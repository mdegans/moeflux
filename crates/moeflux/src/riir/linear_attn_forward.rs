//! End-to-end linear-attention layer forward — Phase 4c.
//!
//! Composes:
//!
//! 1. **Pre-attn input norm** (CPU `rms_norm`)
//! 2. **4 batched projection matvecs** (qkv → 12288, z → 8192, beta →
//!    64, alpha → 64) — `dequant_matvec_4bit_v3` against `wf_buf` at
//!    offsets
//! 3. **5 linear-attn fused kernels**: conv1d_step → rms_norm_qk →
//!    compute_decay_beta → gated_delta_net_step → gated_rms_norm
//!    (output staged into `buffers.batch_out[6]`)
//! 4. Hand-off to [`post_attention_tail`] which runs:
//!    - **o_proj** (matvec from `linear_total_value` → HIDDEN_DIM)
//!    - **Residual add + post-attn RMSNorm**
//!    - **MoE router** (gate matvec → CPU softmax/topK/normalize)
//!      + shared-expert gate score matvec
//!    - **Shared expert FFN** (gate / up / SwiGLU / down)
//!    - **K-expert MoE dispatch + combine** via the 9b path
//!      (`gpu_batched_experts_forward`)
//!
//! Output: post-combine HIDDEN_DIM hidden state in `buffers.input`.
//!
//! Mirrors `fused_layer_forward`'s `!is_full` GPU production path
//! (infer.m:4253–~5085), minus the `prev_gpu_combined` fast path and
//! the deferred-experts state machine — for the dump-hook diff this
//! runs synchronously: each command buffer is committed and waited
//! before the next.
//!
//! ## Why one big function
//!
//! Per the plan: the compose is a straight sequence of encode-and-go
//! steps. Splitting it across helpers obscures the data flow without
//! reducing complexity. The function is long but linear; comment
//! markers (`// ── step N: …`) make it scannable.

use metal::{
    Buffer, CommandBufferRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize, NSUInteger,
};

use super::deferred::{
    gpu_batched_experts_begin, gpu_batched_experts_begin_pre_staged,
    DeferredError,
};
use super::expert_forward::{ChainToNormed, MoeBuffers};
use super::expert_io::ExpertFiles;
use super::gpu_attn::{
    encode_attn_scores_batched_into, encode_attn_softmax_batched_into,
    encode_attn_values_batched_into, encode_sigmoid_gate_into,
    GpuAttnPipelines,
};
use super::gpu_linear_attn::{
    encode_compute_decay_beta, encode_conv1d_step, encode_delta_net_step,
    encode_gated_rms_norm, encode_rms_norm_qk, LinearAttnPipelines,
};
use super::gpu_matvec::{encode_matvec, MatvecPipelines, MatvecSpec};
use super::gpu_norm::{encode_rms_norm_bf16_into, RmsNormBf16Pipelines};
use super::layer_weight_cache::LayerWeightCache;
use super::metal::{MetalBackend, MetalError};
use super::moe_router::moe_router_cpu;
use super::mtl_weight_buf::MtlWeightBuf;
use super::state::LinearAttnState;
use super::variants::{Variant, RMS_NORM_EPS, VARIANT};
use super::weight_file::WeightFile;

/// Errors that can surface during a layer forward (linear or full
/// attention). 4d renamed from `LinearAttnForwardError` once
/// [`post_attention_tail`] became shared between the two paths.
#[derive(Debug, thiserror::Error)]
pub enum LayerForwardError {
    #[error("missing tensor for layer {layer}: {tensor}")]
    MissingTensor {
        layer: usize,
        tensor: &'static str,
    },
    #[error("hidden_in must be HIDDEN_DIM={expected} floats, got {actual}")]
    BadHiddenLen { expected: usize, actual: usize },
    #[error("Metal: {0}")]
    Metal(#[from] MetalError),
    #[error("MoE router: {0}")]
    Router(#[from] super::moe_router::MoeRouterError),
    #[error("expert FFN: {0}")]
    Expert(#[from] super::expert_forward::ExpertForwardError),
    #[error("expert I/O: {0}")]
    ExpertIo(#[from] super::expert_io::ExpertIoError),
    #[error("RoPE: {0}")]
    Rope(#[from] super::rope::RopeError),
    #[error("SDPA: {0}")]
    Sdpa(#[from] super::sdpa::SdpaError),
    #[error("RMSNorm: {0}")]
    RmsNorm(#[from] super::rms_norm::RmsNormError),
    #[error("deferred experts: {0}")]
    Deferred(#[from] DeferredError),
}

/// Backwards-compat alias. `LinearAttnForwardError` was the original
/// name in 4c; 4d generalised it.
pub type LinearAttnForwardError = LayerForwardError;

/// Persistent GPU scratch + recurrence-state buffers needed by the
/// per-layer forward (linear and full attention). Allocated once per
/// [`crate::riir::RsCtx`].
///
/// Renamed from `LinearAttnBuffers` in 4d. Hosts buffers shared across
/// the two attention paths (input/normed/residual/h_mid/output, the
/// 7-slot batch_out, MoE shared FFN scratch) plus path-specific
/// recurrence/cache:
///
/// - Linear-attn: `conv_state`, `delta_state`, `conv_output`,
///   `delta_g_decay`, `delta_beta`, `delta_output`.
/// - Full-attn: `q_proj_out`, `k_out`, `v_out` (the 3 projection
///   outputs read back to host for CPU per-head norm + RoPE + KV
///   append + SDPA).
pub struct LayerForwardBuffers {
    pub input: Buffer,
    pub normed: Buffer,
    pub residual: Buffer,
    pub h_mid: Buffer,
    pub output: Buffer,
    /// 7 batch-output slots. Sized per slot:
    /// - [0]: LINEAR_CONV_DIM (qkv) — only used by linear-attn
    /// - [1]: LINEAR_TOTAL_VALUE (z) — only used by linear-attn
    /// - [2]: LINEAR_NUM_V_HEADS (beta) — only used by linear-attn
    /// - [3]: LINEAR_NUM_V_HEADS (alpha) — only used by linear-attn
    /// - [4]: NUM_EXPERTS (router gate logits) — both paths
    /// - [5]: 1 (shared-expert gate scalar) — both paths
    /// - [6]: max(LINEAR_TOTAL_VALUE, NUM_ATTN_HEADS * HEAD_DIM) —
    ///   o_proj input staging slot for both paths. On qwen3_5_moe
    ///   variants these two values match exactly (linear: 32*128 =
    ///   4096 for A3B; full: 16*256 = 4096), so the slot is reused.
    pub batch_out: [Buffer; 7],
    /// Per-linear-layer recurrence state.
    pub conv_state: Vec<Buffer>,
    pub delta_state: Vec<Buffer>,
    /// Scratch for one layer's linear-attn pipeline (reused across
    /// layers).
    pub conv_output: Buffer,
    pub delta_g_decay: Buffer,
    pub delta_beta: Buffer,
    pub delta_output: Buffer,
    pub sum_sq: Buffer,
    /// Shared-expert intermediate (SHARED_INTERMEDIATE floats).
    pub shared_gate_out: Buffer,
    pub shared_up_out: Buffer,
    pub shared_act: Buffer,
    pub shared_out: Buffer,
    /// Full-attn projection outputs. `q_proj_out` carries the raw
    /// per-head `(q, gate)` interleave (`num_attn_heads * head_dim *
    /// 2` floats); `k_out` / `v_out` carry the `kv_dim` raw outputs
    /// before per-head norm + RoPE + KV append.
    pub q_proj_out: Buffer,
    pub k_out: Buffer,
    pub v_out: Buffer,

    /// Slice 5d-7b — GPU full-attention buffers.
    ///
    /// Per-full-attn-layer KV mirrors (host KV stays canonical for
    /// `state_save`; these get one-way-synced on append + state_load):
    /// `gpu_kv_k[fa_idx]` / `gpu_kv_v[fa_idx]` are `GPU_KV_SEQ * kv_dim`
    /// floats each. `fa_idx` = `full_attn_layer_idx_for(layer_idx)`.
    /// Mirrors C `g_metal->buf_kv_k[NUM_FULL_ATTN_LAYERS]` allocation
    /// at `infer.m:1255..1260`.
    pub gpu_kv_k: Vec<Buffer>,
    pub gpu_kv_v: Vec<Buffer>,
    /// Shared scratch for the GPU SDPA fast path. Reused across layers
    /// because SDPA is layer-sequential per token (matches C). Sizes:
    /// - `gpu_attn_q` / `gpu_attn_out` / `gpu_attn_gate`:
    ///   `num_attn_heads * head_dim` floats each
    /// - `gpu_attn_scores`: `num_attn_heads * GPU_KV_SEQ` floats
    pub gpu_attn_q: Buffer,
    pub gpu_attn_scores: Buffer,
    pub gpu_attn_out: Buffer,
    pub gpu_attn_gate: Buffer,
}

/// Backwards-compat alias for the original 4c name.
pub type LinearAttnBuffers = LayerForwardBuffers;

impl LayerForwardBuffers {
    pub fn new(device: &Device) -> Self {
        let v = VARIANT;
        // Allocate + zero. `device.new_buffer` doesn't guarantee
        // zeroed storage; the recurrence buffers (`conv_state`,
        // `delta_state`) read from this region on first call so
        // garbage initial values would diverge from the C path's
        // metal_setup (which calloc-zeros the same buffers).
        let f32_buf = |n: usize| {
            let b = device.new_buffer(
                (n * std::mem::size_of::<f32>()) as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            );
            // SAFETY: shared storage; no GPU work in flight on a
            // freshly allocated buffer.
            unsafe {
                std::ptr::write_bytes(
                    b.contents() as *mut u8,
                    0,
                    n * std::mem::size_of::<f32>(),
                );
            }
            b
        };
        let q_dim_full = v.num_attn_heads * v.head_dim;
        let q_proj_dim_full = q_dim_full * 2;
        let kv_dim_full = v.num_kv_heads * v.head_dim;
        // batch_out[6] is the o_proj input. Both paths land their
        // attention output here. Linear: linear_total_value; full:
        // q_dim_full. Pick max so either fits.
        let oproj_in_max = v.linear_total_value().max(q_dim_full);
        let batch_sizes = [
            v.linear_conv_dim(),
            v.linear_total_value(),
            v.linear_num_v_heads,
            v.linear_num_v_heads,
            v.num_experts,
            1,
            oproj_in_max,
        ];
        let batch_out: [Buffer; 7] =
            std::array::from_fn(|i| f32_buf(batch_sizes[i]));

        let num_linear = v.num_layers - num_full_attn_layers(&v);
        let conv_state = (0..num_linear)
            .map(|_| {
                f32_buf((Variant::CONV_KERNEL_SIZE - 1) * v.linear_conv_dim())
            })
            .collect();
        let delta_state = (0..num_linear)
            .map(|_| {
                f32_buf(
                    v.linear_num_v_heads
                        * Variant::LINEAR_VALUE_DIM
                        * Variant::LINEAR_KEY_DIM,
                )
            })
            .collect();

        let num_full_attn = num_full_attn_layers(&v);
        let gpu_kv_floats =
            super::variants::GPU_KV_SEQ * kv_dim_full;
        let gpu_kv_k =
            (0..num_full_attn).map(|_| f32_buf(gpu_kv_floats)).collect();
        let gpu_kv_v =
            (0..num_full_attn).map(|_| f32_buf(gpu_kv_floats)).collect();

        Self {
            input: f32_buf(v.hidden_dim),
            normed: f32_buf(v.hidden_dim),
            residual: f32_buf(v.hidden_dim),
            h_mid: f32_buf(v.hidden_dim),
            output: f32_buf(v.hidden_dim),
            batch_out,
            conv_state,
            delta_state,
            conv_output: f32_buf(v.linear_conv_dim()),
            delta_g_decay: f32_buf(v.linear_num_v_heads),
            delta_beta: f32_buf(v.linear_num_v_heads),
            delta_output: f32_buf(v.linear_total_value()),
            sum_sq: f32_buf(1),
            shared_gate_out: f32_buf(v.shared_intermediate),
            shared_up_out: f32_buf(v.shared_intermediate),
            shared_act: f32_buf(v.shared_intermediate),
            shared_out: f32_buf(v.hidden_dim),
            q_proj_out: f32_buf(q_proj_dim_full),
            k_out: f32_buf(kv_dim_full),
            v_out: f32_buf(kv_dim_full),
            gpu_kv_k,
            gpu_kv_v,
            gpu_attn_q: f32_buf(q_dim_full),
            gpu_attn_scores: f32_buf(
                v.num_attn_heads * super::variants::GPU_KV_SEQ,
            ),
            gpu_attn_out: f32_buf(q_dim_full),
            gpu_attn_gate: f32_buf(q_dim_full),
        }
    }

    /// Reset every per-layer state buffer to zero. Called from
    /// `RsCtx::memory_clear` to clear the recurrence in addition to
    /// the host-side state vector (which today the GPU doesn't read,
    /// so this is the source of truth on the GPU side).
    pub fn reset_recurrence(&mut self) {
        for b in &self.conv_state {
            zero_f32_buffer(b);
        }
        for b in &self.delta_state {
            zero_f32_buffer(b);
        }
    }

    /// Slice 5d-7b — zero the GPU full-attn KV mirrors. Called from
    /// `RsCtx::memory_clear` alongside `reset_recurrence`. The host
    /// KV cache is cleared via `clear_all(layer_states)`; this is the
    /// matching reset on the GPU side. Mirrors the C path's reset of
    /// `buf_kv_k` / `buf_kv_v` at `mf_memory_clear`.
    pub fn reset_gpu_attn_kv_mirrors(&mut self) {
        for b in &self.gpu_kv_k {
            zero_f32_buffer(b);
        }
        for b in &self.gpu_kv_v {
            zero_f32_buffer(b);
        }
    }
}

/// Zero every byte of a shared-storage Metal buffer. Used by
/// `memory_clear` to reset GPU-resident state (linear-attn
/// recurrence, full-attn KV mirrors).
///
/// # Safety
///
/// `memory_clear` is the only caller and must run after all
/// in-flight dispatches have completed. The deferred-ring drain at
/// the top of `memory_clear` enforces this; no other path reaches
/// this function.
fn zero_f32_buffer(b: &Buffer) {
    let bytes = b.length() as usize;
    // SAFETY: see fn docs.
    unsafe {
        std::ptr::write_bytes(b.contents() as *mut u8, 0, bytes);
    }
}

/// `linear_layer_idx = layer_idx - (layer_idx + 1) / FULL_ATTN_INTERVAL`.
/// Returns `None` if `layer_idx` is a full-attention layer. The
/// modulo arithmetic for the linear index is qwen3_5_moe-specific
/// (full-attn layers are evenly spaced); the kind dispatch goes
/// through [`Variant::layer_kind`] so a future variant can plug in a
/// different layer-kind sequence without touching this helper's
/// callers.
pub fn linear_layer_idx_for(layer_idx: usize) -> Option<usize> {
    use super::variants::LayerKind;
    if VARIANT.layer_kind(layer_idx) == LayerKind::FullAttn {
        None
    } else {
        Some(layer_idx - (layer_idx + 1) / VARIANT.full_attn_interval)
    }
}

/// `fa_idx = (layer_idx + 1) / FULL_ATTN_INTERVAL - 1`. Returns
/// `None` if `layer_idx` is a linear-attn layer. Mirrors C
/// `(layer_idx + 1) / FULL_ATTN_INTERVAL - 1` at `infer.m:5092`.
pub fn full_attn_layer_idx_for(layer_idx: usize) -> Option<usize> {
    use super::variants::LayerKind;
    if VARIANT.layer_kind(layer_idx) == LayerKind::FullAttn {
        Some((layer_idx + 1) / VARIANT.full_attn_interval - 1)
    } else {
        None
    }
}

pub(super) fn num_full_attn_layers(v: &Variant) -> usize {
    v.num_layers / v.full_attn_interval
}

/// Per-tensor bit-width lookup for the matvec dispatcher. Defaults to
/// 4-bit for tensors not in the manifest; the dispatcher's max(_, 4)
/// floor guards against misreads.
pub(super) fn bits_of(wf: &WeightFile, name: &str) -> u32 {
    wf.tensor_info(name)
        .map(|i| i.bits as u32)
        .unwrap_or(4)
        .max(4)
}

/// Adapter naming the o_proj weights + input shape to use for one
/// call into [`post_attention_tail`]. Linear-attn fills with
/// `linear_o_proj_*`; full-attn fills with `full_o_proj_*`.
pub(super) struct OProj {
    pub w_off: u64,
    pub s_off: u64,
    pub b_off: u64,
    pub bits: u32,
    /// Number of input floats the matvec reads from
    /// `buffers.batch_out[6]` (CPU SDPA path) or
    /// `buffers.gpu_attn_out` (GPU SDPA path). Linear:
    /// `linear_total_value`. Full: `num_attn_heads * head_dim`.
    pub in_dim: u32,
}

/// Slice 5d-7b — args for the GPU SDPA fast path encoded at the top
/// of CMD2 inside [`post_attention_tail`]. Carries the per-call
/// inputs not derivable from `VARIANT`: which full-attn KV mirror
/// slot to use, and the current KV length. When `Some`, the tail
/// encodes the 4 attn kernels (`attn_scores_batched` →
/// `attn_softmax_batched` → `attn_values_batched` → `sigmoid_gate`)
/// into the same cmdbuf as `o_proj`, residual_add, and post-attn
/// rms_norm — no extra commit-wait. Q + q_gate are pre-staged into
/// `buffers.gpu_attn_q` / `buffers.gpu_attn_gate` by the caller; K/V
/// mirrors are pre-populated by the per-token KV-append memcpy.
///
/// When `None`, the tail follows the existing CPU-attn path: o_proj
/// reads from `buffers.batch_out[6]` (caller-staged via
/// `sdpa_cpu` + memcpy).
pub(super) struct GpuAttnEncodeArgs {
    /// Index into `LayerForwardBuffers::gpu_kv_k` / `gpu_kv_v`. From
    /// [`full_attn_layer_idx_for`].
    pub fa_idx: usize,
    /// `kv_state.len` after this token's KV append — the number of
    /// positions the kernels read from the mirror.
    pub kv_len: u32,
}

/// Run one linear-attention layer's forward pass on the GPU.
///
/// Pre: `buffers.input` holds the input hidden state (HIDDEN_DIM
/// floats). The targeted layer's `conv_state` / `delta_state` are
/// mutated in place. `state` is the host-side mirror used for
/// `memory_*` ops; for 4c we keep it in lockstep with the GPU
/// buffers via reset only — partial truncation will resync GPU
/// buffers via `reset_recurrence` (a faithful port of the lossy
/// semantic).
///
/// Post: `*deferred` holds an in-flight K-expert dispatch (committed
/// without `wait`). The caller is responsible for draining it via
/// `complete_deferred_experts_into` (writes the post-combine hidden
/// into a host slice or the next layer's `buffers.input`) or
/// `discard_deferred_experts_in` (drop without readback). `buffers.
/// input` is **not** the output target after slice 4f-3.
#[allow(clippy::too_many_arguments)]
pub fn linear_attn_layer_forward(
    metal: &mut MetalBackend,
    wf: &WeightFile,
    wf_buf: &MtlWeightBuf,
    layer_cache: &LayerWeightCache,
    buffers: &mut LayerForwardBuffers,
    moe: &mut MoeBuffers,
    deferred: &mut super::DeferredRing,
    layer_idx: usize,
    k_active: usize,
    expert_files: &ExpertFiles,
    pool: &rayon::ThreadPool,
    prefetch: &mut super::PrefetchState,
    // Slice 5d-9: which `data_prefetch` set this layer reads from
    // (parity ping-pong: `layer_idx % 2`). Plumbed through to
    // `post_attention_tail`'s K-expert dispatch.
    prefetch_set: usize,
    _layer_state: &mut LinearAttnState,
    gpu_combine: bool,
    // Slice 5d-8: previous layer's K-expert dispatch chained the
    // post-combine rms_norm into its cmdbuf, so `buffers.normed` is
    // already populated when we land here. Skip CMD1's input-norm
    // prelude in that case.
    prev_layer_chained: bool,
    // Slice 5d-8: when `Some`, this layer's K-expert cmdbuf appends
    // rms_norm_sum_sq + rms_norm_apply_bf16 against the next layer's
    // input_layernorm.weight at this offset (in `wf_buf`). Output lands
    // in `buffers.normed`, ready for the next layer's CMD1. `None`
    // disables the chain — used for the last layer and the dump hook.
    chain_next_norm_off: Option<u64>,
) -> Result<(), LayerForwardError> {
    let v = VARIANT;
    let linear_layer_idx = linear_layer_idx_for(layer_idx).ok_or(
        LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "linear_layer_idx (called on full-attn layer)",
        },
    )?;

    // Per-tensor bit width lookup. 4-bit is the default; A3B uses
    // 8-bit for `mlp.gate.weight` and `mlp.shared_expert_gate.weight`.
    let qkv_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight"),
    );
    let z_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.linear_attn.in_proj_z.weight"),
    );
    let alpha_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.linear_attn.in_proj_a.weight"),
    );
    let beta_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.linear_attn.in_proj_b.weight"),
    );
    let o_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.linear_attn.out_proj.weight"),
    );

    // Pull the linear-attn-specific offsets out of the tagged-enum
    // cache. Returning early with `MissingTensor` here also guards
    // against accidentally calling this function on a full-attn layer
    // (the dispatcher in `layer_forward_dump` already filters; this
    // is defense in depth and matches the symmetric guard in
    // `full_attn_layer_forward`).
    let attn = layer_cache.attn.linear().ok_or(
        LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "linear_attn weights (called on full-attn layer)",
        },
    )?;
    let qkv_w = attn.qkv_w;
    let qkv_s = attn.qkv_s;
    let qkv_b = attn.qkv_b;
    let z_w = attn.z_w;
    let z_s = attn.z_s;
    let z_b = attn.z_b;
    let beta_w = attn.beta_w;
    let beta_s = attn.beta_s;
    let beta_b = attn.beta_b;
    let alpha_w = attn.alpha_w;
    let alpha_s = attn.alpha_s;
    let alpha_b = attn.alpha_b;
    let conv1d_w = attn.conv1d_w;
    let a_log = attn.a_log;
    let dt_bias = attn.dt_bias;
    let gnorm_w = attn.gated_norm_w;
    let o_w = attn.o_proj_w;
    let o_s = attn.o_proj_s;
    let o_b = attn.o_proj_b;

    // Pre-fetch every pipeline.
    let lp = LinearAttnPipelines::fetch(metal)?;
    let mv = MatvecPipelines::fetch(metal)?;
    let rms_pipes = RmsNormBf16Pipelines::fetch(metal)?;

    // ── CMD1: input rms_norm + projections + linear-attn pipeline ─
    //
    // Slice 5d-2: input rms_norm runs on the GPU as the prelude to
    // CMD1 (was CPU + 4 host↔GPU memcopies before). C path runs CPU
    // rms_norm in the slow branch (`infer.m:4492`) but the fast branch
    // chains a GPU rms_norm at the tail of the previous CMD3
    // (`infer.m:5712..5744`); functionally we get the latter shape by
    // running the same kernel pair as the first dispatches in CMD1.
    // Slice 9e established that the `rms_norm_sum_sq` /
    // `rms_norm_apply_bf16` pair is bit-exact per-PSO; agreement
    // against the C fast path is bit-exact, agreement against the C
    // slow path drifts by a few ULPs per layer (well within the
    // existing diff floor `cosine ≥ 0.9999`).
    //
    // `buffers.input` is the residual source consumed by
    // `post_attention_tail`'s `encode_residual_add` later this layer;
    // nothing in this layer's forward writes to it (the next layer's
    // top-of-forward `complete_deferred_experts_into` is the next
    // mutation), so the dual-use is safe.
    {
        let cmdbuf = metal.queue().new_command_buffer();

        // Slice 5d-8: skip the input-norm prelude when the previous
        // layer chained — `buffers.normed` is already populated by the
        // appended rms_norm at the tail of the previous K-expert cmdbuf
        // (drained at the top of `step_internal`'s layer iteration).
        if !prev_layer_chained {
            encode_rms_norm_bf16_into(
                cmdbuf,
                &rms_pipes,
                &buffers.input,
                wf_buf.buffer(),
                layer_cache.input_layernorm_w,
                &buffers.sum_sq,
                &buffers.normed,
                v.hidden_dim as u32,
                RMS_NORM_EPS,
            );
        }

        // 4 batched projections from buffers.normed:
        let specs = [
            MatvecSpec {
                w_off: qkv_w,
                s_off: qkv_s,
                b_off: qkv_b,
                input: &buffers.normed,
                output: &buffers.batch_out[0],
                out_dim: v.linear_conv_dim() as u32,
                in_dim: v.hidden_dim as u32,
                bits: qkv_bits,
            },
            MatvecSpec {
                w_off: z_w,
                s_off: z_s,
                b_off: z_b,
                input: &buffers.normed,
                output: &buffers.batch_out[1],
                out_dim: v.linear_total_value() as u32,
                in_dim: v.hidden_dim as u32,
                bits: z_bits,
            },
            MatvecSpec {
                w_off: beta_w,
                s_off: beta_s,
                b_off: beta_b,
                input: &buffers.normed,
                output: &buffers.batch_out[2],
                out_dim: v.linear_num_v_heads as u32,
                in_dim: v.hidden_dim as u32,
                bits: beta_bits,
            },
            MatvecSpec {
                w_off: alpha_w,
                s_off: alpha_s,
                b_off: alpha_b,
                input: &buffers.normed,
                output: &buffers.batch_out[3],
                out_dim: v.linear_num_v_heads as u32,
                in_dim: v.hidden_dim as u32,
                bits: alpha_bits,
            },
        ];
        for s in &specs {
            encode_matvec(cmdbuf, &mv, wf_buf, s);
        }

        encode_conv1d_step(
            cmdbuf,
            &lp.conv1d_step,
            &buffers.conv_state[linear_layer_idx],
            &buffers.batch_out[0],
            wf_buf.buffer(),
            conv1d_w,
            &buffers.conv_output,
            v.linear_conv_dim() as u32,
        );

        encode_rms_norm_qk(
            cmdbuf,
            &lp.rms_norm_qk,
            &buffers.conv_output,
            v.linear_num_k_heads as u32,
            Variant::LINEAR_KEY_DIM as u32,
        );

        encode_compute_decay_beta(
            cmdbuf,
            &lp.compute_decay_beta,
            &buffers.batch_out[3], // alpha
            &buffers.batch_out[2], // beta
            wf_buf.buffer(),
            a_log,
            dt_bias,
            &buffers.delta_g_decay,
            &buffers.delta_beta,
            v.linear_num_v_heads as u32,
        );

        let k_heads_per_v =
            (v.linear_num_v_heads / v.linear_num_k_heads) as u32;
        encode_delta_net_step(
            cmdbuf,
            &lp.delta_net_step,
            &buffers.delta_state[linear_layer_idx],
            &buffers.conv_output,
            &buffers.delta_g_decay,
            &buffers.delta_beta,
            &buffers.delta_output,
            v.linear_num_v_heads as u32,
            Variant::LINEAR_VALUE_DIM as u32,
            k_heads_per_v,
        );

        encode_gated_rms_norm(
            cmdbuf,
            &lp.gated_rms_norm,
            &buffers.delta_output,
            &buffers.batch_out[1], // z
            wf_buf.buffer(),
            gnorm_w,
            &buffers.batch_out[6],
            v.linear_num_v_heads as u32,
            Variant::LINEAR_VALUE_DIM as u32,
        );

        cmdbuf.commit();
        cmdbuf.wait_until_completed();
    }

    // ── Hand off to the shared post-attention tail ───────────────
    // `batch_out[6]` already holds the `gated_rms_norm` output —
    // exactly the o_proj input the tail consumes. The tail leaves an
    // in-flight K-expert dispatch in `*deferred`. Linear-attn never
    // takes the GPU SDPA fast path (it has no attention-kernel
    // pipeline), so `gpu_attn_args = None`.
    post_attention_tail(
        metal,
        wf,
        wf_buf,
        layer_cache,
        buffers,
        moe,
        deferred,
        layer_idx,
        k_active,
        expert_files,
        pool,
        prefetch,
        prefetch_set,
        OProj {
            w_off: o_w,
            s_off: o_s,
            b_off: o_b,
            bits: o_bits,
            in_dim: v.linear_total_value() as u32,
        },
        gpu_combine,
        /* gpu_attn_args = */ None,
        chain_next_norm_off,
    )
}

/// Shared post-attention tail used by both linear- and full-attention
/// layer forwards. Reads the attention output from
/// `buffers.batch_out[6]` (caller-staged) and runs the rest of
/// `fused_layer_forward`:
///
/// 1. CMD2: o_proj matvec → residual_add → post-attn rms_norm.
/// 2. CMD3a: gate logits + shared-expert gate scalar + shared FFN
///    `gate_proj` + `up_proj`.
/// 3. CPU swiglu of shared_gate × shared_up → `shared_act`.
/// 4. CMD3a-b: shared `down_proj` matvec.
/// 5. CPU MoE router on the gate logits.
/// 6. Load K expert blobs from disk via [`ExpertFiles::read_expert`].
/// 7. CMD3b: K-expert FFN + combine — submits the dispatch via
///    [`gpu_batched_experts_begin`] without waiting; ownership of the
///    in-flight cmdbuf transfers to `*deferred`.
///
/// On return, `*deferred` holds the in-flight K-expert dispatch. The
/// caller is responsible for either `complete_deferred_experts_into`
/// (drain into a host slice / next layer's `buffers.input`) or
/// `discard_deferred_experts_in` (drop without readback) before the
/// next `post_attention_tail` call. Slice 4f-3 wired this rewire;
/// before, the synchronous `gpu_batched_experts_forward` ran inline
/// and the result was written into `buffers.input` here. The async
/// path matches the C-side `g_deferred` lifecycle and unlocks the
/// fast/slow split landing in 4f-perf.
#[allow(clippy::too_many_arguments)]
pub(super) fn post_attention_tail(
    metal: &mut MetalBackend,
    wf: &WeightFile,
    wf_buf: &MtlWeightBuf,
    layer_cache: &LayerWeightCache,
    buffers: &mut LayerForwardBuffers,
    moe: &mut MoeBuffers,
    deferred: &mut super::DeferredRing,
    layer_idx: usize,
    k_active: usize,
    expert_files: &ExpertFiles,
    pool: &rayon::ThreadPool,
    prefetch: &mut super::PrefetchState,
    // Slice 5d-9: which `data_prefetch` set this layer reads from. The
    // caller assigns set `layer_idx % 2`; the prefetch state machine
    // wrote that same set at the top of this layer's iteration.
    prefetch_set: usize,
    o_proj: OProj,
    gpu_combine: bool,
    gpu_attn_args: Option<GpuAttnEncodeArgs>,
    // Slice 5d-8: when `Some` AND `gpu_combine` is true, the K-expert
    // cmdbuf appends rms_norm_sum_sq + rms_norm_apply_bf16 against the
    // next layer's input_layernorm.weight at this offset (in `wf_buf`).
    // Output lands in `buffers.normed`, ready for the next layer's
    // CMD1. `None` (or CPU-combine) disables the chain.
    chain_next_norm_off: Option<u64>,
) -> Result<(), LayerForwardError> {
    let v = VARIANT;

    // Per-tensor bit widths for the MoE-side matvecs.
    let gate_bits =
        bits_of(wf, &format!("model.layers.{layer_idx}.mlp.gate.weight"));
    let seg_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert_gate.weight"
        ),
    );
    let s_gate_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight"
        ),
    );
    let s_up_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight"
        ),
    );
    let s_down_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight"
        ),
    );

    let post_attn_norm_w = layer_cache.post_attention_layernorm_w;
    let gate_w = layer_cache.gate.w;
    let gate_s = layer_cache.gate.s;
    let gate_b = layer_cache.gate.b;
    let shared_up_w = layer_cache.shared.up_w;
    let shared_up_s = layer_cache.shared.up_s;
    let shared_up_b = layer_cache.shared.up_b;
    let shared_gate_w = layer_cache.shared.gate_w;
    let shared_gate_s = layer_cache.shared.gate_s;
    let shared_gate_b = layer_cache.shared.gate_b;
    let shared_down_w = layer_cache.shared.down_w;
    let shared_down_s = layer_cache.shared.down_s;
    let shared_down_b = layer_cache.shared.down_b;
    let seg_w = layer_cache.shared.seg_w;
    let seg_s = layer_cache.shared.seg_s;
    let seg_b = layer_cache.shared.seg_b;

    let mv = MatvecPipelines::fetch(metal)?;
    let sum_sq = metal.pipeline("rms_norm_sum_sq")?.clone();
    let apply = metal.pipeline("rms_norm_apply_bf16")?.clone();
    let resid_add = metal.pipeline("residual_add")?.clone();
    let swiglu = metal.pipeline("swiglu_fused")?.clone();
    // Slice 5d-7b: pre-fetch attn pipelines only when the GPU SDPA
    // fast path is active. Keeps the CPU-SDPA / linear-attn paths
    // free of unrelated pipeline lookups.
    let attn_pipes = if gpu_attn_args.is_some() {
        Some(GpuAttnPipelines::fetch(metal)?)
    } else {
        None
    };

    // ── CMD2+3: post-attn + shared FFN + gate logits, single cmdbuf ─
    //
    // Slice 5d-3: collapses the previous CMD2 / CMD3a / CMD3a-b
    // commit+wait sequence into a single command buffer. The C path
    // also fuses post-attn + routing + shared-FFN gate/up into ONE
    // cmdbuf (`infer.m:5088..5258`, the `cmd_fused` block); we now
    // additionally fold the shared-FFN swiglu (was CPU) and the
    // shared_down matvec into the same buffer, eliminating the
    // CPU-side swiglu loop and the separate `cmd_dn` shape.
    //
    // GPU swiglu replaces the CPU SiLU loop the C path uses for the
    // shared FFN (`infer.m:2977 cpu_swiglu`). Per slice 9a's finding,
    // `swiglu_fused` is bit-exact per-PSO; the drift against C's
    // CPU swiglu here is small libm-precision territory and remains
    // within the diff oracle's `cosine ≥ 0.9999` floor. The K-expert
    // FFN already used GPU swiglu inside `gpu_batched_experts_*`, so
    // this is the only remaining cpu_swiglu site outside the experts.
    //
    // Encoders within one cmdbuf serialize on Metal, so the data
    // dependencies (o_proj → residual_add → rms_norm → projections →
    // swiglu → shared_down) are honored without per-encoder waits.
    {
        let cmdbuf = metal.queue().new_command_buffer();

        // ── Slice 5d-7b: GPU full-attn fast path (Enc A1..A4) ──────
        //
        // When active, encode the 4 attn kernels at the head of CMD2
        // so SDPA + sigmoid gate piggyback on the same commit-wait as
        // o_proj + residual + post-attn rms_norm. Mirrors the C path's
        // `gpu_attn_fuse` block at `infer.m:5091..5163`. Q + q_gate
        // are pre-staged into `buffers.gpu_attn_q` / `gpu_attn_gate` by
        // the caller; K/V mirrors are pre-populated by the per-token
        // KV-append memcpy.
        if let (Some(args), Some(attn_pipes)) =
            (gpu_attn_args.as_ref(), attn_pipes.as_ref())
        {
            let head_dim = v.head_dim as u32;
            let kv_dim = (v.num_kv_heads * v.head_dim) as u32;
            let num_heads = v.num_attn_heads as u32;
            let heads_per_kv = (v.num_attn_heads / v.num_kv_heads) as u32;
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let seq_stride = super::variants::GPU_KV_SEQ as u32;

            encode_attn_scores_batched_into(
                cmdbuf,
                &attn_pipes.scores,
                &buffers.gpu_attn_q,
                &buffers.gpu_kv_k[args.fa_idx],
                &buffers.gpu_attn_scores,
                num_heads,
                head_dim,
                kv_dim,
                args.kv_len,
                seq_stride,
                heads_per_kv,
                scale,
            );
            encode_attn_softmax_batched_into(
                cmdbuf,
                &attn_pipes.softmax,
                &buffers.gpu_attn_scores,
                num_heads,
                args.kv_len,
                seq_stride,
            );
            encode_attn_values_batched_into(
                cmdbuf,
                &attn_pipes.values,
                &buffers.gpu_attn_scores,
                &buffers.gpu_kv_v[args.fa_idx],
                &buffers.gpu_attn_out,
                num_heads,
                head_dim,
                kv_dim,
                args.kv_len,
                seq_stride,
                heads_per_kv,
            );
            encode_sigmoid_gate_into(
                cmdbuf,
                &attn_pipes.gate,
                &buffers.gpu_attn_out,
                &buffers.gpu_attn_gate,
                num_heads * head_dim,
            );
        }

        // o_proj + residual_add + post-attn rms_norm (was CMD2).
        // GPU SDPA path: read from `gpu_attn_out` (zero-host-stage).
        // CPU SDPA / linear-attn paths: read from `batch_out[6]`.
        let oproj_input = if gpu_attn_args.is_some() {
            &buffers.gpu_attn_out
        } else {
            &buffers.batch_out[6]
        };
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: o_proj.w_off,
                s_off: o_proj.s_off,
                b_off: o_proj.b_off,
                input: oproj_input,
                output: &buffers.output,
                out_dim: v.hidden_dim as u32,
                in_dim: o_proj.in_dim,
                bits: o_proj.bits,
            },
        );
        encode_residual_add(
            cmdbuf,
            &resid_add,
            &buffers.output,
            &buffers.input, // residual source — see slice 5d-2 note
            &buffers.h_mid,
            v.hidden_dim as u32,
        );
        encode_rms_norm_pair(
            cmdbuf,
            &sum_sq,
            &apply,
            &buffers.h_mid,
            wf_buf.buffer(),
            post_attn_norm_w,
            &buffers.normed,
            &buffers.sum_sq,
            v.hidden_dim as u32,
        );

        // gate logits + shared-expert gate scalar + shared FFN
        // gate/up matvecs (was CMD3a).
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: gate_w,
                s_off: gate_s,
                b_off: gate_b,
                input: &buffers.normed,
                output: &buffers.batch_out[4],
                out_dim: v.num_experts as u32,
                in_dim: v.hidden_dim as u32,
                bits: gate_bits,
            },
        );
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: seg_w,
                s_off: seg_s,
                b_off: seg_b,
                input: &buffers.normed,
                output: &buffers.batch_out[5],
                out_dim: 1,
                in_dim: v.hidden_dim as u32,
                bits: seg_bits,
            },
        );
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: shared_gate_w,
                s_off: shared_gate_s,
                b_off: shared_gate_b,
                input: &buffers.normed,
                output: &buffers.shared_gate_out,
                out_dim: v.shared_intermediate as u32,
                in_dim: v.hidden_dim as u32,
                bits: s_gate_bits,
            },
        );
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: shared_up_w,
                s_off: shared_up_s,
                b_off: shared_up_b,
                input: &buffers.normed,
                output: &buffers.shared_up_out,
                out_dim: v.shared_intermediate as u32,
                in_dim: v.hidden_dim as u32,
                bits: s_up_bits,
            },
        );

        // GPU swiglu — was the CPU loop between CMD3a and CMD3a-b.
        encode_swiglu_buf(
            cmdbuf,
            &swiglu,
            &buffers.shared_gate_out,
            &buffers.shared_up_out,
            &buffers.shared_act,
            v.shared_intermediate as u32,
        );

        // shared_down matvec (was CMD3a-b).
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: shared_down_w,
                s_off: shared_down_s,
                b_off: shared_down_b,
                input: &buffers.shared_act,
                output: &buffers.shared_out,
                out_dim: v.hidden_dim as u32,
                in_dim: v.shared_intermediate as u32,
                bits: s_down_bits,
            },
        );

        cmdbuf.commit();
        cmdbuf.wait_until_completed();
    }

    // ── CPU: MoE router on the gate logits ───────────────────────
    let mut scores =
        read_buffer_to_vec(&buffers.batch_out[4], v.num_experts);
    let mut indices = vec![0i32; k_active];
    let mut weights = vec![0f32; k_active];
    moe_router_cpu(&mut scores, k_active, &mut indices, &mut weights)?;

    // Read shared-gate score scalar (pre-sigmoid).
    let shared_gate_score = {
        let s = read_buffer_to_vec(&buffers.batch_out[5], 1);
        s[0]
    };

    // ── CMD3b: K-expert FFN + combine via slice 9b — async ───────
    //
    // Slice 5d-5 (production fast path, `gpu_combine = true`): pread
    // K expert blobs DIRECTLY into `moe.data[slot]`'s shared-storage
    // pages, then encode the dispatch with GPU buffer refs for the
    // post-attn-norm/residual/shared-out inputs. Saves ~7 MB / layer
    // of host memcpy (the intermediate `expert_data: Vec<u8>` is
    // gone) on top of slice 5d-4's ~2 MB / layer of input
    // round-tripping.
    //
    // The slot-reuse pattern is sound: each layer's K-expert dispatch
    // is waited at the top of the next layer's
    // `complete_deferred_experts_into`, so `moe.data[slot]` is GPU-
    // quiescent by the time this layer preads new bytes into it.
    //
    // CPU-combine fallback path (`gpu_combine = false`) still routes
    // through the host-slice variant — `DeferredMode::Cpu` needs host
    // snapshots of `h_mid` / `shared_out` for the finalize pass.
    let k = k_active;
    if gpu_combine {
        // Slice 5d-6b: speculative-prefetch state machine.
        //
        // 1. Wait for any in-flight prefetch targeting THIS layer.
        // 2. For each slot, decide if the prefetch hit (read from
        //    `data_prefetch[slot]`) or missed (need a sync pread
        //    into `data_synced[slot]`).
        // 3. Parallel sync-pread the missed slots via the io_pool.
        // 4. Encode K-expert dispatch with per-slot SlotSource.
        // 5. Record this layer's actual indices as the prediction
        //    for the next token's same layer.
        // 6. If not the last layer, fire async prefetch for layer
        //    N+1 using the prediction we have for it.
        use rayon::prelude::*;
        use super::prefetch::SlotSource;
        const MAX_K: usize = super::expert_forward::MAX_K;

        // Step 1: drain any in-flight prefetch and check whether it
        // was for this layer.
        let prefetch_status = prefetch.wait_for(layer_idx);

        // Step 2: per-slot hit/miss decision. Set-based, not
        // position-locked — for each actual expert at slot `s`, scan
        // the prefetched indices for a match and record the buffer
        // index where it landed. The K active experts are picked by
        // the router with no canonical ordering, so a position-locked
        // match would fail even when the SET overlaps perfectly with
        // last-token's experts.
        let mut data_set_per_slot: [SlotSource; MAX_K] =
            [SlotSource::Synced; MAX_K];
        let mut hit_count: u64 = 0;
        if let Some(status) = prefetch_status {
            for slot in 0..k {
                let actual = indices[slot];
                for buf_idx in 0..status.k {
                    if status.loaded_indices[buf_idx] == actual {
                        data_set_per_slot[slot] =
                            SlotSource::Prefetched(buf_idx);
                        hit_count += 1;
                        break;
                    }
                }
            }
        }
        prefetch.record_outcome(hit_count, k as u64 - hit_count);

        // Step 3: parallel sync-pread the misses into data_synced.
        let mut dsts = moe.data_synced_slots_mut_array();
        let active = &mut dsts[..k];
        pool.install(|| -> Result<(), super::expert_io::ExpertIoError> {
            active
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(slot, dst)| {
                    if data_set_per_slot[slot] == SlotSource::Synced {
                        let expert_idx = indices[slot] as usize;
                        expert_files
                            .read_expert(layer_idx, expert_idx, *dst)
                    } else {
                        Ok(())
                    }
                })
        })?;

        // Step 5: record actuals (the prediction for the next
        // token's same layer). Done before dispatch so it doesn't
        // depend on the dispatch's success/failure path.
        let mut actuals: [i32; MAX_K] = [0; MAX_K];
        actuals[..k].copy_from_slice(&indices[..k]);
        prefetch.record_actual(layer_idx, actuals);

        // Step 4: encode K-expert dispatch with the per-slot mix.
        // The prefetch for the *next* layer is NOT fired here — it
        // lives in the per-layer loop in `step_internal`, after the
        // drain of THIS layer's deferred dispatch. That ordering is
        // load-bearing: firing the prefetch here would race with
        // the in-flight GPU read of data_prefetch[slot] for the
        // current layer's hits.
        //
        // Slice 5d-8 chain: when `chain_next_norm_off` is `Some`, the
        // K-expert cmdbuf rebinds combine output to `buffers.input` and
        // appends rms_norm_sum_sq + rms_norm_apply_bf16, leaving the
        // next layer's normed input in `buffers.normed`. Allocated only
        // when the chain is active so the borrow doesn't outlive the
        // call.
        let chain_rms_pipes = chain_next_norm_off.map(|_| {
            super::gpu_norm::RmsNormBf16Pipelines {
                sum: sum_sq.clone(),
                apply: apply.clone(),
            }
        });
        let chain = chain_next_norm_off.and_then(|off| {
            chain_rms_pipes.as_ref().map(|pipes| ChainToNormed {
                pipes,
                wf_buf: wf_buf.buffer(),
                next_norm_off: off,
                combine_out: &buffers.input,
                chain_sum_sq: &buffers.sum_sq,
                chain_normed: &buffers.normed,
                eps: RMS_NORM_EPS,
            })
        });
        gpu_batched_experts_begin_pre_staged(
            metal,
            moe,
            deferred,
            k as i32,
            &buffers.normed,     // h_post (post-attn-norm input)
            &buffers.h_mid,      // residual hidden (combine input)
            &buffers.shared_out, // shared FFN output (combine input)
            &weights,
            shared_gate_score,
            layer_idx as i32,
            &data_set_per_slot,
            prefetch_set,
            chain,
        )?;
    } else {
        let expert_size = v.expert_size_4bit();
        let mut expert_data = vec![0u8; k * expert_size];
        for slot in 0..k {
            let expert_idx = indices[slot] as usize;
            let dst = &mut expert_data
                [slot * expert_size..(slot + 1) * expert_size];
            expert_files.read_expert(layer_idx, expert_idx, dst)?;
        }
        let h_mid_host = read_buffer_to_vec(&buffers.h_mid, v.hidden_dim);
        let shared_out_host =
            read_buffer_to_vec(&buffers.shared_out, v.hidden_dim);
        let normed_host = read_buffer_to_vec(&buffers.normed, v.hidden_dim);
        gpu_batched_experts_begin(
            metal,
            moe,
            deferred,
            k as i32,
            &expert_data,
            &normed_host,
            &h_mid_host,
            &shared_out_host,
            &weights,
            shared_gate_score,
            layer_idx as i32,
            /* gpu_combine = */ false,
        )?;
    }

    // No write to `buffers.input` here — the dispatch is in flight.
    // The caller drains it (next layer's top-of-forward
    // `complete_deferred_experts_into`, or `RsCtx::layer_forward_dump`'s
    // post-dispatch drain).
    Ok(())
}


/// Copy `len` f32s from a shared-storage Metal buffer into a fresh
/// `Vec`. Used by the layer-forward dump path and full-attn host
/// staging where the persistent buffers are bare `metal::Buffer`s
/// (not [`MtlBuffer<f32>`](super::metal::MtlBuffer)). Direct
/// counterpart to [`MtlBuffer::to_vec`](super::metal::MtlBuffer::to_vec)
/// for the unwrapped-buffer case.
///
/// # Safety
///
/// Caller must ensure no GPU command buffer writing to `b` is in
/// flight. Typical discipline: a `wait_until_completed` on the most
/// recent dispatch, or a `complete_deferred_experts_into` drain
/// before the read.
pub(super) fn read_buffer_to_vec(b: &Buffer, len: usize) -> Vec<f32> {
    let ptr = b.contents() as *const f32;
    // SAFETY: see fn docs.
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}

#[allow(clippy::too_many_arguments)]
fn encode_rms_norm_pair(
    cmdbuf: &CommandBufferRef,
    sum_pipe: &ComputePipelineState,
    apply_pipe: &ComputePipelineState,
    input: &Buffer,
    weight_buf: &Buffer,
    weight_off: u64,
    output: &Buffer,
    sum_sq: &Buffer,
    dim: u32,
) {
    {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(sum_pipe);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(sum_sq), 0);
        enc.set_bytes(2, 4, (&dim as *const u32).cast());
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
    {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(apply_pipe);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(weight_buf), weight_off as NSUInteger);
        enc.set_buffer(2, Some(sum_sq), 0);
        enc.set_buffer(3, Some(output), 0);
        let eps = RMS_NORM_EPS;
        enc.set_bytes(4, 4, (&dim as *const u32).cast());
        enc.set_bytes(5, 4, (&eps as *const f32).cast());
        let num_tgs = (dim + 255) / 256;
        enc.dispatch_thread_groups(
            MTLSize::new(num_tgs as NSUInteger, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
}

/// One `swiglu_fused` dispatch into a fresh encoder. Mirrors the
/// shared-expert-FFN swiglu (`infer.m` `cpu_swiglu` at the production
/// path's `infer.m:2977`); replaces the CPU loop between the
/// shared `gate`/`up` matvecs and `shared_down`. Same kernel the
/// K-expert FFN uses (slice 9a — bit-exact per-PSO).
fn encode_swiglu_buf(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    gate: &Buffer,
    up: &Buffer,
    act: &Buffer,
    dim: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(gate), 0);
    enc.set_buffer(1, Some(up), 0);
    enc.set_buffer(2, Some(act), 0);
    enc.set_bytes(3, 4, (&dim as *const u32).cast());
    let num_tgs = (dim + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

fn encode_residual_add(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    dim: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(a), 0);
    enc.set_buffer(1, Some(b), 0);
    enc.set_buffer(2, Some(out), 0);
    enc.set_bytes(3, 4, (&dim as *const u32).cast());
    let num_tgs = (dim + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}
