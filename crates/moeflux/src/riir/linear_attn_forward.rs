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

use super::deferred::{gpu_batched_experts_begin, DeferredError, DeferredState};
use super::expert_forward::MoeBuffers;
use super::expert_io::ExpertFiles;
use super::gpu_linear_attn::{
    encode_compute_decay_beta, encode_conv1d_step, encode_delta_net_step,
    encode_gated_rms_norm, encode_rms_norm_qk, LinearAttnPipelines,
};
use super::gpu_matvec::{encode_matvec, MatvecPipelines, MatvecSpec};
use super::layer_weight_cache::LayerWeightCache;
use super::metal::{MetalBackend, MetalError};
use super::moe_router::moe_router_cpu;
use super::mtl_weight_buf::MtlWeightBuf;
use super::rms_norm::rms_norm_cpu;
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
}

fn zero_f32_buffer(b: &Buffer) {
    let bytes = b.length() as usize;
    // SAFETY: Shared-storage buffer; no GPU work is in flight when
    // memory_clear is called (caller invariant).
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

fn num_full_attn_layers(v: &Variant) -> usize {
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
    /// `buffers.batch_out[6]`. Linear: `linear_total_value`. Full:
    /// `num_attn_heads * head_dim`. Equal on qwen3_5_moe variants.
    pub in_dim: u32,
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
    deferred: &mut Option<DeferredState>,
    layer_idx: usize,
    k_active: usize,
    expert_files: &ExpertFiles,
    _layer_state: &mut LinearAttnState,
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

    // ── snapshot residual + CPU input norm ───────────────────────
    // The C path uses `cpu_rms_norm` here (infer.m:4492) and memcpy's
    // the result into `buf_input` (line 4498). Mirror that exactly so
    // the post-norm bytes match bit-for-bit between sides; the GPU
    // rms_norm chain (which slice 9e tested bit-exact GPU-vs-GPU)
    // would still drift ~5 ULPs vs CPU rms_norm here. Compounded
    // through the projection matvec that drift becomes substantial.
    let input_host = read_buffer_to_vec(&buffers.input, v.hidden_dim);
    {
        let dst = buffers.residual.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(
                input_host.as_ptr(),
                dst,
                v.hidden_dim,
            );
        }
    }
    {
        let mut normed_host = vec![0.0f32; v.hidden_dim];
        let weight_name =
            format!("model.layers.{layer_idx}.input_layernorm.weight");
        rms_norm_cpu(wf, &weight_name, &input_host, &mut normed_host)
            .map_err(|_| LayerForwardError::MissingTensor {
                layer: layer_idx,
                tensor: "input_layernorm.weight (cpu rms_norm failed)",
            })?;
        let dst = buffers.normed.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(
                normed_host.as_ptr(),
                dst,
                v.hidden_dim,
            );
        }
    }

    // ── CMD1: projections + linear-attn pipeline ─────────────────
    {
        let cmdbuf = metal.queue().new_command_buffer();

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
    // in-flight K-expert dispatch in `*deferred`.
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
        OProj {
            w_off: o_w,
            s_off: o_s,
            b_off: o_b,
            bits: o_bits,
            in_dim: v.linear_total_value() as u32,
        },
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
    deferred: &mut Option<DeferredState>,
    layer_idx: usize,
    k_active: usize,
    expert_files: &ExpertFiles,
    o_proj: OProj,
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

    // ── CMD2: o_proj + residual_add + post-attn rms_norm ─────────
    {
        let cmdbuf = metal.queue().new_command_buffer();

        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: o_proj.w_off,
                s_off: o_proj.s_off,
                b_off: o_proj.b_off,
                input: &buffers.batch_out[6],
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
            &buffers.residual,
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
            &buffers.normed, // post-attn-norm output
            &buffers.sum_sq,
            v.hidden_dim as u32,
        );

        cmdbuf.commit();
        cmdbuf.wait_until_completed();
    }

    // ── CMD3a: gate logits + shared-gate score + shared FFN ──────
    {
        let cmdbuf = metal.queue().new_command_buffer();

        // Gate logits: HIDDEN_DIM → NUM_EXPERTS.
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

        // Shared-expert gate scalar: HIDDEN_DIM → 1.
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

        // Shared expert FFN: gate + up matvecs only — swiglu runs on
        // CPU after readback to match the C path's cpu_swiglu
        // (infer.m:2977). Then a final cmdbuf does shared_down on GPU.
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

        cmdbuf.commit();
        cmdbuf.wait_until_completed();
    }

    // ── CPU swiglu of shared_gate × shared_up → shared_act ───────
    {
        let g = read_buffer_to_vec(
            &buffers.shared_gate_out,
            v.shared_intermediate,
        );
        let u = read_buffer_to_vec(
            &buffers.shared_up_out,
            v.shared_intermediate,
        );
        let act_dst = buffers.shared_act.contents() as *mut f32;
        for i in 0..v.shared_intermediate {
            // SiLU(g) * u, matching cpu_swiglu (infer.m:~2400):
            // silu(x) = x / (1 + exp(-x))
            let silu = g[i] / (1.0 + (-g[i]).exp());
            unsafe {
                *act_dst.add(i) = silu * u[i];
            }
        }
    }

    // ── CMD3a-b: shared_down matvec (separate cmdbuf to match C) ─
    {
        let cmdbuf = metal.queue().new_command_buffer();
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

    // ── Load K expert blobs from disk ────────────────────────────
    let k = k_active;
    let expert_size = v.expert_size_4bit();
    let mut expert_data = vec![0u8; k * expert_size];
    for slot in 0..k {
        let expert_idx = indices[slot] as usize;
        let dst =
            &mut expert_data[slot * expert_size..(slot + 1) * expert_size];
        expert_files.read_expert(layer_idx, expert_idx, dst)?;
    }

    // ── CMD3b: K-expert FFN + combine via slice 9b — async ───────
    // Stage host-side mirror copies of the three CMD3 inputs the
    // expert encoders read. `gpu_batched_experts_encode` (called via
    // `_begin`) memcpys these into its own per-call MtlBuffers, so
    // we don't need to keep the host buffers alive past the call.
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
        &normed_host, // h_post (post-attn-norm); experts read this as their input
        &h_mid_host,
        &shared_out_host,
        &weights,
        shared_gate_score,
        layer_idx as i32,
    )?;

    // No write to `buffers.input` here — the dispatch is in flight.
    // The caller drains it (next layer's top-of-forward
    // `complete_deferred_experts_into`, or `RsCtx::layer_forward_dump`'s
    // post-dispatch drain).
    Ok(())
}


pub(super) fn read_buffer_to_vec(b: &Buffer, len: usize) -> Vec<f32> {
    let ptr = b.contents() as *const f32;
    // SAFETY: caller ensures no GPU work in flight on `b`.
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
