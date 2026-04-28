//! Pure-Rust port of moeflux's host-side dispatch.
//!
//! Under construction. While the port is in progress, this module
//! coexists with the C-via-`moeflux-sys` path that
//! [`crate::imp::Ctx`] wraps. The differential test harness in
//! `tests/diff_oracle.rs` runs both side-by-side and asserts they
//! agree at well-defined checkpoints — bit-exact for deterministic
//! per-kernel comparisons (Phase 3), within Metal-MoE nondeterminism
//! tolerances for end-to-end logits (Phase 4+).
//!
//! # Status
//!
//! Phase 3 (in progress): bottom-up kernel ports. [`RsCtx::open`]
//! loads only what the kernels ported so far need — currently the
//! `WeightFile` for embedding lookup. Methods that depend on
//! unported kernels still panic with `todo!()`; each kernel landing
//! flips one or more methods to a real impl.
//!
//! See the drama_llama in-repo `riir_moeflux_strategy.md` for the
//! phase breakdown.

#![allow(missing_docs)] // Phase 3 — types fill in incrementally.

use std::path::Path;

pub mod cpu_ops;
pub mod deferred;
pub mod embedding;
pub mod expert_forward;
pub mod expert_io;
pub mod full_attn_forward;
pub mod gpu_attn;
pub mod gpu_linear_attn;
pub mod gpu_lm_head;
pub mod gpu_matvec;
pub mod gpu_norm;
pub mod layer_weight_cache;
pub mod linear_attn;
pub mod linear_attn_forward;
pub mod lm_head;
pub mod metal;
pub mod moe_router;
pub mod mtl_weight_buf;
pub mod prefetch;
pub mod rms_norm;
pub mod rope;
pub mod sdpa;
pub mod state;
pub mod state_snapshot;
pub mod variants;
pub mod weight_file;
pub use deferred::{DeferredError, DeferredState};
pub use embedding::{bf16_to_f32, embed_lookup, EmbeddingError};
pub use expert_forward::{
    gpu_batched_experts_forward, gpu_expert_forward, ExpertForwardError,
    MoeBuffers, MAX_K,
};
pub use expert_io::{ExpertFiles, ExpertIoError};
pub use gpu_attn::{
    gpu_attn_scores_batched, gpu_attn_softmax_batched,
    gpu_attn_values_batched, gpu_sigmoid_gate, GpuAttnError,
};
pub use gpu_lm_head::{GpuLmHead, GpuLmHeadError};
pub use gpu_norm::{gpu_rms_norm_fused, GpuNormError};
pub use linear_attn::{
    conv1d_step, gated_delta_recurrence, rms_norm_bare, rms_norm_gated,
    LinearAttnError,
};
pub use lm_head::{lm_head_cpu, LmHeadError};
pub use metal::{MetalBackend, MetalError, MtlBuffer};
pub use moe_router::{moe_router_cpu, MoeRouterError};
pub use rms_norm::{rms_norm_cpu, rms_norm_per_head_cpu, RmsNormError};
pub use rope::{apply_rotary_emb, RopeError};
pub use layer_weight_cache::LayerWeightCache;
pub use full_attn_forward::full_attn_layer_forward;
pub use linear_attn_forward::{
    linear_attn_layer_forward, linear_layer_idx_for, LayerForwardBuffers,
    LayerForwardError,
};
// Backwards-compat aliases — 4d renamed the buffer struct + error.
#[allow(deprecated)]
pub use linear_attn_forward::{LinearAttnBuffers, LinearAttnForwardError};
pub use mtl_weight_buf::{MtlWeightBuf, MtlWeightBufError};
pub use prefetch::{PrefetchState, PrefetchStatus, SlotSource};
pub use sdpa::{sdpa_cpu, SdpaError};
pub use state::{
    alloc_layer_states, clear_all, pos_max, truncate, KvCache, LayerState,
    LinearAttnState,
};
pub use variants::{Variant, VARIANT};
pub use weight_file::{TensorInfo, WeightFile, WeightFileError};

/// Pure-Rust analogue of [`crate::imp::Ctx`]. API surface mirrors the
/// C wrapper 1:1 during the port — the diff harness compares behavior
/// at this boundary.
///
/// # Phase 3 progress
///
/// Init loads the [`WeightFile`] only (enough for embedding lookup).
/// The per-layer state, expert mmaps, and vocab are deferred to
/// Phase 1b / Phase 4 alongside the kernels that actually consume
/// them. Calling an unported method panics with a `todo!()` pinned
/// to the phase that will implement it.
pub struct RsCtx {
    wf: WeightFile,
    /// Lazily-built Metal backend. CPU-only kernels skip the cost; GPU
    /// kernels (`gpu_expert_forward` and friends) construct it on
    /// first use via [`Self::metal_mut`].
    metal: Option<MetalBackend>,
    /// Lazily-built persistent multi-expert + combine buffer set.
    /// Allocated on first [`Self::gpu_batched_experts_forward`] call;
    /// reused thereafter. ~28 MB on A3B.
    moe_buffers: Option<MoeBuffers>,
    /// Per-layer expert-file handles. Opened eagerly at
    /// [`Self::open`] from `experts_dir/packed_experts/`. Missing
    /// files leave the slot empty per the C path's tolerance
    /// semantics.
    experts: ExpertFiles,
    /// Active top-K (`experts_per_tok` from [`Self::open`]). Mirrors
    /// `mf_ctx.K` — the runtime number of experts to route per
    /// token. The variant's `num_experts_per_tok` is an architectural
    /// MAX; this is the user-selected active value (typically
    /// smaller, e.g. 4 for the dump-hook test even though A3B's
    /// architectural max is 8).
    k_active: usize,
    /// Per-layer KV / linear-attn recurrence state. One entry per
    /// layer; the variant tag matches the C-side
    /// `(i + 1) % FULL_ATTN_INTERVAL == 0` test. Allocated zeroed at
    /// [`Self::open`]; mutated in place by the forward pass and the
    /// `memory_*` ops.
    layer_states: Vec<LayerState>,
    /// Lazily-built `MTLBuffer` wrapping the [`WeightFile`] mmap.
    /// First GPU call constructs it via [`Self::weight_resources_mut`].
    wf_buf: Option<MtlWeightBuf>,
    /// Lazily-built per-layer tensor-offset cache. Per-Ctx (not file-
    /// scope) — the Phase 4b cross-Ctx bug fix.
    layer_caches: Option<Vec<LayerWeightCache>>,
    /// Lazily-built persistent buffer set for the linear-attn forward.
    linear_buffers: Option<LinearAttnBuffers>,
    /// Slice 4e — pending deferred-experts state. `Some` ↔ a
    /// `begin_deferred_experts` call has committed a cmdbuf without
    /// waiting; the matching `complete_deferred_experts` /
    /// `discard_deferred_experts` consumes it. C-side analogue is
    /// the file-scope `g_deferred` global; lifetime-binding to
    /// `RsCtx` here is what eliminates the cross-Ctx NaN bug class
    /// (see [`deferred`] module docs).
    deferred: Option<DeferredState>,
    /// Persistent GPU LM head dispatcher. Lazily built by
    /// [`Self::ensure_linear_resources`] alongside the other GPU
    /// resources. Replaces the per-token `lm_head_cpu` call (which
    /// dominated the 2026-04-27 perf profile at 59% of CPU time).
    lm_head_gpu: Option<GpuLmHead>,
    /// Slice 5d-6 — work-stealing thread pool for parallel K-expert
    /// pread (8 workers on M2 Max P-cores). Eagerly built at
    /// [`Self::open`] so the per-token hot path can `pool.install` /
    /// `pool.spawn` without paying init cost. C analogue is
    /// `g_io_pool` (4 pthreads); we use 8 since M2 Max has 8 P-cores
    /// and there's no contention with other moeflux work.
    io_pool: rayon::ThreadPool,
    /// Slice 5d-6b — speculative-prefetch state machine. One entry
    /// per layer of last-token K indices (used as next-token same-
    /// layer prediction) plus an in-flight async pread handle.
    /// Drained at `memory_clear`, `state_save`, `state_load`, and
    /// `Drop`. See [`prefetch`] module docs for the soundness
    /// argument.
    prefetch: PrefetchState,
    // Future phases populate: vocab.
}

impl RsCtx {
    /// Open a model. Argument order matches [`crate::imp::Ctx::open`].
    ///
    /// Phase 3: only the `weights` + `manifest` paths are consumed
    /// today. The remaining args are accepted for signature stability
    /// — they'll be wired into init as their consuming kernels land.
    pub fn open(
        weights: &Path,
        manifest: &Path,
        _vocab: &Path,
        experts_dir: &Path,
        experts_per_tok: u32,
        _use_2bit: bool,
    ) -> Result<Self, RsError> {
        let wf = WeightFile::open(weights, manifest)
            .map_err(|_| RsError::InitFailed)?;
        let experts = ExpertFiles::open(experts_dir)
            .map_err(|_| RsError::InitFailed)?;
        let layer_states = alloc_layer_states();
        let k_active = (experts_per_tok as usize).clamp(1, VARIANT.num_experts_per_tok);
        let io_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .thread_name(|i| format!("moeflux-io-{}", i))
            .build()
            .map_err(|_| RsError::InitFailed)?;
        let prefetch = PrefetchState::new(VARIANT.num_layers);
        Ok(Self {
            wf,
            metal: None,
            moe_buffers: None,
            experts,
            layer_states,
            k_active,
            wf_buf: None,
            layer_caches: None,
            linear_buffers: None,
            deferred: None,
            lm_head_gpu: None,
            io_pool,
            prefetch,
        })
    }

    /// Build (or return) the Metal backend on demand. CPU-only kernels
    /// don't need it; GPU kernels go through this accessor so the
    /// shader-compile cost is paid lazily on first GPU use.
    fn metal_mut(&mut self) -> Result<&mut MetalBackend, RsError> {
        if self.metal.is_none() {
            self.metal =
                Some(MetalBackend::new().map_err(|_| RsError::InitFailed)?);
        }
        Ok(self.metal.as_mut().expect("just-set"))
    }

    /// Ensure both the Metal backend and the persistent multi-expert
    /// buffer set exist, then return mutable refs to both.
    /// Field-disjoint borrows so two `&mut`s on the same `&mut self`
    /// are valid.
    fn metal_and_moe_mut(
        &mut self,
    ) -> Result<(&mut MetalBackend, &mut MoeBuffers), RsError> {
        if self.metal.is_none() {
            self.metal =
                Some(MetalBackend::new().map_err(|_| RsError::InitFailed)?);
        }
        if self.moe_buffers.is_none() {
            let device =
                self.metal.as_ref().expect("just-set").device().to_owned();
            self.moe_buffers = Some(MoeBuffers::new(&device));
        }
        let Self {
            metal, moe_buffers, ..
        } = self;
        Ok((
            metal.as_mut().expect("just-set"),
            moe_buffers.as_mut().expect("just-set"),
        ))
    }

    pub fn n_vocab(&self) -> usize {
        VARIANT.vocab_size
    }

    pub fn n_ctx(&self) -> usize {
        variants::MAX_SEQ_LEN
    }

    pub fn eos(&self) -> i32 {
        VARIANT.eos_token_1
    }

    pub fn model_name(&self) -> &'static str {
        VARIANT.name
    }

    /// Embed a single token. Writes `HIDDEN_DIM` floats into `out`.
    /// First per-kernel entry point landed in Phase 3; bit-exact
    /// against the C `mf_embed_lookup`.
    pub fn embed(
        &self,
        token_id: i32,
        out: &mut [f32],
    ) -> Result<(), RsError> {
        embed_lookup(&self.wf, token_id, out).map_err(|_| RsError::EvalFailed)
    }

    /// CPU RMSNorm against the weight tensor `weight_name`. `x` and
    /// `out` are both `HIDDEN_DIM` long. Bit-exact against
    /// `mf_rms_norm_cpu` on the same hardware (deterministic CPU
    /// arithmetic, sequential reduction order).
    pub fn rms_norm_cpu(
        &self,
        weight_name: &str,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        rms_norm_cpu(&self.wf, weight_name, x, out)
            .map_err(|_| RsError::EvalFailed)
    }

    /// Per-head CPU RMSNorm, mutating `x_inout` in place. The buffer
    /// holds `num_heads * head_dim` floats (contiguous per head); each
    /// head's slice is RMS-normalized independently and scaled by the
    /// same `head_dim`-long bf16 weight loaded from `weight_name`.
    /// Bit-exact against `mf_rms_norm_per_head_cpu`.
    pub fn rms_norm_per_head_cpu(
        &self,
        weight_name: &str,
        num_heads: usize,
        head_dim: usize,
        x_inout: &mut [f32],
    ) -> Result<(), RsError> {
        rms_norm_per_head_cpu(&self.wf, weight_name, num_heads, head_dim, x_inout)
            .map_err(|_| RsError::EvalFailed)
    }

    /// Apply rotary position embedding to Q and K in place at
    /// position `pos`. `q` is `num_attn_heads * head_dim` floats; `k`
    /// is `num_kv_heads * head_dim`. Bit-exact against
    /// `mf_apply_rotary_emb` on the same hardware.
    pub fn apply_rotary_emb(
        &self,
        pos: i32,
        q: &mut [f32],
        k: &mut [f32],
    ) -> Result<(), RsError> {
        apply_rotary_emb(pos, q, k).map_err(|_| RsError::EvalFailed)
    }

    /// Scaled dot-product attention with sigmoid-gated output, single
    /// query position. ULP-bounded against `mf_sdpa_cpu` (libm `expf`
    /// in softmax + sigmoid sit in the same compiler-choice territory
    /// as RoPE's trig calls).
    pub fn sdpa_cpu(
        &self,
        kv_len: i32,
        q: &[f32],
        q_gate: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        sdpa_cpu(kv_len, q, q_gate, k_cache, v_cache, out)
            .map_err(|_| RsError::EvalFailed)
    }

    /// CPU LM head matvec. `x` is `HIDDEN_DIM` floats (the post-final-
    /// norm hidden state); `out` is `VOCAB_SIZE` floats (raw logits).
    /// Bit-exact target against `mf_lm_head_cpu` on the same hardware.
    pub fn lm_head_cpu(
        &self,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        lm_head_cpu(&self.wf, x, out).map_err(|_| RsError::EvalFailed)
    }

    /// MoE router: softmax → top-K → normalize. `scores` is mutated in
    /// place (post-call it holds the softmaxed probabilities).
    /// `indices` (length `k`) receives the top-K expert IDs in the
    /// selection-sort slot order matching `mf_moe_router_cpu`;
    /// `weights` (length `k`) receives the normalized expert weights.
    /// ULP-bounded against the C path (libm `expf` in softmax).
    pub fn moe_router_cpu(
        &self,
        scores: &mut [f32],
        k: usize,
        indices: &mut [i32],
        weights: &mut [f32],
    ) -> Result<(), RsError> {
        moe_router_cpu(scores, k, indices, weights)
            .map_err(|_| RsError::EvalFailed)
    }

    /// Depthwise 1D conv step + SiLU. `weight_name` references a bf16
    /// tensor of length `channels * kernel_size`. ULP-bounded against
    /// `mf_conv1d_step_cpu` (one libm `expf` per channel in the SiLU
    /// tail; dot product matches clang via `mul_add`).
    pub fn conv1d_step_cpu(
        &self,
        weight_name: &str,
        channels: usize,
        kernel_size: usize,
        conv_state: &[f32],
        new_input: &[f32],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        let bytes = self
            .wf
            .tensor_bytes(weight_name)
            .ok_or(RsError::EvalFailed)?;
        conv1d_step(
            conv_state,
            new_input,
            bytes,
            channels,
            kernel_size,
            out,
        )
        .map_err(|_| RsError::EvalFailed)
    }

    /// Bare CPU RMSNorm (no weight). Bit-exact against
    /// `mf_rms_norm_bare_cpu` on the same hardware.
    pub fn rms_norm_bare_cpu(
        &self,
        eps: f32,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        rms_norm_bare(x, eps, out).map_err(|_| RsError::EvalFailed)
    }

    /// CPU RMSNormGated: `out[i] = rms_norm(x)[i] * w[i] * silu(z[i])`.
    /// ULP-bounded against `mf_rms_norm_gated_cpu` (libm `expf` in SiLU).
    pub fn rms_norm_gated_cpu(
        &self,
        weight_name: &str,
        eps: f32,
        x: &[f32],
        z: &[f32],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        rms_norm_gated(&self.wf, weight_name, x, z, eps, out)
            .map_err(|_| RsError::EvalFailed)
    }

    /// Gated-delta-net recurrence step. Loads `A_log` (f32) and
    /// `dt_bias` (bf16) for the named layer, then runs the per-v-head
    /// decay → kv_mem → delta → state update → output sequence.
    /// `ssm_state` is mutated in place; `out_values` is overwritten.
    /// ULP-bounded against `mf_gated_delta_recurrence_cpu` (libm
    /// `expf`/`logf` per head, `mul_add` matched to clang's FMA).
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_recurrence_cpu(
        &self,
        layer_idx: usize,
        alpha: &[f32],
        beta: &[f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        v_heads: usize,
        k_heads: usize,
        key_dim: usize,
        value_dim: usize,
        ssm_state: &mut [f32],
        out_values: &mut [f32],
    ) -> Result<(), RsError> {
        let a_log_name =
            format!("model.layers.{layer_idx}.linear_attn.A_log");
        let dt_bias_name =
            format!("model.layers.{layer_idx}.linear_attn.dt_bias");
        let a_log_bytes = self
            .wf
            .tensor_bytes(&a_log_name)
            .ok_or(RsError::EvalFailed)?;
        let dt_bias_bytes = self
            .wf
            .tensor_bytes(&dt_bias_name)
            .ok_or(RsError::EvalFailed)?;

        if a_log_bytes.len() != v_heads * 4 {
            return Err(RsError::EvalFailed);
        }
        let mut a_log = vec![0.0f32; v_heads];
        for (i, chunk) in a_log_bytes.chunks_exact(4).enumerate() {
            a_log[i] = f32::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
            ]);
        }

        gated_delta_recurrence(
            &a_log,
            dt_bias_bytes,
            alpha,
            beta,
            q,
            k,
            v,
            v_heads,
            k_heads,
            key_dim,
            value_dim,
            ssm_state,
            out_values,
        )
        .map_err(|_| RsError::EvalFailed)
    }

    /// GPU RMSNorm with bf16 weights (slice 9e). Chains
    /// `rms_norm_sum_sq` + `rms_norm_apply_bf16` into one cmdbuf.
    /// `weight_bf16` is the raw little-endian bf16 byte sequence
    /// (typically from `WeightFile::tensor_bytes(name)`).
    /// First GPU kernel under diff with threadgroup-shared
    /// reduction — empirical question whether this engages the
    /// cosine/Jaccard floors.
    pub fn gpu_rms_norm_fused(
        &mut self,
        x: &[f32],
        weight_bf16: &[u8],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        let metal = self.metal_mut()?;
        gpu_rms_norm_fused(metal, x, weight_bf16, out)
            .map_err(|_| RsError::EvalFailed)
    }

    /// `attn_scores_batched` (slice 5d-7a). Per-head Q · K^T scaled.
    /// Stride-tight oracle entry (`seq_stride = seq_len`).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_scores_batched(
        &mut self,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        q: &[f32],
        k_cache: &[f32],
        scale: f32,
        scores_out: &mut [f32],
    ) -> Result<(), RsError> {
        let metal = self.metal_mut()?;
        gpu_attn_scores_batched(
            metal, num_heads, num_kv_heads, head_dim, seq_len, q, k_cache,
            scale, scores_out,
        )
        .map_err(|_| RsError::EvalFailed)
    }

    /// `attn_softmax_batched` (slice 5d-7a). Per-head softmax over
    /// `[0, seq_len)`, in place.
    pub fn attn_softmax_batched(
        &mut self,
        num_heads: u32,
        seq_len: u32,
        scores_inout: &mut [f32],
    ) -> Result<(), RsError> {
        let metal = self.metal_mut()?;
        gpu_attn_softmax_batched(metal, num_heads, seq_len, scores_inout)
            .map_err(|_| RsError::EvalFailed)
    }

    /// `attn_values_batched` (slice 5d-7a). Per-head scores · V.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_values_batched(
        &mut self,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        scores: &[f32],
        v_cache: &[f32],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        let metal = self.metal_mut()?;
        gpu_attn_values_batched(
            metal, num_heads, num_kv_heads, head_dim, seq_len, scores,
            v_cache, out,
        )
        .map_err(|_| RsError::EvalFailed)
    }

    /// `sigmoid_gate` (slice 5d-7a). `x_inout[i] *= sigmoid(gate[i])`.
    pub fn sigmoid_gate(
        &mut self,
        dim: u32,
        gate: &[f32],
        x_inout: &mut [f32],
    ) -> Result<(), RsError> {
        let metal = self.metal_mut()?;
        gpu_sigmoid_gate(metal, dim, gate, x_inout)
            .map_err(|_| RsError::EvalFailed)
    }

    /// Read one expert's `EXPERT_SIZE`-byte 4-bit blob from disk
    /// (slice 9c). Bypasses every cache; equivalent to a cold pread
    /// against `packed_experts/layer_NN.bin`. Diff-oracle dump point
    /// for the expert-loader.
    pub fn load_expert_bytes(
        &self,
        layer_idx: usize,
        expert_idx: usize,
        out: &mut [u8],
    ) -> Result<(), RsError> {
        self.experts
            .read_expert(layer_idx, expert_idx, out)
            .map_err(|_| RsError::EvalFailed)
    }

    /// Single-expert GPU FFN forward (slice 9a). `expert_data` is one
    /// expert's `EXPERT_SIZE`-byte 4-bit packed blob laid out as
    /// `[gate | up | down]` per `model_variant.h`. `h_post` is the
    /// post-attn-norm hidden state (HIDDEN_DIM floats); `expert_out`
    /// receives the HIDDEN_DIM-float expert output. Cosine/Jaccard
    /// territory against `mf_gpu_expert_forward` (Metal SIMD-reduce
    /// nondeterminism).
    pub fn gpu_expert_forward(
        &mut self,
        expert_data: &[u8],
        h_post: &[f32],
        expert_out: &mut [f32],
    ) -> Result<(), RsError> {
        let metal = self.metal_mut()?;
        gpu_expert_forward(metal, expert_data, h_post, expert_out)
            .map_err(|_| RsError::EvalFailed)
    }

    /// Batched K-expert FFN forward + GPU combine (slice 9b). Encodes
    /// `actual_K` parallel expert FFNs (`gate matvec → up matvec →
    /// SwiGLU → down matvec`) followed by `moe_combine_residual` into
    /// one command buffer. The combine yields
    /// `h_mid + Σ weights[k] × expert_out[k] + sigmoid(gate) × shared_out`.
    /// Cosine/Jaccard territory against `mf_gpu_batched_experts_forward`.
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_batched_experts_forward(
        &mut self,
        actual_k: i32,
        expert_data: &[u8],
        h_post: &[f32],
        h_mid: &[f32],
        shared_out: &[f32],
        expert_weights: &[f32],
        shared_gate_score: f32,
        hidden_out: &mut [f32],
    ) -> Result<(), RsError> {
        let (metal, bufs) = self.metal_and_moe_mut()?;
        gpu_batched_experts_forward(
            metal,
            bufs,
            actual_k,
            expert_data,
            h_post,
            h_mid,
            shared_out,
            expert_weights,
            shared_gate_score,
            hidden_out,
        )
        .map_err(|_| RsError::EvalFailed)
    }

    /// Phase 4 layer-boundary checkpoint hook. Runs a single layer's
    /// forward pass starting from `hidden_in`, returning the post-
    /// layer hidden state in `hidden_out`. The targeted layer's
    /// recurrence state (KV cache for full-attn, conv/SSM state for
    /// linear-attn) is mutated in place. 4c landed the linear-attn
    /// path; 4d added the full-attn path via [`full_attn_layer_forward`].
    ///
    /// Default `gpu_combine = true` (slice 4f-3 production behavior).
    /// Use [`Self::layer_forward_dump_with_gpu_combine`] to exercise
    /// the slice 4f-4 CPU-combine path.
    pub fn layer_forward_dump(
        &mut self,
        layer_idx: i32,
        pos: i32,
        hidden_in: &[f32],
        hidden_out: &mut [f32],
    ) -> Result<(), RsError> {
        self.layer_forward_dump_inner(
            layer_idx, pos, hidden_in, hidden_out, true,
        )
    }

    /// As [`Self::layer_forward_dump`] but lets the caller select
    /// `gpu_combine`. Slice 4f-4 added this entry point so the diff
    /// oracle can exercise the CPU-combine fallback (the path the C
    /// side takes when the next layer's `input_layernorm_w` is
    /// unavailable, or the combine pipelines failed to compile). In
    /// production today every caller uses `true`; slice 4f-perf will
    /// thread the C-mirrored `should_gpu_combine` predicate through
    /// `step_internal`.
    pub fn layer_forward_dump_with_gpu_combine(
        &mut self,
        layer_idx: i32,
        pos: i32,
        hidden_in: &[f32],
        hidden_out: &mut [f32],
        gpu_combine: bool,
    ) -> Result<(), RsError> {
        self.layer_forward_dump_inner(
            layer_idx, pos, hidden_in, hidden_out, gpu_combine,
        )
    }

    fn layer_forward_dump_inner(
        &mut self,
        layer_idx: i32,
        pos: i32,
        hidden_in: &[f32],
        hidden_out: &mut [f32],
        gpu_combine: bool,
    ) -> Result<(), RsError> {
        let v = VARIANT;
        if layer_idx < 0 || (layer_idx as usize) >= v.num_layers {
            return Err(RsError::EvalFailed);
        }
        if pos < 0 {
            return Err(RsError::EvalFailed);
        }
        if hidden_in.len() != v.hidden_dim || hidden_out.len() != v.hidden_dim
        {
            return Err(RsError::EvalFailed);
        }

        let layer_idx_us = layer_idx as usize;
        let is_full =
            v.layer_kind(layer_idx_us) == variants::LayerKind::FullAttn;

        // Ensure all lazy resources exist.
        self.ensure_linear_resources()?;

        // Field-disjoint mutable borrows for the forward call.
        let k_active = self.k_active;
        let Self {
            wf,
            metal,
            moe_buffers,
            experts,
            layer_states,
            wf_buf,
            layer_caches,
            linear_buffers,
            deferred,
            io_pool,
            prefetch,
            ..
        } = self;

        let metal = metal.as_mut().expect("ensure_linear_resources");
        let wf_buf = wf_buf.as_ref().expect("ensure_linear_resources");
        let layer_caches =
            layer_caches.as_ref().expect("ensure_linear_resources");
        let linear_buffers =
            linear_buffers.as_mut().expect("ensure_linear_resources");
        let moe_buffers =
            moe_buffers.as_mut().expect("ensure_linear_resources");
        let io_pool: &rayon::ThreadPool = &*io_pool;

        // Defensive: drain any leaked deferred state from a prior
        // failed call so this entry point stays safe to call
        // back-to-back without `AlreadyActive` errors. After slice
        // 4f-3 the layer forwards leave an in-flight dispatch on
        // return, and a buggy caller might forget to drain — this
        // bracketing guarantees the dump-hook contract (single
        // synchronous layer step) holds regardless.
        //
        // Slice 5d-6b: also drain any in-flight prefetch + clear
        // last-token predictions. The dump hook tests one layer at
        // a time; predictions from a previous test would either be
        // stale or wouldn't match this test's routing anyway.
        deferred::discard_deferred_experts_in(deferred);
        prefetch.invalidate_all();

        // Stage hidden_in into the persistent input buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden_in.as_ptr(),
                linear_buffers.input.contents() as *mut f32,
                v.hidden_dim,
            );
        }

        if is_full {
            let kv_state = match &mut layer_states[layer_idx_us] {
                LayerState::FullAttn(kv) => kv,
                LayerState::LinearAttn(_) => {
                    return Err(RsError::EvalFailed);
                }
            };
            full_attn_layer_forward(
                metal,
                wf,
                wf_buf,
                &layer_caches[layer_idx_us],
                linear_buffers,
                moe_buffers,
                deferred,
                layer_idx_us,
                pos,
                k_active,
                experts,
                io_pool,
                prefetch,
                kv_state,
                gpu_combine,
            )
            .map_err(|_| RsError::EvalFailed)?;
        } else {
            let layer_state = match &mut layer_states[layer_idx_us] {
                LayerState::LinearAttn(la) => la,
                LayerState::FullAttn(_) => {
                    return Err(RsError::EvalFailed);
                }
            };
            linear_attn_layer_forward(
                metal,
                wf,
                wf_buf,
                &layer_caches[layer_idx_us],
                linear_buffers,
                moe_buffers,
                deferred,
                layer_idx_us,
                k_active,
                experts,
                io_pool,
                prefetch,
                layer_state,
                gpu_combine,
            )
            .map_err(|_| RsError::EvalFailed)?;
        }

        // Drain the in-flight K-expert dispatch into `linear_buffers.
        // input` so the existing readback below sees the post-combine
        // hidden state. Slice 4f-3 made the layer forwards async; the
        // dump hook reconstitutes the synchronous single-step contract
        // by completing the dispatch right here.
        // SAFETY: shared-storage buffer; the GPU work for this layer
        // is the dispatch we're about to wait on, and `complete_*` is
        // what does the wait. After it returns, no GPU work is in
        // flight against `linear_buffers.input`.
        let buf_input_slice = unsafe {
            std::slice::from_raw_parts_mut(
                linear_buffers.input.contents() as *mut f32,
                v.hidden_dim,
            )
        };
        deferred::complete_deferred_experts_into(
            deferred,
            moe_buffers,
            buf_input_slice,
        )
        .map_err(|_| RsError::EvalFailed)?;

        // Read post-forward hidden state out of buffers.input.
        unsafe {
            std::ptr::copy_nonoverlapping(
                linear_buffers.input.contents() as *const f32,
                hidden_out.as_mut_ptr(),
                v.hidden_dim,
            );
        }
        Ok(())
    }

    /// 4c diagnostic — runs `layer_forward_dump` and additionally
    /// copies out the post-attn-norm hidden, the post-residual h_mid,
    /// the pre-sigmoid-gate shared expert output, and the shared gate
    /// score. Test-only.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_forward_dump_intermediates(
        &mut self,
        layer_idx: i32,
        pos: i32,
        hidden_in: &[f32],
        hidden_out: &mut [f32],
        h_post_out: Option<&mut [f32]>,
        h_mid_out: Option<&mut [f32]>,
        shared_out_out: Option<&mut [f32]>,
        gate_score_out: Option<&mut f32>,
    ) -> Result<(), RsError> {
        // Run the forward, then read the intermediates from the
        // persistent buffers before the next layer (or the test) can
        // overwrite them.
        self.layer_forward_dump(layer_idx, pos, hidden_in, hidden_out)?;
        let bufs = self
            .linear_buffers
            .as_ref()
            .ok_or(RsError::EvalFailed)?;
        let v = VARIANT;
        let read_into = |buf: &::metal::Buffer, dst: Option<&mut [f32]>| {
            if let Some(dst) = dst {
                let n = dst.len();
                let src = buf.contents() as *const f32;
                // SAFETY: shared storage; no in-flight GPU work because
                // layer_forward_dump waits internally.
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), n);
                }
            }
        };
        read_into(&bufs.normed, h_post_out);
        read_into(&bufs.h_mid, h_mid_out);
        read_into(&bufs.shared_out, shared_out_out);
        if let Some(gate_dst) = gate_score_out {
            let s = bufs.batch_out[5].contents() as *const f32;
            // SAFETY: shared storage.
            *gate_dst = unsafe { *s };
        }
        let _ = v; // silence
        Ok(())
    }

    /// Lazily build the Metal backend, weight buffer, layer caches,
    /// linear-attn persistent buffers, and MoE buffer set. Idempotent
    /// — subsequent calls are no-ops.
    fn ensure_linear_resources(&mut self) -> Result<(), RsError> {
        if self.metal.is_none() {
            self.metal =
                Some(MetalBackend::new().map_err(|_| RsError::InitFailed)?);
        }
        if self.wf_buf.is_none() {
            let device =
                self.metal.as_ref().expect("just-set").device().to_owned();
            self.wf_buf = Some(MtlWeightBuf::wrap(&self.wf, &device));
        }
        if self.layer_caches.is_none() {
            let wf_buf = self.wf_buf.as_ref().expect("just-set");
            let caches = LayerWeightCache::build_all(&self.wf, wf_buf)
                .map_err(|_| RsError::InitFailed)?;
            self.layer_caches = Some(caches);
        }
        if self.linear_buffers.is_none() {
            let device =
                self.metal.as_ref().expect("just-set").device().to_owned();
            self.linear_buffers = Some(LinearAttnBuffers::new(&device));
        }
        if self.moe_buffers.is_none() {
            let device =
                self.metal.as_ref().expect("just-set").device().to_owned();
            self.moe_buffers = Some(MoeBuffers::new(&device));
        }
        if self.lm_head_gpu.is_none() {
            let metal = self.metal.as_mut().expect("just-set");
            let wf_buf = self.wf_buf.as_ref().expect("just-set");
            self.lm_head_gpu = Some(
                GpuLmHead::new(metal, &self.wf, wf_buf)
                    .map_err(|_| RsError::InitFailed)?,
            );
        }
        Ok(())
    }

    /// Process `tokens.len()` tokens at positions `[start_pos,
    /// start_pos + tokens.len())`. Only the final token emits logits
    /// into `logits` (the prefix is state-update-only). Mirrors C
    /// `mf_eval_prompt` (infer.m:7723..7744).
    ///
    /// `seq_id` is accepted for signature parity with the C API and
    /// ignored — moeflux is single-stream.
    ///
    /// Empty `tokens`: returns `Ok(())` without writing `logits` (the
    /// loop body never runs, matching the C-side empty-loop case).
    pub fn eval_prompt(
        &mut self,
        tokens: &[i32],
        start_pos: usize,
        _seq_id: i32,
        logits: &mut [f32],
    ) -> Result<(), RsError> {
        if logits.len() != VARIANT.vocab_size {
            return Err(RsError::EvalFailed);
        }
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = (start_pos + i) as i32;
            let last = i + 1 == tokens.len();
            let logits_arg: Option<&mut [f32]> =
                if last { Some(&mut logits[..]) } else { None };
            self.step_internal(tok, pos, logits_arg)?;
        }
        Ok(())
    }

    /// Decode-style single-token step. Always emits logits. Mirrors C
    /// `mf_eval_token` (infer.m:7746..7757).
    pub fn eval_token(
        &mut self,
        token: i32,
        pos: usize,
        _seq_id: i32,
        logits: &mut [f32],
    ) -> Result<(), RsError> {
        if logits.len() != VARIANT.vocab_size {
            return Err(RsError::EvalFailed);
        }
        self.step_internal(token, pos as i32, Some(logits))
    }

    /// Per-token forward orchestrator. Mirrors C `mf_step_internal`
    /// (infer.m:7687..7721): embed → layer loop → optional drain +
    /// final norm + lm_head. If `logits_out` is `Some`, the deferred
    /// dispatch from the final layer is drained, the result is
    /// `model.norm`-normalized CPU-side, and the LM head writes the
    /// vocabulary-size logits buffer. If `None`, the deferred
    /// dispatch is discarded and no logits are produced.
    ///
    /// Slice 4f-3 made `post_attention_tail` async; this orchestrator
    /// drains the previous layer's dispatch at the top of each
    /// iteration (no-op on iteration 0). Drain target is
    /// `linear_buffers.input` so the next layer's CPU input rms_norm
    /// reads the correct hidden state. The final drain (after the
    /// loop) writes into a host scratch so the model.norm + lm_head
    /// pair don't have to share the GPU buffer.
    fn step_internal(
        &mut self,
        token: i32,
        pos: i32,
        logits_out: Option<&mut [f32]>,
    ) -> Result<(), RsError> {
        let v = VARIANT;
        if pos < 0 {
            return Err(RsError::EvalFailed);
        }
        if let Some(ref l) = logits_out {
            if l.len() != v.vocab_size {
                return Err(RsError::EvalFailed);
            }
        }

        self.ensure_linear_resources()?;
        let k_active = self.k_active;

        // Field-disjoint mutable borrows for the layer loop. Same
        // pattern as `layer_forward_dump_inner`.
        let Self {
            wf,
            metal,
            moe_buffers,
            experts,
            layer_states,
            wf_buf,
            layer_caches,
            linear_buffers,
            deferred,
            lm_head_gpu,
            io_pool,
            prefetch,
            ..
        } = self;
        let metal = metal.as_mut().expect("ensure_linear_resources");
        let wf_buf = wf_buf.as_ref().expect("ensure_linear_resources");
        let layer_caches =
            layer_caches.as_ref().expect("ensure_linear_resources");
        let linear_buffers =
            linear_buffers.as_mut().expect("ensure_linear_resources");
        let moe_buffers =
            moe_buffers.as_mut().expect("ensure_linear_resources");
        let lm_head_gpu =
            lm_head_gpu.as_ref().expect("ensure_linear_resources");
        let io_pool: &rayon::ThreadPool = &*io_pool;

        // Defensive bracket — drain stale state from a buggy prior
        // call so re-entrancy holds.
        deferred::discard_deferred_experts_in(deferred);

        // Embed token into the persistent input buffer in-place.
        // SAFETY: shared-storage buffer; no GPU work is in flight
        // because we just discarded any deferred state.
        {
            let buf_input_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    linear_buffers.input.contents() as *mut f32,
                    v.hidden_dim,
                )
            };
            embedding::embed_lookup(wf, token, buf_input_slice)
                .map_err(|_| RsError::EvalFailed)?;
        }

        // Per-layer loop. Each layer leaves a deferred K-expert
        // dispatch active; the next iteration drains it into
        // `linear_buffers.input` before running its own forward.
        // gpu_combine = true everywhere preserves the slice 4f-3
        // production behavior; slice 4f-perf will gate this on
        // `should_gpu_combine`'s C-mirrored conditions.
        for layer_idx in 0..v.num_layers {
            if layer_idx > 0 {
                // Drain previous layer's deferred dispatch into
                // linear_buffers.input. SAFETY: shared-storage
                // buffer; complete_* waits before reading.
                let buf_input_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        linear_buffers.input.contents() as *mut f32,
                        v.hidden_dim,
                    )
                };
                deferred::complete_deferred_experts_into(
                    deferred,
                    moe_buffers,
                    buf_input_slice,
                )
                .map_err(|_| RsError::EvalFailed)?;
            }

            // Slice 5d-6b: kick off async prefetch for THIS layer
            // using its prediction (= last token's same-layer
            // indices). Runs concurrently with this layer's
            // CMD1+CMD2 GPU compute and finishes before the K-expert
            // dispatch (which `wait_for`s inside post_attention_tail).
            //
            // Ordering is load-bearing — the prefetch must run AFTER
            // layer N-1's deferred drain (above) so the previous
            // layer's GPU read of data_prefetch[slot] is complete
            // before we overwrite it. First token has no predictions,
            // skip the fire (predict_for returns None).
            if let Some(predicted) = prefetch.predict_for(layer_idx) {
                let data_prefetch =
                    moe_buffers.data_prefetch_slots_mut_array();
                prefetch.dispatch(
                    layer_idx,
                    predicted,
                    k_active,
                    data_prefetch,
                    io_pool,
                    experts,
                );
            }

            let is_full = v.layer_kind(layer_idx)
                == variants::LayerKind::FullAttn;
            if is_full {
                let kv_state = match &mut layer_states[layer_idx] {
                    LayerState::FullAttn(kv) => kv,
                    LayerState::LinearAttn(_) => {
                        return Err(RsError::EvalFailed);
                    }
                };
                full_attn_layer_forward(
                    metal,
                    wf,
                    wf_buf,
                    &layer_caches[layer_idx],
                    linear_buffers,
                    moe_buffers,
                    deferred,
                    layer_idx,
                    pos,
                    k_active,
                    experts,
                    io_pool,
                    prefetch,
                    kv_state,
                    /* gpu_combine = */ true,
                )
                .map_err(|_| RsError::EvalFailed)?;
            } else {
                let layer_state = match &mut layer_states[layer_idx] {
                    LayerState::LinearAttn(la) => la,
                    LayerState::FullAttn(_) => {
                        return Err(RsError::EvalFailed);
                    }
                };
                linear_attn_layer_forward(
                    metal,
                    wf,
                    wf_buf,
                    &layer_caches[layer_idx],
                    linear_buffers,
                    moe_buffers,
                    deferred,
                    layer_idx,
                    k_active,
                    experts,
                    io_pool,
                    prefetch,
                    layer_state,
                    /* gpu_combine = */ true,
                )
                .map_err(|_| RsError::EvalFailed)?;
            }
        }

        // Final layer left a deferred dispatch active. Drain (or
        // discard) based on emit policy.
        match logits_out {
            None => {
                deferred::discard_deferred_experts_in(deferred);
            }
            Some(logits) => {
                let mut hidden_final = vec![0.0f32; v.hidden_dim];
                deferred::complete_deferred_experts_into(
                    deferred,
                    moe_buffers,
                    &mut hidden_final,
                )
                .map_err(|_| RsError::EvalFailed)?;

                // Final RMSNorm (`model.norm.weight`) is CPU — small
                // (HIDDEN_DIM = 2048), bit-exact against C per slice 1.
                let mut hidden_normed = vec![0.0f32; v.hidden_dim];
                rms_norm_cpu(
                    wf,
                    "model.norm.weight",
                    &hidden_final,
                    &mut hidden_normed,
                )
                .map_err(|_| RsError::EvalFailed)?;

                // LM head is GPU — was 59% of CPU time per the
                // 2026-04-27 profile. The C path's `lm_head_forward`
                // (infer.m:3090) takes the same Metal route through
                // `dequant_matvec_4bit_v3`; per-PSO bit-exactness
                // (slice 9 finding) keeps the end-to-end logits
                // bit-equal.
                lm_head_gpu
                    .forward(metal, wf_buf, &hidden_normed, logits)
                    .map_err(|_| RsError::EvalFailed)?;
            }
        }

        Ok(())
    }

    /// Reset every layer's state to empty. Mirrors `mf_memory_clear`
    /// (infer.m:7759 → `mf_state_clear_all` infer.m:2271). `seq_id`
    /// is unused — moeflux's sequence-id argument is a no-op stub on
    /// the C side too; KV is single-stream.
    ///
    /// Resets both the host-side per-layer state vector AND the
    /// GPU-side linear-attn recurrence buffers
    /// (`linear_buffers.conv_state` / `delta_state`). The GPU reset
    /// is required because the Rust port treats the GPU buffers as
    /// the canonical recurrence storage (kernels mutate in place,
    /// never read back to host); the C side stores recurrence on the
    /// host and pushes to GPU each call, so resetting host alone
    /// suffices there. Without the GPU reset, back-to-back forwards
    /// after `memory_clear` see stale recurrence and diverge from a
    /// freshly-allocated Ctx.
    pub fn memory_clear(&mut self) {
        clear_all(&mut self.layer_states);
        if let Some(bufs) = self.linear_buffers.as_mut() {
            bufs.reset_recurrence();
        }
        // Slice 5d-6b: drain any in-flight prefetch and clear all
        // last-token predictions. After memory_clear the next token
        // starts from cold-prediction state (no stale predictions
        // from a different prefix).
        self.prefetch.invalidate_all();
    }

    /// Drain any in-flight prefetch and clear all per-layer
    /// last-token predictions. After this call the next forward
    /// starts from cold-prediction state — every K-expert slot in
    /// every layer takes the all-miss (sync-pread) path.
    ///
    /// Slice 5d-6b. Exposed mainly for diff tests that need to force
    /// the all-miss path; production callers shouldn't need this
    /// (prefetch is a perf hint, not a correctness toggle).
    pub fn clear_prefetch_predictions(&mut self) {
        self.prefetch.invalidate_all();
    }

    /// Truncate every layer's state to positions `[0, p0)`. Linear-attn
    /// layers reset to empty (lossy — see `state` module docs and the
    /// FIXME for the Phase 7 typed-error fix). Mirrors
    /// `mf_memory_seq_rm` (infer.m:7752): always returns `true` if the
    /// ctx is valid, since the truncation primitive itself is
    /// infallible.
    pub fn memory_seq_rm(&mut self, _seq_id: i32, p0: i32, p1: i32) -> bool {
        truncate(&mut self.layer_states, p0, p1);
        true
    }

    /// Largest occupied position across full-attn layers, or `-1` if
    /// none has any entries. Mirrors `mf_memory_seq_pos_max`
    /// (infer.m:7759).
    pub fn memory_seq_pos_max(&self, _seq_id: i32) -> i32 {
        pos_max(&self.layer_states)
    }

    /// Bytes the caller must allocate to hold a snapshot of the
    /// current state. Mirrors C `mf_state_size` (infer.m:8505). Grows
    /// linearly with the largest KV length across full-attn layers;
    /// re-query after each evaluation if the state has changed.
    pub fn state_size(&self) -> usize {
        state_snapshot::state_size(&self.layer_states)
    }

    /// Serialize the current state into `buf`. Returns the number of
    /// bytes written. Mirrors C `mf_state_save` (infer.m:8525).
    ///
    /// Drains any pending deferred K-expert dispatch first (the
    /// moeflux.h:481 contract — call only at token boundaries).
    /// Errors if `buf.len() < self.state_size()` or if `linear_
    /// buffers` aren't initialized yet (call `eval_prompt` /
    /// `eval_token` / `memory_clear` once before the first save).
    pub fn state_save(
        &mut self,
        buf: &mut [u8],
    ) -> Result<usize, state_snapshot::StateSnapshotError> {
        // Drain deferred state so the snapshot reflects post-token
        // state, not mid-flight.
        state_snapshot::drain_deferred(&mut self.deferred);
        // Slice 5d-6b: drain any in-flight prefetch (no contribution
        // to the snapshot — predictions are per-token, not part of
        // the wire format — but we need to quiesce the worker pool
        // before any subsequent ctx mutation).
        self.prefetch.drain();
        let linear_buffers = self
            .linear_buffers
            .as_ref()
            .ok_or(state_snapshot::StateSnapshotError::BuffersNotReady)?;
        state_snapshot::state_save(buf, &self.layer_states, linear_buffers)
    }

    /// Replace current state with the one encoded in `buf`. Mirrors
    /// C `mf_state_load` (infer.m:8599). Two-pass: header + per-
    /// layer length preflight before any state is mutated; restore
    /// then memcpys into KV caches and pushes into the GPU
    /// recurrence buffers.
    ///
    /// On error the ctx state is left unchanged (preflight rejects
    /// before the destructive write).
    pub fn state_load(
        &mut self,
        buf: &[u8],
    ) -> Result<(), state_snapshot::StateSnapshotError> {
        // Drain any pending dispatch — load overwrites the state the
        // dispatch was producing for.
        state_snapshot::drain_deferred(&mut self.deferred);
        // Slice 5d-6b: drain any in-flight prefetch + clear
        // last-token predictions. After load, the prefix is whatever
        // the loaded snapshot represents — predictions from the
        // pre-load state would be stale.
        self.prefetch.invalidate_all();
        // Ensure linear_buffers exist so we have somewhere to push
        // the linear-attn recurrence into. Fresh-Ctx state_load
        // before any eval would otherwise hit BuffersNotReady; load
        // is supposed to be a stand-alone restoration primitive.
        self.ensure_linear_resources().map_err(|_| {
            state_snapshot::StateSnapshotError::BuffersNotReady
        })?;
        let Self {
            layer_states,
            linear_buffers,
            ..
        } = self;
        let linear_buffers = linear_buffers
            .as_mut()
            .expect("ensure_linear_resources just ran");
        state_snapshot::state_load(buf, layer_states, linear_buffers)
    }
}

impl std::fmt::Debug for RsCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RsCtx")
            .field("model", &VARIANT.name)
            .field("weights", &self.wf)
            .finish()
    }
}

/// Error type for the Rust port. Mirrors [`crate::imp::Error`] until
/// Phase 7's API cleanup, at which point we'll likely refine
/// (e.g. `CannotTruncateLinear` for the typed `memory_seq_rm`).
#[derive(Debug, thiserror::Error)]
pub enum RsError {
    #[error("path contained an interior NUL byte")]
    PathHasNul,
    #[error("init failed (file missing, mmap, vocab, Metal)")]
    InitFailed,
    #[error("eval call failed")]
    EvalFailed,
    #[error("state save/load failed")]
    StateFailed,
    /// Caller-supplied buffer too small for the snapshot. Variant kept
    /// for API parity with [`crate::imp::Error`]; not yet emitted by
    /// the Rust [`RsCtx::state_save`] (which still returns
    /// `state_snapshot::StateSnapshotError`).
    #[error("state buffer too small (have {have}, need {need})")]
    StateBufferTooSmall {
        /// Bytes the caller provided.
        have: usize,
        /// Bytes the snapshot requires.
        need: usize,
    },
}
