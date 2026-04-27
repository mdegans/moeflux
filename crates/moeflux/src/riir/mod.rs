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

pub mod embedding;
pub mod expert_forward;
pub mod expert_io;
pub mod gpu_norm;
pub mod linear_attn;
pub mod lm_head;
pub mod metal;
pub mod moe_router;
pub mod rms_norm;
pub mod rope;
pub mod sdpa;
pub mod state;
pub mod variants;
pub mod weight_file;
pub use embedding::{bf16_to_f32, embed_lookup, EmbeddingError};
pub use expert_forward::{
    gpu_batched_experts_forward, gpu_expert_forward, ExpertForwardError,
    MoeBuffers, MAX_K,
};
pub use expert_io::{ExpertFiles, ExpertIoError};
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
    /// Per-layer KV / linear-attn recurrence state. One entry per
    /// layer; the variant tag matches the C-side
    /// `(i + 1) % FULL_ATTN_INTERVAL == 0` test. Allocated zeroed at
    /// [`Self::open`]; mutated in place by the forward pass and the
    /// `memory_*` ops.
    layer_states: Vec<LayerState>,
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
        _experts_per_tok: u32,
        _use_2bit: bool,
    ) -> Result<Self, RsError> {
        let wf = WeightFile::open(weights, manifest)
            .map_err(|_| RsError::InitFailed)?;
        let experts = ExpertFiles::open(experts_dir)
            .map_err(|_| RsError::InitFailed)?;
        let layer_states = alloc_layer_states();
        Ok(Self {
            wf,
            metal: None,
            moe_buffers: None,
            experts,
            layer_states,
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
    /// layer hidden state in `hidden_out`. The targeted layer's KV /
    /// recurrence state is mutated in place. Lands in 4c (linear-attn)
    /// / 4d (full-attn) once `fused_layer_forward` is ported.
    pub fn layer_forward_dump(
        &mut self,
        _layer_idx: i32,
        _pos: i32,
        _hidden_in: &[f32],
        _hidden_out: &mut [f32],
    ) -> Result<(), RsError> {
        todo!("RIIR Phase 4c/4d: fused_layer_forward")
    }

    pub fn eval_prompt(
        &mut self,
        _tokens: &[i32],
        _start_pos: usize,
        _seq_id: i32,
        _logits: &mut [f32],
    ) -> Result<(), RsError> {
        todo!("RIIR Phase 4: forward-pass top-level")
    }

    pub fn eval_token(
        &mut self,
        _token: i32,
        _pos: usize,
        _seq_id: i32,
        _logits: &mut [f32],
    ) -> Result<(), RsError> {
        todo!("RIIR Phase 4: forward-pass top-level")
    }

    /// Reset every layer's state to empty. Mirrors `mf_memory_clear`
    /// (infer.m:7747). `seq_id` is unused — moeflux's sequence-id
    /// argument is a no-op stub on the C side too; KV is single-stream.
    pub fn memory_clear(&mut self) {
        clear_all(&mut self.layer_states);
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
}
