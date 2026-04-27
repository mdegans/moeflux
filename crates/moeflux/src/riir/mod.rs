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
pub mod lm_head;
pub mod metal;
pub mod rms_norm;
pub mod rope;
pub mod sdpa;
pub mod variants;
pub mod weight_file;
pub use embedding::{bf16_to_f32, embed_lookup, EmbeddingError};
pub use lm_head::{lm_head_cpu, LmHeadError};
pub use metal::{MetalBackend, MetalError, MtlBuffer};
pub use rms_norm::{rms_norm_cpu, rms_norm_per_head_cpu, RmsNormError};
pub use rope::{apply_rotary_emb, RopeError};
pub use sdpa::{sdpa_cpu, SdpaError};
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
    // Future phases populate: metal: MetalBackend, vocab,
    // per-layer state, expert mmaps, working buffers.
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
        _experts_dir: &Path,
        _experts_per_tok: u32,
        _use_2bit: bool,
    ) -> Result<Self, RsError> {
        let wf = WeightFile::open(weights, manifest)
            .map_err(|_| RsError::InitFailed)?;
        Ok(Self { wf })
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

    pub fn memory_clear(&mut self) {
        todo!("RIIR Phase 4: state management")
    }

    pub fn memory_seq_rm(&mut self, _seq_id: i32, _p0: i32, _p1: i32) -> bool {
        todo!("RIIR Phase 4: state management")
    }

    pub fn memory_seq_pos_max(&self, _seq_id: i32) -> i32 {
        todo!("RIIR Phase 4: state management")
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
