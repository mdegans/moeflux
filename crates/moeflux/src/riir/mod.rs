//! Pure-Rust port of moeflux's host-side dispatch.
//!
//! Under construction. While the port is in progress, this module
//! coexists with the C-via-`moeflux-sys` path that
//! [`crate::imp::Ctx`] wraps. The differential test harness in
//! `tests/diff_oracle.rs` runs both side-by-side and asserts they
//! agree at well-defined checkpoints (prefill logits, post-decode
//! logits) within Metal nondeterminism tolerances.
//!
//! # Status
//!
//! Phase 0: scaffold only. [`RsCtx`] exists as a placeholder so the
//! diff-harness code paths compile; every method panics with
//! `todo!()`. Phase 1+ fills in the implementation bottom-up.
//!
//! See `~/.claude/plans/crystalline-knitting-koala.md` (or the
//! drama_llama in-repo `riir_moeflux_strategy.md`) for the phase
//! breakdown.

#![allow(missing_docs)] // Phase 0 — types fill in incrementally.

use std::path::Path;

/// Pure-Rust analogue of [`crate::imp::Ctx`]. API surface mirrors the
/// C wrapper 1:1 during the port — the diff harness compares behavior
/// at this boundary.
///
/// # Phase 0
///
/// Construction always panics (`todo!()`). The placeholder lets
/// downstream test code spell out the symmetric two-backend test
/// shape today; it'll start working once Phase 1 lands the
/// foundations and Phase 2 stands up the Metal infrastructure.
#[derive(Debug)]
pub struct RsCtx {
    // Fields filled in starting Phase 1 (variants, weight file,
    // vocab) and Phase 2 (Metal device, pipelines, buffers).
    _marker: (),
}

impl RsCtx {
    /// Open a model. Argument order matches [`crate::imp::Ctx::open`].
    pub fn open(
        _weights: &Path,
        _manifest: &Path,
        _vocab: &Path,
        _experts_dir: &Path,
        _experts_per_tok: u32,
        _use_2bit: bool,
    ) -> Result<Self, RsError> {
        todo!("RIIR Phase 1+: open model from disk")
    }

    pub fn n_vocab(&self) -> usize {
        todo!("RIIR Phase 1: variants + weight_file")
    }

    pub fn n_ctx(&self) -> usize {
        todo!("RIIR Phase 1: variants")
    }

    pub fn eos(&self) -> i32 {
        todo!("RIIR Phase 1: variants")
    }

    pub fn model_name(&self) -> &'static str {
        todo!("RIIR Phase 1: variants")
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
