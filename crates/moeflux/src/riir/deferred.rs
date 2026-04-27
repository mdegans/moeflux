//! Slice 4e — deferred-experts state machine.
//!
//! Mirrors the production async path inside `fused_layer_forward`
//! (`infer.m:5747..5776`). Where [`super::gpu_batched_experts_forward`]
//! commits + waits + reads back in one synchronous call, this module
//! exposes a three-call API:
//!
//! - [`RsCtx::begin_deferred_experts`] — encode K-expert FFN +
//!   `moe_combine_residual` into one command buffer, commit async,
//!   stash the cmdbuf + per-expert metadata in `RsCtx::deferred`.
//! - [`RsCtx::complete_deferred_experts`] — wait, read back from
//!   `bufs.moe_hidden` into the caller-supplied `hidden_out`, clear
//!   state.
//! - [`RsCtx::discard_deferred_experts`] — wait (so the persistent
//!   `MoeBuffers` are no longer in use), clear state without readback.
//!   Used for prefill tokens whose hidden state is overwritten by the
//!   next token's embedding.
//!
//! The smallest standalone slice — `_begin` is wired up by the diff
//! oracle, not yet by `linear_attn_layer_forward` /
//! `full_attn_layer_forward` (their `post_attention_tail` still calls
//! the synchronous [`super::gpu_batched_experts_forward`]). Slice 4f
//! integrates the begin/complete pair into the per-layer forward when
//! `mf_step_internal` lands.
//!
//! ## Cross-Ctx NaN bug structurally absent
//!
//! In C, `g_deferred` is a file-scope `static` and `g_deferred.hidden`
//! is a raw `float *` into `mf_ctx.hidden`. `mf_free_model` frees the
//! hidden buffer but does not clear `g_deferred`; opening a second
//! `mf_ctx` in the same process and running its first layer
//! dereferences the dangling pointer in `finalize_deferred_experts`
//! and produces all-NaN logits. (Phase 0 bisect finding; documented
//! in `blallama_session_state_pollution.md`.)
//!
//! Lifetime-binding the state to `&mut RsCtx` makes the dangling
//! pointer uncompilable: [`DeferredState`] holds the owned
//! [`metal::CommandBuffer`] only; the destination hidden buffer is
//! caller-supplied to `complete_deferred_experts` rather than stored.
//! A second `RsCtx::open` cannot inherit a previous Ctx's deferred
//! state because there is no shared state to inherit. The cross-Ctx
//! bug class is structurally absent in this port.

use super::expert_forward::{
    gpu_batched_experts_encode, ExpertForwardError, MoeBuffers,
};
use super::metal::MetalBackend;
use super::variants::VARIANT;
use super::{RsCtx, RsError};

/// State carried between [`RsCtx::begin_deferred_experts`] and the
/// matching `complete` / `discard`. Mirrors C
/// `DeferredExpertState` (`infer.m:4015..4030`) for the
/// `gpu_combined = 1` mode — the `h_mid` / `expert_weights` /
/// `valid` / `shared_gate_score` fields used by the CPU-combine
/// branch are omitted; FIXME(riir): port that mode when slice 4f
/// integrates the production fast/slow split.
pub struct DeferredState {
    /// The committed but un-awaited command buffer holding the
    /// per-expert encoders + `moe_combine_residual` dispatch.
    cmd_buffer: metal::CommandBuffer,
    /// Diagnostic only — for parity with C `g_deferred.layer_idx`.
    /// `-1` denotes a synthetic dispatch with no associated layer
    /// (the diff oracle path).
    #[allow(dead_code)]
    layer_idx: i32,
    /// Diagnostic only — number of experts in the active dispatch.
    /// Stored for parity with C's debug-dump path; not used by
    /// `complete` (which reads from `bufs.moe_hidden`, GPU-combined).
    #[allow(dead_code)]
    actual_k: usize,
}

/// Errors from the deferred-experts state machine.
#[derive(Debug, thiserror::Error)]
pub enum DeferredError {
    #[error(
        "a deferred-experts dispatch is already active; call \
         complete_deferred_experts or discard_deferred_experts before \
         begin_deferred_experts"
    )]
    AlreadyActive,
    #[error(
        "hidden_out must be HIDDEN_DIM={expected} floats, got {actual}"
    )]
    BadHiddenOutLen { expected: usize, actual: usize },
    #[error("Metal backend or MoE buffers init failed")]
    Init,
    #[error(transparent)]
    Encode(#[from] ExpertForwardError),
}

impl From<RsError> for DeferredError {
    fn from(_: RsError) -> Self {
        DeferredError::Init
    }
}

/// Free-function variant of [`RsCtx::begin_deferred_experts`]. Takes
/// the disjoint `(metal, bufs, slot)` borrows directly so callers
/// like `post_attention_tail` and `RsCtx::layer_forward_dump` can use
/// it without holding a `&mut RsCtx`. Slice 4f-3 wires this into the
/// per-layer forward.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_batched_experts_begin(
    metal: &mut MetalBackend,
    bufs: &mut MoeBuffers,
    slot: &mut Option<DeferredState>,
    actual_k: i32,
    expert_data: &[u8],
    h_post: &[f32],
    h_mid: &[f32],
    shared_out: &[f32],
    expert_weights: &[f32],
    shared_gate_score: f32,
    layer_idx: i32,
) -> Result<(), DeferredError> {
    if slot.is_some() {
        return Err(DeferredError::AlreadyActive);
    }
    let cmd_buffer = gpu_batched_experts_encode(
        metal,
        bufs,
        actual_k,
        expert_data,
        h_post,
        h_mid,
        shared_out,
        expert_weights,
        shared_gate_score,
    )?;
    cmd_buffer.commit();
    *slot = Some(DeferredState {
        cmd_buffer,
        layer_idx,
        actual_k: actual_k as usize,
    });
    Ok(())
}

/// Free-function variant of [`RsCtx::complete_deferred_experts`].
/// Reads back the post-combine hidden from `bufs.moe_hidden()` into
/// `hidden_out` and clears `slot`. No-op if `slot` is `None`.
pub(crate) fn complete_deferred_experts_into(
    slot: &mut Option<DeferredState>,
    bufs: &MoeBuffers,
    hidden_out: &mut [f32],
) -> Result<(), DeferredError> {
    if hidden_out.len() != VARIANT.hidden_dim {
        return Err(DeferredError::BadHiddenOutLen {
            expected: VARIANT.hidden_dim,
            actual: hidden_out.len(),
        });
    }
    let Some(state) = slot.take() else {
        return Ok(());
    };
    state.cmd_buffer.wait_until_completed();
    hidden_out.copy_from_slice(&bufs.moe_hidden().to_vec());
    Ok(())
}

/// Free-function variant of [`RsCtx::discard_deferred_experts`]. Used
/// to defensively drain `slot` without reading back — both for
/// prefill tokens whose hidden state will be overwritten by the next
/// embedding and as a bracketing guard in `RsCtx::layer_forward_dump`
/// against stale state from a buggy caller.
pub(crate) fn discard_deferred_experts_in(slot: &mut Option<DeferredState>) {
    if let Some(state) = slot.take() {
        state.cmd_buffer.wait_until_completed();
    }
}

impl RsCtx {
    /// Begin an asynchronous K-expert dispatch. Thin wrapper over
    /// [`gpu_batched_experts_begin`] that pulls the disjoint borrows
    /// out of `&mut self`. Kept for the diff-oracle entry path which
    /// holds a `&mut RsCtx` directly.
    ///
    /// `layer_idx` is diagnostic only — pass `-1` for synthetic /
    /// non-layer dispatches.
    ///
    /// Errors:
    /// - [`DeferredError::AlreadyActive`] if a previous deferred
    ///   dispatch is still pending. Mirrors the C path's
    ///   single-active-buffer invariant.
    /// - [`DeferredError::Encode`] for any input-validation failure
    ///   from [`gpu_batched_experts_encode`].
    /// - [`DeferredError::Init`] if Metal / MoE buffers fail to
    ///   initialize on first call.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_deferred_experts(
        &mut self,
        actual_k: i32,
        expert_data: &[u8],
        h_post: &[f32],
        h_mid: &[f32],
        shared_out: &[f32],
        expert_weights: &[f32],
        shared_gate_score: f32,
        layer_idx: i32,
    ) -> Result<(), DeferredError> {
        // Ensure resources first, then re-borrow disjointly so we can
        // pass `&mut self.deferred` alongside the metal+moe pair.
        let _ = self.metal_and_moe_mut()?;
        let Self {
            metal,
            moe_buffers,
            deferred,
            ..
        } = self;
        let metal = metal.as_mut().expect("metal_and_moe_mut just-set");
        let bufs =
            moe_buffers.as_mut().expect("metal_and_moe_mut just-set");
        gpu_batched_experts_begin(
            metal,
            bufs,
            deferred,
            actual_k,
            expert_data,
            h_post,
            h_mid,
            shared_out,
            expert_weights,
            shared_gate_score,
            layer_idx,
        )
    }

    /// Wait for the deferred GPU dispatch, read back the post-combine
    /// hidden state from `bufs.moe_hidden` into `hidden_out`, clear
    /// state. Thin wrapper over [`complete_deferred_experts_into`].
    ///
    /// No-op (returns `Ok(())`) if no deferred dispatch is active —
    /// mirrors C `complete_deferred_experts`'s
    /// `if (!g_deferred.active) return;` guard.
    pub fn complete_deferred_experts(
        &mut self,
        hidden_out: &mut [f32],
    ) -> Result<(), DeferredError> {
        let Self {
            moe_buffers,
            deferred,
            ..
        } = self;
        let Some(bufs) = moe_buffers.as_ref() else {
            // No moe_buffers means no dispatch could have happened.
            return Ok(());
        };
        complete_deferred_experts_into(deferred, bufs, hidden_out)
    }

    /// Wait for the deferred GPU dispatch (so the persistent
    /// `MoeBuffers` are no longer in use by the GPU) and clear state
    /// without reading back. Used in production for prefill tokens
    /// whose hidden state will be overwritten by the next token's
    /// embedding.
    ///
    /// No-op if no deferred dispatch is active.
    pub fn discard_deferred_experts(&mut self) {
        discard_deferred_experts_in(&mut self.deferred);
    }
}
