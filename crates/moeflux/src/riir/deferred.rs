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
    gpu_batched_experts_encode, gpu_batched_experts_encode_pre_staged,
    ExpertForwardError, MoeBuffers,
};
use super::metal::MetalBackend;
use super::variants::VARIANT;
use super::{RsCtx, RsError};

/// Finalize-policy for a deferred K-expert dispatch. Slice 4f-3
/// shipped only [`Self::Gpu`] (CMD3 includes
/// `moe_combine_residual`; `complete` reads from `bufs.moe_hidden`);
/// slice 4f-4 added [`Self::Cpu`] (CMD3 stops after the per-expert
/// FFNs; `complete` reads K outputs from `bufs.out[..k]` and runs
/// the CPU-side combine — sum + sigmoid-gated shared + residual).
///
/// Mirrors the C `g_deferred.gpu_combined` boolean
/// (`infer.m:4015..4030`); the CPU branch carries the inputs the
/// finalize loop needs because they're not stored in any other
/// long-lived buffer between `begin` and `complete`.
pub enum DeferredMode {
    /// `bufs.moe_hidden` is the readback target. Set when the
    /// encode chain included `moe_combine_residual`.
    Gpu,
    /// CPU-side finalize: `complete_deferred_experts_into` reads
    /// `bufs.out[0..k]` plus the host snapshots below and produces
    /// `hidden = h_mid + Σ_k weights[k] * out[k] + sigmoid(gate) *
    /// shared_out`. Mirrors `infer.m:4106..4129`.
    Cpu {
        h_mid: Vec<f32>,
        shared_out: Vec<f32>,
        expert_weights: Vec<f32>,
        shared_gate_score: f32,
    },
}

/// State carried between [`RsCtx::begin_deferred_experts`] and the
/// matching `complete` / `discard`. Mirrors C
/// `DeferredExpertState` (`infer.m:4015..4030`).
pub struct DeferredState {
    /// The committed but un-awaited command buffer holding the
    /// per-expert encoders (and optionally `moe_combine_residual`
    /// when `mode = Gpu`).
    cmd_buffer: metal::CommandBuffer,
    /// Finalize policy — see [`DeferredMode`].
    mode: DeferredMode,
    /// Diagnostic only — for parity with C `g_deferred.layer_idx`.
    /// `-1` denotes a synthetic dispatch with no associated layer
    /// (the diff oracle path).
    #[allow(dead_code)]
    layer_idx: i32,
    /// Number of experts in the active dispatch. Read by the CPU-
    /// combine path to know how many `bufs.out[k]` slots to read.
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
///
/// `gpu_combine` selects the finalize policy (see [`DeferredMode`]).
/// Slice 4f-4 plumbed the parameter through; today every production
/// caller passes `true` (slice 4f-perf will flip per-layer based on
/// `should_gpu_combine`'s C-mirrored conditions). The CPU-combine
/// path is reached only via the diff-oracle test that forces it on.
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
    gpu_combine: bool,
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
        gpu_combine,
    )?;
    cmd_buffer.commit();
    let mode = if gpu_combine {
        DeferredMode::Gpu
    } else {
        // Snapshot the CPU-finalize inputs that the per-expert FFN
        // dispatch doesn't preserve in any long-lived buffer. We
        // could read `bufs.h_mid` / `bufs.shared_out` back at
        // complete-time (they were just staged at begin-time), but
        // a between-call begin/discard could overwrite them — owning
        // the host snapshots keeps complete deterministic. Mirrors
        // C's `g_deferred.h_mid[HIDDEN_DIM]` field.
        DeferredMode::Cpu {
            h_mid: h_mid.to_vec(),
            shared_out: shared_out.to_vec(),
            expert_weights: expert_weights.to_vec(),
            shared_gate_score,
        }
    };
    *slot = Some(DeferredState {
        cmd_buffer,
        mode,
        layer_idx,
        actual_k: actual_k as usize,
    });
    Ok(())
}

/// Pre-staged variant — slice 5d-5. Caller has populated
/// `bufs.data[0..actual_k]` already (typically via `pread` directly
/// into [`MoeBuffers::data_slot_mut`]) AND the GPU input buffers are
/// passed by reference (typically `LayerForwardBuffers.{normed, h_mid,
/// shared_out}` the post-attention path already wrote). Eliminates
/// both the K × EXPERT_SIZE host memcpy AND the 3 × HIDDEN_DIM
/// host↔GPU round-trip that earlier variants paid.
///
/// Always uses GPU combine ([`DeferredMode::Gpu`]). The CPU-finalize
/// path needs host snapshots of `h_mid` / `shared_out` for the
/// finalize pass; if a caller needs that, it routes through the
/// host-slice variant ([`gpu_batched_experts_begin`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_batched_experts_begin_pre_staged(
    metal: &mut MetalBackend,
    bufs: &mut MoeBuffers,
    slot: &mut Option<DeferredState>,
    actual_k: i32,
    input: &metal::BufferRef,
    h_mid: &metal::BufferRef,
    shared_out: &metal::BufferRef,
    expert_weights: &[f32],
    shared_gate_score: f32,
    layer_idx: i32,
    data_set_per_slot: &[super::SlotSource; super::MAX_K],
) -> Result<(), DeferredError> {
    if slot.is_some() {
        return Err(DeferredError::AlreadyActive);
    }
    let cmd_buffer = gpu_batched_experts_encode_pre_staged(
        metal,
        bufs,
        actual_k,
        input,
        h_mid,
        shared_out,
        expert_weights,
        shared_gate_score,
        data_set_per_slot,
    )?;
    cmd_buffer.commit();
    *slot = Some(DeferredState {
        cmd_buffer,
        mode: DeferredMode::Gpu,
        layer_idx,
        actual_k: actual_k as usize,
    });
    Ok(())
}

/// Free-function variant of [`RsCtx::complete_deferred_experts`].
/// Drains the in-flight dispatch into `hidden_out` and clears
/// `slot`. No-op if `slot` is `None`.
///
/// In [`DeferredMode::Gpu`] mode, the readback target is
/// `bufs.moe_hidden`. In [`DeferredMode::Cpu`] mode, the K per-
/// expert outputs are read back from `bufs.out[..k]` and combined
/// CPU-side per `infer.m:4106..4129`:
/// `hidden = h_mid + Σ_k weights[k] × out[k] + sigmoid(gate) × shared_out`.
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
    match state.mode {
        DeferredMode::Gpu => {
            hidden_out.copy_from_slice(&bufs.moe_hidden().to_vec());
        }
        DeferredMode::Cpu {
            h_mid,
            shared_out,
            expert_weights,
            shared_gate_score,
        } => {
            cpu_combine(
                bufs,
                state.actual_k,
                &h_mid,
                &shared_out,
                &expert_weights,
                shared_gate_score,
                hidden_out,
            );
        }
    }
    Ok(())
}

/// CPU-side equivalent of `moe_combine_residual` —
/// `infer.m:4106..4129`. Reads K per-expert outputs from
/// `bufs.out[0..k]`, accumulates the weighted sum into `hidden_out`,
/// adds the sigmoid-gated shared expert and the residual. Bit-
/// identical to the C path modulo `mul_add` contraction (which both
/// sides use; see slice 6 / cpu_ops.rs for the FMA rationale).
fn cpu_combine(
    bufs: &MoeBuffers,
    actual_k: usize,
    h_mid: &[f32],
    shared_out: &[f32],
    expert_weights: &[f32],
    shared_gate_score: f32,
    hidden_out: &mut [f32],
) {
    let dim = VARIANT.hidden_dim;
    debug_assert_eq!(h_mid.len(), dim);
    debug_assert_eq!(shared_out.len(), dim);
    debug_assert_eq!(expert_weights.len(), actual_k);
    debug_assert_eq!(hidden_out.len(), dim);

    // Start with the routed-experts weighted sum. Per-expert
    // readback into a host scratch then a single fmadd pass keeps
    // the inner loop branch-free. C path's order is the same:
    // moe_out[i] starts at 0, accumulates K experts.
    let mut moe_out = vec![0.0f32; dim];
    for k in 0..actual_k {
        let expert_k = bufs.out(k).to_vec();
        debug_assert_eq!(expert_k.len(), dim);
        super::cpu_ops::cpu_vec_madd(
            &mut moe_out,
            &expert_k,
            expert_weights[k],
        );
    }

    // Apply sigmoid gate to shared expert output (in place into a
    // local `gated_shared` so we don't mutate the caller's
    // snapshot). Mirrors `infer.m:4117..4124`.
    let shared_weight = super::cpu_ops::cpu_sigmoid_scalar(shared_gate_score);

    // Final fold: hidden[i] = h_mid[i] + moe_out[i] + shared_weight
    // * shared_out[i]. Single sequential pass — `mul_add` to match
    // clang's likely FMA contraction (per the slice 6 LM-head
    // finding).
    for i in 0..dim {
        let s = shared_out[i].mul_add(shared_weight, moe_out[i]);
        hidden_out[i] = h_mid[i] + s;
    }
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
            /* gpu_combine = */ true,
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
