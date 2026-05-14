//! GPU port of [`super::moe_router::moe_router_cpu`] — softmax →
//! selection-sort top-K → divide-by-sum normalize, all on the GPU
//! without a host bounce.
//!
//! Phase A of the graph-mode plan
//! ([`qwen_graph_mode_session6_plan.md`]): the routing host bounce
//! at every layer is what forces the per-layer commit boundary
//! today. Moving the router on-GPU is the *enabler* for Phase B's
//! graph-mode submission; on its own it's near-perf-neutral.
//!
//! This module only emits the encoder. The orchestrator still
//! decides whether to dispatch the GPU router or the CPU oracle
//! (a future flag toggles per call); Phase B will swap callsites.
//!
//! ## Diff target
//!
//! Slot order is the running-min replacement order from the CPU
//! oracle (see `moe_router_cpu`'s `cpu_topk` block), so per-slot
//! comparison is well-defined. Floating-point drift comes only from
//! the `exp` reduction order — softmax happens in tg-mem with a
//! tree-shaped reduction on GPU vs. a left-to-right scan on CPU.
//! The diff battery asserts:
//!
//! - Indices: bit-exact set (sorted) — magnitude separation between
//!   adjacent expert scores dominates the per-element ULP drift.
//! - Weights: cosine ≥ 0.9999 — values match within the softmax
//!   reduction-order tolerance.

use metal::{
    Buffer, CommandBufferRef, ComputePipelineState, MTLSize, NSUInteger,
};

use super::metal::{MetalContext, MetalError};

/// Pipelines used by [`encode_moe_router`]. Fetch once per orchestrator
/// scope so subsequent calls are O(1).
pub struct MoeRouterPipelines {
    pub softmax_topk: ComputePipelineState,
    pub normalize: ComputePipelineState,
}

impl MoeRouterPipelines {
    pub fn fetch(metal: &mut MetalContext) -> Result<Self, MetalError> {
        let softmax_topk = metal.pipeline("moe_softmax_topk")?.clone();
        let normalize = metal.pipeline("moe_normalize_weights")?.clone();
        Ok(Self {
            softmax_topk,
            normalize,
        })
    }
}

/// Encode the full GPU router pipeline (softmax + top-K + normalize) into
/// `cmdbuf`. No commit_and_wait — the caller controls the cmdbuf boundary.
///
/// `logits` is the gate-matvec output stacked across `n_tokens` rows of
/// `n_experts` f32 each (row-major). `indices_out` and `weights_out` are
/// caller-owned `n_tokens * k` element output buffers (i32 and f32
/// respectively).
///
/// The softmax dispatch picks `tg_size = 64` threads per token — a single
/// SIMD group covers 32 lanes for the parallel max/sum reductions; 64 keeps
/// per-token work well above the launch overhead without burning extra
/// lanes idling on a 128-expert vector. The selection-sort tail runs on
/// lane 0 only and is the same wall-clock cost regardless of `tg_size`.
///
/// `k` is bounded by the kernel-side `MAX_K = 16`; current models use 8.
/// `n_experts` is bounded by `MAX_EXPERTS = 512`; current models use ≤ 256.
pub fn encode_moe_router(
    cmdbuf: &CommandBufferRef,
    pipes: &MoeRouterPipelines,
    logits: &Buffer,
    indices_out: &Buffer,
    weights_out: &Buffer,
    n_tokens: u32,
    n_experts: u32,
    k: u32,
) {
    // Stage 1: per-token softmax + selection-sort top-K.
    {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipes.softmax_topk);
        enc.set_buffer(0, Some(logits), 0);
        enc.set_buffer(1, Some(indices_out), 0);
        enc.set_buffer(2, Some(weights_out), 0);
        enc.set_bytes(3, 4, (&n_experts as *const u32).cast());
        enc.set_bytes(4, 4, (&k as *const u32).cast());
        enc.dispatch_thread_groups(
            MTLSize::new(n_tokens as NSUInteger, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();
    }

    // Stage 2: per-token weight normalize (divide-by-sum, guarded).
    {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipes.normalize);
        enc.set_buffer(0, Some(weights_out), 0);
        enc.set_bytes(1, 4, (&k as *const u32).cast());
        enc.dispatch_thread_groups(
            MTLSize::new(n_tokens as NSUInteger, 1, 1),
            MTLSize::new(k as NSUInteger, 1, 1),
        );
        enc.end_encoding();
    }
}
