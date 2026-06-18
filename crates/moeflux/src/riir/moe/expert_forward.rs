//! GPU MoE expert FFN dispatch — slices 9a + 9b.
//!
//! Two entry points:
//!
//! - [`gpu_expert_forward`] (slice 9a) runs one expert end-to-end with
//!   transient `MtlBuffer` allocation per call. Used for diff-oracle
//!   bring-up of the four-dispatch sequence.
//! - [`gpu_batched_experts_forward`] (slice 9b) runs K experts in
//!   parallel across one command buffer, then a `moe_combine_residual`
//!   dispatch that adds h_mid + Σ weights × expert_out + sigmoid(gate)
//!   × shared_out. Uses persistent buffers ([`MoeBuffers`]) sized for
//!   the architectural max `K = 16`.
//!
//! The single-expert path mirrors `gpu_expert_forward` in
//! `metal_infer/infer.m`:
//!
//! 1. `dequant_matvec_4bit_v3` over `gate` → `gate_out` `[MOE_INTERMEDIATE]`
//! 2. `dequant_matvec_4bit_v3` over `up`   → `up_out`   `[MOE_INTERMEDIATE]`
//! 3. `swiglu_fused(gate_out, up_out)`     → `act`      `[MOE_INTERMEDIATE]`
//! 4. `dequant_matvec_4bit_v3` over `down` → `expert_out` `[HIDDEN_DIM]`
//!
//! The batched path mirrors `gpu_encode_experts_batched` followed by
//! the production `moe_combine_residual` dispatch (the path that
//! `fused_layer_forward` takes when GPU combine is on, minus the
//! RMSNorm fusion — that's slice 9e).
//!
//! ## Tolerance regime
//!
//! First GPU kernel under diff. Per the strategy's three-band split,
//! GPU kernels live in cosine/Jaccard territory because Metal's
//! threadgroup reduction and SIMD-group sum order are not specified
//! to be deterministic across pipeline-state recompiles. The diff
//! oracle test uses `cosine ≥ 0.9999` and
//! `max_abs_diff ≤ 1e-3 × max_abs_out` against the same C-side
//! pipelines.
//!
//! ## 4-bit only
//!
//! `g_use_2bit` selects a different pipeline (`matvec_2bit`) and a
//! different expert-block layout (`EXPERT_SIZE_2BIT`) on the C side.
//! Surfacing it through the diff oracle is a separate slice; today
//! this module hard-codes the 4-bit pipeline and the 4-bit offsets.
//!
//! FIXME(riir): port the 2-bit path before Phase 6 cutover or the
//! consumer drops 2-bit support — `MoefluxEngine` currently exposes
//! `use_2bit` so users can opt in.

use metal::{Buffer, BufferRef, CommandBufferRef, ComputePipelineState, MTLSize, NSUInteger};

use moeflux_metal::{GatherQmmCall, Kernels, QuantWeights};

use crate::riir::backend::buftype::DeprecatedCogitoBuf;
use crate::riir::backend::gpu::gpu_matvec::{MatvecPipelines, encode_matvec_n_tokens};
use crate::riir::backend::gpu::gpu_norm::{RmsNormBf16Pipelines, encode_rms_norm_bf16_into};
use crate::riir::backend::gpu::metal::{
    MetalContext, MetalError, MtlBuffer, buffer_as_mut_slice, buffer_as_slice,
};
use crate::riir::backend::{BufId, BufferPool, MetalBufferPool};
use crate::riir::moe::moe_router::ExpertBuckets;
use crate::riir::variants::{GROUP_SIZE, SharedExpertGate, VARIANT, Variant};

/// Pipeline name for the MoE combine kernel — varies by variant.
/// Cogito-V2 / DeepSeek-V3 use the unscaled raw-add path; Qwen3 MoE
/// keeps the sigmoid-gated path. Mirrors `deepseek_moe_cpu`'s
/// unconditional `out[i] += shared[i]` (moe_cpu.rs:168-173).
fn combine_kernel_name() -> &'static str {
    match VARIANT.shared_expert_gate {
        SharedExpertGate::SigmoidGate => "moe_combine_residual",
        SharedExpertGate::Unscaled => "moe_combine_residual_unscaled",
    }
}

/// Chained-norm targets for slice 5d-8. When `Some`, the K-expert
/// encoder rebinds `moe_combine_residual` to write into `combine_out`
/// (instead of `bufs.moe_hidden`) and appends `rms_norm_sum_sq` +
/// `rms_norm_apply_bf16` so the next layer's normed input is ready
/// when this cmdbuf completes. Mirrors C's `gpu_combine` path
/// (`infer.m:5677..5750`). `combine_out` doubles as combine output
/// target and chain rms_norm input — in production this is
/// `linear_buffers.input`, the same buffer that serves as the next
/// layer's residual source for CMD2.
pub(crate) struct ChainToNormed<'a> {
    pub pipes: &'a RmsNormBf16Pipelines,
    pub wf_buf: &'a Buffer,
    pub next_norm_off: u64,
    pub combine_out: &'a Buffer,
    pub chain_sum_sq: &'a Buffer,
    pub chain_normed: &'a Buffer,
    pub eps: f32,
}

/// Architectural maximum top-K. Mirrors `MAX_K` in `infer.m` (16).
/// Sets the slot count of [`MoeBuffers`] and the binding-table width
/// of `moe_combine_residual` (which expects 16 expert-output buffers
/// regardless of the active `K`).
pub const MAX_K: usize = 16;

/// Host-side per-token inputs to a batched K-expert dispatch.
///
/// `gpu_batched_experts_forward` / `_encode` and
/// `gpu_batched_experts_begin` all thread these five values in
/// lockstep; bundling them keeps the call sites legible. All slices
/// are `[HIDDEN_DIM]` except `expert_weights` (`[actual_K]`).
#[derive(Clone, Copy)]
pub struct ExpertPayload<'a> {
    /// Shared input to every expert's matvec (post-attn-norm hidden).
    pub h_post: &'a [f32],
    /// Residual added by the combine kernel.
    pub h_mid: &'a [f32],
    /// The shared expert's output.
    pub shared_out: &'a [f32],
    /// Per-slot routing weights.
    pub expert_weights: &'a [f32],
    /// Pre-sigmoid gate logit for the shared expert.
    pub shared_gate_score: f32,
}

/// Errors from GPU expert FFN dispatch (slice 9a + 9b).
#[derive(Debug, thiserror::Error)]
pub enum ExpertForwardError {
    #[error(
        "expert_data is the wrong length: expected {expected} bytes \
         (4-bit layout), got {actual}"
    )]
    BadExpertDataLen { expected: usize, actual: usize },
    #[error("h_post must be HIDDEN_DIM={expected} floats, got {actual}")]
    BadHPostLen { expected: usize, actual: usize },
    #[error("expert_out must be HIDDEN_DIM={expected} floats, got {actual}")]
    BadExpertOutLen { expected: usize, actual: usize },
    #[error("h_mid must be HIDDEN_DIM={expected} floats, got {actual}")]
    BadHMidLen { expected: usize, actual: usize },
    #[error("shared_out must be HIDDEN_DIM={expected} floats, got {actual}")]
    BadSharedOutLen { expected: usize, actual: usize },
    #[error("hidden_out must be HIDDEN_DIM={expected} floats, got {actual}")]
    BadHiddenOutLen { expected: usize, actual: usize },
    #[error("actual_K out of range: must be 1..={max}, got {actual}")]
    BadK { actual: i32, max: usize },
    #[error("expert_weights must be {expected} floats, got {actual}")]
    BadWeightsLen { expected: usize, actual: usize },
    #[error("Metal backend: {0}")]
    Metal(#[from] MetalError),
}

/// One expert's FFN forward on the GPU. `expert_data` is `EXPERT_SIZE`
/// bytes laid out as `[gate_block | up_block | down_block]` per the
/// 4-bit packing in `model_variant.h`. `h_post` is the post-attention-
/// norm hidden state (HIDDEN_DIM floats); `expert_out` receives the
/// expert's HIDDEN_DIM-float output.
///
/// Allocates four transient `MtlBuffer`s per call (data, input, gate,
/// up, act, out). At ~5 MB total this is fine for the diff-oracle test
/// path; persistent reuse arrives with slice 9b.
pub fn gpu_expert_forward(
    metal: &mut MetalContext,
    expert_data: &[u8],
    h_post: &[f32],
    expert_out: &mut [f32],
) -> Result<(), ExpertForwardError> {
    let v = VARIANT;
    let expected_data_len = v.expert_size_4bit();
    if expert_data.len() != expected_data_len {
        return Err(ExpertForwardError::BadExpertDataLen {
            expected: expected_data_len,
            actual: expert_data.len(),
        });
    }
    if h_post.len() != v.hidden_dim {
        return Err(ExpertForwardError::BadHPostLen {
            expected: v.hidden_dim,
            actual: h_post.len(),
        });
    }
    if expert_out.len() != v.hidden_dim {
        return Err(ExpertForwardError::BadExpertOutLen {
            expected: v.hidden_dim,
            actual: expert_out.len(),
        });
    }

    // Compile / fetch pipelines first; nothing else holds &mut self.
    let matvec = metal.pipeline("dequant_matvec_4bit_v3")?.clone();
    let swiglu = metal.pipeline("swiglu_fused")?.clone();

    let device = metal.device();

    // Buffers. `data` holds the full expert blob; the matvec dispatches
    // bind it at three different offsets (weights / scales / biases),
    // mirroring the C side's single `buf_expert_data` shared across
    // dispatches.
    let data = MtlBuffer::<u8>::with_data(device, expert_data);
    let input = MtlBuffer::<f32>::with_data(device, h_post);
    let gate_out = MtlBuffer::<f32>::with_len(device, v.moe_intermediate);
    let up_out = MtlBuffer::<f32>::with_len(device, v.moe_intermediate);
    let act = MtlBuffer::<f32>::with_len(device, v.moe_intermediate);
    let out = MtlBuffer::<f32>::with_len(device, v.hidden_dim);

    let cmdbuf = metal.queue().new_command_buffer();

    encode_matvec(
        cmdbuf,
        &matvec,
        &data,
        v.gate_w_off_4bit(),
        v.gate_s_off_4bit(),
        v.gate_b_off_4bit(),
        &input,
        &gate_out,
        v.moe_intermediate as u32,
        v.hidden_dim as u32,
    );

    encode_matvec(
        cmdbuf,
        &matvec,
        &data,
        v.up_w_off_4bit(),
        v.up_s_off_4bit(),
        v.up_b_off_4bit(),
        &input,
        &up_out,
        v.moe_intermediate as u32,
        v.hidden_dim as u32,
    );

    encode_swiglu(
        cmdbuf,
        &swiglu,
        &gate_out,
        &up_out,
        &act,
        v.moe_intermediate as u32,
    );

    encode_matvec(
        cmdbuf,
        &matvec,
        &data,
        v.down_w_off_4bit(),
        v.down_s_off_4bit(),
        v.down_b_off_4bit(),
        &act,
        &out,
        v.hidden_dim as u32,
        v.moe_intermediate as u32,
    );

    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    expert_out.copy_from_slice(&out.to_vec());
    Ok(())
}

/// One `dequant_matvec_4bit_v3` dispatch into a fresh encoder.
/// Threadgroup config matches `gpu_expert_forward` in `infer.m`:
/// 8 rows per threadgroup × 256 threads (8 SIMD groups of 32 lanes).
fn encode_matvec(
    cmdbuf: &metal::CommandBufferRef,
    pipeline: &metal::ComputePipelineState,
    data: &MtlBuffer<u8>,
    w_off: usize,
    s_off: usize,
    b_off: usize,
    input: &MtlBuffer<f32>,
    output: &MtlBuffer<f32>,
    out_dim: u32,
    in_dim: u32,
) {
    let group_size = GROUP_SIZE as u32;
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(data.raw()), w_off as NSUInteger);
    enc.set_buffer(1, Some(data.raw()), s_off as NSUInteger);
    enc.set_buffer(2, Some(data.raw()), b_off as NSUInteger);
    enc.set_buffer(3, Some(input.raw()), 0);
    enc.set_buffer(4, Some(output.raw()), 0);
    enc.set_bytes(5, 4, (&out_dim as *const u32).cast());
    enc.set_bytes(6, 4, (&in_dim as *const u32).cast());
    enc.set_bytes(7, 4, (&group_size as *const u32).cast());
    let num_tgs = (out_dim + 7) / 8;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

/// One `swiglu_fused` dispatch. 256 threads per threadgroup; the
/// kernel guards `tid >= dim` so the tail threadgroup is safe.
fn encode_swiglu(
    cmdbuf: &metal::CommandBufferRef,
    pipeline: &metal::ComputePipelineState,
    gate: &MtlBuffer<f32>,
    up: &MtlBuffer<f32>,
    act: &MtlBuffer<f32>,
    dim: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(gate.raw()), 0);
    enc.set_buffer(1, Some(up.raw()), 0);
    enc.set_buffer(2, Some(act.raw()), 0);
    enc.set_bytes(3, 4, (&dim as *const u32).cast());
    let num_tgs = (dim + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

// ---------------------------------------------------------------------------
// Slice 9b — persistent multi-expert buffers + batched dispatch
// ---------------------------------------------------------------------------

/// Persistent GPU buffer set for the batched K-expert path. Mirrors
/// the multi-expert + combine buffers on `MetalCtx` in `infer.m`:
///
/// - `data[k]`  — one expert's `EXPERT_SIZE` packed bytes, slot k.
/// - `gate[k]`  — slot k's gate matvec output, `MOE_INTERMEDIATE` floats.
/// - `up[k]`    — slot k's up matvec output, `MOE_INTERMEDIATE` floats.
/// - `act[k]`   — slot k's SwiGLU activation, `MOE_INTERMEDIATE` floats.
/// - `out[k]`   — slot k's down matvec output, `HIDDEN_DIM` floats.
/// - `input`    — shared post-attn-norm hidden (`HIDDEN_DIM` floats).
/// - `h_mid`, `shared_out`, `moe_hidden` — combine inputs / output.
/// - `combine_params` — 18-float buffer for `moe_combine_residual`:
///   layout `[weights[0..16], shared_gate_score, padding]`.
///
/// Allocated once and reused across every batched call. Total ~28 MB
/// for A3B (dominated by `MAX_K × EXPERT_SIZE` ≈ 27 MB) for slice
/// 5d-6a; ~56 MB for slice 5d-6b after `data_prefetch` is added; ~63 MB
/// for slice 5d-9 after the prefetch set ping-pongs.
///
/// The data sets have **fixed roles**:
/// - [`Self::data_synced`] is the on-demand sync-pread target. The
///   GPU dispatch reads from this set for slots whose actual expert
///   index missed the prefetch prediction.
/// - [`Self::data_prefetch`] is the speculative prefetch target —
///   slice 5d-9 widened it to TWO physical sets indexed by `set: 0|1`.
///   Layer N's prefetch writes set `N % 2`; layer N+1's writes the
///   OTHER set. With the depth-2 deferred ring, layer N+1's prefetch
///   can fire while layer N's K-expert is still reading set `N % 2`
///   from the GPU, because the two layers reference different
///   physical buffers — no race. By the time layer N+2's prefetch
///   writes set `N % 2` again, the depth-2 ring has drained layer
///   N's deferred so set `N % 2` is safe to overwrite.
/// Per-token cogito MoE path buffers — every field tagged
/// [`DeprecatedCogitoBuf`].
///
/// This struct is **frozen** pending M5 Studio (need 512 GB unified RAM
/// for cogito 600B-class models). When the cogito path is rewritten
/// post-M5, retype the fields properly (per-role tags from
/// [`crate::riir::backend::buftype`]). Do not iterate on this path
/// without retagging — the single-tag scheme exists because every
/// `pool.alloc` here lives behind the same legacy code path; splitting
/// the tags ad hoc would invite the bug class the rest of the backend
/// just eliminated.
pub struct MoeBuffers {
    /// Sync-pread (miss) target. See type-level docs. `u8`-typed
    /// (expert blob bytes); 2 MiB aligned via `pool.alloc_aligned`.
    data_synced: [BufId<DeprecatedCogitoBuf>; MAX_K],
    /// Async-prefetch (hit) target — two parity-keyed sets. Layer N's
    /// prefetch writes set `N % 2`; the encoder for layer N reads from
    /// the same set. See type-level docs for the soundness argument.
    /// 2 MiB aligned via `pool.alloc_aligned`.
    data_prefetch: [[BufId<DeprecatedCogitoBuf>; MAX_K]; 2],
    gate: [BufId<DeprecatedCogitoBuf>; MAX_K],
    up: [BufId<DeprecatedCogitoBuf>; MAX_K],
    act: [BufId<DeprecatedCogitoBuf>; MAX_K],
    out: [BufId<DeprecatedCogitoBuf>; MAX_K],
    /// Flat [k_active, moe_intermediate] buffers for batched expert dispatch.
    gate_flat: BufId<DeprecatedCogitoBuf>,
    up_flat: BufId<DeprecatedCogitoBuf>,
    act_flat: BufId<DeprecatedCogitoBuf>,
    /// Flat [k_active, hidden_dim] buffer for batched expert down output.
    out_flat: BufId<DeprecatedCogitoBuf>,
    input: BufId<DeprecatedCogitoBuf>,
    h_mid: BufId<DeprecatedCogitoBuf>,
    shared_out: BufId<DeprecatedCogitoBuf>,
    moe_hidden: BufId<DeprecatedCogitoBuf>,
    combine_params: BufId<DeprecatedCogitoBuf>,
    /// Phase 2 (cogito-v2 full-GPU): MoE router-gate logits buffer,
    /// `[num_experts]` f32. The GPU `bf16_matvec` dispatch writes
    /// here; CPU `noaux_tc` routing reads it back via
    /// [`Self::gate_logits_to_vec`]. ~1 KB at num_experts=256.
    gate_logits: BufId<DeprecatedCogitoBuf>,
}

impl MoeBuffers {
    /// Allocate the full multi-expert + combine buffer set in the pool.
    /// Sizes come from the active [`Variant`] and architectural
    /// `MAX_K`. All buffers are persistent — they survive
    /// `pool.reset_transient()` and live for the lifetime of the
    /// `RsCtx`. Buffers are zeroed by the pool on alloc; every call to
    /// [`gpu_batched_experts_forward`] writes the slots it uses before
    /// dispatch.
    ///
    /// S10b-pre-3: migrated from `&Device` + `MtlBuffer<T>` ownership
    /// to `&mut MetalBufferPool` + `BufId`. Pread DMA destinations
    /// (`data_synced`, `data_prefetch`) use `pool.alloc_aligned` with
    /// 2 MiB alignment per the C path's 3.6× DMA win
    /// (`metal_infer/infer.m:1196`).
    pub fn new(pool: &mut MetalBufferPool) -> Self {
        let v: Variant = VARIANT;
        const TWO_MIB: usize = 2 * 1024 * 1024;
        let f32_size = std::mem::size_of::<f32>();
        let data_synced: [BufId<DeprecatedCogitoBuf>; MAX_K] = std::array::from_fn(|_| {
            pool.alloc_aligned(v.expert_size_4bit(), TWO_MIB, "moe.data_synced", true)
        });
        let data_prefetch: [[BufId<DeprecatedCogitoBuf>; MAX_K]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                pool.alloc_aligned(v.expert_size_4bit(), TWO_MIB, "moe.data_prefetch", true)
            })
        });
        // Defense-in-depth: assert 2 MiB alignment held. The pool's
        // `alloc_aligned` is posix_memalign-equivalent by construction;
        // a release-mode failure would indicate a global-allocator bug
        // we don't try to recover from. The probe accepts a slice of
        // BufIds and resolves each through the pool.
        let probe = |label: &str, ids: &[BufId<DeprecatedCogitoBuf>]| {
            for (slot, &id) in ids.iter().enumerate() {
                let addr = pool.handle(id).contents() as usize;
                debug_assert_eq!(
                    addr % TWO_MIB,
                    0,
                    "data_{label} slot {slot} not 2 MB aligned (contents=0x{addr:x})",
                );
            }
        };
        probe("synced", &data_synced[..]);
        probe("prefetch[0]", &data_prefetch[0][..]);
        probe("prefetch[1]", &data_prefetch[1][..]);
        let gate: [BufId<DeprecatedCogitoBuf>; MAX_K] = std::array::from_fn(|_| {
            pool.alloc(v.moe_intermediate * f32_size, "moe.gate", true)
                .expect("pool.alloc moe.gate")
        });
        let up: [BufId<DeprecatedCogitoBuf>; MAX_K] = std::array::from_fn(|_| {
            pool.alloc(v.moe_intermediate * f32_size, "moe.up", true)
                .expect("pool.alloc moe.up")
        });
        let act: [BufId<DeprecatedCogitoBuf>; MAX_K] = std::array::from_fn(|_| {
            pool.alloc(v.moe_intermediate * f32_size, "moe.act", true)
                .expect("pool.alloc moe.act")
        });
        let out: [BufId<DeprecatedCogitoBuf>; MAX_K] = std::array::from_fn(|_| {
            pool.alloc(v.hidden_dim * f32_size, "moe.out", true)
                .expect("pool.alloc moe.out")
        });
        let k_max = v.num_experts_per_tok;
        let gate_flat = pool
            .alloc(k_max * v.moe_intermediate * f32_size, "moe.gate_flat", true)
            .expect("pool.alloc moe.gate_flat");
        let up_flat = pool
            .alloc(k_max * v.moe_intermediate * f32_size, "moe.up_flat", true)
            .expect("pool.alloc moe.up_flat");
        let act_flat = pool
            .alloc(k_max * v.moe_intermediate * f32_size, "moe.act_flat", true)
            .expect("pool.alloc moe.act_flat");
        let out_flat = pool
            .alloc(k_max * v.hidden_dim * f32_size, "moe.out_flat", true)
            .expect("pool.alloc moe.out_flat");
        let input: BufId<DeprecatedCogitoBuf> = pool
            .alloc(v.hidden_dim * f32_size, "moe.input", true)
            .expect("pool.alloc moe.input");
        let h_mid: BufId<DeprecatedCogitoBuf> = pool
            .alloc(v.hidden_dim * f32_size, "moe.h_mid", true)
            .expect("pool.alloc moe.h_mid");
        let shared_out: BufId<DeprecatedCogitoBuf> = pool
            .alloc(v.hidden_dim * f32_size, "moe.shared_out", true)
            .expect("pool.alloc moe.shared_out");
        let moe_hidden: BufId<DeprecatedCogitoBuf> = pool
            .alloc(v.hidden_dim * f32_size, "moe.moe_hidden", true)
            .expect("pool.alloc moe.moe_hidden");
        let combine_params: BufId<DeprecatedCogitoBuf> = pool
            .alloc(18 * f32_size, "moe.combine_params", true)
            .expect("pool.alloc moe.combine_params");
        let gate_logits: BufId<DeprecatedCogitoBuf> = pool
            .alloc(v.num_experts.max(1) * f32_size, "moe.gate_logits", true)
            .expect("pool.alloc moe.gate_logits");
        Self {
            data_synced,
            data_prefetch,
            gate,
            up,
            act,
            out,
            gate_flat,
            up_flat,
            act_flat,
            out_flat,
            input,
            h_mid,
            shared_out,
            moe_hidden,
            combine_params,
            gate_logits,
        }
    }

    // -------- BufId accessors (graph-mode addressing) --------

    pub(crate) fn moe_hidden_id(&self) -> BufId<DeprecatedCogitoBuf> {
        self.moe_hidden
    }
    pub(crate) fn out_id(&self, slot: usize) -> BufId<DeprecatedCogitoBuf> {
        self.out[slot]
    }
    pub(crate) fn gate_id(&self, slot: usize) -> BufId<DeprecatedCogitoBuf> {
        self.gate[slot]
    }
    pub(crate) fn up_id(&self, slot: usize) -> BufId<DeprecatedCogitoBuf> {
        self.up[slot]
    }
    pub(crate) fn act_id(&self, slot: usize) -> BufId<DeprecatedCogitoBuf> {
        self.act[slot]
    }
    pub(crate) fn h_mid_id(&self) -> BufId<DeprecatedCogitoBuf> {
        self.h_mid
    }
    pub(crate) fn shared_out_id(&self) -> BufId<DeprecatedCogitoBuf> {
        self.shared_out
    }
    pub(crate) fn combine_params_id(&self) -> BufId<DeprecatedCogitoBuf> {
        self.combine_params
    }
    pub(crate) fn data_synced_id(&self, slot: usize) -> BufId<DeprecatedCogitoBuf> {
        self.data_synced[slot]
    }
    pub(crate) fn data_prefetch_id(&self, set: usize, slot: usize) -> BufId<DeprecatedCogitoBuf> {
        debug_assert!(set < 2);
        debug_assert!(slot < MAX_K);
        self.data_prefetch[set][slot]
    }
    // -------- Flat buffer accessors (batched expert dispatch) --------

    pub(crate) fn gate_flat_id(&self) -> BufId<DeprecatedCogitoBuf> {
        self.gate_flat
    }
    pub(crate) fn up_flat_id(&self) -> BufId<DeprecatedCogitoBuf> {
        self.up_flat
    }
    pub(crate) fn act_flat_id(&self) -> BufId<DeprecatedCogitoBuf> {
        self.act_flat
    }
    pub(crate) fn out_flat_id(&self) -> BufId<DeprecatedCogitoBuf> {
        self.out_flat
    }

    // -------- &Buffer accessors via pool (imperative encoders) --------

    /// All per-slot data_synced buffers as disjoint `&mut [u8]` views
    /// for parallel pread. SAFETY: caller ensures no GPU dispatch is
    /// reading from any slot.
    pub(crate) fn data_synced_slots_mut_array<'p>(
        &self,
        pool: &'p MetalBufferPool,
    ) -> [&'p mut [u8]; MAX_K] {
        pool.as_mut_slices_u8(self.data_synced)
    }

    /// All per-slot data_prefetch buffers in `set` as disjoint
    /// `&mut [u8]` views. Same SAFETY contract as
    /// [`Self::data_synced_slots_mut_array`].
    pub(crate) fn data_prefetch_slots_mut_array<'p>(
        &self,
        pool: &'p MetalBufferPool,
        set: usize,
    ) -> [&'p mut [u8]; MAX_K] {
        debug_assert!(set < 2, "prefetch set index must be 0 or 1");
        pool.as_mut_slices_u8(self.data_prefetch[set])
    }

    /// Owned-form accessor for the post-attn-norm input buffer.
    pub(crate) fn input_buffer<'p>(&self, pool: &'p MetalBufferPool) -> &'p metal::Buffer {
        pool.handle(self.input)
    }

    /// Owned-form accessor for the residual-input buffer.
    pub(crate) fn h_mid_buffer<'p>(&self, pool: &'p MetalBufferPool) -> &'p metal::Buffer {
        pool.handle(self.h_mid)
    }

    /// Owned-form accessor for the shared-expert output buffer.
    pub(crate) fn shared_out_buffer<'p>(&self, pool: &'p MetalBufferPool) -> &'p metal::Buffer {
        pool.handle(self.shared_out)
    }

    /// Stage `hidden` into `bufs.input` (host copy into shared-storage
    /// Metal buffer). Caller must ensure no GPU work in flight.
    pub(crate) fn stage_host_input(&self, pool: &MetalBufferPool, hidden: &[f32]) {
        debug_assert_eq!(hidden.len(), VARIANT.hidden_dim);
        let buf = pool.handle(self.input);
        // SAFETY: same discipline as `MtlBuffer::as_mut_slice` — caller
        // ensures no concurrent GPU read. Shared-storage buffer is
        // alive while `buf` is held; alignment of f32 on the
        // posix_memalign / Metal allocation is guaranteed.
        let dst: &mut [f32] = unsafe { buffer_as_mut_slice::<f32>(buf, VARIANT.hidden_dim) };
        dst.copy_from_slice(hidden);
    }

    /// Zero `bufs.h_mid`. Used when the layer's residual contribution
    /// should equal `Σ moe + shared_out` exactly. Caller must ensure
    /// no GPU work touches `bufs.h_mid` concurrently — call before
    /// encoding the layer's command buffer or after waiting on the
    /// previous one.
    pub(crate) fn stage_host_h_mid_zero(&self, pool: &MetalBufferPool) {
        let buf = pool.handle(self.h_mid);
        // SAFETY: see method docs — caller's GPU-quiescence invariant
        // covers the canonical `buffer_as_mut_slice` contract.
        let dst: &mut [f32] = unsafe { buffer_as_mut_slice::<f32>(buf, VARIANT.hidden_dim) };
        dst.fill(0.0);
    }

    /// Read back `bufs.moe_hidden` to a host `Vec<f32>`. Caller must
    /// ensure the cmdbuf that wrote it has completed.
    pub(crate) fn moe_hidden_to_vec(&self, pool: &MetalBufferPool) -> Vec<f32> {
        let buf = pool.handle(self.moe_hidden);
        // SAFETY: see method docs — caller has drained the writing
        // cmdbuf, so no concurrent GPU access. Forwards to
        // `buffer_as_slice`.
        let src: &[f32] = unsafe { buffer_as_slice::<f32>(buf, VARIANT.hidden_dim) };
        src.to_vec()
    }

    /// Owned-form accessor for the post-combine hidden buffer. Phase 5
    /// GPU residual stream uses this to read the MoE output directly
    /// into a downstream `residual_add` dispatch without a host bounce.
    pub fn moe_hidden_ref<'p>(&self, pool: &'p MetalBufferPool) -> &'p metal::Buffer {
        pool.handle(self.moe_hidden)
    }

    /// Owned-form accessor for the router-gate-logits buffer. Used as
    /// the output target by the GPU `bf16_matvec` dispatch.
    pub(crate) fn gate_logits_buffer<'p>(&self, pool: &'p MetalBufferPool) -> &'p metal::Buffer {
        pool.handle(self.gate_logits)
    }

    /// Read back `bufs.gate_logits` to a host `Vec<f32>`. Caller must
    /// ensure the cmdbuf that wrote it has completed.
    pub(crate) fn gate_logits_to_vec(&self, pool: &MetalBufferPool) -> Vec<f32> {
        let buf = pool.handle(self.gate_logits);
        let n = VARIANT.num_experts.max(1);
        // SAFETY: see method docs — caller has drained the writing
        // cmdbuf. Forwards to `buffer_as_slice`.
        let src: &[f32] = unsafe { buffer_as_slice::<f32>(buf, n) };
        src.to_vec()
    }
}

impl std::fmt::Debug for MoeBuffers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeBuffers")
            .field("max_k", &MAX_K)
            .field("hidden_dim", &VARIANT.hidden_dim)
            .field("moe_intermediate", &VARIANT.moe_intermediate)
            .field("expert_size_4bit", &VARIANT.expert_size_4bit())
            .finish()
    }
}

/// Batched K-expert FFN forward + GPU combine. Single command buffer:
/// 2K expert encoders ([`gpu_encode_experts_batched`'s shape]) followed
/// by one `moe_combine_residual` dispatch. Reads back the
/// `HIDDEN_DIM`-float post-combine hidden state.
///
/// Inputs:
///
/// - `expert_data` — `actual_K * EXPERT_SIZE` bytes, K expert blobs in
///   slot order.
/// - `h_post` — `[HIDDEN_DIM]` shared input to every expert's matvec.
/// - `h_mid` — `[HIDDEN_DIM]` residual added by the combine.
/// - `shared_out` — `[HIDDEN_DIM]` shared expert's output.
/// - `expert_weights` — `[actual_K]` routing weights.
/// - `shared_gate_score` — pre-sigmoid gate logit for the shared
///   expert.
/// - `hidden_out` — `[HIDDEN_DIM]` post-combine hidden state.
///
/// Cosine/Jaccard tolerance regime against the C-side
/// `mf_gpu_batched_experts_forward`. Floor placeholders today —
/// empirically this kernel pair lacks atomic ops (`weighted_sum` / the
/// combine kernel's `Σ_k weights[k] * expert_out_k[tid]` loop are both
/// per-thread sequential), so it may also land bit-exact like 9a.
#[allow(clippy::too_many_arguments)]
pub fn gpu_batched_experts_forward(
    metal: &mut MetalContext,
    bufs: &mut MoeBuffers,
    buffer_pool: &MetalBufferPool,
    actual_k: i32,
    expert_data: &[u8],
    payload: ExpertPayload<'_>,
    hidden_out: &mut [f32],
) -> Result<(), ExpertForwardError> {
    let v = VARIANT;
    if hidden_out.len() != v.hidden_dim {
        return Err(ExpertForwardError::BadHiddenOutLen {
            expected: v.hidden_dim,
            actual: hidden_out.len(),
        });
    }
    let cmdbuf = gpu_batched_experts_encode(
        metal,
        bufs,
        buffer_pool,
        actual_k,
        expert_data,
        payload,
        /* gpu_combine = */ true,
    )?;
    cmdbuf.commit();
    cmdbuf.wait_until_completed();
    hidden_out.copy_from_slice(&bufs.moe_hidden_to_vec(buffer_pool));
    Ok(())
}

/// Encode the K-expert FFN (and optionally `moe_combine_residual`)
/// into a fresh command buffer. Stages caller inputs into `bufs`,
/// returns the (uncommitted) owned command buffer.
///
/// `gpu_combine`:
/// - `true` — encode the combine kernel as the final dispatch;
///   `bufs.moe_hidden` holds the post-combine hidden state on
///   completion. This is the slice 9b shape and the default for
///   `post_attention_tail`.
/// - `false` — omit the combine kernel; the per-expert outputs
///   remain in `bufs.out[0..k]` for a CPU-side combine in
///   [`super::deferred::complete_deferred_experts_into`]. Mirrors C
///   `gpu_combine = 0` (`infer.m:5668..5673` decision; finalize at
///   `infer.m:4106..4129`). Used by the slice 4f-4 CPU-combine path
///   when next layer's `input_layernorm_w` is missing or the
///   pipelines aren't available.
///
/// Callers decide commit + wait policy:
///
/// - [`gpu_batched_experts_forward`] commits + waits + reads back
///   from `bufs.moe_hidden` synchronously.
/// - The slice 4e deferred-experts state machine (see
///   [`super::deferred`]) commits async and stashes the cmdbuf in
///   `RsCtx::deferred` for a later `complete` / `discard` call.
///
/// Mirrors C `oracle_batched_experts_encode` (`infer.m` slice 4e
/// refactor).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_batched_experts_encode(
    metal: &mut MetalContext,
    bufs: &mut MoeBuffers,
    buffer_pool: &MetalBufferPool,
    actual_k: i32,
    expert_data: &[u8],
    payload: ExpertPayload<'_>,
    _gpu_combine: bool,
) -> Result<metal::CommandBuffer, ExpertForwardError> {
    let ExpertPayload {
        h_post,
        h_mid,
        shared_out,
        expert_weights,
        shared_gate_score,
    } = payload;
    let v = VARIANT;
    validate_inputs(actual_k, expert_data, expert_weights)?;
    let k = actual_k as usize;
    if h_post.len() != v.hidden_dim {
        return Err(ExpertForwardError::BadHPostLen {
            expected: v.hidden_dim,
            actual: h_post.len(),
        });
    }
    if h_mid.len() != v.hidden_dim {
        return Err(ExpertForwardError::BadHMidLen {
            expected: v.hidden_dim,
            actual: h_mid.len(),
        });
    }
    if shared_out.len() != v.hidden_dim {
        return Err(ExpertForwardError::BadSharedOutLen {
            expected: v.hidden_dim,
            actual: shared_out.len(),
        });
    }

    // Compile/fetch pipelines first so no `&mut metal` borrow holds across
    // encoder construction. Both 4-bit matvec PSOs are fetched so the
    // emit helper can pick v3 (in_dim ≤ 4096) vs fast (> 4096) per
    // dispatch — Cogito-V2's hidden_dim=7168 needs `_fast` for the
    // gate/up matvecs while down (in_dim=2048) stays on v3.
    let matvec = MatvecPipelines::fetch(metal)?;
    let swiglu = metal.pipeline("swiglu_fused")?.clone();
    let v3_experts = metal
        .pipeline("dequant_matvec_4bit_v3_experts")
        .ok()
        .cloned();
    let combine_flat = metal.pipeline("moe_combine_residual_flat")?.clone();

    let expert_size = v.expert_size_4bit();
    for slot in 0..k {
        let src = &expert_data[slot * expert_size..(slot + 1) * expert_size];
        let dst = buffer_pool.as_mut_slice_u8(bufs.data_synced_id(slot));
        dst.copy_from_slice(src);
    }
    bufs.stage_host_input(buffer_pool, h_post);
    {
        let buf = buffer_pool.handle(bufs.h_mid_id());
        let dst: &mut [f32] = unsafe { buffer_as_mut_slice::<f32>(buf, v.hidden_dim) };
        dst.copy_from_slice(h_mid);
    }
    {
        let buf = buffer_pool.handle(bufs.shared_out_id());
        let dst: &mut [f32] = unsafe { buffer_as_mut_slice::<f32>(buf, v.hidden_dim) };
        dst.copy_from_slice(shared_out);
    }
    {
        let buf = buffer_pool.handle(bufs.combine_params_id());
        let params: &mut [f32] = unsafe { buffer_as_mut_slice::<f32>(buf, 18) };
        params.fill(0.0);
        params[..k].copy_from_slice(expert_weights);
        params[16] = shared_gate_score;
    }

    let bindings: Vec<(&Buffer, u64)> = (0..k)
        .map(|slot| (buffer_pool.handle(bufs.data_synced_id(slot)), 0u64))
        .collect();

    let cmdbuf = metal.queue().new_command_buffer();
    let input_buf = bufs.input_buffer(buffer_pool).clone();
    let h_mid_buf = bufs.h_mid_buffer(buffer_pool).clone();
    let shared_out_buf = bufs.shared_out_buffer(buffer_pool).clone();
    emit_batched_experts(
        cmdbuf,
        &matvec,
        &swiglu,
        v3_experts.as_ref(),
        &combine_flat,
        bufs,
        buffer_pool,
        &input_buf,
        &h_mid_buf,
        &shared_out_buf,
        k,
        v,
        &bindings,
        None,
    );
    Ok(cmdbuf.to_owned())
}

/// Encode K-expert dispatch using mmap'd expert buffers directly.
/// No staging, no pread — expert weights are read by the GPU from
/// the mmap'd Metal buffers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_batched_experts_encode_mmap(
    metal: &mut MetalContext,
    bufs: &mut MoeBuffers,
    buffer_pool: &MetalBufferPool,
    actual_k: i32,
    input: &BufferRef,
    h_mid: &BufferRef,
    shared_out: &BufferRef,
    expert_weights: &[f32],
    shared_gate_score: f32,
    expert_bindings: &[(&Buffer, u64)],
    chain: Option<ChainToNormed<'_>>,
) -> Result<metal::CommandBuffer, ExpertForwardError> {
    let v = VARIANT;
    if actual_k < 1 || (actual_k as usize) > MAX_K {
        return Err(ExpertForwardError::BadK {
            actual: actual_k,
            max: MAX_K,
        });
    }
    let k = actual_k as usize;
    if expert_weights.len() != k {
        return Err(ExpertForwardError::BadWeightsLen {
            expected: k,
            actual: expert_weights.len(),
        });
    }

    let matvec = MatvecPipelines::fetch(metal)?;
    let swiglu = metal.pipeline("swiglu_fused")?.clone();
    let v3_experts = metal
        .pipeline("dequant_matvec_4bit_v3_experts")
        .ok()
        .cloned();
    let combine_flat = metal.pipeline("moe_combine_residual_flat")?.clone();

    {
        let buf = buffer_pool.handle(bufs.combine_params_id());
        let params: &mut [f32] = unsafe { buffer_as_mut_slice::<f32>(buf, 18) };
        params.fill(0.0);
        params[..k].copy_from_slice(expert_weights);
        params[16] = shared_gate_score;
    }

    let cmdbuf = metal.queue().new_command_buffer();
    emit_batched_experts(
        cmdbuf,
        &matvec,
        &swiglu,
        v3_experts.as_ref(),
        &combine_flat,
        bufs,
        buffer_pool,
        input,
        h_mid,
        shared_out,
        k,
        v,
        expert_bindings,
        chain,
    );
    Ok(cmdbuf.to_owned())
}

/// Shared input validation for both encode entry points.
fn validate_inputs(
    actual_k: i32,
    expert_data: &[u8],
    expert_weights: &[f32],
) -> Result<(), ExpertForwardError> {
    let v = VARIANT;
    if actual_k < 1 || (actual_k as usize) > MAX_K {
        return Err(ExpertForwardError::BadK {
            actual: actual_k,
            max: MAX_K,
        });
    }
    let k = actual_k as usize;
    let expected_data_len = k * v.expert_size_4bit();
    if expert_data.len() != expected_data_len {
        return Err(ExpertForwardError::BadExpertDataLen {
            expected: expected_data_len,
            actual: expert_data.len(),
        });
    }
    if expert_weights.len() != k {
        return Err(ExpertForwardError::BadWeightsLen {
            expected: k,
            actual: expert_weights.len(),
        });
    }
    Ok(())
}

/// Inner encoder helper — emits the K-expert FFN dispatches and
/// combine into `cmdbuf`. Expert weights come from mmap'd Metal
/// buffers via `expert_bindings` (buffer + byte offset per expert).
#[allow(clippy::too_many_arguments)]
fn emit_batched_experts(
    cmdbuf: &CommandBufferRef,
    matvec: &MatvecPipelines,
    swiglu: &ComputePipelineState,
    v3_experts: Option<&ComputePipelineState>,
    combine_flat: &ComputePipelineState,
    bufs: &MoeBuffers,
    buffer_pool: &MetalBufferPool,
    input: &BufferRef,
    h_mid: &BufferRef,
    shared_out: &BufferRef,
    k: usize,
    v: Variant,
    expert_bindings: &[(&Buffer, u64)],
    chain: Option<ChainToNormed<'_>>,
) {
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::Relaxed) {
            // The buffer the kernel reads from is whatever the caller
            // bound (mmap layer slice in Mmap mode, `data_synced` /
            // `data_prefetch` slot in Pread mode); this log line is
            // about the dispatch *shape*, not the IO source.
            eprintln!(
                "[moe] expert dispatch: hidden_dim={} k={} v3_experts={} → {}",
                v.hidden_dim,
                k,
                v3_experts.is_some(),
                if v.hidden_dim <= 4096 && k <= MAX_K && v3_experts.is_some() {
                    "batched"
                } else {
                    "per-expert (fallback)"
                },
            );
        }
    }

    let use_batched = v.hidden_dim <= 4096 && k <= MAX_K && v3_experts.is_some();

    if use_batched {
        let v3e = v3_experts.unwrap();
        let gate_flat = buffer_pool.handle(bufs.gate_flat_id());
        let up_flat = buffer_pool.handle(bufs.up_flat_id());
        let act_flat = buffer_pool.handle(bufs.act_flat_id());
        let out_flat = buffer_pool.handle(bufs.out_flat_id());
        let group_size = GROUP_SIZE as u32;
        let k_u32 = k as u32;

        // Encoder 1: gate + up (shared input, two dispatches)
        {
            let enc = cmdbuf.new_compute_command_encoder();
            encode_matvec_experts(
                enc,
                v3e,
                expert_bindings,
                input,
                gate_flat,
                v.moe_intermediate as u32,
                v.hidden_dim as u32,
                group_size,
                v.gate_w_off_4bit() as u32,
                v.gate_s_off_4bit() as u32,
                v.gate_b_off_4bit() as u32,
                k_u32,
                0,
            );
            encode_matvec_experts(
                enc,
                v3e,
                expert_bindings,
                input,
                up_flat,
                v.moe_intermediate as u32,
                v.hidden_dim as u32,
                group_size,
                v.up_w_off_4bit() as u32,
                v.up_s_off_4bit() as u32,
                v.up_b_off_4bit() as u32,
                k_u32,
                0,
            );
            enc.end_encoding();
        }

        // Encoder 2: SwiGLU over all K experts
        {
            let enc = cmdbuf.new_compute_command_encoder();
            encode_swiglu_into_buf(
                enc,
                swiglu,
                gate_flat,
                up_flat,
                act_flat,
                k_u32 * v.moe_intermediate as u32,
            );
            enc.end_encoding();
        }

        // Encoder 3: down
        {
            let enc = cmdbuf.new_compute_command_encoder();
            encode_matvec_experts(
                enc,
                v3e,
                expert_bindings,
                act_flat,
                out_flat,
                v.hidden_dim as u32,
                v.moe_intermediate as u32,
                group_size,
                v.down_w_off_4bit() as u32,
                v.down_s_off_4bit() as u32,
                v.down_b_off_4bit() as u32,
                k_u32,
                v.moe_intermediate as u32,
            );
            enc.end_encoding();
        }
    } else {
        // Fallback: per-expert loop (hidden_dim > 4096 — i.e. Cogito-V2,
        // whose 7168 hidden won't fit the batched kernel's x_shared, or
        // k > MAX_K). The batched matvec now chunks experts in groups of
        // 8, so a17b's k=10 takes the batched path above, not this one.
        for slot in 0..k {
            let (wb, base_off) = expert_bindings[slot];
            let gate_buf = buffer_pool.handle(bufs.gate_id(slot));
            let up_buf = buffer_pool.handle(bufs.up_id(slot));
            let act_buf = buffer_pool.handle(bufs.act_id(slot));
            let out_buf = buffer_pool.handle(bufs.out_id(slot));
            {
                let enc = cmdbuf.new_compute_command_encoder();
                encode_matvec_into(
                    enc,
                    matvec,
                    wb,
                    base_off,
                    v.gate_w_off_4bit(),
                    v.gate_s_off_4bit(),
                    v.gate_b_off_4bit(),
                    input,
                    gate_buf,
                    v.moe_intermediate as u32,
                    v.hidden_dim as u32,
                );
                encode_matvec_into(
                    enc,
                    matvec,
                    wb,
                    base_off,
                    v.up_w_off_4bit(),
                    v.up_s_off_4bit(),
                    v.up_b_off_4bit(),
                    input,
                    up_buf,
                    v.moe_intermediate as u32,
                    v.hidden_dim as u32,
                );
                enc.end_encoding();
            }
            {
                let enc = cmdbuf.new_compute_command_encoder();
                encode_swiglu_into_buf(
                    enc,
                    swiglu,
                    gate_buf,
                    up_buf,
                    act_buf,
                    v.moe_intermediate as u32,
                );
                encode_matvec_into(
                    enc,
                    matvec,
                    wb,
                    base_off,
                    v.down_w_off_4bit(),
                    v.down_s_off_4bit(),
                    v.down_b_off_4bit(),
                    act_buf,
                    out_buf,
                    v.hidden_dim as u32,
                    v.moe_intermediate as u32,
                );
                enc.end_encoding();
            }
        }
    }

    // Combine
    {
        if use_batched {
            let moe_hidden_buf = buffer_pool.handle(bufs.moe_hidden_id());
            let combine_out: &BufferRef = match chain.as_ref() {
                Some(c) => c.combine_out,
                None => moe_hidden_buf,
            };
            let combine_params_buf = buffer_pool.handle(bufs.combine_params_id());
            let enc = cmdbuf.new_compute_command_encoder();
            encode_combine_flat(
                enc,
                combine_flat,
                h_mid,
                shared_out,
                combine_out,
                buffer_pool.handle(bufs.out_flat_id()),
                combine_params_buf,
                v.hidden_dim as u32,
                k as u32,
            );
            enc.end_encoding();
        } else {
            // Fallback: per-slot combine (same as before but need the
            // old per-slot combine kernel — keep it available)
            // For now, this path is only hit for Cogito-V2 (hidden>4096)
            // which is blocked on hardware anyway. Panic to surface if
            // we ever hit it unexpectedly.
            panic!(
                "[moe] per-expert fallback combine not wired for mmap \
                 (hidden_dim={}, k={}). This path is only for Cogito-V2.",
                v.hidden_dim, k,
            );
        }

        // Slice 5d-8 chain appendix: rms_norm_sum_sq +
        // rms_norm_apply_bf16 reading combine_out, writing chain_normed
        // via chain_sum_sq scratch. Bound to the next layer's
        // input_layernorm.weight in wf_buf at chain.next_norm_off. Same
        // kernel pair the CMD1 input-norm prelude uses (slice 9e
        // bit-exact per-PSO); equivalent to chaining Enc C2/C3 onto
        // the K-expert cmdbuf in C.
        if let Some(c) = chain {
            encode_rms_norm_bf16_into(
                cmdbuf,
                c.pipes,
                c.combine_out,
                c.wf_buf,
                c.next_norm_off,
                c.chain_sum_sq,
                c.chain_normed,
                v.hidden_dim as u32,
                c.eps,
            );
        }
    }
}

/// Inner-loop matvec encoder — same shape as [`encode_matvec`] but
/// takes a pre-existing encoder so the caller can fold multiple
/// dispatches into one encoder (matches the C path's
/// `gpu_encode_experts_batched` 2-encoder-per-expert layout).
///
/// Picks `dequant_matvec_4bit_v3` (cached input in 4096-float
/// threadgroup memory; bounded by Apple's 32 KB tg limit) when
/// `in_dim ≤ 4096`, else `dequant_matvec_4bit_fast` (no input cache).
/// Mirrors [`encode_matvec`]'s pipeline-selection so the experts path
/// works for variants whose `hidden_dim` exceeds 4096 (Cogito-V2 is
/// 7168 — `dequant_matvec_4bit_v3` would silently truncate the input
/// to 4096 and produce ~3% drift on the gate/up matvecs).
fn encode_matvec_into(
    enc: &metal::ComputeCommandEncoderRef,
    pipes: &MatvecPipelines,
    data: &Buffer,
    base_off: u64,
    w_off: usize,
    s_off: usize,
    b_off: usize,
    input: &BufferRef,
    output: &Buffer,
    out_dim: u32,
    in_dim: u32,
) {
    let group_size = GROUP_SIZE as u32;
    let use_v3 = in_dim <= 4096;
    let pipeline = if use_v3 {
        &pipes.v3_4bit
    } else {
        &pipes.fast_4bit
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(data), (base_off as usize + w_off) as NSUInteger);
    enc.set_buffer(1, Some(data), (base_off as usize + s_off) as NSUInteger);
    enc.set_buffer(2, Some(data), (base_off as usize + b_off) as NSUInteger);
    enc.set_buffer(3, Some(input), 0);
    enc.set_buffer(4, Some(output), 0);
    enc.set_bytes(5, 4, (&out_dim as *const u32).cast());
    enc.set_bytes(6, 4, (&in_dim as *const u32).cast());
    enc.set_bytes(7, 4, (&group_size as *const u32).cast());
    if use_v3 {
        let num_tgs = (out_dim + 7) / 8;
        enc.dispatch_thread_groups(
            MTLSize::new(num_tgs as NSUInteger, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    } else {
        enc.dispatch_thread_groups(
            MTLSize::new(out_dim as NSUInteger, 1, 1),
            MTLSize::new(64, 1, 1),
        );
    }
}

fn encode_swiglu_into_buf(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    gate: &Buffer,
    up: &Buffer,
    act: &Buffer,
    dim: u32,
) {
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
}

/// Encode a multi-expert matvec into the given encoder.
///
/// All `k` experts' projection (gate, up, or down) in a single dispatch.
/// Each expert reads from its own blob buffer; the projection within the
/// blob is selected by `(w_byte_off, s_byte_off, b_byte_off)`.
///
/// `input_stride`:
///   - 0: all experts share the same input (gate/up — broadcast)
///   - in_dim: expert k reads `input[k * in_dim..]` (down — per-expert)
fn encode_matvec_experts(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    expert_bindings: &[(&Buffer, u64)],
    input: &BufferRef,
    output: &Buffer,
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    w_byte_off: u32,
    s_byte_off: u32,
    b_byte_off: u32,
    k: u32,
    input_stride: u32,
) {
    enc.set_compute_pipeline_state(pipeline);
    // Bindings/params that don't vary across chunks.
    enc.set_buffer(8, Some(input), 0);
    enc.set_buffer(9, Some(output), 0);
    enc.set_bytes(10, 4, (&out_dim as *const u32).cast());
    enc.set_bytes(11, 4, (&in_dim as *const u32).cast());
    enc.set_bytes(12, 4, (&group_size as *const u32).cast());
    enc.set_bytes(13, 4, (&w_byte_off as *const u32).cast());
    enc.set_bytes(14, 4, (&s_byte_off as *const u32).cast());
    enc.set_bytes(15, 4, (&b_byte_off as *const u32).cast());
    let num_row_tiles = (out_dim + 7) / 8;
    enc.set_bytes(16, 4, (&num_row_tiles as *const u32).cast());
    enc.set_bytes(18, 4, (&input_stride as *const u32).cast());

    // The kernel binds at most 8 expert blobs (blob0..blob7), so cover
    // any k (a3b=8, a17b=10, up to MAX_K) by dispatching in chunks of
    // <=8 experts. Each chunk binds its 8 blobs, passes `expert_base`
    // (its global offset) and `k_active` = the chunk's expert count.
    // Chunks write disjoint `out[(base+e)*out_dim..]` regions, so they
    // are hazard-free within this one encoder.
    const CHUNK: u32 = 8;
    let mut base = 0u32;
    while base < k {
        let chunk_k = (k - base).min(CHUNK);
        for i in 0..CHUNK as usize {
            let global = base as usize + i;
            let (buf, off) = if global < expert_bindings.len() {
                expert_bindings[global]
            } else {
                expert_bindings[0]
            };
            enc.set_buffer(i as NSUInteger, Some(buf), off as NSUInteger);
        }
        enc.set_bytes(17, 4, (&chunk_k as *const u32).cast());
        enc.set_bytes(19, 4, (&base as *const u32).cast());
        let total_tgs = num_row_tiles * chunk_k;
        enc.dispatch_thread_groups(
            MTLSize::new(total_tgs as NSUInteger, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        base += CHUNK;
    }
}

/// Encode the flat-buffer variant of moe_combine_residual.
fn encode_combine_flat(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    h_mid: &BufferRef,
    shared_out: &BufferRef,
    hidden_out: &BufferRef,
    expert_out_flat: &Buffer,
    params: &Buffer,
    dim: u32,
    k: u32,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(h_mid), 0);
    enc.set_buffer(1, Some(shared_out), 0);
    enc.set_buffer(2, Some(hidden_out), 0);
    enc.set_buffer(3, Some(expert_out_flat), 0);
    enc.set_buffer(4, Some(params), 0);
    enc.set_bytes(5, 4, (&dim as *const u32).cast());
    enc.set_bytes(6, 4, (&k as *const u32).cast());
    let tgs = (dim + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Encode a batched MoE permute-and-fuse forward over `n_tokens` tokens.
///
/// Replaces the per-token K-expert dispatch ([`emit_batched_experts`])
/// for prefill: instead of reading each expert blob K times per token,
/// bucket (token, slot, weight) tuples by expert id and process each
/// bucket once. Arithmetic matches the tokenwise path (same
/// gate/up/swiglu/down math, same routing weights) — only the dispatch
/// order differs, so the diff oracle expects cosine ≥ 0.9999 against
/// the tokenwise reference (FP reorder envelope only).
///
/// Two encode paths, selected by `gather`:
/// - `gather = true` — one MLX `affine_gather_qmm_rhs` GEMM per
///   gate/up/down over *all* assignments at once (the kernel walks the
///   per-row `expert_indices` and GEMMs each contiguous same-expert
///   run), a flat swiglu, then the per-bucket scatter.
/// - `gather = false` — the per-bucket fallback: a hand-rolled matvec
///   per non-empty bucket. Kept as the shape the [`CpuBackend`] oracle
///   mirrors and the `MOEFLUX_MOE_GATHER=0` escape hatch.
///
/// Expert weights live in one `expert_base` buffer at uniform
/// `expert_stride` byte spacing; bucket `bi` uses the expert block at
/// `expert_base + expert_slots[bi] * expert_stride`, and the gather
/// kernel reaches the same block via `expert_indices[row]`.
///
/// Caller responsibilities:
/// - `expert_base` holds every needed expert's 4-bit-quantized weight
///   block; `expert_slots[bi]` is the block index for
///   `buckets.expert_ids[bi]` (parallel arrays). `expert_indices` is
///   `expert_slots` expanded per assignment row (`offsets`-driven).
/// - `bucket_input` is a packed `[total_assignments, hidden_dim]` f32
///   buffer of post-attn-norm hidden states gathered per-bucket. The
///   bucket-`bi` rows live at offset
///   `offsets[bi] * hidden_dim * sizeof::<f32>()`.
/// - `bucket_gate` / `bucket_up` / `bucket_act` are scratch buffers each
///   sized for `total_assignments * MOE_INTERMEDIATE` f32s.
/// - `bucket_out` is sized for `total_assignments * hidden_dim` f32s.
/// - `bucket_token_idx` is an i32 buffer holding `buckets.token_idx`.
/// - `bucket_weights` is an f32 buffer holding `buckets.weights`.
/// - `out_sum` is the `[n_tokens, hidden_dim]` MoE accumulator,
///   pre-initialized by the caller (typically zeroed, or seeded with
///   residual / shared-expert contribution).
///
/// Empty buckets are skipped (they don't appear in
/// [`ExpertBuckets::expert_ids`] by construction; the per-bucket loop
/// also guards `b_size == 0` defensively).
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_batched_permute_fuse(
    cmdbuf: &CommandBufferRef,
    matvec: &MatvecPipelines,
    kernels: &Kernels,
    swiglu: &ComputePipelineState,
    bucket_accumulate: &ComputePipelineState,
    expert_base: &Buffer,
    expert_stride: u64,
    expert_indices: &Buffer,
    expert_slots: &[u32],
    bucket_input: &Buffer,
    bucket_gate: &Buffer,
    bucket_up: &Buffer,
    bucket_act: &Buffer,
    bucket_out: &Buffer,
    bucket_token_idx: &Buffer,
    bucket_weights: &Buffer,
    out_sum: &Buffer,
    buckets: &ExpertBuckets,
    v: Variant,
    gather: bool,
) {
    debug_assert_eq!(expert_slots.len(), buckets.expert_ids.len());
    if gather {
        encode_moe_gather(
            cmdbuf,
            kernels,
            swiglu,
            bucket_accumulate,
            expert_base,
            expert_stride,
            expert_indices,
            bucket_input,
            bucket_gate,
            bucket_up,
            bucket_act,
            bucket_out,
            bucket_token_idx,
            bucket_weights,
            out_sum,
            buckets,
            v,
        );
    } else {
        encode_moe_per_bucket(
            cmdbuf,
            matvec,
            swiglu,
            bucket_accumulate,
            expert_base,
            expert_stride,
            expert_slots,
            bucket_input,
            bucket_gate,
            bucket_up,
            bucket_act,
            bucket_out,
            bucket_token_idx,
            bucket_weights,
            out_sum,
            buckets,
            v,
        );
    }
}

/// Element-wise swiglu `silu(gate) * up → act`, flat-dispatched over
/// `dim` elements. All three buffers are read/written at byte offset
/// `off`. Element-wise so the flat dispatch matches the per-token
/// kernel's row-by-row arithmetic exactly.
fn encode_swiglu_at(
    cmdbuf: &CommandBufferRef,
    swiglu: &ComputePipelineState,
    gate: &Buffer,
    up: &Buffer,
    act: &Buffer,
    off: u64,
    dim: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(swiglu);
    enc.set_buffer(0, Some(gate), off as NSUInteger);
    enc.set_buffer(1, Some(up), off as NSUInteger);
    enc.set_buffer(2, Some(act), off as NSUInteger);
    enc.set_bytes(3, 4, (&dim as *const u32).cast());
    let num_tgs = (dim + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

/// `moe_bucket_accumulate`: scatter-add one bucket's weighted expert
/// outputs into `out_sum`. No atomics — `token_idx[b]` is unique
/// within a bucket, and cross-bucket sequencing comes from Metal's
/// encoder ordering within the cmdbuf (each call here is one encoder).
#[allow(clippy::too_many_arguments)]
fn encode_bucket_scatter(
    cmdbuf: &CommandBufferRef,
    bucket_accumulate: &ComputePipelineState,
    bucket_out: &Buffer,
    bucket_token_idx: &Buffer,
    bucket_weights: &Buffer,
    out_sum: &Buffer,
    start: u64,
    b_size: u32,
    hidden_dim: u32,
) {
    let f32_sz = std::mem::size_of::<f32>() as u64;
    let i32_sz = std::mem::size_of::<i32>() as u64;
    let out_off = start * hidden_dim as u64 * f32_sz;
    let idx_off = start * i32_sz;
    let w_off_b = start * f32_sz;

    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(bucket_accumulate);
    enc.set_buffer(0, Some(bucket_out), out_off as NSUInteger);
    enc.set_buffer(1, Some(bucket_token_idx), idx_off as NSUInteger);
    enc.set_buffer(2, Some(bucket_weights), w_off_b as NSUInteger);
    enc.set_buffer(3, Some(out_sum), 0);
    enc.set_bytes(4, 4, (&hidden_dim as *const u32).cast());
    enc.set_bytes(5, 4, (&b_size as *const u32).cast());
    let tgs_x = b_size as NSUInteger;
    let tgs_y = ((hidden_dim + 255) / 256) as NSUInteger;
    enc.dispatch_thread_groups(MTLSize::new(tgs_x, tgs_y, 1), MTLSize::new(1, 256, 1));
    enc.end_encoding();
}

/// Gather path: one MLX `affine_gather_qmm_rhs` GEMM per gate/up/down
/// over every assignment row at once, a flat swiglu, then the
/// per-bucket scatter. The gather kernel walks `expert_indices` (one
/// `u32` per row) and GEMMs each contiguous same-expert run — so
/// `bucket_input` must be bucket-permuted (it is, by construction).
#[allow(clippy::too_many_arguments)]
fn encode_moe_gather(
    cmdbuf: &CommandBufferRef,
    kernels: &Kernels,
    swiglu: &ComputePipelineState,
    bucket_accumulate: &ComputePipelineState,
    expert_base: &Buffer,
    expert_stride: u64,
    expert_indices: &Buffer,
    bucket_input: &Buffer,
    bucket_gate: &Buffer,
    bucket_up: &Buffer,
    bucket_act: &Buffer,
    bucket_out: &Buffer,
    bucket_token_idx: &Buffer,
    bucket_weights: &Buffer,
    out_sum: &Buffer,
    buckets: &ExpertBuckets,
    v: Variant,
) {
    let total = buckets.token_idx.len() as u32;
    if total == 0 {
        return;
    }
    let hidden_dim = v.hidden_dim as u32;
    let moe_inter = v.moe_intermediate as u32;
    // `stride_s` is in bf16 *elements*; the per-expert block stride is
    // `expert_stride` bytes, scales are bf16 → /2.
    let stride_s = expert_stride / 2;

    // One GEMM whose RHS weight is gathered per row. `packed/scales/
    // biases_offset` are expert 0's sub-tensor offsets within its
    // block; the kernel reaches expert e via `e * stride_w/stride_s`.
    let gather = |w_off: u64,
                  s_off: u64,
                  b_off: u64,
                  input: &Buffer,
                  output: &Buffer,
                  in_dim: u32,
                  out_dim: u32| {
        kernels.encode(
            cmdbuf,
            &GatherQmmCall {
                weights: QuantWeights {
                    buffer: expert_base,
                    packed_offset: w_off,
                    scales_offset: s_off,
                    biases_offset: b_off,
                },
                input,
                input_offset: 0,
                output,
                output_offset: 0,
                indices: expert_indices,
                indices_offset: 0,
                in_dim,
                out_dim,
                n_tokens: total,
                stride_w: expert_stride,
                stride_s,
            },
        );
    };

    // 1. gate, 2. up — bucket_input → bucket_gate / bucket_up.
    gather(
        v.gate_w_off_4bit() as u64,
        v.gate_s_off_4bit() as u64,
        v.gate_b_off_4bit() as u64,
        bucket_input,
        bucket_gate,
        hidden_dim,
        moe_inter,
    );
    gather(
        v.up_w_off_4bit() as u64,
        v.up_s_off_4bit() as u64,
        v.up_b_off_4bit() as u64,
        bucket_input,
        bucket_up,
        hidden_dim,
        moe_inter,
    );

    // 3. swiglu over the whole flat `[total, moe_inter]` region.
    encode_swiglu_at(
        cmdbuf,
        swiglu,
        bucket_gate,
        bucket_up,
        bucket_act,
        0,
        total * moe_inter,
    );

    // 4. down — bucket_act → bucket_out.
    gather(
        v.down_w_off_4bit() as u64,
        v.down_s_off_4bit() as u64,
        v.down_b_off_4bit() as u64,
        bucket_act,
        bucket_out,
        moe_inter,
        hidden_dim,
    );

    // 5. per-bucket scatter into out_sum.
    for bi in 0..buckets.expert_ids.len() {
        let start = buckets.offsets[bi] as u64;
        let b_size = (buckets.offsets[bi + 1] - buckets.offsets[bi]) as u32;
        if b_size == 0 {
            continue;
        }
        encode_bucket_scatter(
            cmdbuf,
            bucket_accumulate,
            bucket_out,
            bucket_token_idx,
            bucket_weights,
            out_sum,
            start,
            b_size,
            hidden_dim,
        );
    }
}

/// One-dispatch MoE matmul via `moeflux_mm_id` (Differentiator 1 from
/// `.claude/memory/llama_cpp_moe_differentiators.md`). Eliminates the
/// host-side bucket-permute and per-row expert indirection that
/// [`encode_moe_gather`] above carries.
///
/// Pipeline (per layer):
/// 1. `MoeIdMap0Call` — build `htpe[n_experts]` + `hids[n_experts,
///    n_tokens]` from the router's per-token `indices[n_tokens, k]`.
/// 2. `MoeGatherIdCall` for **gate** — `mlp_in → gate_mid`.
/// 3. `MoeGatherIdCall` for **up** — `mlp_in → up_mid`.
/// 4. SwiGLU — `silu(gate_mid) * up_mid → gate_mid` (in-place; gate
///    and "act" share the same buffer; safe because each thread reads
///    gate[i] + up[i] before writing gate[i]).
/// 5. `MoeGatherIdCall` for **down** — `gate_mid (now silu*up) →
///    down_mid`. Note `ne11 = k` here: input is per-(token, slot).
/// 6. `MoeCombineTopkCall` — `down_mid × weights → out_sum`.
///
/// `mlp_in` is the **post-RmsNorm** hidden state — the input the
/// gate/up matmuls expect. NOT the pre-norm residual (that's a
/// different buffer; see `Op::MoeGatherIdFuse::mlp_in` doc).
///
/// All scratch (htpe, hids, gate_mid, up_mid, down_mid) is provided
/// by the producer; sized for the chunk width at construction.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_gather_id_fuse(
    cmdbuf: &CommandBufferRef,
    kernels: &Kernels,
    swiglu: &ComputePipelineState,
    expert_base: &Buffer,
    expert_stride: u64,
    indices: &Buffer,
    weights: &Buffer,
    mlp_in: &Buffer,
    out_sum: &Buffer,
    htpe: &Buffer,
    hids: &Buffer,
    gate_mid: &Buffer,
    up_mid: &Buffer,
    down_mid: &Buffer,
    n_tokens: u32,
    n_experts: u32,
    k: u32,
    v: Variant,
) {
    if n_tokens == 0 {
        return;
    }
    let hidden_dim = v.hidden_dim as u32;
    let moe_inter = v.moe_intermediate as u32;

    // --- 1. map0 ---
    kernels.encode(
        cmdbuf,
        &moeflux_metal::MoeIdMap0Call {
            indices,
            indices_offset: 0,
            htpe,
            htpe_offset: 0,
            hids,
            hids_offset: 0,
            n_experts,
            n_tokens,
            k,
        },
    );

    // Helper closure: encode one MoE matmul. `ne11`, `nb11`, `nb12`
    // differ between gate/up (input = [n_tokens, hidden]) and down
    // (input = [n_tokens, k, moe_inter]).
    let mm_id = |src1: &Buffer,
                 dst: &Buffer,
                 k_in: u32,
                 n_out: u32,
                 packed_off: u64,
                 scales_off: u64,
                 biases_off: u64,
                 ne11: u32,
                 nb11: u64,
                 nb12: u64| {
        kernels.encode(
            cmdbuf,
            &moeflux_metal::MoeGatherIdCall {
                src0: expert_base,
                src0_offset: 0,
                src1,
                src1_offset: 0,
                htpe,
                htpe_offset: 0,
                hids,
                hids_offset: 0,
                dst,
                dst_offset: 0,
                k_in,
                n_out,
                n_experts,
                n_tokens,
                k,
                ne11,
                nb02: expert_stride,
                nb01_w: (k_in / 2) as u64,
                nb01_s: (k_in / 32) as u64, // (K/64)*2 bytes
                packed_off,
                scales_off,
                biases_off,
                nb10: 4,
                nb11,
                nb12,
            },
        );
    };

    // --- 2. gate: mlp_in → gate_mid ---
    mm_id(
        mlp_in,
        gate_mid,
        hidden_dim,
        moe_inter,
        v.gate_w_off_4bit() as u64,
        v.gate_s_off_4bit() as u64,
        v.gate_b_off_4bit() as u64,
        /* ne11 */ 1,
        /* nb11 */ 0,
        /* nb12 */ (hidden_dim * 4) as u64,
    );
    // --- 3. up: mlp_in → up_mid ---
    mm_id(
        mlp_in,
        up_mid,
        hidden_dim,
        moe_inter,
        v.up_w_off_4bit() as u64,
        v.up_s_off_4bit() as u64,
        v.up_b_off_4bit() as u64,
        /* ne11 */ 1,
        /* nb11 */ 0,
        /* nb12 */ (hidden_dim * 4) as u64,
    );

    // --- 4. SwiGLU: silu(gate_mid) * up_mid → gate_mid (in-place) ---
    // The flat element count covers all (n_tokens × k × moe_inter)
    // slots; the SwiGLU kernel is element-wise so layout is moot.
    encode_swiglu_at(
        cmdbuf,
        swiglu,
        gate_mid,
        up_mid,
        gate_mid, // in-place
        0,
        n_tokens * k * moe_inter,
    );

    // --- 5. down: gate_mid (silu*up) → down_mid ---
    // Input is [n_tokens, k, moe_inter] → ne11 = k (per-slot stride).
    mm_id(
        gate_mid,
        down_mid,
        moe_inter,
        hidden_dim,
        v.down_w_off_4bit() as u64,
        v.down_s_off_4bit() as u64,
        v.down_b_off_4bit() as u64,
        /* ne11 */ k,
        /* nb11 */ (moe_inter * 4) as u64,
        /* nb12 */ (k * moe_inter * 4) as u64,
    );

    // --- 6. combine_topk: down_mid × weights → out_sum ---
    kernels.encode(
        cmdbuf,
        &moeflux_metal::MoeCombineTopkCall {
            mid: down_mid,
            mid_offset: 0,
            weights,
            weights_offset: 0,
            out: out_sum,
            out_offset: 0,
            n_tokens,
            hidden_dim,
            k,
        },
    );
}

/// Per-bucket fallback: a hand-rolled matvec sequence per non-empty
/// bucket. Bucket `bi` uses the expert block at `expert_base +
/// expert_slots[bi] * expert_stride`. Kept as the shape the
/// [`CpuBackend`] oracle mirrors and the `MOEFLUX_MOE_GATHER=0`
/// escape hatch.
#[allow(clippy::too_many_arguments)]
fn encode_moe_per_bucket(
    cmdbuf: &CommandBufferRef,
    matvec: &MatvecPipelines,
    swiglu: &ComputePipelineState,
    bucket_accumulate: &ComputePipelineState,
    expert_base: &Buffer,
    expert_stride: u64,
    expert_slots: &[u32],
    bucket_input: &Buffer,
    bucket_gate: &Buffer,
    bucket_up: &Buffer,
    bucket_act: &Buffer,
    bucket_out: &Buffer,
    bucket_token_idx: &Buffer,
    bucket_weights: &Buffer,
    out_sum: &Buffer,
    buckets: &ExpertBuckets,
    v: Variant,
) {
    let hidden_dim = v.hidden_dim as u32;
    let moe_inter = v.moe_intermediate as u32;
    let f32_sz = std::mem::size_of::<f32>() as u64;

    for (bi, &slot) in expert_slots.iter().enumerate() {
        let start = buckets.offsets[bi] as u64;
        let end = buckets.offsets[bi + 1] as u64;
        let b_size = (end - start) as u32;
        if b_size == 0 {
            continue;
        }

        let expert_off = slot as u64 * expert_stride;
        let in_off = start * v.hidden_dim as u64 * f32_sz;
        let mid_off = start * v.moe_intermediate as u64 * f32_sz;
        let out_off = start * v.hidden_dim as u64 * f32_sz;

        // 1. gate matvec: bucket_input → bucket_gate
        encode_matvec_n_tokens(
            cmdbuf,
            matvec,
            expert_base,
            expert_off + v.gate_w_off_4bit() as u64,
            expert_off + v.gate_s_off_4bit() as u64,
            expert_off + v.gate_b_off_4bit() as u64,
            bucket_input,
            in_off,
            bucket_gate,
            mid_off,
            v.hidden_dim as u32,
            moe_inter,
            b_size,
            4,
        );

        // 2. up matvec: bucket_input → bucket_up
        encode_matvec_n_tokens(
            cmdbuf,
            matvec,
            expert_base,
            expert_off + v.up_w_off_4bit() as u64,
            expert_off + v.up_s_off_4bit() as u64,
            expert_off + v.up_b_off_4bit() as u64,
            bucket_input,
            in_off,
            bucket_up,
            mid_off,
            v.hidden_dim as u32,
            moe_inter,
            b_size,
            4,
        );

        // 3. swiglu: silu(gate) * up → act
        encode_swiglu_at(
            cmdbuf,
            swiglu,
            bucket_gate,
            bucket_up,
            bucket_act,
            mid_off,
            b_size * moe_inter,
        );

        // 4. down matvec: bucket_act → bucket_out
        encode_matvec_n_tokens(
            cmdbuf,
            matvec,
            expert_base,
            expert_off + v.down_w_off_4bit() as u64,
            expert_off + v.down_s_off_4bit() as u64,
            expert_off + v.down_b_off_4bit() as u64,
            bucket_act,
            mid_off,
            bucket_out,
            out_off,
            moe_inter,
            v.hidden_dim as u32,
            b_size,
            4,
        );

        // 5. scatter-add weighted into out_sum.
        encode_bucket_scatter(
            cmdbuf,
            bucket_accumulate,
            bucket_out,
            bucket_token_idx,
            bucket_weights,
            out_sum,
            start,
            b_size,
            hidden_dim,
        );
    }
}

/// Encode the batched MoE combine kernel into `cmdbuf`. Replaces the
/// CPU loop that computed
/// `hidden_out[t,i] = h_mid[t,i] + moe_sum[t,i] + sigmoid(shared_gate[t]) * shared_out[t,i]`
/// after each layer's MoE permute-fuse. With this on GPU the
/// orchestrator can keep `hidden_out` on the GPU as the next layer's
/// `hidden_in`, eliminating the inter-layer host bounce.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_combine_residual_n_tokens(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    h_mid: &Buffer,
    moe_sum: &Buffer,
    shared_out: &Buffer,
    shared_gate: &Buffer,
    hidden_out: &Buffer,
    n_tokens: u32,
    dim: u32,
) {
    let total = n_tokens * dim;
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(h_mid), 0);
    enc.set_buffer(1, Some(moe_sum), 0);
    enc.set_buffer(2, Some(shared_out), 0);
    enc.set_buffer(3, Some(shared_gate), 0);
    enc.set_buffer(4, Some(hidden_out), 0);
    enc.set_bytes(5, 4, (&n_tokens as *const u32).cast());
    enc.set_bytes(6, 4, (&dim as *const u32).cast());
    let num_tgs = (total + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: build a synthetic expert + h_post, run the forward,
    /// verify the output is finite and not all zero. Doesn't compare
    /// against C — the C-vs-Rust diff lives in `tests/diff_oracle.rs`.
    #[test]
    #[ignore = "needs Metal device + access to shaders.metal source"]
    fn gpu_expert_forward_runs_and_produces_finite_output() {
        let mut metal = MetalContext::new().expect("MetalContext::new");
        let expert_data = synth::expert_data_seeded();
        let h_post = synth::h_post_seeded();
        let mut out = vec![0.0f32; VARIANT.hidden_dim];
        gpu_expert_forward(&mut metal, &expert_data, &h_post, &mut out)
            .expect("gpu_expert_forward");
        assert!(out.iter().all(|x| x.is_finite()), "output has NaN/Inf");
        assert!(
            out.iter().any(|&x| x.abs() > 0.0),
            "output is all zero — kernel didn't write?"
        );
    }
}

/// Synthetic-input helpers for the diff harness in
/// `tests/diff_oracle.rs`. Both backends consume identical bytes so
/// any output divergence must come from the kernel-encoding paths
/// themselves.
pub mod synth {
    use super::*;

    /// PRNG-seeded synthetic expert data — BF16 scales = 0x3C00
    /// (≈0.0078125), biases = 0. Identical bytes regardless of
    /// platform / build.
    pub fn expert_data_seeded() -> Vec<u8> {
        let v: Variant = VARIANT;
        let mut data = vec![0u8; v.expert_size_4bit()];
        for block in 0..3 {
            let block_off = block * v.expert_block_bytes_4bit();
            let w_end = block_off + v.expert_weight_bytes_4bit();
            let mut state: u64 = 0xCAFE_BEEF + block as u64;
            for byte in &mut data[block_off..w_end] {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *byte = (state >> 32) as u8;
            }
            let s_end = w_end + v.expert_scale_bytes();
            for chunk in data[w_end..s_end].chunks_exact_mut(2) {
                chunk[0] = 0x00;
                chunk[1] = 0x3C;
            }
        }
        data
    }

    /// Deterministic synthetic post-attn-norm hidden state.
    pub fn h_post_seeded() -> Vec<f32> {
        let v = VARIANT;
        (0..v.hidden_dim)
            .map(|i| (i as f32 - v.hidden_dim as f32 / 2.0) * 1e-3 / v.hidden_dim as f32)
            .collect()
    }

    /// `k * EXPERT_SIZE` bytes of synthetic expert blobs, slot-major.
    /// Each slot uses a different PRNG seed so the K experts produce
    /// distinct outputs through the kernels.
    pub fn k_expert_data_seeded(k: usize) -> Vec<u8> {
        let v: Variant = VARIANT;
        let per_expert = v.expert_size_4bit();
        let mut data = vec![0u8; k * per_expert];
        for slot in 0..k {
            let dst = &mut data[slot * per_expert..(slot + 1) * per_expert];
            for block in 0..3 {
                let block_off = block * v.expert_block_bytes_4bit();
                let w_end = block_off + v.expert_weight_bytes_4bit();
                let mut state: u64 = 0xCAFE_BEEF ^ ((slot as u64) << 32) ^ (block as u64);
                for byte in &mut dst[block_off..w_end] {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    *byte = (state >> 32) as u8;
                }
                let s_end = w_end + v.expert_scale_bytes();
                for chunk in dst[w_end..s_end].chunks_exact_mut(2) {
                    chunk[0] = 0x00;
                    chunk[1] = 0x3C;
                }
            }
        }
        data
    }

    /// Deterministic synthetic h_mid (residual). Slightly different
    /// shape from `h_post_seeded` so the combine pulls in distinct
    /// values rather than the same vector twice.
    pub fn h_mid_seeded() -> Vec<f32> {
        let v = VARIANT;
        (0..v.hidden_dim)
            .map(|i| (i as f32 * 0.0007 - 0.05).sin() * 0.001)
            .collect()
    }

    /// Deterministic synthetic shared expert output.
    pub fn shared_out_seeded() -> Vec<f32> {
        let v = VARIANT;
        (0..v.hidden_dim)
            .map(|i| (i as f32 * 0.0011 + 0.03).cos() * 0.001)
            .collect()
    }

    /// Sum-to-1 routing weights for K experts. Mirrors what the MoE
    /// router would emit after softmax + top-K + normalize.
    pub fn expert_weights_seeded(k: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.37 + 1.0).abs()).collect();
        let total: f32 = raw.iter().sum();
        raw.iter().map(|w| w / total).collect()
    }
}
