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

use metal::{
    Buffer, BufferRef, CommandBufferRef, ComputePipelineState, MTLSize,
    NSUInteger,
};

use super::gpu_norm::{encode_rms_norm_bf16_into, RmsNormBf16Pipelines};
use super::metal::{MetalBackend, MetalError, MtlBuffer};
use super::variants::{Variant, GROUP_SIZE, VARIANT};

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
    #[error(
        "actual_K out of range: must be 1..={max}, got {actual}"
    )]
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
    metal: &mut MetalBackend,
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
/// 5d-6a; ~56 MB for slice 5d-6b after `data_prefetch` is added.
///
/// The two K-element data sets have **fixed roles**, not a ping-pong:
/// - [`Self::data_synced`] is the on-demand sync-pread target. The
///   GPU dispatch reads from this set for slots whose actual expert
///   index missed the prefetch prediction.
/// - [`Self::data_prefetch`] is the speculative prefetch target. The
///   GPU dispatch reads from this set for slots that hit the
///   prediction. Filled async by the prefetch thread pool ahead of
///   the layer that consumes it.
pub struct MoeBuffers {
    /// Sync-pread (miss) target. See type-level docs.
    data_synced: [MtlBuffer<u8>; MAX_K],
    /// Async-prefetch (hit) target. See type-level docs.
    data_prefetch: [MtlBuffer<u8>; MAX_K],
    gate: [MtlBuffer<f32>; MAX_K],
    up: [MtlBuffer<f32>; MAX_K],
    act: [MtlBuffer<f32>; MAX_K],
    out: [MtlBuffer<f32>; MAX_K],
    input: MtlBuffer<f32>,
    h_mid: MtlBuffer<f32>,
    shared_out: MtlBuffer<f32>,
    moe_hidden: MtlBuffer<f32>,
    combine_params: MtlBuffer<f32>,
}

impl MoeBuffers {
    /// Allocate the full multi-expert + combine buffer set on
    /// `device`. Sizes come from the active [`Variant`] and architectural
    /// `MAX_K`. Buffers are uninitialized — every call to
    /// [`gpu_batched_experts_forward`] writes the slots it uses before
    /// dispatch.
    pub fn new(device: &metal::Device) -> Self {
        let v: Variant = VARIANT;
        // Build arrays without requiring Default on MtlBuffer.
        let data_synced = std::array::from_fn(|_| {
            MtlBuffer::<u8>::with_len(device, v.expert_size_4bit())
        });
        let data_prefetch = std::array::from_fn(|_| {
            MtlBuffer::<u8>::with_len(device, v.expert_size_4bit())
        });
        // Slice 5d-6: probe 2 MB DMA alignment on the data slots.
        // Apple's malloc-via-mmap path *often* lands large (>= 1 MB)
        // shared-storage allocations on 2 MB boundaries; the C path
        // claims a 3.6× DMA improvement over 16 KB alignment for the
        // pread destination (`metal_infer/infer.m:1196`). We rely on
        // that incidental behavior rather than a custom allocator;
        // warn once if it ever fails so we can revisit (fallback
        // would be `posix_memalign + new_buffer_with_bytes_no_copy`).
        const TWO_MIB: usize = 2 * 1024 * 1024;
        for (label, set) in
            [("synced", &data_synced[..]), ("prefetch", &data_prefetch[..])]
        {
            for (slot, buf) in set.iter().enumerate() {
                let addr = buf.raw().contents() as usize;
                if addr % TWO_MIB != 0 {
                    eprintln!(
                        "[moe] WARNING: data_{label} slot {slot} not 2 MB \
                         aligned (contents=0x{addr:x}, off=0x{off:x}); \
                         pread DMA may use scatter-gather. See slice 5d-6 \
                         plan.",
                        off = addr % TWO_MIB
                    );
                    break;
                }
            }
        }
        let gate =
            std::array::from_fn(|_| MtlBuffer::<f32>::with_len(device, v.moe_intermediate));
        let up = std::array::from_fn(|_| {
            MtlBuffer::<f32>::with_len(device, v.moe_intermediate)
        });
        let act = std::array::from_fn(|_| {
            MtlBuffer::<f32>::with_len(device, v.moe_intermediate)
        });
        let out =
            std::array::from_fn(|_| MtlBuffer::<f32>::with_len(device, v.hidden_dim));
        Self {
            data_synced,
            data_prefetch,
            gate,
            up,
            act,
            out,
            input: MtlBuffer::with_len(device, v.hidden_dim),
            h_mid: MtlBuffer::with_len(device, v.hidden_dim),
            shared_out: MtlBuffer::with_len(device, v.hidden_dim),
            moe_hidden: MtlBuffer::with_len(device, v.hidden_dim),
            combine_params: MtlBuffer::with_len(device, 18),
        }
    }

    /// The post-combine hidden buffer (`buf_moe_hidden` in C). The
    /// destination for `moe_combine_residual` and the readback target
    /// for the deferred-experts state machine
    /// ([`super::deferred::RsCtx::complete_deferred_experts`]).
    pub(crate) fn moe_hidden(&self) -> &MtlBuffer<f32> {
        &self.moe_hidden
    }

    /// One per-expert output slot — the readback source for the CPU-
    /// combine path. `slot` must be `< actual_k` (caller-checked
    /// against the dispatch's `actual_k`); slots `[actual_k, MAX_K)`
    /// hold stale data and must not be read.
    pub(crate) fn out(&self, slot: usize) -> &MtlBuffer<f32> {
        &self.out[slot]
    }

    /// All per-slot data_synced (sync-pread / miss target) buffers as
    /// disjoint `&mut [u8]` views, suitable for parallel pread. Slice
    /// 5d-6a entry point: K concurrent `read_expert` calls can each
    /// take one slot's `&mut [u8]` from this array without aliasing
    /// the others.
    ///
    /// SAFETY/CORRECTNESS: caller ensures no GPU dispatch reading
    /// from any slot is in flight. The deferred-state `complete_*` /
    /// `discard_*` calls at the top of each layer's forward establish
    /// this invariant.
    pub(crate) fn data_synced_slots_mut_array(
        &mut self,
    ) -> [&mut [u8]; MAX_K] {
        self.data_synced.each_mut().map(|b| b.as_mut_slice())
    }

    /// All per-slot data_prefetch (async-prefetch / hit target)
    /// buffers as disjoint `&mut [u8]` views, suitable for parallel
    /// async pread by the prefetch state machine. Slice 5d-6b entry
    /// point.
    ///
    /// SAFETY/CORRECTNESS: caller ensures no GPU dispatch reading
    /// from any prefetch slot is in flight, AND that the previous
    /// async prefetch into these buffers has been drained.
    /// [`super::prefetch::PrefetchState::wait_for`] /
    /// [`super::prefetch::PrefetchState::drain`] are the established
    /// synchronization points.
    pub(crate) fn data_prefetch_slots_mut_array(
        &mut self,
    ) -> [&mut [u8]; MAX_K] {
        self.data_prefetch.each_mut().map(|b| b.as_mut_slice())
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
    metal: &mut MetalBackend,
    bufs: &mut MoeBuffers,
    actual_k: i32,
    expert_data: &[u8],
    h_post: &[f32],
    h_mid: &[f32],
    shared_out: &[f32],
    expert_weights: &[f32],
    shared_gate_score: f32,
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
        actual_k,
        expert_data,
        h_post,
        h_mid,
        shared_out,
        expert_weights,
        shared_gate_score,
        /* gpu_combine = */ true,
    )?;
    cmdbuf.commit();
    cmdbuf.wait_until_completed();
    hidden_out.copy_from_slice(&bufs.moe_hidden.to_vec());
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
    metal: &mut MetalBackend,
    bufs: &mut MoeBuffers,
    actual_k: i32,
    expert_data: &[u8],
    h_post: &[f32],
    h_mid: &[f32],
    shared_out: &[f32],
    expert_weights: &[f32],
    shared_gate_score: f32,
    gpu_combine: bool,
) -> Result<metal::CommandBuffer, ExpertForwardError> {
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
    // encoder construction.
    let matvec = metal.pipeline("dequant_matvec_4bit_v3")?.clone();
    let swiglu = metal.pipeline("swiglu_fused")?.clone();
    let combine = if gpu_combine {
        Some(metal.pipeline("moe_combine_residual")?.clone())
    } else {
        None
    };

    // Stage K expert blobs, the 18-float combine params, and host
    // inputs into bufs' own input/h_mid/shared_out — production skips
    // the host-input staging via [`gpu_batched_experts_encode_pre_staged`].
    let expert_size = v.expert_size_4bit();
    for slot in 0..k {
        let src = &expert_data[slot * expert_size..(slot + 1) * expert_size];
        bufs.data_synced[slot].as_mut_slice().copy_from_slice(src);
    }
    bufs.input.as_mut_slice().copy_from_slice(h_post);
    bufs.h_mid.as_mut_slice().copy_from_slice(h_mid);
    bufs.shared_out.as_mut_slice().copy_from_slice(shared_out);
    {
        let params = bufs.combine_params.as_mut_slice();
        params.fill(0.0);
        params[..k].copy_from_slice(expert_weights);
        params[16] = shared_gate_score;
    }

    let cmdbuf = metal.queue().new_command_buffer();
    // Bind bufs' own input/h_mid/shared_out — kernels read whatever we
    // just staged. Field-disjoint borrows: bufs is borrowed shared via
    // the bufs.input.raw() etc. derivations; emit_batched_experts only
    // reads bufs (also shared). All consistent.
    //
    // Host-slice variant always stages into `data_synced`; emit reads
    // accordingly.
    let data_set_per_slot: [super::SlotSource; MAX_K] =
        [super::SlotSource::Synced; MAX_K];
    emit_batched_experts(
        cmdbuf,
        &matvec,
        &swiglu,
        combine.as_ref(),
        bufs,
        bufs.input.raw(),
        bufs.h_mid.raw(),
        bufs.shared_out.raw(),
        k,
        v,
        &data_set_per_slot,
        None,
    );
    Ok(cmdbuf.to_owned())
}

/// Pre-staged variant — slice 5d-5. Caller has already populated
/// `bufs.data[0..actual_k]` (typically via `pread` directly into
/// [`MoeBuffers::data_slot_mut`]). Skips the K × EXPERT_SIZE host
/// memcpy that [`gpu_batched_experts_encode_buf`] does — saves ~7 MB
/// of memcpy per layer on A3B (K=4 × ~1.77 MB), or ~280 MB / token
/// at 40 layers. Eliminates the host-resident `expert_data` Vec
/// entirely from the production hot path.
///
/// The slot reuse pattern is sound because every layer's K-expert
/// dispatch is waited at the top of the next layer's
/// [`super::deferred::complete_deferred_experts_into`] before that
/// layer preads new data into `bufs.data[slot]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_batched_experts_encode_pre_staged(
    metal: &mut MetalBackend,
    bufs: &mut MoeBuffers,
    actual_k: i32,
    input: &BufferRef,
    h_mid: &BufferRef,
    shared_out: &BufferRef,
    expert_weights: &[f32],
    shared_gate_score: f32,
    data_set_per_slot: &[super::SlotSource; MAX_K],
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

    // Pipelines first; no `&mut metal` borrow held across encoder
    // construction.
    let matvec = metal.pipeline("dequant_matvec_4bit_v3")?.clone();
    let swiglu = metal.pipeline("swiglu_fused")?.clone();
    let combine = metal.pipeline("moe_combine_residual")?.clone();

    // Only stage the 18-float combine params — bufs' data buffers are
    // the caller's responsibility (data_synced via 5d-6a's parallel
    // pread, data_prefetch via 5d-6b's async prefetch).
    {
        let params = bufs.combine_params.as_mut_slice();
        params.fill(0.0);
        params[..k].copy_from_slice(expert_weights);
        params[16] = shared_gate_score;
    }

    let cmdbuf = metal.queue().new_command_buffer();
    emit_batched_experts(
        cmdbuf,
        &matvec,
        &swiglu,
        Some(&combine),
        bufs,
        input,
        h_mid,
        shared_out,
        k,
        v,
        data_set_per_slot,
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
/// (optionally) `moe_combine_residual` into `cmdbuf`. Takes the input
/// buffers as explicit refs so the host-slice and buf-ref entry points
/// can share this body.
///
/// `data_set_per_slot[slot]` selects whether to bind
/// `bufs.data_synced[slot]` ([`super::SlotSource::Synced`]) or
/// `bufs.data_prefetch[slot]` ([`super::SlotSource::Prefetched`]) as
/// the expert-weights source. Synchronous callers pass
/// `[SlotSource::Synced; MAX_K]`; the 5d-6b prefetch hot path passes
/// a per-slot mix.
#[allow(clippy::too_many_arguments)]
fn emit_batched_experts(
    cmdbuf: &CommandBufferRef,
    matvec: &ComputePipelineState,
    swiglu: &ComputePipelineState,
    combine: Option<&ComputePipelineState>,
    bufs: &MoeBuffers,
    input: &BufferRef,
    h_mid: &BufferRef,
    shared_out: &BufferRef,
    k: usize,
    v: Variant,
    data_set_per_slot: &[super::SlotSource; MAX_K],
    chain: Option<ChainToNormed<'_>>,
) {
    // Per-expert: Encoder A (gate+up — both read the shared `input`),
    // Encoder B (SwiGLU then down). Two encoders per expert exposes
    // GPU parallelism across slots; within an encoder the dispatches
    // serialize by Metal's encoder-internal ordering.
    let pick = |slot: usize| -> &MtlBuffer<u8> {
        match data_set_per_slot[slot] {
            super::SlotSource::Synced => &bufs.data_synced[slot],
            super::SlotSource::Prefetched => &bufs.data_prefetch[slot],
        }
    };
    for slot in 0..k {
        let weights_buf = pick(slot);
        // Encoder A — gate + up
        {
            let enc = cmdbuf.new_compute_command_encoder();
            // gate
            encode_matvec_into(
                enc,
                matvec,
                weights_buf,
                v.gate_w_off_4bit(),
                v.gate_s_off_4bit(),
                v.gate_b_off_4bit(),
                input,
                bufs.gate[slot].raw(),
                v.moe_intermediate as u32,
                v.hidden_dim as u32,
            );
            // up — same encoder, serialized after gate
            encode_matvec_into(
                enc,
                matvec,
                weights_buf,
                v.up_w_off_4bit(),
                v.up_s_off_4bit(),
                v.up_b_off_4bit(),
                input,
                bufs.up[slot].raw(),
                v.moe_intermediate as u32,
                v.hidden_dim as u32,
            );
            enc.end_encoding();
        }

        // Encoder B — SwiGLU + down
        {
            let enc = cmdbuf.new_compute_command_encoder();
            encode_swiglu_into_buf(
                enc,
                swiglu,
                bufs.gate[slot].raw(),
                bufs.up[slot].raw(),
                bufs.act[slot].raw(),
                v.moe_intermediate as u32,
            );
            encode_matvec_into(
                enc,
                matvec,
                weights_buf,
                v.down_w_off_4bit(),
                v.down_s_off_4bit(),
                v.down_b_off_4bit(),
                bufs.act[slot].raw(),
                bufs.out[slot].raw(),
                v.hidden_dim as u32,
                v.moe_intermediate as u32,
            );
            enc.end_encoding();
        }
    }

    // moe_combine_residual: 16 expert-output bindings regardless of K
    // (kernel branches on K, unused slots have weight=0). Bind all
    // MAX_K out buffers, weights via the params buffer. Skipped when
    // `gpu_combine == false` — caller will run the equivalent on the
    // CPU side after wait.
    //
    // Slice 5d-8: when `chain.is_some()`, the combine writes directly
    // into the next layer's input buffer (`chain.combine_out`) instead
    // of `bufs.moe_hidden`, and two more dispatches are appended to
    // produce the next layer's normalized input (`chain.chain_normed`).
    // Mirrors C's gpu_combine fast path (`infer.m:5677..5750`).
    if let Some(combine) = combine {
        let combine_out: &BufferRef = match chain.as_ref() {
            Some(c) => c.combine_out,
            None => bufs.moe_hidden.raw(),
        };
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(combine);
        enc.set_buffer(0, Some(h_mid), 0);
        enc.set_buffer(1, Some(shared_out), 0);
        enc.set_buffer(2, Some(combine_out), 0);
        for slot in 0..MAX_K {
            enc.set_buffer(
                3 + slot as NSUInteger,
                Some(bufs.out[slot].raw()),
                0,
            );
        }
        enc.set_buffer(19, Some(bufs.combine_params.raw()), 0);
        let dim = v.hidden_dim as u32;
        let k_val = k as u32;
        enc.set_bytes(20, 4, (&dim as *const u32).cast());
        enc.set_bytes(21, 4, (&k_val as *const u32).cast());
        let tgs = (dim + 255) / 256;
        enc.dispatch_thread_groups(
            MTLSize::new(tgs as NSUInteger, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();

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
fn encode_matvec_into(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    data: &MtlBuffer<u8>,
    w_off: usize,
    s_off: usize,
    b_off: usize,
    input: &BufferRef,
    output: &BufferRef,
    out_dim: u32,
    in_dim: u32,
) {
    let group_size = GROUP_SIZE as u32;
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(data.raw()), w_off as NSUInteger);
    enc.set_buffer(1, Some(data.raw()), s_off as NSUInteger);
    enc.set_buffer(2, Some(data.raw()), b_off as NSUInteger);
    enc.set_buffer(3, Some(input), 0);
    enc.set_buffer(4, Some(output), 0);
    enc.set_bytes(5, 4, (&out_dim as *const u32).cast());
    enc.set_bytes(6, 4, (&in_dim as *const u32).cast());
    enc.set_bytes(7, 4, (&group_size as *const u32).cast());
    let num_tgs = (out_dim + 7) / 8;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

fn encode_swiglu_into_buf(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    gate: &BufferRef,
    up: &BufferRef,
    act: &BufferRef,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: build a synthetic expert + h_post, run the forward,
    /// verify the output is finite and not all zero. Doesn't compare
    /// against C — the C-vs-Rust diff lives in `tests/diff_oracle.rs`.
    #[test]
    #[ignore = "needs Metal device + access to shaders.metal source"]
    fn gpu_expert_forward_runs_and_produces_finite_output() {
        let mut metal = MetalBackend::new().expect("MetalBackend::new");
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
            .map(|i| {
                (i as f32 - v.hidden_dim as f32 / 2.0) * 1e-3
                    / v.hidden_dim as f32
            })
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
                let mut state: u64 = 0xCAFE_BEEF
                    ^ ((slot as u64) << 32)
                    ^ (block as u64);
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
        let raw: Vec<f32> = (0..k)
            .map(|i| ((i as f32) * 0.37 + 1.0).abs())
            .collect();
        let total: f32 = raw.iter().sum();
        raw.iter().map(|w| w / total).collect()
    }
}
