//! GPU full-attention kernels — slice 5d-7a.
//!
//! The four kernels that make up `gpu_attn_fuse` (`infer.m:5051..5163`):
//!
//! 1. `attn_scores_batched` — Q · K^T per head, scaled. One
//!    threadgroup per (position, head), 256 threads each, two-stage
//!    SIMD-then-shared reduction over `head_dim`.
//! 2. `attn_softmax_batched` — per-head softmax over `[0, seq_len)`.
//!    One threadgroup per head, 256 threads each, three passes
//!    (max → exp+sum → normalize) with two SIMD/shared reductions.
//! 3. `attn_values_batched` — `scores · V` per head. One thread per
//!    `(head, dim)` output element; per-thread loop over `seq_len`.
//! 4. `sigmoid_gate` — `out[i] *= sigmoid(gate[i])`. Per-thread
//!    elementwise; no reductions, no atomics — bit-exact per-PSO.
//!
//! Same per-PSO determinism finding as slice 9 (rms_norm, swiglu,
//! expert FFN): kernels with only SIMD reductions hit sub-ULP per-PSO
//! agreement; tests reserve `cosine ≥ 0.9999` + `rel_max_abs_diff ≤
//! 1e-3` as the defensive floor.
//!
//! ## Per-call allocation (oracle entry points)
//!
//! Same shape as [`super::gpu_norm::gpu_rms_norm_fused`]: scratch
//! buffers are allocated fresh per call. Production reuses the
//! persistent `buf_kv_k[fa_idx]` / `buf_kv_v[fa_idx]` / `buf_attn_*`
//! buffers from the model context (`infer.m:1251..1268`); slice 5d-7b
//! introduces the riir-side equivalents on
//! [`super::linear_attn_forward::LayerForwardBuffers`].
//!
//! ## Stride convention
//!
//! Oracle output uses `seq_stride = seq_len` (tight packing).
//! Production callers use `seq_stride = GPU_KV_SEQ` for the persistent
//! score buffer; the kernels write the same per-row values regardless
//! of stride choice (each row's first `seq_len` entries are populated;
//! the rest are unused), so test vs. production agree per-element
//! after slicing.

use metal::{
    BufferRef, CommandBufferRef, ComputePipelineState, MTLSize, NSUInteger,
};

use super::metal::{MetalBackend, MetalError, MtlBuffer};

/// Errors from the GPU attention kernels.
#[derive(Debug, thiserror::Error)]
pub enum GpuAttnError {
    #[error("buffer too short: {what} expected {expected} floats, got {actual}")]
    BadLen {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("non-positive shape: {what} = {value}")]
    BadShape {
        what: &'static str,
        value: i64,
    },
    #[error("num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})")]
    BadGqa { num_heads: u32, num_kv_heads: u32 },
    #[error("Metal backend: {0}")]
    Metal(#[from] MetalError),
}

fn check_pos(what: &'static str, value: i64) -> Result<(), GpuAttnError> {
    if value <= 0 {
        return Err(GpuAttnError::BadShape { what, value });
    }
    Ok(())
}

fn check_len(
    what: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), GpuAttnError> {
    if actual != expected {
        return Err(GpuAttnError::BadLen {
            what,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Pre-fetched pipelines for the 4 GPU attention kernels. Used by the
/// per-layer fast path (slice 5d-7b); fetching once per layer-forward
/// avoids the lazy-compile in the hot inner dispatch.
pub struct GpuAttnPipelines {
    pub scores: ComputePipelineState,
    pub softmax: ComputePipelineState,
    pub values: ComputePipelineState,
    pub gate: ComputePipelineState,
}

impl GpuAttnPipelines {
    pub fn fetch(metal: &mut MetalBackend) -> Result<Self, MetalError> {
        Ok(Self {
            scores: metal.pipeline("attn_scores_batched")?.clone(),
            softmax: metal.pipeline("attn_softmax_batched")?.clone(),
            values: metal.pipeline("attn_values_batched")?.clone(),
            gate: metal.pipeline("sigmoid_gate")?.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Encoders (production fast path) — slice 5d-7b
// ---------------------------------------------------------------------------

/// Encode `attn_scores_batched` into `cmdbuf`.
///
/// Computes `scores[h, p] = (Q[h] · K[p, kv_h]) * scale` for `h ∈
/// [0, num_heads)`, `p ∈ [0, seq_len)`. GQA: `kv_h = h /
/// heads_per_kv`. `scores` is laid out as `[num_heads, seq_stride]`
/// row-major; positions `≥ seq_len` are unwritten.
#[allow(clippy::too_many_arguments)]
pub fn encode_attn_scores_batched_into(
    cmdbuf: &CommandBufferRef,
    pipe: &ComputePipelineState,
    q: &BufferRef,
    k_cache: &BufferRef,
    scores: &BufferRef,
    num_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    seq_len: u32,
    seq_stride: u32,
    heads_per_kv: u32,
    scale: f32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(q), 0);
    enc.set_buffer(1, Some(k_cache), 0);
    enc.set_buffer(2, Some(scores), 0);
    enc.set_bytes(3, 4, (&head_dim as *const u32).cast());
    enc.set_bytes(4, 4, (&kv_dim as *const u32).cast());
    enc.set_bytes(5, 4, (&seq_len as *const u32).cast());
    enc.set_bytes(6, 4, (&seq_stride as *const u32).cast());
    enc.set_bytes(7, 4, (&scale as *const f32).cast());
    enc.set_bytes(8, 4, (&heads_per_kv as *const u32).cast());
    // `num_seq_tgs` (atIndex:9) — set to seq_len, matching production
    // (`infer.m:5115`).
    enc.set_bytes(9, 4, (&seq_len as *const u32).cast());
    let total_tgs = seq_len * num_heads;
    enc.dispatch_thread_groups(
        MTLSize::new(total_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

/// Encode `attn_softmax_batched` into `cmdbuf`. Mutates `scores` in
/// place: per-head softmax over `[0, seq_len)`.
pub fn encode_attn_softmax_batched_into(
    cmdbuf: &CommandBufferRef,
    pipe: &ComputePipelineState,
    scores: &BufferRef,
    num_heads: u32,
    seq_len: u32,
    seq_stride: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(scores), 0);
    enc.set_bytes(1, 4, (&seq_len as *const u32).cast());
    enc.set_bytes(2, 4, (&seq_stride as *const u32).cast());
    enc.dispatch_thread_groups(
        MTLSize::new(num_heads as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

/// Encode `attn_values_batched` into `cmdbuf`.
///
/// Computes `out[h * head_dim + d] = Σ_p scores[h, p] * V[p, kv_h,
/// d]`. One thread per `(head, dim)` output element; per-thread loop
/// over `seq_len`.
#[allow(clippy::too_many_arguments)]
pub fn encode_attn_values_batched_into(
    cmdbuf: &CommandBufferRef,
    pipe: &ComputePipelineState,
    scores: &BufferRef,
    v_cache: &BufferRef,
    out: &BufferRef,
    num_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    seq_len: u32,
    seq_stride: u32,
    heads_per_kv: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(scores), 0);
    enc.set_buffer(1, Some(v_cache), 0);
    enc.set_buffer(2, Some(out), 0);
    enc.set_bytes(3, 4, (&head_dim as *const u32).cast());
    enc.set_bytes(4, 4, (&kv_dim as *const u32).cast());
    enc.set_bytes(5, 4, (&seq_len as *const u32).cast());
    enc.set_bytes(6, 4, (&seq_stride as *const u32).cast());
    enc.set_bytes(7, 4, (&heads_per_kv as *const u32).cast());
    let total_threads = head_dim * num_heads;
    let tgs = (total_threads + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

/// Encode `sigmoid_gate` into `cmdbuf`. Mutates `x_inout` in place:
/// `x_inout[i] *= sigmoid(gate[i])`.
pub fn encode_sigmoid_gate_into(
    cmdbuf: &CommandBufferRef,
    pipe: &ComputePipelineState,
    x_inout: &BufferRef,
    gate: &BufferRef,
    dim: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(x_inout), 0);
    enc.set_buffer(1, Some(gate), 0);
    enc.set_bytes(2, 4, (&dim as *const u32).cast());
    let tgs = (dim + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

// ---------------------------------------------------------------------------
// Freestanding oracle entry points — slice 5d-7a
// ---------------------------------------------------------------------------
//
// Synchronous one-kernel-at-a-time wrappers used by the diff oracle.
// Allocate fresh scratch, encode, commit + wait, copy back.

/// `attn_scores_batched`: per-head Q · K^T scaled. Stride-tight
/// (`seq_stride = seq_len`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_attn_scores_batched(
    metal: &mut MetalBackend,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    q: &[f32],
    k_cache: &[f32],
    scale: f32,
    scores_out: &mut [f32],
) -> Result<(), GpuAttnError> {
    check_pos("num_heads", num_heads as i64)?;
    check_pos("num_kv_heads", num_kv_heads as i64)?;
    check_pos("head_dim", head_dim as i64)?;
    check_pos("seq_len", seq_len as i64)?;
    if num_heads % num_kv_heads != 0 {
        return Err(GpuAttnError::BadGqa {
            num_heads,
            num_kv_heads,
        });
    }
    let kv_dim = num_kv_heads * head_dim;
    let heads_per_kv = num_heads / num_kv_heads;
    check_len("q", (num_heads * head_dim) as usize, q.len())?;
    check_len("k_cache", (seq_len * kv_dim) as usize, k_cache.len())?;
    check_len("scores_out", (num_heads * seq_len) as usize, scores_out.len())?;

    let pipe = metal.pipeline("attn_scores_batched")?.clone();
    let device = metal.device();
    let buf_q = MtlBuffer::<f32>::with_data(device, q);
    let buf_k = MtlBuffer::<f32>::with_data(device, k_cache);
    let buf_scores = MtlBuffer::<f32>::with_len(device, scores_out.len());

    let cmdbuf = metal.queue().new_command_buffer();
    encode_attn_scores_batched_into(
        cmdbuf,
        &pipe,
        buf_q.raw(),
        buf_k.raw(),
        buf_scores.raw(),
        num_heads,
        head_dim,
        kv_dim,
        seq_len,
        /* seq_stride = */ seq_len,
        heads_per_kv,
        scale,
    );
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    scores_out.copy_from_slice(&buf_scores.to_vec());
    Ok(())
}

/// `attn_softmax_batched`: per-head softmax over `[0, seq_len)`,
/// in-place. Stride-tight.
pub fn gpu_attn_softmax_batched(
    metal: &mut MetalBackend,
    num_heads: u32,
    seq_len: u32,
    scores_inout: &mut [f32],
) -> Result<(), GpuAttnError> {
    check_pos("num_heads", num_heads as i64)?;
    check_pos("seq_len", seq_len as i64)?;
    check_len(
        "scores_inout",
        (num_heads * seq_len) as usize,
        scores_inout.len(),
    )?;

    let pipe = metal.pipeline("attn_softmax_batched")?.clone();
    let device = metal.device();
    let buf_scores = MtlBuffer::<f32>::with_data(device, scores_inout);

    let cmdbuf = metal.queue().new_command_buffer();
    encode_attn_softmax_batched_into(
        cmdbuf,
        &pipe,
        buf_scores.raw(),
        num_heads,
        seq_len,
        /* seq_stride = */ seq_len,
    );
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    scores_inout.copy_from_slice(&buf_scores.to_vec());
    Ok(())
}

/// `attn_values_batched`: per-head scores · V. Stride-tight.
#[allow(clippy::too_many_arguments)]
pub fn gpu_attn_values_batched(
    metal: &mut MetalBackend,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    scores: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
) -> Result<(), GpuAttnError> {
    check_pos("num_heads", num_heads as i64)?;
    check_pos("num_kv_heads", num_kv_heads as i64)?;
    check_pos("head_dim", head_dim as i64)?;
    check_pos("seq_len", seq_len as i64)?;
    if num_heads % num_kv_heads != 0 {
        return Err(GpuAttnError::BadGqa {
            num_heads,
            num_kv_heads,
        });
    }
    let kv_dim = num_kv_heads * head_dim;
    let heads_per_kv = num_heads / num_kv_heads;
    check_len("scores", (num_heads * seq_len) as usize, scores.len())?;
    check_len("v_cache", (seq_len * kv_dim) as usize, v_cache.len())?;
    check_len("out", (num_heads * head_dim) as usize, out.len())?;

    let pipe = metal.pipeline("attn_values_batched")?.clone();
    let device = metal.device();
    let buf_scores = MtlBuffer::<f32>::with_data(device, scores);
    let buf_v = MtlBuffer::<f32>::with_data(device, v_cache);
    let buf_out = MtlBuffer::<f32>::with_len(device, out.len());

    let cmdbuf = metal.queue().new_command_buffer();
    encode_attn_values_batched_into(
        cmdbuf,
        &pipe,
        buf_scores.raw(),
        buf_v.raw(),
        buf_out.raw(),
        num_heads,
        head_dim,
        kv_dim,
        seq_len,
        /* seq_stride = */ seq_len,
        heads_per_kv,
    );
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    out.copy_from_slice(&buf_out.to_vec());
    Ok(())
}

/// `sigmoid_gate`: `x_inout[i] *= sigmoid(gate[i])`, in place.
pub fn gpu_sigmoid_gate(
    metal: &mut MetalBackend,
    dim: u32,
    gate: &[f32],
    x_inout: &mut [f32],
) -> Result<(), GpuAttnError> {
    check_pos("dim", dim as i64)?;
    check_len("gate", dim as usize, gate.len())?;
    check_len("x_inout", dim as usize, x_inout.len())?;

    let pipe = metal.pipeline("sigmoid_gate")?.clone();
    let device = metal.device();
    let buf_x = MtlBuffer::<f32>::with_data(device, x_inout);
    let buf_g = MtlBuffer::<f32>::with_data(device, gate);

    let cmdbuf = metal.queue().new_command_buffer();
    encode_sigmoid_gate_into(cmdbuf, &pipe, buf_x.raw(), buf_g.raw(), dim);
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    x_inout.copy_from_slice(&buf_x.to_vec());
    Ok(())
}
