//! GPU RMSNorm with bf16 weights — slice 9e.
//!
//! Mirrors the production CMD3 fast-path (`infer.m:5712..5744`),
//! standalone and synchronous so the diff oracle can read the result
//! back at a known boundary.
//!
//! Two-kernel chain in one MTLCommandBuffer:
//!
//! 1. `rms_norm_sum_sq` — single threadgroup of 256 threads computes
//!    `Σ x[i]²` via `simd_sum` + threadgroup-shared second-stage
//!    reduction. **First kernel under diff that uses
//!    threadgroup-shared memory across SIMD groups.** If Metal
//!    nondeterminism is going to engage anywhere in the suite, this
//!    is the most plausible spot.
//! 2. `rms_norm_apply_bf16` — per-element; each thread reads
//!    `sum_sq[0]`, computes `rsqrt(sum_sq/dim + eps)`, multiplies by
//!    `bf16_to_f32(weight[i])`.
//!
//! ## Per-call allocation
//!
//! Same shape as slice 9a's single-expert path: scratch buffers are
//! allocated fresh per call. The production C path reuses
//! `buf_moe_hidden / buf_cmd3_sum_sq / buf_input` from the model
//! context; we don't carry that layout into the diff harness because
//! it would couple this slice to the deferred-state plumbing
//! (slice 9d). Per-call alloc is ~µs.

use metal::{
    Buffer, CommandBufferRef, ComputePipelineState, MTLSize, NSUInteger,
};

use super::metal::{MetalBackend, MetalError, MtlBuffer};
use super::variants::{RMS_NORM_EPS, VARIANT};

/// Errors from GPU RMSNorm.
#[derive(Debug, thiserror::Error)]
pub enum GpuNormError {
    #[error("x must be HIDDEN_DIM={expected} floats, got {actual}")]
    BadXLen { expected: usize, actual: usize },
    #[error("out must be HIDDEN_DIM={expected} floats, got {actual}")]
    BadOutLen { expected: usize, actual: usize },
    #[error(
        "weight_bf16 must be HIDDEN_DIM*2={expected} bytes, got {actual}"
    )]
    BadWeightLen { expected: usize, actual: usize },
    #[error("Metal backend: {0}")]
    Metal(#[from] MetalError),
}

/// GPU RMSNorm: `out[i] = x[i] / sqrt(mean(x²) + eps) * bf16_to_f32(weight[i])`.
/// `weight_bf16` is the raw little-endian BF16 byte sequence (typically
/// from `WeightFile::tensor_bytes`).
pub fn gpu_rms_norm_fused(
    metal: &mut MetalBackend,
    x: &[f32],
    weight_bf16: &[u8],
    out: &mut [f32],
) -> Result<(), GpuNormError> {
    let v = VARIANT;
    if x.len() != v.hidden_dim {
        return Err(GpuNormError::BadXLen {
            expected: v.hidden_dim,
            actual: x.len(),
        });
    }
    if out.len() != v.hidden_dim {
        return Err(GpuNormError::BadOutLen {
            expected: v.hidden_dim,
            actual: out.len(),
        });
    }
    let expected_w = v.hidden_dim * 2;
    if weight_bf16.len() != expected_w {
        return Err(GpuNormError::BadWeightLen {
            expected: expected_w,
            actual: weight_bf16.len(),
        });
    }

    let sum_pipe = metal.pipeline("rms_norm_sum_sq")?.clone();
    let apply_pipe = metal.pipeline("rms_norm_apply_bf16")?.clone();

    let device = metal.device();
    let buf_x = MtlBuffer::<f32>::with_data(device, x);
    let buf_w = MtlBuffer::<u8>::with_data(device, weight_bf16);
    let buf_sum_sq = MtlBuffer::<f32>::with_len(device, 1);
    let buf_out = MtlBuffer::<f32>::with_len(device, v.hidden_dim);

    let cmdbuf = metal.queue().new_command_buffer();

    // Stage 1: sum_sq — single threadgroup of 256 threads.
    {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&sum_pipe);
        enc.set_buffer(0, Some(buf_x.raw()), 0);
        enc.set_buffer(1, Some(buf_sum_sq.raw()), 0);
        let dim = v.hidden_dim as u32;
        enc.set_bytes(2, 4, (&dim as *const u32).cast());
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }

    // Stage 2: apply with bf16 weight — per-element.
    {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&apply_pipe);
        enc.set_buffer(0, Some(buf_x.raw()), 0);
        enc.set_buffer(1, Some(buf_w.raw()), 0);
        enc.set_buffer(2, Some(buf_sum_sq.raw()), 0);
        enc.set_buffer(3, Some(buf_out.raw()), 0);
        let dim = v.hidden_dim as u32;
        let eps = RMS_NORM_EPS;
        enc.set_bytes(4, 4, (&dim as *const u32).cast());
        enc.set_bytes(5, 4, (&eps as *const f32).cast());
        let num_tgs = (dim + 255) / 256;
        enc.dispatch_thread_groups(
            MTLSize::new(num_tgs as NSUInteger, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }

    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    out.copy_from_slice(&buf_out.to_vec());
    Ok(())
}

/// Pre-fetched pipelines for the bf16-weighted RMSNorm chain. Used by
/// the per-layer fast path ([`encode_rms_norm_bf16_into`]); fetching
/// once per layer-forward avoids the lazy-compile in the hot inner
/// dispatch.
pub struct RmsNormBf16Pipelines {
    pub sum: ComputePipelineState,
    pub apply: ComputePipelineState,
}

impl RmsNormBf16Pipelines {
    pub fn fetch(metal: &mut MetalBackend) -> Result<Self, MetalError> {
        Ok(Self {
            sum: metal.pipeline("rms_norm_sum_sq")?.clone(),
            apply: metal.pipeline("rms_norm_apply_bf16")?.clone(),
        })
    }
}

/// Encode the bf16-weighted RMSNorm pair into `cmdbuf`. Two dispatches:
///
/// 1. `rms_norm_sum_sq` — single threadgroup of 256 threads computes
///    `Σ x[i]²` into `sum_sq[0]`.
/// 2. `rms_norm_apply_bf16` — per-element `out[i] = x[i] *
///    rsqrt(sum_sq[0]/dim + eps) * bf16_to_f32(weight[i])`.
///
/// Used by the per-layer fast path (slice 5d-2) to replace the CPU
/// `rms_norm_cpu` + host-staging block at the top of
/// `linear_attn_layer_forward` / `full_attn_layer_forward`. The
/// weight is read from `weight_buf` at `weight_off` bytes — typically
/// the shared [`super::mtl_weight_buf::MtlWeightBuf`] with a per-
/// layer `input_layernorm.weight` offset from
/// [`super::layer_weight_cache::LayerWeightCache`].
///
/// Both dispatches go into fresh encoders within `cmdbuf`. Metal
/// orders encoders in commit order, so subsequent dispatches reading
/// from `out_buf` see the normalized result. Bit-exact against the
/// C path's CMD3 fast-path encoders (`infer.m:5712..5744`); see
/// slice 9e for the per-PSO determinism finding.
#[allow(clippy::too_many_arguments)]
pub fn encode_rms_norm_bf16_into(
    cmdbuf: &CommandBufferRef,
    pipes: &RmsNormBf16Pipelines,
    input: &Buffer,
    weight_buf: &Buffer,
    weight_off: u64,
    sum_sq: &Buffer,
    out: &Buffer,
    dim: u32,
    eps: f32,
) {
    // Stage 1: sum_sq — single threadgroup of 256 threads, two-stage
    // reduction (simd_sum + threadgroup-shared second stage).
    {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipes.sum);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(sum_sq), 0);
        enc.set_bytes(2, 4, (&dim as *const u32).cast());
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
    // Stage 2: apply with bf16 weight — per-element. 256 threads/group.
    {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipes.apply);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(weight_buf), weight_off as NSUInteger);
        enc.set_buffer(2, Some(sum_sq), 0);
        enc.set_buffer(3, Some(out), 0);
        enc.set_bytes(4, 4, (&dim as *const u32).cast());
        enc.set_bytes(5, 4, (&eps as *const f32).cast());
        let num_tgs = (dim + 255) / 256;
        enc.dispatch_thread_groups(
            MTLSize::new(num_tgs as NSUInteger, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
}
