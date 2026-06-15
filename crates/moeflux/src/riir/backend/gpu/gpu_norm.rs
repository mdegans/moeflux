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
    Buffer, CommandBufferRef, ComputePipelineState, Device, MTLResourceOptions, MTLSize, NSUInteger,
};

use super::encoder::pipeline_bundle;
use super::metal::{MetalContext, MetalError, MtlBuffer};
use crate::riir::variants::{RMS_NORM_EPS, VARIANT};

/// Errors from GPU RMSNorm.
#[derive(Debug, thiserror::Error)]
pub enum GpuNormError {
    #[error("x must be HIDDEN_DIM={expected} floats, got {actual}")]
    BadXLen { expected: usize, actual: usize },
    #[error("out must be HIDDEN_DIM={expected} floats, got {actual}")]
    BadOutLen { expected: usize, actual: usize },
    #[error("weight_bf16 must be HIDDEN_DIM*2={expected} bytes, got {actual}")]
    BadWeightLen { expected: usize, actual: usize },
    #[error("Metal backend: {0}")]
    Metal(#[from] MetalError),
}

/// GPU RMSNorm: `out[i] = x[i] / sqrt(mean(x²) + eps) * bf16_to_f32(weight[i])`.
/// `weight_bf16` is the raw little-endian BF16 byte sequence (typically
/// from `WeightFile::tensor_bytes`).
pub fn gpu_rms_norm_fused(
    metal: &mut MetalContext,
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
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
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

pipeline_bundle! {
    /// Pre-fetched pipelines for the bf16-weighted RMSNorm chain. Used by
    /// the per-layer fast path ([`encode_rms_norm_bf16_into`]); fetching
    /// once per layer-forward avoids the lazy-compile in the hot inner
    /// dispatch.
    pub struct RmsNormBf16Pipelines {
        sum => "rms_norm_sum_sq",
        apply => "rms_norm_apply_bf16",
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
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
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

pipeline_bundle! {
    /// Pipeline for the fused batched RMSNorm-bf16 kernel. One dispatch
    /// covers the full `[n_tokens, dim]` stack; intermediate `sum_sq`
    /// lives in threadgroup memory only.
    pub struct RmsNormBf16FusedNTokensPipeline {
        pso => "rms_norm_bf16_fused_n_tokens",
    }
}

/// Encode the fused batched RMSNorm into `cmdbuf`. One threadgroup per
/// token; intra-tg parallel sum_sq reduction + apply. Output is
/// `[n_tokens, dim]` f32, ready for downstream batched matvec.
///
/// Replaces the per-token loop that called `encode_rms_norm_bf16_into`
/// once per token with `commit_and_wait_labeled` between iterations
/// (the dominant per-layer commit churn in the existing "batched" path).
#[allow(clippy::too_many_arguments)]
pub fn encode_rms_norm_bf16_fused_n_tokens(
    cmdbuf: &CommandBufferRef,
    pipe: &RmsNormBf16FusedNTokensPipeline,
    input: &Buffer,
    weight_buf: &Buffer,
    weight_off: u64,
    out: &Buffer,
    dim: u32,
    n_tokens: u32,
    eps: f32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipe.pso);
    enc.set_buffer(0, Some(input), 0);
    enc.set_buffer(1, Some(weight_buf), weight_off as NSUInteger);
    enc.set_buffer(2, Some(out), 0);
    enc.set_bytes(3, 4, (&dim as *const u32).cast());
    enc.set_bytes(4, 4, (&eps as *const f32).cast());
    enc.dispatch_thread_groups(
        MTLSize::new(n_tokens as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

/// Encode `out[t*dim + i] = a[t*dim + i] + b[t*dim + i]` over the full
/// `n_tokens * dim` element range. Single 1D dispatch.
pub fn encode_residual_add_n_tokens_into(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    n_tokens: u32,
    dim: u32,
) {
    let total = n_tokens * dim;
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(a), 0);
    enc.set_buffer(1, Some(b), 0);
    enc.set_buffer(2, Some(out), 0);
    enc.set_bytes(3, 4, (&total as *const u32).cast());
    let num_tgs = (total + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

/// Encode in-place vanilla RoPE over an `[n_tokens, num_heads,
/// head_dim]` stack — see the `rope_n_tokens` kernel in
/// shaders.metal. One thread per `(token, head, i)` with
/// `i ∈ [0, rotary_dim/2)`; `inv_freq` is the precomputed
/// `rotary_dim/2`-length frequency table.
#[allow(clippy::too_many_arguments)]
pub fn encode_rope_n_tokens_into(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    x: &Buffer,
    inv_freq: &Buffer,
    n_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    start_pos: i32,
) {
    let total = n_tokens * num_heads * (rotary_dim / 2);
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(x), 0);
    enc.set_buffer(1, Some(inv_freq), 0);
    enc.set_bytes(2, 4, (&n_tokens as *const u32).cast());
    enc.set_bytes(3, 4, (&num_heads as *const u32).cast());
    enc.set_bytes(4, 4, (&head_dim as *const u32).cast());
    enc.set_bytes(5, 4, (&rotary_dim as *const u32).cast());
    enc.set_bytes(6, 4, (&start_pos as *const i32).cast());
    let num_tgs = (total + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

/// Encode a batched 4-bit token-embedding gather — see the
/// `embed_gather_4bit` kernel in shaders.metal. For each of `n_tokens`
/// tokens, row `token_ids[t]` of the affine-packed embedding tensor is
/// dequantized into `out` (`[n_tokens, hidden_dim]`). `w_buf` is the
/// shared [`MtlWeightBuf`] buffer; `w_off` / `s_off` / `b_off` are the
/// byte offsets of `model.embed_tokens.{weight,scales,biases}`.
#[allow(clippy::too_many_arguments)]
pub fn encode_embed_gather_4bit_into(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    w_buf: &Buffer,
    w_off: u64,
    s_off: u64,
    b_off: u64,
    token_ids: &Buffer,
    out: &Buffer,
    n_tokens: u32,
    hidden_dim: u32,
    group_size: u32,
) {
    let total = n_tokens * hidden_dim;
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(w_buf), w_off as NSUInteger);
    enc.set_buffer(1, Some(w_buf), s_off as NSUInteger);
    enc.set_buffer(2, Some(w_buf), b_off as NSUInteger);
    enc.set_buffer(3, Some(token_ids), 0);
    enc.set_buffer(4, Some(out), 0);
    enc.set_bytes(5, 4, (&n_tokens as *const u32).cast());
    enc.set_bytes(6, 4, (&hidden_dim as *const u32).cast());
    enc.set_bytes(7, 4, (&group_size as *const u32).cast());
    let num_tgs = (total + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

/// Encode a GPU-side buffer-to-buffer memcpy via Metal's blit encoder.
/// `dim` floats from `src` → `dst`. Used by the orchestrator's GPU
/// residual stream (Phase 5) to snapshot `hidden → residual` at the
/// top of each sub-block without a CPU bounce.
pub fn encode_buffer_copy_f32(cmdbuf: &CommandBufferRef, src: &Buffer, dst: &Buffer, dim: u32) {
    let bytes = (dim as NSUInteger) * std::mem::size_of::<f32>() as NSUInteger;
    let blit = cmdbuf.new_blit_command_encoder();
    blit.copy_from_buffer(src, 0, dst, 0, bytes);
    blit.end_encoding();
}

/// Public wrapper for the `residual_add` Metal kernel. `out[i] = a[i]
/// + b[i]`. Used by the orchestrator's GPU residual stream (Phase 5)
/// to avoid CPU host-bounces for per-layer residual additions.
pub fn encode_residual_add_into(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    dim: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(a), 0);
    enc.set_buffer(1, Some(b), 0);
    enc.set_buffer(2, Some(out), 0);
    enc.set_bytes(3, 4, (&dim as *const u32).cast());
    let num_tgs = (dim + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

/// Persistent GPU scratch for the orchestrator's residual + norm
/// stream across an MLA-variant token-step. One set lives on RsCtx
/// and is reused across every layer + every token (Phase 5 — full-GPU
/// residual stream landing for cogito-v2/DeepSeek-V3).
///
/// `hidden` is the per-token accumulator. It's staged once from
/// embedding host-side at the top of the step, then the layer loop
/// reads + writes it via GPU dispatches (rms_norm, attention,
/// residual_add). It's read back once at the end of the step for the
/// final lm_head matvec (which still takes a host slice today).
///
/// `residual`, `normed`, and `block_out` are per-layer scratch.
/// `sum_sq` is the 1-float scratch needed by `rms_norm_sum_sq` /
/// `rms_norm_apply_bf16`.
pub struct MlaForwardScratch {
    /// Per-token accumulator carried across the layer loop. After the
    /// embedding lookup, every per-layer residual_add updates this.
    pub hidden: Buffer,
    /// Snapshot of `hidden` taken at the top of each sub-block (pre-
    /// attn or pre-MLP), used as the residual addend.
    pub residual: Buffer,
    /// Output of the per-layer rms_norm — input to attention or MLP.
    pub normed: Buffer,
    /// Output of the post-attn MLP / MoE — addend for the post-MLP
    /// residual_add.
    pub block_out: Buffer,
    /// 1-float scratch for the rms_norm reduction.
    pub sum_sq: Buffer,
}

impl MlaForwardScratch {
    pub fn new(device: &Device) -> Self {
        let v = VARIANT;
        let f32_buf = |n: usize| {
            device.new_buffer(
                (n * std::mem::size_of::<f32>()) as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            )
        };
        Self {
            hidden: f32_buf(v.hidden_dim),
            residual: f32_buf(v.hidden_dim),
            normed: f32_buf(v.hidden_dim),
            block_out: f32_buf(v.hidden_dim),
            sum_sq: f32_buf(1),
        }
    }
}
