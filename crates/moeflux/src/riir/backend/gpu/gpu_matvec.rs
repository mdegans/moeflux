//! Generic 4-bit dequant matvec encoder — Phase 4c.
//!
//! Port of `gpu_encode_batch_matvec` (infer.m:1520). Encodes one or
//! more `dequant_matvec_4bit_v3` (or `_fast` for in_dim > 4096)
//! dispatches into a command buffer, reading weights / scales /
//! biases at byte offsets within an [`MtlWeightBuf`] and writing
//! into caller-supplied output buffers.
//!
//! This is the centerpiece for projection matvecs (qkv / z / beta /
//! alpha for linear-attn; q / k / v for full-attn; o_proj for both;
//! gate logits for the MoE router; shared-expert FFN). The
//! per-expert FFN matvecs in `expert_forward.rs` use a different
//! shape — they read a single packed expert blob, not the global
//! weight buffer — and stay in their own module.
//!
//! ## Pipeline selection
//!
//! - `dequant_matvec_4bit_v3` when `in_dim <= 4096`. Threadgroup
//!   shape: 256 threads, `(out_dim + 7) / 8` threadgroups (8 rows
//!   per group).
//! - `dequant_matvec_4bit_fast` otherwise. Threadgroup shape: 64
//!   threads, `out_dim` threadgroups (1 row per group).
//!
//! Per the Phase 3 finding, both kernels are bit-exact per-PSO on
//! the same device.

use metal::{Buffer, CommandBufferRef, MTLSize, NSUInteger};

use super::encoder::pipeline_bundle;
use crate::riir::io::mtl_weight_buf::MtlWeightBuf;
use crate::riir::variants::GROUP_SIZE;

/// One projection matvec to encode. Weight / scales / biases live at
/// byte offsets within the shared [`MtlWeightBuf`]; input and output
/// are caller-owned buffers (typically scratch / persistent state).
pub struct MatvecSpec<'a> {
    /// Byte offset of the packed-weight tensor inside the shared
    /// weight buffer.
    pub w_off: u64,
    /// Byte offset of the bf16 scales tensor.
    pub s_off: u64,
    /// Byte offset of the bf16 biases tensor.
    pub b_off: u64,
    /// Input vector buffer (`HIDDEN_DIM` floats typically).
    pub input: &'a Buffer,
    /// Output vector buffer (`out_dim` floats).
    pub output: &'a Buffer,
    /// Output dimension (matvec produces `out_dim` floats).
    pub out_dim: u32,
    /// Input dimension. Selects v3 vs fast dispatch via the 4096
    /// threshold for 4-bit weights.
    pub in_dim: u32,
    /// Quantization bits — 4 (default) or 8. On A3B `mlp.gate.weight`
    /// and `mlp.shared_expert_gate.weight` are 8-bit; everything else
    /// is 4-bit.
    pub bits: u32,
}

pipeline_bundle! {
    /// Pre-fetched matvec pipelines. All three flavors compile lazily on
    /// first request via [`crate::riir::backend::gpu::metal::MetalContext::pipeline`].
    ///
    /// `*_n_tokens` variants are the batched-prefill versions: same weights
    /// applied to N stacked input vectors in one dispatch. See
    /// [`encode_matvec_n_tokens`]. 8-bit batched is not implemented yet —
    /// the 8-bit projections (gate logits etc.) on Qwen3-A3B are small
    /// enough that per-token dispatch is cheap; revisit if measurement
    /// flags it.
    pub struct MatvecPipelines {
        v3_4bit => "dequant_matvec_4bit_v3",
        fast_4bit => "dequant_matvec_4bit_fast",
        v3_8bit => "dequant_matvec_8bit_v3",
        v3_4bit_n => "dequant_matvec_4bit_v3_n_tokens",
        fast_4bit_n => "dequant_matvec_4bit_fast_n_tokens",
        v3_8bit_n => "dequant_matvec_8bit_v3_n_tokens",
    }
}

/// Encode one matvec dispatch into `cmdbuf`. Reuses the
/// pre-fetched pipelines so the encoder doesn't borrow `metal`.
/// Mirrors `gpu_encode_batch_matvec`'s pipeline selection
/// (infer.m:1534-1542): 8-bit always uses the v3-shaped 8-bit kernel,
/// 4-bit uses v3 when `in_dim ≤ 4096` else fast.
pub fn encode_matvec(
    cmdbuf: &CommandBufferRef,
    pipes: &MatvecPipelines,
    wf_buf: &MtlWeightBuf,
    spec: &MatvecSpec,
) {
    let group_size = GROUP_SIZE as u32;
    let (pipeline, use_v3_layout) = if spec.bits == 8 {
        (&pipes.v3_8bit, true)
    } else if spec.in_dim <= 4096 {
        (&pipes.v3_4bit, true)
    } else {
        (&pipes.fast_4bit, false)
    };

    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(wf_buf.buffer()), spec.w_off as NSUInteger);
    enc.set_buffer(1, Some(wf_buf.buffer()), spec.s_off as NSUInteger);
    enc.set_buffer(2, Some(wf_buf.buffer()), spec.b_off as NSUInteger);
    enc.set_buffer(3, Some(spec.input), 0);
    enc.set_buffer(4, Some(spec.output), 0);
    enc.set_bytes(5, 4, (&spec.out_dim as *const u32).cast());
    enc.set_bytes(6, 4, (&spec.in_dim as *const u32).cast());
    enc.set_bytes(7, 4, (&group_size as *const u32).cast());
    if use_v3_layout {
        let num_tgs = (spec.out_dim + 7) / 8;
        enc.dispatch_thread_groups(
            MTLSize::new(num_tgs as NSUInteger, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    } else {
        enc.dispatch_thread_groups(
            MTLSize::new(spec.out_dim as NSUInteger, 1, 1),
            MTLSize::new(64, 1, 1),
        );
    }
    enc.end_encoding();
}

/// Batched-prefill variant: apply one 4-bit-quantized weight matrix to
/// `n_tokens` stacked f32 input vectors, writing `[n_tokens, out_dim]`
/// f32 output. Mirrors [`encode_matvec`]'s pipeline selection: v3 when
/// `in_dim ≤ 4096`, fast otherwise. 8-bit is not (yet) batched — see
/// [`MatvecPipelines`] doc.
///
/// Per-(row, token) arithmetic matches the corresponding single-row
/// kernel exactly, so N=1 is bit-exact vs [`encode_matvec`] on the
/// same path. The 8-bit case panics — caller must not pass `bits=8`.
///
/// `input_off` / `output_off` are byte offsets into `input` / `output`,
/// letting one big buffer hold multiple sub-batches (used by MoE
/// permute-and-fuse to keep all buckets in a single packed buffer).
/// Pass 0 for plain `[n_tokens, dim]` layouts.
#[allow(clippy::too_many_arguments)]
pub fn encode_matvec_n_tokens(
    cmdbuf: &CommandBufferRef,
    pipes: &MatvecPipelines,
    w_buf: &Buffer,
    w_off: u64,
    s_off: u64,
    b_off: u64,
    input: &Buffer,
    input_off: u64,
    output: &Buffer,
    output_off: u64,
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    bits: u32,
) {
    assert!(
        bits == 4 || bits == 8,
        "encode_matvec_n_tokens: only 4-bit / 8-bit supported (got bits={})",
        bits
    );
    if n_tokens == 0 {
        return;
    }
    let group_size = GROUP_SIZE as u32;
    let use_v3 = bits == 8 || in_dim <= 4096;
    let pipeline = if bits == 8 {
        &pipes.v3_8bit_n
    } else if use_v3 {
        &pipes.v3_4bit_n
    } else {
        &pipes.fast_4bit_n
    };

    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(w_buf), w_off as NSUInteger);
    enc.set_buffer(1, Some(w_buf), s_off as NSUInteger);
    enc.set_buffer(2, Some(w_buf), b_off as NSUInteger);
    enc.set_buffer(3, Some(input), input_off as NSUInteger);
    enc.set_buffer(4, Some(output), output_off as NSUInteger);
    enc.set_bytes(5, 4, (&out_dim as *const u32).cast());
    enc.set_bytes(6, 4, (&in_dim as *const u32).cast());
    enc.set_bytes(7, 4, (&group_size as *const u32).cast());
    enc.set_bytes(8, 4, (&n_tokens as *const u32).cast());

    if use_v3 {
        let num_row_tiles = (out_dim + 7) / 8;
        enc.set_bytes(9, 4, (&num_row_tiles as *const u32).cast());
        let total_tgs =
            (num_row_tiles as u64).saturating_mul(n_tokens as u64);
        enc.dispatch_thread_groups(
            MTLSize::new(total_tgs as NSUInteger, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    } else {
        let total_tgs = (out_dim as u64).saturating_mul(n_tokens as u64);
        enc.dispatch_thread_groups(
            MTLSize::new(total_tgs as NSUInteger, 1, 1),
            MTLSize::new(64, 1, 1),
        );
    }
    enc.end_encoding();
}

// ---------------------------------------------------------------------------
// Tiled 4-bit dequant matmul (simdgroup_matrix) — batched-prefill path
// ---------------------------------------------------------------------------

/// Token-tile height of the `mul_mm_4bit` kernel — mirrors `MM_BM` in
/// `shaders.metal`. The encoder needs it to size the dispatch grid.
pub const MM_BM: u32 = 64;
/// Output-row tile width of `mul_mm_4bit` — mirrors `MM_BN`.
pub const MM_BN: u32 = 64;

pipeline_bundle! {
    /// Pipeline for [`encode_mul_mm_4bit`] — the tiled `simdgroup_matrix`
    /// 4-bit dequant matmul. Replaces the per-(token, row) `*_n_tokens`
    /// matvec kernels on the batched-prefill path: one threadgroup
    /// computes a `MM_BM × MM_BN` output tile, so each weight slab is
    /// staged + dequantized once and reused across `MM_BM` tokens.
    pub struct MulMm4bitPipelines {
        mul_mm_4bit => "mul_mm_4bit",
    }
}

/// Encode one tiled 4-bit dequant matmul:
/// `out[t][r] = sum_k dequant_4bit(W[r, k]) * input[t][k]`.
///
/// Drop-in for [`encode_matvec_n_tokens`] on 4-bit weights — same
/// buffer/offset contract — but tiles the token axis so the weight
/// matrix is read once per `n_tokens / MM_BM` tiles instead of once per
/// token. `input_off` / `output_off` are byte offsets, matching
/// [`encode_matvec_n_tokens`].
#[allow(clippy::too_many_arguments)]
pub fn encode_mul_mm_4bit(
    cmdbuf: &CommandBufferRef,
    pipes: &MulMm4bitPipelines,
    w_buf: &Buffer,
    w_off: u64,
    s_off: u64,
    b_off: u64,
    input: &Buffer,
    input_off: u64,
    output: &Buffer,
    output_off: u64,
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
) {
    if n_tokens == 0 {
        return;
    }
    let group_size = GROUP_SIZE as u32;
    let num_m_tiles = n_tokens.div_ceil(MM_BM);
    let num_n_tiles = out_dim.div_ceil(MM_BN);
    let total_tgs = (num_m_tiles as u64) * (num_n_tiles as u64);

    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipes.mul_mm_4bit);
    enc.set_buffer(0, Some(w_buf), w_off as NSUInteger);
    enc.set_buffer(1, Some(w_buf), s_off as NSUInteger);
    enc.set_buffer(2, Some(w_buf), b_off as NSUInteger);
    enc.set_buffer(3, Some(input), input_off as NSUInteger);
    enc.set_buffer(4, Some(output), output_off as NSUInteger);
    enc.set_bytes(5, 4, (&out_dim as *const u32).cast());
    enc.set_bytes(6, 4, (&in_dim as *const u32).cast());
    enc.set_bytes(7, 4, (&group_size as *const u32).cast());
    enc.set_bytes(8, 4, (&n_tokens as *const u32).cast());
    enc.set_bytes(9, 4, (&num_m_tiles as *const u32).cast());
    enc.dispatch_thread_groups(
        MTLSize::new(total_tgs as NSUInteger, 1, 1),
        MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
}

// ---------------------------------------------------------------------------
// BF16-weight matvec (no dequant) — Cogito-V2 / DeepSeek-V3 router gate
// ---------------------------------------------------------------------------

pipeline_bundle! {
    /// Pipelines for the BF16-weight matvec. Used by the MoE router gate
    /// (`model.layers.{i}.mlp.gate.weight`, `[num_experts, hidden_dim]`
    /// BF16) which the 4-bit dequant matvec can't handle. Sibling of
    /// [`MatvecPipelines`].
    ///
    /// `bf16_n` is the batched-prefill variant: same weights applied to N
    /// token activations in one dispatch. See [`encode_bf16_matmul_n_tokens`].
    pub struct BfMatvecPipelines {
        bf16 => "bf16_matvec",
        bf16_n => "bf16_matmul_n_tokens",
    }
}

/// One BF16-weight matvec dispatch. Reads `out_dim × in_dim` BF16
/// weights row-major from `w_buf` at `w_off` byte offset (typically
/// the shared [`MtlWeightBuf`]), input from `input`, writes f32
/// output to `output`. Threadgroup-per-output-row, 256 threads/group.
#[allow(clippy::too_many_arguments)]
pub fn encode_bf16_matvec(
    cmdbuf: &CommandBufferRef,
    pipes: &BfMatvecPipelines,
    w_buf: &Buffer,
    w_off: u64,
    input: &Buffer,
    output: &Buffer,
    in_dim: u32,
    out_dim: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipes.bf16);
    enc.set_buffer(0, Some(w_buf), w_off as NSUInteger);
    enc.set_buffer(1, Some(input), 0);
    enc.set_buffer(2, Some(output), 0);
    enc.set_bytes(3, 4, (&in_dim as *const u32).cast());
    enc.set_bytes(4, 4, (&out_dim as *const u32).cast());
    enc.dispatch_thread_groups(
        MTLSize::new(out_dim as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

/// BF16-weight matmul over N tokens (same weights, different inputs).
///
/// Applies `[out_dim, in_dim]` BF16 weights at `w_off` to `n_tokens`
/// stacked `in_dim`-wide f32 activations and writes `[n_tokens, out_dim]`
/// f32. Per-(row, token) arithmetic matches [`encode_bf16_matvec`]
/// exactly, so the two are bit-exact-mod-fp-reorder when fed identical
/// inputs. Grid: `out_dim * n_tokens` threadgroups, 256 threads/group.
#[allow(clippy::too_many_arguments)]
pub fn encode_bf16_matmul_n_tokens(
    cmdbuf: &CommandBufferRef,
    pipes: &BfMatvecPipelines,
    w_buf: &Buffer,
    w_off: u64,
    input: &Buffer,
    output: &Buffer,
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
) {
    if n_tokens == 0 {
        return;
    }
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipes.bf16_n);
    enc.set_buffer(0, Some(w_buf), w_off as NSUInteger);
    enc.set_buffer(1, Some(input), 0);
    enc.set_buffer(2, Some(output), 0);
    enc.set_bytes(3, 4, (&in_dim as *const u32).cast());
    enc.set_bytes(4, 4, (&out_dim as *const u32).cast());
    enc.set_bytes(5, 4, (&n_tokens as *const u32).cast());
    let total_tgs = (out_dim as u64).saturating_mul(n_tokens as u64);
    enc.dispatch_thread_groups(
        MTLSize::new(total_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}
