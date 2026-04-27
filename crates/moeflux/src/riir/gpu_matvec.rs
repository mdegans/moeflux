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

use metal::{
    Buffer, CommandBufferRef, ComputePipelineState, MTLSize, NSUInteger,
};

use super::metal::{MetalBackend, MetalError};
use super::mtl_weight_buf::MtlWeightBuf;
use super::variants::GROUP_SIZE;

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

/// Pre-fetched matvec pipelines. All three flavors compile lazily on
/// first request via [`MetalBackend::pipeline`].
pub struct MatvecPipelines {
    pub v3_4bit: ComputePipelineState,
    pub fast_4bit: ComputePipelineState,
    pub v3_8bit: ComputePipelineState,
}

impl MatvecPipelines {
    pub fn fetch(metal: &mut MetalBackend) -> Result<Self, MetalError> {
        Ok(Self {
            v3_4bit: metal.pipeline("dequant_matvec_4bit_v3")?.clone(),
            fast_4bit: metal.pipeline("dequant_matvec_4bit_fast")?.clone(),
            v3_8bit: metal.pipeline("dequant_matvec_8bit_v3")?.clone(),
        })
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
