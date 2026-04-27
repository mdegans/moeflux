//! GPU LM head — port of C `lm_head_forward` (infer.m:3090) which
//! delegates to `fast_dequant_matvec` against `lm_head.{weight,scales,
//! biases}`.
//!
//! Replaces [`super::lm_head::lm_head_cpu`] in [`crate::riir::RsCtx::
//! step_internal`]'s logits-emission tail. The 2026-04-27 profile of
//! the riir port pointed `lm_head_cpu` at 59% of CPU time per token —
//! a 2048×248320 4-bit dequant matvec is hopeless on CPU. Routing it
//! through the existing `dequant_matvec_4bit_v3` pipeline (the same
//! kernel used for projection matvecs in linear-attn / full-attn /
//! shared-expert FFN) reduces this to one Metal dispatch (~31040
//! threadgroups × 256 threads) per token.
//!
//! The bare per-kernel `lm_head_cpu` accessor on [`crate::riir::RsCtx`]
//! stays — slice 6's diff oracle calls it directly to verify the CPU
//! arithmetic is bit-exact against the C `cpu_dequant_matvec` path.
//! This module is the production path; the end-to-end `eval_token` /
//! `eval_prompt` diff tests cover its bit/cosine-1.0 agreement against
//! the C `lm_head_forward` GPU dispatch.

use metal::{Buffer, MTLResourceOptions, NSUInteger};

use super::gpu_matvec::{encode_matvec, MatvecPipelines, MatvecSpec};
use super::metal::{MetalBackend, MetalError};
use super::mtl_weight_buf::{MtlWeightBuf, MtlWeightBufError};
use super::variants::VARIANT;
use super::weight_file::WeightFile;

/// Errors specific to the GPU LM head dispatch.
#[derive(Debug, thiserror::Error)]
pub enum GpuLmHeadError {
    #[error("Metal: {0}")]
    Metal(#[from] MetalError),
    #[error("weight buffer: {0}")]
    WeightBuf(#[from] MtlWeightBufError),
    #[error("missing lm_head tensor: {0}")]
    MissingTensor(&'static str),
    #[error("input length {got} != HIDDEN_DIM {expected}")]
    InputLen { got: usize, expected: usize },
    #[error("output length {got} != VOCAB_SIZE {expected}")]
    OutputLen { got: usize, expected: usize },
}

/// Persistent GPU state for the LM head dispatch. Allocated once per
/// [`crate::riir::RsCtx`] (lazily on first `eval_token` / `eval_prompt`
/// that emits logits) and reused on every subsequent token.
///
/// Owns:
/// - `pipelines`: pre-fetched matvec pipelines (LM head uses
///   `dequant_matvec_4bit_v3` since `HIDDEN_DIM = 2048 ≤ 4096`).
/// - `input_buf`: shared-storage MTL buffer for the post-final-norm
///   hidden state (`HIDDEN_DIM` floats, ~8 KB).
/// - `logits_buf`: shared-storage MTL buffer for the output logits
///   (`VOCAB_SIZE` floats, ~993 KB on A3B's 248320 vocab).
/// - `w_off` / `s_off` / `b_off`: byte offsets of the lm_head weight /
///   scales / biases tensors inside the shared [`MtlWeightBuf`],
///   resolved once at construction.
pub struct GpuLmHead {
    pipelines: MatvecPipelines,
    input_buf: Buffer,
    logits_buf: Buffer,
    w_off: u64,
    s_off: u64,
    b_off: u64,
}

// metal-rs's `Buffer` and `ComputePipelineState` are not auto-`Send`.
// `MetalBackend` opts in for the same reason: single-owner discipline
// makes cross-thread move safe; concurrent access on the same `RsCtx`
// is forbidden by the public API contract.
unsafe impl Send for GpuLmHead {}

impl GpuLmHead {
    /// Pre-fetch pipelines, allocate persistent input + logits buffers,
    /// and resolve the lm_head tensor offsets. Called lazily from
    /// [`crate::riir::RsCtx::step_internal`] when logits are first
    /// requested.
    pub fn new(
        metal: &mut MetalBackend,
        wf: &WeightFile,
        wf_buf: &MtlWeightBuf,
    ) -> Result<Self, GpuLmHeadError> {
        let pipelines = MatvecPipelines::fetch(metal)?;
        let device = metal.device();
        let v = VARIANT;

        let input_buf = device.new_buffer(
            (v.hidden_dim * std::mem::size_of::<f32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        let logits_buf = device.new_buffer(
            (v.vocab_size * std::mem::size_of::<f32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );

        let w_off = wf_buf
            .tensor_offset(wf, "lm_head.weight")?
            .ok_or(GpuLmHeadError::MissingTensor("lm_head.weight"))?;
        let s_off = wf_buf
            .tensor_offset(wf, "lm_head.scales")?
            .ok_or(GpuLmHeadError::MissingTensor("lm_head.scales"))?;
        let b_off = wf_buf
            .tensor_offset(wf, "lm_head.biases")?
            .ok_or(GpuLmHeadError::MissingTensor("lm_head.biases"))?;

        Ok(Self {
            pipelines,
            input_buf,
            logits_buf,
            w_off,
            s_off,
            b_off,
        })
    }

    /// Compute logits for `hidden` (the post-final-norm hidden state)
    /// and write `VOCAB_SIZE` floats into `logits`.
    ///
    /// Mirrors `lm_head_forward` (infer.m:3109) → `fast_dequant_matvec`.
    /// Single command buffer, single dispatch, blocking wait —
    /// `step_internal`'s emission point is the only caller and runs
    /// after every layer's deferred dispatch has been drained.
    pub fn forward(
        &self,
        metal: &MetalBackend,
        wf_buf: &MtlWeightBuf,
        hidden: &[f32],
        logits: &mut [f32],
    ) -> Result<(), GpuLmHeadError> {
        let v = VARIANT;
        if hidden.len() != v.hidden_dim {
            return Err(GpuLmHeadError::InputLen {
                got: hidden.len(),
                expected: v.hidden_dim,
            });
        }
        if logits.len() != v.vocab_size {
            return Err(GpuLmHeadError::OutputLen {
                got: logits.len(),
                expected: v.vocab_size,
            });
        }

        // Stage hidden state into the shared-storage input buffer.
        // SAFETY: shared-storage allocation, written to from the host;
        // no GPU work is in flight (caller drained all deferred
        // dispatches into `hidden_final` before invoking lm_head).
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden.as_ptr(),
                self.input_buf.contents() as *mut f32,
                v.hidden_dim,
            );
        }

        let cmdbuf = metal.queue().new_command_buffer();
        let spec = MatvecSpec {
            w_off: self.w_off,
            s_off: self.s_off,
            b_off: self.b_off,
            input: &self.input_buf,
            output: &self.logits_buf,
            out_dim: v.vocab_size as u32,
            in_dim: v.hidden_dim as u32,
            bits: 4,
        };
        encode_matvec(cmdbuf, &self.pipelines, wf_buf, &spec);
        cmdbuf.commit();
        cmdbuf.wait_until_completed();

        // Read back into caller-supplied `logits`. Shared storage
        // means this is a memcpy out of the same pages the GPU just
        // wrote — no cross-storage transfer.
        // SAFETY: dispatch has completed; logits_buf contents are
        // settled f32 values produced by the kernel.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.logits_buf.contents() as *const f32,
                logits.as_mut_ptr(),
                v.vocab_size,
            );
        }
        Ok(())
    }
}

impl std::fmt::Debug for GpuLmHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuLmHead")
            .field("w_off", &self.w_off)
            .field("s_off", &self.s_off)
            .field("b_off", &self.b_off)
            .finish()
    }
}
