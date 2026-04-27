//! Linear-attention GPU kernel encoders — Phase 4c.
//!
//! Five thin wrappers around the linear-attn-specific pipelines in
//! `shaders.metal`. Each takes a command buffer, the buffers /
//! offsets the kernel reads, and emits the dispatch shape the C path
//! uses (infer.m:4353–4439).
//!
//! Kernels:
//!
//! - `conv1d_step` — depthwise 1D conv + SiLU (qkv_in → conv_out,
//!   updates conv_state).
//! - `rms_norm_qk` — bare per-head RMSNorm for q and k in place.
//! - `compute_decay_beta` — folds (alpha, beta, A_log, dt_bias) →
//!   (g_decay, beta_gate) per v-head.
//! - `gated_delta_net_step` — the recurrence step.
//! - `gated_rms_norm` — RMSNormGated for the recurrence output, with
//!   z gating + bf16 weight scaling.
//!
//! All five live in this module because they're only used by the
//! linear-attn forward and have no other consumers in the port.

use metal::{
    Buffer, CommandBufferRef, ComputePipelineState, MTLSize, NSUInteger,
};

use super::metal::{MetalBackend, MetalError};
use super::variants::{Variant, VARIANT};

/// All linear-attn pipelines pre-fetched. Used by the layer-forward
/// composer so the encode loop doesn't borrow `metal` mid-encode.
pub struct LinearAttnPipelines {
    pub conv1d_step: ComputePipelineState,
    pub rms_norm_qk: ComputePipelineState,
    pub compute_decay_beta: ComputePipelineState,
    pub delta_net_step: ComputePipelineState,
    pub gated_rms_norm: ComputePipelineState,
}

impl LinearAttnPipelines {
    pub fn fetch(metal: &mut MetalBackend) -> Result<Self, MetalError> {
        Ok(Self {
            conv1d_step: metal.pipeline("conv1d_step")?.clone(),
            rms_norm_qk: metal.pipeline("rms_norm_qk")?.clone(),
            compute_decay_beta: metal.pipeline("compute_decay_beta")?.clone(),
            delta_net_step: metal.pipeline("gated_delta_net_step")?.clone(),
            gated_rms_norm: metal.pipeline("gated_rms_norm")?.clone(),
        })
    }
}

/// Encode `conv1d_step` with the C-side dispatch shape.
/// `(conv_dim + 255) / 256` threadgroups × 256 threads.
#[allow(clippy::too_many_arguments)]
pub fn encode_conv1d_step(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    conv_state: &Buffer,
    qkv_in: &Buffer,
    weight_buf: &Buffer,
    weight_off: u64,
    conv_out: &Buffer,
    conv_dim: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(conv_state), 0);
    enc.set_buffer(1, Some(qkv_in), 0);
    enc.set_buffer(2, Some(weight_buf), weight_off as NSUInteger);
    enc.set_buffer(3, Some(conv_out), 0);
    enc.set_bytes(4, 4, (&conv_dim as *const u32).cast());
    let num_tgs = (conv_dim + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

/// Encode `rms_norm_qk` with q at offset 0, k at offset
/// `LINEAR_TOTAL_KEY * sizeof(float)` of the conv-output buffer.
/// `num_k_heads` threadgroups × `key_dim` threads.
pub fn encode_rms_norm_qk(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    conv_out: &Buffer,
    num_k_heads: u32,
    key_dim: u32,
) {
    let inv_scale = 1.0f32 / (key_dim as f32).sqrt();
    let k_off = (VARIANT.linear_total_key()) * std::mem::size_of::<f32>();

    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(conv_out), 0);
    enc.set_buffer(1, Some(conv_out), k_off as NSUInteger);
    enc.set_bytes(2, 4, (&key_dim as *const u32).cast());
    enc.set_bytes(3, 4, (&inv_scale as *const f32).cast());
    enc.dispatch_thread_groups(
        MTLSize::new(num_k_heads as NSUInteger, 1, 1),
        MTLSize::new(key_dim as NSUInteger, 1, 1),
    );
    enc.end_encoding();
}

/// Encode `compute_decay_beta`. 1 threadgroup × `num_v_heads`
/// threads, one thread per v-head.
#[allow(clippy::too_many_arguments)]
pub fn encode_compute_decay_beta(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    alpha_in: &Buffer,
    beta_in: &Buffer,
    weight_buf: &Buffer,
    a_log_off: u64,
    dt_bias_off: u64,
    g_decay_out: &Buffer,
    beta_gate_out: &Buffer,
    num_v_heads: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(alpha_in), 0);
    enc.set_buffer(1, Some(beta_in), 0);
    enc.set_buffer(2, Some(weight_buf), a_log_off as NSUInteger);
    enc.set_buffer(3, Some(weight_buf), dt_bias_off as NSUInteger);
    enc.set_buffer(4, Some(g_decay_out), 0);
    enc.set_buffer(5, Some(beta_gate_out), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(num_v_heads as NSUInteger, 1, 1),
    );
    enc.end_encoding();
}

/// Encode `gated_delta_net_step`. q / k / v live at offsets in the
/// conv-output buffer. `num_v_heads` threadgroups × `value_dim`
/// threads (note: dispatch is `value_dim`, not `key_dim` — `vi`
/// indexes the value channel).
#[allow(clippy::too_many_arguments)]
pub fn encode_delta_net_step(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    state: &Buffer,
    conv_out: &Buffer,
    g_decay: &Buffer,
    beta_gate: &Buffer,
    output: &Buffer,
    num_v_heads: u32,
    value_dim: u32,
    k_heads_per_v: u32,
) {
    let key_size = std::mem::size_of::<f32>() * VARIANT.linear_total_key();
    let q_off: u64 = 0;
    let k_off: u64 = key_size as u64;
    let v_off: u64 = (key_size * 2) as u64;

    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(state), 0);
    enc.set_buffer(1, Some(conv_out), q_off as NSUInteger);
    enc.set_buffer(2, Some(conv_out), k_off as NSUInteger);
    enc.set_buffer(3, Some(conv_out), v_off as NSUInteger);
    enc.set_buffer(4, Some(g_decay), 0);
    enc.set_buffer(5, Some(beta_gate), 0);
    enc.set_buffer(6, Some(output), 0);
    enc.set_bytes(7, 4, (&k_heads_per_v as *const u32).cast());
    enc.dispatch_thread_groups(
        MTLSize::new(num_v_heads as NSUInteger, 1, 1),
        MTLSize::new(value_dim as NSUInteger, 1, 1),
    );
    enc.end_encoding();
}

/// Encode `gated_rms_norm`. `num_v_heads` threadgroups × `value_dim`
/// threads.
#[allow(clippy::too_many_arguments)]
pub fn encode_gated_rms_norm(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    values: &Buffer,
    z: &Buffer,
    weight_buf: &Buffer,
    weight_off: u64,
    output: &Buffer,
    num_v_heads: u32,
    value_dim: u32,
) {
    let eps = crate::riir::variants::RMS_NORM_EPS;
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(values), 0);
    enc.set_buffer(1, Some(z), 0);
    enc.set_buffer(2, Some(weight_buf), weight_off as NSUInteger);
    enc.set_buffer(3, Some(output), 0);
    enc.set_bytes(4, 4, (&value_dim as *const u32).cast());
    enc.set_bytes(5, 4, (&eps as *const f32).cast());
    enc.dispatch_thread_groups(
        MTLSize::new(num_v_heads as NSUInteger, 1, 1),
        MTLSize::new(value_dim as NSUInteger, 1, 1),
    );
    enc.end_encoding();
}

// Avoid unused-import warning if Variant isn't referenced in helpers.
#[allow(dead_code)]
const _VARIANT_USE: Variant = VARIANT;
