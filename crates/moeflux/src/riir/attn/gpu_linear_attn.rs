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

use crate::riir::backend::gpu::encoder::pipeline_bundle;
use crate::riir::variants::{Variant, VARIANT};

pipeline_bundle! {
    /// All linear-attn pipelines pre-fetched. Used by the layer-forward
    /// composer so the encode loop doesn't borrow `metal` mid-encode.
    pub struct LinearAttnPipelines {
        conv1d_step => "conv1d_step",
        conv1d_state_update => "conv1d_state_update",
        rms_norm_qk => "rms_norm_qk",
        compute_decay_beta => "compute_decay_beta",
        delta_net_step => "gated_delta_net_step",
        delta_net_chunkwise => "gated_delta_net_chunkwise",
        gated_rms_norm => "gated_rms_norm",
    }
}

/// Encode `conv1d_step` + `conv1d_state_update` for a single token —
/// the `n_tokens = 1` case of the batched two-pass conv1d.
/// `(conv_dim + 255) / 256` threadgroups × 256 threads each pass.
///
/// `qkv_in_off` is the byte offset into `qkv_in` where this token's
/// `linear_conv_dim` floats start — the per-token oracle caller binds
/// a stacked projection buffer with a per-token offset.
#[allow(clippy::too_many_arguments)]
pub fn encode_conv1d_step(
    cmdbuf: &CommandBufferRef,
    compute_pso: &ComputePipelineState,
    state_update_pso: &ComputePipelineState,
    conv_state: &Buffer,
    qkv_in: &Buffer,
    qkv_in_off: u64,
    weight_buf: &Buffer,
    weight_off: u64,
    conv_out: &Buffer,
    conv_dim: u32,
) {
    let num_tgs = (conv_dim + 255) / 256;
    let n_tokens: u32 = 1;
    // Pass 1 — compute (reads conv_state, writes conv_out).
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(compute_pso);
    enc.set_buffer(0, Some(conv_state), 0);
    enc.set_buffer(1, Some(qkv_in), qkv_in_off as NSUInteger);
    enc.set_buffer(2, Some(weight_buf), weight_off as NSUInteger);
    enc.set_buffer(3, Some(conv_out), 0);
    enc.set_bytes(4, 4, (&conv_dim as *const u32).cast());
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    // Pass 2 — history-state update.
    let enc2 = cmdbuf.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(state_update_pso);
    enc2.set_buffer(0, Some(conv_state), 0);
    enc2.set_buffer(1, Some(qkv_in), qkv_in_off as NSUInteger);
    enc2.set_bytes(2, 4, (&conv_dim as *const u32).cast());
    enc2.set_bytes(3, 4, (&n_tokens as *const u32).cast());
    enc2.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc2.end_encoding();
}

/// Encode `rms_norm_qk` for a single token — the `n_tokens = 1` case
/// of the batched kernel. q region at offset 0, k region at
/// `LINEAR_TOTAL_KEY` floats into the conv-output buffer.
/// `num_k_heads` threadgroups × `key_dim` threads.
pub fn encode_rms_norm_qk(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    conv_out: &Buffer,
    num_k_heads: u32,
    key_dim: u32,
) {
    let inv_scale = 1.0f32 / (key_dim as f32).sqrt();
    let key_offset_per_token = VARIANT.linear_total_key() as u32;
    // Stride between tokens — unused at n_tokens=1 (t=0); passed as a
    // consistent value covering the q and k regions.
    let per_token_total = key_offset_per_token + num_k_heads * key_dim;

    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(conv_out), 0);
    enc.set_bytes(1, 4, (&key_dim as *const u32).cast());
    enc.set_bytes(2, 4, (&inv_scale as *const f32).cast());
    enc.set_bytes(3, 4, (&per_token_total as *const u32).cast());
    enc.set_bytes(4, 4, (&key_offset_per_token as *const u32).cast());
    enc.dispatch_thread_groups(
        MTLSize::new(num_k_heads as NSUInteger, 1, 1),
        MTLSize::new(key_dim as NSUInteger, 1, 1),
    );
    enc.end_encoding();
}

/// Encode `compute_decay_beta` for a single token. 1 threadgroup ×
/// `num_v_heads` threads — the `n_tokens = 1` case of the batched
/// kernel (`idx` flattens to `head`).
///
/// `alpha_in_off` / `beta_in_off` are byte offsets into the stacked
/// projection buffers — the per-token oracle caller binds the alpha /
/// beta stacks with a per-token offset.
#[allow(clippy::too_many_arguments)]
pub fn encode_compute_decay_beta(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    alpha_in: &Buffer,
    alpha_in_off: u64,
    beta_in: &Buffer,
    beta_in_off: u64,
    weight_buf: &Buffer,
    a_log_off: u64,
    dt_bias_off: u64,
    g_decay_out: &Buffer,
    beta_gate_out: &Buffer,
    num_v_heads: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(alpha_in), alpha_in_off as NSUInteger);
    enc.set_buffer(1, Some(beta_in), beta_in_off as NSUInteger);
    enc.set_buffer(2, Some(weight_buf), a_log_off as NSUInteger);
    enc.set_buffer(3, Some(weight_buf), dt_bias_off as NSUInteger);
    enc.set_buffer(4, Some(g_decay_out), 0);
    enc.set_buffer(5, Some(beta_gate_out), 0);
    enc.set_bytes(6, 4, (&num_v_heads as *const u32).cast());
    enc.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(num_v_heads as NSUInteger, 1, 1),
    );
    enc.end_encoding();
}

/// Encode `gated_delta_net_step` for a single token — the
/// `n_tokens = 1` case of the batched kernel. `conv_out` holds
/// q | k | v packed; the kernel computes the offsets. `num_v_heads`
/// threadgroups × `value_dim` threads (`vi` indexes the value channel).
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
    let key_total = VARIANT.linear_total_key() as u32;
    let n_tokens: u32 = 1;

    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(state), 0);
    enc.set_buffer(1, Some(conv_out), 0);
    enc.set_buffer(2, Some(g_decay), 0);
    enc.set_buffer(3, Some(beta_gate), 0);
    enc.set_buffer(4, Some(output), 0);
    enc.set_bytes(5, 4, (&k_heads_per_v as *const u32).cast());
    enc.set_bytes(6, 4, (&n_tokens as *const u32).cast());
    enc.set_bytes(7, 4, (&key_total as *const u32).cast());
    enc.set_bytes(8, 4, (&num_v_heads as *const u32).cast());
    enc.dispatch_thread_groups(
        MTLSize::new(num_v_heads as NSUInteger, 1, 1),
        MTLSize::new(value_dim as NSUInteger, 1, 1),
    );
    enc.end_encoding();
}

/// Encode `gated_rms_norm` for a single token — the `n_tokens = 1`
/// case of the batched kernel. `num_v_heads` threadgroups ×
/// `value_dim` threads.
///
/// `z_off` / `output_off` are byte offsets into the stacked z input
/// and the stacked gated-rms-norm output — the per-token oracle caller
/// binds the per-token z slice and the per-token output slot. The
/// kernel adds `head * value_dim` on top (t = 0).
#[allow(clippy::too_many_arguments)]
pub fn encode_gated_rms_norm(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    values: &Buffer,
    z: &Buffer,
    z_off: u64,
    weight_buf: &Buffer,
    weight_off: u64,
    output: &Buffer,
    output_off: u64,
    num_v_heads: u32,
    value_dim: u32,
) {
    let eps = crate::riir::variants::RMS_NORM_EPS;
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(values), 0);
    enc.set_buffer(1, Some(z), z_off as NSUInteger);
    enc.set_buffer(2, Some(weight_buf), weight_off as NSUInteger);
    enc.set_buffer(3, Some(output), output_off as NSUInteger);
    enc.set_bytes(4, 4, (&value_dim as *const u32).cast());
    enc.set_bytes(5, 4, (&eps as *const f32).cast());
    enc.set_bytes(6, 4, (&num_v_heads as *const u32).cast());
    enc.dispatch_thread_groups(
        MTLSize::new(num_v_heads as NSUInteger, 1, 1),
        MTLSize::new(value_dim as NSUInteger, 1, 1),
    );
    enc.end_encoding();
}

// Avoid unused-import warning if Variant isn't referenced in helpers.
#[allow(dead_code)]
const _VARIANT_USE: Variant = VARIANT;
