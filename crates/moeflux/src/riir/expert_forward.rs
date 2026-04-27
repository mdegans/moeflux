//! Single-expert GPU FFN forward pass — slice 9a.
//!
//! Mirrors `gpu_expert_forward` in `metal_infer/infer.m`: four GPU
//! dispatches encoded into one command buffer, then a CPU read-back.
//!
//! 1. `dequant_matvec_4bit_v3` over `gate` → `gate_out` `[MOE_INTERMEDIATE]`
//! 2. `dequant_matvec_4bit_v3` over `up`   → `up_out`   `[MOE_INTERMEDIATE]`
//! 3. `swiglu_fused(gate_out, up_out)`     → `act`      `[MOE_INTERMEDIATE]`
//! 4. `dequant_matvec_4bit_v3` over `down` → `expert_out` `[HIDDEN_DIM]`
//!
//! All transient buffers are allocated fresh per call. The C path
//! reuses pre-allocated buffers on the model context; persistent
//! per-`RsCtx` buffers come in slice 9b alongside the batched K-expert
//! dispatch where reuse actually matters.
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

use metal::{MTLSize, NSUInteger};

use super::metal::{MetalBackend, MetalError, MtlBuffer};
use super::variants::{Variant, GROUP_SIZE, VARIANT};

/// Errors from single-expert GPU FFN forward.
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
}
