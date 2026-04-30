//! Cogito-V2 / DeepSeek-V3 MoE forward — GPU experts + unscaled combine.
//!
//! Wraps the existing [`super::expert_forward::gpu_batched_experts_forward`]
//! with the noaux_tc routing + unconditional-shared-expert composition
//! the DeepSeek-V3 architecture uses. Mirrors
//! [`super::moe_cpu::deepseek_moe_cpu`] step-for-step but moves the
//! K-expert FFN dispatch and the residual combine onto Metal:
//!
//! 1. **CPU**: gate matvec (BF16 weights × `[hidden_dim]`) →
//!    `[num_experts]` logits. Cheap (~1 ms on M-series for 256×7168).
//! 2. **CPU**: [`super::moe_router::noaux_tc_router_cpu`] →
//!    `(indices[K], weights[K])`.
//! 3. **CPU**: shared-expert SwiGLU
//!    ([`super::mlp_cpu::shared_expert_swiglu_cpu`]) — separate FFN
//!    at `shared_intermediate` width.
//! 4. **CPU**: read K expert blobs from
//!    [`super::expert_io::ExpertFiles`] into a contiguous buffer.
//! 5. **GPU**: [`super::expert_forward::gpu_batched_experts_forward`]
//!    runs the K experts + the unscaled combine kernel selected at
//!    `expert_forward::combine_kernel_name()`.
//!
//! The combine kernel writes
//! `out = h_mid + Σ_k weights[k] * expert_out[k] + shared_out`. This
//! function passes **`h_mid = zeros`** so the returned `hidden_out`
//! matches `deepseek_moe_cpu`'s contract: the **post-MoE residual
//! contribution** that the caller adds to the post-attn residual,
//! not a fully-combined hidden state. The orchestrator (Phase 3) is
//! free to pass the actual `h_mid` once it threads the residual
//! through end-to-end.
//!
//! GPU routing (the `noaux_tc_router_gpu` kernel from the plan) is a
//! follow-up perf slice — CPU routing is sub-100µs per layer and lets
//! Phase 2 land before the kernel-design risk on top-K-of-256 with
//! tied tie-break.

use rayon::prelude::*;

use super::cpu_matvec::{bf16_matvec_cpu, CpuMatvecError};
use super::embedding::bf16_to_f32;
use super::expert_forward::{
    gpu_batched_experts_forward, ExpertForwardError, MoeBuffers,
};
use super::expert_io::{ExpertFiles, ExpertIoError};
use super::metal::MetalBackend;
use super::mlp_cpu::{shared_expert_swiglu_cpu, MlpForwardError};
use super::moe_router::{noaux_tc_router_cpu, MoeRouterError};
use super::variants::VARIANT;
use super::weight_file::WeightFile;

/// Errors specific to the GPU Cogito-V2 / DeepSeek-V3 MoE path.
#[derive(Debug, thiserror::Error)]
pub enum CogitoMoeGpuError {
    #[error("hidden length {got} != hidden_dim ({expected})")]
    HiddenLen { got: usize, expected: usize },
    #[error("hidden_out length {got} != hidden_dim ({expected})")]
    HiddenOutLen { got: usize, expected: usize },
    #[error("missing tensor '{name}'")]
    MissingTensor { name: String },
    #[error(
        "tensor '{name}' size {got} bytes, expected {expected} bytes"
    )]
    TensorSize {
        name: String,
        got: usize,
        expected: usize,
    },
    #[error("CPU matvec: {0}")]
    Matvec(#[from] CpuMatvecError),
    #[error("router: {0}")]
    Router(#[from] MoeRouterError),
    #[error("shared-expert MLP: {0}")]
    SharedMlp(#[from] MlpForwardError),
    #[error("expert I/O: {0}")]
    Io(#[from] ExpertIoError),
    #[error("GPU experts: {0}")]
    Experts(#[from] ExpertForwardError),
}

/// Per-token MoE composition for one DeepSeek-V3 / Cogito-V2 layer.
/// Computes
/// `out = Σ_k weights[k] * routed_expert(indices[k], hidden) +
/// shared_expert(hidden)`. Routing on CPU; expert FFNs + combine on
/// GPU.
///
/// `hidden` is the post-rms-norm input; `out` is the post-MoE
/// residual contribution to add back into the layer's residual stream.
/// Same contract as [`super::moe_cpu::deepseek_moe_cpu`] modulo Metal
/// reduction order.
pub fn cogito_moe_layer_forward_gpu(
    metal: &mut MetalBackend,
    bufs: &mut MoeBuffers,
    wf: &WeightFile,
    expert_files: &ExpertFiles,
    pool: &rayon::ThreadPool,
    layer_idx: usize,
    hidden: &[f32],
    out: &mut [f32],
) -> Result<(), CogitoMoeGpuError> {
    let v = VARIANT;
    let hidden_dim = v.hidden_dim;
    let num_experts = v.num_experts;
    let k = v.num_experts_per_tok;

    if hidden.len() != hidden_dim {
        return Err(CogitoMoeGpuError::HiddenLen {
            got: hidden.len(),
            expected: hidden_dim,
        });
    }
    if out.len() != hidden_dim {
        return Err(CogitoMoeGpuError::HiddenOutLen {
            got: out.len(),
            expected: hidden_dim,
        });
    }

    // ---- CPU: router-gate matvec + noaux_tc ----
    let gate_w_name = format!("model.layers.{layer_idx}.mlp.gate.weight");
    let bias_name = format!(
        "model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
    );

    let gate_w_bytes = wf
        .tensor_bytes(&gate_w_name)
        .ok_or_else(|| CogitoMoeGpuError::MissingTensor {
            name: gate_w_name.clone(),
        })?;
    let expected_gate_bytes = num_experts * hidden_dim * 2;
    if gate_w_bytes.len() != expected_gate_bytes {
        return Err(CogitoMoeGpuError::TensorSize {
            name: gate_w_name,
            got: gate_w_bytes.len(),
            expected: expected_gate_bytes,
        });
    }
    let gate_w = bytemuck_u16(gate_w_bytes);

    let mut gate_logits = vec![0f32; num_experts];
    bf16_matvec_cpu(
        gate_w,
        hidden_dim,
        num_experts,
        hidden,
        &mut gate_logits,
    )?;

    let bias_bytes = wf.tensor_bytes(&bias_name).ok_or_else(|| {
        CogitoMoeGpuError::MissingTensor {
            name: bias_name.clone(),
        }
    })?;
    let expected_bias_bytes = num_experts * 2;
    if bias_bytes.len() != expected_bias_bytes {
        return Err(CogitoMoeGpuError::TensorSize {
            name: bias_name,
            got: bias_bytes.len(),
            expected: expected_bias_bytes,
        });
    }
    let bias_u16 = bytemuck_u16(bias_bytes);
    let bias_f32: Vec<f32> =
        bias_u16.iter().map(|&b| bf16_to_f32(b)).collect();

    let mut indices = vec![0i32; k];
    let mut weights = vec![0f32; k];
    noaux_tc_router_cpu(
        &mut gate_logits,
        &bias_f32,
        v.n_group,
        v.topk_group,
        k,
        v.routed_scaling_factor,
        &mut indices,
        &mut weights,
    )?;

    // ---- CPU: shared-expert SwiGLU ----
    let mut shared_out = vec![0f32; hidden_dim];
    shared_expert_swiglu_cpu(wf, layer_idx, hidden, &mut shared_out)?;

    // ---- CPU: read K expert blobs into one contiguous Vec ----
    // Parallelize across the io_pool — 8 P-cores can issue 8 pread
    // calls concurrently, dropping K-serial wall-time from
    // K × per_blob to ~max(per_blob). On Cogito's 8 × 24 MB blobs
    // that's ~7-8× the SSD bandwidth utilization vs serial.
    let expert_size = v.expert_size_4bit();
    let mut expert_data = vec![0u8; k * expert_size];
    pool.install(|| -> Result<(), ExpertIoError> {
        expert_data
            .par_chunks_mut(expert_size)
            .zip(indices.par_iter().take(k))
            .try_for_each(|(chunk, &e)| {
                expert_files.read_expert(layer_idx, e as usize, chunk)
            })
    })?;

    // ---- GPU: K-expert FFNs + unscaled combine ----
    // h_mid = zeros so the kernel's `h_mid + moe + shared` produces
    // exactly the residual contribution. shared_gate_score is bound
    // for the kernel signature but the unscaled kernel ignores it.
    let h_mid_zeros = vec![0f32; hidden_dim];
    let _ = pool; // io_pool is consumed by the parallel reads above.
    gpu_batched_experts_forward(
        metal,
        bufs,
        k as i32,
        &expert_data,
        /* h_post = */ hidden,
        /* h_mid  = */ &h_mid_zeros,
        /* shared_out = */ &shared_out,
        /* expert_weights = */ &weights,
        /* shared_gate_score = */ 0.0,
        /* hidden_out = */ out,
    )?;
    Ok(())
}

/// Reinterpret a `&[u8]` as `&[u16]` via `align_to`. The MLX
/// converter aligns BF16 tensors to 2-byte boundaries; the head/tail
/// must therefore be empty. Panics if not — that's a converter bug
/// we want loud, not soft. Same logic as `moe_cpu::bytemuck_u16` —
/// duplicated to keep the modules independent.
fn bytemuck_u16(bytes: &[u8]) -> &[u16] {
    // SAFETY: align_to is safe by definition.
    let (head, body, tail) = unsafe { bytes.align_to::<u16>() };
    assert!(
        head.is_empty() && tail.is_empty(),
        "BF16 tensor not aligned to 2-byte boundary (head={}, tail={})",
        head.len(),
        tail.len()
    );
    body
}
