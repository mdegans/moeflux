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

use metal::{Buffer, Device, MTLResourceOptions, NSUInteger};
use rayon::prelude::*;

use super::dense_mlp_gpu::{
    encode_swiglu_ffn_layer_forward_gpu, DenseMlpGpuError, DenseMlpPipelines,
};
use super::embedding::bf16_to_f32;
use super::expert_forward::{
    gpu_batched_experts_encode_pre_staged, ExpertForwardError, MoeBuffers,
    MAX_K,
};
use super::expert_io::{ExpertFiles, ExpertIoError};
use super::gpu_matvec::{encode_bf16_matvec, BfMatvecPipelines};
use super::metal::MetalBackend;
use super::moe_router::{noaux_tc_router_cpu, MoeRouterError};
use super::mtl_weight_buf::MtlWeightBuf;
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
    #[error("router: {0}")]
    Router(#[from] MoeRouterError),
    #[error("shared-expert FFN (GPU): {0}")]
    SharedFfn(#[from] DenseMlpGpuError),
    #[error("expert I/O: {0}")]
    Io(#[from] ExpertIoError),
    #[error("GPU experts: {0}")]
    Experts(#[from] ExpertForwardError),
}

/// Per-token GPU scratch for the Cogito-V2 / DeepSeek-V3 shared-expert
/// SwiGLU FFN. One set is reused across every MoE layer (the shared-
/// expert FFN runs unconditionally per layer at the same width). At
/// `shared_intermediate=2048` for Cogito-V2 the total cost is ~24 KB of
/// shared-storage Metal buffers per token.
///
/// The shared-expert input reuses `MoeBuffers.input` (already populated
/// with the post-attn-norm hidden vector by the dispatcher); the
/// shared-expert output writes into `MoeBuffers.shared_out` so the
/// downstream `gpu_batched_experts_encode_pre_staged` reads it
/// directly via `bufs.shared_out.raw()` without a host roundtrip.
pub struct SharedExpertBuffers {
    pub gate_out: Buffer,
    pub up_out: Buffer,
    pub act: Buffer,
}

impl SharedExpertBuffers {
    pub fn new(device: &Device) -> Self {
        let v = VARIANT;
        let f32_buf = |n: usize| {
            let b = device.new_buffer(
                (n * std::mem::size_of::<f32>()) as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            );
            // SAFETY: shared storage, no GPU work in flight on a fresh buffer.
            unsafe {
                std::ptr::write_bytes(
                    b.contents() as *mut u8,
                    0,
                    n * std::mem::size_of::<f32>(),
                );
            }
            b
        };
        Self {
            gate_out: f32_buf(v.shared_intermediate),
            up_out: f32_buf(v.shared_intermediate),
            act: f32_buf(v.shared_intermediate),
        }
    }
}

/// Per-token MoE composition for one DeepSeek-V3 / Cogito-V2 layer.
/// Computes
/// `out = Σ_k weights[k] * routed_expert(indices[k], hidden) +
/// shared_expert(hidden)`. Routing on CPU; expert FFNs + shared expert
/// + combine all on GPU. The shared-expert SwiGLU dispatch and the
/// K-routed-expert dispatches are issued on the same Metal command
/// queue (FIFO ordering) so the experts' `bufs.shared_out` read sees
/// the populated tensor without an explicit wait between cmdbufs.
///
/// `hidden` is the post-rms-norm input; `out` is the post-MoE
/// residual contribution to add back into the layer's residual stream.
/// Same contract as [`super::moe_cpu::deepseek_moe_cpu`] modulo Metal
/// reduction order.
#[allow(clippy::too_many_arguments)]
pub fn cogito_moe_layer_forward_gpu(
    metal: &mut MetalBackend,
    bufs: &mut MoeBuffers,
    shared_bufs: &SharedExpertBuffers,
    dense_pipes: &DenseMlpPipelines,
    bf_pipes: &BfMatvecPipelines,
    wf: &WeightFile,
    wf_buf: &MtlWeightBuf,
    expert_files: &ExpertFiles,
    pool: &rayon::ThreadPool,
    layer_idx: usize,
    hidden: &[f32],
    out: &mut [f32],
) -> Result<(), CogitoMoeGpuError> {
    let v = VARIANT;
    let hidden_dim = v.hidden_dim;

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

    // Stage hidden into bufs.input now so both the GPU gate matvec and
    // the GPU shared expert read it directly from bufs.input.
    bufs.stage_host_input(hidden);
    let input_clone: Buffer = bufs.input_buffer().clone();
    cogito_moe_layer_forward_gpu_inner(
        metal,
        bufs,
        shared_bufs,
        dense_pipes,
        bf_pipes,
        wf,
        wf_buf,
        expert_files,
        pool,
        layer_idx,
        input_clone,
    )?;
    out.copy_from_slice(&bufs.moe_hidden_to_vec());
    Ok(())
}

/// Buffer-IO sibling of [`cogito_moe_layer_forward_gpu`]. Reads input
/// directly from `input_buf` (a caller-owned Metal buffer holding the
/// post-rms-norm hidden vector) and leaves the result in
/// `bufs.moe_hidden` on completion — the caller reads it via a
/// subsequent GPU dispatch (e.g. `encode_residual_add_into`) without a
/// CPU bounce. Phase 5 entry point for the GPU residual stream.
#[allow(clippy::too_many_arguments)]
pub fn cogito_moe_layer_forward_gpu_buf_io(
    metal: &mut MetalBackend,
    bufs: &mut MoeBuffers,
    shared_bufs: &SharedExpertBuffers,
    dense_pipes: &DenseMlpPipelines,
    bf_pipes: &BfMatvecPipelines,
    wf: &WeightFile,
    wf_buf: &MtlWeightBuf,
    expert_files: &ExpertFiles,
    pool: &rayon::ThreadPool,
    layer_idx: usize,
    input_buf: &Buffer,
) -> Result<(), CogitoMoeGpuError> {
    cogito_moe_layer_forward_gpu_inner(
        metal,
        bufs,
        shared_bufs,
        dense_pipes,
        bf_pipes,
        wf,
        wf_buf,
        expert_files,
        pool,
        layer_idx,
        input_buf.clone(),
    )
}

#[allow(clippy::too_many_arguments)]
fn cogito_moe_layer_forward_gpu_inner(
    metal: &mut MetalBackend,
    bufs: &mut MoeBuffers,
    shared_bufs: &SharedExpertBuffers,
    dense_pipes: &DenseMlpPipelines,
    bf_pipes: &BfMatvecPipelines,
    wf: &WeightFile,
    wf_buf: &MtlWeightBuf,
    expert_files: &ExpertFiles,
    pool: &rayon::ThreadPool,
    layer_idx: usize,
    input_buf: Buffer,
) -> Result<(), CogitoMoeGpuError> {
    let v = VARIANT;
    let hidden_dim = v.hidden_dim;
    let num_experts = v.num_experts;
    let k = v.num_experts_per_tok;

    // ---- GPU: router-gate matvec → bufs.gate_logits ----
    // Mirrors the C path's per-layer router gate (`model.layers.{i}
    // .mlp.gate.weight`, [num_experts=256, hidden_dim=7168] BF16).
    // Issued on its own cmdbuf because we need the result host-side
    // for `noaux_tc_router_cpu` before continuing.
    let gate_w_name = format!("model.layers.{layer_idx}.mlp.gate.weight");
    let bias_name = format!(
        "model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
    );
    let gate_w_off = wf_buf
        .tensor_offset(wf, &gate_w_name)
        .map_err(|e| CogitoMoeGpuError::SharedFfn(e.into()))?
        .ok_or_else(|| CogitoMoeGpuError::MissingTensor {
            name: gate_w_name.clone(),
        })?;
    {
        let cmdbuf_gate = metal.queue().new_command_buffer();
        encode_bf16_matvec(
            cmdbuf_gate,
            bf_pipes,
            wf_buf.buffer(),
            gate_w_off,
            &input_buf,
            bufs.gate_logits_buffer(),
            hidden_dim as u32,
            num_experts as u32,
        );
        cmdbuf_gate.commit();
        cmdbuf_gate.wait_until_completed();
    }
    let mut gate_logits = bufs.gate_logits_to_vec();

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

    // ---- Stage h_mid as zeros for the unscaled combine ----
    // bufs.input was already staged above for the GPU gate matvec; the
    // shared-expert and routed-expert dispatches read the same buffer.
    bufs.stage_host_h_mid_zero();

    // ---- CPU: read K expert blobs into bufs.data_synced (parallel) ----
    // Slice 5d-6 / 5d-9 path via rayon. `data_set_per_slot` is set to
    // Synced uniformly below — we don't use the prefetch ring on the
    // Cogito path yet (Phase 4b lands it).
    //
    // Hoist the disjoint-slice array outside the rayon closure so the
    // closure only captures `[&mut [u8]; MAX_K]` (Send), not
    // `&mut MoeBuffers`. Same trick the linear-attn path uses at
    // `linear_attn_forward.rs:1106-1129`.
    let mut dsts = bufs.data_synced_slots_mut_array();
    pool.install(|| -> Result<(), ExpertIoError> {
        dsts[..k]
            .par_iter_mut()
            .zip(indices.par_iter().take(k))
            .try_for_each(|(slot, &e)| {
                expert_files.read_expert(layer_idx, e as usize, slot)
            })
    })?;

    // ---- GPU: shared expert SwiGLU into bufs.shared_out ----
    // Encode + commit on the same command queue as the experts. Metal
    // command queues are FIFO: the pre-staged experts cmdbuf below
    // sees the shared-expert dispatch's output without an explicit
    // wait between cmdbufs.
    {
        let cmdbuf_shared = metal.queue().new_command_buffer();
        let prefix =
            format!("model.layers.{layer_idx}.mlp.shared_experts");
        encode_swiglu_ffn_layer_forward_gpu(
            cmdbuf_shared,
            dense_pipes,
            wf,
            wf_buf,
            &prefix,
            v.shared_intermediate as u32,
            &input_buf,
            &shared_bufs.gate_out,
            &shared_bufs.up_out,
            &shared_bufs.act,
            bufs.shared_out_buffer(),
        )?;
        cmdbuf_shared.commit();
    }

    // ---- GPU: K-expert FFNs + unscaled combine ----
    // h_mid was staged to zeros above, so the combine kernel's
    // `h_mid + moe + shared` produces exactly the residual
    // contribution. shared_gate_score is bound for the kernel
    // signature but the unscaled kernel ignores it.
    let data_set_per_slot: [super::SlotSource; MAX_K] =
        [super::SlotSource::Synced; MAX_K];
    // Borrow split: clone the &Buffer references (Metal buffers are
    // Arc-like, clone is cheap retain) so the &mut bufs borrow into
    // the pre-staged encoder doesn't conflict with the input refs.
    // Pass the caller's input_buf directly (replaces bufs.input clone).
    let h_mid_clone: Buffer = bufs.h_mid_buffer().clone();
    let shared_clone: Buffer = bufs.shared_out_buffer().clone();
    let cmdbuf = gpu_batched_experts_encode_pre_staged(
        metal,
        bufs,
        k as i32,
        &input_buf,
        &h_mid_clone,
        &shared_clone,
        &weights,
        /* shared_gate_score = */ 0.0,
        &data_set_per_slot,
        /* prefetch_set = */ 0,
        /* chain = */ None,
    )?;
    cmdbuf.commit();
    cmdbuf.wait_until_completed();
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
