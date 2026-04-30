//! GPU dense SwiGLU MLP for layers `[0, first_k_dense_replace)`.
//!
//! Mirrors [`super::mlp_cpu::dense_mlp_swiglu_cpu`] step-for-step on
//! Metal:
//!
//! 1. `gate_out = gate_proj @ hidden`            (4-bit matvec)
//! 2. `up_out   = up_proj   @ hidden`            (4-bit matvec)
//! 3. `act      = silu(gate_out) * up_out`       (`swiglu_fused` kernel)
//! 4. `out      = down_proj @ act`               (4-bit matvec)
//!
//! Same kernels the existing shared-expert FFN uses
//! (`linear_attn_forward.rs:1004-1035`); the only difference is the
//! intermediate width — `dense_intermediate` (18432 for Cogito-V2) vs
//! `shared_intermediate` (2048). Existing scratch buffers in
//! [`super::linear_attn_forward::LayerForwardBuffers`] are sized to
//! `shared_intermediate`, so the caller must supply scratch sized to
//! `dense_intermediate` here. The Phase 3 orchestrator threads new
//! `dense_*` buffers through; the standalone test allocates them ad-hoc.

use metal::{
    Buffer, CommandBufferRef, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize, NSUInteger,
};

use super::gpu_matvec::{encode_matvec, MatvecPipelines, MatvecSpec};
use super::metal::{MetalBackend, MetalError};
use super::mtl_weight_buf::{MtlWeightBuf, MtlWeightBufError};
use super::variants::VARIANT;
use super::weight_file::WeightFile;

/// Per-token GPU scratch for the dense MLP path. One set is reused
/// across the (small number of) dense layers — `first_k_dense_replace`
/// is 3 for Cogito-V2 — so layer N+1 doesn't need its own copy.
/// ~290 KB total at f32 (gate/up/act at dense_intermediate=18432 + two
/// hidden_dim buffers).
pub struct DenseMlpBuffers {
    /// Caller-staged input (`[hidden_dim]` floats).
    pub hidden_in: Buffer,
    pub gate_out: Buffer,
    pub up_out: Buffer,
    pub act: Buffer,
    /// Post-down-proj output (`[hidden_dim]` floats).
    pub out: Buffer,
}

impl DenseMlpBuffers {
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
            hidden_in: f32_buf(v.hidden_dim),
            gate_out: f32_buf(v.dense_intermediate),
            up_out: f32_buf(v.dense_intermediate),
            act: f32_buf(v.dense_intermediate),
            out: f32_buf(v.hidden_dim),
        }
    }
}

/// Synchronous wrapper around [`encode_dense_mlp_layer_forward_gpu`]
/// that stages `hidden` → `bufs.hidden_in`, runs the dispatch,
/// commits + waits, and reads back `bufs.out` → `out`. Mirrors the
/// shape of [`super::expert_forward::gpu_batched_experts_forward`] for
/// host-slice consumers.
pub fn dense_mlp_layer_forward_gpu(
    metal: &mut MetalBackend,
    pipes: &DenseMlpPipelines,
    bufs: &mut DenseMlpBuffers,
    wf: &WeightFile,
    wf_buf: &MtlWeightBuf,
    layer_idx: usize,
    hidden: &[f32],
    out: &mut [f32],
) -> Result<(), DenseMlpGpuError> {
    let v = VARIANT;
    if hidden.len() != v.hidden_dim {
        return Err(DenseMlpGpuError::HiddenLen {
            got: hidden.len(),
            expected: v.hidden_dim,
        });
    }
    if out.len() != v.hidden_dim {
        return Err(DenseMlpGpuError::OutLen {
            got: out.len(),
            expected: v.hidden_dim,
        });
    }

    // SAFETY: shared-storage; no GPU work in flight on this buffer set
    // (caller obeys the synchronous contract — we commit + wait below).
    unsafe {
        std::ptr::copy_nonoverlapping(
            hidden.as_ptr(),
            bufs.hidden_in.contents() as *mut f32,
            v.hidden_dim,
        );
    }

    let cmdbuf = metal.queue().new_command_buffer();
    encode_dense_mlp_layer_forward_gpu(
        cmdbuf,
        pipes,
        wf,
        wf_buf,
        layer_idx,
        &bufs.hidden_in,
        &bufs.gate_out,
        &bufs.up_out,
        &bufs.act,
        &bufs.out,
    )?;
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    // SAFETY: cmdbuf completed.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bufs.out.contents() as *const f32,
            out.as_mut_ptr(),
            v.hidden_dim,
        );
    }
    Ok(())
}

/// Errors from [`encode_dense_mlp_layer_forward_gpu`].
#[derive(Debug, thiserror::Error)]
pub enum DenseMlpGpuError {
    #[error("hidden length {got} != hidden_dim ({expected})")]
    HiddenLen { got: usize, expected: usize },
    #[error("out length {got} != hidden_dim ({expected})")]
    OutLen { got: usize, expected: usize },
    #[error("missing tensor '{name}'")]
    MissingTensor { name: String },
    #[error("Metal: {0}")]
    Metal(#[from] MetalError),
    #[error("weight-buffer offset: {0}")]
    Offset(#[from] MtlWeightBufError),
    #[error(
        "variant has dense_intermediate={got}; this build's variant has \
         no dense MLP (first_k_dense_replace=0). Don't dispatch dense \
         layers."
    )]
    NoDenseMlp { got: usize },
}

/// Pipelines for the dense MLP path. Both kernels are dim-parametric
/// — same PSOs the shared-expert and routed-expert FFNs use.
pub struct DenseMlpPipelines {
    pub matvec: MatvecPipelines,
    pub swiglu: ComputePipelineState,
}

impl DenseMlpPipelines {
    pub fn fetch(metal: &mut MetalBackend) -> Result<Self, MetalError> {
        Ok(Self {
            matvec: MatvecPipelines::fetch(metal)?,
            swiglu: metal.pipeline("swiglu_fused")?.clone(),
        })
    }
}

/// Encode one dense-MLP layer forward into `cmdbuf`. Synchronous: all
/// dispatches encode in order; the caller commits + waits (or chains
/// the cmdbuf in Phase 4b's deferred-ring path).
///
/// Buffer contract:
/// - `hidden`: post-rms-norm input, `[hidden_dim]` f32, shared.
/// - `gate_out`, `up_out`, `act`: scratch, `[dense_intermediate]` f32
///   each. May not alias each other.
/// - `out`: post-down-proj output, `[hidden_dim]` f32. May not alias
///   `hidden`.
///
/// Reads tensors `model.layers.{layer_idx}.mlp.{gate_proj,up_proj,
/// down_proj}` from `wf_buf`; layer_idx must be `< first_k_dense_replace`.
pub fn encode_dense_mlp_layer_forward_gpu(
    cmdbuf: &CommandBufferRef,
    pipes: &DenseMlpPipelines,
    wf: &WeightFile,
    wf_buf: &MtlWeightBuf,
    layer_idx: usize,
    hidden: &Buffer,
    gate_out: &Buffer,
    up_out: &Buffer,
    act: &Buffer,
    out: &Buffer,
) -> Result<(), DenseMlpGpuError> {
    let v = VARIANT;
    let hidden_dim = v.hidden_dim as u32;
    let intermediate = v.dense_intermediate as u32;
    if intermediate == 0 {
        return Err(DenseMlpGpuError::NoDenseMlp {
            got: v.dense_intermediate,
        });
    }

    let resolve_proj =
        |name: &str| -> Result<(u64, u64, u64), DenseMlpGpuError> {
            let w = format!("{name}.weight");
            let s = format!("{name}.scales");
            let b = format!("{name}.biases");
            let w_off = wf_buf
                .tensor_offset(wf, &w)?
                .ok_or(DenseMlpGpuError::MissingTensor { name: w })?;
            let s_off = wf_buf
                .tensor_offset(wf, &s)?
                .ok_or(DenseMlpGpuError::MissingTensor { name: s })?;
            let b_off = wf_buf
                .tensor_offset(wf, &b)?
                .ok_or(DenseMlpGpuError::MissingTensor { name: b })?;
            Ok((w_off, s_off, b_off))
        };

    let prefix = format!("model.layers.{layer_idx}.mlp");
    let gate_off = resolve_proj(&format!("{prefix}.gate_proj"))?;
    let up_off = resolve_proj(&format!("{prefix}.up_proj"))?;
    let down_off = resolve_proj(&format!("{prefix}.down_proj"))?;

    // gate_out = gate_proj @ hidden
    encode_matvec(
        cmdbuf,
        &pipes.matvec,
        wf_buf,
        &MatvecSpec {
            w_off: gate_off.0,
            s_off: gate_off.1,
            b_off: gate_off.2,
            input: hidden,
            output: gate_out,
            out_dim: intermediate,
            in_dim: hidden_dim,
            bits: 4,
        },
    );
    // up_out = up_proj @ hidden
    encode_matvec(
        cmdbuf,
        &pipes.matvec,
        wf_buf,
        &MatvecSpec {
            w_off: up_off.0,
            s_off: up_off.1,
            b_off: up_off.2,
            input: hidden,
            output: up_out,
            out_dim: intermediate,
            in_dim: hidden_dim,
            bits: 4,
        },
    );
    // act = silu(gate_out) * up_out
    encode_swiglu(cmdbuf, &pipes.swiglu, gate_out, up_out, act, intermediate);
    // out = down_proj @ act
    encode_matvec(
        cmdbuf,
        &pipes.matvec,
        wf_buf,
        &MatvecSpec {
            w_off: down_off.0,
            s_off: down_off.1,
            b_off: down_off.2,
            input: act,
            output: out,
            out_dim: hidden_dim,
            in_dim: intermediate,
            bits: 4,
        },
    );
    Ok(())
}

/// One `swiglu_fused` dispatch. 256 threads per threadgroup; same
/// shape as `linear_attn_forward::encode_swiglu_buf`.
fn encode_swiglu(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    gate: &Buffer,
    up: &Buffer,
    act: &Buffer,
    dim: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(gate), 0);
    enc.set_buffer(1, Some(up), 0);
    enc.set_buffer(2, Some(act), 0);
    enc.set_bytes(3, 4, (&dim as *const u32).cast());
    let num_tgs = (dim + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}
