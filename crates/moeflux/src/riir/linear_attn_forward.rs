//! End-to-end linear-attention layer forward — Phase 4c.
//!
//! Composes:
//!
//! 1. **Pre-attn input norm** (`rms_norm_sum_sq` + `rms_norm_apply_bf16`)
//! 2. **4 batched projection matvecs** (qkv → 12288, z → 8192, beta → 64,
//!    alpha → 64) — `dequant_matvec_4bit_v3` against `wf_buf` at offsets
//! 3. **5 linear-attn fused kernels**: conv1d_step → rms_norm_qk →
//!    compute_decay_beta → gated_delta_net_step → gated_rms_norm
//! 4. **o_proj** (matvec from `value_dim * num_v_heads` → HIDDEN_DIM)
//! 5. **Residual add + post-attn RMSNorm**
//! 6. **MoE router** (gate matvec → CPU softmax/topK/normalize) +
//!    shared-expert gate score matvec
//! 7. **Shared expert FFN** (gate / up / SwiGLU / down)
//! 8. **K-expert MoE dispatch + combine** via the 9b path
//!    (`gpu_batched_experts_forward`)
//!
//! Output: post-combine HIDDEN_DIM hidden state in `buffers.input`.
//!
//! Mirrors `fused_layer_forward`'s `!is_full` GPU production path
//! (infer.m:4253–~5085), minus the `prev_gpu_combined` fast path and
//! the deferred-experts state machine — for the dump-hook diff this
//! runs synchronously: each command buffer is committed and waited
//! before the next.
//!
//! ## Why one big function
//!
//! Per the plan: the compose is a straight sequence of encode-and-go
//! steps. Splitting it across helpers obscures the data flow without
//! reducing complexity. The function is long but linear; comment
//! markers (`// ── step N: …`) make it scannable.

use metal::{
    Buffer, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    NSUInteger,
};

use super::expert_forward::{gpu_batched_experts_forward, MoeBuffers};
use super::rms_norm::rms_norm_cpu;
use super::gpu_linear_attn::{
    encode_compute_decay_beta, encode_conv1d_step, encode_delta_net_step,
    encode_gated_rms_norm, encode_rms_norm_qk, LinearAttnPipelines,
};
use super::gpu_matvec::{encode_matvec, MatvecPipelines, MatvecSpec};
use super::layer_weight_cache::LayerWeightCache;
use super::metal::{MetalBackend, MetalError};
use super::moe_router::moe_router_cpu;
use super::mtl_weight_buf::MtlWeightBuf;
use super::state::LinearAttnState;
use super::variants::{Variant, RMS_NORM_EPS, VARIANT};
use super::weight_file::WeightFile;

/// Errors that can surface during the linear-attn layer forward.
#[derive(Debug, thiserror::Error)]
pub enum LinearAttnForwardError {
    #[error("missing tensor for layer {layer}: {tensor}")]
    MissingTensor {
        layer: usize,
        tensor: &'static str,
    },
    #[error("hidden_in must be HIDDEN_DIM={expected} floats, got {actual}")]
    BadHiddenLen { expected: usize, actual: usize },
    #[error("Metal: {0}")]
    Metal(#[from] MetalError),
    #[error("MoE router: {0}")]
    Router(#[from] super::moe_router::MoeRouterError),
    #[error("expert FFN: {0}")]
    Expert(#[from] super::expert_forward::ExpertForwardError),
    #[error("expert I/O: {0}")]
    ExpertIo(#[from] super::expert_io::ExpertIoError),
}

/// Persistent GPU scratch + recurrence-state buffers needed by the
/// linear-attention layer forward. Allocated once per [`crate::riir::RsCtx`].
pub struct LinearAttnBuffers {
    pub input: Buffer,
    pub normed: Buffer,
    pub residual: Buffer,
    pub h_mid: Buffer,
    pub output: Buffer,
    /// 7 batch-output slots. Sized per slot:
    /// - [0]: LINEAR_CONV_DIM (qkv)
    /// - [1]: LINEAR_TOTAL_VALUE (z)
    /// - [2]: LINEAR_NUM_V_HEADS (beta)
    /// - [3]: LINEAR_NUM_V_HEADS (alpha)
    /// - [4]: NUM_EXPERTS (router gate logits)
    /// - [5]: 1 (shared-expert gate scalar)
    /// - [6]: LINEAR_TOTAL_VALUE (gated-norm output / o_proj input)
    pub batch_out: [Buffer; 7],
    /// Per-linear-layer recurrence state.
    pub conv_state: Vec<Buffer>,
    pub delta_state: Vec<Buffer>,
    /// Scratch for one layer's linear-attn pipeline (reused across layers).
    pub conv_output: Buffer,
    pub delta_g_decay: Buffer,
    pub delta_beta: Buffer,
    pub delta_output: Buffer,
    pub sum_sq: Buffer,
    /// Shared-expert intermediate (SHARED_INTERMEDIATE floats).
    pub shared_gate_out: Buffer,
    pub shared_up_out: Buffer,
    pub shared_act: Buffer,
    pub shared_out: Buffer,
}

impl LinearAttnBuffers {
    pub fn new(device: &Device) -> Self {
        let v = VARIANT;
        // Allocate + zero. `device.new_buffer` doesn't guarantee
        // zeroed storage; the recurrence buffers (`conv_state`,
        // `delta_state`) read from this region on first call so
        // garbage initial values would diverge from the C path's
        // metal_setup (which calloc-zeros the same buffers).
        let f32_buf = |n: usize| {
            let b = device.new_buffer(
                (n * std::mem::size_of::<f32>()) as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            );
            // SAFETY: shared storage; no GPU work in flight on a
            // freshly allocated buffer.
            unsafe {
                std::ptr::write_bytes(
                    b.contents() as *mut u8,
                    0,
                    n * std::mem::size_of::<f32>(),
                );
            }
            b
        };
        let batch_sizes = [
            v.linear_conv_dim(),
            v.linear_total_value(),
            v.linear_num_v_heads,
            v.linear_num_v_heads,
            v.num_experts,
            1,
            v.linear_total_value(),
        ];
        let batch_out: [Buffer; 7] = std::array::from_fn(|i| f32_buf(batch_sizes[i]));

        let num_linear =
            v.num_layers - num_full_attn_layers(&v);
        let conv_state = (0..num_linear)
            .map(|_| f32_buf((Variant::CONV_KERNEL_SIZE - 1) * v.linear_conv_dim()))
            .collect();
        let delta_state = (0..num_linear)
            .map(|_| {
                f32_buf(
                    v.linear_num_v_heads * Variant::LINEAR_VALUE_DIM
                        * Variant::LINEAR_KEY_DIM,
                )
            })
            .collect();

        Self {
            input: f32_buf(v.hidden_dim),
            normed: f32_buf(v.hidden_dim),
            residual: f32_buf(v.hidden_dim),
            h_mid: f32_buf(v.hidden_dim),
            output: f32_buf(v.hidden_dim),
            batch_out,
            conv_state,
            delta_state,
            conv_output: f32_buf(v.linear_conv_dim()),
            delta_g_decay: f32_buf(v.linear_num_v_heads),
            delta_beta: f32_buf(v.linear_num_v_heads),
            delta_output: f32_buf(v.linear_total_value()),
            sum_sq: f32_buf(1),
            shared_gate_out: f32_buf(v.shared_intermediate),
            shared_up_out: f32_buf(v.shared_intermediate),
            shared_act: f32_buf(v.shared_intermediate),
            shared_out: f32_buf(v.hidden_dim),
        }
    }

    /// Reset every per-layer state buffer to zero. Called from
    /// `RsCtx::memory_clear` to clear the recurrence in addition to
    /// the host-side state vector (which today the GPU doesn't read,
    /// so this is the source of truth on the GPU side).
    pub fn reset_recurrence(&mut self) {
        for b in &self.conv_state {
            zero_f32_buffer(b);
        }
        for b in &self.delta_state {
            zero_f32_buffer(b);
        }
    }
}

fn zero_f32_buffer(b: &Buffer) {
    let bytes = b.length() as usize;
    // SAFETY: Shared-storage buffer; no GPU work is in flight when
    // memory_clear is called (caller invariant).
    unsafe {
        std::ptr::write_bytes(b.contents() as *mut u8, 0, bytes);
    }
}

/// `linear_layer_idx = layer_idx - (layer_idx + 1) / FULL_ATTN_INTERVAL`.
/// Returns `None` if `layer_idx` is a full-attention layer.
pub fn linear_layer_idx_for(layer_idx: usize) -> Option<usize> {
    if (layer_idx + 1) % VARIANT.full_attn_interval == 0 {
        None
    } else {
        Some(layer_idx - (layer_idx + 1) / VARIANT.full_attn_interval)
    }
}

fn num_full_attn_layers(v: &Variant) -> usize {
    v.num_layers / v.full_attn_interval
}

/// Run one linear-attention layer's forward pass on the GPU.
///
/// Pre: `buffers.input` holds the input hidden state (HIDDEN_DIM
/// floats). Post: `buffers.input` holds the output hidden state.
/// The targeted layer's `conv_state` / `delta_state` are mutated in
/// place. `state` is the host-side mirror used for `memory_*` ops;
/// for 4c we keep it in lockstep with the GPU buffers via reset
/// only — partial truncation will resync GPU buffers via
/// `reset_recurrence` (a faithful port of the lossy semantic).
#[allow(clippy::too_many_arguments)]
pub fn linear_attn_layer_forward(
    metal: &mut MetalBackend,
    wf: &WeightFile,
    wf_buf: &MtlWeightBuf,
    layer_cache: &LayerWeightCache,
    buffers: &mut LinearAttnBuffers,
    moe: &mut MoeBuffers,
    layer_idx: usize,
    k_active: usize,
    expert_files: &super::expert_io::ExpertFiles,
    _layer_state: &mut LinearAttnState,
) -> Result<(), LinearAttnForwardError> {
    let v = VARIANT;
    let linear_layer_idx = linear_layer_idx_for(layer_idx).ok_or_else(|| {
        LinearAttnForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "linear_layer_idx (called on full-attn layer)",
        }
    })?;

    // Per-tensor bit width lookup. 4-bit is the default; A3B uses
    // 8-bit for `mlp.gate.weight` and `mlp.shared_expert_gate.weight`.
    let bits_of = |name: &str| -> u32 {
        wf.tensor_info(name).map(|i| i.bits as u32).unwrap_or(4).max(4)
    };
    let qkv_bits = bits_of(&format!(
        "model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight"
    ));
    let z_bits = bits_of(&format!(
        "model.layers.{layer_idx}.linear_attn.in_proj_z.weight"
    ));
    let alpha_bits = bits_of(&format!(
        "model.layers.{layer_idx}.linear_attn.in_proj_a.weight"
    ));
    let beta_bits = bits_of(&format!(
        "model.layers.{layer_idx}.linear_attn.in_proj_b.weight"
    ));
    let o_bits = bits_of(&format!(
        "model.layers.{layer_idx}.linear_attn.out_proj.weight"
    ));
    let gate_bits =
        bits_of(&format!("model.layers.{layer_idx}.mlp.gate.weight"));
    let seg_bits = bits_of(&format!(
        "model.layers.{layer_idx}.mlp.shared_expert_gate.weight"
    ));
    let s_gate_bits = bits_of(&format!(
        "model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight"
    ));
    let s_up_bits = bits_of(&format!(
        "model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight"
    ));
    let s_down_bits = bits_of(&format!(
        "model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight"
    ));

    // Resolve required offsets up front so we error early if anything
    // is missing.
    // input_layernorm.weight is read indirectly via `rms_norm_cpu`
    // below, but require it up front for fail-fast behavior.
    let _input_norm_w = layer_cache.input_layernorm_w.ok_or(
        LinearAttnForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "input_layernorm.weight",
        },
    )?;
    let post_attn_norm_w = layer_cache.post_attention_layernorm_w.ok_or(
        LinearAttnForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "post_attention_layernorm.weight",
        },
    )?;
    let qkv_w = require(layer_cache.qkv_w, layer_idx, "qkv_proj.weight")?;
    let qkv_s = require(layer_cache.qkv_s, layer_idx, "qkv_proj.scales")?;
    let qkv_b = require(layer_cache.qkv_b, layer_idx, "qkv_proj.biases")?;
    let z_w = require(layer_cache.z_w, layer_idx, "z_proj.weight")?;
    let z_s = require(layer_cache.z_s, layer_idx, "z_proj.scales")?;
    let z_b = require(layer_cache.z_b, layer_idx, "z_proj.biases")?;
    let beta_w = require(layer_cache.beta_w, layer_idx, "beta_proj.weight")?;
    let beta_s = require(layer_cache.beta_s, layer_idx, "beta_proj.scales")?;
    let beta_b = require(layer_cache.beta_b, layer_idx, "beta_proj.biases")?;
    let alpha_w = require(layer_cache.alpha_w, layer_idx, "alpha_proj.weight")?;
    let alpha_s = require(layer_cache.alpha_s, layer_idx, "alpha_proj.scales")?;
    let alpha_b = require(layer_cache.alpha_b, layer_idx, "alpha_proj.biases")?;
    let conv1d_w = require(layer_cache.conv1d_w, layer_idx, "linear_attn.conv1d.weight")?;
    let a_log = require(layer_cache.a_log, layer_idx, "linear_attn.A_log")?;
    let dt_bias = require(layer_cache.dt_bias, layer_idx, "linear_attn.dt_bias")?;
    let gnorm_w = require(layer_cache.gated_norm_w, layer_idx, "linear_attn.g_norm.weight")?;
    let o_w = require(layer_cache.linear_o_proj_w, layer_idx, "linear_attn.o_proj.weight")?;
    let o_s = require(layer_cache.linear_o_proj_s, layer_idx, "linear_attn.o_proj.scales")?;
    let o_b = require(layer_cache.linear_o_proj_b, layer_idx, "linear_attn.o_proj.biases")?;
    let gate_w = require(layer_cache.gate_w, layer_idx, "mlp.gate.weight")?;
    let gate_s = require(layer_cache.gate_s, layer_idx, "mlp.gate.scales")?;
    let gate_b = require(layer_cache.gate_b, layer_idx, "mlp.gate.biases")?;
    let shared_up_w = require(layer_cache.shared_up_w, layer_idx, "shared.up_proj.w")?;
    let shared_up_s = require(layer_cache.shared_up_s, layer_idx, "shared.up_proj.s")?;
    let shared_up_b = require(layer_cache.shared_up_b, layer_idx, "shared.up_proj.b")?;
    let shared_gate_w = require(layer_cache.shared_gate_w, layer_idx, "shared.gate_proj.w")?;
    let shared_gate_s = require(layer_cache.shared_gate_s, layer_idx, "shared.gate_proj.s")?;
    let shared_gate_b = require(layer_cache.shared_gate_b, layer_idx, "shared.gate_proj.b")?;
    let shared_down_w = require(layer_cache.shared_down_w, layer_idx, "shared.down_proj.w")?;
    let shared_down_s = require(layer_cache.shared_down_s, layer_idx, "shared.down_proj.s")?;
    let shared_down_b = require(layer_cache.shared_down_b, layer_idx, "shared.down_proj.b")?;
    let seg_w = require(layer_cache.seg_w, layer_idx, "shared_expert_gate.w")?;
    let seg_s = require(layer_cache.seg_s, layer_idx, "shared_expert_gate.s")?;
    let seg_b = require(layer_cache.seg_b, layer_idx, "shared_expert_gate.b")?;

    // Pre-fetch every pipeline.
    let lp = LinearAttnPipelines::fetch(metal)?;
    let mv = MatvecPipelines::fetch(metal)?;
    let sum_sq = metal.pipeline("rms_norm_sum_sq")?.clone();
    let apply = metal.pipeline("rms_norm_apply_bf16")?.clone();
    let resid_add = metal.pipeline("residual_add")?.clone();

    // ── snapshot residual + CPU input norm ───────────────────────
    // The C path uses `cpu_rms_norm` here (infer.m:4492) and memcpy's
    // the result into `buf_input` (line 4498). Mirror that exactly so
    // the post-norm bytes match bit-for-bit between sides; the GPU
    // rms_norm chain (which slice 9e tested bit-exact GPU-vs-GPU)
    // would still drift ~5 ULPs vs CPU rms_norm here. Compounded
    // through the projection matvec that drift becomes substantial.
    let input_host = read_buffer_to_vec(&buffers.input, v.hidden_dim);
    {
        let dst = buffers.residual.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(
                input_host.as_ptr(),
                dst,
                v.hidden_dim,
            );
        }
    }
    {
        let mut normed_host = vec![0.0f32; v.hidden_dim];
        let weight_name =
            format!("model.layers.{layer_idx}.input_layernorm.weight");
        rms_norm_cpu(wf, &weight_name, &input_host, &mut normed_host)
            .map_err(|_| LinearAttnForwardError::MissingTensor {
                layer: layer_idx,
                tensor: "input_layernorm.weight (cpu rms_norm failed)",
            })?;
        let dst = buffers.normed.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(
                normed_host.as_ptr(),
                dst,
                v.hidden_dim,
            );
        }
    }

    // ── CMD1: projections + linear-attn pipeline ─────────────────
    {
        let cmdbuf = metal.queue().new_command_buffer();

        // 4 batched projections from buffers.normed:
        let specs = [
            MatvecSpec {
                w_off: qkv_w,
                s_off: qkv_s,
                b_off: qkv_b,
                input: &buffers.normed,
                output: &buffers.batch_out[0],
                out_dim: v.linear_conv_dim() as u32,
                in_dim: v.hidden_dim as u32,
                bits: qkv_bits,
            },
            MatvecSpec {
                w_off: z_w,
                s_off: z_s,
                b_off: z_b,
                input: &buffers.normed,
                output: &buffers.batch_out[1],
                out_dim: v.linear_total_value() as u32,
                in_dim: v.hidden_dim as u32,
                bits: z_bits,
            },
            MatvecSpec {
                w_off: beta_w,
                s_off: beta_s,
                b_off: beta_b,
                input: &buffers.normed,
                output: &buffers.batch_out[2],
                out_dim: v.linear_num_v_heads as u32,
                in_dim: v.hidden_dim as u32,
                bits: beta_bits,
            },
            MatvecSpec {
                w_off: alpha_w,
                s_off: alpha_s,
                b_off: alpha_b,
                input: &buffers.normed,
                output: &buffers.batch_out[3],
                out_dim: v.linear_num_v_heads as u32,
                in_dim: v.hidden_dim as u32,
                bits: alpha_bits,
            },
        ];
        for s in &specs {
            encode_matvec(cmdbuf, &mv, wf_buf, s);
        }

        encode_conv1d_step(
            cmdbuf,
            &lp.conv1d_step,
            &buffers.conv_state[linear_layer_idx],
            &buffers.batch_out[0],
            wf_buf.buffer(),
            conv1d_w,
            &buffers.conv_output,
            v.linear_conv_dim() as u32,
        );

        encode_rms_norm_qk(
            cmdbuf,
            &lp.rms_norm_qk,
            &buffers.conv_output,
            v.linear_num_k_heads as u32,
            Variant::LINEAR_KEY_DIM as u32,
        );

        encode_compute_decay_beta(
            cmdbuf,
            &lp.compute_decay_beta,
            &buffers.batch_out[3], // alpha
            &buffers.batch_out[2], // beta
            wf_buf.buffer(),
            a_log,
            dt_bias,
            &buffers.delta_g_decay,
            &buffers.delta_beta,
            v.linear_num_v_heads as u32,
        );

        let k_heads_per_v =
            (v.linear_num_v_heads / v.linear_num_k_heads) as u32;
        encode_delta_net_step(
            cmdbuf,
            &lp.delta_net_step,
            &buffers.delta_state[linear_layer_idx],
            &buffers.conv_output,
            &buffers.delta_g_decay,
            &buffers.delta_beta,
            &buffers.delta_output,
            v.linear_num_v_heads as u32,
            Variant::LINEAR_VALUE_DIM as u32,
            k_heads_per_v,
        );

        encode_gated_rms_norm(
            cmdbuf,
            &lp.gated_rms_norm,
            &buffers.delta_output,
            &buffers.batch_out[1], // z
            wf_buf.buffer(),
            gnorm_w,
            &buffers.batch_out[6],
            v.linear_num_v_heads as u32,
            Variant::LINEAR_VALUE_DIM as u32,
        );

        cmdbuf.commit();
        cmdbuf.wait_until_completed();
    }

    // ── CMD2: o_proj + residual_add + post-attn rms_norm ─────────
    {
        let cmdbuf = metal.queue().new_command_buffer();

        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: o_w,
                s_off: o_s,
                b_off: o_b,
                input: &buffers.batch_out[6],
                output: &buffers.output,
                out_dim: v.hidden_dim as u32,
                in_dim: v.linear_total_value() as u32,
                bits: o_bits,
            },
        );

        encode_residual_add(
            cmdbuf,
            &resid_add,
            &buffers.output,
            &buffers.residual,
            &buffers.h_mid,
            v.hidden_dim as u32,
        );

        encode_rms_norm_pair(
            cmdbuf,
            &sum_sq,
            &apply,
            &buffers.h_mid,
            wf_buf.buffer(),
            post_attn_norm_w,
            &buffers.normed, // post-attn-norm output
            &buffers.sum_sq,
            v.hidden_dim as u32,
        );

        cmdbuf.commit();
        cmdbuf.wait_until_completed();
    }

    // ── CMD3a: gate logits + shared-gate score + shared FFN ──────
    {
        let cmdbuf = metal.queue().new_command_buffer();

        // Gate logits: HIDDEN_DIM → NUM_EXPERTS.
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: gate_w,
                s_off: gate_s,
                b_off: gate_b,
                input: &buffers.normed,
                output: &buffers.batch_out[4],
                out_dim: v.num_experts as u32,
                in_dim: v.hidden_dim as u32,
                bits: gate_bits,
            },
        );

        // Shared-expert gate scalar: HIDDEN_DIM → 1.
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: seg_w,
                s_off: seg_s,
                b_off: seg_b,
                input: &buffers.normed,
                output: &buffers.batch_out[5],
                out_dim: 1,
                in_dim: v.hidden_dim as u32,
                bits: seg_bits,
            },
        );

        // Shared expert FFN: gate + up matvecs only — swiglu runs on
        // CPU after readback to match the C path's cpu_swiglu
        // (infer.m:2977). Then a final cmdbuf does shared_down on GPU.
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: shared_gate_w,
                s_off: shared_gate_s,
                b_off: shared_gate_b,
                input: &buffers.normed,
                output: &buffers.shared_gate_out,
                out_dim: v.shared_intermediate as u32,
                in_dim: v.hidden_dim as u32,
                bits: s_gate_bits,
            },
        );
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: shared_up_w,
                s_off: shared_up_s,
                b_off: shared_up_b,
                input: &buffers.normed,
                output: &buffers.shared_up_out,
                out_dim: v.shared_intermediate as u32,
                in_dim: v.hidden_dim as u32,
                bits: s_up_bits,
            },
        );

        cmdbuf.commit();
        cmdbuf.wait_until_completed();
    }

    // ── CPU swiglu of shared_gate × shared_up → shared_act ───────
    {
        let g = read_buffer_to_vec(
            &buffers.shared_gate_out,
            v.shared_intermediate,
        );
        let u =
            read_buffer_to_vec(&buffers.shared_up_out, v.shared_intermediate);
        let act_dst = buffers.shared_act.contents() as *mut f32;
        for i in 0..v.shared_intermediate {
            // SiLU(g) * u, matching cpu_swiglu (infer.m:~2400):
            // silu(x) = x / (1 + exp(-x))
            let silu = g[i] / (1.0 + (-g[i]).exp());
            unsafe {
                *act_dst.add(i) = silu * u[i];
            }
        }
    }

    // ── CMD3a-b: shared_down matvec (separate cmdbuf to match C) ─
    {
        let cmdbuf = metal.queue().new_command_buffer();
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: shared_down_w,
                s_off: shared_down_s,
                b_off: shared_down_b,
                input: &buffers.shared_act,
                output: &buffers.shared_out,
                out_dim: v.hidden_dim as u32,
                in_dim: v.shared_intermediate as u32,
                bits: s_down_bits,
            },
        );
        cmdbuf.commit();
        cmdbuf.wait_until_completed();
    }

    // ── CPU: MoE router on the gate logits ───────────────────────
    let mut scores = read_buffer_to_vec(&buffers.batch_out[4], v.num_experts);
    let mut indices = vec![0i32; k_active];
    let mut weights = vec![0f32; k_active];
    moe_router_cpu(&mut scores, k_active, &mut indices, &mut weights)?;

    // Read shared-gate score scalar (pre-sigmoid).
    let shared_gate_score = {
        let s = read_buffer_to_vec(&buffers.batch_out[5], 1);
        s[0]
    };

    // ── Load K expert blobs from disk ────────────────────────────
    let k = k_active;
    let expert_size = v.expert_size_4bit();
    let mut expert_data = vec![0u8; k * expert_size];
    for slot in 0..k {
        let expert_idx = indices[slot] as usize;
        let dst = &mut expert_data[slot * expert_size..(slot + 1) * expert_size];
        expert_files.read_expert(layer_idx, expert_idx, dst)?;
    }

    // ── CMD3b: K-expert FFN + combine via slice 9b ───────────────
    let h_mid_host = read_buffer_to_vec(&buffers.h_mid, v.hidden_dim);
    let shared_out_host = read_buffer_to_vec(&buffers.shared_out, v.hidden_dim);
    let normed_host = read_buffer_to_vec(&buffers.normed, v.hidden_dim);

    let mut hidden_out = vec![0f32; v.hidden_dim];
    gpu_batched_experts_forward(
        metal,
        moe,
        k as i32,
        &expert_data,
        &normed_host, // h_post (post-attn-norm); experts read this as their input
        &h_mid_host,
        &shared_out_host,
        &weights,
        shared_gate_score,
        &mut hidden_out,
    )?;

    // Write hidden_out back into buffers.input so the next layer (or
    // caller's dump-hook readback) sees it there.
    {
        let dst = buffers.input.contents() as *mut f32;
        // SAFETY: shared storage; no GPU work in flight (cmdbuf above
        // committed and waited).
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden_out.as_ptr(),
                dst,
                v.hidden_dim,
            );
        }
    }

    Ok(())
}

fn require(
    val: Option<u64>,
    layer: usize,
    tensor: &'static str,
) -> Result<u64, LinearAttnForwardError> {
    val.ok_or(LinearAttnForwardError::MissingTensor { layer, tensor })
}

fn read_buffer_to_vec(b: &Buffer, len: usize) -> Vec<f32> {
    let ptr = b.contents() as *const f32;
    // SAFETY: caller ensures no GPU work in flight on `b`.
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}

#[allow(clippy::too_many_arguments)]
fn encode_rms_norm_pair(
    cmdbuf: &metal::CommandBufferRef,
    sum_pipe: &ComputePipelineState,
    apply_pipe: &ComputePipelineState,
    input: &Buffer,
    weight_buf: &Buffer,
    weight_off: u64,
    output: &Buffer,
    sum_sq: &Buffer,
    dim: u32,
) {
    {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(sum_pipe);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(sum_sq), 0);
        enc.set_bytes(2, 4, (&dim as *const u32).cast());
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
    {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(apply_pipe);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(weight_buf), weight_off as NSUInteger);
        enc.set_buffer(2, Some(sum_sq), 0);
        enc.set_buffer(3, Some(output), 0);
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
}

fn encode_residual_add(
    cmdbuf: &metal::CommandBufferRef,
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

