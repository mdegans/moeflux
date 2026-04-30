//! CPU MoE composition for the DeepSeek-V3 / Cogito-V2 path.
//!
//! Wraps the existing pieces:
//!
//! - Router gate: BF16 matvec via [`super::cpu_matvec::project_bf16_cpu`]
//!   (the gate is `model.layers.{i}.mlp.gate.weight`, BF16 not 4-bit).
//! - Expert selection: [`super::moe_router::noaux_tc_router_cpu`].
//! - Per-expert FFN: read `EXPERT_SIZE` bytes via
//!   [`super::expert_io::ExpertFiles::read_expert`], then run a SwiGLU
//!   matvec over the `gate / up / down` sub-blocks.
//! - Shared expert: [`super::mlp_cpu::shared_expert_swiglu_cpu`], added
//!   **unconditionally** (no gate, contrasting with Qwen3's gated path).
//!
//! Per-token cost on Cogito-V2: 8 experts × ~24 MB blob read + 3 ×
//! ~30M ops per expert. SSD read dominates wall-time (~200 MB I/O per
//! MoE layer per token). At single-threaded ~400 MB/s sustained read
//! that's ~500 ms / layer × 58 MoE layers ≈ 30 s / token. Slow, but
//! not the load-bearing bottleneck the diff oracle exists to verify.

use super::cpu_matvec::{
    bf16_matvec_cpu, dequant_matvec_4bit_bytes_cpu, CpuMatvecError,
};
use super::embedding::bf16_to_f32;
use super::expert_io::{ExpertFiles, ExpertIoError};
use super::mlp_cpu::{shared_expert_swiglu_cpu, MlpForwardError};
use super::moe_router::{noaux_tc_router_cpu, MoeRouterError};
use super::variants::{Variant, GROUP_SIZE, VARIANT};
use super::weight_file::WeightFile;

/// Errors specific to the CPU MoE forward.
#[derive(Debug, thiserror::Error)]
pub enum MoeForwardError {
    #[error("hidden buffer length {got} != hidden_dim ({expected})")]
    HiddenLen { got: usize, expected: usize },
    #[error("output buffer length {got} != hidden_dim ({expected})")]
    OutLen { got: usize, expected: usize },
    #[error("missing tensor '{name}'")]
    MissingTensor { name: String },
    #[error(
        "router-gate bias '{name}' size {got} bytes, expected {expected}"
    )]
    BiasSize {
        name: String,
        got: usize,
        expected: usize,
    },
    #[error("4-bit matvec error in MoE: {0}")]
    Matvec(#[from] CpuMatvecError),
    #[error("router error in MoE: {0}")]
    Router(#[from] MoeRouterError),
    #[error("shared-expert MLP error in MoE: {0}")]
    Mlp(#[from] MlpForwardError),
    #[error("expert I/O error in MoE: {0}")]
    Io(#[from] ExpertIoError),
}

/// Per-token MoE composition for one DeepSeek-V3 layer. Computes
/// `out = sum_k weights[k] * expert(indices[k], hidden) +
/// shared_expert(hidden)`, where `(indices, weights)` come from the
/// noaux_tc router.
///
/// `out` is the post-MoE residual contribution; the caller adds it
/// to the post-attn residual hidden state. `out` may not alias
/// `hidden`.
pub fn deepseek_moe_cpu(
    wf: &WeightFile,
    expert_files: &ExpertFiles,
    layer_idx: usize,
    hidden: &[f32],
    out: &mut [f32],
) -> Result<(), MoeForwardError> {
    let v = VARIANT;
    let hidden_dim = v.hidden_dim;
    let num_experts = v.num_experts;
    let k = v.num_experts_per_tok;

    if hidden.len() != hidden_dim {
        return Err(MoeForwardError::HiddenLen {
            got: hidden.len(),
            expected: hidden_dim,
        });
    }
    if out.len() != hidden_dim {
        return Err(MoeForwardError::OutLen {
            got: out.len(),
            expected: hidden_dim,
        });
    }

    // ---- Router: gate logits + correction bias ----
    let gate_w_name =
        format!("model.layers.{layer_idx}.mlp.gate.weight");
    let bias_name = format!(
        "model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
    );

    // gate.weight is BF16 [num_experts, hidden_dim] — read as raw
    // u16 slice via `tensor_bytes`, then bf16_matvec.
    let gate_w_bytes = wf
        .tensor_bytes(&gate_w_name)
        .ok_or_else(|| MoeForwardError::MissingTensor {
            name: gate_w_name.clone(),
        })?;
    // 2 bytes per BF16; expected count = num_experts * hidden_dim.
    let expected_gate_bytes = num_experts * hidden_dim * 2;
    if gate_w_bytes.len() != expected_gate_bytes {
        return Err(MoeForwardError::BiasSize {
            name: gate_w_name,
            got: gate_w_bytes.len(),
            expected: expected_gate_bytes,
        });
    }
    let gate_w = bytemuck_u16(gate_w_bytes);

    let mut gate_logits = vec![0.0f32; num_experts];
    bf16_matvec_cpu(
        gate_w,
        hidden_dim,
        num_experts,
        hidden,
        &mut gate_logits,
    )?;

    let bias_bytes = wf.tensor_bytes(&bias_name).ok_or_else(|| {
        MoeForwardError::MissingTensor {
            name: bias_name.clone(),
        }
    })?;
    let expected_bias_bytes = num_experts * 2;
    if bias_bytes.len() != expected_bias_bytes {
        return Err(MoeForwardError::BiasSize {
            name: bias_name,
            got: bias_bytes.len(),
            expected: expected_bias_bytes,
        });
    }
    let bias_u16 = bytemuck_u16(bias_bytes);
    let bias_f32: Vec<f32> =
        bias_u16.iter().map(|&b| bf16_to_f32(b)).collect();

    let mut indices = vec![0i32; k];
    let mut weights = vec![0.0f32; k];
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

    // ---- Routed experts ----
    out.fill(0.0);
    let mut blob = vec![0u8; v.expert_size_4bit()];
    let mut expert_out = vec![0.0f32; hidden_dim];
    for slot in 0..k {
        let expert_idx = indices[slot] as usize;
        let w = weights[slot];
        expert_files.read_expert(layer_idx, expert_idx, &mut blob)?;
        run_packed_expert_swiglu(&v, &blob, hidden, &mut expert_out)?;
        for i in 0..hidden_dim {
            out[i] = w.mul_add(expert_out[i], out[i]);
        }
    }

    // ---- Shared expert (UNCONDITIONAL — no gate) ----
    let mut shared = vec![0.0f32; hidden_dim];
    shared_expert_swiglu_cpu(wf, layer_idx, hidden, &mut shared)?;
    for i in 0..hidden_dim {
        out[i] += shared[i];
    }

    Ok(())
}

/// Run a SwiGLU FFN on a single packed-expert byte blob. The blob
/// layout is [`Variant::expert_size_4bit`] bytes containing
/// `[gate_w | gate_s | gate_b | up_w | up_s | up_b | down_w |
/// down_s | down_b]`, with offsets given by the variant's
/// `*_off_4bit()` helpers.
///
/// Shape:
/// - `gate.weight`: `[moe_intermediate, hidden_dim / 8]` U32
/// - `gate.scales`: `[moe_intermediate, hidden_dim / GROUP_SIZE]` BF16
/// - `up`:    same shape as gate
/// - `down.weight`: `[hidden_dim, moe_intermediate / 8]` U32
/// - `down.scales`: `[hidden_dim, moe_intermediate / GROUP_SIZE]` BF16
fn run_packed_expert_swiglu(
    v: &Variant,
    blob: &[u8],
    hidden: &[f32],
    out: &mut [f32],
) -> Result<(), MoeForwardError> {
    let hidden_dim = v.hidden_dim;
    let intermediate = v.moe_intermediate;
    let w_bytes = v.expert_weight_bytes_4bit();
    let s_bytes = v.expert_scale_bytes();
    debug_assert_eq!(intermediate % GROUP_SIZE, 0);
    debug_assert_eq!(hidden_dim % GROUP_SIZE, 0);

    let gate_w = &blob[v.gate_w_off_4bit()..v.gate_w_off_4bit() + w_bytes];
    let gate_s = &blob[v.gate_s_off_4bit()..v.gate_s_off_4bit() + s_bytes];
    let gate_b = &blob[v.gate_b_off_4bit()..v.gate_b_off_4bit() + s_bytes];
    let up_w = &blob[v.up_w_off_4bit()..v.up_w_off_4bit() + w_bytes];
    let up_s = &blob[v.up_s_off_4bit()..v.up_s_off_4bit() + s_bytes];
    let up_b = &blob[v.up_b_off_4bit()..v.up_b_off_4bit() + s_bytes];
    let down_w = &blob[v.down_w_off_4bit()..v.down_w_off_4bit() + w_bytes];
    let down_s = &blob[v.down_s_off_4bit()..v.down_s_off_4bit() + s_bytes];
    let down_b = &blob[v.down_b_off_4bit()..v.down_b_off_4bit() + s_bytes];

    let mut gate = vec![0.0f32; intermediate];
    let mut up = vec![0.0f32; intermediate];
    dequant_matvec_4bit_bytes_cpu(
        gate_w, gate_s, gate_b, hidden_dim, intermediate, hidden, &mut gate,
    )?;
    dequant_matvec_4bit_bytes_cpu(
        up_w, up_s, up_b, hidden_dim, intermediate, hidden, &mut up,
    )?;
    for i in 0..intermediate {
        let g = gate[i];
        let silu = g / (1.0 + (-g).exp());
        gate[i] = silu * up[i];
    }
    dequant_matvec_4bit_bytes_cpu(
        down_w, down_s, down_b, intermediate, hidden_dim, &gate, out,
    )?;
    Ok(())
}

/// Reinterpret a `&[u8]` as `&[u16]` via `align_to`. The MLX
/// converter aligns BF16 tensors to 2-byte boundaries; the head/tail
/// must therefore be empty. Panics if not — this is a converter bug
/// we want loud, not soft.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke against real weights: layer 3 is the first MoE layer
    /// on Cogito-V2 (`first_k_dense_replace = 3`). Run the full MoE
    /// composition on a pulse input and assert finite + sensible
    /// output. This exercises the router path, expert blob read, all
    /// three SwiGLU matvecs per expert, and the shared-expert add.
    #[cfg(feature = "model-cogito-v2-671b")]
    #[test]
    #[ignore = "needs Cogito-V2 weights and packed_experts/ on /Volumes/Temp Backup"]
    fn moe_layer3_smoke() {
        use std::path::Path;
        let bin = Path::new(
            "/Volumes/Temp Backup/models/blallama/cogito-v2-671b/artifacts/model_weights.bin",
        );
        let manifest = Path::new(
            "/Volumes/Temp Backup/models/blallama/cogito-v2-671b/artifacts/model_weights.json",
        );
        let experts_dir = Path::new(
            "/Volumes/Temp Backup/models/blallama/cogito-v2-671b/root",
        );
        let wf = WeightFile::open(bin, manifest).expect("open weights");
        let ef = ExpertFiles::open(experts_dir).expect("open experts");

        let v = VARIANT;
        let mut hidden = vec![0.0f32; v.hidden_dim];
        // Pulse plus a small spread, so the router gate produces a
        // non-degenerate score distribution across experts.
        for (i, h) in hidden.iter_mut().enumerate() {
            *h = ((i as f32) * 0.001).sin();
        }
        let mut out = vec![0.0f32; v.hidden_dim];
        deepseek_moe_cpu(&wf, &ef, 3, &hidden, &mut out)
            .expect("MoE layer 3");
        assert!(
            out.iter().all(|x| x.is_finite()),
            "non-finite output at index {:?}",
            out.iter().position(|x| !x.is_finite()),
        );
        let max_abs = out.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        assert!(
            max_abs > 0.0,
            "all-zero MoE output — likely router or expert wiring bug"
        );
        assert!(
            max_abs < 1e6,
            "MoE output magnitude {max_abs} suspiciously large"
        );
    }
}
