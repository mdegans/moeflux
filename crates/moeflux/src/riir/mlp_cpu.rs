//! CPU SwiGLU MLP for the Cogito-V2 / DeepSeek-V3 path.
//!
//! Used for two block flavors:
//!
//! 1. **Dense MLP** — layers `[0, first_k_dense_replace)` use a single
//!    SwiGLU MLP with width [`super::variants::Variant::dense_intermediate`]
//!    (18432 for Cogito-V2). Tensors live under
//!    `model.layers.{i}.mlp.{gate,up,down}_proj`.
//! 2. **Shared expert** — every MoE layer also runs a single
//!    "shared expert" MLP at width [`super::variants::Variant::shared_intermediate`]
//!    (2048 for Cogito-V2) on the same hidden state, added to the
//!    routed-experts sum unconditionally. Tensors live under
//!    `model.layers.{i}.mlp.shared_experts.{gate,up,down}_proj`.
//!
//! Both flavors share the same shape:
//! `down(silu(gate(h)) * up(h))`. Parameterize over the tensor prefix
//! and intermediate width.
//!
//! ## Performance note
//!
//! At 7168 → 18432 → 7168 (dense), each layer is 3 * 7168 * 18432 ≈
//! 400M ops per token. Single-threaded ~40ms; with rayon-parallel
//! [`super::cpu_matvec::project_4bit_cpu`] inside, much less. Three
//! dense layers per token is ~120ms warm — acceptable for first-run.

use super::cpu_matvec::{project_4bit_cpu, CpuMatvecError};
use super::variants::VARIANT;
use super::weight_file::WeightFile;

/// Errors specific to the CPU MLP path.
#[derive(Debug, thiserror::Error)]
pub enum MlpForwardError {
    #[error("hidden buffer length {got} != hidden_dim ({expected})")]
    HiddenLen { got: usize, expected: usize },
    #[error("output buffer length {got} != hidden_dim ({expected})")]
    OutLen { got: usize, expected: usize },
    #[error("4-bit matvec error in MLP: {0}")]
    Matvec(#[from] CpuMatvecError),
}

/// Dense SwiGLU MLP using `model.layers.{layer_idx}.mlp.{gate,up,down}_proj`
/// at width [`super::variants::Variant::dense_intermediate`]. Used for
/// layers `[0, first_k_dense_replace)` on Cogito-V2 (layers 0–2).
///
/// `out` is the post-MLP residual contribution; the caller adds it to
/// the post-attn residual hidden state. `out` may not alias `hidden`.
pub fn dense_mlp_swiglu_cpu(
    wf: &WeightFile,
    layer_idx: usize,
    hidden: &[f32],
    out: &mut [f32],
) -> Result<(), MlpForwardError> {
    let prefix = format!("model.layers.{layer_idx}.mlp");
    swiglu_ffn_4bit_cpu(wf, &prefix, VARIANT.dense_intermediate, hidden, out)
}

/// Shared-expert SwiGLU MLP using
/// `model.layers.{layer_idx}.mlp.shared_experts.{gate,up,down}_proj`
/// at width [`super::variants::Variant::shared_intermediate`]. Run on
/// every MoE layer; the result is added to the routed-experts sum
/// unconditionally (no gate) for DeepSeek-V3 / Cogito-V2.
pub fn shared_expert_swiglu_cpu(
    wf: &WeightFile,
    layer_idx: usize,
    hidden: &[f32],
    out: &mut [f32],
) -> Result<(), MlpForwardError> {
    let prefix = format!("model.layers.{layer_idx}.mlp.shared_experts");
    swiglu_ffn_4bit_cpu(wf, &prefix, VARIANT.shared_intermediate, hidden, out)
}

/// Generic SwiGLU FFN reading 4-bit-packed weights from the manifest
/// at `<prefix>.{gate,up,down}_proj`. Computes
/// `out = down(silu(gate(h)) * up(h))`.
///
/// All sizing comes from `VARIANT.hidden_dim` and `intermediate`; no
/// implicit assumptions about which block-kind we are.
pub fn swiglu_ffn_4bit_cpu(
    wf: &WeightFile,
    prefix: &str,
    intermediate: usize,
    hidden: &[f32],
    out: &mut [f32],
) -> Result<(), MlpForwardError> {
    let hidden_dim = VARIANT.hidden_dim;
    if hidden.len() != hidden_dim {
        return Err(MlpForwardError::HiddenLen {
            got: hidden.len(),
            expected: hidden_dim,
        });
    }
    if out.len() != hidden_dim {
        return Err(MlpForwardError::OutLen {
            got: out.len(),
            expected: hidden_dim,
        });
    }

    let gate_name = format!("{prefix}.gate_proj");
    let up_name = format!("{prefix}.up_proj");
    let down_name = format!("{prefix}.down_proj");

    let mut gate = vec![0.0f32; intermediate];
    let mut up = vec![0.0f32; intermediate];
    project_4bit_cpu(wf, &gate_name, hidden_dim, intermediate, hidden, &mut gate)?;
    project_4bit_cpu(wf, &up_name, hidden_dim, intermediate, hidden, &mut up)?;

    // mid = silu(gate) * up, in place into `gate`.
    for i in 0..intermediate {
        let g = gate[i];
        let silu = g / (1.0 + (-g).exp());
        gate[i] = silu * up[i];
    }

    project_4bit_cpu(wf, &down_name, intermediate, hidden_dim, &gate, out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke against real weights: layer 0 is dense (Cogito-V2 has
    /// `first_k_dense_replace = 3`). Run the MLP on a pulse input
    /// and assert finite + sensible-magnitude output.
    #[cfg(feature = "model-cogito-v2-671b")]
    #[test]
    #[ignore = "needs Cogito-V2 weights mmap'd from /Volumes/Temp Backup"]
    fn dense_mlp_layer0_smoke() {
        use std::path::Path;
        let bin = Path::new(
            "/Volumes/Temp Backup/models/blallama/cogito-v2-671b/artifacts/model_weights.bin",
        );
        let manifest = Path::new(
            "/Volumes/Temp Backup/models/blallama/cogito-v2-671b/artifacts/model_weights.json",
        );
        let wf = WeightFile::open(bin, manifest).expect("open weights");

        let v = VARIANT;
        let mut hidden = vec![0.0f32; v.hidden_dim];
        hidden[42] = 1.0;
        let mut out = vec![0.0f32; v.hidden_dim];
        dense_mlp_swiglu_cpu(&wf, 0, &hidden, &mut out)
            .expect("dense MLP layer 0");
        assert!(out.iter().all(|x| x.is_finite()));
        let max_abs = out.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        assert!(max_abs > 0.0, "all-zero output — likely a wiring bug");
        assert!(max_abs < 1e6, "magnitude {max_abs} suspiciously large");
    }
}
