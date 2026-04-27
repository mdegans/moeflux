//! RMS normalization — Rust port of `infer.m`'s `cpu_rms_norm`.
//!
//! `out[i] = x[i] / sqrt(mean(x*x) + eps) * bf16_to_f32(w[i])`.
//!
//! ## Bit-exact contract
//!
//! Bit-equal against the C path on the same hardware: same
//! sequential accumulation order in the sum-of-squares loop, same
//! `f32` arithmetic for `sqrt` / division, same bf16-as-shift weight
//! decode. The diff oracle asserts byte-equal output.
//!
//! Floating-point determinism note: this hold for the CPU path on
//! Apple silicon's `expf`/`sqrtf` (libm). Once the Metal `rms_norm_*`
//! kernels land in a future slice, bit-exactness moves to a per-
//! kernel loose-tolerance comparison (cosine / Jaccard floors), since
//! Metal's `fast_math` and reduction order do diverge from the CPU
//! reference.

use crate::riir::embedding::bf16_to_f32;
use crate::riir::variants::{RMS_NORM_EPS, VARIANT};
use crate::riir::weight_file::WeightFile;

/// Errors specific to the RMSNorm port.
#[derive(Debug, thiserror::Error)]
pub enum RmsNormError {
    #[error("RMSNorm weight tensor '{name}' missing from manifest")]
    MissingTensor { name: String },
    #[error(
        "RMSNorm weight '{name}' size {got} bytes, expected {expected} ({dim} bf16 elements)"
    )]
    WeightSize {
        name: String,
        got: u64,
        expected: u64,
        dim: usize,
    },
    #[error("input length {got} != expected {expected}")]
    InputLen { got: usize, expected: usize },
    #[error("output length {got} != expected {expected}")]
    OutputLen { got: usize, expected: usize },
    #[error("non-positive shape: num_heads={num_heads}, head_dim={head_dim}")]
    NonPositiveShape { num_heads: usize, head_dim: usize },
}

/// Apply RMSNorm with the bf16 weight tensor named `weight_name`.
/// `x` and `out` must both be `HIDDEN_DIM` long; they may not alias.
pub fn rms_norm_cpu(
    wf: &WeightFile,
    weight_name: &str,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), RmsNormError> {
    let dim = VARIANT.hidden_dim;
    if x.len() != dim {
        return Err(RmsNormError::InputLen {
            got: x.len(),
            expected: dim,
        });
    }
    if out.len() != dim {
        return Err(RmsNormError::OutputLen {
            got: out.len(),
            expected: dim,
        });
    }

    let bytes = wf
        .tensor_bytes(weight_name)
        .ok_or_else(|| RmsNormError::MissingTensor {
            name: weight_name.to_string(),
        })?;
    let expected_bytes = (dim * 2) as u64;
    if bytes.len() as u64 != expected_bytes {
        return Err(RmsNormError::WeightSize {
            name: weight_name.to_string(),
            got: bytes.len() as u64,
            expected: expected_bytes,
            dim,
        });
    }

    let mut sum_sq: f32 = 0.0;
    for &xi in x.iter() {
        sum_sq += xi * xi;
    }
    let rms = (sum_sq / dim as f32 + RMS_NORM_EPS).sqrt();
    let inv_rms = 1.0f32 / rms;

    for i in 0..dim {
        let w_bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        let weight = bf16_to_f32(w_bits);
        out[i] = x[i] * inv_rms * weight;
    }

    Ok(())
}

/// Per-head CPU RMSNorm, mutating in place. `x_inout` is
/// `num_heads * head_dim` floats laid out contiguously per head; each
/// head's `head_dim`-long slice is independently RMS-normalized and
/// scaled by the same shared bf16 weight tensor of length `head_dim`
/// loaded from `weight_name`. Bit-exact against
/// `mf_rms_norm_per_head_cpu` on the same hardware.
pub fn rms_norm_per_head_cpu(
    wf: &WeightFile,
    weight_name: &str,
    num_heads: usize,
    head_dim: usize,
    x_inout: &mut [f32],
) -> Result<(), RmsNormError> {
    if num_heads == 0 || head_dim == 0 {
        return Err(RmsNormError::NonPositiveShape {
            num_heads,
            head_dim,
        });
    }
    let expected_len = num_heads * head_dim;
    if x_inout.len() != expected_len {
        return Err(RmsNormError::InputLen {
            got: x_inout.len(),
            expected: expected_len,
        });
    }

    let bytes = wf
        .tensor_bytes(weight_name)
        .ok_or_else(|| RmsNormError::MissingTensor {
            name: weight_name.to_string(),
        })?;
    let expected_bytes = (head_dim * 2) as u64;
    if bytes.len() as u64 != expected_bytes {
        return Err(RmsNormError::WeightSize {
            name: weight_name.to_string(),
            got: bytes.len() as u64,
            expected: expected_bytes,
            dim: head_dim,
        });
    }

    let head_dim_f = head_dim as f32;
    for h in 0..num_heads {
        let xh = &mut x_inout[h * head_dim..(h + 1) * head_dim];
        let mut sum_sq: f32 = 0.0;
        for &xi in xh.iter() {
            sum_sq += xi * xi;
        }
        let inv_rms = 1.0f32 / (sum_sq / head_dim_f + RMS_NORM_EPS).sqrt();
        for i in 0..head_dim {
            let w_bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
            let weight = bf16_to_f32(w_bits);
            xh[i] = xh[i] * inv_rms * weight;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-only sanity. Real bit-exact diff lives in the harness.
    #[test]
    fn rms_norm_eps_is_1e6() {
        assert!((RMS_NORM_EPS - 1e-6).abs() < 1e-12);
    }
}
