//! Rotary position embedding — Rust port of `infer.m`'s
//! `apply_rotary_emb`.
//!
//! Non-traditional MLX-style pairing: rotates `(x[i], x[i+half])`
//! over the first `ROTARY_DIM` channels of each head, where
//! `half = ROTARY_DIM / 2`. The remaining `head_dim - rotary_dim`
//! channels of each head are left untouched.
//!
//! Frequencies are derived from `ROPE_THETA`:
//!
//! ```text
//! freq_i  = 1 / pow(theta, 2*i / rotary_dim)
//! angle_i = pos * freq_i
//! ```
//!
//! ## Tolerance contract — not bit-exact
//!
//! RoPE is the first kernel where the diff oracle relaxes from
//! bit-exact to a tight ULP-bounded tolerance. Two reasons compound:
//!
//! 1. The C side at `-O3` auto-vectorizes the trig loop using
//!    Apple's libm vector variants (`__cosf4` etc.), while extern
//!    `"C"` calls from Rust are opaque to LLVM auto-vectorization
//!    and resolve to the scalar `cosf` / `sinf`. Different code
//!    path → different rounding by ≤ 1 ULP.
//! 2. Even with both paths forced to scalar libm, `llvm.cos.f32`
//!    is explicitly documented as implementation-precision — LLVM
//!    is free to lower it differently per target / opt-level.
//!
//! These are compiler-choice artifacts, not porting bugs. Insisting
//! on bit-exact would be the wrong assertion. Instead, the test asserts
//! max-ULP-distance ≤ 4 across all rotated channels, which is well
//! below any meaningful behavior threshold.
//!
//! The scalar libm bindings stay because they make the drift smaller
//! (~1 ULP rather than the few-ULP-but-occasional-larger drift seen
//! when `f32::cos` lowers to a polynomial approximation).

use crate::riir::variants::{ROPE_THETA, VARIANT};

unsafe extern "C" {
    fn cosf(x: f32) -> f32;
    fn sinf(x: f32) -> f32;
    fn powf(base: f32, exp: f32) -> f32;
}

/// Errors specific to the RoPE port.
#[derive(Debug, thiserror::Error)]
pub enum RopeError {
    #[error("position must be non-negative (got {pos})")]
    NegativePos { pos: i32 },
    #[error(
        "Q buffer length {got} != num_attn_heads * head_dim ({expected})"
    )]
    QLen { got: usize, expected: usize },
    #[error(
        "K buffer length {got} != num_kv_heads * head_dim ({expected})"
    )]
    KLen { got: usize, expected: usize },
}

/// Apply rotary position embedding to Q and K at position `pos`.
/// `q` and `k` are mutated in place. Shape comes from the active
/// `VARIANT`.
pub fn apply_rotary_emb(
    pos: i32,
    q: &mut [f32],
    k: &mut [f32],
) -> Result<(), RopeError> {
    if pos < 0 {
        return Err(RopeError::NegativePos { pos });
    }
    let head_dim = VARIANT.head_dim;
    let num_heads = VARIANT.num_attn_heads;
    let num_kv_heads = VARIANT.num_kv_heads;
    let rotary_dim = VARIANT.rotary_dim();

    let q_expected = num_heads * head_dim;
    if q.len() != q_expected {
        return Err(RopeError::QLen {
            got: q.len(),
            expected: q_expected,
        });
    }
    let k_expected = num_kv_heads * head_dim;
    if k.len() != k_expected {
        return Err(RopeError::KLen {
            got: k.len(),
            expected: k_expected,
        });
    }

    let half = rotary_dim / 2;
    let pos_f = pos as f32;
    let rdim_f = rotary_dim as f32;

    for h in 0..num_heads {
        let qh = &mut q[h * head_dim..h * head_dim + head_dim];
        for i in 0..half {
            // SAFETY: cosf/sinf/powf are scalar libm functions with
            // no preconditions beyond well-formed f32 inputs.
            let freq = unsafe {
                1.0f32 / powf(ROPE_THETA, (2 * i) as f32 / rdim_f)
            };
            let angle = pos_f * freq;
            let cos_a = unsafe { cosf(angle) };
            let sin_a = unsafe { sinf(angle) };
            let q0 = qh[i];
            let q1 = qh[i + half];
            qh[i] = q0 * cos_a - q1 * sin_a;
            qh[i + half] = q0 * sin_a + q1 * cos_a;
        }
    }
    for h in 0..num_kv_heads {
        let kh = &mut k[h * head_dim..h * head_dim + head_dim];
        for i in 0..half {
            let freq = unsafe {
                1.0f32 / powf(ROPE_THETA, (2 * i) as f32 / rdim_f)
            };
            let angle = pos_f * freq;
            let cos_a = unsafe { cosf(angle) };
            let sin_a = unsafe { sinf(angle) };
            let k0 = kh[i];
            let k1 = kh[i + half];
            kh[i] = k0 * cos_a - k1 * sin_a;
            kh[i + half] = k0 * sin_a + k1 * cos_a;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// pos=0 leaves q untouched on the rotated channels (cos=1, sin=0).
    #[test]
    fn rope_at_pos_zero_is_identity_on_rotated_channels() {
        let head_dim = VARIANT.head_dim;
        let num_heads = VARIANT.num_attn_heads;
        let num_kv_heads = VARIANT.num_kv_heads;
        let rotary_dim = VARIANT.rotary_dim();

        let mut q: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| i as f32 * 0.001)
            .collect();
        let mut k: Vec<f32> = (0..num_kv_heads * head_dim)
            .map(|i| i as f32 * 0.001)
            .collect();
        let q_orig = q.clone();
        let k_orig = k.clone();

        apply_rotary_emb(0, &mut q, &mut k).unwrap();

        // Rotated half: q[i] = q0 * 1 - q1 * 0 = q0; q[i+half] = q0*0 + q1*1 = q1.
        for h in 0..num_heads {
            for i in 0..rotary_dim {
                assert_eq!(q[h * head_dim + i], q_orig[h * head_dim + i]);
            }
        }
        for h in 0..num_kv_heads {
            for i in 0..rotary_dim {
                assert_eq!(k[h * head_dim + i], k_orig[h * head_dim + i]);
            }
        }
    }
}
