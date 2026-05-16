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

// =====================================================================
// YaRN — extended-context RoPE for DeepSeek-V3 / Cogito-V2.
//
// YaRN extends the usable context window beyond the model's
// pretrain `original_max_position_embeddings` by interpolating between
// the original frequencies (`freq_extra`) and a scaled-context-aware
// set (`freq_inter`), with a smooth ramp between them so the
// transition is gentle. It also applies a "mscale" multiplier to the
// cos/sin tables and to the attention softmax scale (the latter is
// the caller's responsibility — see `yarn_get_mscale_full`).
//
// Reference: `modeling_deepseek.py::DeepseekV3YarnRotaryEmbedding`.
// We compute the inv_freq table once at engine init and pass it in
// to [`apply_rotary_emb_yarn`] each token.
// =====================================================================

/// Compute the dimension at which a given number of rotations occurs
/// at `max_position_embeddings`. Used to find the smooth-ramp range.
///
/// Mirrors `yarn_find_correction_dim(num_rotations, dim, base,
/// max_position_embeddings)` from `modeling_deepseek.py`. The 2π
/// factor in the denominator converts "full rotations" to radians.
pub fn yarn_find_correction_dim(
    num_rotations: f32,
    dim: usize,
    base: f32,
    max_position_embeddings: f32,
) -> f32 {
    let two_pi = 2.0f32 * std::f32::consts::PI;
    let ln_base = base.ln();
    (dim as f32 * (max_position_embeddings / (num_rotations * two_pi)).ln())
        / (2.0 * ln_base)
}

/// Find `(low, high)` correction range. `beta_fast` controls the
/// high-frequency boundary (channels that stay at original freqs);
/// `beta_slow` controls the low-frequency boundary (channels that
/// switch to scaled freqs). Both rotation counts default to 32 / 1
/// for DeepSeek-V3.
pub fn yarn_find_correction_range(
    beta_fast: f32,
    beta_slow: f32,
    dim: usize,
    base: f32,
    max_position_embeddings: f32,
) -> (f32, f32) {
    let low =
        yarn_find_correction_dim(beta_fast, dim, base, max_position_embeddings)
            .floor();
    let high =
        yarn_find_correction_dim(beta_slow, dim, base, max_position_embeddings)
            .ceil();
    let dim_max = (dim - 1) as f32;
    (low.clamp(0.0, dim_max), high.clamp(0.0, dim_max))
}

/// Soft mscale used as a factor on cos/sin and on the attention
/// softmax scale. Identity when `scale <= 1` (no extension), grows
/// logarithmically with the scale factor.
///
/// Mirrors `yarn_get_mscale(scale, mscale)`:
/// `0.1 * mscale * log(scale) + 1` for `scale > 1`.
pub fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// Final mscale used in DeepSeek-V3 (a ratio of two `yarn_get_mscale`
/// calls, with `mscale_all_dim` as the divisor). This factor is
/// multiplied into both `cos`/`sin` AND the attention softmax scale
/// (squared in the latter case — `softmax_scale *= mscale * mscale`).
pub fn yarn_get_mscale_full(
    scale: f32,
    mscale: f32,
    mscale_all_dim: f32,
) -> f32 {
    yarn_get_mscale(scale, mscale) / yarn_get_mscale(scale, mscale_all_dim)
}

/// Build the smooth-ramp interpolation mask used to blend
/// `freq_inter` (scaled, low-freq dims) with `freq_extra` (original,
/// high-freq dims). Returns a `dim`-element vector in `[0, 1]`.
///
/// `min == max` is handled by widening `max` infinitesimally per the
/// reference implementation (avoids a div-by-zero).
pub fn yarn_linear_ramp_mask(min: f32, max: f32, dim: usize) -> Vec<f32> {
    let max_eff = if max == min { max + 0.001 } else { max };
    let mut out = Vec::with_capacity(dim);
    for i in 0..dim {
        let v = (i as f32 - min) / (max_eff - min);
        out.push(v.clamp(0.0, 1.0));
    }
    out
}

/// Compute the per-(2i, half-dim) inverse-frequency table for YaRN.
/// Returns a `dim/2`-element vector ready to multiply by `pos` to get
/// rotation angles.
///
/// `dim` is the rotation dimension (`qk_rope_head_dim` for MLA — 64
/// for Cogito-V2). `base` is the original RoPE theta (10000 for
/// DeepSeek-V3, NOT the Qwen-family 1e7). `factor` is the YaRN
/// extension factor (40 for Cogito-V2). `original_max_position` is
/// the pretrain context length (4096 for Cogito-V2).
///
/// At `factor = 1` this collapses to vanilla `1 / base^(2i/dim)`.
pub fn compute_yarn_inv_freq(
    dim: usize,
    base: f32,
    factor: f32,
    original_max_position: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Vec<f32> {
    let half = dim / 2;
    // freq_extra = 1 / base^(2i / dim)  — original RoPE freqs
    // freq_inter = 1 / (factor * base)^(2i / dim)  — scaled freqs
    let mut freq_extra = Vec::with_capacity(half);
    let mut freq_inter = Vec::with_capacity(half);
    for i in 0..half {
        let exp_i = (2 * i) as f32 / dim as f32;
        // SAFETY: scalar libm; well-formed f32 inputs.
        let extra = 1.0 / unsafe { powf(base, exp_i) };
        let inter = 1.0 / unsafe { powf(factor * base, exp_i) };
        freq_extra.push(extra);
        freq_inter.push(inter);
    }

    // Smooth ramp: high-frequency dims (small i) stay at extra;
    // low-frequency dims (large i) switch to inter. The mask returns
    // 1.0 → keep extra, 0.0 → use inter.
    let (low, high) = yarn_find_correction_range(
        beta_fast,
        beta_slow,
        dim,
        base,
        original_max_position,
    );
    let ramp = yarn_linear_ramp_mask(low, high, half);
    // Reference uses `(1 - ramp)` as the keep-extra mask, so:
    // inv_freq = freq_inter * (1 - mask) + freq_extra * mask
    //          = freq_inter * ramp + freq_extra * (1 - ramp)  [mask = 1-ramp]
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        let mask_extra = 1.0 - ramp[i];
        inv_freq
            .push(freq_inter[i] * ramp[i] + freq_extra[i] * mask_extra);
    }
    inv_freq
}

/// Errors specific to the YaRN RoPE port.
#[derive(Debug, thiserror::Error)]
pub enum YarnError {
    #[error("position must be non-negative (got {pos})")]
    NegativePos { pos: i32 },
    #[error("buffer length {got} != num_heads * rotary_dim ({expected})")]
    BufLen { got: usize, expected: usize },
    #[error("inv_freq length {got} != rotary_dim/2 ({expected})")]
    InvFreqLen { got: usize, expected: usize },
}

/// Apply YaRN RoPE to a per-head rope-half buffer of shape
/// `[num_heads, rotary_dim]`. Mutates `x` in place. The mscale
/// multiplier is applied to both cos and sin (per
/// `modeling_deepseek.py::DeepseekV3YarnRotaryEmbedding._set_cos_sin_cache`).
///
/// Pairing convention matches MLX (`x[i]` paired with `x[i + half]`),
/// since moeflux's existing vanilla RoPE uses that convention and the
/// MLX-converted weights are pre-arranged for it.
pub fn apply_rotary_emb_yarn(
    pos: i32,
    x: &mut [f32],
    rotary_dim: usize,
    inv_freq: &[f32],
    mscale: f32,
) -> Result<(), YarnError> {
    if pos < 0 {
        return Err(YarnError::NegativePos { pos });
    }
    let half = rotary_dim / 2;
    if inv_freq.len() != half {
        return Err(YarnError::InvFreqLen {
            got: inv_freq.len(),
            expected: half,
        });
    }
    if x.len() % rotary_dim != 0 {
        return Err(YarnError::BufLen {
            got: x.len(),
            expected: rotary_dim,
        });
    }
    let num_heads = x.len() / rotary_dim;
    let pos_f = pos as f32;
    for h in 0..num_heads {
        let xh = &mut x[h * rotary_dim..(h + 1) * rotary_dim];
        for i in 0..half {
            let angle = pos_f * inv_freq[i];
            // SAFETY: scalar libm; well-formed f32 inputs.
            let cos_a = unsafe { cosf(angle) } * mscale;
            let sin_a = unsafe { sinf(angle) } * mscale;
            let x0 = xh[i];
            let x1 = xh[i + half];
            xh[i] = x0 * cos_a - x1 * sin_a;
            xh[i + half] = x0 * sin_a + x1 * cos_a;
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

    /// `yarn_get_mscale(scale=1, mscale=*)` is identity. Catches a
    /// math sign / branch bug.
    #[test]
    fn yarn_mscale_is_identity_at_scale_one() {
        assert_eq!(yarn_get_mscale(1.0, 1.0), 1.0);
        assert_eq!(yarn_get_mscale(1.0, 5.0), 1.0);
        assert_eq!(yarn_get_mscale(0.5, 5.0), 1.0);
    }

    /// `yarn_get_mscale_full(scale=1, ...) = 1.0` since both the
    /// numerator and denominator collapse to identity.
    #[test]
    fn yarn_mscale_full_is_identity_at_scale_one() {
        assert_eq!(yarn_get_mscale_full(1.0, 1.0, 1.0), 1.0);
    }

    /// At YaRN factor = 1 (no extension) the inv_freq table must
    /// equal the vanilla RoPE table `1 / base^(2i/dim)` exactly.
    /// This is the "no-op" sanity check — if we're ever uncertain
    /// about the YaRN math, this should hold by construction.
    #[test]
    fn yarn_inv_freq_at_factor_one_collapses_to_vanilla() {
        let dim = 64;
        let base = 10_000.0f32;
        let factor = 1.0f32;
        let original_max = 4096.0f32;
        let beta_fast = 32.0f32;
        let beta_slow = 1.0f32;

        let inv = compute_yarn_inv_freq(
            dim,
            base,
            factor,
            original_max,
            beta_fast,
            beta_slow,
        );
        assert_eq!(inv.len(), dim / 2);
        for i in 0..dim / 2 {
            let expected =
                1.0 / unsafe { powf(base, (2 * i) as f32 / dim as f32) };
            // factor=1 makes freq_extra == freq_inter, so the ramp
            // blend is the identity. Tolerance for f32 powf drift.
            let diff = (inv[i] - expected).abs();
            assert!(
                diff < 1e-7 * expected.max(1e-30),
                "inv[{i}] = {} vs vanilla {}",
                inv[i],
                expected,
            );
        }
    }

    /// Cogito-V2 inv_freq table is monotonically decreasing across
    /// channels (high-freq dims have larger inv_freq). Sanity check
    /// that we didn't reverse the ramp direction.
    #[test]
    fn yarn_inv_freq_is_monotone_decreasing() {
        let inv = compute_yarn_inv_freq(
            64,    // qk_rope_head_dim
            10_000.0,
            40.0,  // factor
            4096.0,
            32.0,  // beta_fast
            1.0,   // beta_slow
        );
        for i in 1..inv.len() {
            assert!(
                inv[i] < inv[i - 1] || (inv[i] - inv[i - 1]).abs() < 1e-12,
                "inv_freq not monotone at i={i}: {} vs {}",
                inv[i - 1],
                inv[i]
            );
        }
    }

    /// `apply_rotary_emb_yarn` at pos=0 with mscale=1 is the
    /// identity: cos(0)*1 = 1, sin(0)*1 = 0.
    #[test]
    fn yarn_rope_at_pos_zero_mscale_one_is_identity() {
        let rotary_dim = 64;
        let num_heads = 4;
        let inv_freq = compute_yarn_inv_freq(
            rotary_dim,
            10_000.0,
            40.0,
            4096.0,
            32.0,
            1.0,
        );
        let mut x: Vec<f32> = (0..num_heads * rotary_dim)
            .map(|i| i as f32 * 0.01)
            .collect();
        let x_orig = x.clone();

        apply_rotary_emb_yarn(0, &mut x, rotary_dim, &inv_freq, 1.0).unwrap();

        for i in 0..x.len() {
            assert!(
                (x[i] - x_orig[i]).abs() < 1e-6,
                "x[{i}] changed: {} → {}",
                x_orig[i],
                x[i]
            );
        }
    }
}
