//! Small CPU helpers used by the CPU-combine fallback in the
//! deferred-experts state machine. Mirror the equivalent C-side
//! primitives one-for-one (`cpu_vec_madd`, `cpu_sigmoid` —
//! `infer.m:~2300..2350`); the FMA contraction site uses `mul_add` in
//! line with the LM-head finding (slice 6) so the matvec on either
//! side fuses identically.

/// `dst[i] += src[i] * scale`. Mirrors C `cpu_vec_madd`. Uses
/// `mul_add` so clang's default `-ffp-contract=on` and Rust's
/// explicit FMA produce the same instruction sequence.
#[inline]
pub fn cpu_vec_madd(dst: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = s.mul_add(scale, *d);
    }
}

/// Standard sigmoid `1 / (1 + exp(-x))`. Scalar; no SIMD lowering on
/// either C or Rust because it's used once per layer (shared-expert
/// gate scoring), not in a hot loop.
#[inline]
pub fn cpu_sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
