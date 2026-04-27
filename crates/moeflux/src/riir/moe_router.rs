//! MoE router — Rust port of `infer.m`'s `cpu_softmax` +
//! `cpu_topk` + `cpu_normalize_weights` pipeline.
//!
//! Given the post-gate-matvec gate scores (`NUM_EXPERTS` raw logits),
//! pick the top-K experts and emit them with normalized weights:
//!
//! 1. Softmax the score vector in place.
//! 2. Selection-sort top-K (selects K largest values; output slot
//!    order matches the C `cpu_topk` — values land in the slot that
//!    was the running minimum at the moment of replacement, NOT
//!    sorted by score).
//! 3. Normalize the K weights to sum to 1.
//!
//! Returns parallel arrays of `K` expert indices and `K` weights.
//!
//! ## Tolerance regime
//!
//! ULP-bounded: softmax calls `expf` per element and Apple clang at
//! `-O3` auto-vectorizes those calls into Apple libm's `vexpf`, while
//! Rust extern-`"C"` calls into libm don't vectorize. Same drift
//! shape RoPE / SDPA already characterized — bounded per call,
//! single-digit ULPs for typical inputs. Top-K selection is bit-exact
//! given identical softmax inputs, but inherits whatever drift the
//! softmax produced; normalization is one further sum + divide.
//!
//! In practice the per-element ULP drift on softmax is far below the
//! magnitude separation between adjacent expert scores, so the
//! selected index *set* is bit-exact across both sides. The diff
//! test asserts set equality after sorting by index, then compares
//! weights with a small absolute tolerance.

/// Errors specific to the MoE router port.
#[derive(Debug, thiserror::Error)]
pub enum MoeRouterError {
    #[error(
        "k {k} must satisfy 1 ≤ k ≤ scores.len() ({n})"
    )]
    BadK { k: usize, n: usize },
    #[error("scores empty")]
    EmptyScores,
    #[error(
        "indices length {got} != k {k}"
    )]
    IndicesLen { got: usize, k: usize },
    #[error(
        "weights length {got} != k {k}"
    )]
    WeightsLen { got: usize, k: usize },
}

/// Softmax in place over the entire `x`. Matches the C `cpu_softmax`
/// reduction order exactly: max → subtract & exp → sum → multiply by
/// `1/sum`. Uses `f32::exp` (libm `expf` on macOS).
///
/// Public so the linear-attention port (next slice) can reuse the same
/// helper for its own softmax over score vectors. Operates on `&mut [f32]`
/// for symmetry with the C signature.
pub fn softmax(x: &mut [f32]) -> Result<(), MoeRouterError> {
    if x.is_empty() {
        return Err(MoeRouterError::EmptyScores);
    }
    let mut max_val = x[0];
    for &v in &x[1..] {
        if v > max_val {
            max_val = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv_sum = 1.0f32 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
    Ok(())
}

/// Select the K largest values from `scores` into parallel
/// `(indices, values)` arrays. Slot order matches the C
/// `cpu_topk` selection-sort: each new winner overwrites the running
/// minimum slot, so the output is *not* sorted by score. Indices and
/// values both have length `k`.
///
/// Tie-breaking: the C uses strict `>`, so when two scores are equal
/// the earlier index wins (later equal values fail the `> values[min_k]`
/// check). This Rust port keeps the same comparison.
pub fn topk(
    scores: &[f32],
    k: usize,
    indices: &mut [i32],
    values: &mut [f32],
) -> Result<(), MoeRouterError> {
    let n = scores.len();
    if k == 0 || k > n {
        return Err(MoeRouterError::BadK { k, n });
    }
    if indices.len() != k {
        return Err(MoeRouterError::IndicesLen {
            got: indices.len(),
            k,
        });
    }
    if values.len() != k {
        return Err(MoeRouterError::WeightsLen {
            got: values.len(),
            k,
        });
    }

    for slot in 0..k {
        values[slot] = -1e30f32;
        indices[slot] = 0;
    }

    for (i, &score) in scores.iter().enumerate() {
        let mut min_k = 0usize;
        for slot in 1..k {
            if values[slot] < values[min_k] {
                min_k = slot;
            }
        }
        if score > values[min_k] {
            values[min_k] = score;
            indices[min_k] = i as i32;
        }
    }
    Ok(())
}

/// Normalize the K-element weight vector to sum to 1 in place.
/// Matches the C `cpu_normalize_weights`: a guarded `sum > 0` divide
/// (no-op if all weights are non-positive — defensive but unreachable
/// for softmax outputs in practice).
pub fn normalize_weights(weights: &mut [f32]) {
    let mut sum = 0.0f32;
    for &w in weights.iter() {
        sum += w;
    }
    if sum > 0.0 {
        let inv = 1.0f32 / sum;
        for w in weights.iter_mut() {
            *w *= inv;
        }
    }
}

/// Full router pipeline: softmax → top-K → normalize. `scores` is
/// `NUM_EXPERTS` raw gate logits (mutated in place — softmax happens
/// on this buffer). `indices` and `weights` are output parallel arrays
/// of length `k`. Convenience wrapper composing the three primitives.
pub fn moe_router_cpu(
    scores: &mut [f32],
    k: usize,
    indices: &mut [i32],
    weights: &mut [f32],
) -> Result<(), MoeRouterError> {
    softmax(scores)?;
    topk(scores, k, indices, weights)?;
    normalize_weights(weights);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_normalizes_to_one() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        softmax(&mut x).unwrap();
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {sum}");
    }

    #[test]
    fn topk_picks_largest() {
        let scores = [0.1f32, 0.5, 0.3, 0.9, 0.2];
        let mut idx = [0i32; 3];
        let mut val = [0.0f32; 3];
        topk(&scores, 3, &mut idx, &mut val).unwrap();
        let mut pairs: Vec<(i32, f32)> =
            idx.iter().copied().zip(val.iter().copied()).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert_eq!(pairs[0].0, 3); // 0.9
        assert_eq!(pairs[1].0, 1); // 0.5
        assert_eq!(pairs[2].0, 2); // 0.3
    }

    #[test]
    fn normalize_sums_to_one() {
        let mut w = [0.5f32, 1.5, 2.0];
        normalize_weights(&mut w);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "normalize sum = {sum}");
    }
}
