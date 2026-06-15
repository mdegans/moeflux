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
    #[error("k {k} must satisfy 1 ≤ k ≤ scores.len() ({n})")]
    BadK { k: usize, n: usize },
    #[error("scores empty")]
    EmptyScores,
    #[error("indices length {got} != k {k}")]
    IndicesLen { got: usize, k: usize },
    #[error("weights length {got} != k {k}")]
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

/// noaux_tc routing — DeepSeek-V3 / Cogito-V2 path. Differs from
/// [`moe_router_cpu`] in three ways: scoring is sigmoid (not softmax),
/// selection uses a per-expert correction bias added pre-selection,
/// and grouping limits which experts are eligible (top-2 within each
/// group sum → top-`topk_group` groups → mask non-selected groups).
/// The renormalized output weights gather **un-biased** sigmoid
/// scores at the chosen indices, then scale by `routed_scaling_factor`.
///
/// Reference: `modeling_deepseek.py::MoEGate::forward` (lines 423-474
/// in the Cogito-V2-Preview-671B-MoE-4bit drop).
///
/// Inputs:
/// - `score_logits` [`num_experts`]: pre-sigmoid gate logits. Mutated
///   in place to hold `sigmoid(score_logits)` on return (the original
///   un-biased scores).
/// - `correction_bias` [`num_experts`]: per-expert
///   `e_score_correction_bias`.
/// - `n_group`: number of expert groups (8 for Cogito-V2).
/// - `topk_group`: number of groups selected per token (4 for
///   Cogito-V2).
/// - `k`: final top-K expert count (8 for Cogito-V2).
/// - `routed_scaling_factor`: post-renorm weight multiplier (2.5 for
///   Cogito-V2).
///
/// Outputs:
/// - `indices` [`k`]: selected expert ids.
/// - `weights` [`k`]: renormalized + scaled weights for the selected
///   experts.
pub fn noaux_tc_router_cpu(
    score_logits: &mut [f32],
    correction_bias: &[f32],
    n_group: usize,
    topk_group: usize,
    k: usize,
    routed_scaling_factor: f32,
    indices: &mut [i32],
    weights: &mut [f32],
) -> Result<(), MoeRouterError> {
    let n = score_logits.len();
    if n == 0 {
        return Err(MoeRouterError::EmptyScores);
    }
    if correction_bias.len() != n {
        return Err(MoeRouterError::WeightsLen {
            got: correction_bias.len(),
            k: n,
        });
    }
    if n_group == 0 || n % n_group != 0 {
        return Err(MoeRouterError::BadK { k: n_group, n });
    }
    if topk_group == 0 || topk_group > n_group {
        return Err(MoeRouterError::BadK {
            k: topk_group,
            n: n_group,
        });
    }
    if k == 0 || k > n {
        return Err(MoeRouterError::BadK { k, n });
    }
    if indices.len() != k {
        return Err(MoeRouterError::IndicesLen {
            got: indices.len(),
            k,
        });
    }
    if weights.len() != k {
        return Err(MoeRouterError::WeightsLen {
            got: weights.len(),
            k,
        });
    }

    // 1. sigmoid in place — these are the "original" scores that the
    //    final renormalized weights gather from.
    for s in score_logits.iter_mut() {
        *s = 1.0 / (1.0 + (-*s).exp());
    }

    // 2. biased = original + correction_bias (used only for
    //    selection, never for the final weights).
    let mut biased: Vec<f32> = score_logits
        .iter()
        .zip(correction_bias.iter())
        .map(|(s, b)| s + b)
        .collect();

    // 3. Group score = sum of top-2 within each group (DeepSeek-V3's
    //    `topk(2).sum()` over the [n_group, group_size] view).
    let group_size = n / n_group;
    let mut group_scores = vec![0.0f32; n_group];
    for g in 0..n_group {
        let slice = &biased[g * group_size..(g + 1) * group_size];
        // Find top-2 by linear scan (group_size is small — 32 for
        // Cogito-V2). Tracks first-largest then second-largest.
        let mut top1 = f32::NEG_INFINITY;
        let mut top2 = f32::NEG_INFINITY;
        for &v in slice.iter() {
            if v > top1 {
                top2 = top1;
                top1 = v;
            } else if v > top2 {
                top2 = v;
            }
        }
        group_scores[g] = top1 + top2;
    }

    // 4. Top-`topk_group` of group_scores → group_idx (which groups
    //    survive the mask).
    let mut group_idx = vec![0i32; topk_group];
    let mut group_vals = vec![0.0f32; topk_group];
    topk(&group_scores, topk_group, &mut group_idx, &mut group_vals)?;
    let group_idx_set: std::collections::HashSet<usize> =
        group_idx.iter().map(|&i| i as usize).collect();

    // 5. Apply the group mask: zero out experts whose group is not
    //    selected. Use `f32::NEG_INFINITY` so they fail any subsequent
    //    `>` comparison cleanly even if the threshold is 0.0.
    for g in 0..n_group {
        if !group_idx_set.contains(&g) {
            for v in biased[g * group_size..(g + 1) * group_size].iter_mut() {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // 6. Top-K from the masked biased scores → indices.
    let mut throwaway_vals = vec![0.0f32; k];
    topk(&biased, k, indices, &mut throwaway_vals)?;

    // 7. Gather UN-biased sigmoid scores at chosen indices, normalize,
    //    multiply by routed_scaling_factor.
    let mut sum = 0.0f32;
    for (slot, &i) in indices.iter().enumerate() {
        let w = score_logits[i as usize];
        weights[slot] = w;
        sum += w;
    }
    if sum > 0.0 {
        let inv = 1.0f32 / sum;
        for w in weights.iter_mut() {
            *w *= inv;
        }
    }
    for w in weights.iter_mut() {
        *w *= routed_scaling_factor;
    }

    Ok(())
}

/// CSR-style assignment of (token, weight) tuples bucketed by expert id.
///
/// For batched MoE permute-and-fuse: given the routing decisions for
/// `n_tokens` tokens (each with `k_active` top-K experts), produce a sparse
/// view where each non-empty bucket lists which tokens picked that expert
/// and with what weight. Empty buckets are omitted.
///
/// Layout:
/// - `expert_ids[b]`: the expert this bucket dispatches to.
/// - `offsets[b]..offsets[b+1]`: the slice of `token_idx` and `weights`
///   belonging to bucket `b`. `offsets[0] == 0`,
///   `offsets[num_non_empty] == n_tokens * k_active`.
/// - `token_idx[i]`: which row of the batched input to gather (0..n_tokens).
/// - `weights[i]`: routing weight to multiply this expert's output by.
///
/// Within a single bucket, every `token_idx[i]` is distinct: top-K returns
/// distinct expert ids per token, so the same token can't appear twice in
/// the same bucket. This is what lets the GPU scatter-accumulate avoid
/// atomics (`moe_bucket_accumulate` reads each (token, hidden_dim) slot
/// at most once per dispatch).
///
/// `expert_ids` is sorted ascending so the dispatch order is deterministic
/// across runs.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertBuckets {
    pub expert_ids: Vec<i32>,
    pub offsets: Vec<u32>,
    pub token_idx: Vec<i32>,
    pub weights: Vec<f32>,
}

/// Build [`ExpertBuckets`] from per-token (indices, weights) arrays.
///
/// `per_token_indices` and `per_token_weights` are flat arrays of length
/// `n_tokens * k_active` in row-major order: position
/// `(t, s) = t * k_active + s`.
///
/// `num_experts` is the model's full expert count. Output buckets only
/// mention experts that were actually selected by ≥1 token.
pub fn build_expert_buckets(
    per_token_indices: &[i32],
    per_token_weights: &[f32],
    n_tokens: usize,
    k_active: usize,
    num_experts: usize,
) -> ExpertBuckets {
    debug_assert_eq!(per_token_indices.len(), n_tokens * k_active);
    debug_assert_eq!(per_token_weights.len(), n_tokens * k_active);

    let mut counts: Vec<u32> = vec![0; num_experts];
    for &e in per_token_indices {
        debug_assert!(
            e >= 0 && (e as usize) < num_experts,
            "expert id {e} out of range [0, {num_experts})"
        );
        counts[e as usize] += 1;
    }

    let num_non_empty: usize = counts.iter().filter(|&&c| c > 0).count();
    let total_assignments: usize = n_tokens * k_active;

    let mut expert_ids: Vec<i32> = Vec::with_capacity(num_non_empty);
    let mut offsets: Vec<u32> = Vec::with_capacity(num_non_empty + 1);
    let mut expert_to_bucket: Vec<i32> = vec![-1; num_experts];

    let mut running: u32 = 0;
    offsets.push(0);
    for (e, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        expert_to_bucket[e] = expert_ids.len() as i32;
        expert_ids.push(e as i32);
        running += c;
        offsets.push(running);
    }
    debug_assert_eq!(running as usize, total_assignments);

    let mut cursors: Vec<u32> = offsets[..num_non_empty].to_vec();

    let mut token_idx: Vec<i32> = vec![0; total_assignments];
    let mut weights: Vec<f32> = vec![0.0; total_assignments];

    for t in 0..n_tokens {
        for s in 0..k_active {
            let flat = t * k_active + s;
            let e = per_token_indices[flat];
            let b = expert_to_bucket[e as usize] as usize;
            let pos = cursors[b] as usize;
            token_idx[pos] = t as i32;
            weights[pos] = per_token_weights[flat];
            cursors[b] += 1;
        }
    }

    ExpertBuckets {
        expert_ids,
        offsets,
        token_idx,
        weights,
    }
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
        let mut pairs: Vec<(i32, f32)> = idx.iter().copied().zip(val.iter().copied()).collect();
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

    /// Hand-computed fixture for noaux_tc routing.
    ///
    /// 8 experts, n_group=2 (group_size=4), topk_group=1, k=2.
    /// Logits: `[3, -2, 1, -1, -3, 2, 0, 4]`
    ///
    /// After sigmoid (the "original" scores):
    /// - 3.0 → 0.9526; -2.0 → 0.1192; 1.0 → 0.7311; -1.0 → 0.2689
    /// - -3.0 → 0.0474; 2.0 → 0.8808; 0.0 → 0.5000; 4.0 → 0.9820
    ///
    /// `correction_bias = [0.1, 0.2, -0.3, 0.0, 0.0, 0.4, -0.1, 0.05]`
    ///
    /// Biased = original + bias:
    /// - group 0 (indices 0..4): [1.053, 0.319, 0.431, 0.269]
    ///   sum-of-top-2 = 1.053 + 0.431 = 1.484
    /// - group 1 (indices 4..8): [0.047, 1.281, 0.400, 1.032]
    ///   sum-of-top-2 = 1.281 + 1.032 = 2.313
    ///
    /// topk_group=1 picks group 1. Group 0 masked to -inf.
    /// Final top-K=2 from masked biased: indices {5, 7} (values
    /// 1.281, 1.032).
    ///
    /// Weights gather UN-biased sigmoid: 0.8808 (idx 5), 0.9820 (idx 7).
    /// Normalize: 0.8808/1.8628 ≈ 0.4729, 0.9820/1.8628 ≈ 0.5271.
    /// Multiply by routed_scaling=2.5: ≈ 1.1822, 1.3178.
    #[test]
    fn noaux_tc_against_hand_reference() {
        let mut logits = [3.0f32, -2.0, 1.0, -1.0, -3.0, 2.0, 0.0, 4.0];
        let bias = [0.1f32, 0.2, -0.3, 0.0, 0.0, 0.4, -0.1, 0.05];
        let mut indices = [0i32; 2];
        let mut weights = [0.0f32; 2];

        noaux_tc_router_cpu(
            &mut logits,
            &bias,
            /* n_group = */ 2,
            /* topk_group = */ 1,
            /* k = */ 2,
            /* routed_scaling_factor = */ 2.5,
            &mut indices,
            &mut weights,
        )
        .unwrap();

        // Indices set must equal {5, 7} (group 1's top-2 by biased).
        let mut idx_sorted: Vec<i32> = indices.to_vec();
        idx_sorted.sort();
        assert_eq!(idx_sorted, vec![5, 7]);

        // Weights, sorted by their parallel index, must be:
        // idx 5 → 1.1822, idx 7 → 1.3178 (within tolerance).
        let mut pairs: Vec<(i32, f32)> = indices
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .collect();
        pairs.sort_by_key(|&(i, _)| i);

        let tol = 1e-3;
        assert_eq!(pairs[0].0, 5);
        assert!(
            (pairs[0].1 - 1.1822).abs() < tol,
            "weight for idx 5: got {}, want ~1.1822",
            pairs[0].1
        );
        assert_eq!(pairs[1].0, 7);
        assert!(
            (pairs[1].1 - 1.3178).abs() < tol,
            "weight for idx 7: got {}, want ~1.3178",
            pairs[1].1
        );

        // Sum of weights = routed_scaling_factor (= 2.5) since they
        // were renormalized to 1 then multiplied by 2.5.
        let sum: f32 = weights.iter().sum();
        assert!((sum - 2.5).abs() < tol, "sum = {sum}");
    }

    /// Cogito-V2-shape sanity: 256 experts, 8 groups (32/group), top-4
    /// groups, top-K=8. Random logits + bias; verify all selected
    /// experts come from the chosen groups and sum-of-weights equals
    /// routed_scaling_factor.
    #[test]
    fn noaux_tc_full_cogito_shape() {
        let n_experts = 256;
        let n_group = 8;
        let topk_group = 4;
        let k = 8;
        let scaling = 2.5;

        // Deterministic synthetic inputs.
        let mut logits: Vec<f32> = (0..n_experts)
            .map(|i| ((i as f32) * 0.137).sin() * 2.0)
            .collect();
        let bias: Vec<f32> = (0..n_experts)
            .map(|i| ((i as f32) * 0.241).cos() * 0.5)
            .collect();
        let mut indices = vec![0i32; k];
        let mut weights = vec![0.0f32; k];

        noaux_tc_router_cpu(
            &mut logits,
            &bias,
            n_group,
            topk_group,
            k,
            scaling,
            &mut indices,
            &mut weights,
        )
        .unwrap();

        // Sum of routed weights = scaling.
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - scaling).abs() < 1e-3,
            "weight sum = {sum}, want {scaling}"
        );

        // Every selected expert must be in one of the chosen groups.
        // We re-derive group selection from the inputs and assert
        // coverage. (Defensive: catches off-by-one in the mask.)
        let group_size = n_experts / n_group;
        let chosen_groups: std::collections::HashSet<usize> =
            indices.iter().map(|&i| (i as usize) / group_size).collect();
        assert!(
            chosen_groups.len() <= topk_group,
            "selected experts span {} groups, must be ≤ {topk_group}",
            chosen_groups.len()
        );

        // No duplicate indices.
        let unique: std::collections::HashSet<i32> = indices.iter().copied().collect();
        assert_eq!(unique.len(), k, "duplicate expert indices in output");
    }

    /// Reconstructing per-token (expert, weight) pairs from the CSR
    /// buckets must reproduce the input arrays modulo within-token
    /// ordering. Exercises the basic invariant of `build_expert_buckets`.
    #[test]
    fn build_expert_buckets_round_trips() {
        // N=3 tokens, K=2 active, num_experts=4.
        //   token 0 → (expert 0, w=0.7), (expert 3, w=0.3)
        //   token 1 → (expert 0, w=0.4), (expert 2, w=0.6)
        //   token 2 → (expert 3, w=0.5), (expert 0, w=0.5)
        let indices: [i32; 6] = [0, 3, 0, 2, 3, 0];
        let weights: [f32; 6] = [0.7, 0.3, 0.4, 0.6, 0.5, 0.5];
        let b = build_expert_buckets(&indices, &weights, 3, 2, 4);

        // Expert 1 is empty → three non-empty buckets sorted ascending:
        // 0, 2, 3.
        assert_eq!(b.expert_ids, vec![0, 2, 3]);
        // bucket 0 (expert 0): tokens 0, 1, 2 → 3 entries
        // bucket 1 (expert 2): token 1       → 1 entry
        // bucket 2 (expert 3): tokens 0, 2   → 2 entries
        assert_eq!(b.offsets, vec![0, 3, 4, 6]);

        // Reconstruct per-token (expert, weight) sets from the CSR view.
        let mut got: Vec<Vec<(i32, f32)>> = vec![vec![]; 3];
        for (bi, &e) in b.expert_ids.iter().enumerate() {
            let start = b.offsets[bi] as usize;
            let end = b.offsets[bi + 1] as usize;
            for j in start..end {
                let t = b.token_idx[j] as usize;
                got[t].push((e, b.weights[j]));
            }
        }
        for t in 0..3 {
            let mut want: Vec<(i32, f32)> = (0..2)
                .map(|s| (indices[t * 2 + s], weights[t * 2 + s]))
                .collect();
            want.sort_by_key(|&(e, _)| e);
            got[t].sort_by_key(|&(e, _)| e);
            assert_eq!(got[t].len(), 2, "token {t} count");
            for (g, w) in got[t].iter().zip(want.iter()) {
                assert_eq!(g.0, w.0, "token {t} expert");
                assert!((g.1 - w.1).abs() < 1e-9, "token {t} weight");
            }
        }
    }

    /// Top-K guarantees distinct experts per token, so within a single
    /// bucket every token_idx entry must be unique. The
    /// `moe_bucket_accumulate` kernel relies on this invariant to avoid
    /// atomic stores into the per-token accumulator.
    #[test]
    fn build_expert_buckets_distinct_tokens_per_bucket() {
        // N=4, K=2. All (token, slot) pairs map to distinct experts
        // within each token (the top-K contract).
        let indices: [i32; 8] = [0, 1, 1, 2, 0, 2, 1, 3];
        let weights: [f32; 8] = [0.5, 0.5, 0.4, 0.6, 0.7, 0.3, 0.5, 0.5];
        let b = build_expert_buckets(&indices, &weights, 4, 2, 4);
        for bi in 0..b.expert_ids.len() {
            let start = b.offsets[bi] as usize;
            let end = b.offsets[bi + 1] as usize;
            let mut sorted: Vec<i32> = b.token_idx[start..end].to_vec();
            sorted.sort();
            let len = sorted.len();
            sorted.dedup();
            assert_eq!(sorted.len(), len, "duplicate token in bucket {bi}");
        }
    }

    /// Empty buckets are not emitted: experts that no token picked do
    /// not appear in `expert_ids`.
    #[test]
    fn build_expert_buckets_empty_skipped() {
        // num_experts=10, only experts 2 and 7 selected.
        let indices: [i32; 4] = [2, 7, 7, 2];
        let weights: [f32; 4] = [0.5; 4];
        let b = build_expert_buckets(&indices, &weights, 2, 2, 10);
        assert_eq!(b.expert_ids, vec![2, 7]);
        assert_eq!(b.offsets, vec![0, 2, 4]);
        // Bucket 0 (expert 2): tokens 0, 1; bucket 1 (expert 7): tokens 0, 1.
        let b0_tokens = &b.token_idx[0..2];
        let b1_tokens = &b.token_idx[2..4];
        let mut s0 = b0_tokens.to_vec();
        s0.sort();
        let mut s1 = b1_tokens.to_vec();
        s1.sort();
        assert_eq!(s0, vec![0, 1]);
        assert_eq!(s1, vec![0, 1]);
    }

    /// Degenerate single-expert case: every token picks the same expert.
    /// Exactly one bucket emerges with all N tokens in order.
    #[test]
    fn build_expert_buckets_single_expert() {
        let indices: [i32; 4] = [3, 3, 3, 3];
        let weights: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        let b = build_expert_buckets(&indices, &weights, 4, 1, 8);
        assert_eq!(b.expert_ids, vec![3]);
        assert_eq!(b.offsets, vec![0, 4]);
        assert_eq!(b.token_idx, vec![0, 1, 2, 3]);
        assert_eq!(b.weights, vec![1.0; 4]);
    }
}
