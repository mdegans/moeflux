---
name: moeflux-hardening-session-c-plan
description: Session C plan — engine-level diff harness. New test `crates/moeflux/tests/engine_op_diff.rs` runs `Op::MoeBatchedPermuteFuse` vs `Op::MoeGatherIdFuse` against real Qwen3-A3B layer weights, asserts bit-exact. ~270 LOC one file, ~1.5h session. Designed by Plan agent in session 20.
metadata:
  type: project
---

# TL;DR

A new `#[ignore]`-gated integration test at `crates/moeflux/tests/engine_op_diff.rs` that diffs `Op::MoeBatchedPermuteFuse` vs `Op::MoeGatherIdFuse` against real Qwen3-A3B layer weights + real GPU router output. Asserts bit-exact equality with rich per-token-per-channel diagnostic on failure. Reverting `linear_attn_forward.rs:2802` from `h_post_id` to `h_mid_id` MUST cause this test to fail with a clear signal.

Single new file, ~270 LOC, ~1.5h execution.

# Decisions

## 1. Test setup — real model, one layer, lazy

Load real Qwen3.6-A3B from `/Volumes/Temp Backup/models/blallama`, build `LayerWeightCache::build(layer_idx, &wf, &wf_buf)` for **one MoE layer**, use real `ExpertFiles::open(...).attach_to_device(pool)`.

**Rationale.** Synthetic W4_gs64 weights would pass the diff trivially even with bad wiring (random-norm × random-pre-norm cosines fine with random weights). Real weights make the post-norm-vs-pre-norm value distribution actually different — which is what gave session 19 its garbage logits. Cost: one-layer load ~tens of ms after page-cache warm. Behind `#[ignore]` so never in default `cargo test`.

## 2. A/B mechanism — explicit graph construction

Build two `Graph`s in the same process — one with `Op::MoeBatchedPermuteFuse`, one with `Op::MoeGatherIdFuse` — run each against the same boundary buffers. No env-flag toggling.

**Rationale.** Env-flag is `OnceLock`-cached, process-global, can't toggle mid-run. Explicit graphs sidestep that *and* read literally as "here are the two Ops we want to diff."

## 3. Comparison granularity — per-token-per-channel max-abs + assert bit-exact

Compute and *print* per-token max-abs-error and global max-abs-error with `(token, channel)` location; then `assert_eq!` byte-for-byte.

**Rationale.** Session 20 verified the two paths are bit-identical in production at temp=0+seed=42. Bit-exact is the contract. But `assert_eq!([u8; N], [u8; N])` produces useless failure output — rich diagnostic first, assert second. Fallback floor cosine ≥ 0.9999 logged but not asserted; if bit-exact ever drifts (Metal driver change), cosine tells us drift vs wiring.

## 4. Number of layers — one × one chunk

Single MoE layer (first MoE layer per `VARIANT.layer_kind`), n_tokens=64, k_active=8 (production prefill chunk). No multi-chunk sweep.

**Rationale.** Bug class is wiring (layer-invariant). One MoE layer with realistic dims sufficient. Chunk-shape bugs already covered by kernel oracle `down_shape` / `a3b_scale_*`.

## 5. Routing distribution — real GPU router on synthetic seeded input

Run the *real GPU router* (`encode_moe_router`) on a synthetic-but-deterministic `h_post` (seeded xorshift) to produce indices + weights. Use those as input to both Ops.

**Rationale.** Kernel oracle uses random-distinct uniform routing — exactly what missed session 19's bug. Real router output has the distributional properties (some experts hot, htpe early-return paths exercised) that production cares about, without needing real text.

## 6. Runtime budget — `#[ignore]`-gated, ~5s warm, never in CI

```
cargo test -p moeflux --no-default-features \
    --features model-qwen3-6-35b-a3b --release \
    --test engine_op_diff -- --ignored --nocapture
```

Mike runs locally before any MoE-block producer change (complement to [[feedback-coherence-test-before-pipeline-commit]]). 50GB model can't live on CI. Document invocation in doc-comment header.

## 7. Failure signal — rich diagnostic + assert_eq

Before assert:
- Header: `[engine-op-diff] layer={L} n_tokens={N} k_active={K} hidden={H}`
- Per-token max-abs with worst channel, capped at 10 rows
- Global: `max_abs_diff = X.XXXe-N at (t=T, c=C); cosine = Y.YYYYYY`
- Assert: bit-exact

When session-20 fix reverted, report shows large max-abs at every token — Mike immediately sees scale to confirm bug class vs Metal-driver noise.

## 8. File location — `crates/moeflux/tests/engine_op_diff.rs`

Parallels `graph_diff_oracle.rs` / `batched_diff_oracle.rs`. "engine" signals real-engine-state; "op_diff" signals Op-pair level. Re-uses `mod common; use common::diff_helpers::*;`.

## 9. Re-usability — one-Op-pair test, not generic harness

Minimal scope. No second Op pair queued up. Internal structure clean (private `run_one_layer_op_diff(layer_idx, n_tokens, seed)` helper) so hypothetical second test = ~50 LOC. No trait gymnastics today.

## 10. Migration shape — one commit, harness + test together

~250-350 LOC in single file in `tests/`. No separate scaffolding commit.

# Test code structure

```rust
//! `Op::MoeBatchedPermuteFuse` vs `Op::MoeGatherIdFuse` — engine-level
//! diff against real Qwen3-A3B layer weights + real router output.
//!
//! Catches producer-wiring bugs of the session-19 class. The kernel-
//! level oracle (`crates/moeflux-metal/tests/gather_mm_id_diff.rs`)
//! uses synthetic everything; this test uses real mmap'd expert
//! weights and the real GPU router on synthetic (seeded) hidden states.
//!
//! Reverting `linear_attn_forward.rs:2802` from `h_post_id` (or
//! whatever the post-rename equivalent is) back to `h_mid_id` MUST
//! cause this test to fail.
//!
//! Run:
//!   cargo test -p moeflux --no-default-features \
//!       --features model-qwen3-6-35b-a3b --release \
//!       --test engine_op_diff -- --ignored --nocapture
#![cfg(all(target_os = "macos", feature = "model-qwen3-6-35b-a3b"))]

mod common;
use common::diff_helpers::{cosine_sim, default_a3b_paths, A3BPaths};

struct Rng(u64);  // deterministic xorshift, same shape as graph_diff_oracle.rs

fn run_one_layer_op_diff(layer_idx: usize, n_tokens: usize, seed: u64);
fn report_diff(layer_idx: usize, n_tokens: usize, hidden: usize, a: &[f32], b: &[f32]);
fn first_moe_layer_idx() -> usize;

#[test]
#[ignore = "requires real Qwen3-6-A3B weights on /Volumes/Temp Backup"]
fn moe_gather_id_matches_batched_permute_fuse_engine_level() {
    let layer_idx = first_moe_layer_idx();
    run_one_layer_op_diff(layer_idx, /* n_tokens */ 64, /* seed */ 0xA3B_C0DE);
}
```

# `run_one_layer_op_diff` steps

1. **Load real weights.** `WeightFile::open` + `ExpertFiles::open` + `MetalBackend::new` + `MtlWeightBuf::wrap` + `ef.attach_to_device(pool)` + `LayerWeightCache::build(layer_idx, &wf, &wf_buf)`.
2. **Allocate `MoeGraphScratch`** (two instances — one per path, to avoid `commit_planned` latch conflict). Also two `hidden_out_{A,B}` buffers.
3. **Build synthetic input.** Seeded xorshift fills `h_post_host`, `h_mid_host`, `shared_gate_host`. Upload to each scratch's `h_post` / `h_mid` / `shared_gate`.
4. **Real GPU router → routing tables.** Allocate `router_logits` BufId of `[n_tokens, n_experts]`, fill seeded, dispatch `encode_moe_router(...)` → populates `moe.routing_indices` and `moe.routing_weights`. One dispatch.
5. **Bucket prep (CPU-side).** Mirror `moe_block_forward:2660-2716`. Both paths need uploads — Path A uses bucket_input; Path B doesn't but extra upload is harmless.
6. **Run Path A graph.** Mirror lines 2722-2842 forcing the `else`-branch (`MoeBatchedPermuteFuse`), target `hidden_out_A`. `backend.execute(&g, "engine_diff_A")`.
7. **Run Path B graph.** Same shape forcing the `if`-branch (`MoeGatherIdFuse`), target `hidden_out_B`.
8. **Download + diff.**
   ```rust
   let af: Vec<f32> = bytemuck::cast_slice(&a).to_vec();
   let bf: Vec<f32> = bytemuck::cast_slice(&b).to_vec();
   report_diff(layer_idx, n_tokens, hidden, &af, &bf);
   assert_eq!(a, b, "bit-exact A vs B mismatch — see diagnostic above");
   ```
9. **`report_diff`** — ~30 LOC. Per-token loop, accumulate `(max_abs, t, c)` + top-10 worst tokens, print global cosine + max-abs-diff location. Eprintln only, no asserts.

# LOC estimate

| File | LOC | Notes |
|---|---|---|
| `crates/moeflux/tests/engine_op_diff.rs` | 260-320 | header doc (~25), Rng (~25), `first_moe_layer_idx` (~10), `run_one_layer_op_diff` (~200), `report_diff` (~35), `#[test]` (~10) |
| `crates/moeflux/tests/common/diff_helpers.rs` | +0 to +15 | Optional: `f32_chunks_to_max_abs(t, c)` helper if worth sharing |

**Total new code: ~270 LOC.** Single file, single test, single commit. **No production-code changes required** (all relevant items already `pub`).

# Risk surface

| Risk | Likelihood | Detection | Mitigation |
|---|---|---|---|
| `MoeGraphScratch::commit_plan` latch — calling twice in one process may break | Medium | Path B panics / silently zeros | Use TWO `MoeGraphScratch` instances. Cost: 2× scratch (~16 MB extra). Acceptable. |
| Real model not present | Always | `#[ignore]` + `WeightFile::open` error | Self-skips; manual runner sees clear path. |
| `MetalBackend::new` signature drift | Medium | Compile error | Follow `cogito_moe_gpu.rs` test pattern; adjust on first compile. |
| `expert_base_id` mismatch | Always | Path divergence | Use `expert_files.mmap_id_for_expert(layer_idx, 0)` exactly as `moe_block_forward:2650-2653`. |
| Cold-cache > 5s | Low | Wall-clock obs | Document cold/warm split. |
| CPU `Backend` oracle as third leg | Low | Not in plan | **Skip.** Path A and Path B are bit-exact in production. CPU stub for `MoeGatherIdFuse` is `todo!()`. Redundant signal. |

**Rollback.** Single new file. `git rm` is total revert. No production-code impact.

# Open questions for Mike

1. **Layer index.** Propose `first_moe_layer_idx()` returns lowest layer where `VARIANT.layer_kind(i)` is MoE. Sound?
2. **Fail-fast if `bf` all zeros?** Defensive against silent kernel-dispatch drop. Kernel oracle does this. Default: yes.
3. **Env-flag interaction note?** `MOEFLUX_MOE_GATHER_ID` is bypassed entirely by the explicit-graph test. Default: yes, comment in test body.
4. **Note on Path A `bucket_input` host permute?** We unconditionally do it (vs production's env=on skip). Default: yes, comment that test I/O time ≠ production for Path A.
5. **Tombstone signal in header doc?** "Reverting `linear_attn_forward.rs:2802` MUST cause this test to fail." Default: yes — anchors test to purpose.

None block execution.

# Execution time estimate

- Write file: 45 min
- First compile + iterate on signature mismatches: 20-30 min
- First green run + verify revert-test causes failure with rich signal: 15-20 min

**Total: ~1.5 hours.**

# Cross-references

- [[moeflux-hardening-arc-plan]] — the arc this slots into
- [[prefill-gather-id-session-19-landed]] — the bug class this test catches the *revert* of
- [[feedback-coherence-test-before-pipeline-commit]] — the discipline this test structurally complements (engine-level diff is the test version of the curl)
- [[feedback-design-before-execute]] — Plan agent before execution

# Critical files for implementation

- `crates/moeflux/src/riir/attn/linear_attn_forward.rs:348-500` (`MoeGraphScratch`)
- `crates/moeflux/src/riir/attn/linear_attn_forward.rs:2722-2842` (graph-build pattern to mirror)
- `crates/moeflux/tests/cogito_moe_gpu.rs` (canonical engine-state-setup pattern)
- `crates/moeflux/tests/common/diff_helpers.rs` (paths, cosine, COSINE_FLOOR)
- `crates/moeflux/src/riir/io/layer_weight_cache.rs:145-263` (`LayerWeightCache::build`)
- `crates/moeflux-metal/tests/gather_mm_id_diff.rs` (per-row failure-report idiom in `run_down_shape_case`)
