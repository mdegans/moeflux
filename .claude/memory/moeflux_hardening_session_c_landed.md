---
name: moeflux-hardening-session-c-landed
description: 2026-05-21 outcome — Session C of the moeflux hardening arc shipped. Two test artifacts in two repos. moeflux engine_op_diff (per-layer bit-exact A/B against real A3B weights, ~280 LOC) + drama_llama moeflux_coherence (engine-level greedy A/B over 3 completion prompts × 30 steps via subprocess, ~250+140 LOC). Both compile clean; runs deferred to Mike per gpu-launch-from-cc memo.
metadata:
  type: project
---

# What landed

Closes the engine-level test arc that began with
[[moeflux-hardening-arc-plan]]. Sessions A (rename, commit `ae8c71b`)
and B (typed-`BufId<RoleTag>` refactor, commit `12dc016`) are still
the load-bearing protections. This session adds the *engine-level*
defense for the residual class of producer-wiring bugs the type
system can't see: wrong stride, missing upload, wrong layer offset,
wrong expert base, layout-assumption drift.

## Artifact 1 — `crates/moeflux/tests/engine_op_diff.rs`

Single new file (~290 LOC, `#[ignore]`-gated, single-purpose).
No production-code changes.

**Body shape:**
- Real Qwen3-A3B layer-0 weights (`first_k_dense_replace = 0`).
- Two `MoeGraphScratch::new(pool, K_ACTIVE=8)` instances — independent
  `commit_planned` latches; `commit_plan` deliberately skipped (test
  runs once; transients don't need lifetime coloring).
- Two `Graph`s per path: a *router* graph (RmsNorm + gate matvec +
  shared_gate matvec + MoeSoftmaxTopK + MoeNormalizeWeights) and a
  *MoE-block* graph (shared FFN + ZeroBuffer + permute-fuse OR
  gather-id-fuse + MoeCombineResidualNTokens).
- Both scratches see identical seeded `h_mid` bytes → routing tables
  are bit-identical (sanity-checked by `debug_assert_eq!` on
  downloaded indices). `h_post` is derived by the in-graph
  `Op::RmsNormBf16NTokens` to preserve the value-distribution
  difference that makes "wrong wiring" produce broken output.
- Asserts on `cosine >= 0.9999` and `rel max-abs-diff <= 1e-3`
  (matching `tests/cogito_moe_gpu.rs:124-133`), preceded by
  `report_diff` printing per-token max-abs (top-10 worst rows),
  global max-abs `(t, c)`, cosine, and nonzero counts.

**Numerical-equivalence calibration (first-green result, 2026-05-21):**
The original plan-of-record said "bit-exact is the contract." That
was wrong. `MoeGatherIdFuse` and `MoeBatchedPermuteFuse` use
different Metal kernels with different reduction orders, so they are
NOT bit-identical at the output. First green run reported:
- cosine = 1.0000000 (seven nines)
- global max-abs-diff = 7.6e-6 over ±1-magnitude values
- rel max-abs-diff ≈ 7e-6 (four orders below the 1e-3 floor)

That's f32-reduction-order noise. The bug-of-record class (wrong
buffer wired into `mlp_in`) would produce cosine ≈ 0 or argmax-flip
on most tokens — the cos ≥ 0.9999 floor catches it loudly. The
escape valve was actually anticipated in the plan-of-record section
3 ("if bit-exact ever drifts, cosine tells us drift vs wiring") but
the assertion shape didn't follow through. Fixed in this session;
the test is now self-calibrating against future kernel changes.

**Verification:** `cargo build -p moeflux --no-default-features
--features model-qwen3-6-35b-a3b --tests --release` is green.

**Run command** (Mike runs locally per [[feedback-gpu-launch-from-claude-code]]):
```bash
cargo test -p moeflux --no-default-features \
    --features model-qwen3-6-35b-a3b --release \
    --test engine_op_diff -- --ignored --nocapture
```

**Drift from the plan-of-record draft** ([[moeflux-hardening-session-c-plan]]):
- Dropped `first_moe_layer_idx()` helper (Qwen3-A3B has no dense
  prefix; `LAYER_IDX = 0` constant).
- Real router via 3 Ops (`MatvecNTokens` + `MoeSoftmaxTopK` +
  `MoeNormalizeWeights`) pushed at graph head — not `encode_moe_router`
  (which exists at `gpu_moe_router.rs:58` but is cmdbuf-level, wrong
  layer for the test).
- `LayerKind` only has `LinearAttn`/`FullAttn`; the MoE check uses
  `Variant::mlp_kind_at(i) == MlpKind::MoE` instead.
- `bits_of` inlined (the production helper is `pub(in crate::riir)`).
- Single mega-graph per path was rejected because graph_1 must execute
  before bucket build / upload can happen on host. Two graphs per path
  (router-prelude + MoE-block) mirrors production exactly.

## Artifact 2 — drama_llama `bin/moeflux_coherence_decode.rs` +
`tests/moeflux_coherence.rs`

Engine-level greedy-decode A/B over 3 completion-style prompts × 30
tokens, via subprocess isolation. `OnceLock<bool>` cache of
`MOEFLUX_MOE_GATHER_ID` (at moeflux `linear_attn_forward.rs:46-55`)
makes in-process toggling impossible — each pass is its own process.

**Plan deviation:** the original plan put the worker in `examples/`
to leverage `env!("CARGO_BIN_EXE_*")` — but that env var is bin-only
in stable Cargo. Moved to `bin/moeflux_coherence_decode.rs` with a
`required-features = ["moeflux-model-qwen3-6-35b-a3b"]` gate. Same
shape, same artifact count.

**Worker** (~150 LOC):
- argv: `<prompt-id>` ∈ {`tobe`, `constitution`, `hobbit`}.
  Three completion-style prompts (Mike's pick, all famously
  trained-on for cross-model reusability):
  - `tobe` → "To be, or not to be, that is"
  - `constitution` → "We the People of the United States, in Order to form a more perfect Union,"
  - `hobbit` → "In a hole in the ground there lived a hobbit."
- Inherits `MOEFLUX_MOE_GATHER_ID` and `DRAMA_LLAMA_MOEFLUX_*_DIR`
  from the orchestrator's env.
- Stdout: JSONL header line (with `stop_set = eos | eot |
  extra_eos_tokens` — narrow set per discussion;
  `special_tokens()` is too broad), 30 step lines
  (`{step, chosen, top: [[id, logit] × 50]}`), trailer line.
- Greedy via `partial_top_k` pattern from
  `tests/regression.rs:92-103`. Decodes all 30 steps regardless of
  stop tokens — the orchestrator's per-pass coherence check handles
  the stop-token guard.

**Orchestrator** (~270 LOC, `#[ignore]`):
- `Command::new(env!("CARGO_BIN_EXE_moeflux_coherence_decode"))`
  twice per prompt.
- Parses JSONL into `WorkerOutput { stop_set, steps, chosen_seq }`.
- Four assertions per prompt:
  1. Coherence — neither path emits a stop token in first 10 chosen;
     both produce all 30 steps.
  2. Argmax agreement — chosen sequences match for first 20 steps
     (the spectacular-failure detector for the bug-of-record class).
  3. Top-20 Jaccard mean — floor `JACCARD_MEAN_FLOOR = 0.9`
     (calibrate on first green run; tighten to 0.95 if practical
     mean is 0.99+).
  4. Cosine over union of `A.top_50 ∪ B.top_50` — missing-side
     padding uses each side's `top[49].logit` (min-observed floor);
     mean ≥ `COSINE_MEAN_FLOOR = 0.99`.
- Sanity: per-step `chosen` field matches `chosen_seq[i]` on each
  side (catches worker output corruption).

**Verification:** `cargo build --features "moeflux,moeflux-model-qwen3-6-35b-a3b"
--bin moeflux_coherence_decode --test moeflux_coherence --release` is green.

**Run command:**
```bash
cargo test --features "moeflux,moeflux-model-qwen3-6-35b-a3b" \
    --test moeflux_coherence -- --ignored --nocapture
```

## Pre-existing drama_llama test breakage (not Session C scope)

`cargo build --features moeflux,... --tests` surfaces ~17 errors and
~30 `multiple applicable items in scope` warnings in unrelated
existing tests:
- `tests/cross_backend.rs` — uses `Engine<D, M>` (2 generics), but
  v0.8.0's `Engine<B: Backend>` takes 1.
- `tests/session.rs` — same `D: Decoder` bound problem; also a
  `predict_candidates` method-not-found error.
- `tests/hash_cache_smoke.rs` — same shape.
- The lib has 11 warnings (mostly elided lifetime / unused mut /
  ambiguous numeric methods on f32) but no errors.

Per [[feedback-triage-failures]]: pre-existing breakage, not session
work. Flagging for a future cleanup pass. Possibly a `future_work_test_drift_post_v080.md`
memo is warranted in `~/Projects/drama_llama/.claude/memory/` if
this keeps biting bench/canary runs.

## First green run — practical numbers (2026-05-21)

Both artifacts green on first run.

**Artifact 1** — cosine = 1.0000000, rel max-abs-diff ≈ 7e-6 over
±1-magnitude values. Four orders below the 1e-3 rel floor. Paths
are mathematically equivalent.

**Artifact 2** — 3 prompts × 2 paths = 6 subprocess spawns,
total wall-clock 59.86s (warm; cold first run ~3-5 min):

| Prompt | Mean top-20 Jaccard | Mean cosine |
|---|---|---|
| `tobe` (Shakespeare completion) | 0.9469 | 0.999973 |
| `constitution` (verbatim recall) | 1.0000 | 1.000000 |
| `hobbit` (verbatim recall) | 0.9873 | 0.999984 |

Argmax-agree-first-20 passed cleanly on all three. Stop-set guard
fired no false positives. Verbatim-recall prompts are essentially
bit-perfect across paths; `tobe` carries the noise (completion of
high-entropy mid-sentence text has plausible alternates in the 15-
20th places of top-K, and they shuffle under f32 reduction-order
noise).

## Risk register (residual, post-green)

| Risk | When to act |
|---|---|
| `JACCARD_MEAN_FLOOR = 0.9` (artifact 2): 4.7pp margin on `tobe` | Don't tighten yet — collect ≥3 data points across reboots first per [[feedback-bench-discipline]]. Tightening to 0.93 would still pass `tobe` and bite harder on real wiring drift; reconsider after more runs. |
| `COSINE_MEAN_FLOOR = 0.99` (artifact 2): 5pp+ margin everywhere | Could tighten to 0.999 with healthy margin (5e-3 margin on `tobe`), would catch subtler kernel drift the 0.99 floor misses. Optional follow-up. |
| Bit-exact assertion in artifact 1 | RESOLVED 2026-05-21: loosened to cosine ≥ 0.9999, rel ≤ 1e-3. Now green at cosine=1.0000000. |
| `extra_eos_tokens` legitimately appearing in early tokens | Vet on first green; if false positives, narrow further to `eos | eot` only. |
| Routing nondeterminism in Metal scheduler | `debug_assert_eq!` on indices between scratches will fire; add `MOEFLUX_FORCE_DETERMINISTIC=1` to subprocesses if so. |
| `predict_candidates` internal early-stop on EoS | Worker errors if it gets fewer than 30 steps — diagnostic is loud enough. |
| Tombstone (revert + confirm test fails) deferred to first green | Plan section in `~/.claude/plans/lively-soaring-karp.md`. |

## Commit shape (deferred to Mike — not yet committed)

Two repos, two commits:
1. **moeflux**: single commit, `crates/moeflux/tests/engine_op_diff.rs`.
   No production code.
2. **drama_llama**: single commit, three files —
   `bin/moeflux_coherence_decode.rs`, `tests/moeflux_coherence.rs`,
   `Cargo.toml` (`[[bin]]` entry only).

Suggested subjects:
- moeflux: `tests: engine-level Op::MoeGatherIdFuse vs MoeBatchedPermuteFuse diff (hardening session C)`
- drama_llama: `tests: moeflux coherence A/B harness (hardening session C)`

## Cross-references

- [[moeflux-hardening-arc-plan]] — the arc this closes (Session C
  was the last item).
- [[moeflux-hardening-session-c-plan]] — the plan-of-record draft
  this refreshed; corrections folded into the new code.
- [[moeflux-hardening-session-b-v2-landed]] — Session B (typed
  BufIds) precedent.
- [[prefill-gather-id-session-19-landed]] — the bug class.
- [[feedback-coherence-test-before-pipeline-commit]] — the
  discipline these tests structurally complement.
- [[feedback-gpu-launch-from-claude-code]] — why the `--ignored`
  runs are Mike-runs-locally only.
- [[feedback-triage-failures]] — applied to the cross_backend
  pre-existing drift (flagged, not silently absorbed).
