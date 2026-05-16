# Cleanup arc — session 2 landed

2026-05-16. Continues `cleanup_arc_session1_landed.md`. Master plan:
`~/.claude/plans/stateful-booping-acorn.md`; this session's plan:
`~/.claude/plans/melodic-herding-river.md`.

User chose full remaining scope (multi-session). This session re-verified
the remaining phases against real code first (session-1 lesson) and
**three items shrank**: Phase 7 dropped (the linear-attn 600-line fn is
already graph-decomposed; the full-attn one has commit boundaries that
cross its phase banners — tangled, low value), Phase 5b dropped
(`state_size`/`state_save` already specialised), Phase 6 already
abandoned in session 1.

## Landed — 5 commits on moeflux `main`, all gated green

- `af963a9` **Phase 8** — module reorg. Flat 38-file `src/riir/` →
  `backend/{,cpu/,gpu/}` · `attn/` · `moe/` · `io/` · `snapshot/`
  (`mod.rs` + `variants.rs` stay at root). Pure `git mv` (93–100%
  rename similarity) + every `use super::X` → absolute `crate::riir::…`
  + `pub(super)` → `pub(in crate::riir)` where depth would lose scope.
  Done as ONE atomic commit (not the plan's per-subdir — per-subdir
  churns imports 2–3× through intermediate states; atomic rewrites each
  once). Delegated to a Sonnet subagent. Tag `cleanup-phase8`.
- `199db66` **Phase 5a** — `full_kv_mut` helper. The byte-identical
  `match &mut layer_states[idx] { FullAttn(kv)=>kv, Mla|LinearAttn=>Err }`
  block ×3 in `mod.rs` → one free fn, call sites use `?`.
- `44f9c85` **Phase 4b** — `pipeline_bundle!` macro. New
  `backend/gpu/encoder.rs`; 9 of 10 pipeline-bundle structs converted
  (−91 lines of boilerplate). `MlaPipelines::new` → `fetch` for
  uniformity (5 call sites). `DenseMlpPipelines` left hand-rolled — it
  has a nested `MatvecPipelines` field, genuinely heterogeneous. Tag
  `cleanup-phase4b`.
- 3 memory commits (`634c35c`, `0e98c49`, `02ac764`) — the C-oracle
  finding below.

**Phase 5c skipped** — the prefetch-dispatch block dedup needs an
8-argument helper (the audit's own proposed signature). That trades 12
duplicated lines for an 8-arg signature + a clippy allow — relocates
complexity, doesn't reduce it. Prefetch is perf-only (no computed-value
effect), so drift between the two sites isn't a correctness risk.
Skipped, same call as session 1's clusters 3–7.

## Two findings worth carrying forward

1. **The C diff-oracle is broken and non-deterministic.** Full detail
   in `future_work_c_oracle_broken_post_riir.md`. Short version: C
   end-to-end paths emit NaN/garbage (likely RIIR-cutover fallout,
   `2039619`); the failing-test set drifts run-to-run (11 → 13).
   Pre-existing, not reorg/4b-caused (proven: C code + `c_backend.rs`
   untouched; all Rust-path diff tests pass bit-exact). Mike's steer:
   C is to be removed eventually; lean on the Rust/CPU oracle. Decide
   its fate before the kernel arc.

2. **The plan doc's test-gate command is wrong.** It says run
   `--include-ignored` under `--features model-qwen3-5-a17b`, but the
   diff-oracle/canary suites load the **a3b** model — under a17b they
   fail at weight-load. Correct gate:
   `cargo test --no-fail-fast --features model-qwen3-6-35b-a3b -- --include-ignored`.
   The compile gate stays per-feature (all 3). `checkpoint_restore.rs`
   is a3b-only by design (fails under a17b) — a second pre-existing
   baseline failure the plan doc missed.

## Gate baseline (a3b, this session's tree)

`batched_diff_oracle` 16/16 · `graph_diff_oracle` 10/10 ·
`diff_oracle` ~33–35 Rust-path pass / ~11–13 C-vs-Rust fail (the
non-deterministic broken-C set). "Green" = all Rust-path tests pass;
C-comparison failures are the known C-oracle breakage.

## Resume point — Phase 4a (encoder builder)

The last real cleanup item. New `ComputeEncoder` builder in
`backend/gpu/encoder.rs` (the file Phase 4b created — extend it).

**Design refinement from this session's audit (important):** there are
42 `new_compute_command_encoder()` sites, not ~20; 40 are uniform
single-dispatch but 2–3 encode MULTIPLE dispatches per encoder
(`moe/expert_forward.rs` batched K-expert gate/up + swiglu/down;
`attn/gpu_attn.rs` tiled-SDPA accumulate loop). So the builder must NOT
auto-`end_encoding()` after `.dispatch()` — make `.dispatch()`
re-callable and end on an explicit `.end()` or `Drop`. That covers all
42 uniformly.

Phase 4a is the **headline risk** of the arc — a wrong `set_buffer`
index is silent GPU corruption. Convert one file per commit, full a3b
gate (corrected command above) after each. ~6–7 commits — a session of
its own.

Then Phase 9: `n≥3` post-reboot bench vs `pre-cleanup-baseline`
(expect flat) + optional clippy `--fix` sweep.
