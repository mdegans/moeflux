# Future work — the C diff-oracle is broken end-to-end (post-RIIR)

Discovered 2026-05-16 during the cleanup arc (Phase 8 reorg gate run).

## Symptom

`cargo test -p moeflux --test diff_oracle --features model-qwen3-6-35b-a3b
-- --include-ignored` → **35 passed, 11 failed**. Every failure is a
C-vs-Rust comparison test; every Rust-vs-Rust test passes. `batched_diff_oracle`
(16) and `graph_diff_oracle` (10) are fully green.

Failing 11: `eval_token_matches_c_single_step`, `eval_prompt_matches_c_multi_token`,
`layer_forward_dump_close_c_vs_rust{,_cpu_combine,_full_attn,...}`,
`rms_norm_per_head_cpu_bit_exact_c_vs_rust`, `deferred_experts_discard_clears_state_c_vs_rust`,
`state_load_c_from_rust_save`, `state_load_rust_from_c_save`,
`state_size_matches_c_after_prefill`.

Severity is split:
- **End-to-end C paths produce garbage.** `eval_token`: C output is an
  all-zero vector (`max_abs_c=0.000e0`, argmax c=0 vs rs=17).
  `layer_forward_dump`: layer-0 cosine **0.385**.
- **C primitives are ~fine.** `rms_norm_per_head`: C vs Rust differ by
  1 ULP on 230/4096 elements (`c=-1.6078763 rs=-1.6078762`) — a
  tolerance/precision nit, not garbage.

## Likely cause

Commit `2039619` (2026-04-28) "RIIR: replace C host-side dispatch with
pure Rust port [Phases 0-6]" replaced the C host-side orchestration with
Rust. The pattern above is exactly what that leaves behind: low-level C
kernels still compute, but C *end-to-end inference* has no working host
dispatch, so it returns zeros. The `#[ignore]` reasons on these tests
still say only "long running; needs moeflux artifacts" — they were not
marked broken, so this rotted silently after the RIIR cutover.

## Not caused by the Phase 8 reorg

Deductive: the reorg touched only Rust files under `src/riir/` (path
edits); `tests/common/c_backend.rs` has zero diff; `moeflux-sys` (the C
code) was untouched; 61 Rust diff-oracle tests pass bit-exact. Both
sides of every C comparison are byte-identical pre/post reorg.

## Why it matters / what to do

The diff oracle is the regression net for the upcoming kernel/prefill
arc. The **C end-to-end oracle is currently dead**; the surviving
end-to-end net is the Rust per-token oracle
(`eval_prompt_matches_per_token_oracle`, green under a3b) plus the
synthetic `batched_diff_oracle` / `graph_diff_oracle` suites.

Before the kernel arc, decide the C oracle's fate (intersects the
existing `project_phase6_gate_c` plan): either re-wire enough C
host-dispatch to make it a real end-to-end oracle again, or formally
retire the C end-to-end tests and lean on the Rust per-token oracle.
The 1-ULP `rms_norm_per_head` nit is separate — a tolerance review.

## Gate-command correction (also discovered here)

The cleanup plan doc (`~/.claude/plans/stateful-booping-acorn.md`) and
`cleanup_arc_session1_landed.md` say the `--include-ignored` test gate
runs under `--features model-qwen3-5-a17b`. **Wrong** — the
diff-oracle/canary suites load the a3b model; under a17b they fail at
weight-load (`tensor 'model.layers.59...' not found`). The compile gate
is per-feature (all 3); the *behavioral* gate is
`--features model-qwen3-6-35b-a3b -- --include-ignored`. Also pass
`--no-fail-fast` or one model-mismatched binary masks the rest.
`checkpoint_restore.rs` is a3b-only by design and fails under a17b —
a second pre-existing baseline failure the plan doc missed.
