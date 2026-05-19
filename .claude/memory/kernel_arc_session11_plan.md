# Kernel arc — session 11 plan: profile-first fork

Continues `kernel_arc_session10_landed.md`. Session 10 closed the
linear-attn chunkwise kernel's scalar-matmul story (phases 3, 5-GEMM2,
6 all on `simdgroup_matrix`; only the Tier-2 `s0q` GEMM1 remains
scalar, and that one is small).

**Open the session with a fresh post-reboot per-op profile**
(`MOEFLUX_PROFILE_PER_OP`, the session-9 method) — Tier 1 changed the
op mix, so the session-9 numbers are stale. Let the profile pick the
target. The two standing candidates:

## Candidate A — Tier 2: `s0q` (Phase 5 GEMM1)

`s0q[C,D] = qc[C,D] @ S_0ᵀ[D,D]`, contraction D=128. The blocker:
`q` is read direct from `conv_out`, not tg-staged — a `simdgroup_load`
of a ragged `q` tile overreads `conv_out` (decode n=1: up to ~86 KB
past the buffer; a real fault). Two ways out:
- **Staging cascade** (shaders.metal-only): stage `q` (+8 KB) → 34.75
  KB > 32 KB cap → `sdc` 8→4 KB → Phase-6 strip 16→8 + phase-3/5
  output stripped 64-wide → re-verify Phase 6. Own plan-mode pass.
- **Producer-side `conv_out` pad** of CW_C tokens → `q` loads direct
  from device, whole cascade evaporates. Relaxes the "shaders.metal
  only" constraint but is structurally simpler. Weigh both at plan time.

Honest ceiling: `s0q` is one scalar GEMM in an op that is already
~7% of wall — small. Do this only if the profile says linear-attn is
still worth touching.

## Candidate B — MoE GEMMs (likely the bigger fish)

`moe_permute_fuse` profiled at ~17–20 ms/commit in session 9 —
comparable to the *whole* `gated_delta_net_step` op. Once linear-attn's
matmul story is closed, MoE is where the FLOPs are. The session-9
`simdgroup_matrix` pattern transfers directly. This folds in the
"orchestration gap → MoE-kernel phase" noted in the session-12 graph
memo. Needs its own design pass — bigger surface than Tier 2.

## Decision rule

If the post-Tier-1 profile shows `gated_delta_net_step` has dropped
near the noise floor (likely), **go Candidate B** and leave `s0q`
scalar — it isn't worth the staging cascade for a sub-1% slice.
If linear-attn is still a visible pole, Candidate A first.

## Also queued

Delete `GatedDeltaNetStepNTokens` (per-token Op + kernel + Cpu/GPU
arms + `mod.rs` helper + `graph_diff_oracle.rs` sites) once a clean
post-reboot bench confirms chunkwise wins and decode (n=1) doesn't
regress. Cheap cleanup; do it early in session 11.
