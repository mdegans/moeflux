# Kernel arc — session 10 plan: phases 3 + 5 simdgroup_matrix rewrite

Continues `kernel_arc_session9_landed.md`. Session 9 proved the
chunkwise kernel's matmul phases are compute-bound (Phase 6: 1.85–2.2×
from `simdgroup_matrix`). Session 10 applies the same lever to the
remaining scalar-matmul phases. **Gets a plan-mode formalization pass
next session** per `feedback_design_before_execute` — this memo is the
plan-of-record sketch.

## Scope

Rewrite phases 3 and 5 of `gated_delta_net_chunkwise` (`shaders.metal`)
to cooperative `simdgroup_matrix<float,8,8>`. Phase 2 optional, last.
Same discipline as session 9: **one kernel, one phase at a time**, Op /
backends / producer / CPU oracle unchanged, untouched phases stay
scalar so the diff oracle isolates each change. f32 v1.

## The GEMMs (per head, per chunk; C=CW_C=16, D=128)

`S0` = `state` device buffer, `[D,D]` row-major per head. We want
`sum_d state[vi,d]·x[l,d] = (x @ S0ᵀ)[l,vi]` → load `x` as operand a,
`S0` **transposed** as operand b.

- **Phase 3** — `s0k[C,D] = kc[C,D] @ S0ᵀ[D,D]`, contraction D=128
  (16 tiles of 8). Output `[16,128]` = 8 KB → reuse the `sdc` buffer
  (already `CW_C*128`). Epilogue `U_l = beta_l·v_l - beta_l·gamma_l·s0k`
  stays per-`vi` scalar.
- **Phase 5** — two GEMMs:
  1. `s0q[C,D] = qc[C,D] @ S0ᵀ[D,D]`, contraction D=128. `qc` is read
     from `conv_out` (q-region) — may need staging into tg memory
     first (kc is staged; q currently is not). Decide in plan mode:
     stage q, or `simdgroup_load` direct from the device `conv_out`
     with the right stride/offset.
  2. `kqg[C,C] @ U[C,D] → [C,D]`, contraction C=16. `kqg` already
     carries the `i≤l` triangular mask from Phase 2 (dense matmul OK).
  Output `[C,D]` reuses `sdc`. Epilogue: `out = gamma_l·s0q + (kqg@U)`.

These contractions (D=128) are **8× longer** than Phase 6's (C=16),
so the wins may be larger still.

## Carry-overs from session 9

- `sg = vi/32` local (no kernel attribute).
- **Ragged zero-fill** is the #1 risk again: zero rows `c..CW_C` of
  every tg operand before each matmul. `kc` is already zeroed by
  Phase 6's prologue *after* Phase 5 runs — Phase 3/5 run *before*
  Phase 6, so they need their own zero-fill of `kc`/`qc`/`Umat`
  ragged rows. `kqg` ragged entries are already 0 (Phase 2 writes 0
  for `l≥c`). Diff oracle n=1/n=4 catches a miss.
- tg budget: `sdc` (8 KB) already allocated; phases 3/5 outputs
  `[C,128]` reuse it. If q-staging is added that is +8 KB → total
  ~34.8 KB **over** the 32 KB cap → must alias q-staging over a dead
  buffer (Amat/kqg are alive through Phase 5; Umat is the RHS).
  Resolve in plan mode — likely `simdgroup_load` q direct from device.

## Verification

Per phase: smoke + `graph_metal_matches_cpu_gated_delta_chunkwise`
(cos ≥ 0.9999, n ∈ {1,4,16,64} incl. g=0). Final: `diff_oracle`
canary 12/12 + post-reboot `bench.py --model a3b -n 3` A/B.

## After session 10

- Delete `GatedDeltaNetStepNTokens` (Op + per-token kernel +
  CpuBackend/GPU arms + `mod.rs` helper + `graph_diff_oracle.rs`
  sites) once a clean post-reboot bench confirms chunkwise wins and
  decode (n=1) doesn't regress.
- f16 v2 (mixed `simdgroup_matrix<half>` in / `<float>` accum) —
  marked follow-up, validate independently against the 0.9999 gate.
