# Kernel arc — session 10 plan: phases 3 + 5 simdgroup_matrix rewrite

Continues `kernel_arc_session9_landed.md`. Session 9 proved the
chunkwise kernel's matmul phases are compute-bound (Phase 6: 1.85–2.2×
from `simdgroup_matrix`). Session 10 applies the same lever to phases
3 and 5. **Design pass done at the close of session 9** (this memo
reflects the resolved decomposition); a short plan-mode confirmation
opens session 10, then execute.

## Decomposition — a free tier and a costly tier

The chunk index `l` is an **output row** of every phase-3/5 GEMM; the
contraction runs over `d` (D=128) or `i` (C=16). So a ragged `l` row
only corrupts an *unused* output row — it does **not** poison the
contraction. Zero-fill is only needed where a ragged row sits on the
**contraction** axis (the `0*NaN=NaN` trap).

### Tier 1 — Phase 3 + Phase 5 GEMM2 (no new staging, no budget cost)

Do this first. Same discipline as session 9: one phase at a time,
untouched phases stay scalar, per-phase diff-oracle gate.

- **Phase 3** — `s0k[C,D] = kc[C,D] @ S0ᵀ[D,D]`, contraction D=128
  (16 tiles of 8). Output `[C=16,D=128]` → reuse `sdc`. Operands:
  `a` = `kc` (already tg-staged, stride 128); `b` = `state` device
  buffer loaded **transposed** (S0 is `[vi,d]` row-major at
  `head*128*128`, stride 128 → transpose gives `[d,vi]`). Epilogue
  `U_l = beta_l v_l - beta_l gamma_l s0k` stays per-`vi` scalar.
  **No zero-fill** (raggedness is on the output `l` axis only).
  Big win expected — D=128 contraction was fully scalar.
- **Phase 5 GEMM2** — `kqg[C,C] @ U[C,D] → [C,D]`, contraction C=16
  (2 tiles). `a` = `kqg` (tg, stride CW_C); `b` = `Umat` (tg, stride
  128). `kqg` already carries the `i≤l` mask from Phase 2 (dense
  matmul OK; `kqg` ragged entries are already 0).
  **Fix:** zero-fill `Umat` rows `c..CW_C` before the GEMM —
  contraction is over `i`, and `kqg(0)*U(NaN)=NaN`. Phase 3/4 only
  write `l<c`.
  **New barrier:** one after Phase 4, before Phase 5 GEMM2 — the GEMM
  reads `Umat` cross-thread (all columns); today phases 4→5 have no
  barrier because the scalar code reads only its own `vi` column.

### Tier 2 — Phase 5 GEMM1 (`s0q`) — staging cascade, deferred

`s0q[C,D] = qc[C,D] @ S0ᵀ[D,D]`, contraction D=128. `q` is **not**
tg-staged (read direct from `conv_out`). A `simdgroup_load` of a
ragged `q` tile reads `conv_out` **out of bounds** — worst case
decode (n=1) overreads 7 tokens ≈ 86 KB past the buffer; a real
fault risk. Clamping doesn't save `n_tokens<8`.

Fix requires staging `q` (+8 KB tg) → total 34.75 KB **over** the
32 KB cap → forces `sdc` 8 KB→4 KB → forces Phase 6 strip 16→8 and
phase-3/5 output stripped 64-wide. Re-verify Phase 6.

**Decision:** Tier 2 is its own plan-mode pass (session 11), or a
Tier-1 follow-on only if session 10 has the energy. Alternative worth
weighing then: a producer-side `conv_out` pad of `CW_C` tokens
(relaxes "shaders.metal only", but kills the whole cascade — q can
then load direct from device). Phase 2 (`[16,16]` QK matmul) is also
deferred — smallest output, optional per the original session-9 plan.

## Carry-overs from session 9

- `sg = vi/32` local (no kernel attribute).
- `sdc` (8 KB) already allocated — Tier 1 reuses it for the
  `[C,128]` phase-3/5 GEMM outputs (different shape, different time
  than Phase 6's `[128,16]` use; one buffer, time-shared).
- tg budget after Tier 1: unchanged at 26.75 KB / 32 KB.

## Verification

Per phase: smoke + `graph_metal_matches_cpu_gated_delta_chunkwise`
(cos ≥ 0.9999, n ∈ {1,4,16,64} incl. g=0). Final: `diff_oracle`
canary 12/12 + **post-reboot** `bench.py --model a3b -n 3` A/B vs the
session-9 (Phase-6-only) baseline.

## After session 10

- Delete `GatedDeltaNetStepNTokens` (Op + per-token kernel +
  CpuBackend/GPU arms + `mod.rs` helper + `graph_diff_oracle.rs`
  sites) once a clean post-reboot bench confirms chunkwise wins and
  decode (n=1) doesn't regress.
- f16 v2 (mixed `simdgroup_matrix<half>` in / `<float>` accum) —
  marked follow-up, validate independently against the 0.9999 gate.
