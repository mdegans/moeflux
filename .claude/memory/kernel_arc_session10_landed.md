# Kernel arc — session 10: phases 3 + 5-GEMM2 simdgroup_matrix landed

2026-05-19. Continues `kernel_arc_session9_landed.md`. Followed
`kernel_arc_session10_plan.md` — **Tier 1 only** (Tier 2 `s0q`
deliberately deferred). Scope confirmed with Mike before execution.

## Landed — `shaders.metal` only

`gated_delta_net_chunkwise`, two scalar matmul phases rewritten as
cooperative `simdgroup_matrix<float,8,8>` GEMMs — same lever as
session 9's Phase 6. Op enum / backends / producer / CPU oracle all
unchanged. Both produce a `[C=16,D=128]` matrix into the existing
`sdc` buffer (time-shared with Phase 6 — **no new tg memory**, budget
still 26.75/32 KB). Tiling: 32 output tiles (2 row × 16 col), 4
simdgroups (`sg = vi/32`).

- **Phase 3** — `s0k[C,D] = kc[C,D] @ S_0ᵀ[D,D]`, contraction over
  D=128 (16 ktiles). `a` = `kc` tg-staged stride 128; `b` = the
  `[vi,d]` row-major device `state` loaded **transposed** → `[d,vi]`.
  Epilogue `U_l = beta_l v_l - beta_l gamma_l s0k` stays per-`vi`
  scalar. **No zero-fill** — raggedness is on the output `l` axis,
  not the D contraction; a ragged `l` only corrupts an unused output
  row (NaN cannot cross simdgroup_matrix rows). New barrier after the
  GEMM store, before the epilogue.
- **Phase 5 GEMM2** — `kqg[C,C] @ U[C,D]`, contraction over C=16
  (2 ktiles). `a` = `kqg` stride CW_C (already carries the `i≤l`
  mask from Phase 2); `b` = `Umat` stride 128. GEMM1 (`s0q`) **stays
  scalar** — that's Tier 2. **Zero-fill** `Umat` rows `c..CW_C`
  before the GEMM (contraction axis IS the ragged axis here →
  `0*NaN` trap, unlike Phase 3). New barrier after Phase 4 (the GEMM
  reads `Umat` across all columns; the old scalar code read only its
  own `vi` column so phases 4→5 had none).

## Correctness — green first try, both phases

`graph_metal_matches_cpu_gated_delta_chunkwise`, n ∈ {1,4,16,64}:
all `out cos=1.000000000`, `state cos=1.000000000`, worst
`max_abs=1.9e-6` (float reassociation from the 8-wide cooperative
contraction — identical magnitude to session 9's Phase 6, far inside
the 0.9999 gate). Canary `graph_diff_oracle` **12/12**. The ragged
n=1/n=4 shapes passing confirm the Phase-5 zero-fill and the Phase-3
no-zero-fill reasoning.

## Bench — PENDING, Mike to trigger

Not yet benched end-to-end. Release `blallama` **built** with Tier 1
(`target/release/blallama`, drama_llama). Clean A/B = post-reboot
`bench.py --model a3b -n 3 --no-build` on `prefill_prompt.txt` and
`prefill_prompt_long.txt`, vs the **f0f8edb baseline measured this
session post-reboot: 992 warm ≈ 250 tok/s, 15.7k ≈ 251 tok/s**
(recorded in drama_llama `benchmarks.md`).

**Honest expectation:** `gated_delta_net_step` was ~7% of prefill
wall after session 9. Tier 1 attacks the scalar matmuls *inside*
that 7%, so the whole-pipeline win is low-single-digit percent —
possibly near the bench noise floor. The sensitive confirmation is a
per-op profile (`MOEFLUX_PROFILE_PER_OP`), as session 9 used; do
that if the end-to-end bench is ambiguous.

## Not committed

Work is correctness-gated but uncommitted pending Mike's call at the
checkpoint (session 9 precedent: commit on correctness, bench
confirms after). One commit, `shaders.metal` only.

## Next — session 11 (fork, decide post-bench)

See `kernel_arc_session11_plan.md`. Two candidates:
1. **Tier 2 — `s0q`** (Phase 5 GEMM1). Needs staging `q` (+8 KB) →
   over the 32 KB cap → cascades into `sdc` 8→4 KB → Phase-6 strip
   16→8 + re-verify. Its own plan-mode pass. Alternative: a
   producer-side `conv_out` pad of CW_C tokens kills the cascade
   (q loads direct from device) but relaxes "shaders.metal only".
2. **Pivot to MoE GEMMs.** `moe_permute_fuse` profiled at a
   magnitude comparable to `gated_delta_net_step` (session 9:
   ~17–20 ms/commit) — likely the bigger fish now that linear-attn's
   matmul story is nearly closed.

Also queued: delete `GatedDeltaNetStepNTokens` (per-token Op +
kernel + arms) once a clean post-reboot bench confirms chunkwise
wins and decode (n=1) doesn't regress.
