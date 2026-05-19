# Kernel arc — session 9: Phase 6 simdgroup_matrix rewrite landed

2026-05-19. Continues `kernel_arc_session8_landed.md`. Followed
`kernel_arc_session9_plan.md` + plan `flickering-purring-platypus.md`
(checkpoint-gated: Phase 6 only, then reassess).

## Landed — 1 commit on moeflux main

`gated_delta_net_chunkwise` (`shaders.metal`) Phase 6 — the new-state
update `S_C = gamma_{c-1} S_0 + sum_i Gamma_{c-1,i} U_i k_i^T` —
rewritten from a hand-rolled scalar dot-product loop to a cooperative
`simdgroup_matrix<float,8,8>` GEMM. moeflux's **first** use of
hardware cooperative matmul.

Four edits, `shaders.metal` only (Op / backends / producer / CPU
oracle all unchanged, as planned):
1. `#include <metal_simdgroup_matrix>`.
2. `#define CW_STRIP 16` — Phase-6 GEMM column-strip width.
3. `threadgroup float sdc[CW_C*128]` — delta-strip staging
   (tg memory 18.6 KB → 26.8 KB of 32 KB).
4. Phase 6 body: `delta[D,D] = RUᵀ[D,C] @ kc[C,D]`. RU built in place
   over the dead `Umat` buffer; ragged rows `c..CW_C` zero-filled in
   **both** RU and `kc` (`0*NaN=NaN`, so one operand is not enough);
   128 threads act as 4 simdgroups tiling the matmul into `sdc` one
   `[D,16]` column strip at a time; `gamma·S0 + delta` epilogue stays
   per-`vi` scalar (precision-sensitive).

**Deviation from plan:** did *not* add the
`simdgroup_index_in_threadgroup` kernel attribute — computed
`uint sg = vi / 32` as a local. Equivalent for a 1-D threadgroup,
smaller diff, no unused-param warning. `thread_index_in_simdgroup`
was unneeded (simdgroup_load/store/mac distribute lanes internally).

## Correctness — green first try

- Smoke 2/2 (kernel compiles clean with `simdgroup_matrix`, runs).
- Chunkwise diff oracle, n ∈ {1,4,16,64} incl. forced g=0:
  all `cos=1.000000000`, worst `max_abs=1.9e-6` (float reassociation
  from the cooperative 16-wide contraction — far inside the 0.9999
  gate). The ragged n=1/n=4 shapes passing confirms the zero-fill.
- Canary `diff_oracle` 12/12, incl. `eval_prompt_matches_per_token_oracle`
  (full forward, chunkwise kernel live) + `prompt_cache_start_pos_nonzero`.

The session-8 ragged-chunk risk did not bite — zero-fill written in
from the start per the plan.

## Bench — DECISIVE op-level win (dirty, directional)

Back-to-back per-op profile A/B, `prefill_profile.rs profile_1536`,
`MOEFLUX_PROFILE_PER_OP`. **Dirty**: uptime 1d20h, load ~3.6, no
reboot — directional only per `feedback_bench_discipline.md`.

| arm | `gated_delta_net_step` | ms/commit | % wall |
|---|---|---|---|
| A — scalar Phase 6 | 991.25 ms | 33.04 | 17.7% |
| B — simdgroup Phase 6 | 535.09 ms | 17.84 | 7.0% |

**1.85× on the op raw.** The A-arm 17.7% reproduces session 8's
recorded 17.5% — baseline consistent. Control op `moe_permute_fuse`
(unchanged code) drifted 16.81 → 20.13 ms/commit between runs: the
B-run was ~1.2× *more* contended, so contention-corrected the win is
~**2.2×**, and B won big despite worse measurement conditions. The
−15.2 ms/commit signal is 4–5× any control-op drift.

`bench.py` end-to-end deliberately skipped — the B profile run was
visibly more contended; a dirty whole-pipeline bench would be noise.
Clean confirmation = post-reboot `bench.py --model a3b -n 3` on 992
+ `prefill_prompt_long.txt` (Mike to trigger).

## Verdict — the checkpoint question, answered

**Phase 6 was compute-bound, and the kernel's matmul phases ARE
compute-bound.** The session-9-plan "honest expectation" (kernel might
be memory/occupancy-bound, simdgroup moves it little) is **refuted**
for Phase 6. The estimate of ~1.6% FLOP-peak utilisation was real
headroom. → Phases 3 and 5 (and optionally 2) — the same scalar-matmul
pattern — are worth rewriting. Session 10 plan written.

## Next — session 10

simdgroup_matrix rewrite of phases 3 + 5 (then optionally 2). Full
plan: `kernel_arc_session10_plan.md`. `GatedDeltaNetStepNTokens`
(per-token Op/kernel) still in tree as bench A-arm — delete once a
clean post-reboot bench confirms.
