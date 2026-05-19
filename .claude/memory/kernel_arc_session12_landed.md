# Kernel arc — session 12 landed (SDPA QK^T → MMATile: built, benched, REVERTED)

2026-05-19. Executed "Phase 1" of the SDPA simdgroup-matrix rewrite
(`kernel_arc_session11_plan.md`). The rewrite was built and verified
correct, then **benched and reverted as a measured regression**. The
session's durable outputs are (a) the bench result that kills the
whole simdgroup-matrix-SDPA approach, and (b) a real upstream MLX bug
fix.

## The QK^T → MMATile rewrite — built, correct, REVERTED

Commit `bfa1be5` rewrote `attn_sdpa_causal_flash` Phase 2 (QK^T) as a
steel `MMATile` GEMM (Q staged into a register MMATile, K streamed
transposed, `BaseMMAFrag::mma`). It verified at **cosine = 1.0** on all
5 `sdpa_causal_flash_*` diff tests + canary 11/11 — functionally
perfect.

Then `kernel_bench` A/B (clean idle machine, n=3, <1% trial variance):

| shape | baseline (scalar+`simd_sum`) | MMATile QK^T | Δ |
|---|---|---|---|
| M=1536  kv=1536  | 33.76 ms  | 34.56 ms  | +2.3% |
| M=8192  kv=8192  | 821.7 ms  | 895.0 ms  | +8.9% |
| M=8192  kv=32768 | 5834.8 ms | 6598.8 ms | +13.1% |

**Regression, worsening with KV length.** Reverted in `2e2ef70` (the
`FA_BR=32→64` comment fix was kept — correct and independent).

## The load-bearing finding — SDPA is memory-bound, not compute-bound

`kernel_bench` reports GFLOP/s. SDPA: **~600–670 GFLOP/s** in *both*
versions. The `qmm_t` quantized matmuls in the *same* bench:
**~9000+ GFLOP/s**. SDPA runs at ~7% of simdgroup-matrix peak — it is
**memory / overhead-bound** (threadgroup traffic, the serial online-
softmax phases, barriers), not compute-bound.

Consequence: converting SDPA's matmul *math* onto the tensor ALU
cannot help. Worse — the `MMATile` path *adds* threadgroup traffic
(the Q-staging round-trip + per-ktile `B_k` loads) the scalar path
didn't have, so it's a net loss that compounds per KV block. This
**kills the whole simdgroup-matrix-SDPA arc**: QK^T is reverted, and
**P·V is dead by the same argument** (also memory-bound; Phase 5 was
already reduction-free, so there was even less to gain).

The session-11 plan had a **"Phase 0 — GPU capture: compute- vs
memory-bound"** that was *skipped*. It explicitly said "if memory-
bound, GQA-fold is the bigger lever." Skipping Phase 0 cost the
session its headline. **Lesson: do not skip the bound-analysis
capture before a kernel-math rewrite.**

## Upstream MLX bug — found, fixed, PR filed (real win)

While validating the `MMATile` encoding against `steel/gemm/mma.h`,
found a genuine bug: `BaseMMAFrag::load_safe` (`mma.h:92`) computes
the address with `off_x` in the column term where `off_y` belongs —
contradicting its own bounds check and the sibling `store_safe` /
`store_slice`. Confirmed in current upstream MLX. **Filed:
`ml-explore/mlx` PR #3565** (`(off_x+j)` → `(off_y+j)`), discovery
credited to Claude Opus 4.7. This stands regardless of the revert —
it's a real contribution.

## Net for the session

- SDPA simdgroup-matrix arc: **closed, negative result, crossed off.**
  The bench is the deliverable — "won't help a memory-bound kernel"
  is now measured fact.
- `kernel_bench` is confirmed as the right tool for SDPA work (pure
  GPU, no weights, tight variance, gives GFLOP/s → bound analysis).
- moeflux `main` is back to the pre-session SDPA kernel; only the
  `FA_BR` comment fix and memos remain.
- Next: `kernel_arc_session13_plan.md` — GQA-fold, the actual lever.
