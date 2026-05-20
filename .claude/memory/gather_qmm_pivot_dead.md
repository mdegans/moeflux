---
name: gather-qmm-pivot-dead
description: 2026-05-20 — the per-expert dense pivot proposed in `gather_qmm_arch_pivot_plan.md` is dead via our kernels. Empirical evidence from small-M sweep + real htpe distribution + microbench + llama.cpp head-to-head. The actual differentiators are elsewhere; see `llama_cpp_moe_differentiators.md`.
metadata:
  type: project
---

# What got measured

Three load-bearing measurements landed today, in order.

## 1. Small-M dense `affine_qmm_t` sweep at per-expert shapes

`bench_qmm_t_a3b_per_expert_small_m` in
`crates/moeflux-metal/tests/kernel_bench.rs`. Run on M2 Max,
post-reboot, n=5 trials per shape.

| M | gate/up (in=2048 out=768) | down (in=768 out=2048) |
|---|---|---|
| 64 | 2799 GFLOP/s (37% of gather) | 5061 GFLOP/s (70% of gather) |
| 256 | 6770 GFLOP/s | 7141 GFLOP/s |
| 512 | 7538 GFLOP/s | 8069 GFLOP/s |
| 1024 | 8272 GFLOP/s | 8636 GFLOP/s |
| 4096 | 8973 GFLOP/s | 9052 GFLOP/s |
| 8192 | 9105 GFLOP/s (ceiling) | 9110 GFLOP/s |

Gather baseline (synthetic-uniform distribution, n_tokens=65536):
gate/up 7768 GFLOP/s, down 7186 GFLOP/s.

Reading: dense crosses gather around M≈700 for gate/up, M≈400 for
down. Below M=64 the dense kernel collapses (dispatch + tile-fill
overhead dominates).

## 2. Real htpe distribution from a3b prefill

Instrumentation: `MOEFLUX_LOG_HTPE=1` env gate added at
`crates/moeflux/src/riir/attn/linear_attn_forward.rs` ~ line 2566.
Dumps per-expert htpe arrays to stderr; one line per
layer × chunk. Run via `bench.py --model a3b --prompt-file
prefill_prompt_long.txt -n 1 --max-tokens 1`. Output captured at
`/tmp/bench_blallama.log`.

**The plan's mean-tokens-per-expert assumption was wrong by 2×.**
The plan assumed `n_tokens × k_active / n_experts = 65536 / 128 =
512` mean htpe. But a3b has **256 experts**, not 128. True mean is
`65536 / 256 = 256`. The cross-over point (M=400-700) means dense
loses on average at the mean. The pivot's "+8-11% end-to-end"
projection was load-bearing on the wrong mean.

**Distribution is heavy-left, not heavy-right.** Across all 80
(layer, chunk) cells of a 15692-token prefill:

| M range | % cells | % compute |
|---|---|---|
| 1–15 | 12.6% | 0.3% |
| 16–63 | 16.9% | 2.7% |
| 64–255 | 29.9% | 17.4% |
| 256–511 | 15.4% | 22.6% |
| 512–1023 | 9.4% | 27.1% |
| 1024+ | 4.4% | 29.8% |

So ~30% of cells live in M<64 (dispatch-overhead zone) and only ~14%
are at M≥512 (where dense wins). Compute is broader but concentrated
in the M<512 zone where dense is at-best a tie.

Analysis script saved at `/tmp/htpe_analyze.py` (not committed —
gitignored regen-from-log path). It interpolates the small-M curve
against the real distribution and predicts compute-weighted dense
GFLOP/s vs gather. Prediction: dense 0.919× gather on the MoE matmul
stage → ~6% end-to-end prefill regression if shipped.

## 3. Microbench: per-expert dense vs gather on real distribution

`bench_per_expert_vs_gather_real_distribution` in
`crates/moeflux-metal/tests/kernel_bench.rs`. Uses the exact htpe
from layer 0 chunk 0 as `HTPE_A3B_L0C0` constant.

| matmul | gather (1 dispatch) | dense (196 dispatches) | ratio |
|---|---|---|---|
| gate/up | 27.73 ms (7434 GFLOP/s) | 33.66 ms (6124 GFLOP/s) | **1.214×** gather |
| down | 29.70 ms (6940 GFLOP/s) | 28.14 ms (7325 GFLOP/s) | 0.947× dense |

Per-layer-per-chunk total (gate + up + down):
- gather: 85.16 ms
- dense:  95.46 ms
- **Dense loses by 12.1% on MoE matmul stage.**

Empirical answer matches analytic prediction (within 4 percentage
points). The pivot via our `affine_qmm_t` kernel is dead.

# Why the kernel-bench numbers are interesting on their own

- Gather throughput on the **real** distribution (7434 GFLOP/s) is
  4.3% lower than on the synthetic-uniform distribution (7768
  GFLOP/s) used in `bench_gather_qmm_a3b_moe`. Cost: tile-boundary
  fragmentation when small buckets straddle the 32-row tile edge.
  Closes another tile-tuning lever (BM=32 is still the apex, but the
  real distribution costs ~4% vs the synthetic case).

- At M=64 dense hits 30% of peak. At M=512 it's 83%. The kernel is
  tuned for large M (MLX default); there's no straightforward
  tile-tuning fix (BM sweep already closed).

# llama.cpp head-to-head — the framing-shift

Mike ran `bench.py --model a3b --backend llama-cpp --prompt-file
prefill_prompt_long.txt -n 3 --max-tokens 1` on the same M2 Max.

| iter | llama.cpp prefill tok/s | moeflux (today) |
|---|---|---|
| 1 | 858.33 | 287.76 |
| 2 | 857.86 | 332.95 |
| 3 | 850.33 | — |

**Gap is 2.5–3× on this hardware, not the 26-32× from session 5.**
Variance: llama.cpp <1% (very stable). moeflux ~16% across two runs
(suggests page-cache state matters for us — see
`llama_cpp_moe_differentiators.md`).

The "1k tok/s llama.cpp" figure is now a measured ~855 tok/s on this
specific hardware/prompt. Closer than the round-number prior
suggested; still significantly faster than us.

# Decision recorded

1. **Do not build the production pivot** using our current
   `affine_qmm_t` kernel. It will regress prefill by ~6%
   end-to-end on our distribution.

2. **Do not delete `gather_qmm_arch_pivot_plan.md`** — it now stands
   as a pre-empirical proposal that didn't survive the data. Keep
   it for future reference; mark DEAD with pointer to this memo.

3. **The real attackable gap is in three differentiators identified
   today**, not in the dispatch pattern. See
   `llama_cpp_moe_differentiators.md` for the breakdown and ranking.

# What's still useful from the dead pivot

- The two new kernel benches landed in this session (small-M sweep
  and per-expert vs gather) are **permanent diagnostic tools**.
  Re-run them after kernel changes or quant-format changes to
  measure progress.

- The `MOEFLUX_LOG_HTPE=1` instrumentation is small (~20 lines) and
  may stay in-tree as a debug knob — useful any time we want a real
  router distribution snapshot.

- The htpe constant `HTPE_A3B_L0C0` in `kernel_bench.rs` is a
  reproducible reference — future kernel benches can use it without
  re-running the model.

# Cross-references

- [[gather_qmm_arch_pivot_plan]] — the now-dead plan this memo
  supersedes.
- [[llama_cpp_moe_differentiators]] — what to attack instead.
- [[pread_teardown_landed]] — earlier teardown that ruled out the
  staging-blob path. Still correct; just doesn't explain the
  llama.cpp gap.
- [[kernel_bench_bm_sweep_landed]] — tile-tuning closed lever.
