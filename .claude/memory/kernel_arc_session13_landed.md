# Kernel arc — session 13 landed (SDPA: ablation + GQA-fold G=2 + vec4 shipped)

2026-05-19. Session 13 did the Phase-0 bound analysis the session-12
memo demanded, then built/measured/shipped GQA-fold. Everything below
landed on moeflux `main`.

## Phase 0 — SDPA ablation (the bound analysis)

Added phase-skip function constants (`ABLATE_SKIP_QK/SOFTMAX/PV/STAGE`)
to `attn_sdpa_causal_flash` and an ablation harness in `kernel_bench`.
Result, decisive across all three shapes (1536 / 8192 / 8192-at-32k):

- **K/V staging is 96.7–96.9% of the kernel.** `skip-stage` collapses
  it ~30× (5916 ms → 183 ms deep). Compute (QK+softmax+P·V) is ~3%.
- **Latency/transaction-bound, not bandwidth-bound:** the deep shape
  moves ~137 GB of K/V over 5.73 s = **~24 GB/s ≈ 6% of the M2 Max's
  ~400 GB/s peak**. ~7/8 of those bytes are redundant (the 8 query-heads
  per KV head each re-stage the same K/V).
- Caveat on the attribution table: skipping a *consumer* phase lets the
  compiler DCE its *producer* staging, so the "QK^T" / "P·V" rows
  actually measure K-staging / V-staging. The 96.7% headline (from
  `skip-stage`) is unconfounded.
- **Crossed off:** double-buffered staging (compute is 3% — nothing to
  hide latency behind), softmax-opt (1–6%), simdgroup-matrix (session
  12). The phase-skip ablation constants stay in `sdpa.metal` as the
  permanent bound-analysis tool.

## vec4 staging — shipped

`stage_kv_block` rewritten to float4 loads (4 contiguous head_dim values
per load vs 16 strided scalar). Measured +10% at M=8192, +0.6% deep,
+4% at 1536. Now the only staging path — no flag. Zero register cost.

## GQA-fold G=2 — built, measured, shipped

New templated kernel `sdpa_gqa_impl<FA_GROUP>` + entry points
`attn_sdpa_causal_flash_gqa{2,4,8}` (`sdpa.metal`). One threadgroup per
`(q_tile, head-group of G query-heads)`; K+V co-staged once
(`FA_GQA_BC=8`, 16 KB) and reused across the folded heads.

Measured (`kernel_bench` GQA A/B):

| shape | G=2 | G=4 | G=8 |
|---|---|---|---|
| M=1536 | **1.24×** | 1.05× | 0.77× |
| M=8192 | **1.30×** | 1.21× | 0.99× |
| M=8192 / kv=32768 | **1.26×** | 1.22× | 1.00× |

**G=2 ≈ 1.3×, not the projected 1.9×.** Hard ceiling, measured: folding
doubles registers (qreg+acc 128→256 floats/thread), every folded PSO
caps at **448 threads/TG → 1 threadgroup per core**. Halved occupancy
eats ~half the transaction-count win on this latency-bound kernel. G=4/8
worse — more registers, less occupancy. G=2 is the sweet spot.

**Shipped to production:** `Kernels` builds `sdpa_gqa2`; `SdpaCall` has a
`fold` field; `SdpaCall::encode` routes `fold==2` to the folded kernel
with the `/2` grid. `full_attn_forward` sets `fold = 2` when
`heads_per_kv` is even (a3b: 8, a17b: 16), else `1` (Cogito:
heads_per_kv==1 → unfolded). gqa4/gqa8 exist for the bench only.

Gate: 10/10 `sdpa_causal_flash{,_gqa2}_*` diff tests cosine 1.0 through
the production path; canary battery 12/12 (incl. full-forward
`eval_prompt_matches_per_token_oracle` running the folded kernel).

## The parallel-execution anomaly → issue #1

Pre-reboot, the folded path produced wrong output ~2× in ~80 parallel
test-runs (gross errors, ~half the vector). **0 in 240+ runs
post-reboot**; 0 serial ever; 0 on the unfolded kernel ever. Exhaustive
static review found no defect (barriers, bounds, write-conflicts,
simd-convergence, divergence all clean). Reboot destroyed the repro, so
"latent race" vs "machine-state artifact" is unresolved. Filed as
moeflux **issue #1**; see [[sdpa_gqa_parallel_anomaly]]. Shipped G=2
anyway — keeps the repro window open; the gqa2 diff tests are the net.
**Lesson:** GPU correctness weirdness → reboot rules out Apple but kills
the repro; capture state / run a subagent *before* rebooting next time.

## Corrections / notes

- a3b is **16 heads / 2 KV → heads_per_kv = 8** (`variants.rs:419`,
  `#[cfg(model-qwen3-6-35b-a3b)]`). The 32/2 block at `:377` is the
  *a17b* variant. Grep `variants.rs` by `#[cfg]` feature, not first hit.
- `Kernels::new`'s `build` closure now passes an empty
  `FunctionConstantValues` (not `None`) — required once `sdpa.metal`
  declares function constants.

## Net for the session / next

SDPA's two structural levers are both measured and closed: simdgroup-
matrix (session 12, dead) and GQA-fold (1.3×, occupancy-bound, shipped).
Two further leads were investigated late in the session and **both ruled
out**:

- **Metal async-copy** (a `cp.async` analog — register-free global→
  threadgroup staging, which would have un-stuck the fold's occupancy
  ceiling): the instruction exists (`air.simdgroup_async_copy_2d`) but
  is *private/undocumented*. Using it = a reverse-engineered AIR
  intrinsic, ABI-fragile across macOS updates. Not worth it.
- **The `v3` matvec** (~700 GFLOP/s vs `qmm_t` ~9000 — a 13× microbench
  gap): an Explore sweep confirmed the prefill hot path *already* routes
  every significant 4-bit matmul through `qmm_t` / `GatherQmmCall`. Only
  the LM head (last-token-only) and two tiny 8-bit router gates still
  touch v3. Not a lever.

So a fresh a3b prefill profile was run (`profile.py`, samply, 15.7k-token
prompt) to actually locate the ~20× gap. **Verdict: prefill is CPU/
memory-bound, not GPU-bound** — GPU-wait is 2.5%; the time is 41%
memmove + 26% `pread` + CPU-side QK-norm/RoPE + per-layer Metal-buffer
creation. Session 12's "GPU-bound" reading was the linear-attn path; the
**full-attn path (`batched_full_attn_layer_forward`, 57% of prefill)
never got the GPU migration.** Session 14 plan-of-record:
[[prefill_arc_session14_plan]].
