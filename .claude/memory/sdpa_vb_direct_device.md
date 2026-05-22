# SDPA vB — direct-device rewrite (2026-05-22)

## Context

The 2026-05-22 ablation (`kernel_microbench` → `[sdpa-ablation]`)
pinned vA at:

```
attribution (A = 33.126 ms, M=1536 kv_len=1536):
  staging     31.958 ms  ( 96.5%)
  P-V         20.766 ms  ( 62.7%)
  QK^T         3.764 ms  ( 11.4%)
  softmax      3.029 ms  (  9.1%)
  floor        0.036 ms  (  0.1%)
```

Phases overlap (sum > 100%) but staging is the critical-path
dominator. Skip-stage drops the kernel to **1.169 ms** — a 28×
ceiling if we can avoid threadgroup staging entirely.

The previous v2 (simdgroup-MMA port) kept the staging structure and
added a diag-MMA rescale that turned out to be unique to us (a
session 18 hallucination about llama.cpp's pattern that was caught
when I read the kernel directly this session). Measured neutral to
~10% slower than vA. Dormant since 2026-05-21.

## What vB does

Replaces the v2 slot in-place with a port of llama.cpp's
`kernel_flash_attn_ext_impl` design — which Apple's MPP guide also
endorses for FA_HD=256 on Apple silicon:

| | vA | vB |
|---|---|---|
| Q tile | registers (`qreg[8][8]`) | **TG memory (`sq[8 × 256]`)** |
| O accum | registers (`acc[8][8]`) | **TG memory (`so[8 × 256]`)** |
| K source | TG mem (`stage_kv_block`) | **direct device** via `simdgroup_load(..., true)` |
| V source | TG mem (`stage_kv_block`) | **direct device** via `simdgroup_load(...)` |
| QK^T | scalar `fma` + `simd_sum` | `simdgroup_multiply_accumulate` 8×8 MMA |
| P·V | scalar `fma` | `simdgroup_multiply_accumulate` 8×8 MMA |
| max/sum | TG mem (`row_m[64]`, `row_l[64]`) | **per-thread scalar** `M, S` (simdgroup-replicated) |
| O rescale | scalar `acc *= corr` | scalar `so[i] *= corr` (TG mem) |
| Softmax exp | `fast::exp2` (vA's session-22 patch) | `fast::exp2` |
| Q tile | FA_BR=64 | **VB_Q=8** |
| KV block | FA_BC=16 | **VB_C=64** |
| Simdgroups/TG | FA_SIMDS=8 | VB_NSG=8 |

Per-lane register footprint drops **10×** — from ~512 B (vA's
`qreg + acc`) to ~50 B. That's the occupancy lever: direct device
reads only overlap with compute "naturally" at high occupancy
(MPP guide section 2.2 is explicit about this).

## Naming convention (forward-looking)

`vA` = production slot. `vB` = experimental slot. When a new design
beats vA on a clean rebooted A/B, the slots rotate: the new design
becomes vA, the previous vA becomes vB. Two slots indefinitely —
no v3/v4/v5. Diff oracle and bench always show A vs B; production
default is `vb: false`.

Promotion flow when vB wins:
1. Swap kernel bodies in `sdpa.metal` (rename entry points).
2. Update host `SDPA_VA` / `SDPA_VB` constants to point at the
   swapped names.
3. Toggle the gate default if appropriate.
4. The old vA body becomes the new vB body — and we have a fresh
   experimental slot for the next round.

## Key files (provenance for future readers)

- Kernel: `crates/moeflux-metal/shaders/sdpa.metal`
  - vA unfolded: `attn_sdpa_causal_flash_va` (lines ~86-266)
  - vA GQA-fold: `attn_sdpa_causal_flash_gqa2_va` and friends
    (entries at ~511-513, impl shared via `sdpa_gqa_impl<G>`)
  - vB unfolded: `attn_sdpa_causal_flash_vb` (lines ~561-end)
  - vB GQA fold: NOT implemented this session.
- Host: `crates/moeflux-metal/src/lib.rs`
  - Constants: `SDPA_VA`, `SDPA_GQA2_VA`, `SDPA_VB`, `VB_TILE_Q`
  - `SdpaCall::vb: bool` (was `v2`)
  - `SdpaCall::encode` branches on `(fold > 1, vb)` and picks
    `tile_q ∈ {FA_BR=64, VB_TILE_Q=8}` accordingly
- Gate: `crates/moeflux/src/riir/attn/linear_attn_forward.rs::sdpa_vb_enabled`
  reads `MOEFLUX_SDPA_VB=1|true|on`. Default off.
- Diff oracle: `crates/moeflux/tests/batched_diff_oracle.rs`
  - `sdpa_va_causal_flash_*` (5) + `sdpa_va_causal_flash_gqa2_*` (5)
  - `sdpa_vb_causal_flash_*` (5)
- Bench: `crates/moeflux/tests/kernel_bench.rs::bench_sdpa`
  reports vA + vB side-by-side at 3 shapes.

## Result — dirty bench, 2026-05-22

Diff oracle: **15/15 green, cosine 1.000000000** across all vA + vB
shapes. vB matches the CPU oracle bit-for-9-decimals.

Bench (dirty — no reboot, default power, just measure-as-we-go):

```
[sdpa] heads=16 kv_heads=2 head_dim=256 kv_dim=512
  vA sdpa M=1536  kv_len=1536    32.71 ms   591 GFLOP/s
  vB sdpa M=1536  kv_len=1536     4.55 ms  4247 GFLOP/s   7.18× speedup
  vA sdpa M=8192  kv_len=8192   812.42 ms   677 GFLOP/s
  vB sdpa M=8192  kv_len=8192   122.81 ms  4477 GFLOP/s   6.61× speedup
  vA sdpa M=8192  kv_len=32768 5657.07 ms   680 GFLOP/s
  vB sdpa M=8192  kv_len=32768  862.96 ms  4460 GFLOP/s   6.56× speedup
```

**6.5-7× speedup across all three prefill shapes.** Predicted 2-5×;
actual far exceeded that. The direct-device reads, small-tile
geometry (8 query rows/TG instead of 64), per-thread M/S, and the
simdgroup-MMA matmuls all stack — each individually plausible, the
multiplicative compounding wasn't.

Clean rebooted A/B still pending — Mike said he'll run it when home.
But 6.5× is well above noise; this is real.

## Clean rebooted A/B — 2026-05-22 (post-cold-boot)

```
[sdpa] heads=16 kv_heads=2 head_dim=256 kv_dim=512
  vA sdpa M=1536  kv_len=1536    33.12 ms   584 GFLOP/s
  vB sdpa M=1536  kv_len=1536     4.55 ms  4249 GFLOP/s   7.28× speedup
  vA sdpa M=8192  kv_len=8192   821.51 ms   669 GFLOP/s
  vB sdpa M=8192  kv_len=8192   122.25 ms  4497 GFLOP/s   6.72× speedup
  vA sdpa M=8192  kv_len=32768 5707.10 ms   674 GFLOP/s
  vB sdpa M=8192  kv_len=32768  840.24 ms  4580 GFLOP/s   6.79× speedup
```

Trial spreads on vB are ~0.3% (e.g. 122.139 / 122.253 / 122.481 for
M=8192/8192). Tight enough that the gap to vA isn't noise.

Confirmed. **vB is now the production default** (commit `1cc2163`).

## Promotion (2026-05-22, later same day)

The initial gate flip (`MOEFLUX_SDPA_VB` default ON) showed **no
change** — still 370 tok/s. Root cause: `fold == 1` guard. The a3b
model has `heads_per_kv=8` (even) → `fold=2` → vB never fired.

Fix: force `fold=1` when vB is active. Unfolded doubles threadgroups
(8 vs 4 for fold=2) but the 7× kernel win dwarfs the fold loss.

Engine-level prefill (a3b, ~15k tokens, max_tokens=1, n=3):
- **Before (vA, fold=2):** 370 tok/s
- **After (vB, fold=1):** 684-686 tok/s — **+85%**
- **llama.cpp baseline:** 810-820 tok/s
- **Remaining gap:** ~1.19× (down from ~2.2×)

## Follow-ups

1. vB GQA-fold (G≥2). For prefill at heads_per_kv=8, the fold reuses
   K/V across 2-8 query heads. Same trick applies under direct-device
   — just hoist the head loop inside the c0 loop. This would reclaim
   the 2× threadgroup overhead and likely close most of the remaining
   ~1.2× gap to llama.cpp.
2. Bigger VB_C (128 or 256)? More compute per direct-device fetch,
   but each block needs `min(VB_C, kv_len - c0)` valid → fewer total
   c0 iterations. Worth measuring once vB GQA baseline is solid.
3. Mask predicate cost at large kv_len. `needs_mask` fires for
   `O(M/64)` blocks per q-tile (the causal boundary). Bounded but
   nonzero; profile if it shows up.
