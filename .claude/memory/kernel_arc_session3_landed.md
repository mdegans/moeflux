# Kernel arc — session 3 landed (SDPA FlashAttention rewrite)

2026-05-16. Continues `kernel_arc_session2_landed.md` (which fingered
the full-attn SDPA kernel as the prefill bottleneck). This session
rewrote it. Plan: `~/.claude/plans/virtual-petting-pelican.md`.

## What landed (moeflux main)

The old 3-kernel `attn_sdpa_causal_*` SDPA path — one threadgroup per
`(query, head)`, re-reading all of K/V from global memory per query —
is **replaced** by `attn_sdpa_causal_flash`, a single-dispatch
FlashAttention-2 kernel.

Design: one threadgroup owns a tile of `FA_BR=64` query tokens for one
head and runs the whole online-softmax KV-block loop internally.
- Q tile **register-resident** (simdgroup-distributed, 8 floats/lane),
  read from global once.
- Each KV block staged once into a `threadgroup` buffer (`kv_stage`),
  time-multiplexed K → V, reused across the whole query tile.
- O accumulator register-resident.
- Causal block-skipping at `FA_BC=16` granularity.
- moeflux's `simd_sum`/`simd_shuffle` idiom — no global scratch, no
  `simdgroup_matrix`.

Commits: `0339ac6` kernel+encoder+5 diff tests · `cac6832` FA_BR tune
+ Phase-3 generalization · `1aa33f3` production swap + old-kernel
deletion (+ a latent Phase-1 fix, see below).

## Results

Correctness: all 5 `sdpa_causal_flash_*` diff tests **cosine =
1.000000000** vs the tokenwise `sdpa_cpu` oracle (n1 single/multi
block, n4, M=512 square-causal, M=1500 deep-chunk partial-tail).

Microbench A/B (`kernel_bench.rs`, n=3, M2 Max):

| shape | old tiled | flash (FA_BR=64) | speedup |
|---|---|---|---|
| M=8192 kv=8192  | 88 GFLOP/s  | **658 GFLOP/s** | 7.5× |
| M=8192 kv=32768 | 106 GFLOP/s | 639 GFLOP/s     | 6.0× |
| M=1536 kv=1536  | 94 GFLOP/s  | 556 GFLOP/s     | 5.9× |

Production re-profile (`prefill_profile`, M=8192 a3b chunk):

|                       | before  | after  |
|-----------------------|---------|--------|
| `batched_sdpa` phase  | 62.8 s (43.7%) | **8.1 s (8.7%)** |
| **prefill wall**      | **143.8 s** | **93.0 s** |

SDPA **7.8× faster in production**; overall a3b prefill **1.55×**
(143.8→93.0 s for 8192 tokens). Gap to llama.cpp prefill: ~20× → ~13×.
Logits finite, no NaN — swap correct end-to-end.

## Where prefill time goes now (M=8192)

| phase | share |
|---|---|
| `graph_linear_attn` | **40.5%** |
| `graph_moe` | 23.9% |
| `batched_shared_ffn_moe_combine` | 8.3% |
| `batched_sdpa_causal_flash` | 8.7% |
| norms (`rms_norm_qkv`, `oproj_route`) | ~7% |
| unaccounted (setup/lm_head) | ~12% |

**The next prefill arc should target `graph_linear_attn`** (the
GatedDeltaNet linear-attention chain, 40.5%) — not SDPA. See below.

## Strategic note — Path B is now low-priority

The plan named two SDPA follow-ups: **Path B** (simdgroup_matrix
QKᵀ/AV — the lever to reach llama.cpp's ~4.6 TFLOP/s) and GQA-fold.
The flash kernel lands at ~640-660 GFLOP/s ≈ 5% of M2 Max peak —
still ~7× behind llama.cpp's scalar-FA throughput, so Path B *would*
make the kernel faster.

**But** SDPA is now only 8.7% of prefill. Even an ideal Path B
(SDPA → ~0%) shaves at most ~8% off prefill wall. `graph_linear_attn`
at 40.5% is where the next arc's leverage is. Decision: Path B and
GQA-fold are **shelved** as low-priority; the next kernel arc profiles
and attacks the linear-attn graph. Revisit SDPA Path B only if a
future profile re-elevates it.

## Latent Phase-1 fix

The Phase-1 instrumentation commit (`1911430`) added a `label` arg to
`Backend::execute`/`submit_and_wait` but missed the 20 `.execute()`
callsites in `tests/graph_diff_oracle.rs` — the Phase-1 `--all-targets`
check was `| tail`-truncated and the test-target compile error
scrolled off. Fixed in `1aa33f3`. Lesson (already in drama_llama
CLAUDE.md): don't `| tail` a build you need to inspect — redirect to a
file and `grep error`.

## Gate status

`cargo build` zero-warnings (lib). All test targets compile. 5 flash
diff tests cosine 1.0. a3b `prefill_profile` produces finite logits.
Pre-existing test-target warnings (`graph_diff_oracle` `super::*`,
`batched_diff_oracle` `unused device`/`encode_matvec`) untouched —
not introduced this arc.

## Tunables / follow-ups

- `FA_BR=64`, `FA_BC=16` are `constant`s in `shaders.metal`
  (`attn_sdpa_causal_flash`); `FA_BR` also mirrored as `pub const` in
  `gpu_attn.rs`. `FA_BR=128` would spill (256 regs/thread). Phase 3's
  per-row reduction is `FA_BR`-agnostic (`FA_TPR = FA_THREADS/FA_BR`).
- head_dim is compile-time-fixed at `FA_HD=256` (a3b); the encoder
  asserts. Another head_dim needs a second specialized kernel.
- Next arc: profile + attack `graph_linear_attn` (40.5% of prefill).
