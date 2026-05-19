# SDPA GQA-fold — parallel-execution anomaly (moeflux issue #1)

2026-05-19, session 13. Breadcrumb for a non-reproducible wrong-output
event in the GQA-folded SDPA kernel. **If this recurs, see below — and
read the "if it recurs" note before touching anything.**

## What happened

`attn_sdpa_causal_flash_gqa2` produced **wrong output** ~2× in ~80
parallel test-runs of `cargo test --test batched_diff_oracle --
--ignored sdpa_causal_flash` (default test-threads). Failures:
`sdpa_causal_flash_gqa2_m512_square_causal` token 64 cosine 0.707,
`..._m1500_deep_chunk` token 128 cosine 0.866 — gross errors, ~half the
output vector zero/wrong, not FP noise.

- Serial (`--test-threads=1`): clean, 25/25.
- **Post-reboot: 0 failures in 240+ test-executions** (24 parallel runs,
  two operators). The race did not survive a reboot.
- Unfolded kernel (`fold=1`): never observed to fail.

## What was checked (no defect found)

Exhaustive static review of `sdpa_gqa_impl` (`sdpa.metal`): every
threadgroup-memory producer→consumer is barriered; no OOB; all
threadgroup writes single-writer; `simd_sum`/`simd_shuffle_xor`
convergent; control flow uniform (no barrier divergence). Nothing.

## Leading explanation

Machine state. Pre-reboot the box had been up ~4 h running back-to-back
GPU benches. Mike: llama.cpp Metal builds have left this machine able to
do things that "shouldn't be possible." Blame Apple/Metal state first.
But the reboot also destroyed the repro — can't prove "machine state"
vs "real Heisenbug that any code/layout change perturbs away."

## Status

GQA-fold G=2 **shipped to production anyway** — keeps the repro window
open. The 5 `sdpa_causal_flash_gqa2_*` diff tests are the regression
net. Tracked: github.com/mdegans/moeflux/issues/1.

## If it recurs

**Do NOT reboot first.** Capture the live repro: save logs, loop the
failing test, launch a dedicated investigation subagent on the live
state. A reboot rules out Apple but throws away the only chance to learn
what it actually was.
