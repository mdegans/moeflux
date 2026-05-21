---
name: prefill-sdpa-v2-diagnose-arc
description: 2026-05-21 — v2 simdgroup-MMA kernels landed correct (diff oracle cosine 1.0 across 20 tests) but ~9% SLOWER than v1 at engine level (warm mean 328 vs 361 tok/s on a3b). Diagnose arc to find what's eating the predicted lift.
metadata:
  type: project
---

# Initial result (UNCONFIRMED — thermal caveat below)

Post-cold-boot, n=5 prefill_prompt_long benchmarks (15,692
tokens, max-tokens=1):

| iter | v1 (v2=OFF) | v2 (v2=ON) |
|---:|---:|---:|
| 1 (cold) | 313.11 | 272 |
| 2 | 370.69 | 322 |
| 3 | 367.54 | 330 |
| 4 | 365.16 | 335 |
| 5 | 342.72 | 327 |

Warm mean (iter 2-5): v1 = 361.5, v2 = 328.5. Naive read:
v2 is ~9% slower.

**Thermal caveat (Mike, 2026-05-21):** v1 iter 5 dropped to
342 from a 365 peak — a 6% mid-run decline that looks like
thermal throttling. The v2 bench ran *immediately after* on
a thermally-loaded chip (fans already on). v2's numbers are
likely depressed by an unknown amount from thermal state.

**Re-bench planned**: cold-boot, dinner break to let laptop
cool, then re-bench v2 on a thermally-fresh chip. Do not
conclude v2 < v1 until that data is in.

**Mike's bench protocol note for future**: iter 1 is cold-
chip-penalty, iter 5 may be thermal-throttle. Sweet-spot is
iters 2-4. May want to land `--warmup 1 --max-iters K` on
bench.py with a thermal-watch heuristic that bails if
iter-N drops >5% from peak.

# Correctness was solid

20-test diff oracle pass at worst per-token cosine =
1.000000000 (5 shapes × 2 folds × v1/v2). v2 is bit-equivalent
to v1 within the same ULP envelope. **This is not a
correctness regression — it's a perf regression.** v2 stays
landed default-OFF; we diagnose without rollback.

# Suspect 1: finalize serialization (PRIMARY)

The v2 finalize writes lo (FA_BR × FA_HD = 64 KB) to device.
Threadgroup mem budget is only 32 KB so we can't stage all of
lo at once. My implementation serializes per-simdgroup:

- 8 iterations × (2 barriers + 1 simdgroup-only store + 1
  cross-thread scalar finalize) = **24 barriers per layer**.
- For GQA2 v2: 16 iterations × same = 48 barriers per layer.

v1 has zero such serialization — it writes per-lane directly
to device with scalar arithmetic. Each lane handles 8
head_dim positions per row, no TG staging, no barriers.

The 24/48-barrier overhead is the obvious eater. Each
barrier is ~few hundred GPU cycles; at ~9 full-attn layers
× ~few chunks per prefill, this could trivially be 10-20%
overhead.

**Fix**: direct `simdgroup_store(lo[k], device float*, stride)`
for full tiles (br_valid == FA_BR), pre-rescaled by a 1/row_l
diag matrix. Only the partial-tile case (last tile of a
prefill where br_valid < FA_BR) needs staged finalize. This
matches llama.cpp's approach (direct device writes).

# Suspect 2: Metal compiler loop unroll (SECONDARY)

The Plan agent flagged this: "aggressive unrolling on the
simdgroup-matrix loops could undo the register-pressure win."
My QK^T and PV loops are nested 32×2; if Metal unrolls them
fully, all 64 mk/mv matrices could be live simultaneously,
blowing the register budget back up.

**Diagnostic if Suspect 1 fix isn't enough**: GPU capture
with v2=ON, check `attn_sdpa_causal_flash_gqa2_v2`'s actual
occupancy and spilled bytes. If still 18.7% / 1+ KB spill,
unroll is the issue.

**Fix candidates**:
- `#pragma unroll(1)` on the offending loops
- `[[loop]]` attribute
- Refactor: keep one `mk` declared outside, manually overwrite
- Use MLX `MMATile<>` abstraction which has explicit unroll
  control via `STEEL_PRAGMA_UNROLL`

# Suspect 3: simdgroup_float8x8 vs simdgroup_half8x8

llama.cpp uses half-precision K/V/Q matrices. We use float
throughout because moeflux's KV cache is f32. Apple's matrix
pipes are roughly 2× throughput for half vs float per
Apple's Performance Primitives docs.

**Fix**: half-precision Q/K/V mixed-precision pipeline. Big
arc — touches KV-cache layout, KV-append kernel, diff
oracle. Defer until Suspects 1+2 are exhausted.

# What we learned

- Simdgroup MMA correctness is achievable (cosine 1.0).
- Direct port of v1's structure with MMA substitution wasn't
  enough — the finalize required architectural change.
- The diff oracle was a strong correctness gate but NOT a
  perf gate; an engine-level A/B is mandatory before any
  default flip. (Reinforces
  [[feedback-coherence-test-before-pipeline-commit]] but
  for *perf* not coherence.)
- Predicted +30% lift was overconfident given the finalize
  cost. Future: don't promise specific numbers until at
  least Suspect-1-level diagnostic exists.

# Updates (2026-05-21, same session)

## Cold-chip re-bench confirmed regression

n=5 v2-ON, cold-chip, fans off, ambient 21°C:
- 332.78, 332.65, 331.77, 334.65, 331.33 — mean 332.6,
  peak-to-peak 1.0% (very tight, steady-state)

vs v1 baseline cold-chip (warm-iter mean ~367.5):
**v2 is ~9.5% slower in steady state.** Not thermal artifact.

## Suspect 1 (finalize serialization) — RULED OUT

Implemented direct simdgroup_store-to-device fast path for
full tiles (br_valid == FA_BR) + per-simdgroup staged slow
path for partial tiles. Diff oracle re-passed at cosine 1.0
across all 20 cases.

Re-bench n=5: 329, 326, 333, 332, 330 — mean 330. **Same
range as before the fix.** The finalize serialization was
NOT the bottleneck.

This is informative — eliminates one of the two leading
suspects. The kernel itself (Q load + QK^T MMA + softmax +
PV MMA + rescale) is what's slow, not the post-loop
finalize.

The finalize fix is kept — it's correct, cleaner (zero
barriers in the common path vs 24-48 before), and a better
foundation for future work. Just not a win in isolation.

## Suspect 2 (Metal compiler unroll → register spill) — likely

The QK^T and P·V loops have nested 32×2 structure. If the
Metal compiler aggressively unrolls them, all 64 mk/mv
matrices could be live simultaneously, blowing the register
budget. Plan agent flagged this ahead of time.

**Confirming experiment**: GPU capture with
`MOEFLUX_SDPA_V2=1` on the bench command, inspect
`attn_sdpa_causal_flash_gqa2_v2`'s `Max Theor. Occupancy`
and `spilled bytes` in the Shaders tab. If still <40%
occupancy and >0 spill, unroll is the cause.

Fix candidates (in order):
- `#pragma unroll(1)` on the offending inner loops
- `[[loop]]` attribute on outer loops
- Refactor to use a single ephemeral `mk`/`mv` declared
  outside the loop, manually overwritten each iteration
- Port to MLX's `MMATile` abstraction with explicit
  `STEEL_PRAGMA_UNROLL` controls

## Surprising data point: `fast::exp2` opportunity

In looking at MLX's steel_attention as a reference (see
[[future-work-mlx-steel-attention]]), I noticed MLX uses
`fast::exp2` in the softmax with the scale pre-multiplied
by `1/ln2`. `fast::exp2` is ~3-5× faster than `exp` on
Apple Metal per Apple docs.

Both our v1 and v2 use `exp`. Switching to `fast::exp2`
is a small standalone change that could improve BOTH v1
and v2. Worth trying as a low-risk experiment.

# Revised path forward

This session ended at: v2 correct + finalize-fixed but
~10% slower than v1. Default-OFF, kernels remain. Two
commits clean (memory + code). Diagnose arc memo updated.

Next session options (Mike's call):
1. **GPU capture v2** to confirm Suspect 2 occupancy /
   spill state. Definitive diagnostic.
2. **Try `fast::exp2`** in both v1 and v2 — small,
   focused, may help v1 enough to be worth landing
   regardless of v2's fate.
3. **MLX steel_attention port** — bigger lift, more
   reward potential. See [[future-work-mlx-steel-attention]].
4. **Try `#pragma unroll(1)`** on v2 inner loops as a
   minimal Suspect-2 fix.

Recommended order: 1 (diagnose) → 2 (low-risk standalone
improvement) → 4 (minimal v2 fix) → 3 (architecture port
if still gap).

Do NOT delete v2 kernels. Mike, 2026-05-21: "We've
learned what doesn't work. We will find what does."

Do NOT delete v2 kernels. Mike: "We've learned what doesn't
work. We will find what does."

# Cross-references

- [[prefill-sdpa-dominant-finding]] — the trace that
  motivated v2.
- [[feedback-coherence-test-before-pipeline-commit]] — same
  principle but for perf.
- Plan agent's trouble-spot #3 (Metal compiler unroll) —
  flagged ahead of time, now confirmed plausible.
