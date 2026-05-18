# Kernel arc — session 8: chunkwise DeltaNet Phase 4 landed + NaN fix

2026-05-18. Continues `kernel_arc_session7_landed.md`. Followed
`kernel_arc_session8_plan.md` (Phase 4 — make the chunkwise win real).
Plan file (ephemeral): `~/.claude/plans/parallel-herding-patterson.md`.

## Landed — 3 commits on moeflux main

| commit | what |
|---|---|
| `841e001` | NaN fix — `g_decay` clamp before `ln` in chunkwise DeltaNet |
| `75f20cf` | metal: fail fast on a command-buffer error |
| `c219715` | producer emits `Op::GatedDeltaNetChunkwise` (the Phase-4 swap) |

Gates green: canary 6/6, lib 75/75, smoke 2/2, both chunkwise diff
tests `cos=1.0` with new `g=0` coverage.

## The NaN bug (the session's real work)

The Phase-4 swap was one line. It immediately failed the canary 6/6.
Root cause was a **latent bug in the session-7 chunkwise kernel**, not
the swap:

- The chunkwise reformulation accumulates `acc += ln(g_decay)` then
  computes `exp(L_l - L_i)`. A real forget gate `g_decay` can be
  exactly `0.0` (strong forget, or `f32` underflow of the `exp` that
  produces it). `ln(0) = -inf`; for any token pair past a zero,
  `exp((-inf) - (-inf)) = exp(NaN) = NaN`. NaN propagates through the
  residual stream → whole forward pass poisoned.
- The per-token `gated_delta_net_step` kernel is immune: it uses `g`
  directly (`state *= g`), never takes a log. That asymmetry is why
  the kernel passed isolated diff tests (synthetic `g_decay ∈
  [0.85,0.95]`) yet failed live.

**Debugging lesson worth keeping:** the symptom *looked* like all-zero
logits — `max_abs_diff=0.000e0` alongside `cosine=NaN`. That is the
signature of **NaN logits**, not zero: the test folds `(a-b).abs()`
with `f32::max` starting at `0.0`, and `f32::max(0.0, NaN)` returns
`0.0` (Rust's `f32::max` discards NaN). Cost ~3 wrong hypotheses
(GPU fault, timeout, threadgroup-mem) before the `f32::max` insight
flipped "zero" → "NaN". If you see `max_abs=0.000` + `cosine=NaN`,
suspect NaN first.

Fix: clamp `ln` input to `G_DECAY_LN_FLOOR = 1e-30`, identically in
the CPU reference `gated_delta_chunkwise` (`linear_attn.rs`) and the
Metal kernel `gated_delta_net_chunkwise`. A zero gate → `ln ≈ -69`
(≈hard reset), error ~1e-30 vs a true reset. Both chunkwise diff
tests now force a per-head zero gate (`graph_diff_oracle.rs`) — they'd
fail pre-fix (`cosine(NaN,NaN) < 0.9999`).

Also: `commit_and_wait_labeled` never checked `cmdbuf.status()` — a
GPU fault was silently swallowed. Now panics fail-fast with the label.

## Bench — chunkwise is performance-NEUTRAL

Dirty A/B (no reboot, contended machine — directional only):

| prompt | A: StepNTokens | B: Chunkwise |
|---|---|---|
| 992 tok | ~226 tok/s | ~225 tok/s |
| 15.7k tok | ~234 tok/s | ~232 tok/s |

Chunkwise ties the step kernel — not a win, not a regression.

Per-op profile (`prefill_profile.rs`, M=1536, `MOEFLUX_PROFILE_PER_OP`):
the delta op (`linear_attn.gated_delta_net_step` label) is **17.5% —
the #1 single-op pole**. Next: `moe_permute_fuse` 9.0%, `sdpa_flash`
6.1%. Fused profile (M=8192): `graph_linear_attn` 27.8%,
`batched_sdpa_causal_flash` 27.7%, ~30% unaccounted.

**Why neutral:** Phase 3 of the kernel arc was correctness-first —
every matmul inside the chunkwise kernel is a hand-rolled scalar
dot-product loop. The chunkwise form *exposes* parallelism; it doesn't
*realize* it. moeflux has zero `simdgroup_matrix` usage anywhere.
Estimated kernel utilization ~1.6% of f32 FLOP peak.

## Next — session 9

Optimize the chunkwise kernel's matmuls with hardware
`simdgroup_matrix`. Full plan: `kernel_arc_session9_plan.md`.
`GatedDeltaNetStepNTokens` (the per-token Op/kernel) is still in the
tree as the bench A-arm — delete it once session 9's bench confirms
chunkwise stays correct + decode doesn't regress.
