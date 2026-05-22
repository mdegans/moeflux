# Prefill arc — next session plan (updated 2026-05-22)

## Current state

- **moeflux prefill:** 685 tok/s (a3b, ~15k tokens, direct-device vA)
- **llama.cpp prefill:** 810-820 tok/s (same prompt, same model)
- **Gap:** ~1.19× (~3.6 s on 15k tokens)

## Where the gap lives (Amdahl's law post-SDPA-7× win)

| Category | Est. % of GPU time | Notes |
|----------|-------------------|-------|
| Matmul projections | ~58% | qkv, o_proj, shared-expert FFN (qmm_t) |
| MoE dispatch | ~24% | gather_mm_id |
| SDPA | ~9% | direct-device vA — diminishing returns |
| Other (norms, etc) | ~9% | rms_norm, residual, router |

These are estimates from pre-improvement GPU capture + Amdahl scaling.
**Step 1 is to validate with a fresh capture.**

## Step 1: GPU capture (Mike runs from terminal)

Get a Metal capture with the improved SDPA kernel to identify the
actual dominant op. The 2026-05-21 capture showed SDPA at 61% — that
has changed dramatically. We need updated numbers.

## Step 2: attack the dominant op

### If qmm_t (batched 4-bit matmul) dominates:
- Compare our MLX-vendored `affine_qmm_rhs` against llama.cpp's
  `kernel_mul_mat` for the same shapes
- Tile-size sweep (BM/BN/BK)
- Q4_K_S format comparison — llama.cpp's sub-block scaling may be
  cheaper to decode. Measure before porting.

### If gather_mm_id (MoE) dominates:
- Already ported from llama.cpp — should be near parity
- Check capture for dispatch differences (barriers, grid sizing)

### If fixed per-layer overhead dominates:
- Command buffer submission frequency
- CPU-side router work between dispatches
- cmdbuf consolidation

## Dead ends (don't revisit)

- **GQA fold on direct-device SDPA:** 3% slower (2026-05-22 dirty
  A/B). Compute-bound, not bandwidth-bound. Serializing heads inside
  each TG loses parallelism.
- **SDPA staging kernel (vB):** 7× slower. Diff-oracle reference only.
- **LTO + codegen-units=1:** neutral at 9× build time (2026-04-28).

## Env var cheat sheet

| Var | Default | Effect |
|-----|---------|--------|
| `MOEFLUX_SDPA_VB` | OFF | Use old staging SDPA kernel |
| `MOEFLUX_SDPA_GQA` | OFF | GQA fold (G=2) on direct-device |
| `MOEFLUX_MOE_GATHER_ID` | ON | Use gather_mm_id MoE kernel |

## Cross-references

- [[sdpa_vb_direct_device]] — full SDPA arc history + promotion
- [[sdpa_session_learnings]] — what worked, what didn't, numerical
  precision notes
- [[prefill_sdpa_dominant_finding]] — pre-improvement GPU capture
  (SDPA 61%, MoE 11%)
