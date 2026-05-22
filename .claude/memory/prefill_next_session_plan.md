# Prefill arc — next session plan (updated 2026-05-22, session 2)

## Current state

- **moeflux prefill:** 775 tok/s (a3b, ~15k tokens, sequential vB)
- **llama.cpp prefill:** 810-820 tok/s (same prompt, same model)
- **Gap:** ~1.05× (~0.8 s on 15k tokens)

## Where the gap lives (post-delta-net-sequential GPU capture)

| Category | % of GPU time | Notes |
|----------|--------------|-------|
| `gated_delta_net_chunkwise` | 33% | → sequential vB: +15% e2e |
| `affine_qmm_t_float` | 29% | 4-bit matmul projections |
| `moeflux_mm_id` | 22% | MoE dispatch |
| SDPA + other | ~16% | Already optimized (7× last session) |

These % are from the pre-sequential-improvement capture. The sequential
kernel's 15% e2e win means delta-net dropped to roughly ~20% of GPU
time. **qmm_t (29%) is now the #1 target.**

## Step 1: promote sequential kernel to vA

The sequential kernel is correct (diff oracle cosine 1.000000000) and
15% faster. Slot rotation: rename entry points so vA = sequential,
vB = chunkwise. Same procedure as the SDPA promotion.

## Step 2: fresh GPU capture

Need updated % breakdown with the sequential kernel active. The
Amdahl estimates above are approximations — a real capture will show
whether qmm_t or mm_id moved.

## Step 3: attack qmm_t

If `affine_qmm_t_float` dominates:
- Compare our MLX-vendored `affine_qmm_rhs` against llama.cpp's
  `kernel_mul_mat` for the same shapes (projections: [n_tokens, dim])
- Tile-size sweep (BM/BN/BK)
- Q4_K_S format comparison — llama.cpp's sub-block scaling may be
  cheaper to decode

If `moeflux_mm_id` dominates:
- Already ported from llama.cpp — should be near parity
- Check for dispatch differences (barriers, grid sizing)

## Dead ends (don't revisit)

- **GQA fold on direct-device SDPA:** 3% slower (2026-05-22 session 1).
- **SDPA staging kernel (vB):** 7× slower.
- **Chunkwise delta-net:** 15% slower than sequential (2026-05-22
  session 2). Barrier + TG memory overhead outweighs chunkwise
  parallelism for this state size.
- **LTO + codegen-units=1:** neutral at 9× build time.

## Env var cheat sheet

| Var | Default | Effect |
|-----|---------|--------|
| `MOEFLUX_SDPA_VB` | OFF | Use old staging SDPA kernel |
| `MOEFLUX_SDPA_GQA` | OFF | GQA fold on direct-device |
| `MOEFLUX_DELTA_NET_VB` | OFF | Use sequential delta-net kernel |
| `MOEFLUX_MOE_GATHER_ID` | ON | Use gather_mm_id MoE kernel |

## Cross-references

- [[sdpa_vb_direct_device]] — SDPA arc history
- [[sdpa_session_learnings]] — SDPA what worked / didn't
- [[delta_net_sequential_session]] — this session's findings
