# Kernel arc — session 4 landed (MLX quantized GEMM)

2026-05-17. Continues `kernel_arc_session3_landed.md` (SDPA FlashAttention
rewrite). That session's re-profile fingered `graph_linear_attn` (40.5% of
a3b prefill) as the next rock. Plan: `~/.claude/plans/mighty-purring-nebula.md`.

## What landed

The 4-bit dequant **matvecs** (`qkv/z/o_proj`, `gate_router`, …) — ~25% of
prefill — were memory-bound at ~720 GFLOP/s (~5% of GPU peak). The old
`dequant_matvec_4bit_v3_n_tokens` is a *matvec*: one threadgroup per
`(token, row)`, re-streaming the weight matrix once per token.

Two steps:

1. **Profiling instrumentation** — env-gated `MOEFLUX_PROFILE_PER_OP` mode in
   `MetalBackend::execute` (commit `95afb42`): commits each Op as its own
   labeled cmdbuf so `prefill_profile` gives a per-op breakdown. That split
   `graph_linear_attn`'s 40.5% into ~25% 4-bit matvecs + ~16% per-token-loop
   ops.

2. **Hand-rolled `mul_mm_4bit`** (committed `f3941bc`, deleted `0cd8c2b`) — a
   tiled `simdgroup_matrix` matmul. Correct (cosine 1.0) but only ~950
   GFLOP/s, ~3.5% of f16 peak: a naive single-buffered GEMM. Interim. Two
   findings banked: **`simdgroup_bfloat8x8` matmul is emulated on M2**
   (~3.5× slower than f16 — use `half` tiles); and out-tuning Apple's own
   GEMM by hand is not the bet.

3. **`moeflux-mlx` crate** — adopt MLX's tuned quantized GEMM. moeflux already
   uses MLX's affine 4-bit format, so MLX's `affine_qmm_t` consumes moeflux's
   weight layout directly. New workspace crate vendoring the MLX kernel
   headers (MIT, Apple © — `NOTICE` records provenance, commit `7b7c124`).

## Results

`qmm_t` microbench (a3b shapes, M=8192) — **decisive**:

| shape | v3 matvec | qmm_t (MLX) | speedup |
|---|---|---|---|
| qkv 2048→12288 | 716 GFLOP/s | **9136** | 12.8× |
| z 2048→8192 | 715 | 8814 | 12.3× |
| o 8192→2048 | 336 | 8569 | 25.5× |
| q 2048→4096 | 737 | 8306 | 11.3× |

qmm_t reaches **~8.3–9.1 TFLOP/s ≈ 65% of M2 Max f32 peak** (vs the matvec's
~5%). a3b **prefill wall 93.0 s → 70.3 s (1.32×)**; gap to llama.cpp prefill
~13× → ~10×. a3b smoke PASS (prefill + decode + state roundtrip, logits
finite).

## The moeflux-mlx crate

`crates/moeflux-mlx/` — the moeflux family's home for MLX-derived Metal
kernels (not qmm-specific; the gather variant + future lifts land here too).

- `shaders/` — 15 vendored MLX headers (5343 LOC), verbatim, every Apple
  copyright header intact. `qmm.metal` instantiates `affine_qmm_t`.
- `src/lib.rs` — `assemble_source` concatenates the headers in `#include`-DAG
  order, stripping quoted includes + `#pragma once` (MLX's headers `#include`
  by full repo-root path; `new_library_with_source` can't resolve that — the
  flatten-then-compile approach MLX's own JIT uses). `QmmKernels::new`
  compiles + builds both pipelines (fail-fast); `encode_qmm_t(cmdbuf,
  &QmmCall)` is the infallible hot path + the shape-aware selection seam
  (aligned/unaligned by `out_dim % 32`).
- **The one adaptation:** a `ScaleT` template parameter (defaulted to `T`)
  threaded through `QuantizedBlockLoader` / `qmm_t_impl` / `affine_qmm_t`, so
  the loader reads moeflux's bf16 scales/biases while computing in f32. Every
  change to vendored MLX source is marked inline `moeflux-mlx:`.
- Diff-tested in-crate vs a self-contained CPU oracle — cosine 1.0 across
  aligned + ragged shapes.

`MetalBackend` holds a `QmmKernels`; `encode_op`'s `MatvecNTokens` arm routes
4-bit through `qmm_t`, 8-bit (a3b `mlp.gate`/`shared_expert_gate`) stays on
the old `encode_matvec_n_tokens`.

## Where prefill time goes now (M=8192, wall 70.3 s)

| phase | share |
|---|---|
| `graph_moe` | **28.4%** |
| `graph_linear_attn` | 23.6% |
| `batched_sdpa_causal_flash` | 11.4% |
| `batched_shared_ffn_moe_combine` | 11.0% |
| `batched_rms_norm_qkv_proj` | 6.2% |
| `batched_oproj_post_attn_route` | 3.0% |
| unaccounted | ~16% |

## Next — P7 (its own session)

`graph_moe` (28.4%) is now the top phase: the **gathered** MoE expert
matmul. MLX has `affine_gather_qmm_t` for exactly this — adapt it the same
way (it shares `qmm_t_impl`, so the `ScaleT` work carries over). `shared_ffn`
(11.0%) is dense matvecs — route through `qmm_t`. Together ~39% of prefill,
and at ~12× that is the next big step.

Also still open: the **per-token-loop ops** inside `graph_linear_attn`
(`gated_delta_net_step`, `rms_norm_qk`, `gated_rms_norm`, `conv1d`,
`compute_decay_beta`) — ~16% of prefill, 8192 encoders/op/layer, occupancy-
starved. Batching them (one dispatch over the token axis) is a separate,
lower-risk arc. The `MOEFLUX_PROFILE_PER_OP` instrumentation makes it easy to
re-measure.

## Commits (moeflux main)

`95afb42` profiling instrumentation · `f3941bc` interim mul_mm_4bit ·
`8551704` moeflux-mlx scaffold + vendored headers · `eb2e7c3` compile the
MLX library · `bf26bd5` bf16-scale (`ScaleT`) adaptation · `9cc4179`
`QmmKernels` API + crate diff test · `e27c51b` integration + microbench ·
`8d3edf5` wire `qmm_t` into `MatvecNTokens` · `0cd8c2b` delete interim
mul_mm_4bit + re-profile.

## Gate status

`cargo build` zero-warning (lib). a3b smoke PASS. moeflux-mlx crate diff
tests cosine 1.0. Pre-existing test-target warnings (`graph_diff_oracle`
`super::*`, `batched_diff_oracle` `device`/`encode_matvec`) untouched.
