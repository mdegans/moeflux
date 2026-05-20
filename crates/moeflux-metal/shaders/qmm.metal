// moeflux-mlx: instantiation entry point for the MLX affine quantized GEMM.
//
// This file is NOT a standalone Metal translation unit. moeflux-mlx's
// library builder (src/lib.rs) concatenates the vendored MLX headers in
// dependency order — with their quoted `#include "..."` lines stripped —
// and appends this file. So by the time these macros are expanded,
// `instantiate_kernel` (defines.h) and `affine_qmm_t` (quantized.h) are
// already in scope.
//
// `affine_qmm_t` template params: <T, group_size, bits, aligned_N,
// batched, BM=32, BK=32, BN=32, ScaleT=T>. moeflux's a3b projections are
// 4-bit affine weights at group_size 64, f32 activations/output,
// non-batched. `aligned_N` (N % 32 == 0) is picked per-call by the host
// encoder, so both variants are instantiated.
//
// `ScaleT=bfloat16_t` is the moeflux-mlx adaptation: moeflux's quantized
// weights ship bf16 scales/biases, so the kernel reads them as bf16 and
// converts to f32 internally (see the ScaleT param in quantized.h). The
// BM/BK/BN defaults (32) must be spelled explicitly to reach ScaleT.
// Kernel-name format matches MLX's `instantiate_quantized_aligned_batched`.

instantiate_kernel(
    "affine_qmm_t_float_gs_64_b_4_alN_true_batch_0",
    affine_qmm_t, float, 64, 4, true, 0, 32, 32, 32, bfloat16_t)

instantiate_kernel(
    "affine_qmm_t_float_gs_64_b_4_alN_false_batch_0",
    affine_qmm_t, float, 64, 4, false, 0, 32, 32, 32, bfloat16_t)

// moeflux-mlx: gathered quantized GEMM for the MoE expert matmul. One
// dispatch handles all experts; `indices` gives each row's expert and the
// kernel collects contiguous same-expert runs. Template params:
// <T, group_size, bits, BM, BN, BK, WM, WN, transpose, ScaleT>. transpose
// = true matches moeflux's [out_dim, in_dim] weight layout (same as the
// dense affine_qmm_t path). align_M/N/K are function constants (200/201/
// 202), set per-PSO by the host, NOT a template axis.

instantiate_kernel(
    "affine_gather_qmm_rhs_float_gs_64_b_4_t_true",
    affine_gather_qmm_rhs, float, 64, 4, 32, 32, 32, 2, 2, true, bfloat16_t)

// Experimental BM=64 variant — 2× rows per threadgroup, same K/N work.
// Aimed at lifting the 20.83% occupancy ceiling we measured on the
// BM=32 PSO by amortizing register cost across more output elements
// per thread. Selected at runtime via MOEFLUX_GATHER_QMM_TILE=64.
// Template axis order is <T, group_size, bits, BM, BN, BK, WM, WN,
// transpose, ScaleT>; WM scales with BM to keep one simdgroup per
// (BM/16) rows.
//
// 2026-05-20 result: 23-26% SLOWER than BM=32 (post-reboot,
// confirmed). Kept compiled for reproducibility — see
// `kernel_bench_bm_sweep.md`.
instantiate_kernel(
    "affine_gather_qmm_rhs_float_gs_64_b_4_t_true_bm64",
    affine_gather_qmm_rhs, float, 64, 4, 64, 32, 32, 4, 2, true, bfloat16_t)

// Experimental BM=16 variant — 0.5× rows per threadgroup, same
// per-thread MMA-tiles-along-M. Tests the opposite hypothesis from
// BM=64: smaller tile → more threadgroups → potentially higher
// achieved occupancy if the per-threadgroup resource budget is what
// limits us (rather than per-thread register pressure).
// (BM/WM)/8 = (16/2)/8 = 1 simdgroup_matrix tile along M per
// simdgroup, vs 2 for BM=32 and BM=64.
//
// 2026-05-20 result: 7-10% SLOWER than BM=32. Tile-M dimension is
// fully explored; BM=32 is the apex on this hardware. See
// `gather_qmm_arch_pivot_plan.md` for why the next lever isn't
// tile-tuning.
instantiate_kernel(
    "affine_gather_qmm_rhs_float_gs_64_b_4_t_true_bm16",
    affine_gather_qmm_rhs, float, 64, 4, 16, 32, 32, 2, 2, true, bfloat16_t)
