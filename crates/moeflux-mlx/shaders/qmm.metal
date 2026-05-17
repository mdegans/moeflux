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
// batched, BM=32, BK=32, BN=32>. moeflux's a3b projections are 4-bit
// affine weights at group_size 64, f32 activations/output, non-batched.
// `aligned_N` (N % 32 == 0) is picked per-call by the host encoder, so
// both variants are instantiated. Kernel-name format matches MLX's
// `instantiate_quantized_aligned_batched` macro.

instantiate_kernel(
    "affine_qmm_t_float_gs_64_b_4_alN_true_batch_0",
    affine_qmm_t, float, 64, 4, true, 0)

instantiate_kernel(
    "affine_qmm_t_float_gs_64_b_4_alN_false_batch_0",
    affine_qmm_t, float, 64, 4, false, 0)
