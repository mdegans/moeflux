//! MLX-derived Metal compute kernels for the moeflux inference stack.
//!
//! This crate vendors a subset of [MLX](https://github.com/ml-explore/mlx)'s
//! Metal kernel source — the **affine quantized GEMM** (`affine_qmm_t` and,
//! later, the gathered/MoE variant) — and exposes it behind a thin Rust
//! dispatch wrapper. moeflux's hand-rolled 4-bit dequant matvec is
//! memory-bound at ~5% of GPU peak; MLX's `qmm_t` is a properly tiled,
//! `simdgroup_matrix` GEMM, and moeflux already uses MLX's affine
//! quantization format, so the kernel consumes moeflux's weight layout
//! directly.
//!
//! ## Provenance & license
//!
//! The Metal source under `shaders/` is vendored from MLX (MIT,
//! Copyright © 2023-2024 Apple Inc.). Every vendored file keeps its
//! upstream copyright header; the small adaptations moeflux needs (notably
//! a `ScaleT` template parameter so the quantized loader reads bf16
//! scales while computing in f32) are marked inline as `moeflux-mlx`
//! changes. See `NOTICE` for the upstream commit and details.
//!
//! ## Scope
//!
//! The crate is the moeflux family's home for MLX-derived Metal kernels in
//! general — `affine_qmm_t` is the first lift; the gathered/MoE variant
//! and any future MLX kernel lift land here too.
//!
//! ## Status
//!
//! Scaffold only. The Metal-library compilation and the `encode_qmm_t`
//! dispatch wrapper land in subsequent phases (see plan
//! `mighty-purring-nebula`).

#![cfg(target_os = "macos")]
