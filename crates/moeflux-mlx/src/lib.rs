//! MLX-derived Metal compute kernels for the moeflux inference stack.
//!
//! This crate vendors a subset of [MLX](https://github.com/ml-explore/mlx)'s
//! Metal kernel source — the **affine quantized GEMM** (`affine_qmm_t` and,
//! later, the gathered/MoE variant) — and exposes it behind a thin Rust
//! dispatch wrapper. moeflux's hand-rolled 4-bit dequant matvec is
//! memory-bound at ~5% of GPU peak; MLX's `qmm_t` is a properly tiled
//! `simdgroup_matrix` GEMM, and moeflux already uses MLX's affine
//! quantization format, so the kernel consumes moeflux's weight layout
//! directly.
//!
//! ## Usage
//!
//! [`QmmKernels::new`] compiles the vendored kernels and builds the
//! pipelines once (fail-fast). [`QmmKernels::encode_qmm_t`] then encodes a
//! quantized matmul into a command buffer — infallible, hot-path-friendly:
//!
//! ```no_run
//! # use moeflux_mlx::{QmmKernels, QmmCall, QuantWeights};
//! # fn go(device: &metal::DeviceRef, cmd: &metal::CommandBufferRef,
//! #       wf: &metal::BufferRef, x: &metal::BufferRef, y: &metal::BufferRef) {
//! let kernels = QmmKernels::new(device).expect("build MLX kernels");
//! kernels.encode_qmm_t(cmd, &QmmCall {
//!     weights: QuantWeights { buffer: wf, packed_offset: 0,
//!                             scales_offset: 1 << 20, biases_offset: 2 << 20 },
//!     input: x, input_offset: 0,
//!     output: y, output_offset: 0,
//!     in_dim: 2048, out_dim: 4096, n_tokens: 8192,
//! });
//! # }
//! ```
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
//! ## How the library is built
//!
//! MLX's headers `#include` each other by full repo-root path
//! (`#include "mlx/backend/metal/kernels/..."`). `new_library_with_source`
//! does no include resolution, so [`assemble_source`] concatenates the
//! vendored files in dependency order and strips the quoted `#include`
//! lines — the same flatten-then-compile approach MLX's own JIT uses.

#![cfg(target_os = "macos")]

use metal::{
    BufferRef, CommandBufferRef, CompileOptions, ComputePipelineState,
    DeviceRef, MTLSize, NSUInteger,
};

/// Errors from building the MLX kernel library.
#[derive(Debug, thiserror::Error)]
pub enum MlxError {
    /// `new_library_with_source` rejected the assembled Metal source.
    #[error("compiling the moeflux-mlx Metal library: {0}")]
    Compile(String),
    /// A kernel function or pipeline could not be created.
    #[error("creating pipeline for kernel `{0}`: {1}")]
    Pipeline(String, String),
    /// The device cannot host the kernel's threadgroup size.
    #[error(
        "kernel `{0}` needs {needed} threads/threadgroup; device allows {got}"
    )]
    ThreadgroupTooSmall {
        /// Kernel name.
        kernel: String,
        /// Threads the kernel must launch per threadgroup.
        needed: u64,
        /// `maxTotalThreadsPerThreadgroup` the device reports.
        got: u64,
    },
}

// `affine_qmm_t` runs a fixed (32, 2, 2) threadgroup — 4 simdgroups.
const QMM_T_THREADS_PER_GROUP: u64 = 128;
// `affine_qmm_t` tiles the output in BM (token) x BN (row) blocks.
const QMM_T_TILE: u32 = 32;

const QMM_T_ALIGNED: &str = "affine_qmm_t_float_gs_64_b_4_alN_true_batch_0";
const QMM_T_UNALIGNED: &str = "affine_qmm_t_float_gs_64_b_4_alN_false_batch_0";

/// Vendored MLX headers + the moeflux-mlx instantiation entry, embedded at
/// build time. Order is the topological order of the files' `#include`
/// DAG — dependencies before dependents — so concatenation with quoted
/// includes stripped yields a well-formed translation unit.
const SHADER_PARTS: &[&str] = &[
    // bf16 / bf16_math / complex / defines / logging — in the exact order
    // MLX's own `utils.h` includes them (`complex.h` uses `bfloat16_t`
    // from `bf16.h`, an implicit dep not expressed as an `#include`).
    include_str!("../shaders/bf16.h"),
    include_str!("../shaders/bf16_math.h"),
    include_str!("../shaders/complex.h"),
    include_str!("../shaders/defines.h"),
    include_str!("../shaders/logging.h"),
    include_str!("../shaders/utils.h"),
    include_str!("../shaders/steel/defines.h"),
    include_str!("../shaders/steel/utils.h"),
    include_str!("../shaders/steel/utils/type_traits.h"),
    include_str!("../shaders/steel/utils/integral_constant.h"),
    include_str!("../shaders/steel/gemm/transforms.h"),
    include_str!("../shaders/steel/gemm/loader.h"),
    include_str!("../shaders/steel/gemm/mma.h"),
    include_str!("../shaders/quantized_utils.h"),
    include_str!("../shaders/quantized.h"),
    include_str!("../shaders/qmm.metal"),
];

/// Concatenate [`SHADER_PARTS`] into one Metal translation unit. Two kinds
/// of line are dropped:
///
/// - quoted `#include "..."` — the vendored files reference each other by
///   full repo-root path; the content is supplied by ordering instead.
/// - `#pragma once` — meaningless once every file is a single concatenated
///   TU (each appears exactly once), and it warns `-Wpragma-once-outside-
///   header`.
///
/// System `#include <...>` lines are kept — the Metal toolchain resolves
/// those.
pub fn assemble_source() -> String {
    let mut out = String::with_capacity(256 * 1024);
    for part in SHADER_PARTS {
        for line in part.lines() {
            let t = line.trim_start();
            if t.starts_with("#include \"") || t == "#pragma once" {
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// A quantized-weight tensor for [`QmmKernels::encode_qmm_t`].
///
/// MLX affine 4-bit format: `packed` holds 8 four-bit nibbles per `u32`
/// (`[out_dim, in_dim/8]` row-major); `scales` and `biases` hold one bf16
/// value per group of 64 input elements (`[out_dim, in_dim/64]`
/// row-major); the dequantized weight is `nibble * scale + bias`.
///
/// All three live at byte offsets within one shared Metal buffer — the
/// layout moeflux's mmap'd weight file already uses.
#[derive(Clone, Copy)]
pub struct QuantWeights<'a> {
    /// Buffer holding the packed weights, scales, and biases.
    pub buffer: &'a BufferRef,
    /// Byte offset of the packed-weight (`u32`) region.
    pub packed_offset: u64,
    /// Byte offset of the bf16 scales region.
    pub scales_offset: u64,
    /// Byte offset of the bf16 biases region.
    pub biases_offset: u64,
}

/// One quantized matmul: `output[n_tokens, out_dim] =
/// input[n_tokens, in_dim] @ dequant(weights)[out_dim, in_dim]ᵀ`.
///
/// In GEMM terms `M = n_tokens`, `N = out_dim`, `K = in_dim`. `input` and
/// `output` are f32, row-major, contiguous.
#[derive(Clone, Copy)]
pub struct QmmCall<'a> {
    /// 4-bit affine quantized weight matrix.
    pub weights: QuantWeights<'a>,
    /// Activation buffer — `[n_tokens, in_dim]` f32.
    pub input: &'a BufferRef,
    /// Byte offset of the activations within `input`.
    pub input_offset: u64,
    /// Output buffer — `[n_tokens, out_dim]` f32.
    pub output: &'a BufferRef,
    /// Byte offset of the output within `output`.
    pub output_offset: u64,
    /// `K` — input / contraction dimension.
    pub in_dim: u32,
    /// `N` — output dimension (rows of the weight matrix).
    pub out_dim: u32,
    /// `M` — number of token rows.
    pub n_tokens: u32,
}

/// The compiled MLX quantized-GEMM kernels, ready to dispatch.
///
/// Construct once via [`Self::new`]; it compiles the vendored MLX library
/// and builds both `affine_qmm_t` pipeline variants up front, so encoding
/// is infallible. Cheap to keep alive for a process; not cheap to rebuild.
pub struct QmmKernels {
    /// `aligned_N` variant — used when `out_dim % 32 == 0`.
    qmm_t_aligned: ComputePipelineState,
    /// Fallback variant with bounds-guarded loads.
    qmm_t_unaligned: ComputePipelineState,
}

impl QmmKernels {
    /// Compile the vendored MLX kernels on `device` and build the
    /// `affine_qmm_t` pipelines (4-bit affine weights, group_size 64, f32
    /// compute, bf16 scales, non-batched).
    pub fn new(device: &DeviceRef) -> Result<Self, MlxError> {
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&assemble_source(), &options)
            .map_err(MlxError::Compile)?;

        let build = |name: &str| -> Result<ComputePipelineState, MlxError> {
            let function = library
                .get_function(name, None)
                .map_err(|e| MlxError::Pipeline(name.to_string(), e))?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| MlxError::Pipeline(name.to_string(), e))?;
            let got = pipeline.max_total_threads_per_threadgroup();
            if got < QMM_T_THREADS_PER_GROUP {
                return Err(MlxError::ThreadgroupTooSmall {
                    kernel: name.to_string(),
                    needed: QMM_T_THREADS_PER_GROUP,
                    got,
                });
            }
            Ok(pipeline)
        };

        Ok(Self {
            qmm_t_aligned: build(QMM_T_ALIGNED)?,
            qmm_t_unaligned: build(QMM_T_UNALIGNED)?,
        })
    }

    /// Encode one quantized matmul ([`QmmCall`]) into `cmdbuf`.
    ///
    /// Picks the `aligned_N` kernel when `out_dim` is a multiple of the
    /// 32-wide output tile, else the bounds-guarded variant — the
    /// shape-aware selection seam. Encoding cannot fail.
    pub fn encode_qmm_t(&self, cmdbuf: &CommandBufferRef, call: &QmmCall) {
        let pipeline = if call.out_dim % QMM_T_TILE == 0 {
            &self.qmm_t_aligned
        } else {
            &self.qmm_t_unaligned
        };

        // The kernel's K/N/M are `int`; moeflux's dims fit i32.
        let k = call.in_dim as i32;
        let n = call.out_dim as i32;
        let m = call.n_tokens as i32;

        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(
            0,
            Some(call.weights.buffer),
            call.weights.packed_offset,
        );
        enc.set_buffer(
            1,
            Some(call.weights.buffer),
            call.weights.scales_offset,
        );
        enc.set_buffer(
            2,
            Some(call.weights.buffer),
            call.weights.biases_offset,
        );
        enc.set_buffer(3, Some(call.input), call.input_offset);
        enc.set_buffer(4, Some(call.output), call.output_offset);
        enc.set_bytes(5, 4, (&k as *const i32).cast());
        enc.set_bytes(6, 4, (&n as *const i32).cast());
        enc.set_bytes(7, 4, (&m as *const i32).cast());
        // Buffers 8-15 (batch shapes/strides) are unread: the kernel is
        // the `batched = 0` instantiation, so they stay unbound.

        // One threadgroup per 32x32 output tile; threadgroup (32, 2, 2).
        let grid = MTLSize::new(
            call.out_dim.div_ceil(QMM_T_TILE) as NSUInteger,
            call.n_tokens.div_ceil(QMM_T_TILE) as NSUInteger,
            1,
        );
        enc.dispatch_thread_groups(grid, MTLSize::new(32, 2, 2));
        enc.end_encoding();
    }
}
