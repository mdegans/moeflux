//! Metal compute kernels for the moeflux inference stack.
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
//! [`Kernels::new`] compiles the vendored kernels and builds the
//! pipelines once (fail-fast). [`Kernels::encode`] then encodes any
//! [`KernelCall`] into a command buffer — infallible, hot-path-friendly:
//!
//! ```no_run
//! # use moeflux_metal::{Kernels, QmmCall, QuantWeights};
//! # fn go(device: &metal::DeviceRef, cmd: &metal::CommandBufferRef,
//! #       wf: &metal::BufferRef, x: &metal::BufferRef, y: &metal::BufferRef) {
//! let kernels = Kernels::new(device).expect("build MLX kernels");
//! kernels.encode(cmd, &QmmCall {
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
//! scales while computing in f32) are marked inline as `moeflux-metal`
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

use std::ffi::c_void;

use metal::{
    BufferRef, CommandBufferRef, CompileOptions, ComputePipelineState,
    DeviceRef, FunctionConstantValues, MTLDataType, MTLSize, NSUInteger,
};

/// Errors from building the MLX kernel library.
#[derive(Debug, thiserror::Error)]
pub enum MlxError {
    /// `new_library_with_source` rejected the assembled Metal source.
    #[error("compiling the moeflux-metal Metal library: {0}")]
    Compile(String),
    /// A kernel function or pipeline could not be created.
    #[error("creating pipeline for kernel `{0}`: {1}")]
    Pipeline(String, String),
    /// The device cannot host the kernel's threadgroup size.
    #[error(
        "kernel `{kernel}` needs {needed} threads/threadgroup; \
         device allows {got}"
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

/// The gathered MoE GEMM. One Metal function; its `align_M/N/K` are
/// function constants (indices 200/201/202), specialised per PSO.
const GATHER_QMM_RHS: &str = "affine_gather_qmm_rhs_float_gs_64_b_4_t_true";
/// Experimental BM=64 / WM=4 variant of the gather kernel. Same
/// alignment-flag PSO matrix as the BM=32 default; selected at
/// runtime by [`gather_tile_variant`] (env `MOEFLUX_GATHER_QMM_TILE`).
const GATHER_QMM_RHS_BM64: &str =
    "affine_gather_qmm_rhs_float_gs_64_b_4_t_true_bm64";
/// Experimental BM=16 / WM=2 variant of the gather kernel.
const GATHER_QMM_RHS_BM16: &str =
    "affine_gather_qmm_rhs_float_gs_64_b_4_t_true_bm16";
// Function-constant indices for the gather kernel's alignment flags.
const FC_ALIGN_M: NSUInteger = 200;
const FC_ALIGN_N: NSUInteger = 201;
const FC_ALIGN_K: NSUInteger = 202;

/// Threadgroup-size + M-tile selector for the gather kernel.
/// `Bm32Wm2` matches MLX's default. `Bm16Wm2` halves the M tile;
/// `Bm64Wm4` doubles it. All three are always built; switch at
/// runtime via [`gather_tile_variant`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatherTileVariant {
    Bm16Wm2,
    Bm32Wm2,
    Bm64Wm4,
}

impl GatherTileVariant {
    /// M-axis tile size (rows per threadgroup).
    fn bm(self) -> u32 {
        match self {
            GatherTileVariant::Bm16Wm2 => 16,
            GatherTileVariant::Bm32Wm2 => 32,
            GatherTileVariant::Bm64Wm4 => 64,
        }
    }
    /// Threadgroup shape `(simd_lanes, WM, WN)`. Threads per group =
    /// `simd_lanes × WM × WN` = 128 for BM=16/32, 256 for BM=64.
    fn threadgroup_yxz(self) -> (u64, u64, u64) {
        match self {
            GatherTileVariant::Bm16Wm2 | GatherTileVariant::Bm32Wm2 => {
                (32, 2, 2)
            }
            GatherTileVariant::Bm64Wm4 => (32, 4, 2),
        }
    }
}

// Override sentinels for `OVERRIDE`. 0 = no override (use env-cached
// default); 1 = `Bm32Wm2`; 2 = `Bm64Wm4`; 3 = `Bm16Wm2`.
static OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// Process-wide gather-tile selection. Override (if set) wins;
/// otherwise reads `MOEFLUX_GATHER_QMM_TILE` once at first access.
/// Unrecognised env values fall back to BM=32 with a startup warning.
pub fn gather_tile_variant() -> GatherTileVariant {
    use std::sync::atomic::Ordering;
    match OVERRIDE.load(Ordering::Relaxed) {
        1 => return GatherTileVariant::Bm32Wm2,
        2 => return GatherTileVariant::Bm64Wm4,
        3 => return GatherTileVariant::Bm16Wm2,
        _ => {}
    }
    use std::sync::OnceLock;
    static CACHED: OnceLock<GatherTileVariant> = OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("MOEFLUX_GATHER_QMM_TILE").as_deref() {
            Ok("32") | Err(_) => GatherTileVariant::Bm32Wm2,
            Ok("64") => GatherTileVariant::Bm64Wm4,
            Ok("16") => GatherTileVariant::Bm16Wm2,
            Ok(other) => {
                eprintln!(
                    "[moeflux-metal] MOEFLUX_GATHER_QMM_TILE={other:?} \
                     unrecognised; using BM=32. Valid: 16 | 32 | 64."
                );
                GatherTileVariant::Bm32Wm2
            }
        }
    })
}

/// Set a process-wide override that bypasses the env-cached default.
/// `None` clears the override (env-cached default takes over again).
/// Intended for benches that A/B tile variants back-to-back in
/// one process — eliminates between-run system-state drift.
pub fn set_gather_tile_variant_override(v: Option<GatherTileVariant>) {
    use std::sync::atomic::Ordering;
    let val: u8 = match v {
        None => 0,
        Some(GatherTileVariant::Bm32Wm2) => 1,
        Some(GatherTileVariant::Bm64Wm4) => 2,
        Some(GatherTileVariant::Bm16Wm2) => 3,
    };
    OVERRIDE.store(val, Ordering::Relaxed);
}

/// The FA2 batched-prefill causal SDPA kernel (`sdpa.metal`).
const SDPA: &str = "attn_sdpa_causal_flash";
/// The GQA-folded SDPA kernel — one threadgroup per `(q_tile, head-pair)`,
/// staging each K/V block once for both query-heads. Used when
/// `SdpaCall::fold == 2`.
const SDPA_GQA2: &str = "attn_sdpa_causal_flash_gqa2";
/// SDPA query-tile size — must match `FA_BR` in `sdpa.metal`.
const FA_BR: u32 = 64;
/// SDPA threads per threadgroup — must match `FA_THREADS` in `sdpa.metal`.
const FA_THREADS: u32 = 256;

/// Vendored MLX headers + the moeflux-metal instantiation entry, embedded at
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
    // moeflux's own kernels — concatenated after the steel headers so
    // they can build on `mlx::steel::MMATile`.
    include_str!("../shaders/sdpa.metal"),
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

/// A quantized-weight tensor for [`QmmCall`] and [`GatherQmmCall`].
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

/// One gathered quantized matmul for the MoE expert path.
///
/// All experts' weights live contiguously in `weights.buffer` at a uniform
/// per-expert stride; `indices[row]` picks the expert for each input row.
/// Rows must be grouped into contiguous same-expert runs (the moeflux
/// bucket-permuted layout) — the kernel collects a run, then GEMMs it.
///
/// `output[m, out_dim] = input[m, in_dim] @ dequant(expert indices[m])ᵀ`.
#[derive(Clone, Copy)]
pub struct GatherQmmCall<'a> {
    /// Weight tensor of **expert 0** — `packed_offset`/`scales_offset`/
    /// `biases_offset` are the sub-tensor offsets within expert 0's block;
    /// expert `e` is reached by adding `e * stride_w` / `e * stride_s`.
    pub weights: QuantWeights<'a>,
    /// Activation buffer — `[n_tokens, in_dim]` f32.
    pub input: &'a BufferRef,
    /// Byte offset of the activations within `input`.
    pub input_offset: u64,
    /// Output buffer — `[n_tokens, out_dim]` f32.
    pub output: &'a BufferRef,
    /// Byte offset of the output within `output`.
    pub output_offset: u64,
    /// Per-row expert index — `[n_tokens]` `u32`.
    pub indices: &'a BufferRef,
    /// Byte offset of the indices within `indices`.
    pub indices_offset: u64,
    /// `K` — input / contraction dimension.
    pub in_dim: u32,
    /// `N` — output dimension (rows of one expert's weight matrix).
    pub out_dim: u32,
    /// `M` — number of token rows (total expert assignments).
    pub n_tokens: u32,
    /// Bytes between consecutive experts' packed-weight blocks.
    pub stride_w: u64,
    /// `ScaleT` (bf16) elements between consecutive experts' scale blocks.
    pub stride_s: u64,
}

/// One FlashAttention-2 batched-prefill causal SDPA dispatch.
///
/// `q` / `out` are `[n_tokens, num_heads, head_dim]` f32, contiguous;
/// `k_cache` / `v_cache` are `[kv_len, kv_dim]` f32 with
/// `kv_dim = num_kv_heads * head_dim`. Per-query causal mask: query `i`
/// attends to `[0, start_pos + i + 1)`. Single dispatch, no scratch —
/// the kernel keeps its online-softmax state threadgroup/register-
/// resident. `head_dim` must equal 256 (the kernel's threadgroup arrays
/// are sized at compile time around `FA_HD = 256`, a3b).
#[derive(Clone, Copy)]
pub struct SdpaCall<'a> {
    /// Query tensor — `[n_tokens, num_heads, head_dim]` f32.
    pub q: &'a BufferRef,
    /// Key cache — `[kv_len, kv_dim]` f32.
    pub k_cache: &'a BufferRef,
    /// Value cache — `[kv_len, kv_dim]` f32.
    pub v_cache: &'a BufferRef,
    /// Output tensor — `[n_tokens, num_heads, head_dim]` f32, written.
    pub out: &'a BufferRef,
    /// `M` — number of query tokens.
    pub n_tokens: u32,
    /// Number of attention (query) heads.
    pub num_heads: u32,
    /// Query heads sharing one KV head (GQA fan-out).
    pub heads_per_kv: u32,
    /// Per-head dimension. Must be 256.
    pub head_dim: u32,
    /// `kv_dim = num_kv_heads * head_dim` — row stride of the KV caches.
    pub kv_dim: u32,
    /// Absolute position of query 0 — the causal-mask offset.
    pub start_pos: u32,
    /// Total KV length the caches hold.
    pub kv_len: u32,
    /// Softmax scale, typically `1 / sqrt(head_dim)`.
    pub softmax_scale: f32,
    /// GQA-fold factor — query-heads folded into one threadgroup so each
    /// K/V block is staged once and reused. `1` = unfolded
    /// (`attn_sdpa_causal_flash`); `2` = the folded kernel
    /// (`attn_sdpa_causal_flash_gqa2`). Must divide both `num_heads` and
    /// `heads_per_kv`.
    pub fold: u32,
}

/// A kernel invocation that encodes itself into a command buffer using
/// the compiled [`Kernels`] pipelines.
///
/// Each kernel family is one `Call` struct (its parameters) plus one
/// `impl KernelCall` (its buffer bindings and dispatch). Adding a kernel
/// touches nothing in [`Kernels`] — dispatch routes through
/// [`Kernels::encode`].
pub trait KernelCall {
    /// Encode this call into `cmdbuf` against `kernels`' pipelines.
    fn encode(&self, kernels: &Kernels, cmdbuf: &CommandBufferRef);
}

/// The compiled Metal kernels, ready to dispatch.
///
/// Construct once via [`Self::new`]; it compiles the library and builds
/// every pipeline up front, so [`Self::encode`] is infallible. `Clone` is
/// cheap — per-pipeline NSObject refcount bumps; rebuilding is not.
#[derive(Clone)]
pub struct Kernels {
    /// `aligned_N` variant — used when `out_dim % 32 == 0`.
    qmm_t_aligned: ComputePipelineState,
    /// Fallback variant with bounds-guarded loads.
    qmm_t_unaligned: ComputePipelineState,
    /// `affine_gather_qmm_rhs` PSOs, indexed by the alignment triple
    /// `(align_M) | (align_N << 1) | (align_K << 2)`. All eight are built
    /// up front so a [`GatherQmmCall`] dispatch picks the exact-fit PSO
    /// for any M/N/K without a bounds-unsafe fallback in release builds.
    gather: [ComputePipelineState; 8],
    /// BM=64 / WM=4 variant of the gather kernel — same alignment-PSO
    /// matrix shape as `gather`. Selected at dispatch time by
    /// [`gather_tile_variant`]. All arrays always built; the extra
    /// metallib bytes (~20 KB per variant) are negligible.
    gather_bm64: [ComputePipelineState; 8],
    /// BM=16 / WM=2 variant of the gather kernel — same alignment-PSO
    /// matrix shape as `gather`.
    gather_bm16: [ComputePipelineState; 8],
    /// FA2 batched-prefill causal SDPA (`attn_sdpa_causal_flash`).
    sdpa: ComputePipelineState,
    /// GQA-folded SDPA (`attn_sdpa_causal_flash_gqa2`) — `SdpaCall::fold == 2`.
    sdpa_gqa2: ComputePipelineState,
}

impl Kernels {
    /// Compile the vendored MLX kernels on `device` and build the
    /// `affine_qmm_t` pipelines (4-bit affine weights, group_size 64, f32
    /// compute, bf16 scales, non-batched).
    pub fn new(device: &DeviceRef) -> Result<Self, MlxError> {
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&assemble_source(), &options)
            .map_err(MlxError::Compile)?;

        let build = |name: &str| -> Result<ComputePipelineState, MlxError> {
            // An empty `FunctionConstantValues` (not `None`): `sdpa.metal`
            // declares the `ABLATE_*` function constants, and Metal then
            // requires a specialized function even when every constant
            // takes its `is_function_constant_defined` default. Harmless
            // for the constant-free `qmm_t` kernels.
            let function = library
                .get_function(name, Some(FunctionConstantValues::new()))
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

        // Build one `affine_gather_qmm_rhs` PSO per alignment triple. The
        // kernel's align_M/N/K are function constants; `idx` bit 0/1/2 is
        // align_M/N/K. Specialising all eight keeps the gather dispatch
        // exact-fit (no bounds-unsafe align=true path for ragged dims).
        let build_gather = |name: &'static str,
                            needed_threads: u64,
                            idx: usize|
         -> Result<ComputePipelineState, MlxError> {
            let fcv = FunctionConstantValues::new();
            for (fc_index, bit) in
                [(FC_ALIGN_M, 0), (FC_ALIGN_N, 1), (FC_ALIGN_K, 2)]
            {
                let value = (idx >> bit) & 1 == 1;
                fcv.set_constant_value_at_index(
                    &value as *const bool as *const c_void,
                    MTLDataType::Bool,
                    fc_index,
                );
            }
            let function = library
                .get_function(name, Some(fcv))
                .map_err(|e| MlxError::Pipeline(name.to_string(), e))?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| MlxError::Pipeline(name.to_string(), e))?;
            let got = pipeline.max_total_threads_per_threadgroup();
            if got < needed_threads {
                return Err(MlxError::ThreadgroupTooSmall {
                    kernel: name.to_string(),
                    needed: needed_threads,
                    got,
                });
            }
            Ok(pipeline)
        };
        let build_gather_array =
            |name: &'static str, needed_threads: u64|
             -> Result<[ComputePipelineState; 8], MlxError> {
                Ok([
                    build_gather(name, needed_threads, 0)?,
                    build_gather(name, needed_threads, 1)?,
                    build_gather(name, needed_threads, 2)?,
                    build_gather(name, needed_threads, 3)?,
                    build_gather(name, needed_threads, 4)?,
                    build_gather(name, needed_threads, 5)?,
                    build_gather(name, needed_threads, 6)?,
                    build_gather(name, needed_threads, 7)?,
                ])
            };

        let gather =
            build_gather_array(GATHER_QMM_RHS, QMM_T_THREADS_PER_GROUP)?;
        // BM=64/WM=4 variant runs 256 threads/threadgroup.
        let gather_bm64 = build_gather_array(GATHER_QMM_RHS_BM64, 256)?;
        // BM=16/WM=2 variant runs the same 128 threads/threadgroup as
        // the BM=32 default.
        let gather_bm16 = build_gather_array(
            GATHER_QMM_RHS_BM16,
            QMM_T_THREADS_PER_GROUP,
        )?;

        Ok(Self {
            qmm_t_aligned: build(QMM_T_ALIGNED)?,
            qmm_t_unaligned: build(QMM_T_UNALIGNED)?,
            gather,
            gather_bm64,
            gather_bm16,
            sdpa: build(SDPA)?,
            sdpa_gqa2: build(SDPA_GQA2)?,
        })
    }

    /// Encode any [`KernelCall`] into `cmdbuf`. Infallible — every
    /// pipeline was built in [`Self::new`]. The uniform dispatch entry
    /// point: `kernels.encode(cmdbuf, &QmmCall { .. })`.
    pub fn encode<C: KernelCall>(
        &self,
        cmdbuf: &CommandBufferRef,
        call: &C,
    ) {
        call.encode(self, cmdbuf);
    }
}

/// Encode one quantized matmul ([`QmmCall`]) into `cmdbuf`.
///
/// Picks the `aligned_N` kernel when `out_dim` is a multiple of the
/// 32-wide output tile, else the bounds-guarded variant — the
/// shape-aware selection seam. Encoding cannot fail.
impl KernelCall for QmmCall<'_> {
    fn encode(&self, kernels: &Kernels, cmdbuf: &CommandBufferRef) {
        let pipeline = if self.out_dim % QMM_T_TILE == 0 {
            &kernels.qmm_t_aligned
        } else {
            &kernels.qmm_t_unaligned
        };

        // The kernel's K/N/M are `int`; moeflux's dims fit i32.
        let k = self.in_dim as i32;
        let n = self.out_dim as i32;
        let m = self.n_tokens as i32;

        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(
            0,
            Some(self.weights.buffer),
            self.weights.packed_offset,
        );
        enc.set_buffer(
            1,
            Some(self.weights.buffer),
            self.weights.scales_offset,
        );
        enc.set_buffer(
            2,
            Some(self.weights.buffer),
            self.weights.biases_offset,
        );
        enc.set_buffer(3, Some(self.input), self.input_offset);
        enc.set_buffer(4, Some(self.output), self.output_offset);
        enc.set_bytes(5, 4, (&k as *const i32).cast());
        enc.set_bytes(6, 4, (&n as *const i32).cast());
        enc.set_bytes(7, 4, (&m as *const i32).cast());
        // Buffers 8-15 (batch shapes/strides) are unread: the kernel is
        // the `batched = 0` instantiation, so they stay unbound.

        // One threadgroup per 32x32 output tile; threadgroup (32, 2, 2).
        let grid = MTLSize::new(
            self.out_dim.div_ceil(QMM_T_TILE) as NSUInteger,
            self.n_tokens.div_ceil(QMM_T_TILE) as NSUInteger,
            1,
        );
        enc.dispatch_thread_groups(grid, MTLSize::new(32, 2, 2));
        enc.end_encoding();
    }
}

/// Encode one gathered MoE matmul ([`GatherQmmCall`]) into `cmdbuf`.
///
/// Picks the exact-fit PSO for the call's M/N/K alignment — no
/// bounds-unsafe path. Encoding cannot fail.
impl KernelCall for GatherQmmCall<'_> {
    fn encode(&self, kernels: &Kernels, cmdbuf: &CommandBufferRef) {
        let variant = gather_tile_variant();
        let bm = variant.bm();
        // BN stays 32 in both variants; only the M-tile changes with BM.
        let align_m = self.n_tokens % bm == 0;
        let align_n = self.out_dim % QMM_T_TILE == 0;
        let align_k = self.in_dim % QMM_T_TILE == 0;
        let idx = usize::from(align_m)
            | (usize::from(align_n) << 1)
            | (usize::from(align_k) << 2);
        let pipeline = match variant {
            GatherTileVariant::Bm32Wm2 => &kernels.gather[idx],
            GatherTileVariant::Bm64Wm4 => &kernels.gather_bm64[idx],
            GatherTileVariant::Bm16Wm2 => &kernels.gather_bm16[idx],
        };

        // The kernel's K/N/M are `int`; moeflux's dims fit i32. Strides
        // are `uint64_t` (per-expert block stride — bytes / ScaleT elems).
        let k = self.in_dim as i32;
        let n = self.out_dim as i32;
        let m = self.n_tokens as i32;
        let stride_w = self.stride_w;
        let stride_s = self.stride_s;

        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(self.input), self.input_offset);
        enc.set_buffer(1, Some(self.weights.buffer), self.weights.packed_offset);
        enc.set_buffer(2, Some(self.weights.buffer), self.weights.scales_offset);
        enc.set_buffer(3, Some(self.weights.buffer), self.weights.biases_offset);
        enc.set_buffer(4, Some(self.indices), self.indices_offset);
        enc.set_buffer(5, Some(self.output), self.output_offset);
        enc.set_bytes(6, 4, (&m as *const i32).cast());
        enc.set_bytes(7, 4, (&n as *const i32).cast());
        enc.set_bytes(8, 4, (&k as *const i32).cast());
        enc.set_bytes(9, 8, (&stride_w as *const u64).cast());
        enc.set_bytes(10, 8, (&stride_s as *const u64).cast());

        // Grid: one threadgroup per (BN × BM) output tile. BN=32 fixed;
        // BM scales with the variant.
        let grid = MTLSize::new(
            self.out_dim.div_ceil(QMM_T_TILE) as NSUInteger,
            self.n_tokens.div_ceil(bm) as NSUInteger,
            1,
        );
        let (tg_x, tg_y, tg_z) = variant.threadgroup_yxz();
        enc.dispatch_thread_groups(grid, MTLSize::new(tg_x, tg_y, tg_z));
        enc.end_encoding();
    }
}

/// Encode the FA2 batched-prefill causal SDPA ([`SdpaCall`]) into
/// `cmdbuf`. One threadgroup per `(query tile, head)`; the kernel runs
/// the whole online-softmax KV-block loop internally. Encoding cannot
/// fail.
impl KernelCall for SdpaCall<'_> {
    fn encode(&self, kernels: &Kernels, cmdbuf: &CommandBufferRef) {
        if self.n_tokens == 0 || self.kv_len == 0 {
            return;
        }
        assert_eq!(
            self.head_dim, 256,
            "attn_sdpa_causal_flash is compiled for head_dim == 256"
        );

        debug_assert!(
            self.fold == 1 || self.fold == 2,
            "SdpaCall::fold must be 1 or 2, got {}",
            self.fold
        );
        debug_assert_eq!(
            self.num_heads % self.fold,
            0,
            "fold must divide num_heads"
        );

        // GQA-fold: one threadgroup per (q_tile, head-group of `fold`
        // query-heads). fold == 1 is the unfolded kernel.
        let pso = if self.fold > 1 {
            &kernels.sdpa_gqa2
        } else {
            &kernels.sdpa
        };
        let num_q_tiles = self.n_tokens.div_ceil(FA_BR);
        let total_tgs = num_q_tiles * (self.num_heads / self.fold);

        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(self.q), 0);
        enc.set_buffer(1, Some(self.k_cache), 0);
        enc.set_buffer(2, Some(self.v_cache), 0);
        enc.set_buffer(3, Some(self.out), 0);
        enc.set_bytes(4, 4, (&self.n_tokens as *const u32).cast());
        enc.set_bytes(5, 4, (&self.num_heads as *const u32).cast());
        enc.set_bytes(6, 4, (&self.heads_per_kv as *const u32).cast());
        enc.set_bytes(7, 4, (&self.kv_dim as *const u32).cast());
        enc.set_bytes(8, 4, (&self.start_pos as *const u32).cast());
        enc.set_bytes(9, 4, (&self.kv_len as *const u32).cast());
        enc.set_bytes(10, 4, (&self.softmax_scale as *const f32).cast());
        enc.dispatch_thread_groups(
            MTLSize::new(total_tgs as NSUInteger, 1, 1),
            MTLSize::new(FA_THREADS as NSUInteger, 1, 1),
        );
        enc.end_encoding();
    }
}
