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

pub mod residency_set;

pub use residency_set::ResidencySet;

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

/// FA2 batched-prefill causal SDPA — **slot vA**: the current
/// production kernel (`sdpa.metal`). The vA/vB scheme keeps two
/// kernel slots indefinitely; new designs overwrite whichever slot
/// is older. See `attn_sdpa_causal_flash_vb` for the experimental
/// slot.
const SDPA_VA: &str = "attn_sdpa_causal_flash_va";
/// GQA-folded vA SDPA — one threadgroup per `(q_tile, head-pair)`,
/// staging each K/V block once for both query-heads. Used when
/// `SdpaCall::fold == 2`.
const SDPA_GQA2_VA: &str = "attn_sdpa_causal_flash_gqa2_va";
/// vB SDPA — the experimental slot. Currently the direct-device,
/// llama.cpp-style design: Q and O in threadgroup memory, K and V read
/// direct from device per block, per-thread scalar M/S. Different
/// dispatch geometry from vA (smaller per-TG query tile — see
/// `VB_TILE_Q`). Selected when `SdpaCall::vb && SdpaCall::fold == 1`.
/// vB GQA-fold is not implemented yet; `vb && fold == 2` asserts.
const SDPA_VB: &str = "attn_sdpa_causal_flash_vb";
/// vA query-tile size — must match `FA_BR` in `sdpa.metal`.
const FA_BR: u32 = 64;
/// vB query-tile size — must match `VB_Q` in `sdpa.metal`.
const VB_TILE_Q: u32 = 8;
/// SDPA threads per threadgroup — must match `FA_THREADS` in `sdpa.metal`.
/// Same for both slots.
const FA_THREADS: u32 = 256;

/// MoE map0 pre-pass kernel name — builds `htpe[]` + `hids[][]` from
/// per-token expert indices. Single instantiation for top-k = 8.
const MOE_MM_ID_MAP0_K8: &str = "moeflux_mm_id_map0_k8";
/// MoE single-dispatch gathered matmul (one of gate / up / down).
const MOE_MM_ID: &str = "moeflux_mm_id";
/// Topk-combine reduction kernel — `[n_tokens, k, hidden] × [n_tokens, k]
/// → [n_tokens, hidden]`.
const MOE_COMBINE_TOPK: &str = "moeflux_combine_topk";

/// Top-k for the map0 specialization — a3b. If other variants ship,
/// add their own `MOE_MM_ID_MAP0_K*` instantiation in
/// `gather_mm_id.metal` and a sibling PSO entry here.
pub const MOE_MM_ID_TOPK: u32 = 8;

/// Threadgroup-memory budget for the main matmul. NR0×NK halfs
/// (sa = 4 KB) + NR1×NK halfs (sb = 2 KB) during the K-loop;
/// reused as NR0×NR1 floats (8 KB) for the output stage. Max wins.
const MOE_MM_ID_TG_MEM: NSUInteger = 8192;
/// Threadgroup-memory budget for map0 — `n_experts × topk × u16`.
/// For a3b: 256 × 8 × 2 = 4096 bytes. Hardcoded for the k=8 variant.
const MOE_MM_ID_MAP0_TG_MEM: NSUInteger = 4096;
/// Threads-per-threadgroup for the main matmul (4 simdgroups × 32).
const MOE_MM_ID_THREADS: NSUInteger = 128;
/// Output-row tile (must match `NR0` in `gather_mm_id.metal`).
const MOE_MM_ID_NR0: u32 = 64;
/// M-row tile (must match `NR1` in `gather_mm_id.metal`).
const MOE_MM_ID_NR1: u32 = 32;

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
    // Port of llama.cpp's `kernel_mul_mm_id` (one-dispatch MoE matmul)
    // with W4_gs64 dequant inlined. See ../NOTICE for attribution.
    include_str!("../shaders/gather_mm_id.metal"),
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
    /// (`attn_sdpa_causal_flash_va`); `2` = the folded kernel
    /// (`attn_sdpa_causal_flash_gqa2_va`). Must divide both `num_heads`
    /// and `heads_per_kv`.
    pub fold: u32,
    /// Route through the vB experimental kernel
    /// (`attn_sdpa_causal_flash_vb`) instead of vA. Default `false` for
    /// safe landing; flipped by `MOEFLUX_SDPA_VB` at the engine layer.
    /// vB GQA fold is not yet implemented — `vb && fold > 1` asserts.
    pub vb: bool,
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
    /// FA2 batched-prefill causal SDPA — vA slot
    /// (`attn_sdpa_causal_flash_va`).
    sdpa_va: ComputePipelineState,
    /// GQA-folded vA SDPA (`attn_sdpa_causal_flash_gqa2_va`) —
    /// `SdpaCall::fold == 2 && !v2`.
    sdpa_gqa2_va: ComputePipelineState,
    /// vB unfolded SDPA — direct-device, small-tile design.
    /// `SdpaCall::vb && fold == 1`. No GQA-folded vB yet.
    sdpa_vb: ComputePipelineState,
    /// MoE map0 pre-pass — `htpe[]` + `hids[][]` from per-token expert
    /// indices. Top-k = 8 instantiation (`moeflux_mm_id_map0_k8`).
    moe_mm_id_map0_k8: ComputePipelineState,
    /// MoE one-dispatch gathered matmul (`moeflux_mm_id`). One PSO
    /// services gate, up, and down — args differ per call site.
    moe_mm_id: ComputePipelineState,
    /// Topk-combine reduction (`moeflux_combine_topk`).
    moe_combine_topk: ComputePipelineState,
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

        // MoE map0 + mm_id PSOs. The map0 needs at least n_experts
        // threads in one threadgroup; we'd surface the requirement at
        // the dispatch site via `MoeIdMap0Call::encode`. The main
        // mm_id PSO needs at least 128 threads (4 simdgroups).
        let moe_mm_id_map0_k8 = build(MOE_MM_ID_MAP0_K8)?;
        let moe_mm_id = build(MOE_MM_ID)?;
        let moe_combine_topk = build(MOE_COMBINE_TOPK)?;

        Ok(Self {
            qmm_t_aligned: build(QMM_T_ALIGNED)?,
            qmm_t_unaligned: build(QMM_T_UNALIGNED)?,
            gather,
            gather_bm64,
            gather_bm16,
            sdpa_va: build(SDPA_VA)?,
            sdpa_gqa2_va: build(SDPA_GQA2_VA)?,
            sdpa_vb: build(SDPA_VB)?,
            moe_mm_id_map0_k8,
            moe_mm_id,
            moe_combine_topk,
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
            "attn_sdpa_causal_flash_va is compiled for head_dim == 256"
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
        debug_assert!(
            !(self.vb && self.fold > 1),
            "vB GQA fold not yet implemented; \
             use `vb: false` for fold > 1"
        );

        // Slot × fold dispatch. vA is the production kernel (unfolded
        // and `_gqa2_va`); vB is the experimental slot (unfolded only
        // so far). Each slot has its own query-tile size, so the grid
        // sizing has to branch.
        let (pso, tile_q) = match (self.fold > 1, self.vb) {
            (false, false) => (&kernels.sdpa_va, FA_BR),
            (true, false) => (&kernels.sdpa_gqa2_va, FA_BR),
            (false, true) => (&kernels.sdpa_vb, VB_TILE_Q),
            (true, true) => unreachable!("vB GQA — guarded by debug_assert"),
        };
        let num_q_tiles = self.n_tokens.div_ceil(tile_q);
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

// ===============================================================
// MoE one-dispatch gathered matmul (`gather_mm_id.metal`)
// ===============================================================

/// `constant` args struct for [`MoeIdMap0Call`]. Layout must match
/// `moeflux_mm_id_map0_args` in `gather_mm_id.metal`.
#[repr(C)]
#[derive(Clone, Copy)]
struct MoeMmIdMap0Args {
    n_experts: i32,
    n_tokens: i32,
    k: i32,
}

/// `constant` args struct for [`MoeGatherIdCall`]. Layout must match
/// `moeflux_mm_id_args` in `gather_mm_id.metal`. C++ struct alignment
/// rules: int32_t (4) and uint64_t (8) — int32_t fields are grouped
/// at the top to avoid padding; uint64_t fields are contiguous; the
/// trailing int32_t triple has no padding (8-byte stride is satisfied
/// after the 3*4 = 12 bytes pad to 16; we explicitly add one i32 of
/// padding to match Metal MSL's auto-alignment).
#[repr(C)]
#[derive(Clone, Copy)]
struct MoeMmIdArgs {
    k_in: i32,
    n_out: i32,
    nb02: u64,
    nb01_w: u64,
    nb01_s: u64,
    packed_off: u64,
    scales_off: u64,
    biases_off: u64,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    n_tokens: i32,
    k: i32,
    ne11: i32,
    _pad: i32,
}

/// `constant` args struct for [`MoeCombineTopkCall`]. Layout must
/// match `moeflux_combine_topk_args` in `gather_mm_id.metal`.
#[repr(C)]
#[derive(Clone, Copy)]
struct MoeCombineTopkArgs {
    n_tokens: i32,
    hidden_dim: i32,
    k: i32,
}

/// Run the map0 pre-pass: build `htpe[n_experts]` (per-expert
/// assignment count) + `hids[n_experts, n_tokens]` (per-expert
/// assignment list, encoded as `token*k + slot`) from the router's
/// per-token expert indices.
///
/// One dispatch per layer, before the three [`MoeGatherIdCall`]s
/// (gate/up/down) for that layer.
#[derive(Clone, Copy)]
pub struct MoeIdMap0Call<'a> {
    /// Router indices — `[n_tokens, k]` i32.
    pub indices: &'a BufferRef,
    pub indices_offset: u64,
    /// Output htpe — `[n_experts]` u32.
    pub htpe: &'a BufferRef,
    pub htpe_offset: u64,
    /// Output hids — `[n_experts, n_tokens]` i32. Only the first
    /// `htpe[e]` entries per expert are valid.
    pub hids: &'a BufferRef,
    pub hids_offset: u64,
    pub n_experts: u32,
    pub n_tokens: u32,
    /// Top-k. Currently must equal [`MOE_MM_ID_TOPK`] = 8.
    pub k: u32,
}

/// Run one MoE matmul: `dst[t, s, r] = dequant(W[e, r, :]) @ src1[t, ?]`
/// for every assignment `(t, s)` to expert `e`, where `e =
/// indices[t, s]`. One dispatch covers all experts (`tgpig.z =
/// expert`); per-expert M-aware tile work via `htpe[e]` early-return.
///
/// `ne11 == 1`: `src1` is `[n_tokens, K]` — used for gate and up
/// projections (each token's activation feeds all k expert slots).
/// `ne11 == k`: `src1` is `[n_tokens, k, K]` — used for the down
/// projection (each (token, slot) has its own activation row from
/// the SwiGLU output).
#[derive(Clone, Copy)]
pub struct MoeGatherIdCall<'a> {
    /// Layer's expert-blob base — all `n_experts` experts stacked
    /// at per-expert stride `nb02`.
    pub src0: &'a BufferRef,
    pub src0_offset: u64,
    /// Activations (see `ne11` doc above).
    pub src1: &'a BufferRef,
    pub src1_offset: u64,
    /// `[n_experts]` u32 — from a prior [`MoeIdMap0Call`].
    pub htpe: &'a BufferRef,
    pub htpe_offset: u64,
    /// `[n_experts, n_tokens]` i32 — from a prior [`MoeIdMap0Call`].
    pub hids: &'a BufferRef,
    pub hids_offset: u64,
    /// Output `[n_tokens, k, n_out]` f32. Slot-indexed per token;
    /// reduce via [`MoeCombineTopkCall`].
    pub dst: &'a BufferRef,
    pub dst_offset: u64,
    /// K — contraction dim. Gate/up: hidden_dim. Down: moe_inter.
    pub k_in: u32,
    /// N — output dim per assignment. Gate/up: moe_inter. Down:
    /// hidden_dim.
    pub n_out: u32,
    pub n_experts: u32,
    pub n_tokens: u32,
    /// Top-k (must equal [`MOE_MM_ID_TOPK`]).
    pub k: u32,
    /// Activation broadcast dim. 1 for gate/up (no slot dim); k
    /// for down (per-slot activations).
    pub ne11: u32,
    /// Per-expert byte stride into `src0`. For a3b W4_gs64:
    /// `expert_size_4bit = 1,769,472`.
    pub nb02: u64,
    /// Packed-weight byte row stride. Gate/up: 1024. Down: 256.
    pub nb01_w: u64,
    /// Scales/biases byte row stride. Gate/up: 64. Down: 16.
    pub nb01_s: u64,
    /// Within-expert byte offset to this projection's packed bytes.
    /// Gate: 0. Up: `expert_block_bytes_4bit`. Down:
    /// `2 * expert_block_bytes_4bit`.
    pub packed_off: u64,
    /// Within-expert byte offset to this projection's scales.
    pub scales_off: u64,
    /// Within-expert byte offset to this projection's biases.
    pub biases_off: u64,
    /// Activation per-element byte stride. f32 → 4.
    pub nb10: u64,
    /// Activation per-slot byte stride. 0 for gate/up; `4 * K_in`
    /// for down.
    pub nb11: u64,
    /// Activation per-token byte stride. `4 * K_in` for gate/up;
    /// `4 * k * K_in` for down.
    pub nb12: u64,
}

/// Reduce `mid[n_tokens, k, hidden]` × `weights[n_tokens, k]` →
/// `out[n_tokens, hidden]`. Single small kernel; trivial vs the
/// matmul. Runs once per layer after gate/up/down.
#[derive(Clone, Copy)]
pub struct MoeCombineTopkCall<'a> {
    pub mid: &'a BufferRef,
    pub mid_offset: u64,
    pub weights: &'a BufferRef,
    pub weights_offset: u64,
    pub out: &'a BufferRef,
    pub out_offset: u64,
    pub n_tokens: u32,
    pub hidden_dim: u32,
    pub k: u32,
}

impl KernelCall for MoeIdMap0Call<'_> {
    fn encode(&self, kernels: &Kernels, cmdbuf: &CommandBufferRef) {
        if self.n_tokens == 0 || self.n_experts == 0 {
            return;
        }
        debug_assert_eq!(
            self.k, MOE_MM_ID_TOPK,
            "MoeIdMap0Call::k must equal MOE_MM_ID_TOPK ({} != {})",
            self.k, MOE_MM_ID_TOPK
        );

        let args = MoeMmIdMap0Args {
            n_experts: self.n_experts as i32,
            n_tokens: self.n_tokens as i32,
            k: self.k as i32,
        };

        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernels.moe_mm_id_map0_k8);
        enc.set_bytes(
            0,
            std::mem::size_of::<MoeMmIdMap0Args>() as NSUInteger,
            (&args as *const MoeMmIdMap0Args).cast(),
        );
        enc.set_buffer(1, Some(self.indices), self.indices_offset);
        enc.set_buffer(2, Some(self.htpe), self.htpe_offset);
        enc.set_buffer(3, Some(self.hids), self.hids_offset);
        enc.set_threadgroup_memory_length(0, MOE_MM_ID_MAP0_TG_MEM);

        // One threadgroup; one thread per expert.
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(self.n_experts as NSUInteger, 1, 1),
        );
        enc.end_encoding();
    }
}

impl KernelCall for MoeGatherIdCall<'_> {
    fn encode(&self, kernels: &Kernels, cmdbuf: &CommandBufferRef) {
        if self.n_tokens == 0 || self.n_experts == 0 {
            return;
        }
        debug_assert_eq!(
            self.k, MOE_MM_ID_TOPK,
            "MoeGatherIdCall::k must equal MOE_MM_ID_TOPK ({} != {})",
            self.k, MOE_MM_ID_TOPK
        );

        let args = MoeMmIdArgs {
            k_in: self.k_in as i32,
            n_out: self.n_out as i32,
            nb02: self.nb02,
            nb01_w: self.nb01_w,
            nb01_s: self.nb01_s,
            packed_off: self.packed_off,
            scales_off: self.scales_off,
            biases_off: self.biases_off,
            nb10: self.nb10,
            nb11: self.nb11,
            nb12: self.nb12,
            n_tokens: self.n_tokens as i32,
            k: self.k as i32,
            ne11: self.ne11 as i32,
            _pad: 0,
        };

        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernels.moe_mm_id);
        enc.set_bytes(
            0,
            std::mem::size_of::<MoeMmIdArgs>() as NSUInteger,
            (&args as *const MoeMmIdArgs).cast(),
        );
        enc.set_buffer(1, Some(self.src0), self.src0_offset);
        enc.set_buffer(2, Some(self.src1), self.src1_offset);
        enc.set_buffer(3, Some(self.htpe), self.htpe_offset);
        enc.set_buffer(4, Some(self.hids), self.hids_offset);
        enc.set_buffer(5, Some(self.dst), self.dst_offset);
        enc.set_threadgroup_memory_length(0, MOE_MM_ID_TG_MEM);

        // Grid: (ceil(n_tokens/NR1), ceil(n_out/NR0), n_experts).
        let grid_x = self.n_tokens.div_ceil(MOE_MM_ID_NR1) as NSUInteger;
        let grid_y = self.n_out.div_ceil(MOE_MM_ID_NR0) as NSUInteger;
        enc.dispatch_thread_groups(
            MTLSize::new(grid_x, grid_y, self.n_experts as NSUInteger),
            MTLSize::new(MOE_MM_ID_THREADS, 1, 1),
        );
        enc.end_encoding();
    }
}

impl KernelCall for MoeCombineTopkCall<'_> {
    fn encode(&self, kernels: &Kernels, cmdbuf: &CommandBufferRef) {
        if self.n_tokens == 0 || self.hidden_dim == 0 {
            return;
        }
        let args = MoeCombineTopkArgs {
            n_tokens: self.n_tokens as i32,
            hidden_dim: self.hidden_dim as i32,
            k: self.k as i32,
        };

        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernels.moe_combine_topk);
        enc.set_bytes(
            0,
            std::mem::size_of::<MoeCombineTopkArgs>() as NSUInteger,
            (&args as *const MoeCombineTopkArgs).cast(),
        );
        enc.set_buffer(1, Some(self.mid), self.mid_offset);
        enc.set_buffer(2, Some(self.weights), self.weights_offset);
        enc.set_buffer(3, Some(self.out), self.out_offset);

        // One threadgroup per token; 128 threads stride over
        // hidden_dim. Could tune later if `combine_topk` shows up
        // in profile.
        enc.dispatch_thread_groups(
            MTLSize::new(self.n_tokens as NSUInteger, 1, 1),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
    }
}

#[cfg(test)]
mod gather_mm_id_tests {
    use super::*;
    use metal::Device;

    /// Build the full `Kernels` set, including the new MoE pipelines.
    /// Failure here = Metal compile error in `gather_mm_id.metal`.
    /// Fast — no kernel launch, just library + PSO build.
    #[test]
    fn moe_mm_id_pipelines_compile() {
        let device = Device::system_default()
            .expect("a Metal device for moe_mm_id compile test");
        let _kernels =
            Kernels::new(&device).expect("compile gather_mm_id.metal");
    }
}

