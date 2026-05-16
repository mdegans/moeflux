//! Per-layer GPU execution context.
//!
//! The layer-forward functions all thread the same five shared
//! borrows — the weight file, its Metal-resident mirror, the layer
//! weight cache, the per-layer buffer set, and the buffer pool — in
//! lockstep. [`GpuLayerCtx`] bundles them so a single `&GpuLayerCtx`
//! replaces five parameters at every call site.
//!
//! The exclusive `&mut MetalContext` is deliberately *not* bundled:
//! it is borrowed mutably (pipeline fetches mutate the pipeline
//! cache), so it stays a separate parameter to keep the struct a
//! cheap `Copy` of shared references.

use crate::riir::backend::MetalBufferPool;
use crate::riir::io::layer_weight_cache::LayerWeightCache;
use crate::riir::attn::linear_attn_forward::LayerForwardBuffers;
use crate::riir::io::mtl_weight_buf::MtlWeightBuf;
use crate::riir::io::weight_file::WeightFile;

/// The shared-borrow half of a layer-forward call's context. See the
/// module docs for why `&mut MetalContext` is excluded.
///
/// All fields are shared references, so the struct is `Copy` — passing
/// it by value or `&` is equally cheap, and it compiles away entirely.
#[derive(Clone, Copy)]
pub struct GpuLayerCtx<'a> {
    /// The on-disk weight file (tensor metadata + mmap).
    pub wf: &'a WeightFile,
    /// Metal-resident mirror of the resident weight tensors.
    pub wf_buf: &'a MtlWeightBuf,
    /// Per-layer resolved weight offsets / bit widths.
    pub layer_cache: &'a LayerWeightCache,
    /// The per-layer GPU buffer set (input, normed, residual, …).
    pub buffers: &'a LayerForwardBuffers,
    /// The lifetime-colored Metal buffer pool.
    pub buffer_pool: &'a MetalBufferPool,
}
