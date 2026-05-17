//! `MetalBackend` — wires the `Backend` trait against the existing
//! Metal kernels.
//!
//! Wraps the existing `encode_X_into` helpers. The pool stores
//! `metal::Buffer` directly; `encode_op` writes compute dispatches into
//! a `MetalEncodeCtx` that owns a `CommandBuffer`; `submit_and_wait`
//! commits the cmdbuf and blocks.
//!
//! **S7-3 / S7-6b scope:** 13 of 15 `Op` variants are fully wired.
//! `LmHead` (workspace BufId) and `SdpaCausalTiled` (kv_dim
//! disambiguation) remain as `todo!()` and will wire alongside the
//! full-attn producer rewrite in S7-7.

pub mod dense_mlp_gpu;
pub mod encoder;
pub mod gpu_ctx;
pub mod gpu_matvec;
pub mod gpu_norm;
pub mod metal;

use super::{Backend, BufId, BufferPool, Graph, GraphError, Op};

use ::metal::{Buffer, CommandBuffer, MTLSize, NSRange};
use metal::{
    CommandBufferRef, ComputePipelineState, Device, MTLResourceOptions, NSUInteger,
};

use crate::riir::attn::gpu_attn::FlashSdpaPipelines;
use crate::riir::attn::gpu_linear_attn::LinearAttnPipelines;
use crate::riir::backend::gpu::gpu_matvec::{
    encode_matvec_n_tokens, BfMatvecPipelines, MatvecPipelines,
};
use crate::riir::moe::gpu_moe_router::MoeRouterPipelines;
use crate::riir::backend::gpu::gpu_norm::{
    encode_residual_add_n_tokens_into, encode_rms_norm_bf16_fused_n_tokens,
    RmsNormBf16FusedNTokensPipeline, RmsNormBf16Pipelines,
};
use crate::riir::backend::gpu::metal::{MetalContext, MetalError, MtlBuffer};
use crate::riir::io::mtl_weight_buf::MtlWeightBuf;
use moeflux_mlx::{QmmCall, QuantWeights};

/// Metal buffer pool. Storage is `Vec<Buffer>` indexed *indirectly*
/// by `BufId` through `bufid_to_physical`. Pre-`commit_plan` the
/// mapping is identity; after `commit_plan`, colorable BufIds may
/// share a single `metal::Buffer`.
pub struct MetalBufferPool {
    device: Device,
    buffers: Vec<Buffer>,
    labels: Vec<&'static str>,
    persistent: Vec<bool>,
    byte_sizes: Vec<usize>,
    bufid_to_physical: Vec<u32>,
    /// S10b-pre-1 — anchors for `alloc_aligned`-created buffers.
    /// `MtlBuffer::with_aligned_len_u8` uses `newBufferWithBytesNoCopy:`
    /// with `deallocator=None`; the wrapper's `AlignedBacking` owns
    /// the heap memory the Metal buffer points at. Keeping the
    /// `MtlBuffer` alive in this vec keeps the backing alive for the
    /// lifetime of the pool. Append-only; order is independent of
    /// `buffers` because we never index into it.
    aligned_anchors: Vec<MtlBuffer<u8>>,
}

impl MetalBufferPool {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            buffers: Vec::new(),
            labels: Vec::new(),
            persistent: Vec::new(),
            byte_sizes: Vec::new(),
            bufid_to_physical: Vec::new(),
            aligned_anchors: Vec::new(),
        }
    }

    pub fn physical_buffer_count(&self) -> usize {
        self.buffers.len()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Allocate a buffer of `bytes` bytes aligned to `alignment`.
    /// `alignment` must be a power of two.
    ///
    /// Backed by posix_memalign-equivalent heap memory wrapped via
    /// `newBufferWithBytesNoCopy:` (deallocator=None). The pool owns
    /// the heap memory; the Metal buffer references it. Required for
    /// pread DMA destinations — the C path documents a 3.6× DMA win
    /// from 2 MiB alignment over 16 KB (`metal_infer/infer.m:1196`).
    pub fn alloc_aligned(
        &mut self,
        bytes: usize,
        alignment: usize,
        label: &'static str,
        persistent: bool,
    ) -> BufId {
        let anchor = MtlBuffer::<u8>::with_aligned_len_u8(
            &self.device,
            bytes,
            alignment,
        );
        let buf = anchor.buffer().clone();
        let id = BufId(self.bufid_to_physical.len() as u32);
        let physical = self.buffers.len() as u32;
        self.buffers.push(buf);
        self.aligned_anchors.push(anchor);
        self.labels.push(label);
        self.persistent.push(persistent);
        self.byte_sizes.push(bytes);
        self.bufid_to_physical.push(physical);
        id
    }

    /// Register an externally-owned `metal::Buffer` as a pool slot.
    /// The pool stores a refcounted clone — for Metal that's an
    /// NSObject retain, essentially free. The caller continues to own
    /// the underlying memory's lifetime guarantee (e.g. an mmap that
    /// outlives the pool, or another retainer of the same `Buffer`).
    ///
    /// `bytes` is informational — used for `upload` / `download` size
    /// checks and for `as_mut_slice_u8` length. Pass the buffer's
    /// actual length.
    ///
    /// Used for mmap'd expert files: `ExpertFiles` owns the `Mmap`
    /// + the wrapping `Buffer`; the pool registers the `Buffer` so
    /// graph-mode `Op`s can address it via `BufId`.
    pub fn register_borrowed(
        &mut self,
        buf: Buffer,
        bytes: usize,
        label: &'static str,
        persistent: bool,
    ) -> BufId {
        let id = BufId(self.bufid_to_physical.len() as u32);
        let physical = self.buffers.len() as u32;
        self.buffers.push(buf);
        self.labels.push(label);
        self.persistent.push(persistent);
        self.byte_sizes.push(bytes);
        self.bufid_to_physical.push(physical);
        id
    }

    /// Returns the buffer's underlying memory as a mut byte slice.
    ///
    /// SAFETY/CORRECTNESS: same discipline as today's
    /// `MtlBuffer::as_mut_slice` — caller ensures no GPU dispatch
    /// is reading from this buffer concurrently. `&self` because
    /// `metal::Buffer` is NSObject-interior-mut; Rust borrow rules
    /// don't apply to its contents.
    pub fn as_mut_slice_u8(&self, id: BufId) -> &mut [u8] {
        let idx = id.0 as usize;
        let physical = self.bufid_to_physical[idx] as usize;
        let buf = &self.buffers[physical];
        let bytes = self.byte_sizes[idx];
        // SAFETY: `buf.contents()` is non-null for any shared-storage
        // Metal buffer (Apple guarantee). The byte count matches the
        // allocation; aliasing concerns are caller-managed per the
        // contract above.
        unsafe {
            std::slice::from_raw_parts_mut(
                buf.contents() as *mut u8,
                bytes,
            )
        }
    }

    /// Disjoint mut byte slices for the common pread-worker pattern.
    /// Same soundness contract as [`Self::as_mut_slice_u8`] per slot,
    /// PLUS the caller guarantees the `ids` array is duplicate-free —
    /// otherwise the returned references alias and the &mut semantics
    /// is violated.
    pub fn as_mut_slices_u8<const N: usize>(
        &self,
        ids: [BufId; N],
    ) -> [&mut [u8]; N] {
        ids.map(|id| self.as_mut_slice_u8(id))
    }
}

impl BufferPool for MetalBufferPool {
    type Handle = Buffer;
    type Error = GraphError;

    fn alloc(
        &mut self,
        bytes: usize,
        label: &'static str,
        persistent: bool,
    ) -> Result<BufId, GraphError> {
        let id = BufId(self.bufid_to_physical.len() as u32);
        let physical = self.buffers.len() as u32;
        let buf = self.device.new_buffer(
            bytes as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        // Zero on alloc so encoders that assume a clean slot don't
        // read stale memory. Matches the CPU pool's vec![0u8; bytes]
        // behaviour.
        unsafe {
            std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
        }
        self.buffers.push(buf);
        self.labels.push(label);
        self.persistent.push(persistent);
        self.byte_sizes.push(bytes);
        self.bufid_to_physical.push(physical);
        Ok(id)
    }

    fn handle(&self, id: BufId) -> &Buffer {
        let physical = self.bufid_to_physical[id.0 as usize] as usize;
        &self.buffers[physical]
    }

    fn upload(&mut self, id: BufId, host: &[u8]) -> Result<(), GraphError> {
        let idx = id.0 as usize;
        let label = *self.labels.get(idx).ok_or(GraphError::BadBufId(id))?;
        let expected = self.byte_sizes[idx];
        // Prefix semantics: `host` may be shorter than the buffer
        // (once-per-run buffers are sized at max chunk width; a
        // smaller chunk uploads only its rows). Too-large is rejected.
        if host.len() > expected {
            return Err(GraphError::SizeMismatch {
                label,
                expected,
                actual: host.len(),
            });
        }
        let physical = self.bufid_to_physical[idx] as usize;
        let buf = &self.buffers[physical];
        unsafe {
            std::ptr::copy_nonoverlapping(
                host.as_ptr(),
                buf.contents() as *mut u8,
                host.len(),
            );
        }
        Ok(())
    }

    fn upload_at(
        &mut self,
        id: BufId,
        offset: usize,
        host: &[u8],
    ) -> Result<(), GraphError> {
        let idx = id.0 as usize;
        let label = *self.labels.get(idx).ok_or(GraphError::BadBufId(id))?;
        let expected = self.byte_sizes[idx];
        if offset + host.len() > expected {
            return Err(GraphError::SizeMismatch {
                label,
                expected,
                actual: offset + host.len(),
            });
        }
        let physical = self.bufid_to_physical[idx] as usize;
        let buf = &self.buffers[physical];
        unsafe {
            std::ptr::copy_nonoverlapping(
                host.as_ptr(),
                (buf.contents() as *mut u8).add(offset),
                host.len(),
            );
        }
        Ok(())
    }

    fn download(
        &self,
        id: BufId,
        host: &mut [u8],
    ) -> Result<(), GraphError> {
        let idx = id.0 as usize;
        let label = *self.labels.get(idx).ok_or(GraphError::BadBufId(id))?;
        let expected = self.byte_sizes[idx];
        // Prefix semantics: see `upload`.
        if host.len() > expected {
            return Err(GraphError::SizeMismatch {
                label,
                expected,
                actual: host.len(),
            });
        }
        let physical = self.bufid_to_physical[idx] as usize;
        let buf = &self.buffers[physical];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.contents() as *const u8,
                host.as_mut_ptr(),
                host.len(),
            );
        }
        Ok(())
    }

    fn reset_transient(&mut self) {
        // Mirrors CpuBufferPool::reset_transient: keep the persistent
        // prefix in BufId space; drop physical buffers no longer
        // referenced. After `commit_plan`, persistents retain their
        // original physical indices.
        let mut keep_bufids = 0;
        for (i, &p) in self.persistent.iter().enumerate() {
            if p {
                keep_bufids = i + 1;
            }
        }
        self.labels.truncate(keep_bufids);
        self.persistent.truncate(keep_bufids);
        self.byte_sizes.truncate(keep_bufids);
        self.bufid_to_physical.truncate(keep_bufids);

        let max_physical = self
            .bufid_to_physical
            .iter()
            .copied()
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(0);
        self.buffers.truncate(max_physical);
    }

    fn label(&self, id: BufId) -> &'static str {
        self.labels
            .get(id.0 as usize)
            .copied()
            .unwrap_or("<bad-bufid>")
    }

    fn commit_plan(&mut self, graph: &Graph) {
        use super::lifetime::{analyze_lifetimes, greedy_color, ColorId};
        use std::collections::HashMap;

        let lifetimes = analyze_lifetimes(graph);
        let coloring = greedy_color(&lifetimes);

        let n_bufids = self.bufid_to_physical.len();
        let aliasable: HashMap<BufId, ColorId> = coloring
            .bufid_to_color
            .iter()
            .filter(|(b, _)| !self.persistent[b.0 as usize])
            .map(|(b, c)| (*b, *c))
            .collect();

        // Phase 1: place non-aliasable BufIds (persistent + non-
        // colorable transients) in the new layout, preserving the
        // underlying metal::Buffer (and its content) via swap.
        //
        // A prior `commit_plan` may already have aliased BufIds onto
        // a shared physical buffer, so several non-aliasable BufIds
        // can map to the same `old_physical`. Move each physical
        // exactly once (`old_to_new`) and remap every BufId that
        // shared it to that single new slot — otherwise the second
        // and later BufIds would swap out an already-moved-away
        // placeholder.
        let mut new_buffers: Vec<Buffer> = Vec::new();
        let mut new_bufid_to_physical: Vec<u32> = vec![u32::MAX; n_bufids];
        let mut old_to_new: HashMap<usize, u32> = HashMap::new();

        // We need a placeholder Buffer to swap with — use a 1-byte
        // throwaway allocation, deferred to the first swap.
        let placeholder = self
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);

        for bufid_idx in 0..n_bufids {
            let buf = BufId(bufid_idx as u32);
            if aliasable.contains_key(&buf) {
                continue;
            }
            let old_physical = self.bufid_to_physical[bufid_idx] as usize;
            let new_phys = *old_to_new.entry(old_physical).or_insert_with(|| {
                let old_buf = std::mem::replace(
                    &mut self.buffers[old_physical],
                    placeholder.clone(),
                );
                let np = new_buffers.len() as u32;
                new_buffers.push(old_buf);
                np
            });
            new_bufid_to_physical[bufid_idx] = new_phys;
        }

        // Phase 2: one Metal buffer per color, sized to max(byte_size).
        let mut color_to_physical: HashMap<ColorId, u32> = HashMap::new();
        for color in 0..coloring.color_count {
            let max_size = aliasable
                .iter()
                .filter(|&(_, c)| *c == color)
                .map(|(b, _)| self.byte_sizes[b.0 as usize])
                .max()
                .unwrap_or(0);
            if max_size == 0 {
                continue;
            }
            let buf = self.device.new_buffer(
                max_size as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            );
            unsafe {
                std::ptr::write_bytes(
                    buf.contents() as *mut u8,
                    0,
                    max_size,
                );
            }
            color_to_physical.insert(color, new_buffers.len() as u32);
            new_buffers.push(buf);
        }

        for (buf, color) in &aliasable {
            let phys = color_to_physical[color];
            new_bufid_to_physical[buf.0 as usize] = phys;
        }

        debug_assert!(new_bufid_to_physical.iter().all(|&p| p != u32::MAX));
        self.buffers = new_buffers;
        self.bufid_to_physical = new_bufid_to_physical;

        // S10b-2: pin every colored BufId. Its physical layout is now
        // frozen for the run; flipping `persistent` keeps it (and the
        // shared color buffer it points at) across `reset_transient`.
        for buf in aliasable.keys() {
            self.persistent[buf.0 as usize] = true;
        }
    }
}

/// Encoding context for [`MetalBackend`]: owns a `CommandBuffer`
/// that `encode_op` appends dispatches to and `submit_and_wait`
/// commits + waits on.
pub struct MetalEncodeCtx {
    cmdbuf: CommandBuffer,
}

/// Metal `Backend` trait impl.
///
/// Composes a renamed [`MetalContext`] (device + library + pipeline
/// cache + stats) and a shared [`MtlWeightBuf`] (mmap'd weight file
/// wrapped as a Metal buffer). Pre-fetches all pipelines we touch at
/// construction time so `encode_op` can stay `&self`-typed and
/// thread-friendly.
pub struct MetalBackend {
    metal: MetalContext,
    wf_buf: MtlWeightBuf,
    pool: MetalBufferPool,
    // Pre-warmed pipeline caches.
    matvec_pipes: MatvecPipelines,
    #[allow(dead_code)]
    bf_matvec_pipes: BfMatvecPipelines,
    rms_n_pipe: RmsNormBf16FusedNTokensPipeline,
    #[allow(dead_code)]
    rms_pipes: RmsNormBf16Pipelines,
    router_pipes: MoeRouterPipelines,
    #[allow(dead_code)]
    sdpa_pipes: FlashSdpaPipelines,
    #[allow(dead_code)]
    linear_attn_pipes: LinearAttnPipelines,
    residual_add_n_pso: ComputePipelineState,
    swiglu_fused_batched_pso: ComputePipelineState,
    swiglu_fused_pso: ComputePipelineState,
    moe_combine_residual_n_pso: ComputePipelineState,
    moe_bucket_accumulate_pso: ComputePipelineState,
    /// When set (env `MOEFLUX_PROFILE_PER_OP`), [`Backend::execute`]
    /// commits each op as its own labeled cmdbuf so `prefill_profile`
    /// reports a per-op breakdown instead of one figure per graph.
    /// Instrumentation only — it forfeits the S7-1a commit fusion, so
    /// it inflates wall time; use it for proportion analysis, never
    /// for an absolute bench.
    profile_per_op: bool,
    /// When false (env `MOEFLUX_MOE_GATHER=0`), `MoeBatchedPermuteFuse`
    /// uses the per-bucket matvec fallback instead of the MLX
    /// `affine_gather_qmm_rhs` GEMM path. Default on; the `=0` escape
    /// hatch keeps the slower path reachable for A/B and bisecting.
    moe_gather: bool,
}

impl MetalBackend {
    pub fn new(
        mut metal: MetalContext,
        wf_buf: MtlWeightBuf,
    ) -> Result<Self, MetalError> {
        // Fetch all pipelines we'll need. Each is a cheap NSObject
        // refcount bump after first compilation; subsequent
        // operations reuse the cache.
        let matvec_pipes = MatvecPipelines::fetch(&mut metal)?;
        let bf_matvec_pipes = BfMatvecPipelines::fetch(&mut metal)?;
        let rms_n_pipe = RmsNormBf16FusedNTokensPipeline::fetch(&mut metal)?;
        let rms_pipes = RmsNormBf16Pipelines::fetch(&mut metal)?;
        let router_pipes = MoeRouterPipelines::fetch(&mut metal)?;
        let sdpa_pipes = FlashSdpaPipelines::fetch(&mut metal)?;
        let linear_attn_pipes = LinearAttnPipelines::fetch(&mut metal)?;
        let residual_add_n_pso =
            metal.pipeline("residual_add_n_tokens")?.clone();
        let swiglu_fused_batched_pso =
            metal.pipeline("swiglu_fused_batched")?.clone();
        let swiglu_fused_pso = metal.pipeline("swiglu_fused")?.clone();
        let moe_combine_residual_n_pso =
            metal.pipeline("moe_combine_residual_n_tokens")?.clone();
        let moe_bucket_accumulate_pso =
            metal.pipeline("moe_bucket_accumulate")?.clone();

        let device = metal.device().clone();
        Ok(Self {
            metal,
            wf_buf,
            pool: MetalBufferPool::new(device),
            matvec_pipes,
            bf_matvec_pipes,
            rms_n_pipe,
            rms_pipes,
            router_pipes,
            sdpa_pipes,
            linear_attn_pipes,
            residual_add_n_pso,
            swiglu_fused_batched_pso,
            swiglu_fused_pso,
            moe_combine_residual_n_pso,
            moe_bucket_accumulate_pso,
            profile_per_op: std::env::var_os("MOEFLUX_PROFILE_PER_OP")
                .is_some(),
            moe_gather: std::env::var("MOEFLUX_MOE_GATHER")
                .map_or(true, |v| v != "0"),
        })
    }

    pub fn metal(&self) -> &MetalContext {
        &self.metal
    }

    pub fn metal_mut(&mut self) -> &mut MetalContext {
        &mut self.metal
    }

    pub fn weight_buf(&self) -> &MtlWeightBuf {
        &self.wf_buf
    }

    /// Disjoint mutable borrow of the three graph-mode fields.
    /// Lets `RsCtx::ensure_*_resources` and the imperative MLA
    /// step body pass `(&mut MetalContext, &MtlWeightBuf, &mut
    /// MetalBufferPool)` to existing helpers without manually
    /// splitting the borrow at each call site.
    pub fn parts_mut(
        &mut self,
    ) -> (&mut MetalContext, &MtlWeightBuf, &mut MetalBufferPool) {
        (&mut self.metal, &self.wf_buf, &mut self.pool)
    }
}

/// Construction inputs for [`MetalBackend::open`]. Carries the
/// already-built [`MetalContext`] (device + library + queue +
/// pipeline cache) and the mmap'd weight file wrapped as a Metal
/// buffer. The backend takes ownership of both.
pub struct MetalConfig {
    pub metal: MetalContext,
    pub wf_buf: MtlWeightBuf,
}

impl Backend for MetalBackend {
    type Pool = MetalBufferPool;
    type EncodeCtx = MetalEncodeCtx;
    type Config = MetalConfig;
    type Error = GraphError;

    fn open(config: MetalConfig) -> Result<Self, GraphError>
    where
        Self: Sized,
    {
        Self::new(config.metal, config.wf_buf)
            .map_err(|e| GraphError::Backend(Box::new(e)))
    }

    fn pool(&self) -> &MetalBufferPool {
        &self.pool
    }
    fn pool_mut(&mut self) -> &mut MetalBufferPool {
        &mut self.pool
    }

    fn begin_encoding(&self) -> MetalEncodeCtx {
        let cmdbuf = self.metal.queue().new_command_buffer().to_owned();
        MetalEncodeCtx { cmdbuf }
    }

    fn submit_and_wait(
        &self,
        ctx: MetalEncodeCtx,
        label: &'static str,
    ) -> Result<(), GraphError> {
        self.metal.commit_and_wait_labeled(&ctx.cmdbuf, label);
        Ok(())
    }

    fn execute(
        &self,
        graph: &Graph,
        label: &'static str,
    ) -> Result<(), GraphError> {
        if self.profile_per_op {
            // Instrumentation: one labeled commit per op so
            // `cmdbuf_stats` carries a per-op breakdown. Forfeits the
            // commit fusion — see `profile_per_op`.
            for op in &graph.ops {
                let mut ctx = self.begin_encoding();
                self.encode_op(op, &mut ctx);
                self.submit_and_wait(ctx, op.label())?;
            }
            return Ok(());
        }
        let mut ctx = self.begin_encoding();
        self.encode_graph(graph, &mut ctx);
        self.submit_and_wait(ctx, label)
    }

    fn encode_op(&self, op: &Op, ctx: &mut MetalEncodeCtx) {
        let cmd: &CommandBufferRef = &ctx.cmdbuf;
        match op {
            Op::RmsNormBf16NTokens {
                x,
                weight_off,
                out,
                dim,
                n_tokens,
                eps,
                ..
            } => {
                encode_rms_norm_bf16_fused_n_tokens(
                    cmd,
                    &self.rms_n_pipe,
                    self.pool.handle(*x),
                    self.wf_buf.buffer(),
                    *weight_off,
                    self.pool.handle(*out),
                    *dim,
                    *n_tokens,
                    *eps,
                );
            }
            Op::ResidualAddNTokens {
                a,
                b,
                out,
                n_tokens,
                dim,
                ..
            } => {
                encode_residual_add_n_tokens_into(
                    cmd,
                    &self.residual_add_n_pso,
                    self.pool.handle(*a),
                    self.pool.handle(*b),
                    self.pool.handle(*out),
                    *n_tokens,
                    *dim,
                );
            }
            Op::ZeroBuffer { buf, n_bytes, .. } => {
                let blit = cmd.new_blit_command_encoder();
                blit.fill_buffer(
                    self.pool.handle(*buf),
                    NSRange::new(0, *n_bytes as NSUInteger),
                    0,
                );
                blit.end_encoding();
            }
            Op::MatvecNTokens {
                weight,
                input,
                input_off,
                output,
                output_off,
                in_dim,
                out_dim,
                n_tokens,
                ..
            } => {
                if weight.bits == 4 {
                    // 4-bit: MLX's tiled quantized GEMM (~12x the old
                    // dequant_matvec_4bit matvec at prefill shapes).
                    self.metal.qmm().encode_qmm_t(
                        cmd,
                        &QmmCall {
                            weights: QuantWeights {
                                buffer: self.wf_buf.buffer(),
                                packed_offset: weight.w_off,
                                scales_offset: weight.s_off,
                                biases_offset: weight.b_off,
                            },
                            input: self.pool.handle(*input),
                            input_offset: *input_off,
                            output: self.pool.handle(*output),
                            output_offset: *output_off,
                            in_dim: *in_dim,
                            out_dim: *out_dim,
                            n_tokens: *n_tokens,
                        },
                    );
                } else {
                    // 8-bit (a3b mlp.gate / shared_expert_gate) stays on
                    // the per-token dequant matvec — moeflux-mlx's qmm_t
                    // is instantiated 4-bit only.
                    encode_matvec_n_tokens(
                        cmd,
                        &self.matvec_pipes,
                        self.wf_buf.buffer(),
                        weight.w_off,
                        weight.s_off,
                        weight.b_off,
                        self.pool.handle(*input),
                        *input_off,
                        self.pool.handle(*output),
                        *output_off,
                        *in_dim,
                        *out_dim,
                        *n_tokens,
                        weight.bits,
                    );
                }
            }
            Op::SwigluFusedBatched {
                gate,
                up,
                out,
                total,
                ..
            } => {
                // Kernel takes (dim, K) where total = K * dim; the inner
                // loop only sees `total`. Pass K=1 and dim=total so the
                // arithmetic resolves to our flat dispatch shape.
                let dim = *total;
                let k_one: u32 = 1;
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.swiglu_fused_batched_pso);
                enc.set_buffer(0, Some(self.pool.handle(*gate)), 0);
                enc.set_buffer(1, Some(self.pool.handle(*up)), 0);
                enc.set_buffer(2, Some(self.pool.handle(*out)), 0);
                enc.set_bytes(3, 4, (&dim as *const u32).cast());
                enc.set_bytes(4, 4, (&k_one as *const u32).cast());
                let num_tgs = (*total + 255) / 256;
                enc.dispatch_thread_groups(
                    MTLSize::new(num_tgs as NSUInteger, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                enc.end_encoding();
            }
            Op::MoeSoftmaxTopK {
                logits,
                indices_out,
                weights_out,
                n_tokens,
                n_experts,
                k,
                ..
            } => {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.router_pipes.softmax_topk);
                enc.set_buffer(0, Some(self.pool.handle(*logits)), 0);
                enc.set_buffer(1, Some(self.pool.handle(*indices_out)), 0);
                enc.set_buffer(2, Some(self.pool.handle(*weights_out)), 0);
                enc.set_bytes(3, 4, (n_experts as *const u32).cast());
                enc.set_bytes(4, 4, (k as *const u32).cast());
                enc.dispatch_thread_groups(
                    MTLSize::new(*n_tokens as NSUInteger, 1, 1),
                    MTLSize::new(64, 1, 1),
                );
                enc.end_encoding();
            }
            Op::MoeNormalizeWeights {
                weights,
                n_tokens,
                k,
                ..
            } => {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.router_pipes.normalize);
                enc.set_buffer(0, Some(self.pool.handle(*weights)), 0);
                enc.set_bytes(1, 4, (k as *const u32).cast());
                enc.dispatch_thread_groups(
                    MTLSize::new(*n_tokens as NSUInteger, 1, 1),
                    MTLSize::new(*k as NSUInteger, 1, 1),
                );
                enc.end_encoding();
            }
            Op::MoeCombineResidualNTokens {
                h_mid,
                moe_sum,
                shared_out,
                shared_gate,
                hidden_out,
                n_tokens,
                dim,
                ..
            } => {
                crate::riir::moe::expert_forward::encode_moe_combine_residual_n_tokens(
                    cmd,
                    &self.moe_combine_residual_n_pso,
                    self.pool.handle(*h_mid),
                    self.pool.handle(*moe_sum),
                    self.pool.handle(*shared_out),
                    self.pool.handle(*shared_gate),
                    self.pool.handle(*hidden_out),
                    *n_tokens,
                    *dim,
                );
            }
            Op::LmHead { .. } => {
                todo!(
                    "LmHead encode_op: needs a persistent workspace BufId in Op shape for the intermediate (post-final-norm) hidden bytes; defer to S7-7 producer wire-up where the orchestrator can allocate it"
                );
            }
            Op::RmsNormQkNTokens {
                x,
                num_k_heads,
                key_dim,
                key_offset_per_token,
                per_token_total,
                n_tokens,
                ..
            } => {
                // In-place per-head RMS-norm on q and k regions of `x`.
                // Single batched dispatch: `(num_k_heads, n_tokens)`
                // threadgroups × `key_dim` threads. Each token's slot
                // is `per_token_total` floats; q region at offset 0, k
                // region at offset `key_offset_per_token`. For q|k|v
                // layouts (linear-attn `conv_out`) `per_token_total`
                // includes the V region trailing K. Matches
                // `rms_norm_qk_n_tokens_cpu`.
                let inv_scale = 1.0f32 / (*key_dim as f32).sqrt();
                let x_buf = self.pool.handle(*x);
                let key_dim_arg = *key_dim;
                let ptt = *per_token_total;
                let kopt = *key_offset_per_token;
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(
                    &self.linear_attn_pipes.rms_norm_qk,
                );
                enc.set_buffer(0, Some(x_buf), 0);
                enc.set_bytes(1, 4, (&key_dim_arg as *const u32).cast());
                enc.set_bytes(2, 4, (&inv_scale as *const f32).cast());
                enc.set_bytes(3, 4, (&ptt as *const u32).cast());
                enc.set_bytes(4, 4, (&kopt as *const u32).cast());
                enc.dispatch_thread_groups(
                    MTLSize::new(
                        *num_k_heads as NSUInteger,
                        *n_tokens as NSUInteger,
                        1,
                    ),
                    MTLSize::new(*key_dim as NSUInteger, 1, 1),
                );
                enc.end_encoding();
            }
            Op::SdpaCausalTiled { .. } => {
                todo!(
                    "SdpaCausalTiled encode_op: needs kv_dim arg disambiguation; defer to S7-7 producer wire-up"
                )
            }
            Op::MoeBatchedPermuteFuse {
                expert_base,
                expert_stride,
                expert_indices,
                expert_slots,
                bucket_input,
                bucket_gate,
                bucket_up,
                bucket_act,
                bucket_out,
                bucket_token_idx,
                bucket_weights,
                out_sum,
                buckets,
                ..
            } => {
                crate::riir::moe::expert_forward::encode_moe_batched_permute_fuse(
                    cmd,
                    &self.matvec_pipes,
                    self.metal.qmm(),
                    &self.swiglu_fused_pso,
                    &self.moe_bucket_accumulate_pso,
                    self.pool.handle(*expert_base),
                    *expert_stride,
                    self.pool.handle(*expert_indices),
                    expert_slots,
                    self.pool.handle(*bucket_input),
                    self.pool.handle(*bucket_gate),
                    self.pool.handle(*bucket_up),
                    self.pool.handle(*bucket_act),
                    self.pool.handle(*bucket_out),
                    self.pool.handle(*bucket_token_idx),
                    self.pool.handle(*bucket_weights),
                    self.pool.handle(*out_sum),
                    buckets,
                    crate::riir::variants::VARIANT,
                    self.moe_gather,
                );
            }
            Op::Conv1dStepNTokens {
                qkv_in,
                conv_state,
                weight_off,
                conv_out,
                conv_dim,
                n_tokens,
                ..
            } => {
                // Two batched dispatches in one cmdbuf. Pass 1
                // (`conv1d_step`) reads `conv_state` + the whole
                // `qkv_in` chunk and writes `conv_out`; pass 2
                // (`conv1d_state_update`) reads the originals and
                // overwrites `conv_state` with the chunk-tail history.
                // The split avoids the cross-token-threadgroup state
                // read/write hazard a single kernel would have.
                let qkv_buf = self.pool.handle(*qkv_in);
                let state_buf = self.pool.handle(*conv_state);
                let conv_out_buf = self.pool.handle(*conv_out);
                let conv_dim_arg = *conv_dim;
                let n_tokens_arg = *n_tokens;
                let num_tgs = (conv_dim_arg + 255) / 256;
                // Pass 1 — compute.
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(
                    &self.linear_attn_pipes.conv1d_step,
                );
                enc.set_buffer(0, Some(state_buf), 0);
                enc.set_buffer(1, Some(qkv_buf), 0);
                enc.set_buffer(
                    2,
                    Some(self.wf_buf.buffer()),
                    *weight_off as NSUInteger,
                );
                enc.set_buffer(3, Some(conv_out_buf), 0);
                enc.set_bytes(4, 4, (&conv_dim_arg as *const u32).cast());
                enc.dispatch_thread_groups(
                    MTLSize::new(
                        num_tgs as NSUInteger,
                        n_tokens_arg as NSUInteger,
                        1,
                    ),
                    MTLSize::new(256, 1, 1),
                );
                enc.end_encoding();
                // Pass 2 — history-state update.
                let enc2 = cmd.new_compute_command_encoder();
                enc2.set_compute_pipeline_state(
                    &self.linear_attn_pipes.conv1d_state_update,
                );
                enc2.set_buffer(0, Some(state_buf), 0);
                enc2.set_buffer(1, Some(qkv_buf), 0);
                enc2.set_bytes(2, 4, (&conv_dim_arg as *const u32).cast());
                enc2.set_bytes(3, 4, (&n_tokens_arg as *const u32).cast());
                enc2.dispatch_thread_groups(
                    MTLSize::new(num_tgs as NSUInteger, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                enc2.end_encoding();
            }
            Op::ComputeDecayBetaNTokens {
                alpha_in,
                beta_in,
                a_log_off,
                dt_bias_off,
                g_decay_out,
                beta_gate_out,
                num_v_heads,
                n_tokens,
                ..
            } => {
                // Single batched dispatch: `(n_tokens)` threadgroups ×
                // `(num_v_heads)` threads. alpha / beta / g_decay /
                // beta_gate are token-major `[n_tokens * num_v_heads]`;
                // the kernel flattens `idx = t * num_v_heads + head`.
                // a_log + dt_bias are shared per-head weights.
                let alpha_buf = self.pool.handle(*alpha_in);
                let beta_buf = self.pool.handle(*beta_in);
                let g_decay_buf = self.pool.handle(*g_decay_out);
                let beta_gate_buf = self.pool.handle(*beta_gate_out);
                let nvh = *num_v_heads;
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(
                    &self.linear_attn_pipes.compute_decay_beta,
                );
                enc.set_buffer(0, Some(alpha_buf), 0);
                enc.set_buffer(1, Some(beta_buf), 0);
                enc.set_buffer(
                    2,
                    Some(self.wf_buf.buffer()),
                    *a_log_off as NSUInteger,
                );
                enc.set_buffer(
                    3,
                    Some(self.wf_buf.buffer()),
                    *dt_bias_off as NSUInteger,
                );
                enc.set_buffer(4, Some(g_decay_buf), 0);
                enc.set_buffer(5, Some(beta_gate_buf), 0);
                enc.set_bytes(6, 4, (&nvh as *const u32).cast());
                enc.dispatch_thread_groups(
                    MTLSize::new(*n_tokens as NSUInteger, 1, 1),
                    MTLSize::new(nvh as NSUInteger, 1, 1),
                );
                enc.end_encoding();
            }
            Op::GatedDeltaNetStepNTokens {
                state,
                conv_out,
                g_decay,
                beta_gate,
                output,
                num_v_heads,
                value_dim,
                k_heads_per_v,
                n_tokens,
                ..
            } => {
                // Single batched dispatch: `num_v_heads` threadgroups ×
                // `value_dim` threads. The recurrence is sequential
                // over time but parallel over (head, vi); the kernel
                // runs the `for t` loop internally over its private
                // state row. `conv_out` is [n_tokens * (2*key_total +
                // num_v_heads*value_dim)] — q | k | v per token; the
                // kernel computes the per-token offsets. State is
                // persistent and mutated in-place.
                let nvh = *num_v_heads;
                let vd = *value_dim;
                let kpv = *k_heads_per_v;
                let key_total =
                    crate::riir::variants::VARIANT.linear_total_key() as u32;
                let n_tokens_arg = *n_tokens;
                let state_buf = self.pool.handle(*state);
                let conv_buf = self.pool.handle(*conv_out);
                let g_buf = self.pool.handle(*g_decay);
                let bg_buf = self.pool.handle(*beta_gate);
                let out_buf = self.pool.handle(*output);
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(
                    &self.linear_attn_pipes.delta_net_step,
                );
                enc.set_buffer(0, Some(state_buf), 0);
                enc.set_buffer(1, Some(conv_buf), 0);
                enc.set_buffer(2, Some(g_buf), 0);
                enc.set_buffer(3, Some(bg_buf), 0);
                enc.set_buffer(4, Some(out_buf), 0);
                enc.set_bytes(5, 4, (&kpv as *const u32).cast());
                enc.set_bytes(6, 4, (&n_tokens_arg as *const u32).cast());
                enc.set_bytes(7, 4, (&key_total as *const u32).cast());
                enc.set_bytes(8, 4, (&nvh as *const u32).cast());
                enc.dispatch_thread_groups(
                    MTLSize::new(nvh as NSUInteger, 1, 1),
                    MTLSize::new(vd as NSUInteger, 1, 1),
                );
                enc.end_encoding();
            }
            Op::GatedRmsNormNTokens {
                values,
                z,
                weight_off,
                output,
                num_v_heads,
                value_dim,
                n_tokens,
                eps,
                ..
            } => {
                // Single batched dispatch: `(num_v_heads, n_tokens)`
                // threadgroups × `value_dim` threads. values / z /
                // output are token-major `[n_tokens * num_v_heads *
                // value_dim]`; the kernel addresses each per-head slot
                // via `(t * num_v_heads + head) * value_dim`. Weight is
                // shared across heads and tokens (value_dim bf16).
                let values_buf = self.pool.handle(*values);
                let z_buf = self.pool.handle(*z);
                let output_buf = self.pool.handle(*output);
                let value_dim_arg = *value_dim;
                let eps_arg = *eps;
                let nvh = *num_v_heads;
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(
                    &self.linear_attn_pipes.gated_rms_norm,
                );
                enc.set_buffer(0, Some(values_buf), 0);
                enc.set_buffer(1, Some(z_buf), 0);
                enc.set_buffer(
                    2,
                    Some(self.wf_buf.buffer()),
                    *weight_off as NSUInteger,
                );
                enc.set_buffer(3, Some(output_buf), 0);
                enc.set_bytes(4, 4, (&value_dim_arg as *const u32).cast());
                enc.set_bytes(5, 4, (&eps_arg as *const f32).cast());
                enc.set_bytes(6, 4, (&nvh as *const u32).cast());
                enc.dispatch_thread_groups(
                    MTLSize::new(
                        nvh as NSUInteger,
                        *n_tokens as NSUInteger,
                        1,
                    ),
                    MTLSize::new(value_dim_arg as NSUInteger, 1, 1),
                );
                enc.end_encoding();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests — S10b-pre-1 pool primitives
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Device {
        Device::system_default().expect("no Metal device available")
    }

    #[test]
    #[ignore = "needs Metal device"]
    fn alloc_aligned_returns_aligned_pointer() {
        let mut pool = MetalBufferPool::new(dev());
        const TWO_MIB: usize = 2 * 1024 * 1024;
        // Use a size that wouldn't naturally land on a 2 MiB boundary
        // (Apple's allocator only does that incidentally for large
        // allocations).
        let id = pool.alloc_aligned(64 * 1024, TWO_MIB, "test.aligned", true);
        let buf = pool.handle(id);
        let addr = buf.contents() as usize;
        assert_eq!(
            addr % TWO_MIB,
            0,
            "alloc_aligned returned 0x{addr:x}, not 2 MiB-aligned",
        );
    }

    #[test]
    #[ignore = "needs Metal device"]
    fn register_borrowed_round_trip_via_handle() {
        let device = dev();
        let raw = device.new_buffer(
            128,
            MTLResourceOptions::StorageModeShared,
        );
        let raw_ptr_before = raw.contents() as usize;
        let mut pool = MetalBufferPool::new(device);
        let id = pool.register_borrowed(raw, 128, "test.borrowed", true);
        let pooled = pool.handle(id);
        // The buffer the pool returns should point to the same memory
        // as the buffer we registered (refcounted clone, same backing).
        assert_eq!(pooled.contents() as usize, raw_ptr_before);
    }

    #[test]
    #[ignore = "needs Metal device"]
    fn as_mut_slice_u8_writes_visible_through_handle() {
        let mut pool = MetalBufferPool::new(dev());
        let id = pool
            .alloc(64, "test.scratch", true)
            .expect("alloc");
        {
            let slice = pool.as_mut_slice_u8(id);
            assert_eq!(slice.len(), 64);
            for (i, b) in slice.iter_mut().enumerate() {
                *b = (i as u8).wrapping_mul(7);
            }
        }
        // Read back through the regular handle path.
        let buf = pool.handle(id);
        let read = unsafe {
            std::slice::from_raw_parts(buf.contents() as *const u8, 64)
        };
        for (i, &b) in read.iter().enumerate() {
            assert_eq!(b, (i as u8).wrapping_mul(7));
        }
    }

    #[test]
    #[ignore = "needs Metal device"]
    fn as_mut_slices_u8_disjoint_writes_dont_clobber() {
        let mut pool = MetalBufferPool::new(dev());
        let a = pool.alloc(32, "a", true).expect("alloc");
        let b = pool.alloc(32, "b", true).expect("alloc");
        let c = pool.alloc(32, "c", true).expect("alloc");
        {
            let [sa, sb, sc] = pool.as_mut_slices_u8([a, b, c]);
            sa.fill(0xAA);
            sb.fill(0xBB);
            sc.fill(0xCC);
        }
        for (id, want) in [(a, 0xAAu8), (b, 0xBBu8), (c, 0xCCu8)] {
            let buf = pool.handle(id);
            let read = unsafe {
                std::slice::from_raw_parts(buf.contents() as *const u8, 32)
            };
            assert!(
                read.iter().all(|&v| v == want),
                "slot for {id:?} should be filled with 0x{want:x}"
            );
        }
    }
}
