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

use super::{Backend, BufId, BufferPool, Graph, GraphError, Op};

use metal::{
    Buffer, CommandBuffer, CommandBufferRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize, NSUInteger,
};

use crate::riir::gpu_attn::BatchedSdpaPipelines;
use crate::riir::gpu_linear_attn::LinearAttnPipelines;
use crate::riir::gpu_matvec::{
    encode_matvec_n_tokens, BfMatvecPipelines, MatvecPipelines,
};
use crate::riir::gpu_moe_router::MoeRouterPipelines;
use crate::riir::gpu_norm::{
    encode_residual_add_n_tokens_into, encode_rms_norm_bf16_fused_n_tokens,
    RmsNormBf16FusedNTokensPipeline, RmsNormBf16Pipelines,
};
use crate::riir::metal::{MetalContext, MetalError};
use crate::riir::mtl_weight_buf::MtlWeightBuf;

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
        }
    }

    pub fn physical_buffer_count(&self) -> usize {
        self.buffers.len()
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
        if host.len() != expected {
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
                expected,
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
        if host.len() != expected {
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
                expected,
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
        let mut new_buffers: Vec<Buffer> = Vec::new();
        let mut new_bufid_to_physical: Vec<u32> = vec![u32::MAX; n_bufids];

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
            let old_buf =
                std::mem::replace(&mut self.buffers[old_physical], placeholder.clone());
            new_bufid_to_physical[bufid_idx] = new_buffers.len() as u32;
            new_buffers.push(old_buf);
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
    sdpa_pipes: BatchedSdpaPipelines,
    #[allow(dead_code)]
    linear_attn_pipes: LinearAttnPipelines,
    residual_add_n_pso: ComputePipelineState,
    swiglu_fused_batched_pso: ComputePipelineState,
    swiglu_fused_pso: ComputePipelineState,
    moe_combine_residual_n_pso: ComputePipelineState,
    moe_bucket_accumulate_pso: ComputePipelineState,
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
        let sdpa_pipes = BatchedSdpaPipelines::fetch(&mut metal)?;
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
    ) -> Result<(), GraphError> {
        ctx.cmdbuf.commit();
        ctx.cmdbuf.wait_until_completed();
        Ok(())
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
                crate::riir::expert_forward::encode_moe_combine_residual_n_tokens(
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
                n_tokens,
                ..
            } => {
                // In-place per-head RMS-norm on q and k regions of `x`.
                // Layout: each token spans `key_offset_per_token +
                // num_k_heads*key_dim` floats — q region at offset 0,
                // k region at offset `key_offset_per_token`. Matches
                // `rms_norm_qk_n_tokens_cpu`.
                let inv_scale = 1.0f32 / (*key_dim as f32).sqrt();
                let per_token_elems =
                    *key_offset_per_token + *num_k_heads * *key_dim;
                let per_token_bytes = (per_token_elems as u64) * 4;
                let k_byte_off = (*key_offset_per_token as u64) * 4;
                let x_buf = self.pool.handle(*x);
                let key_dim_arg = *key_dim;
                for t in 0..*n_tokens {
                    let token_base = (t as u64) * per_token_bytes;
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(
                        &self.linear_attn_pipes.rms_norm_qk,
                    );
                    enc.set_buffer(0, Some(x_buf), token_base as NSUInteger);
                    enc.set_buffer(
                        1,
                        Some(x_buf),
                        (token_base + k_byte_off) as NSUInteger,
                    );
                    enc.set_bytes(2, 4, (&key_dim_arg as *const u32).cast());
                    enc.set_bytes(3, 4, (&inv_scale as *const f32).cast());
                    enc.dispatch_thread_groups(
                        MTLSize::new(*num_k_heads as NSUInteger, 1, 1),
                        MTLSize::new(*key_dim as NSUInteger, 1, 1),
                    );
                    enc.end_encoding();
                }
            }
            Op::SdpaCausalTiled { .. } => {
                todo!(
                    "SdpaCausalTiled encode_op: needs kv_dim arg disambiguation; defer to S7-7 producer wire-up"
                )
            }
            Op::MoeBatchedPermuteFuse {
                expert_refs,
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
                // Resolve per-bucket expert blob BufIds to (Buffer, offset)
                // pairs in the format the existing encoder consumes. The
                // `expert_refs` vec is parallel to `buckets.expert_ids`.
                let blobs: Vec<&Buffer> = expert_refs
                    .iter()
                    .map(|(id, _)| self.pool.handle(*id))
                    .collect();
                let resolved: Vec<crate::riir::expert_forward::ExpertRef<'_>> =
                    blobs
                        .iter()
                        .zip(expert_refs.iter())
                        .map(|(buf, (_, off))| (*buf, *off))
                        .collect();
                crate::riir::expert_forward::encode_moe_batched_permute_fuse(
                    cmd,
                    &self.matvec_pipes,
                    &self.swiglu_fused_pso,
                    &self.moe_bucket_accumulate_pso,
                    &resolved,
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
                // Per-token loop. Each token reads `conv_dim` floats
                // from `qkv_in` and writes `conv_dim` floats to
                // `conv_out`. The Metal kernel updates `conv_state`
                // in-place (history shift) at the end of each call,
                // so consecutive token dispatches see the post-shifted
                // state.
                let per_token_bytes = (*conv_dim as u64) * 4;
                let qkv_buf = self.pool.handle(*qkv_in);
                let state_buf = self.pool.handle(*conv_state);
                let conv_out_buf = self.pool.handle(*conv_out);
                let conv_dim_arg = *conv_dim;
                let num_tgs = (conv_dim_arg + 255) / 256;
                for t in 0..*n_tokens {
                    let off = (t as u64) * per_token_bytes;
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(
                        &self.linear_attn_pipes.conv1d_step,
                    );
                    enc.set_buffer(0, Some(state_buf), 0);
                    enc.set_buffer(1, Some(qkv_buf), off as NSUInteger);
                    enc.set_buffer(
                        2,
                        Some(self.wf_buf.buffer()),
                        *weight_off as NSUInteger,
                    );
                    enc.set_buffer(3, Some(conv_out_buf), off as NSUInteger);
                    enc.set_bytes(4, 4, (&conv_dim_arg as *const u32).cast());
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_tgs as NSUInteger, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    enc.end_encoding();
                }
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
                // Per-token loop: each token's alpha / beta are
                // `num_v_heads` floats, tightly packed token-major.
                // Outputs `g_decay` and `beta_gate` follow the same
                // layout. a_log + dt_bias are shared per-head weights.
                let per_token_bytes = (*num_v_heads as u64) * 4;
                let alpha_buf = self.pool.handle(*alpha_in);
                let beta_buf = self.pool.handle(*beta_in);
                let g_decay_buf = self.pool.handle(*g_decay_out);
                let beta_gate_buf = self.pool.handle(*beta_gate_out);
                let nvh = *num_v_heads;
                for t in 0..*n_tokens {
                    let off = (t as u64) * per_token_bytes;
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(
                        &self.linear_attn_pipes.compute_decay_beta,
                    );
                    enc.set_buffer(0, Some(alpha_buf), off as NSUInteger);
                    enc.set_buffer(1, Some(beta_buf), off as NSUInteger);
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
                    enc.set_buffer(4, Some(g_decay_buf), off as NSUInteger);
                    enc.set_buffer(
                        5,
                        Some(beta_gate_buf),
                        off as NSUInteger,
                    );
                    enc.dispatch_thread_groups(
                        MTLSize::new(1, 1, 1),
                        MTLSize::new(nvh as NSUInteger, 1, 1),
                    );
                    enc.end_encoding();
                }
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
                // Per-token loop. State is persistent and mutated
                // in-place by the kernel. conv_out is laid out as
                // [q | k | v] tightly packed per token (3 *
                // VARIANT.linear_total_key() floats per token); the
                // kernel reads q/k/v from byte offsets within the
                // current token's slot. g_decay / beta_gate / output
                // stride by `num_v_heads` (or `num_v_heads *
                // value_dim`) per token.
                let nvh = *num_v_heads;
                let vd = *value_dim;
                let kpv = *k_heads_per_v;
                let key_total =
                    crate::riir::variants::VARIANT.linear_total_key();
                let key_total_bytes = (key_total * 4) as u64;
                let value_total_bytes = (nvh as u64) * (vd as u64) * 4;
                // Per-token conv layout = q (key_total) | k (key_total)
                // | v (num_v_heads * value_dim); v differs from k in
                // size because v uses `value_dim` channels per v-head.
                let conv_per_token =
                    2 * key_total_bytes + value_total_bytes;
                let gdb_per_token = (nvh as u64) * 4;
                let out_per_token = value_total_bytes;
                let state_buf = self.pool.handle(*state);
                let conv_buf = self.pool.handle(*conv_out);
                let g_buf = self.pool.handle(*g_decay);
                let bg_buf = self.pool.handle(*beta_gate);
                let out_buf = self.pool.handle(*output);
                for t in 0..*n_tokens {
                    let conv_off = (t as u64) * conv_per_token;
                    let gdb_off = (t as u64) * gdb_per_token;
                    let out_off = (t as u64) * out_per_token;
                    let q_off = conv_off;
                    let k_off = conv_off + key_total_bytes;
                    let v_off = conv_off + 2 * key_total_bytes;
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(
                        &self.linear_attn_pipes.delta_net_step,
                    );
                    enc.set_buffer(0, Some(state_buf), 0);
                    enc.set_buffer(1, Some(conv_buf), q_off as NSUInteger);
                    enc.set_buffer(2, Some(conv_buf), k_off as NSUInteger);
                    enc.set_buffer(3, Some(conv_buf), v_off as NSUInteger);
                    enc.set_buffer(4, Some(g_buf), gdb_off as NSUInteger);
                    enc.set_buffer(5, Some(bg_buf), gdb_off as NSUInteger);
                    enc.set_buffer(6, Some(out_buf), out_off as NSUInteger);
                    enc.set_bytes(7, 4, (&kpv as *const u32).cast());
                    enc.dispatch_thread_groups(
                        MTLSize::new(nvh as NSUInteger, 1, 1),
                        MTLSize::new(vd as NSUInteger, 1, 1),
                    );
                    enc.end_encoding();
                }
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
                // Per-token loop. Each token reads one per-head slot
                // from `values` and `z` (tightly packed token-major)
                // and writes to `output` at the matching slot.
                // Weight is shared across tokens (value_dim bf16).
                let per_token_bytes =
                    (*num_v_heads * *value_dim) as u64 * 4;
                let values_buf = self.pool.handle(*values);
                let z_buf = self.pool.handle(*z);
                let output_buf = self.pool.handle(*output);
                let value_dim_arg = *value_dim;
                let eps_arg = *eps;
                for t in 0..*n_tokens {
                    let off = (t as u64) * per_token_bytes;
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(
                        &self.linear_attn_pipes.gated_rms_norm,
                    );
                    enc.set_buffer(0, Some(values_buf), off as NSUInteger);
                    enc.set_buffer(1, Some(z_buf), off as NSUInteger);
                    enc.set_buffer(
                        2,
                        Some(self.wf_buf.buffer()),
                        *weight_off as NSUInteger,
                    );
                    enc.set_buffer(3, Some(output_buf), off as NSUInteger);
                    enc.set_bytes(4, 4, (&value_dim_arg as *const u32).cast());
                    enc.set_bytes(5, 4, (&eps_arg as *const f32).cast());
                    enc.dispatch_thread_groups(
                        MTLSize::new(*num_v_heads as NSUInteger, 1, 1),
                        MTLSize::new(*value_dim as NSUInteger, 1, 1),
                    );
                    enc.end_encoding();
                }
            }
        }
    }
}
