//! End-to-end full-attention layer forward — Phase 4d.
//!
//! Companion to [`crate::riir::attn::linear_attn_forward::linear_attn_layer_forward`]
//! for the layers that satisfy `(layer_idx + 1) % full_attn_interval
//! == 0` (every 4th layer in the qwen3_5_moe family). Same shape as
//! the linear-attn forward — input rms_norm + projection batch + per-
//! head ops + hand-off to the shared post-attention tail — but with
//! the attention pipeline swapped:
//!
//! - **Linear-attn (4c)**: 4 batched projections (qkv/z/beta/alpha)
//!   → 5 GPU fused kernels (conv1d / qk_norm / decay_beta /
//!   delta_net / gated_norm) → o_proj.
//! - **Full-attn (4d, this module)**: 3 batched projections (q/k/v)
//!   → CPU per-head Q/K rms_norm → CPU RoPE → KV append (host) →
//!   CPU SDPA → o_proj.
//!
//! Everything from o_proj onward is identical between the two paths
//! and lives in [`crate::riir::attn::linear_attn_forward::post_attention_tail`].
//!
//! Mirrors the `is_full` branches of `fused_layer_forward`
//! (infer.m:4283–5777), excluding the GPU-attn fast path
//! (`gpu_attn_fuse`, gated on `kv->len >= 32`) and the deferred-
//! experts state machine. Both are out of scope for the dump-hook
//! diff and are queued for later slices.
//!
//! ## Tolerance contract
//!
//! Predicted compounded drift, per the strategy doc's per-stage
//! findings:
//!
//! - input rms_norm (CPU, slice 2): bit-exact
//! - 3 GPU projection matvecs (slices 9a/9b finding): bit-exact
//!   per-PSO
//! - per-head Q/K rms_norm (slice 4): bit-exact
//! - RoPE (slice 3): ≤ 4 ULP per channel
//! - KV append (memcpy): bit-exact
//! - SDPA (slice 5): cosine ≥ 0.9999, max_abs_diff ≤ 1.5e-8 at
//!   small kv_len
//! - post-attention tail: bit-exact / ULP-bounded throughout
//!
//! Test floor: cosine ≥ 0.9999, `max_abs_diff / max_abs_out` ≤ 1e-3.
//! Predicted observed: well under those — comparable to 4c's
//! cosine ≈ 1.0, max_abs_diff ≈ 4.1e-8.

use metal::NSUInteger;

use crate::riir::attn::linear_attn_forward::{
    GpuAttnEncodeArgs, LayerForwardError, MoeGraphScratch, OProj, PostAttnIntermediates, bits_of,
    full_attn_layer_idx_for, moe_block_forward, moe_dispatch_per_token, post_attention_pre_moe,
    read_buffer_to_vec,
};
use crate::riir::attn::rms_norm::rms_norm_per_head_cpu;
use crate::riir::attn::rope::apply_rotary_emb;
use crate::riir::attn::sdpa::sdpa_cpu;
use crate::riir::backend::buftype::{
    AttnInputBuf, AttnOutBuf, HiddenBuf, KProjOutBuf, OProjOutBuf, QBuf, QGateBuf, QProjOutBuf,
    RopeInvFreqBuf, RouterLogitsBuf, VProjOutBuf,
};
use crate::riir::backend::gpu::gpu_ctx::GpuLayerCtx;
use crate::riir::backend::gpu::gpu_matvec::{MatvecPipelines, MatvecSpec, encode_matvec};
use crate::riir::backend::gpu::gpu_norm::{RmsNormBf16Pipelines, encode_rms_norm_bf16_into};
use crate::riir::backend::gpu::metal::MetalContext;
use crate::riir::backend::{Backend, BufId, BufferPool, MetalBufferPool};
use crate::riir::io::expert_io::ExpertFiles;
use crate::riir::io::layer_weight_cache::LayerWeightCache;
use crate::riir::io::weight_file::WeightFile;
use crate::riir::moe::expert_forward::MoeBuffers;
use crate::riir::snapshot::state::KvCache;
use crate::riir::variants::VARIANT;

/// Run-lifetime scratch for the batched full-attn `graph1`.
///
/// The full-attn counterpart of
/// [`crate::riir::attn::linear_attn_forward::LinearAttnGraphScratch`]:
/// holds only the full-attn-specific intra-`graph1` transients plus the
/// precomputed RoPE frequency table. Every buffer allocated once at
/// `BATCHED_CHUNK_SIZE` width; the `*NTokens` Ops stride by the real
/// `n_tokens` and the max-chunk tail is simply unused.
///
/// - **Intra-graph1 transients** (`normed` … `gate_logits`):
///   `persistent = false`. The first producer call `commit_plan`s the
///   graph, which lifetime-colors them down to a small physical set.
/// - `inv_freq` is the precomputed vanilla-RoPE frequency table — a
///   read-only graph input, `persistent = true`, uploaded once at
///   construction.
///
/// The graph1→MoE boundary buffers live in the shared
/// [`MoeGraphScratch`](crate::riir::attn::linear_attn_forward::MoeGraphScratch);
/// the hidden double-buffer in
/// [`HiddenDoubleBuffer`](crate::riir::attn::linear_attn_forward::HiddenDoubleBuffer).
/// Both are orchestrator-owned and passed in per layer.
pub struct FullAttnGraphScratch {
    // Intra-graph1 transients (colorable).
    /// Pre-attention RMS-norm output — feeds Q/K/V proj matvecs.
    /// Allocated as the canonical `AttnInputBuf`; the producer's
    /// `Op::RmsNormBf16NTokens` push converts to the `RmsNormOut`
    /// union via the existing `From<AttnInputBuf> for BufId<RmsNormOut>`
    /// impl, and the projection matvecs accept it as input via
    /// `From<AttnInputBuf> for BufId<MatvecIn>`.
    pub normed: BufId<AttnInputBuf>,
    pub q_proj_stack: BufId<QProjOutBuf>,
    pub k_proj_stack: BufId<KProjOutBuf>,
    pub v_proj_stack: BufId<VProjOutBuf>,
    pub q_stack: BufId<QBuf>,
    pub q_gate_stack: BufId<QGateBuf>,
    pub attn_out_stack: BufId<AttnOutBuf>,
    pub o_proj_stack: BufId<OProjOutBuf>,
    pub gate_logits: BufId<RouterLogitsBuf>,
    /// Precomputed vanilla-RoPE `inv_freq` table — persistent,
    /// read-only, uploaded once at construction.
    pub inv_freq: BufId<RopeInvFreqBuf>,
    /// One-time `commit_plan` latch for `graph1` — cleared at
    /// construction, set by the first producer call. The MoE `graph2`
    /// has its own latch on `MoeGraphScratch`.
    pub commit_planned: std::cell::Cell<bool>,
}

impl FullAttnGraphScratch {
    /// Allocate the `graph1` transients at max chunk width and upload
    /// the RoPE frequency table. The transients are `persistent =
    /// false` (the first producer call `commit_plan`s + pins them);
    /// `inv_freq` is `persistent = true`.
    pub fn new(pool: &mut MetalBufferPool) -> Self {
        let v = VARIANT;
        let chunk = crate::riir::BATCHED_CHUNK_SIZE;
        let f32_sz = std::mem::size_of::<f32>();
        let hidden = v.hidden_dim;
        let q_dim = v.num_attn_heads * v.head_dim;
        let kv_dim = v.num_kv_heads * v.head_dim;

        // Colorable transients first (`persistent = false`). Each
        // `pool.alloc::<Tag>` returns a `BufId<Tag>` directly — the
        // tag at the call site is the only place that names the role.
        let bytes_of = |elems: usize| chunk * elems * f32_sz;
        let normed: BufId<AttnInputBuf> = pool
            .alloc(bytes_of(hidden), "fags.normed", false)
            .expect("FullAttnGraphScratch::new pool alloc");
        let q_proj_stack: BufId<QProjOutBuf> = pool
            .alloc(bytes_of(q_dim * 2), "fags.q_proj_stack", false)
            .expect("FullAttnGraphScratch::new pool alloc");
        let k_proj_stack: BufId<KProjOutBuf> = pool
            .alloc(bytes_of(kv_dim), "fags.k_proj_stack", false)
            .expect("FullAttnGraphScratch::new pool alloc");
        let v_proj_stack: BufId<VProjOutBuf> = pool
            .alloc(bytes_of(kv_dim), "fags.v_proj_stack", false)
            .expect("FullAttnGraphScratch::new pool alloc");
        let q_stack: BufId<QBuf> = pool
            .alloc(bytes_of(q_dim), "fags.q_stack", false)
            .expect("FullAttnGraphScratch::new pool alloc");
        let q_gate_stack: BufId<QGateBuf> = pool
            .alloc(bytes_of(q_dim), "fags.q_gate_stack", false)
            .expect("FullAttnGraphScratch::new pool alloc");
        let attn_out_stack: BufId<AttnOutBuf> = pool
            .alloc(bytes_of(q_dim), "fags.attn_out_stack", false)
            .expect("FullAttnGraphScratch::new pool alloc");
        let o_proj_stack: BufId<OProjOutBuf> = pool
            .alloc(bytes_of(hidden), "fags.o_proj_stack", false)
            .expect("FullAttnGraphScratch::new pool alloc");
        let gate_logits: BufId<RouterLogitsBuf> = pool
            .alloc(bytes_of(v.num_experts), "fags.gate_logits", false)
            .expect("FullAttnGraphScratch::new pool alloc");

        // RoPE frequency table: inv_freq[i] = 1 / theta^(2i/rotary_dim).
        let rotary_dim = v.rotary_dim();
        let half = rotary_dim / 2;
        let theta = crate::riir::variants::ROPE_THETA;
        let inv_freq_host: Vec<f32> = (0..half)
            .map(|i| 1.0f32 / theta.powf((2 * i) as f32 / rotary_dim as f32))
            .collect();

        // `inv_freq` is the sole persistent buffer — allocated last so
        // the highest BufId index is `persistent = true`, keeping the
        // transients above safe under `reset_transient`.
        let inv_freq: BufId<RopeInvFreqBuf> = pool
            .alloc(half * f32_sz, "fags.inv_freq", true)
            .expect("FullAttnGraphScratch::new pool alloc");
        pool.upload(inv_freq, unsafe {
            std::slice::from_raw_parts(inv_freq_host.as_ptr() as *const u8, half * f32_sz)
        })
        .expect("FullAttnGraphScratch::new inv_freq upload");

        Self {
            normed,
            q_proj_stack,
            k_proj_stack,
            v_proj_stack,
            q_stack,
            q_gate_stack,
            attn_out_stack,
            o_proj_stack,
            gate_logits,
            inv_freq,
            commit_planned: std::cell::Cell::new(false),
        }
    }
}

/// Run one full-attention layer's forward pass — the pre-MoE half.
///
/// Returns the [`PostAttnIntermediates`] needed by either
/// [`moe_dispatch_per_token`] (per-token path; called by the
/// [`full_attn_layer_forward`] wrapper) or by the batched-prefill
/// orchestrator (Phase B+) which collects intermediates across all
/// tokens in a chunk before dispatching MoE in batch.
///
/// Pre: `buffers.input` holds the input hidden state (HIDDEN_DIM
/// floats). Post: `buffers.h_mid`, `buffers.normed` (= h_post), and
/// `buffers.shared_out` hold the per-token GPU outputs; KV cache
/// advances by 1.
///
/// `pos` is the absolute KV position (matches the C side's `pos`
/// argument to `apply_rotary_emb`).
#[allow(clippy::too_many_arguments)]
pub(in crate::riir) fn full_attn_pre_moe_layer_forward(
    metal: &mut MetalContext,
    gpu: &GpuLayerCtx<'_>,
    layer_idx: usize,
    pos: i32,
    k_active: usize,
    kv_state: &mut KvCache,
    // Slice 5d-8: see `linear_attn_layer_forward` for the contract.
    prev_layer_chained: bool,
) -> Result<PostAttnIntermediates, LayerForwardError> {
    let GpuLayerCtx {
        wf,
        wf_buf,
        layer_cache,
        buffers,
        buffer_pool,
        ..
    } = *gpu;
    let v = VARIANT;

    // Phase 0b (prefill arc): the KV cache is GPU-resident, registered
    // into the pool by `ensure_linear_resources` before any forward
    // runs — no per-layer alloc here.

    // Reject linear-attn layers up front. Mirror the symmetric guard
    // in `linear_attn_layer_forward`.
    if v.layer_kind(layer_idx) != crate::riir::variants::LayerKind::FullAttn {
        return Err(LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "full_attn_layer_forward called on linear-attn layer",
        });
    }

    // Per-tensor bit width lookup for the projection matvecs.
    let q_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
    );
    let k_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
    );
    let v_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
    );
    let o_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
    );

    // Pull the full-attn-specific offsets out of the tagged-enum
    // cache. Every slot is required at `LayerWeightCache::build` time
    // for full-attn layers, so this is a single match instead of a
    // require-ladder.
    let attn = layer_cache
        .attn
        .full()
        .ok_or(LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "full_attn weights (called on linear-attn layer)",
        })?;
    let q_w = attn.q_proj_w;
    let q_s = attn.q_proj_s;
    let q_b = attn.q_proj_b;
    let k_w = attn.k_proj_w;
    let k_s = attn.k_proj_s;
    let k_b = attn.k_proj_b;
    let v_w = attn.v_proj_w;
    let v_s = attn.v_proj_s;
    let v_b = attn.v_proj_b;
    let o_w = attn.o_proj_w;
    let o_s = attn.o_proj_s;
    let o_b = attn.o_proj_b;
    // q_norm / k_norm are loaded by name inside `rms_norm_per_head_cpu`;
    // they're guaranteed present here because `LayerWeightCache::build`
    // populated `attn.q_norm_w` / `attn.k_norm_w` as required slots.

    let q_dim = v.num_attn_heads * v.head_dim; // total Q channels
    let q_proj_dim = q_dim * 2; // q + per-head sigmoid gate
    let kv_dim = v.num_kv_heads * v.head_dim;

    // Pre-fetch the matvec pipelines.
    let mv = MatvecPipelines::fetch(metal)?;
    let rms_pipes = RmsNormBf16Pipelines::fetch(metal)?;

    // ── CMD1: input rms_norm + 3 batched projection matvecs ──────
    //
    // Slice 5d-2: input rms_norm runs on the GPU as the prelude to
    // CMD1. Same shape as the linear-attn forward; see that module
    // for the rationale + bit-exactness against the C fast-path
    // chain. `buffers.input` is the residual source consumed later
    // by `post_attention_tail`'s `encode_residual_add`; it's not
    // mutated within the layer's forward, so dual-use is safe.
    {
        let cmdbuf = metal.queue().new_command_buffer();

        // Slice 5d-8: skip the input-norm prelude when the previous
        // layer chained — `buffers.normed` is already populated.
        if !prev_layer_chained {
            encode_rms_norm_bf16_into(
                cmdbuf,
                &rms_pipes,
                buffer_pool.handle(buffers.input),
                wf_buf.buffer(),
                layer_cache.input_layernorm_w,
                buffer_pool.handle(buffers.sum_sq),
                buffer_pool.handle(buffers.normed),
                v.hidden_dim as u32,
                crate::riir::variants::RMS_NORM_EPS,
            );
        }

        let specs = [
            MatvecSpec {
                w_off: q_w,
                s_off: q_s,
                b_off: q_b,
                input: buffer_pool.handle(buffers.normed),
                output: buffer_pool.handle(buffers.q_proj_out),
                out_dim: q_proj_dim as u32,
                in_dim: v.hidden_dim as u32,
                bits: q_bits,
            },
            MatvecSpec {
                w_off: k_w,
                s_off: k_s,
                b_off: k_b,
                input: buffer_pool.handle(buffers.normed),
                output: buffer_pool.handle(buffers.k_out),
                out_dim: kv_dim as u32,
                in_dim: v.hidden_dim as u32,
                bits: k_bits,
            },
            MatvecSpec {
                w_off: v_w,
                s_off: v_s,
                b_off: v_b,
                input: buffer_pool.handle(buffers.normed),
                output: buffer_pool.handle(buffers.v_out),
                out_dim: kv_dim as u32,
                in_dim: v.hidden_dim as u32,
                bits: v_bits,
            },
        ];
        for s in &specs {
            encode_matvec(cmdbuf, &mv, wf_buf, s);
        }

        metal.commit_and_wait_labeled(cmdbuf, "full_attn.cmd1");
    }

    // ── Read q_proj_out, k, v back to host ───────────────────────
    let q_proj_host = read_buffer_to_vec(buffer_pool.handle(buffers.q_proj_out), q_proj_dim);
    let mut k_host = read_buffer_to_vec(buffer_pool.handle(buffers.k_out), kv_dim);
    let v_host = read_buffer_to_vec(buffer_pool.handle(buffers.v_out), kv_dim);

    // ── Per-head split: q_proj_out → q + q_gate ──────────────────
    // q_proj_out layout per head: `[q_h (HEAD_DIM) | gate_h
    // (HEAD_DIM)]`, i.e. contiguous head-by-head as 2*HEAD_DIM stride.
    // Matches the C path at infer.m:4760–4764 (and 2428–2432 in the
    // standalone full_attention_forward).
    let mut q_host = vec![0.0f32; q_dim];
    let mut q_gate_host = vec![0.0f32; q_dim];
    for h in 0..v.num_attn_heads {
        let src_off = h * (2 * v.head_dim);
        let dst_off = h * v.head_dim;
        q_host[dst_off..dst_off + v.head_dim]
            .copy_from_slice(&q_proj_host[src_off..src_off + v.head_dim]);
        q_gate_host[dst_off..dst_off + v.head_dim]
            .copy_from_slice(&q_proj_host[src_off + v.head_dim..src_off + 2 * v.head_dim]);
    }

    // ── Per-head Q rms_norm ──────────────────────────────────────
    let q_norm_name = format!("model.layers.{layer_idx}.self_attn.q_norm.weight");
    rms_norm_per_head_cpu(wf, &q_norm_name, v.num_attn_heads, v.head_dim, &mut q_host)?;

    // ── Per-head K rms_norm ──────────────────────────────────────
    let k_norm_name = format!("model.layers.{layer_idx}.self_attn.k_norm.weight");
    rms_norm_per_head_cpu(wf, &k_norm_name, v.num_kv_heads, v.head_dim, &mut k_host)?;

    // ── RoPE on q + k ────────────────────────────────────────────
    apply_rotary_emb(pos, &mut q_host, &mut k_host)?;

    // ── KV append: canonical pool KV ─────────────────────────────
    let cache_pos = kv_state.len as usize;
    if cache_pos + 1 > crate::riir::variants::MAX_SEQ_LEN {
        return Err(LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "kv cache overflow",
        });
    }
    // SAFETY: shared-storage KV buffers; this CPU append runs before
    // any GPU dispatch reads the cache this layer (CMD1 above
    // committed and waited; CMD2 is built below).
    unsafe {
        kv_state
            .k_slice_mut(buffer_pool, cache_pos, cache_pos + 1)
            .copy_from_slice(&k_host);
        kv_state
            .v_slice_mut(buffer_pool, cache_pos, cache_pos + 1)
            .copy_from_slice(&v_host);
    }
    kv_state.len += 1;

    // The oracle GPU SDPA fast path (below) reads the canonical pool
    // KV directly — it is GPU-resident post-Phase-0b, so the old
    // per-token GPU mirror copy is gone (#2: the mirror was never
    // populated by batched prefill, so decode after a ≥32-token
    // batched eval_prompt attended zeros over the prompt region).
    // `fa_idx` still gates the fast path: only full-attn layers take it.
    let fa_idx = full_attn_layer_idx_for(layer_idx);

    // ── Decide between GPU SDPA fast path and CPU SDPA ──────────
    //
    // Base gate (matches C `infer.m:5054`): full-attn layer, KV fits the
    // persistent buffer, and past the GPU dispatch break-even point
    // (kv_len < 32 keeps command-encoder overhead from dominating).
    //
    // Plus the I/O-mode gate (#2 perf regression): the GPU SDPA fast path
    // binds the canonical KV pool buffers (~2 GB/full-attn-layer). In
    // `Pread` mode — the working set exceeds RAM (a17b: 202 GiB experts /
    // 96 GiB RAM) and experts stream from disk — making those buffers
    // GPU-resident evicts the streamed expert pages and collapses decode
    // (~8×, idle GPU / disk-bound). So route full-attn through CPU SDPA,
    // which reads only the host-resident KV prefix and leaves the GPU
    // free for expert streaming; the full-attn SDPA is a tiny slice of an
    // expert-streaming-bound decode, so the cost is negligible. In `Mmap`
    // mode (a3b — fits RAM, no streaming pressure) keep GPU SDPA: ~17%
    // faster full-attn with no eviction.
    let kv_len = kv_state.len;
    let gpu_attn_ready = fa_idx.is_some()
        && kv_len >= 32
        && (kv_len as usize) < crate::riir::variants::GPU_KV_SEQ
        && !gpu.expert_io_mode.is_pread();

    let gpu_attn_args = if gpu_attn_ready {
        // Stage Q + q_gate (both post-norm + RoPE for q) into the
        // shared GPU scratch buffers. Read by Enc A1 (scores) and
        // Enc A4 (sigmoid gate). SAFETY: shared-storage; no GPU work
        // in flight on these buffers (CMD1 above committed and waited;
        // CMD2 hasn't been built yet).
        unsafe {
            let q_dst = buffer_pool.handle(buffers.gpu_attn_q).contents() as *mut f32;
            let g_dst = buffer_pool.handle(buffers.gpu_attn_gate).contents() as *mut f32;
            std::ptr::copy_nonoverlapping(q_host.as_ptr(), q_dst, q_dim);
            std::ptr::copy_nonoverlapping(q_gate_host.as_ptr(), g_dst, q_dim);
        }
        Some(GpuAttnEncodeArgs {
            k_cache: kv_state.k_id.expect("gpu_attn_ready ⇒ KV K allocated"),
            v_cache: kv_state.v_id.expect("gpu_attn_ready ⇒ KV V allocated"),
            kv_len: kv_len as u32,
        })
    } else {
        // CPU SDPA fallback. Slice the caches to the occupied prefix
        // and stage the result into batch_out[6] for o_proj.
        let mut attn_out = vec![0.0f32; q_dim];
        // SAFETY: shared-storage KV buffers; CPU SDPA fallback runs
        // with no GPU work in flight on the cache (CMD1 committed and
        // waited above; CMD2 not yet built).
        let (k_prefix, v_prefix) = unsafe {
            (
                kv_state.k_slice(buffer_pool, kv_len as usize),
                kv_state.v_slice(buffer_pool, kv_len as usize),
            )
        };
        sdpa_cpu(
            kv_len,
            &q_host,
            &q_gate_host,
            k_prefix,
            v_prefix,
            &mut attn_out,
        )?;

        let dst = buffer_pool.handle(buffers.o_proj_stack).contents() as *mut f32;
        // SAFETY: shared-storage; no GPU work in flight (CMD1
        // committed and waited above).
        unsafe {
            std::ptr::copy_nonoverlapping(attn_out.as_ptr(), dst, q_dim);
        }
        debug_assert!(
            buffer_pool.handle(buffers.o_proj_stack).length() as usize
                >= q_dim * std::mem::size_of::<f32>(),
            "batch_out[6] sized {} bytes, need {} for full-attn o_proj input",
            buffer_pool.handle(buffers.o_proj_stack).length() as NSUInteger,
            q_dim * std::mem::size_of::<f32>(),
        );
        None
    };

    // ── Hand off to the shared pre-MoE tail ──────────────────────
    // When `gpu_attn_args` is `Some`, pre_moe encodes the 4 attn
    // kernels at the head of CMD2 and reads o_proj from
    // `gpu_attn_out`. Otherwise it reads from `batch_out[6]` (the
    // CPU-SDPA staging slot above).
    //
    // Slice cmdbuf-fold-1: full-attn cannot fold CMD1 into CMD2+3
    // because the host-bounce above (q/k/v readback + CPU per-head
    // norm + RoPE + KV append) is interposed. A fresh cmdbuf for
    // the tail is the correct shape until Phase 3b moves the host-
    // bounce work to GPU.
    //
    // `queue_clone` (not `queue`) so the cmdbuf borrow doesn't pin
    // `*metal` and block the `&mut metal` arg below.
    let queue = metal.queue_clone();
    let cmdbuf = queue.new_command_buffer();
    let intermediates = post_attention_pre_moe(
        metal,
        cmdbuf,
        gpu,
        layer_idx,
        k_active,
        OProj {
            w_off: o_w,
            s_off: o_s,
            b_off: o_b,
            bits: o_bits,
            in_dim: q_dim as u32,
        },
        gpu_attn_args,
    )?;
    Ok(intermediates)
}

/// Run one full-attention layer's forward pass — per-token wrapper.
/// Composes [`full_attn_pre_moe_layer_forward`] with
/// [`moe_dispatch_per_token`]. Behaviour mirrors the pre-Phase-B
/// `full_attn_layer_forward`: `buffers.input` in, post-combine
/// hidden state in `buffers.input` after the deferred dispatch
/// completes (drained at the top of the next layer's forward).
#[allow(clippy::too_many_arguments)]
pub(in crate::riir) fn full_attn_layer_forward(
    metal: &mut MetalContext,
    gpu: &GpuLayerCtx<'_>,
    moe: &mut MoeBuffers,
    deferred: &mut crate::riir::moe::deferred::DeferredRing,
    layer_idx: usize,
    pos: i32,
    k_active: usize,
    expert_files: &ExpertFiles,
    pool: &rayon::ThreadPool,
    prefetch: &mut crate::riir::io::prefetch::PrefetchState,
    // Slice 5d-9: which `data_prefetch` set this layer reads from
    // (parity ping-pong: `layer_idx % 2`).
    prefetch_set: usize,
    kv_state: &mut KvCache,
    gpu_combine: bool,
    // Slice 5d-8: see `linear_attn_layer_forward` for the contract.
    prev_layer_chained: bool,
    chain_next_norm_off: Option<u64>,
) -> Result<(), LayerForwardError> {
    let GpuLayerCtx {
        wf: _,
        wf_buf,
        layer_cache: _,
        buffers,
        buffer_pool,
        ..
    } = *gpu;
    let intermediates = full_attn_pre_moe_layer_forward(
        metal,
        gpu,
        layer_idx,
        pos,
        k_active,
        kv_state,
        prev_layer_chained,
    )?;
    moe_dispatch_per_token(
        metal,
        wf_buf,
        buffers,
        buffer_pool,
        moe,
        deferred,
        layer_idx,
        expert_files,
        pool,
        prefetch,
        prefetch_set,
        &intermediates,
        gpu_combine,
        chain_next_norm_off,
    )
}

/// Batched full-attention layer forward for chunked prefill.
///
/// The pre-MoE half — input norm, Q/K/V projections, the per-head
/// Q/gate split, per-head Q/K norm, RoPE, KV-cache append, SDPA, the
/// sigmoid query gate, O-projection, the post-attention residual +
/// norm, and the MoE router — is expressed as one [`Graph`] of typed
/// [`Op`]s and submitted via `backend.execute` in a single cmdbuf
/// (`commit_plan`'d once per run). Structurally identical to
/// [`batched_linear_attn_layer_forward`](crate::riir::attn::linear_attn_forward::batched_linear_attn_layer_forward)'s
/// `graph1`.
///
/// The MoE block (shared FFN + expert permute-fuse + combine) is still
/// imperative — graph-ifying it is the next slice.
///
/// Pre: `hidden_in_id` holds token `t`'s input hidden state at
/// `[t*hidden_dim..]`. Post: `hidden_out_id` holds the post-combine
/// hidden state; `kv_state.len` advances by `n_tokens`.
///
/// **No deferred dispatch.** The batched path reads expert blobs
/// synchronously per bucket; `prefetch` is the decode-only (N=1)
/// state machine, threaded through to the MoE block.
#[allow(clippy::too_many_arguments)]
pub(in crate::riir) fn batched_full_attn_layer_forward<B>(
    backend: &mut B,
    wf: &WeightFile,
    layer_cache: &LayerWeightCache,
    layer_idx: usize,
    start_pos: i32,
    n_tokens: usize,
    k_active: usize,
    expert_files: &ExpertFiles,
    moe_buffers: &mut MoeBuffers,
    kv_state: &mut KvCache,
    // Session-5 Phase 3: prefetch env when caller has fired async
    // prefetch for this layer. See `batched_linear_attn_layer_forward`
    // for the semantic.
    prefetch: Option<crate::riir::attn::linear_attn_forward::PrefetchEnv<'_>>,
    // S7-2: hidden_in / hidden_out are the orchestrator's run-lifetime
    // double-buffer BufIds, swapped between layers.
    hidden_in_id: BufId<HiddenBuf>,
    hidden_out_id: BufId<HiddenBuf>,
    // Run-lifetime scratch, allocated once at max chunk width by
    // `ensure_linear_resources`. `scratch` holds the full-attn
    // `graph1` transients; `moe` is the shared MoE-block scratch
    // (boundary buffers + `graph2` working set).
    scratch: &FullAttnGraphScratch,
    moe: &MoeGraphScratch,
) -> Result<(), LayerForwardError>
where
    B: Backend,
    LayerForwardError: From<B::Error>,
    LayerForwardError: From<<B::Pool as BufferPool>::Error>,
{
    use crate::riir::backend::{Graph, Op, WeightRef};
    use crate::riir::moe::expert_forward::MAX_K;

    let v = VARIANT;
    debug_assert!(k_active <= MAX_K);

    let hidden_dim = v.hidden_dim;
    let q_dim = v.num_attn_heads * v.head_dim;
    let kv_dim = v.num_kv_heads * v.head_dim;
    let q_proj_dim = q_dim * 2;
    let num_attn_heads = v.num_attn_heads as u32;
    let num_kv_heads = v.num_kv_heads as u32;
    let head_dim = v.head_dim as u32;
    let rotary_dim = v.rotary_dim() as u32;
    let eps = crate::riir::variants::RMS_NORM_EPS;

    // Phase 0b (prefill arc): the KV cache is GPU-resident, registered
    // into the pool by `ensure_linear_resources`.
    let k_cache_id = kv_state
        .k_id
        .expect("kv cache registered by ensure_linear_resources");
    let v_cache_id = kv_state
        .v_id
        .expect("kv cache registered by ensure_linear_resources");

    let kv_start = kv_state.len;
    if (kv_start as usize) + n_tokens > crate::riir::variants::MAX_SEQ_LEN {
        return Err(LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "kv cache overflow",
        });
    }

    let attn = layer_cache
        .attn
        .full()
        .ok_or(LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "full_attn weights (batched graph path)",
        })?;

    // Per-tensor bit widths for the projection / router matvecs.
    let q_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
    );
    let k_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
    );
    let v_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
    );
    let o_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
    );
    let gate_bits = bits_of(wf, &format!("model.layers.{layer_idx}.mlp.gate.weight"));
    let seg_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.mlp.shared_expert_gate.weight"),
    );

    let softmax_scale = 1.0f32 / (v.head_dim as f32).sqrt();
    let heads_per_kv = num_attn_heads / num_kv_heads;
    let n = n_tokens as u32;

    // ── graph1: input norm → Q/K/V proj → split → per-head norm →
    //    RoPE → KV append → SDPA → sigmoid gate → o_proj → residual →
    //    post-norm → MoE router. One Graph, one commit_plan, one
    //    cmdbuf — mirrors `batched_linear_attn_layer_forward`. ──────
    let graph = {
        let mut g = Graph::new();
        g.push(Op::RmsNormBf16NTokens {
            label: "full_attn.input_norm",
            x: hidden_in_id.into(),
            weight_off: layer_cache.input_layernorm_w,
            out: scratch.normed.into(),
            dim: hidden_dim as u32,
            n_tokens: n,
            eps,
        });
        g.push(Op::MatvecNTokens {
            label: "full_attn.q_proj",
            weight: WeightRef {
                w_off: attn.q_proj_w,
                s_off: attn.q_proj_s,
                b_off: attn.q_proj_b,
                bits: q_bits,
            },
            input: scratch.normed.into(),
            input_off: 0,
            output: scratch.q_proj_stack.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: q_proj_dim as u32,
            n_tokens: n,
        });
        g.push(Op::MatvecNTokens {
            label: "full_attn.k_proj",
            weight: WeightRef {
                w_off: attn.k_proj_w,
                s_off: attn.k_proj_s,
                b_off: attn.k_proj_b,
                bits: k_bits,
            },
            input: scratch.normed.into(),
            input_off: 0,
            output: scratch.k_proj_stack.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: kv_dim as u32,
            n_tokens: n,
        });
        g.push(Op::MatvecNTokens {
            label: "full_attn.v_proj",
            weight: WeightRef {
                w_off: attn.v_proj_w,
                s_off: attn.v_proj_s,
                b_off: attn.v_proj_b,
                bits: v_bits,
            },
            input: scratch.normed.into(),
            input_off: 0,
            output: scratch.v_proj_stack.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: kv_dim as u32,
            n_tokens: n,
        });
        g.push(Op::SplitQGate {
            label: "full_attn.split_q_gate",
            q_proj: scratch.q_proj_stack,
            q_out: scratch.q_stack,
            gate_out: scratch.q_gate_stack,
            num_heads: num_attn_heads,
            head_dim,
            n_tokens: n,
        });
        g.push(Op::RmsNormPerHeadNTokens {
            label: "full_attn.q_norm",
            // Per-head norm is in-place on the Q stack; QBuf → RmsNormIn
            // via the union's `From<QBuf>` impl.
            x: scratch.q_stack.into(),
            weight_off: attn.q_norm_w,
            num_heads: num_attn_heads,
            head_dim,
            n_tokens: n,
            eps,
        });
        g.push(Op::RmsNormPerHeadNTokens {
            label: "full_attn.k_norm",
            // In-place on the K-projection buffer; KProjOutBuf → RmsNormIn
            // via the union impl.
            x: scratch.k_proj_stack.into(),
            weight_off: attn.k_norm_w,
            num_heads: num_kv_heads,
            head_dim,
            n_tokens: n,
            eps,
        });
        g.push(Op::RopeNTokens {
            label: "full_attn.q_rope",
            x: scratch.q_stack.into(),
            inv_freq: scratch.inv_freq,
            n_tokens: n,
            num_heads: num_attn_heads,
            head_dim,
            rotary_dim,
            start_pos,
        });
        g.push(Op::RopeNTokens {
            label: "full_attn.k_rope",
            x: scratch.k_proj_stack.into(),
            inv_freq: scratch.inv_freq,
            n_tokens: n,
            num_heads: num_kv_heads,
            head_dim,
            rotary_dim,
            start_pos,
        });
        g.push(Op::KvCacheAppendNTokens {
            label: "full_attn.kv_append",
            k_src: scratch.k_proj_stack,
            v_src: scratch.v_proj_stack,
            k_cache: k_cache_id,
            v_cache: v_cache_id,
            kv_dim: kv_dim as u32,
            n_tokens: n,
            kv_start: kv_start as u32,
        });
        g.push(Op::SdpaCausalTiled {
            label: "full_attn.sdpa",
            q: scratch.q_stack,
            k: k_cache_id,
            v: v_cache_id,
            attn_out: scratch.attn_out_stack,
            n_tokens: n,
            num_heads: num_attn_heads,
            heads_per_kv,
            head_dim,
            kv_dim: kv_dim as u32,
            kv_start: kv_start as u32,
            kv_len_total: kv_start as u32 + n,
            softmax_scale,
        });
        g.push(Op::SigmoidGateNTokens {
            label: "full_attn.sigmoid_gate",
            x: scratch.attn_out_stack,
            gate: scratch.q_gate_stack,
            dim: q_dim as u32,
            n_tokens: n,
        });
        g.push(Op::MatvecNTokens {
            label: "full_attn.o_proj",
            weight: WeightRef {
                w_off: attn.o_proj_w,
                s_off: attn.o_proj_s,
                b_off: attn.o_proj_b,
                bits: o_bits,
            },
            input: scratch.attn_out_stack.into(),
            input_off: 0,
            output: scratch.o_proj_stack.into(),
            output_off: 0,
            in_dim: q_dim as u32,
            out_dim: hidden_dim as u32,
            n_tokens: n,
        });
        g.push(Op::ResidualAddNTokens {
            label: "full_attn.residual_add",
            a: scratch.o_proj_stack,
            // The residual `b` slot accepts the hidden double-buffer
            // input via the RmsNormIn union (HiddenBuf → RmsNormIn).
            b: hidden_in_id.into(),
            out: moe.h_mid,
            n_tokens: n,
            dim: hidden_dim as u32,
        });
        g.push(Op::RmsNormBf16NTokens {
            label: "full_attn.post_attn_norm",
            // ResidualBuf → RmsNormIn via the union impl.
            x: moe.h_mid.into(),
            weight_off: layer_cache.post_attention_layernorm_w,
            // MoeGraphScratch::h_post is the post-norm activation
            // (MoeInputBuf); MoeInputBuf → RmsNormOut via the union impl.
            out: moe.h_post.into(),
            dim: hidden_dim as u32,
            n_tokens: n,
            eps,
        });
        g.push(Op::MatvecNTokens {
            label: "full_attn.gate_router",
            weight: WeightRef {
                w_off: layer_cache.gate.w,
                s_off: layer_cache.gate.s,
                b_off: layer_cache.gate.b,
                bits: gate_bits,
            },
            input: moe.h_post.into(),
            input_off: 0,
            output: scratch.gate_logits.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: v.num_experts as u32,
            n_tokens: n,
        });
        g.push(Op::MatvecNTokens {
            label: "full_attn.shared_gate",
            weight: WeightRef {
                w_off: layer_cache.shared.seg_w,
                s_off: layer_cache.shared.seg_s,
                b_off: layer_cache.shared.seg_b,
                bits: seg_bits,
            },
            input: moe.h_post.into(),
            input_off: 0,
            output: moe.shared_gate.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: 1,
            n_tokens: n,
        });
        g.push(Op::MoeSoftmaxTopK {
            label: "full_attn.router_softmax_topk",
            logits: scratch.gate_logits,
            indices_out: moe.routing_indices,
            weights_out: moe.routing_weights,
            n_tokens: n,
            n_experts: v.num_experts as u32,
            k: k_active as u32,
        });
        g.push(Op::MoeNormalizeWeights {
            label: "full_attn.router_normalize",
            weights: moe.routing_weights,
            n_tokens: n,
            k: k_active as u32,
        });
        g
    };

    // First call lifetime-colors graph1's transients (topology is
    // layer- and step-invariant); then latch.
    if !scratch.commit_planned.get() {
        backend.pool_mut().commit_plan(&graph);
        scratch.commit_planned.set(true);
    }
    backend.execute(&graph, "graph_full_attn")?;
    // The `KvCacheAppendNTokens` op extended the cache by `n_tokens`.
    kv_state.len += n_tokens as i32;

    // Prefill-arc Phase 3b: the MoE block — host readback, CPU
    // bucket build, expert staging, graph2 — is the shared
    // `moe_block_forward`, identical to the linear-attn path.
    moe_block_forward(
        backend,
        moe,
        wf,
        layer_cache,
        layer_idx,
        n_tokens,
        k_active,
        expert_files,
        moe_buffers,
        prefetch,
        hidden_out_id,
    )
}
