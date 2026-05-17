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

use crate::riir::moe::expert_forward::MoeBuffers;
use crate::riir::backend::BufferPool;
use crate::riir::io::expert_io::ExpertFiles;
use crate::riir::attn::gpu_attn::{encode_sdpa_causal_flash, FlashSdpaPipelines};
use crate::riir::backend::gpu::gpu_matvec::{encode_matvec, MatvecPipelines, MatvecSpec};
use crate::riir::backend::gpu::gpu_norm::{encode_rms_norm_bf16_into, RmsNormBf16Pipelines};
use crate::riir::backend::gpu::gpu_ctx::GpuLayerCtx;
use crate::riir::backend::gpu::gpu_matvec::encode_dense_matmul_n_tokens;
use crate::riir::attn::linear_attn_forward::{
    bits_of, full_attn_layer_idx_for, moe_dispatch_per_token,
    post_attention_pre_moe, read_buffer_to_vec, GpuAttnEncodeArgs,
    LayerForwardError, OProj, PostAttnIntermediates,
};
use crate::riir::backend::gpu::metal::MetalContext;
use crate::riir::attn::rms_norm::rms_norm_per_head_cpu;
use crate::riir::attn::rope::apply_rotary_emb;
use crate::riir::attn::sdpa::sdpa_cpu;
use crate::riir::snapshot::state::KvCache;
use crate::riir::variants::VARIANT;

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
    let GpuLayerCtx { wf, wf_buf, layer_cache, buffers, buffer_pool } =
        *gpu;
    let v = VARIANT;

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
    let attn = layer_cache.attn.full().ok_or(
        LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "full_attn weights (called on linear-attn layer)",
        },
    )?;
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
        q_host[dst_off..dst_off + v.head_dim].copy_from_slice(
            &q_proj_host[src_off..src_off + v.head_dim],
        );
        q_gate_host[dst_off..dst_off + v.head_dim].copy_from_slice(
            &q_proj_host[src_off + v.head_dim..src_off + 2 * v.head_dim],
        );
    }

    // ── Per-head Q rms_norm ──────────────────────────────────────
    let q_norm_name =
        format!("model.layers.{layer_idx}.self_attn.q_norm.weight");
    rms_norm_per_head_cpu(
        wf,
        &q_norm_name,
        v.num_attn_heads,
        v.head_dim,
        &mut q_host,
    )?;

    // ── Per-head K rms_norm ──────────────────────────────────────
    let k_norm_name =
        format!("model.layers.{layer_idx}.self_attn.k_norm.weight");
    rms_norm_per_head_cpu(
        wf,
        &k_norm_name,
        v.num_kv_heads,
        v.head_dim,
        &mut k_host,
    )?;

    // ── RoPE on q + k ────────────────────────────────────────────
    apply_rotary_emb(pos, &mut q_host, &mut k_host)?;

    // ── KV append: host-canonical + GPU mirror (slice 5d-7b) ─────
    let cache_pos = kv_state.len as usize;
    if cache_pos + 1 > crate::riir::variants::MAX_SEQ_LEN {
        return Err(LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "kv cache overflow",
        });
    }
    let row_start = cache_pos * kv_dim;
    let row_end = row_start + kv_dim;
    kv_state.k_cache[row_start..row_end].copy_from_slice(&k_host);
    kv_state.v_cache[row_start..row_end].copy_from_slice(&v_host);
    kv_state.len += 1;

    // GPU mirror of host KV — feeds the GPU SDPA fast path when the
    // gate predicate fires below. Mirrors C `infer.m:4796..4802`. Only
    // populated for full-attn layers; bounded by `GPU_KV_SEQ` to avoid
    // overrunning the persistent buffer (above that, the C path also
    // falls back to CPU SDPA).
    let fa_idx = full_attn_layer_idx_for(layer_idx);
    if let Some(fa_idx) = fa_idx {
        if cache_pos < crate::riir::variants::GPU_KV_SEQ {
            // SAFETY: shared-storage buffer; no GPU work in flight on
            // gpu_kv_k/v at this point (no encode has been dispatched
            // yet this layer; previous dispatch's CMD2 commit-wait
            // happened in last layer's `complete_deferred_experts_into`
            // drain at the top of this layer's eval call).
            unsafe {
                let k_dst = buffer_pool.handle(buffers.gpu_kv_k[fa_idx]).contents() as *mut f32;
                let v_dst = buffer_pool.handle(buffers.gpu_kv_v[fa_idx]).contents() as *mut f32;
                std::ptr::copy_nonoverlapping(
                    k_host.as_ptr(),
                    k_dst.add(row_start),
                    kv_dim,
                );
                std::ptr::copy_nonoverlapping(
                    v_host.as_ptr(),
                    v_dst.add(row_start),
                    kv_dim,
                );
            }
        }
    }

    // ── Decide between GPU SDPA fast path and CPU SDPA ──────────
    //
    // Match C `infer.m:5054` exactly: gate fires when the layer is
    // full-attn, KV mirror fits in the persistent buffer, and we're
    // past the GPU dispatch break-even point (kv_len < 32 keeps
    // command-encoder overhead from dominating).
    let kv_len = kv_state.len;
    let gpu_attn_ready = fa_idx.is_some()
        && kv_len >= 32
        && (kv_len as usize) < crate::riir::variants::GPU_KV_SEQ;

    let gpu_attn_args = if gpu_attn_ready {
        let fa_idx = fa_idx.expect("gpu_attn_ready ⇒ Some(fa_idx)");
        // Stage Q + q_gate (both post-norm + RoPE for q) into the
        // shared GPU scratch buffers. Read by Enc A1 (scores) and
        // Enc A4 (sigmoid gate). SAFETY: shared-storage; no GPU work
        // in flight on these buffers (CMD1 above committed and waited;
        // CMD2 hasn't been built yet).
        unsafe {
            let q_dst = buffer_pool.handle(buffers.gpu_attn_q).contents() as *mut f32;
            let g_dst = buffer_pool.handle(buffers.gpu_attn_gate).contents() as *mut f32;
            std::ptr::copy_nonoverlapping(q_host.as_ptr(), q_dst, q_dim);
            std::ptr::copy_nonoverlapping(
                q_gate_host.as_ptr(),
                g_dst,
                q_dim,
            );
        }
        Some(GpuAttnEncodeArgs {
            fa_idx,
            kv_len: kv_len as u32,
        })
    } else {
        // CPU SDPA fallback. Slice the caches to the occupied prefix
        // and stage the result into batch_out[6] for o_proj.
        let kv_total = (kv_len as usize) * kv_dim;
        let mut attn_out = vec![0.0f32; q_dim];
        sdpa_cpu(
            kv_len,
            &q_host,
            &q_gate_host,
            &kv_state.k_cache[..kv_total],
            &kv_state.v_cache[..kv_total],
            &mut attn_out,
        )?;

        let dst = buffer_pool.handle(buffers.batch_out[6]).contents() as *mut f32;
        // SAFETY: shared-storage; no GPU work in flight (CMD1
        // committed and waited above).
        unsafe {
            std::ptr::copy_nonoverlapping(
                attn_out.as_ptr(),
                dst,
                q_dim,
            );
        }
        debug_assert!(
            buffer_pool.handle(buffers.batch_out[6]).length() as usize
                >= q_dim * std::mem::size_of::<f32>(),
            "batch_out[6] sized {} bytes, need {} for full-attn o_proj input",
            buffer_pool.handle(buffers.batch_out[6]).length() as NSUInteger,
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
    let GpuLayerCtx { wf: _, wf_buf, layer_cache: _, buffers, buffer_pool } =
        *gpu;
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
/// Runs the per-token pre-MoE forward in a loop (steps 1-14 of the
/// Phase B plan), capturing per-token intermediates and the GPU
/// outputs (`h_mid`, `h_post`, `shared_out`) to host stacks. Then
/// builds the joint N×k_active expert-bucket CSR and runs
/// [`crate::riir::moe::expert_forward::encode_moe_batched_permute_fuse`] once
/// for the whole chunk — one expert blob read per non-empty bucket
/// instead of K reads per token. The per-token combine
/// (h_mid + moe_sum + sigmoid(shared_gate) * shared_out) runs on
/// CPU; B4 batches it via a GPU kernel.
///
/// **B1 sub-step:** steps 1-14 per-token, step 16 (MoE permute-fuse)
/// batched. Sub-steps B2-B4 progressively batch SDPA, projections,
/// shared FFN, and combine.
///
/// Pre: `hidden_in[t*hidden_dim..(t+1)*hidden_dim]` holds token t's
/// input hidden state. Post: `hidden_out[t*hidden_dim..]` holds the
/// post-combine hidden state. `kv_state.len` advances by `n_tokens`.
///
/// **No deferred dispatch, no prefetch.** The batched path reads
/// expert blobs synchronously per bucket; the per-token prefetch
/// state machine is decode-only after Phase H.
#[allow(clippy::too_many_arguments)]
pub(in crate::riir) fn batched_full_attn_layer_forward(
    metal: &mut MetalContext,
    gpu: &GpuLayerCtx<'_>,
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
    mut prefetch: Option<crate::riir::attn::linear_attn_forward::PrefetchEnv<'_>>,
    // S7-2 (session 6): hidden_in / hidden_out live on GPU. The
    // orchestrator double-buffers two MtlBuffers and swaps between
    // layers, eliminating the inter-layer host bounce. Layer body
    // reads `hidden_in_buf` for embedding/residual + writes
    // `hidden_out_buf` via the GPU combine kernel.
    hidden_in_buf: &metal::Buffer,
    hidden_out_buf: &metal::Buffer,
) -> Result<(), LayerForwardError> {
    use crate::riir::moe::expert_forward::{
        encode_moe_batched_permute_fuse, encode_moe_combine_residual_n_tokens,
        MAX_K,
    };
    use crate::riir::backend::gpu::metal::MtlBuffer;
    use crate::riir::moe::moe_router::build_expert_buckets;

    let GpuLayerCtx { wf, wf_buf, layer_cache, buffers: _, buffer_pool } =
        *gpu;
    let v = VARIANT;
    debug_assert!(k_active <= MAX_K);

    let hidden_dim = v.hidden_dim;
    let q_dim = v.num_attn_heads * v.head_dim;
    let kv_dim = v.num_kv_heads * v.head_dim;

    // S7-2: h_mid + shared_gate stay on GPU (consumed by GPU combine).
    // h_post_stack is still read back so the per-token bucket_input
    // gather (`bucket_input_host[assignment_idx]`) can index into it.
    let mut h_post_stack = vec![0.0f32; n_tokens * hidden_dim];
    let all_routing_indices: Vec<i32>;
    let all_routing_weights: Vec<f32>;

    let kv_start = kv_state.len;
    if (kv_start as usize) + n_tokens > crate::riir::variants::MAX_SEQ_LEN {
        return Err(LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "kv cache overflow",
        });
    }
    let attn = layer_cache.attn.full().ok_or(
        LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "full_attn weights (B3 batched path)",
        },
    )?;
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
    let q_proj_dim = q_dim * 2;
    let device = metal.device().clone();
    let mv = MatvecPipelines::fetch(metal)?;
    let rms_pipes = RmsNormBf16Pipelines::fetch(metal)?;

    // ── Phase 1a+1b: fused input rms_norm + Q/K/V projections. ───
    // No host bounce between normed_buf and the Q/K/V matvecs, so
    // both phases share one cmdbuf. S7-1a fusion.
    let rms_n_pipe =
        crate::riir::backend::gpu::gpu_norm::RmsNormBf16FusedNTokensPipeline::fetch(metal)?;
    // hidden_in_buf comes from the orchestrator's double-buffer (S7-2).
    let normed_buf = MtlBuffer::<f32>::with_len(&device, n_tokens * hidden_dim);
    let q_proj_buf =
        MtlBuffer::<f32>::with_len(&device, n_tokens * q_proj_dim);
    let k_proj_buf = MtlBuffer::<f32>::with_len(&device, n_tokens * kv_dim);
    let v_proj_buf = MtlBuffer::<f32>::with_len(&device, n_tokens * kv_dim);
    {
        let queue = metal.queue_clone();
        let cmdbuf = queue.new_command_buffer();
        crate::riir::backend::gpu::gpu_norm::encode_rms_norm_bf16_fused_n_tokens(
            cmdbuf,
            &rms_n_pipe,
            hidden_in_buf,
            wf_buf.buffer(),
            layer_cache.input_layernorm_w,
            normed_buf.buffer(),
            hidden_dim as u32,
            n_tokens as u32,
            crate::riir::variants::RMS_NORM_EPS,
        );
        encode_dense_matmul_n_tokens(
            cmdbuf,
            metal.qmm(),
            &mv,
            wf_buf.buffer(),
            attn.q_proj_w,
            attn.q_proj_s,
            attn.q_proj_b,
            normed_buf.buffer(),
            0,
            q_proj_buf.buffer(),
            0,
            hidden_dim as u32,
            q_proj_dim as u32,
            n_tokens as u32,
            q_bits,
        );
        encode_dense_matmul_n_tokens(
            cmdbuf,
            metal.qmm(),
            &mv,
            wf_buf.buffer(),
            attn.k_proj_w,
            attn.k_proj_s,
            attn.k_proj_b,
            normed_buf.buffer(),
            0,
            k_proj_buf.buffer(),
            0,
            hidden_dim as u32,
            kv_dim as u32,
            n_tokens as u32,
            k_bits,
        );
        encode_dense_matmul_n_tokens(
            cmdbuf,
            metal.qmm(),
            &mv,
            wf_buf.buffer(),
            attn.v_proj_w,
            attn.v_proj_s,
            attn.v_proj_b,
            normed_buf.buffer(),
            0,
            v_proj_buf.buffer(),
            0,
            hidden_dim as u32,
            kv_dim as u32,
            n_tokens as u32,
            v_bits,
        );
        metal.commit_and_wait_labeled(cmdbuf, "batched_rms_norm_qkv_proj");
    }
    let _ = &rms_pipes;
    let q_proj_stack = q_proj_buf.to_vec();
    let k_stack = k_proj_buf.to_vec();
    let v_stack = v_proj_buf.to_vec();

    // ── Phase 1c: per-token Q split + per-head Q/K norm + RoPE +
    //    KV append into kv_state. ─────────────────────────────────
    let mut q_stack = vec![0.0f32; n_tokens * q_dim];
    let mut q_gate_stack = vec![0.0f32; n_tokens * q_dim];
    for t in 0..n_tokens {
        let pos = start_pos + t as i32;
        let q_proj_t = &q_proj_stack
            [t * q_proj_dim..(t + 1) * q_proj_dim];
        let mut q_host = vec![0.0f32; q_dim];
        let mut q_gate_host = vec![0.0f32; q_dim];
        let mut k_host =
            k_stack[t * kv_dim..(t + 1) * kv_dim].to_vec();
        let v_host =
            v_stack[t * kv_dim..(t + 1) * kv_dim].to_vec();
        for h in 0..v.num_attn_heads {
            let src_off = h * (2 * v.head_dim);
            let dst_off = h * v.head_dim;
            q_host[dst_off..dst_off + v.head_dim].copy_from_slice(
                &q_proj_t[src_off..src_off + v.head_dim],
            );
            q_gate_host[dst_off..dst_off + v.head_dim].copy_from_slice(
                &q_proj_t[src_off + v.head_dim..src_off + 2 * v.head_dim],
            );
        }
        rms_norm_per_head_cpu(
            wf,
            &format!("model.layers.{layer_idx}.self_attn.q_norm.weight"),
            v.num_attn_heads,
            v.head_dim,
            &mut q_host,
        )?;
        rms_norm_per_head_cpu(
            wf,
            &format!("model.layers.{layer_idx}.self_attn.k_norm.weight"),
            v.num_kv_heads,
            v.head_dim,
            &mut k_host,
        )?;
        apply_rotary_emb(pos, &mut q_host, &mut k_host)?;
        let cache_pos = (kv_start as usize) + t;
        let row_start = cache_pos * kv_dim;
        let row_end = row_start + kv_dim;
        kv_state.k_cache[row_start..row_end].copy_from_slice(&k_host);
        kv_state.v_cache[row_start..row_end].copy_from_slice(&v_host);
        q_stack[t * q_dim..(t + 1) * q_dim].copy_from_slice(&q_host);
        q_gate_stack[t * q_dim..(t + 1) * q_dim]
            .copy_from_slice(&q_gate_host);
    }
    kv_state.len += n_tokens as i32;

    // ── Phase 2: batched tiled SDPA over the joint q stack against
    //    the (now-extended) kv state. ──────────────────────────────
    let device = metal.device().clone();
    let kv_len_total = kv_state.len as u32;
    let kv_floats = (kv_len_total as usize) * kv_dim;
    let q_buf = MtlBuffer::<f32>::with_data(&device, &q_stack);
    let k_gpu = MtlBuffer::<f32>::with_data(
        &device,
        &kv_state.k_cache[..kv_floats],
    );
    let v_gpu = MtlBuffer::<f32>::with_data(
        &device,
        &kv_state.v_cache[..kv_floats],
    );
    let attn_out_buf =
        MtlBuffer::<f32>::with_len(&device, n_tokens * q_dim);

    let sdpa_pipes = FlashSdpaPipelines::fetch(metal)?;
    let softmax_scale = 1.0f32 / (v.head_dim as f32).sqrt();
    let heads_per_kv = (v.num_attn_heads / v.num_kv_heads) as u32;
    let queue = metal.queue_clone();
    let cmdbuf = queue.new_command_buffer();
    encode_sdpa_causal_flash(
        cmdbuf,
        &sdpa_pipes,
        q_buf.buffer(),
        k_gpu.buffer(),
        v_gpu.buffer(),
        attn_out_buf.buffer(),
        n_tokens as u32,
        v.num_attn_heads as u32,
        heads_per_kv,
        v.head_dim as u32,
        kv_dim as u32,
        kv_start as u32,
        kv_len_total,
        softmax_scale,
    );
    metal.commit_and_wait_labeled(cmdbuf, "batched_sdpa_causal_flash");
    // Tiled SDPA output is gate-free (per session-2 Phase 3 design).
    // Apply sigmoid_gate per token on host below.
    let attn_out_stack = attn_out_buf.to_vec();

    // ── Phase 3a: per-token sigmoid_gate (CPU) → attn_with_gate_stack ─
    let mut attn_with_gate_stack = vec![0.0f32; n_tokens * q_dim];
    for t in 0..n_tokens {
        let q_gate = &q_gate_stack[t * q_dim..(t + 1) * q_dim];
        let attn_raw = &attn_out_stack[t * q_dim..(t + 1) * q_dim];
        let out_t = &mut attn_with_gate_stack[t * q_dim..(t + 1) * q_dim];
        for i in 0..q_dim {
            let sg = 1.0f32 / (1.0f32 + (-q_gate[i]).exp());
            out_t[i] = sg * attn_raw[i];
        }
    }

    // ── Phase 3b+3c: fused O projection + post-attn residual_add +
    //    rms_norm + gate matvec + shared-gate matvec + GPU MoE router.
    //    o_proj_buf is consumed only by the residual_add inside the
    //    post-attn chain — no host bounce between them, so one cmdbuf
    //    covers both. S7-1a fusion.
    let attn_with_gate_buf =
        MtlBuffer::<f32>::with_data(&device, &attn_with_gate_stack);
    let o_proj_buf =
        MtlBuffer::<f32>::with_len(&device, n_tokens * hidden_dim);
    let gate_bits =
        bits_of(wf, &format!("model.layers.{layer_idx}.mlp.gate.weight"));
    let seg_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert_gate.weight"
        ),
    );
    let resid_add_n_pso = metal
        .pipeline("residual_add_n_tokens")?
        .clone();
    let router_pipes =
        crate::riir::moe::gpu_moe_router::MoeRouterPipelines::fetch(metal)?;
    let h_mid_buf =
        MtlBuffer::<f32>::with_len(&device, n_tokens * hidden_dim);
    let h_post_buf =
        MtlBuffer::<f32>::with_len(&device, n_tokens * hidden_dim);
    let gate_logits_buf =
        MtlBuffer::<f32>::with_len(&device, n_tokens * v.num_experts);
    let shared_gate_buf = MtlBuffer::<f32>::with_len(&device, n_tokens);
    let routing_indices_buf = MtlBuffer::<i32>::with_len(&device, n_tokens * k_active);
    let routing_weights_buf =
        MtlBuffer::<f32>::with_len(&device, n_tokens * k_active);
    {
        let queue = metal.queue_clone();
        let cmdbuf = queue.new_command_buffer();
        encode_dense_matmul_n_tokens(
            cmdbuf,
            metal.qmm(),
            &mv,
            wf_buf.buffer(),
            attn.o_proj_w,
            attn.o_proj_s,
            attn.o_proj_b,
            attn_with_gate_buf.buffer(),
            0,
            o_proj_buf.buffer(),
            0,
            q_dim as u32,
            hidden_dim as u32,
            n_tokens as u32,
            o_bits,
        );
        crate::riir::backend::gpu::gpu_norm::encode_residual_add_n_tokens_into(
            cmdbuf,
            &resid_add_n_pso,
            o_proj_buf.buffer(),
            hidden_in_buf,
            h_mid_buf.buffer(),
            n_tokens as u32,
            hidden_dim as u32,
        );
        crate::riir::backend::gpu::gpu_norm::encode_rms_norm_bf16_fused_n_tokens(
            cmdbuf,
            &rms_n_pipe,
            h_mid_buf.buffer(),
            wf_buf.buffer(),
            layer_cache.post_attention_layernorm_w,
            h_post_buf.buffer(),
            hidden_dim as u32,
            n_tokens as u32,
            crate::riir::variants::RMS_NORM_EPS,
        );
        encode_dense_matmul_n_tokens(
            cmdbuf,
            metal.qmm(),
            &mv,
            wf_buf.buffer(),
            layer_cache.gate.w,
            layer_cache.gate.s,
            layer_cache.gate.b,
            h_post_buf.buffer(),
            0,
            gate_logits_buf.buffer(),
            0,
            hidden_dim as u32,
            v.num_experts as u32,
            n_tokens as u32,
            gate_bits,
        );
        encode_dense_matmul_n_tokens(
            cmdbuf,
            metal.qmm(),
            &mv,
            wf_buf.buffer(),
            layer_cache.shared.seg_w,
            layer_cache.shared.seg_s,
            layer_cache.shared.seg_b,
            h_post_buf.buffer(),
            0,
            shared_gate_buf.buffer(),
            0,
            hidden_dim as u32,
            1,
            n_tokens as u32,
            seg_bits,
        );
        crate::riir::moe::gpu_moe_router::encode_moe_router(
            cmdbuf,
            &router_pipes,
            gate_logits_buf.buffer(),
            routing_indices_buf.buffer(),
            routing_weights_buf.buffer(),
            n_tokens as u32,
            v.num_experts as u32,
            k_active as u32,
        );
        metal.commit_and_wait_labeled(
            cmdbuf,
            "batched_oproj_post_attn_route",
        );
    }
    // Bulk readback (S7-2): only h_post_stack (for bucket_input
    // gather) and the small routing buffers — h_mid + shared_gate
    // stay on GPU and feed directly into the GPU combine kernel.
    h_post_stack.copy_from_slice(&h_post_buf.to_vec());
    all_routing_indices = routing_indices_buf.to_vec();
    all_routing_weights = routing_weights_buf.to_vec();

    // ── Phase 3d: batched shared FFN (gate matvec, up matvec,
    //    flat-element-wise swiglu, down matvec) over h_post_stack. ─
    let s_gate_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight"
        ),
    );
    let s_up_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight"
        ),
    );
    let s_down_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight"
        ),
    );
    let shared_gate_w = layer_cache.shared.gate_w;
    let shared_gate_s = layer_cache.shared.gate_s;
    let shared_gate_b = layer_cache.shared.gate_b;
    let shared_up_w = layer_cache.shared.up_w;
    let shared_up_s = layer_cache.shared.up_s;
    let shared_up_b = layer_cache.shared.up_b;
    let shared_down_w = layer_cache.shared.down_w;
    let shared_down_s = layer_cache.shared.down_s;
    let shared_down_b = layer_cache.shared.down_b;
    let h_post_buf =
        MtlBuffer::<f32>::with_data(&device, &h_post_stack);
    // Shared FFN's gate-proj output (NOT the per-token sigmoid scalar
    // from Phase 3c — that's `shared_gate_buf` and is still in scope).
    // Renamed from `shared_gate_buf` to `shared_ffn_gate_buf` so the
    // GPU combine kernel below reads the right buffer.
    let shared_ffn_gate_buf = MtlBuffer::<f32>::with_len(
        &device,
        n_tokens * v.shared_intermediate,
    );
    let shared_up_buf = MtlBuffer::<f32>::with_len(
        &device,
        n_tokens * v.shared_intermediate,
    );
    let shared_act_buf = MtlBuffer::<f32>::with_len(
        &device,
        n_tokens * v.shared_intermediate,
    );
    let shared_down_buf =
        MtlBuffer::<f32>::with_len(&device, n_tokens * hidden_dim);
    let swiglu_pso = metal.pipeline("swiglu_fused")?.clone();
    // S7-1a fusion: one cmdbuf covers Phase 3d (shared FFN) AND
    // Phase 3e (MoE permute-fuse). CPU bucket-build + bucket setup
    // happen *between* encoding 3d and 3e; the GPU runs 3d while
    // CPU prepares 3e's bucket inputs.
    let queue = metal.queue_clone();
    let cmdbuf = queue.new_command_buffer();
    encode_dense_matmul_n_tokens(
        cmdbuf,
        metal.qmm(),
        &mv,
        wf_buf.buffer(),
        shared_gate_w,
        shared_gate_s,
        shared_gate_b,
        h_post_buf.buffer(),
        0,
        shared_ffn_gate_buf.buffer(),
        0,
        hidden_dim as u32,
        v.shared_intermediate as u32,
        n_tokens as u32,
        s_gate_bits,
    );
    encode_dense_matmul_n_tokens(
        cmdbuf,
        metal.qmm(),
        &mv,
        wf_buf.buffer(),
        shared_up_w,
        shared_up_s,
        shared_up_b,
        h_post_buf.buffer(),
        0,
        shared_up_buf.buffer(),
        0,
        hidden_dim as u32,
        v.shared_intermediate as u32,
        n_tokens as u32,
        s_up_bits,
    );
    // swiglu_fused is element-wise → flat dispatch over
    // n_tokens * shared_intermediate.
    {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&swiglu_pso);
        enc.set_buffer(0, Some(shared_ffn_gate_buf.buffer()), 0);
        enc.set_buffer(1, Some(shared_up_buf.buffer()), 0);
        enc.set_buffer(2, Some(shared_act_buf.buffer()), 0);
        let dim = (n_tokens * v.shared_intermediate) as u32;
        enc.set_bytes(3, 4, (&dim as *const u32).cast());
        let num_tgs = (dim + 255) / 256;
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs as NSUInteger, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
    encode_dense_matmul_n_tokens(
        cmdbuf,
        metal.qmm(),
        &mv,
        wf_buf.buffer(),
        shared_down_w,
        shared_down_s,
        shared_down_b,
        shared_act_buf.buffer(),
        0,
        shared_down_buf.buffer(),
        0,
        v.shared_intermediate as u32,
        hidden_dim as u32,
        n_tokens as u32,
        s_down_bits,
    );
    // 3d encoded — fall through to 3e CPU prep without committing.

    let buckets = build_expert_buckets(
        &all_routing_indices,
        &all_routing_weights,
        n_tokens,
        k_active,
        v.num_experts,
    );

    let device = metal.device().clone();
    let total_assignments = buckets.token_idx.len();
    debug_assert_eq!(total_assignments, n_tokens * k_active);

    // Expert weight buffers — see `batched_linear_attn_layer_forward`
    // for the three-source explanation (prefetch hit / mmap / pread).
    let expert_size = v.expert_size_4bit();
    let mode = expert_files.mode();
    let num_buckets = buckets.expert_ids.len();
    let mut bucket_prefetch_slot: Vec<Option<usize>> = vec![None; num_buckets];
    let mut hit_count: u64 = 0;
    if let Some(pe) = prefetch.as_mut() {
        if let Some(status) = pe.prefetch.wait_for(layer_idx) {
            for (bi, &expert_id) in buckets.expert_ids.iter().enumerate() {
                for buf_idx in 0..status.k {
                    if status.loaded_indices[buf_idx] == expert_id {
                        bucket_prefetch_slot[bi] = Some(buf_idx);
                        hit_count += 1;
                        break;
                    }
                }
            }
        }
        pe.prefetch.record_outcome(
            hit_count,
            num_buckets as u64 - hit_count,
        );
    }

    let mut owned_blobs: Vec<Option<MtlBuffer<u8>>> = (0..num_buckets)
        .map(|_| None)
        .collect();
    if mode == crate::riir::io::expert_io::ExpertIoMode::Pread {
        let mut blob_scratch = vec![0u8; expert_size];
        for (bi, &expert_id) in buckets.expert_ids.iter().enumerate() {
            if bucket_prefetch_slot[bi].is_some() {
                continue;
            }
            expert_files
                .read_expert(layer_idx, expert_id as usize, &mut blob_scratch)?;
            owned_blobs[bi] =
                Some(MtlBuffer::<u8>::with_data(&device, &blob_scratch));
        }
    }
    let expert_refs: Vec<crate::riir::moe::expert_forward::ExpertRef<'_>> = buckets
        .expert_ids
        .iter()
        .enumerate()
        .map(|(bi, &expert_id)| {
            if let Some(buf_idx) = bucket_prefetch_slot[bi] {
                let pe = prefetch.as_ref().expect("prefetch slot requires env");
                let buf = moe_buffers.data_prefetch_buffer(
                    buffer_pool,
                    pe.prefetch_set,
                    buf_idx,
                );
                (buf, 0u64)
            } else if mode == crate::riir::io::expert_io::ExpertIoMode::Mmap {
                expert_files
                    .mmap_buffer_for_expert(layer_idx, expert_id as u32)
                    .expect("mmap layer present in mmap mode")
            } else {
                let blob =
                    owned_blobs[bi].as_ref().expect("synced miss has owned blob");
                (blob.buffer(), 0u64)
            }
        })
        .collect();

    let mut bucket_input_host = vec![0.0f32; total_assignments * hidden_dim];
    for assignment_idx in 0..total_assignments {
        let t = buckets.token_idx[assignment_idx] as usize;
        let src = &h_post_stack[t * hidden_dim..(t + 1) * hidden_dim];
        let dst_off = assignment_idx * hidden_dim;
        bucket_input_host[dst_off..dst_off + hidden_dim].copy_from_slice(src);
    }

    let bucket_input =
        MtlBuffer::<f32>::with_data(&device, &bucket_input_host);
    let bucket_gate = MtlBuffer::<f32>::with_len(
        &device,
        total_assignments * v.moe_intermediate,
    );
    let bucket_up = MtlBuffer::<f32>::with_len(
        &device,
        total_assignments * v.moe_intermediate,
    );
    let bucket_act = MtlBuffer::<f32>::with_len(
        &device,
        total_assignments * v.moe_intermediate,
    );
    let bucket_out =
        MtlBuffer::<f32>::with_len(&device, total_assignments * hidden_dim);
    let bucket_token_idx =
        MtlBuffer::<i32>::with_data(&device, &buckets.token_idx);
    let bucket_weights =
        MtlBuffer::<f32>::with_data(&device, &buckets.weights);
    let out_sum_zeros = vec![0.0f32; n_tokens * hidden_dim];
    let out_sum = MtlBuffer::<f32>::with_data(&device, &out_sum_zeros);

    let matvec = MatvecPipelines::fetch(metal)?;
    let swiglu = metal.pipeline("swiglu_fused")?.clone();
    let bucket_accumulate =
        metal.pipeline("moe_bucket_accumulate")?.clone();
    // 3e encodes into the same cmdbuf the 3d block opened above —
    // GPU runs 3d's matvecs+swiglu+down concurrently with CPU's bucket
    // setup, then 3e dispatches after them in the same buffer.
    encode_moe_batched_permute_fuse(
        cmdbuf,
        &matvec,
        &swiglu,
        &bucket_accumulate,
        &expert_refs,
        bucket_input.buffer(),
        bucket_gate.buffer(),
        bucket_up.buffer(),
        bucket_act.buffer(),
        bucket_out.buffer(),
        bucket_token_idx.buffer(),
        bucket_weights.buffer(),
        out_sum.buffer(),
        &buckets,
        v,
    );
    // S7-2: GPU combine into hidden_out_buf — replaces the CPU loop
    // that read h_mid_stack / shared_out_stack / shared_gate_scores /
    // moe_sum_host. With this on the same cmdbuf as 3d/3e, the final
    // hidden state for this layer is on GPU and feeds directly into
    // the next layer's input without a host bounce.
    let combine_pso = metal
        .pipeline("moe_combine_residual_n_tokens")?
        .clone();
    encode_moe_combine_residual_n_tokens(
        cmdbuf,
        &combine_pso,
        h_mid_buf.buffer(),
        out_sum.buffer(),
        shared_down_buf.buffer(),
        shared_gate_buf.buffer(),
        hidden_out_buf,
        n_tokens as u32,
        hidden_dim as u32,
    );
    metal.commit_and_wait_labeled(cmdbuf, "batched_shared_ffn_moe_combine");

    // Phase 3: record_actual for prefetch's next-token prediction.
    // See `batched_linear_attn_layer_forward` for the semantic.
    if let Some(pe) = prefetch.as_mut() {
        use crate::riir::moe::expert_forward::MAX_K;
        let mut actuals: [i32; MAX_K] = [0; MAX_K];
        let len = k_active.min(MAX_K).min(all_routing_indices.len());
        actuals[..len].copy_from_slice(&all_routing_indices[..len]);
        pe.prefetch.record_actual(layer_idx, actuals);
    }

    Ok(())
}
