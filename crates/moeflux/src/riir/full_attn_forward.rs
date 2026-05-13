//! End-to-end full-attention layer forward — Phase 4d.
//!
//! Companion to [`super::linear_attn_forward::linear_attn_layer_forward`]
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
//! and lives in [`super::linear_attn_forward::post_attention_tail`].
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

use super::expert_forward::MoeBuffers;
use super::expert_io::ExpertFiles;
use super::gpu_attn::{encode_sdpa_causal_tiled, BatchedSdpaPipelines};
use super::gpu_matvec::{encode_matvec, MatvecPipelines, MatvecSpec};
use super::gpu_norm::{encode_rms_norm_bf16_into, RmsNormBf16Pipelines};
use super::layer_weight_cache::LayerWeightCache;
use super::linear_attn_forward::{
    bits_of, full_attn_layer_idx_for, moe_dispatch_per_token,
    post_attention_pre_moe, read_buffer_to_vec, GpuAttnEncodeArgs,
    LayerForwardBuffers, LayerForwardError, OProj, PostAttnIntermediates,
};
use super::metal::MetalBackend;
use super::mtl_weight_buf::MtlWeightBuf;
use super::rms_norm::rms_norm_per_head_cpu;
use super::rope::apply_rotary_emb;
use super::sdpa::sdpa_cpu;
use super::state::KvCache;
use super::variants::VARIANT;
use super::weight_file::WeightFile;

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
pub(super) fn full_attn_pre_moe_layer_forward(
    metal: &mut MetalBackend,
    wf: &WeightFile,
    wf_buf: &MtlWeightBuf,
    layer_cache: &LayerWeightCache,
    buffers: &mut LayerForwardBuffers,
    layer_idx: usize,
    pos: i32,
    k_active: usize,
    kv_state: &mut KvCache,
    // Slice 5d-8: see `linear_attn_layer_forward` for the contract.
    prev_layer_chained: bool,
) -> Result<PostAttnIntermediates, LayerForwardError> {
    let v = VARIANT;

    // Reject linear-attn layers up front. Mirror the symmetric guard
    // in `linear_attn_layer_forward`.
    if v.layer_kind(layer_idx) != super::variants::LayerKind::FullAttn {
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
                &buffers.input,
                wf_buf.buffer(),
                layer_cache.input_layernorm_w,
                &buffers.sum_sq,
                &buffers.normed,
                v.hidden_dim as u32,
                super::variants::RMS_NORM_EPS,
            );
        }

        let specs = [
            MatvecSpec {
                w_off: q_w,
                s_off: q_s,
                b_off: q_b,
                input: &buffers.normed,
                output: &buffers.q_proj_out,
                out_dim: q_proj_dim as u32,
                in_dim: v.hidden_dim as u32,
                bits: q_bits,
            },
            MatvecSpec {
                w_off: k_w,
                s_off: k_s,
                b_off: k_b,
                input: &buffers.normed,
                output: &buffers.k_out,
                out_dim: kv_dim as u32,
                in_dim: v.hidden_dim as u32,
                bits: k_bits,
            },
            MatvecSpec {
                w_off: v_w,
                s_off: v_s,
                b_off: v_b,
                input: &buffers.normed,
                output: &buffers.v_out,
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
    let q_proj_host = read_buffer_to_vec(&buffers.q_proj_out, q_proj_dim);
    let mut k_host = read_buffer_to_vec(&buffers.k_out, kv_dim);
    let v_host = read_buffer_to_vec(&buffers.v_out, kv_dim);

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
    if cache_pos + 1 > super::variants::MAX_SEQ_LEN {
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
        if cache_pos < super::variants::GPU_KV_SEQ {
            // SAFETY: shared-storage buffer; no GPU work in flight on
            // gpu_kv_k/v at this point (no encode has been dispatched
            // yet this layer; previous dispatch's CMD2 commit-wait
            // happened in last layer's `complete_deferred_experts_into`
            // drain at the top of this layer's eval call).
            unsafe {
                let k_dst = buffers.gpu_kv_k[fa_idx].contents() as *mut f32;
                let v_dst = buffers.gpu_kv_v[fa_idx].contents() as *mut f32;
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
        && (kv_len as usize) < super::variants::GPU_KV_SEQ;

    let gpu_attn_args = if gpu_attn_ready {
        let fa_idx = fa_idx.expect("gpu_attn_ready ⇒ Some(fa_idx)");
        // Stage Q + q_gate (both post-norm + RoPE for q) into the
        // shared GPU scratch buffers. Read by Enc A1 (scores) and
        // Enc A4 (sigmoid gate). SAFETY: shared-storage; no GPU work
        // in flight on these buffers (CMD1 above committed and waited;
        // CMD2 hasn't been built yet).
        unsafe {
            let q_dst = buffers.gpu_attn_q.contents() as *mut f32;
            let g_dst = buffers.gpu_attn_gate.contents() as *mut f32;
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

        let dst = buffers.batch_out[6].contents() as *mut f32;
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
            buffers.batch_out[6].length() as usize
                >= q_dim * std::mem::size_of::<f32>(),
            "batch_out[6] sized {} bytes, need {} for full-attn o_proj input",
            buffers.batch_out[6].length() as NSUInteger,
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
        wf,
        wf_buf,
        layer_cache,
        buffers,
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
pub(super) fn full_attn_layer_forward(
    metal: &mut MetalBackend,
    wf: &WeightFile,
    wf_buf: &MtlWeightBuf,
    layer_cache: &LayerWeightCache,
    buffers: &mut LayerForwardBuffers,
    moe: &mut MoeBuffers,
    deferred: &mut super::deferred::DeferredRing,
    layer_idx: usize,
    pos: i32,
    k_active: usize,
    expert_files: &ExpertFiles,
    pool: &rayon::ThreadPool,
    prefetch: &mut super::PrefetchState,
    // Slice 5d-9: which `data_prefetch` set this layer reads from
    // (parity ping-pong: `layer_idx % 2`).
    prefetch_set: usize,
    kv_state: &mut KvCache,
    gpu_combine: bool,
    // Slice 5d-8: see `linear_attn_layer_forward` for the contract.
    prev_layer_chained: bool,
    chain_next_norm_off: Option<u64>,
) -> Result<(), LayerForwardError> {
    let intermediates = full_attn_pre_moe_layer_forward(
        metal,
        wf,
        wf_buf,
        layer_cache,
        buffers,
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

/// Pre-SDPA half for the batched prefill orchestrator. Runs the
/// per-token CMD1 (input rms_norm + Q/K/V projections), host-bounces
/// q/k/v, applies per-head Q/K rms_norm + RoPE, and appends the new
/// K/V row to `kv_state`. Returns per-token (q_host, q_gate_host)
/// for the caller to stack into a batched-SDPA input.
///
/// Mirrors lines ~106-313 of [`full_attn_pre_moe_layer_forward`] but
/// skips the SDPA decision (`gpu_attn_args` setup, sdpa_cpu fallback)
/// since the batched path replaces those with a single tiled SDPA
/// dispatch over all N tokens. Also skips populating
/// `buffers.gpu_kv_k/v` mirrors — those are decode-only after Phase G.
#[allow(clippy::too_many_arguments)]
fn full_attn_pre_sdpa_capture(
    metal: &mut MetalBackend,
    wf: &WeightFile,
    wf_buf: &MtlWeightBuf,
    layer_cache: &LayerWeightCache,
    buffers: &mut LayerForwardBuffers,
    layer_idx: usize,
    pos: i32,
    kv_state: &mut KvCache,
) -> Result<(Vec<f32>, Vec<f32>), LayerForwardError> {
    let v = VARIANT;
    if v.layer_kind(layer_idx) != super::variants::LayerKind::FullAttn {
        return Err(LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "full_attn_pre_sdpa_capture called on linear-attn layer",
        });
    }

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
    let attn = layer_cache.attn.full().ok_or(
        LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "full_attn weights (batched pre-SDPA)",
        },
    )?;
    let q_dim = v.num_attn_heads * v.head_dim;
    let q_proj_dim = q_dim * 2;
    let kv_dim = v.num_kv_heads * v.head_dim;

    let mv = MatvecPipelines::fetch(metal)?;
    let rms_pipes = RmsNormBf16Pipelines::fetch(metal)?;

    {
        let cmdbuf = metal.queue().new_command_buffer();
        encode_rms_norm_bf16_into(
            cmdbuf,
            &rms_pipes,
            &buffers.input,
            wf_buf.buffer(),
            layer_cache.input_layernorm_w,
            &buffers.sum_sq,
            &buffers.normed,
            v.hidden_dim as u32,
            super::variants::RMS_NORM_EPS,
        );
        let specs = [
            MatvecSpec {
                w_off: attn.q_proj_w,
                s_off: attn.q_proj_s,
                b_off: attn.q_proj_b,
                input: &buffers.normed,
                output: &buffers.q_proj_out,
                out_dim: q_proj_dim as u32,
                in_dim: v.hidden_dim as u32,
                bits: q_bits,
            },
            MatvecSpec {
                w_off: attn.k_proj_w,
                s_off: attn.k_proj_s,
                b_off: attn.k_proj_b,
                input: &buffers.normed,
                output: &buffers.k_out,
                out_dim: kv_dim as u32,
                in_dim: v.hidden_dim as u32,
                bits: k_bits,
            },
            MatvecSpec {
                w_off: attn.v_proj_w,
                s_off: attn.v_proj_s,
                b_off: attn.v_proj_b,
                input: &buffers.normed,
                output: &buffers.v_out,
                out_dim: kv_dim as u32,
                in_dim: v.hidden_dim as u32,
                bits: v_bits,
            },
        ];
        for s in &specs {
            encode_matvec(cmdbuf, &mv, wf_buf, s);
        }
        metal.commit_and_wait_labeled(cmdbuf, "batched_full_attn.cmd1");
    }

    let q_proj_host = read_buffer_to_vec(&buffers.q_proj_out, q_proj_dim);
    let mut k_host = read_buffer_to_vec(&buffers.k_out, kv_dim);
    let v_host = read_buffer_to_vec(&buffers.v_out, kv_dim);

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

    let cache_pos = kv_state.len as usize;
    if cache_pos + 1 > super::variants::MAX_SEQ_LEN {
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

    Ok((q_host, q_gate_host))
}

/// Batched full-attention layer forward for chunked prefill.
///
/// Runs the per-token pre-MoE forward in a loop (steps 1-14 of the
/// Phase B plan), capturing per-token intermediates and the GPU
/// outputs (`h_mid`, `h_post`, `shared_out`) to host stacks. Then
/// builds the joint N×k_active expert-bucket CSR and runs
/// [`super::expert_forward::encode_moe_batched_permute_fuse`] once
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
pub(super) fn batched_full_attn_layer_forward(
    metal: &mut MetalBackend,
    wf: &WeightFile,
    wf_buf: &MtlWeightBuf,
    layer_cache: &LayerWeightCache,
    buffers: &mut LayerForwardBuffers,
    layer_idx: usize,
    start_pos: i32,
    n_tokens: usize,
    k_active: usize,
    expert_files: &ExpertFiles,
    kv_state: &mut KvCache,
    hidden_in: &[f32],
    hidden_out: &mut [f32],
) -> Result<(), LayerForwardError> {
    use super::expert_forward::{encode_moe_batched_permute_fuse, MAX_K};
    use super::metal::MtlBuffer;
    use super::moe_router::build_expert_buckets;

    let v = VARIANT;
    debug_assert!(k_active <= MAX_K);
    debug_assert_eq!(hidden_in.len(), n_tokens * v.hidden_dim);
    debug_assert_eq!(hidden_out.len(), n_tokens * v.hidden_dim);

    let hidden_dim = v.hidden_dim;
    let q_dim = v.num_attn_heads * v.head_dim;
    let kv_dim = v.num_kv_heads * v.head_dim;

    let mut h_mid_stack = vec![0.0f32; n_tokens * hidden_dim];
    let mut h_post_stack = vec![0.0f32; n_tokens * hidden_dim];
    let mut shared_out_stack = vec![0.0f32; n_tokens * hidden_dim];
    let mut all_routing_indices = Vec::with_capacity(n_tokens * k_active);
    let mut all_routing_weights = Vec::with_capacity(n_tokens * k_active);
    let mut shared_gate_scores = Vec::with_capacity(n_tokens);

    // ── Phase 1: per-token pre-SDPA. Capture q + q_gate per token,
    //    append k/v to kv_state. KV state advances by n_tokens. ──
    let mut q_stack = vec![0.0f32; n_tokens * q_dim];
    let mut q_gate_stack = vec![0.0f32; n_tokens * q_dim];
    let kv_start = kv_state.len;
    for t in 0..n_tokens {
        let pos = start_pos + t as i32;
        // SAFETY: shared-storage buffer; pre-SDPA's CMD1 ends in
        // commit_and_wait, so no GPU work is in flight on
        // buffers.input by the next iteration.
        unsafe {
            let dst = buffers.input.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(
                hidden_in[t * hidden_dim..].as_ptr(),
                dst,
                hidden_dim,
            );
        }
        let (q_host, q_gate_host) = full_attn_pre_sdpa_capture(
            metal,
            wf,
            wf_buf,
            layer_cache,
            buffers,
            layer_idx,
            pos,
            kv_state,
        )?;
        q_stack[t * q_dim..(t + 1) * q_dim].copy_from_slice(&q_host);
        q_gate_stack[t * q_dim..(t + 1) * q_dim].copy_from_slice(&q_gate_host);
    }

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
    let state_total = (n_tokens as u32) * (v.num_attn_heads as u32);
    let running_max = MtlBuffer::<f32>::with_len(&device, state_total as usize);
    let running_denom =
        MtlBuffer::<f32>::with_len(&device, state_total as usize);
    let v_partial = MtlBuffer::<f32>::with_len(
        &device,
        (state_total as usize) * v.head_dim,
    );

    let sdpa_pipes = BatchedSdpaPipelines::fetch(metal)?;
    let softmax_scale = 1.0f32 / (v.head_dim as f32).sqrt();
    let heads_per_kv = (v.num_attn_heads / v.num_kv_heads) as u32;
    let queue = metal.queue_clone();
    let cmdbuf = queue.new_command_buffer();
    encode_sdpa_causal_tiled(
        cmdbuf,
        &sdpa_pipes,
        q_buf.buffer(),
        k_gpu.buffer(),
        v_gpu.buffer(),
        attn_out_buf.buffer(),
        running_max.buffer(),
        running_denom.buffer(),
        v_partial.buffer(),
        n_tokens as u32,
        v.num_attn_heads as u32,
        heads_per_kv,
        v.head_dim as u32,
        kv_dim as u32,
        kv_start as u32,
        kv_len_total,
        softmax_scale,
    );
    metal.commit_and_wait_labeled(cmdbuf, "batched_sdpa_causal_tiled");
    // Tiled SDPA output is gate-free (per session-2 Phase 3 design).
    // Apply sigmoid_gate per token on host below.
    let attn_out_stack = attn_out_buf.to_vec();

    // ── Phase 3: per-token post-SDPA. Apply sigmoid gate, stage into
    //    batch_out[6], run post_attention_pre_moe (gpu_attn_args=None),
    //    snapshot intermediates + per-token GPU outputs. ────────────
    for t in 0..n_tokens {
        let q_gate = &q_gate_stack[t * q_dim..(t + 1) * q_dim];
        let attn_raw = &attn_out_stack[t * q_dim..(t + 1) * q_dim];
        let mut attn_with_gate = vec![0.0f32; q_dim];
        for i in 0..q_dim {
            let sg = 1.0f32 / (1.0f32 + (-q_gate[i]).exp());
            attn_with_gate[i] = sg * attn_raw[i];
        }
        // SAFETY: shared-storage; no GPU work in flight on
        // batch_out[6] (Phase 2 committed+waited, Phase 3 hasn't
        // dispatched yet for token t).
        unsafe {
            let dst = buffers.batch_out[6].contents() as *mut f32;
            std::ptr::copy_nonoverlapping(
                attn_with_gate.as_ptr(),
                dst,
                q_dim,
            );
        }
        // Re-stage hidden_in[t] into buffers.input — it's the residual
        // source consumed by post_attention_pre_moe's residual_add.
        // Each Phase 3 iteration overwrites buffers.input for the
        // CURRENT token, so the previous iteration's value is no
        // longer reliable. Always restage.
        unsafe {
            let dst = buffers.input.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(
                hidden_in[t * hidden_dim..].as_ptr(),
                dst,
                hidden_dim,
            );
        }

        // Reproduce the post-SDPA tail of `full_attn_pre_moe_layer_forward`:
        // create a fresh cmdbuf for the o_proj-onward CMD2+3, pass to
        // post_attention_pre_moe with gpu_attn_args=None (reads attn
        // input from batch_out[6]).
        let attn_full = layer_cache.attn.full().ok_or(
            LayerForwardError::MissingTensor {
                layer: layer_idx,
                tensor: "full_attn weights (B2 batched path)",
            },
        )?;
        let o_bits = bits_of(
            wf,
            &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
        );
        let tail_queue = metal.queue_clone();
        let tail_cmdbuf = tail_queue.new_command_buffer();
        let intermediates = post_attention_pre_moe(
            metal,
            tail_cmdbuf,
            wf,
            wf_buf,
            layer_cache,
            buffers,
            layer_idx,
            k_active,
            OProj {
                w_off: attn_full.o_proj_w,
                s_off: attn_full.o_proj_s,
                b_off: attn_full.o_proj_b,
                bits: o_bits,
                in_dim: q_dim as u32,
            },
            /* gpu_attn_args = */ None,
        )?;

        h_mid_stack[t * hidden_dim..(t + 1) * hidden_dim]
            .copy_from_slice(&read_buffer_to_vec(&buffers.h_mid, hidden_dim));
        h_post_stack[t * hidden_dim..(t + 1) * hidden_dim]
            .copy_from_slice(&read_buffer_to_vec(&buffers.normed, hidden_dim));
        shared_out_stack[t * hidden_dim..(t + 1) * hidden_dim]
            .copy_from_slice(&read_buffer_to_vec(
                &buffers.shared_out,
                hidden_dim,
            ));
        all_routing_indices.extend_from_slice(&intermediates.routing_indices);
        all_routing_weights.extend_from_slice(&intermediates.routing_weights);
        shared_gate_scores.push(intermediates.shared_gate_score);
    }

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

    let expert_size = v.expert_size_4bit();
    let mut expert_blobs: Vec<MtlBuffer<u8>> =
        Vec::with_capacity(buckets.expert_ids.len());
    let mut blob_scratch = vec![0u8; expert_size];
    for &expert_id in &buckets.expert_ids {
        expert_files
            .read_expert(layer_idx, expert_id as usize, &mut blob_scratch)?;
        expert_blobs
            .push(MtlBuffer::<u8>::with_data(&device, &blob_scratch));
    }

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
    let queue = metal.queue_clone();
    let cmdbuf = queue.new_command_buffer();
    encode_moe_batched_permute_fuse(
        cmdbuf,
        &matvec,
        &swiglu,
        &bucket_accumulate,
        &expert_blobs,
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
    metal.commit_and_wait_labeled(cmdbuf, "batched_moe_permute_fuse");

    let moe_sum_host = out_sum.to_vec();
    for t in 0..n_tokens {
        let sg = shared_gate_scores[t];
        let sigmoid_sg = 1.0f32 / (1.0f32 + (-sg).exp());
        let h_mid = &h_mid_stack[t * hidden_dim..(t + 1) * hidden_dim];
        let moe = &moe_sum_host[t * hidden_dim..(t + 1) * hidden_dim];
        let shared = &shared_out_stack[t * hidden_dim..(t + 1) * hidden_dim];
        let out = &mut hidden_out[t * hidden_dim..(t + 1) * hidden_dim];
        for i in 0..hidden_dim {
            out[i] = h_mid[i] + moe[i] + sigmoid_sg * shared[i];
        }
    }

    Ok(())
}
