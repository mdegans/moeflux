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
use super::gpu_matvec::{encode_matvec, MatvecPipelines, MatvecSpec};
use super::gpu_norm::{encode_rms_norm_bf16_into, RmsNormBf16Pipelines};
use super::layer_weight_cache::LayerWeightCache;
use super::linear_attn_forward::{
    bits_of, post_attention_tail, read_buffer_to_vec,
    LayerForwardBuffers, LayerForwardError, OProj,
};
use super::metal::MetalBackend;
use super::mtl_weight_buf::MtlWeightBuf;
use super::rms_norm::rms_norm_per_head_cpu;
use super::rope::apply_rotary_emb;
use super::sdpa::sdpa_cpu;
use super::state::KvCache;
use super::variants::VARIANT;
use super::weight_file::WeightFile;

/// Run one full-attention layer's forward pass.
///
/// Pre: `buffers.input` holds the input hidden state (HIDDEN_DIM
/// floats). Post: `buffers.input` holds the output hidden state.
/// `kv_state.len` advances by 1 and the new K/V row is appended at
/// the previous-`len` position.
///
/// `pos` is the absolute KV position (matches the C side's `pos`
/// argument to `apply_rotary_emb`). Today the dump-hook test calls
/// this at `pos=0` against a `memory_clear`-reset KV cache, so
/// `kv_state.len` starts at 0 and `pos` matches `kv_state.len`.
/// Once 4f wires `eval_prompt`, callers will pre-set `pos` to the
/// absolute sequence position.
#[allow(clippy::too_many_arguments)]
pub fn full_attn_layer_forward(
    metal: &mut MetalBackend,
    wf: &WeightFile,
    wf_buf: &MtlWeightBuf,
    layer_cache: &LayerWeightCache,
    buffers: &mut LayerForwardBuffers,
    moe: &mut MoeBuffers,
    deferred: &mut Option<super::deferred::DeferredState>,
    layer_idx: usize,
    pos: i32,
    k_active: usize,
    expert_files: &ExpertFiles,
    pool: &rayon::ThreadPool,
    prefetch: &mut super::PrefetchState,
    kv_state: &mut KvCache,
    gpu_combine: bool,
) -> Result<(), LayerForwardError> {
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

        cmdbuf.commit();
        cmdbuf.wait_until_completed();
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

    // ── KV append into host-side cache ───────────────────────────
    // FIXME(riir): the C path also mirrors k/v into the GPU-resident
    // KV buffers (`buf_kv_k[fa_idx]`, `buf_kv_v[fa_idx]` at
    // infer.m:4796–4802). Those feed the GPU-attention fast path that
    // engages once `kv_len >= 32`. Skipped here because the dump-hook
    // diff runs at `pos=0` (kv_len=1 after this append) and never
    // exercises the GPU path. Wire when 4e/4f integration lands.
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

    // ── CPU SDPA ─────────────────────────────────────────────────
    // Slice the caches to the occupied prefix; sdpa_cpu validates
    // length against `kv_len * num_kv_heads * head_dim`.
    let kv_len = kv_state.len;
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

    // ── Stage attn_out into batch_out[6] for o_proj input ────────
    {
        let dst = buffers.batch_out[6].contents() as *mut f32;
        // SAFETY: shared storage; no GPU work in flight (CMD1 above
        // committed and waited).
        unsafe {
            std::ptr::copy_nonoverlapping(
                attn_out.as_ptr(),
                dst,
                q_dim,
            );
        }
        // batch_out[6] was sized to `max(linear_total_value, q_dim)`
        // floats at allocation; assert the expected slot is large
        // enough so a future Variant change can't silently overflow.
        debug_assert!(
            buffers.batch_out[6].length() as usize
                >= q_dim * std::mem::size_of::<f32>(),
            "batch_out[6] sized {} bytes, need {} for full-attn o_proj input",
            buffers.batch_out[6].length() as NSUInteger,
            q_dim * std::mem::size_of::<f32>(),
        );
    }

    // ── Hand off to the shared post-attention tail ───────────────
    // The tail leaves an in-flight K-expert dispatch in `*deferred`;
    // caller drains.
    post_attention_tail(
        metal,
        wf,
        wf_buf,
        layer_cache,
        buffers,
        moe,
        deferred,
        layer_idx,
        k_active,
        expert_files,
        pool,
        prefetch,
        OProj {
            w_off: o_w,
            s_off: o_s,
            b_off: o_b,
            bits: o_bits,
            in_dim: q_dim as u32,
        },
        gpu_combine,
    )
}
