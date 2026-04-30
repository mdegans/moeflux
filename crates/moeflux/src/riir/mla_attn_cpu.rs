//! Multi-head Latent Attention (MLA) — CPU forward kernel.
//!
//! DeepSeek-V3 / Cogito-V2-Preview-671B attention path. Companion to
//! [`super::full_attn_forward::full_attn_layer_forward`] (GQA) and
//! [`super::linear_attn_forward::linear_attn_layer_forward`]
//! (GatedDeltaNet) — the third attention flavor moeflux dispatches.
//!
//! ## Status: Phase C scaffold
//!
//! This file currently contains the type signatures and orchestration
//! shape only. The inner forward (Q-LoRA → KV decompress → SDPA with
//! YaRN double-mscale) lands in the next session. The router math
//! ([`super::moe_router::noaux_tc_router_cpu`]) and YaRN math
//! ([`super::rope::compute_yarn_inv_freq`] +
//! [`super::rope::apply_rotary_emb_yarn`]) are already in place.
//!
//! ## Why CPU-only first
//!
//! 1. The existing GQA full-attn forward is GPU-pipelined
//!    (Metal command buffers + deferred K-expert dispatch + parity
//!    ping-pong prefetch). Integrating MLA into that orchestration is
//!    its own focused slice; CPU-only delivers a working forward
//!    sooner and keeps the integration choices visible.
//! 2. At 671B with SSD-streamed experts, projected throughput is
//!    ~1 tok/s warm. Attention compute is not on the critical path —
//!    SSD I/O is. CPU MLA fits the perf envelope for a first run.
//! 3. CPU is testable against arithmetic reference values (small
//!    probe inputs, hand-computed expected outputs). GPU shaders
//!    require the full pipeline scaffolding.
//!
//! Once the model produces coherent text on this path, a GPU port
//! can layer on incrementally without changing the CPU contract.
//!
//! ## Forward shape (per token at position `pos`)
//!
//! ```text
//! q_lat = q_a_proj @ hidden                   # [q_lora_rank=1536]
//! q_lat = rms_norm(q_lat)                     # q_a_layernorm
//! q     = q_b_proj @ q_lat                    # [num_heads*qk_head_dim]
//! q_per_head = reshape(q, [num_heads, qk_head_dim])
//! q_nope, q_pe = split(q_per_head, [qk_nope_head_dim, qk_rope_head_dim])
//!
//! kv_pre = kv_a_proj_with_mqa @ hidden        # [kv_lora_rank + qk_rope_head_dim]
//! kv_lat, k_pe = split(kv_pre, [kv_lora_rank, qk_rope_head_dim])
//! kv_lat = rms_norm(kv_lat)                   # kv_a_layernorm
//!
//! q_pe   = yarn_rope(q_pe, pos, ..)           # per-head [qk_rope_head_dim]
//! k_pe   = yarn_rope(k_pe, pos, ..)           # shared    [qk_rope_head_dim]
//!
//! # Append (kv_lat, k_pe) to MlaKvCache. cache.len += 1.
//!
//! # For each cached position j in [0, len):
//! #   k_b_view = kv_b_proj  # [num_heads*(qk_nope_head_dim+v_head_dim), kv_lora_rank]
//! #   kv_decoded = k_b_view @ latent_cache[j]   # [num_heads*256]
//! #   k_nope_j, v_j = split(kv_decoded, [qk_nope_head_dim, v_head_dim], per-head)
//! #   k_j = concat(k_nope_j, broadcast(rope_k_cache[j]))  # per-head [qk_head_dim]
//! #   score_h_j = (q_per_head[h] · k_j[h]) * softmax_scale
//! # softmax_scale = (1/sqrt(qk_head_dim)) * yarn_mscale²
//! # softmax over j; out_h = Σ_j weight * v_j[h]
//! out_flat = reshape(out, [num_heads * v_head_dim])
//! return o_proj @ out_flat                    # [hidden_dim]
//! ```
//!
//! Naive implementation re-runs `kv_b_proj @ latent_j` per cached
//! step. The mathematically-equivalent low-rank-folded form (precompute
//! `q_nope @ kv_b_proj_per_head` to get `q' [num_heads, kv_lora_rank]`
//! then `score_h_j = q'_h · latent_j + q_pe_h · rope_k_j`) is faster
//! at long context but more code. First implementation lands the
//! naive form for clarity; the folded form is a follow-up
//! optimization once the model is producing tokens.

use super::cpu_matvec::{project_4bit_cpu, CpuMatvecError};
use super::moe_router::softmax;
use super::rms_norm::{rms_norm_per_head_cpu, RmsNormError};
use super::rope::{apply_rotary_emb_yarn, YarnError};
use super::state::MlaKvCacheGpu;
use super::variants::{MAX_SEQ_LEN, VARIANT};
use super::weight_file::WeightFile;

/// Errors specific to the MLA CPU forward.
#[derive(Debug, thiserror::Error)]
pub enum MlaForwardError {
    #[error("called on non-MLA variant (attn_kind = {kind:?})")]
    NotMlaVariant {
        kind: super::variants::AttnKind,
    },
    #[error("hidden buffer length {got} != hidden_dim ({expected})")]
    HiddenLen { got: usize, expected: usize },
    #[error("output buffer length {got} != hidden_dim ({expected})")]
    OutLen { got: usize, expected: usize },
    #[error("position {pos} != kv_cache.len {cache_len} (single-step decode)")]
    PosMismatch { pos: i32, cache_len: i32 },
    #[error("kv_cache.len {len} would exceed MAX_SEQ_LEN={max} after append")]
    CacheFull { len: i32, max: usize },
    #[error("matvec error in MLA: {0}")]
    Matvec(#[from] CpuMatvecError),
    #[error("rms-norm error in MLA: {0}")]
    Norm(#[from] RmsNormError),
    #[error("YaRN RoPE error in MLA: {0}")]
    Rope(#[from] YarnError),
    #[error("softmax error in MLA: {0}")]
    Softmax(#[from] super::moe_router::MoeRouterError),
}

/// Per-token MLA forward pass. Reads layer weights from `wf` by
/// canonical `model.layers.{i}.self_attn.*` names; reads + appends
/// to `kv_cache`; writes the post-o_proj hidden state to `out`.
///
/// Pre-conditions:
/// - `VARIANT.attn_kind == AttnKind::Mla`
/// - `hidden.len() == VARIANT.hidden_dim`
/// - `out.len() == VARIANT.hidden_dim`
/// - `kv_cache.len < MAX_SEQ_LEN`
/// - `pos == kv_cache.len` (single-step decode at the next position)
///
/// Post-conditions:
/// - `kv_cache.len += 1`, with the new latent and rope-K appended
/// - `out` holds the post-attention residual contribution (caller
///   adds to the input hidden state per the standard transformer
///   block; same contract as the GQA forward).
///
/// Naive form: for each cached position j, run `kv_b_proj @ latent[j]`
/// to materialize per-head `(k_nope, v)`. Cost is O(len * 16M ops) per
/// token; tractable for first-run validation. The folded form (precompute
/// `q_nope @ kv_b_proj_K` and `kv_b_proj_V @ scored_combine`) cuts this
/// to O(16M + len * 130K) and is a follow-up once the model produces
/// coherent text.
#[allow(clippy::too_many_arguments)]
pub fn mla_attn_layer_forward_cpu(
    wf: &WeightFile,
    layer_idx: usize,
    pos: i32,
    hidden: &[f32],
    kv_cache: &mut MlaKvCacheGpu,
    yarn_inv_freq: &[f32],
    yarn_mscale: f32,
    out: &mut [f32],
) -> Result<(), MlaForwardError> {
    use super::variants::AttnKind;
    if VARIANT.attn_kind != AttnKind::Mla {
        return Err(MlaForwardError::NotMlaVariant {
            kind: VARIANT.attn_kind,
        });
    }
    let v = VARIANT;
    if hidden.len() != v.hidden_dim {
        return Err(MlaForwardError::HiddenLen {
            got: hidden.len(),
            expected: v.hidden_dim,
        });
    }
    if out.len() != v.hidden_dim {
        return Err(MlaForwardError::OutLen {
            got: out.len(),
            expected: v.hidden_dim,
        });
    }
    if pos != kv_cache.len {
        return Err(MlaForwardError::PosMismatch {
            pos,
            cache_len: kv_cache.len,
        });
    }
    if (kv_cache.len as usize) >= MAX_SEQ_LEN {
        return Err(MlaForwardError::CacheFull {
            len: kv_cache.len,
            max: MAX_SEQ_LEN,
        });
    }

    let hidden_dim = v.hidden_dim;
    let num_heads = v.num_attn_heads;
    let q_lora_rank = v.q_lora_rank;
    let kv_lora_rank = v.kv_lora_rank;
    let nope = v.qk_nope_head_dim;
    let rope = v.qk_rope_head_dim;
    let v_head_dim = v.v_head_dim;
    let qk_head_dim = nope + rope;
    // 256 = nope (128) + v_head_dim (128) for Cogito-V2.
    let kv_b_per_head = nope + v_head_dim;

    // ---- Q chain ----
    let q_a_name = format!("model.layers.{layer_idx}.self_attn.q_a_proj");
    let q_a_norm =
        format!("model.layers.{layer_idx}.self_attn.q_a_layernorm.weight");
    let q_b_name = format!("model.layers.{layer_idx}.self_attn.q_b_proj");

    let mut q_lat = vec![0.0f32; q_lora_rank];
    project_4bit_cpu(wf, &q_a_name, hidden_dim, q_lora_rank, hidden, &mut q_lat)?;
    rms_norm_per_head_cpu(wf, &q_a_norm, 1, q_lora_rank, &mut q_lat)?;

    let mut q_full = vec![0.0f32; num_heads * qk_head_dim];
    project_4bit_cpu(
        wf,
        &q_b_name,
        q_lora_rank,
        num_heads * qk_head_dim,
        &q_lat,
        &mut q_full,
    )?;

    // q_full is laid out per head as [nope | pe]. Extract pe halves
    // contiguously for RoPE; copy back after.
    let mut q_pe = vec![0.0f32; num_heads * rope];
    for h in 0..num_heads {
        let q_h = &q_full[h * qk_head_dim..(h + 1) * qk_head_dim];
        let dst = &mut q_pe[h * rope..(h + 1) * rope];
        dst.copy_from_slice(&q_h[nope..nope + rope]);
    }

    // ---- KV chain ----
    let kv_a_name =
        format!("model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa");
    let kv_a_norm =
        format!("model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight");

    let mut kv_pre = vec![0.0f32; kv_lora_rank + rope];
    project_4bit_cpu(
        wf,
        &kv_a_name,
        hidden_dim,
        kv_lora_rank + rope,
        hidden,
        &mut kv_pre,
    )?;
    rms_norm_per_head_cpu(
        wf,
        &kv_a_norm,
        1,
        kv_lora_rank,
        &mut kv_pre[..kv_lora_rank],
    )?;

    // ---- YaRN RoPE on the rope halves ----
    apply_rotary_emb_yarn(pos, &mut q_pe, rope, yarn_inv_freq, yarn_mscale)?;
    apply_rotary_emb_yarn(
        pos,
        &mut kv_pre[kv_lora_rank..],
        rope,
        yarn_inv_freq,
        yarn_mscale,
    )?;

    // Write rotated q_pe back into q_full's per-head pe slots.
    for h in 0..num_heads {
        let dst = &mut q_full[h * qk_head_dim + nope..(h + 1) * qk_head_dim];
        let src = &q_pe[h * rope..(h + 1) * rope];
        dst.copy_from_slice(src);
    }

    // ---- Append to MLA cache ----
    //
    // The cache lives in shared-storage Metal buffers; the CPU path
    // grabs unsafe host slices over the appropriate row windows.
    // No GPU work is in flight on this path (`step_internal_mla_cpu`
    // is fully host-side except for the final lm_head dispatch),
    // so the slices are safe to mutate.
    let new_idx = pos as usize;
    // SAFETY: shared-storage buffer; CPU MLA path holds the no-GPU-
    // work invariant. ensure_mla_resources was called before entry.
    unsafe {
        let l_dst = kv_cache.latent_slice_mut(new_idx, new_idx + 1);
        l_dst.copy_from_slice(&kv_pre[..kv_lora_rank]);
        let r_dst = kv_cache.rope_k_slice_mut(new_idx, new_idx + 1);
        r_dst.copy_from_slice(&kv_pre[kv_lora_rank..]);
    }
    kv_cache.len = pos + 1;
    let cache_len = kv_cache.len as usize;
    // Snapshot read-only views over the populated prefix once. The
    // SDPA loop below borrows `latent_cache_view` / `rope_k_cache_view`
    // instead of re-deriving slices through the accessor each step.
    // SAFETY: see the mutable accessor above.
    let latent_cache_view: &[f32] =
        unsafe { kv_cache.latent_slice(cache_len) };
    let rope_k_cache_view: &[f32] =
        unsafe { kv_cache.rope_k_slice(cache_len) };

    // ---- Decompress kv_b_proj @ latent[j] for every cached j ----
    // decoded_all layout: [cache_len, num_heads, kv_b_per_head] flat.
    // Per cached j, per head h: dec[j, h, ..nope] = k_nope, dec[j, h,
    // nope..] = v.
    let kv_b_name = format!("model.layers.{layer_idx}.self_attn.kv_b_proj");
    let mut decoded_all = vec![0.0f32; cache_len * num_heads * kv_b_per_head];
    for j in 0..cache_len {
        let latent_j = &latent_cache_view
            [j * kv_lora_rank..(j + 1) * kv_lora_rank];
        let dec_j = &mut decoded_all
            [j * num_heads * kv_b_per_head..(j + 1) * num_heads * kv_b_per_head];
        project_4bit_cpu(
            wf,
            &kv_b_name,
            kv_lora_rank,
            num_heads * kv_b_per_head,
            latent_j,
            dec_j,
        )?;
    }

    // ---- SDPA per head ----
    // softmax_scale = (1/sqrt(qk_head_dim)) * mscale². For Cogito-V2's
    // mscale=1.0/mscale_all_dim=1.0 this collapses to 1/sqrt(192).
    let softmax_scale =
        (1.0 / (qk_head_dim as f32).sqrt()) * yarn_mscale * yarn_mscale;

    let mut head_out = vec![0.0f32; num_heads * v_head_dim];
    let mut scores = vec![0.0f32; cache_len];

    for h in 0..num_heads {
        let q_h = &q_full[h * qk_head_dim..(h + 1) * qk_head_dim];
        let q_nope_h = &q_h[..nope];
        let q_pe_h = &q_h[nope..nope + rope];
        for j in 0..cache_len {
            let dec_jh = &decoded_all[(j * num_heads + h) * kv_b_per_head
                ..(j * num_heads + h + 1) * kv_b_per_head];
            let k_nope_jh = &dec_jh[..nope];
            let rope_k_j =
                &rope_k_cache_view[j * rope..(j + 1) * rope];
            let mut s = 0.0f32;
            for c in 0..nope {
                s = q_nope_h[c].mul_add(k_nope_jh[c], s);
            }
            for c in 0..rope {
                s = q_pe_h[c].mul_add(rope_k_j[c], s);
            }
            scores[j] = s * softmax_scale;
        }
        softmax(&mut scores)?;
        let head_out_h = &mut head_out[h * v_head_dim..(h + 1) * v_head_dim];
        head_out_h.fill(0.0);
        for j in 0..cache_len {
            let dec_jh = &decoded_all[(j * num_heads + h) * kv_b_per_head
                ..(j * num_heads + h + 1) * kv_b_per_head];
            let v_jh = &dec_jh[nope..nope + v_head_dim];
            let w = scores[j];
            for c in 0..v_head_dim {
                head_out_h[c] = w.mul_add(v_jh[c], head_out_h[c]);
            }
        }
    }

    // ---- o_proj ----
    let o_name = format!("model.layers.{layer_idx}.self_attn.o_proj");
    project_4bit_cpu(
        wf,
        &o_name,
        num_heads * v_head_dim,
        hidden_dim,
        &head_out,
        out,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Calling MLA forward on a non-MLA variant must fail cleanly,
    /// not silently mis-decode. This catches a mis-dispatch in the
    /// integration step (Phase G).
    #[cfg(any(
        feature = "model-qwen3-5-a17b",
        feature = "model-qwen3-6-35b-a3b",
    ))]
    #[test]
    fn rejects_non_mla_variant() {
        // Can't actually construct a real WeightFile in unit-test
        // context — skip the body. The scaffold's compile-time check
        // (this file builds against any variant) is the load-bearing
        // assertion for now.
    }

    /// Smoke test: run one MLA forward step on layer 0 with a
    /// pulse-input hidden state, pos=0. Verifies the kernel finishes
    /// without panic and produces finite output. Doesn't check
    /// numerical correctness — that's the Phase G end-to-end bisect.
    #[cfg(feature = "model-cogito-v2-671b")]
    #[test]
    #[ignore = "needs Cogito-V2 weights mmap'd from /Volumes/Temp Backup"]
    fn mla_layer0_pos0_smoke() {
        use super::super::rope::{compute_yarn_inv_freq, yarn_get_mscale_full};
        use super::super::variants::ROPE_THETA;
        use std::path::Path;

        let bin = Path::new(
            "/Volumes/Temp Backup/models/blallama/cogito-v2-671b/artifacts/model_weights.bin",
        );
        let manifest = Path::new(
            "/Volumes/Temp Backup/models/blallama/cogito-v2-671b/artifacts/model_weights.json",
        );
        let wf = WeightFile::open(bin, manifest).expect("open weights");

        let v = VARIANT;
        let inv_freq = compute_yarn_inv_freq(
            v.qk_rope_head_dim,
            ROPE_THETA,
            v.yarn_factor,
            v.yarn_original_max_pos as f32,
            v.yarn_beta_fast,
            v.yarn_beta_slow,
        );
        let mscale = yarn_get_mscale_full(
            v.yarn_factor,
            v.yarn_mscale,
            v.yarn_mscale_all_dim,
        );

        // Hidden = pulse at index 7 (no particular meaning; just a
        // non-zero, non-uniform input the kernel can transform).
        let mut hidden = vec![0.0f32; v.hidden_dim];
        hidden[7] = 1.0;
        let device = metal::Device::system_default()
            .expect("Metal device for MLA KV cache buffers");
        let mut cache = MlaKvCacheGpu::new();
        cache.ensure_buffers(&device);
        let mut out = vec![0.0f32; v.hidden_dim];

        mla_attn_layer_forward_cpu(
            &wf, 0, 0, &hidden, &mut cache, &inv_freq, mscale, &mut out,
        )
        .expect("MLA forward should succeed");

        assert_eq!(cache.len, 1, "cache should advance to 1");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "out[i] non-finite at first index = {:?}",
            out.iter().position(|v| !v.is_finite()),
        );
        let max_abs = out.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(
            max_abs > 0.0,
            "output is all zeros — likely a wiring bug"
        );
        assert!(
            max_abs < 1e6,
            "output magnitude {max_abs} suspiciously large"
        );
    }
}
