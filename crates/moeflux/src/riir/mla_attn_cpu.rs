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

use super::state::MlaKvCache;
use super::variants::VARIANT;
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
    #[error("MLA forward not yet implemented (Phase C scaffold; next session)")]
    NotImplemented,
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
/// Phase C scaffold returns [`MlaForwardError::NotImplemented`].
/// The kernel implementation arrives in the next session.
#[allow(unused_variables, clippy::too_many_arguments)]
pub fn mla_attn_layer_forward_cpu(
    wf: &WeightFile,
    layer_idx: usize,
    pos: i32,
    hidden: &[f32],
    kv_cache: &mut MlaKvCache,
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
    if hidden.len() != VARIANT.hidden_dim {
        return Err(MlaForwardError::HiddenLen {
            got: hidden.len(),
            expected: VARIANT.hidden_dim,
        });
    }
    if out.len() != VARIANT.hidden_dim {
        return Err(MlaForwardError::OutLen {
            got: out.len(),
            expected: VARIANT.hidden_dim,
        });
    }

    // Phase C scaffold — kernel implementation in the next session.
    // The cargo of work for that session lives in the next-session
    // continuation memo at
    // `~/Projects/drama_llama/.claude/memory/cogito_v2_landing_state.md`.
    Err(MlaForwardError::NotImplemented)
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

    /// Stub returns NotImplemented for MLA variants — placeholder
    /// until the kernel lands. Skipped because we can't construct a
    /// WeightFile here; lives as documentation of intent.
    #[cfg(feature = "model-cogito-v2-671b")]
    #[test]
    #[ignore = "Phase C scaffold; kernel implementation pending"]
    fn cogito_returns_not_implemented_until_kernel_lands() {}
}
