//! Per-layer pre-computed tensor offsets — Phase 4c plumbing.
//!
//! Mirrors the C `LayerWeightCache` (infer.m:3825) but stored as a
//! field on [`crate::riir::RsCtx`] instead of a file-scope global.
//! That fixes the cross-Ctx `layer_cache_built` bug 4b discovered:
//! when the C-side global outlives its source Ctx, downstream Ctx
//! instances read freed pointers. Per-Ctx storage in the Rust port
//! makes that bug class uncompilable.
//!
//! ## Shape (slice 4f-2)
//!
//! Originally a single struct with every linear-attn / full-attn /
//! MoE / norm slot side-by-side as `Option<u64>`. Slice 4f-2 split it
//! into a nested form so attention-kind-specific slots live in a
//! tagged enum [`LayerAttnW`] — Mike's "future-arch openings" note in
//! the RIIR strategy doc, motivated by the eventual DeepSeek-V3 port
//! whose third attention shape (MLA) doesn't fit the linear/full
//! dichotomy. Common slots (input + post-attn norms, MoE router,
//! shared expert) stay flat on [`LayerWeightCache`]; consumers no
//! longer carry the per-slot `require()` ladder that 4c's flat shape
//! forced.

use super::mtl_weight_buf::{MtlWeightBuf, MtlWeightBufError};
use super::variants::{LayerKind, VARIANT};
use super::weight_file::WeightFile;

/// Tensor offsets specific to a linear-attention (GatedDeltaNet)
/// layer. Every slot is required for layers of this kind; missing
/// tensors fail at [`LayerWeightCache::build`] time.
#[derive(Debug, Clone)]
pub struct LinearAttnW {
    pub qkv_w: u64,
    pub qkv_s: u64,
    pub qkv_b: u64,
    pub z_w: u64,
    pub z_s: u64,
    pub z_b: u64,
    pub alpha_w: u64,
    pub alpha_s: u64,
    pub alpha_b: u64,
    pub beta_w: u64,
    pub beta_s: u64,
    pub beta_b: u64,
    pub conv1d_w: u64,
    pub a_log: u64,
    pub dt_bias: u64,
    pub gated_norm_w: u64,
    pub o_proj_w: u64,
    pub o_proj_s: u64,
    pub o_proj_b: u64,
}

/// Tensor offsets specific to a standard full-attention (SDPA + per-
/// head Q/K norms) layer.
#[derive(Debug, Clone)]
pub struct FullAttnW {
    pub q_proj_w: u64,
    pub q_proj_s: u64,
    pub q_proj_b: u64,
    pub k_proj_w: u64,
    pub k_proj_s: u64,
    pub k_proj_b: u64,
    pub v_proj_w: u64,
    pub v_proj_s: u64,
    pub v_proj_b: u64,
    pub q_norm_w: u64,
    pub k_norm_w: u64,
    pub o_proj_w: u64,
    pub o_proj_s: u64,
    pub o_proj_b: u64,
}

/// Attention-kind-specific weight offsets for one layer. The variant
/// matches [`LayerKind`] for the layer's index in the active variant.
#[derive(Debug, Clone)]
pub enum LayerAttnW {
    LinearAttn(LinearAttnW),
    FullAttn(FullAttnW),
}

impl LayerAttnW {
    /// Returns the inner [`LinearAttnW`] if this is a linear-attn
    /// layer, otherwise `None`. Consumers in
    /// `linear_attn_layer_forward` use this to fail-fast with a
    /// clearer error than a `match` arm could give.
    pub fn linear(&self) -> Option<&LinearAttnW> {
        match self {
            Self::LinearAttn(la) => Some(la),
            Self::FullAttn(_) => None,
        }
    }

    /// Returns the inner [`FullAttnW`] if this is a full-attn layer.
    pub fn full(&self) -> Option<&FullAttnW> {
        match self {
            Self::FullAttn(fa) => Some(fa),
            Self::LinearAttn(_) => None,
        }
    }
}

/// MoE routing gate weights — produces the per-expert logits that the
/// CPU router (`moe_router_cpu`) reads. `bias` is `None` for variants
/// whose manifest folds the per-output bias into `gate.biases` (the
/// per-group dequant bias) rather than carrying a separate
/// `gate.bias` tensor — true for every variant we ship today.
#[derive(Debug, Clone)]
pub struct GateW {
    pub w: u64,
    pub s: u64,
    pub b: u64,
    pub bias: Option<u64>,
}

/// Shared-expert FFN + scoring gate — runs alongside the routed
/// experts every layer. `seg_*` is the scalar gate that produces the
/// shared expert's mixing weight; `gate/up/down` are its FFN matvecs.
#[derive(Debug, Clone)]
pub struct SharedExpertW {
    pub seg_w: u64,
    pub seg_s: u64,
    pub seg_b: u64,
    pub gate_w: u64,
    pub gate_s: u64,
    pub gate_b: u64,
    pub up_w: u64,
    pub up_s: u64,
    pub up_b: u64,
    pub down_w: u64,
    pub down_s: u64,
    pub down_b: u64,
}

/// Pre-computed tensor byte offsets within an [`MtlWeightBuf`] for
/// one layer of the model.
#[derive(Debug, Clone)]
pub struct LayerWeightCache {
    pub input_layernorm_w: u64,
    pub post_attention_layernorm_w: u64,
    pub attn: LayerAttnW,
    pub gate: GateW,
    pub shared: SharedExpertW,
}

impl LayerWeightCache {
    /// Resolve every tensor offset for layer `layer_idx`. Returns a
    /// fully-populated cache; tensors that the active layer kind
    /// doesn't carry are simply absent from the matching enum
    /// variant. Tensor names match the manifest exported by
    /// `extract_weights.py`. Errors with
    /// [`MtlWeightBufError::MissingTensor`] if any required slot for
    /// the layer kind is missing.
    pub fn build(
        layer_idx: usize,
        wf: &WeightFile,
        wf_buf: &MtlWeightBuf,
    ) -> Result<Self, MtlWeightBufError> {
        let need = |name: String| -> Result<u64, MtlWeightBufError> {
            wf_buf
                .tensor_offset(wf, &name)?
                .ok_or(MtlWeightBufError::MissingTensor { name })
        };

        let input_layernorm_w =
            need(format!("model.layers.{layer_idx}.input_layernorm.weight"))?;
        let post_attention_layernorm_w = need(format!(
            "model.layers.{layer_idx}.post_attention_layernorm.weight"
        ))?;

        let attn = match VARIANT.layer_kind(layer_idx) {
            LayerKind::LinearAttn => {
                let p = |suffix: &str| {
                    format!("model.layers.{layer_idx}.linear_attn.{suffix}")
                };
                LayerAttnW::LinearAttn(LinearAttnW {
                    qkv_w: need(p("in_proj_qkv.weight"))?,
                    qkv_s: need(p("in_proj_qkv.scales"))?,
                    qkv_b: need(p("in_proj_qkv.biases"))?,
                    z_w: need(p("in_proj_z.weight"))?,
                    z_s: need(p("in_proj_z.scales"))?,
                    z_b: need(p("in_proj_z.biases"))?,
                    alpha_w: need(p("in_proj_a.weight"))?,
                    alpha_s: need(p("in_proj_a.scales"))?,
                    alpha_b: need(p("in_proj_a.biases"))?,
                    beta_w: need(p("in_proj_b.weight"))?,
                    beta_s: need(p("in_proj_b.scales"))?,
                    beta_b: need(p("in_proj_b.biases"))?,
                    conv1d_w: need(p("conv1d.weight"))?,
                    a_log: need(p("A_log"))?,
                    dt_bias: need(p("dt_bias"))?,
                    gated_norm_w: need(p("norm.weight"))?,
                    o_proj_w: need(p("out_proj.weight"))?,
                    o_proj_s: need(p("out_proj.scales"))?,
                    o_proj_b: need(p("out_proj.biases"))?,
                })
            }
            LayerKind::FullAttn => {
                let s = |suffix: &str| {
                    format!("model.layers.{layer_idx}.self_attn.{suffix}")
                };
                LayerAttnW::FullAttn(FullAttnW {
                    q_proj_w: need(s("q_proj.weight"))?,
                    q_proj_s: need(s("q_proj.scales"))?,
                    q_proj_b: need(s("q_proj.biases"))?,
                    k_proj_w: need(s("k_proj.weight"))?,
                    k_proj_s: need(s("k_proj.scales"))?,
                    k_proj_b: need(s("k_proj.biases"))?,
                    v_proj_w: need(s("v_proj.weight"))?,
                    v_proj_s: need(s("v_proj.scales"))?,
                    v_proj_b: need(s("v_proj.biases"))?,
                    q_norm_w: need(s("q_norm.weight"))?,
                    k_norm_w: need(s("k_norm.weight"))?,
                    o_proj_w: need(s("o_proj.weight"))?,
                    o_proj_s: need(s("o_proj.scales"))?,
                    o_proj_b: need(s("o_proj.biases"))?,
                })
            }
        };

        let m =
            |suffix: &str| format!("model.layers.{layer_idx}.mlp.{suffix}");
        let gate = GateW {
            w: need(m("gate.weight"))?,
            s: need(m("gate.scales"))?,
            b: need(m("gate.biases"))?,
            // No separate `gate.bias` in the manifest for the
            // qwen3_5_moe family — `gate.biases` (above) carries the
            // per-group dequant bias.
            bias: None,
        };
        let shared = SharedExpertW {
            seg_w: need(m("shared_expert_gate.weight"))?,
            seg_s: need(m("shared_expert_gate.scales"))?,
            seg_b: need(m("shared_expert_gate.biases"))?,
            gate_w: need(m("shared_expert.gate_proj.weight"))?,
            gate_s: need(m("shared_expert.gate_proj.scales"))?,
            gate_b: need(m("shared_expert.gate_proj.biases"))?,
            up_w: need(m("shared_expert.up_proj.weight"))?,
            up_s: need(m("shared_expert.up_proj.scales"))?,
            up_b: need(m("shared_expert.up_proj.biases"))?,
            down_w: need(m("shared_expert.down_proj.weight"))?,
            down_s: need(m("shared_expert.down_proj.scales"))?,
            down_b: need(m("shared_expert.down_proj.biases"))?,
        };

        Ok(Self {
            input_layernorm_w,
            post_attention_layernorm_w,
            attn,
            gate,
            shared,
        })
    }

    /// Build a cache for every layer in the active variant.
    pub fn build_all(
        wf: &WeightFile,
        wf_buf: &MtlWeightBuf,
    ) -> Result<Vec<Self>, MtlWeightBufError> {
        (0..VARIANT.num_layers)
            .map(|i| Self::build(i, wf, wf_buf))
            .collect()
    }
}
