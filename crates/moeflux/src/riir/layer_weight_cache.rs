//! Per-layer pre-computed tensor offsets — Phase 4c plumbing.
//!
//! Mirrors the C `LayerWeightCache` (infer.m:3825) but stored as a
//! field on [`crate::riir::RsCtx`] instead of a file-scope global.
//! That fixes the cross-Ctx `layer_cache_built` bug 4b discovered:
//! when the C-side global outlives its source Ctx, downstream Ctx
//! instances read freed pointers. Per-Ctx storage in the Rust port
//! makes that bug class uncompilable.
//!
//! For the linear-attn forward (4c) we need the byte offsets of
//! every tensor a single layer's pipeline reads. The struct holds
//! `Option<u64>` so layers that don't carry a particular tensor
//! (e.g., full-attn layers don't have linear-attn-specific weights)
//! leave those slots `None`. The cache is built once per layer at
//! [`LayerWeightCache::build`] and stays immutable thereafter.

use super::mtl_weight_buf::{MtlWeightBuf, MtlWeightBufError};
use super::variants::VARIANT;
use super::weight_file::WeightFile;

/// Pre-computed tensor byte offsets within an [`MtlWeightBuf`] for
/// one layer of the model. Linear-attn-specific fields are `Some`
/// for linear-attn layers and `None` for full-attn layers, and vice
/// versa for the attention-projection fields. Norm + MoE
/// fields are populated for every layer.
#[derive(Debug, Default, Clone)]
pub struct LayerWeightCache {
    // --- Layer-wide norms (every layer) -------------------------
    pub input_layernorm_w: Option<u64>,
    pub post_attention_layernorm_w: Option<u64>,

    // --- Linear-attn (`!is_full` layers) ------------------------
    pub qkv_w: Option<u64>,
    pub qkv_s: Option<u64>,
    pub qkv_b: Option<u64>,
    pub z_w: Option<u64>,
    pub z_s: Option<u64>,
    pub z_b: Option<u64>,
    pub alpha_w: Option<u64>,
    pub alpha_s: Option<u64>,
    pub alpha_b: Option<u64>,
    pub beta_w: Option<u64>,
    pub beta_s: Option<u64>,
    pub beta_b: Option<u64>,
    pub conv1d_w: Option<u64>,
    pub a_log: Option<u64>,
    pub dt_bias: Option<u64>,
    pub gated_norm_w: Option<u64>,
    pub linear_o_proj_w: Option<u64>,
    pub linear_o_proj_s: Option<u64>,
    pub linear_o_proj_b: Option<u64>,

    // --- Full-attn (`is_full` layers) ---------------------------
    // 4d will populate these. Listed here so the struct shape stays
    // stable across 4c → 4d.
    pub q_proj_w: Option<u64>,
    pub q_proj_s: Option<u64>,
    pub q_proj_b: Option<u64>,
    pub k_proj_w: Option<u64>,
    pub k_proj_s: Option<u64>,
    pub k_proj_b: Option<u64>,
    pub v_proj_w: Option<u64>,
    pub v_proj_s: Option<u64>,
    pub v_proj_b: Option<u64>,
    pub q_norm_w: Option<u64>,
    pub k_norm_w: Option<u64>,
    pub full_o_proj_w: Option<u64>,
    pub full_o_proj_s: Option<u64>,
    pub full_o_proj_b: Option<u64>,

    // --- MoE router + shared (every layer) ----------------------
    /// `mlp.gate.weight/scales/biases` — routing gate logits matvec.
    pub gate_w: Option<u64>,
    pub gate_s: Option<u64>,
    pub gate_b: Option<u64>,
    pub gate_bias: Option<u64>,
    /// `mlp.shared_expert_gate.weight/scales/biases` — scoring gate
    /// for the shared-expert mixing weight (`seg_*` in the C side,
    /// infer.m:2879).
    pub seg_w: Option<u64>,
    pub seg_s: Option<u64>,
    pub seg_b: Option<u64>,
    /// `mlp.shared_experts.gate_proj.*` — shared expert FFN gate.
    pub shared_gate_w: Option<u64>,
    pub shared_gate_s: Option<u64>,
    pub shared_gate_b: Option<u64>,
    /// `mlp.shared_experts.up_proj.*` — shared expert FFN up.
    pub shared_up_w: Option<u64>,
    pub shared_up_s: Option<u64>,
    pub shared_up_b: Option<u64>,
    /// `mlp.shared_experts.down_proj.*` — shared expert FFN down.
    pub shared_down_w: Option<u64>,
    pub shared_down_s: Option<u64>,
    pub shared_down_b: Option<u64>,
}

impl LayerWeightCache {
    /// Resolve every tensor offset for layer `layer_idx`. Returns a
    /// fully-populated cache for that layer; tensors that don't
    /// exist for this layer's type leave their slots `None`. Tensor
    /// names match the manifest exported by `extract_weights.py`.
    pub fn build(
        layer_idx: usize,
        wf: &WeightFile,
        wf_buf: &MtlWeightBuf,
    ) -> Result<Self, MtlWeightBufError> {
        let mut c = Self::default();

        // Helper closure capturing wf + wf_buf so each tensor lookup
        // is one line.
        let off = |name: &str| -> Result<Option<u64>, MtlWeightBufError> {
            wf_buf.tensor_offset(wf, name)
        };

        // Layer-wide norms.
        c.input_layernorm_w =
            off(&format!("model.layers.{layer_idx}.input_layernorm.weight"))?;
        c.post_attention_layernorm_w = off(&format!(
            "model.layers.{layer_idx}.post_attention_layernorm.weight"
        ))?;

        // Linear-attn projections + recurrence weights. Manifest
        // names follow `linear_attn.in_proj_{qkv,z,a,b}` and
        // `linear_attn.out_proj`; `linear_attn.norm` is the gated-
        // norm weight.
        let p = |suffix: &str| {
            format!("model.layers.{layer_idx}.linear_attn.{suffix}")
        };
        c.qkv_w = off(&p("in_proj_qkv.weight"))?;
        c.qkv_s = off(&p("in_proj_qkv.scales"))?;
        c.qkv_b = off(&p("in_proj_qkv.biases"))?;
        c.z_w = off(&p("in_proj_z.weight"))?;
        c.z_s = off(&p("in_proj_z.scales"))?;
        c.z_b = off(&p("in_proj_z.biases"))?;
        c.alpha_w = off(&p("in_proj_a.weight"))?;
        c.alpha_s = off(&p("in_proj_a.scales"))?;
        c.alpha_b = off(&p("in_proj_a.biases"))?;
        c.beta_w = off(&p("in_proj_b.weight"))?;
        c.beta_s = off(&p("in_proj_b.scales"))?;
        c.beta_b = off(&p("in_proj_b.biases"))?;
        c.conv1d_w = off(&p("conv1d.weight"))?;
        c.a_log = off(&p("A_log"))?;
        c.dt_bias = off(&p("dt_bias"))?;
        c.gated_norm_w = off(&p("norm.weight"))?;
        c.linear_o_proj_w = off(&p("out_proj.weight"))?;
        c.linear_o_proj_s = off(&p("out_proj.scales"))?;
        c.linear_o_proj_b = off(&p("out_proj.biases"))?;

        // Full-attn projections (4d will populate; listed here so the
        // build path is uniform across layer types).
        let s = |suffix: &str| {
            format!("model.layers.{layer_idx}.self_attn.{suffix}")
        };
        c.q_proj_w = off(&s("q_proj.weight"))?;
        c.q_proj_s = off(&s("q_proj.scales"))?;
        c.q_proj_b = off(&s("q_proj.biases"))?;
        c.k_proj_w = off(&s("k_proj.weight"))?;
        c.k_proj_s = off(&s("k_proj.scales"))?;
        c.k_proj_b = off(&s("k_proj.biases"))?;
        c.v_proj_w = off(&s("v_proj.weight"))?;
        c.v_proj_s = off(&s("v_proj.scales"))?;
        c.v_proj_b = off(&s("v_proj.biases"))?;
        c.q_norm_w = off(&s("q_norm.weight"))?;
        c.k_norm_w = off(&s("k_norm.weight"))?;
        c.full_o_proj_w = off(&s("o_proj.weight"))?;
        c.full_o_proj_s = off(&s("o_proj.scales"))?;
        c.full_o_proj_b = off(&s("o_proj.biases"))?;

        // MoE router + shared.
        let m = |suffix: &str| format!("model.layers.{layer_idx}.mlp.{suffix}");
        c.gate_w = off(&m("gate.weight"))?;
        c.gate_s = off(&m("gate.scales"))?;
        c.gate_b = off(&m("gate.biases"))?;
        // No separate `gate.bias` in the manifest — `gate.biases` (above)
        // already carries the per-group dequant bias for the matvec.
        c.gate_bias = None;
        c.shared_gate_w = off(&m("shared_expert.gate_proj.weight"))?;
        c.shared_gate_s = off(&m("shared_expert.gate_proj.scales"))?;
        c.shared_gate_b = off(&m("shared_expert.gate_proj.biases"))?;
        c.seg_w = off(&m("shared_expert_gate.weight"))?;
        c.seg_s = off(&m("shared_expert_gate.scales"))?;
        c.seg_b = off(&m("shared_expert_gate.biases"))?;
        c.shared_up_w = off(&m("shared_expert.up_proj.weight"))?;
        c.shared_up_s = off(&m("shared_expert.up_proj.scales"))?;
        c.shared_up_b = off(&m("shared_expert.up_proj.biases"))?;
        c.shared_down_w = off(&m("shared_expert.down_proj.weight"))?;
        c.shared_down_s = off(&m("shared_expert.down_proj.scales"))?;
        c.shared_down_b = off(&m("shared_expert.down_proj.biases"))?;

        Ok(c)
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
