//! Per-model shape constants — Rust port of `metal_infer/model_variant.h`.
//!
//! Mirrors the C header line-for-line: a single [`Variant`] struct
//! plus a feature-gated [`VARIANT`] const carrying the active model's
//! shape. Architecture-wide constants (RMS norm epsilon, RoPE theta,
//! quant group size, …) are plain module-level `pub const`s since
//! they don't change between variants — except where they do, in
//! which case the const itself is `cfg`-gated per variant.
//!
//! ## Adding a new variant
//!
//! 1. Add the matching Cargo feature to `crates/moeflux/Cargo.toml`
//!    (and in `moeflux-sys/Cargo.toml` if there's a C-side oracle for
//!    it; the C header has compile-time variant gating that we keep
//!    in lockstep with the Rust variants).
//! 2. Add a new `#[cfg(feature = "…")]` `VARIANT` block below.
//! 3. Confirm via the integration-test helper `assert_matches_c`
//!    in `tests/common/c_backend.rs` — boots a C ctx and asserts
//!    every public field matches. New variants without a C oracle
//!    skip this check (see e.g. `model-cogito-v2-671b`).
//!
//! ## Sync invariant
//!
//! The Rust constants here MUST agree with the C-side `model_variant.h`
//! values for the same Cargo feature, when both sides ship that
//! feature. The compile-time variant selection in C and the
//! `cfg(feature = …)` selection here are kept in lockstep manually.
//! Any drift is caught at runtime by the integration-test
//! `assert_matches_c` and at test time by the diff oracle.

/// Kind of a single transformer layer. The qwen3_5_moe family
/// alternates linear-attention layers with periodic full-attention
/// layers spaced by [`Variant::full_attn_interval`]; DeepSeek-V3
/// (Cogito-V2-671B) uses MLA on every layer with no linear-attn at
/// all. Kernel dispatch picks the attn flavor from
/// [`Variant::attn_kind`] when [`LayerKind::FullAttn`] applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// GatedDeltaNet linear-attention layer with conv1d + recurrence
    /// state. Cheap per token; constant memory per layer.
    LinearAttn,
    /// Full-attention layer with a KV cache. Memory grows linearly
    /// with sequence length. Inner kernel selected by
    /// [`Variant::attn_kind`] (`Gqa` → existing GQA path, `Mla` →
    /// new MLA path with latent KV cache).
    FullAttn,
}

/// Full-attention kernel flavor.
///
/// `Gqa` is grouped-query attention (Qwen3 family): each KV head is
/// shared across `num_attn_heads / num_kv_heads` query heads, full
/// per-head K/V cached.
///
/// `Mla` is multi-head latent attention (DeepSeek-V3 / Cogito-V2):
/// K and V are jointly compressed to a 576-dim latent (= `kv_lora_rank
/// + qk_rope_head_dim`) per token, decompressed at use time. Q is
/// also LoRA-compressed and split into nope + rope halves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    Gqa,
    Mla,
}

/// MLP kernel flavor for a layer. DeepSeek-V3's first
/// [`Variant::first_k_dense_replace`] layers use a dense SwiGLU MLP
/// (no routing); the rest use the routed-MoE path with optional
/// shared experts. Qwen3-MoE variants are MoE on every layer that
/// gets [`LayerKind::FullAttn`] (linear-attn layers have their own
/// FFN block in the linear-attn forward).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpKind {
    Dense,
    MoE,
}

/// RoPE flavor. The qwen3_5_moe family uses vanilla partial-rotary
/// RoPE with a fixed `ROPE_THETA = 1e7`. DeepSeek-V3 uses YaRN
/// extended-context scaling on top of vanilla RoPE: smooth-ramp
/// frequency interpolation between `freq_inter` (scaled) and
/// `freq_extra` (original) plus a softened mscale on cos/sin and
/// an additional mscale² factor on the attention softmax scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeKind {
    Vanilla,
    Yarn,
}

/// MoE router flavor.
///
/// `Softmax` is the qwen3_5_moe path: softmax over expert logits,
/// top-K selection, renormalize.
///
/// `NoauxTc` is DeepSeek-V3's load-balancing path: sigmoid scoring,
/// per-expert correction-bias added pre-selection, group-limit
/// (sum-of-top-2-per-group → top-`topk_group` groups → mask), final
/// top-K from the masked space, renormalize on **original** sigmoid
/// scores (not biased), multiply by `routed_scaling_factor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterKind {
    Softmax,
    NoauxTc,
}

/// Composition rule for shared experts in MoE blocks.
///
/// `SigmoidGate` is the qwen3_5_moe path: shared-expert output is
/// scaled by `sigmoid(shared_gate_score)`.
///
/// `Unscaled` is the DeepSeek-V3 path: shared-expert output is added
/// to the routed-experts sum unconditionally, no gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedExpertGate {
    SigmoidGate,
    Unscaled,
}

/// All shape parameters for one model variant. `usize` everywhere
/// the C side uses an `int` macro so const arithmetic stays in one
/// type; the [`Variant::eos_*`] / [`Variant::think_*`] tokens stay
/// `i32` to match `mf_eos`'s C signature.
///
/// The MLA / noaux_tc / YaRN / dense-MLP fields are zero-valued (or
/// 1.0 for scale-like parameters) on Qwen3 variants since those
/// kernels are never reached from the GQA / softmax-router /
/// vanilla-RoPE paths.
///
/// `Eq` is intentionally not derived: YaRN / routing-scale fields
/// are `f32`. `PartialEq` is sufficient for the test-side equality
/// checks against the C oracle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Variant {
    pub name: &'static str,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_attn_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate: usize,
    pub shared_intermediate: usize,
    pub full_attn_interval: usize,
    pub linear_num_v_heads: usize,
    pub linear_num_k_heads: usize,

    // Tokenizer specials. `-1` means "not present" (e.g. DeepSeek-V3
    // has no dedicated `<think>` tokens — those are inline ASCII in
    // the chat template).
    pub eos_token_1: i32,
    pub eos_token_2: i32,
    pub think_start_token: i32,
    pub think_end_token: i32,

    // --- Per-variant kernel-flavor selectors ---
    pub attn_kind: AttnKind,
    pub rope_kind: RopeKind,
    pub router_kind: RouterKind,
    pub shared_expert_gate: SharedExpertGate,

    // --- MLA dimensions (zero for Gqa) ---
    /// Q-side LoRA bottleneck. Cogito-V2: 1536. (q_a_proj down,
    /// q_b_proj up.)
    pub q_lora_rank: usize,
    /// KV-side latent dimension cached per token. Cogito-V2: 512.
    pub kv_lora_rank: usize,
    /// Per-head non-rotated portion of Q and K. Cogito-V2: 128.
    pub qk_nope_head_dim: usize,
    /// Per-head rotated portion of Q and K (also the shared-K rope
    /// dim, broadcast across heads). Cogito-V2: 64.
    pub qk_rope_head_dim: usize,
    /// Per-head V dimension (independent of qk_*; V has no rope
    /// split). Cogito-V2: 128.
    pub v_head_dim: usize,

    // --- noaux_tc routing (zero for Softmax) ---
    /// Number of expert groups for group-limited top-K. Cogito-V2: 8.
    pub n_group: usize,
    /// Number of groups selected per token before global top-K.
    /// Cogito-V2: 4.
    pub topk_group: usize,
    /// Multiplier applied to the renormalized expert weights.
    /// Cogito-V2: 2.5. Qwen variants: 1.0.
    pub routed_scaling_factor: f32,

    // --- Dense-FFN-first-K dispatch ---
    /// Number of leading layers using a dense MLP instead of MoE.
    /// Cogito-V2: 3 (layers 0-2 are dense, 3-60 are MoE). Qwen: 0.
    pub first_k_dense_replace: usize,
    /// Hidden width of the dense MLP for the first K layers.
    /// Cogito-V2: 18432. Qwen: 0 (unused).
    pub dense_intermediate: usize,

    // --- YaRN parameters (zero/1.0 for Vanilla rope_kind) ---
    /// YaRN scaling factor. Original-context length is multiplied by
    /// this to get the supported extended context. Cogito-V2: 40.
    pub yarn_factor: f32,
    /// Original (pre-YaRN) max position embedding length used during
    /// pretraining. Cogito-V2: 4096.
    pub yarn_original_max_pos: usize,
    /// YaRN low-end correction-range bound (high-frequency dims
    /// stay at original freqs). Cogito-V2: 32.
    pub yarn_beta_fast: f32,
    /// YaRN high-end correction-range bound (low-frequency dims
    /// switch to scaled freqs). Cogito-V2: 1.
    pub yarn_beta_slow: f32,
    /// YaRN mscale (input to `yarn_get_mscale`). Cogito-V2: 1.0.
    pub yarn_mscale: f32,
    /// YaRN mscale_all_dim (denominator in the mscale ratio that
    /// gets applied to cos/sin and to the softmax scale).
    /// Cogito-V2: 1.0.
    pub yarn_mscale_all_dim: f32,
}

impl Variant {
    /// Linear-attention key dim per head (constant for the qwen3_5_moe
    /// architecture).
    pub const LINEAR_KEY_DIM: usize = 128;
    /// Linear-attention value dim per head.
    pub const LINEAR_VALUE_DIM: usize = 128;
    /// Conv1D kernel width on linear-attention conv state.
    pub const CONV_KERNEL_SIZE: usize = 4;

    /// Total linear-attention key channels = `LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM`.
    pub const fn linear_total_key(&self) -> usize {
        self.linear_num_k_heads * Self::LINEAR_KEY_DIM
    }

    /// Total linear-attention value channels.
    pub const fn linear_total_value(&self) -> usize {
        self.linear_num_v_heads * Self::LINEAR_VALUE_DIM
    }

    /// Linear-attn conv channels (Q + K + V, with 2× key per the
    /// Delta-net split).
    pub const fn linear_conv_dim(&self) -> usize {
        self.linear_total_key() * 2 + self.linear_total_value()
    }

    /// Number of channels the RoPE rotation acts on per Q/K head.
    /// For GQA variants this is `head_dim / 4` (qwen3_5_moe partial
    /// rotary). For MLA variants it's `qk_rope_head_dim` directly —
    /// RoPE is applied only to the rope half of the split Q/K.
    pub const fn rotary_dim(&self) -> usize {
        match self.attn_kind {
            AttnKind::Gqa => self.head_dim / 4,
            AttnKind::Mla => self.qk_rope_head_dim,
        }
    }

    /// Per-layer attn dispatch. The qwen3_5_moe family alternates
    /// linear-attn with full-attn at `full_attn_interval`; DeepSeek-V3
    /// (and any future variant where every layer is MLA) sets
    /// `full_attn_interval = 1` so every layer takes the full-attn
    /// branch and the kernel-dispatch site reads
    /// [`Variant::attn_kind`] to pick MLA vs GQA.
    pub const fn layer_kind(&self, layer_idx: usize) -> LayerKind {
        if (layer_idx + 1) % self.full_attn_interval == 0 {
            LayerKind::FullAttn
        } else {
            LayerKind::LinearAttn
        }
    }

    /// Per-layer MLP dispatch. Layers `< first_k_dense_replace` use a
    /// dense SwiGLU MLP with `dense_intermediate` width; the rest use
    /// the routed-MoE path. For variants without dense early layers
    /// (`first_k_dense_replace = 0`) every layer is MoE.
    pub const fn mlp_kind_at(&self, layer_idx: usize) -> MlpKind {
        if layer_idx < self.first_k_dense_replace {
            MlpKind::Dense
        } else {
            MlpKind::MoE
        }
    }

    // --- 4-bit packed-expert layout (derived) ---------------------

    /// `MOE_INTERMEDIATE * HIDDEN_DIM * 4 / 8` packed nibbles per
    /// gate/up/down weight matrix.
    pub const fn expert_weight_bytes_4bit(&self) -> usize {
        self.moe_intermediate * self.hidden_dim * BITS / 8
    }

    /// `MOE_INTERMEDIATE * (HIDDEN_DIM / GROUP_SIZE) * 2` BF16
    /// scales per matrix; biases share the layout.
    pub const fn expert_scale_bytes(&self) -> usize {
        self.moe_intermediate * (self.hidden_dim / GROUP_SIZE) * 2
    }

    /// One `[weights | scales | biases]` block (gate, up, or down).
    pub const fn expert_block_bytes_4bit(&self) -> usize {
        self.expert_weight_bytes_4bit() + 2 * self.expert_scale_bytes()
    }

    /// Bytes per expert for the 4-bit layout: gate + up + down.
    pub const fn expert_size_4bit(&self) -> usize {
        3 * self.expert_block_bytes_4bit()
    }

    // --- 4-bit expert block offsets ------------------------------
    // Mirror the GATE_*_OFF / UP_*_OFF / DOWN_*_OFF macros in
    // `model_variant.h`. Each block is laid out as
    // `[weights | scales | biases]`; gate is at offset 0, up follows,
    // down follows up.

    pub const fn gate_w_off_4bit(&self) -> usize {
        0
    }
    pub const fn gate_s_off_4bit(&self) -> usize {
        self.expert_weight_bytes_4bit()
    }
    pub const fn gate_b_off_4bit(&self) -> usize {
        self.expert_weight_bytes_4bit() + self.expert_scale_bytes()
    }
    pub const fn up_w_off_4bit(&self) -> usize {
        self.expert_block_bytes_4bit()
    }
    pub const fn up_s_off_4bit(&self) -> usize {
        self.expert_block_bytes_4bit() + self.expert_weight_bytes_4bit()
    }
    pub const fn up_b_off_4bit(&self) -> usize {
        self.expert_block_bytes_4bit()
            + self.expert_weight_bytes_4bit()
            + self.expert_scale_bytes()
    }
    pub const fn down_w_off_4bit(&self) -> usize {
        2 * self.expert_block_bytes_4bit()
    }
    pub const fn down_s_off_4bit(&self) -> usize {
        2 * self.expert_block_bytes_4bit() + self.expert_weight_bytes_4bit()
    }
    pub const fn down_b_off_4bit(&self) -> usize {
        2 * self.expert_block_bytes_4bit()
            + self.expert_weight_bytes_4bit()
            + self.expert_scale_bytes()
    }

    // --- 2-bit packed-expert layout (derived) ---------------------

    pub const fn expert_weight_bytes_2bit(&self) -> usize {
        self.moe_intermediate * self.hidden_dim * 2 / 8
    }

    pub const fn expert_block_bytes_2bit(&self) -> usize {
        self.expert_weight_bytes_2bit() + 2 * self.expert_scale_bytes()
    }

    pub const fn expert_size_2bit(&self) -> usize {
        3 * self.expert_block_bytes_2bit()
    }
}

// --- Architecture-wide constants ---------------------------------

/// RMS-norm epsilon. Same value llama.cpp + most modern decoders use.
pub const RMS_NORM_EPS: f32 = 1e-6;
/// Quantization group size — 64 weights share one BF16 scale.
pub const GROUP_SIZE: usize = 64;
/// 4-bit weight quantization (fixed for the qwen3_5_moe and
/// DeepSeek-V3 families; both use MLX 4-bit `group_size = 64`).
pub const BITS: usize = 4;

/// RoPE base frequency. Per-variant: Qwen3 family uses 1e7 (much
/// higher than the 1e4 default); DeepSeek-V3 / Cogito-V2 uses 1e4
/// (the YaRN scaling layers on top of this base).
#[cfg(any(feature = "model-qwen3-5-a17b", feature = "model-qwen3-6-35b-a3b"))]
pub const ROPE_THETA: f32 = 10_000_000.0;
#[cfg(feature = "model-cogito-v2-671b")]
pub const ROPE_THETA: f32 = 10_000.0;

// --- KV cache / runtime limits (architecture-wide) ----------------

/// Maximum sequence length the architecture supports. Mirrors
/// `MAX_SEQ_LEN` in `model_variant.h`.
pub const MAX_SEQ_LEN: usize = 1_048_576;
/// GPU-resident KV window — KV positions beyond this swap to host.
pub const GPU_KV_SEQ: usize = 8192;

// --- Per-variant selection ---------------------------------------

#[cfg(feature = "model-qwen3-5-a17b")]
pub const VARIANT: Variant = Variant {
    name: "Qwen3.5-397B-A17B-4bit",
    hidden_dim: 4096,
    num_layers: 60,
    num_attn_heads: 32,
    num_kv_heads: 2,
    head_dim: 256,
    vocab_size: 248320,
    num_experts: 512,
    num_experts_per_tok: 10,
    moe_intermediate: 1024,
    shared_intermediate: 1024,
    full_attn_interval: 4,
    linear_num_v_heads: 64,
    linear_num_k_heads: 16,
    eos_token_1: 248046,
    eos_token_2: 248044,
    think_start_token: 248068,
    think_end_token: 248069,
    attn_kind: AttnKind::Gqa,
    rope_kind: RopeKind::Vanilla,
    router_kind: RouterKind::Softmax,
    shared_expert_gate: SharedExpertGate::SigmoidGate,
    q_lora_rank: 0,
    kv_lora_rank: 0,
    qk_nope_head_dim: 0,
    qk_rope_head_dim: 0,
    v_head_dim: 0,
    n_group: 0,
    topk_group: 0,
    routed_scaling_factor: 1.0,
    first_k_dense_replace: 0,
    dense_intermediate: 0,
    yarn_factor: 1.0,
    yarn_original_max_pos: 0,
    yarn_beta_fast: 0.0,
    yarn_beta_slow: 0.0,
    yarn_mscale: 1.0,
    yarn_mscale_all_dim: 1.0,
};

#[cfg(feature = "model-qwen3-6-35b-a3b")]
pub const VARIANT: Variant = Variant {
    name: "Qwen3.6-35B-A3B-4bit",
    hidden_dim: 2048,
    num_layers: 40,
    num_attn_heads: 16,
    num_kv_heads: 2,
    head_dim: 256,
    vocab_size: 248320,
    num_experts: 256,
    num_experts_per_tok: 8,
    moe_intermediate: 512,
    shared_intermediate: 512,
    full_attn_interval: 4,
    linear_num_v_heads: 32,
    linear_num_k_heads: 16,
    eos_token_1: 248046,
    eos_token_2: 248044,
    think_start_token: 248068,
    think_end_token: 248069,
    attn_kind: AttnKind::Gqa,
    rope_kind: RopeKind::Vanilla,
    router_kind: RouterKind::Softmax,
    shared_expert_gate: SharedExpertGate::SigmoidGate,
    q_lora_rank: 0,
    kv_lora_rank: 0,
    qk_nope_head_dim: 0,
    qk_rope_head_dim: 0,
    v_head_dim: 0,
    n_group: 0,
    topk_group: 0,
    routed_scaling_factor: 1.0,
    first_k_dense_replace: 0,
    dense_intermediate: 0,
    yarn_factor: 1.0,
    yarn_original_max_pos: 0,
    yarn_beta_fast: 0.0,
    yarn_beta_slow: 0.0,
    yarn_mscale: 1.0,
    yarn_mscale_all_dim: 1.0,
};

/// Cogito-V2-Preview-671B (DeepSeek-V3 architecture). Source MLX
/// 4-bit safetensors: `mlx-community/cogito-v2-preview-deepseek-
/// 671B-MoE-4bit/config.json`.
///
/// MLA attention with KV-latent compression to 576 dims/token.
/// noaux_tc routing with sigmoid + group-limit. First 3 layers
/// dense (no MoE). YaRN RoPE extension (factor=40 over original
/// 4096-token context).
#[cfg(feature = "model-cogito-v2-671b")]
pub const VARIANT: Variant = Variant {
    name: "Cogito-V2-Preview-671B-4bit",
    hidden_dim: 7168,
    num_layers: 61,
    num_attn_heads: 128,
    // MLA shares K and V state across heads via the latent — the
    // GQA-style num_kv_heads concept doesn't apply. Set equal to
    // num_attn_heads so divisibility checks gated on Gqa stay
    // self-consistent if ever evaluated.
    num_kv_heads: 128,
    // Documentation only for MLA: this is qk_head_dim = nope+rope =
    // 128+64 = 192. Kernels read qk_nope_head_dim / qk_rope_head_dim
    // / v_head_dim directly.
    head_dim: 192,
    vocab_size: 128815,
    num_experts: 256,
    num_experts_per_tok: 8,
    moe_intermediate: 2048,
    shared_intermediate: 2048,
    // Every layer is MLA full-attn; no linear-attn.
    full_attn_interval: 1,
    linear_num_v_heads: 0,
    linear_num_k_heads: 0,
    // bos=0, eos=1. No dedicated <think> tokens — the chat template
    // uses inline ASCII <think>/</think> that the BPE tokenizer
    // splits across multiple tokens.
    eos_token_1: 1,
    eos_token_2: 1,
    think_start_token: -1,
    think_end_token: -1,
    attn_kind: AttnKind::Mla,
    rope_kind: RopeKind::Yarn,
    router_kind: RouterKind::NoauxTc,
    shared_expert_gate: SharedExpertGate::Unscaled,
    q_lora_rank: 1536,
    kv_lora_rank: 512,
    qk_nope_head_dim: 128,
    qk_rope_head_dim: 64,
    v_head_dim: 128,
    n_group: 8,
    topk_group: 4,
    routed_scaling_factor: 2.5,
    first_k_dense_replace: 3,
    dense_intermediate: 18432,
    yarn_factor: 40.0,
    yarn_original_max_pos: 4096,
    yarn_beta_fast: 32.0,
    yarn_beta_slow: 1.0,
    yarn_mscale: 1.0,
    yarn_mscale_all_dim: 1.0,
};

#[cfg(not(any(
    feature = "model-qwen3-5-a17b",
    feature = "model-qwen3-6-35b-a3b",
    feature = "model-cogito-v2-671b",
)))]
compile_error!(
    "moeflux: enable exactly one model variant feature \
     (`model-qwen3-5-a17b`, `model-qwen3-6-35b-a3b`, or \
     `model-cogito-v2-671b`)."
);

// --- Static sanity checks ----------------------------------------
//
// Architecture-wide invariants run for every variant; GQA-specific
// invariants (Q-expansion, KV-head divisibility) only run when the
// active variant uses GQA. MLA has its own constraints (latent
// dimension multiple of group_size, etc.) checked in the MLA-block.

const _: () = {
    assert!(
        VARIANT.hidden_dim % GROUP_SIZE == 0,
        "HIDDEN_DIM must be a multiple of GROUP_SIZE"
    );
    assert!(
        VARIANT.num_experts_per_tok <= VARIANT.num_experts,
        "num_experts_per_tok must be ≤ num_experts"
    );

    // GQA-only: Q is projected into an expanded space whose total
    // channel count must factor cleanly into hidden_dim.
    if matches!(VARIANT.attn_kind, AttnKind::Gqa) {
        assert!(
            VARIANT.num_attn_heads % VARIANT.num_kv_heads == 0,
            "num_attn_heads must be a multiple of num_kv_heads (GQA)"
        );
        assert!(
            (VARIANT.num_attn_heads * VARIANT.head_dim) % VARIANT.hidden_dim
                == 0,
            "num_attn_heads * head_dim must be a multiple of hidden_dim"
        );
    }

    // MLA-only: latent dims must align with the MLX group_size for
    // 4-bit packing of the latent-down and latent-up projections.
    if matches!(VARIANT.attn_kind, AttnKind::Mla) {
        assert!(
            VARIANT.kv_lora_rank % GROUP_SIZE == 0,
            "kv_lora_rank must be a multiple of GROUP_SIZE"
        );
        assert!(
            VARIANT.q_lora_rank % GROUP_SIZE == 0,
            "q_lora_rank must be a multiple of GROUP_SIZE"
        );
        assert!(
            VARIANT.qk_nope_head_dim + VARIANT.qk_rope_head_dim > 0,
            "MLA must define qk_nope_head_dim + qk_rope_head_dim"
        );
        assert!(VARIANT.v_head_dim > 0, "MLA must define v_head_dim");
    }

    // Routing invariants for noaux_tc.
    if matches!(VARIANT.router_kind, RouterKind::NoauxTc) {
        assert!(
            VARIANT.n_group > 0 && VARIANT.topk_group > 0,
            "noaux_tc requires n_group and topk_group > 0"
        );
        assert!(
            VARIANT.num_experts % VARIANT.n_group == 0,
            "num_experts must be divisible by n_group for group-limit routing"
        );
        assert!(
            VARIANT.topk_group <= VARIANT.n_group,
            "topk_group must be ≤ n_group"
        );
    }

    // Dense-FFN-first-K must be < num_layers (otherwise no MoE
    // layers exist and the routed-expert kernels are dead code).
    assert!(
        VARIANT.first_k_dense_replace < VARIANT.num_layers,
        "first_k_dense_replace must be strictly less than num_layers"
    );
};

// --- Runtime cross-check against the C path ----------------------
//
// `assert_matches_c` lives in the integration-test layer at
// `tests/common/c_backend.rs` since Phase 6 — it's the only consumer
// of the C-API safe wrapper, and that wrapper is no longer part of
// moeflux's lib surface. Variants without a C-side counterpart
// (currently `model-cogito-v2-671b`) skip this check; their Phase A
// regression coverage is the static asserts above.

/// Compile-time invariants this module relies on. No-op at runtime;
/// presence forces the static asserts above to be evaluated when
/// the function is referenced (e.g. by tests).
pub const fn assert_static_invariants() {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-fast unit test: invariants and derived sizes don't
    /// blow up. Doesn't need a real model.
    #[test]
    fn variant_is_well_formed() {
        assert_static_invariants();
        let v = VARIANT;
        assert!(v.expert_size_4bit() > 0);
        assert!(v.expert_size_2bit() > 0);
        assert!(v.rotary_dim() > 0);
        // 2-bit experts are half the size of 4-bit (only the weight
        // tensor shrinks; scales/biases are unchanged BF16).
        assert!(v.expert_size_2bit() < v.expert_size_4bit());
        // GQA: heads must group cleanly.
        if matches!(v.attn_kind, AttnKind::Gqa) {
            assert_eq!(v.num_attn_heads % v.num_kv_heads, 0);
            assert!(v.linear_conv_dim() > 0);
        }
    }

    /// `layer_kind(i)` must agree with the legacy modulo predicate for
    /// every layer index in the active variant.
    #[test]
    fn layer_kind_matches_legacy_modulo() {
        let v = VARIANT;
        for i in 0..v.num_layers {
            let legacy_full = (i + 1) % v.full_attn_interval == 0;
            let kind = v.layer_kind(i);
            assert_eq!(
                kind == LayerKind::FullAttn,
                legacy_full,
                "layer_kind({i}) disagrees with legacy modulo predicate \
                 (full_attn_interval = {})",
                v.full_attn_interval,
            );
        }
        let n_full = (0..v.num_layers)
            .filter(|&i| v.layer_kind(i) == LayerKind::FullAttn)
            .count();
        assert!(n_full > 0, "every shipping variant has full-attn layers");
        assert_eq!(n_full, v.num_layers / v.full_attn_interval);
    }

    /// `mlp_kind_at(i)` boundary check: layers strictly below
    /// `first_k_dense_replace` are Dense, the rest are MoE.
    #[test]
    fn mlp_kind_dense_then_moe() {
        let v = VARIANT;
        for i in 0..v.num_layers {
            let expected = if i < v.first_k_dense_replace {
                MlpKind::Dense
            } else {
                MlpKind::MoE
            };
            assert_eq!(v.mlp_kind_at(i), expected, "mlp_kind_at({i})");
        }
    }
}
