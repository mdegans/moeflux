//! Per-model shape constants — Rust port of `metal_infer/model_variant.h`.
//!
//! Mirrors the C header line-for-line: a single [`Variant`] struct
//! plus a feature-gated [`VARIANT`] const carrying the active model's
//! shape. Architecture-wide constants (RMS norm epsilon, RoPE theta,
//! quant group size, …) are plain module-level `pub const`s since
//! they don't change between variants.
//!
//! ## Adding a new variant
//!
//! 1. Add the matching Cargo feature to `crates/moeflux/Cargo.toml`
//!    (and in `moeflux-sys/Cargo.toml`).
//! 2. Add a new `#[cfg(feature = "…")]` `VARIANT` block below.
//! 3. Confirm with the runtime [`assert_matches_c`] sanity check —
//!    boots a C ctx and asserts every public field matches.
//!
//! ## Sync invariant
//!
//! The Rust constants here MUST agree with the C-side `model_variant.h`
//! values for the same Cargo feature. The compile-time variant
//! selection in C and the `cfg(feature = …)` selection here are kept
//! in lockstep manually. Any drift is caught at runtime by
//! [`assert_matches_c`] (uses `mf_n_vocab` / `mf_n_ctx` / `mf_eos`
//! / `mf_model_name`) and at test time by the diff oracle.

/// Kind of a single transformer layer. The qwen3_5_moe family
/// alternates linear-attention layers with periodic full-attention
/// layers spaced by [`Variant::full_attn_interval`]; future variants
/// (notably DeepSeek-V3 with MLA + dense early layers) will need
/// per-layer dispatch that the modulo predicate can't express, so the
/// dispatch goes through [`Variant::layer_kind`] from the start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// GatedDeltaNet linear-attention layer with conv1d + recurrence
    /// state. Cheap per token; constant memory per layer.
    LinearAttn,
    /// Standard scaled-dot-product full-attention layer with a KV
    /// cache. Memory grows linearly with sequence length.
    FullAttn,
}

/// All shape parameters for one model variant. `usize` everywhere
/// the C side uses an `int` macro so const arithmetic stays in one
/// type; the [`Variant::eos_*`] / [`Variant::think_*`] tokens stay
/// `i32` to match `mf_eos`'s C signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    // Tokenizer specials. Same across the qwen3_5_moe family today,
    // but kept per-variant so a future non-Qwen variant can override.
    pub eos_token_1: i32,
    pub eos_token_2: i32,
    pub think_start_token: i32,
    pub think_end_token: i32,
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

    /// Partial-rotary fraction × `head_dim` — number of channels the
    /// RoPE actually rotates. Always `head_dim / 4` for the qwen3_5_moe
    /// family.
    pub const fn rotary_dim(&self) -> usize {
        self.head_dim / 4
    }

    /// Per-layer dispatch predicate. For the qwen3_5_moe family this
    /// is the `(layer_idx + 1) % full_attn_interval == 0` test that
    /// `infer.m:4695` uses inline; we name it so future variants
    /// (notably DeepSeek-V3 with dense early layers + MLA + MoE) can
    /// override the implementation without churning every callsite.
    pub const fn layer_kind(&self, layer_idx: usize) -> LayerKind {
        if (layer_idx + 1) % self.full_attn_interval == 0 {
            LayerKind::FullAttn
        } else {
            LayerKind::LinearAttn
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
/// 4-bit weight quantization (fixed for the qwen3_5_moe family).
pub const BITS: usize = 4;
/// RoPE theta. Qwen3 uses 1e7, much higher than the 1e4 default.
pub const ROPE_THETA: f32 = 10_000_000.0;

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
};

#[cfg(not(any(
    feature = "model-qwen3-5-a17b",
    feature = "model-qwen3-6-35b-a3b",
)))]
compile_error!(
    "moeflux: enable exactly one model variant feature \
     (`model-qwen3-5-a17b` or `model-qwen3-6-35b-a3b`)."
);

// --- Static sanity checks ----------------------------------------
//
// `MOE_INTERMEDIATE * HIDDEN_DIM` must be a multiple of GROUP_SIZE
// for the 4-bit packed layout to align. The C code assumes this
// implicitly; we make it a compile-time error here.

const _: () = {
    assert!(
        VARIANT.hidden_dim % GROUP_SIZE == 0,
        "HIDDEN_DIM must be a multiple of GROUP_SIZE"
    );
    assert!(
        VARIANT.num_attn_heads % VARIANT.num_kv_heads == 0,
        "num_attn_heads must be a multiple of num_kv_heads (GQA)"
    );
    assert!(
        VARIANT.num_experts_per_tok <= VARIANT.num_experts,
        "num_experts_per_tok must be ≤ num_experts"
    );
    // Q is projected into an expanded space in qwen3_5_moe: total
    // Q channels = num_attn_heads * head_dim, NOT hidden_dim. The
    // ratio (called "Q expansion") is 2 for the Qwen3 family. We
    // don't assert exact ratio because it's an architectural choice
    // future variants may change; we only assert it's an integer
    // multiple, which is what the matmul shapes require.
    assert!(
        (VARIANT.num_attn_heads * VARIANT.head_dim) % VARIANT.hidden_dim == 0,
        "num_attn_heads * head_dim must be a multiple of hidden_dim"
    );
};

// --- Runtime cross-check against the C path ----------------------

/// Open a C `Ctx` and assert every shape constant exposed by `mf_*`
/// matches the Rust constant. Used by integration tests; the input
/// `ctx` is borrowed read-only.
///
/// Catches drift if `model_variant.h` is updated but [`VARIANT`]
/// here isn't (or vice versa). Currently the C side only exposes
/// `n_vocab`, `n_ctx`, `eos`, and `model_name` via `mf_*` getters —
/// the rest of the shape parameters aren't reachable through the
/// public C API. Those rely on [`assert_static_invariants`]'s
/// compile-time checks plus the diff oracle's end-to-end behavioral
/// agreement.
pub fn assert_matches_c(ctx: &crate::Ctx) {
    assert_eq!(
        ctx.n_vocab(),
        VARIANT.vocab_size,
        "C n_vocab disagrees with VARIANT.vocab_size",
    );
    assert_eq!(
        ctx.n_ctx(),
        MAX_SEQ_LEN,
        "C n_ctx disagrees with MAX_SEQ_LEN",
    );
    assert_eq!(
        ctx.eos(),
        VARIANT.eos_token_1,
        "C eos disagrees with VARIANT.eos_token_1",
    );
    assert_eq!(
        ctx.model_name(),
        VARIANT.name,
        "C model_name disagrees with VARIANT.name",
    );
}

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
        assert!(v.linear_conv_dim() > 0);
        assert!(v.rotary_dim() > 0);
        // 2-bit experts are half the size of 4-bit (only the weight
        // tensor shrinks; scales/biases are unchanged BF16).
        assert!(v.expert_size_2bit() < v.expert_size_4bit());
        // GQA: heads must group cleanly.
        assert_eq!(v.num_attn_heads % v.num_kv_heads, 0);
    }

    /// `layer_kind(i)` must agree with the legacy modulo predicate for
    /// every layer index in the active variant. Catches drift if a
    /// future variant's `layer_kind` impl diverges from the qwen3_5_moe
    /// shape without the rest of the dispatch being updated.
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
        // Sanity: at least one full-attn layer exists in every variant
        // we ship today.
        let n_full = (0..v.num_layers)
            .filter(|&i| v.layer_kind(i) == LayerKind::FullAttn)
            .count();
        assert!(n_full > 0, "every shipping variant has full-attn layers");
        assert_eq!(n_full, v.num_layers / v.full_attn_interval);
    }
}
