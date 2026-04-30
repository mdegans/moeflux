//! Differential test harness for the RIIR port.
//!
//! Gated to C-supported variants — variants without a C-side oracle
//! (`model-cogito-v2-671b`) skip this whole test file, since the
//! oracle backend doesn't exist for them.
//!
//! Two implementations of [`DiffBackend`]:
//!
//! - **`CBackend`** wraps the existing C-via-`moeflux-sys` path
//!   ([`moeflux::Ctx`]).
//! - **`RsBackend`** wraps the new pure-Rust path ([`moeflux::riir::RsCtx`]).
//!   Phase 0: every method panics. Phase 4 onwards: real impl.
//!
//! Phase 0+ tests run identical inputs through both and compare
//! outputs. Tolerances account for Metal MoE atomic-op nondeterminism
//! (a known issue, not a porting bug):
//!
//! - argmax must match exactly
//! - top-K (K=20) Jaccard ≥ 0.95
//! - full-vector cosine similarity ≥ 0.99
//!
//! `#[ignore]` on every test that touches a real model — needs the
//! ~18 GB of artifacts mounted.
//!
//! ## End-to-end logits are NOT a useful diff oracle
//!
//! Empirically (Phase 0 finding): the C path is non-deterministic
//! across `memory_clear` for the same prompt. Two identical
//! `eval_prompt` calls on one Ctx, with `memory_clear` between,
//! produce logit vectors with cosine sim ≈ 0.65–0.76 and top-20
//! Jaccard ≈ 0.10–0.18 — well below the floors above. Argmax
//! matches some prompts and not others; trajectory match works
//! only because greedy decoding lands in attractors regardless of
//! starting state.
//!
//! Conclusion: the Rust port cannot be diff-tested at the
//! end-to-end-logits boundary against C. The real diff strategy
//! starting Phase 3 will be **intermediate-tensor checkpoints** —
//! both backends will expose hooks to dump per-layer outputs, and
//! comparison happens layer-by-layer where Metal nondeterminism
//! has had less chance to accumulate.
//!
//! ## Phase 0 contents
//!
//! - The trait + C impl + Rust stub + comparison helpers.
//! - One scaffold-validation test (`harness_loads`) that opens the C
//!   backend, prints model metadata, runs a single prefill, and
//!   confirms the helpers don't crash. No equality assertion.
//!
//! Phase 1+ adds tests that run `CBackend` vs `RsBackend` for each
//! ported subsystem, with the per-layer hooks landing in Phase 3.
//!
//! ```bash
//! cargo test -p moeflux \
//!     --features "model-qwen3-6-35b-a3b" \
//!     --test diff_oracle --release \
//!     -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(all(
    target_os = "macos",
    any(
        feature = "model-qwen3-5-a17b",
        feature = "model-qwen3-6-35b-a3b",
    ),
))]

use std::path::{Path, PathBuf};

mod common;
use common::c_backend::Ctx;
use moeflux::riir::RsCtx;

// ---------------------------------------------------------------------------
// Trait + impls
// ---------------------------------------------------------------------------

/// Common behavior the harness exercises across both backends.
///
/// Methods mirror [`moeflux::Ctx`]'s public API 1:1 — the diff
/// harness treats this surface as the boundary at which behavior
/// must agree. Each impl forwards to its underlying ctx; the
/// abstraction is purely a witness for trait-generic test code.
pub trait DiffBackend {
    fn open(
        weights: &Path,
        manifest: &Path,
        vocab: &Path,
        experts_dir: &Path,
        experts_per_tok: u32,
        use_2bit: bool,
    ) -> Self;

    fn n_vocab(&self) -> usize;
    fn n_ctx(&self) -> usize;
    fn eos(&self) -> i32;
    fn model_name(&self) -> &'static str;

    /// Embed a single token. Returns a `HIDDEN_DIM`-long f32 vector.
    /// First per-kernel diff point landed in Phase 3.
    fn embed(&self, token_id: i32) -> Vec<f32>;

    /// CPU RMSNorm against the BF16 weight tensor `weight_name`.
    /// Returns a `HIDDEN_DIM`-long f32 vector.
    fn rms_norm_cpu(&self, weight_name: &str, x: &[f32]) -> Vec<f32>;

    /// Apply rotary position embedding to Q and K at `pos`. Returns
    /// `(q_out, k_out)`; inputs are not mutated.
    fn apply_rotary_emb(
        &self,
        pos: i32,
        q: &[f32],
        k: &[f32],
    ) -> (Vec<f32>, Vec<f32>);

    /// Per-head CPU RMSNorm against the bf16 weight tensor
    /// `weight_name` (length `head_dim`). Returns the
    /// `num_heads * head_dim`-long output; the input is not mutated.
    fn rms_norm_per_head_cpu(
        &self,
        weight_name: &str,
        num_heads: usize,
        head_dim: usize,
        x: &[f32],
    ) -> Vec<f32>;

    /// CPU scaled dot-product attention with sigmoid-gated output for
    /// one query position against `kv_len` cached positions. Returns
    /// the `num_attn_heads * head_dim`-long gated attention output.
    fn sdpa_cpu(
        &self,
        kv_len: i32,
        q: &[f32],
        q_gate: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
    ) -> Vec<f32>;

    /// CPU LM head matvec. `x` is `HIDDEN_DIM` floats; the returned
    /// vector is `VOCAB_SIZE` floats (raw logits).
    fn lm_head_cpu(&self, x: &[f32]) -> Vec<f32>;

    /// MoE router: softmax → top-K → normalize. Takes the raw gate
    /// logits, returns `(indices, weights)` parallel arrays of length
    /// `k`. `scores` is consumed as input; the C path mutates it in
    /// place but the trait surface hands over an owned copy each call.
    fn moe_router_cpu(&self, scores: Vec<f32>, k: usize) -> (Vec<i32>, Vec<f32>);

    /// Depthwise 1D conv step + SiLU. `weight_name` is a bf16 tensor
    /// of length `channels * kernel_size`. Returns `channels` floats.
    fn conv1d_step_cpu(
        &self,
        weight_name: &str,
        channels: usize,
        kernel_size: usize,
        conv_state: &[f32],
        new_input: &[f32],
    ) -> Vec<f32>;

    /// Bare CPU RMSNorm (no weight). Returns `x.len()` floats.
    fn rms_norm_bare_cpu(&self, eps: f32, x: &[f32]) -> Vec<f32>;

    /// CPU RMSNormGated. Returns `x.len()` floats.
    fn rms_norm_gated_cpu(
        &self,
        weight_name: &str,
        eps: f32,
        x: &[f32],
        z: &[f32],
    ) -> Vec<f32>;

    /// Gated-delta-net recurrence step. Returns the post-step
    /// `(ssm_state, out_values)` pair — input state is consumed; the
    /// trait surface clones it per call so the harness can run both
    /// backends from identical starting states.
    #[allow(clippy::too_many_arguments)]
    fn gated_delta_recurrence_cpu(
        &self,
        layer_idx: usize,
        alpha: &[f32],
        beta: &[f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        v_heads: usize,
        k_heads: usize,
        key_dim: usize,
        value_dim: usize,
        ssm_state_in: Vec<f32>,
    ) -> (Vec<f32>, Vec<f32>);

    /// Read one expert's `EXPERT_SIZE`-byte 4-bit blob from disk
    /// (slice 9c). Returns the raw on-disk bytes.
    fn load_expert_bytes(&self, layer_idx: i32, expert_idx: i32) -> Vec<u8>;

    /// GPU RMSNorm with bf16 weights (slice 9e). `x` is HIDDEN_DIM
    /// floats; `weight_bf16` is HIDDEN_DIM × 2 bytes (typically the
    /// raw `model.norm.weight` mmap region). Returns HIDDEN_DIM floats.
    fn gpu_rms_norm_fused(
        &mut self,
        x: &[f32],
        weight_bf16: &[u8],
    ) -> Vec<f32>;

    /// Single-expert GPU FFN forward (slice 9a). `expert_data` is one
    /// expert's `EXPERT_SIZE`-byte 4-bit blob; `h_post` is HIDDEN_DIM
    /// floats. Returns the HIDDEN_DIM-float expert output. Takes
    /// `&mut self` because the Rust backend builds the Metal device
    /// lazily on first GPU call.
    fn gpu_expert_forward(
        &mut self,
        expert_data: &[u8],
        h_post: &[f32],
    ) -> Vec<f32>;

    /// Batched K-expert FFN forward + GPU combine (slice 9b).
    /// `expert_data` is `actual_k * EXPERT_SIZE` bytes (K blobs in slot
    /// order). Returns the HIDDEN_DIM-float post-combine hidden state.
    #[allow(clippy::too_many_arguments)]
    fn gpu_batched_experts_forward(
        &mut self,
        actual_k: i32,
        expert_data: &[u8],
        h_post: &[f32],
        h_mid: &[f32],
        shared_out: &[f32],
        expert_weights: &[f32],
        shared_gate_score: f32,
    ) -> Vec<f32>;

    /// `attn_scores_batched` (slice 5d-7a). Returns `[num_heads * seq_len]`
    /// scaled per-head Q · K^T scores (stride-tight).
    #[allow(clippy::too_many_arguments)]
    fn attn_scores_batched(
        &mut self,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        q: &[f32],
        k_cache: &[f32],
        scale: f32,
    ) -> Vec<f32>;

    /// `attn_softmax_batched` (slice 5d-7a). Per-head softmax over
    /// `[0, seq_len)`. Input is `[num_heads * seq_len]` raw scores;
    /// output is the same shape, post-softmax.
    fn attn_softmax_batched(
        &mut self,
        num_heads: u32,
        seq_len: u32,
        scores_in: &[f32],
    ) -> Vec<f32>;

    /// `attn_values_batched` (slice 5d-7a). Returns `[num_heads *
    /// head_dim]` per-head value aggregation.
    #[allow(clippy::too_many_arguments)]
    fn attn_values_batched(
        &mut self,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        scores: &[f32],
        v_cache: &[f32],
    ) -> Vec<f32>;

    /// `sigmoid_gate` (slice 5d-7a). Returns `[dim]` gated values
    /// (`x_in[i] * sigmoid(gate[i])`). Caller passes the pre-gate
    /// values in `x_in`; the trait surface clones to the in/out buffer
    /// internally.
    fn sigmoid_gate(
        &mut self,
        dim: u32,
        gate: &[f32],
        x_in: &[f32],
    ) -> Vec<f32>;

    /// Slice 4e — begin a deferred K-expert dispatch (commits async,
    /// no readback). Pair with [`Self::complete_deferred_experts`] or
    /// [`Self::discard_deferred_experts`].
    #[allow(clippy::too_many_arguments)]
    fn begin_deferred_experts(
        &mut self,
        actual_k: i32,
        expert_data: &[u8],
        h_post: &[f32],
        h_mid: &[f32],
        shared_out: &[f32],
        expert_weights: &[f32],
        shared_gate_score: f32,
    );

    /// Slice 4e — wait for the deferred dispatch and read back the
    /// post-combine hidden state. Returns HIDDEN_DIM floats; an
    /// all-zero vector if no deferred dispatch was active (matches the
    /// C-side no-op semantics).
    fn complete_deferred_experts(&mut self) -> Vec<f32>;

    /// Slice 4e — wait for the deferred dispatch and clear state
    /// without readback. Used in production for prefill tokens whose
    /// hidden state is overwritten by the next token's embedding.
    fn discard_deferred_experts(&mut self);

    /// Phase 4 layer-boundary diff checkpoint. Runs one layer's
    /// forward starting from `hidden_in` and returns the post-layer
    /// HIDDEN_DIM state. Drives the layer's per-layer state in place
    /// (callers are expected to `memory_clear` between independent
    /// trials so the KV / recurrence start state matches across
    /// backends). Tests land in 4c (linear-attn) / 4d (full-attn);
    /// the trait method is here in 4b so both backend impls can be
    /// wired ahead of the kernel landing.
    fn layer_forward_dump(
        &mut self,
        layer_idx: i32,
        pos: i32,
        hidden_in: &[f32],
    ) -> Vec<f32>;

    /// Prefill `tokens` at `start_pos`. Returns the n_vocab-length
    /// logit vector for the position immediately after the last
    /// token in `tokens`.
    fn eval_prompt(&mut self, tokens: &[i32], start_pos: usize) -> Vec<f32>;

    /// Decode a single token at `pos`. Returns the next-token logit
    /// vector.
    fn eval_token(&mut self, token: i32, pos: usize) -> Vec<f32>;

    fn memory_clear(&mut self);
    fn memory_seq_rm(&mut self, p0: i32, p1: i32) -> bool;
    fn memory_seq_pos_max(&self) -> i32;
}

/// C-via-`moeflux-sys` impl. Thin wrapper around [`moeflux::Ctx`].
/// Field is public so tests in this file can call inherent `Ctx`
/// methods that aren't on the [`DiffBackend`] trait (e.g.
/// state_save / state_load checks in later phases).
pub struct CBackend(pub Ctx);

impl DiffBackend for CBackend {
    fn open(
        weights: &Path,
        manifest: &Path,
        vocab: &Path,
        experts_dir: &Path,
        experts_per_tok: u32,
        use_2bit: bool,
    ) -> Self {
        Self(
            Ctx::open(
                weights,
                manifest,
                vocab,
                experts_dir,
                experts_per_tok,
                use_2bit,
            )
            .expect("CBackend Ctx::open"),
        )
    }

    fn n_vocab(&self) -> usize {
        self.0.n_vocab()
    }
    fn n_ctx(&self) -> usize {
        self.0.n_ctx()
    }
    fn eos(&self) -> i32 {
        self.0.eos()
    }
    fn model_name(&self) -> &'static str {
        self.0.model_name()
    }

    fn embed(&self, token_id: i32) -> Vec<f32> {
        let mut out = vec![0.0f32; moeflux::riir::VARIANT.hidden_dim];
        self.0.embed(token_id, &mut out).expect("CBackend embed");
        out
    }

    fn rms_norm_cpu(&self, weight_name: &str, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; moeflux::riir::VARIANT.hidden_dim];
        self.0
            .rms_norm_cpu(weight_name, x, &mut out)
            .expect("CBackend rms_norm_cpu");
        out
    }

    fn apply_rotary_emb(
        &self,
        pos: i32,
        q: &[f32],
        k: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut q_out = q.to_vec();
        let mut k_out = k.to_vec();
        self.0
            .apply_rotary_emb(pos, &mut q_out, &mut k_out)
            .expect("CBackend apply_rotary_emb");
        (q_out, k_out)
    }

    fn rms_norm_per_head_cpu(
        &self,
        weight_name: &str,
        num_heads: usize,
        head_dim: usize,
        x: &[f32],
    ) -> Vec<f32> {
        let mut out = x.to_vec();
        self.0
            .rms_norm_per_head_cpu(weight_name, num_heads, head_dim, &mut out)
            .expect("CBackend rms_norm_per_head_cpu");
        out
    }

    fn sdpa_cpu(
        &self,
        kv_len: i32,
        q: &[f32],
        q_gate: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; q.len()];
        self.0
            .sdpa_cpu(kv_len, q, q_gate, k_cache, v_cache, &mut out)
            .expect("CBackend sdpa_cpu");
        out
    }

    fn lm_head_cpu(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.0.n_vocab()];
        self.0
            .lm_head_cpu(x, &mut out)
            .expect("CBackend lm_head_cpu");
        out
    }

    fn moe_router_cpu(&self, scores: Vec<f32>, k: usize) -> (Vec<i32>, Vec<f32>) {
        let mut s = scores;
        let mut idx = vec![0i32; k];
        let mut w = vec![0.0f32; k];
        self.0
            .moe_router_cpu(&mut s, k, &mut idx, &mut w)
            .expect("CBackend moe_router_cpu");
        (idx, w)
    }

    fn conv1d_step_cpu(
        &self,
        weight_name: &str,
        channels: usize,
        kernel_size: usize,
        conv_state: &[f32],
        new_input: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; channels];
        self.0
            .conv1d_step_cpu(
                weight_name,
                channels,
                kernel_size,
                conv_state,
                new_input,
                &mut out,
            )
            .expect("CBackend conv1d_step_cpu");
        out
    }

    fn rms_norm_bare_cpu(&self, eps: f32, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        self.0
            .rms_norm_bare_cpu(eps, x, &mut out)
            .expect("CBackend rms_norm_bare_cpu");
        out
    }

    fn rms_norm_gated_cpu(
        &self,
        weight_name: &str,
        eps: f32,
        x: &[f32],
        z: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        self.0
            .rms_norm_gated_cpu(weight_name, eps, x, z, &mut out)
            .expect("CBackend rms_norm_gated_cpu");
        out
    }

    fn gated_delta_recurrence_cpu(
        &self,
        layer_idx: usize,
        alpha: &[f32],
        beta: &[f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        v_heads: usize,
        k_heads: usize,
        key_dim: usize,
        value_dim: usize,
        ssm_state_in: Vec<f32>,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut state = ssm_state_in;
        let mut out = vec![0.0f32; v_heads * value_dim];
        self.0
            .gated_delta_recurrence_cpu(
                layer_idx,
                alpha,
                beta,
                q,
                k,
                v,
                v_heads,
                k_heads,
                key_dim,
                value_dim,
                &mut state,
                &mut out,
            )
            .expect("CBackend gated_delta_recurrence_cpu");
        (state, out)
    }

    fn load_expert_bytes(&self, layer_idx: i32, expert_idx: i32) -> Vec<u8> {
        let mut out = vec![0u8; moeflux::riir::VARIANT.expert_size_4bit()];
        self.0
            .load_expert_bytes(layer_idx, expert_idx, &mut out)
            .expect("CBackend load_expert_bytes");
        out
    }

    fn gpu_rms_norm_fused(
        &mut self,
        x: &[f32],
        weight_bf16: &[u8],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; moeflux::riir::VARIANT.hidden_dim];
        self.0
            .gpu_rms_norm_fused(x, weight_bf16, &mut out)
            .expect("CBackend gpu_rms_norm_fused");
        out
    }

    fn gpu_expert_forward(
        &mut self,
        expert_data: &[u8],
        h_post: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; moeflux::riir::VARIANT.hidden_dim];
        self.0
            .gpu_expert_forward(expert_data, h_post, &mut out)
            .expect("CBackend gpu_expert_forward");
        out
    }

    fn gpu_batched_experts_forward(
        &mut self,
        actual_k: i32,
        expert_data: &[u8],
        h_post: &[f32],
        h_mid: &[f32],
        shared_out: &[f32],
        expert_weights: &[f32],
        shared_gate_score: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; moeflux::riir::VARIANT.hidden_dim];
        self.0
            .gpu_batched_experts_forward(
                actual_k,
                expert_data,
                h_post,
                h_mid,
                shared_out,
                expert_weights,
                shared_gate_score,
                &mut out,
            )
            .expect("CBackend gpu_batched_experts_forward");
        out
    }

    fn attn_scores_batched(
        &mut self,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        q: &[f32],
        k_cache: &[f32],
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; (num_heads * seq_len) as usize];
        self.0
            .attn_scores_batched(
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                seq_len as i32,
                q,
                k_cache,
                scale,
                &mut out,
            )
            .expect("CBackend attn_scores_batched");
        out
    }

    fn attn_softmax_batched(
        &mut self,
        num_heads: u32,
        seq_len: u32,
        scores_in: &[f32],
    ) -> Vec<f32> {
        let mut out = scores_in.to_vec();
        self.0
            .attn_softmax_batched(num_heads as i32, seq_len as i32, &mut out)
            .expect("CBackend attn_softmax_batched");
        out
    }

    fn attn_values_batched(
        &mut self,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        scores: &[f32],
        v_cache: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; (num_heads * head_dim) as usize];
        self.0
            .attn_values_batched(
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                seq_len as i32,
                scores,
                v_cache,
                &mut out,
            )
            .expect("CBackend attn_values_batched");
        out
    }

    fn sigmoid_gate(
        &mut self,
        dim: u32,
        gate: &[f32],
        x_in: &[f32],
    ) -> Vec<f32> {
        let mut out = x_in.to_vec();
        self.0
            .sigmoid_gate(dim as i32, gate, &mut out)
            .expect("CBackend sigmoid_gate");
        out
    }

    fn begin_deferred_experts(
        &mut self,
        actual_k: i32,
        expert_data: &[u8],
        h_post: &[f32],
        h_mid: &[f32],
        shared_out: &[f32],
        expert_weights: &[f32],
        shared_gate_score: f32,
    ) {
        self.0
            .begin_deferred_experts(
                actual_k,
                expert_data,
                h_post,
                h_mid,
                shared_out,
                expert_weights,
                shared_gate_score,
            )
            .expect("CBackend begin_deferred_experts");
    }

    fn complete_deferred_experts(&mut self) -> Vec<f32> {
        let mut out = vec![0.0f32; moeflux::riir::VARIANT.hidden_dim];
        self.0
            .complete_deferred_experts(&mut out)
            .expect("CBackend complete_deferred_experts");
        out
    }

    fn discard_deferred_experts(&mut self) {
        self.0
            .discard_deferred_experts()
            .expect("CBackend discard_deferred_experts");
    }

    fn layer_forward_dump(
        &mut self,
        layer_idx: i32,
        pos: i32,
        hidden_in: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; moeflux::riir::VARIANT.hidden_dim];
        self.0
            .layer_forward_dump(layer_idx, pos, hidden_in, &mut out)
            .expect("CBackend layer_forward_dump");
        out
    }

    fn eval_prompt(&mut self, tokens: &[i32], start_pos: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.0.n_vocab()];
        self.0
            .eval_prompt(tokens, start_pos, 0, &mut logits)
            .expect("CBackend eval_prompt");
        logits
    }

    fn eval_token(&mut self, token: i32, pos: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.0.n_vocab()];
        self.0
            .eval_token(token, pos, 0, &mut logits)
            .expect("CBackend eval_token");
        logits
    }

    fn memory_clear(&mut self) {
        self.0.memory_clear()
    }
    fn memory_seq_rm(&mut self, p0: i32, p1: i32) -> bool {
        self.0.memory_seq_rm(0, p0, p1)
    }
    fn memory_seq_pos_max(&self) -> i32 {
        self.0.memory_seq_pos_max(0)
    }
}

/// Pure-Rust impl. Phase 3: methods become real as their kernels are
/// ported (embedding landed; the rest still `todo!()`).
pub struct RsBackend(RsCtx);

impl DiffBackend for RsBackend {
    fn open(
        weights: &Path,
        manifest: &Path,
        vocab: &Path,
        experts_dir: &Path,
        experts_per_tok: u32,
        use_2bit: bool,
    ) -> Self {
        Self(
            RsCtx::open(
                weights,
                manifest,
                vocab,
                experts_dir,
                experts_per_tok,
                use_2bit,
            )
            .expect("RsBackend RsCtx::open"),
        )
    }

    fn n_vocab(&self) -> usize {
        self.0.n_vocab()
    }
    fn n_ctx(&self) -> usize {
        self.0.n_ctx()
    }
    fn eos(&self) -> i32 {
        self.0.eos()
    }
    fn model_name(&self) -> &'static str {
        self.0.model_name()
    }

    fn embed(&self, token_id: i32) -> Vec<f32> {
        let mut out = vec![0.0f32; moeflux::riir::VARIANT.hidden_dim];
        self.0.embed(token_id, &mut out).expect("RsBackend embed");
        out
    }

    fn rms_norm_cpu(&self, weight_name: &str, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; moeflux::riir::VARIANT.hidden_dim];
        self.0
            .rms_norm_cpu(weight_name, x, &mut out)
            .expect("RsBackend rms_norm_cpu");
        out
    }

    fn apply_rotary_emb(
        &self,
        pos: i32,
        q: &[f32],
        k: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut q_out = q.to_vec();
        let mut k_out = k.to_vec();
        self.0
            .apply_rotary_emb(pos, &mut q_out, &mut k_out)
            .expect("RsBackend apply_rotary_emb");
        (q_out, k_out)
    }

    fn rms_norm_per_head_cpu(
        &self,
        weight_name: &str,
        num_heads: usize,
        head_dim: usize,
        x: &[f32],
    ) -> Vec<f32> {
        let mut out = x.to_vec();
        self.0
            .rms_norm_per_head_cpu(weight_name, num_heads, head_dim, &mut out)
            .expect("RsBackend rms_norm_per_head_cpu");
        out
    }

    fn sdpa_cpu(
        &self,
        kv_len: i32,
        q: &[f32],
        q_gate: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; q.len()];
        self.0
            .sdpa_cpu(kv_len, q, q_gate, k_cache, v_cache, &mut out)
            .expect("RsBackend sdpa_cpu");
        out
    }

    fn lm_head_cpu(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.0.n_vocab()];
        self.0
            .lm_head_cpu(x, &mut out)
            .expect("RsBackend lm_head_cpu");
        out
    }

    fn moe_router_cpu(&self, scores: Vec<f32>, k: usize) -> (Vec<i32>, Vec<f32>) {
        let mut s = scores;
        let mut idx = vec![0i32; k];
        let mut w = vec![0.0f32; k];
        self.0
            .moe_router_cpu(&mut s, k, &mut idx, &mut w)
            .expect("RsBackend moe_router_cpu");
        (idx, w)
    }

    fn conv1d_step_cpu(
        &self,
        weight_name: &str,
        channels: usize,
        kernel_size: usize,
        conv_state: &[f32],
        new_input: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; channels];
        self.0
            .conv1d_step_cpu(
                weight_name,
                channels,
                kernel_size,
                conv_state,
                new_input,
                &mut out,
            )
            .expect("RsBackend conv1d_step_cpu");
        out
    }

    fn rms_norm_bare_cpu(&self, eps: f32, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        self.0
            .rms_norm_bare_cpu(eps, x, &mut out)
            .expect("RsBackend rms_norm_bare_cpu");
        out
    }

    fn rms_norm_gated_cpu(
        &self,
        weight_name: &str,
        eps: f32,
        x: &[f32],
        z: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        self.0
            .rms_norm_gated_cpu(weight_name, eps, x, z, &mut out)
            .expect("RsBackend rms_norm_gated_cpu");
        out
    }

    fn gated_delta_recurrence_cpu(
        &self,
        layer_idx: usize,
        alpha: &[f32],
        beta: &[f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        v_heads: usize,
        k_heads: usize,
        key_dim: usize,
        value_dim: usize,
        ssm_state_in: Vec<f32>,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut state = ssm_state_in;
        let mut out = vec![0.0f32; v_heads * value_dim];
        self.0
            .gated_delta_recurrence_cpu(
                layer_idx,
                alpha,
                beta,
                q,
                k,
                v,
                v_heads,
                k_heads,
                key_dim,
                value_dim,
                &mut state,
                &mut out,
            )
            .expect("RsBackend gated_delta_recurrence_cpu");
        (state, out)
    }

    fn load_expert_bytes(&self, layer_idx: i32, expert_idx: i32) -> Vec<u8> {
        let mut out = vec![0u8; moeflux::riir::VARIANT.expert_size_4bit()];
        self.0
            .load_expert_bytes(
                layer_idx as usize,
                expert_idx as usize,
                &mut out,
            )
            .expect("RsBackend load_expert_bytes");
        out
    }

    fn gpu_rms_norm_fused(
        &mut self,
        x: &[f32],
        weight_bf16: &[u8],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; moeflux::riir::VARIANT.hidden_dim];
        self.0
            .gpu_rms_norm_fused(x, weight_bf16, &mut out)
            .expect("RsBackend gpu_rms_norm_fused");
        out
    }

    fn gpu_expert_forward(
        &mut self,
        expert_data: &[u8],
        h_post: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; moeflux::riir::VARIANT.hidden_dim];
        self.0
            .gpu_expert_forward(expert_data, h_post, &mut out)
            .expect("RsBackend gpu_expert_forward");
        out
    }

    fn gpu_batched_experts_forward(
        &mut self,
        actual_k: i32,
        expert_data: &[u8],
        h_post: &[f32],
        h_mid: &[f32],
        shared_out: &[f32],
        expert_weights: &[f32],
        shared_gate_score: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; moeflux::riir::VARIANT.hidden_dim];
        self.0
            .gpu_batched_experts_forward(
                actual_k,
                expert_data,
                h_post,
                h_mid,
                shared_out,
                expert_weights,
                shared_gate_score,
                &mut out,
            )
            .expect("RsBackend gpu_batched_experts_forward");
        out
    }

    fn attn_scores_batched(
        &mut self,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        q: &[f32],
        k_cache: &[f32],
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; (num_heads * seq_len) as usize];
        self.0
            .attn_scores_batched(
                num_heads, num_kv_heads, head_dim, seq_len, q, k_cache,
                scale, &mut out,
            )
            .expect("RsBackend attn_scores_batched");
        out
    }

    fn attn_softmax_batched(
        &mut self,
        num_heads: u32,
        seq_len: u32,
        scores_in: &[f32],
    ) -> Vec<f32> {
        let mut out = scores_in.to_vec();
        self.0
            .attn_softmax_batched(num_heads, seq_len, &mut out)
            .expect("RsBackend attn_softmax_batched");
        out
    }

    fn attn_values_batched(
        &mut self,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        scores: &[f32],
        v_cache: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; (num_heads * head_dim) as usize];
        self.0
            .attn_values_batched(
                num_heads, num_kv_heads, head_dim, seq_len, scores, v_cache,
                &mut out,
            )
            .expect("RsBackend attn_values_batched");
        out
    }

    fn sigmoid_gate(
        &mut self,
        dim: u32,
        gate: &[f32],
        x_in: &[f32],
    ) -> Vec<f32> {
        let mut out = x_in.to_vec();
        self.0
            .sigmoid_gate(dim, gate, &mut out)
            .expect("RsBackend sigmoid_gate");
        out
    }

    fn begin_deferred_experts(
        &mut self,
        actual_k: i32,
        expert_data: &[u8],
        h_post: &[f32],
        h_mid: &[f32],
        shared_out: &[f32],
        expert_weights: &[f32],
        shared_gate_score: f32,
    ) {
        // layer_idx = -1 mirrors the C hook (synthetic / no real layer).
        self.0
            .begin_deferred_experts(
                actual_k,
                expert_data,
                h_post,
                h_mid,
                shared_out,
                expert_weights,
                shared_gate_score,
                -1,
            )
            .expect("RsBackend begin_deferred_experts");
    }

    fn complete_deferred_experts(&mut self) -> Vec<f32> {
        let mut out = vec![0.0f32; moeflux::riir::VARIANT.hidden_dim];
        self.0
            .complete_deferred_experts(&mut out)
            .expect("RsBackend complete_deferred_experts");
        out
    }

    fn discard_deferred_experts(&mut self) {
        self.0.discard_deferred_experts();
    }

    fn layer_forward_dump(
        &mut self,
        layer_idx: i32,
        pos: i32,
        hidden_in: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; moeflux::riir::VARIANT.hidden_dim];
        self.0
            .layer_forward_dump(layer_idx, pos, hidden_in, &mut out)
            .expect("RsBackend layer_forward_dump");
        out
    }

    fn eval_prompt(&mut self, tokens: &[i32], start_pos: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.0.n_vocab()];
        self.0
            .eval_prompt(tokens, start_pos, 0, &mut logits)
            .expect("RsBackend eval_prompt");
        logits
    }

    fn eval_token(&mut self, token: i32, pos: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.0.n_vocab()];
        self.0
            .eval_token(token, pos, 0, &mut logits)
            .expect("RsBackend eval_token");
        logits
    }

    fn memory_clear(&mut self) {
        self.0.memory_clear()
    }
    fn memory_seq_rm(&mut self, p0: i32, p1: i32) -> bool {
        self.0.memory_seq_rm(0, p0, p1)
    }
    fn memory_seq_pos_max(&self) -> i32 {
        self.0.memory_seq_pos_max(0)
    }
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

/// Argmax (id of largest logit). Ties broken by lowest id.
pub fn argmax(logits: &[f32]) -> i32 {
    let mut best_id = 0i32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_id = i as i32;
        }
    }
    best_id
}

/// Top-K ids by descending logit, ties broken by ascending id.
pub fn topk(logits: &[f32], k: usize) -> Vec<i32> {
    let mut idx: Vec<(i32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as i32, v))
        .collect();
    idx.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    idx.truncate(k);
    idx.into_iter().map(|(i, _)| i).collect()
}

/// Jaccard set overlap of two id lists.
pub fn jaccard(a: &[i32], b: &[i32]) -> f32 {
    use std::collections::HashSet;
    let sa: HashSet<i32> = a.iter().copied().collect();
    let sb: HashSet<i32> = b.iter().copied().collect();
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    if union == 0.0 { 1.0 } else { inter / union }
}

/// Cosine similarity over the full logit vector. Robust to scale
/// differences; catches both directional and magnitude drift up to
/// a global multiplicative factor.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine_sim: length mismatch");
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 { 1.0 } else { (dot / denom) as f32 }
}

/// Tolerances applied at every diff-check site. Conservative defaults
/// — Metal MoE atomic-op nondeterminism is the dominant noise source
/// and stays well below these floors empirically.
pub const TOPK_K: usize = 20;
pub const TOPK_JACCARD_MIN: f32 = 0.95;
pub const COSINE_SIM_MIN: f32 = 0.99;

/// Full diff check on one logit vector. Asserts argmax match,
/// top-K Jaccard floor, and cosine-sim floor; logs all three.
pub fn assert_logits_close(label: &str, c: &[f32], rs: &[f32]) {
    let c_arg = argmax(c);
    let rs_arg = argmax(rs);
    let c_top = topk(c, TOPK_K);
    let rs_top = topk(rs, TOPK_K);
    let jac = jaccard(&c_top, &rs_top);
    let cos = cosine_sim(c, rs);

    eprintln!(
        "[diff:{label}] argmax c={c_arg} rs={rs_arg} \
         top-{TOPK_K} jaccard={jac:.4} cosine={cos:.5}"
    );

    assert_eq!(
        c_arg, rs_arg,
        "[diff:{label}] argmax mismatch (c={c_arg} rs={rs_arg})"
    );
    assert!(
        jac >= TOPK_JACCARD_MIN,
        "[diff:{label}] top-{TOPK_K} jaccard {jac:.4} below {TOPK_JACCARD_MIN}"
    );
    assert!(
        cos >= COSINE_SIM_MIN,
        "[diff:{label}] cosine sim {cos:.5} below {COSINE_SIM_MIN}"
    );
}

// ---------------------------------------------------------------------------
// Path resolution (env-var override, mirrors smoke.rs)
// ---------------------------------------------------------------------------

fn artifacts_dir() -> PathBuf {
    let default =
        "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-artifacts";
    PathBuf::from(
        std::env::var("MOEFLUX_SMOKE_ARTIFACTS").unwrap_or(default.into()),
    )
}

fn root_dir() -> PathBuf {
    let default =
        "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-root";
    PathBuf::from(std::env::var("MOEFLUX_SMOKE_ROOT").unwrap_or(default.into()))
}

/// Open a backend with the standard A3B artifacts layout. Used by
/// every test in the harness so the path resolution lives in one
/// place.
pub fn open_backend<B: DiffBackend>() -> B {
    let art = artifacts_dir();
    let root = root_dir();
    B::open(
        &art.join("model_weights.bin"),
        &art.join("model_weights.json"),
        &art.join("vocab.bin"),
        &root,
        /* experts_per_tok */ 4,
        /* use_2bit */ false,
    )
}

// ---------------------------------------------------------------------------
// Phase 0 sanity test — validates the harness itself
// ---------------------------------------------------------------------------

/// Phase 0 scaffold-validation test. Opens the C backend, runs a
/// single prefill, and exercises the comparison helpers (against
/// itself — the only way to get a 1.0 cosine / Jaccard with a
/// non-deterministic backend is to compare logits to themselves).
///
/// Does not assert any equality between *different* C calls. See
/// the file-level docs for why end-to-end logits aren't a useful
/// oracle: the C path is non-deterministic across `memory_clear`,
/// and both `mf_free_model + Ctx::open` and `memory_clear` leave
/// process-global state we don't control. The real diff strategy
/// (Phase 3+) compares intermediate tensors at layer boundaries.
/// Smoke-test the Rust `WeightFile` against the real A3B artifacts.
/// Loads the manifest + mmap, asserts tensor count matches what the
/// C path's `[manifest]` log line reports (1397 tensors for A3B),
/// and that a couple of well-known tensors are present with the
/// expected dtype.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn weight_file_loads_a3b() {
    let art = artifacts_dir();
    let wf = moeflux::riir::WeightFile::open(
        &art.join("model_weights.bin"),
        &art.join("model_weights.json"),
    )
    .expect("WeightFile::open");
    eprintln!(
        "[diff:weight_file] {} tensors in {:.2} GB",
        wf.len(),
        wf.file_size() as f64 / 1e9,
    );

    // 1397 is the value the C `[manifest]` log prints for A3B.
    assert_eq!(wf.len(), 1397, "tensor count drifted from C");

    // The token-embedding tensor exists in every Qwen MoE export.
    let embed = wf
        .tensor_info("model.embed_tokens.weight")
        .expect("model.embed_tokens.weight");
    assert!(!embed.dtype.is_empty(), "embed_tokens dtype empty");
    eprintln!(
        "[diff:weight_file] embed_tokens dtype={} shape={:?} bits={} size={}",
        embed.dtype, embed.shape, embed.bits, embed.size,
    );
    let bytes = wf
        .tensor_bytes("model.embed_tokens.weight")
        .expect("embed bytes");
    assert_eq!(bytes.len() as u64, embed.size);
}

/// Cross-check that the pure-Rust `riir::VARIANT` shape constants
/// agree with the C runtime values for the same Cargo feature.
/// Catches drift between `model_variant.h` and `riir/variants.rs`.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn variants_match_c() {
    let c: CBackend = open_backend();
    common::c_backend::assert_matches_c(&c.0);
    eprintln!(
        "[diff:variants] {} n_vocab={} n_ctx={} eos={}",
        c.model_name(),
        c.n_vocab(),
        c.n_ctx(),
        c.eos(),
    );
}

/// ULP-bounded diff for the RoPE kernel (Phase 3, slice 3).
///
/// Not bit-exact — see `riir/rope.rs` module docs. Two compiler
/// choices compound: (1) Apple clang at `-O3` auto-vectorizes the
/// scalar `cosf`/`sinf` calls through Apple's libm vector variants,
/// while Rust extern-`"C"` calls don't vectorize; (2) clang with
/// `-ffp-contract=on` fuses the rotation's `q0*cos_a - q1*sin_a`
/// into FMA instructions, which Rust's plain `*` / `-` do not.
///
/// Both are compiler-choice artifacts, not porting bugs. We set the
/// threshold to 128 ULPs — well above the observed ~30 ULP drift
/// from FMA-vs-non-FMA but still tight enough to flag any real
/// algorithmic bug (which would produce thousands of ULPs of drift,
/// or NaN/inf, or sign flips).
///
/// Relative-error perspective: 128 ULPs at f32-near-1.0 ≈ 1.5e-5,
/// far below the noise floor of the surrounding 4-bit-quantized
/// weights and bf16 scales the rest of the model uses.
const MAX_ULP_DRIFT: u32 = 128;

/// Distance between two `f32` values measured in ULPs (units in last
/// place). Same-sign only; sign disagreement returns u32::MAX so it
/// fails any reasonable bound.
fn ulp_diff(a: f32, b: f32) -> u32 {
    if a.to_bits() == b.to_bits() {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    if a.is_sign_negative() != b.is_sign_negative() {
        return u32::MAX;
    }
    let ai = a.to_bits();
    let bi = b.to_bits();
    if ai > bi { ai - bi } else { bi - ai }
}

#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn rope_close_c_vs_rust() {
    use moeflux::riir::VARIANT;

    let c: CBackend = open_backend();
    let rs: RsBackend = open_backend();

    let head_dim = VARIANT.head_dim;
    let q_len = VARIANT.num_attn_heads * head_dim;
    let k_len = VARIANT.num_kv_heads * head_dim;

    let q_in: Vec<f32> = (0..q_len)
        .map(|i| ((i as f32) * 0.013).sin() * 0.7 + 0.1)
        .collect();
    let k_in: Vec<f32> = (0..k_len)
        .map(|i| ((i as f32) * 0.019).cos() * 0.5 - 0.2)
        .collect();

    let positions: [i32; 5] = [0, 1, 17, 1024, 65535];
    for &pos in &positions {
        let (c_q, c_k) = c.apply_rotary_emb(pos, &q_in, &k_in);
        let (rs_q, rs_k) = rs.apply_rotary_emb(pos, &q_in, &k_in);

        let q_max_ulp = c_q
            .iter()
            .zip(rs_q.iter())
            .map(|(&a, &b)| ulp_diff(a, b))
            .max()
            .unwrap_or(0);
        let k_max_ulp = c_k
            .iter()
            .zip(rs_k.iter())
            .map(|(&a, &b)| ulp_diff(a, b))
            .max()
            .unwrap_or(0);
        let q_diff_count = c_q
            .iter()
            .zip(rs_q.iter())
            .filter(|&(&a, &b)| a.to_bits() != b.to_bits())
            .count();
        let k_diff_count = c_k
            .iter()
            .zip(rs_k.iter())
            .filter(|&(&a, &b)| a.to_bits() != b.to_bits())
            .count();

        eprintln!(
            "[diff:rope pos={pos}] Q max_ulp={q_max_ulp} ({}/{} differ); \
             K max_ulp={k_max_ulp} ({}/{} differ)",
            q_diff_count,
            c_q.len(),
            k_diff_count,
            c_k.len(),
        );

        assert!(
            q_max_ulp <= MAX_ULP_DRIFT,
            "[diff:rope pos={pos}] Q max ULP drift {q_max_ulp} > {MAX_ULP_DRIFT}"
        );
        assert!(
            k_max_ulp <= MAX_ULP_DRIFT,
            "[diff:rope pos={pos}] K max ULP drift {k_max_ulp} > {MAX_ULP_DRIFT}"
        );
    }
}

/// Bit-exact diff for the CPU RMSNorm kernel (Phase 3, slice 2).
///
/// Composes on top of slice 1: feed the embedding output of several
/// token IDs through RMSNorm and compare bit-for-bit against the C
/// path. Three weight tensors exercised:
///
/// - `model.norm.weight` (the final RMSNorm before LM head)
/// - `model.layers.0.input_layernorm.weight` (per-layer attention
///   input norm)
/// - `model.layers.0.post_attention_layernorm.weight` (per-layer MLP
///   input norm)
///
/// All three are BF16 weight tensors of length `HIDDEN_DIM`. CPU
/// RMSNorm is deterministic on a given machine, so we expect every
/// output element to be byte-identical between C and Rust.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn rms_norm_cpu_bit_exact_c_vs_rust() {
    use moeflux::riir::VARIANT;

    let c: CBackend = open_backend();
    let rs: RsBackend = open_backend();

    let token_ids: [i32; 4] = [0, 1, VARIANT.eos_token_1, VARIANT.vocab_size as i32 - 1];
    let weight_names: [&str; 3] = [
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    ];

    for &tok in &token_ids {
        let x = c.embed(tok);
        for w_name in &weight_names {
            let c_out = c.rms_norm_cpu(w_name, &x);
            let rs_out = rs.rms_norm_cpu(w_name, &x);

            let diffs: Vec<(usize, f32, f32)> = c_out
                .iter()
                .zip(rs_out.iter())
                .enumerate()
                .filter_map(|(i, (&a, &b))| {
                    if a.to_bits() != b.to_bits() {
                        Some((i, a, b))
                    } else {
                        None
                    }
                })
                .collect();

            if !diffs.is_empty() {
                let first = &diffs[0];
                panic!(
                    "[diff:rms_norm_cpu token={tok} w={w_name}] {} of {} elements differ; \
                     first at index {} (c={} rs={})",
                    diffs.len(),
                    c_out.len(),
                    first.0,
                    first.1,
                    first.2,
                );
            }

            eprintln!(
                "[diff:rms_norm_cpu token={tok} w={w_name}] {} elements bit-equal; \
                 first 4: {:?}",
                c_out.len(),
                &c_out[..4],
            );
        }
    }
}

/// ULP-bounded diff for the SDPA core (Phase 3, slice 5).
///
/// `cpu_sdpa` does scaled dot-product attention + sigmoid gating in
/// one pass: per-head GQA dot product Q·K, scale, softmax, weighted
/// sum of V, then `out *= sigmoid(q_gate)`. Two libm calls per output
/// element (`expf` in softmax, `expf` in sigmoid) put it firmly in
/// ULP-bounded territory — same compiler-choice considerations as RoPE
/// (auto-vectorized libm vector variants on the C side, scalar libm
/// from Rust extern `"C"` calls), but compounded across more
/// reductions and more transcendental calls per element.
///
/// Synthetic Q/K/V buffers at three `kv_len` points to characterize
/// how drift scales with cache length — the load-bearing assertion
/// here is "Rust output and C output have the same shape" (high
/// cosine similarity, small max-abs-diff relative to the magnitude of
/// the output) rather than "every element matches to N ULPs," because
/// the compounded transcendental calls easily push individual
/// elements past any tight ULP bound.
///
/// Cosine ≥ 0.9999 catches porting bugs (sign flips, GQA mis-grouping,
/// off-by-one indexing all collapse it well below this), and max-abs
/// drift / max-abs output ≤ 1e-3 catches localized blow-ups while
/// staying well above the per-element FMA-vs-non-FMA noise floor.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn sdpa_cpu_close_c_vs_rust() {
    use moeflux::riir::VARIANT;

    let c: CBackend = open_backend();
    let rs: RsBackend = open_backend();

    let head_dim = VARIANT.head_dim;
    let num_attn_heads = VARIANT.num_attn_heads;
    let num_kv_heads = VARIANT.num_kv_heads;
    let q_dim = num_attn_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    // Synthetic per-position vectors. Magnitudes loosely match
    // post-RMSNorm + post-RoPE statistics observed in practice.
    let q: Vec<f32> = (0..q_dim)
        .map(|i| ((i as f32) * 0.011).sin() * 0.6 + 0.05)
        .collect();
    let q_gate: Vec<f32> = (0..q_dim)
        .map(|i| ((i as f32) * 0.007).cos() * 1.2 - 0.1)
        .collect();

    // Probe at small / medium / longer cache to see how drift scales.
    for &kv_len in &[1i32, 8, 64, 512] {
        let kv_len_u = kv_len as usize;
        let kv_total = kv_len_u * kv_dim;
        let k_cache: Vec<f32> = (0..kv_total)
            .map(|i| ((i as f32) * 0.013).sin() * 0.5 + 0.02)
            .collect();
        let v_cache: Vec<f32> = (0..kv_total)
            .map(|i| ((i as f32) * 0.017).cos() * 0.4 - 0.05)
            .collect();

        let c_out = c.sdpa_cpu(kv_len, &q, &q_gate, &k_cache, &v_cache);
        let rs_out = rs.sdpa_cpu(kv_len, &q, &q_gate, &k_cache, &v_cache);

        let max_ulp = c_out
            .iter()
            .zip(rs_out.iter())
            .map(|(&a, &b)| ulp_diff(a, b))
            .max()
            .unwrap_or(0);
        let diff_count = c_out
            .iter()
            .zip(rs_out.iter())
            .filter(|&(&a, &b)| a.to_bits() != b.to_bits())
            .count();
        let max_abs_diff = c_out
            .iter()
            .zip(rs_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_abs_out = c_out
            .iter()
            .map(|&a| a.abs())
            .fold(0.0f32, f32::max);
        let cos = cosine_sim(&c_out, &rs_out);

        eprintln!(
            "[diff:sdpa kv_len={kv_len}] cosine={cos:.6} max_abs_diff={max_abs_diff:.3e} \
             max_abs_out={max_abs_out:.3e} max_ulp={max_ulp} ({}/{} differ)",
            diff_count,
            c_out.len(),
        );

        assert!(
            cos >= 0.9999,
            "[diff:sdpa kv_len={kv_len}] cosine sim {cos:.6} below 0.9999"
        );
        assert!(
            max_abs_diff <= 1e-3 * max_abs_out.max(1e-6),
            "[diff:sdpa kv_len={kv_len}] max abs diff {max_abs_diff:.3e} \
             > 1e-3 * max abs out ({max_abs_out:.3e})"
        );
    }
}

/// Bit-exact diff for the per-head Q/K RMSNorm kernel (Phase 3, slice 4).
///
/// The per-head variant fires inside `full_attention_forward` after the
/// QKV projection — each Q head's `head_dim`-long slice (and each K
/// head's slice) gets RMS-normed independently against a shared
/// `head_dim`-long bf16 weight. The arithmetic is the same shape as
/// the whole-vector `cpu_rms_norm` (sequential sum-of-squares,
/// `1/sqrt(.)`, multiply-by-bf16-weight), so we expect bit-exact
/// agreement on the same hardware.
///
/// Two weight tensors exercised — Q and K norms for layer 0 (the only
/// full-attention layer in the first interval; both are bf16 of length
/// `HEAD_DIM`). Layer 0's norms are guaranteed present whether the
/// model variant places its first full-attn layer at index 0 or later,
/// since these tensors exist for every full-attention layer.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn rms_norm_per_head_cpu_bit_exact_c_vs_rust() {
    use moeflux::riir::VARIANT;

    let c: CBackend = open_backend();
    let rs: RsBackend = open_backend();

    let head_dim = VARIANT.head_dim;
    let num_attn_heads = VARIANT.num_attn_heads;
    let num_kv_heads = VARIANT.num_kv_heads;

    // Find the first full-attention layer index. The variant's
    // FULL_ATTN_INTERVAL determines this; layer i is full-attn iff
    // (i+1) % FULL_ATTN_INTERVAL == 0. For Qwen3 MoE this is layer 3
    // (interval 4) — i.e. the 4th, 8th, ... layers.
    let mut fa_layer: Option<usize> = None;
    for i in 0..VARIANT.num_layers {
        if (i + 1) % VARIANT.full_attn_interval == 0 {
            fa_layer = Some(i);
            break;
        }
    }
    let fa_layer = fa_layer.expect("at least one full-attention layer");
    let q_norm_name =
        format!("model.layers.{fa_layer}.self_attn.q_norm.weight");
    let k_norm_name =
        format!("model.layers.{fa_layer}.self_attn.k_norm.weight");

    // Synthetic per-head input vectors — large enough to exercise the
    // sum-of-squares reduction across all heads, with non-trivial
    // magnitudes so the inv_rms term is interesting.
    let q_in: Vec<f32> = (0..num_attn_heads * head_dim)
        .map(|i| ((i as f32) * 0.011).sin() * 0.6 + 0.05)
        .collect();
    let k_in: Vec<f32> = (0..num_kv_heads * head_dim)
        .map(|i| ((i as f32) * 0.023).cos() * 0.4 - 0.1)
        .collect();

    for (label, w_name, num_heads, x_in) in [
        ("Q", q_norm_name.as_str(), num_attn_heads, &q_in),
        ("K", k_norm_name.as_str(), num_kv_heads, &k_in),
    ] {
        let c_out = c.rms_norm_per_head_cpu(w_name, num_heads, head_dim, x_in);
        let rs_out = rs.rms_norm_per_head_cpu(w_name, num_heads, head_dim, x_in);

        let diffs: Vec<(usize, f32, f32)> = c_out
            .iter()
            .zip(rs_out.iter())
            .enumerate()
            .filter_map(|(i, (&a, &b))| {
                if a.to_bits() != b.to_bits() {
                    Some((i, a, b))
                } else {
                    None
                }
            })
            .collect();

        if !diffs.is_empty() {
            let first = &diffs[0];
            panic!(
                "[diff:rms_norm_per_head_cpu {label} layer={fa_layer}] {} of {} elements differ; \
                 first at index {} (c={} rs={})",
                diffs.len(),
                c_out.len(),
                first.0,
                first.1,
                first.2,
            );
        }

        eprintln!(
            "[diff:rms_norm_per_head_cpu {label} layer={fa_layer} \
             num_heads={num_heads} head_dim={head_dim}] {} elements bit-equal; \
             first 4: {:?}",
            c_out.len(),
            &c_out[..4],
        );
    }
}

/// Bit-exact diff for the embedding kernel (Phase 3, slice 1).
///
/// The embedding lookup is fully deterministic CPU code on both
/// sides — quantized 4-bit dequant + bf16-via-bit-shift. Outputs
/// must match to the last bit. Running both backends against the
/// same mmap'd weight blob, anything but byte-equal output indicates
/// a porting bug.
///
/// Picks 8 token IDs spanning the vocabulary's interesting points:
/// 0 (BOS in most setups), 1, the canonical EOS, the secondary EOS,
/// the think-start / think-end specials, a mid-vocabulary id, and
/// `vocab_size - 1`. Out-of-range ids are tested separately for the
/// error path.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn embed_bit_exact_c_vs_rust() {
    use moeflux::riir::VARIANT;

    let c: CBackend = open_backend();
    let rs: RsBackend = open_backend();

    let vocab = VARIANT.vocab_size as i32;
    let token_ids: [i32; 8] = [
        0,
        1,
        VARIANT.eos_token_1,
        VARIANT.eos_token_2,
        VARIANT.think_start_token,
        VARIANT.think_end_token,
        vocab / 2,
        vocab - 1,
    ];

    for &tok in &token_ids {
        let c_emb = c.embed(tok);
        let rs_emb = rs.embed(tok);
        assert_eq!(
            c_emb.len(),
            VARIANT.hidden_dim,
            "[diff:embed token={tok}] C output length mismatch"
        );
        assert_eq!(
            rs_emb.len(),
            VARIANT.hidden_dim,
            "[diff:embed token={tok}] Rust output length mismatch"
        );

        let diffs: Vec<(usize, f32, f32)> = c_emb
            .iter()
            .zip(rs_emb.iter())
            .enumerate()
            .filter_map(|(i, (&a, &b))| {
                if a.to_bits() != b.to_bits() {
                    Some((i, a, b))
                } else {
                    None
                }
            })
            .collect();

        if !diffs.is_empty() {
            let first = &diffs[0];
            panic!(
                "[diff:embed token={tok}] {} of {} elements differ; \
                 first at index {} (c={} rs={})",
                diffs.len(),
                c_emb.len(),
                first.0,
                first.1,
                first.2,
            );
        }

        eprintln!(
            "[diff:embed token={tok}] {} elements bit-equal; first 4: {:?}",
            c_emb.len(),
            &c_emb[..4],
        );
    }
}

/// Bit-exact diff for the LM head matvec (Phase 3, slice 6).
///
/// Symmetric to the embedding kernel: 4-bit dequant + per-group bf16
/// scale/bias, but as a fused matvec rather than a single-row lookup.
/// `mf_lm_head_cpu` routes through `cpu_dequant_matvec` regardless of
/// Metal availability, so both backends do deterministic CPU work
/// against the same mmap'd weight blob.
///
/// Two synthetic hidden vectors are run through the matvec — a
/// trig-modulated wave (the same shape used by RoPE / SDPA tests) and
/// a hidden state derived from the embedding-then-RMSNorm pipeline of
/// a real token, so we exercise both adversarial and realistic input
/// magnitudes. Bit-exactness is the target; the lazy-mode floor is
/// "no diffs at all," and any drift falls back to the same per-element
/// reporting the embedding test uses.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn lm_head_cpu_bit_exact_c_vs_rust() {
    use moeflux::riir::VARIANT;

    let c: CBackend = open_backend();
    let rs: RsBackend = open_backend();

    let hidden_dim = VARIANT.hidden_dim;

    // Input 1: synthetic trig-modulated hidden vector, magnitudes
    // similar to post-norm hidden states.
    let x_synth: Vec<f32> = (0..hidden_dim)
        .map(|i| ((i as f32) * 0.011).sin() * 0.6 + 0.05)
        .collect();

    // Input 2: realistic hidden derived from embedding(EOS) → final norm.
    // Composes on top of slices 1 + 2, so a regression here is more
    // likely to be in the LM head than upstream.
    let emb = c.embed(VARIANT.eos_token_1);
    let x_real = c.rms_norm_cpu("model.norm.weight", &emb);

    for (label, x) in [("synth", &x_synth), ("real", &x_real)] {
        let c_out = c.lm_head_cpu(x);
        let rs_out = rs.lm_head_cpu(x);

        assert_eq!(c_out.len(), VARIANT.vocab_size, "C output len");
        assert_eq!(rs_out.len(), VARIANT.vocab_size, "Rust output len");

        let diffs: Vec<(usize, f32, f32)> = c_out
            .iter()
            .zip(rs_out.iter())
            .enumerate()
            .filter_map(|(i, (&a, &b))| {
                if a.to_bits() != b.to_bits() {
                    Some((i, a, b))
                } else {
                    None
                }
            })
            .collect();

        if !diffs.is_empty() {
            let max_ulp = c_out
                .iter()
                .zip(rs_out.iter())
                .map(|(&a, &b)| ulp_diff(a, b))
                .max()
                .unwrap_or(0);
            let max_abs_diff = c_out
                .iter()
                .zip(rs_out.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let max_abs_out = c_out
                .iter()
                .map(|&a| a.abs())
                .fold(0.0f32, f32::max);
            let cos = cosine_sim(&c_out, &rs_out);
            let first = &diffs[0];
            panic!(
                "[diff:lm_head {label}] {} of {} elements differ; \
                 max_ulp={max_ulp} max_abs_diff={max_abs_diff:.3e} \
                 max_abs_out={max_abs_out:.3e} cosine={cos:.6}; \
                 first at index {} (c={} rs={})",
                diffs.len(),
                c_out.len(),
                first.0,
                first.1,
                first.2,
            );
        }

        eprintln!(
            "[diff:lm_head {label}] {} elements bit-equal; \
             argmax={} first 4: {:?}",
            c_out.len(),
            argmax(&c_out),
            &c_out[..4],
        );
    }
}

/// ULP-bounded diff for the MoE router (Phase 3, slice 7).
///
/// Pipeline: softmax → top-K → normalize. The softmax's per-element
/// `expf` is the only ULP source — Apple clang `-O3` auto-vectorizes
/// scalar `expf` calls into Apple libm's `vexpf`, while Rust extern
/// `"C"` calls don't, so we expect single-digit ULP drift on the
/// softmaxed scores. The selected expert *index set* should be
/// identical across both sides — top-K selection is deterministic
/// once you have identical inputs, and the softmax ULP gap is
/// orders of magnitude below the score separation that would change
/// which experts are picked.
///
/// Two synthetic gate-score patterns:
///
/// 1. **clear-winner**: NUM_EXPERTS scores with K large bumps
///    several orders of magnitude above the noise floor. The set of
///    top-K indices must match exactly.
/// 2. **mild-spread**: scores drawn from a smooth trig sweep. The
///    set match is still expected (gaps still much wider than ULP
///    drift) but stresses tighter gaps in the top-K cutoff zone.
///
/// Weights compared after canonicalizing slot order by index — the
/// C selection-sort produces a slot ordering that depends on the
/// historical replacement order, which is itself input-dependent
/// but identical across our two backends since both walk the score
/// array in the same order.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn moe_router_cpu_close_c_vs_rust() {
    use moeflux::riir::VARIANT;

    let c: CBackend = open_backend();
    let rs: RsBackend = open_backend();

    let n_experts = VARIANT.num_experts;
    let k = VARIANT.num_experts_per_tok;
    assert!(n_experts >= k && k >= 1);

    // Pattern 1: clear-winner. Smooth low-magnitude background plus K
    // large bumps placed at deterministic indices.
    let mut clear: Vec<f32> = (0..n_experts)
        .map(|i| ((i as f32) * 0.011).sin() * 0.01 + 0.0)
        .collect();
    let bump_indices: Vec<usize> =
        (0..k).map(|i| (i * (n_experts / k.max(1))) % n_experts).collect();
    for (slot, &idx) in bump_indices.iter().enumerate() {
        clear[idx] = 5.0 + slot as f32 * 0.5;
    }

    // Pattern 2: mild-spread. All scores within a narrower band so
    // the top-K cutoff is closer to the noise floor.
    let spread: Vec<f32> = (0..n_experts)
        .map(|i| ((i as f32) * 0.013).cos() * 0.7 + ((i as f32) * 0.041).sin() * 0.3)
        .collect();

    for (label, scores) in [("clear", &clear), ("spread", &spread)] {
        let (c_idx, c_w) = c.moe_router_cpu(scores.clone(), k);
        let (rs_idx, rs_w) = rs.moe_router_cpu(scores.clone(), k);

        // Set equality on indices.
        let mut c_sorted = c_idx.clone();
        let mut rs_sorted = rs_idx.clone();
        c_sorted.sort();
        rs_sorted.sort();
        assert_eq!(
            c_sorted, rs_sorted,
            "[diff:moe_router {label}] index set mismatch (c={c_idx:?} rs={rs_idx:?})"
        );

        // Map C/Rust slot → score for canonical comparison by index.
        let c_pairs: Vec<(i32, f32)> =
            c_idx.iter().copied().zip(c_w.iter().copied()).collect();
        let rs_pairs: Vec<(i32, f32)> =
            rs_idx.iter().copied().zip(rs_w.iter().copied()).collect();
        let mut c_by_idx: std::collections::HashMap<i32, f32> = c_pairs.into_iter().collect();
        let mut rs_by_idx: std::collections::HashMap<i32, f32> = rs_pairs.into_iter().collect();

        let mut max_ulp = 0u32;
        let mut max_abs_diff = 0.0f32;
        for &idx in &c_sorted {
            let cw = c_by_idx.remove(&idx).unwrap();
            let rw = rs_by_idx.remove(&idx).unwrap();
            max_ulp = max_ulp.max(ulp_diff(cw, rw));
            max_abs_diff = max_abs_diff.max((cw - rw).abs());
        }
        let weight_sum_c: f32 = c_w.iter().sum();
        let weight_sum_rs: f32 = rs_w.iter().sum();

        eprintln!(
            "[diff:moe_router {label} n={n_experts} k={k}] \
             max_ulp={max_ulp} max_abs_diff={max_abs_diff:.3e} \
             c_sum={weight_sum_c:.6} rs_sum={weight_sum_rs:.6}"
        );

        // ULP bound: same regime as RoPE. Single libm call per element
        // in softmax, bounded per call; normalization is one further
        // sum + divide on K floats.
        assert!(
            max_ulp <= MAX_ULP_DRIFT,
            "[diff:moe_router {label}] max ULP drift {max_ulp} > {MAX_ULP_DRIFT}"
        );
        assert!(
            (weight_sum_c - 1.0).abs() < 1e-5,
            "[diff:moe_router {label}] C weights don't sum to 1 ({weight_sum_c})"
        );
        assert!(
            (weight_sum_rs - 1.0).abs() < 1e-5,
            "[diff:moe_router {label}] Rust weights don't sum to 1 ({weight_sum_rs})"
        );
    }
}

/// Bit-exact diff for bare RMSNorm (Phase 3, slice 8a-1).
///
/// Used inside `linear_attention_forward` to RMS-normalize each head
/// of Q and K (with no learned weight tensor — just a plain
/// normalization). Same arithmetic shape as the existing weighted
/// `cpu_rms_norm`, just without the bf16-multiply step. CPU,
/// deterministic, bit-exact territory.
///
/// Probe shape: a `LINEAR_KEY_DIM`-long head, since that's the
/// production callsite. Mid-magnitude trig input.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn rms_norm_bare_cpu_bit_exact_c_vs_rust() {
    use moeflux::riir::variants::Variant;

    let c: CBackend = open_backend();
    let rs: RsBackend = open_backend();

    let dim = Variant::LINEAR_KEY_DIM;
    let x: Vec<f32> = (0..dim)
        .map(|i| ((i as f32) * 0.011).sin() * 0.6 + 0.05)
        .collect();
    let eps = 1e-6f32;

    let c_out = c.rms_norm_bare_cpu(eps, &x);
    let rs_out = rs.rms_norm_bare_cpu(eps, &x);

    let diffs: Vec<(usize, f32, f32)> = c_out
        .iter()
        .zip(rs_out.iter())
        .enumerate()
        .filter_map(|(i, (&a, &b))| {
            if a.to_bits() != b.to_bits() {
                Some((i, a, b))
            } else {
                None
            }
        })
        .collect();

    if !diffs.is_empty() {
        let first = &diffs[0];
        panic!(
            "[diff:rms_norm_bare] {} of {} elements differ; \
             first at index {} (c={} rs={})",
            diffs.len(),
            c_out.len(),
            first.0,
            first.1,
            first.2,
        );
    }

    eprintln!(
        "[diff:rms_norm_bare dim={dim}] {} elements bit-equal; first 4: {:?}",
        c_out.len(),
        &c_out[..4],
    );
}

/// ULP-bounded diff for the conv1d step + SiLU (Phase 3, slice 8a-2).
///
/// Used inside `linear_attention_forward` to do the depthwise 1D conv
/// over the linear-attention (Q|K|V) channels with the previous
/// `(kernel_size-1)` time-steps of state. Tested against layer-0's
/// real `conv1d.weight` tensor so the bf16 decode path is exercised.
///
/// The dot-product half is bit-exact (matched via `mul_add` to clang's
/// FMA contraction) but the SiLU tail introduces one `expf` per
/// channel — same compiler-choice ULP territory as the rest of the
/// libm-bearing kernels. Tolerance: ULP-bounded with the same
/// `MAX_ULP_DRIFT = 128` budget RoPE uses.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn conv1d_step_cpu_close_c_vs_rust() {
    use moeflux::riir::variants::Variant;
    use moeflux::riir::VARIANT;

    let c: CBackend = open_backend();
    let rs: RsBackend = open_backend();

    let channels = VARIANT.linear_conv_dim();
    let kernel_size = Variant::CONV_KERNEL_SIZE;
    let weight_name = "model.layers.0.linear_attn.conv1d.weight";

    // (kernel_size-1) past time-steps × channels values.
    let conv_state: Vec<f32> = (0..(kernel_size - 1) * channels)
        .map(|i| ((i as f32) * 0.013).sin() * 0.4 + 0.02)
        .collect();
    let new_input: Vec<f32> = (0..channels)
        .map(|i| ((i as f32) * 0.019).cos() * 0.5 - 0.1)
        .collect();

    let c_out =
        c.conv1d_step_cpu(weight_name, channels, kernel_size, &conv_state, &new_input);
    let rs_out =
        rs.conv1d_step_cpu(weight_name, channels, kernel_size, &conv_state, &new_input);

    let max_ulp = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(&a, &b)| ulp_diff(a, b))
        .max()
        .unwrap_or(0);
    let diff_count = c_out
        .iter()
        .zip(rs_out.iter())
        .filter(|&(&a, &b)| a.to_bits() != b.to_bits())
        .count();
    let max_abs_diff = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_abs_out =
        c_out.iter().map(|&a| a.abs()).fold(0.0f32, f32::max);

    eprintln!(
        "[diff:conv1d_step channels={channels} kernel={kernel_size}] \
         max_ulp={max_ulp} max_abs_diff={max_abs_diff:.3e} \
         max_abs_out={max_abs_out:.3e} ({diff_count}/{} differ)",
        c_out.len(),
    );

    assert!(
        max_ulp <= MAX_ULP_DRIFT,
        "[diff:conv1d_step] max ULP drift {max_ulp} > {MAX_ULP_DRIFT}"
    );
}

/// ULP-bounded diff for RMSNormGated (Phase 3, slice 8a-3).
///
/// Used inside `linear_attention_forward` to apply `rms_norm × silu × weight`
/// to each per-head output of the gate-delta-net recurrence. Tested
/// against layer-0's real `linear_attn.norm.weight` tensor.
///
/// One libm `expf` per element via SiLU, plus the bf16 weight decode
/// the existing weighted RMSNorm already exercises. Same ULP budget
/// as the other libm-bearing kernels.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn rms_norm_gated_cpu_close_c_vs_rust() {
    use moeflux::riir::variants::Variant;

    let c: CBackend = open_backend();
    let rs: RsBackend = open_backend();

    let dim = Variant::LINEAR_VALUE_DIM;
    let weight_name = "model.layers.0.linear_attn.norm.weight";
    let eps = 1e-6f32;

    let x: Vec<f32> = (0..dim)
        .map(|i| ((i as f32) * 0.011).sin() * 0.6 + 0.05)
        .collect();
    let z: Vec<f32> = (0..dim)
        .map(|i| ((i as f32) * 0.017).cos() * 1.2 - 0.1)
        .collect();

    let c_out = c.rms_norm_gated_cpu(weight_name, eps, &x, &z);
    let rs_out = rs.rms_norm_gated_cpu(weight_name, eps, &x, &z);

    let max_ulp = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(&a, &b)| ulp_diff(a, b))
        .max()
        .unwrap_or(0);
    let diff_count = c_out
        .iter()
        .zip(rs_out.iter())
        .filter(|&(&a, &b)| a.to_bits() != b.to_bits())
        .count();
    let max_abs_diff = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_abs_out =
        c_out.iter().map(|&a| a.abs()).fold(0.0f32, f32::max);

    eprintln!(
        "[diff:rms_norm_gated dim={dim}] max_ulp={max_ulp} \
         max_abs_diff={max_abs_diff:.3e} max_abs_out={max_abs_out:.3e} \
         ({diff_count}/{} differ)",
        c_out.len(),
    );

    assert!(
        max_ulp <= MAX_ULP_DRIFT,
        "[diff:rms_norm_gated] max ULP drift {max_ulp} > {MAX_ULP_DRIFT}"
    );
}

/// ULP-bounded diff for the gated-delta-net recurrence (Phase 3, slice 8b).
///
/// The novel kernel inside `linear_attention_forward`. Arithmetic
/// shape per v-head:
///
///   g       = exp(-exp(A_log) * softplus(alpha + dt_bias))
///   b_gate  = sigmoid(beta)
///   S      *= g
///   for vi:
///     kv_mem = Σ_ki S[vi,ki] * k[ki]
///     delta  = (v[vi] - kv_mem) * b_gate
///     S[vi]+= k * delta
///   for vi:
///     out[vi] = Σ_ki S[vi,ki] * q[ki]
///
/// Three FMA-shaped contractions in the per-vi inner loops; all use
/// `mul_add` on the Rust side to match clang's AArch64 codegen
/// (per the LM head finding). The per-head precomputation has libm
/// `expf`/`logf`/`sigmoid` calls — same scalar-libm regime as the
/// other linear-attn primitives.
///
/// Probe shape: layer 0 (linear-attn under FULL_ATTN_INTERVAL=4), one
/// step from a zero initial state. Tests both the post-step state
/// (the in-place mutation) and the per-head output values.
///
/// Compares both `ssm_state` and `out_values` element-wise. Tolerance:
/// ULP-bounded with the shared `MAX_ULP_DRIFT = 128` budget; if it
/// lands tighter (bit-exact, like the 8a primitives), the eprintln
/// will say so.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn gated_delta_recurrence_cpu_close_c_vs_rust() {
    use moeflux::riir::variants::Variant;
    use moeflux::riir::VARIANT;

    let c: CBackend = open_backend();
    let rs: RsBackend = open_backend();

    let v_heads = VARIANT.linear_num_v_heads;
    let k_heads = VARIANT.linear_num_k_heads;
    let key_dim = Variant::LINEAR_KEY_DIM;
    let value_dim = Variant::LINEAR_VALUE_DIM;
    let layer_idx = 0usize; // layer 0 is linear-attn (full_attn_interval=4)

    // Synthetic per-step inputs, magnitudes matching post-conv-and-bare-norm
    // statistics. alpha/beta land near zero so softplus and sigmoid have
    // non-degenerate working ranges.
    let alpha: Vec<f32> = (0..v_heads)
        .map(|i| ((i as f32) * 0.013).sin() * 0.3)
        .collect();
    let beta: Vec<f32> = (0..v_heads)
        .map(|i| ((i as f32) * 0.017).cos() * 0.5)
        .collect();
    let q: Vec<f32> = (0..k_heads * key_dim)
        .map(|i| ((i as f32) * 0.011).sin() * 0.4 + 0.05)
        .collect();
    let k_in: Vec<f32> = (0..k_heads * key_dim)
        .map(|i| ((i as f32) * 0.019).cos() * 0.5 - 0.1)
        .collect();
    let v_in: Vec<f32> = (0..v_heads * value_dim)
        .map(|i| ((i as f32) * 0.023).sin() * 0.6 + 0.02)
        .collect();
    let ssm_state_in: Vec<f32> = vec![0.0f32; v_heads * value_dim * key_dim];

    let (c_state, c_out) = c.gated_delta_recurrence_cpu(
        layer_idx, &alpha, &beta, &q, &k_in, &v_in,
        v_heads, k_heads, key_dim, value_dim, ssm_state_in.clone(),
    );
    let (rs_state, rs_out) = rs.gated_delta_recurrence_cpu(
        layer_idx, &alpha, &beta, &q, &k_in, &v_in,
        v_heads, k_heads, key_dim, value_dim, ssm_state_in,
    );

    let state_max_ulp = c_state
        .iter()
        .zip(rs_state.iter())
        .map(|(&a, &b)| ulp_diff(a, b))
        .max()
        .unwrap_or(0);
    let state_diff_count = c_state
        .iter()
        .zip(rs_state.iter())
        .filter(|&(&a, &b)| a.to_bits() != b.to_bits())
        .count();
    let out_max_ulp = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(&a, &b)| ulp_diff(a, b))
        .max()
        .unwrap_or(0);
    let out_diff_count = c_out
        .iter()
        .zip(rs_out.iter())
        .filter(|&(&a, &b)| a.to_bits() != b.to_bits())
        .count();
    let out_max_abs = c_out
        .iter()
        .map(|&a| a.abs())
        .fold(0.0f32, f32::max);
    let out_max_diff = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!(
        "[diff:gated_delta_recurrence layer={layer_idx} v_heads={v_heads} \
         k_heads={k_heads} key_dim={key_dim} value_dim={value_dim}] \
         state max_ulp={state_max_ulp} ({}/{} differ); \
         out max_ulp={out_max_ulp} max_abs_diff={:.3e} max_abs_out={:.3e} \
         ({}/{} differ)",
        state_diff_count,
        c_state.len(),
        out_max_diff,
        out_max_abs,
        out_diff_count,
        c_out.len(),
    );

    assert!(
        state_max_ulp <= MAX_ULP_DRIFT,
        "[diff:gated_delta_recurrence] state max ULP drift \
         {state_max_ulp} > {MAX_ULP_DRIFT}"
    );
    assert!(
        out_max_ulp <= MAX_ULP_DRIFT,
        "[diff:gated_delta_recurrence] out max ULP drift \
         {out_max_ulp} > {MAX_ULP_DRIFT}"
    );
}

/// Phase 3 slice 9a — single-expert GPU FFN forward.
///
/// Both backends consume identical synthetic bytes (PRNG-seeded 4-bit
/// weights with BF16 0x3C00 scales / 0 biases per `expert_forward::synth`)
/// and run the same four Metal pipelines (`dequant_matvec_4bit_v3` ×3
/// + `swiglu_fused`). First GPU kernel under diff — Metal SIMD-reduce
/// ordering is not specified to be deterministic across pipeline-state
/// recompiles, so the tolerance regime is cosine/Jaccard rather than
/// bit-exact / ULP-bounded.
///
/// Empirically tighter floors than the end-to-end `COSINE_SIM_MIN`
/// because there's only one stage of nondeterminism stacked here, not
/// 60 layers' worth.
#[test]
#[ignore = "long running; needs Metal device + moeflux artifacts"]
fn gpu_expert_forward_close_c_vs_rust() {
    use moeflux::riir::expert_forward::synth;
    use moeflux::riir::VARIANT;

    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();

    let expert_data = synth::expert_data_seeded();
    let h_post = synth::h_post_seeded();
    assert_eq!(expert_data.len(), VARIANT.expert_size_4bit());
    assert_eq!(h_post.len(), VARIANT.hidden_dim);

    let c_out = c.gpu_expert_forward(&expert_data, &h_post);
    let rs_out = rs.gpu_expert_forward(&expert_data, &h_post);
    assert_eq!(c_out.len(), VARIANT.hidden_dim);
    assert_eq!(rs_out.len(), VARIANT.hidden_dim);

    let cos = cosine_sim(&c_out, &rs_out);
    let max_abs_out = c_out
        .iter()
        .chain(rs_out.iter())
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    let max_abs_diff = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = if max_abs_out > 0.0 {
        max_abs_diff / max_abs_out
    } else {
        0.0
    };

    eprintln!(
        "[diff:gpu_expert_forward] cosine={cos:.7} \
         max_abs_diff={max_abs_diff:.3e} max_abs_out={max_abs_out:.3e} \
         relative={rel:.3e}"
    );

    // Sanity — outputs must be finite and not all-zero. If either side
    // bombs we've got something more than nondeterminism going on.
    assert!(
        c_out.iter().all(|x| x.is_finite()),
        "[diff:gpu_expert_forward] C output has NaN/Inf"
    );
    assert!(
        rs_out.iter().all(|x| x.is_finite()),
        "[diff:gpu_expert_forward] Rust output has NaN/Inf"
    );
    assert!(
        c_out.iter().any(|&x| x != 0.0),
        "[diff:gpu_expert_forward] C output is all zero"
    );
    assert!(
        rs_out.iter().any(|&x| x != 0.0),
        "[diff:gpu_expert_forward] Rust output is all zero"
    );

    // Tolerances: cosine 0.9999 floor (one stage of Metal SIMD-reduce
    // nondeterminism); max_abs_diff scaled to 1e-3 of max output to
    // accept reorder-induced drift on small-magnitude entries.
    const COSINE_FLOOR: f32 = 0.9999;
    const REL_DIFF_FLOOR: f32 = 1e-3;
    assert!(
        cos >= COSINE_FLOOR,
        "[diff:gpu_expert_forward] cosine {cos:.7} below {COSINE_FLOOR}"
    );
    assert!(
        rel <= REL_DIFF_FLOOR,
        "[diff:gpu_expert_forward] relative max_abs_diff {rel:.3e} \
         above {REL_DIFF_FLOOR:.3e}"
    );
}

/// Phase 3 slice 9e — GPU RMSNorm fused chain.
///
/// `rms_norm_sum_sq` is the first kernel in the diff suite using
/// **threadgroup-shared memory across SIMD groups** (256 threads
/// reduce to 1 scalar via simd_sum + 32-element shared array +
/// second-stage simd_sum). If Metal nondeterminism is going to
/// engage, this is the most plausible spot in the slices ported so
/// far.
///
/// Test pattern: load real `model.norm.weight` bytes via the C side's
/// existing wrapper (the Rust port hasn't published the weight file
/// over its public API surface; we read from the C-side ctx since
/// both ctxs were opened over the same artifacts). Run the chain on
/// the same x on both backends; compare.
///
/// Empirical floors: cosine ≥ 0.9999, rel ≤ 1e-3 (the GPU "atomic-op
/// tolerance" envelope). If both kernels turn out to also be
/// deterministic-per-PSO (likely — `simd_sum` is documented as such,
/// the threadgroup-shared write/read is barrier-fenced), this lands
/// bit-exact like 9a/9b.
#[test]
#[ignore = "long running; needs Metal device + moeflux artifacts"]
fn gpu_rms_norm_fused_close_c_vs_rust() {
    use moeflux::riir::VARIANT;

    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();

    // Load `model.norm.weight` as raw bf16 bytes via the Rust
    // weight-file wrapper (which both backends opened over the same
    // artifacts). Only the bytes are needed; same on both sides.
    let art = artifacts_dir();
    let wf = moeflux::riir::WeightFile::open(
        &art.join("model_weights.bin"),
        &art.join("model_weights.json"),
    )
    .expect("WeightFile::open");
    let weight = wf
        .tensor_bytes("model.norm.weight")
        .expect("model.norm.weight present in manifest");
    assert_eq!(weight.len(), VARIANT.hidden_dim * 2);

    // Synthetic x: not symmetric around zero so sum_sq is meaningful.
    let x: Vec<f32> = (0..VARIANT.hidden_dim)
        .map(|i| (i as f32 * 0.013).sin() * 0.5 + 0.1)
        .collect();

    let c_out = c.gpu_rms_norm_fused(&x, weight);
    let rs_out = rs.gpu_rms_norm_fused(&x, weight);
    assert_eq!(c_out.len(), VARIANT.hidden_dim);
    assert_eq!(rs_out.len(), VARIANT.hidden_dim);

    let cos = cosine_sim(&c_out, &rs_out);
    let max_abs_out = c_out
        .iter()
        .chain(rs_out.iter())
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    let max_abs_diff = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = if max_abs_out > 0.0 {
        max_abs_diff / max_abs_out
    } else {
        0.0
    };

    eprintln!(
        "[diff:gpu_rms_norm_fused] cosine={cos:.7} \
         max_abs_diff={max_abs_diff:.3e} max_abs_out={max_abs_out:.3e} \
         relative={rel:.3e}"
    );

    assert!(
        c_out.iter().all(|x| x.is_finite()),
        "[diff:gpu_rms_norm_fused] C output has NaN/Inf"
    );
    assert!(
        rs_out.iter().all(|x| x.is_finite()),
        "[diff:gpu_rms_norm_fused] Rust output has NaN/Inf"
    );

    const COSINE_FLOOR: f32 = 0.9999;
    const REL_DIFF_FLOOR: f32 = 1e-3;
    assert!(
        cos >= COSINE_FLOOR,
        "[diff:gpu_rms_norm_fused] cosine {cos:.7} below {COSINE_FLOOR}"
    );
    assert!(
        rel <= REL_DIFF_FLOOR,
        "[diff:gpu_rms_norm_fused] relative max_abs_diff {rel:.3e} \
         above {REL_DIFF_FLOOR:.3e}"
    );
}

/// Phase 3 slice 9c — expert blob I/O.
///
/// Reads a few (layer, expert) pairs through both backends and
/// asserts byte-for-byte equality. The C path uses
/// `pread(ctx->layer_fds[layer_idx], …)`; the Rust path uses
/// `File::read_at(self.layers[layer_idx], …)`. Both end up issuing
/// a `pread64` against the same file at the same offset, so the
/// expected outcome is bit-exact.
///
/// Fixed sample of probe pairs — covers a full-attn layer (idx 3,
/// `(i+1) % 4 == 0`), a linear-attn layer (idx 0), and a couple of
/// mid-stack layers. Expert indices 0, 7, 255 stress the
/// first-block / mid-block / last-block byte regions of the file.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn load_expert_bytes_byte_exact_c_vs_rust() {
    use moeflux::riir::VARIANT;

    let c: CBackend = open_backend();
    let rs: RsBackend = open_backend();

    let v = VARIANT;
    let probes: &[(i32, i32)] = &[
        (0, 0),
        (0, 7),
        (3, 0),
        (3, 255),
        ((v.num_layers / 2) as i32, 42),
        ((v.num_layers - 1) as i32, (v.num_experts - 1) as i32),
    ];

    for &(layer, expert) in probes {
        let c_bytes = c.load_expert_bytes(layer, expert);
        let rs_bytes = rs.load_expert_bytes(layer, expert);
        assert_eq!(c_bytes.len(), v.expert_size_4bit());
        assert_eq!(rs_bytes.len(), v.expert_size_4bit());

        let first_diff =
            c_bytes.iter().zip(rs_bytes.iter()).position(|(a, b)| a != b);
        eprintln!(
            "[diff:load_expert_bytes layer={layer} expert={expert}] \
             {} bytes; first_diff={first_diff:?}",
            c_bytes.len(),
        );

        assert!(
            first_diff.is_none(),
            "[diff:load_expert_bytes layer={layer} expert={expert}] \
             byte mismatch at offset {first_diff:?} \
             (c=0x{:02x} rs=0x{:02x})",
            first_diff.map(|i| c_bytes[i]).unwrap_or(0),
            first_diff.map(|i| rs_bytes[i]).unwrap_or(0),
        );
    }
}

/// Phase 3 slice 9b — batched K-expert FFN + GPU combine.
///
/// K=4 synthetic experts (distinct PRNG seeds per slot), normalized
/// random weights, deterministic h_post / h_mid / shared_out, and
/// `shared_gate_score = -1.0` so `sigmoid(-1) ≈ 0.269` brings a
/// non-trivial fraction of `shared_out` into the combine.
///
/// Both backends consume identical bytes; the only difference is the
/// host-side encoding path (which buffers, which encoders, which
/// dispatch helper). `weighted_sum`/`moe_combine_residual` per-thread
/// loops have no atomic ops, so empirically this *might* land
/// bit-exact like 9a — but the cosine ≥ 0.9999 / rel ≤ 1e-3 floors
/// stay loose to absorb any genuine SIMD-reduce nondeterminism that
/// would surface here before it matters in 9c+.
#[test]
#[ignore = "long running; needs Metal device + moeflux artifacts"]
fn gpu_batched_experts_forward_close_c_vs_rust() {
    use moeflux::riir::expert_forward::synth;
    use moeflux::riir::VARIANT;

    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();

    let k: usize = 4;
    let expert_data = synth::k_expert_data_seeded(k);
    let h_post = synth::h_post_seeded();
    let h_mid = synth::h_mid_seeded();
    let shared_out = synth::shared_out_seeded();
    let weights = synth::expert_weights_seeded(k);
    let shared_gate_score: f32 = -1.0;

    assert_eq!(expert_data.len(), k * VARIANT.expert_size_4bit());
    assert_eq!(weights.len(), k);
    let weight_sum: f32 = weights.iter().sum();
    assert!(
        (weight_sum - 1.0).abs() < 1e-5,
        "synth weights don't sum to 1: {weight_sum}"
    );

    let c_out = c.gpu_batched_experts_forward(
        k as i32,
        &expert_data,
        &h_post,
        &h_mid,
        &shared_out,
        &weights,
        shared_gate_score,
    );
    let rs_out = rs.gpu_batched_experts_forward(
        k as i32,
        &expert_data,
        &h_post,
        &h_mid,
        &shared_out,
        &weights,
        shared_gate_score,
    );
    assert_eq!(c_out.len(), VARIANT.hidden_dim);
    assert_eq!(rs_out.len(), VARIANT.hidden_dim);

    let cos = cosine_sim(&c_out, &rs_out);
    let max_abs_out = c_out
        .iter()
        .chain(rs_out.iter())
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    let max_abs_diff = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = if max_abs_out > 0.0 {
        max_abs_diff / max_abs_out
    } else {
        0.0
    };

    eprintln!(
        "[diff:gpu_batched_experts_forward k={k}] cosine={cos:.7} \
         max_abs_diff={max_abs_diff:.3e} max_abs_out={max_abs_out:.3e} \
         relative={rel:.3e}"
    );

    assert!(
        c_out.iter().all(|x| x.is_finite()),
        "[diff:gpu_batched_experts_forward] C output has NaN/Inf"
    );
    assert!(
        rs_out.iter().all(|x| x.is_finite()),
        "[diff:gpu_batched_experts_forward] Rust output has NaN/Inf"
    );
    assert!(
        c_out.iter().any(|&x| x != 0.0),
        "[diff:gpu_batched_experts_forward] C output is all zero"
    );
    assert!(
        rs_out.iter().any(|&x| x != 0.0),
        "[diff:gpu_batched_experts_forward] Rust output is all zero"
    );

    const COSINE_FLOOR: f32 = 0.9999;
    const REL_DIFF_FLOOR: f32 = 1e-3;
    assert!(
        cos >= COSINE_FLOOR,
        "[diff:gpu_batched_experts_forward] cosine {cos:.7} below {COSINE_FLOOR}"
    );
    assert!(
        rel <= REL_DIFF_FLOOR,
        "[diff:gpu_batched_experts_forward] relative max_abs_diff \
         {rel:.3e} above {REL_DIFF_FLOOR:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Slice 5d-7a — per-kernel diff coverage for GPU full-attention
// ---------------------------------------------------------------------------
//
// Synthetic deterministic inputs against the same per-kernel
// tolerance regime as slices 9a / 9b / 9e (cosine ≥ 0.9999,
// rel_max_abs_diff ≤ 1e-3). The four kernels have only SIMD or
// threadgroup-shared reductions (no atomics), so empirical
// expectation is bit-exact per-PSO; the floor is defensive.

/// Generate `n` deterministic floats from a low-discrepancy sequence
/// shifted by a per-test seed. Avoids per-test PRNG plumbing while
/// keeping the inputs non-symmetric (so reductions are meaningful).
fn synth_floats(seed: u32, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let phase = (seed as f32 * 0.13) + (i as f32 * 0.017);
            phase.sin() * scale + (phase * 1.7).cos() * (scale * 0.5)
        })
        .collect()
}

/// `attn_scores_batched` (slice 5d-7a). Synthetic Q + K, varying
/// `seq_len`. Stride-tight output. Empirically bit-exact per-PSO; the
/// cosine/rel floors are defensive.
#[test]
#[ignore = "long running; needs Metal device + moeflux artifacts"]
fn attn_scores_close_c_vs_rust() {
    use moeflux::riir::VARIANT;

    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();

    let v = VARIANT;
    let num_heads = v.num_attn_heads as u32;
    let num_kv_heads = v.num_kv_heads as u32;
    let head_dim = v.head_dim as u32;
    let kv_dim = num_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for &seq_len in &[32u32, 64, 128, 512] {
        let q = synth_floats(seq_len, (num_heads * head_dim) as usize, 0.4);
        let k_cache =
            synth_floats(seq_len + 1, (seq_len * kv_dim) as usize, 0.3);

        let c_out = c.attn_scores_batched(
            num_heads, num_kv_heads, head_dim, seq_len, &q, &k_cache, scale,
        );
        let rs_out = rs.attn_scores_batched(
            num_heads, num_kv_heads, head_dim, seq_len, &q, &k_cache, scale,
        );
        assert_eq!(c_out.len(), (num_heads * seq_len) as usize);
        assert_eq!(rs_out.len(), c_out.len());

        let cos = cosine_sim(&c_out, &rs_out);
        let max_abs_out = c_out
            .iter()
            .chain(rs_out.iter())
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        let max_abs_diff = c_out
            .iter()
            .zip(rs_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rel = if max_abs_out > 0.0 {
            max_abs_diff / max_abs_out
        } else {
            0.0
        };

        eprintln!(
            "[diff:attn_scores seq_len={seq_len}] cosine={cos:.7} \
             max_abs_diff={max_abs_diff:.3e} \
             max_abs_out={max_abs_out:.3e} relative={rel:.3e}"
        );

        assert!(
            c_out.iter().all(|x| x.is_finite()),
            "[diff:attn_scores seq_len={seq_len}] C output has NaN/Inf"
        );
        assert!(
            rs_out.iter().all(|x| x.is_finite()),
            "[diff:attn_scores seq_len={seq_len}] Rust output has NaN/Inf"
        );

        const COSINE_FLOOR: f32 = 0.9999;
        const REL_DIFF_FLOOR: f32 = 1e-3;
        assert!(
            cos >= COSINE_FLOOR,
            "[diff:attn_scores seq_len={seq_len}] cosine {cos:.7} \
             below {COSINE_FLOOR}"
        );
        assert!(
            rel <= REL_DIFF_FLOOR,
            "[diff:attn_scores seq_len={seq_len}] relative max_abs_diff \
             {rel:.3e} above {REL_DIFF_FLOOR:.3e}"
        );
    }
}

/// `attn_softmax_batched` (slice 5d-7a). Synthetic per-row logits,
/// varying `seq_len`. Three-pass softmax with two SIMD reductions —
/// empirically bit-exact per-PSO.
#[test]
#[ignore = "long running; needs Metal device + moeflux artifacts"]
fn attn_softmax_close_c_vs_rust() {
    use moeflux::riir::VARIANT;

    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();

    let v = VARIANT;
    let num_heads = v.num_attn_heads as u32;

    for &seq_len in &[32u32, 64, 128, 512] {
        let scores =
            synth_floats(seq_len * 7, (num_heads * seq_len) as usize, 1.5);

        let c_out = c.attn_softmax_batched(num_heads, seq_len, &scores);
        let rs_out = rs.attn_softmax_batched(num_heads, seq_len, &scores);
        assert_eq!(c_out.len(), (num_heads * seq_len) as usize);

        // Per-row softmax sums should be ~1.
        for h in 0..num_heads as usize {
            let row = &c_out[h * seq_len as usize..(h + 1) * seq_len as usize];
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "[diff:attn_softmax seq_len={seq_len}] row {h} sum {sum} \
                 not ~1"
            );
        }

        let cos = cosine_sim(&c_out, &rs_out);
        let max_abs_out = c_out
            .iter()
            .chain(rs_out.iter())
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        let max_abs_diff = c_out
            .iter()
            .zip(rs_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rel = if max_abs_out > 0.0 {
            max_abs_diff / max_abs_out
        } else {
            0.0
        };

        eprintln!(
            "[diff:attn_softmax seq_len={seq_len}] cosine={cos:.7} \
             max_abs_diff={max_abs_diff:.3e} \
             max_abs_out={max_abs_out:.3e} relative={rel:.3e}"
        );

        assert!(
            c_out.iter().all(|x| x.is_finite()),
            "[diff:attn_softmax seq_len={seq_len}] C output has NaN/Inf"
        );
        assert!(
            rs_out.iter().all(|x| x.is_finite()),
            "[diff:attn_softmax seq_len={seq_len}] Rust output has NaN/Inf"
        );

        const COSINE_FLOOR: f32 = 0.9999;
        const REL_DIFF_FLOOR: f32 = 1e-3;
        assert!(
            cos >= COSINE_FLOOR,
            "[diff:attn_softmax seq_len={seq_len}] cosine {cos:.7} \
             below {COSINE_FLOOR}"
        );
        assert!(
            rel <= REL_DIFF_FLOOR,
            "[diff:attn_softmax seq_len={seq_len}] relative max_abs_diff \
             {rel:.3e} above {REL_DIFF_FLOOR:.3e}"
        );
    }
}

/// `attn_values_batched` (slice 5d-7a). Synthetic post-softmax probs +
/// V, varying `seq_len`. Per-thread inner loop over `seq_len` — no
/// reductions; bit-exact per-PSO.
#[test]
#[ignore = "long running; needs Metal device + moeflux artifacts"]
fn attn_values_close_c_vs_rust() {
    use moeflux::riir::VARIANT;

    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();

    let v = VARIANT;
    let num_heads = v.num_attn_heads as u32;
    let num_kv_heads = v.num_kv_heads as u32;
    let head_dim = v.head_dim as u32;
    let kv_dim = num_kv_heads * head_dim;

    for &seq_len in &[32u32, 64, 128, 512] {
        // Build a valid softmax by exp-then-normalize per row.
        let raw = synth_floats(seq_len + 11, (num_heads * seq_len) as usize, 1.0);
        let mut scores = vec![0.0f32; raw.len()];
        for h in 0..num_heads as usize {
            let row_start = h * seq_len as usize;
            let row = &raw[row_start..row_start + seq_len as usize];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> =
                row.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for i in 0..seq_len as usize {
                scores[row_start + i] = exps[i] / sum;
            }
        }
        let v_cache =
            synth_floats(seq_len + 13, (seq_len * kv_dim) as usize, 0.5);

        let c_out = c.attn_values_batched(
            num_heads, num_kv_heads, head_dim, seq_len, &scores, &v_cache,
        );
        let rs_out = rs.attn_values_batched(
            num_heads, num_kv_heads, head_dim, seq_len, &scores, &v_cache,
        );
        assert_eq!(c_out.len(), (num_heads * head_dim) as usize);
        assert_eq!(rs_out.len(), c_out.len());

        let cos = cosine_sim(&c_out, &rs_out);
        let max_abs_out = c_out
            .iter()
            .chain(rs_out.iter())
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        let max_abs_diff = c_out
            .iter()
            .zip(rs_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rel = if max_abs_out > 0.0 {
            max_abs_diff / max_abs_out
        } else {
            0.0
        };

        eprintln!(
            "[diff:attn_values seq_len={seq_len}] cosine={cos:.7} \
             max_abs_diff={max_abs_diff:.3e} \
             max_abs_out={max_abs_out:.3e} relative={rel:.3e}"
        );

        assert!(
            c_out.iter().all(|x| x.is_finite()),
            "[diff:attn_values seq_len={seq_len}] C output has NaN/Inf"
        );
        assert!(
            rs_out.iter().all(|x| x.is_finite()),
            "[diff:attn_values seq_len={seq_len}] Rust output has NaN/Inf"
        );

        const COSINE_FLOOR: f32 = 0.9999;
        const REL_DIFF_FLOOR: f32 = 1e-3;
        assert!(
            cos >= COSINE_FLOOR,
            "[diff:attn_values seq_len={seq_len}] cosine {cos:.7} \
             below {COSINE_FLOOR}"
        );
        assert!(
            rel <= REL_DIFF_FLOOR,
            "[diff:attn_values seq_len={seq_len}] relative max_abs_diff \
             {rel:.3e} above {REL_DIFF_FLOOR:.3e}"
        );
    }
}

/// `sigmoid_gate` (slice 5d-7a). Per-thread elementwise — no
/// reductions, no atomics. Expected bit-exact.
#[test]
#[ignore = "long running; needs Metal device + moeflux artifacts"]
fn sigmoid_gate_close_c_vs_rust() {
    use moeflux::riir::VARIANT;

    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();

    let v = VARIANT;
    let dim = (v.num_attn_heads * v.head_dim) as u32;
    let x = synth_floats(101, dim as usize, 0.7);
    let gate = synth_floats(202, dim as usize, 1.2);

    let c_out = c.sigmoid_gate(dim, &gate, &x);
    let rs_out = rs.sigmoid_gate(dim, &gate, &x);
    assert_eq!(c_out.len(), dim as usize);
    assert_eq!(rs_out.len(), dim as usize);

    let max_abs_diff = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cos = cosine_sim(&c_out, &rs_out);
    let max_abs_out = c_out
        .iter()
        .chain(rs_out.iter())
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);

    eprintln!(
        "[diff:sigmoid_gate] cosine={cos:.7} max_abs_diff={max_abs_diff:.3e} \
         max_abs_out={max_abs_out:.3e}"
    );

    assert!(
        c_out.iter().all(|x| x.is_finite()),
        "[diff:sigmoid_gate] C output has NaN/Inf"
    );
    assert!(
        rs_out.iter().all(|x| x.is_finite()),
        "[diff:sigmoid_gate] Rust output has NaN/Inf"
    );

    // Bit-exact expected (no reductions, no atomics — `sigmoid_gate`
    // is a pure per-thread elementwise op). Floors held loose to absorb
    // any libm-vs-MSL exp/sigmoid drift we haven't characterized; if
    // observed in practice we'll tighten.
    let first_diff = c_out.iter().zip(rs_out.iter()).position(|(a, b)| a != b);
    if let Some(i) = first_diff {
        eprintln!(
            "[diff:sigmoid_gate] first non-bit-exact at i={i}: \
             c={} rs={}",
            c_out[i], rs_out[i]
        );
    }
    assert_eq!(
        max_abs_diff, 0.0,
        "[diff:sigmoid_gate] expected bit-exact, got max_abs_diff {max_abs_diff:.3e}"
    );
}

/// Phase 4e: paired begin → complete via the deferred-experts state
/// machine, C path vs Rust port. Same kernels as 9b
/// (`gpu_batched_experts_forward`) split across an async commit and
/// a separate wait + readback. Confirms the begin/complete pair
/// produces the same final hidden state as the synchronous
/// single-call path.
///
/// Same tolerance regime as 9b (cosine ≥ 0.9999, rel ≤ 1e-3 — likely
/// bit-exact in practice; the underlying kernels are unchanged).
/// `layer_idx = -1` on both sides (synthetic dispatch — see the C
/// hook's invariant in `mf_begin_deferred_experts`).
#[test]
#[ignore = "long running; needs Metal device + moeflux artifacts"]
fn deferred_experts_begin_complete_close_c_vs_rust() {
    use moeflux::riir::expert_forward::synth;
    use moeflux::riir::VARIANT;

    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();

    let k: usize = 4;
    let expert_data = synth::k_expert_data_seeded(k);
    let h_post = synth::h_post_seeded();
    let h_mid = synth::h_mid_seeded();
    let shared_out = synth::shared_out_seeded();
    let weights = synth::expert_weights_seeded(k);
    let shared_gate_score: f32 = -1.0;

    assert_eq!(expert_data.len(), k * VARIANT.expert_size_4bit());
    assert_eq!(weights.len(), k);

    c.begin_deferred_experts(
        k as i32, &expert_data, &h_post, &h_mid, &shared_out, &weights,
        shared_gate_score,
    );
    let c_out = c.complete_deferred_experts();

    rs.begin_deferred_experts(
        k as i32, &expert_data, &h_post, &h_mid, &shared_out, &weights,
        shared_gate_score,
    );
    let rs_out = rs.complete_deferred_experts();

    assert_eq!(c_out.len(), VARIANT.hidden_dim);
    assert_eq!(rs_out.len(), VARIANT.hidden_dim);

    let cos = cosine_sim(&c_out, &rs_out);
    let max_abs_out = c_out
        .iter()
        .chain(rs_out.iter())
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    let max_abs_diff = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = if max_abs_out > 0.0 {
        max_abs_diff / max_abs_out
    } else {
        0.0
    };

    eprintln!(
        "[diff:deferred_experts_begin_complete k={k}] cosine={cos:.7} \
         max_abs_diff={max_abs_diff:.3e} max_abs_out={max_abs_out:.3e} \
         relative={rel:.3e}"
    );

    assert!(
        c_out.iter().all(|x| x.is_finite()),
        "[diff:deferred_experts_begin_complete] C output has NaN/Inf"
    );
    assert!(
        rs_out.iter().all(|x| x.is_finite()),
        "[diff:deferred_experts_begin_complete] Rust output has NaN/Inf"
    );
    assert!(
        c_out.iter().any(|&x| x != 0.0),
        "[diff:deferred_experts_begin_complete] C output is all zero"
    );
    assert!(
        rs_out.iter().any(|&x| x != 0.0),
        "[diff:deferred_experts_begin_complete] Rust output is all zero"
    );

    const COSINE_FLOOR: f32 = 0.9999;
    const REL_DIFF_FLOOR: f32 = 1e-3;
    assert!(
        cos >= COSINE_FLOOR,
        "[diff:deferred_experts_begin_complete] cosine {cos:.7} below {COSINE_FLOOR}"
    );
    assert!(
        rel <= REL_DIFF_FLOOR,
        "[diff:deferred_experts_begin_complete] relative max_abs_diff \
         {rel:.3e} above {REL_DIFF_FLOOR:.3e}"
    );
}

/// Phase 4e: discard semantics. Asserts that on each backend, a
/// `discard` between two begins doesn't taint the second dispatch's
/// output — i.e. the wait-then-clear actually clears state, and the
/// next begin starts from a clean slate. Performs the discard cycle
/// on both backends and compares the second dispatch's hidden state
/// across C vs Rust.
///
/// The first (discarded) dispatch uses K=2 with one set of synthetic
/// inputs; the second (kept) dispatch uses K=4 with a different set.
/// The cross-backend comparison confirms both impls handle the
/// discard-then-begin sequence identically.
#[test]
#[ignore = "long running; needs Metal device + moeflux artifacts"]
fn deferred_experts_discard_clears_state_c_vs_rust() {
    use moeflux::riir::expert_forward::synth;
    use moeflux::riir::VARIANT;

    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();

    // First dispatch (will be discarded): K=2.
    let k1: usize = 2;
    let data1 = synth::k_expert_data_seeded(k1);
    let h_post1 = synth::h_post_seeded();
    let h_mid1 = synth::h_mid_seeded();
    let shared_out1 = synth::shared_out_seeded();
    let weights1 = synth::expert_weights_seeded(k1);

    // Second dispatch (will be completed and compared): K=4 with a
    // different set of inputs derived by reseeding through K.
    let k2: usize = 4;
    let data2 = synth::k_expert_data_seeded(k2);
    let h_post2 = synth::h_post_seeded();
    let h_mid2 = synth::h_mid_seeded();
    let shared_out2 = synth::shared_out_seeded();
    let weights2 = synth::expert_weights_seeded(k2);

    // C side.
    c.begin_deferred_experts(
        k1 as i32, &data1, &h_post1, &h_mid1, &shared_out1, &weights1, -1.0,
    );
    c.discard_deferred_experts();
    c.begin_deferred_experts(
        k2 as i32, &data2, &h_post2, &h_mid2, &shared_out2, &weights2, -1.0,
    );
    let c_out = c.complete_deferred_experts();

    // Rust side.
    rs.begin_deferred_experts(
        k1 as i32, &data1, &h_post1, &h_mid1, &shared_out1, &weights1, -1.0,
    );
    rs.discard_deferred_experts();
    rs.begin_deferred_experts(
        k2 as i32, &data2, &h_post2, &h_mid2, &shared_out2, &weights2, -1.0,
    );
    let rs_out = rs.complete_deferred_experts();

    assert_eq!(c_out.len(), VARIANT.hidden_dim);
    assert_eq!(rs_out.len(), VARIANT.hidden_dim);

    let cos = cosine_sim(&c_out, &rs_out);
    let max_abs_out = c_out
        .iter()
        .chain(rs_out.iter())
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    let max_abs_diff = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = if max_abs_out > 0.0 {
        max_abs_diff / max_abs_out
    } else {
        0.0
    };

    eprintln!(
        "[diff:deferred_experts_discard k1={k1} k2={k2}] cosine={cos:.7} \
         max_abs_diff={max_abs_diff:.3e} max_abs_out={max_abs_out:.3e} \
         relative={rel:.3e}"
    );

    assert!(
        c_out.iter().all(|x| x.is_finite()),
        "[diff:deferred_experts_discard] C output has NaN/Inf"
    );
    assert!(
        rs_out.iter().all(|x| x.is_finite()),
        "[diff:deferred_experts_discard] Rust output has NaN/Inf"
    );
    assert!(
        c_out.iter().any(|&x| x != 0.0),
        "[diff:deferred_experts_discard] C output is all zero"
    );
    assert!(
        rs_out.iter().any(|&x| x != 0.0),
        "[diff:deferred_experts_discard] Rust output is all zero"
    );

    const COSINE_FLOOR: f32 = 0.9999;
    const REL_DIFF_FLOOR: f32 = 1e-3;
    assert!(
        cos >= COSINE_FLOOR,
        "[diff:deferred_experts_discard] cosine {cos:.7} below {COSINE_FLOOR}"
    );
    assert!(
        rel <= REL_DIFF_FLOOR,
        "[diff:deferred_experts_discard] relative max_abs_diff \
         {rel:.3e} above {REL_DIFF_FLOOR:.3e}"
    );
}

/// Phase 4c: end-to-end linear-attention layer forward, C path vs
/// Rust port, via the dump hook. Token 1 → embedding → layer 0
/// (linear-attn) on both sides; cosine ≥ 0.9999 / max_abs_diff /
/// max_abs_out ≤ 1e-3 floors per the strategy doc's Phase 3+ tolerance
/// regime. The numerical signal that 4c was building toward.
///
/// Both sides reset their per-layer state via `memory_clear` before
/// the call so the recurrence starts empty. The C side uses CPU
/// rms_norm for the input norm + GPU upload to `buf_input`; the Rust
/// side does GPU rms_norm directly. They differ on reduction order
/// (CPU sequential vs GPU parallel simd_sum + threadgroup-shared
/// second stage); per Phase 9e and the strategy doc that's at most
/// ULP-bounded drift on the norm output, which composes through the
/// rest of the pipeline staying inside the cosine floor.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn layer_forward_dump_close_c_vs_rust() {
    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();
    let hidden_dim = moeflux::riir::VARIANT.hidden_dim;

    c.memory_clear();
    rs.memory_clear();

    // Embed token 1 on the C side and use that as input on both. (We
    // could embed on each side, but `embed` is bit-exact across them
    // per slice 1, so a single source keeps the test focused on the
    // layer forward.)
    let hidden_in = c.embed(1);
    assert_eq!(hidden_in.len(), hidden_dim);

    let layer_idx = 0i32; // linear-attn (full_attn_interval = 4)
    let pos = 0i32;

    let c_out = c.layer_forward_dump(layer_idx, pos, &hidden_in);
    let rs_out = rs.layer_forward_dump(layer_idx, pos, &hidden_in);
    assert_eq!(c_out.len(), hidden_dim);
    assert_eq!(rs_out.len(), hidden_dim);

    assert!(
        c_out.iter().all(|x| x.is_finite()),
        "[diff:layer_forward_dump] C output has NaN/Inf"
    );
    assert!(
        rs_out.iter().all(|x| x.is_finite()),
        "[diff:layer_forward_dump] Rust output has NaN/Inf"
    );

    // Magnitude check — output isn't trivially all-zero.
    let max_abs_out =
        c_out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs_out > 1e-6,
        "[diff:layer_forward_dump] C output magnitude {max_abs_out:.3e} \
         too small — production path likely no-op'd (cross-Ctx layer_cache?)"
    );

    let cos = cosine_sim(&c_out, &rs_out);
    let max_abs_diff = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = max_abs_diff / max_abs_out.max(f32::EPSILON);

    eprintln!(
        "[diff:layer_forward_dump layer=0] cosine={cos:.7} \
         max_abs_diff={max_abs_diff:.3e} max_abs_out={max_abs_out:.3e} \
         rel={rel:.3e}"
    );

    const COSINE_FLOOR: f32 = 0.9999;
    const REL_DIFF_FLOOR: f32 = 1e-3;
    assert!(
        cos >= COSINE_FLOOR,
        "[diff:layer_forward_dump] cosine {cos:.7} below {COSINE_FLOOR}"
    );
    assert!(
        rel <= REL_DIFF_FLOOR,
        "[diff:layer_forward_dump] relative max_abs_diff \
         {rel:.3e} above {REL_DIFF_FLOOR:.3e}"
    );
}

/// Phase 4f-4: CPU-combine path matches the C side. Forces the Rust
/// port to take the `gpu_combine = false` branch (CMD3 omits
/// `moe_combine_residual`; the K per-expert outputs are read back to
/// host and combined CPU-side per `infer.m:4106..4129`). The C side
/// always picks `gpu_combine = true` for layer 0 with all pipelines
/// available — but the **arithmetic** of CPU combine is identical
/// (`hidden = h_mid + Σ weights[k] × out[k] + sigmoid(gate) ×
/// shared_out`), so the two paths must agree at the cosine floor.
///
/// Catches porting drift in the CPU-combine code path itself
/// (cpu_vec_madd / cpu_sigmoid_scalar / final fold) — the GPU-combine
/// path is exercised by every other layer-forward diff test, so this
/// is the only test that hits the CPU branch.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn layer_forward_dump_close_c_vs_rust_cpu_combine() {
    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();
    let hidden_dim = moeflux::riir::VARIANT.hidden_dim;

    c.memory_clear();
    rs.memory_clear();

    let hidden_in = c.embed(1);
    assert_eq!(hidden_in.len(), hidden_dim);

    let layer_idx = 0i32;
    let pos = 0i32;

    let c_out = c.layer_forward_dump(layer_idx, pos, &hidden_in);
    let mut rs_out = vec![0.0f32; hidden_dim];
    rs.0.layer_forward_dump_with_gpu_combine(
        layer_idx,
        pos,
        &hidden_in,
        &mut rs_out,
        /* gpu_combine = */ false,
    )
    .expect("RsBackend layer_forward_dump_with_gpu_combine");

    assert!(
        c_out.iter().all(|x| x.is_finite()),
        "[diff:layer_forward_dump_cpu_combine] C output has NaN/Inf"
    );
    assert!(
        rs_out.iter().all(|x| x.is_finite()),
        "[diff:layer_forward_dump_cpu_combine] Rust output has NaN/Inf"
    );

    let max_abs_out =
        c_out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs_out > 1e-6,
        "[diff:layer_forward_dump_cpu_combine] C output magnitude \
         {max_abs_out:.3e} too small"
    );

    let cos = cosine_sim(&c_out, &rs_out);
    let max_abs_diff = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = max_abs_diff / max_abs_out.max(f32::EPSILON);

    eprintln!(
        "[diff:layer_forward_dump_cpu_combine layer=0] cosine={cos:.7} \
         max_abs_diff={max_abs_diff:.3e} max_abs_out={max_abs_out:.3e} \
         rel={rel:.3e}"
    );

    const COSINE_FLOOR: f32 = 0.9999;
    const REL_DIFF_FLOOR: f32 = 1e-3;
    assert!(
        cos >= COSINE_FLOOR,
        "[diff:layer_forward_dump_cpu_combine] cosine {cos:.7} below \
         {COSINE_FLOOR}"
    );
    assert!(
        rel <= REL_DIFF_FLOOR,
        "[diff:layer_forward_dump_cpu_combine] relative max_abs_diff \
         {rel:.3e} above {REL_DIFF_FLOOR:.3e}"
    );
}

/// Phase 4f-3 parity: five back-to-back `layer_forward_dump` calls
/// on the Rust side must all succeed without leaking a deferred-
/// experts dispatch across calls. After slice 4f-3,
/// `post_attention_tail` leaves an in-flight K-expert dispatch in
/// `RsCtx::deferred` instead of reading back inline.
/// `RsCtx::layer_forward_dump` reconstitutes the synchronous single-
/// step contract by draining the dispatch into `linear_buffers.input`
/// post-call, with a defensive `discard_deferred_experts_in` bracket
/// pre-call to guard against stale state.
///
/// Failure modes this test catches:
/// - **Missing post-call drain**: iteration 2+ would see a `Some`
///   deferred slot from iter 1 and hit
///   `DeferredError::AlreadyActive` → `RsError::EvalFailed`.
/// - **Missing pre-call discard**: a buggy caller leaving stale
///   state would propagate the same way.
/// - **Drain wrote into wrong buffer**: outputs would be all-zero or
///   NaN, caught by the finite + magnitude assertions.
///
/// Numerical equivalence across iterations IS asserted: with
/// `memory_clear` between calls (which slice 4f-3 extended to also
/// reset GPU-side linear-attn recurrence), each iteration starts
/// from the same state. Per-PSO Metal determinism (slice 9 finding)
/// means the outputs must be byte-identical.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn layer_forward_dump_back_to_back_no_deferred_leak() {
    let mut rs: RsBackend = open_backend();
    let hidden_dim = moeflux::riir::VARIANT.hidden_dim;

    let hidden_in = rs.embed(1);
    assert_eq!(hidden_in.len(), hidden_dim);

    let layer_idx = 0i32; // linear-attn
    let pos = 0i32;
    let n_iters = 5usize;

    let mut outs: Vec<Vec<f32>> = Vec::with_capacity(n_iters);
    for i in 0..n_iters {
        // memory_clear resets both host LayerState and GPU recurrence
        // (per slice 4f-3's RsCtx::memory_clear extension), so each
        // iteration starts from the same state. Without the GPU
        // reset, iterations 1..N would see stale conv_state /
        // delta_state from iter 0 and diverge.
        rs.memory_clear();
        let out = rs.layer_forward_dump(layer_idx, pos, &hidden_in);
        assert_eq!(out.len(), hidden_dim, "iter {i}: output length");
        assert!(
            out.iter().all(|x| x.is_finite()),
            "iter {i}: output has NaN/Inf — likely stale deferred state"
        );
        let max_abs = out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs > 1e-6,
            "iter {i}: output magnitude {max_abs:.3e} too small — drain \
             likely reading from wrong buffer or hitting AlreadyActive"
        );
        outs.push(out);
    }

    // All five outputs must be byte-identical: same layer, same
    // input, fully-reset state between calls. Drift here implies
    // either the deferred bracketing is reading from a buffer that
    // wasn't drained, or memory_clear failed to reset some piece of
    // recurrence.
    for i in 1..n_iters {
        let drift_max = outs[0]
            .iter()
            .zip(outs[i].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            drift_max, 0.0,
            "iter 0 vs iter {i} differ by max_abs_diff={drift_max:.3e} — \
             deferred-experts state leaked across calls or memory_clear \
             did not reset all recurrence"
        );
    }
    eprintln!(
        "[diff:layer_forward_dump_back_to_back] {n_iters} iterations \
         bit-identical (max_abs_diff=0)"
    );
}

/// Phase 4d: end-to-end **full-attention** layer forward, C path vs
/// Rust port, via the dump hook. Companion to the linear-attn test
/// above. Same floors / shape; the only differences are
/// `layer_idx = 3` (first full-attn layer when
/// `full_attn_interval == 4`) and the eprintln tag.
///
/// The Rust path swaps the linear-attn pipeline (4 batched
/// projections + 5 fused GPU kernels) for the full-attn pipeline
/// (3 batched projections + per-head Q/K rms_norm + RoPE + KV append
/// + SDPA), then hands off to the same shared `post_attention_tail`
/// the linear-attn forward uses. So if 4c was green and 4d is red,
/// the bug is somewhere in the swapped attention pipeline — not the
/// shared tail.
///
/// `memory_clear` runs on both sides before the call so the KV cache
/// starts at `len = 0` and the test runs at `pos = 0`. After the
/// call `kv.len = 1`.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn layer_forward_dump_close_c_vs_rust_full_attn() {
    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();
    let hidden_dim = moeflux::riir::VARIANT.hidden_dim;

    c.memory_clear();
    rs.memory_clear();

    let hidden_in = c.embed(1);
    assert_eq!(hidden_in.len(), hidden_dim);

    let layer_idx = 3i32; // first full-attn layer (full_attn_interval = 4)
    let pos = 0i32;

    let c_out = c.layer_forward_dump(layer_idx, pos, &hidden_in);
    let rs_out = rs.layer_forward_dump(layer_idx, pos, &hidden_in);
    assert_eq!(c_out.len(), hidden_dim);
    assert_eq!(rs_out.len(), hidden_dim);

    assert!(
        c_out.iter().all(|x| x.is_finite()),
        "[diff:layer_forward_dump_full] C output has NaN/Inf"
    );
    assert!(
        rs_out.iter().all(|x| x.is_finite()),
        "[diff:layer_forward_dump_full] Rust output has NaN/Inf"
    );

    let max_abs_out =
        c_out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs_out > 1e-6,
        "[diff:layer_forward_dump_full] C output magnitude {max_abs_out:.3e} \
         too small — production path likely no-op'd (cross-Ctx layer_cache?)"
    );

    let cos = cosine_sim(&c_out, &rs_out);
    let max_abs_diff = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = max_abs_diff / max_abs_out.max(f32::EPSILON);

    eprintln!(
        "[diff:layer_forward_dump layer=3 (full-attn)] cosine={cos:.7} \
         max_abs_diff={max_abs_diff:.3e} max_abs_out={max_abs_out:.3e} \
         rel={rel:.3e}"
    );

    const COSINE_FLOOR: f32 = 0.9999;
    const REL_DIFF_FLOOR: f32 = 1e-3;
    assert!(
        cos >= COSINE_FLOOR,
        "[diff:layer_forward_dump_full] cosine {cos:.7} below {COSINE_FLOOR}"
    );
    assert!(
        rel <= REL_DIFF_FLOOR,
        "[diff:layer_forward_dump_full] relative max_abs_diff \
         {rel:.3e} above {REL_DIFF_FLOOR:.3e}"
    );
}

/// Slice 5d-7b: full-attention layer forward at `kv_len ≥ 32` —
/// exercises the GPU SDPA fast path on both backends. The
/// `layer_forward_dump_close_c_vs_rust_full_attn` sibling above runs
/// at `pos = 0` (kv_len = 1 after append), which is below the gate
/// (`kv_len >= 32`); this test prefills the KV cache so the test
/// call lands on the gate predicate and the 4 GPU attn kernels run
/// inside CMD2.
///
/// Both backends share the same gate predicate (`kv_len >= 32 && <
/// GPU_KV_SEQ`); both will run GPU SDPA on the test call, so the
/// comparison stays at the same tolerance regime as the existing
/// full-attn dump test.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn layer_forward_dump_close_c_vs_rust_full_attn_gpu_path() {
    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();
    let hidden_dim = moeflux::riir::VARIANT.hidden_dim;

    c.memory_clear();
    rs.memory_clear();

    let layer_idx = 3i32; // first full-attn layer (full_attn_interval = 4)

    // Prefill: call layer_forward_dump at pos=0..31 to fill layer 3's
    // KV cache to length 32 on both backends. Distinct token per
    // position so each entry is non-trivial.
    for pos in 0i32..32 {
        let hidden_in = c.embed(1 + pos);
        let _c_drain = c.layer_forward_dump(layer_idx, pos, &hidden_in);
        let _rs_drain = rs.layer_forward_dump(layer_idx, pos, &hidden_in);
    }

    // Test call: pos=32 → kv_len becomes 33 after append, satisfying
    // the GPU-path gate (`kv_len >= 32`). The 4 attn kernels fire on
    // both backends.
    let pos = 32i32;
    let hidden_in = c.embed(1 + pos);
    let c_out = c.layer_forward_dump(layer_idx, pos, &hidden_in);
    let rs_out = rs.layer_forward_dump(layer_idx, pos, &hidden_in);
    assert_eq!(c_out.len(), hidden_dim);
    assert_eq!(rs_out.len(), hidden_dim);

    assert!(
        c_out.iter().all(|x| x.is_finite()),
        "[diff:layer_forward_dump_full_gpu] C output has NaN/Inf"
    );
    assert!(
        rs_out.iter().all(|x| x.is_finite()),
        "[diff:layer_forward_dump_full_gpu] Rust output has NaN/Inf"
    );

    let max_abs_out =
        c_out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs_out > 1e-6,
        "[diff:layer_forward_dump_full_gpu] C output magnitude \
         {max_abs_out:.3e} too small — GPU path likely no-op'd"
    );

    let cos = cosine_sim(&c_out, &rs_out);
    let max_abs_diff = c_out
        .iter()
        .zip(rs_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = max_abs_diff / max_abs_out.max(f32::EPSILON);

    eprintln!(
        "[diff:layer_forward_dump layer=3 (full-attn, GPU SDPA at kv_len=33)] \
         cosine={cos:.7} max_abs_diff={max_abs_diff:.3e} \
         max_abs_out={max_abs_out:.3e} rel={rel:.3e}"
    );

    const COSINE_FLOOR: f32 = 0.9999;
    const REL_DIFF_FLOOR: f32 = 1e-3;
    assert!(
        cos >= COSINE_FLOOR,
        "[diff:layer_forward_dump_full_gpu] cosine {cos:.7} below \
         {COSINE_FLOOR}"
    );
    assert!(
        rel <= REL_DIFF_FLOOR,
        "[diff:layer_forward_dump_full_gpu] relative max_abs_diff \
         {rel:.3e} above {REL_DIFF_FLOOR:.3e}"
    );
}

/// Phase 4b sanity: the C-side `mf_layer_forward_dump` hook is
/// callable and returns finite output. The numerical-correctness
/// signal lands in 4c when `RsCtx::layer_forward_dump` exists and
/// the diff oracle compares per-layer outputs head-on.
///
/// Permissive on purpose: 4b discovered a third cross-Ctx state bug
/// in addition to the two captured during the bisect. The C side has
/// a process-global `layer_cache` (host-side weight-pointer cache,
/// `infer.m:~3960`) that's populated on first call to
/// `fused_layer_forward` from whichever ctx ran first. When that ctx
/// is freed and a later test opens a fresh ctx, the cache's
/// `layer_cache_built` flag stays 1 so the cache is never rebuilt —
/// its pointers now reference freed memory. On this M2 Max with
/// these tests run serially, those pages typically come back zeroed,
/// making `fused_layer_forward` for the second ctx an effective
/// no-op (`hidden_out == hidden_in`). This is the same bug class as
/// the documented `g_deferred` cross-Ctx NaN finding; the Phase 7
/// fix lifts both into Ctx-owned state.
///
/// FIXME(riir): `layer_cache` cross-Ctx state pollution. Phase 7 fix
/// alongside the typed-`memory_seq_rm` work. Recorded in the
/// drama_llama in-repo `blallama_session_state_pollution.md`.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn layer_forward_dump_c_self_sanity() {
    let mut c: CBackend = open_backend();
    let hidden_dim = moeflux::riir::VARIANT.hidden_dim;

    c.memory_clear();
    let hidden_in = c.embed(/* token_id */ 1);
    assert_eq!(hidden_in.len(), hidden_dim);

    let hidden_out = c.layer_forward_dump(/* layer_idx */ 0, /* pos */ 0, &hidden_in);
    assert_eq!(hidden_out.len(), hidden_dim);

    // The only assertion we can make order-independently — finite
    // output. NaN/Inf would indicate uninitialized state being read or
    // a corrupted weight tensor.
    assert!(
        hidden_out.iter().all(|x| x.is_finite()),
        "[diff:layer_forward_dump_c] hidden_out contains NaN/Inf"
    );

    eprintln!(
        "[diff:layer_forward_dump_c] layer=0 hidden_dim={hidden_dim} \
         first 4 in={:?} out={:?}",
        &hidden_in[..4],
        &hidden_out[..4],
    );
}

/// Phase 4a: validate the Rust port's `memory_*` ops against the C
/// path on fresh state. End-to-end forward isn't ported yet, so the
/// diff is structural — empty-state queries and truncation arguments
/// must produce matching `pos_max` readings on both sides.
///
/// When 4f lands and `eval_prompt` is real, this test grows a
/// "prefill, truncate, compare pos_max" body that exercises a
/// non-empty cache. Today it's a structural-equivalence guard against
/// the layer-state allocation drifting out of sync with the C side.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn memory_ops_match_c_on_empty_state() {
    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();

    eprintln!(
        "[diff:memory_ops] model={} num_layers reachable via VARIANT",
        c.model_name(),
    );

    // Fresh ctx: pos_max equals 0 on both sides for every supported
    // variant (each has at least one full-attn layer, so the -1 sentinel
    // never fires).
    assert_eq!(c.memory_seq_pos_max(), rs.memory_seq_pos_max());
    assert_eq!(c.memory_seq_pos_max(), 0);

    // memory_clear is a no-op on empty state and must stay matched.
    c.memory_clear();
    rs.memory_clear();
    assert_eq!(c.memory_seq_pos_max(), rs.memory_seq_pos_max());

    // Several truncation argument shapes — empty state stays empty in
    // every case, but exercises the (p0, p1) argument plumbing on both
    // sides.
    for (p0, p1) in [(0, -1), (5, 10), (-1, -1), (100, 50), (0, 0)] {
        let c_ok = c.memory_seq_rm(p0, p1);
        let rs_ok = rs.memory_seq_rm(p0, p1);
        assert_eq!(
            c_ok, rs_ok,
            "[diff:memory_ops] memory_seq_rm({p0}, {p1}) return mismatch \
             (c={c_ok} rs={rs_ok})"
        );
        assert_eq!(
            c.memory_seq_pos_max(),
            rs.memory_seq_pos_max(),
            "[diff:memory_ops] pos_max mismatch after memory_seq_rm({p0}, {p1})"
        );
    }

    eprintln!("[diff:memory_ops] structural equivalence on empty state: OK");
}

#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn harness_loads() {
    let mut c: CBackend = open_backend();
    eprintln!(
        "[diff:scaffold] model={} n_vocab={} n_ctx={} eos={}",
        c.model_name(),
        c.n_vocab(),
        c.n_ctx(),
        c.eos(),
    );

    let prompt: [i32; 4] = [1, 200, 600, 1100];
    let logits = c.eval_prompt(&prompt, 0);
    assert_eq!(logits.len(), c.n_vocab());

    let arg = argmax(&logits);
    let top = topk(&logits, TOPK_K);
    let self_jac = jaccard(&top, &top);
    let self_cos = cosine_sim(&logits, &logits);
    eprintln!(
        "[diff:scaffold] argmax={arg} top-{TOPK_K}={top:?} \
         self-jaccard={self_jac:.4} self-cosine={self_cos:.5}"
    );

    // Sanity: comparing a vector against itself must hit 1.0 exactly
    // (within float). If these fail, the helpers are broken.
    assert!((self_jac - 1.0).abs() < 1e-6, "self-jaccard != 1.0");
    assert!((self_cos - 1.0).abs() < 1e-4, "self-cosine != 1.0");
}

// ---------------------------------------------------------------------------
// Phase 4f-6 — end-to-end eval_prompt / eval_token diff
// ---------------------------------------------------------------------------
//
// The file-level "end-to-end logits NOT useful" note (lines 21..37)
// is about *intra-Ctx* `memory_clear`-between calls — that path
// reveals C's lossy `memory_clear` semantic and produces cosine
// 0.65..0.76 for two C runs of the same prompt. The fresh-Ctx-per-
// test pattern below sidesteps that: each backend gets its own
// freshly-opened Ctx, runs one eval_prompt / eval_token, and we
// compare logits across backends. Both paths are deterministic given
// fresh state (Metal kernels are bit-exact per-PSO per the slice 9
// finding); the only float drift is CPU swiglu / sigmoid in the
// shared post-attention tail, which is identical word-by-word
// between C and Rust.
//
// Floor: cosine ≥ 0.9999 + top-1 argmax match. If 0.9999 fails on
// real runs, drop to 0.999 + Jaccard@20 ≥ 0.95 with a FIXME(riir):
// note pinpointing the bisect-by-layer finding.

const E2E_COSINE_FLOOR: f32 = 0.9999;

fn assert_e2e_logits_close(label: &str, c_logits: &[f32], rs_logits: &[f32]) {
    assert_eq!(c_logits.len(), rs_logits.len(), "[{label}] logits length");
    assert!(
        c_logits.iter().all(|x| x.is_finite()),
        "[{label}] C logits contain NaN/Inf"
    );
    assert!(
        rs_logits.iter().all(|x| x.is_finite()),
        "[{label}] Rust logits contain NaN/Inf"
    );

    let c_arg = argmax(c_logits);
    let rs_arg = argmax(rs_logits);
    let cos = cosine_sim(c_logits, rs_logits);
    let max_abs_c =
        c_logits.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let max_abs_diff = c_logits
        .iter()
        .zip(rs_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = max_abs_diff / max_abs_c.max(f32::EPSILON);
    let c_top = topk(c_logits, TOPK_K);
    let rs_top = topk(rs_logits, TOPK_K);
    let jac = jaccard(&c_top, &rs_top);

    eprintln!(
        "[diff:{label}] argmax c={c_arg} rs={rs_arg} cosine={cos:.7} \
         max_abs_diff={max_abs_diff:.3e} max_abs_c={max_abs_c:.3e} \
         rel={rel:.3e} top-{TOPK_K} jaccard={jac:.4}"
    );

    assert_eq!(
        c_arg, rs_arg,
        "[{label}] argmax mismatch (c={c_arg} rs={rs_arg})"
    );
    assert!(
        cos >= E2E_COSINE_FLOOR,
        "[{label}] cosine {cos:.7} below {E2E_COSINE_FLOOR}"
    );
}

/// One-token decode against a 4-token synthetic prefill. Both
/// backends open fresh, prefill `[1, 200, 600, 1100]` via
/// `eval_prompt` (state-update only on the first 3 tokens, emit on
/// the 4th — the per-step contract `mf_eval_prompt` advertises),
/// then run `eval_token(7)` at `pos = 4` and compare logits.
///
/// Exercises both code paths in slice 4f-5: the prefix loop in
/// `eval_prompt` (no-emit step_internal) and the single-step
/// `eval_token`. Catches positional bugs (e.g. dropped `pos`),
/// state-update vs emit divergence, and the final norm + lm_head
/// path that only runs on the emitting step.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn eval_token_matches_c_single_step() {
    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();

    // Prefill 4 synthetic tokens. eval_prompt emits logits for the
    // 4th; we discard them — they're a separate diff from the one
    // this test asserts.
    let prefill: [i32; 4] = [1, 200, 600, 1100];
    let _c_prefill_logits = c.eval_prompt(&prefill, 0);
    let _rs_prefill_logits = rs.eval_prompt(&prefill, 0);

    // Decode one more token at the next position.
    let next_token = 7i32;
    let next_pos = prefill.len();
    let c_logits = c.eval_token(next_token, next_pos);
    let rs_logits = rs.eval_token(next_token, next_pos);

    assert_e2e_logits_close(
        "eval_token_after_4tok_prefill",
        &c_logits,
        &rs_logits,
    );
}

// ---------------------------------------------------------------------------
// Phase 4g — state_save / state_load
// ---------------------------------------------------------------------------

/// `state_size()` on Rust must match `state_size()` on C after the
/// same prefill. The header is fixed-size and shape-driven; the body
/// scales with KV length on full-attn layers (15 of 40 on A3B) and
/// is fixed-size on linear-attn layers (45 of 40 on A3B). If sizes
/// disagree it means the wire-format math diverged from C.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn state_size_matches_c_after_prefill() {
    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();

    let prompt: [i32; 4] = [1, 200, 600, 1100];
    let _ = c.eval_prompt(&prompt, 0);
    let _ = rs.eval_prompt(&prompt, 0);

    let c_size = c.0.state_size();
    let rs_size = rs.0.state_size();

    eprintln!(
        "[diff:state_size_after_4tok] c={c_size} rs={rs_size}"
    );
    assert_eq!(
        c_size, rs_size,
        "state_size mismatch: c={c_size} rs={rs_size}"
    );

    // Sanity: should be > header bytes; KV grows with len.
    assert!(c_size > 32, "state_size {c_size} suspiciously small");
}

/// Rust↔Rust round-trip: prefill → save → memory_clear → load →
/// eval_token → compare to a fresh Rust eval that prefills + eval_
/// tokens without the save/load cycle. Both should produce
/// bit-identical logits since save/load preserves all per-layer
/// state and per-PSO Metal kernels are deterministic.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn state_round_trip_rust() {
    // Reference: fresh Ctx, prefill + eval_token without save/load.
    let mut rs_ref: RsBackend = open_backend();
    let prompt: [i32; 4] = [1, 200, 600, 1100];
    let next_token = 7i32;
    let next_pos = prompt.len();
    let _ = rs_ref.eval_prompt(&prompt, 0);
    let ref_logits = rs_ref.eval_token(next_token, next_pos);

    // Test path: fresh Ctx, prefill, save, memory_clear, load,
    // eval_token. Should match `ref_logits` exactly.
    let mut rs: RsBackend = open_backend();
    let _ = rs.eval_prompt(&prompt, 0);

    let snap_size = rs.0.state_size();
    let mut snap = vec![0u8; snap_size];
    let written = rs.0.state_save(&mut snap).expect("Rust state_save");
    assert_eq!(written, snap_size, "state_save wrote unexpected length");

    rs.memory_clear();
    rs.0.state_load(&snap).expect("Rust state_load");

    let test_logits = rs.eval_token(next_token, next_pos);

    assert_eq!(test_logits.len(), ref_logits.len());
    let drift_max = ref_logits
        .iter()
        .zip(test_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cos = cosine_sim(&ref_logits, &test_logits);
    eprintln!(
        "[diff:state_round_trip_rust] snap_bytes={snap_size} \
         max_abs_diff={drift_max:.3e} cosine={cos:.7}"
    );
    assert_eq!(
        argmax(&ref_logits),
        argmax(&test_logits),
        "round-trip changed argmax"
    );
    assert!(
        cos >= 0.9999,
        "round-trip cosine {cos:.7} below 0.9999"
    );
}

/// Wire compat (Rust → C): Rust prefills, saves; a fresh C Ctx
/// loads the snapshot and decodes one more token. The resulting
/// logits must match Rust's direct-eval continuation at the same
/// position. Confirms the Rust port writes bytes the C side can
/// read.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn state_load_c_from_rust_save() {
    let mut rs: RsBackend = open_backend();
    let prompt: [i32; 4] = [1, 200, 600, 1100];
    let next_token = 7i32;
    let next_pos = prompt.len();

    let _ = rs.eval_prompt(&prompt, 0);

    let snap_size = rs.0.state_size();
    let mut snap = vec![0u8; snap_size];
    rs.0.state_save(&mut snap).expect("Rust state_save");

    // Reference: Rust direct continuation.
    let rs_logits = rs.eval_token(next_token, next_pos);

    // Test: fresh C Ctx, load Rust snapshot, eval_token.
    let mut c: CBackend = open_backend();
    c.memory_clear();
    c.0.state_load(&snap).expect("C state_load(rust_snap)");
    let mut c_logits = vec![0.0f32; c.0.n_vocab()];
    c.0
        .eval_token(next_token, next_pos, 0, &mut c_logits)
        .expect("C eval_token after Rust state_load");

    assert_e2e_logits_close(
        "state_load_c_from_rust_save",
        &c_logits,
        &rs_logits,
    );
}

/// Wire compat (C → Rust): C prefills, saves; a fresh Rust Ctx
/// loads the snapshot and decodes one more token. The resulting
/// logits must match C's direct-eval continuation. Confirms the
/// Rust port reads C-produced snapshot bytes correctly.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn state_load_rust_from_c_save() {
    let mut c: CBackend = open_backend();
    let prompt: [i32; 4] = [1, 200, 600, 1100];
    let next_token = 7i32;
    let next_pos = prompt.len();

    let _ = c.eval_prompt(&prompt, 0);

    let snap_size = c.0.state_size();
    let mut snap = vec![0u8; snap_size];
    c.0.state_save(&mut snap).expect("C state_save");

    // Reference: C direct continuation.
    let mut c_logits = vec![0.0f32; c.0.n_vocab()];
    c.0
        .eval_token(next_token, next_pos, 0, &mut c_logits)
        .expect("C eval_token reference");

    // Test: fresh Rust Ctx, load C snapshot, eval_token.
    let mut rs: RsBackend = open_backend();
    rs.memory_clear();
    rs.0.state_load(&snap).expect("Rust state_load(c_snap)");
    let rs_logits = rs.eval_token(next_token, next_pos);

    assert_e2e_logits_close(
        "state_load_rust_from_c_save",
        &c_logits,
        &rs_logits,
    );
}

/// Multi-token prefill via `eval_prompt`. Both backends open fresh,
/// run `eval_prompt` over `[1..=8]`, compare last-token logits.
///
/// Catches positional bugs in the prefill loop (each step advances
/// `pos` by 1; if step_internal mishandles the position the per-
/// layer KV append in full_attn_layer_forward writes to the wrong
/// row and the final logits diverge).
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn eval_prompt_matches_c_multi_token() {
    let mut c: CBackend = open_backend();
    let mut rs: RsBackend = open_backend();

    let prompt: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let c_logits = c.eval_prompt(&prompt, 0);
    let rs_logits = rs.eval_prompt(&prompt, 0);

    assert_e2e_logits_close(
        "eval_prompt_8tok",
        &c_logits,
        &rs_logits,
    );
}

// ---------------------------------------------------------------------------
// Phase 5d-6b — speculative-prefetch correctness
// ---------------------------------------------------------------------------

/// The prefetch hit path (normal flow) and the all-miss path
/// (predictions cleared between every token) must produce
/// **bit-identical** logits. Per-PSO Metal kernels are deterministic
/// (slice 9 finding); the only difference between the two paths is
/// which buffer (`data_prefetch[slot]` vs `data_synced[slot]`) the
/// expert weights came from. Both buffers should hold identical
/// bytes for the same expert.
///
/// Catches: any bug where `data_prefetch[slot]` ends up loaded with
/// the wrong expert, or where the encoder binds the wrong buffer
/// for a given `SlotSource`.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn prefetch_hit_miss_equivalence_rust() {
    let prompt: [i32; 4] = [1, 200, 600, 1100];
    let next_token = 7i32;
    let next_pos = prompt.len();

    // Reference: normal eval with prefetch hits where they apply.
    let mut rs_normal: RsBackend = open_backend();
    let _ = rs_normal.eval_prompt(&prompt, 0);
    let normal_logits = rs_normal.eval_token(next_token, next_pos);

    // Test: same prompt+token, but clear prefetch predictions just
    // before the token-decode. With no predictions, every layer
    // takes the all-miss (sync-pread into data_synced) path.
    let mut rs_miss: RsBackend = open_backend();
    let _ = rs_miss.eval_prompt(&prompt, 0);
    rs_miss.0.clear_prefetch_predictions();
    let miss_logits = rs_miss.eval_token(next_token, next_pos);

    assert_eq!(normal_logits.len(), miss_logits.len());
    let drift_max = normal_logits
        .iter()
        .zip(miss_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cos = cosine_sim(&normal_logits, &miss_logits);
    eprintln!(
        "[diff:prefetch_hit_miss_equivalence] \
         max_abs_diff={drift_max:.3e} cosine={cos:.7} \
         argmax(normal)={a} argmax(miss)={b}",
        a = argmax(&normal_logits),
        b = argmax(&miss_logits),
    );
    assert_eq!(
        argmax(&normal_logits),
        argmax(&miss_logits),
        "prefetch hit and all-miss paths produced different argmax"
    );
    assert_eq!(
        drift_max, 0.0,
        "prefetch hit and all-miss paths should be bit-identical, \
         got drift {drift_max:.3e}"
    );
}

/// `step → memory_clear → step` must produce the same logits as a
/// fresh-Ctx `step → step`. Catches: prefetch state leaking across
/// `memory_clear` (stale predictions, in-flight prefetch not
/// drained, last_token_indices not cleared).
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn memory_clear_cancels_prefetch_no_leak() {
    let prompt_a: [i32; 4] = [1, 200, 600, 1100];
    let prompt_b: [i32; 4] = [2, 300, 700, 1200];
    let next_token = 7i32;
    let next_pos = prompt_b.len();

    // Reference: fresh ctx, eval prompt_b only, get next-token logits.
    let mut rs_ref: RsBackend = open_backend();
    let _ = rs_ref.eval_prompt(&prompt_b, 0);
    let ref_logits = rs_ref.eval_token(next_token, next_pos);

    // Test: same ctx, eval prompt_a, memory_clear, eval prompt_b,
    // get next-token logits. Should match ref_logits.
    let mut rs: RsBackend = open_backend();
    let _ = rs.eval_prompt(&prompt_a, 0);
    rs.memory_clear();
    let _ = rs.eval_prompt(&prompt_b, 0);
    let test_logits = rs.eval_token(next_token, next_pos);

    assert_eq!(test_logits.len(), ref_logits.len());
    let drift_max = ref_logits
        .iter()
        .zip(test_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cos = cosine_sim(&ref_logits, &test_logits);
    eprintln!(
        "[diff:memory_clear_cancels_prefetch] \
         max_abs_diff={drift_max:.3e} cosine={cos:.7}"
    );
    assert_eq!(
        argmax(&ref_logits),
        argmax(&test_logits),
        "memory_clear leaked prefetch state across reset"
    );
    assert!(
        cos >= 0.9999,
        "memory_clear leak: cosine {cos:.7} below 0.9999"
    );
}

/// Two consecutive `eval_token` calls with `clear_prefetch_predictions`
/// between them must produce the same logits as a fresh ctx running
/// the same sequence with no prefetch state at all. Catches: stale
/// `data_synced[slot]` bytes from token N polluting token N+1's
/// dispatch (would only happen if the parallel pread or the slot-
/// reuse contract were broken).
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn slot_reuse_race_regression_rust() {
    let prompt: [i32; 4] = [1, 200, 600, 1100];
    let token_t1 = 7i32;
    let token_t2 = 42i32;
    let pos_t1 = prompt.len();
    let pos_t2 = pos_t1 + 1;

    // Reference: fresh ctx per token, no prefetch state ever exists.
    let mut rs_ref1: RsBackend = open_backend();
    let _ = rs_ref1.eval_prompt(&prompt, 0);
    let _ = rs_ref1.eval_token(token_t1, pos_t1);
    let ref_t2 = rs_ref1.eval_token(token_t2, pos_t2);

    // Test: same ctx through both tokens, but clear predictions
    // before each call. Should match ref_t2.
    let mut rs: RsBackend = open_backend();
    let _ = rs.eval_prompt(&prompt, 0);
    rs.0.clear_prefetch_predictions();
    let _ = rs.eval_token(token_t1, pos_t1);
    rs.0.clear_prefetch_predictions();
    let test_t2 = rs.eval_token(token_t2, pos_t2);

    let drift_max = ref_t2
        .iter()
        .zip(test_t2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cos = cosine_sim(&ref_t2, &test_t2);
    eprintln!(
        "[diff:slot_reuse_race_regression] \
         max_abs_diff={drift_max:.3e} cosine={cos:.7}"
    );
    assert_eq!(
        argmax(&ref_t2),
        argmax(&test_t2),
        "slot-reuse race: argmax changed across consecutive evals"
    );
    assert!(
        cos >= 0.9999,
        "slot-reuse race regression: cosine {cos:.7} below 0.9999"
    );
}
