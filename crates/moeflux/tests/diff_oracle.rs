//! Differential test harness for the RIIR port.
//!
//! Gated to C-supported variants — variants without a C-side oracle
//! (`model-cogito-v2-671b`) skip this whole test file, since the
//! oracle backend doesn't exist for them.
//!
//! One implementation of [`DiffBackend`]:
//!
//! - **`RsBackend`** wraps the pure-Rust path ([`moeflux::riir::RsCtx`]).
//!   Used for Rust-only correctness tests (state round-trip, prefetch
//!   equivalence, chunked eval, prompt-cache scenarios, directional
//!   benchmarks).
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
//! ```bash
//! cargo test -p moeflux \
//!     --features "model-qwen3-6-35b-a3b" \
//!     --test diff_oracle --release \
//!     -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(all(
    target_os = "macos",
    any(feature = "model-qwen3-5-a17b", feature = "model-qwen3-6-35b-a3b",),
))]

use std::path::Path;
use std::time::Instant;

mod common;
use common::diff_helpers::{argmax, artifacts_dir, cosine_sim, default_a3b_paths};
use moeflux::riir::RsCtx;

// ---------------------------------------------------------------------------
// Trait + impls
// ---------------------------------------------------------------------------

/// Common behavior the harness exercises across backends.
///
/// Methods mirror [`moeflux::riir::RsCtx`]'s public API 1:1 — the diff
/// harness treats this surface as the boundary at which behavior
/// must agree. Each impl forwards to its underlying ctx; the
/// abstraction is purely a witness for trait-generic test code.
pub trait DiffBackend {
    fn open(
        weights: &Path,
        manifest: &Path,
        vocab: &Path,
        experts_dir: &Path,
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
    fn apply_rotary_emb(&self, pos: i32, q: &[f32], k: &[f32]) -> (Vec<f32>, Vec<f32>);

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
    fn rms_norm_gated_cpu(&self, weight_name: &str, eps: f32, x: &[f32], z: &[f32]) -> Vec<f32>;

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
    fn gpu_rms_norm_fused(&mut self, x: &[f32], weight_bf16: &[u8]) -> Vec<f32>;

    /// Single-expert GPU FFN forward (slice 9a). `expert_data` is one
    /// expert's `EXPERT_SIZE`-byte 4-bit blob; `h_post` is HIDDEN_DIM
    /// floats. Returns the HIDDEN_DIM-float expert output. Takes
    /// `&mut self` because the Rust backend builds the Metal device
    /// lazily on first GPU call.
    fn gpu_expert_forward(&mut self, expert_data: &[u8], h_post: &[f32]) -> Vec<f32>;

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
    fn attn_softmax_batched(&mut self, num_heads: u32, seq_len: u32, scores_in: &[f32])
    -> Vec<f32>;

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
    fn sigmoid_gate(&mut self, dim: u32, gate: &[f32], x_in: &[f32]) -> Vec<f32>;

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
    fn layer_forward_dump(&mut self, layer_idx: i32, pos: i32, hidden_in: &[f32]) -> Vec<f32>;

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

/// Pure-Rust impl. Phase 3: methods become real as their kernels are
/// ported (embedding landed; the rest still `todo!()`).
pub struct RsBackend(RsCtx);

impl DiffBackend for RsBackend {
    fn open(
        weights: &Path,
        manifest: &Path,
        vocab: &Path,
        experts_dir: &Path,
        use_2bit: bool,
    ) -> Self {
        Self(
            RsCtx::open(weights, manifest, vocab, experts_dir, use_2bit)
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

    fn apply_rotary_emb(&self, pos: i32, q: &[f32], k: &[f32]) -> (Vec<f32>, Vec<f32>) {
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

    fn rms_norm_gated_cpu(&self, weight_name: &str, eps: f32, x: &[f32], z: &[f32]) -> Vec<f32> {
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
                layer_idx, alpha, beta, q, k, v, v_heads, k_heads, key_dim, value_dim, &mut state,
                &mut out,
            )
            .expect("RsBackend gated_delta_recurrence_cpu");
        (state, out)
    }

    fn load_expert_bytes(&self, layer_idx: i32, expert_idx: i32) -> Vec<u8> {
        let mut out = vec![0u8; moeflux::riir::VARIANT.expert_size_4bit()];
        self.0
            .load_expert_bytes(layer_idx as usize, expert_idx as usize, &mut out)
            .expect("RsBackend load_expert_bytes");
        out
    }

    fn gpu_rms_norm_fused(&mut self, x: &[f32], weight_bf16: &[u8]) -> Vec<f32> {
        let mut out = vec![0.0f32; moeflux::riir::VARIANT.hidden_dim];
        self.0
            .gpu_rms_norm_fused(x, weight_bf16, &mut out)
            .expect("RsBackend gpu_rms_norm_fused");
        out
    }

    fn gpu_expert_forward(&mut self, expert_data: &[u8], h_post: &[f32]) -> Vec<f32> {
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
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                q,
                k_cache,
                scale,
                &mut out,
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
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                scores,
                v_cache,
                &mut out,
            )
            .expect("RsBackend attn_values_batched");
        out
    }

    fn sigmoid_gate(&mut self, dim: u32, gate: &[f32], x_in: &[f32]) -> Vec<f32> {
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

    fn layer_forward_dump(&mut self, layer_idx: i32, pos: i32, hidden_in: &[f32]) -> Vec<f32> {
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
// Backend opener (path resolution + tolerance constants live in
// `tests/common/diff_helpers.rs` so they're sharable with the
// `batched_diff_oracle.rs` test binary).
// ---------------------------------------------------------------------------

/// Open a backend with the standard A3B artifacts layout. Used by
/// every test in the harness so the path resolution lives in one
/// place.
pub fn open_backend<B: DiffBackend>() -> B {
    let p = default_a3b_paths();
    B::open(&p.weights, &p.manifest, &p.vocab, &p.root, p.use_2bit)
}

// ---------------------------------------------------------------------------
// Phase 0 sanity test — validates the harness itself
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Phase 4c — layer-by-layer Rust-only correctness
// ---------------------------------------------------------------------------

/// Rust-only back-to-back layer-forward sanity. Five consecutive
/// `layer_forward_dump` calls on the same layer with `memory_clear`
/// between must produce bit-identical outputs — catches deferred-
/// expert state leaking across calls.
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

// ---------------------------------------------------------------------------
// Phase 4f-6 — Rust-only eval_prompt / eval_token / state round-trip
// ---------------------------------------------------------------------------

/// Rust state save/load round-trip. Prefill, snapshot, memory_clear,
/// reload, decode one token — must match the direct-eval continuation
/// at the same position.
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
    assert!(cos >= 0.9999, "round-trip cosine {cos:.7} below 0.9999");
}

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

/// Phase 6 scaffolding: canonical `eval_prompt` produces last-token
/// logits matching a tokenwise reference built from per-token
/// `eval_token` calls (which route through
/// `step_internal_per_token_oracle`).
///
/// Session-3 status: `RsCtx::step_internal` is currently a tokenwise
/// loop calling the per-token oracle, so this test trivially passes
/// on the per-PSO-determinism floor. When session 4 swaps the loop
/// body for the GPU batched-prefill primitives landed in sessions 1-3
/// (`encode_moe_batched_permute_fuse`, causal-masked tiled SDPA,
/// batched matmul), the two paths diverge in implementation while
/// staying equivalent in output — the FP-reorder envelope from
/// per-bucket vs per-slot MoE accumulation should keep cosine ≥
/// 0.9999, the same floor the synthetic Phase 4 diff test hit at
/// cosine = 1.000000000.
///
/// Catches: (a) regressions in `eval_prompt`'s emission contract,
/// (b) any future per-layer batched primitive that diverges from the
/// per-token oracle beyond the FP-reorder envelope, (c) state
/// advancement bugs (KV append position, deferred ring drain,
/// prefetch interactions).
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn eval_prompt_matches_per_token_oracle() {
    let prompt: [i32; 16] = [
        1, 200, 600, 1100, 2, 300, 700, 1200, 3, 400, 800, 1300, 4, 500, 900, 1400,
    ];
    let next_token = 7i32;
    let next_pos = prompt.len();

    // Reference path: per-token oracle via `eval_token` per position.
    // `eval_token` always emits logits, so the buffer ends up holding
    // the last token's logits after the loop.
    let mut rs_ref: RsBackend = open_backend();
    let n_vocab = rs_ref.0.n_vocab();
    let mut ref_prompt_logits = vec![0.0f32; n_vocab];
    for (i, &tok) in prompt.iter().enumerate() {
        rs_ref
            .0
            .eval_token(tok, i, 0, &mut ref_prompt_logits)
            .expect("oracle eval_token");
    }
    let ref_continuation = rs_ref.eval_token(next_token, next_pos);

    // Test path: canonical `eval_prompt` (slice-taking).
    let mut rs: RsBackend = open_backend();
    let mut prompt_logits = vec![0.0f32; n_vocab];
    rs.0.eval_prompt(&prompt, 0, 0, &mut prompt_logits)
        .expect("canonical eval_prompt");
    let test_continuation = rs.eval_token(next_token, next_pos);

    // Compare both the end-of-prompt logits and the post-prompt
    // continuation. End-of-prompt catches divergence inside the
    // prefill path; continuation catches divergence in the KV state
    // left behind after the prompt completes.
    let prompt_cos = cosine_sim(&ref_prompt_logits, &prompt_logits);
    let prompt_drift = ref_prompt_logits
        .iter()
        .zip(prompt_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cont_cos = cosine_sim(&ref_continuation, &test_continuation);
    let cont_drift = ref_continuation
        .iter()
        .zip(test_continuation.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[diff:eval_prompt_matches_per_token_oracle] \
         prompt cosine={prompt_cos:.7} max_abs_diff={prompt_drift:.3e} | \
         continuation cosine={cont_cos:.7} max_abs_diff={cont_drift:.3e}"
    );
    assert_eq!(
        argmax(&ref_prompt_logits),
        argmax(&prompt_logits),
        "eval_prompt last-token argmax diverged from oracle"
    );
    assert_eq!(
        argmax(&ref_continuation),
        argmax(&test_continuation),
        "post-prompt continuation argmax diverged from oracle"
    );
    assert!(
        prompt_cos >= 0.9999,
        "eval_prompt last-token cosine {prompt_cos:.7} below 0.9999"
    );
    assert!(
        cont_cos >= 0.9999,
        "post-prompt continuation cosine {cont_cos:.7} below 0.9999"
    );
}

/// Phase D — chunkwise iteration. With `BATCHED_CHUNK_SIZE` overridden
/// to 4 via the test hook, a 16-token prompt evaluates as 4 chunks of
/// 4 tokens each (instead of one chunk of 16). The end-of-prompt
/// logits must match the per-token oracle within the same FP-reorder
/// envelope that single-chunk eval_prompt achieves. Catches:
/// chunk-boundary `start_pos` arithmetic, KV state advance across
/// chunks, scratch buffer reuse, last-chunk logits emission gating.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn eval_prompt_chunked_matches_eval_prompt_whole_prompt() {
    let prompt: [i32; 16] = [
        1, 200, 600, 1100, 2, 300, 700, 1200, 3, 400, 800, 1300, 4, 500, 900, 1400,
    ];
    let next_token = 7i32;
    let next_pos = prompt.len();

    let mut rs_ref: RsBackend = open_backend();
    let n_vocab = rs_ref.0.n_vocab();
    let mut ref_prompt_logits = vec![0.0f32; n_vocab];
    for (i, &tok) in prompt.iter().enumerate() {
        rs_ref
            .0
            .eval_token(tok, i, 0, &mut ref_prompt_logits)
            .expect("oracle eval_token");
    }
    let ref_continuation = rs_ref.eval_token(next_token, next_pos);

    // Test path: chunk size = 4, so the 16-token prompt evaluates as
    // 4 chunks of 4 tokens each. Restore the override at the end so
    // later tests on the same thread use the production default.
    moeflux::riir::set_batched_chunk_size_for_test(Some(4));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut rs: RsBackend = open_backend();
        let mut prompt_logits = vec![0.0f32; n_vocab];
        rs.0.eval_prompt(&prompt, 0, 0, &mut prompt_logits)
            .expect("chunked eval_prompt");
        let test_continuation = rs.eval_token(next_token, next_pos);
        (prompt_logits, test_continuation)
    }));
    moeflux::riir::set_batched_chunk_size_for_test(None);
    let (prompt_logits, test_continuation) = match result {
        Ok(t) => t,
        Err(payload) => std::panic::resume_unwind(payload),
    };

    let prompt_cos = cosine_sim(&ref_prompt_logits, &prompt_logits);
    let prompt_drift = ref_prompt_logits
        .iter()
        .zip(prompt_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cont_cos = cosine_sim(&ref_continuation, &test_continuation);
    let cont_drift = ref_continuation
        .iter()
        .zip(test_continuation.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[diff:eval_prompt_chunked_matches_eval_prompt_whole_prompt] \
         chunk=4 prompt cosine={prompt_cos:.7} max_abs={prompt_drift:.3e} | \
         continuation cosine={cont_cos:.7} max_abs={cont_drift:.3e}"
    );
    assert_eq!(
        argmax(&ref_prompt_logits),
        argmax(&prompt_logits),
        "chunked eval_prompt last-token argmax diverged from oracle"
    );
    assert_eq!(
        argmax(&ref_continuation),
        argmax(&test_continuation),
        "post-chunked-prompt continuation argmax diverged from oracle"
    );
    assert!(
        prompt_cos >= 0.9999,
        "chunked eval_prompt cosine {prompt_cos:.7} below 0.9999"
    );
    assert!(
        cont_cos >= 0.9999,
        "post-chunked continuation cosine {cont_cos:.7} below 0.9999"
    );
}

/// Diagnostic: B2 batched-SDPA cosine at chunk_size=1. If this passes
/// but multi-token N fails, the bug is in cross-token state (Phase 1
/// pre-SDPA per-token, Phase 2 batched SDPA across tokens, or Phase 3
/// post-SDPA per-token). If this fails, the bug is in single-token
/// orchestration of pre-SDPA + tiled SDPA + post-SDPA.
#[test]
#[ignore = "long running; needs moeflux artifacts; diagnostic"]
fn diag_b2_eval_prompt_chunk_1() {
    let prompt: [i32; 16] = [
        1, 200, 600, 1100, 2, 300, 700, 1200, 3, 400, 800, 1300, 4, 500, 900, 1400,
    ];
    let next_token = 7i32;
    let next_pos = prompt.len();

    let mut rs_ref: RsBackend = open_backend();
    let n_vocab = rs_ref.0.n_vocab();
    let mut ref_logits = vec![0.0f32; n_vocab];
    for (i, &tok) in prompt.iter().enumerate() {
        rs_ref
            .0
            .eval_token(tok, i, 0, &mut ref_logits)
            .expect("oracle eval_token");
    }
    let ref_cont = rs_ref.eval_token(next_token, next_pos);

    moeflux::riir::set_batched_chunk_size_for_test(Some(1));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut rs: RsBackend = open_backend();
        let mut prompt_logits = vec![0.0f32; n_vocab];
        rs.0.eval_prompt(&prompt, 0, 0, &mut prompt_logits)
            .expect("chunked eval_prompt @ chunk=1");
        let cont = rs.eval_token(next_token, next_pos);
        (prompt_logits, cont)
    }));
    moeflux::riir::set_batched_chunk_size_for_test(None);
    let (prompt_logits, test_cont) = match result {
        Ok(t) => t,
        Err(payload) => std::panic::resume_unwind(payload),
    };
    let prompt_cos = cosine_sim(&ref_logits, &prompt_logits);
    let cont_cos = cosine_sim(&ref_cont, &test_cont);
    eprintln!("[diag:b2_chunk_1] prompt_cos={prompt_cos:.7} cont_cos={cont_cos:.7}");
    assert!(
        prompt_cos >= 0.9999,
        "chunk_size=1 prompt cosine {prompt_cos:.7}"
    );
    assert!(cont_cos >= 0.9999, "chunk_size=1 cont cosine {cont_cos:.7}");
}

/// Phase F headline — directional bench of batched eval_prompt vs
/// per-token oracle on a 256-token synthetic prompt.
///
/// Run with `cargo test --release ... bench_batched_eval_prompt_vs_per_token
/// -- --ignored --nocapture`. **Not a proper bench**: single iteration,
/// no reboot between, no power-mode control. The proper bench protocol
/// (n≥3, reboot between revisions, high-perf power) lives at
/// drama_llama/.claude/memory/feedback_bench_discipline.md. Use this for
/// in-session directional answers; the headline number for memos comes
/// from the protocol-compliant bench.
///
/// Synthetic prompt = repeating token 200, 256 tokens. Real-world
/// prompts have richer routing distributions; this is a controlled
/// floor on the I/O-batching win since 256 repeated tokens still
/// route to ~all experts via per-position routing.
#[test]
#[ignore = "long running; needs moeflux artifacts; directional only"]
fn bench_batched_eval_prompt_vs_per_token() {
    const N: usize = 256;
    let prompt: Vec<i32> = (0..N).map(|i| ((i * 37 + 5) % 50000 + 1) as i32).collect();

    // Path A: per-token oracle via eval_token loop.
    let mut rs_oracle: RsBackend = open_backend();
    let n_vocab = rs_oracle.0.n_vocab();
    let mut oracle_logits = vec![0.0f32; n_vocab];
    let t0 = Instant::now();
    for (i, &tok) in prompt.iter().enumerate() {
        rs_oracle
            .0
            .eval_token(tok, i, 0, &mut oracle_logits)
            .expect("oracle eval_token");
    }
    let oracle_elapsed = t0.elapsed();

    // Path B: canonical batched eval_prompt (single chunk since N=256
    // < CHUNK_SIZE=8192).
    let mut rs_batched: RsBackend = open_backend();
    let mut batched_logits = vec![0.0f32; n_vocab];
    let t1 = Instant::now();
    rs_batched
        .0
        .eval_prompt(&prompt, 0, 0, &mut batched_logits)
        .expect("batched eval_prompt");
    let batched_elapsed = t1.elapsed();

    let oracle_tok_s = N as f64 / oracle_elapsed.as_secs_f64();
    let batched_tok_s = N as f64 / batched_elapsed.as_secs_f64();
    let speedup = batched_tok_s / oracle_tok_s;

    eprintln!(
        "[bench:eval_prompt_vs_per_token N={N}] \
         per-token: {oracle_elapsed:?} ({oracle_tok_s:.2} tok/s) | \
         batched: {batched_elapsed:?} ({batched_tok_s:.2} tok/s) | \
         speedup: {speedup:.2}×"
    );

    // Cosine sanity — but we already verify this elsewhere with
    // higher precision. Just make sure the bench wasn't a no-op.
    let cos = cosine_sim(&oracle_logits, &batched_logits);
    eprintln!("[bench:eval_prompt_vs_per_token] sanity cosine={cos:.7}");
    assert!(cos >= 0.99, "bench cosine {cos:.7} below sanity floor");
}

/// Phase G — decode regression bench. Compares the per-token oracle
/// (`eval_token` loop hitting `step_internal_per_token_oracle` directly)
/// against the batched-path-via-N=1 (`eval_prompt(&[tok], pos, ...)`
/// hitting `step_internal_batched_gqa` with chunk size effectively 1
/// since the prompt has length 1).
///
/// Phase G's go/no-go: if batched-decode is within 5% of per-token at
/// small kv_len AND faster at large kv_len, route eval_token through
/// the batched path and remove the per-token attn kernels.
///
/// **Not protocol-compliant** (n=1, no reboot). Directional only.
#[test]
#[ignore = "long running; needs moeflux artifacts; directional only"]
fn bench_decode_per_token_vs_batched_n1() {
    const PROMPT_LEN: usize = 32; // warm prefix
    const DECODE_N: usize = 32;
    let prompt: Vec<i32> = (0..PROMPT_LEN)
        .map(|i| ((i * 37 + 5) % 50000 + 1) as i32)
        .collect();

    // Path A: per-token oracle via eval_token.
    let mut rs_oracle: RsBackend = open_backend();
    let n_vocab = rs_oracle.0.n_vocab();
    // Warm-up prefill via the oracle path.
    let mut prompt_logits = vec![0.0f32; n_vocab];
    for (i, &tok) in prompt.iter().enumerate() {
        rs_oracle
            .0
            .eval_token(tok, i, 0, &mut prompt_logits)
            .expect("oracle warm-up");
    }
    let mut last_logits = prompt_logits.clone();
    let t0 = Instant::now();
    for d in 0..DECODE_N {
        // Greedy next token.
        let next_tok = argmax(&last_logits) as i32;
        rs_oracle
            .0
            .eval_token(next_tok, PROMPT_LEN + d, 0, &mut last_logits)
            .expect("oracle decode");
    }
    let oracle_elapsed = t0.elapsed();
    let oracle_decode_tok_s = DECODE_N as f64 / oracle_elapsed.as_secs_f64();

    // Path B: batched-path with N=1 chunks via eval_prompt(&[tok], pos).
    let mut rs_batched: RsBackend = open_backend();
    // Warm-up prefill via eval_prompt (batched path).
    let mut prompt_logits_b = vec![0.0f32; n_vocab];
    rs_batched
        .0
        .eval_prompt(&prompt, 0, 0, &mut prompt_logits_b)
        .expect("batched warm-up");
    let mut last_logits_b = prompt_logits_b.clone();
    let t1 = Instant::now();
    for d in 0..DECODE_N {
        let next_tok = argmax(&last_logits_b) as i32;
        rs_batched
            .0
            .eval_prompt(&[next_tok], PROMPT_LEN + d, 0, &mut last_logits_b)
            .expect("batched decode N=1");
    }
    let batched_elapsed = t1.elapsed();
    let batched_decode_tok_s = DECODE_N as f64 / batched_elapsed.as_secs_f64();

    let regression = (oracle_decode_tok_s - batched_decode_tok_s) / oracle_decode_tok_s * 100.0;
    eprintln!(
        "[bench:decode_per_token_vs_batched_n1] kv_start={PROMPT_LEN} \
         decode_n={DECODE_N} | per-token: {oracle_elapsed:?} \
         ({oracle_decode_tok_s:.2} tok/s) | batched-N1: \
         {batched_elapsed:?} ({batched_decode_tok_s:.2} tok/s) | \
         regression: {regression:.1}%"
    );
    // Sanity: both should produce the same greedy trajectory.
    let cos = cosine_sim(&last_logits, &last_logits_b);
    eprintln!("[bench:decode_per_token_vs_batched_n1] final-logit cos={cos:.7}");
    assert!(
        cos >= 0.99,
        "decode bench cosine {cos:.7} below sanity floor — \
         per-token and batched-N1 diverged greedily"
    );
}

/// Phase D — prompt-cache scenario. Eval a "cached prefix" on ctx_A,
/// snapshot, reset, load, then eval the suffix with
/// `start_pos = prefix.len()`. The resulting continuation logits must
/// match a control: eval the full `prefix ++ suffix` on a fresh ctx_B
/// from `start_pos = 0`.
///
/// This is the cache-hit pattern drama_llama's prefix-reuse layer
/// uses (Session: hash-keyed lookup → state_load → eval_prompt(suffix,
/// cached_pos)). Failure mode: KV state at `start_pos` doesn't quite
/// match what a fresh forward at the same position produces — could
/// be a snapshot serialization bug, a chunk-boundary arithmetic bug,
/// or a linear-attn recurrent-state issue.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn prompt_cache_start_pos_nonzero_matches() {
    let prefix: [i32; 4] = [1, 200, 600, 1100];
    let suffix: [i32; 5] = [2, 300, 700, 1200, 3];
    let next_token = 7i32;
    let full_pos = prefix.len() + suffix.len();

    // Control: full prompt from start_pos=0.
    let mut rs_ctrl: RsBackend = open_backend();
    let mut full_prompt = Vec::with_capacity(full_pos);
    full_prompt.extend_from_slice(&prefix);
    full_prompt.extend_from_slice(&suffix);
    let n_vocab = rs_ctrl.0.n_vocab();
    let mut ctrl_prompt_logits = vec![0.0f32; n_vocab];
    rs_ctrl
        .0
        .eval_prompt(&full_prompt, 0, 0, &mut ctrl_prompt_logits)
        .expect("control eval_prompt");
    let ctrl_continuation = rs_ctrl.eval_token(next_token, full_pos);

    // Test: prefix → snapshot → reset → load → suffix at start_pos=4.
    let mut rs: RsBackend = open_backend();
    let mut _prefix_logits = vec![0.0f32; n_vocab];
    rs.0.eval_prompt(&prefix, 0, 0, &mut _prefix_logits)
        .expect("prefix eval_prompt");

    let snap_size = rs.0.state_size();
    let mut snap = vec![0u8; snap_size];
    rs.0.state_save(&mut snap).expect("state_save");

    rs.memory_clear();
    rs.0.state_load(&snap).expect("state_load");

    let mut test_prompt_logits = vec![0.0f32; n_vocab];
    rs.0.eval_prompt(&suffix, prefix.len(), 0, &mut test_prompt_logits)
        .expect("suffix eval_prompt at start_pos != 0");
    let test_continuation = rs.eval_token(next_token, full_pos);

    let prompt_cos = cosine_sim(&ctrl_prompt_logits, &test_prompt_logits);
    let prompt_drift = ctrl_prompt_logits
        .iter()
        .zip(test_prompt_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cont_cos = cosine_sim(&ctrl_continuation, &test_continuation);
    let cont_drift = ctrl_continuation
        .iter()
        .zip(test_continuation.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[diff:prompt_cache_start_pos_nonzero_matches] \
         prompt cosine={prompt_cos:.7} max_abs={prompt_drift:.3e} | \
         continuation cosine={cont_cos:.7} max_abs={cont_drift:.3e}"
    );
    assert_eq!(
        argmax(&ctrl_prompt_logits),
        argmax(&test_prompt_logits),
        "prompt-cache last-token argmax diverged from control"
    );
    assert!(
        prompt_cos >= 0.9999,
        "prompt-cache prompt cosine {prompt_cos:.7} below 0.9999"
    );
    assert!(
        cont_cos >= 0.9999,
        "prompt-cache continuation cosine {cont_cos:.7} below 0.9999"
    );
}
