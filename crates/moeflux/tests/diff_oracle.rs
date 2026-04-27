//! Differential test harness for the RIIR port.
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
//!     --features "model-qwen3-6-35b-a3b,riir-port" \
//!     --test diff_oracle --release \
//!     -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(all(target_os = "macos", feature = "riir-port"))]

use std::path::{Path, PathBuf};

use moeflux::{riir::RsCtx, Ctx};

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
    moeflux::riir::variants::assert_matches_c(&c.0);
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
