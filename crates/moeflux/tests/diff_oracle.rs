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
pub struct CBackend(Ctx);

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

/// Pure-Rust impl. Phase 0: stub — every method panics. Phase 4
/// onwards: real implementation.
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
