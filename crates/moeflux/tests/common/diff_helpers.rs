//! Shared diff-oracle helpers used by `diff_oracle.rs` and
//! `batched_diff_oracle.rs`.
//!
//! Cargo treats each `tests/<name>.rs` file as its own test binary,
//! so utility code that two test binaries both want lives in
//! `tests/common/` and gets pulled in via `mod common; use
//! common::diff_helpers::*;`. The per-binary `DiffBackend` traits stay
//! in their respective binaries — only the generic comparison utilities
//! and path resolution are factored out here.
//!
//! `#[allow(dead_code)]` is intentional: not every test binary uses
//! every helper (e.g., `diff_oracle.rs` redeclares per-test
//! `COSINE_FLOOR` inline and doesn't need the module constant), but
//! the symbols must exist here so the other binary can import them.

#![allow(dead_code)]

use std::path::PathBuf;

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
    if denom == 0.0 {
        1.0
    } else {
        (dot / denom) as f32
    }
}

// ---------------------------------------------------------------------------
// Tolerance constants
// ---------------------------------------------------------------------------

/// End-to-end logits tolerance (Metal MoE atomic-op nondeterminism).
pub const TOPK_K: usize = 20;
pub const TOPK_JACCARD_MIN: f32 = 0.95;
pub const COSINE_SIM_MIN: f32 = 0.99;

/// Intermediate-tensor diff tolerance (per-layer / per-kernel checks).
/// Tighter than the end-to-end floors because diffs happen before
/// Metal MoE nondeterminism has accumulated. Hoisted from the per-test
/// inline constants at the GPU diff sites.
pub const COSINE_FLOOR: f32 = 0.9999;
pub const REL_DIFF_FLOOR: f32 = 1e-3;

// ---------------------------------------------------------------------------
// Logit assertion
// ---------------------------------------------------------------------------

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

pub fn artifacts_dir() -> PathBuf {
    let default = "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-artifacts";
    PathBuf::from(std::env::var("MOEFLUX_SMOKE_ARTIFACTS").unwrap_or(default.into()))
}

pub fn root_dir() -> PathBuf {
    let default = "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-root";
    PathBuf::from(std::env::var("MOEFLUX_SMOKE_ROOT").unwrap_or(default.into()))
}

/// Standard A3B artifact paths used by every diff test. Each test
/// binary's typed `open_backend` builds its concrete backend from
/// these paths; the resolution logic lives in one place.
pub struct A3BPaths {
    pub weights: PathBuf,
    pub manifest: PathBuf,
    pub vocab: PathBuf,
    pub root: PathBuf,
    pub experts_per_tok: u32,
    pub use_2bit: bool,
}

pub fn default_a3b_paths() -> A3BPaths {
    let art = artifacts_dir();
    let root = root_dir();
    A3BPaths {
        weights: art.join("model_weights.bin"),
        manifest: art.join("model_weights.json"),
        vocab: art.join("vocab.bin"),
        root,
        experts_per_tok: 4,
        use_2bit: false,
    }
}
