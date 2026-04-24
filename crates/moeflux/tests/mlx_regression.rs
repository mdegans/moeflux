//! MLX regression test — moeflux vs a pre-captured MLX reference.
//!
//! MLX is the trusted source-of-truth for this codebase's output
//! (higher confidence than llama.cpp). This test asserts that
//! moeflux's top-K logits on a fixed prompt agree with MLX's to
//! within a tight threshold. It's the regression test that would
//! have caught the A3B gate_proj-offset bug immediately (moeflux
//! commit 925f7a0) — pre-fix, argmax wouldn't match; post-fix,
//! top-20 overlap is 20/20.
//!
//! ## Regenerating fixtures
//!
//! When a new variant is added, or when moeflux's numerical
//! behavior is intentionally changed (e.g. bf16 migration), regen
//! the goldens via:
//!
//! ```text
//! uv run --with mlx --with mlx-lm python3 \
//!   metal_infer/tests/mlx_reference/generate_goldens.py \
//!   --model <mlx-model-dir> --variant <slug> \
//!   --out crates/moeflux/tests/fixtures/mlx_golden_<slug>.txt
//! ```
//!
//! ## Running
//!
//! ```text
//! # A3B (18 GB model)
//! cargo test -p moeflux --features model-qwen3-6-35b-a3b \
//!   mlx_regression_a3b -- --ignored
//!
//! # A17B (~220 GB model)
//! cargo test -p moeflux --features model-qwen3-5-a17b \
//!   mlx_regression_a17b -- --ignored
//! ```

#![cfg(target_os = "macos")]

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use moeflux::Ctx;

/// "The quick brown fox" per the Qwen3 tokenizer. Shared across all
/// qwen3_5_moe variants (same vocab).
const PROMPT_TOKENS: [i32; 4] = [760, 3841, 13477, 37550];

/// Top-K overlap threshold. moeflux's post-fix result is 20/20;
/// threshold is 19/20 (95%) to allow one cosmetic swap between
/// adjacent logits that differ by < 0.1.
const TOP_K: usize = 20;
const MIN_OVERLAP: usize = 19;

struct Golden {
    argmax: u32,
    top: Vec<(u32, f32)>, // (token_id, logit), sorted desc by logit
}

fn parse_golden(path: &Path) -> Option<Golden> {
    let text = std::fs::read_to_string(path).ok()?;
    Some(parse_golden_text(&text))
}

fn parse_golden_text(text: &str) -> Golden {
    let mut argmax = None;
    let mut top = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# variant=") {
            // "variant=<slug> prompt=<...> argmax=<N> vocab=<...> top=<...>"
            for field in rest.split_whitespace() {
                if let Some(v) = field.strip_prefix("argmax=") {
                    argmax = Some(v.parse::<u32>().expect("argmax parse"));
                }
            }
            continue;
        }
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let mut parts = line.split('|');
        let _rank = parts.next().expect("rank");
        let tok: u32 = parts.next().expect("tok").parse().expect("tok int");
        let logit: f32 = parts.next().expect("logit").parse().expect("logit f32");
        top.push((tok, logit));
    }
    Golden {
        argmax: argmax.expect("golden header missing argmax"),
        top,
    }
}

fn load_golden_or_skip(path: &Path, variant: &str) -> Golden {
    match parse_golden(path) {
        Some(g) => g,
        None => {
            // Emit a clear skip message rather than panicking. A17B's
            // MLX golden needs a host with ~220 GB RAM to regenerate
            // (the MLX 4-bit checkpoint is 209 GB; MLX loads it all at
            // once, unlike moeflux's streaming path). If you're on
            // such a host, see the generator invocation at the top of
            // this file.
            eprintln!(
                "[{variant}] SKIP  fixture not found at {path:?} — \
                 regenerate with metal_infer/tests/mlx_reference/generate_goldens.py"
            );
            std::process::exit(0);
        }
    }
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn run_moeflux(art: &Path, root: &Path, experts_per_tok: u32) -> Vec<f32> {
    let mut ctx = Ctx::open(
        &art.join("model_weights.bin"),
        &art.join("model_weights.json"),
        &art.join("vocab.bin"),
        root,
        experts_per_tok,
        /* use_2bit */ false,
    )
    .expect("Ctx::open");
    let mut logits = vec![0.0f32; ctx.n_vocab()];
    ctx.eval_prompt(&PROMPT_TOKENS, 0usize, 0, &mut logits)
        .expect("eval_prompt");
    logits
}

fn argmax_of(logits: &[f32]) -> u32 {
    let (idx, _) = logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        });
    idx as u32
}

fn top_k_set(logits: &[f32], k: usize) -> HashSet<u32> {
    let mut pairs: Vec<(f32, u32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    pairs.into_iter().take(k).map(|(_, i)| i).collect()
}

fn assert_matches_golden(logits: &[f32], golden: &Golden, variant: &str) {
    let mf_argmax = argmax_of(logits);
    assert_eq!(
        mf_argmax, golden.argmax,
        "[{variant}] moeflux argmax {} != MLX argmax {}",
        mf_argmax, golden.argmax
    );

    let mf_top: HashSet<u32> = top_k_set(logits, TOP_K);
    let mx_top: HashSet<u32> = golden.top.iter().take(TOP_K).map(|(id, _)| *id).collect();
    let overlap = mf_top.intersection(&mx_top).count();
    assert!(
        overlap >= MIN_OVERLAP,
        "[{variant}] top-{TOP_K} overlap {overlap}/{TOP_K} below threshold {MIN_OVERLAP}.\n  \
         moeflux: {:?}\n  mlx:     {:?}\n  mf-only: {:?}\n  mx-only: {:?}",
        {
            let mut v: Vec<u32> = mf_top.iter().copied().collect();
            v.sort();
            v
        },
        {
            let mut v: Vec<u32> = mx_top.iter().copied().collect();
            v.sort();
            v
        },
        {
            let mut v: Vec<u32> = mf_top.difference(&mx_top).copied().collect();
            v.sort();
            v
        },
        {
            let mut v: Vec<u32> = mx_top.difference(&mf_top).copied().collect();
            v.sort();
            v
        },
    );

    // Diagnostic (always emitted on pass — useful for noticing drift
    // before it crosses the threshold).
    let mf_top_logit = logits[mf_argmax as usize];
    let mx_top_logit = golden.top[0].1;
    eprintln!(
        "[{variant}] PASS  argmax={mf_argmax}  overlap={overlap}/{TOP_K}  \
         top-1 logit moeflux={mf_top_logit:.4} mlx={mx_top_logit:.4}"
    );
}

#[cfg(feature = "model-qwen3-6-35b-a3b")]
#[test]
#[ignore]
fn mlx_regression_a3b() {
    let art = PathBuf::from(
        std::env::var("MOEFLUX_SMOKE_ARTIFACTS")
            .unwrap_or("/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-artifacts".into()),
    );
    let root = PathBuf::from(
        std::env::var("MOEFLUX_SMOKE_ROOT")
            .unwrap_or("/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-root".into()),
    );
    let golden = load_golden_or_skip(
        &fixtures_dir().join("mlx_golden_qwen3-6-35b-a3b.txt"),
        "qwen3-6-35b-a3b",
    );
    let logits = run_moeflux(&art, &root, /* K = */ 8);
    assert_matches_golden(&logits, &golden, "qwen3-6-35b-a3b");
}

#[cfg(feature = "model-qwen3-5-a17b")]
#[test]
#[ignore]
fn mlx_regression_a17b() {
    let art = PathBuf::from(
        std::env::var("MOEFLUX_SMOKE_ARTIFACTS")
            .unwrap_or("/Volumes/Temp Backup/models/moeflux/qwen3-5-a17b-artifacts".into()),
    );
    let root = PathBuf::from(
        std::env::var("MOEFLUX_SMOKE_ROOT")
            .unwrap_or("/Volumes/Temp Backup/models/moeflux/qwen3-5-a17b-root".into()),
    );
    let golden = load_golden_or_skip(
        &fixtures_dir().join("mlx_golden_qwen3-5-a17b.txt"),
        "qwen3-5-a17b",
    );
    // A17B's config specifies num_experts_per_tok=10, but moeflux's MAX_K=8
    // truncates internally — pass 10 here to match MLX's top-K selection
    // (moeflux still only runs top 8, but it's the same 8 MLX picks).
    let logits = run_moeflux(&art, &root, /* K = */ 10);
    assert_matches_golden(&logits, &golden, "qwen3-5-a17b");
}
