//! End-to-end smoke for the Cogito-V2 / DeepSeek-V3 CPU MLA path.
//!
//! Exercises `Ctx::open` + `eval_token` against the real on-disk
//! 671B weights at `/Volumes/Temp Backup/models/blallama/cogito-v2-671b/`.
//! Asserts logits come back finite and non-degenerate. Doesn't check
//! token-level correctness — that's a follow-up once we wire blallama
//! and a tokenizer.
//!
//! `#[ignore]` because it needs ~352 GB of weights on disk and takes
//! ~10 s per token even single-step. Run with:
//!
//! ```text
//! cargo test -p moeflux \
//!     --features model-cogito-v2-671b --no-default-features \
//!     --release --test cogito_v2_smoke -- --ignored --nocapture
//! ```

#![cfg(all(target_os = "macos", feature = "model-cogito-v2-671b"))]

use std::path::PathBuf;

use moeflux::{Ctx, Error};

const ROOT: &str = "/Volumes/Temp Backup/models/blallama/cogito-v2-671b";

fn art(name: &str) -> PathBuf {
    PathBuf::from(ROOT).join("artifacts").join(name)
}

#[test]
#[ignore]
fn cogito_v2_eval_token_smoke() -> Result<(), Error> {
    let mut ctx = Ctx::open(
        &art("model_weights.bin"),
        &art("model_weights.json"),
        // vocab.bin is unused on the Rust path; HF tokenizer.json
        // lives at <root>/mlx/tokenizer.json and is loaded by
        // blallama, not moeflux.
        &PathBuf::from(ROOT).join("mlx/tokenizer.json"),
        &PathBuf::from(ROOT).join("root"),
        /* experts_per_tok = */ 8,
        /* use_2bit       = */ false,
    )?;

    let vocab = 128_815usize;
    let mut logits = vec![0.0f32; vocab];

    eprintln!("[cogito-smoke] running eval_token(BOS, pos=0)…");
    let t0 = std::time::Instant::now();
    // BOS = 0 per the variant config.
    ctx.eval_token(0, 0, 0, &mut logits)?;
    let elapsed = t0.elapsed();
    eprintln!(
        "[cogito-smoke] eval_token finished in {:.2}s",
        elapsed.as_secs_f64()
    );

    let finite = logits.iter().all(|v| v.is_finite());
    let nonzero = logits.iter().any(|&v| v != 0.0);
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let min = logits.iter().fold(f32::INFINITY, |m, &v| m.min(v));
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    eprintln!(
        "[cogito-smoke] logits: finite={finite} nonzero={nonzero} \
         min={min:.3} max={max:.3} argmax={argmax}",
    );

    assert!(finite, "logits contain NaN/Inf");
    assert!(nonzero, "logits are all zero");
    // Magnitude sanity — typical pre-softmax logits are in single-digit
    // to low-double-digit range. Anything > 1000 strongly suggests a
    // wiring bug (missing norm, wrong residual, etc.).
    assert!(
        max.abs() < 1000.0 && min.abs() < 1000.0,
        "logit magnitudes outside sane range (min={min}, max={max})",
    );
    Ok(())
}
