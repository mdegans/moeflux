//! Rust port of `tests/smoke.c`. Exercises the safe API end-to-end
//! against a real model dir. `#[ignore]` because it needs ~18 GB of
//! weights + packed experts on disk; run with
//! `cargo test -p moeflux --features model-qwen3-6-35b-a3b -- --ignored`.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use moeflux::{Ctx, Error};

fn model_root() -> PathBuf {
    // Default to the layout produced by the session's prep pipeline;
    // override with MOEFLUX_SMOKE_ROOT to point elsewhere.
    let default =
        "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-root";
    PathBuf::from(std::env::var("MOEFLUX_SMOKE_ROOT").unwrap_or(default.into()))
}

fn artifacts() -> PathBuf {
    let default =
        "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-artifacts";
    PathBuf::from(std::env::var("MOEFLUX_SMOKE_ARTIFACTS").unwrap_or(default.into()))
}

fn logits_look_sane(logits: &[f32], label: &str) {
    let finite = logits.iter().all(|v| v.is_finite());
    let nonzero = logits.iter().any(|&v| v != 0.0);
    assert!(finite, "[{label}] logits contain NaN/Inf");
    assert!(nonzero, "[{label}] logits are all zero");
}

#[test]
#[ignore]
fn rust_smoke_35b_a3b() {
    let art = artifacts();
    let root = model_root();

    let mut ctx = Ctx::open(
        &art.join("model_weights.bin"),
        &art.join("model_weights.json"),
        &art.join("vocab.bin"),
        &root,
        /* experts_per_tok */ 4,
        /* use_2bit */ false,
    )
    .expect("Ctx::open");

    eprintln!(
        "[smoke] model={} n_vocab={} n_ctx={} eos={}",
        ctx.model_name(),
        ctx.n_vocab(),
        ctx.n_ctx(),
        ctx.eos(),
    );
    assert!(ctx.n_vocab() > 0);

    let mut logits = vec![0.0f32; ctx.n_vocab()];
    let mut logits_prev = vec![0.0f32; ctx.n_vocab()];

    // Prefill 4 fabricated low-id tokens.
    let prompt = [1i32, 100, 500, 1000];
    ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill");
    logits_look_sane(&logits, "after-prefill");
    assert_eq!(ctx.memory_seq_pos_max(0), 4);

    // Three decodes, each must change logits.
    let mut next = 42i32;
    for step in 0..3 {
        logits_prev.copy_from_slice(&logits);
        ctx.eval_token(next, 4 + step, 0, &mut logits)
            .expect("decode");
        logits_look_sane(&logits, "after-decode");
        assert_ne!(&logits[..1024], &logits_prev[..1024], "step {step}");
        next = next.wrapping_mul(31).wrapping_add(7) % ctx.n_vocab() as i32;
    }

    // Truncate back to 4 via seq_rm.
    assert!(ctx.memory_seq_rm(0, 4, -1));
    assert_eq!(ctx.memory_seq_pos_max(0), 4);

    // State save/load round-trip. Clear + re-prefill so size matches
    // save (same ordering fix as tests/smoke.c).
    ctx.memory_clear();
    ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("re-prefill");

    let snap_size = ctx.state_size();
    assert!(snap_size > 0);
    let mut snap = vec![0u8; snap_size];
    let wrote = ctx.state_save(&mut snap).expect("state_save");
    assert_eq!(wrote, snap_size);

    // Mutate state, then restore.
    ctx.eval_token(42, 4, 0, &mut logits).expect("mutate-decode-1");
    ctx.eval_token(43, 5, 0, &mut logits).expect("mutate-decode-2");
    assert_eq!(ctx.memory_seq_pos_max(0), 6);

    ctx.state_load(&snap[..wrote]).expect("state_load");
    assert_eq!(ctx.memory_seq_pos_max(0), 4);

    ctx.memory_clear();
    assert_eq!(ctx.memory_seq_pos_max(0), 0);
    eprintln!("[smoke] PASS");
}

#[test]
fn open_with_missing_files_fails_cleanly() {
    // Non-existent paths → mf_init_model returns NULL → Error::InitFailed.
    // Does not need a real model.
    let bogus = PathBuf::from("/nonexistent/moeflux/path");
    let err = Ctx::open(&bogus, &bogus, &bogus, &bogus, 4, false).unwrap_err();
    assert!(matches!(err, Error::InitFailed));
}
