//! Phase 7 — checkpoint/restore on `Ctx`.
//!
//! Lossless rewinds at breakpoint positions. Verifies that
//! [`Ctx::checkpoint_pos`] + [`Ctx::restore_to`] form a true
//! round-trip: state captured at position P, then mutated by
//! decode, then restored, must reproduce the post-snapshot logits
//! exactly.
//!
//! `#[ignore]` because each test loads ~18 GB of artifacts. Run with:
//!
//! ```bash
//! cargo test -p moeflux --features model-qwen3-6-35b-a3b \
//!     --test checkpoint_restore --release \
//!     -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use moeflux::{Ctx, CheckpointError, DEFAULT_MAX_CHECKPOINTS};

fn model_root() -> PathBuf {
    let default =
        "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-root";
    PathBuf::from(std::env::var("MOEFLUX_SMOKE_ROOT").unwrap_or(default.into()))
}

fn artifacts() -> PathBuf {
    let default =
        "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-artifacts";
    PathBuf::from(
        std::env::var("MOEFLUX_SMOKE_ARTIFACTS").unwrap_or(default.into()),
    )
}

fn open_ctx() -> Ctx {
    let art = artifacts();
    let root = model_root();
    Ctx::open(
        &art.join("model_weights.bin"),
        &art.join("model_weights.json"),
        &art.join("vocab.bin"),
        &root,
        /* experts_per_tok */ 4,
        /* use_2bit */ false,
    )
    .expect("Ctx::open")
}

fn argmax(logits: &[f32]) -> i32 {
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

const PROMPT: &[i32] = &[1, 100, 500, 1000, 2000, 3000];

/// After a checkpoint at position P, mutating decode steps, and a
/// restore_to(P), the next forward pass at position P must produce
/// the same logits the model would have produced immediately after
/// the checkpoint with no intervening work.
#[test]
#[ignore]
fn checkpoint_then_restore_round_trips() {
    let mut ctx = open_ctx();
    let mut logits = vec![0.0f32; ctx.n_vocab()];

    // Establish state at pos = PROMPT.len().
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits).expect("eval_prompt");
    ctx.checkpoint_pos(PROMPT.len() as i32)
        .expect("checkpoint_pos");

    // Capture the immediately-post-checkpoint logits by feeding one
    // probe token. We need a deterministic measurement of "the state
    // right after checkpoint" — eval_token at pos = PROMPT.len()
    // gives us exactly that.
    let probe = argmax(&logits);
    let probe_pos = PROMPT.len();
    ctx.eval_token(probe, probe_pos, 0, &mut logits)
        .expect("eval_token probe");
    let logits_post_checkpoint = logits.clone();

    // Restore — this should rewind state to right after the
    // checkpoint_pos call (i.e., before the probe token).
    ctx.restore_to(PROMPT.len() as i32).expect("restore_to");

    // Re-feed the same probe at the same position. Must match.
    ctx.eval_token(probe, probe_pos, 0, &mut logits)
        .expect("eval_token probe (after restore)");
    assert_eq!(
        argmax(&logits_post_checkpoint),
        argmax(&logits),
        "argmax diverged across checkpoint/restore round-trip",
    );
    let max_abs = logits_post_checkpoint
        .iter()
        .zip(logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-4,
        "logits diverged across round-trip: max abs delta = {max_abs}",
    );
}

/// `restore_to(pos)` with no snapshot at `pos` returns
/// `NoCheckpoint`. drama_llama's `Session` interprets this as "fall
/// back to full clear".
#[test]
#[ignore]
fn restore_to_unknown_pos_errors() {
    let mut ctx = open_ctx();
    let mut logits = vec![0.0f32; ctx.n_vocab()];
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits).expect("eval_prompt");

    match ctx.restore_to(9999) {
        Err(CheckpointError::NoCheckpoint { pos }) => {
            assert_eq!(pos, 9999);
        }
        other => panic!("expected NoCheckpoint(9999), got {other:?}"),
    }
}

/// `memory_clear` drops every snapshot. After a clear, no position
/// is restorable.
#[test]
#[ignore]
fn memory_clear_drops_snapshots() {
    let mut ctx = open_ctx();
    let mut logits = vec![0.0f32; ctx.n_vocab()];
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits).expect("eval_prompt");
    ctx.checkpoint_pos(PROMPT.len() as i32)
        .expect("checkpoint_pos");
    assert_eq!(ctx.checkpoint_count(), 1);

    ctx.memory_clear();
    assert_eq!(ctx.checkpoint_count(), 0);

    match ctx.restore_to(PROMPT.len() as i32) {
        Err(CheckpointError::NoCheckpoint { .. }) => {}
        other => panic!("expected NoCheckpoint after memory_clear, got {other:?}"),
    }
}

/// LRU cap keeps the most recent N. With max=4 and 5 inserts at
/// distinct non-zero positions, the oldest non-protected key is
/// evicted. Position 0 (if present) and the just-inserted position
/// are always retained.
#[test]
#[ignore]
fn lru_eviction_keeps_last_n() {
    let mut ctx = open_ctx();
    let mut logits = vec![0.0f32; ctx.n_vocab()];
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits).expect("eval_prompt");

    assert_eq!(DEFAULT_MAX_CHECKPOINTS, 4, "test assumes default cap of 4");
    // Insert 5 distinct non-zero positions. After the 5th, count
    // should drop back to 4 — the oldest (50) gets evicted.
    for &pos in &[50, 100, 150, 200, 250] {
        ctx.checkpoint_pos(pos).expect("checkpoint_pos");
    }
    assert_eq!(ctx.checkpoint_count(), 4);

    // 50 should be gone; 250 (just inserted) and the rest should be present.
    assert!(matches!(ctx.restore_to(50), Err(CheckpointError::NoCheckpoint { .. })));
    // After failed restore, count is still 4 (failure is non-mutating).
    assert_eq!(ctx.checkpoint_count(), 4);

    // 250 (most recent) is restorable.
    ctx.restore_to(250).expect("restore_to most-recent");
}

/// LRU eviction protects pos=0 even when older than other entries.
/// Used by drama_llama's `Session` to keep the empty-prefix snapshot
/// available across long conversations.
#[test]
#[ignore]
fn lru_protects_pos_zero() {
    let mut ctx = open_ctx();
    let mut logits = vec![0.0f32; ctx.n_vocab()];
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits).expect("eval_prompt");

    // Order: 0 first (oldest), then 4 more. With cap=4 and 5 entries,
    // the oldest non-protected is 50, not 0.
    for &pos in &[0, 50, 100, 150, 200] {
        ctx.checkpoint_pos(pos).expect("checkpoint_pos");
    }
    assert_eq!(ctx.checkpoint_count(), 4);
    // pos=0 must survive even though it's the oldest.
    ctx.restore_to(0).expect("pos=0 must remain restorable");
}
