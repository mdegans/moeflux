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

use moeflux::{CheckpointError, Ctx, DEFAULT_MAX_CHECKPOINTS};

fn model_root() -> PathBuf {
    let default = "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-root";
    PathBuf::from(std::env::var("MOEFLUX_SMOKE_ROOT").unwrap_or(default.into()))
}

fn artifacts() -> PathBuf {
    let default = "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-artifacts";
    PathBuf::from(std::env::var("MOEFLUX_SMOKE_ARTIFACTS").unwrap_or(default.into()))
}

fn open_ctx() -> Ctx {
    let art = artifacts();
    let root = model_root();
    Ctx::open(
        &art.join("model_weights.bin"),
        &art.join("model_weights.json"),
        &art.join("vocab.bin"),
        &root,
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
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits)
        .expect("eval_prompt");
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
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits)
        .expect("eval_prompt");

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
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits)
        .expect("eval_prompt");
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
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits)
        .expect("eval_prompt");

    assert_eq!(DEFAULT_MAX_CHECKPOINTS, 4, "test assumes default cap of 4");
    // Insert 5 distinct non-zero positions. After the 5th, count
    // should drop back to 4 — the oldest (50) gets evicted.
    for &pos in &[50, 100, 150, 200, 250] {
        ctx.checkpoint_pos(pos).expect("checkpoint_pos");
    }
    assert_eq!(ctx.checkpoint_count(), 4);

    // 50 should be gone; 250 (just inserted) and the rest should be present.
    assert!(matches!(
        ctx.restore_to(50),
        Err(CheckpointError::NoCheckpoint { .. })
    ));
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
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits)
        .expect("eval_prompt");

    // Order: 0 first (oldest), then 4 more. With cap=4 and 5 entries,
    // the oldest non-protected is 50, not 0.
    for &pos in &[0, 50, 100, 150, 200] {
        ctx.checkpoint_pos(pos).expect("checkpoint_pos");
    }
    assert_eq!(ctx.checkpoint_count(), 4);
    // pos=0 must survive even though it's the oldest.
    ctx.restore_to(0).expect("pos=0 must remain restorable");
}

/// `forget_pos(P)` drops the snapshot at P without disturbing any
/// other snapshot. After forgetting, `restore_to(P)` reports
/// `NoCheckpoint`; siblings remain restorable.
#[test]
#[ignore]
fn forget_pos_drops_one_entry() {
    let mut ctx = open_ctx();
    let mut logits = vec![0.0f32; ctx.n_vocab()];
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits)
        .expect("eval_prompt");

    for &pos in &[50, 100, 150] {
        ctx.checkpoint_pos(pos).expect("checkpoint_pos");
    }
    assert_eq!(ctx.checkpoint_count(), 3);

    ctx.forget_pos(100);
    assert_eq!(ctx.checkpoint_count(), 2);
    assert!(matches!(
        ctx.restore_to(100),
        Err(CheckpointError::NoCheckpoint { .. })
    ));

    // Siblings still restorable.
    ctx.restore_to(150)
        .expect("150 must survive forget_pos(100)");
    // restore_to(150) drops keys > 150 (none here), so 50 still lives.
    ctx.restore_to(50).expect("50 must survive forget_pos(100)");
}

/// `forget_pos` is idempotent: calling it on a position with no
/// snapshot is a no-op (matches the `Decoder::forget_pos` contract
/// drama_llama's `Session` relies on for double-eviction tolerance).
#[test]
#[ignore]
fn forget_pos_idempotent_on_missing() {
    let mut ctx = open_ctx();
    let mut logits = vec![0.0f32; ctx.n_vocab()];
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits)
        .expect("eval_prompt");

    ctx.checkpoint_pos(100).expect("checkpoint_pos");
    assert_eq!(ctx.checkpoint_count(), 1);

    // No snapshot at 9999 — must not panic, must not touch the live entry.
    ctx.forget_pos(9999);
    assert_eq!(ctx.checkpoint_count(), 1);

    // Calling twice on the same valid pos: first removes, second is no-op.
    ctx.forget_pos(100);
    assert_eq!(ctx.checkpoint_count(), 0);
    ctx.forget_pos(100);
    assert_eq!(ctx.checkpoint_count(), 0);
}

/// Three-arm discriminator for the partial-hit divergence drama_llama
/// caught during v0.8.0 validation (2026-06-12): a Session that
/// restores to a breakpoint and chunk-prefills a new tail produced a
/// different greedy continuation than a fresh session prefilling the
/// same tokens in one shot.
///
/// - **Arm A** (ground truth): one-shot `eval_prompt(PREFIX ++ TAIL)`,
///   greedy-decode `N_OUT`.
/// - **Arm B** (chunking only): `eval_prompt(PREFIX)` then
///   `eval_prompt(TAIL)`, decode. A≠B → chunked-prefill numerics
///   alone shift the state.
/// - **Arm C** (full scenario): prefill PREFIX, checkpoint, greedy-
///   decode `N_GEN` (simulated prior-turn generation), restore,
///   prefill TAIL, decode. B≠C → restore-after-generation is lossy.
#[test]
#[ignore]
fn restore_after_generation_matches_fresh_prefill() {
    const PREFIX: &[i32] = &[
        1, 100, 500, 1000, 2000, 3000, 42, 77, 1234, 5678, 9012, 345, 678, 901, 2345, 6789, 111,
        222, 333, 444, 555, 666, 777, 888,
    ];
    const TAIL: &[i32] = &[
        999, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 123, 456, 789, 1010, 2020, 3030,
    ];
    const N_GEN: usize = 24; // simulated prior-turn generation length
    const N_OUT: usize = 16; // compared continuation length

    fn decode_n(ctx: &mut Ctx, first_logits: &[f32], start_pos: usize, n: usize) -> Vec<i32> {
        let mut logits = first_logits.to_vec();
        let mut out = Vec::with_capacity(n);
        let mut pos = start_pos;
        for _ in 0..n {
            let tok = argmax(&logits);
            out.push(tok);
            ctx.eval_token(tok, pos, 0, &mut logits)
                .expect("eval_token");
            pos += 1;
        }
        out
    }

    let full: Vec<i32> = PREFIX.iter().chain(TAIL.iter()).copied().collect();

    // Arm A — one-shot ground truth.
    let seq_a = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(&full, 0, 0, &mut logits)
            .expect("eval_prompt A");
        decode_n(&mut ctx, &logits, full.len(), N_OUT)
    };

    // Arm B — two-chunk prefill, no checkpoint machinery.
    let seq_b = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PREFIX, 0, 0, &mut logits)
            .expect("eval_prompt B prefix");
        ctx.eval_prompt(TAIL, PREFIX.len(), 0, &mut logits)
            .expect("eval_prompt B tail");
        decode_n(&mut ctx, &logits, full.len(), N_OUT)
    };

    // Arm C — checkpoint, decode N_GEN, restore, prefill tail.
    let seq_c = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PREFIX, 0, 0, &mut logits)
            .expect("eval_prompt C prefix");
        ctx.checkpoint_pos(PREFIX.len() as i32)
            .expect("checkpoint_pos");
        let _discarded = decode_n(&mut ctx, &logits, PREFIX.len(), N_GEN);
        ctx.restore_to(PREFIX.len() as i32).expect("restore_to");
        ctx.eval_prompt(TAIL, PREFIX.len(), 0, &mut logits)
            .expect("eval_prompt C tail");
        decode_n(&mut ctx, &logits, full.len(), N_OUT)
    };

    eprintln!("[three-arm] A (one-shot):       {seq_a:?}");
    eprintln!("[three-arm] B (chunked):        {seq_b:?}");
    eprintln!("[three-arm] C (restore+chunk):  {seq_c:?}");

    assert_eq!(
        seq_a, seq_b,
        "Arm B: chunked prefill diverged from one-shot ground truth",
    );
    assert_eq!(
        seq_b, seq_c,
        "Arm C: restore-after-generation diverged from plain chunked \
         prefill — snapshot/restore is lossy",
    );
}

/// Variant of `checkpoint_then_restore_round_trips` with a *long*
/// drift between checkpoint and restore (24 greedy decode steps vs
/// the original's single probe token). Isolates "how far the state
/// drifted before restore" from the tail-prefill variable in
/// `restore_after_generation_matches_fresh_prefill`.
#[test]
#[ignore]
fn restore_round_trips_after_long_decode() {
    let mut ctx = open_ctx();
    let mut logits = vec![0.0f32; ctx.n_vocab()];

    ctx.eval_prompt(PROMPT, 0, 0, &mut logits)
        .expect("eval_prompt");
    ctx.checkpoint_pos(PROMPT.len() as i32)
        .expect("checkpoint_pos");

    // Reference: the immediate post-checkpoint probe logits.
    let probe = argmax(&logits);
    let probe_pos = PROMPT.len();
    ctx.eval_token(probe, probe_pos, 0, &mut logits)
        .expect("eval_token probe (reference)");
    let logits_ref = logits.clone();

    // Drift: 24 more greedy decode steps on top of the probe.
    let mut pos = probe_pos + 1;
    for _ in 0..24 {
        let tok = argmax(&logits);
        ctx.eval_token(tok, pos, 0, &mut logits)
            .expect("eval_token drift");
        pos += 1;
    }

    // Restore and re-probe.
    ctx.restore_to(PROMPT.len() as i32).expect("restore_to");
    ctx.eval_token(probe, probe_pos, 0, &mut logits)
        .expect("eval_token probe (after restore)");

    assert_eq!(
        argmax(&logits_ref),
        argmax(&logits),
        "argmax diverged after restore from a 24-step drift",
    );
    let max_abs = logits_ref
        .iter()
        .zip(logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-4,
        "logits diverged after long-drift restore: max abs delta = {max_abs}",
    );
}

/// Discriminator V1: long drift that crosses the 32-position
/// boundary (prompt 6 + 41 decode steps → pos 47), restore, probe.
/// No batched prefill after restore. Fails → KV capacity growth
/// during drift breaks restore.
#[test]
#[ignore]
fn restore_round_trips_after_capacity_growth() {
    let mut ctx = open_ctx();
    let mut logits = vec![0.0f32; ctx.n_vocab()];

    ctx.eval_prompt(PROMPT, 0, 0, &mut logits)
        .expect("eval_prompt");
    ctx.checkpoint_pos(PROMPT.len() as i32)
        .expect("checkpoint_pos");

    let probe = argmax(&logits);
    let probe_pos = PROMPT.len();
    ctx.eval_token(probe, probe_pos, 0, &mut logits)
        .expect("eval_token probe (reference)");
    let logits_ref = logits.clone();

    let mut pos = probe_pos + 1;
    for _ in 0..40 {
        let tok = argmax(&logits);
        ctx.eval_token(tok, pos, 0, &mut logits)
            .expect("eval_token drift");
        pos += 1;
    }

    ctx.restore_to(PROMPT.len() as i32).expect("restore_to");
    ctx.eval_token(probe, probe_pos, 0, &mut logits)
        .expect("eval_token probe (after restore)");

    let max_abs = logits_ref
        .iter()
        .zip(logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-4,
        "logits diverged after capacity-crossing restore: \
         max abs delta = {max_abs}",
    );
}

/// Discriminator V2: batched prefill after restore, with everything
/// staying far below the 32-position boundary (6 + 8 drift + 8 tail
/// + 8 decode = max pos 22). Fails → batched-prefill-after-restore
/// is broken regardless of KV capacity.
#[test]
#[ignore]
fn restore_then_batched_tail_small() {
    const TAIL: &[i32] = &[999, 1111, 2222, 3333, 4444, 5555, 6666, 7777];
    const N_OUT: usize = 8;

    fn decode_n(ctx: &mut Ctx, first_logits: &[f32], start_pos: usize, n: usize) -> Vec<i32> {
        let mut logits = first_logits.to_vec();
        let mut out = Vec::with_capacity(n);
        let mut pos = start_pos;
        for _ in 0..n {
            let tok = argmax(&logits);
            out.push(tok);
            ctx.eval_token(tok, pos, 0, &mut logits)
                .expect("eval_token");
            pos += 1;
        }
        out
    }

    // Reference arm: chunked prefill, no checkpoint machinery.
    let seq_ref = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PROMPT, 0, 0, &mut logits).expect("prefix");
        ctx.eval_prompt(TAIL, PROMPT.len(), 0, &mut logits)
            .expect("tail");
        decode_n(&mut ctx, &logits, PROMPT.len() + TAIL.len(), N_OUT)
    };

    // Restore arm: prefill, checkpoint, short drift, restore, tail.
    let seq_restore = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PROMPT, 0, 0, &mut logits).expect("prefix");
        ctx.checkpoint_pos(PROMPT.len() as i32)
            .expect("checkpoint_pos");
        let _ = decode_n(&mut ctx, &logits, PROMPT.len(), 8);
        ctx.restore_to(PROMPT.len() as i32).expect("restore_to");
        ctx.eval_prompt(TAIL, PROMPT.len(), 0, &mut logits)
            .expect("tail");
        decode_n(&mut ctx, &logits, PROMPT.len() + TAIL.len(), N_OUT)
    };

    eprintln!("[v2] ref:     {seq_ref:?}");
    eprintln!("[v2] restore: {seq_restore:?}");
    assert_eq!(
        seq_ref, seq_restore,
        "batched tail prefill after restore diverged (capacity not a \
         factor at these lengths)",
    );
}

/// Parameterized grid over the arm-C shape: which knob flips the
/// restore+tail divergence? Each config runs a reference arm
/// (chunked prefill, no restore) and a restore arm, comparing greedy
/// continuations.
#[test]
#[ignore]
fn restore_divergence_grid() {
    const POOL: &[i32] = &[
        1, 100, 500, 1000, 2000, 3000, 42, 77, 1234, 5678, 9012, 345, 678, 901, 2345, 6789, 111,
        222, 333, 444, 555, 666, 777, 888, 999, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888,
        9999, 123, 456, 789, 1010, 2020, 3030,
    ];

    fn decode_n(ctx: &mut Ctx, first_logits: &[f32], start_pos: usize, n: usize) -> Vec<i32> {
        let mut logits = first_logits.to_vec();
        let mut out = Vec::with_capacity(n);
        let mut pos = start_pos;
        for _ in 0..n {
            let tok = argmax(&logits);
            out.push(tok);
            ctx.eval_token(tok, pos, 0, &mut logits)
                .expect("eval_token");
            pos += 1;
        }
        out
    }

    fn run_config(
        prefix_len: usize,
        drift: usize,
        tail_len: usize,
        n_out: usize,
    ) -> (Vec<i32>, Vec<i32>) {
        let prefix = &POOL[..prefix_len];
        let tail = &POOL[prefix_len..prefix_len + tail_len];

        let reference = {
            let mut ctx = open_ctx();
            let mut logits = vec![0.0f32; ctx.n_vocab()];
            ctx.eval_prompt(prefix, 0, 0, &mut logits)
                .expect("ref prefix");
            ctx.eval_prompt(tail, prefix_len, 0, &mut logits)
                .expect("ref tail");
            decode_n(&mut ctx, &logits, prefix_len + tail_len, n_out)
        };

        let restored = {
            let mut ctx = open_ctx();
            let mut logits = vec![0.0f32; ctx.n_vocab()];
            ctx.eval_prompt(prefix, 0, 0, &mut logits)
                .expect("rst prefix");
            ctx.checkpoint_pos(prefix_len as i32)
                .expect("checkpoint_pos");
            let _ = decode_n(&mut ctx, &logits, prefix_len, drift);
            ctx.restore_to(prefix_len as i32).expect("restore_to");
            ctx.eval_prompt(tail, prefix_len, 0, &mut logits)
                .expect("rst tail");
            decode_n(&mut ctx, &logits, prefix_len + tail_len, n_out)
        };

        (reference, restored)
    }

    let configs: &[(usize, usize, usize, usize, &str)] = &[
        (24, 24, 16, 16, "arm-C replica"),
        (24, 8, 16, 16, "short drift"),
        (6, 24, 16, 16, "small prefix"),
        (24, 24, 8, 8, "small tail+out"),
    ];

    let mut failures = Vec::new();
    for &(p, d, t, n, label) in configs {
        let (a, b) = run_config(p, d, t, n);
        let verdict = if a == b { "MATCH" } else { "DIVERGE" };
        eprintln!("[grid] {label} (p={p} d={d} t={t} n={n}): {verdict}");
        if a != b {
            eprintln!("[grid]   ref: {a:?}");
            eprintln!("[grid]   rst: {b:?}");
            failures.push(label);
        }
    }
    assert!(failures.is_empty(), "diverging configs: {failures:?}",);
}

/// Grid 2: drift=0 — checkpoint then *immediately* restore, then
/// tail. If this diverges, the save/load round-trip alone corrupts
/// state; the prefix-length sweep finds the threshold (tile size?).
#[test]
#[ignore]
fn restore_divergence_prefix_sweep_zero_drift() {
    const POOL: &[i32] = &[
        1, 100, 500, 1000, 2000, 3000, 42, 77, 1234, 5678, 9012, 345, 678, 901, 2345, 6789, 111,
        222, 333, 444, 555, 666, 777, 888, 999, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888,
        9999, 123, 456, 789, 1010, 2020, 3030, 4040, 5050, 6060, 7070, 8080, 9090, 1212, 3434,
    ];
    const TAIL_LEN: usize = 16;
    const N_OUT: usize = 8;

    fn decode_n(ctx: &mut Ctx, first_logits: &[f32], start_pos: usize, n: usize) -> Vec<i32> {
        let mut logits = first_logits.to_vec();
        let mut out = Vec::with_capacity(n);
        let mut pos = start_pos;
        for _ in 0..n {
            let tok = argmax(&logits);
            out.push(tok);
            ctx.eval_token(tok, pos, 0, &mut logits)
                .expect("eval_token");
            pos += 1;
        }
        out
    }

    let mut failures = Vec::new();
    for &p in &[6usize, 8, 12, 16, 17, 20, 24] {
        let prefix = &POOL[..p];
        let tail = &POOL[p..p + TAIL_LEN];

        let reference = {
            let mut ctx = open_ctx();
            let mut logits = vec![0.0f32; ctx.n_vocab()];
            ctx.eval_prompt(prefix, 0, 0, &mut logits)
                .expect("ref prefix");
            ctx.eval_prompt(tail, p, 0, &mut logits).expect("ref tail");
            decode_n(&mut ctx, &logits, p + TAIL_LEN, N_OUT)
        };
        let restored = {
            let mut ctx = open_ctx();
            let mut logits = vec![0.0f32; ctx.n_vocab()];
            ctx.eval_prompt(prefix, 0, 0, &mut logits)
                .expect("rst prefix");
            ctx.checkpoint_pos(p as i32).expect("checkpoint_pos");
            ctx.restore_to(p as i32).expect("restore_to");
            ctx.eval_prompt(tail, p, 0, &mut logits).expect("rst tail");
            decode_n(&mut ctx, &logits, p + TAIL_LEN, N_OUT)
        };
        let verdict = if reference == restored {
            "MATCH"
        } else {
            "DIVERGE"
        };
        eprintln!("[sweep] p={p}: {verdict}");
        if reference != restored {
            eprintln!("[sweep]   ref: {reference:?}");
            eprintln!("[sweep]   rst: {restored:?}");
            failures.push(p);
        }
    }
    assert!(
        failures.is_empty(),
        "diverging prefix lengths: {failures:?}"
    );
}

/// In-flight capture probe: after a 24-token prefill, `state_save`
/// twice — immediately, and again after 300 ms (plenty for any
/// in-flight GPU work to land). If the two captures differ, the
/// first read raced the prefill's GPU writes and the snapshot drain
/// contract is broken for the batched path.
#[test]
#[ignore]
fn state_save_is_stable_after_prefill() {
    const PREFIX: &[i32] = &[
        1, 100, 500, 1000, 2000, 3000, 42, 77, 1234, 5678, 9012, 345, 678, 901, 2345, 6789, 111,
        222, 333, 444, 555, 666, 777, 888,
    ];
    let mut ctx = open_ctx();
    let mut logits = vec![0.0f32; ctx.n_vocab()];
    ctx.eval_prompt(PREFIX, 0, 0, &mut logits)
        .expect("eval_prompt");

    let mut snap_a = vec![0u8; ctx.state_size()];
    ctx.state_save(&mut snap_a).expect("state_save A");

    std::thread::sleep(std::time::Duration::from_millis(300));

    let mut snap_b = vec![0u8; ctx.state_size()];
    ctx.state_save(&mut snap_b).expect("state_save B");

    let diff = snap_a
        .iter()
        .zip(snap_b.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff, 0,
        "state_save captured {diff} differing bytes across a 300 ms \
         pause — snapshot read raced in-flight GPU work",
    );
}

/// Save→load→save identity: after a 16-token prefill (one full
/// gated-delta chunk — the corruption threshold), snapshot A, load
/// it back, snapshot C. A≠C → `state_load` does not faithfully
/// install what `state_save` captured (or save can't read back what
/// load wrote). Reports the first differing byte offset to localize
/// the section.
#[test]
#[ignore]
fn state_load_save_identity_at_chunk_boundary() {
    const PREFIX: &[i32] = &[
        1, 100, 500, 1000, 2000, 3000, 42, 77, 1234, 5678, 9012, 345, 678, 901, 2345, 6789,
    ];
    let mut ctx = open_ctx();
    let mut logits = vec![0.0f32; ctx.n_vocab()];
    ctx.eval_prompt(PREFIX, 0, 0, &mut logits)
        .expect("eval_prompt");

    let mut snap_a = vec![0u8; ctx.state_size()];
    ctx.state_save(&mut snap_a).expect("state_save A");

    ctx.state_load(&snap_a).expect("state_load");

    let mut snap_c = vec![0u8; ctx.state_size()];
    ctx.state_save(&mut snap_c).expect("state_save C");

    if snap_a != snap_c {
        let first = snap_a
            .iter()
            .zip(snap_c.iter())
            .position(|(a, c)| a != c)
            .unwrap();
        let diff = snap_a
            .iter()
            .zip(snap_c.iter())
            .filter(|(a, c)| a != c)
            .count();
        panic!(
            "load→save not identity: {diff} bytes differ, first at \
             offset {first} (total {})",
            snap_a.len(),
        );
    }
}

/// Does `state_save` alone perturb live state? Prefill one full
/// gated-delta chunk, call `state_save`, THROW THE SNAPSHOT AWAY
/// (no load), then tail-prefill and decode. Divergence from the
/// no-save reference means the save path (or its deferred/prefetch
/// drains) corrupts the live context.
#[test]
#[ignore]
fn state_save_alone_does_not_perturb_live_state() {
    const PREFIX: &[i32] = &[
        1, 100, 500, 1000, 2000, 3000, 42, 77, 1234, 5678, 9012, 345, 678, 901, 2345, 6789,
    ];
    const TAIL: &[i32] = &[
        111, 222, 333, 444, 555, 666, 777, 888, 999, 1111, 2222, 3333, 4444, 5555, 6666, 7777,
    ];
    const N_OUT: usize = 8;

    fn decode_n(ctx: &mut Ctx, first_logits: &[f32], start_pos: usize, n: usize) -> Vec<i32> {
        let mut logits = first_logits.to_vec();
        let mut out = Vec::with_capacity(n);
        let mut pos = start_pos;
        for _ in 0..n {
            let tok = argmax(&logits);
            out.push(tok);
            ctx.eval_token(tok, pos, 0, &mut logits)
                .expect("eval_token");
            pos += 1;
        }
        out
    }

    let reference = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PREFIX, 0, 0, &mut logits)
            .expect("ref prefix");
        ctx.eval_prompt(TAIL, PREFIX.len(), 0, &mut logits)
            .expect("ref tail");
        decode_n(&mut ctx, &logits, PREFIX.len() + TAIL.len(), N_OUT)
    };

    let with_save = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PREFIX, 0, 0, &mut logits)
            .expect("sv prefix");
        let mut snap = vec![0u8; ctx.state_size()];
        ctx.state_save(&mut snap).expect("state_save");
        drop(snap); // never loaded
        ctx.eval_prompt(TAIL, PREFIX.len(), 0, &mut logits)
            .expect("sv tail");
        decode_n(&mut ctx, &logits, PREFIX.len() + TAIL.len(), N_OUT)
    };

    eprintln!("[save-only] ref:  {reference:?}");
    eprintln!("[save-only] save: {with_save:?}");
    assert_eq!(
        reference, with_save,
        "state_save alone perturbed the live context",
    );
}

/// Snapshot-free repro candidate: `clear_prefetch_predictions` runs
/// the same `prefetch.invalidate_all()` that `state_load` does — the
/// only non-no-op side effect a zero-drift restore performs. If
/// prefix → clear → tail diverges from prefix → tail, the restore
/// divergence has nothing to do with snapshots at all: invalidating
/// prefetch between batched prefills corrupts the next call's expert
/// staging.
#[test]
#[ignore]
fn prefetch_invalidate_between_prefills_is_sound() {
    const PREFIX: &[i32] = &[
        1, 100, 500, 1000, 2000, 3000, 42, 77, 1234, 5678, 9012, 345, 678, 901, 2345, 6789,
    ];
    const TAIL: &[i32] = &[
        111, 222, 333, 444, 555, 666, 777, 888, 999, 1111, 2222, 3333, 4444, 5555, 6666, 7777,
    ];
    const N_OUT: usize = 8;

    fn decode_n(ctx: &mut Ctx, first_logits: &[f32], start_pos: usize, n: usize) -> Vec<i32> {
        let mut logits = first_logits.to_vec();
        let mut out = Vec::with_capacity(n);
        let mut pos = start_pos;
        for _ in 0..n {
            let tok = argmax(&logits);
            out.push(tok);
            ctx.eval_token(tok, pos, 0, &mut logits)
                .expect("eval_token");
            pos += 1;
        }
        out
    }

    let reference = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PREFIX, 0, 0, &mut logits)
            .expect("ref prefix");
        ctx.eval_prompt(TAIL, PREFIX.len(), 0, &mut logits)
            .expect("ref tail");
        decode_n(&mut ctx, &logits, PREFIX.len() + TAIL.len(), N_OUT)
    };

    let with_invalidate = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PREFIX, 0, 0, &mut logits)
            .expect("inv prefix");
        ctx.clear_prefetch_predictions();
        ctx.eval_prompt(TAIL, PREFIX.len(), 0, &mut logits)
            .expect("inv tail");
        decode_n(&mut ctx, &logits, PREFIX.len() + TAIL.len(), N_OUT)
    };

    eprintln!("[invalidate] ref: {reference:?}");
    eprintln!("[invalidate] inv: {with_invalidate:?}");
    assert_eq!(
        reference, with_invalidate,
        "prefetch invalidation between batched prefills changed the \
         continuation — expert staging desync",
    );
}

/// Oracle-tail variant at the p=16 threshold: both arms feed TAIL
/// per-token through `eval_token` (oracle path); only the zero-drift
/// checkpoint+restore differs. Diverges → the poisoned state is on
/// the decode path too (i.e., restore corrupts shared state).
/// Matches → the corruption is specific to a *batched* prefill
/// running after a restore.
#[test]
#[ignore]
fn restore_with_oracle_tail_at_threshold() {
    const PREFIX: &[i32] = &[
        1, 100, 500, 1000, 2000, 3000, 42, 77, 1234, 5678, 9012, 345, 678, 901, 2345, 6789,
    ];
    const TAIL: &[i32] = &[
        111, 222, 333, 444, 555, 666, 777, 888, 999, 1111, 2222, 3333, 4444, 5555, 6666, 7777,
    ];
    const N_OUT: usize = 8;

    fn feed_tail_oracle(ctx: &mut Ctx, logits: &mut [f32]) {
        let mut pos = PREFIX.len();
        for &tok in TAIL {
            ctx.eval_token(tok, pos, 0, logits)
                .expect("eval_token tail");
            pos += 1;
        }
    }

    fn decode_n(ctx: &mut Ctx, first_logits: &[f32], start_pos: usize, n: usize) -> Vec<i32> {
        let mut logits = first_logits.to_vec();
        let mut out = Vec::with_capacity(n);
        let mut pos = start_pos;
        for _ in 0..n {
            let tok = argmax(&logits);
            out.push(tok);
            ctx.eval_token(tok, pos, 0, &mut logits)
                .expect("eval_token");
            pos += 1;
        }
        out
    }

    let reference = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PREFIX, 0, 0, &mut logits)
            .expect("ref prefix");
        feed_tail_oracle(&mut ctx, &mut logits);
        decode_n(&mut ctx, &logits, PREFIX.len() + TAIL.len(), N_OUT)
    };

    let restored = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PREFIX, 0, 0, &mut logits)
            .expect("rst prefix");
        ctx.checkpoint_pos(PREFIX.len() as i32)
            .expect("checkpoint_pos");
        ctx.restore_to(PREFIX.len() as i32).expect("restore_to");
        feed_tail_oracle(&mut ctx, &mut logits);
        decode_n(&mut ctx, &logits, PREFIX.len() + TAIL.len(), N_OUT)
    };

    eprintln!("[oracle-tail] ref: {reference:?}");
    eprintln!("[oracle-tail] rst: {restored:?}");
    assert_eq!(
        reference, restored,
        "oracle-fed tail diverged after restore"
    );
}

/// Composition probe: save + prefetch-invalidate (both of state_load's
/// drains/side-effects) but NO buffer writes, then batched tail.
/// Fails → the side-effect composition breaks the next batched call.
/// Passes → the (content-identical!) buffer writes in state_load are
/// the trigger, pointing at a CPU-write/GPU-plan coherency or
/// aliasing issue.
#[test]
#[ignore]
fn save_plus_invalidate_then_batched_tail() {
    const PREFIX: &[i32] = &[
        1, 100, 500, 1000, 2000, 3000, 42, 77, 1234, 5678, 9012, 345, 678, 901, 2345, 6789,
    ];
    const TAIL: &[i32] = &[
        111, 222, 333, 444, 555, 666, 777, 888, 999, 1111, 2222, 3333, 4444, 5555, 6666, 7777,
    ];
    const N_OUT: usize = 8;

    fn decode_n(ctx: &mut Ctx, first_logits: &[f32], start_pos: usize, n: usize) -> Vec<i32> {
        let mut logits = first_logits.to_vec();
        let mut out = Vec::with_capacity(n);
        let mut pos = start_pos;
        for _ in 0..n {
            let tok = argmax(&logits);
            out.push(tok);
            ctx.eval_token(tok, pos, 0, &mut logits)
                .expect("eval_token");
            pos += 1;
        }
        out
    }

    let reference = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PREFIX, 0, 0, &mut logits)
            .expect("ref prefix");
        ctx.eval_prompt(TAIL, PREFIX.len(), 0, &mut logits)
            .expect("ref tail");
        decode_n(&mut ctx, &logits, PREFIX.len() + TAIL.len(), N_OUT)
    };

    let composed = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PREFIX, 0, 0, &mut logits)
            .expect("cmp prefix");
        let mut snap = vec![0u8; ctx.state_size()];
        ctx.state_save(&mut snap).expect("state_save");
        drop(snap);
        ctx.clear_prefetch_predictions();
        ctx.eval_prompt(TAIL, PREFIX.len(), 0, &mut logits)
            .expect("cmp tail");
        decode_n(&mut ctx, &logits, PREFIX.len() + TAIL.len(), N_OUT)
    };

    eprintln!("[compose] ref: {reference:?}");
    eprintln!("[compose] cmp: {composed:?}");
    assert_eq!(
        reference, composed,
        "save+invalidate (no load writes) broke the batched tail",
    );
}

/// Forensic diff: run the reference arm and the zero-drift restore
/// arm through identical tokens (prefix, tail — same chunk
/// boundaries), then `state_save` both and byte-diff. Any difference
/// is state the load path corrupted; the offset decode names the
/// layer and section (full-attn KV vs linear conv vs linear ssm).
#[test]
#[ignore]
fn post_tail_state_diff_names_corrupted_section() {
    const PREFIX: &[i32] = &[
        1, 100, 500, 1000, 2000, 3000, 42, 77, 1234, 5678, 9012, 345, 678, 901, 2345, 6789,
    ];
    const TAIL: &[i32] = &[
        111, 222, 333, 444, 555, 666, 777, 888, 999, 1111, 2222, 3333, 4444, 5555, 6666, 7777,
    ];

    let snap_ref = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PREFIX, 0, 0, &mut logits)
            .expect("ref prefix");
        ctx.eval_prompt(TAIL, PREFIX.len(), 0, &mut logits)
            .expect("ref tail");
        let mut s = vec![0u8; ctx.state_size()];
        ctx.state_save(&mut s).expect("save ref");
        s
    };

    let snap_rst = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PREFIX, 0, 0, &mut logits)
            .expect("rst prefix");
        ctx.checkpoint_pos(PREFIX.len() as i32)
            .expect("checkpoint_pos");
        ctx.restore_to(PREFIX.len() as i32).expect("restore_to");
        ctx.eval_prompt(TAIL, PREFIX.len(), 0, &mut logits)
            .expect("rst tail");
        let mut s = vec![0u8; ctx.state_size()];
        ctx.state_save(&mut s).expect("save rst");
        s
    };

    assert_eq!(snap_ref.len(), snap_rst.len(), "snapshot size mismatch");
    if snap_ref == snap_rst {
        eprintln!(
            "[forensic] snapshots identical — corruption is OUTSIDE \
                   snapshot-covered state"
        );
        return;
    }

    // Walk differing byte runs and report offsets; section decode is
    // done by hand against the layout in state_snapshot.rs (header,
    // then per-layer: full-attn `4 + len*stride*2` / linear
    // `conv + ssm`).
    let mut runs = Vec::new();
    let mut run_start: Option<usize> = None;
    for (i, (a, b)) in snap_ref.iter().zip(snap_rst.iter()).enumerate() {
        match (a == b, run_start) {
            (false, None) => run_start = Some(i),
            (true, Some(s)) => {
                runs.push((s, i));
                run_start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = run_start {
        runs.push((s, snap_ref.len()));
    }
    eprintln!(
        "[forensic] {} differing runs over {} bytes total",
        runs.len(),
        snap_ref.len(),
    );
    for (s, e) in runs.iter().take(20) {
        eprintln!("[forensic]   bytes {s}..{e} ({} bytes)", e - s);
    }
    panic!("snapshots differ — see offsets above");
}

/// Is the canonical (snapshot-visible) KV actually synced after a
/// BATCHED prefill? Feed the same 16 tokens batched vs per-token
/// (oracle), snapshot both, and compare. Large differences mean the
/// batched path leaves canonical KV stale — and any state_load then
/// clobbers the live GPU mirrors with stale data.
#[test]
#[ignore]
fn batched_prefill_syncs_canonical_kv() {
    const PREFIX: &[i32] = &[
        1, 100, 500, 1000, 2000, 3000, 42, 77, 1234, 5678, 9012, 345, 678, 901, 2345, 6789,
    ];

    let snap_batched = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PREFIX, 0, 0, &mut logits).expect("batched");
        let mut s = vec![0u8; ctx.state_size()];
        ctx.state_save(&mut s).expect("save batched");
        s
    };

    let snap_oracle = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        // Feed token 0 via eval_prompt (size-1 batch primes buffers),
        // then the rest per-token through the oracle path.
        ctx.eval_prompt(&PREFIX[..1], 0, 0, &mut logits)
            .expect("first");
        for (i, &tok) in PREFIX.iter().enumerate().skip(1) {
            ctx.eval_token(tok, i, 0, &mut logits).expect("eval_token");
        }
        let mut s = vec![0u8; ctx.state_size()];
        ctx.state_save(&mut s).expect("save oracle");
        s
    };

    assert_eq!(snap_batched.len(), snap_oracle.len());
    // Value-level comparison: byte-diff can't distinguish benign FP
    // jitter (mantissa bytes of every float differ) from staleness.
    // Interpret the body as f32s and look at magnitudes.
    let body_a = &snap_batched[40..];
    let body_b = &snap_oracle[40..];
    let n = body_a.len() / 4;
    let mut max_abs = 0.0f32;
    let mut big = 0usize; // |delta| > 0.25 — way past reduction jitter
    for i in 0..n {
        let a = f32::from_le_bytes(body_a[i * 4..i * 4 + 4].try_into().unwrap());
        let b = f32::from_le_bytes(body_b[i * 4..i * 4 + 4].try_into().unwrap());
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        if d > 0.25 {
            big += 1;
        }
    }
    eprintln!(
        "[canon] f32 body: n={n}, max |delta|={max_abs}, big(>0.25)={big} \
         ({:.2}%)",
        big as f64 / n as f64 * 100.0,
    );
    assert!(
        (big as f64) < n as f64 * 0.001,
        "batched-prefill snapshot diverges from oracle snapshot at \
         value level — canonical state is stale/garbled after batched \
         prefill (max |delta| = {max_abs})",
    );
}

/// Magnitude check: how big are the logit deltas the zero-drift
/// restore introduces? Tiny (≲1e-2) → the divergence is numeric
/// jitter flipping near-tie argmaxes (restore is sound, exact-match
/// expectations are too strict). Large → genuine corruption.
#[test]
#[ignore]
fn restore_logit_delta_magnitude() {
    const PREFIX: &[i32] = &[
        1, 100, 500, 1000, 2000, 3000, 42, 77, 1234, 5678, 9012, 345, 678, 901, 2345, 6789,
    ];
    const TAIL: &[i32] = &[
        111, 222, 333, 444, 555, 666, 777, 888, 999, 1111, 2222, 3333, 4444, 5555, 6666, 7777,
    ];
    const N_STEPS: usize = 8;

    fn run(restore: bool) -> Vec<Vec<f32>> {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(PREFIX, 0, 0, &mut logits).expect("prefix");
        if restore {
            ctx.checkpoint_pos(PREFIX.len() as i32).expect("checkpoint");
            ctx.restore_to(PREFIX.len() as i32).expect("restore");
        }
        ctx.eval_prompt(TAIL, PREFIX.len(), 0, &mut logits)
            .expect("tail");
        let mut all = vec![logits.clone()];
        let mut pos = PREFIX.len() + TAIL.len();
        for _ in 0..N_STEPS {
            // Follow the REFERENCE arm's greedy choice in both runs so
            // positions stay comparable after a flip.
            let tok = argmax(&all[all.len() - 1]);
            ctx.eval_token(tok, pos, 0, &mut logits).expect("step");
            all.push(logits.clone());
            pos += 1;
        }
        all
    }

    let ref_logits = run(false);
    let rst_logits = run(true);

    for (step, (a, b)) in ref_logits.iter().zip(rst_logits.iter()).enumerate() {
        let max_abs = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let am_a = argmax(a);
        let am_b = argmax(b);
        eprintln!(
            "[delta] step {step}: max|dlogit|={max_abs:.4} argmax \
             ref={am_a} rst={am_b} {}",
            if am_a == am_b { "" } else { "<-- FLIP" },
        );
    }
}

/// THE BIG ONE: does oracle decode (GPU-SDPA-over-mirror at
/// kv_len ≥ 32) even work after a *batched* prefill? The batched
/// graph path appends KV to the canonical pool buffers only —
/// nothing populates `gpu_kv_k/v` mirrors. Arm 1: batched prefill
/// 40 tokens, oracle-decode 8. Arm 2: oracle-feed the same 40
/// tokens per-token (mirror stays synced), decode 8. Wild
/// divergence → every oracle decode after a batched prefill ≥ 32
/// tokens attends a partially-zero mirror.
#[test]
#[ignore]
fn oracle_decode_after_batched_prefill_matches_oracle_prefill() {
    const TOKENS: &[i32] = &[
        1, 100, 500, 1000, 2000, 3000, 42, 77, 1234, 5678, 9012, 345, 678, 901, 2345, 6789, 111,
        222, 333, 444, 555, 666, 777, 888, 999, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888,
        9999, 123, 456, 789, 1010, 2020, 3030,
    ];
    const N_OUT: usize = 8;

    fn decode_n(ctx: &mut Ctx, first_logits: &[f32], start_pos: usize, n: usize) -> Vec<i32> {
        let mut logits = first_logits.to_vec();
        let mut out = Vec::with_capacity(n);
        let mut pos = start_pos;
        for _ in 0..n {
            let tok = argmax(&logits);
            out.push(tok);
            ctx.eval_token(tok, pos, 0, &mut logits)
                .expect("eval_token");
            pos += 1;
        }
        out
    }

    let batched = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(TOKENS, 0, 0, &mut logits).expect("batched");
        decode_n(&mut ctx, &logits, TOKENS.len(), N_OUT)
    };

    let oracle = {
        let mut ctx = open_ctx();
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.eval_prompt(&TOKENS[..1], 0, 0, &mut logits)
            .expect("first");
        for (i, &tok) in TOKENS.iter().enumerate().skip(1) {
            ctx.eval_token(tok, i, 0, &mut logits).expect("feed");
        }
        decode_n(&mut ctx, &logits, TOKENS.len(), N_OUT)
    };

    eprintln!("[mirror] batched-prefill arm: {batched:?}");
    eprintln!("[mirror] oracle-prefill arm:  {oracle:?}");
    assert_eq!(
        batched, oracle,
        "oracle decode diverges after batched prefill — GPU KV \
         mirrors are not populated by the batched path",
    );
}
