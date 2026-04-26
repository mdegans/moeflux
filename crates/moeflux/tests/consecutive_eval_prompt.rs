//! Bisect tests for `Ctx` state-pollution.
//!
//! Originally written to chase a blallama symptom (degenerate output
//! on the second/third consecutive completion through
//! `Session<MoefluxBackend>`). The bisect found two real upstream
//! issues *and* exonerated `mf_memory_clear` itself.
//!
//! ## Findings
//!
//! - ✅ `memory_clear_resets_for_same_prompt` — passes. Within a
//!   single Ctx, `mf_memory_clear` resets state correctly across
//!   identical prompts.
//! - ✅ `memory_clear_after_dirty_decode_resets_for_different_prompt`
//!   — passes. Even after a dirty decode on a different prompt,
//!   `mf_memory_clear` fully resets. The original `g_deferred`
//!   hypothesis (file-scope global persisting across `memory_clear`)
//!   is empirically refuted for the within-Ctx case.
//! - ❌ `fresh_ctx_open_after_drop_works` — fails. Two independent
//!   Ctx instances should produce identical inference for the same
//!   prompt; instead, the second `Ctx::open` after the first is
//!   dropped produces all-NaN logits. Process-global state
//!   (`g_deferred` and/or its dangling pointer to `ctx->hidden`)
//!   survives `mf_free_model`.
//! - ❌ `resuming_prefill_after_seq_rm_matches_full_prefill` — fails.
//!   `mf_memory_seq_rm(0, l_hit, -1)` + `eval_prompt(suffix,
//!   start_pos=l_hit, …)` produces materially different output
//!   (verbatim 8-token loop on synthetic input) than a fresh full
//!   prefill of the same final token sequence. Per `moeflux.h`
//!   design notes, partial truncation of linear-attention layers
//!   resets recurrence state by design — this test confirms the
//!   implementation matches the lossy spec, *and* that the
//!   re-prefilled state diverges from a fresh prefill.
//!
//! ## Status
//!
//! These tests are left in their failing state intentionally. The
//! fix path is the upstream RIIR (see drama_llama's
//! `.claude/memory/riir_moeflux_strategy.md`). When the Rust port's
//! Phase 7 lands `memory_seq_rm -> Result<(), CannotTruncateLinear>`
//! and removes process-globals, both tests will pass. No need to
//! toggle `#[should_panic]`.
//!
//! `#[ignore]` because each test loads ~18 GB of artifacts. Run with:
//!
//! ```bash
//! cargo test -p moeflux --features model-qwen3-6-35b-a3b \
//!     --test consecutive_eval_prompt --release \
//!     -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use moeflux::Ctx;

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

fn topk(logits: &[f32], k: usize) -> Vec<i32> {
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

fn jaccard(a: &[i32], b: &[i32]) -> f32 {
    use std::collections::HashSet;
    let sa: HashSet<i32> = a.iter().copied().collect();
    let sb: HashSet<i32> = b.iter().copied().collect();
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    if union == 0.0 { 1.0 } else { inter / union }
}

fn logits_summary(logits: &[f32]) -> String {
    let finite = logits.iter().filter(|v| v.is_finite()).count();
    let nonzero = logits.iter().filter(|v| **v != 0.0).count();
    let any_nan = logits.iter().any(|v| v.is_nan());
    format!(
        "len={} finite={} nonzero={} any_nan={} argmax={}",
        logits.len(),
        finite,
        nonzero,
        any_nan,
        argmax(logits)
    )
}

/// Greedy continuation of length `n` starting from `tokens` at
/// position 0. Returns the n-token greedy trajectory plus the prefill
/// logits (logits returned by `eval_prompt` itself, before any decode
/// steps).
fn greedy(ctx: &mut Ctx, tokens: &[i32], n: usize) -> (Vec<i32>, Vec<f32>) {
    let mut logits = vec![0.0f32; ctx.n_vocab()];
    ctx.eval_prompt(tokens, 0, 0, &mut logits).expect("eval_prompt");
    let prefill_logits = logits.clone();

    let mut out = Vec::with_capacity(n);
    let mut next = argmax(&logits);
    for step in 0..n {
        out.push(next);
        let pos = tokens.len() + step;
        ctx.eval_token(next, pos, 0, &mut logits).expect("eval_token");
        next = argmax(&logits);
    }
    (out, prefill_logits)
}

/// Like `greedy`, but `eval_prompt` is called with `start_pos > 0`,
/// asserting that positions `[0, start_pos)` already contain matching
/// KV. Mirrors the resuming-prefill that `drama_llama::Session` does
/// after `memory_seq_rm(0, l_hit, -1)`.
fn greedy_resuming(
    ctx: &mut Ctx,
    suffix: &[i32],
    start_pos: usize,
    n: usize,
) -> (Vec<i32>, Vec<f32>) {
    let mut logits = vec![0.0f32; ctx.n_vocab()];
    ctx.eval_prompt(suffix, start_pos, 0, &mut logits)
        .expect("eval_prompt resuming");
    let prefill_logits = logits.clone();

    let mut out = Vec::with_capacity(n);
    let mut next = argmax(&logits);
    for step in 0..n {
        out.push(next);
        let pos = start_pos + suffix.len() + step;
        ctx.eval_token(next, pos, 0, &mut logits).expect("eval_token");
        next = argmax(&logits);
    }
    (out, prefill_logits)
}

/// Two distinct, equal-length token sequences. First token shared
/// (BOS-like) so divergence starts at index 1 — mirrors how real
/// chat prompts share a template prefix and diverge in the user
/// message.
const PROMPT_1: &[i32] = &[1, 100, 500, 1000];
const PROMPT_2: &[i32] = &[1, 200, 600, 1100];

/// Prompts with a shared prefix and distinct suffixes — the shape
/// drama_llama's `Session::kv_setup_for_call` exercises via
/// `memory_seq_rm(0, l_hit, -1)` + `eval_prompt(suffix, start_pos=l_hit)`.
const SHARED_PREFIX: &[i32] = &[1, 100, 500];
const P1_SUFFIX: &[i32] = &[1000, 2000, 3000];
const P2_SUFFIX: &[i32] = &[4000, 5000, 6000];

const N_DECODE: usize = 16;
const TOP_K: usize = 20;

#[test]
#[ignore]
fn memory_clear_resets_for_same_prompt() {
    let mut ctx = open_ctx();
    eprintln!(
        "[same-prompt] model={} n_vocab={}",
        ctx.model_name(),
        ctx.n_vocab(),
    );

    let (traj_a, logits_a) = greedy(&mut ctx, PROMPT_2, N_DECODE);
    eprintln!("[same-prompt] run A traj: {traj_a:?}");
    eprintln!("[same-prompt] run A logits: {}", logits_summary(&logits_a));

    ctx.memory_clear();
    assert_eq!(ctx.memory_seq_pos_max(0), 0);

    let (traj_b, logits_b) = greedy(&mut ctx, PROMPT_2, N_DECODE);
    eprintln!("[same-prompt] run B traj: {traj_b:?}");
    eprintln!("[same-prompt] run B logits: {}", logits_summary(&logits_b));

    assert_eq!(
        argmax(&logits_a),
        argmax(&logits_b),
        "memory_clear failed to reset state: same prompt produced \
         different prefill argmax across two runs on one Ctx"
    );
    assert_eq!(
        traj_a, traj_b,
        "memory_clear failed to reset state: same prompt diverged \
         during greedy decode across two runs on one Ctx"
    );
}

#[test]
#[ignore]
fn memory_clear_after_dirty_decode_resets_for_different_prompt() {
    let mut ctx = open_ctx();
    eprintln!(
        "[dirty-decode] model={} n_vocab={}",
        ctx.model_name(),
        ctx.n_vocab(),
    );

    // Baseline: clean Ctx, prompt 2 trajectory.
    let (baseline_traj, baseline_logits) =
        greedy(&mut ctx, PROMPT_2, N_DECODE);
    eprintln!("[dirty-decode] baseline traj: {baseline_traj:?}");
    eprintln!(
        "[dirty-decode] baseline logits: {}",
        logits_summary(&baseline_logits)
    );

    ctx.memory_clear();

    // Dirty: process prompt 1 with 8 decode steps to exercise
    // `g_deferred` and `ctx->hidden`, then memory_clear, then prompt 2.
    let (p1_traj, _) = greedy(&mut ctx, PROMPT_1, 8);
    eprintln!("[dirty-decode] prompt 1 traj: {p1_traj:?}");

    ctx.memory_clear();
    assert_eq!(ctx.memory_seq_pos_max(0), 0);

    let (dirty_traj, dirty_logits) = greedy(&mut ctx, PROMPT_2, N_DECODE);
    eprintln!("[dirty-decode] dirty traj: {dirty_traj:?}");
    eprintln!(
        "[dirty-decode] dirty logits: {}",
        logits_summary(&dirty_logits)
    );

    let baseline_top = topk(&baseline_logits, TOP_K);
    let dirty_top = topk(&dirty_logits, TOP_K);
    let overlap = jaccard(&baseline_top, &dirty_top);
    let traj_matches = baseline_traj
        .iter()
        .zip(dirty_traj.iter())
        .filter(|(a, b)| a == b)
        .count();
    eprintln!(
        "[dirty-decode] top-{TOP_K} jaccard={overlap:.3} traj_match={traj_matches}/{N_DECODE}"
    );

    assert_eq!(
        argmax(&baseline_logits),
        argmax(&dirty_logits),
        "Ctx state pollutes across consecutive eval_prompt calls: \
         prompt 2 prefill argmax differs after a prior prompt 1 + \
         decode + memory_clear (memory_clear is incomplete)"
    );
    assert!(
        overlap >= 0.95,
        "top-{TOP_K} overlap {overlap:.3} below 0.95 — Ctx state \
         pollutes across calls"
    );
    assert_eq!(
        baseline_traj, dirty_traj,
        "greedy trajectory diverges after a prior prompt 1 + decode \
         + memory_clear — Ctx state pollutes across calls"
    );
}

#[test]
#[ignore]
fn fresh_ctx_open_after_drop_works() {
    // Two independent Ctx instances must produce identical inference
    // for the same prompt. If they don't, moeflux has process-global
    // state that survives `mf_free_model`.
    let (traj_a, logits_a) = {
        let mut ctx = open_ctx();
        eprintln!(
            "[fresh-ctx] ctx_a model={} n_vocab={}",
            ctx.model_name(),
            ctx.n_vocab(),
        );
        greedy(&mut ctx, PROMPT_2, N_DECODE)
    };
    eprintln!("[fresh-ctx] ctx_a traj: {traj_a:?}");
    eprintln!("[fresh-ctx] ctx_a logits: {}", logits_summary(&logits_a));

    let (traj_b, logits_b) = {
        let mut ctx = open_ctx();
        eprintln!(
            "[fresh-ctx] ctx_b model={} n_vocab={}",
            ctx.model_name(),
            ctx.n_vocab(),
        );
        greedy(&mut ctx, PROMPT_2, N_DECODE)
    };
    eprintln!("[fresh-ctx] ctx_b traj: {traj_b:?}");
    eprintln!("[fresh-ctx] ctx_b logits: {}", logits_summary(&logits_b));

    assert_eq!(
        argmax(&logits_a),
        argmax(&logits_b),
        "two fresh Ctx instances disagree on prefill argmax — \
         moeflux has process-global state surviving mf_free_model \
         (suspect: g_deferred at infer.m:4003)"
    );
    assert_eq!(
        traj_a, traj_b,
        "two fresh Ctx instances produce different greedy trajectories \
         — process-global state survives mf_free_model"
    );
}

#[test]
#[ignore]
fn resuming_prefill_after_seq_rm_matches_full_prefill() {
    // Mirrors the exact path drama_llama::Session takes: prompt 1 is
    // prefilled and decoded, the suffix-and-generation are dropped
    // via memory_seq_rm(0, l_hit, -1), then prompt 2's suffix is
    // prefilled with start_pos=l_hit. Compare against a baseline
    // where prompt 2 is prefilled fresh from scratch.
    //
    // If these diverge, moeflux's resuming prefill (or memory_seq_rm
    // truncation, particularly of linear-attention state) is
    // incorrect — directly explaining the blallama symptom catalog.
    let mut ctx = open_ctx();
    eprintln!(
        "[resume] model={} n_vocab={}",
        ctx.model_name(),
        ctx.n_vocab(),
    );

    // Baseline: full prompt 2 from a fresh memory_clear.
    let p2_full: Vec<i32> = SHARED_PREFIX
        .iter()
        .chain(P2_SUFFIX.iter())
        .copied()
        .collect();
    let (baseline_traj, baseline_logits) =
        greedy(&mut ctx, &p2_full, N_DECODE);
    eprintln!("[resume] baseline traj: {baseline_traj:?}");
    eprintln!(
        "[resume] baseline logits: {}",
        logits_summary(&baseline_logits)
    );

    ctx.memory_clear();
    assert_eq!(ctx.memory_seq_pos_max(0), 0);

    // Dirty: full prompt 1 → decode → seq_rm to shared prefix → prompt 2 suffix.
    let p1_full: Vec<i32> = SHARED_PREFIX
        .iter()
        .chain(P1_SUFFIX.iter())
        .copied()
        .collect();
    let (p1_traj, _) = greedy(&mut ctx, &p1_full, 8);
    eprintln!("[resume] prompt 1 traj: {p1_traj:?}");

    let l_hit = SHARED_PREFIX.len() as i32;
    assert!(
        ctx.memory_seq_rm(0, l_hit, -1),
        "memory_seq_rm(0, {l_hit}, -1) returned false"
    );
    assert_eq!(
        ctx.memory_seq_pos_max(0),
        l_hit,
        "after seq_rm, pos_max should equal l_hit"
    );

    let (resume_traj, resume_logits) =
        greedy_resuming(&mut ctx, P2_SUFFIX, l_hit as usize, N_DECODE);
    eprintln!("[resume] resume traj:   {resume_traj:?}");
    eprintln!(
        "[resume] resume logits: {}",
        logits_summary(&resume_logits)
    );

    let baseline_top = topk(&baseline_logits, TOP_K);
    let resume_top = topk(&resume_logits, TOP_K);
    let overlap = jaccard(&baseline_top, &resume_top);
    let traj_matches = baseline_traj
        .iter()
        .zip(resume_traj.iter())
        .filter(|(a, b)| a == b)
        .count();
    eprintln!(
        "[resume] top-{TOP_K} jaccard={overlap:.3} traj_match={traj_matches}/{N_DECODE}"
    );

    assert_eq!(
        argmax(&baseline_logits),
        argmax(&resume_logits),
        "resuming prefill (after memory_seq_rm) produces different \
         prefill argmax than fresh full prefill of the same final \
         token sequence — moeflux's resuming-prefill or \
         memory_seq_rm path is incorrect"
    );
    assert!(
        overlap >= 0.95,
        "resuming-prefill top-{TOP_K} overlap {overlap:.3} below 0.95"
    );
    assert_eq!(
        baseline_traj, resume_traj,
        "resuming-prefill greedy trajectory diverges from fresh \
         full prefill — moeflux's resuming-prefill or memory_seq_rm \
         path is incorrect"
    );
}
