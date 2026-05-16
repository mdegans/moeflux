//! Phase 1 of the kernel arc — in-situ per-phase prefill breakdown.
//!
//! Drives one a3b prefill and dumps the Metal backend's per-label
//! cmdbuf timing (`Ctx::cmdbuf_stats`), answering: which phase —
//! attention, MoE, linear-attn, norms — owns prefill GPU time. The
//! labels come from the `commit_and_wait_labeled` callsites in the
//! batched full-attn path and the linear-attn graph path. This is
//! instrumentation only; it changes no prefill behavior.
//!
//! `#[ignore]` — loads ~18 GB of artifacts and runs a multi-second
//! prefill. Run **one shape per `cargo test` invocation** so each
//! prefill gets a pristine process (no cross-run `Ctx` state):
//!
//! ```bash
//! cargo test -p moeflux --features model-qwen3-6-35b-a3b \
//!     --test prefill_profile --release \
//!     -- --ignored --nocapture --test-threads=1 profile_1536
//!
//! cargo test -p moeflux --features model-qwen3-6-35b-a3b \
//!     --test prefill_profile --release \
//!     -- --ignored --nocapture --test-threads=1 profile_8192
//! ```

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use moeflux::riir::CmdbufStat;
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

/// Open the a3b Ctx. `experts_per_tok = 8` is the model's real
/// routing width (`variants.rs` `num_experts_per_tok`) — the MoE
/// phase should be profiled at production cost.
fn open_ctx() -> Ctx {
    let art = artifacts();
    let root = model_root();
    Ctx::open(
        &art.join("model_weights.bin"),
        &art.join("model_weights.json"),
        &art.join("vocab.bin"),
        &root,
        /* experts_per_tok */ 8,
        /* use_2bit */ false,
    )
    .expect("Ctx::open")
}

/// Deterministic in-vocab synthetic token stream. Exact tokens don't
/// matter for a per-phase *proportion* breakdown; kept off the low
/// special-token range.
fn synth_tokens(n: usize, vocab: usize) -> Vec<i32> {
    let span = vocab.saturating_sub(20).max(1);
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (10 + (s as usize % span)) as i32
        })
        .collect()
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Print the per-label breakdown sorted by total wall time desc.
fn dump(m: usize, wall: Duration, stats: &[(&str, CmdbufStat)], logits: &[f32]) {
    let wall_ms = ms(wall);
    let labeled_ns: u64 = stats.iter().map(|(_, s)| s.cpu_wait_ns).sum();
    let labeled_ms = labeled_ns as f64 / 1e6;

    let mut rows: Vec<&(&str, CmdbufStat)> = stats.iter().collect();
    rows.sort_by(|a, b| b.1.cpu_wait_ns.cmp(&a.1.cpu_wait_ns));

    let finite = logits.iter().filter(|v| v.is_finite()).count();
    let any_nan = logits.iter().any(|v| v.is_nan());

    eprintln!();
    eprintln!("=== prefill profile: M={m} tokens ===");
    eprintln!(
        "  wall {wall_ms:.1} ms   labeled {labeled_ms:.1} ms   \
         unaccounted {:.1} ms",
        wall_ms - labeled_ms
    );
    eprintln!(
        "  logits: len={} finite={finite} any_nan={any_nan}",
        logits.len()
    );
    eprintln!(
        "  {:<34} {:>6} {:>11} {:>11} {:>8}",
        "label", "count", "total ms", "ms/commit", "% wall"
    );
    for (label, s) in rows {
        let total_ms = s.cpu_wait_ns as f64 / 1e6;
        let per = if s.count > 0 {
            total_ms / s.count as f64
        } else {
            0.0
        };
        eprintln!(
            "  {label:<34} {:>6} {total_ms:>11.2} {per:>11.3} {:>7.1}%",
            s.count,
            100.0 * total_ms / wall_ms,
        );
    }
    eprintln!("=== end M={m} ===");
}

fn run_profile(m: usize) {
    let mut ctx = open_ctx();
    let vocab = ctx.n_vocab();
    eprintln!(
        "[prefill-profile] model={} n_vocab={vocab} M={m}",
        ctx.model_name(),
    );

    let tokens = synth_tokens(m, vocab);
    let mut logits = vec![0.0f32; vocab];

    ctx.reset_cmdbuf_stats(); // no-op on a fresh Ctx; explicit anyway
    let t0 = Instant::now();
    ctx.eval_prompt(&tokens, 0, 0, &mut logits)
        .expect("eval_prompt");
    let wall = t0.elapsed();

    let stats = ctx.cmdbuf_stats();
    dump(m, wall, &stats, &logits);

    assert!(
        !stats.is_empty(),
        "no labeled cmdbuf stats recorded — the prefill did not run \
         through any commit_and_wait_labeled path"
    );
    assert!(
        logits.iter().any(|v| v.is_finite() && *v != 0.0),
        "prefill produced no finite non-zero logits"
    );
}

/// ~1536-token prefill — sub-chunk shape (`BATCHED_CHUNK_SIZE` is
/// 8192), useful for the proportion breakdown.
#[test]
#[ignore = "loads ~18 GB of a3b artifacts; profiling driver"]
fn profile_1536() {
    run_profile(1536);
}

/// ~8192-token prefill — one full `BATCHED_CHUNK_SIZE` chunk, the
/// real production prefill shape.
#[test]
#[ignore = "loads ~18 GB of a3b artifacts; profiling driver"]
fn profile_8192() {
    run_profile(8192);
}
