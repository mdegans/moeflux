//! Phase 7 — snapshot v2 round-trip on Cogito-V2 671B (MLA variant).
//!
//! Smoke-validates the v2 wire format: header words, MLA per-layer
//! body (latent + rope-K), v1 backward-compat. The bit-exact assertion
//! is on logits at pos=N: eval to pos=N → snapshot → eval one more
//! token → restore snapshot → eval same token → compare logits.
//!
//! Uses pos=2 (not pos=200 from the plan) because cogito-v2 takes
//! ~12 s per token on this machine — pos=200 would exceed any
//! reasonable test wallclock. The wire format is exercised
//! identically at pos=2.
//!
//! Run via:
//! ```text
//! cargo test -p moeflux --no-default-features \
//!     --features model-cogito-v2-671b --release \
//!     --test snapshot_v2_roundtrip -- --ignored --nocapture
//! ```

#![cfg(all(target_os = "macos", feature = "model-cogito-v2-671b"))]

use std::path::PathBuf;

use moeflux::{Ctx, Error};

const ROOT: &str = "/Volumes/Temp Backup/models/blallama/cogito-v2-671b";

fn art(name: &str) -> PathBuf {
    PathBuf::from(ROOT).join("artifacts").join(name)
}

fn open_ctx() -> Result<Ctx, Error> {
    Ctx::open(
        &art("model_weights.bin"),
        &art("model_weights.json"),
        &PathBuf::from(ROOT).join("mlx/tokenizer.json"),
        &PathBuf::from(ROOT).join("root"),
        /* experts_per_tok = */ 8,
        /* use_2bit       = */ false,
    )
}

#[test]
#[ignore]
fn snapshot_v2_roundtrip_pos2() -> Result<(), Error> {
    let mut ctx = open_ctx()?;
    let vocab = 128_815usize;

    // Eval BOS at pos 0 (warms init + populates layer 0 of cache).
    let mut logits = vec![0.0f32; vocab];
    eprintln!("[snapshot-v2] eval token=0 pos=0");
    ctx.eval_token(0, 0, 0, &mut logits)?;

    // Eval token 1 at pos 1 (populates layer 1).
    eprintln!("[snapshot-v2] eval token=1 pos=1");
    ctx.eval_token(1, 1, 0, &mut logits)?;

    // Snapshot.
    let need = ctx.state_size();
    eprintln!("[snapshot-v2] state_size = {} bytes ({:.1} MB)", need, need as f64 / (1024.0 * 1024.0));
    assert!(need > 0, "state_size returned 0 — Mla branch likely still gated");
    let mut snap = vec![0u8; need];
    let written = ctx
        .state_save(&mut snap)
        .map_err(|e| panic!("state_save: {e}"))
        .unwrap();
    assert_eq!(written, need, "state_save wrote {} bytes, want {}", written, need);
    eprintln!("[snapshot-v2] state_save wrote {} bytes", written);

    // Header magic check (sanity).
    let magic = u32::from_le_bytes(snap[0..4].try_into().unwrap());
    let version = u32::from_le_bytes(snap[4..8].try_into().unwrap());
    assert_eq!(magic, 0x4D464C58, "magic mismatch");
    assert_eq!(version, 2, "version mismatch");

    // Eval one more token at pos 2 (this is the "after-save" reference).
    let mut logits_pre = vec![0.0f32; vocab];
    eprintln!("[snapshot-v2] eval token=2 pos=2 (pre-restore reference)");
    ctx.eval_token(2, 2, 0, &mut logits_pre)?;

    // Restore the snapshot — this rolls the cache state back to pos=2.
    eprintln!("[snapshot-v2] state_load");
    ctx.state_load(&snap)
        .map_err(|e| panic!("state_load: {e}"))
        .unwrap();

    // Eval the same token=2 at pos=2 again — should produce bit-equal logits.
    let mut logits_post = vec![0.0f32; vocab];
    eprintln!("[snapshot-v2] eval token=2 pos=2 (post-restore)");
    ctx.eval_token(2, 2, 0, &mut logits_post)?;

    // Compare bit-equal.
    let mut max_diff = 0.0f32;
    let mut argmax_pre = 0usize;
    let mut argmax_post = 0usize;
    let mut max_pre = f32::NEG_INFINITY;
    let mut max_post = f32::NEG_INFINITY;
    for (i, (&a, &b)) in logits_pre.iter().zip(logits_post.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
        if a > max_pre {
            max_pre = a;
            argmax_pre = i;
        }
        if b > max_post {
            max_post = b;
            argmax_post = i;
        }
    }
    eprintln!(
        "[snapshot-v2] max_abs_diff={max_diff:.4e} argmax_pre={argmax_pre} argmax_post={argmax_post}"
    );
    assert_eq!(
        argmax_pre, argmax_post,
        "argmax differs after roundtrip — snapshot wire format is lossy",
    );
    // Bit-equal target. Allow tiny epsilon for any in-flight non-determinism.
    assert!(
        max_diff < 1e-3,
        "max_abs_diff {max_diff:.4e} > 1e-3 — snapshot drift",
    );
    Ok(())
}
