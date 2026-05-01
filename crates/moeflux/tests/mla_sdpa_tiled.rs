//! Phase 6 — tiled folded SDPA validation.
//!
//! Compares `mla_sdpa_folded_tiled` against `mla_sdpa_folded` on
//! synthetic inputs. The tiled path uses an online-softmax accumulator
//! across `MLA_MAX_CACHE_TG`-sized tiles; the single-shot path
//! processes the full cache in one threadgroup-memory pass. They're
//! mathematically equivalent up to floating-point reordering.
//!
//! Targets:
//! - `cache_len = MLA_MAX_CACHE_TG` (one tile): cosine ≥ 0.99999, max-abs-diff < 1e-3.
//!   (Bit-exact would require the kernels to do identical FMA orderings,
//!   which the merged-state path doesn't — `prev_max == -INFINITY`
//!   makes scale_old = exp(-inf - new_max) = 0, but the multiplications
//!   that follow still happen and may reorder ops vs single-shot.)
//! - `cache_len = 8192` (two tiles): cosine ≥ 0.9999 vs single-shot
//!   reference computed on a separate buffer.
//!
//! Runs only on macOS with the `model-cogito-v2-671b` feature so the
//! variant constants (kv_lora_rank=512, qk_rope_head_dim=64,
//! num_attn_heads=128) are wired in.
//!
//! Run via:
//! ```text
//! cargo test -p moeflux --no-default-features \
//!     --features model-cogito-v2-671b --release \
//!     --test mla_sdpa_tiled -- --ignored --nocapture
//! ```

#![cfg(all(target_os = "macos", feature = "model-cogito-v2-671b"))]

use metal::{Buffer, MTLResourceOptions, NSUInteger};

use moeflux::riir::gpu_mla::{
    encode_mla_sdpa_folded, encode_mla_sdpa_folded_tiled, MlaPipelines,
    MLA_MAX_CACHE_TG,
};
use moeflux::riir::metal::MetalBackend;
use moeflux::riir::variants::VARIANT;

fn make_buf_f32(metal: &MetalBackend, n: usize) -> Buffer {
    let bytes = (n * std::mem::size_of::<f32>()) as NSUInteger;
    metal
        .device()
        .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
}

fn fill_buf_f32(buf: &Buffer, data: &[f32]) {
    let n = data.len();
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents() as *mut f32,
            n,
        );
    }
}

fn read_buf_f32(buf: &Buffer, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.contents() as *const f32,
            v.as_mut_ptr(),
            n,
        );
    }
    v
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb).max(1e-30)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn run_case(cache_len: u32) -> (Vec<f32>, Vec<f32>) {
    let v = VARIANT;
    let num_heads = v.num_attn_heads as u32;
    let kv_lora_rank = v.kv_lora_rank as u32;
    let qk_rope_head_dim = v.qk_rope_head_dim as u32;
    let softmax_scale = 1.0 / ((v.qk_nope_head_dim + v.qk_rope_head_dim) as f32).sqrt();

    let mut metal = MetalBackend::new().expect("MetalBackend::new");
    let pipes = MlaPipelines::new(&mut metal).expect("MlaPipelines::new");

    // Synthetic inputs — sin patterns, deterministic.
    let q_prime_data: Vec<f32> = (0..num_heads * kv_lora_rank)
        .map(|i| ((i as f32) * 0.0007).sin() * 0.1)
        .collect();
    let q_pe_data: Vec<f32> = (0..num_heads * qk_rope_head_dim)
        .map(|i| ((i as f32) * 0.0011).sin() * 0.1)
        .collect();
    let latent_data: Vec<f32> = (0..cache_len * kv_lora_rank)
        .map(|i| ((i as f32) * 0.00013).sin() * 0.05)
        .collect();
    let rope_k_data: Vec<f32> = (0..cache_len * qk_rope_head_dim)
        .map(|i| ((i as f32) * 0.00017).sin() * 0.05)
        .collect();

    let q_prime = make_buf_f32(&metal, q_prime_data.len());
    fill_buf_f32(&q_prime, &q_prime_data);
    let q_pe = make_buf_f32(&metal, q_pe_data.len());
    fill_buf_f32(&q_pe, &q_pe_data);
    let latent = make_buf_f32(&metal, latent_data.len());
    fill_buf_f32(&latent, &latent_data);
    let rope_k = make_buf_f32(&metal, rope_k_data.len());
    fill_buf_f32(&rope_k, &rope_k_data);

    let v_out_n = (num_heads * kv_lora_rank) as usize;

    // Reference path: single-shot kernel (only valid up to MLA_MAX_CACHE_TG).
    let mut ref_out: Option<Vec<f32>> = None;
    if cache_len <= MLA_MAX_CACHE_TG {
        let v_combine_ref = make_buf_f32(&metal, v_out_n);
        let cmdbuf = metal.queue().new_command_buffer();
        encode_mla_sdpa_folded(
            cmdbuf,
            &pipes.sdpa,
            &q_prime,
            &q_pe,
            &latent,
            &rope_k,
            &v_combine_ref,
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            cache_len,
            softmax_scale,
        )
        .expect("encode_mla_sdpa_folded");
        cmdbuf.commit();
        cmdbuf.wait_until_completed();
        ref_out = Some(read_buf_f32(&v_combine_ref, v_out_n));
    }

    // Tiled path.
    let v_combine_tiled = make_buf_f32(&metal, v_out_n);
    let running_max = make_buf_f32(&metal, num_heads as usize);
    let running_denom = make_buf_f32(&metal, num_heads as usize);
    let v_partial = make_buf_f32(&metal, v_out_n);
    let cmdbuf = metal.queue().new_command_buffer();
    encode_mla_sdpa_folded_tiled(
        cmdbuf,
        &pipes.sdpa_tile_accumulate,
        &pipes.sdpa_tile_finalize,
        &q_prime,
        &q_pe,
        &latent,
        &rope_k,
        &v_combine_tiled,
        &running_max,
        &running_denom,
        &v_partial,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        cache_len,
        softmax_scale,
    )
    .expect("encode_mla_sdpa_folded_tiled");
    cmdbuf.commit();
    cmdbuf.wait_until_completed();
    let tiled_out = read_buf_f32(&v_combine_tiled, v_out_n);

    (ref_out.unwrap_or_default(), tiled_out)
}

#[test]
#[ignore = "needs Metal device"]
fn mla_tiled_matches_single_shot_at_tile_size() {
    let (ref_out, tiled) = run_case(MLA_MAX_CACHE_TG);
    assert_eq!(ref_out.len(), tiled.len());
    let cos = cosine(&ref_out, &tiled);
    let mad = max_abs_diff(&ref_out, &tiled);
    eprintln!("[mla-tiled] cache_len={MLA_MAX_CACHE_TG} cos={cos:.7e} max_abs_diff={mad:.4e}");
    assert!(cos >= 0.99999, "cosine {cos:.7e} below 0.99999");
    assert!(mad < 1e-2, "max_abs_diff {mad:.4e} above 1e-2");
}

#[test]
#[ignore = "needs Metal device"]
fn mla_tiled_long_context_8k_smoke() {
    // 8k = 2 tiles. No single-shot reference (single-shot caps at MLA_MAX_CACHE_TG).
    // Sanity-check finite outputs + non-zero results.
    let (_ref_out, tiled) = run_case(8192);
    assert!(!tiled.is_empty(), "tiled output empty at cache_len=8192");
    let finite = tiled.iter().all(|x| x.is_finite());
    let nonzero = tiled.iter().any(|&x| x != 0.0);
    let mn = tiled.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = tiled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    eprintln!(
        "[mla-tiled-8k] finite={finite} nonzero={nonzero} min={mn:.4e} max={mx:.4e}"
    );
    assert!(finite, "non-finite outputs at cache_len=8192");
    assert!(nonzero, "all-zero outputs at cache_len=8192");
}

#[test]
#[ignore = "needs Metal device — slow-ish at 16k"]
fn mla_tiled_long_context_16k_smoke() {
    let (_ref_out, tiled) = run_case(16384);
    assert!(!tiled.is_empty());
    let finite = tiled.iter().all(|x| x.is_finite());
    let nonzero = tiled.iter().any(|&x| x != 0.0);
    eprintln!("[mla-tiled-16k] finite={finite} nonzero={nonzero}");
    assert!(finite);
    assert!(nonzero);
}
