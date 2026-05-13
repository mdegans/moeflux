//! Batched-prefill diff oracle.
//!
//! Per-kernel validation for the GPU batched-prefill primitives. Each
//! new kernel lands here first with its own small-fixture diff target
//! before composing into the full batched forward (`step_internal_batched`,
//! future session).
//!
//! Run via:
//! ```text
//! cargo test -p moeflux --no-default-features \
//!     --features model-qwen3-6-35b-a3b --release \
//!     --test batched_diff_oracle -- --ignored --nocapture
//! ```
//!
//! Tests don't need model weights — they generate synthetic inputs with
//! a seeded RNG, so the variant feature is only required for the
//! `variants::VARIANT` constants used to size buffers (not load tensors).

#![cfg(target_os = "macos")]

use metal::{Buffer, MTLResourceOptions, NSUInteger};

use moeflux::riir::cpu_matvec::bf16_matvec_cpu;
use moeflux::riir::gpu_matvec::{
    encode_bf16_matmul_n_tokens, BfMatvecPipelines,
};
use moeflux::riir::metal::MetalBackend;

mod common;

use common::diff_helpers::{cosine_sim, COSINE_FLOOR};

// ---------------------------------------------------------------------------
// Local helpers — buffer plumbing + deterministic synthetic data.
// ---------------------------------------------------------------------------

fn make_buf<T>(metal: &MetalBackend, n: usize) -> Buffer {
    let bytes = (n * std::mem::size_of::<T>()) as NSUInteger;
    metal
        .device()
        .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
}

fn write_buf<T: Copy>(buf: &Buffer, data: &[T]) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents() as *mut T,
            data.len(),
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

/// Round-to-nearest-even f32 → bf16. Same algorithm as the production
/// weight pipeline (see `gpu_mla.rs::tests::f32_to_bf16`).
fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounding_bias = ((bits >> 16) & 1) + 0x7fff;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// xorshift64* — deterministic, no dependency on rand.
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// f32 in (-1.0, 1.0).
    fn next_f32(&mut self) -> f32 {
        let u = (self.next_u64() >> 8) as f32 / ((1u64 << 56) as f32);
        u * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Phase 1: bf16_matmul_n_tokens vs CPU per-row reference.
// ---------------------------------------------------------------------------

/// Diff `encode_bf16_matmul_n_tokens` against N independent calls of
/// `bf16_matvec_cpu`. Per-token cosine must be ≥ COSINE_FLOOR (0.9999).
///
/// The GPU kernel does its sum via a tree reduction across 256 partials;
/// the CPU reference uses a linear `mul_add` accumulator. They are
/// mathematically equivalent up to floating-point reordering, so we
/// expect very close (but not bit-exact) agreement.
#[test]
#[ignore = "long-running GPU test"]
fn bf16_matmul_n_tokens_matches_cpu() {
    let n_tokens: u32 = 4;
    let in_dim: u32 = 2048;
    let out_dim: u32 = 512;

    let mut rng = XorShift64::new(0xBA7C_4ED0_D0FF_05AC);

    // ---- Synthetic bf16 weights [out_dim, in_dim] ----
    let weights_f32: Vec<f32> = (0..(out_dim as usize * in_dim as usize))
        .map(|_| rng.next_f32() * 0.1)
        .collect();
    let weights_bf16: Vec<u16> =
        weights_f32.iter().copied().map(f32_to_bf16).collect();
    // Re-decode for the CPU oracle so it sees the same quantized values
    // the GPU sees (avoids spurious mismatch from f32→bf16 rounding).
    let weights_f32_decoded: Vec<f32> = weights_bf16
        .iter()
        .map(|b| f32::from_bits((*b as u32) << 16))
        .collect();

    // ---- Synthetic inputs [n_tokens, in_dim] ----
    let inputs_f32: Vec<f32> = (0..(n_tokens as usize * in_dim as usize))
        .map(|_| rng.next_f32())
        .collect();

    // ---- CPU oracle, per-token ----
    let mut cpu_out =
        vec![0.0f32; n_tokens as usize * out_dim as usize];
    for t in 0..(n_tokens as usize) {
        let x = &inputs_f32[t * in_dim as usize..(t + 1) * in_dim as usize];
        let out = &mut cpu_out
            [t * out_dim as usize..(t + 1) * out_dim as usize];
        bf16_matvec_cpu(
            &weights_bf16,
            in_dim as usize,
            out_dim as usize,
            x,
            out,
        )
        .expect("bf16_matvec_cpu");
    }
    assert!(
        cpu_out.iter().all(|x| x.is_finite()),
        "CPU oracle produced non-finite output"
    );
    // Reference decoded weights are used; the unused `weights_f32` here
    // would be the pre-quantization values. Touch to silence unused-var
    // if future edits drop the bf16 round-trip.
    let _ = weights_f32_decoded;

    // ---- GPU dispatch ----
    let mut metal = MetalBackend::new().expect("open Metal");
    let device = metal.device().clone();
    let pipes = BfMatvecPipelines::fetch(&mut metal)
        .expect("fetch BfMatvecPipelines");

    let w_buf = make_buf::<u16>(&metal, weights_bf16.len());
    write_buf(&w_buf, &weights_bf16);
    let in_buf = make_buf::<f32>(&metal, inputs_f32.len());
    write_buf(&in_buf, &inputs_f32);
    let out_buf =
        make_buf::<f32>(&metal, n_tokens as usize * out_dim as usize);

    let queue = metal.queue();
    let cmdbuf = queue.new_command_buffer();
    encode_bf16_matmul_n_tokens(
        cmdbuf, &pipes, &w_buf, 0, &in_buf, &out_buf, in_dim, out_dim,
        n_tokens,
    );
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    let gpu_out =
        read_buf_f32(&out_buf, n_tokens as usize * out_dim as usize);
    assert!(
        gpu_out.iter().all(|x| x.is_finite()),
        "GPU output has non-finite values"
    );

    // ---- Compare per-token ----
    for t in 0..(n_tokens as usize) {
        let g = &gpu_out[t * out_dim as usize..(t + 1) * out_dim as usize];
        let c = &cpu_out[t * out_dim as usize..(t + 1) * out_dim as usize];
        let cos = cosine_sim(g, c);
        let max_abs: f32 = g
            .iter()
            .zip(c.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!(
            "token {}: cosine = {:.9}, max_abs_diff = {:.6}",
            t, cos, max_abs
        );
        assert!(
            cos >= COSINE_FLOOR,
            "token {} cosine {} below floor {}",
            t,
            cos,
            COSINE_FLOOR
        );
    }
}

/// N=1 degenerate case: `bf16_matmul_n_tokens` with n_tokens=1 must
/// match `encode_bf16_matvec` against the same single input — these
/// have *bit-exact* arithmetic (same per-row reduction order). Catches
/// dispatch / indexing bugs sharper than the CPU diff above.
#[test]
#[ignore = "long-running GPU test"]
fn bf16_matmul_n_tokens_n1_matches_single_matvec() {
    use moeflux::riir::gpu_matvec::encode_bf16_matvec;

    let in_dim: u32 = 1024;
    let out_dim: u32 = 256;
    let mut rng = XorShift64::new(0xDEAD_BEEF_C0FE_BABE);

    let weights_bf16: Vec<u16> = (0..(out_dim as usize * in_dim as usize))
        .map(|_| f32_to_bf16(rng.next_f32() * 0.1))
        .collect();
    let input_f32: Vec<f32> =
        (0..in_dim as usize).map(|_| rng.next_f32()).collect();

    let mut metal = MetalBackend::new().expect("open Metal");
    let device = metal.device().clone();
    let pipes = BfMatvecPipelines::fetch(&mut metal)
        .expect("fetch BfMatvecPipelines");

    let w_buf = make_buf::<u16>(&metal, weights_bf16.len());
    write_buf(&w_buf, &weights_bf16);
    let in_buf = make_buf::<f32>(&metal, input_f32.len());
    write_buf(&in_buf, &input_f32);
    let out_single = make_buf::<f32>(&metal, out_dim as usize);
    let out_batched = make_buf::<f32>(&metal, out_dim as usize);

    let queue = metal.queue();
    let cmdbuf = queue.new_command_buffer();
    encode_bf16_matvec(
        cmdbuf, &pipes, &w_buf, 0, &in_buf, &out_single, in_dim, out_dim,
    );
    encode_bf16_matmul_n_tokens(
        cmdbuf, &pipes, &w_buf, 0, &in_buf, &out_batched, in_dim,
        out_dim, 1,
    );
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    let single = read_buf_f32(&out_single, out_dim as usize);
    let batched = read_buf_f32(&out_batched, out_dim as usize);
    for (i, (s, b)) in single.iter().zip(batched.iter()).enumerate() {
        assert_eq!(
            s.to_bits(),
            b.to_bits(),
            "row {}: single={} batched={} (expected bit-exact for N=1)",
            i,
            s,
            b
        );
    }
    let _ = device; // silence unused
}
