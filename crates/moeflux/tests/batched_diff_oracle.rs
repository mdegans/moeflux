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

use moeflux::riir::cpu_matvec::{bf16_matvec_cpu, dequant_matvec_4bit_cpu};
use moeflux::riir::expert_forward::{
    encode_moe_batched_permute_fuse, gpu_expert_forward,
};
use moeflux::riir::gpu_attn::{
    encode_sdpa_causal_tiled, BatchedSdpaPipelines,
};
use moeflux::riir::gpu_matvec::{
    encode_bf16_matmul_n_tokens, encode_matvec_n_tokens, BfMatvecPipelines,
    MatvecPipelines,
};
use moeflux::riir::metal::MetalBackend;
use moeflux::riir::moe_router::build_expert_buckets;
use moeflux::riir::sdpa::sdpa_cpu;
use moeflux::riir::variants::VARIANT;
use moeflux::riir::MtlBuffer;

const GROUP_SIZE: u32 = 64;

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

// ---------------------------------------------------------------------------
// Phase 2: dequant_matvec_4bit_n_tokens vs CPU per-row reference.
// ---------------------------------------------------------------------------

/// Generate synthetic 4-bit weights + bf16 scales/biases for a
/// `[out_dim, in_dim]` quantized weight matrix. Returns (packed,
/// scales, biases) in the same layout the production weight pipeline
/// emits. `in_dim` must be a multiple of GROUP_SIZE=64 (and of 8).
fn gen_4bit_weights(
    rng: &mut XorShift64,
    out_dim: usize,
    in_dim: usize,
) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
    assert!(in_dim % GROUP_SIZE as usize == 0);
    let in_packed = in_dim / 8;
    let num_groups = in_dim / GROUP_SIZE as usize;

    let mut packed = vec![0u32; out_dim * in_packed];
    for w in packed.iter_mut() {
        *w = rng.next_u64() as u32;
    }
    let scales: Vec<u16> = (0..(out_dim * num_groups))
        .map(|_| f32_to_bf16(rng.next_f32() * 0.05))
        .collect();
    let biases: Vec<u16> = (0..(out_dim * num_groups))
        .map(|_| f32_to_bf16(rng.next_f32() * 0.02))
        .collect();
    (packed, scales, biases)
}

/// Build a single Metal buffer holding (packed, scales, biases)
/// concatenated. Returns the buffer and the byte offsets of each
/// section. uint32 packed first (natural 4-byte alignment), then
/// uint16 scales, then uint16 biases.
fn pack_weights_into_buf(
    metal: &MetalBackend,
    packed: &[u32],
    scales: &[u16],
    biases: &[u16],
) -> (Buffer, u64, u64, u64) {
    let w_bytes = packed.len() * std::mem::size_of::<u32>();
    let s_bytes = scales.len() * std::mem::size_of::<u16>();
    let b_bytes = biases.len() * std::mem::size_of::<u16>();
    let total = w_bytes + s_bytes + b_bytes;
    let buf = metal.device().new_buffer(
        total as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let base = buf.contents() as *mut u8;
        std::ptr::copy_nonoverlapping(
            packed.as_ptr() as *const u8,
            base,
            w_bytes,
        );
        std::ptr::copy_nonoverlapping(
            scales.as_ptr() as *const u8,
            base.add(w_bytes),
            s_bytes,
        );
        std::ptr::copy_nonoverlapping(
            biases.as_ptr() as *const u8,
            base.add(w_bytes + s_bytes),
            b_bytes,
        );
    }
    (buf, 0, w_bytes as u64, (w_bytes + s_bytes) as u64)
}

fn run_4bit_n_tokens_test(in_dim: u32, out_dim: u32, n_tokens: u32, seed: u64) {
    let mut rng = XorShift64::new(seed);
    let (packed, scales, biases) =
        gen_4bit_weights(&mut rng, out_dim as usize, in_dim as usize);
    let inputs_f32: Vec<f32> = (0..(n_tokens as usize * in_dim as usize))
        .map(|_| rng.next_f32())
        .collect();

    // ---- CPU oracle, per-token ----
    let mut cpu_out =
        vec![0.0f32; n_tokens as usize * out_dim as usize];
    for t in 0..(n_tokens as usize) {
        let x =
            &inputs_f32[t * in_dim as usize..(t + 1) * in_dim as usize];
        let out = &mut cpu_out
            [t * out_dim as usize..(t + 1) * out_dim as usize];
        dequant_matvec_4bit_cpu(
            &packed,
            &scales,
            &biases,
            in_dim as usize,
            out_dim as usize,
            x,
            out,
        )
        .expect("dequant_matvec_4bit_cpu");
    }
    assert!(
        cpu_out.iter().all(|x| x.is_finite()),
        "CPU oracle produced non-finite output"
    );

    // ---- GPU dispatch ----
    let mut metal = MetalBackend::new().expect("open Metal");
    let pipes = MatvecPipelines::fetch(&mut metal)
        .expect("fetch MatvecPipelines");

    let (w_buf, w_off, s_off, b_off) =
        pack_weights_into_buf(&metal, &packed, &scales, &biases);
    let in_buf = make_buf::<f32>(&metal, inputs_f32.len());
    write_buf(&in_buf, &inputs_f32);
    let out_buf =
        make_buf::<f32>(&metal, n_tokens as usize * out_dim as usize);

    let queue = metal.queue();
    let cmdbuf = queue.new_command_buffer();
    encode_matvec_n_tokens(
        cmdbuf, &pipes, &w_buf, w_off, s_off, b_off, &in_buf, 0,
        &out_buf, 0, in_dim, out_dim, n_tokens, 4,
    );
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    let gpu_out =
        read_buf_f32(&out_buf, n_tokens as usize * out_dim as usize);
    assert!(
        gpu_out.iter().all(|x| x.is_finite()),
        "GPU output has non-finite values"
    );

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
            "in_dim={} token {}: cosine = {:.9}, max_abs_diff = {:.6}",
            in_dim, t, cos, max_abs
        );
        assert!(
            cos >= COSINE_FLOOR,
            "in_dim={} token {} cosine {} below floor {}",
            in_dim,
            t,
            cos,
            COSINE_FLOOR
        );
    }
}

/// v3 path (in_dim ≤ 4096): cached x_shared, ROWS_PER_TG=8.
#[test]
#[ignore = "long-running GPU test"]
fn dequant_matvec_4bit_n_tokens_v3_matches_cpu() {
    run_4bit_n_tokens_test(2048, 512, 4, 0xD3CAFE_BABE_0001);
}

/// fast path (in_dim > 4096): no x_shared, one TG per (row, token).
#[test]
#[ignore = "long-running GPU test"]
fn dequant_matvec_4bit_n_tokens_fast_matches_cpu() {
    run_4bit_n_tokens_test(8192, 256, 4, 0xD3CAFE_BABE_0002);
}

/// N=1 degenerate case for the v3 path: bit-exact vs encode_matvec.
#[test]
#[ignore = "long-running GPU test"]
fn dequant_matvec_4bit_n_tokens_v3_n1_matches_single() {
    use moeflux::riir::gpu_matvec::encode_matvec;

    let in_dim: u32 = 1024;
    let out_dim: u32 = 256;
    let mut rng = XorShift64::new(0xD3CAFE_BABE_0003);
    let (packed, scales, biases) =
        gen_4bit_weights(&mut rng, out_dim as usize, in_dim as usize);
    let input: Vec<f32> =
        (0..in_dim as usize).map(|_| rng.next_f32()).collect();

    let mut metal = MetalBackend::new().expect("open Metal");
    let pipes = MatvecPipelines::fetch(&mut metal)
        .expect("fetch MatvecPipelines");

    let (w_buf, w_off, s_off, b_off) =
        pack_weights_into_buf(&metal, &packed, &scales, &biases);
    let in_buf = make_buf::<f32>(&metal, input.len());
    write_buf(&in_buf, &input);
    let out_single = make_buf::<f32>(&metal, out_dim as usize);
    let out_batched = make_buf::<f32>(&metal, out_dim as usize);

    let queue = metal.queue();
    let cmdbuf = queue.new_command_buffer();

    // Single-row reference: inline-encode the v3_4bit pipeline.
    // (encode_matvec wants an MtlWeightBuf which needs a WeightFile
    // we don't want to load for a synthetic test.)
    use metal::{MTLSize, NSUInteger};
    {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipes.v3_4bit);
        enc.set_buffer(0, Some(&w_buf), w_off as NSUInteger);
        enc.set_buffer(1, Some(&w_buf), s_off as NSUInteger);
        enc.set_buffer(2, Some(&w_buf), b_off as NSUInteger);
        enc.set_buffer(3, Some(&in_buf), 0);
        enc.set_buffer(4, Some(&out_single), 0);
        enc.set_bytes(5, 4, (&out_dim as *const u32).cast());
        enc.set_bytes(6, 4, (&in_dim as *const u32).cast());
        enc.set_bytes(7, 4, (&GROUP_SIZE as *const u32).cast());
        let num_tgs = (out_dim + 7) / 8;
        enc.dispatch_thread_groups(
            MTLSize::new(num_tgs as NSUInteger, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
    encode_matvec_n_tokens(
        cmdbuf, &pipes, &w_buf, w_off, s_off, b_off, &in_buf, 0,
        &out_batched, 0, in_dim, out_dim, 1, 4,
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
}

// ---------------------------------------------------------------------------
// Phase 3: batched causal SDPA vs tokenwise sdpa_cpu loop.
// ---------------------------------------------------------------------------

/// Run the batched causal SDPA on a single command buffer and return
/// the [N, H, head_dim] output as a Vec<f32>.
#[allow(clippy::too_many_arguments)]
fn run_batched_sdpa(
    metal: &mut MetalBackend,
    pipes: &BatchedSdpaPipelines,
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    n_tokens: u32,
    num_heads: u32,
    heads_per_kv: u32,
    head_dim: u32,
    kv_dim: u32,
    start_pos: u32,
    kv_len: u32,
    scale: f32,
) -> Vec<f32> {
    let out_total =
        n_tokens as usize * num_heads as usize * head_dim as usize;
    let state_total = n_tokens as usize * num_heads as usize;

    let q_buf = make_buf::<f32>(metal, q_data.len());
    write_buf(&q_buf, q_data);
    let k_buf = make_buf::<f32>(metal, k_data.len());
    write_buf(&k_buf, k_data);
    let v_buf = make_buf::<f32>(metal, v_data.len());
    write_buf(&v_buf, v_data);
    let out_buf = make_buf::<f32>(metal, out_total);
    let rmax_buf = make_buf::<f32>(metal, state_total);
    let rden_buf = make_buf::<f32>(metal, state_total);
    let vpart_buf = make_buf::<f32>(metal, out_total);

    let queue = metal.queue();
    let cmdbuf = queue.new_command_buffer();
    encode_sdpa_causal_tiled(
        cmdbuf,
        pipes,
        &q_buf,
        &k_buf,
        &v_buf,
        &out_buf,
        &rmax_buf,
        &rden_buf,
        &vpart_buf,
        n_tokens,
        num_heads,
        heads_per_kv,
        head_dim,
        kv_dim,
        start_pos,
        kv_len,
        scale,
    );
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    read_buf_f32(&out_buf, out_total)
}

/// q_gate filled with 1000.0 so sigmoid(1000) = 1.0 exactly in f32 —
/// turns sdpa_cpu's gate into a no-op, matching our gate-free batched
/// kernel.
fn gate_off(q_dim: usize) -> Vec<f32> {
    vec![1000.0f32; q_dim]
}

/// Phase 3c: N=1 against single-shot sdpa_cpu (no-gate via large q_gate).
/// Single tile (kv_len ≤ TILE_SIZE) — kernel arithmetic should be very
/// close to the CPU reference. Floor: cosine ≥ 0.9999.
#[test]
#[ignore = "long-running GPU test"]
fn sdpa_causal_tiled_n1_matches_cpu_single_tile() {
    let num_heads = VARIANT.num_attn_heads as u32;
    let num_kv_heads = VARIANT.num_kv_heads as u32;
    let head_dim = VARIANT.head_dim as u32;
    let heads_per_kv = num_heads / num_kv_heads;
    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_heads as usize * head_dim as usize;

    let n_tokens: u32 = 1;
    let kv_len: u32 = 64;
    let start_pos: u32 = kv_len - 1; // query attends to [0, kv_len)
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut rng = XorShift64::new(0x5DA0_5D5D_C0FE_AAAA);

    let q_data: Vec<f32> = (0..q_dim).map(|_| rng.next_f32() * 0.1).collect();
    let k_data: Vec<f32> = (0..(kv_len as usize * kv_dim as usize))
        .map(|_| rng.next_f32() * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..(kv_len as usize * kv_dim as usize))
        .map(|_| rng.next_f32() * 0.1)
        .collect();

    let gate = gate_off(q_dim);
    let mut cpu_out = vec![0.0f32; q_dim];
    sdpa_cpu(
        kv_len as i32,
        &q_data,
        &gate,
        &k_data,
        &v_data,
        &mut cpu_out,
    )
    .expect("sdpa_cpu");

    let mut metal = MetalBackend::new().expect("open Metal");
    let pipes = BatchedSdpaPipelines::fetch(&mut metal)
        .expect("fetch BatchedSdpaPipelines");

    let gpu_out = run_batched_sdpa(
        &mut metal,
        &pipes,
        &q_data,
        &k_data,
        &v_data,
        n_tokens,
        num_heads,
        heads_per_kv,
        head_dim,
        kv_dim,
        start_pos,
        kv_len,
        scale,
    );

    let cos = cosine_sim(&gpu_out, &cpu_out);
    let max_abs: f32 = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    eprintln!(
        "N=1 single-tile: cosine = {:.9}, max_abs_diff = {:.6}",
        cos, max_abs
    );
    assert!(
        gpu_out.iter().all(|x| x.is_finite()),
        "GPU output has non-finite values"
    );
    assert!(
        cos >= COSINE_FLOOR,
        "N=1 single-tile cosine {} below floor {}",
        cos,
        COSINE_FLOOR
    );
}

/// Phase 3c multi-tile: N=1, kv_len > TILE_SIZE. Exercises the
/// online-softmax merge across multiple tile dispatches.
#[test]
#[ignore = "long-running GPU test"]
fn sdpa_causal_tiled_n1_matches_cpu_multi_tile() {
    let num_heads = VARIANT.num_attn_heads as u32;
    let num_kv_heads = VARIANT.num_kv_heads as u32;
    let head_dim = VARIANT.head_dim as u32;
    let heads_per_kv = num_heads / num_kv_heads;
    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_heads as usize * head_dim as usize;

    let n_tokens: u32 = 1;
    let kv_len: u32 = 5000; // > 4096 tile size → multi-tile
    let start_pos: u32 = kv_len - 1;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut rng = XorShift64::new(0x5DA0_5D5D_C0FE_BBBB);

    let q_data: Vec<f32> = (0..q_dim).map(|_| rng.next_f32() * 0.1).collect();
    let k_data: Vec<f32> = (0..(kv_len as usize * kv_dim as usize))
        .map(|_| rng.next_f32() * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..(kv_len as usize * kv_dim as usize))
        .map(|_| rng.next_f32() * 0.1)
        .collect();

    let gate = gate_off(q_dim);
    let mut cpu_out = vec![0.0f32; q_dim];
    sdpa_cpu(
        kv_len as i32,
        &q_data,
        &gate,
        &k_data,
        &v_data,
        &mut cpu_out,
    )
    .expect("sdpa_cpu");

    let mut metal = MetalBackend::new().expect("open Metal");
    let pipes = BatchedSdpaPipelines::fetch(&mut metal)
        .expect("fetch BatchedSdpaPipelines");

    let gpu_out = run_batched_sdpa(
        &mut metal,
        &pipes,
        &q_data,
        &k_data,
        &v_data,
        n_tokens,
        num_heads,
        heads_per_kv,
        head_dim,
        kv_dim,
        start_pos,
        kv_len,
        scale,
    );

    let cos = cosine_sim(&gpu_out, &cpu_out);
    let max_abs: f32 = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    eprintln!(
        "N=1 multi-tile (kv_len=5000): cosine = {:.9}, max_abs_diff = {:.6}",
        cos, max_abs
    );
    assert!(
        gpu_out.iter().all(|x| x.is_finite()),
        "GPU output has non-finite values"
    );
    assert!(
        cos >= COSINE_FLOOR,
        "N=1 multi-tile cosine {} below floor {}",
        cos,
        COSINE_FLOOR
    );
}

/// Phase 3d: N>1 with per-query causal cutoff. Diff target is a
/// tokenwise sdpa_cpu loop where each query attends to a growing prefix
/// of the K/V cache.
#[test]
#[ignore = "long-running GPU test"]
fn sdpa_causal_tiled_n4_matches_tokenwise_cpu() {
    let num_heads = VARIANT.num_attn_heads as u32;
    let num_kv_heads = VARIANT.num_kv_heads as u32;
    let head_dim = VARIANT.head_dim as u32;
    let heads_per_kv = num_heads / num_kv_heads;
    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_heads as usize * head_dim as usize;

    let n_tokens: u32 = 4;
    let start_pos: u32 = 4;
    let kv_len: u32 = start_pos + n_tokens; // 8
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut rng = XorShift64::new(0x5DA0_5D5D_C0FE_CCCC);

    let q_data: Vec<f32> = (0..(n_tokens as usize * q_dim))
        .map(|_| rng.next_f32() * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..(kv_len as usize * kv_dim as usize))
        .map(|_| rng.next_f32() * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..(kv_len as usize * kv_dim as usize))
        .map(|_| rng.next_f32() * 0.1)
        .collect();

    // Tokenwise CPU oracle: for each query q at absolute position
    // start_pos + q, attend to KV[0..start_pos+q+1].
    let gate = gate_off(q_dim);
    let mut cpu_out_all = vec![0.0f32; n_tokens as usize * q_dim];
    for q in 0..(n_tokens as usize) {
        let kv_max = start_pos as usize + q + 1;
        let q_slice = &q_data[q * q_dim..(q + 1) * q_dim];
        let k_slice =
            &k_data[..kv_max * kv_dim as usize];
        let v_slice =
            &v_data[..kv_max * kv_dim as usize];
        let out_slice = &mut cpu_out_all[q * q_dim..(q + 1) * q_dim];
        sdpa_cpu(
            kv_max as i32,
            q_slice,
            &gate,
            k_slice,
            v_slice,
            out_slice,
        )
        .expect("sdpa_cpu");
    }

    let mut metal = MetalBackend::new().expect("open Metal");
    let pipes = BatchedSdpaPipelines::fetch(&mut metal)
        .expect("fetch BatchedSdpaPipelines");

    let gpu_out = run_batched_sdpa(
        &mut metal,
        &pipes,
        &q_data,
        &k_data,
        &v_data,
        n_tokens,
        num_heads,
        heads_per_kv,
        head_dim,
        kv_dim,
        start_pos,
        kv_len,
        scale,
    );

    assert!(
        gpu_out.iter().all(|x| x.is_finite()),
        "GPU output has non-finite values"
    );

    for q in 0..(n_tokens as usize) {
        let g = &gpu_out[q * q_dim..(q + 1) * q_dim];
        let c = &cpu_out_all[q * q_dim..(q + 1) * q_dim];
        let cos = cosine_sim(g, c);
        let max_abs: f32 = g
            .iter()
            .zip(c.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!(
            "N=4 token {}: cosine = {:.9}, max_abs_diff = {:.6}",
            q, cos, max_abs
        );
        assert!(
            cos >= COSINE_FLOOR,
            "token {} cosine {} below floor {}",
            q,
            cos,
            COSINE_FLOOR
        );
    }
}

// ---------------------------------------------------------------------------
// Phase 4: MoE permute-and-fuse vs tokenwise gpu_expert_forward + sum.
// ---------------------------------------------------------------------------

/// Build a synthetic 4-bit expert blob with the same on-disk layout the
/// production weight pipeline emits. Uses `VARIANT`'s offsets so the
/// produced bytes are consumed unchanged by `gpu_expert_forward` and
/// `encode_moe_batched_permute_fuse`.
fn gen_synth_expert_blob(rng: &mut XorShift64) -> Vec<u8> {
    let v = VARIANT;
    let h = v.hidden_dim;
    let mi = v.moe_intermediate;
    let mut buf = vec![0u8; v.expert_size_4bit()];

    let write_at = |buf: &mut Vec<u8>, off: usize, src_bytes: &[u8]| {
        buf[off..off + src_bytes.len()].copy_from_slice(src_bytes);
    };
    let as_bytes_u32 = |v: &[u32]| -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                v.as_ptr() as *const u8,
                std::mem::size_of_val(v),
            )
        }
    };
    let as_bytes_u16 = |v: &[u16]| -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                v.as_ptr() as *const u8,
                std::mem::size_of_val(v),
            )
        }
    };

    // gate: [mi, h]
    let (packed, scales, biases) = gen_4bit_weights(rng, mi, h);
    write_at(&mut buf, v.gate_w_off_4bit(), as_bytes_u32(&packed));
    write_at(&mut buf, v.gate_s_off_4bit(), as_bytes_u16(&scales));
    write_at(&mut buf, v.gate_b_off_4bit(), as_bytes_u16(&biases));

    // up: [mi, h]
    let (packed, scales, biases) = gen_4bit_weights(rng, mi, h);
    write_at(&mut buf, v.up_w_off_4bit(), as_bytes_u32(&packed));
    write_at(&mut buf, v.up_s_off_4bit(), as_bytes_u16(&scales));
    write_at(&mut buf, v.up_b_off_4bit(), as_bytes_u16(&biases));

    // down: [h, mi]
    let (packed, scales, biases) = gen_4bit_weights(rng, h, mi);
    write_at(&mut buf, v.down_w_off_4bit(), as_bytes_u32(&packed));
    write_at(&mut buf, v.down_s_off_4bit(), as_bytes_u16(&scales));
    write_at(&mut buf, v.down_b_off_4bit(), as_bytes_u16(&biases));

    buf
}

/// Diff `encode_moe_batched_permute_fuse` against a tokenwise loop of
/// `gpu_expert_forward` summed per-token with the routing weights. Same
/// expert blobs, same routing weights, same arithmetic — only the
/// dispatch order differs (per-expert bucket vs per-token slot). The
/// FP-reorder envelope keeps cosine ≥ COSINE_FLOOR per token.
///
/// Fixture: N=4 tokens, K=4 active experts each, num_experts=12 (with 4
/// experts empty by construction — exercises the empty-bucket skip).
/// Routing creates 8 non-empty buckets, each of size 2.
#[test]
#[ignore = "long-running GPU test"]
fn moe_permute_fuse_n_tokens_matches_tokenwise() {
    let v = VARIANT;
    let n_tokens: usize = 4;
    let k_active: usize = 4;
    let num_experts: usize = 12;
    let h = v.hidden_dim;
    let mi = v.moe_intermediate;

    let mut rng = XorShift64::new(0xCAFE_BABE_DEAD_BEEF);

    // Routing fixture: 8 distinct experts in use (0..7), 4 empty (8..11).
    // Each (token, slot) pair distinct within a token (top-K invariant).
    // Bucket sizes: every non-empty bucket has 2 tokens.
    let routing_experts: [[i32; 4]; 4] = [
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [4, 5, 6, 7],
        [6, 7, 0, 1],
    ];

    let mut per_token_indices = vec![0i32; n_tokens * k_active];
    let mut per_token_weights = vec![0.0f32; n_tokens * k_active];
    for t in 0..n_tokens {
        for s in 0..k_active {
            per_token_indices[t * k_active + s] = routing_experts[t][s];
            // Bounded positive weights — keep arithmetic well-conditioned
            // for the FP-reorder envelope.
            per_token_weights[t * k_active + s] =
                rng.next_f32() * 0.4 + 0.3;
        }
    }

    // Post-attn-norm hidden states per token. Synthesized in (-1, 1).
    let mut h_post = vec![0.0f32; n_tokens * h];
    for x in h_post.iter_mut() {
        *x = rng.next_f32();
    }

    // Generate one blob per unique selected expert (parallel to a sorted
    // unique_experts list, which matches buckets.expert_ids order).
    let unique_experts: Vec<i32> = {
        let mut seen = std::collections::BTreeSet::new();
        for &e in &per_token_indices {
            seen.insert(e);
        }
        seen.into_iter().collect()
    };
    assert_eq!(unique_experts.len(), 8);

    let synth_blobs: Vec<Vec<u8>> = (0..unique_experts.len())
        .map(|_| gen_synth_expert_blob(&mut rng))
        .collect();
    let expert_to_blob_idx: std::collections::HashMap<i32, usize> =
        unique_experts
            .iter()
            .enumerate()
            .map(|(i, &e)| (e, i))
            .collect();

    // ---- Tokenwise reference: gpu_expert_forward per (t, s), sum
    // weighted into per_token_ref[t].
    let mut metal = MetalBackend::new().expect("open Metal");
    let mut per_token_ref = vec![0.0f32; n_tokens * h];
    for t in 0..n_tokens {
        let h_post_t = &h_post[t * h..(t + 1) * h];
        for s in 0..k_active {
            let e = per_token_indices[t * k_active + s];
            let w = per_token_weights[t * k_active + s];
            let blob_idx = expert_to_blob_idx[&e];
            let mut out_expert = vec![0.0f32; h];
            gpu_expert_forward(
                &mut metal,
                &synth_blobs[blob_idx],
                h_post_t,
                &mut out_expert,
            )
            .expect("gpu_expert_forward");
            let dst = &mut per_token_ref[t * h..(t + 1) * h];
            for (d, &x) in dst.iter_mut().zip(out_expert.iter()) {
                *d += w * x;
            }
        }
    }

    // ---- Batched (system under test): build CSR, pack inputs, dispatch.
    let buckets = build_expert_buckets(
        &per_token_indices,
        &per_token_weights,
        n_tokens,
        k_active,
        num_experts,
    );
    assert_eq!(buckets.expert_ids, unique_experts);
    let total_assignments = buckets.token_idx.len();
    assert_eq!(total_assignments, n_tokens * k_active);

    // Upload per-bucket expert blobs (parallel to buckets.expert_ids).
    let blobs_mtl: Vec<MtlBuffer<u8>> = buckets
        .expert_ids
        .iter()
        .map(|&e| {
            let idx = expert_to_blob_idx[&e];
            MtlBuffer::<u8>::with_data(metal.device(), &synth_blobs[idx])
        })
        .collect();

    // Host gather: pack post-attn-norm rows into bucket-major layout.
    let mut packed_input = vec![0.0f32; total_assignments * h];
    for (i, &t) in buckets.token_idx.iter().enumerate() {
        let src = &h_post[(t as usize) * h..((t as usize) + 1) * h];
        packed_input[i * h..(i + 1) * h].copy_from_slice(src);
    }

    let in_buf = make_buf::<f32>(&metal, packed_input.len());
    write_buf(&in_buf, &packed_input);
    let gate_buf = make_buf::<f32>(&metal, total_assignments * mi);
    let up_buf = make_buf::<f32>(&metal, total_assignments * mi);
    let act_buf = make_buf::<f32>(&metal, total_assignments * mi);
    let out_buf = make_buf::<f32>(&metal, total_assignments * h);
    let idx_buf = make_buf::<i32>(&metal, total_assignments);
    write_buf(&idx_buf, &buckets.token_idx);
    let w_buf = make_buf::<f32>(&metal, total_assignments);
    write_buf(&w_buf, &buckets.weights);
    let out_sum_buf = make_buf::<f32>(&metal, n_tokens * h);
    write_buf(&out_sum_buf, &vec![0.0f32; n_tokens * h]);

    let matvec_pipes =
        MatvecPipelines::fetch(&mut metal).expect("fetch MatvecPipelines");
    let swiglu = metal
        .pipeline("swiglu_fused")
        .expect("swiglu_fused pipeline")
        .clone();
    let bucket_accumulate = metal
        .pipeline("moe_bucket_accumulate")
        .expect("moe_bucket_accumulate pipeline")
        .clone();

    let queue = metal.queue();
    let cmdbuf = queue.new_command_buffer();
    encode_moe_batched_permute_fuse(
        cmdbuf,
        &matvec_pipes,
        &swiglu,
        &bucket_accumulate,
        &blobs_mtl,
        &in_buf,
        &gate_buf,
        &up_buf,
        &act_buf,
        &out_buf,
        &idx_buf,
        &w_buf,
        &out_sum_buf,
        &buckets,
        v,
    );
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    let gpu_out = read_buf_f32(&out_sum_buf, n_tokens * h);
    assert!(
        gpu_out.iter().all(|x| x.is_finite()),
        "permute-fuse output has non-finite values"
    );

    for t in 0..n_tokens {
        let g = &gpu_out[t * h..(t + 1) * h];
        let c = &per_token_ref[t * h..(t + 1) * h];
        let cos = cosine_sim(g, c);
        let max_abs: f32 = g
            .iter()
            .zip(c.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!(
            "[diff:moe_permute_fuse] token={} cosine={:.9} max_abs={:.3e}",
            t, cos, max_abs
        );
        assert!(
            cos >= COSINE_FLOOR,
            "token {} cosine {:.9} below floor {}",
            t,
            cos,
            COSINE_FLOOR
        );
    }
}
