//! P4 diff gate: `encode_qmm_t` output vs a self-contained CPU reference.
//!
//! The CPU reference dequantizes the 4-bit affine weights and runs the
//! matmul in f64-free f32 — the same arithmetic the kernel performs, in a
//! different order. The simdgroup reduction reorders the FP adds, so
//! bit-exactness is not expected; the contract is cosine >= 0.9999 per
//! token row, across tile-aligned and ragged shapes.

#![cfg(target_os = "macos")]

use metal::{Device, MTLResourceOptions};
use moeflux_metal::{QmmCall, QmmKernels, QuantWeights};

const COSINE_FLOOR: f32 = 0.9999;

// --- bf16 <-> f32 -----------------------------------------------------------

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round_bias = ((bits >> 16) & 1) + 0x7fff;
    (bits.wrapping_add(round_bias) >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// --- deterministic RNG ------------------------------------------------------

struct XorShift(u64);

impl XorShift {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        let u = (self.next_u64() >> 8) as f32 / ((1u64 << 56) as f32);
        u * 2.0 - 1.0
    }
}

// --- synthetic 4-bit affine weights -----------------------------------------

/// `[out_dim, in_dim]` 4-bit affine weights in MLX's layout: packed
/// `[out_dim, in_dim/8]` u32 (8 nibbles/word), bf16 scales + biases
/// `[out_dim, in_dim/64]`. `in_dim` must be a multiple of 64.
fn gen_weights(
    rng: &mut XorShift,
    out_dim: usize,
    in_dim: usize,
) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
    assert_eq!(in_dim % 64, 0, "in_dim must be a multiple of group_size 64");
    let in_packed = in_dim / 8;
    let n_groups = in_dim / 64;
    let packed: Vec<u32> =
        (0..out_dim * in_packed).map(|_| rng.next_u64() as u32).collect();
    let scales: Vec<u16> = (0..out_dim * n_groups)
        .map(|_| f32_to_bf16(rng.unit() * 0.05))
        .collect();
    let biases: Vec<u16> = (0..out_dim * n_groups)
        .map(|_| f32_to_bf16(rng.unit() * 0.02))
        .collect();
    (packed, scales, biases)
}

/// `out[t][r] = sum_k dequant(W[r,k]) * x[t][k]`, dequant =
/// `nibble * scale + bias`. The diff oracle.
fn cpu_qmm(
    packed: &[u32],
    scales: &[u16],
    biases: &[u16],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    n_tokens: usize,
) -> Vec<f32> {
    let in_packed = in_dim / 8;
    let n_groups = in_dim / 64;
    let mut out = vec![0.0f32; n_tokens * out_dim];
    for t in 0..n_tokens {
        for r in 0..out_dim {
            let mut acc = 0.0f32;
            for k in 0..in_dim {
                let word = packed[r * in_packed + k / 8];
                let nibble = (word >> ((k % 8) * 4)) & 0xF;
                let g = k / 64;
                let scale = bf16_to_f32(scales[r * n_groups + g]);
                let bias = bf16_to_f32(biases[r * n_groups + g]);
                let w = nibble as f32 * scale + bias;
                acc += w * x[t * in_dim + k];
            }
            out[t * out_dim + r] = acc;
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

// --- the test ---------------------------------------------------------------

fn run_case(in_dim: u32, out_dim: u32, n_tokens: u32, seed: u64) {
    let mut rng = XorShift::new(seed);
    let (packed, scales, biases) =
        gen_weights(&mut rng, out_dim as usize, in_dim as usize);
    let x: Vec<f32> = (0..(n_tokens * in_dim) as usize)
        .map(|_| rng.unit())
        .collect();

    let cpu = cpu_qmm(
        &packed,
        &scales,
        &biases,
        &x,
        out_dim as usize,
        in_dim as usize,
        n_tokens as usize,
    );

    // --- GPU ---
    let device = Device::system_default().expect("no Metal device");
    let kernels = QmmKernels::new(&device).expect("build moeflux-metal kernels");

    // One buffer: [packed u32][scales u16][biases u16], as moeflux's
    // weight file lays them out.
    let w_bytes = std::mem::size_of_val(&packed[..]);
    let s_bytes = std::mem::size_of_val(&scales[..]);
    let b_bytes = std::mem::size_of_val(&biases[..]);
    let wf = device.new_buffer(
        (w_bytes + s_bytes + b_bytes) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let base = wf.contents() as *mut u8;
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

    let in_buf = device.new_buffer_with_data(
        x.as_ptr() as *const _,
        std::mem::size_of_val(&x[..]) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_len = (n_tokens * out_dim) as usize;
    let out_buf = device.new_buffer(
        (out_len * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let queue = device.new_command_queue();
    let cmd = queue.new_command_buffer();
    kernels.encode_qmm_t(
        cmd,
        &QmmCall {
            weights: QuantWeights {
                buffer: &wf,
                packed_offset: 0,
                scales_offset: w_bytes as u64,
                biases_offset: (w_bytes + s_bytes) as u64,
            },
            input: &in_buf,
            input_offset: 0,
            output: &out_buf,
            output_offset: 0,
            in_dim,
            out_dim,
            n_tokens,
        },
    );
    cmd.commit();
    cmd.wait_until_completed();

    let gpu: Vec<f32> = unsafe {
        std::slice::from_raw_parts(out_buf.contents() as *const f32, out_len)
            .to_vec()
    };
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "qmm_t {in_dim}->{out_dim} N={n_tokens}: non-finite output"
    );

    let mut worst = 1.0f32;
    for t in 0..n_tokens as usize {
        let g = &gpu[t * out_dim as usize..(t + 1) * out_dim as usize];
        let c = &cpu[t * out_dim as usize..(t + 1) * out_dim as usize];
        let cos = cosine(g, c);
        worst = worst.min(cos);
        assert!(
            cos >= COSINE_FLOOR,
            "qmm_t {in_dim}->{out_dim} N={n_tokens} token {t}: \
             cosine {cos} below floor {COSINE_FLOOR}"
        );
    }
    eprintln!(
        "qmm_t {in_dim}->{out_dim} N={n_tokens}: worst cosine = {worst:.9}"
    );
}

/// Tile-aligned — n_tokens and out_dim both multiples of 32.
#[test]
fn qmm_t_aligned_matches_cpu() {
    run_case(2048, 512, 128, 0x0A11);
}

/// Ragged out_dim — last N-tile is a partial 32-wide block.
#[test]
fn qmm_t_ragged_out_dim_matches_cpu() {
    run_case(2048, 580, 64, 0x0B22);
}

/// Ragged n_tokens — last M-tile is partial.
#[test]
fn qmm_t_ragged_n_tokens_matches_cpu() {
    run_case(2048, 512, 100, 0x0C33);
}

/// Both axes ragged, small — stresses the bounds-guarded edges.
#[test]
fn qmm_t_ragged_both_matches_cpu() {
    run_case(2048, 300, 70, 0x0D44);
}

/// A production a3b projection shape: qkv_proj 2048 -> 12288.
#[test]
fn qmm_t_a3b_qkv_shape_matches_cpu() {
    run_case(2048, 12288, 256, 0x0E55);
}
