//! P7 diff gate: `encode_gather_qmm_rhs` output vs a self-contained CPU
//! reference.
//!
//! The gather kernel does the MoE expert matmul: all experts' 4-bit affine
//! weights live contiguously in one buffer at a uniform per-expert stride,
//! and `indices[row]` picks the expert for each input row. The kernel
//! collects contiguous same-expert runs (the moeflux bucket-permuted
//! layout) and GEMMs each run.
//!
//! The CPU reference routes each row to `indices[row]`'s expert, dequantizes
//! `nibble * scale + bias`, and matmuls in f32. The simdgroup reduction
//! reorders the FP adds, so the contract is cosine >= 0.9999 per row,
//! across tile-aligned / ragged M and N, single- and multi-expert index
//! arrays, and sorted vs shuffled indices.

#![cfg(target_os = "macos")]

use metal::{Device, MTLResourceOptions};
use moeflux_mlx::{GatherQmmCall, QmmKernels, QuantWeights};

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

/// One expert's `[out_dim, in_dim]` 4-bit affine weights in MLX's layout:
/// packed `[out_dim, in_dim/8]` u32 (8 nibbles/word), bf16 scales + biases
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

/// `out[r] = sum_k dequant(W_e[row_out, k]) * x[r, k]` with `e =
/// indices[r]`, dequant = `nibble * scale + bias`. The diff oracle.
#[allow(clippy::too_many_arguments)]
fn cpu_gather_qmm(
    experts: &[(Vec<u32>, Vec<u16>, Vec<u16>)],
    indices: &[u32],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    n_tokens: usize,
) -> Vec<f32> {
    let in_packed = in_dim / 8;
    let n_groups = in_dim / 64;
    let mut out = vec![0.0f32; n_tokens * out_dim];
    for t in 0..n_tokens {
        let (packed, scales, biases) = &experts[indices[t] as usize];
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

// --- per-row expert indices -------------------------------------------------

/// Contiguous per-expert runs — the moeflux bucket-permuted layout: row 0
/// onward fills expert 0's bucket, then expert 1's, etc. Bucket sizes vary
/// so runs straddle the kernel's 32-row tile boundaries.
fn sorted_indices(rng: &mut XorShift, n_tokens: usize, n_experts: usize) -> Vec<u32> {
    let mut idx = Vec::with_capacity(n_tokens);
    let mut e = 0usize;
    while idx.len() < n_tokens {
        // Run length 1..=47 so some runs are sub-tile and some span tiles.
        let run = 1 + (rng.next_u64() % 47) as usize;
        let expert = (e % n_experts) as u32;
        for _ in 0..run {
            if idx.len() == n_tokens {
                break;
            }
            idx.push(expert);
        }
        e += 1;
    }
    idx
}

/// Fully shuffled — each row independently picks a random expert. Stresses
/// the kernel's run-collection loop with single-row runs.
fn shuffled_indices(rng: &mut XorShift, n_tokens: usize, n_experts: usize) -> Vec<u32> {
    (0..n_tokens)
        .map(|_| (rng.next_u64() % n_experts as u64) as u32)
        .collect()
}

// --- the test ---------------------------------------------------------------

fn run_gather_case(
    in_dim: u32,
    out_dim: u32,
    n_tokens: u32,
    indices: &[u32],
    seed: u64,
    label: &str,
) {
    let mut rng = XorShift::new(seed);
    let n_experts =
        indices.iter().copied().max().unwrap_or(0) as usize + 1;

    let experts: Vec<(Vec<u32>, Vec<u16>, Vec<u16>)> = (0..n_experts)
        .map(|_| gen_weights(&mut rng, out_dim as usize, in_dim as usize))
        .collect();
    let x: Vec<f32> = (0..(n_tokens * in_dim) as usize)
        .map(|_| rng.unit())
        .collect();

    let cpu = cpu_gather_qmm(
        &experts,
        indices,
        &x,
        out_dim as usize,
        in_dim as usize,
        n_tokens as usize,
    );

    // --- GPU ---
    let device = Device::system_default().expect("no Metal device");
    let kernels = QmmKernels::new(&device).expect("build moeflux-mlx kernels");

    // One expert block: [packed u32][scales u16][biases u16] — moeflux's
    // weight-file layout. All experts contiguous at uniform `block_bytes`
    // stride; the kernel reaches expert e via `e * stride_w / stride_s`.
    let w_bytes = std::mem::size_of_val(&experts[0].0[..]);
    let s_bytes = std::mem::size_of_val(&experts[0].1[..]);
    let b_bytes = std::mem::size_of_val(&experts[0].2[..]);
    let block_bytes = w_bytes + s_bytes + b_bytes;
    assert_eq!(block_bytes % 2, 0, "block must be bf16-element aligned");

    let wf = device.new_buffer(
        (block_bytes * n_experts) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let base = wf.contents() as *mut u8;
        for (e, (packed, scales, biases)) in experts.iter().enumerate() {
            let blk = base.add(e * block_bytes);
            std::ptr::copy_nonoverlapping(
                packed.as_ptr() as *const u8,
                blk,
                w_bytes,
            );
            std::ptr::copy_nonoverlapping(
                scales.as_ptr() as *const u8,
                blk.add(w_bytes),
                s_bytes,
            );
            std::ptr::copy_nonoverlapping(
                biases.as_ptr() as *const u8,
                blk.add(w_bytes + s_bytes),
                b_bytes,
            );
        }
    }

    let in_buf = device.new_buffer_with_data(
        x.as_ptr() as *const _,
        std::mem::size_of_val(&x[..]) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let idx_buf = device.new_buffer_with_data(
        indices.as_ptr() as *const _,
        std::mem::size_of_val(indices) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_len = (n_tokens * out_dim) as usize;
    let out_buf = device.new_buffer(
        (out_len * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let queue = device.new_command_queue();
    let cmd = queue.new_command_buffer();
    kernels.encode_gather_qmm_rhs(
        cmd,
        &GatherQmmCall {
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
            indices: &idx_buf,
            indices_offset: 0,
            in_dim,
            out_dim,
            n_tokens,
            stride_w: block_bytes as u64,
            stride_s: (block_bytes / 2) as u64,
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
        "gather {label} {in_dim}->{out_dim} N={n_tokens}: non-finite output"
    );

    let mut worst = 1.0f32;
    for t in 0..n_tokens as usize {
        let g = &gpu[t * out_dim as usize..(t + 1) * out_dim as usize];
        let c = &cpu[t * out_dim as usize..(t + 1) * out_dim as usize];
        let cos = cosine(g, c);
        worst = worst.min(cos);
        assert!(
            cos >= COSINE_FLOOR,
            "gather {label} {in_dim}->{out_dim} N={n_tokens} row {t} \
             (expert {}): cosine {cos} below floor {COSINE_FLOOR}",
            indices[t]
        );
    }
    eprintln!(
        "gather {label} {in_dim}->{out_dim} N={n_tokens} E={n_experts}: \
         worst cosine = {worst:.9}"
    );
}

/// Tile-aligned — n_tokens and out_dim both multiples of 32, sorted runs.
#[test]
fn gather_aligned_matches_cpu() {
    let mut rng = XorShift::new(0x6A11);
    let indices = sorted_indices(&mut rng, 128, 4);
    run_gather_case(2048, 512, 128, &indices, 0x6A11, "aligned");
}

/// Ragged n_tokens — last M-tile is a partial 32-row block.
#[test]
fn gather_ragged_n_tokens_matches_cpu() {
    let mut rng = XorShift::new(0x6B22);
    let indices = sorted_indices(&mut rng, 100, 5);
    run_gather_case(2048, 512, 100, &indices, 0x6B22, "ragged_m");
}

/// Ragged out_dim — last N-tile is partial.
#[test]
fn gather_ragged_out_dim_matches_cpu() {
    let mut rng = XorShift::new(0x6C33);
    let indices = sorted_indices(&mut rng, 96, 3);
    run_gather_case(2048, 580, 96, &indices, 0x6C33, "ragged_n");
}

/// Both axes ragged, small — stresses the bounds-guarded edges.
#[test]
fn gather_ragged_both_matches_cpu() {
    let mut rng = XorShift::new(0x6D44);
    let indices = sorted_indices(&mut rng, 70, 4);
    run_gather_case(2048, 300, 70, &indices, 0x6D44, "ragged_both");
}

/// Single expert — every row routes to expert 0 (degenerate run spanning
/// all tiles).
#[test]
fn gather_single_expert_matches_cpu() {
    let indices = vec![0u32; 128];
    run_gather_case(2048, 512, 128, &indices, 0x6E55, "single_expert");
}

/// Shuffled indices — random expert per row, mostly single-row runs.
#[test]
fn gather_shuffled_indices_matches_cpu() {
    let mut rng = XorShift::new(0x6F66);
    let indices = shuffled_indices(&mut rng, 100, 6);
    run_gather_case(2048, 512, 100, &indices, 0x6F66, "shuffled");
}

/// A production-ish a3b MoE expert shape: hidden 2048 -> moe_intermediate
/// 768, 512 assignments across 8 experts.
#[test]
fn gather_a3b_moe_shape_matches_cpu() {
    let mut rng = XorShift::new(0x7077);
    let indices = sorted_indices(&mut rng, 512, 8);
    run_gather_case(2048, 768, 512, &indices, 0x7077, "a3b_moe");
}
