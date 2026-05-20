//! Kernel microbench — times the two GPU kernels that dominate
//! moeflux's prefill GPU trace: `affine_qmm_t` (dense matmul, 25%
//! of GPU time per the 2026-05-20 capture) and `affine_gather_qmm_rhs`
//! (MoE expert matmul, 53.72%). Pure kernel-level — no moeflux
//! orchestrator deps. Synthetic weights/inputs sized at a3b
//! production shapes.
//!
//! `#[ignore]` — GPU-only, real timing. Run with:
//!
//! ```bash
//! cargo test -p moeflux-metal --release \
//!     --test kernel_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Encodes K dispatches into one cmdbuf, `commit` + `wait` once,
//! reports median ms/dispatch + GFLOP/s. K is chosen per shape so
//! the cmdbuf wall is ~`TARGET_MS`, amortizing per-cmdbuf overhead
//! out of the per-dispatch number. One untimed warm-up absorbs
//! shader-compile cost.

#![cfg(target_os = "macos")]

use std::time::{Duration, Instant};

use metal::{Buffer, CommandBufferRef, Device, MTLResourceOptions};
use moeflux_metal::{GatherQmmCall, Kernels, QmmCall, QuantWeights};

// Target measured-cmdbuf GPU time; K is chosen per shape so
// `K * single_dispatch ≈ TARGET_MS`.
const TARGET_MS: f64 = 300.0;
const MAX_K: u32 = 64;
const TRIALS: usize = 5;
const GROUP_SIZE: usize = 64;

// a3b production shapes — kept as constants here so the bench
// doesn't pull in `moeflux::riir::variants::VARIANT`.
const A3B_HIDDEN: u32 = 2048;
const A3B_MOE_INTERMEDIATE: u32 = 768;
const A3B_NUM_EXPERTS: u32 = 128;
const A3B_K_ACTIVE: u32 = 8;
const A3B_CHUNK: u32 = 8192;

// ---------------------------------------------------------------------------
// bf16 helpers + RNG.
// ---------------------------------------------------------------------------

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round_bias = ((bits >> 16) & 1) + 0x7fff;
    (bits.wrapping_add(round_bias) >> 16) as u16
}

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
    fn unit(&mut self) -> f32 {
        let u = (self.next_u64() >> 8) as f32 / ((1u64 << 56) as f32);
        u * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// 4-bit weight generation. Matches gather_qmm_diff.rs layout: packed
// `[out_dim, in_dim/8]` u32 + bf16 scales/biases `[out_dim, in_dim/64]`.
// ---------------------------------------------------------------------------

fn gen_weights(
    rng: &mut XorShift,
    out_dim: usize,
    in_dim: usize,
) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
    assert_eq!(in_dim % GROUP_SIZE, 0, "in_dim must be a multiple of GROUP_SIZE");
    let in_packed = in_dim / 8;
    let n_groups = in_dim / GROUP_SIZE;
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

/// Pack `(packed, scales, biases)` into a single Metal buffer
/// matching moeflux's on-disk per-expert layout. Returns the buffer
/// + byte offsets for the scales / biases slabs (packed is at 0).
fn pack_one_block_into_buf(
    device: &Device,
    packed: &[u32],
    scales: &[u16],
    biases: &[u16],
) -> (Buffer, u64, u64, u64) {
    let w_bytes = std::mem::size_of_val(packed);
    let s_bytes = std::mem::size_of_val(scales);
    let b_bytes = std::mem::size_of_val(biases);
    let block_bytes = w_bytes + s_bytes + b_bytes;
    let buf = device.new_buffer(
        block_bytes as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let base = buf.contents() as *mut u8;
        std::ptr::copy_nonoverlapping(packed.as_ptr() as *const u8, base, w_bytes);
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
    (
        buf,
        0,
        w_bytes as u64,
        (w_bytes + s_bytes) as u64,
    )
}

/// Pack `n_experts` distinct expert blocks at uniform `block_bytes`
/// stride. Returns the buffer + stride values for `GatherQmmCall`.
fn pack_experts_into_buf(
    device: &Device,
    experts: &[(Vec<u32>, Vec<u16>, Vec<u16>)],
) -> (Buffer, u64, u64, u64) {
    assert!(!experts.is_empty());
    let w_bytes = std::mem::size_of_val(&experts[0].0[..]);
    let s_bytes = std::mem::size_of_val(&experts[0].1[..]);
    let b_bytes = std::mem::size_of_val(&experts[0].2[..]);
    let block_bytes = w_bytes + s_bytes + b_bytes;
    let buf = device.new_buffer(
        (block_bytes * experts.len()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let base = buf.contents() as *mut u8;
        for (e, (packed, scales, biases)) in experts.iter().enumerate() {
            let blk = base.add(e * block_bytes);
            std::ptr::copy_nonoverlapping(packed.as_ptr() as *const u8, blk, w_bytes);
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
    (
        buf,
        w_bytes as u64,
        (w_bytes + s_bytes) as u64,
        block_bytes as u64,
    )
}

// ---------------------------------------------------------------------------
// Bucket-permuted indices — `n_tokens × k_active` assignments grouped
// by expert, run lengths varying so some buckets straddle the 32-row
// tile boundaries.
// ---------------------------------------------------------------------------

fn bucket_indices(
    rng: &mut XorShift,
    n_assignments: usize,
    n_experts: u32,
) -> Vec<u32> {
    let mut idx = Vec::with_capacity(n_assignments);
    let mut e = 0u32;
    while idx.len() < n_assignments {
        // Mean run ~= n_assignments / n_experts; sampled around it.
        let target = (n_assignments / n_experts as usize).max(1);
        let run = (target.saturating_sub(target / 4))
            + (rng.next_u64() as usize % target.max(1));
        for _ in 0..run {
            if idx.len() == n_assignments {
                break;
            }
            idx.push(e);
        }
        e = (e + 1) % n_experts;
    }
    idx
}

// ---------------------------------------------------------------------------
// Timing harness.
// ---------------------------------------------------------------------------

fn time_cmdbuf(
    device: &Device,
    queue: &metal::CommandQueue,
    k: u32,
    encode: &dyn Fn(&CommandBufferRef),
) -> Duration {
    let _ = device;
    let cmd = queue.new_command_buffer();
    for _ in 0..k {
        encode(cmd);
    }
    let t0 = Instant::now();
    cmd.commit();
    cmd.wait_until_completed();
    t0.elapsed()
}

/// Warm-up + probe + `TRIALS` timed cmdbufs. Returns `(k, sorted
/// per-dispatch ms across trials)`.
fn measure(
    device: &Device,
    queue: &metal::CommandQueue,
    encode: &dyn Fn(&CommandBufferRef),
) -> (u32, Vec<f64>) {
    // Untimed warm-up — absorbs shader compile.
    let _ = time_cmdbuf(device, queue, 1, encode);
    // Probe to size K.
    let probe = time_cmdbuf(device, queue, 1, encode).as_secs_f64() * 1e3;
    let k = ((TARGET_MS / probe).round() as u32).clamp(1, MAX_K);
    let mut per: Vec<f64> = (0..TRIALS)
        .map(|_| {
            time_cmdbuf(device, queue, k, encode).as_secs_f64() * 1e3
                / k as f64
        })
        .collect();
    per.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (k, per)
}

fn median(per: &[f64]) -> f64 {
    per[per.len() / 2]
}

fn report(
    label: &str,
    k: u32,
    per: &[f64],
    flops_per_dispatch: f64,
) {
    let med = median(per);
    let gflops = flops_per_dispatch / (med * 1e6);
    eprintln!(
        "  {label:<48} K={k:<3} {med:>10.3} ms  {gflops:>9.1} GFLOP/s  \
         (trials {:.3}/{:.3}/{:.3})",
        per[0],
        med,
        per[per.len() - 1],
    );
}

// ---------------------------------------------------------------------------
// Dense matmul bench — `QmmCall` at a3b shapes.
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn bench_qmm_t_a3b_dense() {
    let device = Device::system_default().expect("no Metal device");
    let kernels = Kernels::new(&device).expect("build moeflux-metal kernels");
    let queue = device.new_command_queue();
    let mut rng = XorShift::new(0xCAFE_BABE);

    eprintln!(
        "\nQmmCall (dense matmul) — a3b shapes, 4-bit gs_64 weights:"
    );

    let cases: &[(u32, u32, &str)] = &[
        // (in_dim, out_dim, label) — M is fixed at A3B_CHUNK = 8192.
        (A3B_HIDDEN, A3B_HIDDEN, "qkv_proj (in=2048 out=2048)"),
        (A3B_HIDDEN, A3B_MOE_INTERMEDIATE * 2, "gate_up_shared (in=2048 out=1536)"),
        (A3B_MOE_INTERMEDIATE, A3B_HIDDEN, "down_shared (in=768 out=2048)"),
    ];

    for &(in_dim, out_dim, label) in cases {
        let n_tokens = A3B_CHUNK;
        let (packed, scales, biases) =
            gen_weights(&mut rng, out_dim as usize, in_dim as usize);
        let (wf, w_off, s_off, b_off) =
            pack_one_block_into_buf(&device, &packed, &scales, &biases);
        let x: Vec<f32> = (0..(n_tokens * in_dim) as usize)
            .map(|_| rng.unit())
            .collect();
        let in_buf = device.new_buffer_with_data(
            x.as_ptr() as *const _,
            std::mem::size_of_val(&x[..]) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            n_tokens as u64 * out_dim as u64 * 4,
            MTLResourceOptions::StorageModeShared,
        );

        let encode = |cmd: &CommandBufferRef| {
            kernels.encode(
                cmd,
                &QmmCall {
                    weights: QuantWeights {
                        buffer: &wf,
                        packed_offset: w_off,
                        scales_offset: s_off,
                        biases_offset: b_off,
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
        };

        let (k, per) = measure(&device, &queue, &encode);
        let flops =
            2.0 * n_tokens as f64 * in_dim as f64 * out_dim as f64;
        report(label, k, &per, flops);
    }
}

// ---------------------------------------------------------------------------
// MoE gather bench — `GatherQmmCall` at a3b shapes. M = chunk ×
// k_active = 65536 (one full prefill chunk's assignments).
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn bench_gather_qmm_a3b_moe() {
    let device = Device::system_default().expect("no Metal device");
    let kernels = Kernels::new(&device).expect("build moeflux-metal kernels");
    let queue = device.new_command_queue();
    let mut rng = XorShift::new(0xDEAD_BEEF);

    eprintln!(
        "\nGatherQmmCall (MoE expert matmul) — a3b shapes, 4-bit gs_64, \
         128 experts:"
    );

    let n_tokens = A3B_CHUNK * A3B_K_ACTIVE; // 65536 assignments
    let n_experts = A3B_NUM_EXPERTS;
    let indices = bucket_indices(&mut rng, n_tokens as usize, n_experts);

    let cases: &[(u32, u32, &str)] = &[
        // (in_dim, out_dim, label) for the per-expert matmuls.
        (A3B_HIDDEN, A3B_MOE_INTERMEDIATE, "expert_gate_up (in=2048 out=768)"),
        (A3B_MOE_INTERMEDIATE, A3B_HIDDEN, "expert_down (in=768 out=2048)"),
    ];

    for &(in_dim, out_dim, label) in cases {
        // Build n_experts distinct weight blocks at uniform stride.
        let experts: Vec<(Vec<u32>, Vec<u16>, Vec<u16>)> = (0..n_experts)
            .map(|_| gen_weights(&mut rng, out_dim as usize, in_dim as usize))
            .collect();
        let (wf, s_off, b_off, stride_bytes) =
            pack_experts_into_buf(&device, &experts);
        // QmmCall convention: packed_offset is within-block (0 from
        // the block start, but the kernel adds expert_idx * stride_w).
        // s_off / b_off are within-block scale / bias offsets in their
        // bf16 element domain (stride_s == stride_w / 2 from the
        // bf16-2-byte vs. packed-4-byte ratio in this layout).
        let x: Vec<f32> = (0..(n_tokens * in_dim) as usize)
            .map(|_| rng.unit())
            .collect();
        let in_buf = device.new_buffer_with_data(
            x.as_ptr() as *const _,
            std::mem::size_of_val(&x[..]) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let idx_buf = device.new_buffer_with_data(
            indices.as_ptr() as *const _,
            std::mem::size_of_val(&indices[..]) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            n_tokens as u64 * out_dim as u64 * 4,
            MTLResourceOptions::StorageModeShared,
        );

        let encode = |cmd: &CommandBufferRef| {
            kernels.encode(
                cmd,
                &GatherQmmCall {
                    weights: QuantWeights {
                        buffer: &wf,
                        packed_offset: 0,
                        scales_offset: s_off,
                        biases_offset: b_off,
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
                    stride_w: stride_bytes,
                    stride_s: stride_bytes / 2,
                },
            );
        };

        let (k, per) = measure(&device, &queue, &encode);
        let flops =
            2.0 * n_tokens as f64 * in_dim as f64 * out_dim as f64;
        report(label, k, &per, flops);
    }
}
