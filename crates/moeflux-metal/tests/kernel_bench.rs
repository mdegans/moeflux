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
use moeflux_metal::{
    set_gather_tile_variant_override, GatherQmmCall, GatherTileVariant,
    Kernels, QmmCall, QuantWeights,
};

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
// Small-M dense bench — `QmmCall` at *per-expert* a3b shapes, swept
// across M. Motivated by `gather_qmm_arch_pivot_plan.md`: a two-stage
// routing design would replace the in-kernel gather with one
// `affine_qmm_t` per expert at M = htpe[e] (mean = chunk×k_active /
// num_experts = 65536 / 128 = 512). The pivot only wins if dense
// throughput holds up at M=512 against the gather kernel's
// ~7186-7768 GFLOP/s.
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn bench_qmm_t_a3b_per_expert_small_m() {
    let device = Device::system_default().expect("no Metal device");
    let kernels = Kernels::new(&device).expect("build moeflux-metal kernels");
    let queue = device.new_command_queue();
    let mut rng = XorShift::new(0xFEED_F00D);

    eprintln!(
        "\nQmmCall (dense matmul) — per-expert a3b shapes, M-sweep, 4-bit gs_64:"
    );

    // (in_dim, out_dim, label) — per-expert MoE matmul shapes. Gate
    // and up share the same shape; down is the transpose-ish.
    let cases: &[(u32, u32, &str)] = &[
        (A3B_HIDDEN, A3B_MOE_INTERMEDIATE, "expert_gate|up (in=2048 out=768)"),
        (A3B_MOE_INTERMEDIATE, A3B_HIDDEN, "expert_down    (in=768 out=2048)"),
    ];

    // M values: 64/256/512/1024 are the pivot-plan's small-M sweep.
    // 4096/8192 anchor the curve against the existing dense bench.
    let m_values: &[u32] = &[64, 256, 512, 1024, 4096, 8192];

    for &(in_dim, out_dim, label) in cases {
        eprintln!("\n  {label}:");
        let (packed, scales, biases) =
            gen_weights(&mut rng, out_dim as usize, in_dim as usize);
        let (wf, w_off, s_off, b_off) =
            pack_one_block_into_buf(&device, &packed, &scales, &biases);

        // Allocate input + output sized for the largest M; re-use the
        // same buffers across the sweep (the kernel reads only the
        // first `n_tokens` rows).
        let m_max = *m_values.iter().max().unwrap();
        let x: Vec<f32> = (0..(m_max * in_dim) as usize)
            .map(|_| rng.unit())
            .collect();
        let in_buf = device.new_buffer_with_data(
            x.as_ptr() as *const _,
            std::mem::size_of_val(&x[..]) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            m_max as u64 * out_dim as u64 * 4,
            MTLResourceOptions::StorageModeShared,
        );

        for &n_tokens in m_values {
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
            report(&format!("M={n_tokens:<5}"), k, &per, flops);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-expert dense vs gather, on the *real* router distribution from
// a3b prefill. Settles the question raised by
// `gather_qmm_arch_pivot_plan.md`: does the per-expert dispatch
// pattern (llama.cpp's `mul_mat_id`) beat the in-kernel gather, on
// *our* `affine_qmm_t` kernel, on the real htpe distribution? Result
// isolates dispatch-pattern from kernel-throughput — if dense loses
// here, the issue isn't the pattern, it's the kernel at small M.
//
// Distribution sourced from a real `prefill_prompt_long.txt` run
// (15692 tokens, chunk 0, layer 0). Sum = 65536 (= 8192 × 8).
// num_zero = ~29 experts unused. Distribution shape captured by
// `htpe_analyze.py`:
//   1..15     12.6% of cells   0.3% of compute
//   16..63    16.9% of cells   2.7% of compute
//   64..255   29.9% of cells  17.4% of compute
//   256..511  15.4% of cells  22.6% of compute
//   512..1023  9.4% of cells  27.1% of compute
//   1024+      4.4% of cells  29.8% of compute
// ---------------------------------------------------------------------------

const HTPE_A3B_L0C0: [u32; 256] = [
    59, 0, 402, 0, 0, 523, 371, 174, 588, 29, 428, 704, 0, 1120, 119, 0,
    1010, 292, 58, 681, 924, 176, 293, 56, 58, 58, 309, 0, 1151, 877, 1, 0,
    232, 116, 0, 59, 117, 1, 466, 112, 58, 877, 58, 2, 117, 176, 0, 103,
    748, 234, 702, 293, 351, 0, 167, 0, 0, 59, 118, 59, 58, 512, 1873, 0,
    312, 232, 333, 520, 117, 243, 293, 0, 979, 0, 114, 302, 752, 1, 780, 425,
    116, 236, 116, 118, 0, 234, 584, 230, 38, 132, 0, 0, 427, 239, 394, 1,
    246, 1411, 176, 29, 0, 211, 0, 0, 112, 92, 59, 165, 176, 58, 0, 56,
    2, 1403, 0, 118, 602, 351, 0, 95, 496, 117, 59, 524, 1247, 53, 0, 0,
    550, 120, 0, 231, 128, 0, 177, 580, 59, 0, 0, 293, 112, 0, 1403, 0,
    175, 0, 175, 58, 410, 271, 1291, 0, 0, 234, 58, 304, 234, 0, 0, 1222,
    268, 194, 58, 326, 252, 43, 1403, 292, 208, 0, 550, 117, 58, 1, 821, 31,
    170, 28, 60, 0, 174, 58, 0, 0, 0, 292, 176, 295, 120, 0, 0, 818,
    2898, 118, 0, 267, 174, 60, 526, 233, 1, 275, 0, 6, 291, 1338, 53, 149,
    59, 351, 232, 1072, 58, 1, 0, 898, 174, 0, 98, 0, 1302, 198, 802, 716,
    0, 58, 293, 0, 0, 174, 1, 0, 0, 0, 150, 1, 0, 139, 1, 280, 352,
    294, 60, 346, 186, 0, 0, 295, 0, 174, 58, 58, 58, 1, 1054, 0,
];

#[test]
#[ignore]
fn bench_per_expert_vs_gather_real_distribution() {
    let device = Device::system_default().expect("no Metal device");
    let kernels = Kernels::new(&device).expect("build moeflux-metal kernels");
    let queue = device.new_command_queue();
    let mut rng = XorShift::new(0xC0FF_EE42);

    let htpe = HTPE_A3B_L0C0;
    let n_experts = htpe.len() as u32;
    let total: u32 = htpe.iter().sum();
    assert_eq!(total, 65536, "expected 8192 × 8 = 65536 assignments");

    // Bucket-permuted offsets. `offsets[e+1] - offsets[e] = htpe[e]`.
    let mut offsets = vec![0u32; (n_experts + 1) as usize];
    for e in 0..n_experts as usize {
        offsets[e + 1] = offsets[e] + htpe[e];
    }
    let n_active = htpe.iter().filter(|&&c| c > 0).count();

    // expert_indices for gather: bucket e occupies rows
    // [offsets[e], offsets[e+1]) and all hold the index `e`.
    let mut indices = vec![0u32; total as usize];
    for e in 0..n_experts as usize {
        let (lo, hi) = (offsets[e] as usize, offsets[e + 1] as usize);
        indices[lo..hi].fill(e as u32);
    }

    eprintln!(
        "\nPer-expert dense vs gather on real a3b distribution \
         (n_experts={n_experts}, active={n_active}, total={total}):"
    );

    let cases: &[(u32, u32, &str)] = &[
        (A3B_HIDDEN, A3B_MOE_INTERMEDIATE, "gate|up (in=2048 out=768)"),
        (A3B_MOE_INTERMEDIATE, A3B_HIDDEN, "down    (in=768 out=2048)"),
    ];

    for &(in_dim, out_dim, label) in cases {
        // Build n_experts distinct weight blocks at uniform stride.
        let experts: Vec<(Vec<u32>, Vec<u16>, Vec<u16>)> = (0..n_experts)
            .map(|_| gen_weights(&mut rng, out_dim as usize, in_dim as usize))
            .collect();
        let (wf, s_off, b_off, stride_bytes) =
            pack_experts_into_buf(&device, &experts);

        let x: Vec<f32> = (0..(total * in_dim) as usize)
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
            total as u64 * out_dim as u64 * 4,
            MTLResourceOptions::StorageModeShared,
        );

        // Total FLOPs for the matmul (sum across all M).
        let flops =
            2.0 * total as f64 * in_dim as f64 * out_dim as f64;

        eprintln!("\n  {label}:");

        // --- Gather, BM=32 (default). ---
        set_gather_tile_variant_override(Some(GatherTileVariant::Bm32Wm2));
        let encode_gather = |cmd: &CommandBufferRef| {
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
                    n_tokens: total,
                    stride_w: stride_bytes,
                    stride_s: stride_bytes / 2,
                },
            );
        };
        let (k_g, per_g) = measure(&device, &queue, &encode_gather);
        report("gather  (1 dispatch)", k_g, &per_g, flops);
        set_gather_tile_variant_override(None);

        // --- Per-expert dense: N dispatches into one cmdbuf. ---
        // Mirrors what the pivot would issue. Skips empty experts.
        let encode_dense = |cmd: &CommandBufferRef| {
            for e in 0..n_experts as usize {
                let m = htpe[e];
                if m == 0 {
                    continue;
                }
                let row_off = offsets[e] as u64;
                let exp_base = e as u64 * stride_bytes;
                kernels.encode(
                    cmd,
                    &QmmCall {
                        weights: QuantWeights {
                            buffer: &wf,
                            packed_offset: exp_base,
                            scales_offset: exp_base + s_off,
                            biases_offset: exp_base + b_off,
                        },
                        input: &in_buf,
                        input_offset: row_off * in_dim as u64 * 4,
                        output: &out_buf,
                        output_offset: row_off * out_dim as u64 * 4,
                        in_dim,
                        out_dim,
                        n_tokens: m,
                    },
                );
            }
        };
        let (k_d, per_d) = measure(&device, &queue, &encode_dense);
        report(
            &format!("dense   ({n_active} dispatches)"),
            k_d,
            &per_d,
            flops,
        );

        let med_g = median(&per_g);
        let med_d = median(&per_d);
        eprintln!(
            "    → dense/gather = {:.3}× ({})",
            med_d / med_g,
            if med_d < med_g { "dense WINS" } else { "gather WINS" }
        );
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

    let n_tokens = A3B_CHUNK * A3B_K_ACTIVE; // 65536 assignments
    let n_experts = A3B_NUM_EXPERTS;
    let indices = bucket_indices(&mut rng, n_tokens as usize, n_experts);

    let cases: &[(u32, u32, &str)] = &[
        // (in_dim, out_dim, label) for the per-expert matmuls.
        (A3B_HIDDEN, A3B_MOE_INTERMEDIATE, "expert_gate_up (in=2048 out=768)"),
        (A3B_MOE_INTERMEDIATE, A3B_HIDDEN, "expert_down (in=768 out=2048)"),
    ];

    // Per-shape: build buffers once, then re-run measure under each
    // tile variant. Variants A/B back-to-back in one process keeps
    // page cache / thermal / scheduler state identical across the
    // pair (per `feedback_bench_discipline.md`).
    for &(in_dim, out_dim, label) in cases {
        // Build n_experts distinct weight blocks at uniform stride.
        let experts: Vec<(Vec<u32>, Vec<u16>, Vec<u16>)> = (0..n_experts)
            .map(|_| gen_weights(&mut rng, out_dim as usize, in_dim as usize))
            .collect();
        let (wf, s_off, b_off, stride_bytes) =
            pack_experts_into_buf(&device, &experts);
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

        let flops =
            2.0 * n_tokens as f64 * in_dim as f64 * out_dim as f64;

        eprintln!(
            "\nGatherQmmCall {label} — a3b 4-bit gs_64, {n_experts} experts:"
        );
        for (variant, vlabel) in &[
            (GatherTileVariant::Bm16Wm2, "BM=16 WM=2 (experimental)"),
            (GatherTileVariant::Bm32Wm2, "BM=32 WM=2 (default)"),
            (GatherTileVariant::Bm64Wm4, "BM=64 WM=4 (experimental)"),
        ] {
            set_gather_tile_variant_override(Some(*variant));
            let (k, per) = measure(&device, &queue, &encode);
            report(vlabel, k, &per, flops);
        }
        set_gather_tile_variant_override(None);
    }
}
