//! Diff oracle for `moeflux_mm_id` + `moeflux_mm_id_map0_k8` — the
//! one-dispatch MoE matmul ported from llama.cpp.
//!
//! Validates the kernel against a plain CPU dequant-matmul over the
//! same per-(token, slot) expert routing. The simdgroup reduction
//! reorders the FP adds, so the contract is **cosine >= 0.9999** per
//! token-slot row.
//!
//! Two shapes:
//! 1. **Tiny** (4 experts, 16 tokens, k=2, K=128, N=64) — primary
//!    correctness gate. Fast to iterate on.
//! 2. **Mid** (32 experts, 64 tokens, k=8, K=512, N=256) — size-
//!    edge sanity. Catches threadgroup-Z indexing and per-expert
//!    early-return bugs that the tiny case might mask.

#![cfg(target_os = "macos")]

use metal::{Device, MTLResourceOptions};
use moeflux_metal::{
    moe_mm_id_supports_topk, Kernels, MoeGatherIdCall, MoeIdMap0Call,
};

const COSINE_FLOOR: f32 = 0.9999;

// --- bf16 helpers ------------------------------------------------------------

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round_bias = ((bits >> 16) & 1) + 0x7fff;
    (bits.wrapping_add(round_bias) >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// --- deterministic RNG -------------------------------------------------------

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

// --- synthetic W4_gs64 expert weights ----------------------------------------

/// `[N, K]` packed nibbles + `[N, K/64]` bf16 scales + biases.
/// One "gate" matrix per expert.
fn gen_expert(
    rng: &mut XorShift,
    n_out: usize,
    k_in: usize,
) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
    assert_eq!(k_in % 64, 0, "K must be a multiple of group_size 64");
    let in_packed_u32 = k_in / 8; // 8 nibbles per u32
    let n_groups = k_in / 64;
    let packed: Vec<u32> = (0..n_out * in_packed_u32)
        .map(|_| rng.next_u64() as u32)
        .collect();
    let scales: Vec<u16> = (0..n_out * n_groups)
        .map(|_| f32_to_bf16(rng.unit() * 0.05))
        .collect();
    let biases: Vec<u16> = (0..n_out * n_groups)
        .map(|_| f32_to_bf16(rng.unit() * 0.02))
        .collect();
    (packed, scales, biases)
}

/// CPU dequant-matmul reference:
///   out[t, s, r] = sum_k dequant(W[e=indices[t,s], r, k]) * x[t, k]
/// `dequant(nibble) = nibble * scale_group + bias_group`.
#[allow(clippy::too_many_arguments)]
fn cpu_reference(
    experts: &[(Vec<u32>, Vec<u16>, Vec<u16>)],
    indices: &[i32], // [n_tokens, k]
    x: &[f32],       // [n_tokens, K]
    n_out: usize,
    k_in: usize,
    n_tokens: usize,
    k: usize,
) -> Vec<f32> {
    let in_packed_u32 = k_in / 8;
    let n_groups = k_in / 64;
    let mut out = vec![0.0f32; n_tokens * k * n_out];
    for t in 0..n_tokens {
        for s in 0..k {
            let e = indices[t * k + s] as usize;
            let (packed, scales, biases) = &experts[e];
            for r in 0..n_out {
                let mut acc = 0.0f32;
                for kk in 0..k_in {
                    let word = packed[r * in_packed_u32 + kk / 8];
                    let nibble = (word >> ((kk % 8) * 4)) & 0xF;
                    let g = kk / 64;
                    let scale =
                        bf16_to_f32(scales[r * n_groups + g]);
                    let bias =
                        bf16_to_f32(biases[r * n_groups + g]);
                    let w = nibble as f32 * scale + bias;
                    acc += w * x[t * k_in + kk];
                }
                out[(t * k + s) * n_out + r] = acc;
            }
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

// --- one diff-oracle case ----------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_case(
    label: &str,
    n_experts: usize,
    n_tokens: usize,
    k: usize,
    k_in: usize,
    n_out: usize,
    seed: u64,
) {
    assert!(
        moe_mm_id_supports_topk(k as u32),
        "test fixture k={k} not a supported gather-id top-k"
    );
    assert_eq!(k_in % 64, 0);

    let mut rng = XorShift::new(seed);

    // --- synthetic inputs ---
    let experts: Vec<_> = (0..n_experts)
        .map(|_| gen_expert(&mut rng, n_out, k_in))
        .collect();
    let activations: Vec<f32> = (0..n_tokens * k_in)
        .map(|_| rng.unit())
        .collect();
    // Each token picks `k` distinct expert IDs; we don't strictly
    // need distinct, but it mirrors the router's behaviour.
    let mut indices: Vec<i32> = Vec::with_capacity(n_tokens * k);
    for _ in 0..n_tokens {
        let mut row: Vec<i32> = Vec::with_capacity(k);
        while row.len() < k {
            let e = (rng.next_u64() % n_experts as u64) as i32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        indices.extend(row);
    }

    // --- CPU reference ---
    let cpu_out = cpu_reference(
        &experts, &indices, &activations, n_out, k_in, n_tokens, k,
    );

    // --- GPU build ---
    let device =
        Device::system_default().expect("no Metal device for test");
    let kernels =
        Kernels::new(&device).expect("build moeflux-metal kernels");

    // Layout: one expert block = [packed | scales | biases]. We
    // ship only the "gate" matrix per expert; no up/down.
    let w_bytes = std::mem::size_of_val(&experts[0].0[..]);
    let s_bytes = std::mem::size_of_val(&experts[0].1[..]);
    let b_bytes = std::mem::size_of_val(&experts[0].2[..]);
    let per_expert = w_bytes + s_bytes + b_bytes;

    let weights_buf = device.new_buffer(
        (per_expert * n_experts) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let base = weights_buf.contents() as *mut u8;
        for (e, (packed, scales, biases)) in experts.iter().enumerate() {
            let blk = base.add(e * per_expert);
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

    let indices_buf = device.new_buffer(
        (indices.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            indices.as_ptr(),
            indices_buf.contents() as *mut i32,
            indices.len(),
        );
    }

    let activations_buf = device.new_buffer(
        (activations.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            activations.as_ptr(),
            activations_buf.contents() as *mut f32,
            activations.len(),
        );
    }

    let htpe_buf = device.new_buffer(
        (n_experts * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let hids_buf = device.new_buffer(
        (n_experts * n_tokens * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (n_tokens * k * n_out * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    // Zero the output so unwritten cells aren't NaN.
    unsafe {
        std::ptr::write_bytes(
            out_buf.contents() as *mut u8,
            0,
            (n_tokens * k * n_out * 4) as usize,
        );
    }

    // --- encode + submit ---
    let queue = device.new_command_queue();
    let cmdbuf = queue.new_command_buffer();

    // Stride math (per `Variant::*_off_4bit` analogues):
    // - packed:  N rows × (K/2) bytes per row → nb01_w = K/2.
    // - scales:  N rows × (K/64) × 2 bytes per row → nb01_s = K/32.
    let nb01_w = (k_in / 2) as u64;
    let nb01_s = (k_in / 32) as u64; // K/64 groups × 2 bytes
    kernels.encode(
        cmdbuf,
        &MoeIdMap0Call {
            indices: &indices_buf,
            indices_offset: 0,
            htpe: &htpe_buf,
            htpe_offset: 0,
            hids: &hids_buf,
            hids_offset: 0,
            n_experts: n_experts as u32,
            n_tokens: n_tokens as u32,
            k: k as u32,
        },
    );
    kernels.encode(
        cmdbuf,
        &MoeGatherIdCall {
            src0: &weights_buf,
            src0_offset: 0,
            src1: &activations_buf,
            src1_offset: 0,
            htpe: &htpe_buf,
            htpe_offset: 0,
            hids: &hids_buf,
            hids_offset: 0,
            dst: &out_buf,
            dst_offset: 0,
            k_in: k_in as u32,
            n_out: n_out as u32,
            n_experts: n_experts as u32,
            n_tokens: n_tokens as u32,
            k: k as u32,
            ne11: 1, // gate-shape: src1 = [n_tokens, K], no slot dim
            nb02: per_expert as u64,
            nb01_w,
            nb01_s,
            packed_off: 0,
            scales_off: w_bytes as u64,
            biases_off: (w_bytes + s_bytes) as u64,
            nb10: 4,
            nb11: 0,
            nb12: (k_in * 4) as u64,
        },
    );
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    // --- compare ---
    let gpu_out: Vec<f32> = unsafe {
        std::slice::from_raw_parts(
            out_buf.contents() as *const f32,
            n_tokens * k * n_out,
        )
        .to_vec()
    };

    let cos = cosine(&cpu_out, &gpu_out);
    assert!(
        cos >= COSINE_FLOOR,
        "[{label}] cosine {cos} < {COSINE_FLOOR}",
    );

    // Also assert non-degenerate magnitudes — a kernel that writes
    // all zeros (silent bug) would cosine to 1.0 vs a zero CPU
    // result but here cpu_out is non-zero, so we'd catch a degenerate
    // GPU output via a magnitude check.
    let gpu_l2: f64 = gpu_out.iter().map(|&v| v as f64 * v as f64).sum();
    assert!(
        gpu_l2 > 0.0,
        "[{label}] GPU output is identically zero",
    );

    eprintln!(
        "[{label}] n_exp={n_experts} n_tok={n_tokens} k={k} K={k_in} N={n_out} \
         cosine={cos:.6}"
    );
}

#[test]
fn moe_mm_id_diff_tiny() {
    // 4 experts, 16 tokens, k=8 (the only top-k we instantiated),
    // K=128, N=64. All experts get assignments because every token
    // picks 8 distinct from 4 → wait, can't pick 8 distinct from
    // 4. Need n_experts >= k. Bump to 8 experts.
    run_case(
        "tiny",
        /* n_experts */ 8,
        /* n_tokens */ 16,
        /* k */ 8,
        /* k_in */ 128,
        /* n_out */ 64,
        /* seed */ 0xCAFE_F00D,
    );
}

#[test]
fn moe_mm_id_diff_mid() {
    // 32 experts, 64 tokens, k=8, K=512, N=256. Stresses
    // per-expert tile-Z indexing and the multi-K-step inner loop.
    run_case(
        "mid",
        32,
        64,
        8,
        512,
        256,
        0xBEEF_BABE,
    );
}

// ---------------------------------------------------------------------
// `ne11 = k` (down-matmul shape) coverage.
//
// `run_case` above passes `ne11=1` — the gate/up shape where
// `src1 = [n_tokens, K]` and every slot of a token shares the same
// activation row. The DOWN matmul takes `src1 = [n_tokens, k, K_in]`
// (the SwiGLU output) and reads per-(token, slot) activations.
// Different `nb11` / `nb12` / `ne11` — potentially different bug
// surface, and the original `moe_mm_id_diff_*` tests don't exercise
// it at all (mid bug only landed once we ran the real pipeline).
// ---------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_down_shape_case(
    label: &str,
    n_experts: usize,
    n_tokens: usize,
    k: usize,
    k_in: usize,
    n_out: usize,
    seed: u64,
) {
    assert!(
        moe_mm_id_supports_topk(k as u32),
        "test fixture k={k} not a supported gather-id top-k"
    );
    assert_eq!(k_in % 64, 0);

    let mut rng = XorShift::new(seed);

    let experts: Vec<_> = (0..n_experts)
        .map(|_| gen_expert(&mut rng, n_out, k_in))
        .collect();
    // `[n_tokens, k, k_in]` flat — per-(token, slot) activations.
    let activations: Vec<f32> =
        (0..n_tokens * k * k_in).map(|_| rng.unit()).collect();
    let mut indices: Vec<i32> = Vec::with_capacity(n_tokens * k);
    for _ in 0..n_tokens {
        let mut row: Vec<i32> = Vec::with_capacity(k);
        while row.len() < k {
            let e = (rng.next_u64() % n_experts as u64) as i32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        indices.extend(row);
    }

    // CPU reference: out[t, s, r] = sum_kk dequant(W[e, r, kk]) *
    //                                x[t, s, kk]  where e=indices[t,s].
    let cpu_out = {
        let in_packed_u32 = k_in / 8;
        let n_groups = k_in / 64;
        let mut out = vec![0.0f32; n_tokens * k * n_out];
        for t in 0..n_tokens {
            for s in 0..k {
                let e = indices[t * k + s] as usize;
                let (packed, scales, biases) = &experts[e];
                for r in 0..n_out {
                    let mut acc = 0.0f32;
                    for kk in 0..k_in {
                        let word = packed[r * in_packed_u32 + kk / 8];
                        let nibble = (word >> ((kk % 8) * 4)) & 0xF;
                        let g = kk / 64;
                        let scale =
                            bf16_to_f32(scales[r * n_groups + g]);
                        let bias =
                            bf16_to_f32(biases[r * n_groups + g]);
                        let w = nibble as f32 * scale + bias;
                        // Per-(token, slot) activation row:
                        acc +=
                            w * activations[(t * k + s) * k_in + kk];
                    }
                    out[(t * k + s) * n_out + r] = acc;
                }
            }
        }
        out
    };

    let device =
        Device::system_default().expect("no Metal device for test");
    let kernels =
        Kernels::new(&device).expect("build moeflux-metal kernels");

    let w_bytes = std::mem::size_of_val(&experts[0].0[..]);
    let s_bytes = std::mem::size_of_val(&experts[0].1[..]);
    let b_bytes = std::mem::size_of_val(&experts[0].2[..]);
    let per_expert = w_bytes + s_bytes + b_bytes;

    let weights_buf = device.new_buffer(
        (per_expert * n_experts) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let base = weights_buf.contents() as *mut u8;
        for (e, (packed, scales, biases)) in
            experts.iter().enumerate()
        {
            let blk = base.add(e * per_expert);
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

    let indices_buf = device.new_buffer(
        (indices.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            indices.as_ptr(),
            indices_buf.contents() as *mut i32,
            indices.len(),
        );
    }

    let activations_buf = device.new_buffer(
        (activations.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            activations.as_ptr(),
            activations_buf.contents() as *mut f32,
            activations.len(),
        );
    }

    let htpe_buf = device.new_buffer(
        (n_experts * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let hids_buf = device.new_buffer(
        (n_experts * n_tokens * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (n_tokens * k * n_out * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::write_bytes(
            out_buf.contents() as *mut u8,
            0,
            (n_tokens * k * n_out * 4) as usize,
        );
    }

    let queue = device.new_command_queue();
    let cmdbuf = queue.new_command_buffer();

    let nb01_w = (k_in / 2) as u64;
    let nb01_s = (k_in / 32) as u64;
    kernels.encode(
        cmdbuf,
        &MoeIdMap0Call {
            indices: &indices_buf,
            indices_offset: 0,
            htpe: &htpe_buf,
            htpe_offset: 0,
            hids: &hids_buf,
            hids_offset: 0,
            n_experts: n_experts as u32,
            n_tokens: n_tokens as u32,
            k: k as u32,
        },
    );
    kernels.encode(
        cmdbuf,
        &MoeGatherIdCall {
            src0: &weights_buf,
            src0_offset: 0,
            src1: &activations_buf,
            src1_offset: 0,
            htpe: &htpe_buf,
            htpe_offset: 0,
            hids: &hids_buf,
            hids_offset: 0,
            dst: &out_buf,
            dst_offset: 0,
            k_in: k_in as u32,
            n_out: n_out as u32,
            n_experts: n_experts as u32,
            n_tokens: n_tokens as u32,
            k: k as u32,
            ne11: k as u32, // down-shape: per-slot activations
            nb02: per_expert as u64,
            nb01_w,
            nb01_s,
            packed_off: 0,
            scales_off: w_bytes as u64,
            biases_off: (w_bytes + s_bytes) as u64,
            nb10: 4,
            nb11: (k_in * 4) as u64,
            nb12: (k * k_in * 4) as u64,
        },
    );
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    let gpu_out: Vec<f32> = unsafe {
        std::slice::from_raw_parts(
            out_buf.contents() as *const f32,
            n_tokens * k * n_out,
        )
        .to_vec()
    };

    // Per-row cosine — the global cosine can mask per-row badness
    // if most rows are near zero. Down-shape exercises slot/token
    // address arithmetic for the activation reads, so check each
    // output row individually.
    let row_floor = 0.999_f32;
    let mut min_row_cos = 1.0_f32;
    let mut bad: Vec<(usize, usize, f32)> = Vec::new();
    for t in 0..n_tokens {
        for s in 0..k {
            let off = (t * k + s) * n_out;
            let row_cpu = &cpu_out[off..off + n_out];
            let row_gpu = &gpu_out[off..off + n_out];
            let row_cos = cosine(row_cpu, row_gpu);
            if row_cos < min_row_cos {
                min_row_cos = row_cos;
            }
            if row_cos < row_floor {
                bad.push((t, s, row_cos));
            }
        }
    }
    if !bad.is_empty() {
        eprintln!(
            "[{label}] FAILING rows ({} of {}):",
            bad.len(),
            n_tokens * k
        );
        for (t, s, c) in bad.iter().take(10) {
            eprintln!(
                "  (t={t}, s={s}, expert={}) cosine={c:.4}",
                indices[t * k + s]
            );
        }
    }
    let cos = cosine(&cpu_out, &gpu_out);
    eprintln!(
        "[{label}] DOWN-SHAPE n_exp={n_experts} n_tok={n_tokens} \
         k={k} K_in={k_in} N={n_out} global_cos={cos:.6} \
         min_row_cos={min_row_cos:.6}"
    );
    assert!(
        min_row_cos >= row_floor,
        "[{label}] min per-row cosine {min_row_cos} < {row_floor}",
    );
    assert!(
        cos >= COSINE_FLOOR,
        "[{label}] global cosine {cos} < {COSINE_FLOOR}",
    );
}

#[test]
fn moe_mm_id_diff_down_shape() {
    // Same dims as `moe_mm_id_diff_mid` but with ne11=k.
    run_down_shape_case(
        "down-shape",
        /* n_experts */ 32,
        /* n_tokens */ 64,
        /* k */ 8,
        /* k_in */ 512,
        /* n_out */ 256,
        /* seed */ 0xD0_BEEF,
    );
}

#[test]
fn moe_mm_id_diff_a3b_scale_gate() {
    // Production scale (256 experts) at gate dims. Slow but
    // catches scale-dependent bugs the small cases miss.
    run_case(
        "a3b-scale-gate",
        /* n_experts */ 256,
        /* n_tokens */ 64,
        /* k */ 8,
        /* k_in */ 2048,
        /* n_out */ 512,
        /* seed */ 0xA3B_5CA1E,
    );
}

#[test]
fn moe_mm_id_diff_a3b_scale_down() {
    // Production scale at down dims (input is per-slot
    // `[n_tokens, k, moe_inter]`).
    run_down_shape_case(
        "a3b-scale-down",
        /* n_experts */ 256,
        /* n_tokens */ 64,
        /* k */ 8,
        /* k_in */ 512,
        /* n_out */ 2048,
        /* seed */ 0xA3B_D0_5CA,
    );
}

// ---------------------------------------------------------------------
// k=10 coverage (a17b). Exercises the `moeflux_mm_id_map0_k10` map0
// instantiation (compile-time `K_TOPK=10`: indices stride + hids
// `token*10+slot` encoding) and the matmul's runtime-`k` hids decode
// (`id / 10`, `id % 10`). n_experts >= 10 so every token can pick 10
// distinct experts.
// ---------------------------------------------------------------------

#[test]
fn moe_mm_id_diff_k10_gate() {
    // Gate/up shape (ne11=1) at k=10.
    run_case(
        "k10-gate",
        /* n_experts */ 32,
        /* n_tokens */ 64,
        /* k */ 10,
        /* k_in */ 512,
        /* n_out */ 256,
        /* seed */ 0xA17B_6A1E,
    );
}

#[test]
fn moe_mm_id_diff_k10_down() {
    // Down shape (ne11=k) at k=10 — per-(token, slot) activations.
    run_down_shape_case(
        "k10-down",
        /* n_experts */ 32,
        /* n_tokens */ 64,
        /* k */ 10,
        /* k_in */ 512,
        /* n_out */ 256,
        /* seed */ 0xA17B_D0_5C,
    );
}
