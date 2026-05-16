//! Phase 2 of the kernel arc — moeflux hot-kernel microbench.
//!
//! Times the GPU kernels Phase 1 (`prefill_profile.rs`) fingered as
//! prefill-dominant, isolated from the orchestrator, at a3b prefill
//! shapes. For each kernel/shape: encode K dispatches into one command
//! buffer, `commit` + `wait` once, report ms/dispatch + GFLOP/s.
//! Encoding K per commit amortizes per-cmdbuf overhead out of the
//! measurement (irrelevant for the slow kernels, load-bearing for the
//! fast ones); a warm-up pass first absorbs shader-compile cost.
//!
//! Phase 1 result: `batched_sdpa_causal_tiled` is 43.7% of prefill
//! wall at the production 8192-token chunk and the only superlinear
//! phase, so SDPA is the headline here. The 4-bit dequant matvec
//! (`encode_matvec_n_tokens`) feeds the linear-attn / MoE graphs and
//! is benched second. `encode_bf16_matmul_n_tokens` is not on a
//! labeled hot path — deferred.
//!
//! `#[ignore]` — pure GPU, no model weights (synthetic inputs sized
//! from `variants::VARIANT`). Run with:
//!
//! ```bash
//! cargo test -p moeflux --no-default-features \
//!     --features model-qwen3-6-35b-a3b --release \
//!     --test kernel_bench -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use std::time::{Duration, Instant};

use metal::{Buffer, CommandBufferRef, MTLResourceOptions, NSUInteger};

use moeflux::riir::attn::gpu_attn::{
    encode_sdpa_causal_flash, encode_sdpa_causal_tiled, BatchedSdpaPipelines,
    FlashSdpaPipelines,
};
use moeflux::riir::backend::gpu::gpu_matvec::{
    encode_matvec_n_tokens, MatvecPipelines,
};
use moeflux::riir::variants::VARIANT;
use moeflux::riir::MetalContext;

const GROUP_SIZE: usize = 64;

/// Target measured-cmdbuf GPU time; K is chosen per kernel/shape so
/// `K * single_dispatch ≈ this`. Caps amortization cost on the slow
/// kernels (where K=1 already suffices) while giving the fast ones
/// enough repeats to drown per-cmdbuf overhead.
const TARGET_MS: f64 = 300.0;
const MAX_K: u32 = 64;
const TRIALS: usize = 3;

// ---------------------------------------------------------------------------
// Buffer + RNG helpers (mirrors batched_diff_oracle.rs scaffold).
// ---------------------------------------------------------------------------

fn make_buf<T>(metal: &MetalContext, n: usize) -> Buffer {
    let bytes = (n * std::mem::size_of::<T>()) as NSUInteger;
    metal
        .device()
        .new_buffer(bytes.max(4), MTLResourceOptions::StorageModeShared)
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

/// xorshift64* — deterministic, no `rand` dependency.
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
    fn next_f32(&mut self) -> f32 {
        let u = (self.next_u64() >> 8) as f32 / ((1u64 << 56) as f32);
        u * 2.0 - 1.0
    }
}

fn rand_f32s(rng: &mut XorShift64, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.next_f32()).collect()
}

/// f32 → bf16, round-to-nearest-even.
fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounding_bias = ((bits >> 16) & 1) + 0x7fff;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Synthetic 4-bit weights + bf16 scales/biases for a `[out_dim,
/// in_dim]` quantized matrix, in the production weight-pipeline
/// layout. `in_dim` must be a multiple of `GROUP_SIZE`.
fn gen_4bit_weights(
    rng: &mut XorShift64,
    out_dim: usize,
    in_dim: usize,
) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
    assert!(in_dim % GROUP_SIZE == 0);
    let in_packed = in_dim / 8;
    let num_groups = in_dim / GROUP_SIZE;
    let packed: Vec<u32> =
        (0..out_dim * in_packed).map(|_| rng.next_u64() as u32).collect();
    let scales: Vec<u16> = (0..out_dim * num_groups)
        .map(|_| f32_to_bf16(rng.next_f32() * 0.05))
        .collect();
    let biases: Vec<u16> = (0..out_dim * num_groups)
        .map(|_| f32_to_bf16(rng.next_f32() * 0.02))
        .collect();
    (packed, scales, biases)
}

/// One Metal buffer holding (packed u32, scales u16, biases u16)
/// concatenated. Returns `(buf, w_off, s_off, b_off)`.
fn pack_weights_into_buf(
    metal: &MetalContext,
    packed: &[u32],
    scales: &[u16],
    biases: &[u16],
) -> (Buffer, u64, u64, u64) {
    let w_bytes = std::mem::size_of_val(packed);
    let s_bytes = std::mem::size_of_val(scales);
    let b_bytes = std::mem::size_of_val(biases);
    let buf = metal.device().new_buffer(
        (w_bytes + s_bytes + b_bytes) as NSUInteger,
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

// ---------------------------------------------------------------------------
// Timing harness.
// ---------------------------------------------------------------------------

/// Time `commit` + `wait` of a command buffer built by `encode`,
/// which appends `k` dispatches. Returns the wall duration.
fn time_cmdbuf(
    metal: &MetalContext,
    k: u32,
    encode: &dyn Fn(&CommandBufferRef),
) -> Duration {
    let cmdbuf = metal.queue().new_command_buffer();
    for _ in 0..k {
        encode(cmdbuf);
    }
    let t0 = Instant::now();
    cmdbuf.commit();
    cmdbuf.wait_until_completed();
    t0.elapsed()
}

/// Warm up (one untimed dispatch, absorbs shader-compile), probe a
/// single dispatch to size `K`, then run `TRIALS` timed cmdbufs of K
/// dispatches each. Returns `(k, ms_per_dispatch_median)`.
fn bench(
    metal: &MetalContext,
    label: &str,
    flops_per_dispatch: f64,
    encode: &dyn Fn(&CommandBufferRef),
) {
    // Warm-up + single-dispatch probe.
    let _ = time_cmdbuf(metal, 1, encode);
    let probe = time_cmdbuf(metal, 1, encode).as_secs_f64() * 1e3;
    let k = ((TARGET_MS / probe).round() as u32).clamp(1, MAX_K);

    let mut per: Vec<f64> = (0..TRIALS)
        .map(|_| {
            let d = time_cmdbuf(metal, k, encode);
            d.as_secs_f64() * 1e3 / k as f64
        })
        .collect();
    per.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = per[per.len() / 2];
    let gflops = flops_per_dispatch / (median * 1e6);

    eprintln!(
        "  {label:<40} K={k:<3} {median:>10.3} ms  {gflops:>9.1} GFLOP/s  \
         (trials {:.3}/{:.3}/{:.3})",
        per[0],
        per[per.len() / 2],
        per[per.len() - 1],
    );
}

// ---------------------------------------------------------------------------
// SDPA — the headline kernel.
// ---------------------------------------------------------------------------

/// Useful (causal) FLOP for `encode_sdpa_causal_tiled`: query `i`
/// attends to `start_pos + i + 1` keys; QK and AV each cost
/// `2 * head_dim` per score.
fn sdpa_flops(m: u64, start_pos: u64, num_heads: u64, head_dim: u64) -> f64 {
    let scores = m * start_pos + m * (m + 1) / 2; // per head
    4.0 * head_dim as f64 * num_heads as f64 * scores as f64
}

fn bench_sdpa(metal: &mut MetalContext) {
    let pipes =
        BatchedSdpaPipelines::fetch(metal).expect("fetch BatchedSdpaPipelines");
    let flash_pipes =
        FlashSdpaPipelines::fetch(metal).expect("fetch FlashSdpaPipelines");

    let num_heads = VARIANT.num_attn_heads as u32;
    let num_kv_heads = VARIANT.num_kv_heads as u32;
    let head_dim = VARIANT.head_dim as u32;
    let heads_per_kv = num_heads / num_kv_heads;
    let kv_dim = num_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    eprintln!(
        "\n[sdpa] heads={num_heads} kv_heads={num_kv_heads} \
         head_dim={head_dim} kv_dim={kv_dim}"
    );

    // (M, kv_len): first-chunk shapes (kv_len == M) plus a deep-chunk
    // shape (kv_len > M) — later chunks of a long Agora prompt.
    let configs: &[(u32, u32)] =
        &[(1536, 1536), (8192, 8192), (8192, 32768)];

    let mut rng = XorShift64::new(0x5D_0A_0001);
    for &(m, kv_len) in configs {
        let start_pos = kv_len - m;
        let q = rand_f32s(
            &mut rng,
            m as usize * num_heads as usize * head_dim as usize,
        );
        let k = rand_f32s(&mut rng, kv_len as usize * kv_dim as usize);
        let v = rand_f32s(&mut rng, kv_len as usize * kv_dim as usize);

        let q_buf = make_buf::<f32>(metal, q.len());
        write_buf(&q_buf, &q);
        let k_buf = make_buf::<f32>(metal, k.len());
        write_buf(&k_buf, &k);
        let v_buf = make_buf::<f32>(metal, v.len());
        write_buf(&v_buf, &v);
        let out_total =
            m as usize * num_heads as usize * head_dim as usize;
        let state_total = m as usize * num_heads as usize;
        let out_buf = make_buf::<f32>(metal, out_total);
        let rmax = make_buf::<f32>(metal, state_total);
        let rden = make_buf::<f32>(metal, state_total);
        let vpart = make_buf::<f32>(metal, out_total);

        let flops = sdpa_flops(
            m as u64,
            start_pos as u64,
            num_heads as u64,
            head_dim as u64,
        );
        let encode = |cmd: &CommandBufferRef| {
            encode_sdpa_causal_tiled(
                cmd, &pipes, &q_buf, &k_buf, &v_buf, &out_buf, &rmax,
                &rden, &vpart, m, num_heads, heads_per_kv, head_dim,
                kv_dim, start_pos, kv_len, scale,
            );
        };
        bench(
            metal,
            &format!("sdpa-tiled M={m} kv_len={kv_len}"),
            flops,
            &encode,
        );

        let encode_flash = |cmd: &CommandBufferRef| {
            encode_sdpa_causal_flash(
                cmd, &flash_pipes, &q_buf, &k_buf, &v_buf, &out_buf, m,
                num_heads, heads_per_kv, head_dim, kv_dim, start_pos,
                kv_len, scale,
            );
        };
        bench(
            metal,
            &format!("sdpa-flash M={m} kv_len={kv_len}"),
            flops,
            &encode_flash,
        );
    }
}

// ---------------------------------------------------------------------------
// 4-bit dequant matvec.
// ---------------------------------------------------------------------------

fn bench_matvec(metal: &mut MetalContext) {
    let pipes =
        MatvecPipelines::fetch(metal).expect("fetch MatvecPipelines");

    eprintln!("\n[matvec-4bit]");

    // (in_dim, out_dim, name) — a3b projection shapes.
    let shapes: &[(u32, u32, &str)] = &[
        (2048, 4096, "q_proj"),
        (4096, 2048, "o_proj"),
        (2048, 512, "kv_proj/expert_gate_up"),
        (512, 2048, "expert_down"),
    ];

    let mut rng = XorShift64::new(0x4B17_0002_u64);
    for &m in &[1536u32, 8192u32] {
        for &(in_dim, out_dim, name) in shapes {
            let (packed, scales, biases) = gen_4bit_weights(
                &mut rng,
                out_dim as usize,
                in_dim as usize,
            );
            let (w_buf, w_off, s_off, b_off) =
                pack_weights_into_buf(metal, &packed, &scales, &biases);
            let input =
                rand_f32s(&mut rng, m as usize * in_dim as usize);
            let in_buf = make_buf::<f32>(metal, input.len());
            write_buf(&in_buf, &input);
            let out_buf =
                make_buf::<f32>(metal, m as usize * out_dim as usize);

            let flops = 2.0 * in_dim as f64 * out_dim as f64 * m as f64;
            let encode = |cmd: &CommandBufferRef| {
                encode_matvec_n_tokens(
                    cmd, &pipes, &w_buf, w_off, s_off, b_off, &in_buf,
                    0, &out_buf, 0, in_dim, out_dim, m, 4,
                );
            };
            bench(
                metal,
                &format!("matvec {name} {in_dim}->{out_dim} M={m}"),
                flops,
                &encode,
            );
        }
    }
}

#[test]
#[ignore = "GPU microbench — long-running"]
fn kernel_microbench() {
    let mut metal = MetalContext::new().expect("open Metal");
    eprintln!("=== moeflux kernel microbench (a3b shapes) ===");
    bench_sdpa(&mut metal);
    bench_matvec(&mut metal);
    eprintln!("=== end microbench ===");
}
