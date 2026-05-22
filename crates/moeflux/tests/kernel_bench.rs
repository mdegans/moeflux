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

use std::ffi::c_void;
use std::time::{Duration, Instant};

use metal::{
    Buffer, CommandBufferRef, CompileOptions, ComputePipelineState,
    FunctionConstantValues, MTLDataType, MTLResourceOptions, MTLSize,
    NSUInteger,
};

use moeflux::riir::backend::gpu::gpu_matvec::{
    encode_matvec_n_tokens, MatvecPipelines,
};
use moeflux_metal::{Kernels, QmmCall, QuantWeights, SdpaCall};
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
/// dispatches each. Returns `(k, sorted per-dispatch ms)`.
fn measure(
    metal: &MetalContext,
    encode: &dyn Fn(&CommandBufferRef),
) -> (u32, Vec<f64>) {
    let _ = time_cmdbuf(metal, 1, encode);
    let probe = time_cmdbuf(metal, 1, encode).as_secs_f64() * 1e3;
    let k = ((TARGET_MS / probe).round() as u32).clamp(1, MAX_K);
    let mut per: Vec<f64> = (0..TRIALS)
        .map(|_| time_cmdbuf(metal, k, encode).as_secs_f64() * 1e3 / k as f64)
        .collect();
    per.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (k, per)
}

/// Median of a sorted trial vector.
fn median(per: &[f64]) -> f64 {
    per[per.len() / 2]
}

/// [`measure`] + print a GFLOP/s line.
fn bench(
    metal: &MetalContext,
    label: &str,
    flops_per_dispatch: f64,
    encode: &dyn Fn(&CommandBufferRef),
) {
    let (k, per) = measure(metal, encode);
    let med = median(&per);
    let gflops = flops_per_dispatch / (med * 1e6);
    eprintln!(
        "  {label:<40} K={k:<3} {med:>10.3} ms  {gflops:>9.1} GFLOP/s  \
         (trials {:.3}/{:.3}/{:.3})",
        per[0],
        med,
        per[per.len() - 1],
    );
}

// ---------------------------------------------------------------------------
// SDPA ablation — Phase 0 bound analysis (session 13).
//
// Builds `attn_sdpa_causal_flash_va` with the `ABLATE_*` phase-skip /
// vec4-stage function constants (see `sdpa.metal`) and times each
// variant. The dominant cost (staging / QK / softmax / P·V /
// loop+barrier floor) falls out by difference: `A − skip(X) = cost(X)`.
// ---------------------------------------------------------------------------

/// SDPA query-tile size — must match `FA_BR` in `sdpa.metal`.
const FA_BR: u32 = 64;
/// SDPA threads per threadgroup — must match `FA_THREADS` in `sdpa.metal`.
const FA_THREADS: u32 = 256;

// `ABLATE_*` function-constant indices — must match `sdpa.metal`.
const FC_SKIP_QK: NSUInteger = 100;
const FC_SKIP_SOFTMAX: NSUInteger = 101;
const FC_SKIP_PV: NSUInteger = 102;
const FC_SKIP_STAGE: NSUInteger = 103;

struct SdpaArgs<'a> {
    q: &'a Buffer,
    k: &'a Buffer,
    v: &'a Buffer,
    out: &'a Buffer,
    n_tokens: u32,
    num_heads: u32,
    heads_per_kv: u32,
    kv_dim: u32,
    start_pos: u32,
    kv_len: u32,
    scale: f32,
}

/// Encode one SDPA dispatch with `pso` — mirrors `SdpaCall::encode`
/// (moeflux-metal). `fold` is the GQA-fold factor: the grid is
/// `num_q_tiles × num_heads / fold` (1 for the unfolded ablation PSOs).
fn encode_sdpa(
    cmd: &CommandBufferRef,
    pso: &ComputePipelineState,
    a: &SdpaArgs,
    fold: u32,
) {
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    enc.set_buffer(0, Some(a.q), 0);
    enc.set_buffer(1, Some(a.k), 0);
    enc.set_buffer(2, Some(a.v), 0);
    enc.set_buffer(3, Some(a.out), 0);
    enc.set_bytes(4, 4, (&a.n_tokens as *const u32).cast());
    enc.set_bytes(5, 4, (&a.num_heads as *const u32).cast());
    enc.set_bytes(6, 4, (&a.heads_per_kv as *const u32).cast());
    enc.set_bytes(7, 4, (&a.kv_dim as *const u32).cast());
    enc.set_bytes(8, 4, (&a.start_pos as *const u32).cast());
    enc.set_bytes(9, 4, (&a.kv_len as *const u32).cast());
    enc.set_bytes(10, 4, (&a.scale as *const f32).cast());
    let total_tgs = a.n_tokens.div_ceil(FA_BR) * (a.num_heads / fold);
    enc.dispatch_thread_groups(
        MTLSize::new(total_tgs as NSUInteger, 1, 1),
        MTLSize::new(FA_THREADS as NSUInteger, 1, 1),
    );
    enc.end_encoding();
}

/// One SDPA PSO under test: label, pipeline, GQA-fold factor.
struct SdpaPso {
    label: &'static str,
    pso: ComputePipelineState,
    fold: u32,
}

/// Build the SDPA PSOs from one compiled library: 6 unfolded ablation
/// variants (fold 1) + the 3 GQA-folded kernels (fold 2/4/8).
fn build_sdpa_psos(metal: &MetalContext) -> Vec<SdpaPso> {
    let device = metal.device();
    let library = device
        .new_library_with_source(
            &moeflux_metal::assemble_source(),
            &CompileOptions::new(),
        )
        .expect("compile sdpa ablation library");
    let build = |name: &str, flags: &[NSUInteger]| -> ComputePipelineState {
        let fcv = FunctionConstantValues::new();
        let set = true;
        for &idx in flags {
            fcv.set_constant_value_at_index(
                &set as *const bool as *const c_void,
                MTLDataType::Bool,
                idx,
            );
        }
        let function = library
            .get_function(name, Some(fcv))
            .unwrap_or_else(|e| panic!("get {name}: {e}"));
        device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap_or_else(|e| panic!("build {name} pso: {e}"))
    };
    const UNFOLDED: &str = "attn_sdpa_causal_flash_va";
    let mk = |label, pso, fold| SdpaPso { label, pso, fold };
    vec![
        mk("A baseline", build(UNFOLDED, &[]), 1),
        mk("B skip-QK", build(UNFOLDED, &[FC_SKIP_QK]), 1),
        mk("C skip-softmax", build(UNFOLDED, &[FC_SKIP_SOFTMAX]), 1),
        mk("D skip-PV", build(UNFOLDED, &[FC_SKIP_PV]), 1),
        mk("E skip-stage", build(UNFOLDED, &[FC_SKIP_STAGE]), 1),
        mk(
            "F floor",
            build(
                UNFOLDED,
                &[FC_SKIP_QK, FC_SKIP_SOFTMAX, FC_SKIP_PV, FC_SKIP_STAGE],
            ),
            1,
        ),
        mk("G gqa-fold G=2", build("attn_sdpa_causal_flash_gqa2_va", &[]), 2),
        mk("H gqa-fold G=4", build("attn_sdpa_causal_flash_gqa4_va", &[]), 4),
        mk("I gqa-fold G=8", build("attn_sdpa_causal_flash_gqa8_va", &[]), 8),
    ]
}

fn bench_sdpa_ablation(metal: &mut MetalContext) {
    let num_heads = VARIANT.num_attn_heads as u32;
    let num_kv_heads = VARIANT.num_kv_heads as u32;
    let head_dim = VARIANT.head_dim as u32;
    let heads_per_kv = num_heads / num_kv_heads;
    let kv_dim = num_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    assert_eq!(head_dim, 256, "ablation kernel is compiled for head_dim 256");

    let psos = build_sdpa_psos(metal);
    eprintln!(
        "\n[sdpa-ablation] heads={num_heads} kv_heads={num_kv_heads} \
         heads_per_kv={heads_per_kv} head_dim={head_dim}"
    );

    let configs: &[(u32, u32)] =
        &[(1536, 1536), (8192, 8192), (8192, 32768)];
    let mut rng = XorShift64::new(0x5D_0A_0013);
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
        let out_buf = make_buf::<f32>(
            metal,
            m as usize * num_heads as usize * head_dim as usize,
        );
        let args = SdpaArgs {
            q: &q_buf,
            k: &k_buf,
            v: &v_buf,
            out: &out_buf,
            n_tokens: m,
            num_heads,
            heads_per_kv,
            kv_dim,
            start_pos,
            kv_len,
            scale,
        };

        eprintln!("  -- M={m} kv_len={kv_len} --");
        let mut med = vec![f64::NAN; psos.len()];
        for (i, p) in psos.iter().enumerate() {
            // A folded PSO whose register footprint forces the
            // threadgroup below FA_THREADS can't be dispatched at 256
            // threads — that is the register-ceiling readout.
            let max_tg = p.pso.max_total_threads_per_threadgroup();
            if max_tg < FA_THREADS as NSUInteger {
                eprintln!(
                    "  {:<16} occupancy {max_tg}<{FA_THREADS} thr/TG \
                     — register spill, skipped",
                    p.label,
                );
                continue;
            }
            let encode = |cmd: &CommandBufferRef| {
                encode_sdpa(cmd, &p.pso, &args, p.fold)
            };
            let (reps, per) = measure(metal, &encode);
            med[i] = median(&per);
            let occ = if p.fold > 1 {
                format!("  [{max_tg} thr/TG]")
            } else {
                String::new()
            };
            eprintln!(
                "  {:<16} K={reps:<3} {:>10.3} ms  \
                 (trials {:.3}/{:.3}/{:.3}){occ}",
                p.label,
                med[i],
                per[0],
                med[i],
                per[per.len() - 1],
            );
        }

        // Attribution: A − skip(X) = cost of X.
        let a = med[0];
        let qk = a - med[1];
        let sm = a - med[2];
        let pv = a - med[3];
        let stage = a - med[4];
        let floor = med[5];
        let sum = floor + qk + sm + pv + stage;
        let pct = |x: f64| 100.0 * x / a;
        eprintln!(
            "  attribution (A = {a:.3} ms):\n\
             \x20   staging  {stage:>9.3} ms  ({:>5.1}%)\n\
             \x20   QK^T     {qk:>9.3} ms  ({:>5.1}%)\n\
             \x20   softmax  {sm:>9.3} ms  ({:>5.1}%)\n\
             \x20   P-V      {pv:>9.3} ms  ({:>5.1}%)\n\
             \x20   floor    {floor:>9.3} ms  ({:>5.1}%)  (KV loop + barriers)\n\
             \x20   sum {sum:.3} ms vs A {a:.3} ms  (residual {:.3} ms)",
            pct(stage),
            pct(qk),
            pct(sm),
            pct(pv),
            pct(floor),
            a - sum,
        );
        // GQA-fold (G/H/I) vs baseline A — the headline A/B.
        eprintln!("  GQA-fold vs baseline A ({a:.3} ms):");
        for (i, p) in psos.iter().enumerate().skip(6) {
            if med[i].is_nan() {
                eprintln!("    {:<16}  — skipped", p.label);
            } else {
                eprintln!(
                    "    {:<16} {:>10.3} ms   {:.2}× speedup ({:+.1}%)",
                    p.label,
                    med[i],
                    a / med[i],
                    100.0 * (a - med[i]) / a,
                );
            }
        }
    }
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
    let kernels = metal.kernels().clone();

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
        let out_buf = make_buf::<f32>(metal, out_total);

        let flops = sdpa_flops(
            m as u64,
            start_pos as u64,
            num_heads as u64,
            head_dim as u64,
        );
        let encode_flash = |cmd: &CommandBufferRef| {
            kernels.encode(
                cmd,
                &SdpaCall {
                    q: &q_buf,
                    k_cache: &k_buf,
                    v_cache: &v_buf,
                    out: &out_buf,
                    n_tokens: m,
                    num_heads,
                    heads_per_kv,
                    head_dim,
                    kv_dim,
                    start_pos,
                    kv_len,
                    softmax_scale: scale,
                    fold: 1,
                    v2: false,
                },
            );
        };
        bench(
            metal,
            &format!("sdpa M={m} kv_len={kv_len}"),
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
    let kernels = Kernels::new(metal.device())
        .expect("build moeflux-metal Kernels");

    eprintln!("\n[matvec-4bit]  v3 matvec vs qmm_t (MLX)");

    // (in_dim, out_dim, name) — a3b projection shapes. qkv_proj / z_proj
    // are the real linear-attn production shapes (out = conv_dim / total
    // value); q_proj / o_proj are kept for continuity with session 2.
    let shapes: &[(u32, u32, &str)] = &[
        (2048, 12288, "qkv_proj"),
        (2048, 8192, "z_proj"),
        (8192, 2048, "o_proj"),
        (2048, 4096, "q_proj"),
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
                &format!("v3     {name} {in_dim}->{out_dim} M={m}"),
                flops,
                &encode,
            );
            let encode_qmm = |cmd: &CommandBufferRef| {
                kernels.encode(
                    cmd,
                    &QmmCall {
                        weights: QuantWeights {
                            buffer: &w_buf,
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
                        n_tokens: m,
                    },
                );
            };
            bench(
                metal,
                &format!("qmm_t  {name} {in_dim}->{out_dim} M={m}"),
                flops,
                &encode_qmm,
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
    bench_sdpa_ablation(&mut metal);
    bench_matvec(&mut metal);
    eprintln!("=== end microbench ===");
}
