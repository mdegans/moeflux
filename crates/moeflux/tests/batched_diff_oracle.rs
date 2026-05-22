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

use moeflux::riir::backend::cpu::cpu_matvec::{bf16_matvec_cpu, dequant_matvec_4bit_cpu};
use moeflux::riir::moe::expert_forward::{
    encode_moe_batched_permute_fuse, gpu_expert_forward,
};
use moeflux_metal::SdpaCall;
use moeflux::riir::backend::gpu::gpu_matvec::{
    encode_bf16_matmul_n_tokens, encode_matvec_n_tokens, BfMatvecPipelines,
    MatvecPipelines,
};
use moeflux::riir::moe::gpu_moe_router::{
    encode_moe_router, MoeRouterPipelines,
};
use moeflux::riir::backend::gpu::gpu_norm::{
    encode_residual_add_n_tokens_into, encode_rms_norm_bf16_fused_n_tokens,
    RmsNormBf16FusedNTokensPipeline,
};
use moeflux::riir::MetalContext;
use moeflux::riir::moe::moe_router::{build_expert_buckets, moe_router_cpu};
use moeflux::riir::sdpa_cpu;
use moeflux::riir::variants::VARIANT;
use moeflux::riir::MtlBuffer;
use moeflux::riir::backend::{Backend, BufferPool, CpuBackend, Graph, Op};
use moeflux::riir::WeightFile;

const GROUP_SIZE: u32 = 64;

mod common;

use common::diff_helpers::{cosine_sim, COSINE_FLOOR};

// ---------------------------------------------------------------------------
// Local helpers — buffer plumbing + deterministic synthetic data.
// ---------------------------------------------------------------------------

fn make_buf<T>(metal: &MetalContext, n: usize) -> Buffer {
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
    let mut metal = MetalContext::new().expect("open Metal");
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
    use moeflux::riir::backend::gpu::gpu_matvec::encode_bf16_matvec;

    let in_dim: u32 = 1024;
    let out_dim: u32 = 256;
    let mut rng = XorShift64::new(0xDEAD_BEEF_C0FE_BABE);

    let weights_bf16: Vec<u16> = (0..(out_dim as usize * in_dim as usize))
        .map(|_| f32_to_bf16(rng.next_f32() * 0.1))
        .collect();
    let input_f32: Vec<f32> =
        (0..in_dim as usize).map(|_| rng.next_f32()).collect();

    let mut metal = MetalContext::new().expect("open Metal");
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
/// Reinterpret a `&[T]` of plain-old-data as raw bytes — for pool
/// `upload` of typed host buffers.
fn as_u8<T>(v: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            v.as_ptr() as *const u8,
            std::mem::size_of_val(v),
        )
    }
}

/// Minimal on-disk [`WeightFile`] (one tiny bf16 tensor) so a
/// [`CpuBackend`] can be constructed for Op-level diff tests. The MoE
/// Op reads only pool buffers, never the weight file, so the contents
/// are irrelevant — only that `WeightFile::open` succeeds.
fn dummy_weight_file(tag: &str) -> WeightFile {
    let dir = std::env::temp_dir()
        .join(format!("moeflux-bdo-{}-{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir dummy WF dir");
    let bin = dir.join("model_weights.bin");
    let json = dir.join("model_weights.json");
    std::fs::write(&bin, vec![0u8; 64]).expect("write dummy .bin");
    std::fs::write(
        &json,
        r#"{"tensors":{"dummy":{"offset":0,"size":64,"shape":[32],"dtype":"BF16","bits":0}}}"#,
    )
    .expect("write dummy .json");
    WeightFile::open(&bin, &json).expect("open dummy WF")
}

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
    metal: &MetalContext,
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
    let mut metal = MetalContext::new().expect("open Metal");
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
    use moeflux::riir::backend::gpu::gpu_matvec::encode_matvec;

    let in_dim: u32 = 1024;
    let out_dim: u32 = 256;
    let mut rng = XorShift64::new(0xD3CAFE_BABE_0003);
    let (packed, scales, biases) =
        gen_4bit_weights(&mut rng, out_dim as usize, in_dim as usize);
    let input: Vec<f32> =
        (0..in_dim as usize).map(|_| rng.next_f32()).collect();

    let mut metal = MetalContext::new().expect("open Metal");
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

/// q_gate filled with 1000.0 so sigmoid(1000) = 1.0 exactly in f32 —
/// turns sdpa_cpu's gate into a no-op, matching our gate-free batched
/// kernel.
fn gate_off(q_dim: usize) -> Vec<f32> {
    vec![1000.0f32; q_dim]
}

// ---------------------------------------------------------------------------
// FlashAttention-2 causal SDPA — diff vs tokenwise sdpa_cpu oracle.
// ---------------------------------------------------------------------------

/// Dispatch the SDPA kernel over freshly-written input buffers and read
/// back `out`. `fold` selects unfolded (1) vs the GQA-folded kernel (2)
/// via the production `SdpaCall` path. No scratch buffers — the flash
/// kernel keeps its running state threadgroup/register-resident.
#[allow(clippy::too_many_arguments)]
fn run_batched_sdpa_flash(
    metal: &mut MetalContext,
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
    fold: u32,
    vb: bool,
) -> Vec<f32> {
    let out_total =
        n_tokens as usize * num_heads as usize * head_dim as usize;

    let q_buf = make_buf::<f32>(metal, q_data.len());
    write_buf(&q_buf, q_data);
    let k_buf = make_buf::<f32>(metal, k_data.len());
    write_buf(&k_buf, k_data);
    let v_buf = make_buf::<f32>(metal, v_data.len());
    write_buf(&v_buf, v_data);
    let out_buf = make_buf::<f32>(metal, out_total);

    let queue = metal.queue();
    let cmdbuf = queue.new_command_buffer();
    metal.kernels().encode(
        cmdbuf,
        &SdpaCall {
            q: &q_buf,
            k_cache: &k_buf,
            v_cache: &v_buf,
            out: &out_buf,
            n_tokens,
            num_heads,
            heads_per_kv,
            head_dim,
            kv_dim,
            start_pos,
            kv_len,
            softmax_scale: scale,
            fold,
            vb,
        },
    );
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    read_buf_f32(&out_buf, out_total)
}

/// Diff `attn_sdpa_causal_flash_va` for `n_tokens` queries starting at
/// absolute position `start_pos` against a per-token `sdpa_cpu` oracle
/// (each query `q` attends to `KV[0 .. start_pos+q+1]`). Per-token
/// cosine must clear `COSINE_FLOOR`.
fn flash_diff_tokenwise(
    n_tokens: u32,
    start_pos: u32,
    seed: u64,
    fold: u32,
    vb: bool,
) {
    let num_heads = VARIANT.num_attn_heads as u32;
    let num_kv_heads = VARIANT.num_kv_heads as u32;
    let head_dim = VARIANT.head_dim as u32;
    let heads_per_kv = num_heads / num_kv_heads;
    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_heads as usize * head_dim as usize;
    let kv_len = start_pos + n_tokens;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut rng = XorShift64::new(seed);
    let q_data: Vec<f32> = (0..n_tokens as usize * q_dim)
        .map(|_| rng.next_f32() * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..kv_len as usize * kv_dim as usize)
        .map(|_| rng.next_f32() * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..kv_len as usize * kv_dim as usize)
        .map(|_| rng.next_f32() * 0.1)
        .collect();

    // Tokenwise CPU oracle: query q attends to KV[0 .. start_pos+q+1].
    let gate = gate_off(q_dim);
    let mut cpu = vec![0.0f32; n_tokens as usize * q_dim];
    for q in 0..(n_tokens as usize) {
        let kv_max = start_pos as usize + q + 1;
        sdpa_cpu(
            kv_max as i32,
            &q_data[q * q_dim..(q + 1) * q_dim],
            &gate,
            &k_data[..kv_max * kv_dim as usize],
            &v_data[..kv_max * kv_dim as usize],
            &mut cpu[q * q_dim..(q + 1) * q_dim],
        )
        .expect("sdpa_cpu");
    }

    let mut metal = MetalContext::new().expect("open Metal");
    let gpu = run_batched_sdpa_flash(
        &mut metal, &q_data, &k_data, &v_data, n_tokens, num_heads,
        heads_per_kv, head_dim, kv_dim, start_pos, kv_len, scale, fold,
        vb,
    );

    assert!(
        gpu.iter().all(|x| x.is_finite()),
        "flash GPU output has non-finite values"
    );
    let mut worst = 1.0f32;
    for q in 0..(n_tokens as usize) {
        let g = &gpu[q * q_dim..(q + 1) * q_dim];
        let c = &cpu[q * q_dim..(q + 1) * q_dim];
        let cos = cosine_sim(g, c);
        worst = worst.min(cos);
        if cos < COSINE_FLOOR {
            let hd = head_dim as usize;
            for h in 0..num_heads as usize {
                let gh = &g[h * hd..(h + 1) * hd];
                let ch = &c[h * hd..(h + 1) * hd];
                let gn: f32 = gh.iter().map(|x| x * x).sum::<f32>().sqrt();
                let cn: f32 = ch.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!(
                    "  token {q} head {h}: cos={:.6} |gpu|={gn:.4} \
                     |cpu|={cn:.4}",
                    cosine_sim(gh, ch),
                );
            }
        }
        assert!(
            cos >= COSINE_FLOOR,
            "flash N={n_tokens} start_pos={start_pos} fold={fold} vb={vb} \
             token {q} cosine {cos} below floor {COSINE_FLOOR}"
        );
    }
    eprintln!(
        "flash N={n_tokens} start_pos={start_pos} fold={fold} vb={vb}: \
         worst per-token cosine = {worst:.9}"
    );
}

/// N=1, kv_len=64 — single KV block.
#[test]
#[ignore = "long-running GPU test"]
fn sdpa_va_causal_flash_n1_single_block() {
    flash_diff_tokenwise(1, 63, 0x5DA0_F1A5_0000_0001, 1, false);
}

/// N=1, kv_len=5000 — many KV blocks, exercises the online merge.
#[test]
#[ignore = "long-running GPU test"]
fn sdpa_va_causal_flash_n1_multi_block() {
    flash_diff_tokenwise(1, 4999, 0x5DA0_F1A5_0000_0002, 1, false);
}

/// N=4, start_pos=4 — per-query causal cutoff across a small tile.
#[test]
#[ignore = "long-running GPU test"]
fn sdpa_va_causal_flash_n4_tokenwise() {
    flash_diff_tokenwise(4, 4, 0x5DA0_F1A5_0000_0003, 1, false);
}

/// M=512, start_pos=0 — square causal: 16 query tiles, heavy causal
/// block-skipping.
#[test]
#[ignore = "long-running GPU test"]
fn sdpa_va_causal_flash_m512_square_causal() {
    flash_diff_tokenwise(512, 0, 0x5DA0_F1A5_0000_0004, 1, false);
}

/// M=1500, start_pos=4096 — deep chunk (kv_len > M), and 1500 is not a
/// multiple of FA_BR=64 so the last tile is partial (`br_valid < FA_BR`).
#[test]
#[ignore = "long-running GPU test"]
fn sdpa_va_causal_flash_m1500_deep_chunk() {
    flash_diff_tokenwise(1500, 4096, 0x5DA0_F1A5_0000_0005, 1, false);
}

// GQA-folded staging kernel (`attn_sdpa_causal_flash_gqa2_va`, fold=2) —
// same five shapes, diffed against the same per-token `sdpa_cpu`
// oracle. fold=2 divides a3b's heads_per_kv=8. Dispatched via the vB
// staging path (vb=true) since vA (direct-device) has no GQA fold yet.

#[test]
#[ignore = "long-running GPU test"]
fn sdpa_gqa2_causal_flash_n1_single_block() {
    flash_diff_tokenwise(1, 63, 0x5DA0_F1A5_0000_0001, 2, true);
}

#[test]
#[ignore = "long-running GPU test"]
fn sdpa_gqa2_causal_flash_n1_multi_block() {
    flash_diff_tokenwise(1, 4999, 0x5DA0_F1A5_0000_0002, 2, true);
}

#[test]
#[ignore = "long-running GPU test"]
fn sdpa_gqa2_causal_flash_n4_tokenwise() {
    flash_diff_tokenwise(4, 4, 0x5DA0_F1A5_0000_0003, 2, true);
}

#[test]
#[ignore = "long-running GPU test"]
fn sdpa_gqa2_causal_flash_m512_square_causal() {
    flash_diff_tokenwise(512, 0, 0x5DA0_F1A5_0000_0004, 2, true);
}

#[test]
#[ignore = "long-running GPU test"]
fn sdpa_gqa2_causal_flash_m1500_deep_chunk() {
    flash_diff_tokenwise(1500, 4096, 0x5DA0_F1A5_0000_0005, 2, true);
}

// ---------------------------------------------------------------------------
// vB staging kernel (`attn_sdpa_causal_flash_vb`) — the former production
// kernel, now kept as the experimental slot. Same five shapes diffed
// against the same `sdpa_cpu` oracle, identical `COSINE_FLOOR = 0.9999`.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "long-running GPU test"]
fn sdpa_vb_causal_flash_n1_single_block() {
    flash_diff_tokenwise(1, 63, 0x5DA0_F1A5_0000_0001, 1, true);
}

#[test]
#[ignore = "long-running GPU test"]
fn sdpa_vb_causal_flash_n1_multi_block() {
    flash_diff_tokenwise(1, 4999, 0x5DA0_F1A5_0000_0002, 1, true);
}

#[test]
#[ignore = "long-running GPU test"]
fn sdpa_vb_causal_flash_n4_tokenwise() {
    flash_diff_tokenwise(4, 4, 0x5DA0_F1A5_0000_0003, 1, true);
}

#[test]
#[ignore = "long-running GPU test"]
fn sdpa_vb_causal_flash_m512_square_causal() {
    flash_diff_tokenwise(512, 0, 0x5DA0_F1A5_0000_0004, 1, true);
}

#[test]
#[ignore = "long-running GPU test"]
fn sdpa_vb_causal_flash_m1500_deep_chunk() {
    flash_diff_tokenwise(1500, 4096, 0x5DA0_F1A5_0000_0005, 1, true);
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
    let mut metal = MetalContext::new().expect("open Metal");
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

    // Pack every bucket's expert blob into one `expert_base` buffer at
    // uniform `expert_size` stride — the layout both the gather GEMM
    // and the per-bucket fallback index. `expert_slots[bi] = bi`;
    // `expert_indices` expands that per assignment row.
    let num_buckets = buckets.expert_ids.len();
    let expert_size = v.expert_size_4bit();
    let mut expert_base_host = vec![0u8; num_buckets * expert_size];
    for (bi, &e) in buckets.expert_ids.iter().enumerate() {
        let blob = &synth_blobs[expert_to_blob_idx[&e]];
        expert_base_host[bi * expert_size..(bi + 1) * expert_size]
            .copy_from_slice(blob);
    }
    let expert_base =
        MtlBuffer::<u8>::with_data(metal.device(), &expert_base_host);
    let expert_slots: Vec<u32> = (0..num_buckets as u32).collect();
    let mut expert_indices_host = vec![0u32; total_assignments];
    for bi in 0..num_buckets {
        let start = buckets.offsets[bi] as usize;
        let end = buckets.offsets[bi + 1] as usize;
        expert_indices_host[start..end].fill(bi as u32);
    }
    let expert_indices_buf = make_buf::<u32>(&metal, total_assignments);
    write_buf(&expert_indices_buf, &expert_indices_host);

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

    // Exercise both encode paths: the MLX gather GEMM (`gather =
    // true`) and the per-bucket matvec fallback (`gather = false`).
    // Both must match the tokenwise reference.
    for &gather in &[true, false] {
        write_buf(&out_sum_buf, &vec![0.0f32; n_tokens * h]);
        let cmdbuf = metal.queue().new_command_buffer();
        encode_moe_batched_permute_fuse(
            cmdbuf,
            &matvec_pipes,
            metal.kernels(),
            &swiglu,
            &bucket_accumulate,
            expert_base.buffer(),
            expert_size as u64,
            &expert_indices_buf,
            &expert_slots,
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
            gather,
        );
        cmdbuf.commit();
        cmdbuf.wait_until_completed();

        let gpu_out = read_buf_f32(&out_sum_buf, n_tokens * h);
        assert!(
            gpu_out.iter().all(|x| x.is_finite()),
            "permute-fuse output (gather={gather}) has non-finite values"
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
                "[diff:moe_permute_fuse gather={}] token={} cosine={:.9} max_abs={:.3e}",
                gather, t, cos, max_abs
            );
            assert!(
                cos >= COSINE_FLOOR,
                "gather={} token {} cosine {:.9} below floor {}",
                gather,
                t,
                cos,
                COSINE_FLOOR
            );
        }
    }

    // ---- Op-level: run `Op::MoeBatchedPermuteFuse` through the
    // `CpuBackend` executor (the diff oracle) and check it matches the
    // same tokenwise reference. Exercises the executor arm's
    // `expert_base` per-bucket slicing; the GPU Op arm is covered
    // end-to-end by the `eval_prompt_matches_per_token_oracle` canary.
    {
        let f32_sz = std::mem::size_of::<f32>();
        let mut cpu = CpuBackend::new(dummy_weight_file("moe_pf"));
        let pool = cpu.pool_mut();
        let eb = pool
            .alloc(expert_base_host.len(), "expert_base", false)
            .unwrap();
        let ei = pool
            .alloc(total_assignments * 4, "expert_indices", false)
            .unwrap();
        let bin = pool
            .alloc(total_assignments * h * f32_sz, "bucket_input", false)
            .unwrap();
        let bg = pool
            .alloc(total_assignments * mi * f32_sz, "bucket_gate", false)
            .unwrap();
        let bu = pool
            .alloc(total_assignments * mi * f32_sz, "bucket_up", false)
            .unwrap();
        let ba = pool
            .alloc(total_assignments * mi * f32_sz, "bucket_act", false)
            .unwrap();
        let bo = pool
            .alloc(total_assignments * h * f32_sz, "bucket_out", false)
            .unwrap();
        let bti = pool
            .alloc(total_assignments * 4, "bucket_token_idx", false)
            .unwrap();
        let bw = pool
            .alloc(total_assignments * f32_sz, "bucket_weights", false)
            .unwrap();
        let os = pool
            .alloc(n_tokens * h * f32_sz, "out_sum", false)
            .unwrap();
        pool.upload(eb, &expert_base_host).unwrap();
        pool.upload(ei, as_u8(&expert_indices_host)).unwrap();
        pool.upload(bin, as_u8(&packed_input)).unwrap();
        pool.upload(bti, as_u8(&buckets.token_idx)).unwrap();
        pool.upload(bw, as_u8(&buckets.weights)).unwrap();
        // `out_sum` is zero from `alloc`; the per-bucket scatter adds.

        let mut g = Graph::new();
        g.push(Op::MoeBatchedPermuteFuse {
            label: "test_moe_pf_cpu",
            expert_base: eb,
            expert_stride: expert_size as u64,
            expert_indices: ei,
            expert_slots: expert_slots.clone(),
            bucket_input: bin,
            bucket_gate: bg,
            bucket_up: bu,
            bucket_act: ba,
            bucket_out: bo,
            bucket_token_idx: bti,
            bucket_weights: bw,
            out_sum: os,
            buckets: buckets.clone(),
        });
        cpu.execute(&g, "moe_pf_op_diff").expect("CpuBackend execute");

        let mut out_bytes = vec![0u8; n_tokens * h * f32_sz];
        cpu.pool().download(os, &mut out_bytes).unwrap();
        let cpu_out: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!(
            cpu_out.iter().all(|x| x.is_finite()),
            "CpuBackend MoE Op output has non-finite values"
        );
        for t in 0..n_tokens {
            let c = &cpu_out[t * h..(t + 1) * h];
            let r = &per_token_ref[t * h..(t + 1) * h];
            let cos = cosine_sim(c, r);
            eprintln!(
                "[diff:moe_permute_fuse cpu-op] token={} cosine={:.9}",
                t, cos
            );
            assert!(
                cos >= COSINE_FLOOR,
                "cpu-op token {} cosine {:.9} below floor {}",
                t, cos, COSINE_FLOOR
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Phase A (session 6): GPU MoE router vs CPU oracle.
// ---------------------------------------------------------------------------

/// Per-token GPU `encode_moe_router` against `moe_router_cpu`.
///
/// Both sides start from the same f32 logits tensor `[n_tokens, n_experts]`.
/// The CPU oracle runs softmax → selection-sort top-K → divide-by-sum
/// normalize per token, the GPU does the same on-device.
///
/// Comparison policy (mirrors `moe_router_cpu_close_c_vs_rust` in
/// `diff_oracle.rs`):
///   - **Indices**: bit-exact as a *set* per token (sort ascending, compare).
///     Slot order can drift if two probs collide within ULP, but for
///     random uniform-ish inputs the magnitude separation dominates.
///   - **Weights**: cosine ≥ COSINE_FLOOR per token after aligning by
///     index (gather GPU weights at CPU's index order). ULP drift comes
///     from the softmax reduction order (tree on GPU vs scan on CPU).
fn run_moe_router_diff(n_tokens: usize, n_experts: usize, k: usize, seed: u64) {
    let mut rng = XorShift64::new(seed);

    // Logits: random in (-2, 2) per element. Large enough magnitude that
    // softmax has real selection pressure (rather than near-uniform output).
    let logits_f32: Vec<f32> = (0..(n_tokens * n_experts))
        .map(|_| rng.next_f32() * 2.0)
        .collect();

    // --- CPU oracle ---
    let mut cpu_indices: Vec<i32> = vec![0; n_tokens * k];
    let mut cpu_weights: Vec<f32> = vec![0.0; n_tokens * k];
    for t in 0..n_tokens {
        let mut scores = logits_f32[t * n_experts..(t + 1) * n_experts].to_vec();
        moe_router_cpu(
            &mut scores,
            k,
            &mut cpu_indices[t * k..(t + 1) * k],
            &mut cpu_weights[t * k..(t + 1) * k],
        )
        .expect("moe_router_cpu");
    }

    // --- GPU ---
    let mut metal = MetalContext::new().expect("MetalContext::new");
    let pipes = MoeRouterPipelines::fetch(&mut metal).expect("router pipes");

    let logits_buf = make_buf::<f32>(&metal, n_tokens * n_experts);
    write_buf(&logits_buf, &logits_f32);

    let indices_buf = make_buf::<i32>(&metal, n_tokens * k);
    let weights_buf = make_buf::<f32>(&metal, n_tokens * k);

    let queue = metal.queue_clone();
    let cmdbuf = queue.new_command_buffer();
    encode_moe_router(
        cmdbuf,
        &pipes,
        &logits_buf,
        &indices_buf,
        &weights_buf,
        n_tokens as u32,
        n_experts as u32,
        k as u32,
    );
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    let gpu_indices: Vec<i32> = {
        let mut v = vec![0i32; n_tokens * k];
        unsafe {
            std::ptr::copy_nonoverlapping(
                indices_buf.contents() as *const i32,
                v.as_mut_ptr(),
                v.len(),
            );
        }
        v
    };
    let gpu_weights = read_buf_f32(&weights_buf, n_tokens * k);

    // Compare per token.
    let mut min_cos = f32::INFINITY;
    let mut max_abs_w: f32 = 0.0;
    let mut slot_matches = 0usize;
    for t in 0..n_tokens {
        let ci = &cpu_indices[t * k..(t + 1) * k];
        let gi = &gpu_indices[t * k..(t + 1) * k];
        let cw = &cpu_weights[t * k..(t + 1) * k];
        let gw = &gpu_weights[t * k..(t + 1) * k];

        // Set equality after sort.
        let mut ci_sorted = ci.to_vec();
        let mut gi_sorted = gi.to_vec();
        ci_sorted.sort();
        gi_sorted.sort();
        assert_eq!(
            gi_sorted, ci_sorted,
            "token {} index set mismatch: gpu={:?} cpu={:?}",
            t, gi_sorted, ci_sorted
        );

        if gi == ci {
            slot_matches += 1;
        }

        // Gather GPU weights at CPU's index order (so we compare same-expert
        // weights) — pick out each CPU slot's expert from the GPU output.
        let mut gw_aligned = vec![0.0f32; k];
        for (cs, &cpu_e) in ci.iter().enumerate() {
            let gs = gi.iter().position(|&e| e == cpu_e).unwrap();
            gw_aligned[cs] = gw[gs];
        }

        let cos = cosine_sim(&gw_aligned, cw);
        let max_abs = gw_aligned
            .iter()
            .zip(cw.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        min_cos = min_cos.min(cos);
        max_abs_w = max_abs_w.max(max_abs);
        assert!(
            cos >= COSINE_FLOOR,
            "token {} weight cosine {:.9} below floor {} (max_abs={:.3e})",
            t,
            cos,
            COSINE_FLOOR,
            max_abs
        );
    }

    eprintln!(
        "[diff:moe_router_gpu] N={} E={} K={}: slot-match {}/{}, min_cos={:.9}, max_abs_w={:.3e}",
        n_tokens, n_experts, k, slot_matches, n_tokens, min_cos, max_abs_w
    );
}

/// Qwen3.6-A3B shape: 256 experts, K=8. Single token (decode N=1).
#[test]
#[ignore = "long-running GPU test"]
fn moe_router_gpu_matches_cpu_a3b_n1() {
    run_moe_router_diff(1, 256, 8, 0xA3B_0001);
}

/// Qwen3.6-A3B shape, small batch (N=8 — sub-chunk).
#[test]
#[ignore = "long-running GPU test"]
fn moe_router_gpu_matches_cpu_a3b_n8() {
    run_moe_router_diff(8, 256, 8, 0xA3B_0008);
}

/// Qwen3.6-A3B shape, mid batch (N=256).
#[test]
#[ignore = "long-running GPU test"]
fn moe_router_gpu_matches_cpu_a3b_n256() {
    run_moe_router_diff(256, 256, 8, 0xA3B_0256);
}

/// Smaller variant shape: 128 experts, K=8 (Qwen2-style for breadth).
#[test]
#[ignore = "long-running GPU test"]
fn moe_router_gpu_matches_cpu_e128_k8() {
    run_moe_router_diff(64, 128, 8, 0xE128_0064);
}

/// Large-K stress: 512 experts (kernel cap), K=10 (Qwen3.5-A17B shape).
#[test]
#[ignore = "long-running GPU test"]
fn moe_router_gpu_matches_cpu_e512_k10() {
    run_moe_router_diff(32, 512, 10, 0xE512_000A);
}

// ---------------------------------------------------------------------------
// Phase B-0a/B-0b (session 6): batched rms_norm + residual_add.
// ---------------------------------------------------------------------------

/// Reference CPU implementation of bf16-weighted RMSNorm, matching the
/// fused GPU kernel's behavior for a single token. Self-contained so the
/// diff test doesn't depend on `WeightFile` plumbing.
fn rms_norm_bf16_ref(x: &[f32], weight_bf16: &[u16], eps: f32, out: &mut [f32]) {
    let dim = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let inv_rms = 1.0f32 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        // bf16 → f32: shift left 16 bits into the high half of the f32.
        let w_u32 = (weight_bf16[i] as u32) << 16;
        let w = f32::from_bits(w_u32);
        out[i] = x[i] * inv_rms * w;
    }
}

#[test]
#[ignore = "long-running GPU test"]
fn rms_norm_bf16_fused_n_tokens_matches_cpu() {
    let n_tokens: usize = 16;
    let dim: usize = 2048;
    let eps: f32 = 1e-6;

    let mut rng = XorShift64::new(0xB0_FE_DD_01);
    let x: Vec<f32> = (0..(n_tokens * dim))
        .map(|_| rng.next_f32() * 0.5)
        .collect();
    let weight_bf16: Vec<u16> =
        (0..dim).map(|_| f32_to_bf16(rng.next_f32() * 0.1)).collect();

    // CPU reference: per-token rms_norm.
    let mut cpu_out = vec![0.0f32; n_tokens * dim];
    for t in 0..n_tokens {
        rms_norm_bf16_ref(
            &x[t * dim..(t + 1) * dim],
            &weight_bf16,
            eps,
            &mut cpu_out[t * dim..(t + 1) * dim],
        );
    }

    // GPU: single fused dispatch.
    let mut metal = MetalContext::new().expect("MetalContext::new");
    let pipe = RmsNormBf16FusedNTokensPipeline::fetch(&mut metal)
        .expect("rms_norm fused pipe");

    let x_buf = make_buf::<f32>(&metal, n_tokens * dim);
    write_buf(&x_buf, &x);
    let w_buf = make_buf::<u16>(&metal, dim);
    write_buf(&w_buf, &weight_bf16);
    let out_buf = make_buf::<f32>(&metal, n_tokens * dim);

    let queue = metal.queue_clone();
    let cmdbuf = queue.new_command_buffer();
    encode_rms_norm_bf16_fused_n_tokens(
        cmdbuf,
        &pipe,
        &x_buf,
        &w_buf,
        0,
        &out_buf,
        dim as u32,
        n_tokens as u32,
        eps,
    );
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    let gpu_out = read_buf_f32(&out_buf, n_tokens * dim);

    let mut min_cos = f32::INFINITY;
    let mut max_abs: f32 = 0.0;
    for t in 0..n_tokens {
        let g = &gpu_out[t * dim..(t + 1) * dim];
        let c = &cpu_out[t * dim..(t + 1) * dim];
        let cos = cosine_sim(g, c);
        let m = g
            .iter()
            .zip(c.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        min_cos = min_cos.min(cos);
        max_abs = max_abs.max(m);
        assert!(
            cos >= COSINE_FLOOR,
            "token {} cosine {:.9} below floor {} (max_abs={:.3e})",
            t, cos, COSINE_FLOOR, m
        );
    }
    eprintln!(
        "[diff:rms_norm_bf16_fused_n_tokens] N={} dim={}: min_cos={:.9} max_abs={:.3e}",
        n_tokens, dim, min_cos, max_abs
    );
}

#[test]
#[ignore = "long-running GPU test"]
fn residual_add_n_tokens_matches_cpu() {
    let n_tokens: usize = 32;
    let dim: usize = 2048;

    let mut rng = XorShift64::new(0xB0_FE_DD_02);
    let a: Vec<f32> = (0..(n_tokens * dim))
        .map(|_| rng.next_f32())
        .collect();
    let b: Vec<f32> = (0..(n_tokens * dim))
        .map(|_| rng.next_f32())
        .collect();

    // CPU ref: element-wise add.
    let cpu_out: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    let metal = MetalContext::new().expect("MetalContext::new");
    let mut metal = metal;
    let pso = metal
        .pipeline("residual_add_n_tokens")
        .expect("residual_add_n_tokens pso")
        .clone();

    let a_buf = make_buf::<f32>(&metal, n_tokens * dim);
    write_buf(&a_buf, &a);
    let b_buf = make_buf::<f32>(&metal, n_tokens * dim);
    write_buf(&b_buf, &b);
    let out_buf = make_buf::<f32>(&metal, n_tokens * dim);

    let queue = metal.queue_clone();
    let cmdbuf = queue.new_command_buffer();
    encode_residual_add_n_tokens_into(
        cmdbuf,
        &pso,
        &a_buf,
        &b_buf,
        &out_buf,
        n_tokens as u32,
        dim as u32,
    );
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    let gpu_out = read_buf_f32(&out_buf, n_tokens * dim);

    // Element-wise add is bit-exact between CPU and GPU on the same inputs.
    let max_abs: f32 = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs == 0.0,
        "residual_add_n_tokens not bit-exact: max_abs={:.3e}",
        max_abs
    );
    eprintln!(
        "[diff:residual_add_n_tokens] N={} dim={}: max_abs={:.3e}",
        n_tokens, dim, max_abs
    );
}
