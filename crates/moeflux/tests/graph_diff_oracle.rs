//! Load-bearing acceptance gate for the typed-Op graph compiler
//! (session 7 S7-4).
//!
//! Builds synthetic [`Graph`]s and runs them through both
//! [`CpuBackend`] and [`MetalBackend`], comparing per-Op output
//! buffers via cosine similarity. The check `cosine ≥ 0.9999` is
//! the same floor used by `batched_diff_oracle.rs`'s kernel-level
//! tests and is what we require of every Op variant.
//!
//! ## Scope
//!
//! Exercises the 8 Op variants that are wired in both backends per
//! S7-3:
//! - ResidualAddNTokens (weight-free)
//! - RmsNormBf16NTokens (bf16 weight from synthetic file)
//! - MatvecNTokens (4-bit quantized weight)
//! - SwigluFusedBatched (weight-free)
//! - MoeSoftmaxTopK (weight-free)
//! - MoeNormalizeWeights (in-place, weight-free)
//! - MoeCombineResidualNTokens (weight-free)
//! - MatvecNTokens 8-bit variant (8-bit quantized weight)
//!
//! The remaining 7 variants (RmsNormQkNTokens, SdpaCausalTiled,
//! MoeBatchedPermuteFuse, Conv1dStepNTokens, ComputeDecayBetaNTokens,
//! GatedDeltaNetStepNTokens, GatedRmsNormNTokens, LmHead) wire in
//! later checkpoints alongside their producers, and are not
//! exercised here.
//!
//! ## Synthetic weight file
//!
//! The weight-touching tests build a small `.bin` + `.json` fixture
//! in a tempdir per-test, containing the minimum tensors needed by
//! the Ops under test. Bytes are deterministic (seeded) so failures
//! are reproducible.

#![cfg(feature = "model-qwen3-6-35b-a3b")]

use std::fs;
use std::path::PathBuf;

use moeflux::riir::graph::{
    Backend, BufferPool, CpuBackend, Graph, MetalBackend, Op,
};
use moeflux::riir::metal::MetalContext;
use moeflux::riir::mtl_weight_buf::MtlWeightBuf;
use moeflux::riir::weight_file::WeightFile;

/// Cosine-similarity floor for Op-level diffs. Same as
/// `batched_diff_oracle.rs`'s COSINE_FLOOR.
const COSINE_FLOOR: f32 = 0.9999;

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Deterministic PRNG modeled on xorshift64. Bit-stable across
/// platforms.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f32_unit(&mut self) -> f32 {
        // Uniform in [-1, 1].
        let bits = (self.next_u64() >> 32) as u32;
        let normalized = (bits as f32) / (u32::MAX as f32);
        normalized * 2.0 - 1.0
    }
}

/// Write a minimal `.bin` + `.json` fixture to a per-test tempdir
/// and return `(WeightFile, tempdir_path)`. The caller keeps the
/// tempdir alive until tests are done (drops on RAII).
struct SyntheticWf {
    wf: Option<WeightFile>,
    dir: PathBuf,
}
impl SyntheticWf {
    /// Build a fixture containing exactly the tensors named in
    /// `tensors`, where each entry is
    /// `(name, dtype, bits, shape, bytes)`. Bytes are concatenated
    /// in order with no padding; manifest offsets are computed
    /// automatically.
    fn build(test_name: &str, tensors: &[(&str, &str, i32, Vec<usize>, Vec<u8>)]) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "moeflux-s7-{}-{}",
            test_name,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("mkdir tempdir");
        let bin_path = dir.join("model_weights.bin");
        let json_path = dir.join("model_weights.json");

        let mut blob: Vec<u8> = Vec::new();
        let mut manifest = serde_json::json!({ "tensors": {} });
        let tensors_obj = manifest["tensors"].as_object_mut().unwrap();
        for (name, dtype, bits, shape, bytes) in tensors.iter() {
            let offset = blob.len();
            blob.extend_from_slice(bytes);
            let entry = serde_json::json!({
                "offset": offset,
                "size": bytes.len(),
                "shape": shape,
                "dtype": dtype,
                "bits": *bits,
            });
            tensors_obj.insert(name.to_string(), entry);
        }
        fs::write(&bin_path, &blob).expect("write .bin");
        fs::write(&json_path, manifest.to_string()).expect("write .json");
        let wf = WeightFile::open(&bin_path, &json_path).expect("open WF");
        Self { wf: Some(wf), dir }
    }
    fn take(&mut self) -> WeightFile {
        self.wf.take().expect("WF already taken")
    }
}
impl Drop for SyntheticWf {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn bytes_of_f32(v: &[f32]) -> &[u8] {
    let (head, body, tail) = unsafe { v.align_to::<u8>() };
    assert!(head.is_empty() && tail.is_empty());
    body
}

fn f32_of_bytes(v: &[u8]) -> Vec<f32> {
    assert_eq!(v.len() % 4, 0);
    let mut out = vec![0.0f32; v.len() / 4];
    let (head, body, tail) = unsafe { out.align_to_mut::<u8>() };
    assert!(head.is_empty() && tail.is_empty());
    body.copy_from_slice(v);
    out
}

/// Smallest plausible WeightFile: one tiny bf16 tensor. Used by
/// tests that don't actually use weights but need to construct
/// the Backend (CpuBackend / MetalBackend require a real WF for
/// their wf field / MtlWeightBuf wrap).
fn dummy_weight_file(test_name: &str) -> SyntheticWf {
    let dummy_bf16_bytes = vec![0u8; 64]; // 32 bf16 elements
    SyntheticWf::build(
        test_name,
        &[(
            "dummy.weight",
            "BF16",
            0,
            vec![32],
            dummy_bf16_bytes,
        )],
    )
}

// ============================================================================
// Test: ResidualAddNTokens through both backends
// ============================================================================

#[test]
#[ignore = "long-running GPU test"]
fn graph_metal_matches_cpu_residual_add() {
    let n_tokens: u32 = 8;
    let dim: u32 = 64;
    let total = (n_tokens * dim) as usize;
    let mut rng = Rng::new(0xA3B_AAAA);
    let a_data: Vec<f32> = (0..total).map(|_| rng.next_f32_unit()).collect();
    let b_data: Vec<f32> = (0..total).map(|_| rng.next_f32_unit()).collect();

    // CPU path
    let mut cpu_wf = dummy_weight_file("residual_add_cpu");
    let mut cpu = CpuBackend::new(cpu_wf.take());
    let cpu_a = cpu.pool_mut().alloc(total * 4, "a", false).unwrap();
    let cpu_b = cpu.pool_mut().alloc(total * 4, "b", false).unwrap();
    let cpu_out = cpu.pool_mut().alloc(total * 4, "out", false).unwrap();
    cpu.pool_mut().upload(cpu_a, bytes_of_f32(&a_data)).unwrap();
    cpu.pool_mut().upload(cpu_b, bytes_of_f32(&b_data)).unwrap();
    let mut g_cpu = Graph::new();
    g_cpu.push(Op::ResidualAddNTokens {
        label: "test_residual",
        a: cpu_a,
        b: cpu_b,
        out: cpu_out,
        n_tokens,
        dim,
    });
    cpu.execute(&g_cpu).unwrap();
    let mut cpu_out_bytes = vec![0u8; total * 4];
    cpu.pool().download(cpu_out, &mut cpu_out_bytes).unwrap();
    let cpu_out_f32 = f32_of_bytes(&cpu_out_bytes);

    // GPU path
    let mut gpu_wf = dummy_weight_file("residual_add_gpu");
    let metal_ctx = MetalContext::new().expect("MetalContext::new");
    let wf_ref = gpu_wf.wf.as_ref().unwrap();
    let wf_buf = MtlWeightBuf::wrap(wf_ref, metal_ctx.device());
    let _ = gpu_wf.take(); // give ownership to backend below
    let mut gpu = MetalBackend::new(metal_ctx, wf_buf).expect("MetalBackend::new");
    let gpu_a = gpu.pool_mut().alloc(total * 4, "a", false).unwrap();
    let gpu_b = gpu.pool_mut().alloc(total * 4, "b", false).unwrap();
    let gpu_out = gpu.pool_mut().alloc(total * 4, "out", false).unwrap();
    gpu.pool_mut().upload(gpu_a, bytes_of_f32(&a_data)).unwrap();
    gpu.pool_mut().upload(gpu_b, bytes_of_f32(&b_data)).unwrap();
    let mut g_gpu = Graph::new();
    g_gpu.push(Op::ResidualAddNTokens {
        label: "test_residual",
        a: gpu_a,
        b: gpu_b,
        out: gpu_out,
        n_tokens,
        dim,
    });
    gpu.execute(&g_gpu).unwrap();
    let mut gpu_out_bytes = vec![0u8; total * 4];
    gpu.pool().download(gpu_out, &mut gpu_out_bytes).unwrap();
    let gpu_out_f32 = f32_of_bytes(&gpu_out_bytes);

    // Compare
    let cos = cosine_sim(&cpu_out_f32, &gpu_out_f32);
    let max_abs = cpu_out_f32
        .iter()
        .zip(gpu_out_f32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[s7-4 residual_add] N={} dim={}: cos={:.9}, max_abs={:.3e}",
        n_tokens, dim, cos, max_abs
    );
    assert!(
        cos >= COSINE_FLOOR,
        "cosine {} below floor {} (max_abs={})",
        cos,
        COSINE_FLOOR,
        max_abs
    );
}

// ============================================================================
// Test: SwigluFusedBatched through both backends
// ============================================================================

#[test]
#[ignore = "long-running GPU test"]
fn graph_metal_matches_cpu_swiglu() {
    let total: u32 = 1024;
    let mut rng = Rng::new(0xA3B_5555);
    let gate_data: Vec<f32> = (0..total as usize)
        .map(|_| rng.next_f32_unit() * 2.0)
        .collect();
    let up_data: Vec<f32> = (0..total as usize)
        .map(|_| rng.next_f32_unit() * 2.0)
        .collect();

    // CPU
    let mut cpu_wf = dummy_weight_file("swiglu_cpu");
    let mut cpu = CpuBackend::new(cpu_wf.take());
    let cpu_gate = cpu.pool_mut().alloc(total as usize * 4, "gate", false).unwrap();
    let cpu_up = cpu.pool_mut().alloc(total as usize * 4, "up", false).unwrap();
    let cpu_out = cpu.pool_mut().alloc(total as usize * 4, "out", false).unwrap();
    cpu.pool_mut().upload(cpu_gate, bytes_of_f32(&gate_data)).unwrap();
    cpu.pool_mut().upload(cpu_up, bytes_of_f32(&up_data)).unwrap();
    let mut g_cpu = Graph::new();
    g_cpu.push(Op::SwigluFusedBatched {
        label: "test_swiglu",
        gate: cpu_gate,
        up: cpu_up,
        out: cpu_out,
        total,
    });
    cpu.execute(&g_cpu).unwrap();
    let mut cpu_out_bytes = vec![0u8; total as usize * 4];
    cpu.pool().download(cpu_out, &mut cpu_out_bytes).unwrap();
    let cpu_out_f32 = f32_of_bytes(&cpu_out_bytes);

    // GPU
    let mut gpu_wf = dummy_weight_file("swiglu_gpu");
    let metal_ctx = MetalContext::new().expect("MetalContext::new");
    let wf_ref = gpu_wf.wf.as_ref().unwrap();
    let wf_buf = MtlWeightBuf::wrap(wf_ref, metal_ctx.device());
    let _ = gpu_wf.take();
    let mut gpu = MetalBackend::new(metal_ctx, wf_buf).expect("MetalBackend::new");
    let gpu_gate = gpu.pool_mut().alloc(total as usize * 4, "gate", false).unwrap();
    let gpu_up = gpu.pool_mut().alloc(total as usize * 4, "up", false).unwrap();
    let gpu_out = gpu.pool_mut().alloc(total as usize * 4, "out", false).unwrap();
    gpu.pool_mut().upload(gpu_gate, bytes_of_f32(&gate_data)).unwrap();
    gpu.pool_mut().upload(gpu_up, bytes_of_f32(&up_data)).unwrap();
    let mut g_gpu = Graph::new();
    g_gpu.push(Op::SwigluFusedBatched {
        label: "test_swiglu",
        gate: gpu_gate,
        up: gpu_up,
        out: gpu_out,
        total,
    });
    gpu.execute(&g_gpu).unwrap();
    let mut gpu_out_bytes = vec![0u8; total as usize * 4];
    gpu.pool().download(gpu_out, &mut gpu_out_bytes).unwrap();
    let gpu_out_f32 = f32_of_bytes(&gpu_out_bytes);

    let cos = cosine_sim(&cpu_out_f32, &gpu_out_f32);
    let max_abs = cpu_out_f32
        .iter()
        .zip(gpu_out_f32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[s7-4 swiglu] total={}: cos={:.9}, max_abs={:.3e}",
        total, cos, max_abs
    );
    assert!(
        cos >= COSINE_FLOOR,
        "cosine {} below floor {} (max_abs={})",
        cos,
        COSINE_FLOOR,
        max_abs
    );
}

// ============================================================================
// Test: MoeSoftmaxTopK + MoeNormalizeWeights through both backends
// ============================================================================

#[test]
#[ignore = "long-running GPU test"]
fn graph_metal_matches_cpu_moe_router_normalize() {
    let n_tokens: u32 = 4;
    let n_experts: u32 = 32;
    let k: u32 = 4;
    let mut rng = Rng::new(0xA3B_F00D);
    let logits_data: Vec<f32> = (0..(n_tokens * n_experts) as usize)
        .map(|_| rng.next_f32_unit() * 3.0)
        .collect();

    // CPU
    let mut cpu_wf = dummy_weight_file("router_cpu");
    let mut cpu = CpuBackend::new(cpu_wf.take());
    let cpu_l = cpu.pool_mut().alloc(logits_data.len() * 4, "logits", false).unwrap();
    let cpu_i = cpu.pool_mut().alloc((n_tokens * k) as usize * 4, "idx", false).unwrap();
    let cpu_w_buf = cpu.pool_mut().alloc((n_tokens * k) as usize * 4, "w", false).unwrap();
    cpu.pool_mut().upload(cpu_l, bytes_of_f32(&logits_data)).unwrap();
    let mut g_cpu = Graph::new();
    g_cpu.push(Op::MoeSoftmaxTopK {
        label: "topk",
        logits: cpu_l,
        indices_out: cpu_i,
        weights_out: cpu_w_buf,
        n_tokens,
        n_experts,
        k,
    });
    g_cpu.push(Op::MoeNormalizeWeights {
        label: "norm",
        weights: cpu_w_buf,
        n_tokens,
        k,
    });
    cpu.execute(&g_cpu).unwrap();
    let mut cpu_ib = vec![0u8; (n_tokens * k) as usize * 4];
    let mut cpu_wb = vec![0u8; (n_tokens * k) as usize * 4];
    cpu.pool().download(cpu_i, &mut cpu_ib).unwrap();
    cpu.pool().download(cpu_w_buf, &mut cpu_wb).unwrap();
    let cpu_idx: Vec<i32> = cpu_ib
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let cpu_w = f32_of_bytes(&cpu_wb);

    // GPU
    let mut gpu_wf = dummy_weight_file("router_gpu");
    let metal_ctx = MetalContext::new().expect("MetalContext::new");
    let wf_buf = MtlWeightBuf::wrap(gpu_wf.wf.as_ref().unwrap(), metal_ctx.device());
    let _ = gpu_wf.take();
    let mut gpu = MetalBackend::new(metal_ctx, wf_buf).unwrap();
    let gpu_l = gpu.pool_mut().alloc(logits_data.len() * 4, "logits", false).unwrap();
    let gpu_i = gpu.pool_mut().alloc((n_tokens * k) as usize * 4, "idx", false).unwrap();
    let gpu_w_buf = gpu.pool_mut().alloc((n_tokens * k) as usize * 4, "w", false).unwrap();
    gpu.pool_mut().upload(gpu_l, bytes_of_f32(&logits_data)).unwrap();
    let mut g_gpu = Graph::new();
    g_gpu.push(Op::MoeSoftmaxTopK {
        label: "topk",
        logits: gpu_l,
        indices_out: gpu_i,
        weights_out: gpu_w_buf,
        n_tokens,
        n_experts,
        k,
    });
    g_gpu.push(Op::MoeNormalizeWeights {
        label: "norm",
        weights: gpu_w_buf,
        n_tokens,
        k,
    });
    gpu.execute(&g_gpu).unwrap();
    let mut gpu_ib = vec![0u8; (n_tokens * k) as usize * 4];
    let mut gpu_wb = vec![0u8; (n_tokens * k) as usize * 4];
    gpu.pool().download(gpu_i, &mut gpu_ib).unwrap();
    gpu.pool().download(gpu_w_buf, &mut gpu_wb).unwrap();
    let gpu_idx: Vec<i32> = gpu_ib
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let gpu_w = f32_of_bytes(&gpu_wb);
    eprintln!(
        "[s7-4 router/cpu] first indices: {:?}",
        &cpu_idx[..k as usize]
    );
    eprintln!(
        "[s7-4 router/gpu] first indices: {:?}",
        &gpu_idx[..k as usize]
    );

    // Verify top-K index sets match (order may differ slot-by-slot).
    for t in 0..n_tokens as usize {
        let mut ci = cpu_idx[t * k as usize..(t + 1) * k as usize].to_vec();
        let mut gi = gpu_idx[t * k as usize..(t + 1) * k as usize].to_vec();
        ci.sort();
        gi.sort();
        assert_eq!(gi, ci, "token {} top-K index set mismatch", t);
    }

    // Compare weights ALIGNED by index — gather GPU weights at CPU slot order.
    let mut cpu_w_aligned = cpu_w.clone();
    let mut gpu_w_aligned = vec![0.0f32; cpu_w.len()];
    for t in 0..n_tokens as usize {
        let ci = &cpu_idx[t * k as usize..(t + 1) * k as usize];
        let gi = &gpu_idx[t * k as usize..(t + 1) * k as usize];
        let gw = &gpu_w[t * k as usize..(t + 1) * k as usize];
        for (cs, &cpu_e) in ci.iter().enumerate() {
            let gs = gi.iter().position(|&e| e == cpu_e).unwrap();
            gpu_w_aligned[t * k as usize + cs] = gw[gs];
        }
    }
    let _ = &mut cpu_w_aligned;
    let cos = cosine_sim(&cpu_w_aligned, &gpu_w_aligned);
    let max_abs = cpu_w_aligned
        .iter()
        .zip(gpu_w_aligned.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[s7-4 router/norm] N={} E={} K={}: cos={:.9}, max_abs={:.3e}",
        n_tokens, n_experts, k, cos, max_abs
    );
    assert!(
        cos >= COSINE_FLOOR,
        "weights cosine {} below floor {} (max_abs={})",
        cos,
        COSINE_FLOOR,
        max_abs
    );
}

// ============================================================================
// Test: graph_metal_matches_cpu_colored — load-bearing S7-5 gate
// ============================================================================
//
// 10-op residual chain forces aliasing opportunities. Per-op:
//
//     tmp_0 = a + b
//     tmp_i = tmp_{i-1} + b   for i in 1..10
//
// `a` and `b` are non-colorable inputs (read but never written by an
// Op); `tmp_0..tmp_9` are all colorable. Adjacent tmps overlap at one
// op so greedy coloring produces 2 colors. Total BufIds = 12;
// physical_buffer_count after `commit_plan` = 2 (inputs) + 2 (colors)
// = 4.
//
// The test asserts:
// 1. cpu_out matches gpu_out at cosine ≥ COSINE_FLOOR (correctness).
// 2. Both pools have `physical_buffer_count() < 12` (aliasing happened).
// 3. CPU and Metal pools agree on `physical_buffer_count()`
//    (deterministic coloring).

#[test]
#[ignore = "long-running GPU test"]
fn graph_metal_matches_cpu_colored() {
    let n_tokens: u32 = 4;
    let dim: u32 = 32;
    let total = (n_tokens * dim) as usize;
    let n_intermediates: u32 = 10;
    let bufid_count: u32 = 2 + n_intermediates; // a, b, tmp_0..tmp_9
    let mut rng = Rng::new(0xA3B_C010_4ED);
    let a_data: Vec<f32> = (0..total).map(|_| rng.next_f32_unit()).collect();
    let b_data: Vec<f32> = (0..total).map(|_| rng.next_f32_unit()).collect();

    // Build a Graph against the given (pool, a, b, tmps) IDs.
    let build_graph =
        |a: moeflux::riir::graph::BufId,
         b: moeflux::riir::graph::BufId,
         tmps: &[moeflux::riir::graph::BufId]| {
            let mut g = Graph::new();
            // op 0: tmp_0 = a + b
            g.push(Op::ResidualAddNTokens {
                label: "tmp_0",
                a,
                b,
                out: tmps[0],
                n_tokens,
                dim,
            });
            // op i: tmp_i = tmp_{i-1} + b
            for i in 1..(n_intermediates as usize) {
                g.push(Op::ResidualAddNTokens {
                    label: "tmp_i",
                    a: tmps[i - 1],
                    b,
                    out: tmps[i],
                    n_tokens,
                    dim,
                });
            }
            g
        };

    // CPU path
    let mut cpu_wf = dummy_weight_file("colored_cpu");
    let mut cpu = CpuBackend::new(cpu_wf.take());
    let cpu_a = cpu.pool_mut().alloc(total * 4, "a", false).unwrap();
    let cpu_b = cpu.pool_mut().alloc(total * 4, "b", false).unwrap();
    let mut cpu_tmps = Vec::new();
    for _ in 0..n_intermediates {
        cpu_tmps.push(cpu.pool_mut().alloc(total * 4, "tmp", false).unwrap());
    }
    cpu.pool_mut().upload(cpu_a, bytes_of_f32(&a_data)).unwrap();
    cpu.pool_mut().upload(cpu_b, bytes_of_f32(&b_data)).unwrap();
    let g_cpu = build_graph(cpu_a, cpu_b, &cpu_tmps);
    // Apply coloring.
    cpu.pool_mut().commit_plan(&g_cpu);
    let cpu_phys = cpu.pool().physical_buffer_count();
    cpu.execute(&g_cpu).unwrap();
    let mut cpu_out_bytes = vec![0u8; total * 4];
    cpu.pool()
        .download(*cpu_tmps.last().unwrap(), &mut cpu_out_bytes)
        .unwrap();
    let cpu_out_f32 = f32_of_bytes(&cpu_out_bytes);

    // GPU path
    let mut gpu_wf = dummy_weight_file("colored_gpu");
    let metal_ctx = MetalContext::new().expect("MetalContext::new");
    let wf_ref = gpu_wf.wf.as_ref().unwrap();
    let wf_buf = MtlWeightBuf::wrap(wf_ref, metal_ctx.device());
    let _ = gpu_wf.take();
    let mut gpu = MetalBackend::new(metal_ctx, wf_buf).expect("MetalBackend::new");
    let gpu_a = gpu.pool_mut().alloc(total * 4, "a", false).unwrap();
    let gpu_b = gpu.pool_mut().alloc(total * 4, "b", false).unwrap();
    let mut gpu_tmps = Vec::new();
    for _ in 0..n_intermediates {
        gpu_tmps.push(gpu.pool_mut().alloc(total * 4, "tmp", false).unwrap());
    }
    gpu.pool_mut().upload(gpu_a, bytes_of_f32(&a_data)).unwrap();
    gpu.pool_mut().upload(gpu_b, bytes_of_f32(&b_data)).unwrap();
    let g_gpu = build_graph(gpu_a, gpu_b, &gpu_tmps);
    gpu.pool_mut().commit_plan(&g_gpu);
    let gpu_phys = gpu.pool().physical_buffer_count();
    gpu.execute(&g_gpu).unwrap();
    let mut gpu_out_bytes = vec![0u8; total * 4];
    gpu.pool()
        .download(*gpu_tmps.last().unwrap(), &mut gpu_out_bytes)
        .unwrap();
    let gpu_out_f32 = f32_of_bytes(&gpu_out_bytes);

    let cos = cosine_sim(&cpu_out_f32, &gpu_out_f32);
    let max_abs = cpu_out_f32
        .iter()
        .zip(gpu_out_f32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[s7-5 colored] N={} dim={} chain={} bufid_count={} cpu_phys={} gpu_phys={} cos={:.9} max_abs={:.3e}",
        n_tokens,
        dim,
        n_intermediates,
        bufid_count,
        cpu_phys,
        gpu_phys,
        cos,
        max_abs
    );

    // Correctness: CPU and Metal agree on the final tmp.
    assert!(
        cos >= COSINE_FLOOR,
        "cosine {} below floor {} (max_abs={})",
        cos,
        COSINE_FLOOR,
        max_abs
    );
    // Aliasing happened: both pools should be well under bufid_count.
    assert!(
        cpu_phys < bufid_count as usize,
        "CPU pool physical={} not less than bufid_count={}",
        cpu_phys,
        bufid_count
    );
    assert!(
        gpu_phys < bufid_count as usize,
        "GPU pool physical={} not less than bufid_count={}",
        gpu_phys,
        bufid_count
    );
    // Determinism: CPU and Metal pools should pick the same physical
    // layout because the coloring algorithm is deterministic and they
    // both see identical Graphs.
    assert_eq!(
        cpu_phys, gpu_phys,
        "CPU and Metal pools disagree on physical_buffer_count"
    );
}

// ============================================================================
// Test: RmsNormQkNTokens through both backends (S7-6b)
// ============================================================================

#[test]
#[ignore = "long-running GPU test"]
fn graph_metal_matches_cpu_rms_norm_qk() {
    let n_tokens: u32 = 4;
    let num_k_heads: u32 = 8;
    let key_dim: u32 = 32;
    // No gap between q and k (key_offset_per_token == q region size).
    let key_offset_per_token: u32 = num_k_heads * key_dim;
    let per_token_elems =
        (key_offset_per_token + num_k_heads * key_dim) as usize;
    let total = n_tokens as usize * per_token_elems;
    let mut rng = Rng::new(0xA3B_BB01);
    let x_data: Vec<f32> = (0..total).map(|_| rng.next_f32_unit()).collect();

    // CPU path
    let mut cpu_wf = dummy_weight_file("rms_norm_qk_cpu");
    let mut cpu = CpuBackend::new(cpu_wf.take());
    let cpu_x = cpu.pool_mut().alloc(total * 4, "x", false).unwrap();
    cpu.pool_mut().upload(cpu_x, bytes_of_f32(&x_data)).unwrap();
    let mut g_cpu = Graph::new();
    g_cpu.push(Op::RmsNormQkNTokens {
        label: "test_rms_norm_qk",
        x: cpu_x,
        num_k_heads,
        key_dim,
        key_offset_per_token,
        n_tokens,
    });
    cpu.execute(&g_cpu).unwrap();
    let mut cpu_out_bytes = vec![0u8; total * 4];
    cpu.pool().download(cpu_x, &mut cpu_out_bytes).unwrap();
    let cpu_out_f32 = f32_of_bytes(&cpu_out_bytes);

    // GPU path
    let mut gpu_wf = dummy_weight_file("rms_norm_qk_gpu");
    let metal_ctx = MetalContext::new().expect("MetalContext::new");
    let wf_ref = gpu_wf.wf.as_ref().unwrap();
    let wf_buf = MtlWeightBuf::wrap(wf_ref, metal_ctx.device());
    let _ = gpu_wf.take();
    let mut gpu = MetalBackend::new(metal_ctx, wf_buf).expect("MetalBackend::new");
    let gpu_x = gpu.pool_mut().alloc(total * 4, "x", false).unwrap();
    gpu.pool_mut().upload(gpu_x, bytes_of_f32(&x_data)).unwrap();
    let mut g_gpu = Graph::new();
    g_gpu.push(Op::RmsNormQkNTokens {
        label: "test_rms_norm_qk",
        x: gpu_x,
        num_k_heads,
        key_dim,
        key_offset_per_token,
        n_tokens,
    });
    gpu.execute(&g_gpu).unwrap();
    let mut gpu_out_bytes = vec![0u8; total * 4];
    gpu.pool().download(gpu_x, &mut gpu_out_bytes).unwrap();
    let gpu_out_f32 = f32_of_bytes(&gpu_out_bytes);

    // Compare
    let cos = cosine_sim(&cpu_out_f32, &gpu_out_f32);
    let max_abs = cpu_out_f32
        .iter()
        .zip(gpu_out_f32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[s7-6b rms_norm_qk] N={} h={} kd={} kop={}: cos={:.9}, max_abs={:.3e}",
        n_tokens, num_k_heads, key_dim, key_offset_per_token, cos, max_abs
    );
    assert!(
        cos >= COSINE_FLOOR,
        "cosine {} below floor {} (max_abs={})",
        cos,
        COSINE_FLOOR,
        max_abs
    );
}

// ============================================================================
// Test: GatedRmsNormNTokens through both backends (S7-6b)
// ============================================================================

#[test]
#[ignore = "long-running GPU test"]
fn graph_metal_matches_cpu_gated_rms_norm() {
    let n_tokens: u32 = 4;
    let num_v_heads: u32 = 8;
    let value_dim: u32 = 32;
    let eps: f32 = 1e-6;
    let per_token = (num_v_heads * value_dim) as usize;
    let total = n_tokens as usize * per_token;
    let mut rng = Rng::new(0xA3B_BB02);
    let values_data: Vec<f32> =
        (0..total).map(|_| rng.next_f32_unit()).collect();
    let z_data: Vec<f32> = (0..total).map(|_| rng.next_f32_unit()).collect();
    // Weight = value_dim bf16. Construct random finite bf16 around
    // ~1.0 (avoid NaN/inf exponents from raw random u16).
    let mut weight_bytes = vec![0u8; (value_dim as usize) * 2];
    for chunk in weight_bytes.chunks_exact_mut(2) {
        let w_f32 = rng.next_f32_unit() * 0.25 + 1.0;
        let bits = (w_f32.to_bits() >> 16) as u16;
        chunk[0] = bits.to_le_bytes()[0];
        chunk[1] = bits.to_le_bytes()[1];
    }
    let tensors = &[(
        "gated_rms_weight",
        "BF16",
        0,
        vec![value_dim as usize],
        weight_bytes.clone(),
    )];

    // CPU path
    let mut cpu_wf = SyntheticWf::build("gated_rms_norm_cpu", tensors);
    let weight_off = {
        let wf = cpu_wf.wf.as_ref().unwrap();
        wf.tensor_info("gated_rms_weight").unwrap().offset as u64
    };
    let mut cpu = CpuBackend::new(cpu_wf.take());
    let cpu_values = cpu.pool_mut().alloc(total * 4, "values", false).unwrap();
    let cpu_z = cpu.pool_mut().alloc(total * 4, "z", false).unwrap();
    let cpu_out = cpu.pool_mut().alloc(total * 4, "out", false).unwrap();
    cpu.pool_mut().upload(cpu_values, bytes_of_f32(&values_data)).unwrap();
    cpu.pool_mut().upload(cpu_z, bytes_of_f32(&z_data)).unwrap();
    let mut g_cpu = Graph::new();
    g_cpu.push(Op::GatedRmsNormNTokens {
        label: "test_gated_rms_norm",
        values: cpu_values,
        z: cpu_z,
        weight_off,
        output: cpu_out,
        num_v_heads,
        value_dim,
        n_tokens,
        eps,
    });
    cpu.execute(&g_cpu).unwrap();
    let mut cpu_out_bytes = vec![0u8; total * 4];
    cpu.pool().download(cpu_out, &mut cpu_out_bytes).unwrap();
    let cpu_out_f32 = f32_of_bytes(&cpu_out_bytes);

    // GPU path
    let mut gpu_wf = SyntheticWf::build("gated_rms_norm_gpu", tensors);
    let metal_ctx = MetalContext::new().expect("MetalContext::new");
    let wf_ref = gpu_wf.wf.as_ref().unwrap();
    let wf_buf = MtlWeightBuf::wrap(wf_ref, metal_ctx.device());
    let _ = gpu_wf.take();
    let mut gpu = MetalBackend::new(metal_ctx, wf_buf).expect("MetalBackend::new");
    let gpu_values = gpu.pool_mut().alloc(total * 4, "values", false).unwrap();
    let gpu_z = gpu.pool_mut().alloc(total * 4, "z", false).unwrap();
    let gpu_out = gpu.pool_mut().alloc(total * 4, "out", false).unwrap();
    gpu.pool_mut().upload(gpu_values, bytes_of_f32(&values_data)).unwrap();
    gpu.pool_mut().upload(gpu_z, bytes_of_f32(&z_data)).unwrap();
    let mut g_gpu = Graph::new();
    g_gpu.push(Op::GatedRmsNormNTokens {
        label: "test_gated_rms_norm",
        values: gpu_values,
        z: gpu_z,
        weight_off,
        output: gpu_out,
        num_v_heads,
        value_dim,
        n_tokens,
        eps,
    });
    gpu.execute(&g_gpu).unwrap();
    let mut gpu_out_bytes = vec![0u8; total * 4];
    gpu.pool().download(gpu_out, &mut gpu_out_bytes).unwrap();
    let gpu_out_f32 = f32_of_bytes(&gpu_out_bytes);

    let cos = cosine_sim(&cpu_out_f32, &gpu_out_f32);
    let max_abs = cpu_out_f32
        .iter()
        .zip(gpu_out_f32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[s7-6b gated_rms_norm] N={} h={} vd={}: cos={:.9}, max_abs={:.3e}",
        n_tokens, num_v_heads, value_dim, cos, max_abs
    );
    assert!(
        cos >= COSINE_FLOOR,
        "cosine {} below floor {} (max_abs={})",
        cos,
        COSINE_FLOOR,
        max_abs
    );
}
