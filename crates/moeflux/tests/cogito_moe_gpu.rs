//! Phase 2 — GPU MoE diff against the CPU oracle (`deepseek_moe_cpu`).
//!
//! Drives [`cogito_moe_gpu::cogito_moe_layer_forward_gpu`] on
//! Cogito-V2 layer 3 (the first MoE layer; layers 0-2 are dense MLP)
//! with a sin-pattern input and compares the output to
//! [`moe_cpu::deepseek_moe_cpu`].
//!
//! Tolerance:
//! - cosine ≥ 0.9999
//! - max(|gpu - cpu|) / max(|cpu|) ≤ 1e-3
//!
//! Run with:
//! ```text
//! cargo test -p moeflux --no-default-features \
//!     --features model-cogito-v2-671b --release \
//!     --test cogito_moe_gpu -- --ignored --nocapture
//! ```

#![cfg(all(target_os = "macos", feature = "model-cogito-v2-671b"))]

use std::path::Path;

use moeflux::riir::cogito_moe_gpu::{
    cogito_moe_layer_forward_gpu, SharedExpertBuffers,
};
use moeflux::riir::dense_mlp_gpu::DenseMlpPipelines;
use moeflux::riir::expert_forward::MoeBuffers;
use moeflux::riir::expert_io::ExpertFiles;
use moeflux::riir::gpu_matvec::BfMatvecPipelines;
use moeflux::riir::metal::MetalBackend;
use moeflux::riir::moe_cpu::deepseek_moe_cpu;
use moeflux::riir::mtl_weight_buf::MtlWeightBuf;
use moeflux::riir::variants::VARIANT;
use moeflux::riir::weight_file::WeightFile;

const ROOT: &str =
    "/Volumes/Temp Backup/models/blallama/cogito-v2-671b";

#[test]
#[ignore = "needs Cogito-V2 weights and packed_experts/ on /Volumes/Temp Backup"]
fn cogito_moe_layer3_gpu_matches_cpu() {
    let v = VARIANT;
    let hidden_dim = v.hidden_dim;

    // Layer 3 = first MoE layer (first_k_dense_replace = 3).
    let layer_idx = v.first_k_dense_replace;

    // ---- Open weights / experts / wrap as Metal buffer ----
    let bin = Path::new(ROOT).join("artifacts/model_weights.bin");
    let manifest = Path::new(ROOT).join("artifacts/model_weights.json");
    let experts_dir = Path::new(ROOT).join("root");
    let wf = WeightFile::open(&bin, &manifest).expect("open weights");
    let ef = ExpertFiles::open(&experts_dir).expect("open experts");
    let mut metal = MetalBackend::new().expect("open Metal");
    let device = metal.device().clone();
    let mut bufs = MoeBuffers::new(&device);
    let shared_bufs = SharedExpertBuffers::new(&device);
    let dense_pipes =
        DenseMlpPipelines::fetch(&mut metal).expect("fetch DenseMlpPipelines");
    let bf_pipes =
        BfMatvecPipelines::fetch(&mut metal).expect("fetch BfMatvecPipelines");
    let wf_buf = MtlWeightBuf::wrap(&wf, &device);

    // ---- Synthetic input (sin pattern; same as moe_layer3_smoke) ----
    let mut hidden = vec![0.0f32; hidden_dim];
    for (i, h) in hidden.iter_mut().enumerate() {
        *h = ((i as f32) * 0.001).sin();
    }

    // ---- CPU oracle ----
    let mut out_cpu = vec![0.0f32; hidden_dim];
    let t0 = std::time::Instant::now();
    deepseek_moe_cpu(&wf, &ef, layer_idx, &hidden, &mut out_cpu)
        .expect("deepseek_moe_cpu");
    eprintln!(
        "[cogito-moe-gpu] CPU oracle in {:.3}s",
        t0.elapsed().as_secs_f64()
    );
    assert!(
        out_cpu.iter().all(|x| x.is_finite()),
        "CPU oracle produced non-finite output"
    );

    // ---- GPU path ----
    let mut out_gpu = vec![0.0f32; hidden_dim];
    let t0 = std::time::Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("build io_pool");
    cogito_moe_layer_forward_gpu(
        &mut metal,
        &mut bufs,
        &shared_bufs,
        &dense_pipes,
        &bf_pipes,
        &wf,
        &wf_buf,
        &ef,
        &pool,
        layer_idx,
        &hidden,
        &mut out_gpu,
    )
    .expect("cogito_moe_layer_forward_gpu");
    eprintln!(
        "[cogito-moe-gpu] GPU path in {:.3}s",
        t0.elapsed().as_secs_f64()
    );
    assert!(
        out_gpu.iter().all(|x| x.is_finite()),
        "GPU output contains NaN/Inf"
    );

    // ---- Diff ----
    let cosine = cosine_similarity(&out_gpu, &out_cpu);
    let (max_abs_diff, max_abs_cpu) = max_abs_stats(&out_gpu, &out_cpu);
    let rel = max_abs_diff / max_abs_cpu.max(1e-30);
    eprintln!(
        "[cogito-moe-gpu] cosine={cosine:.6}  \
         max_abs_diff={max_abs_diff:.4e}  \
         max_abs_cpu={max_abs_cpu:.4e}  rel={rel:.4e}",
    );
    assert!(
        cosine >= 0.9999,
        "cosine {cosine:.6} below 0.9999 — likely router \
         disagreement (different expert indices) or shared-expert \
         composition mismatch."
    );
    assert!(
        rel <= 1e-3,
        "rel max-abs-diff {rel:.4e} above 1e-3 — math drift"
    );
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f64 =
        a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum();
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn max_abs_stats(gpu: &[f32], cpu: &[f32]) -> (f32, f32) {
    let mut max_abs_diff = 0.0f32;
    let mut max_abs_cpu = 0.0f32;
    for (&g, &c) in gpu.iter().zip(cpu.iter()) {
        max_abs_diff = max_abs_diff.max((g - c).abs());
        max_abs_cpu = max_abs_cpu.max(c.abs());
    }
    (max_abs_diff, max_abs_cpu)
}
