//! Phase 1 — GPU dense SwiGLU MLP diff against the CPU oracle.
//!
//! Drives [`dense_mlp_gpu::encode_dense_mlp_layer_forward_gpu`] on
//! Cogito-V2 layer 0 with a sin-pattern input (same shape as
//! `moe_cpu::tests::moe_layer3_smoke`) and compares the output to
//! [`mlp_cpu::dense_mlp_swiglu_cpu`]. Bit-equality is *not* required —
//! GPU and CPU use different reduction orders. Tolerance:
//!
//! - cosine ≥ 0.9999
//! - max(|gpu - cpu|) / max(|cpu|) ≤ 1e-3
//!
//! Run with:
//! ```text
//! cargo test -p moeflux --no-default-features \
//!     --features model-cogito-v2-671b --release \
//!     --test dense_mlp_gpu -- --ignored --nocapture
//! ```

#![cfg(all(target_os = "macos", feature = "model-cogito-v2-671b"))]

use std::path::Path;

use metal::{MTLResourceOptions, NSUInteger};
use moeflux::riir::dense_mlp_gpu::{
    encode_dense_mlp_layer_forward_gpu, DenseMlpPipelines,
};
use moeflux::riir::metal::MetalBackend;
use moeflux::riir::mlp_cpu::dense_mlp_swiglu_cpu;
use moeflux::riir::mtl_weight_buf::MtlWeightBuf;
use moeflux::riir::variants::VARIANT;
use moeflux::riir::weight_file::WeightFile;

const ROOT: &str =
    "/Volumes/Temp Backup/models/blallama/cogito-v2-671b";

#[test]
#[ignore = "needs Cogito-V2 weights mmap'd from /Volumes/Temp Backup"]
fn dense_mlp_layer0_gpu_matches_cpu() {
    let v = VARIANT;
    assert!(
        v.first_k_dense_replace > 0,
        "variant has no dense MLP layers — wrong build feature?"
    );
    let hidden_dim = v.hidden_dim;
    let intermediate = v.dense_intermediate;
    assert!(intermediate > 0, "variant has dense_intermediate=0");

    // ---- Open weights / wrap as Metal buffer ----
    let bin = Path::new(ROOT).join("artifacts/model_weights.bin");
    let manifest = Path::new(ROOT).join("artifacts/model_weights.json");
    let wf = WeightFile::open(&bin, &manifest).expect("open weights");
    let mut metal = MetalBackend::new().expect("open Metal");
    let device = metal.device().clone();
    let wf_buf = MtlWeightBuf::wrap(&wf, &device);

    let pipes =
        DenseMlpPipelines::fetch(&mut metal).expect("fetch pipelines");

    // ---- Synthetic input (sin pattern; same as moe_layer3_smoke) ----
    let mut hidden_host = vec![0.0f32; hidden_dim];
    for (i, h) in hidden_host.iter_mut().enumerate() {
        *h = ((i as f32) * 0.001).sin();
    }

    // ---- CPU oracle ----
    let mut out_cpu = vec![0.0f32; hidden_dim];
    dense_mlp_swiglu_cpu(&wf, 0, &hidden_host, &mut out_cpu)
        .expect("dense_mlp_swiglu_cpu");
    assert!(
        out_cpu.iter().all(|x| x.is_finite()),
        "CPU oracle produced non-finite output"
    );

    // ---- GPU buffers + dispatch ----
    let f32_buf = |n: usize| -> metal::Buffer {
        device.new_buffer(
            (n * std::mem::size_of::<f32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let hidden_buf = f32_buf(hidden_dim);
    let gate_out_buf = f32_buf(intermediate);
    let up_out_buf = f32_buf(intermediate);
    let act_buf = f32_buf(intermediate);
    let out_buf = f32_buf(hidden_dim);

    // SAFETY: shared-storage; no GPU work in flight.
    unsafe {
        std::ptr::copy_nonoverlapping(
            hidden_host.as_ptr(),
            hidden_buf.contents() as *mut f32,
            hidden_dim,
        );
    }

    let queue = metal.queue();
    let cmdbuf = queue.new_command_buffer();
    let t0 = std::time::Instant::now();
    encode_dense_mlp_layer_forward_gpu(
        cmdbuf,
        &pipes,
        &wf,
        &wf_buf,
        /* layer_idx = */ 0,
        &hidden_buf,
        &gate_out_buf,
        &up_out_buf,
        &act_buf,
        &out_buf,
    )
    .expect("encode dense MLP");
    cmdbuf.commit();
    cmdbuf.wait_until_completed();
    let elapsed = t0.elapsed();
    eprintln!(
        "[dense-mlp-gpu] layer 0 forward in {:.3}s",
        elapsed.as_secs_f64()
    );

    // SAFETY: cmdbuf completed.
    let out_gpu: Vec<f32> = unsafe {
        let p = out_buf.contents() as *const f32;
        std::slice::from_raw_parts(p, hidden_dim).to_vec()
    };
    assert!(
        out_gpu.iter().all(|x| x.is_finite()),
        "GPU output contains NaN/Inf"
    );

    // ---- Diff ----
    let cosine = cosine_similarity(&out_gpu, &out_cpu);
    let (max_abs_diff, max_abs_cpu) = max_abs_stats(&out_gpu, &out_cpu);
    let rel = max_abs_diff / max_abs_cpu.max(1e-30);
    eprintln!(
        "[dense-mlp-gpu] cosine={cosine:.6}  \
         max_abs_diff={max_abs_diff:.4e}  \
         max_abs_cpu={max_abs_cpu:.4e}  rel={rel:.4e}",
    );
    assert!(
        cosine >= 0.9999,
        "cosine {cosine:.6} below 0.9999 — math drift"
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
