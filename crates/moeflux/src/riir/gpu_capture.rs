//! Programmatic GPU capture hook for mid-prefill Metal traces.
//!
//! Bounded capture window driven by env vars so a `.gputrace` doesn't
//! blow up to gigabytes. Disabled when env is unset — zero cost on the
//! hot path (one `Option::is_some` per layer).
//!
//! ```text
//! METAL_CAPTURE_ENABLED=1                       # required by Metal itself
//! MOEFLUX_GPU_CAPTURE_PATH=/tmp/moeflux.gputrace # output (.gputrace bundle)
//! MOEFLUX_GPU_CAPTURE_AT_CHUNK=0                # which prefill chunk (default 0)
//! MOEFLUX_GPU_CAPTURE_AT_LAYER=30               # start layer index
//! MOEFLUX_GPU_CAPTURE_LAYERS=2                  # window size in layers
//! ```
//!
//! The capture starts just before the begin_layer hook fires for
//! `(at_chunk, at_layer)` and stops just before the begin_layer hook
//! fires for `(at_chunk, at_layer + n_layers)`. Effective window:
//! all commits issued by layers `[at_layer, at_layer + n_layers)` of
//! chunk `at_chunk`.

use std::path::PathBuf;
use std::sync::OnceLock;

use metal::{CaptureDescriptor, CaptureManager, Device, MTLCaptureDestination};

/// Parsed env. Returned by [`config()`] once at first access.
#[derive(Debug, Clone)]
pub struct GpuCaptureConfig {
    pub at_chunk: usize,
    pub at_layer: usize,
    pub n_layers: usize,
    pub path: PathBuf,
}

impl GpuCaptureConfig {
    /// `(start, stop)` boundary check. The capture starts immediately
    /// before the begin_layer hook for `start`; the next layer hook
    /// equal to `stop` ends it.
    pub fn start_at(&self, chunk_idx: usize, layer_idx: usize) -> bool {
        chunk_idx == self.at_chunk && layer_idx == self.at_layer
    }
    pub fn stop_at(&self, chunk_idx: usize, layer_idx: usize) -> bool {
        chunk_idx == self.at_chunk
            && layer_idx == self.at_layer.saturating_add(self.n_layers)
    }
}

/// Parsed once at first access. `None` when `MOEFLUX_GPU_CAPTURE_PATH`
/// is unset — the universal "capture disabled" signal.
pub fn config() -> Option<&'static GpuCaptureConfig> {
    static CFG: OnceLock<Option<GpuCaptureConfig>> = OnceLock::new();
    CFG.get_or_init(|| {
        let path = std::env::var("MOEFLUX_GPU_CAPTURE_PATH").ok()?;
        let at_chunk = std::env::var("MOEFLUX_GPU_CAPTURE_AT_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0usize);
        let at_layer = std::env::var("MOEFLUX_GPU_CAPTURE_AT_LAYER")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0usize);
        let n_layers = std::env::var("MOEFLUX_GPU_CAPTURE_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2usize)
            .max(1);
        if std::env::var("METAL_CAPTURE_ENABLED").ok().as_deref()
            != Some("1")
        {
            eprintln!(
                "[gpu_capture] MOEFLUX_GPU_CAPTURE_PATH is set but \
                 METAL_CAPTURE_ENABLED is not 1 — Metal will reject \
                 startCapture. Set METAL_CAPTURE_ENABLED=1 to enable."
            );
            return None;
        }
        Some(GpuCaptureConfig {
            at_chunk,
            at_layer,
            n_layers,
            path: PathBuf::from(path),
        })
    })
    .as_ref()
}

/// Begin capture targeting `device`, writing to `path` as a
/// `.gputrace` bundle. Removes any pre-existing path so Metal's
/// "output URL already exists" error doesn't fire on repeat runs.
pub fn start(device: &Device, cfg: &GpuCaptureConfig) {
    let manager = CaptureManager::shared();
    if manager.is_capturing() {
        eprintln!("[gpu_capture] already capturing; skipping start");
        return;
    }
    if cfg.path.exists() {
        let _ = std::fs::remove_dir_all(&cfg.path);
        let _ = std::fs::remove_file(&cfg.path);
    }
    let desc = CaptureDescriptor::new();
    desc.set_capture_device(device);
    desc.set_destination(MTLCaptureDestination::GpuTraceDocument);
    desc.set_output_url(&cfg.path);
    match manager.start_capture(&desc) {
        Ok(()) => eprintln!(
            "[gpu_capture] started → {} (chunk={} layers={}..{})",
            cfg.path.display(),
            cfg.at_chunk,
            cfg.at_layer,
            cfg.at_layer + cfg.n_layers,
        ),
        Err(e) => eprintln!("[gpu_capture] start_capture failed: {e}"),
    }
}

pub fn stop() {
    let manager = CaptureManager::shared();
    if !manager.is_capturing() {
        return;
    }
    manager.stop_capture();
    eprintln!("[gpu_capture] stopped");
}
