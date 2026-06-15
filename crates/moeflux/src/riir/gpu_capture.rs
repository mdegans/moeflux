//! Programmatic GPU capture hook for Metal traces.
//!
//! Bounded capture window driven by env vars so a `.gputrace` doesn't
//! blow up to gigabytes. Disabled when env is unset — zero cost on the
//! hot path (one `Option::is_some` per layer).
//!
//! ```text
//! METAL_CAPTURE_ENABLED=1                        # required by Metal itself
//! MOEFLUX_GPU_CAPTURE_PATH=/tmp/moeflux.gputrace # output (.gputrace bundle)
//! MOEFLUX_GPU_CAPTURE_MODE=prefill               # prefill (default) | decode
//! MOEFLUX_GPU_CAPTURE_LAYER=30                   # start layer index (both modes)
//!
//! # prefill-only:
//! MOEFLUX_GPU_CAPTURE_CHUNK=0                    # which prefill chunk (default 0)
//! MOEFLUX_GPU_CAPTURE_N_LAYERS=2                 # window size in layers (default 2)
//!
//! # decode-only:
//! MOEFLUX_GPU_CAPTURE_TOKEN=10                   # which decode token to capture (default 10)
//! ```
//!
//! In prefill mode, capture starts just before `begin_layer` fires for
//! `(chunk, layer)` and stops at `(chunk, layer + n_layers)`.
//!
//! In decode mode, capture waits for the Nth token (countdown from
//! `MOEFLUX_GPU_CAPTURE_TOKEN`), then captures exactly 1 layer.
//! Defaulting to token 10 lets the GPU warm up before tracing.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use metal::{CaptureDescriptor, CaptureManager, Device, MTLCaptureDestination};

/// Set by [`decode_begin_token()`] when the countdown reaches zero;
/// cleared by [`stop()`]. Guards [`decode_start`] / [`decode_stop`]
/// so the layer-level checks only fire on the target token.
static DECODE_ARMED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone)]
pub enum CaptureMode {
    Prefill {
        at_chunk: usize,
        at_layer: usize,
        n_layers: usize,
    },
    Decode {
        at_layer: usize,
        at_token: usize,
    },
}

/// Parsed env. Returned by [`config()`] once at first access.
#[derive(Debug, Clone)]
pub struct GpuCaptureConfig {
    pub path: PathBuf,
    pub mode: CaptureMode,
}

impl GpuCaptureConfig {
    /// Prefill path: should capture start here?
    pub fn prefill_start(&self, chunk_idx: usize, layer_idx: usize) -> bool {
        match &self.mode {
            CaptureMode::Prefill {
                at_chunk, at_layer, ..
            } => chunk_idx == *at_chunk && layer_idx == *at_layer,
            CaptureMode::Decode { .. } => false,
        }
    }

    /// Prefill path: should capture stop here?
    pub fn prefill_stop(&self, chunk_idx: usize, layer_idx: usize) -> bool {
        match &self.mode {
            CaptureMode::Prefill {
                at_chunk,
                at_layer,
                n_layers,
            } => chunk_idx == *at_chunk && layer_idx == at_layer.saturating_add(*n_layers),
            CaptureMode::Decode { .. } => false,
        }
    }

    /// Decode path: should capture start at this layer?
    /// Only returns true when [`DECODE_ARMED`] is set (by
    /// [`decode_begin_token`]).
    pub fn decode_start(&self, layer_idx: usize) -> bool {
        match &self.mode {
            CaptureMode::Decode { at_layer, .. } => {
                DECODE_ARMED.load(Ordering::Relaxed) && layer_idx == *at_layer
            }
            CaptureMode::Prefill { .. } => false,
        }
    }

    /// Decode path: should capture stop at this layer?
    pub fn decode_stop(&self, layer_idx: usize) -> bool {
        match &self.mode {
            CaptureMode::Decode { at_layer, .. } => {
                DECODE_ARMED.load(Ordering::Relaxed) && layer_idx == at_layer.saturating_add(1)
            }
            CaptureMode::Prefill { .. } => false,
        }
    }
}

/// Call once per decode token, **before** the layer loop. Counts
/// down from `at_token`; returns `true` exactly once — on the
/// target token — and arms [`DECODE_ARMED`] for that token's
/// layer pass.
pub fn decode_begin_token() -> bool {
    static COUNTDOWN: OnceLock<AtomicUsize> = OnceLock::new();
    let counter = COUNTDOWN.get_or_init(|| {
        let n = config()
            .and_then(|cfg| match &cfg.mode {
                CaptureMode::Decode { at_token, .. } => Some(*at_token),
                _ => None,
            })
            .unwrap_or(0);
        AtomicUsize::new(n)
    });

    // fetch_sub wraps on underflow — fine, the value will never
    // be 0 again (would take usize::MAX more calls).
    let prev = counter.fetch_sub(1, Ordering::Relaxed);
    if prev == 0 {
        DECODE_ARMED.store(true, Ordering::Relaxed);
        if let Some(cfg) = config() {
            if let CaptureMode::Decode { at_layer, at_token } = &cfg.mode {
                eprintln!(
                    "[gpu_capture] token {at_token} reached, \
                     arming capture at layer {at_layer}",
                );
            }
        }
        true
    } else {
        false
    }
}

/// Parsed once at first access. `None` when `MOEFLUX_GPU_CAPTURE_PATH`
/// is unset — the universal "capture disabled" signal.
pub fn config() -> Option<&'static GpuCaptureConfig> {
    static CFG: OnceLock<Option<GpuCaptureConfig>> = OnceLock::new();
    CFG.get_or_init(|| {
        let path = std::env::var("MOEFLUX_GPU_CAPTURE_PATH").ok()?;
        if std::env::var("METAL_CAPTURE_ENABLED").ok().as_deref() != Some("1") {
            eprintln!(
                "[gpu_capture] MOEFLUX_GPU_CAPTURE_PATH is set but \
                 METAL_CAPTURE_ENABLED is not 1 — Metal will reject \
                 startCapture. Set METAL_CAPTURE_ENABLED=1 to enable."
            );
            return None;
        }

        let mode_str =
            std::env::var("MOEFLUX_GPU_CAPTURE_MODE").unwrap_or_else(|_| "prefill".into());

        let mode = match mode_str.as_str() {
            "prefill" => {
                let at_chunk = std::env::var("MOEFLUX_GPU_CAPTURE_CHUNK")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0usize);
                let at_layer = std::env::var("MOEFLUX_GPU_CAPTURE_LAYER")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0usize);
                let n_layers = std::env::var("MOEFLUX_GPU_CAPTURE_N_LAYERS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2usize)
                    .max(1);
                CaptureMode::Prefill {
                    at_chunk,
                    at_layer,
                    n_layers,
                }
            }
            "decode" => {
                let at_layer = std::env::var("MOEFLUX_GPU_CAPTURE_LAYER")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0usize);
                let at_token = std::env::var("MOEFLUX_GPU_CAPTURE_TOKEN")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10usize);
                CaptureMode::Decode { at_layer, at_token }
            }
            other => {
                eprintln!(
                    "[gpu_capture] MOEFLUX_GPU_CAPTURE_MODE={other:?} \
                     unrecognised; valid: prefill | decode."
                );
                return None;
            }
        };

        Some(GpuCaptureConfig {
            path: PathBuf::from(path),
            mode,
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
        Ok(()) => {
            let window = match &cfg.mode {
                CaptureMode::Prefill {
                    at_chunk,
                    at_layer,
                    n_layers,
                } => {
                    format!(
                        "prefill chunk={at_chunk} layers={at_layer}..{}",
                        at_layer + n_layers,
                    )
                }
                CaptureMode::Decode { at_layer, .. } => {
                    format!("decode layer={at_layer}")
                }
            };
            eprintln!("[gpu_capture] started → {} ({window})", cfg.path.display(),);
        }
        Err(e) => eprintln!("[gpu_capture] start_capture failed: {e}"),
    }
}

pub fn stop() {
    let manager = CaptureManager::shared();
    if !manager.is_capturing() {
        return;
    }
    manager.stop_capture();
    DECODE_ARMED.store(false, Ordering::Relaxed);
    eprintln!("[gpu_capture] stopped");
}
