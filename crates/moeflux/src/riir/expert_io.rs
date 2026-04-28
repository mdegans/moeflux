//! Per-layer expert-blob I/O — slice 9c.
//!
//! Mirrors the C path's expert-loading sub-system in
//! `metal_infer/infer.m` (lines 7570–7605 + 3222–3740) at the
//! "plain pread / mmap" level. Caches and exotic decompression
//! modes are deliberately out of scope:
//!
//! - **In:** open `experts_dir/packed_experts/layer_NN.bin` per
//!   layer (RAII, missing files are tolerated mirroring the C
//!   semantics), read `EXPERT_SIZE` bytes at offset
//!   `expert_idx * EXPERT_SIZE` via [`std::os::unix::fs::FileExt::read_at`].
//! - **Out:** LZ4 decompression (`packed_experts_lz4/`), the LRU
//!   metal-buffer expert cache (`g_expert_cache`), the malloc
//!   expert cache (`g_malloc_cache`), and the async pread
//!   thread pool (`g_async_pread`). Those are slice 9f material.
//! - **Out:** 2-bit experts (`packed_experts_2bit/`). The 4-bit
//!   `EXPERT_SIZE` is wired in here; 2-bit lands as a Phase 7
//!   commit alongside the matvec_2bit pipeline.
//!
//! ## Why no thread pool / mmap yet
//!
//! The C async-pread thread pool exists because pread on macOS has
//! per-syscall overhead and parallelizing K preads gives a real win
//! during decode. For the diff oracle we just need byte-equal
//! output; a synchronous `read_at` produces identical bytes faster
//! than is worth measuring. mmap lands in slice 9d/9e if a real-
//! prompt benchmark shows the cache-warm path matters.
//!
//! ## RAII
//!
//! [`ExpertFiles`] owns one [`std::fs::File`] per layer. Files
//! close on drop. No `Arc`, no global state — fits the strategy's
//! "single-`&mut Ctx` discipline" rule.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use super::variants::{Variant, VARIANT};

/// Errors from expert-blob I/O.
#[derive(Debug, thiserror::Error)]
pub enum ExpertIoError {
    #[error("layer_idx {layer} out of range (must be < {num_layers})")]
    BadLayerIdx { layer: i32, num_layers: usize },
    #[error("expert_idx {expert} out of range (must be < {num_experts})")]
    BadExpertIdx { expert: i32, num_experts: usize },
    #[error("layer {layer} file not opened (probably missing on disk)")]
    LayerFileMissing { layer: usize },
    #[error(
        "expert blob read short: expected {expected} bytes at \
         layer={layer} expert={expert}, got {actual}"
    )]
    ShortRead {
        layer: usize,
        expert: usize,
        expected: usize,
        actual: usize,
    },
    #[error("out buffer must be EXPERT_SIZE={expected} bytes, got {actual}")]
    BadOutLen { expected: usize, actual: usize },
    #[error("I/O error reading layer {layer}: {source}")]
    Io {
        layer: usize,
        #[source]
        source: io::Error,
    },
}

/// Per-layer file handles for the active variant. Slot `i` is `None`
/// if `experts_dir/packed_experts/layer_<i>.bin` was missing at
/// open time — the C path tolerates missing files (zeroes expert
/// outputs in `fused_layer_forward`); we mirror that semantic.
pub struct ExpertFiles {
    /// `Some(file)` if the layer file opened, `None` if it was missing.
    layers: Vec<Option<File>>,
    /// Bytes per expert blob for the active variant. Hard-coded to
    /// 4-bit; 2-bit needs a separate field if the path lands.
    expert_size: usize,
    /// Directory the files were opened relative to. Kept for
    /// diagnostics / debug-impl; never re-read.
    experts_dir: PathBuf,
}

impl ExpertFiles {
    /// Open `experts_dir/packed_experts/layer_NN.bin` for every layer
    /// in the active variant. Missing files leave the slot at `None`.
    /// I/O errors other than `NotFound` propagate.
    ///
    /// Mirrors the C loop at `metal_infer/infer.m:7580..7605` minus
    /// the mmap step (we use `pread` only).
    pub fn open(experts_dir: &Path) -> Result<Self, ExpertIoError> {
        let v: Variant = VARIANT;
        let subdir = experts_dir.join("packed_experts");
        let mut layers = Vec::with_capacity(v.num_layers);
        for i in 0..v.num_layers {
            let path = subdir.join(format!("layer_{i:02}.bin"));
            match File::open(&path) {
                Ok(f) => layers.push(Some(f)),
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    layers.push(None);
                }
                Err(e) => {
                    return Err(ExpertIoError::Io {
                        layer: i,
                        source: e,
                    });
                }
            }
        }
        Ok(Self {
            layers,
            expert_size: v.expert_size_4bit(),
            experts_dir: experts_dir.to_path_buf(),
        })
    }

    /// Read one expert's `EXPERT_SIZE` bytes into `out`. `out.len()`
    /// must equal `expert_size`.
    ///
    /// Equivalent to the C `pread(layer_fds[layer_idx], out,
    /// EXPERT_SIZE, expert_idx * EXPERT_SIZE)` call site at
    /// `infer.m:2915`.
    pub fn read_expert(
        &self,
        layer_idx: usize,
        expert_idx: usize,
        out: &mut [u8],
    ) -> Result<(), ExpertIoError> {
        let v: Variant = VARIANT;
        if layer_idx >= v.num_layers {
            return Err(ExpertIoError::BadLayerIdx {
                layer: layer_idx as i32,
                num_layers: v.num_layers,
            });
        }
        if expert_idx >= v.num_experts {
            return Err(ExpertIoError::BadExpertIdx {
                expert: expert_idx as i32,
                num_experts: v.num_experts,
            });
        }
        if out.len() != self.expert_size {
            return Err(ExpertIoError::BadOutLen {
                expected: self.expert_size,
                actual: out.len(),
            });
        }
        let Some(file) = self.layers[layer_idx].as_ref() else {
            return Err(ExpertIoError::LayerFileMissing { layer: layer_idx });
        };
        let off = (expert_idx as u64) * (self.expert_size as u64);
        let n = file.read_at(out, off).map_err(|e| ExpertIoError::Io {
            layer: layer_idx,
            source: e,
        })?;
        if n != self.expert_size {
            return Err(ExpertIoError::ShortRead {
                layer: layer_idx,
                expert: expert_idx,
                expected: self.expert_size,
                actual: n,
            });
        }
        Ok(())
    }

    /// Number of layers (matches [`Variant::num_layers`]).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// `true` iff the file for `layer_idx` opened successfully.
    pub fn has_layer(&self, layer_idx: usize) -> bool {
        self.layers
            .get(layer_idx)
            .map(Option::is_some)
            .unwrap_or(false)
    }
}

impl std::fmt::Debug for ExpertFiles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let opened = self.layers.iter().filter(|s| s.is_some()).count();
        f.debug_struct("ExpertFiles")
            .field("experts_dir", &self.experts_dir)
            .field("num_layers", &self.layers.len())
            .field("opened", &opened)
            .field("expert_size", &self.expert_size)
            .finish()
    }
}
