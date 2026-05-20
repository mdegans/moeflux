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

use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
#[cfg(feature = "model-cogito-v2-671b")]
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use metal::{Buffer, MTLResourceOptions, NSUInteger};
use moeflux_metal::ResidencySet;

use crate::riir::backend::{BufId, MetalBufferPool};
use crate::riir::variants::{Variant, VARIANT};

/// Disable kernel readahead on a successfully-opened layer fd.
/// Mirrors `fcntl(fd, F_RDAHEAD, 0)` in the C path at
/// `metal_infer/infer.m:7593`.
///
/// **Variant-gated to `model-cogito-v2-671b` only.** Theoretical
/// rationale: when the expert working set is *much larger* than
/// physical RAM (Cogito-V2 ≈ 340 GB at 4-bit on a 96 GB UMA Mac),
/// kernel readahead is wasteful — adjacent pages get pulled into
/// the page cache, evicting other expert pages we still need.
/// When the working set fits (Qwen3.5-397B-A17B streams from
/// per-layer files into a ~24 MB pool, comfortably resident),
/// readahead might help warm-token cache hits. The C path
/// applied F_RDAHEAD=0 unconditionally; we gate it because the
/// Rust port spans a wider variant range than the C path was
/// tuned for, and the disable was speculative for Cogito too —
/// not driven by profiling on either model.
///
/// **Empirical note (2026-05-01):** A/B'd Qwen at max_tokens=512
/// with and without this disable. Result was perf-neutral
/// (1.7822 vs 1.7805 tok/s). Suspected as a regression vector
/// vs the 1.96 baseline; turned out not to be. Gate stays as a
/// documented variant difference; the rationale above remains
/// theoretical and untested for Cogito.
#[cfg(feature = "model-cogito-v2-671b")]
fn disable_readahead(file: &File) {
    // SAFETY: `file.as_raw_fd()` is owned by `file` for the duration
    // of this call; `F_RDAHEAD` with arg=0 has no failure modes that
    // affect correctness — if it errors, readahead stays at default
    // and we eat a tiny perf loss. Log for diagnostics but don't
    // propagate.
    #[cfg(target_os = "macos")]
    unsafe {
        let rc = libc::fcntl(file.as_raw_fd(), libc::F_RDAHEAD, 0i32);
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            eprintln!("[experts] fcntl(F_RDAHEAD, 0) failed: {err}");
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = file;
    }
}

/// No-op for variants whose expert working set fits in RAM with
/// headroom — kernel readahead is helpful there. See the
/// cogito-gated sibling for the full rationale.
#[cfg(not(feature = "model-cogito-v2-671b"))]
fn disable_readahead(_file: &File) {}

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

// Expert I/O is always mmap. The previous `ExpertIoMode::Pread` mode
// staged each layer's experts into a single GPU buffer via a tight
// synchronous `pread` + `pool.upload_at` loop on the calling thread
// (see git history for the deleted code). On 2026-05-20 we measured
// that path against mmap on a 15692-token prefill: same wall-clock
// (74.5s vs 73.8s, within noise), but pread mode burned ~36% more
// main-thread CPU on a workload that was already GPU-bound at the
// MLX `affine_gather_qmm_rhs` kernel's 20.83% register-pressure
// occupancy ceiling. The OS page cache + readahead does the work
// pread mode was meant to manage, and unified-memory Metal can read
// the mmap'd region directly with no staging copy.
//
// If you are tempted to add a pread fallback back: don't. The path
// it was solving (low-RAM hardware where the expert working set
// can't fit in the page cache) has never been hit on supported
// devices and would benefit more from `madvise(MADV_RANDOM)` or
// `MADV_DONTNEED` on the mmap than from re-introducing serial pread
// on the GPU's critical path. See
// `.claude/memory/pread_teardown_landed.md` for the full rationale.

/// Per-layer file handles for the active variant. Slot `i` is `None`
/// if `experts_dir/packed_experts/layer_<i>.bin` was missing at
/// open time — the C path tolerates missing files (zeroes expert
/// outputs in `fused_layer_forward`); we mirror that semantic.
///
/// **Drop order:** `mmap_buffers` (Metal `Buffer`s wrapping the
/// mmap'd regions, deallocator `None`) must drop before
/// `mmap_layers` (the `Mmap`s that own the kernel mappings). Fields
/// drop in declaration order, so `mmap_buffers` is declared above
/// `mmap_layers`. In practice both live for the whole session.
pub struct ExpertFiles {
    /// Pin every `mmap_buffers` allocation as GPU-resident across
    /// cmdbuf boundaries via `MTLResidencySet` (macOS 15+). Populated
    /// by [`Self::attach_to_device`]. `None` if the residency-set API
    /// isn't available on this OS or no buffers were attached yet.
    ///
    /// **Declared first** so it drops first — `endResidency` runs
    /// while the registered allocations are still alive.
    expert_rset: Option<ResidencySet>,
    /// `Some(file)` if the layer file opened, `None` if it was missing.
    layers: Vec<Option<File>>,
    /// Bytes per expert blob for the active variant. Hard-coded to
    /// 4-bit; 2-bit needs a separate field if the path lands.
    expert_size: usize,
    /// Directory the files were opened relative to. Kept for
    /// diagnostics / debug-impl; never re-read.
    experts_dir: PathBuf,
    /// Per-layer Metal buffer wrapping the mmap'd region via
    /// `newBufferWithBytesNoCopy`. Populated by `attach_to_device`
    /// once the Metal backend exists. `None` for missing layers.
    /// Declared before `mmap_layers` so it drops first.
    mmap_buffers: Vec<Option<Buffer>>,
    /// S10b-pre-2 — per-layer `BufId` parallel to `mmap_buffers`.
    /// Populated by [`Self::attach_to_device`] via
    /// `MetalBufferPool::register_borrowed`. Lets graph-mode `Op`s
    /// (`Op::MoeBatchedPermuteFuse`) address mmap'd expert data by
    /// `BufId` rather than `&Buffer`. The pool holds a refcounted
    /// clone of the same `Buffer` that lives in `mmap_buffers`;
    /// drop order is irrelevant (BufId is `Copy`).
    mmap_buf_ids: Vec<Option<BufId>>,
    /// Per-layer mmap. Populated at `open` time when `mode == Mmap`.
    /// `None` for the Pread mode and for missing layers.
    mmap_layers: Vec<Option<Mmap>>,
}

impl ExpertFiles {
    /// Open `experts_dir/packed_experts/layer_NN.bin` for every layer
    /// in the active variant. Missing files leave the slot at `None`.
    /// I/O errors other than `NotFound` propagate.
    ///
    /// Mirrors the C loop at `metal_infer/infer.m:7580..7605`. Each
    /// opened layer file is mmap'd unconditionally; the matching
    /// Metal buffer is built lazily by [`Self::attach_to_device`]
    /// once the Metal backend exists. The `MOEFLUX_EXPERT_IO` env
    /// var is now inert — set to anything other than `mmap` and it
    /// emits a friendly "ignored" warning, then proceeds with mmap.
    pub fn open(experts_dir: &Path) -> Result<Self, ExpertIoError> {
        let v: Variant = VARIANT;
        if let Ok(val) = std::env::var("MOEFLUX_EXPERT_IO") {
            if val != "mmap" {
                eprintln!(
                    "[experts] MOEFLUX_EXPERT_IO={val:?} ignored; \
                     expert I/O is always mmap (pread mode was removed \
                     2026-05-20 — see expert_io.rs comment)."
                );
            }
        }
        let subdir = experts_dir.join("packed_experts");
        let mut layers = Vec::with_capacity(v.num_layers);
        let mut mmap_layers: Vec<Option<Mmap>> =
            (0..v.num_layers).map(|_| None).collect();
        for i in 0..v.num_layers {
            let path = subdir.join(format!("layer_{i:02}.bin"));
            match File::open(&path) {
                Ok(f) => {
                    disable_readahead(&f);
                    // SAFETY: file is owned by the closure for the
                    // duration of the mmap call; the resulting mapping
                    // survives the file handle (closed when `f` drops
                    // at end of this match arm). Returned mapping is
                    // read-only, no aliasing concerns.
                    let mmap = unsafe { Mmap::map(&f) }.map_err(|e| {
                        ExpertIoError::Io {
                            layer: i,
                            source: e,
                        }
                    })?;
                    mmap_layers[i] = Some(mmap);
                    layers.push(Some(f));
                }
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
        let mmap_buffers = (0..v.num_layers).map(|_| None).collect();
        let mmap_buf_ids = (0..v.num_layers).map(|_| None).collect();
        eprintln!(
            "[experts] {} layer(s) mmap'd \
             (Metal buffers built on first batched call)",
            mmap_layers.iter().filter(|m| m.is_some()).count()
        );
        Ok(Self {
            expert_rset: None,
            layers,
            expert_size: v.expert_size_4bit(),
            experts_dir: experts_dir.to_path_buf(),
            mmap_buffers,
            mmap_buf_ids,
            mmap_layers,
        })
    }

    /// Wrap each mmap'd layer as a Metal-shared buffer via
    /// `newBufferWithBytesNoCopy`, and register the buffer with the
    /// graph-mode buffer pool. Idempotent: the second call is a no-op.
    /// Called once from `ensure_linear_resources` after the Metal
    /// backend is created.
    ///
    /// S10b-pre-2: parameter changed from `&Device` to
    /// `&mut MetalBufferPool`. The buffer is stored in both
    /// `mmap_buffers` (existing `&Buffer` accessor still works) AND
    /// `mmap_buf_ids` (new `BufId` accessor for graph-mode `Op`s);
    /// they share an NSObject-refcounted clone of the same backing.
    pub fn attach_to_device(&mut self, pool: &mut MetalBufferPool) {
        // Lazy-init the residency set on the first attach call. The
        // capacity hint is the layer count — a slight over-estimate
        // when some layers are missing, but it's a hint, not a cap.
        // Returns `None` on macOS < 15; we then attach buffers without
        // pinning.
        if self.expert_rset.is_none() {
            self.expert_rset = ResidencySet::new(
                pool.device(),
                "moeflux.experts",
                self.mmap_layers.len() as u64,
            );
            if self.expert_rset.is_none()
                && !moeflux_metal::residency_set::is_available()
            {
                eprintln!(
                    "[experts] MTLResidencySet unavailable (macOS < 15) \
                     — expert buffers will not be pinned"
                );
            }
        }
        let mut added_any = false;
        for i in 0..self.mmap_layers.len() {
            if self.mmap_buffers[i].is_some() {
                continue;
            }
            let Some(mmap) = self.mmap_layers[i].as_ref() else {
                continue;
            };
            let raw_len = mmap.len();
            // Page-align the length the way `MtlWeightBuf::wrap` does:
            // Metal accepts a length, the kernel maps in page-granular
            // units. The trailing padding (if any) past the file end
            // is OS-zero-filled and never indexed by kernels because
            // expert offsets bound at `(num_experts - 1) * expert_size
            // + expert_size <= raw_len`.
            let page = 16384usize;
            let aligned_len = (raw_len + page - 1) & !(page - 1);
            // SAFETY: `mmap.as_ptr()` is valid for `raw_len` bytes; the
            // trailing pad to `aligned_len` is kernel-zero-filled (mmap
            // contract). The `Mmap` outlives this `Buffer` because both
            // live on `ExpertFiles` and drop order puts `Buffer` first
            // (see struct docs). Deallocator `None` — Metal does NOT
            // own the mapping; `Mmap`'s drop unmaps after the Buffer
            // is released.
            let buf = pool.device().new_buffer_with_bytes_no_copy(
                mmap.as_ptr() as *const c_void,
                aligned_len as NSUInteger,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            // Register a refcounted clone with the pool. The pool's
            // BufId entry holds an NSObject retain; this `mmap_buffers`
            // slot holds another. Both die when `ExpertFiles` drops
            // (Buffer refcount → 0 → newBufferWithBytesNoCopy releases
            // the non-owning Metal reference), after which `Mmap`
            // drops and unmaps the file. Order is enforced by struct
            // field declaration order.
            let id = pool.register_borrowed(
                buf.clone(),
                aligned_len,
                "expert_io.mmap_layer",
                /* persistent = */ true,
            );
            if let Some(rset) = self.expert_rset.as_ref() {
                rset.add_allocation(&buf);
            }
            self.mmap_buffers[i] = Some(buf);
            self.mmap_buf_ids[i] = Some(id);
            added_any = true;
        }
        // Commit + requestResidency once at the end of the call, only
        // if we actually staged new allocations. The second
        // (idempotent) call to `attach_to_device` exits with
        // `added_any == false` and skips this — re-requesting on an
        // unchanged set is harmless but pointless.
        if added_any {
            if let Some(rset) = self.expert_rset.as_ref() {
                rset.commit();
                rset.request_residency();
            }
        }
    }

    /// Pool BufId for the mmap'd layer Buffer + expert byte offset.
    /// Mmap-mode sibling of [`Self::mmap_buffer_for_expert`]. Returns
    /// `None` in Pread mode and for missing layers.
    ///
    /// Graph-mode `Op::MoeBatchedPermuteFuse` uses this to address
    /// the layer's mmap buffer directly as its `expert_base` (the
    /// layer mmap is already `num_experts` blocks at `expert_size`
    /// stride, so `expert_idx = 0` gives the base).
    pub fn mmap_id_for_expert(
        &self,
        layer_idx: usize,
        expert_idx: u32,
    ) -> Option<(BufId, u64)> {
        let id = self.mmap_buf_ids.get(layer_idx)?.as_ref()?;
        let off = (expert_idx as u64) * (self.expert_size as u64);
        Some((*id, off))
    }

    /// Returns `Some((layer_buf, byte_offset))` for `(layer_idx,
    /// expert_idx)` when the layer file opened. Returns `None` for
    /// missing layers (caller treats the expert output as zeros, per
    /// the C-side `fused_layer_forward` semantic).
    pub fn mmap_buffer_for_expert(
        &self,
        layer_idx: usize,
        expert_idx: u32,
    ) -> Option<(&Buffer, u64)> {
        let buf = self.mmap_buffers.get(layer_idx)?.as_ref()?;
        let off = (expert_idx as u64) * (self.expert_size as u64);
        Some((buf, off))
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
