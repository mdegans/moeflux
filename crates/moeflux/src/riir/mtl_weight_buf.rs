//! Wrap the [`WeightFile`] mmap as a single Metal buffer for in-place
//! GPU access — Phase 4c plumbing.
//!
//! Mirrors `metal_set_weights` (infer.m:1334): wraps the mmap region
//! as one [`metal::Buffer`] via `newBufferWithBytesNoCopy:length:options:deallocator:`
//! with `MTLResourceStorageModeShared`. macOS unified memory means
//! the GPU reads directly out of the same pages the CPU mmap'd; no
//! copy on Ctx open. The deallocator is `nil` because [`WeightFile`]
//! owns the mmap lifetime — the Metal buffer just borrows.
//!
//! Tensor offsets within the buffer are computed via pointer
//! subtraction against the mmap base, matching the C path's
//! `(const char *)tensor_ptr - (const char *)[wf_buf contents]`
//! pattern. Kernels that take `(buffer, byte_offset)` pairs (every
//! kernel that reads weights) point at this single buffer with the
//! per-tensor offset.

use std::ffi::c_void;
use std::ptr::NonNull;

use metal::{Buffer, Device, MTLResourceOptions, NSUInteger};

use super::weight_file::WeightFile;

/// Errors specific to wrapping a [`WeightFile`] as a Metal buffer.
#[derive(Debug, thiserror::Error)]
pub enum MtlWeightBufError {
    /// Tensor name resolved but its bytes don't lie inside the mmap
    /// — should be caught by [`WeightFile::open`]'s bounds check, but
    /// re-checked at offset compute time as defense in depth.
    #[error("tensor '{name}' bytes outside mmap region")]
    TensorOutOfBounds { name: String },
    /// A tensor required by the layer-weight cache build path was not
    /// present in the manifest. Catches truncated / partial weight
    /// files at `LayerWeightCache::build` time instead of at the per-
    /// layer forward call.
    #[error("required tensor '{name}' missing from weight manifest")]
    MissingTensor { name: String },
}

/// `MTLBuffer` wrapping a [`WeightFile`]'s mmap region. Constructed
/// once per [`crate::riir::RsCtx`]; lives as long as both the
/// `WeightFile` and the [`metal::Device`] that produced it.
///
/// **Lifetime invariant**: `WeightFile` must outlive every kernel
/// dispatch that reads from this buffer. Callers ensure this by
/// keeping both as fields of the same `RsCtx`.
pub struct MtlWeightBuf {
    buf: Buffer,
    /// Pointer to the start of the mmap region. Used to compute byte
    /// offsets via subtraction. `NonNull<u8>` for the right covariance;
    /// the actual lifetime is bound to the [`WeightFile`] this was
    /// built from (caller invariant).
    base_ptr: NonNull<u8>,
    /// Bytes accessible via the wrapped buffer. Equals the mmap size
    /// rounded up to the page boundary required by
    /// `newBufferWithBytesNoCopy`.
    aligned_len: usize,
}

// Safe to send across threads: contents are read-only weight data
// borrowed from a `WeightFile` that lives as long as `Self`. Metal
// buffers are themselves `Send`-safe per Apple's documentation
// (concurrent access requires per-buffer external synchronization,
// which we provide via the single-`&mut Ctx` discipline).
unsafe impl Send for MtlWeightBuf {}

impl MtlWeightBuf {
    /// Wrap `wf`'s mmap region as a Metal buffer on `device`.
    /// Length is rounded up to the 16 KiB page boundary required by
    /// `newBufferWithBytesNoCopy`.
    pub fn wrap(wf: &WeightFile, device: &Device) -> Self {
        // SAFETY: WeightFile guarantees the mmap is alive for at
        // least as long as `&wf`. Caller's invariant: keep `wf` alive
        // for as long as this `MtlWeightBuf`.
        //
        // We construct `base_ptr` from the first byte of the first
        // tensor's slice — there's at least one tensor and its
        // slice's first byte is the mmap base + first-tensor-offset.
        // Subtract that to recover the mmap base. The simpler path
        // would be a `WeightFile::mmap_ptr()` accessor; adding one
        // here keeps the API surface tight.
        let base_ptr = wf
            .iter()
            .next()
            .and_then(|(name, _)| {
                let bytes = wf.tensor_bytes(name)?;
                let info = wf.tensor_info(name)?;
                let off = info.offset as usize;
                // bytes.as_ptr() points at mmap[off]; back off to mmap[0]
                let mmap_base = unsafe { bytes.as_ptr().sub(off) };
                NonNull::new(mmap_base as *mut u8)
            })
            .expect("WeightFile is non-empty");

        let raw_len = wf.file_size();
        let page = 16384;
        let aligned_len = (raw_len + page - 1) & !(page - 1);

        // base_ptr is valid for `aligned_len` bytes — the first
        // `raw_len` are mmap'd weight data, the trailing padding
        // into the next page is OS-zero-filled by the kernel page
        // allocator (mmap's standard behavior). Kernels never index
        // past `raw_len` because tensor offsets are bounds-checked
        // at WeightFile load. Deallocator is None — Metal does not
        // own the mapping; WeightFile does.
        let buf = device.new_buffer_with_bytes_no_copy(
            base_ptr.as_ptr() as *const c_void,
            aligned_len as NSUInteger,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        Self {
            buf,
            base_ptr,
            aligned_len,
        }
    }

    /// Underlying [`metal::Buffer`] for binding into encoders. Use
    /// [`Self::tensor_offset`] to get the byte offset for the tensor
    /// you want to read.
    pub fn buffer(&self) -> &Buffer {
        &self.buf
    }

    /// Byte offset of `name`'s data within the wrapped buffer.
    /// `Some(off)` if the tensor exists in the manifest and its bytes
    /// lie within the mmap; `None` if the tensor is missing.
    /// Errors if the tensor info is internally inconsistent (should
    /// never happen for a `WeightFile` built via [`WeightFile::open`]
    /// — that already bounds-checks every entry).
    pub fn tensor_offset(
        &self,
        wf: &WeightFile,
        name: &str,
    ) -> Result<Option<u64>, MtlWeightBufError> {
        let Some(info) = wf.tensor_info(name) else {
            return Ok(None);
        };
        let off = info.offset;
        if (off as usize) + (info.size as usize) > self.aligned_len {
            return Err(MtlWeightBufError::TensorOutOfBounds {
                name: name.to_string(),
            });
        }
        Ok(Some(off))
    }

    /// Total bytes accessible via the wrapped buffer (page-aligned).
    pub fn aligned_len(&self) -> usize {
        self.aligned_len
    }

    /// Pointer to the mmap base. Diagnostics / debugging only —
    /// callers shouldn't dereference it.
    pub fn base_ptr(&self) -> *const u8 {
        self.base_ptr.as_ptr()
    }
}

impl std::fmt::Debug for MtlWeightBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtlWeightBuf")
            .field("aligned_len", &self.aligned_len)
            .field(
                "size_gb",
                &(self.aligned_len as f64 / 1e9),
            )
            .finish()
    }
}
