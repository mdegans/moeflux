//! Metal backend for the RIIR port.
//!
//! Wraps `metal-rs` into a single-owner [`MetalBackend`] holding the
//! Metal device, command queue, the compiled shader library, and a
//! per-kernel pipeline-state cache. RAII; no globals.
//!
//! ## Shader source location
//!
//! `shaders.metal` lives at `crates/moeflux/shaders/shaders.metal`
//! and is embedded into the binary at compile time via
//! [`include_str!`]. No env vars, no path discovery, no runtime IO
//! to find the source. The C-side oracle (`mod imp`, gated behind
//! the `diff-oracle` feature) reads from the same file via the path
//! `moeflux-sys`'s `build.rs` bakes in.
//!
//! ## What's cached, what isn't
//!
//! - **Cached**: device, command queue, library, pipeline states.
//!   These are immutable once created and reused across every
//!   forward pass. Pipeline cache is lazy — a kernel is compiled
//!   the first time it's requested.
//! - **Not cached**: command buffers, encoders, transient buffers.
//!   These are per-call, RAII'd within method scopes.
//!
//! ## Threading
//!
//! Metal command queues are thread-safe; pipeline states are
//! thread-safe; buffers used as kernel inputs are not — the caller
//! must serialize access to any buffer being written. We hold the
//! single-`&mut Ctx` discipline (per the RIIR plan), so this is a
//! non-issue at the public API layer.

use std::collections::HashMap;

use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions, NSUInteger,
};

/// Errors from the Metal backend.
#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("no Metal device available (system has no GPU?)")]
    NoDevice,
    #[error("compiling shaders.metal: {0}")]
    LibraryCompile(String),
    #[error("kernel '{name}' not found in compiled library")]
    FunctionNotFound { name: String },
    #[error("pipeline-state creation failed for '{name}': {err}")]
    PipelineCreate { name: String, err: String },
}

/// Embedded `shaders.metal` source — compiled into the binary so
/// runtime has no path-discovery requirement. See module doc.
const SHADER_SOURCE: &str =
    include_str!("../../shaders/shaders.metal");

/// All kernels in `shaders.metal`. The smoke test compiles every
/// one of these at startup; if any fails, the shader source is
/// broken (or metal-rs is parsing it wrong) and we want to know
/// before downstream code tries to dispatch.
///
/// Keep alphabetized by kernel name. When a kernel is added /
/// removed in `shaders.metal`, update this list — the smoke test
/// is the canary.
pub const ALL_KERNELS: &[&str] = &[
    "attn_scores_batched",
    "attn_softmax_batched",
    "attn_values_batched",
    "bf16_matvec",
    "compute_decay_beta",
    "conv1d_step",
    "dequant_matvec_2bit",
    "dequant_matvec_4bit",
    "dequant_matvec_4bit_batched",
    "dequant_matvec_4bit_fast",
    "dequant_matvec_4bit_v3",
    "dequant_matvec_4bit_v4",
    "dequant_matvec_4bit_v5",
    "dequant_matvec_8bit_v3",
    "fused_gate_up_swiglu",
    "gated_delta_net_step",
    "gated_rms_norm",
    "mla_sdpa_tile_accumulate",
    "mla_sdpa_tile_finalize",
    "moe_combine_residual",
    "residual_add",
    "rms_norm_apply",
    "rms_norm_apply_bf16",
    "rms_norm_qk",
    "rms_norm_sum_sq",
    "sigmoid_gate",
    "swiglu_fused",
    "swiglu_fused_batched",
    "swiglu_fused_vec4",
    "weighted_sum",
];

/// Single-owner Metal backend. One per `Ctx`. Owns the device,
/// command queue, compiled library, and pipeline cache.
pub struct MetalBackend {
    device: Device,
    queue: CommandQueue,
    library: Library,
    /// Lazily-populated cache of compute pipelines keyed by kernel name.
    /// `&'static str` keys come from [`ALL_KERNELS`] or string literals
    /// in dispatcher helpers — never user input, never owned strings.
    pipelines: HashMap<&'static str, ComputePipelineState>,
}

impl MetalBackend {
    /// Open the system-default Metal device, build a command queue,
    /// and compile `shaders.metal` into a `Library`.
    ///
    /// Pipeline states are *not* eagerly compiled here — they're
    /// built on first request via [`Self::pipeline`]. To force-build
    /// every kernel up front (e.g. for diagnostics or to amortize
    /// JIT cost), call [`Self::warm_all`].
    pub fn new() -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let queue = device.new_command_queue();

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .map_err(MetalError::LibraryCompile)?;

        Ok(Self {
            device,
            queue,
            library,
            pipelines: HashMap::new(),
        })
    }

    /// Underlying Metal device. Exposed so dispatcher helpers can
    /// allocate buffers without re-acquiring `system_default`.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Command queue. Reused across every forward pass.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Get-or-compile the pipeline state for kernel `name`. The
    /// pipeline is cached; subsequent calls are O(1) hash lookup.
    /// `name` must be `'static` so cache keys don't outlive the
    /// strings that produced them — caller passes string literals
    /// from [`ALL_KERNELS`] or from inline `dispatch_*` helpers.
    pub fn pipeline(
        &mut self,
        name: &'static str,
    ) -> Result<&ComputePipelineState, MetalError> {
        if !self.pipelines.contains_key(name) {
            let function = self.library.get_function(name, None).map_err(
                |_| MetalError::FunctionNotFound {
                    name: name.to_string(),
                },
            )?;
            let state = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|err| MetalError::PipelineCreate {
                    name: name.to_string(),
                    err,
                })?;
            self.pipelines.insert(name, state);
        }
        Ok(&self.pipelines[name])
    }

    /// Pre-compile every kernel in [`ALL_KERNELS`]. Used by the
    /// smoke test and as a startup-time cost amortizer if a caller
    /// wants every dispatch to be hot.
    pub fn warm_all(&mut self) -> Result<(), MetalError> {
        for &name in ALL_KERNELS {
            self.pipeline(name)?;
        }
        Ok(())
    }

    /// Number of pipelines currently cached. Diagnostics only.
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }
}

impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBackend")
            .field("device", &self.device.name())
            .field("pipelines_cached", &self.pipelines.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// MtlBuffer — typed RAII wrapper around metal::Buffer
// ---------------------------------------------------------------------------

/// Owned, custom-aligned heap allocation backing an [`MtlBuffer`]
/// constructed via [`MtlBuffer::with_aligned_len_u8`]. Frees on drop.
///
/// The Metal buffer that wraps this allocation uses `deallocator=None`
/// (Metal does not own the bytes). Drop order in [`MtlBuffer`] runs
/// fields in declaration order, so `inner` drops first (releasing the
/// GPU-side reference), then `_backing` drops here (freeing the
/// allocation). Reordering the field list would corrupt this.
struct AlignedBacking {
    ptr: std::ptr::NonNull<u8>,
    layout: std::alloc::Layout,
}

// SAFETY: `AlignedBacking` is a logically-owned heap region. The
// `NonNull<u8>` is the unique owner (no aliasing); access is
// serialized by the enclosing `MtlBuffer` which moeflux holds via
// the single-`&mut Ctx` discipline. `metal::Buffer`'s `Send` bound
// (it's Objective-C reference-counted, thread-safe per Apple docs)
// is what makes `MtlBuffer<u8>` useful across rayon boundaries; the
// backing must match it.
unsafe impl Send for AlignedBacking {}
unsafe impl Sync for AlignedBacking {}

impl Drop for AlignedBacking {
    fn drop(&mut self) {
        // SAFETY: `ptr` was allocated by the global allocator with
        // `layout` in `MtlBuffer::with_aligned_len_u8`. This is the
        // only `dealloc` site for it, and the matching `MtlBuffer`'s
        // `inner` (Buffer) field drops before this (declaration
        // order), so no GPU work can still reference the bytes.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

/// Typed wrapper around a Metal buffer. Tracks element count and
/// element type for type-safe `to_vec` round-trips. All buffers use
/// shared storage mode (CPU+GPU accessible) — moeflux's working set
/// fits in unified memory and the tradeoff favors simplicity over
/// the small bandwidth win of private storage.
pub struct MtlBuffer<T> {
    inner: metal::Buffer,
    len: usize,
    /// Owned backing for the [`MtlBuffer::with_aligned_len_u8`] path,
    /// where Metal wraps externally-allocated bytes via
    /// `newBufferWithBytesNoCopy:` with `deallocator=None`. `None` for
    /// the standard `with_len` / `with_data` paths where Metal owns
    /// the allocation. **Field order matters** — `inner` must drop
    /// before this so the Buffer releases its borrow first.
    _backing: Option<AlignedBacking>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Copy> MtlBuffer<T> {
    /// Allocate `len` elements, uninitialized (zero-filled by Metal
    /// on first GPU access; CPU reads of unwritten regions are
    /// implementation-defined but in practice zero on shared mode).
    pub fn with_len(device: &Device, len: usize) -> Self {
        let bytes = (len * std::mem::size_of::<T>()) as NSUInteger;
        let inner = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        Self {
            inner,
            len,
            _backing: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Allocate `len` elements pre-filled with `data`. Length is
    /// taken from `data`.
    pub fn with_data(device: &Device, data: &[T]) -> Self {
        let bytes = (std::mem::size_of_val(data)) as NSUInteger;
        let inner = device.new_buffer_with_data(
            data.as_ptr().cast(),
            bytes,
            MTLResourceOptions::StorageModeShared,
        );
        Self {
            inner,
            len: data.len(),
            _backing: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Underlying `metal::Buffer` for passing to encoder calls.
    pub fn raw(&self) -> &metal::BufferRef {
        &self.inner
    }

    /// Owned-buffer accessor — same value as [`Self::raw`] but at
    /// `&metal::Buffer` rather than `&metal::BufferRef`. Useful when
    /// downstream APIs (e.g. [`super::gpu_matvec::MatvecSpec`]) want
    /// `&Buffer` specifically. `Buffer` derefs to `BufferRef`, so
    /// callers expecting either can use this; some can't take `&BufferRef`
    /// because their lifetimes are tied to `&Buffer`.
    pub fn buffer(&self) -> &metal::Buffer {
        &self.inner
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copy buffer contents into a `Vec<T>`. For testing / debugging.
    /// Caller is responsible for ensuring any pending GPU writes
    /// have completed (via `wait_until_completed` on the command
    /// buffer that wrote into this buffer).
    ///
    /// # Safety
    ///
    /// Only call when no GPU command buffer that writes to this
    /// buffer is still in flight. Shared-storage memory is read
    /// directly from CPU; concurrent GPU writes produce UB.
    pub fn to_vec(&self) -> Vec<T> {
        let ptr = self.inner.contents() as *const T;
        // SAFETY: see method docs. Caller has ensured no in-flight
        // writers. Length matches the allocation.
        unsafe { std::slice::from_raw_parts(ptr, self.len).to_vec() }
    }

    /// Mutable byte slice view. Only valid while no GPU operation
    /// is reading or writing this buffer. Used by callers that need
    /// to write input data after allocation (e.g. weight loading).
    ///
    /// # Safety
    ///
    /// See [`Self::to_vec`].
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let ptr = self.inner.contents() as *mut T;
        // SAFETY: see method docs.
        unsafe { std::slice::from_raw_parts_mut(ptr, self.len) }
    }
}

impl MtlBuffer<u8> {
    /// Allocate `len` bytes with explicit alignment (e.g. 2 MB for
    /// pread DMA destinations) and wrap as a Metal shared-storage
    /// buffer via `newBufferWithBytesNoCopy:`. Apple's allocator only
    /// lands large allocations on 2 MB boundaries incidentally — for
    /// the expert-pool buffers the C path documents a 3.6× DMA
    /// throughput cliff if we miss the alignment, so we control it
    /// explicitly here.
    ///
    /// `align` must be a power of two and a multiple of `T`'s native
    /// alignment (trivially true for `u8`). The Metal buffer holds a
    /// non-owning reference to the bytes; the [`AlignedBacking`] in
    /// the returned [`MtlBuffer`] frees on drop after `inner` releases.
    pub fn with_aligned_len_u8(
        device: &Device,
        len: usize,
        align: usize,
    ) -> Self {
        assert!(align.is_power_of_two(), "align must be power of two");
        assert!(len > 0, "with_aligned_len_u8 len must be > 0");
        let layout = std::alloc::Layout::from_size_align(len, align)
            .expect("invalid alignment for len");
        // SAFETY: `layout` has nonzero size; OOM is handled by aborting
        // via `handle_alloc_error`, matching Box / Vec behavior.
        let raw = unsafe { std::alloc::alloc(layout) };
        let ptr = std::ptr::NonNull::new(raw)
            .unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        // Wrap as a Metal buffer with deallocator=None — Metal does
        // not own the allocation; `AlignedBacking::drop` does, and
        // runs strictly after `inner` (declaration order).
        let inner = device.new_buffer_with_bytes_no_copy(
            ptr.as_ptr() as *const std::ffi::c_void,
            len as NSUInteger,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        Self {
            inner,
            len,
            _backing: Some(AlignedBacking { ptr, layout }),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> std::fmt::Debug for MtlBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtlBuffer")
            .field("len", &self.len)
            .field("element_size", &std::mem::size_of::<T>())
            .field("byte_size", &(self.len * std::mem::size_of::<T>()))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Smoke tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile every kernel in `shaders.metal`. Catches: shader-source
    /// I/O issues, syntax errors, missing kernel functions, kernel
    /// signatures that don't satisfy Metal's pipeline requirements.
    /// Slow (Metal compiles every kernel) but only runs on macOS
    /// with a real device — CI on other platforms skips via cfg.
    #[test]
    #[ignore = "needs Metal device + access to shaders.metal source"]
    fn metal_backend_compiles_all_kernels() {
        let mut backend =
            MetalBackend::new().expect("MetalBackend::new failed");
        eprintln!("[metal] device: {}", backend.device().name());
        eprintln!("[metal] kernels to compile: {}", ALL_KERNELS.len());

        backend.warm_all().expect("warm_all failed");
        assert_eq!(backend.pipeline_count(), ALL_KERNELS.len());
        eprintln!(
            "[metal] all {} kernels compiled successfully",
            backend.pipeline_count()
        );
    }

    /// Buffer round-trip: write data, read it back. Doesn't dispatch
    /// any kernels — purely tests the host-visible side of shared
    /// storage and the byte-count arithmetic.
    #[test]
    #[ignore = "needs Metal device"]
    fn buffer_round_trip() {
        let backend = MetalBackend::new().expect("MetalBackend::new");
        let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
        let buf = MtlBuffer::with_data(backend.device(), &data);
        assert_eq!(buf.len(), 1024);
        let read = buf.to_vec();
        assert_eq!(read, data);
    }
}
