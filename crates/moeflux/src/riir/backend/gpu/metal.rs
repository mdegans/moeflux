//! Metal backend for the RIIR port.
//!
//! Wraps `metal-rs` into a single-owner [`MetalContext`] holding the
//! Metal device, command queue, the compiled shader library, and a
//! per-kernel pipeline-state cache. RAII; no globals.
//!
//! ## Shader source location
//!
//! `shaders.metal` lives at `crates/moeflux/shaders/shaders.metal`
//! and is embedded into the binary at compile time via
//! [`include_str!`]. No env vars, no path discovery, no runtime IO
//! to find the source.
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
use std::sync::Mutex;

pub use metal::{
    CommandBufferRef, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions, NSUInteger,
};

use objc::{msg_send, sel, sel_impl};

use moeflux_metal::Kernels;

/// Read the UTF-8 bytes out of an `NSString *`. Pure-Objective-C helper —
/// `metal-rs`'s `nsstring_as_str` is crate-private, so we have our own.
/// Caller must hold a live reference to the underlying NSString.
unsafe fn nsstring_as_str<'a>(nsstr: &'a objc::runtime::Object) -> &'a str {
    let bytes: *const i8 = unsafe { msg_send![nsstr, UTF8String] };
    let len: NSUInteger = unsafe { msg_send![nsstr, length] };
    if bytes.is_null() || len == 0 {
        return "";
    }
    let slice = unsafe { std::slice::from_raw_parts(bytes.cast::<u8>(), len as usize) };
    std::str::from_utf8(slice).unwrap_or("<invalid utf-8>")
}

/// Format the `NSError` attached to a faulted `MTLCommandBuffer` for
/// panic output. Maps the integer error code to the
/// [`MTLCommandBufferError`](metal::MTLCommandBufferError) variant name
/// (Timeout / PageFault / OutOfMemory / …) so the panic message names
/// the failure class, not just "completed with error status".
///
/// Returns `"<no NSError>"` when `cmdbuf.error` is nil (the cmdbuf is
/// in `Error` status but Apple didn't attach a reason — rare).
fn cmdbuf_error_detail(cmdbuf: &CommandBufferRef) -> String {
    unsafe {
        let err: *mut objc::runtime::Object = msg_send![cmdbuf, error];
        if err.is_null() {
            return "<no NSError>".to_string();
        }
        let code: isize = msg_send![err, code];
        let desc_obj: *mut objc::runtime::Object = msg_send![err, localizedDescription];
        let desc = if desc_obj.is_null() {
            "<no description>"
        } else {
            nsstring_as_str(&*desc_obj)
        };
        // Names from MTLCommandBufferError. See metal-rs commandbuffer.rs.
        let kind = match code {
            0 => "None",
            1 => "Internal",
            2 => "Timeout",
            3 => "PageFault",
            4 => "Blacklisted",
            7 => "NotPermitted",
            8 => "OutOfMemory",
            9 => "InvalidResource",
            10 => "Memoryless",
            11 => "DeviceRemoved",
            _ => "Unknown",
        };
        format!("{kind}({code}): {desc}")
    }
}

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
    #[error("building the moeflux-metal kernel library: {0}")]
    MlxKernels(String),
}

/// Embedded `shaders.metal` source — compiled into the binary so
/// runtime has no path-discovery requirement. See module doc.
const SHADER_SOURCE: &str = include_str!("../../../../shaders/shaders.metal");

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
    "conv1d_state_update",
    "conv1d_step",
    "dequant_matvec_2bit",
    "dequant_matvec_4bit",
    "dequant_matvec_4bit_batched",
    "dequant_matvec_4bit_fast",
    "dequant_matvec_4bit_v3",
    "dequant_matvec_4bit_v4",
    "dequant_matvec_4bit_v5",
    "dequant_matvec_8bit_v3",
    "dequant_matvec_8bit_v3_n_tokens",
    "fused_gate_up_swiglu",
    "gated_delta_net_chunkwise",
    "gated_delta_net_step",
    "gated_rms_norm",
    "kv_cache_append_n_tokens",
    "mla_sdpa_tile_accumulate",
    "mla_sdpa_tile_finalize",
    "moe_combine_residual",
    "moe_combine_residual_n_tokens",
    "moe_normalize_weights",
    "moe_softmax_topk",
    "residual_add",
    "residual_add_n_tokens",
    "rms_norm_apply",
    "rms_norm_apply_bf16",
    "rms_norm_bf16_fused_n_tokens",
    "rms_norm_per_head_n_tokens",
    "rms_norm_qk",
    "rms_norm_sum_sq",
    "rope_n_tokens",
    "sigmoid_gate",
    "split_q_gate",
    "swiglu_fused",
    "swiglu_fused_batched",
    "swiglu_fused_vec4",
    "weighted_sum",
];

/// Per-label cmdbuf submission stats. Aggregated under
/// [`MetalContext::cmdbuf_stats`] via the labeled commit+wait wrapper.
///
/// `gpu_ns` is reserved for a future port to a metal-rs version that
/// surfaces `gpu_start_time` / `gpu_end_time`; for now it stays at
/// zero and consumers should ignore it.
#[derive(Debug, Default, Clone, Copy)]
pub struct CmdbufStat {
    pub count: u64,
    pub cpu_wait_ns: u64,
    pub gpu_ns: u64,
}

/// Single-owner Metal backend. One per `Ctx`. Owns the device,
/// command queue, compiled library, and pipeline cache.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    library: Library,
    /// Lazily-populated cache of compute pipelines keyed by kernel name.
    /// `&'static str` keys come from [`ALL_KERNELS`] or string literals
    /// in dispatcher helpers — never user input, never owned strings.
    pipelines: HashMap<&'static str, ComputePipelineState>,
    /// Per-label cmdbuf timing accumulator. Populated by
    /// [`Self::commit_and_wait_labeled`]. Behind a [`Mutex`] so the
    /// helper can take `&self` — contention is irrelevant (one
    /// lock per cmdbuf, dwarfed by the wait itself).
    cmdbuf_stats: Mutex<HashMap<&'static str, CmdbufStat>>,
    /// The compiled `moeflux-metal` kernels. Built once here so every
    /// consumer — the `Op` executor and the direct attn-forward callers
    /// alike — shares one compiled library.
    kernels: Kernels,
}

impl MetalContext {
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

        let kernels = Kernels::new(&device)
            .map_err(|e| MetalError::MlxKernels(e.to_string()))?;

        Ok(Self {
            device,
            queue,
            library,
            pipelines: HashMap::new(),
            cmdbuf_stats: Mutex::new(HashMap::new()),
            kernels,
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

    /// The compiled `moeflux-metal` kernels, built once in [`Self::new`].
    /// Shared by the `Op` executor and the direct attn-forward callers.
    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    /// Clone the command-queue handle. `metal::CommandQueue` is an
    /// NSObject-backed reference-counted handle, so this is cheap
    /// (one Objective-C `retain`).
    ///
    /// Use this when a caller needs to allocate a cmdbuf from the queue
    /// *and* hold `&mut MetalContext` simultaneously for pipeline
    /// fetches or labeled commit-waits. Calling
    /// `metal.queue().new_command_buffer()` returns a `&CommandBufferRef`
    /// whose lifetime is tied to `metal`, which conflicts with later
    /// `&mut metal` use. Cloning the queue out severs that lifetime
    /// dependency.
    pub fn queue_clone(&self) -> CommandQueue {
        self.queue.clone()
    }

    /// Get-or-compile the pipeline state for kernel `name`. The
    /// pipeline is cached; subsequent calls are O(1) hash lookup.
    /// `name` must be `'static` so cache keys don't outlive the
    /// strings that produced them — caller passes string literals
    /// from [`ALL_KERNELS`] or from inline `dispatch_*` helpers.
    pub fn pipeline(&mut self, name: &'static str) -> Result<&ComputePipelineState, MetalError> {
        if !self.pipelines.contains_key(name) {
            let function = self.library.get_function(name, None).map_err(|_| {
                MetalError::FunctionNotFound {
                    name: name.to_string(),
                }
            })?;
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

    /// Commit `cmdbuf`, wait for completion, and record the CPU wait
    /// time under `label` in [`Self::cmdbuf_stats`].
    ///
    /// `label` must be `'static` so the stats map can key on it
    /// without owning a string per call.
    ///
    /// Panics if the command buffer completes with an error status —
    /// a faulting kernel otherwise leaves its output buffers unwritten
    /// and the failure surfaces far downstream as garbage / NaN logits
    /// with no hint of where it began. Rerun under `MTL_DEBUG_LAYER=1`
    /// `MTL_SHADER_VALIDATION=1` for the precise fault.
    pub fn commit_and_wait_labeled(&self, cmdbuf: &CommandBufferRef, label: &'static str) {
        let t0 = std::time::Instant::now();
        cmdbuf.commit();
        cmdbuf.wait_until_completed();
        let cpu_wait_ns = t0.elapsed().as_nanos() as u64;

        // Fail fast — previously a cmdbuf error was silently swallowed.
        // `cmdbuf_error_detail` extracts the NSError code + localized
        // description so the panic names the failure class (Timeout /
        // PageFault / OutOfMemory / …) rather than "error status".
        if cmdbuf.status() == metal::MTLCommandBufferStatus::Error {
            let detail = cmdbuf_error_detail(cmdbuf);
            panic!(
                "Metal command buffer '{label}' completed with error \
                 status: {detail}. Rerun with MTL_DEBUG_LAYER=1 \
                 MTL_SHADER_VALIDATION=1 for the fault detail."
            );
        }

        let mut stats = self
            .cmdbuf_stats
            .lock()
            .expect("cmdbuf_stats mutex poisoned");
        let entry = stats.entry(label).or_default();
        entry.count += 1;
        entry.cpu_wait_ns += cpu_wait_ns;
    }

    /// Snapshot the per-label cmdbuf stats. Returns `(label, stat)`
    /// pairs sorted by label for deterministic reporting.
    pub fn cmdbuf_stats(&self) -> Vec<(&'static str, CmdbufStat)> {
        let stats = self
            .cmdbuf_stats
            .lock()
            .expect("cmdbuf_stats mutex poisoned");
        let mut out: Vec<_> = stats.iter().map(|(k, v)| (*k, *v)).collect();
        out.sort_by_key(|(k, _)| *k);
        out
    }

    /// Zero out the per-label cmdbuf stats. Call before a measured
    /// segment (e.g. per-request) to get clean numbers.
    pub fn reset_cmdbuf_stats(&self) {
        self.cmdbuf_stats
            .lock()
            .expect("cmdbuf_stats mutex poisoned")
            .clear();
    }
}

impl std::fmt::Debug for MetalContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalContext")
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
        // SAFETY: see method docs — caller has ensured no in-flight
        // writers; `self.len` matches the allocation.
        unsafe { buffer_as_slice::<T>(&self.inner, self.len).to_vec() }
    }

    /// Immutable slice view over the buffer's contents.
    ///
    /// # Safety
    ///
    /// See [`Self::to_vec`].
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: see method docs.
        unsafe { buffer_as_slice::<T>(&self.inner, self.len) }
    }

    /// Mutable byte slice view. Only valid while no GPU operation
    /// is reading or writing this buffer. Used by callers that need
    /// to write input data after allocation (e.g. weight loading).
    ///
    /// # Safety
    ///
    /// See [`Self::to_vec`].
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: see method docs.
        unsafe { buffer_as_mut_slice::<T>(&self.inner, self.len) }
    }
}

/// View `n` elements of type `T` over a shared-storage Metal buffer's
/// `contents()` pointer.
///
/// # Safety
///
/// This is the canonical safety contract for shared-storage Metal
/// buffer CPU access in moeflux. Every `unsafe` slice over an
/// `MTLBuffer` in this crate forwards to this contract.
///
/// - **No GPU work in flight**: `MTLResourceStorageModeShared` puts
///   the buffer in unified memory; the bytes the GPU touches *are*
///   the bytes `contents()` returns. A GPU command buffer that reads
///   or writes `buf` running concurrently with this CPU access is
///   undefined behaviour. Callers must have driven the relevant
///   command buffer to completion (`wait_until_completed`) or have
///   independent evidence no kernel touches `buf`.
/// - **Aliasing**: the mutable variant ([`buffer_as_mut_slice`])
///   requires the caller hold unique access for the returned slice's
///   lifetime — no other CPU or GPU reader/writer.
/// - **Bounds**: `n * size_of::<T>()` must not exceed the buffer's
///   byte length.
/// - **Alignment**: `contents()` must be `T`-aligned. In practice
///   shared buffers are page-aligned (16 KiB on Apple Silicon), so
///   this holds for every native scalar moeflux uses. Compound
///   element types over `T`-aligned bases are fine.
pub unsafe fn buffer_as_slice<T>(buf: &metal::BufferRef, n: usize) -> &[T] {
    // SAFETY: forwarded to the caller's contract above.
    unsafe { std::slice::from_raw_parts(buf.contents() as *const T, n) }
}

/// Mutable counterpart of [`buffer_as_slice`].
///
/// # Safety
///
/// See [`buffer_as_slice`]; the caller additionally holds unique
/// access to the buffer for the returned slice's lifetime — no other
/// CPU or GPU reader/writer.
pub unsafe fn buffer_as_mut_slice<T>(
    buf: &metal::BufferRef,
    n: usize,
) -> &mut [T] {
    // SAFETY: forwarded to the caller's contract above.
    unsafe {
        std::slice::from_raw_parts_mut(buf.contents() as *mut T, n)
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
    pub fn with_aligned_len_u8(device: &Device, len: usize, align: usize) -> Self {
        assert!(align.is_power_of_two(), "align must be power of two");
        assert!(len > 0, "with_aligned_len_u8 len must be > 0");
        let layout =
            std::alloc::Layout::from_size_align(len, align).expect("invalid alignment for len");
        // SAFETY: `layout` has nonzero size; OOM is handled by aborting
        // via `handle_alloc_error`, matching Box / Vec behavior.
        let raw = unsafe { std::alloc::alloc(layout) };
        let ptr =
            std::ptr::NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
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
        let mut backend = MetalContext::new().expect("MetalContext::new failed");
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
        let backend = MetalContext::new().expect("MetalContext::new");
        let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
        let buf = MtlBuffer::with_data(backend.device(), &data);
        assert_eq!(buf.len(), 1024);
        let read = buf.to_vec();
        assert_eq!(read, data);
    }
}
