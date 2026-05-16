//! GPU command-encoder ergonomics.
//!
//! Phase 4 of the cleanup arc. Houses two things:
//!
//! - [`pipeline_bundle!`], the macro that generates the crate's
//!   pipeline-bundle structs.
//! - [`ComputeEncoder`], the builder that wraps a Metal compute
//!   command encoder so kernel modules stop hand-rolling the
//!   `set_compute_pipeline_state` → `set_buffer` → `set_bytes` →
//!   `dispatch_thread_groups` → `end_encoding` recipe.

use metal::{
    BufferRef, CommandBufferRef, ComputeCommandEncoderRef, ComputePipelineStateRef, MTLSize,
};

/// Generate a pipeline-bundle struct and its `fetch` constructor.
///
/// Every field is a `metal::ComputePipelineState` fetched by kernel
/// name; `fetch` builds the whole bundle. Replaces the hand-rolled
/// `pub struct … {} impl … { pub fn fetch … }` boilerplate that
/// recurred across the GPU kernel modules.
///
/// ```ignore
/// pipeline_bundle! {
///     /// Optional struct doc comment.
///     pub struct RmsNormBf16Pipelines {
///         sum   => "rms_norm_sum_sq",
///         /// Optional field doc comment.
///         apply => "rms_norm_apply_bf16",
///     }
/// }
/// ```
macro_rules! pipeline_bundle {
    (
        $(#[$struct_meta:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[$field_meta:meta])*
                $field:ident => $kernel:literal
            ),+ $(,)?
        }
    ) => {
        $(#[$struct_meta])*
        $vis struct $name {
            $(
                $(#[$field_meta])*
                pub $field: ::metal::ComputePipelineState,
            )+
        }

        impl $name {
            /// Fetch every kernel pipeline by name. `MetalContext`
            /// caches compiled pipelines, so repeat calls are O(1).
            pub fn fetch(
                metal: &mut $crate::riir::backend::gpu::metal::MetalContext,
            ) -> ::core::result::Result<
                Self,
                $crate::riir::backend::gpu::metal::MetalError,
            > {
                ::core::result::Result::Ok(Self {
                    $( $field: metal.pipeline($kernel)?.clone(), )+
                })
            }
        }
    };
}

pub(crate) use pipeline_bundle;

/// Ergonomic builder over a Metal compute command encoder.
///
/// Replaces the hand-rolled `set_compute_pipeline_state` → N×
/// `set_buffer` → N× `set_bytes` → `dispatch_thread_groups` →
/// `end_encoding` recipe that recurred across every GPU kernel
/// module.
///
/// The builder *borrows* an encoder — [`begin`](Self::begin) creates
/// one on a command buffer, [`wrap`](Self::wrap) adopts one the
/// caller already holds. It does not own the command buffer.
///
/// Lifecycle: [`dispatch`](Self::dispatch) is **re-callable** — a
/// single encoder may drive several dispatches (the batched-expert
/// path folds two matvecs onto one encoder). Encoding closes on an
/// explicit [`end`](Self::end) *or* on `Drop`; whichever comes first,
/// `end_encoding` is called exactly once.
///
/// # Index discipline
///
/// A wrong `set_buffer` *index* is silent GPU corruption — wrong
/// math, no panic. The chaining API keeps each index visually
/// adjacent to its value so a transposed pair is easier to catch in
/// review; [`bytes`](Self::bytes) additionally computes the byte
/// length from the value's type, so a host/kernel width mismatch
/// cannot slip in unnoticed.
pub(crate) struct ComputeEncoder<'e> {
    enc: &'e ComputeCommandEncoderRef,
    ended: bool,
}

impl<'e> ComputeEncoder<'e> {
    /// Adopt an encoder the caller already created. Use this when the
    /// caller owns lifecycle — e.g. a helper that encodes one of
    /// several dispatches onto a shared encoder.
    pub fn wrap(enc: &'e ComputeCommandEncoderRef) -> Self {
        Self { enc, ended: false }
    }

    /// Create a fresh compute encoder on `cmdbuf` and wrap it. The
    /// common single-dispatch case. The encoder borrows `cmdbuf`'s
    /// autorelease pool, so the builder cannot outlive `cmdbuf`.
    pub fn begin(cmdbuf: &'e CommandBufferRef) -> Self {
        Self::wrap(cmdbuf.new_compute_command_encoder())
    }

    /// Bind the compute pipeline. Called once, first.
    pub fn pipeline(&mut self, state: &ComputePipelineStateRef) -> &mut Self {
        self.enc.set_compute_pipeline_state(state);
        self
    }

    /// Bind `buffer` at argument `index` with byte `offset`. Every
    /// call site passes a real buffer — `None` is not modeled.
    pub fn buffer(&mut self, index: u64, buffer: &BufferRef, offset: u64) -> &mut Self {
        self.enc.set_buffer(index, Some(buffer), offset);
        self
    }

    /// Bind a small POD scalar at argument `index` as a kernel
    /// `constant`.
    ///
    /// Replaces the `set_bytes(i, 4, (&v as *const T).cast())`
    /// pointer dance. The byte length is `size_of::<T>()`, *computed*
    /// — so a host value whose width does not match the kernel's
    /// expected constant cannot silently mis-copy.
    ///
    /// `T: Copy` rules out drop glue; `'static` rules out any
    /// borrowed reference inside `T`. Together they guarantee `value`
    /// is a self-contained bit-pattern safe to copy verbatim.
    pub fn bytes<T: Copy + 'static>(&mut self, index: u64, value: &T) -> &mut Self {
        debug_assert!(
            std::mem::size_of::<T>() <= 4096,
            "set_bytes is for small inline constants, not buffers",
        );
        // SAFETY: `T: Copy + 'static` ⇒ `value` is a self-contained
        // POD bit-pattern with no drop glue or interior references.
        // `set_bytes` copies `size_of::<T>()` bytes out immediately
        // (it does not retain the pointer), and `value` is a live
        // borrow for the whole call — so the read is in-bounds and
        // non-dangling.
        self.enc.set_bytes(
            index,
            std::mem::size_of::<T>() as u64,
            (value as *const T).cast::<std::ffi::c_void>(),
        );
        self
    }

    /// Encode one threadgroup dispatch. Re-callable: a single encoder
    /// may drive multiple dispatches. Returns `&mut self` — does not
    /// consume the builder and does not end encoding.
    pub fn dispatch(&mut self, grid: MTLSize, threadgroup: MTLSize) -> &mut Self {
        self.enc.dispatch_thread_groups(grid, threadgroup);
        self
    }

    /// Close encoding explicitly. Idempotent — a second `end`, or a
    /// `Drop` after `end`, is a no-op. Use this only when later code
    /// in the same scope must run after encoding closes (e.g.
    /// `cmdbuf.commit()`); otherwise let `Drop` close the encoder.
    pub fn end(&mut self) {
        if !self.ended {
            self.enc.end_encoding();
            self.ended = true;
        }
    }
}

impl Drop for ComputeEncoder<'_> {
    /// End encoding if the caller did not. Calling `end_encoding`
    /// twice on one Metal encoder is an API violation — the `ended`
    /// flag guarantees exactly one call.
    fn drop(&mut self) {
        self.end();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::riir::backend::gpu::metal::MetalContext;

    /// `end()` then `Drop` must call `end_encoding` exactly once. A
    /// double `end_encoding` on a live Metal encoder is an API
    /// violation; the `ended` flag is what prevents it. The test
    /// exercises the guarded path — `end()`, a redundant `end()`, and
    /// then `Drop` — and asserts the flag tracks state. Reaching the
    /// end of the test (process intact) is itself the assertion that
    /// no double-end occurred.
    #[test]
    fn end_then_drop_is_single_end() {
        let mut metal = match MetalContext::new() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[encoder] skipping: Metal init failed: {e:?}");
                return;
            }
        };
        let cmdbuf = metal.queue().new_command_buffer();
        let mut ce = ComputeEncoder::begin(cmdbuf);
        assert!(!ce.ended, "fresh encoder must not be ended");
        ce.end();
        assert!(ce.ended, "end() must mark the encoder ended");
        ce.end();
        assert!(ce.ended, "redundant end() must stay a no-op");
        drop(ce); // Drop after end() — must not end_encoding again.
    }

    /// An encoder dropped without an explicit `end()` must still be
    /// closed by `Drop`. Calling `end_encoding` once on a live
    /// encoder is valid; the test asserts that path does not abort.
    #[test]
    fn drop_without_explicit_end_still_ends() {
        let mut metal = match MetalContext::new() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[encoder] skipping: Metal init failed: {e:?}");
                return;
            }
        };
        let cmdbuf = metal.queue().new_command_buffer();
        {
            let _ce = ComputeEncoder::begin(cmdbuf);
            // No explicit end() — Drop must close the encoder.
        }
    }
}
