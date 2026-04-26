//! macOS implementation.

use std::ffi::{CStr, CString};
use std::fmt;
use std::path::Path;
use std::ptr::NonNull;

use moeflux_sys as sys;

/// Errors from the moeflux FFI layer.
#[derive(Debug)]
pub enum Error {
    /// A path contained an interior NUL byte — cannot be passed to C.
    PathHasNul,
    /// `mf_init_model` returned NULL. Typical causes: missing file,
    /// mmap failure, vocab parse failure, Metal unavailable. moeflux
    /// currently does not distinguish these — check stderr for the C
    /// side's diagnostic.
    InitFailed,
    /// `mf_eval_*` returned nonzero.
    EvalFailed,
    /// `mf_state_save` returned -1 or `mf_state_load` rejected the
    /// buffer. The most common cause is a shape-header mismatch
    /// (snapshot taken on a different model variant) or truncated
    /// buffer.
    StateFailed,
    /// The caller-supplied buffer was too small for the snapshot.
    /// Call [`Ctx::state_size`] to get the required size.
    StateBufferTooSmall {
        /// Bytes the caller provided.
        have: usize,
        /// Bytes the snapshot requires.
        need: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PathHasNul => f.write_str("path contained an interior NUL byte"),
            Self::InitFailed => f.write_str("moeflux: mf_init_model returned NULL"),
            Self::EvalFailed => f.write_str("moeflux: mf_eval_* returned nonzero"),
            Self::StateFailed => f.write_str("moeflux: state save/load failed"),
            Self::StateBufferTooSmall { have, need } => write!(
                f,
                "moeflux: state buffer too small (have {have}, need {need})"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// A loaded moeflux model context. Drops via `mf_free_model`.
pub struct Ctx {
    inner: NonNull<sys::mf_ctx>,
}

// The C side is single-threaded per ctx but the struct contains no
// thread-local state that would prevent move across threads — the
// Metal context is a process global, not a ctx field. So Send is OK
// under moeflux.h's contract. Not Sync: concurrent calls on the same
// ctx are explicitly undefined.
unsafe impl Send for Ctx {}

impl Ctx {
    /// Load a model and initialize the Metal backend.
    ///
    /// - `weights`: `model_weights.bin` produced by `extract_weights.py`.
    /// - `manifest`: `model_weights.json` sibling of the above.
    /// - `vocab`: `vocab.bin` produced by `export_vocab.py`.
    /// - `experts_dir`: directory containing a `packed_experts/`
    ///   subdirectory with `layer_XX.bin` files.
    /// - `experts_per_tok`: `K` — how many experts to route to per
    ///   token at inference time. Must be `<= NUM_EXPERTS_PER_TOK` of
    ///   the compile-time variant.
    /// - `use_2bit`: select the 2-bit packed-experts layout (reads
    ///   `packed_experts_2bit/` instead of `packed_experts/`).
    pub fn open(
        weights: &Path,
        manifest: &Path,
        vocab: &Path,
        experts_dir: &Path,
        experts_per_tok: u32,
        use_2bit: bool,
    ) -> Result<Self, Error> {
        // Point the Metal backend at shaders.metal via env var unless
        // the caller has already set it. The compile-time default
        // path comes from moeflux-sys's build.rs, so this works from
        // any cwd / downstream crate without extra configuration.
        // SAFETY: set_var is safe in the single-threaded init path
        // before any Ctx is created; if a user later spawns threads
        // and races open, they should set MOEFLUX_SHADERS_PATH
        // themselves before launching.
        if std::env::var_os("MOEFLUX_SHADERS_PATH").is_none() {
            // set_var became unsafe in edition 2024; the guard above
            // documents our invariant.
            unsafe {
                std::env::set_var(
                    "MOEFLUX_SHADERS_PATH",
                    sys::DEFAULT_SHADERS_PATH,
                );
            }
        }

        let weights = path_to_c(weights)?;
        let manifest = path_to_c(manifest)?;
        let vocab = path_to_c(vocab)?;
        let experts_dir = path_to_c(experts_dir)?;

        // SAFETY: all four CStrings live until the call returns; the
        // FFI copies what it needs. experts_per_tok is clamped to a
        // reasonable u32; mf_init_model's C signature is signed int,
        // hence the `as i32`. use_2bit maps to C int.
        let raw = unsafe {
            sys::mf_init_model(
                weights.as_ptr(),
                manifest.as_ptr(),
                vocab.as_ptr(),
                experts_dir.as_ptr(),
                experts_per_tok as i32,
                i32::from(use_2bit),
            )
        };
        NonNull::new(raw).map(|inner| Self { inner }).ok_or(Error::InitFailed)
    }

    /// Vocabulary size. Logit buffers must be at least this long.
    pub fn n_vocab(&self) -> usize {
        // SAFETY: self.inner is a valid ctx until Drop.
        unsafe { sys::mf_n_vocab(self.inner.as_ptr()) }
    }

    /// Maximum sequence length the model supports.
    pub fn n_ctx(&self) -> usize {
        unsafe { sys::mf_n_ctx(self.inner.as_ptr()) }
    }

    /// Canonical EOS token id for this model.
    pub fn eos(&self) -> i32 {
        unsafe { sys::mf_eos(self.inner.as_ptr()) }
    }

    /// Static display name of the compiled-in model variant.
    pub fn model_name(&self) -> &'static str {
        // SAFETY: moeflux returns a pointer to static storage;
        // lifetime is bound to the process. UTF-8 by construction.
        let cptr = unsafe { sys::mf_model_name(self.inner.as_ptr()) };
        if cptr.is_null() {
            return "";
        }
        unsafe { CStr::from_ptr(cptr) }.to_str().unwrap_or("")
    }

    /// Prefill the KV cache with `tokens` starting at `start_pos`,
    /// then fill `logits` with the next-token distribution.
    ///
    /// `logits` must hold at least [`Self::n_vocab`] elements.
    pub fn eval_prompt(
        &mut self,
        tokens: &[i32],
        start_pos: usize,
        seq_id: i32,
        logits: &mut [f32],
    ) -> Result<(), Error> {
        assert!(
            logits.len() >= self.n_vocab(),
            "logits buffer too small: {} < n_vocab={}",
            logits.len(),
            self.n_vocab()
        );
        // SAFETY: tokens/logits slices live for the duration of the
        // call; moeflux reads tokens, writes at most n_vocab floats
        // into logits.
        let rc = unsafe {
            sys::mf_eval_prompt(
                self.inner.as_ptr(),
                tokens.as_ptr(),
                tokens.len(),
                start_pos,
                seq_id,
                logits.as_mut_ptr(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(Error::EvalFailed) }
    }

    /// Decode a single token at `pos`, writing next-token logits.
    pub fn eval_token(
        &mut self,
        token: i32,
        pos: usize,
        seq_id: i32,
        logits: &mut [f32],
    ) -> Result<(), Error> {
        assert!(
            logits.len() >= self.n_vocab(),
            "logits buffer too small: {} < n_vocab={}",
            logits.len(),
            self.n_vocab()
        );
        let rc = unsafe {
            sys::mf_eval_token(
                self.inner.as_ptr(),
                token,
                pos,
                seq_id,
                logits.as_mut_ptr(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(Error::EvalFailed) }
    }

    /// Embed a single token via the C path's `embed_lookup`. Writes
    /// `HIDDEN_DIM` floats into `out`. Used by the RIIR diff oracle as
    /// the per-layer dump point for the embedding kernel; not part of
    /// production decode.
    pub fn embed(&self, token_id: i32, out: &mut [f32]) -> Result<(), Error> {
        let rc = unsafe {
            sys::mf_embed_lookup(self.inner.as_ptr(), token_id, out.as_mut_ptr())
        };
        if rc == 0 { Ok(()) } else { Err(Error::EvalFailed) }
    }

    /// CPU RMSNorm against the weight tensor named `weight_name`
    /// (typically `model.norm.weight` or one of the per-layer
    /// `*.input_layernorm.weight` / `*.post_attention_layernorm.weight`
    /// tensors). Diff-oracle dump point for the RMSNorm kernel.
    pub fn rms_norm_cpu(
        &self,
        weight_name: &str,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), Error> {
        let cname =
            std::ffi::CString::new(weight_name).map_err(|_| Error::PathHasNul)?;
        let rc = unsafe {
            sys::mf_rms_norm_cpu(
                self.inner.as_ptr(),
                cname.as_ptr(),
                x.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(Error::EvalFailed) }
    }

    /// Reset the sequence to empty.
    pub fn memory_clear(&mut self) {
        unsafe { sys::mf_memory_clear(self.inner.as_ptr()) }
    }

    /// Remove positions `[p0, p1)` from `seq_id`. Pass `p1 = -1` for
    /// "to end of sequence". Returns `true` on success.
    pub fn memory_seq_rm(&mut self, seq_id: i32, p0: i32, p1: i32) -> bool {
        (unsafe { sys::mf_memory_seq_rm(self.inner.as_ptr(), seq_id, p0, p1) }) != 0
    }

    /// Highest occupied position + 1 in `seq_id`, or 0 if empty.
    pub fn memory_seq_pos_max(&self, seq_id: i32) -> i32 {
        unsafe { sys::mf_memory_seq_pos_max(self.inner.as_ptr(), seq_id) }
    }

    /// Current snapshot size in bytes. Pass to [`Self::state_save`].
    pub fn state_size(&self) -> usize {
        unsafe { sys::mf_state_size(self.inner.as_ptr()) }
    }

    /// Serialize the current state into `buf`. Returns the number of
    /// bytes written.
    pub fn state_save(&self, buf: &mut [u8]) -> Result<usize, Error> {
        let need = self.state_size();
        if buf.len() < need {
            return Err(Error::StateBufferTooSmall { have: buf.len(), need });
        }
        let rc = unsafe {
            sys::mf_state_save(
                self.inner.as_ptr(),
                buf.as_mut_ptr().cast(),
                buf.len(),
            )
        };
        if rc < 0 {
            Err(Error::StateFailed)
        } else {
            Ok(rc as usize)
        }
    }

    /// Restore a snapshot previously produced by [`Self::state_save`].
    /// On failure the ctx state is undefined; call
    /// [`Self::memory_clear`] before continuing.
    pub fn state_load(&mut self, buf: &[u8]) -> Result<(), Error> {
        let rc = unsafe {
            sys::mf_state_load(
                self.inner.as_ptr(),
                buf.as_ptr().cast(),
                buf.len(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(Error::StateFailed) }
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        // SAFETY: mf_free_model takes ownership of the ctx. We hold
        // a NonNull that we never clone, so this runs exactly once.
        unsafe { sys::mf_free_model(self.inner.as_ptr()) }
    }
}

impl fmt::Debug for Ctx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ctx")
            .field("model", &self.model_name())
            .field("n_vocab", &self.n_vocab())
            .finish()
    }
}

fn path_to_c(p: &Path) -> Result<CString, Error> {
    // macOS paths are bytes; go through OsStr → bytes rather than
    // requiring UTF-8. CString::new fails only on interior NULs.
    use std::os::unix::ffi::OsStrExt;
    CString::new(p.as_os_str().as_bytes()).map_err(|_| Error::PathHasNul)
}
