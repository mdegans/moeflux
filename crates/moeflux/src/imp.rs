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

    /// Apply rotary position embedding to Q and K at `pos`. Both
    /// buffers are mutated in place. Diff-oracle dump point for the
    /// RoPE kernel.
    pub fn apply_rotary_emb(
        &self,
        pos: i32,
        q: &mut [f32],
        k: &mut [f32],
    ) -> Result<(), Error> {
        let rc = unsafe {
            sys::mf_apply_rotary_emb(
                self.inner.as_ptr(),
                pos,
                q.as_mut_ptr(),
                k.as_mut_ptr(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(Error::EvalFailed) }
    }

    /// Per-head CPU RMSNorm, mutating `x_inout` in place. Layout is
    /// `num_heads` contiguous head slices of `head_dim` floats each;
    /// the same `head_dim`-long bf16 weight (from `weight_name`) is
    /// applied to every head. Diff-oracle dump point for the per-head
    /// Q/K norm used inside attention layers.
    pub fn rms_norm_per_head_cpu(
        &self,
        weight_name: &str,
        num_heads: usize,
        head_dim: usize,
        x_inout: &mut [f32],
    ) -> Result<(), Error> {
        let cname =
            std::ffi::CString::new(weight_name).map_err(|_| Error::PathHasNul)?;
        let rc = unsafe {
            sys::mf_rms_norm_per_head_cpu(
                self.inner.as_ptr(),
                cname.as_ptr(),
                num_heads as i32,
                head_dim as i32,
                x_inout.as_mut_ptr(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(Error::EvalFailed) }
    }

    /// CPU scaled dot-product attention with sigmoid-gated output,
    /// single query position against `kv_len` cached positions. Uses
    /// the active variant's NUM_ATTN_HEADS / NUM_KV_HEADS / HEAD_DIM
    /// (GQA: `num_attn_heads / num_kv_heads` query heads share one
    /// kv head). `out` is overwritten. Diff-oracle dump point for
    /// the SDPA core of full-attention layers.
    pub fn sdpa_cpu(
        &self,
        kv_len: i32,
        q: &[f32],
        q_gate: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        out: &mut [f32],
    ) -> Result<(), Error> {
        let rc = unsafe {
            sys::mf_sdpa_cpu(
                self.inner.as_ptr(),
                kv_len,
                q.as_ptr(),
                q_gate.as_ptr(),
                k_cache.as_ptr(),
                v_cache.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(Error::EvalFailed) }
    }

    /// CPU LM head matvec. `x` is `HIDDEN_DIM` floats, `out` is
    /// `VOCAB_SIZE` floats. Diff-oracle dump point for the final
    /// logits projection; routes through `cpu_dequant_matvec`
    /// regardless of Metal availability so the diff harness compares
    /// CPU outputs head-on.
    pub fn lm_head_cpu(&self, x: &[f32], out: &mut [f32]) -> Result<(), Error> {
        let rc = unsafe {
            sys::mf_lm_head_cpu(
                self.inner.as_ptr(),
                x.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(Error::EvalFailed) }
    }

    /// CPU MoE router: softmax → top-K → normalize. `scores` is the
    /// raw gate logit vector (mutated in place — afterwards holds
    /// softmax probabilities). `indices` and `weights` are output
    /// parallel arrays both of length `k`; the slot order matches the
    /// C selection-sort and is not sorted by score. Diff-oracle dump
    /// point for the per-layer expert-routing decision.
    pub fn moe_router_cpu(
        &self,
        scores: &mut [f32],
        k: usize,
        indices: &mut [i32],
        weights: &mut [f32],
    ) -> Result<(), Error> {
        let rc = unsafe {
            sys::mf_moe_router_cpu(
                self.inner.as_ptr(),
                scores.as_mut_ptr(),
                scores.len() as i32,
                k as i32,
                indices.as_mut_ptr(),
                weights.as_mut_ptr(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(Error::EvalFailed) }
    }

    /// Depthwise 1D conv step + SiLU. `weight_name` is a bf16 tensor of
    /// length `channels * kernel_size`. Diff-oracle dump point for the
    /// linear-attention conv1d primitive.
    pub fn conv1d_step_cpu(
        &self,
        weight_name: &str,
        channels: usize,
        kernel_size: usize,
        conv_state: &[f32],
        new_input: &[f32],
        out: &mut [f32],
    ) -> Result<(), Error> {
        let cname =
            std::ffi::CString::new(weight_name).map_err(|_| Error::PathHasNul)?;
        let rc = unsafe {
            sys::mf_conv1d_step_cpu(
                self.inner.as_ptr(),
                cname.as_ptr(),
                channels as i32,
                kernel_size as i32,
                conv_state.as_ptr(),
                new_input.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(Error::EvalFailed) }
    }

    /// CPU bare RMSNorm (no weight). Diff-oracle dump point for the
    /// per-head Q/K bare-norm inside linear-attention layers.
    pub fn rms_norm_bare_cpu(
        &self,
        eps: f32,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), Error> {
        let rc = unsafe {
            sys::mf_rms_norm_bare_cpu(
                self.inner.as_ptr(),
                x.len() as i32,
                eps,
                x.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(Error::EvalFailed) }
    }

    /// CPU RMSNormGated: rms_norm(x) × weight × silu(z). Diff-oracle
    /// dump point for the gated-norm tail of the linear-attention
    /// recurrence output.
    pub fn rms_norm_gated_cpu(
        &self,
        weight_name: &str,
        eps: f32,
        x: &[f32],
        z: &[f32],
        out: &mut [f32],
    ) -> Result<(), Error> {
        let cname =
            std::ffi::CString::new(weight_name).map_err(|_| Error::PathHasNul)?;
        let rc = unsafe {
            sys::mf_rms_norm_gated_cpu(
                self.inner.as_ptr(),
                cname.as_ptr(),
                x.len() as i32,
                eps,
                x.as_ptr(),
                z.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(Error::EvalFailed) }
    }

    /// CPU gated-delta-net recurrence step. Loads `A_log` and
    /// `dt_bias` from the manifest for `layer_idx`. Diff-oracle dump
    /// point for the heart of GatedDeltaNet — the production decode
    /// path uses the GPU `gated_delta_net_step` shader; the diff
    /// oracle exercises the parallel CPU helper that mirrors the same
    /// arithmetic.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_recurrence_cpu(
        &self,
        layer_idx: usize,
        alpha: &[f32],
        beta: &[f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        v_heads: usize,
        k_heads: usize,
        key_dim: usize,
        value_dim: usize,
        ssm_state: &mut [f32],
        out_values: &mut [f32],
    ) -> Result<(), Error> {
        let rc = unsafe {
            sys::mf_gated_delta_recurrence_cpu(
                self.inner.as_ptr(),
                layer_idx as i32,
                alpha.as_ptr(),
                beta.as_ptr(),
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                v_heads as i32,
                k_heads as i32,
                key_dim as i32,
                value_dim as i32,
                ssm_state.as_mut_ptr(),
                out_values.as_mut_ptr(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(Error::EvalFailed) }
    }

    /// Single-expert GPU FFN forward. Wraps the C-side
    /// `mf_gpu_expert_forward` hook. `expert_data` must be exactly
    /// `EXPERT_SIZE` bytes for the active 4-bit variant — the C path
    /// rejects 2-bit ctxs (those use a different pipeline + layout).
    /// `h_post` is HIDDEN_DIM floats; `expert_out` receives HIDDEN_DIM
    /// floats. Diff-oracle dump point for the GPU expert FFN.
    pub fn gpu_expert_forward(
        &self,
        expert_data: &[u8],
        h_post: &[f32],
        expert_out: &mut [f32],
    ) -> Result<(), Error> {
        let rc = unsafe {
            sys::mf_gpu_expert_forward(
                self.inner.as_ptr(),
                expert_data.as_ptr().cast(),
                expert_data.len(),
                h_post.as_ptr(),
                expert_out.as_mut_ptr(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(Error::EvalFailed) }
    }

    /// Batched K-expert FFN forward + GPU combine. Wraps
    /// `mf_gpu_batched_experts_forward`. `expert_data` is K expert
    /// blobs concatenated in slot order (K × `EXPERT_SIZE` bytes).
    /// `h_post`, `h_mid`, `shared_out`, `hidden_out` are HIDDEN_DIM
    /// floats; `expert_weights` is K floats. `actual_k` must satisfy
    /// `1 ≤ actual_k ≤ MAX_K=16`. Diff-oracle dump point for the
    /// production MoE expert + combine path.
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_batched_experts_forward(
        &self,
        actual_k: i32,
        expert_data: &[u8],
        h_post: &[f32],
        h_mid: &[f32],
        shared_out: &[f32],
        expert_weights: &[f32],
        shared_gate_score: f32,
        hidden_out: &mut [f32],
    ) -> Result<(), Error> {
        let rc = unsafe {
            sys::mf_gpu_batched_experts_forward(
                self.inner.as_ptr(),
                actual_k,
                expert_data.as_ptr().cast(),
                expert_data.len(),
                h_post.as_ptr(),
                h_mid.as_ptr(),
                shared_out.as_ptr(),
                expert_weights.as_ptr(),
                shared_gate_score,
                hidden_out.as_mut_ptr(),
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
