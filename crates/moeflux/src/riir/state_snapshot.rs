//! Per-Ctx state snapshot — Phase 4g port of `mf_state_save` /
//! `mf_state_load` (infer.m:8525..8700).
//!
//! Wire format (little-endian, fixed by the C side):
//!
//! - **Header** (8 × `u32`):
//!   - `magic = 0x4D464C58` (`'MFLX'`)
//!   - `version = 1`
//!   - `num_layers`
//!   - `full_attn_interval`
//!   - `num_kv_heads`
//!   - `head_dim`
//!   - `linear_conv_bytes` =
//!     `(CONV_KERNEL_SIZE - 1) * linear_conv_dim() * 4`
//!   - `linear_ssm_bytes` =
//!     `linear_num_v_heads * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * 4`
//! - **Per-layer body** (in 0..num_layers order):
//!   - Full-attn layer: `i32 len`, then `len * fa_stride` bytes K
//!     followed by the same many bytes V (where `fa_stride =
//!     num_kv_heads * head_dim * 4`).
//!   - Linear-attn layer: `linear_conv_bytes` of conv_state then
//!     `linear_ssm_bytes` of ssm_state.
//!
//! The Rust port uses GPU buffers as the canonical store for the
//! linear-attn recurrence (`linear_buffers.conv_state[i]` /
//! `delta_state[i]`); save reads them back, load memcpys into them.
//! This matches C's Metal-fast-path layout — see the C path's
//! `g_metal->buf_conv_state[linear_idx]` writes / reads in
//! infer.m:8568..8595 / 8678..8693. The CPU-fallback path the C side
//! also handles (infer.m:8579..8593) doesn't apply: there is no CPU
//! fallback in the Rust port.
//!
//! ## Concurrency contract
//!
//! Mirror of moeflux.h:481-487 — call only at token boundaries
//! (after `eval_prompt` / `eval_token` returns). Pending deferred
//! GPU expert compute must have finalized; this is enforced by the
//! `step_internal` orchestrator draining `RsCtx::deferred` before
//! returning, but a defensive `discard_deferred_experts_in` in
//! `state_save` keeps the contract robust against a buggy caller.

use metal::Buffer;

use super::deferred;
use super::linear_attn_forward::{
    full_attn_layer_idx_for, linear_layer_idx_for, LayerForwardBuffers,
};
use super::state::LayerState;
use super::variants::{LayerKind, Variant, GPU_KV_SEQ, VARIANT};

/// `'MFLX'` little-endian. Wire-compatible with the C side's
/// `MF_SNAPSHOT_MAGIC` (infer.m:8487).
pub const SNAPSHOT_MAGIC: u32 = 0x4D464C58;
/// Format version. Bumped when wire layout changes; load() rejects
/// any version it doesn't understand.
pub const SNAPSHOT_VERSION: u32 = 1;
/// Header is exactly this many `u32` fields. Mirrors C
/// `MF_SNAPSHOT_HEADER_U32`.
pub const SNAPSHOT_HEADER_U32: usize = 8;

/// Errors from the snapshot save/load path.
#[derive(Debug, thiserror::Error)]
pub enum StateSnapshotError {
    /// Caller-supplied buffer too small for the snapshot.
    #[error("buffer too small: need {need} bytes, got {got}")]
    BufferTooSmall { need: usize, got: usize },
    /// Header magic doesn't match.
    #[error("snapshot magic mismatch (got 0x{got:08X}, want 0x{want:08X})")]
    BadMagic { got: u32, want: u32 },
    /// Header version doesn't match.
    #[error("snapshot version mismatch (got {got}, want {want})")]
    BadVersion { got: u32, want: u32 },
    /// Header shape constants don't match the active build.
    #[error("snapshot shape mismatch on field '{field}' (got {got}, want {want})")]
    ShapeMismatch {
        field: &'static str,
        got: u32,
        want: u32,
    },
    /// Per-layer length value is negative.
    #[error("snapshot layer {layer} has negative KV length {len}")]
    NegativeLen { layer: usize, len: i32 },
    /// Per-layer length exceeds the architectural maximum.
    #[error(
        "snapshot layer {layer} KV length {len} exceeds MAX_SEQ_LEN={max}"
    )]
    LenOverflow { layer: usize, len: i32, max: usize },
    /// Truncated payload (preflight failure during load).
    #[error("snapshot truncated at layer {layer}: need {need} more bytes, got {got}")]
    Truncated {
        layer: usize,
        need: usize,
        got: usize,
    },
    /// linear_buffers must exist before save/load. Set up by
    /// `RsCtx::ensure_linear_resources` on first eval.
    #[error("linear_buffers not initialized — call eval_prompt or memory_clear first")]
    BuffersNotReady,
}

#[inline]
fn full_attn_stride_bytes(v: &Variant) -> usize {
    v.num_kv_heads * v.head_dim * std::mem::size_of::<f32>()
}

#[inline]
fn linear_conv_bytes(v: &Variant) -> usize {
    (Variant::CONV_KERNEL_SIZE - 1)
        * v.linear_conv_dim()
        * std::mem::size_of::<f32>()
}

#[inline]
fn linear_ssm_bytes(v: &Variant) -> usize {
    v.linear_num_v_heads
        * Variant::LINEAR_VALUE_DIM
        * Variant::LINEAR_KEY_DIM
        * std::mem::size_of::<f32>()
}

/// Bytes the caller must allocate to hold the current snapshot.
/// Mirrors C `mf_state_size` (infer.m:8505..8523). Re-query after
/// every evaluation: the value grows with KV length.
pub fn state_size(layer_states: &[LayerState]) -> usize {
    let v = VARIANT;
    let mut n = SNAPSHOT_HEADER_U32 * std::mem::size_of::<u32>();
    let fa_stride = full_attn_stride_bytes(&v);
    let la_conv = linear_conv_bytes(&v);
    let la_ssm = linear_ssm_bytes(&v);
    for (i, layer) in layer_states.iter().enumerate().take(v.num_layers) {
        match v.layer_kind(i) {
            LayerKind::FullAttn => {
                n += std::mem::size_of::<i32>();
                let len = match layer {
                    LayerState::FullAttn(kv) => kv.len.max(0) as usize,
                    LayerState::LinearAttn(_) => 0,
                };
                n += 2 * len * fa_stride;
            }
            LayerKind::LinearAttn => {
                n += la_conv + la_ssm;
            }
        }
    }
    n
}

/// Serialize the current state into `buf`. Returns the number of
/// bytes written. Mirrors C `mf_state_save` (infer.m:8525..8597).
///
/// Reads the linear-attn GPU recurrence buffers back to host as
/// part of writing the body. Caller must ensure no GPU work is in
/// flight against those buffers — the public `RsCtx::state_save`
/// wrapper enforces this by draining any pending deferred dispatch
/// first.
pub fn state_save(
    buf: &mut [u8],
    layer_states: &[LayerState],
    linear_buffers: &LayerForwardBuffers,
) -> Result<usize, StateSnapshotError> {
    let v = VARIANT;
    let need = state_size(layer_states);
    if buf.len() < need {
        return Err(StateSnapshotError::BufferTooSmall {
            need,
            got: buf.len(),
        });
    }

    let fa_stride = full_attn_stride_bytes(&v);
    let la_conv = linear_conv_bytes(&v);
    let la_ssm = linear_ssm_bytes(&v);

    let mut off = 0usize;

    // Header.
    let header: [u32; SNAPSHOT_HEADER_U32] = [
        SNAPSHOT_MAGIC,
        SNAPSHOT_VERSION,
        v.num_layers as u32,
        v.full_attn_interval as u32,
        v.num_kv_heads as u32,
        v.head_dim as u32,
        la_conv as u32,
        la_ssm as u32,
    ];
    for word in header.iter() {
        buf[off..off + 4].copy_from_slice(&word.to_le_bytes());
        off += 4;
    }

    // Per-layer body.
    for (i, layer) in layer_states.iter().enumerate().take(v.num_layers) {
        match v.layer_kind(i) {
            LayerKind::FullAttn => {
                let kv = match layer {
                    LayerState::FullAttn(kv) => kv,
                    LayerState::LinearAttn(_) => {
                        return Err(StateSnapshotError::ShapeMismatch {
                            field: "layer_state_kind",
                            got: 0,
                            want: 1,
                        });
                    }
                };
                let len = kv.len.max(0);
                buf[off..off + 4].copy_from_slice(&len.to_le_bytes());
                off += 4;
                if len > 0 {
                    let bytes = (len as usize) * fa_stride;
                    let k_src = unsafe {
                        std::slice::from_raw_parts(
                            kv.k_cache.as_ptr() as *const u8,
                            bytes,
                        )
                    };
                    let v_src = unsafe {
                        std::slice::from_raw_parts(
                            kv.v_cache.as_ptr() as *const u8,
                            bytes,
                        )
                    };
                    buf[off..off + bytes].copy_from_slice(k_src);
                    off += bytes;
                    buf[off..off + bytes].copy_from_slice(v_src);
                    off += bytes;
                }
            }
            LayerKind::LinearAttn => {
                let linear_idx = linear_layer_idx_for(i)
                    .expect("layer_kind says LinearAttn");
                read_buffer_bytes(
                    &linear_buffers.conv_state[linear_idx],
                    &mut buf[off..off + la_conv],
                );
                off += la_conv;
                read_buffer_bytes(
                    &linear_buffers.delta_state[linear_idx],
                    &mut buf[off..off + la_ssm],
                );
                off += la_ssm;
            }
        }
    }

    debug_assert_eq!(off, need, "state_save wrote {off} bytes, expected {need}");
    Ok(off)
}

/// Replace current state with the one encoded in `buf`. Mirrors C
/// `mf_state_load` (infer.m:8599..8700). Two-pass: preflight verifies
/// the header + per-layer lengths fit in `buf` before any state is
/// mutated; restore then performs the writes.
pub fn state_load(
    buf: &[u8],
    layer_states: &mut [LayerState],
    linear_buffers: &mut LayerForwardBuffers,
) -> Result<(), StateSnapshotError> {
    let v = VARIANT;
    let header_bytes = SNAPSHOT_HEADER_U32 * std::mem::size_of::<u32>();
    if buf.len() < header_bytes {
        return Err(StateSnapshotError::Truncated {
            layer: 0,
            need: header_bytes,
            got: buf.len(),
        });
    }

    let read_u32 = |off: usize| -> u32 {
        u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
    };

    let magic = read_u32(0);
    if magic != SNAPSHOT_MAGIC {
        return Err(StateSnapshotError::BadMagic {
            got: magic,
            want: SNAPSHOT_MAGIC,
        });
    }
    let version = read_u32(4);
    if version != SNAPSHOT_VERSION {
        return Err(StateSnapshotError::BadVersion {
            got: version,
            want: SNAPSHOT_VERSION,
        });
    }
    let check = |off: usize, field: &'static str, want: u32| -> Result<(), StateSnapshotError> {
        let got = read_u32(off);
        if got != want {
            return Err(StateSnapshotError::ShapeMismatch { field, got, want });
        }
        Ok(())
    };
    check(8, "num_layers", v.num_layers as u32)?;
    check(12, "full_attn_interval", v.full_attn_interval as u32)?;
    check(16, "num_kv_heads", v.num_kv_heads as u32)?;
    check(20, "head_dim", v.head_dim as u32)?;
    let la_conv = linear_conv_bytes(&v);
    let la_ssm = linear_ssm_bytes(&v);
    check(24, "linear_conv_bytes", la_conv as u32)?;
    check(28, "linear_ssm_bytes", la_ssm as u32)?;

    let fa_stride = full_attn_stride_bytes(&v);

    // Preflight: walk the body without mutating state. Any size
    // mismatch is caught here so a partial load doesn't leave the
    // ctx in a half-restored state. Mirrors the C-side preflight at
    // infer.m:8629..8649.
    {
        let mut q = header_bytes;
        for i in 0..v.num_layers {
            match v.layer_kind(i) {
                LayerKind::FullAttn => {
                    if buf.len() - q < 4 {
                        return Err(StateSnapshotError::Truncated {
                            layer: i,
                            need: 4,
                            got: buf.len() - q,
                        });
                    }
                    let len = i32::from_le_bytes(
                        buf[q..q + 4].try_into().unwrap(),
                    );
                    q += 4;
                    if len < 0 {
                        return Err(StateSnapshotError::NegativeLen {
                            layer: i,
                            len,
                        });
                    }
                    if (len as usize) > super::variants::MAX_SEQ_LEN {
                        return Err(StateSnapshotError::LenOverflow {
                            layer: i,
                            len,
                            max: super::variants::MAX_SEQ_LEN,
                        });
                    }
                    let bytes = 2 * (len as usize) * fa_stride;
                    if buf.len() - q < bytes {
                        return Err(StateSnapshotError::Truncated {
                            layer: i,
                            need: bytes,
                            got: buf.len() - q,
                        });
                    }
                    q += bytes;
                }
                LayerKind::LinearAttn => {
                    let bytes = la_conv + la_ssm;
                    if buf.len() - q < bytes {
                        return Err(StateSnapshotError::Truncated {
                            layer: i,
                            need: bytes,
                            got: buf.len() - q,
                        });
                    }
                    q += bytes;
                }
            }
        }
    }

    // Preflight ok — restore.
    let mut off = header_bytes;
    for i in 0..v.num_layers {
        match v.layer_kind(i) {
            LayerKind::FullAttn => {
                let len = i32::from_le_bytes(
                    buf[off..off + 4].try_into().unwrap(),
                );
                off += 4;
                let kv = match &mut layer_states[i] {
                    LayerState::FullAttn(kv) => kv,
                    LayerState::LinearAttn(_) => {
                        return Err(StateSnapshotError::ShapeMismatch {
                            field: "layer_state_kind",
                            got: 0,
                            want: 1,
                        });
                    }
                };
                if len > 0 {
                    let bytes = (len as usize) * fa_stride;
                    let k_dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            kv.k_cache.as_mut_ptr() as *mut u8,
                            bytes,
                        )
                    };
                    k_dst.copy_from_slice(&buf[off..off + bytes]);
                    off += bytes;
                    let v_dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            kv.v_cache.as_mut_ptr() as *mut u8,
                            bytes,
                        )
                    };
                    v_dst.copy_from_slice(&buf[off..off + bytes]);
                    off += bytes;
                }
                // Zero the [len, MAX_SEQ_LEN) tail so stale K/V from a
                // prior eval doesn't bleed into attention reads. Mirrors
                // C infer.m:8669..8678.
                let stride = v.num_kv_heads * v.head_dim;
                let from = (len as usize) * stride;
                kv.k_cache[from..].fill(0.0);
                kv.v_cache[from..].fill(0.0);
                kv.len = len;

                // Slice 5d-7b — mirror restored host KV into the GPU
                // KV mirrors so the GPU SDPA fast path sees the same
                // prefix. Mirrors C `infer.m:6570..6577`. Capped at
                // `GPU_KV_SEQ` (the persistent buffer's slot count);
                // restored sequences past that fall back to CPU SDPA
                // exactly as the C side does.
                if let Some(fa_idx) = full_attn_layer_idx_for(i) {
                    let mirror_len = (len as usize).min(GPU_KV_SEQ);
                    if mirror_len > 0 {
                        let n = mirror_len * stride;
                        // SAFETY: shared-storage GPU buffer; called at a
                        // token boundary (per moeflux.h:481 contract +
                        // the `discard_deferred_experts` guard at the
                        // top of state_load callers), so no GPU work is
                        // in flight on the mirror.
                        unsafe {
                            let k_dst = linear_buffers.gpu_kv_k[fa_idx]
                                .contents()
                                as *mut f32;
                            let v_dst = linear_buffers.gpu_kv_v[fa_idx]
                                .contents()
                                as *mut f32;
                            std::ptr::copy_nonoverlapping(
                                kv.k_cache.as_ptr(),
                                k_dst,
                                n,
                            );
                            std::ptr::copy_nonoverlapping(
                                kv.v_cache.as_ptr(),
                                v_dst,
                                n,
                            );
                        }
                    }
                }
            }
            LayerKind::LinearAttn => {
                let linear_idx = linear_layer_idx_for(i)
                    .expect("layer_kind says LinearAttn");
                write_buffer_bytes(
                    &linear_buffers.conv_state[linear_idx],
                    &buf[off..off + la_conv],
                );
                off += la_conv;
                write_buffer_bytes(
                    &linear_buffers.delta_state[linear_idx],
                    &buf[off..off + la_ssm],
                );
                off += la_ssm;
            }
        }
    }
    Ok(())
}

/// Public-facing convenience for `RsCtx`-aware callers that don't
/// have direct access to the `linear_buffers` field. Drains EVERY
/// in-flight deferred dispatch (the moeflux.h:481 contract) so the
/// snapshot reflects post-token state, not mid-flight state. Slice
/// 5d-9 widened the deferred slot to a depth-2 ring; this drains
/// the whole ring.
pub(super) fn drain_deferred(deferred: &mut deferred::DeferredRing) {
    deferred::discard_deferred_experts_in(deferred);
}

fn read_buffer_bytes(buf: &Buffer, dst: &mut [u8]) {
    let n = dst.len();
    debug_assert!(buf.length() as usize >= n);
    // SAFETY: shared-storage Metal buffer; caller ensures no GPU
    // work is in flight (state_save's contract).
    unsafe {
        std::ptr::copy_nonoverlapping(buf.contents() as *const u8, dst.as_mut_ptr(), n);
    }
}

fn write_buffer_bytes(buf: &Buffer, src: &[u8]) {
    let n = src.len();
    debug_assert!(buf.length() as usize >= n);
    // SAFETY: shared-storage Metal buffer; caller ensures no GPU
    // work is in flight (state_load's contract).
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), buf.contents() as *mut u8, n);
    }
}
