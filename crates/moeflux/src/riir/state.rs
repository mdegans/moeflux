//! Per-layer KV / recurrent state for the forward pass (Phase 4a).
//!
//! Mirrors the C-side `KVCache` + `LinearAttnState` allocations and the
//! `mf_state_*` truncation primitives from `metal_infer/infer.m`. The
//! Rust port owns these as fields on [`crate::riir::RsCtx`] instead of
//! addressing them through `KVCache **kv_caches` / `void **layer_states`
//! arrays — Rust enums replace the parallel-array + `is_full` modulo
//! pattern.
//!
//! ## Faithful-port note
//!
//! [`truncate`] preserves the C-side "Option A" semantic: any partial
//! truncation of a linear-attn layer resets that layer's recurrence
//! state to empty. This is lossy (the `(conv_state, ssm_state)` pair
//! folds the entire history; you can't unwind it). The bisect findings
//! flagged this as a bug source — drama_llama's prefix-cache reuse
//! silently diverges when truncating a position inside a linear-attn
//! span. The faithful port keeps the lossy semantic and the typed
//! `Result<(), CannotTruncateLinear>` lands as a Phase 7 post-cutover
//! slice, per `riir_moeflux_strategy.md`.
//!
//! FIXME(riir): lossy partial-linear truncation; faithful port of
//! `infer.m:2291 mf_state_truncate`. Phase 7 introduces the typed
//! error; in the meantime the call still resets to empty.

use crate::riir::variants::{Variant, MAX_SEQ_LEN, VARIANT};

/// Full-attention key/value cache for one layer. Allocated to
/// [`MAX_SEQ_LEN`] capacity up front; `len` tracks the occupied prefix.
///
/// Layout matches the C `KVCache`: `k_cache` and `v_cache` are
/// `[max_seq, num_kv_heads * head_dim]` row-major float arrays. Rust
/// uses `Box<[f32]>` instead of `float *` — same bytes, lifetime-bound.
#[derive(Debug)]
pub struct KvCache {
    pub k_cache: Box<[f32]>,
    pub v_cache: Box<[f32]>,
    pub len: i32,
}

impl KvCache {
    /// Allocate a zeroed cache sized for the active variant. Routes
    /// through `vec![0.0; n]` so the system allocator uses
    /// `alloc_zeroed`/`calloc` and gets lazy-committed pages on macOS
    /// for the multi-GB virtual reservation.
    pub fn new() -> Self {
        let entries = MAX_SEQ_LEN * VARIANT.num_kv_heads * VARIANT.head_dim;
        Self {
            k_cache: vec![0.0f32; entries].into_boxed_slice(),
            v_cache: vec![0.0f32; entries].into_boxed_slice(),
            len: 0,
        }
    }

    /// Reset to positions `[0, new_len)`. No-op if already shorter.
    /// Zeros the `[new_len, old_len)` window so stale K/V doesn't bleed
    /// into later decodes — mirrors `kv_cache_truncate` at infer.m:2243.
    pub fn truncate(&mut self, new_len: i32) {
        if new_len < 0 || new_len > self.len {
            return;
        }
        let old_len = self.len;
        let stride = VARIANT.num_kv_heads * VARIANT.head_dim;
        if new_len < old_len {
            let start = (new_len as usize) * stride;
            let end = (old_len as usize) * stride;
            self.k_cache[start..end].fill(0.0);
            self.v_cache[start..end].fill(0.0);
        }
        self.len = new_len;
    }
}

/// Multi-head Latent Attention KV cache for one MLA layer (DeepSeek-V3
/// architecture — Cogito-V2-Preview-671B).
///
/// MLA jointly compresses K and V to a `kv_lora_rank`-dim latent
/// (512 for Cogito-V2) plus a shared `qk_rope_head_dim`-dim rope-K
/// (64 for Cogito-V2) per token. Total cached width per token is
/// `kv_lora_rank + qk_rope_head_dim` (= 576 for Cogito-V2), giving
/// ~28× memory compression vs GQA at long context.
///
/// At use time, K_nope and V are reconstructed from the latent via
/// `kv_b_proj` per cached position; only the rope-K is stored
/// directly.
///
/// Allocated to [`MAX_SEQ_LEN`] capacity. Pages are lazy-committed
/// via `alloc_zeroed`, so the initial reservation is virtual-only on
/// macOS.
///
/// Phase C scaffold: type defined; allocation and dispatch into
/// [`LayerState`] land alongside the MLA forward kernel in
/// [`super::mla_attn_cpu`].
#[derive(Debug)]
pub struct MlaKvCache {
    /// `kv_a_layernorm`-output cache: post-down-projection,
    /// post-norm latent. Shape `[max_seq, kv_lora_rank]` row-major.
    pub latent_cache: Box<[f32]>,
    /// Pre-RoPE'd rope-K cache (already RoPE-applied at the position
    /// it was stored at — broadcast across all heads at use time).
    /// Shape `[max_seq, qk_rope_head_dim]`.
    pub rope_k_cache: Box<[f32]>,
    /// Number of populated positions.
    pub len: i32,
}

impl MlaKvCache {
    /// Allocate a zeroed MLA KV cache sized for the active variant.
    /// Reads `kv_lora_rank` and `qk_rope_head_dim` from `VARIANT`.
    /// On non-MLA variants these are 0 and the resulting cache is
    /// empty — guarded against accidentally calling this at
    /// allocation dispatch in [`alloc_layer_states`].
    pub fn new() -> Self {
        let latent_entries = MAX_SEQ_LEN * VARIANT.kv_lora_rank;
        let rope_k_entries = MAX_SEQ_LEN * VARIANT.qk_rope_head_dim;
        Self {
            latent_cache: vec![0.0f32; latent_entries].into_boxed_slice(),
            rope_k_cache: vec![0.0f32; rope_k_entries].into_boxed_slice(),
            len: 0,
        }
    }

    /// Reset to positions `[0, new_len)`. No-op if already shorter or
    /// `new_len` is invalid.
    pub fn truncate(&mut self, new_len: i32) {
        if new_len < 0 || new_len > self.len {
            return;
        }
        let old_len = self.len;
        if new_len < old_len {
            let l_start =
                (new_len as usize) * VARIANT.kv_lora_rank;
            let l_end = (old_len as usize) * VARIANT.kv_lora_rank;
            self.latent_cache[l_start..l_end].fill(0.0);
            let r_start =
                (new_len as usize) * VARIANT.qk_rope_head_dim;
            let r_end =
                (old_len as usize) * VARIANT.qk_rope_head_dim;
            self.rope_k_cache[r_start..r_end].fill(0.0);
        }
        self.len = new_len;
    }
}

impl Default for MlaKvCache {
    fn default() -> Self {
        Self::new()
    }
}

/// GatedDeltaNet recurrent state for one linear-attention layer.
/// `conv_state` holds the depthwise conv1d's last `(kernel_size - 1)`
/// inputs; `ssm_state` holds the per-v-head outer-product state of
/// shape `[num_v_heads, value_dim, key_dim]` (flattened).
#[derive(Debug)]
pub struct LinearAttnState {
    pub conv_state: Box<[f32]>,
    pub ssm_state: Box<[f32]>,
}

impl LinearAttnState {
    /// Allocate a zeroed state sized for the active variant.
    pub fn new() -> Self {
        let conv_entries =
            (Variant::CONV_KERNEL_SIZE - 1) * VARIANT.linear_conv_dim();
        let ssm_entries = VARIANT.linear_num_v_heads
            * Variant::LINEAR_VALUE_DIM
            * Variant::LINEAR_KEY_DIM;
        Self {
            conv_state: vec![0.0f32; conv_entries].into_boxed_slice(),
            ssm_state: vec![0.0f32; ssm_entries].into_boxed_slice(),
        }
    }

    /// Reset to the empty-sequence state. Lossy by construction — see
    /// the module docs. Mirrors `linear_attn_state_reset` at
    /// infer.m:2260.
    pub fn reset(&mut self) {
        self.conv_state.fill(0.0);
        self.ssm_state.fill(0.0);
    }
}

/// Per-layer state. The variant tag matches the C-side
/// `(layer + 1) % FULL_ATTN_INTERVAL == 0` test: every Nth layer is a
/// full-attention layer with a KV cache, the rest are linear-attention
/// layers with a GatedDeltaNet recurrence.
///
/// The full-attention slot has two flavors selected by
/// [`Variant::attn_kind`]: GQA (`FullAttn`) caches per-head K and V
/// directly; MLA (`Mla`) caches a compressed latent + a shared rope-K
/// per token and reconstructs K/V at use time.
#[derive(Debug)]
pub enum LayerState {
    FullAttn(KvCache),
    Mla(MlaKvCache),
    LinearAttn(LinearAttnState),
}

impl LayerState {
    /// True for any flavor that grows a sequence-length-shaped cache
    /// (GQA `FullAttn` or `Mla`). Used by callers that distinguish
    /// "real" KV layers from the constant-state linear-attn layers.
    pub fn is_full(&self) -> bool {
        matches!(self, Self::FullAttn(_) | Self::Mla(_))
    }
}

/// Allocate the per-layer state vector for the active variant.
/// Dispatched via [`Variant::layer_kind`] — for qwen3_5_moe that's
/// `(i + 1) % full_attn_interval == 0`; for DeepSeek-V3 / Cogito-V2
/// every layer is full-attn (`full_attn_interval = 1`). The flavor
/// of full-attn (GQA vs MLA) is selected by [`Variant::attn_kind`]:
/// GQA gets a [`KvCache`], MLA gets a compressed [`MlaKvCache`].
/// This matters at allocation because [`KvCache::new`] would reserve
/// `num_kv_heads * head_dim * MAX_SEQ_LEN * 2 * 4` bytes — for
/// Cogito-V2 that's a ~196 GB virtual reservation per layer, all of
/// it lazy-committed but still wasted address space we never touch.
/// Mirrors the C-side allocation in `mf_init_model` (infer.m:7511+).
pub fn alloc_layer_states() -> Vec<LayerState> {
    use super::variants::{AttnKind, LayerKind};
    (0..VARIANT.num_layers)
        .map(|i| match VARIANT.layer_kind(i) {
            LayerKind::FullAttn => match VARIANT.attn_kind {
                AttnKind::Gqa => LayerState::FullAttn(KvCache::new()),
                AttnKind::Mla => LayerState::Mla(MlaKvCache::new()),
            },
            LayerKind::LinearAttn => {
                LayerState::LinearAttn(LinearAttnState::new())
            }
        })
        .collect()
}

/// Reset every layer's state to empty. Mirrors `mf_state_clear_all`
/// (infer.m:2271).
pub fn clear_all(layers: &mut [LayerState]) {
    for layer in layers {
        match layer {
            LayerState::FullAttn(kv) => kv.truncate(0),
            LayerState::Mla(mla) => mla.truncate(0),
            LayerState::LinearAttn(la) => la.reset(),
        }
    }
}

/// Truncate every layer to positions `[0, p0)`. Linear-attn layers
/// reset to empty (lossy — see module docs). `p0 < 0` is treated as 0
/// (full clear); `p1 < 0` means "to end". Mirrors `mf_state_truncate`
/// (infer.m:2291).
pub fn truncate(layers: &mut [LayerState], p0: i32, p1: i32) {
    let new_len = p0.max(0);
    for layer in layers {
        match layer {
            LayerState::FullAttn(kv) => {
                let effective_end =
                    if p1 < 0 || p1 > kv.len { kv.len } else { p1 };
                let truncate_to = new_len.min(effective_end);
                kv.truncate(truncate_to);
            }
            LayerState::Mla(mla) => {
                let effective_end =
                    if p1 < 0 || p1 > mla.len { mla.len } else { p1 };
                let truncate_to = new_len.min(effective_end);
                mla.truncate(truncate_to);
            }
            // FIXME(riir): faithful port of the lossy semantic. A
            // partial truncation of a linear-attn span resets the
            // recurrence to empty. Phase 7 introduces a typed
            // `Result<(), CannotTruncateLinear>` so callers must
            // explicitly handle the unwind failure.
            LayerState::LinearAttn(la) => la.reset(),
        }
    }
}

/// Largest occupied position across full-attn layers (GQA `FullAttn`
/// or `Mla`), or `-1` if no full-attn layer exists at all. Mirrors
/// `mf_state_pos_max` (infer.m:2320).
pub fn pos_max(layers: &[LayerState]) -> i32 {
    let mut max_len = -1;
    for layer in layers {
        let len = match layer {
            LayerState::FullAttn(kv) => kv.len,
            LayerState::Mla(mla) => mla.len,
            LayerState::LinearAttn(_) => continue,
        };
        if len > max_len {
            max_len = len;
        }
    }
    max_len
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty state on a fresh allocation: pos_max returns 0 (the
    /// largest occupied position across full-attn layers, all of which
    /// start at len=0). The `-1` sentinel only fires when no full-attn
    /// layer exists at all — not the case for any supported variant.
    #[test]
    fn empty_state_pos_max_is_zero() {
        let mut layers = alloc_layer_states();
        assert_eq!(pos_max(&layers), 0);
        clear_all(&mut layers);
        assert_eq!(pos_max(&layers), 0);
    }

    /// Truncate on empty state is a no-op; pos_max stays at 0
    /// regardless of arguments.
    #[test]
    fn truncate_empty_is_noop() {
        let mut layers = alloc_layer_states();
        truncate(&mut layers, 0, -1);
        assert_eq!(pos_max(&layers), 0);
        truncate(&mut layers, 5, 10);
        assert_eq!(pos_max(&layers), 0);
        truncate(&mut layers, -1, -1);
        assert_eq!(pos_max(&layers), 0);
    }

    /// Synthetic: hand-set one full-attn layer's len, observe pos_max
    /// pick it up, truncate, observe pos_max drop. Handles both
    /// flavors (`FullAttn` for GQA variants, `Mla` for DeepSeek-V3
    /// variants); the test asserts the same observable behavior on
    /// either side of the dispatch.
    #[test]
    fn truncate_drops_full_attn_len() {
        let mut layers = alloc_layer_states();
        let injected = layers
            .iter_mut()
            .find_map(|l| match l {
                LayerState::FullAttn(kv) => {
                    kv.len = 7;
                    Some(())
                }
                LayerState::Mla(mla) => {
                    mla.len = 7;
                    Some(())
                }
                LayerState::LinearAttn(_) => None,
            });
        assert!(
            injected.is_some(),
            "variant must have at least one full-attn (GQA or MLA) layer",
        );
        assert_eq!(pos_max(&layers), 7);

        truncate(&mut layers, 3, -1);
        assert_eq!(pos_max(&layers), 3);

        truncate(&mut layers, 10, -1);
        // new_len > len: kv_cache_truncate is a no-op when new_len >
        // self.len. p0=10 with current len=3 → effective_end=3,
        // truncate_to=min(10, 3)=3. Stays at 3.
        assert_eq!(pos_max(&layers), 3);

        clear_all(&mut layers);
        assert_eq!(pos_max(&layers), 0);
    }
}
