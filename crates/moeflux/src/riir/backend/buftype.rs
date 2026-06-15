//! Typed buffer identifiers for the backend graph IR.
//!
//! Every [`BufId<B>`] in the graph carries a compile-time role tag.
//! Producer code can only construct a `BufId<T>` of one role; the
//! pool's `alloc::<T>` returns a `BufId<T>` for the requested role,
//! and `Op` enum fields take specific concrete tags (e.g.
//! [`Op::MoeGatherIdFuse::mlp_in: BufId<MoeInputBuf>`]). Wrong-wiring
//! becomes a compile error.
//!
//! ## Why
//!
//! Session 19 landed an MoE gather-id producer that passed the
//! pre-RmsNorm residual (`MoeGraphScratch::h_mid`) into a field on
//! the consumer named `h_mid` — which the consuming kernel actually
//! reads as the *post-norm* activation. End-to-end output was
//! `</think>` + EoT. Kernel diff oracle missed it (synthetic input).
//! Code review missed it. The Session A rename (`h_mid` → `mlp_in`)
//! reduced the trap but didn't eliminate the class. This module
//! eliminates the class at compile time.
//!
//! See `.claude/memory/prefill_gather_id_session_19_landed.md` for
//! the full bug trace and `.claude/memory/moeflux_hardening_arc_plan.md`
//! for the three-session arc this refactor closes.
//!
//! ## Shape
//!
//! - [`Buf`] is a sealed marker trait. Implementors are zero-sized
//!   unit structs ([`MoeInputBuf`], [`AttnInputBuf`], ...). The seal
//!   prevents downstream crates / future modules from adding tags
//!   ad-hoc.
//! - [`BufId<B>`] wraps a `u32` index plus a `PhantomData<B>`. The
//!   inner field is `pub(crate)` so only the `backend` module can
//!   mint or unwrap a `BufId`; producers can't bypass the tag.
//! - Concrete role tags name the buffer's role (e.g. [`MoeInputBuf`],
//!   not `PostNormBuf`) — what the buffer is *for*, not where it
//!   came from.
//! - Union tags (e.g. [`MatvecIn`], [`RmsNormIn`]) cover `Op` fields
//!   that legitimately accept multiple roles. Each constituent has
//!   an explicit `From<BufId<X>> for BufId<Union>` impl below.
//!   Producer push sites do `.into()` at the call site — that
//!   explicit `.into()` is the documentation of "this role is legal
//!   here."
//! - The bucket-permute scratch tags ([`BucketGateBuf`],
//!   [`BucketUpBuf`], [`BucketOutBuf`]) have unidirectional `From`
//!   impls into the gather-id-path tags ([`GateMidBuf`],
//!   [`UpMidBuf`], [`DownMidBuf`]). The bucket-permute path is the
//!   canonical allocation; the env-on gather-id path converts at
//!   the push site.
//! - **No `Raw: Buf` catch-all.** A `Raw` tag would invite `.into()`
//!   abuse that defeats the entire safety argument. If a buffer has
//!   no role, we invent one.
//!
//! ## Examples of what now fails to compile
//!
//! Passing the pre-norm residual buffer where the post-norm MoE
//! input is expected (the session-19 bug):
//!
//! ```compile_fail
//! # use moeflux::riir::backend::buftype::{BufId, MoeInputBuf, ResidualBuf};
//! let h_mid: BufId<ResidualBuf> = unsafe { std::mem::zeroed() };
//! // session-19: this is the wrong field to pass h_mid into.
//! let mlp_in: BufId<MoeInputBuf> = h_mid;
//! ```
//!
//! Passing a non-attention-input buffer where the attention input
//! is expected:
//!
//! ```compile_fail
//! # use moeflux::riir::backend::buftype::{BufId, AttnInputBuf, MoeInputBuf};
//! let post_attn: BufId<MoeInputBuf> = unsafe { std::mem::zeroed() };
//! let qkv_in: BufId<AttnInputBuf> = post_attn;
//! ```
//!
//! Going through the union conversion works:
//!
//! ```
//! # use moeflux::riir::backend::buftype::{BufId, AttnInputBuf, MatvecIn};
//! # let normed: BufId<AttnInputBuf> = unsafe { std::mem::zeroed() };
//! let matvec_in: BufId<MatvecIn> = normed.into();
//! ```

use std::fmt;
use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}

/// Marker trait for buffer role tags. Sealed — only the tags
/// defined in this module implement it.
pub trait Buf: sealed::Sealed + Copy + Clone + 'static {}

/// Identifier into a [`super::BufferPool`], parameterized by the
/// buffer's role tag. Backend-agnostic; each backend translates
/// `BufId<B>` to its native handle internally (e.g. `metal::Buffer`,
/// `RefCell<Vec<u8>>`).
///
/// Producers cannot mint a `BufId<B>` directly — the inner field is
/// `pub(crate)`. The only ways to obtain one are:
///
/// 1. `Pool::alloc::<B>(...) -> BufId<B>` (the pool's typed alloc).
/// 2. `Pool::register_borrowed::<B>(...) -> BufId<B>` for KV-cache
///    and weight-file slots.
/// 3. A `From<BufId<X>> for BufId<B>` impl defined in this module.
pub struct BufId<B: Buf> {
    pub(crate) idx: u32,
    pub(crate) _tag: PhantomData<B>,
}

impl<B: Buf> BufId<B> {
    /// Construct from a raw `u32` index. Crate-internal only —
    /// producers obtain `BufId<B>`s through the pool's typed `alloc`
    /// or through `From` impls.
    #[inline]
    pub(crate) fn from_raw(idx: u32) -> Self {
        Self {
            idx,
            _tag: PhantomData,
        }
    }

    /// The raw `u32` index. Crate-internal only — pool impls and
    /// the lifetime-coloring pass use this to operate on indices
    /// without caring about tags.
    #[inline]
    pub(crate) fn raw(self) -> u32 {
        self.idx
    }
}

// Manual derives — `#[derive]` works for `PhantomData<B>` in stable
// Rust for these traits but only when the `B` parameter itself
// satisfies them. Since `B: Buf` only requires `Copy + Clone +
// 'static`, we'd need `B: Eq + Hash + Ord + ...` bounds in the
// derive expansion — which would force consumers to write those
// bounds on every tag. Implementing manually keeps `B` to just
// `Buf` and works because `PhantomData<B>` is unconditionally
// `Copy + Clone + Eq + PartialEq + Ord + PartialOrd + Hash`.

impl<B: Buf> Copy for BufId<B> {}

impl<B: Buf> Clone for BufId<B> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: Buf> PartialEq for BufId<B> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
    }
}

impl<B: Buf> Eq for BufId<B> {}

impl<B: Buf> std::hash::Hash for BufId<B> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.idx.hash(state);
    }
}

impl<B: Buf> PartialOrd for BufId<B> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<B: Buf> Ord for BufId<B> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.idx.cmp(&other.idx)
    }
}

impl<B: Buf> fmt::Debug for BufId<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BufId<_>({})", self.idx)
    }
}

impl<B: Buf> fmt::Display for BufId<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.idx)
    }
}

// ---------------------------------------------------------------------------
// Concrete role tags.
//
// Each is a ZST unit struct. The `tag!` macro keeps the boilerplate
// short: declare the tag + Sealed + Buf in one line.
// ---------------------------------------------------------------------------

macro_rules! tag {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Copy, Clone, Debug)]
        pub struct $name;
        impl sealed::Sealed for $name {}
        impl Buf for $name {}
    };
}

tag!(
    /// Output of the embedding-gather kernel. Feeds the first layer's
    /// `RmsNormBf16NTokens` input.
    EmbedOutBuf
);
tag!(
    /// The pre-norm residual stream. `MoeGraphScratch::h_mid` and
    /// the output of each layer's `ResidualAddNTokens`. Reads into
    /// the post-attn `RmsNormBf16NTokens` to produce a `MoeInputBuf`.
    ResidualBuf
);
tag!(
    /// The cross-layer hidden double-buffer pair (`hidden_a` /
    /// `hidden_b` on [`crate::riir::attn::linear_attn_forward::HiddenDoubleBuffer`]).
    /// Both physical buffers share this tag — the orchestrator swaps
    /// the pair between layers; type-system parity tracking is not
    /// the goal here.
    HiddenBuf
);
tag!(
    /// Output of the pre-attention RMS-norm. Feeds the Q/K/V (or
    /// fused QKV) projection matvec inputs.
    AttnInputBuf
);
tag!(
    /// Output of the post-attention RMS-norm — the **post-norm MoE
    /// input**. Feeds the router matvec, the shared-FFN matvec, and
    /// (the session-19 bug-of-record) `Op::MoeGatherIdFuse::mlp_in`.
    /// `MoeGraphScratch::h_post`.
    MoeInputBuf
);
tag!(
    /// Output of the final tail RMS-norm. Feeds the lm_head matvec.
    TailNormedBuf
);

// Attention projections.
tag!(
    /// Output of the q-projection matvec on the full-attn path.
    /// `[n_tokens, num_heads, 2*head_dim]` (q + gate interleaved per
    /// head — split by [`crate::riir::backend::Op::SplitQGate`]).
    QProjOutBuf
);
tag!(
    /// The query stack proper — `[n_tokens, num_heads, head_dim]`.
    /// Output of `SplitQGate`'s `q_out`. RoPE and per-head RMS-norm
    /// run in-place; same tag pre- and post-rope.
    QBuf
);
tag!(
    /// Output of the k-projection matvec.
    /// `[n_tokens, num_k_heads * head_dim]`. RoPE and per-head
    /// RMS-norm in-place; same tag pre- and post-rope. Reused as
    /// the `k_src` input to `KvCacheAppendNTokens`.
    KProjOutBuf
);
tag!(
    /// Output of the v-projection matvec.
    /// `[n_tokens, num_k_heads * head_dim]`. Reused as the `v_src`
    /// input to `KvCacheAppendNTokens`.
    VProjOutBuf
);
tag!(
    /// The per-head query gate, split out of the fused q-projection
    /// by `SplitQGate`. Feeds `SigmoidGateNTokens` to multiply
    /// against `AttnOutBuf` in-place.
    QGateBuf
);
tag!(
    /// Output of the SDPA kernel — pre-`o_proj`. The sigmoid-gate
    /// runs in-place against this buffer (full-attn path).
    AttnOutBuf
);
tag!(
    /// Output of the `o_proj` matvec — the residual delta for this
    /// layer's attention block. Feeds `ResidualAddNTokens.a`.
    OProjOutBuf
);

// KV cache (persistent, per-layer).
tag!(
    /// Persistent per-layer K-cache slot. Registered via
    /// `Pool::register_borrowed` from the underlying Metal buffer
    /// owned by [`crate::riir::attn::kv_cache::KvCacheLayerState`].
    /// RMW under `KvCacheAppendNTokens`; read by `SdpaCausalTiled`.
    KvCacheKBuf
);
tag!(
    /// Persistent per-layer V-cache slot. See [`KvCacheKBuf`].
    KvCacheVBuf
);

// Router / MoE shared block.
tag!(
    /// Output of the router gate matvec — `[n_tokens, n_experts]`.
    /// Feeds `MoeSoftmaxTopK`.
    RouterLogitsBuf
);
tag!(
    /// Per-token top-k expert indices — `[n_tokens, k]` i32.
    /// Output of `MoeSoftmaxTopK`; input to `MoeGatherIdFuse` and to
    /// the host-side bucket builder.
    RouterIdxBuf
);
tag!(
    /// Per-token top-k expert weights — `[n_tokens, k]` f32.
    /// Output of `MoeSoftmaxTopK`; normalized in-place by
    /// `MoeNormalizeWeights`; consumed by `MoeGatherIdFuse` and the
    /// bucket-permute path.
    RouterWeightsBuf
);
tag!(
    /// Shared-expert gating scalar — `[n_tokens, 1]` matvec output.
    /// Feeds `MoeCombineResidualNTokens.shared_gate`.
    SharedGateBuf
);
tag!(
    /// Shared-FFN gate-proj output. `[n_tokens, moe_inter]`.
    SharedFfnGateBuf
);
tag!(
    /// Shared-FFN up-proj output. `[n_tokens, moe_inter]`.
    SharedFfnUpBuf
);
tag!(
    /// Shared-FFN SwiGLU activation — output of fused
    /// `SwigluFusedBatched` over gate * up. Feeds the down-proj
    /// matvec.
    SharedFfnActBuf
);
tag!(
    /// Shared-FFN down-proj output — the shared expert's contribution
    /// to the residual. Feeds `MoeCombineResidualNTokens.shared_out`.
    SharedFfnDownBuf
);
tag!(
    /// MoE per-token accumulator — `[n_tokens, hidden_dim]`. Zeroed
    /// before each MoE block, scatter-added into by the bucket
    /// kernel, or written by the gather-id kernel. Feeds
    /// `MoeCombineResidualNTokens.moe_sum`.
    MoeOutSumBuf
);

// Bucket-permute MoE path (env=off, the original).
tag!(
    /// Host-uploaded bucket-permuted MoE input — `[total_assignments,
    /// hidden_dim]`. Built CPU-side from the post-norm activation by
    /// permuting per-expert.
    BucketInputBuf
);
tag!(
    /// Bucket-permute gate-proj output. **Allocation canonical for
    /// `Op::MoeGatherIdFuse::gate_mid` via `From<BucketGateBuf> for
    /// BufId<GateMidBuf>` (unidirectional).**
    BucketGateBuf
);
tag!(
    /// Bucket-permute up-proj output. Canonical allocation for
    /// `Op::MoeGatherIdFuse::up_mid` via unidirectional `From`.
    BucketUpBuf
);
tag!(
    /// Bucket-permute SwiGLU activation (gate * up).
    BucketActBuf
);
tag!(
    /// Bucket-permute down-proj output. Canonical allocation for
    /// `Op::MoeGatherIdFuse::down_mid` via unidirectional `From`.
    BucketOutBuf
);
tag!(
    /// Per-assignment-row token index into the original `[n_tokens]`
    /// space. Used by the bucket combine to scatter-add into
    /// `MoeOutSumBuf`.
    BucketTokenIdxBuf
);
tag!(
    /// Per-assignment-row routing weight. Read by the bucket combine.
    BucketWeightsBuf
);
tag!(
    /// Per-assignment-row expert slot (expansion of
    /// `Op::MoeBatchedPermuteFuse::expert_slots`). The gather kernel's
    /// row→slot table.
    ExpertIndicesBuf
);
tag!(
    /// Layer's expert blob — every expert's packed weight block
    /// stacked at uniform stride. Registered via `Pool::register_borrowed`
    /// from `ExpertFiles`.
    ExpertBaseBuf
);

// Gather-id MoE path (env=on).
tag!(
    /// Per-expert assignment count — `[n_experts]` u32. Scratch for
    /// `Op::MoeGatherIdFuse`'s map0 pre-pass.
    HtpeBuf
);
tag!(
    /// Per-expert assignment list, encoded as `token*k + slot` —
    /// `[n_experts, n_tokens]` i32. Scratch for `Op::MoeGatherIdFuse`.
    HidsBuf
);
tag!(
    /// Gather-id gate-proj output / SwiGLU output (reused in place).
    /// `[n_tokens, k, moe_inter]` f32. **Receives** a `BufId` from
    /// `BucketGateBuf` via unidirectional `From`.
    GateMidBuf
);
tag!(
    /// Gather-id up-proj output. `[n_tokens, k, moe_inter]` f32.
    /// Receives from `BucketUpBuf`.
    UpMidBuf
);
tag!(
    /// Gather-id down-proj output. `[n_tokens, k, hidden_dim]` f32.
    /// Receives from `BucketOutBuf`. Input to `combine_topk` →
    /// `MoeOutSumBuf`.
    DownMidBuf
);

// Linear-attn (Qwen3 gated DeltaNet).
tag!(
    /// Conv1d output / per-token `[q | k | v]` stack — output of
    /// `Conv1dStepNTokens`. Operated on in-place by
    /// `RmsNormQkNTokens`. Consumed by the SSM step.
    ConvOutBuf
);
tag!(
    /// Persistent per-layer Conv1d state (carries the kernel-window
    /// forward across tokens). RMW under `Conv1dStepNTokens`.
    ConvStateBuf
);
tag!(
    /// QKV-stack — output of the fused qkv-proj matvec.
    /// `[n_tokens, conv_dim]`. Feeds `Conv1dStepNTokens`.
    QkvStackBuf
);
tag!(
    /// Z-stack — output of the z-proj matvec. Gate for the gated
    /// RMS-norm at the end of the linear-attn layer.
    ZStackBuf
);
tag!(
    /// Alpha-stack — output of the alpha-proj matvec. Feeds
    /// `ComputeDecayBetaNTokens.alpha_in`.
    AlphaStackBuf
);
tag!(
    /// Beta-stack — output of the beta-proj matvec. Feeds
    /// `ComputeDecayBetaNTokens.beta_in`.
    BetaStackBuf
);
tag!(
    /// Per-token decay factor. Output of
    /// `ComputeDecayBetaNTokens.g_decay_out`.
    GDecayBuf
);
tag!(
    /// Per-token beta gate. Output of
    /// `ComputeDecayBetaNTokens.beta_gate_out`.
    BetaGateBuf
);
tag!(
    /// Persistent per-layer SSM hidden state — `[num_v_heads,
    /// value_dim, key_dim]`-shaped. RMW under
    /// `GatedDeltaNetStepNTokens` / `GatedDeltaNetChunkwise`.
    DeltaStateBuf
);
tag!(
    /// Output of the gated DeltaNet recurrence — `[n_tokens,
    /// num_v_heads * value_dim]`. Feeds `GatedRmsNormNTokens.values`.
    DeltaOutBuf
);
tag!(
    /// Output of the gated RMS-norm at the tail of the linear-attn
    /// layer. Feeds the o-proj matvec input.
    ValueOutBuf
);

// Misc.
tag!(
    /// RoPE inverse-frequency table. Persistent, host-uploaded.
    /// Read-only.
    RopeInvFreqBuf
);
tag!(
    /// Token IDs — `[n_tokens]` i32. Host-uploaded; input to
    /// `EmbedGatherNTokens`.
    TokenIdsBuf
);
tag!(
    /// Final lm_head matvec output — `[1, vocab_size]` (decode) or
    /// `[n_tokens, vocab_size]` (prefill last-token).
    LogitsBuf
);

// Legacy.
tag!(
    /// Per-token cogito MoE path buffers. **Deprecated** —
    /// `[crate::riir::moe::expert_forward::MoeBuffers]`'s entire
    /// field set carries this tag. Frozen pending M5 Studio
    /// availability (need 512 GB unified RAM for cogito 600B-class
    /// models). When the cogito path is rewritten post-M5, retype
    /// the fields properly. Do not iterate on this path without
    /// retagging.
    DeprecatedCogitoBuf
);

// ---------------------------------------------------------------------------
// Union tags.
//
// Each union tag is a ZST unit struct that's the target of
// `From<BufId<Source>> for BufId<Union>` impls below. Producer push
// sites doing `.into()` convert from a concrete role tag to the
// union — that `.into()` is the documentation of "this role is
// legal in this slot."
// ---------------------------------------------------------------------------

tag!(
    /// Union tag for `Op::MatvecNTokens.input` — the matvec is the
    /// graph's fan-in node; many roles legitimately feed it.
    ///
    /// Convertible from: [`AttnInputBuf`], [`MoeInputBuf`],
    /// [`AttnOutBuf`], [`ValueOutBuf`], [`SharedFfnActBuf`],
    /// [`TailNormedBuf`].
    MatvecIn
);
tag!(
    /// Union tag for `Op::MatvecNTokens.output` — the matvec is the
    /// graph's fan-out node; many roles are legitimate matvec outputs.
    ///
    /// Convertible from: every matvec-output role tag (Q/K/V/O proj,
    /// router logits, shared-FFN gate/up/down, shared-gate, lm_head
    /// logits, linear-attn qkv/z/alpha/beta stacks).
    MatvecOut
);
tag!(
    /// Union tag for the input of all RMS-norm `Op` variants
    /// (`RmsNormBf16NTokens.x`, `RmsNormQkNTokens.x`,
    /// `RmsNormPerHeadNTokens.x`).
    ///
    /// Convertible from: [`EmbedOutBuf`], [`ResidualBuf`],
    /// [`HiddenBuf`], [`ConvOutBuf`], [`QBuf`], [`KProjOutBuf`].
    RmsNormIn
);
tag!(
    /// Union tag for `Op::RmsNormBf16NTokens.out` — the same Op is
    /// pushed at multiple points (input-norm, post-attn-norm,
    /// tail-norm) writing to different role buffers.
    ///
    /// Convertible from: [`AttnInputBuf`], [`MoeInputBuf`],
    /// [`TailNormedBuf`].
    RmsNormOut
);

// ---------------------------------------------------------------------------
// From impls.
//
// Per-pair, explicit. No global `SubtypeOf` trait. Each conversion
// is auditable by reading this file.
// ---------------------------------------------------------------------------

macro_rules! impl_from {
    ($src:ident -> $dst:ident) => {
        impl From<BufId<$src>> for BufId<$dst> {
            #[inline]
            fn from(x: BufId<$src>) -> Self {
                BufId::from_raw(x.raw())
            }
        }
    };
}

// Unidirectional: bucket-permute scratch → gather-id scratch.
// The bucket path is the canonical allocation; env-on gather-id
// converts at the push site.
impl_from!(BucketGateBuf -> GateMidBuf);
impl_from!(BucketUpBuf -> UpMidBuf);
impl_from!(BucketOutBuf -> DownMidBuf);

// Head/tail plumbing — added by `riir/mod.rs:step_internal_batched_gqa`'s
// head graph (embed_gather writes the model's input, which lives in
// the cross-layer hidden double-buffer) and tail graph (final_norm
// reads the same double-buffer slot the last layer wrote).
//
// `From<HiddenBuf> for BufId<EmbedOutBuf>`: the head writes embedding
//   output into `hidden_a` of the run-lifetime `HiddenDoubleBuffer`
//   (`BufId<HiddenBuf>`). `Op::EmbedGatherNTokens::hidden_out` is
//   typed `BufId<EmbedOutBuf>`. Same physical role at the top of
//   the model.
// `From<HiddenBuf> for BufId<TailNormedBuf>`: the tail's final-norm
//   writes into the *other* double-buffer slot (`hidden_b`), which
//   is `BufId<HiddenBuf>`. `Op::RmsNormBf16NTokens.out` accepts
//   `RmsNormOut`, which is the union over the norm-output role
//   tags including `TailNormedBuf`. This impl gives the producer a
//   single `.into()` to reach the union — the alternative would be
//   double-`.into()` (HiddenBuf → TailNormedBuf → RmsNormOut), which
//   is fine but verbose; this shortcut keeps the push site to one
//   `.into()`.
impl_from!(HiddenBuf -> EmbedOutBuf);
impl_from!(HiddenBuf -> TailNormedBuf);

// MatvecIn union.
impl_from!(AttnInputBuf -> MatvecIn);
impl_from!(MoeInputBuf -> MatvecIn);
impl_from!(AttnOutBuf -> MatvecIn);
impl_from!(ValueOutBuf -> MatvecIn);
impl_from!(SharedFfnActBuf -> MatvecIn);
impl_from!(TailNormedBuf -> MatvecIn);

// MatvecOut union.
impl_from!(QProjOutBuf -> MatvecOut);
impl_from!(KProjOutBuf -> MatvecOut);
impl_from!(VProjOutBuf -> MatvecOut);
impl_from!(OProjOutBuf -> MatvecOut);
impl_from!(RouterLogitsBuf -> MatvecOut);
impl_from!(SharedGateBuf -> MatvecOut);
impl_from!(SharedFfnGateBuf -> MatvecOut);
impl_from!(SharedFfnUpBuf -> MatvecOut);
impl_from!(SharedFfnDownBuf -> MatvecOut);
impl_from!(LogitsBuf -> MatvecOut);
impl_from!(QkvStackBuf -> MatvecOut);
impl_from!(ZStackBuf -> MatvecOut);
impl_from!(AlphaStackBuf -> MatvecOut);
impl_from!(BetaStackBuf -> MatvecOut);

// RmsNormIn union.
impl_from!(EmbedOutBuf -> RmsNormIn);
impl_from!(ResidualBuf -> RmsNormIn);
impl_from!(HiddenBuf -> RmsNormIn);
impl_from!(ConvOutBuf -> RmsNormIn);
impl_from!(QBuf -> RmsNormIn);
impl_from!(KProjOutBuf -> RmsNormIn);

// RmsNormOut union.
impl_from!(AttnInputBuf -> RmsNormOut);
impl_from!(MoeInputBuf -> RmsNormOut);
impl_from!(TailNormedBuf -> RmsNormOut);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buf_id_construction_and_round_trip() {
        let id: BufId<MoeInputBuf> = BufId::from_raw(42);
        assert_eq!(id.raw(), 42);
    }

    #[test]
    fn buf_id_eq_hash_by_index_only() {
        let a: BufId<MoeInputBuf> = BufId::from_raw(7);
        let b: BufId<MoeInputBuf> = BufId::from_raw(7);
        let c: BufId<MoeInputBuf> = BufId::from_raw(8);
        assert_eq!(a, b);
        assert_ne!(a, c);

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn buf_id_display_matches_legacy() {
        let id: BufId<MoeInputBuf> = BufId::from_raw(13);
        assert_eq!(format!("{id}"), "%13");
    }

    #[test]
    fn unidirectional_bucket_to_mid_conversion() {
        let bucket: BufId<BucketGateBuf> = BufId::from_raw(1);
        let mid: BufId<GateMidBuf> = bucket.into();
        assert_eq!(mid.raw(), 1);
    }

    #[test]
    fn union_conversion_from_concrete() {
        let normed: BufId<AttnInputBuf> = BufId::from_raw(5);
        let matvec_in: BufId<MatvecIn> = normed.into();
        assert_eq!(matvec_in.raw(), 5);
    }

    #[test]
    fn moe_input_converts_to_matvec_and_rms_out() {
        let moe_in: BufId<MoeInputBuf> = BufId::from_raw(99);
        let m: BufId<MatvecIn> = moe_in.into();
        assert_eq!(m.raw(), 99);
        let r: BufId<RmsNormOut> = moe_in.into();
        assert_eq!(r.raw(), 99);
    }
}
