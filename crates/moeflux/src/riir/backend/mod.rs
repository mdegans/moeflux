//! Backend-agnostic graph compiler for the RIIR forward path.
//!
//! This module defines the *shape* of the IR: a typed [`Op`] enum,
//! a [`Graph`] = `Vec<Op>` dispatch list, the [`BufferPool`] and
//! [`Backend`] traits that abstract over Metal / CPU / future
//! CoreML or CUDA backends, and the [`BufId`] / [`WeightRef`]
//! handles producers use to reference intermediate and weight
//! tensors.
//!
//! S7-1 ships types and tests only — no backend impls. S7-2 wires
//! [`CpuBackend`]; S7-3 wires [`MetalBackend`]. S7-4 ships the
//! [`graph_metal_matches_cpu`] diff oracle.
//!
//! ## Design tenets
//!
//! - **Model-driven op vocabulary.** Variants exist for the ops
//!   our supported models need. No general-purpose tensor algebra;
//!   no constant folding; no shape inference. The graph is a
//!   *dispatch list*, not a compute graph in the GGML sense.
//! - **Insulation from llama.cpp upstream churn.** Producer code
//!   speaks `Op`; the Metal encoder layer is the only thing that
//!   knows about specific kernels. Swap a kernel without touching
//!   producers.
//! - **Backend portability without speculative abstraction.** The
//!   [`Backend`] trait is shaped so a second impl (CoreML, CUDA)
//!   is mechanical — but we don't ship one until we need it.
//!
//! ## What is *not* here
//!
//! - In-place tensor operations are expressed by a single [`BufId`]
//!   appearing in both `reads()` and `writes()` for the same op.
//!   The pool's coloring pass (S7-5) treats this as a "RMW" — the
//!   slot must stay alive across the op.
//! - The graph does not track types beyond `BufId` semantic
//!   labelling. Each [`Op`] variant statically knows the dtype it
//!   expects in each buffer (f32, bf16-packed-u16, u8 quantized,
//!   etc.).

use std::fmt;

use crate::riir::moe::moe_router::ExpertBuckets;

/// Identifier into a [`BufferPool`]. Backend-agnostic; each backend
/// translates `BufId` to its native handle internally (e.g.
/// `metal::Buffer`, `RefCell<Vec<u8>>`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct BufId(pub u32);

impl fmt::Display for BufId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// Reference to a weight tensor in the backend's mmap'd weight
/// file. Carries byte offsets into the file; the backend resolves
/// these against its own representation:
///
/// - `MetalBackend` reads them as offsets into the shared
///   [`crate::riir::mtl_weight_buf::MtlWeightBuf`].
/// - `CpuBackend` reads them as offsets into the
///   [`crate::riir::weight_file::WeightFile`] mmap.
/// - A future CoreML impl would resolve to a pre-loaded MPSGraph
///   constant (keyed by offset for cache reuse).
///
/// Producer code constructs `WeightRef`s from
/// [`crate::riir::layer_weight_cache::LayerWeightCache`] entries;
/// the (w, s, b) triple corresponds to packed-weight bytes, bf16
/// scales, bf16 biases for quantized matvec. For non-quantized
/// weights (bf16, fp32) `s_off` / `b_off` / `bits` are ignored.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct WeightRef {
    pub w_off: u64,
    pub s_off: u64,
    pub b_off: u64,
    pub bits: u32,
}

/// Errors common to graph building / execution that aren't
/// backend-specific. Backends define their own error types with
/// `From` impls into this where appropriate.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("buffer id {0:?} out of range")]
    BadBufId(BufId),
    #[error("buffer size mismatch for {label:?}: expected {expected} bytes, got {actual}")]
    SizeMismatch { label: &'static str, expected: usize, actual: usize },
    /// Backend-specific error escape hatch. Used by [`Backend::open`]
    /// impls to box their typed init error (e.g. `MetalError`) without
    /// forcing this module to depend on backend-specific symbols.
    #[error("backend error: {0}")]
    Backend(Box<dyn std::error::Error + Send + Sync + 'static>),
}

/// Backend-specific buffer pool.
///
/// Each backend's `Handle` is its native buffer representation
/// (e.g. `metal::Buffer` for Metal, `RefCell<Vec<u8>>` for CPU).
/// The pool owns the storage; `BufId`s are stable indices into it.
///
/// Persistent allocations (KV cache, hidden state across layers,
/// weight file views) opt out of [`Self::reset_transient`] via the
/// `persistent` flag and are excluded from the S7-5 lifetime
/// coloring pass.
pub trait BufferPool {
    type Handle;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Reserve a buffer of `bytes` bytes. The returned `BufId` is
    /// stable for the lifetime of the pool (or until
    /// [`Self::reset_transient`] for non-persistent ids).
    ///
    /// `label` is a `&'static str` for debug / inspection only —
    /// the pool may store it, the producer doesn't depend on it
    /// being unique.
    fn alloc(
        &mut self,
        bytes: usize,
        label: &'static str,
        persistent: bool,
    ) -> Result<BufId, Self::Error>;

    /// Look up a buffer's backend-native handle.
    ///
    /// Returns `&Self::Handle` so callers can use whatever interior
    /// mutability the backend provides (Metal: writes go through
    /// `.contents()` regardless of Rust-level mutability; CPU:
    /// `RefCell<Vec<u8>>` ).
    fn handle(&self, id: BufId) -> &Self::Handle;

    /// Bulk-copy `host` bytes into the buffer at `id`. Used at
    /// graph-build time for inputs (embeddings, routing tables).
    ///
    /// Prefix semantics: `host` may be *shorter* than the buffer, in
    /// which case only the leading `host.len()` bytes are written.
    /// This lets once-per-run buffers be sized at max chunk width
    /// while a smaller chunk uploads only its rows. A `host` *longer*
    /// than the buffer is rejected with `SizeMismatch`.
    fn upload(&mut self, id: BufId, host: &[u8]) -> Result<(), Self::Error>;

    /// Bulk-copy `host` bytes into the buffer at `id` starting at
    /// byte `offset`. Like [`Self::upload`] but for filling one
    /// sub-region of a larger buffer — e.g. one expert's weight block
    /// within a packed expert-weight staging buffer. `offset +
    /// host.len()` must fit the buffer or `SizeMismatch` is returned.
    fn upload_at(
        &mut self,
        id: BufId,
        offset: usize,
        host: &[u8],
    ) -> Result<(), Self::Error>;

    /// Bulk-copy bytes out of the buffer at `id` into `host`. Used
    /// for the routing readback at the two-phase split. Prefix
    /// semantics mirror [`Self::upload`].
    fn download(&self, id: BufId, host: &mut [u8]) -> Result<(), Self::Error>;

    /// Release all non-persistent allocations. Persistent buffers
    /// keep their `BufId`s; transient ones are eligible to be
    /// recycled or dropped on the next [`Self::alloc`].
    fn reset_transient(&mut self);

    /// Label of the buffer at `id`, for [`Graph::dump`] inspection.
    fn label(&self, id: BufId) -> &'static str;

    /// Apply lifetime-aware buffer aliasing for `graph`. After this
    /// call, multiple `BufId`s with disjoint live ranges may share a
    /// single physical buffer, reducing `physical_buffer_count()`.
    ///
    /// Default impl is a no-op (preserves backwards compatibility
    /// for tests that don't need aliasing). Concrete impls override
    /// it via [`lifetime::analyze_lifetimes`] +
    /// [`lifetime::greedy_color`].
    ///
    /// **Contract:**
    /// - Persistent BufIds are never aliased (their physical buffer
    ///   survives [`Self::reset_transient`]).
    /// - Non-colorable transient BufIds (those that appear only in
    ///   `Op::reads()`, never in `Op::writes()`) are never aliased;
    ///   their content is preserved (it was uploaded externally).
    /// - Colorable BufIds (written by some `Op`) may share physical
    ///   storage with other colorable BufIds whose live ranges don't
    ///   overlap. Their pre-`commit_plan` content is NOT preserved
    ///   — they'll be re-written when `Backend::execute` runs.
    /// - Called once, after all `alloc()`s for the graph are done
    ///   and before `Backend::execute`. Multiple calls are allowed
    ///   but unnecessary.
    /// - After coloring, every colored BufId is **pinned**
    ///   (`persistent` set true): its physical layout is frozen for
    ///   the run, so it — and the shared color buffer it now points
    ///   at — survive [`Self::reset_transient`]. A run-lifetime
    ///   scratch set is therefore allocated `persistent = false`,
    ///   `commit_plan`'d once, and thereafter behaves as persistent.
    fn commit_plan(&mut self, _graph: &Graph) {}
}

/// Backend trait. Owns the device / executor + pool + pipeline /
/// compiled-graph cache. Encoding is `&self`-typed; backends use
/// interior mutability for any state mutation during encode.
///
/// The three-step flow ([`Self::begin_encoding`],
/// [`Self::encode_op`], [`Self::submit_and_wait`]) is exposed for
/// callers that need fine-grained control; most callsites use the
/// convenience [`Self::execute`] which runs the full cycle.
pub trait Backend {
    type Pool: BufferPool;
    type EncodeCtx;
    /// Backend-specific construction inputs. Concrete backends define
    /// their own `Config` struct (e.g. `MetalConfig { metal, wf_buf }`,
    /// `CpuConfig { wf }`); [`Self::open`] consumes one to produce a
    /// ready-to-use backend instance.
    type Config;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Construct a backend from its `Config`. Each impl pre-warms
    /// whatever pipelines / compiled-graphs / etc. it needs so the
    /// encode surface (`encode_op`, `submit_and_wait`) can stay
    /// `&self`-typed afterwards.
    ///
    /// `Self: Sized` is a method-level bound (not trait-level) so
    /// trait-object use of the rest of `Backend` is not foreclosed by
    /// this constructor.
    fn open(config: Self::Config) -> Result<Self, Self::Error>
    where
        Self: Sized;

    fn pool(&self) -> &Self::Pool;
    fn pool_mut(&mut self) -> &mut Self::Pool;

    /// Open an encoding session.
    ///
    /// - **Metal:** allocates and returns a fresh `CommandBuffer`.
    /// - **CPU:** returns `()` — encoding *is* execution.
    /// - **CoreML (future):** returns a fresh MPSGraph builder.
    fn begin_encoding(&self) -> Self::EncodeCtx;

    /// Encode one op into `ctx`.
    ///
    /// - **Metal:** appends compute dispatches to the cmdbuf.
    /// - **CPU:** runs the kernel inline, writing through the
    ///   pool's `RefCell<Vec<u8>>` handles.
    ///
    /// `&self`-typed by design; if a backend needs mutable internal
    /// state during encode (e.g. a stats accumulator), it uses
    /// interior mutability (Mutex / RefCell).
    fn encode_op(&self, op: &Op, ctx: &mut Self::EncodeCtx);

    /// Encode an entire graph. Default impl walks ops linearly.
    /// Backends override only for non-linear scheduling (e.g.
    /// parallel encode — session 8+ work).
    fn encode_graph(&self, graph: &Graph, ctx: &mut Self::EncodeCtx) {
        for op in &graph.ops {
            self.encode_op(op, ctx);
        }
    }

    /// Submit the encoded work and block until done.
    ///
    /// `label` is a `'static` tag for this submission. Backends that
    /// track per-label cmdbuf timing record the wall-clock under it
    /// (Metal, via `commit_and_wait_labeled`); other backends ignore
    /// it.
    ///
    /// - **Metal:** `cmdbuf.commit()` + `wait_until_completed()`,
    ///   timed under `label`.
    /// - **CPU:** no-op (already executed inline during
    ///   [`Self::encode_op`]).
    /// - **CoreML (future):** `executable.run()`.
    fn submit_and_wait(
        &self,
        ctx: Self::EncodeCtx,
        label: &'static str,
    ) -> Result<(), Self::Error>;

    /// Convenience: full begin → encode → submit cycle. `label` tags
    /// the submission for per-label timing — see [`Self::submit_and_wait`].
    fn execute(
        &self,
        graph: &Graph,
        label: &'static str,
    ) -> Result<(), Self::Error> {
        let mut ctx = self.begin_encoding();
        self.encode_graph(graph, &mut ctx);
        self.submit_and_wait(ctx, label)
    }
}

/// One typed dispatch into the backend.
///
/// Each variant carries the buffers, weight refs, and scalar dims
/// needed for one kernel call. The `label` field is producer-
/// supplied (`"layer_5_q_proj_matvec"`, `"input_rms_norm"`, etc.)
/// and surfaces in [`Graph::dump`] and any future backend
/// inspection.
///
/// ## Field naming convention
///
/// - Buffers that are *read* by the op are listed first.
/// - Buffers that are *written* by the op are listed after.
/// - In-place ops (e.g. [`Op::RmsNormQkNTokens`]) declare the same
///   buffer in both `reads()` and `writes()`.
/// - Dims are `u32` (matches Metal kernel arg types).
/// - Weight-file offsets are `u64` (file is mmap'd, may exceed
///   4 GiB).
///
/// ## What is *not* in this enum
///
/// MLA variants (`MlaQPrime4Bit`, `MlaSdpaTileAccumulate`,
/// `MlaSdpaTileFinalize`) and full-attn-GPU variants
/// (`RopeApplyNTokens`, `KvCacheAppendNTokens`,
/// `SigmoidGateNTokens`) are reserved for future sessions when
/// their producers are rewritten. Don't add unused variants —
/// each one expands the `encode_op` match arms in every backend.
#[derive(Debug)]
pub enum Op {
    /// Fused RMS-norm with bf16 weight over `[n_tokens, dim]`.
    ///
    /// Used for input rms_norm + post-attn rms_norm. One
    /// threadgroup per token; sum_sq stays in tg-mem.
    RmsNormBf16NTokens {
        label: &'static str,
        x: BufId,
        weight_off: u64,
        out: BufId,
        dim: u32,
        n_tokens: u32,
        eps: f32,
    },

    /// Per-head Q/K RMS norm, in-place on `conv_out` / projection
    /// buffer. Operates on the q region at offset 0 and the k
    /// region at offset `key_offset_per_token` (in floats) within
    /// each token's slot. The slot itself is `per_token_total`
    /// floats — for `q|k` layouts this equals `key_offset_per_token
    /// + num_k_heads * key_dim`, but for `q|k|v` layouts (linear-
    /// attn `conv_out`) it must include the trailing V region.
    ///
    /// Dispatched as a single batched kernel — `(num_k_heads,
    /// n_tokens)` threadgroups, one per (head, token).
    RmsNormQkNTokens {
        label: &'static str,
        x: BufId,
        num_k_heads: u32,
        key_dim: u32,
        key_offset_per_token: u32,
        /// Per-token slot stride in floats. Must match the actual
        /// per-token element count of `x`. The kernel computes the
        /// per-token base offset as `t * per_token_total * 4`.
        per_token_total: u32,
        n_tokens: u32,
    },

    /// Residual add over `[n_tokens, dim]`: `out = a + b`.
    ResidualAddNTokens {
        label: &'static str,
        a: BufId,
        b: BufId,
        out: BufId,
        n_tokens: u32,
        dim: u32,
    },

    /// Zero the leading `n_bytes` of `buf`. Used to clear a
    /// run-lifetime scratch accumulator (e.g. the MoE permute-fuse
    /// `out_sum`, which the bucket kernel scatter-*adds* into) before
    /// it is reused for a new step. `buf` is treated as written-only
    /// (no read) so lifetime coloring sees a clean def point.
    ZeroBuffer {
        label: &'static str,
        buf: BufId,
        n_bytes: u32,
    },

    /// Quantized matvec over n_tokens. 4-bit or 8-bit is selected
    /// by `weight.bits`. Offsets allow the input/output to be
    /// views into larger stacked buffers (Q/K/V proj split).
    MatvecNTokens {
        label: &'static str,
        weight: WeightRef,
        input: BufId,
        input_off: u64,
        output: BufId,
        output_off: u64,
        in_dim: u32,
        out_dim: u32,
        n_tokens: u32,
    },

    /// SwiGLU element-wise fused: `out[i] = silu(gate[i]) * up[i]`
    /// over `total` elements (typically `n_tokens *
    /// ffn_intermediate_dim` or, for permuted MoE, the bucket-flat
    /// equivalent).
    SwigluFusedBatched {
        label: &'static str,
        gate: BufId,
        up: BufId,
        out: BufId,
        total: u32,
    },

    /// Batched tiled causal SDPA. Three accumulator buffers
    /// (`running_max`, `running_denom`, `v_partial`) survive across
    /// tile dispatches; the encoder issues N tile-passes + a
    /// finalize pass.
    SdpaCausalTiled {
        label: &'static str,
        q: BufId,
        k: BufId,
        v: BufId,
        attn_out: BufId,
        running_max: BufId,
        running_denom: BufId,
        v_partial: BufId,
        n_tokens: u32,
        num_heads: u32,
        heads_per_kv: u32,
        head_dim: u32,
        kv_dim: u32,
        kv_start: u32,
        kv_len_total: u32,
        softmax_scale: f32,
    },

    /// MoE softmax + top-K selection. Reads `[n_tokens, n_experts]`
    /// logits, writes `[n_tokens, k]` indices and `[n_tokens, k]`
    /// weights.
    MoeSoftmaxTopK {
        label: &'static str,
        logits: BufId,
        indices_out: BufId,
        weights_out: BufId,
        n_tokens: u32,
        n_experts: u32,
        k: u32,
    },

    /// Normalize MoE weights to sum=1 per token. Operates in-place
    /// on `[n_tokens, k]` weights.
    MoeNormalizeWeights {
        label: &'static str,
        weights: BufId,
        n_tokens: u32,
        k: u32,
    },

    /// Bucket-driven expert FFN dispatch. The producer pre-builds
    /// [`ExpertBuckets`] on CPU (from the routing readback) and
    /// embeds the metadata directly in the op.
    ///
    /// Expert weights live in a single `expert_base` buffer addressed
    /// at uniform `expert_stride` byte stride: bucket `bi` uses the
    /// expert block at `expert_base + expert_slots[bi] * expert_
    /// stride`. `expert_indices` is the per-assignment-row expansion
    /// of `expert_slots` (`expert_slots` expanded by `buckets.
    /// offsets`) — the gather kernel's row→slot table.
    ///
    /// All bucket buffers in this op are bucket-flat: indexed by
    /// `(bucket_offset + slot_within_bucket)`. The combine into
    /// per-token `out_sum` is the kernel's responsibility (or the
    /// CPU oracle's, in [`CpuBackend`]).
    MoeBatchedPermuteFuse {
        label: &'static str,
        /// Base buffer holding every expert's packed weight block.
        expert_base: BufId,
        /// Byte stride between consecutive expert blocks in
        /// `expert_base`.
        expert_stride: u64,
        /// Per-assignment-row expert slot (`u32`). Length =
        /// `total_assignments`; the gather kernel's `indices`.
        expert_indices: BufId,
        /// Per-bucket expert slot into `expert_base`. Length =
        /// `buckets.expert_ids.len()`; the per-bucket fallback's
        /// selector.
        expert_slots: Vec<u32>,
        bucket_input: BufId,
        bucket_gate: BufId,
        bucket_up: BufId,
        bucket_act: BufId,
        bucket_out: BufId,
        bucket_token_idx: BufId,
        bucket_weights: BufId,
        out_sum: BufId,
        buckets: ExpertBuckets,
    },

    /// MoE combine + residual:
    /// `hidden_out[t,i] = h_mid[t,i] + moe_sum[t,i]
    ///                  + sigmoid(shared_gate[t]) * shared_out[t,i]`.
    MoeCombineResidualNTokens {
        label: &'static str,
        h_mid: BufId,
        moe_sum: BufId,
        shared_out: BufId,
        shared_gate: BufId,
        hidden_out: BufId,
        n_tokens: u32,
        dim: u32,
    },

    /// Linear-attn 1b: per-token conv1d step over n_tokens. The
    /// `conv_state` buffer is persistent per layer (carries the
    /// kernel-window of prior values forward across tokens).
    Conv1dStepNTokens {
        label: &'static str,
        qkv_in: BufId,
        conv_state: BufId,
        weight_off: u64,
        conv_out: BufId,
        conv_dim: u32,
        n_tokens: u32,
    },

    /// Linear-attn 1c: compute g_decay + beta_gate from alpha / beta
    /// projections, with bf16 a_log + dt_bias weights.
    ComputeDecayBetaNTokens {
        label: &'static str,
        alpha_in: BufId,
        beta_in: BufId,
        a_log_off: u64,
        dt_bias_off: u64,
        g_decay_out: BufId,
        beta_gate_out: BufId,
        num_v_heads: u32,
        n_tokens: u32,
    },

    /// Linear-attn 1d: gated DeltaNet SSM recurrence step. The
    /// `state` buffer is persistent per layer (carries the SSM
    /// hidden state forward).
    GatedDeltaNetStepNTokens {
        label: &'static str,
        state: BufId,
        conv_out: BufId,
        g_decay: BufId,
        beta_gate: BufId,
        output: BufId,
        num_v_heads: u32,
        value_dim: u32,
        k_heads_per_v: u32,
        n_tokens: u32,
    },

    /// Linear-attn 1e: gated RMS norm over `[n_tokens, num_v_heads
    /// * value_dim]`.
    GatedRmsNormNTokens {
        label: &'static str,
        values: BufId,
        z: BufId,
        weight_off: u64,
        output: BufId,
        num_v_heads: u32,
        value_dim: u32,
        n_tokens: u32,
        eps: f32,
    },

    /// Final RMS norm + lm_head matvec for a single token row of
    /// `hidden`. Used at chunk end to produce logits for the last
    /// (or specified) token. `last_token_row` indexes into
    /// `hidden`'s n_tokens dimension.
    LmHead {
        label: &'static str,
        hidden: BufId,
        last_token_row: u32,
        logits_out: BufId,
    },
}

impl Op {
    /// Producer-supplied label for inspection / debug.
    pub fn label(&self) -> &'static str {
        match self {
            Op::RmsNormBf16NTokens { label, .. } => label,
            Op::RmsNormQkNTokens { label, .. } => label,
            Op::ResidualAddNTokens { label, .. } => label,
            Op::ZeroBuffer { label, .. } => label,
            Op::MatvecNTokens { label, .. } => label,
            Op::SwigluFusedBatched { label, .. } => label,
            Op::SdpaCausalTiled { label, .. } => label,
            Op::MoeSoftmaxTopK { label, .. } => label,
            Op::MoeNormalizeWeights { label, .. } => label,
            Op::MoeBatchedPermuteFuse { label, .. } => label,
            Op::MoeCombineResidualNTokens { label, .. } => label,
            Op::Conv1dStepNTokens { label, .. } => label,
            Op::ComputeDecayBetaNTokens { label, .. } => label,
            Op::GatedDeltaNetStepNTokens { label, .. } => label,
            Op::GatedRmsNormNTokens { label, .. } => label,
            Op::LmHead { label, .. } => label,
        }
    }

    /// The variant name as a `&'static str`. For `Graph::dump`
    /// and any future IR text format.
    pub fn variant_name(&self) -> &'static str {
        match self {
            Op::RmsNormBf16NTokens { .. } => "RmsNormBf16NTokens",
            Op::RmsNormQkNTokens { .. } => "RmsNormQkNTokens",
            Op::ResidualAddNTokens { .. } => "ResidualAddNTokens",
            Op::ZeroBuffer { .. } => "ZeroBuffer",
            Op::MatvecNTokens { .. } => "MatvecNTokens",
            Op::SwigluFusedBatched { .. } => "SwigluFusedBatched",
            Op::SdpaCausalTiled { .. } => "SdpaCausalTiled",
            Op::MoeSoftmaxTopK { .. } => "MoeSoftmaxTopK",
            Op::MoeNormalizeWeights { .. } => "MoeNormalizeWeights",
            Op::MoeBatchedPermuteFuse { .. } => "MoeBatchedPermuteFuse",
            Op::MoeCombineResidualNTokens { .. } => "MoeCombineResidualNTokens",
            Op::Conv1dStepNTokens { .. } => "Conv1dStepNTokens",
            Op::ComputeDecayBetaNTokens { .. } => "ComputeDecayBetaNTokens",
            Op::GatedDeltaNetStepNTokens { .. } => "GatedDeltaNetStepNTokens",
            Op::GatedRmsNormNTokens { .. } => "GatedRmsNormNTokens",
            Op::LmHead { .. } => "LmHead",
        }
    }

    /// BufIds this op *reads from*. Includes RMW buffers (which
    /// also appear in [`Self::writes`]).
    ///
    /// Consumed by the S7-5 lifetime-coloring pass to compute each
    /// BufId's last-read index. Producers don't need to call this
    /// directly.
    pub fn reads(&self) -> Vec<BufId> {
        match self {
            Op::RmsNormBf16NTokens { x, .. } => vec![*x],
            Op::RmsNormQkNTokens { x, .. } => vec![*x],
            Op::ResidualAddNTokens { a, b, .. } => vec![*a, *b],
            Op::ZeroBuffer { .. } => vec![],
            Op::MatvecNTokens { input, .. } => vec![*input],
            Op::SwigluFusedBatched { gate, up, .. } => vec![*gate, *up],
            Op::SdpaCausalTiled {
                q,
                k,
                v,
                running_max,
                running_denom,
                v_partial,
                ..
            } => vec![*q, *k, *v, *running_max, *running_denom, *v_partial],
            Op::MoeSoftmaxTopK { logits, .. } => vec![*logits],
            Op::MoeNormalizeWeights { weights, .. } => vec![*weights],
            Op::MoeBatchedPermuteFuse {
                expert_base,
                expert_indices,
                bucket_input,
                bucket_token_idx,
                bucket_weights,
                ..
            } => vec![
                *expert_base,
                *expert_indices,
                *bucket_input,
                *bucket_token_idx,
                *bucket_weights,
            ],
            Op::MoeCombineResidualNTokens {
                h_mid,
                moe_sum,
                shared_out,
                shared_gate,
                ..
            } => vec![*h_mid, *moe_sum, *shared_out, *shared_gate],
            Op::Conv1dStepNTokens {
                qkv_in,
                conv_state,
                ..
            } => vec![*qkv_in, *conv_state],
            Op::ComputeDecayBetaNTokens {
                alpha_in, beta_in, ..
            } => vec![*alpha_in, *beta_in],
            Op::GatedDeltaNetStepNTokens {
                state,
                conv_out,
                g_decay,
                beta_gate,
                ..
            } => vec![*state, *conv_out, *g_decay, *beta_gate],
            Op::GatedRmsNormNTokens { values, z, .. } => vec![*values, *z],
            Op::LmHead { hidden, .. } => vec![*hidden],
        }
    }

    /// BufIds this op *writes to*. RMW buffers also appear in
    /// [`Self::reads`].
    pub fn writes(&self) -> Vec<BufId> {
        match self {
            Op::RmsNormBf16NTokens { out, .. } => vec![*out],
            Op::RmsNormQkNTokens { x, .. } => vec![*x], // in-place
            Op::ResidualAddNTokens { out, .. } => vec![*out],
            Op::ZeroBuffer { buf, .. } => vec![*buf],
            Op::MatvecNTokens { output, .. } => vec![*output],
            Op::SwigluFusedBatched { out, .. } => vec![*out],
            Op::SdpaCausalTiled {
                attn_out,
                running_max,
                running_denom,
                v_partial,
                ..
            } => vec![*attn_out, *running_max, *running_denom, *v_partial],
            Op::MoeSoftmaxTopK {
                indices_out,
                weights_out,
                ..
            } => vec![*indices_out, *weights_out],
            Op::MoeNormalizeWeights { weights, .. } => vec![*weights], // in-place
            Op::MoeBatchedPermuteFuse {
                bucket_gate,
                bucket_up,
                bucket_act,
                bucket_out,
                out_sum,
                ..
            } => vec![*bucket_gate, *bucket_up, *bucket_act, *bucket_out, *out_sum],
            Op::MoeCombineResidualNTokens { hidden_out, .. } => vec![*hidden_out],
            Op::Conv1dStepNTokens {
                conv_state,
                conv_out,
                ..
            } => vec![*conv_state, *conv_out], // conv_state is RMW
            Op::ComputeDecayBetaNTokens {
                g_decay_out,
                beta_gate_out,
                ..
            } => vec![*g_decay_out, *beta_gate_out],
            Op::GatedDeltaNetStepNTokens { state, output, .. } => vec![*state, *output], // state is RMW
            Op::GatedRmsNormNTokens { output, .. } => vec![*output],
            Op::LmHead { logits_out, .. } => vec![*logits_out],
        }
    }
}

/// A backend-agnostic dispatch list. Built incrementally by
/// producer code (`graph.push(Op::...)`) and consumed by
/// [`Backend::execute`].
#[derive(Debug, Default)]
pub struct Graph {
    pub ops: Vec<Op>,
}

impl Graph {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn push(&mut self, op: Op) {
        self.ops.push(op);
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Iterate labels in op order.
    pub fn labels(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.ops.iter().map(Op::label)
    }

    /// Multi-line debug dump, one line per op. Polished in S7-9
    /// with per-variant arg summaries; for now produces
    /// `{idx:3}  {variant:<28}  {label}`.
    pub fn dump(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        for (i, op) in self.ops.iter().enumerate() {
            let _ = writeln!(
                s,
                "{i:3}  {variant:<28}  {label}",
                variant = op.variant_name(),
                label = op.label(),
            );
        }
        s
    }
}

pub mod cpu;
pub mod gpu;
pub mod lifetime;

pub use cpu::{CpuBackend, CpuBufferPool};
pub use gpu::{MetalBackend, MetalBufferPool, MetalConfig, MetalEncodeCtx};

#[cfg(test)]
mod tests {
    use super::*;

    fn buf(n: u32) -> BufId {
        BufId(n)
    }

    /// One of every variant, with minimal-ish stub fields.
    fn one_of_each() -> Graph {
        let mut g = Graph::new();
        g.push(Op::RmsNormBf16NTokens {
            label: "rms_in",
            x: buf(0),
            weight_off: 0,
            out: buf(1),
            dim: 4096,
            n_tokens: 8,
            eps: 1e-6,
        });
        g.push(Op::RmsNormQkNTokens {
            label: "qk_norm",
            x: buf(2),
            num_k_heads: 4,
            key_dim: 128,
            key_offset_per_token: 512,
            per_token_total: 1024,
            n_tokens: 8,
        });
        g.push(Op::ResidualAddNTokens {
            label: "resid",
            a: buf(3),
            b: buf(4),
            out: buf(5),
            n_tokens: 8,
            dim: 4096,
        });
        g.push(Op::MatvecNTokens {
            label: "q_proj",
            weight: WeightRef { w_off: 0, s_off: 0, b_off: 0, bits: 4 },
            input: buf(6),
            input_off: 0,
            output: buf(7),
            output_off: 0,
            in_dim: 4096,
            out_dim: 4096,
            n_tokens: 8,
        });
        g.push(Op::SwigluFusedBatched {
            label: "ffn_swiglu",
            gate: buf(8),
            up: buf(9),
            out: buf(10),
            total: 8 * 1024,
        });
        g.push(Op::SdpaCausalTiled {
            label: "sdpa",
            q: buf(11),
            k: buf(12),
            v: buf(13),
            attn_out: buf(14),
            running_max: buf(15),
            running_denom: buf(16),
            v_partial: buf(17),
            n_tokens: 8,
            num_heads: 16,
            heads_per_kv: 2,
            head_dim: 128,
            kv_dim: 1024,
            kv_start: 0,
            kv_len_total: 8,
            softmax_scale: 0.088_388_35,
        });
        g.push(Op::MoeSoftmaxTopK {
            label: "moe_topk",
            logits: buf(18),
            indices_out: buf(19),
            weights_out: buf(20),
            n_tokens: 8,
            n_experts: 128,
            k: 8,
        });
        g.push(Op::MoeNormalizeWeights {
            label: "moe_norm",
            weights: buf(20),
            n_tokens: 8,
            k: 8,
        });
        g.push(Op::MoeBatchedPermuteFuse {
            label: "moe_pf",
            expert_base: buf(21),
            expert_stride: 0,
            expert_indices: buf(35),
            expert_slots: vec![0],
            bucket_input: buf(22),
            bucket_gate: buf(23),
            bucket_up: buf(24),
            bucket_act: buf(25),
            bucket_out: buf(26),
            bucket_token_idx: buf(27),
            bucket_weights: buf(28),
            out_sum: buf(29),
            buckets: ExpertBuckets {
                expert_ids: vec![0],
                offsets: vec![0, 8],
                token_idx: vec![0, 1, 2, 3, 4, 5, 6, 7],
                weights: vec![0.125; 8],
            },
        });
        g.push(Op::MoeCombineResidualNTokens {
            label: "moe_combine",
            h_mid: buf(30),
            moe_sum: buf(31),
            shared_out: buf(32),
            shared_gate: buf(33),
            hidden_out: buf(34),
            n_tokens: 8,
            dim: 4096,
        });
        g.push(Op::Conv1dStepNTokens {
            label: "conv1d",
            qkv_in: buf(35),
            conv_state: buf(36),
            weight_off: 0,
            conv_out: buf(37),
            conv_dim: 5120,
            n_tokens: 8,
        });
        g.push(Op::ComputeDecayBetaNTokens {
            label: "decay_beta",
            alpha_in: buf(38),
            beta_in: buf(39),
            a_log_off: 0,
            dt_bias_off: 0,
            g_decay_out: buf(40),
            beta_gate_out: buf(41),
            num_v_heads: 16,
            n_tokens: 8,
        });
        g.push(Op::GatedDeltaNetStepNTokens {
            label: "delta_net",
            state: buf(42),
            conv_out: buf(43),
            g_decay: buf(44),
            beta_gate: buf(45),
            output: buf(46),
            num_v_heads: 16,
            value_dim: 128,
            k_heads_per_v: 2,
            n_tokens: 8,
        });
        g.push(Op::GatedRmsNormNTokens {
            label: "gated_rms",
            values: buf(47),
            z: buf(48),
            weight_off: 0,
            output: buf(49),
            num_v_heads: 16,
            value_dim: 128,
            n_tokens: 8,
            eps: 1e-6,
        });
        g.push(Op::LmHead {
            label: "lm_head",
            hidden: buf(50),
            last_token_row: 7,
            logits_out: buf(51),
        });
        g
    }

    #[test]
    fn push_round_trips() {
        let g = one_of_each();
        assert_eq!(g.len(), 15);
        assert!(!g.is_empty());
    }

    #[test]
    fn labels_iter_matches_push_order() {
        let g = one_of_each();
        let labels: Vec<&str> = g.labels().collect();
        assert_eq!(
            labels,
            vec![
                "rms_in",
                "qk_norm",
                "resid",
                "q_proj",
                "ffn_swiglu",
                "sdpa",
                "moe_topk",
                "moe_norm",
                "moe_pf",
                "moe_combine",
                "conv1d",
                "decay_beta",
                "delta_net",
                "gated_rms",
                "lm_head",
            ]
        );
    }

    #[test]
    fn variant_name_matches_label_for_each_variant() {
        let g = one_of_each();
        let pairs: Vec<(&str, &str)> = g
            .ops
            .iter()
            .map(|op| (op.variant_name(), op.label()))
            .collect();
        // Spot-check the discriminant-aware naming.
        assert!(pairs.contains(&("RmsNormBf16NTokens", "rms_in")));
        assert!(pairs.contains(&("MoeBatchedPermuteFuse", "moe_pf")));
        assert!(pairs.contains(&("LmHead", "lm_head")));
        assert_eq!(pairs.len(), 15);
    }

    #[test]
    fn reads_and_writes_are_non_empty_for_every_variant() {
        let g = one_of_each();
        for op in &g.ops {
            assert!(
                !op.reads().is_empty(),
                "{} produced empty reads()",
                op.variant_name()
            );
            assert!(
                !op.writes().is_empty(),
                "{} produced empty writes()",
                op.variant_name()
            );
        }
    }

    #[test]
    fn in_place_ops_appear_in_both_reads_and_writes() {
        let g = Graph {
            ops: vec![
                Op::RmsNormQkNTokens {
                    label: "qk",
                    x: buf(2),
                    num_k_heads: 4,
                    key_dim: 128,
                    key_offset_per_token: 512,
                    per_token_total: 1024,
                    n_tokens: 8,
                },
                Op::MoeNormalizeWeights {
                    label: "moe_norm",
                    weights: buf(20),
                    n_tokens: 8,
                    k: 8,
                },
                Op::Conv1dStepNTokens {
                    label: "conv1d",
                    qkv_in: buf(35),
                    conv_state: buf(36),
                    weight_off: 0,
                    conv_out: buf(37),
                    conv_dim: 5120,
                    n_tokens: 8,
                },
            ],
        };
        // RmsNormQkNTokens: x is read and written
        assert!(g.ops[0].reads().contains(&buf(2)));
        assert!(g.ops[0].writes().contains(&buf(2)));
        // MoeNormalizeWeights: weights is read and written
        assert!(g.ops[1].reads().contains(&buf(20)));
        assert!(g.ops[1].writes().contains(&buf(20)));
        // Conv1dStepNTokens: conv_state is read and written
        assert!(g.ops[2].reads().contains(&buf(36)));
        assert!(g.ops[2].writes().contains(&buf(36)));
    }

    #[test]
    fn dump_emits_one_line_per_op() {
        let g = one_of_each();
        let dump = g.dump();
        let line_count = dump.lines().count();
        assert_eq!(line_count, 15);
        // Spot-check formatting: each line has the variant name and label.
        assert!(dump.contains("RmsNormBf16NTokens"));
        assert!(dump.contains("rms_in"));
        assert!(dump.contains("MoeBatchedPermuteFuse"));
        assert!(dump.contains("moe_pf"));
    }

    #[test]
    fn dump_snapshot_tiny_graph() {
        let mut g = Graph::new();
        g.push(Op::RmsNormBf16NTokens {
            label: "rms_in",
            x: buf(0),
            weight_off: 0,
            out: buf(1),
            dim: 64,
            n_tokens: 2,
            eps: 1e-6,
        });
        g.push(Op::ResidualAddNTokens {
            label: "resid",
            a: buf(1),
            b: buf(0),
            out: buf(2),
            n_tokens: 2,
            dim: 64,
        });
        let expected = concat!(
            "  0  RmsNormBf16NTokens            rms_in\n",
            "  1  ResidualAddNTokens            resid\n",
        );
        assert_eq!(g.dump(), expected);
    }

    #[test]
    fn bufid_display_uses_percent_prefix() {
        assert_eq!(format!("{}", buf(42)), "%42");
    }

}
