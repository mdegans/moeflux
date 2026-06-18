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

use crate::riir::moe::moe_router::ExpertBuckets;

pub mod buftype;
pub use buftype::*;

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
    #[error("buffer id {0} out of range")]
    BadBufId(u32),
    #[error("buffer size mismatch for {label:?}: expected {expected} bytes, got {actual}")]
    SizeMismatch {
        label: &'static str,
        expected: usize,
        actual: usize,
    },
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

    /// Reserve a buffer of `bytes` bytes. The returned `BufId<B>` is
    /// stable for the lifetime of the pool (or until
    /// [`Self::reset_transient`] for non-persistent ids).
    ///
    /// Generic on the buffer's role tag `B: Buf` — producers spell the
    /// tag explicitly at the alloc site (e.g. `pool.alloc::<MoeInputBuf>(...)`)
    /// so the returned id carries the role through the type system.
    /// `label` is a `&'static str` for debug / inspection only.
    fn alloc<B: Buf>(
        &mut self,
        bytes: usize,
        label: &'static str,
        persistent: bool,
    ) -> Result<BufId<B>, Self::Error>;

    /// Look up a buffer's backend-native handle.
    ///
    /// Returns `&Self::Handle` so callers can use whatever interior
    /// mutability the backend provides (Metal: writes go through
    /// `.contents()` regardless of Rust-level mutability; CPU:
    /// `RefCell<Vec<u8>>` ).
    fn handle<B: Buf>(&self, id: BufId<B>) -> &Self::Handle;

    /// Bulk-copy `host` bytes into the buffer at `id`. Prefix semantics
    /// (see trait doc).
    fn upload<B: Buf>(&mut self, id: BufId<B>, host: &[u8]) -> Result<(), Self::Error>;

    /// Bulk-copy `host` bytes into the buffer at `id` starting at
    /// byte `offset`. `offset + host.len()` must fit the buffer or
    /// `SizeMismatch` is returned.
    fn upload_at<B: Buf>(
        &mut self,
        id: BufId<B>,
        offset: usize,
        host: &[u8],
    ) -> Result<(), Self::Error>;

    /// Bulk-copy bytes out of the buffer at `id` into `host`. Used
    /// for the routing readback at the two-phase split. Prefix
    /// semantics mirror [`Self::upload`].
    fn download<B: Buf>(&self, id: BufId<B>, host: &mut [u8]) -> Result<(), Self::Error>;

    /// Release all non-persistent allocations. Persistent buffers
    /// keep their `BufId`s; transient ones are eligible to be
    /// recycled or dropped on the next [`Self::alloc`].
    fn reset_transient(&mut self);

    /// Label of the buffer at `id`, for [`Graph::dump`] inspection.
    fn label<B: Buf>(&self, id: BufId<B>) -> &'static str;

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
    fn submit_and_wait(&self, ctx: Self::EncodeCtx, label: &'static str)
    -> Result<(), Self::Error>;

    /// Convenience: full begin → encode → submit cycle. `label` tags
    /// the submission for per-label timing — see [`Self::submit_and_wait`].
    fn execute(&self, graph: &Graph, label: &'static str) -> Result<(), Self::Error> {
        let mut ctx = self.begin_encoding();
        self.encode_graph(graph, &mut ctx);
        self.submit_and_wait(ctx, label)
    }

    /// Hook fired by the prefill orchestrator at the top of each
    /// layer iteration. Default no-op. Currently used by `MetalBackend`
    /// to gate `MTLCaptureManager` start/stop against the env-driven
    /// [`crate::riir::gpu_capture`] window.
    fn begin_layer(&mut self, _chunk_idx: usize, _layer_idx: usize) {}
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
/// `MlaSdpaTileFinalize`) are reserved for future sessions when their
/// producers are rewritten. Don't add unused variants — each one
/// expands the `encode_op` match arms in every backend.
#[derive(Debug)]
pub enum Op {
    /// Fused RMS-norm with bf16 weight over `[n_tokens, dim]`.
    ///
    /// Used for input rms_norm + post-attn rms_norm. One
    /// threadgroup per token; sum_sq stays in tg-mem.
    RmsNormBf16NTokens {
        label: &'static str,
        x: BufId<RmsNormIn>,
        weight_off: u64,
        out: BufId<RmsNormOut>,
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
    /// Linear-attn-only — `x` is the conv1d output (q | k | v stack).
    /// Dispatched as a single batched kernel — `(num_k_heads,
    /// n_tokens)` threadgroups, one per (head, token).
    RmsNormQkNTokens {
        label: &'static str,
        x: BufId<ConvOutBuf>,
        num_k_heads: u32,
        key_dim: u32,
        key_offset_per_token: u32,
        /// Per-token slot stride in floats. Must match the actual
        /// per-token element count of `x`. The kernel computes the
        /// per-token base offset as `t * per_token_total * 4`.
        per_token_total: u32,
        n_tokens: u32,
    },

    /// Weighted per-head RMS norm, in-place over `[n_tokens,
    /// num_heads*head_dim]`. Each `(token, head)` `head_dim`-slice is
    /// independently RMS-normalized and scaled by a shared learned
    /// bf16 weight of length `head_dim` (`weight_off` into the weight
    /// file). Used for full-attn's per-head `q_norm` / `k_norm` — one
    /// Op pushed once per buffer (q and k live in separate buffers,
    /// so unlike [`Op::RmsNormQkNTokens`] they can't fuse into one
    /// dispatch). Diff oracle: `attn::rms_norm::rms_norm_per_head_cpu`.
    ///
    /// In-place; `x` accepts QBuf or KProjOutBuf via the [`RmsNormIn`]
    /// union (existing `From` impls).
    RmsNormPerHeadNTokens {
        label: &'static str,
        x: BufId<RmsNormIn>,
        weight_off: u64,
        num_heads: u32,
        head_dim: u32,
        n_tokens: u32,
        eps: f32,
    },

    /// Vanilla RoPE over an `[n_tokens, num_heads, head_dim]` stack,
    /// in-place. Rotates the first `rotary_dim` channels of each head
    /// (GPT-NeoX half-split layout); token `t`'s absolute position is
    /// `start_pos + t`. `inv_freq` is a precomputed `rotary_dim/2`-
    /// length frequency table — the kernel is agnostic to vanilla vs
    /// YaRN-rescaled, so a future `factor` knob only touches the
    /// table, never this Op. Diff oracle: `attn::rope::apply_rotary_emb`.
    ///
    /// In-place on `x`; `x` accepts QBuf or KProjOutBuf via the
    /// [`RmsNormIn`] union.
    RopeNTokens {
        label: &'static str,
        x: BufId<RmsNormIn>,
        inv_freq: BufId<RopeInvFreqBuf>,
        n_tokens: u32,
        num_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        start_pos: i32,
    },

    /// Deinterleave a fused q-projection into separate q + gate
    /// stacks. `q_proj` is `[n_tokens, num_heads, 2*head_dim]` — each
    /// head laid out `[q (head_dim) | gate (head_dim)]`. Writes
    /// `q_out` and `gate_out`, both `[n_tokens, num_heads*head_dim]`
    /// contiguous. Full-attn's `q_proj` carries the per-head query
    /// gate interleaved with the query; this splits them apart.
    SplitQGate {
        label: &'static str,
        q_proj: BufId<QProjOutBuf>,
        q_out: BufId<QBuf>,
        gate_out: BufId<QGateBuf>,
        num_heads: u32,
        head_dim: u32,
        n_tokens: u32,
    },

    /// Residual add over `[n_tokens, dim]`: `out = a + b`.
    ///
    /// `a` is the o_proj output (the layer's attention contribution);
    /// `b` is the residual stream (`RmsNormIn` union — accepts
    /// `EmbedOutBuf`/`ResidualBuf`/`HiddenBuf` via existing impls).
    ResidualAddNTokens {
        label: &'static str,
        a: BufId<OProjOutBuf>,
        b: BufId<RmsNormIn>,
        out: BufId<ResidualBuf>,
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
        buf: BufId<MoeOutSumBuf>,
        n_bytes: u32,
    },

    /// Quantized matvec over n_tokens. 4-bit or 8-bit is selected
    /// by `weight.bits`. Offsets allow the input/output to be
    /// views into larger stacked buffers (Q/K/V proj split).
    MatvecNTokens {
        label: &'static str,
        weight: WeightRef,
        input: BufId<MatvecIn>,
        input_off: u64,
        output: BufId<MatvecOut>,
        output_off: u64,
        in_dim: u32,
        out_dim: u32,
        n_tokens: u32,
    },

    /// SwiGLU element-wise fused: `out[i] = silu(gate[i]) * up[i]`
    /// over `total` elements. Used for the shared-FFN SwiGLU on
    /// post-norm activations.
    SwigluFusedBatched {
        label: &'static str,
        gate: BufId<SharedFfnGateBuf>,
        up: BufId<SharedFfnUpBuf>,
        out: BufId<SharedFfnActBuf>,
        total: u32,
    },

    /// Batched causal SDPA. `q` is `[n_tokens, num_heads, head_dim]`;
    /// `k`/`v` are the GPU-resident KV cache, `[kv_len_total, kv_dim]`
    /// row-major — token `t` (absolute position `kv_start + t`)
    /// attends causally over rows `[0, kv_start + t]`. The Metal arm
    /// dispatches the production flash kernel (`SdpaCall`); the GQA
    /// fold is derived internally from `heads_per_kv`. Diff oracle:
    /// `attn::sdpa` (single-pass per-token compute).
    SdpaCausalTiled {
        label: &'static str,
        q: BufId<QBuf>,
        k: BufId<KvCacheKBuf>,
        v: BufId<KvCacheVBuf>,
        attn_out: BufId<AttnOutBuf>,
        n_tokens: u32,
        num_heads: u32,
        heads_per_kv: u32,
        head_dim: u32,
        kv_dim: u32,
        kv_start: u32,
        kv_len_total: u32,
        softmax_scale: f32,
    },

    /// Per-token sigmoid gate, in-place on the attention output:
    /// `attn_out[t,i] *= sigmoid(gate[t,i])` over `[n_tokens, dim]`.
    /// Element-wise — `dim` and `n_tokens` are kept for clarity but
    /// the encoder collapses them into one flat dispatch of
    /// `dim * n_tokens` threads. Full-attn per-head query gate.
    SigmoidGateNTokens {
        label: &'static str,
        x: BufId<AttnOutBuf>,
        gate: BufId<QGateBuf>,
        dim: u32,
        n_tokens: u32,
    },

    /// Append per-token k/v scratch into the GPU-resident KV cache.
    /// `k_src`/`v_src` are `[n_tokens, kv_dim]`; the cache buffers are
    /// `[MAX_SEQ_LEN, kv_dim]`. Writes rows `[kv_start, kv_start +
    /// n_tokens)`. Both backends do a strided copy — bit-exact.
    KvCacheAppendNTokens {
        label: &'static str,
        k_src: BufId<KProjOutBuf>,
        v_src: BufId<VProjOutBuf>,
        k_cache: BufId<KvCacheKBuf>,
        v_cache: BufId<KvCacheVBuf>,
        kv_dim: u32,
        n_tokens: u32,
        kv_start: u32,
    },

    /// MoE softmax + top-K selection. Reads `[n_tokens, n_experts]`
    /// logits, writes `[n_tokens, k]` indices and `[n_tokens, k]`
    /// weights.
    MoeSoftmaxTopK {
        label: &'static str,
        logits: BufId<RouterLogitsBuf>,
        indices_out: BufId<RouterIdxBuf>,
        weights_out: BufId<RouterWeightsBuf>,
        n_tokens: u32,
        n_experts: u32,
        k: u32,
    },

    /// Normalize MoE weights to sum=1 per token. Operates in-place
    /// on `[n_tokens, k]` weights.
    MoeNormalizeWeights {
        label: &'static str,
        weights: BufId<RouterWeightsBuf>,
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
        expert_base: BufId<ExpertBaseBuf>,
        /// Byte stride between consecutive expert blocks in
        /// `expert_base`.
        expert_stride: u64,
        /// Per-assignment-row expert slot (`u32`). Length =
        /// `total_assignments`; the gather kernel's `indices`.
        expert_indices: BufId<ExpertIndicesBuf>,
        /// Per-bucket expert slot into `expert_base`. Length =
        /// `buckets.expert_ids.len()`; the per-bucket fallback's
        /// selector.
        expert_slots: Vec<u32>,
        bucket_input: BufId<BucketInputBuf>,
        bucket_gate: BufId<BucketGateBuf>,
        bucket_up: BufId<BucketUpBuf>,
        bucket_act: BufId<BucketActBuf>,
        bucket_out: BufId<BucketOutBuf>,
        bucket_token_idx: BufId<BucketTokenIdxBuf>,
        bucket_weights: BufId<BucketWeightsBuf>,
        out_sum: BufId<MoeOutSumBuf>,
        buckets: ExpertBuckets,
    },

    /// MoE gather-by-expert-id fuse (the `gather_mm_id.metal` path).
    ///
    /// One-dispatch-per-projection MoE matmul (Diff #1 from
    /// `.claude/memory/llama_cpp_moe_differentiators.md`). Replaces
    /// the bucket-permute + per-row-indexed gather of
    /// [`Self::MoeBatchedPermuteFuse`] with: a map0 pre-pass that
    /// builds per-expert assignment lists from the router's
    /// `indices[n_tokens, k]`, then three `moeflux_mm_id` dispatches
    /// (gate / up / down) that internally early-return per-expert
    /// based on `htpe[e]` and gather activations via `hids`.
    ///
    /// Output of gate/up/down is `[n_tokens, k, dim]` slot-indexed;
    /// `combine_topk` reduces by `weights[n_tokens, k]` to produce
    /// `[n_tokens, hidden_dim]` in `out_sum`.
    ///
    /// All scratch buffers (`htpe`, `hids`, `gate_mid`, `up_mid`,
    /// `down_mid`) are pool-allocated per chunk; sized at construction
    /// for the max chunk width. Currently GPU-only — CPU encode arm
    /// `todo!()`s with a pointer to the engine-level GPU/GPU diff
    /// test for validation.
    MoeGatherIdFuse {
        label: &'static str,
        /// Layer's expert blob — `n_experts` experts stacked at
        /// `expert_stride` byte spacing.
        expert_base: BufId<ExpertBaseBuf>,
        /// Per-expert byte stride — `Variant::expert_size_4bit()`.
        expert_stride: u64,
        /// Router output — `[n_tokens, k]` i32.
        indices: BufId<RouterIdxBuf>,
        /// Router output — `[n_tokens, k]` f32.
        weights: BufId<RouterWeightsBuf>,
        /// Input to the MoE-MLP matmul stack — the **post-RmsNorm**
        /// hidden state (`[n_tokens, hidden_dim]` f32). This is the
        /// same value the bucket-permute path feeds its bucket_input
        /// host-permute from. Do NOT pass the pre-norm residual
        /// (`MoeGraphScratch::h_mid`) — that was the session-19 bug,
        /// now a compile error: pre-norm = `BufId<ResidualBuf>`,
        /// post-norm = `BufId<MoeInputBuf>`; the type system rejects
        /// the swap.
        mlp_in: BufId<MoeInputBuf>,
        /// Output — `[n_tokens, hidden_dim]` f32. Accumulator; the
        /// kernel WRITES (not adds). Combine with shared/residual
        /// downstream via [`Self::MoeCombineResidualNTokens`].
        out_sum: BufId<MoeOutSumBuf>,
        /// Scratch — `[n_experts]` u32, per-expert assignment count.
        htpe: BufId<HtpeBuf>,
        /// Scratch — `[n_experts, n_tokens]` i32, per-expert
        /// assignment list (encoded `token*k + slot`).
        hids: BufId<HidsBuf>,
        /// Scratch — `[n_tokens, k, moe_inter]` f32, gate-proj output
        /// (also reused in-place as the SwiGLU output that feeds down).
        gate_mid: BufId<GateMidBuf>,
        /// Scratch — `[n_tokens, k, moe_inter]` f32, up-proj output.
        up_mid: BufId<UpMidBuf>,
        /// Scratch — `[n_tokens, k, hidden_dim]` f32, down-proj
        /// output (input to combine_topk).
        down_mid: BufId<DownMidBuf>,
        n_tokens: u32,
        n_experts: u32,
        /// Top-k. Must be one of `moeflux_metal::MOE_MM_ID_SUPPORTED_TOPK`
        /// (a3b routes 8, a17b routes 10).
        k: u32,
    },

    /// MoE combine + residual:
    /// `hidden_out[t,i] = h_mid[t,i] + moe_sum[t,i]
    ///                  + sigmoid(shared_gate[t]) * shared_out[t,i]`.
    MoeCombineResidualNTokens {
        label: &'static str,
        h_mid: BufId<ResidualBuf>,
        moe_sum: BufId<MoeOutSumBuf>,
        shared_out: BufId<SharedFfnDownBuf>,
        shared_gate: BufId<SharedGateBuf>,
        hidden_out: BufId<HiddenBuf>,
        n_tokens: u32,
        dim: u32,
    },

    /// Linear-attn 1b: per-token conv1d step over n_tokens. The
    /// `conv_state` buffer is persistent per layer (carries the
    /// kernel-window of prior values forward across tokens).
    Conv1dStepNTokens {
        label: &'static str,
        qkv_in: BufId<QkvStackBuf>,
        conv_state: BufId<ConvStateBuf>,
        weight_off: u64,
        conv_out: BufId<ConvOutBuf>,
        conv_dim: u32,
        n_tokens: u32,
    },

    /// Linear-attn 1c: compute g_decay + beta_gate from alpha / beta
    /// projections, with bf16 a_log + dt_bias weights.
    ComputeDecayBetaNTokens {
        label: &'static str,
        alpha_in: BufId<AlphaStackBuf>,
        beta_in: BufId<BetaStackBuf>,
        a_log_off: u64,
        dt_bias_off: u64,
        g_decay_out: BufId<GDecayBuf>,
        beta_gate_out: BufId<BetaGateBuf>,
        num_v_heads: u32,
        n_tokens: u32,
    },

    /// Linear-attn 1d: gated DeltaNet SSM recurrence step. The
    /// `state` buffer is persistent per layer (carries the SSM
    /// hidden state forward).
    GatedDeltaNetStepNTokens {
        label: &'static str,
        state: BufId<DeltaStateBuf>,
        conv_out: BufId<ConvOutBuf>,
        g_decay: BufId<GDecayBuf>,
        beta_gate: BufId<BetaGateBuf>,
        output: BufId<DeltaOutBuf>,
        num_v_heads: u32,
        value_dim: u32,
        k_heads_per_v: u32,
        n_tokens: u32,
    },

    /// Linear-attn 1d, chunkwise-parallel variant of
    /// [`Op::GatedDeltaNetStepNTokens`]. Same delta-rule recurrence
    /// and same buffer roles (`state` is persistent RMW; `conv_out`
    /// is the per-token `q | k | v` stack), but the within-chunk
    /// computation is reformulated as matmuls + a triangular solve so
    /// only the chunk-to-chunk state carry stays sequential.
    /// `chunk_size` is the inner chunk length `C`.
    GatedDeltaNetChunkwise {
        label: &'static str,
        state: BufId<DeltaStateBuf>,
        conv_out: BufId<ConvOutBuf>,
        g_decay: BufId<GDecayBuf>,
        beta_gate: BufId<BetaGateBuf>,
        output: BufId<DeltaOutBuf>,
        num_v_heads: u32,
        value_dim: u32,
        k_heads_per_v: u32,
        n_tokens: u32,
        chunk_size: u32,
    },

    /// Linear-attn 1e: gated RMS norm over `[n_tokens, num_v_heads
    /// * value_dim]`.
    GatedRmsNormNTokens {
        label: &'static str,
        values: BufId<DeltaOutBuf>,
        z: BufId<ZStackBuf>,
        weight_off: u64,
        output: BufId<ValueOutBuf>,
        num_v_heads: u32,
        value_dim: u32,
        n_tokens: u32,
        eps: f32,
    },

    /// Batched 4-bit token-embedding gather. For each of `n_tokens`
    /// tokens, reads row `token_ids[t]` of the affine-packed embedding
    /// weight and dequantizes `hidden_dim` f32 channels into
    /// `hidden_out` (`[n_tokens, hidden_dim]`). GPU port of
    /// `io::embedding::embed_lookup` — that function is the CPU oracle.
    ///
    /// `token_ids` is an `[n_tokens]` `i32` buffer. `weight` carries
    /// the offsets of `model.embed_tokens.{weight,scales,biases}`; the
    /// Metal arm indexes the shared weight buffer by those offsets,
    /// the CPU arm resolves the same bytes via `WeightFile::bytes_at`.
    EmbedGatherNTokens {
        label: &'static str,
        token_ids: BufId<TokenIdsBuf>,
        weight: WeightRef,
        hidden_out: BufId<EmbedOutBuf>,
        hidden_dim: u32,
        n_tokens: u32,
    },
}

impl Op {
    /// Producer-supplied label for inspection / debug.
    pub fn label(&self) -> &'static str {
        match self {
            Op::RmsNormBf16NTokens { label, .. } => label,
            Op::RmsNormQkNTokens { label, .. } => label,
            Op::RopeNTokens { label, .. } => label,
            Op::ResidualAddNTokens { label, .. } => label,
            Op::ZeroBuffer { label, .. } => label,
            Op::MatvecNTokens { label, .. } => label,
            Op::SwigluFusedBatched { label, .. } => label,
            Op::SdpaCausalTiled { label, .. } => label,
            Op::SigmoidGateNTokens { label, .. } => label,
            Op::SplitQGate { label, .. } => label,
            Op::RmsNormPerHeadNTokens { label, .. } => label,
            Op::KvCacheAppendNTokens { label, .. } => label,
            Op::MoeSoftmaxTopK { label, .. } => label,
            Op::MoeNormalizeWeights { label, .. } => label,
            Op::MoeBatchedPermuteFuse { label, .. } => label,
            Op::MoeGatherIdFuse { label, .. } => label,
            Op::MoeCombineResidualNTokens { label, .. } => label,
            Op::Conv1dStepNTokens { label, .. } => label,
            Op::ComputeDecayBetaNTokens { label, .. } => label,
            Op::GatedDeltaNetStepNTokens { label, .. } => label,
            Op::GatedDeltaNetChunkwise { label, .. } => label,
            Op::GatedRmsNormNTokens { label, .. } => label,
            Op::EmbedGatherNTokens { label, .. } => label,
        }
    }

    /// The variant name as a `&'static str`. For `Graph::dump`
    /// and any future IR text format.
    pub fn variant_name(&self) -> &'static str {
        match self {
            Op::RmsNormBf16NTokens { .. } => "RmsNormBf16NTokens",
            Op::RmsNormQkNTokens { .. } => "RmsNormQkNTokens",
            Op::RopeNTokens { .. } => "RopeNTokens",
            Op::ResidualAddNTokens { .. } => "ResidualAddNTokens",
            Op::ZeroBuffer { .. } => "ZeroBuffer",
            Op::MatvecNTokens { .. } => "MatvecNTokens",
            Op::SwigluFusedBatched { .. } => "SwigluFusedBatched",
            Op::SdpaCausalTiled { .. } => "SdpaCausalTiled",
            Op::SigmoidGateNTokens { .. } => "SigmoidGateNTokens",
            Op::SplitQGate { .. } => "SplitQGate",
            Op::RmsNormPerHeadNTokens { .. } => "RmsNormPerHeadNTokens",
            Op::KvCacheAppendNTokens { .. } => "KvCacheAppendNTokens",
            Op::MoeSoftmaxTopK { .. } => "MoeSoftmaxTopK",
            Op::MoeNormalizeWeights { .. } => "MoeNormalizeWeights",
            Op::MoeBatchedPermuteFuse { .. } => "MoeBatchedPermuteFuse",
            Op::MoeGatherIdFuse { .. } => "MoeGatherIdFuse",
            Op::MoeCombineResidualNTokens { .. } => "MoeCombineResidualNTokens",
            Op::Conv1dStepNTokens { .. } => "Conv1dStepNTokens",
            Op::ComputeDecayBetaNTokens { .. } => "ComputeDecayBetaNTokens",
            Op::GatedDeltaNetStepNTokens { .. } => "GatedDeltaNetStepNTokens",
            Op::GatedDeltaNetChunkwise { .. } => "GatedDeltaNetChunkwise",
            Op::GatedRmsNormNTokens { .. } => "GatedRmsNormNTokens",
            Op::EmbedGatherNTokens { .. } => "EmbedGatherNTokens",
        }
    }

    /// Raw `u32` indices this op *reads from* — tag-agnostic, used by
    /// [`lifetime::analyze_lifetimes`]. Includes RMW buffers (which
    /// also appear in [`Self::writes_raw`]).
    ///
    /// The lifetime / coloring pass operates on indices only — it
    /// doesn't care about role tags — so this helper unwraps every
    /// `BufId<B>` to its inner `u32`. Producers don't call this
    /// directly.
    pub fn reads_raw(&self) -> Vec<u32> {
        match self {
            Op::RmsNormBf16NTokens { x, .. } => vec![x.raw()],
            Op::RmsNormQkNTokens { x, .. } => vec![x.raw()],
            Op::RopeNTokens { x, inv_freq, .. } => {
                vec![x.raw(), inv_freq.raw()]
            }
            Op::ResidualAddNTokens { a, b, .. } => vec![a.raw(), b.raw()],
            Op::ZeroBuffer { .. } => vec![],
            Op::MatvecNTokens { input, .. } => vec![input.raw()],
            Op::SwigluFusedBatched { gate, up, .. } => {
                vec![gate.raw(), up.raw()]
            }
            Op::SdpaCausalTiled { q, k, v, .. } => {
                vec![q.raw(), k.raw(), v.raw()]
            }
            Op::SigmoidGateNTokens { x, gate, .. } => {
                vec![x.raw(), gate.raw()]
            }
            Op::SplitQGate { q_proj, .. } => vec![q_proj.raw()],
            Op::RmsNormPerHeadNTokens { x, .. } => vec![x.raw()],
            Op::KvCacheAppendNTokens { k_src, v_src, .. } => {
                vec![k_src.raw(), v_src.raw()]
            }
            Op::MoeSoftmaxTopK { logits, .. } => vec![logits.raw()],
            Op::MoeNormalizeWeights { weights, .. } => vec![weights.raw()],
            Op::MoeBatchedPermuteFuse {
                expert_base,
                expert_indices,
                bucket_input,
                bucket_token_idx,
                bucket_weights,
                ..
            } => vec![
                expert_base.raw(),
                expert_indices.raw(),
                bucket_input.raw(),
                bucket_token_idx.raw(),
                bucket_weights.raw(),
            ],
            Op::MoeGatherIdFuse {
                expert_base,
                indices,
                weights,
                mlp_in,
                ..
            } => vec![
                expert_base.raw(),
                indices.raw(),
                weights.raw(),
                mlp_in.raw(),
            ],
            Op::MoeCombineResidualNTokens {
                h_mid,
                moe_sum,
                shared_out,
                shared_gate,
                ..
            } => vec![
                h_mid.raw(),
                moe_sum.raw(),
                shared_out.raw(),
                shared_gate.raw(),
            ],
            Op::Conv1dStepNTokens {
                qkv_in, conv_state, ..
            } => vec![qkv_in.raw(), conv_state.raw()],
            Op::ComputeDecayBetaNTokens {
                alpha_in, beta_in, ..
            } => vec![alpha_in.raw(), beta_in.raw()],
            Op::GatedDeltaNetStepNTokens {
                state,
                conv_out,
                g_decay,
                beta_gate,
                ..
            } => vec![state.raw(), conv_out.raw(), g_decay.raw(), beta_gate.raw()],
            Op::GatedDeltaNetChunkwise {
                state,
                conv_out,
                g_decay,
                beta_gate,
                ..
            } => vec![state.raw(), conv_out.raw(), g_decay.raw(), beta_gate.raw()],
            Op::GatedRmsNormNTokens { values, z, .. } => {
                vec![values.raw(), z.raw()]
            }
            Op::EmbedGatherNTokens { token_ids, .. } => vec![token_ids.raw()],
        }
    }

    /// Raw `u32` indices this op *writes to* — tag-agnostic, used by
    /// [`lifetime::analyze_lifetimes`]. RMW buffers also appear in
    /// [`Self::reads_raw`].
    pub fn writes_raw(&self) -> Vec<u32> {
        match self {
            Op::RmsNormBf16NTokens { out, .. } => vec![out.raw()],
            Op::RmsNormQkNTokens { x, .. } => vec![x.raw()], // in-place
            Op::RopeNTokens { x, .. } => vec![x.raw()],      // in-place
            Op::ResidualAddNTokens { out, .. } => vec![out.raw()],
            Op::ZeroBuffer { buf, .. } => vec![buf.raw()],
            Op::MatvecNTokens { output, .. } => vec![output.raw()],
            Op::SwigluFusedBatched { out, .. } => vec![out.raw()],
            Op::SdpaCausalTiled { attn_out, .. } => vec![attn_out.raw()],
            Op::SigmoidGateNTokens { x, .. } => vec![x.raw()], // in-place
            Op::SplitQGate {
                q_out, gate_out, ..
            } => vec![q_out.raw(), gate_out.raw()],
            Op::RmsNormPerHeadNTokens { x, .. } => vec![x.raw()], // in-place
            Op::KvCacheAppendNTokens {
                k_cache, v_cache, ..
            } => vec![k_cache.raw(), v_cache.raw()],
            Op::MoeSoftmaxTopK {
                indices_out,
                weights_out,
                ..
            } => vec![indices_out.raw(), weights_out.raw()],
            Op::MoeNormalizeWeights { weights, .. } => {
                vec![weights.raw()] // in-place
            }
            Op::MoeBatchedPermuteFuse {
                bucket_gate,
                bucket_up,
                bucket_act,
                bucket_out,
                out_sum,
                ..
            } => vec![
                bucket_gate.raw(),
                bucket_up.raw(),
                bucket_act.raw(),
                bucket_out.raw(),
                out_sum.raw(),
            ],
            Op::MoeGatherIdFuse {
                htpe,
                hids,
                gate_mid,
                up_mid,
                down_mid,
                out_sum,
                ..
            } => vec![
                htpe.raw(),
                hids.raw(),
                gate_mid.raw(),
                up_mid.raw(),
                down_mid.raw(),
                out_sum.raw(),
            ],
            Op::MoeCombineResidualNTokens { hidden_out, .. } => {
                vec![hidden_out.raw()]
            }
            Op::Conv1dStepNTokens {
                conv_state,
                conv_out,
                ..
            } => vec![conv_state.raw(), conv_out.raw()], // conv_state is RMW
            Op::ComputeDecayBetaNTokens {
                g_decay_out,
                beta_gate_out,
                ..
            } => vec![g_decay_out.raw(), beta_gate_out.raw()],
            Op::GatedDeltaNetStepNTokens { state, output, .. } => {
                vec![state.raw(), output.raw()] // state is RMW
            }
            Op::GatedDeltaNetChunkwise { state, output, .. } => {
                vec![state.raw(), output.raw()] // state is RMW
            }
            Op::GatedRmsNormNTokens { output, .. } => vec![output.raw()],
            Op::EmbedGatherNTokens { hidden_out, .. } => vec![hidden_out.raw()],
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

    /// Mint a typed `BufId<B>` from a raw index, for fixtures. Tests
    /// don't care about which physical pool the id came from; they
    /// just need stable index values across `one_of_each`.
    fn buf<B: Buf>(n: u32) -> BufId<B> {
        BufId::from_raw(n)
    }

    /// One of every variant, with minimal-ish stub fields.
    fn one_of_each() -> Graph {
        let mut g = Graph::new();
        g.push(Op::RmsNormBf16NTokens {
            label: "rms_in",
            x: buf::<EmbedOutBuf>(0).into(),
            weight_off: 0,
            out: buf::<AttnInputBuf>(1).into(),
            dim: 4096,
            n_tokens: 8,
            eps: 1e-6,
        });
        g.push(Op::RmsNormQkNTokens {
            label: "qk_norm",
            x: buf::<ConvOutBuf>(2),
            num_k_heads: 4,
            key_dim: 128,
            key_offset_per_token: 512,
            per_token_total: 1024,
            n_tokens: 8,
        });
        g.push(Op::ResidualAddNTokens {
            label: "resid",
            a: buf::<OProjOutBuf>(3),
            b: buf::<HiddenBuf>(4).into(),
            out: buf::<ResidualBuf>(5),
            n_tokens: 8,
            dim: 4096,
        });
        g.push(Op::MatvecNTokens {
            label: "q_proj",
            weight: WeightRef {
                w_off: 0,
                s_off: 0,
                b_off: 0,
                bits: 4,
            },
            input: buf::<AttnInputBuf>(6).into(),
            input_off: 0,
            output: buf::<QProjOutBuf>(7).into(),
            output_off: 0,
            in_dim: 4096,
            out_dim: 4096,
            n_tokens: 8,
        });
        g.push(Op::SwigluFusedBatched {
            label: "ffn_swiglu",
            gate: buf::<SharedFfnGateBuf>(8),
            up: buf::<SharedFfnUpBuf>(9),
            out: buf::<SharedFfnActBuf>(10),
            total: 8 * 1024,
        });
        g.push(Op::SdpaCausalTiled {
            label: "sdpa",
            q: buf::<QBuf>(11),
            k: buf::<KvCacheKBuf>(12),
            v: buf::<KvCacheVBuf>(13),
            attn_out: buf::<AttnOutBuf>(14),
            n_tokens: 8,
            num_heads: 16,
            heads_per_kv: 2,
            head_dim: 128,
            kv_dim: 1024,
            kv_start: 0,
            kv_len_total: 8,
            softmax_scale: 0.088_388_35,
        });
        g.push(Op::SigmoidGateNTokens {
            label: "sigmoid_gate",
            x: buf::<AttnOutBuf>(15),
            gate: buf::<QGateBuf>(16),
            dim: 1024,
            n_tokens: 8,
        });
        g.push(Op::SplitQGate {
            label: "split_q_gate",
            q_proj: buf::<QProjOutBuf>(40),
            q_out: buf::<QBuf>(41),
            gate_out: buf::<QGateBuf>(42),
            num_heads: 16,
            head_dim: 128,
            n_tokens: 8,
        });
        g.push(Op::RmsNormPerHeadNTokens {
            label: "rms_norm_per_head",
            x: buf::<QBuf>(43).into(),
            weight_off: 0,
            num_heads: 16,
            head_dim: 128,
            n_tokens: 8,
            eps: 1e-6,
        });
        g.push(Op::KvCacheAppendNTokens {
            label: "kv_cache_append",
            k_src: buf::<KProjOutBuf>(44),
            v_src: buf::<VProjOutBuf>(45),
            k_cache: buf::<KvCacheKBuf>(46),
            v_cache: buf::<KvCacheVBuf>(47),
            kv_dim: 1024,
            n_tokens: 8,
            kv_start: 0,
        });
        g.push(Op::MoeSoftmaxTopK {
            label: "moe_topk",
            logits: buf::<RouterLogitsBuf>(18),
            indices_out: buf::<RouterIdxBuf>(19),
            weights_out: buf::<RouterWeightsBuf>(20),
            n_tokens: 8,
            n_experts: 128,
            k: 8,
        });
        g.push(Op::MoeNormalizeWeights {
            label: "moe_norm",
            weights: buf::<RouterWeightsBuf>(20),
            n_tokens: 8,
            k: 8,
        });
        g.push(Op::MoeBatchedPermuteFuse {
            label: "moe_pf",
            expert_base: buf::<ExpertBaseBuf>(21),
            expert_stride: 0,
            expert_indices: buf::<ExpertIndicesBuf>(35),
            expert_slots: vec![0],
            bucket_input: buf::<BucketInputBuf>(22),
            bucket_gate: buf::<BucketGateBuf>(23),
            bucket_up: buf::<BucketUpBuf>(24),
            bucket_act: buf::<BucketActBuf>(25),
            bucket_out: buf::<BucketOutBuf>(26),
            bucket_token_idx: buf::<BucketTokenIdxBuf>(27),
            bucket_weights: buf::<BucketWeightsBuf>(28),
            out_sum: buf::<MoeOutSumBuf>(29),
            buckets: ExpertBuckets {
                expert_ids: vec![0],
                offsets: vec![0, 8],
                token_idx: vec![0, 1, 2, 3, 4, 5, 6, 7],
                weights: vec![0.125; 8],
            },
        });
        g.push(Op::MoeCombineResidualNTokens {
            label: "moe_combine",
            h_mid: buf::<ResidualBuf>(30),
            moe_sum: buf::<MoeOutSumBuf>(31),
            shared_out: buf::<SharedFfnDownBuf>(32),
            shared_gate: buf::<SharedGateBuf>(33),
            hidden_out: buf::<HiddenBuf>(34),
            n_tokens: 8,
            dim: 4096,
        });
        g.push(Op::Conv1dStepNTokens {
            label: "conv1d",
            qkv_in: buf::<QkvStackBuf>(35),
            conv_state: buf::<ConvStateBuf>(36),
            weight_off: 0,
            conv_out: buf::<ConvOutBuf>(37),
            conv_dim: 5120,
            n_tokens: 8,
        });
        g.push(Op::ComputeDecayBetaNTokens {
            label: "decay_beta",
            alpha_in: buf::<AlphaStackBuf>(38),
            beta_in: buf::<BetaStackBuf>(39),
            a_log_off: 0,
            dt_bias_off: 0,
            g_decay_out: buf::<GDecayBuf>(40),
            beta_gate_out: buf::<BetaGateBuf>(41),
            num_v_heads: 16,
            n_tokens: 8,
        });
        g.push(Op::GatedDeltaNetStepNTokens {
            label: "delta_net",
            state: buf::<DeltaStateBuf>(42),
            conv_out: buf::<ConvOutBuf>(43),
            g_decay: buf::<GDecayBuf>(44),
            beta_gate: buf::<BetaGateBuf>(45),
            output: buf::<DeltaOutBuf>(46),
            num_v_heads: 16,
            value_dim: 128,
            k_heads_per_v: 2,
            n_tokens: 8,
        });
        g.push(Op::GatedRmsNormNTokens {
            label: "gated_rms",
            values: buf::<DeltaOutBuf>(47),
            z: buf::<ZStackBuf>(48),
            weight_off: 0,
            output: buf::<ValueOutBuf>(49),
            num_v_heads: 16,
            value_dim: 128,
            n_tokens: 8,
            eps: 1e-6,
        });
        g.push(Op::EmbedGatherNTokens {
            label: "embed_gather",
            token_ids: buf::<TokenIdsBuf>(50),
            weight: WeightRef {
                w_off: 0,
                s_off: 0,
                b_off: 0,
                bits: 4,
            },
            hidden_out: buf::<EmbedOutBuf>(51),
            hidden_dim: 2048,
            n_tokens: 8,
        });
        g
    }

    #[test]
    fn push_round_trips() {
        let g = one_of_each();
        assert_eq!(g.len(), 19);
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
                "sigmoid_gate",
                "split_q_gate",
                "rms_norm_per_head",
                "kv_cache_append",
                "moe_topk",
                "moe_norm",
                "moe_pf",
                "moe_combine",
                "conv1d",
                "decay_beta",
                "delta_net",
                "gated_rms",
                "embed_gather",
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
        assert!(pairs.contains(&("EmbedGatherNTokens", "embed_gather")));
        assert_eq!(pairs.len(), 19);
    }

    #[test]
    fn reads_and_writes_are_non_empty_for_every_variant() {
        let g = one_of_each();
        for op in &g.ops {
            assert!(
                !op.reads_raw().is_empty(),
                "{} produced empty reads_raw()",
                op.variant_name()
            );
            assert!(
                !op.writes_raw().is_empty(),
                "{} produced empty writes_raw()",
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
                    x: buf::<ConvOutBuf>(2),
                    num_k_heads: 4,
                    key_dim: 128,
                    key_offset_per_token: 512,
                    per_token_total: 1024,
                    n_tokens: 8,
                },
                Op::MoeNormalizeWeights {
                    label: "moe_norm",
                    weights: buf::<RouterWeightsBuf>(20),
                    n_tokens: 8,
                    k: 8,
                },
                Op::Conv1dStepNTokens {
                    label: "conv1d",
                    qkv_in: buf::<QkvStackBuf>(35),
                    conv_state: buf::<ConvStateBuf>(36),
                    weight_off: 0,
                    conv_out: buf::<ConvOutBuf>(37),
                    conv_dim: 5120,
                    n_tokens: 8,
                },
            ],
        };
        // RmsNormQkNTokens: x is read and written
        assert!(g.ops[0].reads_raw().contains(&2));
        assert!(g.ops[0].writes_raw().contains(&2));
        // MoeNormalizeWeights: weights is read and written
        assert!(g.ops[1].reads_raw().contains(&20));
        assert!(g.ops[1].writes_raw().contains(&20));
        // Conv1dStepNTokens: conv_state is read and written
        assert!(g.ops[2].reads_raw().contains(&36));
        assert!(g.ops[2].writes_raw().contains(&36));
    }

    #[test]
    fn dump_emits_one_line_per_op() {
        let g = one_of_each();
        let dump = g.dump();
        let line_count = dump.lines().count();
        assert_eq!(line_count, 19);
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
            x: buf::<EmbedOutBuf>(0).into(),
            weight_off: 0,
            out: buf::<AttnInputBuf>(1).into(),
            dim: 64,
            n_tokens: 2,
            eps: 1e-6,
        });
        g.push(Op::ResidualAddNTokens {
            label: "resid",
            a: buf::<OProjOutBuf>(1),
            b: buf::<EmbedOutBuf>(0).into(),
            out: buf::<ResidualBuf>(2),
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
        let id: BufId<MoeInputBuf> = BufId::from_raw(42);
        assert_eq!(format!("{id}"), "%42");
    }
}
