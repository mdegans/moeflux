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

use std::cell::{Ref, RefCell, RefMut};
use std::fmt;

use crate::riir::cpu_matvec::{
    dequant_matvec_4bit_cpu, dequant_matvec_8bit_v3_cpu,
};
use crate::riir::cpu_ops::{cpu_sigmoid_scalar, residual_add_n_tokens_cpu};
use crate::riir::embedding::bf16_to_f32;
use crate::riir::linear_attn::{
    compute_decay_beta_cpu, conv1d_step, gated_delta_recurrence,
};
use crate::riir::moe_cpu::moe_permute_fuse_cpu;
use crate::riir::moe_router::ExpertBuckets;
use crate::riir::variants::{GROUP_SIZE, VARIANT};
use crate::riir::weight_file::WeightFile;

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
    fn upload(&mut self, id: BufId, host: &[u8]) -> Result<(), Self::Error>;

    /// Bulk-copy bytes out of the buffer at `id` into `host`. Used
    /// for the routing readback at the two-phase split.
    fn download(&self, id: BufId, host: &mut [u8]) -> Result<(), Self::Error>;

    /// Release all non-persistent allocations. Persistent buffers
    /// keep their `BufId`s; transient ones are eligible to be
    /// recycled or dropped on the next [`Self::alloc`].
    fn reset_transient(&mut self);

    /// Label of the buffer at `id`, for [`Graph::dump`] inspection.
    fn label(&self, id: BufId) -> &'static str;
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
    type Error: std::error::Error + Send + Sync + 'static;

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
    /// - **Metal:** `cmdbuf.commit()` + `wait_until_completed()`.
    /// - **CPU:** no-op (already executed inline during
    ///   [`Self::encode_op`]).
    /// - **CoreML (future):** `executable.run()`.
    fn submit_and_wait(&self, ctx: Self::EncodeCtx) -> Result<(), Self::Error>;

    /// Convenience: full begin → encode → submit cycle.
    fn execute(&self, graph: &Graph) -> Result<(), Self::Error> {
        let mut ctx = self.begin_encoding();
        self.encode_graph(graph, &mut ctx);
        self.submit_and_wait(ctx)
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
    /// region at offset `key_offset_per_token`.
    ///
    /// `n_tokens` triggers an internal per-token loop in
    /// [`Backend::encode_op`] for now — future work may add a
    /// truly batched kernel without changing the Op shape.
    RmsNormQkNTokens {
        label: &'static str,
        x: BufId,
        num_k_heads: u32,
        key_dim: u32,
        key_offset_per_token: u32,
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
    /// embeds the metadata directly in the op. Each non-empty
    /// bucket is mapped to one expert weight blob via
    /// `expert_refs[bucket_idx]`.
    ///
    /// All buffers in this op are bucket-flat: indexed by
    /// `(bucket_offset + slot_within_bucket)`. The combine into
    /// per-token `out_sum` is the kernel's responsibility (or the
    /// CPU oracle's, in [`CpuBackend`]).
    MoeBatchedPermuteFuse {
        label: &'static str,
        /// Per-bucket `(expert blob buf, byte offset)`. Length =
        /// `buckets.expert_ids.len()`.
        expert_refs: Vec<(BufId, u64)>,
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
                expert_refs,
                bucket_input,
                bucket_token_idx,
                bucket_weights,
                ..
            } => {
                let mut v = Vec::with_capacity(expert_refs.len() + 3);
                v.push(*bucket_input);
                v.push(*bucket_token_idx);
                v.push(*bucket_weights);
                for (id, _off) in expert_refs {
                    v.push(*id);
                }
                v
            }
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

// ============================================================================
// CpuBackend — first customer of the Backend trait
// ============================================================================
//
// Wraps the existing CPU oracle helpers in a Backend impl. Encoding
// is inline execution: `encode_op` runs the kernel and writes through
// the pool's RefCell-backed buffers; `submit_and_wait` is a no-op.
//
// Used by S7-4's graph_metal_matches_cpu diff test as the reference
// truth against which MetalBackend's encoded output is compared. Also
// stays in-tree post-S7 as a regression net for future kernel work.

/// Naive CPU buffer pool: one `RefCell<Vec<u8>>` per `BufId`.
/// `reset_transient` truncates back to the persistent prefix
/// (assumes persistent allocations come before transient ones, which
/// matches how producer code uses the pool).
pub struct CpuBufferPool {
    buffers: Vec<RefCell<Vec<u8>>>,
    labels: Vec<&'static str>,
    persistent: Vec<bool>,
}

impl CpuBufferPool {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            labels: Vec::new(),
            persistent: Vec::new(),
        }
    }

    /// Total number of physical buffers (one per non-aliased BufId
    /// for the naive pool). Useful for memory-pressure assertions.
    pub fn physical_buffer_count(&self) -> usize {
        self.buffers.len()
    }
}

impl Default for CpuBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferPool for CpuBufferPool {
    type Handle = RefCell<Vec<u8>>;
    type Error = GraphError;

    fn alloc(
        &mut self,
        bytes: usize,
        label: &'static str,
        persistent: bool,
    ) -> Result<BufId, GraphError> {
        let id = BufId(self.buffers.len() as u32);
        self.buffers.push(RefCell::new(vec![0u8; bytes]));
        self.labels.push(label);
        self.persistent.push(persistent);
        Ok(id)
    }

    fn handle(&self, id: BufId) -> &RefCell<Vec<u8>> {
        &self.buffers[id.0 as usize]
    }

    fn upload(&mut self, id: BufId, host: &[u8]) -> Result<(), GraphError> {
        let idx = id.0 as usize;
        let label = *self.labels.get(idx).ok_or(GraphError::BadBufId(id))?;
        let buf = &self.buffers[idx];
        let mut buf_mut = buf.borrow_mut();
        if buf_mut.len() != host.len() {
            return Err(GraphError::SizeMismatch {
                label,
                expected: buf_mut.len(),
                actual: host.len(),
            });
        }
        buf_mut.copy_from_slice(host);
        Ok(())
    }

    fn download(&self, id: BufId, host: &mut [u8]) -> Result<(), GraphError> {
        let idx = id.0 as usize;
        let label = *self.labels.get(idx).ok_or(GraphError::BadBufId(id))?;
        let buf = self.buffers[idx].borrow();
        if buf.len() != host.len() {
            return Err(GraphError::SizeMismatch {
                label,
                expected: buf.len(),
                actual: host.len(),
            });
        }
        host.copy_from_slice(&buf);
        Ok(())
    }

    fn reset_transient(&mut self) {
        // Keep the longest persistent prefix and drop everything past it.
        // Producer convention: persistent allocations (KV cache, hidden
        // double-buffer, weight file views) are made before transient
        // intermediates.
        let mut keep = 0;
        for (i, &p) in self.persistent.iter().enumerate() {
            if p {
                keep = i + 1;
            }
        }
        self.buffers.truncate(keep);
        self.labels.truncate(keep);
        self.persistent.truncate(keep);
    }

    fn label(&self, id: BufId) -> &'static str {
        self.labels
            .get(id.0 as usize)
            .copied()
            .unwrap_or("<bad-bufid>")
    }
}

/// CPU Backend implementation. Each encode_op variant runs the
/// kernel inline using the [`CpuBufferPool`]'s RefCell-backed
/// buffers; `submit_and_wait` is a no-op (execution already
/// happened).
///
/// Weight resolution: each variant that reads weights uses
/// [`WeightFile::bytes_at`] against the backend's owned mmap.
/// `WeightRef` carries `(w_off, s_off, b_off, bits)`; the encode_op
/// arm computes byte lengths from dims and slices the mmap.
pub struct CpuBackend {
    pool: CpuBufferPool,
    wf: WeightFile,
}

impl CpuBackend {
    pub fn new(wf: WeightFile) -> Self {
        Self {
            pool: CpuBufferPool::new(),
            wf,
        }
    }

    pub fn weight_file(&self) -> &WeightFile {
        &self.wf
    }

    // ------------------------------------------------------------------
    // Slice accessors over pool buffers
    // ------------------------------------------------------------------

    fn read_f32(&self, id: BufId) -> Ref<'_, [f32]> {
        Ref::map(self.pool.handle(id).borrow(), bytes_as_f32_panic)
    }

    fn write_f32(&self, id: BufId) -> RefMut<'_, [f32]> {
        RefMut::map(self.pool.handle(id).borrow_mut(), bytes_as_f32_mut_panic)
    }

    #[allow(dead_code)]
    fn read_i32(&self, id: BufId) -> Ref<'_, [i32]> {
        Ref::map(self.pool.handle(id).borrow(), bytes_as_i32_panic)
    }

    fn write_i32(&self, id: BufId) -> RefMut<'_, [i32]> {
        RefMut::map(self.pool.handle(id).borrow_mut(), bytes_as_i32_mut_panic)
    }

    fn read_bytes(&self, id: BufId) -> Ref<'_, [u8]> {
        Ref::map(self.pool.handle(id).borrow(), |v| v.as_slice())
    }
}

// ----------------------------------------------------------------------------
// Byte → typed-slice cast helpers. Pool stores Vec<u8>; the kernels
// want &[f32] / &[i32]. align_to panics if the head/tail of the
// alignment isn't empty — that signals a misaligned alloc, which is a
// caller bug we want loud.
// ----------------------------------------------------------------------------

fn bytes_as_f32_panic(v: &Vec<u8>) -> &[f32] {
    // SAFETY: align_to is safe by definition; we panic on misalignment.
    let (head, body, tail) = unsafe { v.align_to::<f32>() };
    assert!(
        head.is_empty() && tail.is_empty(),
        "pool buffer not f32-aligned (head={}, tail={})",
        head.len(),
        tail.len()
    );
    body
}

fn bytes_as_f32_mut_panic(v: &mut Vec<u8>) -> &mut [f32] {
    let (head, body, tail) = unsafe { v.align_to_mut::<f32>() };
    assert!(
        head.is_empty() && tail.is_empty(),
        "pool buffer not f32-aligned (head={}, tail={})",
        head.len(),
        tail.len()
    );
    body
}

#[allow(dead_code)]
fn bytes_as_i32_panic(v: &Vec<u8>) -> &[i32] {
    let (head, body, tail) = unsafe { v.align_to::<i32>() };
    assert!(
        head.is_empty() && tail.is_empty(),
        "pool buffer not i32-aligned (head={}, tail={})",
        head.len(),
        tail.len()
    );
    body
}

fn bytes_as_i32_mut_panic(v: &mut Vec<u8>) -> &mut [i32] {
    let (head, body, tail) = unsafe { v.align_to_mut::<i32>() };
    assert!(
        head.is_empty() && tail.is_empty(),
        "pool buffer not i32-aligned (head={}, tail={})",
        head.len(),
        tail.len()
    );
    body
}

fn bytes_as_u16_panic(b: &[u8]) -> &[u16] {
    let (head, body, tail) = unsafe { b.align_to::<u16>() };
    assert!(
        head.is_empty() && tail.is_empty(),
        "byte slice not u16-aligned (head={}, tail={})",
        head.len(),
        tail.len()
    );
    body
}

fn bytes_as_u32_panic(b: &[u8]) -> &[u32] {
    let (head, body, tail) = unsafe { b.align_to::<u32>() };
    assert!(
        head.is_empty() && tail.is_empty(),
        "byte slice not u32-aligned (head={}, tail={})",
        head.len(),
        tail.len()
    );
    body
}

// ----------------------------------------------------------------------------
// Byte-variant CPU primitives for kernels whose existing helpers take
// tensor names (rms_norm, lm_head). Inlined here rather than refactor
// every existing call site.
// ----------------------------------------------------------------------------

fn rms_norm_bf16_n_tokens_cpu(
    weight_bf16: &[u8],
    x: &[f32],
    dim: usize,
    n_tokens: usize,
    eps: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), n_tokens * dim);
    debug_assert_eq!(out.len(), n_tokens * dim);
    debug_assert!(weight_bf16.len() >= dim * 2);
    for t in 0..n_tokens {
        let xt = &x[t * dim..(t + 1) * dim];
        let ot = &mut out[t * dim..(t + 1) * dim];
        let mut sum_sq = 0.0f32;
        for &xi in xt.iter() {
            sum_sq += xi * xi;
        }
        let inv_rms = 1.0f32 / (sum_sq / dim as f32 + eps).sqrt();
        for i in 0..dim {
            let w_bits = u16::from_le_bytes([
                weight_bf16[i * 2],
                weight_bf16[i * 2 + 1],
            ]);
            let w = bf16_to_f32(w_bits);
            ot[i] = xt[i] * inv_rms * w;
        }
    }
}

fn rms_norm_qk_n_tokens_cpu(
    x_inout: &mut [f32],
    num_k_heads: usize,
    key_dim: usize,
    key_offset_per_token: usize,
    n_tokens: usize,
    eps: f32,
) {
    // Apply per-head RMSNorm to the q region [base..base+num_k_heads*key_dim]
    // AND the k region [base+key_offset_per_token..]. Unweighted —
    // matches the Metal kernel which uses an inv_scale = 1/sqrt(key_dim)
    // (the kernel doesn't multiply by a separate `weight` tensor).
    let inv_scale = 1.0f32 / (key_dim as f32).sqrt();
    // q stride per token mirrors what the encoder binds; we assume both
    // q and k regions are within one per-token slot of size
    // `key_offset_per_token + num_k_heads * key_dim`.
    let per_token_stride = key_offset_per_token + num_k_heads * key_dim;
    debug_assert_eq!(x_inout.len(), n_tokens * per_token_stride);
    for t in 0..n_tokens {
        let base = t * per_token_stride;
        // Q region
        for h in 0..num_k_heads {
            let off = base + h * key_dim;
            let row = &mut x_inout[off..off + key_dim];
            normalize_unweighted(row, eps, inv_scale);
        }
        // K region
        for h in 0..num_k_heads {
            let off = base + key_offset_per_token + h * key_dim;
            let row = &mut x_inout[off..off + key_dim];
            normalize_unweighted(row, eps, inv_scale);
        }
    }
}

fn normalize_unweighted(row: &mut [f32], eps: f32, inv_scale: f32) {
    let dim = row.len();
    let mut sum_sq = 0.0f32;
    for &v in row.iter() {
        sum_sq += v * v;
    }
    let inv_rms = 1.0f32 / (sum_sq / dim as f32 + eps).sqrt();
    for v in row.iter_mut() {
        *v = *v * inv_rms * inv_scale;
    }
}

fn gated_rms_norm_n_tokens_cpu(
    values: &[f32],
    z: &[f32],
    weight_bf16: &[u8],
    output: &mut [f32],
    num_v_heads: usize,
    value_dim: usize,
    n_tokens: usize,
    eps: f32,
) {
    let per_token = num_v_heads * value_dim;
    debug_assert_eq!(values.len(), n_tokens * per_token);
    debug_assert_eq!(z.len(), n_tokens * per_token);
    debug_assert_eq!(output.len(), n_tokens * per_token);
    debug_assert!(weight_bf16.len() >= value_dim * 2);
    for t in 0..n_tokens {
        for h in 0..num_v_heads {
            let base = t * per_token + h * value_dim;
            let v = &values[base..base + value_dim];
            let zr = &z[base..base + value_dim];
            let o = &mut output[base..base + value_dim];
            let mut sum_sq = 0.0f32;
            for &vi in v.iter() {
                sum_sq += vi * vi;
            }
            let inv_rms =
                1.0f32 / (sum_sq / value_dim as f32 + eps).sqrt();
            for i in 0..value_dim {
                let normed = v[i] * inv_rms;
                let zval = zr[i];
                let gate = zval / (1.0 + (-zval).exp()); // SiLU
                let w_bits = u16::from_le_bytes([
                    weight_bf16[i * 2],
                    weight_bf16[i * 2 + 1],
                ]);
                let w = bf16_to_f32(w_bits);
                o[i] = normed * gate * w;
            }
        }
    }
}

fn swiglu_fused_cpu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    for i in 0..gate.len() {
        let g = gate[i];
        let silu = g / (1.0 + (-g).exp());
        out[i] = silu * up[i];
    }
}

fn moe_softmax_topk_cpu(
    logits: &[f32],
    indices_out: &mut [i32],
    weights_out: &mut [f32],
    n_tokens: usize,
    n_experts: usize,
    k: usize,
) {
    debug_assert_eq!(logits.len(), n_tokens * n_experts);
    debug_assert_eq!(indices_out.len(), n_tokens * k);
    debug_assert_eq!(weights_out.len(), n_tokens * k);
    for t in 0..n_tokens {
        let lr = &logits[t * n_experts..(t + 1) * n_experts];
        // Softmax (numerically stable).
        let mut maxv = f32::NEG_INFINITY;
        for &v in lr.iter() {
            if v > maxv {
                maxv = v;
            }
        }
        let mut sum = 0.0f32;
        let mut probs = vec![0.0f32; n_experts];
        for (i, &v) in lr.iter().enumerate() {
            let p = (v - maxv).exp();
            probs[i] = p;
            sum += p;
        }
        let inv_sum = 1.0f32 / sum;
        for p in probs.iter_mut() {
            *p *= inv_sum;
        }
        // Top-K via running-minimum selection sort: matches the
        // Metal `moe_softmax_topk` kernel's slot order exactly so
        // diff is bit-exact-per-slot (no set sort needed).
        let ir = &mut indices_out[t * k..(t + 1) * k];
        let wr = &mut weights_out[t * k..(t + 1) * k];
        for slot in 0..k {
            ir[slot] = -1;
            wr[slot] = f32::NEG_INFINITY;
        }
        for (e, &p) in probs.iter().enumerate() {
            // Find the slot with the running minimum.
            let mut min_slot = 0;
            let mut min_val = wr[0];
            for s in 1..k {
                if wr[s] < min_val {
                    min_val = wr[s];
                    min_slot = s;
                }
            }
            if p > min_val {
                ir[min_slot] = e as i32;
                wr[min_slot] = p;
            }
        }
    }
}

fn moe_normalize_weights_cpu(weights: &mut [f32], n_tokens: usize, k: usize) {
    debug_assert_eq!(weights.len(), n_tokens * k);
    for t in 0..n_tokens {
        let wr = &mut weights[t * k..(t + 1) * k];
        let sum: f32 = wr.iter().sum();
        if sum > 0.0 {
            let inv = 1.0f32 / sum;
            for w in wr.iter_mut() {
                *w *= inv;
            }
        }
    }
}

fn moe_combine_residual_n_tokens_cpu(
    h_mid: &[f32],
    moe_sum: &[f32],
    shared_out: &[f32],
    shared_gate: &[f32],
    hidden_out: &mut [f32],
    n_tokens: usize,
    dim: usize,
) {
    debug_assert_eq!(h_mid.len(), n_tokens * dim);
    debug_assert_eq!(moe_sum.len(), n_tokens * dim);
    debug_assert_eq!(shared_out.len(), n_tokens * dim);
    debug_assert_eq!(shared_gate.len(), n_tokens);
    debug_assert_eq!(hidden_out.len(), n_tokens * dim);
    for t in 0..n_tokens {
        let g = cpu_sigmoid_scalar(shared_gate[t]);
        for i in 0..dim {
            let idx = t * dim + i;
            hidden_out[idx] = h_mid[idx] + moe_sum[idx] + g * shared_out[idx];
        }
    }
}

fn sdpa_causal_tiled_n_tokens_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    attn_out: &mut [f32],
    n_tokens: usize,
    num_heads: usize,
    heads_per_kv: usize,
    head_dim: usize,
    kv_start: usize,
    kv_len_total: usize,
    softmax_scale: f32,
) {
    // Tile the GPU kernel's running-max/denom/partial pattern with
    // a single-pass per-token compute. Causal mask: token t attends
    // to positions [0..kv_start + t + 1).
    let q_stride = num_heads * head_dim;
    let kv_dim = (num_heads / heads_per_kv) * head_dim;
    debug_assert_eq!(q.len(), n_tokens * q_stride);
    debug_assert_eq!(k.len(), kv_len_total * kv_dim);
    debug_assert_eq!(v.len(), kv_len_total * kv_dim);
    debug_assert_eq!(attn_out.len(), n_tokens * q_stride);
    for t in 0..n_tokens {
        let kv_len_t = kv_start + t + 1;
        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_off = t * q_stride + h * head_dim;
            let q_h = &q[q_off..q_off + head_dim];
            let mut scores = vec![0.0f32; kv_len_t];
            let mut max_score = f32::NEG_INFINITY;
            for pos in 0..kv_len_t {
                let k_off = pos * kv_dim + kv_head * head_dim;
                let mut dot = 0.0f32;
                for i in 0..head_dim {
                    dot += q_h[i] * k[k_off + i];
                }
                scores[pos] = dot * softmax_scale;
                if scores[pos] > max_score {
                    max_score = scores[pos];
                }
            }
            let mut sum_exp = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0f32 / sum_exp;
            for s in scores.iter_mut() {
                *s *= inv_sum;
            }
            let o_off = t * q_stride + h * head_dim;
            for i in 0..head_dim {
                let mut acc = 0.0f32;
                for pos in 0..kv_len_t {
                    let v_off = pos * kv_dim + kv_head * head_dim;
                    acc += scores[pos] * v[v_off + i];
                }
                attn_out[o_off + i] = acc;
            }
        }
    }
}

fn lm_head_cpu(
    hidden: &[f32],
    last_token_row: usize,
    hidden_dim: usize,
    final_norm_weight: &[u8],
    lm_head_weight: &[u8],
    vocab_size: usize,
    logits_out: &mut [f32],
) {
    debug_assert_eq!(logits_out.len(), vocab_size);
    // Final RMSNorm on the last_token_row of hidden.
    let hr =
        &hidden[last_token_row * hidden_dim..(last_token_row + 1) * hidden_dim];
    let mut normed = vec![0.0f32; hidden_dim];
    let mut sum_sq = 0.0f32;
    for &v in hr.iter() {
        sum_sq += v * v;
    }
    let inv_rms = 1.0f32 / (sum_sq / hidden_dim as f32 + 1e-6).sqrt();
    for i in 0..hidden_dim {
        let w_bits = u16::from_le_bytes([
            final_norm_weight[i * 2],
            final_norm_weight[i * 2 + 1],
        ]);
        let w = bf16_to_f32(w_bits);
        normed[i] = hr[i] * inv_rms * w;
    }
    // lm_head matvec: BF16 weight, shape [vocab_size, hidden_dim].
    let lm_w = bytes_as_u16_panic(lm_head_weight);
    for r in 0..vocab_size {
        let mut acc = 0.0f32;
        for c in 0..hidden_dim {
            let w = bf16_to_f32(lm_w[r * hidden_dim + c]);
            acc = w.mul_add(normed[c], acc);
        }
        logits_out[r] = acc;
    }
}

// ----------------------------------------------------------------------------
// CpuBackend::Backend impl
// ----------------------------------------------------------------------------

impl Backend for CpuBackend {
    type Pool = CpuBufferPool;
    type EncodeCtx = ();
    type Error = GraphError;

    fn pool(&self) -> &CpuBufferPool {
        &self.pool
    }
    fn pool_mut(&mut self) -> &mut CpuBufferPool {
        &mut self.pool
    }
    fn begin_encoding(&self) {}
    fn submit_and_wait(&self, _: ()) -> Result<(), GraphError> {
        Ok(())
    }

    fn encode_op(&self, op: &Op, _ctx: &mut ()) {
        match op {
            Op::RmsNormBf16NTokens {
                x,
                weight_off,
                out,
                dim,
                n_tokens,
                eps,
                ..
            } => {
                let dim = *dim as usize;
                let n_tokens = *n_tokens as usize;
                let weight_bytes = self
                    .wf
                    .bytes_at(*weight_off, dim * 2)
                    .expect("weight_off out of mmap");
                let x_buf = self.read_f32(*x);
                let mut out_buf = self.write_f32(*out);
                rms_norm_bf16_n_tokens_cpu(
                    weight_bytes, &x_buf, dim, n_tokens, *eps, &mut out_buf,
                );
            }
            Op::RmsNormQkNTokens {
                x,
                num_k_heads,
                key_dim,
                key_offset_per_token,
                n_tokens,
                ..
            } => {
                let mut x_buf = self.write_f32(*x);
                rms_norm_qk_n_tokens_cpu(
                    &mut x_buf,
                    *num_k_heads as usize,
                    *key_dim as usize,
                    *key_offset_per_token as usize,
                    *n_tokens as usize,
                    1e-6,
                );
            }
            Op::ResidualAddNTokens { a, b, out, .. } => {
                let a_buf = self.read_f32(*a);
                let b_buf = self.read_f32(*b);
                let mut out_buf = self.write_f32(*out);
                residual_add_n_tokens_cpu(&a_buf, &b_buf, &mut out_buf);
            }
            Op::MatvecNTokens {
                weight,
                input,
                input_off,
                output,
                output_off,
                in_dim,
                out_dim,
                n_tokens,
                ..
            } => {
                let in_dim = *in_dim as usize;
                let out_dim = *out_dim as usize;
                let n_tokens = *n_tokens as usize;
                let bits = weight.bits;
                let input_buf = self.read_f32(*input);
                let mut output_buf = self.write_f32(*output);
                let in_skip = (*input_off as usize) / 4;
                let out_skip = (*output_off as usize) / 4;
                let in_packed_words = in_dim * out_dim / (if bits == 4 { 8 } else { 4 });
                let in_scales = out_dim * (in_dim / GROUP_SIZE);
                let w_bytes = self
                    .wf
                    .bytes_at(weight.w_off, in_packed_words * 4)
                    .expect("weight.w_off out of mmap");
                let s_bytes = self
                    .wf
                    .bytes_at(weight.s_off, in_scales * 2)
                    .expect("weight.s_off out of mmap");
                let b_bytes = self
                    .wf
                    .bytes_at(weight.b_off, in_scales * 2)
                    .expect("weight.b_off out of mmap");
                let packed = bytes_as_u32_panic(w_bytes);
                let scales = bytes_as_u16_panic(s_bytes);
                let biases = bytes_as_u16_panic(b_bytes);
                for t in 0..n_tokens {
                    let x_t =
                        &input_buf[in_skip + t * in_dim..in_skip + (t + 1) * in_dim];
                    let out_t = &mut output_buf
                        [out_skip + t * out_dim..out_skip + (t + 1) * out_dim];
                    if bits == 4 {
                        dequant_matvec_4bit_cpu(
                            packed, scales, biases, in_dim, out_dim, x_t, out_t,
                        )
                        .expect("4-bit matvec");
                    } else if bits == 8 {
                        dequant_matvec_8bit_v3_cpu(
                            packed, scales, biases, in_dim, out_dim, x_t, out_t,
                        )
                        .expect("8-bit matvec");
                    } else {
                        panic!("unsupported MatvecNTokens bits={bits}");
                    }
                }
            }
            Op::SwigluFusedBatched { gate, up, out, .. } => {
                let g = self.read_f32(*gate);
                let u = self.read_f32(*up);
                let mut o = self.write_f32(*out);
                swiglu_fused_cpu(&g, &u, &mut o);
            }
            Op::SdpaCausalTiled {
                q,
                k,
                v,
                attn_out,
                n_tokens,
                num_heads,
                heads_per_kv,
                head_dim,
                kv_start,
                kv_len_total,
                softmax_scale,
                ..
            } => {
                let q_buf = self.read_f32(*q);
                let k_buf = self.read_f32(*k);
                let v_buf = self.read_f32(*v);
                let mut o_buf = self.write_f32(*attn_out);
                sdpa_causal_tiled_n_tokens_cpu(
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    &mut o_buf,
                    *n_tokens as usize,
                    *num_heads as usize,
                    *heads_per_kv as usize,
                    *head_dim as usize,
                    *kv_start as usize,
                    *kv_len_total as usize,
                    *softmax_scale,
                );
            }
            Op::MoeSoftmaxTopK {
                logits,
                indices_out,
                weights_out,
                n_tokens,
                n_experts,
                k,
                ..
            } => {
                let logits_buf = self.read_f32(*logits);
                let mut idx_buf = self.write_i32(*indices_out);
                let mut w_buf = self.write_f32(*weights_out);
                moe_softmax_topk_cpu(
                    &logits_buf,
                    &mut idx_buf,
                    &mut w_buf,
                    *n_tokens as usize,
                    *n_experts as usize,
                    *k as usize,
                );
            }
            Op::MoeNormalizeWeights {
                weights, n_tokens, k, ..
            } => {
                let mut w_buf = self.write_f32(*weights);
                moe_normalize_weights_cpu(
                    &mut w_buf,
                    *n_tokens as usize,
                    *k as usize,
                );
            }
            Op::MoeBatchedPermuteFuse {
                expert_refs,
                bucket_input,
                buckets,
                out_sum,
                ..
            } => {
                // Each (BufId, offset) in expert_refs identifies a per-bucket
                // expert blob already uploaded into the pool. We hold all
                // borrows simultaneously — the buffer IDs are distinct, so
                // RefCell borrows don't collide.
                let input_buf = self.read_f32(*bucket_input);
                let mut out_buf = self.write_f32(*out_sum);
                let blobs: Vec<Ref<'_, [u8]>> = expert_refs
                    .iter()
                    .map(|(id, _off)| self.read_bytes(*id))
                    .collect();
                let blob_refs: Vec<&[u8]> =
                    blobs.iter().map(|b| &**b).collect();
                moe_permute_fuse_cpu(
                    &VARIANT, &blob_refs, &input_buf, buckets, &mut out_buf,
                )
                .expect("moe permute-fuse");
            }
            Op::MoeCombineResidualNTokens {
                h_mid,
                moe_sum,
                shared_out,
                shared_gate,
                hidden_out,
                n_tokens,
                dim,
                ..
            } => {
                let h_mid_buf = self.read_f32(*h_mid);
                let moe_sum_buf = self.read_f32(*moe_sum);
                let shared_out_buf = self.read_f32(*shared_out);
                let shared_gate_buf = self.read_f32(*shared_gate);
                let mut hidden_out_buf = self.write_f32(*hidden_out);
                moe_combine_residual_n_tokens_cpu(
                    &h_mid_buf,
                    &moe_sum_buf,
                    &shared_out_buf,
                    &shared_gate_buf,
                    &mut hidden_out_buf,
                    *n_tokens as usize,
                    *dim as usize,
                );
            }
            Op::Conv1dStepNTokens {
                qkv_in,
                conv_state,
                weight_off,
                conv_out,
                conv_dim,
                n_tokens,
                ..
            } => {
                let conv_dim = *conv_dim as usize;
                let n_tokens = *n_tokens as usize;
                // Conv1d kernel size is 4 for Qwen3.6-A3B. Hardcoded
                // here because the kernel arity isn't in the Op for
                // historical reasons (Metal kernel pulls from the
                // bf16 weight tensor size). Refine if needed.
                let kernel_size = 4;
                let weight_bytes = self
                    .wf
                    .bytes_at(*weight_off, conv_dim * kernel_size * 2)
                    .expect("conv1d weight_off out of mmap");
                let qkv_in_buf = self.read_f32(*qkv_in);
                let mut conv_state_buf = self.write_f32(*conv_state);
                let mut conv_out_buf = self.write_f32(*conv_out);
                let mut tmp_out = vec![0.0f32; conv_dim];
                for t in 0..n_tokens {
                    let input =
                        &qkv_in_buf[t * conv_dim..(t + 1) * conv_dim];
                    conv1d_step(
                        &conv_state_buf,
                        input,
                        weight_bytes,
                        conv_dim,
                        kernel_size,
                        &mut tmp_out,
                    )
                    .expect("conv1d_step");
                    conv_out_buf[t * conv_dim..(t + 1) * conv_dim]
                        .copy_from_slice(&tmp_out);
                    // Shift state forward by one step: state =
                    // [state[channels..], input]
                    let cs_len = conv_state_buf.len();
                    conv_state_buf.copy_within(conv_dim..cs_len, 0);
                    conv_state_buf[cs_len - conv_dim..].copy_from_slice(input);
                }
            }
            Op::ComputeDecayBetaNTokens {
                alpha_in,
                beta_in,
                a_log_off,
                dt_bias_off,
                g_decay_out,
                beta_gate_out,
                num_v_heads,
                n_tokens,
                ..
            } => {
                let num_v_heads = *num_v_heads as usize;
                let n_tokens = *n_tokens as usize;
                let a_log_bytes = self
                    .wf
                    .bytes_at(*a_log_off, num_v_heads * 4)
                    .expect("a_log_off out of mmap");
                let dt_bias_bytes = self
                    .wf
                    .bytes_at(*dt_bias_off, num_v_heads * 2)
                    .expect("dt_bias_off out of mmap");
                let a_log: &[f32] = bytemuck_f32(a_log_bytes);
                let alpha_buf = self.read_f32(*alpha_in);
                let beta_buf = self.read_f32(*beta_in);
                let mut g_decay_buf = self.write_f32(*g_decay_out);
                let mut beta_gate_buf = self.write_f32(*beta_gate_out);
                for t in 0..n_tokens {
                    let a =
                        &alpha_buf[t * num_v_heads..(t + 1) * num_v_heads];
                    let b = &beta_buf[t * num_v_heads..(t + 1) * num_v_heads];
                    let g = &mut g_decay_buf
                        [t * num_v_heads..(t + 1) * num_v_heads];
                    let bg = &mut beta_gate_buf
                        [t * num_v_heads..(t + 1) * num_v_heads];
                    compute_decay_beta_cpu(a, b, a_log, dt_bias_bytes, g, bg)
                        .expect("compute_decay_beta");
                }
            }
            Op::GatedDeltaNetStepNTokens {
                state,
                conv_out,
                g_decay,
                beta_gate,
                output,
                num_v_heads,
                value_dim,
                k_heads_per_v,
                n_tokens,
                ..
            } => {
                let v_heads = *num_v_heads as usize;
                let value_dim = *value_dim as usize;
                let k_heads_per_v = *k_heads_per_v as usize;
                let k_heads = v_heads / k_heads_per_v;
                let key_dim = crate::riir::variants::Variant::LINEAR_KEY_DIM;
                let n_tokens = *n_tokens as usize;
                let conv_out_buf = self.read_f32(*conv_out);
                let g_decay_buf = self.read_f32(*g_decay);
                let beta_gate_buf = self.read_f32(*beta_gate);
                let mut state_buf = self.write_f32(*state);
                let mut output_buf = self.write_f32(*output);
                let key_total = VARIANT.linear_total_key();
                let value_total = v_heads * value_dim;
                let per_token_conv = key_total * 3; // q, k, v stacked
                for t in 0..n_tokens {
                    let conv_t = &conv_out_buf
                        [t * per_token_conv..(t + 1) * per_token_conv];
                    let q = &conv_t[0..key_total];
                    let k = &conv_t[key_total..2 * key_total];
                    let v = &conv_t[2 * key_total..3 * key_total];
                    // a_log + dt_bias not used in this Op (consumed by
                    // ComputeDecayBeta upstream); gated_delta_recurrence
                    // takes their results.
                    let g = &g_decay_buf
                        [t * v_heads..(t + 1) * v_heads];
                    let bg = &beta_gate_buf
                        [t * v_heads..(t + 1) * v_heads];
                    let mut out_t = vec![0.0f32; value_total];
                    // Build the recurrence inputs: alpha/beta were
                    // already consumed; we need a_log / dt_bias_bf16.
                    // For the CPU oracle, we pass dummy a_log/dt_bias
                    // because gated_delta_recurrence is structured to
                    // include the decay-beta math. Refinement: split
                    // gated_delta_recurrence into pre-decay + post-decay
                    // pieces. For now, this Op assumes ComputeDecayBeta
                    // has already produced g_decay/beta_gate, and the
                    // recurrence consumes those directly.
                    //
                    // TODO(s7-graph-cleanup): produce a recurrence-only
                    // CPU helper that takes (g, bg, q, k, v) and writes
                    // out_values without re-computing the decay step.
                    let a_log_dummy = vec![0.0f32; v_heads];
                    let dt_dummy = vec![0u8; v_heads * 2];
                    gated_delta_recurrence(
                        &a_log_dummy,
                        &dt_dummy,
                        g, // alpha treated as g_decay input
                        bg, // beta treated as beta_gate input
                        q,
                        k,
                        v,
                        v_heads,
                        k_heads,
                        key_dim,
                        value_dim,
                        &mut state_buf,
                        &mut out_t,
                    )
                    .expect("delta net step");
                    output_buf[t * value_total..(t + 1) * value_total]
                        .copy_from_slice(&out_t);
                }
            }
            Op::GatedRmsNormNTokens {
                values,
                z,
                weight_off,
                output,
                num_v_heads,
                value_dim,
                n_tokens,
                eps,
                ..
            } => {
                let value_dim = *value_dim as usize;
                let weight_bytes = self
                    .wf
                    .bytes_at(*weight_off, value_dim * 2)
                    .expect("gated_rms_norm weight_off out of mmap");
                let v_buf = self.read_f32(*values);
                let z_buf = self.read_f32(*z);
                let mut o_buf = self.write_f32(*output);
                gated_rms_norm_n_tokens_cpu(
                    &v_buf,
                    &z_buf,
                    weight_bytes,
                    &mut o_buf,
                    *num_v_heads as usize,
                    value_dim,
                    *n_tokens as usize,
                    *eps,
                );
            }
            Op::LmHead {
                hidden,
                last_token_row,
                logits_out,
                ..
            } => {
                let hidden_dim = VARIANT.hidden_dim;
                let vocab_size = VARIANT.vocab_size;
                let final_norm_bytes = self
                    .wf
                    .tensor_bytes("model.norm.weight")
                    .expect("final norm tensor missing");
                let lm_head_bytes = self
                    .wf
                    .tensor_bytes("lm_head.weight")
                    .expect("lm_head tensor missing");
                let hidden_buf = self.read_f32(*hidden);
                let mut logits_buf = self.write_f32(*logits_out);
                lm_head_cpu(
                    &hidden_buf,
                    *last_token_row as usize,
                    hidden_dim,
                    final_norm_bytes,
                    lm_head_bytes,
                    vocab_size,
                    &mut logits_buf,
                );
            }
        }
    }
}

fn bytemuck_f32(b: &[u8]) -> &[f32] {
    let (head, body, tail) = unsafe { b.align_to::<f32>() };
    assert!(
        head.is_empty() && tail.is_empty(),
        "byte slice not f32-aligned (head={}, tail={})",
        head.len(),
        tail.len()
    );
    body
}

// ============================================================================
// MetalBackend — wires the Backend trait against the existing Metal kernels
// ============================================================================
//
// Wraps the existing `encode_X_into` helpers. The pool stores
// `metal::Buffer` directly; encode_op writes compute dispatches into a
// `MetalEncodeCtx` that owns a `CommandBuffer`; submit_and_wait commits
// the cmdbuf and blocks.
//
// **S7-3 scope:** 8 of 15 Op variants are fully wired (the ones
// exercised by S7-4's load-bearing diff test): RmsNormBf16NTokens,
// ResidualAddNTokens, MatvecNTokens, SwigluFusedBatched, MoeSoftmaxTopK,
// MoeNormalizeWeights, MoeCombineResidualNTokens, LmHead. The
// remaining 7 (RmsNormQkNTokens, SdpaCausalTiled, MoeBatchedPermuteFuse,
// Conv1dStepNTokens, ComputeDecayBetaNTokens, GatedDeltaNetStepNTokens,
// GatedRmsNormNTokens) are stubbed `todo!()` and will wire alongside
// their producers in S7-6/S7-7. This is honest scoping — those Ops
// have complex multi-pipeline composition or per-token internal loops
// that are easier to wire end-to-end with a real producer than via a
// synthetic harness.

use metal::{
    Buffer, CommandBuffer, CommandBufferRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize, NSUInteger,
};

use crate::riir::gpu_attn::BatchedSdpaPipelines;
use crate::riir::gpu_linear_attn::LinearAttnPipelines;
use crate::riir::gpu_matvec::{
    encode_matvec_n_tokens, BfMatvecPipelines, MatvecPipelines,
};
use crate::riir::gpu_moe_router::MoeRouterPipelines;
use crate::riir::gpu_norm::{
    encode_residual_add_n_tokens_into, encode_rms_norm_bf16_fused_n_tokens,
    RmsNormBf16FusedNTokensPipeline, RmsNormBf16Pipelines,
};
use crate::riir::metal::{MetalContext, MetalError};
use crate::riir::mtl_weight_buf::MtlWeightBuf;

/// Naive Metal buffer pool: one `metal::Buffer` per `BufId`.
pub struct MetalBufferPool {
    device: Device,
    buffers: Vec<Buffer>,
    labels: Vec<&'static str>,
    persistent: Vec<bool>,
}

impl MetalBufferPool {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            buffers: Vec::new(),
            labels: Vec::new(),
            persistent: Vec::new(),
        }
    }

    pub fn physical_buffer_count(&self) -> usize {
        self.buffers.len()
    }
}

impl BufferPool for MetalBufferPool {
    type Handle = Buffer;
    type Error = GraphError;

    fn alloc(
        &mut self,
        bytes: usize,
        label: &'static str,
        persistent: bool,
    ) -> Result<BufId, GraphError> {
        let id = BufId(self.buffers.len() as u32);
        let buf = self.device.new_buffer(
            bytes as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        // Zero on alloc so encoders that assume a clean slot don't
        // read stale memory. Matches the CPU pool's vec![0u8; bytes]
        // behaviour.
        unsafe {
            std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
        }
        self.buffers.push(buf);
        self.labels.push(label);
        self.persistent.push(persistent);
        Ok(id)
    }

    fn handle(&self, id: BufId) -> &Buffer {
        &self.buffers[id.0 as usize]
    }

    fn upload(&mut self, id: BufId, host: &[u8]) -> Result<(), GraphError> {
        let idx = id.0 as usize;
        let label = *self.labels.get(idx).ok_or(GraphError::BadBufId(id))?;
        let buf = &self.buffers[idx];
        let len = buf.length() as usize;
        if len != host.len() {
            return Err(GraphError::SizeMismatch {
                label,
                expected: len,
                actual: host.len(),
            });
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                host.as_ptr(),
                buf.contents() as *mut u8,
                len,
            );
        }
        Ok(())
    }

    fn download(
        &self,
        id: BufId,
        host: &mut [u8],
    ) -> Result<(), GraphError> {
        let idx = id.0 as usize;
        let label = *self.labels.get(idx).ok_or(GraphError::BadBufId(id))?;
        let buf = &self.buffers[idx];
        let len = buf.length() as usize;
        if len != host.len() {
            return Err(GraphError::SizeMismatch {
                label,
                expected: len,
                actual: host.len(),
            });
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.contents() as *const u8,
                host.as_mut_ptr(),
                len,
            );
        }
        Ok(())
    }

    fn reset_transient(&mut self) {
        let mut keep = 0;
        for (i, &p) in self.persistent.iter().enumerate() {
            if p {
                keep = i + 1;
            }
        }
        self.buffers.truncate(keep);
        self.labels.truncate(keep);
        self.persistent.truncate(keep);
    }

    fn label(&self, id: BufId) -> &'static str {
        self.labels
            .get(id.0 as usize)
            .copied()
            .unwrap_or("<bad-bufid>")
    }
}

/// Encoding context for [`MetalBackend`]: owns a `CommandBuffer`
/// that `encode_op` appends dispatches to and `submit_and_wait`
/// commits + waits on.
pub struct MetalEncodeCtx {
    cmdbuf: CommandBuffer,
}

/// Metal `Backend` trait impl.
///
/// Composes a renamed [`MetalContext`] (device + library + pipeline
/// cache + stats) and a shared [`MtlWeightBuf`] (mmap'd weight file
/// wrapped as a Metal buffer). Pre-fetches all pipelines we touch at
/// construction time so `encode_op` can stay `&self`-typed and
/// thread-friendly.
pub struct MetalBackend {
    metal: MetalContext,
    wf_buf: MtlWeightBuf,
    pool: MetalBufferPool,
    // Pre-warmed pipeline caches.
    matvec_pipes: MatvecPipelines,
    #[allow(dead_code)]
    bf_matvec_pipes: BfMatvecPipelines,
    rms_n_pipe: RmsNormBf16FusedNTokensPipeline,
    #[allow(dead_code)]
    rms_pipes: RmsNormBf16Pipelines,
    router_pipes: MoeRouterPipelines,
    #[allow(dead_code)]
    sdpa_pipes: BatchedSdpaPipelines,
    #[allow(dead_code)]
    linear_attn_pipes: LinearAttnPipelines,
    residual_add_n_pso: ComputePipelineState,
    swiglu_fused_batched_pso: ComputePipelineState,
    moe_combine_residual_n_pso: ComputePipelineState,
}

impl MetalBackend {
    pub fn new(
        mut metal: MetalContext,
        wf_buf: MtlWeightBuf,
    ) -> Result<Self, MetalError> {
        // Fetch all pipelines we'll need. Each is a cheap NSObject
        // refcount bump after first compilation; subsequent
        // operations reuse the cache.
        let matvec_pipes = MatvecPipelines::fetch(&mut metal)?;
        let bf_matvec_pipes = BfMatvecPipelines::fetch(&mut metal)?;
        let rms_n_pipe = RmsNormBf16FusedNTokensPipeline::fetch(&mut metal)?;
        let rms_pipes = RmsNormBf16Pipelines::fetch(&mut metal)?;
        let router_pipes = MoeRouterPipelines::fetch(&mut metal)?;
        let sdpa_pipes = BatchedSdpaPipelines::fetch(&mut metal)?;
        let linear_attn_pipes = LinearAttnPipelines::fetch(&mut metal)?;
        let residual_add_n_pso =
            metal.pipeline("residual_add_n_tokens")?.clone();
        let swiglu_fused_batched_pso =
            metal.pipeline("swiglu_fused_batched")?.clone();
        let moe_combine_residual_n_pso =
            metal.pipeline("moe_combine_residual_n_tokens")?.clone();

        let device = metal.device().clone();
        Ok(Self {
            metal,
            wf_buf,
            pool: MetalBufferPool::new(device),
            matvec_pipes,
            bf_matvec_pipes,
            rms_n_pipe,
            rms_pipes,
            router_pipes,
            sdpa_pipes,
            linear_attn_pipes,
            residual_add_n_pso,
            swiglu_fused_batched_pso,
            moe_combine_residual_n_pso,
        })
    }

    pub fn metal(&self) -> &MetalContext {
        &self.metal
    }

    pub fn weight_buf(&self) -> &MtlWeightBuf {
        &self.wf_buf
    }
}

impl Backend for MetalBackend {
    type Pool = MetalBufferPool;
    type EncodeCtx = MetalEncodeCtx;
    type Error = GraphError;

    fn pool(&self) -> &MetalBufferPool {
        &self.pool
    }
    fn pool_mut(&mut self) -> &mut MetalBufferPool {
        &mut self.pool
    }

    fn begin_encoding(&self) -> MetalEncodeCtx {
        let cmdbuf = self.metal.queue().new_command_buffer().to_owned();
        MetalEncodeCtx { cmdbuf }
    }

    fn submit_and_wait(
        &self,
        ctx: MetalEncodeCtx,
    ) -> Result<(), GraphError> {
        ctx.cmdbuf.commit();
        ctx.cmdbuf.wait_until_completed();
        Ok(())
    }

    fn encode_op(&self, op: &Op, ctx: &mut MetalEncodeCtx) {
        let cmd: &CommandBufferRef = &ctx.cmdbuf;
        match op {
            Op::RmsNormBf16NTokens {
                x,
                weight_off,
                out,
                dim,
                n_tokens,
                eps,
                ..
            } => {
                encode_rms_norm_bf16_fused_n_tokens(
                    cmd,
                    &self.rms_n_pipe,
                    self.pool.handle(*x),
                    self.wf_buf.buffer(),
                    *weight_off,
                    self.pool.handle(*out),
                    *dim,
                    *n_tokens,
                    *eps,
                );
            }
            Op::ResidualAddNTokens {
                a,
                b,
                out,
                n_tokens,
                dim,
                ..
            } => {
                encode_residual_add_n_tokens_into(
                    cmd,
                    &self.residual_add_n_pso,
                    self.pool.handle(*a),
                    self.pool.handle(*b),
                    self.pool.handle(*out),
                    *n_tokens,
                    *dim,
                );
            }
            Op::MatvecNTokens {
                weight,
                input,
                input_off,
                output,
                output_off,
                in_dim,
                out_dim,
                n_tokens,
                ..
            } => {
                encode_matvec_n_tokens(
                    cmd,
                    &self.matvec_pipes,
                    self.wf_buf.buffer(),
                    weight.w_off,
                    weight.s_off,
                    weight.b_off,
                    self.pool.handle(*input),
                    *input_off,
                    self.pool.handle(*output),
                    *output_off,
                    *in_dim,
                    *out_dim,
                    *n_tokens,
                    weight.bits,
                );
            }
            Op::SwigluFusedBatched {
                gate,
                up,
                out,
                total,
                ..
            } => {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.swiglu_fused_batched_pso);
                enc.set_buffer(0, Some(self.pool.handle(*gate)), 0);
                enc.set_buffer(1, Some(self.pool.handle(*up)), 0);
                enc.set_buffer(2, Some(self.pool.handle(*out)), 0);
                enc.set_bytes(3, 4, (total as *const u32).cast());
                let num_tgs = (*total + 255) / 256;
                enc.dispatch_thread_groups(
                    MTLSize::new(num_tgs as NSUInteger, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                enc.end_encoding();
            }
            Op::MoeSoftmaxTopK {
                logits,
                indices_out,
                weights_out,
                n_tokens,
                n_experts,
                k,
                ..
            } => {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.router_pipes.softmax_topk);
                enc.set_buffer(0, Some(self.pool.handle(*logits)), 0);
                enc.set_buffer(1, Some(self.pool.handle(*indices_out)), 0);
                enc.set_buffer(2, Some(self.pool.handle(*weights_out)), 0);
                enc.set_bytes(3, 4, (n_experts as *const u32).cast());
                enc.set_bytes(4, 4, (k as *const u32).cast());
                enc.dispatch_thread_groups(
                    MTLSize::new(*n_tokens as NSUInteger, 1, 1),
                    MTLSize::new(64, 1, 1),
                );
                enc.end_encoding();
            }
            Op::MoeNormalizeWeights {
                weights,
                n_tokens,
                k,
                ..
            } => {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.router_pipes.normalize);
                enc.set_buffer(0, Some(self.pool.handle(*weights)), 0);
                enc.set_bytes(1, 4, (k as *const u32).cast());
                enc.dispatch_thread_groups(
                    MTLSize::new(*n_tokens as NSUInteger, 1, 1),
                    MTLSize::new(*k as NSUInteger, 1, 1),
                );
                enc.end_encoding();
            }
            Op::MoeCombineResidualNTokens {
                h_mid,
                moe_sum,
                shared_out,
                shared_gate,
                hidden_out,
                n_tokens,
                dim,
                ..
            } => {
                crate::riir::expert_forward::encode_moe_combine_residual_n_tokens(
                    cmd,
                    &self.moe_combine_residual_n_pso,
                    self.pool.handle(*h_mid),
                    self.pool.handle(*moe_sum),
                    self.pool.handle(*shared_out),
                    self.pool.handle(*shared_gate),
                    self.pool.handle(*hidden_out),
                    *n_tokens,
                    *dim,
                );
            }
            Op::LmHead { .. } => {
                todo!(
                    "LmHead encode_op: needs a persistent workspace BufId in Op shape for the intermediate (post-final-norm) hidden bytes; defer to S7-7 producer wire-up where the orchestrator can allocate it"
                );
            }
            Op::RmsNormQkNTokens { .. } => {
                todo!(
                    "RmsNormQkNTokens encode_op: per-token loop with existing encode_rms_norm_qk; defer to S7-6 producer wire-up"
                )
            }
            Op::SdpaCausalTiled { .. } => {
                todo!(
                    "SdpaCausalTiled encode_op: needs kv_dim arg disambiguation; defer to S7-7 producer wire-up"
                )
            }
            Op::MoeBatchedPermuteFuse { .. } => {
                todo!(
                    "MoeBatchedPermuteFuse encode_op: multi-pipeline composition (gather/gate/up/swiglu/down/bucket_accumulate); defer to S7-6 producer wire-up"
                )
            }
            Op::Conv1dStepNTokens { .. } => {
                todo!(
                    "Conv1dStepNTokens encode_op: per-token loop with encode_conv1d_step; defer to S7-6 producer wire-up"
                )
            }
            Op::ComputeDecayBetaNTokens { .. } => {
                todo!(
                    "ComputeDecayBetaNTokens encode_op: per-token loop with encode_compute_decay_beta; defer to S7-6 producer wire-up"
                )
            }
            Op::GatedDeltaNetStepNTokens { .. } => {
                todo!(
                    "GatedDeltaNetStepNTokens encode_op: per-token loop with encode_delta_net_step; defer to S7-6 producer wire-up"
                )
            }
            Op::GatedRmsNormNTokens { .. } => {
                todo!(
                    "GatedRmsNormNTokens encode_op: per-token loop with encode_gated_rms_norm; defer to S7-6 producer wire-up"
                )
            }
        }
    }
}

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
            expert_refs: vec![(buf(21), 0)],
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

    // ---------- CpuBufferPool contract ----------

    #[test]
    fn cpu_pool_alloc_returns_sequential_bufids() {
        let mut p = CpuBufferPool::new();
        let a = p.alloc(64, "a", false).unwrap();
        let b = p.alloc(128, "b", true).unwrap();
        let c = p.alloc(32, "c", false).unwrap();
        assert_eq!(a, BufId(0));
        assert_eq!(b, BufId(1));
        assert_eq!(c, BufId(2));
        assert_eq!(p.physical_buffer_count(), 3);
    }

    #[test]
    fn cpu_pool_upload_download_round_trips() {
        let mut p = CpuBufferPool::new();
        let id = p.alloc(16, "x", false).unwrap();
        let payload = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        p.upload(id, &payload).unwrap();
        let mut out = vec![0u8; 16];
        p.download(id, &mut out).unwrap();
        assert_eq!(out, payload);
    }

    #[test]
    fn cpu_pool_upload_rejects_size_mismatch() {
        let mut p = CpuBufferPool::new();
        let id = p.alloc(16, "x", false).unwrap();
        let too_big = vec![0u8; 17];
        match p.upload(id, &too_big) {
            Err(GraphError::SizeMismatch { label, expected, actual }) => {
                assert_eq!(label, "x");
                assert_eq!(expected, 16);
                assert_eq!(actual, 17);
            }
            _ => panic!("expected SizeMismatch"),
        }
    }

    #[test]
    fn cpu_pool_reset_transient_keeps_persistent_prefix() {
        let mut p = CpuBufferPool::new();
        let _persistent_a = p.alloc(64, "kv_a", true).unwrap();
        let _persistent_b = p.alloc(64, "kv_b", true).unwrap();
        let _transient = p.alloc(32, "intermed", false).unwrap();
        assert_eq!(p.physical_buffer_count(), 3);
        p.reset_transient();
        assert_eq!(p.physical_buffer_count(), 2);
        // The persistent ids are still valid.
        assert_eq!(p.label(BufId(0)), "kv_a");
        assert_eq!(p.label(BufId(1)), "kv_b");
    }

    #[test]
    fn cpu_pool_handle_returns_refcell_with_zeros() {
        let mut p = CpuBufferPool::new();
        let id = p.alloc(12, "z", false).unwrap();
        let handle = p.handle(id);
        let borrowed = handle.borrow();
        assert_eq!(&*borrowed, &[0u8; 12]);
    }
}
