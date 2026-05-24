//! End-to-end linear-attention layer forward — Phase 4c.
//!
//! Composes:
//!
//! 1. **Pre-attn input norm** (CPU `rms_norm`)
//! 2. **4 batched projection matvecs** (qkv → 12288, z → 8192, beta →
//!    64, alpha → 64) — `dequant_matvec_4bit_v3` against `wf_buf` at
//!    offsets
//! 3. **5 linear-attn fused kernels**: conv1d_step → rms_norm_qk →
//!    compute_decay_beta → gated_delta_net_step → gated_rms_norm
//!    (output staged into `buffers.o_proj_stack`)
//! 4. Hand-off to [`post_attention_tail`] which runs:
//!    - **o_proj** (matvec from `linear_total_value` → HIDDEN_DIM)
//!    - **Residual add + post-attn RMSNorm**
//!    - **MoE router** (gate matvec → CPU softmax/topK/normalize)
//!      + shared-expert gate score matvec
//!    - **Shared expert FFN** (gate / up / SwiGLU / down)
//!    - **K-expert MoE dispatch + combine** via the 9b path
//!      (`gpu_batched_experts_forward`)
//!
//! Output: post-combine HIDDEN_DIM hidden state in `buffers.input`.
//!
//! Mirrors `fused_layer_forward`'s `!is_full` GPU production path
//! (infer.m:4253–~5085), minus the `prev_gpu_combined` fast path and
//! the deferred-experts state machine — for the dump-hook diff this
//! runs synchronously: each command buffer is committed and waited
//! before the next.
//!
//! ## Why one big function
//!
//! Per the plan: the compose is a straight sequence of encode-and-go
//! steps. Splitting it across helpers obscures the data flow without
//! reducing complexity. The function is long but linear; comment
//! markers (`// ── step N: …`) make it scannable.

use metal::{
    Buffer, CommandBufferRef, ComputePipelineState, MTLSize, NSUInteger,
};

/// Process-wide cache for `MOEFLUX_MOE_GATHER_ID`. Routes the
/// batched MoE block through `Op::MoeGatherIdFuse` (the one-
/// dispatch `gather_mm_id.metal` kernel ported from llama.cpp)
/// vs `Op::MoeBatchedPermuteFuse` (the older MLX-vendored
/// `affine_gather_qmm_rhs` path).
///
/// **Default: ON.** Set `MOEFLUX_MOE_GATHER_ID=0` / `false` /
/// `off` to force the old path (kept for diff-oracle and A/B work).
///
/// 2026-05-21: flipped to ON-by-default after engine-level A/B on
/// a3b + 15.7k prefill measured **+11% throughput** (333 → 369.8
/// mean tok/s, n=5, peak-to-peak 0.49%) with no coherence
/// regression (jaccard 0.9469 / 1.0 / 0.9873; cosine ≥0.999973
/// across the three completion prompts in the moeflux_coherence
/// harness).
fn moe_gather_id_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("MOEFLUX_MOE_GATHER_ID").as_deref() {
            Ok("0") | Ok("false") | Ok("off") => false,
            _ => true,
        }
    })
}

/// Process-wide cache for `MOEFLUX_SDPA_VB`. Routes the full-attn SDPA
/// dispatch through the vB staging kernel instead of the vA direct-device
/// production kernel. Opt-in only — kept for diff-oracle and A/B work.
///
/// **Default: OFF.** Set `MOEFLUX_SDPA_VB=1` / `true` / `on` to use
/// the staging kernel.
pub(in crate::riir) fn sdpa_vb_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        matches!(
            std::env::var("MOEFLUX_SDPA_VB").as_deref(),
            Ok("1") | Ok("true") | Ok("on")
        )
    })
}

/// Process-wide cache for `MOEFLUX_SDPA_GQA`. Controls GQA fold (fold=2)
/// for even `heads_per_kv`.
///
/// **Default: OFF.** Dirty A/B (2026-05-22) showed GQA fold is ~3%
/// slower than unfolded vA on a3b — serializing heads inside each TG
/// costs more in parallelism than it saves in L2 cache reuse. The
/// direct-device kernel isn't bandwidth-bound at the fold boundary.
///
/// Set `MOEFLUX_SDPA_GQA=1` / `true` / `on` to opt into fold=2.
pub(in crate::riir) fn sdpa_gqa_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        matches!(
            std::env::var("MOEFLUX_SDPA_GQA").as_deref(),
            Ok("1") | Ok("true") | Ok("on")
        )
    })
}

/// Process-wide cache for `MOEFLUX_DELTA_NET_VB`. Routes the batched
/// delta-net dispatch through the sequential-recurrent kernel (vB)
/// instead of the chunkwise-parallel production kernel (vA).
///
/// **Default: ON.** Set `MOEFLUX_DELTA_NET_VB=0` / `false` / `off` to
/// fall back to the chunkwise-parallel kernel (vA).
pub(in crate::riir) fn delta_net_vb_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        !matches!(
            std::env::var("MOEFLUX_DELTA_NET_VB").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        )
    })
}

use crate::riir::backend::buftype::{
    AlphaStackBuf, AttnInputBuf, AttnOutBuf, BetaGateBuf, BetaStackBuf,
    BucketActBuf, BucketGateBuf, BucketInputBuf, BucketOutBuf,
    BucketTokenIdxBuf, BucketUpBuf, BucketWeightsBuf, ConvOutBuf,
    ConvStateBuf, DeltaOutBuf, DeltaStateBuf, ExpertBaseBuf,
    ExpertIndicesBuf, GDecayBuf, HiddenBuf, HidsBuf, HtpeBuf, KProjOutBuf,
    KvCacheKBuf, KvCacheVBuf, LogitsBuf, MoeInputBuf, MoeOutSumBuf,
    OProjOutBuf, QBuf, QGateBuf, QProjOutBuf, QkvStackBuf, ResidualBuf,
    RouterIdxBuf, RouterLogitsBuf, RouterWeightsBuf, SharedFfnActBuf,
    SharedFfnDownBuf, SharedFfnGateBuf, SharedFfnUpBuf, SharedGateBuf,
    TokenIdsBuf, ValueOutBuf, VProjOutBuf, ZStackBuf,
};
use crate::riir::backend::{Backend, BufId, BufferPool, MetalBufferPool};
use crate::riir::moe::deferred::{
    gpu_batched_experts_begin, gpu_batched_experts_begin_mmap,
    DeferredError,
};
use crate::riir::moe::expert_forward::{ChainToNormed, ExpertPayload, MoeBuffers};
use crate::riir::io::expert_io::ExpertFiles;
use crate::riir::attn::gpu_attn::{
    encode_attn_scores_batched_into, encode_attn_softmax_batched_into,
    encode_attn_values_batched_into, encode_sigmoid_gate_into,
    GpuAttnPipelines,
};
use crate::riir::attn::gpu_linear_attn::{
    encode_compute_decay_beta, encode_conv1d_step, encode_delta_net_step,
    encode_gated_rms_norm, encode_rms_norm_qk, LinearAttnPipelines,
};
use crate::riir::backend::gpu::gpu_matvec::{encode_matvec, MatvecPipelines, MatvecSpec};
use crate::riir::backend::gpu::gpu_norm::{encode_rms_norm_bf16_into, RmsNormBf16Pipelines};
use crate::riir::backend::gpu::gpu_ctx::GpuLayerCtx;
use crate::riir::io::layer_weight_cache::LayerWeightCache;
use crate::riir::backend::gpu::metal::{MetalContext, MetalError};
use crate::riir::moe::moe_router::moe_router_cpu;
use crate::riir::io::mtl_weight_buf::MtlWeightBuf;
use crate::riir::snapshot::state::LinearAttnState;
use crate::riir::variants::{Variant, RMS_NORM_EPS, VARIANT};
use crate::riir::io::weight_file::WeightFile;

/// Errors that can surface during a layer forward (linear or full
/// attention). 4d renamed from `LinearAttnForwardError` once
/// [`post_attention_tail`] became shared between the two paths.
#[derive(Debug, thiserror::Error)]
pub enum LayerForwardError {
    #[error("missing tensor for layer {layer}: {tensor}")]
    MissingTensor {
        layer: usize,
        tensor: &'static str,
    },
    #[error("hidden_in must be HIDDEN_DIM={expected} floats, got {actual}")]
    BadHiddenLen { expected: usize, actual: usize },
    #[error("Metal: {0}")]
    Metal(#[from] MetalError),
    #[error("MoE router: {0}")]
    Router(#[from] crate::riir::moe::moe_router::MoeRouterError),
    #[error("expert FFN: {0}")]
    Expert(#[from] crate::riir::moe::expert_forward::ExpertForwardError),
    #[error("expert I/O: {0}")]
    ExpertIo(#[from] crate::riir::io::expert_io::ExpertIoError),
    #[error("RoPE: {0}")]
    Rope(#[from] crate::riir::attn::rope::RopeError),
    #[error("SDPA: {0}")]
    Sdpa(#[from] crate::riir::attn::sdpa::SdpaError),
    #[error("RMSNorm: {0}")]
    RmsNorm(#[from] crate::riir::attn::rms_norm::RmsNormError),
    #[error("deferred experts: {0}")]
    Deferred(#[from] DeferredError),
    #[error("graph: {0}")]
    Graph(#[from] crate::riir::backend::GraphError),
}

/// Backwards-compat alias. `LinearAttnForwardError` was the original
/// name in 4c; 4d generalised it.
pub type LinearAttnForwardError = LayerForwardError;

/// Persistent GPU scratch + recurrence-state buffers needed by the
/// per-layer forward (linear and full attention). Allocated once per
/// [`crate::riir::RsCtx`].
///
/// Renamed from `LinearAttnBuffers` in 4d. Hosts buffers shared across
/// the two attention paths (input/normed/residual/h_mid/output, the
/// 7-slot batch_out, MoE shared FFN scratch) plus path-specific
/// recurrence/cache:
///
/// - Linear-attn: `conv_state`, `delta_state`, `conv_output`,
///   `delta_g_decay`, `delta_beta`, `delta_output`.
/// - Full-attn: `q_proj_out`, `k_out`, `v_out` (the 3 projection
///   outputs read back to host for CPU per-head norm + RoPE + KV
///   append + SDPA).
pub struct LayerForwardBuffers {
    pub input: BufId<HiddenBuf>,
    pub normed: BufId<AttnInputBuf>,
    pub residual: BufId<ResidualBuf>,
    pub h_mid: BufId<ResidualBuf>,
    pub output: BufId<HiddenBuf>,
    // Former `batch_out: [BufId; 7]` — broken into named fields per
    // tag (plan Q7). The original 7 slots were heterogeneous; the
    // array layout was a smell, the named fields make the per-slot
    // role explicit. Slot index ↔ field name mapping:
    //   [0] → q_stack (linear-attn qkv stack)
    //   [1] → z_stack (linear-attn z proj)
    //   [2] → beta_stack
    //   [3] → alpha_stack
    //   [4] → gate_logits (router output, both paths)
    //   [5] → shared_gate (shared-expert gate scalar, both paths)
    //   [6] → o_proj_stack (o_proj input staging; both paths land
    //         attention output here)
    pub q_stack: BufId<QkvStackBuf>,
    pub z_stack: BufId<ZStackBuf>,
    pub beta_stack: BufId<BetaStackBuf>,
    pub alpha_stack: BufId<AlphaStackBuf>,
    pub gate_logits: BufId<RouterLogitsBuf>,
    pub shared_gate: BufId<SharedGateBuf>,
    pub o_proj_stack: BufId<OProjOutBuf>,
    /// Per-linear-layer recurrence state.
    pub conv_state: Vec<BufId<ConvStateBuf>>,
    pub delta_state: Vec<BufId<DeltaStateBuf>>,
    /// Scratch for one layer's linear-attn pipeline (reused across
    /// layers).
    pub conv_output: BufId<ConvOutBuf>,
    pub delta_g_decay: BufId<GDecayBuf>,
    pub delta_beta: BufId<BetaGateBuf>,
    pub delta_output: BufId<DeltaOutBuf>,
    /// 1-float scratch for `rms_norm_sum_sq` intermediate.
    pub sum_sq: BufId<HiddenBuf>,
    /// Shared-expert intermediate (SHARED_INTERMEDIATE floats).
    pub shared_gate_out: BufId<SharedFfnGateBuf>,
    pub shared_up_out: BufId<SharedFfnUpBuf>,
    pub shared_act: BufId<SharedFfnActBuf>,
    pub shared_out: BufId<SharedFfnDownBuf>,
    /// Full-attn projection outputs. `q_proj_out` carries the raw
    /// per-head `(q, gate)` interleave (`num_attn_heads * head_dim *
    /// 2` floats); `k_out` / `v_out` carry the `kv_dim` raw outputs
    /// before per-head norm + RoPE + KV append.
    pub q_proj_out: BufId<QProjOutBuf>,
    pub k_out: BufId<KProjOutBuf>,
    pub v_out: BufId<VProjOutBuf>,

    /// Slice 5d-7b — GPU full-attention buffers.
    ///
    /// Per-full-attn-layer KV mirrors (host KV stays canonical for
    /// `state_save`; these get one-way-synced on append + state_load):
    /// `gpu_kv_k[fa_idx]` / `gpu_kv_v[fa_idx]` are `GPU_KV_SEQ * kv_dim`
    /// floats each. `fa_idx` = `full_attn_layer_idx_for(layer_idx)`.
    /// Mirrors C `g_metal->buf_kv_k[NUM_FULL_ATTN_LAYERS]` allocation
    /// at `infer.m:1255..1260`.
    pub gpu_kv_k: Vec<BufId<KvCacheKBuf>>,
    pub gpu_kv_v: Vec<BufId<KvCacheVBuf>>,
    /// Shared scratch for the GPU SDPA fast path. Reused across layers
    /// because SDPA is layer-sequential per token (matches C). Sizes:
    /// - `gpu_attn_q` / `gpu_attn_out` / `gpu_attn_gate`:
    ///   `num_attn_heads * head_dim` floats each
    /// - `gpu_attn_scores`: `num_attn_heads * GPU_KV_SEQ` floats
    pub gpu_attn_q: BufId<QBuf>,
    pub gpu_attn_scores: BufId<HiddenBuf>,
    pub gpu_attn_out: BufId<AttnOutBuf>,
    pub gpu_attn_gate: BufId<QGateBuf>,
}

/// Backwards-compat alias for the original 4c name.
pub type LinearAttnBuffers = LayerForwardBuffers;

/// Run-lifetime scratch BufIds for the batched linear-attn `graph1`
/// (S10b-2 / prefill-arc Phase 3). Allocated once, sized at
/// `BATCHED_CHUNK_SIZE` token width; every step and every layer reuses
/// these ids. A smaller chunk (or a decode step at `n_tokens = 1`)
/// processes only a prefix — the `*NTokens` Ops stride by the real
/// `n_tokens`.
///
/// Holds *only* the linear-attn-specific intra-`graph1` transients.
/// The graph1→MoE boundary + the MoE block's working set live in the
/// shared [`MoeGraphScratch`]; the cross-layer hidden double-buffer in
/// [`HiddenDoubleBuffer`]. All allocated `persistent = false` — the
/// first producer call `commit_plan`s, which lifetime-colors them down
/// to a small physical set and pins them for the run.
pub struct LinearAttnGraphScratch {
    // Intra-graph1 transients (colorable).
    pub normed: BufId<AttnInputBuf>,
    pub qkv_stack: BufId<QkvStackBuf>,
    pub z_stack: BufId<ZStackBuf>,
    pub beta_stack: BufId<BetaStackBuf>,
    pub alpha_stack: BufId<AlphaStackBuf>,
    pub conv_out_stack: BufId<ConvOutBuf>,
    pub g_decay_stack: BufId<GDecayBuf>,
    pub beta_gate_stack: BufId<BetaGateBuf>,
    pub delta_out_stack: BufId<DeltaOutBuf>,
    pub value_out_stack: BufId<ValueOutBuf>,
    pub o_proj_stack: BufId<OProjOutBuf>,
    pub gate_logits: BufId<RouterLogitsBuf>,
    /// One-time latch: cleared at construction, set by the first
    /// producer call after it `commit_plan`s `graph1`. Gates the
    /// lifetime-coloring pass to run exactly once per run. The MoE
    /// `graph2` has its own latch on [`MoeGraphScratch`].
    pub commit_planned: std::cell::Cell<bool>,
}

impl LinearAttnGraphScratch {
    /// Allocate the `graph1` transients at max chunk width, all
    /// `persistent = false` (the first producer call `commit_plan`s +
    /// pins them).
    pub fn new(pool: &mut MetalBufferPool) -> Self {
        let v = VARIANT;
        let chunk = crate::riir::BATCHED_CHUNK_SIZE;
        let f32_sz = std::mem::size_of::<f32>();
        let hidden = v.hidden_dim;
        let conv = v.linear_conv_dim();
        let total_value = v.linear_total_value();
        let num_v = v.linear_num_v_heads;
        let delta_out = num_v * Variant::LINEAR_VALUE_DIM;
        // Closures over `pool` can only fix one tag-arg per call, so
        // we expand alloc sites inline. `bytes_of` keeps the size
        // computation close to the site for readability.
        let bytes_of = |elems: usize| chunk * elems * f32_sz;
        Self {
            normed: pool
                .alloc(bytes_of(hidden), "lags.normed", false)
                .expect("LinearAttnGraphScratch::new pool alloc"),
            qkv_stack: pool
                .alloc(bytes_of(conv), "lags.qkv_stack", false)
                .expect("LinearAttnGraphScratch::new pool alloc"),
            z_stack: pool
                .alloc(bytes_of(total_value), "lags.z_stack", false)
                .expect("LinearAttnGraphScratch::new pool alloc"),
            beta_stack: pool
                .alloc(bytes_of(num_v), "lags.beta_stack", false)
                .expect("LinearAttnGraphScratch::new pool alloc"),
            alpha_stack: pool
                .alloc(bytes_of(num_v), "lags.alpha_stack", false)
                .expect("LinearAttnGraphScratch::new pool alloc"),
            conv_out_stack: pool
                .alloc(bytes_of(conv), "lags.conv_out_stack", false)
                .expect("LinearAttnGraphScratch::new pool alloc"),
            g_decay_stack: pool
                .alloc(bytes_of(num_v), "lags.g_decay_stack", false)
                .expect("LinearAttnGraphScratch::new pool alloc"),
            beta_gate_stack: pool
                .alloc(bytes_of(num_v), "lags.beta_gate_stack", false)
                .expect("LinearAttnGraphScratch::new pool alloc"),
            delta_out_stack: pool
                .alloc(bytes_of(delta_out), "lags.delta_out_stack", false)
                .expect("LinearAttnGraphScratch::new pool alloc"),
            value_out_stack: pool
                .alloc(bytes_of(total_value), "lags.value_out_stack", false)
                .expect("LinearAttnGraphScratch::new pool alloc"),
            o_proj_stack: pool
                .alloc(bytes_of(hidden), "lags.o_proj_stack", false)
                .expect("LinearAttnGraphScratch::new pool alloc"),
            gate_logits: pool
                .alloc(bytes_of(v.num_experts), "lags.gate_logits", false)
                .expect("LinearAttnGraphScratch::new pool alloc"),
            commit_planned: std::cell::Cell::new(false),
        }
    }
}

/// Cross-layer hidden double-buffer — run-lifetime, orchestrator-owned.
/// Each layer reads `hidden_in` and writes `hidden_out`; the
/// orchestrator swaps the pair between layers. Lifted out of the
/// per-layer scratch (prefill-arc Phase 3) so the attention scratch
/// structs hold only per-layer state, not run-level orchestrator state.
pub struct HiddenDoubleBuffer {
    pub hidden_a: BufId<HiddenBuf>,
    pub hidden_b: BufId<HiddenBuf>,
}

impl HiddenDoubleBuffer {
    /// Allocate the alternating hidden pair at max chunk width, both
    /// `persistent = true` (never aliased — they bridge layers).
    pub fn new(pool: &mut MetalBufferPool) -> Self {
        let v = VARIANT;
        let chunk = crate::riir::BATCHED_CHUNK_SIZE;
        let bytes = chunk * v.hidden_dim * std::mem::size_of::<f32>();
        Self {
            hidden_a: pool
                .alloc(bytes, "hdb.hidden_a", true)
                .expect("HiddenDoubleBuffer::new pool alloc"),
            hidden_b: pool
                .alloc(bytes, "hdb.hidden_b", true)
                .expect("HiddenDoubleBuffer::new pool alloc"),
        }
    }
}

/// Run-lifetime scratch for the orchestrator's head and tail —
/// owned at run scope, both BufIds `persistent = true`, allocated
/// once by [`Self::new`] (prefill-arc Phase 4).
///
/// - `token_ids` — `[BATCHED_CHUNK_SIZE]` i32. The head uploads this
///   step's token ids here for `Op::EmbedGatherNTokens`.
/// - `logits` — `[vocab_size]` f32. The tail's lm-head matvec writes
///   the final logits here, downloaded to the caller's slice.
pub struct HeadTailScratch {
    pub token_ids: BufId<TokenIdsBuf>,
    pub logits: BufId<LogitsBuf>,
}

impl HeadTailScratch {
    /// Allocate the token-id input and logits-output buffers.
    pub fn new(pool: &mut MetalBufferPool) -> Self {
        let v = VARIANT;
        let chunk = crate::riir::BATCHED_CHUNK_SIZE;
        let token_ids: BufId<TokenIdsBuf> = pool
            .alloc(
                chunk * std::mem::size_of::<i32>(),
                "hts.token_ids",
                true,
            )
            .expect("HeadTailScratch::new token_ids alloc");
        let logits: BufId<LogitsBuf> = pool
            .alloc(
                v.vocab_size * std::mem::size_of::<f32>(),
                "hts.logits",
                true,
            )
            .expect("HeadTailScratch::new logits alloc");
        Self { token_ids, logits }
    }
}

/// Run-lifetime scratch for the shared MoE block (`graph2`) — the
/// graph1→MoE boundary buffers plus the shared-FFN + permute-fuse
/// working set. One instance, owned by the orchestrator, reused by
/// every layer's MoE block regardless of attention kind (layers run
/// sequentially — no aliasing hazard). Both attention producers
/// receive it as `&MoeGraphScratch` (prefill-arc Phase 3).
///
/// Two `commit_plan` classes:
/// - **graph1→MoE boundary** (`h_mid` … `routing_weights`): written by
///   the attention `graph1`, read by the MoE block. `persistent = true`
///   — never aliased.
/// - **MoE working set** (`shared_ffn_gate` … `expert_indices`):
///   `shared_ffn_gate` … `out_sum` are colorable transients
///   (`persistent = false`); the host-uploaded inputs (`bucket_input`,
///   `bucket_token_idx`, `bucket_weights`, `expert_indices`) are
///   `persistent = true`.
pub struct MoeGraphScratch {
    // Graph1→MoE boundary (persistent, never aliased).
    pub h_mid: BufId<ResidualBuf>,
    pub h_post: BufId<MoeInputBuf>,
    pub shared_gate: BufId<SharedGateBuf>,
    pub routing_indices: BufId<RouterIdxBuf>,
    pub routing_weights: BufId<RouterWeightsBuf>,
    // Shared FFN intermediates (colorable transients).
    pub shared_ffn_gate: BufId<SharedFfnGateBuf>,
    pub shared_up: BufId<SharedFfnUpBuf>,
    pub shared_act: BufId<SharedFfnActBuf>,
    pub shared_down: BufId<SharedFfnDownBuf>,
    // Bucket-flat scratch. Sized for `BATCHED_CHUNK_SIZE * k_active`
    // assignments.
    pub bucket_input: BufId<BucketInputBuf>,
    pub bucket_gate: BufId<BucketGateBuf>,
    pub bucket_up: BufId<BucketUpBuf>,
    pub bucket_act: BufId<BucketActBuf>,
    pub bucket_out: BufId<BucketOutBuf>,
    pub bucket_token_idx: BufId<BucketTokenIdxBuf>,
    pub bucket_weights: BufId<BucketWeightsBuf>,
    /// MoE permute-fuse scatter accumulator. `Op::ZeroBuffer` clears
    /// it each layer before `Op::MoeBatchedPermuteFuse` accumulates.
    pub out_sum: BufId<MoeOutSumBuf>,
    /// Per-assignment-row expert slot table (`u32`), `chunk *
    /// k_active` wide — the gather kernel's `indices`. Filled per
    /// layer from `buckets`.
    pub expert_indices: BufId<ExpertIndicesBuf>,
    /// **`Op::MoeGatherIdFuse` scratch** — per-expert assignment
    /// count `[n_experts]` u32. Always allocated; only touched by
    /// the new-kernel path. Tiny (~1 KB at n_experts=256).
    pub htpe: BufId<HtpeBuf>,
    /// **`Op::MoeGatherIdFuse` scratch** — per-expert assignment
    /// list `[n_experts, chunk]` i32 (encoded `token*k + slot`).
    /// Always allocated; only touched by the new-kernel path.
    /// `~8 MB` at n_experts=256, chunk=8192. The `bucket_gate` /
    /// `bucket_up` / `bucket_out` slots above are byte-equivalent
    /// to the new path's `gate_mid` / `up_mid` / `down_mid` and
    /// are reused in-place by `Op::MoeGatherIdFuse` via the
    /// unidirectional `From<BucketGateBuf> for BufId<GateMidBuf>`
    /// (etc.) impls in `buftype.rs` — no second allocation; the
    /// `.into()` at the push site documents the path swap.
    pub hids: BufId<HidsBuf>,
    /// One-time latch: cleared at construction, set by the first MoE
    /// block call after it `commit_plan`s `graph2`. `graph2`'s
    /// topology is identical regardless of which attention kind
    /// produced its inputs, so one pass holds for the whole run.
    pub commit_planned: std::cell::Cell<bool>,
}

impl MoeGraphScratch {
    /// Allocate the MoE-block scratch at max chunk width. The shared-
    /// FFN intermediates are `persistent = false` (the first MoE block
    /// call `commit_plan`s + pins them); the boundary buffers +
    /// host-uploaded inputs are `persistent = true`. `k_active` sizes
    /// the bucket-flat buffers.
    pub fn new(
        pool: &mut MetalBufferPool,
        k_active: usize,
    ) -> Self {
        let v = VARIANT;
        let chunk = crate::riir::BATCHED_CHUNK_SIZE;
        let f32_sz = std::mem::size_of::<f32>();
        let hidden = v.hidden_dim;
        let shared_inter = v.shared_intermediate;
        let moe_inter = v.moe_intermediate;
        let chk = |elems: usize| chunk * elems * f32_sz;
        // Bucket-flat buffers hold `chunk * k_active` assignments.
        let bkt = |per: usize| chunk * k_active * per * f32_sz;
        let max_k = crate::riir::moe::expert_forward::MAX_K;
        // Colorable transients first (`persistent = false`).
        let shared_ffn_gate: BufId<SharedFfnGateBuf> = pool
            .alloc(chk(shared_inter), "mgs.shared_ffn_gate", false)
            .expect("MoeGraphScratch::new pool alloc");
        let shared_up: BufId<SharedFfnUpBuf> = pool
            .alloc(chk(shared_inter), "mgs.shared_up", false)
            .expect("MoeGraphScratch::new pool alloc");
        let shared_act: BufId<SharedFfnActBuf> = pool
            .alloc(chk(shared_inter), "mgs.shared_act", false)
            .expect("MoeGraphScratch::new pool alloc");
        let shared_down: BufId<SharedFfnDownBuf> = pool
            .alloc(chk(hidden), "mgs.shared_down", false)
            .expect("MoeGraphScratch::new pool alloc");
        let bucket_gate: BufId<BucketGateBuf> = pool
            .alloc(chk(k_active * moe_inter), "mgs.bucket_gate", false)
            .expect("MoeGraphScratch::new pool alloc");
        let bucket_up: BufId<BucketUpBuf> = pool
            .alloc(chk(k_active * moe_inter), "mgs.bucket_up", false)
            .expect("MoeGraphScratch::new pool alloc");
        let bucket_act: BufId<BucketActBuf> = pool
            .alloc(chk(k_active * moe_inter), "mgs.bucket_act", false)
            .expect("MoeGraphScratch::new pool alloc");
        let bucket_out: BufId<BucketOutBuf> = pool
            .alloc(chk(k_active * hidden), "mgs.bucket_out", false)
            .expect("MoeGraphScratch::new pool alloc");
        let out_sum: BufId<MoeOutSumBuf> = pool
            .alloc(chk(hidden), "mgs.out_sum", false)
            .expect("MoeGraphScratch::new pool alloc");

        // Pinned buffers (`persistent = true`) — graph1→MoE boundary
        // and host-uploaded inputs.
        let h_mid: BufId<ResidualBuf> = pool
            .alloc(chk(hidden), "mgs.h_mid", true)
            .expect("MoeGraphScratch::new pool alloc");
        let h_post: BufId<MoeInputBuf> = pool
            .alloc(chk(hidden), "mgs.h_post", true)
            .expect("MoeGraphScratch::new pool alloc");
        let shared_gate: BufId<SharedGateBuf> = pool
            .alloc(chk(1), "mgs.shared_gate", true)
            .expect("MoeGraphScratch::new pool alloc");
        let routing_indices: BufId<RouterIdxBuf> = pool
            .alloc(chk(max_k), "mgs.routing_indices", true)
            .expect("MoeGraphScratch::new pool alloc");
        let routing_weights: BufId<RouterWeightsBuf> = pool
            .alloc(chk(max_k), "mgs.routing_weights", true)
            .expect("MoeGraphScratch::new pool alloc");
        let bucket_input: BufId<BucketInputBuf> = pool
            .alloc(bkt(hidden), "mgs.bucket_input", true)
            .expect("MoeGraphScratch::new pool alloc");
        let bucket_token_idx: BufId<BucketTokenIdxBuf> = pool
            .alloc(chunk * k_active * f32_sz, "mgs.bucket_token_idx", true)
            .expect("MoeGraphScratch::new pool alloc");
        let bucket_weights: BufId<BucketWeightsBuf> = pool
            .alloc(chunk * k_active * f32_sz, "mgs.bucket_weights", true)
            .expect("MoeGraphScratch::new pool alloc");
        let expert_indices: BufId<ExpertIndicesBuf> = pool
            .alloc(
                chunk * k_active * std::mem::size_of::<u32>(),
                "mgs.expert_indices",
                true,
            )
            .expect("MoeGraphScratch::new pool alloc");
        // `Op::MoeGatherIdFuse` scratch — sized for the max chunk
        // width. Persistent because the new kernel's map0 + matmul
        // dispatches assume these buffers don't get colored under
        // them by `commit_plan`. `htpe` is small (~1 KB); `hids`
        // is `n_experts * chunk * 4` bytes.
        let n_experts = v.num_experts.max(1);
        let htpe: BufId<HtpeBuf> = pool
            .alloc(n_experts * std::mem::size_of::<u32>(), "mgs.htpe", true)
            .expect("MoeGraphScratch::new pool alloc");
        let hids: BufId<HidsBuf> = pool
            .alloc(
                n_experts * chunk * std::mem::size_of::<i32>(),
                "mgs.hids",
                true,
            )
            .expect("MoeGraphScratch::new pool alloc");
        Self {
            h_mid,
            h_post,
            shared_gate,
            routing_indices,
            routing_weights,
            shared_ffn_gate,
            shared_up,
            shared_act,
            shared_down,
            bucket_input,
            bucket_gate,
            bucket_up,
            bucket_act,
            bucket_out,
            bucket_token_idx,
            bucket_weights,
            out_sum,
            expert_indices,
            htpe,
            hids,
            commit_planned: std::cell::Cell::new(false),
        }
    }
}

impl LayerForwardBuffers {
    /// Allocate every per-layer / per-step buffer as a persistent BufId
    /// in the graph-mode pool. `pool.alloc` zero-fills under the hood,
    /// so the recurrence buffers (`conv_state` / `delta_state`) start
    /// at zero — matching the C `metal_setup` calloc semantics.
    pub fn new(pool: &mut MetalBufferPool) -> Self {
        let v = VARIANT;
        let f32_bytes = |n: usize| n * std::mem::size_of::<f32>();
        let q_dim_full = v.num_attn_heads * v.head_dim;
        let q_proj_dim_full = q_dim_full * 2;
        let kv_dim_full = v.num_kv_heads * v.head_dim;
        // The former `batch_out[6]` "o_proj input" slot has to fit
        // either the linear-attn (`linear_total_value`) or full-attn
        // (`q_dim_full`) attention output. Pick max so either fits.
        let oproj_in_max = v.linear_total_value().max(q_dim_full);

        let num_linear = v.num_layers - num_full_attn_layers(&v);
        let conv_state: Vec<BufId<ConvStateBuf>> = (0..num_linear)
            .map(|_| {
                pool.alloc(
                    f32_bytes(
                        (Variant::CONV_KERNEL_SIZE - 1) * v.linear_conv_dim(),
                    ),
                    "lfb.conv_state",
                    true,
                )
                .expect("LayerForwardBuffers::new pool alloc")
            })
            .collect();
        let delta_state: Vec<BufId<DeltaStateBuf>> = (0..num_linear)
            .map(|_| {
                pool.alloc(
                    f32_bytes(
                        v.linear_num_v_heads
                            * Variant::LINEAR_VALUE_DIM
                            * Variant::LINEAR_KEY_DIM,
                    ),
                    "lfb.delta_state",
                    true,
                )
                .expect("LayerForwardBuffers::new pool alloc")
            })
            .collect();

        let num_full_attn = num_full_attn_layers(&v);
        let gpu_kv_floats = crate::riir::variants::GPU_KV_SEQ * kv_dim_full;
        let gpu_kv_k: Vec<BufId<KvCacheKBuf>> = (0..num_full_attn)
            .map(|_| {
                pool.alloc(f32_bytes(gpu_kv_floats), "lfb.gpu_kv_k", true)
                    .expect("LayerForwardBuffers::new pool alloc")
            })
            .collect();
        let gpu_kv_v: Vec<BufId<KvCacheVBuf>> = (0..num_full_attn)
            .map(|_| {
                pool.alloc(f32_bytes(gpu_kv_floats), "lfb.gpu_kv_v", true)
                    .expect("LayerForwardBuffers::new pool alloc")
            })
            .collect();

        Self {
            input: pool
                .alloc(f32_bytes(v.hidden_dim), "lfb.input", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            normed: pool
                .alloc(f32_bytes(v.hidden_dim), "lfb.normed", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            residual: pool
                .alloc(f32_bytes(v.hidden_dim), "lfb.residual", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            h_mid: pool
                .alloc(f32_bytes(v.hidden_dim), "lfb.h_mid", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            output: pool
                .alloc(f32_bytes(v.hidden_dim), "lfb.output", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            // Former batch_out[0..7] expanded to named fields.
            q_stack: pool
                .alloc(
                    f32_bytes(v.linear_conv_dim()),
                    "lfb.batch_out[0:qkv]",
                    true,
                )
                .expect("LayerForwardBuffers::new pool alloc"),
            z_stack: pool
                .alloc(
                    f32_bytes(v.linear_total_value()),
                    "lfb.batch_out[1:z]",
                    true,
                )
                .expect("LayerForwardBuffers::new pool alloc"),
            beta_stack: pool
                .alloc(
                    f32_bytes(v.linear_num_v_heads),
                    "lfb.batch_out[2:beta]",
                    true,
                )
                .expect("LayerForwardBuffers::new pool alloc"),
            alpha_stack: pool
                .alloc(
                    f32_bytes(v.linear_num_v_heads),
                    "lfb.batch_out[3:alpha]",
                    true,
                )
                .expect("LayerForwardBuffers::new pool alloc"),
            gate_logits: pool
                .alloc(
                    f32_bytes(v.num_experts),
                    "lfb.batch_out[4:router_gate]",
                    true,
                )
                .expect("LayerForwardBuffers::new pool alloc"),
            shared_gate: pool
                .alloc(f32_bytes(1), "lfb.batch_out[5:shared_gate]", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            o_proj_stack: pool
                .alloc(
                    f32_bytes(oproj_in_max),
                    "lfb.batch_out[6:oproj_in]",
                    true,
                )
                .expect("LayerForwardBuffers::new pool alloc"),
            conv_state,
            delta_state,
            conv_output: pool
                .alloc(f32_bytes(v.linear_conv_dim()), "lfb.conv_output", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            delta_g_decay: pool
                .alloc(f32_bytes(v.linear_num_v_heads), "lfb.delta_g_decay", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            delta_beta: pool
                .alloc(f32_bytes(v.linear_num_v_heads), "lfb.delta_beta", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            delta_output: pool
                .alloc(
                    f32_bytes(v.linear_total_value()),
                    "lfb.delta_output",
                    true,
                )
                .expect("LayerForwardBuffers::new pool alloc"),
            sum_sq: pool
                .alloc(f32_bytes(1), "lfb.sum_sq", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            shared_gate_out: pool
                .alloc(
                    f32_bytes(v.shared_intermediate),
                    "lfb.shared_gate_out",
                    true,
                )
                .expect("LayerForwardBuffers::new pool alloc"),
            shared_up_out: pool
                .alloc(
                    f32_bytes(v.shared_intermediate),
                    "lfb.shared_up_out",
                    true,
                )
                .expect("LayerForwardBuffers::new pool alloc"),
            shared_act: pool
                .alloc(f32_bytes(v.shared_intermediate), "lfb.shared_act", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            shared_out: pool
                .alloc(f32_bytes(v.hidden_dim), "lfb.shared_out", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            q_proj_out: pool
                .alloc(f32_bytes(q_proj_dim_full), "lfb.q_proj_out", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            k_out: pool
                .alloc(f32_bytes(kv_dim_full), "lfb.k_out", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            v_out: pool
                .alloc(f32_bytes(kv_dim_full), "lfb.v_out", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            gpu_kv_k,
            gpu_kv_v,
            gpu_attn_q: pool
                .alloc(f32_bytes(q_dim_full), "lfb.gpu_attn_q", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            gpu_attn_scores: pool
                .alloc(
                    f32_bytes(
                        v.num_attn_heads * crate::riir::variants::GPU_KV_SEQ,
                    ),
                    "lfb.gpu_attn_scores",
                    true,
                )
                .expect("LayerForwardBuffers::new pool alloc"),
            gpu_attn_out: pool
                .alloc(f32_bytes(q_dim_full), "lfb.gpu_attn_out", true)
                .expect("LayerForwardBuffers::new pool alloc"),
            gpu_attn_gate: pool
                .alloc(f32_bytes(q_dim_full), "lfb.gpu_attn_gate", true)
                .expect("LayerForwardBuffers::new pool alloc"),
        }
    }

    /// Reset every per-layer state buffer to zero. Called from
    /// `RsCtx::memory_clear` to clear the recurrence in addition to
    /// the host-side state vector (which today the GPU doesn't read,
    /// so this is the source of truth on the GPU side).
    pub fn reset_recurrence(&self, pool: &MetalBufferPool) {
        for &id in &self.conv_state {
            zero_f32_buffer(pool.handle(id));
        }
        for &id in &self.delta_state {
            zero_f32_buffer(pool.handle(id));
        }
    }

    /// Slice 5d-7b — zero the GPU full-attn KV mirrors. Called from
    /// `RsCtx::memory_clear` alongside `reset_recurrence`. The host
    /// KV cache is cleared via `clear_all(layer_states)`; this is the
    /// matching reset on the GPU side. Mirrors the C path's reset of
    /// `buf_kv_k` / `buf_kv_v` at `mf_memory_clear`.
    pub fn reset_gpu_attn_kv_mirrors(&self, pool: &MetalBufferPool) {
        for &id in &self.gpu_kv_k {
            zero_f32_buffer(pool.handle(id));
        }
        for &id in &self.gpu_kv_v {
            zero_f32_buffer(pool.handle(id));
        }
    }
}

/// Zero every byte of a shared-storage Metal buffer. Used by
/// `memory_clear` to reset GPU-resident state (linear-attn
/// recurrence, full-attn KV mirrors).
///
/// # Safety
///
/// `memory_clear` is the only caller and must run after all
/// in-flight dispatches have completed. The deferred-ring drain at
/// the top of `memory_clear` enforces this; no other path reaches
/// this function.
fn zero_f32_buffer(b: &Buffer) {
    let bytes = b.length() as usize;
    // SAFETY: see fn docs.
    unsafe {
        std::ptr::write_bytes(b.contents() as *mut u8, 0, bytes);
    }
}

/// `linear_layer_idx = layer_idx - (layer_idx + 1) / FULL_ATTN_INTERVAL`.
/// Returns `None` if `layer_idx` is a full-attention layer. The
/// modulo arithmetic for the linear index is qwen3_5_moe-specific
/// (full-attn layers are evenly spaced); the kind dispatch goes
/// through [`Variant::layer_kind`] so a future variant can plug in a
/// different layer-kind sequence without touching this helper's
/// callers.
pub fn linear_layer_idx_for(layer_idx: usize) -> Option<usize> {
    use crate::riir::variants::LayerKind;
    if VARIANT.layer_kind(layer_idx) == LayerKind::FullAttn {
        None
    } else {
        Some(layer_idx - (layer_idx + 1) / VARIANT.full_attn_interval)
    }
}

/// `fa_idx = (layer_idx + 1) / FULL_ATTN_INTERVAL - 1`. Returns
/// `None` if `layer_idx` is a linear-attn layer. Mirrors C
/// `(layer_idx + 1) / FULL_ATTN_INTERVAL - 1` at `infer.m:5092`.
pub fn full_attn_layer_idx_for(layer_idx: usize) -> Option<usize> {
    use crate::riir::variants::LayerKind;
    if VARIANT.layer_kind(layer_idx) == LayerKind::FullAttn {
        Some((layer_idx + 1) / VARIANT.full_attn_interval - 1)
    } else {
        None
    }
}

pub(in crate::riir) fn num_full_attn_layers(v: &Variant) -> usize {
    v.num_layers / v.full_attn_interval
}

/// Per-tensor bit-width lookup for the matvec dispatcher. Defaults to
/// 4-bit for tensors not in the manifest; the dispatcher's max(_, 4)
/// floor guards against misreads.
pub(in crate::riir) fn bits_of(wf: &WeightFile, name: &str) -> u32 {
    wf.tensor_info(name)
        .map(|i| i.bits as u32)
        .unwrap_or(4)
        .max(4)
}

/// Prefetch context for the batched layer forwards — session-5
/// Phase 3.
///
/// When `Some`, the caller has fired
/// [`crate::riir::io::prefetch::PrefetchState::dispatch`] for THIS layer before invoking
/// the batched forward and wants the in-flight pread to be drained +
/// set-matched against the bucket experts. Hits route through the
/// per-slot `moe.data_prefetch[set]` buffers (zero-I/O); misses fall
/// through to the existing mmap-or-pread path.
///
/// Caller fires the dispatch only at small N (the per-layer
/// prefetch is wasted work at N ≥ 32 — the bucket spans most or all
/// experts) and only in pread mode (mmap mode has zero pread cost
/// to begin with; the prefetch's pread would duplicate work the
/// kernel's demand-fault handler is about to do anyway).
pub(in crate::riir) struct PrefetchEnv<'a> {
    pub prefetch: &'a mut crate::riir::io::prefetch::PrefetchState,
    // `prefetch_set` field removed 2026-05-20 alongside the pread
    // teardown: it indexed the cold-path lookup of
    // `moe_buffers.data_prefetch_id(set, buf_idx)` inside
    // `moe_block_forward`, which is gone. `pe.prefetch.record_actual`
    // (still used by the producer) doesn't need it.
}

/// Adapter naming the o_proj weights + input shape to use for one
/// call into [`post_attention_tail`]. Linear-attn fills with
/// `linear_o_proj_*`; full-attn fills with `full_o_proj_*`.
pub(in crate::riir) struct OProj {
    pub w_off: u64,
    pub s_off: u64,
    pub b_off: u64,
    pub bits: u32,
    /// Number of input floats the matvec reads from
    /// `buffers.o_proj_stack` (CPU SDPA path) or
    /// `buffers.gpu_attn_out` (GPU SDPA path). Linear:
    /// `linear_total_value`. Full: `num_attn_heads * head_dim`.
    pub in_dim: u32,
}

/// Slice 5d-7b — args for the GPU SDPA fast path encoded at the top
/// of CMD2 inside [`post_attention_tail`]. Carries the per-call
/// inputs not derivable from `VARIANT`: which full-attn KV mirror
/// slot to use, and the current KV length. When `Some`, the tail
/// encodes the 4 attn kernels (`attn_scores_batched` →
/// `attn_softmax_batched` → `attn_values_batched` → `sigmoid_gate`)
/// into the same cmdbuf as `o_proj`, residual_add, and post-attn
/// rms_norm — no extra commit-wait. Q + q_gate are pre-staged into
/// `buffers.gpu_attn_q` / `buffers.gpu_attn_gate` by the caller; K/V
/// mirrors are pre-populated by the per-token KV-append memcpy.
///
/// When `None`, the tail follows the existing CPU-attn path: o_proj
/// reads from `buffers.o_proj_stack` (caller-staged via
/// `sdpa_cpu` + memcpy).
pub(in crate::riir) struct GpuAttnEncodeArgs {
    /// Index into `LayerForwardBuffers::gpu_kv_k` / `gpu_kv_v`. From
    /// [`full_attn_layer_idx_for`].
    pub fa_idx: usize,
    /// `kv_state.len` after this token's KV append — the number of
    /// positions the kernels read from the mirror.
    pub kv_len: u32,
}

/// Run one linear-attention layer's forward pass on the GPU.
///
/// Pre: `buffers.input` holds the input hidden state (HIDDEN_DIM
/// floats). The targeted layer's `conv_state` / `delta_state` are
/// mutated in place. `state` is the host-side mirror used for
/// `memory_*` ops; for 4c we keep it in lockstep with the GPU
/// buffers via reset only — partial truncation will resync GPU
/// buffers via `reset_recurrence` (a faithful port of the lossy
/// semantic).
///
/// Post: `*deferred` holds an in-flight K-expert dispatch (committed
/// without `wait`). The caller is responsible for draining it via
/// `complete_deferred_experts_into` (writes the post-combine hidden
/// into a host slice or the next layer's `buffers.input`) or
/// `discard_deferred_experts_in` (drop without readback). `buffers.
/// input` is **not** the output target after slice 4f-3.
#[allow(clippy::too_many_arguments)]
pub fn linear_attn_layer_forward(
    metal: &mut MetalContext,
    gpu: &GpuLayerCtx<'_>,
    moe: &mut MoeBuffers,
    deferred: &mut crate::riir::moe::deferred::DeferredRing,
    layer_idx: usize,
    k_active: usize,
    expert_files: &ExpertFiles,
    pool: &rayon::ThreadPool,
    prefetch: &mut crate::riir::io::prefetch::PrefetchState,
    // Slice 5d-9: which `data_prefetch` set this layer reads from
    // (parity ping-pong: `layer_idx % 2`). Plumbed through to
    // `post_attention_tail`'s K-expert dispatch.
    prefetch_set: usize,
    _layer_state: &mut LinearAttnState,
    gpu_combine: bool,
    // Slice 5d-8: previous layer's K-expert dispatch chained the
    // post-combine rms_norm into its cmdbuf, so `buffers.normed` is
    // already populated when we land here. Skip CMD1's input-norm
    // prelude in that case.
    prev_layer_chained: bool,
    // Slice 5d-8: when `Some`, this layer's K-expert cmdbuf appends
    // rms_norm_sum_sq + rms_norm_apply_bf16 against the next layer's
    // input_layernorm.weight at this offset (in `wf_buf`). Output lands
    // in `buffers.normed`, ready for the next layer's CMD1. `None`
    // disables the chain — used for the last layer and the dump hook.
    chain_next_norm_off: Option<u64>,
) -> Result<(), LayerForwardError> {
    let GpuLayerCtx { wf, wf_buf, layer_cache, buffers, buffer_pool } =
        *gpu;
    let v = VARIANT;
    let linear_layer_idx = linear_layer_idx_for(layer_idx).ok_or(
        LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "linear_layer_idx (called on full-attn layer)",
        },
    )?;

    // Per-tensor bit width lookup. 4-bit is the default; A3B uses
    // 8-bit for `mlp.gate.weight` and `mlp.shared_expert_gate.weight`.
    let qkv_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight"),
    );
    let z_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.linear_attn.in_proj_z.weight"),
    );
    let alpha_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.linear_attn.in_proj_a.weight"),
    );
    let beta_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.linear_attn.in_proj_b.weight"),
    );
    let o_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.linear_attn.out_proj.weight"),
    );

    // Pull the linear-attn-specific offsets out of the tagged-enum
    // cache. Returning early with `MissingTensor` here also guards
    // against accidentally calling this function on a full-attn layer
    // (the dispatcher in `layer_forward_dump` already filters; this
    // is defense in depth and matches the symmetric guard in
    // `full_attn_layer_forward`).
    let attn = layer_cache.attn.linear().ok_or(
        LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "linear_attn weights (called on full-attn layer)",
        },
    )?;
    let qkv_w = attn.qkv_w;
    let qkv_s = attn.qkv_s;
    let qkv_b = attn.qkv_b;
    let z_w = attn.z_w;
    let z_s = attn.z_s;
    let z_b = attn.z_b;
    let beta_w = attn.beta_w;
    let beta_s = attn.beta_s;
    let beta_b = attn.beta_b;
    let alpha_w = attn.alpha_w;
    let alpha_s = attn.alpha_s;
    let alpha_b = attn.alpha_b;
    let conv1d_w = attn.conv1d_w;
    let a_log = attn.a_log;
    let dt_bias = attn.dt_bias;
    let gnorm_w = attn.gated_norm_w;
    let o_w = attn.o_proj_w;
    let o_s = attn.o_proj_s;
    let o_b = attn.o_proj_b;

    // Pre-fetch every pipeline.
    let lp = LinearAttnPipelines::fetch(metal)?;
    let mv = MatvecPipelines::fetch(metal)?;
    let rms_pipes = RmsNormBf16Pipelines::fetch(metal)?;

    // ── CMD1: input rms_norm + projections + linear-attn pipeline ─
    //
    // Slice 5d-2: input rms_norm runs on the GPU as the prelude to
    // CMD1 (was CPU + 4 host↔GPU memcopies before). C path runs CPU
    // rms_norm in the slow branch (`infer.m:4492`) but the fast branch
    // chains a GPU rms_norm at the tail of the previous CMD3
    // (`infer.m:5712..5744`); functionally we get the latter shape by
    // running the same kernel pair as the first dispatches in CMD1.
    // Slice 9e established that the `rms_norm_sum_sq` /
    // `rms_norm_apply_bf16` pair is bit-exact per-PSO; agreement
    // against the C fast path is bit-exact, agreement against the C
    // slow path drifts by a few ULPs per layer (well within the
    // existing diff floor `cosine ≥ 0.9999`).
    //
    // `buffers.input` is the residual source consumed by
    // `post_attention_tail`'s `encode_residual_add` later this layer;
    // nothing in this layer's forward writes to it (the next layer's
    // top-of-forward `complete_deferred_experts_into` is the next
    // mutation), so the dual-use is safe.
    //
    // Slice cmdbuf-fold-1: the cmdbuf is hoisted out of CMD1 and
    // reused by `post_attention_tail` for CMD2+3, eliminating the
    // CMD1 commit-wait. Encoders within a single cmdbuf serialize
    // on Metal so the data flow is honored without the synchronous
    // boundary. Net per-layer count: 3 → 2 cmdbufs for linear-attn.
    //
    // `queue_clone` (not `queue`) so the cmdbuf borrow doesn't pin
    // `*metal` and block the `&mut metal` arg to `post_attention_tail`.
    let queue = metal.queue_clone();
    let cmdbuf = queue.new_command_buffer();
    {
        // Slice 5d-8: skip the input-norm prelude when the previous
        // layer chained — `buffers.normed` is already populated by the
        // appended rms_norm at the tail of the previous K-expert cmdbuf
        // (drained at the top of `step_internal`'s layer iteration).
        if !prev_layer_chained {
            encode_rms_norm_bf16_into(
                cmdbuf,
                &rms_pipes,
                buffer_pool.handle(buffers.input),
                wf_buf.buffer(),
                layer_cache.input_layernorm_w,
                buffer_pool.handle(buffers.sum_sq),
                buffer_pool.handle(buffers.normed),
                v.hidden_dim as u32,
                RMS_NORM_EPS,
            );
        }

        // 4 batched projections from buffers.normed:
        let specs = [
            MatvecSpec {
                w_off: qkv_w,
                s_off: qkv_s,
                b_off: qkv_b,
                input: buffer_pool.handle(buffers.normed),
                output: buffer_pool.handle(buffers.q_stack),
                out_dim: v.linear_conv_dim() as u32,
                in_dim: v.hidden_dim as u32,
                bits: qkv_bits,
            },
            MatvecSpec {
                w_off: z_w,
                s_off: z_s,
                b_off: z_b,
                input: buffer_pool.handle(buffers.normed),
                output: buffer_pool.handle(buffers.z_stack),
                out_dim: v.linear_total_value() as u32,
                in_dim: v.hidden_dim as u32,
                bits: z_bits,
            },
            MatvecSpec {
                w_off: beta_w,
                s_off: beta_s,
                b_off: beta_b,
                input: buffer_pool.handle(buffers.normed),
                output: buffer_pool.handle(buffers.beta_stack),
                out_dim: v.linear_num_v_heads as u32,
                in_dim: v.hidden_dim as u32,
                bits: beta_bits,
            },
            MatvecSpec {
                w_off: alpha_w,
                s_off: alpha_s,
                b_off: alpha_b,
                input: buffer_pool.handle(buffers.normed),
                output: buffer_pool.handle(buffers.alpha_stack),
                out_dim: v.linear_num_v_heads as u32,
                in_dim: v.hidden_dim as u32,
                bits: alpha_bits,
            },
        ];
        for s in &specs {
            encode_matvec(cmdbuf, &mv, wf_buf, s);
        }

        encode_conv1d_step(
            cmdbuf,
            &lp.conv1d_step,
            &lp.conv1d_state_update,
            buffer_pool.handle(buffers.conv_state[linear_layer_idx]),
            buffer_pool.handle(buffers.q_stack),
            0,
            wf_buf.buffer(),
            conv1d_w,
            buffer_pool.handle(buffers.conv_output),
            v.linear_conv_dim() as u32,
        );

        encode_rms_norm_qk(
            cmdbuf,
            &lp.rms_norm_qk,
            buffer_pool.handle(buffers.conv_output),
            v.linear_num_k_heads as u32,
            Variant::LINEAR_KEY_DIM as u32,
        );

        encode_compute_decay_beta(
            cmdbuf,
            &lp.compute_decay_beta,
            buffer_pool.handle(buffers.alpha_stack), // alpha
            0,
            buffer_pool.handle(buffers.beta_stack), // beta
            0,
            wf_buf.buffer(),
            a_log,
            dt_bias,
            buffer_pool.handle(buffers.delta_g_decay),
            buffer_pool.handle(buffers.delta_beta),
            v.linear_num_v_heads as u32,
        );

        let k_heads_per_v =
            (v.linear_num_v_heads / v.linear_num_k_heads) as u32;
        encode_delta_net_step(
            cmdbuf,
            &lp.delta_net_step,
            buffer_pool.handle(buffers.delta_state[linear_layer_idx]),
            buffer_pool.handle(buffers.conv_output),
            buffer_pool.handle(buffers.delta_g_decay),
            buffer_pool.handle(buffers.delta_beta),
            buffer_pool.handle(buffers.delta_output),
            v.linear_num_v_heads as u32,
            Variant::LINEAR_VALUE_DIM as u32,
            k_heads_per_v,
        );

        encode_gated_rms_norm(
            cmdbuf,
            &lp.gated_rms_norm,
            buffer_pool.handle(buffers.delta_output),
            buffer_pool.handle(buffers.z_stack), // z
            0,
            wf_buf.buffer(),
            gnorm_w,
            buffer_pool.handle(buffers.o_proj_stack),
            0,
            v.linear_num_v_heads as u32,
            Variant::LINEAR_VALUE_DIM as u32,
        );

        // Slice cmdbuf-fold-1: CMD1's commit+wait is gone — the tail
        // commits the same cmdbuf below after encoding CMD2+3.
    }

    // ── Hand off to the shared post-attention tail ───────────────
    // `batch_out[6]` already holds the `gated_rms_norm` output —
    // exactly the o_proj input the tail consumes. The tail leaves an
    // in-flight K-expert dispatch in `*deferred`. Linear-attn never
    // takes the GPU SDPA fast path (it has no attention-kernel
    // pipeline), so `gpu_attn_args = None`.
    post_attention_tail(
        metal,
        cmdbuf,
        gpu,
        moe,
        deferred,
        layer_idx,
        k_active,
        expert_files,
        pool,
        prefetch,
        prefetch_set,
        OProj {
            w_off: o_w,
            s_off: o_s,
            b_off: o_b,
            bits: o_bits,
            in_dim: v.linear_total_value() as u32,
        },
        gpu_combine,
        /* gpu_attn_args = */ None,
        chain_next_norm_off,
    )
}

/// Host-side bookkeeping captured by [`post_attention_pre_moe`] and
/// consumed by [`moe_dispatch_per_token`] (per-token path) or by the
/// batched-prefill orchestrator (Phase B+). The GPU buffers
/// (`h_mid`, `normed`/`h_post`, `shared_out`) live in
/// [`LayerForwardBuffers`] and are referenced by callers directly;
/// this struct only carries the host-side values produced by the
/// CPU MoE router + shared-gate readback.
pub(in crate::riir) struct PostAttnIntermediates {
    /// Top-K expert indices from the CPU router, length = `k_active`.
    pub routing_indices: Vec<i32>,
    /// Top-K expert weights (normalized softmax), length = `k_active`.
    pub routing_weights: Vec<f32>,
    /// Pre-sigmoid shared-expert gate scalar.
    pub shared_gate_score: f32,
}

/// Shared post-attention tail used by both linear- and full-attention
/// layer forwards. Composed of [`post_attention_pre_moe`] (CMD2+3
/// work up to and including the CPU MoE router) and
/// [`moe_dispatch_per_token`] (K-expert dispatch + combine + optional
/// chained rms_norm into the next layer's input). Reads the
/// attention output from `buffers.o_proj_stack` (caller-staged) and
/// runs the rest of `fused_layer_forward`:
///
/// 1. CMD2: o_proj matvec → residual_add → post-attn rms_norm.
/// 2. CMD3a: gate logits + shared-expert gate scalar + shared FFN
///    `gate_proj` + `up_proj`.
/// 3. GPU swiglu of shared_gate × shared_up → `shared_act`.
/// 4. CMD3a-b: shared `down_proj` matvec.
/// 5. CPU MoE router on the gate logits.
/// 6. Load K expert blobs from disk via [`ExpertFiles::read_expert`].
/// 7. CMD3b: K-expert FFN + combine — submits the dispatch via
///    [`gpu_batched_experts_begin`] without waiting; ownership of
///    the in-flight cmdbuf transfers to `*deferred`.
///
/// On return, `*deferred` holds the in-flight K-expert dispatch.
/// The caller is responsible for either `complete_deferred_experts_into`
/// (drain into a host slice / next layer's `buffers.input`) or
/// `discard_deferred_experts_in` (drop without readback) before the
/// next `post_attention_tail` call. Slice 4f-3 wired this rewire;
/// before, the synchronous `gpu_batched_experts_forward` ran inline
/// and the result was written into `buffers.input` here. The async
/// path matches the C-side `g_deferred` lifecycle and unlocks the
/// fast/slow split landing in 4f-perf.
///
/// Session 4 split this function in two so the batched-prefill
/// orchestrator can run [`post_attention_pre_moe`] in a per-token
/// loop to populate the joint N×k_active routing CSR before
/// [`crate::riir::moe::expert_forward::encode_moe_batched_permute_fuse`], and
/// then call a batched combine kernel instead of the per-token
/// K-expert dispatch. The per-token path here remains identical in
/// behaviour.
#[allow(clippy::too_many_arguments)]
pub(in crate::riir) fn post_attention_tail(
    metal: &mut MetalContext,
    // Caller-owned cmdbuf into which CMD2+3 work is encoded. The
    // linear-attn caller passes its CMD1 cmdbuf so projections +
    // linear-attn fused kernels share the same submit (slice
    // cmdbuf-fold-1). Full-attn callers must create a fresh cmdbuf
    // *after* the host-bounce (q/k/v readback + CPU per-head norm +
    // RoPE + KV append) and pass it here. `commit + wait` of CMD2+3
    // happens inside this function; the caller must not commit before
    // returning.
    cmdbuf: &CommandBufferRef,
    gpu: &GpuLayerCtx<'_>,
    moe: &mut MoeBuffers,
    deferred: &mut crate::riir::moe::deferred::DeferredRing,
    layer_idx: usize,
    k_active: usize,
    expert_files: &ExpertFiles,
    pool: &rayon::ThreadPool,
    prefetch: &mut crate::riir::io::prefetch::PrefetchState,
    // Slice 5d-9: which `data_prefetch` set this layer reads from. The
    // caller assigns set `layer_idx % 2`; the prefetch state machine
    // wrote that same set at the top of this layer's iteration.
    prefetch_set: usize,
    o_proj: OProj,
    gpu_combine: bool,
    gpu_attn_args: Option<GpuAttnEncodeArgs>,
    // Slice 5d-8: when `Some` AND `gpu_combine` is true, the K-expert
    // cmdbuf appends rms_norm_sum_sq + rms_norm_apply_bf16 against the
    // next layer's input_layernorm.weight at this offset (in `wf_buf`).
    // Output lands in `buffers.normed`, ready for the next layer's
    // CMD1. `None` (or CPU-combine) disables the chain.
    chain_next_norm_off: Option<u64>,
) -> Result<(), LayerForwardError> {
    let GpuLayerCtx { wf: _, wf_buf, layer_cache: _, buffers, buffer_pool } =
        *gpu;
    let intermediates = post_attention_pre_moe(
        metal,
        cmdbuf,
        gpu,
        layer_idx,
        k_active,
        o_proj,
        gpu_attn_args,
    )?;
    moe_dispatch_per_token(
        metal,
        wf_buf,
        buffers,
        buffer_pool,
        moe,
        deferred,
        layer_idx,
        expert_files,
        pool,
        prefetch,
        prefetch_set,
        &intermediates,
        gpu_combine,
        chain_next_norm_off,
    )
}

/// First half of [`post_attention_tail`]: encode CMD2+3 (o_proj →
/// residual → post-attn rms_norm → gate logits + shared-gate scalar
/// → shared FFN gate/up/swiglu/down) into the caller-owned `cmdbuf`,
/// commit + wait, then run the CPU MoE router on the gate logits and
/// read back the shared-gate scalar. Returns
/// [`PostAttnIntermediates`] for the dispatch / combine stages.
///
/// Linear-attn callers pass their CMD1 cmdbuf so projections +
/// linear-attn fused kernels share the same submit (slice
/// cmdbuf-fold-1); full-attn callers create a fresh cmdbuf after the
/// host-bounce (q/k/v readback + CPU per-head norm + RoPE + KV
/// append).
///
/// Phase B's batched-prefill orchestrator runs this in a per-token
/// loop to populate the joint N×k_active routing CSR before
/// [`crate::riir::moe::expert_forward::encode_moe_batched_permute_fuse`].
#[allow(clippy::too_many_arguments)]
pub(in crate::riir) fn post_attention_pre_moe(
    metal: &mut MetalContext,
    cmdbuf: &CommandBufferRef,
    gpu: &GpuLayerCtx<'_>,
    layer_idx: usize,
    k_active: usize,
    o_proj: OProj,
    gpu_attn_args: Option<GpuAttnEncodeArgs>,
) -> Result<PostAttnIntermediates, LayerForwardError> {
    let GpuLayerCtx { wf, wf_buf, layer_cache, buffers, buffer_pool } =
        *gpu;
    let v = VARIANT;

    // Per-tensor bit widths for the MoE-side matvecs.
    let gate_bits =
        bits_of(wf, &format!("model.layers.{layer_idx}.mlp.gate.weight"));
    let seg_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert_gate.weight"
        ),
    );
    let s_gate_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight"
        ),
    );
    let s_up_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight"
        ),
    );
    let s_down_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight"
        ),
    );

    let post_attn_norm_w = layer_cache.post_attention_layernorm_w;
    let gate_w = layer_cache.gate.w;
    let gate_s = layer_cache.gate.s;
    let gate_b = layer_cache.gate.b;
    let shared_up_w = layer_cache.shared.up_w;
    let shared_up_s = layer_cache.shared.up_s;
    let shared_up_b = layer_cache.shared.up_b;
    let shared_gate_w = layer_cache.shared.gate_w;
    let shared_gate_s = layer_cache.shared.gate_s;
    let shared_gate_b = layer_cache.shared.gate_b;
    let shared_down_w = layer_cache.shared.down_w;
    let shared_down_s = layer_cache.shared.down_s;
    let shared_down_b = layer_cache.shared.down_b;
    let seg_w = layer_cache.shared.seg_w;
    let seg_s = layer_cache.shared.seg_s;
    let seg_b = layer_cache.shared.seg_b;

    let mv = MatvecPipelines::fetch(metal)?;
    let sum_sq = metal.pipeline("rms_norm_sum_sq")?.clone();
    let apply = metal.pipeline("rms_norm_apply_bf16")?.clone();
    let resid_add = metal.pipeline("residual_add")?.clone();
    let swiglu = metal.pipeline("swiglu_fused")?.clone();
    // Slice 5d-7b: pre-fetch attn pipelines only when the GPU SDPA
    // fast path is active. Keeps the CPU-SDPA / linear-attn paths
    // free of unrelated pipeline lookups.
    let attn_pipes = if gpu_attn_args.is_some() {
        Some(GpuAttnPipelines::fetch(metal)?)
    } else {
        None
    };

    // ── CMD2+3: post-attn + shared FFN + gate logits, single cmdbuf ─
    //
    // Slice 5d-3: collapses the previous CMD2 / CMD3a / CMD3a-b
    // commit+wait sequence into a single command buffer. The C path
    // also fuses post-attn + routing + shared-FFN gate/up into ONE
    // cmdbuf (`infer.m:5088..5258`, the `cmd_fused` block); we now
    // additionally fold the shared-FFN swiglu (was CPU) and the
    // shared_down matvec into the same buffer, eliminating the
    // CPU-side swiglu loop and the separate `cmd_dn` shape.
    //
    // GPU swiglu replaces the CPU SiLU loop the C path uses for the
    // shared FFN (`infer.m:2977 cpu_swiglu`). Per slice 9a's finding,
    // `swiglu_fused` is bit-exact per-PSO; the drift against C's
    // CPU swiglu here is small libm-precision territory and remains
    // within the diff oracle's `cosine ≥ 0.9999` floor. The K-expert
    // FFN already used GPU swiglu inside `gpu_batched_experts_*`, so
    // this is the only remaining cpu_swiglu site outside the experts.
    //
    // Encoders within one cmdbuf serialize on Metal, so the data
    // dependencies (o_proj → residual_add → rms_norm → projections →
    // swiglu → shared_down) are honored without per-encoder waits.
    //
    // The cmdbuf is now caller-owned (slice cmdbuf-fold-1). Linear-attn
    // callers pass their CMD1 cmdbuf so projections + linear-attn fused
    // kernels share the same submit; full-attn callers create a fresh
    // cmdbuf after the host-bounce.
    {
        // ── Slice 5d-7b: GPU full-attn fast path (Enc A1..A4) ──────
        //
        // When active, encode the 4 attn kernels at the head of CMD2
        // so SDPA + sigmoid gate piggyback on the same commit-wait as
        // o_proj + residual + post-attn rms_norm. Mirrors the C path's
        // `gpu_attn_fuse` block at `infer.m:5091..5163`. Q + q_gate
        // are pre-staged into `buffers.gpu_attn_q` / `gpu_attn_gate` by
        // the caller; K/V mirrors are pre-populated by the per-token
        // KV-append memcpy.
        if let (Some(args), Some(attn_pipes)) =
            (gpu_attn_args.as_ref(), attn_pipes.as_ref())
        {
            let head_dim = v.head_dim as u32;
            let kv_dim = (v.num_kv_heads * v.head_dim) as u32;
            let num_heads = v.num_attn_heads as u32;
            let heads_per_kv = (v.num_attn_heads / v.num_kv_heads) as u32;
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let seq_stride = crate::riir::variants::GPU_KV_SEQ as u32;

            encode_attn_scores_batched_into(
                cmdbuf,
                &attn_pipes.scores,
                buffer_pool.handle(buffers.gpu_attn_q),
                buffer_pool.handle(buffers.gpu_kv_k[args.fa_idx]),
                buffer_pool.handle(buffers.gpu_attn_scores),
                num_heads,
                head_dim,
                kv_dim,
                args.kv_len,
                seq_stride,
                heads_per_kv,
                scale,
            );
            encode_attn_softmax_batched_into(
                cmdbuf,
                &attn_pipes.softmax,
                buffer_pool.handle(buffers.gpu_attn_scores),
                num_heads,
                args.kv_len,
                seq_stride,
            );
            encode_attn_values_batched_into(
                cmdbuf,
                &attn_pipes.values,
                buffer_pool.handle(buffers.gpu_attn_scores),
                buffer_pool.handle(buffers.gpu_kv_v[args.fa_idx]),
                buffer_pool.handle(buffers.gpu_attn_out),
                num_heads,
                head_dim,
                kv_dim,
                args.kv_len,
                seq_stride,
                heads_per_kv,
            );
            encode_sigmoid_gate_into(
                cmdbuf,
                &attn_pipes.gate,
                buffer_pool.handle(buffers.gpu_attn_out),
                buffer_pool.handle(buffers.gpu_attn_gate),
                num_heads * head_dim,
            );
        }

        // o_proj + residual_add + post-attn rms_norm (was CMD2).
        // GPU SDPA path: read from `gpu_attn_out` (zero-host-stage).
        // CPU SDPA / linear-attn paths: read from `batch_out[6]`.
        let oproj_input = if gpu_attn_args.is_some() {
            buffer_pool.handle(buffers.gpu_attn_out)
        } else {
            buffer_pool.handle(buffers.o_proj_stack)
        };
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: o_proj.w_off,
                s_off: o_proj.s_off,
                b_off: o_proj.b_off,
                input: oproj_input,
                output: buffer_pool.handle(buffers.output),
                out_dim: v.hidden_dim as u32,
                in_dim: o_proj.in_dim,
                bits: o_proj.bits,
            },
        );
        encode_residual_add(
            cmdbuf,
            &resid_add,
            buffer_pool.handle(buffers.output),
            buffer_pool.handle(buffers.input), // residual source — see slice 5d-2 note
            buffer_pool.handle(buffers.h_mid),
            v.hidden_dim as u32,
        );
        encode_rms_norm_pair(
            cmdbuf,
            &sum_sq,
            &apply,
            buffer_pool.handle(buffers.h_mid),
            wf_buf.buffer(),
            post_attn_norm_w,
            buffer_pool.handle(buffers.normed),
            buffer_pool.handle(buffers.sum_sq),
            v.hidden_dim as u32,
        );

        // gate logits + shared-expert gate scalar + shared FFN
        // gate/up matvecs (was CMD3a).
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: gate_w,
                s_off: gate_s,
                b_off: gate_b,
                input: buffer_pool.handle(buffers.normed),
                output: buffer_pool.handle(buffers.gate_logits),
                out_dim: v.num_experts as u32,
                in_dim: v.hidden_dim as u32,
                bits: gate_bits,
            },
        );
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: seg_w,
                s_off: seg_s,
                b_off: seg_b,
                input: buffer_pool.handle(buffers.normed),
                output: buffer_pool.handle(buffers.shared_gate),
                out_dim: 1,
                in_dim: v.hidden_dim as u32,
                bits: seg_bits,
            },
        );
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: shared_gate_w,
                s_off: shared_gate_s,
                b_off: shared_gate_b,
                input: buffer_pool.handle(buffers.normed),
                output: buffer_pool.handle(buffers.shared_gate_out),
                out_dim: v.shared_intermediate as u32,
                in_dim: v.hidden_dim as u32,
                bits: s_gate_bits,
            },
        );
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: shared_up_w,
                s_off: shared_up_s,
                b_off: shared_up_b,
                input: buffer_pool.handle(buffers.normed),
                output: buffer_pool.handle(buffers.shared_up_out),
                out_dim: v.shared_intermediate as u32,
                in_dim: v.hidden_dim as u32,
                bits: s_up_bits,
            },
        );

        // GPU swiglu — was the CPU loop between CMD3a and CMD3a-b.
        encode_swiglu_buf(
            cmdbuf,
            &swiglu,
            buffer_pool.handle(buffers.shared_gate_out),
            buffer_pool.handle(buffers.shared_up_out),
            buffer_pool.handle(buffers.shared_act),
            v.shared_intermediate as u32,
        );

        // shared_down matvec (was CMD3a-b).
        encode_matvec(
            cmdbuf,
            &mv,
            wf_buf,
            &MatvecSpec {
                w_off: shared_down_w,
                s_off: shared_down_s,
                b_off: shared_down_b,
                input: buffer_pool.handle(buffers.shared_act),
                output: buffer_pool.handle(buffers.shared_out),
                out_dim: v.hidden_dim as u32,
                in_dim: v.shared_intermediate as u32,
                bits: s_down_bits,
            },
        );

        metal.commit_and_wait_labeled(cmdbuf, "post_attn_tail.cmd2_3");
    }

    // ── CPU: MoE router on the gate logits ───────────────────────
    let mut scores =
        read_buffer_to_vec(buffer_pool.handle(buffers.gate_logits), v.num_experts);
    let mut routing_indices = vec![0i32; k_active];
    let mut routing_weights = vec![0f32; k_active];
    moe_router_cpu(
        &mut scores,
        k_active,
        &mut routing_indices,
        &mut routing_weights,
    )?;

    // Read shared-gate score scalar (pre-sigmoid).
    let shared_gate_score = {
        let s = read_buffer_to_vec(buffer_pool.handle(buffers.shared_gate), 1);
        s[0]
    };

    Ok(PostAttnIntermediates {
        routing_indices,
        routing_weights,
        shared_gate_score,
    })
}

/// B3-era variant of [`post_attention_pre_moe`] that assumes
/// `buffers.output` already holds the o_proj result (typically
/// populated via batched o_proj outside this function). Skips the
/// o_proj matvec entirely; runs residual_add → post-attn rms_norm →
/// gate logits → shared-gate scalar → shared FFN inside the caller-
/// owned `cmdbuf`.
///
/// B4 superseded this for `batched_full_attn_layer_forward` by also
/// batching the shared FFN externally.
/// Kept here as a forward-looking building block for callers (e.g.
/// a future eval_token path) that want o_proj batched but per-token
/// shared FFN inside one cmdbuf.
#[allow(dead_code, clippy::too_many_arguments)]
pub(in crate::riir) fn post_attention_post_o_proj_to_intermediates(
    metal: &mut MetalContext,
    cmdbuf: &CommandBufferRef,
    wf: &WeightFile,
    wf_buf: &MtlWeightBuf,
    layer_cache: &LayerWeightCache,
    buffers: &LayerForwardBuffers,
    buffer_pool: &MetalBufferPool,
    layer_idx: usize,
    k_active: usize,
) -> Result<PostAttnIntermediates, LayerForwardError> {
    let v = VARIANT;

    let gate_bits =
        bits_of(wf, &format!("model.layers.{layer_idx}.mlp.gate.weight"));
    let seg_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert_gate.weight"
        ),
    );
    let s_gate_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight"
        ),
    );
    let s_up_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight"
        ),
    );
    let s_down_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight"
        ),
    );

    let post_attn_norm_w = layer_cache.post_attention_layernorm_w;
    let gate_w = layer_cache.gate.w;
    let gate_s = layer_cache.gate.s;
    let gate_b = layer_cache.gate.b;
    let shared_up_w = layer_cache.shared.up_w;
    let shared_up_s = layer_cache.shared.up_s;
    let shared_up_b = layer_cache.shared.up_b;
    let shared_gate_w = layer_cache.shared.gate_w;
    let shared_gate_s = layer_cache.shared.gate_s;
    let shared_gate_b = layer_cache.shared.gate_b;
    let shared_down_w = layer_cache.shared.down_w;
    let shared_down_s = layer_cache.shared.down_s;
    let shared_down_b = layer_cache.shared.down_b;
    let seg_w = layer_cache.shared.seg_w;
    let seg_s = layer_cache.shared.seg_s;
    let seg_b = layer_cache.shared.seg_b;

    let mv = MatvecPipelines::fetch(metal)?;
    let sum_sq = metal.pipeline("rms_norm_sum_sq")?.clone();
    let apply = metal.pipeline("rms_norm_apply_bf16")?.clone();
    let resid_add = metal.pipeline("residual_add")?.clone();
    let swiglu = metal.pipeline("swiglu_fused")?.clone();

    encode_residual_add(
        cmdbuf,
        &resid_add,
        buffer_pool.handle(buffers.output),
        buffer_pool.handle(buffers.input),
        buffer_pool.handle(buffers.h_mid),
        v.hidden_dim as u32,
    );
    encode_rms_norm_pair(
        cmdbuf,
        &sum_sq,
        &apply,
        buffer_pool.handle(buffers.h_mid),
        wf_buf.buffer(),
        post_attn_norm_w,
        buffer_pool.handle(buffers.normed),
        buffer_pool.handle(buffers.sum_sq),
        v.hidden_dim as u32,
    );
    encode_matvec(
        cmdbuf,
        &mv,
        wf_buf,
        &MatvecSpec {
            w_off: gate_w,
            s_off: gate_s,
            b_off: gate_b,
            input: buffer_pool.handle(buffers.normed),
            output: buffer_pool.handle(buffers.gate_logits),
            out_dim: v.num_experts as u32,
            in_dim: v.hidden_dim as u32,
            bits: gate_bits,
        },
    );
    encode_matvec(
        cmdbuf,
        &mv,
        wf_buf,
        &MatvecSpec {
            w_off: seg_w,
            s_off: seg_s,
            b_off: seg_b,
            input: buffer_pool.handle(buffers.normed),
            output: buffer_pool.handle(buffers.shared_gate),
            out_dim: 1,
            in_dim: v.hidden_dim as u32,
            bits: seg_bits,
        },
    );
    encode_matvec(
        cmdbuf,
        &mv,
        wf_buf,
        &MatvecSpec {
            w_off: shared_gate_w,
            s_off: shared_gate_s,
            b_off: shared_gate_b,
            input: buffer_pool.handle(buffers.normed),
            output: buffer_pool.handle(buffers.shared_gate_out),
            out_dim: v.shared_intermediate as u32,
            in_dim: v.hidden_dim as u32,
            bits: s_gate_bits,
        },
    );
    encode_matvec(
        cmdbuf,
        &mv,
        wf_buf,
        &MatvecSpec {
            w_off: shared_up_w,
            s_off: shared_up_s,
            b_off: shared_up_b,
            input: buffer_pool.handle(buffers.normed),
            output: buffer_pool.handle(buffers.shared_up_out),
            out_dim: v.shared_intermediate as u32,
            in_dim: v.hidden_dim as u32,
            bits: s_up_bits,
        },
    );
    encode_swiglu_buf(
        cmdbuf,
        &swiglu,
        buffer_pool.handle(buffers.shared_gate_out),
        buffer_pool.handle(buffers.shared_up_out),
        buffer_pool.handle(buffers.shared_act),
        v.shared_intermediate as u32,
    );
    encode_matvec(
        cmdbuf,
        &mv,
        wf_buf,
        &MatvecSpec {
            w_off: shared_down_w,
            s_off: shared_down_s,
            b_off: shared_down_b,
            input: buffer_pool.handle(buffers.shared_act),
            output: buffer_pool.handle(buffers.shared_out),
            out_dim: v.hidden_dim as u32,
            in_dim: v.shared_intermediate as u32,
            bits: s_down_bits,
        },
    );
    metal.commit_and_wait_labeled(cmdbuf, "post_attn_post_oproj.cmd");

    let mut scores =
        read_buffer_to_vec(buffer_pool.handle(buffers.gate_logits), v.num_experts);
    let mut routing_indices = vec![0i32; k_active];
    let mut routing_weights = vec![0f32; k_active];
    moe_router_cpu(
        &mut scores,
        k_active,
        &mut routing_indices,
        &mut routing_weights,
    )?;
    let shared_gate_score = {
        let s = read_buffer_to_vec(buffer_pool.handle(buffers.shared_gate), 1);
        s[0]
    };
    Ok(PostAttnIntermediates {
        routing_indices,
        routing_weights,
        shared_gate_score,
    })
}

/// Second half of [`post_attention_tail`]: K-expert FFN dispatch +
/// combine (and optional chain-rms-norm into the next layer's input)
/// against the [`PostAttnIntermediates`] from
/// [`post_attention_pre_moe`]. Handles the prefetch hit/miss state
/// machine, sync-pread for misses, records actuals for the next
/// token's prediction, and fires the async K-expert dispatch.
///
/// On return, `*deferred` holds the in-flight K-expert dispatch.
/// The caller drains it (next layer's top-of-forward
/// `complete_deferred_experts_into`, or `RsCtx::layer_forward_dump`'s
/// post-dispatch drain).
#[allow(clippy::too_many_arguments)]
pub(in crate::riir) fn moe_dispatch_per_token(
    metal: &mut MetalContext,
    wf_buf: &MtlWeightBuf,
    buffers: &LayerForwardBuffers,
    buffer_pool: &MetalBufferPool,
    moe: &mut MoeBuffers,
    deferred: &mut crate::riir::moe::deferred::DeferredRing,
    layer_idx: usize,
    expert_files: &ExpertFiles,
    _pool: &rayon::ThreadPool,
    _prefetch: &mut crate::riir::io::prefetch::PrefetchState,
    _prefetch_set: usize,
    intermediates: &PostAttnIntermediates,
    gpu_combine: bool,
    chain_next_norm_off: Option<u64>,
) -> Result<(), LayerForwardError> {
    let v = VARIANT;

    // ── CMD3b: K-expert FFN + combine via slice 9b — async ───────
    //
    // Slice 5d-5 (production fast path, `gpu_combine = true`): pread
    // K expert blobs DIRECTLY into `moe.data[slot]`'s shared-storage
    // pages, then encode the dispatch with GPU buffer refs for the
    // post-attn-norm/residual/shared-out inputs. Saves ~7 MB / layer
    // of host memcpy (the intermediate `expert_data: Vec<u8>` is
    // gone) on top of slice 5d-4's ~2 MB / layer of input
    // round-tripping.
    //
    // The slot-reuse pattern is sound: each layer's K-expert dispatch
    // is waited at the top of the next layer's
    // `complete_deferred_experts_into`, so `moe.data[slot]` is GPU-
    // quiescent by the time this layer preads new bytes into it.
    //
    // CPU-combine fallback path (`gpu_combine = false`) still routes
    // through the host-slice variant — `DeferredMode::Cpu` needs host
    // snapshots of `h_mid` / `shared_out` for the finalize pass.
    let indices = &intermediates.routing_indices;
    let weights = &intermediates.routing_weights;
    let shared_gate_score = intermediates.shared_gate_score;
    let k = indices.len();

    if gpu_combine {
        // Direct mmap path: GPU reads expert weights from mmap'd Metal
        // buffers. No pread, no prefetch, no staging.
        let bindings: Vec<(&metal::Buffer, u64)> = indices.iter()
            .map(|&idx| {
                expert_files
                    .mmap_buffer_for_expert(layer_idx, idx as u32)
                    .expect("mmap buffer missing for expert")
            })
            .collect();

        let chain_rms_pipes = if chain_next_norm_off.is_some() {
            Some(crate::riir::backend::gpu::gpu_norm::RmsNormBf16Pipelines {
                sum: metal.pipeline("rms_norm_sum_sq")?.clone(),
                apply: metal.pipeline("rms_norm_apply_bf16")?.clone(),
            })
        } else {
            None
        };
        let chain = chain_next_norm_off.and_then(|off| {
            chain_rms_pipes.as_ref().map(|pipes| ChainToNormed {
                pipes,
                wf_buf: wf_buf.buffer(),
                next_norm_off: off,
                combine_out: buffer_pool.handle(buffers.input),
                chain_sum_sq: buffer_pool.handle(buffers.sum_sq),
                chain_normed: buffer_pool.handle(buffers.normed),
                eps: RMS_NORM_EPS,
            })
        });
        gpu_batched_experts_begin_mmap(
            metal,
            moe,
            buffer_pool,
            deferred,
            k as i32,
            buffer_pool.handle(buffers.normed),
            buffer_pool.handle(buffers.h_mid),
            buffer_pool.handle(buffers.shared_out),
            weights,
            shared_gate_score,
            layer_idx as i32,
            &bindings,
            chain,
        )?;
    } else {
        let expert_size = v.expert_size_4bit();
        let mut expert_data = vec![0u8; k * expert_size];
        for slot in 0..k {
            let expert_idx = indices[slot] as usize;
            let dst = &mut expert_data
                [slot * expert_size..(slot + 1) * expert_size];
            expert_files.read_expert(layer_idx, expert_idx, dst)?;
        }
        let h_mid_host = read_buffer_to_vec(buffer_pool.handle(buffers.h_mid), v.hidden_dim);
        let shared_out_host =
            read_buffer_to_vec(buffer_pool.handle(buffers.shared_out), v.hidden_dim);
        let normed_host = read_buffer_to_vec(buffer_pool.handle(buffers.normed), v.hidden_dim);
        let payload = ExpertPayload {
            h_post: &normed_host,
            h_mid: &h_mid_host,
            shared_out: &shared_out_host,
            expert_weights: weights,
            shared_gate_score,
        };
        gpu_batched_experts_begin(
            metal,
            moe,
            buffer_pool,
            deferred,
            k as i32,
            &expert_data,
            payload,
            layer_idx as i32,
            /* gpu_combine = */ false,
        )?;
    }

    Ok(())
}


/// Copy `len` f32s from a shared-storage Metal buffer into a fresh
/// `Vec`. Used by the layer-forward dump path and full-attn host
/// staging where the persistent buffers are bare `metal::Buffer`s
/// (not [`MtlBuffer<f32>`](crate::riir::backend::gpu::metal::MtlBuffer)). Direct
/// counterpart to [`MtlBuffer::to_vec`](crate::riir::backend::gpu::metal::MtlBuffer::to_vec)
/// for the unwrapped-buffer case.
///
/// # Safety
///
/// Caller must ensure no GPU command buffer writing to `b` is in
/// flight. Typical discipline: a `wait_until_completed` on the most
/// recent dispatch, or a `complete_deferred_experts_into` drain
/// before the read.
pub(in crate::riir) fn read_buffer_to_vec(b: &Buffer, len: usize) -> Vec<f32> {
    let ptr = b.contents() as *const f32;
    // SAFETY: see fn docs.
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}

#[allow(clippy::too_many_arguments)]
fn encode_rms_norm_pair(
    cmdbuf: &CommandBufferRef,
    sum_pipe: &ComputePipelineState,
    apply_pipe: &ComputePipelineState,
    input: &Buffer,
    weight_buf: &Buffer,
    weight_off: u64,
    output: &Buffer,
    sum_sq: &Buffer,
    dim: u32,
) {
    {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(sum_pipe);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(sum_sq), 0);
        enc.set_bytes(2, 4, (&dim as *const u32).cast());
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
    {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(apply_pipe);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(weight_buf), weight_off as NSUInteger);
        enc.set_buffer(2, Some(sum_sq), 0);
        enc.set_buffer(3, Some(output), 0);
        let eps = RMS_NORM_EPS;
        enc.set_bytes(4, 4, (&dim as *const u32).cast());
        enc.set_bytes(5, 4, (&eps as *const f32).cast());
        let num_tgs = (dim + 255) / 256;
        enc.dispatch_thread_groups(
            MTLSize::new(num_tgs as NSUInteger, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
}

/// One `swiglu_fused` dispatch into a fresh encoder. Mirrors the
/// shared-expert-FFN swiglu (`infer.m` `cpu_swiglu` at the production
/// path's `infer.m:2977`); replaces the CPU loop between the
/// shared `gate`/`up` matvecs and `shared_down`. Same kernel the
/// K-expert FFN uses (slice 9a — bit-exact per-PSO).
fn encode_swiglu_buf(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    gate: &Buffer,
    up: &Buffer,
    act: &Buffer,
    dim: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(gate), 0);
    enc.set_buffer(1, Some(up), 0);
    enc.set_buffer(2, Some(act), 0);
    enc.set_bytes(3, 4, (&dim as *const u32).cast());
    let num_tgs = (dim + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

fn encode_residual_add(
    cmdbuf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    dim: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(a), 0);
    enc.set_buffer(1, Some(b), 0);
    enc.set_buffer(2, Some(out), 0);
    enc.set_bytes(3, 4, (&dim as *const u32).cast());
    let num_tgs = (dim + 255) / 256;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

/// Batched-prefill linear-attention forward — session-5 Phase 1.
///
/// Sibling of [`crate::riir::attn::full_attn_forward::batched_full_attn_layer_forward`]
/// for the linear-attn layers in the qwen3_5_moe family. Replaces the
/// per-token fallback inside [`crate::riir::step_internal_batched_gqa`] that
/// the post-session-4 profile flagged as 55% of prefill inclusive time
/// (vs. 9.9% for the batched full-attn path).
///
/// ## Phases
///
/// 1. **Per-token input rms_norm** (Phase 1a, host bounce — mirrors
///    full-attn's Phase 1a).
/// 2. **Batched projections** (Phase 1b): qkv / z / beta / alpha via
///    four `encode_matvec_n_tokens` calls into per-projection stacks.
/// 3. **Per-token recurrent kernels in one cmdbuf** (Phase 1c): N×5
///    dispatches (conv1d_step → rms_norm_qk → compute_decay_beta →
///    delta_net_step → gated_rms_norm) encoded into a single command
///    buffer. The recurrence buffers (`conv_state`, `delta_state`)
///    serialise on Metal encoder order within the cmdbuf, so state[t]
///    is visible to state[t+1] without a per-token commit. **This is
///    the win** — eliminates the per-token commit_and_wait dominating
///    the linear_attn_layer_forward profile.
/// 4. **Batched o_proj** (Phase 1d): one `encode_matvec_n_tokens` from
///    `value_out_stack` (gated_rms_norm output) → `o_proj_stack`.
/// 5. **Per-token post-attn tail** (Phase 1e): residual_add +
///    post-attn rms_norm + gate logits + shared-expert gate scalar +
///    CPU MoE routing. Same
///    pattern as full-attn's Phase 3c.
/// 6. **Batched shared FFN** (Phase 1f): three `encode_matvec_n_tokens`
///    + flat-dispatch `swiglu_fused` over `h_post_stack`. Mirrors
///    full-attn's Phase 3d exactly.
/// 7. **Batched MoE permute-fuse** (Phase 1g): one
///    `encode_moe_batched_permute_fuse` over the joint N×k_active
///    routing CSR. Mirrors full-attn's Phase 3e.
/// 8. **CPU combine** (Phase 1h): per-token
///    `h_mid + moe_sum + sigmoid(shared_gate) * shared_out`.
///
/// Recurrence state (`buffers.conv_state[layer_idx]`,
/// `buffers.delta_state[layer_idx]`) is advanced by N steps. The
/// 1-token-sized scratch buffers (`conv_output`, `delta_g_decay`,
/// `delta_beta`, `delta_output`) are reused per-token within the
/// Phase 1c cmdbuf — write-then-read serialises on encoder order.
///
/// `layer_state` is the host-side recurrence mirror. Not consumed
/// today (parity with [`linear_attn_layer_forward`]'s `_layer_state`),
/// kept on the signature for symmetry with the per-token oracle.
#[allow(clippy::too_many_arguments)]
pub(in crate::riir) fn batched_linear_attn_layer_forward<B>(
    backend: &mut B,
    wf: &WeightFile,
    layer_cache: &LayerWeightCache,
    buffers: &LayerForwardBuffers,
    layer_idx: usize,
    n_tokens: usize,
    k_active: usize,
    expert_files: &ExpertFiles,
    moe_buffers: &mut crate::riir::moe::expert_forward::MoeBuffers,
    _layer_state: &mut LinearAttnState,
    // Session-5 Phase 3: when `Some`, the caller has fired
    // `prefetch.dispatch` for this layer and wants the bucket-vs-
    // prediction set match + record_outcome / record_actual. Only
    // enabled by `step_internal_batched_gqa` at N == 1 (decode
    // re-routed eval_token shape) and only in pread mode. Moved into
    // `moe_block_forward` unchanged.
    prefetch: Option<PrefetchEnv<'_>>,
    // S10b-1a-ii (session 11): hidden in/out are pool BufIds. The
    // orchestrator owns the alternating double-buffer pair and passes
    // ids per layer; this producer resolves to `&metal::Buffer` via
    // `buffer_pool.handle(...)` at the top of the body.
    hidden_in_id: BufId<HiddenBuf>,
    hidden_out_id: BufId<HiddenBuf>,
    // S10b-2: run-lifetime scratch BufIds for the batched graph,
    // allocated once at max chunk width. Replaces the per-layer
    // `pool.alloc` of the pre-MoE transients. `scratch` holds the
    // linear-attn `graph1` transients; `moe` is the shared MoE-block
    // scratch (boundary buffers + `graph2` working set).
    scratch: &LinearAttnGraphScratch,
    moe: &MoeGraphScratch,
) -> Result<(), LayerForwardError>
where
    B: Backend,
    LayerForwardError: From<B::Error>,
    LayerForwardError: From<<B::Pool as BufferPool>::Error>,
{
    use crate::riir::moe::expert_forward::MAX_K;

    // S10b-1b (session 11): no parts_mut destructure here — Phase 2's
    // graph build owns its own pool borrow inside an inner scope, then
    // re-borrows after `backend.execute(&graph)` for the imperative
    // MoE block (1f-1h).

    let v = VARIANT;
    debug_assert!(k_active <= MAX_K);

    let linear_layer_idx = linear_layer_idx_for(layer_idx).ok_or(
        LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "linear_layer_idx (batched called on full-attn layer)",
        },
    )?;

    let hidden_dim = v.hidden_dim;
    let conv_dim = v.linear_conv_dim();
    let total_value = v.linear_total_value();
    let num_v_heads = v.linear_num_v_heads;

    // Per-tensor bit widths. Same lookups as the per-token path.
    let qkv_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight"),
    );
    let z_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.linear_attn.in_proj_z.weight"),
    );
    let alpha_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.linear_attn.in_proj_a.weight"),
    );
    let beta_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.linear_attn.in_proj_b.weight"),
    );
    let o_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.linear_attn.out_proj.weight"),
    );

    let attn = layer_cache.attn.linear().ok_or(
        LayerForwardError::MissingTensor {
            layer: layer_idx,
            tensor: "linear_attn weights (batched called on full-attn layer)",
        },
    )?;
    let qkv_w = attn.qkv_w;
    let qkv_s = attn.qkv_s;
    let qkv_b = attn.qkv_b;
    let z_w = attn.z_w;
    let z_s = attn.z_s;
    let z_b = attn.z_b;
    let beta_w = attn.beta_w;
    let beta_s = attn.beta_s;
    let beta_b = attn.beta_b;
    let alpha_w = attn.alpha_w;
    let alpha_s = attn.alpha_s;
    let alpha_b = attn.alpha_b;
    let conv1d_w = attn.conv1d_w;
    let a_log = attn.a_log;
    let dt_bias = attn.dt_bias;
    let gnorm_w = attn.gated_norm_w;
    let o_w = attn.o_proj_w;
    let o_s = attn.o_proj_s;
    let o_b = attn.o_proj_b;

    // ── Phase 1a–1e: pre-MoE chain via the Graph compiler. ───────
    // S10b-1b (session 11): the imperative encode_X_into chain that
    // S7-1a (session 6) fused into one cmdbuf is now expressed as a
    // [`crate::riir::backend::Graph`]. Each former encoder call becomes one
    // [`crate::riir::backend::Op`] push; `MetalBackend::execute` walks the op
    // list and records them into a single cmdbuf, preserving the
    // commit-fusion win. The recurrent `*NTokens` Ops loop internally
    // over n_tokens — they expect *stacked* transient buffers
    // (n_tokens × per_token_size), not the single-token scratch the
    // imperative loop reused.
    use crate::riir::backend::{Graph, Op, WeightRef};

    // Hoist bits lookups for gate + shared_expert_gate so we can build
    // the WeightRefs at graph-build time.
    let gate_bits =
        bits_of(wf, &format!("model.layers.{layer_idx}.mlp.gate.weight"));
    let seg_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert_gate.weight"
        ),
    );

    let value_dim = Variant::LINEAR_VALUE_DIM as u32;
    let k_heads_per_v =
        (v.linear_num_v_heads / v.linear_num_k_heads) as u32;
    let key_offset_per_token =
        (v.linear_num_k_heads * Variant::LINEAR_KEY_DIM) as u32;

    // Build the graph + allocate transient pool BufIds in an inner
    // scope so the `pool` borrow ends before `backend.execute(&graph)`.
    // BufIds for buffers consumed by the imperative MoE block (1f-1h)
    // and the host-side routing readback are returned out.
    let graph = {
        let mut g = Graph::new();

        // S10b-2: transient BufIds come from the run-lifetime
        // `LinearAttnGraphScratch`; the graph1→MoE boundary BufIds from
        // the shared `MoeGraphScratch` (both allocated once at max
        // chunk width) instead of a fresh per-layer `pool.alloc`.
        // Same local names so the Op pushes below are unchanged; the
        // `*NTokens` Ops stride by the real `n_tokens` and the
        // scratch buffers' max-chunk tail is simply unused.
        let normed_id = scratch.normed;
        let qkv_stack_id = scratch.qkv_stack;
        let z_stack_id = scratch.z_stack;
        let beta_stack_id = scratch.beta_stack;
        let alpha_stack_id = scratch.alpha_stack;
        let conv_out_stack_id = scratch.conv_out_stack;
        let g_decay_stack_id = scratch.g_decay_stack;
        let beta_gate_stack_id = scratch.beta_gate_stack;
        let delta_out_stack_id = scratch.delta_out_stack;
        let value_out_stack_id = scratch.value_out_stack;
        let o_proj_stack_id = scratch.o_proj_stack;
        let gate_logits_id = scratch.gate_logits;
        let h_mid_id = moe.h_mid;
        let h_post_id = moe.h_post;
        let shared_gate_id = moe.shared_gate;
        let routing_indices_id = moe.routing_indices;
        let routing_weights_id = moe.routing_weights;

        // Phase 1a — input rms_norm.
        g.push(Op::RmsNormBf16NTokens {
            label: "linear_attn.input_norm",
            // HiddenBuf → RmsNormIn via the union impl.
            x: hidden_in_id.into(),
            weight_off: layer_cache.input_layernorm_w,
            out: normed_id.into(),
            dim: hidden_dim as u32,
            n_tokens: n_tokens as u32,
            eps: RMS_NORM_EPS,
        });

        // Phase 1b — 4 batched projections (qkv / z / beta / alpha).
        g.push(Op::MatvecNTokens {
            label: "linear_attn.qkv_proj",
            weight: WeightRef { w_off: qkv_w, s_off: qkv_s, b_off: qkv_b, bits: qkv_bits },
            input: normed_id.into(),
            input_off: 0,
            output: qkv_stack_id.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: conv_dim as u32,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::MatvecNTokens {
            label: "linear_attn.z_proj",
            weight: WeightRef { w_off: z_w, s_off: z_s, b_off: z_b, bits: z_bits },
            input: normed_id.into(),
            input_off: 0,
            output: z_stack_id.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: total_value as u32,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::MatvecNTokens {
            label: "linear_attn.beta_proj",
            weight: WeightRef { w_off: beta_w, s_off: beta_s, b_off: beta_b, bits: beta_bits },
            input: normed_id.into(),
            input_off: 0,
            output: beta_stack_id.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: num_v_heads as u32,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::MatvecNTokens {
            label: "linear_attn.alpha_proj",
            weight: WeightRef { w_off: alpha_w, s_off: alpha_s, b_off: alpha_b, bits: alpha_bits },
            input: normed_id.into(),
            input_off: 0,
            output: alpha_stack_id.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: num_v_heads as u32,
            n_tokens: n_tokens as u32,
        });

        // Phase 1c — recurrent loop. Each *NTokens Op below loops
        // internally over n_tokens. The Rust per-token loop is gone.
        g.push(Op::Conv1dStepNTokens {
            label: "linear_attn.conv1d_step",
            qkv_in: qkv_stack_id,
            conv_state: buffers.conv_state[linear_layer_idx],
            weight_off: conv1d_w,
            conv_out: conv_out_stack_id,
            conv_dim: conv_dim as u32,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::RmsNormQkNTokens {
            label: "linear_attn.rms_norm_qk",
            x: conv_out_stack_id,
            num_k_heads: v.linear_num_k_heads as u32,
            key_dim: Variant::LINEAR_KEY_DIM as u32,
            key_offset_per_token,
            // conv_out is `q | k | v` per token: stride = conv_dim.
            per_token_total: conv_dim as u32,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::ComputeDecayBetaNTokens {
            label: "linear_attn.compute_decay_beta",
            alpha_in: alpha_stack_id,
            beta_in: beta_stack_id,
            a_log_off: a_log,
            dt_bias_off: dt_bias,
            g_decay_out: g_decay_stack_id,
            beta_gate_out: beta_gate_stack_id,
            num_v_heads: num_v_heads as u32,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::GatedDeltaNetChunkwise {
            label: "linear_attn.gated_delta_net_step",
            state: buffers.delta_state[linear_layer_idx],
            conv_out: conv_out_stack_id,
            g_decay: g_decay_stack_id,
            beta_gate: beta_gate_stack_id,
            output: delta_out_stack_id,
            num_v_heads: num_v_heads as u32,
            value_dim,
            k_heads_per_v,
            n_tokens: n_tokens as u32,
            chunk_size: 16,
        });
        g.push(Op::GatedRmsNormNTokens {
            label: "linear_attn.gated_rms_norm",
            values: delta_out_stack_id,
            z: z_stack_id,
            weight_off: gnorm_w,
            output: value_out_stack_id,
            num_v_heads: num_v_heads as u32,
            value_dim,
            n_tokens: n_tokens as u32,
            eps: RMS_NORM_EPS,
        });

        // Phase 1d — o_proj.
        g.push(Op::MatvecNTokens {
            label: "linear_attn.o_proj",
            weight: WeightRef { w_off: o_w, s_off: o_s, b_off: o_b, bits: o_bits },
            input: value_out_stack_id.into(),
            input_off: 0,
            output: o_proj_stack_id.into(),
            output_off: 0,
            in_dim: total_value as u32,
            out_dim: hidden_dim as u32,
            n_tokens: n_tokens as u32,
        });

        // Phase 1e — residual + post-norm + gate matvec + shared-gate
        // matvec + GPU MoE router (softmax-topK + normalize).
        g.push(Op::ResidualAddNTokens {
            label: "linear_attn.residual_add",
            a: o_proj_stack_id,
            // HiddenBuf → RmsNormIn via the union impl.
            b: hidden_in_id.into(),
            out: h_mid_id,
            n_tokens: n_tokens as u32,
            dim: hidden_dim as u32,
        });
        g.push(Op::RmsNormBf16NTokens {
            label: "linear_attn.post_attn_norm",
            // ResidualBuf → RmsNormIn; MoeInputBuf → RmsNormOut.
            x: h_mid_id.into(),
            weight_off: layer_cache.post_attention_layernorm_w,
            out: h_post_id.into(),
            dim: hidden_dim as u32,
            n_tokens: n_tokens as u32,
            eps: RMS_NORM_EPS,
        });
        g.push(Op::MatvecNTokens {
            label: "linear_attn.gate_router",
            weight: WeightRef {
                w_off: layer_cache.gate.w,
                s_off: layer_cache.gate.s,
                b_off: layer_cache.gate.b,
                bits: gate_bits,
            },
            input: h_post_id.into(),
            input_off: 0,
            output: gate_logits_id.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: v.num_experts as u32,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::MatvecNTokens {
            label: "linear_attn.shared_gate",
            weight: WeightRef {
                w_off: layer_cache.shared.seg_w,
                s_off: layer_cache.shared.seg_s,
                b_off: layer_cache.shared.seg_b,
                bits: seg_bits,
            },
            input: h_post_id.into(),
            input_off: 0,
            output: shared_gate_id.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: 1,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::MoeSoftmaxTopK {
            label: "linear_attn.router_softmax_topk",
            logits: gate_logits_id,
            indices_out: routing_indices_id,
            weights_out: routing_weights_id,
            n_tokens: n_tokens as u32,
            n_experts: v.num_experts as u32,
            k: k_active as u32,
        });
        g.push(Op::MoeNormalizeWeights {
            label: "linear_attn.router_normalize",
            weights: routing_weights_id,
            n_tokens: n_tokens as u32,
            k: k_active as u32,
        });

        g
    };

    // S10b-2 Phase 4: on the first call, lifetime-color graph1's
    // transient BufIds (and pin them). Topology is layer- and
    // step-invariant, so one pass holds for the whole run. graph2 has
    // its own latch on `moe` (prefill-arc Phase 3 split the latches).
    if !scratch.commit_planned.get() {
        backend.pool_mut().commit_plan(&graph);
        scratch.commit_planned.set(true);
    }
    // Submit + wait. Single cmdbuf, single commit (the S7-1a fusion
    // is preserved by `MetalBackend::execute`'s default loop body).
    backend.execute(&graph, "graph_linear_attn")?;

    // Prefill-arc Phase 3b: host readback → CPU bucket build → expert
    // staging → graph2 are the shared `moe_block_forward`, driven
    // identically by the full-attn producer.
    moe_block_forward(
        backend,
        moe,
        wf,
        layer_cache,
        layer_idx,
        n_tokens,
        k_active,
        expert_files,
        moe_buffers,
        prefetch,
        hidden_out_id,
    )
}

/// Run the shared MoE block — the tail half of every batched layer
/// forward, after the attention `graph1` has produced its outputs into
/// the [`MoeGraphScratch`] boundary buffers (`h_mid`, `h_post`,
/// `shared_gate`, `routing_indices`, `routing_weights`).
///
/// Identical for linear-attn and full-attn layers: host-reads the
/// router output, builds the expert buckets on the CPU, stages expert
/// weights (pread) or points at the mmap layer, then runs `graph2` —
/// shared FFN + MoE permute-fuse + combine into `hidden_out_id`. The
/// `graph2` `commit_plan` is gated by `moe.commit_planned` (its
/// topology is identical regardless of which attention kind produced
/// the inputs, so one pass holds for the whole run).
pub(in crate::riir) fn moe_block_forward<B>(
    backend: &mut B,
    moe: &MoeGraphScratch,
    wf: &WeightFile,
    layer_cache: &LayerWeightCache,
    layer_idx: usize,
    n_tokens: usize,
    k_active: usize,
    expert_files: &ExpertFiles,
    moe_buffers: &mut crate::riir::moe::expert_forward::MoeBuffers,
    mut prefetch: Option<PrefetchEnv<'_>>,
    hidden_out_id: BufId<HiddenBuf>,
) -> Result<(), LayerForwardError>
where
    B: Backend,
    LayerForwardError: From<B::Error>,
    LayerForwardError: From<<B::Pool as BufferPool>::Error>,
{
    use crate::riir::backend::{Graph, Op, WeightRef};
    use crate::riir::moe::moe_router::build_expert_buckets;

    let v = VARIANT;
    let hidden_dim = v.hidden_dim;
    let f32_sz_u = std::mem::size_of::<f32>();
    // Boundary BufIds produced by the attention `graph1`.
    let h_mid_id = moe.h_mid;
    let h_post_id = moe.h_post;
    let shared_gate_id = moe.shared_gate;
    let routing_indices_id = moe.routing_indices;
    let routing_weights_id = moe.routing_weights;

    // Host readback at the graph1 → MoE-graph split: h_post (for the
    // bucket_input gather) + routing indices/weights (for
    // build_expert_buckets). h_mid / shared_gate stay on GPU.
    let mut h_post_stack = vec![0.0f32; n_tokens * hidden_dim];
    let mut all_routing_indices = vec![0i32; n_tokens * k_active];
    let mut all_routing_weights = vec![0.0f32; n_tokens * k_active];
    {
        let pool = backend.pool();
        pool.download(h_post_id, unsafe {
            std::slice::from_raw_parts_mut(
                h_post_stack.as_mut_ptr() as *mut u8,
                n_tokens * hidden_dim * f32_sz_u,
            )
        })?;
        pool.download(routing_indices_id, unsafe {
            std::slice::from_raw_parts_mut(
                all_routing_indices.as_mut_ptr() as *mut u8,
                n_tokens * k_active * std::mem::size_of::<i32>(),
            )
        })?;
        pool.download(routing_weights_id, unsafe {
            std::slice::from_raw_parts_mut(
                all_routing_weights.as_mut_ptr() as *mut u8,
                n_tokens * k_active * f32_sz_u,
            )
        })?;
    }

    // ── Phases 1f-1h as graph2: shared FFN + MoE permute-fuse +
    //    combine. The shared-expert weight-file bit widths feed the
    //    matvec WeightRefs.
    let s_gate_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight"
        ),
    );
    let s_up_bits = bits_of(
        wf,
        &format!("model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight"),
    );
    let s_down_bits = bits_of(
        wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight"
        ),
    );

    // CPU bucket prep — needs the routing readback above.
    let buckets = build_expert_buckets(
        &all_routing_indices,
        &all_routing_weights,
        n_tokens,
        k_active,
        v.num_experts,
    );
    let total_assignments = buckets.token_idx.len();
    debug_assert_eq!(total_assignments, n_tokens * k_active);

    let num_buckets = buckets.expert_ids.len();

    // Optional htpe (hits-tokens-per-expert) distribution dump. Gated
    // by `MOEFLUX_LOG_HTPE=1`. One stderr line per layer per chunk,
    // 128 comma-separated counts. Used to decide the gather→per-expert
    // dense pivot (see `gather_qmm_arch_pivot_plan.md`): the pivot
    // win depends on the routing distribution, not just the mean.
    if std::env::var_os("MOEFLUX_LOG_HTPE").is_some() {
        let mut counts = vec![0u32; v.num_experts];
        for bi in 0..num_buckets {
            let e = buckets.expert_ids[bi] as usize;
            counts[e] = buckets.offsets[bi + 1] - buckets.offsets[bi];
        }
        let mut line = String::with_capacity(v.num_experts * 5);
        for (i, &c) in counts.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            line.push_str(&c.to_string());
        }
        eprintln!(
            "HTPE layer={layer_idx} n_tokens={n_tokens} k_active={k_active} \
             num_experts={} counts=[{line}]",
            v.num_experts,
        );
    }

    // 2026-05-20 pread teardown: the staging-blob path (per-bucket
    // `read_expert` + `pool.upload_at`) was deleted. Expert weights
    // are read by the gather GEMM straight from the layer's mmap'd
    // buffer — zero copy, OS page cache manages the working set.
    // The `prefetch` env is also unused on the batched path
    // (the orchestrator never fires prefetch dispatch here); the
    // `Option<PrefetchEnv>` arg stays in the signature for the
    // per-token oracle caller but is always `None` from prefill.
    // Both `prefetch` and `moe_buffers` will become removable
    // once task-7 confirms decode prefetch isn't load-bearing.
    let _ = prefetch;
    let _ = moe_buffers;
    let expert_base_id: BufId<ExpertBaseBuf> = expert_files
        .mmap_id_for_expert(layer_idx, 0)
        .expect("mmap layer present (expert I/O is always mmap)")
        .0;

    // `expert_slots[bi]` is the block index of `buckets.expert_ids[bi]`
    // within the layer's mmap buffer (`num_experts` blocks at
    // `expert_size` stride). `expert_indices` expands it per
    // assignment row for the gather kernel's per-row `indices`.
    let expert_slots: Vec<u32> =
        buckets.expert_ids.iter().map(|&e| e as u32).collect();
    let mut expert_indices_host = vec![0u32; total_assignments];
    for bi in 0..num_buckets {
        let start = buckets.offsets[bi] as usize;
        let end = buckets.offsets[bi + 1] as usize;
        expert_indices_host[start..end].fill(expert_slots[bi]);
    }

    // bucket_input host permute: gather each assignment's token row
    // from h_post into bucket-flat order. Then upload it + the small
    // routing tables into the run-lifetime scratch buffers.
    //
    // `Op::MoeGatherIdFuse` (the new-kernel path) does NOT consume
    // `bucket_input` — its kernel gathers per-(token, slot)
    // activations internally via `hids`. Skipping the host permute
    // + upload here is the load-bearing "no host-side permute" win
    // promised by the new path; running it anyway would inflate the
    // env=on bench by ~tens of ms per layer.
    if !moe_gather_id_enabled() {
        let mut bucket_input_host =
            vec![0.0f32; total_assignments * hidden_dim];
        for assignment_idx in 0..total_assignments {
            let t = buckets.token_idx[assignment_idx] as usize;
            let src =
                &h_post_stack[t * hidden_dim..(t + 1) * hidden_dim];
            let dst_off = assignment_idx * hidden_dim;
            bucket_input_host[dst_off..dst_off + hidden_dim]
                .copy_from_slice(src);
        }
        let pool = backend.pool_mut();
        pool.upload(moe.bucket_input, unsafe {
            std::slice::from_raw_parts(
                bucket_input_host.as_ptr() as *const u8,
                total_assignments * hidden_dim * f32_sz_u,
            )
        })?;
    }
    {
        let pool = backend.pool_mut();
        pool.upload(moe.bucket_token_idx, unsafe {
            std::slice::from_raw_parts(
                buckets.token_idx.as_ptr() as *const u8,
                total_assignments * std::mem::size_of::<i32>(),
            )
        })?;
        pool.upload(moe.bucket_weights, unsafe {
            std::slice::from_raw_parts(
                buckets.weights.as_ptr() as *const u8,
                total_assignments * f32_sz_u,
            )
        })?;
        pool.upload(moe.expert_indices, unsafe {
            std::slice::from_raw_parts(
                expert_indices_host.as_ptr() as *const u8,
                total_assignments * std::mem::size_of::<u32>(),
            )
        })?;
    }

    // graph2: shared FFN (gate/up matvec → swiglu → down matvec),
    // zero the out_sum accumulator, MoE permute-fuse, combine into
    // hidden_out.
    let graph2 = {
        let mut g = Graph::new();
        g.push(Op::MatvecNTokens {
            label: "moe.shared_gate_proj",
            weight: WeightRef {
                w_off: layer_cache.shared.gate_w,
                s_off: layer_cache.shared.gate_s,
                b_off: layer_cache.shared.gate_b,
                bits: s_gate_bits,
            },
            input: h_post_id.into(),
            input_off: 0,
            output: moe.shared_ffn_gate.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: v.shared_intermediate as u32,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::MatvecNTokens {
            label: "moe.shared_up_proj",
            weight: WeightRef {
                w_off: layer_cache.shared.up_w,
                s_off: layer_cache.shared.up_s,
                b_off: layer_cache.shared.up_b,
                bits: s_up_bits,
            },
            input: h_post_id.into(),
            input_off: 0,
            output: moe.shared_up.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: v.shared_intermediate as u32,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::SwigluFusedBatched {
            label: "moe.shared_swiglu",
            gate: moe.shared_ffn_gate,
            up: moe.shared_up,
            out: moe.shared_act,
            total: (n_tokens * v.shared_intermediate) as u32,
        });
        g.push(Op::MatvecNTokens {
            label: "moe.shared_down_proj",
            weight: WeightRef {
                w_off: layer_cache.shared.down_w,
                s_off: layer_cache.shared.down_s,
                b_off: layer_cache.shared.down_b,
                bits: s_down_bits,
            },
            input: moe.shared_act.into(),
            input_off: 0,
            output: moe.shared_down.into(),
            output_off: 0,
            in_dim: v.shared_intermediate as u32,
            out_dim: hidden_dim as u32,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::ZeroBuffer {
            label: "moe.out_sum_zero",
            buf: moe.out_sum,
            n_bytes: (n_tokens * hidden_dim * f32_sz_u) as u32,
        });
        if moe_gather_id_enabled() {
            // New path: one-dispatch MoE matmul (`gather_mm_id.metal`).
            // Reuses the `bucket_gate` / `bucket_up` / `bucket_out`
            // BufIds as `gate_mid` / `up_mid` / `down_mid` scratch
            // via the unidirectional `From<BucketGateBuf> for
            // BufId<GateMidBuf>` (etc.) impls in `buftype.rs`. The
            // `.into()` at the push site is the documentation of
            // the path swap.
            g.push(Op::MoeGatherIdFuse {
                label: "moe.gather_id_fuse",
                expert_base: expert_base_id,
                expert_stride: v.expert_size_4bit() as u64,
                indices: moe.routing_indices,
                weights: moe.routing_weights,
                // h_post_id is BufId<MoeInputBuf>: the post-norm MoE
                // input. This is the load-bearing bug-of-record check —
                // session 19 passed `h_mid` (a BufId<ResidualBuf>) here,
                // and the type system now rejects that swap at compile
                // time (no From<ResidualBuf> for BufId<MoeInputBuf>).
                mlp_in: h_post_id,
                out_sum: moe.out_sum,
                htpe: moe.htpe,
                hids: moe.hids,
                gate_mid: moe.bucket_gate.into(),
                up_mid: moe.bucket_up.into(),
                down_mid: moe.bucket_out.into(),
                n_tokens: n_tokens as u32,
                n_experts: v.num_experts as u32,
                k: k_active as u32,
            });
        } else {
            g.push(Op::MoeBatchedPermuteFuse {
                label: "moe.permute_fuse",
                expert_base: expert_base_id,
                expert_stride: v.expert_size_4bit() as u64,
                expert_indices: moe.expert_indices,
                expert_slots,
                bucket_input: moe.bucket_input,
                bucket_gate: moe.bucket_gate,
                bucket_up: moe.bucket_up,
                bucket_act: moe.bucket_act,
                bucket_out: moe.bucket_out,
                bucket_token_idx: moe.bucket_token_idx,
                bucket_weights: moe.bucket_weights,
                out_sum: moe.out_sum,
                buckets,
            });
        }
        g.push(Op::MoeCombineResidualNTokens {
            label: "moe.combine",
            h_mid: h_mid_id,
            moe_sum: moe.out_sum,
            shared_out: moe.shared_down,
            shared_gate: shared_gate_id,
            hidden_out: hidden_out_id,
            n_tokens: n_tokens as u32,
            dim: hidden_dim as u32,
        });
        g
    };
    // S10b-2 Phase 4: color graph2's transients on the first call,
    // then latch — `commit_plan` runs exactly once per run.
    if !moe.commit_planned.get() {
        backend.pool_mut().commit_plan(&graph2);
        moe.commit_planned.set(true);
    }
    backend.execute(&graph2, "graph_moe")?;

    // Phase 3: record this layer's actual routing as the prediction
    // for the next token's same layer. Only meaningful at N=1
    // (eval_token shape) — at larger N the layer routes many tokens
    // and the per-token prediction semantic is murky. The caller
    // gates the env to `None` for N != 1.
    if let Some(pe) = prefetch.as_mut() {
        use crate::riir::moe::expert_forward::MAX_K;
        let mut actuals: [i32; MAX_K] = [0; MAX_K];
        let len = k_active.min(MAX_K).min(all_routing_indices.len());
        actuals[..len].copy_from_slice(&all_routing_indices[..len]);
        pe.prefetch.record_actual(layer_idx, actuals);
    }

    Ok(())
}
