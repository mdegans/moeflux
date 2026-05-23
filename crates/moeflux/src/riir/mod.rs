//! Pure-Rust port of moeflux's host-side dispatch.
//!
//! This module is the sole host-side path: the C/Objective-C
//! reference implementation it was ported from has been retired.
//! Correctness is held by the Rust-internal differential oracles in
//! `tests/graph_diff_oracle.rs` and `tests/batched_diff_oracle.rs`
//! (CPU backend vs. Metal backend, and batched vs. per-token).
//!
//! # Status
//!
//! Phase 3 (in progress): bottom-up kernel ports. [`RsCtx::open`]
//! loads only what the kernels ported so far need — currently the
//! `WeightFile` for embedding lookup. Methods that depend on
//! unported kernels still panic with `todo!()`; each kernel landing
//! flips one or more methods to a real impl.
//!
//! See the drama_llama in-repo `riir_moeflux_strategy.md` for the
//! phase breakdown.

#![allow(missing_docs)] // Phase 3 — types fill in incrementally.

use std::collections::{HashMap, VecDeque};
use std::path::Path;

use ::metal::Buffer;

pub mod attn;
pub mod backend;
pub mod gpu_capture;
pub mod io;
pub mod moe;
pub mod snapshot;
pub mod variants;
pub use moe::deferred::{DeferredError, DeferredRing, DeferredState};
pub use io::embedding::{bf16_to_f32, embed_lookup, EmbeddingError};
pub use moe::expert_forward::{
    gpu_batched_experts_forward, gpu_expert_forward, ExpertForwardError,
    MoeBuffers, MAX_K,
};
pub use io::expert_io::{ExpertFiles, ExpertIoError};
pub use attn::gpu_attn::{
    gpu_attn_scores_batched, gpu_attn_softmax_batched,
    gpu_attn_values_batched, gpu_sigmoid_gate, GpuAttnError,
};
pub use io::gpu_lm_head::{GpuLmHead, GpuLmHeadError};
pub use backend::gpu::gpu_norm::{gpu_rms_norm_fused, GpuNormError};
pub use attn::linear_attn::{
    conv1d_step, gated_delta_recurrence, rms_norm_bare, rms_norm_gated,
    LinearAttnError,
};
// S7-6c: trait scope for `pool.alloc/handle/upload/download/reset_transient`
// in `step_internal_batched_gqa`.
use backend::BufferPool as _;
use backend::{Backend, Graph, MetalBackend, Op, WeightRef};
pub use io::lm_head::{lm_head_cpu, LmHeadError};
pub use backend::gpu::metal::{
    CmdbufStat, MetalContext, MetalError, MtlBuffer,
};
pub use moe::moe_router::{moe_router_cpu, MoeRouterError};
pub use attn::rms_norm::{rms_norm_cpu, rms_norm_per_head_cpu, RmsNormError};
pub use attn::rope::{apply_rotary_emb, RopeError};
pub use io::layer_weight_cache::LayerWeightCache;
// full_attn_layer_forward dropped from pub re-export with the Phase A/B
// refactor — it's now pub(super) inside riir, and the only external
// callers in this crate use it transitively via eval_prompt / eval_token.
use attn::full_attn_forward::full_attn_layer_forward;
pub use attn::linear_attn_forward::{
    linear_attn_layer_forward, linear_layer_idx_for, LayerForwardBuffers,
    LayerForwardError,
};
// Backwards-compat aliases — 4d renamed the buffer struct + error.
#[allow(deprecated)]
pub use attn::linear_attn_forward::{LinearAttnBuffers, LinearAttnForwardError};
pub use io::mtl_weight_buf::{MtlWeightBuf, MtlWeightBufError};
pub use io::prefetch::{PrefetchState, PrefetchStatus, SlotSource};
pub use attn::sdpa::{sdpa_cpu, SdpaError};
pub use snapshot::state::{
    alloc_layer_states, clear_all, pos_max, truncate, KvCache, LayerState,
    LinearAttnState,
};
pub use variants::{Variant, VARIANT};
pub use io::weight_file::{TensorInfo, WeightFile, WeightFileError};

/// Default cap on stored snapshot checkpoints per [`RsCtx`]. Matches
/// Anthropic's API limit on `cache_control` breakpoints per prompt
/// (4) — the cap aligns with the upstream contract by construction,
/// not coincidence.
pub const DEFAULT_MAX_CHECKPOINTS: usize = 4;

/// Errors from [`RsCtx::restore_to`].
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    /// No snapshot exists at the requested position. Caller should
    /// fall back to a full clear + reprefill.
    #[error("no checkpoint stored at position {pos}")]
    NoCheckpoint { pos: i32 },
    /// State-snapshot deserialization failed. By construction this
    /// shouldn't fire (we only restore from buffers we wrote with
    /// `state_save` ourselves) — surface it rather than panic so
    /// callers can log and recover.
    #[error("snapshot reload failed: {0}")]
    Snapshot(#[source] snapshot::state_snapshot::StateSnapshotError),
}

/// Pure-Rust analogue of [`crate::imp::Ctx`]. API surface mirrors the
/// C wrapper 1:1 during the port — the diff harness compares behavior
/// at this boundary.
///
/// # Phase 3 progress
///
/// Init loads the [`WeightFile`] only (enough for embedding lookup).
/// The per-layer state, expert mmaps, and vocab are deferred to
/// Phase 1b / Phase 4 alongside the kernels that actually consume
/// them. Calling an unported method panics with a `todo!()` pinned
/// to the phase that will implement it.
/// Chunked-prefill default for the batched-GQA orchestrator. Hardware-
/// tuned for 96 GB unified RAM + SSD profile per the Phase D plan;
/// Phase F sweep will validate / refine the value. Production callers
/// don't override; tests that need to exercise multi-chunk boundary
/// math at small N use [`set_batched_chunk_size_for_test`].
pub const BATCHED_CHUNK_SIZE: usize = 8192;

std::thread_local! {
    static BATCHED_CHUNK_OVERRIDE: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Set / clear the per-thread chunk-size override for
/// [`RsCtx::step_internal`]. **Test-only.** Production callers must
/// not call this — the only purpose is exercising chunk-boundary
/// arithmetic at small N from integration tests. Pass `None` to
/// restore the default.
pub fn set_batched_chunk_size_for_test(n: Option<usize>) {
    BATCHED_CHUNK_OVERRIDE.with(|c| c.set(n));
}

fn batched_chunk_size() -> usize {
    BATCHED_CHUNK_OVERRIDE
        .with(|c| c.get())
        .unwrap_or(BATCHED_CHUNK_SIZE)
}

/// Session-5 Phase 3: `eval_token` routing backend, read once from
/// `MOEFLUX_EVAL_TOKEN` at first call.
///
/// - `oracle` (default): the historical per-token oracle path with
///   deferred K-expert async pipelining.
/// - `batched`: routes through `step_internal(&[tok], pos, ...)`,
///   exercising the batched orchestrator at N=1 with the Phase 3
///   prefetch state machine.
///
/// The two paths produce identical logits (cosine=1.0); only
/// per-token decode wall-clock differs. See [`RsCtx::eval_token`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvalTokenMode {
    Oracle,
    Batched,
}

fn eval_token_mode() -> EvalTokenMode {
    use std::sync::OnceLock;
    static MODE: OnceLock<EvalTokenMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("MOEFLUX_EVAL_TOKEN").as_deref() {
            Ok("batched") => EvalTokenMode::Batched,
            Ok("oracle") | Err(_) => EvalTokenMode::Oracle,
            Ok(other) => {
                eprintln!(
                    "[eval_token] MOEFLUX_EVAL_TOKEN={other:?} \
                     unrecognised; using `oracle`. Valid values: \
                     oracle | batched."
                );
                EvalTokenMode::Oracle
            }
        }
    })
}

/// Resolve the full-attention KV cache for `layer_states[idx]`.
///
/// MLA / linear-attention layers route through a separate dispatch
/// (Phase F); reaching the GPU GQA path with one means the dispatch
/// wasn't wired — fail loudly at runtime.
fn full_kv_mut(
    layer_states: &mut [LayerState],
    idx: usize,
) -> Result<&mut KvCache, RsError> {
    match &mut layer_states[idx] {
        LayerState::FullAttn(kv) => Ok(kv),
        LayerState::Mla(_) | LayerState::LinearAttn(_) => {
            Err(RsError::EvalFailed)
        }
    }
}

pub struct RsCtx<B: Backend = MetalBackend> {
    wf: WeightFile,
    /// Session 10a — graph-mode backend. Owns
    /// `MetalContext + MtlWeightBuf + MetalBufferPool` (and pre-warmed
    /// pipelines). Replaces the formerly-separate `metal`, `wf_buf`,
    /// `pool` fields. Lazily built by
    /// [`Self::ensure_linear_resources`] / [`Self::ensure_mla_resources`].
    ///
    /// Default type parameter `B = MetalBackend` preserves source
    /// compatibility for external callers that write plain `RsCtx`.
    /// A `CpuBackend` instantiation is the cross-backend diff-oracle
    /// path for synthetic and (future) end-to-end tests.
    backend: Option<B>,
    /// Lazily-built persistent multi-expert + combine buffer set.
    /// Allocated on first [`Self::gpu_batched_experts_forward`] call;
    /// reused thereafter. ~28 MB on A3B.
    moe_buffers: Option<MoeBuffers>,
    /// Per-layer expert-file handles. Opened eagerly at
    /// [`Self::open`] from `experts_dir/packed_experts/`. Missing
    /// files leave the slot empty per the C path's tolerance
    /// semantics.
    experts: ExpertFiles,
    /// Active top-K (`experts_per_tok` from [`Self::open`]). Mirrors
    /// `mf_ctx.K` — the runtime number of experts to route per
    /// token. The variant's `num_experts_per_tok` is an architectural
    /// MAX; this is the user-selected active value (typically
    /// smaller, e.g. 4 for the dump-hook test even though A3B's
    /// architectural max is 8).
    k_active: usize,
    /// Per-layer KV / linear-attn recurrence state. One entry per
    /// layer; the variant tag matches the C-side
    /// `(i + 1) % FULL_ATTN_INTERVAL == 0` test. Allocated zeroed at
    /// [`Self::open`]; mutated in place by the forward pass and the
    /// `memory_*` ops.
    layer_states: Vec<LayerState>,
    /// Lazily-built per-layer tensor-offset cache. Per-Ctx (not file-
    /// scope) — the Phase 4b cross-Ctx bug fix.
    layer_caches: Option<Vec<LayerWeightCache>>,
    /// Lazily-built persistent buffer set for the linear-attn forward.
    linear_buffers: Option<LinearAttnBuffers>,
    /// S10b-2: run-lifetime scratch BufIds for the batched linear-attn
    /// `graph1` (allocated once at max chunk width, reused every step /
    /// layer). Built by [`Self::ensure_linear_resources`].
    linear_attn_graph_scratch:
        Option<attn::linear_attn_forward::LinearAttnGraphScratch>,
    /// Prefill arc Phase 2 — run-lifetime scratch for the batched
    /// full-attn `graph1` (allocated once at max chunk width). Built
    /// by [`Self::ensure_linear_resources`] alongside
    /// `linear_attn_graph_scratch`.
    full_attn_graph_scratch:
        Option<attn::full_attn_forward::FullAttnGraphScratch>,
    /// Prefill arc Phase 3 — shared run-lifetime scratch for the MoE
    /// block (`graph2`): graph1→MoE boundary buffers + the shared-FFN /
    /// permute-fuse working set. One instance, reused by every layer's
    /// MoE block regardless of attention kind.
    moe_graph_scratch: Option<attn::linear_attn_forward::MoeGraphScratch>,
    /// Prefill arc Phase 3 — the cross-layer hidden double-buffer,
    /// lifted out of the per-layer attention scratch so it is owned at
    /// run scope. The orchestrator swaps the pair between layers.
    hidden_double_buffer:
        Option<attn::linear_attn_forward::HiddenDoubleBuffer>,
    /// Prefill arc Phase 4 — run-lifetime scratch for the orchestrator
    /// head (GPU embedding gather) and tail (GPU final norm + lm_head).
    /// Built by [`Self::ensure_linear_resources`].
    head_tail_scratch:
        Option<attn::linear_attn_forward::HeadTailScratch>,
    /// Slice 4e — pending deferred-experts state. Originally a
    /// single `Option<DeferredState>` (one in-flight K-expert dispatch
    /// at a time); slice 5d-9 widened it to a depth-2
    /// [`DeferredRing`] so layer N+1's CMD1 can be submitted while
    /// layer N's K-expert is still running on the GPU. Drains by
    /// `complete_deferred_experts_*` (pop oldest) or
    /// `discard_deferred_experts_*` (drain all). C-side analogue is
    /// the file-scope `g_deferred` global; lifetime-binding to
    /// `RsCtx` here is what eliminates the cross-Ctx NaN bug class
    /// (see [`deferred`] module docs).
    deferred: DeferredRing,
    /// Persistent GPU LM head dispatcher. Lazily built by
    /// [`Self::ensure_linear_resources`] alongside the other GPU
    /// resources. Replaces the per-token `lm_head_cpu` call (which
    /// dominated the 2026-04-27 perf profile at 59% of CPU time).
    lm_head_gpu: Option<GpuLmHead>,
    /// Slice 5d-6 — work-stealing thread pool for parallel K-expert
    /// pread (8 workers on M2 Max P-cores). Eagerly built at
    /// [`Self::open`] so the per-token hot path can `pool.install` /
    /// `pool.spawn` without paying init cost. C analogue is
    /// `g_io_pool` (4 pthreads); we use 8 since M2 Max has 8 P-cores
    /// and there's no contention with other moeflux work.
    io_pool: rayon::ThreadPool,
    /// Slice 5d-6b — speculative-prefetch state machine. One entry
    /// per layer of last-token K indices (used as next-token same-
    /// layer prediction) plus an in-flight async pread handle.
    /// Drained at `memory_clear`, `state_save`, `state_load`, and
    /// `Drop`. See [`prefetch`] module docs for the soundness
    /// argument.
    prefetch: PrefetchState,
    /// Phase 7 — checkpoint snapshots keyed by sequence position.
    /// Populated by [`Self::checkpoint_pos`] (typically called by
    /// drama_llama's `Session` after prefilling each cache-breakpoint
    /// chunk), consumed by [`Self::restore_to`] on partial-hit cache
    /// reuse. Each value is the byte buffer produced by
    /// [`Self::state_save`] at that position. Drained on
    /// [`Self::memory_clear`] and on every `restore_to(pos)` for keys
    /// `> pos` (their futures are invalidated).
    checkpoints: HashMap<i32, Vec<u8>>,
    /// LRU order for `checkpoints`. Front = oldest, back = newest.
    /// When `checkpoints.len() > max_checkpoints`, evict from the
    /// front while skipping pos-0 (kept for repeat full-prefill
    /// reuse) and the most recently inserted position.
    checkpoint_order: VecDeque<i32>,
    /// Cap on stored snapshots. Defaults to 4 — matches Anthropic's
    /// API limit on `cache_control` breakpoints per prompt, so the
    /// cap aligns with the upstream contract by construction.
    /// Adjustable via [`Self::set_max_checkpoints`].
    max_checkpoints: usize,
    /// MLA per-token GPU scratch (q/k chains, q_prime, v_combine,
    /// out_per_head etc.). Built by `ensure_mla_gpu_resources` on
    /// first GPU MLA eval; reused across every token.
    mla_buffers: Option<attn::mla_attn_forward::MlaForwardBuffers>,
    /// MLA YaRN tables (inv_freq buffer + mscale scalar). Lazy.
    mla_yarn: Option<attn::mla_attn_forward::MlaYarnTables>,
    /// MLA per-kernel compute pipelines (q_prime / sdpa / out_per_head
    /// + matvec / norms / yarn_rope). Lazy.
    mla_pipes: Option<attn::mla_attn_forward::MlaForwardPipelines>,
    /// Phase 3 — Cogito-V2 / DeepSeek-V3 dense-MLP GPU buffers. One set
    /// reused across the `first_k_dense_replace` dense layers. Allocated
    /// only on MLA variants whose `dense_intermediate > 0`.
    dense_mlp_bufs: Option<backend::gpu::dense_mlp_gpu::DenseMlpBuffers>,
    /// Phase 3 — pipelines for the dense MLP forward (matvec + swiglu).
    dense_mlp_pipes: Option<backend::gpu::dense_mlp_gpu::DenseMlpPipelines>,
    /// Cogito-V2 / DeepSeek-V3 GPU shared-expert SwiGLU scratch
    /// (gate_out / up_out / act at `shared_intermediate=2048`). One set
    /// reused across every MoE layer. Allocated lazily in
    /// `ensure_mla_resources`.
    shared_expert_bufs: Option<moe::cogito_moe_gpu::SharedExpertBuffers>,
    /// Phase 2 (cogito-v2 full-GPU): pipelines for the BF16-weight
    /// matvec used by the MoE router gate. Sibling of `dense_mlp_pipes`
    /// — only present on variants whose router gate is BF16 (today:
    /// Cogito-V2 / DeepSeek-V3).
    bf_matvec_pipes: Option<backend::gpu::gpu_matvec::BfMatvecPipelines>,
    /// Phase 5 (cogito-v2 full-GPU): persistent GPU scratch for the
    /// orchestrator's residual + norm stream — keeps `hidden`,
    /// `residual`, `normed`, and `sum_sq` resident across layers so
    /// the per-layer pre/post-norm + residual_add stay on GPU. ~112 KB
    /// total at hidden_dim=7168.
    mla_residual_scratch: Option<backend::gpu::gpu_norm::MlaForwardScratch>,
    /// Phase 5 (cogito-v2 full-GPU): pipelines for the per-layer
    /// rms_norm + residual_add the orchestrator dispatches. Compiled
    /// once on first MLA eval.
    mla_norm_pipes: Option<backend::gpu::gpu_norm::RmsNormBf16Pipelines>,
    /// Phase 5 (cogito-v2 full-GPU): pre-fetched `residual_add`
    /// pipeline for the GPU residual stream. Same kernel the
    /// linear-attn path uses internally.
    residual_add_pipe: Option<::metal::ComputePipelineState>,
    // Future phases populate: vocab.
}

/// Probe a freshly-opened [`WeightFile`] against the compile-time
/// [`VARIANT`], rejecting weights whose shape doesn't match the model
/// this binary was built for.
///
/// A moeflux binary is feature-gated to exactly one model
/// (`moeflux-model-*` → [`VARIANT`]). Pointing it at a different
/// model's weights otherwise loads cleanly, logs `session_ready`, and
/// then panics deep in prefill — an opaque failure that cost a real
/// debugging detour. This turns it into a descriptive
/// [`RsError::ModelMismatch`] at open time.
///
/// Extracts the two robust signals from the manifest and hands the
/// decision to [`check_variant_dims`] (the unit-tested half).
fn probe_variant_match(wf: &WeightFile) -> Result<(), RsError> {
    // Layer count: highest `model.layers.{N}.` index in the manifest,
    // plus one. Every variant in `variants.rs` has a distinct
    // `num_layers`, so this alone discriminates them.
    let top_layer = wf
        .iter()
        .filter_map(|(name, _)| {
            name.strip_prefix("model.layers.")?
                .split('.')
                .next()?
                .parse::<usize>()
                .ok()
        })
        .max();
    // Hidden dim: the layer-0 `input_layernorm.weight` is a 1-D
    // rms-norm vector of length `hidden_dim` (the product covers a
    // `[1, hidden_dim]` storage variant).
    let hidden_dim = wf
        .tensor_info("model.layers.0.input_layernorm.weight")
        .map(|info| info.shape.iter().product::<usize>());
    check_variant_dims(top_layer, hidden_dim)
}

/// Decision half of [`probe_variant_match`]: compare the manifest's
/// top layer index and hidden dim against [`VARIANT`]. Pure (no I/O)
/// so it is unit-testable without a model on disk.
///
/// `top_layer` is the highest `model.layers.{N}` index (so layer
/// *count* is `top_layer + 1`); `None` means the manifest carried no
/// layer tensors at all. `hidden_dim` is the layer-0
/// `input_layernorm.weight` length; `None` means that tensor was
/// absent and the dim check is skipped.
fn check_variant_dims(
    top_layer: Option<usize>,
    hidden_dim: Option<usize>,
) -> Result<(), RsError> {
    let mismatch = |detail: String| RsError::ModelMismatch {
        expected: VARIANT.name,
        detail,
    };

    match top_layer {
        None => {
            return Err(mismatch(
                "manifest carries no `model.layers.*` tensors — not a \
                 moeflux-converted model directory?"
                    .to_string(),
            ));
        }
        Some(top) if top + 1 != VARIANT.num_layers => {
            return Err(mismatch(format!(
                "weights have {} layers, this binary expects {}. \
                 Rebuild with the matching `--features moeflux-model-…` \
                 or point at the {} model directory.",
                top + 1,
                VARIANT.num_layers,
                VARIANT.name,
            )));
        }
        Some(_) => {}
    }

    // Belt-and-suspenders for any future variants that happen to
    // share a `num_layers`.
    if let Some(found) = hidden_dim {
        if found != VARIANT.hidden_dim {
            return Err(mismatch(format!(
                "weights have hidden_dim {}, this binary expects {} \
                 ({}). Rebuild with the matching \
                 `--features moeflux-model-…`.",
                found, VARIANT.hidden_dim, VARIANT.name,
            )));
        }
    }

    Ok(())
}

impl RsCtx<MetalBackend> {
    /// Open a model. Argument order matches [`crate::imp::Ctx::open`].
    ///
    /// Phase 3: only the `weights` + `manifest` paths are consumed
    /// today. The remaining args are accepted for signature stability
    /// — they'll be wired into init as their consuming kernels land.
    pub fn open(
        weights: &Path,
        manifest: &Path,
        _vocab: &Path,
        experts_dir: &Path,
        experts_per_tok: u32,
        _use_2bit: bool,
    ) -> Result<Self, RsError> {
        let wf = WeightFile::open(weights, manifest)
            .map_err(|_| RsError::InitFailed)?;
        probe_variant_match(&wf)?;
        let experts = ExpertFiles::open(experts_dir)
            .map_err(|_| RsError::InitFailed)?;
        let layer_states = alloc_layer_states();
        let k_active = (experts_per_tok as usize).clamp(1, VARIANT.num_experts_per_tok);
        let io_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .thread_name(|i| format!("moeflux-io-{}", i))
            .build()
            .map_err(|_| RsError::InitFailed)?;
        let prefetch = PrefetchState::new(VARIANT.num_layers);
        Ok(Self {
            wf,
            backend: None,
            moe_buffers: None,
            experts,
            layer_states,
            k_active,
            layer_caches: None,
            linear_buffers: None,
            linear_attn_graph_scratch: None,
            full_attn_graph_scratch: None,
            moe_graph_scratch: None,
            hidden_double_buffer: None,
            head_tail_scratch: None,
            deferred: DeferredRing::new(),
            lm_head_gpu: None,
            io_pool,
            prefetch,
            checkpoints: HashMap::new(),
            checkpoint_order: VecDeque::new(),
            max_checkpoints: DEFAULT_MAX_CHECKPOINTS,
            mla_buffers: None,
            mla_yarn: None,
            mla_pipes: None,
            dense_mlp_bufs: None,
            dense_mlp_pipes: None,
            shared_expert_bufs: None,
            bf_matvec_pipes: None,
            mla_residual_scratch: None,
            mla_norm_pipes: None,
            residual_add_pipe: None,
        })
    }

    /// Build (or return) the Metal backend on demand. CPU-only kernels
    /// don't need it; GPU kernels go through this accessor so the
    /// shader-compile cost is paid lazily on first GPU use.
    fn metal_mut(&mut self) -> Result<&mut MetalContext, RsError> {
        self.ensure_backend()?;
        Ok(self.backend.as_mut().expect("just-set").metal_mut())
    }

    /// Lazy-build the [`MetalBackend`] (device + library + queue +
    /// `MtlWeightBuf` + `MetalBufferPool` + pre-warmed pipelines).
    /// Idempotent. Internal helper for the various ensure_* methods
    /// and the [`Self::metal_mut`] accessor.
    fn ensure_backend(&mut self) -> Result<(), RsError> {
        if self.backend.is_none() {
            let metal =
                MetalContext::new().map_err(|_| RsError::InitFailed)?;
            let device = metal.device().to_owned();
            let wf_buf = MtlWeightBuf::wrap(&self.wf, &device);
            self.backend = Some(
                MetalBackend::open(backend::gpu::MetalConfig { metal, wf_buf })
                    .map_err(|_| RsError::InitFailed)?,
            );
        }
        Ok(())
    }

    /// Ensure both the Metal backend and the persistent multi-expert
    /// buffer set exist, then return mutable refs to both.
    /// Field-disjoint borrows so two `&mut`s on the same `&mut self`
    /// are valid.
    fn metal_and_moe_mut(
        &mut self,
    ) -> Result<(&mut MetalContext, &mut MoeBuffers), RsError> {
        self.ensure_backend()?;
        if self.moe_buffers.is_none() {
            let pool =
                self.backend.as_mut().expect("just-set").pool_mut();
            self.moe_buffers = Some(MoeBuffers::new(pool));
        }
        let Self {
            backend, moe_buffers, ..
        } = self;
        Ok((
            backend.as_mut().expect("just-set").metal_mut(),
            moe_buffers.as_mut().expect("just-set"),
        ))
    }

    pub fn n_vocab(&self) -> usize {
        VARIANT.vocab_size
    }

    pub fn n_ctx(&self) -> usize {
        variants::MAX_SEQ_LEN
    }

    pub fn eos(&self) -> i32 {
        VARIANT.eos_token_1
    }

    pub fn model_name(&self) -> &'static str {
        VARIANT.name
    }

    /// Accumulated `(hits, misses)` for the speculative expert
    /// prefetch. Each value is a count of K-expert *slots*, not tokens
    /// — a single decode token contributes up to `K * num_layers` slot
    /// outcomes. Useful for per-request hit-rate telemetry; pair with
    /// [`Self::reset_prefetch_stats`] to scope to a single request.
    pub fn prefetch_stats(&self) -> (u64, u64) {
        self.prefetch.stats()
    }

    /// Zero the prefetch hit/miss counters.
    pub fn reset_prefetch_stats(&self) {
        self.prefetch.reset_stats();
    }

    /// Snapshot the Metal backend's per-label cmdbuf timing stats
    /// (see [`MetalContext::commit_and_wait_labeled`]). Returns
    /// `(label, stat)` pairs sorted by label. Empty until the backend
    /// is built (lazily, on the first forward pass) and at least one
    /// labeled commit has run. Profiling/diagnostics only.
    pub fn cmdbuf_stats(&self) -> Vec<(&'static str, CmdbufStat)> {
        self.backend
            .as_ref()
            .map(|b| b.metal().cmdbuf_stats())
            .unwrap_or_default()
    }

    /// Zero the Metal backend's per-label cmdbuf timing stats. Call
    /// before a measured prefill to scope the numbers to it. No-op if
    /// the backend has not been built yet.
    pub fn reset_cmdbuf_stats(&self) {
        if let Some(b) = self.backend.as_ref() {
            b.metal().reset_cmdbuf_stats();
        }
    }

    /// Embed a single token. Writes `HIDDEN_DIM` floats into `out`.
    /// First per-kernel entry point landed in Phase 3; bit-exact
    /// against the C `mf_embed_lookup`.
    pub fn embed(
        &self,
        token_id: i32,
        out: &mut [f32],
    ) -> Result<(), RsError> {
        embed_lookup(&self.wf, token_id, out).map_err(|_| RsError::EvalFailed)
    }

    /// CPU RMSNorm against the weight tensor `weight_name`. `x` and
    /// `out` are both `HIDDEN_DIM` long. Bit-exact against
    /// `mf_rms_norm_cpu` on the same hardware (deterministic CPU
    /// arithmetic, sequential reduction order).
    pub fn rms_norm_cpu(
        &self,
        weight_name: &str,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        rms_norm_cpu(&self.wf, weight_name, x, out)
            .map_err(|_| RsError::EvalFailed)
    }

    /// Per-head CPU RMSNorm, mutating `x_inout` in place. The buffer
    /// holds `num_heads * head_dim` floats (contiguous per head); each
    /// head's slice is RMS-normalized independently and scaled by the
    /// same `head_dim`-long bf16 weight loaded from `weight_name`.
    /// Bit-exact against `mf_rms_norm_per_head_cpu`.
    pub fn rms_norm_per_head_cpu(
        &self,
        weight_name: &str,
        num_heads: usize,
        head_dim: usize,
        x_inout: &mut [f32],
    ) -> Result<(), RsError> {
        rms_norm_per_head_cpu(&self.wf, weight_name, num_heads, head_dim, x_inout)
            .map_err(|_| RsError::EvalFailed)
    }

    /// Apply rotary position embedding to Q and K in place at
    /// position `pos`. `q` is `num_attn_heads * head_dim` floats; `k`
    /// is `num_kv_heads * head_dim`. Bit-exact against
    /// `mf_apply_rotary_emb` on the same hardware.
    pub fn apply_rotary_emb(
        &self,
        pos: i32,
        q: &mut [f32],
        k: &mut [f32],
    ) -> Result<(), RsError> {
        apply_rotary_emb(pos, q, k).map_err(|_| RsError::EvalFailed)
    }

    /// Scaled dot-product attention with sigmoid-gated output, single
    /// query position. ULP-bounded against `mf_sdpa_cpu` (libm `expf`
    /// in softmax + sigmoid sit in the same compiler-choice territory
    /// as RoPE's trig calls).
    pub fn sdpa_cpu(
        &self,
        kv_len: i32,
        q: &[f32],
        q_gate: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        sdpa_cpu(kv_len, q, q_gate, k_cache, v_cache, out)
            .map_err(|_| RsError::EvalFailed)
    }

    /// CPU LM head matvec. `x` is `HIDDEN_DIM` floats (the post-final-
    /// norm hidden state); `out` is `VOCAB_SIZE` floats (raw logits).
    /// Bit-exact target against `mf_lm_head_cpu` on the same hardware.
    pub fn lm_head_cpu(
        &self,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        lm_head_cpu(&self.wf, x, out).map_err(|_| RsError::EvalFailed)
    }

    /// MoE router: softmax → top-K → normalize. `scores` is mutated in
    /// place (post-call it holds the softmaxed probabilities).
    /// `indices` (length `k`) receives the top-K expert IDs in the
    /// selection-sort slot order matching `mf_moe_router_cpu`;
    /// `weights` (length `k`) receives the normalized expert weights.
    /// ULP-bounded against the C path (libm `expf` in softmax).
    pub fn moe_router_cpu(
        &self,
        scores: &mut [f32],
        k: usize,
        indices: &mut [i32],
        weights: &mut [f32],
    ) -> Result<(), RsError> {
        moe_router_cpu(scores, k, indices, weights)
            .map_err(|_| RsError::EvalFailed)
    }

    /// Depthwise 1D conv step + SiLU. `weight_name` references a bf16
    /// tensor of length `channels * kernel_size`. ULP-bounded against
    /// `mf_conv1d_step_cpu` (one libm `expf` per channel in the SiLU
    /// tail; dot product matches clang via `mul_add`).
    pub fn conv1d_step_cpu(
        &self,
        weight_name: &str,
        channels: usize,
        kernel_size: usize,
        conv_state: &[f32],
        new_input: &[f32],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        let bytes = self
            .wf
            .tensor_bytes(weight_name)
            .ok_or(RsError::EvalFailed)?;
        conv1d_step(
            conv_state,
            new_input,
            bytes,
            channels,
            kernel_size,
            out,
        )
        .map_err(|_| RsError::EvalFailed)
    }

    /// Bare CPU RMSNorm (no weight). Bit-exact against
    /// `mf_rms_norm_bare_cpu` on the same hardware.
    pub fn rms_norm_bare_cpu(
        &self,
        eps: f32,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        rms_norm_bare(x, eps, out).map_err(|_| RsError::EvalFailed)
    }

    /// CPU RMSNormGated: `out[i] = rms_norm(x)[i] * w[i] * silu(z[i])`.
    /// ULP-bounded against `mf_rms_norm_gated_cpu` (libm `expf` in SiLU).
    pub fn rms_norm_gated_cpu(
        &self,
        weight_name: &str,
        eps: f32,
        x: &[f32],
        z: &[f32],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        rms_norm_gated(&self.wf, weight_name, x, z, eps, out)
            .map_err(|_| RsError::EvalFailed)
    }

    /// Gated-delta-net recurrence step. Loads `A_log` (f32) and
    /// `dt_bias` (bf16) for the named layer, then runs the per-v-head
    /// decay → kv_mem → delta → state update → output sequence.
    /// `ssm_state` is mutated in place; `out_values` is overwritten.
    /// ULP-bounded against `mf_gated_delta_recurrence_cpu` (libm
    /// `expf`/`logf` per head, `mul_add` matched to clang's FMA).
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
    ) -> Result<(), RsError> {
        let a_log_name =
            format!("model.layers.{layer_idx}.linear_attn.A_log");
        let dt_bias_name =
            format!("model.layers.{layer_idx}.linear_attn.dt_bias");
        let a_log_bytes = self
            .wf
            .tensor_bytes(&a_log_name)
            .ok_or(RsError::EvalFailed)?;
        let dt_bias_bytes = self
            .wf
            .tensor_bytes(&dt_bias_name)
            .ok_or(RsError::EvalFailed)?;

        if a_log_bytes.len() != v_heads * 4 {
            return Err(RsError::EvalFailed);
        }
        let mut a_log = vec![0.0f32; v_heads];
        for (i, chunk) in a_log_bytes.chunks_exact(4).enumerate() {
            a_log[i] = f32::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
            ]);
        }

        gated_delta_recurrence(
            &a_log,
            dt_bias_bytes,
            alpha,
            beta,
            q,
            k,
            v,
            v_heads,
            k_heads,
            key_dim,
            value_dim,
            ssm_state,
            out_values,
        )
        .map_err(|_| RsError::EvalFailed)
    }

    /// GPU RMSNorm with bf16 weights (slice 9e). Chains
    /// `rms_norm_sum_sq` + `rms_norm_apply_bf16` into one cmdbuf.
    /// `weight_bf16` is the raw little-endian bf16 byte sequence
    /// (typically from `WeightFile::tensor_bytes(name)`).
    /// First GPU kernel under diff with threadgroup-shared
    /// reduction — empirical question whether this engages the
    /// cosine/Jaccard floors.
    pub fn gpu_rms_norm_fused(
        &mut self,
        x: &[f32],
        weight_bf16: &[u8],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        let metal = self.metal_mut()?;
        gpu_rms_norm_fused(metal, x, weight_bf16, out)
            .map_err(|_| RsError::EvalFailed)
    }

    /// `attn_scores_batched` (slice 5d-7a). Per-head Q · K^T scaled.
    /// Stride-tight oracle entry (`seq_stride = seq_len`).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_scores_batched(
        &mut self,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        q: &[f32],
        k_cache: &[f32],
        scale: f32,
        scores_out: &mut [f32],
    ) -> Result<(), RsError> {
        let metal = self.metal_mut()?;
        gpu_attn_scores_batched(
            metal, num_heads, num_kv_heads, head_dim, seq_len, q, k_cache,
            scale, scores_out,
        )
        .map_err(|_| RsError::EvalFailed)
    }

    /// `attn_softmax_batched` (slice 5d-7a). Per-head softmax over
    /// `[0, seq_len)`, in place.
    pub fn attn_softmax_batched(
        &mut self,
        num_heads: u32,
        seq_len: u32,
        scores_inout: &mut [f32],
    ) -> Result<(), RsError> {
        let metal = self.metal_mut()?;
        gpu_attn_softmax_batched(metal, num_heads, seq_len, scores_inout)
            .map_err(|_| RsError::EvalFailed)
    }

    /// `attn_values_batched` (slice 5d-7a). Per-head scores · V.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_values_batched(
        &mut self,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        scores: &[f32],
        v_cache: &[f32],
        out: &mut [f32],
    ) -> Result<(), RsError> {
        let metal = self.metal_mut()?;
        gpu_attn_values_batched(
            metal, num_heads, num_kv_heads, head_dim, seq_len, scores,
            v_cache, out,
        )
        .map_err(|_| RsError::EvalFailed)
    }

    /// `sigmoid_gate` (slice 5d-7a). `x_inout[i] *= sigmoid(gate[i])`.
    pub fn sigmoid_gate(
        &mut self,
        dim: u32,
        gate: &[f32],
        x_inout: &mut [f32],
    ) -> Result<(), RsError> {
        let metal = self.metal_mut()?;
        gpu_sigmoid_gate(metal, dim, gate, x_inout)
            .map_err(|_| RsError::EvalFailed)
    }

    /// Read one expert's `EXPERT_SIZE`-byte 4-bit blob from disk
    /// (slice 9c). Bypasses every cache; equivalent to a cold pread
    /// against `packed_experts/layer_NN.bin`. Diff-oracle dump point
    /// for the expert-loader.
    pub fn load_expert_bytes(
        &self,
        layer_idx: usize,
        expert_idx: usize,
        out: &mut [u8],
    ) -> Result<(), RsError> {
        self.experts
            .read_expert(layer_idx, expert_idx, out)
            .map_err(|_| RsError::EvalFailed)
    }

    /// Single-expert GPU FFN forward (slice 9a). `expert_data` is one
    /// expert's `EXPERT_SIZE`-byte 4-bit packed blob laid out as
    /// `[gate | up | down]` per `model_variant.h`. `h_post` is the
    /// post-attn-norm hidden state (HIDDEN_DIM floats); `expert_out`
    /// receives the HIDDEN_DIM-float expert output. Cosine/Jaccard
    /// territory against `mf_gpu_expert_forward` (Metal SIMD-reduce
    /// nondeterminism).
    pub fn gpu_expert_forward(
        &mut self,
        expert_data: &[u8],
        h_post: &[f32],
        expert_out: &mut [f32],
    ) -> Result<(), RsError> {
        let metal = self.metal_mut()?;
        gpu_expert_forward(metal, expert_data, h_post, expert_out)
            .map_err(|_| RsError::EvalFailed)
    }

    /// Batched K-expert FFN forward + GPU combine (slice 9b). Encodes
    /// `actual_K` parallel expert FFNs (`gate matvec → up matvec →
    /// SwiGLU → down matvec`) followed by `moe_combine_residual` into
    /// one command buffer. The combine yields
    /// `h_mid + Σ weights[k] × expert_out[k] + sigmoid(gate) × shared_out`.
    /// Cosine/Jaccard territory against `mf_gpu_batched_experts_forward`.
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_batched_experts_forward(
        &mut self,
        actual_k: i32,
        expert_data: &[u8],
        h_post: &[f32],
        h_mid: &[f32],
        shared_out: &[f32],
        expert_weights: &[f32],
        shared_gate_score: f32,
        hidden_out: &mut [f32],
    ) -> Result<(), RsError> {
        self.ensure_backend()?;
        if self.moe_buffers.is_none() {
            let pool =
                self.backend.as_mut().expect("just-set").pool_mut();
            self.moe_buffers = Some(MoeBuffers::new(pool));
        }
        let Self {
            backend, moe_buffers, ..
        } = self;
        let backend =
            backend.as_mut().expect("ensure_backend just-set");
        let (metal, _wf_buf, pool) = backend.parts_mut();
        let bufs = moe_buffers.as_mut().expect("just-set");
        let payload = moe::expert_forward::ExpertPayload {
            h_post,
            h_mid,
            shared_out,
            expert_weights,
            shared_gate_score,
        };
        gpu_batched_experts_forward(
            metal,
            bufs,
            pool,
            actual_k,
            expert_data,
            payload,
            hidden_out,
        )
        .map_err(|_| RsError::EvalFailed)
    }

    /// Phase 4 layer-boundary checkpoint hook. Runs a single layer's
    /// forward pass starting from `hidden_in`, returning the post-
    /// layer hidden state in `hidden_out`. The targeted layer's
    /// recurrence state (KV cache for full-attn, conv/SSM state for
    /// linear-attn) is mutated in place. 4c landed the linear-attn
    /// path; 4d added the full-attn path via [`full_attn_layer_forward`].
    ///
    /// Default `gpu_combine = true` (slice 4f-3 production behavior).
    /// Use [`Self::layer_forward_dump_with_gpu_combine`] to exercise
    /// the slice 4f-4 CPU-combine path.
    pub fn layer_forward_dump(
        &mut self,
        layer_idx: i32,
        pos: i32,
        hidden_in: &[f32],
        hidden_out: &mut [f32],
    ) -> Result<(), RsError> {
        self.layer_forward_dump_inner(
            layer_idx, pos, hidden_in, hidden_out, true,
        )
    }

    /// As [`Self::layer_forward_dump`] but lets the caller select
    /// `gpu_combine`. Slice 4f-4 added this entry point so the diff
    /// oracle can exercise the CPU-combine fallback (the path the C
    /// side takes when the next layer's `input_layernorm_w` is
    /// unavailable, or the combine pipelines failed to compile). In
    /// production today every caller uses `true`; slice 4f-perf will
    /// thread the C-mirrored `should_gpu_combine` predicate through
    /// `step_internal`.
    pub fn layer_forward_dump_with_gpu_combine(
        &mut self,
        layer_idx: i32,
        pos: i32,
        hidden_in: &[f32],
        hidden_out: &mut [f32],
        gpu_combine: bool,
    ) -> Result<(), RsError> {
        self.layer_forward_dump_inner(
            layer_idx, pos, hidden_in, hidden_out, gpu_combine,
        )
    }

    fn layer_forward_dump_inner(
        &mut self,
        layer_idx: i32,
        pos: i32,
        hidden_in: &[f32],
        hidden_out: &mut [f32],
        gpu_combine: bool,
    ) -> Result<(), RsError> {
        let v = VARIANT;
        if layer_idx < 0 || (layer_idx as usize) >= v.num_layers {
            return Err(RsError::EvalFailed);
        }
        if pos < 0 {
            return Err(RsError::EvalFailed);
        }
        if hidden_in.len() != v.hidden_dim || hidden_out.len() != v.hidden_dim
        {
            return Err(RsError::EvalFailed);
        }

        let layer_idx_us = layer_idx as usize;
        let is_full =
            v.layer_kind(layer_idx_us) == variants::LayerKind::FullAttn;

        // Ensure all lazy resources exist.
        self.ensure_linear_resources()?;

        // Field-disjoint mutable borrows for the forward call.
        let k_active = self.k_active;
        let Self {
            wf,
            backend,
            moe_buffers,
            experts,
            layer_states,
            layer_caches,
            linear_buffers,
            deferred,
            io_pool,
            prefetch,
            ..
        } = self;

        let backend =
            backend.as_mut().expect("ensure_linear_resources");
        let (metal, wf_buf, buffer_pool) = backend.parts_mut();
        let layer_caches =
            layer_caches.as_ref().expect("ensure_linear_resources");
        let linear_buffers =
            linear_buffers.as_mut().expect("ensure_linear_resources");
        let moe_buffers =
            moe_buffers.as_mut().expect("ensure_linear_resources");
        let io_pool: &rayon::ThreadPool = &*io_pool;

        // Defensive: drain any leaked deferred state from a prior
        // failed call so this entry point stays safe to call
        // back-to-back without `AlreadyActive` errors. After slice
        // 4f-3 the layer forwards leave an in-flight dispatch on
        // return, and a buggy caller might forget to drain — this
        // bracketing guarantees the dump-hook contract (single
        // synchronous layer step) holds regardless.
        //
        // Slice 5d-6b: also drain any in-flight prefetch + clear
        // last-token predictions. The dump hook tests one layer at
        // a time; predictions from a previous test would either be
        // stale or wouldn't match this test's routing anyway.
        moe::deferred::discard_deferred_experts_in(deferred);
        prefetch.invalidate_all();

        // Stage hidden_in into the persistent input buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden_in.as_ptr(),
                buffer_pool.handle(linear_buffers.input).contents()
                    as *mut f32,
                v.hidden_dim,
            );
        }

        // Slice 5d-9: dump hook tests one layer at a time. Match the
        // production parity convention so the dispatched layer reads
        // from the same set its prefetch (if any) wrote to. The
        // `invalidate_all` above means no prefetch is in flight, but
        // production-path data_set_per_slot stays consistent.
        let prefetch_set = layer_idx_us % 2;

        if is_full {
            let kv_state = full_kv_mut(layer_states, layer_idx_us)?;
            let layer_ctx = backend::gpu::gpu_ctx::GpuLayerCtx {
                wf,
                wf_buf,
                layer_cache: &layer_caches[layer_idx_us],
                buffers: linear_buffers,
                buffer_pool,
            };
            full_attn_layer_forward(
                metal,
                &layer_ctx,
                moe_buffers,
                deferred,
                layer_idx_us,
                pos,
                k_active,
                experts,
                io_pool,
                prefetch,
                prefetch_set,
                kv_state,
                gpu_combine,
                /* prev_layer_chained = */ false,
                /* chain_next_norm_off = */ None,
            )
            .map_err(|_| RsError::EvalFailed)?;
        } else {
            let layer_state = match &mut layer_states[layer_idx_us] {
                LayerState::LinearAttn(la) => la,
                LayerState::FullAttn(_) | LayerState::Mla(_) => {
                    return Err(RsError::EvalFailed);
                }
            };
            let layer_ctx = backend::gpu::gpu_ctx::GpuLayerCtx {
                wf,
                wf_buf,
                layer_cache: &layer_caches[layer_idx_us],
                buffers: linear_buffers,
                buffer_pool,
            };
            linear_attn_layer_forward(
                metal,
                &layer_ctx,
                moe_buffers,
                deferred,
                layer_idx_us,
                k_active,
                experts,
                io_pool,
                prefetch,
                prefetch_set,
                layer_state,
                gpu_combine,
                /* prev_layer_chained = */ false,
                /* chain_next_norm_off = */ None,
            )
            .map_err(|_| RsError::EvalFailed)?;
        }

        // Drain the in-flight K-expert dispatch into `linear_buffers.
        // input` so the existing readback below sees the post-combine
        // hidden state. Slice 4f-3 made the layer forwards async; the
        // dump hook reconstitutes the synchronous single-step contract
        // by completing the dispatch right here.
        // SAFETY: shared-storage buffer; the GPU work for this layer
        // is the dispatch we're about to wait on, and `complete_*` is
        // what does the wait. After it returns, no GPU work is in
        // flight against `linear_buffers.input`.
        let buf_input_slice = unsafe {
            backend::gpu::metal::buffer_as_mut_slice::<f32>(
                buffer_pool.handle(linear_buffers.input),
                v.hidden_dim,
            )
        };
        moe::deferred::complete_deferred_experts_into(
            deferred,
            moe_buffers,
            buffer_pool,
            buf_input_slice,
        )
        .map_err(|_| RsError::EvalFailed)?;

        // Read post-forward hidden state out of buffers.input.
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer_pool.handle(linear_buffers.input).contents() as *const f32,
                hidden_out.as_mut_ptr(),
                v.hidden_dim,
            );
        }
        Ok(())
    }

    /// 4c diagnostic — runs `layer_forward_dump` and additionally
    /// copies out the post-attn-norm hidden, the post-residual h_mid,
    /// the pre-sigmoid-gate shared expert output, and the shared gate
    /// score. Test-only.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_forward_dump_intermediates(
        &mut self,
        layer_idx: i32,
        pos: i32,
        hidden_in: &[f32],
        hidden_out: &mut [f32],
        h_post_out: Option<&mut [f32]>,
        h_mid_out: Option<&mut [f32]>,
        shared_out_out: Option<&mut [f32]>,
        gate_score_out: Option<&mut f32>,
    ) -> Result<(), RsError> {
        // Run the forward, then read the intermediates from the
        // persistent buffers before the next layer (or the test) can
        // overwrite them.
        self.layer_forward_dump(layer_idx, pos, hidden_in, hidden_out)?;
        let bufs = self
            .linear_buffers
            .as_ref()
            .ok_or(RsError::EvalFailed)?;
        let pool =
            self.backend.as_ref().ok_or(RsError::EvalFailed)?.pool();
        let v = VARIANT;
        let read_into = |buf: &::metal::Buffer, dst: Option<&mut [f32]>| {
            if let Some(dst) = dst {
                let n = dst.len();
                let src = buf.contents() as *const f32;
                // SAFETY: shared storage; no in-flight GPU work because
                // layer_forward_dump waits internally.
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), n);
                }
            }
        };
        read_into(pool.handle(bufs.normed), h_post_out);
        read_into(pool.handle(bufs.h_mid), h_mid_out);
        read_into(pool.handle(bufs.shared_out), shared_out_out);
        if let Some(gate_dst) = gate_score_out {
            let s = pool.handle(bufs.shared_gate).contents() as *const f32;
            // SAFETY: shared storage.
            *gate_dst = unsafe { *s };
        }
        let _ = v; // silence
        Ok(())
    }

    /// Lazily build the Metal backend, weight buffer, layer caches,
    /// linear-attn persistent buffers, and MoE buffer set. Idempotent
    /// — subsequent calls are no-ops.
    /// Slim sibling of [`Self::ensure_linear_resources`] for the
    /// MLA / DeepSeek-V3 CPU path. The MLA pipeline runs entirely on
    /// host buffers and only touches the GPU for the final
    /// `lm_head` matvec. Skipping the GQA-specific
    /// [`LayerWeightCache::build_all`] is load-bearing — it requires
    /// `q_proj` / `k_proj` / `v_proj` tensor names that don't exist
    /// on MLA variants (Cogito-V2 has `q_a_proj` / `q_b_proj` /
    /// `kv_a_proj_with_mqa` / `kv_b_proj` instead).
    fn ensure_mla_resources(&mut self) -> Result<(), RsError> {
        self.ensure_backend()?;
        if self.lm_head_gpu.is_none() {
            let Self {
                wf,
                backend,
                lm_head_gpu,
                ..
            } = self;
            let backend = backend.as_mut().expect("just-set");
            let (metal, wf_buf, _) = backend.parts_mut();
            *lm_head_gpu = Some(
                GpuLmHead::new(metal, wf, wf_buf)
                    .map_err(|_| RsError::InitFailed)?,
            );
        }
        // Lazy-allocate per-layer MLA KV caches in shared-storage Metal
        // buffers. Idempotent — `ensure_buffers` no-ops on already-
        // populated layers, so re-entering after `memory_clear` (which
        // truncates `len` to 0 but keeps the buffers) is cheap.
        let device = self
            .backend
            .as_ref()
            .expect("just-set")
            .metal()
            .device()
            .to_owned();
        for state in self.layer_states.iter_mut() {
            if let LayerState::Mla(mla) = state {
                mla.ensure_buffers(&device);
            }
        }
        // GPU MLA forward resources — buffers + YaRN tables + pipelines.
        // Used by `step_internal_mla_gpu`; absent on the env-gated CPU
        // fallback path.
        if self.mla_buffers.is_none() {
            self.mla_buffers =
                Some(attn::mla_attn_forward::MlaForwardBuffers::new(&device));
        }
        if self.mla_yarn.is_none() {
            self.mla_yarn =
                Some(attn::mla_attn_forward::MlaYarnTables::new(&device));
        }
        if self.mla_pipes.is_none() {
            let metal =
                self.backend.as_mut().expect("just-set").metal_mut();
            self.mla_pipes = Some(
                attn::mla_attn_forward::MlaForwardPipelines::new(metal)
                    .map_err(|_| RsError::InitFailed)?,
            );
        }
        // Phase 3 — full-GPU dense MLP + MoE for the MLA path.
        if VARIANT.first_k_dense_replace > 0 && self.dense_mlp_bufs.is_none()
        {
            self.dense_mlp_bufs =
                Some(backend::gpu::dense_mlp_gpu::DenseMlpBuffers::new(&device));
        }
        if VARIANT.first_k_dense_replace > 0
            && self.dense_mlp_pipes.is_none()
        {
            let metal =
                self.backend.as_mut().expect("just-set").metal_mut();
            self.dense_mlp_pipes = Some(
                backend::gpu::dense_mlp_gpu::DenseMlpPipelines::fetch(metal)
                    .map_err(|_| RsError::InitFailed)?,
            );
        }
        if self.moe_buffers.is_none() {
            let pool =
                self.backend.as_mut().expect("just-set").pool_mut();
            self.moe_buffers = Some(MoeBuffers::new(pool));
        }
        if self.shared_expert_bufs.is_none() && VARIANT.shared_intermediate > 0
        {
            self.shared_expert_bufs =
                Some(moe::cogito_moe_gpu::SharedExpertBuffers::new(&device));
        }
        // Phase 1 (cogito-v2 full-GPU): the GPU shared-expert SwiGLU
        // reuses `DenseMlpPipelines` (matvec + swiglu_fused are
        // dim-parametric), so even variants with `first_k_dense_replace
        // == 0` need the dense pipes if they have a shared expert.
        if self.dense_mlp_pipes.is_none() && VARIANT.shared_intermediate > 0 {
            let metal =
                self.backend.as_mut().expect("just-set").metal_mut();
            self.dense_mlp_pipes = Some(
                backend::gpu::dense_mlp_gpu::DenseMlpPipelines::fetch(metal)
                    .map_err(|_| RsError::InitFailed)?,
            );
        }
        // Phase 2 (cogito-v2 full-GPU): BF16 matvec PSO for the MoE
        // router gate. Only allocated for MLA variants (the linear-attn
        // path's gate is 8-bit dequant via the existing MatvecPipelines).
        if self.bf_matvec_pipes.is_none() {
            let metal =
                self.backend.as_mut().expect("just-set").metal_mut();
            self.bf_matvec_pipes = Some(
                backend::gpu::gpu_matvec::BfMatvecPipelines::fetch(metal)
                    .map_err(|_| RsError::InitFailed)?,
            );
        }
        // Phase 5 (cogito-v2 full-GPU): persistent GPU scratch + norm
        // pipelines for the orchestrator's residual stream.
        if self.mla_residual_scratch.is_none() {
            self.mla_residual_scratch =
                Some(backend::gpu::gpu_norm::MlaForwardScratch::new(&device));
        }
        if self.mla_norm_pipes.is_none() {
            let metal =
                self.backend.as_mut().expect("just-set").metal_mut();
            self.mla_norm_pipes = Some(
                backend::gpu::gpu_norm::RmsNormBf16Pipelines::fetch(metal)
                    .map_err(|_| RsError::InitFailed)?,
            );
        }
        if self.residual_add_pipe.is_none() {
            let metal =
                self.backend.as_mut().expect("just-set").metal_mut();
            self.residual_add_pipe = Some(
                metal
                    .pipeline("residual_add")
                    .map_err(|_| RsError::InitFailed)?
                    .clone(),
            );
        }
        Ok(())
    }

    fn ensure_linear_resources(&mut self) -> Result<(), RsError> {
        self.ensure_backend()?;
        // Wrap each mmap'd layer file as a Metal-shared buffer via
        // newBufferWithBytesNoCopy. Idempotent (skipped after first
        // call). Expert I/O is unconditionally mmap as of
        // `pread_teardown_landed.md` (2026-05-20).
        {
            let pool =
                self.backend.as_mut().expect("just-set").pool_mut();
            self.experts.attach_to_device(pool);
        }
        // Prefill arc Phase 0b: register the GPU-resident KV cache for
        // every full-attn (GQA) layer into the pool, once, up front.
        // Eager — not lazy-per-eval — because `KvCache::ensure_buffers`
        // needs `&mut pool`, available only here. Idempotent:
        // `ensure_buffers` skips already-registered caches. MLA
        // variants have no `FullAttn(KvCache)` layers, so the loop is
        // empty for them.
        {
            let Self {
                layer_states,
                backend,
                ..
            } = self;
            let pool =
                backend.as_mut().expect("just-set").pool_mut();
            for layer in layer_states.iter_mut() {
                if let LayerState::FullAttn(kv) = layer {
                    kv.ensure_buffers(pool);
                }
            }
        }
        if self.layer_caches.is_none() {
            let wf_buf =
                self.backend.as_ref().expect("just-set").weight_buf();
            let caches = LayerWeightCache::build_all(&self.wf, wf_buf)
                .map_err(|_| RsError::InitFailed)?;
            self.layer_caches = Some(caches);
        }
        if self.linear_buffers.is_none() {
            let pool =
                self.backend.as_mut().expect("just-set").pool_mut();
            self.linear_buffers = Some(LinearAttnBuffers::new(pool));
        }
        // Construction order matters: `LinearAttnGraphScratch` is
        // all-transient (`persistent = false`); the structs after it
        // (`full_attn_graph_scratch`, `moe_graph_scratch`,
        // `hidden_double_buffer`, `moe_buffers`) all end on a
        // persistent alloc, so the highest pool BufId stays persistent
        // and `reset_transient` keeps the transients above it.
        if self.linear_attn_graph_scratch.is_none() {
            let pool =
                self.backend.as_mut().expect("just-set").pool_mut();
            self.linear_attn_graph_scratch = Some(
                attn::linear_attn_forward::LinearAttnGraphScratch::new(
                    pool,
                ),
            );
        }
        if self.full_attn_graph_scratch.is_none() {
            let pool =
                self.backend.as_mut().expect("just-set").pool_mut();
            self.full_attn_graph_scratch = Some(
                attn::full_attn_forward::FullAttnGraphScratch::new(pool),
            );
        }
        if self.moe_graph_scratch.is_none() {
            let k_active = self.k_active;
            let pool =
                self.backend.as_mut().expect("just-set").pool_mut();
            self.moe_graph_scratch = Some(
                attn::linear_attn_forward::MoeGraphScratch::new(
                    pool, k_active,
                ),
            );
        }
        if self.hidden_double_buffer.is_none() {
            let pool =
                self.backend.as_mut().expect("just-set").pool_mut();
            self.hidden_double_buffer = Some(
                attn::linear_attn_forward::HiddenDoubleBuffer::new(pool),
            );
        }
        if self.head_tail_scratch.is_none() {
            let pool =
                self.backend.as_mut().expect("just-set").pool_mut();
            self.head_tail_scratch = Some(
                attn::linear_attn_forward::HeadTailScratch::new(pool),
            );
        }
        if self.moe_buffers.is_none() {
            let pool =
                self.backend.as_mut().expect("just-set").pool_mut();
            self.moe_buffers = Some(MoeBuffers::new(pool));
        }
        if self.lm_head_gpu.is_none() {
            let Self {
                wf,
                backend,
                lm_head_gpu,
                ..
            } = self;
            let backend = backend.as_mut().expect("just-set");
            let (metal, wf_buf, _) = backend.parts_mut();
            *lm_head_gpu = Some(
                GpuLmHead::new(metal, wf, wf_buf)
                    .map_err(|_| RsError::InitFailed)?,
            );
        }
        Ok(())
    }

    /// Process `tokens.len()` tokens at positions `[start_pos,
    /// start_pos + tokens.len())`. Only the final token emits logits
    /// into `logits` (the prefix is state-update-only). Mirrors C
    /// `mf_eval_prompt` (infer.m:7723..7744).
    ///
    /// `seq_id` is accepted for signature parity with the C API and
    /// ignored — moeflux is single-stream.
    ///
    /// Empty `tokens`: returns `Ok(())` without writing `logits` (the
    /// loop body never runs, matching the C-side empty-loop case).
    pub fn eval_prompt(
        &mut self,
        tokens: &[i32],
        start_pos: usize,
        _seq_id: i32,
        logits: &mut [f32],
    ) -> Result<(), RsError> {
        if logits.len() != VARIANT.vocab_size {
            return Err(RsError::EvalFailed);
        }
        if tokens.is_empty() {
            return Ok(());
        }
        self.step_internal(tokens, start_pos as i32, Some(&mut logits[..]))
    }

    /// Decode-style single-token step. Always emits logits. Mirrors C
    /// `mf_eval_token` (infer.m:7746..7757).
    ///
    /// **Routing is env-gated** via `MOEFLUX_EVAL_TOKEN`:
    /// - `oracle` (default): per-token oracle, with the deferred
    ///   K-expert ring giving cross-layer pipelining.
    /// - `batched`: routes through `step_internal(&[tok], pos, ...)`,
    ///   which fires the prefetch state machine wired in Phase 3.
    ///   Cosine=1.0 against oracle at N=1. In-process A/B
    ///   (`bench_decode_per_token_vs_batched_n1`) measures
    ///   batched-N=1 ~10.5% slower than oracle on decode (the gap is
    ///   the oracle's cross-layer deferred-K-expert pipelining —
    ///   layer N's async K-expert dispatch overlapping layer N+1's
    ///   CMD1+CMD2+3 — which the batched permute-fuse can't yet
    ///   replicate at N=1). The opt-in lets the user measure the
    ///   batched-N=1 path's improvements as cross-layer pipelining
    ///   inside batched lands, without a recompile.
    pub fn eval_token(
        &mut self,
        token: i32,
        pos: usize,
        _seq_id: i32,
        logits: &mut [f32],
    ) -> Result<(), RsError> {
        if logits.len() != VARIANT.vocab_size {
            return Err(RsError::EvalFailed);
        }
        match eval_token_mode() {
            EvalTokenMode::Oracle => self
                .step_internal_per_token_oracle(
                    token,
                    pos as i32,
                    Some(logits),
                ),
            EvalTokenMode::Batched => {
                self.step_internal(&[token], pos as i32, Some(logits))
            }
        }
    }

    /// Canonical multi-token forward orchestrator. Processes
    /// `tokens` at positions `[start_pos, start_pos + tokens.len())`
    /// and (when requested) writes the last token's logits.
    ///
    /// **Session-4 implementation (Gqa variants):** batched MoE
    /// permute-fuse path. For each layer:
    /// - Full-attention layers (Qwen3 GQA) route through
    ///   [`full_attn_forward::batched_full_attn_layer_forward`], which
    ///   runs the per-token pre-MoE forward (B1 sub-step: tokenwise)
    ///   then a single batched MoE dispatch over the joint N×K_active
    ///   routing CSR.
    /// - Linear-attention layers stay tokenwise: each token gets a
    ///   `linear_attn_layer_forward` call with sync deferred drain,
    ///   advancing the recurrent state position-by-position. Chunkwise
    ///   linear-attn is a future-work primitive.
    ///
    /// MLA variants fall back to a tokenwise loop over the per-token
    /// oracle — batched MLA is out of scope for session 4.
    ///
    /// `eval_prompt` routes here; `eval_token` stays on the per-token
    /// oracle since single-token decode doesn't benefit from batching.
    pub(crate) fn step_internal(
        &mut self,
        tokens: &[i32],
        start_pos: i32,
        logits_out: Option<&mut [f32]>,
    ) -> Result<(), RsError> {
        if tokens.is_empty() {
            return Ok(());
        }
        if start_pos < 0 {
            return Err(RsError::EvalFailed);
        }
        if let Some(ref l) = logits_out {
            if l.len() != VARIANT.vocab_size {
                return Err(RsError::EvalFailed);
            }
        }

        // MLA: tokenwise oracle for now (batched MLA = session-5+).
        if matches!(VARIANT.attn_kind, variants::AttnKind::Mla) {
            let last_idx = tokens.len() - 1;
            let mut logits_owned = logits_out;
            for (i, &tok) in tokens.iter().enumerate() {
                let pos = start_pos + i as i32;
                let logits_arg: Option<&mut [f32]> = if i == last_idx {
                    logits_owned.take()
                } else {
                    None
                };
                self.step_internal_per_token_oracle(tok, pos, logits_arg)?;
            }
            return Ok(());
        }

        // Chunked iteration over the batched-GQA path. Phase D fixes
        // CHUNK_SIZE = 8192 — the Phase F sweep will validate this
        // hardware-tuned default. Only the last chunk's last token
        // emits logits. `start_pos` advances by `CHUNK_SIZE` per
        // chunk so the KV cache reflects the cumulative position.
        //
        // Tests can override CHUNK_SIZE via [`BATCHED_CHUNK_OVERRIDE`]
        // (thread-local) so multi-chunk boundary math is exercisable
        // at small N without paying long-prompt wall-clock.
        let chunk_size = batched_chunk_size();
        let n = tokens.len();
        let mut chunk_start = 0usize;
        let mut chunk_idx = 0usize;
        let mut logits_owned = logits_out;
        while chunk_start < n {
            let chunk_end = (chunk_start + chunk_size).min(n);
            let is_last_chunk = chunk_end == n;
            let chunk_tokens = &tokens[chunk_start..chunk_end];
            let chunk_start_pos = start_pos + chunk_start as i32;
            let chunk_logits = if is_last_chunk {
                logits_owned.take()
            } else {
                None
            };
            self.step_internal_batched_gqa(
                chunk_tokens,
                chunk_start_pos,
                chunk_logits,
                chunk_idx,
            )?;
            chunk_start = chunk_end;
            chunk_idx += 1;
        }
        Ok(())
    }

    /// GQA batched orchestrator — the Phase B+C implementation that
    /// `step_internal` dispatches to when the variant is Qwen3-shape
    /// (full-attn + linear-attn layers). Sub-steps B1-B4 progressively
    /// batch more operations inside
    /// [`full_attn_forward::batched_full_attn_layer_forward`]; this
    /// orchestrator stays stable across sub-steps.
    fn step_internal_batched_gqa(
        &mut self,
        tokens: &[i32],
        start_pos: i32,
        logits_out: Option<&mut [f32]>,
        chunk_idx: usize,
    ) -> Result<(), RsError> {
        use attn::full_attn_forward::batched_full_attn_layer_forward;

        let v = VARIANT;
        let n = tokens.len();
        let hidden_dim = v.hidden_dim;

        self.ensure_linear_resources()?;
        let k_active = self.k_active;

        // Field-disjoint mutable borrows. `io_pool` dropped here on
        // 2026-05-20 alongside the pread teardown — the batched
        // orchestrator no longer fires prefetch dispatch (the N=1
        // re-route to this path was reverted in session 5). `prefetch`
        // stays for the `prefetch.drain()` housekeeping below; the
        // `Option<PrefetchEnv>` plumbed into producers is always
        // `None`. Task-7 will rip the rest once decode A/B confirms.
        let Self {
            wf,
            backend,
            moe_buffers,
            experts,
            layer_states,
            layer_caches,
            linear_buffers,
            linear_attn_graph_scratch,
            full_attn_graph_scratch,
            moe_graph_scratch,
            hidden_double_buffer,
            head_tail_scratch,
            deferred,
            prefetch,
            ..
        } = self;
        let backend =
            backend.as_mut().expect("ensure_linear_resources");
        // S10b-1a-ii: parts_mut destructure is now scoped per-branch
        // (full-attn branch inside the layer loop, post-loop block).
        // The linear-attn branch takes &mut backend directly so the
        // producer can `backend.execute(&graph)` itself in S10b-1b.
        let layer_caches =
            layer_caches.as_ref().expect("ensure_linear_resources");
        let linear_buffers =
            linear_buffers.as_mut().expect("ensure_linear_resources");
        let linear_attn_graph_scratch = linear_attn_graph_scratch
            .as_ref()
            .expect("ensure_linear_resources");
        let full_attn_graph_scratch = full_attn_graph_scratch
            .as_ref()
            .expect("ensure_linear_resources");
        let moe_graph_scratch = moe_graph_scratch
            .as_ref()
            .expect("ensure_linear_resources");
        let hidden_double_buffer = hidden_double_buffer
            .as_ref()
            .expect("ensure_linear_resources");
        let head_tail_scratch = head_tail_scratch
            .as_ref()
            .expect("ensure_linear_resources");
        let moe_buffers =
            moe_buffers.as_mut().expect("ensure_linear_resources");

        // Drain any stale deferred state — neither batched path uses
        // the ring, but a prior per-token oracle call on the same
        // RsCtx may have left in-flight K-expert dispatches that the
        // batched path's MoE permute-fuse would race with. Defensive.
        moe::deferred::discard_deferred_experts_in(deferred);

        // The batched orchestrator never fires prefetch — eval_token
        // routes through the per-token oracle (see session-5 revert),
        // and at N > 1 the bucket spans multiple unique experts per
        // layer which the per-token K-slot prefetch state machine
        // can't satisfy. Drain any in-flight prefetch from a prior
        // eval_token so the state machine stays consistent for the
        // next eval_token's re-prime via record_actuals.
        //
        // The `prefetch_enabled` gate was tied to `pread_mode` until
        // the 2026-05-20 pread teardown; the dead `if prefetch_enabled`
        // dispatch branch in this loop was removed at the same time.
        // The per-token oracle keeps its own prefetch dispatch
        // (mod.rs around line 2565) for the decode path — pending the
        // task-7 A/B on whether it actually helps decode.
        prefetch.drain();

        // Prefill arc Phase 4 — GPU embedding gather. Upload the N
        // token ids (N×4 bytes), then one `EmbedGatherNTokens` Op
        // dequantizes each token's row of `model.embed_tokens.*`
        // straight into `hidden_a` — no host stack, no n×hidden_dim
        // upload. `io::embedding::embed_lookup` stays the CPU oracle.
        //
        // hidden_a / hidden_b are run-lifetime scratch BufIds
        // (allocated once at max chunk width); the orchestrator owns
        // the alternating pair and swaps it between layers. The
        // embed gather writes a prefix (the max-chunk tail is unused
        // for n < BATCHED_CHUNK_SIZE).
        let (mut hidden_a_id, mut hidden_b_id) =
            (hidden_double_buffer.hidden_a, hidden_double_buffer.hidden_b);
        // The `embed_gather_4bit` kernel does not range-check token
        // ids; the orchestrator does, before the upload.
        for &tok in tokens {
            if tok < 0 || tok as usize >= v.vocab_size {
                return Err(RsError::EvalFailed);
            }
        }
        let embed_off = |name: &str| -> Result<u64, RsError> {
            wf.tensor_info(name)
                .map(|t| t.offset)
                .ok_or(RsError::EvalFailed)
        };
        let embed_weight = WeightRef {
            w_off: embed_off("model.embed_tokens.weight")?,
            s_off: embed_off("model.embed_tokens.scales")?,
            b_off: embed_off("model.embed_tokens.biases")?,
            bits: 4,
        };
        backend
            .pool_mut()
            .upload(head_tail_scratch.token_ids, bytemuck::cast_slice(tokens))
            .expect("pool upload token ids");
        let mut g_head = Graph::new();
        // The head writes the embedding-gather output into the
        // run-lifetime `hidden_a` slot of the cross-layer double-
        // buffer (`BufId<HiddenBuf>`). The Op's `hidden_out` slot is
        // typed `BufId<EmbedOutBuf>` — same physical role at the
        // top of the model, with a dedicated `From<HiddenBuf> for
        // BufId<EmbedOutBuf>` impl that documents the bridge here.
        g_head.push(Op::EmbedGatherNTokens {
            label: "embed_gather",
            token_ids: head_tail_scratch.token_ids,
            weight: embed_weight,
            hidden_out: hidden_a_id.into(),
            hidden_dim: hidden_dim as u32,
            n_tokens: n as u32,
        });
        backend
            .execute(&g_head, "graph_head")
            .map_err(|_| RsError::EvalFailed)?;

        for layer_idx in 0..v.num_layers {
            backend.begin_layer(chunk_idx, layer_idx);
            // Prefetch is never fired by the batched orchestrator
            // (see the drain() above this loop and its surrounding
            // comment). `PrefetchEnv` is always None on this path —
            // producers' `bucket_prefetch_slot` cold-correctness
            // branches that key on `Some(prefetch_env)` are dead
            // code post-2026-05-20 pread teardown and will be
            // pruned alongside the task-7 A/B.
            let prefetch_env: Option<attn::linear_attn_forward::PrefetchEnv> =
                None;
            let is_full = v.layer_kind(layer_idx)
                == variants::LayerKind::FullAttn;
            if is_full {
                let kv_state = full_kv_mut(layer_states, layer_idx)?;
                // Prefill arc Phase 2: full-attn now mirrors the
                // linear-attn branch — `&mut backend` + BufIds +
                // graph scratch; the producer resolves parts via
                // `parts_mut` internally.
                batched_full_attn_layer_forward(
                    backend,
                    wf,
                    &layer_caches[layer_idx],
                    layer_idx,
                    start_pos,
                    n,
                    k_active,
                    experts,
                    &mut *moe_buffers,
                    kv_state,
                    prefetch_env,
                    hidden_a_id,
                    hidden_b_id,
                    full_attn_graph_scratch,
                    moe_graph_scratch,
                )
                .map_err(|_| RsError::EvalFailed)?;
            } else {
                let layer_state = match &mut layer_states[layer_idx] {
                    LayerState::LinearAttn(la) => la,
                    LayerState::FullAttn(_) | LayerState::Mla(_) => {
                        return Err(RsError::EvalFailed);
                    }
                };
                // Session-5 Phase 1: batched linear-attn forward. The
                // per-token fallback (previously a loop over
                // `linear_attn_layer_forward` + per-token deferred
                // drain) was 55% of prefill inclusive time per the
                // post-session-4 profile. The batched form keeps the
                // 5 recurrent kernels per-token but folds N×5
                // dispatches into one cmdbuf, batches the 4
                // projections + o_proj + shared FFN, and routes MoE
                // through the same permute-fuse path as the full-attn
                // batched forward.
                // S10b-1a-ii: passes &mut backend + BufIds; the
                // producer resolves parts via parts_mut internally
                // so S10b-1b can replace its imperative pre-MoE chain
                // with backend.execute(&graph) without orchestrator
                // changes.
                attn::linear_attn_forward::batched_linear_attn_layer_forward(
                    backend,
                    wf,
                    &layer_caches[layer_idx],
                    linear_buffers,
                    layer_idx,
                    n,
                    k_active,
                    experts,
                    &mut *moe_buffers,
                    layer_state,
                    prefetch_env,
                    hidden_a_id,
                    hidden_b_id,
                    linear_attn_graph_scratch,
                    moe_graph_scratch,
                )
                .map_err(|_| RsError::EvalFailed)?;
            }
            // Swap GPU buffers: this layer wrote `hidden_b`; next
            // layer reads it as `hidden_a` and writes the previously-
            // input buffer. Swapping BufIds (not buffer handles) is
            // safe because we re-fetch via `pool.handle(...)` each
            // iteration.
            std::mem::swap(&mut hidden_a_id, &mut hidden_b_id);
        }
        // Safety net for the gpu_capture hook: if the window extends
        // past the chunk's last layer (e.g. user set n_layers too
        // large), stop here so the trace still gets written.
        if gpu_capture::config().is_some() {
            gpu_capture::stop();
        }

        // Prefill arc Phase 4 — GPU final norm + lm_head. The layer
        // loop's post-iteration swap leaves `hidden_a_id` on the
        // buffer the last layer wrote; `hidden_b_id` is the *other*
        // physical buffer — no layer reads it again, so it is free to
        // norm into. (Use the post-loop local, never the struct
        // field: under odd num_layers `hidden_double_buffer.hidden_b`
        // aliases the final hidden state we still need to read.)
        //
        // `Op::LmHead`'s job is exactly RmsNorm + Matvec, and both
        // are already wired — norm `hidden_a -> hidden_b`, then an
        // lm_head matvec of the last token's row produces the logits.
        if let Some(logits) = logits_out {
            let norm_off = wf
                .tensor_info("model.norm.weight")
                .map(|t| t.offset)
                .ok_or(RsError::EvalFailed)?;
            let lm_off = |name: &str| -> Result<u64, RsError> {
                wf.tensor_info(name)
                    .map(|t| t.offset)
                    .ok_or(RsError::EvalFailed)
            };
            let lm_head_weight = WeightRef {
                w_off: lm_off("lm_head.weight")?,
                s_off: lm_off("lm_head.scales")?,
                b_off: lm_off("lm_head.biases")?,
                bits: 4,
            };
            let mut g_tail = Graph::new();
            g_tail.push(Op::RmsNormBf16NTokens {
                label: "final_norm",
                // Last layer wrote into `hidden_a`; norm reads it
                // (HiddenBuf → RmsNormIn) and writes the post-norm
                // value into `hidden_b` (HiddenBuf → TailNormedBuf →
                // RmsNormOut, via the union impl).
                x: hidden_a_id.into(),
                weight_off: norm_off,
                out: {
                    // Two-step to RmsNormOut: HiddenBuf → TailNormedBuf
                    // → RmsNormOut. Documented in buftype.rs above
                    // the `HiddenBuf -> TailNormedBuf` impl.
                    let tail: crate::riir::backend::BufId<
                        crate::riir::backend::buftype::TailNormedBuf,
                    > = hidden_b_id.into();
                    tail.into()
                },
                dim: hidden_dim as u32,
                n_tokens: n as u32,
                eps: variants::RMS_NORM_EPS,
            });
            g_tail.push(Op::MatvecNTokens {
                label: "lm_head",
                weight: lm_head_weight,
                // lm_head matvec reads the tail-normed buffer (we just
                // wrote it through `final_norm`). HiddenBuf →
                // TailNormedBuf → MatvecIn via the `TailNormedBuf ->
                // MatvecIn` union impl.
                input: {
                    let tail: crate::riir::backend::BufId<
                        crate::riir::backend::buftype::TailNormedBuf,
                    > = hidden_b_id.into();
                    tail.into()
                },
                input_off: ((n - 1)
                    * hidden_dim
                    * std::mem::size_of::<f32>())
                    as u64,
                output: head_tail_scratch.logits.into(),
                output_off: 0,
                in_dim: hidden_dim as u32,
                out_dim: v.vocab_size as u32,
                n_tokens: 1,
            });
            backend
                .execute(&g_tail, "graph_tail")
                .map_err(|_| RsError::EvalFailed)?;
            backend
                .pool()
                .download(
                    head_tail_scratch.logits,
                    bytemuck::cast_slice_mut(logits),
                )
                .expect("pool download logits");
        }
        // Release transient BufIds so the pool doesn't grow across
        // steps. The head/tail graphs use only persistent BufIds.
        backend.pool_mut().reset_transient();
        Ok(())
    }

    /// Per-token CPU MLA forward for DeepSeek-V3 / Cogito-V2. Pure
    /// host-side compute except for the final `lm_head` matvec. No
    /// deferred dispatch, no GPU pipeline — just embed → 61× layers
    /// → final norm → GPU lm_head.
    ///
    /// Each layer runs the standard transformer block: pre-norm →
    /// MLA → +residual → post-norm → MLP-or-MoE → +residual. Layers
    /// `[0, first_k_dense_replace)` use the dense MLP; the rest use
    /// the routed-MoE path with shared expert added unconditionally.
    ///
    /// This is the baseline path for first-run validation. The folded
    /// MLA form (q' = q_nope @ kv_b_proj, then `q' · latent_j` per
    /// cached position) and a GPU MLA kernel are follow-up slices.
    fn step_internal_mla_cpu(
        &mut self,
        token: i32,
        pos: i32,
        logits_out: Option<&mut [f32]>,
    ) -> Result<(), RsError> {
        use attn::mla_attn_cpu::mla_attn_layer_forward_cpu;
        use moe::mlp_cpu::dense_mlp_swiglu_cpu;
        use moe::moe_cpu::deepseek_moe_cpu;
        use attn::rope::{compute_yarn_inv_freq, yarn_get_mscale_full};
        use variants::ROPE_THETA;

        self.ensure_mla_resources()?;
        let v = VARIANT;

        let Self {
            wf,
            backend,
            experts,
            layer_states,
            lm_head_gpu,
            ..
        } = self;
        let backend = backend.as_mut().expect("ensure_mla_resources");
        let (metal, wf_buf, _) = backend.parts_mut();
        let lm_head_gpu =
            lm_head_gpu.as_ref().expect("ensure_mla_resources");

        // YaRN constants — recomputed per token; cache-worthy if perf
        // matters but it's microseconds vs the per-layer matvec cost.
        let yarn_inv_freq = compute_yarn_inv_freq(
            v.qk_rope_head_dim,
            ROPE_THETA,
            v.yarn_factor,
            v.yarn_original_max_pos as f32,
            v.yarn_beta_fast,
            v.yarn_beta_slow,
        );
        let yarn_mscale = yarn_get_mscale_full(
            v.yarn_factor,
            v.yarn_mscale,
            v.yarn_mscale_all_dim,
        );

        // Embed → host hidden buffer.
        let mut hidden = vec![0.0f32; v.hidden_dim];
        io::embedding::embed_lookup(wf, token, &mut hidden)
            .map_err(|_| RsError::EvalFailed)?;

        // Per-layer scratch.
        let mut residual = vec![0.0f32; v.hidden_dim];
        let mut normed = vec![0.0f32; v.hidden_dim];
        let mut block_out = vec![0.0f32; v.hidden_dim];

        for layer_idx in 0..v.num_layers {
            // ---- Attention sub-block: residual = h; h = h + mla(norm(h)) ----
            residual.copy_from_slice(&hidden);
            let pre_norm_name =
                format!("model.layers.{layer_idx}.input_layernorm.weight");
            rms_norm_cpu(wf, &pre_norm_name, &hidden, &mut normed)
                .map_err(|_| RsError::EvalFailed)?;

            let kv_cache = match &mut layer_states[layer_idx] {
                LayerState::Mla(c) => c,
                LayerState::FullAttn(_) | LayerState::LinearAttn(_) => {
                    return Err(RsError::EvalFailed);
                }
            };
            mla_attn_layer_forward_cpu(
                wf,
                layer_idx,
                pos,
                &normed,
                kv_cache,
                &yarn_inv_freq,
                yarn_mscale,
                &mut block_out,
            )
            .map_err(|_| RsError::EvalFailed)?;
            for i in 0..v.hidden_dim {
                hidden[i] = residual[i] + block_out[i];
            }

            // ---- MLP sub-block: residual = h; h = h + mlp(norm(h)) ----
            residual.copy_from_slice(&hidden);
            let post_norm_name = format!(
                "model.layers.{layer_idx}.post_attention_layernorm.weight"
            );
            rms_norm_cpu(wf, &post_norm_name, &hidden, &mut normed)
                .map_err(|_| RsError::EvalFailed)?;

            if layer_idx < v.first_k_dense_replace {
                dense_mlp_swiglu_cpu(wf, layer_idx, &normed, &mut block_out)
                    .map_err(|_| RsError::EvalFailed)?;
            } else {
                deepseek_moe_cpu(
                    wf, experts, layer_idx, &normed, &mut block_out,
                )
                .map_err(|_| RsError::EvalFailed)?;
            }
            for i in 0..v.hidden_dim {
                hidden[i] = residual[i] + block_out[i];
            }
        }

        // Final RMSNorm.
        let mut hidden_normed = vec![0.0f32; v.hidden_dim];
        rms_norm_cpu(wf, "model.norm.weight", &hidden, &mut hidden_normed)
            .map_err(|_| RsError::EvalFailed)?;

        // LM head — GPU (the only GPU dispatch on this path). Skipped
        // when `logits_out` is None (prompt-prefix step).
        if let Some(logits) = logits_out {
            lm_head_gpu
                .forward(metal, wf_buf, &hidden_normed, logits)
                .map_err(|_| RsError::EvalFailed)?;
        }

        Ok(())
    }

    /// Per-token GPU MLA forward — same shape as
    /// [`Self::step_internal_mla_cpu`] but runs the attention block on
    /// Metal via [`attn::mla_attn_forward::mla_attn_layer_forward_gpu`].
    /// Dense MLP / MoE remain on CPU for first run; the GPU bounce
    /// per layer is one shared-storage memcpy of `hidden_dim` floats,
    /// cheap relative to the projections + SDPA we just moved off the
    /// CPU. Full GPU MoE integration is a follow-up perf slice.
    fn step_internal_mla_gpu(
        &mut self,
        token: i32,
        pos: i32,
        logits_out: Option<&mut [f32]>,
    ) -> Result<(), RsError> {
        use moe::cogito_moe_gpu::cogito_moe_layer_forward_gpu_buf_io;
        use backend::gpu::dense_mlp_gpu::encode_dense_mlp_layer_forward_gpu;
        use backend::gpu::gpu_norm::{
            encode_buffer_copy_f32, encode_residual_add_into,
            encode_rms_norm_bf16_into,
        };
        use attn::mla_attn_forward::mla_attn_layer_forward_gpu;

        self.ensure_mla_resources()?;
        let v = VARIANT;

        let Self {
            wf,
            backend,
            experts,
            layer_states,
            lm_head_gpu,
            mla_buffers,
            mla_yarn,
            mla_pipes,
            dense_mlp_bufs,
            dense_mlp_pipes,
            shared_expert_bufs,
            bf_matvec_pipes,
            moe_buffers,
            mla_residual_scratch,
            mla_norm_pipes,
            residual_add_pipe,
            io_pool,
            ..
        } = self;
        let backend = backend.as_mut().expect("ensure_mla_resources");
        let (metal, wf_buf, pool) = backend.parts_mut();
        let lm_head_gpu =
            lm_head_gpu.as_ref().expect("ensure_mla_resources");
        let mla_buffers =
            mla_buffers.as_mut().expect("ensure_mla_resources");
        let mla_yarn = mla_yarn.as_ref().expect("ensure_mla_resources");
        let mla_pipes = mla_pipes.as_ref().expect("ensure_mla_resources");
        let dense_mlp_bufs =
            dense_mlp_bufs.as_mut().expect("ensure_mla_resources");
        let dense_mlp_pipes =
            dense_mlp_pipes.as_ref().expect("ensure_mla_resources");
        let shared_expert_bufs =
            shared_expert_bufs.as_ref().expect("ensure_mla_resources");
        let bf_matvec_pipes =
            bf_matvec_pipes.as_ref().expect("ensure_mla_resources");
        let moe_buffers =
            moe_buffers.as_mut().expect("ensure_mla_resources");
        let scratch =
            mla_residual_scratch.as_ref().expect("ensure_mla_resources");
        let norm_pipes =
            mla_norm_pipes.as_ref().expect("ensure_mla_resources");
        let residual_add_pipe = residual_add_pipe
            .as_ref()
            .expect("ensure_mla_resources");

        // Embed → host hidden vec → GPU scratch.hidden buffer (one
        // host->GPU bounce per token; the layer loop is fully GPU
        // resident from here).
        let mut hidden_host = vec![0.0f32; v.hidden_dim];
        io::embedding::embed_lookup(wf, token, &mut hidden_host)
            .map_err(|_| RsError::EvalFailed)?;
        // SAFETY: shared-storage GPU buffer; no GPU work in flight at
        // top of step (caller honors the moeflux.h:481 contract).
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden_host.as_ptr(),
                scratch.hidden.contents() as *mut f32,
                v.hidden_dim,
            );
        }

        let dim = v.hidden_dim as u32;

        for layer_idx in 0..v.num_layers {
            // ---- Pre-attn block: residual := hidden, normed := rms_norm(hidden) ----
            // Single cmdbuf for the residual snapshot + pre-norm. We
            // commit + wait here because mla_attn_layer_forward_gpu
            // (called next) builds its own cmdbuf and the synchronous
            // `mla_buffers.pre_norm` host stage further down would race
            // with our normed write if we didn't wait.
            let pre_norm_name = format!(
                "model.layers.{layer_idx}.input_layernorm.weight"
            );
            let pre_norm_off = wf_buf
                .tensor_offset(wf, &pre_norm_name)
                .map_err(|_| RsError::EvalFailed)?
                .ok_or(RsError::EvalFailed)?;
            {
                let cmdbuf = metal.queue().new_command_buffer();
                encode_buffer_copy_f32(
                    cmdbuf,
                    &scratch.hidden,
                    &scratch.residual,
                    dim,
                );
                encode_rms_norm_bf16_into(
                    cmdbuf,
                    norm_pipes,
                    &scratch.hidden,
                    wf_buf.buffer(),
                    pre_norm_off,
                    &scratch.sum_sq,
                    &scratch.normed,
                    dim,
                    variants::RMS_NORM_EPS,
                );
                // Mirror normed into `mla_buffers.pre_norm` so the
                // existing `mla_attn_layer_forward_gpu` (which reads
                // from there) sees our normed input. Phase 5b refactor
                // would parameterize mla_attn's input buffer; for now
                // we add one GPU buffer-copy per layer.
                encode_buffer_copy_f32(
                    cmdbuf,
                    &scratch.normed,
                    &mla_buffers.pre_norm,
                    dim,
                );
                cmdbuf.commit();
                cmdbuf.wait_until_completed();
            }

            let kv_cache = match &mut layer_states[layer_idx] {
                LayerState::Mla(c) => c,
                LayerState::FullAttn(_) | LayerState::LinearAttn(_) => {
                    return Err(RsError::EvalFailed);
                }
            };
            mla_attn_layer_forward_gpu(
                metal,
                mla_pipes,
                wf,
                wf_buf,
                mla_yarn,
                mla_buffers,
                kv_cache,
                layer_idx,
                pos,
            )
            .map_err(|_| RsError::EvalFailed)?;

            // Post-attn residual_add: hidden := residual + mla_buffers.out.
            {
                let cmdbuf = metal.queue().new_command_buffer();
                encode_residual_add_into(
                    cmdbuf,
                    residual_add_pipe,
                    &scratch.residual,
                    &mla_buffers.out,
                    &scratch.hidden,
                    dim,
                );
                cmdbuf.commit();
                cmdbuf.wait_until_completed();
            }

            // ---- Pre-MLP block: residual := hidden, normed := rms_norm(hidden) ----
            let post_norm_name = format!(
                "model.layers.{layer_idx}.post_attention_layernorm.weight"
            );
            let post_norm_off = wf_buf
                .tensor_offset(wf, &post_norm_name)
                .map_err(|_| RsError::EvalFailed)?
                .ok_or(RsError::EvalFailed)?;
            {
                let cmdbuf = metal.queue().new_command_buffer();
                encode_buffer_copy_f32(
                    cmdbuf,
                    &scratch.hidden,
                    &scratch.residual,
                    dim,
                );
                encode_rms_norm_bf16_into(
                    cmdbuf,
                    norm_pipes,
                    &scratch.hidden,
                    wf_buf.buffer(),
                    post_norm_off,
                    &scratch.sum_sq,
                    &scratch.normed,
                    dim,
                    variants::RMS_NORM_EPS,
                );
                cmdbuf.commit();
                cmdbuf.wait_until_completed();
            }

            // ---- MLP / MoE: read scratch.normed, write to mlp output buf ----
            // The post-MLP residual_add reads from whichever buffer the
            // dispatch wrote to (dense_mlp_bufs.out for dense, or
            // moe_buffers.moe_hidden for MoE).
            let mlp_out: &Buffer = if layer_idx < v.first_k_dense_replace
            {
                {
                    let cmdbuf = metal.queue().new_command_buffer();
                    encode_dense_mlp_layer_forward_gpu(
                        cmdbuf,
                        dense_mlp_pipes,
                        wf,
                        wf_buf,
                        layer_idx,
                        &scratch.normed,
                        &dense_mlp_bufs.gate_out,
                        &dense_mlp_bufs.up_out,
                        &dense_mlp_bufs.act,
                        &dense_mlp_bufs.out,
                    )
                    .map_err(|_| RsError::EvalFailed)?;
                    cmdbuf.commit();
                    cmdbuf.wait_until_completed();
                }
                &dense_mlp_bufs.out
            } else {
                cogito_moe_layer_forward_gpu_buf_io(
                    metal,
                    moe_buffers,
                    pool,
                    shared_expert_bufs,
                    dense_mlp_pipes,
                    bf_matvec_pipes,
                    wf,
                    wf_buf,
                    experts,
                    io_pool,
                    layer_idx,
                    &scratch.normed,
                )
                .map_err(|_| RsError::EvalFailed)?;
                moe_buffers.moe_hidden_ref(pool)
            };

            // Post-MLP residual_add: hidden := residual + mlp_out.
            {
                let cmdbuf = metal.queue().new_command_buffer();
                encode_residual_add_into(
                    cmdbuf,
                    residual_add_pipe,
                    &scratch.residual,
                    mlp_out,
                    &scratch.hidden,
                    dim,
                );
                cmdbuf.commit();
                cmdbuf.wait_until_completed();
            }
        }

        // Final RMSNorm + GPU lm_head. Stage hidden GPU → host → final
        // norm CPU → lm_head host slice. lm_head_gpu currently takes a
        // host slice; making it Buffer-IO is a follow-up.
        let final_norm_name = "model.norm.weight";
        // SAFETY: shared-storage; no GPU work in flight (the last
        // residual_add committed + waited above).
        let hidden_host: Vec<f32> = unsafe {
            let p = scratch.hidden.contents() as *const f32;
            std::slice::from_raw_parts(p, v.hidden_dim).to_vec()
        };
        let mut hidden_normed = vec![0.0f32; v.hidden_dim];
        rms_norm_cpu(wf, final_norm_name, &hidden_host, &mut hidden_normed)
            .map_err(|_| RsError::EvalFailed)?;
        if let Some(logits) = logits_out {
            lm_head_gpu
                .forward(metal, wf_buf, &hidden_normed, logits)
                .map_err(|_| RsError::EvalFailed)?;
        }
        Ok(())
    }

    /// Per-token forward orchestrator. Mirrors C `mf_step_internal`
    /// (infer.m:7687..7721): embed → layer loop → optional drain +
    /// final norm + lm_head. If `logits_out` is `Some`, the deferred
    /// dispatch from the final layer is drained, the result is
    /// `model.norm`-normalized CPU-side, and the LM head writes the
    /// vocabulary-size logits buffer. If `None`, the deferred
    /// dispatch is discarded and no logits are produced.
    ///
    /// Slice 4f-3 made `post_attention_tail` async; this orchestrator
    /// drains the previous layer's dispatch at the top of each
    /// iteration (no-op on iteration 0). Drain target is
    /// `linear_buffers.input` so the next layer's CPU input rms_norm
    /// reads the correct hidden state. The final drain (after the
    /// loop) writes into a host scratch so the model.norm + lm_head
    /// pair don't have to share the GPU buffer.
    ///
    /// ## Role: per-token oracle for the batched-prefill path
    ///
    /// This is the tokenwise forward — it processes exactly one token
    /// per call. The canonical multi-token orchestrator
    /// [`Self::step_internal`] takes a slice and, after the
    /// session-4 batched-primitive integration, will use GPU batched
    /// kernels (MoE permute-and-fuse, tiled SDPA, batched matmul) to
    /// amortize per-layer I/O across the prompt. This function stays
    /// as the diff oracle so any future divergence in the batched
    /// path is caught against the tokenwise reference.
    ///
    /// Production callers: [`Self::eval_token`] (decode) routes here
    /// directly. The slice-taking [`Self::step_internal`] currently
    /// wraps this in a loop.
    pub(crate) fn step_internal_per_token_oracle(
        &mut self,
        token: i32,
        pos: i32,
        logits_out: Option<&mut [f32]>,
    ) -> Result<(), RsError> {
        let v = VARIANT;
        if pos < 0 {
            return Err(RsError::EvalFailed);
        }
        if let Some(ref l) = logits_out {
            if l.len() != v.vocab_size {
                return Err(RsError::EvalFailed);
            }
        }

        // MLA variants (DeepSeek-V3 / Cogito-V2) run a separate
        // pipeline that bypasses the GPU GQA dispatch. The GPU
        // attention path is the default; setting
        // `MOEFLUX_FORCE_CPU_MLA=1` falls back to the original
        // host-only path (used as a diff oracle and as a last-resort
        // fallback if the GPU path regresses).
        if matches!(v.attn_kind, variants::AttnKind::Mla) {
            let force_cpu =
                std::env::var_os("MOEFLUX_FORCE_CPU_MLA").is_some();
            return if force_cpu {
                self.step_internal_mla_cpu(token, pos, logits_out)
            } else {
                self.step_internal_mla_gpu(token, pos, logits_out)
            };
        }

        self.ensure_linear_resources()?;
        let k_active = self.k_active;

        // Field-disjoint mutable borrows for the layer loop. Same
        // pattern as `layer_forward_dump_inner`.
        let Self {
            wf,
            backend,
            moe_buffers,
            experts,
            layer_states,
            layer_caches,
            linear_buffers,
            deferred,
            lm_head_gpu,
            io_pool,
            prefetch,
            ..
        } = self;
        let backend =
            backend.as_mut().expect("ensure_linear_resources");
        let (metal, wf_buf, buffer_pool) = backend.parts_mut();
        let layer_caches =
            layer_caches.as_ref().expect("ensure_linear_resources");
        let linear_buffers =
            linear_buffers.as_mut().expect("ensure_linear_resources");
        let moe_buffers =
            moe_buffers.as_mut().expect("ensure_linear_resources");
        let lm_head_gpu =
            lm_head_gpu.as_ref().expect("ensure_linear_resources");
        let io_pool: &rayon::ThreadPool = &*io_pool;

        // Defensive bracket — drain stale state from a buggy prior
        // call so re-entrancy holds.
        moe::deferred::discard_deferred_experts_in(deferred);

        // Embed token into the persistent input buffer in-place.
        // SAFETY: shared-storage buffer; no GPU work is in flight
        // because we just discarded any deferred state.
        {
            let buf_input_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    buffer_pool.handle(linear_buffers.input).contents() as *mut f32,
                    v.hidden_dim,
                )
            };
            io::embedding::embed_lookup(wf, token, buf_input_slice)
                .map_err(|_| RsError::EvalFailed)?;
        }

        // Per-layer loop. Slice 5d-9 widened the deferred slot to a
        // depth-2 ring and dropped the explicit per-layer wait — the
        // CPU now submits layer N+1's CMD1+CMD2+chain without waiting
        // for layer N's K-expert. Metal queue serialization on a
        // single command queue ensures N+1's CMD1 reads
        // `linear_buffers.normed` only after N's chain wrote it.
        //
        // The wait collapses into ring-cleanup: drain the oldest
        // dispatch only when the ring is at capacity. With depth 2,
        // entering layer N drains layer N-2's dispatch — by then it's
        // had ~2 layer-times of GPU runtime so the wait is short or
        // zero.
        //
        // Soundness for prefetch: layer N's prefetch wrote set
        // `N % 2`; layer N+1 writes the OTHER set (no race with N's
        // GPU read). Layer N+2's prefetch writes set `N % 2` again,
        // but by then layer N has been drained — set `N % 2` is
        // free. See `MoeBuffers::data_prefetch` docs.
        //
        // gpu_combine = true everywhere preserves the slice 4f-3
        // production behavior. Every non-last layer chains (slice
        // 5d-8) so the drained N-2 dispatch is always chained — no
        // host readback needed during the loop.
        let mut prev_layer_chained = false;
        for layer_idx in 0..v.num_layers {
            if let Some(cfg) = gpu_capture::config() {
                if cfg.decode_start(layer_idx) {
                    gpu_capture::start(metal.device(), cfg);
                } else if cfg.decode_stop(layer_idx) {
                    gpu_capture::stop();
                }
            }

            if deferred.is_full() {
                moe::deferred::complete_deferred_experts_chained(deferred)
                    .map_err(|_| RsError::EvalFailed)?;
            }

            // Slice 5d-6b: kick off async prefetch for THIS layer
            // using its prediction (= last token's same-layer
            // indices). Runs concurrently with this layer's
            // CMD1+CMD2 GPU compute and finishes before the K-expert
            // dispatch (which `wait_for`s inside post_attention_tail).
            //
            // Ordering is load-bearing — the prefetch must run AFTER
            // layer N-1's deferred drain (above) so the previous
            // layer's GPU read of data_prefetch[slot] is complete
            // before we overwrite it. First token has no predictions,
            // skip the fire (predict_for returns None).
            // Slice 5d-9: layer N writes set `N % 2` so layer N+1's
            // prefetch (which writes the OTHER set) doesn't race layer
            // N's GPU read of set `N % 2`. The encoder for layer N
            // reads from the same set this prefetch wrote to.
            let prefetch_set = layer_idx % 2;
            if let Some(predicted) = prefetch.predict_for(layer_idx) {
                let data_prefetch = moe_buffers
                    .data_prefetch_slots_mut_array(buffer_pool, prefetch_set);
                prefetch.dispatch(
                    layer_idx,
                    predicted,
                    k_active,
                    data_prefetch,
                    io_pool,
                    experts,
                );
            }

            // Slice 5d-8: chain enabled for every non-last layer
            // (gpu_combine is hardcoded true here). Last layer's
            // combine still writes to bufs.moe_hidden so the post-loop
            // final drain reads from it as before.
            let chain_next = layer_idx + 1 < v.num_layers;
            let chain_next_norm_off = chain_next.then(|| {
                layer_caches[layer_idx + 1].input_layernorm_w
            });

            let is_full = v.layer_kind(layer_idx)
                == variants::LayerKind::FullAttn;
            if is_full {
                let kv_state = full_kv_mut(layer_states, layer_idx)?;
                let layer_ctx = backend::gpu::gpu_ctx::GpuLayerCtx {
                    wf,
                    wf_buf,
                    layer_cache: &layer_caches[layer_idx],
                    buffers: linear_buffers,
                    buffer_pool,
                };
                full_attn_layer_forward(
                    metal,
                    &layer_ctx,
                    moe_buffers,
                    deferred,
                    layer_idx,
                    pos,
                    k_active,
                    experts,
                    io_pool,
                    prefetch,
                    prefetch_set,
                    kv_state,
                    /* gpu_combine = */ true,
                    prev_layer_chained,
                    chain_next_norm_off,
                )
                .map_err(|_| RsError::EvalFailed)?;
            } else {
                let layer_state = match &mut layer_states[layer_idx] {
                    LayerState::LinearAttn(la) => la,
                    LayerState::FullAttn(_) | LayerState::Mla(_) => {
                        return Err(RsError::EvalFailed);
                    }
                };
                let layer_ctx = backend::gpu::gpu_ctx::GpuLayerCtx {
                    wf,
                    wf_buf,
                    layer_cache: &layer_caches[layer_idx],
                    buffers: linear_buffers,
                    buffer_pool,
                };
                linear_attn_layer_forward(
                    metal,
                    &layer_ctx,
                    moe_buffers,
                    deferred,
                    layer_idx,
                    k_active,
                    experts,
                    io_pool,
                    prefetch,
                    prefetch_set,
                    layer_state,
                    /* gpu_combine = */ true,
                    prev_layer_chained,
                    chain_next_norm_off,
                )
                .map_err(|_| RsError::EvalFailed)?;
            }
            prev_layer_chained = chain_next;
        }

        if gpu_capture::config().is_some() {
            gpu_capture::stop();
        }

        // Slice 5d-9 post-loop drain. With the depth-2 ring, the
        // tail of the loop typically leaves [layer N-1, layer N]
        // in flight, where N = num_layers - 1 is the unchained last
        // layer and N-1 is chained (chain wrote `linear_buffers.normed`
        // for what would be layer N+1 — but there is no layer N+1, so
        // the chained-write is harmless and we just drain it without
        // readback). Layer N's combine wrote into `bufs.moe_hidden`
        // (chain disabled at last layer), so its drain readback is
        // canonical for the LM head.
        //
        // For 1-layer models the ring has only [layer 0] and the
        // while-loop is a no-op; for 2-layer models the ring has
        // [layer 0 chained, layer 1 unchained] and the while-loop
        // drains layer 0 once. Generalizes to N layers.
        while deferred.len() > 1 {
            moe::deferred::complete_deferred_experts_chained(deferred)
                .map_err(|_| RsError::EvalFailed)?;
        }
        match logits_out {
            None => {
                moe::deferred::discard_deferred_experts_in(deferred);
            }
            Some(logits) => {
                let mut hidden_final = vec![0.0f32; v.hidden_dim];
                moe::deferred::complete_deferred_experts_into(
                    deferred,
                    moe_buffers,
                    buffer_pool,
                    &mut hidden_final,
                )
                .map_err(|_| RsError::EvalFailed)?;

                // Final RMSNorm (`model.norm.weight`) is CPU — small
                // (HIDDEN_DIM = 2048), bit-exact against C per slice 1.
                let mut hidden_normed = vec![0.0f32; v.hidden_dim];
                rms_norm_cpu(
                    wf,
                    "model.norm.weight",
                    &hidden_final,
                    &mut hidden_normed,
                )
                .map_err(|_| RsError::EvalFailed)?;

                // LM head is GPU — was 59% of CPU time per the
                // 2026-04-27 profile. The C path's `lm_head_forward`
                // (infer.m:3090) takes the same Metal route through
                // `dequant_matvec_4bit_v3`; per-PSO bit-exactness
                // (slice 9 finding) keeps the end-to-end logits
                // bit-equal.
                lm_head_gpu
                    .forward(metal, wf_buf, &hidden_normed, logits)
                    .map_err(|_| RsError::EvalFailed)?;
            }
        }

        Ok(())
    }

    /// Reset every layer's state to empty. Mirrors `mf_memory_clear`
    /// (infer.m:7759 → `mf_state_clear_all` infer.m:2271). `seq_id`
    /// is unused — moeflux's sequence-id argument is a no-op stub on
    /// the C side too; KV is single-stream.
    ///
    /// Resets both the host-side per-layer state vector AND the
    /// GPU-side linear-attn recurrence buffers
    /// (`linear_buffers.conv_state` / `delta_state`). The GPU reset
    /// is required because the Rust port treats the GPU buffers as
    /// the canonical recurrence storage (kernels mutate in place,
    /// never read back to host); the C side stores recurrence on the
    /// host and pushes to GPU each call, so resetting host alone
    /// suffices there. Without the GPU reset, back-to-back forwards
    /// after `memory_clear` see stale recurrence and diverge from a
    /// freshly-allocated Ctx.
    pub fn memory_clear(&mut self) {
        clear_all(&mut self.layer_states);
        if let (Some(bufs), Some(backend)) =
            (self.linear_buffers.as_ref(), self.backend.as_ref())
        {
            let pool = backend.pool();
            bufs.reset_recurrence(pool);
            // Slice 5d-7b — zero the GPU full-attn KV mirrors
            // alongside the host-side clear. Without this, the GPU
            // SDPA fast path would read stale k/v from the previous
            // sequence at positions [0, prev_len).
            bufs.reset_gpu_attn_kv_mirrors(pool);
        }
        // Slice 5d-6b: drain any in-flight prefetch and clear all
        // last-token predictions. After memory_clear the next token
        // starts from cold-prediction state (no stale predictions
        // from a different prefix).
        self.prefetch.invalidate_all();
        // Phase 7: snapshot store is keyed by sequence position; a
        // full clear invalidates every key.
        self.checkpoints.clear();
        self.checkpoint_order.clear();
    }

    /// Drain any in-flight prefetch and clear all per-layer
    /// last-token predictions. After this call the next forward
    /// starts from cold-prediction state — every K-expert slot in
    /// every layer takes the all-miss (sync-pread) path.
    ///
    /// Slice 5d-6b. Exposed mainly for diff tests that need to force
    /// the all-miss path; production callers shouldn't need this
    /// (prefetch is a perf hint, not a correctness toggle).
    pub fn clear_prefetch_predictions(&mut self) {
        self.prefetch.invalidate_all();
    }

    /// Truncate every layer's state to positions `[0, p0)`. Linear-attn
    /// layers reset to empty (lossy — see `state` module docs and the
    /// FIXME for the Phase 7 typed-error fix). Mirrors
    /// `mf_memory_seq_rm` (infer.m:7752): always returns `true` if the
    /// ctx is valid, since the truncation primitive itself is
    /// infallible.
    pub fn memory_seq_rm(&mut self, _seq_id: i32, p0: i32, p1: i32) -> bool {
        truncate(&mut self.layer_states, p0, p1);
        true
    }

    /// Snapshot the current ctx state at sequence position `pos`.
    /// Subsequent [`Self::restore_to`] calls with the same `pos` will
    /// reload exactly this state.
    ///
    /// Internally allocates a buffer of [`Self::state_size`] bytes and
    /// calls [`Self::state_save`] into it — drains pending GPU work
    /// and serializes the full ctx (linear-attn recurrence + full-attn
    /// KV) into the wire format. Bytes-per-snapshot grow with KV
    /// length at `pos`.
    ///
    /// Eviction: if storing this snapshot pushes the count past
    /// `max_checkpoints`, evict from the LRU front while skipping
    /// `pos == 0` and the position just inserted. If a snapshot
    /// already exists at `pos`, it is overwritten (and the LRU is
    /// updated).
    ///
    /// Errors only on [`StateSnapshotError`] from `state_save` —
    /// effectively, `BuffersNotReady` if called before the first
    /// eval/`memory_clear`. Callers (drama_llama `Session`) check
    /// after the first prefill on a sequence so this should not fire
    /// in normal use.
    pub fn checkpoint_pos(
        &mut self,
        pos: i32,
    ) -> Result<(), snapshot::state_snapshot::StateSnapshotError> {
        let mut buf = vec![0u8; self.state_size()];
        self.state_save(&mut buf)?;

        match self.checkpoints.insert(pos, buf) {
            Some(_) => {
                // Overwrite: refresh LRU position.
                if let Some(idx) =
                    self.checkpoint_order.iter().position(|&p| p == pos)
                {
                    self.checkpoint_order.remove(idx);
                }
            }
            None => {}
        }
        self.checkpoint_order.push_back(pos);

        // Evict oldest until under cap, skipping pos=0 and the just-
        // inserted position. `pos=0` is kept because it's the natural
        // resume point for a fresh full-prefix reuse; the just-
        // inserted position is the freshest data we have.
        while self.checkpoints.len() > self.max_checkpoints {
            let mut evicted = false;
            for i in 0..self.checkpoint_order.len() {
                let candidate = self.checkpoint_order[i];
                if candidate == 0 || candidate == pos {
                    continue;
                }
                self.checkpoint_order.remove(i);
                self.checkpoints.remove(&candidate);
                evicted = true;
                break;
            }
            if !evicted {
                // Only pos=0 and `pos` remain (or pathological config
                // with max_checkpoints < 2). Stop — refusing to evict
                // protected entries beats silently corrupting
                // future-fast-resume.
                break;
            }
        }

        Ok(())
    }

    /// Restore ctx state to a previously-snapshotted position. After
    /// success, KV state matches what was current immediately after
    /// the [`Self::checkpoint_pos`] call at `pos`, and any snapshots
    /// stored at positions `> pos` are dropped (their futures are now
    /// invalid — the sequence is being rewritten from `pos` forward).
    ///
    /// Returns [`CheckpointError::NoCheckpoint`] if no snapshot exists
    /// at exactly `pos`. drama_llama's `Session` interprets this as a
    /// signal to fall back to full-clear + full reprefill.
    pub fn restore_to(&mut self, pos: i32) -> Result<(), CheckpointError> {
        let buf = self
            .checkpoints
            .get(&pos)
            .ok_or(CheckpointError::NoCheckpoint { pos })?
            .clone();
        self.state_load(&buf).map_err(CheckpointError::Snapshot)?;

        // Drop snapshots whose key > pos: the resuming prefill will
        // write fresh content at those positions. Keep `pos` itself
        // (we may restore here again) and any earlier snapshots.
        self.checkpoints.retain(|&k, _| k <= pos);
        self.checkpoint_order.retain(|&k| k <= pos);

        Ok(())
    }

    /// Drop the snapshot stored at `pos` from the checkpoint map and
    /// the LRU, without touching any other snapshot or the live ctx
    /// state. Idempotent: returns silently if no entry exists at `pos`.
    ///
    /// Used by drama_llama's `Session` to evict a previous "internal
    /// tip" snapshot being replaced by a fresher tip, and to prune
    /// orphan breakpoints from prior calls — both essential for
    /// keeping the most-valuable cross-agent anchor (system+tools at
    /// pos-0) alive in the bounded LRU.
    pub fn forget_pos(&mut self, pos: i32) {
        if self.checkpoints.remove(&pos).is_some() {
            if let Some(idx) =
                self.checkpoint_order.iter().position(|&p| p == pos)
            {
                self.checkpoint_order.remove(idx);
            }
        }
    }

    /// Set the snapshot count cap. Default is
    /// [`DEFAULT_MAX_CHECKPOINTS`] = 4. Lowering past the current
    /// `checkpoints.len()` triggers eviction at the next
    /// `checkpoint_pos`; this method does not retroactively evict.
    pub fn set_max_checkpoints(&mut self, n: usize) {
        self.max_checkpoints = n;
    }

    /// Current snapshot count. Useful for tests.
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Largest occupied position across full-attn layers, or `-1` if
    /// none has any entries. Mirrors `mf_memory_seq_pos_max`
    /// (infer.m:7759).
    pub fn memory_seq_pos_max(&self, _seq_id: i32) -> i32 {
        pos_max(&self.layer_states)
    }

    /// Bytes the caller must allocate to hold a snapshot of the
    /// current state. Mirrors C `mf_state_size` (infer.m:8505). Grows
    /// linearly with the largest KV length across full-attn layers;
    /// re-query after each evaluation if the state has changed.
    pub fn state_size(&self) -> usize {
        snapshot::state_snapshot::state_size(&self.layer_states)
    }

    /// Serialize the current state into `buf`. Returns the number of
    /// bytes written. Mirrors C `mf_state_save` (infer.m:8525).
    ///
    /// Drains any pending deferred K-expert dispatch first (the
    /// moeflux.h:481 contract — call only at token boundaries).
    /// Errors if `buf.len() < self.state_size()` or if `linear_
    /// buffers` aren't initialized yet (call `eval_prompt` /
    /// `eval_token` / `memory_clear` once before the first save).
    pub fn state_save(
        &mut self,
        buf: &mut [u8],
    ) -> Result<usize, snapshot::state_snapshot::StateSnapshotError> {
        // Drain deferred state so the snapshot reflects post-token
        // state, not mid-flight.
        snapshot::state_snapshot::drain_deferred(&mut self.deferred);
        // Slice 5d-6b: drain any in-flight prefetch (no contribution
        // to the snapshot — predictions are per-token, not part of
        // the wire format — but we need to quiesce the worker pool
        // before any subsequent ctx mutation).
        self.prefetch.drain();
        // linear_buffers is optional — pure-MLA variants don't have
        // LinearAttn layers and don't need it. The Option pattern
        // here lets us serve both Gqa-with-LinearAttn (Qwen) and
        // pure-Mla (Cogito-V2) variants from the same wrapper.
        snapshot::state_snapshot::state_save(
            buf,
            &self.layer_states,
            self.linear_buffers.as_ref(),
            self.backend.as_ref().map(|b| b.pool()),
        )
    }

    /// Replace current state with the one encoded in `buf`. Mirrors
    /// C `mf_state_load` (infer.m:8599). Two-pass: header + per-
    /// layer length preflight before any state is mutated; restore
    /// then memcpys into KV caches and pushes into the GPU
    /// recurrence buffers.
    ///
    /// On error the ctx state is left unchanged (preflight rejects
    /// before the destructive write).
    pub fn state_load(
        &mut self,
        buf: &[u8],
    ) -> Result<(), snapshot::state_snapshot::StateSnapshotError> {
        // Drain any pending dispatch — load overwrites the state the
        // dispatch was producing for.
        snapshot::state_snapshot::drain_deferred(&mut self.deferred);
        // Slice 5d-6b: drain any in-flight prefetch + clear
        // last-token predictions. After load, the prefix is whatever
        // the loaded snapshot represents — predictions from the
        // pre-load state would be stale.
        self.prefetch.invalidate_all();
        // Ensure linear_buffers exist so we have somewhere to push
        // the linear-attn recurrence into. Fresh-Ctx state_load
        // before any eval would otherwise hit BuffersNotReady; load
        // is supposed to be a stand-alone restoration primitive.
        // For Mla variants, ensure_linear_resources fails (no
        // linear-attn tensors); for Gqa variants we still need it to
        // populate gpu_kv_k/v mirrors. Run it best-effort and pass
        // linear_buffers as Option — the Mla path doesn't read it.
        let _ = self.ensure_linear_resources();
        // Always need a Metal device for MLA buffer alloc on load. It
        // exists if either ensure_linear_resources or any prior eval
        // ran, OR initialize a backend on demand.
        self.ensure_backend()
            .map_err(|_| snapshot::state_snapshot::StateSnapshotError::BuffersNotReady)?;
        let device = self
            .backend
            .as_ref()
            .expect("just-set")
            .metal()
            .device()
            .to_owned();
        let Self {
            layer_states,
            linear_buffers,
            backend,
            ..
        } = self;
        snapshot::state_snapshot::state_load(
            buf,
            layer_states,
            linear_buffers.as_mut(),
            backend.as_ref().map(|b| b.pool()),
            &device,
        )
    }
}

impl std::fmt::Debug for RsCtx<MetalBackend> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RsCtx")
            .field("model", &VARIANT.name)
            .field("weights", &self.wf)
            .finish()
    }
}

/// Error type for the Rust port. Mirrors [`crate::imp::Error`] until
/// Phase 7's API cleanup, at which point we'll likely refine
/// (e.g. `CannotTruncateLinear` for the typed `memory_seq_rm`).
#[derive(Debug, thiserror::Error)]
pub enum RsError {
    #[error("path contained an interior NUL byte")]
    PathHasNul,
    #[error("init failed (file missing, mmap, vocab, Metal)")]
    InitFailed,
    #[error("eval call failed")]
    EvalFailed,
    #[error("state save/load failed")]
    StateFailed,
    /// Caller-supplied buffer too small for the snapshot. Variant kept
    /// for API parity with [`crate::imp::Error`]; not yet emitted by
    /// the Rust [`RsCtx::state_save`] (which still returns
    /// `snapshot::state_snapshot::StateSnapshotError`).
    #[error("state buffer too small (have {have}, need {need})")]
    StateBufferTooSmall {
        /// Bytes the caller provided.
        have: usize,
        /// Bytes the snapshot requires.
        need: usize,
    },
    /// The opened weights don't match the model this binary was built
    /// for. A moeflux binary is feature-gated to one model
    /// (`moeflux-model-*` → compile-time [`VARIANT`]); pointing it at
    /// another model's weights otherwise loads cleanly and then
    /// panics deep in prefill. Detected by [`probe_variant_match`] at
    /// [`RsCtx::open`].
    #[error("model mismatch: binary built for {expected} — {detail}")]
    ModelMismatch {
        /// `VARIANT.name` — the model this binary was compiled for.
        expected: &'static str,
        /// What was found in the weights, and how to fix it.
        detail: String,
    },
}

#[cfg(test)]
mod variant_guard_tests {
    use super::{check_variant_dims, RsError, VARIANT};

    /// The dims this binary was actually built for pass cleanly.
    #[test]
    fn matching_dims_ok() {
        let top = VARIANT.num_layers - 1;
        assert!(
            check_variant_dims(Some(top), Some(VARIANT.hidden_dim)).is_ok()
        );
    }

    /// A manifest with no `model.layers.*` tensors is rejected.
    #[test]
    fn no_layer_tensors_rejected() {
        assert!(matches!(
            check_variant_dims(None, None),
            Err(RsError::ModelMismatch { .. })
        ));
    }

    /// Wrong layer count is rejected — guards the `top + 1` off-by-one.
    #[test]
    fn wrong_layer_count_rejected() {
        // `top == num_layers` means count `num_layers + 1` — a mismatch.
        assert!(matches!(
            check_variant_dims(
                Some(VARIANT.num_layers),
                Some(VARIANT.hidden_dim),
            ),
            Err(RsError::ModelMismatch { .. })
        ));
    }

    /// Right layer count but wrong hidden dim is rejected.
    #[test]
    fn wrong_hidden_dim_rejected() {
        assert!(matches!(
            check_variant_dims(
                Some(VARIANT.num_layers - 1),
                Some(VARIANT.hidden_dim + 1),
            ),
            Err(RsError::ModelMismatch { .. })
        ));
    }

    /// An absent `input_layernorm.weight` skips the hidden-dim check
    /// rather than failing — layer count alone still discriminates.
    #[test]
    fn absent_hidden_dim_tensor_skips_check() {
        assert!(check_variant_dims(Some(VARIANT.num_layers - 1), None).is_ok());
    }
}
