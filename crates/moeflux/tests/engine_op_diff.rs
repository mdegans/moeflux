//! Engine-level per-layer diff of [`Op::MoeBatchedPermuteFuse`] (env=off
//! path) vs [`Op::MoeGatherIdFuse`] (env=on path) against real
//! Qwen3-A3B layer-0 weights and real GPU router output on a seeded
//! synthetic `h_mid`.
//!
//! ## Why this exists — and what it actually catches
//!
//! Session 19 landed `Op::MoeGatherIdFuse` with the kernel-level diff
//! oracle green and the bench green — but the producer site passed the
//! pre-norm `h_mid` BufId into the post-norm `h_post` slot. Engine
//! output collapsed to `</think>` + EoT.
//!
//! Session B (the typed-`BufId<RoleTag>` refactor, moeflux commit
//! `12dc016`) closed the **BufId mix-up class** at compile time. The
//! literal Session-19 revert no longer compiles — there is no
//! `From<ResidualBuf> for BufId<MoeInputBuf>` impl, so feeding `h_mid`
//! to `Op::MoeGatherIdFuse::mlp_in` is a compile error.
//!
//! This test closes the **residual producer-wiring class** the type
//! system can't see: a wrong stride, a missing upload, a wrong layer
//! offset, a wrong expert base id, a layout assumption that drifted
//! from the consuming kernel. The kernel-level diff oracle in
//! `moeflux-metal/tests/gather_mm_id_diff.rs` exercises synthetic
//! everything; this test exercises real mmap'd expert weights and the
//! real Op-graph form of the router on synthetic-but-deterministic
//! hidden state — exactly the distributional shape the kernel oracle
//! cannot reach.
//!
//! ## Tombstone
//!
//! The two paths are *mathematically equivalent* at temp=0, seed-fixed
//! inputs — cosine ~1.0, max-abs-diff in f32-reduction-noise territory
//! (~1e-6 over ±1-magnitude values). They are NOT bit-identical:
//! `MoeGatherIdFuse` and `MoeBatchedPermuteFuse` execute different
//! Metal kernels with different reduction orders. The contract is
//! `cosine >= COSINE_FLOOR (0.9999)` and `rel_max_abs <= 1e-3` —
//! mirroring `tests/cogito_moe_gpu.rs:124-133`.
//!
//! Any **wire-swap that the type system does NOT catch** (e.g.
//! perturbing `n_tokens: (n_tokens - 1) as u32`, or swapping
//! `out_sum` for a different correctly-typed `BufId<MoeOutSumBuf>`
//! buffer) MUST cause cosine to drop spectacularly — the bug-of-record
//! class produces cosine ≈ 0 or argmax-flip on most tokens, not
//! f32-noise-level disagreement.
//!
//! ## Run
//!
//! ```bash
//! cargo test -p moeflux --no-default-features \
//!     --features model-qwen3-6-35b-a3b --release \
//!     --test engine_op_diff -- --ignored --nocapture
//! ```

#![cfg(all(target_os = "macos", feature = "model-qwen3-6-35b-a3b"))]

mod common;

use std::mem::size_of;

use moeflux::riir::attn::linear_attn_forward::MoeGraphScratch;
use moeflux::riir::backend::buftype::{
    BufId, ExpertBaseBuf, HiddenBuf, ResidualBuf, RouterLogitsBuf,
};
use moeflux::riir::backend::{
    Backend, BufferPool, Graph, MetalBackend, Op, WeightRef,
};
use moeflux::riir::moe::moe_router::build_expert_buckets;
use moeflux::riir::variants::{MlpKind, VARIANT, RMS_NORM_EPS};
use moeflux::riir::{
    ExpertFiles, LayerWeightCache, MetalContext, MtlWeightBuf, WeightFile,
};

use common::diff_helpers::cosine_sim;

// ────────────────────────────────────────────────────────────────────
// Test pins
// ────────────────────────────────────────────────────────────────────

const LAYER_IDX: usize = 0;
const N_TOKENS: usize = 64;
const K_ACTIVE: usize = 8;
const SEED: u64 = 0xA3B_C0DE;

/// Cosine floor — mirrors `tests/common/diff_helpers.rs:COSINE_FLOOR`
/// and `tests/cogito_moe_gpu.rs:124`. A wiring bug produces cosine
/// catastrophically below this; f32-reduction-order noise stays at 1.0.
const COSINE_FLOOR: f32 = 0.9999;

/// Relative max-abs-diff floor — `max(|a-b|) / max(|a|) <= REL_DIFF_FLOOR`.
/// Matches `tests/cogito_moe_gpu.rs:131`. f32-reduction noise on
/// ±1-magnitude values is ~1e-6, four orders below this floor.
const REL_DIFF_FLOOR: f32 = 1e-3;

// Default paths (overridable via env) — mirror the convention used by
// drama_llama/tests/moeflux_smoke.rs and tests/common/diff_helpers.rs.
const MLX_DIR_DEFAULT: &str =
    "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-mlx-4bit";
const ARTIFACTS_DIR_DEFAULT: &str =
    "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-artifacts";
const EXPERTS_DIR_DEFAULT: &str =
    "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-root";

// ────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────

/// Deterministic xorshift, same shape as `graph_diff_oracle.rs`.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in roughly `[-0.5, 0.5]`. Matches the value range of a
    /// post-residual `h_mid` reasonably enough that the gate-router's
    /// softmax doesn't collapse all probability onto one expert.
    fn next_f32_centered(&mut self) -> f32 {
        let bits = (self.next_u64() >> 32) as u32;
        ((bits as f32) / (u32::MAX as f32)) - 0.5
    }
}

/// Mirror of `linear_attn_forward.rs:bits_of` (which is `pub(in
/// crate::riir)`). Defaults to 4-bit for tensors not in the manifest,
/// floor at 4.
fn bits_of(wf: &WeightFile, name: &str) -> u32 {
    wf.tensor_info(name).map(|i| i.bits as u32).unwrap_or(4).max(4)
}

fn env_path(var: &str, default: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(
        std::env::var(var).unwrap_or_else(|_| default.to_string()),
    )
}

// ────────────────────────────────────────────────────────────────────
// Diagnostic reporter
// ────────────────────────────────────────────────────────────────────

fn report_diff(
    layer_idx: usize,
    n_tokens: usize,
    hidden: usize,
    a: &[f32],
    b: &[f32],
) {
    let a_nz = a.iter().filter(|&&v| v != 0.0).count();
    let b_nz = b.iter().filter(|&&v| v != 0.0).count();
    eprintln!(
        "[engine-op-diff] layer={layer_idx} n_tokens={n_tokens} hidden={hidden} \
         elems_a={} elems_b={} nonzero_a={a_nz} nonzero_b={b_nz}",
        a.len(),
        b.len(),
    );
    let mut global_max = 0.0f32;
    let mut global_at = (0usize, 0usize);
    let mut per_token: Vec<(usize, f32, usize)> = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        let mut row_max = 0.0f32;
        let mut row_at = 0usize;
        for c in 0..hidden {
            let idx = t * hidden + c;
            let d = (a[idx] - b[idx]).abs();
            if d > row_max {
                row_max = d;
                row_at = c;
            }
            if d > global_max {
                global_max = d;
                global_at = (t, c);
            }
        }
        per_token.push((t, row_max, row_at));
    }
    per_token.sort_by(|x, y| {
        y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("[engine-op-diff]   top-10 worst tokens (by row max-abs):");
    for &(t, d, c) in per_token.iter().take(10) {
        let idx = t * hidden + c;
        eprintln!(
            "[engine-op-diff]     t={t:>3} max_abs={d:.4e} at c={c} \
             (a={:+.4e} b={:+.4e})",
            a[idx], b[idx],
        );
    }
    let cos = cosine_sim(a, b);
    eprintln!(
        "[engine-op-diff]   global max_abs_diff={global_max:.4e} \
         at (t={}, c={})  cosine={cos:.7}",
        global_at.0, global_at.1,
    );
}

// ────────────────────────────────────────────────────────────────────
// Body
// ────────────────────────────────────────────────────────────────────

fn run_one_layer_op_diff(layer_idx: usize, n_tokens: usize, seed: u64) {
    let v = VARIANT;
    let hidden_dim = v.hidden_dim;
    let f32_sz = size_of::<f32>();
    let i32_sz = size_of::<i32>();
    let u32_sz = size_of::<u32>();

    // ── 1. Open real model ──────────────────────────────────────────
    let artifacts_dir =
        env_path("MOEFLUX_SMOKE_ARTIFACTS", ARTIFACTS_DIR_DEFAULT);
    let experts_dir = env_path("MOEFLUX_SMOKE_ROOT", EXPERTS_DIR_DEFAULT);
    let _ = env_path("MOEFLUX_SMOKE_MLX", MLX_DIR_DEFAULT); // unused at this layer

    let weights_bin = artifacts_dir.join("model_weights.bin");
    let manifest = artifacts_dir.join("model_weights.json");

    let wf = WeightFile::open(&weights_bin, &manifest)
        .expect("open WeightFile");
    let mut ef = ExpertFiles::open(&experts_dir).expect("open ExpertFiles");
    let metal = MetalContext::new().expect("open MetalContext");
    let device = metal.device().clone();
    let wf_buf = MtlWeightBuf::wrap(&wf, &device);
    let mut backend =
        MetalBackend::new(metal, wf_buf).expect("MetalBackend::new");
    ef.attach_to_device(backend.pool_mut());
    let layer_cache =
        LayerWeightCache::build(layer_idx, &wf, backend.weight_buf())
            .expect("LayerWeightCache::build");

    debug_assert!(
        matches!(v.mlp_kind_at(layer_idx), MlpKind::MoE),
        "engine_op_diff requires an MoE layer; mlp_kind_at({layer_idx}) was not MoE",
    );

    // ── 2. Allocate per-path scratches + outputs ───────────────────
    // Two independent scratches → two independent `commit_planned`
    // latches. For a one-shot test we intentionally skip `commit_plan`
    // entirely: persistent buffers stay allocated, transients are
    // simply not lifetime-colored. Memory waste is acceptable for an
    // `#[ignore]`-gated test.
    let scratch_a = MoeGraphScratch::new(backend.pool_mut(), K_ACTIVE);
    let scratch_b = MoeGraphScratch::new(backend.pool_mut(), K_ACTIVE);
    let hidden_out_a: BufId<HiddenBuf> = backend
        .pool_mut()
        .alloc(
            n_tokens * hidden_dim * f32_sz,
            "engine_diff.hidden_out_a",
            true,
        )
        .expect("alloc hidden_out_a");
    let hidden_out_b: BufId<HiddenBuf> = backend
        .pool_mut()
        .alloc(
            n_tokens * hidden_dim * f32_sz,
            "engine_diff.hidden_out_b",
            true,
        )
        .expect("alloc hidden_out_b");
    // Per-path router-logits scratch; consumed by softmax_topk, then
    // discarded.
    let gate_logits_a: BufId<RouterLogitsBuf> = backend
        .pool_mut()
        .alloc(
            n_tokens * v.num_experts * f32_sz,
            "engine_diff.gate_logits_a",
            true,
        )
        .expect("alloc gate_logits_a");
    let gate_logits_b: BufId<RouterLogitsBuf> = backend
        .pool_mut()
        .alloc(
            n_tokens * v.num_experts * f32_sz,
            "engine_diff.gate_logits_b",
            true,
        )
        .expect("alloc gate_logits_b");

    // ── 3. Seed synthetic h_mid + upload to both scratches ─────────
    // h_post is DERIVED by Op::RmsNormBf16NTokens from h_mid (mirroring
    // production semantics); h_mid is the residual stream the layer
    // received. Seeding both independently would give them implausibly
    // similar magnitudes — the load-bearing distributional difference
    // between h_mid (residual scale) and h_post (unit-variance) is
    // precisely what makes "passing the wrong one" produce broken
    // output. We need the rms_norm to enforce that difference.
    let mut rng = Rng::new(seed);
    let mut h_mid_host = vec![0.0f32; n_tokens * hidden_dim];
    for v in h_mid_host.iter_mut() {
        // Scale slightly to mimic residual-stream magnitudes —
        // ~unit-variance Gaussian after a few residuals.
        *v = rng.next_f32_centered() * 4.0;
    }

    {
        let pool = backend.pool_mut();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                h_mid_host.as_ptr() as *const u8,
                h_mid_host.len() * f32_sz,
            )
        };
        pool.upload(scratch_a.h_mid, bytes).expect("upload h_mid_a");
        pool.upload(scratch_b.h_mid, bytes).expect("upload h_mid_b");
    }

    // ── 4. Resolve weight bits + tensor refs for the layer ─────────
    let post_attn_norm_off = wf
        .tensor_info(&format!(
            "model.layers.{layer_idx}.post_attention_layernorm.weight"
        ))
        .expect("post_attention_layernorm.weight in manifest")
        .offset as u64;
    let gate_bits =
        bits_of(&wf, &format!("model.layers.{layer_idx}.mlp.gate.weight"));
    let seg_bits = bits_of(
        &wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert_gate.weight"
        ),
    );
    let s_gate_bits = bits_of(
        &wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight"
        ),
    );
    let s_up_bits = bits_of(
        &wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight"
        ),
    );
    let s_down_bits = bits_of(
        &wf,
        &format!(
            "model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight"
        ),
    );
    let expert_base_id: BufId<ExpertBaseBuf> = ef
        .mmap_id_for_expert(layer_idx, 0)
        .expect("mmap layer present")
        .0;

    // ── 5. Build & execute the router prelude per path ─────────────
    // graph_router pushes RmsNorm(h_mid → h_post) + gate matvec +
    // shared_gate matvec + softmax_topk + normalize. Same shape both
    // paths; mirrors `linear_attn_forward.rs:2553-2611`.
    let push_router = |g: &mut Graph,
                       scratch: &MoeGraphScratch,
                       gate_logits: BufId<RouterLogitsBuf>| {
        g.push(Op::RmsNormBf16NTokens {
            label: "engine_diff.post_attn_rms_norm",
            x: BufId::<ResidualBuf>::from(scratch.h_mid).into(),
            weight_off: post_attn_norm_off,
            out: scratch.h_post.into(),
            dim: hidden_dim as u32,
            n_tokens: n_tokens as u32,
            eps: RMS_NORM_EPS,
        });
        g.push(Op::MatvecNTokens {
            label: "engine_diff.gate_router",
            weight: WeightRef {
                w_off: layer_cache.gate.w,
                s_off: layer_cache.gate.s,
                b_off: layer_cache.gate.b,
                bits: gate_bits,
            },
            input: scratch.h_post.into(),
            input_off: 0,
            output: gate_logits.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: v.num_experts as u32,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::MatvecNTokens {
            label: "engine_diff.shared_gate",
            weight: WeightRef {
                w_off: layer_cache.shared.seg_w,
                s_off: layer_cache.shared.seg_s,
                b_off: layer_cache.shared.seg_b,
                bits: seg_bits,
            },
            input: scratch.h_post.into(),
            input_off: 0,
            output: scratch.shared_gate.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: 1,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::MoeSoftmaxTopK {
            label: "engine_diff.router_softmax_topk",
            logits: gate_logits,
            indices_out: scratch.routing_indices,
            weights_out: scratch.routing_weights,
            n_tokens: n_tokens as u32,
            n_experts: v.num_experts as u32,
            k: K_ACTIVE as u32,
        });
        g.push(Op::MoeNormalizeWeights {
            label: "engine_diff.router_normalize",
            weights: scratch.routing_weights,
            n_tokens: n_tokens as u32,
            k: K_ACTIVE as u32,
        });
    };

    let mut router_a = Graph::new();
    push_router(&mut router_a, &scratch_a, gate_logits_a);
    backend
        .execute(&router_a, "engine_diff_router_a")
        .expect("router_a execute");

    let mut router_b = Graph::new();
    push_router(&mut router_b, &scratch_b, gate_logits_b);
    backend
        .execute(&router_b, "engine_diff_router_b")
        .expect("router_b execute");

    // ── 6. Download path A's routing tables for CPU bucket build ──
    let mut all_routing_indices = vec![0i32; n_tokens * K_ACTIVE];
    let mut all_routing_weights = vec![0.0f32; n_tokens * K_ACTIVE];
    {
        let pool = backend.pool();
        pool.download(scratch_a.routing_indices, unsafe {
            std::slice::from_raw_parts_mut(
                all_routing_indices.as_mut_ptr() as *mut u8,
                n_tokens * K_ACTIVE * i32_sz,
            )
        })
        .expect("download routing_indices_a");
        pool.download(scratch_a.routing_weights, unsafe {
            std::slice::from_raw_parts_mut(
                all_routing_weights.as_mut_ptr() as *mut u8,
                n_tokens * K_ACTIVE * f32_sz,
            )
        })
        .expect("download routing_weights_a");
    }
    // Sanity: scratch_b's router got the same inputs and the same
    // (deterministic) kernels — its routing tables must match.
    {
        let pool = backend.pool();
        let mut b_idx = vec![0i32; n_tokens * K_ACTIVE];
        pool.download(scratch_b.routing_indices, unsafe {
            std::slice::from_raw_parts_mut(
                b_idx.as_mut_ptr() as *mut u8,
                n_tokens * K_ACTIVE * i32_sz,
            )
        })
        .expect("download routing_indices_b");
        debug_assert_eq!(
            all_routing_indices, b_idx,
            "routing indices diverged between scratches — check Metal determinism"
        );
    }

    // ── 7. Build buckets + expert_indices_host on host ────────────
    let buckets = build_expert_buckets(
        &all_routing_indices,
        &all_routing_weights,
        n_tokens,
        K_ACTIVE,
        v.num_experts,
    );
    let total_assignments = buckets.token_idx.len();
    debug_assert_eq!(total_assignments, n_tokens * K_ACTIVE);

    let expert_slots: Vec<u32> =
        buckets.expert_ids.iter().map(|&e| e as u32).collect();
    let mut expert_indices_host = vec![0u32; total_assignments];
    for bi in 0..buckets.expert_ids.len() {
        let start = buckets.offsets[bi] as usize;
        let end = buckets.offsets[bi + 1] as usize;
        expert_indices_host[start..end].fill(expert_slots[bi]);
    }

    // Path A's `bucket_input` host permute (mirrors `2816-2832`). Path
    // B doesn't consume `bucket_input`; uploading it on path B would
    // be wasted but harmless. We upload it only to scratch_a.
    let mut h_post_stack = vec![0.0f32; n_tokens * hidden_dim];
    {
        let pool = backend.pool();
        pool.download(scratch_a.h_post, unsafe {
            std::slice::from_raw_parts_mut(
                h_post_stack.as_mut_ptr() as *mut u8,
                n_tokens * hidden_dim * f32_sz,
            )
        })
        .expect("download h_post_a");
    }
    let mut bucket_input_host =
        vec![0.0f32; total_assignments * hidden_dim];
    for a in 0..total_assignments {
        let t = buckets.token_idx[a] as usize;
        let src = &h_post_stack[t * hidden_dim..(t + 1) * hidden_dim];
        let dst_off = a * hidden_dim;
        bucket_input_host[dst_off..dst_off + hidden_dim]
            .copy_from_slice(src);
    }

    // ── 8. Upload CPU-built bucket tables to both scratches ──────
    {
        let pool = backend.pool_mut();
        let upload_per_scratch = |pool: &mut <MetalBackend as Backend>::Pool,
                                  s: &MoeGraphScratch| {
            pool.upload(s.bucket_token_idx, unsafe {
                std::slice::from_raw_parts(
                    buckets.token_idx.as_ptr() as *const u8,
                    total_assignments * i32_sz,
                )
            })
            .expect("upload bucket_token_idx");
            pool.upload(s.bucket_weights, unsafe {
                std::slice::from_raw_parts(
                    buckets.weights.as_ptr() as *const u8,
                    total_assignments * f32_sz,
                )
            })
            .expect("upload bucket_weights");
            pool.upload(s.expert_indices, unsafe {
                std::slice::from_raw_parts(
                    expert_indices_host.as_ptr() as *const u8,
                    total_assignments * u32_sz,
                )
            })
            .expect("upload expert_indices");
        };
        upload_per_scratch(pool, &scratch_a);
        upload_per_scratch(pool, &scratch_b);
        // Path A only.
        pool.upload(scratch_a.bucket_input, unsafe {
            std::slice::from_raw_parts(
                bucket_input_host.as_ptr() as *const u8,
                total_assignments * hidden_dim * f32_sz,
            )
        })
        .expect("upload bucket_input_a");
    }

    // ── 9. Build the MoE-block graph per path + execute ──────────
    // Shape mirrors `linear_attn_forward.rs:2859-2980`. Path A forces
    // the env=off `else` branch; path B forces the env=on `if` branch.
    let push_moe_block = |g: &mut Graph,
                          scratch: &MoeGraphScratch,
                          path_is_gather_id: bool,
                          hidden_out: BufId<HiddenBuf>| {
        g.push(Op::MatvecNTokens {
            label: "engine_diff.shared_gate_proj",
            weight: WeightRef {
                w_off: layer_cache.shared.gate_w,
                s_off: layer_cache.shared.gate_s,
                b_off: layer_cache.shared.gate_b,
                bits: s_gate_bits,
            },
            input: scratch.h_post.into(),
            input_off: 0,
            output: scratch.shared_ffn_gate.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: v.shared_intermediate as u32,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::MatvecNTokens {
            label: "engine_diff.shared_up_proj",
            weight: WeightRef {
                w_off: layer_cache.shared.up_w,
                s_off: layer_cache.shared.up_s,
                b_off: layer_cache.shared.up_b,
                bits: s_up_bits,
            },
            input: scratch.h_post.into(),
            input_off: 0,
            output: scratch.shared_up.into(),
            output_off: 0,
            in_dim: hidden_dim as u32,
            out_dim: v.shared_intermediate as u32,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::SwigluFusedBatched {
            label: "engine_diff.shared_swiglu",
            gate: scratch.shared_ffn_gate,
            up: scratch.shared_up,
            out: scratch.shared_act,
            total: (n_tokens * v.shared_intermediate) as u32,
        });
        g.push(Op::MatvecNTokens {
            label: "engine_diff.shared_down_proj",
            weight: WeightRef {
                w_off: layer_cache.shared.down_w,
                s_off: layer_cache.shared.down_s,
                b_off: layer_cache.shared.down_b,
                bits: s_down_bits,
            },
            input: scratch.shared_act.into(),
            input_off: 0,
            output: scratch.shared_down.into(),
            output_off: 0,
            in_dim: v.shared_intermediate as u32,
            out_dim: hidden_dim as u32,
            n_tokens: n_tokens as u32,
        });
        g.push(Op::ZeroBuffer {
            label: "engine_diff.out_sum_zero",
            buf: scratch.out_sum,
            n_bytes: (n_tokens * hidden_dim * f32_sz) as u32,
        });
        if path_is_gather_id {
            // Path B: env=on, the new-kernel one-dispatch path.
            // Mirrors `linear_attn_forward.rs:2929-2950`.
            g.push(Op::MoeGatherIdFuse {
                label: "engine_diff.gather_id_fuse",
                expert_base: expert_base_id,
                expert_stride: v.expert_size_4bit() as u64,
                indices: scratch.routing_indices,
                weights: scratch.routing_weights,
                mlp_in: scratch.h_post,
                out_sum: scratch.out_sum,
                htpe: scratch.htpe,
                hids: scratch.hids,
                gate_mid: scratch.bucket_gate.into(),
                up_mid: scratch.bucket_up.into(),
                down_mid: scratch.bucket_out.into(),
                n_tokens: n_tokens as u32,
                n_experts: v.num_experts as u32,
                k: K_ACTIVE as u32,
            });
        } else {
            // Path A: env=off, the bucket-permute fallback.
            // Mirrors `linear_attn_forward.rs:2952-2967`. `buckets` is
            // cloned because the Op consumes it by value.
            g.push(Op::MoeBatchedPermuteFuse {
                label: "engine_diff.permute_fuse",
                expert_base: expert_base_id,
                expert_stride: v.expert_size_4bit() as u64,
                expert_indices: scratch.expert_indices,
                expert_slots: expert_slots.clone(),
                bucket_input: scratch.bucket_input,
                bucket_gate: scratch.bucket_gate,
                bucket_up: scratch.bucket_up,
                bucket_act: scratch.bucket_act,
                bucket_out: scratch.bucket_out,
                bucket_token_idx: scratch.bucket_token_idx,
                bucket_weights: scratch.bucket_weights,
                out_sum: scratch.out_sum,
                buckets: buckets.clone(),
            });
        }
        g.push(Op::MoeCombineResidualNTokens {
            label: "engine_diff.combine",
            h_mid: scratch.h_mid,
            moe_sum: scratch.out_sum,
            shared_out: scratch.shared_down,
            shared_gate: scratch.shared_gate,
            hidden_out,
            n_tokens: n_tokens as u32,
            dim: hidden_dim as u32,
        });
    };

    let mut graph_a = Graph::new();
    push_moe_block(&mut graph_a, &scratch_a, false, hidden_out_a);
    backend
        .execute(&graph_a, "engine_diff_moe_a")
        .expect("graph_a execute");

    let mut graph_b = Graph::new();
    push_moe_block(&mut graph_b, &scratch_b, true, hidden_out_b);
    backend
        .execute(&graph_b, "engine_diff_moe_b")
        .expect("graph_b execute");

    // ── 10. Download outputs + diagnostic + assert ──────────────
    let mut a_bytes = vec![0u8; n_tokens * hidden_dim * f32_sz];
    let mut b_bytes = vec![0u8; n_tokens * hidden_dim * f32_sz];
    {
        let pool = backend.pool();
        pool.download(hidden_out_a, &mut a_bytes).expect("download A");
        pool.download(hidden_out_b, &mut b_bytes).expect("download B");
    }
    let a_f32: &[f32] = bytemuck::cast_slice(&a_bytes);
    let b_f32: &[f32] = bytemuck::cast_slice(&b_bytes);
    report_diff(layer_idx, n_tokens, hidden_dim, a_f32, b_f32);

    // Belt-and-suspenders: a path that silently dropped every dispatch
    // (zero kernels actually run) would pass cosine vacuously (NaN /
    // 0-vector). Require at least some nonzero output first.
    assert!(
        a_f32.iter().any(|&v| v != 0.0),
        "path A produced all-zeros output — kernel dispatch likely skipped",
    );
    assert!(
        b_f32.iter().any(|&v| v != 0.0),
        "path B produced all-zeros output — kernel dispatch likely skipped",
    );

    // Numerical-equivalence contract: the two MoE-block paths use
    // different Metal kernels with different reduction orders, so
    // they are NOT bit-identical at the output. They ARE
    // mathematically equivalent — cosine 1.0 and rel max-abs-diff
    // in f32-noise territory.
    let cos = cosine_sim(a_f32, b_f32);
    let mut max_abs_diff = 0.0f32;
    let mut max_abs_a = 0.0f32;
    for (&a, &b) in a_f32.iter().zip(b_f32.iter()) {
        max_abs_diff = max_abs_diff.max((a - b).abs());
        max_abs_a = max_abs_a.max(a.abs());
    }
    let rel = max_abs_diff / max_abs_a.max(1e-30);
    assert!(
        cos >= COSINE_FLOOR,
        "Op::MoeBatchedPermuteFuse vs Op::MoeGatherIdFuse: cosine {cos:.7} \
         below floor {COSINE_FLOOR} — likely a producer-wiring bug. \
         See per-token diagnostic above.",
    );
    assert!(
        rel <= REL_DIFF_FLOOR,
        "Op::MoeBatchedPermuteFuse vs Op::MoeGatherIdFuse: rel max-abs-diff \
         {rel:.4e} above floor {REL_DIFF_FLOOR:.1e} — math drift beyond \
         f32-reduction-order noise.",
    );
}

// ────────────────────────────────────────────────────────────────────
// #[test]
// ────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires real Qwen3-6-A3B weights on /Volumes/Temp Backup"]
fn moe_gather_id_matches_batched_permute_fuse_engine_level() {
    run_one_layer_op_diff(LAYER_IDX, N_TOKENS, SEED);
}
