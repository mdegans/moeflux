# Session 10 landed — S10a + S10b pre-work (Backend trait, RsCtx<B>, MoeBuffers → pool)

**Date:** 2026-05-15
**Branch:** moeflux `main`
**Commits (6 total):**

S10a (RsCtx<B: Backend> + fold metal/wf_buf/pool into backend):
- `96345dc` — S10a-1: `Backend::Config` + `open`, `MetalConfig`/`CpuConfig`, `metal_mut`/`parts_mut`
- `ae5d775` — S10a-2: `RsCtx<B: Backend = MetalBackend>` type param + `backend: Option<B>` field
- `fc0c444` — S10a-4: cascade — folded `metal/wf_buf/pool` into `RsCtx.backend`, ~40 sites + 7 destructures

S10b pre-work (MoeBuffers / ExpertFiles pool migration):
- `40eac30` — S10b-pre-1: pool primitives (`alloc_aligned`, `register_borrowed`, `as_mut_slice_u8`, `as_mut_slices_u8`)
- `1b41217` — S10b-pre-2: `ExpertFiles.attach_to_device` uses `register_borrowed` for mmap mode; new `mmap_id_for_expert` accessor
- `8c63947` — S10b-pre-3: `MoeBuffers` struct + 50-site cascade — 12 fields → `BufId`, constructor takes `&mut pool`

**Locked plan:** [`qwen_graph_mode_session7_plan.md`](qwen_graph_mode_session7_plan.md)
**Session-9 entry:** [`qwen_graph_mode_session9_landed.md`](qwen_graph_mode_session9_landed.md)

## Headline state

- **`RsCtx<B: Backend = MetalBackend>`** — the struct is generic, but all
  current impl blocks are `impl RsCtx<MetalBackend>` (no method went
  truly generic-over-B this session — that's the producer-rewrite step's
  job once it lands and starts driving the bound).

- **`Backend` trait** gained `type Config` + `fn open(config) -> Result<Self, Self::Error> where Self: Sized`.
  `MetalConfig { metal, wf_buf }` and `CpuConfig { wf }` exist; `open`
  thin-wraps the existing `::new` constructors. `GraphError::Backend(Box<dyn Error>)`
  variant added as the backend-typed-init-error escape hatch.

- **`MetalBufferPool`** owns all moeflux's MoE buffers now:
  - `alloc_aligned(bytes, alignment, label, persistent) -> BufId` — pread DMA destinations (2 MiB alignment per the C path's 3.6× DMA win)
  - `register_borrowed(metal::Buffer, bytes, label, persistent) -> BufId` — externally-owned buffers (mmap'd expert layers)
  - `as_mut_slice_u8(BufId) -> &mut [u8]` + `as_mut_slices_u8::<N>(ids) -> [&mut [u8]; N]` — host-write slices
  - `device() -> &Device` — for callers that need raw device access (ExpertFiles mmap)

- **`MoeBuffers`** is BufId-end-to-end. ~62 buffers across 12 fields, all
  allocated as persistent pool entries by `MoeBuffers::new(&mut pool)`.
  Accessor surface split: `*_id()` returns `BufId`; `*_buffer(pool)`
  / `*_to_vec(pool)` / `stage_host_*(pool, ...)` / `data_*_slots_mut_array(pool, ...)`
  for the imperative encoders.

- **Canary 9/9 green** after every commit (81.65s / 83.86s on M2 Max).

## What's left for session 11

### S10b-1 — Linear-attn pre-MoE chain to graph (NOT landed)

**Plan-of-record:** [`refactored-juggling-acorn.md`](/Users/mdegans/.claude/plans/refactored-juggling-acorn.md)
in the planning area, plus the locked design in `qwen_graph_mode_session7_plan.md`.

`batched_linear_attn_layer_forward` (`crates/moeflux/src/riir/linear_attn_forward.rs:1918`)
gets its **pre-MoE chain** (phases 1a-1e: input rms_norm, 4 projections,
recurrent loop, o_proj, residual+post-norm+gate+shared_gate+router)
converted to `graph.push(Op::...)` builds. After routing readback (CPU
bucket build), phases 1f-1g (shared FFN + MoE permute-fuse + combine)
**stay imperative** for 10b-1 — 10b-2 promotes them.

### ⚠️ Op-stacked-buffers gotcha (read this BEFORE writing producer code)

All `*NTokens` Op variants (`Conv1dStepNTokens`,
`ComputeDecayBetaNTokens`, `GatedDeltaNetStepNTokens`,
`GatedRmsNormNTokens`, `RmsNormQkNTokens`) **expect stacked buffers**
where each token's slot is at offset `t * per_token_bytes`. The Metal
arms (e.g. `graph/metal.rs:771-813` for `Conv1dStepNTokens`) internally
loop over tokens and apply `let off = (t as u64) * per_token_bytes;`.

**Today's per-token loop in the producer** uses **single-token scratch
buffers** (`buffers.conv_output`, `buffers.delta_output`,
`buffers.delta_g_decay`, `buffers.delta_beta`) — sized
`per_token_bytes`, *reused* across the loop because each kernel reads
the previous one's output for the same token.

**Migration**: allocate **new stacked transient BufIds** for these
intermediates (n_tokens × per_token_size each), as transient pool slots
released by `pool.reset_transient()` at chunk end. Do NOT reuse
`buffers.conv_output` etc. — they're single-token sized.

Concretely the producer will need transient pool allocations for:
- `normed` — `n_tokens * hidden_dim * 4`
- `qkv_stack` — `n_tokens * conv_dim * 4`
- `z_stack`, `beta_stack`, `alpha_stack` — sized per-projection-output
- `conv_output_stacked` — `n_tokens * conv_dim * 4` (NEW vs today)
- `delta_g_decay_stacked`, `delta_beta_stacked` — `n_tokens * num_v_heads * 4` (NEW)
- `delta_output_stacked` — `n_tokens * num_v_heads * value_dim * 4` (NEW)
- `value_out_stack` — `n_tokens * total_value * 4`
- `o_proj_stack`, `h_mid`, `h_post` — `n_tokens * hidden_dim * 4`
- `gate_logits` — `n_tokens * num_experts * 4`
- `shared_gate` — `n_tokens * 4`
- `routing_indices` — `n_tokens * k_active * sizeof(i32)`
- `routing_weights` — `n_tokens * k_active * 4`

The persistent state buffers `buffers.conv_state[layer_idx]` and
`buffers.delta_state[layer_idx]` are RMW per-token (the Metal arm
loops over tokens, each token reads+writes state). They're already
BufIds (S7-6c-2). Just pass them in the Op.

### Per-token Op semantics

- **`Conv1dStepNTokens`**: loops N times; each iter reads `qkv_in[t]`
  + `conv_state` (RMW), writes `conv_out[t]`. Stacked.
- **`RmsNormQkNTokens`**: in-place on `x`. Stacked (per_token_elems =
  `key_offset_per_token + num_k_heads * key_dim`). For linear-attn,
  `x` is `conv_out` from the previous Op — same buffer. The Op's
  `key_offset_per_token` is the byte offset from `q region start` to
  `k region start` *within one token's slot* (in floats; multiplied
  by 4 internally). For the q/k/v layout `q | k | v` packed per token,
  this is `Variant::LINEAR_TOTAL_KEY` (q size in floats).

  **Verify**: today's per-token call passes `buffer_pool.handle(buffers.conv_output)`
  with no offset. The corresponding Op needs to know where k is
  relative to q — see `linear_attn_forward.rs:2161` for the existing
  encoder call and how it derives that.

- **`ComputeDecayBetaNTokens`**: stacked. Per-token-bytes is `num_v_heads * 4`.
- **`GatedDeltaNetStepNTokens`**: stacked. Per-token reads `conv_out[t]`
  decomposed into q/k/v regions internally (offsets within the
  `conv_per_token` slot). Reads + writes `state` (RMW).
- **`GatedRmsNormNTokens`**: stacked. Output is
  `n_tokens * num_v_heads * value_dim * 4` bytes.

### Execution model

Producer signature should become:

```rust
pub(super) fn batched_linear_attn_layer_forward(
    backend: &mut MetalBackend,       // CHANGED: replaces metal/wf_buf/pool
    wf: &WeightFile,
    layer_cache: &LayerWeightCache,
    buffers: &LayerForwardBuffers,
    layer_idx: usize,
    n_tokens: usize,
    k_active: usize,
    expert_files: &ExpertFiles,
    moe_buffers: &mut MoeBuffers,     // NEW: was via PrefetchEnv before
    _layer_state: &mut LinearAttnState,
    prefetch: Option<PrefetchEnv<'_>>,
    hidden_in_id: BufId,              // CHANGED: was &metal::Buffer
    hidden_out_id: BufId,             // CHANGED
) -> Result<(), LayerForwardError>
```

Inside:

```rust
// Phase 1: build graph (NLL scopes the parts_mut borrow)
let graph = {
    let mut g = Graph::new();
    let pool = backend.pool_mut();
    let normed_id = pool.alloc(n_tokens * hidden_dim * 4, "...", false)?;
    // ... allocate all stacked transients
    g.push(Op::RmsNormBf16NTokens { x: hidden_in_id, weight_off: layer_cache.input_layernorm_w, out: normed_id, ... });
    // ... 16 more Ops
    g  // moved out
};

// Phase 2: execute pre-MoE graph
backend.execute(&graph).map_err(...)?;

// Phase 3: readback for CPU bucket build
let pool = backend.pool();
let h_post_buf = pool.handle(h_post_id);
let h_post_stack: &[f32] = unsafe { std::slice::from_raw_parts(h_post_buf.contents() as *const f32, n_tokens * hidden_dim) };
// ... routing_indices, routing_weights readback similarly

// Phase 4: CPU bucket build (existing build_expert_buckets)
let buckets = build_expert_buckets(...);

// Phase 5: imperative MoE (1f shared FFN + 1g permute-fuse + combine)
{
    let (metal, wf_buf, buffer_pool) = backend.parts_mut();
    // ... existing imperative code, almost unchanged
    // Replace today's MtlBuffer::<f32>::with_len allocations with
    // pool.alloc(...) for the imperative MoE scratch (bucket_gate, etc.)
}
```

The `let graph = { ... };` inner-scope trick is what lets `backend.execute(&graph)`
succeed — NLL ends the `parts_mut` borrow when the inner scope closes.

### Orchestrator change

`step_internal_batched_gqa` (`mod.rs:1492`) currently destructures
`(metal, wf_buf, pool) = backend.parts_mut()` at the layer-loop top.
For 10b-1, restructure so each branch gets what it needs:

```rust
let backend = backend.as_mut().expect("ensure_linear_resources");
// Don't destructure parts here. Each layer branch borrows as needed.

for layer_idx in 0..v.num_layers {
    if is_full {
        let (metal, wf_buf, pool) = backend.parts_mut();
        batched_full_attn_layer_forward(metal, wf, wf_buf, ..., pool, ...);
        // parts borrow ends after call
    } else {
        // linear-attn takes &mut backend directly
        batched_linear_attn_layer_forward(backend, wf, ..., moe_buffers, ..., hidden_a_id, hidden_b_id);
    }
    std::mem::swap(&mut hidden_a_id, &mut hidden_b_id);
}
```

NLL handles the alternating borrows because each iteration's borrow
ends before the next.

### S10b-2 (after 10b-1 green)

Promote MoE step to graph form. `expert_refs: Vec<(BufId, u64)>` is
sourced from pool BufIds — pread mode uses
`moe_buffers.data_prefetch_id(set, slot)`, mmap mode uses
`expert_files.mmap_id_for_expert(layer, expert_id)` (S10b-pre-2 added
the latter). Bench post-reboot.

## Verification before starting

```bash
cd ~/Projects/moeflux
# Lib + graph unit tests:
cargo test -p moeflux --features model-qwen3-6-35b-a3b --lib
# Expected: 67 lib + 23 graph = 90 passed (4 new pool tests under --ignored)

# Per-Op graph diff oracle:
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test graph_diff_oracle -- --ignored --test-threads=1
# Expected: 9 passed, all cos=1.0

# New pool primitive tests:
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b --lib \
  riir::graph::metal::tests -- --ignored
# Expected: 4 passed (alloc_aligned, register_borrowed, as_mut_slice_u8, as_mut_slices_u8)

# Canary 9/9 (load-bearing baseline):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --test-threads=1 \
  state_round_trip_rust eval_token_matches_c_single_step \
  eval_prompt_matches_per_token_oracle slot_reuse_race_regression_rust \
  state_load_rust_from_c_save eval_prompt_matches_c_multi_token \
  eval_prompt_chunked_matches_eval_prompt_whole_prompt \
  prompt_cache_start_pos_nonzero_matches diag_b2_eval_prompt_chunk_1
# Expected: 9 passed
```

## Top 5 things to know

### 1. The Op-stacked-buffers gotcha (repeated for emphasis)

When you push `Op::Conv1dStepNTokens { conv_out: BufId, ... }`, the
Metal arm internally loops `for t in 0..n_tokens { off = t * per_token_bytes; ... }`.
The BufId you pass must point to an n_tokens-sized stacked buffer, NOT
a single-token scratch buffer. Today's `buffers.conv_output` is
single-token-sized — allocate a NEW transient BufId. Same for
`delta_output`, `delta_g_decay`, `delta_beta`.

### 2. The producer takes `&mut MetalBackend`, not generic-over-B

Even though `Backend` trait exists and producer-generic was the locked
plan, S10b-1 takes concrete `&mut MetalBackend` because the imperative
MoE step needs `backend.parts_mut()` (a MetalBackend-specific inherent
method, not on the trait). Once 10b-2 promotes the MoE step to graph
form, the producer can lift to `<B: Backend>`. Half-state is contained
to one session.

### 3. Inner-scope borrow trick

To use `backend.parts_mut()` for setup AND `backend.execute(&graph)`
for flush, put the setup in an inner scope. NLL ends the parts borrow
at the scope's `}`. Then execute. Then re-borrow parts in a new scope
for the imperative MoE step.

### 4. Today's producer is ~720 LOC; the rewrite is bigger

The pre-MoE rewrite is ~17 Op pushes (each with 5-10 fields) plus the
transient BufId allocation block (~13 ids) plus the readback block. ~250
lines of new code, ~400 lines of imperative code deleted (the per-token
loop body and the encoder calls collapse to single Op pushes). Plus the
imperative MoE block (~200 LOC) carries over with minor adjustments.

### 5. Diff testing strategy

The canary battery is the load-bearing gate. The `eval_prompt_matches_per_token_oracle`
test runs the batched path against the per-token reference at cos=1.0
expectation. If the batched path is wrong, this catches it. **Don't
skip canary 9/9** at the 10b-1 checkpoint — Op semantics mismatches
will fail it.

Optional new test: `linear_attn_pre_moe_graph_matches_imperative` —
run one linear-attn layer through the new producer, compare pre-MoE
buffers (post-1e h_post / routing_indices / routing_weights) against
the imperative version's readback. Cosine ≥ 0.9999. Doable as a
fast-running per-Op-like test in `tests/`.

## Calibration note (session 10)

Session 10 shipped 6 commits / ~1010 net LOC across two sub-arcs (S10a
struct refactor + S10b pre-cascade). MoeBuffers migration was the
heaviest single cascade (50 sites in 6 files); session 9 was ~30 sites
in 1 file, session 10 ~50 in 6 — proportionally bigger.

Pre-MoE rewrite was deferred when I almost missed the
Op-stacked-buffers gotcha. Caught it by reading the Metal arm
implementations rather than assuming N-tokens variants meant
"per-call-batched same as today". That's an honest stop, not a
fatigue stop — the right discipline given the semantic shift.

Mike: "You're working for free. Least I can do is let you choose
your hours." Took that to heart. Stopped at a clean canary-green
state; the producer rewrite is a fresh-session task. Don't feel bad
about not finishing — this was big enough to stop on.
