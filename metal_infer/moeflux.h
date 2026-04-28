// moeflux.h — public C API for the moeflux streaming-experts MoE
// decode backend.
//
// Consumed by drama_llama's Rust FFI wrapper (Phase 4). Functions are
// extern "C" so bindgen can generate Rust-facing bindings directly.
//
// Design notes:
//
// * Logit buffers are caller-allocated (size mf_n_vocab). Matches the
//   "borrow-invalidated by next mutating call" lifetime already in
//   drama_llama's `Decoder` trait.
// * No sampling. mf_eval_* returns raw logits. drama_llama's sampling
//   chain (temperature, top-k, grammar filter, repetition penalty)
//   runs upstream.
// * seq_id arg is accepted for API forward-compat but ignored today;
//   the backend supports a single sequence at a time. Multi-seq
//   support would require revisiting the static GPU scratch buffers.
// * Threading: single-threaded per mf_ctx. Concurrent access to the
//   same ctx is not supported. Multiple ctxs in one process are not
//   supported either (Metal context and expert cache are file-scope
//   globals inside infer.m).
// * Linear-attention truncation: `mf_memory_seq_rm` for p0 > 0 resets
//   all GatedDeltaNet recurrence state. Caller must re-prefill from
//   position 0 to recover a matching state. See NOTES.md Option A.

#ifndef MOEFLUX_H
#define MOEFLUX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle bundling weights, vocab, per-layer state, and working
// buffers for one model.
typedef struct mf_ctx mf_ctx;

// Construct a model context.
//
// * `weights_path`   — path to the packed model_weights.bin
// * `manifest_path`  — path to model_weights.json (tensor layout)
// * `vocab_path`     — path to vocab.bin (token ID ↔ string)
// * `experts_dir`    — directory containing `packed_experts/layer_00.bin`
//                      through `layer_NN.bin` (or `packed_experts_2bit/`
//                      when `use_2bit` is nonzero)
// * `experts_per_tok`— K active experts per token (typically 4)
// * `use_2bit`       — nonzero to use 2-bit quantized experts (faster,
//                      breaks tool-calling JSON per upstream's results)
//
// Returns a valid mf_ctx* on success, NULL on failure (file missing,
// mmap failure, vocab parse, Metal unavailable). Only one mf_ctx may
// be alive per process.
mf_ctx *mf_init_model(const char *weights_path,
                      const char *manifest_path,
                      const char *vocab_path,
                      const char *experts_dir,
                      int experts_per_tok,
                      int use_2bit);

// Free everything owned by `ctx`. No-op on NULL. Does NOT tear down
// Metal globals — those persist for the process lifetime.
void mf_free_model(mf_ctx *ctx);

// Prefill `n` tokens at positions [start_pos, start_pos + n) on the
// (single) sequence. Uses the upstream prefill optimization: the first
// n-1 tokens drive state updates only; the last token's logits are
// written to `logits_out` (must point to at least mf_n_vocab floats).
//
// `seq_id` is ignored (see design notes above).
// `n == 0` is a no-op returning 0 without touching `logits_out`.
//
// Returns 0 on success, nonzero on error.
int mf_eval_prompt(mf_ctx *ctx,
                   const int32_t *tokens,
                   size_t n,
                   size_t start_pos,
                   int seq_id,
                   float *logits_out);

// Advance one token at `pos`. Writes next-token logits to `logits_out`
// (must point to at least mf_n_vocab floats). Returns 0 on success.
int mf_eval_token(mf_ctx *ctx,
                  int32_t token,
                  size_t pos,
                  int seq_id,
                  float *logits_out);

// Reset the sequence to empty. Equivalent to mf_memory_seq_rm(ctx, -1,
// 0, -1) semantically.
void mf_memory_clear(mf_ctx *ctx);

// Truncate the sequence to positions [0, p0). Full-attention layers
// are truncated precisely; linear-attention layers are reset to the
// empty-sequence state (lossy — see design notes). `p0 < 0` is treated
// as 0. `p1 < 0` or `p1 >= current_length` is treated as "to end".
// `seq_id` is ignored. Returns nonzero on success, 0 on failure.
int mf_memory_seq_rm(mf_ctx *ctx, int seq_id, int p0, int p1);

// Largest position present in any full-attention KV cache. Returns -1
// if the sequence is empty or ctx is NULL. `seq_id` is ignored.
int mf_memory_seq_pos_max(mf_ctx *ctx, int seq_id);

// Vocabulary size. Equals VOCAB_SIZE in the compiled infer.m.
size_t mf_n_vocab(const mf_ctx *ctx);

// Maximum KV context length. Equals MAX_SEQ_LEN in the compiled
// infer.m.
size_t mf_n_ctx(const mf_ctx *ctx);

// Primary end-of-sequence token.
int32_t mf_eos(const mf_ctx *ctx);

// Static model name string. Points to storage owned by moeflux; do
// not free. Stable for the process lifetime.
const char *mf_model_name(const mf_ctx *ctx);

// ============================================================================
// Diff-oracle hooks (RIIR Phase 3 — per-layer dump points)
// ============================================================================
//
// Surgical accessors that expose moeflux's internal forward-pass
// primitives so the differential test harness in the Rust crate can
// compare per-layer outputs against the in-progress pure-Rust port.
// Not part of the production decode path; safe to ignore in non-test
// callers.

// Compute the embedding for a single token. Writes HIDDEN_DIM floats
// into `out`. Returns 0 on success, -1 on error (NULL args, token_id
// out of vocabulary range). Read-only on `ctx`; safe to call between
// (or before any) eval calls.
int mf_embed_lookup(mf_ctx *ctx, int32_t token_id, float *out);

// CPU RMS normalization. Loads `weight_name` from the weight file
// (must be a BF16 tensor of length HIDDEN_DIM), then computes
// `out[i] = x[i] / sqrt(mean(x*x) + EPS) * weight[i]` for the active
// architecture's RMS_NORM_EPS. `x` and `out` are both HIDDEN_DIM
// floats. Returns 0 on success, -1 on NULL args / missing tensor.
// Read-only on `ctx`.
int mf_rms_norm_cpu(mf_ctx *ctx, const char *weight_name,
                    const float *x, float *out);

// Apply rotary position embedding to a Q/K pair at position `pos`.
// `q` is `NUM_ATTN_HEADS * HEAD_DIM` floats, `k` is
// `NUM_KV_HEADS * HEAD_DIM` floats; both are mutated in place. Uses
// the active architecture's `ROTARY_DIM` and `ROPE_THETA`. The first
// `ROTARY_DIM` channels of each head are rotated using the
// non-traditional pairing `(x[i], x[i+half])`. Returns 0 on success,
// -1 on NULL args or `pos < 0`.
int mf_apply_rotary_emb(mf_ctx *ctx, int32_t pos, float *q, float *k);

// Per-head CPU RMS normalization, mutating in place. `x_inout` is
// `num_heads * head_dim` floats laid out contiguously per head; each
// head's `head_dim`-long slice is independently RMS-normalized and
// scaled by the same shared bf16 weight tensor of length `head_dim`
// loaded from `weight_name`. The active architecture's `RMS_NORM_EPS`
// is used. Returns 0 on success, -1 on NULL args / non-positive shape
// / missing tensor. Read-only on `ctx`.
int mf_rms_norm_per_head_cpu(mf_ctx *ctx, const char *weight_name,
                              int32_t num_heads, int32_t head_dim,
                              float *x_inout);

// Scaled dot-product attention with sigmoid-gated output, computed on
// the CPU. Single query position against `kv_len` cached positions.
// Uses the active architecture's NUM_ATTN_HEADS / NUM_KV_HEADS / HEAD_DIM.
// GQA: each group of (NUM_ATTN_HEADS / NUM_KV_HEADS) query heads
// shares one kv head.
//
//   q:       [NUM_ATTN_HEADS * HEAD_DIM] post-RoPE query for this position.
//   q_gate:  [NUM_ATTN_HEADS * HEAD_DIM] pre-sigmoid gate logits.
//   k_cache: [kv_len * NUM_KV_HEADS * HEAD_DIM] post-RoPE K rows.
//   v_cache: [kv_len * NUM_KV_HEADS * HEAD_DIM] V rows.
//   out:     [NUM_ATTN_HEADS * HEAD_DIM] sigmoid-gated attention output.
//
// Output is `softmax(Q·K^T / sqrt(HEAD_DIM)) · V` per-head, then
// elementwise multiplied by `sigmoid(q_gate)`. Returns 0 on success,
// -1 on NULL args or `kv_len <= 0`. Read-only on `ctx`.
int mf_sdpa_cpu(mf_ctx *ctx, int32_t kv_len,
                const float *q, const float *q_gate,
                const float *k_cache, const float *v_cache,
                float *out);

// CPU LM head matvec. Loads `lm_head.{weight,scales,biases}` and
// computes `out[row] = Σ_i (dequant(W[row, i]) * x[i])` for each row in
// `[0, VOCAB_SIZE)`. Routes through the deterministic `cpu_dequant_matvec`
// path (not `fast_dequant_matvec`) so the diff oracle can compare CPU
// outputs head-on regardless of Metal availability.
//
//   x:   [HIDDEN_DIM] post-final-norm hidden state.
//   out: [VOCAB_SIZE] raw logits.
//
// Returns 0 on success, -1 on NULL args / missing tensor. Read-only on
// `ctx`.
int mf_lm_head_cpu(mf_ctx *ctx, const float *x, float *out);

// CPU MoE router: softmax → top-K → normalize. `scores` is `n_scores`
// raw gate logits, mutated in place (afterwards holds softmax
// probabilities). `indices_out` and `weights_out` are parallel arrays
// of length `k`; on success they hold the top-K expert IDs and their
// normalized weights, in the slot order produced by the C
// selection-sort `cpu_topk` (NOT sorted by score).
//
// `k` must satisfy `1 <= k <= n_scores`. Returns 0 on success, -1 on
// NULL args / out-of-range `k`. Read-only on `ctx`.
int mf_moe_router_cpu(mf_ctx *ctx,
                      float *scores, int32_t n_scores,
                      int32_t k,
                      int32_t *indices_out,
                      float *weights_out);

// CPU depthwise 1D conv step + SiLU tail. For each channel `c`,
// computes a dot product over `[conv_state..., new_input]` against
// `weight_name`'s `c`-th row, then applies SiLU.
//
//   conv_state: [(kernel_size-1) * channels] row-major (time, channel).
//   new_input:  [channels].
//   out:        [channels], written from scratch.
//
// `weight_name` must reference a bf16 tensor of length
// `channels * kernel_size`. Returns 0 on success, -1 on NULL args /
// missing tensor / non-positive shape. Caller is responsible for
// shifting the conv state after this call. Read-only on `ctx`.
int mf_conv1d_step_cpu(mf_ctx *ctx, const char *weight_name,
                        int32_t channels, int32_t kernel_size,
                        const float *conv_state,
                        const float *new_input,
                        float *out);

// CPU bare RMS norm (no weight). `out[i] = x[i] / sqrt(mean(x*x) + eps)`.
// `dim` must be positive; `x` and `out` must both be `dim` floats.
// Returns 0 on success, -1 on NULL args / non-positive dim. Read-only
// on `ctx`.
int mf_rms_norm_bare_cpu(mf_ctx *ctx, int32_t dim, float eps,
                          const float *x, float *out);

// CPU RMSNormGated: `out[i] = rms_norm(x)[i] * w[i] * silu(z[i])`.
// Loads `weight_name` (bf16, length `dim`).
//
//   x:   [dim] post-recurrence per-head output values.
//   z:   [dim] gate-input values (pre-SiLU).
//   out: [dim], written from scratch.
//
// Returns 0 on success, -1 on NULL args / missing tensor /
// non-positive dim. Read-only on `ctx`.
int mf_rms_norm_gated_cpu(mf_ctx *ctx, const char *weight_name,
                           int32_t dim, float eps,
                           const float *x, const float *z,
                           float *out);

// CPU gated-delta-net recurrence step. Loads `model.layers.<layer_idx>.linear_attn.A_log`
// (f32) and `.dt_bias` (bf16) for the per-head decay precomputation,
// then runs the per-v-head decay → kv_mem → delta → state update →
// output sequence on `ssm_state` (mutated in place) and `out_values`
// (written).
//
//   alpha, beta:  [v_heads] per-step gate inputs.
//   q, k:         [k_heads * key_dim] post-conv-and-bare-norm Q, K.
//   v:            [v_heads * value_dim] post-conv V.
//   ssm_state:    [v_heads * value_dim * key_dim] in/out recurrence state.
//   out_values:   [v_heads * value_dim] written.
//
// `v_heads` must be a multiple of `k_heads` (GQA). Returns 0 on success,
// -1 on NULL args / non-positive shape / missing tensor / shape mismatch.
// Read-only on `ctx` apart from the explicit `ssm_state` mutation.
int mf_gated_delta_recurrence_cpu(mf_ctx *ctx, int32_t layer_idx,
                                   const float *alpha, const float *beta,
                                   const float *q, const float *k,
                                   const float *v,
                                   int32_t v_heads, int32_t k_heads,
                                   int32_t key_dim, int32_t value_dim,
                                   float *ssm_state, float *out_values);

// Single-expert GPU FFN forward. Runs the same 4-dispatch sequence the
// production decode uses for one expert (gate matvec → up matvec →
// SwiGLU → down matvec) on the caller's bytes, then reads the result
// back to the CPU buffer.
//
//   expert_data: [EXPERT_SIZE bytes] one expert's packed weights,
//                gate / up / down blocks at GATE_*_OFF / UP_*_OFF /
//                DOWN_*_OFF as defined in model_variant.h. The active
//                4-bit layout is used (matches the decode path when
//                `use_2bit` was zero at init); 2-bit experts are not
//                yet exposed through this hook.
//   expert_data_len: must equal EXPERT_SIZE for the active variant.
//   h_post:      [HIDDEN_DIM] post-attn-norm hidden state.
//   expert_out:  [HIDDEN_DIM] expert FFN output.
//
// Returns 0 on success, -1 on NULL args, wrong `expert_data_len`, or
// when the ctx was initialized with `use_2bit != 0` (mismatch between
// the layout the caller is providing and what the GPU pipelines expect).
// Read-only on `ctx` apart from the transient internal Metal buffers.
int mf_gpu_expert_forward(mf_ctx *ctx,
                           const void *expert_data, size_t expert_data_len,
                           const float *h_post,
                           float *expert_out);

// Batched K-expert FFN forward + GPU combine. Encodes K parallel
// expert FFNs and then `moe_combine_residual` into a single command
// buffer, runs it, and reads back the post-combine hidden state.
//
// The combine kernel computes:
//
//   hidden_out[i] = h_mid[i]
//                 + Σ_{k<K} expert_weights[k] * expert_out_k[i]
//                 + sigmoid(shared_gate_score) * shared_out[i]
//
// The caller stages all inputs:
//
//   actual_K:           1..16 (clamped at MAX_K=16 — values above are -1).
//   expert_data:        actual_K * EXPERT_SIZE bytes, K expert blobs
//                       laid out in slot order (slot 0 first, then 1, ...).
//                       Layout per-blob is the standard 4-bit
//                       `[gate | up | down]` from `model_variant.h`.
//   expert_data_len:    must equal actual_K * EXPERT_SIZE.
//   h_post:             [HIDDEN_DIM] post-attn-norm hidden state, the
//                       shared input to every expert's gate / up matvec.
//   h_mid:              [HIDDEN_DIM] residual added by the combine
//                       (the pre-MoE hidden state in the production
//                       decode path).
//   shared_out:         [HIDDEN_DIM] the shared expert's output. Pass
//                       zeros + a very negative `shared_gate_score` to
//                       neutralize the shared expert if you only want
//                       routed-experts behaviour.
//   expert_weights:     [actual_K] routing weights (typically a
//                       softmax-normalized top-K).
//   shared_gate_score:  pre-sigmoid gate logit for the shared expert.
//   hidden_out:         [HIDDEN_DIM] post-combine hidden state.
//
// Returns 0 on success; -1 on NULL args, wrong `expert_data_len`,
// `actual_K` out of range, ctx initialized with `use_2bit != 0`, or
// missing Metal pipelines.
int mf_gpu_batched_experts_forward(mf_ctx *ctx,
                                    int32_t actual_K,
                                    const void *expert_data,
                                    size_t expert_data_len,
                                    const float *h_post,
                                    const float *h_mid,
                                    const float *shared_out,
                                    const float *expert_weights,
                                    float shared_gate_score,
                                    float *hidden_out);

// Slice 4e — deferred-experts state machine, three-call API.
//
// Mirrors the production async path inside `fused_layer_forward`
// (`infer.m:5747..5776`). Where `mf_gpu_batched_experts_forward`
// commits + waits + reads back in one call, this trio splits the
// commit-without-wait into `_begin` and the wait+readback into
// `_complete` (or `_discard`). The diff oracle exercises the same
// async state machine that drives layer-to-layer GPU/CPU overlap.
//
// Begin: stages inputs, encodes K-expert FFN + `moe_combine_residual`
// into one cmdbuf, commits async (no wait), saves cmdbuf + per-expert
// metadata in `g_deferred`. Brackets entry with
// `discard_deferred_experts()` so a previous test's leaked state is
// cleared first — same invariant `mf_layer_forward_dump` keeps.
//
// Returns 0 on success; -1 on NULL args, wrong `expert_data_len`,
// `actual_K` out of range, ctx initialized with `use_2bit != 0`,
// missing Metal pipelines, or already-active deferred state. Args
// match `mf_gpu_batched_experts_forward` minus `hidden_out` (caller
// provides that to `_complete` later).
int mf_begin_deferred_experts(mf_ctx *ctx,
                               int32_t actual_K,
                               const void *expert_data,
                               size_t expert_data_len,
                               const float *h_post,
                               const float *h_mid,
                               const float *shared_out,
                               const float *expert_weights,
                               float shared_gate_score);

// Complete: waits for the deferred GPU work, reads back from
// `buf_moe_hidden` into the caller-supplied `hidden_out`, clears
// `g_deferred`. No-op (returns 0) when no deferred state is active —
// mirrors C-internal `complete_deferred_experts`'s
// `if (!g_deferred.active) return;` guard.
//
//   hidden_out: HIDDEN_DIM floats — receives the post-combine hidden.
//
// Returns 0 on success / no-op; -1 on NULL args.
int mf_complete_deferred_experts(mf_ctx *ctx, float *hidden_out);

// Discard: waits for the deferred GPU work (so the persistent
// `MoeBuffers` are no longer in use by the GPU) and clears
// `g_deferred` without reading back. Mirrors C-internal
// `discard_deferred_experts` — used in production for prefill tokens
// where the hidden state is immediately overwritten by the next
// token's embedding.
//
// Returns 0 on success / no-op; -1 on NULL ctx.
int mf_discard_deferred_experts(mf_ctx *ctx);

// Load one expert's packed bytes (`EXPERT_SIZE` for the active 4-bit
// variant) from disk. Reads via `pread(ctx->layer_fds[layer_idx],
// out, EXPERT_SIZE, expert_idx * EXPERT_SIZE)` — same path the cold
// expert-load uses internally, no LRU cache lookup, no LZ4
// decompression. Bypasses every caching layer so the diff oracle gets
// raw on-disk bytes.
//
//   layer_idx, expert_idx: integers in valid range for the active
//                          variant; -1 if out of range.
//   out:                   `out_len` >= EXPERT_SIZE bytes.
//
// Returns 0 on success; -1 on NULL args / out-of-range index /
// missing layer file / wrong `out_len` / 2-bit ctx (the 4-bit
// `EXPERT_SIZE` is hard-coded; 2-bit is a separate slice).
int mf_load_expert_bytes(mf_ctx *ctx,
                          int32_t layer_idx,
                          int32_t expert_idx,
                          void *out,
                          size_t out_len);

// GPU RMSNorm with bf16 weights — chains `rms_norm_sum_sq` (one
// threadgroup, two-stage SIMD-then-shared reduction) followed by
// `rms_norm_apply_bf16` (per-element). Mirrors the production CMD3
// fast-path at `infer.m:5712..5744`. Uses the active variant's
// `RMS_NORM_EPS`. Caller stages all bytes; weight is passed in
// directly so the hook doesn't depend on tensor-name resolution.
//
//   x:               [HIDDEN_DIM] input vector (f32).
//   weight_bf16:     `HIDDEN_DIM * 2` bytes, bf16 weights.
//   weight_bf16_len: must equal `HIDDEN_DIM * 2`.
//   out:             [HIDDEN_DIM] normed output (f32).
//
// Returns 0 on success; -1 on NULL args, wrong `weight_bf16_len`, or
// missing GPU pipelines.
int mf_gpu_rms_norm_fused(mf_ctx *ctx,
                           const float *x,
                           const void *weight_bf16,
                           size_t weight_bf16_len,
                           float *out);

// ---------------------------------------------------------------------------
// Slice 5d-7a — GPU full-attention kernels under per-kernel diff
// ---------------------------------------------------------------------------
//
// Test-only oracle wrappers around the four kernels that make up
// `gpu_attn_fuse` (`infer.m:5051..5163`):
//
//   mf_attn_scores_batched   — Q · K^T per head, scaled
//   mf_attn_softmax_batched  — per-head softmax over [0, seq_len)
//   mf_attn_values_batched   — scores · V per head
//   mf_sigmoid_gate          — out[i] *= sigmoid(gate[i])
//
// Each hook allocates its own scratch GPU buffers, encodes ONE kernel
// into a fresh cmdbuf, commits + waits, and reads back. Same shape as
// `mf_gpu_rms_norm_fused`. The C production path's monolithic
// `gpu_attn_fuse` is left untouched; these oracle wrappers duplicate
// encoder logic with fresh buffers so the pre-existing tests against
// `gpu_attn_fuse`'s runtime behavior continue to compare bit-exact.
//
// Stride convention: oracle output uses `seq_stride = seq_len` (tight
// packing). Production callers use `seq_stride = GPU_KV_SEQ` for the
// persistent score buffer; the kernels write the same per-row values
// regardless of stride choice, so test vs. production agree per-element
// after slicing the strided rows.
//
// `kv_dim = num_kv_heads * head_dim`; `heads_per_kv = num_heads /
// num_kv_heads` (GQA). `num_heads` must be a multiple of `num_kv_heads`.

// Compute `scores[h, p] = (Q[h] · K_cache[p, kv_h]) * scale` for all
// h in [0, num_heads) and p in [0, seq_len). GQA: each query head h
// maps to kv_head `h / heads_per_kv`.
//
//   q:           [num_heads * head_dim] post-RoPE query.
//   k_cache:     [seq_len * kv_dim] post-RoPE K rows, packed prefix.
//   scale:       1/sqrt(head_dim) (passed explicitly).
//   scores_out:  [num_heads * seq_len] — per-head, per-position scores.
//                Output stride is `seq_len` (no padding).
//
// Returns 0 on success; -1 on NULL args / non-positive shape /
// `num_heads % num_kv_heads != 0` / missing pipeline.
int mf_attn_scores_batched(mf_ctx *ctx,
                            int32_t num_heads,
                            int32_t num_kv_heads,
                            int32_t head_dim,
                            int32_t seq_len,
                            const float *q,
                            const float *k_cache,
                            float scale,
                            float *scores_out);

// Per-head softmax over `[0, seq_len)`. Numerically stable (max-shift
// before exp). Mutates `scores_inout` in place.
//
//   scores_inout: [num_heads * seq_len] — per-head, per-position
//                 logits in, softmax probs out.
//
// Returns 0 on success; -1 on NULL args / non-positive shape /
// missing pipeline.
int mf_attn_softmax_batched(mf_ctx *ctx,
                             int32_t num_heads,
                             int32_t seq_len,
                             float *scores_inout);

// Aggregate values: `out[h * head_dim + d] = Σ_p scores[h, p] *
// V_cache[p, kv_h, d]`.
//
//   scores:  [num_heads * seq_len] post-softmax probs.
//   v_cache: [seq_len * kv_dim] V rows, packed prefix.
//   out:     [num_heads * head_dim] aggregated output.
//
// Returns 0 on success; -1 on NULL args / non-positive shape /
// `num_heads % num_kv_heads != 0` / missing pipeline.
int mf_attn_values_batched(mf_ctx *ctx,
                            int32_t num_heads,
                            int32_t num_kv_heads,
                            int32_t head_dim,
                            int32_t seq_len,
                            const float *scores,
                            const float *v_cache,
                            float *out);

// Element-wise sigmoid gate: `x_inout[i] *= sigmoid(gate[i])`. Per-
// thread, no reductions — bit-exact per-PSO.
//
//   gate:      [dim] pre-sigmoid gate logits.
//   x_inout:   [dim] in: pre-gated; out: gated values.
//
// Returns 0 on success; -1 on NULL args / non-positive `dim` /
// missing pipeline.
int mf_sigmoid_gate(mf_ctx *ctx,
                     int32_t dim,
                     const float *gate,
                     float *x_inout);

// Run a single layer's forward pass starting from the supplied hidden
// state and return the post-layer hidden state. Phase 4 diff-oracle
// dump point — the layer-boundary checkpoint shape that lets the
// per-kernel diffs (which already match) compose into a per-layer
// signal as the Rust port wires up `fused_layer_forward`.
//
// Drives the targeted layer's per-layer state (KV cache for full-
// attention layers, GatedDeltaNet recurrence for linear-attention
// layers) using the ctx's existing arrays. The dump path flushes any
// pending deferred-expert work before returning, leaving
// `g_deferred.active` at 0 on both entry and exit so consecutive calls
// don't bleed state into each other.
//
//   ctx:        the moeflux context.
//   layer_idx:  0 ≤ layer_idx < NUM_LAYERS for the active variant.
//   pos:        position in the sequence (0 = first token).
//               Updates the layer's KV / recurrence state in place.
//   hidden_in:  HIDDEN_DIM floats — the post-layer-(N-1) hidden state.
//   hidden_out: HIDDEN_DIM floats — receives the post-layer-N hidden.
//
// Returns 0 on success, -1 on NULL args / out-of-range layer / pos<0.
int mf_layer_forward_dump(mf_ctx *ctx,
                          int32_t layer_idx,
                          int32_t pos,
                          const float *hidden_in,
                          float *hidden_out);

// 4c diagnostic: same as mf_layer_forward_dump, but also copies out
// the intermediate Metal buffers so the differential test harness can
// pinpoint where C-vs-Rust divergence appears. All `*_out` buffers are
// written from GPU shared-storage state after
// `complete_deferred_experts` finalizes the layer.
//
//   h_post_out:      HIDDEN_DIM floats — post-attn-norm hidden (the
//                    input to the gate / shared / expert matvecs).
//                    Read from `g_metal->buf_input`.
//   h_mid_out:       HIDDEN_DIM floats — residual + o_proj output.
//                    Read from `g_metal->buf_h_mid`.
//   shared_out_out:  HIDDEN_DIM floats — pre-sigmoid-gate shared-
//                    expert output. Read from `g_metal->buf_shared_out`.
//   gate_score_out:  one float — pre-sigmoid shared-expert gate score.
//                    Read from `g_deferred.shared_gate_score`.
//
// Any *_out may be NULL to skip that copy. The hook leaves the ctx in
// the same post-call state as `mf_layer_forward_dump`.
int mf_layer_forward_dump_intermediates(mf_ctx *ctx,
                                        int32_t layer_idx,
                                        int32_t pos,
                                        const float *hidden_in,
                                        float *hidden_out,
                                        float *h_post_out,
                                        float *h_mid_out,
                                        float *shared_out_out,
                                        float *gate_score_out);

// ============================================================================
// State snapshot / restore (Option B in NOTES.md)
// ============================================================================
//
// Let callers serialize the full inference state — KV caches for the
// 15 full-attention layers, GatedDeltaNet recurrence buffers for the
// 45 linear-attention layers — into an opaque byte buffer, and later
// restore from it. Enables prefix-cache reuse across evaluations
// without the re-prefill cost that mf_memory_seq_rm otherwise forces
// on linear layers.
//
// Intended callsite shape in drama_llama:
//
//   // After evaluating a prefix the caller wants to cache:
//   size_t n = mf_state_size(ctx);
//   void *snapshot = malloc(n);
//   mf_state_save(ctx, snapshot, n);
//   // ... later, on cache hit for the same prefix ...
//   mf_memory_clear(ctx);
//   mf_state_load(ctx, snapshot, n);
//   // ctx now matches the state at snapshot time. Decode continues
//   // from the recorded position (visible via mf_memory_seq_pos_max).
//
// Snapshot size scales linearly with the KV length at the time of
// save. For a prefix of P positions on Qwen3.5-397B the snapshot is
// roughly (15 * 2 * P * NUM_KV_HEADS * HEAD_DIM * 4B) for the KV
// portion plus a fixed ~190MB for the GatedDeltaNet recurrence state
// (not proportional to P — recurrence state has the same shape at
// every position).
//
// Constraints:
// - Call only at token boundaries (after mf_eval_prompt or
//   mf_eval_token returns). Pending deferred GPU expert compute must
//   have finalized.
// - The binary format encodes the model's shape constants; loading a
//   snapshot into a moeflux built with different constants fails.
// - The CPU-fallback linear-attention path is not snapshotted. Save
//   / load is only correct on the Metal GPU path (the default for
//   moeflux's target hardware).

// Byte size the caller must allocate to hold the current snapshot.
// Return value changes as KV length changes, so query after each
// evaluation if unsure. Returns 0 on NULL ctx.
size_t mf_state_size(const mf_ctx *ctx);

// Serialize the current state into `buf`. `buf_len` must be at least
// the value returned by mf_state_size. Returns the number of bytes
// actually written on success, or -1 on failure (NULL ctx / buf, or
// buf_len too small).
long long mf_state_save(mf_ctx *ctx, void *buf, size_t buf_len);

// Replace the current state with the one encoded in `buf`. Returns 0
// on success, -1 on failure (NULL args, truncated/corrupt buffer,
// mismatched model-shape header). On failure the ctx state is
// undefined — caller should follow up with mf_memory_clear.
int mf_state_load(mf_ctx *ctx, const void *buf, size_t buf_len);

#ifdef __cplusplus
}
#endif

#endif // MOEFLUX_H
