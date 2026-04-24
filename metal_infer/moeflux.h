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
