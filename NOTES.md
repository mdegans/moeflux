# moeflux — working notes

Session-persistent notes for Claude + Mike while reshaping
`danveloper/flash-moe` into a Rust-downstream library. Not a user-
facing doc; think of this as the lab notebook.

## What this fork is for

Exposed as a Rust crate with a stable C FFI so drama_llama's Phase
4 can drop in a `MoefluxDecoder` / `MoefluxModel` alongside
`LlamaCppDecoder` / `LlamaCppModel`. Downstream goal: Cogito 600B
running locally for Agora's Council independence path.

## Code orientation (settled Phase 3b.1)

### chat.m is an HTTP/SSE client, NOT the reference inference loop

Scout doc was wrong; fresh survey was right. `chat.m` opens with
"Interactive TUI chat client for Flash-MoE inference server — Thin
HTTP/SSE client with session persistence." It posts to a local
server and streams responses. Safe to delete — no inference logic
inside.

The **real** reference inference loop is `serve_loop()` in
`metal_infer/infer.m` (line 5965). That's the HTTP server end; it
orchestrates the full per-token compute flow. Our `mf_eval_token`
and `mf_eval_prompt` should mirror the inner loop of serve_loop.

### No existing `eval_token(token, pos) -> logits[vocab]`

The per-token forward pass is driven manually by serve_loop:
1. `embed_lookup()` (infer.m:2009) — token → hidden state
2. For each of 60 layers: `fused_layer_forward()` (infer.m:3998)
3. `complete_deferred_experts()` (infer.m:3884) — finalize GPU
   expert output for the last layer
4. `lm_head_forward()` (infer.m:2914) — hidden → logits
5. (optional) `cpu_argmax()` (infer.m:843) — sampling; we skip this

`mf_eval_token` orchestrates steps 1–4 and returns raw logits.
`mf_eval_prompt` does the same in batch (multi-token prefill) —
serve_loop has the precedent.

### KV cache is per-layer, simple, single-sequence

```c
typedef struct {
    float *k_cache;  // [max_seq, num_kv_heads * head_dim]
    float *v_cache;  // [max_seq, num_kv_heads * head_dim]
    int len;         // current number of cached entries
} KVCache;
```

- One per full-attention layer (15 layers).
- `kv->len` is the write cursor. Updated at infer.m:2264, 4531.
- **No seq_id support.** Single sequence only. `mf_*` API accepts
  `seq_id` for forward compatibility but ignores it.
- `mf_memory_seq_rm(start, end)` = reset `kv->len = start` on every
  full-attn layer, optionally memset `[start*stride, end*stride)`.
  ~50 LOC as the scout doc predicted.

### ⚠ Linear-attention layers are not position-truncatable (Option A now, B later)

45 of the 60 layers are GatedDeltaNet linear-attention. They hold
a recurrence state (`conv_state` + `ssm_state`) that is NOT
position-indexed. `mf_memory_seq_rm` cannot cleanly truncate them.

**Phase 3/4 posture: Option A — full reset.** `mf_memory_seq_rm`
zeroes linear-attn state entirely. The caller (drama_llama) must
re-prefill from position 0 after any truncation. This means the
drama_llama prefix cache gets NO linear-attn benefit — full
re-prefill on every prefix-miss.

**Option B (deferred; try at end of current session if context
allows):** snapshot `(conv_state, ssm_state)` for all linear
layers at `Prompt`-primitive breakpoints, restore on cache hit.
Avoids re-ingestion at the cost of extra memory per cached prefix.
Without B, Cogito 600B on 96GB Mac will barely hit its tok/s
target because every Council turn re-prefills the full thread
history.

Required for B:
- `mf_state_save(buf) / mf_state_load(buf)` — serialize/deserialize
  all per-layer state (full-attn KV + linear-attn recurrence).
- drama_llama-side: cache entry stores the opaque state bytes in
  addition to the prefix tokens.
- Not "major surgery if we're careful" — Mike's words; agreed.

### Shape is runtime-parameterizable at the shader level

shaders.metal takes shape as `constant uint` buffer args
(out_dim, in_dim, group_size, etc.). Kernel compile is NOT shape-
specialized. Swapping models with different HIDDEN_DIM / NUM_LAYERS
/ etc. only needs:
- Update `#define`s in infer.m (or eventually: pass as runtime
  params)
- Rebuild infer.m (but not shaders)

For Phase 3 smoke, rebuild-per-model is fine. Runtime params are a
Phase 5 (Cogito 600B) concern.

### Model shape constants (Qwen3.5-397B-A17B as checked in)

```c
// infer.m:72–100
HIDDEN_DIM          4096
NUM_LAYERS          60
NUM_ATTN_HEADS      32
NUM_KV_HEADS        2
HEAD_DIM            256
VOCAB_SIZE          248320
RMS_NORM_EPS        1e-6f
NUM_EXPERTS         512
NUM_EXPERTS_PER_TOK 10
MOE_INTERMEDIATE    1024
SHARED_INTERMEDIATE 1024
FULL_ATTN_INTERVAL  4  // 1 full, 3 linear, repeat
GROUP_SIZE          64
BITS                4

// Linear attention (GatedDeltaNet)
LINEAR_NUM_V_HEADS  64
LINEAR_NUM_K_HEADS  16
LINEAR_KEY_DIM      128
LINEAR_VALUE_DIM    128
CONV_KERNEL_SIZE    4

// RoPE
ROPE_THETA          10000000.0f
PARTIAL_ROTARY      0.25f

// Expert sizes (packed layout)
EXPERT_SIZE         7077888
EXPERT_SIZE_2BIT    3932160
```

### Global state (flag for later)

File-scope globals in infer.m: `g_metal` (line 992), `g_expert_cache`
(line 3226), `g_io_pool`, `g_deferred`. Prevents two models in one
process. Acceptable for drama_llama's current use — one Engine per
process via Session. If multi-model is ever needed, wrap in an
`mf_ctx` struct.

### Tokenizer plug points

- Encode: `encode_prompt_text_to_tokens()` (infer.m:668) — wraps
  `bpe_encode` from tokenizer.h.
- Decode: `decode_token()` (infer.m:613).

Phase 3: keep tokenizer.h for smoke-test convenience. Phase 4:
drama_llama tokenizes via HuggingFace `tokenizers` crate and passes
token IDs across FFI; `mf_*` API never sees text.

## Sampling

`cpu_argmax()` (infer.m:843) — greedy argmax only. moeflux does
NO sampling; `mf_eval_prompt` / `mf_eval_token` return raw logits
and drama_llama's sampling chain handles the rest.

## Build

Existing Makefile produces two binaries: `infer` and `chat`. We
need:
- A new target: `libmoeflux.a` — static library for Rust FFI.
- Drop the `chat` target after chat.m is stripped.
- Keep `infer` target (useful for C-level smoke testing).

## Strip list — Phase 3b.2 (being conservative: keep what infer.m still needs)

**Deleting (standalone, no inbound deps):**
- `metal_infer/chat.m` — HTTP/SSE client, replaced by drama_llama
- `metal_infer/linenoise.c`, `linenoise.h` — chat.m's dep
- `metal_infer/test_apfs_compress.sh`, `test_lzfse.c` — failed
  compression experiments per upstream's paper
- `metal_infer/train_predictor.py` — router training; not runtime
- `metal_infer/repack_experts_2bit.py` — 2-bit pipeline, breaks
  tool calling per upstream README; can re-add if we want 2-bit
- `metal_infer/repack_experts_lz4.c` — LZ4 compressed cache, marked
  degraded in upstream
- `metal_infer/Makefile` — will be updated (chat target removed,
  libmoeflux.a target added)

**Keeping (still needed for now):**
- `metal_infer/infer.m` — the core; heavy edits in 3b.3 / 3b.4
- `metal_infer/shaders.metal` — Metal kernels, untouched
- `metal_infer/tokenizer.h` — delete only when Rust tokenization
  lands in Phase 4
- `metal_infer/export_tokenizer.py` — produces tokenizer.bin that
  tokenizer.h consumes; delete with tokenizer.h
- `metal_infer/main.m` — keep until 3b.4; may extract test code
  for `tests/smoke.c`
- Top-level: `repack_experts.py`, `extract_weights.py`,
  `expert_index.json`, `progress.png`, `progress.py`,
  `results.tsv`, `paper/`, `docs/`, `CLAUDE.md` — preserve for
  attribution and model-prep recipe

## Open tasks in order

- [ ] 3b.2 strip + Makefile update (+ verify `infer` still builds)
- [ ] 3b.3 add `mf_memory_seq_rm`, `mf_memory_clear`,
      `mf_memory_seq_pos_max`
- [ ] 3b.4 add `mf_*` C API (moeflux.h + wrapper functions) +
      `libmoeflux.a` target
- [ ] 3b.5 C smoke test (tests/smoke.c)
- [ ] (deferred) Option B: `mf_state_save` / `mf_state_load` for
      linear-attn state snapshotting
- [ ] (Phase 4) Replace tokenizer.h with Rust-side tokenization

## Key line-number references

- `infer.m:72–100` — model shape #defines
- `infer.m:521` — `open_weights()`
- `infer.m:581` — `load_vocab()`
- `infer.m:613` — `decode_token()`
- `infer.m:668` — `encode_prompt_text_to_tokens()`
- `infer.m:843` — `cpu_argmax()` (ignore; we don't sample here)
- `infer.m:994` — `metal_setup()`
- `infer.m:2009` — `embed_lookup()`
- `infer.m:2071` — `kv_cache_new()`
- `infer.m:2079` — `kv_cache_free()`
- `infer.m:2096` — `linear_attn_state_new()`
- `infer.m:2264` — first `kv->len` write site (seq_rm hook)
- `infer.m:2914` — `lm_head_forward()`
- `infer.m:3226` — `g_expert_cache` global
- `infer.m:3884` — `complete_deferred_experts()`
- `infer.m:3998` — `fused_layer_forward()`
- `infer.m:4531` — second `kv->len` write site
- `infer.m:5965` — `serve_loop()` (reference per-token flow)
