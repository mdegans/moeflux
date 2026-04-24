# Model variants

moeflux compiles for one model shape at a time. Model shape is
*compile-time*, not runtime, because:

- C's static array sizes (`int arr[NUM_LAYERS][NUM_EXPERTS]`) require
  compile-time constants.
- Metal shader dispatch dims and tile sizes are easier to reason about
  when the dims are folded into the binary.
- Runtime model parameterization is deferred to a future "Phase 5"
  effort (see `.claude/memory/plan_v0.8.0_backend_split.md` on the
  drama_llama side). For now, we ship one binary per target model.

## How selection works

A single header — [`metal_infer/model_variant.h`](../metal_infer/model_variant.h)
— owns all shape-dependent `#define`s. Select a variant by defining
exactly one `-DMOEFLUX_MODEL_*` flag at compile time:

```
cc -DMOEFLUX_MODEL_QWEN3_5_A17B       ...  # original target
cc -DMOEFLUX_MODEL_QWEN3_6_35B_A3B    ...  # smaller drama_llama test target
```

No flag → defaults to `QWEN3_5_A17B` (preserves pre-refactor behavior).
Two flags → `#error`.

From `metal_infer/`, the Makefile accepts `MODEL=`:

```
make lib                             # A17B (default)
make lib MODEL=qwen3-5-a17b          # A17B (explicit)
make lib MODEL=qwen3-6-35b-a3b       # 35B-A3B
make smoke MODEL=qwen3-6-35b-a3b     # smoke linked against 35B-A3B build
```

Unknown `MODEL=` values fail fast with an `$(error …)`.

## Supported variants

| Variant                      | Flag                              | Source                                       |
| ---------------------------- | --------------------------------- | -------------------------------------------- |
| Qwen3.5-397B-A17B-4bit       | `MOEFLUX_MODEL_QWEN3_5_A17B`      | `mlx-community/Qwen3.5-397B-A17B-4bit`       |
| Qwen3.6-35B-A3B-4bit         | `MOEFLUX_MODEL_QWEN3_6_35B_A3B`   | `Qwen/Qwen3.6-35B-A3B` (convert to MLX 4-bit) |

### Shape table

All variants are the `qwen3_5_moe` architecture family: attention-
output-gate, mixed linear-attention + full-attention (3:1 ratio,
`FULL_ATTN_INTERVAL=4`), MTP head, same Qwen tokenizer (vocab
248320). The `qwen3_moe` (original Qwen3, no SSM) family is **not**
currently supported.

| Parameter              | A17B   | 35B-A3B |
| ---------------------- | ------ | ------- |
| `HIDDEN_DIM`           | 4096   | 2048    |
| `NUM_LAYERS`           | 60     | 40      |
| `NUM_ATTN_HEADS`       | 32     | 16      |
| `NUM_KV_HEADS`         | 2      | 2       |
| `HEAD_DIM`             | 256    | 256     |
| `NUM_EXPERTS`          | 512    | 256     |
| `NUM_EXPERTS_PER_TOK`  | 10     | 8       |
| `MOE_INTERMEDIATE`     | 1024   | 512     |
| `SHARED_INTERMEDIATE`  | 1024   | 512     |
| `LINEAR_NUM_V_HEADS`   | 64     | 32      |
| `LINEAR_NUM_K_HEADS`   | 16     | 16      |
| `EXPERT_SIZE` (bytes)  | 7077888 | 1769472 |

`EXPERT_SIZE` and all `*_OFF` byte offsets are **derived** from
`HIDDEN_DIM`, `MOE_INTERMEDIATE`, `GROUP_SIZE`, and `BITS` — do not
hand-tune them. See the `_MF_EXPERT_*` helper macros in the header.

## Adding a new variant

1. Pull the shape constants from the model's HuggingFace `config.json`.
   Read `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
   `num_key_value_heads`, `head_dim`, `num_experts`,
   `num_experts_per_tok`, `moe_intermediate_size`,
   `shared_expert_intermediate_size`, `linear_num_value_heads`,
   `linear_num_key_heads`.
2. Confirm the architecture is `qwen3_5_moe` (check `model_type`). If
   it isn't — especially `qwen3_moe` without SSM — expect substantial
   kernel work beyond just flipping `#define`s; not supported yet.
3. Add an `#elif defined(MOEFLUX_MODEL_YOUR_NAME)` block in
   `model_variant.h`.
4. Add a matching `ifeq ($(MODEL),your-name)` branch in the Makefile.
5. When `crates/moeflux-sys` exists, add a matching Cargo feature and
   wire `build.rs` to pass the `-D` flag to `cc`.
6. Generate `expert_index.json` for your MLX-converted weights and run
   `repack_experts.py` with parameters matching your shape.
7. Run `make smoke MODEL=your-name && ./smoke …` to validate.

## Why one binary per model

The alternative is runtime parameterization: pass `NUM_LAYERS`,
`NUM_EXPERTS`, etc. as function arguments and heap-allocate all
previously-static arrays. That's a substantial rewrite affecting
most of `infer.m`'s ~7800 lines and would not improve the drama_llama
consumer path (drama_llama picks the backend at build time too).

If moeflux ever grows a config-driven "load any Qwen MoE" mode, it
lives alongside the compile-time path, not in place of it.
