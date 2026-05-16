# Model variants

moeflux compiles for one model shape at a time. Model shape is
*compile-time*, not runtime: the per-variant constants are folded into
the binary, and Metal shader dispatch dims are simpler to reason about
that way. Runtime model parameterization is future work — for now,
one build per target model.

## How selection works

Select a variant with exactly one Cargo feature:

```bash
cargo build --features model-qwen3-5-a17b      # original target
cargo build --features model-qwen3-6-35b-a3b   # smaller test target
cargo build --features model-cogito-v2-671b    # DeepSeek-V3 architecture
```

The feature gates the shape constants in `src/riir/variants.rs`.
Selecting no feature, or more than one, is a build error.

## Supported variants

| Variant                | Feature                 | Source                                        |
| ---------------------- | ----------------------- | --------------------------------------------- |
| Qwen3.5-397B-A17B-4bit | `model-qwen3-5-a17b`    | `mlx-community/Qwen3.5-397B-A17B-4bit`        |
| Qwen3.6-35B-A3B-4bit   | `model-qwen3-6-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` (convert to MLX 4-bit) |
| Cogito-V2-Preview-671B | `model-cogito-v2-671b`  | DeepSeek-V3 architecture                      |

### Shape table — Qwen variants

The Qwen variants are the `qwen3_5_moe` architecture family:
attention-output-gate, mixed linear-attention + full-attention (3:1,
`FULL_ATTN_INTERVAL=4`), MTP head, Qwen tokenizer (vocab 248320).

| Parameter             | A17B    | 35B-A3B |
| --------------------- | ------- | ------- |
| `HIDDEN_DIM`          | 4096    | 2048    |
| `NUM_LAYERS`          | 60      | 40      |
| `NUM_ATTN_HEADS`      | 32      | 16      |
| `NUM_KV_HEADS`        | 2       | 2       |
| `HEAD_DIM`            | 256     | 256     |
| `NUM_EXPERTS`         | 512     | 256     |
| `NUM_EXPERTS_PER_TOK` | 10      | 8       |
| `MOE_INTERMEDIATE`    | 1024    | 512     |
| `SHARED_INTERMEDIATE` | 1024    | 512     |
| `LINEAR_NUM_V_HEADS`  | 64      | 32      |
| `LINEAR_NUM_K_HEADS`  | 16      | 16      |
| `EXPERT_SIZE` (bytes) | 7077888 | 1769472 |

`EXPERT_SIZE` and the `*_OFF` byte offsets are *derived* from
`HIDDEN_DIM`, `MOE_INTERMEDIATE`, `GROUP_SIZE`, and `BITS` — do not
hand-tune them.

Cogito-V2-671B is a separate (DeepSeek-V3) architecture — MLA
attention, not the qwen3_5_moe linear/full mix. Its shape lives in
`variants.rs` under the same feature mechanism.

## Adding a new variant

1. Pull the shape constants from the model's HuggingFace
   `config.json`.
2. Confirm the architecture. The `qwen3_5_moe` family drops in by
   constants alone; a different architecture (e.g. plain `qwen3_moe`
   without SSM, or a new MLA model) needs kernel work.
3. Add a `model-your-name` feature in `crates/moeflux/Cargo.toml` and
   a matching shape block in `src/riir/variants.rs`.
4. Run the model-prep pipeline in `scripts/` against your
   MLX-converted weights.
5. Build with the new feature and run the oracle suites against it.
