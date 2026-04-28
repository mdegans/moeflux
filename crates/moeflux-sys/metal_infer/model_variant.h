// moeflux — per-model shape constants.
//
// Model shape is compile-time in moeflux: C requires static array sizes
// and Metal shader kernels reference these dimensions. Switch variants
// by defining exactly one of the MOEFLUX_MODEL_* flags at build time.
//
// Currently supported variants (Qwen MoE family, qwen3_5_moe architecture):
//   -DMOEFLUX_MODEL_QWEN3_5_A17B      (default — Qwen3.5-397B-A17B-4bit)
//   -DMOEFLUX_MODEL_QWEN3_6_35B_A3B   (Qwen3.6-35B-A3B-4bit — smaller
//                                      drama_llama cross-backend test target)
//
// Adding a new variant:
//   1. Add a new `#elif defined(MOEFLUX_MODEL_*)` block below.
//   2. Fill the shape constants from the HuggingFace config.json.
//   3. The derived offsets (EXPERT_SIZE, UP_W_OFF, etc.) are computed
//      from the shape — you do not hand-tune them.
//   4. Add a matching Cargo feature in crates/moeflux-sys.
//   5. Regenerate expert_index.json for the converted MLX weights.
//
// Constraints all current variants must satisfy (not hard-enforced,
// but the MoE and linear-attention code paths assume them):
//   - 4-bit quantization with GROUP_SIZE=64, affine mode (matches MLX default).
//   - qwen3_5_moe architecture (attn_output_gate + linear_attention + MTP).
//   - Same Qwen tokenizer (vocab 248320).
//
// See docs/model_variants.md for the full per-variant parameter table.

#ifndef MOEFLUX_MODEL_VARIANT_H
#define MOEFLUX_MODEL_VARIANT_H

// --- Variant selection --------------------------------------------------

#if !defined(MOEFLUX_MODEL_QWEN3_5_A17B) && \
    !defined(MOEFLUX_MODEL_QWEN3_6_35B_A3B)
#define MOEFLUX_MODEL_QWEN3_5_A17B
#endif

#if (defined(MOEFLUX_MODEL_QWEN3_5_A17B) ? 1 : 0) + \
    (defined(MOEFLUX_MODEL_QWEN3_6_35B_A3B) ? 1 : 0) > 1
#error "Define exactly one MOEFLUX_MODEL_* variant."
#endif

// --- Per-variant shape constants ---------------------------------------

#if defined(MOEFLUX_MODEL_QWEN3_5_A17B)

#define MOEFLUX_MODEL_NAME    "Qwen3.5-397B-A17B-4bit"
#define HIDDEN_DIM            4096
#define NUM_LAYERS            60
#define NUM_ATTN_HEADS        32
#define NUM_KV_HEADS          2
#define HEAD_DIM              256
#define VOCAB_SIZE            248320
#define NUM_EXPERTS           512
#define NUM_EXPERTS_PER_TOK   10
#define MOE_INTERMEDIATE      1024
#define SHARED_INTERMEDIATE   1024
#define FULL_ATTN_INTERVAL    4
#define LINEAR_NUM_V_HEADS    64
#define LINEAR_NUM_K_HEADS    16

#elif defined(MOEFLUX_MODEL_QWEN3_6_35B_A3B)

#define MOEFLUX_MODEL_NAME    "Qwen3.6-35B-A3B-4bit"
#define HIDDEN_DIM            2048
#define NUM_LAYERS            40
#define NUM_ATTN_HEADS        16
#define NUM_KV_HEADS          2
#define HEAD_DIM              256
#define VOCAB_SIZE            248320
#define NUM_EXPERTS           256
#define NUM_EXPERTS_PER_TOK   8
#define MOE_INTERMEDIATE      512
#define SHARED_INTERMEDIATE   512
#define FULL_ATTN_INTERVAL    4
#define LINEAR_NUM_V_HEADS    32
#define LINEAR_NUM_K_HEADS    16

#else
#error "No MOEFLUX_MODEL_* variant defined (internal: default-selection bug)."
#endif

// --- Shared constants (architecture-wide, not shape-dependent) --------

#define RMS_NORM_EPS          1e-6f
#define GROUP_SIZE            64
#define BITS                  4

#define LINEAR_KEY_DIM        128
#define LINEAR_VALUE_DIM      128
#define LINEAR_TOTAL_KEY      (LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM)
#define LINEAR_TOTAL_VALUE    (LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM)
#define LINEAR_CONV_DIM       (LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE)
#define CONV_KERNEL_SIZE      4

#define ROPE_THETA            10000000.0f
#define PARTIAL_ROTARY        0.25f
#define ROTARY_DIM            (int)(HEAD_DIM * PARTIAL_ROTARY)

// Qwen tokenizer special tokens (shared across qwen3_5_moe variants).
#define EOS_TOKEN_1           248046
#define EOS_TOKEN_2           248044
#define THINK_START_TOKEN     248068
#define THINK_END_TOKEN       248069

// --- 4-bit packed-expert layout (derived from shape) -------------------
//
// Layout per expert:
//   [gate_w | gate_s | gate_b | up_w | up_s | up_b | down_w | down_s | down_b]
// gate/up/down all have the same byte count in 4-bit:
//   weights: MOE_INTERMEDIATE * HIDDEN_DIM * BITS / 8 (U32-packed)
//   scales:  MOE_INTERMEDIATE * (HIDDEN_DIM / GROUP_SIZE) * 2 (BF16)
//   biases:  same as scales
// (down_w/s/b have the dims transposed but the byte totals are identical.)

#define _MF_EXPERT_WEIGHT_BYTES  (MOE_INTERMEDIATE * HIDDEN_DIM * BITS / 8)
#define _MF_EXPERT_SCALE_BYTES   (MOE_INTERMEDIATE * (HIDDEN_DIM / GROUP_SIZE) * 2)
#define _MF_EXPERT_BLOCK_BYTES   (_MF_EXPERT_WEIGHT_BYTES + 2 * _MF_EXPERT_SCALE_BYTES)

#define GATE_W_OFF   0
#define GATE_S_OFF   _MF_EXPERT_WEIGHT_BYTES
#define GATE_B_OFF   (_MF_EXPERT_WEIGHT_BYTES + _MF_EXPERT_SCALE_BYTES)
#define UP_W_OFF     _MF_EXPERT_BLOCK_BYTES
#define UP_S_OFF     (_MF_EXPERT_BLOCK_BYTES + _MF_EXPERT_WEIGHT_BYTES)
#define UP_B_OFF     (_MF_EXPERT_BLOCK_BYTES + _MF_EXPERT_WEIGHT_BYTES + _MF_EXPERT_SCALE_BYTES)
#define DOWN_W_OFF   (2 * _MF_EXPERT_BLOCK_BYTES)
#define DOWN_S_OFF   (2 * _MF_EXPERT_BLOCK_BYTES + _MF_EXPERT_WEIGHT_BYTES)
#define DOWN_B_OFF   (2 * _MF_EXPERT_BLOCK_BYTES + _MF_EXPERT_WEIGHT_BYTES + _MF_EXPERT_SCALE_BYTES)
#define EXPERT_SIZE  (3 * _MF_EXPERT_BLOCK_BYTES)

// --- 2-bit packed-expert layout (derived from shape) -------------------
//
// Same block structure as 4-bit, but weights are half the byte count
// (U32 packs 16 × 2-bit values instead of 8 × 4-bit).
// Scales/biases are unchanged (still BF16, grouped by GROUP_SIZE).

#define _MF_EXPERT_WEIGHT_BYTES_2BIT  (MOE_INTERMEDIATE * HIDDEN_DIM * 2 / 8)
#define _MF_EXPERT_BLOCK_BYTES_2BIT   (_MF_EXPERT_WEIGHT_BYTES_2BIT + 2 * _MF_EXPERT_SCALE_BYTES)

#define GATE_W_OFF_2   0
#define GATE_S_OFF_2   _MF_EXPERT_WEIGHT_BYTES_2BIT
#define GATE_B_OFF_2   (_MF_EXPERT_WEIGHT_BYTES_2BIT + _MF_EXPERT_SCALE_BYTES)
#define UP_W_OFF_2     _MF_EXPERT_BLOCK_BYTES_2BIT
#define UP_S_OFF_2     (_MF_EXPERT_BLOCK_BYTES_2BIT + _MF_EXPERT_WEIGHT_BYTES_2BIT)
#define UP_B_OFF_2     (_MF_EXPERT_BLOCK_BYTES_2BIT + _MF_EXPERT_WEIGHT_BYTES_2BIT + _MF_EXPERT_SCALE_BYTES)
#define DOWN_W_OFF_2   (2 * _MF_EXPERT_BLOCK_BYTES_2BIT)
#define DOWN_S_OFF_2   (2 * _MF_EXPERT_BLOCK_BYTES_2BIT + _MF_EXPERT_WEIGHT_BYTES_2BIT)
#define DOWN_B_OFF_2   (2 * _MF_EXPERT_BLOCK_BYTES_2BIT + _MF_EXPERT_WEIGHT_BYTES_2BIT + _MF_EXPERT_SCALE_BYTES)
#define EXPERT_SIZE_2BIT  (3 * _MF_EXPERT_BLOCK_BYTES_2BIT)

// --- KV cache / runtime limits (not model-variant-specific) ------------

#define MAX_SEQ_LEN  1048576
#define GPU_KV_SEQ   8192

#endif  // MOEFLUX_MODEL_VARIANT_H
