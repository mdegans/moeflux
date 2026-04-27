//! LM head matvec — Rust port of `infer.m`'s `cpu_dequant_matvec`
//! specialized for the `lm_head.{weight,scales,biases}` tensors.
//!
//! `out[row] = Σ_i (dequant(W[row, i]) * x[i])` for `row` in
//! `0..VOCAB_SIZE`, where dequant decodes a 4-bit nibble through a
//! per-group bf16 scale + bias. The C path's `lm_head_forward`
//! delegates to `fast_dequant_matvec`, which hands off to the GPU when
//! Metal is available; for the diff oracle we route both sides through
//! the deterministic CPU path so per-row outputs can be compared head-
//! on.
//!
//! ## Tolerance regime
//!
//! Bit-exact on the same hardware via explicit `mul_add`. The C source
//! statement `acc += ((float)val * scale + bias) * x[i]` is FMA-fused
//! by Apple clang at `-O3` (default `-ffp-contract=on`) into two
//! `fmadd` instructions on AArch64: `T = fma(val, scale, bias)` and
//! `acc = fma(T, x[i], acc)`. Rust's plain `*` and `+` don't contract,
//! so a literal port produces ~3.5e-7 relative drift even though the
//! cosine similarity stays at 1.0 — see the rope.rs module note about
//! the same FMA-vs-non-FMA gap.
//!
//! The fix: call `mul_add` at the two contraction points so the Rust
//! side matches clang's pattern bit-for-bit. Same arithmetic shape as
//! the embedding kernel (which is bit-exact today because its single
//! `nibble*scale + bias` step happens to land identically without
//! requiring the explicit FMA), now with the extra reduction made
//! deterministic across compilers.
//!
//! ## Layout reminder
//!
//! - `lm_head.weight`: `U32`, shape `[vocab_size, hidden_dim / 8]`,
//!   bits=4. Each `u32` packs 8 little-endian nibbles.
//! - `lm_head.scales`: `BF16`, shape `[vocab_size, hidden_dim / GROUP_SIZE]`.
//! - `lm_head.biases`: `BF16`, shape `[vocab_size, hidden_dim / GROUP_SIZE]`.

use crate::riir::embedding::bf16_to_f32;
use crate::riir::variants::{GROUP_SIZE, VARIANT};
use crate::riir::weight_file::WeightFile;

/// Errors specific to the LM head port.
#[derive(Debug, thiserror::Error)]
pub enum LmHeadError {
    #[error("LM head tensor '{name}' missing from manifest")]
    MissingTensor { name: &'static str },
    #[error(
        "LM head tensor '{name}' has unexpected shape {shape:?} (expected {expected:?})"
    )]
    ShapeMismatch {
        name: &'static str,
        shape: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error("input length {got} != HIDDEN_DIM {expected}")]
    InputLen { got: usize, expected: usize },
    #[error("output length {got} != VOCAB_SIZE {expected}")]
    OutputLen { got: usize, expected: usize },
}

const WEIGHT_NAME: &str = "lm_head.weight";
const SCALES_NAME: &str = "lm_head.scales";
const BIASES_NAME: &str = "lm_head.biases";

/// Compute the LM head logits for `x` (the post-final-norm hidden
/// state) and write `VOCAB_SIZE` floats into `out`. Sequential
/// reduction order matches `cpu_dequant_matvec`'s nested loops so the
/// partial sums see the same operands in the same order as the C path.
pub fn lm_head_cpu(
    wf: &WeightFile,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), LmHeadError> {
    let hidden_dim = VARIANT.hidden_dim;
    let vocab_size = VARIANT.vocab_size;

    if x.len() != hidden_dim {
        return Err(LmHeadError::InputLen {
            got: x.len(),
            expected: hidden_dim,
        });
    }
    if out.len() != vocab_size {
        return Err(LmHeadError::OutputLen {
            got: out.len(),
            expected: vocab_size,
        });
    }

    let num_groups = hidden_dim / GROUP_SIZE;
    let packed_cols = hidden_dim / 8;
    let packed_per_group = GROUP_SIZE / 8;

    let w_bytes = tensor_or_missing(wf, WEIGHT_NAME)?;
    let s_bytes = tensor_or_missing(wf, SCALES_NAME)?;
    let b_bytes = tensor_or_missing(wf, BIASES_NAME)?;

    expect_shape(wf, WEIGHT_NAME, &[vocab_size, packed_cols])?;
    expect_shape(wf, SCALES_NAME, &[vocab_size, num_groups])?;
    expect_shape(wf, BIASES_NAME, &[vocab_size, num_groups])?;

    for row in 0..vocab_size {
        let w_row_off = row * packed_cols * 4;
        let s_row_off = row * num_groups * 2;
        let b_row_off = row * num_groups * 2;
        let w_row = &w_bytes[w_row_off..w_row_off + packed_cols * 4];
        let s_row = &s_bytes[s_row_off..s_row_off + num_groups * 2];
        let b_row = &b_bytes[b_row_off..b_row_off + num_groups * 2];

        let mut acc: f32 = 0.0;
        for g in 0..num_groups {
            let scale = bf16_to_f32(read_u16_le(s_row, g));
            let bias = bf16_to_f32(read_u16_le(b_row, g));
            let base_x = g * GROUP_SIZE;
            for p in 0..packed_per_group {
                let packed = read_u32_le(w_row, g * packed_per_group + p);
                let x_base = base_x + p * 8;
                for n in 0..8 {
                    let val = (packed >> (n * 4)) & 0xF;
                    let t = (val as f32).mul_add(scale, bias);
                    acc = t.mul_add(x[x_base + n], acc);
                }
            }
        }
        out[row] = acc;
    }

    Ok(())
}

fn tensor_or_missing<'a>(
    wf: &'a WeightFile,
    name: &'static str,
) -> Result<&'a [u8], LmHeadError> {
    wf.tensor_bytes(name)
        .ok_or(LmHeadError::MissingTensor { name })
}

fn expect_shape(
    wf: &WeightFile,
    name: &'static str,
    expected: &[usize],
) -> Result<(), LmHeadError> {
    let info = wf
        .tensor_info(name)
        .ok_or(LmHeadError::MissingTensor { name })?;
    if info.shape != expected {
        return Err(LmHeadError::ShapeMismatch {
            name,
            shape: info.shape.clone(),
            expected: expected.to_vec(),
        });
    }
    Ok(())
}

fn read_u32_le(buf: &[u8], idx: usize) -> u32 {
    let off = idx * 4;
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn read_u16_le(buf: &[u8], idx: usize) -> u16 {
    let off = idx * 2;
    u16::from_le_bytes([buf[off], buf[off + 1]])
}
