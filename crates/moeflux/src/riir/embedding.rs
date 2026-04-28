//! Token-embedding lookup — Rust port of `infer.m`'s `embed_lookup`.
//!
//! The embedding tensor is 4-bit-packed (`U32`-encoded, 8 nibbles per
//! word) with per-group BF16 scales and biases. Lookup for a single
//! token reads exactly one `weight` row plus its scale/bias rows and
//! dequantizes into a `hidden_dim`-long `f32` vector.
//!
//! ## Bit-exact contract
//!
//! Every step is integer / bit-cast arithmetic with no rounding choices:
//! the `f32::from_bits((bf16 as u32) << 16)` decode mirrors the C
//! `bf16_to_f32`, and the `nibble * scale + bias` accumulation runs in
//! the same order the C loop does. The diff oracle asserts byte-equal
//! output against the C path.
//!
//! ## Layout reminder (matches `model_variant.h` / `extract_weights.py`)
//!
//! - `weight`:  `U32`, shape `[vocab_size, hidden_dim / 8]`, bits=4.
//!   Row stride is `hidden_dim / 8` u32 words; each word packs 8
//!   nibbles in little-endian nibble order (nibble 0 = bits 0..4).
//! - `scales`:  `BF16`, shape `[vocab_size, hidden_dim / GROUP_SIZE]`.
//! - `biases`:  `BF16`, shape `[vocab_size, hidden_dim / GROUP_SIZE]`.

use crate::riir::variants::{GROUP_SIZE, VARIANT};
use crate::riir::weight_file::WeightFile;

/// Errors specific to the embedding port.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("token id {token_id} out of vocabulary range (0..{vocab_size})")]
    TokenOutOfRange { token_id: i32, vocab_size: usize },
    #[error("embedding tensor '{name}' missing from manifest")]
    MissingTensor { name: &'static str },
    #[error(
        "embedding tensor '{name}' has unexpected shape {shape:?} (expected {expected:?})"
    )]
    ShapeMismatch {
        name: &'static str,
        shape: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error("output buffer length {got} does not match HIDDEN_DIM {expected}")]
    OutputLen { got: usize, expected: usize },
}

const WEIGHT_NAME: &str = "model.embed_tokens.weight";
const SCALES_NAME: &str = "model.embed_tokens.scales";
const BIASES_NAME: &str = "model.embed_tokens.biases";

/// Compute the embedding for `token_id` and write `HIDDEN_DIM` floats
/// into `out`. Returns an error if `token_id` is out of range, if any
/// of the three tensors is missing, or if `out.len() != HIDDEN_DIM`.
///
/// `out` is fully written on success; on error its contents are
/// unspecified.
pub fn embed_lookup(
    wf: &WeightFile,
    token_id: i32,
    out: &mut [f32],
) -> Result<(), EmbeddingError> {
    let hidden_dim = VARIANT.hidden_dim;
    let vocab_size = VARIANT.vocab_size;

    if out.len() != hidden_dim {
        return Err(EmbeddingError::OutputLen {
            got: out.len(),
            expected: hidden_dim,
        });
    }
    if token_id < 0 || (token_id as usize) >= vocab_size {
        return Err(EmbeddingError::TokenOutOfRange {
            token_id,
            vocab_size,
        });
    }

    let num_groups = hidden_dim / GROUP_SIZE;
    let packed_cols = hidden_dim / 8;
    let group_size = GROUP_SIZE;
    let packed_per_group = group_size / 8;

    let w_bytes = tensor_or_missing(wf, WEIGHT_NAME)?;
    let s_bytes = tensor_or_missing(wf, SCALES_NAME)?;
    let b_bytes = tensor_or_missing(wf, BIASES_NAME)?;

    expect_shape(wf, WEIGHT_NAME, &[vocab_size, packed_cols])?;
    expect_shape(wf, SCALES_NAME, &[vocab_size, num_groups])?;
    expect_shape(wf, BIASES_NAME, &[vocab_size, num_groups])?;

    let token = token_id as usize;
    let w_row_off = token * packed_cols * 4;
    let s_row_off = token * num_groups * 2;
    let b_row_off = token * num_groups * 2;
    let w_row = &w_bytes[w_row_off..w_row_off + packed_cols * 4];
    let s_row = &s_bytes[s_row_off..s_row_off + num_groups * 2];
    let b_row = &b_bytes[b_row_off..b_row_off + num_groups * 2];

    for g in 0..num_groups {
        let scale = bf16_to_f32(read_u16_le(s_row, g));
        let bias = bf16_to_f32(read_u16_le(b_row, g));
        for p in 0..packed_per_group {
            let packed = read_u32_le(w_row, g * packed_per_group + p);
            let base = g * group_size + p * 8;
            for n in 0..8 {
                let nibble = (packed >> (n * 4)) & 0xF;
                out[base + n] = (nibble as f32) * scale + bias;
            }
        }
    }

    Ok(())
}

fn tensor_or_missing<'a>(
    wf: &'a WeightFile,
    name: &'static str,
) -> Result<&'a [u8], EmbeddingError> {
    wf.tensor_bytes(name)
        .ok_or(EmbeddingError::MissingTensor { name })
}

fn expect_shape(
    wf: &WeightFile,
    name: &'static str,
    expected: &[usize],
) -> Result<(), EmbeddingError> {
    let info = wf
        .tensor_info(name)
        .ok_or(EmbeddingError::MissingTensor { name })?;
    if info.shape != expected {
        return Err(EmbeddingError::ShapeMismatch {
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

/// BF16 → F32 by zero-extending to the high half of an `f32`. Matches
/// the C `bf16_to_f32` byte-for-byte.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_trip_known_values() {
        // 0x3F80 = 1.0, 0xC000 = -2.0, 0x0000 = 0.0, 0x7F80 = +inf.
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC000), -2.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert!(bf16_to_f32(0x7F80).is_infinite());
    }
}
