//! CPU 4-bit dequant + matvec for the Cogito-V2 / DeepSeek-V3 path.
//!
//! moeflux's existing matvec / dequant kernels live GPU-side in
//! [`super::gpu_matvec`]. The first-run CPU MLA path needs an
//! equivalent on-host primitive for the MLA projections (q_a, q_b,
//! kv_a_with_mqa, kv_b, o_proj) and the dense / MoE FFN matrices.
//! Performance is intentionally not load-bearing — the goal is
//! coherent first-token output, not interactive throughput. GPU MLA
//! is a follow-up slice.
//!
//! ## Layout assumption
//!
//! Mirrors [`super::embedding`] and the MLX 4-bit safetensors
//! convention emitted by `tools/convert_cogito_v2.py`. For a
//! conceptual `[out_dim, in_dim]` matrix:
//!
//! - **`weight`**: `U32`, shape `[out_dim, in_dim / 8]`. Each `u32`
//!   packs 8 nibbles in little-endian nibble order (nibble `n` lives
//!   in bits `n*4 .. n*4+4`).
//! - **`scales`**: `BF16`, shape `[out_dim, in_dim / GROUP_SIZE]`.
//!   One scale per group of [`GROUP_SIZE`] consecutive in-features.
//! - **`biases`**: `BF16`, same shape as `scales`.
//!
//! Dequant of the value at `(r, c)` is
//! `nibble * scale[r, c / GROUP_SIZE] + bias[r, c / GROUP_SIZE]`.
//!
//! The matvec computes `out[r] = sum_c W[r, c] * x[c]` for `r in
//! 0..out_dim`. Output rows are independent and run in parallel via
//! rayon.

use rayon::prelude::*;

use crate::riir::embedding::bf16_to_f32;
use crate::riir::variants::GROUP_SIZE;
use crate::riir::weight_file::WeightFile;

/// Errors from the CPU matvec primitive.
#[derive(Debug, thiserror::Error)]
pub enum CpuMatvecError {
    /// `in_dim` must be a multiple of [`GROUP_SIZE`] so the group
    /// boundaries align with the BF16 scale / bias arrays.
    #[error("in_dim {in_dim} is not a multiple of GROUP_SIZE={group_size}")]
    InDimNotMultiple { in_dim: usize, group_size: usize },
    /// Sliced lengths don't match the declared shape.
    #[error(
        "slice length mismatch on '{field}': got {got} elements, expected {expected}"
    )]
    SliceLen {
        field: &'static str,
        got: usize,
        expected: usize,
    },
    /// One of the three tensors `<name>.{weight,scales,biases}` was
    /// not present in the manifest.
    #[error("missing tensor '{name}'")]
    MissingTensor { name: String },
    /// Tensor shape from the manifest doesn't match `[out_dim,
    /// in_dim_packed]`.
    #[error(
        "tensor '{name}' has unexpected shape {shape:?} (expected {expected:?})"
    )]
    ShapeMismatch {
        name: String,
        shape: Vec<usize>,
        expected: Vec<usize>,
    },
    /// Tensor byte buffer wasn't aligned to its dtype (`u32` for
    /// weights, `u16` for scales / biases). Indicates a converter
    /// bug or an unaligned mmap; the upstream converter pads to
    /// 8-byte boundaries so this should never fire in practice.
    #[error("tensor '{name}' not aligned to {align}-byte boundary")]
    Misaligned { name: String, align: usize },
}

/// Compute `out[r] = sum_c dequant(W[r, c]) * x[c]` for `r in
/// 0..out_dim`. Rows are processed in parallel via rayon; each row
/// performs a single-pass fused dequant + dot, no materialized
/// dequant matrix.
///
/// `in_dim` must be a multiple of [`GROUP_SIZE`]; `packed` /
/// `scales` / `biases` must have lengths consistent with `(in_dim,
/// out_dim)`. See module docs for the layout convention.
pub fn dequant_matvec_4bit_cpu(
    packed: &[u32],
    scales: &[u16],
    biases: &[u16],
    in_dim: usize,
    out_dim: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), CpuMatvecError> {
    if in_dim % GROUP_SIZE != 0 {
        return Err(CpuMatvecError::InDimNotMultiple {
            in_dim,
            group_size: GROUP_SIZE,
        });
    }
    let in_packed = in_dim / 8;
    let in_groups = in_dim / GROUP_SIZE;
    let packed_per_group = GROUP_SIZE / 8;

    let expect_packed = out_dim * in_packed;
    let expect_scales = out_dim * in_groups;
    if packed.len() != expect_packed {
        return Err(CpuMatvecError::SliceLen {
            field: "packed",
            got: packed.len(),
            expected: expect_packed,
        });
    }
    if scales.len() != expect_scales {
        return Err(CpuMatvecError::SliceLen {
            field: "scales",
            got: scales.len(),
            expected: expect_scales,
        });
    }
    if biases.len() != expect_scales {
        return Err(CpuMatvecError::SliceLen {
            field: "biases",
            got: biases.len(),
            expected: expect_scales,
        });
    }
    if x.len() != in_dim {
        return Err(CpuMatvecError::SliceLen {
            field: "x",
            got: x.len(),
            expected: in_dim,
        });
    }
    if out.len() != out_dim {
        return Err(CpuMatvecError::SliceLen {
            field: "out",
            got: out.len(),
            expected: out_dim,
        });
    }

    out.par_iter_mut().enumerate().for_each(|(r, out_r)| {
        let packed_row = &packed[r * in_packed..(r + 1) * in_packed];
        let scale_row = &scales[r * in_groups..(r + 1) * in_groups];
        let bias_row = &biases[r * in_groups..(r + 1) * in_groups];
        let mut acc = 0.0f32;
        for g in 0..in_groups {
            let scale = bf16_to_f32(scale_row[g]);
            let bias = bf16_to_f32(bias_row[g]);
            let group_packed = &packed_row
                [g * packed_per_group..(g + 1) * packed_per_group];
            let x_group = &x[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
            for p in 0..packed_per_group {
                let word = group_packed[p];
                let x_chunk = &x_group[p * 8..p * 8 + 8];
                for n in 0..8 {
                    let nibble = ((word >> (n * 4)) & 0xF) as f32;
                    let w = nibble.mul_add(scale, bias);
                    acc = w.mul_add(x_chunk[n], acc);
                }
            }
        }
        *out_r = acc;
    });
    Ok(())
}

/// Bytes-level variant of [`dequant_matvec_4bit_cpu`] used by the
/// per-layer expert path: the packed expert blob is read via
/// `pread` into a single `Vec<u8>` and we want to slice the `gate`
/// / `up` / `down` weight + scales + biases sub-blocks out of it
/// without copying.
///
/// Each `*_bytes` slice is reinterpreted as the appropriate dtype
/// (`u32` for `weight_bytes`, `u16` for `scales_bytes` /
/// `biases_bytes`) using `align_to`. The packed expert layout the
/// converter emits is naturally aligned, so the head/tail must be
/// empty — anything else is a converter bug we want loud.
pub fn dequant_matvec_4bit_bytes_cpu(
    weight_bytes: &[u8],
    scales_bytes: &[u8],
    biases_bytes: &[u8],
    in_dim: usize,
    out_dim: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), CpuMatvecError> {
    let packed = bytes_as_u32("packed", weight_bytes)?;
    let scales = bytes_as_u16("scales", scales_bytes)?;
    let biases = bytes_as_u16("biases", biases_bytes)?;
    dequant_matvec_4bit_cpu(packed, scales, biases, in_dim, out_dim, x, out)
}

/// `out[r] = sum_c W[r, c] * x[c]` for an unquantized BF16 matrix
/// stored as `[out_dim, in_dim]` row-major. Used for the
/// noaux_tc router gate (`mlp.gate.weight` is BF16, not 4-bit
/// packed) and any other small unquantized projection.
///
/// Single-pass per row, parallelized via rayon. The BF16 → F32
/// decode happens inline in the dot product.
pub fn bf16_matvec_cpu(
    weight: &[u16],
    in_dim: usize,
    out_dim: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), CpuMatvecError> {
    let expect_w = in_dim * out_dim;
    if weight.len() != expect_w {
        return Err(CpuMatvecError::SliceLen {
            field: "weight",
            got: weight.len(),
            expected: expect_w,
        });
    }
    if x.len() != in_dim {
        return Err(CpuMatvecError::SliceLen {
            field: "x",
            got: x.len(),
            expected: in_dim,
        });
    }
    if out.len() != out_dim {
        return Err(CpuMatvecError::SliceLen {
            field: "out",
            got: out.len(),
            expected: out_dim,
        });
    }
    out.par_iter_mut().enumerate().for_each(|(r, out_r)| {
        let row = &weight[r * in_dim..(r + 1) * in_dim];
        let mut acc = 0.0f32;
        for (c, &w_bits) in row.iter().enumerate() {
            let w = bf16_to_f32(w_bits);
            acc = w.mul_add(x[c], acc);
        }
        *out_r = acc;
    });
    Ok(())
}

/// Convenience wrapper around [`bf16_matvec_cpu`] that reads
/// `<name>` (a single tensor name, no `.weight` / `.scales`
/// suffix) from the [`WeightFile`]. Confirms shape `[out_dim,
/// in_dim]` and BF16 dtype.
pub fn project_bf16_cpu(
    wf: &WeightFile,
    name: &str,
    in_dim: usize,
    out_dim: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), CpuMatvecError> {
    let bytes = wf
        .tensor_bytes(name)
        .ok_or_else(|| CpuMatvecError::MissingTensor {
            name: name.to_string(),
        })?;
    expect_shape(wf, name, &[out_dim, in_dim])?;
    let weight = bytes_as_u16(name, bytes)?;
    bf16_matvec_cpu(weight, in_dim, out_dim, x, out)
}

/// Convenience: read `{name}.weight`, `{name}.scales`,
/// `{name}.biases` from the [`WeightFile`] and run the matvec.
/// Caller passes `(in_dim, out_dim)` rather than reading from the
/// manifest so we get a tight typed assertion at the call site
/// (the MLA / MLP paths know their layer shapes from
/// [`super::variants::VARIANT`]).
pub fn project_4bit_cpu(
    wf: &WeightFile,
    name: &str,
    in_dim: usize,
    out_dim: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), CpuMatvecError> {
    let w_name = format!("{name}.weight");
    let s_name = format!("{name}.scales");
    let b_name = format!("{name}.biases");

    let w_bytes = wf
        .tensor_bytes(&w_name)
        .ok_or_else(|| CpuMatvecError::MissingTensor {
            name: w_name.clone(),
        })?;
    let s_bytes = wf
        .tensor_bytes(&s_name)
        .ok_or_else(|| CpuMatvecError::MissingTensor {
            name: s_name.clone(),
        })?;
    let b_bytes = wf
        .tensor_bytes(&b_name)
        .ok_or_else(|| CpuMatvecError::MissingTensor {
            name: b_name.clone(),
        })?;

    let in_packed = in_dim / 8;
    let in_groups = in_dim / GROUP_SIZE;
    let expected_w = vec![out_dim, in_packed];
    let expected_s = vec![out_dim, in_groups];
    expect_shape(wf, &w_name, &expected_w)?;
    expect_shape(wf, &s_name, &expected_s)?;
    expect_shape(wf, &b_name, &expected_s)?;

    let packed = bytes_as_u32(&w_name, w_bytes)?;
    let scales = bytes_as_u16(&s_name, s_bytes)?;
    let biases = bytes_as_u16(&b_name, b_bytes)?;
    dequant_matvec_4bit_cpu(packed, scales, biases, in_dim, out_dim, x, out)
}

fn expect_shape(
    wf: &WeightFile,
    name: &str,
    expected: &[usize],
) -> Result<(), CpuMatvecError> {
    let info = wf
        .tensor_info(name)
        .ok_or_else(|| CpuMatvecError::MissingTensor {
            name: name.to_string(),
        })?;
    if info.shape != expected {
        return Err(CpuMatvecError::ShapeMismatch {
            name: name.to_string(),
            shape: info.shape.clone(),
            expected: expected.to_vec(),
        });
    }
    Ok(())
}

fn bytes_as_u32<'a>(
    name: &str,
    bytes: &'a [u8],
) -> Result<&'a [u32], CpuMatvecError> {
    // SAFETY: `align_to` is safe by definition. We additionally
    // assert no unaligned head/tail so the body covers the full
    // tensor; the converter pads to 8 bytes so this never fires
    // in practice but if it ever does it's a real bug we want loud.
    let (head, body, tail) = unsafe { bytes.align_to::<u32>() };
    if !head.is_empty() || !tail.is_empty() {
        return Err(CpuMatvecError::Misaligned {
            name: name.to_string(),
            align: 4,
        });
    }
    Ok(body)
}

fn bytes_as_u16<'a>(
    name: &str,
    bytes: &'a [u8],
) -> Result<&'a [u16], CpuMatvecError> {
    let (head, body, tail) = unsafe { bytes.align_to::<u16>() };
    if !head.is_empty() || !tail.is_empty() {
        return Err(CpuMatvecError::Misaligned {
            name: name.to_string(),
            align: 2,
        });
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-constructed 1×64 matrix: every nibble is 1, scale=1.0,
    /// bias=0.0. Dequant → all-ones row; matvec with x = [1, 1, ..., 1]
    /// returns 64.0.
    #[test]
    fn dequant_matvec_all_ones() {
        let in_dim = 64;
        let out_dim = 1;
        // 8 u32 words = 64 nibbles, each = 1 (0x11111111).
        let packed = vec![0x1111_1111u32; 8];
        // scale = 1.0 (BF16 0x3F80), bias = 0.0 (BF16 0x0000).
        let scales = vec![0x3F80u16; 1];
        let biases = vec![0x0000u16; 1];
        let x = vec![1.0f32; in_dim];
        let mut out = vec![0.0f32; out_dim];
        dequant_matvec_4bit_cpu(
            &packed, &scales, &biases, in_dim, out_dim, &x, &mut out,
        )
        .unwrap();
        // 64 ones, dot with 64 ones = 64.
        assert_eq!(out[0], 64.0);
    }

    /// Two output rows, one zero-input column-vector → output rows
    /// must both be 0.0. Exercises the row stride and the zero-x
    /// fast case.
    #[test]
    fn zero_input_yields_zero_output() {
        let in_dim = 64;
        let out_dim = 2;
        let packed = vec![0xFFFF_FFFFu32; 8 * out_dim];
        let scales = vec![0x3F80u16; out_dim];
        let biases = vec![0x3F80u16; out_dim];
        let x = vec![0.0f32; in_dim];
        let mut out = vec![999.0f32; out_dim];
        dequant_matvec_4bit_cpu(
            &packed, &scales, &biases, in_dim, out_dim, &x, &mut out,
        )
        .unwrap();
        // Even with bias=1.0, x=0 → every term is 0, sum=0.
        assert_eq!(out, vec![0.0; out_dim]);
    }

    /// nibble=0 with bias=2.0, scale=irrelevant → dequant value is
    /// 2.0 in every cell. Dot with x=[1.0; 64] returns 128.0 (= 2 ×
    /// 64). Confirms the bias path and the FMA accumulation order.
    #[test]
    fn bias_only_path() {
        let in_dim = 64;
        let out_dim = 1;
        let packed = vec![0u32; 8];
        // scale = 5.0 (BF16 0x40A0), bias = 2.0 (BF16 0x4000).
        let scales = vec![0x40A0u16; 1];
        let biases = vec![0x4000u16; 1];
        let x = vec![1.0f32; in_dim];
        let mut out = vec![0.0f32; out_dim];
        dequant_matvec_4bit_cpu(
            &packed, &scales, &biases, in_dim, out_dim, &x, &mut out,
        )
        .unwrap();
        // nibble=0 → dequant = 0 * 5 + 2 = 2; sum = 2 × 64 = 128.
        assert_eq!(out[0], 128.0);
    }

    /// Slice-length mismatch must surface as a typed error, not a
    /// panic.
    #[test]
    fn slice_length_mismatch_errors() {
        let in_dim = 64;
        let out_dim = 1;
        let packed = vec![0u32; 7]; // wrong length (need 8)
        let scales = vec![0x3F80u16; 1];
        let biases = vec![0x0000u16; 1];
        let x = vec![1.0f32; in_dim];
        let mut out = vec![0.0f32; out_dim];
        let err = dequant_matvec_4bit_cpu(
            &packed, &scales, &biases, in_dim, out_dim, &x, &mut out,
        )
        .unwrap_err();
        match err {
            CpuMatvecError::SliceLen { field, got, expected } => {
                assert_eq!(field, "packed");
                assert_eq!(got, 7);
                assert_eq!(expected, 8);
            }
            _ => panic!("wrong error variant: {err:?}"),
        }
    }

    /// BF16 matvec sanity: identity matrix × x = x.
    #[test]
    fn bf16_matvec_identity() {
        let in_dim = 4;
        let out_dim = 4;
        // 4×4 identity matrix in BF16. 1.0 = 0x3F80, 0.0 = 0x0000.
        let mut weight = vec![0x0000u16; in_dim * out_dim];
        for r in 0..out_dim {
            weight[r * in_dim + r] = 0x3F80;
        }
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; out_dim];
        bf16_matvec_cpu(&weight, in_dim, out_dim, &x, &mut out).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// Smoke test against the real Cogito-V2 weights: load layer 0
    /// `q_a_proj`, run the matvec on a hand-built input, check the
    /// output is finite (no NaN/Inf) and within a sane magnitude
    /// range. Catches a wrong-orientation layout mistake before the
    /// MLA kernel + first-forward bisect.
    ///
    /// Requires the converted weights at the path below; ignored by
    /// default. Run with `cargo test -- --ignored
    /// q_a_proj_smoke_against_real_weights`.
    #[cfg(feature = "model-cogito-v2-671b")]
    #[test]
    #[ignore = "needs Cogito-V2 weights mmap'd from /Volumes/Temp Backup"]
    fn q_a_proj_smoke_against_real_weights() {
        use std::path::Path;
        let bin = Path::new(
            "/Volumes/Temp Backup/models/blallama/cogito-v2-671b/artifacts/model_weights.bin",
        );
        let manifest = Path::new(
            "/Volumes/Temp Backup/models/blallama/cogito-v2-671b/artifacts/model_weights.json",
        );
        let wf = WeightFile::open(bin, manifest).expect("open weights");
        // Hidden = 7168 → q_lora_rank = 1536 (Cogito-V2 dims).
        let in_dim = 7168;
        let out_dim = 1536;
        // x = a sparse pulse: x[3] = 1.0, rest = 0.
        let mut x = vec![0.0f32; in_dim];
        x[3] = 1.0;
        let mut out = vec![0.0f32; out_dim];
        project_4bit_cpu(
            &wf,
            "model.layers.0.self_attn.q_a_proj",
            in_dim,
            out_dim,
            &x,
            &mut out,
        )
        .unwrap();
        // Every output must be finite. Any NaN/Inf points to a
        // layout / dequant bug.
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite values in q_a_proj output: \
             first nonfinite at index {:?}",
            out.iter().position(|v| !v.is_finite()),
        );
        // The output corresponds to column 3 of the (dequantized)
        // weight matrix. For a 4-bit quantization with sane scales,
        // weight magnitudes are O(1); column 3 of a [1536, 7168]
        // matrix should produce a [1536]-vector with magnitude in
        // single-digit range, not 0 or 1e10.
        let max_abs =
            out.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(
            max_abs > 1e-4 && max_abs < 1e3,
            "q_a_proj column-3 magnitude {max_abs} outside sane range — \
             possible wrong layout or wrong tensor",
        );
    }

    /// `in_dim` not divisible by `GROUP_SIZE` must surface a typed
    /// error.
    #[test]
    fn in_dim_alignment_check() {
        let in_dim = 65; // not multiple of 64
        let out_dim = 1;
        let packed = vec![0u32; 8];
        let scales = vec![0x3F80u16; 1];
        let biases = vec![0x0000u16; 1];
        let x = vec![1.0f32; in_dim];
        let mut out = vec![0.0f32; out_dim];
        let err = dequant_matvec_4bit_cpu(
            &packed, &scales, &biases, in_dim, out_dim, &x, &mut out,
        )
        .unwrap_err();
        assert!(matches!(err, CpuMatvecError::InDimNotMultiple { .. }));
    }
}
