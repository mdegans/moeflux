//! Weight-file loader — Rust port of `infer.m`'s `WeightFile` +
//! `TensorManifest`.
//!
//! `model_weights.bin` is mmap'd; `model_weights.json` is parsed
//! into a per-tensor offset/shape/dtype index. Tensors are looked up
//! by name, returning a borrowed slice into the mmap. The slice's
//! lifetime is tied to [`WeightFile`]'s — drop the file and your
//! tensor views go away.
//!
//! ## Schema
//!
//! ```json
//! {
//!   "tensors": {
//!     "model.layers.0.self_attn.q_proj.weight": {
//!       "offset": 0,
//!       "size": 8388608,
//!       "shape": [4096, 4096],
//!       "dtype": "U32",
//!       "bits": 4
//!     },
//!     ...
//!   }
//! }
//! ```
//!
//! `bits` is only emitted for `U32`-packed quantized tensors by
//! `extract_weights.py >= 2026-04-26`. We default it to 4 for older
//! manifests so A17B / pre-bits extractions keep working — matches
//! the C path's behavior.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde::Deserialize;

/// Errors from the weight-file loader.
#[derive(Debug, thiserror::Error)]
pub enum WeightFileError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("manifest JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("manifest missing 'tensors' key")]
    MissingTensors,
    #[error(
        "tensor '{name}' span [{offset}, {end}) extends past mmap'd file ({file_size} bytes)"
    )]
    OutOfBounds {
        name: String,
        offset: u64,
        end: u64,
        file_size: u64,
    },
}

/// One tensor's metadata. Name lives in the index's [`HashMap`] key;
/// the index returns `(&str, &TensorInfo)` pairs.
///
/// `dtype` mirrors the C `dtype[8]` field — typical values are
/// `"U32"` (4-bit or 8-bit packed weights), `"BF16"` (scales /
/// biases), and `"F32"`. Use [`TensorInfo::dtype`] as a string-match
/// site rather than parsing into an enum: the C side does the same
/// and new dtypes can show up via the conversion script without a
/// crate-side change.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub offset: u64,
    pub size: u64,
    pub shape: Vec<usize>,
    pub dtype: String,
    /// Quantization bits for `U32`-packed tensors (typically 4 or
    /// 8). 0 means "not quantized" / non-`U32` dtype.
    pub bits: i32,
}

/// mmap'd weight blob plus tensor index. Construct via
/// [`WeightFile::open`]; tensor data via [`WeightFile::tensor_bytes`]
/// (slice view into the mmap, lives as long as `&self`).
pub struct WeightFile {
    mmap: Mmap,
    tensors: HashMap<String, TensorInfo>,
}

impl WeightFile {
    /// Open a weight blob and its sibling manifest. The two paths
    /// are typically `<artifacts>/model_weights.bin` and
    /// `<artifacts>/model_weights.json` (matching the C `Ctx::open`
    /// argument order).
    pub fn open(
        bin_path: &Path,
        manifest_path: &Path,
    ) -> Result<Self, WeightFileError> {
        let file = File::open(bin_path)?;
        // SAFETY: mmap of a regular file we just opened. The file
        // descriptor is closed when `file` drops (after Mmap takes
        // ownership of the mapping). On Linux/macOS a `MAP_PRIVATE`
        // mmap stays valid until munmap regardless of fd state.
        let mmap = unsafe { Mmap::map(&file)? };
        let tensors = parse_manifest(manifest_path)?;
        let wf = WeightFile { mmap, tensors };
        wf.bounds_check_all()?;
        eprintln!(
            "[weights] mmap'd {:.2} GB from {}",
            wf.mmap.len() as f64 / 1e9,
            bin_path.display()
        );
        eprintln!(
            "[manifest] Loaded {} tensors from {}",
            wf.tensors.len(),
            manifest_path.display()
        );
        Ok(wf)
    }

    /// Total bytes in the mmap'd file.
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Number of tensors in the manifest.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Tensor info by name, or `None` if missing. Cheap O(1) hash
    /// lookup.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Borrowed byte slice of tensor `name`'s data, or `None` if the
    /// tensor isn't in the manifest. The slice is exactly
    /// `info.size` bytes long, starting at `info.offset` in the
    /// mmap.
    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensor_info(name)?;
        let start = info.offset as usize;
        let end = start + info.size as usize;
        Some(&self.mmap[start..end])
    }

    /// Iterator over `(name, info)` pairs. Order is HashMap-defined
    /// (i.e., unspecified). Useful for debugging / dumping.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TensorInfo)> {
        self.tensors.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Verify every tensor's `[offset, offset + size)` span fits
    /// inside the mmap. Run once at load time so later
    /// [`Self::tensor_bytes`] calls can skip per-call bounds checks
    /// — a typical forward pass walks ~1400 tensors per token, the
    /// per-call branch matters in the hot loop.
    fn bounds_check_all(&self) -> Result<(), WeightFileError> {
        let file_size = self.mmap.len() as u64;
        for (name, info) in &self.tensors {
            let end = info.offset.saturating_add(info.size);
            if end > file_size {
                return Err(WeightFileError::OutOfBounds {
                    name: name.clone(),
                    offset: info.offset,
                    end,
                    file_size,
                });
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for WeightFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightFile")
            .field("file_size", &self.mmap.len())
            .field("num_tensors", &self.tensors.len())
            .finish()
    }
}

// --- Manifest parser ---------------------------------------------

#[derive(Debug, Deserialize)]
struct RawManifest {
    tensors: HashMap<String, RawTensor>,
}

/// Mirrors the per-tensor shape `extract_weights.py` emits. `bits`
/// is optional (older manifests don't carry it) — see module docs.
#[derive(Debug, Deserialize)]
struct RawTensor {
    offset: u64,
    size: u64,
    shape: Vec<usize>,
    dtype: String,
    #[serde(default)]
    bits: Option<i32>,
}

fn parse_manifest(
    path: &Path,
) -> Result<HashMap<String, TensorInfo>, WeightFileError> {
    let bytes = std::fs::read(path)?;
    let raw: RawManifest = serde_json::from_slice(&bytes)?;
    let mut out = HashMap::with_capacity(raw.tensors.len());
    for (name, t) in raw.tensors {
        // Default bits=4 for U32-packed tensors that lack the
        // explicit field (older manifests). Non-U32 tensors are
        // unquantized — bits=0.
        let bits = t.bits.unwrap_or_else(|| {
            if t.dtype == "U32" {
                4
            } else {
                0
            }
        });
        out.insert(
            name,
            TensorInfo {
                offset: t.offset,
                size: t.size,
                shape: t.shape,
                dtype: t.dtype,
                bits,
            },
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Manifest parser handles the documented schema, including
    /// older manifests without `bits`. Doesn't need a real model.
    #[test]
    fn parse_manifest_with_and_without_bits() {
        let json = br#"{
            "tensors": {
                "weights_a": {
                    "offset": 0,
                    "size": 1024,
                    "shape": [16, 16],
                    "dtype": "U32",
                    "bits": 4
                },
                "weights_b_old": {
                    "offset": 1024,
                    "size": 2048,
                    "shape": [32, 16],
                    "dtype": "U32"
                },
                "scales": {
                    "offset": 3072,
                    "size": 64,
                    "shape": [32],
                    "dtype": "BF16"
                }
            }
        }"#;
        let raw: RawManifest = serde_json::from_slice(json).unwrap();
        assert_eq!(raw.tensors.len(), 3);
        let a = &raw.tensors["weights_a"];
        assert_eq!(a.bits, Some(4));
        assert_eq!(a.dtype, "U32");
        let b = &raw.tensors["weights_b_old"];
        assert_eq!(b.bits, None);
        assert_eq!(b.dtype, "U32");
    }
}
