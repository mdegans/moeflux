//! Pure-Rust [moeflux](https://github.com/mdegans/moeflux) — streaming-experts
//! MoE inference on Metal.
//!
//! Forked from [flash-moe](https://github.com/SuperEpic/flash-moe);
//! the Metal kernels at `crates/moeflux/shaders/shaders.metal` retain
//! flash-moe's attribution. Everything else is original Rust.
//!
//! ## Platform
//!
//! macOS only. On non-macOS targets this crate exposes no symbols.
//!
//! ## Model variant
//!
//! Exactly one Cargo feature must select the compile-time model
//! shape:
//!
//! - `model-qwen3-5-a17b`
//! - `model-qwen3-6-35b-a3b`
//!
//! See `docs/model_variants.md` for the full shape table.
//!
//! ## Concurrency
//!
//! Single-threaded per [`Ctx`]. The Metal command queue is owned per
//! [`Ctx`], so two `Ctx`s can drive the GPU concurrently from
//! different threads (the Council reactor uses this for parallel
//! agent deliberation). [`Ctx`] is [`Send`] but not [`Sync`].
//!
//! ## Differential oracle
//!
//! The original C-via-Objective-C implementation is wrapped as a
//! test-only helper at `tests/common/c_backend.rs`. moeflux's lib
//! ships only the Rust port; the C oracle exists solely to validate
//! the port via `tests/diff_oracle.rs`.

#![cfg_attr(not(target_os = "macos"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

/// Pure-Rust port of the host-side dispatch.
#[cfg(target_os = "macos")]
pub mod riir;

/// Default backend re-export. The Rust port is the only path.
#[cfg(target_os = "macos")]
pub use riir::{
    CheckpointError, RsCtx as Ctx, RsError as Error, DEFAULT_MAX_CHECKPOINTS,
};
