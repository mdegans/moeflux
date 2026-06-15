//! Pure-Rust [moeflux](https://github.com/mdegans/moeflux) — streaming-experts
//! MoE inference on Metal.
//!
//! Forked from [flash-moe](https://github.com/danveloper/flash-moe);
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
//! moeflux is a pure-Rust port; the original C/Objective-C reference
//! implementation has been retired. The port is validated by
//! Rust-internal differential oracles — `tests/graph_diff_oracle.rs`
//! (the CPU backend vs. the Metal backend, per `Op`) and
//! `tests/batched_diff_oracle.rs` (batched vs. per-token paths), with
//! `tests/diff_oracle.rs` carrying the remaining Rust-internal checks.

#![cfg_attr(not(target_os = "macos"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

/// Pure-Rust port of the host-side dispatch.
#[cfg(target_os = "macos")]
pub mod riir;

/// Default backend re-export. The Rust port is the only path.
#[cfg(target_os = "macos")]
pub use riir::{CheckpointError, DEFAULT_MAX_CHECKPOINTS, RsCtx as Ctx, RsError as Error};
