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
//! The original C-via-Objective-C implementation is preserved behind
//! the `diff-oracle` cargo feature for regression testing. Production
//! consumers don't enable it; moeflux's `tests/diff_oracle.rs` does.

#![cfg_attr(not(target_os = "macos"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

/// Pure-Rust port of the host-side dispatch.
#[cfg(target_os = "macos")]
pub mod riir;

/// C-via-`moeflux-sys` implementation. Available behind the
/// `diff-oracle` cargo feature for differential regression testing.
/// Production consumers should ignore this module.
#[cfg(all(target_os = "macos", feature = "diff-oracle"))]
pub mod imp;

/// Default backend re-export. The Rust port is the only path; the
/// `diff-oracle` feature exposes the C path under [`mod@imp`] but
/// does not change the [`Ctx`] / [`Error`] aliases.
#[cfg(target_os = "macos")]
pub use riir::{RsCtx as Ctx, RsError as Error};
