//! Safe Rust API for [moeflux](https://github.com/mdegans/moeflux).
//!
//! Wraps the `mf_*` C API from `moeflux-sys` with RAII, typed
//! errors, and lifetime-checked logit slices.
//!
//! ## Platform
//!
//! macOS only. On non-macOS targets this crate exposes no symbols.
//!
//! ## Model variant
//!
//! Exactly one Cargo feature must select the compile-time model
//! shape. The variant flows through to `moeflux-sys`:
//!
//! - `model-qwen3-5-a17b`
//! - `model-qwen3-6-35b-a3b`
//!
//! See `docs/model_variants.md` in the moeflux repo for the
//! full shape table.
//!
//! ## Concurrency
//!
//! Per moeflux.h: single-threaded per `Ctx`, and the backend
//! initializes a process-global Metal context on first `Ctx::open`.
//! `Ctx` is `Send` (you may move it across threads) but not `Sync`
//! (concurrent calls on the same instance are undefined).

#![cfg_attr(not(target_os = "macos"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

/// C-via-`moeflux-sys` implementation. Direct access kept public so
/// the differential test harness can pin-import [`imp::Ctx`] without
/// caring which backend the top-level [`Ctx`] alias resolves to.
#[cfg(target_os = "macos")]
pub mod imp;

/// Pure-Rust port of the host-side dispatch. Available when the
/// `riir-port` Cargo feature is enabled. Runs alongside the C path so
/// the differential test harness can compare them.
#[cfg(all(target_os = "macos", feature = "riir-port"))]
pub mod riir;

/// Default backend re-export: when `riir-port` is on, `Ctx`/`Error`
/// resolve to the Rust port (`riir::RsCtx`/`riir::RsError`); otherwise
/// to the C wrapper (`imp::Ctx`/`imp::Error`). The Phase 6 cutover
/// drops this alias and lifts the riir types to the crate root
/// directly.
#[cfg(all(target_os = "macos", not(feature = "riir-port")))]
pub use imp::{Ctx, Error};

#[cfg(all(target_os = "macos", feature = "riir-port"))]
pub use riir::{RsCtx as Ctx, RsError as Error};
