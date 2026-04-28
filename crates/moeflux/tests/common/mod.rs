//! Helper modules shared across integration tests.
//!
//! Cargo treats `tests/<name>.rs` files as separate test binaries, but
//! files under `tests/common/` are picked up only when an integration
//! test does `mod common;`. This is the canonical place for shared
//! test scaffolding.

pub mod c_backend;
