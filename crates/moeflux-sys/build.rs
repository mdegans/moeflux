//! Build script for moeflux-sys.
//!
//! Compiles the C/Objective-C sources from `metal_infer/`, generates
//! Rust bindings against `moeflux.h`, and links the macOS frameworks
//! the Metal pipeline needs. macOS-only; on other platforms build.rs
//! is a no-op so `cargo check` at least succeeds in CI.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=build.rs");

    // moeflux is Metal-only today. Non-macOS targets get no code —
    // the safe wrapper crate above us will `#[cfg]` away its impls.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target_os != "macos" {
        println!("cargo:warning=moeflux-sys: target is {target_os}, not macos — skipping build");
        return;
    }

    // Exactly one C-supported model variant must be selected by the
    // consuming crate when the C oracle is wanted. Variants without a
    // C counterpart (e.g. `model-cogito-v2-671b`) skip the C build:
    // we still compile this crate as a no-op so the diff-oracle test
    // target compiles, but lib.rs's `include!()` reads an empty
    // bindings.rs and the c_backend wrapper is gated to only the
    // C-supported variants.
    let model_defines: Vec<(&str, &str)> = [
        ("CARGO_FEATURE_MODEL_QWEN3_5_A17B",    "MOEFLUX_MODEL_QWEN3_5_A17B"),
        ("CARGO_FEATURE_MODEL_QWEN3_6_35B_A3B", "MOEFLUX_MODEL_QWEN3_6_35B_A3B"),
    ]
    .into_iter()
    .filter(|(feat, _)| env::var(feat).is_ok())
    .collect();
    let model_define = match model_defines.len() {
        0 => {
            println!(
                "cargo:warning=moeflux-sys: no C-side model variant enabled \
                 (only Rust-side variants like `model-cogito-v2-671b`); \
                 skipping C build. Empty bindings.rs will be emitted."
            );
            // Write an empty bindings.rs so lib.rs's `include!()` succeeds.
            let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
            std::fs::write(out_dir.join("bindings.rs"), b"")
                .expect("writing empty bindings.rs");
            return;
        }
        1 => model_defines[0].1,
        _ => panic!(
            "moeflux-sys: multiple C-side model variants enabled ({model_defines:?}). \
             Exactly one must be selected."
        ),
    };

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let metal_infer = manifest.join("metal_infer");
    let infer_m = metal_infer.join("infer.m");
    let moeflux_h = metal_infer.join("moeflux.h");

    // Shader source-of-truth lives in the moeflux crate (Phase 6).
    // moeflux-sys does NOT bake a shader path — consumers of this
    // crate (the diff oracle suite via the moeflux crate's `imp`
    // module) own shader-path discovery. This keeps moeflux-sys
    // self-contained for `cargo publish`.

    for p in [&infer_m, &moeflux_h] {
        println!("cargo:rerun-if-changed={}", p.display());
    }
    // model_variant.h is #included from infer.m, track it too.
    println!(
        "cargo:rerun-if-changed={}",
        metal_infer.join("model_variant.h").display()
    );

    // Compile libmoeflux.a from infer.m. Matches the Makefile's `lib`
    // target: -DMOEFLUX_LIB excludes main(), -fobjc-arc enables ARC,
    // the model define selects the shape.
    cc::Build::new()
        .file(&infer_m)
        .include(&metal_infer)
        .define("MOEFLUX_LIB", None)
        .define(model_define, None)
        .define("ACCELERATE_NEW_LAPACK", None)
        .flag("-fobjc-arc")
        .flag("-Wno-unused-function")
        .flag("-Wno-unused-variable")
        .flag("-Wno-unused-parameter")
        .compile("moeflux");

    // Frameworks + system libs the Metal pipeline needs.
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=compression");

    // Bindings. We only need the mf_* public API from moeflux.h, not
    // the whole internal universe, so allowlist by function prefix.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", metal_infer.display()))
        .allowlist_function("mf_.*")
        .allowlist_type("mf_.*")
        .allowlist_var("MF_.*")
        // C bool should map to Rust bool — and size_t to usize, etc.
        .size_t_is_usize(true)
        .generate_comments(true)
        .layout_tests(false)
        .generate()
        .expect("bindgen failed");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("writing bindings.rs");
}
