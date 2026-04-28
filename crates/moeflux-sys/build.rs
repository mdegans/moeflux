//! Build script for moeflux-sys.
//!
//! Compiles the C/Objective-C sources from `../../metal_infer`, generates
//! Rust bindings against `moeflux.h`, and links the macOS frameworks the
//! Metal pipeline needs. macOS-only; on other platforms build.rs is a
//! no-op so `cargo check` at least succeeds in CI.

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

    // Exactly one model variant must be selected by the consuming crate.
    let model_defines: Vec<(&str, &str)> = [
        ("CARGO_FEATURE_MODEL_QWEN3_5_A17B",    "MOEFLUX_MODEL_QWEN3_5_A17B"),
        ("CARGO_FEATURE_MODEL_QWEN3_6_35B_A3B", "MOEFLUX_MODEL_QWEN3_6_35B_A3B"),
    ]
    .into_iter()
    .filter(|(feat, _)| env::var(feat).is_ok())
    .collect();
    let model_define = match model_defines.len() {
        0 => panic!(
            "moeflux-sys: no model variant feature enabled. Enable exactly one of: \
             `model-qwen3-5-a17b`, `model-qwen3-6-35b-a3b`."
        ),
        1 => model_defines[0].1,
        _ => panic!(
            "moeflux-sys: multiple model variants enabled ({model_defines:?}). \
             Exactly one must be selected."
        ),
    };

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let metal_infer = manifest.join("../../metal_infer");
    let infer_m = metal_infer.join("infer.m");
    let moeflux_h = metal_infer.join("moeflux.h");

    // Bake the shader-source absolute path into the crate so consumers
    // don't have to manage cwd. metal_setup() picks this up via the
    // MOEFLUX_SHADERS_PATH env var at runtime; the safe wrapper sets
    // that env var from this const on Ctx::open. Phase 6 moved the
    // source-of-truth shader into the safe-wrapper crate; the C-side
    // oracle now reads from there.
    let shaders_src = manifest.join("../moeflux/shaders/shaders.metal");
    let shaders_abs = shaders_src.canonicalize()
        .unwrap_or_else(|_| shaders_src.clone());
    println!("cargo:rustc-env=MOEFLUX_COMPILED_SHADERS_PATH={}",
             shaders_abs.display());
    println!("cargo:rerun-if-changed={}", shaders_src.display());

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
