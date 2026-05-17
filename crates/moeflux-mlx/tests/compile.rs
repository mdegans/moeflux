//! P2 gate: the vendored MLX library assembles, compiles, and its
//! `affine_qmm_t` pipelines can be created.

#![cfg(target_os = "macos")]

use metal::Device;
use moeflux_mlx::QmmLibrary;

#[test]
fn qmm_library_compiles_and_pipelines_fetch() {
    let device = Device::system_default().expect("no Metal device");
    let lib = QmmLibrary::new(&device).expect("compile moeflux-mlx library");
    for aligned in [true, false] {
        lib.qmm_t_pipeline(&device, aligned).unwrap_or_else(|e| {
            panic!("fetch affine_qmm_t (aligned_n={aligned}): {e}");
        });
    }
}
