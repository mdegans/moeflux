//! Smoke gate: the vendored MLX library assembles, compiles, and both
//! `affine_qmm_t` pipelines build (incl. the threadgroup-size check).

#![cfg(target_os = "macos")]

use metal::Device;
use moeflux_mlx::QmmKernels;

#[test]
fn qmm_kernels_build() {
    let device = Device::system_default().expect("no Metal device");
    QmmKernels::new(&device).expect("build moeflux-mlx kernels");
}
