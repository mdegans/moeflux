//! FFI link sanity. Confirms the `mf_*` symbols actually resolve at
//! runtime and that `mf_init_model` honors its "null inputs → null
//! output" contract without crashing. Does not load a real model.

#![cfg(target_os = "macos")]

use std::ptr;

use moeflux_sys::{mf_free_model, mf_init_model};

#[test]
fn null_inputs_return_null() {
    // Per moeflux.h: any NULL path argument must yield NULL return.
    // If bindgen missed a symbol or the C ABI drifted, we'd either
    // fail to link (compile-time) or segfault here (runtime).
    let ctx = unsafe {
        mf_init_model(
            ptr::null(), ptr::null(), ptr::null(), ptr::null(),
            /* experts_per_tok */ 4,
            /* use_2bit */ 0,
        )
    };
    assert!(ctx.is_null());

    // Freeing a null ctx must also be a no-op per the contract.
    unsafe { mf_free_model(ptr::null_mut()) };
}
