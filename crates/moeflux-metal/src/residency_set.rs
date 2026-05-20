//! `MTLResidencySet` bindings — pin a fixed set of buffer allocations
//! to GPU-resident memory across command-buffer boundaries.
//!
//! The Metal driver may page out [`metal::Buffer`] allocations between
//! command submissions; on cold or memory-pressured runs that shows up
//! as run-to-run variance in our prefill bench (16% iter-1 → iter-2
//! jump on a3b). `MTLResidencySet` (macOS 15+) is the Apple-blessed
//! way to ask the driver to keep specific allocations resident.
//!
//! `metal-rs` 0.32 does not expose this API, so we go direct via
//! `objc::msg_send!`. The implementation mirrors llama.cpp's
//! `ggml_metal_buffer_rset_init` (`ggml-metal-device.m:1351-1386`):
//!
//! ```text
//! desc = [[MTLResidencySetDescriptor alloc] init]
//! desc.label = "moeflux"
//! desc.initialCapacity = N
//! rset = [device newResidencySetWithDescriptor:desc error:&err]
//! [desc release]
//! for buf in buffers: [rset addAllocation:buf]
//! [rset commit]
//! [rset requestResidency]
//! ```
//!
//! Teardown — in `Drop`:
//!
//! ```text
//! [rset endResidency]
//! [rset removeAllAllocations]
//! [rset release]
//! ```

use metal::foreign_types::ForeignTypeRef;
use metal::objc::rc::autoreleasepool;
use metal::objc::runtime::{Class, Object};
use metal::objc::{class, msg_send, sel, sel_impl};
use metal::{BufferRef, DeviceRef};

/// A pinned set of Metal buffer allocations.
///
/// Construct with [`ResidencySet::new`]; add buffers with
/// [`Self::add_allocation`]; finalise with [`Self::commit`] +
/// [`Self::request_residency`]. After `request_residency`, the Metal
/// driver keeps the registered allocations GPU-resident until `Drop`
/// (which calls `endResidency`).
///
/// Returns `None` from `new` on macOS < 15 (the underlying API is
/// unavailable). All other methods are no-ops on that platform because
/// the value can't have been constructed.
///
/// Manual ARC: the `*mut Object` is +1-retained on construction and
/// released in `Drop`. Not `Send` / `Sync` — drive from one thread.
pub struct ResidencySet {
    /// `id<MTLResidencySet>`, +1-retained.
    obj: *mut Object,
}

// `MTLResidencySet` is documented thread-safe for membership-management
// (`addAllocation:`, `commit`, `requestResidency`). In our use the set
// is built on one thread during `attach_to_device`, then read-only for
// the lifetime of inference — no method is called concurrently. The
// raw `*mut Object` would otherwise make any struct that holds a
// `ResidencySet` `!Send + !Sync`; explicitly opting in here keeps the
// caller side (e.g. `ExpertFiles` shared across rayon threads) clean.
unsafe impl Send for ResidencySet {}
unsafe impl Sync for ResidencySet {}

impl ResidencySet {
    /// Build a residency set on `device`. `label` is the
    /// `MTLResidencySetDescriptor.label` (debug tag in Instruments);
    /// `initial_capacity` is the descriptor's initial allocation
    /// capacity hint (the set still grows beyond this).
    ///
    /// Returns `None` on macOS < 15.
    pub fn new(
        device: &DeviceRef,
        label: &str,
        initial_capacity: u64,
    ) -> Option<Self> {
        if !is_available() {
            return None;
        }
        autoreleasepool(|| unsafe {
            // [[MTLResidencySetDescriptor alloc] init]
            let desc_cls = class!(MTLResidencySetDescriptor);
            let desc: *mut Object = msg_send![desc_cls, alloc];
            let desc: *mut Object = msg_send![desc, init];

            // desc.label = @"<label>"
            let ns_label = ns_string(label);
            let _: () = msg_send![desc, setLabel: ns_label];

            // desc.initialCapacity = initial_capacity
            let _: () =
                msg_send![desc, setInitialCapacity: initial_capacity];

            // rset = [device newResidencySetWithDescriptor:desc error:&err]
            let mut err: *mut Object = std::ptr::null_mut();
            let rset: *mut Object = msg_send![
                device.as_ptr(),
                newResidencySetWithDescriptor: desc
                error: &mut err
            ];

            let _: () = msg_send![desc, release];

            if rset.is_null() {
                if !err.is_null() {
                    let desc_str: *mut Object = msg_send![err, description];
                    let utf8: *const std::os::raw::c_char =
                        msg_send![desc_str, UTF8String];
                    if !utf8.is_null() {
                        let s = std::ffi::CStr::from_ptr(utf8)
                            .to_string_lossy();
                        eprintln!(
                            "[moeflux-metal] MTLResidencySet alloc \
                             failed: {s}"
                        );
                    }
                }
                return None;
            }

            Some(Self { obj: rset })
        })
    }

    /// `[rset addAllocation:buffer]`. May be called any time; the
    /// registration is staged until [`Self::commit`].
    pub fn add_allocation(&self, buffer: &BufferRef) {
        unsafe {
            let _: () =
                msg_send![self.obj, addAllocation: buffer.as_ptr()];
        }
    }

    /// `[rset commit]`. Apply the staged add/remove operations.
    pub fn commit(&self) {
        unsafe {
            let _: () = msg_send![self.obj, commit];
        }
    }

    /// `[rset requestResidency]`. Hint to the driver to bring the
    /// committed allocations resident and keep them so.
    pub fn request_residency(&self) {
        unsafe {
            let _: () = msg_send![self.obj, requestResidency];
        }
    }
}

impl Drop for ResidencySet {
    fn drop(&mut self) {
        // No-op if construction failed — we wouldn't have a value at
        // all. Mirror llama.cpp's teardown order: end → remove → release.
        unsafe {
            let _: () = msg_send![self.obj, endResidency];
            let _: () = msg_send![self.obj, removeAllAllocations];
            let _: () = msg_send![self.obj, release];
        }
    }
}

/// Runtime probe: is the `MTLResidencySetDescriptor` class linked?
///
/// `Class::get` resolves through `objc_getClass`, which returns nil for
/// classes absent from the linked framework. Cached once because the
/// answer is process-lifetime stable.
pub fn is_available() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        Class::get("MTLResidencySetDescriptor").is_some()
    })
}

/// Build an autoreleased `NSString` from a Rust `&str`.
///
/// The returned pointer is owned by the surrounding autorelease pool
/// — do not release it.
unsafe fn ns_string(s: &str) -> *mut Object {
    let cstr = std::ffi::CString::new(s)
        .expect("ns_string: label contained NUL byte");
    let ns_cls = class!(NSString);
    let obj: *mut Object =
        msg_send![ns_cls, stringWithUTF8String: cstr.as_ptr()];
    obj
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::{Device, MTLResourceOptions};

    #[test]
    fn residency_set_smoke() {
        // Skip on macOS < 15.
        if !is_available() {
            eprintln!(
                "MTLResidencySet not available on this OS — skipping"
            );
            return;
        }
        let device = Device::system_default()
            .expect("a Metal device for the residency-set smoke test");
        let rset = ResidencySet::new(&device, "moeflux-smoke", 1)
            .expect("newResidencySetWithDescriptor returned nil");
        let buf = device.new_buffer(
            4096,
            MTLResourceOptions::StorageModeShared,
        );
        rset.add_allocation(&buf);
        rset.commit();
        rset.request_residency();
        // Drop runs end → remove → release; if any of those crash the
        // test will fail.
    }
}
