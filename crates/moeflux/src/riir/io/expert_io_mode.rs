//! Expert-IO mode selection — covers both prefill and decode.
//!
//! On 2026-05-20 we tore out the `pread` expert-IO path across both
//! the prefill batched gather (see [`super::expert_io`] and
//! `pread_teardown_landed.md`) and the per-token decode dispatch
//! (see `completed_decode_gen_arc.md`), moving everything to direct
//! GPU reads of the mmap'd expert buffers. That was load-bearing for
//! Qwen3-A3B (+51 % decode on M2 Max) but turned out to break
//! Qwen3-A17B at both phases: its expert working set is many times
//! physical RAM, so the OS can't keep it page-resident and
//! `MTLResidencySet` either fails or thrashes. The GPU then stalls
//! on demand-fault VM activity.
//!
//! The fix is a single runtime gate consulted by every site that
//! used to inspect `ExpertIoMode`:
//!
//! - **`ExpertFiles::attach_to_device`** — skip the
//!   `newBufferWithBytesNoCopy` + residency-set pin in `Pread` mode.
//!   The on-disk mmap stays for reading via `read_at`; the layer
//!   files are not exposed to the GPU at all.
//! - **`MoeGraphScratch::expert_base`** — `Some` in `Pread` mode
//!   (a `num_experts * expert_size` staging buffer the prefill
//!   gather kernel reads from); `None` in `Mmap` mode.
//! - **`moe_block_forward`** (prefill) — `Pread` arm reads each
//!   bucketed expert from disk and uploads it into `expert_base`
//!   at `expert_id * expert_size`; `Mmap` arm points the kernel at
//!   the layer mmap buffer.
//! - **`moe_dispatch_per_token`** (decode) — `Pread` arm runs the
//!   speculative-prefetch / sync-pread state machine from
//!   [`super::prefetch`]; `Mmap` arm binds the layer mmap buffer.
//! - **`step_internal_per_token_oracle`** — skips the
//!   `prefetch.dispatch` fire in `Mmap` mode.
//!
//! The gate: `total_expert_bytes > 0.75 * physical_ram`, with
//! `MOEFLUX_EXPERT_IO=mmap|pread|auto` as an override for
//! benchmarking. `auto` (the default) runs the gate.

use crate::riir::variants::{VARIANT, Variant};

/// Which path moves expert bytes onto the GPU. Picked once at
/// [`crate::riir::RsCtx::open`] and threaded into the prefetch state
/// machine, the MoE graph scratch, and `ExpertFiles`. The choice is
/// constant for the session.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExpertIoMode {
    /// GPU reads expert weights directly from per-layer mmap'd Metal
    /// buffers. Zero staging copy, zero CPU on the per-token critical
    /// path. Suitable when the expert working set comfortably fits in
    /// physical RAM (`MTLResidencySet` can pin it).
    Mmap,
    /// CPU stages expert weights into scratch via `pread` (prefill:
    /// one `num_experts * expert_size` staging buffer per layer
    /// dispatch; decode: K-slot `data_synced` ring with prefetch
    /// overlap). Suitable when the working set exceeds page-cache
    /// capacity and mmap-direct stalls the GPU on VM activity.
    Pread,
}

impl ExpertIoMode {
    pub fn is_mmap(self) -> bool {
        matches!(self, ExpertIoMode::Mmap)
    }
    pub fn is_pread(self) -> bool {
        matches!(self, ExpertIoMode::Pread)
    }
}

/// Total bytes of expert weights across every layer at the active
/// variant's 4-bit layout. This is the working set
/// `ExpertFiles::attach_to_device` would mmap into Metal-resident
/// buffers when every layer file is present, and is what the gate
/// compares against physical RAM.
pub fn total_expert_bytes(v: Variant) -> u64 {
    (v.expert_size_4bit() as u64) * (v.num_experts as u64) * (v.num_layers as u64)
}

/// Total physical RAM in bytes via the [`sysinfo`] crate. Sampled
/// once at `open()` — the value is constant for the lifetime of the
/// session.
fn physical_ram_bytes() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.total_memory()
}

/// Pick the expert-IO mode for this run. Reads `MOEFLUX_EXPERT_IO`
/// if set (`mmap`, `pread`, or `auto`); otherwise (or on `auto`)
/// applies the `expert_bytes > 0.75 * physical_ram` gate. Logs the
/// decision once on stderr.
pub fn select() -> ExpertIoMode {
    let v = VARIANT;
    let expert_bytes = total_expert_bytes(v);
    let ram_bytes = physical_ram_bytes();
    let threshold = ram_bytes / 4 * 3; // 0.75 * ram (integer-safe)

    let env = std::env::var("MOEFLUX_EXPERT_IO").ok();
    let forced = match env.as_deref() {
        Some("mmap") => Some(ExpertIoMode::Mmap),
        Some("pread") => Some(ExpertIoMode::Pread),
        Some("auto") | None => None,
        Some(other) => {
            eprintln!(
                "[expert_io] MOEFLUX_EXPERT_IO={other:?} unrecognised; \
                 expected mmap|pread|auto. Falling back to auto."
            );
            None
        }
    };

    let auto = if expert_bytes > threshold {
        ExpertIoMode::Pread
    } else {
        ExpertIoMode::Mmap
    };
    let mode = forced.unwrap_or(auto);

    let gib = |b: u64| (b as f64) / (1024.0 * 1024.0 * 1024.0);
    let forced_str = match (forced, env.as_deref()) {
        (Some(_), Some(s)) => format!(" (forced via MOEFLUX_EXPERT_IO={s})"),
        _ => String::new(),
    };
    eprintln!(
        "[expert_io] expert_bytes={:.2} GiB, physical_ram={:.2} GiB, \
         threshold={:.2} GiB → {:?}{}",
        gib(expert_bytes),
        gib(ram_bytes),
        gib(threshold),
        mode,
        forced_str,
    );
    mode
}
