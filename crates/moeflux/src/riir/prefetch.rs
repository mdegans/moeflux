//! Speculative-prefetch state machine for K-expert disk reads.
//!
//! Slice 5d-6b. Mirrors the C path's `g_async_pread` + `data_B`
//! prefetch (`metal_infer/infer.m:3240-3282, 4460-4470, 5510-5601`).
//!
//! # Concept
//!
//! Each layer's K active-expert disk reads are normally on the
//! critical path between MoE router (CPU) and CMD3 dispatch (GPU).
//! Slice 5d-6a parallelized them across 8 worker threads but kept
//! them synchronous. This slice (5d-6b) overlaps them with GPU
//! compute by prefetching the next layer's predicted experts into a
//! separate buffer set, asynchronously, so by the time CMD3 of layer
//! N completes and layer N+1 begins its expert dispatch, the experts
//! are already resident.
//!
//! Prediction = "the K indices the router selected for this same
//! layer in the previous token" — same as C
//! (`metal_infer/infer.m:5510-5562`). Token-to-token expert locality
//! is high in practice, so this hit rate is empirically good.
//!
//! # Soundness
//!
//! Async prefetch closures receive raw pointers into the
//! `MoeBuffers.data_prefetch[slot]` byte buffers and into
//! [`ExpertFiles`] via the [`DataPrefetchPtr`] / [`ExpertFilesPtr`]
//! newtypes (both with `unsafe impl Send`). Sound because:
//!
//! - **`MoeBuffers.data_prefetch[slot]`** is allocated once at
//!   `RsCtx` lazy-init and never reallocated. The pointer is valid
//!   for as long as `RsCtx` is alive.
//! - **`ExpertFiles`** is on `RsCtx` and exposes only `&self`
//!   methods; `pread64` is thread-safe per POSIX (per-call offset,
//!   no shared file position).
//! - **No aliasing of `data_prefetch[slot]`**: the K dispatched
//!   tasks each get a *disjoint* slot pointer (one per slot index);
//!   no GPU dispatch reads from `data_prefetch[slot]` until the
//!   prefetch has been [`PrefetchState::wait_for`]ed.
//! - **No use-after-free**: the drain points
//!   ([`PrefetchState::wait_for`], [`PrefetchState::invalidate_all`],
//!   [`PrefetchState::drain`], and [`PrefetchState`]'s `Drop` impl)
//!   complete all in-flight prefetches before the caller can mutate
//!   `MoeBuffers` or drop the `RsCtx`.
//!
//! Violating any of those requires misuse that the type signatures
//! don't enforce — same shape of contract as the deferred-experts
//! state machine in [`super::deferred`].

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver};

use super::expert_forward::MAX_K;
use super::expert_io::{ExpertFiles, ExpertIoError};

/// Per-slot decision: which buffer the K-expert encoder reads from.
/// Set by [`PrefetchState::wait_for`] (one entry per slot per layer)
/// and consumed by `gpu_batched_experts_*` in
/// [`super::expert_forward`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SlotSource {
    /// Slot was sync-pread into `MoeBuffers.data_synced[slot]` (miss
    /// or first-touch). Encoder binds `data_synced[slot]`.
    Synced,
    /// Slot was async-prefetched into `MoeBuffers.data_prefetch[slot]`
    /// and the prediction matched the actual routing for this token.
    /// Encoder binds `data_prefetch[slot]`.
    Prefetched,
}

/// State machine owned by `RsCtx`. Tracks per-layer predictions and
/// at most one in-flight async prefetch.
#[derive(Debug)]
pub struct PrefetchState {
    /// Per-layer last-token K indices, used as the prediction for
    /// the next token's same-layer prefetch. `None` until the layer
    /// has been run at least once (or until `invalidate_all` resets).
    last_token_indices: Vec<Option<[i32; MAX_K]>>,
    /// In-flight prefetch, if any.
    in_flight: Option<InFlight>,
    /// Slot-level hit counter (per-slot prediction matched the
    /// per-token routing). Accumulates across the lifetime of the
    /// state; reset via [`PrefetchState::reset_stats`] for per-request
    /// scoping. Atomic so the counters can be read from any thread,
    /// though the increment site is single-threaded (the orchestrator
    /// loop in `linear_attn_forward::post_attention_tail`).
    hits: AtomicU64,
    /// Slot-level miss counter (no in-flight prefetch, prefetch was
    /// for a different layer, prediction didn't match, or the slot
    /// index ran past the prefetch's K). Same lifecycle as `hits`.
    misses: AtomicU64,
}

#[derive(Debug)]
struct InFlight {
    /// Layer this prefetch was launched for.
    target_layer: usize,
    /// The K indices the prefetch loaded into
    /// `data_prefetch[0..k]`. [`PrefetchState::wait_for`] returns
    /// these so the caller can match per-slot against the actual
    /// routing for `target_layer`.
    loaded_indices: [i32; MAX_K],
    /// Number of K tasks dispatched (= K-active for the layer).
    k: usize,
    /// Per-task completion channel.
    rx: Receiver<Result<(), ExpertIoError>>,
}

/// Result of [`PrefetchState::wait_for`] when a prefetch was both
/// in-flight and targeted at the requested layer.
#[derive(Copy, Clone, Debug)]
pub struct PrefetchStatus {
    /// Indices the prefetch loaded — caller matches per-slot vs
    /// actual routing for the layer.
    pub loaded_indices: [i32; MAX_K],
    /// Number of valid entries in `loaded_indices` (= K-active).
    pub k: usize,
}

/// `Send`-able handle to one `data_prefetch[slot]` byte buffer. Stored
/// as a `usize` (the pointer's address) so the closure-capture
/// auto-trait inference doesn't see a `*mut u8`/`NonNull<u8>` that
/// `!Send` propagates from. Reconstituted to `*mut u8` only inside
/// the worker closure under the `unsafe` block.
#[derive(Copy, Clone, Debug)]
struct DataPrefetchPtr {
    ptr_addr: usize,
    len: usize,
}

impl DataPrefetchPtr {
    fn from_slice(s: &mut [u8]) -> Self {
        debug_assert!(!s.is_empty(), "data_prefetch slot must be non-empty");
        Self {
            ptr_addr: s.as_mut_ptr() as usize,
            len: s.len(),
        }
    }

    /// SAFETY: caller upholds the drain-before-touch invariant from
    /// the module docs.
    unsafe fn as_mut_slice<'a>(self) -> &'a mut [u8] {
        // SAFETY: forwarded — caller's invariant covers this.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr_addr as *mut u8,
                self.len,
            )
        }
    }
}

/// `Send + Sync` handle to the `ExpertFiles` instance. Same `usize`
/// trick as [`DataPrefetchPtr`] so the closure's auto-trait
/// derivation doesn't see a raw pointer.
#[derive(Copy, Clone, Debug)]
struct ExpertFilesPtr {
    addr: usize,
}

impl ExpertFilesPtr {
    fn from_ref(r: &ExpertFiles) -> Self {
        Self {
            addr: (r as *const ExpertFiles) as usize,
        }
    }

    /// SAFETY: caller upholds the drain-before-touch invariant from
    /// the module docs (specifically, the referent outlives all
    /// in-flight readers).
    unsafe fn as_ref<'a>(self) -> &'a ExpertFiles {
        // SAFETY: forwarded — caller's invariant covers this.
        unsafe { &*(self.addr as *const ExpertFiles) }
    }
}

impl PrefetchState {
    /// Create a fresh state with `num_layers` slots, all unprimed.
    pub fn new(num_layers: usize) -> Self {
        Self {
            last_token_indices: vec![None; num_layers],
            in_flight: None,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Record a per-layer outcome: how many of the K slots were
    /// satisfied by a prefetch hit, and how many fell back to a sync
    /// pread. Called from the orchestrator at the resolution site
    /// (`linear_attn_forward::post_attention_tail`).
    pub fn record_outcome(&self, hits: u64, misses: u64) {
        if hits > 0 {
            self.hits.fetch_add(hits, Ordering::Relaxed);
        }
        if misses > 0 {
            self.misses.fetch_add(misses, Ordering::Relaxed);
        }
    }

    /// Read the accumulated `(hits, misses)` counters.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }

    /// Zero the counters (e.g. for per-request scoping).
    pub fn reset_stats(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Drain any in-flight prefetch (wait for all K tasks). Used by
    /// `memory_clear`, `state_save`, `state_load`, and the `Drop`
    /// impl. Per-task errors are dropped — caller of `drain` (as
    /// distinct from `wait_for`) shouldn't be reading
    /// `data_prefetch[slot]` afterward.
    pub fn drain(&mut self) {
        if let Some(in_flight) = self.in_flight.take() {
            for _ in 0..in_flight.k {
                let _ = in_flight.rx.recv();
            }
        }
    }

    /// Drain any in-flight prefetch AND clear all per-layer
    /// predictions. Used by `memory_clear` so the next token starts
    /// from cold-prediction state (no stale predictions from before
    /// the clear).
    pub fn invalidate_all(&mut self) {
        self.drain();
        for slot in self.last_token_indices.iter_mut() {
            *slot = None;
        }
    }

    /// Wait for any in-flight prefetch targeted at `layer_idx`,
    /// returning the loaded indices if (a) the target matches, and
    /// (b) all K tasks completed without error. Returns `None` if
    /// no prefetch was in flight, the target was a different layer
    /// (stale prefetch), or any task errored. In all cases, the
    /// in-flight slot is drained and consumed.
    pub fn wait_for(&mut self, layer_idx: usize) -> Option<PrefetchStatus> {
        let in_flight = self.in_flight.take()?;
        let mut all_ok = true;
        for _ in 0..in_flight.k {
            match in_flight.rx.recv() {
                Ok(Ok(())) => {}
                _ => all_ok = false,
            }
        }
        if !all_ok || in_flight.target_layer != layer_idx {
            None
        } else {
            Some(PrefetchStatus {
                loaded_indices: in_flight.loaded_indices,
                k: in_flight.k,
            })
        }
    }

    /// The prediction for `layer_idx`, if one exists. `None` means
    /// no prediction yet (first token, or post-`invalidate_all`).
    pub fn predict_for(&self, layer_idx: usize) -> Option<[i32; MAX_K]> {
        self.last_token_indices.get(layer_idx).copied().flatten()
    }

    /// Record this token's actual routing for `layer_idx`, becoming
    /// the prediction for the next token's same layer.
    pub fn record_actual(
        &mut self,
        layer_idx: usize,
        actual: [i32; MAX_K],
    ) {
        if let Some(slot) = self.last_token_indices.get_mut(layer_idx) {
            *slot = Some(actual);
        }
    }

    /// Fire-and-forget: kick off async prefetch of `predicted[0..k]`
    /// into the K caller-supplied `data_prefetch` slot pointers via
    /// `pool`. Stores the task receiver in `self.in_flight`. Caller
    /// must drain (via `wait_for` / `drain` / `invalidate_all`)
    /// before any subsequent mutation of `data_prefetch[0..k]`,
    /// before any GPU read of `data_prefetch[slot]`, and before
    /// dropping the `RsCtx`.
    ///
    /// `target_layer` is the layer the prefetch is FOR (not the
    /// layer currently running). `k` is the number of valid
    /// entries in `predicted` and the count of slot pointers used
    /// from `data_prefetch`.
    ///
    /// If a prior prefetch is still in-flight, drains it first.
    /// Correct callers won't hit that path; the drain is defensive.
    pub fn dispatch(
        &mut self,
        target_layer: usize,
        predicted: [i32; MAX_K],
        k: usize,
        data_prefetch: [&mut [u8]; MAX_K],
        pool: &rayon::ThreadPool,
        expert_files: &ExpertFiles,
    ) {
        self.drain();
        let (tx, rx) = sync_channel::<Result<(), ExpertIoError>>(MAX_K);
        let efp = ExpertFilesPtr::from_ref(expert_files);
        // Capture the K disjoint slot pointers BEFORE spawning;
        // the borrow checker's `data_prefetch: [&mut [u8]; MAX_K]`
        // input is consumed by this conversion.
        let mut slot_ptrs: [Option<DataPrefetchPtr>; MAX_K] =
            std::array::from_fn(|_| None);
        for (i, dst) in data_prefetch.into_iter().enumerate() {
            slot_ptrs[i] = Some(DataPrefetchPtr::from_slice(dst));
        }
        for slot in 0..k {
            let expert_idx = predicted[slot] as usize;
            let dst_ptr = slot_ptrs[slot].expect("slot 0..k populated");
            let tx = tx.clone();
            // SAFETY: see module-level docs. `dst_ptr` and `efp` are
            // valid for the duration of the in-flight prefetch; the
            // drain-before-touch discipline guarantees no aliasing.
            pool.spawn(move || {
                let dst = unsafe { dst_ptr.as_mut_slice() };
                let efs = unsafe { efp.as_ref() };
                let r = efs.read_expert(target_layer, expert_idx, dst);
                let _ = tx.send(r);
            });
        }
        self.in_flight = Some(InFlight {
            target_layer,
            loaded_indices: predicted,
            k,
            rx,
        });
    }
}

impl Drop for PrefetchState {
    fn drop(&mut self) {
        // Final drain — soundness contract requires no in-flight
        // tasks holding pointers when the underlying `MoeBuffers` /
        // `ExpertFiles` are dropped. `RsCtx`'s field drop order is
        // declaration order; `prefetch` field is declared after
        // `moe_buffers` and `experts` so this drop runs first.
        self.drain();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_for_returns_none_until_recorded() {
        let mut st = PrefetchState::new(8);
        assert_eq!(st.predict_for(3), None);
        let actual = [0i32; MAX_K];
        st.record_actual(3, actual);
        assert_eq!(st.predict_for(3), Some(actual));
        st.invalidate_all();
        assert_eq!(st.predict_for(3), None);
    }

    #[test]
    fn predict_for_out_of_range_layer_is_none() {
        let st = PrefetchState::new(2);
        assert_eq!(st.predict_for(99), None);
    }
}
