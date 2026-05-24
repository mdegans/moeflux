---
name: expert-io-gate-landed
description: 2026-05-24 — restored pread+prefetch behind a runtime gate covering both prefill and decode (mmap default; pread when expert working set > 0.75 × physical RAM). Fixes a17b at both phases without regressing a3b.
metadata:
  type: project
---

# What

Expert IO is now runtime-selected, not hard-coded. Two paths,
applied identically to prefill and decode:

- **Mmap** — the 2026-05-20 tearout path. The per-layer mmap is
  wrapped as a Metal buffer and pinned via `MTLResidencySet`; the
  GPU reads expert weights directly. Zero staging copy, zero
  main-thread CPU on the per-token critical path.
- **Pread** — the speculative-prefetch + parallel-`pread` state
  machine the tearout removed, plus a `num_experts × expert_size`
  staging buffer for the prefill gather. The mmap'd layer files
  are still opened (we use `file.read_at` against them) but never
  exposed to the GPU.

Selection lives in [`io/expert_io_mode.rs`](../../crates/moeflux/src/riir/io/expert_io_mode.rs):
`auto` (default) picks Pread when
`expert_size × num_experts × num_layers > 0.75 × physical_ram`,
otherwise Mmap. `MOEFLUX_EXPERT_IO=mmap|pread|auto` overrides.
The decision is logged at `RsCtx::open` and constant for the
session.

# Why

- a3b's 24 GB working set fits in M2 Max's 96 GB and mmap-direct
  is +51 % decode (11.35 → 17.16 tok/s — see
  [[completed-decode-gen-arc]]).
- a17b's ~245 GB working set blows past page-cache capacity and
  the `MTLResidencySet` pin either fails or thrashes. The GPU
  stalls on VM demand-faults. Mike noticed the prefill regression
  this session; decode was already known to be broken.

# Wiring

Single field `expert_io_mode: ExpertIoMode` on `RsCtx`, set once
at `open()` via `io::expert_io_mode::select()`. Consumers:

- `ExpertFiles::attach_to_device(pool, mode)` — early-return in
  Pread mode. The mmap is still built (cheap virtual mapping), but
  no Metal buffer wrap and no residency pin.
- `MoeGraphScratch::new(pool, k_active, mode)` — allocates the
  `expert_base: Option<BufId<ExpertBaseBuf>>` staging buffer only
  in Pread mode (mmap mode addresses the per-layer mmap directly).
- `moe_block_forward` (prefill) — `Some(base_id)` arm reads each
  bucketed expert from disk and uploads it to
  `expert_id * expert_size`; `None` arm calls
  `expert_files.mmap_id_for_expert(layer, 0)`.
- `PrefetchState::new(num_layers, mode)` — carries the mode for
  decode-side consumers.
- `step_internal_per_token_oracle` — skips `prefetch.dispatch`
  fire in Mmap mode (the OS page cache covers what the prefetch
  would pump; serial pread on top is wasted CPU).
- `moe_dispatch_per_token` (decode) — Pread arm runs the
  wait-for / set-match / parallel-pread sequence and binds per-slot
  `(data_synced[slot])` (miss) or `(data_prefetch[set][buf_idx])`
  (hit), offset 0; Mmap arm binds the layer mmap at
  `(expert_idx * expert_size)` offset.

Both paths fan in through the same
`gpu_batched_experts_begin_mmap` (now misnamed — it's the
generic-bindings path).

# Sysinfo dep

Added `sysinfo = "0.32"` with `default-features = false, features
= ["system"]` to the workspace. Used in exactly one spot
(`physical_ram_bytes()` at `open` time). Well-known crate per
Mike's preference; libc::sysctl was the alternative.

# Expected behavior

- **a3b** — gate picks Mmap (~24 GB ≪ 0.75 × 96 GB). Preserves
  17.16 tok/s decode and the prefill mmap-direct path. No code
  difference from 2026-05-21.
- **a17b** — gate picks Pread (~245 GB ≫ threshold). Restores
  the pre-tearout prefill staging path and decode prefetch state
  machine. Should restore working tok/s at both phases.

# Untested

Tok/s benchmarks. Builds for all three model features clean;
81 lib tests pass. Mike to run profile.py / bench.py for both
variants.

# Cross-references

- [[pread-teardown-landed]] — the tearout this reverts (gated).
- drama_llama `project_mmap_vs_pread_tradeoff.md` — problem
  statement.
- [[completed-decode-gen-arc]] — the arc that landed the a3b
  mmap-direct win.
