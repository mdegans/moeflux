# Kernel arc — session 13 plan

Opens after session 12 landed SDPA Phase 1 (QK^T → steel `MMATile`,
commit `bfa1be5`, cosine 1.0). Three items, in order.

## 1. Bench Phase 1 (deferred measurement — do first)

Session 12 could not bench (Mike on mobile, no reboot). Per
`feedback_bench_discipline`: reboot, high-perf power, n≥3.
- `prefill_profile` with `MOEFLUX_PROFILE_PER_OP=1` at the 8192 shape —
  expect `batched_sdpa_causal_flash`'s share to drop from ~33%.
- 15.7k `bench.py`-style end-to-end A/B vs the pre-`bfa1be5` binary.
QK^T removed a real cross-lane `simd_sum`, so a genuine win is
expected; this just confirms + sizes it.

## 2. GQA-fold — the headline lever (NEW design work)

The session-12 P·V validation agent's read: GQA-fold is a *far larger*
structural win than P·V-on-tensors. Today one threadgroup serves one
`(q_tile, head)`; each of the `heads_per_kv` query-heads sharing a KV
head re-stages the *same* K/V blocks from global. GQA-fold = one
threadgroup per `(q_tile, kv_h)`, stage each K/V block **once**, inner-
loop over the `heads_per_kv` shared query-heads → K/V global traffic
and `kv_stage` writes cut by `heads_per_kv`×. A real bandwidth
reduction.

Scope (this touches Rust — needs `feedback_design_before_execute`:
design conversation → plan mode → execute):
- Encoder grid changes: `gpu_attn.rs` — grid becomes
  `num_q_tiles × num_kv_heads` instead of `× num_heads`.
- Kernel restructure (`sdpa.metal`): outer KV-block loop stages K/V
  once; inner loop over the `heads_per_kv` query-heads. `O` and the
  online-softmax running state (`row_m/l/corr`) are per-query-head —
  process heads sequentially so only one head's `O` is live, or widen
  the register state by `heads_per_kv` (check register budget; the
  P·V agent measured Phase-5 peak ≈196 floats/lane already).
- The `MMATile` Q/QK^T idiom from Phase 1 carries over per-head.
Gate: the 5 `sdpa_causal_flash_*` diff tests + canary + bench.

## 3. P·V → steel `MMATile` — optional uniformity follow-up

Validated spec below — executable in ~20 min, no further validation
needed. **It is a uniformity refactor, not a perf win** (P·V has no
reduction to delete; `FA_BC=16` contraction is too short for the
tensor unit — expect ~0% or slight regression). Do it only if the
codebase-uniformity value (consistent `MMATile` idiom for both
matmuls, a consistent base for GQA-fold) is judged worth it. Gate on
its own bench; revert if it regresses.

### Validated P·V spec (Plan agent, 2026-05-19)

Replaces Phase 5 (`sdpa.metal` ~Phase 5) and the finalize block.
`Frag`, `QTile` (= `MMATile<float,1,FA_KTILES>`), `co` already exist
from Phase 1.

**O accumulator.** Replace the `acc[FA_RPS][FA_VPL]` register array
with `QTile O;` declared before the KV loop (the `MMATile` default
ctor zero-inits `val_frags` — no `.clear()` needed; do NOT re-zero
inside the loop — it is the running accumulator).

**Phase 5, per KV block** (`O ← diag(corr)·O + P·V`):
1. corr rescale, element-wise — corr is row-constant (a lane's 32
   frags × 2 elems all belong to query row `simd_id*FA_RPS + co.y`):
   ```
   float rc = row_corr[simd_id*FA_RPS + co.y];
   for (uint j=0;j<FA_KTILES;++j) for (uint e=0;e<2;++e)
       O.frag_at(0,j)[e] *= rc;
   ```
2. load P (post-softmax scores), A-layout — same offset form as
   Phase 2's `C.store`:
   ```
   mlx::steel::MMATile<float,1,2> P;
   P.load<float,1,1,FA_BC,1>(
       scores + simd_id*FA_RPS*FA_BC + co.y*FA_BC + co.x);
   ```
3. stream V per ktile (k∈{0,1}, contraction FA_BC=16 = 2 ktiles).
   V is natural-layout (NO transpose: `str_x=FA_HD`=kv-row,
   `str_y=1`=head_dim). Stream — a full `MMATile<2,32>` V would push
   Phase-5 register peak to ~260 floats/lane and spill; streamed
   `Vk` keeps it ≈196 (`A`64 + `O`64 + `P`4 + `Vk`64):
   ```
   for (uint k=0;k<2;++k) {
       QTile Vk;
       Vk.load<float,1,1,FA_HD,1>(
           kv_stage + k*8*FA_HD + co.y*FA_HD + co.x);
       for (uint n=0;n<FA_KTILES;++n)
           Frag::mma(O.frag_at(0,n), P.frag_at(0,k),
                     Vk.frag_at(0,n), O.frag_at(0,n));
   }
   ```
No new barrier before Phase 5 (Phase 4's trailing barrier fences V;
Phase 3's fences `scores`/`row_corr`). Keep the existing trailing
barrier after Phase 5.

**Finalize** — `O` is fragment-layout, can't be scalar-written in the
old `lane+k*32` layout, and a device `MMATile::store` of all 8 rows
over-writes `out` rows past `n_tokens` on the ragged last tile
(`store_safe` predicates tile-frags, not per-lane rows → does NOT fix
it). Use a `kv_stage` round-trip, symmetric with Phase 1's Q staging:
```
// per-row 1/row_l prescale of O, element-wise (row-constant)
{
    uint trow = simd_id*FA_RPS + co.y;
    float denom = row_l[trow];
    float inv = (denom > 0.0f) ? (1.0f/denom) : 0.0f;
    for (uint j=0;j<FA_KTILES;++j) for (uint e=0;e<2;++e)
        O.frag_at(0,j)[e] *= inv;
}
// round-trip O -> kv_stage -> out, FA_BR/FA_BC = 4 rounds
for (uint r=0;r<FA_BR/FA_BC;++r) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id/2 == r) {
        uint local_row0 = (simd_id - 2*r)*FA_RPS;
        O.store<float,1,1,FA_HD,1>(
            kv_stage + local_row0*FA_HD + co.y*FA_HD + co.x);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx=lid; idx<FA_BC*FA_HD; idx+=FA_THREADS) {
        uint c = idx/FA_HD, d = idx%FA_HD;
        uint trow = r*FA_BC + c;
        if (trow < br_valid)
            out[((size_t)(q0+trow)*num_heads + h)*FA_HD + d]
                = kv_stage[idx];
    }
}
```
Both barriers required (producer→consumer fence on `kv_stage`, and
the next round's overwrite fence). Ragged guard `trow < br_valid`
lives on the scalar write.

Gate: 5 `sdpa_causal_flash_*` diff tests cosine ≥ 0.9999 + canary.
