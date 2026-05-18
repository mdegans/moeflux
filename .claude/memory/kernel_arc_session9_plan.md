# Kernel arc — session 9 plan: simdgroup_matrix rewrite of `gated_delta_net_chunkwise`

Continues `kernel_arc_session8_landed.md`. Plan reviewed via plan mode
and approved by Mike 2026-05-18.

## Why

The chunkwise DeltaNet kernel is **17.5% of a3b prefill — the #1
single-op pole** — but performance-neutral vs the old per-token kernel,
because every matmul inside it is a hand-rolled scalar dot-product
loop. moeflux uses `simdgroup_matrix` (hardware cooperative matmul)
**nowhere**. This session makes the kernel actually exploit the
parallelism the chunkwise form exposes. ("The MLX lever" — but the
right tool here is the raw Metal builtin, see below.)

## Approach (decisions locked — do not re-litigate)

1. **Raw Metal `simdgroup_matrix<float,8,8>` builtin.** NOT MLX
   `mma.h` (forces one element type — no f16-in/f32-acc), NOT steel
   GEMM (built for large matrices). MLX's own small kernels (Winograd
   conv, `mlx/backend/metal/kernels/conv.metal:378`) use raw
   simdgroup_matrix — this IS the MLX-appropriate approach for these
   shapes, and needs no new vendoring. `#include
   <metal_simdgroup_matrix>` at the top of `shaders.metal` (default
   `CompileOptions` is fine — Metal 3.x, builtin available).
2. **Single kernel, incremental.** Keep `gated_delta_net_chunkwise`
   as ONE kernel. Rewrite ONE phase at a time; the other phases stay
   scalar, so the diff oracle isolates each change. The Op, both
   backend arms, the producer, and the CPU oracle `gated_delta_chunkwise`
   (`linear_attn.rs:474`) are ALL unchanged — `shaders.metal` is the
   only file edited.
3. **f32 for v1.** All `simdgroup_matrix<float,8,8>`. f16 is a marked
   v2 follow-up.
4. 128 threads/threadgroup = **4 simdgroups**. Matmul phases run
   cooperatively across the 4; per-`vi` phases (1, 4) and the
   elementwise epilogues are unchanged. Barriers + threadgroup memory
   hand off between the two thread interpretations. Add
   `simdgroup_index_in_threadgroup` + `thread_index_in_simdgroup`
   kernel attributes.

## The kernel

`shaders.metal:1768-1913`, `kernel void gated_delta_net_chunkwise`.
Dispatch: 64 threadgroups (1/v-head) × 128 threads (`vi`). `CW_C 16`
= chunk length C. head_dim = key_dim = value_dim = D = 128. Six phases
inside a `for chunk_start` loop; current threadgroup memory 18,560 B
of 32,768.

The four matmul-bearing phases as GEMMs (per head, per chunk; C=16, D=128):

| Phase | lines | GEMM | output |
|---|---|---|---|
| 2 | 1820-1848 | `kc·kcᵀ`→`kk[C,C]`, `kc·qᵀ`→`kq[C,C]`; then ×β/γ elementwise + triangular mask | 16×16 |
| 3 | 1850-1863 | `S0[D,D]·kcᵀ[D,C]` → `s0k[D,C]` | 128×16 |
| 5 | 1875-1889 | `S0·qᵀ`→`s0q[D,C]`; and `kqg[C,C]·U[C,D]`→`[C,D]` | two GEMMs |
| 6 | 1891-1910 | `RUᵀ[D,C]·kc[C,D]` → delta`[D,D]`; then `state = γ·S0 + delta` | 128×128 |

where `S0` = the persistent `state` buffer (device, row-major `[D,D]`
per head, f32, offset `head_id*128*128`); `kc` = K staged in tg mem
`[C,128]`; `q` = read from device `conv_out` (k-region offset
`key_total + kh*128`); `Umat` = U `[C,128]` in tg mem; `RU[i,vi] =
exp(L_{c-1}-L_i)·U[i,vi]`.

`simdgroup_load`/`simdgroup_store` move 8×8 tiles to/from device or
threadgroup pointers with a stride + optional `transpose` flag (for
the `ᵀ` operands). Contraction D=128 → 16 steps of 8; C=16 → 2 steps.

## Threadgroup-memory budget

Add one `threadgroup float sdc[CW_C*128]` (8,192 B) — staging reused
across phases 3/5/6 (and q-staging for 2). Total 26,752 B of 32,768 —
safe. Phase 6's `[D,D]` delta (64 KB) exceeds tg memory → process it
in **column strips**: matmul a `[128, strip]` block → tg strip buffer
→ per-`vi` scalar `state = γ·S0 + delta` RMW into device `state`.
Strip width tuned at execution; diff oracle is the check.

## Execution order: 6 → 3 → 5 → (2)

Phase 6 first: heaviest (only `[D,D]` output), biggest isolated signal.
Phase 2 last + **optional** — `[16,16]` output, only win is the K=128
contraction; defer if the post-Phase-5 bench already satisfies.

### Setup (before Phase 6)
1. `#include <metal_simdgroup_matrix>` near `shaders.metal:28`.
2. Add the two simdgroup kernel attributes; add `sdc` tg buffer.
3. Smoke test compiles clean with `simdgroup_matrix` in scope.

### Per phase
Rewrite the phase's scalar matmul to cooperative `simdgroup_matrix`
(4 simdgroups split the output tiles; 8×8 `simdgroup_multiply_accumulate`
over the contraction). Store the result into tg memory (or device
`state` for Phase 6). **Keep the elementwise / precision-sensitive
epilogue** — β/γ scaling, the `i<l`/`i≤l` triangular masks,
`exp(log_decay)` — as the existing per-`vi` scalar code, now reading
the matmul's output buffer instead of recomputing a dot. Then verify.

### ⚠ Highest correctness risk — ragged chunks
For `n_tokens ∈ {1,4}`, chunk length `c < 8`, but `simdgroup_load`
always reads a full 8×8 tile. Every tg operand buffer (`kc`, `sdc`,
the `RU` staging) must be **explicitly zero-filled for rows `c..CW_C`**
before each matmul — Phase 1 staging only writes `l < c`. The diff
oracle's n=1/n=4 shapes catch a miss; expect to need this.

## Verification

After **each** phase:
1. `cargo test --release -p moeflux --features model-qwen3-6-35b-a3b
   --test smoke -- --include-ignored`
2. Diff oracle (gate, cos ≥ 0.9999, n∈{1,4,16,64} incl. g=0):
   `cargo test --release -p moeflux --features model-qwen3-6-35b-a3b
   --test graph_diff_oracle -- --ignored --test-threads=1
   graph_metal_matches_cpu_gated_delta_chunkwise`
   Untouched phases stay scalar → a failure isolates to the just-
   rewritten phase.

Final gate (after chosen phases):
3. Canary battery — the 6-test `diff_oracle` set (command in
   `kernel_arc_session8_landed.md` / session-11 verification block).
4. A/B bench: `prefill_profile.rs` `profile_1536`/`profile_8192`
   per-op breakdown — compare the `gated_delta_net` pole vs the 17.5%
   baseline. `bench.py` 992 + `prefill_prompt_long.txt`.

## Honest expectation

Estimated current kernel utilization is ~1.6% of f32 FLOP peak — so
the kernel may be occupancy / memory-latency bound, not compute-bound,
in which case simdgroup_matrix moves it little. Mike accepted this
("pull the lever, learn either way"). Phase 6 first gives the early
read: if it doesn't move the bench, the kernel isn't compute-bound and
later phases can be reconsidered. GPU capture was explicitly ruled out
(prior attempt produced untriageable GB/s of trace).

## Deferred follow-ups (not v1)

- **Phase 2** — do if Phase-5 bench leaves headroom; else a later session.
- **f16 v2** — stage `kc`/`RU`/`q` as `half`, `simdgroup_matrix<half>`
  inputs + `simdgroup_matrix<float>` accumulators (the raw builtin's
  mixed overload — the reason raw was chosen over `mma.h`). Halves tg
  footprint. Validate against the 0.9999 gate independently — the
  decay-weighted sums are precision-sensitive.
- **Delete `GatedDeltaNetStepNTokens`** (Op + per-token kernel +
  CpuBackend/GPU arms + `mod.rs:955` helper + 3 `graph_diff_oracle.rs`
  sites) once this session's bench confirms chunkwise is correct and
  decode (n=1) doesn't regress. It is currently kept only as the
  bench A-arm.

## Files

- `crates/moeflux/shaders/shaders.metal` — the only file edited.
- `crates/moeflux/src/riir/attn/linear_attn.rs:474` — CPU oracle, frozen.
- `crates/moeflux/tests/graph_diff_oracle.rs` — per-phase diff gate.
- `crates/moeflux/tests/prefill_profile.rs` — final A/B bench.
