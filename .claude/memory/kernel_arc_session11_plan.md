# Kernel arc — session 11 plan: SDPA → simdgroup_matrix ("make it go brrr")

**Supersedes the Tier-2-vs-MoE draft.** That fork was scoped off a
session-9 mindset where SDPA looked like ~7% of wall. The session-10
post-Tier-1 per-op profile (below) refuted it: at production prefill
shape, full-attention SDPA is **33% of wall and O(n²)**. This is the
prefill bottleneck and where the llama.cpp gap lives. Pivot locked
with Mike at the close of session 10.

## The profile that decided this

`MOEFLUX_PROFILE_PER_OP=1`, `prefill_profile.rs`, post-Tier-1 binary
(a236a0e). Two shapes — note how SDPA scaling differs from everything
else:

| op | 1536-tok | 8192-tok | ms/commit scaling |
|----|----------|----------|-------------------|
| `batched_sdpa_causal_flash` | 6.7% | **33.2%** | 34→799 = **23.5× (O(n²))** |
| `gated_delta_net_step` | 8.5% | 9.6% | 14.4→77 = 5.3× (O(n)) |
| `moe_permute_fuse` | 10.0% | 8.6% | O(n) |
| `batched_shared_ffn_moe_combine` | 5.0% | 4.5% | O(n) |
| unaccounted | 52% | 27% | fixed-overhead amortizes |

Linear attention is O(n); full attention is O(n²). At 8192 SDPA is
already 2.4× the entire MoE chunk; at 16k it is ~4× larger again; at
Agora's 40–60k it dominates everything. **SDPA is THE target.** MoE
GEMMs become session 12; Tier-2 `s0q` stays parked (a sub-1% sliver).

(Tier 1 itself is confirmed working: `gated_delta_net_step` went
17.84→14.4 ms/commit at 1536 since session 9 — a real ~19% op-level
drop; the +1.6% end-to-end was honest for a 9%-of-wall op.)

## The kernel — `attn_sdpa_causal_flash` (`shaders.metal:1473-1643`)

Flash-attention online-softmax; one threadgroup per (q_tile, head),
FA_THREADS=256 (8 simdgroups). Constants: FA_BR=64 (query tile),
FA_BC=16 (KV block), FA_HD=256 (head_dim, a3b), FA_RPS=8, FA_VPL=8.
Encoder `gpu_attn.rs:445` — grid `num_q_tiles × num_heads`, no dynamic
tg memory (all `threadgroup` arrays static). KV-block loop
(`:1529`) per threadgroup:

- **Phase 1** stage K block → `kv_stage[16,256]` (16 KB tg).
- **Phase 2 — QK^T** (`:1542-1562`): per (row,c) a scalar `fma` dot
  over FA_VPL + a `simd_sum` 32-lane reduce. **← GEMM target #1.**
- **Phase 3** online-softmax: block max, m_new, corr, exponentiate
  `scores` in place → P, update `row_l`. Stays scalar (cheap, not a
  matmul).
- **Phase 4** stage V block → `kv_stage` (reused, K now dead).
- **Phase 5 — P·V** (`:1614-1627`): `acc[ri][k] = acc*corr + Σ_c
  P·V`, fully scalar `fma`. **← GEMM target #2.**
- **Finalize** (`:1631`): `out = acc / row_l`.

`qreg[8][8]` and `acc[8][8]` are **register-resident**, per-lane
partitioned (lane holds head-dim `lane + k*32`). tg use today:
kv_stage 16 KB + scores 4 KB + row_{m,l,corr} 0.75 KB ≈ **20.75/32
KB** — ~11 KB free.

## The two GEMMs

**GEMM 1 — QK^T.** `scores[64,16] = Q[64,256] @ Kᵀ[256,16]`,
contraction HD=256 (32 ktiles). Output → tg `scores` (exists).
8 row-tiles × 2 col-tiles = 16 output tiles / 8 simdgroups.
`a` = Q tile (row=query, d=contraction); `b` = K staged as
`kv_stage[c,d]` loaded **transposed** → `[d,c]`. Post-GEMM: a cheap
elementwise pass applies `softmax_scale` + the causal mask (logic
unchanged from `:1554-1559`). Contraction is exact (256/8) — **no
zero-fill on the contraction axis**; raggedness is only on the BR
output rows (last q-tile, `br_valid<64`).

**GEMM 2 — P·V.** `O[64,256] += diag(corr)·O + P[64,16] @ V[16,256]`,
contraction BC=16 (2 ktiles). `a` = P = `scores` post-softmax (tg);
`b` = V = `kv_stage[c,d]` (no transpose — `[c=contraction, d=col]`).
Output O[64,256] = 8×32 = 256 tiles. **Lower-confidence than GEMM 1**
— Phase 5 is already embarrassingly parallel (each thread owns its
`acc[ri][k]`, no reduction), so the simdgroup win is the tensor-ALU
throughput, not removing a `simd_sum`. Gate it on its own bench;
keep GEMM 1 if GEMM 2 doesn't pay.

## Three design decisions — lock these in the opening conversation

Per `feedback_design_before_execute`, session 11 opens with a short
design pass on exactly these, then plan-mode, then execute.

1. **Ragged-q-tile OOB.** GEMM 1's `a` must read Q from tg or device
   (simdgroup_load can't read registers). The last q-tile has
   `br_valid<64`; loading Q tiles direct from device overreads the Q
   buffer (real fault, cf. the Tier-2 `s0q` hazard). Q[64,256]=64 KB
   won't tg-stage. Candidates: (a) **producer-side pad Q to a
   multiple of FA_BR** — 1 alloc change in `full_attn_forward.rs`,
   relaxes "shaders.metal-only" by one line, kills the problem
   cleanly; (b) re-stage a `[64,8]` Q ktile-strip into tg (2 KB,
   fits the 11 KB headroom), zero-filled for ragged rows, per ktile
   — shaders-only but re-stages 32× per KV block (Q is block-
   invariant → wasteful). **Lean (a).**
2. **O-accumulator residency.** O[64,256]=64 KB across the KV loop
   won't fit tg. Candidate: **O as register-resident
   `simdgroup_matrix<float,8,8>` accumulators** (32 tiles/simdgroup =
   64 floats/lane — identical footprint to today's `acc[8][8]`; this
   is the FlashAttention-2-on-tensor-cores standard).
   `simdgroup_multiply_accumulate` accumulates P·V in-register across
   blocks. **Lean register-resident simdgroup_matrix.**
3. **Per-row `corr` rescale of O.** With O in simdgroup_matrix tiles,
   `O ← diag(corr)·O` each block = left-multiply each row-tile by an
   8×8 diagonal `corr` matrix (`simdgroup_multiply`). Candidates:
   (a) build 8 diagonal matrices/block and multiply; (b) deferred-
   rescale variants. **Lean (a)** — explicit, matches the math.

## Phasing

- **Phase 0 — GPU capture first.** Before any rewrite, Xcode GPU
  capture / occupancy read on `attn_sdpa_causal_flash` at the 8192
  shape: is it compute-bound (→ simdgroup_matrix is the lever) or
  memory-bound on K/V re-reads (→ GQA-fold is the lever)? This is
  the session-12 plan-of-record's "GPU capture first" applied here,
  and it orders Phases 1–3. Session 9's discipline: measure before
  assuming the lever works.
- **Phase 1 — QK^T → simdgroup_matrix.** Highest-confidence win
  (removes the `simd_sum`). `shaders.metal` + (likely) the 1-line Q
  pad. Gate: all 4 `flash_diff_tokenwise` tests.
- **Phase 2 — P·V → simdgroup_matrix** with register-resident O +
  diag-`corr` rescale. `shaders.metal`-only. Gate: flash diff tests
  + its own per-op bench (keep only if it pays).
- **Phase 3 — GQA-fold** (stretch; likely session 12). One
  threadgroup per (q_tile, **kv_head**); stage each K/V block once,
  inner-loop over the 8 query-heads that share it → 8× less K/V
  global traffic. Restructures the kernel + changes the encoder grid
  (`gpu_attn.rs` — touches Rust). O stays one head's worth (heads
  processed sequentially in the inner loop). Gate: flash diff tests
  + bench.

## Verification

Per phase: the `flash_diff_tokenwise` battery in
`batched_diff_oracle.rs` — `sdpa_causal_flash_{n1,n4_tokenwise,
m512_square_causal,m1500_deep_chunk}`, per-token cosine ≥ 0.9999 vs
`sdpa_cpu`. `m1500_deep_chunk` is the ragged-tile + deep-KV case —
the load-bearing one for the OOB fix. Final: `graph_diff_oracle`
canary 12/12 + post-Tier-1-style per-op profile (`profile_8192`,
expect SDPA's % to drop) + post-reboot `bench.py`.

## Honest expectation

QK^T+PV is the bulk of SDPA's *compute*; sessions 9–10 got
1.85–2.2× converting scalar matmul to simdgroup_matrix. If SDPA is
compute-bound, ~2× on it ⇒ 33%→~18% ⇒ ~15% whole-prefill win, and it
**compounds with context length** (the O(n²) term). If Phase 0 shows
it's memory-bound, GQA-fold (8× less K/V traffic) is the bigger
lever and Phase 3 jumps the queue. Either way the headline metric is
`profile_8192` SDPA-% and the 15.7k `bench.py` number.

## Carry-overs

- Stale comment: `batched_diff_oracle.rs` ~line 720 says "FA_BR=32"
  — it's 64. Fix in passing.
- Still queued: delete `GatedDeltaNetStepNTokens` (per-token Op +
  kernel + arms) once a clean post-reboot bench confirms chunkwise
  wins and decode (n=1) doesn't regress — cheap, do it early.
- Tier-2 `s0q` and MoE GEMMs (`moe_permute_fuse` ~8.6%,
  `moe_combine` ~4.5%) remain real but secondary — session 12.
