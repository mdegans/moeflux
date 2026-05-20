---
name: moeflux-hardening-session-b-v2-plan
description: Plan-of-record for the full typed-BufId refactor. Supersedes the narrow Session B plan. Every BufId in the backend gets a role tag via `BufId<B: Buf>(u32, PhantomData<B>)`. ~400-450 net LOC, ~950 churn, one squashed commit. Designed by Mike + Opus 4.7 + Plan agent on 2026-05-20.
metadata:
  type: project
---

# TL;DR

Replace the untyped `pub struct BufId(pub u32)` with a generic
`BufId<B: Buf>(pub(crate) u32, PhantomData<B>)` in a new
`crates/moeflux/src/riir/backend/buftype.rs`. Every `BufId` field on
every `Op` variant, every `BufId` field on every scratch struct,
every method on the `BufferPool` trait, every `BufId` in producer
code gets a concrete role tag. The pool keeps `u32` internally; only
the public surface is generic-on-tag. Where multiple roles legitimately
feed the same `Op` field, we use either (a) a union tag with per-source
`From<BufId<X>> for BufId<Union>` impls or (b) a unidirectional
`From` from a canonical role tag to the consumer's role tag. No `Raw`
catch-all. Producer push sites do `.into()` at the call site — that
explicit `.into()` is documentation of "this role is legal here." End
state: zero bare `BufId` in producer code. Bug-of-record (session 19's
`h_mid_id` → `mlp_in`) becomes a compile error.

Single squashed commit; partial typing leaves a half-typed system
that's worse than untyped.

# Why this exists

Session 19 landed `Op::MoeGatherIdFuse` with kernel-level diff oracle
green and bench green. The producer at
`linear_attn_forward.rs:2796` passed `h_mid_id` (pre-RmsNorm residual)
into a field named `h_mid` on the Op — which the kernel uses as the
post-norm activation for gate/up matmuls. End-to-end output was
`</think>` + EoT. Session 20 found and fixed it with a one-line swap.
The bug survived a kernel diff oracle, code review, **and** the
Session A rename (`h_mid` → `mlp_in`). It cost two Claude sessions
plus Mike's time.

See [[prefill-gather-id-session-19-landed]] (RESOLVED) for the full
trace and [[moeflux-hardening-arc-plan]] for the three-session arc
this slots into.

This supersedes [[moeflux-hardening-session-b-plan]]
(scoped too narrowly — wrapped exactly one BufId class. Mike's call:
"this will make the code more verbose but also pay off in the bugs we
won't have, the sessions we won't waste").

# Locked decisions

1. **Module:** new `crates/moeflux/src/riir/backend/buftype.rs` holds
   `mod sealed`, the `Buf` trait, every concrete + union tag ZST, the
   `BufId<B>` struct, and every `From<BufId<A>> for BufId<B>` impl.

2. **`BufId<B>` shape:**
   `pub struct BufId<B: Buf>(pub(crate) u32, PhantomData<B>);`
   Derives: `Copy, Clone, Debug, Eq, PartialEq, Hash`. The `Buf` trait
   is sealed (`mod sealed { pub trait Sealed {} }`,
   `pub trait Buf: sealed::Sealed + Copy + Clone + 'static {}`).
   `pub(crate) fn from_raw(n: u32)` and `pub(crate) fn raw(self) -> u32`
   are the only constructors/extractors — producers can't mint or
   unwrap. **Keep `u32`, not `usize`** (less collateral; existing index
   width is fine).

3. **Pool internal representation: `u32` indices.** `BufferPool` trait
   methods themselves become generic-on-tag (e.g.
   `fn alloc<B: Buf>(...) -> Result<BufId<B>, _>`). This loses
   object-safety, but `Backend` uses an associated `type Pool: BufferPool`
   — never `dyn`. Internal impl bodies operate on `id.raw()`.
   **No extension trait, no untyped escape.** Full migration.

4. **No `Raw: Buf` catch-all.** A `Raw` tag would invite
   `from()`/`into()` abuse that defeats the entire point. If a buffer
   has no role, we name a role for it.

5. **`Op` enum: concrete tagged fields.** Not generic Ops. Union tags
   (ZST unit structs with `From` impls from sources) cover fields that
   legitimately accept multiple roles.

6. **Per-pair `From` impls in `buftype.rs`.** No global `SubtypeOf`
   trait. Every legal conversion is its own three-line impl, auditable
   by reading one file.

7. **Role-based naming:** `MoeInputBuf` (not `PostNormBuf`),
   `AttnInputBuf` (not `PreAttnNormOutBuf`).

8. **Single squashed commit.** If we hit a logical break mid-refactor,
   we re-plan, not partial-land.

9. **Scope: wrap everything.** Zero bare `BufId` in producer code after
   the refactor.

10. **`HiddenDoubleBuffer` (Q4):** one tag `HiddenBuf` for both
    `hidden_a` and `hidden_b`. Layer-parity is orchestrator state; the
    type system can't catch parity bugs anyway.

11. **`BucketGate ↔ GateMid` aliasing (Q1):** unidirectional `From`.
    Allocate canonical as `BucketGateBuf` (the original path); env-on
    gather-id path does `.into()` at the push site to convert to
    `GateMidBuf`. Same for `BucketUpBuf → UpMidBuf` and
    `BucketOutBuf → DownMidBuf`. The `From` site **is** the
    documentation of which way the buffer flows. One canonical
    allocation tag per slot; no union needed.

12. **`RmsNormBf16NTokens` field tags (Q6):** input `x: BufId<RmsNormIn>`
    (union: `EmbedOutBuf`, `ResidualBuf`, `HiddenBuf`), output
    `out: BufId<RmsNormOut>` (union: `AttnInputBuf`, `MoeInputBuf`,
    `TailNormedBuf`). Producer push sites do `.into()` at both ends.

13. **`LayerForwardBuffers::batch_out: [BufId; 7]` (Q7):** break to 7
    named fields, each with its distinct role tag. The array indexing
    was already a smell.

14. **`MoeBuffers` legacy cogito path (Q8):** one tag
    `DeprecatedCogitoBuf` covers every field on the struct. Doc the tag
    itself: "Per-token cogito MoE path; frozen pending M5 Studio with
    512GB RAM. When we rewrite this we'll type the buffers properly.
    Do not iterate without retagging." Single tag for the whole struct
    — when the cogito path is rewritten (post-M5) we re-type from
    scratch.

15. **Compile-fail tests (Q9):** doc-comment examples with
    `compile_fail` annotations, no `trybuild` dep. Runs in
    `cargo test`'s doc-test phase.

# Tag vocabulary

## Concrete role tags (40 total)

`EmbedOutBuf`, `ResidualBuf`, `HiddenBuf`, `AttnInputBuf`,
`MoeInputBuf`, `TailNormedBuf`,
`QProjOutBuf`, `QBuf`, `KProjOutBuf`, `VProjOutBuf`, `QGateBuf`,
`AttnOutBuf`, `OProjOutBuf`,
`KvCacheKBuf`, `KvCacheVBuf`,
`RouterLogitsBuf`, `RouterIdxBuf`, `RouterWeightsBuf`,
`SharedGateBuf`, `SharedFfnGateBuf`, `SharedFfnUpBuf`,
`SharedFfnActBuf`, `SharedFfnDownBuf`,
`MoeOutSumBuf`,
`BucketInputBuf`, `BucketGateBuf`, `BucketUpBuf`, `BucketActBuf`,
`BucketOutBuf`, `BucketTokenIdxBuf`, `BucketWeightsBuf`,
`ExpertIndicesBuf`, `ExpertBaseBuf`,
`HtpeBuf`, `HidsBuf`, `GateMidBuf`, `UpMidBuf`, `DownMidBuf`,
`ConvOutBuf`, `ConvStateBuf`,
`QkvStackBuf`, `ZStackBuf`, `AlphaStackBuf`, `BetaStackBuf`,
`GDecayBuf`, `BetaGateBuf`, `DeltaStateBuf`, `DeltaOutBuf`,
`ValueOutBuf`,
`RopeInvFreqBuf`, `TokenIdsBuf`, `LogitsBuf`,
`DeprecatedCogitoBuf`.

(53 concrete tags — yes, that's a lot. Each is justified by a distinct
role at a producer/consumer boundary; collapsing erases the
documentation value.)

## Union tags (4 total)

| Union | `From` sources | Used in |
|---|---|---|
| `MatvecIn` | `AttnInputBuf`, `MoeInputBuf`, `AttnOutBuf`, `ValueOutBuf`, `SharedFfnActBuf` | `Op::MatvecNTokens.input` |
| `MatvecOut` | `QProjOutBuf`, `KProjOutBuf`, `VProjOutBuf`, `OProjOutBuf`, `RouterLogitsBuf`, `SharedGateBuf`, `SharedFfnGateBuf`, `SharedFfnUpBuf`, `SharedFfnDownBuf`, `LogitsBuf`, `QkvStackBuf`, `ZStackBuf`, `AlphaStackBuf`, `BetaStackBuf` | `Op::MatvecNTokens.output` |
| `RmsNormIn` | `EmbedOutBuf`, `ResidualBuf`, `HiddenBuf`, `ConvOutBuf` (for `RmsNormQk`) | `Op::RmsNorm*.x` |
| `RmsNormOut` | `AttnInputBuf`, `MoeInputBuf`, `TailNormedBuf` | `Op::RmsNormBf16NTokens.out` |

`MatvecOut` is 14 sources — well above the "≥4 = smell" threshold, but
matvec is genuinely the graph's fan-out node and splitting
`MatvecNTokens` would multiply Op surface across both backends. Mike
confirmed: keep as union.

# Migration phases

## Phase 1 — Create `buftype.rs` (standalone)

- `mod sealed` + `Buf` trait
- `BufId<B>` struct with `pub(crate)` field, derives, methods
- 53 concrete tag declarations + 4 union tag declarations
- ~30 `From` impls (per the union-source tables above + the 3
  bucket→mid unidirectional)
- 4-6 doc-comment `compile_fail` examples demonstrating the bug-of-
  record class
- Re-export from `backend/mod.rs`; delete old `pub struct BufId`
- Update `GraphError::BadBufId(BufId)` → `BadBufId(u32)`

**Net LOC:** ~300. Standalone — compiles in isolation.

## Phase 2 — Migrate the `BufferPool` trait (signatures)

- `BufferPool::alloc::<B: Buf>(...) -> Result<BufId<B>, _>` generic-on-
  tag; internal body operates on `u32`.
- Same for `handle`, `upload`, `upload_at`, `download`,
  `register_borrowed`, `as_mut_slice_u8`, `as_mut_slices_u8`,
  `label`, `alloc_aligned`.
- Both impls (`MetalBufferPool`, `CpuBufferPool`) update mechanically.
- `lifetime.rs` stays tag-agnostic — operates via new
  `Op::reads_raw() -> Vec<u32>` / `writes_raw() -> Vec<u32>` (added
  alongside the existing typed `reads()` / `writes()`).
- Test helper `fn buf<B: Buf>(n: u32) -> BufId<B>` (test-only) for
  the `tests::one_of_each` style fixtures.

**Net LOC:** ~80 net (~200 churn).

## Phase 3 — Migrate `Op` enum field types + dispatch arms (main thread)

22 `Op` variants. Each variant's `BufId` fields get concrete tags per
the inventory tables in the v1 plan / agent output. Both backend
dispatch arms update in lockstep. Single-threaded (the central enum
is the conflict point; subagent fan-out would just merge-conflict).

**Net LOC:** ~50 net (~200 churn).

## Phase 4 — Migrate scratch structs (subagent fan-out per-struct)

9 scratch structs: `MoeGraphScratch`, `LinearAttnGraphScratch`,
`FullAttnGraphScratch`, `LayerForwardBuffers` (largest — includes the
7-named-field rewrite of `batch_out`), `HiddenDoubleBuffer`,
`HeadTailScratch`, `MoeBuffers` (all fields tagged
`DeprecatedCogitoBuf`), `KvCacheLayerState`, `ExpertFiles::mmap_buf_ids`.

Fan out one subagent per struct (or pair small ones).

**Net LOC:** ~30 net (~200 churn).

## Phase 5 — Migrate producer code (subagent fan-out per-file)

4 producer files: `attn/linear_attn_forward.rs` (heaviest, ~69 sites),
`attn/full_attn_forward.rs` (~14 sites), `moe/expert_forward.rs` (~32
sites; mostly `DeprecatedCogitoBuf`), `riir/mod.rs` (~1 site).

Fan out one subagent per file. Each subagent's contract:

1. Read the file once, find every `Op::...{ field: bufid, .. }` push.
2. Type-check immediately, OR add `.into()` at the push site, OR
   report "no `From` impl exists for this conversion" as a finding
   (= either bug or missing impl in `buftype.rs`).
3. Return a list of `.into()` sites added and any findings.

**Net LOC:** ~50 net (~150 churn).

## Phase 6 — Verification

Per [[feedback-coherence-test-before-pipeline-commit]] and
[[feedback-bench-discipline]]:

1. `cargo build --release --features "moeflux-model-qwen3-6-35b-a3b,axum,cli,toml"` clean
2. `cargo test -p moeflux` — canary 9/9 still green
3. `cargo test -p moeflux --test '*'` — integration tests
4. Coherence curl with env=off: "What is the capital of France?" →
   "The capital of France is Paris."
5. Same curl with env=on (`MOEFLUX_MOE_GATHER_ID=1`) — same answer
   (this is the original session-19 trap)
6. Post-reboot bench, n=5: expect no perf change (ZST overhead = 0
   bytes)

## Phase 7 — Squashed commit

Subject: `Type-tag every BufId in the backend; producer-wiring bugs become compile errors.`

# Risk surface

1. **`BufferPool` trait restructure (Phase 2).** Loses object-safety.
   Confirmed not load-bearing — `Backend::Pool` is associated, never
   dyn. If the build surfaces a `dyn BufferPool` we didn't know about,
   that's a real surprise; flag and re-plan.

2. **`lifetime.rs` / coloring stays tag-agnostic.** Operates on `u32`
   via `Op::reads_raw()` / `writes_raw()`. If the diff shows
   `lifetime.rs` importing tag types, that's a smell — back out.

3. **Some `From` impls may be missing for legal conversions.** Phase 5
   subagents are instructed to report rather than silently add.
   Mike (or main thread) confirms each addition before landing.

4. **`MoeBuffers` is mostly dead code.** Tagging it `DeprecatedCogitoBuf`
   is the cheapest correct answer; deletion is independent work post-M5.

# Subagent fan-out plan

| Phase | Mode | Boundary |
|---|---|---|
| 1 | Main thread | Foundation; must verify standalone build |
| 2 | Main thread | Trait restructure; biggest design call |
| 3 | Main thread | Central enum; subagent fan-out would conflict |
| 4 | Subagents | Per-struct (9 structs, ~5 agents) |
| 5 | Subagents | Per-file (4 files, 4 agents) |
| 6 | Main thread | Verification |
| 7 | Main thread | Commit |

# LOC estimate (rolled up)

| Phase | Net LOC | Churn LOC |
|---|---|---|
| 1 | +300 | +300 |
| 2 | +80 | ~200 |
| 3 | +50 | ~200 |
| 4 | +30 | ~200 |
| 5 | +50 | ~150 |
| **Total** | **~+510** | **~1050** |

One focused session.

# Cross-references

- [[prefill-gather-id-session-19-landed]] — the bug class
- [[moeflux-hardening-arc-plan]] — the arc this slots into (Session B)
- [[moeflux-hardening-session-b-plan]] — superseded narrow version
- [[feedback-coherence-test-before-pipeline-commit]] — Phase 6 gate
- [[feedback-bench-discipline]] — Phase 6 bench protocol
- [[feedback-design-before-execute]] — discipline applied
- [[feedback-pretty-is-a-goal]] — ergonomic motivation
