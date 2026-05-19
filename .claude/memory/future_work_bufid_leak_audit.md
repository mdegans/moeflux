# Future work — BufId arena: stale-handle audit + RAII design

## Accurate model (corrected session 16, 2026-05-19)

`MetalBufferPool` is an **arena**, not a malloc/free allocator.
Storage is `Vec<Buffer>`, indexed indirectly via `bufid_to_physical`.
There *are* free paths — the earlier "no free path" framing was wrong:

- `reset_transient()` truncates `buffers`/`labels`/`persistent`/
  `byte_sizes`/`bufid_to_physical` past the last persistent BufId.
  `Vec::truncate` drops the `metal::Buffer` values, and `metal::Buffer`
  is RAII (Drop sends ObjC `release`). Transients are bulk-freed.
- Pool drop frees everything.

So there is **no memory leak**. The arena model is exactly bumpalo's:
no per-allocation free, bulk-free at arena drop / `reset_transient`.

## The real latent hazard — stale BufId (slot use-after-free)

`BufId(u32)` is an arena index. It is `#[derive(Copy, Clone, Debug,
Eq, PartialEq, Hash, Ord, PartialOrd)]` and used as a `HashMap` key in
`commit_plan`'s lifetime coloring. A `BufId` used *after*
`reset_transient` truncated its slot is a use-after-free of the
*slot* (OOB or wrong-buffer). `label()`'s `unwrap_or("<bad-bufid>")`
is a tell someone anticipated this.

Currently NOT live: every BufId lives in a scratch struct whose
lifetime ⊆ the pool's. `commit_plan` remaps *physical* indices, not
BufId indices, so BufIds survive `commit_plan`. They die only at
`reset_transient`.

Also still true: any *per-layer / per-token* `register_borrowed` /
`alloc` would be an unbounded BufId-space slot leak. P7 Part 1
rejected an Option-B design for exactly this. Audit pass: confirm
every BufId call site is once-per-run or once-per-resource, never
per-step; count BufIds at run end vs. expected.

## Options if the hazard goes live (do NONE now)

- **Epoch counter (cheap, recommended first):** pool holds a
  generation counter; `reset_transient` bumps it; `BufId` carries its
  epoch; `debug_assert` epoch match on every access. Catches the
  stale-handle bug class with zero lifetime infection. The
  bumpalo-lite move.
- **Non-Copy RAII BufId (Mike's preferred end-state, big redesign):**
  `BufId` not `Copy`/`Clone`, single-ownership, `Drop` returns the
  slice to the pool (optionally zeroes it — or zero the whole pool on
  drop instead). Needs the pool reachable from the handle (back-ref /
  shared handle). Collides with current usage: `commit_plan` aliasing,
  `HashMap` keys, free copies into `Op`s. Its own design conversation.
- **Drop-bomb:** `BufId` panics on `Drop` unless explicitly returned.
  Wrong for an arena (you *want* to drop handles freely) — rejected
  this session.
- **`'bump` lifetime on `BufId`:** compile-time safety, but infects
  every `Op`/`Graph`/scratch struct + the `Backend` trait signature.
  Too invasive for a non-live hazard.

Also noted: pool storage may go contiguous (single backing buffer)
instead of `Vec<Buffer>` in the future. Fine as-is for now.

Decision (session 16): none of this mid-arc. Epoch counter is the
first escalation; the non-Copy RAII redesign gets its own design
conversation once the graph-mode/prefill arc settles.
