# Future work — BufId-space leak audit

The pool's `register_borrowed` (and `alloc`) only ever *push* to
`buffers` / `bufid_to_physical` — there is no free path. That is fine
for once-per-resource registrations (mmap layers, run-lifetime
scratch) but any per-layer / per-token `register_borrowed` would be an
unbounded BufId-space leak. P7 Part 1 rejected an Option-B design for
exactly this reason.

Once the graph-mode arc settles, do a pass: confirm every BufId handed
out has a once-per-run (or once-per-resource) call site, not a
per-step one. Count BufIds at run end vs. expected.

Mike's options (he's written an allocator like this before):
- **Drop-bomb**: `BufId` panics on `Drop` unless explicitly returned
  to the allocator. Extreme but turns leaks into immediate runtime
  failures — the version he used in the past.
- **RAII free-on-drop**: give `BufId` a `Drop` impl that calls back
  into the allocator and frees the slot. Softer — no panic, the leak
  just can't happen. Needs the pool reachable from the handle (a
  back-reference / shared handle), which is a real design change
  given the current `BufId(u32)` newtype.

Neither is planned mid-arc. Recorded as the escalation path; the
plain inspection audit comes first.
