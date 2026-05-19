# Future work — rustfmt config + pre-commit hook

Set up a `rustfmt.toml` for the moeflux workspace plus a pre-commit
hook that runs `cargo fmt`. Mike does this on another project and it
helps.

**Why:** when an editor/linter reformats on save, an agent's in-flight
`Edit` reads desync ("file modified since read"). If every file is
already rustfmt-canonical, a reformat is a no-op and produces no diff —
the desync stops happening. Also keeps the tree consistently formatted.

**Status:** deferred — Mike flagged it 2026-05-19, "one of these
sessions." Not blocking the kernel arc.
