#!/usr/bin/env bash
#
# Build a moeflux test binary with the flags samply needs on macOS,
# capture a CPU profile via samply (with presymbolicate sidecar so
# we don't depend on the binary still being on disk later), and
# aggregate the top hot symbols.
#
# Usage:
#   scripts/profile_smoke.sh                      # default: cogito_v2_smoke
#   scripts/profile_smoke.sh <test-name>          # any test in tests/
#   scripts/profile_smoke.sh <test-name> <fn>     # specific test fn
#   scripts/profile_smoke.sh --open               # open in browser
#                                                 #   (default: --save-only)
#
# Outputs:
#   /tmp/moeflux_profile.json
#   /tmp/moeflux_profile.syms.json
#   prints aggregated top self/inclusive on stdout
#
# Why these flags:
#   * --release          — match the perf path; otherwise we profile
#                          unoptimized code.
#   * `-C force-frame-pointers=yes` (RUSTFLAGS) — without frame
#                          pointers samply walks only one stack
#                          frame per sample on macOS arm64 and
#                          everything aliases to the test harness.
#   * `CARGO_PROFILE_RELEASE_DEBUG=true` — embed DWARF for atos /
#                          samply to resolve symbols.
#   * `--unstable-presymbolicate` (samply) — sidecar `.syms.json`
#                          with all referenced symbols, so the
#                          aggregator doesn't need the binary on
#                          disk later.
set -euo pipefail

TEST="${1:-cogito_v2_smoke}"
TESTFN="${2:-}"
OPEN="${OPEN:-no}"
if [[ "${1:-}" == "--open" ]]; then OPEN="yes"; TEST="${2:-cogito_v2_smoke}"; TESTFN="${3:-}"; fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_JSON=/tmp/moeflux_profile.json
OUT_SYMS=/tmp/moeflux_profile.syms.json

echo "[profile] building $TEST with debuginfo + frame pointers..." >&2
RUSTFLAGS="-C force-frame-pointers=yes" \
  CARGO_PROFILE_RELEASE_DEBUG=true \
  cargo build -p moeflux \
    --no-default-features \
    --features model-cogito-v2-671b \
    --release \
    --tests >&2

# Find the most recent test binary (deps has many, pick by name + mtime).
BIN=$(ls -t "$REPO_ROOT"/target/release/deps/${TEST}-* 2>/dev/null \
        | grep -vE '\.(d|dSYM)$' | head -1 || true)
if [[ -z "$BIN" ]]; then
  echo "[profile] could not find test binary for '$TEST'" >&2
  exit 1
fi
echo "[profile] binary=$BIN" >&2

SAMPLY_FLAGS=( record --unstable-presymbolicate -o "$OUT_JSON" )
if [[ "$OPEN" == "no" ]]; then
  SAMPLY_FLAGS+=( --save-only )
fi

if [[ -n "$TESTFN" ]]; then
  TEST_ARGS=( "$TESTFN" --ignored --nocapture )
else
  TEST_ARGS=( --ignored --nocapture )
fi

echo "[profile] samply ${SAMPLY_FLAGS[*]} -- $BIN ${TEST_ARGS[*]}" >&2
samply "${SAMPLY_FLAGS[@]}" -- "$BIN" "${TEST_ARGS[@]}" >&2

if [[ "$OPEN" == "no" ]]; then
  python3 "$REPO_ROOT/scripts/profile_aggregate.py" "$OUT_JSON"
fi
