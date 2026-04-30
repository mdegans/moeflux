#!/usr/bin/env python3
"""
Aggregate a samply CPU profile (Firefox-Profiler JSON) plus its
`--unstable-presymbolicate` sidecar into a top-N self/inclusive
report. Designed for `scripts/profile_smoke.sh`.

Self-time = leaf-frame samples per symbol. Inclusive = any-frame.

The sidecar resolves symbols against the test binary's RVA table
without needing the original binary on disk later.

Usage:
    profile_aggregate.py <profile.json> [--top N] [--filter SUBSTR]
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from collections import Counter


def load_profile_and_syms(profile_path: str) -> tuple[dict, dict]:
    with open(profile_path) as f:
        profile = json.load(f)
    syms_path = profile_path.replace(".json", ".syms.json")
    if not os.path.exists(syms_path):
        print(
            f"[warn] no presymbolicate sidecar at {syms_path}; "
            "addresses will stay raw",
            file=sys.stderr,
        )
        syms = {"string_table": [], "data": []}
    else:
        with open(syms_path) as f:
            syms = json.load(f)
    return profile, syms


def build_resolver(profile: dict, syms: dict):
    """Pick the largest user-binary in the profile's libs and build
    a sorted (rva, symbol) lookup against the syms sidecar."""
    sym_strings = syms["string_table"]
    libs = profile.get("libs", [])

    # Heuristic: the test binary is the one whose name doesn't look
    # like a system lib (no `.dylib`, not in `/usr/lib`).
    target = None
    for lib in libs:
        name = lib.get("name", "")
        if name.endswith(".dylib") or "/usr/lib" in lib.get("path", ""):
            continue
        if "dyld" in name:
            continue
        target = lib
        break
    if target is None and libs:
        target = libs[0]

    code_id = target.get("codeId") if target else None
    rvas: list[tuple[int, str]] = []
    for entry in syms.get("data", []):
        if entry.get("code_id") == code_id:
            for s in entry.get("symbol_table", []):
                rvas.append((s["rva"], sym_strings[s["symbol"]]))
            break
    rvas.sort()
    sorted_rvas = [r[0] for r in rvas]
    rva_to_sym = {r[0]: r[1] for r in rvas}

    def resolve(addr: int) -> str | None:
        i = bisect.bisect_right(sorted_rvas, addr)
        if i == 0:
            return None
        return rva_to_sym[sorted_rvas[i - 1]]

    return resolve, target


def aggregate(thread: dict, resolve) -> tuple[Counter, Counter]:
    samples = thread["samples"]
    stack_table = thread["stackTable"]
    frame_table = thread["frameTable"]
    fr_addr = frame_table["address"]
    s_frame = stack_table["frame"]
    s_prefix = stack_table["prefix"]

    self_time: Counter = Counter()
    inclusive: Counter = Counter()

    for stk_idx in samples["stack"]:
        if stk_idx is None:
            continue
        leaf_frame = s_frame[stk_idx]
        leaf_sym = resolve(fr_addr[leaf_frame]) or f"<?>0x{fr_addr[leaf_frame]:x}"
        self_time[leaf_sym] += 1

        seen: set[int] = set()
        s = stk_idx
        while s is not None and s not in seen:
            seen.add(s)
            fr = s_frame[s]
            sym = resolve(fr_addr[fr]) or f"<?>0x{fr_addr[fr]:x}"
            inclusive[sym] += 1
            s = s_prefix[s]
    return self_time, inclusive


def print_table(title: str, counter: Counter, total: int, top: int, filt: str | None) -> None:
    print(f"\n=== {title} ===")
    items = counter.most_common()
    if filt:
        items = [(n, t) for n, t in items if filt in n]
    for name, t in items[:top]:
        pct = 100.0 * t / total if total else 0.0
        print(f"  {pct:5.1f}%  {t:6}  {name[:140]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", help="path to samply profile JSON")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--filter", default=None,
                    help="only show symbols containing this substring")
    args = ap.parse_args()

    profile, syms = load_profile_and_syms(args.profile)
    resolve, target = build_resolver(profile, syms)
    print(f"# resolved against: {target.get('name', '?') if target else '<no lib>'}")

    threads = sorted(
        profile["threads"],
        key=lambda t: t.get("samples", {}).get("length", 0),
        reverse=True,
    )
    main_thread = threads[0]
    n_samples = main_thread.get("samples", {}).get("length", 0)
    print(f"# top thread: {main_thread.get('name', '?')}  ({n_samples} samples)")

    self_time, inclusive = aggregate(main_thread, resolve)
    total = sum(self_time.values())

    print_table(
        f"Top {args.top} SELF-TIME",
        self_time,
        total,
        args.top,
        args.filter,
    )
    print_table(
        f"Top {args.top} INCLUSIVE",
        inclusive,
        total,
        args.top,
        args.filter,
    )

    # Other busy threads — useful for spotting rayon-pool work.
    print()
    print("# other busy threads:")
    for t in threads[1:5]:
        n = t.get("samples", {}).get("length", 0)
        if n > 0:
            print(f"  {t.get('name', '?')}: {n} samples")


if __name__ == "__main__":
    main()
