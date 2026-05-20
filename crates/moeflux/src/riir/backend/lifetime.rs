//! Lifetime analysis and interval-coloring for the buffer pool.
//!
//! We use per-BufId intervals over op-index coordinates rather than
//! full liveness sets. For our domain — straight-line graphs with
//! single-def BufIds (the producer always allocates a fresh `BufId`
//! for each new logical value) — interval-overlap reduces *exactly*
//! to the interference graph that full liveness analysis builds. The
//! two approaches produce identical colorings; intervals are simply
//! cheaper.
//!
//! MoE doesn't break this: each MoE block is a single fused `Op`,
//! so data-dependent expert routing happens *inside* the kernel, not
//! in the graph topology. Coloring sees one node with fixed
//! reads/writes.
//!
//! If a future workload introduces genuine branching, disjoint live
//! ranges per BufId, or parallel cmdbuf scheduling, the `Lifetimes`
//! type and `analyze_lifetimes` body would swap to a backward-
//! liveness pass without touching `greedy_color` or the pool
//! integration.
//!
//! ## Colorability discriminator
//!
//! A `BufId` is colorable iff *some `Op` writes to it*. BufIds that
//! appear only in `Op::reads()` (never in `Op::writes()`) were
//! uploaded externally and their content must be preserved — they
//! must NOT alias internal scratch. These are absent from the
//! returned `Lifetimes::intervals`.

use std::collections::HashMap;

use super::Graph;

/// Live range of a transient BufId in op-index coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interval {
    /// Op index where the BufId is first written (becomes live).
    pub first_write_op: u32,
    /// Op index of the last read (becomes dead at end of this op).
    /// Inclusive: while encoding op `last_read_op`, the buffer is
    /// still live.
    pub last_read_op: u32,
}

/// Per-BufId lifetime intervals — keyed by the raw `u32` index.
/// Tag-agnostic: this pass cares only about index coordinates, not
/// role tags, so it never imports a [`super::buftype::Buf`] tag.
/// Only BufIds that are written by at least one `Op` appear in
/// `intervals` — pure external inputs are absent and must not be
/// aliased.
#[derive(Debug, Default)]
pub struct Lifetimes {
    pub intervals: HashMap<u32, Interval>,
}

/// Walk `graph.ops` once, building per-BufId `Interval`s.
///
/// Algorithm:
/// 1. For each op, set `first_write` for each written BufId (if not
///    already set — guards against repeated writes preserving the
///    earliest live point).
/// 2. For each op, set `last_read` for each read BufId that has been
///    written by a prior or current op. Overwriting `last_read`
///    naturally tracks the latest read.
/// 3. In-place RMW (a BufId in both `reads()` and `writes()` at the
///    same op): the BufId either gets first_write_op set this op
///    (then last_read_op also = this op), or first_write_op is
///    earlier and last_read_op extends to this op. Both branches
///    produce a correct interval.
pub fn analyze_lifetimes(graph: &Graph) -> Lifetimes {
    let mut first_write: HashMap<u32, u32> = HashMap::new();
    let mut last_read: HashMap<u32, u32> = HashMap::new();

    for (op_idx, op) in graph.ops.iter().enumerate() {
        let op_idx = op_idx as u32;
        for buf in op.writes_raw() {
            first_write.entry(buf).or_insert(op_idx);
        }
        for buf in op.reads_raw() {
            if first_write.contains_key(&buf) {
                last_read.insert(buf, op_idx);
            }
        }
    }

    let mut intervals = HashMap::with_capacity(first_write.len());
    for (buf, first) in first_write {
        // A BufId written but never read still needs to exist after
        // the write (its content is the "value" the graph produced).
        // Treat its live range as a single point at the write op.
        let last = last_read.get(&buf).copied().unwrap_or(first);
        intervals.insert(
            buf,
            Interval {
                first_write_op: first,
                last_read_op: last,
            },
        );
    }

    Lifetimes { intervals }
}

/// One physical-buffer slot index after coloring. Multiple `BufId`s
/// mapping to the same `ColorId` share a physical buffer.
pub type ColorId = u32;

/// Result of greedy linear-scan interval coloring.
#[derive(Debug, Default)]
pub struct ColoringMap {
    /// `BufId` raw index → physical-slot index. Multiple BufIds may
    /// share a color when their intervals don't overlap.
    pub bufid_to_color: HashMap<u32, ColorId>,
    /// Number of distinct colors (= physical slots needed for the
    /// colorable BufIds). Tight upper bound on simultaneous-live
    /// transient buffers.
    pub color_count: u32,
}

/// Greedy linear-scan coloring (register-allocator style).
///
/// Sort intervals by start; maintain `active` (currently-live
/// colors). For each interval: free any expired entries from
/// `active`, reuse the lowest free color if one exists below
/// `next_color`, else allocate `next_color` and increment.
///
/// Deterministic: ties on `first_write_op` break by `BufId.0` so
/// the same `Lifetimes` always produces the same `ColoringMap`
/// regardless of `HashMap` iteration order. CPU and Metal pools
/// running this on identical lifetimes therefore agree on the
/// physical layout.
pub fn greedy_color(lifetimes: &Lifetimes) -> ColoringMap {
    let mut intervals: Vec<(u32, Interval)> = lifetimes
        .intervals
        .iter()
        .map(|(b, i)| (*b, *i))
        .collect();
    intervals.sort_by_key(|(b, i)| (i.first_write_op, *b));

    let mut active: Vec<(ColorId, u32)> = Vec::new();
    let mut next_color: ColorId = 0;
    let mut bufid_to_color: HashMap<u32, ColorId> =
        HashMap::with_capacity(intervals.len());

    for (buf, interval) in &intervals {
        active.retain(|(_, last_read)| *last_read >= interval.first_write_op);

        // Find lowest free color. Scan active colors in ascending
        // order; first gap below `next_color` is reused.
        let mut active_colors: Vec<ColorId> =
            active.iter().map(|(c, _)| *c).collect();
        active_colors.sort();
        let mut expected: ColorId = 0;
        let mut chosen: Option<ColorId> = None;
        for &c in &active_colors {
            if c != expected {
                chosen = Some(expected);
                break;
            }
            expected += 1;
        }
        let color = match chosen {
            Some(c) => c,
            None => {
                // No gap was found inside active_colors. If `expected`
                // walked off the end below `next_color`, the gap is at
                // `expected` (a previously-freed color). Else allocate
                // a brand-new color.
                if expected < next_color {
                    expected
                } else {
                    let c = next_color;
                    next_color += 1;
                    c
                }
            }
        };

        active.push((color, interval.last_read_op));
        bufid_to_color.insert(*buf, color);
    }

    ColoringMap {
        bufid_to_color,
        color_count: next_color,
    }
}

#[cfg(test)]
mod tests {
    use super::super::buftype::{
        Buf, BufId, ConvOutBuf, HiddenBuf, OProjOutBuf, ResidualBuf,
    };
    use super::super::Op;
    use super::*;

    /// Mint a typed `BufId<B>` from a raw index for fixtures.
    fn buf<B: Buf>(n: u32) -> BufId<B> {
        BufId::from_raw(n)
    }

    /// Synthesize a `ResidualAddNTokens` against raw indices; the tag
    /// choices below match the `Op::ResidualAddNTokens` field-tag
    /// inventory (a = `OProjOutBuf`, b = `RmsNormIn` union over
    /// `HiddenBuf`, out = `ResidualBuf`). Live-range coloring is
    /// tag-agnostic so the assignments are just there to satisfy
    /// the type checker.
    fn resid(a: u32, b: u32, out: u32) -> Op {
        Op::ResidualAddNTokens {
            label: "test",
            a: buf::<OProjOutBuf>(a),
            b: buf::<HiddenBuf>(b).into(),
            out: buf::<ResidualBuf>(out),
            n_tokens: 1,
            dim: 1,
        }
    }

    #[test]
    fn empty_graph_has_no_intervals() {
        let g = Graph::new();
        let lt = analyze_lifetimes(&g);
        assert!(lt.intervals.is_empty());
    }

    #[test]
    fn pure_input_bufids_absent_from_intervals() {
        // Single residual_add: a, b are read-only inputs, out is
        // written. Only `out` should appear.
        let mut g = Graph::new();
        g.push(resid(0, 1, 2));
        let lt = analyze_lifetimes(&g);
        assert_eq!(lt.intervals.len(), 1);
        assert!(lt.intervals.contains_key(&2));
        assert!(!lt.intervals.contains_key(&0));
        assert!(!lt.intervals.contains_key(&1));
        // `out` is never read after being written → live range is
        // just the write op.
        assert_eq!(
            lt.intervals[&2],
            Interval { first_write_op: 0, last_read_op: 0 }
        );
    }

    #[test]
    fn chain_of_residuals_has_one_interval_per_intermediate() {
        // resid(0,1) -> tmp1
        // resid(tmp1, 2) -> tmp2
        // resid(tmp2, 3) -> tmp3
        // resid(tmp3, 4) -> final
        // tmp1, tmp2, tmp3 are colorable; final is written but never
        // read (still appears in intervals as single-point).
        let mut g = Graph::new();
        g.push(resid(0, 1, 5)); // op 0: tmp1=resid(0,1)
        g.push(resid(5, 2, 6)); // op 1: tmp2=resid(tmp1,2)
        g.push(resid(6, 3, 7)); // op 2: tmp3=resid(tmp2,3)
        g.push(resid(7, 4, 8)); // op 3: final=resid(tmp3,4)
        let lt = analyze_lifetimes(&g);
        // tmp1..tmp3 and final → 4 entries.
        assert_eq!(lt.intervals.len(), 4);
        assert_eq!(
            lt.intervals[&5],
            Interval { first_write_op: 0, last_read_op: 1 }
        );
        assert_eq!(
            lt.intervals[&6],
            Interval { first_write_op: 1, last_read_op: 2 }
        );
        assert_eq!(
            lt.intervals[&7],
            Interval { first_write_op: 2, last_read_op: 3 }
        );
        assert_eq!(
            lt.intervals[&8],
            Interval { first_write_op: 3, last_read_op: 3 }
        );
    }

    #[test]
    fn rmw_bufid_has_single_point_interval() {
        // RmsNormQkNTokens reads and writes the same BufId.
        let mut g = Graph::new();
        g.push(Op::RmsNormQkNTokens {
            label: "qk",
            x: buf::<ConvOutBuf>(0),
            num_k_heads: 4,
            key_dim: 128,
            key_offset_per_token: 512,
            per_token_total: 1024,
            n_tokens: 1,
        });
        let lt = analyze_lifetimes(&g);
        // x is written at op 0, read at op 0 → single-point.
        assert_eq!(
            lt.intervals[&0],
            Interval { first_write_op: 0, last_read_op: 0 }
        );
    }

    #[test]
    fn coloring_empty_lifetimes() {
        let cm = greedy_color(&Lifetimes::default());
        assert_eq!(cm.color_count, 0);
        assert!(cm.bufid_to_color.is_empty());
    }

    #[test]
    fn coloring_disjoint_intervals_reuses_color() {
        // tmp1 live [0,1], tmp2 live [2,3] — disjoint, share one
        // color.
        let mut lt = Lifetimes::default();
        lt.intervals.insert(
            0,
            Interval { first_write_op: 0, last_read_op: 1 },
        );
        lt.intervals.insert(
            1,
            Interval { first_write_op: 2, last_read_op: 3 },
        );
        let cm = greedy_color(&lt);
        assert_eq!(cm.color_count, 1);
        assert_eq!(cm.bufid_to_color[&0], 0);
        assert_eq!(cm.bufid_to_color[&1], 0);
    }

    #[test]
    fn coloring_overlapping_intervals_use_two_colors() {
        // tmp1 live [0,2], tmp2 live [1,3] — overlap at op 1-2.
        let mut lt = Lifetimes::default();
        lt.intervals.insert(
            0,
            Interval { first_write_op: 0, last_read_op: 2 },
        );
        lt.intervals.insert(
            1,
            Interval { first_write_op: 1, last_read_op: 3 },
        );
        let cm = greedy_color(&lt);
        assert_eq!(cm.color_count, 2);
        assert_ne!(cm.bufid_to_color[&0], cm.bufid_to_color[&1]);
    }

    #[test]
    fn coloring_ping_pong_chain_uses_two_colors() {
        // 10-op residual chain: each op writes a fresh tmp, reads
        // the previous one. Live ranges are [i, i+1] for i in 0..9.
        // Adjacent intervals overlap at one op → 2 colors suffice.
        let mut g = Graph::new();
        // BufIds 100..200 reserved for transients; 0..10 for inputs.
        for i in 0..10 {
            let a = if i == 0 { 0 } else { 100 + i - 1 };
            let out = 100 + i;
            // Use i+1 as the "second input" (a distinct read-only).
            g.push(resid(a, i + 1, out));
        }
        let lt = analyze_lifetimes(&g);
        let cm = greedy_color(&lt);
        // 10 transient outputs (BufIds 100..110). Adjacent ones
        // overlap → 2 colors max. The final output (100+9=109)
        // never read after write — its single-point interval can
        // share with whatever color is free at op 9.
        assert!(
            cm.color_count <= 2,
            "expected ≤ 2 colors, got {}",
            cm.color_count
        );
    }

    #[test]
    fn coloring_residual_chain_compresses_intermediates() {
        // Same as `chain_of_residuals_has_one_interval_per_intermediate`
        // but assert the coloring outcome.
        let mut g = Graph::new();
        g.push(resid(0, 1, 5));
        g.push(resid(5, 2, 6));
        g.push(resid(6, 3, 7));
        g.push(resid(7, 4, 8));
        let lt = analyze_lifetimes(&g);
        let cm = greedy_color(&lt);
        // 4 entries (3 intermediates + final). Intervals:
        //   buf 5: [0,1]
        //   buf 6: [1,2]  — overlaps buf 5 at op 1
        //   buf 7: [2,3]  — overlaps buf 6 at op 2
        //   buf 8: [3,3]  — overlaps buf 7 at op 3
        // Chain depth = 2 (only ever 2 intermediates live at once).
        assert_eq!(cm.color_count, 2);
    }

    #[test]
    fn coloring_is_deterministic_across_runs() {
        // Run the same colorer twice on isomorphic Lifetimes built
        // in different insertion orders; result must agree.
        let mk = |order: &[(u32, u32, u32)]| -> Lifetimes {
            let mut lt = Lifetimes::default();
            for &(b, fw, lr) in order {
                lt.intervals.insert(
                    b,
                    Interval { first_write_op: fw, last_read_op: lr },
                );
            }
            lt
        };
        let lt_a = mk(&[(10, 0, 1), (20, 1, 2), (30, 2, 3)]);
        let lt_b = mk(&[(30, 2, 3), (10, 0, 1), (20, 1, 2)]);
        let cm_a = greedy_color(&lt_a);
        let cm_b = greedy_color(&lt_b);
        assert_eq!(cm_a.color_count, cm_b.color_count);
        assert_eq!(cm_a.bufid_to_color, cm_b.bufid_to_color);
    }
}
