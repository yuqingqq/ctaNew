# Review of the take + maker exit result — 2026-11-25

**Verdict.** No invalidating defect: the accounting reconciles, every take and exit fill re-derives from the raw book and trade
tape, and the strategy stays positive on all four days under every stress tested. But the reported +$3,934 is the optimistic end
of the range. The conservative full-path bound (strict take fills + the measured 58 ms order latency) is **+1,618.55**
(+5.76 per market, 217/281 markets positive). The result also applies only to the 24 % of markets
the replay can value, and those have about twice the per-share edge of the rest. The value is essentially the taker leg; the maker
exit adds consistency, not total. Consumed days, development evidence only.

Two parts:

- an independent fresh-agent code review, given code and outputs only ([reviewer_report.md](reviewer_report.md));
- quantitative checks and full-path sensitivities declared in [review_plan.md](review_plan.md) before each ran.

## The range of the result

| Version (PASSIVE_EXIT, 281 markets) | END net USD | Per market | Days positive | Markets positive | Takes filled | Without young exit value |
|---|---:|---:|---:|---:|---:|---:|
| v2 as reported: lenient take fills, 40 ms | +3,933.54 | +14.00 | 4/4 | 256/281 | 3,575 | +4,054.33 |
| S1: strict take fills, 40 ms | +2,021.19 | +7.19 | 4/4 | 233/281 | 2,600 | +2,172.43 |
| S2: strict take fills, measured 58 ms | +1,618.55 | +5.76 | 4/4 | 217/281 | 2,338 | +1,738.05 |

## Findings (ranked)

1. **Material: take fills against quotes that may already be cancelled.**
   - The replay's IOC fills against the last venue snapshot before its effect and never checks the next one. Predict publishes
     ~50 ms snapshots.
   - 1,799 of 3,575 take fills (62,160 shares) took depth that had
     vanished without trades by the next snapshot ([diag.json](diag.json)). These fills carry most of the take value.
   - TAKE-SIGNAL's two-frame rule, which established the edge, was not carried into the replay. S1 restores it on the full paths.
2. **Material: 40 ms order delay vs the measured 58 ms.** The exec probe's measured order-live latency is p50 58 ms (pf_clock_rl.py).
   S2 applies it together with S1. The 50 ms decision grid adds ~25 ms on top, so an event-driven client would sit between S1 and S2.
3. **Material for generalising: the population.**
   - "Complete" requires the book to stay two-sided and fresh to 298 s, which cannot be known live.
   - On TAKE-SIGNAL's triggers its markets earn +3.81 ¢/sh at 3 s vs +2.22 / +1.75 in the excluded classes.
   - The v2 report's "mostly still contested at the close" is **wrong**: 65 % of complete markets end with the mark at ≤ 5 or ≥ 95.
     What they share is a two-sided book, and 91 % near-money triggers.
4. **Minor: the exits.**
   - About half of exit shares fill only at our price (queue-dependent).
   - PASSIVE_EXIT − HOLD (+$6.90) is not robust: exit timing used ASYNC_FAST10, and the Tokyo 58 / 37 ms effects exceed the difference.
   - Exits add consistency (more markets positive, ~¼ the inventory), not total value.
5. **Minor: the standing checklist (C1–C3, C12, C15) was not run in full.** The reviewer's partial computations:
   - M 250 on the IOC trade debit −3.6 %;
   - cancel-pending exits 2.4 % of shares;
   - WS-timed exits +$242 vs E2 −$37.

   M 250 on the exits needs a replay.
6. **Minor: v2 checks.** G3 does not re-derive the venue frame (the reviewer did, with all matching) and has no falsifier. The
   Binance median feed delay leaves 14 % of triggers within the p99 delay tail.

**Sound:**
- client-known limit and own-cross;
- depth not reused, partial fills, fee;
- totals reconcile;
- settlement vs END mark barely matters ($3,911.84 vs $3,933.54).

## Corrections (in band)

- v2's headline is the lenient upper bound, not a central estimate.
- Its population description is withdrawn.
- "Beats HOLD" is withdrawn as not robust.

These supersede the v2 report's readings in the state files. The v2 receipt stays as provenance.

## What would make it decision-grade

1. **Untouched days:** the frozen rule with the strict IOC rule and the measured latency.
2. **The full C1–C3 / C12 / C15 checklist**, including M 250 on the exits.
3. **A valuation for non-complete markets** (e.g. settlement), so the population is not selected by a future property.
4. **Live facts:**
   - whether Predict supports a true IOC / marketable order path, and its real latency;
   - competitor speed: a Tokyo shadow log of how fast stale quotes disappear.
