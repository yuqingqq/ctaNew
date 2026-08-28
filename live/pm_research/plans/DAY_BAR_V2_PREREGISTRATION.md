# Forward-day admissibility bar v2 — PRE-REGISTERED RULING

**Committed:** 2026-08-28T05:43Z, coordinator.
**Authority:** the user's O2 ruling (R-232, structured answer 2026-08-28 ~05:38Z)
approving the DIRECTION: judge days on data loss and contamination, not raw
event count. This document fixes the EXACT thresholds, per rule 6/11:
committed BEFORE the first day it judges.
**Applies to:** UTC days beginning **2026-08-29T00:00:00Z**, per coin, under the
per-coin verdict regime. Days before that stay under the old count bar and
their verdicts stand: 08-26 FAIL (Q-DA-72), 08-27 EXCLUDED (R-222),
08-28 judged tonight under the OLD bar. Nothing retroactive.

## 1. Predicates (ALL must hold, per coin, for a complete-day PASS)

Window grid: 288 contiguous 300 s UTC windows. Gap scope: COIN-LEVEL
(R-191) — a gap on the coin's feed blinds every window it intersects,
regardless of which slug logged it. Source: the LIVE `collector_gaps.jsonl`
with its as-of recorded; the tape6e ledger pin is NOT a valid source for
full-day figures (frozen 08-27T11:56Z — Q-DA source-trap note).

- **P1 — severity.** `lost_seconds / 24 ≤ 120 s/hr` (≡ ≤3.33% coverage loss).
- **P2 — material contamination breadth.** Windows with **≥75 s** (25% of
  span) intersected by gap: **≤5% of windows** (≤14 of 288).
- **P3 — concentration.** Max rolling-60-minute lost seconds **≤900 s**
  (no hour loses more than 25% of itself).

**Reported diagnostics, NO bar:** raw windows-gap-affected share (any-overlap);
gap count/hr; per-cause duration distributions. The raw breadth share is
near-invariant under the O1 fix (a 1.3 s gap blinds a window as surely as an
11.3 s one) and is dominated by sub-2 s gaps; a bar on it would fail good days
forever. It is kept as information: severity collapsing while breadth persists
is the expected post-O1 signature, not a failure.

## 2. Derivations (anchors chosen from scoring needs, not from what passes)

- **P1** anchors to the 3.33% coverage-loss sketch written in
  `BTC_FEED_MITIGATION_DESIGN_2026-08-28.md` (§O2) BEFORE today's
  measurements existed. Row-level GAP statuses already exclude contaminated
  rows one by one; the day bar's job is to cap the aggregate loss those
  exclusions represent.
- **P2** anchors to row-population materiality: a window with ≥25% of its
  span in gap loses ≥~25% of its decision times; capping such windows at 5%
  of the day bounds the day's materially-degraded breadth. This predicate is
  NOT binding on the frequent-tiny-gap regime (measured worst on the three
  degraded days: 0.3%); it exists to catch the RARE-LONG-OUTAGE regime the
  old count bar was originally built for.
- **P3** anchors to rule 7 (controls matched on hour): an hour losing >25% of
  itself cannot be credibly hour-matched. Measured worst rolling-60-min loss
  on the degraded days: 301.2 s.

## 3. Grounding measurements (btc, coin-level, live ledger as-of 08-28T05:31Z)

| day | lost s/hr (P1) | mat≥25% windows (P2) | worst-60min s (P3) | raw breadth (diagnostic) | verdict under v2 |
|---|---|---|---|---|---|
| 08-25 | 151.9 FAIL | 0.0% pass | 301.2 pass | 80.6% | FAIL (P1) |
| 08-26 | 123.8 FAIL | 0.3% pass | 278.6 pass | 62.2% | FAIL (P1) |
| 08-27 | 130.6 FAIL | 0.0% pass | 256.3 pass | 70.1% | FAIL (P1) |
| post-O1 counterfactual (crude: −10 s detection lag per long gap) | 54–79 | ~0% | 131–158 | ~unchanged | PASS expected |
| post-O1 counterfactual (DA, measured per-gap lag) | ~28–36 | ~0% | — | ~unchanged | PASS expected |

**AMENDED 2026-08-28T06:04Z, before any judged day (supersedes the three-branch
text below, rule 13; original kept for provenance).** DA corrected its own
counterfactual: O1a SHORTENS detection to ~3 s (ping-tracked), it does not
eliminate it — the det=0 model was for a fix that is not the one deploying.
Corrected per-gap model: **~60–77 s/hr** (76.5/59.9/63.9 for the three
reference days), which sits ON TOP of the cruder 54–79 band — the two models
no longer discriminate. The pre-registered reading is now ONE band:

- **expect ~55–80 s/hr** — consistent with both models; the fix worked as
  modelled (roughly halving, not quartering);
- **below ~45 s/hr** → something else ALSO improved — most plausibly O1b's
  backoff shortening the reconnect residual, which neither model prices;
- **above ~120 s/hr (P1 FAIL)** → the detection-lag diagnosis was WRONG and
  the mechanism is not what either model assumed — NOT "the fix
  underperformed." This branch survives the correction unchanged and is the
  one worth having declared.

*Superseded original (det=0 mis-calibration):* the counterfactuals differ
~2.6x; ~30 supports the per-gap subtraction, ~79 the cruder aggregate, >120
falsifies both.

Both independent breadth computations agree exactly (232/179/202 of 288).
eth passes every predicate trivially (~3.6 lost s/hr over the same span).

## 4. Implementation requirements (DA, after v2.3 verification + blocker 7)

1. Extend the day-verdict tool with P1/P2/P3, per coin, evaluated at
   00:06Z against the live ledger with as-of recorded in the verdict.
2. COIN-LEVEL evaluation is mandatory; the verifier's existing per-slug
   `windows_gap_affected` field must be renamed or dual-reported so the two
   definitions cannot be conflated (they differ by construction on bad days).
3. Falsifiers ship with the checker (rule 15): a synthetic long-outage day
   must FAIL P2 and P3; a synthetic high-loss day must FAIL P1; a malformed
   or truncated ledger must REFUSE, not pass.
4. The verdict artifact names this document and its commit as the governing
   bar for days ≥2026-08-29.

## 5. Interaction with the O1 boundary

O1 deploys at 2026-08-29T00:00Z (user-approved, R-232) — the same instant this
bar takes effect. Era ruling (recorded in-band): no row-stamping change; the
boundary is distributional; O1d makes never-connected gap RECORDS complete
going forward, so cross-boundary duration comparisons treat pre-boundary
never-connected gaps as understated. First post-deploy structural check:
the PING_TIMEOUT duration distribution must collapse toward ~ping_interval
+ 1.3 s (R-182 within-cause verification; never a throughput A/B).
