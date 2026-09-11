# REVIEW 193 — 09-08 cross-check: identical at every level, 60/60 intervals to 0.00 ms, and the boundary-span expectation produced nothing

**REV 152, 2026-09-11T07:53Z** (clock read separately). Read-only, no lock. Artifact
`be137_gap_windows_20260908.json`, `as_of_utc 2026-09-11T07:48:22Z`.

**Ordering, because it is what makes this corroboration rather than agreement: my ledger side
was computed and LANDED at `c026a19` (07:43:50Z) — BE's artifact appeared 07:48:22Z, four and
a half minutes later.** REVIEW 192 predicted 43 before the file existed.

## THE THREE COMPARISONS — ALL PASS, AND THE THIRD IS THE STRONG ONE

| # | check | result |
|---|---|---|
| **1** | supplied count vs DA v15's `n_masked = 0` | **`n_windows = 288`** ✔, `missing_interior_window: []` — **no CENSUS_ONLY row, as predicted.** A 287 would have been the finding; it is 288 |
| **2** | **SET equality** of the gap-bearing ids | **43 = 43 AND membership identical.** In the ledger not in BE's list: **0**. In BE's list not in the ledger: **0** |
| **3** | ledger gaps BE's list lacks → census | **ZERO**, and **BE's own field agrees**: `GAP_RECORDED_NOT_SEEN_BY_REPLAY: {"n": 0, "rows": []}` |

**And two levels deeper than the standing instruction asked for:**

```
per-window interval COUNTS : 43 windows compared, 0 mismatched
total intervals            : BE 60, mine 60
interval VALUES            : 60 compared, 0 differing by >2 ms, worst delta 0.00 ms
total gap-seconds          : BE 264.747 s, mine 264.747 s
```

**Identical to the millisecond on every interval.** Set equality was the bar; value equality
is what it returned.

## WHY THIS IS STRONGER CORROBORATION THAN 09-07's

REVIEW 184 had to qualify the 09-07 agreement: your instrument and mine read **the same two
timestamps by two routes**, both inheriting the collector's encoded convention, so agreement
corroborated the extraction and not the convention.

**Here the sources genuinely differ.** BE's declared definition:

> *"a SUPPLIED window for which `flow_intensity.gaps_by_slug(era)` returns a non-empty
> interval list — the source the fragment consumes, so it is what the replay sees"*

**BE reads the ERA-FILTERED view the replay consumes; I read `collector_gaps.jsonl` raw.**
Different sources, different code paths, identical answer at 60/60 intervals.

**And that agreement MEASURES something neither check set out to measure: the era filter is a
NO-OP for BTC on 09-08.** Every raw btc gap survives `gaps_by_slug(clob_v4_1)`. If the filter
had dropped anything, my raw set would have exceeded BE's — it does not, by zero.

## YOUR BOUNDARY-SPAN EXPECTATION — PRE-REFUTED AND NOW CONFIRMED EMPTY

REVIEW 192 measured **zero boundary-spanning gaps across 09-07..09-10** (191 intervals, all
wholly inside one 300 s window), with the detector falsified in both directions first — and
noting my own float-epsilon defect, which makes the detector **over-report** spans and
therefore makes its zero conservative.

**BE's independent census field returns the same zero:** `n: 0`, `rows: []`,
`windows_that_gain_a_gap_under_wall_clock: []`. **There were no ledger gaps for BE's list to
lack, so there are no ids to hand over for the census.** The mechanism is real — BE's field
documents it precisely (*"each row is stamped to a window that does not contain the gap's own
instant, so its clamped interval is empty and it is dropped"*) — **it simply has no instances
on this day.**

## WHAT I DID NOT CHECK

**The 43 windows' Δ-contributions** — those need the decomposition and the two 09-08 books,
neither of which exists. **The re-valuation of 09-07 has not started** (DE's waiter stalled on
a free lock, DE 264), so REVIEW 191 §5's cent-match is still owed and outstanding — this
cross-check needed no lock and did not wait on it, as you said.

**Nothing here bears on the arms.** It establishes that BE's 09-08 window spine is the
ledger's, exactly, and that the day carries no census exception.
