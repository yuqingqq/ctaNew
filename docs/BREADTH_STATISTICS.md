# Two breadth statistics, and why they must not be conflated

**Status:** definitional note (DA, 2026-09-01). No bar, no ruling. It exists
because two different measurements are printed side by side on every day
verdict and they nearly agree — which is the shape that gets conflated.

## The near-agreement

On 2026-08-29, btc:

| statistic | value | share |
|---|---|---|
| COIN-LEVEL touched windows | 95 / 288 | 33.0% |
| PER-SLUG affected windows | 93 / 288 | 32.3% |

0.7 points apart. A reader who takes one for the other is right to within a
point — until the day they are not. Same day, **eth**: COIN-LEVEL 1, PER-SLUG
0. There the two disagree about whether *any* window was affected at all.

This is the class Q-BE-175 named: two quantities that nearly collide
numerically are more dangerous than two that obviously differ, because
nothing about the numbers themselves prompts the check.

## The definitions

### A — COIN-LEVEL touched windows
`da_forward_day_verify.coin_level_affected_windows(lo, hi, coin, ...)`

For each of the day's **288 window spans** `[w0, w0+300)`, does it intersect
any **merged coin-level gap interval**? Input is the **gap ledger** only.

- Denominator: always 288 (or, on an open day, the complete elapsed windows —
  both are carried, see below).
- It counts a window blinded by a gap logged against a **neighbouring** slug.
  That is deliberate and ruled: a gap on the adjacent stream still blinds this
  window (R-191, day-bar doc §4.2).
- **This is the governing scope** for anything that reads breadth, and it is
  the numerator of the 0h disclosure carried beside P1/P2/P3.

### B — PER-SLUG affected windows
`da_forward_day_verify.per_slug_affected(coin, lo, hi, covered_slugs, gaps_by_slug)`

Over the slugs the **era selector covers** inside `[lo, hi)`, how many carry a
non-empty gap list **keyed to that slug**? Input is the **slug map** only.

- Denominator: `era_covered_windows`, the covered-slug count — **not always
  288**. Under the stale `fi.ERA` literal it read 0 on days holding 288
  windows (B1); it is now derived per day.
- It cannot see a neighbour's gap, and it *can* flag a window whose own 300 s
  span holds no interval (the slug's stream gapped while recording past its
  own window end).
- Retained for continuity. **It is not the bar's input.**

## Why they differ, in two directions

Both are exercised by a fixture in `da_forward_day_verify._selftests`:

1. A gap keyed to slug 5 that runs into window 6 → **A counts 6, B does not.**
2. A gap keyed to slug 20 whose interval sits inside window 25's span →
   **B counts slug 20, A counts window 25.**

So neither is a bound on the other, and neither can be derived from the other.
The suite asserts this three ways: the counts must disagree on that fixture;
the per-slug denominator must move with the covered-slug set while the
coin-level one stays 288; and **neither function's signature can see the
other's inputs** — the coin-level count takes no slug map, the per-slug count
takes no ledger. A mutation that derives one from the other fails the suite.

## Which statistic each receipt carries

| receipt / field | statistic |
|---|---|
| `day_bar_v2[coin].windows_affected_disclosure` (0h) | **A**, with both denominators: `affected_over_elapsed` and `affected_over_288` |
| `windows_gap_affected[coin].gap_affected_COIN_LEVEL` / `_pct_COIN_LEVEL` | **A**, /288 |
| `windows_gap_affected[coin].gap_affected_PER_SLUG` / `_pct_PER_SLUG` | **B**, / `era_covered_windows` |
| HANDOFF contamination survey, "windows w/ gap" column | **B** |
| HANDOFF contamination survey, "est. rows gap-before-cutoff" column | neither — a **row-level** estimator over decision rows, a third quantity again |

## Three denominators, stated once

Even within statistic A there are two, and on an open day they differ:
`affected / complete elapsed windows` is the live rate, `affected / 288` is
progress toward the closing-day denominator. Only the second is quoted in a
complete-day receipt; on a closed day they coincide. The 09-01 open-day report
printed 52/113 and 52/288 from one numerator, and both were correct.

## What none of them are

A count of **contaminated** windows. A gap opens a blind interval and forces a
modeled queue reset and repost; the replay clears state, resynchronizes and
re-anchors from the next quote, and busy windows carry thousands of `book`
snapshots. Overlap does not mean the rest of the window is stale — the
opposite claim was made on 2026-09-01 and withdrawn the same day.
