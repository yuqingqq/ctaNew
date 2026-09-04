# P003 preliminary-result reliability audit

As of: **2026-09-04T09:59Z**  
Economic-result tip inspected: **`659ed66`**; race/coverage status refreshed
through **`ad68601`**  
Scope: read-only review of the released two-arm forward result, its declarations,
consumer code, and the script used for the reported profitability calculation.
No large feed was rebuilt and no model was refit.

## Verdict

The narrow preliminary finding is credible:

> On the two pre-declared read days, 2026-09-01 and 2026-09-02, the frozen
> candidate did not beat the frozen incumbent on BTC when the two arms were
> compared at the same realised number of cancellation actions.

This is a **descriptive two-day ranking result**, not validation-grade evidence
and not a profitability result. It is enough to reject the claim that the
candidate has already demonstrated an improvement. It is not enough to prove
that the candidate is structurally worse.

The published profitability block is **withdrawn as unreliable**. The reported
`$226,594` filled notional, `$1,801.29` no-cancel P&L, `0.7949%` return,
`+$620.58` overlay improvement, `+34.5%`, and `$807/day` are not supported by
the population that the calculation actually sums.

## What was reproduced

The nine BTC `MATCHED_VOLUME` values reproduce exactly from the preserved
two-arm feeds for 09-01, 09-02, and their pool:

| population | 5% | 10% | 15% |
|---|---:|---:|---:|
| 09-01 | -789.12c | -2,016.71c | -1,476.01c |
| 09-02 | -227.60c | -1,237.84c | -2,975.36c |
| pooled | -1,012.68c | -3,038.75c | -3,949.76c |

The declaration ordering also checks: the interim declaration was committed at
`eeb02ba` (12:55:32Z), before the 09-01 result (`7719588`, 15:41:26Z) and the
09-02 addendum (`40b49fb`, 16:21:39Z).

Module-level controls passed during this audit:

- `be_interim_declaration.py --selftest`: **21/21**.
- `be_forward_metric.py --selftest`: **102/102**.

These checks corroborate the arithmetic and declaration machinery. They do not
close the pipeline-wiring and inference limitations below.

## What the comparison means

`MATCHED_VOLUME` is the repository's label, but it matches the **number of
cancellation actions**, not shares, notional, or capital. The candidate acts at
its frozen threshold. The incumbent is then ranked over the complete evaluated
population and its cutoff is lowered retrospectively until its action count
equals the candidate's realised count (`be_read_cells.py:86-99,144-175`).

The incumbent is **not** a prediction-free benchmark. Both arms are linear
harmful-flow predictors using 54 Polymarket features plus six fine-flow
features. Both score expected cancellation value as predicted fill hazard
multiplied by predicted conditional value
(`harmful_forward_scorer.py:167-174`). The difference is:

- candidate: frozen `PM_PLUS_FINE` fit;
- incumbent: per-coin `INCUMBENT_REWEIGHTED_ONLY` fit, trained with
  generation-balanced weights.

Therefore a negative value means that the candidate selected less valuable
cancellations than the incumbent at equal action count. It is not realised
trading loss and it does not answer whether prediction beats no cancellation or
a non-predictive rule.

## Reliability limits on the negative result

1. **Only two pre-declared economic days have been read.** The 08-29 read is
   development evidence. The race now has `G=3` because 09-03 accrued under
   R-503, but 09-03 has not become a third economic read.
2. **No day-cluster interval is claimable.** The reported permutation p-values
   use window-level sign flips even though UTC day is the ruled cluster unit.
   The declaration itself marks this as weaker/optimistic. A high one-sided p
   for `candidate > incumbent` is failure to show a win, not proof of a loss.
3. **The equal-count comparator is retrospective.** The incumbent's top-K
   cutoff is determined using the full evaluated day/pool. That is useful for a
   descriptive ranking-quality comparison, but it is not an executable policy
   operating point.
4. **The primary result is not on the canonical consumer path.** The committed
   `be_read_cells.compute()` still produces `BY_THRESHOLD` and `BY_COUNT` cells
   and never calls `matched_volume()` (`be_read_cells.py:230-303`). Repository
   search finds no committed caller for `matched_volume()`; the published read
   was assembled by a scratch `interim_report.py`. Thus the number is
   reproducible today but not yet a stable, in-repo result pipeline.

## Why the profitability claims are withdrawn

The scratch calculation `prof.py` keeps only the **first row** encountered for
each `(slug, side, gen)` action (`prof.py:17-20`) and labels sums from those rows
as total filled notional and no-cancel P&L (`prof.py:24-28`). Actions can have
multiple rows, so first-row selection is not a total-fill aggregation.

More fundamentally, the emitted scale is explicitly `preventable_shares`
(`be_forward_metric.py:619-624`), not all filled shares. Its producer includes
only fills inside the one-second action horizon and, at 50 ms latency, only
tranches at or after the latency cutoff (`harmful_exposure_rows.py:339-370`).
Fills before 50 ms are recorded separately as `stale_shares`, and fills outside
the one-second horizon are outside this population.

Consequently:

- the denominator is not total filled notional;
- the baseline is not the whole no-cancel book;
- baseline and overlay need not use the same row within a multi-row action;
- the resulting percentages and daily dollar extrapolations do not have their
  stated meanings.

Even after repairing that aggregation, any profitability result must still add
fees, realised exit/settlement economics, quote size, inventory, and capital.
The current five-second gross markout is not a net return.

## Current programme state

R-503 re-admitted 09-03 on its covered complement: 287/288 windows, with the
missing 15:20Z window named and counted as accounted loss. The latest verdict
says it accrues, moving the race to **G=3** (09-01, 09-02, 09-03). Two more
accruing days are needed. The superseding 09-03 verdict lost the scheduled-unit
attribution prefix, which remains an open provenance issue; this does not turn
09-03 into an additional read result. The later `589af56` repair now emits
coverage-absent windows as their own named/countable status, distinct from
blackout-masked windows. That closes the missing-status defect but does not
change this economic-result audit.

## Required next steps before stronger claims

1. Wire `matched_volume()` into a committed result runner, with a positive
   control, known-bad refusal, and durable result artifact.
2. Keep the retrospective equal-count result labelled as a diagnostic, or
   predeclare a causal incumbent operating point before subsequent days.
3. Accumulate the ruled day unit and report day-cluster uncertainty; do not
   interpret window-level permutation p-values as validation.
4. Replace `prof.py` with an action-native fill ledger that aggregates every
   fill exactly once and carries fees, exits/settlement, quote size, inventory,
   capital, and explicit statuses for unavailable terms.
