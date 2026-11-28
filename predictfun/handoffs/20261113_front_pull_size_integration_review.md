# Size and ladder results: implications for the model goal

The new study does not show that more size or a six-level ladder fixes the
selection problem. A larger order can nevertheless retain earned priority across
partial fills; the reported losses do not disprove that mechanism. The user
has clarified this objective explicitly; see [the size/priority comparison](20261113_size_priority_objective.md). It also does not establish that reproducing makers' order
shape leaves only queue position as the remaining gap. Its ladder is a sum of
independent lanes, and its repeated-fill statistic differs from the public-
order statistic used for comparison.

The [published report](20261113_front_pull_size_results.md) and its raw/review
JSON copies were inspected. Report as of **2026-11-13T07:22:49Z**: **nine
consumed dates**, 2,592 windows / 5,184 window-sides. Daily status records retain
15,984 untimed first-minute legs (188,923.35 shares); 591,868 model-alignment
rows have zero reported mismatches. These counts identify the reported
population; this review does not independently rerun its historical controls.

## What the reported numbers support

Under its M/50 queue/clock setting, the FPQ single-level policy has one-second
markout of **-1.1826 / -1.1868 / -1.1775 cents per non-young filled share** at
5 / 30 / 100 shares. Reported daily values are **-$48.18 / -$234.22 / -$562.08**.
The six-rung, 30-share-per-rung arm reports **-$5,094.91/day**, and loses against
the single-level 30-share arm on all nine dates at both M/0 and M/50.
These are the saved fill-markout statistics, not matched-inventory portfolio
returns. Reported R410 is `NOT-SHOWN`, R411 `LADDER-HURTS`, R412 `0 of 6`, and
both R413 positivity counts are `0 of 3`.

One prose correction: R410's saved `R410_days_up_down` is `[[5,3],[4,6]]`.
Thirty shares improves cents/share over five shares on **5/9** M/0 dates and
**3/9** M/50 dates, not the report's 5/4. The `NOT-SHOWN` conclusion is unchanged.

## Source limits that matter for our next decision

1. **The ladder has no common allocation of trade volume.**
   `pf_front_pull_size.fp_eval` calls `lane` separately for each rung and sums
   their fills. Each `pf_kp_s4.sim2` starts from the full same trade tape and
   resets `left = tq[i]`. Consumption by one rung is not passed to another.
   There is no shared inventory cap, reservation or risk controller either.
   This is a comparison of summed independent lanes, not a reconciled joint
   ladder portfolio. Source inspection establishes that missing conservation
   mechanism; it does not quantify historical double allocation or give a
   corrected ladder result. Also, the deeper rungs use a different rule: no
   entry block, and the learned/guard pull applies only at the known best.
   The negative result pertains to this implementation and risk allocation.

2. **Repeated fills are counted differently in simulation and the maker study.**
   `pf_front_pull_size.order_fill_counts` counts fill records inside a simulated
   order's time interval. `pf_kp_s4.load_day` aligns legs sharing a taker order
   to one time but does not aggregate those legs; `window_sides` retains them
   as separate trade records. Thus multiple legs from one taker order can count
   repeatedly. In contrast, `pf_orders.day_measures` uses `og.fid.nunique()` with
   `fid` equal to the taker-order identity, or an anchor-time fallback when
   unknown. The report's **41.1% / 60.6%** simulated orders filled at least three
   times at 30 / 100 shares cannot directly establish that the strategy has
   matched or exceeded makers' **25.8%** repeated-distinct-event retention.
   This review does not estimate how often the definitions disagree. The subsequent [synthetic mechanism check](20261113_size_priority_objective.md) confirms that the report counter counts three legs from one taker event as three fills; it does not quantify historical incidence.

3. **Removing young fills is a reporting filter.** `FP.add` bins realized fills
   by order age; the report sums bins with age at least 100 ms. It does not
   rerun the path with those fills suppressed and subsequent inventory/order
   decisions changed. This differs from our full-path age500 sensitivity.
   Per-fill marks use `np.nansum`; these summaries do not carry explicit
   missing-mark counts. No missing-mark frequency is inferred here.

4. **Queue sensitivity is not a demonstrated model gain.** The 30-share FPQ
   result changes from **-$234.22/day** under M/50 to **+$491.08/day** with U/50's
   nothing-ahead assumption; the ladder remains negative at **-$4,148.89/day**
   under U/50. That identifies strong dependence on the fill assumption.
   It does not identify our actual queue place or prove that forecast quality
   is sufficient. Nothing-ahead is an optimistic queue assumption, not a
   general mathematical upper bound on P&L when extra fills can lose money.

The inherited level-depletion target, fits on the other eight dates (including
later dates), guard-score exposure and unmatched quantity/inventory remain as
qualified in [the front-pull review](20261113_front_pull_integration_review.md).
Public at-best-maker observations and simulated fills also need common event,
age, level and coverage definitions before their swept shares are comparable.

## Consequence for the active experiment

Preserve the negative size/ladder result, without promoting larger orders or
claiming that all multi-level policies fail. Our frozen comparison stays at
its existing size and shared inventory limit. It must first show better
KEEP/PULL selection, recovery and risk in full sequential replay. The weak
[fixed recovery forecast](20261113_recovery_proxy_audit_review.md) and inconclusive
[local selective-rule comparison](20261113_recovery_local_review.md) remain
separate evidence that the prediction/policy gap is unresolved.

No other owner's report, source or data were edited. No new model or replay
variant, historical rerun, protected payload, live action or promotion.

Evidence identities are preserved in
[the source-review receipt](../reviews/model_recovery_20261113/front_pull_size_source_review.json).
Source introduction commit: `b8e7592a317ec85b7704e0b07d56312c2f4fa219`.
Report JSON SHA: `2f15449aeb87293cf12206c41f169cd986eef4244ebdd164e5cf95f98c44ceda`.
