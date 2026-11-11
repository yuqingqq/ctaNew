# Maker comparison: the missing decision target

Review as of **2026-11-11 14:22 UTC**. Sources: committed B12–B18 artifacts through `ea2f107a`, their builders, the frozen v3 strategy and guard specification; the ongoing B16c design at `5deb1103` was inspected separately. This is a review of existing evidence, not a new backtest. Protected Nov 4–6 / Nov 9–13 prices and outcomes were not read.

**The model concern is well founded, but the strongest structural issue is the decision it estimates: a generic order at the back of the queue, rather than the incremental value of keeping our actual resting order.** Horizon choice, information processing and cancellation execution are separate issues. We do not know the top makers' private targets or exact cancel decisions.

## What the comparison actually establishes

| Question | Evidence | Implication |
|---|---|---|
| Do we throw away resting priority? | B13c: 12,240 at-best first-minute pulls in 220 Nov 8 windows, median observed life 23.17 ms; 82.49% replaced at the same price within one second. B13d corrected the earlier guard-clock attribution: brief pull-line crossings are the principal trigger. | There is substantial churn in the **old live build**. It is not enough to improve the price predictor while applying the same immediate entry/exit threshold. |
| Are imbalance and flow missing? | The live model's 32 features already include `bn_imb`, `bn_flow250/1000/5000`, `bn_ntr1000`, `pf_imb1/3`, depth, price changes and pending-flow fields. Explicit G1/G2 guards use adverse Binance **price returns**; they are not dedicated burst/imbalance triggers. | Pressure before a price move must get through the learned score or the existing book rules. There is no evidence that simply adding the same raw imbalance feature again solves this. |
| Does the new target represent our earned queue position? | B16 assigns a five-share order to the back of the displayed queue on every grid row. Its feature list lacks our order's age, queue progress, remaining quantity and cancel status. On 438,951 checked sweep rows, the replacement label equals the old label. | Stage 1 does not test the main keep-versus-cancel distinction. An old order near the front and a hypothetical new order at the back receive the same market-only inputs. |
| Do pressure features help the right objective? | B16 qb improves one-second replay markouts but has lower sixty-second and settlement totals, below. B14 did not improve its declared quote-decision criterion. | Overall predictive fit and short markouts are insufficient promotion criteria. The evidence suggests an objective mismatch; it does not prove horizon choice alone caused the result. |
| Do top makers simply avoid pressure? | B18: 62.94% of top-five filled shares lie in its union of pressure classes. Their markouts there are +0.043 c/share at 1 s and −0.150 at 60 s, versus +0.759 / +0.627 outside. | They still get filled during pressure. These **fill-conditional** statistics do not measure their cancellation rate or resting exposure, so neither “they always pull” nor “they do not pull more” follows. |

B13c's population is one consumed partial live day; it is not current-build performance. The recorded native handoff says incremental sizing and narrow soft pull are already implemented on main `0c3b115`; that handoff is not a verification of Tokyo deployment or the new live behavior. Current native files were not inspected in this review. Preserve the distinction between implemented mechanics and measured results.

B12g/B18 cover the nine consumed replay dates, Oct 23–27, Oct 31, Nov 1–3. “Top five” means ranked by that day's filled volume, not five identities selected in advance or ranked by independently observed profitability. Their 0–240 s sample contains **24,411,430.78 shares**. B18's underlying tape population contains 5,185,723 timed maker legs across all groups; it reports no missing feature-row lookups. That zero does not establish complete inputs or precise match times. The loader's excluded untimed legs/unresolved markets and stale midpoint ages are not completely enumerated in B12g, so its values remain qualified descriptive comparisons.

## The target mismatch, in numbers

From B16's saved daily CSVs and report, qb minus the current guard, **nine consumed dates, 2,592 markets per arm/timing**; J and X are the existing alternative fill-timing assumptions, not independent experiments:

| Metric | J | X |
|---|---:|---:|
| One-second fill markout value | +$71.97 | +$18.55 |
| Sixty-second fill markout value | −$66.61 | −$191.41 |
| Settlement P&L | −$148.08 | −$182.88 |
| Filled events, reference → qb | 5,278 → 5,662 | 5,025 → 5,302 |

Results were produced around 13:37 UTC Nov 11. These columns are alternative valuations, **not additive P&L components**. Longer-horizon differences are noisy and their reported day intervals include zero. The one-second J gain occurs on all nine dates; the X gain on seven. A higher total one-second markout partly coincides with more trading; assess per-share value and matched exposure as well.

B12g's public top-five fills have +0.309 / +0.138 / +0.163 c/share at 1 s / 60 s / settlement across the nine dates (24.41M shares, report at 13:58 UTC). This describes the value of their observed maker fills. It excludes their full taker activity, hedges, inventory path, rewards and private costs. It is not a reconstruction of their strategy P&L or a label revealing their objective.

Two earlier interpretations need qualification:

1. **“The extra fills cause the long-horizon loss” is not isolated by B16.** Changing the guard changes the whole order/inventory path. The comparison does not split shared fills, added fills, removed fills and changed quantities. A common-fill decomposition is needed before attributing the loss to only the added trades.
2. **“Not inventory because mean absolute terminal inventory is similar” is too strong.** Signed inventory, its covariance with subsequent moves, its duration and the path to the stop can differ despite similar mean absolute inventory. B16's terminal-mark/carry decomposition is informative, but the scalar absolute-inventory statistic cannot rule this out.

The combined 1 s / 5 s / 60 s target is already being tested by the other research session as **B16c** (`5deb1103`). No B16c outcomes were inspected here. It addresses one horizon concern, but retains the generic back-of-queue label and lacks actual order state. Its equal-weight average is an experimental objective, not a measured distribution of our holding/exit times.

**Important evaluation caveat:** B16c calls R190 a “tape” referee, but its `qlm` label is built using B16's simulated order and rule-M fills. Public-tape inputs do not make the label independent of the fill model. Moving the primary score from the full replay to that grid cannot, by itself, remove queue/fill-model bias.

## What we can and cannot infer about maker behavior

Public snapshots expose aggregate levels and executed makers; they do not identify each maker's outstanding orders and cancellations. Repeated fills of the same maker can mean a persistent order, replenishment or another order from that maker. Absence of another fill can mean cancellation, exhaustion or no incoming trade. A queue-reach proxy cannot exclude a queue mechanism, nor prove an exact priority rank.

This matches the current API distinction: the [public book stream](https://dev.predict.fun/websocket/topics/order-book) contains aggregate levels and can combine multiple changes into a single update, while [wallet events](https://dev.predict.fun/websocket/how-tos/follow-your-orders) describe the authenticated wallet's own orders. This limits both event ordering and attempts to infer other makers' cancels.

The defensible finding is that **top makers receive less damaging flow within broad pressure/guard classes**, while our old build churns and our public-fill comparison includes poor late-cancel fills. Queue position, order size, quote depth, information, execution latency and selection can all contribute. We cannot identify their objective or copy their hidden cancellation policy from this tape.

B12b also found a separate execution problem on Nov 8: 11 first-minute fills (65 shares, eight windows) on orders with slow cancel confirmations account for the selected guard-OFF loss, with requests already sent before the estimated match. Match times are uncertain and three legs dominate the loss. This is evidence to inspect cancel-request → venue removal → fill-match/notice clocks, not evidence that a more accurate prediction alone would have prevented those fills.

## Issues to correct before treating the pressure classes as pull signals

These are source-review findings against frozen B18, not newly measured historical effects. Keep its old outputs as provenance and issue a superseding definition/report for changes.

- **“2× activity” is not twice the arithmetic mean.** B13a computes `log1p(ntr_now) - rolling_mean(log1p(ntr))`; B18 thresholds it at `log(2)`. This compares with a geometric scale of `1+ntr`. A bursty history can trigger it when current activity is below its arithmetic mean. Either describe the feature accurately or define and test the intended arithmetic-rate ratio separately.
- **“Thinning” does not establish cancellations at our price.** B18 compares current best-level size with the best-level size five rows earlier, without requiring the same price. The levels may differ; even at the same price, shrinkage includes executions as well as cancellations. Off-touch/rung fills are classified using the best level, not necessarily the resting order's price. Track same-price level identity and separate trade consumption from unexplained depletion, with an unknown status where timing is ambiguous.
- **Unknown inputs become false flags.** Comparisons with NaN yield `False`, after which `isfinite(bool)` is always true. A valid feature-row lookup therefore does not imply all class inputs are available. Preserve class-specific unknown statuses and quote coverage before comparing rates. Historical impact is not quantified by this review.
- **Filled-share fraction is not exposure or avoidance.** `P(pressure | filled volume)` depends on offered size, fill intensity, depth, queue position and pull behavior. Comparing it with an unweighted fraction of clock time does not identify pulling. The new model should be evaluated on neutral action opportunities, not trained to copy labels inferred only from completed fills.
- **Tape versus replay is not a matched fill-model test.** B18's public makers and our selective simulated strategy have different populations. Their different markouts motivate calibration but alone do not establish its cause. B1's same-own-order comparison is the appropriate evidence. Changes evaluated under one biased fill model can have biased **differences** too.

## Proposed adjustment to the research direction

Keep the short-horizon pressure estimate, but make its decision target specific to **our existing order and inventory**:

> How much more value do we expect by keeping this order, compared with cancelling it and following the same subsequent policy?

For a new order, compare posting with waiting. For a resting order, compare keeping with cancelling/reposting. These are different alternatives. A cancellation has latency and can still fill before removal; a replacement loses priority and may be rejected or delayed. Treating pulling as an immediate zero-payoff escape is wrong when those effects matter.

The next useful experiment is an **order-state decision layer**, with the current market/pressure inputs held fixed:

1. **Build action rows from a neutral reference path.** Retain every eligible order generation/action, including unfilled orders and exclusions. Use actual receipt/decision clocks. Record accepted age, current price/depth, remaining size, queue-ahead bounds, trades since entry, cancel status, recent partial fills, signed inventory and all potentially filling outstanding orders. Do not credit every displayed shrink as a known cancellation ahead.
2. **Compare keep/cancel on the same state and tape with effective latencies.** Before the cancel can take effect, both branches retain the exposure. Continue a fixed subsequent policy after the one changed action so that preserving priority has a future value. De-duplicate overlapping decision rows or keep them within their generation/day cluster; do not sum overlapping potential gains as realizable P&L.
3. **Use one coherent value objective.** For a fixed evaluation stop, use cash change plus marked inventory change, minus execution costs, with inventory risk reported separately and a predeclared penalty only if used for selection. Keep 1 s / 5 s / 60 s markouts as diagnostics. Adding several markouts to the same cash P&L would count the same economics repeatedly. Model unconditional expected fill value on all reference actions, not only observed fills.
4. **Allow asymmetric post/keep decisions.** A narrow soft pull already moves in this direction. The general rule should value earned queue position when information is weak and react to a strong adverse signal when that outweighs continuation value. A thin side or increased trade count alone is not sufficient; direction, depletion, expected post-fill value, order location and cancellation latency matter together. Keep hard risk/staleness overrides.
5. **Require evidence at the decision and portfolio levels.** Compare against the current model, the existing soft pull, and random pulls matched on side/time/action count/exposure. Report avoided bad value, forfeited good value, fill count, spread, short and long markouts, signed inventory path, square-inventory time and stop/settlement decomposition. Show shared/added/removed fills. Freeze candidates and test later untouched complete days; the consumed dates are exploratory. Actual own-fill calibration remains necessary before promotion.

This should distinguish a missing signal from a mispriced action. A longer target alone cannot make the model aware that a resting order has acquired a valuable place in the queue. None of the findings justifies a blanket “pull on every pressure flag” rule, a wider retention band, or a deployment change today. Maker rebates remain separate from trading P&L and do not replenish the test loss budget.

## Evidence and verification status

- Read source and saved artifacts for B12b/d/g, B13a/c/d, B14, B16 and B18. **All eight existing artifact SHA256SUMS manifests checked successfully** on the isolated `research/predict-maker-comparison-20261111` branch.
- Existing numeric results are cited above with their populations and original report times. No new model, strategy replay, raw-data population or statistical test was run for this review.
- An additional arithmetic/synthetic audit helper is committed at `b6a44a72`: `predictfun/research/pf_maker_comparison_audit.py`. Its launch was refused because the single research slot was occupied by B16c. **It has not run**, and no `maker_comparison_20261111/audit.json` receipt is claimed. Run it from the isolated worktree after B16c finishes; do not interrupt that pipeline or overwrite its outputs.
- Preserved original studies and declarations. This review qualifies their interpretation; it does not silently revise frozen receipts. B16c outcomes remain outside this review.

Key existing reports: [churn diagnosis](20261111_b13d_churn_diagnosis.md), [queue/cancel comparison](20261111_b12b_reason.md), [maker horizons](20261111_b12g_horizons.md), [B16 target experiment](20261111_b16_queue_guard.md), [pressure comparison](20261111_b18_pressure_fills.md), [current soft-pull handoff](20261111_soft_pull_port.md).
