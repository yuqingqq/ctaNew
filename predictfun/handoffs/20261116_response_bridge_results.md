# Richer response forecasts for coherent order valuation

The richer response forecasts improve simulated fill markout, but the resulting
controller churns more, quotes less and earns less total simulated P&L. This
candidate does **not** replace the audited SOFT_JOINT baseline. The new heads
are useful research components; the forecast-to-value mapping remains unsolved.

V2 adds absolute liquidity scale consistently to the new forecast inputs after
a V1 omission was caught. Both versions and all completed/partial artifacts
are preserved. The completed V2 ledger/action audit passes. Results are consumed
development evidence, with no confidence interval or promotion claim.

## What changed

The previous fifteen public-flow/price classes omitted economically relevant
differences inside each category. Four new public response distributions use
the existing point-in-time data:

* Quantity reaching the focal price during37–100ms,37–250ms and37–1000ms.
  Each has seven categories: zero,0–10,10–30,30–60,60–120,120–300 and>300
  shares (upper endpoints included).
* Side-signed midpoint movement to1s, with boundaries−8,−4,−2,−0.5,+0.5,+2,
  +4,+8 cents. This adds severity inside the previous open-ended tails.

Volume is queried at the current best and1,3,5 ticks behind it; early volume
is queried at best. Inputs are the same71 causal market features, plus the
query price's depth, signed edge and causal W30 absolute liquidity scale. Bid/ask orientation and the side indicator
let the same pooled learner behave differently on the two sides. Own queue,
remaining size and order age stay in the execution/value calculation, rather
than being inputs to a forecast of public incoming flow.

The three quantity heads train on all357,041 eligible neutral action rows from
Oct23–25. The price head uses56,110 at-best rung0 actions to avoid repeating
the same endpoint across price depths. Zero-flow actions remain included.
Pinned LGBM parameters and200 rounds are unchanged. Oct26 sets temperature on
the inherited six-value grid. There is one fixed bin/depth design and two
response versions; the first omitted the absolute liquidity scale and is
superseded by V2. Both versions count in multiplicity. No bin, threshold or
parameter search was performed against economic results.

## Mapping into economic values

The new marginals and existing coarse joint forecasts reweight the same paired
training tape/book paths. The execution kernel allocates each path's public
volume once across competing orders, preserving price/time priority. It then
values whole-portfolio branch cash, inventory and common endpoint wealth.
No independent per-order fill head can duplicate the same incoming trade.

SOFT_JOINT is the existing audited mapping. RICH_JOINT adds learned response
constraints. RICH_EMPIRICAL adds historical side/phase/depth response-frequency
constraints to the same learned coarse model, isolating the incremental use of
market features in the new heads. All use the same soft-spread prior and
24 damped projection cycles. Unsupported mass and residual constraint errors
are reported; the extra marginals can conflict on a finite scenario bank.
The generic marginal residual averages two constraints for SOFT_JOINT versus
sixteen for RICH_JOINT. Those averages are not directly comparable: the result
index separately measures error against the same two coarse forecasts.

The controller,30-share six-rung ladder, cap180, risk coefficient1/360,
58ms placement,37ms cancellation,100ms notification,50ms review and1s endpoint
are unchanged. No cooldown, minimum rest, new threshold or size search.

## Evaluation contract

Recreate causal client states for the exact archived neutral post/wait and
keep/cancel identities on Oct26/27. Reuse canonical labels only by checked
identity/hash and reproduce the archived SOFT_JOINT predictions. Labels include
no-fill actions and share historical/unavoidable prefixes; they value the
action versus baseline with no later placements. They do not include the full
future learned cancellation policy. Report statuses, actual-value calibration,
quantity error, ranks and200 decision-count-matched null draws.

Replay all12 preselected Oct27 markets for their full five-minute contracts,
with raw fill and portfolio reconciliation. Report markout, initial edge,
terminal marked P&L, passive settlement P&L, quoting, inventory and churn.
Oct26 is a calibration diagnostic; Oct27 was already consumed for research.
Neither is untouched validation. No confidence interval, deployment or
general profitability claim; protected dates remain unopened.

A separate future-signature diagnostic, committed before economic outcomes,
supplies actual future response categories only to a retrospective calculation.
It compares common action identities with the prior true-coarse-category and
actual-future-path diagnostics. This separates forecast error from finite-bank
representation error; it is not a usable strategy or achievable upper bound.

## V2 forecast diagnostics

As of2026-11-16T16:02:23Z, consumed Oct27 action population. Lower categorical
logloss is better. Historical frequencies are matched on side, phase and depth;
the price head uses at-best rung0 actions. These are distribution diagnostics,
not order-value or strategy acceptance metrics.

| Target | n | Learned logloss | Historical-frequency logloss |
|---|---:|---:|---:|
| 100ms volume | 118,062 | 0.18771 | 0.21176 |
| 250ms volume | 118,062 | 0.40808 | 0.46241 |
| 1s volume | 118,062 | 0.84576 | 0.93183 |
| 1s price severity | 18,132 | 1.43809 | 1.67642 |

All four heads beat their baseline on logloss and Brier score. Scale restoration
slightly improves these forecast metrics, rather than explaining the entire
economic gap. Training scale range is13.255–6061.390 shares; Oct27 has45 of
118,062 eligible quantity rows above that range and none below. This is a
scale-coverage check, not proof of complete state or tail-event coverage.

## V2 neutral economic action tests

As of2026-11-16T16:06:19Z (Oct26) and16:10:14Z (Oct27). All418/443 selected
actions are supported by all three arms; source candidate statuses remain in
the reports. Baseline predictions and causal reference replays reproduce exactly.
Lower RMSE and higher rank are better; values are cents per whole action.

| Date / action | n | SOFT RMSE | RICH V2 RMSE | SOFT rank | RICH V2 rank |
|---|---:|---:|---:|---:|---:|
| 1026 / cancel | 205 | 8.449 | 8.285 | 0.117 | 0.064 |
| 1026 / post | 213 | 36.111 | 35.495 | 0.067 | 0.100 |
| 1027 / cancel | 221 | 31.826 | 31.636 | 0.339 | 0.335 |
| 1027 / post | 222 | 24.829 | 25.063 | 0.224 | 0.159 |

The richer forecast is not a uniform action-value improvement. Both dates show
a small cancellation RMSE reduction, but cancellation ranking is weaker. Entry
ranking improves only on Oct26; Oct27 entry error and ranking worsen. The
scale correction changes little in these economic metrics. The head-level
forecast gains therefore cannot be used to declare the controller improved.

## V2 matched continuous replay

As of2026-11-16T16:32:20Z,12 preselected Oct27 markets, full five-minute
contracts;24 strategy/market records including the exact archived baseline.
These are gross simulated totals. No untouched-day, live or full-day claim.

| Metric | SOFT_JOINT | RICH_JOINT V2 |
|---|---:|---:|
| Terminal marked P&L, USD | 367.66 | 187.83 |
| Passive settlement P&L, USD | 353.96 | 172.42 |
| Initial edge, cents/share | 1.917 | 1.701 |
| 1s markout, cents/share | -0.961 | 0.020 |
| 5s markout, cents/share | 2.650 | 3.366 |
| Filled shares | 7,895.73 | 5,556.47 |
| Orders | 10,847 | 15,040 |
| Pulls | 9,775 | 14,247 |
| Both sides quoting, % | 32.857 | 18.345 |
| Any side quoting, % | 77.613 | 60.849 |
| First-review model cancellations | 3,838 | 8,168 |
| All model cancellations | 9,762 | 14,231 |
| Median cancellation age after activation, ms | 92 | 42 |
| Mean absolute ending inventory, shares | 71.80 | 75.66 |

First-review cancellation fractions rise39.32%->57.40%. Pull count increases
and both-sided quote time decreases in all12 markets. Terminal/settlement P&L
improve only4/12; pooled differences are−$179.82/−$181.55. Maximum inventory
remains within180 (176.99->177.44), while ending absolute inventory worsens.
Better per-filled-share markout comes with lower trading volume and much more
quote turnover. The replay therefore does not establish an improved controller.

## Projection and future-information diagnostic

On the same443 Oct27 neutral actions, mean error against the **same two coarse
joint forecasts** increases0.09105->0.20675 when the richer marginals are added.
Oct26 increases0.07182->0.18607 (418 actions). The generic residual averaged
over all16 constraints hides this loss of coarse consistency. Extra constraints
can conflict on the sparse path bank; cyclic projection does not preserve the
original joint event distribution. These values are in the computed result index.

Retrospective diagnostic asof2026-11-16T16:34:00Z:443 actions,423 supported by
true fine marginals and20 explicitly unsupported. Common support across all
five compared arms is416 actions (207 cancellation,209 posting).

| Information supplied | Cancel RMSE, c/action | Post RMSE, c/action |
|---|---:|---:|
| Ordinary coarse forecasts | 32.736 | 25.539 |
| Ordinary richer forecasts | 32.518 | 25.734 |
| True coarse class pair, conditional prior | 26.011 | 24.143 |
| True fine categories, current projection | 28.930 | 37.132 |
| Actual tape/book path, same client order state | 2.032 | 0.377 |

True finer inputs still leave average projection residual about0.30. This
does **not** show that richer information is harmful or that its value is
unlearnable. The bank/projection cannot faithfully represent those inputs;
conditional responses remain approximate. The actual-path control supports
internal executor/target alignment, not external simulator realism. Future
information is absent from every learned model and continuous policy run.

The coarse-preservation check completed at16:39:48Z on the same416 common
actions. Of443 total,23 lack the true coarse pair and4 more lack a true fine
marginal. Before conditioning, the fine projection assigns only71.35% average
posterior mass to the known coarse pair (mean coarse L1 error0.35594). Hard
conditioning preserves that pair and reduces cancellation RMSE28.930->25.582c,
but posting RMSE only37.132->36.496c, still worse than24.143c using the true
coarse pair alone. Incorrect coarse weighting is one problem; the conditional
within-category scenario response remains poor even after that is removed.
Known-bad value, action identity and coarse-information relaxation controls
are refused. This remains a retrospective diagnostic, not a tested policy fix.

## Decision and next implementation boundary

Keep SOFT_JOINT as the comparison controller. Preserve the useful richer
forecasts and corrected scale input, but do not promote their current projection.

The next change should make the joint conditional response coherent: finer
information should refine compatible events, with unsupported combinations
explicit, rather than quietly distorting the coarse joint event probabilities.
Order values also need adequate resolution around each order's queue threshold,
arrival/activation timing and associated price/book evolution. Merely predicting
more marginal buckets or switching LGBM family does not address the observed
response failure. Entry versus newly resting value consistency and queue
continuation remain separate requirements; no fixed retry timer was introduced.

The completed action/ledger audit, inherited canonical execution controls,
1,929 donor-label reconciliations, scale-separability control and diagnostic
identity/falsifier checks pass. Single Codex/core8 process throughout; peak RSS
1,277MiB, full replay1,325.98s. No research job remains running.

Active evidence index: `reviews/response_bridge_20261116_v2/latest.json`.
It links the main audited results, the separate coarse-preservation diagnostic
and the V1 supersession receipt. The first draft of that last diagnostic had a
receipt-path typo caught before execution; its unused freeze is preserved and
the corrected source/freeze supersede it explicitly.

## Remaining structural limits

The original sparse1,929-path bank and approximate transplantation of donor
futures remain. Marginal distributions still do not identify full event paths,
exact quantities inside bins, or tail severity beyond the last boundary.
Early public labels begin37ms after decision; the execution kernel, rather than
the label, applies58ms post activation. Client queue/unreceived-fill uncertainty,
estimated trade timestamps and zero continuation value beyond1s remain.
Neutral generation sampling versus repeated controller queries also remains a
population-transport limitation; adding scale does not establish full coverage.

V2 source5e67d068; freezeb5206f6a. Reproduction:
`bash predictfun/research/run_response_bridge_v2.sh` after creating both immutable
freezes. Output is under ignored `data/predict_fun/research/response_bridge_20261116_v2/`.
Existing output directories are refused; do not overwrite frozen artifacts.

## Preserved V1 correction

V1 sourcef71c34f6/freeze69ef2372 trained four heads and completed both action
tests, but omitted the W30 scale while predicting absolute shares from market
features normalized by that scale. Equal relative book states at10 and1000
shares were indistinguishable in those inputs. V2 adds the same causal scale
in training and query vectors; explicit separability and invalid-scale controls
pass. No other target/controller settings changed. The five-day input audit
at15:54:30Z found no missing/nonpositive scale in612,128 eligible actions.

V1 action reports asof15:46:26Z/15:50:19Z are preserved. Against SOFT_JOINT,
Oct26 cancellation/posting RMSE changed8.449/36.111 to8.252/35.420c
(n205/213); Oct27 changed31.826/24.829 to31.479/25.145c (n221/222).
Cancellation rank worsened on both days; entry rank improved only on Oct26.
Thus more detailed categories were not a uniform economic improvement.
The unfinished V1 pilot was terminated, rc143, with no completed replay receipt
or final P&L artifact. Its log remains; it is not a completed strategy test.
