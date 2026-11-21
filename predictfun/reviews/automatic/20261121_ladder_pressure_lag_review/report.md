# PRESSURE-LEAD-SENSITIVITY-v1 — combined independent review

2026-11-21, Codex. **Target: 6b6f977bac5eee954033c7d9326506832d935b31**.
Range: `909c671e9411746b6900f5bcd7b03b21e9ba16ff..6b6f977bac5eee954033c7d9326506832d935b31`.
Reviewed freeze `b19b7ca0e26670696315e52d860e9ecb91f718f4`, checker supersession
`9cc6161ac5335586b21844a33e650f101f4b7491`, and the pinned results. Source was
read and imported from a detached checkout; implementation was not changed.

**Accept the timing sensitivity as a consumed-day diagnostic. No blocking
finding or required rerun.** The +50 ms guard retains a material part of the
improvement and passes the predeclared screen. Both prior findings are
addressed. This does not establish profitability or out-of-sample performance.

Population, as of the receipt at 2026-11-21 19:58Z: **19 complete markets**,
7/5/5/2 on Oct 24/25/26/27. The original 96 selected markets retain 54
missing-endpoint and 23 stale-future-book statuses. All four days are consumed
adaptive development data; neither the original signal choice nor this
timing-margin exercise is untouched validation. No day intervals.

## What the evidence establishes

| Metric | Baseline ladder | Original pressure | Pressure +50 ms |
|---|---:|---:|---:|
| END marked value, cents | −209,557.709 | −86,109.607 | −122,383.003 |
| Gain over baseline, cents | — | +123,448.101 | +87,174.706 |
| Filled shares | 96,991.1 | 79,399.5 | 83,182.2 |
| Eligible +1 s markout, cents | −203,083.922 | −85,164.277 | −109,560.079 |
| +1 s per valid filled share, cents | −2.102 | −1.077 | −1.324 |
| Mean exact both-side active time | 40.70% | 40.19% | 41.39% |

The delayed arm improves END and +1 s/share versus baseline on each of the
four days. Eighteen markets improve and one worsens. It retains **70.62% of
the END improvement** and **75.93% of the per-share improvement**. Those are
ratios of improvements over baseline, not a fraction of profitable trading
or a win rate. The pooled per-share result uses each arm's own eligible
quantity denominator, not an average of daily ratios.

Timing still matters: retained END gains are **113.94%, 77.58%, 37.76%,
48.45%** by day. Relative to the original pressure arm, the delay costs
**36,273.395 c** in total. The additive decomposition is **−24,395.801 c**
through eligible +1 s markout, **−12,500.022 c** from those fills' later
movement to END, and **+622.428 c** in late/missing-horizon contributions.
These are accounting contributions, not causal attribution to individual
late cancels. Ten markets improve versus the original pressure arm, nine
worsen. Delaying the signal changes admissions, clearing, subsequent queue
positions and inventory, so the replay difference is not a monotone physical
latency-cost curve.

The lagged local screen reproduces: **110 selected / 4,831 uncensored live
states**, 62 movable strata with 103 selected; 500 matched draws. Selected
mean **−90.2 c**, null p05 **−57.9 c**, empirical percentile **0.2%**. Bids
alone are at 37.2%, asks at 0.2%. Thus the declared pooled screen passes,
with stronger evidence on asks. It remains a local baseline-state screen,
not a null for full policy P&L or an entry-action selection test.

The delay does not recreate the old near-universal first-review cancellation:
**33 / 4,523** reviewed generations receive a pressure pull at their first
review. There are 472 pressure pulls and 1,402 withheld slot proposals.
Both-side activity is slightly higher than baseline in the pooled table;
fewer filled shares and changed paths still matter. Uptime is not synonymous
with submitted size, fill rate, or the economic value of exposure.

## Ranked consolidated findings

1. **P3 — NONBLOCKING; defer or bundle cheaply: the exported re-entry rows
   have a narrower checker than its header suggests.**
   `lag_checks_v2.py:20` describes reconciliation of `reentry.csv`, but
   `:291–302` verifies membership and `pull_gid` only. `:134–149,236–241`
   recomputes arm-level pair summaries from the ledger, not the other CSV
   fields. Adding 1 to every exported `next_price` still passes H; this
   targeted in-memory control is saved in `evidence.json`.
   **Verified workaround:** every field of all **905 current re-entry rows**
   was compared to the executed order/queue records through the inspected
   `reentry_pairs` transformation; all match. This does not affect any
   headline economic reading. If the next round uses the exported pairs,
   add field comparisons and a changed-price/gid refusal; otherwise narrow
   the checker claim and retain this limitation. Do not launch a repair-only
   cycle or replay the current experiment for it.

No other result-invalidating issue was found in the affected signal, runner,
selection, measurement and acceptance boundaries. Routine hardening is not
a prerequisite for the next bounded experiment.

## Prior findings and code review

**Prior finding 1 — closed for the requested participation/inventory fields.**
`lag_checks_v2.py:93–131` independently rebuilds exact order-active interval
unions/intersections and fill-based inventory integrals. `:240–241` compares
those arm fields, and `:268–272` checks their reading deltas. The coordinated
+25 pp uptime corruption and the changed-inventory case now fail. Independent
event sweeps in this review also agree on all **57 arm ledgers**.

**Prior finding 2 — closed as a diagnostic-label correction.**
`pf_ladder_pressure_lag.py:52–54,220–248` labels birth ahead as activation-clock
initialization, exports both gids/prices/rungs/times/refusal status, and calls
the comparison the next same-side submission. The report explicitly excludes
replacement/counterfactual-queue-cost interpretations. The original pressure
arm's 430 pairs include 57 same-price and 156 same-rung pairs, matching the
prior review. The −102.5 same-price ahead difference is a market-mean
descriptive statistic; it is not evidence that canceling improved queue
position relative to keeping the order. Raw `ahead0` remains an inherited
submission-stamped field, correctly explained by the new measurement layer.

**Lag and warm-up.** `LaggedPressureGuard.value` at `:76–83` computes
`round(t/50) − lag`, chooses that side's signal/status/age, and returns warm-up
before reading a negative index. Production instantiates only lag 0 and 1 at
`:163–164`. The parent signal/live/slot methods all dispatch through this
lookup; no path substitutes current tick k for delayed pressure. The source
timestamp and effective age in `:85–86` are consistent with k−1. Decisions
are exact grid ticks. Current structural/book/inventory state remains allowed
and unchanged: this delays the pressure input, not all market information.

The existing synthetic ON/OFF crossing, warm-up, side and future-read tests
pass. An additional **152 direct read-barrier controls** on saved caches
changed both sides' signal/status/age at tick k and later: value(k) remained
unchanged for lag 1. Changing the used k−1 value changed the result. All
intervention logs reconcile to source ticks/ages, actual commands, client
position known after +100 ms, and live pull effect at request +37 ms. Warm-up
is a policy status, not a blank reading of the last array element.

**Controls and identity.** `:99–109` binds parent day records to their
committed receipt copies and checks their gz/npz hashes, rule identity and
expected day-record hash. `:143–147` checks the new source closure and rule
before a run. Parent arrays are reused rather than rebuilt. Independently
verified those identities, the parent manifest hash, every source hash,
every receipt-copy hash, and current output hashes. The 19 zero-lag arms
reproduce parent ledger/fingerprint, orders, fills, requests, logs, queue
records and command arrays. Additional comparisons of decisions, cash,
mid, reservations and venue/client inventory arrays also agree. Common
marks, tape ids and episode identities agree across arms. The inherited
disabled arms reproduce the original ladder.

**Join and screen.** `:185–215` checks the zero-lag key alignment before
substituting the k−1 input. Reading k to validate an offline join does not
feed it into the lagged policy or selection mask. All **4,840** live-state
keys align, with zero off-grid/mismatch/warm-up/non-ok cases in this sampled
population. The same action-relative windows are retained after the current
cancel effect, not shifted back to the source tick. All selection fields,
including per-side diagnostics, censoring, quantities and draws, reproduce.

**Measurement and reading.** `:245–299` uses the same arm accounting and
common cutoff. Fills/mids lengths, fill times within [0, END], exact cash/
inventory, END identity and both lost/added pairs reconcile. `:255–274`
implements the requested retained-improvement formula and conjunction;
`lag_checks_v2.py:258–283` rebuilds these checks. The conjunction does not
require positive absolute value or positive gains relative to the zero-lag
arm; the receipt does not claim either. No new lag or threshold was selected.

**Checker supersession.** The v1/v2 diff is exactly the disclosed change
from `dict(PC.recompute(...))` to `collections.defaultdict(float, ...)`, plus
the supersession description, inclusion of v2 in committed-file checks and
new output suffixes. No acceptance threshold or population changed. V1's
pilot-rehearsal crash is preserved; v2 was committed before its run and
passes 20 checks on the pilot with full-membership false by design, then
all 21 on the full population. Source did not change in the results commit.
The saved first mutation escape and final 21/21 receipt were inspected;
the whole mutation queue was not redundantly rerun.

## Bounded verification

`reproduce.py`, through **PF_SLOT=codex** and the absolute `run_light.sh`,
passed the pinned selftest, **21 checks and 14 falsifiers**, all artifact/
parent/source hashes, 19 zero-lag comparisons, 57 ledgers and exact exposure
reconciliations, every intervention, all 905 re-entry rows, the join/screen
and 152 direct read-barrier controls. **22.49 user CPU seconds; peak 643 MiB.**
Checker source/data paths were relocated in the review harness only; no
development artifact was edited. No real-tape replay, model fit, label build,
new market or new day was run. Passing synthetic controls proves implementation
behavior, not economic skill. Evidence, logs and independent arm metrics are
beside this report.

Resource accounting from retained logs is in `resources.json`: development
receipts plus available review logs total about **1.540 core-hours**. The
current development round totals **0.25905** including its later checks/report,
consistent with 0.25481 at the saved checker. This is below its 0.50 ceiling.
The proposed 0.35-hour next round fits the original 2-hour allocation at this
recorded use; recheck cumulative logs before launch. The retained first resource
pass omitted the earliest review directory; the corrected pass includes it.

## Largest limitation and ONE next priority

**The economic blocker remains negative value on the exposure still taken.**
The +50 ms arm loses **1.324 c/share at +1 s** and **122,383 c at END**;
even the nominal-clock original loses 1.077 c/share. The experiment establishes
some timing margin in the adverse-movement signal, not that latency alone
explains the remaining loss. The opportunity population is small and consumed,
bid-only local separation is unproven, and large posted orders still assume
unchanged external flow. The lag test does not resolve that counterfactual
assumption or actual feed-tail latency.

**ONE targeted experiment: PRESSURE-ACTION-ATTRIBUTION-v1.** Is the useful
economic effect primarily from withholding new exposure, pulling existing
orders, or their interaction? Resolve that before changing thresholds,
adding features, or fitting another value model. Use one predeclared 2×2
ablation of the original nominal-clock pressure rule, not a lag search.

1. Define bits **C** = enable pressure cancellation of would-be KEEP live
   orders; **W** = enable pressure withholding of would-be POST slots.
   Four arms: `BASELINE_00`, `PULL_ONLY_10`, `WITHHOLD_ONLY_01`, `FULL_11`.
   Disabled components delegate exactly to LadderInventory. Structural and
   inventory pulls/restrictions always remain active, with their own reasons.
   A pull-only arm may submit during pressure; this is part of the declared
   simulation comparison.
2. Keep **all** other choices pinned: original pressure edges and current
   cached tick (no additional 50 ms), both sides, nominal 100 shares/two
   levels, 200/100/50 inventory settings, no timer, 58/37/100 ms clocks,
   M50 fills, same full paths and marks, same 19 markets and all 96 statuses.
   “No additional lag” still includes the declared feed/grid/execution
   latencies; it is not zero-latency trading. Do not stack the reservation
   guard or choose different thresholds for the two ablations.
3. Reuse hash-bound caches and baseline evidence. Rerun `FULL_11` and require
   equality to the accepted original pressure arm on every market. Prove
   `BASELINE_00` delegation with synthetic/control fixtures and retain the
   already reconciled baseline identity. The two new real arms are 10 and 01;
   record both in multiplicity before output. Test that each disabled
   component cannot intervene, that the other can, and that inventory and
   structural precedence, notice clocks and reservations remain unchanged.
4. Primary contrasts per market and UTC day are:
   - cancellation added to withholding: **R11 − R01**;
   - withholding added to cancellation: **R11 − R10**;
   - standalone effects: **R10 − R00**, **R01 − R00**;
   - interaction: **R11 − R10 − R01 + R00**.
   Compute these for additive END and eligible +1 s value. Never sum the
   standalone effects as the full effect without the interaction term.
   These compare complete feedback policies, not isolated per-order avoided
   losses. Present per-share value, quantities and denominators separately.
5. Retain side/rung results, exact uptime/inventory, END decomposition,
   good/bad markouts, reducing-side suppressions, first-review pulls,
   generations and lifetimes. Show quantities lost/gained with the declared
   fill-matching convention. Carry queue diagnostics with their corrected
   interpretation; bundle the minor CSV check only if those rows are used.
   Reuse the original 500-draw local screen as context, explicitly not an
   entry-action or whole-policy null. No inference that an ablation has
   predictive skill from a favorable P&L contrast alone.
6. Freeze readings before any new replay. For each incremental contrast,
   classify END and +1 s/share independently as positive on every day,
   negative on every day, or mixed; count better/worse markets and show
   concentration. Keep the no-improvement reference at **zero incremental
   value**. If all absolute values remain negative, say so regardless of
   relative gains. No interval, automated winner/promotion, new feature,
   threshold selection or additional candidate after seeing the result.
7. **At most 0.35 additional core-hours**, still within the original 2-hour
   cumulative limit. Pilot the same first two complete markets per date;
   continue only if the complete declared round fits. Existing slots and
   memory limits. A budget failure gets an explicit stop, not a smaller
   favorable population. One combined freeze/control/result receipt.

**No additional USER permission is needed for this bounded consumed-day
ablation.** More days, untouched validation, an increased resource allocation,
live execution or a new market-impact model require a separate USER decision.
