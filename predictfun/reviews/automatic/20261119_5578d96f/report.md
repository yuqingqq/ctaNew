# Latest result review — 2026-11-19

Reviewed result commit **5578d96f684d472b04ed25325295a9a2a6989b2e** and its saved
Oct23 six-rule/19-scenario experiment. Working HEAD was
`c2717dc86fc637548c8e90275227cabeb99a476b`; the seven declared implementation
sources were matched to the reviewed commit and the published manifest before
execution. Existing development edits were preserved. No implementation changed.

**The numerical results reproduce. One reporting issue needs correction; no new
blocking numerical/accounting defect was established in this review.** These
are simple fair-price rules, not a test of the new learned pipeline.

## Finding, ranked by severity

**P2 — the PULL-ALL column mixes an improvement with KEEP's absolute value.**
`pf_keep_pull_variants.py:234` labels adjacent columns `KEEP` and `PULL-ALL`, and
line 235 prints `KEEP` beside the raw `PULLALL` improvement. The CSV similarly
exports `PULLALL` at line 218 without an absolute `V_pullall`. The published
`variants_20261023.txt:5` consequently presents -416,243 and +389,554 side by side
at B, although PULL-ALL's absolute local value is **-26,688.8971 cents**.

The actual comparisons at `pf_keep_pull_variants.py:102` are correct:
`score - PULLALL` equals the difference of the two policies' total values because
their KEEP baseline is common. This is a misleading output label, not a reversal
of the candidate rankings. Publish explicit `pull_all_improvement_cents` and
`pull_all_value_cents = KEEP + PULLALL` in a superseding receipt/table. Never read
the positive improvement as positive profit. The independent checker includes
both a positive control and a planted delta-as-total error it must refuse.

## What the result says

Population: **30,466 neutral order opportunities**, 30,088 with paths; at baseline
28,799 eligible, 1,289 refused post-only, 354 missing endpoint, 24 stale future
book. One consumed complete day, **2026-10-23**, 288 markets. Result as-of
**2026-11-19T02:49:13.598345Z**; scenario inputs as-of
**2026-11-18T20:28:18.245546Z**. Six declared rules; no fitted predictor; no
untouched validation or day-cluster confidence interval.

At baseline M50/A58/L37/N100, local END marks in cents:

| Rule | Own local value | Excess over its matched STATES expectation |
|---|---:|---:|
| KEEP | -416,243 | — |
| A+0: pull below zero fair edge | -174,961 | +34,948 |
| A+1: pull below +1 cent fair edge | -153,379 | +29,847 |
| B1: pull after a 1 cent adverse fair move | -244,914 | +102,767 |
| PULL-ALL at first live decision | -26,689 | — |
| Never post, no exposure | 0 | Mechanical control only; provides no quotes |

- **A larger E does not imply a better strategy value.** B1 beats its own
  participation-matched expectation by more than A+0, but cancels only 3,420
  orders versus 14,155 and leaves a much worse local value. E has a different
  matched control for each policy. It is useful selection evidence within that
  comparison, not a common absolute objective across policies.
- **Threshold tuning has not established a profitable or stable rule.** A+1
  improves total value over A+0 in 18/19 scenarios, but no candidate improves it
  in all 19. Every rule has negative E at M250; all six have negative total
  value in 17/19 scenarios. The 19 assumptions are not independent days or
  measured scenario probabilities.
- **The opening/retention trade-off matters.** PULL-ALL beats each score rule's
  local value in 17/19 scenarios, while its own value remains negative at
  M0/M50/M250 (-18,863 / -26,689 / -37,083 cents). Thus blanket entry followed by
  fast cancellation is still loss-making in these clock scenarios. It does not
  follow that every individual order has negative value, or that no feature can
  identify good orders. Entry selection and the remaining value of an existing
  order need separate, aligned estimates. A new fixed cooldown does not follow
  from this experiment.
- **This does not isolate a failure of LightGBM, the richer features, or all
  market information.** The tested rules have no learned quantity/severity
  predictor. M0 is a trade-clock assumption, not zero end-to-end latency.
  Even the L0 cancellation sensitivity is not a zero-delay data/feed experiment.

## Checks and limitations

[Verifier](verify.py), [receipt](receipt.json), [synthetic selftest](selftest.txt).

- Published text/CSV hashes, input reports, day receipts and source hashes match.
- The variants module's synthetic selftest passes, including its planted bad
  inputs. Existing mutation-suite logs were inspected; those suites were not
  rerun. No synthetic success is interpreted as profitability.
- All **114 policy/scenario rows**, comparison tables and computed readings
  reproduce from the saved JSON. Existing V0a–V0e checks pass.
- Independently reconstruct **18 policy/clock cells** (six policies at
  M0/M50/M250), with separate first-trigger logic, direct per-fill removals and
  explicit per-step live risk-set expectations. KEEP, improvement, null, E,
  PULL-ALL, pull counts and eligible populations agree within 1e-6 cents/exact
  counts. The null here is an analytic expectation, not an undersampled Monte
  Carlo maximum or a significance test.
- Codex slot: **21.80 s, 679 MiB**, exit 0. No raw tape, model fit, full tape
  replay or untouched-day access. This bounded review verifies the experiment's
  accounting and interpretation, not the venue's fill realism.

Previous readiness findings remain resolved within their tested scope. Remaining
limits are unchanged: timing premises, unvalidated order-level queue/fills,
cancel refusals, no sequential repost/inventory controller or quote-availability
measurement, and local END marks rather than self-financing portfolio or
settlement P&L. The new contract at c2717dc8 passes an interface gate; it is not
a trained strategy and this result does not evaluate it.

## One next priority

Build the **corrected neutral-action feature/label exporter and a chronological
economic prediction benchmark**, covering POST versus no-entry and KEEP versus
latency-delayed PULL separately. Reuse the richer causal book/flow/order inputs,
keep no-fill rows and exclusion statuses, and require kernel parity. Include
fair-edge, mechanical, direct-value, never-post and first-live-pull controls;
judge calibrated gains/loss amounts and net value at matched participation,
before tuning or connecting a full sequential controller. This is the same next
step as [the pipeline plan](../../../handoffs/20261119_prediction_pipeline.md),
now supported by the score-variant result. Preserve timing sensitivity and do
not promote a candidate from these consumed-day comparisons.
