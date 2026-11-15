# Timing coverage with a reference-path certificate

The user authorized improving defensible timing coverage after the accounting
audit. New code is versioned separately: `pf_stop_ev_timing_certificate.py`,
`pf_stop_ev_timing_check.py`, `pf_stop_ev_timing_build.py`, and
`run_stop_ev_timing.sh`. Existing running model/data sources are unchanged.

## What the new filter proves

For the uncapped neutral reference only, opposite-side executions cannot alter
the focal side's orders. On the same side, an uncertain event is irrelevant only
if it cannot reach **any accepted order** anywhere in its possible interval
before the focal label horizon ends. Queue drain counts as an influence even
without an immediate fill. The test includes the history before the decision,
not merely the focal order's future one-second interval.

Before the first possible uncertain state change, reference states agree by
induction. Multiple events proved to be no-ops cannot jointly change that path.
This is a sufficient certificate; failure to certify does not prove a nominal
label is wrong. The implementation assumes no recovery after an uncertain event.
No empirical timing quantile is treated as a hard bound. Missing anchors remain
unbounded; known out-of-range events retain their source intervals. Every source
leg and exclusion still reconciles in the original ledger. A local label view
can disregard a missing event only with the explicit certificate attached.

Two declared variants:

- `certified_ws`: missing events and E2 timing changes cannot influence the
  relevant side prefix. E2 intervals are the existing declared +/-250 ms
  assumption, not proven physical bounds.
- `certified_estimated`: missing events cannot influence the prefix, conditional
  on the E2 point anchors. Its labels remain explicitly estimated-timing labels.

Neither variant proves actual venue fills. WS anchors themselves come from
matched settlement-pending events, not an independently verified execution
timestamp. Book data are treated as fixed external inputs. The certificate
does not authorize inventory/cross-side inputs, a cap, learned actions, or a
different replay engine; those require another dependency proof.

## Why future-only exclusion is insufficient

The other session's `ce6edfa7` change, authorized there as a broader training
population, excludes a missing leg only if it might reach the focal order in
`[t, t+1s]`. That is a conditional sensitivity rather than a path-invariance
guarantee. Synthetic counterexamples demonstrate:

1. A missing trade before `t` drains queue ahead; a later trade fills 5 shares
   instead of none, with identical observed book features.
2. A missing earlier full fill changes which order generation is resting at `t`.
3. A missing trade at a better price changes how a later leg is split across
   our rungs, even though that earlier trade cannot directly reach the focal price.

All nine synthetic checks passed at **2026-11-15T13:47:35Z**, including both
sides, inserted-event no-op controls, invalid provenance/cap refusals, and
retention of invalid-mark statuses. The first test attempt is retained: one
fixture compared array indices after inserting events; the corrected check
compares stable per-tranche executions/economics, not renumbered tape positions.

## Frozen historical comparison

Source `f0682872`, freeze commit `3bb6773f`; declaration at
`reviews/stop_ev_timing_20261115/freeze_v1.json`. Nine consumed dates only,
Oct 23–27/31 and Nov 1–3. Same neutral probes, 30 shares, 6 rungs, M50,
unknown queue zero, no cap, one-second exposure and fill+one-second marks.

Output namespace:
`data/predict_fun/research/stop_ev_timing_certified_20261115_v1/`.
Each day retains original strict/estimated statuses alongside the new labels,
raw input/source hashes, per-window missing-event identities and certificates.
Previously valid label values and order features must match exactly. A
future-only comparator measures additional admissions lacking a certificate;
that count is **not** an observed mislabel rate. There is no P&L-based selection,
model fitting, significance test, or forward-validation claim in this study.

## Completed comparison and saved-artifact audit

Build as of **2026-11-15T13:58:13Z**, saved-artifact audit as of
**2026-11-15T14:02:20Z**. All nine declared days completed, 2,586 windows and
393,154 unique neutral probes per variant. The development subset is Oct 23–26,
1,152 windows and 187,384 probes per variant. These are consumed data.

| Population / timing assumptions | Probes | Original eligible | Certified eligible | Coverage before → after |
|---|---:|---:|---:|---:|
| All nine days / strict interval invariance | 393,154 | 2,809 | 18,911 | 0.71% → 4.81% |
| All nine days / conditional E2 point anchors | 393,154 | 72,255 | 149,658 | 18.38% → 38.07% |
| Development / strict interval invariance | 187,384 | 969 | 7,337 | 0.52% → 3.92% |
| Development / conditional E2 point anchors | 187,384 | 30,076 | 68,028 | 16.05% → 36.30% |

The estimated-timing development population gains 37,952 labels (2.26x total);
the strict population gains 6,368 (7.57x total). These fractions measure label
coverage of scheduled probes, **not quote uptime or fill rate**.

For the same all-day population/as-of, bid has 195,440 probes and ask 197,714.
Strict certified labels are 9,590 bid / 9,321 ask; estimated-timing labels are
73,663 bid / 75,995 ask. Per-day/per-side counts and every exclusion status
are in `reviews/stop_ev_timing_20261115/audit_v2.json`.

| All nine days: status (n = 393,154 per variant) | Strict | Estimated |
|---|---:|---:|
| Eligible under the declared assumptions | 18,911 | 149,658 |
| Uncertain prior reference state | 245,673 | 119,517 |
| Uncertain future exposure | 5,984 | 1,393 |
| Not client-resting | 114,175 | 114,175 |
| Probe not reached | 3,766 | 3,766 |
| Incomplete book coverage | 4,639 | 4,639 |
| Missing/stale mark | 6 | 6 |

All **75,064 baseline-overlap comparisons** preserve label values and order
features exactly (2,809 strict + 72,255 estimated comparisons; not 75,064
distinct actions). Saved files pass unique-key, quantity/residual, component,
normalization, invalid-status nullness, certificate and source-count checks.
All five planted artifact corruptions are refused. Nine synthetic tests passed.
The extended v2 audit adds the declared side summaries; the passing v1 receipt
and both logs remain. Frozen sources and raw inputs match before/after every
day; saved artifact hashes match their receipts.

The sequential Codex-slot build completed in 360 seconds with exit code 0 and
an external 2 GiB RSS guard. Build report, runner result, all nine day manifests,
and audit receipts are versioned in `reviews/stop_ev_timing_20261115/`.

## What remains and how to use the artifacts

The future-only comparator admits 265,373 of the same 393,154 probes; **115,715
of those admissions lack the new prefix certificate** (development: 59,642 of
127,670 admissions among 187,384 probes). This is not an observed error rate.
The synthetic counterexamples establish that its rule cannot generally prove
the neutral queue, generation or value unchanged. The user-approved broader
model population should remain explicitly a sensitivity.

Missing execution times and uncertain prior queue state remain the main
coverage limit. The strict sample is still small and potentially selected;
both populations need representativeness checks before extrapolating model
results. Neither coverage improvement establishes prediction quality,
profitability, live fills, or physical timestamp accuracy. No model was fitted
or running model pipeline changed in this work.

Per-day Parquets contain all probes, statuses, baseline statuses, label scalar
columns, `x_*` known-order features, and source/certificate counts. Keep the
two variants separate. `certified_ws` uses `eligible_timing_invariant`;
`certified_estimated` uses `eligible_estimated_timing`. Existing evaluators
that only accept the original statuses need an explicit adapter with timing
provenance; do not silently relabel the strict variant as physical truth.
Certificate counts and exclusion statuses use outcome information and must
never become model inputs or live eligibility gates.

For the next model experiment, join market features at exactly
`(day, market, side, decision_ms)` and verify the known-book depth identity.
These Parquets do not include the other builder's `k` or `hour` fields: recover
`k` from the exact decision grid, derive UTC hour from the timestamp, and refuse
missing/duplicate/time-shifted matches. Compare the certified population with
the broader sensitivity under a newly declared experiment; do not overwrite
the already-running model's frozen inputs or claim untouched validation.

Reproduction uses an isolated checkout of builder commit `f0682872` and the
committed `freeze_v1.json` from `3bb6773f`, then `run_stop_ev_timing.sh` with a
new output directory. Its source-hash guard deliberately refuses a checkout
whose imported sources have since changed. Audit checker `1e6c71d2` reads the
saved artifacts only and can run through `run_light.sh` on the Codex slot.
