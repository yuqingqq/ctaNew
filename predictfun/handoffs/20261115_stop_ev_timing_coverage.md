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

The run is sequential on the Codex slot with an external 2 GiB RSS guard.
Results will be recorded after all declared days complete. Do not substitute
partial-day counts for the final comparison.
