# STOP-EV accounting status after the historical build

As of 2026-11-15T13:37:11Z, no new primary quantity/value accounting defect was
found. The saved-artifact audit at `c677eef8` passed on nine consumed dates,
Oct 23–27/31 and Nov 1–3. Build receipts have latest as-of 13:00:52Z. Evidence:
`reviews/stop_ev_accounting_status_20261115/receipt_v1.json` and `audit_v1.log`.
No model fitting, raw-feed read, or reserved-day access was performed.

## What was verified

- All nine day receipts pass E1–E7/E9, use the same recorded source identities,
  and match the current committed builder/accounting sources.
- All 393,154 saved generation probes were checked in both timing variants.
  The estimated-timing variant has 72,255 eligible actions, matching the E9
  population exactly; the E9 receipts report zero value/quantity mismatches
  and zero ledger failures. Strict timing has 2,809 eligible actions.
- Saved actions have unique generation keys. Filled quantity does not exceed
  actual residual quantity after common exposure. Remaining shares, known-fill
  quantities, value components and per-known-share normalization reconcile.
- Excluded actions retain null labels; valid unfilled actions have zero value.
  Timing variants have identical action identities, probe times and causal
  features; strict eligible labels match their sensitivity counterparts.
- Positive controls pass. Planted duplicate actions, overused quantity,
  corrupted value, non-null invalid labels and changed source identities are
  all refused. Peak wrapper RSS was 204 MiB, under the 2 GiB guard.

## What remains open

**Timing coverage and selection.** On the four development dates there are
187,384 probes, but only 969 strict eligible actions and 30,076 eligible under
estimated timing. More estimated-timing data does not establish exact execution
times. The untimed-leg exclusion is deliberately conservative. Narrowing it to
the focal order's future fills alone is not justified: an earlier missing event
can change queue state, order birth, inventory and shared-volume allocation.
Any less conservative rule needs demonstrated bounds or path-invariance checks.

**Diagnostic tables have a different contract.** E7's single-order grid still
diverges on SHARED/ENDED states; E9 and the primary saved generation labels
handle these through the joint ledger. The grid also uses forward-filled mids
and frame-age checks (`SideCtx`), whereas primary labels use actual valid quote
timestamps. Do not substitute grid values for primary labels or treat their
approximate results as a clean primary-label performance measurement.

**Fill realism and raw identity.** Engine agreement does not establish venue
queue position or exact WS/E2 execution time. Equal-content trade flags remain
in the data report. The loader deduplicates by transaction/maker-order hash,
which explains how distinct equal clips can survive; this review did not
re-read raw maker identities to independently prove every flagged case.
Do not deduplicate trades merely because price/size/time fields match.

The remaining work is data coverage/identification and model validation; the
original double-counting, partial-fill, cancel-prefix and per-tranche value
defects are not reproduced in the checked primary artifacts. This is not a
profitability or forward-validation result.
