# Live fix for the port session: count a cancelled order for a 1 s late-fill window, not until the 5 s REST audit

USER 2026-11-09: "We can use the shorter window for phantom inventory." Research diagnosis only; the port session implements and
verifies. Evidence: `predictfun/research/pf_live_vs_replay.py` (`kernel`, `phantom`), Tokyo runs tokyo-6 `1794108465172` and
basis-tokyo-1 `1794147344737`, 220 complete windows of 2026-11-08 (data branch `data/tokyo-live-20261109` e41bb5c).

## Problem

- The kernel's NetCap (`[live.risk] net_shares = 150`) refused 4,430 places in tokyo-6 and 6,017 in the basis run's 11-08 windows
  (21% / 29% of places). It refused 24% (J) / 18% (X) of the shares the frozen v3 replay fills on the same windows.
- Cause: after a cancel is confirmed, the order keeps its full unfilled size in `risk::exposure` until the REST trade audit runs.
  - That audit starts `CANCEL_AUDIT_GRACE_MS = 5000` after the cancel (median 5.3 s from the cancel to `CancelAuditComplete`).
  - The fast per-order read (`CancelQuantityReconciled`) clears only 47%, at a median of about 0.8 s. Its budget is burst 4, then
    1 per second.
- With the strategy's ~50 ms place / cancel churn, these cancelled orders average about 140 of the ~160 shares counted at a refusal.
  Real net plus live orders is under 10 shares.
- Without them, 2 of 10,447 refusals would remain. The loss-budget refusals (116 / 29) count the same orders.

## Evidence for the window length

- 71 cancelled orders got their first fill notice after the cancel was confirmed (44 tokyo-6, 27 basis).
- All 71 arrived within 593 ms of the cancel confirmation (p50 about 320 ms, p90 about 510 ms).

Rebuilt refusals at each window length:

| Window W after the cancel is confirmed | NetCap refusals left (tokyo-6 / basis) | Late fills not covered |
|---|---|---|
| until the audit (today) | 3,968 / 5,577 rebuilt over the cap | 0 |
| 1,000 ms | 212 / 334 | 0 |
| 2,000 ms | 495 / 866 | 0 |

## Change requested

- New live config `cancel_late_fill_window_ms`, default 1000.
- A cancelled order counts its unseen remainder in `risk::exposure` (NetCap and the loss worst case) only until `W` ms after the
  cancel is confirmed. After that it counts only booked or owed quantity (`unbooked` debt), as after a reconciliation that found none.
- The REST cancel audit is unchanged and stays the backstop.
  - A fill it finds later is booked as today ("late fill after a cancel, booked, no halt").
  - Raise an alert when a fill notice arrives more than `W` after its cancel. The window should then be widened.
- Unchanged: `net_shares` 150, the loss budget, the fast-reconcile budget, and the strategy.

Code pointers, at 4d65884 / 3c724fe:
- `crates/oms/src/lib.rs`: `OrderRec::unbooked_exposure()` returns `remaining` while `cancel_audit_pending` has no matched bound.
- `crates/oms/src/derived.rs`: `Row::read` uses `unbooked_exposure()` for orders that are not live, so the token `buy_quantity` and
  cost include them.
- `crates/exec/src/service.rs`: `CancelPendingAudit` is emitted on the cancel confirmation; `CancelAuditComplete` after
  `CANCEL_AUDIT_GRACE_MS`.
- `crates/exec/src/service_reconciliation.rs`: the fast reconcile budget.
- `crates/risk/src/lib.rs`: `exposure`, `check_with_inputs` (NetCap, LossCap).

Implementation note: derived totals change only at mutation boundaries. The window needs its own event at confirmation + W (for
example an exec report or timer such as `LateFillWindowElapsed { order_id }`) that re-marks the order, so the totals drop at that
moment. Otherwise the order would keep counting until its next mutation.

## Verify

- Unit tests:
  - a cancelled order counts in NetCap before W and not after;
  - a fill notice after W is still booked by the audit path, and the alert fires;
  - a cancel reconciled before W behaves as today.
- Offline: rebuild 11-08 admission with the new rule. Expect NetCap refusals of about 212 / 334 (from 4,430 / 6,017).
- Live: watch NetCap and loss-budget refusals per window, and count fill notices later than W after a cancel (expected 0 on
  current evidence).

## Not this fix

The larger replay-vs-live volume gap is the replay crediting fills to orders that were not yet on the book. It is not a live problem;
see predictfun/STATUS.yml 2026-11-09.
