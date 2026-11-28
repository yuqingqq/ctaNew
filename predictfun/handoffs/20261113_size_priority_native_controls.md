# Larger-order native construction: controls and remaining work

As of 2026-11-13 08:48 UTC, the quantity/reservation and causal observation
components have passed their synthetic cases. Full native residual forks and
the value-label component are implemented, but their runtime verification is
still pending. No historical larger-order data, fit or performance result exists.
The main recovery experiment is unchanged and continues on core8.

## What ran

- Original quantity suite: **8/8 passed**, 08:32:49.817599 UTC, rc0, 187 MiB,
  0.99 seconds. Receipt `size_priority_policy_controls.json`.
- First native suite: **20 passed / 3 errors of23**, 08:33:03.927226 UTC.
  Ten installed-reservation and nine ledger cases passed, as did the source
  instrumentation check. Three native execution cases refused a generation
  outside the declared window. The fixture omitted the existing G5/S240 stop
  rule while supplying book frames beyond240s. The ledger correctly refused
  those extra orders. Original fixture, log and receipt remain preserved.
- Corrected native fixture plus reset tests: **30 passed / 6 errors of36**,
  2026-11-13T08:37:20.415424+00:00; receipt
  `size_priority_native_controls_v2.json`. The corrected five native cases
  passed, including direct reproduction of the old missing-stop-rule failure,
  exact observed/unobserved orders/fills/generations/independent-accounting
  equality, client-notice timing and exhausted-generation retention. All six
  pure residual-reset protocol cases also passed. The six fork cases refused
  an unmatched source anchor before any fork ran: the generated drain function
  adds four spaces to the original probe block.
- Queued attempt v3 was superseded before a test worker started. A pure source
  compilation check found a second indentation mismatch: the fill-notification
  callback still assumed the flat event-loop indentation. Its actual
  `IndentationError`, bound sources and queued-launch termination are preserved
  in `size_priority_native_controls_v3.json` and its log. This was a confirmed
  code error, not a restart after a wait timeout.

The correction at **52b3d8f9** handles the probe block's nested indentation and
matches the complete fill-notification statement, preserving its actual indent.
Generated M and L engines both compile. It retains the original native loop,
complete-state cloning, heap reordering after pruning, actual cancel/place
clocks and the original failed source versions. **38 runtime cases are queued**
under launcher **3820900**, after the other owner's active `run_keep_place.sh`
chain (3771985) releases core9. The declared source manifest is
`size_priority_native_controls_v4_started.json`; do not modify its bound files
or launch a duplicate worker. No passing full-fork result is claimed yet.

## Residual intervention and target

`pf_size_priority_reset.py` cancels one known partial order, waits until its
actual cancellation effect plus the last possible fill-notification delay,
and derives replacement quantity from the original order's current notified
fills. Its replacement has the same still-valid price and never exceeds the
neutral policy's available capacity. Price/data/budget/minimum-notional failures
and exhaustion during cancellation are explicit terminal protocol statuses;
the neutral continuation resumes and these actions remain in the target
population. No latent remaining quantity or queue-ahead state selects actions.

`pf_size_priority_forks_v3.source` and the inherited hook in
`pf_size_priority_forks.py` fork KEEP/RESET on the same full native event state.
They reconcile KEEP against the neutral path, both branches' cash/inventory,
all captured fill quantities and the unavoidable pre-cancel prefix. They retain
actual within-clock event order and check pathwise maximum inventory against
the explicit profile. The pending cases include a fill during cancellation and
a trade exactly at the decision timestamp: that trade belongs to the physical
prefix, while its delayed notification must not leak into decision features.
These are implementation intentions until the runtime tests pass.

The label component, **61fb8eea**, is `pf_size_priority_target.py`. It computes
KEEP-minus-RESET continuation utility in cents from every fill's own price and
one/five-second future marks, minus the explicit endpoint inventory-square
penalty. Missing marks produce statuses, never a partial `nansum` value. It
also reports full inventory-square time and peak inventory using the native
fill order, including equal-clock fills. Aborted resets remain eligible when
their marks are complete; the population is not restricted to successful
reposts. This value includes cancellation interruption and later policy effects;
it is not a pure queue bonus. Seven analytic/corruption tests are committed in
`tests/test_size_priority_target.py` but **not yet run or queued**. Syntax passes.

## Next

1. Inspect the actual38-case native result; resolve any failures while retaining
   prior receipts. Then run the seven target controls and connect the target to
   successful native synthetic forks, with hand-reconciled economics.
2. Complete the neutral larger-order builder and saved-capture reconstruction,
   chronological fitting, risk-matched comparators and full replay/reporting.
   Freeze the complete pipeline, sampling/nulls and candidate count before any
   historical size experiment. Existing five-share fits are not size50 models.
3. Keep the eighteen-run recovery experiment running unchanged. Oct26/X and
   Oct27/X are complete; Oct31/X is active. No multi-day improvement verdict,
   untouched-day validation, promotion or live change.
