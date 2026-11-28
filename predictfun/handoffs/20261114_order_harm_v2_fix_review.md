# Review of the ORDER-HARM and JOINT-LADDER v2 repairs

As of 2026-11-14 around 06:40 UTC. Reviewed ORDER-HARM v2 `abf5638e`, results
`12f9b183`, and JOINT-LADDER v2 `3040edf2`. This supersedes the open status of
the three specific ORDER-HARM v1 defects in the earlier critical review.

## Repairs verified

- **Queue-feature look-ahead:** the new `queue_known` uses the client-known
  clock, does not use the simulator's raised `a0`, and preserves unknown
  activation depth as NaN. The earlier synthetic case now gives identical
  features when only the future book changes. A second case with a venue
  frame stamped before the decision but received after it also passes.
  Reintroducing either next-frame raising or venue-time selection fails.
- **Overlapping economics:** R622 and its summed-value verdict are removed
  from the report. Remaining state-weighted ranking/calibration diagnostics
  are explicitly distinguished from sequential policy earnings. The rows
  still overlap, so they are not independent orders or additive action values.
- **Window boundary:** the builder now rejects primary horizons crossing the
  reference lane's end and bounds the pull fork by that end. A decision at
  58 seconds is eligible for a two-second label; one at 59 seconds is censored.
  Five-second continuation labels get a separate censor rule. This addresses
  the earlier unequal-end example.

Saved report as of **06:29:41 UTC**, nine consumed dates (Oct 23–27/31,
Nov 1–3), first-minute windows: 2,287,617 retained decision rows, 64,182 primary
censored states, 110,575 additional five-second censored states, 56,205
no-order states and zero missing-input states. The nine row receipts reconcile
to 2,408,004 considered states. They report 114,383 future-size perturbation
checks with zero differences and zero reference-path differences. This tests
the nine order inputs conditional on the chosen reference order; it is not
an end-to-end causality certificate for every upstream feature or decision.

## New critical finding: future rejection releases hard-cap capacity early

In `pf_joint_ladder_v2.py:102`, `place()` looks at the venue book at the future
activation time to determine whether an order will be rejected. That is an
execution outcome. Only accepted orders are appended to `ST[s]['allo']`
(line 116). `believed()` (line 124) then uses that list for the next placement's
capacity check, even at the same submission timestamp.

Thus a rejection not yet observable by the client immediately releases its
reservation. This lets future information change an earlier placement.

The synthetic full-lane counterexample has a 30-share cap and 30-share orders:

| Identical known book at t=0; only t=40 ms differs | Bid at 50c | Bid at 49c submitted at t=0? |
|---|---|---|
| Future opposite best remains 52c | Will be accepted | No: capacity reserved |
| Future opposite best falls to 50c | Will be rejected | Yes: capacity already treated as free |

Both runs have identical information at t=0. The actual `sim_joint` function
produces the differing second-order submissions; **all seven existing violation
counters remain zero**. Those counters check accounting, not causal admission.
The demonstrated issue may improve apparent queue position by placing the
second order before the first rejection is known. Its historical frequency
and economic effect have not been measured; no historical result is assigned
a numerical correction by this review.

Fix: reserve capacity for every submitted request independently of its eventual
accept/reject outcome. Update the client-known ledger only when the acceptance,
rejection, fill or cancellation notification becomes available. Keep latent
venue execution state separate. Add a whole-lane prefix-invariance control:
altering events not known by t must not change submissions/cancels at or before
t. Include future rejection as a planted error, with a known-information
change as a positive control.

Two other limits remain: `maxexp_hard` currently records maximum filled
position (`maxpos`), not position plus executable orders; and settlement marks
still inherit the previously documented tie conversion. The hard admission
check is an improvement, but its receipt should distinguish reserved exposure
from realized inventory and measure both.

## What the new model result supports

The saved AUC 0.753 and DELTA rank correlation 0.151 describe predictive
ranking of the specified historical state labels. The best predicted DELTA
decile still averages -0.06c per original order share. This does not prove
adaptive keeping is unprofitable: KEEP in the target does not reconsider its
cancel decision later, whereas the proposed policy would. Independent rungs,
overlapping states and leave-one-day-out training on consumed dates remain
declared limitations. Joint, sequential and chronological evaluation is still
needed before a model/policy improvement claim.

## Verification and execution

The reviewer ran only tiny extracted-source synthetic cases and receipt/hash
checks, not another model fit or historical replay. The new instrument is
`research/model_order_harm_v2_fix_review.py`; its immutable output is
`reviews/model_order_harm_v2_fix_review_20261114/review.json`.
The saved report text/JSON match their SHA256SUMS and runtime copies.

ORDER-HARM v2 completed at 06:29:50 UTC. At 06:39 UTC JOINT-LADDER v2 had
finished its three parity dates and two historical evaluation dates; the third
was running. No frozen scientific source or active worker was changed.
