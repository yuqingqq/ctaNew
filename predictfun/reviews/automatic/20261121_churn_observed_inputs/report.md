# Churn: observed input changes, not a causal attribution

2026-11-21; evidence as of 09:33:35 UTC. Pinned R4E source/results commit:
**141e820e4b736c6a8c4f27b0a0dbf53789e95e89**. This is a follow-up diagnosis of
existing consumed Oct26–27 artifacts, not a review of the advancing
ENTRY-EXPOSURE development commits.

**Real input updates are present in almost every first-review cancellation.
That does not establish that the updates justify cancellation.** The earlier
observation that the mid-price usually did not move cannot support a claim
that the market information was unchanged. Similarly, switching model heads
is a code mechanism and a possible contributor, not a demonstrated dominant
cause of the churn.

## Evidence

All 12 committed part hashes and records match; 384 planned worlds contain
380 ok worlds and four explicit path exclusions. All 16,455 retained
generation-level first-policy-review records reconcile to the corresponding
POST and review decisions, and every saved churn CSV field matches the raw
parts and the CSV committed at the pinned SHA. Counts below are unweighted
generation events within repeated worlds, not independent samples, and no
interval is calculated. All counts are computed in [results.json](results.json).

| SMOOTH first-policy-review PULL records | Oct26 | Oct27 |
|---|---:|---:|
| PULL events / worlds containing them | 3,325 / 188 | 3,491 / 192 |
| Mid-price and spread both unchanged | 3,012 | 3,204 |
| Top-level imbalance changed | 2,845 | 3,049 |
| At least one recorded market field changed | 3,319 | 3,486 |
| Recorded side-adjusted fair edge increased | 1,915 | 2,156 |
| Recorded fair edge decreased | 1,358 | 1,304 |
| All seven non-clock recorded fields unchanged | 6 | 5 |

The recorded market group is mid, spread, top-level imbalance and fair edge.
The seven non-clock fields additionally include order edge, estimated queue
ahead and known remaining quantity. Equality means exact equality of the
recorded float32 inputs; tiny changes count. This is deliberately not a
post-hoc materiality threshold or an economic screen. An increase in fair
edge is an improvement in that **estimated** component, not realized profit.
Fair-edge updates can also reflect the fair model's time evolution rather
than a new adverse external event.

Thus there are two simultaneous observations: book/fair inputs usually
update, but the recorded fair-edge change alone points in a favorable
direction for more than half of SMOOTH's pulled orders on each day. Other
signals may still imply adverse selection. Neither “the market changed” nor
“the mid did not move” explains the economic correctness of a PULL.

The same issue exists within PI0's own POST/RETAIN heads: its first-review
PULL counts are 223 / 266, with a recorded market-field change in all of
them. This does not establish that either those changes or the model-head
transition caused the action. Source: `pf_r2_teacher.py:407–423` and
`pf_adv2.py:318–328`; values use different POST and retention targets even
though both refer to the original END.

## What remains unidentified and how to complete the diagnosis

The saved churn records retain **eight fields**, including the clock, from
the 107-input vector. Full pre/post vectors, the PI0 moment decomposition
and all contemporaneous order-flow inputs were not retained for every
first-review pair (`pf_adv2_replay.py:63,235–254`;
`pf_adv2_entry_replay.py:172–184`). Decision/fill ledgers suffice for the
previous accounting review; they do not suffice for this new question.

The next diagnostic integration should retain the actual 107 inputs at
submission and first policy review, model identities/support status and
the teacher's P, expected quantity, expected signed movement, edge and V,
alongside the actual order/generation and decision. Separate:

1. Market/fair/order-flow changes.
2. Queue, age, known fills, reservations and opposite-side state changes.
3. The estimator and target transition from POST to RETAIN.

Replay actual decisions first. Any fixed-market feature-block rescoring is
a **sensitivity diagnostic**, not causal proof: rebuild dependent features
consistently, report out-of-support combinations, and do not apply a RETAIN
head to an absent-order row and call it a valid counterfactual. Test a
synthetic unchanged-market transition to check whether the transition alone
can reproduce a reversal. Establish economic benefit with matched KEEP/PULL
outcomes, not score changes alone.

**The accepted next priority remains ENTRY-EXPOSURE-v1 and its combined
review.** That experiment asks whether avoiding the early discretionary
pulls improves value. The input-attribution detail belongs in the next
relevant bounded diagnostic; do not interrupt or amend the frozen run,
launch a broad replay, add a timer, or require a separate cleanup review.
No assertion here that short-term predictions are reliable or that keeping
orders longer will improve performance.

## Verification and limitations

[inspect.py](inspect.py) uses only saved JSON/gzip/CSV artifacts and the
pinned Git receipt. The unchanged/change/missing classifications have
synthetic positive and known-bad controls. All executable checks ran through
PF_SLOT=codex and the absolute run_light.sh/script paths. The two successful
passes took 1.31 / 1.30 s, peak 121 MiB; the latter also records computed
fractions and the fair-edge-majority predicate. No fit, real tape simulation
or label build.

Three stopped reviewer attempts are retained: the first two exposed a
checker assumption that the first engine live tick must equal the first
policy callback; an engine no-book fallback can precede the callback. The
checker now excludes the explicitly logged engine fallback, with a synthetic
control. One SMOOTH Oct26 callback is consequently 150 ms after posting;
the other 3,324 are 100 ms. The third attempt looked for the CSV in the part
hash manifest; its actual identity is the committed compressed CSV, now
verified directly. A shell redirection typo also stopped before launching
the checker. These were reviewer-instrument issues; no source/data repair
was needed. Logs: [initial](check.log), [detail](check_diagnostic.log),
[second checker](check_v2.log), [passing check](check_v3.log),
[final check with computed fractions](check_v4.log).
