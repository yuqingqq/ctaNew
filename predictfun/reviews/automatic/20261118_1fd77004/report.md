# Manual result review — 2026-11-18

Pinned target: **1fd77004be70f436119c9eaf8079a94fe322099f**.
Range reviewed: de084f8af0d48c4346f453033874450a851e0711..1fd77004be70f436119c9eaf8079a94fe322099f,
including 0529a820d1d0472a17587f3dd112b65b625accc1 and
d0088e90632a3220eb3dbd9a0096dd8f4349ef3d. Source was inspected in the detached
checkout `/tmp/predictfun-review-1fd77004`; other-session edits were preserved.

**The previous two findings are fixed. The saved results reproduce, but the
execution-evidence gate fails: the timing windows do not fit known order
lifecycles, and the score's measured advantage changes sign across scenarios.**
Two P2 diagnostic issues remain below. Neither establishes a new arithmetic
error in the saved 15-scenario economic table.

## What the results establish

Consumed **Oct 23**, scenario receipt as of **2026-11-18T07:23:56.848599Z**:
30,466 counted states, comprising 28,799 eligible, 1,289 post-only refusals,
354 missing endpoints and 24 stale future books. The 30,088 states with paths
were replayed, including orders without baseline fills. Baseline has 17,670
fill rows. Values below are summed local END (+5 s) values in cents, not
settlement P&L, portfolio P&L, or profitability after all costs.

| Scenario | Score advantage over matched STATES null, E | KEEP + score |
|---|---:|---:|
| M50, activation 58 / cancel 37 / notice 100 ms | +34,948 | -174,961 |
| S1 earliest timeline, same action latencies | -11,098 | -296,755 |
| S2 earliest timeline, same action latencies | -2,848 | -336,357 |
| M250, same action latencies | -13,342 | -297,803 |
| M50, cancel latency 0 ms | +38,255 | -146,417 |
| M50, cancel latency 100 ms | +17,964 | -230,841 |

All 15 saved scenario rows, their day splits and readings reproduce exactly
from the saved order/fill tables. The pinned self-tests and saved-output hashes
pass. This supports the implementation's internal accounting, not the realism
of the execution assumptions. M0/M250 are trade-clock shifts; L0 is the
cancellation-latency scenario. These should not all be called "zero latency."

The ledger's conditional exact joint ranges still span zero (S1 at 37 ms:
[-91,533, +113,454] c). The saved ledger labels all components exact with no
relaxed component; this is exact for its declared fixed-M50-fill/identity
problem, not a bound over the queue replay. Under evidence-only timing,
99.5% of baseline removed gross value remains UNKNOWN. The broader scenario
replay discovers and loses fills, but tests 15 selected, mostly one-factor
timelines, not all admissible joint timelines.

Known-order evidence, **consumed Nov 8**, as of **2026-11-18T07:21:12.820661Z**:
99 legs from 86 orders; **93 legs have research anchors and six do not**.
S1 misses 12 of those 93 lifecycles and calls three known fills avoidable;
S2 misses six and calls one avoidable. A constant offset cannot reconcile
either run. The miss counts and offset feasibility reproduce from the pinned
inputs. The evaluation population should be named as 93 anchored legs,
with six unknowns, rather than describing 99 tested lifecycles.

The experiment evaluates the existing fair-edge A+0 rule under execution
assumptions. It does not isolate learned-model quality. The engine can be
internally consistent while its timing inputs remain unsuitable for economic
labels. These results justify repairing that evidence before model tuning.

## Findings, ranked by severity

### P2 — K5 is not a valid test of the declared effect-time ranges

`predictfun/research/pf_keep_pull_exec_windows.py:722–726` tests only whether
the p95 acknowledgment-derived upper bounds fit the ranges' upper endpoints.
It ignores the activation lower endpoint and does not distinguish a bound
that contradicts a range from one that merely cannot certify it.

- **Missed lower-end contradiction:** among the 86 distinct known orders, one
  acceptance acknowledgment is at +21.165 ms. Under the module's own premise
  that activation precedes acknowledgment, that order cannot activate in the
  declared [25,100] ms range. The headline calls activation covered because its
  p95 upper bound is below 100 ms. A synthetic positive control with acceptance
  at +40 ms and a known-bad case at +1 ms both read `COVERED` when cancellation
  acknowledgment is +20 ms. The checker fails to reject the known-bad range.
- **Upper-bound insufficiency is not a measured slow effect:** a request at
  0 ms and acknowledgment at 5,033 ms permits an effect at either 20 ms or
  2,000 ms. The observations alone do not establish which side of 100 ms the
  cancellation effect occupies. Thus K5 does not prove a five-second venue
  cancellation latency or falsify the cancellation range. It shows that these
  acknowledgment bounds cannot certify that range. HANDOFF's statement that
  the declared cancel range "does not hold" is too strong.
- **The sampling unit is a leg:** K5 currently counts 99 activation records and
  88 cancellation records. Deduplicating `(run,cid)` gives 86 / 77 orders;
  cancellation acknowledgment-bound p95 remains about 5.031 s. The
  interpretation changes even though this sample's rounded tail number does
  not. It remains an outcome-selected, filled-order sample.

Requested correction: retain intervals, check both endpoints, report
contradicted / compatible-but-unidentified / certified coverage separately,
and compute order-path statistics once per order. Keep actual effect time,
acknowledgment receipt, retries/timeouts and unknown clock components distinct.
Do not replace the simulator latency with the five-second acknowledgment tail
without evidence of when cancellation actually took effect.

### P2 — The contradiction headline attributes proxy disagreements to REST evidence

`pf_keep_pull_exec_windows.py:540` includes both `contradictory_evidence` and
`contradictory_legs` (`st >= 2`), while `:651–654` labels their combined totals
"window contradicted by the REST evidence." The same attribution appears in
the committed HANDOFF/FINDINGS. Those are different causes.

Recomputed from the saved Oct 23 ledger (17,670 fills, 50,553 candidate events):

| All candidates have this status | S1 fill rows | S2 fill rows |
|---|---:|---:|
| Either contradiction type | 748 | 129 |
| Explicit evidence-cap contradiction (`st == 2`) | 37 | 0 |
| Leg-window intersection contradiction (`st == 3`) | 708 | 129 |

The remaining three S1 all-contradictory fills mix the two status types across
their candidates. Among events, S1 has 79 evidence-cap contradictions and
1,087 leg-intersection contradictions; S2 has **zero** evidence-cap
contradictions and 225 leg-intersection contradictions. A leg intersection
can already be empty before the REST cap is applied. These counts classify
the stored status, not an assertion that the categories exhaust all possible
underlying causes.

Requested correction: name the total "contradictory timing assumptions" and
retain the separate cause/status split at event and fill level. Investigate
WS/E2 linkage, clock alignment and the one-instant event assumption where leg
windows disagree. Do not treat all 748 cases as independent REST evidence
refuting their match time. The separate K4 known-lifecycle failures remain
valid evidence of a timing problem regardless of this headline correction.

## Prior findings and remaining limits

- **Resolved:** the multi-event contradiction omission. All/some/none/unlinked
  statuses and gross amounts now reconcile to the full saved fill population.
- **Resolved:** independent candidate timings labelled as exact. The old
  three-fill counterexample now produces joint [0,10]; forcing the cap gives
  [-10,20] explicitly labelled relaxed. No relaxed component is reported for
  the saved day. The full real-data bound solver was not rerun in this review.
- **Implemented:** a queue-aware scenario replay of the neutral states with
  paths, including initially unfilled and post-only refused states; queue,
  fills, remaining size, activation, cancellation and notice eligibility are
  recomputed. All 30,088 baseline state matches and 816 sampled kernel checks
  are recorded in the hashed day receipt. This review recomputed the saved
  economic report, not the full tape replay.
- **Implemented:** bound types, basis, representative-time rules and a checked
  column dictionary; K1 is a consistency check, K4 is explicit, and the known
  verdict disclaims venue-engine acceptance. The zero-latency structural tie
  correction matches the unchanged kernel convention in the synthetic checks.
- **Still open:** real execution-window calibration, venue/book/client clock
  alignment, live queue-rule validation, and full portfolio/repost/inventory
  evaluation. Contradictory events are held at M50 in the S1/S2 sensitivity
  timelines; their total event denominator and economic contribution are not
  reported. The held counts (9,576 / 1,412) therefore do not quantify the
  uncertain share of the replay. Development already notes the missing
  denominator. Adding it is useful but does not resolve K4.

## Verification and limitations

Read AGENTS, the binding CLAUDE reliability rules, HANDOFF/STATUS/FINDINGS, the
[prior review](../20261118_c23e6f29/report.md), and the execution-window
addendum. Reviewed the new source, driver, mutation definitions, committed
logs, receipts, labels and populations. Saved source hashes match the pinned
checkout; saved output hashes match the committed receipts.

Through **PF_SLOT=codex** and the absolute `run_light.sh` path:

- [verify.py](verify.py): both pinned complete synthetic self-tests, prior
  counterexample, saved ledger reconciliation, all 15 scenario metric rows,
  K0–K5 on existing known-case tables, and a corrupted-output hash refusal.
  Passed in 15.27 s, peak 573 MiB. The first review-harness attempt correctly
  received `ValueError` from the corrupted-hash control but caught only
  `SystemExit`; the harness was corrected and rerun. No implementation change.
- [check_diagnostics.py](check_diagnostics.py): saved contradiction-cause split,
  unique-order activation bounds, and K5 positive/known-bad examples. Passed
  in 2.23 s, peak 260 MiB.

Evidence: [receipt](receipt.json), [diagnostics](diagnostics.json),
[recomputed report](scenario_recompute.txt), [self-tests](selftests.txt),
[failed lifecycle cases](known_misses.csv), [run records](runs.txt).
The case CSV records order IDs and relative offsets; multiple rows can belong
to one order and must be resolved to the original clock-case leg identities
before a raw-event diagnosis. Existing 52/52 and 19/19 mutation logs were
reviewed; those suites were not rerun.

No full tape experiment, training, new capture, live action or untouched
validation day was used. One consumed day is insufficient for intervals or
forward validation; Nov 8 is a consumed partial-day diagnostic sample, not
validation of Oct 23's failure frequency. No corrected strategy profitability
claim follows from passing implementation controls.

## One concrete next priority

**Reconcile the failed known-order timing cases to raw event identities and
clock semantics before expanding the scenario run or tuning a model.**
Deliver one versioned case ledger covering the 93 anchored legs plus six
unknowns, with each of the 12 S1 failures traced to its original trade/order
identity, source status, receipt clock, and lifecycle bounds. Carry distinct
causes for proxy/leg disagreement, evidence conflict and unresolved linkage;
carry cancellation/activation intervals per unique order and correct the K5
interpretation above. The acceptance gate is no false certified avoidance on
known fills, with unsupported cases retained as unknown/contradictory rather
than moved into a convenient window. Additional capture is needed only where
the existing observations cannot resolve the intended claim.

Review artifacts and review-owned state additions only; another session owns
implementation. Scheduled reviews remain removed. Nothing was pushed or merged.
