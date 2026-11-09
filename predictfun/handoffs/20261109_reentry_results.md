# Pull and re-entry study: delays do not preserve capture coverage

This study addresses B3's pull/repost behavior, separate from A2 size adjustment.
None of the tested re-entry policies achieved fewer requests with unchanged or
better coverage at the original intended prices. Keep these policies off pending
evidence of an economic benefit. This is a negative engineering result, not proof
that all forms of hysteresis or model stabilization fail.

Producer as-of **2026-11-09 14:59:08 UTC**, independently verified **14:59:57 UTC**.
Population: **220 previously consumed Nov8 windows**, first 60 seconds, four lanes,
**279,059 recorded decisions** and **30,796 reference quote opportunities**.
Zero excluded windows. Reference and candidates all use the same incremental
sizing policy from the previous study. No fills or PnL are simulated.

## Paired results

All percentages compare with the same reference, including every original quote
opportunity. Coverage is active time at the exact original desired price, not the
fraction of the candidate's smaller selected opportunity set.

| Policy | Requests | Request change | Aligned quote-time change | Original desired time retained | Off-target quote-time change |
|---|---:|---:|---:|---:|---:|
| Reference | 65,211 | — | — | 100.00% | — |
| Wait for prior same-price cancel acknowledgement | 65,067 | −0.22% | −0.14% | 99.71% | +0.12% |
| Rapid re-entry: confirm 25 ms | 59,697 | −8.46% | −8.19% | 89.86% | −2.46% |
| Rapid re-entry: confirm 50 ms | 56,215 | −13.80% | −11.19% | 86.81% | −10.71% |
| Rapid re-entry: confirm 100 ms | 52,504 | −19.49% | −15.84% | 81.96% | −17.61% |
| Confirm every entry 50 ms, cost control | 37,074 | −43.15% | −27.06% | 66.04% | −39.71% |

The tradeoff has the same sign in both recorded runs. For the 25 ms rule,
v6 (122 windows) has requests −9.65% and aligned time −8.83%; basis (98 windows)
has −7.18% and −7.47%. Request reductions are not a demonstrated improvement in
capture. Reduced off-target time could matter economically, but this experiment
does not value that tradeoff or establish lower inventory variance.

The mean oldest active quote rises from 515 ms to 536/550/567 ms with the three
re-entry delays, even as coverage falls. Delaying every entry raises it to 643 ms
while losing 27.06% of aligned time. Longer observed quote age alone can reward
skipping short opportunities; it does not establish better queue access.

## What was tested

- A rapid re-entry means the recorded lane was pulled and becomes eligible again
  at the same price within 100 ms. First entries, longer-gap entries and new-price
  entries remain immediate, except in the explicit all-entry control.
- Confirmation requires continuous recorded eligibility at the same price for
  the specified minimum elapsed time. A size change does not restart the timer.
  A pull resets it and immediately requests cancellation. A reprice immediately
  cancels old-price orders; it never preserves an obsolete quote to save requests.
- Cancel-ack gating waits only for an outstanding same-lane, same-price cancel
  acknowledgement to arrive. It does not wait out the subsequent one-second
  reservation grace, and does not use the future remaining latency to decide.
- All decisions occur at the original recorded decision events. There is no new
  independent timer. Delay quantiles, as-of **15:04:07 UTC**, among delayed admitted
  opportunities: 25 ms minimum gives median **46.93 ms** (n=7,792); 50 ms gives
  **75.84 ms** (n=5,976); 100 ms gives **125.15 ms** (n=4,101). These conditional
  timing summaries explain the mechanism; they are not performance estimates.
- Order model unchanged: 40 ms placement, 22 ms accepted-order cancellation,
  95 ms pending-create cancellation, one-second cancellation reservation grace,
  150-share side cap, three pending creates per side and $1 minimum notional.

Reference opportunities are maximal intervals with a nonempty lane at the same
price. They are independent of whether a candidate posts or fills, and their
identities/endpoints are identical across all arms. Of 30,796 opportunities,
26,152 end in a pull, 4,547 in a reprice, and 97 are censored at minute end.
There are 11,218 rapid re-entries, of which 11,185 complete before the boundary.
Among all 30,699 completed opportunities, median lifetime is 100 ms and 3,348
last less than 40 ms. These are lane-intent intervals, not actual order lifetimes
or independently observed prediction-only decisions.

## Interpretation and next work

Waiting to repost does not keep an order's priority across the original pull.
The old order is already being cancelled, and a new order joins later. Under
this latency model, waiting only for cancellation acknowledgement has a small
aggregate effect. Broader confirmation removes requests but also a material
amount of original coverage. Capacity and in-flight refusals fall; that saving
does not recover the lost quoting time in this workload.

The remaining strategy question is whether some **pulls themselves** should
change. A symmetric/exit-only band or a more stable forecast would retain orders
after some current prediction-line crossings. Such a change must keep hard
guards immediate and be evaluated for future adverse moves, not just fewer
cancels. These exit changes were not tested here.

The current capture omits exact per-decision guard values and pull reasons.
The earlier 8.77% reconstructed-score disagreement means an empty recorded lane
cannot safely be classified as a soft prediction pull from those scores alone.
Complete existing **A3** (exact guard values, active lines, known inventory and
per-lane reasons), or obtain an exact native-input replay with decision parity,
before testing selective retention on the live decision path. This is not a new
size-adjustment item. B3 exit-band/EMA research remains open.

Faster timer-driven re-entry is also outside this test; the nominal 25 ms result
must not be presented as a guaranteed 25 ms execution delay. Venue RPM limits,
backoff and post-only rejection are absent from the model, so the small cancel-ack
result is not a universal statement about cancellation latency tails or throttling.

## Verification and provenance

- Branch `research/predict-reentry-churn-20261109`, checkout
  `/tmp/cta-reentry-churn-20261109`; producer freeze `dfbe5ddf`, declaration
  `252bd26f`. Six arms, five new policies; cumulative distinct policies across
  this and the previous capture study: twelve. No fresh-day validation claim.
- **30 unit controls pass**, including causal prefixes, zero-delay parity,
  immediate pulls, same-price gating, pending creates, cancellation confirmation
  versus reservation grace, exact lost/gained coverage, and invalid-input refusal.
- Reference output matches the prior incremental artifact exactly on **18 metrics
  for all 220 windows**. Verifier reconciles **1,320 window-arm rows** and
  **184,776 action-arm rows**, including every admitted/withheld/censored status.
  Coverage obeys reference coverage minus lost time plus gained time per window.
  Artifact hashes, populations, ratios and comparison predicates reconcile.
  Corrupted counts and negative admission waits are refused by positive/refusal
  controls in the verifier and diagnostic.
- Exact pull reasons are unavailable; policies use only recorded intent, not
  reconstructed scores. All historical inventory feedback in that intent path
  remains factual, not a simulated counterfactual position. No fill rate, PnL,
  settlement edge, queue rank, variance reduction or confidence interval is claimed.
- No live code/settings, frozen engine, prior artifacts or fill-clock gate changed.
  Protected Nov9–13 were not read. All jobs completed serially under the resource
  wrapper; producer peak RSS 294 MiB, elapsed 48.66 seconds.

Receipts: `reviews/reentry_churn_20261109/{declaration,receipt,verification,diagnostics}.json`.
Parquets/logs: `data/predict_fun/research/reentry_churn_20261109/` (not committed).
Code: `research/pf_reentry.py`, `pf_reentry_study.py`, `pf_reentry_verify.py`,
`pf_reentry_diagnostics.py`, and `research/tests/test_reentry*.py`.

Controls, without market data:

```bash
predictfun/research/run_light.sh reentry_controls -- bash -c \
  'cd /tmp/cta-reentry-churn-20261109; exec python3 -m unittest predictfun.research.tests.test_reentry predictfun.research.tests.test_reentry_study predictfun.research.tests.test_quote_capture predictfun.research.tests.test_quote_capture_study predictfun.research.pf_quote_coverage'
```

Study entrypoints refuse overwriting immutable artifacts. Reproduction requires
the recorded source/input hashes and a separate destination; do not overwrite the
original receipts or remove the historical fill-clock refusal to produce a PnL.
