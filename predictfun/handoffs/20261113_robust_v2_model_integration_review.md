# Corrected maker study: useful refill evidence, residual quantity still unknown

Review as of **2026-11-13T16:09:38.900898+00:00**. The corrected producer report
is dated **2026-11-13T15:53:51Z** and uses the nine consumed dates Oct 23–27/31
and Nov 1–3. Its complete report text and K1–K7 verdict JSON reconstruct
exactly from its saved tables. This checks report arithmetic and interpretation;
it does not independently replay its raw data or establish joint-budget profit.

The corrected first-trade and per-order measurements strengthen the evidence
for testing useful residuals. They do not identify original order size, quantity
remaining at a decision, or the policy that caused a missing later fill. The
frozen five/50-share experiment supplies those states through native captures
and retains neutral actions, rather than selecting only orders that later fill.

## What the saved tables establish

The following observations concern the side from which the best price moves
away. Refill rates condition on **at least 300 shares subsequently trading at
the old price** and a complete ten-second observation window. Counts are
**order-event records**, not independently identified unique orders.

| Direction | Group | Observed refills / qualifying records | Refill rate | Records with under 10 observed prior shares | Refill rate in that subgroup |
|---|---|---:|---:|---:|---:|
| UP | top five | 251 / 1,004 | 25.00% | 199 | 46.23% |
| UP | field | 234 / 2,808 | 8.33% | 1,081 | 11.10% |
| DOWN | top five | 351 / 1,450 | 24.21% | 289 | 45.67% |
| DOWN | field | 325 / 3,731 | 8.71% | 1,487 | 11.84% |

The separate value calculation uses **all observed later-refill records**,
without the 300-share condition. For top makers, the UP population has 1,634
records and 52,148.60 later-filled shares, marked at +0.5313 cents/share at
one second and +0.1627 at five seconds. DOWN has 1,999 records and 69,715.30
shares, marked at +0.6017 and +0.4392 cents/share. These are descriptive,
event-weighted fill marks. They do not value every submitted order, inventory,
unfilled exposure, cancellation alternatives or a complete portfolio.

On the side into which the price moves, same-hash fills at least 250 ms before
birth prove some prior presence. For top makers the UP population has 13,006
order-event records, with 1,045 carrying that proof; the proven fraction of
later shares is 6.64%. DOWN has 11,554 records, 1,028 proven and 6.89% of
later shares. Field fractions are 4.09% / 4.44% on 34,301 / 31,260 records.
These are lower-bound observations and do not establish majority pre-positioning.

K3b includes every hashed order attributed to the first identified taker,
removing arbitrary ordering within a known trade. Top makers account for
32.54% of its 56,179 first-trade order-event records, compared with 27.25% of
all 90,121 adverse-side records. This supports the descriptive priority
hypothesis, subject to the unknown-taker fallback described below.

The producer retains exclusion metadata: 63 adverse, 61 other, 36 calm-fallback
and 36 calm-improvement events have invalid birth frames; 15,161 maker events
lack the full ten-second window. Certain hashes cover 247,868 / 248,416 top-five
WS first-minute legs and 648,070 / 674,137 field legs. The source tables still
omit event/order/taker identities needed to de-duplicate repeated event exposure.

## Exact-code interpretation controls

The review executes the committed producer's extraction functions on controlled
inputs. Its original positive test and three planted-error refusals also pass.

1. **Observed prior fills do not identify a residual.** The exact `old_orders`
   extractor returns identical records for a five-share order that has filled
   five shares and a 50-share order that has filled five shares, if neither has
   a later recorded fill. Their hypothetical residuals are zero and 45 shares.
   Both satisfy `pre_q < 10`. A later same-hash fill correctly identifies the
   second order's refill, but only after that outcome occurs.
2. **The prior-fill quantity is local, not lifetime.** It sums only the
   two-second proof window. A 50-share order with 45 earlier fills and five
   recent fills is exhausted but still receives `pre_q = 5` in the control.
   Calling this subgroup “not used up” is therefore unsupported.
3. **A nearby two-level taker is not uniquely the transition-causing taker.**
   The sweep function accepts a common taker at both prices in the declared
   `(birth - 1 s, birth + 100 ms]` window. It receives no price-run end or
   transition-taker identity. A controlled post-birth two-level trade is
   admitted even though all its new-price hashes lack prior-fill proof.
   Its top-maker market-volume share and the top makers' own proven prior
   share have different denominators. Historical affected frequency is unknown.
4. **Unknown first takers retain a one-leg fallback.** Known-taker ties are
   invariant to row order. If the first taker id is unknown, the same function
   uses the first leg only; swapping tied rows changes group attribution in
   the control. The saved `firsttrade` table omits taker ids, so its historical
   frequency cannot be measured from those outputs. Hash coverage does not
   answer the taker-identity question.

These controls limit inference; they do not erase the observed refill or
prior-fill evidence. No public no-fill outcome is converted into an observed
cancellation, and no future-volume-conditioned group becomes a causal training
population in the model work.

## Policy and goal implications

The corrected early-entry arms are now compared with their corresponding
ordinary/fast path. Their negative or not-shown K7 readings reconstruct. All
inspected K7 day/arm/scenario cells satisfy the declared 30-record minimum.
The gated arms still inherit the previously identified depth-score leakage;
their aged-fill marks are not the native joint-budget objective. K8 is inherited
from v1 and was not independently rechecked by this review.

The useful design consequence is to estimate keep/pull value using actual
notified residual quantity and inventory, and to price re-entry consistently.
The existing size experiment already implements that comparison. This study
does not authorize a larger fixed live order, demonstrate an optimal maker
policy, or resolve the recovery candidate's low presence. The separate
[absence diagnostic](20261113_recovery_absence_overlap.md) and
[negative portfolio comparison](20261113_recovery_primary_stress_review.md)
remain part of that assessment. No frozen experiment was changed or selected
using these findings.

## Evidence

Review source **c3a60809**, producer **9b218820**. Before aggregation the
declaration binds nine producer source identities, three review sources and
66 input artifacts. The preserved tables contain 49,216 sweep rows, 123,440
order rows, 123,440 first-trade rows, 163,254 maker rows, 170,290 old-order rows
and 1,352,363 simulated variant-fill rows. These populations are separate.
No simulated fill lacks its one-second value. All daily coverage/control
metadata pass; incorrect clock metadata and artifact digests are refused.

Execution exited zero on the free auxiliary slot, using 390 MiB peak RSS,
39.67 seconds wall and 39.02 seconds user CPU. All declared sources/inputs
and three receipt copies matched after execution.

- [Result](../reviews/model_recovery_20261113/robust_v2_integration_review_v1.json),
  SHA `8a6e7378ca5b157ab88826304d5c59fba497ad188d0577e98dc7fa9d1b42dd61`.
- [Declaration](../reviews/model_recovery_20261113/robust_v2_integration_review_v1_declaration.json),
  SHA `613ce6968c90674b2184f7e3cbb103a271c16e3ded1bb812f9629419809f15fe`.
- Execution log SHA
  `e3f6eef878ef49f7cdba62e4144e3d28b7af81d4600ef6f179b5b821cc68a675`.

Recovery is still running; the fixed size and depth studies remain queued.
The better-model goal is active. No live changes or forward-validation claim.
