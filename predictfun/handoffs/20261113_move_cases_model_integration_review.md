# Maker move cases: event-weighted offsets are not verified portfolio P&L

Review as of **2026-11-13T17:46:07.922570+00:00**. Producer MOVE-CASES report as of **2026-11-13T16:12:36Z**, source state `81ae9c60`. Nine consumed dates: Oct 23–27/31 and Nov 1–3. The report and its post-hoc side table reconstruct exactly from their saved inputs. No model, policy, threshold or live behavior changed.

The source can attribute the same fill to multiple overlapping market-move events. Its daily totals sum those event records, so they cannot be interpreted as verified unique-fill portfolio value. This does not establish how much historical duplication occurred or disprove that the makers are profitable; the saved identities are insufficient to answer either question.

## Scope and populations

The audit binds seven producer sources, four review sources and 40 inputs; all hashes and three receipt copies match. It checks original daily coverage metadata and exercises the exact producer event accumulator, path classifier, original positive selftest and three planted-error controls. Changed input hashes and failed metadata are refused.

At the producer as-of time, the saved populations contain **100,719 event records**, **1,233,693 real maker/taker segment records**, and **21,738 simulated segment records**. Segment rows aggregate fills within an event, side, group and segment; they are not individual trades. The producer reports 99 invalid-event exclusions, 8,414 truncated-event exclusions and zero zero-move exclusions. Its repeated-event accounting is unchanged by this review.

Original top-five maker summaries, restricted to adverse signal events:

| Realized case | Aggregate segment rows | INTO-side mark, USD/day | AWAY-side mark, USD/day | Sum, USD/day |
|---|---:|---:|---:|---:|
| Big trends | 56,555 | -5,923.77 | +6,168.34 | +244.56 |
| Big reversions | 30,099 | +4,393.46 | -1,101.52 | +3,291.93 |
| All cases | 195,859 | +1,589.74 | +6,375.05 | +7,964.79 |

These are reproduced **event-weighted five-second fill marks divided by nine days**, not independently verified profits. They show opposite signs across sides/cases in the selected records. They do not establish that the same maker hedged each event, held continuous quotes or maintained comparable inventory risk.

## Source counterexamples

- **Overlapping event attribution:** two transitions can have different start times and old prices but share a later peak and the same subsequent fills. The exact producer accumulation block admits the same bid and ask fill into both events. In the synthetic control, two physical maker fills total ten shares and five cents of marked value. The two overlapping event records sum to twenty shares and ten cents. A disjoint-event control adds nothing. No historical duplication multiplier is claimed.
- **Missing identity:** the saved event/segment tables omit event IDs, fill IDs, order hashes and signer identities. They therefore cannot support exact fill deduplication or attribution of both sides to the same individual maker. The producer’s source-level event equality check does not supply those missing output identities.
- **Unobserved endpoint:** the exact path function falls back to the last valid frame when selecting the price three seconds after the peak. A synthetic tape ending at 1,500 ms is classified as a trend at a requested 4,200 ms endpoint; adding a real endpoint frame can change it to a reversion. The historical affected count is unknown. Calendar-window truncation alone does not prove observed endpoint coverage.

The realized case and peak are future-conditioned descriptions. They are useful for examining failure modes, but are not features available at a quoting decision. Simulated ladder fills in this study also come from six independently replayed rungs; equality to an earlier simulation does not establish a shared inventory budget or reservation constraint.

## Consequence for the model goal

Two-sided value and inventory reduction remain important. This report cannot supply a portfolio profitability target, an identified maker hedge policy, or a causal cancellation label. The active experiments continue to require neutral action populations, retained order identity, a single sequential inventory path, joint capacity constraints and marked value at measured risk. Their declarations and success gates are unchanged.

The separate [next-trade label audit](20261113_learn_pull_model_integration_review.md) explains why a maker’s absence from the next fill also does not identify cancellation. Neither public-study shortcut replaces the controlled five/50-share test.

## Provenance and validation

The first review attempt, frozen at `6c442ecb`, stopped during synthetic controls because the audit wrapper omitted the producer’s `xa = 1 - x` binding. It wrote no declaration or historical result. Corrected source `24e6a426` supplies that binding; the v2 declaration preserves the failed attempt and log hash. Producer code and tables were unchanged. The successful auxiliary job exited zero in 4.18 seconds with 977 MiB peak RSS; no raw historical replay or model fitting occurred.

- [Review receipt](../reviews/model_recovery_20261113/move_cases_integration_review_v2.json), SHA `c337f2ad0f839506434857442dfa69cb1c49dec920844ca3144eb384ec871e75`.
- [Declaration and failed-attempt provenance](../reviews/model_recovery_20261113/move_cases_integration_review_v2_declaration.json), SHA `5aa9086f73993b3ec3cdae85b23ed82fc5eae19473206932a20056327020b35e`.
- [Review instrument](../research/model_move_cases_integration_review.py).

Original outputs remain preserved. No improved-model, forward-validation, live-profit or top-maker-parity claim is established.
