# Concrete partial-order chains and next-quote choices

As of 2026-11-24T05:18:17Z. Source freeze **551932c10**; comparator-only checker supersession **a5b947041**. Full source SHA: a5b94704167202e9520bb81a74412082dac7ac80. Consumed Oct24–27, top-three makers in the accepted fill-linked order census. No new days, tape replay, fit, validation or private action log.

We can reconstruct partial action chains and test explanations, but this round establishes no general BTC or inventory trigger. The useful result is that both lower and higher subsequent quotes occur while the old signed order still has unfilled budget; exhaustion alone cannot describe these conditional chains. This does not identify the private cancel command or prove the new order replaced the old one.

## Full observed-action census

4,149 source hashes give 3,286 distinct linked successor actions after de-duplication. There are 1,097 conditionally ordered posting windows, 1,361 overlapping/ambiguous windows and 828 refused windows. The 1,097 ordered actions split into **255 less-aggressive prices, 133 unchanged prices, 709 more-aggressive prices**. This is a fill-observed population: it omits orders that never fill, and is not a posting/nonposting decision census. G2/G4 posting windows remain conditional/scenario bounds.

Comparing prices relative to the contemporaneous touch gives 15 definitely farther-back,54 definitely closer,8 exactly unchanged and1,020 ambiguous ranges among the1,097. A lower bid or higher ask by itself cannot distinguish following a moving market from deliberately retreating. These are cross-window bounds, not exact decision instants. The offset outcome tests whether retreat is identifiable, not whether all ambiguous cases were truly non-retreats.

## Every ordered partial-budget chain

All56 signed positive-remainder cases are retained in partial_cases.csv:8 sequential under inherited ending bounds,23 ambiguous,21 unavailable and4 with no observed successor. The8 sequential cases divide into4 less-aggressive,3 more-aggressive and1 same-price quote. Every old ending here is conditional unavailability; none is a privately observed cancellation. Remaining signed budget is an upper bound, not proof the quantity was resting. No successor in this eight has a verified original signed size.

The gap is from the inferred old ending to the next posting window. BTC is signed to the order side; inventory “offsetting” means that side would reduce the reconstructed signed exposure. Both current-window and1s-lag classifications are shown, including mixed categories. All8 are listed, not hand-selected winners.

| Date / market / maker | Later quote | Old filled / original shares | Conditional gap | BTC D0 / lag | Inventory D0 / lag | Old hash prefix |
|---|---|---:|---:|---|---|---|
| 20261026 / 3789484 / #1 | bid 46→32¢ | 10.000 / 50 | 1.447–5.958s | against / against | offsetting / offsetting | `0x3be8243bef` |
| 20261027 / 3877472 / #1 | bid 56→50¢ | 11.364 / 160 | 0.848–4.834s | against|flat / against|flat | adding|offsetting / adding|offsetting | `0x714e35d667` |
| 20261024 / 3704004 / #2 | bid 62→66¢ | 22.182 / 50 | 7.120–14.073s | flat / flat | offsetting / offsetting | `0x7d51569cfb` |
| 20261024 / 3699278 / #3 | ask 77→66¢ | 7.692 / 10 | 6.998–13.088s | with / with | offsetting / offsetting | `0x9325239722` |
| 20261026 / 3803131 / #3 | ask 87→95¢ | 100.489 / 220 | 70.445–77.411s | flat|with / flat|with | offsetting / adding|offsetting | `0xab150bc108` |
| 20261026 / 3804696 / #2 | ask 11→12¢ | 30.833 / 50 | 1.491–7.654s | flat / flat | offsetting / offsetting | `0xae94dc82d9` |
| 20261025 / 3734864 / #3 | bid 31→31¢ | 1.449 / 10 | 3.049–12.798s | flat / flat | adding / adding | `0xb50f04e54b` |
| 20261024 / 3712725 / #3 | bid 40→61¢ | 21.951 / 220 | 11.913–14.520s | flat|with / flat | offsetting / offsetting | `0xf8a1df6fe1` |

Three concrete hypotheses emerge from these records:

1. **Reprice lower while still wanting buys.** Maker#1, Oct26 market3789484:50-share bid46¢,10 filled; conditional ending269.035–271.500s; next bid32¢ posted272.947–274.993s after market start. BTC is against the bid in both clock views; buying offsets a short position. Compatible with repricing the desired buy downward as conditions deteriorate. It does not prove a BTC-triggered cancel, or movement farther behind the contemporaneous touch.
2. **Quote more aggressively on the inventory-reducing side.** Maker#2, Oct24 market3704004:50-share bid62¢,22.182 filled; conditional ending19.530–24.642s; next bid66¢ at31.762–33.603s. BTC is flat and buying offsets short exposure in both views. Compatible with continuing to work a buy while short, but a rising market could also explain the price change.
3. **Same-price new order after a partial ending.** Maker#3, Oct25 market3734864:10-share bid31¢,1.449 filled; conditional ending120.521–124.724s; another signed hash at31¢ at127.773–133.319s. BTC is flat and buying adds exposure in both views. This is a candidate renewal with unexhausted old budget; it does not establish depletion, a desired queue reset or the new size.

The independent 146-hash packet also contains22 ordered successors after exact budget exhaustion (7 same-price /15 repriced). That is a separate, stronger physical exhaustion category. Multiple fills under one hash consume that hash's fixed original budget; a different hash is a different signed order. Public records do not prove continuous queue residence between fills.

## Tests of general logic

Eight comparisons were declared before computation:two exposures × four price/offset outcomes,500 conditional draws each, fixed support and multiplicity rules. Only observed quotes are compared with observed quotes; no absent fill is called WAIT.

- BTC-against versus flat:only3 against/4 flat cases in3 common strata/2 markets after identity,phase,old-touch,inventory and timing matching. **Insufficient support** for a general reversal rule. We did not loosen matching after this result.
- Inventory-adding versus offsetting:41/34 quotes in31 strata/12 markets meet the declared count floor. Less-aggressive-price difference **−4.89 percentage points**, raw exploratory p=.760; same-price +1.15pp,p=1;more-aggressive +3.74pp,p=.804. Day signs disagree. **No demonstrated inventory-driven repricing rule.** This concerns price choice conditional on another observed quote, not whether inventory affects sizes, takes or the decision to quote at all.
- None of the8 tests meets the declared exploratory-pattern rule. Exact day values,all null draws and every status are committed. The permutation is a descriptive conditional allocation; related actions and unmeasured confounding preclude causal claims. These consumed days supply no validation claim.

## What the evidence supports now

Observed skeleton: signed order → partial/exhausted executions → sometimes a bounded conditional ending → next observed same-side quote at the same or a different price. Price changes occur in both directions, including on the exposure-reducing side. Market-following, adverse-flow protection and inventory management are plausible explanations for particular chains; a universal controller has not been recovered. A useful strategy rule cannot yet be promoted from these examples. This packet measures neither queue-profit causality nor whether remaining at the front pays for early losses.

The largest limitation is timing/ownership identification, not a missing classifier:most posting-offset intervals span several feasible states, and unseen/cancelled-before-fill orders are missing. Inherited unavailability and G4 scenarios remain assumptions. Reconstructed holdings are not private client-known inventory; no proof of the maker's motivation is claimed.

## Verification and disposition

Source identities,canonical successor membership,all prices/statuses/conditions/offset bounds,all scalar matched rates/null draws and all56 partial cases were independently recomputed from hash-bound accepted artifacts. Eight real-output corruptions were refused. Synthetic controls include a known association and no-association fixture. The first checker failed on Python bool-as-int parsing;v2 changes only boolean comparison,with explicit positive/negative controls. Measurement and frozen predicates are unchanged;failed logs remain.

Measured packet CPU before this publishing job:2.00s;conservative publishing allowance .15s;packet cap5.04s. Task before publishing 1.9382167coreh,line 5.2489167;both still fit task2.00/line5.30 plus .05 standing reserve. The final ledger receipt records this job after it closes. No implementation or other-session file is changed.

ONE next priority, if further compute is authorized:an **ending-anchored market-following versus retreat audit** on the56 signed partial-budget cases, comparing their old price against the touch/fair-value ranges at the inferred ending and at the next post, with overlapping and unidentified histories retained. The present post-to-post offset ranges cannot answer that narrower question. Do not fit a policy, weaken matching or claim queue advantage first. This is existing consumed evidence; it needs a bounded resource top-up if it exceeds the remaining allowance. No new days or live use are implied.
