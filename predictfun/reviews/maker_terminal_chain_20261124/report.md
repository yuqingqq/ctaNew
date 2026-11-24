# TERMINAL-ORDER-AND-WHOLE-CHAIN-v1 — results and action-chain interpretation

2026-11-24. USER continuation: bridge maker-action identification gaps and show the whole action and logic chain. This is research implementation, not a review-only receipt. Source freeze **335797b52963916d490e2f9c60428c882ab80416**; independent checker and presentation freeze **d2b86cd2703f22e4421aad83fd29bd2a4fcdd1b4**. [Contract](contract.md), [manifest](manifest.json), [computed summary](summary.json), [independent checks](checks.json), [all six chains](whole_chain.md).

## What the new evidence establishes

The fixed sample contains **88 orders and 152 settlement transactions**, as of 2026-11-24T02:08:07Z. It takes eight hash-ranked orders from each nonempty consumed Oct 24–27 day × top-three volume-rank cell; rank 3 on Oct 27 has no eligible orders. Selection precedes new signed-size and ending evidence. The starting population is the old fill-selected, scenario-labelled S_FIFO cancel-compatible cohort, not neutral quoting opportunities. These stratified counts are not estimates of all-maker cancellation rates.

| New ending evidence | Orders | Interpretation |
|---|---:|---|
| Entire signed making budget consumed by observed fills | 41 | No unfilled signed remainder at final completion; an earlier temporary withdrawal is still possible. |
| Observed making below signed budget | 47 | Remaining history unresolved; not a proven cancellation, expiry or live quote. |
| Linked on-chain cancellation in the queried settlement receipts | 0 | Narrow search scope; does not establish absence of off-chain or separately transacted cancellations. |

All 152 required transactions were acquired and reconciled. The independent checker recomputes full order hashes and integer making budgets from raw receipt logs, rather than trusting exported labels. Four planted corruptions are refused: changed amount, unsupported private-pull claim, unsupported queue-front claim, omitted raw event. No source discrepancy is absorbed as a missing row.

The old successor windows yield 39 without a recorded successor, 10 potentially overlapping, 14 successors with an exhausted predecessor, and 25 unresolved replacement/parallel alternatives. The exported label `later_order_after_exhaustion_command_unidentified` describes the old conditional chronology, **not an identified submit-after-fill command**. Unfilled successor orders are invisible to this source. No private pull or queue-front position is newly identified.

The old BTC/inventory window cross-tabs remain descriptive. They use fill-anchored, scenario-timed windows and occur in both exhausted and unresolved groups. They cannot become a decision-time pull model by changing their labels. No new null, model, threshold, strategy candidate, significance claim or validation claim is introduced.

## Historical state limitation — retained, not replaced with latest state

An optional `orderStatus(bytes32)` lookup was prepared for each order at its last recorded fill block. Of 88 prepared requests, **64 request slots were submitted in four batches**. The first three batches returned 48 `missing trie node` errors; the fourth returned one batch-level rate-limit error, failing response-membership validation. The remaining 24 requests were not submitted. No historical state was evaluable. All 88 therefore retain `archive_or_rpc_unavailable`; this is not a claim that 88 individual archive calls failed.

No provider retry, current-state substitution or inferred remainder was used. This failure does not invalidate exhaustion proven directly by complete signed-budget consumption. It prevents the optional historical-state bridge for unresolved endings. Even a successful on-chain getter would not reveal off-chain withdrawal or distinguish every cancellation/exhaustion alternative by its boolean alone.

## Whole action chain: a concrete 220-share example

All six examples were selected earlier and are retained in [whole_chain.md](whole_chain.md), with [summary rows](six_chain_summary.csv) and [fill timelines](six_fill_timelines.json). They are six orders from three consumed Oct 24 markets, not an independently selected profitable sample. Chain 5 is rank 3 in market 3712725, order `0x01ebf81129c65381fd72b5b33ddf3f4aa5e0b84e52b146e789e224ee313344b6`.

| Stage | Evidence | Economic interpretation and limit |
|---|---|---|
| Market conditions and quote decision | Saved nearby context includes a one-tick spread and flat five-second BTC movement. The inferred posting interval has `G1_conflict`. | Reject that interval as a posting-time bound. These are nearby alternative contexts, not established conditions at the private decision. |
| Signed order | 220 shares at a YES-equivalent ask of 32c. The actual signed instrument is a NO buy at 68c. | Known original signed terms; displayed size, initial queue ahead and continuous resting are unknown. |
| Repeated execution | Seven records, 187.544–195.961 seconds after opening, totaling 220 shares at 32c in YES space. | An 8.417-second first-to-last record span; not 8.417 seconds of pre-fill rest or proof of uninterrupted queue ownership. |
| Inventory turnover | FIFO assigns 211.856953 shares to closing earlier inventory for **+$6.632623 net**. The other 8.143047 shares later close through maker fills for approximately zero net. | The passive order principally completes earlier turnover. Its standalone END markout cannot replace the full acquisition-and-close accounting. |
| Order ending | Observed fills exhaust its signed budget. No filled same-side successor is recorded in the old episode window. | Completion is identified. Absence of an unfilled successor, earlier pull or repost is not identified. |
| Later aggressive trades | Five YES-buy takes, 58 shares at 34c, recorded at 206.010–206.078 seconds. The assigned 204-second anchor reconstructs a roughly **397.12-share short** position; all three timing views classify the takes as reducing exposure. | These takes reduce other book exposure. Exact FIFO attribution assigns none to closing this 220-share order. The anchor is reconstructed, not the maker's private decision-time inventory; intervening trades preclude deriving the final position by simple subtraction. |

The supported chain is therefore: **signed passive quote → repeated execution → profitable closing of earlier inventory → full budget exhaustion → later aggressive reduction of other short exposure**. The “why” is an accounting-supported economic explanation, not a recovered private algorithm. A +$6.63 FIFO close does not establish that selling was optimal versus retaining the inventory; opportunity cost and whole-book risk still matter.

The six examples include losing completed turnover as well: chain 1 new-entry closes +$2.27; chains 3, 4 and 6 approximately −$3.46, −$4.49 and −$0.71; chain 2 retains 6.6666 observed shares without an identified close. Those mixed outcomes prevent promoting this example into a profitable policy or an optimal-size recommendation.

## Interpretation boundaries carried into the handoff

1. **Economic logic is clearer than command logic.** Passive orders can close existing inventory as well as open new exposure. Later takes can manage a different part of the book. A useful quote-value target must include existing inventory, subsequent opposite-side executions and residual exposure, rather than the first fill's short markout alone.
2. **Queue acquisition remains unknown.** Repeated execution of one hash and a large signed size do not prove early posting, continuous rest or front priority. This packet does not infer that queue advantage explains most maker profit.
3. **Pull/reprice motives remain unknown.** Full exhaustion removes one terminal ambiguity for 41 orders. It does not rule out temporary off-chain withdrawal or establish the cause/timing of later quotes. The 47 incomplete observations remain unknown, not labelled bad actions.
4. **Presentation qualifications:** the generic exhaustion explanation in `whole_chain.md` applies only where its preceding status says exhausted; chain 2 is incomplete. Old `sequential_identified` labels are conditional scenario links, not identified private actions. G1 posting intervals are explicitly rejected. Earlier take-window summaries and continuation-to-END accounting have different horizons and must not be summed.

## Verification, resource and publication

Frozen selftests cover sample order invariance, scenario/day guards, integer budget boundaries, exhaustion/partial/cancel precedence, state responses and successor overlap; accepted decoder/RPC controls are reused. Independent raw-log recount and four falsifiers pass in `checks.json`. Whole-chain assembly checks exact keys and previously accepted order/fill/FIFO identities. A static reading of the existing chain-5 evidence was also supplied by the existing review agent; it made no changes or new measurement.

All result-bearing runs used `PF_SLOT=codex run_light.sh`, tags `mt_terminal_*`. The ledger after checks is **task 1.9108/2.00 core-hours; line 5.2215/5.30**, including this packet once, with the standing 0.05-hour reserve still fitting. Packet increase is approximately 0.0011 core-hours against the declared 0.025 ceiling. Public read-only transaction lookups only; no replay, training, new day, live use, push or schedule. Raw RPC data remain under ignored `data/predict_fun/research/maker_terminal_chain_20261124/`; their hashes are in `summary.json`. Committed outputs are bounded small files. Publication preserves other-session working changes and stages only this packet plus additive owned handoff/state blocks.

## ONE next priority

**An attributed terminal-event evidence gate for the same 47 unresolved orders, before another action-condition fit.** First establish an available historical source that can reconcile signed terms with complete per-hash fills and cancellation events through the declared episode endpoint. Freeze source coverage and failure statuses before extracting action-condition tables. Successful archived `orderStatus` is supplementary; it cannot certify an off-chain quote. Require direct lifecycle/command evidence to claim a private pull or repost, and retain an unidentified category otherwise. Stop if the source adds no identifying information; do not spend another round tuning anonymous disappearance associations.

Bounded read-only work on these same consumed days fits the USER's continuation authorization, subject to a fresh count/resource gate within existing caps. Access credentials, paid archival service, untouched/new days, live recording/trading or a cap increase are separate USER decisions. No such dependency has been acquired or launched. The current endpoint has already failed its historical-state gate; do not retry it in a loop. Meanwhile the six full chains above are usable evidence for designing whole-book economic values, with their private posting and queue mechanisms explicitly unresolved.
