# Quantity-normalized indirect event reconstruction

2026-11-24. USER: normalize first, enumerate possible indirect relations, check those relations, then connect actions into their logic chains. This is the execution plan and relation inventory, not a new empirical result or a claim that all relations are usable. Base **463470cd680f407a9cc42c0531fb28f7225077b1**. It replaces concentrating the next measurement only on three reversal-compatible examples. All 88 fixed signed orders remain the first reconstruction cohort; three reversal cases are annotations, never selection criteria or constraints.

## Goal and order of work

1. Verify identity, quantities, prices, sides, timestamps and source completeness.
2. Build order/fill/book/pending events without consulting BTC or intended action hypotheses.
3. Test the catalog below, retain competing histories, and combine compatible constraints jointly.
4. Report estimated posting, matching, exhaustion, withdrawal and successor intervals with their evidence and assumptions.
5. Join BTC, local book pressure, inventory and time-to-END only after these intervals are fixed. Test the proposed condition -> action chains across the eligible population, not only the compatible examples.

We can estimate without certifying a private command. Never force one history merely because it fits the desired trading logic. Conversely, missing private events do not prevent useful conditional estimates.

## Normalization contract

| Item | Canonical representation / verification |
|---|---|
| Order identity | Chain/exchange domain + full signed order hash, plus market/token/maker cross-checks. Keep native fields and YES-space side. Same price/size/address is not a unique order. A hash collision in metadata is a refusal, not a new order. |
| Event identity | Successful on-chain event: chain + tx hash + log index; retain block/transaction/log order. REST settlement/leg and on-chain event are representations of the same execution and must be linked, not added twice. Repeated WebSocket snapshots also are not new orders. |
| Signed budget | Original makerAmount/takerAmount are immutable raw integers; preserve token/collateral units and decimal scale. Exact cumulative making is checked in raw units. Derive quoted shares at the signed price with explicit side/outcome conversion; do not treat collateral wei as share wei. |
| Shares / price | Common YES-equivalent side and price for book joins; native price/outcome beside them. NO buy becomes YES ask at the complementary price; quantity stays shares. Keep integer ticks/decimal arithmetic where possible; flag rounding rather than silently clipping contradictions. |
| Filled quantity | Distinct successful fills across all observed roles for the hash. Cumulative confirmed making must not exceed the signed budget. Original size does not reset after a partial fill. Coverage-incomplete history yields a bound, not an exact live remainder. |
| Pending quantity | Preserve raw per-price pending USDT separately from displayed shares. Convert with the verified price convention before comparing amounts. No division by invalid/zero prices; unknown stays unknown. Pending is aggregate, not maker-attributed, unless a unique allocation is supported. A negative pending delta is not itself a cancellation. |
| Quote availability | A confirmed fill consumes budget; a matched-unsettled quantity is reserved. Do not count it again when settlement arrives. Releasing a failed settlement can restore availability without a new post. Verify whether the historical displayed field includes/excludes reservations before a conservation equation uses it. |
| Clock | Retain venue book update, local receipt, settlement record, chain time/order, and execution interval separately. Normalize ms/ns explicitly. Never substitute block/UUID/receipt time for the private post or cancel time. Same-time records remain ties where order is unknown. |
| State validity | Zero size, missing cell, truncated book, out-of-order frame, reconnect, stale/gap and coalesced updates are distinct statuses. Missing/invalid frames do not certify an empty level or create a new lifetime. |
| Source coverage | Every constraint records its exact input rows, historical schema/collector version, coverage interval and any missing fills/events. Current documentation helps specify a convention but does not prove the October raw payload follows it. |

Existing code inspection: `pf_mt_extract.py` retains raw pending price/value lists; `pf_mt_lifecycle.py::_pend` reads them as USD, `_book` keeps displayed shares and pending in separate arrays, and match evidence compares pending increments to shares × YES price. This is an existing unit distinction, not evidence of a newly found normalization bug. The next packet must verify the actual saved records and prevent conversion twice. Source: [extract](../../research/pf_mt_extract.py), [lifecycle](../../research/pf_mt_lifecycle.py), [signed identity and exact budget checks](../../research/pf_mt_signed_evidence_v3.py).

Useful accounting, in one consistent native budget unit:

- `B_confirmed_remaining = B_signed − sum(distinct successful making amounts)`.
- If history is incomplete, this is an upper bound on remaining budget. It is never proof that the remainder is live on the book.
- If active and the per-order reservation is known, `B_available = B_confirmed_remaining − B_pending_reserved`. Otherwise retain a range over feasible pending allocations.
- For a verified FREE displayed level in shares: `ΔD = new/restored displayed shares − newly reserved matches − withdrawals`, with amendments/other causes retained where applicable. Settling an already-reserved match does not subtract it a second time. Aggregate coalescing can hide offsetting additions and removals; this is a conservation constraint, not a unique decomposition.

## Relation catalog — practical candidates in the available sources

The table is the inventory to check, not 20 new independent statistical hypotheses. Evidence types: **O** identity-bearing observation, **C** constraint under listed premises, **S** scenario-dependent link, **H** behavioural hypothesis. The same raw change supporting several rows counts once in joint inference.

| ID | Candidate relation | What it can establish or narrow | Main alternative / required check | Order |
|---|---|---|---|---|
| R01 | Same signed hash -> multiple fills (O) | One immutable budget, cumulative execution, remaining-budget bound | Deduplicate representations; complete all roles; same hash does not prove continuous resting | First |
| R02 | Cumulative raw making equals original (O) | Budget exhaustion at completion of those executions; excludes positive terminal remainder | Earlier temporary withdrawal still possible; settlement time != execution time | First |
| R03 | Book decrease + pending increase + later linked settlement (C/S) | Candidate match interval; distinguish match from pull | Units, aggregate pending allocation, delayed/coalesced updates, concurrent unrelated orders | First |
| R04 | Pending release + failure evidence + restored depth (C/S) | Possible release of reserved liquidity rather than a new quote | Without failure identity, refill or unrelated additions remain possible | First |
| R05 | First fill of a hash (O/C) | Order was accepted by its execution; upper bound on posting | Fill execution interval uncertain; first fill does not prove early rest | First |
| R06 | Later fill of the same hash (O/S) | Remaining budget continued to execute at a later event; possible continuous rest | Off-chain removal/re-entry, reservation before withdrawal, settlement reordering | First |
| R07 | Isolated level zero -> positive, compatible original/remaining size (S) | Candidate add/accept window | Other orders, masked netting, partially observed display, unknown vs zero | Second |
| R08 | Displayed/pending capacity cannot accommodate a required remainder (C/S) | Exclude some active-order histories; narrow posting or removal | Complete fill coverage, full-display assumption, reservations, rounding and gaps required | Second |
| R09 | Joint allocation of one book change across several candidate hashes (C/S) | Avoid assigning the same addition/drop to multiple orders; eliminate incompatible combinations | Unfilled anonymous orders and unknown makers retain slack; no forced unique allocation | Second |
| R10 | Depth decrease unexplained by accounted matches/reservations (S) | Candidate non-trade removal interval at that level | Cancellation, expiry, invalidation, unrelated removals; not automatically this maker | Second |
| R11 | Trade executes strictly through the quote's price without that hash (C/S) | Bound absence/availability under price priority and feasible execution ordering | Already reserved/exhausted, self-trade prevention, different outcome mapping, incomplete trade set | Second |
| R12 | Post-partial-fill same-price executions by other orders (S) | Under FIFO, test whether a remainder could still have retained priority | Aggregated/tied execution, maker-list order not queue order, re-entry, priority exceptions | Second |
| R13 | Same hash later appears after a supposed terminal removal (O/C) | Refute an irreversible-terminal or no-reentry history where timing is certain | Match may precede removal but settle later; failed match release; off-chain signature remains valid | Second |
| R14 | Signed expiry / explicit linked cancellation or invalidation (O/C) | Separate terminal causes and constrain time | Broad search coverage absent; API removal usually off-chain; expiry timestamp alone is not observed processing | Second |
| R15 | Old remainder disappears + new hash same maker/side nearby price (S) | Candidate replacement/reprice chain and interval order | Independent new order, parallel ladder, temporary overlap; a new hash is not proof of replacement intent | Third |
| R16 | Exhaustion + later new hash at same price (S) | Candidate replenishment chain | New order may already have rested; unfilled successors missing | Third |
| R17 | Overlapping old/new hashes on same side across prices (C/S) | Candidate ladder / overlapping replacement instead of sequential repost | Need genuine overlap in availability, not overlapping settlement records alone | Third |
| R18 | Attributed opposing fills/takes + inventory path (O/C/H) | Inventory reduction, completed turnover and residual exposure; possible risk-management explanation | Maker's known inventory/notices unobserved; opposing trade need not reduce absolute inventory; FIFO assignment != intent | After linkage |
| R19 | BTC/local book/flow reversal, inventory pressure or market phase -> action (H) | Test condition/action associations using independently fixed intervals | Same evidence cannot define action and verify cause; shifts/wrong-side/exhaustion controls; outcome-selected sampling | After linkage |
| R20 | Reconstructed posting/partial fills + intervening competing volume (S) | Candidate queue acquisition, depletion and retained priority histories | Continuous rest/FIFO assumptions; public flow not counterfactual if we change policy; no queue-front certificate from repeated fills alone | After linkage |

R19 encompasses a hypothesis register rather than one unrestricted search: BTC reversal, local fair-value deterioration, inventory pressure, queue-preserving retention, replenishment after exhaustion, and expiry/end-phase actions. Declare one operational definition and its comparison before a new measurement; do not tune all combinations on these four consumed days. The prior frozen reversal check remains as written.

## Combining relations without inventing certainty

Represent each order and event as a node. A candidate link records exact IDs, [earliest, latest] time, native/normalized amount, evidence rows, assumptions, conflicts, competing explanations and grade. A successor link may remain one-to-many. Statuses are `observed`, `constrained_under_premises`, `scenario_possible`, `contradicted_under_premises`, `unknown`.

Within one consistent scenario, intersect compatible constraints. Across incompatible scenarios, retain the union of feasible histories and report their separate bounds. Never intersect FIFO and no-priority conclusions as if both were unconditional facts. An empty feasible set becomes a conflict to diagnose; it must not be repaired by silently moving a fill, treating a gap as zero or forcing a cancel.

Book drop and pending growth can be two manifestations of one match. A hash's multiple fills can link several snapshots. Allocate quantities jointly; do not add confidence simply because correlated clues agree. Do not manufacture a calibrated probability from a count of matching clues. Fewer hidden commands can be a labelled representative history, not a certified actual history.

Maintain a ledger of surviving alternatives: continued resting, newly matched/pending, fully exhausted, withdrawn, expired/invalidated, failed-settlement release, re-entry, replacement and parallel ladder. Unknown remains a valid outcome.

## ONE next executable packet

**Normalize and reconcile the fixed 88 signed-order chains, then apply the quantity/pending/book-capacity relations R01–R10 in one bounded batch.** R01/R02 already have exact signed-budget evidence; reuse it and verify joins rather than rebuild extraction. All 40 conditional withdrawal candidates, 41 exhausted controls and seven unresolved orders stay in coverage tables. Do not select only the three reversal-compatible cases. The three are separately flagged after signal-blind linkage to see whether post-before-reversal becomes clearer.

The implementation freeze must bind the data adapter, historical units/clock checks, relation definitions, outputs and controls before the first new relation measurement. Read saved sources only, stream by market and retain failures. Use existing higher-level R11–R20 outputs as alternatives/context; no full-day replay, refit, new data, archive retry or full condition rebuild. Weak or unavailable relations get statuses; they must not hold up stronger useful relations or create a cleanup-only round.

Deliverables:

- `normalization_receipt`: field/units/clock provenance, raw-to-normalized reconciliation, duplicate/coverage/unknown counts.
- `order_ledger`: original budget, cumulative execution, pending allocation range, remaining bound and terminal alternatives at each event.
- `relation_edges`: every tested relation with source IDs, assumptions and accepted/rejected/unknown status.
- `chain_bounds`: old and new post/end bounds beside each other, surviving histories and reasons for narrowing or conflict.
- One report across all 88: orders gaining information, width reductions at unchanged grades, grades changing with assumptions, contradictions, abstentions and negative-control false-pull count. Include denominator/status for every rate; no cancellation-rate or policy-profit claim from this sample.

Controls must include: partial→partial→exhaustion; newly matched→settled without double decrement; failed settlement→release; same size/different hash; duplicate REST/on-chain representation; NO-to-YES conversion; USD/shares mismatch; unknown/reconnect frame; simultaneous add+remove hidden in net depth; multiple hashes competing for one level change; expiry without owner cancel; later settlement after earlier matching; explicit false-pull on an exhausted order. Require positive controls that actually narrow a known timeline and bad controls that force refusal. Use available own-order ground truth only if already saved and within this packet; otherwise label conditional reconstruction rather than requiring a new source.

## Resource and scope

Plan/catalog only in this commit: no new outcome, field measurement or hypothesis verdict. Last measurement ledger: task about 1.9143/2.00 coreh, line about 5.2250/5.30; standing reserve .05. Count publication overhead once in mt_relplan. Next packet <=.015 coreh all-in, PF_SLOT=codex/run_light, with a count/schema/runtime projection before the real pass and no more than the remaining ledger headroom. If full relation application cannot fit, preserve normalization/coverage and report which relations were not evaluated; do not claim them checked. Same consumed-day saved-data scope is authorized by the USER; no new permission required within these caps. New source/access, new days, live use or larger caps remain separate decisions. No push; preserve all other-session changes.

## Source conventions already checked in the prior turn

Current documentation: [order lifecycle](https://dev.predict.fun/trading/order-lifecycle) (matched reservations vs settlement), [book](https://dev.predict.fun/concepts/order-book) (shares vs pending USDT, YES axis), [book stream](https://dev.predict.fun/websocket/topics/order-book) (coalescing), [cancel](https://dev.predict.fun/trading/cancel-orders) (off-chain removal vs on-chain invalidation). Historical compatibility is a normalization gate, not assumed from these current pages.
