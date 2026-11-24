# Signed-order evidence: a real bridge from the maker handoffs

2026-11-24 UTC. USER resumed identification: “Can you take the handoffs and continue to identify actions and bridge the gap.” Base handoff commit **3464fc641**; accepted source freeze **c125c3b63** (v3); six-chain acquisition/attachment freeze **ba09b710b**. This is new research work, not a review-only commit. No strategy, simulator or live code changed.

**Result: the previous aggregate-book information limit was not the limit of all available evidence. Signed settlement orders recover original terms and remove the exhaustion-versus-cancelled-remainder ambiguity in five of the six existing examples. Posting times, queue front and off-chain cancel decisions remain unidentified.** No profitability or general maker-behaviour claim follows from this small, filled-order-selected sample.

## What we established

The deferred on-chain cache from consumed Oct 23 contains the full signed order tuple. We decoded its historical operator wrapper, recomputed the standard exchange EIP-712 order hash, and required equality with each successful receipt's order hash, maker, asset IDs and role. Thus identity does not rely on address/price coincidence or event ordering.

As of 2026-11-24 01:30Z, the original Oct23 14:00–14:15 cache including its saved padding provides:

| Evidence population | Result |
|---|---:|
| Cached transactions decoded | 8,255 / 8,255 |
| Known-token fill events bound by exact identity | 19,671 / 19,671 |
| Distinct signed orders | 13,834 |
| Orders with at least one maker execution | 6,537 |
| Existing volume-ranked top-three maker orders | 836 |
| Top-three orders filled in multiple transactions | 430 |
| Top-three orders whose observed making fully exhausts the signed budget | 350 |

The ranking is the handoff's Oct23 volume ranking. These are observations on an already-consumed ranking day, not ex-ante validation. The maker-order group includes any order with a maker execution; **37 orders overall appear in both roles**, so that group is not an exclusively passive-fill population. The full tables retain every role and status. Exact input/output hashes are in v3/manifest.json and v3/summary.json.

**Source distinction:** original signed terms are now available for these executed orders. Original *submission quantity visible on the book*, actual acceptance time, amendments, private pulls and continuously maintained queue position are not established. Repeated executions under one signed hash do not rule out intervening off-chain cancellation/reposting with the same hash.

## Directly updating the six handoff chains

We kept the previously selected six orders unchanged, identified their **14 existing settlement transactions**, and made only read-only transaction/receipt lookups. All six are on consumed Oct24, in three markets. Full signed identities and raw integer fills reconcile; the original exported REST maker quantities also agree. No new market/day was selected by an outcome. Raw responses and their hashes are retained.

| Existing chain / maker rank | Signed quote size, shares | Observed filled shares | Fill transactions | Terminal-budget evidence |
|---|---:|---:|---:|---|
| 1 / rank 1 | 30 | 30 | 1 | Fully exhausted |
| 2 / rank 1 | **30** | **6.6666** | 1 | Observed budget incomplete |
| 3 / rank 2 | 50 | 50 | 3 | Fully exhausted |
| 4 / rank 2 | 50 | 50 | 1 | Fully exhausted |
| 5 / rank 3 | **220** | **220** | **7** | Fully exhausted |
| 6 / rank 3 | 10 | 10 | 1 | Fully exhausted |

For chains1/3/4/5/6, the old `EX|PL` alternative “a larger order left an unfilled remainder” is ruled out **at completion of the recorded fills**: the actual signed making budget is fully consumed. This does not rule out a temporary pull before eventual completion, nor timestamp the book disappearance. Do not continue calling these final endings evidence of a discretionary cancellation.

For chain2, the old6.6666-share lower bound becomes a30-share signed quote size. The14 selected transactions show only6.6666shares filled. There is23.3334shares of quote-equivalent unobserved budget at its fixed signed price; this is **an upper bound, not a proven live remainder**. Unseen executions or off-chain withdrawal remain possible. A subsequent different order still is not automatically a cancel/reprice replacement.

Chain5 is concrete evidence that a top maker used one220-share signed order across seven executions. Chain3 similarly used50shares across three. These examples support investigating repeated-fill economics and replenishment; they do **not** establish that220shares is optimal, that larger orders generally profit, or that the maker held queue front continuously. The previous order-specific FIFO economics and their limitations stay unchanged.

## Cancellation evidence and clocks

The cached successful receipts contain **124 on-chain OrderCancelled events,124 distinct hashes**. **Zero** join to the decoded known-token order population. These unmatched events are exchange-wide cache observations, not124 known BTC cancels or124 top-maker pulls. Their absence from the joined population says nothing about off-chain cancel frequency. Even a joined on-chain cancel can follow exhaustion and would not prove a positive remaining budget.

Fills and cancels retain block number, transaction index and log index. Block time / settlement span is not a private decision time or continuous resting duration. The wrapper's trailing `created_at` text is deliberately uninterpreted; do not use it as an order-submission clock without separate validation.

## Verification and provenance

- Synthetic controls: official type/domain/order-hash vector; same-address/same-price orders with distinct salts; maker/taker identity; malformed offsets/signatures; wrong hash/amount/expiry/owner/assets/role; impossible cumulative fill; partial-not-cancel and repeated-events-not-repeated-transactions.
- Independent offline recount: all13,834 orders /19,671 raw events, signed hashes and integer budgets,124cancel events. Four planted corruptions refused (original amount, unsupported cancel grade, span, omitted event).
- Independent six-chain recount: all6orders /14raw events against saved RPC receipts. Four corruptions refused (size, exhaustion class, unsupported cancel, missing event). RPC controls verify requested transaction identity and complete response membership.
- V1 correctly refused the historical wrapper selector; v2 correctly refused its non-word-aligned suffix under an overly strict whole-payload check. Both failures remain as superseded receipts. V3 validates bounded order arguments and padded signatures while permitting the opaque wrapper suffix. Every admitted historical tuple then matches its emitted full hash. A mistyped selftest path failed before loading anything and was retried; it remains in run_light.log.
- The official [order-signing schema and reference vector](https://dev.predict.fun/resources/order-signing) and [SDK ABI](https://github.com/PredictDotFun/sdk-python/blob/main/src/predict_sdk/abis/CTFExchange.json) guide the field decoder. Applicability to historical data is established by the actual receipt-hash reconciliations, not by assuming today's wrapper ABI applied in October.

All result checks ran through PF_SLOT=codex. Recorded task ledger after the checks: **1.9097 /2.00core-hours**, line **5.2204 /5.30**; including standing0.05reserve:1.9597 /5.2704. The packet added approximately0.0024core-hours, below its0.030cap. Tags `mt_bridge_*` are counted automatically; do not add a separate review charge for them. No pushes, untouched-day test, model fitting, strategy replay or trading.

## One next priority

**Attach signed terms to a fixed sample of the existing cancel-compatible terminal-order cohort, then reclassify its endings before testing their market-condition logic.** This directly addresses the most consequential remaining gap: an exhausted order and a discretionary pull must not share the same action label.

Use the already-consumed Oct24–27 BOOK markets and existing top-three order identities. Before new lookups, freeze a deterministic hash-ranked sample of up to8orders per day×rank (up to96; retain empty/thin cells) from the old cancel-compatible cohort, plus its complete recorded same-hash transaction list. Do not select by P&L, original size, later market movement or newly decoded status. Obtain original signed amounts/expiry; reconcile all recorded fills in either role; retain explicit missing-transaction/source statuses. Separate (1) observed exhaustion, (2) explicit linked on-chain cancellation if present, and (3) unresolved remainder/expiry/withdrawal. None of the third group is automatically a pull. Regrade successor links as exhaustion-followed-by-new-order versus unresolved replacement, retaining overlap and timing ambiguity.

First deliverable is the identity-bound reclassification table, coverage by day/rank, and superseding chain labels. Preserve the old action-condition associations as estimates on their old population; do not silently claim they describe confirmed pulls. Only after the new action labels have adequate coverage should a separate frozen conditional rule be evaluated. Do not repeat anonymous transition tuning or choose a larger-size strategy from the220-share example.

Budget the next packet at<=0.025core-hours all-in including review; prepare its transaction-count/projection gate before any acquisition and stop if it does not fit. On the current ledger plus the standing reserve this fits the existing2.00 /5.30caps. The USER's current continuation instruction authorizes this bounded, same-consumed-day identification work; no further confirmation is needed within scope. New days, untouched validation, live use or a larger resource envelope remain separate decisions.

**Largest remaining limitation:** even exact signed sizes and exhaustion labels do not supply private submission/cancel clocks or continuous queue history. We have narrowed real alternatives; we have not recovered the makers' complete decision policy.
