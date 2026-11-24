# SIGNED-ORDER-EVIDENCE-GATE-v1

2026-11-24. USER: “Can you take the handoffs and continue to identify actions and bridge the gap.” This resumes the evidence gate from review 0742bf95e, rather than another variant of the anonymous book-transition analysis. One bounded offline source-capability measurement; zero strategy/model candidates. Existing private/live code stays untouched.

## Scope and question

Use ONLY the previously cached `onchain/20261023_1400_1415` inputs, successful receipts, blocks and token map, including the cache's original padding. Oct 23 is already consumed; it is also the top-maker ranking day. Ranking here labels examples, never validates a performance claim. No untouched days, RPC fetch, tape replay, fitting, economic outcomes or live actions. The old 19-market Oct 24–27 findings are not extended to this slice.

The older decoder in `pf_chain_calldata.py` dropped salt/expiry/original quantities and joined fills on transaction/address/role. The new decoder must verify the EIP-712 hash of the **entire signed order** against the receipt's order hash and match asset IDs, maker and maker/taker role. Two same-address orders must remain separate. Decode all cached calls without choosing orders by size or result. Unsupported calls and every missing BTC event binding are counted; any unbound known-token fill fails the source gate. No fallback address-only join.

## Declared outputs and grades

- All successfully bound known-token orders, exact original signed amount fields, expiry, owner, role, and fill events. Quantities and reconciliation use integer wei. JSON/CSV output amounts are strings.
- `exhausted_by_observed_fills` only when all observed **making** amounts equal the original signed making budget. Buying consumes collateral; selling consumes shares. Signed share target and actual share receipts are separate because taker price improvement can differ.
- Otherwise `observed_budget_incomplete`: original budget minus observed making is an **upper bound**, not exact remaining quantity. Prior fills outside the cache may exist. Never infer a pull, expiry, continuous resting or non-exhaustion from this status.
- Distinguish repeated log events from repeated transactions. A repeated hash proves repeated executions of the same signed order; settlement span is not a resting-time or queue-priority measurement. Queue may have changed invisibly.
- Enumerate on-chain cancellation events in the saved successful receipts, with an explicit scoped denominator. Their absence says nothing about off-chain cancellation coverage.
- Summary for all, maker-role, and the existing volume-ranked top three. Predetermined examples: first 12 top-three rows in hash order. Complete tables remain under data/ with hashes; no large data committed.
- Gate PASS requires at least one bound BTC fill and zero unbound BTC fill events, plus synthetic/independent checks. PASS establishes a field source, not a profitable rule, a complete lifecycle source, or selection skill. No association null/interval applies to this mechanical identity gate.

## Verification and limits

Official schema/reference vector: [Predict order signing](https://dev.predict.fun/resources/order-signing) and [official SDK ABI](https://github.com/PredictDotFun/sdk-python/blob/main/src/predict_sdk/abis/CTFExchange.json), checked 2026-11-24. Historical receipt hashes, rather than current documentation alone, must validate applicability to the October cache. Only the cached standard exchange `0x8bc070bedab741406f4b1eb65a72bee27894b689` is supported. No silent use of other exchange domains.

Synthetic controls cover official known hash, two same-owner/same-price different salts, maker/taker roles, hash/owner/asset/expiry/amount corruption, malformed ABI, integer budget accounting, partial-not-cancel, repeated-events-not-repeated-transactions, and impossible overfill. Independent output recount must read raw receipt events and reconstruct membership and totals without the decoder. Refuse a deliberately corrupted original amount and an omitted event. Write outputs once; preserve failures.

Source and input hashes committed before the real decoding run. CPU budget <=0.030 core-hours all-in (including tests/checks), no more than 50 CPU seconds for the main gate; peak guard 1,000 MiB. Run via PF_SLOT=codex run_light.sh, tags `mt_bridge_*` counted automatically by the existing ledger. Pre-work task1.9073 / line5.2180; with .030 and standing .050 reserve =>1.9873 /5.2980, within2.00 /5.30. No cap change. Stop at budget rather than quietly enlarge scope.

## Next action decided from evidence

If identity fails, retain the refusal and locate the concrete schema/accounting disagreement; do not infer actions. If it passes, document exactly which gaps it narrows and the remaining ones. The next packet would attach signed terms to the existing consumed-day, preselected maker action chains and reclassify exhausted-versus-unresolved endings. A missing cache is a concrete acquisition dependency; no broad crawl or untouched-day use is authorized by this small gate. Do not restart anonymous association tuning.
