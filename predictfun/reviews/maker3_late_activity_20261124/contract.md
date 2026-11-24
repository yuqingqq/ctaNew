# Maker 3 late-market activity — declared 2026-11-24

USER asks whether maker 3's active takes buy the ask or sell, and whether most fills occur in the last two minutes. Saved-data descriptive audit only. No new replay, model, day, hypothesis-selection claim or live use.

Fixed address: 0x57FbC891E89184DA071AE9D97E745531cd7F718a (Oct 23 ex-ante rank 3). Principal example market 3789484; also tabulate every captured actor leg in Oct 24–27, by day and pooled. The population is the existing canonical fills_valued.parquet, status=ok, non-self, positive quantity. Record excluded statuses and pre/post-window quantities. All 1,152 market registry rows remain represented; fill absence does not identify quote absence. Final Oct 27 market is boundary-censored; retain that status.

Time is settlement upper-bound time relative to opening, NOT observed posting/taking decision time. Five half-open minutes [0,60), ..., [240,300); last two minutes [180,300). Reconcile the named example separately to the existing dashboard [0,298]. Show both allocated maker-counterparty legs and unique (day,market,settlement-record,role) events: one taker execution can have multiple counterparty legs. Volume is shares, not marker count. No forecast/null/causal interpretation.

Join native taker direction from unique records.parquet; native Bid buys the indicated YES/NO token, native Ask sells it. A displayed YES-equivalent ask can be a NO buy, not a YES sale. Verify normalization, signer/hash and complete maker-counterparty leg quantity reconciliation. Publish all named-market takes, not selected examples.

Trade-only net flow starts at zero before the earliest captured market record, includes pre-open captured fills, and uses simultaneous settlement-time batches. At a take show net immediately before the batch, its signed change, and after the batch; separately flag mixed simultaneous other activity. This is NOT private inventory (unknown initial balances, token fees, transfers and merges). An opposing trade and an absolute reduction after crossing zero are different classifications. Do not infer liquidation motive from either.

Inputs are hashed. Independent check of named-market chart rows and native/raw maker-leg aggregation; positive cases and corruptions for direction, duplication, timing and volume. No economic P&L recomputation or strategy edits required. Small chart wording corrections may clarify existing actor-side labels without altering frozen data.

Budget: at most 90 CPU seconds (0.025 core-hours) for extraction/checks/publication, wall <30min, RSS <1.5GiB. run_light codex only, absolute scripts, each guarded. Standing USER 2026-11-23 CPU authorization (maker_trace/PENDING.md) permits this bounded top-up without asking again: task cap 2.10, line 5.40 including standing 0.05 reserve; current conservative task~2.00906/line~5.31976. Charge all attempts. Preserve other sessions' changes; local commits only.
