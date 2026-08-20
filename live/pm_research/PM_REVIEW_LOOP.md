# PM_REVIEW_LOOP — data-completeness review loop (PM-E0.5, P-2026-003)

Charter (user-initiated 2026-08-19): **before any experiment runs, loop an
adversarial review of the experiment design + collectors until the data layer
is provably complete.** Convergence = two consecutive iterations with zero
MUST-FIX findings. Protocol per iteration:

1. Review agent audits, against `PM_MM_PLAN.md` (esp. §2 model inputs, §4
   layers, §5 ladder):
   a. **Traceability matrix** — every symbol the model consumes (F_t, σ̂, μ̂,
      K=X_0, X_T, book state, q, fees, rebates, rewards params, resolution)
      mapped to a concrete collected series + file path + field, or flagged.
   b. **Collector correctness** — schemas, gaps, restart behavior, silent
      failure modes of collect_pm.py / collect_pm_prices.py / collect_hf.py.
   c. **Open data gaps** — currently: L3 identity (PM feed vs Chainlink TWAP
      vs Binance mirror), L4 settlement-truth recovery (on-chain reports /
      data.chain.link / API), program-docs snapshot, backfill enumeration.
   d. **Design hazards** the plan misses.
2. Findings triaged MUST-FIX / SHOULD-FIX / NOTED; orchestrator applies fixes
   (collector patches within the iteration; plan amendments logged).
3. Next iteration re-audits, including verification that prior fixes hold on
   REAL collected data (not just code reading).

Iteration log:

| iter | date | MUST-FIX | verdict doc |
|---|---|---|---|
| 1 | 2026-08-19 | 2 found, 2 fixed same-day | PM_REVIEW_ITER1.md |

Iteration 1 outcomes (fixes applied ~15:45 UTC):
- **L3 = MIRROR** of Binance SPOT last (100% string-match vs spot 1s kline,
  +4.6 bps vs our futures mid) — not settlement.
- **L4 SOLVED**: Chainlink RTDS TWAP relayed publicly on the SAME live-data WS,
  topics `crypto_prices_twap_sixty`/`_thirty` (1 s, 1e18-scaled, window_s
  field). Subscribed as of ~15:45. NO REPLAY exists — pre-15:45 truth is lost
  (paid backfill: pmdata.dev parquet since Aug-01, if ever needed). On-chain
  resolution txs carry payouts only, no prices. Synthetic X̂ from Binance:
  −5.4 bps basis (std 0.7) — unusable for near-tie truth, fine for coarse work.
- **MUST-FIX both applied**: (1) in-flight window resume + numbered gz shards
  (restart previously lost windows — verified against the 15:12 restart);
  (2) TWAP topic subscriptions.
- SHOULD-FIX applied: doge/bnb/hype coins added; per-market CLOB fee/rewards
  fields captured. Remaining for iter 2: fee-source conflict (docs 7% vs CLOB
  base_fee 1000 vs observed fee_rate_bps=0 on all 19,141 trades — resolve
  empirically per side), settlement boundary convention (X_T≥X_0 vs literal
  full-window-TWAP description reading — decide from collected TWAP stream vs
  actual winners), fix-verification on real post-fix data (restart test,
  shard concat, TWAP stream continuity), gap scan on prices CSVs.
- Positive iter-1 finding: `price_change` messages are LEVEL TOTALS (99.89%
  verified) → the E3 queue-bracket reconstruction is sound.

Iteration 2: run after ≥12 h of post-fix data (or next session start,
whichever later). Convergence rule unchanged.
