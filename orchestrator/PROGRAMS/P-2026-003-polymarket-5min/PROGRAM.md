# P-2026-003: Polymarket Crypto 5-min Markets

## Goal

Determine whether the P-2026-002 fair-value machinery (flow-aware fair value
from HF Binance data) prices Polymarket's recurring crypto 5-minute binary
markets ("{BTC,ETH,SOL,XRP} Up or Down", 300 s windows) better than their own
order book — and if yes, whether that mispricing is monetizable as (a) taker
against stale quotes and/or (b) maker on the binary CLOB with a fair-value
center. Research + paper only in this program; no live trading.

## The idea (transfer from P-2026-002)

A 5-min up/down binary is a digital option on the window's price change. Fair
probability from our side:

```
P(up) ≈ Φ( (F_t − K + μ̂·τ_rem) / (σ̂·√τ_rem) )
F_t   = Binance microprice/flow fair value (components a-b of P-2026-002)
K     = window open (strike);  τ_rem = time to window end
σ̂     = short-horizon vol from our 100 ms feed;  μ̂ = flow-implied drift
```

We already stream the signal side (collect_hf.py: bookTicker + depth20@100ms +
trade for the same 4 coins). The market side (this program's collector) records
the binary book at message level. Edge hypotheses, in falsifiability order:
1. **Latency/staleness**: binary quotes lag the underlying by enough to cross
   profitably near window end (taker; needs fee ≤ mispricing).
2. **Digital-pricing quality**: book prices are systematically miscalibrated vs
   Φ(·) even without latency (e.g. long-shot bias inside 5-min windows).
3. **MM on the binary**: quote both sides around our P(up) with GLFT-style
   inventory control; adverse selection = informed flow from other
   Binance-watchers (measurable by markout in probability space).

## Open questions the data answers first

- Fee structure: `fee_rate_bps` observed per trade in `last_trade_price`
  messages (seen non-empty 2026-08-19); taker/maker fee schedule on these
  markets decides hypotheses 1/3 viability. Measure, don't assume.
- Resolution source + strike definition: extracted from market `description`
  (recorded); which oracle (Pyth/Chainlink), and open/close timestamp
  conventions — needed for exact strike K and for resolution-risk near ties.
- Liquidity: typical book depth ($), trade sizes, and whether flow is
  concentrated in the last seconds of the window.
- Calibration ground truth: resolutions.jsonl accumulates (outcome, final book
  price) pairs — realized calibration curve for free.

## Kill gates (sketch; pre-register properly before the first experiment)

- G1 (economics): median half-spread + fee on the binary book vs the
  |P̂(up) − book| mispricing distribution — if fees+spread ≥ p95 mispricing,
  taker thesis dead.
- G2 (calibration): our Φ-model must beat the book's own price as a predictor
  of resolution (log-loss / Brier, out-of-sample by day) — else no edge exists
  to make markets around either.
- Evaluation discipline inherited from P-2026-002: day-clustered stats, block
  bootstrap, notional-weighted, no eq-weighted gates.

## Infrastructure

| Piece | Status |
|---|---|
| Market-side collector `live/pm_research/collect_pm.py` | this session |
| Signal-side Binance HF feed | running since 2026-08-19 (P-2026-002 E0) |
| Storage `data/pm_5min/` (markets.jsonl, resolutions.jsonl, raw/<day>/<slug>.jsonl.gz) | gitignored |
| Trading access (Polymarket CLOB API keys, US-access status) | out of scope until gates pass |

## Non-goals

No live orders; no UMA-dispute modeling; no non-crypto Polymarket markets;
no on-chain settlement infra.
