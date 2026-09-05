# P-2026-003: Polymarket Crypto 5-min Markets

> **Current state (updated 2026-09-05T09:54:58Z): GATE 1F INPUT AUDIT REFUSED—
> NO OWNED EXECUTION EXPORT; GATE 1 REFUSED FOR UNAVAILABLE OWNED MAKER FEES;
> V2 TERMINALLY
> STOPPED AT 1/7; GATE 0 ONLY FULLY CLEARED; BROAD/HEAVY RUNS OFF.**
> The R-531 halt remains the
> last registered state; the later direct user instruction authorises the
> prospective plan, lightweight implementation and required one-window capped
> smokes but has not yet been entered
> in the coordinator register. This file is the original research charter, not
> live status. The latest
> economic evidence is one BTC development hour (12 windows, 4,315 fills), not
> validation or net profitability. The planned broad `V_oracle` survey failed
> before producing a result; no ~40-cell or 138-cell result exists. The Gate-0
> smoke is pipeline evidence only, not an economic result; overall v2 progress
> is 1 of 7 gates. The iid stateful sampler produced only 1 of 200 required
> exact matched draws in 4,000 proposals, so it published no null. The
> constrained exact-fiber replacement found 399 distinct states but failed its
> mixing bar (ESS 10.53 < 100). Overall v2 progress remains 1/7. The user has
> ratified a different sequential random action-quota estimand with a fixed
> 200-of-1,000 one-window gate and unchanged one-CPU/3 GiB ceiling. That gate
> also refused: only 16 proposals reached the actual issued-action quota and
> only 16 distinct action sets existed versus the fixed 50 minimum. The
> matched null is absent; this does not reopen any failed sampler or authorise
> Gate 2. New work is
> governed by `live/pm_research/plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN_V2.md`.
> All 17 current v2 module/wrapper batteries pass (182 checks total), and the
> 223-check parent suite passes under one-CPU/1-GiB/no-swap caps;
> the Gate-1c receipt has 11/12 current-tree identity matches, with only its
> prospectively hashed plan changed by the post-run result annotation; no named
> source file drifted. Full identity detail is disclosed in the handoff.
> Gate 1d now preserves the clustered score sequence and enumerates every
> cyclic phase, requiring at least 200 distinct exact-count joint phases before
> 200 uniform without-replacement full replays. It changes none of the prior
> failures and does not authorise Gate 2.
> The complete enumeration found 18 BUY and 40 SELL exact-count phases, 720
> joint assignments, and all fixed 200 distinct full replays passed identities.
> Gate 1e pinned and reproduced those phases. QR_SKEW_ONLY, treatment and all
> 200 controls passed every gross accounting/population identity, but all 202
> owned-order per-fill maker-fee ledgers are unavailable. Receipt
> `p003_v2_gate1_economics_smoke__20260905T052605Z.json`, sha256
> `e78fe495846cf22e834b63e04aea445cf1616563cb932a11f304d3a7ba2abd42`.
> All strategy nets and the matched decision statistic are null; no public
> taker fee or assumed zero was substituted. This is an input-identification
> refusal, not negative P&L. Gate 2 and Gates 3–6 did not start.
> The subsequent bounded acquisition audit found no
> `data/pm_5min/owned_execution/manifest.json`. Corrected receipt
> `p003_v2_gate1f_owned_source_audit__20260905T054941Z.json`, sha256
> `c99109943de37d37d2fc8358628640214d489752e96bb8ca4f86e144bf197f47`.
> Public raw/Tier-1 coverage cannot supply the missing owned order/ack/fill/fee
> join; decision metric remains null and Gate 2 remains off.
> Read `workspace/RESULTS.md` §0, then `workspace/HANDOFF.md`.

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
