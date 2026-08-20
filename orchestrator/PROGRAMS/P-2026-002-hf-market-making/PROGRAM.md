# P-2026-002: HF Market Making on Crypto Perps

## Goal

Determine whether a non-colocated solo operator can run a profitable high-frequency
market-making strategy on crypto perpetual futures (Binance USDM first; Hyperliquid
as the candidate execution venue given its maker-rebate ladder), and if yes, build it
up through simulation to a small live paper test. The strategy concept: estimate a
flow-aware fair price, post bid/ask around it, and earn spread + rebate net of
adverse selection, inventory risk, and funding.

The hypothesis is **not assumed** — the program runs a pre-registered experiment
ladder (E0–E5) with kill gates. The prior from P-2026-001 research is that costs
kill most free-data edges at retail tiers; here the maker side of the fee schedule
and wide-spread mid-caps are the candidate niche, and that arithmetic is tested
before any model is built.

## Background (from session 2026-08-19 literature sweep)

- Fair value from book+flow = Stoikov micro-price / OFI; flow impact = propagator
  models. The user's "mid + simulated flow impact" idea maps onto these.
- Quoting = GLFT closed forms + Cartea-Wang alpha-signal drift + funding-aware
  skew (perp-specific). RL deferred — simulator-quality-limited.
- The loss term is adverse selection measured by fill-conditional markout, not
  "impermanent loss" (that is the AMM analogue, LVR).
- Fills are endogenous: you are filled precisely when flow is against you. All
  evaluation is markout-based, never gross spread capture.
- On majors the spread is pinned at ~1 tick (~0.01 bps on BTC) vs ~2 bps maker
  fee at VIP-0 — touch-quoting majors at retail tiers is off by orders of
  magnitude. The candidate niche is wide-spread mid-caps and/or rebate venues.
- No free historical futures L2 at MM resolution exists (bookDepth is 30s bands);
  live collection of bookTicker + depth20@100ms + aggTrade starts at E0.

## Agent pipeline (session structure)

```
Phase R  4 parallel research agents → live/mm_research/R1..R4 briefs
Phase S  review agent → STRATEGY_SKETCH.md (component architecture)
Phase D  experiment-design agent → EXPERIMENT_PLAN.md (prereg ladder + gates)
Phase I  implementation → E0 collector (done first — calendar-time long pole) + E1 scan
Phase V  adversarial review agent → audit implementation + first results
```

## Experiment ladder (detail in live/mm_research/EXPERIMENT_PLAN.md once written)

| ID | Question | Data | Kill gate (sketch — prereg in plan doc) |
|---|---|---|---|
| E0 | collector infra | live WS | n/a (infrastructure) |
| E1 | which symbols have spread economics that clear fees? | Vision tick aggTrades | no symbol clears fee+adverse-selection floor → program stops or pivots venue |
| E2 | does a naive touch-quoter's markout eat the spread? | collected L2 + trades | markout-adjusted spread ≤ 0 everywhere → need signal or die |
| E3 | do microprice/OFI/flow signals predict enough to fix E2? | collected L2 | signal lift doesn't flip E2 negative cells → stop |
| E4 | full sim (hftbacktest queue models) net of latency | collected L2 | net PnL ≤ 0 under pessimistic queue model → stop |
| E5 | small live paper test | live | tracking error vs sim; go/no-go for capital |

## Scope decisions (defaults; redirect to override)

| Decision | Default | Rationale |
|---|---|---|
| Venue for research data | Binance USDM (free tick aggTrades + live WS L2) | Data access; largest venue |
| Candidate execution venue | Hyperliquid (maker rebates, no colocation moat) — decided at E1/E4 | Binance VIP ladder unreachable for solo size |
| Historical L2 | Forward-collect from 2026-08-19; Tardis purchase deferred until E1 passes | Don't spend before the arithmetic clears |
| Simulator | hftbacktest (queue + latency models) | Standard OSS; avoids optimistic fill models |
| Code location | live/mm_research/ | Research code, matches repo convention |
| Evaluation | markout curves + realized-spread decomposition, block-bootstrap CIs, day-clustered | Repo methodology standards |

## Non-goals

- No live order placement in this program (paper/sim only; E5 is paper).
- No colocation/racing at the touch on majors — explicitly conceded to HFT firms.
- No RL until a closed-form + signal baseline exists and a validated simulator exists.
