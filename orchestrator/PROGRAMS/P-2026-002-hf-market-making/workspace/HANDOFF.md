# HANDOFF — P-2026-002 HF Market Making

Updated: 2026-08-19 end of session 1. **E1 COMPLETE — program is overlay-only.**

## Session 1 outcome (full pipeline ran: R → S → D → I → V)

- **Docs** (all in `live/mm_research/`): R1–R4 research briefs →
  STRATEGY_SKETCH.md (canonical architecture, 7 components, variants A/B) →
  EXPERIMENT_PLAN.md (pre-registered E1–E5 ladder) → E1_CODE_REVIEW.md (blind
  code audit + results audit) → **E1_RESULTS.md (the verdict — read this
  first)**.
- **E1-B (standalone Binance MM): no real pass.** Majors NEGATIVE pre-fee
  (BTC −0.19, ETH −0.23 bps at 30 s). ADA's +2.44 screen pass is an H1
  fat-tick artifact: notional-weighted it is **−0.32 bps** (small prints earn
  the half-tick, the dollars are adversely selected; same flip on every
  wide-tick name). Pre-read amendment for E1x/E2.0: gate quantity =
  notional-weighted rs on true mids.
- **E1-A (passive-execution overlay for the XS book): PASS, audit-robust.**
  T_p=600 s bracket: touch 3.45 [3.11,3.79] / sweep 6.26 [5.76,6.75] ≤ 8 bps
  capstone threshold; stale-shadow 7.20; excl-ICP 6.15; per-symbol max 7.53.
  Still maker-optimistic (H1; no queue position) — E2-A/E4-A decide for real.
- **Infra**: collect_hf.py LIVE since ~12:45 UTC (16 syms × bookTicker +
  depth20@100ms + trade; @aggTrade dead on URL-subscribed fstream — @trade
  used; combined /stream?streams= endpoint required; fapi REST geo-blocked
  from this box → Tokyo VPS before any order). 31 d tick aggTrades on disk.

## Next steps (in order)

1. **E2.0 + E2/E2-A when 14 d of L2 exist (~2026-09-03)**: true-mid recompute
   (notional-weighted, per amendment) voids/settles ADA; E2-A resolves the
   overlay bracket + the ICP cell with real books under the queue-model
   bracket (RiskAverse vs ProbQueue-f3; sign-flip = fail).
2. **E1x (optional, ADA only)**: 12-month quarterly confirmation with
   notional-weighted gate + fixed tick_size(). Expectation: kill. Low priority
   given E2.0 supersedes.
3. **Hyperliquid forward collection: RUNNING since 2026-08-19 ~13:20 UTC**
   (`live/mm_research/collect_hl.py` → `data/mm_hf/hl_raw/`, log
   `data/mm_hf/hl_collector.log`). bbo + l2Book(top-10) + trades, 16 coins
   (Binance-pilot equivalents; all HL-listed). HL market-data WS and /info
   are NOT geo-blocked from this box. App-level ping required ({"method":
   "ping"}) — handled. Trade side field ("B"/"A") recorded raw; side
   semantics must be verified empirically vs prevailing bbo before the
   screen. HL E1-style screen (notional-weighted gate) readable ~2026-09-18.
   Restart: `nohup python3 live/mm_research/collect_hl.py >
   data/mm_hf/hl_collector.log 2>&1 &`
4. Keep the collector alive (restart cmd below); check heartbeats when
   session starts.

## Watch out for

- All E1 numbers are maker-optimistic upper bounds (H1 population-vs-marginal,
  H2 staleness) — fails final, passes provisional; only E4-pessimistic-queue
  establishes economic sign.
- Weighting matters more than anything: eq-weighted markout is a per-EVENT
  statistic; economics are per-DOLLAR. Never gate on eq-weighted again.
- Queue bracket rule everywhere: sign-flip across RiskAverse/ProbQueue = fail.
- The 16-sym pilot is not PIT (H5); conclusions attach to named symbols.
- Collector restart:
  `nohup python3 live/mm_research/collect_hf.py > data/mm_hf/collector.log 2>&1 &`
  then `pgrep -af collect_hf` + check heartbeat lines.
