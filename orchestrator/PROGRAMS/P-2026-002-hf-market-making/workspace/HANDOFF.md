# HANDOFF — P-2026-002 HF Market Making

Updated: 2026-09-06T04:47:21Z (DA seat). **E2.0 RAN AND IS READ: ADA IS SETTLED
DEAD ON TRUE BOOKS, AND E1 IS VINDICATED AS A MEASUREMENT.** E1-B is now empty
with no pending cell. Read this section, then session 1's below it.

## READ FIRST — E2.0, 2026-09-06

### The verdict: `NOT_VOIDED+DEAD`, and the two legs say different things

**ADAUSDT, 16 admissible UTC days (2026-08-20..09-05), τ\* = 30 s (16 of 16
days fast, threshold 13), 2,077,916 sweeps.** Receipt
`data/mm_hf/e1/p002_e2_0_smoke_ADAUSDT__20260906T044455Z.json`
(sha256 `6942eb5ff574a42e`, carrying_commit `2cf2029`), declaration v2
`054ff4e7536291ee`.

| leg | quantity | threshold | result |
|---|---:|---:|---|
| **VOID** | Δ_rs(τ\*) on the intersection, eq-weighted | > +1.0 bps voids | **−0.0066 bps → does NOT void** |
| **SETTLE** | notional-weighted rs(τ\*) on true mids | < 1.8 bps kills | **−0.5552 bps → DEAD** |

**The proxy was fine.** Δ_rs is seven thousandths of a basis point: E1's
two-sided last-print mid agrees with the real book. *E1 measured correctly, and
the +2.44 was never a mid artifact.* **ADA dies on the WEIGHTING**, exactly as
the results audit predicted — now confirmed against real bookTicker rather than
inferred from prints.

**The kill is interval-robust.** CI95 `[−1.86, +0.12]`, entirely below the
1.8 bps fee. All four readable §1.5 gates fail:

| gate | bar | value | |
|---|---:|---:|---|
| 1 point ≥ fee + c_safe | 2.3 | −0.5552 | FAIL |
| 2 CI-lo ≥ fee | 1.8 | −1.8612 | FAIL |
| 3 positive on ≥ 70% of days | 12 of 16 | 6 of 16 | FAIL |
| 4 both sides > 0 | — | maker-bid −1.1462 / maker-ask +0.0834 | FAIL |

### The mechanism, monotone and per-dollar

Eq-weighted **+1.3639** against notional **−0.5552** — a **1.92 bps** weighting
gap (E1 measured 2.76 on its own window). Day-clustered rs by **notional
quintile**:

| quintile | n | notional | rs (notional-w) |
|---|---:|---:|---:|
| 0 smallest | 415,433 | $2.7 M | **+2.29** |
| 1 | 415,727 | $9.2 M | +1.98 |
| 2 | 415,577 | $39.4 M | +1.73 |
| 3 | 415,584 | $212 M | +0.98 |
| 4 largest | 415,593 | **$2,951 M** | **−0.69** |

**The top quintile carries 92% of the $3.21 bn.** Small prints earn the
half-tick; the dollars are adversely selected. Even the smallest quintile only
just clears the 2.3 gate, and it is 0.08% of the notional.

### What this does NOT settle — stated in the declaration, not discovered after

- **The window is 2026-08-20..09-05, not E1's 2026-07-18..08-17.** The two do
  not intersect (the collector started 08-19), so Δ_rs was computed with BOTH
  mids on the SAME collected days. A dead cell here is dead **on this period**.
- **H1 is measured by the notional weighting, not removed by it.** rs is still
  a population statistic over all sweeps. Only E2's queue replay puts a
  *marginal* order at the back of the queue.
- **No queue position, no own impact.** E2.0 can kill or fail to kill; it
  cannot establish economic sign.
- **8.60% of true-mid-valid events (178,659) the proxy could not see.** Δ_rs
  therefore speaks about the mid *on the events both mids see*; that share
  bounds the reach of the void reading.

### Two defects this step found

1. **The collected `@trade` stream carries zero-quantity prints** — rows of the
   form `trade_id,0,0`. **22,639 across ADA's 16 days.** A sweep made only of
   them collapses to Q = 0 and price 0, and `mo = −q(m₁−p)/p` explodes: the
   first real day came back at **4.7e12 bps** and the §1.3c identity check
   failed. **E1's Vision aggTrades contain none** (0 of 582,765 rows over 8 ADA
   days), so **E1's published numbers are not exposed** — this is a property of
   the surface E2.0 introduces. Excluded and counted; four falsifiers drive it
   both ways.
2. **`git checkout --detach` destroys the worktree's `data` symlink** and
   leaves a near-empty shell. The first smoke read **zero days and refused** —
   the instrument worked, but the standing refresh procedure silently undoes
   the fix. Restored twice this session. See the Q-DA row.

## E2-A is DECLARED — and two things need a ruling before it runs

`live/mm_research/declarations/p002_e2_a_declaration_v2.json`
(sha256 `6567a25f04d7fb89`, carrying_commit `0cbaba6`; v1 `405ddb7ab10486c2`
superseded in band, untouched), **no data touched under either**. v2 adds the
data-root discipline the E2.0 result review requires: the P-002 surface
resolves through the same imported resolver as P-003, every receipt records the
root and branch, and a result-bearing run off the canonical ledger REFUSES —
with the falsifier being a **partial** root (a real tape holding 2 days of 19),
which must refuse rather than report a smaller census.
Gate: `eff_RT <= 8 bps` under **RiskAverse** at T_p = 600 s, interval binding
on the PASS side. A bracket that **straddles** the threshold is a FAIL, never
averaged. **ProbQueue-f3 costing more than RiskAverse refutes the INSTRUMENT**,
not the symbol. ICP: above a **50% episode-skip bar** a symbol is UNRESOLVED
and excluded, aggregate reported both ways — the bar is declared before any
census and applies to all twelve (E1-A measured ICP at 72%).

> **RULING NEEDED (a): `hftbacktest` is NOT installed on this box.** The plan
> names it for the bracket. Installing a dependency is an environment change
> and not the seat's to make, so both queue models are declared in closed form
> and implemented in the runner — which makes their **correctness mine**. Each
> ships a falsifier, and the ordering property is a computed predicate. Either
> the direct implementation is accepted, the dependency is authorised, or E2-A
> waits.
>
> **RULING NEEDED (b): the XS rebalance notional does not exist.** "Depth-aware
> sizes at the XS book's actual rebalance notionals" needs a per-symbol
> notional this programme has never pinned — E1-A's episodes were explicitly
> *min-size, notional-free*. Its absence **REFUSES the size-aware arm** and
> reports the min-size arm labelled NOT the gate, because a min-size answer is
> E1-A's answer with a better fill model.

**The reviewer files on the declaration before any run.**

## Next steps (in order)

1. **E2.0 on the remaining 15 symbols.** The smoke was one symbol by dispatch.
   ADA cost **67 s wall / 657 MiB RSS** for 16 days at one CPU under 8G.
   **BTC is ~12× ADA's daily volume** (480 MB/day gz bookTicker vs 41 MB) — a
   resource check before it runs, and the cap is never raised: a symbol that
   exceeds it refuses. Expected reading: the E1 audit found the same eq→notional
   flip on SOL/DOGE/XRP/ATOM/ICP, so the 15 are a *confirmation* set, not a
   search for a survivor.
2. **E2-A (overlay bracket on real books)** — still the decision-bearing step
   for the programme's live track, and it resolves the fragile ICP cell.
   Untouched by this result.
3. **E2 proper** (hftbacktest queue replay, RiskAverse vs ProbQueue-f3).
4. **E1x is MOOT.** It existed to re-measure ADA's population statistic; E2.0
   supersedes it and the cell is dead.
5. **HL Variant-B screen** still ~2026-09-18.

## Watch out for

- **Never gate on eq-weighted.** This step is the second independent
  demonstration; the quintile table is the picture to show anyone who asks why.
- All E1-A numbers remain maker-optimistic upper bounds; only E4-pessimistic-
  queue establishes economic sign.
- The 16-sym pilot is not PIT (H5); conclusions attach to named symbols.
- Collector restart (do NOT restart without the coordinator — collector
  surface, R-110):
  `nohup python3 live/mm_research/collect_hf.py > data/mm_hf/collector.log 2>&1 &`

---


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
