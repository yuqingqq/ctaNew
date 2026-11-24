# MAKER-TRACE-RECONSTRUCTION — conclusion record (how the top makers make money, and the logic of their actions)

2026-11-24, session aea5bfb9. Recorded at the USER's request ("Record first") after review 0742bf95e closed the same-data identification line.

**What this is:** a consolidation, not a new result. Every number is copied from a committed receipt named beside it; the receipts and their reviews remain the authority.

**Scope:**
- the ex-ante top makers on Predict.fun BTC 5-minute markets;
- consumed days Oct 24–27, 2026 (19 BOOK markets for every book-based study);
- public data only: REST matches, WS full-depth books, the Binance BTC grid;
- 4 days, so point estimates only, no intervals, nothing validated on untouched days.

## 1. Where the money is (economic decomposition)

**Population, Oct 24–27:**

| Group | Net | Round trip | Residual | Fees | Maker legs | Taker legs |
|---|---|---|---|---|---|---|
| Ex-ante top 10 | **+$35,586** | +$130,789 | −$72,444 | −$22,759 | +$7,509 | +$28,077 |
| Other makers (2,943) | +$110,136 | +$356,551 | −$168,077 | −$78,338 | | |
| Pure takers (18,346) | **−$427,378** | | | −$180,559 | | |

Source: `v2/report.md`.

**Reading:** the top makers earn on **completed round trips** and give much of it back on **inventory held to resolution**. Takers as a group pay.

**Completed closes by entry → exit role** (c/share, net of fees; `v2/report_closes.md`):

| Entry → exit | Rank 1 | Rank 2 | Rank 3 | Top 10 |
|---|---|---|---|---|
| maker → maker | +2.87 | +1.64 | +1.70 | **+1.93** |
| taker → maker | +4.02 | +2.87 | +1.96 | **+3.95** |
| maker → taker | −2.87 | −6.69 | −6.85 | −3.18 |
| taker → taker | −1.40 | −6.95 | −4.33 | −1.59 |

Exits made passively (as maker) earn; exits made by taking cost. Each role cell is selected, so its causes are not identified (`report_interpretation_v3.md`).

**Our strategy vs them, same 19 markets and intervals** (`v2/report_matched_gap.md`, qualified by `report_matched_gap_interpretation_v2.md`):
- Our completed round trips lose about −1.8 to −2.1 c/share, where the top makers' earn +0.75 to +3.39 c/share. The reading is `our_turnover_loses` for every comparison party.
- This compares our simulator's fills with recorded fills, under different clocks.
- Which control decisions cause the gap is unmeasured. The earlier sentence "fixing our exposure would not close it" was withdrawn.

## 2. The logic of their actions (what public data supports)

Every action was tested against all the moments it did not happen, with the decision rules frozen before computation.

| Action | Established observable association | Grade | Receipt |
|---|---|---|---|
| **Take** | Large inventory (> 200 sh) roughly **halves the seconds containing same-direction (adding) takes**: 0.44–0.58×, in all six clock / ledger views. | robust (lower tail, descriptive) | v3, v7 |
| | **Opposing takes follow bursts of their own maker fills** (≥ 3 in 5 s): 1.42× pooled, 18/19 markets. | timing-sensitive | v3 |
| | **Taking falls sharply in the last 30 s**: opposing 0.44×, adding 0.03×. | robust | v3 |
| | Inventory size does NOT raise opposing takes. The trigger is **withdrawn** (0.91× pooled, ambiguity-aware). | declared decision | v3 |
| | No BTC-direction association passed the screen: no stop-loss and no take-profit shown. | not supported / insufficient | v2, v3 |
| **Pull** (cancel-compatible endings) | Adverse BTC within the last second is enriched under both reference families on all 4 days (+9.06 pp, fill-anchored). | timing-sensitive; pull vs exhaustion tied | v2 |
| **Post** (fresh price levels, market layer) | New levels follow a wide spread (1.36×, 16/16) and a thin own best level (1.12×, 17/19). Deep placements surge in the last 30 s (7.60×). | supported associations; 96.5% of creators unknown | v4 |
| **Reprice after a one-side retreat** (3,796) | The next detected change is usually **the other side improving** (1,853). Same-side response occurs in 26.3% (exact restoration 19.8%). | census | v5–v7 |
| | Same-side response is **more likely in the last 30 s (+14.8 pts)** and **less likely when that side's best level was thin (−9.7 pts)**. | adequate, consistent on 4/4 days and both clocks | v6–v7 |
| | BTC-against is sign-consistent only on the matched population. BTC-with and fair value are inconsistent. | caveated / inconsistent | v7 |
| **Queue front** | Front holders cannot be certified: masked cancel / replace inside a book frame is always possible. Whether the top 3 win the front more than their fill share is **unidentified** (2.1–35.3% vs 11.5%). | unidentified | v4 |

**Who does it:** most quote changes cannot be tied to a maker. 96.5% of new levels and 91% of retreat responses have no candidate owner, and only 12 of 999 responses link to a top-3 maker. Linked restoring orders show no profit within the window (23 exact orders, −$9.65 pooled; 21 exact restorations −$1.51).

## 3. Claims made and withdrawn along the way (kept so they are not repeated)

| Claim | Withdrawn in favour of |
|---|---|
| "≥ 70% of filled orders rested ≤ 10 s" | 940 / 4,149 bound-constrained; the rest unknown |
| "takes follow BTC against the position" | the statistic was against the take side |
| "inventory size is the take trigger" | the classification bias at small positions |
| "fast reaction within ~1 s" / "answered within milliseconds" / "54 ms latency" | these are frame endpoints, not latencies |
| "they do NOT win the front more often" | unidentified |
| "the front pays" | not a queue effect |
| "55% posted with no queue" | an endpoint observation only |
| "quotes move mostly as a whole" | 57% of improvement creations fall in two-side intervals, while one-side changed pairs outnumber two-side ones |
| "restoration +14.8 pts" | same-side response, all subtypes |
| "BTC gives no consistent direction" | BTC-against is matched-consistent with caveats |
| the inventory prescription in the matched gap | unmeasured |

## 4. Identification limit

These saved public records leave most maker identities, posting and cancel times, original sizes and queue histories **tied across compatible explanations**. That is a limit of this information, not proof that every source is uninformative.

More days of the same format can test the observable associations. They cannot recover an absent owner-tagged cancel message.

## 5. Decisions open for the USER (review 0742bf95e)

1. **Apply:** test the established patterns in our own simulator as a separate, planned strategy task. Candidates: end-of-market behaviour, opposing takes after fill bursts, no adding when inventory is large, the refill dynamics of a thin side.
2. **Validate:** check the key associations on untouched later days. This consumes held-out days.
3. **Calibrate:** run the public-data reconstruction blind on our own order logs. This measures reconstruction accuracy for our actions only.
4. **Lifecycle evidence:** a maker-attributed order-lifecycle source, which does not exist for competitors.
5. **Close:** keep the findings as they are.

## Receipts (all under `reviews/maker_trace_20261122/`)

| Receipt | Content |
|---|---|
| `v2/report.md` | economics |
| `v2/report_closes.md`, `v2/report_interpretation_v3.md` | closes |
| `v2/report_matched_gap.md`, `v2/report_matched_gap_interpretation_v2.md` | our gap |
| `v2/report_logic_chains*.md` (v1 → v7) | the action-logic line |

The contracts are `*_contract.md` in this directory. The Codex reviews are under `reviews/automatic/20261123_*`; the last is 0742bf95e.

**Task CPU ledger:** 1.9073 of 2.00 core-hours (`tools/cpu_ledger.py`).
