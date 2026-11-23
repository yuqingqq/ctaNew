# MAKER-TRACE v2 — completed-close attribution and identifiability control (review 567f741d9 next priority)

2026-11-23, session aea5bfb9.

**Provenance:**
- **Frozen tools:** amendment 6 `cddafa895`, plus fix 6b `e2201c258`. The frozen run first refused on its own cross-group guard, before any write; the float-cumsum cause was fixed with a selftest case.
- **Run:** `mt_x_closes_v2_r1` (7.2 CPU s, 513 MiB). All 15 reconciliation checks against econ pass: close P&L + residual = econ gross; matched shares = econ `mq` / `rq`.
- **Outputs:** `closes/`. The headline is `answer.json`, by role pair `closes_summary.csv`, the residuals `residual_summary.csv`, and the horizon cohorts `cohorts.csv`.

**Scope:** 4 consumed days, point estimates, descriptive. The clock is the FIFO clock: settlement-ID creation, an upper bound of each match.

## Completed closes by role pair (Oct 24–27 pooled; net of fees at the fill price; c / share)

| Entry → exit | Rank 1 | Rank 2 | Rank 3 | Top 10 | Top-10 shares | Median hold (top 10) |
|---|---|---|---|---|---|---|
| maker → maker | **+2.87** | **+1.64** | **+1.70** | **+1.93** | 6.20 M | 19.6 s |
| taker → maker | **+4.02** | **+2.87** | **+1.96** | **+3.95** | 1.01 M | 16.2 s |
| maker → taker | −2.87 | −6.69 | −6.85 | −3.18 | 1.23 M | 17.6 s |
| taker → taker | −1.40 | −6.95 | −4.33 | −1.59 | 0.72 M | 14.4 s |
| **all closes** | +0.66 | +0.60 | +1.19 | +1.19 | 9.16 M | 18.5 s |

**Totals for the top 10:**

| Entry → exit | Net |
|---|---|
| maker → maker | +119,870 |
| taker → maker | +39,887 |
| maker → taker | −39,182 |
| taker → taker | −11,429 |
| **all closes** | **+109,146** |

**Reading the table:**
- Closes that **exit passively**, whether entered as maker or taker, earn.
- Closes that **exit by taking** lose, because they pay the spread and the taker fee to unwind.
- This is per closed share. The role attribution is not independent strategy value, because an exit-by-take is how inventory gets flattened, and it is coupled to the maker fills that created that inventory.

## Is good turnover lost to residual exposure? (`answer.json`)

| Address | Close net (c / sh, USD) | Residual to resolution | Econ net | Max \|position\| (recorded fills) |
|---|---|---|---|---|
| Rank 1 | +0.66, +18,711 | −8,367 | +9,706 | 5,126 sh |
| Rank 2 | +0.60, +6,436 | −13,974 | **−7,556** | 3,423 sh |
| Rank 3 | +1.19, +14,998 | −10,499 | +4,491 | 3,000 sh |

All three read **`good_turnover_lost_to_residual`** pooled.

**Per address-day:**
- **Top 3:** 10 of 12 are `good_turnover_lost_to_residual`; 2 are `turnover_itself_poor` (rank 2 on Oct 24, rank 3 on Oct 25).
- **Top 10:** 24 of 40 are good turnover lost to the residual; 10 are good turnover with a non-negative residual; 4 are poor turnover; 2 have no rows.
- No reading flips between the two fee conventions.

**Computed answer:** the top makers' completed round trips are profitable, at about +0.6 to +1.2 c/share net, mostly from passive exits. The unclosed inventory then gives back a large share of it: 45% for rank 1, more than all of it for rank 2, and 70% for rank 3. Their recorded position reaches 3,000–5,100 shares in a market.

## Identifiability control (`tools/decision_identifiability_control.py`; run `mt_dec_identifiability`)

Two synthetic command histories yield byte-identical public records and 6,005 book frames. Run through the frozen lifecycle and decision code, they give identical outputs in all 19 files.

**Unidentified from public data:**
- an order's true size (5 vs 50 shares);
- whether a remainder was cancelled or filled;
- another maker's posting time and queue place;
- a 40 ms posting shift;
- an unseen cancelled order.

**Controls:** the falsifier (one visible order changes 7 outputs) and the frozen positive controls (78 / 78) pass. Every posting-decision statement of the decision section is limited accordingly (`report_interpretation_v2.md`).
