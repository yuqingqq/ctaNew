# Research insight — iter-005 (ORTHOGONAL DATA: Deribit DVOL)

**Directive:** First orthogonal-data experiment. Approved data = Deribit BTC/ETH DVOL (forward-30d
implied-vol index), the *forward-looking* signal price/funding could not provide. Decisive
pre-analysis: **does DVOL LEAD the strategy's pain (esp. the f5/f6 −57% DD episode)?** If yes and a
DVOL overlay beats a matched-magnitude placebo at ≥p95, propose it. If DVOL is coincident/lagging
(like the failed price features), return a clean NO-CANDIDATE.

**Outcome: NO-CANDIDATE.** DVOL does NOT lead the drawdown — it *lags/coincides*. No DVOL-conditioned
overlay beats the G4 matched-random-timing placebo (best p92 < p95). Honest negative result. Champion
stays = baseline (HL70 Calmar +1.68).

---

## Data & PIT setup
- `deribit_dvol.parquet`: BTC+ETH DVOL on 4h grid, 2024-12-01 → 2026-05-11 (covers the entire HL70
  backtest 2025-03-30 → 2026-05-06). Features: level, expanding-percentile (PIT), chg_1d, chg_3d.
- **PIT merge:** backward as-of merge with a **+4h (1-bar) publish lag** — feature at decision-time t
  uses DVOL as-of (t − 4h). 2405/2405 cycles matched. No look-ahead.
- Book PnL: `X121_percycle_HL70.parquet::pnl_base` (cumsum = +10,472 bps, maxDD −5,674 = byte-matches
  baseline). The −57% DD episode = **peak 2025-09-30 (cum +10,190) → trough 2025-12-24 (cum +4,516)**,
  spanning fold f5 into f6.

## 1. Does DVOL LEAD the pain? NO — it lags.
**Lead-lag test** (Spearman of PIT DVOL feature vs PAST-7d book PnL vs FUTURE-7d book PnL):

| feature | IC vs PAST-7d PnL | IC vs FUTURE-7d PnL |
|---|---|---|
| dvol_btc_pctile | **−0.259** | −0.228 |
| dvol_btc_chg_3d | −0.012 | −0.071 |

`dvol_btc_pctile` correlates **more strongly with what already happened (−0.26) than with what's
coming (−0.23)**. DVOL rises *in response to* drawdowns, it does not precede them — the same
coincident/lagging signature as the failed price-corr signal (iter-002). Confirmed visually around
the episode: at the cum peak (2025-09-30) DVOL pctile was **0.18 (low)**; it only climbed to 0.52
*as* the first −2,863 bps loss week (Sep30→Oct7) was happening, not before it.

**Forward-horizon IC (full overlap, PIT-lagged):**
- `dvol_btc_pctile`: next-1-cycle IC −0.023 (p=0.26, noise); **fwd-7d IC −0.228 (p<0.001)**.
- chg_3d / eth_pctile similar sign but weaker. So there IS an aggregate "high IV → weak next week"
  relation — but the next-CYCLE signal is pure noise, and the weekly signal is a **between-fold
  artifact, not a within-fold lead** (next point).

## 2. The aggregate −0.23 fwd-IC is a between-fold artifact, NOT a DD warning.
Per-fold IC of dvol_btc_pctile vs fwd-7d book PnL **sign-flips every fold**:

| fold | n | fwd-7d IC | book-7d mean | dvol_pctile mean |
|---|---|---|---|---|
| f2 | 340 | −0.531 | +678 | 0.16 |
| f3 | 348 | +0.495 | +339 | 0.04 |
| f4 | 348 | −0.138 | +161 | 0.07 |
| **f5** | 349 | **+0.387** | **−454** | 0.37 |
| f6 | 348 | −0.418 | +29 | 0.49 |
| f7 | 348 | +0.041 | −119 | 0.61 |
| f8 | 314 | −0.433 | +715 | 0.49 |

Critically, **inside f5 (the disaster fold) the IC is POSITIVE (+0.39)** — higher DVOL → *higher*
forward book PnL within the very fold we want to protect. The negative aggregate IC comes only from
the cross-fold pattern (high-DVOL folds f5–f8 happen to be lower-PnL folds), which is *not* a
forward-actionable timing signal. DVOL does not separate f5's losing cycles from the rest.

## 3. G4 pre-check — DVOL-conditioned de-gross overlay FAILS placebo.
Overlay tested: de-gross the new sleeve to 0.5× when `dvol_btc_pctile` is high (regime overlay,
matched-COUNT random-timing placebo, 300 seeds):

| variant | active | REAL Calmar | maxDD | placebo p95 / max | **real rank** |
|---|---|---|---|---|---|
| pctile ≥0.6 | 20% | +1.65 | −5,683 | +2.03 / +2.61 | **p56** |
| pctile ≥0.7 | 14% | +1.68 | −5,674 | +2.05 / +2.25 | **p56** |
| pctile ≥0.8 | 5% | +1.68 | −5,674 | +1.87 / +2.16 | **p59** |
| chg_3d top-25% (rising vol) | 25% | +2.01 | −4,734 | +2.25 | **p89** |
| chg_3d top-15% | 15% | +2.00 | −5,094 | +2.07 | **p92** |
| high & rising (pctile≥0.6 & chg_3d>0) | 11% | +1.70 | −5,646 | — | (Sh +1.95, DD unchanged) |

Best variant (rising-vol chg_3d) reaches Calmar +2.01 / maxDD −4,734 but ranks only **p89–p92 < the
p95 bar** — a *blindfolded* random de-gross of the same magnitude does as well or better. The
pctile-level overlays don't even move the maxDD (high-DVOL cycles aren't the loss cycles: flagged
cycles average **+0.79 / −0.10 bps** — neutral cycles, not the f5 losers).

**G4 pre-check FAILS.** Per the AGENT.md PRE-CHECK-G4 rule, do not propose it — it would be another
"run smaller / skip some" effect, not skill, and a guaranteed G4 casualty downstream.

---

## Synthesis
DVOL is the textbook forward-looking crash-fear signal, and the hope was it would LEAD the regime
DD that price/funding could not. It does not: implied vol **reacts** to realized stress (lags book
PnL more than it leads), and within the f5 disaster fold its sign is actually *positive*. The
aggregate "high IV → weak week" relation is a slow between-fold co-movement, not a forward DD-onset
detector, and no DVOL overlay clears the matched placebo. This is the same coincident/lagging wall
the price features hit (iter-001/002/004), now confirmed on the first orthogonal feed. The −57% DD
remains structural and not honestly reducible by any *level/percentile*-based regime gate.

## Recommendation (next data source)
DVOL **level/percentile** is coincident — exhausted. Two genuinely-different orthogonal angles
remain before declaring the data axis closed:
1. **Deribit options skew / 25Δ risk-reversal & term-structure** (data_sources.md "available, heavier
   fetch"). *Skew* (put-over-call premium) and a *backwardated term structure* are crowding/positioning
   signals distinct from the DVOL *level* — they can move while the level is still low (the Sep-2025
   pre-episode window where pctile was 0.18). Worth one lead-lag test of `rr_25d` / term-slope vs the
   f5 onset; only build if it actually leads (the DVOL level did not).
2. **Coinglass aggregated liquidations** (blocked — paid, needs human key). Forced-deleverage cascades
   are the mechanism *behind* a regime DD and could plausibly lead at the cycle scale where DVOL fails.

If the skew lead-lag test (no new key needed) also shows coincidence/lag, the orthogonal-data axis
should be considered closed for this construction and the loop escalated to human (the manual-research
conclusion: free + light-orthogonal data on the 4h horizon is a structural local optimum).

Scripts: `/tmp/iter005_*.py` (merge + lead-lag + per-fold IC + G4 pre-check). Merged panel at
`/tmp/iter005_merged.parquet`. Will move to research/ on request.
