# Evaluation Report — iter-011 (REACTIVE risk-control track)

**Verdict: ACCEPTABLE FOR DEPLOYMENT as a risk option** (with a stated trade-off).
This is **NOT** an alpha ADOPT and does **NOT** change the alpha champion (which stays = baseline,
Calmar +1.68). It is a characterized DD-vs-cost risk overlay, graded against the reactive-track gates
R1–R7 — the alpha G4≥p95 disqualifier does **not** apply here.

## The change
A reactive **equity-drawdown stop** (mechanical damage-control, NOT a forecast). De-gross the whole
held book to `g_floor = 0.40` when the strategy's OWN cumulative equity is **≥ X bps below its running
peak** (equity/peak/DD computed through t−1, PIT). Re-enter (gross→1) when equity heals **50%** of the
drawdown back toward the peak (and is above the trough — never buy back AT the trough) OR after **90
bars (~15d)** as a fail-safe. **Recommended X = 1,600 bps.** The threshold X is the only tuned
parameter; g_floor/heal/timeout are fixed policy.

Script: `research/convexity_portable_2026-05-20/scripts/X124_reactive_dd_stop.py` (canonical, gross
applied to positions BEFORE turnover/cost). Artifacts: `results/X124_tradeoff_curve.parquet`,
`results/X124_r4_const_degross.parquet`. All numbers below independently re-run and reconciled.

## Base reproduction (gross=1.0 == X117) — CLEAN
| universe @4.5bps | Sharpe | maxDD | Calmar | totPnL |
|---|---|---|---|---|
| HL70 | +1.93 | −5,674 | +1.68 | +10,472 | ← X117 EXACT |
| EXT | +0.87 | −4,953 | +0.66 | +15,448 |
| S44 | +1.84 | −4,170 | +2.10 | +25,620 |

## R-gate results

| # | Gate | Result | Number |
|---|---|---|---|
| **R1** | Look-ahead | **PASS** | Review PASS. Trigger reads equity/peak through t−1 only; gross fixed before pnl[t] realized; running peak is PIT cummax; re-entry uses realized trough/heal/timeout with `eq>trough` guard. No forward peek. |
| **R2** | Tail reduction | **PASS** | HL70 maxDD −5,674 → **−3,292 (−42.0%)** at X=1600/g_floor=0.40. Well above the ≥25–30% bar. |
| **R3** | Bounded cost | **PASS (stated)** | totPnL +10,472 → +7,914 (**−24.4% cost**). Sharpe +1.93→+1.77. **Calmar +1.68→+2.19 (IMPROVES).** 52.9% time at reduced gross, 15 round-trips/402d. Cost-robust: 23.2%/23.3%/24.4% at 1/3/4.5bps. |
| **R4** | Concentration vs constant-de-gross (honesty gate) | **MIXED → effectively FAIL the skill claim** | HL70: stop modestly beats constant at X=1200–2500 (STOP−CONST maxDD +150…+582 bps; +582 at X=1600). EXT: constant MATCHES/BEATS at almost every X (negative). S44: mixed. **R4-PLACEBO (200 seeds, matched %-time): real ranks HL70 p91, EXT p42–p52 — all < p95.** The tail-cap is **~proportional to exposure removed, NOT skill-selective.** |
| **R5** | Cross-episode + LOFO (DECISIVE) | **PASS** | X=1600 caps DD in **3/4 EXT episodes** (luna 60%, ftx 46%, 2024_summer 62%; 2025_q4 EXT-slice −0%, shallow — the real −57% lives on HL70, cut 42%). **Episode-LOFO (X=2000): ddRed +27.4/+30.1/+28.1/+30.1% dropping luna/ftx/2024/q4 — does NOT vanish dropping any episode.** First DD mechanism in the run to survive episode-LOFO. |
| **R6** | Nested-OOS of the threshold | **HL70 PASS, EXT FAIL** | HL70: forward ddRed **+21.8% at +2.6% cost** (chosen X drifts 1200–1600; OOS maxDD −5,674→−4,435, Sharpe +1.09→+1.19). EXT: forward ddRed **−7.3% at +44.8% cost** (X drifts to deep 2500–3000, barely fires). |
| **R7** | Re-entry sanity | **PASS** | 15 firings / 15 re-entries, heal-or-timeout, `eq>trough` guard (no buy-at-trough). g_floor=0.40>0 so equity heals while stopped → no frozen-equity permanent-kill pathology (the g_floor=0 / re-enter-at-new-high variant degenerates to a 90%-time-stopped 1-RT kill — REJECTED). |

## DD-vs-cost trade-off curve (HL70, g_floor=0.40, @4.5bps) — the deliverable
| X (bps) | maxDD | ddRed% | totPnL | totCost% | Sharpe | Calmar | %stop | RT | avgG |
|---|---|---|---|---|---|---|---|---|---|
| 800 | −3,115 | 45.1 | 4,208 | 59.8 | 1.22 | 1.23 | 76.7 | 27 | 0.54 |
| 1200 | −3,175 | 44.0 | 7,493 | 28.4 | 1.76 | 2.15 | 58.8 | 17 | 0.65 |
| **1600** | **−3,292** | **42.0** | **7,914** | **24.4** | **1.77** | **2.19** | 52.9 | 15 | 0.68 |
| 2000 | −3,606 | 36.5 | 6,731 | 35.7 | 1.50 | 1.70 | 52.5 | 15 | 0.68 |
| 2500 | −3,794 | 33.1 | 8,617 | 17.7 | 1.85 | 2.07 | 50.8 | 14 | 0.70 |
| 3000 | −4,314 | 24.0 | 8,084 | 22.8 | 1.72 | 1.71 | 51.3 | 14 | 0.69 |

Reading: deeper X removes less DD at less cost (a clean risk dial). X≈1,200–1,600 is the knee — ~42–44%
DD cut, ~24–28% cost, Calmar > base. Shallower X (800) just runs perpetually small (76% time stopped,
Sharpe collapses). **EXT** (X=1600): ddRed 36% / cost 39%. **S44** (X=1200): ddRed 41% / cost 38%,
Calmar +2.10→+2.23. The cut holds on every universe but costs more per unit of ddRed off HL70 (HL70 is
the favorable case — one deep contiguous −57% episode the stop catches cleanly).

## R4 — stop vs constant-de-gross of equal average exposure (the honesty comparison)
STOP_maxDD − CONST_maxDD (positive = the *triggered* stop caps the LEFT TAIL better than always
running at the same average gross):

| X | HL70 (avgG) | EXT | S44 |
|---|---|---|---|
| 800 | −51 (0.54) | −465 | +31 |
| 1200 | +499 (0.65) | −622 | +390 |
| **1600** | **+582 (0.68)** | **−511** | −120 |
| 2000 | +280 | −692 | +415 |
| 2500 | +150 | −7 | +377 |
| 3000 | −386 | +143 | +505 |

**Honest reading:** on HL70 the equity-triggered stop beats a flat constant-gross book by a modest
+150…+582 bps of maxDD across the knee — i.e. concentrating the de-gross in the deep episode buys a
little. On EXT, constant-de-gross **matches or beats** the stop at almost every threshold. The
R4-PLACEBO seals it: against a random de-gross of matched %-time, the real stop ranks **p91 (HL70) /
p42–p52 (EXT)** — below p95. **The tail-cap is essentially "run smaller specifically while underwater,"
roughly proportional to exposure removed; it is NOT a skillful tail-selector.** This is the *expected*
reactive-track outcome (a stop reacts to a DD already underway and cannot forecast), not a bug.

## Why this is DEPLOYABLE despite R4/R6 caveats
Per the reactive-track verdict rule (contract §TWO TRACKS): a stop is "ACCEPTABLE FOR DEPLOYMENT" iff
**R1/R2/R5/R6 hold and R3 cost is acceptable.** On the production universe (HL70) all four hold:
- R1 PIT-clean, R2 −42% maxDD, R3 bounded +24% cost with **Calmar improving** (+1.68→+2.19), R6
  nested-OOS PASS (+21.8% ddRed at +2.6% cost — the threshold generalizes forward and is nearly
  cost-free OOS), R5 cross-episode + LOFO PASS (genuine general rule, not fit to one episode).

It is the **first deployable result of the run** — a real, robust, PIT-clean live capital-preservation
policy. The honest caveats that must be stated to the human:
1. **R4: it is NOT free DD reduction and NOT skill.** The cost is real and ~proportional. The honest
   equivalent is running the book at ~0.65–0.70 constant gross.
2. **R6: the specific threshold does NOT generalize to EXT** (−7.3% forward ddRed / +44.8% cost on the
   multi-episode panel — X drifts deep and barely fires). The +21.8% nested-OOS result is HL70-specific.

## Deployment recommendation: equity-stop vs constant 0.67-gross
Both achieve a similar DD/return profile (R4). Choose by behavioral preference:
- **Equity-DD stop (X=1,600, g_floor=0.40)** — recommended **IF** the desk wants *full exposure in
  calm periods and automatic protection once a deep loss is underway* ("let it run, then protect").
  This is a genuine behavioral advantage over constant-de-gross — it does not give up upside in calm
  regimes — at the cost of whipsaw round-trips (~15/402d) and a small implementation state machine.
- **Constant ~0.67 gross** — the simpler near-equivalent: no state machine, no whipsaw, same average
  DD/return profile. Choose this if operational simplicity is preferred and the "full-in-calm" behavior
  is not valued.

Recommendation: **deploy the equity-stop form** as an optional risk overlay (separate from the alpha
champion) for desks that want the let-it-run-then-protect behavior; otherwise the constant-0.67-gross
is a defensible simpler alternative. Either way the alpha champion is unchanged.

## Insight for next cycle
The prediction axis (iters 5–10) and now the reaction axis (iter-011) are both characterized: the −57%
DD is mechanically reducible at ~proportional cost (a reactive stop, robust across episodes — the first
to survive LOFO because it reacts rather than predicts), but it is **not reducible for free** on free
data. There is no skillful tail-selector. The deliverable for the human is the trade-off curve + the
X≈1,600 recommendation. Remaining classes are out-of-scope for free-data construction work
(bull-only beta strategy, different alpha, paid leading data, or accept DD + this kill-switch).
