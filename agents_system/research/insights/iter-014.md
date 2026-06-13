# iter-014 — STRUCTURAL K × hold/sleeve sweep (discrete architecture validation)

**Question:** are the INHERITED structural choices **K=5 legs/side** and **HOLD=6 sleeves (24h hold)**
robust-optimal on this system, or does a different DISCRETE config robustly raise Sharpe/Calmar across
HL70 (prod) + EXT (2021-26) + S44? K/HOLD were never validated here. The sister vBTC line found K=3 beat
K=4 (+0.82 Sharpe, p100 placebo) — so K matters and 5 may be suboptimal.

K and HOLD are **discrete** structural choices, so **G3 is waivable IF not tuned-for-Sharpe** — but the
bar (per the contract + the 51-panel K=4-vs-K=3 lesson) is that a different config must be **ROBUSTLY
better cross-universe (G7) + nested-OOS + paired-CI (G6) on the production universe**, not just
in-sample-best on HL70. Evaluated on the **alpha champion's structure CLEAN — no reactive overlay**.

Scripts: `X128_struct_K_hold_sweep.py` (full grid + G7 ranks + nested-OOS + G6), `X128b_hold9_candidate.py`
(the single cleanest discrete change isolated). Reuse X123 `build_universe`/held-book verbatim; K
parameterizes the L/S tail-selection + beta-neutral sizing, HOLD the sleeve-averaging. Base K=5/H6 @4.5bps
reproduces **X117 EXACT** (HL70 Sharpe +1.93 / maxDD −5674 / Calmar +1.68; S44 +1.84 / −4170).

## STEP 2 — K × HOLD grid @4.5bps (Calmar; Sharpe; maxDD)

K=5 baseline highlighted. **HOLD is monotone-helpful on all 3 universes; K is universe-dependent/noisy.**

**HL70** (prod) — Calmar:
| K\H | 3 | 6 | 9 | 12 |
|---|---|---|---|---|
| 2 | 1.66 | 1.58 | **1.96** | 1.75 |
| 3 | 0.94 | 1.14 | 1.58 | 1.32 |
| 4 | 1.19 | 1.40 | 1.60 | 1.06 |
| **5** | 1.43 | **1.68 (base)** | 1.88 | 1.42 |
| 6 | 1.60 | 1.73 | 1.93 | 1.65 |
| 7 | 1.34 | 1.49 | 1.69 | 1.37 |

**S44** — Calmar (longer-hold effect strongest here):
| K\H | 3 | 6 | 9 | 12 |
|---|---|---|---|---|
| 2 | 2.29 | 2.49 | 2.64 | 2.56 |
| 4 | 2.00 | 2.18 | 2.79 | **2.96** |
| **5** | 1.63 | **2.10 (base)** | 2.73 | 2.82 |
| 7 | 1.73 | 2.05 | 2.74 | 2.82 |

**EXT** — Calmar (flattest; everything ~0.6–0.9; longer hold mildly helps):
K5: H3 0.35 / H6 0.66 / H9 0.81 / H12 0.72. Best EXT Calmar = K3 H9 (0.90).

**Marginals (the headline):**
- **HOLD-marginal at K=5 (Calmar):** HL70 H3 1.43 → **H6 1.68 → H9 1.88** → H12 1.42; EXT 0.35→0.66→**0.81**→0.72; S44 1.63→2.10→2.73→**2.82**. Longer hold (9–12 sleeves) raises Calmar + cuts maxDD on ALL THREE (cost amortization — same mechanism as the vBTC sleeve-overlap finding). H12 starts to fade Sharpe on HL70 (turnover too low / stale book).
- **K-marginal at HOLD=6 (Calmar):** HL70 best K6 (1.73); EXT best K2/K3 (0.79); S44 best K2 (2.49). **No consistent K** — best K disagrees across universes (HL70→6, EXT→2/3, S44→2/4). The vBTC "K=3 wins" does NOT transport: K=3 is the *worst* HL70 cell (Cal 1.14) and only middling elsewhere.
- **Cost (G8):** longer-hold edge holds / widens at every cost {1,3,4.5}bps (S44 K5 ΔCal H9-vs-H6 = +0.58/+0.61/+0.63); not a low-cost artifact.

## STEP 3 — the honest tests

### G7 cross-universe consistency (avg Sharpe-rank across the 3 universes, 1=best of 24 cells)
| config | avg-Sh-rank | worst | HL70 | EXT | S44 |
|---|---|---|---|---|---|
| K2 H9 | 5.0 | 12 | 1 | 2 | 12 |
| K5 H9 | 5.0 | 8 | 8 | 4 | 3 |
| K5 H6 (BASE) | 13.0 | 17 | 5 | 17 | 17 |

The two top cross-universe cells are **K2 H9** and **K5 H9** — both have HOLD=9. **Best-per-universe
DISAGREES** (HL70→K2H9, EXT→K2H6/K3H9, S44→K7H12/K4H12), confirming K is universe-overfit; the only
shared signal is *longer hold*.

### G3/G5 nested-OOS (choose config on past folds, apply forward) — THE TRAP, and it bites
- **choose-K-only (H6):** HL70 picks K=2 every fold → forward Δcal **−0.19 (CHURNS)**, maxDD blows to −9629; S44 picks K2-4 → **−1.11 (CHURNS)**; EXT +0.10 (barely generalizes). The HL70-best K does NOT survive forward.
- **choose-(K,HOLD):** HL70 −0.15, EXT −0.04, S44 −1.41 — **all churn/lose.**
- **choose-HOLD-only (K=5, the cleanest lever):** HL70 picks H3 every fold (early folds favored it) → Δcal **−0.22 (CHURNS)**; EXT −0.10 churns; S44 +0.30 generalizes. Even the monotone hold lever fails honest forward selection on HL70.

### G6 paired block-bootstrap CI — the single cleanest discrete change, K=5 HOLD 6→9 (36h)
| univ | H6 → H9 | maxDD ddRed | G5 wins | G6 paired diff CI |
|---|---|---|---|---|
| **HL70 (prod)** | Cal 1.68→1.88, Sh 1.93→1.83 | **+18%** | **2/7** | −0.39 bps/cyc **[−1.20,+0.65] CROSSES 0** |
| EXT | Cal 0.66→0.81 | +14% | 6/8 | +0.07 **[−0.30,+0.37] crosses 0** |
| S44 | Cal 2.10→2.73, Sh 1.84→2.08 | +15% | 8/8 | +0.41 **[+0.15,+0.75] clears 0** |

H9 looks great in-sample (Calmar up + maxDD down on all 3, same-signed, cost-robust) but on the
**production universe (HL70)** it FAILS G5 (2/7 folds), FAILS G6 (CI crosses 0, Sharpe actually drops
−0.10), and FAILS nested-OOS (selection picks the wrong hold). The clean win is only on S44.

## Verdict — K=5 / 6-sleeve is ROBUST-OPTIMAL under honest validation

**No discrete config robustly beats the inherited K=5/6-sleeve across all three universes + nested-OOS +
production-universe paired-CI.** Specifically:
- **K is universe-overfit / noisy.** Best K disagrees (HL70 6, EXT 2, S44 4); choosing K on past folds
  CHURNS forward (HL70 −0.19, S44 −1.11). The vBTC K=3 finding does NOT transport — K=3 is the worst
  HL70 cell. K=5 is a defensible middle (HL70 Sh-rank 5/24, never the loser).
- **HOLD has a real same-signed direction (longer = lower DD, higher Calmar via cost amortization)** —
  the one genuine cross-universe signal in the sweep — **but it fails the production honest gates**: H9
  on HL70 fails G5 (2/7), G6 (CI crosses 0, Sharpe −0.10), and nested-OOS (churns). The Calmar lift is
  real but statistically indistinguishable per-cycle on HL70 and not forward-selectable; only S44 clears.

This is the **same pattern the whole iteration log keeps surfacing**: an in-sample Calmar improvement
that is real but does not clear honest validation on the production universe (cf the 51-panel K=4-vs-K=3
lesson — discrete choices that DON'T transport are universe-overfit). The inherited structure is sound.

**Useful robustness confirmation, not a failure.** Documenting that the inherited K=5 / 6-sleeve is the
robust choice retires the "K/HOLD never validated" open question.

### Honest note for a risk-preference decision (NOT an ADOPT)
If a human explicitly prefers lower drawdown and accepts a flat-to-slightly-lower Sharpe, **HOLD=9
(36h hold)** is the one same-signed lever: it cuts maxDD ~14–18% on every universe and raises Calmar
in-sample everywhere, at the same/lower cost. It is offered as a *risk dial* (like the iter-012 reactive
stop), characterized but NOT adopted, because it fails G5/G6/nested-OOS on the production universe. It is
NOT a Sharpe improvement and must not be sold as one.

Artifacts: `research/convexity_portable_2026-05-20/results/X128_K_hold_grid.parquet`;
scripts `X128_struct_K_hold_sweep.py`, `X128b_hold9_candidate.py`.
