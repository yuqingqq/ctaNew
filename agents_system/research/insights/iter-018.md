# Research Insights — iter-018 (HUMAN idea: DYNAMIC thesis-based EXIT on realized residual convergence)

**Question (human):** the held book holds 24h via 6 overlapping 4h sleeves (cost-amortization, not
signal capture — iter-016). Instead of the FIXED hold, EXIT a position EARLY when the residual it bet
on has CONVERGED (the predicted mean-reversion has been realized / the edge is gone), and keep
HOLDING while the residual RETAINS (reversion still pending). KEY DISTINCTION from iter-016: iter-016
set the hold from ENTRY-time signal strength (PREDICT the horizon → failed nested-OOS). THIS is a
DYNAMIC exit on REALIZED convergence — observe the position's residual evolve and exit when the bet
has played out. That is PIT-valid (convergence is OBSERVED, not predicted).

**The crux (tested both):** "residual retains" splits into (a) not-yet-reverted-but-will (hold =
correct) vs (b) moving AGAINST you / WIDENING (oversold getting more oversold = the iter-006 falling
knife = the −57% DD). The human's rule "hold if retains" holds BOTH — and holding (b) is the disaster.
So I tested the human's rule AS STATED (P2) AND a variant with a divergence cut (P3).

Diagnostic, not a build. Scripts: `iter018_dynamic_exit.py` (engine + G4/G5/G3 on HL70+EXT),
`iter018_ext_episodes.py` (EXT per-episode + LOFO). Reuses X122/X123 `build_universe` + held-book
verbatim. Per-cycle outputs: `results/iter018_dynexit_{HL70,EXT}.parquet`.

---

## TL;DR

The human's literal rule (**P2: exit-on-converge, HOLD-through-divergence**) does **almost nothing** —
the convergence signal rarely fires, P2 ≈ baseline on HL70 (Calmar 1.68→1.73, 52 early exits) and is
slightly WORSE on EXT (0.66→0.64), and it FAILS G4 on both (HL70 p78 / EXT p25). It does NOT re-create
a *worse* falling knife in aggregate (because the mean-rev legs are beta-neutral and the per-cycle
residual mean is ~0), but it gives no protection either — it just holds the same stale book.

The **divergence-cut variant (P3: exit-on-converge OR cut-on-divergence)** is the classic trap. On the
production universe **HL70 it looks spectacular** — Calmar 1.68→**2.97**, maxDD −5674→**−3536 (−38%)**,
and it **PASSES G4 at p100** (real Calmar +2.97 vs matched-random-exit p95 +2.49; real maxDD −3536 vs
random-exit p05 −5509) AND nested-OOS +0.21. The G4 pass is genuine *within HL70*: cutting the legs
whose realized residual has gone against the bet caps the tail better than a random exit of the same
average hold. **But it is a SINGLE-EPISODE artifact.** On the multi-episode **EXT panel P3 HURTS on
every metric** — Calmar 0.66→**0.52**, maxDD −4953→**−5410 (WORSE)**, Sharpe +0.87→+0.79, FAILS G4 at
**p18**, per-fold **2/8**, nested-OOS **−0.12**, and **episode-LOFO is NEGATIVE dropping EVERY one of
the 4 episodes** (−0.11 to −0.20). HL70 has exactly one big DD episode (2025-Q4) and the divergence-cut
fits it; on EXT's four episodes it whipsaws — by the time the residual has moved against the bet you
are near the local bottom, so the cut exits into the bounce (the same "alt-bear flag is a coincident
bottom-detector" pathology that killed iter-007/008/010/017).

**Verdict: NO-CANDIDATE.** Convergence-exit (the human's literal rule) is inert and fails G4. The
divergence-cut that makes it look good is universe-overfit to the single HL70 episode (G7 FAIL, EXT
episode-LOFO uniformly negative). The 24h fixed sleeve hold remains robust-optimal. Champion unchanged.

---

## STEP 2 — operational PIT definitions + engine

A sleeve entered at cycle t0 has long basket L (top-K pred) and short basket S (bottom-K pred),
beta-neutral leg weights `legw` (matches X117). At each later cycle u = t0+s, decided with info ≤ u−1:
- **realized capture** `cap` = Σ over legs of `sign(legw) · alpha_A` cumulated over the prior steps
  (the residual the bet has actually earned so far; PIT — alpha_A from t0→u−1 is realized by u).
- **fresh re-score** `fresh` = mean over legs of `sign(legw_entry) · fresh_pred(u)` (the model's
  current signal in the bet's direction; the fresh pred is already in the preds file, PIT at u).
- **CONVERGED** (exit) := `fresh ≤ conv_band` (directional pred decayed toward 0 / flipped → edge gone).
- **DIVERGES** (cut, P3 only) := `cap ≤ −div_band` (realized capture went against the bet = falling knife).
- **RETAINS** := neither → hold. Once a sleeve exits/cuts it stays out (no re-entry). Book at u =
  average of LIVE sleeves' legw/HOLD (gross drops when sleeves exit — the honest behavior; G4 then
  tests whether the EXIT TIMING beats a random exit of matched avg-hold, not just "run smaller").

Policies on the SAME entries, dynamic exit on SIDE sleeves only (mean-rev = the DD source; bull
momentum sleeves held full HOLD). Defaults conv_band=0.0, div_band=0.05 (≈2.6σ of single-bar resid).
**P1 = fixed 24h (== production); P2 = exit-on-converge, hold-divergence (HUMAN); P3 = + cut-on-diverge.**

P1 reproduces X117 EXACTLY (HL70 Sharpe +1.93 / maxDD −5674 / Calmar +1.68) — engine validated.

## STEP 3 — P1 vs P2 vs P3, net of cost

**HL70 (production)** @4.5bps:
| policy | Sharpe | maxDD | Calmar | totPnL | avgHold(side) | earlyExits |
|---|---|---|---|---|---|---|
| **P1 fixed-24h** | +1.93 | −5674 | +1.68 | +10472 | 6.00 | 0 |
| **P2 human (hold-diverge)** | +1.94 | −5538 | +1.73 | +10497 | 5.94 | 52 |
| **P3 +cut-diverge** | +2.31 | **−3536 (−38%)** | **+2.97** | +11529 | 4.35 | 691 |

**EXT (2021–26 multi-episode)** @4.5bps:
| policy | Sharpe | maxDD | Calmar | totPnL | avgHold(side) | earlyExits |
|---|---|---|---|---|---|---|
| **P1 fixed-24h** | +0.87 | −4953 | +0.66 | +15448 | 6.00 | 0 |
| **P2 human (hold-diverge)** | +0.85 | −5001 | +0.64 | +15040 | 5.89 | 262 |
| **P3 +cut-diverge** | +0.79 | **−5410 (WORSE)** | **+0.52** | +13199 | 4.96 | 1770 |

**Reads:**
- **P2 (the human's rule as stated) is inert.** Convergence (fresh signed-pred ≤ 0) fires on only ~3%
  of side sleeve-steps → 52 early exits on HL70, 262 on EXT; avg side hold barely moves (5.94/5.89 vs
  6.00). Calmar ≈ baseline (HL70 +1.73 ≈ +1.68; EXT +0.64 < +0.66). It does NOT re-create a worse
  aggregate falling knife (beta-neutral legs, ~0-mean residual), but it gives zero protection — it
  just keeps holding the same stale book. **The exit-on-convergence half of the human idea carries no
  edge.** This converges with iter-016: the signal has already decayed by the time you'd "exit on
  convergence," so a convergence exit is just a slightly-shorter fixed hold (already swept, iter-014).
- **P3 (the divergence cut) is where ALL the HL70 action is** — and it is the falling-knife protection
  the human's literal rule omits. On HL70 it cuts maxDD 38% and lifts Calmar to +2.97. **But it does
  NOT transport:** on EXT it is WORSE than baseline on Sharpe, maxDD AND Calmar at every cost level.

## STEP 3b — G8 cost (Calmar by policy)
| univ | cost | P1 | P2 | P3 |
|---|---|---|---|---|
| HL70 | 1 / 3 / 4.5 bps | 1.99 / 1.81 / 1.68 | 2.05 / 1.86 / 1.73 | **3.70 / 3.27 / 2.97** |
| EXT | 1 / 3 / 4.5 bps | 1.02 / 0.80 / 0.66 | 0.99 / 0.77 / 0.64 | **0.84 / 0.64 / 0.52** |

P3 dominates at every cost on HL70 and LOSES at every cost on EXT — not a cost artifact either way.

## STEP 3c — DECISIVE honesty gates

### G4 matched-random-exit placebo (200 seeds; force a RANDOM exit step on side sleeves drawn from the SAME empirical hold-length distribution the real policy produced → matched avg-hold + matched dist)
| univ | policy | real Calmar | placebo p50 / p95 | **Calmar rank** | real maxDD | random maxDD p50/p05 | **DD rank** |
|---|---|---|---|---|---|---|---|
| **HL70** | P2 | +1.73 | +1.69 / +1.77 | **p78 FAIL** | −5538 | −5648 / −5797 | p88 |
| **HL70** | **P3** | **+2.97** | +1.97 / +2.49 | **p100 PASS** | **−3536** | −4692 / −5509 | **p100** |
| **EXT** | P2 | +0.64 | +0.66 / +0.70 | **p25 FAIL** | −5001 | −4954 / −5224 | p38 |
| **EXT** | **P3** | **+0.52** | +0.58 / +0.69 | **p18 FAIL** | −5410 | −5075 / −5661 | p22 |

The P3 G4 result **flips sign across universes**: on HL70 the divergence-cut TIMING genuinely beats a
random exit of matched avg-hold (p100 — cutting the legs that have moved against you caps the tail far
better than random); on EXT a RANDOM exit of the same avg-hold does BETTER than the divergence-cut
(p18). That sign-flip is the signature of a single-episode fit (HL70 has one DD episode the cut is
tuned to; EXT has four where it whipsaws).

### G5 per-fold (P3 Calmar ≥ P1)
HL70 **5/7** (fails 6/9 narrowly). EXT **2/8** (fails hard). The HL70 5/7 is itself concentrated —
fold 5 (the deep-DD fold) drives it (P1 −4.60→P3 −4.11, maxDD −4623→−2843).

### G3 nested-OOS of (conv_band, div_band) for P3 (choose on past folds, apply forward)
| univ | P1 (nested window) | P3 nested | lift |
|---|---|---|---|
| HL70 | +0.84 | +1.04 | **+0.21** (generalizes IN-UNIVERSE) |
| EXT | +0.65 | +0.52 | **−0.12** (does NOT generalize) |

On HL70 the chosen bands stabilize to (conv 0.25, div 0.05) and beat P1 forward — but this is forward
selection *within the one HL70 episode structure*. On EXT the bands churn and lose forward. The
HL70-only nested-OOS pass is necessary-not-sufficient; G7 is the binding gate.

### EXT episode-LOFO (the decisive multi-episode test) — P3 lift NEGATIVE dropping EVERY episode
| drop | P1 Calmar | P3 Calmar | lift |
|---|---|---|---|
| full | +0.66 | +0.52 | **−0.14** |
| −2022_luna | +0.67 | +0.53 | −0.14 |
| −2022_ftx | +0.79 | +0.59 | −0.20 |
| −2024_summer | +0.77 | +0.61 | −0.16 |
| −2025_q4 | +0.49 | +0.38 | −0.11 |

This is the **inverse of a one-episode artifact at the EXT level** — P3 uniformly HURTS across ALL four
episodes (not "wins on one, neutral elsewhere"; it *loses* everywhere on EXT). Per-episode totPnL/maxDD:

| episode | P1 pnl | P3 pnl | P1 maxDD | P3 maxDD | read |
|---|---|---|---|---|---|
| 2022_luna | +756 | +300 | −765 | −765 | cuts the winners, no DD help |
| 2022_ftx | −2039 | −1066 | −2474 | **−1490** | cut DOES help DD in the persistent crash |
| 2024_summer | −267 | **−722** | −1266 | −1288 | cuts into the chop → worse PnL |
| 2025_q4 | +4834 | +4256 | −900 | **−525** | cut helps DD, costs PnL |

The divergence cut *does* shrink the maxDD inside the two genuinely-persistent crashes (ftx, q4) — that
is the real, intuitive part — but (i) it cuts PnL in all four episodes (giving back more in the
bounce-y luna/2024 than it saves), and (ii) the **aggregate** EXT maxDD gets WORSE (−4953→−5410)
because early sleeve exits desynchronize the 6-sleeve overlap and open new drawdown paths between
episodes. Net: the cut is a per-episode DD-vs-PnL trade that loses on the portfolio.

---

## Why this is the expected result, and what it teaches

1. **The convergence half (the human's actual ask) is inert** — confirms iter-016: the 4h mean-rev
   signal has already decayed (IC zero-cross ~h10–12) before any "convergence" is observable, so
   exit-on-convergence = a marginally shorter fixed hold, which iter-014 already swept (longer fixed
   hold is the only same-signed lever, and it fails production nested-OOS/G6). There is no *realized*
   convergence event with edge to time the exit on.
2. **The divergence cut is the iter-006/007/008/012 falling-knife problem in a new costume.** "Cut when
   the residual has moved against the bet" is a reactive stop keyed on realized adverse move. PIT-valid,
   but the adverse move is a **coincident bottom-detector**: by the time `cap ≤ −div_band` the position
   is near a local trough, so the cut exits into the mean-reversion bounce it was supposed to harvest.
   On HL70's single sustained 2025-Q4 bleed that cut looks like genius (p100); across EXT's four
   episodes it whipsaws (p18, LOFO −0.14 everywhere). **Same wall as every reactive directional/skip
   overlay: real-looking on the one HL70 episode, universe-overfit, fails the multi-episode panel.**
3. **The HL70 p100 G4 is the most seductive single number in the log so far** — it is the first DD
   overlay where the *timing* genuinely beats matched-random-exit on the production universe. That is
   exactly why G7 (EXT) and episode-LOFO are mandatory: the timing skill is real *only for the specific
   shape of the one HL70 episode* and is anti-skill on average across regimes.

This converges with iter-012's reactive equity-stop characterization: a divergence/drawdown-reactive
cut is offered (at best) as a **risk dial** that trades PnL for a smaller maxDD *inside persistent
crashes* (ftx/q4 here), NOT as a Calmar/alpha improvement, and on a drifting multi-episode universe it
does not even reliably cut the aggregate DD. The iter-012 portable equity-stop already occupies that
risk-dial slot more cleanly (unitless, transports, R6 3/3).

## Decision (pre-registered gates)

Objective = raise HL70 Calmar without single-episode artifact; mandatory G4 matched-random-exit +
G7(EXT) + episode-LOFO + nested-OOS of the bands.
- P2 (human's literal rule): **FAIL G2** (≈baseline/worse) **+ G4** (p78/p25). Inert.
- P3 (+divergence cut): **PASS** G2/G4/G3 **on HL70 only** → **FAIL G7** (EXT Calmar 0.66→0.52, maxDD
  worse), **FAIL EXT G4** (p18), **FAIL EXT nested-OOS** (−0.12), **FAIL EXT episode-LOFO** (−0.14, NEG
  on all 4).

**→ NO-CANDIDATE. No change proposed. Champion = baseline (HL70 Calmar +1.68), unchanged.** The 24h
fixed sleeve hold is robust-optimal; dynamic thesis-based exit adds nothing honest.

Artifacts: `research/convexity_portable_2026-05-20/scripts/iter018_dynamic_exit.py`,
`iter018_ext_episodes.py`; per-cycle `results/iter018_dynexit_{HL70,EXT}.parquet`.
