# Evaluation Report — iter-007

**Change:** Parameter-free 2-axis alt-bear SIDE gate (F1). In the SIDEWAYS regime, FLAT the held
book when the equal-weight alt complex underperforms BTC over the trailing 30d
(`regime==side AND alt_index_30d < btc_30d → emit {}`). bull→mom30 and bear→FLAT unchanged. No new
model feature, no retrain, no swept threshold. Implements the iter-006 root-cause diagnostic's last
specific feature hypothesis — the alt-complex-bear regime axis — in its most defensible
parameter-free form.

**Script:** `research/convexity_portable_2026-05-20/scripts/X122_altbear_gate.py`
**Per-cycle outputs:** `results/X122_percycle_{HL70,EXT,S44}.parquet`

## Verdict: **REJECT**

Fails G4 (HL70 p72, EXT p0 — both < p95), G5 multi-episode (HL70 lift collapses dropping fold-5;
EXT helps 1/4 episodes with negative episode-LOFO), G6-EXT (significantly negative), G7 (net-hurts
EXT and S44). The HL70 in-sample +4.73 Calmar is a **single-episode (2025-Q4 / fold-5) artifact**,
not conditional skill — Review certified it is real and non-leaked, which sharpens the point: even a
real, non-leaked, parameter-free, G3-waived change that clears the in-sample bar REJECTS because it
does not generalize across episodes or universes.

All numbers below were **re-derived independently** from the per-cycle parquets (held-book PnL,
folds, flags) and cross-checked against the implementation's own placebo/CI machinery. They agree.

---

## Headline metrics @4.5bps (G2 / G7)

| universe | role | arm | Sharpe | maxDD (bps) | Calmar | totPnL (bps) | flagged |
|---|---|---|---|---|---|---|---|
| **HL70** | production (1 DD episode) | base | +1.93 | −5,674 | +1.68 | +10,472 | — |
| HL70 | | **F1** | +2.61 | −2,239 | **+4.73** | +11,633 | 1101/1455 side |
| **EXT** | multi-episode 2021–26 | base | +0.87 | −4,953 | +0.66 | +15,448 | — |
| EXT | | **F1** | +0.51 | **−6,837** | **+0.25** | +8,084 | 3523/5232 side |
| **S44** | transport | base | +1.84 | −4,170 | +2.10 | +25,620 | — |
| S44 | | **F1** | +1.43 | −3,728 | +1.61 | +17,506 | 2596/3650 side |

Base reproduces X117 on HL70 exactly (Sharpe +1.93 / maxDD −5,674 / Calmar +1.68 / totPnL +10,472).
On HL70 F1 looks like a clean ~3× Calmar win (DD −60%). On **both** out-of-sample universes it
net-HURTS: EXT Calmar halves and maxDD is **38% WORSE**; S44 Calmar 2.10→1.61, totPnL −32%.

---

## Gate-by-gate

### G1 — Look-ahead: **PASS** (Review certified)
Review handoff PASS. `alt_index_30d` is the trailing-180-bar (30d on 4h grid) cum log-return of the
panel's OWN eq-weight, ex-BTC/ETH alts, `.shift(1)` lagged; `btc_30d` for the comparison is lagged
the SAME way (matched lag, no lag-mismatch leak). Independent recompute from raw klines max-diff 0.0;
per-universe isolation confirmed (no cross-universe carry). Base reproduces X117 to the bp. Deleting/
adding a regime branch cannot leak. No IC>+0.10 anomaly in results.

### G2 — In-sample objective (HL70): **PASS** (necessary, not sufficient)
HL70 Calmar +1.68 → **+4.73** (> current_best +1.68). maxDD −5,674 → −2,239 (−60%). Sharpe +1.93 →
+2.61. totPnL ~flat (+10,472 → +11,633). Clears the bar — but G2 is necessary only.

### G3 — Nested-OOS: **WAIVED** (Review-confirmed)
The boundary is the structural ±0 RELATIVE comparison `alt30 < btc30` — no swept/selected scalar
(the iter-006 `−0.10` is gone). Same class as the existing ±10% BTC regime rule. Legitimately waived.

### G4 — Matched side-pool placebo (MANDATORY ≥p95): **FAIL on BOTH** (re-derived, 200 seeds)
FLAT the same COUNT of random side cycles as F1 flags, drawn from the side pool (`replace=False`),
through the identical held-book construction/decay machinery.

| universe | real F1 Calmar | placebo p50 | placebo p95 | placebo max | **rank** | maxDD rank |
|---|---|---|---|---|---|---|
| HL70 | +4.73 | +4.53 | +5.06 | +5.87 | **p72** ✗ | p38 |
| EXT | +0.25 | +0.54 | +0.73 | +0.87 | **p0** ✗ | p0 |

On HL70 the "win" is **no better than FLATting random side cycles** (placebo p95 +5.06 already beats
real +4.73). On EXT, **random side-FLATting does strictly better** than the signal-aligned gate
(placebo p50 +0.54 > real +0.25). The alt-bear axis carries no conditional information beyond the
mechanical "run-smaller in the zero-mean side regime" effect.

### G5 — Multi-episode robustness (the decisive, upgraded gate): **FAIL on BOTH**

**HL70 per-fold + fold-LOFO** (re-derived):
- F1 Calmar ≥ base in **4/7 folds** (f3, f5, f6, f7; loses f2, f4, f8) — fails the 6/9 spirit.
- Fold-LOFO: full lift **+3.05**; dropping each fold leaves +2.78…+3.52 — EXCEPT **drop fold-5
  (the 2025-Q4 episode) → lift −0.96**. The entire HL70 win is one fold.

| drop | −f2 | −f3 | −f4 | **−f5** | −f6 | −f7 | −f8 |
|---|---|---|---|---|---|---|---|
| lift | +2.78 | +3.08 | +3.18 | **−0.96** | +3.52 | +3.50 | +3.24 |

**EXT per-episode maxDD + episode-LOFO** (re-derived — the multi-episode test prior iters lacked):

| episode | base maxDD | F1 maxDD | DDimp% | improved? |
|---|---|---|---|---|
| 2022_luna | −765 | −765 | +0.0% | no (no side flagged; BTC already bear) |
| 2022_ftx | −2,474 | −969 | +60.8% | **YES** |
| 2024_summer | −1,266 | −1,941 | −53.3% | no (**HURTS**) |
| 2025_q4 | −900 | −1,026 | −14.0% | no (HURTS) |

Episodes improved: **1/4** (bar ≥3/4) → FAIL.
Episode-LOFO: full lift **−0.41** (F1 already loses on the full panel); dropping each episode keeps
the lift NEGATIVE (−0.44 / −0.51 / −0.30 / −0.20). The gate does not net-help on EXT and no single
episode is responsible — it is uniformly worse.

### G6 — Paired CI (block-bootstrap by fold, 2000 draws): **HL70 crosses 0; EXT significantly NEGATIVE**
- HL70: obs +0.483 bps/cyc, 95% CI **[−2.68, +4.11] → CROSSES 0** (not significant).
- EXT: obs −0.716 bps/cyc, 95% CI **[−1.63, −0.07] → clears 0 but NEGATIVE** (F1 significantly WORSE
  than base).

### G7 — Universe robustness: **FAIL**
Must hold on HL70 (production) AND not be a single-universe artifact. It **net-hurts EXT** (Calmar
+0.66→+0.25, maxDD 38% worse, totPnL halved) **and S44** (Calmar 2.10→1.61, totPnL −32%, Sharpe
−0.41). The separation sign flips between universes (HL70 flags losers; EXT flags winners) — the
axis carries no consistent cross-episode conditional info.

### G8 — Cost realism: **not exculpatory**
| universe | @1bp F1 Cal | @3bp F1 Cal | @4.5bp F1 Cal | base @4.5bp |
|---|---|---|---|---|
| HL70 | +4.93 | +4.82 | +4.73 | +1.68 |
| EXT | +0.33 | +0.29 | +0.25 | +0.66 |
| S44 | +1.81 | +1.69 | +1.61 | +2.10 |
F1 loses to base at EVERY cost level on EXT and S44. On HL70 the level is a "FLAT cycles → less
cost / run-smaller" confound, not conditional skill — and the binding G4/G5/G7 tests already fail,
so cost behavior cannot rescue it.

---

## Decision rule (pre-registered) check
ADOPT required: G2✓ AND G3-waived✓ AND **G4≥p95 on BOTH** (p72/p0 ✗) AND **G5 survives episode-LOFO
on BOTH** (HL70 −f5 collapse, EXT 1/4 + negative LOFO ✗) AND G6 clears 0 on BOTH (HL70 crosses, EXT
negative ✗) AND G7 holds on HL70+EXT+S44 (EXT+S44 hurt ✗). → **REJECT.** Matches the research and
implementation pre-registration exactly.

---

## Insight — retiring the last feature hypothesis; why the drawdown is irreducible

This iteration formally retires the **alt-complex-bear regime axis**, the final specific feature
hypothesis surfaced by the iter-006 root-cause diagnostic. Recast in its most defensible
parameter-free form (F1 `alt30 < btc30`, G3-waived, PIT-clean, Review-certified non-leaked), and
subjected for the first time to a genuine **multi-episode** test (the EXT 2021–26 panel the prior
five iterations never had), it fails on every honest gate. The HL70 +4.73 Calmar was the 2025-Q4
episode masquerading as conditional skill.

**What the iter-001..007 arc now establishes about the −57% drawdown:**

1. **The drawdown is essentially ONE correlated-alt-bear episode per universe.** HL70's entire DD is
   2025-09→2025-12 (fold-5). EXT's largest is 2022_ftx. Every "DD-reducer" tested (i1 vol-throttle,
   i2 corr gate, i3 flat-whole-side, i6 alt30<−0.10, i7 alt30<btc30) wins in-sample by avoiding the
   one episode in the production universe and then collapses under LOFO / fails the multi-episode
   panel / fails matched-random placebo. With n=1 big episode per universe, any gate tuned to it is
   fitting n=1 — and a parameter-free gate just gets lucky on that draw, as the matched placebo and
   the EXT sign-flip prove.

2. **The loss is ~92% alpha-residual, not beta** (iter-006). The side-regime mean-reversion `pred`
   alpha is a **zero-edge noise process with a fat left tail** (per-fold XS-IC sign-flips, mean
   ≈ +0.002; hit-rate ~54% vs ~51% out — the magnitude of losers blows out, the sign does not
   invert). The beta hedge works (8% of the loss). So the DD is not a hedging or sizing-construction
   problem; it is that the strategy's 60%-of-cycles side regime has no harvestable cross-sectional
   edge and occasionally realizes its tail.

3. **No observable leads that tail consistently across episodes.** Price (realized vol, own
   momentum, corr), regime (BTC-30d), and now alt-complex direction (alt30 vs btc30) have all been
   tested as gates/separators. Each describes the ONE production episode well *in-sample* and then
   either flips sign on another universe/episode (alt-bear axis: HL70 flags losers, EXT flags
   winners) or is indistinguishable from random (corr p27, vol-throttle p0, alt-bear p72/p0). The
   tail is **not forecastable from the free observables available** — iter-006 measured per-cycle IC
   predictability R²≈0.005 from regime features; it is genuinely unpredictable noise.

**What this means for the strategy and what classes of solution remain (for the human):**

- The construction/feature/regime-composition family of drawdown fixes is **exhausted on free data.**
  The strategy sits at a single-episode-limited local optimum: forward Calmar expectation ~+1.0 to
  +2.2, mean ~+1.5, with a fat-left-tail DD that is structural, not fixable by gating.
- Remaining classes of solution, none cheap:
  1. **Don't trade the side regime's mean-reversion at all** — accept a bull-only (momentum) beta
     strategy. This removes the zero-edge majority and its tail, but it is a *different* strategy
     with its own risk profile (long-only crypto beta, large bear/chop give-back, far fewer
     trading cycles) — and must be re-validated as such, not adopted as a "DD fix."
  2. **A fundamentally different alpha** that has a real cross-sectional edge in correlated-alt
     selloffs (the side regime), rather than mean-reversion that this data shows is zero-edge there.
  3. **Paid leading data** (e.g. on-chain deleverage / liquidation flow, options-implied alt skew)
     that might actually *lead* the alt-bear tail where the free price/funding/regime observables do
     not. iter-006/007 pin the missing signal to an alt-complex-bear *leading* indicator — the free
     observables only describe it coincidentally.
- Otherwise the honest path is **accept the structural DD and live-monitor with a kill-switch**,
  acknowledging the universe-overfit risk (delistings/composition drift will move performance).

Champion stays = baseline (Calmar +1.68). Proposed `current_best` delta: **NONE.**
