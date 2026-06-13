# iter-034 — REPORTING: 70-symbol vs full-156 universe head-to-head (within-V0, fair A/B)

**Track:** reporting / decision-support (no new alpha hunt). Clean apples-to-apples comparison.
**One-line read:** within the same V0 model / same construction / same window, going **70 → 156**
*raises capacity (totPnL ~1.5–2×) and base-book Calmar*, but **lowers risk-adjusted Sharpe** (+stop
FULL +1.34 → +1.03) and worsens maxDD. The **history-gated wide set** is the best honest wide config
(recovers most of the lost Sharpe: +1.19) — but it still does not beat the narrow 70-name set on Sharpe.

---

## Method — what's held constant, what varies

- **Model:** identical. The SAME expanded **x132 V0** preds drive every cell
  (`research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet`). The
  70-name arm is simply the x132 preds **restricted to the 70 HL names** (68 are present; ASTERUSDT
  and TSTUSDT are not in x132). This removes the model-retrain confound entirely — the only thing
  that changes across rows is the *set of names the held-book can pick from*.
- **Construction:** identical — champion regime-hybrid held-book (mom-bull / mean-rev side beta-neutral /
  flat-bear), K=5, 6 sleeves, @4.5 bps, BASE and **+iter-012 vol-norm stop (k=2.0)**.
- **Engine:** the iter-032 fast engine **VERBATIM** (which imports the iter-031 / X117+X125 slow engine).
  Re-verified fast == slow on the 70-subset: Sharpe +1.341453 vs +1.341453, max abs per-cycle diff 2.3e-13.
  The full-156 +stop FULL cell reproduces iter-032 exactly (+1.03 / −3960 / +1.16 / +21611).
- **Windows:** (a) **FULL** = all 8 folds, 2021-08 → 2026-05 (multi-episode transport view).
  (b) **HL70-era** = folds **7+8** (2025-02-09 → 2026-05-06) — the production-relevant recent period;
  the actual HL70 preds (2025-03-30 → 2026-05-10) live entirely inside these two folds.
- Universes: **70** = HL68-in-x132; **156** = full x132 (ex-BTC); **hist-gated wide** = full set with a
  per-cycle ≥180-bar (~30d) trailing-history floor (the iter-032 recommended deploy config).

---

## Comparison table (universe × window × {base, +stop}) @4.5 bps, x132 V0 preds

| universe | window | variant | Sharpe | maxDD (bps) | Calmar | totPnL (bps) | %pos | folds_pos |
|---|---|---|---|---|---|---|---|---|
| **70** (HL68 in x132) | FULL 2021-26 | base | **+1.35** | −3508 | +1.64 | +27032 | 40.7 | 7/8 |
| **70** | FULL 2021-26 | +stop | **+1.34** | **−2647** | +1.84 | +22863 | 40.7 | 7/8 |
| **70** | HL70-era 25-26 | base | **+2.02** | −1269 | +6.00 | +8950 | 39.9 | 2/2 |
| **70** | HL70-era 25-26 | +stop | **+1.92** | **−1269** | +5.44 | +8109 | 39.9 | 2/2 |
| **156** (full) | FULL 2021-26 | base | +1.12 | −4262 | +2.02 | +40434 | 41.2 | 7/8 |
| **156** | FULL 2021-26 | +stop | +1.03 | −3960 | +1.16 | +21611 | 41.1 | 6/8 |
| **156** | HL70-era 25-26 | base | +1.19 | −3960 | +3.84 | +17868 | 39.7 | 2/2 |
| **156** | HL70-era 25-26 | +stop | +0.85 | −3960 | +1.34 | +6232 | 39.7 | 2/2 |
| **hist-gated wide** | FULL 2021-26 | base | +1.24 | −4687 | +2.00 | +44035 | 40.3 | 7/8 |
| **hist-gated wide** | FULL 2021-26 | +stop | +1.19 | −3960 | +1.33 | +24804 | 40.3 | 5/8 |
| **hist-gated wide** | HL70-era 25-26 | base | +1.29 | −3960 | +4.11 | +19143 | 39.3 | 2/2 |
| **hist-gated wide** | HL70-era 25-26 | +stop | +0.97 | −3960 | +1.51 | +7035 | 39.2 | 2/2 |

### 70 → 156 deltas (the fair within-V0 A/B), +stop config

| window | metric | 70 | 156 | Δ (156−70) | hist-gated | Δ (hg−70) |
|---|---|---|---|---|---|---|
| FULL 2021-26 | Sharpe | +1.34 | +1.03 | **−0.31** | +1.19 | −0.15 |
| FULL 2021-26 | maxDD | −2647 | −3960 | **−1313 (worse)** | −3960 | −1313 (worse) |
| FULL 2021-26 | Calmar | +1.84 | +1.16 | **−0.68** | +1.33 | −0.51 |
| FULL 2021-26 | totPnL | +22863 | +21611 | −1252 | +24804 | +1941 |
| HL70-era 25-26 | Sharpe | +1.92 | +0.85 | **−1.07** | +0.97 | −0.95 |
| HL70-era 25-26 | maxDD | −1269 | −3960 | **−2691 (worse)** | −3960 | −2691 (worse) |
| HL70-era 25-26 | Calmar | +5.44 | +1.34 | **−4.10** | +1.51 | −3.93 |

---

## Honest read (2–3 sentences)

Holding the model, construction and window constant and varying **only** the universe within the same
x132 V0 preds, going **70 → 156 does not improve risk-adjusted performance — it slightly hurts it**:
+stop Sharpe falls **+1.34 → +1.03** on the full 2021-26 panel (and **+1.92 → +0.85** on the
production-relevant 2025-26 sub-window), Calmar falls, and maxDD worsens (−2647 → −3960); what
expansion buys is **capacity / base-book Calmar**, not Sharpe. The **history-gated wide set** is the
honest best-of-wide (recovers Sharpe to +1.19 FULL / Calmar +1.33 by dropping sub-30d-history names),
consistent with iter-032's finding that breadth helps *directionally* but the headline lift attributed
to expansion was mostly the **model retrain**, and full-156 fails the clean Sharpe gates (iter-032
G4 p69 / G5 6/8 / G6 paired-CI crosses zero). **Critically, the original champion "+1.93 Sharpe" is a
different model (V5_mv3) on the HL70-only 2025-26 window — NOT comparable to these V0 numbers;** the
only fair within-model comparison is this table, and within it the narrow 70-name set is the
Sharpe/Calmar/maxDD winner while the hist-gated wide set is the best honest wide deploy if you want the
extra capacity. (The HL70-era is only 2 folds, so its single-window Sharpes are high-variance — the
70-set's +1.92 there leans on a benign recent regime — but the *direction* of the 70 vs 156 gap is
consistent across both windows.)

---

## Notes / caveats

- **"70" = 68 names** because ASTERUSDT and TSTUSDT (in the HL70 pred set) are not present in x132.
  This is the honest intersect; it cannot be otherwise without a separate x132 build.
- The HL70-era window (folds 7+8) is short (2 folds, ~2575 cycles); treat its absolute Sharpe levels as
  directional, not as a robust estimate. The FULL 2021-26 panel is the robust transport view.
- maxDD for all full/hist-gated +stop cells pins at −3960 because the vol-norm stop's deepest realized
  drawdown episode is shared across the wider books; the 70-set's stop reaches a shallower −2647 / −1269.
- Engine verified == slow path (2.3e-13). Script: `agents_system/research/scratch/iter034_70_vs_full.py`;
  artifact `outputs/iter034/iter034_70_vs_full.csv`.
