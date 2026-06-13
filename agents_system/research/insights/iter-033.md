# iter-033 — TRAINING-CONFIG study (window × cadence): does old data help, does fresher retrain help?

**Track:** deployment / RETRAIN-POLICY (decision-support, honest gates).
**Verdict:** **NO-CANDIDATE — keep EXPANDING / ALL-AVAILABLE-HISTORY, retrain ~quarterly.**
The in-sample winners (faster cadence, 2–3yr rolling) are a single-fold (f5) artifact and **FAIL nested-OOS**:
honest forward config-selection gives Sharpe **+0.85 / Calmar +1.01 — BELOW the incumbent expanding-9-fold
(+1.03 / +1.16)**; paired CI **[−1.57, +0.49] crosses zero**. Same tuned-parameter failure signature as the
K3-margin and decay-sleeve rejects. The current expanding/all-data training is already at its honest optimum.

iter-032's "+0.86→+1.06 from retraining on more data" lever is REAL but it is the **breadth/recency of the
training SET** (more symbols × more rows), **not** a window-shortening or cadence-tuning knob — those don't
generalize forward here.

Engine: the iter-031 deploy engine (X117 held-book regime-hybrid + X125 iter-012 vol-norm stop, k=2.0) VERBATIM,
via the iter-032 fast precompute layer. Panel = `outputs/vBTC_features/panel_expanded_v0.parquet` (175 syms,
2021-01→2026-05, 4h-sampled, PIT target_z). Only the **fold/window construction** in `x6.train_per_sym_ridge`
varies; preds regenerated per config. All numbers @4.5bps, full-156 +stop. Common 9-bin eval grid for per-fold
comparability across cadences. Smoke-check: incumbent reproduces iter-032 full-156 EXACTLY (Sharpe +1.0328,
maxDD −3960, Calmar +1.16, totPnL +21611, IC +0.0146). Script `agents_system/research/scratch/iter033_training_config.py`;
artifacts `outputs/iter033/` (grid.csv, perfold.csv, nested.json, preds/).

---

## 1. The window × cadence grid (full-156, +stop, @4.5bps)

WINDOW = trailing train history (expanding = all; rolling 1/2/3yr). CADENCE = N_FOLDS retrain points
(nf9 ≈ 7mo / current, nf18 ≈ 3.5mo, nf27 ≈ 2.3mo).

| config | Sharpe | maxDD | Calmar | totPnL | full-IC | folds+ | LOFO worst |
|---|---|---|---|---|---|---|---|
| **exp_nf9 (INCUMBENT)** | **+1.03** | −3960 | **+1.16** | +21611 | +0.0146 | 7/9 | +0.86 |
| exp_nf18 | +1.18 | −3960 | +1.31 | +25804 | +0.0118 | 7/9 | +0.79 |
| exp_nf27 | +1.03 | −4124 | +1.14 | +23841 | +0.0115 | 6/9 | +0.69 |
| 3yr_nf9 | +1.03 | −3960 | +1.16 | +21581 | +0.0148 | 7/9 | +0.86 |
| **3yr_nf18 (best-static)** | **+1.28** | −3960 | **+1.43** | +28331 | +0.0114 | 7/9 | +0.90 |
| 2yr_nf9 | +1.19 | −3960 | +1.38 | +25594 | +0.0148 | 7/9 | **+1.03** |
| 2yr_nf18 | +0.99 | −3960 | **+1.68** | +33161 | +0.0110 | 8/9 | +0.73 |
| 1yr_nf9 | +0.93 | −4157 | +1.59 | +31072 | +0.0148 | 7/9 | +0.79 |
| 1yr_nf18 | +1.04 | −4344 | +1.03 | +22299 | +0.0102 | 7/9 | +0.60 |

### Reading the grid (the two clean structural findings)
- **OLD DATA is ~free, not harmful.** `3yr_nf9 ≈ exp_nf9` to 3 decimals (Sharpe +1.03, Calmar +1.16,
  IC +0.0148 vs +0.0146) — dropping >3yr-old rows changes nothing (most folds' trailing window is already <3yr
  on this 2021-start panel). Rolling **2yr** keeps full Sharpe (+1.19) and slightly RAISES Calmar (+1.38) and
  PnL — i.e. trimming the oldest ~1yr is mildly beneficial-to-neutral (regime-drift), NOT the catastrophe a
  short window would be. Rolling **1yr** starts to HURT Sharpe (+0.93) — too little data for per-sym RidgeCV.
  So the data-quantity vs regime-drift tradeoff bottoms out around **2yr** for Sharpe; below that, starvation wins.
- **FRESHER retrain (cadence) is where the in-sample lift lives — but it's f5-concentrated.** nf18 (~3.5mo)
  beats nf9 on the headline (exp +1.03→+1.18, 3yr +1.03→+1.28). nf27 (~2.3mo) gives it back (over-frequent,
  thinner per-fold OOS, noisier). The *Calmar/PnL* improvement of nf18/2yr is large (Calmar up to +1.68,
  totPnL up to +33k) — but see §3 for why it doesn't survive.

---

## 2. Per-fold Sharpe (common 9-fold eval grid) — the f5 trap

| config | f0 | f1 | f2 | f3 | f4 | **f5** | f6 | f7 | f8 |
|---|---|---|---|---|---|---|---|---|---|
| exp_nf9 (inc) | +1.29 | +0.24 | −0.42 | **−1.24** | +2.61 | +1.66 | +1.57 | +0.36 | +1.40 |
| exp_nf18 | +0.92 | +0.12 | +0.60 | −1.34 | −0.09 | **+3.93** | +1.16 | +0.95 | +1.76 |
| 3yr_nf18 | +0.92 | +0.12 | +0.60 | −1.34 | −0.09 | **+3.95** | +1.39 | +1.44 | +1.70 |
| 2yr_nf9 | +1.29 | +0.24 | −0.42 | −1.61 | +2.69 | +1.31 | +2.05 | +1.21 | +1.43 |
| 2yr_nf18 | +0.92 | +0.12 | +0.60 | −1.62 | +0.18 | **+3.95** | +1.39 | +0.78 | +1.54 |
| 1yr_nf18 | +0.92 | +0.05 | +0.38 | −1.40 | −0.87 | **+4.29** | +1.21 | +0.77 | +1.35 |

**The entire cadence "win" is fold f5.** Every nf18 config jumps f5 from ~+1.6 (nf9) to **+3.9–4.3**, while
LOSING f4 (nf9 +2.6 → nf18 −0.09 to −0.87). The faster retrain happens to align a refresh with the f5 episode's
regime turn — pure timing luck — and pays for it in f4. f3 is negative under EVERY config (the structural
correlated-alt-bear drawdown that no prior iter could fix; training config doesn't touch it). So the headline
ranking is driven by which fold a given cadence happens to catch, not by a generalizable property.

---

## 3. NESTED-OOS the config choice — DECISIVE REJECTION

Pick the config with the best CUMULATIVE Sharpe over PAST eval-folds [0..f−1], apply it forward to fold f
(incumbent for the f0/f1 warmup). This is the honest "you must choose without hindsight" test.

| line | Sharpe | maxDD | Calmar |
|---|---|---|---|
| **NESTED-OOS (honest forward selection)** | **+0.85** | −3719 | **+1.01** |
| **INCUMBENT exp_nf9 (no selection)** | **+1.03** | −3960 | **+1.16** |
| BEST-STATIC (hindsight) 3yr_nf18 | +1.28 | — | +1.43 |

- Chosen path: `f2-f4,f6-f7: exp_nf27 · f5: exp_nf9 · f8: 3yr_nf18`. The forward selector chases **exp_nf27**
  (the joint-WORST static config) for most folds — because past-fold cumulative Sharpe does **not** predict the
  next fold's best config. The in-sample winner 3yr_nf18 is picked only once (f8).
- **Nested-OOS LOSES to the incumbent: +0.85 vs +1.03 Sharpe (−0.18), +1.01 vs +1.16 Calmar.**
- **Paired (nested − incumbent) per-cycle diff −0.334 bps; 95% block-bootstrap CI [−1.571, +0.492] CROSSES 0.**

This is the same signature that killed K3-cost-margin, the decay-weighted sleeve, mom_180d, and every other
tuned-knob in this project: **wins in-sample, generalizes nowhere.** A cadence/window choice that only beats the
incumbent with hindsight fails the deploy bar.

---

## 4. Honest gates
- **G1 look-ahead — PASS** (inherited X132/X70 PIT pipeline; target_z `.shift(HORIZON)`; rolling train window
  uses only `open_time ≥ ec−window` / `exit_time < ec`; full-IC +0.010–0.015 well below the +0.10 leak flag).
- **G2 in-sample — several configs PASS** (3yr_nf18 +1.28/+1.43; 2yr_nf9 +1.19/+1.38; 2yr_nf18 Calmar +1.68)
  — but G2 alone is exactly what the nested-OOS test exists to discount.
- **G3 nested-OOS — FAIL (decisive):** honest forward config-selection +0.85/+1.01 < incumbent +1.03/+1.16.
- **G5 per-fold — the lift is f5-concentrated** (nf18's edge is one episode; it gives back f4). Not robust.
- **G6 paired CI — CROSSES ZERO** ([−1.57, +0.49]); nested not distinguishable from incumbent (and point-est worse).
- **G7 transport** — the qualitative shape (old data ~free; 1yr starves; nf27 over-frequent) is internally
  consistent across windows, but the cadence "improvement" itself does not survive forward selection, so there
  is nothing transport-positive to adopt.

---

## VERDICT — NO-CANDIDATE. Deploy retrain policy: EXPANDING (all available history), refresh ~quarterly.

Two clean, deployable **findings** (not a config change):
1. **Train on ALL available history (expanding).** Old data is neutral-to-mildly-helpful — `3yr ≈ expanding`,
   and shortening to **1yr HURTS** (Sharpe +1.03→+0.93). There is NO regime-drift penalty large enough to justify
   dropping history; data quantity dominates. (A 2yr rolling window is a defensible near-equivalent that mildly
   helps Calmar/PnL and never hurts Sharpe, if an operator prefers bounded memory — but it is NOT a Sharpe
   improvement and is not worth the added complexity given nested-OOS.)
2. **Retrain cadence is not a free Sharpe lever.** Faster retrain (~3.5mo) looks better in-sample but the lift is
   one-fold (f5) luck and **fails nested-OOS**; ~2.3mo (nf27) is already over-frequent (noisier OOS). A
   **~quarterly retrain** is a reasonable operational default — frequent enough to fold in new data/symbols,
   not so frequent that each retrain's OOS is noise — but choose it for operational freshness, **not** expecting
   a risk-adjusted gain.

**Forward expectation unchanged:** ~+1.0–1.2 Sharpe / +1.16 Calmar on this harder 2021-26 multi-episode panel
with the expanding/all-data model + iter-012 vol-norm stop. The training-config axis is now characterized and
closed: it confirms iter-032's lever (more training DATA) while showing the window/cadence KNOBS don't extend it.
