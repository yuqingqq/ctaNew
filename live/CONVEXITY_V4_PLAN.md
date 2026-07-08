# Convexity v4 — residual-aligned target: plan

## What v4 is (one change from v3)
v4 = the **exact v3 stack**, with the two-book model **trained on the residual it farms**:
`xs_z(return_pct)` → **`xs_z(alpha_vs_btc_realized)`**. Nothing else changes.

**Rationale (consistency):** the strategy farms cross-sectional BTC-beta residual alpha, but the v3 model trains on
raw-return rank. Training on the residual aligns the objective with what is harvested. This is a *correctness* change
first, a PnL change second.

**Evidence so far (fair simple-book test — gate-free, only the label differs):**
- residual ≥ return at every K; **tied at symmetric K, +0.51 daily Sharpe at production K (K_long=1/K_short=2)**.
- Improvement lands in **side (+138→+157) and bear (+59→+94)** — the regimes v3 *uses* the pred; the bull weakness
  (+24→−3) lands where v3 *ignores* the pred (uses `return_1d`). Regime profile fits v3.
- Monthly breadth **8/9 months** residual>return.

**Honest caveat (do not oversell):** the +0.51 edge is cycle-concentrated (median 0, %pos 20%, top-3 share 61%) and
**decaying** (half-1 +39.8 → half-2 −0.4). So expect **small/neutral net Sharpe**; v4's justification is
consistency + not-worse + monthly-broad, NOT a large lift. The half-2 decay is the #1 thing OOS must resolve.

## What stays IDENTICAL (frozen from v3 — do not touch in v4)
- Model: per-symbol RidgeCV, `V0_LEAN` features, HL=60 recency, WF cuts, two books (base + residrev), per-symbol norm.
- Strategy: K=2/1, `inv_sqrt_vol`, REGIME_GATE, `BULL_DEEP_THR=0.15`, bear depth-ramp, `SHORT_MIN_RET3D`, conc_cap,
  `BULL_SHORT_RANK=return_1d`, cost 9 + depth, funding, sleeves.
- **K_short=3 stays a SEPARATE candidate** — do NOT bundle into v4 (own discrete-architecture validation). v4 isolates
  the target change; K_short=3 can layer on later as v4.1 after its own OOS.

## Confounds — status
- ✅ **Label leak:** residual's beta = `rolling(180,min42).cov/var).shift(1)` — trailing, PIT, no look-ahead. Clean.
- ⚠️ **Config-confound:** v3's gate thresholds (REGIME_GATE, BULL_DEEP_THR, conc_cap) were tuned to *return*-target
  preds. v4 feeds *residual*-target preds into that same frozen config. Do NOT retune the gates for v4 (that would
  overfit) — instead verify they still behave, and if one clearly degrades, flag it rather than patch it.
- ⚠️ **Half-2 decay:** validate OOS before trusting any in-sample lift.

## Validation gates (pass ALL before v4 goes live-parallel; pass + clear win before it replaces v3)
1. ✅ Fast simple-book (done) — residual ≥ return, monthly-broad. [caveat: decaying]
2. **Strategy-layer at fixed K:** retrain both books on residual → run frozen v3 config → report by-regime PnL,
   net daily Sharpe, maxDD, per-fold. Gate: net Sharpe ≥ v3 (≥ neutral), maxDD not worse, no regime harmed,
   per-fold not concentrated.
3. **Gate-behavior check:** confirm REGIME_GATE / BULL_DEEP_THR / conc_cap still fire sensibly on residual preds.
4. **OOS fullhist (2022–2026):** the decisive gate for the half-2 decay concern.
5. **LOO/placebo re-confirm:** the v3 levers still earn their keep with residual preds (regression guard).

## Deployment (only after gates 2–5 pass)
Mirror the v3 deploy package, new artifacts:
- `train_v4_artifact.py` (per-symbol RidgeCV on residual target; base + residrev), `predict_v4_incremental.py`,
  `run_convexity_v4_live.sh`, `parity_v4.sh`, `V4_LIVE_DEPLOY.md`.
- Run as a **parallel forward-test** (separate state dir, alongside v1/v3) — do not replace v3 until v4 wins live +
  OOS. Training box retrains monthly on residual target + pushes; live box pulls.

## Decisions for the user
1. v4 = target change ONLY (recommended), K_short=3 kept separate? 
2. Deploy v4 **parallel** to v3 first (recommended), or replace?
3. Adoption bar: **consistency + not-worse** (recommended, given the honest weak/decaying edge), or require a clear
   Sharpe win at gate 2/4?

## Expected outcome
Realistic: v4 ≈ v3 net Sharpe (small + / neutral), with a cleaner, more defensible design. Treat v4 as the
correct-by-construction baseline; the K_short=3 lead is the bigger separate EV.
