# Alpha integration & testing plan (convexity v3) — DRAFT

## Why one framework is not enough
v3 is **not** a pure prediction system. It applies different machinery per regime, and in some regimes it
barely uses the model pred at all. Direct evidence (2026-07-06, by-regime PnL):

| bull book | bull PnL | mechanism |
|---|---|---|
| v3 FULL (defensive) | **+4,250** | BTC hedge + `return_1d` ranker + DEEP_THR sit-out + hold=1 |
| pure mean-rev pred in bull | **−916** | rank by pred, full alt long |

The bull book's PnL comes from the **mechanism, not the prediction**. So the framework I used so far —
*add alpha to `V0_LEAN` → retrain → new pred → full backtest* — only tests **one** integration point (the
pred), and that point only drives selection in **side** (and partly **bear**). It structurally **cannot** reveal
an alpha whose value is as a **bull ranker** or a **risk/gate signal**. That is the gap this plan fixes.

## The integration points (where an alpha can actually enter v3)
| # | Point | What it changes | Regimes it acts in | Current baseline it replaces/augments |
|---|---|---|---|---|
| **I1** | Model feature (`V0_LEAN`) | base + residrev pred | side, bear (NOT bull) | the 14-feature pred |
| **I2** | **Bull short ranker** | which alts are shorted in bull | **bull only** | `BULL_SHORT_RANK=return_1d` |
| I3 | Long-book ranker (residrev) | long pick | side, bear | residrev pred |
| I4 | Regime gate / gross modulation | book gross | all (regime-timed) | perf regime gate (trailing edge) |
| I5 | Sizing signal | per-name weight | all | `inv_sqrt_vol` |
| I6 | Standalone blended sleeve | a separate book, weight-blended | chosen regime(s) | n/a (new sleeve) |

**Status:** I1 is tested and REJECTED across raw / beta-neutral / per-regime frames (funding, sector, alpha191:
alpha065/082/095/023/052/159/010). I2–I6 are **untested**. The promising lead (alpha070 broad bull alpha) is an
**I2** candidate — a different integration point than everything rejected so far.

## Measurement framework (per integration point)
For each candidate, the test is always: **match execution, vary only the one integration, measure where that
integration is active.** Never dilute a regime-local alpha across all regimes.

1. **Signal layer (fast, gating):** in the target regime, standalone L/S distribution — mean, **median**, %pos,
   **top-3 cycle concentration**, **per-cycle Sharpe**, non-overlapping t, sub-period halves, placebo. Reject if
   tail-driven (median≈0, high concentration) — that was the alpha095 mirage.
2. **Capture layer:** naked gate-free book with the alpha at its integration point vs the current baseline, split
   by regime. Confirms the signal converts to capturable L/S.
3. **Strategy layer:** full frozen v3 with the one integration swapped, by-regime PnL + net Sharpe + maxDD +
   per-fold. This is the only number that decides adoption.
4. **OOS:** repeat 1–3 on the fullhist pred/data (2022–2026) before any production change.

## Fair-comparison rules
- Baseline = the **current mechanism at that point** (e.g. I2 baseline = `return_1d` bull ranker), NOT the pred.
- Vary only the one integration; hold cost, funding, universe, K, hold, all other regimes fixed.
- Score in the **active regime**; also report all-regime net (must not harm other regimes).
- Robustness gate (all required): broad (median≈mean, top-3 share < ~40%), non-overlap significant, both
  sub-period halves same sign, placebo-clean, **and** OOS-stable. Aggregate mean alone is NOT sufficient.

## Priority order (from findings so far)
1. **I2 — bull short ranker: alpha070 (or alpha010) vs `return_1d`.** Highest priority: bull is where the pred
   fails (−916) and the mechanism wins (+4,250), and alpha070 shows broad, sub-period-stable bull alpha
   (non-overlap Sharpe +1.52, median≈mean, top-3 share 20%) that the pred lacks. Test: does alpha070 beat
   `return_1d` as the bull ranker, in bull PnL and net?
2. I4 — dispersion/turnover as a regime-gate or gross modulator (turnover-dispersion factors screened strong).
3. I3 — long-book ranker (long side is the weak leg; top-decile IC only +2.7).
4. I6 — standalone blended sleeve for a regime-local alpha that doesn't fit I2–I4.

## What is already settled (do not re-run)
- I1 (model feature) for all screened Alpha191 / funding / sector factors: **rejected**, robustly, in raw,
  beta-neutral, and per-regime frames. The K=1/2 selection + already-broad side/bear base make them redundant.
- K tuning: **K_short=3 / K_long=1** is a separate, adopted-pending-OOS win (+0.44 Sharpe, −32% maxDD).

## Open decision for the user
Confirm the priority (I2 bull-ranker first?) and the robustness bar before I run the I2 test. The key risk to
guard against is another alpha095 (aggregate-mean win that is tail-concentrated) — hence the median/concentration/
sub-period/placebo gate at the signal layer *before* spending backtest compute.

---
# FAST-CHECK PROTOCOL (how to cheaply test a new feature or mechanism)

The goal: a seconds-to-minutes gate that screens out most candidates BEFORE the slow full backtest, encoding the
lessons from this session. Two paths, because a **feature** (enters the pred) and a **mechanism** (enters position
construction) fail in different ways and need different checks.

## Rule 0 — always, before any check
- **PIT-shift** the signal (use only closed bars ≤ t−1). Skipping this gave a fake Sharpe +21.
- **Beta-neutralize** it (residualize vs trailing BTC beta). Raw-price signals are mostly market beta, which the
  beta-neutral book can't farm. (alpha065 looked great raw, vanished beta-neutral.)

## Path A — new FEATURE (adds to V0_LEAN → pred)  ⏱ seconds
Tool: `live/quick_ic_delta.py` (pooled WF ridge ΔIC, per regime).
1. Compute **ΔIC vs V0_LEAN**, overall AND per regime.
2. GATE: keep only if ΔIC > ~**+0.003** in a regime that **uses the pred** (side/bear). 
   - ΔIC ≈ 0 → reject (redundant). alpha070/alpha095 died here.
   - ΔIC only in **bull** → reject: **bull ignores the pred** (uses return_1d), so a feature can't act there.
3. **IC is a screen-OUT, not a confirm.** Calibration from this session: alpha082 had ΔIC +0.0045 (passed step 2)
   but LOST the strategy backtest (−0.27 Sharpe) — because the lift was in already-strong side/bear and didn't
   survive the K=1/2 selection. So: ΔIC>0 → *proceed* to Path C; it does NOT mean adopt.

## Path B — new MECHANISM (ranker / gate / sizing / hedge; pred unchanged)  ⏱ seconds
Tool: `live/bull_ranker_test.py` / `live/alpha_naked_regime.py` (capture-layer, no bot, no retrain).
1. Measure the **specific leg/decision the mechanism changes**, directly from preds+panel, **in the active regime**,
   vs the incumbent mechanism (e.g. alpha070 vs return_1d for the bull short leg).
2. GATE on the **distribution, not the mean** (this is the alpha095 lesson):
   - non-overlapping per-cycle Sharpe **> 0**, AND
   - both **sub-period halves same sign**, AND
   - **top-3 cycle share < ~40%** (not tail-driven), AND
   - **placebo-clean** (shuffle target → IC collapses).
   - A high aggregate mean with median≈0 / top-3>60% = mirage → reject. (alpha095 bull: mean +61, median 0, 3 cycles.)

## Path C — STRATEGY confirm (only if A or B passes)  ⏱ minutes
Full frozen v3 with the ONE change; report **by-regime PnL + net Sharpe + maxDD + per-fold**. Baseline = the current
mechanism at that point. This is the only number that decides adoption. Must not harm other regimes.

## Path D — OOS (before production)
Repeat A–C on fullhist (2022–2026). Discrete architecture changes (K, ranker choice) generalize; tuned continuous
params usually don't.

## One-line decision tree
PIT+beta-neut → is it a feature or mechanism? → (feature: ΔIC per-regime, reject if ≤0 or bull-only) OR
(mechanism: capture-layer distribution gate) → if pass, strategy backtest by regime → if pass, OOS. Never adopt on
aggregate mean or IC alone.

---
# UPGRADE (2026-07-06): screen on TIP-ACCURACY, not average IC

Deep-cause finding: this strategy trades the TIP of the pred distribution (top-1 long / top-2 shorts), where the base
alpha is hyper-concentrated (single most-extreme short realized −88 bps; whole top-decile avg only −1.7). IC is a
whole-cross-section average — a feature can raise it by fixing the untraded MIDDLE while DILUTING the tip's
high-conviction picks. So average ΔIC is the WRONG screen (necessary-not-sufficient, and noisy: flips sign pooled vs
per-symbol). Replace Path-A's ΔIC gate with:

## TIP-ACCURACY screen (Path A', from preds only — no bot, no retrain-of-full-stack)
Tool: live/tip_accuracy.py. For candidate model/feature vs baseline, at the traded K (1,2,3), measure top-K/bottom-K
realized fwd-alpha L/S on THREE axes:
  1. mean (edge magnitude),  2. per-cycle Sharpe (reliability),  3. hit-rate (% cycles positive).
GATE: keep ONLY if it improves the tip on mean AND reliability at the traded K. Improving average IC while the tip
degrades = REJECT (that's the exact IC↔PnL trap).
Calibration (2026-07-06): +factors avg IC −0.005 (ambiguous) but tip K=2 mean +99→+60, reliability +4.66→+2.93,
hit 59→55% -> tip screen says REJECT -> backtest confirmed −0.67 Sharpe. Tip metric predicts; avg IC doesn't.
A feature helps ONLY if it makes the TIP steeper + more reliable, not on average IC.
