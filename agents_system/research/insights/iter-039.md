# iter-039 — Risk-managed / model-gated new-listing short — NO-CANDIDATE

**Scope (human idea, sharpening iter-037):** iter-037 found new perps FADE (median −19%/30d, 64%
down, every cohort) but a NAKED short has mean≈0 because ~8% moonshot 2–5× (one 5×=−500% on a short
eats ~100 winning shorts). Two sharper ideas: (A) RISK-MANAGE the short with a HARD STOP + position
cap (defined-risk-on-perps) to cap the moonshot loss; (B) TRAIN A MODEL on early-window features to
predict moonshot-vs-fade and gate the short. Tested both HONESTLY: realistic gap-slippage on the stop,
cohort-OOS for the model, G4 placebo, event-bootstrap CI, cohort transport.

## Data
- Same 163 usable listing events as iter-037 (first close ≥2023-02-01; 28/53/78/4 in 2023/24/25/26),
  5m→1h OHLCV. Funding-from-listing only for late-2024/2025 (146). Cost 15 bps/leg (RT 30), swept 5–30.
- Moonshot defined as ret(day3→day30) ≥ +50% (the move that kills the short): **25 moonshots total
  (6/12/7 per cohort)** — THIN.

## (A) RISK-MANAGED STOPPED SHORT — caps the tail, does NOT rescue the mean

Stop = cut the short if hourly HIGH ≥ entry·(1+X). **Gap honesty:** fill at `max(trigger, breach-bar
CLOSE)` (a violently-pumping bar closes far above the trigger) — and a harsher `next`-bar-high model.

| config (gap=close) | n | mean | hit | t | boot CI95 | P(>0) | worst |
|---|---|---|---|---|---|---|---|
| naked short@3d→30d (no stop) | 163 | −0.030 | 66% | −0.50 | [−0.154,+0.085] | 32% | **−4.13** |
| short@3d→30d **stop+30%** | 163 | **+0.035** | 50% | +1.16 | [−0.023,+0.094] | **88%** | **−0.99** |
| short@3d→30d stop+50% | 163 | −0.0003 | 55% | −0.01 | [−0.071,+0.071] | 50% | −0.99 |
| short@3d→14d stop+30% | 163 | +0.019 | 56% | +0.76 | [−0.030,+0.064] | 77% | −0.99 |
| short@3d→30d stop+30% **[next-gap]** | 163 | +0.011 | 50% | +0.35 | [−0.052,+0.074] | 64% | −1.41 |

**The stop genuinely caps the tail** (worst −4.13 → −0.99). The tightest stop (+30%) flips the pooled
mean positive (+0.035, P>0=88%). BUT: (1) it never clears P(>0)≥95%; (2) under the harsher (and more
realistic for a violent pump) `next`-bar gap model the best config decays to +0.011 / P>0=64%;
(3) hit-rate falls to 50% — the stop also cuts winning continuations, so it's "avoid the moonshot" not
"keep the fade body."

**Cohort transport FAILS (the wall):** stopped short@3d→30d stop+30% by listing year —
- 2023 (n=28): mean **+0.004** (P>0=51%)
- 2024 (n=53): mean **−0.036** (P>0=20%)
- 2025 (n=78): mean **+0.089** (P>0=96%)

stop+50%: 2023 −0.027 / 2024 **−0.089 (P>0=4%)** / 2025 +0.064. The entire positive is the 2025
alt-bear cohort; the short LOSES in 2024. Cost is not binding (mean ≈0 even at 5 bps). Same
universe/regime-overfit sign-flip as iter-037.

## (B) MODEL to predict moonshot-vs-fade — no OOS skill on the tail

Early-window features (PIT, first 3d): ret_1d, ret_3d, rv_3d, maxrunup_3d, maxdd_3d, fund_mean,
fund_last. Targets: P(moonshot) (classification) and ret30_d3 (regression). Cohort-OOS:
train 2023→test 2024; train 2023-24→test 2025.

**Moonshot-prediction OOS AUC:**
| split | train moon | test moon | AUC logit | AUC lgbm |
|---|---|---|---|---|
| 2023→2024 | 6 | 12/53 | **0.500** | 0.540 |
| 2023-24→2025 | 18 | 7/78 | 0.650 | 0.636 |

2023→2024 is **random** (AUC 0.50); the 2025 split is only modest (~0.64) and rests on 7 test
moonshots. Regression IC vs ret30_d3 is +0.12 (2024) then −0.02 (2025) — sign-inconsistent OOS.

**Model-gated short (skip top predicted-moonshots), OOS:** 2023→2024 the gate LOSES (mean −0.05 to
−0.11, P>0 3–22%) — no rescue, 2024 is negative regardless. Only 2023-24→2025 looks good
(+0.08 to +0.11, P>0 91–96%) — again the favorable cohort.

**G4 PLACEBO (decisive):** does skipping the model's top-decile predicted-moonshots beat skipping a
RANDOM same-count subset?
- 2024 test: real gated mean −0.102 ranks **p24** of random skips (WORSE than random).
- 2025 test: real gated mean +0.083 ranks **p82** of random skips — **below the p95 bar.**

The model adds no skill the random skip doesn't; the gate ≈ "short fewer names." The univariate early
features have descriptive IC vs ret30 (maxdd_3d +0.30, rv_3d −0.17, ret_1d +0.16) but it does NOT
transport into an OOS moonshot classifier — 25 events / a non-stationary tail is too thin to learn.

## The mechanism that kills BOTH ideas: the moonshot tail is NON-STATIONARY

| cohort | moonshot rate | fade rate | mean ret30_d3 | median ret30_d3 |
|---|---|---|---|---|
| 2023 | 21% | 61% | **+0.24** | −0.16 |
| 2024 | 23% | 55% | **+0.16** | −0.07 |
| 2025 | 9% | 74% | **−0.13** | −0.29 |

The fade (median) is in every cohort, but the **MEAN sign-flips because the moonshot frequency/severity
is regime-driven** — 21–23% of 2023–24 listings moonshot (alt-bull) vs 9% in 2025 (alt-bear). The short
only profits when moonshots are rare (2025). A stop caps each individual moonshot loss but cannot make
the *frequency* stationary; a model cannot predict *which* names moonshot OOS because the tail is a
regime property, not a name property the early features capture. **Both ideas reshuffle the same
fat-tail/regime problem rather than solving it.**

## Verdict: NO-CANDIDATE
1. **Stopped short:** the stop DOES cap the tail (worst −4.13 → −0.99) and the tightest stop flips the
   pooled mean positive (+0.035, P>0=88%) — but never clears P(>0)≥95%, decays under harsher gap
   assumption, and **sign-flips across cohorts** (2024 −0.089 / 2025 +0.089). Capping the per-event loss
   ≠ stabilizing the regime-driven moonshot frequency.
2. **Model:** moonshot-prediction OOS AUC ≈0.50 (2024) / 0.64 (2025) on 25 events; the model-gated short
   **FAILS the G4 random-skip placebo** (p24 in 2024, p82 in 2025 < p95). No learnable moonshot skill.
3. **Cohort transport FAILS for both** — the entire positive lean is the 2025 alt-bear cohort, identical
   to iter-037 and the rejected net-short/trend family (iter-007/008/021).
4. Cost is not the binding constraint (mean ≈0 even at 5 bps).

**New listings stay EXCLUDED via the maturity≥180d filter (iter-032/035/036).** Risk-management caps the
loss tail honestly but the residual edge is a 2025-regime artifact; the model cannot predict the
non-stationary moonshot tail from 25 events. To trade the fade body you would need a genuinely
capped-loss instrument AND a stationary moonshot frequency across full bull/bear cycles — neither
available on free perp data with ~163 events. Champion + universe standard UNCHANGED.

Scripts: `iter039_stopped_short.py` (stop + gap models + transport + cost),
`iter039_model.py` (early feats + cohort-OOS classifier/regressor + model-gated short),
`iter039_placebo.py` (G4 gate placebo + moonshot stationarity).
