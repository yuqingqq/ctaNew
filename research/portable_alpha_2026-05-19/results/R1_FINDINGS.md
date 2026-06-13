> **CORRECTION (2026-05-19, post results-review).** The "broadly diversified
> per-cycle risk (Herfindahl 0.094)" and "robust ex-VVV +1.99" claims below
> were OVER-CLAIMED. Herfindahl here is on year-cumulative *signed* per-name
> PnL (mechanically diffuse). The honest per-cycle *gross-weight* Herfindahl
> is ≈0.19 (~5 effective names — concentrated); ex-VVV concentration ROTATES
> to the next tail name (R1c). Frontier/cost/reconstruction numbers below are
> correct; the concentration interpretation is superseded by
> `R1c_concentration_truth.json` and `R4_SYNTHESIS.md` (v2). Kept for audit.

# R1 — Honest Baseline + Concentration Frontier: FINDINGS (2026-05-19)

Reconstruction validated: uncapped equal/flat-4.5 reproduces the known V3.1
headline **exactly** (Sharpe +2.229, maxDD −3445, 7/9 folds, totPnL +8385) →
the machinery is faithful, so every number below is trustworthy.

## Pre-registered prediction vs outcome (PLAN.md R1)

| prediction | outcome | status |
|---|---|---|
| uncapped Sharpe ∈ [+1.5,+2.6] | **+2.23** | HIT |
| Herfindahl ≥ 0.40 (concentrated) | gross-risk H = **0.094** | **MISS** |
| frontier monotone-decreasing in tightness | yes (1/2=inf, 1/3 .93, 1/5 .84, 1/8 .73) | HIT |
| c=1/3 Sharpe ∈ [+0.6,+1.8] | **+2.06** (8/9 folds) | MISS (higher/better) |
| drop-5 mean ≤ +1.2 | **+2.14** (worst +0.37, 0 neg/30) | MISS (higher/better) |

Per the locked rule, prediction misses **rewrite the Diagnosis, not the gate.**

## The decisive reconciliation (R1b)

The R1 metric (gross |per-cycle PnL|) and PROGRESS.md's "62% from VVV" are
**different metrics; both are correct and they tell opposite stories:**

- **Per-cycle gross risk concentration: LOW.** Herfindahl 0.094 (≈11 effective
  names). The 6-sleeve net-weight book takes broadly diversified risk every
  cycle — it is *not* a single-name bet.
- **Cumulative net-PnL concentration: EXTREME.** VVVUSDT = **+86.1%** of total
  net PnL (top-3 = 132%, i.e. large offsetting flows). On the faithful sleeve
  reconstruction the figure is *higher* than PROGRESS.md's 62%.
- **Yet the edge is robust to removing the concentrating name:**
  ex-VVV Sharpe = **+1.985** (Δ −0.244), and drop-5 (often includes VVV) mean
  **+2.14**, worst-of-30 **+0.37** (zero negative draws).

Why both are true: total net PnL is a small residual of large offsetting
per-name flows, so "86% of net" overstates *risk* importance; the construction
diversifies *risk* (gross H 0.09) so VVV's Sharpe contribution ≪ its PnL share;
when VVV is dropped the rolling-IC + K=3 refill simply rotate to other names
and the diversified engine continues.

## Diagnosis rewrite (honest, per the pre-registered rule)

The project's prior belief — "edge is 62–86% VVV, fragile, universe-overfit,
honest broad ≈ +1.0–1.5" — is **wrong as a deployability statement.** Correct:

> The V3.1 6-sleeve construction takes **broadly diversified per-cycle risk**
> (gross Herfindahl ≈ 0.09), is **robust to symbol removal** (drop-5 mean +2.14,
> worst +0.37, 0/30 negative; ex-VVV +1.99), and **survives cost stress**
> (flat-9 bps +1.96; realized √ADV +2.13). The deployable, concentration-robust
> Sharpe is **≈ +2.0** (ex-VVV / drop-5 mean), not +1.0–1.5. The pessimistic
> "+1.63 fixed-universe" reading was a universe-confound, not the honest broad
> number. The single legitimate residual risk is **operational, not
> statistical**: 86% of *cumulative dollar* PnL routed through one low-float
> name (VVV) is a live liquidity/delisting exposure even though Sharpe survives
> its removal — handled by deployment choice (vol-norm or cap-1/3 variant) +
> the R3 kill-switch, exactly the frontier-decides-tradeoff the user asked for.

## Deployable-criteria check (PLAN.md) — R1 ALONE is deployable

| criterion | uncapped equal | cap 1/3 equal | vol-norm uncapped |
|---|---|---|---|
| 1. Sharpe ≥ +0.8 net (4.5bps) | +2.23 (ex-VVV +1.99) ✓ | +2.06 ✓ | +2.14 ✓ |
| 2. ≥6/9 folds positive | 7/9 ✓ | **8/9** ✓ | 7/9 ✓ |
| 3. ≥+0.5 @9bps AND @√ADV | +1.96 / +2.13 ✓ | (≥, monotone) ✓ | ✓ |
| 4. cost-of-cap retention ≥70% | 1.00 ✓ | 0.93 ✓ | 1.00 ✓ |
| 5. maxDD ≤ 1.5× uncapped DD | −3445 (ref) ✓ | −3580 ✓ | **−2864** ✓ |
| drop-5 mean ≥? / worst | +2.14 / +0.37 | +2.07 / +0.27 | (equal-proxy) |

**Decoupled delivery (per profitability reviewer): R1 already yields a
deployable, robust, concentration-controlled system.** Recommended deployment
candidate = **vol-norm uncapped** (Sharpe +2.13, lowest maxDD −2864, lowest
gross H 0.081) or **equal cap-1/3** (8/9 folds, +2.06) — both reduce single-name
dollar dominance vs raw uncapped while holding Sharpe ≈ +2.0. R2 (profit levers)
is now upside research on top of an already-deployable baseline.

Scripts: `R1_baseline_frontier.py`, `R1b_concentration_reconcile.py`.
Data: `R1_frontier.csv`, `R1_results.json`, `R1b_concentration.json`.
