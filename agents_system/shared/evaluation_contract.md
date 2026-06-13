# Evaluation Contract (MANDATORY)

This is the system's anti-overfitting backbone. **No change is ADOPTED unless it passes every
applicable gate below.** Research proposals MUST pre-register against these gates;
Evaluation MUST report each one with a number; the Orchestrator rejects any ADOPT that skips a gate.

## TWO TRACKS (added iter-011)
- **ALPHA track** (iters 1–10): a change claiming to *predict/improve* — must beat the G4 matched
  random-timing placebo at ≥p95. This is for forecast-skill claims. We have proven no free signal
  leads the alt selloff, so the alpha track for DD-prediction is closed.
- **REACTIVE RISK-CONTROL track** (iter-011+): a change that does NOT claim prediction — a mechanical
  stop / de-gross / kill-switch that flags a drawdown EARLY (fast detector, iter-010 showed fast
  metrics flag ~21d sooner) and reactively caps the tail, ACCEPTING it can't forecast and will
  sometimes whipsaw. Objective here is NOT Calmar/alpha — it is **bounding the maximum drawdown for
  live capital preservation at a bounded, acceptable, robust cost.** G4 is reframed (below); the
  output is a *characterized DD-vs-cost trade-off + recommended config*, not a binary alpha ADOPT.

### Reactive-track gates (use INSTEAD of the alpha gates when the change is a mechanical risk stop)
| # | Gate | Requirement |
|---|---|---|
| R1 | Look-ahead | PASS (PIT trigger; equity/flag uses only realized-to-date info, lagged). |
| R2 | **Tail reduction** | Meaningfully cuts maxDD / worst-tail vs baseline on HL70 (the primary risk metric now). |
| R3 | **Bounded cost** | The return/Sharpe give-up is bounded and explicitly stated (a risk-preference trade is OK; a catastrophic return loss is not). Report the full DD-vs-cost trade-off curve over the trigger threshold. |
| R4 | **Concentration vs constant-de-gross** | Does triggering ON the drawdown cut the *tail* better than a constant flat-gross-reduction of equal average exposure? (The reframed G4: a stop should beat "just always run smaller" at capping the LEFT TAIL specifically — if it only matches constant-de-gross, say so; constant-de-gross is then the honest equivalent.) |
| R5 | **Cross-episode + universe robustness (DECISIVE)** | The stop must cap DD across **multiple** episodes (2021-26 EXT panel) and on S44 — a general risk rule, NOT fit to the f5 episode. Episode-LOFO: the tail-capping must not vanish dropping any one episode. |
| R6 | **Nested-OOS of the threshold** | If the stop has a threshold, choose it on past data and apply forward — the realized forward DD-capping must hold. |
| R7 | Re-entry sanity | The re-entry rule is defined and doesn't itself create a worse path (no buy-back-at-top pathology). |
A reactive stop is "ACCEPTABLE FOR DEPLOYMENT" (the reactive-track verdict) iff R1/R2/R5/R6 hold and
R3 cost is acceptable to the human — it is offered as a *risk option with a stated trade-off*, not an
alpha. The honest negative result is "reactive stops cut DD only at ~proportional return cost" (then
report the trade-off curve so the human picks a risk point).

Motivation: prior research on this strategy produced many in-sample "wins" that died under honest
validation — cost-margin swap (nested-OOS +0.24 vs +1.88 in-sample), decay-weighted sleeves
(paired CI crossed 0), the sign-flip layer (hit-rate 48.5% < coin-flip with multi-regime training).
Raw in-sample Sharpe is **not** evidence.

## Objective
Primary: **raise Calmar** (annualized return / |maxDD|) — i.e. reduce drawdown and/or raise Sharpe.
Secondary (tie-break / must-not-regress): Sharpe, maxDD, total PnL.
A change may trade a little Sharpe for materially less drawdown if Calmar improves and gates pass.

## Gates

| # | Gate | Requirement |
|---|---|---|
| G1 | **Look-ahead audit** (from Review) | PASS — all features trailing/shifted, labels purged by `exit_time`, train-only preprocessing. A FIX-NEEDED review blocks evaluation. |
| G2 | **In-sample objective** | Calmar > current_best Calmar (necessary, not sufficient). Report Sharpe/maxDD/Calmar/totPnL. |
| G3 | **Nested-OOS** (for ANY tuned/selected parameter) | Choose the parameter on past walk-forward blocks, apply forward; nested-OOS Calmar ≥ current_best. If the change has no tuned parameter (a structural choice like K, or an untuned rule), state that and G3 is waived. |
| G4 | **Matched placebo** | The effect must beat a matched-random control (matched basket-size / matched skip-rate / matched flip-rate, ≥100 seeds). Report percentile rank; require **≥ p95**. |
| G5 | **Per-fold robustness** | `folds_positive` reported; require improvement in **≥ 6/9 folds** OR a documented reason the concentration is acceptable (e.g. the lift isn't carried by 1–2 folds — check LOFO). |
| G6 | **Paired CI** | Block-bootstrap the paired per-cycle PnL difference vs current_best; **CI must not cross zero** for ADOPT. |
| G7 | **Universe robustness** | Evaluate on ≥2 universes (e.g. HL70 + 44-sym, or random subsets). The improvement must hold on the production universe (HL70) and not be a single-universe artifact. |
| G8 | **Cost realism** | Report at ≥3 cost levels incl. HL maker (~1bp) and taker (~3bp); the change must not depend on an unrealistically low cost. |

## Decision

- **ADOPT** iff G1 PASS, G2 improved, G3 (if applicable) holds, G4 ≥ p95, G5 met, G6 CI clears zero, G7 holds on HL70, G8 robust.
- Otherwise **REJECT** with the failing gate(s) named and an insight for the next cycle.
- "Mixed" results (improves in-sample, fails honest gates) are **REJECT** — and that is a *useful* result; log the lesson.

## Standard measurement conventions (see also conventions.md)
- Annualization: cycles/yr = `365 * 288 / horizon_bars` (4h horizon = 48 bars → √(6·365)).
- maxDD on cumulative bps equity (additive on constant gross notional).
- Walk-forward: 9 expanding folds, 1-day embargo, label purge by `exit_time`.
- Placebo seeds ≥ 100; bootstrap blocks by fold.
- Always report on the **production universe (HL70)** plus at least one robustness universe.
