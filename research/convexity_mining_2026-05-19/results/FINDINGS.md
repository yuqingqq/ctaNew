# Convexity-Mining — FINDINGS (2026-05-19) — LINE CLOSED BY MEASUREMENT

Plan v1 got 3-agent **DO-NOT-PROCEED** (selection-endogeneity + vol-detector
confound; re-derives closed work). Per discipline, ran the single decisive
precondition the reviewers mandated instead of C0–C2.

## Decisive test (`C0pre_decisive.py`, pre-registered)
Convex-event labels rebuilt on the FULL 51-name panel (every (symbol,cycle)
in OOS folds, entered-or-not, both sides), BTC-beta-neutral realized residual;
12 PIT features; classifier OOS-symbol (5 disjoint groups, seed 20260519);
label-shuffled placebo.

| label | full-panel OOS-symbol AUC |
|---|---|
| big-up (pos)            | 0.669 |
| big-down (neg)          | **0.701** |
| big-\|move\| (abs)      | 0.753 |
| pos, vol-orthogonalized | 0.650 |
| placebo (shuffled)      | 0.500 |
| atr_pct alone pos / neg | 0.61 / 0.65 |

## Verdict: **B — pure volatility detector, not convexity-specific. LINE CLOSED.**
- Real & portable: the signature generalizes off the old-selector manifold
  (OOS-symbol ~0.67, placebo 0.50) — **not** pure selector-echo, and not a
  methodology artifact.
- **No directional content:** predicts big-DOWN ≥ big-UP (|pos−neg|=0.03);
  best at big-|move|. It identifies "name primed for a big idiosyncratic
  move," direction-agnostic — exactly the pre-registered kill condition.
- Direction is the unforecastable part (closed: hit 50.7%, per-cycle IC≈0.02).
  Vol-detector + no direction edge = no tradeable convexity strategy. The
  cost-concentration angle is moot (concentrating a directionless,
  non-portable book just makes a −0.33 thing cheaper, not profitable).

## Reconciliation (honest)
The forensic AUC 0.68 was a *genuine portable volatility detector*
mis-identified as convexity because its label was selection-endogenous and
one-sided. The 3-agent plan gate predicted this confound; the mandated
precondition confirmed it — the discipline prevented a wasted C0–C2 arc and
a likely confident-but-wrong "portable cost-efficient convexity strategy."

## Residual value (not a profit lever)
The vol detector is real and OOS-symbol-portable → potential **risk-control**
use only (size-down / avoid names primed for big moves; vol-targeting
overlay). NOT alpha. Does not change the standing conclusion: no portable
in-scope alpha; only un-refuted lever remains orthogonal data (OI/flow arc,
separate, running) and scope/horizon change.

Artifacts: `C0pre_decisive.py`, `C0pre_decisive.json`, `reviews/ROUND1_plan_review.md`.
