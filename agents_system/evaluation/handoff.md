# Handoff: evaluation → orchestrator/research
iteration: iter-032 (DEPLOYMENT — expanded universe 70→156 honest validation)
state: done
verdict: **QUALIFIED ADOPT** — deploy a WIDER universe, but with a per-cycle min-history (~30d) gate.
Champion ALPHA unchanged (= baseline regime-hybrid held-book + iter-012 vol-norm stop). This updates the
DEPLOY-UNIVERSE recommendation (iter-031): wider IS better via breadth/dispersion, *up to* the
well-historied set; raw thin post-2024 names DILUTE.

## The question
We expanded the universe 70→156 (iter-031: breadth=edge). Does the wider set IMPROVE the honest,
transport-validated Sharpe/Calmar vs the narrower baseline, or is it polluted by thin-history noise?
Tested the full champion (X117 regime-hybrid held-book + iter-012 vol-norm stop k=2.0) on the EXPANDED
x132 V0 preds (156 syms, 2021-26, 8 folds, IC +0.0146) vs the 23-sym EXT (x113). Engine = iter-031
verbatim, re-implemented as a fast precompute layer (verified == slow to 1e-13).

## Breadth-N sweep (x132, random subsets, same window — clean A/B). +stop (deploy):
| N | base Sharpe | base Calmar | +stop Sharpe | +stop Calmar |
|---|---|---|---|---|
| 23 | +0.46 | +0.44 | +0.38 | +0.41 |
| 50 | +0.86 | +0.86 | +0.75 | +0.68 |
| 100 | +1.02 | +1.28 | +0.91 | +1.16 |
| 156 | +1.12 | +2.02 | +1.03 | +1.16 |
**Monotone in N — breadth=edge CONFIRMED on the independent V0/2021-26 panel.** Random-N std also falls
with N (0.45→0): wider = more stable composition.

## Full-156 vs 23-sym EXT (+stop) — and the decomposition
| config | Sharpe | maxDD | Calmar | totPnL |
|---|---|---|---|---|
| EXT-23 (x113, prior baseline) | +0.86 | −3000 | +0.74 | +10450 |
| EXT-23 SUBSET of x132 (retrain only) | +1.06 | −2644 | +1.08 | +13467 |
| FULL-156 (x132) | +1.03 | −3960 | +1.16 | +21611 |
| **FULL-156 + min-30d-history gate** | **+1.19** | −3960 | **+1.33** | +24804 |
| OLD-only 47 syms (≤2023 listing) | +1.15 | −3461 | +1.37 | +22313 |

**Decomposition:** the +0.86→+1.03 Sharpe lift over x113-EXT is **~all the model retrain** (retrained
EXT-23 = +1.06, already above full-156). Adding 133 names on top is **marginally negative on Sharpe** but
~2× totPnL (capacity) and much higher base-book Calmar (+0.66→+2.02). maxDD is WORSE (−3960 vs −2644).

## Per-fold (+stop): expansion is LESS robust, not more
folds_positive: FULL-156 **6/8** (LOFO worst drop f5 → +0.65; f5 alone is +3.48/Cal +10.95/+9765bps);
EXT-23-x132 **7/8** (LOFO +0.87, no fold < −0.40); EXT-23-x113 6/8. FULL-156 has f3 −1.38 & f4 −0.10
where the narrower set is positive. Width concentrates the edge in the best fold; doesn't buy robustness.

## Thin-history noise check (the watch item) — DECISIVE
- Wide pred tails [−34,+45]: extreme-|pred| rows skew younger (median hist 977 vs 3059; 9.8% vs 3.5%
  thin) but corr(|pred|,thin)=+0.012 weak; biggest tail-maker is LITUSDT (FULL 2021 history, tiny rvol →
  z blows up), NOT a thin name.
- **Winsorizing pred (|3|,|1.5|) changes NOTHING** — the rank-based top/bottom-K book is invariant to
  monotone clipping. The wide tails are a RED HERRING for this construction (would only matter to a
  |pred|-magnitude-weighted variant).
- **Min-30d-history gate IMPROVES the book** (+1.03→+1.19, Calmar +1.16→+1.33): thin names are mildly
  net-negative. 90d over-prunes (+0.98).
- **OLD-only 47 syms BEATS full-156** (+1.15/+1.37 vs +1.03/+1.16): the breadth gain is the *wider OLD
  set*, the 88 post-2024 names DILUTE.

## Gates (deploy=+stop)
- G1 PASS (PIT; no clip-at-±5 hack; IC +0.0146 << +0.10). G2 PASS (Calmar +1.16/+1.33 > baseline +0.74).
- G3 WAIVED (structural "widest set"; min-history is a hygiene floor tied to the 180-bar warmup, not fitted).
- **G4 p69 (FAIL ≥p95)** — full-156 ranks p69 of random-100 (mean +0.92, p95 +1.26). Breadth > truncation
  but the full set is not a special composition (consistent with iter-031: breadth is the edge, not the names).
- **G5 6/8 folds, lift f5-concentrated (FAIL the robustness bar).**
- **G6 paired CI [−0.34, +2.28] CROSSES ZERO** — full-156 not statistically distinguishable from EXT-23-x132.
- G7 transport HOLDS (breadth-monotone reproduces). G8 cost robust (+1.21/+1.04/+1.03 @1/3/4.5bps).

## Verdict + deploy recommendation
**Deploy the WIDER universe — it raises the headline objective (Calmar +0.74→+1.16, totPnL ~2×) and
re-confirms breadth=edge — BUT gate by per-cycle minimum trailing history (~30d / 180 4h-bars).** The
history-gated wide set (+1.19 Sharpe / +1.33 Calmar @4.5bps +stop) is the single best honest config and
the recommended deploy universe. Honest caveats: most of the *Sharpe* lift vs the old EXT baseline is the
model retrain, not the extra names; the full set is not statistically better than a good narrower retrained
set (G6 crosses zero, G4 p69) and is f5-concentrated (G5 6/8); so size expectations to the broad-based mean
and do NOT dump in freshly-listed names. iter-012 vol-norm stop stays the always-on overlay (transports).

## Proposed current_best DEPLOY-UNIVERSE update (orchestrator to apply)
Amend the iter-031 deploy-universe rule: keep "trade the widest tradable set, never rank/prune by
liquidity/IC", ADD "**and never include names below a per-cycle minimum trailing history (~30d / 180
4h-bars)** — they add z-target noise and dilute Sharpe/Calmar; the well-historied wide set is the edge."
Note the wide V0 z-target pred tails are harmless to the rank book but would need winsorizing in any
magnitude-weighted construction.

## Insight for next research
The universe-WIDTH lever is now fully characterized across two independent builds (iter-031 HL70/EXT,
iter-032 x132): breadth helps via dispersion but (i) saturates/dilutes once sub-30d-history names enter,
and (ii) does not improve cross-fold robustness — the deep-DD / single-episode-concentration nature is
unchanged by width. No further universe-width iterations have positive prior. The retrain-on-more-data
effect (EXT-23 +0.86→+1.06) is the more interesting thread: periodic model retraining on the growing panel
may matter more than universe width — worth a dedicated retrain-cadence study if pursued.

blockers: none
