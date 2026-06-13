# Evaluation report — iter-022 (ALPHA track)

**Change:** In the **sideways** regime only, replace the held-book side rank key with an
equal-weight cross-sectional reversal ensemble: `score = z_xs(pred) − z_xs(rel_ret_1d)`
(per-cycle XS z-scores, fixed 0.5/0.5, UNTUNED). Long top-K / short bottom-K, existing
beta-neutral leg sizing. Bull (mom30) and bear (FLAT) regimes unchanged. K=5, 6 sleeves, 4.5bps fixed.
Source signal `rel_ret_1d = return_1d − per-cycle XS mean(return_1d)` (PIT, no new data).

**Pre-check that motivated this:** the signal PASSED the universe-transport IC pre-check —
XS IC vs forward 4h alpha-residual **HL70 −0.0360 (t −9.76)**, **EXT −0.0302 (t −12.33)**,
SAME negative sign both universes, significant in EVERY year 2021–2026, and **orthogonal to
`pred`** (XS corr −0.022; IC survives at −0.0350 after removing `pred`, ~8.5× stronger than
`pred`'s own +0.0042 IC). This is exactly where funding (i21) / mom180 (i15) / alt-bear (i7) died.

**Verdict: REJECT. Champion UNCHANGED (= baseline, Calmar +1.68 + optional iter-012 vol-norm stop).**

All headline numbers below were independently re-derived from the implementation per-cycle
parquets (`results/X131_percycle_{HL70,EXT,S44}.parquet`) — Sharpe/Calmar match to the digit,
and the gross-PnL collapse + turnover deltas reproduce exactly via the cost-slope (net@1bp vs
net@4.5bp → gross at cost=0).

## Headline (@4.5bps) — verified from parquets

| universe | arm | Sharpe | maxDD (bps) | Calmar | totPnL net (bps) | GROSS PnL (bps) | turnover |
|---|---|---|---|---|---|---|---|
| HL70 | base (=X117) | **+1.93** | −5674 | **+1.68** | +10472 | **+12272** | 800 |
| HL70 | ens | +0.86 | −9818 | +0.49 | +5301 | **+7241 (−41%)** | 862 (+7.8%) |
| EXT | base | **+0.87** | −4953 | **+0.66** | +15448 | **+20656** | 2315 |
| EXT | ens | +0.19 | −9559 | +0.08 | +3379 | **+9604 (−54%)** | 2767 (+19.5%) |
| S44 | base | **+1.84** | −4170 | **+2.10** | +25620 | **+29875** | 1891 |
| S44 | ens | +0.56 | −8102 | +0.34 | +8108 | **+12913 (−57%)** | 2135 (+12.9%) |

Base reproduces X117 to the digit on HL70.

## Gate table

| Gate | Requirement | Result | Verdict |
|---|---|---|---|
| **G1** look-ahead | PASS from Review; flag if IC>+0.10 | Review **PASS**: `rel_ret_1d` window `[t−288,t]` vs forward target `[t,t+48]` share only the boundary price `close[t]` known at t — no overlap. XS demean within-cycle (groupby open_time), PIT. Pre-check IC −0.036 reproduced exactly, far below +0.10 leak flag; re-rank LOSES gross (opposite of a leak signature). | **PASS** |
| **G2** in-sample objective | Calmar > 1.68 (HL70) | Calmar +0.49 (HL70) / +0.08 (EXT) / +0.34 (S44) — all **far below** base. Sharpe down on all three; maxDD WORSENS 73–94% deeper. | **FAIL** |
| **G3** nested-OOS | required only for tuned param | Weight fixed 0.5/0.5, untuned/structural. | **WAIVED** |
| **G4** matched placebo | ≥ p95 | Within-cycle SCORE-SHUFFLE of `rel_ret_1d` (break rel→symbol link, rebuild same ensemble + BN weights, 200 seeds): HL70 real Calmar +0.49 vs placebo p50 **+2.27** / p95 +3.65 → **rank p0**; EXT real +0.08 vs placebo p50 +0.52 / p95 +0.70 → **rank p0**. The real ordering is WORSE than a random within-cycle re-rank. | **FAIL (p0)** |
| **G5** per-fold | ≥6/9 folds (or LOFO not 1-2-fold) | ens better than base in HL70 **2/7**, EXT **1/8**, S44 **3/8**. fold-LOFO: lift stays NEGATIVE dropping each fold → loss is broad, not concentrated. | **FAIL** |
| **G6** paired CI | CI must not cross zero (positive) | HL70 obs diff (ens−base) **−2.150 bps/cyc**, CI [−6.407, +1.871] crosses 0. EXT **−1.173 bps/cyc**, CI [−2.379, −0.220] **clears 0 NEGATIVE** (significantly HURTS). No universe shows a positive CI clearing 0. | **FAIL** |
| **G7** universe / PnL transport | improve HL70 AND hold on EXT+S44 at PnL layer | Fails HL70 outright. EXT per-episode PnL ens>base in only **1/4** episodes; 2025_q4 INVERTS the strongest base episode (base +4834 / Sh +4.83 → ens −3123 / Sh −3.17). The IC transports across universes; the PnL does NOT. | **FAIL** |
| **G8** cost realism | hold at 1/3/4.5bps; check GROSS | **GROSS PnL FALLS** on all three (−41/−54/−57%) → the loss is NOT a cost artifact; pre-cost return is destroyed. Turnover rises only modestly (+7.8/+19.5/+12.9%). Even at 1bp ens loses by a wide margin (HL70 net +6810 vs base +11872). | **FAIL (no edge to begin with)** |

## The IC-vs-PnL analysis (the point of this iteration)

`rel_ret_1d` has a strong, era-stable, orthogonal univariate cross-sectional IC against the
forward 4h alpha-residual — it PASSED every pre-check we built to fail-fast bad signals
(transport-first sign consistency, era stability, orthogonality to `pred`). Yet it **REJECTS at
the PnL layer on every universe and every honest gate**, and the decisive tell is at the **GROSS**
layer: pre-cost return collapses 41–57%. This is not cost; the re-rank actively *destroys* edge.

The mechanism: the production `pred` book is **already a beta-neutral cross-sectional
mean-reversion basket**. The productive part of the cross-sectional reversal is therefore *already
absorbed* by `pred`. Overlaying `−z(rel_ret_1d)` re-ranks the same names from a **noisier,
higher-turnover angle that trades AGAINST `pred`** — so the ensemble doesn't add the reversal,
it *dilutes/inverts* the part `pred` already captured. G4 confirms this directly: the real
ordering ranks at **p0** — strictly *worse* than a random within-cycle shuffle of the same
signal. A random re-rank that merely adds noise to `pred`'s order does less damage than the
signal-aligned `−z(rel)` re-rank, because the signal-aligned re-rank systematically pulls the
basket toward an ordering `pred` has already (correctly) ranked the other way.

This is the iter-018 failure mode (divergence-cut: great HL70 IC, died on EXT PnL), shown here at
its cleanest — the IC is genuinely real, PIT, orthogonal, AND transports across universes, and it
STILL doesn't monetize through this book.

## Insight for the next research cycle (STANDING RULE — record for the loop)

**Univariate cross-sectional IC — even strong, orthogonal, AND universe-transport-stable — does
NOT imply tradeable portfolio contribution on this book.** The production `pred` (a beta-neutral
XS mean-reversion basket) already absorbs the productive reversal; an overlay measured only by its
*standalone* IC re-ranks from a redundant/anti-correlated angle and destroys gross return.

**New standing pre-check (alongside PRE-CHECK-G4 and check-GROSS-PnL):** an alpha candidate must
pass a **conditional/marginal-contribution pre-check BEFORE build** — does it add gross PnL
*GIVEN* `pred`? Concretely:
1. IC of the signal on **`pred`-RESIDUALIZED forward returns** (regress forward alpha on `pred`,
   take the residual, then IC the candidate against that residual) — measures incremental skill,
   not skill `pred` already has. A near-zero residual IC predicts a PnL-layer reject even when the
   raw IC is large. (Orthogonality of the *signals* — corr(rel,pred)=−0.022 here — is NOT the same
   as orthogonality on the *forward-return residual*; this signal looked orthogonal on signals yet
   was redundant on outcomes.)
2. OR a tiny-weight blend check: does a small (e.g. 0.05–0.10) overlay weight LIFT gross PnL? If a
   minimal overlay already drops gross, a 0.5 ensemble cannot help.

This converts the IC→PnL gap from a 476s held-book build into a cheap pre-check, and would have
fail-fast-killed iter-022 before implementation.

## Artifacts
- script: `research/convexity_portable_2026-05-20/scripts/X131_xs_reversal_ensemble.py`
- per-cycle: `results/X131_percycle_{HL70,EXT,S44}.parquet` (base+ens PnL @ {1,3,4.5}bps, fold/regime tags)
- pre-check: `iter022_leadlag_transport_precheck.py`, `iter022_orth_fast.py`, `iter022_era_stability.py`
