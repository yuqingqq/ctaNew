# Convexity v1 — FROZEN production reference (2026-06-04)

The locked v1 strategy after the 2026-06-03/04 audit + validation. This is the **forward-test
baseline and future-comparison reference**. Do not change in place — fork v2 for changes.

## Strategy
**Book-B-only** (the high-vol "book A" is DROPPED — it is alpha-barren: per-leg A-long −0.19, A-short
mostly BTC beta; no fix generalized).

- **Universe:** rank eligible symbols by **trailing-30d `rvol_7d`** (PIT), **exclude the top ~46%**
  (≈ top-80 at today's ~166-sym universe; cutoff sweep peaked N≈70–80), trade the calmest ~54%
  (~85–96 names). Use the **percentile form** (universe-size-invariant; portable across listings/
  delistings) with a count **floor/cap ~50–120** for K=3 breadth + capacity.
- **Eligibility:** maturity ≥180d, trailing-30d $-volume ≥ liquidity floor, hygiene excludes — all PIT.
- **Construction:** K=3 long / K=3 short, **beta-neutral**, **24h hold via 6 overlapping sleeves**,
  regime gate (BTC 30d ±10%, N=3 hysteresis), **relative-rank** selection.
- **resid_rev overlay (ON):** dual-pred — the **long** leg is ranked by a per-symbol Ridge that adds the
  BTC-residual short-horizon reversal feature (`resid_rev_2/3` = −Σ past 8h/12h residual alpha); the
  **short** leg stays on the base model. (Applying resid_rev globally corrupts the shorts — must be
  long-ranker only.) **EXECUTION-LATENCY-GATED:** capture within ~10–15 min of the 4h close.
- **Cost basis:** 4.5 bps/leg (conservative — realized low-vol effective spread mean ~2.4 bps RT).

## Honest expected performance (NOT a point estimate)
- **Baseline (no resid_rev):** Sharpe ~**+2.0** (nested-OOS cutoff) to **+2.55** (static N=80), monthly;
  ~+2.82 daily. maxDD ~−2400 bps.
- **+ resid_rev (prompt execution):** ~**+3.4–3.6** monthly/daily (lift +0.8–0.9, cost-robust to 13 bps).
  Degrades toward baseline if execution lags (full-bar-late ≈ +2.9 monthly / ~baseline daily).
- **Plan around a RANGE, not a number** — universe-composition variance is high (see Known Risks).

## Validation scorecard
| check | result |
|---|---|
| Leakage audit (purge/embargo, PIT dvol/betas/features) | ✅ clean; survives 1-bar exec delay |
| Symbol-rule form (rank vs abs vs multifac) | ✅ relative-rank (abs drifts, multifac kills dispersion) |
| Hold horizon | ✅ 24h optimal (cost amortization) |
| Drop-book-A stress | ✅ no catastrophic tail (~13% deeper DD, net +0.42 Sharpe) |
| resid_rev genuine (not bid-ask bounce) | ✅ 4h autocorr +0.012, Roll spread 0; resid-alpha lag-1 −0.05 |
| resid_rev latency budget | ✅ ~10–15 min (5m-resolution) |
| resid_rev cost-robustness | ✅ lift +0.81 even at 13 bps/leg |
| Realistic spreads (#175) | ✅ low-vol mean ~2.4 bps RT << 9 bps assumed |
| **Universe overfit (placebo #173)** | ⚠️ **p83 (FAIL p90)** — high composition variance |
| Cutoff nested-OOS (#174) | ✅ +2.01 generalizes (vs static +2.74) |

## Known risks (carry into forward test)
1. **Universe-composition variance (primary):** rvol-selection beats random *mean* (+0.88, 10/12) but
   only ranks p83; random ~94-subsets span −1.0 to +2.1. The Sharpe depends meaningfully on *which*
   symbols are in the book → **forward expectations must be wide**, edge can decay as the universe drifts.
2. **resid_rev is latency-gated:** the +0.8–0.9 lift requires fills within ~15 min of the 4h close; slow
   execution erases most of it. Live latency must be measured.
3. **Live `--cycle` path not yet wired** — real-time feature/pred computation on the execution server is TODO.

## How to run (paper / forward test)
```bash
# wired strategy (book-B-only + resid_rev, default):
USE_RESIDREV=1 bash live/run_bookB_residrev.sh
# baseline reference (resid_rev off):
USE_RESIDREV=0 bash live/run_bookB_residrev.sh
# regenerate preds from current panel first (each forward cycle):
REGEN=1 USE_RESIDREV=1 bash live/run_bookB_residrev.sh
```
Runner: `live/run_bookB_residrev.sh` · preds gen: `live/gen_residrev_wf_preds.py` · ledger:
`live/state/opt_loop/insights.md`.

## Open optimization (v2 candidates — do NOT alter v1)
**★ LEAD v2 CANDIDATE (2026-06-05, mechanism re-audit) — equal-weight + regime-aware stop:**
Two structural fixes vs the side/bull-tuned production: (1) **equal-weight sizing** (drop beta-neutral a/b reweighting —
noise on near-matched leg betas), (2) **DD-stop OFF in bear** (`STOP_SKIP_REGIMES=bear` — the equity-DD stop is
pro-cyclical against mean-reversion: de-grosses into the bounce; engaged 78% in bear). Root weakness was flat-in-bear
forgoing a real edge; true cause was the stop, not the regime. **Assembled: gross +3.82 / NET-OF-FUNDING +3.33 Sharpe (+0.41 vs production's +2.92 net)**; passes per-fold 7/9,
bootstrap P=0.96, robust to spread 13.5 bps/leg AND funding (bear edge +2,721 gross → +2,195 net; funding is a modest
symmetric cost ~−1.4 bps/cyc paid by both configs — the 3.82→3.33 move is purely funding). **CAVEAT — Sharpe win, not
maxDD win:** maxDD doubles (−4,135, bear = fat-tail drawdowns); at matched maxDD, levering production ×1.85 (+15,091)
≈ v2 (+13,662) → v2's +68% PnL is mostly leverage; the real un-leverable edge is **+0.41 Sharpe** (pays off only if
vol/Sharpe is the binding constraint, not maxDD). Env-gated (`SIDE_BETA_NEUT`,`STOP_SKIP_REGIMES`,`BEAR_MODE`); production
byte-unchanged. Decisive arbiter: forward bear test (live now). Full scorecard: `docs/convexity_mechanism_audit.md`.

**v2 composition-variance levers — TESTED, mostly REJECTED (2026-06-04):**
- **#181 cutoff ensemble** — ADOPT for v2 cutoff form (parameter-free blend N=55/70/85 = +2.58 nested-OOS,
  beats single-N +2.01; fixes cutoff non-generalization, not composition variance).
- **#182 breadth/higher-K** — REJECTED. K=3 +3.46 > K4 +2.73 > K5 +2.51 > K6 +2.00. Alpha concentrated in
  top-3 ranks; breadth dilutes alpha faster than composition risk.
- **#180 rank stability** — REJECTED (both halves). Window-smoothing −0.2 Sharpe; band-hysteresis −1.25
  Sharpe at matched book size for 22% churn cut. Monthly re-rank churn is *productive* (rvol non-stationary).
- **Maturity gate** (user hypothesis: redundant with rvol-exclude) — KEEP 180d. gate-sweep 180/+3.46,
  90/+2.65, 0/+3.10 (non-monotone = composition noise); no evidence dropping helps.

**Conclusion:** composition variance (placebo p83) is **irreducible via construction levers** on free data —
it's cross-sectional (which names), not temporal (churn) or breadth (K). Honest mitigation = forward test
with wide expectations + kill-switch.
- **#185 cross-exchange spread FEATURE** (Coinbase/OKX-vs-Binance premium) — **REJECTED**. Real univariate XS IC
  (okx_level −0.04, t−8) but: 36% redundant with resid_rev (corr −0.55), hurts as a model feature on every leg
  (−0.26 to −0.56), tilt form craters. Orthogonal remainder (residual IC −0.021) too small to lift Sharpe. The 4h
  residual-alpha extraction is at ceiling even with external orthogonal data. "Free orthogonal signal" question answered.

**v2 verdict: no construction/feature lever beats book-B + resid_rev + K=3 (+3.46).** Composition variance is
irreducible on free data. The decisive next step is the forward paper-test, not another backtest.
- **#178 forward paper-test** — the decisive arbiter.

## UPDATE 2026-06-04 — deploy reconciliation (corrections to above)
- **Funding RESTORED (full-V0).** The earlier V0_LEAN (funding-dropped) was a two-book result; on book-B-only funding
  HELPS (+0.68 baseline / +0.95 with resid_rev, walk-forward). Deploy = full V0.
- **Single book.** Flow model / book A fully dropped. No two-book combine.
- **Artifacts RENAMED** (no "twobook"): `convexity_v1_short_model.pkl` (V0 base → shorts),
  `convexity_v1_long_model.pkl` (V0+resid_rev → longs), `convexity_v1_universe.json` (exclude top-80 high-vol).
- **Deploy cut = fit_cut 2026-05-29** (latest − 1d embargo). The "5.26" that appeared earlier was only the
  walk-forward backtest's last-fold training cut, NOT a deploy model.
- **Reproduction:** model@5.29 + split@5.29 is DETERMINISTIC (run1==run2 post-cutoff). Live box reproduces exactly.
  Golden (`docs/golden_cycles_v1.json`) regenerated from this aligned config.
- **Honest Sharpe = walk-forward +3.46** (monthly-rerank, no look-ahead). The frozen full-OOS replay (+9.49) is
  LOOK-AHEAD (frozen model on its own training data) — discard; only its post-cutoff tail is valid.
- **Runner:** `live/run_convexity_v1.sh` (single book). `run_convexity_daily.sh` still has legacy two-book logic →
  needs single-book rewrite on the exec server (#179).
