# ORTHOGONAL-DATA LOOP — can genuinely-orthogonal (non-price/vol) data add gross alpha?

Self-paced research loop. Charter + iteration log. Harness: `orthogonal_harness.py`.
**Do NOT commit / update cross-session memory without user review.**

## Why this loop (the one remaining lever)
Five construction/signal cycles (see `docs/CONSTRUCTION_COST_IMPROVEMENTS_2026-07-29.md`) proved the price/vol
signal is exhausted and the edge is ~ONE factor: every attempt to get MORE gross from price/vol transforms
(features, regime/dispersion gating, vol-management, PPP) is null. The incumbent is at the frontier for THAT
information set. The ONLY remaining lever for more gross is **orthogonal information** — data NOT derived from
returns.

## Key enabling finding (2026-07-29) — overturns a stale memory note
Prior memory said positioning metrics are "recent-only -> underpowered." **FALSE now.** Binance Vision FUTURES
`metrics_*.parquet` (176 syms) have BOTH-ERA coverage: **165/176 syms >500 bars in each era, data back to 2021**
(46 syms from 2021, 30 from 2023, 45 from 2024). So positioning/OI/flow can be tested BOTH-ERA with real power —
this is a genuine, well-powered orthogonal information set, not a proposal. (Correct the memory once validated.)

## The orthogonal data (5-min cadence -> 4h PIT features, strictly backward merge_asof)
Source columns: `sum_open_interest[_value]`, `count_toptrader_long_short_ratio` (accounts),
`sum_toptrader_long_short_ratio` (positions = "smart money"), `count_long_short_ratio` (global accounts =
"retail"), `sum_taker_long_short_vol_ratio` (aggressive taker flow). **None is return-derived** => genuinely
orthogonal to the {low-vol, reversal} price factors.

Candidate features (initial): `oi_chg_1d/3d` (OI build/unwind), `tt_pos_ls` + `tt_pos_chg_1d` (smart-money
positioning level/change), `gl_acc_ls` (retail positioning), `smart_dumb = log(tt_pos)-log(gl_acc)` (smart-vs-
dumb spread — classic contrarian), `taker_ls` + `taker_chg_1d` (aggressive flow).

Hypotheses to adjudicate (let IC decide sign): retail-long = bearish (contrarian); smart_dumb high = bullish;
OI-up = positioning build (continuation or exhaustion); taker imbalance = short-horizon pressure.

## Protocol (each iteration)
1. Plan the test; state the prior + what would make it a real signal vs an artifact.
2. Run it (both eras, day-clustered / block bootstrap CI). Honest nulls recorded.
3. Review: is it orthogonal (survives residualizing vs vol+reversal)? incremental through the REAL pipeline?
   both-era stable? net of cost? Distinguish sign-flip (dead) from low-power (scale) from real.
4. Record verdict here; decide next. Skeptical prior: OI/positioning were null in OTHER contexts
   (alpha_v9_oi; pump/dump event-study) — but the rigorous BOTH-ERA INCREMENTAL-vs-{vol,reversal} test through
   the current per-symbol RidgeCV pipeline has NOT been done. That is this loop's job.

## Gates (a candidate must clear all to be "real")
- G1 SCREEN: orthogonalized IC (residual vs {rvol_7d, atr_pct, return_1d, ret_3d}) CI-excludes-0 BOTH eras.
- G2 PIPELINE: adds incremental rank-IC to V0 through gen() (paired day-clustered CI > 0) BOTH eras.
- G3 ROBUST: same sign both eras; not a vol/liquidity proxy; survives cost (turnover/net).
- Anything that dies at G1/G2, or sign-flips by era, is a null (record + move on).

## Iteration log
### iter 1 — coverage + triage screen (DONE)
Built `orthogonal_metrics_feats.parquet` (175 syms merged, all 8 feats present). G1 screen = orthogonalized IC
(residual vs {rvol_7d, atr_pct, return_1d, ret_3d}) vs forward alpha, both eras, day-clustered CI.
**2 of 8 pass G1 (orth IC CI-excludes-0, SAME sign, BOTH eras):**
- `oi_chg_1d` (OI 1d %chg): orth IC **+0.0073 [+.005,+.010] OOS / +0.0247 [+.020,+.030] RECENT** (+). NB raw IC
  was −0.019 → the negative raw was a vol/reversal proxy; the ORTHOGONAL part is positive.
- `tt_pos_chg_1d` (smart-money position 1d chg): orth IC **−0.0064 [−.010,−.004] / −0.0106 [−.015,−.006]** (−).
  Orthogonalization STRENGTHENED it (raw OOS was ns) => genuinely independent of vol/reversal.
FAILS: `oi_chg_3d` (OOS ns), `gl_acc_ls` & `smart_dumb` (sign-flip by era), `tt_pos_ls`/`taker_ls`/`taker_chg_1d`
(ns/dead). Pattern: CHANGE/dynamics features survive; LEVEL features are proxies or era-unstable.
Skeptic notes: both survivors RECENT-stronger (OOS small, +0.007) = same non-stationarity risk as everything;
G1 is only the raw-feature screen. Proceed to G2 (incremental through real pipeline).

### iter 2 — G2 incremental pipeline test (DONE): NULL
`orth_pipeline.py`. Baseline V0 reproduced (RECENT +0.0302 / OOS +0.0210 = gate passed). Adding each survivor:
- RECENT: +oi_chg_1d Δ +0.0001 [-.001,+.001]; +tt_pos_chg_1d +0.0005 [-.001,+.002]; +BOTH +0.0008 [-.001,+.003].
- OOS: all Δ negative, within noise (+oi −0.0004; +tt_pos −0.0005; +BOTH −0.0005).
**All CIs span 0 both eras => NULL through the per-symbol pipeline.** Same as OB/flow ("passes screen, dies
incrementally"). NUANCE: the pipeline is PER-SYMBOL RidgeCV (each symbol its own model), but G1's orth signal is
CROSS-SECTIONAL (rank names by positioning) — a per-symbol model cannot see a cross-sectional signal. So G2 null
= "not capturable as a per-symbol feature," NOT yet "no signal." Proceed to iter-3 (portfolio-diversification:
standalone cross-sectional factor + blend), which is the RIGHT vehicle for an orthogonal cross-sectional signal.

### iter 3 — portfolio-diversification test (DONE): 1st orthogonal both-era signal, but SUB-COST at retail
`orth_diversify.py`. Factors oriented ERA-LOCKED (sign from the OTHER era's mean; no look-ahead). Strategy Sh
RECENT +4.15 / OOS +0.92.
- `oi_chg_1d`: **NULL — non-stationary.** Era-locked orientation gives NEGATIVE Sharpe both eras (RECENT −3.86,
  OOS −0.61): the sign that works OOS is wrong for RECENT & vice versa. Blend hurts/none. Dead.
- `tt_pos_chg_1d`: **FIRST genuinely-orthogonal both-era signal found.** Oriented (stably negative IC both eras
  => sign −1 uniformly, no look-ahead): **gross Sh +2.89 RECENT / +1.96 OOS**, gross **+3.1 / +2.7 bps**, corr to
  strategy **+0.01 / −0.04** (orthogonal). GROSS diversification: **OOS blend ΔSh [+0.15,+2.16] CI>0 DIVERSIFIES**;
  RECENT ΔSh [−1.54,+2.94] no-help (strategy already Sh 4.15). **BUT SUB-COST at retail**: turn 0.33 × 24bps ≈ 8bps
  drag >> +2.7-3.1 gross => net@24 Sh −4.6/−3.8. Same wall as the Amihud premium: real orthogonal info, not
  harvestable at retail.
- Mechanism: `tt_pos_chg` = 1d change in TOP-TRADER POSITION long/short ratio ("smart money" repositioning). Sign
  negative => names where smart money INCREASED net-long subsequently UNDERPERFORM (crowding/exhaustion), and vice
  versa. Genuinely orthogonal to {low-vol, reversal}.

### iter 4 — net viability via turnover-control + fee-tier (DONE): sub-cost, no net diversification
`orth_iter4_net.py`. EWMA works as predicted (positioning is slow): λ=.85 cut factor turnover 0.33->0.14 and
lifted net Sh at every cost. Factor STANDALONE net Sh (RECENT/OOS): c24 −2.4/−1.7, c12 −0.5/−0.0, c6 +0.5/+0.8
=> breakeven ~12bps (fee-tier), positive at 6, sub-cost at retail 24. **NET BLEND vs strategy: NEVER diversifies**
— hurts at c24 (drags), TIE at c12/c6 (every ΔSh CI spans 0). Orthogonality (corr ~0) isn't enough: the factor's
NET Sharpe is ~0 at fee-tier, so blending adds only noise. The iter-3 gross diversification was a GROSS phenomenon;
cost eats the edge.
- **VERDICT (tt_pos_chg): real, both-era, genuinely-orthogonal (not a vol/reversal proxy) — the cleanest signal
  the program has found — but SUB-COST; no harvestable net edge, no net diversification to the strategy.** Same wall
  as the Amihud premium. Turnover-control (EWMA λ.85) is the only thing that even brings it near fee-tier breakeven
  standalone.

### iter 5 — broaden orthogonal candidate families (DONE): found a STRONGER, era-stable signal
`orth_iter5_screen.py`. G1 both-era survivors (orth IC CI-excludes-0, same sign):
- **oi_price_div = oi_chg_1d × sign(return_1d): orth IC +0.0143 [+.011,+.017] OOS / +0.0156 [+.010,+.021] RECENT.**
  STRONGEST orth signal found (~2× tt_pos_chg) and remarkably ERA-STABLE (+.0143 vs +.0156 — most signals are
  RECENT-lopsided; this isn't). Raw IC is negative (vol/reversal proxy) but the ORTHOGONAL part is strongly +.
  Mechanism: OI building INTO up-moves => continuation (higher fwd alpha); OI building into down-moves (new shorts)
  => underperformance. A positioning-confirmation signal.
- `oi_z` (OI vs own 30-bar hist): orth −0.0107 / −0.0060 (OI over-extension mean-reverts). `tt_pos_chg_3d`
  −0.0045 / −0.0074 (confirms 1d smart-money at 3d).
- Fails: gl_acc_chg (OOS ns), smart_dumb_chg (OOS ns), taker_chg_3d (OOS ns) — all OOS-insignificant.
Best candidate = oi_price_div (~2× gross of tt_pos_chg => real shot at surviving cost). Proceed to net viability.

### iter 6 — net viability of oi_price_div + oi_z (DONE): both sub-cost, no diversification
`orth_iter6_net.py`. corr to strategy ~0 both (orthogonal ✓, even oi_price_div despite using sign(return_1d)).
- **oi_price_div**: 2× IC advantage EVAPORATES at net — sign(return_1d) flips often => turnover 0.48 (50% higher
  than tt_pos_chg), so EWMA λ.85 (turn 0.15) net Sh: c24 −1.9/−1.6, c12 −0.3/−0.2, c6 +0.5/+0.5 = SAME ~fee-tier
  breakeven. Blend NEVER diversifies (hurt@24, tie@12/6). Higher gross bought nothing net.
- **oi_z**: worse — net-negative through c6 (OOS −0.40 even at 6bps); blend hurts/tie. NULL.
- Higher-IC did NOT beat cost because higher-IC positioning signals churn faster. Cost is the binding wall.

## SYNTHESIS / CONCLUSION (orthogonal-data loop, 6 iterations)
Tested the ONLY new information source with both-era power: free Binance futures positioning/OI/flow metrics
(genuinely orthogonal — not return-derived). 14 candidate features, 2 rounds, gates G1(screen)→G2(pipeline)→
G3(diversification)→net-viability.
- **REAL orthogonal both-era signal EXISTS** (the cleanest "real information" the whole program has found):
  tt_pos_chg_1d/3d (smart-money repositioning), oi_price_div (OI-price divergence), oi_z (OI over-extension) all
  pass the both-era orthogonalized screen (CI-excludes-0, same sign, NOT vol/reversal proxies). Mechanisms are
  economically sensible (crowding/exhaustion, positioning confirmation).
- **NONE is net-harvestable.** Per-symbol pipeline can't use cross-sectional signals (G2 null). As standalone
  cross-sectional factors: gross Sharpe +2–3, orthogonal (corr ~0), but gross is thin (~+3–5bps) and turnover
  high; EWMA turnover-control (positioning is slow) brings the best to ~FEE-TIER breakeven (~12bps) standalone,
  but SUB-COST at retail 24bps, and NO significant net diversification to the strategy at ANY cost (blend ΔSh CI
  spans 0 at best). Higher-IC candidates churn faster => cost eats the extra gross.
- **VERDICT: the cost wall is UNIVERSAL** — it binds on genuinely-orthogonal positioning data exactly as on
  price/vol. Free-data harvestable cross-sectional alpha is exhausted for a retail-cost taker. Real structure,
  no harvestable edge. Only paths to net-positive: (a) fee-tier/maker execution (≤~12bps) — then positioning
  factors become marginally viable standalone though still non-diversifying; (b) genuinely different data that
  is STRONGER or SLOWER (on-chain settlement flows, options-implied skew) — needs paid acquisition; the bar is
  now empirically set: must clear ~24bps retail OR operator at fee-tier.
- **MEMORY TO CORRECT (pending user review):** the note "positioning metrics recent-only → underpowered" is
  FALSE — metrics_*.parquet is both-era (165/176 syms >500 bars each, back to 2021). The honest finding is
  "positioning is real both-era orthogonal signal but sub-cost," not "underpowered."
- Scripts: orthogonal_harness / orth_pipeline / orth_diversify / orth_iter4_net / orth_iter5_screen / orth_iter6_net.

### iter 7 — squeeze-VETO overlay (borrow Barroso-principle done right; user idea A) — DONE: NULL on net
`orth_iter7_veto.py`. Use validated oi_price_div as a CONDITIONER on the short leg: veto the most squeeze-prone
shorts (highest oi_price_div = OI building into the pump). An overlay needs no standalone net-positive alpha — just
to remove bad shorts. Baseline (top-K=3 band) vs short-veto33/50 + symmetric; net@{24,12}, short-leg tail, block CI.
- **NULL on net**: every variant net@24 CI spans 0 (RECENT Δ[−3.4,+0.2]; OOS [−2.3,+0.7]); point estimates mostly
  WORSE (RECENT net@24 2.87→1.18–1.73; OOS −0.91→−1.27..−1.45). The veto INCREASES turnover (0.29→0.38–0.45 — going
  flat then re-entering) and cuts gross => cost > benefit.
- **Real but INSUFFICIENT tail reduction**: short-veto50 ~halves short-leg maxDD (RECENT 19→11, OOS 103→54) + improves
  skew — the squeeze-veto mechanism genuinely WORKS (removes squeeze drawdown) but not enough to beat the added
  turnover + lost gross. Same shape as vol-management (real tail-shape, no net Sharpe).
- **VERDICT: idea A = NULL for net Sharpe.** Even the "borrow done right" (overlay, no standalone-alpha requirement)
  hits the cost wall: the overlay itself adds turnover and the thin orthogonal signal (~+0.015 IC) can't clear it.
  Adopt ONLY if the goal is short-leg drawdown reduction at low cost, not Sharpe. Confirms the universal-cost-wall
  from the overlay side too. This closes the "borrow-for-alpha" thread — the constraint is cost, full stop.
- Untested-but-low-prior (would need a reason to pursue): positioning COMPOSITE (correlated, both sub-cost →
  unlikely), true daily/weekly horizon (EWMA λ.85 already ~proxies low turnover and still sub-cost).
