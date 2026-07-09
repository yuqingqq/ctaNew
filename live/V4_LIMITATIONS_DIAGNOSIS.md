# v4 production-config limitations diagnosis (2026-07-09)

Config-aware diagnosis of what production v4 ACTUALLY trades (not the vanilla book the variant
cells used). Purpose: locate the real limitations before optimizing. Grounded in `run_convexity_v4_live.sh`
+ this session's validated findings + a per-regime raw-book attribution.

## Production v4 = a regime-SWITCHED strategy

Per `run_convexity_v4_live.sh`, the model preds feed a regime switch (btc_ret_30d):

| regime (btc30) | what production does | share (rec/oos) |
|---|---|---|
| **side** (−0.10..+0.10) | model 1L/2S, REGIME_GATE, inv_sqrt_vol | 54% / 55% |
| **bear** (< −0.10) | equal-weight 1L/2S (BEAR_MODE=equal); **DD-stop OFF** (STOP_SKIP_REGIMES=bear) | 36% / 10% |
| **bull** (0.10..0.15) | **FLAT — gated** (BULL_GROSS_MULT=0) | 7% / 12% |
| **deep bull** (≥ 0.15) | **mom1d long-only** (long top-2 by return_1d, ½ gross) | 3% / 23% |

Plus GLOBAL_GROSS_MULT=0.5 (live cap), kill-switch.

## Per-regime raw-book attribution (net, book 0.5/0.5, 9-bps RT convention, daily-Sharpe)

RAW model-book edge by production regime label (PRE-gate — REGIME_GATE/DD-stop would cut the
losing cycles, so actual production per-regime net is ≥ these; small-sample flags noted):

| regime | REC net / Sharpe | OOS net / Sharpe | production action |
|---|---|---|---|
| side | **+18.7 / +4.09** | **−0.1 / −0.04** | traded |
| bear | −4.9 / −0.88 | **+17.3 / +4.46** | traded |
| bull (mild) | +45.2 / +4.36 (n=114, short-driven, era-fit) | −7.0 / −1.47 | GATED |
| deep bull | −32.3 / −4.55 (n=47) | −0.9 / −0.22 | mom1d long-only |

## The core limitations (what this reveals)

1. **NO regime has a consistent both-era edge — every edge is era-dependent.** Side carries
   RECENT (+4.09) but is FLAT OOS (−0.04). Bear is the anchor OOS (+4.46) but a DRAG recent
   (−0.88). Mild-bull would've earned recent (+4.36) but loses OOS (−1.47). The strategy is a
   *collection of era-specific regime edges*, and the config (gate bull, trade bear) is implicitly
   a bet on which era you're in. This is the deepest limitation and the mechanism behind the 2022
   holdout FAIL + the 0.5× cap.
2. **The bull gate is too blunt.** BULL_GROSS_MULT=0 kills ALL bull (btc30>0.10) uniformly — but
   recent mild-bull (0.10-0.15) would have been strongly positive (short-driven "pump topping
   reverts"), while deep-bull and OOS-bull are destructive. The uniform gate gives up productive
   mild-bull to avoid destructive deep-bull. (Caveat: recent mild-bull is n=114, short-driven,
   likely era-fit — the gate is the conservative choice.)
3. **The deep-bull mom1d patch does not earn** (−32 rec small-n / −0.9 OOS). The return_1d
   long-only overlay is ~flat-to-negative; it provides long-alt exposure but no validated edge
   (Q3: ranking unproven OOS, p=0.215).
4. **Bear is traded unconditionally** (BEAR_MODE=equal, DD-stop OFF in bear) despite the recent
   drag — a bet that bear reverts to its OOS-anchor behavior. The squeeze tail (short median +42
   but mean tail-gutted, skew −2.45) is the risk, and it is unhedged (SQ1 crowding signal exists
   but is too weak/non-stationary on free data — SK1 failed the recent holdout).
5. **The workhorse (side) alpha is thin and event-concentrated** — 76% of side net from 2
   dispersion months, 5/9 months positive, flat OOS. No config fixes this; it is a signal/data
   limit.

## Optimization map (limitation → lever, with honest priors)

| limitation | candidate lever | prior / status |
|---|---|---|
| era-dependent regime edges | finer/adaptive regime detection | LOW — adaptive-timing failed 7× (dynamic K, gates, rvol-scaling); "trailing estimators lag era" |
| blunt bull gate | split gate: trade mild-bull (0.10-0.15), gate deep-bull (≥0.15) | MED but era-fit risk — recent mild-bull is the temptation; needs dual-era + the 2022 caveat |
| deep-bull mom1d weak | drop it, or replace with a validated deep-bull rule | worth a pre-registered test (it's a config, cheaply removable) |
| bear unconditional / squeeze tail | CUSUM gross-throttle (regime-level); liquidation data (SQ1) | CUSUM wired-not-lever; liquidation = the paid-data route |
| thin event-concentrated side alpha | dispersion timing (FAILED); new data | dispersion timing dead; execution/cost is the real lever |
| era-fragility (2022 FAIL, 0.5× cap) | forward ledger (release cap); universe-portable build | operational; the binding path forward |

## The two config choices most worth re-examining (NEW, actionable)

Unlike the model/feature axis (exhausted, 0 promotions), these are CONFIG choices with live edge
questions the config-aware attribution surfaced:
1. **The bull gate granularity** (uniform-off vs split mild/deep) — testable, but the recent
   mild-bull edge is small-sample + era-fit, so a dual-era pre-registered cell with the 2022
   caveat is mandatory; honest prior it's an era bet.
2. **The deep-bull mom1d overlay** — it doesn't earn; a pre-registered test of {mom1d vs flat vs
   alt-rule} in deep bull could simplify the stack at no cost, or find a better deep-bull handling.

Both are OVERLAY/config tests (not model changes), so they need the full-stack replay discipline
(faithful cost, no path-coupled variant noise — the estimator law), NOT the vanilla-book cell
machinery. Everything else points back to the standing levers: forward ledger, execution/cost,
paid positioning-depth data.

## Caveats on the numbers

RAW-book pre-gate attribution (actual post-REGIME_GATE/DD-stop production per-regime net is ≥
these). Recent bull (n=114) and deep-bull (n=47) are small-sample. inv_sqrt_vol sizing and the
gates are NOT applied (this is the raw regime EDGE, the right lens for "where's the signal," not
production PnL). Cost = 9-bps RT per §8 convention; Sharpe daily-aggregated then ×√365.
