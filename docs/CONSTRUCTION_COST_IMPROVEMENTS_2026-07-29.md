# Construction & Cost Improvements — 2026-07-29 (tracking)

> **⚠️ CURRENT CONCLUSION IS `docs/CONCLUSION_2026-08-03.md`** (supersedes the "cost-gated at retail" bottom-line
> in the Executive summary below, which predates the big-names / PIT-review / net-of-cost findings). The doc below
> is the running tracking log; read the consolidated conclusion for the up-to-date verdict.

Literature→proposal→test→validate flow on the crypto per-symbol-Ridge alpha-residual strategy. Signal is
exhausted (see SIGNAL_LATENT_MAP); these are CONSTRUCTION/COST/RISK-side improvements that raise the NET /
robustness of the thin edge. All scripts uncommitted; numbers for review. Eras: OOS<2025-10-01≤RECENT;
RECENT is a shorter, non-stationary, concentrated era — treat OOS as the durable/forward number.

## Executive summary (5-cycle literature→test→validate flow)

**Deployable stack (all validated, cost/risk/robustness-side):**
`per-symbol Ridge → beta-hedge (era-locked) → liquidity filter (top-50% ADV) → turnover-control (band and/or EWMA λ≈0.7)`

| lever | effect (durable OOS unless noted) | status |
|---|---|---|
| no-trade band (K+M hysteresis) | halves turnover; break-even ~2× (6→12 bps) | validated |
| beta-neutralization (era-locked) | era gap 1.11→0.28 bps; OOS Sharpe +1.84→+2.11 | validated |
| liquidity filter (top-50% ADV) | cost ~24→15 bps; OOS net Sharpe −0.9→−0.31 | validated |
| **EWMA weight-smoothing (λ≈0.7)** | **~half turnover at equal net; stacks with hedge** | validated (turnover win, Sharpe tie) |

**Four SOTA attempts at MORE gross alpha — all null:** dispersion/regime gating, volatility management
(timing+weighting), parametric portfolio policies. Root cause re-confirmed independently: the edge is **~one
effective factor** (PPP's covariance machinery was worthless; feature ablation says the same).

**Bottom line:** gross edge ≈ **+3.3 bps/bar, ~one factor, era-unstable, cost-gated at retail**. The signal is
**vol-PRIMARY (low-vol) + secondary short-reversal** (loads ~2× on low-vol; `build_signal_decomp.py`, and it
EARNS via per-symbol-Ridge calibration — crude cross-sectional factor books LOSE OOS — so it is NOT a naive
factor bet; see SIGNAL_LATENT_MAP addendum 2026-07-30). Every robustness/net lever has been found and banked;
every lever for *more gross* on free data is exhausted. The incumbent construction is at the practical frontier
for this information set. **The only remaining lever for more gross is ORTHOGONAL DATA (on-chain / options-implied
/ positioning-at-scale) — needs acquisition, can't be tested on local data (funding/OI/L2 already null).**

## CORRECTION 2026-07-30 — the cost wall is NOT universal: big-names-only is NET-POSITIVE at retail (OOS)
`build_liquidity_tiers.py`. User point: 24bps is long-tail slippage; trade big names only → low cost IF signal
survives. Sweep top-N by ADV, both eras. **The signal SURVIVES (even strengthens) in big names, and cost collapses
→ net-positive at retail in OOS:**
- **OOS (durable):** top15 rankIC **+0.027** (STRONGER than full-universe +0.021!), gross +4.9bps, cost ~7 →
  **net Sharpe +0.56 @model / +0.48 @8bps / +0.86 @5bps**. top25 net +0.10/+0.16/+0.57; top40 +0.05/+0.39/+0.92.
  Signal does NOT degrade in majors (rankIC +0.020–0.028 across tiers) — the low-vol+reversal edge is CLEANER in
  liquid names, and low cost there → **net-POSITIVE at retail without needing fee-tier.**
- **RECENT (shorter/noisier):** marginal — top15 net +0.15 @model / +0.03 @8; mid-tiers net-negative; top25 gross
  anomalously −4.6 (small-n noise). So RECENT big-name net is marginal-to-mixed, not clean.
- **⇒ The earlier "cost-gated at retail" conclusion was an ARTIFACT of trading the full long-tail universe.**
  Restricting to ~top15–40 by ADV keeps the (durable-OOS) edge at ~7–10bps cost → net-positive. This is the
  deployable configuration; EWMA turnover-control + beta-hedge (both validated, not yet stacked here) would push it
  further. Caveats: small concentrated book (12–15 names, higher idio risk, noisier rankIC); RECENT only marginal;
  turnover still ~0.4 here (EWMA would help). Net: **the strategy IS net-positive at retail on big names in OOS.**

## Beta decomposition of the top-40 book (`build_top40_beta.py`, 2026-07-30) — NOT beta-driven; RECENT alpha is weak
Answering "is it beta-driven?": NO. Net beta is small (−0.073 OOS / −0.160 RECENT) and the return is ~all alpha:
- **OOS:** L/S +4.22bps = ALPHA (beta-neutral) **+4.49** (107% of return) + beta contribution **−0.28** (the small
  short-beta was a DRAG). Definitively beta-NEUTRAL alpha; beta is a nuisance to hedge, not the source.
- **RECENT (honest wrinkle):** L/S −1.32bps = ALPHA **−2.10** (NEGATIVE) + beta **+0.78** (short-beta windfall in a
  down-drifting equal-wt tape). So the top-40 beta-neutral alpha is NEGATIVE in RECENT — the 2025-26 froth-era edge
  lived in the LONG-TAIL small-caps we dropped, not the majors. Beta small in both eras (not driving).
- **⇒ Two things:** (1) the book is beta-NEUTRAL (net beta ~−0.1, return is 100%+ alpha) — hedging beta removes a
  small drag, it is NOT a market bet. (2) BUT the top-40 ALPHA is OOS-durable / RECENT-weak-to-negative — so the
  earlier "big-names net-positive at retail" is an OOS phenomenon; in the most recent era the majors did NOT carry
  it. Big-names fixes COST but exposes that the RECENT edge was tail-concentrated. Material caveat for forward use.

## VALIDATION SCORECARD (`build_validate_conclusion.py`, 2026-07-30, day-clustered CIs, robust across N=20/30/40/50)
- **VALIDATED — beta-neutral:** net beta −0.07..−0.20 across N; return is ~all alpha. Not a market bet.
- **VALIDATED — mechanism = structural low-vol:** BETWEEN-rvol IC **−0.041 OOS [−.048,−.036] / −0.031 RECENT
  [−.045,−.016]**, CI excludes 0 BOTH eras. The carrier is real & significant in both eras (the RECENT *book*
  weakness comes from the model mixing in the within-vol component, which flipped + in RECENT — NOT the core
  premium failing).
- **VALIDATED — OOS GROSS beta-neutral alpha is significant:** +4.6..+5.2 bps, Sharpe **+1.6..+2.0, CI excludes 0**,
  robust across N=20–50. Real edge.
- **MARGINAL — NET at retail:** the significant gross alpha is eaten by cost (~4 bps drag at top-40) → net ~+0.8 bps,
  CI SPANS 0. So "net-positive at retail" is a positive point estimate, NOT statistically significant.
- **CORRECTION (I overstated) — "RECENT alpha negative":** actually **NOT significant** — RECENT alpha −1.7..−4.5 bps
  but CI SPANS 0 at every N (e.g. N=40 [−6.8,+3.2]). So RECENT is *uninformative/noisy* for the majors (negative
  lean, wide CI), NOT reliably negative. Do not claim the top-40 fails in RECENT — claim it's unproven either way.
- **EXPLORATORY (not CI-validated) — regime (quiet vs frothy) claims:** point estimates on sub-splits; suggestive,
  not validated; regime-timing is itself non-stationary. Do not hard-gate on it.

## REVIEW: universe look-ahead check (`build_review_pit.py`, 2026-07-30) — the OOS claim SURVIVES
Concern: all "big-names" tests picked top-N by FULL-SAMPLE ADV (forward-looking — names that BECAME big). Honest
PIT re-test ranks by TRAILING-30d ADV per bar (in-era beta, day-CI):
- **OOS SURVIVES and is if anything STRONGER:** N=20 alpha +5.68 [+2.37,+9.10] Sh +2.22 [+0.88,+3.61]; N=40 +5.20
  [+2.37,+7.86] Sh **+2.49 [+1.14,+3.86]** — CI excludes 0. So the universe look-ahead did NOT drive it; the PIT
  (trailing-ADV) majors carry the alpha cleanly. **The OOS gross-alpha claim is robust.**
- **RECENT under PIT:** +2.99 [−2.46,+9.06] (N=20) / +0.42 [−4.57,+5.74] (N=40) — spans 0, mild POSITIVE lean →
  reconfirms RECENT is insignificant/noisy, NOT negative (the earlier "RECENT negative" was noise; PIT leans +).
- **Remaining un-fixable caveat:** delisted-name SURVIVORSHIP (dead names absent from the panel/flow cache entirely)
  — cannot correct on this data; majors are less exposed (rarely delist) but it's a real limit on absolute levels.
- NET: the review UPHELD the load-bearing claim (OOS significant gross beta-neutral alpha, robust to PIT universe,
  in-era beta, N, and day-CI); net-at-retail stays marginal; RECENT stays insignificant.

## Baseline (what we farm, net)
- Gross cross-sectional alpha ≈ **+3.3 bps/bar** (beta-hedged, ~identical both eras); the recent extra is a
  short-beta windfall (non-stationary — don't bank). GROSS, pre-cost. `build_alpha_beta_decomp.py`.
- Net (quintile L/S, 4h rebalance, turnover ~0.40): **negative at retail (24 bps), break-even ~7-10 bps**,
  net-positive only at fee-tier/maker. `build_net_edge.py`.

## Improvement 1 — no-trade band (cost-aware construction)
Literature: Implementable Efficient Frontier (afajof 32368); band-turnover regularization (FR-LUX arXiv:2510.02986).
`build_notrade_band.py`, `build_deployed_band.py`. Enter top-K, hold until exit top-(K+M).
- Halves turnover (0.40→~0.20-0.28) with modest gross loss → **break-even ~doubles**: deployed top-K=3, OOS
  6.1→12.1 bps; RECENT 10→15-20. Net Sharpe@12bps flips −0.72→~0. VALIDATED, deployable, works on top of the
  existing hysteresis. (Tune band width on OOS; don't cherry-pick per era.)

## Improvement 2 — beta-neutralization (risk)
Found: the "BTC-residual" target retains ~+0.25 market beta (higher for high-vol names) → book carries a hidden
non-stationary short-beta (~−0.1). `build_why_beta.py`, `build_beta_neutral.py`.
- Era-locked hedge: era gap in mean return **1.11→0.28 bps** (much more stable); OOS mean +2.84→+3.20,
  Sharpe **+1.84→+2.11**; RECENT gives up its short-beta windfall (correct — it was luck). VALIDATED.

## Combined (band + beta-hedge) — they STACK
`build_combined.py`. Deployed top-K=3: OOS break-even 6.1→**13.4** bps; net Sharpe@6bps +0.01→**+0.56**,
@12bps −0.72→**+0.11** (flips positive), @3bps +0.38→+0.79. Meaningfully better *where it counts* (durable
net at achievable cost) but still cost-gated (net-neg at retail); gains are robustness/break-even, not gross.

## Improvement 3 — per-symbol (liquidity-aware) cost + liquidity filter
`build_net_cost.py`. Cost model cost_RT_i = 6+36·(1−ADV_pct) (liquid ~6-10, illiquid ~38-42, mean 24).
- Full-universe book trades MEDIAN-liquidity names (avg cost ~24) → per-symbol ≈ flat-24; short leg only
  MODESTLY pricier than long (OOS 26 vs 21) — high-vol maps only weakly to illiquidity.
- **Liquidity filter (top-50% ADV): cost 24→~15 bps → OOS net Sharpe −0.9→−0.31 (+0.6), RECENT +0.03→+0.48.**
  Cheaper trading beats the lost gross AND removes illiquid-name tail risk. VALIDATED (actionable).
- Caveat: cost model is a CALIBRATED ASSUMPTION; real per-symbol costs need actual spread/depth (the
  documented "L2 for cost realism" use). Direction robust; exact bps needs real execution data.

## Overall state
- Signal exhausted (confirmed by SOTA: "model complexity limited in crypto, OLS≈NN"). Wins are construction/
  cost/risk-side.
- Durable (OOS) net at realistic liquid-name cost (~15 bps) with band+hedge+filter ≈ **breakeven-to-slightly-
  negative (Sharpe ~−0.3)**; positive only RECENT (non-stationary). The FULL STACK (liquidity filter + band +
  beta-hedge + FEE-TIER getting RT<~13 bps) is the realistic path to a net-positive durable book.
- Remaining lever for MORE gross: ORTHOGONAL DATA (on-chain/positioning/options) — needs acquisition.

## Scripts
build_alpha_beta_decomp / build_net_edge / build_notrade_band / build_deployed_band / build_topk_robust /
build_beta_neutral / build_combined / build_net_cost (+ build_latent_map, build_ridge_map, build_universal_test,
build_why_* for the signal-structure map).

## Research cycle 2 — dispersion / regime gating: NULL (does not transfer to crypto)
Literature: factor-timing via cross-sectional dispersion (equities: high disp = more opportunity; crypto
momentum breaks down in high disp, SSRN 6648082). Proposal: gate/size our edge by cross-sectional dispersion
(the mechanism that makes xyz work). `build_dispersion_gate.py`.
- RESULT: model rank-IC by era-locked dispersion tercile is ~FLAT — OOS 0.019/0.025/0.020 (LOW/MID/HIGH),
  REC 0.023/0.030/0.033; corr(disp, IC) = +0.01 OOS / +0.02 REC (≈0). OOS flat/hump, REC mildly rising =
  NOT both-era-stable.
- VERDICT: **NULL — the crypto edge is dispersion-INDEPENDENT; a dispersion gate does not improve it** (unlike
  xyz, where it's the key lever). 2nd regime-null (with the earlier vol-regime null) ⇒ robust: the crypto edge
  works ~uniformly across regimes and cannot be gated/timed for improvement. Don't add a dispersion/regime gate.

## Research cycle 3 — volatility management (timing + weighting): NULL on Sharpe (crash-shape only)
Literature: Barroso & Santa-Clara 2015 "Momentum Has Its Moments" (scale L/S by inverse trailing realized
variance → "kills crashes, ~doubles Sharpe"); Moreira-Muir 2017 JF. Critique honored: Cederburg 2020 /
Barroso-Detzel 2021 (gains not OOS-achievable, cost erodes); DeMiguel et al. 2024 JF (works OOS only if you
time some factors, hold others constant). `build_vol_managed.py`, `build_vol_weight.py`.

- **3a vol-TIMING** (scale book by median(σ)/σ_{t-1}, strictly PIT, cap 3×): **NULL on Sharpe both eras, both
  books** — quintile ΔSh CI OOS [−0.28,+0.45] / REC [−0.56,+0.67]; deployed ΔSh CI OOS [−0.18,+0.51] /
  REC [−0.27,+0.67]. Point estimate leans slightly + on the deployed book (+0.16..+0.32 across windows) but
  not significant. REAL but MODEST crash-shape gain: kurtosis drops all 4 cells (deployed OOS 6.0→5.2,
  maxDD 58.9→52.8, skew −0.29→−0.07). Extra turnover from re-levering negligible (~0.01-0.03), so unlike the
  equity critique COST isn't the killer — the Sharpe edge just isn't there. Mechanism: our low-vol-led
  book is already near-symmetric (skew −0.3..−0.5) and band+beta-hedged, so there's little crash structure to
  time (vs long-momentum's −2..−4 skew). Optional: adopt the 240-bar scaling purely for tail-shape at ~zero
  cost; NOT a Sharpe/alpha win.
- **3b vol-WEIGHTING** (inverse-rvol_7d within each leg vs equal-weight): **NULL and ERA-UNSTABLE** — RECENT
  ΔSh −0.73 (INV-VOL) / −0.45 (winsor), OOS +0.23 / +0.03; every CI spans 0, signs flip by era. Down-weighting
  high-vol names cuts the short-high-vol exposure = RECENT windfall (hurts) but marginal OOS help. Equal-weight
  top-K stands.
- VERDICT: **the volatility-management family does NOT improve our book** beyond band+beta-hedge. 3rd construction
  overlay that works for equity factors but does NOT transfer to our crypto low-vol-led (reversal-secondary) book (with regime-
  and dispersion-gating). Pattern is now robust: overlays that exploit crash-skew / regime-dependence / vol-
  structure find little to bite on here; the validated wins (band, beta-hedge, liquidity filter) are cost/risk-
  realism, not alpha. Reinforces: remaining lever for MORE is ORTHOGONAL DATA, not more construction.
- Scripts: build_vol_managed / build_vol_weight.

## Research cycle 4 — parametric portfolio policies (PPP): NULL (does not beat predict-then-sort)
Literature: Brandt, Santa-Clara & Valkanov 2009 RFS. Different paradigm: skip return-prediction; set weights
directly w_i=(1/N)theta'z_i and choose theta to max book utility. MV utility -> book return = theta'f_t
(f = characteristic-factor returns) -> closed form theta*=Sigma_f^{-1}mu_f, estimated WALK-FORWARD per cut-window
(expanding, 1d embargo). Same 14 V0_LEAN chars & eval windows as incumbent (per-symbol Ridge->topK=3 band->
era-locked beta-hedge). `build_ppp.py`. Variants: MV vs mean(theta=mu); equal-weight vs td60 (HL60d time-decay
matching incumbent). Paired block-bootstrap CI on gross-Sharpe diff, both eras.
- **No improvement**: with matched discipline (td60) EVERY era comparison is a TIE (CI spans 0). RECENT incumbent
  point est is higher (+4.05 vs PPP-mean-td60 +1.78) but that's the NON-BANKABLE concentration/survivor windfall
  (+22 bps); on the durable OOS metric they're indistinguishable (incumbent grossSh +1.02, PPP-mean +1.75 equal /
  +0.66 td, all CIs span 0). **Reject PPP.**
- **Covariance term adds NOTHING**: PPP-mean(theta=mu) >= PPP-MV(Sigma^{-1}mu) in EVERY cell (RECENT-td +1.78 vs
  +1.00; OOS-equal +1.75 vs +1.62; OOS-td +0.66 vs +0.61). Optimizing a 14x14 factor covariance is noise-fitting
  when there is ~ONE effective factor. **Fresh independent confirmation of "edge = one redundant factor"** from the
  construction side (not feature ablation).
- **Fairness/methodology**: equal-weight PPP was DESTROYED in RECENT (grossSh -1.26, negative) because non-adaptive
  theta can't handle the OOS->RECENT non-stationarity; td60 rescued it to a tie (+1.78). Confirms HL=60 adaptivity
  is essential for any pooled-parameter method here (do NOT conclude from the non-time-decayed run).
- **Texture (NOT a validated win)**: equal-weight PPP-mean had the best OOS net profile in the table (turnover 0.18
  vs incumbent 0.28, break-even 26 vs 13, only positive net@24 = +0.16) — continuous weights churn less than top-K.
  But that config breaks in RECENT and the net-diff wasn't CI'd, so it's only a HINT that turnover could be squeezed,
  not a result. Possible narrow follow-up: a lower-turnover continuous construction with time-decay.
- VERDICT: **4th SOTA idea tested, 4th non-improvement** (regime-gate / dispersion / vol-management / direct
  weight-optimization). The incumbent construction (per-symbol Ridge + band + beta-hedge + liquidity filter) is at
  or near the frontier for THIS information set. Signal AND construction on existing free data are now thoroughly
  exhausted; the only remaining lever for MORE is ORTHOGONAL DATA. Script: build_ppp.

## Research cycle 5 — turnover-optimized construction (EWMA smoothing): TURNOVER WIN, Sharpe TIE
Motivation: user 'cost is mostly long-tail slippage' (= convex) + cycle-4 texture (continuous weights churned
less). Test on incumbent preds: turnover-control levers vs the hysteresis band, NET@cost + block-boot CI on the
NET@24 Sharpe diff, both eras, unhedged (isolates the turnover/net tradeoff; hedge is separable). `build_turnover_opt.py`.
- **EWMA weight-smoothing (keep top-K concentration, ramp positions in/out) CUTS TURNOVER 45-70%** (band 0.35 ->
  lam=.7 0.19 -> lam=.85 0.11) while NET@24 is a **statistical TIE both eras** (CI spans 0; RECENT point est even
  higher +3.16 vs +2.69; OOS -1.12 vs -1.30). Same net, ~half the trading.
- **Why the tie UNDERSTATES it**: net@24 prices cost LINEAR in turnover; real slippage is CONVEX (long-tail), so
  halving turnover saves MORE than the linear model credits -> EWMA likely a modest net WIN in reality + more
  capacity-scalable + less exposed to the calibrated-cost-model risk. Never hurt net in any cell. **Low-risk
  deployable turnover/robustness improvement** (use lam~0.7; NOT a gross-alpha claim). Validation caveat: the net@24
  Sharpe DIFF is a tie under the linear model; the convex-slippage upside is directional (needs real execution data
  to size), so adopt as turnover reduction, not as a proven Sharpe gain.
- **Concentration re-confirmed**: continuous z-weighting (spread across names) collapses gross mean and is
  decisively net-WORSE both eras (RECENT [-6.5,-3.4], OOS [-3.4,-1.6]). Re-confirms convex-sizing CLOSED
  (concentration is where the edge lives). Don't spread. (In OOS continuous had higher gross SHARPE from
  diversification but too little gross MEAN to clear retail cost.)
- **EWMA give-up is era-dependent**: mild in persistent RECENT (+23->+20.6 gross) but severe in fast-decaying OOS
  (+3.4->+0.9), since smoothing lags the signal -> keep lam conservative (0.7 not 0.85).
- VERDICT: first construction result since band+hedge with practical value: **~half the turnover at equal net**,
  net-favorable under the convex slippage the user describes. Adopt EWMA lam~0.7 as a turnover-control (augments/
  replaces the band). Not a Sharpe win. Script: build_turnover_opt.
- **CONFIRMATORY (build_ewma_hedge.py): EWMA STACKS with the beta-hedge** (orthogonal, as expected). In the DEPLOYED
  book (era-locked hedge on): band+hedge turn 0.35/0.34 (REC/OOS) vs EWMA.7+hedge turn 0.19/0.19; NET@24 tie both
  eras (REC [-0.77,+1.64] pt +3.05 vs +2.59; OOS [-0.51,+0.90] pt -1.02 vs -1.20) — point est nudges UP both eras.
  The EWMA turnover win banks cleanly into the deployed stack.

## Overall (5 cycles): stack = per-symbol Ridge + beta-hedge + liquidity filter + turnover-control (band and/or
EWMA lam~0.7). All wins are cost/risk/robustness-side; gross alpha is thin (~+3.3bps), ~one factor, era-unstable,
cost-gated at retail. 4 SOTA attempts to get MORE gross (regime/dispersion/vol-mgmt/PPP) all null. Remaining lever
for MORE = orthogonal data (needs acquisition).
