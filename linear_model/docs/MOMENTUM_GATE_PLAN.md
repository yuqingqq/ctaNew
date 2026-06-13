# Momentum + Meta-Label-Gate Plan (pre-registration)

Date: 2026-05-18
Status: **DRAFT — pre-registration NOT yet locked.** Nothing runs until the
§7 choices are locked by the owner. Locking = copying §5/§7 verbatim into the
script header as the fixed contract, then one run per phase.

## 1. Why this exists

The 76→91 linear arc reached a rigorous terminus: the cross-sectional
4h/24h β-residual on free-perp data (price/vol/funding **and** new OI) has no
edge that survives honest, hindsight-free, net-of-cost evaluation. Core issue
= the feature→residual relationship is **non-stationary** (and for OI the
sign actively inverts OOS). See `STATUS.md` top blocks,
`project_vBTC_linear_model`, `insight-coef-stability-not-predictive-stability`.

This plan tests the one axis the arc never touched, after an explicit user
challenge ("the whole thing is the same, only a gate added?") that was
substantially correct.

## 2. What is genuinely new vs what is identical

**Identical (do not pretend otherwise):** data, the `alpha_beta` β-residual
quantity, folds, PnL accounting (`pos × alpha_beta` net of cost), and the
rigor scaffolding (nested-OOS, matched placebo, block-bootstrap CI,
de-concentration).

**Genuinely new (the only reason to run this):**
1. **Base signal = time-series momentum, not cross-sectional rank.** The
   entire arc ranked symbols each cycle by a *learned* residual predictor.
   Here the base signal is each symbol's *own trailing direction* — a
   time-series axis the residual model never used.
2. **Sparse trade/no-trade economics** (Phase 2 only) vs continuous rank.

**Conceded:** a meta-label *gate* by itself does **not** escape the core
problem — it is the same non-stationary feature→outcome map relabelled as a
classifier, and Phase DDI already measured "predict when it's on" at
R² ≈ 0.005. Therefore the gate is **secondary and lower-prior**; the novel,
parameter-free element is the base rule, tested first and alone (Phase 1).

## 3. Honest prior + why fixed-hold / parameter-free (Steps 62–66 rationale)

**Guarded, and partly mapped.** Steps 62–66 tested the *adaptive*
mean-reversion-exit family (hold until β-residual mean-reverts: hybrid
signal-decay / target / time / stop; ranked-pool refill) on executable
blue-chips and **refuted it decisively**: nested-OOS −0.67; **plain
fixed-24h-hold (+0.77 in-sample) BEAT the adaptive exit by +1.44** (the
mean-rev trigger destroyed value); a **random-exit placebo (p95 −0.33) beat
the nested mean-rev exit** (random exit > the rule); the mean-rev-v2 that
briefly showed +3.51 was on the engine the Step-72 audit found
look-ahead-buggy → invalidated, corrected ≈0/neg. **Consequence (design
rationale):** an adaptive/tunable exit is both *already refuted* and a
K2/K3-class in-sample trap → Phase 1 uses a **fixed, parameter-free hold,
and the SIMPLEST one** (4h hold at the 4h non-overlapping decision cadence,
**no sleeve**) — to isolate "does the momentum-on-residual axis have a
pulse" with the absolute minimum machinery. The V3.1 6-sleeve is **not**
needed for the falsification and is deliberately *excluded* from Phase 1: its
only value is cost-amortization, which is an *enhancement* question (exactly
parallel to the gate), to be layered **only if the raw axis first shows a
pulse** — not baked into the primary test (conflating "is there signal" with
"does cost-amortized construction rescue it" is the over-engineering this
plan now removes). What was *not* run: a pure parameter-free
BTC-hedged-momentum rule, simplest fixed hold, cleanly net-of-cost,
matched-placebo, on the **full HL-executable** universe. That is Phase 1.
This is a **falsification test, not an optimization.** Symmetric-skepticism
rule applies throughout: correct over-claim *and* over-dismissal;
pre-registered gates are the only arbiter.

## 4. Universe & data (per phase — they differ)

- **Phase 1 universe = FULL HL-executable** (no OI dependency; Phase 1 uses
  no OI features). `panel110_hl_map.csv` `on_hl & hl_day_vol_usd ≥ $2M`
  ≈ 42 (hl42, the clean standard used through 76–88) — or all-`on_hl`
  ≈ 70 as the broader option. Open param **U1**.
- **Phase 2 universe = OI ∩ HL-executable** (the gate needs OI features;
  ≈15–20 from the 23 cached-OI symbols filtered to HL-liquid). If Phase 1
  passes, fetch the missing ~86 symbols' OI from Binance Vision
  (`metrics_loader.py`, network) — justified *because* Phase 1 showed
  signal (don't do the big fetch otherwise).
- Target/PnL source: `outputs/vBTC_features_btc_only_111_full_pit/...`
  (`alpha_beta`, `return_1d`, `sigma_idio`→`target_z`, `exit_time`,
  `autocorr_pctile_7d`); folds via `_multi_oos_splits` / `_slice`.
- Gate features (Phase 2): audited `outputs/vBTC_features_oi/oi_panel.parquet`
  + asset-state (return_1d, atr_pct, idio_vol, funding_z) + xs-dispersion +
  BTC-regime. All PIT (already audited).

## 5. Phase 1 — ungated parameter-free β-residual trend/revert test

The whole "is there anything different here" question, with the least
machinery and zero overfitting surface.

- **Signal quantity — LOCKED:** **BTC-hedged (β-residual) trailing return**
  over `L`: `s_t = ret_asset[t−L,t) − β_pit · ret_btc[t−L,t)` (strict-PIT
  `beta_btc_pit`, trailing raw returns). Rationale: PnL is on `alpha_beta`
  (forward β-residual), so the signal must be the *trailing analog of the
  same quantity*. Raw price momentum is **rejected** (for high-β alts ≈
  `sign(BTC return)`, a directional BTC bet the PnL strips → mismatch);
  trailing-sum-`alpha_beta` **rejected** (forward-label overlap hazard).
- **`s_t` PIT CONSTRUCTION — LOCKED (highest look-ahead-risk component;
  built & audited as a dedicated step before any strategy run):**
  - **DO NOT use panel `return_pct` or `btc_ret_fwd` for the trailing
    signal — both are FORWARD 4h returns** (the legs of `alpha_beta =
    return_pct − β·btc_ret_fwd`); compounding them "trailing" = direct
    look-ahead.
  - Build from 5m **klines** (close): `ret_asset[t−L,t) =
    close_asset[t−1]/close_asset[t−1−L] − 1`; `ret_btc[t−L,t) =
    close_btc[t−1]/close_btc[t−1−L] − 1` — computed on the kline series
    then `.shift` so the value at decision bar `t` uses only bars `≤ t−1`
    (the obv_z / OI-builder convention). `β_pit` = panel `beta_btc_pit`
    (already strict-PIT, shift-49). `s_t = ret_asset_L − β_pit·ret_btc_L`.
  - **MANDATORY pre-run PIT audit** (same gates as the volume/OI builders;
    Phase 1 does NOT run until PASS): (a) independent strictly-past
    recompute exact-match on ≥2 symbols (corr 1.0, maxdiff ≤ float32-eps);
    (b) `|corr(s_t, forward alpha_beta)| < 0.10` look-ahead check, pooled
    & worst-symbol. This is the discipline that caught the `inf`/`target_z`
    bugs in Steps 89/91 — non-negotiable here given the forward-field trap.
- **Direction is LOCKED to the project's prior thesis, NOT searched.**
  Base rule = **β-residual CONVERGENCE / fade**: `pos_t = −sign(s_t)` — if
  a coin idiosyncratically ran up (outperformed its β path), short it
  expecting the residual to converge. This is the project's standing
  economic premise (β-residual = a mispricing that converges; the mean-rev
  architecture; "short the one that ran up"). It is pre-registered from a
  genuine prior — the *disciplined* alternative to coin-flipping both
  directions (testing trend & revert symmetrically is itself a
  multiple-testing inflation). **Trend (`+sign`) is NOT a co-hypothesis
  here.**
- **No mid-investigation direction flip (anti-p-hack rule).** If the
  convergence base shows no pulse, the honest conclusion is "the residual-
  convergence thesis has no parameter-free executable pulse" — NOT "try
  momentum instead." Momentum would be a separate, independently
  pre-registered *future* question, never a same-run post-hoc rescue.
  (Honest note: the convergence *sign* was never cleanly validated either
  — the mean-rev-exit line failed for *exit-rule/look-ahead* reasons, not
  sign; pre-registering convergence is justified by the prior, not by
  proven sign.)
- **Direct autocorrelation diagnostic — TRANSPARENCY ONLY, not the
  arbiter and CANNOT flip the pre-registered direction:** pooled &
  per-symbol corr( `s_t` , forward `alpha_beta` ) at the non-overlapping
  cadence — reports what the residual's own sign actually does (never
  measured before); informs interpretation; the gated trade test on the
  *fixed convergence* rule still decides.
- **Hold / construction (pre-registered, ZERO free params, SIMPLEST):**
  `pos_t = −sign(s_t) ∈ {−1,+1}` per symbol, held over the **4h forward bar
  at the 4h non-overlapping decision cadence — NO sleeve**.
- **Position sizing & accounting — LOCKED (exact, removes over/under-charge
  ambiguity):** equal-weight across the symbols *active that cycle* (= those
  with a valid `s_t` and `alpha_beta`), `N_t` = count of active symbols:
  - portfolio cycle GROSS bps = `mean_i( pos_{i,t} · alpha_beta_{i,t} ) · 1e4`
  - portfolio cycle COST bps = `mean_i( |pos_{i,t} − pos_{i,t−1}| ) ·
    COST_PER_UNIT_ABS_DELTA` (s64; `|Δpos| ∈ {0,2}`, a flip = 4.5 bps @
    VIP-0). First appearance: `|Δpos| = |pos_{i,t}|` (entry).
  - NET = GROSS − COST. **Report GROSS and NET separately** (Step-91
    decomposition — "no signal" is never conflated with "killed by cost").
  This is exactly the Step-90/91 equal-weight convention
  (`groupby(open_time).mean()`), now written into the contract.
- The V3.1 6-sleeve cost-amortizing construction is an **optional
  enhancement (Phase 1b)**, layered only if Phase 1 shows a *gross* pulse —
  NOT part of the primary (see §6).
- **Locked choices needed:** **only L** (lookback). Direction is FIXED to
  convergence/fade (`−sign(s_t)`) — not a choice, not searched; hold =
  simplest 4h/no-sleeve (fixed). See §7 **B1**.
- **Evaluation:** per-symbol-timing portfolio, equal-weight, 4h
  non-overlapping cadence; block-bootstrap CI on GROSS and NET; per-symbol
  breakdown + drop-top-2; matched **random-sign** placebo (150 seeds, same
  turnover). Walk-forward only — **no fitted parameter**, so
  nested-vs-insample is moot (nothing to overfit).
- **Pre-registered PASS (fixed before run; ALL, on NET):**
  - P1 portfolio NET annualized Sharpe block-bootstrap CI excludes 0
  - P2 NET Sharpe > matched random-sign placebo p95
  - P3 ≥ 60% of symbols individually NET-positive AND drop-top-2 survives
  - P4 not a single-fold artifact (≥ 6/9 folds NET-positive)
- **Outcome logic:** PASS all → the β-residual-CONVERGENCE rule has a real,
  parameter-free, executable, net-of-cost pulse — **this is the headline
  result**; enhancements (Phase 1b sleeve, Phase 2 gate) are then *optional*.
  GROSS-positive but NET-fails-only-on-cost → no net edge, but Phase 1b
  (cost-amortization) is *warranted* (Step-91 logic). GROSS ≤ 0 → no pulse;
  enhancements moot; momentum line closed honestly, recorded same day.
  (One run.)

## 6. Optional enhancements — ONLY if Phase 1 shows a pulse, NON-mandatory

If Phase 1 passes, **that is the headline result**; the enhancements below
are *optional improvements*, not second mandatory proofs. Symmetric, parallel
treatment: the sleeve and the gate are both enhancement layers on a raw
signal that must first show a pulse.

### Phase 1b — V3.1 cost-amortizing sleeve (optional)
- **Run if:** Phase 1 is GROSS-positive (whether or not NET passed) — i.e.
  there is a real signal whose binding constraint may be cost (Step-91 logic).
- **What:** re-run Phase 1's locked base rule held 24h via the V3.1
  equal-weight 6-overlapping-sleeve (4h entry, HOLD_BARS=288, 1/6 weights —
  a fixed, non-tunable discrete construction, like K=3; its sole purpose is
  turnover/cost amortization, ~21%→~12% cost/gross).
- **Adopted iff:** NET Sharpe lift ≥ +0.5 vs Phase-1 primary **without
  degrading P3/P4 robustness**, and still beats the matched random-sign
  placebo p95. Else: not adopted (context only). Not mandatory.

### Phase 2 — the meta-label gate (optional, lower-prior)
Lower-prior (inherits the non-stationarity per Phase DDI); pursued only to
*concentrate* a base rule already shown to have a pulse. **Optional, not a
second mandatory proof.**
- **Run-precondition — LOCKED (prevents the gate becoming a selection
  loop):** Phase 2 runs **only if Phase 1 passes NET, OR Phase 1b is
  adopted and passes NET.** If Phase 1 is only GROSS-positive but Phase 1b
  fails (cost not rescued), the learned gate is **NOT** run. The gate is a
  pure enhancement of an already-NET-positive base — never a rescue / a
  profitability search on a net-negative base.

- **The gate is binary trade / flat ONLY — it NEVER changes direction.**
  Direction stays the fixed convergence/fade `pos_t = −sign(s_t)`; the gate
  only decides whether to take that position this cycle or sit out.
- **Label:** `y = 1 if (pos_t × alpha_beta_t − cost) > 0 else 0`
  (`pos_t` = the fixed convergence position).
- **Gate features:** §4 set (OI + asset-state + dispersion + BTC-regime).
- **Model (M1):** primary = logistic regression; **plus** a no-fit
  signed-composite gate for estimator-consistency (Step-84 G4). Optional
  shallow LGBM = non-gating secondary.
- **Threshold (T1):** primary = fixed 0.5 (zero free param). Nested-selected
  threshold = secondary, not gated on.
- **Decisive variant:** `gate_nested` — gate trained on folds < k only,
  applied to fold k (Step-88/90 standard). `gate_insample` context only.
- **Gate ADOPTED iff (fixed before run; `gate_nested`, ALL)** — pure
  enhancement logic (the user's preferred fix; the old self-contradicting
  "base must not already pass" clause is removed, since Phase 2 only runs
  *because* Phase 1 / base passed):
  - G1 `gate_nested` NET annualized Sharpe block-CI excludes 0
  - G2 > matched **random-gate** placebo p95 (same fire-fraction, 150) —
    ensures the lift isn't just any sparsification of a heavy-tailed base
  - G3 **improves** the Phase-1 primary: NET-Sharpe lift ≥ +0.5 **AND does
    not reduce P3/P4 robustness** (still ≥60% syms NET-positive +
    drop-top-2 survives + ≥6/9 folds)
  - G4 estimator-consistent: logistic & signed-composite gates same-sign,
    both > 0
  - G5 leakage/PIT clean + shuffle-gate ≈ placebo (signal-dependent)
- Gate fails any → **not adopted; Phase-1 result stands as the headline**
  (the gate was an optional enhancement, never a mandatory second proof).
  One run; budget discipline.

## 7. Pre-registered choices to LOCK (owner sign-off required)

| # | choice | proposed default | alternatives |
|---|---|---|---|
| U1 | Ph1 universe | **full HL-exec, $2M floor ≈ 42 (hl42)** | all-on_hl ≈ 70 (broader). [Ph2 = OI∩HL, fetch more OI if Ph1 passes] |
| B1 | base: L only | **L = 24h** (signal = β-hedged trailing return, §5 LOCKED; direction FIXED to convergence/fade `−sign(s_t)` — NOT searched; hold = simplest 4h/no-sleeve, FIXED) | L ∈ {12h, 48h} pick ONE, no sweep |
| C1 | cost | `s64.COST_PER_UNIT_ABS_DELTA` (2.25 bps/unit |Δw|) | HL maker ~1 bps; 2× stress |
| M1 | gate model (Ph2, optional) | logistic + signed-composite-consistency | + shallow LGBM (non-gating) |
| T1 | gate threshold (Ph2, optional) | fixed 0.5 | nested-as-primary |
| G3 | enhancement margin (Ph1b/Ph2, optional) | NET-Sharpe lift ≥ +0.5 & no P3/P4 loss & beats placebo p95 | tighter/looser |

## 8. Discipline / invariants (non-negotiable)

- One run per phase. No parameter sweeping (a swept lookback/threshold is
  the K2/K3/L4 in-sample trap).
- Decisive metric = nested-honest (Ph2) / parameter-free (Ph1) + matched
  placebo + net-of-cost + block-CI. In-sample numbers are context only.
- Auto-verdict strings in scripts are not authoritative — record the
  honest pre-registered-gate verdict in `STATUS.md` + `project_vBTC_linear
  _model` memory the same day, symmetric (correct over-claim AND
  over-dismissal).
- Executable universe only (meme/non-HL targets are non-deployable —
  triple-confirmed Steps 55–61, 79). Production LGBM untouched throughout.

## 9. Expected artifacts

- Phase 1: `linear_model/scripts/92_tsmom_base.py` →
  `linear_model/results/step92_tsmom_base/{summary,verdict}.csv` + log.
- Phase 1b (optional, if Ph1 gross-positive): `93_tsmom_sleeve.py`.
- Phase 2 (optional, lower-prior): `94_tsmom_gate.py`.

## 10. Status / next action

LOCKED & Phase 1 RUN (2026-05-18, defaults: hl42, L=24h, convergence/fade
`−sign(s_t)`, 4h hold no-sleeve, VIP-0). **Phase 1 = pre-registered FAIL.**
PIT audit PASS (s_t exact-match, look-ahead IC −0.027 fade-sign-confirmed).
GROSS Sharpe +0.78 (+1.29 bps/cyc — real parameter-free pulse) but NET
+0.25 CI[−2.07,+2.47] (P1 fail), 55% syms (P3 fail), 4/9 folds (P4 fail),
P2 weak-pass vs deeply-negative permutation placebo. Same Step-91 sub-cost
+ breadth/fold-fragile pattern. Per §6, **Phase 1b (V3.1 cost-amortization
sleeve) is contractually warranted** (GROSS-positive / NET-fails-on-cost) —
**but guarded prior**: the sleeve amortizes cost only; P3/P4 are breadth/
fold failures it cannot fix, so even a NET lift is unlikely to yield a full
P1–P4 pass.

**Step 92b (2026-05-18) — §7-U1 ALTERNATIVE universe (all-on_hl ≈70),
robustness NOT a retry:** hl42's FAIL is final; this tested whether the
failure is composition-specific or structural. Near-identical FAIL: GROSS
+0.88, NET +0.28 CI[−2.13,+2.56] (P1 fail), 56% syms (P3 fail), 5/9 folds
(P4 fail), P2 only-pass vs placebo p95 −2.60 (uninformative). PIT audit
PASS. **The Phase-1 failure is STRUCTURAL, not composition-specific**
(consistent with Step-79). This *strengthens* the guarded Phase-1b prior:
P3/P4 persist under universe-widening, so cost-amortization (P1-only) is
even less likely to deliver a full pass. No multiple-comparison concern
(also failed; no cherry-pick).

**Step 93 (2026-05-18) — Phase 1b RUN (owner chose option A; §6 V3.1
cost-amortizing 6-sleeve, one run, hl42): NOT adopted.** PIT audit PASS;
Phase-1 exactly reproduced in-harness (validates the harness). Sleeve NET
Sharpe **+0.02** CI[−2.18,+2.06]; **lift −0.23** vs Phase-1 +0.25 (adopt
needed ≥ +0.5); P4 still fails (4/9 folds). **The sleeve *backfires*:**
turnover/cost fell (|Δ| ×0.44, cost ×0.43) but GROSS fell *harder* (×0.32),
so cost/gross got *worse* 68%→93% (contract had hypothesized ~21%→~12%).
Root cause — the β-residual convergence edge is a **fast 4h reversion pulse
that does not survive a 24h hold**; the 6-sleeve MA dilutes the position
while the edge is reverting. (Generalizable: V3.1 amortizes cost only when
alpha persists over the hold — `insight-v31-sleeve-needs-persistent-alpha`.)
P4 fold-fragility (cost-independent, per 92b) persists as the guarded prior
predicted.

**TERMINUS (per §6).** Phase 1b NOT adopted ⇒ Phase-2 gate precondition is
UNMET (Phase 1 net-failed AND Phase 1b not adopted). The pre-registered
momentum-gate line is **contractually closed**: the linear β-residual
convergence signal is a real, PIT-clean, parameter-free GROSS pulse
(~+1.3 bps/cyc) with **no parameter-free, net-of-cost, robust executable
edge on free 4h Binance perp data**; the cost-amortization rescue fails for
a mechanistically clear reason. Honest, well-understood negative — a
property of the signal, not a testing failure. No direction flip
(anti-p-hack). Production LGBM unaffected. Owner's only remaining option is
to override the §6 precondition and run the gate as an explicitly
*exploratory, non-adoptable* diagnostic (strong negative prior: Phase-DDI
R²≈0.005 + Step-90 nested −2.60) — otherwise the line is done.
Artifacts: `step92_tsmom_base/`, `step92b_tsmom_allhl/`,
`step93_tsmom_v31_sleeve/`.
