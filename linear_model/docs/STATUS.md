# Linear Model — STATUS

Last updated: 2026-05-19
Owner: this session

> ## ✅ RESOLUTION (2026-05-19) — retraction discharged; line CLOSED as a
> **narrow, honest negative** (NOT "information-bounded"). The 2026-05-18
> retraction below was acted on in full: `composite_study/harness_v3.py`
> rebuilt to plan v3 §2 — per-fold strict-past per-symbol σ_idio with NO
> cross-symbol fallback (fixes the contaminated TARGET, finding #7); the
> frozen panel `sigma_idio` is never consumed; static frozen transform
> map actually wired; Ridge α-grid + chronologically-early-stopped LGBM
> envelope; native universe on one common 4h calendar; **3 GENUINE
> BLOCKING self-checks** (prefix-causal Δ=0 at interior cuts;
> post-preprocess PIT incl. magnitude sniff <0.10; σ_idio strict-past +
> explicit single-symbol-isolation, Δ=0) — closes finding #9 (fake/prose
> checks). Each step 3-agent-reviewed (R1/R2/R3) vs pre-stated
> expectation; re-initiated once when R2 caught two §4-design
> false-positive vectors (cost asymmetry; baseline survivorship).
> **§4** (the only real profitable path — ensemble-with-production):
> SUCCESS-B FAIL, **3/3-agent reproduced** — vs the *matched-grid* V3.1
> honest-forward **+2.747** (folds 3–9, 1260 cyc; the prelim's "+0.92
> lift" was 100% wrong-baseline survivorship, formally void), nested
> var-min blend lift **−0.92** (paired CI [−1.95,+0.14]); in-sample
> upper bound == V3.1 ⇒ infeasible for ANY weight. corr(lin,V31)=+0.05
> held but a near-uncorrelated *unprofitable* sleeve cannot accrete.
> **§3.5** (target-framing — last in-scope independent hypothesis, flaw
> #8): NEGATIVE, **3/3-agent reproduced** — short-only (DDI) &
> magnitude-conviction (Step-80a), ridge & lgbm, both cost rates: best
> standalone +1.316 < +1.5 (CI incl 0), every Success-B lift negative.
> Magnitude is mildly predictable (IC +0.12) but directional IC≈0 ⇒
> unmonetizable; DDI short-side alpha is real as a cohort IC spread but
> **portfolio-invisible under 4h cost**. **§5-INT** (OWNER-AUTHORIZED
> re-open — owner asked "why not spot-perp / price-volume interactions";
> 26 locked interactions, Tier A 42-sym price-vol/short×long no-shrink +
> Tier B 19-sym spot-perp/order-flow/OI): NEGATIVE, **3/3-agent
> reproduced**. Re-initiated once — R1 caught the first interaction
> magnitude-PIT guard as a mis-calibrated leak test (false-positives on
> benign vol-clustering); fixed via a FRESH pre-registration (§5-INT-v2)
> with a leak-specific G3 asymmetry guard (red-team-proven still catches
> real leaks), NOT a retro-edit. Interactions add only single-fold
> variance-luck marginal structure; every cell fails Success-A &
> Success-B; spot-perp/price-volume/order-flow add no tradeable signal.
> §3 horizon gated-off + out-of-scope (user fixed goal to 4h); §5
> forbidden by the pre-registered kill; §7 N/A. **The findings (1)–(8)
> below were all
> addressed, not waved away; the honest re-test confirms a NEGATIVE but
> a NARROW one** — out-of-scope levers (orthogonal data, longer horizon;
> R2 P≈20–30%) are untested, not refuted. Single source of truth =
> `composite_study/PROGRESS.md`. Production LGBM (V3.1) unaffected
> throughout and remains the production strategy.

> ## ⛔ RETRACTION (2026-05-18) — Steps 94–102 + composite_study "airtight/
> definitive/terminus" verdicts are WITHDRAWN as conditional on a flawed
> harness. A 3-independent-agent review (R1 methodology / R2 profitability /
> R3 red-team) found, validated: (1) the §0.5 preprocessing was never wired
> into the code — every "ceiling"/feature run was StandardScaler-only Ridge
> / raw un-early-stopped LGBM, so "LGBM-can't-beat-Ridge ⇒ information
> ceiling" is unproven (mis-tuned nonlinear arm, not absent structure);
> (2) the per-symbol "t=+9.36, real & broad" stat treats 42
> cross-correlated symbols as iid → significance inflated ~6–10×;
> (3) the +1.5/CI-excl-0 gate calibrated on 42 syms was applied to the
> 19-sym ∩ universe (CI symmetric-about-0 by breadth → FAIL baked in);
> (4) "sub-cost" used worst-case naive sign(pred) 100%-turnover at retail
> 9bps-RT at naked-4h — the one horizon the project's OWN evidence (sister
> 24h-sleeve strategy) shows is cost-pathological; even zero-cost CI
> crosses 0 ⇒ the binding constraint AND success criterion were
> mis-specified; (5) the profitable objective (low-corr sub-bar sleeve,
> portfolio-accretive vs production) was never tested. **"Information-
> bounded / closed" is NOT established.** The linear line is RE-OPENED.
> Corrective harness-rebuild + re-aimed plan:
> `composite_study/FEATURE_REENGINEERING_PLAN.md` (being rewritten;
> 3-agent review loop continuing). Production LGBM unaffected throughout.

> **Round-2 3-agent re-review (2026-05-18) — deeper findings:** (7) the
> `sigma_idio` TARGET is itself contaminated — `s94.build` consumes the
> panel's fold-0-frozen σ with a cross-symbol-MEDIAN fallback for
> late-listed (often "best") symbols ⇒ every ceiling in the whole arc
> inherits a mis-scaled, look-ahead-asymmetric target; (8) the
> symmetric `sign(pred)` 4h-residual-return framing may itself be
> mis-specified — the arc's own evidence (Step-77 decile-inversion,
> Step-80a magnitude/squared structure, production DDI short-side-only)
> points to an asymmetric/magnitude/longer-horizon target, which
> rank→inverse-normal preprocessing actively destroys. Plus: the only
> real path to a *profitable* outcome is the untested ensemble-with-
> production sleeve (§4), not feature work. Plan v3
> (`composite_study/FEATURE_REENGINEERING_PLAN.md`) locks the full
> convergent fix-list; next = code-level harness rebuild incl. the
> target fix, then §4 first. The whole prior closure is not just
> "conditional" — the target it was measured on is contaminated and
> possibly the wrong target. Production LGBM still unaffected.

> **2026-05-18 (superseded by the retraction above):** momentum-gate line
> "terminus" (Step 93) and the D1→D3 "information diagnostic" verdicts —
> all conditional on the flawed harness; do not treat as closed.

---

## ✅ DECISIVE CLOSURE (2026-05-17, Step 77): orientation rank-IC is statistically real but ECONOMICALLY DEAD

Supersedes the Step-76 block below. `77_orientation_decile_diag.py` (346s, no
backtest) reconstructed the exact Step-76 A_4h composite (imported s76 helpers;
sanity cycle-IC = **+0.0517**, byte-matches) and asked the user's question:
*is the +0.052 IC monetizable in any band, or rank signal that inverts at
tradable extremes?*

Answer — **it inverts and is non-monotonic; not economically useful:**

| diagnostic | result |
|---|---|
| Decile monotonicity ρ | **−0.49** (D0 low-score +3.11 bps → D9 high-score +1.34; declining + noisy) |
| Quintile ρ | −0.30 |
| L/S by K=1 | long **−2.19** / short **+13.72** → spread **−15.91** (t −1.74) — extreme inversion |
| L/S by K=2/3/5/10 | −3.64 / −2.03 / −1.68 / −0.95 — all negative, → universe mean as K→all |
| best interior band (D6-8/D1-3) | +0.30 bps, t +0.20 — nothing monetizes anywhere |
| fold-level decile ρ | sign-flips (+0.76 … −0.54); only **4/9** positive; D0 +49.7 (f4) to −25.0 (f5) |

Per the user's pre-stated rule (*deciles not monotonic → IC not economically
useful*): **the orientation line is closed economically.** The +0.052 rank-IC
is broad, weak, mid-rank co-movement whose magnitude-bearing extremes are
*anti*-correlated; "flip the sign" is fold-unstable (K3/decay overfit trap),
not a rescue. The K=1 short=+13.7 is a heavy-tail artifact (53% pos).

**This is the correct, direct closure Step-75 reached via a flawed proxy.**
Net reconciliation of the three steps: Step-75 economic conclusion *right*,
its basis *wrong* (MSE proxy); Step-76 *right* that a real rank-IC lever
exists (Step-75 "no signal" overclaimed); Step-77 *decisive* — the lever has
no monotone/extreme-tradable payoff. **Verdict robust to the pending leakage
audit**: the diagnostic measures payoff, not IC; a leak could only inflate an
IC that already carries zero economic value → the audit is no longer
decision-gating (optional method-hygiene only).

**Scope (do not over-generalize):** established dead = orientation of the
existing **V2 22 features at the 4h `alpha_beta` target on the 42-sym HL≥2M
testbed**. Genuinely *unresolved*: proper-24h (Step-76 Part-B bug). *Untested*:
the conditional Step-3 price×volume×volatility interaction features. No
backtest, no optimization, no further orientation tuning. Production stays
LGBM. Artifacts: `linear_model/results/step77_orientation_decile_diag/`.

### Step 78 addendum (2026-05-17): model-side exhausted, bottleneck = features

Compressed constrained-model trial (`78_nnls_poscoef_payoff.py`, NNLS +
positive-Ridge on sign-oriented features, vs raw-Ridge / signed-equal /
signed-shrunk anchors; same testbed; **sanity: signed_shrunk reproduced
IC +0.0517 / ρ −0.491 byte-match**). Pre-registered payoff gate (ρ≥+0.60 AND
K3≥+9 bps AND ≥6/9 folds): **no model clears → linear-on-current-features
CLOSED.** Honest nuance recorded (not a gate override): model *form* drives
payoff shape — shrunk-IC **inverts** (ρ −0.49, K1 −15.9), equal-sign
**flattens** (ρ +0.64 but K3 +0.8), and **NNLS/pos-Ridge give the only
positive non-inverting fold-robust (7/9) payoff** (~+6–7 bps K3, IC +0.026
t +5.0). It is a **real but sub-cost signal** (~6–7 bps K3 gross vs ~9 bps RT
floor → net ≈ 0), not noise. Decisive implication: **the bottleneck is the
feature set, not the model** — model side is exhausted. Two approved next
probes (cheap, no backtest): (a) **broader-universe attribution diagnostic**
(hl42 / hl-all≈70 executable / Binance-110 research-only) — does breadth push
the +6–7 bps constrained signal past cost, or is any widening a meme/illiquid
tail? per-symbol attribution + drop-top-2 + executability built in; (b) Batch B
price×volume interaction features. Artifacts:
`linear_model/results/step78_nnls_poscoef_payoff/`.

### Step 79 addendum (2026-05-17): broader universe does NOT rescue — meme-tail re-confirmed

`79_broader_universe_attrib.py` (3001s) — 3 universes, shared fold dates,
per-universe PIT rebuild; scores ridge_xsz / nnls_oriented / signed_equal;
Step-77 payoff **+ per-symbol gross attribution + HL-flag + drop-top-2**.
Pre-registered gate (Step-78 payoff gate **+** top-5 ≤ 60% pos-gross **+**
drop-top-2 K3 > 0; decision keyed off executable `hl_all`):

| universe | best K3 | drop-top2 | reading |
|---|---|---|---|
| hl42 (42, exec) | nnls **+6.77** | +4.75 | real but sub-cost, liquid top-5 (consistent w/ Step 78) |
| **hl_all (70, exec — decision)** | nnls **−1.56** | −4.61 | broadening the executable set **degrades to negative** (all 3 scores) |
| binance110 (110, research) | ridge +16.15 | +9.26 | SIREN\*/SOLV\*/JELLYJELLY\*/AVAAI\*/BROCCOLIF3B\* — **non-HL meme tail** |

**Verdict: `hl_all_rescued=False, binance110_memetail=True`.** Breadth does not
rescue the linear line — the executable broadening makes it *worse*; the only
"win" is the non-executable SIREN/JELLYJELLY/BROCCOLI tail (Steps 55–60
reproduced exactly), rejected automatically by the built-in instrumentation.
The linear β-residual line is now closed via **three independent routes**:
direct payoff (77, 42-sym), all model forms (78, 42-sym), all universes incl.
broader-executable (79). Meme-tail confound triple-confirmed. **Remaining
genuinely-untested:** only Step-80 Batch B (price×volume interactions) and the
unresolved proper-24h (Step-76 bug). Everything else on free 4h Binance data
is exhausted. Production stays LGBM. Artifacts:
`linear_model/results/step79_broader_universe/`.

### Step 80a addendum (2026-05-17): group ablation — dead payoff is structural, but the U-shape group is the lone carrier

User-directed (treat 22 feats as 5 groups; ablate by group; gate by group
payoff). `80a_group_ablation_payoff.py` (3961s) on hl42, scores
nnls_oriented + ridge_xsz; baseline / leave-one-group-out ×5 / single-group
×5 through the Step-77 payoff diagnostic + drop-top-2. **Scaling note: the
`*_sq` re-standardize the user proposed is a provable no-op here** —
per-cycle cross-sectional-z is invariant to train-affine rescale
(`xsz(α·c+β)≡xsz(c)`); valid hygiene only for the non-xsz production
`s58.train_ridge` path, flagged not run. **Pre-registered verdict: no
LOGO config clears the full gate → dead payoff is STRUCTURAL across groups,
not one noisy group dominating.** Honest nuance (recorded, not a gate
override): the **squared / U-shape group is the lone ESSENTIAL group**
(`ridge drop_squared` collapses K3 5.94→0.86, ρ 0.55→0.11, drop-top2
+8.27→−2.38) **and the best single-group payoff** (`ridge only_squared`:
K3 **+10.11**, ρ +0.552, **8/9 folds**, drop-top2 +8.16) — the closest
anything has come, failing the gate **only on ρ (0.552 vs 0.60)**. nnls
groups all "neutral" (signal diffuse). **Interpretation: the
residual-predictive structure is NON-MONOTONE (lives in squared/U-shape
terms — magnitude of move, not direction) — explains why every
signed-linear weighting topped out and why composites inverted/flattened
(76–78).** Actionable for Step 80b: build **non-monotone / magnitude
interactions** (|ret|×vol, ret²×vol, dist-from-high×vol) from the
**squared + btc_rel** groups, not naive signed price×volume. Still FAILS
the gate; honest base rate low (one config, ρ 0.05 short, project
near-misses historically don't survive). Artifacts:
`linear_model/results/step80a_group_ablation/`.

### Step 80b + 81 + 82a addendum (2026-05-17): NOT closed — narrow gate-fail; calibration layer rigorously ruled out; proper-24h is the live lead

**Framing correction (standing):** earlier blocks here over-asserted
"DEAD/closed/exhausted". A failed pre-registered gate = "no clean win among
configs tested", NOT "direction closed". See memory
`feedback-gate-fail-vs-research-exhausted`. The user has correctly overturned
my premature "dead" calls ~5× on this line.

- **Volume rebuilt from klines (user-directed):** `build_btc_vol_features.py`
  → volaug panel = full-PIT spine + 9 PIT volume/flow features, rows
  byte-identical. **Step 81 PIT audit PASS** (qvol_z_1d vs independent
  recompute corr 1.00000 maxdiff 0; all 9 features |IC|<0.03 ≪ 0.10).
- **Step 80b (volaug, hl42):** no config clears the pre-registered payoff
  gate. Raw vol-only is *negative* (ridge K3 −5.01, top-5 78% concentrated);
  non-monotone/magnitude interactions *degrade* vs v2 baseline. Best
  non-baseline `ridge sqbtcrel_plus_int` K3 +6.97 ρ +0.47 7/9 — still
  sub-cost. (The script's auto-verdict string says "CLOSED/exhausted" — that
  is the same over-assertion; the honest verdict is narrow gate-fail.)
- **Step 82a (carrier bucket-stability pre-gate, the user's two-layer plan):**
  measured fold-stability of the squared/non-monotone CARRIER itself
  (Step 77 only ever measured the shrunk-composite proxy). Result: sign/
  direction is fold-stable (same-sign decile-ρ ~8/9, dom=+) but the
  bucket-**shape** is NOT — consecutive-fold decile-profile correlation ≈0
  (best +0.06 vs +0.30 bar); top-3 profitable buckets pos in only 4–6/9.
  **Pre-gate FAIL on all 4 carrier×model combos.** Per the pre-registered
  two-layer plan, Step 82b (bucket→bps calibration) is **not run** — it would
  learn a profile shape that demonstrably does not persist OOS (the K2/K3/
  decay nested-OOS pathology), and the only stable component (coarse rank
  direction) is already captured by the rank gate (which tops sub-cost
  ~+6 bps). This is a *rigorous pre-registered closure of the calibration
  idea on the correct carrier*, not a premature quit.

**Net (honest):** orientation (76–78), universe (79), groups (80a), proper
volume + interactions (80b), and the bucket-calibration layer (82a pre-gate)
have each been tested under pre-registered gates and none produced a clean
tradable win. The signal is real but sub-cost and its exploitable structure
is the coarse rank direction only. **Genuinely-unresolved & untested:**
proper-24h (Step-76 `shift(-j*BLOCK)`+β bug; pre-bug +19 bps spread) — now
buildable correctly on the volaug panel. That is the next concrete step
(task #7), not closure. Production LGBM unaffected. Artifacts:
`linear_model/results/{step80b_vol_interaction,step81_verify_volaug,step82a_carrier_bucket_stability}/`,
`scripts/build_btc_vol_features.py`.

### Steps 83-87 addendum (2026-05-17): 24h FROZEN structure PASSES — strongest result of the arc, NOT yet validated

- **83** proper-24h (fixed Step-76 shift bug; β@289 matched-PIT): ridge huge
  but flagged. **84** rigorous gate: sqbtcrel/ridge G1/G2/G3 pass but **G4
  FAIL** (signed_equal −3.52, nnls inverts) → closed as standalone tradable
  by direct test; the scale-free rank-IC (ρ +0.74, 9/9) was real.
- **85** bars_since_low (volaug2, PIT-pass exact): marginal **+0.23 bps** K3,
  still est-inconsistent — clean new feature adds nothing. Adding raw
  features is exhausted.
- **86** (user-run) 24h sqbtcrel ridge diagnostic: edge robust to α
  (0.1–10⁴ all +1.8..+2.3, 8-9/9), concentrated in squared/btc-rel, but
  **in-sample RidgeCV** and signed_equal −3.52. Sign-stable coeffs:
  beta_to_btc_change_5d (9/0), return_1d_sq (9/0), corr_to_btc_1d_sq (9/0),
  dom_btc_z_1d (0/9); *squared-beta* and dom_sq sign-FLIP (1/8).
- **87** pre-registered FROZEN-structure nested-OOS test (the decisive one):
  **`sigstable3`** = zero-param signed-equal `+beta_to_btc_change_5d
  +return_1d_sq −dom_btc_z_1d` → net Sharpe **+2.19, CI [+0.46, +4.52]
  (excl 0), beats matched-placebo p95 +1.05, 7/9**, with **no in-sample
  fitting**. **NO K3 collapse**: in-sample +2.29 / nested-fit +3.21 / frozen
  +2.19 (nested ≥ in-sample ≥ frozen, all positive) — categorically unlike
  every prior K2/K3/W23 failure. G4 mechanistically resolved (the −3.52 was
  equal-weighting the sign-UNSTABLE squared feats; the sign-stable subset
  works frozen).

**Honest status: strongest, most rigorously-gated positive of the entire
investigation — and the first to be K3-collapse-free — but NOT validated.**
Residual bias: `sigstable3`'s 3-feature/sign *selection* used step-86
full-period hindsight; the hindsight-free `twogrp_equal` is +1.99 with **CI
crossing 0** [−0.17, +4.20], so the PASS leans partly on the structure pick.
CI also wide (small non-overlap 24h n + heavy tails). **Next (Step 88, task
#10): close the last loop — nested per-fold feature/sign selection (subset
from folds<k only, applied frozen), + matched placebo + CI + shuffle/negate +
24h-pipeline leak re-confirm.** If the nested-selected frozen composite still
clears → first genuinely-validated linear edge of the investigation; if the
selection churns/fails → it was structure-level hindsight. NO backtest until
that clears. Production LGBM unaffected. Artifacts:
`linear_model/results/{step83_proper_24h,step84_proper_24h_rigor,step85_bars_since_low,step86_ridge_24h_diagnostic,step87_frozen_structure}/`.

### Steps 88-90 addendum (2026-05-17): nested-honest closes 24h; per-symbol + NEW OI data closed by rigorous direct test

- **88** nested hindsight-free feature/sign selection on the Step-87 24h
  structure → FAIL all 4 (netSh −1.05, picks churn 4/9, shuffle≈negate≈0).
  The 24h "edge" was structure-level hindsight; under honest nested selection
  it is OOS-noise. Linear β-residual line has no edge surviving honest nested
  selection at 4h OR 24h — terminal, by direct loop-closed tests.
- **89-90 (per-symbol + OI/positioning, tested at user's insistence — the
  Step-67 rejection was invalidly on the pre-Step-72 leaky engine + wrong
  normalization).** Built a PIT OI panel from cached Binance-Vision metrics
  (23 liquid syms; create_time↔open_time alignment resolved + `.shift(1)`;
  the per-step validation gates caught & fixed two real bugs — pct_change
  `inf`, derived `target_z` — before they reached the model; PIT-audit PASS,
  look-ahead IC<0.10). Per-symbol model + per-symbol PIT trailing-z norm.
  **Promising intermediates** (V2 per-symbol OOS IC +0.015 mean, 70% of syms
  positive; V3 77% of (sym,feat) sign-consistent ≥7/9 — first concrete
  per-symbol-more-stationary signal vs the pooled ≈noise). **But the Step-90
  pre-registered determinant FAILED all 4 gates:** the nested-honest
  (past-only signs) per-symbol-timing portfolio = netSh **−2.60, CI
  [−4.86,−0.29] (significantly NEGATIVE)**, 26% syms+, drop-top2 −1.88,
  vs random-per-symbol-sign placebo p95 +1.32 / mean −0.03 → **below
  random**. Monotone worsening in-fold (−0.73) → full-train (−0.91) →
  nested (−2.60) = textbook non-stationary/overfit; per-symbol OI signs
  *actively invert* across regimes (crowded positioning mean-reverts), so a
  frozen past-sign is worse than random. **Per-symbol + OI direction CLOSED
  by direct rigorous test** (hypothesis was legitimate and given the proper
  test it deserved; verdict is decisive, not premature). Extends Insight-1:
  full-set sign-stability (cross-sectional OR per-symbol) is hindsight-
  flavored; only nested past-only + placebo + net-of-cost is decisive.

- **91 IC-vs-trading decomposition (refines 89-90; earlier "same illusion"
  framing corrected).** User asked why per-symbol IC was best-of-arc but
  trading bad. On the same pred_ridge: IC does **not** collapse at the
  honest non-overlapping 4h cadence (+0.0175, 70% syms+) → **NOT an
  overlap illusion** (unlike everywhere else); Step-90 reconciled exactly
  (not a bug). **GROSS portfolio Sharpe ≈ +1.0 (+0.83 bps/cyc) — a REAL
  positive gross micro-edge, the first of the investigation** — but turnover
  0.64×2.25 = +1.44 bps/cyc cost > +0.79 gross → net −0.73. Two separate
  limiters: the IC-positive ridge variant fails on **cost** (real but
  sub-cost); the honest frozen-sign variant fails on **sign
  non-stationarity** (signed_nested −2.60; the gross edge leans on per-fold
  refit). Honest refinement: per-symbol OI is a *real, ~70%-syms, gross-
  Sharpe-1.0 micro-edge that is both sub-cost AND honest-sign-unstable* —
  not noise, not tradable. Only lever = turnover/horizon reduction
  (longer hold / no-trade band / maker-HL exec), guarded & low-prior, must
  be pre-registered + nested-OOS + placebo. Script `91_ic_vs_trading_diag.py`.

**Net (whole 76→91 linear arc):** no edge survives honest, hindsight-free,
net-of-cost evaluation — across orientation, universe, groups, volume, 4h,
24h, nested structure, and per-symbol on genuinely-new OI/positioning data.
Refined nuance (Step 91): per-symbol OI is the one case with a *real* but
*sub-cost* gross micro-edge whose honest sign is non-stationary — not "pure
illusion," but not tradable. Core issue confirmed: non-stationary
feature→residual relationship on free perp data (price/vol/funding *and*
OI), not a model-capacity problem. Production LGBM unaffected/unchanged. New
PIT OI data + builder retained (`outputs/vBTC_features_oi/`,
`scripts/build_btc_oi_features.py`) for any future work. Artifacts:
`linear_model/results/{step88_nested_selection,step89_per_symbol_oi,step90_per_symbol_oi_rigor,step91_ic_vs_trading}/`.

### Step 92 addendum (2026-05-18): Phase 1 of MOMENTUM_GATE_PLAN — pre-reg FAIL, same sub-cost pattern

New direction (user-driven, fully pre-registered in `MOMENTUM_GATE_PLAN.md`,
locked before run): the project's standing **β-residual convergence/fade**
thesis as a parameter-free time-series rule — `pos_t = −sign(s_t)`,
`s_t = ret_asset[t−L,t) − β_pit·ret_btc[t−L,t)`, L=24h, hl42 (42 syms),
4h hold no sleeve, VIP-0, equal-weight. **PIT audit PASS** (s_t vs
independent strictly-past recompute corr 1.000000 maxdiff ~1e-8; look-ahead
IC −0.027, the −sign confirming fade is correctly oriented; built from
klines, panel forward fields explicitly excluded). **Pre-registered verdict
FAIL:** GROSS Sharpe **+0.78** (+1.29 bps/cyc) — a real positive
parameter-free gross pulse — but NET **+0.25, CI [−2.07,+2.47]** (P1 fail,
indistinguishable from 0), 55% syms & 4/9 folds (P3/P4 fail); P2 passes only
vs a deeply-negative permutation placebo (weak bar). **Same Step-91
sub-cost-micro-edge pattern, now for the cross-sectional fixed-convergence
rule:** binding constraints = transaction cost (+0.88 bps/cyc drag) +
structural breadth/fold inconsistency. Symmetric read: not dead at gross
level (real PIT-clean +0.78, fade direction confirmed) but not a net edge
(fails locked gate). Contract §6 triggers **Phase 1b (V3.1 cost-amortization
sleeve)** — but honest prior **guarded**: the sleeve only amortizes *cost*;
P3/P4 are breadth/fold failures cost-amortization cannot fix, so even a
NET lift likely won't yield a full P1–P4 pass. No direction flip
(anti-p-hack). Production LGBM unaffected. Script `92_tsmom_base.py`;
artifacts `linear_model/results/step92_tsmom_base/`.

### Step 92b addendum (2026-05-18): Phase 1 on pre-registered ALTERNATIVE universe (all-on_hl ≈70) — robustness, NOT a retry

The hl42 (locked primary) Step-92 verdict is **final**; this runs the §7-U1
pre-registered *alternative* universe (all-`on_hl`, drop the $2M floor, 70
syms = 42 + 28 sub-$2M HL names) purely as a **breadth/robustness
diagnostic**: is the hl42 failure composition-specific or structural? Honest
prior was guarded-negative (less-liquid names; VIP-0 under-charges them;
**Step-79 already showed this exact broadening degrades** the cross-sectional
residual). Identical locked contract (reuses Step-92's audited code; only the
universe filter changes). **PIT audit PASS** (s_t exact-match SOL/ADA;
look-ahead IC −0.0275). **Result — near-identical FAIL:** GROSS Sharpe
**+0.88** (+1.29 bps/cyc) but NET **+0.28, CI [−2.13,+2.56]** (P1 fail),
**56%** syms (P3 fail), **5/9** folds (P4 fail); P2 passes only vs deeply-
negative permutation placebo p95 −2.60 (uninformative — says the rule churns
less than random sign-flipping, not that it has alpha). Numbers barely move
vs hl42 (GROSS +0.78→+0.88, NET +0.25→+0.28, syms 55→56%, folds 4→5/9). No
cherry-pick risk (it also failed). **Conclusion: the Phase-1 failure is
STRUCTURAL, not composition-specific — broadening the executable universe
does not rescue it (consistent with Step-79). The β-residual convergence
signal is sub-cost + breadth/fold-fragile across BOTH pre-registered
executable universes.** This also *strengthens the guarded prior on Phase
1b*: P3/P4 are non-cost failures that persist when the universe is widened,
so a cost-amortization sleeve (fixes only P1's cost drag) is even less
likely to deliver a full P1–P4 pass. Production LGBM unaffected. Script
`92b_tsmom_base_allhl.py`; artifacts `linear_model/results/step92b_tsmom_allhl/`.

### Step 93 addendum (2026-05-18): Phase 1b — V3.1 cost-amortizing sleeve — pre-registered NOT adopted; the sleeve *backfires* (mechanism identified)

Owner chose **(A) run Phase 1b** (contractually warranted: Phase 1
GROSS-positive). Locked §6 construction: Phase-1's rule held 24h via the
V3.1 equal-weight 6-overlapping-sleeve = trailing 6-step (24h) MA of the
±1 convergence position per symbol on the 4h grid, applied to the *same*
forward-4h `alpha_beta` (NO forward shift → structurally immune to the
Step-76 24h-shift bug). One run, hl42. **PIT audit PASS; Phase-1 exactly
reproduced in-harness** (GROSS +1.29, NET +0.41 bps/cyc, Sh +0.25 — matches
Step-92 hl42, validates the harness). **Result — NOT adopted (decisive):**
sleeve NET Sharpe **+0.02**, CI [−2.18,+2.06]; **lift −0.23** vs Phase-1
+0.25 (adopt needed ≥ +0.5); P4 still fails (4/9 folds). **Mechanism — the
sleeve backfires, opposite of the production-vBTC case:** turnover did drop
(|Δ| 0.39→0.17, ×0.44) and cost did fall (0.88→0.38 bps/cyc, ×0.43) — but
**GROSS fell *harder*** (+1.29→+0.41 bps/cyc, ×0.32), so cost/gross got
*worse* (68%→**93%**), not better (contract had hypothesized ~21%→~12%).
Root cause: this β-residual convergence edge is a **fast 4h reversion pulse
that does not survive a 24h hold** (autocorr −0.027, fade-confirmed, edge
realized in the immediate forward 4h). Averaging the position across a 24h
window via the 6-sleeve MA dilutes it *exactly while the edge is decaying/
reverting* → destroys ~⅔ of gross alpha while trimming only ~½ the cost. V3.1
amortizes cost **only when the underlying alpha persists over the hold** (the
production LGBM signal does; this short-lived reversion does not) — a sharp,
generalizable boundary condition on when the sleeve helps. P4 fold-fragility
(cost-independent, per Step-92b) also persists, as the guarded prior
predicted. **Per §6: Phase 1b NOT adopted → Phase-1 result stands as the
headline; the Phase-2 gate precondition is UNMET** (Phase 1 net-failed AND
Phase 1b not adopted). **Contractual terminus of the pre-registered
momentum-gate line reached:** the linear β-residual convergence signal is a
real, PIT-clean, parameter-free GROSS pulse (~+1.3 bps/cyc) with **no
parameter-free, net-of-cost, robust executable edge on free 4h Binance perp
data**, and the cost-amortization rescue fails for a mechanistically clear
reason. Honest, well-understood negative — a property of the signal, not a
testing failure. No direction flip. Production LGBM unaffected. Script
`93_tsmom_v31_sleeve.py`; artifacts `linear_model/results/step93_tsmom_v31_sleeve/`.

### Step 94 / 94b addendum (2026-05-18): D1 information ceiling — Q1 = NO, the line is INFORMATION-BOUNDED (definitive)

Owner posed three gated questions (enough info? / fully utilized? / can we
catch the good signals?) — pre-registered as D1→D2→D3 in
`docs/INFORMATION_DIAGNOSTIC_PLAN.md`. D1 = information ceiling: best-case
extraction under a *stationarity assumption* (leak-free CV that interleaves
regimes), pre-registered gate NET Sharpe > +1.5.

**Step 94 (D1 v1) — INVALID, discarded (not ridden as a pass).** Naive
random shuffle leaked (temporal autocorrelation + contemporaneous
cross-section): LGBM IC **+0.376**, NET **+23.71** — the project's own
">0.10 IC = suspicious" rule flagged it; the Ridge→LGBM 10× explosion was
the memorization-via-leak fingerprint. Caught by sanity, design corrected.

**Step 94b (D1 v1.1, the gated test) — leak-free, FAIL → Q1 = NO.**
Time-grouped (whole-timestamp) shuffled 5-fold + 1-day embargo kills both
leaks; shuffled blocks still interleave regimes (the legitimate Q1-vs-Q2
separation). Leak check confirmed v1 was pure leakage: **LGBM IC collapsed
+0.376 → +0.010**, NET +23.71 → −0.35. Clean ceiling: **Ridge NET Sharpe
+0.62, CI [−1.68,+3.14] (straddles 0), IC +0.026; LGBM NET −0.35, IC
+0.010 (≈ pure noise).** Best F_core = +0.62 ≤ +1.5 → **FAIL.** s_t
reference +0.29 (= Step-92, harness validated). F_core+OI (context, 19 syms)
both *negative* (Ridge −2.40, LGBM −2.22) — OI features actively hurt
leak-free (consistent with Step-90 sign-inversion).

**Decisive, definitive conclusion (answers all three questions via the
gate):**
- **Q1 — enough information? NO.** Under the *most generous* leak-free
  setting (stationarity assumed), the ceiling is sub-cost: linear ≈ +0.6
  CI-zero, flexible nonlinear ≈ 0/negative. The features do not carry
  enough net-of-cost tradeable 4h information.
- **Q2 — fully utilized? Effectively YES (moot).** Realized honest results
  (Step-92 s_t +0.29; whole 76–93 arc ≈ 0) sit *just below* the leak-free
  ceiling (+0.62). The gap is ~+0.3 Sharpe and the ceiling itself is below
  cost — we were *not* leaving large signal on the table. The famous
  in-sample→nested "collapses" were the non-stationarity penalty applied to
  *leak-inflated* in-sample numbers; measured cleanly there is barely a
  ceiling to fall from.
- **Q3 — catch the good ones? Moot.** Cannot select a profitable subset
  from a signal whose *total* ceiling is sub-cost and CI-zero (independently
  evidenced: Phase-DDI R²≈0.005).

**The bottleneck is the raw information content of free 4h Binance perp
features — NOT model capacity, NOT extraction efficiency, NOT signal
selection.** This is the cleanest, strongest terminus of the line: it
*explains* the whole 76–93 arc (every apparent positive was
leak/selection/in-sample because the honest ceiling is ≈0). Caveats
(symmetric): D1 bounds the *current* feature family (price/vol/funding/OI/
dominance/beta/idio-vol — the bulk of what's been engineered), not literally
every conceivable free feature; Ridge +0.62 is a faint non-zero pulse (the
real-but-weak convergence we always saw) but CI-zero and sub-cost and the
flexible model can't even find it. **Levers with headroom = NEW orthogonal
information.** Honest scope of the D1 bound: it covers the panel family =
perp-OHLCV-derived price/vol/funding/OI/dominance/beta/idio (its volume
features `vol_zscore_4h_over_7d`, `obv_z_1d` are **perp** volume, inside the
failed ceiling). It does NOT cover two genuinely-untested **free** families:
(1) spot microstructure (spot volume, spot/perp volume divergence, spot CVD,
basis beyond the funding-carry proxy) — free via Binance Vision, not yet on
disk; (2) perp aggTrade order-flow (VPIN/taker-CVD; `features_ml/trade_flow.py`
exists, perp aggTrades on disk, NOT in this panel). Plus paid on-chain/cohort.
Q2/Q3 work on the *current* features is futile by construction; the open
question is whether spot/flow features lift the leak-free ceiling (a cheap
D1-extension), else accept the terminus.
No strategy adopted (D1–D3 are measurements). Production LGBM unaffected.
Scripts `94_info_ceiling_d1.py` (invalid), `94b_info_ceiling_d1_grouped.py`
(gated); artifacts `linear_model/results/step94b_info_ceiling_grouped/`.

### Step 95 addendum (2026-05-18): D1-ext-A — perp aggTrade order-flow — gate FAIL, but the cleanest marginal feature signal of the whole arc

D1-ext-A (pre-registered): built 6 PIT order-flow features directly from
perp aggTrades (`of_tfi_z1d`, `of_imb_4h/1d`, `of_vol_z7d`, `of_kyle_1d`,
`of_tsz_z1d`; all trailing+`.shift(1)`), cached
`outputs/vBTC_features_oflow/`. Universe = hl42∩aggTrades = **20 liquid
majors** (only ones with aggTrades; subset like OI). **PIT audit PASS**
(`of_tfi_z1d` indep strictly-past recompute corr 1.000000 SOL/ADA;
look-ahead |corr(oflow,fwd αβ)| max **0.014**, all <0.10 — genuinely
predictive-but-clean). Same leak-free CV (whole-timestamp 5-fold + 1-day
embargo) + same pre-registered **+1.5 gate**; F_core vs F_core+oflow on the
**same 20-sym rows** (apples-to-apples).
**Result — gate FAIL, but a real clean lift:**
- F_core (20-sym) baseline: Ridge NET **+0.46** (CI[−1.96,+2.94]), LGBM −1.64.
- F_core+ORDERFLOW (gated): Ridge NET **+1.09** (CI[−1.20,+3.56], IC
  +0.026, 70% syms+, 6/9 folds), LGBM −1.46. Best **+1.09 ≤ +1.5 → FAIL.**
- **Δ = +0.63** (Ridge +0.46→+1.09), IC +0.018→+0.026, syms+ 55→70%,
  folds 5→6/9. s_t ref +0.51 (= Step-92, harness re-validated).

**Symmetric read.** *Not over-claimed:* gate fails; +1.09 CI straddles 0;
LGBM is negative (only the linear model extracts it ⇒ weak linear signal);
+1.5 bar was pre-set because the non-stationarity haircut (Q2) destroys
~2–3× — +1.09 stationary leaves nothing after it. *Not over-dismissed:*
**Δ +0.63 is the largest, cleanest marginal feature-family contribution in
the entire 76→95 arc** — the first time a PIT-clean family materially moved
the leak-free ceiling the right way, improving IC *and* breadth *and* folds
together. Order-flow carries real orthogonal information the perp-OHLCV
family lacks. **This re-answers Q1 as a much closer NO and changes the
picture: the line is not information-empty — stacking orthogonal
microstructure is the productive axis.** Per the pre-registration (D1-ext-A
AND D1-ext-B are the two free families), proceed to **D1-ext-B (spot
microstructure: spot/perp volume divergence, spot CVD, basis-beyond-funding)**
and evaluate the **STACKED** ceiling F_core+orderflow+spot vs +1.5 (oflow
features carry forward, not discarded — the honest question is whether the
full free-data stack clears the bar). If the stack still ≤ +1.5 → terminus
(free data exhausted); if > +1.5 → line reopens, D2 live. No strategy
adopted. Production LGBM unaffected. Script `95_d1ext_orderflow.py`;
artifacts `linear_model/results/step95_d1ext_orderflow/`.

### Step 96 addendum (2026-05-18): D1-ext-B — spot microstructure STACKED — gate FAIL, free-data information bound is DEFINITIVE (TERMINUS)

D1-ext-B (pre-registered, the decisive free-data-stack test): downloaded
spot 5m klines (Binance Vision; egress OK) for the 20-sym universe, built
6 PIT spot features (`sp_basis_z1d`, `sp_basis_4h`, `sp_taker_imb_1d`,
`sp_volratio_z1d`, `sp_vol_z7d`, `sp_retdiff_4h` — basis dislocation, spot
CVD, spot/perp volume lead, spot-perp ret lag; all trailing+`.shift(1)`).
*Builder bug found & fixed mid-run, recorded honestly:* Binance switched
daily-kline timestamps ms→µs in 2025 → first run parsed open_time as ms →
year-57489 dates → 0 perp-merge rows → crash; fixed with unit auto-detect,
bad caches purged, rerun. **PIT audit PASS** (`sp_taker_imb_1d` indep
strictly-past recompute corr 1.000000 SOL/ADA; look-ahead |corr| <0.10).
Same leak-free CV + same +1.5 gate; F_core / +oflow / +spot / +oflow+spot
on the **same 20-sym rows**. Harness re-validated: F_core +0.46,
F_core+oflow +1.09, s_t ref +0.51 reproduce Step-95 exactly.

**Result — gate FAIL, and spot adds nothing:**
| stack | best NET Sharpe |
|---|---|
| F_core | +0.46 |
| F_core + order-flow | +1.09 |
| F_core + **spot** | **+0.33** (spot marginal ≈ **−0.13**, slightly *hurts*) |
| F_core + order-flow + spot (GATED) | **+0.91** ≤ +1.5 → **FAIL** |

Spot microstructure carries **no marginal information** beyond
F_core+order-flow (stacked +0.91 < oflow-only +1.09; spot-only +0.33 <
F_core +0.46). Mechanistically expected: `funding_rate` (in F_core)
already proxies the perp–spot basis *carry*; perp volume-z + order-flow
already capture the flow regime; spot/perp divergence at 4h on liquid
majors is dominated by the same systematic moves the BTC-β residual already
removes ⇒ spot is **redundant, not orthogonal**.

**DEFINITIVE TERMINUS.** Both pre-registered free families now tested:
order-flow = a real clean +0.63 marginal but sub-gate (Step 95); spot ≈
zero/negative marginal (Step 96). The **full free-data stack
(perp-OHLCV + order-flow + spot), measured leak-free under the most
generous stationarity assumption, does NOT clear +1.5.** So, completely
and across all free data: **Q1 = NO** (raw information content of free 4h
crypto data is the binding constraint), **Q2/Q3 moot.** This is the
honest, mechanistically-understood close of the linear β-residual line.
Symmetric note (carry forward): Step-95 order-flow's clean +0.63 is the
single genuine orthogonal-information finding of the whole arc — it says
*flow-type* data is where residual-predictive content lives, the right
direction **if** a richer/paid/orthogonal source is ever pursued. Only
remaining levers: paid on-chain/cohort data (Glassnode, the production
memory's deferred >11 cohort-Sharpe-spread bar) or a different data
domain/horizon — else the line is closed. No strategy adopted (D1–D3 are
measurements). Production LGBM unaffected. Script `96_d1ext_spot.py`;
artifacts `linear_model/results/step96_d1ext_spot/`,
`outputs/vBTC_features_spot/`.

### Step 97 addendum (2026-05-18): spot decomposition — redundancy is real at feature level, NOT a block-masking artifact

Owner Q: *why does spot add nothing — what about spot-perp volume diff?*
Step-96 used the 6-feature spot block; a block can mask one useful feature,
so isolated each spot feature (same leak-free CV, diagnostic not gated).
Univariate IC of all 6 spot feats vs fwd αβ ∈ [−0.013,+0.008] (≈0).
Single-feature marginals: F_core +0.46 → **+sp_volratio_z1d (spot/perp
volume divergence, alone) +0.47 (Δ +0.02)**; +sp_retdiff_4h +0.38
(Δ −0.07); both-divergence +0.38 (Δ −0.08); F_core+oflow +1.09 →
+sp_volratio_z1d +1.12 (Δ +0.03). CI literally unchanged
([−1.96,+2.94]→[−1.96,+2.95]). **Verdict: the spot-perp volume-divergence
feature carries no marginal information even isolated — "spot adds nothing"
is NOT a block-dilution artifact.** Mechanism now evidence-backed: (a)
spot↔perp lead-lag is a seconds-minutes effect, fully arbitraged before a
4h forward window (return-divergence *hurts*, −0.07 — the decay
fingerprint); (b) spot/perp volume ~0.8-corr ⇒ flow regime already in
F_core+order-flow; (c) basis ≈ `funding_rate` already in F_core (solo
basis ceilings ≈0/neg). Contrast: trade-level aggressor flow = clean +0.63
(Step 95) vs spot price/volume/basis ≈0 — residual-predictive content
lives in *flow*, not in spot price/volume/basis at 4h. Confirms the
Step-96 terminus. Production LGBM unaffected. Script `97_spot_decomp.py`;
artifacts `linear_model/results/step97_spot_decomp/`.

### Step 98 addendum (2026-05-18): D1-ext-C — perp-vs-spot flow divergence + interactions — gate FAIL, block net-destructive

Owner Q: spot *flow* vs perp *flow*, more interactions. Legitimate
(follows the one real signal — flow +0.63) so given a rigorous
pre-registered one-shot test (LOCKED 6-feature block, no sweep, same
leak-free CV + same +1.5 gate, anti-block-masking solo checks). Block:
`fd_imb`=of_imb_1d−sp_taker_imb_1d, `fd_absdiff`, `fd_prod`, `x_flow_vol`,
`x_flow_fund`, `x_fd_st`=fd_imb·s_t. Univariate IC all ∈[−0.016,+0.015]
(≈0). **Result — FAIL & net-destructive:** F_core +0.46 → F_core+oflow
**+1.09** → F_core+oflow+FLOWINT **+0.44** (Δ **−0.65** — the block *erases*
the order-flow lift via variance inflation). Anti-block-masking: solo
`fd_imb` Δ +0.03, solo `x_fd_st` Δ −0.07 — the core "leverage-led deviation
reverts harder" feature individually carries nothing (not a masking
artifact). LGBM negative throughout (no stationary interaction structure —
consistent with LGBM's own auto-interaction search finding nothing).
**Verdict: perp-vs-spot flow divergence + targeted flow×regime/×deviation
interactions add no ceiling information at 4h.** Honest residual: spot flow
here is spot-*kline* taker (5m aggregate), coarser than the perp *aggTrades*
of Step 95 — so the single remaining untested free probe is
**aggTrade-granularity spot order-flow** (download spot aggTrades, build
true spot VPIN/kyle/tfi à la Step 95, re-test perp-vs-spot at matched
granularity). Heavy (Step-95-scale), prior now guarded-low (everything
beyond perp order-flow ≈0), but legitimate and the last free stone. The
free-data terminus stands pending that one probe; order-flow's clean +0.63
remains the sole genuine orthogonal-information finding of the 76→98 arc.
No strategy adopted. Production LGBM unaffected. Script
`98_flow_interactions.py`; artifacts
`linear_model/results/step98_flow_interactions/`.

### Step 99 addendum (2026-05-18): D1-ext-D — spot-led MOMENTUM-vs-reversion classifier — ABSENT (closes from the opposite polarity)

Owner reframed the spot-volume intuition as a *momentum/continuation*
idea (spot-led = durable move) — distinct from this line's reversion
thesis, never tested on its own terms. LOCKED pre-reg one-shot: bucket by
`sp_volratio_z1d` (spot/perp vol-lead) q0..q4 + extreme deciles;
continuation coef = Spearman(trailing-24h raw move, forward) at 4h/24h/72h
raw + β-resid@4h; q4−q0 spread block-bootstrap CI (block≥horizon). PIT
sanity OK (conditioner corr w/ fwd +0.002). **Result — ABSENT (all 3
pre-reg conditions fail):** continuation coef **negative in every bucket
at every horizon** (−0.03 to −0.11 — everything mildly *reverts*, incl.
spot-led q4); monotonicity ρ = −0.20/+0.10/−0.60 (not ↑); q4−q0 spreads
−3/+4/−12 bps, **all CIs straddle 0**; β-resid@4h same (all buckets
revert). Trailing move ↔ fwd-4h corr = −0.047 pooled (reversion is the
*only* structure, regime-invariant to spot/perp vol). **Transparent
nuance (not buried):** the *single* directionally-consistent cell is the
24h extreme-decile contrast D9(spot-led) +7 vs D0(perp-led) −12 (Δ +19
bps) — but it's one cell of a 2×3 grid, not significant (the 24h quintile
spread CI was [−11,+20]), flanked by 4h Δ≈0 and 72h Δ≈−1; the
pre-registered monotonicity+CI gate correctly rejects it as the
K2/K3-style single-cell artifact, NOT a basis to continue. **Verdict:
the "spot volume outperforms ⇒ durable move" intuition is definitively
not present in Binance free data at 4h/24h/72h, on its own momentum
terms.** This closes the free-data question from *both* polarities:
reversion edge = real-but-sub-cost (92–93); momentum framing = absent
(99); no free feature lifts the leak-free ceiling (94b–98). Nothing in
free data classifies persist-vs-revert — the mechanistic reason the
momentum-gate line and Phase-DDI conditioning always failed. Symmetric
discipline: owner's reframed hypothesis was legitimate and given a
rigorous test on its own terms (not pre-dismissed); the one tempting cell
reported in full and rejected by the pre-registered arbiter (not
goalpost-moved into a false positive). No strategy adopted. Production
LGBM unaffected. Script `99_spot_momentum.py`; artifacts
`linear_model/results/step99_spot_momentum/`.

### Step 100 addendum (2026-05-18): D1-ext-E — structural-EVENT paradigm — events NOT predictable; comprehensive terminus

Owner's best/most-mechanistic hypothesis: reduced-form feature→return is
information-bounded, but a *structural* composites→event(crowding/
liquidation)→return paradigm is untested, and the β-hedge may strip what
events produce — so test hedged AND unhedged. LOCKED pre-reg (D1-ext-E):
6 PIT crowding composites (OI×price, OI×funding, OI-vs-own-history,
price/OI quadrant, taker-LS, OI-accel) from cached OI panel; 3 fixed
forward-24h events (E1 long-liq r≤−2σ, E2 squeeze r≥+2σ, E3 vol/deleverage
range≥p90); Stage-1 leak-free grouped+embargo classifier (logistic+LGBM)
OOF AUC Bonferroni×3; Stage-2 hedged-vs-unhedged fork instrumented
(residual gate +1.5; raw vs market-exposure-matched placebo). *Process
note (honest):* first run killed at 34 min — a perf bug in my AUC
bootstrap (O(timestamps×rows×iters)); fixed perf-only (identical AUC +
timestamp-block-bootstrap math, pre-registration unchanged), rerun = 58s.
19 syms / 29.6k rows; PIT sanity 0.021; event base rates E1/E2 3.0%,
E3 10.5% (well-formed). **Result — Stage 1 FAILS for all 3:** E1 AUC
**0.481** CI[0.433,0.540]; E2 **0.502** CI[0.451,0.552]; E3 **0.506**
CI[0.486,0.526]. All ≈ 0.50 coin-flip, none Bonferroni-significant.
Stage 2 correctly never triggered (pre-registered gate worked).
**Verdict: crowding/forced-deleveraging events are NOT PIT-predictable
from free OI/price/funding composites.** Crucially the failure is at
Stage 1, *before* the hedged-vs-unhedged fork — so it is **not** "the
hedge strips it"; it is deeper: crowding tells you the *fuel* exists, not
*when/whether* it ignites — the cascade trigger is an exogenous price
shock, by construction absent from PIT crowding features (AUC≈0.5 is the
*expected* result for predicting a cascade from crowding alone). **This
generalizes the terminus across BOTH paradigms: reduced-form feature→
return information-bounded (94b–99) AND structural composites→event→return
not even event-predictable (100), hedged and unhedged-instrumented.** The
single best remaining hypothesis, tested rigorously on its own terms,
failed at the first gate — no target/encoding ambiguity remains. Symmetric:
owner's excellent idea given a ffull rigorous structural test (not
pre-dismissed); clean decisive negative with mechanism understood (not
over-claimed). Free-data linear β-residual line is now comprehensively
closed. Only levers: paid/orthogonal data, or accept closure. No strategy
adopted. Production LGBM unaffected. Script `100_oi_events.py`; artifacts
`linear_model/results/step100_oi_events/`.

### Step 101 addendum (2026-05-18): D1-ext-F — MAXIMAL feature set → structural events — still AUC≈0.50; closes the "OI-only" loose end (airtight terminus)

Owner critique (fair): Step 100 was OI-anchored — "OI alone may not take
effect, plenty of other composites." Honest prior stated up front: D1 (94b)
already bounded the full F_core stack for the forward *return*, and E1/E2
are 2σ thresholds of that same return ⇒ strong prior AUC≈0.50; but
full-features→event-classification was a genuinely untested cell, so the
decisive maximal test was warranted (and removes the loose end). LOCKED
D1-ext-F: feature set = strict SUPERSET of Step 100 = **47 PIT features**
(F_core ~24 panel + s_t + FULL OI panel 11 + order-flow panel 6 + the 6
Step-100 composites); events = the 3 LOCKED Step-100 defs UNCHANGED
(anti-p-hack); no new hand-crafted composites (LGBM auto-combines); same
leak-free CV + logistic+LGBM + OOF AUC + timestamp-block bootstrap +
Bonferroni×3 + hedged/unhedged Stage-2. 19 syms (OI∩oflow∩panel∩klines),
29.5k rows; PIT sanity max|corr(feat,resid_fwd24)|=0.090 (corr_to_btc_1d,
a pre-audited panel feat) none>0.15; base rates E1/E2 3.0% E3 10.5%.
**Result — Stage 1 still FAILS all 3:** E1 AUC **0.508** CI[0.470,0.554];
E2 **0.489** CI[0.446,0.528]; E3 **0.502** CI[0.476,0.530]. All ≈0.50
coin-flip, none Bonferroni-sig; Stage 2 never triggered. **Verdict:
structural forced-deleveraging events are NOT predictable from ANY free
feature combination — not OI alone, not the maximal stack incl. the +0.63
order-flow family, with LGBM auto-constructing all interactions.** The
"OI-only" objection is decisively closed: it was never that OI was
insufficient — the cascade *trigger* is an exogenous price shock absent
from every PIT feature (Step-100 mechanism confirmed at full feature
breadth). **The free-data terminus is now AIRTIGHT: both paradigms
(reduced-form feature→return 94b–99 AND structural composites→event→return
100–101), maximal feature set, both polarities, hedged AND
unhedged-instrumented. No distinct free-data test remains** — further
free-data variants would be goalpost-moving. Symmetric: owner critique was
valid and given the principled maximal test (not pre-dismissed); clean
decisive negative, mechanism reconfirmed at breadth (not over-claimed).
Only levers: paid/orthogonal data (the lone real signal = order-flow
+0.63, flow-type) or accept closure. No strategy adopted (pure
measurement). Production LGBM unaffected. Script
`101_oi_events_fullfeat.py`; artifacts
`linear_model/results/step101_oi_events_fullfeat/`.

### Step 102 addendum (2026-05-18): D1-ext-G — LONG vs SHORT composite-consensus sign-reliability — hypothesis falsified (more agreement → WORSE)

Owner request: combine all long composites → check the sign; all short →
check the sign; does more agreement give a more reliable direction. LOCKED
parameter-free signed set (14 composites: OI/funding/price/volume/vol/
cross-asset), net vote V, target = panel alpha_beta (canonical fwd-4h
β-resid). 19 syms, 29.6k rows. **Result — directly falsified:**
- LONG-consensus (V>0, n=46%): mean +1.92 bps CI[−0.19,+4.02], **hit 50.3%
  (coin-flip)**, net Sharpe +0.62, 5/9 folds.
- SHORT-consensus (V<0, n=46%): mean −0.27 bps, hit 53.4%, **net −0.09**,
  4/9 folds. (Short>long hit echoes production DDI-2 long≈below/short>random,
  but net-of-cost it's ≈0.)
- **|V|-monotonicity ρ = −0.60 (NEGATIVE):** |V|=2→+1.7, |V|=3→+4.0, but
  |V|=4→−4.3, |V|=5→−9.0 bps. **More composites agreeing → MORE NEGATIVE**
  signed payoff — the exact opposite of the hypothesis. High-agreement =
  extreme-crowding/trend states where the reversion sign inverts
  (non-stationarity, exactly where reliability was expected).
- Fails matched placebo: real +0.84 vs random-per-symbol-sign p95 +1.24.

**Disclosed spec flaw (honest):** the locked "1σ" cutoff Z=1 was correct
for z-scored features (funding_z, etc.) but mis-scaled for raw-return-scale
`s_t` ⇒ the 4 explicitly-convergence composites gated on |s_t| (P1/V4/R1/X1)
fired ~0%. **Impact assessment: does NOT change the verdict** — those 4 are
all `−sign(s_t)` = the Step-92 convergence signal already bound net ≈0
sub-cost (+0.25, CI crosses 0); including them properly only re-weights V
toward a known-sub-cost direction, cannot manufacture reliability. The
falsification (long-hit 50.3%, anti-monotone |V|, placebo-fail) comes
independently from the diverse composites that DID fire (P2 67%, O5 31%,
V2 39%, O4 24%, V3 23%, V1 14%). A scaling-corrected rerun is confirmatory,
not decision-changing (offered, not auto-run — rigor-anchor: don't burn a
cycle re-confirming a robust null). **Conclusion: signed-composite
consensus does NOT yield a reliable net-of-cost direction; more agreement
makes it worse — empirically confirms the composites are correlated reads
of one sub-cost convergence signal, not independent directional bets
(verify-not-assert: the structural argument now demonstrated, not just
asserted).** Symmetric: owner's specific repeated hypothesis given the
exact decomposition test it asked for; cleanly falsified with the
mechanism (anti-monotonicity) explained; my spec flaw disclosed +
impact-assessed, not buried. Free-data signed-consensus framing closed. No
strategy adopted. Production LGBM unaffected. Script
`102_composite_consensus.py`; artifacts
`linear_model/results/step102_composite_consensus/`.

---

## ⚠️ KILL OVERTURNED (2026-05-17, Step 76) — SUPERSEDED by Step 77 above (rank-IC real but economically dead)

Supersedes the Step-75 kill block immediately below. The Step-75 verdict was a
**proxy-based** stop (pooled MSE-Ridge + IC-magnitude persistence). User
correctly judged it left a gap — Step-75's *sign*-persistence sub-condition had
in fact **passed** (top-10 features all sign-stable ≥7/9 folds) — and approved
one direct test of the simplest signed composite. `76_minimal_orientation.py`
(590s, no backtest, IC gate only, pre-registered IC ≥ +0.02 & t ≥ 3.0 fixed
before run):

| part | composite | OOS cycle-IC | t | folds+ | top-bottom | cheap IC gate | full Phase-6 gate |
|---|---|---|---|---|---|---|---|
| A — 4h `alpha_beta` | signed_all_shrunk_ic_weighted | **+0.0517** | **+8.46** | **9/9** (0.027–0.085) | **−2.03 bps** | PASS | **FAIL** (spread<0) |
| A — 4h (sibling) | signed-**equal** | +0.0505 | +9.41 | 9/9 | — | — | — |
| B — proper 24h resid | ~~signed_all_shrunk_ic_weighted~~ | ~~+0.0575~~ | ~~+3.36~~ | ~~4/8~~ | ~~+19.34~~ | **INVALID** | **INVALID** |

**B_24h_proper is INVALID — discarded (user-caught bug, 2026-05-17).**
`76_minimal_orientation.py:191` `build_24h_target` does
`groupby("symbol")["return_pct"].shift(-j * BLOCK)` on the **already-4h-sampled**
`dec` frame, where one row = 4h, so it shifts ~8 days/block (should be
`shift(-j)`); it also reuses 4h `beta_btc_pit` instead of rebuilding β at the
24h horizon. The 24h question is **unresolved**, not failed. Step-75's
analogous shift was on the raw 5m panel (48 bars = 4h) so correct there, but
its 24h was a non-load-bearing proxy.

**A_4h passes the cheap IC falsifier but FAILS the full plan gate.** The
Phase-2/Phase-6 pre-registered gate also requires top-bottom spread ≥ 9 bps net
and ≥6/9 folds robust; A_4h has a **negative** −2.03 bps extreme-decile spread
→ it does **not** clear the tradeability gate. It is a confirmed rank-IC lever,
**not** a gate-passing signal.

**What is established:** a fold-local signed composite of the same 22 V2
features recovers a **strong, fold-stable cross-sectional rank-IC** (+0.052,
t 8.5, every one of 9 folds positive, equal-weight ≈ shrunk-weight → robust to
the noisy magnitude) that the pooled MSE-Ridge *inverted* (Step 75 −0.011) and
the production Ridge diluted to ~0. The plan's orientation hypothesis is
**vindicated at the rank-IC level.** The Step-75 "no extractable linear signal
/ line STOPPED" verdict is **WRONG and retracted** — it measured an MSE fit and
IC-magnitude noise, not the sign-aggregation channel that actually carries.

**What is NOT established (do not overclaim):**
1. **Not tradeable as-is.** Part A has positive rank-IC but **negative
   extreme-decile spread** (−2.03 bps): the ordering is right mid-rank, the
   3-name tails you'd trade carry negative raw residual (heavy-tail target).
   Rank-IC ≠ monetizable basket.
2. **24h is fragile.** Part B passes only via the noisy IC-weighting (equal
   sibling fails t 2.19), is fold-concentrated (f2-5 carry, recent f6-8
   negative, f9 dropped) — the recurring K2/K3/W23 pattern. Treat as
   unconfirmed.
3. **Not yet audited.** A +0.052/t8.5 jump on features a multivariate fit
   found anti-predictive demands an independent PIT/leakage audit of the
   orientation pipeline (train-slice purge, per-cycle cross-sectional-z, IC
   fit) **before** this reversal is trusted — the project's decisive Step-72
   "audit before battery" lesson applies directly.

**Status: Phase-1.5 kill SUSPENDED, orientation REOPENED as a real lever; NOT
a strategy.** Next per plan = Phase-6-style gate, not a backtest: (a)
independent leakage/PIT audit of Step 76; (b) resolve the IC>0 / extreme-
spread<0 contradiction (decile/monotone-band analysis). No optimization, no
backtest until both clear. Production stays LGBM regardless. Artifacts:
`linear_model/results/step76_minimal_orientation/` + `.log`.

---

## ⚠️ KILL RECORDED (2026-05-17, Step 75 / Phase 1.5) — RETRACTED, see block above

After the Step 72/74 engine timing-bug fix
(headline +4.34 → causal **−0.57**, see prior verdict), `NEXT_PLAN.md`
mandated a pre-registered Phase 1.5 signal probe *before* any orientation /
composite / feature work. It was run (`75_signal_probe.py`, 919s) on the
current drop-BIO+VVV HL≥2M testbed (42 syms, V2 22 features, causal folds).
**Both pre-registered gates FAIL → kill criterion triggered.**

| gate | observed | pre-registered bar | verdict |
|---|---|---|---|
| Feature-IC sign persistence | sign sub-cond **10/10** persist ≥7/9 (8 at 9/9); mean `rho(fₖ→fₖ₊₁)` **+0.049** | ≥8/10 persist AND mean rho ≥ +0.20 | **FAIL** (on rho only) |
| Multivariate reference (pooled MSE-Ridge, any horizon) | 4h IC **−0.0113** t −2.27; 8h −0.0065; 12h −0.0052; 24h +0.0055 t +0.45 | OOS cycle-IC mean ≥ +0.02 AND t ≥ 3.0 | **FAIL** at all 4 horizons |

**Recorded conclusion (per Kill Criterion):** *free-data 4h linear β-residual
has no extractable persistent cross-sectional alpha.* Phase 2/3 (orientation /
model fix) is **skipped** per the plan's both-fail decision branch. Phase 4
feature batches are **not unlocked** — their precondition is "sign-persistence
PASSES but reference is tiny"; sign-persistence FAILED on rho, so the
pre-registered logic is STOP, not feature expansion.

**Structural finding (recorded for method reuse, not goalpost-moving):** the
failure is *not* "features are noise." Every strong feature's IC *sign* is
rock-stable (dom_btc_z_1d −0.046, atr_pct −0.041, idio_vol_to_btc_1h −0.041,
all 9/9 folds same sign). Two things kill it: (a) per-fold IC *magnitude* is
uncorrelated noise (mean rho +0.05), so no fold-adaptive IC weighting
generalizes; (b) the pooled MSE-Ridge is *significantly anti-predictive* at 4h
(t −2.27) — the multivariate fit inverts the weak sign-stable univariate
signal. Longer horizons drift toward IC≈0 (24h +0.0055, not significant; the
+32 bps top-bottom there is just larger residual scale, not signal). This
corroborates the long-standing project finding ("model = noisy magnitude
generator; the apparent edge was the architecture wrapper") — and post the
look-ahead fix, with the wrapper's apparent edge gone (causal −0.57), nothing
remains. Caveats (do not change verdict): horizon targets are the cheap
additive non-overlapping 4h-sum proxy the plan authorized; 24h t-stat uses
n=270 non-overlapping cycles. IC≈0 at every horizon, nowhere near +0.02/t≥3.

**Next:** stop, or change the input (on-chain / Glassnode / different horizon /
different universe). Do not open a step-77+ feature-tuning loop on this data.
Production stays LGBM (unaffected — different PIT-clean engine). Artifacts:
`linear_model/results/step75_signal_probe/{verdict,horizon_reference,multivariate_reference,feature_ic_persistence}.csv`,
`linear_model/results/step75_signal_probe.log`.

---

## ⚠️ FINAL VERDICT (2026-05-14 post Steps 55-57): STRATEGY NOT PRODUCTION-VIABLE

**The entire +3.11 edge is two un-executable meme coins (SIREN + JELLYJELLY).** Triangulated from three independent angles, all agreeing:

| evidence | result |
|---|---|
| Per-symbol attribution | SIREN +14,061 bps + JELLYJELLY +10,542 bps = +24,603; **all other 108 symbols net −630 bps** |
| HL inference-restriction (train 110, trade 70 HL syms) | +3.11 → **−0.17**, both placebos FAIL (P2 = median of random); vol-filtered 50-sym → P2 −4.19 |
| HL-native retrain (train+validate from scratch on 70 HL syms) | **−1.29 Sharpe, gross −1.47/cyc (NEGATIVE)** |

Both SIREN and JELLYJELLY are absent from Hyperliquid (JELLYJELLY was force-delisted by HL after the March 2025 manipulation incident; SIREN never listed). Every prior positive validation (full-PIT, causal accounting, P1/P2/wrapper-symmetric placebos at p100, cost-robustness) was technically valid but measured *"does the machinery capture the SIREN/JELLYJELLY explosions better than random"* — which it does — **not** *"is there a diversified, transportable edge."* There is not.

**This is not a tuning/anchoring artifact**: a model trained *specifically and only* on the executable Hyperliquid universe produces *negative* Sharpe (−1.29) with negative gross. No exploitable 4h β-residual α exists on the liquid HL universe. All four symptoms — HL non-transport, fold concentration (folds 4+5 = the meme explosion periods), composition fragility (drop wrong 2 → collapse), drop-2-kills-it — are the same single fact.

The in-sample validation work + methodology (PIT discipline, gate-consistent & wrapper-symmetric placebos, causal aggregator, cost analytics) is sound and reusable. The **strategy as constructed is dead** for Hyperliquid execution. Retained below for the record and method reuse.

---

## TL;DR — in-sample validation state (Binance 110-panel, NOT executable as-is)

| state | meaning |
|---|---|
| **V2 + V3.1 sleeve = Sharpe +2.19 on 51-panel** (Step 34/35) | PASSES both P1 (p99) and P2 (p97) placebos at p95. First placebo-validated linear in codebase. NOT yet rebuilt with full-PIT for parity. |
| **V2 on 110-panel FULL-PIT + CAUSAL (Step 50/51, FINAL): Sharpe +3.11** | CI [+1.06, +4.92] strictly positive. P1 p100 edge **+1.45**, P2 p100 edge **+1.61** — both placebos pass under correct causal accounting. 7/9 folds+ (fold 3 flipped +0.08; fold 1 still negative −4.39). Step 51 K-drop stress (causal): K=10 +1.26, K=20 +0.81, K=30 +0.97, K=40 +1.02 — survives sign at every K but far below baseline (composition-fragile). |
| V2 on 110-panel FULL-PIT lagged (Step 47+48, superseded by Step 50) | Lagged aggregator gave +3.35 / P1 +1.69 / P2 +1.81; causal correction uniform −0.2 across all metrics. |
| V2 on 110-panel strict-PIT (BTC-frame only, Step 44+45, superseded) | +2.11, P1 p97 (+0.96), P2 p99 (+0.66). Missing the base-OHLCV-shift. |
| V2 on 110-panel non-strict-PIT (Step 41+42, DEPRECATED) | +2.03, P2 FAIL p92 (−0.13), K=10 catastrophic (−1.24). Multiple PIT bugs since fixed. |
| **LGBM Phase UNI-111 = −1.48 (memory)** | NOT apples-to-apples (different cost convention, target convention, dates, no PIT-corrected LGBM rerun). Treat as contextual reference only, not headline proof of V2 advantage. |
| **Raw 4h cycle (no sleeve) Sharpe = −2.55** with correct 9 bps cost (Step 36) | Linear has near-zero gross signal per cycle. Sleeve adds +4.74 via cost amortization + 24h hold extension. |
| **conv_gate contribution +0.34** for B_IC_signed (Step 39) | trail_ic per-symbol wrapper already does most of conv_gate's filtering work. |
| **24h target hurts, aligned features hurt more** (Steps 37/38) | Phase AH's "predict short, hold long via sleeve" lesson holds for linear. The V2 feature+target combo is locally optimal. |

**Bottom line:** V2 + V3.1 sleeve is **placebo-validated on BOTH 51-panel AND 110-panel** with FULL-PIT discipline AND causal PnL accounting. Major gates summary:

| gate | 110-panel full-PIT causal (Step 50, FINAL) | status |
|---|---|---|
| Sharpe (B_IC_signed) | **+3.11** | strong |
| CI | [+1.06, +4.92] | strictly positive |
| folds+ | 7/9 (fold 3 flipped +0.08; fold 1 still −4.39) | strong but folds 4+5 dominate (Sharpe +7.73, +11.23) |
| P1 placebo (broad-liq random, gate-consistent) | p100, edge **+1.45** | PASS |
| P2 placebo (within-univ random, gate-consistent) | p100, edge **+1.61** | PASS |
| **Wrapper-symmetric placebo (Step 52, cleanest test)** | **p100, edge +1.37 over p95** | **PASS — model adds value beyond all wrapper machinery** |
| Cost sensitivity (Step 54, analytic) | HL taker 4.5 one-way → +2.97; breakeven ~53 bps one-way (12× HL fee) | NOT a binding risk — huge margin |
| K-drop universe stress (Step 51, causal) | K=10 mean +1.26 (29/30 +), K=20-40 means +0.81 to +1.02 (23-26/30 +) | survives sign but well below baseline +3.11 |

Caveats to read alongside:
- **Wrapper-symmetric placebo (Step 52) PASSES** at p100 edge +1.37 — the cleanest model-vs-random test (placebo gets identical select_refill/picks_hist/gate/PM/sleeve). Reviewer's biggest concern resolved.
- **K-drop means (+0.66 to +1.24) are far below baseline +3.11.** "Robust to symbol drops" means stays positive; the headline Sharpe DOES NOT survive random subset.
- **Fold concentration is real**: folds 4+5 contribute the bulk; fold 1 strongly negative (−4.39); fold 6 also negative (−0.96). The +3.11 mean is built from a few strong folds rather than uniformly across all 9.
- **Placebo wrapper is gate-consistent but NOT wrapper-symmetric**: real uses `select_refill()` + `picks_hist`; placebos use random candidates without that feedback loop. Step 50 still passes by wide margin, but this remains the cleanest unfinished validation.
- **Entry convention** = "decide at close of bar t, enter at close[t]". Results are valid ONLY under this convention. Switching to "trade at open_time t" would require target rebuild and would not produce these numbers.

Lagged (Step 47/48) gave +3.35 / P1 +1.69 / P2 +1.81 — causal correction is uniform −0.2 across all metrics. Both placebos still pass at p100 even under corrected accounting.

Each PIT leak fix improved everything (non-strict +2.03 → strict-PIT +2.11 → full-PIT lagged +3.35 → full-PIT causal +3.11). Leaks appear to have been ADDING NOISE rather than signal. The Step 47 full-PIT shift (return_1d/atr_pct/vwap_slope_96/bars_since_high `.shift(1)`) coincided with resolving BOTH the within-universe P2 failure AND the K=10 composition-fragility — strong evidence the OHLCV leak was a common upstream factor, but not formal proof (could be partial coincidence with other simultaneous fixes; would need feature-by-feature ablation to fully isolate).

**Open issues** (none invalidating the result):
1. **Placebo wrapper picks_hist asymmetry** in `phase_ah_sleeve.py:216`: real uses refill+picks_hist; placebos don't. Step 48/50 are gate-consistent but not wrapper-symmetric. Lower priority given p100 + wide edges.
2. **51-panel not rebuilt with full-PIT for parity**. The 51-panel +2.19 used the older non-shifted base feature convention; under full-PIT it may shift (likely upward by analogy with 110-panel). Worth doing for consistency.
3. **1000-seed placebos** could tighten p100 claim — p100 over 100 seeds means ~+0.7-1.0 standard error on the tail percentile.
4. **Paired CI vs LGBM** requires LGBM rerun on same 110-panel with same conventions before any "V2 beats LGBM" claim.

Comparison to LGBM Phase UNI-111 (−1.48) remains contextual not apples-to-apples.

---

## Active experiments

### Step 34 — V1+V2 NaN-fix + re-standardization (COMPLETE)

Fixes reviewer-confirmed bugs in Step 32:

1. **NaN bug** — `np.searchsorted` was mapping NaN inputs to max rank (+0.5). 39,902 NaNs in `funding_rate_z_7d` and 14,400 in `funding_rate_1d_change` were all silently mapped to +0.5, corrupting V1's rank features. Fixed: mask finite values before ranking; NaN positions → 0.
2. **Scale mismatch** — rank features had std 0.23–0.30 vs z-score features at 1.0 and squared at 1.4–2.1. Ridge regularization pressure was non-uniform across feature families. Fixed: re-z-score rank columns using fold-0 train stats.
3. **Auditable artifacts** — saves per-cycle PnL, predictions, fold records, LOFO CSVs to `results/step34_v1_fixed/`.

Results: V0 unchanged (+0.67), V1 −0.10 (+1.21), V2 +1.86 (+2.19). See result ledger below.

### Step 35 — P1 + P2 placebos on fixed V1 and V2 (COMPLETE)

100 seeds × P1 + 100 seeds × P2 × 2 variants = 400 placebo runs with **gate-consistent methodology** (real and placebo both gate on `pred_B`). Results in `results/step35_verdict.csv`.

| variant | Sharpe | P1 verdict | P2 verdict |
|---|---|---|---|
| V1 fixed | +1.21 | FAIL p89 (p95=+1.53) edge −0.32 | FAIL p90 (p95=+1.65) edge −0.44 |
| **V2 fixed** | **+2.19** | **PASS p99 (p95=+1.16) edge +1.04** | **PASS p97 (p95=+1.61) edge +0.58** |

**Methodology note**: Step 33's earlier P1 placebo for V1 pre-fix gave p95 = +1.19 (V1 +1.31 passed). Step 35's same P1 placebo for V1 fixed gives p95 = +1.53 (V1 +1.21 fails). The difference is gate consistency: Step 33 gated placebos on `pred_z` while real used `pred_B`; Step 35 gates both on `pred_B`. The gate-consistent placebo is wider and harder — it's the methodologically correct test. Under it, V1 fixed fails but V2 fixed passes by a clear margin.

### Step 56 — HL-native retrain (COMPLETE — confirms not salvageable)

Reviewer/user question: Step 55 only restricted execution of the 110-trained model. Fair test = retrain from scratch on the HL universe. Cloned the Step 47 pipeline restricted to 70 HL symbols, recomputed σ_idio + preprocessing + rolling-IC universe HL-native, retrained 5-seed Ridge, ran causal aggregator.

| approach | Sharpe | gross/cyc | folds+ |
|---|---|---|---|
| Full 110-panel (validated) | +3.11 | +15.42 | 7/9 |
| Inference-restriction (Step 55) | −0.17 | — | — |
| **HL-native retrain (Step 56)** | **−1.29** | **−1.47 (NEGATIVE)** | 4/9 |

Retraining HL-native is *worse* than inference-restriction. Gross is negative — the model's picks lose money before costs on the liquid universe. Definitive: there is no exploitable 4h β-residual α on the HL-executable universe even with a fair purpose-built model. Not an anchoring artifact.

### Step 57 — Drop only SIREN+JELLYJELLY (COMPLETE — cleanest confirmation)

Drops *only* SIREN+JELLYJELLY, keeps all other 108 (NO HL restriction). Isolates "is it literally these 2 tokens" from the HL question.

| | result |
|---|---|
| Sharpe (causal) | **−0.89** (was +3.11 with all 110) |
| P1 placebo | FAIL, edge −2.25 |
| P2 placebo | FAIL, edge −2.48, **rank 29% (worse than random)** |

Removing 2 of 110 symbols collapses +3.11 → −0.89 with the model below the median of random picks. The cleanest possible proof the entire edge is those 2 meme coins. Confirms the attribution (other 108 net −630 bps). Four independent tests now agree (attribution, drop-2, HL inference-restriction, HL-native retrain) — the strategy is a 2-position bet on un-executable meme coins.

### Step 55 — Binance→Hyperliquid executable-universe transport (COMPLETE — DECISIVE NEGATIVE)

The strategy is validated on Binance USDM perps but executes on Hyperliquid (Binance-train / HL-execute design). Mapped the 110 panel symbols to the HL roster (`outputs/vBTC_check_universe/all_hyperliquid.csv`, 183 HL perps; mapping saved `panel110_hl_map.csv`).

**Only 70 of 110 panel symbols are on Hyperliquid.** 40 absent — almost entirely the illiquid meme/AI/recent-launch tail (BROCCOLI*, BANANAS31, AI16Z, ZEREBRO, SWARMS, ALCH, ARC, …). HL liquidity thin even on the 70: 21 with ≥$10M daily vol, 30 ≥$5M, 50 ≥$1M.

Re-ran Step 50 machinery (full-PIT, causal, pred_z universe, P1/P2 placebos) restricted to HL-executable subsets:

| universe | n syms | Sharpe causal | P1 | P2 |
|---|---|---|---|---|
| Full 110 (Binance, validated) | 110 | +3.11 | PASS p100 (+1.45) | PASS p100 (+1.61) |
| **HL-executable (on-HL)** | **70** | **−0.17** | **FAIL (−1.59)** | **FAIL (−1.46), rank 53%** |
| HL vol ≥ $1M | 50 | (running) | | |
| HL vol ≥ $5M | 30 | (running) | | |

**Verdict**: the +3.11 alpha was concentrated in the 40 illiquid symbols that don't exist on Hyperliquid. On the executable universe the model has ZERO edge (−0.17, P2 at the median of random picks). Far worse than Step 51's *random* K=40 drop (+1.02) — the targeted removal of the illiquid tail is much more damaging, direct evidence the alpha lived disproportionately in the non-tradeable names.

Mechanistically expected in hindsight: illiquid meme/AI tokens have the largest idiosyncratic moves (the β-residual α the model targets), and those are exactly what HL won't list. The strategy structurally requires the un-executable part of the universe.

**This is a hard production blocker.** Validating HL transport before forward/paper testing was the correct ordering — it caught the binding failure cheaply (~1h) instead of after weeks of paper trading.

### Step 54 — Cost sensitivity (analytic, from Step 50 saved per-cycle) (COMPLETE)

No re-aggregation needed — gross PnL and turnover are cost-independent and saved in `step50_causal_full_pit/per_cycle_real_causal.csv`. Net Sharpe recomputed analytically at each one-way cost rate (= COST_PER_UNIT_ABS_DELTA = cost per unit |Δweight|).

Execution is on **Hyperliquid** (Binance-train / HL-execute). HL base-tier taker = 4.5 bps one-way (no volume tier / HYPE stake / referral).

| one-way cost rate | interpretation | Sharpe | folds+ | mean net/cyc |
|---|---|---|---|---|
| 0.00 | zero cost (gross only) | +3.24 | 7/9 | +15.42 |
| 2.25 | current code assumption | +3.11 | 7/9 | +14.77 |
| **4.50** | **HL taker fees only (realistic floor)** | **+2.97** | **6/9** | +14.12 |
| 6.50 | HL taker + ~2 bps slippage | +2.85 | 5/9 | +13.54 |
| 9.00 | HL taker + thin-alt slippage | +2.70 | 5/9 | +12.82 |
| 13.0 | pessimistic alt execution | +2.45 | 5/9 | +11.66 |
| 18.0 | very pessimistic | +2.15 | 5/9 | +10.22 |
| **53.4** | **breakeven (mean net → 0)** | **0.00** | — | 0.00 |

**Cost is NOT the binding risk.** At realistic HL taker cost (4.5 one-way) Sharpe = +2.97 (−0.14 vs current assumption). Breakeven is ~53 bps one-way ≈ 12× the HL fee. The sleeve's cost-amortization keeps turnover at only 0.29/cycle vs +15.42 gross/cycle, so cost is ~0.65 bps/cycle — structurally negligible with huge margin, not a fragile assumption.

Caveats: (1) per-fold consistency degrades with cost (7/9 → 6/9 → 5/9 as cost rises) — aggregate robust but marginal folds flip; (2) fold 4+5 concentration unchanged; (3) analytic estimate holds strategy behavior fixed (no cost-aware gate re-opt — which would only help, so conservative); (4) does NOT address composition fragility / forward test / Binance→HL universe transport, which remain the binding risks.

### Step 53 — Universe-selection criterion: pred_z vs pred_B (COMPLETE)

User question: since the buggy Step 52 built the universe from pred_B, how does pred_B-universe compare to pred_z-universe? Held everything else identical (production protocol + causal aggregator + gate-consistent placebos).

| variant | universe (Stage 1) | Sharpe causal | P1 | P2 |
|---|---|---|---|---|
| Step 50 (Z) | pred_z trailing IC | **+3.11** | PASS p100 (+1.45) | PASS p100 (+1.61) |
| Step 53 (B) | pred_B trailing IC | **+1.16** | FAIL p89 (−0.49) | FAIL p88 (−0.69) |

**Universe-selection criterion matters enormously**: +1.95 Sharpe gap, pass-both vs fail-both. pred_z is decisively the correct Stage-1 criterion.

Mechanism: `pred_B = pred_z × trail_ic`, and `trail_ic` is itself derived from past `corr(pred_z, α)`. The rolling-IC universe selector computes trailing `corr(pred, α)`. Using pred_B double-counts the IC weighting (circular/compounding), selecting a noisier universe that doesn't generalize. The two-stage design — Stage 1 universe by clean `pred_z`, Stage 2 ranking by IC-signed `pred_B` — is correct as production has it.

This also confirms the first (buggy) Step 52 +1.16 was the reproducible pred_B-universe number, not noise.

### Step 52 — WRAPPER-SYMMETRIC placebo (COMPLETE) — reviewer's biggest gap, RESOLVED

The cleanest model-vs-random test. Unlike Steps 35-50 (gate-consistent but placebo bypassed `select_refill` and skipped `picks_hist`), here the ONLY difference between real and placebo is the `pred` values — model `pred_B` vs deterministic random scores. conv_gate, select_refill, picks_hist feedback, PM_M, sleeve all run IDENTICALLY on whatever pred is given.

| metric | value |
|---|---|
| Real Sharpe (model pred) | **+3.11** (exactly matches Step 50 → real path now production-identical) |
| Wrapper-symmetric placebo p95 / p99 / max | +1.74 / +2.08 / +2.63 |
| Real rank | **100.0%** (beats all 100 placebos) |
| Edge over p95 | **+1.37** |
| Edge over p99 | +1.03 |

**PASSES decisively.** Comparison to Step 50's gate-consistent-only placebo:

| placebo type | placebo p95 | real edge |
|---|---|---|
| Step 50 gate-consistent (placebo bypasses refill/picks_hist) | +1.50 | +1.61 |
| Step 52 wrapper-symmetric (placebo gets identical machinery) | +1.74 | +1.37 |

Giving random picks the select_refill/picks_hist machinery lifted the placebo p95 by +0.24 (the refill heuristic generates some alpha on any picks). But the model retains **+1.37 genuine ranking edge** on top — beating all 100 wrapper-symmetric placebos. **The model's prediction genuinely adds ranking value beyond the architecture machinery.** Reviewer's biggest remaining concern is resolved in the model's favor.

(Note: first Step 52 run had a universe-construction bug — built rolling-IC universe from pred_B instead of pred_z — giving real +1.16 on a different universe. Corrected to build universe from pred_z per production convention; real recovered to +3.11 confirming the real path is now identical to Step 50.)

### Step 51 — Causal K-drop universe stress (COMPLETE)

K-drop universe stress under causal aggregator (matches Step 50 accounting).

| K_drop | causal mean | std | worst | best | positive | lagged (Step 49) |
|---|---|---|---|---|---|---|
| 10 | +1.26 | 0.66 | −1.76 | +1.98 | 29/30 | +1.24 |
| 20 | +0.81 | 1.40 | −2.21 | +2.29 | 23/30 | +0.66 |
| 30 | +0.97 | 1.27 | −3.75 | +2.98 | 24/30 | +0.86 |
| 40 | +1.02 | 1.06 | −2.48 | +2.40 | 26/30 | +0.90 |

Causal slightly higher than lagged at every K level (+0.02 to +0.15) — the lag effect is universe-dependent rather than uniform. Composition-fragility verdict UNCHANGED: K-drop means (+0.81 to +1.26) are far below the full-panel baseline +3.11.

Step 51 script had a baseline-comparison bug (`shs > 3.35` instead of `> 3.11`). Fortunately the max K-drop Sharpe across all K levels was +2.98 (K=30) — below BOTH thresholds — so `n_beat_baseline` would have been 0/30 either way. The printed counts are correct in this case despite the threshold typo.

### Step 50 — Causal-aligned aggregator + final placebos (COMPLETE)

Built causal-immediate aggregator (`gross[t] = tw × alpha[t]`) to address the PnL timing inconsistency reviewer identified. Rerun Step 47 numbers + Step 48 placebos under causal aggregator.

| metric | lagged (Step 47/48) | causal (Step 50, FINAL) | Δ |
|---|---|---|---|
| Sharpe (B) | +3.35 | **+3.11** | −0.24 |
| folds+ | 6/9 | **7/9** | +1 (fold 3 flipped from −0.49 to +0.08; fold 1 still negative at −4.39) |
| gross/cycle | +16.66 | +15.42 | −1.24 |
| CI | [+1.35, +5.23] | **[+1.06, +4.92]** | both strictly positive |
| P1 placebo edge | +1.69 (p100) | **+1.45** (p100) | both PASS |
| P2 placebo edge | +1.81 (p100) | **+1.61** (p100) | both PASS |
| LOFO drives | folds 4+5 (Δ −0.41, −0.92) | folds 4+5 (Δ −0.56, −0.80) | similar |

Causal correction was uniform across metrics (~−0.2 to −0.25). Reviewer's manual recompute (+3.11–3.13) was exact: causal Sharpe = +3.11. Both placebos still pass at p100 with substantial edges.

**This is the cleanest, most-defensible 110-panel result** — full-PIT features + BTC excluded + causal PnL accounting + gate-consistent placebos at p100 with K=10 composition robustness.

### Step 49 — Full-PIT V2 110-panel universe stress (COMPLETE, LAGGED aggregator)

K-drop universe stress under full-PIT — **uses the lagged aggregator** (same convention as Step 47/48). Causal-aligned version running as Step 51 for parity with Step 50 final numbers.

| K_drop | full-PIT mean (LAGGED) | strict-PIT mean (Step 46) | non-strict mean (Step 43) | 51-panel ref (Step 40) |
|---|---|---|---|---|
| 10 | +1.24 (29/30 positive) | −1.38 | −1.24 | +1.28 |
| 20 | +0.66 (22/30 positive) | −0.72 | −1.02 | +0.69 |
| 30 | +0.86 (23/30 positive) | −0.21 | −1.05 | — |
| 40 | +0.90 (23/30 positive) | +0.19 | −0.43 | — |

**Interpretation**: The Step 47 OHLCV-feature shift coincided with resolving both the within-universe P2 failure and the K=10 sign-flip catastrophe — strong evidence the OHLCV leak was an upstream cause, but not formal proof (would need feature-by-feature ablation to fully isolate). **Important nuance**: K-drop means (+0.66 to +1.24) survive positive sign but are FAR BELOW the baseline +3.11. The headline Sharpe is not invariant to symbol composition — dropping 10 random symbols typically reduces strategy Sharpe to ~+1.2 (similar to 51-panel pattern). Worst-case still gets bad (K=30 worst −3.24).

Step 51 (causal version) will tell us if the same picture holds under corrected accounting; means likely shift ~−0.2 but qualitative pattern should hold.

### Step 48 — Full-PIT V2 110-panel P1+P2 placebos (COMPLETE)

After Step 47 full-PIT V2 +3.35, rerun gate-consistent placebos:

| placebo | description | p95 | V2 rank | edge | verdict |
|---|---|---|---|---|---|
| P1 | random pick from broad liquidity universe | +1.66 | **p100** | **+1.69** | PASS |
| P2 | random pick from V2's own rolling-IC top-15 universe | +1.54 | **p100** | **+1.81** | PASS |

**V2 full-PIT beats ALL 100 random matched picks on both placebos.** Edge progression from each PIT fix:

| version | P1 edge | P2 edge | verdict |
|---|---|---|---|
| Non-strict 110 (Step 42) | +0.60 (p99) | **−0.13** | P2 FAIL |
| Strict-PIT 110 (Step 45) | +0.96 (p97) | +0.66 (p99) | both pass |
| **Full-PIT 110 (Step 48)** | **+1.69 (p100)** | **+1.81 (p100)** | both pass at p100 |

The progression makes mechanical sense: leaky features gave random within-universe picks ARTIFICIAL alpha (a generic IC-inflation effect placebos can exploit). Removing the leak removes both real V2's leak-driven alpha AND the placebo's free benefit, but the real model retains GENUINE alpha while placebo random picks have none. Real edge widens as leaks are cleaned.

### Step 47 — Full-PIT V2 rebuild on 110-panel (COMPLETE)

After reviewer noted base OHLCV-derived features (return_1d, atr_pct, vwap_slope_96, bars_since_high) were copied unshifted from the 111-panel — still leaking bar-t close/high/low/volume:
- Added shift(1) per symbol for these 4 base features in `scripts/build_btc_features_111_full_pit.py:217`
- Retrained V2 (`47_v2_on_110_full_pit.py`)

Reviewer audit confirmed shifted values match `previous-bar` formulas, not current-bar.

| metric | strict-PIT (Step 44) | full-PIT (Step 47) | Δ |
|---|---|---|---|
| Sharpe (B_IC_signed) | +2.11 | **+3.35** | **+1.24** |
| Sharpe (A baseline) | +1.11 | +0.75 | −0.36 |
| folds+ | 6/9 | 6/9 | same |
| gross/cycle | +14.38 | +16.66 | +2.28 |
| overall IC | −0.0051 | −0.0039 | flat |
| **CI** | [+0.14, +3.77] | **[+1.35, +5.23]** | lower bound +1.21 |
| LOFO drives | fold 5 (Δ −0.45) | folds 4+5 (Δ −0.41, −0.92) | similar |

Largest single improvement from PIT discipline so far. A_baseline dropped (less direct prediction power); B_IC_signed jumped (wrapper compensates and then some). CI lower bound is now +1.35 — strongly positive.

**Reviewer-noted caveats**:
1. **PnL alignment**: sleeve aggregator MTMs `prev_weights × alpha[t]` while charging cost for transitioning to new weights at same `t`. This delays new weights by one cycle. Immediate-fill recompute = +3.11, causal close-at-current = +3.13 — reported +3.35 is ~+0.22 inflated by this lag.
2. **Entry convention**: target uses `close[t+48]/close[t] - 1`. If we trade at open_time t with features known through t-1, that's a 5-min entry lookahead. If convention is "decision after bar t closes" (entry at close[t]), then shifted features are correct. Documentation should make this explicit.
3. **Placebo wrapper still asymmetric**: `phase_ah_sleeve.py:181` — real uses `select_refill()` + `picks_hist`; placebos don't. Step 48 is gate-consistent but not wrapper-symmetric.

### Step 45 — Strict-PIT V2 110-panel P1+P2 placebos (COMPLETE, SUPERSEDED by Step 48)

100 seeds × P1 + 100 seeds × P2, gate-consistent (real and placebo both gate on pred_B). Uses Step 44 strict-PIT predictions.

| placebo | description | p95 | V2 rank | edge | verdict |
|---|---|---|---|---|---|
| P1 | random pick from broad liquidity universe (top-30 by 90d $vol within 110-panel) | +1.15 | p97 | **+0.96** | **PASS** |
| P2 | random pick from V2's own rolling-IC top-15 universe | +1.45 | p99 | **+0.66** | **PASS** |

**Both placebos pass cleanly** — the strict-PIT 110-panel V2 is the **first 110-panel result that fully placebo-validates end-to-end** (universe selection AND within-universe ranking).

Compare:
- V2 51-panel (Step 35): P1 p99 (+1.04), P2 p97 (+0.58) — both pass
- V2 110-panel non-strict (Step 42): P1 p99 (+0.60), **P2 FAIL p92 (−0.13)**
- V2 110-panel strict-PIT (this): P1 p97 (+0.96), P2 p99 (+0.66) — both pass

The strict-PIT fix transformed P2 from FAIL to PASS at p99 with edge +0.66. **The feature leak in the non-strict version was confounding the within-universe placebo test**, not just inflating absolute Sharpe.

Mechanism: leaky features gave random within-universe picks ARTIFICIAL alpha (a generic "all features in universe have inflated IC" effect that random picks could exploit). After strict-PIT, real model retains genuine signal but the leak-driven random benefit disappears. Real edge over placebo widens substantially.

### Step 44 — V2 strict-PIT rebuild on 110-panel (COMPLETE)

After reviewer identified feature-PIT issues (beta `.shift(1)` not `.shift(49)`; rolling features end at current bar), rebuilt 110-panel with strict PIT:
- `scripts/build_btc_features_111_strict_pit.py` → `outputs/vBTC_features_btc_only_111_strict_pit/panel_btc_only_111.parquet`
- Beta: `.shift(49)` matching 01_build_target.py
- All rolling-ending-at-current features `.shift(1)` so window ends at bar t-1
- Affected: `corr_to_btc_1d`, `idio_vol_to_btc_1h`, `idio_vol_to_btc_1d`, `dom_btc_z_1d`, `obv_z_1d`, `dom_btc_change_288b`, `btc_realized_vol_1d`, `btc_ret_48b`

Retrained V2 (`44_v2_on_110_strict_pit.py`, results in `step44_110_strict_pit/`).

| metric | non-strict (Step 41) | strict-PIT (Step 44) | Δ |
|---|---|---|---|
| Sharpe (B_IC_signed) | +2.03 | **+2.11** | **+0.08** |
| Sharpe (A baseline) | +1.43 | +1.11 | −0.32 |
| folds+ | 6/9 | 6/9 | same |
| gross/cycle | +10.39 | +14.38 | +3.99 |
| overall IC | −0.0033 | −0.0051 | flat |
| **CI** | **[−0.22, +4.14]** | **[+0.14, +3.77]** | **no longer crosses zero** |
| LOFO drives | folds 5+8 (Δ −0.69, −0.84) | fold 5 only (Δ −0.45) | less concentrated |

**Counter-intuitive result**: strict-PIT made V2 stronger, not weaker. Same pattern as Step 32→34 NaN fix and Step 41 PIT fix — feature-quality fixes improve the model. The "IC inflation" from leaky features (reviewer measured ~2× IC magnitude under current-bar convention) was not helping the model; cleaner features → cleaner signal.

**A_baseline dropped** (−0.32) — model's raw pred_z ranking lost some directional power from the leak. **B_IC_signed rose** (+0.08) — trail_ic wrapper compensates and then some.

**CI is now strictly positive** (was crossing zero). This is the first 110-panel result with statistically significant Sharpe at the 5% level.

Placebos on strict-PIT predictions ran as Step 45: P1 PASS p97 (+0.96), P2 PASS p99 (+0.66). Strict-PIT cleared P2 fail. Result later superseded by Step 47+50 full-PIT.

### Step 43 — V2 110-panel universe stress (COMPLETE, non-strict)

Run on non-strict predictions (Step 41 rerun, DEPRECATED). 30 random draws per K.

| K_drop | mean | std | p5/p50/p95 | worst | best | positive count | 51-panel ref (Step 40) |
|---|---|---|---|---|---|---|---|
| 10 | **−1.24** | 0.59 | −1.97/−1.34/−0.39 | −2.62 | +0.16 | 1/30 | mean +1.28 worst −1.09 |
| 20 | −1.02 | 0.75 | −2.19/−0.95/+0.10 | −2.64 | +0.36 | 2/30 | mean +0.69 worst −2.07 |
| 30 | −1.05 | 0.97 | −2.64/−1.10/+0.55 | −3.28 | +1.11 | 4/30 | — |
| 40 | −0.43 | 1.03 | −2.25/−0.40/+0.79 | −2.28 | +2.05 | 11/30 | — |

**All K-levels catastrophic** on non-strict 110-panel. Removing even 10 random symbols (9% of universe) collapses Sharpe from +2.03 to mean −1.24. By contrast, 51-panel K=10 was still mean +1.28. The non-strict 110-panel V2 result was almost entirely concentrated in specific high-IC symbol composition — random drops destroy it.

Consistent with Step 42 P2 fail: within-universe ranking is noise; the +2.03 came from finding the right ~15 symbols, and that selection is fragile.

### Step 46 — Strict-PIT V2 110-panel universe stress (COMPLETE)

Same K-drop methodology as Step 43, run on strict-PIT predictions.

| K_drop | strict-PIT mean | std | worst | non-strict (Step 43) |
|---|---|---|---|---|
| 10 | **−1.38** | 1.30 | −3.21 | −1.24 |
| 20 | −0.72 | 1.61 | −3.27 | −1.02 |
| 30 | −0.21 | 1.38 | −3.47 | −1.05 |
| 40 | +0.19 | 1.14 | −2.64 | −0.43 |

**Strict-PIT improved at higher K but K=10 catastrophe persists** — slightly worse than non-strict (−1.38 vs −1.24). 0/30 draws at any K-level beat baseline +2.11.

**Interpretation**: V2 110-panel's +2.11 is real for the specific 110-symbol composition (passes placebo + has strictly-positive CI), but **fragile to composition changes**. Dropping just 10 random symbols collapses Sharpe. This means:
- The model + sleeve genuinely finds signal in the present 110-symbol set
- But the signal is concentrated in a few high-IC symbols
- Real-world symbol delistings or composition drift would materially affect performance
- Same pattern as 51-panel Step 40 K=20 mean +0.69 (less catastrophic than 110-panel K=10 −1.38), suggesting 110-panel is MORE composition-fragile than 51-panel was

### Step 42 — P1+P2 placebos on V2 110-panel (COMPLETE)

100 seeds × P1 + 100 seeds × P2, gate-consistent (real and placebo both gate on pred_B). Reuses Step 41 rerun predictions.

| placebo | description | p95 | V2 rank | edge | verdict |
|---|---|---|---|---|---|
| P1 | random pick from broad liquidity universe (top-30 by 90d $vol within 110-panel) | +1.43 | p99 | +0.60 | **PASS** |
| P2 | random pick from V2's own rolling-IC top-15 universe | +2.15 | p92 | −0.13 | **FAIL** |

**Mixed verdict** — universe-selection layer transfers (P1 ✓), within-universe ranking does NOT transfer (P2 ✗).

Compare 51-panel (Step 35): P1 p99 (+1.04), P2 p97 (+0.58) — both PASS. On 51-panel V2 added value at both layers; on 110-panel V2 only adds value at the universe-selection layer.

**Mechanism (most likely)**: 110-panel adds 60 lower-IC alts with shorter histories. Rolling-IC universe filter still surfaces high-IC subset (P1 passes — those names dominate broad-liq random placebo). But within the universe, per-cycle ranking is closer to noise on 110-panel — model's discriminative signal doesn't scale to wider, noisier universe.

**Honest implication**: The +2.03 on 110-panel is mostly universe-selection alpha, not model-prediction alpha. V2 110-panel is **NOT placebo-validated** as an end-to-end trading strategy. The transfer claim from 51-panel to 110-panel is partial: the universe selector survives, the model's per-cycle ranks do not.

### Step 41 — V2 on 110-panel + V3.1 sleeve (RERUN COMPLETE, PIT-fixed)

Reviewer findings addressed in rerun (`outputs/vBTC_features_btc_only_111_pit/`, `linear_model/results/step41_111panel_pit/`):
1. **PIT shifts applied to 2 features**: `return_8h` now `.shift(1)`, `vol_zscore_4h_over_7d` now `.shift(49)` matching Step 29. **Other kline-derived features NOT audited** (`corr_to_btc_1d`, `idio_vol_to_btc_*`, `dom_btc_z_1d`, `obv_z_1d` all use current rolling values — same convention as 51-panel's `build_btc_only_features.py`, but not strict-PIT under the convention where the position is opened at `open_time`). Full feature-by-feature PIT audit is still pending; do not claim "PIT-clean" globally.
2. **BTCUSDT excluded** from training+evaluation (110 symbols, 116,880 rows removed)
3. **Audit artifacts saved**: predictions.parquet, per_cycle_{A,B}.csv, universe_per_cycle.parquet, sleeve_records_B.parquet, lofo_B.csv

**Result comparison (pre-fix vs post-fix):**

| variant | Sharpe (B) | Sharpe (A) | folds+ | gross/cycle | CI |
|---|---|---|---|---|---|
| pre-fix (PIT leak + BTC included) | +1.44 | −0.68 | 5/9 | +7.39 | [−1.19, +3.63] |
| **post-fix (2-feature PIT-shift + BTC excluded)** | **+2.03** | **+1.43** | **6/9** | **+10.39** | **[−0.22, +4.14]** |

Post-fix Sharpe is +0.59 HIGHER than pre-fix, not lower. Same pattern as Step 32→34 NaN fix: removing a feature-quality issue improved the model. The PIT misalignment in `return_8h` (close[t] vs `.shift(1)` for other features) was confusing the model; alignment fixed → cleaner predictions → directional A_baseline.

**Headline finding**: V2 on 110-panel = **+2.03**, only **−0.16** below 51-panel +2.19 despite universe doubling. R3_BTC discipline + universal-model + B_IC_signed wrapper combination preserves nearly all of the 51-panel performance when transferred to the 110-symbol universe. **This is the strongest evidence yet for the linear architecture's universe-portability**.

**Still required for production claim**:
1. 110-panel P1/P2 placebos (matched random picks under same gating)
2. Cost sensitivity sweep
3. Paired diff CI vs LGBM (need rerun of LGBM on 110-panel with same dates/cost convention)
4. CI [−0.22, +4.14] crosses zero — true Sharpe could be near zero
5. Fold concentration on folds 5+8 (Δ −0.69, −0.84) — single-fold dependency exists

**LGBM Phase UNI-111 = −1.48 (memory) is CONTEXTUAL only**, not apples-to-apples (different dates, cost convention, target hack `target_A clip-at-±5`, etc.).

### Step 41 PRE-FIX (HISTORICAL — SUPERSEDED BY RERUN ABOVE)

Original Step 41 run with PIT leak in return_8h/vol_zscore_4h_over_7d and BTCUSDT included in training. Numbers kept for audit trail; **do not cite this section's claims**.

User question: "can we run on a different symbol set to test robustness, LGBM only worked on 51 specifically"

| variant | Sharpe | CI | folds+ | gross | overall IC | LOFO drives |
|---|---|---|---|---|---|---|
| V2 on 51-panel (Step 34/35) | +2.19 | — | 7/9 | +3.40 | −0.0007 | folds 1, 4 |
| V2 on 111-panel pre-fix (deprecated) | +1.44 | [−1.19, +3.63] | 5/9 | +7.39 | −0.0012 | folds 5, 8 |
| LGBM Phase UNI-111 (memory) | −1.48 | — | 3/9 | — | — | — |

Pre-fix issues (all now corrected in the rerun above):
- `return_8h` unshifted (Step 29 used `.shift(1)`)
- `vol_zscore_4h_over_7d` unshifted (Step 29 used `.shift(49)`)
- BTCUSDT in training panel with sigma_idio clipped to 1e-6
- Inflated overstated claim "dramatically better than LGBM" was based on apples-to-oranges memory value
- A_baseline was −0.68 (model not directional)

### Step 39 — V2 fixed WITHOUT conv_gate (COMPLETE)

Isolate conv_gate's contribution by setting GATE_PCTILE=0 (no skip).

| ranking | with conv_gate | without conv_gate | gate Δ |
|---|---|---|---|
| A_baseline (pred_z) | +0.99 | −1.43 | +2.42 |
| B_IC_signed (pred_B) | +2.19 | +1.85 | +0.34 |

Without conv_gate: 8/9 folds+ (improves from 7/9), gross PnL UP (+3.40 → +4.09), but Sharpe DOWN (gate is Sharpe-optimizer not PnL-optimizer — filters high-variance cycles). 98% of cycles traded vs ~50% with gate.

Key insight: **trail_ic already does most of conv_gate's filtering work for B_IC_signed**. The IC-signed wrapper implicitly shrinks bad-cycle/bad-symbol predictions; conv_gate adds only marginal cycle-level skip benefit (+0.34). For A_baseline (no trail_ic), gate is critical (+2.42 lift).

vs LGBM Phase J: conv_gate lift for LGBM was +2.78. For Ridge B_IC_signed it's +0.34. Ridge's narrower pred distribution + trail_ic per-symbol calibration absorb most of the gate's would-be benefit.

### Step 38 — 24h target + 24h-ALIGNED features + sleeve (COMPLETE)

After Step 37 showed 24h target alone hurts (+1.50 vs +2.19), user proposed aligning features to 24h horizon:
- Drop V2 short-horizon features (atr_pct, vwap_slope_96, idio_vol_to_btc_1h, return_8h_orth, vol_zscore_4h_over_7d)
- Add longer-horizon: return_3d, return_5d (compounded from return_1d), btc_realized_vol_1d, btc_ret_288b, log_dollar_volume_7d
- Net 22 features (same count as V2)

| variant | target | features | Sharpe | folds+ | IC |
|---|---|---|---|---|---|
| V2 (Step 34/35) | 4h | V2 (4h-tilted) | +2.19 | 7/9 | −0.0007 |
| Step 37 | 24h | V2 (4h-tilted) | +1.50 | 6/9 | −0.0017 |
| **Step 38** | **24h** | **24h-aligned** | **+0.50** | **3/9** | **+0.0043** |

**Aligning features to 24h horizon DECREASED Sharpe by −1.00 vs Step 37.** Despite Step 38 having the BEST IC of any sleeve variant (+0.0043), it produces the WORST Sharpe (+0.50). Same "architecture-is-the-alpha" pattern: better predictions don't translate when basket-construction dominates. Confirms vBTC's feature-co-fit lesson for linear: changing V2's feature mix breaks the sleeve's implicit co-fit, even when the change is theoretically "horizon-aligned".

### Step 37 — SLEEVE + 24h RETURN PREDICTION (COMPLETE)

User question: "we hold 24h via sleeve, why predict 4h? — try sleeve + 24h target".

| variant | target | execution | Sharpe | folds+ | overall IC |
|---|---|---|---|---|---|
| V2 (Step 34/35) | 4h forward | V3.1 sleeve | +2.19 | 7/9 | −0.0007 |
| **Step 37** | **24h forward** | **V3.1 sleeve** | **+1.50** | **6/9** | **−0.0017** |

24h-target HURTS by −0.69 Sharpe. CI [−0.69, +3.61] crosses zero. Confirms Phase AH's LGBM lesson holds for linear: native long-horizon target is worse than 4h target + sleeve aggregation. **The sleeve's 6-prediction averaging adds variance reduction the model can't replicate by directly predicting 24h.** 4h target gives the sleeve better ingredients.

### Step 36 — V0/V1/V2 raw 4h cycle (NO sleeve) (CORRECTED 2026-05-14)

Tested whether linear variants have lift at the clean 4h-cycle level without V3.1 sleeve overlay.

**Cost correction**: original Step 36 used `2 × K × 4.5 = 27 bps/cycle` which triple-counted cost. Correct: each leg at weight 1/K bears `4.5/K` bps round-trip; K=3 long + K=3 short = `2 × COST_PER_LEG = 9 bps/cycle` at full turnover. Numbers below reflect the corrected cost.

| variant | A_baseline | B_IC_signed | folds+ (B) | gross/cycle (B) | CI on B |
|---|---|---|---|---|---|
| V0 standard | −2.36 | −1.84 | 3/9 | +1.56 | [−5.04, +0.39] |
| V1 fixed | −0.77 | −1.30 | 2/9 | +1.77 | [−5.31, +0.80] |
| V2 fixed | −2.06 | −2.55 | 2/9 | −0.16 | [−5.07, −0.37] |

All linear variants are negative at raw 4h cycle. The 9 bps cost still exceeds the model's gross PnL (~0 to 2 bps per cycle on B_IC_signed).

LGBM K=3 raw 4h cycle = +1.98 (per memory) — LGBM has gross signal large enough to cover 9 bps cost; Ridge does not.

**Sleeve contribution for Ridge**: V2 with sleeve +2.19 − V2 raw −2.55 = **+4.74**. The sleeve adds substantial Sharpe by reducing effective per-cycle cost via turnover smoothing (100% → ~16% turnover) and extending effective hold to 24h (more 4h α accumulated per position).

**Sleeve contribution for LGBM**: +2.23 (sleeve) − +1.98 (raw) = +0.25. LGBM's raw 4h already works; sleeve is a polish.

**Conclusion**: Linear model is **net-negative at raw 4h** but not catastrophically so. The V3.1 sleeve is structurally REQUIRED for linear to be viable. LGBM doesn't have this dependency.

### Open reviewer concerns (still relevant for forensic accuracy)

1. **picks_hist asymmetry in placebos** — `phase_ah_sleeve.py:216` only updates picks_hist when `placebo_seed is None`. Real V2 gets refill/picks_hist feedback loop; placebos don't. Confirmed the architecture (not just gating) is what carries V2's apparent edge.
2. **100 seeds thin for p97 placebo claim** — Step 35's max placebo was +2.65 above V2 +2.19. Pass at p97 with 100 seeds has ~3% noise. For tight claims need 1000+ seeds.
3. **V2 A_baseline = +0.99 with sleeve** — softens "wrapper-only" framing; 2 short-horizon features improve raw ranking too, not just IC-signed wrapper. But Step 36 shows even A_baseline at raw 4h is −3.94 to −5.99, so this lift is also sleeve-driven.
4. **Step 34 didn't save per-fold coefficients / Ridge α / feature contribution by fold** — main interpretability artifact for linear models. Not yet remedied.
5. **Preprocessing fit on fold-0 calendar rows; training filters autocorr_pctile_7d ≥ 0.5** — not leakage but the two distributions don't exactly match.

---

## Result ledger (current best understanding, post-review)

| variant | preprocessing | A_baseline | B_IC_signed | folds+ | overall IC | P1 | P2 | status |
|---|---|---|---|---|---|---|---|---|
| V0 R3_BTC standard (Step 32) | winsorize + z-score | +0.15 | +0.67 | 5/9 | −0.0019 | — | — | re-baseline |
| V1 R3_BTC proper (Step 32, pre-fix) | rank heavy-tail + per-sym funding (BROKEN NaN) | +0.15 | +1.31 | 4/9 | +0.0011 | p96 PASS +0.12 | p94 FAIL −0.03 | NaN bug + scale mismatch — superseded by V1 fixed |
| V2 V1 + short_horizon (Step 32 pre-fix) | V1 + return_8h_orth + vol_z (BROKEN NaN) | — | +0.33 | 5/9 | — | — | — | superseded |
| V1 R3_BTC proper (Step 34 fixed) | V1 + NaN fix + re-std | +0.32 | +1.21 | 4/9 | +0.0008 | **FAIL** p89 (p95=+1.53) edge −0.32 | **FAIL** p90 (p95=+1.65) edge −0.44 | rejected — V1 alone is architecture-noise |
| V2 V1 + short_horizon (Step 34 fixed) | V2 + NaN fix + re-std | +0.99 | +2.19 | 7/9 | −0.0007 | PASS p99 (p95=+1.16) edge +1.04 | PASS p97 (p95=+1.61) edge +0.58 | **placebos passed; V2 raw 4h cycle = −2.55 (CORRECTED cost 9 bps) → sleeve adds +4.74, not +9.6 as initially reported** |
| V0/V1/V2 raw 4h cycle (Step 36 corrected) | various | −2.36/−0.77/−2.06 | −1.84/−1.30/−2.55 | 3/9 / 2/9 / 2/9 | — | — | — | net-negative without sleeve; sleeve structurally required |
| V2 on 110-panel + sleeve (Step 41 rerun + Step 42, DEPRECATED) | V2 features, 2-feature PIT-shift fix + BTC excluded | +1.43 | +2.03 | 6/9 | −0.0033 | PASS p99 (+0.60) | **FAIL** p92 (−0.13) | superseded by Step 44 strict-PIT; CI crosses 0; universe stress K=10 mean −1.24 |
| V2 on 110-panel STRICT-PIT (Step 44+45, BTC-frame only) | V2 features, BTC-frame PIT-shift + BTC excluded; base OHLCV still unshifted | +1.11 | +2.11 | 6/9 | −0.0051 | PASS p97 (+0.96) | PASS p99 (+0.66) | Superseded by full-PIT below |
| V2 on 110-panel FULL-PIT lagged (Step 47+48) | V2 features, full PIT-shift + BTC excluded, LAGGED aggregator | +0.75 | +3.35 | 6/9 | −0.0039 | PASS p100 (+1.69) | PASS p100 (+1.81) | Lagged version; superseded by causal below. |
| **V2 on 110-panel FULL-PIT CAUSAL (Step 50, FINAL)** | V2 features, full PIT-shift + BTC excluded + CAUSAL PnL aggregator | not computed | **+3.11** | 7/9 | −0.0039 | **PASS p100 (+1.45)** | **PASS p100 (+1.61)** | FINAL clean result. CI [+1.06, +4.92]. Both placebos pass at p100 even under corrected accounting. K=10 univ stress mean +1.24 (Step 49 LAGGED; Step 51 causal version running). |

---

## Historical (stale or superseded) results

These were either pre-σ-fix, pre-NaN-fix, or otherwise compromised. Listed for audit trail only.

| label | Sharpe | issue |
|---|---|---|
| R3_BTC + IC-signed (Step 19) | +1.92 | **σ-idio fallback leak inflated this** (HYPE/ASTER were using full-panel std); after σ fix Sharpe is +0.86. Also fold-6-fragile (excl fold 6 → +0.40). |
| R3 + IC-signed corrected (Step 13) | +0.86 | fold-1+2 dependency. P1 p87 FAIL, P2 p82 FAIL. |
| R3 + IC-signed pre-σ-fix | +0.15 | σ leak was SUPPRESSING this number. |
| R7 + IC-signed (Step 15) | +1.60 | fold-4 dependency; trail_ic-shuffle placebo p89 FAIL. |
| Original Ridge w/ sym dummies (Step 3) | −1.62 | sym dummies absorbed 56× more coef mass than numerics. |

All sub-Step-32 results from before the σ-idio fallback fix should be treated as historical. The σ-leak was found to be SUPPRESSING the corrected R3 result (not inflating it), so post-fix numbers are slightly HIGHER not lower than pre-fix — except for R3_BTC, where the leak inflated.

---

## Canonical baseline: R3_BTC (20 features)

Pure BTC-frame, β-hedged target. No basket features, no sym_id.

```
11 frame-neutral W17 numeric:
  return_1d, atr_pct, obv_z_1d, vwap_slope_96, bars_since_high_xs_rank,
  funding_rate, funding_rate_z_7d, funding_rate_1d_change,
  corr_to_btc_1d, idio_vol_to_btc_1h, beta_to_btc_change_5d

3 R3 squared U-shape:
  return_1d², corr_to_btc_1d², beta_to_btc_change_5d²

4 BTC-frame replacements for basket features:
  dom_btc_z_1d, dom_btc_change_288b, corr_to_btc_change_3d, idio_vol_to_btc_1d

2 BTC squared U-shape:
  dom_btc_change_288b², corr_to_btc_change_3d²
```

**Preprocessing (V1 proper, post-fix in Step 34):**
- HEAVY_TAIL (kurt > 50) → rank transform pooled (`vwap_slope_96`, `idio_vol_to_btc_1h`, `idio_max_abs_12b`, `funding_rate*`)
- PER_SYMBOL_Z funding features → per-symbol rank pooled fold-0
- All ranks → re-z-scored to unit variance
- Non-heavy-tail → standard winsorize p1/p99 + z-score
- Squared terms use standard z-scored base (rank-of-rank doesn't help)

**Future variants extend FROM R3_BTC** with same preprocessing discipline. Basket-frame features (`xs_alpha_*`, `*_vs_bk`) are excluded since target is BTC-hedged.

---

## Why the lift sits in the wrapper, not the model

Critical finding from reviewer:
- V1 raw IC = +0.0011 (essentially zero; LGBM production = +0.0157, 14× higher)
- V1 A_baseline Sharpe = +0.15 (using `pred_z` directly to rank — break-even)
- V1 B_IC_signed Sharpe = +1.31 (using `pred_z × trail_ic` to rank)

The B variant's signal source decomposes as:
- Universe selection by rolling-IC top-15 (eligibility filter)
- Per-symbol sign flipping via `sign(trail_ic[sym, t])`
- Magnitude weighting via `pred_z × |trail_ic|`

Since A_baseline ≈ 0, all useful effect comes from the B-specific transformations. The linear model is effectively a **noisy magnitude generator** that the IC-signed wrapper rectifies.

This explains:
- Why V1 PASSES P1 (random pick on broad liquidity universe) but FAILS P2 (random pick within V1 universe): the universe filter contributes most of the alpha; ranking within universe is interchangeable with random.
- Why even very low-IC variants (R3_BTC Step 19 negative IC −0.0019) produced high Sharpe: architecture mechanics dominate.

---

## Calculation conventions (clarified 2026-05-14 per reviewer)

### Entry timing — "decision at close of bar t" convention

- `return_pct` (target) in panel: `close[t+48] / close[t] − 1` (forward 4h return)
- `close[t]` is the closing price at time `t + 5min` (end of 5m bar starting at open_time t)
- Convention: **decision is made at the close of bar t** (i.e., at time t + 5min when close[t] becomes observable). Entry price = close[t].
- Features for row t in the full-PIT panel use data through close[t−1] (one bar before). Under this convention, the panel is **conservative**: features could have used close[t] info (it's known at decision time) but don't, foregoing 5 min of information.
- An alternative "trade at open_time t" interpretation would require features through close[t−1] AND target = close[t+48]/close[t−1]; that's not the convention used. The shifted features under the close-of-bar-t convention are not a leak — they're slightly under-using available info.

### Feature shift discipline (full-PIT version)

| feature | shift |
|---|---|
| `beta_btc_pit` | `.shift(49)` = HORIZON+1 (matches 01_build_target.py) |
| `corr_to_btc_1d`, `idio_vol_to_btc_1h`, `idio_vol_to_btc_1d`, `dom_btc_z_1d`, `dom_btc_change_288b`, `corr_to_btc_change_3d`, `obv_z_1d`, `btc_realized_vol_1d`, `btc_ret_48b` | `.shift(1)` (rolling end at bar t−1) |
| `beta_to_btc_change_5d` | inherits beta's shift(49) via diff |
| `return_1d`, `atr_pct`, `vwap_slope_96`, `bars_since_high` (and `bars_since_high_xs_rank`) | `.shift(1)` per symbol (added in Step 47 full-PIT builder) |
| `return_8h` | `.shift(1)` per Step 29 |
| `vol_zscore_4h_over_7d` | `.shift(49)` per Step 29 |
| `funding_rate`, `funding_rate_z_7d`, `funding_rate_1d_change` | NOT shifted (Binance funding announced before window, known at open_time t) |

### Sleeve PnL aggregator — TWO conventions in use

Two versions of `aggregate_hold_through()` exist:

**Lagged (older steps, Steps 32–49):**
```python
gross[t] = prev_weights × alpha[t]                # MTM uses PREVIOUS weights
cost[t]  = |tw - prev_weights| × cost_per_unit    # cost charged at t for transition to tw
```
Closed weights "earn" the forward 4h alpha starting at t (one cycle "free" after they should close). Inflates Sharpe ~+0.2 vs causal.

**Causal-immediate (Step 50+):**
```python
gross[t] = tw × alpha[t]                          # MTM uses NEW weights established at t
cost[t]  = |tw - prev_weights| × cost_per_unit    # same cost
```
New weights earn the forward alpha from their decision time. Closed weights earn nothing forward.

**Which step uses which**:
| step range | aggregator | Sharpe convention |
|---|---|---|
| 32-49 (incl. 51-panel V2 +2.19, Step 47/48 +3.35, Step 49 K-drop) | lagged | inflated ~+0.2 |
| Step 50, Step 51 | causal-immediate | corrected |

Step 51 (causal K-drop) is running. After it completes, both Sharpe AND K-drop tests will have causal-corrected versions for direct comparison.

For numbers prior to Step 50 (e.g., 51-panel V2 +2.19), the causal-corrected estimate is ~+1.95–2.0 by analogy with the −0.22 correction Step 50 measured on 110-panel. Both panels likely need to be rebuilt with the causal aggregator for a strict apples-to-apples cross-panel comparison.

### Cost convention

- `COST_PER_LEG = 4.5 bps` round-trip per unit-weight position
- K=3 long + K=3 short at weight 1/K each = `2 × COST_PER_LEG = 9 bps` per full-turnover cycle
- Earlier mis-stated as 27 bps (triple-counted); all post-2026-05-14 numbers use the correct 9 bps

## PIT controls — 110-panel full-PIT version (Step 47+)

In the full-PIT 110-panel rebuild (Step 47), all kline-derived features are shifted. The 51-panel V2 (+2.19) used a looser convention and has NOT been rebuilt with the same full-PIT discipline — that's a remaining parity gate.

| component | discipline (110-panel full-PIT) |
|---|---|
| β estimation | `.shift(49)` = horizon+1 bars |
| σ_idio | fold-0 train rows only, cross-symbol fold-0 median fallback |
| Winsorize / z-score | fold-0 train quantiles + mean/std |
| Rank transform (heavy-tail) | fold-0 train distribution, NaN-safe |
| Per-symbol rank (funding) | fold-0 train per-symbol, NaN-safe |
| return_8h, vol_zscore_4h_over_7d | `.shift(1)` / `.shift(49)` per Step 29 |
| return_1d, atr_pct, vwap_slope_96, bars_since_high | `.shift(1)` per symbol (Step 47) |
| corr_to_btc_1d, idio_vol_to_btc_{1h,1d}, dom_btc_z_1d, dom_btc_change_288b, obv_z_1d, btc_realized_vol_1d, btc_ret_48b | `.shift(1)` (rolling end at t−1, Step 44+ strict + Step 47 verified) |
| Trailing IC for ranking | trailing 90d cycles, strict < current |
| Ridge α selection | RidgeCV gcv on fold-train + bootstrap |
| Funding features | NOT shifted (Binance funding announced before window, known at open_time t) |

**51-panel V2 (+2.19) used the older non-shifted convention for base OHLCV features and rolling-end-at-current for BTC-frame features.** Under full-PIT rebuild, it likely shifts upward (by analogy with the 110-panel pattern, where each PIT fix improved the result). 51-panel full-PIT parity rebuild is still pending.

---

## Open questions / next gates (post Step 50 final)

1. **Step 51 (causal universe stress)** — currently running. Will give K-drop means under the same causal aggregator as Step 50 Sharpe/placebos. Step 49 (lagged) gave K=10 mean +1.24; causal version expected modestly lower (~−0.2 by analogy).
2. **51-panel full-PIT + causal parity rebuild** — current 51-panel V2 +2.19 used non-shifted base OHLCV + lagged aggregator. For consistent cross-panel comparison, need rebuild.
3. **Full-wrapper-symmetric placebo** — `phase_ah_sleeve.py:216` only updates picks_hist for real path. Step 50 still passes by wide margin (+1.45/+1.61 edges) but a wrapper-symmetric placebo is the cleanest remaining validation.
4. **1000-seed placebos** — Step 50 used 100 seeds; p100 with 100 seeds has wide standard error on the tail. Tightening would not change verdict but would tighten the claim.
5. **Cost sensitivity sweep** — cost ∈ {1, 3, 6, 9, 12} bps/leg; checks whether V2 lift persists across cost regimes.
6. **Paired diff CI vs LGBM** — need LGBM rerun on same 110-panel with same dates/cost/target/aggregator before any "V2 beats LGBM" production claim.
7. **Feature-by-feature ablation** — would formally isolate which feature(s) were the dominant leak source (vs the observed "Step 47 fixed both symptoms" coincidence).

---

## Files index (relevant subset)

```
linear_model/
├── docs/STATUS.md, HANDOFF.md, RESULTS.md, design.md
├── scripts/
│   ├── 01_build_target.py     z-target, σ_idio fold-0 cross-sym median fallback
│   ├── 32_r3_btc_clean_preprocessing.py  V0/V1/V2 build (NaN bug, pre-fix; superseded by 34)
│   ├── 33_validate_v1.py      V1 P1/P2 placebo validation (gate-inconsistent; superseded by 35)
│   ├── 34_v1_nan_fixed.py     V0/V1/V2 with NaN fix + re-std (CURRENT 51-panel V2)
│   ├── 35_placebo_v1_v2_fixed.py  V1/V2 P1/P2 gate-consistent placebos
│   ├── 36_raw_4h_cycle_no_sleeve.py  V0/V1/V2 raw 4h cycle no sleeve (cost-corrected 9 bps)
│   ├── 37_sleeve_plus_24h_target.py  24h target + sleeve test
│   ├── 38_24h_target_aligned_features.py  24h target + 24h-aligned features test
│   ├── 39_v2_no_conv_gate.py  V2 without conv_gate
│   ├── 40_v2_universe_stress.py  V2 universe stress (drop K random from 51)
│   └── 41_v2_on_111_panel.py  V2 trained+tested on 110-panel (BTC excluded, PIT-fixed for 2 features)
└── results/
    ├── step34_v1_fixed/                       51-panel V2 artifacts
    ├── step35_*.csv, step35_verdict.csv       51-panel placebos
    ├── step36_*.csv                           raw 4h cycle (corrected cost)
    ├── step39_*.csv                           no-conv_gate test
    ├── step40_universe_stress_summary.csv     universe stress 51-panel
    ├── step41_111panel/                       PRE-FIX 111-panel (DEPRECATED)
    ├── step41_111panel_pit/                   POST-FIX 110-panel (CURRENT)
    └── feature_inventory_audit.csv            candidate features (btc_ret_fwd stripped, hour_cos flagged)
```

Companion outputs:
- `outputs/vBTC_features_btc_only_111_pit/panel_btc_only_111.parquet` — 110-panel feature panel (BTC excluded, 2 features PIT-shifted)
- `outputs/vBTC_features_btc_only_111/` — pre-fix version (deprecated; keep for diff)

## Memory references

- `~/.claude/projects/-home-yuqing-ctaNew/memory/project_vBTC_linear_model.md`
- `~/.claude/projects/-home-yuqing-ctaNew/memory/project_vBTC_status.md`
