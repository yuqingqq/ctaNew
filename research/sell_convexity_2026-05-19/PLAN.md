# Sell-Convexity — Pre-Registered Plan v1 (2026-05-19)

## Thesis (theory- and data-aligned, non-redundant)
The lottery/skew premium accrues to the **seller** of convexity, not the
buyer (Bali/Brunnermeier–Kumar–Page; established, replicated). "Just long the
primed names" is therefore negative-EV (confirmed: theory + C0pre down>up +
ex-VVV decay + unseen −0.33). The *un-refuted, theory-positive* side is to
**short the high-skew/primed-for-a-big-move cohort, market-hedged**, harvesting
the negative-skew + post-pump-reversal premium. Our own evidence already
points the right way: the validated full-panel PIT detector (C0pre) predicts
big-**DOWN** at OOS-symbol AUC **0.70** > big-UP 0.67, placebo 0.50, and is
NOT selector-echo (built on the full 51-name panel, every name every cycle).
Binding open question: **does the premium survive realistic frictions
(observable perp funding paid by shorts on exactly these names) and tail risk,
and does it port?**

## Hard priors carried (must not be re-discovered the hard way)
1. Long-skew = negative-EV → not retested; only the SHORT side.
2. **Frictions are THE binding constraint** and are worst on the low-float
   names that carry the premium. Cost must be **observable**, not a knob:
   short pays the realized **funding rate** (panel `funding_rate`,
   `funding_rate_z_7d`) over the hold — on pumped low-float names funding
   spikes positive against shorts (the real, measurable squeeze cost) — plus
   √ADV slippage + tail-stress. NO fictional idio-vol borrow knob (prior
   review kill).
3. **Portability is the gate** (R3c: group-disjoint, no sym_id, unseen
   symbols, beta-neutral, costed). In-universe numbers are diagnostics only.
4. **Use the CORRECTED aggregation** (oi_flow_test_v2 fix): strict
   within-group pairing, NO cartesian `time` join, honest n_eff = cycles/BLK,
   per-group level-CI, LOFO single-group sign-flip. Power-limited
   (~5 groups / ~0.74y) → pre-register MDE; three-way verdict
   (real / detectable-null / underpowered-indeterminate); never "exhausted".
5. Signal built on FULL panel (every name every cycle) — NOT old-selector-
   entered legs (the convexity-mining fatal flaw). Anti-selection-endogeneity.
6. Tail risk is FIRST-CLASS: short-vol has a fat left tail. maxDD / worst-
   cycle / left-tail are pre-registered kill criteria even if Sharpe>0; hard
   per-name cap + stop included by construction.
7. No goalpost-moving; a prediction miss rewrites the diagnosis, not the gate;
   honest negative is a completed iteration that spawns the next hypothesis.

## Locked parameters
- Panel `outputs/vBTC_features/panel_variants_with_funding.parquet` (R0-clean).
  Signature = the 12 PIT C0pre features (`atr_pct, idio_vol_1d_vs_bk,
  idio_vol_to_btc_1d, idio_skew_1d, idio_kurt_1d, idio_max_abs_12b,
  name_idio_share_1d, name_factor_loading_1d, funding_rate_z_7d, return_1d,
  dom_change_288b_vs_bk, corr_to_btc_1d`). Leak guard: denylist +
  blocking `max|rankIC(sigfeat,target_A)|<0.10`; prefix-causal; label uses
  realized fwd (necessarily) but every feature strictly prior.
- Eval = R3c portable protocol; corrected within-group pairing; seed 20260519;
  5 disjoint groups; BLOCK=11; block-bootstrap n=2000; one-sided LCB.
- Costs (all reported; gate keyed to the realistic one): (a) flat 4.5 bps;
  (b) realized √ADV; (c) **observable funding** = short pays
  `funding_rate`×(hold/8h) per cycle (sign-correct: short pays when
  funding>0); (d) tail-stress 3× √ADV on top-vol-decile; (e) gap/squeeze
  stress: extra −X bps on the worst-decile adverse moves of held shorts.
- Construction: each cycle, rank full eligible universe by signature score;
  **short** top-Kσ "primed" names (K∈{3,5}); market hedge = equal-notional
  **long BTC** (and a beta-neutral variant via trailing-288 PIT β); hard
  per-name cap = 1/3 book gross; 6-sleeve overlap (production machinery).

## Tests (pre-registered; absolute, falsifiable)

### S0 — Signal-validity reuse (no recompute; cite C0pre)
Confirm the detector to be shorted is the validated full-panel one:
OOS-symbol AUC big-down 0.70 / placebo 0.50 / not selector-echo
(`convexity_mining_2026-05-19/results/C0pre_decisive.json`). Pre-registered:
if the short cohort is NOT the C0pre detector, STOP (no re-mining).

### S1 — Sell-convexity portable, net of OBSERVABLE funding (the decisive test)
Short top-Kσ primed names, BTC-hedged + beta-neutral variant, R3c portable,
corrected aggregation. Report gross, net-√ADV, **net-observable-funding**,
per-group, LOFO, level-CI, honest n_eff/MDE.
- **PASS:** portable **net-of-observable-funding** Sharpe ≥ **+0.5**, one-sided
  LCB > 0, ≥4/5 groups positive, no LOFO sign-flip.
- **detectable-null:** HCB < +0.2 (premium ruled out above a fundable bar).
- **indeterminate:** neither (power-limited; the honest default given the
  documented ~5-group/0.74y limit) — report as effect-size, NOT "exhausted".
- Pre-registered prediction: **gross/√ADV positive** (premium exists) but
  **net-observable-funding ≤ +0.2** (friction-killed) — the literature-
  consistent expectation. A miss either way rewrites the diagnosis.

### S2 — Tail & friction stress (kill-criteria even if S1 PASS)
On the S1 config: maxDD, p1 worst-cycle, left-tail vs the V3.1 baseline;
net under tail-stress (c) + gap/squeeze (e); capacity (√ADV-scaled size).
- **Kill** (config not deployable regardless of Sharpe): capped maxDD >
  2× R1-uncapped maxDD, OR net-of-tail-stress Sharpe < +0.3, OR a single
  cycle/group drives >60% of net.

### S3 — Synthesis & decision (written only after S0–S2)
Sized: does a *portable, friction-survived, tail-bounded* short-skew premium
exist? Exact gate passed/missed + margin. No pre-written verdict. If FAIL →
re-initiate the loop with the next genuinely-distinct hypothesis (candidates,
ranked: longer-hold sell-convexity for funding amortization; explicit
funding-carry × skew combined harvest; regime-conditional sell-convexity).

## Process
Plan → **3-agent plan review** (methodology / profitability / red-team) →
revise to alignment before any test. Heavy test sequenced **after the running
OI/flow v2 frees CPU** (no thrash). After S0–S2 → **3-agent results review**
vs these pre-registered gates; fudge/leak/goalpost ⇒ re-initiate. Iterate the
whole loop until a hypothesis passes honestly or the genuinely-distinct space
is exhausted (then report that, honestly, as the terminal answer).

## Out of scope
Long-skew (closed); return-direction forecasting (closed, IC≈0.02); the
convexity-mining cost-concentration angle (closed — directionless);
re-deriving any closed arc; paid data; live deployment.
