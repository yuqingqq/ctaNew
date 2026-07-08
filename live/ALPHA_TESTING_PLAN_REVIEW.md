# Review of the re-formed alpha-evaluation framework (other session, 2026-07-06)

Reviewed: ALPHA_TESTING_PLAN.md (integration points I1–I6, Path A/B/C/D, tip-accuracy upgrade),
tip_accuracy.py, V4_PERFORMANCE.md state. Verdict: **framework is a genuine upgrade — the two core
ideas are correct and independently confirmed by this session's validation work. Six fixes below.**

## What is right (cross-validated)

1. **Integration-point taxonomy (I1–I6).** Confirmed mechanically: full-stack leg attribution shows
   bull entries are pred-independent (identical picks across models — return_1d ranker + BTC hedge),
   so I1 feature-tests structurally cannot act in bull. And the wq036 adoption failure proved the
   dual: a signal can be real (pred-orthogonal t +3.79) yet inexpressible at I1 (per-symbol ridge
   cannot encode cross-sectional rank info) — integration point is a first-class dimension, not a detail.
2. **Tip-accuracy over average IC.** Correct for a K=1/2-selection book; consistent with the decile
   findings (productive middle band, anti-calibrated extreme) and with the K2 calibration case
   (avg IC ambiguous, tip predicted the −0.67 backtest). "IC is a screen-out, not a confirm" is the
   right epistemics.
3. **Distribution-shape gates** (median≈mean, top-3<40%, halves, placebo) encode exactly the
   validated failure modes (alpha095 bull mirage; v4's own Nov-2025 concentration).
4. **"Strategy layer is the only number that decides adoption"** — matches the rr_both reversal
   (won gate-free, lost −1.09 Sharpe under gates at fair fees).

## Fixes needed

1. **The tip screen has no significance machinery.** K=1–2 L/S per cycle is maximal-variance, and the
   gate is "improves mean AND reliability" with no error bars — with 3 K's × 3 metrics × many
   candidates, chance passes are guaranteed. Fix: gate on the PAIRED per-cycle L/S diff
   (candidate − base) with non-overlap (stride-H) t or block bootstrap, and apply the concentration/
   halves gates to the DIFF, not just the levels.
2. **"reliability(Sh)" is an overlap-inflated t-stat, not a Sharpe.** 24h-fwd sums at 4h cadence →
   ~2.4× inflation (same as the old screens). Fine for A/B (both sides inflated); never read it
   absolutely. Cheap fix: stride-6 subsample as in alphaset_validate.py.
3. **Regime-frame mismatch + power.** Signal-layer bull stats use the trend-threshold frame
   (thousands of cycles); the strategy's bull is the bot's HYSTERESIS regime — only **137 cycles**
   in the trading window. Calibration from the v4 significance work: even 547 bear cycles peaked at
   NW-t +1.37. **No in-sample bull-mechanism test (I2) can reach significance — plan for the
   fullhist OOS to be the actual decider, and run capture/strategy layers on the bot's regime labels.**
4. **Strategy-layer runs must be post-fee-fix.** FEE_BPS_FILL=4.5 is now the bot default (audit:
   depth CSV was slippage-only; COST_BPS_LEG bypassed). Numbers produced before the fix (e.g. bull
   +4,250, the −0.67 factors backtest) are directionally fine but marginal candidates FLIP under
   fees via the REGIME_GATE trailing-PnL feedback (ret_rrboth −0.91 beyond the linear fee). Re-judge
   anything borderline at 4.5.
5. **Keep one whole-distribution sanity check.** Path A′ drops ΔIC entirely, but pred_disp (XS std
   of preds) feeds REGIME_GATE — a candidate that wrecks the untraded middle changes gate behavior
   even with a good tip. The strategy layer catches it; just don't skip that layer on tip-pass.
6. **The locked v4 default contradicts the framework's own rule.** The tracker still locks
   rr_both (single model, both legs) — a decision made at the capture layer (gross, gate-free).
   The strategy layer says two-book wins at fair fees: **+2.68 vs +2.38** (and ret two-book +2.68 vs
   ret rr_both +1.59). By rule #4 of this plan, the default should revert to two-book pending OOS.

## tip_accuracy_v2.py — built + calibrated (2026-07-06, other session)

Implements the fixes: two-book asymmetric selection at production K=1/2 (long ranked by LONG-book pred,
short by SHORT-book pred), per-cycle eligible universe (maturity>=180d + hygiene snapshot), bot-hysteresis
regime split, PAIRED per-cycle diff with daily-5d-block bootstrap CI as the primary significance test
(stride-6 t kept as diagnostic only — it discards 5/6 of cycles and sign-flips on the heavy-tailed tip
series), concentration/halves gates on the DIFF, shared within-cycle placebo, dispersion-normalized tip.
Verdict: REJECT if CI<0; PASS if CI>0 AND top3<40% AND halves AND placebo-clean; else NEUTRAL.
Overall verdict from pred-active regimes (side+bear) only.

**Validation ledger (5 candidates with known full-stack outcomes):**
| case | known (full stack, fee-fair) | tip-v2 verdict | call |
|---|---|---|---|
| res-target vs ret-target (two-book) | NEUTRAL (+6% PnL, ΔSh 0.00) | NEUTRAL | ✓ |
| res rr_both vs res two-book | LOSES −0.30 | NEUTRAL | ✗ (miss) |
| ret rr_both vs ret two-book | LOSES −1.09 | REJECT (side CI [−1088,−60]) | ✓ |
| wq036_bn vs lean | LOSES −0.42 | REJECT (bear CI [−2396,−161]) | ✓ |
| wq036_raw vs lean | LOSES −0.46 | NEUTRAL (bear CI [−2177,+11], marginal) | ✗ (miss, right direction) |

Read: **no false PASS** — the screen never green-lit a loser. Both misses are candidates whose loss
materializes at the CONSTRUCTION layer (gates/cost/turnover interactions), which is invisible at
entry-tip level by design — the documented blind spot, and exactly why NEUTRAL must route to the
full-stack replay (or deprioritization), never to adoption. Usage: REJECT → kill without backtest;
PASS → prioritize for full-stack; NEUTRAL → no tip evidence, judge by mechanism prior + strategy layer.
Known cosmetic: top3% blows up (>100%) when the net diff ~0 — read it only when the CI is significant.

## WARNING (2026-07-06): tip metrics are being used as a TUNING OBJECTIVE — this inverts the tool

The prune workflow (gen_prune_tip.py LOO scan scored by tipK2/tipK3 → gen_prune_variants.py
{no_ret3d, no_obvz, no_both}) selects feature drops by tip MEAN on the same in-sample window.
Running all three variants through the calibrated tip_accuracy_v2 (paired CI, concentration, halves,
placebo): **ALL THREE ARE NEUTRAL** — every CI crosses zero widely (e.g. no_ret3d bear +29.4 mean but
CI [−446,+2279]), top-3 shares 45–239%, halves flip on two of three. The raw tip means that look
attractive (+9 to +29 bps) are exactly what a mean-only scan selects on, and the significance
machinery says they are noise. Picking argmax-of-variants here reproduces the WINNER_16 pruning /
K4-margin failure pattern (in-sample metric selection → honest-validation loss).

Correct pruning protocol:
1. Shortlist drops by MECHANISM (near-zero standalone IC AND near-zero LOO tip change → droppable
   for simplicity), not by best tip lift. Feature-drop is a discrete change — the class that CAN
   generalize — but variant selection must still be honest.
2. tip_accuracy_v2 as kill/keep gate only (REJECT kills; NEUTRAL = no evidence — default keep the
   simpler model ONLY if the motivation is simplicity, never claim improvement).
3. ONE pre-registered variant → full-stack fee-fair replay → fullhist OOS.
4. Branch note: pruning is being tuned on the V0_LEAN+RR (rr_both) branch, but the best validated
   full-stack cell is base_both (V0_LEAN, NO RR, +3.29). Dropping ret_3d/obv_z while keeping RR
   optimizes within the inferior branch — the biggest defensible "prune" is RR itself, already
   validated at the strategy layer.

## ALPHA RE-EVALUATION under the new framework (2026-07-06, other session) — COMPLETE

All surviving alpha candidates re-scored at their integration points with the calibrated machinery.
I1 = WF retrain into the v4 frame (single model both legs, V0_LEAN+RR+<alpha>, residual target,
beta-neutral factor; gen_alphaI1_wf_preds.py → hl_i1_*) → tip_accuracy_v2 vs hl_tgt_res_long.

| candidate | origin frame | side tip Δ | bear tip Δ | verdict |
|---|---|---|---|---|
| +wq036 | pred-orthogonal winner (t +3.79) | +21.9 [−268,+1610] | −14.7 | NEUTRAL |
| +q158_IMIN60 | feature-frame survivor | −22.3 | −46.4 (halves −81/−12) | NEUTRAL (negative-lean) |
| +q158_RANK60 | feature-frame survivor | +3.3 | −7.0 | NEUTRAL (noise) |
| +alpha082 | a191 top / hot-bull | −6.7 | −10.0 | NEUTRAL (negative-lean) |
| +alpha065 | a191 borderline | −27.9 [−1924,+181] | −14.9 | NEUTRAL (negative-lean) |
| +alpha054 | side specialist | −26.1 [−1813,+31] | −15.4 | NEUTRAL (negative-lean) |

**I1 verdict: 0 PASS / 6 NEUTRAL, four negative-leaning.** No alpha improves the traded tip when
added to the v4 model; most nudge it DOWN — consistent with the wq036 full-stack lesson (a real
cross-sectional signal still dilutes ~170 per-symbol ridges). Under the framework's usage rule,
NEUTRAL earns NO integration and NO full-stack slot: the burden was on the alpha to show tip
evidence. **The published-alpha thread is now closed at the new framework's I1 gate as well.**

**I2 (alpha070_bn as bull short-ranker vs return_1d), capture layer, 137 hysteresis-bull cycles:**
incumbent short edge −69.6 bps/cyc (return_1d shorts LOST in this window's bull) vs candidate +5.7;
diff +75.3, halves +71/+80, but CI [−806,+3531] and top3 103% — DIRECTIONALLY supportive,
statistically nothing at n=137 (as predicted: bull-mechanism tests cannot be powered in-sample).
alpha070-I2 is the single surviving lead → decided ONLY by fullhist OOS.

Integration path for any future PASS: tip-v2 PASS → full-stack fee-4.5 frozen replay (run_v4_ab.sh
pattern) → fullhist 2022-26 OOS → wire into train_v4 artifacts. Nothing currently qualifies.

### VALIDATION of the "alphas don't help" verdict (2026-07-06) — controls run, verdict CONFIRMED + structural discovery
1. **Features entered the models** (not silently zeroed): standardized coef share 3.8–5.9%, median
   rank 9–14 of 17; picks changed on 38–57% of cycles.
2. **Evaluator detects and signs change correctly** (control A): a leaky 4h-oracle (0.15×current-bar
   label + noise) slashed pick overlap to 5% and the tip metric correctly PUNISHED it — current-bar
   residual mean-reverts over the next 24h (xs corr −0.028), so ranking by it hurts the 24h tip.
3. **Tip↔stack agreement**: best-looking candidate (+wq036) at full stack: +552 bps vs its rr_both
   reference, CI [−148,+174], halves flip, top3 423% — noise, exactly the screen's NEUTRAL.
4. **STRUCTURAL DISCOVERY (control B, oracle24): the I1 pipeline has a HORIZON BOTTLENECK.** A
   feature carrying IC≈0.15 on the EXACT 24h tip target (0.15×z(fwd24)+noise) transmits ~nothing
   through the pipeline (side −12, bear +8, all n.s.) — the ridge trains on the CURRENT-4h-bar label
   and loads features only by 4h-bar predictiveness (fwd24↔current-bar xs corr ≈ −0.03). Implications:
   (a) the alphas' I1 failure is OVERDETERMINED — weak signal AND a pipe that cannot pass
   24h-horizon information; (b) every historical I1 rejection has this ceiling — Path A cannot detect
   24h-horizon alpha, period; (c) the only way to exploit a 24h-horizon feature at the model layer is
   a LABEL-HORIZON change (train on fwd-24h residual) — a discrete, no-knob experiment worth its own
   validation (note: the current 4h label works because its predictions persist — the term-structure
   finding — so a 24h label is a hypothesis, not an upgrade by default).

### Per-alpha regime influence (leg-level Δtip, bps/cyc; bull CIs unreliable — only ~23 bull days)
| alpha | side ΔL/ΔS/Δtot | bear ΔL/ΔS/Δtot | bull ΔL/ΔS/Δtot | pattern |
|---|---|---|---|---|
| wq036 | +14/+8/+22 | −7/−8/−15 | −15/+21/+6 | side-bear seesaw, nets to ~0 |
| imin60 | −20/−3/−22 | −34/−12/−46 | +44/0/+44 | hurts earning regimes; value STRANDED in bull |
| rank60 | +16/−13/+3 | −1/−6/−7 | +60/−45/+15 | long-short seesaw, cancels |
| a082 | +8/−15/−7 | +10/−20/−10 | +11/−19/−8 | uniform SHORT-LEG DEGRADER |
| a065 | 0/−28/−28 | −3/−12/−15 | +12/−7/+5 | short-leg damage in side |
| a054 | −12/−15/−26 | −11/−5/−15 | −11/+38/+26 | hurts both legs side/bear; bull-stranded |
Two structural reads: (1) where these alphas DO help, it is disproportionately in BULL — the regime
where production ignores preds (I1 value structurally stranded; reinforces I2 as the only live
direction); (2) the dominant failure mode is SHORT-LEG degradation in side/bear (a082/a065/rank60)
— same mechanism as RR: reversal/positional signals tilt short selection toward squeeze-prone names.

## RECONCILIATION after the OOS run (2026-07-06 13:00+, other session)

The parallel session's OOS work (2023-01→2025-09, 33 folds, V4_REGIME_INVESTIGATION.md) supersedes
two in-sample conclusions of mine — updating the record:
1. **res base_both (+3.29 in-sample) is a WINDOW ARTIFACT.** OOS feature-tip shows resid_rev_3 is
   the STRONGEST long-book feature (+100/+84, both halves positive) — RR on the long book is real
   OOS alpha; its in-sample harm was specific to the 2025-10+ window. My best-of-6/single-window
   caveat fired exactly as written. **Recommendation reverts: keep the two-book split (RR on the
   long book), do NOT adopt base_both.**
2. **Residual target at OOS: defensible wash** (v4 ≈ v3 − 0.5 Sh under identical gating; ungated gap
   is entirely the bull-long leg that gates suppress). Consistent with my "consistency case only,
   net-neutral" verdict — their call (keep v3 production, v4 not adopted / not rejected) is right.
Also endorsed from their OOS batch: K_long=1/K_short=3 (stable rank-2..6 short band, net-Sharpe
rises with KS after fees — mechanism-grounded discrete change, bot-level net runs in flight);
regime-conditional feature tuning DECISIVELY CLOSED (14-config multi-agent search, 0 pass,
H1-decide/H2-validate caught the earlier bear-long false positive; ridge L2 shrinkage already
down-weights regime-reversing features better than binary pruning can).
Caveat flagged to them: the OOS window starts 2023-01 — the 2022 bear (deepest regime) is still
untested; worth one extension run before any adoption freeze.

### Independent VALIDATION of the OOS results (other session, 2026-07-06)
Reproduced from the pred books + panel, own code:
- **(A) books**: 552,082 rows × 4, v3/v4 keys identical ✓ (row-matched as claimed).
- **(B) ungated K=1/2**: v3 +7.1 / v4 +0.2 EXACT; bull-long attribution −3.6 / −17.8 EXACT ✓.
- **(C) gated**: K=1/2 +28.6 / +25.6 and K=1/3 +29.1 / +24.8 EXACT ✓.
- **(D) KS sweep**: net-mean peak at KS=3 (+25.4) EXACT; Sharpe-rises-to-KS4 ordering ✓.
- **(E) resid_rev_3 bear-long tip: +100.2 / +83.8 vs claimed +100 / +84 ✓** — the RR-long-is-real
  finding is solid; the base_both reversal stands.
Two findings: (1) **the gate regime uses UNSHIFTED r30 → 4h look-ahead in the gating decision**
(the bar's own close). PIT-shifted re-run: gated means drop ~1 bp/cyc (v3 K=1/3 +29.1→+27.8, ≈−4%),
SYMMETRIC across models — no ranking changes, but the harness should shift(1) the regime series
before any adoption-grade number is quoted. (2) The per-cycle Sharpe COLUMN is convention-dependent
(their values ≈1.6× mine on identical means/data, on top of the self-labeled 2.4× overlap inflation)
— use it only for within-table comparison, never as an absolute.

## INTERVENTION NOTE — for the tuning-validation loop (other session, 2026-07-06 ~14:45)

Reviewed section F + tuning_harness.py + toos_* sweep. The work is sound; three course-corrections
BEFORE applying the proceed plan:

1. **The OOS-bleed headline is now CONFIRMED ON THE PRODUCTION MODEL (v3 preds)** — I ran the missing
   cells (state: live/state/longtail/v3pred_oos_{full,vanilla}): v3-pred full v3_native stack OOS
   2023-25 = **−1.57 Sh / −12,528 bps / maxDD −13,101 / stop 80%**; v3-pred vanilla = −1.05 / −26,891.
   Statistically same as the v4-pred cells (−1.68/−10.5k, −1.30/−30.8k). Era-confinement is
   MODEL-INDEPENDENT. Use these as the production-reference rows in the harness (add
   WIN={"oos_v3": (hl_lean175_oos, hl_residrev_oos)}).
2. **"APPLY residual target" in the proceed plan contradicts the session's own verdict** (v3 ≥ v4 in
   every OOS cell; v4 not adopted). The harness's VANILLA is also v4/residual. Either revert the
   loop's reference to the v3 return-target preds or state explicitly why the non-adopted target is
   the deployment base. Do not let the target flip back in silently via the harness default.
3. **The tuning loop must not turn the OOS window into in-sample.** Agent-proposed env configs
   selected on 2023-25 performance = fitting the holdout (K2/K3/L-margin pattern). Required:
   decide/validate split INSIDE the loop (e.g. select on 2023-24, confirm on 2025), keep the still-
   ungenerated **2022 window as the untouched final holdout**, and pre-register config count.
   *(2026-07-07 later: 2022 SPENT on KEEPSET4 one-shot: FAIL — V4_PERFORMANCE §1; no further 2022 cells.)*
Endorsed as-is: gates-as-capital-preservation reading (3× loss/DD cut, now confirmed on v3 preds
too); toos_* single-lever bear-ramp isolation (right method for the DROP decision); the
"deployability, not more alpha tuning" meta-conclusion — the recent-era edge (+2.7 net) vs OOS bleed
(−1.6 capped by gates) tradeoff is a BUSINESS decision for the user, not a tuning target.

## MONITOR FLAGS on the tuning loop (other session, 2026-07-06 ~15:10)

1. **bear_ramp isolation cell is INVALID as run** — its OOS output is bit-identical to vanilla
   (−1.30 / −30,790, same per-year). The lever did not engage: BEAR_DEPTH_RAMP scales bear-entry
   gross inside the bear book path, which the VANILLA env (BEAR_K=0, BULL_MODE=default,
   STOP_SKIP_REGIMES=all) may not exercise. Do not record "bear ramp = no OOS effect" — re-run with
   its enabling mode (BEAR_MODE=equal / BEAR_K=2) or test as full-stack-minus-ramp.
2. **Section G's proposed lever ("PIT side-sign detection") is the mode-timing trap, third visit.**
   Prior art in THIS repo: vBTC L.1 mode meta-gate (nested-OOS +0.38, placebo p13), Phase DDI
   (per-cycle IC unpredictable from regime features, R²=0.005), and v3's own
   phase_policy_{meta,regime}_probe (learnability screen failed). The Oct+16k/Jan−7k side swing is
   real but the question is PIT-detectability, which has failed three times. If pursued anyway:
   mode-timing placebo (random side-sign flips of matched frequency), decide-2023-24/validate-2025
   split, and the untouched 2022 holdout are all MANDATORY before any adoption language.
   *(2026-07-07 later: 2022 spent — FAIL, gross-cap consequence active; see V4_PERFORMANCE §1/§7.)*
3. Section G's October-attribution method is good honest work — vanilla ex-Oct NEGATIVE is the
   right way to expose it. Carry the same ex-best-month check into any future headline.

## INTERVENTION — the "dd_stop REJECT (overfit)" verdict is WRONG by its own data (2026-07-06)

The loop's final verdict rejects dd_stop as "overfit: recent +1.59, OOS worse". The cell's own
numbers say the opposite:
- OOS PnL **−15,733 vs vanilla −30,790 — loss HALVED**; maxDD −17,079 vs −32,279 — HALVED;
- per-year: 2023 −3.5k (≈vanilla), **2024 −8.6k vs −19.7k, 2025 −3.6k vs −7.7k** — the ONLY lever
  in the sweep that improves the OOS economics era-uniformly;
- recent window: +1.59 vs +1.26 — ALSO better. "Overfit" (helps recent, hurts OOS) is factually
  inverted: dd_stop helps BOTH windows.
The only metric that worsened is OOS Sharpe (−1.48 vs −1.30) — a truncated-distribution artifact:
a stop on a losing strategy cuts std more than it cuts the (negative) mean, so Sharpe LOOKS worse
while every economic quantity (loss, DD, underwater time) improves. **Sharpe-on-negative-mean is
the wrong acceptance metric for a capital-preservation lever — judge stops by loss/DD/tail, not
Sharpe.** dd_stop should be KEEP (strongest generalizer in the table), and the still-unrun
gate+dd_stop combined cell (see STACK mis-wiring flag below) is the actual candidate keep-stack.
Everything else in the final verdict stands: regime_gate keep, bull_hedge overfit, bear cells dead
code (cause correctly identified), side-sign trailing-edge failure closure, BEAR_MODE=equal as the
new mechanism-grounded revisit.

## NOTE — bear anchor CONFIRMED both windows; run the combined keep-set (2026-07-06, other session)
v2_bear_mode_equal: OOS bear +5,006 (vs −684 untraded), total −24.8k, Sh −0.98 (best OOS of any
lever), 2023 POSITIVE (+1,770); in-sample ALSO better (+1.41/+15.4k). Second era-robust lever after
dd_stop, first that adds alpha. NEXT CELL THAT MATTERS: the combined keep-set
BEAR_MODE=equal + dd_stop(STOP_SKIP_REGIMES=bear, STOP_K_SIGMA=2.0) + regime_gate — never run
together; they act on disjoint failure modes and should compose. Then confirm on the v3 reference
(role doctrine: analyze on v4, confirm on v3).

## MONITOR FLAG — ss_trailedge_long is a NO-OP cell (2026-07-06, other session)
Its output is bit-identical to the regime_gate cell on BOTH windows (OOS −24,099 / INS +16,282 to
the bp) → REGIME_GATE_UNIV=side did not engage (third no-op env after bear_ramp/bear_k2 — check how
REGIME_GATE_UNIV is consumed in the bot before recording "side-scoping = no effect"). The task #27
ledger so far: fast side gate (W=90) = decisively WORSE (−2.00/−35.7k); side-scoping = untested.
Given ss_trailedge's clean failure, the honest closure "side-sign not PIT-detectable by trailing-
edge methods" only needs one valid side-scoped cell — fix the env wiring or implement the scope in
regime_mult directly.

## MONITOR FLAG — STACK cell is mis-wired (2026-07-06, other session)

STACK.json sets only the REGIME_GATE vars and does NOT override STOP_SKIP_REGIMES, so the vanilla
default ("side,bear,bull" = stop disabled everywhere) persists: STACK/oos output is bit-identical
to the regime_gate-only cell (−1.30 / −24,099 / stop_engaged 0%). **The assembled stack is missing
dd_stop — the sweep's only era-uniform lever** (dd_stop alone: OOS −15.7k, every year improved).
Fix: add "STOP_SKIP_REGIMES":"bear","STOP_K_SIGMA":"2.0" (as in dd_stop.json) to STACK and re-run
both windows before recording any keep-stack verdict. Expected: gate+stop combined should beat both
singles OOS (gate cuts side bleed, stop caps everything else).

## On the model/target (v4)

- **Residual target: endorse** (consistency case; net-neutral through the stack; the bear-tilted
  regime signature — +4.0 bps/cyc, NW-t +1.37 — is exactly where the label-decontamination mechanism
  predicts gains; judge it on that signature at fullhist OOS).
- **rr_both: revert** (see fix 6). resid_rev on the SHORT leg degrades side-regime short picks
  (entry edge 117.5→99.6 ret / →112.2 res bps/pos), and side is 57% of cycles.
- **K_short=3/K_long=1 (+0.44, −32% maxDD): plausible** — same discrete-architecture class as the
  historically-generalizing wins (K=3, V3.1) — but confirm it was (re)run post-fee-fix and hold it
  to the same OOS gate; asymmetric-K had a prior failure in the vBTC line (ASYMK rejected).

## AUDIT of CONVEXITY_V4_FLOW.md against the code (other session, 2026-07-07)

**VERIFIED CORRECT (code-checked):** regime thresholds ±0.10 + hysteresis N=3 (gap-free series);
mom30 computed internally with PIT .shift(1), MOM_WINDOW=180; BULL_MODE default=mom → bull ranks by
mom30 (sidealpha→pred) — the doc now correctly documents that vanilla/keep-set cells trade momentum
in bull, never the model; BEAR_MODE default=flat; side default=pred with SIDE_BETA_NEUT default=ON
(note: production v3_native sets it OFF explicitly — config divergence footnote); HOLD=6/24h sleeves;
overlay order conc_cap→DD-stop→REGIME_GATE→vol_target; VolNormStop (floor 0.40, 50%-heal, 90-bar
timeout); regime_gross_mult exit-lagged trailing-W binary; cost/funding formulas (previously audited).

**THREE ERRORS to fix:**
1. **§3/§9's load-bearing rationale FAILS replication.** "pred edge +40bps@4h → −21@24h reverses
   within the hold in bull" — on the CLEANED v4 books the bull pred edge is **−8.2 @4h and −36.1
   @24h: negative at BOTH horizons, no reversal**. (Side +13→+47, bear +20→+102 — persistence.) The
   design conclusion (don't rank bull by pred) survives, but the stated mechanism is wrong — likely a
   dirty-data (LITUSDT) or old-frame artifact. Fix the rationale to: "pred has no positive bull edge
   at any horizon; the bull tip is sign-inverted at the extremes (§M)."
2. **§8 PRODUCTION v3 "recent ~+3.0 / OOS ~−1.7" mixes fee frames** — +3.0 is pre-fee (fee-4.5 =
   +2.68) and −1.7 is the v4-pred OOS cell (v3-pred = −1.57). Should read **+2.68 / −1.57**.
3. **§8 "OPTIMIZED v4 = vanilla + bear_equal + regime_gate (only two logics survived)"** contradicts
   the tracker: dd_stop's rejection is the documented Sharpe-truncation error (§I), and bull0 passed
   their OWN dose-response. The four-lever KEEPSET4 (+2.17/−0.28 v4-frame; +2.77/−0.59 v3-ref) is the
   measured optimum (§N). The flow doc and the tracker currently DISAGREE about the optimized config —
   this inconsistency is the main source of system muddiness; the flow doc should import §N.
Minor: "OPTIMIZED +1.81" provenance unclear (C_bear_rg measured +1.94); REGIME_GATE_K semantics
undocumented; "+0.29 net long-beta", "+45/−6 mild/deep bull short" are unreplicated measurements —
mark as such or re-derive on clean books.
