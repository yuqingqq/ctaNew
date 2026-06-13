# Results Review — 3 agents on the full R0–R3 record (2026-05-19)

Verdicts: Methodology **RESULTS-NEED-REWORK** · Profitability
**GOAL-PARTIALLY-MET (over-claimed)** · Red-team **RESULTS-UNSOUND**.

## What stands (genuinely sound, all 3 agree)
- **R0 PASS** — `target_A` PIT-clean, prefix-causal exactly 0; not circular.
- **R1 reconstruction faithful** — reproduces V3.1 +2.229/−3445/7-9 exactly.
- **R2 both levers honestly refuted** — clean A/B, pre-registered bar applied,
  "else refuted" fired as written. No goalpost-moving in R2.
- **Cost-amortization-artifact concern retired** — edge survives realized √ADV
  (+2.13) and 3× tail-stressed cost (+1.96). Real positive.
- **drop-k random-symbol robustness real** — 90/90 draws positive.

## What is over-claimed / artifact (must re-initiate or retract)
1. **F1 (decisive, 2 agents): "diversified per-cycle risk, Herfindahl 0.094"
   is FALSE.** Herfindahl was computed on year-cumulative *signed* per-name
   PnL (mechanically diffuse). Honest per-cycle **gross-weight** Herfindahl
   ≈ 0.216 → ≈4.6 effective names. The bet IS concentrated. The R1 Diagnosis
   rewrite and R4 headline rest on this false metric.
2. **F2/F5 (decisive): "ex-VVV +1.99 ⇒ robust" is the adaptive-refill
   confound** (Round-2 kill F5, wrongly declared dead). R1b rebuilds the
   universe ex-VVV so filter_refill rotates onto the next tail name (AXS →
   ~41% of net). "Robust to VVV" = "concentration rotates to the next meme" —
   the same universe/meme dependence the user reported, relabelled.
3. **F3/F4 (decisive): portability +1.35 is unsound.** Costless, gate-free
   K=2 raw spread on `alpha_vs_btc_realized` — NOT the pre-registered
   trailing-288 PIT-β residual, NOT the deployable stack, target-mismatched
   vs the model's `target_A`; pooled CI ≈ [−0.86,+3.36] includes 0;
   contradicts STATUS Test-3 (−0.39 on the real stack). R3→R3b column swap is
   a silent spec substitution that produced the only positive portability
   number. Label also pools held-out group via `basket_A_fwd` (not
   label-disjoint).
4. Minor: doc number "+1.96 @9bps" is actually +1.92 (the +1.96 is the R2b
   tail-stress figure); deploy-fraction hand-loosened 0.3→0.5–0.7 post-hoc.

## Re-initiation plan (per the user's process)
- **R1c** — recompute the honest per-cycle gross-WEIGHT Herfindahl; quantify
  the ex-VVV refill-rotation (rebuilt book's new top-name share). Replace the
  "diversified" narrative with the truth.
- **R3c** — the decisive one. Implement the pre-registered trailing-288 PIT-β
  to-BTC residual; make the label group-disjoint (basket excludes held-out
  group); run the FULL deployable stack (rolling-IC + conv_gate + refill +
  PM_M2 + 6-sleeve + cost) on held-out UNSEEN symbols, no sym_id; report
  regardless of sign. This is the true apples-to-apples test of Test-3.
- **R4 rewrite** — honest, de-hyped: what survives (random-drop & cost
  robustness, R2 negatives) vs what does not (per-cycle concentration,
  refill-rotation = meme dependence, true out-of-universe portability).
- Doc fixes: +1.92; respect script's 0.3 deploy fraction unless rigorously
  justified.

Round-3 agent ids: methodology `ab8d72db4f3e7d73c`, profitability
`a9a7d8ec15043351c`, red-team `a307b293e92924858`.
