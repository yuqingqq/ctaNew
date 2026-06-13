# R4 — Synthesis & Decision (2026-05-19, HONEST FINAL)

> v1 (`R4_SYNTHESIS_v1_overclaimed.md`) was retracted after the mandated
> 3-agent results review (`reviews/ROUND3_results_review.md`) found the
> "deployable, robust, portable +2.0" headline over-claimed. The decisive
> components were re-initiated correctly (R1c, R3c). This v2 reports the
> corrected, honest conclusion. No goalpost-moving: the re-initiations
> CONFIRMED the reviewers and reversed the headline.

## Bottom line (honest)

**The research goal — a profitable system that is universe-portable and not
dependent on single-name meme concentration — is NOT met. The honest answer
to the user's original question is: the LGBM edge genuinely does NOT replay
on a different symbol universe.** This is a decisive, well-evidenced negative,
which is the correct and valuable outcome of the process.

## What is TRUE (survives honest scrutiny)

1. **R0: the target is PIT-clean.** `target_A` recompute matches 1.1e-5·std;
   prefix-causal exactly 0. No look-ahead in the label. Solid.
2. **The in-universe V3.1 +2.23 is real but it is a concentrated vol-convexity
   bet, not broad alpha.** Honest *per-cycle gross-weight* Herfindahl ≈ **0.19
   (~5 effective names)** — concentrated (R1c). ~80% of cumulative net PnL is
   one low-float meme (VVV). Removing top names does not diversify it — the
   filter_refill **rotates** concentration to the next tail name
   (VVV→AXS→PENDLE) and Sharpe degrades +2.06→+1.89→+1.15 (R1c).
3. **R2: both profit levers decisively refuted (clean OOS).** rvol/ret as
   model features = −1.84 lift; longer holds monotone-worse. The cost-
   amortization-artifact concern is retired (survives realized √ADV +2.13 and
   3× tail-stressed cost +1.96) — a genuine positive, but it only hardens an
   in-universe edge that does not port.
4. **R3c (decisive, proper test): the deployable stack does NOT port to
   unseen symbols.** Full stack, costed, no sym_id, group-disjoint
   label+features, pre-registered trailing-288 PIT-β-to-BTC residual eval,
   on symbols the model never trained on: **pooled Sharpe −0.33, 2/5 groups
   positive** (g0 +0.54, g1 +0.18, g2 −0.33, g3 −1.54, g4 −0.67). This
   reproduces the prior honest STATUS Test-3 (no-sym_id full-stack = −0.39).

## What was OVER-CLAIMED in v1 and is now CORRECTED

| v1 claim | corrected truth (script) |
|---|---|
| "diversified per-cycle risk, Herfindahl 0.094" | per-cycle gross-weight H ≈ 0.19, ~5 effective names — **concentrated** (R1c) |
| "robust: ex-VVV +1.99" | concentration **rotates** to AXS (37%) then PENDLE; Sharpe decays +2.06→+1.89→+1.15 (R1c) |
| "universe-portable, +1.35 out-of-universe, not-portable belief refuted" | proper full-stack costed test = **−0.33** on unseen symbols; **not portable** (R3c). v1's +1.35 was a costless, gate-free, target-mismatched K=2 proxy (red-team F3) |
| "R1 alone is a deployable system" | only true *in-sample, in this exact 51-universe*; it is universe-overfit |

The drop-k "90/90 positive" is real but is itself largely the refill-rotation
mechanism (any remaining tail names sustain the convexity) — it is robustness
to *random subsetting of the same universe*, NOT portability to a *new* one
(R3c shows the latter fails).

## Honest decision / recommendation

The user's goal (profitable + universe-portable + not meme-concentrated) is
**not achievable from this line as scoped.** Ranked:

**▶ RECOMMENDED — Option A: treat as a decisive honest negative.** On free 4h
Binance-perp residuals, no universe-portable alpha exists at deployable cost
(R3c −0.33 on unseen symbols, reproducing the prior Test-3 −0.39). Every
in-scope construction/feature/horizon lever is now refuted with clean OOS
(R2). This matches the already-closed linear β-residual line. Stop spending
effort here; do not deploy.

**Option B (only if user explicitly wants a niche tactical bet — does NOT
meet the stated portability goal):** the +2.0–2.2 in-universe Sharpe is real
but is a fragile ~5-effective-name meme-convexity bet that REQUIRES the
specific 51-symbol composition incl. low-float pump names. It **fails the
user's universe-portability requirement** (that is precisely what R3c proves).
Deployable only as: tiny size, hard kill-switch (cum-DD −6,265 bps), live
single-name dollar cap, accepting it decays on composition drift / meme
delisting. Not a robust system; not scalable.

**Option C — scope change (un-refuted, needs a user decision):** genuinely
orthogonal data (on-chain/Glassnode) or a fundamentally different signal/
horizon. Out of current scope; not executed here by design.

## ▶ Decision item (not a footnote): live bot tests neither

`live/vBTC_paper_bot.py` ships **K=4, no sleeve** — neither the in-universe
research stack nor anything validated here. **Do not run it for forward
evidence under any option:** it would generate misleading data that validates
neither the in-universe strategy nor this honest negative. Reconciling or
retiring it is a required action before any paper-forward.

## Process integrity

Plan: 3 rounds / 9 agent reviews before any run. Results: 3-agent review
caught the v1 over-claim; the offending components were re-initiated (R1c,
R3c), not rationalised; the corrected results reversed the headline and are
reported straight. No gate was moved. Audit trail: `PLAN.md`,
`reviews/ROUND{1,2,3}_*`, `results/R*_FINDINGS.md`, `results/R4_SYNTHESIS_v1_overclaimed.md`,
all `R*_results.json`, `scripts/R*.py`.
