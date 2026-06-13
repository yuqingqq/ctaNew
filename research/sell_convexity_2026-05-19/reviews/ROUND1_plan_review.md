# Sell-Convexity Plan v1 — 3-Agent Review (2026-05-19) — KILLED, RE-INITIATE

Verdicts: Methodology **FUNDAMENTALLY-FLAWED** · Profitability
**NEEDS-REFOCUS** · Red-team **DO-NOT-PROCEED**. Convergent decisive findings
(the errors are the author's, caught before any compute):

1. **S0 misrepresents C0pre (fatal).** `C0pre_decisive.json` verdict is
   literally **"B: LINE CLOSED — pure volatility detector"**: AUC_pos 0.669 ≈
   AUC_neg 0.701 (gap 0.032 < its own 0.04 bar), **AUC_abs 0.753 > both**.
   The plan cherry-picked the down-leg AUC and presented a LINE-CLOSED
   *symmetric volatility* detector as a *negative-skew* validation. Shorting
   it = short the high-vol/rotating-meme tail (a short-straddle proxy), NOT
   harvesting a negative-skew premium. Same vol-detector confound that
   already closed the convexity line.
2. **Funding model wrong + not binding (fatal).** Binance USDM: when
   funding>0 longs pay shorts → a short *receives* funding when funding>0;
   the plan had the sign inverted. Empirically the primed/pumped cohort has
   *negative* mean funding (≈ −1.5 to −2.4 bps/8h) and panel funding is tiny
   (sub-1 bps/4h) — funding is NOT the binding constraint and the plan's
   "friction-killed by funding" thesis is false on its own panel. Also
   "realized funding over the hold" = look-ahead, and the locked engine
   (R1.aggregate_capped) has no per-position carry path at all.
3. **Re-derives closed work (fatal).** The R3c portable stack already builds
   a short leg every cycle (short lowest-pred, β-neutral, costed) and ported
   to **−0.33**. "Short the cohort by a signature collinear with vol features
   already in pred" ≈ that closed short leg re-ranked → re-derivation,
   explicitly out-of-scope.
4. Power: +0.5/LCB>0 gate unreachable on ~5-group/0.74y; "indeterminate"
   risks unfalsifiability.

## Decision
Sell-convexity v1 KILLED. NOT a candidate for revision-and-proceed (the core
thesis object — a negative-skew detector — does not exist; C0pre proved it's
a symmetric vol detector, LINE CLOSED). Per the loop: this iteration is a
completed honest negative. Re-initiate with the next genuinely-distinct,
non-redundant hypothesis (must NOT be the vol detector, the −0.33 short leg,
or the closed return-forecast ceiling). Agent ids: meth `ac7b9bf75091cb0f4`,
prof `a4ea60579246cf30d`, red `a8a6efe90f14008bd`.
