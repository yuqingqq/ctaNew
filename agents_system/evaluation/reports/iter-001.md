# Evaluation Report — iter-001

**Change:** De-lever-ONLY, parameter-free realized-vol throttle on the held book
(`X119_delever_throttle.py`). Scale all book weights by
`s_t = clip(expanding_median(rv) / rv_t, FLOOR=0.3, 1.0)`, hard cap 1.0 (never levers up),
`rv_t` = trailing std (W=42) of the book's own per-cycle net PnL, HOLD-lagged PIT.

**Verdict: REJECT** — fails G2, G4, G6 on the production universe (HL70).
The objective is **Calmar on HL70**; the throttle *lowers* HL70 Calmar from +1.68 to +0.91.

---

## Metrics table (@4.5 bps)

| universe | arm | Sharpe | maxDD (bps) | Calmar | totPnL (bps) | %pos |
|---|---|---|---|---|---|---|
| **HL70** (PRIMARY) | base / current_best | +1.93 | −5,674 | **+1.68** | +10,472 | 39.9 |
| **HL70** | throttle | +1.38 | −5,627 | **+0.91** | +5,636 | 39.9 |
| HL70 | Δ | −0.55 | +0.8% DD | **−0.77** | −4,836 | — |
| S44 (robustness) | base | +1.84 | −4,170 | +2.10 | +25,620 | 44.8 |
| S44 | throttle | +1.89 | −3,363 | +1.93 | +18,981 | 44.8 |
| S44 | Δ | +0.04 | +19.4% DD | −0.17 | −6,639 | — |

Pre-registered targets (research handoff): HL70 maxDD reduction **≥20%** (≤ −4,540 bps),
Calmar **≥ +2.0**, Sharpe **≥ +1.73**. HL70 result: DD reduction **+0.8%**, Calmar **+0.91**,
Sharpe **+1.38** — misses all three. On S44 the DD target is nearly met (+19.4%) but the
secondary universe does not satisfy the objective, which is HL70 Calmar.

---

## Gate-by-gate

### G1 — Look-ahead audit: **PASS**
Review handoff PASS (fix-round 1). Throttle window is HOLD-lagged PIT (uses only `pnl[:t−HOLD+1]`),
expanding-median reference, hard cap 1.0 (parquet confirms scale ∈ [0.30, 1.00], zero values >1).
No IC-type leakage signal present (N/A here — this is a sizing overlay, not a feature). Confirmed.

### G2 — In-sample objective (Calmar): **FAIL on HL70**
- HL70 Calmar +1.68 → **+0.91** (Δ −0.77). Necessary condition (Calmar > current_best) NOT met.
- maxDD barely moves on HL70 (−5,674 → −5,627, +0.8%); Sharpe drops −0.55 (3× the ≤0.2 budget).
- S44 Calmar +2.10 → +1.93 (also down). On neither universe does Calmar improve.

### G3 — Nested-OOS: **WAIVED (legitimate)**
W=42 and FLOOR=0.3 are fixed module constants (review verified lines 43–44; used once each, no sweep).
Reference is the running expanding median, not an in-sample-chosen value. Cap=1.0 is structural.
No tuned/selected parameter ⇒ G3 waived per contract. (FLOOR sensitivity, if characterized, would be a
robustness band, not a selection — not run, since the change already fails G2/G4 on production.)

### G4 — Matched random-timing placebo (THE KILL TEST): **FAIL on both universes**
Same multiset of `scale` values, timing shuffled across cycles, 500 seeds, applied to `pnl_base`.
Need real ≥ p95 to show *timing* (not just lower average gross) drives any DD benefit.

| universe | real Calmar | placebo median / p95 / max | **real Calmar pctile** | real maxDD pctile |
|---|---|---|---|---|
| HL70 | +0.91 | +1.64 / +2.12 / +2.59 | **p0** | p7 |
| S44 | +1.93 | +2.07 / +2.55 / +3.00 | **p30** | p75 |

Decisive: on HL70 a *random-timing* de-lever of the same average magnitude does **strictly better**
(median placebo Calmar +1.64 vs real +0.91; real is below the entire placebo distribution, p0).
On S44 the real throttle ranks p30 (Calmar) / p75 (maxDD) — well short of p95. The DD reduction is
explained entirely by "run smaller on average," with the throttle's *timing* actively unhelpful.
This is exactly the failure mode the research handoff pre-registered as the kill test → REJECT.

### G5 — Per-fold robustness + LOFO: per-fold direction OK, but lift is hollow
- HL70: DD improved in 6/7 folds; but the *aggregate* DD improvement is only +0.8% and Sharpe falls in
  most folds. LOFO: aggregate DD-improvement is ~+0.8% regardless of which fold is dropped (rises to
  +5.5% only when the worst fold f5 is removed) — i.e. there is essentially no DD benefit to concentrate.
- S44: DD improved in 7/8 folds (+19.4% aggregate), robust to LOFO (+11.4% to +25.5%). The DD cut on S44
  is genuinely broad-based — but G4 shows it's a magnitude effect, not timing skill.
- G5 is not the binding gate; the per-fold direction is fine, the magnitude/skill is the problem.

### G6 — Paired CI (block-bootstrap by fold, 2000 boots): **adverse on HL70**
- HL70 Sharpe-diff (throttle − base): mean −0.47, CI95 **[−1.06, −0.10]** — clears zero on the *negative*
  side (throttle significantly worse on Sharpe). totPnL-diff CI95 [−10,754, −640] (significantly negative).
- S44 Sharpe-diff: mean +0.05, CI95 [−0.13, +0.30] — crosses zero (no Sharpe edge). totPnL-diff
  [−13,345, −370] (negative, as expected for an exposure trim).
- For a pure DD trade a negative PnL-diff CI is acceptable *if* Calmar improves — but Calmar does NOT
  improve on either universe, and on HL70 the Sharpe-diff CI is significantly negative.

### G7 — Universe robustness: **FAIL on production (HL70)**
The improvement must hold on HL70. It does not (Calmar −0.77, DD −0.8% only). S44 shows a DD cut but
fails G4 there too, and S44 is the secondary universe. A change that helps S44 (marginally) but not HL70
does not satisfy G7. REJECT — and this is a clean, informative result.

### G8 — Cost realism: throttle loses at every cost level
| cost | HL70 base Calmar | HL70 throttle Calmar | S44 base | S44 throttle |
|---|---|---|---|---|
| 1.0 bp | +1.99 | +1.17 | +2.48 | +2.37 |
| 3.0 bp | +1.81 | +1.02 | +2.26 | +2.12 |
| 4.5 bp | +1.68 | +0.91 | +2.10 | +1.93 |
The throttle's Calmar is below base at all three cost levels on both universes; the deficit does not
close at low cost (the hoped-for "cost improves under de-lever" effect is swamped by lost gross alpha).

---

## Why REJECT
The objective is to raise **Calmar on HL70**. The throttle *lowers* HL70 Calmar (+1.68 → +0.91) and
barely touches the HL70 drawdown (+0.8%), while costing −0.55 Sharpe. The S44 DD reduction (+19.4%) is
attractive at first glance but (a) S44 is the robustness, not production, universe and its Calmar still
falls, and (b) G4 shows the S44 DD cut is a *magnitude* effect (p30/p75), not timing skill — a random
de-lever of the same average size does as well or better. On HL70 the real throttle is *below the
entire* random-timing placebo distribution (p0 Calmar). The pre-registered kill test (G4) fired exactly
as the research handoff warned.

## Insights for next research cycle
1. **HL70's deep DD is not P&L-vol-clustered the way S44's is.** The book-level realized-vol throttle
   keys on the book's own PnL variance; it bites on S44 (broad DD cut, 7/8 folds) but is inert on HL70
   (+0.8%). The HL70 −57% drawdown is therefore NOT a "vol runs hot during the grind" episode that a
   variance-targeting overlay can attenuate — its drawdown structure differs from S44's. Drawdown
   anatomy (X98) was derived largely on the 44-sym base; **re-do the DD anatomy on HL70 specifically**
   before proposing the next risk overlay. The two universes have structurally different DD mechanisms.
2. **"Lower average gross" is not alpha.** G4 cleanly separates timing skill from de-leveraging. Any
   future sizing overlay must beat the matched-magnitude random-timing placebo at ≥p95 — the S44 result
   (p30) shows a 19% DD cut can be entirely a gross-down artifact. The honest alternative to this trade
   is simply a flat gross reduction, which would give the same S44 DD profile with less machinery.
3. **De-lever-only is confirmed dead on HL70**, alongside the previously-killed lever-up (X97) and tuned
   throttles (X99/X100). The volatility-scaling family is exhausted on the production universe. Pivot
   away from book-level vol targeting toward (a) HL70-specific DD-onset detection (what actually
   precedes the −57% grind on HL70 — regime, breadth, correlation spike?), or (b) the construction layer
   (entry/selection during the identified bad regime), rather than uniform book scaling.
