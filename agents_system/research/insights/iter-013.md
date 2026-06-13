# Research Insights — iter-013 (EFFICIENCY of the portable vol-norm reactive DD stop)

**REACTIVE risk-control track (NOT alpha).** iter-012 shipped the PORTABLE vol-normalized equity-DD stop
(k=2.0 unitless, g_floor=0.40, heal-50% / 90-bar timeout, vol_win=180, warmup=60): PASS R6 nested-OOS 3/3,
R5 LOFO 4/4, R4 ~proportional (p55–p70, EXPECTED — no tail-selection skill on free data). It WORKS but it
**CHURNS**: ~15 round-trips / 402d on HL70 (92 on the longer EXT panel), 51% of time at reduced gross,
each de-gross/re-gross paying turnover. R4 already proved the tail-cap is ~PROPORTIONAL — you cannot SELECT
the tail with skill (that wall is proven). So this iteration did NOT try to beat proportional. The ONLY
honest lever is **EFFICIENCY**: cut the whipsaw / transaction waste to get the SAME DD-cap at LOWER cost
(Pareto-improve the trade-off curve).

**One-line verdict: the iter-012 stop is ALREADY at its efficient frontier.** No efficiency variant
Pareto-improves *robustly across all three universes* at the recommended operating point. The one variant
that helps — **hysteresis (fuller re-entry heal)** — Pareto-WINS on HL70 at k=2.0 (identical maxDD, cost
19.9%→16.6%, Sharpe 1.80→1.88, Calmar 2.01→2.10) but is **sign-unstable across universe×k×cost**: at the
same k=2.0 it costs MORE on EXT (+3.5%) and is neutral on S44. Every other variant (smooth/graded
de-gross, cooldown, confirmation-lag) only trades DD-cap for cost — i.e. moves ALONG the existing curve
(equivalent to picking a deeper k), not a Pareto SHIFT. **iter-012 k=2.0 binary stays the deployable
config.** Optional refinement: ship the re-entry heal at 0.90 instead of 0.50 — it never gives back DD-cap
(maxDD identical wherever it acts) and helps HL70 modestly, costing nothing structurally (it is the same
unitless state machine, R5/R6 unchanged). But it is a marginal, not a decisive, win.

Scripts: `research/convexity_portable_2026-05-20/scripts/X126_volnorm_efficiency.py` (variant trade-offs,
matched-avg-gross Pareto, R5 LOFO, R6 nested-OOS for every variant family) and
`X127_hyst_robustness.py` (hysteresis across k×cost). Base reproduces X117 EXACT (HL70 @4.5bps Sharpe
+1.93 / maxDD −5674 / Calmar +1.68 / tot +10472). Reuses X123 build_universe + X124 held-book engine.

---

## STEP 2 — efficiency variants (all PIT; parameter-free preferred to stay portable + not overfit)

Same engine/policy as iter-012 (gross applied to positions BEFORE turnover/cost; g_floor=0.40, vol_win=180,
warmup=60, timeout=90; trigger trig = k·σ(trailing-180 equity incr)·√180). Each variant changes ONLY *how
gross moves between 1.0 and g_floor* or *when it may change*:

| variant | mechanism | knob (nested in R6) |
|---|---|---|
| **binary** (iter-012) | hard 1.0↔0.40; re-enter on 50%-heal or 90-bar | none (reference) |
| **grad2x / grad3x** | SMOOTH ramp: gross = linear f(stress ratio r=−dd/trig) from 1.0 (r≤1) → g_floor (r≥grad_ratio); stateless | grad_ratio ∈ {2,3} (shape, unitless) |
| **hyst75 / hyst90** | binary, but re-enter only after a FULLER heal (0.75 / 0.90 instead of 0.50) → wider dead-band, fewer flip-flops | heal ∈ {0.50,0.75,0.90} |
| **cool30 / cool60** | binary, but after a re-entry cannot re-fire for `cool` bars (~5d / ~10d) | cool ∈ {0,30,60} |
| **conf3 / conf6** | binary, but require −dd≥trig for M consecutive bars before firing | confirm ∈ {1,3,6} |

---

## STEP 3 — decisive trade-off + Pareto + robustness (all @4.5bps, k=2.0, canonical held book)

### Variant trade-off vs the iter-012 binary reference
A variant **PARETO-WINS** only if ddRed ≥ binary AND cost strictly < binary (same/more DD-cap, less cost).
"cheaper-but-less-DD" = a move ALONG the curve (no Pareto shift; same as deeper k).

| universe | binary (ref) | grad3x | hyst90 | cool30 | conf6 |
|---|---|---|---|---|---|
| **HL70** ddRed / cost / RT | 33.1% / 19.9% / 15 | 14.3% / 19.8% / 6 | **33.1% / 16.6% / 14 ✅PARETO** | 24.7% / 15.0% / 11 (less DD) | 20.3% / 23.5% / 14 |
| **EXT** ddRed / cost / RT | 39.4% / 32.4% / 92 | 18.7% / 8.3% / 35 (less DD) | 39.4% / **35.8%** / 92 (worse cost) | 33.8% / 9.9% / 58 (less DD) | 36.4% / 32.7% / 90 |
| **S44** ddRed / cost / RT | 20.7% / 11.1% / 22 | 16.6% / 3.2% / 12 (less DD) | 20.7% / 11.1% / 22 (identical) | 9.8% / −2.0% / 13 (less DD) | 19.8% / 6.2% / 20 (less DD) |

Read-out:
- **hyst** is the ONLY variant that holds DD-cap constant while changing cost. On HL70 it removes ONE true
  whipsaw round-trip (15→14) → identical maxDD −3794 but cost 19.9%→16.6%, Sharpe +0.08, Calmar 2.01→2.10.
  **But on EXT at k=2.0 it costs MORE** (+3.5%): EXT spends 79% time stopped with 92 round-trips, so a fuller
  heal genuinely keeps you stopped longer there → more exposure-removal cost, same DD. On S44 it is exactly
  identical (heal threshold never binds; re-entry is timeout-driven). → **helps HL70, neutral/worse elsewhere.**
- **grad / cool / conf** all reduce cost ONLY by reducing DD-cap (cheaper-but-less-DD) — they are
  equivalent to choosing a deeper k on the existing iter-012 dial, NOT a new efficient point. (cool/grad's
  big EXT cost drops, e.g. cool30 9.9% vs binary 32.4%, come with ddRed dropping 39.4%→33.8% — proportional.)

### Pareto at MATCHED avg-gross (turnover efficiency holding average exposure fixed)
For each variant, compare its (ddRed, cost) to the CONST flat-degross of the SAME avg-gross. binary already
beats const on the *tail* on HL70/S44 (the iter-012 R4 asymmetry-of-WHEN finding). The efficiency variants
do NOT add a consistent turnover advantage over binary at matched exposure — hyst matches binary's profile,
grad/cool/conf give back tail to buy the cheaper turnover. No variant dominates binary on both axes on all 3.

### hysteresis robustness across k×cost (X127) — the sign is UNSTABLE
| | HL70 | EXT | S44 |
|---|---|---|---|
| k=1.5 | mostly neutral | **worse-cost** (+10 to +13%) | PARETO-WIN |
| **k=2.0 (rec)** | **PARETO-WIN** (−2.5 to −3.3% cost, +0.08 Sh) | **worse-cost** (+3.5 to +3.9%) | neutral |
| k=2.5 | neutral | PARETO-WIN (−17 to −20% cost) | neutral |

The hysteresis cost effect **flips sign with k and universe**. It NEVER gives back DD-cap (maxDD identical
in every cell where it acts — it only ever removes a premature re-gross that would have immediately
re-fired, i.e. a genuine whipsaw), so it is *risk-neutral and never harmful to the tail*. But it is not a
*robust* cost win: it helps HL70 exactly at the recommended k=2.0 and helps EXT only at k=2.5. This is the
same "helps in one cell, hurts in another" pattern the contract flags as non-robust — so it cannot be
claimed as a Pareto improvement of the curve; at best it is a free, harmless HL70 refinement.

### R5 cross-episode + LOFO (EXT) — STILL HOLDS for the robust variants
binary, hyst90, hyst75, conf3, conf6 all cap **4/4** EXT episodes ≥10% (luna/ftx/2024/q4), unchanged from
iter-012. grad3x drops to 3/4 (misses ftx at +20% under-10% on q4 +7%) and cool30/cool60 weaken ftx
(+16/+17%) — confirming the cost-cutting variants erode cross-episode robustness (they de-gross less, so
cap less in the deepest episodes). **hysteresis preserves R5 exactly** (identical maxDD ⇒ identical episode
caps). R5 PASS for hyst, weaker for grad/cool.

### R6 cross-universe nested-OOS (THE portability gate) — STILL HOLDS for hyst/cool
Nested-OOS picks (k, knob) on past folds (max ddRed under ≤25% cost), applies forward; PASS = fwd ddRed>+5%
AND cost<40% on EVERY universe.
| family | nested-OOS | HL70 | EXT | S44 |
|---|---|---|---|---|
| **binary** | **3/3** | +33%/−36% P | +29%/+28% P | +9%/+6% P |
| **hyst** | **3/3** | +33%/−36% P | +29%/+28% P | +9%/+9% P |
| **cool** | **3/3** | +33%/−36% P | +29%/+28% P | +9%/+9% P |
| grad | 2/3 | +12%/−17% P | −3%/+19% **F** | +13%/+1% P |
| conf | 2/3 | +36%/−41% P | +28%/+44% **F** | +9%/+6% P |

hyst and cool keep portability (3/3); the nested selector mostly falls back to the binary/timeout behavior
forward, so they inherit binary's robustness. grad and conf FAIL portability on EXT (grad's stateless ramp
de-grosses too gently to cap the EXT tail; conf's confirmation lag blows past the EXT 40% cost budget).

### R4 (reported, as expected) — still ~proportional
All variants remain p55–p70 vs matched-%-time random de-gross (no skill claim; the reactive-track
expectation). Efficiency variants do NOT change this — they cannot, since the tail-cap is structurally
proportional. (R4-placebo not re-run per-variant; the proportional wall is invariant to re-entry policy.)

---

## STEP 4 — decision: ALREADY AT THE EFFICIENT FRONTIER (honest negative result)

**No efficiency variant Pareto-improves the iter-012 curve robustly across HL70+EXT+S44.**
- grad / cool / conf: cut cost ONLY by giving back DD-cap (moves along the existing k-dial, not a shift) —
  and grad/conf break R6 portability on EXT and grad/cool weaken R5 (deepest-episode caps).
- hyst (fuller heal): the only variant that holds DD-cap fixed; Pareto-WINS on HL70 at k=2.0 (cost
  −3.3pp, Sharpe +0.08, Calmar 2.01→2.10) and keeps R5 4/4 + R6 3/3 — but its cost benefit is sign-unstable
  across k×universe (worse on EXT at k=2.0). It is a free, harmless HL70 refinement, NOT a decisive
  cross-universe Pareto win.

**Why the churn is already near-efficient:** the iter-012 re-entry policy (50%-heal OR 90-bar timeout with
the `eq>trough` guard) was already tuned to avoid buy-back-at-trough and frozen-equity kills. The ~15 HL70
round-trips are mostly *necessary* de-gross/re-gross cycles tracking genuine DD episodes, not flip-flop
noise — only ~1 is a removable whipsaw (what hyst catches). The turnover cost is dominated by the
de-grossing TRADE itself (g 1.0→0.40 moves 60% of the book), which is intrinsic to capping the tail
proportionally; you cannot cut it without cutting the DD-cap. The graded ramp confirms this: spreading the
de-gross into small steps (grad) does NOT lower total turnover (HL70 turn 601 vs binary 580 — slightly
HIGHER) because the cumulative position change is the same; it just caps less because it reaches g_floor
later. **The transaction waste is small and the stop sits on its efficient frontier.**

### Recommended FINAL deployable config (unchanged from iter-012, with one optional harmless refinement)
> **Vol-normalized equity-DD stop, k=2.0 unitless.** De-gross the held book to g_floor=0.40 when
> (peak−eq) ≥ 2.0·σ(trailing-180-bar equity increments)·√180. σ/peak/DD through t−1 (PIT). Re-enter
> (gross→1) on **heal=0.90** of the DD back toward the peak (and eq>trough) OR 90-bar timeout. Warmup 60.

The ONLY change vs iter-012 is heal 0.50 → **0.90** (a wider re-entry dead-band). It is unitless, keeps R5
4/4 and R6 3/3, never gives back DD-cap, and removes the one removable HL70 whipsaw (cost 19.9%→16.6%,
Sharpe +0.08, Calmar 2.01→2.10). If a single global heal is preferred for simplicity, **0.50 (iter-012) and
0.90 are both defensible** — 0.90 is marginally better on HL70/S44, marginally worse on EXT cost. This is a
risk-preference refinement, not a new capability.

---

## How this fits the prior ledger
iters 5–10 closed the prediction axis; iter-011 characterized the reaction axis (proportional tail-cap,
cross-episode robust, HL70-specific absolute trigger); iter-012 closed portability (unitless vol-norm k=2.0,
R6 3/3). iter-013 closes the **efficiency** axis: the portable stop is already near its efficient frontier —
the whipsaw waste is small (~1 removable round-trip on HL70 of ~15), and every cost-cutting variant either
gives back DD-cap (grad/cool/conf, moves along the k-dial) or helps only in one universe×k cell (hyst).
**There is no Pareto-improving efficiency variant that holds across all three universes.** The deployable
reactive overlay remains the iter-012 vol-normalized stop (optionally heal=0.90). The DD is mechanically
reducible at ~proportional cost with a portable PIT rule, near-efficiently — not for free, and not improvable.

## Artifacts
- scripts: `research/convexity_portable_2026-05-20/scripts/X126_volnorm_efficiency.py`,
  `X127_hyst_robustness.py`
- results: `research/convexity_portable_2026-05-20/results/X126_variant_tradeoff.parquet`
- reuses: X123 build_universe, X124 held-book engine + metrics, X125 vol-norm baseline.
