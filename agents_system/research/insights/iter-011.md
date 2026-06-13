# Research Insights — iter-011 (REACTIVE damage-control: equity-DD stop + fast-flag de-risk)

**NEW TRACK (reactive risk-control, not alpha).** Human directive: stop trying to PREDICT the alt-bear
(iters 5–10 proved no free signal leads). Build a MECHANICAL, REACTIVE damage-control layer that
de-grosses when a drawdown is ALREADY HAPPENING and caps the tail for live capital preservation —
accepting it cannot forecast and will whipsaw. Verdict here is a *characterized DD-vs-cost trade-off +
recommended risk config*, NOT a binary alpha ADOPT. Gates = R1–R7 (evaluation_contract.md).

**One-line verdict:** A reactive **equity-drawdown stop** (de-gross to 40% gross when the strategy's own
equity is ≥X bps below its running peak, re-enter on 50%-heal-or-90-bar timeout) **does cut the HL70 tail
~40% at ~24% return cost and actually RAISES Calmar (+1.68→+2.21)** — a usable risk option for live
capital preservation. BUT the honest R4 verdict is that **it does NOT beat a constant flat de-gross of
equal average exposure by more than a marginal amount, and it ranks only p33–p62 vs random de-grossing of
matched %-time** — i.e. the tail-cap is **essentially "run smaller when underwater," ~proportional, not a
skillful tail-selective stop.** It is robust across episodes (R5 PASS, episode-LOFO holds) and survives
nested-OOS on HL70 (R6 PASS: +21.8% ddRed at +2.6% cost) but FAILS on EXT (R6: +0% ddRed). **This is a
risk-preference trade the human can take — not free DD reduction.** Recommended config + trade-off below.

Script: `research/convexity_portable_2026-05-20/scripts/iter011_reactive_dd_stop.py`. Reuses the X123
held-book per-cycle panels verbatim (`pnl_base` = production held-book per-cycle return on constant gross
notional; ×1e4 = bps, additive equity). HL70 baseline reproduces exactly: Sharpe +1.93, maxDD −5,674,
Calmar +1.68, totPnL +10,472.

---

## STEP 2 — the reactive mechanisms (all PIT, mechanical, with a defined re-entry rule)

**(a) EQUITY-DRAWDOWN STOP** — track running cum-equity on realized `pnl_base`; the gross for cycle t is
decided from equity **through t−1** (running peak & DD lagged → R1 PIT clean). De-gross to `g_floor`
when DD-from-peak ≥ X bps. **Re-entry (R7):** book keeps participating at `g_floor`>0 so equity can heal;
re-enter (gross→1) when equity heals 50% of the DD back toward the peak (and is above the trough — never
buy back AT the trough) OR after 90 bars (~15d) as a fail-safe. **This re-entry rule is load-bearing:** an
initial g_floor=0 / re-enter-at-new-high design is a degenerate PERMANENT KILL (equity frozen at the floor
→ never makes a new high → 90–99% time stopped, 1 round-trip, ~90–100% cost). The floor + heal/timeout
fixes that pathology.

**(b) FAST-FLAG de-risk** — use iter-010's fast onset metrics (which flag ~21d earlier) to de-gross to 0
while the flag fires; re-enter when it normalizes.

**(c) COMBO** — fast-flag ARMS, equity-DD CONFIRMS (de-gross only when both fire), sane re-entry.

---

## STEP 2/3 — the DD-vs-COST trade-off curve (arm a, g_floor=0.4, the deliverable)

The interesting region is DEEP thresholds that fire only in catastrophic tails (shallow stops just run
smaller all the time — iter-004). DD reduction (ddRed) and return cost (totCost) vs threshold X:

**HL70 (production universe; baseline maxDD −5,674, Calmar +1.68):**
| X (bps off peak) | maxDD | ddRed% | totPnL | totCost% | Sharpe | Calmar | %time-stop | round-trips |
|---|---|---|---|---|---|---|---|---|
| 800  | −3,093 | 45.5 | 5,179 | 50.5 | 1.45 | 1.52 | 75 | 26 |
| 1200 | −3,158 | 44.3 | 7,541 | 28.0 | 1.77 | 2.17 | 59 | 17 |
| **1600** | **−3,281** | **42.2** | **7,956** | **24.0** | **1.77** | **2.21** | 53 | 15 |
| 2000 | −3,595 | 36.6 | 6,773 | 35.3 | 1.51 | 1.72 | 53 | 15 |
| 2500 | −3,788 | 33.2 | 8,651 | 17.4 | 1.86 | 2.08 | 51 | 14 |
| 3000 | −4,307 | 24.1 | 8,119 | 22.5 | 1.72 | 1.72 | 51 | 14 |
| 4000 | −4,676 | 17.6 | 9,497 |  9.3 | 1.89 | 1.85 | 41 | 11 |

**The HL70 sweet spot is X≈1,600 bps: cuts maxDD 42% (−5,674 → −3,281) at 24% totPnL cost, Sharpe
+1.93→+1.77, and Calmar RISES +1.68→+2.21** (DD cut outweighs the return give-up). R2 PASS (≥25–30% cut),
R3 cost is bounded and explicit (~24% of totPnL).

**EXT (2021-26 multi-episode; base maxDD −4,953):** X=800 ddRed 47% / cost 46%; X=1600 ddRed 36% / cost
39%; X=2000 ddRed 31% / cost 38%. **S44 (base maxDD −4,170):** X=800 ddRed 44% / cost 53%; X=1200 ddRed
41% / cost 38% (Calmar +2.10→+2.23); X=2000 ddRed 23% / cost 27%. The cut holds on every universe but the
cost on EXT/S44 is higher per unit of ddRed than on HL70 (HL70 is the favorable case because its tail is
one deep contiguous −57% episode the stop catches cleanly).

---

## STEP 3 — the decisive honest tests

### R4 (reframed G4) — STOP vs CONSTANT de-gross of EQUAL AVERAGE EXPOSURE — **marginal / mixed**
Does triggering ON the drawdown cut the LEFT TAIL better than just always running at the stop's average
gross? STOP maxDD minus CONST maxDD (positive = stop caps the tail better):

| X | HL70 (avgG) STOP−CONST maxDD | EXT STOP−CONST | S44 STOP−CONST |
|---|---|---|---|
| 800  | +25 (0.55) | −303 | +72 |
| 1200 | +516 | −527 | +403 |
| 1600 | +593 | −409 | −130 |
| 2000 | +291 | −525 | +359 |
| 2500 | +156 | −513 | +410 |

On **HL70 the stop modestly beats constant-de-gross** (by +25 to +593 bps of maxDD — i.e. triggering on
the DD is a little better than always running smaller, because it concentrates the de-gross in the deep
episode). On **EXT constant-de-gross MATCHES or BEATS the stop at every threshold** (STOP−CONST negative
throughout). On S44 it is mixed. **Read: the tail-selectivity edge over "just run smaller" is small and
not universe-robust.**

### R4-PLACEBO — STOP vs RANDOM de-gross of MATCHED %-time (200 seeds) — **FAIL (p33–p62)**
The sharper control: does triggering on the *realized* DD cap the maxDD better than de-grossing the same
%-of-time at *random* cycles? Real-stop maxDD rank within the random distribution:
| | X=1600 | X=2000 |
|---|---|---|
| HL70 | **p60** | **p33** |
| EXT  | **p62** | **p57** |

**Real ranks p33–p62 — well below p95.** A random de-gross of matched %-time caps the tail about as well
(often better). **This is the decisive honest finding: the equity-DD stop's tail-cap is ~proportional to
the exposure it removes, NOT a skillful tail-selective stop.** Same family as the iter-001/002 "run
smaller, not skill" result — which is *expected* on the reactive track (a stop reacts to a DD already
underway; it cannot select the tail forward). The value, if any, is that it removes exposure
*specifically while underwater* (a defined live-capital-preservation policy), not that it beats random.

### R5 (DECISIVE) — cross-episode tail-capping + episode-LOFO — **PASS**
One mechanical rule (X=1600, g_floor=0.4) applied globally on the EXT running equity, maxDD WITHIN each
episode window:
| episode | base maxDD | stop maxDD | ddRed% |
|---|---|---|---|
| 2022_luna | −765 | −306 | 60.0 |
| 2022_ftx | −2,474 | −1,323 | 46.5 |
| 2024_summer | −1,266 | −484 | 61.8 |
| 2025_q4 (EXT slice) | −900 | −900 | −0.0* |

3/4 episodes get ≥10% cut at X=1600 (the 2025_q4 EXT-slice DD is small/shallow *within* EXT's lower-vol
book — the real −57% lives on HL70, which the trade-off table above shows is cut 42%). **Episode-LOFO
(drop each episode, recompute stop on the remainder, X=2000): ddRed stays +22.8% / +31.3% / +30.2% /
+31.3% dropping luna/ftx/2024/q4 respectively — the tail-cap does NOT vanish dropping any one episode.**
R5 PASS — unlike the alpha-track gates (iter-003/006/007 collapsed under LOFO), the *mechanical* stop is
genuinely a general rule, not fit to one episode. (Makes sense: it reacts to whatever DD occurs.)

### R6 — nested-OOS of the threshold (pick X on past folds, apply forward) — **PASS on HL70, FAIL on EXT**
Pick X on past folds (maximize ddRed under a ≤25% cost budget), apply to the next fold:
- **HL70: forward ddRed +21.8% at +2.6% totPnL cost** (OOS maxDD −5,674 → −4,435, Sharpe +1.09→+1.19).
  PASS — the threshold generalizes forward on the production universe and is nearly cost-free OOS.
- **EXT: forward ddRed +0.0% at +23.9% cost** (chosen X drifts to the deepest 4000 in late folds → barely
  fires → no cut but still pays). FAIL on EXT.

### R1 PASS (PIT — equity/DD computed through t−1, lagged). R7 PASS (defined heal/timeout re-entry, never
buys back at the trough; the g_floor>0 design avoids the permanent-kill pathology).

---

## Arms (b) and (c) — fast-flag de-risk and combo — **WORSE than arm (a)**
**(b) Fast-flag de-risk (iter-010 metrics, EXT):** every flag either makes maxDD WORSE (alt_1d −79%,
alt_dd20 −48%, alt_accel −37%) or barely helps (alt_7d −4%), and **constant-de-gross matches/beats every
flag** — consistent with iter-010 (the flags fire on a coincident drawdown, the forward move bounces, so
de-grossing on them removes good cycles too). The reactive equity stop (arm a) is strictly better because
it keys off the strategy's OWN realized loss, not a price flag that bounces.
**(c) Combo (breadth-armed equity stop, EXT):** X=1200 ddRed +23.9% at +13.9% cost, Sharpe +0.97, but
constant-de-gross of equal exposure still beats it (const maxDD −2,858 vs combo −3,768). Adding the
fast-flag arm only DELAYS the stop (worse, since the flag bounces). No improvement over arm (a).

---

## STEP 4 — recommended reactive config + STATED trade-off (the deliverable)

**RECOMMENDED (if the human wants a live capital-preservation stop):**
> **Equity-drawdown stop, X = 1,600 bps off running peak, g_floor = 0.40, re-enter on 50%-heal OR
> 90-bar (~15d) timeout.** PIT, mechanical, parameter-light.

**Stated trade-off (HL70, production):** maxDD −5,674 → **−3,281 (−42%)**, totPnL −24%, Sharpe
+1.93→+1.77, **Calmar +1.68→+2.21 (improves)**, ~53% of time at reduced gross, ~15 round-trips over the
402-day OOS. Nested-OOS forward: +21.8% ddRed at only +2.6% cost. Robust across episodes (R5/LOFO PASS).

**THE HONEST CAVEAT (must be stated to the human):** this is **NOT a skillful tail-selective stop** — it
ranks p33–p62 vs a random de-gross of matched %-time (R4-PLACEBO FAIL) and only marginally beats / ties a
constant flat de-gross of equal average exposure (R4 mixed, EXT favors constant). **The honest equivalent
is "run at lower gross specifically while the book is underwater."** It reduces DD at ~proportional return
cost (no favorable asymmetry beyond the modest HL70 edge). That is *expected and acceptable* on the
reactive track: a stop reacts to a DD already happening — it cannot forecast — but it does mechanically
cap the live downside at a bounded, explicit, robust cost. **The human picks the risk point on the curve
above.** If the goal is purely "never let live capital draw down past ~33%," X≈1,600 g_floor=0.4 delivers
that at a known ~24% long-run return give-up, and Calmar improves — a reasonable capital-preservation
trade. If the goal is "free DD reduction with no return cost," there is none — the cost is real and
~proportional.

**Alternative honest framing for the human:** since the stop barely beats constant-de-gross, an even
simpler policy — **just run the whole book at ~0.65–0.70 constant gross** — gives a similar DD/return
profile without any state machine or whipsaw round-trips. The reactive stop is preferable only if the
human specifically wants full exposure in calm periods and automatic de-grossing once a deep loss is
underway (the "let it run, then protect" preference), accepting the whipsaw.

---

## How this fits the prior ledger
iters 5–10 closed the **prediction** axis (nothing free leads the alt-bear). iter-011 closes the
**reaction** axis honestly: a mechanical equity stop DOES cap the tail at bounded cost and IS
cross-episode robust (the first DD mechanism to survive episode-LOFO — because it reacts rather than
predicts), but it is **not a skillful tail-selector** (p33–p62 vs random, ~ties constant-de-gross). The
DD is reducible *mechanically* at ~proportional cost; it is not reducible *for free* on free data. The
deliverable is the trade-off curve + the X≈1,600/g_floor=0.4 recommendation with its stated ~42%-cut /
~24%-cost / Calmar-up trade — a characterized risk option for the human, exactly as the reactive track
specifies.

## Artifacts
- script: `research/convexity_portable_2026-05-20/scripts/iter011_reactive_dd_stop.py` (reproduces every
  table: full DD-vs-cost trade-off on HL70/EXT/S44, R4 vs constant-de-gross, R5 per-episode + LOFO, R6
  nested-OOS, arms b/c, R4-placebo).
- reuses: `results/X123_altbear_short_{HL70,EXT,S44}.parquet`, `results/iter010_fast_metrics_EXT.parquet`.
