# Research Insights — iter-012 (PORTABLE reactive equity-DD stop: self-normalizing trigger)

**REACTIVE risk-control track (NOT alpha).** iter-011 built a working equity-DD stop but its trigger
was an **ABSOLUTE bps threshold** (X=1600 bps off peak). Different universes have different natural
equity scales / volatility, so one absolute X cannot be right everywhere → iter-011 PASSED nested-OOS
on HL70 but **FAILED on EXT** (R6: forward ddRed −7.3% at +44.8% cost — X drifts deep in late folds and
barely fires). This iteration fixes that one weakness by making the trigger **self-normalizing**, so a
single UNITLESS parameter generalizes across HL70 + EXT + S44 under nested-OOS WITHOUT per-universe
tuning.

**One-line verdict:** The **vol-normalized DD trigger** — fire when the drawdown-from-peak exceeds
`k × σ_equity × √win`, where `k` is a UNITLESS "sigmas of equity" multiple — is the **PORTABLE** form.
It is the **only one of four trigger forms that PASSES R6 nested-OOS on all three universes (3/3)**, and
the nested-OOS selector lands on essentially the **same unitless k≈2.0 on every universe and every fold**
— that stable cross-universe choice is the portability signature. At the recommended **k=2.0** it cuts
maxDD **HL70 +33% / EXT +39% / S44 +21%** at bounded cost (11–32%) and **Calmar IMPROVES on all three**
(HL70 1.68→2.01, EXT 0.66→0.74, S44 2.10→2.36). Cross-episode R5 + LOFO PASS (4/4 EXT episodes capped).
The honest R4 caveat from iter-011 still holds (it ranks p55–p70 vs random matched-%-time de-gross →
~proportional, not a skillful tail-selector) — but that is *expected and acceptable* on the reactive
track; the win condition for THIS iteration was **portability**, which is met.

Script: `research/convexity_portable_2026-05-20/scripts/iter012_portable_dd_stop.py`. Reuses X123
`build_universe` + X124 held-book mechanics verbatim (gross applied to positions BEFORE turnover/cost).
Base reproduces exactly: HL70 Sharpe +1.93 / maxDD −5,674 / Calmar +1.68 / tot +10,472.

---

## STEP 2 — the portable trigger FORMS (each one UNITLESS knob; same fixed re-entry policy as iter-011)

All triggers PIT (R1): equity / peak / DD / vol computed through t−1, expanding or trailing only, no
future. Same fixed re-entry policy carried from iter-011 (g_floor=0.40, heal 50%, timeout 90 bars →
R7 carries over). Fixed trailing windows are policy, not tuned: vol_win=180 bars (~30d), warmup=60.

- **(a) VOLNORM** — fire when `(peak − eq) ≥ k · σ(equity increments, trailing 180) · √win`. `k` is a
  UNITLESS "sigmas of equity" multiple. Self-scales to each universe's own equity volatility.
- **(b) PCTILE** — fire when current DD is in the worst `q`-quantile of the strategy's OWN **expanding**
  DD distribution (PIT). `q` unitless.
- **(c) MAXDDFRAC** — fire when current DD > `f` × the strategy's trailing-360-bar max DD. `f` unitless.
- **(absolute)** — iter-011's `X` bps reference (the thing we are trying to beat on EXT/S44).

---

## STEP 3 — the decisive robustness tests

### R6 — NESTED-OOS of the unitless param (THE TARGET): pick param on past folds, apply forward
Choose the knob on past walk-forward folds (max ddRed under ≤25% cost budget), apply to the next fold;
measure realized FORWARD DD-capping. PASS = forward ddRed > +5% AND cost < 40% on **every** universe.

| form | HL70 fwd ddRed / cost | EXT fwd ddRed / cost | S44 fwd ddRed / cost | universes PASS |
|---|---|---|---|---|
| absolute (iter-011) | **+21.8% / +2.6%** PASS | **−7.3% / +44.8%** FAIL | −16.1% / +8.8% FAIL | **1/3** |
| **volnorm** | **+31.9% / −34.3%** PASS | **+29.1% / +31.4%** PASS | **+5.3% / +3.1%** PASS | **3/3 ← PORTABLE** |
| pctile | +44.0% / +13.3% PASS | +34.1% / +28.6% PASS | +10.8% / +46.2% FAIL | 2/3 |
| maxddfrac | +5.2% / +21.9% PASS | +0.0% / +0.0% FAIL | +12.9% / +16.0% PASS | 2/3 |

**volnorm is the only PORTABLE form.** Why it generalizes where absolute does not: the nested-OOS
selector for `absolute` drifts to deep X (2500–3000) on EXT/S44 because the *base* equity scale on the
longer panels makes the same bps threshold rarely fire — so it pays cost but caps nothing. The nested-OOS
selector for `volnorm` lands on essentially the **same k≈2.0 on every universe and every fold** (HL70: all
k=2.0; EXT: 2.0 except one fold 2.5; S44: 2.0 with two folds 2.5/4.0) — a self-normalizing trigger has the
same "meaning" in every universe, so the chosen knob transports. That stable choice IS the portability.

### R5 (DECISIVE) — cross-episode tail-capping (EXT) + episode-LOFO
volnorm at k=3.0 on the running EXT equity caps **4/4 episodes ≥10%** (vs absolute/pctile 3/4 — they miss
the shallow 2025_q4 EXT-slice; volnorm catches it because its trigger scales DOWN with the lower-vol book):
| episode | base maxDD | volnorm maxDD | ddRed% |
|---|---|---|---|
| 2022_luna | −765 | −306 | 60.0 |
| 2022_ftx | −2,474 | −935 | 62.2 |
| 2024_summer | −1,266 | −499 | 60.6 |
| 2025_q4 (EXT slice) | −900 | −803 | 10.8 |
**Episode-LOFO (drop each, recompute on remainder):** ddRed stays +27/+27/+28/+27% dropping
luna/ftx/2024/q4 — the cap does NOT vanish dropping any one episode. **R5 PASS.**

### R2/R3 — maxDD reduction + bounded cost at the RECOMMENDED k=2.0 (canonical held-book, @4.5bps)
| universe | base maxDD | stop maxDD | ddRed% | totCost% | Sharpe | Calmar (base→stop) |
|---|---|---|---|---|---|---|
| **HL70** (prod) | −5,674 | −3,794 | **+33.1** | +19.9 | 1.93→1.80 | **1.68→2.01** |
| **EXT** | −4,953 | −3,000 | **+39.4** | +32.4 | 0.87→0.86 | **0.66→0.74** |
| **S44** | −4,170 | −3,307 | **+20.7** | +11.1 | 1.84→1.89 | **2.10→2.36** |
**R2 PASS** (≥20% cut on all three). **R3 PASS** — cost bounded (11–32% of totPnL) and explicitly stated;
**Calmar IMPROVES on every universe.** Cost-robust across {1,3,4.5}bps (the firing logic is on equity, not
cost). Note: at k=2.0 the volnorm trade-off knee on HL70 (ddRed 33% / cost 20%) is close to absolute
X=1600 (42% / 24%) — slightly less DD cut on HL70 but it BUYS the EXT/S44 portability the absolute form
lacks. Deeper k (3.0–4.0) reduces firing/cost but cuts less DD — a clean risk dial.

### R4 — STOP vs CONSTANT de-gross of equal average exposure + R4-placebo (the honesty gate)
At k=2.0, R4-placebo (200 seeds, matched %-time random de-gross): real volnorm ranks **HL70 p70 / EXT p55 /
S44 p70** — below p95. STOP−CONST maxDD at k=3.0: HL70 −776, EXT −554, S44 +77 (mixed/slightly negative on
HL70/EXT). **Same conclusion as iter-011: the tail-cap is ~proportional to exposure removed, NOT a skillful
tail-selector.** This is EXPECTED on the reactive track (a stop reacts to a DD already underway; it cannot
forecast). It does NOT disqualify the deliverable — the win condition this iteration was PORTABILITY, and
volnorm is portable where absolute X was not. The honest equivalent is still "run smaller while underwater."

### R1 PASS (PIT — σ_equity, peak, DD all trailing/expanding through t−1). R7 PASS (re-entry policy
unchanged from iter-011: g_floor=0.40>0 so equity heals while stopped, heal-50%-or-90-bar timeout with
`eq>trough` guard — no frozen-equity permanent kill, no buy-back-at-trough).

---

## STEP 4 — recommended PORTABLE config (the deliverable)

> **Vol-normalized equity-DD stop.** De-gross the whole held book to `g_floor=0.40` when the strategy's
> own drawdown-from-peak `(peak − eq)` ≥ **k=2.0** × σ(trailing-180-bar equity increments) × √180. Equity /
> peak / σ computed through t−1 (PIT). Re-enter (gross→1) when equity heals 50% of the DD back toward the
> peak (and `eq > trough`) OR after 90 bars (~15d). Warmup 60 bars before the trigger can fire.

**`k` is the only knob and it is UNITLESS** — it means the same thing on every universe (sigmas of the
strategy's own equity), so it generalizes. The window (180), g_floor, heal, timeout are fixed policy.

**Cross-universe trade-off at k=2.0 (canonical, @4.5bps):**
| universe | maxDD cut | cost | Calmar |
|---|---|---|---|
| HL70 (prod) | −5,674 → −3,794 (**−33%**) | −20% totPnL | 1.68 → **2.01** |
| EXT | −4,953 → −3,000 (**−39%**) | −32% totPnL | 0.66 → **0.74** |
| S44 | −4,170 → −3,307 (**−21%**) | −11% totPnL | 2.10 → **2.36** |

**THE HONEST CAVEAT (must be stated):** like iter-011, this is **NOT free DD reduction and NOT skill** —
it ranks p55–p70 vs random matched-%-time de-gross (R4-placebo) and ~ties / slightly trails a constant
flat de-gross of equal average exposure (R4). The cost is real and ~proportional. **What iter-012 BUYS
over iter-011 is PORTABILITY**: the unitless k=2.0 caps DD forward on HL70, EXT, AND S44 under nested-OOS
(3/3), whereas iter-011's absolute X=1600 worked only on HL70 (1/3) and had to be re-tuned per universe.
For live deployment across a drifting/expanding universe, the portable form is the safer rule — it
re-calibrates itself to each universe's equity scale instead of requiring a hand-set bps threshold.

---

## How this fits the prior ledger
iters 5–10 closed the **prediction** axis (nothing free leads the alt-bear). iter-011 characterized the
**reaction** axis (a mechanical equity stop caps the tail at ~proportional cost, cross-episode robust) but
its absolute trigger was HL70-specific (failed EXT/S44 nested-OOS). iter-012 closes the **portability**
gap: the vol-normalized (self-normalizing) trigger is the form whose single unitless parameter generalizes
across all three universes under honest nested-OOS. The DD is mechanically reducible at ~proportional cost
with a **portable, parameter-light, PIT** rule — that is the deployable reactive risk overlay. There is
still no skillful tail-selector on free data (R4 ~proportional everywhere) — that conclusion is unchanged.

## Artifacts
- script: `research/convexity_portable_2026-05-20/scripts/iter012_portable_dd_stop.py`
- results: `iter012_tradeoff.parquet` (DD-vs-cost per form/universe), `iter012_nested_oos.parquet`
  (R6 per form/universe), `iter012_r5_episodes.parquet` (cross-episode + LOFO).
- reuses: X123 `build_universe`, X124 held-book engine, `results/X123_altbear_short_{HL70,EXT,S44}.parquet`.
