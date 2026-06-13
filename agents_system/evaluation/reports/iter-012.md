# Evaluation — iter-012 (REACTIVE risk-control track)

**Verdict: ACCEPTABLE FOR DEPLOYMENT as a risk option — PORTABLE form. SUPERSEDES the iter-011
absolute-X overlay.** NOT an alpha ADOPT; the alpha champion stays = baseline (Calmar +1.68).

Graded on the reactive-track gates R1–R7 (TWO-TRACKS clause of `evaluation_contract.md`), NOT the
alpha G4≥p95 disqualifier. Reactive verdict = "ACCEPTABLE FOR DEPLOYMENT" iff R1/R2/R5/R6 hold and R3
cost is acceptable to the human.

## The change
**Vol-normalized (portable) reactive equity-DD stop.** De-gross the whole held book to `g_floor=0.40`
when the strategy's own drawdown-from-peak `(peak − eq)` ≥ **k · σ(trailing-180-bar equity increments)
· √180**, where **k is UNITLESS** ("sigmas of equity"), recommended **k=2.0**. Equity / peak / σ
through t−1 (PIT). Re-enter (gross→1) on 50%-heal OR 90-bar timeout, with `eq>trough` guard. Warmup 60.

The ONLY change vs the iter-011 absolute-X stop is the **trigger FORM**: absolute bps (`X=1600` off
peak) → self-normalizing `k·σ·√win`. Same held-book engine (X124 verbatim), same re-entry policy.
**Headline = R6: passes nested-OOS on ALL THREE universes (3/3) vs iter-011's 1/3 (HL70-only).**

Script: `research/convexity_portable_2026-05-20/scripts/X125_volnorm_stop.py` (re-run, 220s,
deterministic seed 12345). Base reproduces X117 EXACT: HL70 @4.5bps Sharpe +1.93 / maxDD −5674 /
Calmar +1.68 / totPnL +10472. All numbers below independently re-run from the script + parquets.

## R-gate results (reactive track)

| # | Gate | Result | Numbers (verified) |
|---|---|---|---|
| **R1** | Look-ahead | **PASS** | Review PASS. PIT vol trigger: dd, σ, peak all through t−1; `incr[t]` written AFTER gross fixed + pnl realized; HOLD-lagged sleeve book; warmup-60 firing guard; k chosen only in R6 from past folds. No forward peek. |
| **R2** | Tail reduction | **PASS** | maxDD ddRed @4.5bps k=2.0: **HL70 +33.1%** (−5674→−3794), **EXT +39.4%** (−4953→−3000), **S44 +20.7%** (−4170→−3307). ≥20% cut on all three. |
| **R3** | Bounded cost | **PASS (stated)** | totPnL give-up: HL70 −19.9%, EXT −32.4%, S44 −11.1%. **Calmar IMPROVES on all three:** HL70 1.68→2.01, EXT 0.66→0.74, S44 2.10→2.36. Sharpe: 1.93→1.80 / 0.87→0.86 / 1.84→1.89. Cost-robust across {1,3,4.5}bps (firing keys off equity, not cost). |
| **R4** | vs constant-de-gross | **~PROPORTIONAL (not skill — expected)** | R4-placebo (200 matched-%-time seeds): real ranks **HL70 p70 / EXT p55 / S44 p70** — all < p95. STOP−CONST maxDD mixed (HL70 +131 / EXT −393 / S44 +100 at k=2.0). Tail-cap is ~proportional to exposure removed, NOT a skillful tail-selector. State plainly: this is the EXPECTED reactive-track outcome, NOT a skill claim. |
| **R5** | Cross-episode + LOFO | **PASS (decisive)** | Caps **4/4** EXT episodes ≥10% @k=2.0: luna 60.0%, ftx 56.5%, 2024_summer 55.9%, 2025_q4 10.8% (beats absolute/pctile, which miss the shallow q4). Episode-LOFO @k=2.0: ddRed stays **+37.5 / +39.4 / +37.7 / +39.4%** dropping luna/ftx/2024/q4 — does NOT vanish dropping any one. |
| **R6** | **Cross-universe nested-OOS (HEADLINE)** | **PASS 3/3 — PORTABLE** | k chosen on PAST folds (max ddRed under ≤25% cost budget), applied FORWARD, PER UNIVERSE. Forward ddRed / cost: **HL70 +33.4% / −36.2%**, **EXT +29.1% / +27.7%**, **S44 +9.0% / +6.2%** — all PASS (ddRed>+5% AND cost<40%). Selector lands on the low-k family (mostly 1.5–2.0) on every universe/fold. **vs iter-011 absolute-X = 1/3** (HL70 only; EXT −7.3%/+44.8% FAIL). |
| **R7** | Re-entry sanity | **PASS** | HL70 @k=2.0: 15 round-trips, 14 re-entries, 51.4% time stopped. g_floor=0.40>0 (equity heals while stopped, no frozen-equity kill); `eq>trough` guard (never buys at trough); heal-50%-or-90-bar timeout. No buy-back-at-top pathology. |

**Decision logic (TWO-TRACKS):** R1/R2/R5/R6 all hold on ALL THREE universes; R3 cost bounded and
explicit (Calmar up everywhere). → **ACCEPTABLE FOR DEPLOYMENT as a portable risk option.**

## R3 k-sweep trade-off dial (HL70 @4.5bps, g_floor=0.40) — the human picks a risk point
| k | maxDD | ddRed% | totPnL cost% | Sharpe | Calmar | %time stop | round-trips |
|---|---|---|---|---|---|---|---|
| 1.5 | −3606 | **+36.5** | +16.0 | 1.90 | 2.22 | 52.3 | 15 |
| **2.0 (rec)** | −3794 | **+33.1** | +19.9 | 1.80 | **2.01** | 51.4 | 15 |
| 2.5 | −4728 | +16.7 | +13.6 | 1.82 | 1.74 | 44.9 | 12 |
| 3.0 | −4947 | +12.8 | +26.5 | 1.58 | 1.42 | 44.2 | 12 |

Lower k = more DD cut, more firing/cost; deeper k = shallower dial. k=2.0 is the recommended knee
(33% DD cut, 20% cost, Calmar 1.68→2.01). Full curve (3 universes × {1,3,4.5}bps × k) in
`results/X125_tradeoff_curve.parquet`.

## Cross-universe summary @4.5bps, k=2.0 (canonical held book)
| universe | base maxDD | stop maxDD | ddRed% | cost% (totPnL) | Sharpe | Calmar (base→stop) | R6 nested-OOS fwd ddRed/cost |
|---|---|---|---|---|---|---|---|
| **HL70** (prod) | −5,674 | −3,794 | **+33.1** | −19.9 | 1.93→1.80 | **1.68→2.01** | **+33.4% / −36.2% PASS** |
| **EXT** (2021–26) | −4,953 | −3,000 | **+39.4** | −32.4 | 0.87→0.86 | **0.66→0.74** | **+29.1% / +27.7% PASS** |
| **S44** | −4,170 | −3,307 | **+20.7** | −11.1 | 1.84→1.89 | **2.10→2.36** | **+9.0% / +6.2% PASS** |

DD cut AND Calmar improves on ALL THREE; cost bounded 11–32%; nested-OOS passes 3/3.

## Why this SUPERSEDES the iter-011 absolute-X overlay
iter-011 (absolute X=1600) was characterized as deployable but **HL70-tuned**: R6 passed on HL70
(+21.8% / +2.6%) but FAILED on EXT (−7.3% / +44.8%) — one absolute bps threshold cannot be right
across universes with different equity scales (X drifts deep on the longer EXT/S44 panels, pays cost,
caps nothing). iter-012 fixes exactly that one weakness:

- **Same family of behavior** (~proportional tail-cap, Calmar improvement, R4 p55–p70 < p95 — NOT a
  new skill claim; honest equivalent is still a constant ~0.67 gross book).
- **Strictly more robust:** the unitless k=2.0 means "the same number of sigmas of equity" on every
  universe → it transports under nested-OOS (3/3 vs 1/3) WITHOUT per-universe re-tuning.
- **Self-recalibrating to each universe's equity scale** — important for a drifting / expanding live
  universe where a hand-set bps threshold would go stale. The new form removes the iter-011 R6 EXT
  caveat entirely.

The trade is the recommended-k DD cut on HL70 is slightly shallower than iter-011's absolute X=1600
(33% vs 42% / cost 20% vs 24%) — but it BUYS the EXT/S44 portability the absolute form lacked. For
live deployment across a drifting universe, the portable form is the safer rule.

## Honest framing (unchanged from iter-011, must be stated)
- This is **NOT free DD reduction and NOT skill.** R4-placebo ranks p55–p70 < p95 on all three →
  the cut is ~proportional to exposure removed. The honest equivalent is running the whole book at a
  constant ~0.67 gross (no state machine, no whipsaw).
- The equity-stop's value over constant-de-gross is the **asymmetry of WHEN it de-risks** (full
  exposure in calm, automatic protection once a deep loss is underway, ~15 whipsaw RT/402d) and now
  the **self-normalizing portability** across universes.
- There is still **no skillful tail-selector on free data** — that conclusion is unchanged. iter-012
  closes the *portability* gap of the reactive overlay, not the skill gap.

## Insight for next cycle
The reaction axis is now fully characterized: a PIT, parameter-light (one unitless knob), portable
equity stop caps the −33 to −39% tail at ~proportional cost, robust cross-episode (LOFO) and
cross-universe (nested-OOS 3/3). The prediction axis (iters 5–10) and the reaction axis (iters 11–12)
are both closed on free data: the DD is mechanically reducible at ~proportional cost, NOT for free.
Remaining DD-reduction classes are out-of-scope for free-data construction (bull-only beta, a
different alpha, paid leading data) or "accept the DD + deploy this portable stop." This is the
recommended deployable reactive overlay for the live, drifting-universe book.
