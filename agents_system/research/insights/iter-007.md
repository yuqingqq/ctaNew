# Research Insights — iter-007 (FINALIZE the alt-bear gate: parameter-free form + multi-episode design)

**Mandate:** take the iter-006 proposed 2-axis alt-bear side gate, (1) make it **parameter-free** so
G3 is genuinely *waivable* not tuned, (2) design the **multi-episode** validation on the 23-sym
2021–26 ext panel (the thing HL70 lacks), (3) note look-ahead traps. Then pre-register and hand off.
ONE change only: the 2-axis alt-bear gate.

Data-check scripts: `research/convexity_portable_2026-05-20/scripts/iter007_structform_check.py`
(+ follow-up full-book per-episode + HL70 LOFO, logged below). Reuses the X117/X121 held-book engine
verbatim (K=5, HOLD=6, mom-bull / mean-rev-side-BN / flat-bear, trailing-180-bar `.shift(1)` betas).

---

## TL;DR — the chosen form, and a decisive honest caveat I am putting up front

**Chosen parameter-free form: `F1 = (alt_index_30d < btc_30d)`** — in the SIDEWAYS regime, FLAT the book
when the equal-weight alt complex is *underperforming BTC over the trailing 30d* (relative alt-bear).
It is fully parameter-free: the comparison boundary is a structural ±0 relative threshold, exactly like
the existing ±10% BTC regime rule and the bear→FLAT rule — **no swept scalar**. So **G3 is waived by
construction.**

**BUT** — and this is the load-bearing finding of the iteration — the multi-episode data check I was
asked to design and run **already refutes the gate honestly**: on the multi-episode ext panel the
alt-bear axis **net-HURTS** (Calmar +0.66 → +0.27, maxDD WORSE −4,953 → −6,691), and on HL70 the
spectacular win (+1.68 → +4.89 Calmar) **collapses to −0.81 lift the moment you drop the single 2025-Q4
episode (fold 5)**. The mechanism does *not* generalize across episodes. I am handing this off as
**READY-to-confirm**, with a pre-registration whose honest expectation is **REJECT on G5/G7**, because
the multi-episode test — the one prior iterations never had the data for — is exactly what should now
adjudicate it, and it is cheap to run cleanly in Evaluation. If anyone reads this as a "win," they are
reading it wrong; it is the *decisive disposal* of the alt-bear regime axis.

---

## STEP 1 — making it parameter-free: the candidate forms and the data check

iter-006's gate was `regime=side AND alt_index_30d < −0.10 → FLAT`. The `−0.10` is a swept scalar (Calmar
spiked *only* at −0.10 → G3 fail). I evaluated four **structural, no-free-threshold** forms of "the alt
complex is in a bear," all using the same PIT alt-index (eq-weight trailing-30d cum log-ret of the traded
universe ex-BTC/ETH, `.shift(1)` lagged):

| form | rule | parameter-free? | rationale |
|---|---|---|---|
| F0 | `alt30 < 0` | yes (0 boundary) | absolute 30d alt drawdown |
| **F1** | **`alt30 < btc30`** | **yes (relative 0 boundary)** | **alts underperform BTC — the cleanest "BTC-invisible alt bear" the H4 mechanism describes** |
| F2 | `alt30 < btc30 AND alt30 < 0` | yes (two 0 boundaries) | relative *and* absolute |
| F3 | `breadth < 0.5` | yes (majority boundary) | <50% of alts up over 30d |
| T | `alt30 < −0.10` | **NO (swept)** | the iter-006 tuned reference |

**Separation data check (side-regime cycles, mean held-book PnL on FLAGGED vs UNFLAGGED, bps/cycle):**

| form | HL70 sep | HL70 DD-removed | EXT sep | EXT DD-removed |
|---|---|---|---|---|
| F0 | +0.31 | +46% | **−1.79** | −34% |
| **F1** | **−2.24** | +55% | **−2.45** | −8% |
| F2 | −0.51 | +44% | −2.06 | −44% |
| F3 | +6.11 | +69% | **−4.03** | −41% |
| T (tuned) | +6.18 | +37% | −3.24 | −50% |

`sep = mean(unflagged) − mean(flagged)`; **positive sep = the form correctly flags the LOSING side
cycles.** The sign flips between universes: on HL70 the forms (F0/F3/T) flag the losers (positive sep),
but **on the multi-episode ext panel EVERY form has negative sep** — i.e. the alt-bear condition flags
the side cycles that were *good*, not the bad ones, and FLATting them makes the ext maxDD *worse* (negative
DD-removed for F1/F2/F3/T). This is the first, cheapest signal that the HL70 result is universe/episode
specific.

**Why F1 is nonetheless the form to commit to** (if the axis is pursued at all): it is the *most
defensible parameter-free encoding of the iter-006 H4 mechanism* ("BTC says sideways while alts bleed" =
alts underperform BTC = `alt30 < btc30`), it has no swept scalar (G3 waived), and on HL70 it is the
strongest full-book performer of the structural forms. F3 (breadth) had a higher HL70 *separation* but
F3 and F0 also flip sign on EXT and are no more defensible; F1 is the cleanest mechanism match. **One form,
committed: F1.**

## STEP 1b — full-book confirmation (the headline numbers)

Applying F1/F2 as a true regime branch (`side AND alt30<btc30 → emit {}`, decaying sleeves like bear):

| universe | arm | Sharpe | maxDD | Calmar | totPnL |
|---|---|---|---|---|---|
| **HL70** (1 episode) | base | +1.93 | −5,674 | +1.68 | +10,472 |
| HL70 | **F1** | **+2.71** | **−2,239** | **+4.89** | +12,026 |
| HL70 | F2 | +2.49 | −2,239 | +4.64 | +11,400 |
| **EXT** (multi-episode) | base | +0.87 | −4,953 | **+0.66** | +15,448 |
| EXT | **F1** | +0.55 | **−6,691** | **+0.27** | +8,570 |
| EXT | F2 | +0.56 | −7,334 | +0.26 | +9,071 |

On HL70 F1 looks like a clean win (Calmar nearly 3×). On the **multi-episode panel it is a clear loss**
(Calmar more than halved, maxDD ~35% worse, totPnL nearly halved). That divergence is the whole story.

## STEP 2 evidence — the multi-episode / multi-fold disposal (the key upgrade vs iter-003)

**EXT per-episode maxDD (full book):**

| episode | base | F1 | F2 | F1 verdict |
|---|---|---|---|---|
| 2022_luna (May–Jul'22) | −765 | −765 | −765 | no side cycles flagged (BTC also bear → already FLAT) |
| 2022_ftx (Nov'22–Jan'23) | −2,474 | **−969** | −1,695 | helps |
| 2024_summer (Jun–Sep'24) | −1,266 | **−1,798** | −1,899 | **HURTS** (flats good cycles) |
| 2025_q4 (Sep–Dec'25) | −900 | **−616** | −616 | helps |

The gate helps in 2 episodes, hurts in 1, is inert in 1 — and **net degrades the panel**. It is not a
consistent DD reducer; it is a coin-flip per episode whose HL70 success was the lucky draw.

**HL70 LOFO across folds/episodes (drop one fold, recompute F1-vs-base Calmar lift):**

| dropped fold | lift | Δ vs full |
|---|---|---|
| full series | **+3.21** | — |
| −f2 | +3.64 | +0.43 |
| −f3 | +3.28 | +0.07 |
| −f4 | +3.23 | +0.02 |
| **−f5 (the deep-DD episode)** | **−0.81** | **−4.03** |
| −f6 | +3.69 | +0.48 |
| −f7 | +3.68 | +0.47 |
| −f8 | +3.42 | +0.21 |

Identical pathology to iter-003's `side→FLAT` (LOFO −0.86) and iter-002's fold-6 dependence: **100% of the
HL70 lift is the single 2025-Q4 episode** (fold 5). Per-fold F1 ≥ base in only **4/7 folds** (fails the 6/9
spirit of G5). Drop the one episode and the gate *loses* to baseline.

## STEP 3 — look-ahead traps (how the alt-index must be computed)

1. **alt-index = trailing PIT cum-return of the equal-weight TRADED universe, `.shift(1)` lagged.** It is
   built from the *same symbols being traded* (ex-BTC, ex-ETH), trailing 180×5m bars (~30d on the 4h grid),
   shifted one bar so the decision at cycle *t* uses only data through *t−1*. No forward window, no
   full-sample normalization.
2. **Relative form needs both legs PIT:** `btc_30d` is the existing PIT regime input (already `.shift`-safe
   via the same trailing window); `alt30` uses the same window so the comparison is apples-to-apples at the
   same lag. Do not mix a lagged alt30 with a contemporaneous btc30.
3. **Universe-consistency trap:** on each panel the alt-index must be the eq-weight of *that panel's* traded
   alts (HL70 → HL70 alts; EXT → the 23 alts; S44 → the 44 alts). Do **not** import an HL70 alt-index onto
   the ext panel — that would be a look-ahead/cross-universe leak.
4. **Breadth form (if ever revisited)** has the same rules: per-symbol trailing-30d cum log-ret, `.shift(1)`,
   fraction>0, decided at *t* from *t−1* data.

---

## STEP 2 (contract) — PRE-REGISTRATION against the gates

**Objective:** raise Calmar on HL70 *without* it being a single-episode artifact (the explicit upgrade:
G5 now means *survive LOFO across MULTIPLE episodes*, tested on the ext panel).

**The ONE change:** regime map gains a 2-axis branch — `regime==side AND alt_index_30d < btc_30d → FLAT`
(emit `{}`, decay sleeves like bear). bull→mom30 and bear→FLAT unchanged. No new feature input to the
model, no retrain, no threshold. (The alt-index is a PIT regime *descriptor*, computed in the construction
layer like mom30/beta.)

| gate | requirement (pre-registered) | honest expectation from the data check |
|---|---|---|
| G1 look-ahead | alt-index trailing eq-wt cum log-ret of the **panel's own** traded alts, `.shift(1)`; relative `alt30<btc30` at matched lag. | PASS (structural, PIT). |
| G2 in-sample | HL70 Calmar > +1.68. | **PASS** (F1 +4.89) — necessary, not sufficient. |
| **G3 nested-OOS** | **WAIVED** iff form is parameter-free. **F1 = `alt30 < btc30` has NO swept scalar** (relative 0-boundary, structural like ±10% BTC rule). State this; G3 waived. | Waived (legitimately). |
| **G4 matched placebo** | **≥ p95 MANDATORY**, re-derived: 100+ seeds of FLAT-ting the *same number* of random side cycles per universe; report percentile on HL70 **and EXT**. | HL70 likely ~p95 (iter-006 saw p95/p96); **EXT expected < p50** (the form flags good cycles there → random does as well or better). |
| **G5 LOFO across MULTIPLE episodes** | **MUST survive dropping ANY single episode/fold.** On HL70: per-fold F1≥base in ≥6/9 AND LOFO lift stays >0 dropping each fold. On EXT: per-episode maxDD improvement in **≥ 3/4** alt-bear episodes AND LOFO across the 4 calendar episodes stays >0. | **FAIL** — HL70 LOFO −f5 = **−0.81**; HL70 per-fold 4/7; EXT helps only 2/4 episodes (hurts 2024_summer). |
| G6 paired CI | Block-bootstrap paired per-cycle PnL diff (F1−base) by fold, on HL70 **and EXT**; CI must not cross 0. | HL70 may clear; **EXT expected to cross/sit negative.** |
| G7 universe | Must hold on **HL70 (production)** AND **EXT 23-sym** AND **S44**. The improvement cannot be a single-universe artifact. | **FAIL** — EXT Calmar +0.66→+0.27 (net hurts). |
| G8 cost | Report @1 / 3 / 4.5 bps; must not depend on unrealistic cost. | The gate FLATs cycles (less cost) so it benefits at high cost — but that is the "run-smaller" confound, hence G4/G7 are the binding tests. |

**Decision rule (pre-committed):** ADOPT only if G4≥p95 **on both HL70 and EXT**, G5 survives episode-LOFO
on both, G6 CI clears 0 on both, G7 holds on HL70+EXT+S44. **The data check predicts REJECT on G5 + G7**
(single-episode artifact; net-hurts multi-episode). This pre-registration is written so Evaluation
produces the clean, contract-compliant numbers that *retire* the alt-bear axis rather than leaving it as
an open "looked promising on HL70" thread.

---

## What this iteration establishes

- The iter-006 alt-bear gate, recast in its **most defensible parameter-free form (F1, `alt30<btc30`)**,
  is **not a real DD reducer**: it is a single-episode (2025-Q4 / fold-5) fit on HL70 that **net-degrades
  the multi-episode ext panel** and helps only 2 of 4 alt-bear episodes. The G4 p95 it cleared on HL70 was
  the n=1 episode masquerading as conditional skill.
- This converges with iter-002/003/004/005/006: the **single −57% episode is the binding constraint**, and
  the alt-complex-bear regime axis — the last specific feature hypothesis the diagnostic floated — **does
  not survive the multi-episode test** that prior iterations lacked the data to run. With this, the
  *regime-axis / composition* family of DD fixes is exhausted on free data.
- **Recommended orchestrator read:** authorize Evaluation to run the pre-registered F1 multi-episode/
  multi-universe protocol to formally log the REJECT (G5/G7), then close the alt-bear regime axis. The
  strategy is confirmed at a single-episode-limited local optimum on free-data HL70; remaining levers are
  out-of-scope (new data / paid feeds) or operational.

Artifacts:
- script: `research/convexity_portable_2026-05-20/scripts/iter007_structform_check.py` (separation + per-episode)
- follow-up logs (full-book per-episode DD, HL70 per-fold + LOFO) in this insight; reproducible from the
  X121 engine with the `side AND alt30<btc30 → FLAT` branch.
