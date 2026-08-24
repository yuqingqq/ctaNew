# Cost/turnover optimization loop — charter

Self-paced research loop opened 2026-08-06. Premise from `docs/CONCLUSION_2026-08-03.md`: the gross XS
alpha-residual edge is real (OOS Sharpe +2.2..+2.5, CI excludes 0) but **net is marginal** (~+1.0 at 8 bps,
CI spans 0; ~0 at retail 24 bps). The doc's own verdict: *"the remaining leverage is EXECUTION COST … NOT
more free-data signal research."* This loop attacks cost/turnover/construction, not signal.

Discipline (inherited from the OB-flow and orthogonal-data loops, non-negotiable):
- Both eras always — RECENT (2025-10→2026-06) and OOS (2023-06→2025-09). A one-era win is a null.
- Clustered CIs: day-clustered for per-bar IC, 7d-block bootstrap for Sharpe/PnL series (positions in a slow
  book are autocorrelated; iid bootstrap would lie).
- Paired deltas vs the incumbent, CI on the DELTA, not on the two levels.
- Pre-register the gate and the falsifier BEFORE running. A result that only passes a post-hoc gate is a null.
- Separate gross-significant from net-marginal. Report turnover with every net number.
- Adopt nothing on one era, one universe, or one cost assumption.

Incumbent baseline (what every variant is compared against):
`per-symbol RidgeCV(V0_LEAN, target=xs_z(alpha_vs_btc_realized), HL=60d) → quintile L/S → 4h rebalance`,
turnover ~0.40/bar, book rank-IC +0.030 RECENT / +0.021 OOS.

Harness: `live/cost_loop_harness.py` (cached walk-forward preds, per-symbol cost map, book builder, CIs).
Iterations: `live/cl_iter*.py`.

---

## Directions (agenda)

| # | direction | status |
|---|---|---|
| D1 | **Static vs dynamic decomposition → low-turnover book.** ~90% of rank-IC sits in a persistent per-name tilt at 1% of the turnover. Does it survive at PORTFOLIO level, net of cost? | iter 1 — running |
| D2 | **Big-names universe ported into the deployed 4h stack.** Research validated top-20-40 ADV; the live script has no ADV cap ($3M/day floor, mean cost 11 bps, tail 35). | open |
| D3 | **Clip-size / capacity realism.** Live prices `cost_10k`; `cost_50k` median is 15 bps, mean 24. Find the AUM at which net → 0. | open |
| D4 | **Era-locked beta hedge re-quote.** `build_net_result.py:77` fits the hedge beta in-era (`np.polyfit` on the evaluation era) — the headline net is mildly optimistic. | open |
| D5 | **Preproc / decay-window mismatch.** `fit_preproc` uses unweighted full history; the fit applies a 60d half-life. Untested. | open |
| D6 | **Holding-horizon alignment.** If the edge is slow, the 4h label is the wrong target — test daily/multi-day labels for the persistent component. | open (gated on D1) |
| D7 | **Simplification.** If D1 holds, 175 Ridge models ≈ a trailing-rvol sort; an explicit factor drops the retrain/artifact-drift surface. | open (gated on D1) |

Closed upstream — do NOT re-open: V0 feature levers, Ridge internals/tail shaping, convex sizing, pooled
coefficient shrinkage, L2/order-book alpha, pump→dump, dispersion/vol-management/PPP.

---

## Iteration 1 — D1: does the persistent component survive at portfolio level, net?

**Context (measured 2026-08-06, real `gen_pred` pipeline, both eras).** Splitting each per-symbol Ridge
prediction into `stat` (PIT expanding mean of that symbol's own past preds, shift-1) and `dyn = pred − stat`:

| era | pred rank-IC | stat | dyn | Δ(stat−pred) day-CI | turnover pred → stat |
|---|---|---|---|---|---|
| RECENT | +0.0300 | +0.0279 | +0.0185 | −0.0022 [−0.0090,+0.0045] | 0.404 → 0.004 /bar |
| OOS | +0.0211 | +0.0190 | +0.0138 | −0.0021 [−0.0071,+0.0029] | 0.406 → 0.004 /bar |

Baseline reproduces the documented +0.030/+0.021 validity gate. The persistent tilt carries ~90% of the
ranking information at ~1% of the turnover, and the shortfall is not distinguishable from zero in either era.

**Hypothesis H1.** Because cost scales with turnover and the edge is mostly persistent, a slow book keeps most
of the gross alpha and converts materially more of it to net than the 4h book.

**This is a lead, not a result.** rank-IC is not net Sharpe. A near-static book is one concentrated persistent
factor bet: worse time-diversification, fatter drawdowns when the low-vol premium reverses, and more exposure
to survivorship (the same names are held for years). Portfolio-level + net + risk is what decides it.

### Assumptions that must hold (validated in this iteration)
- **A1 PIT.** `stat` uses only the symbol's own past preds (shift-1 expanding). Cross-check with a FROZEN
  variant whose tilt stops updating after the first 90 days of each era — if frozen collapses, the "static"
  result was really slow-dynamic and the turnover claim shrinks.
- **A2 Portfolio ≠ IC.** Quintile L/S gross Sharpe both eras, not just rank-IC.
- **A3 Net.** Per-symbol calibrated cost (`persym_cost_cal.csv`, cost_10k) + flat grid 6/8/12/24 bps.
- **A4 Not a plain rvol sort.** Benchmark a slow book built from −trailing-rvol rank alone.
- **A5 Universe.** Full eligible universe AND PIT trailing-ADV top-40.
- **A6 Risk.** maxDD, effective N, turnover, name-persistence — reported, not hidden.

### Pre-registered gates
- **G1 (gross).** Slow book gross Sharpe CI (7d-block) excludes 0 in BOTH eras.
- **G2 (net, the point of the loop).** Paired Δ(slow − fast) net Sharpe at 8 bps has CI excluding 0 in BOTH
  eras, on at least the top-40 universe.
- **G3 (attribution).** If the −rvol slow book matches or beats the model's slow book, the honest conclusion
  is "drop the ML for a factor sort", not "the ML tilt is the edge".
- **Falsifier.** G1 fails in either era, or G2's CI spans 0 in both eras → NULL, adopt nothing, record it.

### Result — **NULL. H1 REJECTED.** (`live/cl_iter1_static.py`, `live/state/cost_loop/iter1_*.csv`)

Harness validity first: the incumbent `fast` book reproduces the published numbers — OOS/top-40 gross Sharpe
**+2.59 [+1.32,+3.76]** (doc: +2.2..+2.5), net@per-symbol-cost **+1.08 [−0.19,+2.27] spans 0** (doc: ~+1.0,
CI spans 0), RECENT gross insignificant (doc: same). Era-locked hedge, 7d-block CI. The harness is sound.

Gross Sharpe, top-40 PIT-ADV universe, era-locked hedge:

| signal | turn/bar | RECENT gross | OOS gross | Δnet@8 vs fast RECENT | Δnet@8 vs fast OOS |
|---|---|---|---|---|---|
| fast (incumbent) | 0.400 | +0.40 spans0 | **+2.59 SIG** | — | — |
| stat (persistent tilt) | 0.010 | **+3.30 SIG** | +0.26 spans0 | +4.48 SIG | −0.79 spans0 |
| stat_ew | 0.013 | +2.20 spans0 | +0.17 spans0 | +3.38 SIG | −0.90 spans0 |
| stat_froz | 0.005 | +3.96 SIG | +0.98 spans0 | +4.94 SIG | −0.47 spans0 |
| rvol_slow | 0.006 | −0.01 spans0 | +0.22 spans0 | +1.18 spans0 | −0.83 spans0 |
| fast_ewma λ=0.7 | 0.162 | +1.07 spans0 | +1.94 SIG | +1.50 SIG | +0.23 spans0 |

**G1 FAIL, G2 FAIL** for every slow variant. The persistent tilt is **era-locked**: it is the best thing in the
table in RECENT and worthless in OOS (Δgross −2.32 [−3.78,−0.82], significantly WORSE than the incumbent). The
turnover saving is real (0.40 → 0.01/bar, per-symbol cost 3.07 → 0.08 bps/bar) but it buys nothing durable.
Exactly the failure mode the both-era rule exists to catch: at IC level the tilt tied in both eras, so a
one-era or IC-only test would have adopted it.

### Iteration 1b — why did IC-equivalence not transfer? (`live/cl_iter1b_diag.py`)
Two candidate explanations pre-registered; **both refuted**:
- **E1 breadth (rejected).** Book-return lag-1 autocorrelation is +0.00..+0.04 for the static book — variance
  inflation ×1.00-1.04. A static book is not making "one long bet"; there is no breadth deflation to blame.
- **E2 universe (rejected).** `stat`'s IC does not collapse inside top-40 OOS (+0.0124 vs fast +0.0186, Δ tie).

The real reason is **tail non-stationarity**: IC is a full-cross-section statistic, the book only holds the
extremes, and a static tilt holds the *same* extremes forever. Its per-bar payoff by year — OOS top-40:
2023 +3.4 bps, 2024 −0.4, 2025 −0.7; RECENT top-40: 2025 +9.0, 2026 +7.6. The tilt paid in 2023 and again in
2025-26 and was dead in between. **Lesson for this loop: rank-IC parity does not imply book parity; a signal
whose ranking is constant converts IC into a bet on one factor's episode structure.**

### Unexpected finding (opens iteration 2)
`rvol_slow` — a one-line slow sort on trailing realized vol, no ML — has **higher rank-IC than the 175-model
Ridge stack, significantly, in 3 of 4 era×universe cells**:

| cell | fast rank-IC | rvol_slow rank-IC | Δ day-CI |
|---|---|---|---|
| RECENT/top40 | +0.0108 | **+0.0501** | +0.0393 [+0.0226,+0.0556] BETTER |
| OOS/top40 | +0.0186 | **+0.0412** | +0.0225 [+0.0133,+0.0320] BETTER |
| OOS/all | +0.0211 | **+0.0402** | +0.0191 [+0.0121,+0.0260] BETTER |
| RECENT/all | +0.0300 | +0.0366 | +0.0066 [−0.0021,+0.0153] tie |

…yet its **book** Sharpe is ~0 in every cell. That gap is diagnostic, not contradictory: the books above are
built on RAW forward returns with a single book-level beta hedge, while the label the IC is measured against
(`alpha_vs_btc_realized`) is a PER-NAME beta residual. A pure low-vol book is exactly the case where those two
differ most — `docs/CONCLUSION_2026-08-03.md` says so explicitly ("the raw low-vol book is short-beta and
bleeds; targeting the BTC-residual + hedging beta is what unlocks it"). So iteration 1's book construction is
mis-specified for this signal, and that is a flaw in MY harness, not a property of the signal.

---

## Iteration 2 — D7/D4: is the incumbent's edge just a slow low-vol factor, and is that factor real?

**H2.** A slow, per-name beta-neutral low-vol sort matches or beats the incumbent per-symbol-Ridge book on NET
Sharpe in both eras inside the deployable top-40 universe — i.e. the 175-model stack is reproducible by one
factor, at a fraction of the operational surface (no monthly retrain, no artifact drift, no 14-feature panel).

**Order of work is deliberate: the artifact control runs FIRST.** If the low-vol IC advantage is an artifact of
how the residual label is built, there is nothing to test at book level and the thread dies cheaply.

### Assumptions to validate
- **A1 harness (must fix before any book claim).** Books must be built on the quantity the strategy actually
  farms — per-name BTC-residual returns — not raw returns with one book-level beta. Refit the INCUMBENT book
  both ways; the residual version must still reproduce the documented OOS gross +2.2..+2.5.
- **A2 label artifact.** `alpha_A = fwd − β̂·btc_fwd` with β̂ from a trailing 1d 5m regression. β̂ is noisier for
  high-vol names, so residual β-error correlates with rvol; if BTC's drift then leaks into the residual, a
  spurious vol premium appears. Control: IC(rvol_slow, alpha_A) split by BTC forward-return tercile. A genuine
  premium is present in all three; a β-artifact flips sign with BTC direction.
- **A3 raw-return control.** IC(rvol_slow, raw return_pct) — expected ≈0 or negative per the documented
  mechanism; a *large positive* raw IC would mean the residual is not what is doing the work.
- **A4 net.** Per-symbol calibrated cost + turnover, both eras, 7d-block CIs.

### Pre-registered gates
- **G1 (harness validity).** Residual-return incumbent book reproduces OOS/top-40 gross ≈ +2.2..+2.6.
- **G2 (artifact control).** IC(rvol_slow, alpha_A) keeps the same sign in all three BTC terciles in BOTH eras.
- **G3 (gross).** rvol factor residual-book gross Sharpe CI excludes 0 in BOTH eras, top-40.
- **G4 (net vs incumbent).** Δ(rvol − incumbent) net@per-symbol-cost CI excludes 0 in ≥1 era and is not
  significantly negative in the other.
- **Falsifier.** G2 fails → label artifact, kill the thread and record it. G3 fails → the IC advantage does not
  convert, record and stop. Adopting requires G1+G2+G3+G4.

### Result — **G1 PASS, G2 PASS, G3 FAIL, G4 FAIL. H2 rejected; the factor is REAL but does not convert.**
(`live/cl_iter2_factor.py`, `live/state/cost_loop/iter2_*.csv`)

**G1 harness validity — PASS.** Rebuilt on per-name BTC-residual returns, the incumbent OOS/top-40 book gives
gross **+2.51 [+1.25,+3.68]** vs +2.59 on the raw+book-beta construction. So the mis-specification I suspected
in iteration 1 was *not* material for the incumbent — both constructions reproduce the documented +2.2..+2.5.
Worth knowing: the incumbent's edge does not depend on which of the two hedging conventions you use.

**G2 artifact control — PASS.** The low-vol IC is not a residual-label beta artifact:

| cell | rvol_slow IC (all) | btc_down | btc_flat | btc_up | vs RAW fwd return |
|---|---|---|---|---|---|
| RECENT/top40 | +0.0501 | +0.0363 | +0.0466 | +0.0675 | +0.0521 |
| OOS/top40 | +0.0412 | +0.0359 | +0.0439 | +0.0438 | +0.0348 |
| OOS/all | +0.0402 | +0.0350 | +0.0398 | +0.0458 | +0.0337 |

Same sign in every BTC tercile, CIs exclude 0, and it is just as strong against RAW forward returns. This is
genuine information, ~2-4× the incumbent's rank-IC, and it costs nothing to compute.

**G3/G4 FAIL — and this is the finding.** The same signal as a quintile book earns nothing:

| era/tier | signal | turn | gross Sharpe | net@per-sym cost | maxDD (bps) |
|---|---|---|---|---|---|
| OOS/top40 | fast (incumbent) | 0.408 | **+2.51 SIG** | +0.98 spans0 | −2428 |
| OOS/top40 | rvol_slow | 0.002 | +0.35 spans0 | +0.35 spans0 | **−8909** |
| RECENT/top40 | fast | 0.400 | +0.96 spans0 | −0.41 spans0 | −2973 |
| RECENT/top40 | rvol_slow | 0.006 | +0.32 spans0 | +0.30 spans0 | −2777 |

A signal with 2-4× the rank-IC produces ~1/8 the book Sharpe, measured under **two independent book
constructions** (raw + book-level beta hedge in iter1; per-name residual here) that agree to ~0.03 Sharpe.
So the gap is a property of the construction↔signal interaction, not of the hedging convention.

**Quantitative hypothesis it hands us.** Sorting *on* volatility makes the two legs variance-mismatched: the
short leg holds the raciest names, so the book's standard deviation is roughly the short leg's, and Sharpe is
deflated by about the leg-vol ratio. If that ratio is ~3, a "true" balanced Sharpe near +1 would show up as
~+0.33 — which is what both constructions measure (+0.30..+0.35 in all four cells). The 3.6× worse maxDD
(−8909 vs −2428) points the same way. That is a mechanism with a number attached, so it is falsifiable.

---

## Iteration 3 — is the low-vol factor's ~0 book Sharpe a leg-variance artifact of equal weighting?

**H3.** The rvol factor's book Sharpe is suppressed by leg variance mismatch inherited from sorting on vol.
Risk-weighting within legs (or a continuous full-cross-section construction) recovers a materially higher
Sharpe from the same IC.

**Not a re-run of closed work.** "Convex/soft rank sizing" and "volatility-management" are both closed nulls —
but both were tested **on the incumbent prediction**, whose legs are already variance-balanced because it does
not sort on vol. The mismatch here is *structural to a vol-sorted book*, so the prior does not transfer. The
incumbent is carried through every construction in this iteration as the control: if inverse-vol weighting
lifts the incumbent too, the closed null is contradicted and I should trust the closed result over this run.

### Constructions tested (each × {fast, rvol_slow, stat} × both eras × {top40, all})
- `eq` equal-weight quintile (iteration-2 baseline)
- `ivol` inverse-vol weights within each leg (∝ 1/rvol_7d, PIT), normalized per side
- `cont` continuous rank weights across the FULL cross-section, inverse-vol scaled, dollar-neutral — harvests
  the middle of the distribution instead of only the tails
- `volscale` per-name position scaled to equal risk contribution

### Assumptions to validate
- **A1 mechanism is measured, not assumed.** Report the realized short-leg/long-leg return-vol ratio per book.
- **A2 control.** Same constructions on the incumbent; a generic lift would mean this is not vol-sort-specific.
- **A3 turnover.** Risk weights move with rvol; confirm turnover stays near the slow book's 0.002-0.006/bar.

### Pre-registered gates
- **G1 (mechanism).** Measured leg-vol ratio ≥ 1.5 for the rvol `eq` book in BOTH eras. If it is ≈1, the
  variance story is wrong and the IC lives in the middle of the distribution — record and stop the thread.
- **G2 (recovery).** Paired Δ(ivol or cont − eq) gross Sharpe CI excludes 0 for rvol_slow in BOTH eras.
- **G3 (level).** The recovered rvol book's gross Sharpe CI excludes 0 in BOTH eras.
- **G4 (net).** Its net@per-symbol-cost CI excludes 0 in BOTH eras.
- **Falsifier.** G1 fails → mechanism refuted. G2 or G3 fails → the factor's IC is not harvestable by
  reweighting; record as a null and close the D7 simplification thread.

### Result — **G1 PASS, G2 FAIL. Mechanism confirmed present but NOT binding. D7 thread closed as a null.**
(`live/cl_iter3_weighting.py`, `live/state/cost_loop/iter3_*.csv`)

**G1 PASS — the mechanism is real and measured.** Realized short-leg/long-leg return-vol ratio for the rvol
equal-weight book: **2.50 RECENT / 1.83 OOS** (the incumbent's is 1.13-1.34). Sorting on vol does mismatch the
legs exactly as predicted, and it does deflate the Sharpe denominator.

**G2 FAIL — fixing it recovers nothing.** Four constructions, paired Δ vs equal-weight, rvol_slow top-40:

| Δ(construction − eq) gross | RECENT | OOS |
|---|---|---|
| ivol | +0.29 [−0.48,+0.80] | +0.07 [−0.20,+0.36] |
| cont | +0.55 [−0.47,+1.58] | +0.18 [−0.36,+0.69] |
| volscale | +0.18 [−1.22,+1.22] | −0.12 [−0.90,+0.74] |

`volscale` drove the leg-vol ratio from 2.50 → 1.10 and the Sharpe did not move (+0.32 → +0.50, CI spans 0).
So the variance mismatch is present, correctable, and **not** what is suppressing the factor. The low-vol IC
simply does not live in the tradeable tails — it is a full-cross-section ordering effect.

**A2 control — the closed nulls hold.** Inverse-vol and volscale *significantly hurt* the incumbent in RECENT
(−1.05 [−1.79,−0.05] and −1.64 [−3.05,−0.23]) and in RECENT/all (−1.17, −0.94). This independently reproduces
the closed "vol-weighting / convex-sizing rejected" results, which is a good sign for the harness. One cell
(incumbent `cont`, OOS/top40, Δ+0.64 [+0.10,+1.29]) is positive but flips sign in RECENT → null by loop rule.

**What iterations 1-3 jointly establish.** The incumbent's value is **tail selection**, not overall ranking: a
plain low-vol sort out-ranks it 2-4× on IC yet earns ~+0.3 as a book, while the incumbent earns +2.5 from a
lower IC. That is consistent with the closed finding that the edge sits in the *prediction* tail (K=2-3
optimal). Signal-side simplification is therefore not available — the ML is doing tail work a factor sort
cannot. **D1 and D7 both close as nulls.**

---

## Iteration 4 — D2+D3+D4: the deployability / capacity frontier

**H4.** With the big-names restriction and turnover control, net Sharpe is significantly > 0, and the capacity
frontier (clip size at which net dies) is at least $50k. Pre-registered 3×3 grid (tier × control) × 3 calibrated
clip tiers, incumbent signal only, all 27 cells reported, a config must pass in BOTH eras and be the SAME config
in both.

### Result — **G1 FAIL as written (0/9 both-era passers) — but the failure is a POWER failure, not instability.**
(`live/cl_iter4_capacity.py`, `live/state/cost_loop/iter4_capacity.csv`)

Net Sharpe @ cost_10k, per-name residual books, 7d-block CI:

| tier / control | turn | cost bps/bar | OOS net | RECENT net |
|---|---|---|---|---|
| all / none *(≈ the live config)* | 0.41 | 4.26 | **−0.91** [−2.21,+0.36] | −0.62 [−3.43,+1.58] |
| all / ewma0.7 | 0.16 | 1.64 | −0.16 [−1.48,+1.22] | +0.78 [−2.07,+3.09] |
| top40 / none | 0.41 | 3.18 | +0.98 [−0.29,+2.11] | −0.41 [−2.90,+1.67] |
| top40 / ewma0.7 | 0.16 | 1.24 | +1.28 [−0.03,+2.56] | +0.87 [−1.63,+2.98] |
| **top40 / band** | 0.26 | 2.11 | **+1.72 [+0.34,+2.98] SIG** | +0.68 [−1.73,+2.51] |
| **top20 / band** | 0.18 | 1.26 | **+1.56 [+0.22,+2.85] SIG** | +0.85 [−1.98,+3.37] |
| top20 / ewma0.7 | 0.16 | 1.11 | +1.26 [+0.04,+2.44] SIG | +1.07 [−1.56,+3.44] |

**Capacity (the D3 answer) — far better than feared for the big-name configs:**

| config | net @ $10k | @ $50k | @ $100k (OOS) |
|---|---|---|---|
| top40 / band | +1.72 SIG | +1.43 SIG | +1.38 SIG |
| top20 / band | +1.56 SIG | +1.41 SIG | +1.37 SIG |
| all / none | −0.91 | −4.41 neg | −4.97 neg |

Cost per bar for top40/band moves only 2.11 → 2.92 → 3.06 bps from $10k to $100k clips, because the band cuts
turnover to 0.26 and the ADV restriction keeps every name cheap. On the full universe cost *doubles* (4.26 →
9.53) and the book is deeply negative at every clip size.

**Reading it honestly.** No config passes the pre-registered both-era gate, so nothing is adopted on this
iteration's evidence. But the reason differs from iteration 1: there, RECENT and OOS *disagreed in sign* with
the delta significantly negative (real instability). Here OOS is consistently positive and significant for three
configs while every RECENT CI is ~±2.5 wide and contains them — RECENT (≈1,400 bars, 8 months) cannot resolve a
±1 Sharpe difference. `docs/CONCLUSION_2026-08-03.md` already calls RECENT "uninformative". **Methodological
correction for this loop: the both-era rule conflates "unstable" with "underpowered".** It did its job in
iteration 1 and is the wrong instrument here.

**Directional result worth carrying forward (not yet adopted):** the currently-deployed universe choice is the
single worst cell in the table. Full-universe/no-control is −0.91 OOS net; top-40 + band is +1.72. That is the
D2 lever, quantified on the residual book.

---

## Iteration 5 — fix the power problem with a genuine hard split

**Why.** Iteration 4's gate cannot distinguish "does not work in RECENT" from "RECENT cannot measure it", and
its OOS winners were *selected on OOS* — circular. Both problems have one fix: a chronological hard split with
selection and evaluation on disjoint windows, sized so the evaluation window has real power.

**Design (pre-registered).**
- **Selection window** 2023-06 → 2024-12 (~2,600 bars). Pick the SINGLE best config by net Sharpe @cost_10k.
- **Held-out window** 2025-01 → 2026-06 (~3,200 bars, spans the OOS tail *and* all of RECENT). Evaluate that
  one config. Nothing about the held-out window informs the choice.
- All 9 configs reported on both windows for transparency, but only the pre-committed selection rule counts.

### Pre-registered gates
- **G1.** Selected config's held-out net Sharpe CI (7d-block) excludes 0 @cost_10k.
- **G2.** Same @cost_50k (capacity claim survives honest selection).
- **G3 (texture, not a gate).** Held-out net Sharpe sign-consistent across calendar years 2025 and 2026.
- **Falsifier.** G1 fails → the iteration-4 OOS positives do not survive honest selection; record as a null and
  the deployability branch closes negative.

### Result — **G1 FAIL, G2 FAIL, G3 FAIL. The iteration-4 positives do NOT survive honest selection.**
(`live/cl_iter5_hardsplit.py`, `live/state/cost_loop/iter5_hardsplit.csv`)

Pre-committed rule selected **top40/band** on 2023-06→2024-12 (select net@10k **+2.00 [+0.40,+3.56] SIG**,
gross +2.73, turn 0.258). Held out on 2025-01→2026-06 (2,712 bars):

| held-out | value |
|---|---|
| gross | +1.55 [−0.16,+3.12] spans0 |
| net @$10k | **+0.85 [−0.86,+2.43] spans0** |
| net @$50k | +0.53 [−1.18,+2.09] spans0 |
| net @$100k | +0.48 [−1.23,+2.04] spans0 |
| by year | 2025 Sharpe +1.30 (+4.36 bps/bar) · 2026 Sharpe −0.13 (−0.40 bps/bar) |

**Every one of the 9 configs has a held-out CI spanning 0 at every clip tier.** The gross edge itself halves
out of selection (+2.73 → +1.55, no longer significant). This is "not demonstrable", not "demonstrably absent"
— the held-out CI is ±1.7 wide and the point estimate is positive — but it is the honest answer, and it is
obtained under a design where the configuration never saw the evaluation data.

**What DOES replicate across the split — the relative ordering:**

| config | select net@10k | holdout net@10k | select→holdout @$100k |
|---|---|---|---|
| all / none *(≈ the live config)* | −0.38 | −1.49 | −4.31 → −5.82 |
| all / ewma0.7 | +0.17 | −0.06 | −1.44 → −1.88 |
| top40 / ewma0.7 | +1.39 | +0.94 | +1.13 → +0.64 |
| **top40 / band** | **+2.00** | **+0.85** | +1.71 → +0.48 |

The *direction* of D2/D3 replicates cleanly even though the level is not significant: trading the full
universe with no turnover control is the worst cell in both windows and is the only configuration that gets
dramatically worse with clip size (−1.49 → −5.82 from $10k to $100k), while big-names + band is nearly
clip-size-flat (+0.85 → +0.48). **That is a robust relative statement about construction, not a claim of a
significant net edge.**

---

## Iteration 6 — D5: preprocessing window vs decay half-life (the last model-side lever)

**H5.** `x6.fit_preproc` computes winsor bounds, z-scores and the heavy-tail empirical CDF on the **entire
unweighted** per-symbol training history, while the RidgeCV fit weights samples with a 60-day half-life. The
scaling therefore describes a sample the model barely uses; matching the preprocessing window to the effective
sample should improve the fit. Never tested (the closed Ridge-internals audit covered the alpha grid, feature
pruning and tail shaping — not the preproc window).

**Variants** (single axis, no knob-fitting): `full` (incumbent) vs `trail120` (≈2 half-lives) vs `trail240`.

### Pre-registered gates
- **G1 (book level).** Paired Δ rank-IC vs incumbent, day-clustered CI excludes 0 in BOTH eras.
- **G2 (portfolio).** Held-out (2025-01→2026-06) net@10k on top40/band improves with paired block CI > 0.
- **Falsifier.** G1 fails → the mismatch is immaterial; D5 closes as a null and the loop's agenda is exhausted.

### Result — **G1 FAIL, G2 FAIL. D5 is a null, and the hypothesis was wrong in DIRECTION.**
(`live/cl_iter6_preproc.py`)

| variant | RECENT rank-IC | OOS rank-IC | held-out net@10k (top40/band) |
|---|---|---|---|
| full (incumbent) | +0.0302 | +0.0210 | +0.85 [−0.86,+2.43] |
| trail120 | +0.0303 Δ+0.0001 spans0 | +0.0201 **Δ−0.0009 [−0.0016,−0.0002] neg** | −0.13, Δ **−0.98 [−1.64,−0.26] neg** |
| trail240 | +0.0307 Δ+0.0005 spans0 | +0.0206 Δ−0.0003 spans0 | +0.49, Δ −0.36 spans0 |

Matching the preprocessing window to the decay half-life does not help — it *hurts*, significantly so at 120
days on both metrics. The mechanism is clear in hindsight and worth recording: the preproc stats (winsor
quantiles, mean/sd, heavy-tail empirical CDFs) are **nuisance parameters**, and they want the largest sample
available. The 60-day decay expresses which *relationships* are currently relevant, not which *scale* is
correct. Shortening the scaling window adds estimation noise without adding relevance. The incumbent's
full-history preproc is right as-is.

---

# Loop close-out (2026-08-06)

All seven directions tested. **Nothing adopted.** The one durable output is a relative construction result.

| # | direction | verdict |
|---|---|---|
| D1 | persistent-tilt / low-turnover book | **NULL** — era-locked: +3.30 RECENT vs +0.26 OOS, Δgross −2.32 [−3.78,−0.82] |
| D2 | big-names universe | **direction replicates across a hard split; level not significant** |
| D3 | clip-size capacity | big-names+band is nearly clip-flat (+0.85→+0.48, $10k→$100k); full universe collapses (−1.49→−5.82) |
| D4 | era-locked vs in-era hedge | **immaterial** — +2.51 vs +2.59 on the same book; the doc's convention is not the issue |
| D5 | preproc window vs decay HL | **NULL** — and directionally wrong; incumbent is right |
| D6 | horizon alignment | not run — gated on D1, which failed; horizon is covered upstream (`WINDOW_HORIZON_RESULTS.md`) |
| D7 | factor-sort simplification | **NULL** — the low-vol factor out-ranks the model 2-4× on IC and earns ~+0.3 as a book |

### The two results worth keeping

**1. The incumbent's value is TAIL SELECTION, not ranking.** A one-line slow low-vol sort beats the 175-model
Ridge stack on rank-IC by 2-4× in three of four era×universe cells (OOS/top40 +0.0412 vs +0.0186), survives the
label-artifact control (same sign in every BTC tercile, and just as strong against raw returns), and still
earns ~+0.3 as a book against the model's +2.5. Four reweighting schemes — including one that drove the
leg-vol ratio from 2.50 to 1.10 — recovered nothing. Signal-side simplification is therefore unavailable: the
ML is doing tail work that a factor sort cannot, which is consistent with the closed finding that the edge
sits in the *prediction* tail (K=2-3 optimal). **Do not re-open "replace the model with a factor".**

**2. Construction ordering replicates; net profitability does not.** Under a chronological hard split
(select 2023-06→2024-12, evaluate 2025-01→2026-06, config never sees the evaluation window):

| config | select net@10k | holdout net@10k | @$10k→$100k holdout |
|---|---|---|---|
| all / none *(≈ `run_convexity_v4_live.sh`)* | −0.38 | −1.49 | −1.49 → −5.82 |
| top40 / ewma0.7 | +1.39 | +0.94 | +0.94 → +0.64 |
| top40 / band | +2.00 SIG | **+0.85 [−0.86,+2.43] spans0** | +0.85 → +0.48 |

Every one of the 9 configs has a held-out CI spanning 0 at every clip tier, and the *gross* edge halves out of
selection (+2.73 → +1.55, no longer significant), with 2026 at −0.13. So: **no configuration demonstrates a
net edge out of selection.** What does replicate is the ordering — the currently-deployed universe/no-control
choice is the worst of the nine and the only one that collapses with clip size.

### Harness validity (why to believe the nulls)
- Reproduces the published book-level numbers: rank-IC +0.0302 RECENT / +0.0210 OOS (docs: +0.030/+0.021).
- Reproduces the published portfolio numbers: OOS/top-40 gross **+2.51..+2.59** (docs: +2.2..+2.5), net at
  majors cost ~**+1.0 with CI spanning 0** (docs: same).
- Independently reproduces two *closed* nulls: inverse-vol and volscale weighting significantly hurt the
  incumbent (−1.05, −1.64 RECENT), matching the closed convex-sizing / vol-management rejections.

### Method lessons for the next loop
1. **rank-IC parity does not imply book parity.** A signal whose ranking is constant converts IC into a bet on
   one factor's episode structure (iteration 1). Test at portfolio level before believing an IC result.
2. **The both-era gate conflates "unstable" with "underpowered."** It correctly killed D1 (a genuine sign flip
   with a significant negative delta) and was the wrong instrument for D4/iteration 4, where RECENT's 1,400
   bars simply cannot resolve ±1 Sharpe. Use a chronological hard split for level questions, the era gate for
   stability questions.
3. **Selecting a config on the window you score it on inflates it by ~1.2 Sharpe here** (+2.00 select → +0.85
   held out). Always pre-commit the selection rule to a disjoint earlier window.
4. Nuisance-parameter estimators (scaling, quantiles, CDFs) want the largest sample; sample-decay logic that
   applies to the *model* does not transfer to them (iteration 6).

### What is NOT closed by this loop
Execution mechanics below the taker-slippage model — maker/limit fills, queue position, venue routing. The
cost calibration in `live/state/v3loop/persym_cost_cal.csv` is a taker depth model; fill-probability modelling
needs data this repo does not own. That remains the only identified route to lifting net, exactly as
`docs/CONCLUSION_2026-08-03.md` states. This loop found nothing on the *construction* side to add to it.

All scripts uncommitted, in `live/cl_iter*.py` + `live/cost_loop_harness.py`; results in
`live/state/cost_loop/*.csv`, logs in `live/state/cost_loop_iter*.log`.

---

# Addendum (2026-08-06) — market cap and open interest

Asked post-loop: has market cap ever been used, and how does OI relate to it?
Scripts: `live/mc_oi_probe.py`, `live/mc_oi_universe.py`, `live/mc_oi_incremental.py`, `live/mc_fetch_365d.py`.

**Market cap has never been used.** The repo's only CoinGecko call (`ml/research/alpha_vBTC_check_universe.py`)
enumerates live perps; it does not read market cap. `agents_system/orchestrator/iteration_log.md:764` flags
float / circulating-supply / tokenomics as a known gap. Binance Vision has no supply data, so MC is not
derivable from owned data, and CoinGecko's free tier caps `market_chart` at 365 days (error 10012) — a PIT
market-cap panel reaching the OOS era is a paid feed.

**OI has been used only as a CHANGE** (`oi_chg_1d/3d`, `oi_z_30d`, `oi_price_div`; `build_positioning_axes.py`,
`orth_iter*`). The OI **level** was never used, despite being PIT and complete (176 syms, 5-min, 2021-01→2026-07).

**OI ↔ MC, current snapshot, n=146** (snapshot used only to calibrate the proxy — ranking history by today's
caps would be look-ahead):

| | log-log Pearson | Spearman |
|---|---|---|
| OI value ↔ market cap | **+0.896** | +0.833 |
| ADV ↔ market cap | +0.797 | +0.693 |

OI/MC: p5 0.66%, median 4.13%, p95 15.2%; **corr(log OI/MC, log MC) = −0.652** — small caps carry
proportionally far more futures OI (CHIP 27.9%, 0G 21.0%, XPL 20.3% vs BTC 0.56%, BNB 0.45%, TRX 0.36%).
**corr(log OI/MC, log OI/ADV) = −0.36, wrong sign — OI/ADV is NOT a proxy for OI/MC.**

### Tests run (both negative)
**Universe by OI value instead of trailing ADV — REJECTED.** 86% overlap; SELECT tie (+2.05 vs +2.03,
Δ +0.02 spans0); HOLDOUT worse (−0.42 vs +0.84, Δ −1.26 [−2.73,+0.16]).

**OI/ADV as an incremental signal — REJECTED on both gates.** It first looked like the session's only both-era
incremental IC (Δ +0.0117 [+0.0057,+0.0177] OOS, +0.0224 [+0.0159,+0.0293] RECENT). Then:
- **C1 orthogonality FAIL.** xs corr(OI/ADV rank, vol rank) = **−0.54 OOS / −0.59 RECENT**. Residualized on
  vol rank its IC is **+0.0028 [−0.0038,+0.0094] spans0 in OOS** (+0.0128 SIG RECENT). The raw vol factor is
  *stronger* than OI/ADV (long-low-vol +0.0383 OOS / +0.0591 RECENT). OI/ADV is the known low-vol factor in a
  new hat — high OI relative to churn just means a calm, slowly-turned-over name.
- **C2 conversion FAIL.** Book level, top40/band, hard split: SELECT Δ(blend−pred) −0.71 [−2.37,+0.97];
  HOLDOUT +1.33 [−0.14,+2.97]. Opposite signs, both span 0 — noise. (The blend does halve turnover, 0.255 →
  0.143, because the OI/ADV component is nearly static — the same slow-component pattern that failed in
  iteration 1, and it fails the same way.)

---

# Addendum 2 (2026-08-06) — passive (maker) execution: measured, not assumed

`live/maker_exec_probe.py`. The loop's terminal finding was that execution cost is the binding constraint but
that fill-probability data was not owned. This prices passive execution on the **actual trades the held-out
top40/band book makes** (22,542 trades, 113 syms, 2025-01→2026-07), from owned 5m klines, using implementation
shortfall against the same decision price the backtest uses.

| scenario | fill % | bps/unit traded | bps/bar | held-out net Sharpe |
|---|---|---|---|---|
| taker, immediate (cost_10k) | 100 | 8.35 | 2.30 | +0.85 [−0.86,+2.43] |
| passive 60m, 1bp inside | 97.3 | 4.45 | 1.23 | +1.17 [−0.53,+2.76] |
| passive 60m, 2bp inside | 96.7 | 4.16 | 1.15 | +1.20 [−0.51,+2.78] |
| passive 60m, 5bp inside | 94.3 | 3.93 | 1.08 | +1.22 [−0.49,+2.80] |
| passive 15m, at touch-price *(optimistic)* | 99.0 | 2.79 | 0.77 | +1.31 [−0.40,+2.89] |
| **gross (zero cost ceiling)** | — | 0 | 0 | **+1.55** |

- **Passive execution roughly halves the cost** (8.35 → ~4 bps/unit) and recovers **~half the net-to-gross
  gap** (+0.85 → ~+1.2 against a +1.55 ceiling). All still span 0 held-out — it improves the point estimate,
  it does not manufacture significance.
- **Adverse selection is visible and self-limiting.** Wider offsets buy a better fill price (cost-when-filled
  2.00 → −3.00 bps) but lose fills (99.0% → 94.3%), and the misses must be chased at ~8-10 bps. Net cost is
  flat across offsets (3.93-4.55). The ~5% that don't fill are exactly the ones that ran away.
- **Patience is not the constraint.** Beyond 15 minutes nothing improves (fill 99.0% → 99.6%, cost flat). The
  book's alpha has a 4h label and 24h hold, so the patience budget is 16-96× the observed fill time.

**The load-bearing caveat.** Klines tell us the price *touched* a level, not that our order *filled* — no queue
position. So these fill rates are an UPPER bound and the true passive cost lies between the passive rows and
the taker row. Settling it needs real fill data, which the existing paper-trading harness could generate by
posting `GTX` (post-only) orders in small size and recording actual fills. That is the cheapest route to the
one dataset the loop identified as missing.

**Frequency implication.** Passive execution does NOT require running faster — it requires patience, which we
have in surplus. What cheaper execution changes is that the turnover suppression the loop adopted to fight an
8.35 bps taker cost (band 0.26/bar, EWMA 0.16/bar) is less necessary; held-out gross is near-identical across
controls (none +1.57 / ewma +1.58 / band +1.55), so re-optimising turnover under ~4 bps is worth one test but
cannot be transformative. **Execution work is bounded above by the +1.55 gross.** Closing the gap to a 6-Sharpe
book is a risk-model / effective-breadth problem, not an execution problem.

### The open question, and whether it is worth money
The one genuinely untested axis is the **leverage-intensity residual**: OI/MC after controlling for size and
vol. Owned data cannot proxy it (OI/ADV is −0.36 correlated with it, and OI value is 0.90 correlated with MC,
i.e. it *is* size). Testing it properly needs paid PIT market-cap history. Before spending: `mc_fetch_365d.py`
pulls the free 365-day window (covers RECENT only) so the residual can be screened at IC level. **Single-era
⇒ diagnostic only, never adoptable** — its sole job is to answer "is there anything here worth buying history
for?" A null there is a cheap reason not to buy; a hit is not a result, only grounds for the paid test.
