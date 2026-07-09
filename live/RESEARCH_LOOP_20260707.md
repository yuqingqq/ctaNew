# Research loop — 2026-07-07 (12h autonomous run)

Ledger for the self-paced validation loop. Flow per item: research/design → adversarial design
review (subagent) → test (bot replays, paired dual-window CI) → adversarial results review
(subagent) → evaluation → docs update → doc-hygiene review (subagent). Canonical results land in
`V4_PERFORMANCE.md` / `REGIME_DISCOVERY_FRAMEWORK.md`; this file tracks loop state only.

Baseline stack for all paired tests: **KEEPSET4 + deep-bull mom1d_long** (the wired v4 forward
config, `run_convexity_v4_live.sh`), clean v4 preds, recent (2025-10-04..2026-06-30) + OOS
(2023-01-01..2025-09-30), fee 4.5/fill. Tests ran at 1.0× gross; the live 0.5× cap is a
post-Q2 consequence, not part of the test baseline.

## Queue

| # | item | status |
|---|---|---|
| Q1 | Side OI-drain de-gross bot test | CLOSED — rejected at reproduction gate (Iter 3) |
| Q2 | 2022 holdout window (final stack) | CLOSED — FAIL, consequences landed (gross cap 0.5×; Iter 4-5) |
| Q3 | Deep-bull overlay robustness (placebo + K/gross band) | CLOSED — ranking unproven OOS, exposure stands (Iter 2) |
| Q4 | Metrics cache rebuild + positioning re-screen | CLOSED — cache restored 174/176, loader hardened, re-screen done (Iter 3, 5) |
| Q5 | Leg-level OI features screen | CLOSED — screening only; 3 caveated watchlist entries (Iter 6-7) |
| Q6 | Side-flat DD-lever characterization | CLOSED — DD lever, mechanism unresolved (mostly exposure-dose; Iter 8-9) |

## Post-loop addendum (2026-07-08) — squeeze-surrender + per-regime holds: REJECTED at pre-tests

Two candidates from the tail-anatomy session, both through adversarial design review, both killed
by the pre-registered zero-DoF pre-tests (no bot code written):

**A. Conditional short-leg early exit ("squeeze surrender") — REJECTED.** The motivating
screening numbers contained a conditioning-bar bug (the reviewer caught the non-reconciliation by
pure arithmetic before the code was re-checked: the quoted E[rest|bad]=−287 included the trigger
bar itself) AND pooled untraded bull cycles. On the REALIZED KEEPSET4+overlay sleeves, per
regime × window with a convention-pinned trigger (worst-quartile bar-1, ~25% fire):
E[rest-of-hold | fired] = **+89 (side) / +198 (bear) recent; −3 / +8 OOS** — fired shorts RECOVER;
exiting forfeits the recovery and pays ~28-29 bps/fire exit+re-entry. Permutation null not
rejected anywhere (p 0.34-0.96) — zero mechanism evidence. The within-hold loss-continuation is
a BULL-only phenomenon, already amputated by bull0. Jaccard with DD-stop ≤0.25 (not redundancy —
just nothing there to be redundant with).

**B(a) side long-leg 8h hold — BLOCKED** (design review: reopens closed §6.2 rejection, breaks
side book neutrality by construction, arithmetic-negative: −4 bps/cyc saved vs ~3× long-leg
turnover at 14.5 bps/fill). Not pursued.

**B(b) bear 48h hold — BLOCKED** until the §7 forward bear-farm confirmation completes (would
contaminate the program's only remaining out-of-era test), and independently fails the
concentration pre-test: the +62-75 bps/cyc bars-7-12 marginal is positive every period BUT
top-5%-cycle share = 110-148% (tail-carried). Direct prior art (536e8fa hold sweep) already
failed the unscoped version. May be revisited as ONE pinned forward-test cell after the
confirmation window.

Term-structure facts (screening, retained for the record): side-LONG tip dies after ~8h
(+8@8h → +4@24h); side-SHORT and bear compound through 48h; bull negative past 4h. These are
diagnostics, not levers — every implementable version failed the ladder above.

## Post-loop addendum 2 (2026-07-08) — CUSUM monitor: built + validated; CUSUM-GATE lever: REJECTED at pre-test

`live/tip_exceedance_monitor.py` (exceedance-CUSUM, blind-validated — see V4_PERFORMANCE §7) stays
MONITORING-ONLY. The lever version ("zero side gross 30d post-m3-alarm") died at the zero-DoF
pre-test on realized v4-stack cycles: in 3 of 4 OOS alarm windows the stop/gate had ALREADY
de-grossed 75-100% of cycles (realized damage left: −196/+447/−274/0 bps — nothing to save), and
the one full-gross window (2025-10-24, recent) realized **+4,299** which the gate would have
forfeited. Net effect of wiring: **−4,276 bps. REJECT.** Lesson (rhymes with squeeze-surrender
A7): equity-based reactive defenses harvest the same clustering FASTER (drawdown is immediate;
event-rate certification needs weeks) — the CUSUM's residual value is attribution and the signals
equity can't see (jackpot-rate side, rate shifts while already flat), i.e., §7 human review.

## Post-loop addendum 7 (2026-07-08, post-review FINAL) — feature-window program: 6 cells, 0 promotions

Canonical results: **live/FEATURE_TUNING_RESULTS.md** (matched-population scoreboard + full
review appendix). The adversarial results review found a HIGH scorer defect (F1: the paired
per-cycle Δ was computed on different symbol populations per arm — the harness's variant
train-row minimum drops symbol-folds, so books differ per cycle: C2 43% of OOS cycles, T1 100%).
Scorer fixed (both arms on incumbent ∩ variant ∩ fwd); all cells re-scored. Consequences:
**C2 downgraded REJECT → KEEP-incumbent** (matched OOS Δ −0.0012, CI [−.0027, +.0004] crosses 0
— "entirely negative" was a population artifact; "affirmatively worse" unsupported);
**C3 OOS CI now EXCLUDES 0** (+0.0016 [+.0002, +.0030], hit 21/33) but stays non-promotable
(spread Δ negative both windows, rec hit 4/9, rec era-flip); **T1 OOS Δ flips to −0.0003**
(first-pass +0.0005 was population-inflated — KEEP reinforced). C4 reworded per F4: NO SWAP is a
power outcome, not inferiority (−2 bps margin was a de facto superiority test, neutral-swap pass
prob ~0.1-0.2%). Also recorded: 6b population-matched control books never built for C2/T1 (F2 —
matched scoring is a proxy); T1 imputation pin not implemented (F3, direction favored variant);
endpoint-3 BTC-move quintile split never computed (F5). Final verdicts: C1/C2/C3/C5 KEEP
incumbent, C4 NO SWAP, T1 no addition. Approved closing: **no promotable variant among
the tested cells at the pre-registered bars; V0_LEAN stays frozen (scoped to the candidate set
— Pack 6 unscreened, endpoint-3 quintile split never computed); improvements, if any, are <~0.002 Δrank-IC or in
endpoints the bars reject; program CLOSED per cell budget** ("locally optimal" claim withdrawn —
the cells bound improvements, they don't confirm optimality). New estimator lesson for the law:
paired per-cycle deltas are only paired if both arms share the per-cycle population — a
0.7-symbol mean mismatch flipped one verdict and one CI sign.

## Post-loop addendum 7b (2026-07-08) — pre-registration COMPLETED: endpoint 3 + matched controls

The two gaps the results review left open are closed; **no verdict changed**; the package is now
complete in the strict pre-registered-endpoint sense.

- **Endpoint 3 (|BTC-4h-move| quintile split, F5)**: implemented in `score_variant_cell.py`
  (BTC move = |close(t)→close(t+4h)| of the scored cycle, quintiles per window) and computed for
  all 6 cells. 2 of 12 Q4 CIs exclude 0 (vs ~0.6 expected at 95%; P(≥2) ≈ 0.12 under
  independence — weak evidence at best): C3 rec Q4
  +0.0036 [+.0003, +.0069] (mechanism-consistent — residualized momentum helps most when the BTC
  move dominates; still non-promotable on spread/recent-hit bars) and C5 rec Q4 −0.0036
  [−.0062, −.0009] (reinforces KEEP). Diagnostic only, per pre-registration.
- **6b population-matched controls (F2)**: `VARIANT_CONTROL=1` harness mode builds incumbent-
  feature books on the variant row mask (`hl_retc1ctl_*`, `hl_takerlsctl_*`); scorer takes
  `SCORE_BASELINE_TAG` to use them as the baseline arm. Honest Δ (variant vs matched control):
  **C2 rec −0.0003 [−.0022, +.0017], OOS −0.0005 [−.0019, +.0009]; T1 rec −0.0002
  [−.0023, +.0019], OOS −0.0001 [−.0016, +.0014]** — all cross zero. Pure population effect
  (control vs incumbent) is small and CI-crossing in all four window-cells. Mechanism statements
  now instrument-backed: ret_6d adds nothing over ret_3d (not "worse"); taker_ls at 36h lag
  carries no incremental information over V0_LEAN. The T1 control also moots the F3 imputation
  deviation for the verdict (both arms share the row mask).

- **Endpoint 4 per-fold tables**: the scorer initially printed per-year/monthly only; per-fold
  Δrank-IC is now printed on every run and recorded for all 8 comparisons (FEATURE_TUNING_RESULTS
  "Endpoint 4" table) — fold structure consistent with the monthly tables, no single-fold
  concentration, no verdict impact.

Final standing verdicts (unchanged): C1/C2/C3/C5 KEEP incumbent, C4 NO SWAP, T1 no addition.
Remaining open scope: Pack 6 liquidity unscreened; C4 parity question untestable at this noise
scale (construction-grounds adjudication if ever revisited).

## Post-loop addendum 6d (2026-07-08) — feature-pack screens + ONE new-modality cell (T1)

User-proposed 8-pack expansion map screened against evidence:
- Pack 1 (mixed-beta features): premise DEAD — the mixed-beta label was rejected as artifact
  (rank-IC Δ≤0); do NOT wire into X70/panel. resid_ret_3d = C3, in flight. beta_spread:
  screening-tier only, no cell.
- Pack 2 (xs-rank versions): CLOSED, mathematically — xs-ranking is a within-cycle monotone
  transform; per-cycle rank-IC identical to raw to 4 decimals for all 5 features; the selection
  layer already consumes only within-cycle ranks.
- Pack 4 (funding retest under residual target): DONE, dead — IC +0.006/+0.002/−0.002 vs
  0.027-0.080 for V0_LEAN. Closed with fresh evidence.
- Pack 5 (regime-context features): closed by prior ledger (btc_rvol_7d already in V0_LEAN;
  sister-program regime feature additions rejected fold-concentrated; DDI R²=0.005; regime is the
  bot's job by validated design).
- Pack 7 (interactions): closed for this model class (per-symbol Ridge, small data; combinatorial
  garden). The legitimate underlying question is model-class (Ridge→GBM), not features.
- Pack 6 (liquidity): unscreened, low prior; may get a screening batch later.
- **Pack 3 (positioning as MODEL features): one REAL flag.** Per-symbol screens: oi_chg_{4h,1d,3d}
  IC ≤ |0.015| (no flag); top_ls 0.009 (no flag); **taker_ls_24h IC +0.037 (t +23.6), decaying
  gracefully to +0.029 (t +18.9) at 36h availability lag** — V0_LEAN-mid-tier grade, new modality,
  deployable via daily Vision ingest with a pinned lag convention.

**T1 pre-registration (BINDING):** cell = V0_LEAN + taker_ls_24h_lag36h (ADDITION — capacity
increase acknowledged; feature-addition prior is 0-for-N and stated). Construction pinned:
trailing-24h mean of sum_taker_long_short_vol_ratio (≥80% window coverage else NaN), sampled at
cycle_open − 36h (worst-case Vision availability), NaN ⇒ preproc imputation as any V0_LEAN
feature. Same books/cuts/machinery; book-level endpoints and promotion bars identical to C-cells
(addendum 6b); population accounting mandatory (metrics coverage 0.88). Runs AFTER C1-C5.
Promotion = forward-test candidate only; live wiring additionally requires a Vision metrics daily
ingest step in the loop (currently absent) — implementation gap stated up front.

## Post-loop addendum 6c (2026-07-08) — C5-C6 gate outcome + cells in flight

Screening extension (±2× perturbation, 12 syms, screening-only) on the 4 untriaged features:
- **autocorr: INERT in window AND percentile-history** (IC 0.018-0.026 across all perturbations;
  corr with incumbent ≥ 0.988) — the proposed 16-combo grid would have been pure noise mining.
- beta_chg_5d: no directional flag (double ≈ base, half weaker). idio_vol_1d: within noise.
- **corr_to_btc: DIRECTIONAL FLAG** — half-window (144 = 12h) IC 0.041 vs base 0.012 (3.4×),
  double-window ≤1× → **C5 GRANTED: corr_to_btc_1d → corr_to_btc_12h**. One grant total; C6 slot
  unused. Caveat printed per protocol: base IC is small (0.012), so 3.4× of small is still small —
  the cell exists to test it properly, not to promise anything.
Cells in flight: C1-C4 books generating (feature_variant_harness.py, final numbering), C5 queued
after. Scorer committed: `live/score_variant_cell.py` (book-level endpoints per estimator law).
## Post-loop addendum 6b (2026-07-08) — FINAL protocol (supersedes 6; reviewer's completed version
incorporating the beta closure + estimator law)

Addendum 6 was written from the reviewer's interim output; the completed review supersedes it.
Deltas from 6: the ret_3d family becomes THREE variants — C1 ret_36h (pct_change 432), C2 ret_6d
(1728), C3 resid_ret_3d (trailing 864-bar sum of incumbent per-bar β_288 alpha, shift 1) — with
two binding structure rules: **if C1 AND C2 both pass → BOTH rejected** (both-directions-better
was pre-identified as the noise signature; only a forward window can validate a dead-middle
story), and if C3 passes alongside one of C1/C2, **C3 takes the family slot** (mechanism over
window). C4 = dd_from_high_288 parity swap (unchanged from 6). C5-C6 contingent on extending the
±2× perturbation screen to the 4 untriaged features (corr_to_btc, beta_change_5d, idio_vol_1h/1d,
autocorr) — a cell is granted only on a DIRECTIONAL flag (≥2× |IC| one direction, ≤1× the other);
both-directions = no cell. C7 combined confirmation; if C7 fails, ALL promotions demote to
watchlist, no subset search. Endpoints/bars: book-level primary (paired rank-IC Δ with day-block
CI; top/bot-K spread; big-|BTC|-quintile split; fold/era tables), overlay-frozen replay secondary
with the uninformative-replay rule (|net Δ| < 20% of churn scale → declared uninformative);
promotion needs rank-IC Δ≥0 both windows + CI excluding 0 in ≥1 + spread Δ≥0 both + F8 rules +
hit-rate ≥5/9 and ≥18/33 + population control (trigger: >0.5% train-row diff in any fold OR any
symbol entry-fold shift). Per-cell false-pass ≈1.2%; ≤8 cells; expected false passes ≈0.06-0.09.
NOT-TEST list per review (beta everything, matching windows, all triage-inert windows, vol
ratios/feature additions, autocorr grids, regime-conditional anything, 2022). Standing law
restated: no feature/label variant is ever judged through the live gate/stop replay path.

## Post-loop addendum 6 (2026-07-08) — FEATURE-WINDOW PROTOCOL (BINDING pre-registration, merged
with the estimator law from the beta post-mortem)

Design review arbitrated the user's 7-family plan (~45-50 cells, "both windows improve" bar →
~11 expected false promotions) vs hardened restrictions. **Verdict: user plan REJECTED as
structured; protocol reduced to TWO cells** (beta family closed by the A/B verdict; families
2/4/6/7 rejected in full — triage-inert, sequential-conditioning, or grid-shaped; autocorr may
EARN a future cell only via a ±2× perturbation triage flag first).

**Cells (each: 4 book-gens recent/OOS × base/long vs frozen incumbent; one change per cell;
replacement not addition):**
- C1 ret_3d → ret_6d (pct_change 1728, shift(1) kept). Mechanism: horizon-ladder gap (8h/12h/1d/3d
  covered, 6d empty; 1.5d would crowd resid_rev + require argmax between near-tied screening ICs).
- C2 bars_since_high (+xs_rank) → dd_from_high_288 (+xs_rank), window unchanged. Engineering
  parity swap (bounded by construction; kills the truncation guard + the xs-rank parity hazard).
  **Non-inferiority ladder chosen NOW; alpha wording forfeited forever for this cell.**

**Endpoints (per the binding estimator law — replay-through-overlays is NOT a valid variant
estimator):** PRIMARY = book-level per-cycle rank-IC Δ (day-block t) + top/bot-K selection alpha
spread Δ + big-|BTC-move|-quintile split. SECONDARY = paired replay with overlays DISABLED
(REGIME_GATE=0, stop off, kill-switch off) for cost/turnover realism. Full-stack replay: reported,
never verdict-bearing.
**C1 promotion (ALL required):** primary Δ ≥ 0 both windows; OOS selection-spread day-block 95% CI
excludes 0; no era-flip (F8 25% rule); top episode ≤ 50%; gates-off replay OOS ≥ 20/33 folds Δ≥0
and recent ≥ 5/9. REJECT if any primary CI entirely negative. Else KEEP INCUMBENT.
**C2 swap:** primary Δ CI lower bound > −2 bps/cyc both windows; no era-flip; gates-off maxDD not
worsened >10%.
**Mandatory population control:** any variant with longer effective history (ret_6d: 6d vs 3d) —
if training rows differ >1% in any fold or any symbol's entry shifts >2d, run the
population-matched control (incumbent features on the variant row mask); honest Δ =
variant vs matched control.
**Multiplicity:** 2 verdict-bearing cells + ≤1 combined confirmation; expected false passes ≈ 0.05;
printed next to any pass. One run per cell, defect-only repeats. Any promotion = FORWARD-TEST
CANDIDATE; production never changes from these windows.
**Not tested (and why), verbatim from review:** beta blends/3d/7d (tuned-continuous), family-2
matching windows (aesthetics), return_1d window (triage-optimal), ATR/rvol/OBV/VWAP windows
(triage-inert), vol ratios (new features, out of scope), high-window sweep (C2 covers the
mechanism), autocorr grid (no triage flag yet), ret_1.5d (crowding+argmax), residualized momentum
(beta closed + 0-for-14 prior), any per-regime conditioning (standing law).

## Post-loop addendum 5 (2026-07-08) — beta-label A/B pre-registration + feature-window triage

**Beta-label A/B (RUNNING; generator committed `live/gen_beta_label_ab.py`).** Variant under test:
the textbook SHRUNK beta label, alpha* = my_fwd − (0.5·β_288 + 0.5·β_1440)·btc_fwd. Isolation pins
(fixed before results): label only — features incl. resid_rev stay on the incumbent alpha; fwd
returns stay row-based (no _fill_grid confound); same WF cuts/machinery as the incumbent books;
clean universe. Cells: KEEPSET4+deepmom, both windows, paired day-block CI vs the incumbent
k4_deepmom cells. Verdict rules: KEEP INCUMBENT if CIs cross 0 in both windows (the expected
outcome — labels correlate 0.971); ADOPT-candidate only if Δ ≥ 0 in BOTH windows with at least one
CI excluding 0 and no era-flip (F8 numeric rule); REJECT variant if either window's CI is negative.
β_5d-pure is the backup variant, run only if the shrunk result is ambiguous.

**Beta-label A/B RESULT (2026-07-08): KEEP INCUMBENT — pre-registered rule fired (CIs cross 0 both
windows) AND attribution shows the large deltas are OVERLAY-PATH ARTIFACTS, not label alpha.**
Headline (recent +6,644, Sh 2.26→2.86; OOS +5,792, Sh −0.19→+0.27) decomposes: preds correlate
0.995 with incumbent; per-cycle rank-IC Δ zero-to-negative (t −0.35/−0.84; no lift in big-|BTC-move|
bars where the churn theory predicted it); selection-only top/bot-K spread flat OOS, WORSE recent
(−4.1 bps/cyc, t −1.8). The PnL Δ was manufactured by path-coupled overlays on near-identical
preds: recent +4,830/6,644 from identical-pick cycles differing only in sigma-stop gross
(stop engagement 35.7%→17.1%); OOS +5,279/5,792 from 682 cycles where the binary gate was flat in
one arm only; differing-pick cycles net −1,337 OOS. Excluding gate-divergent cycles, the
"every-OOS-period-positive" pattern disappears (2023 −20, 2024 +986, 2025H1 −184, 2025H2 −269).
Post-run generator parameterization verified bit-exact by full regeneration (max|pred Δ|=0).
β_1440 min_periods delays 5 young symbols one fold — immaterial and wrong-signed for the Δ.
**NOT a forward-test candidate; beta family CLOSED.** pure5d arm = diagnostic only (pre-registration:
cannot upgrade a fired KEEP verdict). NB `gen_beta_label_ab.py` exists but is untracked — commit
with this entry when the user next commits.
**PROCESS LAW (binding for all future feature/label A/Bs): a paired replay through the
equity-sigma stop + binary regime gate is NOT a valid label-effect estimator (positive-feedback
path bifurcation; net Δ ≈ 5-10% of inter-arm churn). Primary endpoints must be BOOK-LEVEL
(per-cycle rank-IC, top/bot-K selection spread); replay secondary with overlays frozen/disabled.**

**Feature-window sensitivity triage (screening-only, 12 liquid syms, ~160d).** Windows are NOT
load-bearing for atr_pct/obv_z/rvol_7d (corr ≥ 0.86 under ±2× perturbation, IC stable);
return_1d's 288 is locally best (both perturbations weaken IC); vwap_slope/bars_since_high within
noise. ONE flag: **ret_3d (864 = 3d)** — both half (1.5d) and double (6d) windows show ~2× the
univariate IC (−0.072/−0.069 vs −0.038). Both-directions-better is noise-shaped OR the 3d middle
is genuinely dead; either way this is a SCREENING note only. Standing law applies: per-feature
window sweeps are the overfitting garden (regime-conditional feature tuning went 0-for-14);
ret_3d is a bot FILTER input more than a model feature (LONG_MAX_RET3D etc. are off in KEEPSET4;
it enters the model via V0_LEAN cohort) — any change requires the full harness A/B, and none is
scheduled. Recorded as watchlist-style diagnostic.

## Post-loop addendum 4 (2026-07-08) — feature deep-review: per-feature map + parity guards applied

Full construction review of all 14 V0_LEAN features (live path `incremental_xs_feats.py` lean
builder + `incremental_panel.py`/X70 `btc_cross` + X6b cohort; research path identical formulas).
Per-feature verdicts (formula → window → PIT mechanism):
- return_1d (pct_change 288), atr_pct (EWM-14 TR/close), vwap_slope_96 (vwap96/shift5−1): same-bar
  close under the enter-at-close convention — PIT ✓.
- obv_z_1d ((obv−roll288μ)/roll288σ, shift 1) ✓; autocorr_pctile_7d (roll-2016 pct-rank of roll-35
  lag-1 autocorr, shift 1) ✓.
- corr_to_btc_1d (roll-288 corr, shift 1), beta_to_btc_change_5d (shifted β diff 288×5),
  idio_vol_to_btc_1h/1d (roll std of β-residual, shift 1) ✓.
- rvol_7d (roll 7d std logret, shift 1), ret_3d (pct_change 864, shift 1), btc_rvol_7d (broadcast) ✓.
- bars_since_high (cumcount since close==roll-288 max): PIT ✓ but UNBOUNDED — the module docstring's
  "resets within 288 bars" was FALSE (corrected). Empirical max run 3,441 bars ≈ 11.9d vs the ~36d
  windowed-recompute bound (3× margin).
- bars_since_high_xs_rank: per-cycle cross-sectional rank — the one live-vs-backtest parity hazard
  (partial cross-sections on late klines shift every survivor's rank).
All pooled |IC| ≤ 0.057 (audit); no look-ahead anywhere; _fill_grid protects all row-offset
features against 5-min gaps in the xs path (label/beta path still row-based — consistent both
sides, queued for next retrain).

**Fixes applied (definitions UNCHANGED — validated artifacts stay valid):**
1. `incremental_xs_feats.py`: truncation guard — if bars_since_high nears the windowed-recompute
   bound, that symbol full-rebuilds instead of writing a truncated value; docstring corrected.
   Windowed-vs-full validation still bit-identical (max rel diff 9e-10, SOLUSDT).
2. `incremental_panel.py`: thin-cross-section DEFER guard — new cycles with <80% of the trailing
   median universe are deferred (rebuilt on a later pass via --rebuild-days) instead of ranked over
   a partial cross-section.
Queued for next retrain (definition changes): _fill_grid in load_closes/_closes_tail (label/beta
grid alignment); β-window label A/B (5d, shrunk); bars_since_high cap-at-source; HL naming.

## Post-loop addendum 3 (2026-07-08) — full pipeline audit: NO look-ahead; live-parity fixes applied

Adversarial audit of label/target/features/estimation/prediction. **Verdict: no look-ahead
anywhere** (CLAUDE.md's known bug classes verified absent, incl. cousins; all V0_LEAN features PIT
with |IC| ≤ 0.057). Issues found + actions:
- HIGH (fixed): v4/v3 live loops ran `incremental_panel.py` without `--rebuild-days` → late-kline
  holes never backfilled and xs-feats tail repairs never propagated (predictor recompute was a
  no-op for corrections). Both runners now pass `--rebuild-days 10` (v1 already did).
- MED (fixed): first predictor run (and every post-retrain run) could overwrite PIT history with
  in-sample preds (fit_cut = panel−1d). `predict_v4_incremental.py` now floors the recompute
  window at the artifact's fit_cut. (Same latent issue exists in `predict_v3_incremental.py` —
  fix when v3 live is next touched.)
- MED (open, label quality): the 1-day beta window in the v4 label is undocumented and
  contributes ~5.9% of label variance as hedge-ratio churn — **~11% (up to 16%) in top-decile
  |BTC-move| bars, i.e. concentrated in the beta-dominated regime = the known 2024 failure mode.**
  Pure estimation noise only ~1.7% (the churn is mostly REAL beta drift; a longer window lags
  instead). Labels under 1d vs 5d beta correlate 0.971. QUEUED as a pinned label A/B (β_5d and
  shrunk 0.5·β_1d+0.5·β_5d) through the matched-cut harness — the one new backtest-lever item the
  audit legitimately adds.
- MED (open, parity): `bars_since_high` unbounded (45d-recompute truncation divergence in long
  drawdowns) + its xs_rank over partial cross-sections. Fix before/with the next artifact retrain.
- LOW: label horizon row-shift without grid-fill (consistent both paths, embargo-safe; fix with
  _fill_grid at leisure); "HL=60" is e-folding not half-life (naming); unused target_z semantics.

## Iteration log (REVERSE-chronological below Iter 4: entries were inserted newest-first —
read order is Iter 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9)

### Iter 1 — INCIDENT + design-review verdict (15:2x UTC)

**INCIDENT (caught by the Q1 design reviewer, F1): the Q4 metrics top-up DESTROYED cache history.**
`metrics_loader.py` saved only the fetched range (overwrite, not merge); the 2026-05-10..07-06
top-up clobbered the 2021→2026-05 history of **163/176 symbol caches** before it was killed
(intact: W-Z tail only). Recovery (all data is public Binance Vision):
- loader FIXED to merge-with-existing (never overwrite; incident documented in code comment);
- BTCUSDT + full-universe rebuild 2021-01→2026-07-06 running (background);
- the pre-clobber positioning axis values SURVIVE in the merged dataset, preserved to
  `live/V4_GATE_DATASET_POSITIONING.parquet`; axis construction committed as
  `live/build_positioning_axes.py` (was inline-only — reviewer F1).

**Q1 design review verdict: PROCEED-WITH-FIXES (blocked until F1/F3/F4 in place).** Full findings
in the review transcript; accepted pre-registrations now binding for Q1:
- F1: reproduce the motivating diagnostic (side×oi_z q0: net −47.3±10, t sign, 0/4 pattern) on
  rebuilt data BEFORE writing bot code; hard stop if it fails.
- F2 (multiplicity — t −2.3 is a screening extremum over ~70 buckets): max possible outcome of Q1
  is FORWARD-TEST CELL or RISK-LEVER; ADOPT is impossible from these two windows.
- F3: 100-seed random-skip placebo matched on per-period gate rate; alpha claim needs mean > p95;
  risk-lever claim needs DD improvement > placebo DD median. Without this, RISK-LEVER-ONLY is
  unfalsifiable.
- F4 go/no-go pre-test (zero degrees of freedom): baseline KEEPSET4 realized PnL on the would-be-
  gated cycles must be < −10 bps/cyc in ≥3/5 periods, else the gate arithmetically cannot help
  and no bot run happens.
- F5: seed expanding-percentile history from 2021-01; report per-period gate rate; if gate rate
  drifts materially across eras, per-period deltas are declared uninterpretable.
- F6: mechanism demoted to label; report corr(oi_z, btc_ret_30d | side) and Jaccard overlap of
  gated cycles vs SIDE_FLAT_SKIP(0.05) set; overlap >50% = compounded multiplicity, note in verdict.
- F7 pins: contracts OI column (not USD), last-snapshot-≤-open with 2h staleness cap, ≥80% window
  coverage else NaN, NaN⇒gate INERT, percentile over all prior cycles, 180-cycle burn-in.
- F8 numeric verdict rules: era-flip = any opposite-sign period with |Δ| ≥ 25% of Σ|period Δ|;
  concentration fail = top episode > 50% of total; recent-window Δ ≥ 0 required for any positive
  verdict; DD claim requires both windows + beating placebo DD.
- F9: median/concentration/halves reported next to every mean; state which period lacked coverage.
- F10: report deltas split into skipped-entry vs path-divergence cycles; never stack with
  side_flat in the same cell.
- 2022 holdout is RESERVED for the final stack (not spent on Q1).

### Iter 2 — Q3 primary result (16:0x UTC): OOS ranking claim FAILS placebo; exposure claim stands

Q3 design review: PROCEED-WITH-FIXES; all blocking fixes implemented (episode-frozen
persistence-matched placebo, 1000 seeds, stateless book-level primary statistic, control arms,
episode jackknife, exact p-values). Episode pre-check defused the review's F3 for OOS: **67
episodes, ≥6 with material PnL** — the "one 2024 episode" concern was a calendar-attribution
artifact. Recent window IS episode-limited (6 episodes, 1 material) → descriptive-only as
pre-registered. Script committed: `live/q3_deepbull_placebo.py`.

**OOS (test):** signal gross +62,279 / net +54,405; placebo median +53,761 gross (p90 +68,417);
signal rank **p79, exact p=0.215 → the return_1d RANKING is NOT distinguishable from random
alt picks OOS.** Controls: BTC-long +26,853, beta-proxy-ranked +37,245 — both below random-alts.
Turnover matched (0.39 vs 0.40). Jackknife range [+36,287, +65,715] (never negative).
**Recent (descriptive):** signal +7,518, placebo median +2,319, rank p98 (exact p=0.023) —
ranking added value in the single 2026 episode.

**Earned wording (per pre-registered ladder):** the overlay's validated OOS value is
"LONG-ALT EXPOSURE in deep bull" (random alts beat BTC-long by 2×; both positive). The
return_1d ranking is significant only in the 2026 episode so far. Forward-test cell RETAINED.

**Q3 CLOSED (results review: REVISE → fixes applied → landed).** Results-review audit
re-executed the primary statistic (all numbers reproduce; placebo p79/p=0.215 is seed-stable
at p76/0.255 on a 50-seed trim) and found: (F1) the committed script's BTC-long control
silently printed +0 (panel is ex-BTC) — FIXED, control now fetched from Vision klines with a
hard-fail on empty (committed script's own control: +26,853; results-review independent recomputation +25.6k —
the ≈1.2k gap is method noise, both confirm ≈2:1);
(F3) two small biases both favor the signal arm → the FAIL verdict is conservative-safe;
(F4) stateless population = ALL 1,378 OOS deep cycles; the bot entered 749 of them (the 979
figure is the PnL-divergence footprint incl. 230 non-deep carry cycles) — scoped in the doc; (F5) placebo verdict is K=2-scoped; K1_g50 OOS +5,318 is a full-bot path
statistic, logged NOT chased; (F6) yearly table re-attributed to exposure, "mom30 died"
re-scoped to "mom30 harmful", both docs edited; (F7) OOS top-episode share 41.7%, OOS gross
non-linearity (1,343/1,720/3,657 per unit) recorded as fragility evidence; (F8) recent
jackknife flips negative [−1.9k, +8.2k] — quoted. Band: recent linear (+757/unit ×3), center
kept per pre-registration. Forward counterfactual pre-registered (episode-frozen matched
random-pick book, K=2, seeds 1-1000, cumulative signal-minus-placebo-median). Canonical
updates: V4_PERFORMANCE §6.1, CONVEXITY_V4_FLOW §8.

### Iter 3 — Q1 HARD-STOPPED at reproduction gate (16:3x UTC); Q4 BTC re-screen done

BTC metrics fully rebuilt (2021-01→2026-07-07; 579k rows). Integrity: rebuilt axis values
bit-identical to the preserved pre-clobber dataset (max diff 0.000000). **F1 reproduction of the
Q1 motivating diagnostic FAILS the pre-registered pattern**: net −45.3 (within ±10 ✓) but t −0.8
(was −2.3) and 1/5 periods positive (was 0/4) → FRAGILE, not AVOID. Cause: the original battery
ran on coverage truncated at 2026-05-12; the restored 2026 period flips the pattern. This is the
design review's F2 (screening extremum) materializing on schedule. **Q1 CLOSED: REJECTED at
reproduction; zero bot compute spent.** Framework doc corrected (side AVOIDs retracted inline).
Full-coverage battery re-run (24 axis×scope cells): bear all-FARM (unchanged), bull AVOIDs
persist (consistent with bull0), side = weak mid-bucket FARMs only, nothing actionable.
Universe metrics rebuild continues (background, needed for Q5 only).

### Iter 4 — Q2 PRE-REGISTRATION (16:5x UTC, BINDING, written before pred generation)

Design review: PROCEED-WITH-FIXES; all blocking items resolved as follows. Artifacts frozen:
`live/gen_residual_target_2022.py` (F11: 2022 CUTS, logged drops, fold symbol counts, F9
stale-print rule zero-return-frac>0.15 PIT), `live/run_2022_holdout.sh` (F12: full env dumps of
all 3 cells + preds SHA256 + one-shot write-once guard). Verified facts: 2023-OOS convention =
cold start at window open (pinned same); 2022 funding coverage 50-53%/quarter, p25 = −0.66
bps/8h; panel contains NO LUNA/FTT/UST/SRM/ANC/RAY → survivorship CONFIRMED; 43 symbols eligible
at 2022-01 (pre-reg screen; realized bot universe 39 at Jan open → 48 by Dec).

**Verdict rules (fixed now; no changes after unblinding):**
- Cells: A=vanilla reference, B=KEEPSET4, C=B+deep-bull overlay. The 2022 verdict computes on B.
- Rule i (loss containment), branch-complete (F2): if A_pnl<0 → PASS-i needs B_pnl ≥ 0.5×A_pnl
  AND |maxDD_B| ≤ |maxDD_A|. If A_pnl≥0 → PASS-i needs B_pnl > −5,000 bps AND |maxDD_B| ≤
  21,400 bps (=1.25× the validated 2023-25 B-frame maxDD 17,098).
- Rule ii (bear farm), three-way (F3): bear = bot's own regime labels. FAIL if bear-PnL day-block
  95% CI entirely <0; PASS if entirely >0; else INCONCLUSIVE (reported as such, never as pass).
- Funding stress (F5): any PASS must survive penalty_bps = 0.16 × Σ_cycles(gross_after_stop)
  (imputes the 48% missing funding rows at 2022-p25 magnitude, all-short worst case).
- Cost sensitivity (F6): re-score with depth component doubled (post-hoc: +Σ cost×10/14.5);
  a PASS that flips is labeled "cost-fragile" — wording only, verdict decided at 1×.
- C (F8): DESCRIPTIVE ONLY regardless of magnitude; deep-bull episode count reported first;
  C's sole guardrail: |maxDD_C| ≤ 1.1×|maxDD_B|.
- Survivorship wording law (F1): a PASS is "conditional on survivor universe; DD/tail are lower
  bounds on badness" and cannot upgrade any live-risk claim; a FAIL carries full weight.
- Cold-start ladder (F4): per-quarter attribution + per-fold training-history months reported;
  failure concentrated in Q1 with clean Q4 = "cold-start-favored", not era-fragility.
- FAIL consequence (F10, operational): V4_PERFORMANCE §7 gains a blocking item — live gross
  0.5× until the forward ledger independently confirms the bear farm over ≥2 months — and §1 is
  annotated era-fragile. One repeat allowed ONLY for code-defect bugs. No further 2022 cells ever.
- Universe size + per-quarter symbol counts reported next to every 2022 number; kill-switch
  halts (if any) reported with dates.

### Iter 9 — Q6 FINAL VERDICT (corrected placebo, 23:2x UTC): DD lever, mechanism unresolved

Corrected dose-matched placebo (100 exact-dose, run-length-matched, full-bot replays; generator
committed): the voided "beats 100/100" collapses to **recent 48/50 (p=0.059, beats p90 marginally)
/ OOS 41/50 (p=0.196, INCONCLUSIVE)**. Equal-dose random side de-gross replicates ~79-83% of
sf05's DD improvement → the lever is mostly EXPOSURE-DOSE. Joint pre-registered verdict:
**DD lever, mechanism unresolved** — not "flat-middle-specific". Landed in V4_PERFORMANCE §6.3
with all binding caveats (threshold-selection firewall, single-episode load-bearing recent,
no-alpha-claim, adoption deferred behind Q2-F10, 2022 uncharacterized). **Q6 CLOSED. QUEUE
EXHAUSTED (6/6 closed).**

### Iter 8 — Q6 placebo VOIDED by results review; defect-repeat running (21:4x UTC)

The stage-1/stage-2 placebo (expanded from the announced 20 to 50 seeds/window = 100 replays;
sf05 beat 100/100) was **VOIDED**: results review found the
skip-set generator dose-mismatched — it matched full-book de-gross cycles (355 ins / 919 OOS)
instead of sf05's TRUE suppressed-entry dose (581 / 1,931 per sleeves.csv), so every placebo was
under-dosed 40-50% and run-length-mismatched. Cells themselves reproduce exactly and stand
(bit-check, sf05/sf03 numbers, overlay entry identity 1.000). Per-skip normalization on the void
placebos: OOS DD improvement ≈ dose-explained (3.50 vs 3.21 bps/skip), ins ~45% excess untested.
Recent DD win is single-episode load-bearing (one Jan-Feb 2026 episode = 61% of baseline DD).
Additional binding wording (review F5): thr 0.05 inherits §6.3's in-window sweep — a placebo can
control skip-PLACEMENT, never threshold SELECTION; only the forward window tests the threshold.
PnL contrast (sf05 raises PnL where random skips lower it) = §6.3's rejected mean claim restated,
descriptive-only, cannot upgrade anything. Corrected pooled-50 ins placebo PnL median +16,844.
**Defect-repeat authorized (placebo generation only, Q2 precedent): generator now COMMITTED
(`live/q6_gen_skipsets.py`), 100 exact-dose run-length-matched skip-sets verified (581/1,931 ×50
seeds each), 100 replays running.** Earned verdict if the corrected placebo is NOT beaten:
"DD lever, mechanism unresolved" (nonspecific outcome). Adoption stays DEFERRED behind Q2-F10.

### Iter 7 — Q5 CLOSED (battery + watchlist); Q6 escalated to full-bot placebo (19:5x UTC)

**Q5 CLOSED.** Battery per pre-registration: 48 runs / 144 cells; non-FRAGILE count ≈9 beyond
|t|≈2.8 vs null expectation ~7 (count barely above null); redundancy table CLEAN (no |ρ|>0.4 vs
the 8 failed leg axes — genuinely new information). 3 watchlist admissions with caveats (framework
doc updated): bear×long_oi_chg_3d_rank q2 (t 6.4/5.2, the standout; within-bear tilt only),
bear×short_oi_chg_24h_rank q1, side×short_taker_ls_24h_rank q1 (opposite-direction monotonicity —
noise-shaped). Rejected at the gate: both bull cells (sign-flip / non-monotone). No bot ladder.

**Q6 in progress.** Baseline bit-check PASS both windows (today's code changes confirmed no-op at
defaults). Cells atop the current stack (KEEPSET4+deepmom, clean v4 preds): sf05 recent
+22,624 (Δ+2,996), maxDD −11,787→−3,869; OOS +105 (Δ+2,443 — sign flips the OOS total), maxDD
−14,555→−7,800. sf03 (descriptive): +1,111 / +631, maxDD −5,445 / −9,331. Interaction checks:
deep-overlay ENTRY sets bit-identical (1.000 both cells/windows); the nonzero bull deltas
(+758 recent) are stop-path gross scaling on identical entries — the F1 coupling made visible,
F5 tolerance technically exceeded and reported as such. Per pre-registration NO alpha claim is
made from these deltas. **Stateless placebo pre-screen: INCONCLUSIVE zone in both windows**
(signal-proxy maxDD beats placebo median but not p90: recent −5,262 vs median −9,189 / p90
−5,030; OOS −10,219 vs −11,869 / −9,773) → two-stage rule mandates escalation: 20-seed full-bot
persistence-matched placebo per window (skip-sets matched on per-period count + run-length
distribution; deterministic `SIDE_SKIP_SET_FILE` bot hook added, no in-bot RNG), running.

### Iter 6 — Q5 PRE-REGISTRATION (19:2x UTC, BINDING, before the battery runs)

Design review: PROCEED-WITH-FIXES. Reviewer's empirical probe found the committed BTC axis
builder's row-based windows would GUARANTEE a retraction-class artifact on this cache (in-file
gaps up to 1,598 days; universal ~10.5h outage gaps; per-symbol OI zeros and |dlog|≤1.8 spikes).
All blocking fixes implemented in the committed builder `live/build_leg_positioning_axes.py`:
F1 time-based windows + 2h staleness caps + calendar coverage ≥80%; F2 zero→NaN + 0.5-dlog 24h
quarantine (counts reported); F3 any-NaN leg rule + ≥70% pool validity; F5 battery cut to 6
verdict-bearing axes = within-pool percentile RANKS only ({long,short} × {oi_chg_24h, oi_chg_3d,
taker_ls_24h}), raw values retained non-verdict-bearing; F9 picks mirror v4_gate_model_test
(noted: pool conditions on label existence — pre-existing, screening-tolerable); F10 identity
assert on stored leg columns, hard stop on drift; 30d listing-age guard.
**Binding outcome rules:** SCREENING ONLY — no bot test this session under any statistic.
Watchlist admits ≤3 buckets max, ranked by |period-t|, ONLY if tercile-monotone AND same-sign in
both edge frames (F7); q_nan buckets never watchlist-eligible (F4); redundancy table vs the 8
already-failed leg axes required, |ρ|>0.6 candidates annotated "redundant re-test" (F8); WF
routing on per-regime scopes verdict-inert both directions (F11); same-session WF results are
watchlist-only regardless of significance — ladder eligibility needs a LATER-session majority-of-
folds result or one pre-registered forward period (net > MARGIN, same sign, open_time after this
timestamp) (F6); battery = 6 axes × 4 scopes × mechanism-paired edge frames ≈ 48 runs ≈ ~150
cells → null expectation ~4-8 cells beyond |t|=2.8 MUST be printed next to any reported finding
(F7); one run, defect-only repeat (Q2 precedent); costs 15/7.5, sensitivity {10,15,20} only.

### Iter 5 — Q2 verdict LANDED (results review PASS; consequences applied) + Q4 CLOSED (18:4x UTC)

**2022 holdout: FAIL, confirmed by adversarial results review (computation CLEAN, wiring CLEAN,
regime labels verified, no code-defect → no repeat permitted).** Key refinement adopted from the
review: the bear edge did NOT invert in 2022 — bear net −8,949 = gross −3,022 (CI crosses 0) −
costs 6,075 + funding +149; "absent gross and cost-dominated", not sign-flipped. Consequences landed:
V4_PERFORMANCE header + §1 annotation (reviewer-pinned wording incl. equity −76%, universe 39→48,
survivor-universe law) + §7 BLOCKING item (gross 0.5× with the forward bear-farm confirmation
criterion pinned: ≥2 months containing ≥1 bear episode, bear-net day-block CI excluding 0).
Enforcement wired: `GLOBAL_GROSS_MULT` env added to the bot (all three paths, default 1.0) and
set to 0.5 in `run_convexity_v4_live.sh`. Audit gaps for future runs (reviewer): env-dump grep
missed SIZING_MODE/SHORT_MIN_RET3D/LONG_MAX_RET3D; C footprint 135 path-divergent cycles in 8
episodes (the "139/19" figure was the raw deep-cycle/episode count); kill-switch halts: none.

**Q4 CLOSED.** Metrics cache restoration complete: 174/176 full-history; CHIPUSDT = late listing
(no earlier data exists); ICPUSDT = upstream Vision archive hole 2022-01..2026-04 (fragment kept,
logged). Merge path hardened again: a failed merge now ABORTS the write instead of dropping
history. Positioning re-screen was already done on full coverage (Iter 3).

### Iter 0 — loop start (14:5x UTC)
- Q4 top-up launched: metrics 2026-05-10..2026-07-06, BTC + 175-symbol universe (background).
- Q1 design pre-registered and sent to adversarial design-review subagent. Key pre-registrations:
  PIT trailing-expanding percentile threshold 0.333 (single value, no sweep), 180-cycle warm-up,
  skip-new-side-sleeves action (mirrors SIDE_FLAT_SKIP), verdict rules fixed BEFORE the run
  (REJECT on negative OOS delta or era-flip; RISK-LEVER-ONLY if only maxDD improves; ADOPT-candidate
  needs OOS paired CI excluding 0). Acknowledged weaknesses: 4th side lever on same windows
  (multiplicity), threshold-level mismatch vs diagnostic terciles, 3-for-3 diagnostic→bot failures.

## Addendum 8 (2026-07-08, PRE-REGISTERED before any computation) — sleeve-conditional window×horizon program

**Motivation (user):** the closed feature-window program scored every window against the single
production label (4h entry / 24h tip). A window useless there may serve a different holding
sleeve. Term-structure prior supports the axis (signal persists past 24h); STRAT_HOLD prior
constrains it (naive longer holds of the 4h signal did not generalize).

**Phase A — IC surface (SCREEN ONLY, non-verdict-bearing, no multiplicity spend):**
- Labels: per-symbol forward residual-alpha sums over next {1, 3, 6, 12, 18} cycles
  (4h, 12h, 24h, 48h, 72h), from panel `alpha_vs_btc_realized` (X70 conventions).
- Features (all shift(1) at 5m, sampled at 4h cadence, X6b conventions):
  raw ret {12h, 24h, 36h, 3d, 5d, 7d}; resid_ret {12h, 24h, 36h, 3d, 5d} (incumbent β_288
  idio sums, C3 construction); resid_rev lagged sums k∈{2,3,6,12} cycles (from panel alpha);
  corr_to_btc {12h, 1d, 3d}; dd_from_high {1d, 3d, 7d}. ~21 features × 5 horizons × 2 eras.
- Endpoints per cell: per-cycle XS rank-IC (mean, day-block t) AND pred-orthogonal rank-IC
  (feature ranks residualized per-cycle on incumbent base-book pred ranks — absorption screen).
- Eras: recent (2025-10-04→panel end) and OOS (2023-01→2025-09), scored separately.
- **Ridge rule (flag, not verdict):** a (feature, horizon) cell flags only if pred-orthogonal
  |t| ≥ 3 in BOTH eras, same sign, AND ≥1 adjacent window (same family, neighboring L) has same
  sign with |t| ≥ 2 in the pooled frame. Isolated peaks are ignored by construction.
- Multiplicity note printed with any flag: ~210 cells scored; ~10 false |t|≥3 single-era cells
  expected; the both-era + adjacency requirement is the control. No promotion from Phase A.

**Phase B (only if Phase A flags): ≤3 pre-registered book-level cells**, chosen by ridge center
not peak, one change per cell, full 6b machinery (matched-population scorer, all 4 endpoints,
matched control books whenever row availability changes, promotion bars as 6b per sleeve).
A cell targeting a NEW sleeve (label horizon ≠ 4h) additionally requires strategy-layer cost
validation before any adoption (STRAT_HOLD prior; cost per hold, sleeve overlap) — book-level
pass alone is a forward-test candidacy at most. 2022 holdout is SPENT; dual-era + fold bars are
the only era defense; stated limitation, not waivable.

**Not tested here:** beta-window label variants (family closed, addendum 5); funding windows
(dead under residual target); autocorr grids (inert); anything requiring new data classes.

### Addendum 8b (2026-07-08, BEFORE any computation) — design-review amendments (verdict: REVISE, applied)

Adversarial design review returned 13 findings; blocking fixes + pins applied to the
pre-registration and screen implementation BEFORE the first run:

1. **Horizon-length day blocks (HIGH)**: block = ceil(horizon/24h) days (1/1/1/2/3 per h4..h72)
   for every t and CI. Daily blocks under-count overlap: VIF ≈ 2 at 48h (t overstated ×1.41),
   ≈ 3 at 72h (×1.73). The addendum-8 "~10 false |t|≥3" note was itself evidence of the
   miscalibration; calibrated both-era same-sign flag rate ≈ 4e-4 over 105 alpha cells.
2. **Marginal labels (HIGH)**: h>24h flags additionally require same-sign |t|≥2 BOTH eras on the
   24h→h excess label (cycles 7..k: rolling(k−6).sum().shift(−(k−1))). Nested labels otherwise
   flag the already-harvested first 24h mechanically.
3. **Phase B sleeve baseline (HIGH)**: for any cell with label horizon ≠ 4h, the baseline arm is
   pre-registered as V0_LEAN (unchanged features) retrained on the SAME h label with the same
   cuts; Δ = variant vs that baseline; matched-population control on top per 6b. No comparison
   against 4h-label incumbent books.
4. **V0-span orthogonalization (MED-HIGH)**: the flag-bearing column is feature ranks
   residualized per-cycle on the FULL 16-rank span (V0_LEAN ∪ resid_rev_2/3 — includes the
   long-book extras, covering the resid_rev pin). Pred-orthogonal (base-book pred) kept as
   diagnostic only; neither necessary nor sufficient for a retrained sleeve — stated.
5. **Family-common population masks (MED)**: per cycle, all windows within a family scored on the
   intersection mask (every family window ∩ pred ∩ V0-span present); per-cell N reported.
   min_periods pinned: ret = full window (pct_change), resid_ret = w/2 (C3), corr = max(36, w/4)
   (C5), dd = full window (C4 basis).
6. **Universe (MED)**: Phase A rows = clean-book universe (pred present in `hl_*_clean` books —
   enforces EXCL + liveness); stale-print symbols never enter any cycle's XS ranks.
7. **Label grid guard (MED)**: label NaN where open_time[t+k−1] − open_time[t] ≠ (k−1)·4h
   (row-based rolling otherwise stretches horizons over gaps, compounding with k).
8. **Shared-β caveat (LOW-MED)**: resid_* features and the alpha label share β_288 hedge noise
   (~5.9% of label variance, ~16% top-decile |BTC-move|); any flagged resid_* cell must also
   show the raw (unhedged) forward-return-label IC as a robustness column (computed for all
   cells, diagnostic).
9. **Implementability (LOW)**: the multi-cycle alpha sum is a 4h-rebalanced hedge ladder, not a
   static-hedge residual; Phase B targets and STRAT_HOLD cost validation refer to the ladder.
10. **Era tails (LOW)**: per-horizon label masks differ at era tails (up to 3d at 72h); pinned,
    no post-hoc adjustment.
11. **Adjacency demoted to cosmetic**: under window correlation (t-stat ρ ≈ 0.7-0.9 between
    neighbors) adjacency support is nearly free and filters little; it is reported but carries
    no weight. Flag rule = V0-span-orth |t|≥3 both eras same sign (+ marginal-label test for
    h>24h). Ridge-shaped false positives remain the norm under correlation — Phase A flags are
    screen output, never evidence.
12. **New-sleeve ceiling tightened**: a book-level pass on a new sleeve = candidacy for a
    SEPARATELY DESIGNED forward experiment (the existing forward ledger runs the 4h stack under
    the 0.5× gross consequence and cannot adjudicate a new sleeve).

### Addendum 8c (2026-07-08, PRE-REGISTERED before Phase B runs) — Phase A results + Phase B cells

**Phase A outcome** (live/IC_SURFACE_WINDOW_HORIZON.csv, 504 rows; sanity PASSED — deployed
ret_3d/resid_rev_2/3 read absorbed |t_v0|≤0.9 vs raw |t|8-14): 15 flags in 3 coherent ridges.
(1) SHORT-MOMENTUM HOLE: 12h-24h return windows × h4-h24, positive sign (continuation), peak
ret_24h × h4 t_v0 +9.5/+7.0, ic +0.021/+0.010 — V0_LEAN spans 8h reversal (resid_rev_2/3) and
3d momentum (ret_3d) but nothing between; t collapses to ~0 at 36h+. Survives raw-label
robustness. (2) dd_3d × h4-h12 (+4.1/+3.0). (3) resid_ret_3d × h72, the only h>24h flag passing
the marginal 24h→72h test (m72 +4.1/+2.2).

**Phase B cells (≤3 budget, ridge centers, one change each):**
- **B1: + ret_24h** (addition) on the production 4h label. Full 6b machinery/bars (as T1):
  matched-population scorer, all 4 endpoints; matched control book if train-row diff >0.5% any
  fold or symbol entry shifts. Promotion bars identical to 6b additions.
- **B2: + dd_3d** (addition) on the production 4h label. Same machinery/bars. B1 and B2 are
  correlated candidates from different families; if both pass, only the stronger is promoted
  (≤1 promotion/family rule extends: ≤1 promotion for the short-momentum ridge overall).
- **B3: + resid_ret_3d on a 72h sleeve** — baseline arm per 8b-3: V0_LEAN retrained on the h72
  label (xs_z of 18-cycle alpha sum, grid-guarded) on the VARIANT row mask (control built in);
  variant arm: V0_LEAN + resid_ret_3d, same label/cuts. Label purging: exit time for embargo =
  open_time + 72h (NOT the panel 4h exit_time) — mandatory, else training leaks the label
  window. Endpoints: same 4 endpoints with fwd = 72h alpha sum and 3-day blocks. Ceiling per
  8b-12: a pass = candidacy for a separately designed forward experiment only.

Multiplicity: 3 cells, expected false passes ≈ 0.04 (6b bars). No further cells from this
surface regardless of outcome; unflagged families (corr windows, resid_rev extensions) CLOSED.

### Addendum 8d (2026-07-08) — Phase B results: 0/3, program CLOSED (post-review)

All three cells scored on the F1-fixed scorer; adversarial results review reproduced every
number, stress-tested the new 72h paths (3d-block CIs robust at 6d/9d blocks; purge/control in
gen_sleeve72.py verified sound, test-row ratio 1.0000 both arms), and REVISED two explanations:

- **B1 +ret_24h: NO ADDITION** (rec Δic −0.0002, OOS −0.0003, spreads −3.9/−5.6, hits 3/9 &
  13/33). **Mechanism corrected by review (F1): NOT absorption.** ret_24h is a near-duplicate of
  the DEPLOYED return_1d (per-symbol corr median 0.995; per-cycle XS rank corr 0.986). The
  Phase A flag (t_v0 +9.5/+7.0) lives entirely in the ~2-3%-variance residual of that pair — the
  shift(1)-vs-same-bar freshness offset, i.e. a last-5m-bar reversal component, plausibly part
  bid-ask bounce. The incumbent pred carries NONE of it (pred ⊥ flag-residual corr −0.0018), and
  a ridge-regularized addition cell splits weight across a 0.99-collinear pair and shrinks
  exactly the low-variance difference direction — the flag was screen-real but MODEL-INACCESSIBLE
  to the cell as designed. Do not re-mine: the sharp cell would be the difference feature, whose
  content is 5m-close microstructure. Screen postmortem: the V0-span orthogonalization DID
  control return_1d; what survived it is this residual.
- **B2 +dd_3d: NO ADDITION** (Δic +0.0002/+0.0003 at the noise floor, spread Δ negative both
  windows, hits 6/9 & 16/33).
- **B3 resid_ret_3d @ 72h sleeve: REJECT.** Era-independent kill first: selection-spread Δ
  non-positive in BOTH windows — rec −65.3 bps/cyc CI [−129.1, −6.7] ENTIRELY NEGATIVE (robust
  at 6d/9d blocks), OOS −2.9 [−33.1, +28.8] — the rank-IC lift never converts to top/bot-K
  alpha even in the favorable era. Then: recent fails Δ≥0 (Δic −0.0020, hit 3/9). Descriptive:
  OOS Δic +0.0027 CI [+.0002, +.0052] excludes 0 (lower bound razor-thin at every block length),
  22/33, big-move Q4 concentrated — but review's monthly decomposition shows the 2025 OOS
  strength is a Jul-Sep 2025 hot streak (+0.013/+0.011/+0.010) with negative months inside OOS
  too; the "building trend" read is wrong. Era difference ≈1.9σ, no instrument discontinuity.
  No sleeve candidacy (8b-12 ceiling would have applied even to a pass).

**Deviations logged (review F5):** B1/B2 matched-control trigger technically fired (test-row
ratios 0.9993/0.9961, symbol entry-fold shifts) and control books were not built — verdict-safe:
the scorer intersects populations per cycle, the C2/T1 controls bounded the pure population
effect at |Δ|≤0.0005, and the handicap direction penalizes the variant, which cannot rescue
cells failing at 3/9-16/33 hit rates. The k=6 scorer default path has no grid guard (matches all
prior cells; tiny and arm-symmetric).

**PROGRAM CLOSED (approved wording):** Phase A ridges are real at screen level — with the caveat
that the short-momentum ridge's flag component is the low-variance residual of a near-duplicate
of an in-model feature — but 0/3 book cells pass. B3's 72h lift is OOS-only AND never reaches
the selection layer in either era. No promotable window×horizon variant; no sleeve candidacy;
V0_LEAN and the 4h stack stay frozen. Unflagged families (corr windows, resid_rev extensions)
and the ≤3-cell budget are spent; no further cells from this surface.

## Addendum 9 (2026-07-09, PRE-REGISTERED) — W1: training-label winsorization A/B

**Motivation (target-alignment discussion, 2026-07-09):** the strategy consumes ranks; the tip
tails are measured-unpredictable lotteries/squeezes; MSE training nonetheless spends weight
precision chasing extreme label magnitudes. Hypothesis: truncating the TRAINING label's tails
removes unfittable variance and improves the ordering the weights produce — "fit the predictable
middle, decide on the conditional tail." Prior caution on record: vBTC's clip-at-±5 was harmful,
but that was an uncontrolled preprocessing change across panels, not a paired label A/B.

**Cell W1 (single verdict-bearing cell):**
- Incumbent training target: xs_z = per-cycle z(alpha_vs_btc_realized), clip ±10 (clips ~nothing).
- Variant: identical xs_z, **clip ±2** (z first, then clip — incumbent order). ONLY the training
  target changes: features, two-book structure, WF cuts, RidgeCV, HL=60, embargo, min-rows,
  EXCL — all frozen. Row availability unchanged by construction (clip preserves notna) → no
  population mask; matched-population scorer applies regardless.
- Dose choice: ±2 truncates a meaningful tail mass (expected ~2-6% of rows; exact fraction
  REPORTED as the dose check). ±3 would clip ~1% and risk a bit-identical NO-OP (validation
  ladder rule: bit-identical cells are NO-OPs, not null results). NO-OP guard: if per-cycle pred
  rank-corr with incumbent > 0.999 in both windows, declare NO-OP.
- Endpoints & bars: 6b verbatim vs the incumbent V0_LEAN books (fwd24 outcome; the EVAL label is
  never winsorized — only training). Promotion requires: Δrank-IC ≥ 0 both windows; OOS rank-IC
  CI excludes 0; K-spread Δ ≥ 0 both windows; no era-flip (F8); recent hit ≥ 5/9, OOS ≥ 17/33.
- Backup W1b (clip ±3): run ONLY if W1 is ambiguous (defined: exactly one window's rank-IC CI
  excludes 0 and the two windows' Δ signs conflict). No other doses ever — a clip-level sweep is
  argmax mining and is refused in advance.
- Multiplicity: 1 verdict-bearing cell; expected false pass ≈ 0.01.
Books: hl_winz2_{base,long}[_oos]. Generator: live/gen_winsor_label.py (gen_beta_label_ab
pattern — target-only change through the frozen machinery).

### Addendum 9b (2026-07-09, BEFORE any book run) — W1 design-review amendments (verdict: REVISE, applied)

Reviewer measured the label anatomy directly: |z|>2 = 4.95% of rows (positive tail 3.06% /
negative 1.89% — the dose is ~1.6× larger on the long-book side), |z|>3 = 1.97% (pre-reg's "~1%"
corrected), |z|>10 = 0.01%, clip-±2 removes 38.6% of label variance (corr(z, clip) 0.933).
Amendments, all applied pre-run:

1. **Per-side leg decomposition added to the scorer** (Δ long-leg fwd, Δ short-leg fwd, block
   CIs) — DIAGNOSTIC ONLY, never verdict-bearing; pinned now because symmetric clipping is a
   book-asymmetric treatment and a post-hoc side story is otherwise guaranteed.
2. **W1b trigger fixed**: backup (clip ±3) runs ONLY if exactly one window's rank-IC CI excludes
   0 **on the positive side** and the two windows' mean Δrank-IC signs conflict. A negative-side
   excluding CI is a REJECT and takes precedence — no backup after a REJECT (dose-mining).
   W1b's role is confirm-toward-KEEP only (a weaker dose cannot rescue an ambiguous strong dose).
   "Signs" = signs of the two windows' mean Δrank-IC.
3. **Bars pinned exactly (replacing "6b verbatim")**: promotion requires Δrank-IC ≥ 0 both
   windows; rank-IC CI excludes 0 (positive) in ≥1 window; K-spread Δ ≥ 0 both windows; no
   era-flip (F8 25% rule); recent hit ≥ 5/9; OOS hit ≥ 18/33. REJECT if any primary CI entirely
   negative. Else KEEP INCUMBENT.
4. **Identity arm (panel-drift + machinery-parity guard)**: WINSOR_Z=10 run through the same
   generator; require ~0 pred divergence vs the incumbent books on common rows BEFORE the W1
   verdict is read. Catches silent panel drift (known repo hazard) and proves generator parity
   in one shot.
5. **NO-OP guard implementation pinned**: mean per-cycle Spearman(variant pred, incumbent pred)
   on the common population, printed by the scorer per window; NO-OP if > 0.999 both windows.
   Noted: with 38.6% variance removed this guard can only catch a bit-identical bug; the real
   power floor is the paired CI half-width (~0.002 Δrank-IC) — effects below it are undetectable.
6. **RidgeCV alpha endogeneity declared part of the treatment**: the alpha grid is frozen; the
   per-symbol selected alpha is endogenous to the target by design (addendum-5 precedent —
   "what the model learned" includes regularization choice).
7. Reviewer verified clean (no action): preproc touches features only (target never passes
   through it); resid_rev built from unclipped alpha; row masks/fold populations bit-identical
   by construction (matched-control trigger cannot fire); sample weights target-independent;
   eval scale-invariant (ranks + argmax) so prediction shrinkage cannot manufacture a win.
8. Weighted per-fold dose (HL=60 recency weights) printed by the generator.

### Addendum 9c (2026-07-09) — W1 RESULT: KEEP INCUMBENT (post-review FINAL)

**Scoreboard (all numbers reviewer-reproduced bit-for-bit; identity arm bit-exact):**
REC Δrank-IC +0.0243 [+.0199,+.0288] EXCLUDES 0, hit 9/9, folds 9/9; K-spread Δ −19.10
[−82.01,+41.03]; pred rank-corr 0.889. OOS Δrank-IC +0.0202 [+.0178,+.0226] EXCLUDES 0, hit
33/33, folds 33/33; K-spread Δ −9.28 [−27.35,+8.32]; pred rank-corr 0.923. Verdict under the
9b-pinned bars: **KEEP INCUMBENT** (fails K-spread ≥0 both windows; no REJECT — no CI entirely
negative; W1b correctly does not fire).

**Artifact prosecutions, all rejected by the review:** eval-horizon (lift survives at the
trained 4h outcome: +0.0189/+0.0139 CI-solid; fwd24 amplifies ~1.3-1.45× via overlap smoothing —
noted); vol-deweighting tilt (vol-shrink overlays on incumbent preds recover <7% of the lift —
the gain is coefficient-level, from removing the extreme-leverage rows: 5% of rows carried 38.6%
of label variance in a LINEAR model); leakage (clip is per-row monotone with a constant
threshold; identity arm proves pipeline parity); panel drift (bit-exact identity arm).

**Interpretation (reviewer-corrected — supersedes the preliminary "tails carry the tip skill"
wording, which was factually wrong in two places):** winsorization improves rank ordering across
the ENTIRE book — mid-book (+0.016..+0.023) and, even more, among extreme-outcome names
(tails-only Δ +0.028..+0.032, CI-solid). What it does not preserve is top-of-book SELECTION:
the variant's top-1-vs-top-2 pred gap flattens ~40% (near-ties → partial tip-pick
randomization), tip picks change in ~half of cycles, and point-estimate tip alpha falls — but
every tip delta's CI crosses zero, so tip degradation is DIRECTIONAL, NOT ESTABLISHED
(instrument tip-CI half-width ±18-63 bps vs ~10-20 bps effects). Book-level rank-IC and
production-K tip value moved in opposite directions on point estimates. Whether fitting tail
magnitudes is where tip skill comes from remains UNRESOLVED at this instrument's power.

**Follow-up ruling (review R3):** the hybrid two-model cell (winsorized book-model + unclipped
tip-model) is NOT registered — with production consuming only the K=1/2 tips it is behaviorally
identical to the incumbent, and the cell would adjudicate on the tip endpoint the instrument
cannot resolve. The legitimate next step is a DESIGN QUESTION, not a cell: does any layer of the
stack consume book-level ranks (sizing, eligibility, rotation)? Only if such a consumer exists
does the winsorized model's (large, real) ordering edge have a monetizable outlet — and that
cell's endpoint must be the consumer's, not the tip spread. Until then the winz2 books stay as
diagnostic state (hl_winz2_*, hl_winz10_*).

## Addendum 10 (2026-07-09, PRE-REGISTERED) — S1: correlation-aware short selection

**Motivation:** the 2 shorts are currently ranks 1-2 by base pred and can be cluster twins
sharing one squeeze event (joint tail). De-concentrate the short-leg tail with a deterministic,
parameter-light selection rule. Predictions unchanged — pure selection-layer cell.

**Rule (S1):** short-1 = bottom-1 by base pred (unchanged). Short-2 = of the next two ranks
(bottom-2, bottom-3), the one with the LOWER trailing correlation to short-1. Longs unchanged.
Correlation: trailing 180-cycle (30d) pairwise corr of 4h returns, shift(1) (PIT; window pinned
by the REGIME_GATE W=180 convention, not tuned). No continuous knobs; the only discrete choices
(pool = ranks 2-3, window = 180 cycles) are pinned here before any computation.

**Machinery:** no retraining — the rule is evaluated directly on the EXISTING incumbent books
(hl_tgt_res_*_clean / hl_v4*_oos_clean) + a PIT pairwise-corr panel. Estimator-law compliant:
book-level paired endpoints; overlay replay secondary only, overlays disabled.

**Endpoints (primary intent is TAIL de-concentration, not mean lift):**
1. Paired per-cycle K-spread Δ (S1 tips vs incumbent tips), day-block CI — GUARDRAIL: point
   estimate must not be significantly negative (CI upper bound > 0); we do NOT set a
   non-inferiority margin (C4 lesson: margins below CI width are de facto superiority tests).
2. TAIL endpoints (verdict-bearing): (a) short-pair joint-tail frequency — fraction of cycles
   where BOTH shorts realize fwd alpha above +X where X = the incumbent short-leg 90th
   percentile (fixed from the incumbent distribution, both windows separately); (b) short-leg
   worst-decile mean (CVaR-style); (c) short-pair realized correlation achieved. Bars: (a)
   reduced with day-block CI excluding 0 in ≥1 window and point-reduced in both; (b) improved or
   flat both windows; guardrail (1) holds.
3. Diagnostic: how often the rule swaps (rank-3 chosen), Δ in the swap cycles only, per-fold.

**Open questions explicitly delegated to the design review (before any run):** power of the
joint-tail endpoint (event count), whether the corr window/pool pins are adequately justified,
whether achieving corr-reduction mechanically guarantees endpoint (c) (triviality risk), and
whether the guardrail phrasing survives the C4 margin critique.
**Multiplicity:** 1 verdict-bearing cell; adoption (if passed) = forward-test candidate only.

### Addendum 10b (2026-07-09, BEFORE the official run) — S1 design-review amendments (REVISE, applied)

Reviewer measured the incumbent anatomy directly (disclosure: quantifying power required applying
the rule once — the reviewer run is recorded, and the corr-measure pin below was chosen AGAINST
the peeked argmax). Key measurements now on record: recent joint-tail events 24 (18 days), OOS
129 (92 days); always-swap-to-rank-3 counterfactual removes 1 / 7 events (rank-3 marginal tail
rate 8.3-8.7% ≈ rank-2's 9.0-9.1%) — **the joint tail is regime-driven, not twin-driven; the
de-concentration ceiling is a few events**. Honest prior: KEEP INCUMBENT. Amendments:

1. **OOS is the sole CI-bearing window for the joint-tail endpoint** (recent = 24 events cannot
   exclude 0 at any plausible effect; its bar is non-contradiction only: not above placebo p95).
2. **Matched placebo added (verdict-critical)**: 200-seed random-swap-to-rank-3 at the matched
   swap rate; promotion requires OOS events strictly < placebo p5, in addition to the paired CI.
3. **Corr pinned to residual space**: trailing-180-cycle corr of alpha_vs_btc_realized, shift(1),
   min 120 valid obs else no-swap (fallback frequency reported). Raw-return corr = sensitivity
   diagnostic only, never verdict-bearing, no post-hoc switch. Known hazard recorded: residual
   corr level ≈ 2× its estimator SE → ~2/3 of picks are within-noise ties.
4. **Guardrail relabeled**: K-spread CI-upper>0 is a gross-error catch only (CI width 10-20×
   the expected drag), NOT mean-neutrality evidence. Expected cost pre-committed: promotion
   means accepting ~3-11 bps/cyc expected drag for the tail benefit. Swap-cycle-only short-leg
   Δ with block CI reported as the honest cost readout (non-verdict).
5. **Endpoint (c) moved to diagnostics**: dose check, mechanically implied by the rule, never
   supports promotion.
6. **Threshold**: same-window incumbent p90 primary (treatment-independent, shared by arms);
   robustness re-reads (other window's X; fixed +700 bps) diagnostic only, cannot flip a verdict.
7. **Clustering**: endpoint (a) scored with 2-day blocks; event-days reported alongside
   event-cycles; event-day McNemar as tie-breaker.
8. Mechanical pins: fwd24 named as the tail outcome; worst-decile = pooled per-window short-leg
   fwd24, mean of highest-alpha decile; rank-3-missing fallback = keep rank-2 (counted).
9. **One official run, verbatim, no post-hoc dose/measure/window changes** (W1b-style refusal
   recorded in advance; a corr-window or pool sweep after a near-miss is refused).

### Addendum 10c (2026-07-09) — S1 RESULT: KEEP INCUMBENT

Official run (live/s1_corr_shorts_eval.py) matches the design-review peek exactly (rec 25 / OOS
115 events — two independent implementations agree bit-for-bit on event counts; this serves as
the reproduction check). Verdict under the 10b bars:

- OOS joint-tail paired Δ −0.00238 events/cyc, 2d-block CI [−0.00545, +0.00085] — **crosses 0 →
  fails**. Events 129 → 115 (92 → 86 days); McNemar 28 incumbent-only vs 22 S1-only days (ns).
- **Placebo bar fails**: S1's 115 is NOT strictly below the matched random-swap placebo p5 (114)
  — the corr-based choice does not beat random de-concentration at the same swap rate.
- Recent non-contradiction: OK (25 within [19, 28]).
- Worst-decile short-leg mean: marginally WORSE both windows (+1898→+1901, +1475→+1497) — fails
  improved-or-flat.
- Guardrail holds (gross-error catch only): Δ −11.2/−1.3 bps/cyc, CIs cross 0 — consistent with
  the pre-committed ~3-11 bps expected drag.
- Dose WAS delivered (achieved c(s1,pick) 0.160→0.095 rec, 0.325→0.261 OOS; swap rate ~0.50,
  zero corr fallbacks) — the rule did what it was designed to do; the tail didn't care.

**Verdict: KEEP INCUMBENT. Mechanism finding (the durable output): the short-pair joint tail is
regime-driven, not twin-driven** — rank-3's marginal squeeze rate equals rank-2's (8.3-8.7% vs
9.0-9.1%), the always-swap ceiling is 1/24 rec / 7/129 OOS events, and delivered corr reduction
moves nothing beyond random swapping. Within-pool selection cannot de-concentrate this tail;
the levers that address it remain regime-level (CUSUM throttle) and structural (K, bands).
No re-runs, no corr-window/pool sweeps (refused in advance, 10b-9). Cell closed.

## Addendum 11 (2026-07-09, PRE-REGISTERED) — M1: model-class cell (pooled LGBM vs per-symbol Ridge)

**Motivation:** last untested axis at fixed data. Hypothesis: tips are extreme-feature regions
where interactions matter and a linear per-symbol model cannot express them; W1 established the
verdict must ride on the SELECTION endpoints, not mean rank-IC.

**Variant arm:** pooled LGBM across the stacked panel with sym_id (repo convention from the ml/
pipeline), same V0_LEAN features (+RR for the long book), same xs_z target (clip ±10 incumbent),
same WF cuts/embargo, two-book structure preserved. LGBM hyperparameters = the repo's pinned set
(CLAUDE.md: pinned across v1→v4; cite the exact params in the generator) — NO tuning. 3 seeds
{0,1,2}, predictions averaged. Sample weights: same HL=60 recency exp-decay.
**Decomposition arm (diagnostic only):** pooled Ridge (same stacking, sym_id one-hot) to
attribute pooled-vs-per-symbol separately from linear-vs-tree. Never verdict-bearing.
**Baseline:** incumbent per-symbol Ridge books. Scoring: matched-population scorer, all 4
endpoints + per-side legs + NO-OP guard.
**Bars (pinned, as 9b-3):** promotion requires Δrank-IC ≥ 0 both windows; rank-IC CI excludes 0
(positive) in ≥1 window; K-spread Δ ≥ 0 both windows; no era-flip (F8); recent hit ≥ 5/9; OOS
≥ 18/33. REJECT if any primary CI entirely negative. Per the W1 law, a rank-IC-only win without
spread conversion is a KEEP, whatever its size.
**Open questions delegated to design review:** composite-treatment concern (pooling + model
class change together — is the pooled-Ridge decomposition arm sufficient attribution?); exact
pinned LGBM params source; min-rows/population handling for pooled training (target_A-clip
lesson from the 111-panel: no clipping hacks, symbols enter as-is); seed-count adequacy;
compute plan. Multiplicity: 1 verdict-bearing cell; gate for the stage-2 shortlist cell (M1
null ⇒ shortlist cell skipped, conditional structure absent at fixed features).

### Addendum 11b (2026-07-09, BEFORE any computation) — M1 design-review amendments (REVISE, applied)

1. **Params pinned: `x6.LGB_PARAMS_POOLED`** (lr .03, leaves 31, depth 6, min_leaf 300, ff/bf .85,
   bagging_freq 5, reg .1/.1, n_estimators 400 FIXED — no early stopping, no calibration split).
   The alpha_v4_xs dict was considered and rejected: it drags in an ES+cal-set protocol the
   frozen machinery has no analogue of.
2. **Weights NOT dropped**: the in-repo pooled precedent functions silently omit sample_weight —
   the M1 generator passes w = exp(−(t_end−open_time)/60d), GLOBAL fold t_end, to every fit.
   Per-fold Σw and ESS printed (final-fold ESS ≈ 11.5% of rows — accepted under pinned params).
3. **Preproc/encoding pins**: LGBM arm = NO preproc, raw features + sym_id int32 with
   categorical_feature, native NaN (X6 convention; trees invariant to monotone transforms —
   per-symbol standardization would be a second treatment). Pooled-Ridge decomposition arm =
   global x6.fit_preproc REFIT PER FOLD (not X6's fold-0 reuse), sym one-hots drop_first
   unstandardized; partially-pooled intercepts noted. Per-symbol-preproc-then-stack REFUSED as
   post-hoc sensitivity. Unseen-symbol test rows: LGBM default direction; scorer intersection
   makes it verdict-irrelevant.
4. **Gate re-pinned**: the stage-2 shortlist cell is skipped ONLY if Δrank-IC CI fails to exclude
   0 (positive) in BOTH windows. A rank-IC-positive/spread-null KEEP leaves stage-2 alive (W1
   law: the spread endpoint cannot see structure at 10-20 bps). Stage-2 stub registered NOW:
   hypothesis = conditional structure among near-tip candidates (grinder vs squeeze) exists and
   is learnable from V0_LEAN + book context at ~10 candidates/cycle; endpoint = the consumer's
   (tip selection spread under nested-fold selection with matched-shortlist placebo); full
   pre-registration required before it runs.
5. **Priors cited (expectations set BEFORE the run)**: diag_pooling_confirm (#165) — pooled
   linear >> per-symbol, pooled GBM ≤ pooled linear; gen_pooled_wf_preds/diag (#167) — pooled IC
   +0.047 vs +0.029 per-symbol BUT loses at the traded top-K legs; X6 24-cell matrix — pooled
   LGBM replay Sharpe −0.63..−2.55, ≤ pooled Ridge in every feature set; per-symbol LGBM
   train_ic ≈ 0.001. None had matched-population paired CIs, this target, 42 folds, or HL-60
   weights — M1 is legitimate, but the prior-consistent outcome is rank-IC-up / spread-null KEEP.
6. **Per-symbol-LGBM arm REFUSED with the correct claim**: underfit/noise, not overfit — X6
   cells 8-11 per-sym LGBM train_ic +0.0010/−0.0007/+0.0003; per-symbol ESS under HL-60 ≈ 720
   rows vs min_data_in_leaf 100-300. Pooled-Ridge decomposition arm is diagnostic-only and
   CANNOT be promoted this cycle regardless of its numbers.
7. **Population**: no per-fold min-rows floor for pooled training (part of the treatment; no
   clipping/exclusion hacks). Scorer intersection drops pooled-only coverage → verdict is
   conservative w.r.t. pooling's coverage benefit (a KEEP must not be over-read). EXCL test-only,
   matching incumbent. Flatness tripwires printed per fold: per-cycle pred std, n-unique preds
   (111-panel lesson).
8. **Seeds**: 5 = ENSEMBLE_SEEDS (42, 7, 123, 99, 314), all four LGBM seed fields set,
   deterministic=True, force_row_wise=True, num_threads fixed (8); mean of RAW preds; inter-seed
   per-cycle pred rank-corr printed.
9. **Panel-freshness check before the verdict**: panel mtime must predate the W1 winz10 identity
   books (which were verified bit-exact against the incumbents); if the panel changed since,
   rerun the identity arm first.
