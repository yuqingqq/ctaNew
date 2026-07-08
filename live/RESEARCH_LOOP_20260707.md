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
incumbent, C4 NO SWAP, T1 no addition. Approved closing: **no promotable variant at the
pre-registered bars; V0_LEAN stays frozen; improvements, if any, are <~0.002 Δrank-IC or in
endpoints the bars reject; program CLOSED per cell budget** ("locally optimal" claim withdrawn —
the cells bound improvements, they don't confirm optimality). New estimator lesson for the law:
paired per-cycle deltas are only paired if both arms share the per-cycle population — a
0.7-symbol mean mismatch flipped one verdict and one CI sign.

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
