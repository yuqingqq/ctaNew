# Feature-window tuning program — results (2026-07-08, post-review FINAL)

Pre-registration: `RESEARCH_LOOP_20260707.md` addenda 6b/6c/6d. Machinery:
`live/feature_variant_harness.py` (books), `live/score_variant_cell.py` (book-level endpoints per
the estimator law — no live-overlay replays). Incumbent baseline: V0_LEAN books
(`hl_tgt_res_*_clean` recent / `hl_v4base|v4long_oos_clean` OOS). Endpoints: paired per-cycle
rank-IC Δ (day-block 95% CI), production-K selection spread Δ, monthly hit rates, per-year tables.

**Status: FINAL.** Adversarial results review completed 2026-07-08 (appendix below). Its HIGH
finding — the first-pass scorer paired the two arms on *different* per-cycle symbol populations —
changed one verdict (C2 REJECT → KEEP-incumbent) and one CI classification (C3 OOS now excludes
zero). All numbers below are from the **corrected matched-population scorer** (both arms on
incumbent ∩ variant ∩ fwd per cycle), reproduced locally after the fix. The bottom line is
unchanged: **6 cells, 0 promotions.** Scope caveats up front: the package is NOT complete in the
strict pre-registered-endpoint sense (endpoint 3 never computed — see Endpoint accounting), and
"frozen" means **no promotable variant among the tested cells** — not a claim about the complete
feature universe (Pack 6 liquidity unscreened).

## Scoreboard (matched-population, corrected)

| cell | change | recent Δrank-IC (CI) | OOS Δrank-IC (CI) | other decisive numbers | verdict |
|---|---|---|---|---|---|
| C1 | ret_3d → ret_36h | +0.0014 [−.0006,+.0033] | +0.0001 [−.0013,+.0016] | rec K-spread −18.2; OOS hit 18/33 | KEEP INCUMBENT |
| C2 | ret_3d → ret_6d | +0.0001 (crosses 0) | −0.0012 [−.0027,+.0004] | OOS hit 13/33 | KEEP INCUMBENT (see F1/F2) |
| C3 | ret_3d → resid_ret_3d | +0.0006 (rec hit 4/9 FAIL; era-flip F8) | **+0.0016 [+.0002,+.0030] EXCLUDES 0**; hit 21/33 ✓ | K-spread rec −3.5 / OOS −6.9 both FAIL spread bar | KEEP (nearest miss — non-promotable) |
| C4 | bars_since_high → dd_from_high (parity ladder) | spread −32.2 [−82.3,+16.4] | spread −0.85 | power outcome, not evidence of inferiority (F4) | NO SWAP |
| C5 | corr_to_btc_1d → 12h | −0.0015 [−.0030,~.0000] | +0.0001 | rec hit 2/9; 3.4× screening flag absorbed | KEEP |
| T1 | + taker_ls_24h_lag36h (addition) | +0.0002 | −0.0003 [−.0019,+.0014]; hit 11/33 | univariate t=23.6 absorbed; first-pass +0.0005 was population-inflated | KEEP (no addition) |

Expected false passes over 6 cells ≈ 0.07; observed promotions = 0.

Verdict notes (reviewer-ruled wording):

- **C2**: the first-pass "REJECT — OOS CI entirely negative" was a population artifact (43% of OOS
  cycles had arm-population mismatch because ret_6d's 6-day history shifts symbol entry folds).
  Matched CI crosses zero. "ret_6d is affirmatively worse" is **unsupported**; the supportable
  statement is only that it offers no improvement. The pre-registered population-matched *control
  book* (6b) was never built for C2/T1 — the matched-population scoring above is a scoring-side
  proxy (it cannot undo train-mask differences). Verdict is KEEP-incumbent either way, but
  **no mechanism claim about ret_6d is supported** (neither "worse" nor "equivalent") without
  the control books.
- **C3**: under the corrected paired estimator the OOS rank-IC lift is CI-solid (+0.0002..+0.0030,
  hit 21/33). Still non-promotable under the pre-registered bars: selection-spread Δ negative in
  BOTH windows, recent hit 4/9 < 5/9, and the recent per-year split is an F8 era-flip (2025
  −0.0004 vs 2026 +0.0011). "All OOS years ≥0" is technically true but degenerate (2023 =
  +0.0001) — not evidence. The protocol has **no watchlist tier** for near-miss KEEPs; this note
  is diagnostic only, not a status.
- **C4**: NO SWAP is a **power outcome, not evidence of inferiority** — the recent CI [−82, +16]
  is compatible with parity. The −2 bps non-inferiority margin was unpassable by construction
  (~4% of the estimator's CI half-width; joint pass probability under a perfectly neutral swap
  ≈ 0.1–0.2%). The engineering-parity motivation (bounded feature, kills the truncation guard +
  xs-rank hazard) was never testable by this instrument; if revisited, adjudicate on
  pred-correlation/bit-parity and construction grounds — the alpha wording is forfeited per 6b.
  Incumbent-stays remains the correct conservative default.
- **T1**: the matched recompute *flips the sign* of the OOS Δ (−0.0003) — the first-pass +0.0005
  was inflated by the arm-population mismatch (100% of cycles, metric-cache availability), not
  attenuated. KEEP reinforced. Recorded deviation (F3): the pre-registered "NaN ⇒ preproc
  imputation" pin was not implemented — the harness dropped NaN-variant train rows and skipped
  symbols without metrics caches, changing the training population; effect direction favored the
  variant, so the KEEP verdict is robust to it. As with C2, the verdict rests on matched
  *scoring*, not the pre-registered matched control books — sufficient for KEEP / no-addition,
  **insufficient for any mechanism claim** about taker_ls_24h_lag36h itself.

## Endpoint accounting

Pre-registered endpoint 3 (big-|BTC-4h-move| quintile split of the rank-IC Δ) was **never
computed** for any cell (F5) — the package is therefore not complete in the strict
pre-registered-endpoint sense; the no-promotion conclusion rests on endpoints 1/2/4 only. Not
verdict-bearing under the promotion bars, declared here for the record. The scorer's docstring
and inline comment now say NOT IMPLEMENTED (the first version's comment wrongly implied it was
computed elsewhere). The population-matching defect (F1) is fixed in the committed
`score_variant_cell.py`; any future reuse also requires implementing the quintile split first.

## Screening outcomes that avoided cells (evidence-based closures)

- **XS-rank pack: closed mathematically** — xs-ranking is a within-cycle monotone transform; the
  per-cycle rank-IC is identical to raw (verified to 4 decimals on 5 features); the selection
  layer already consumes only within-cycle ranks.
- **Funding retest under the residual target: dead** — IC +0.006 / +0.002 / −0.002 vs 0.027-0.080
  for V0_LEAN keepers.
- **Autocorr window AND percentile-history: inert** (IC 0.018-0.026 under every perturbation) —
  the proposed 16-combo grid would have been noise mining.
- **OI-change / top-trader ratios as model features: no flag** (|IC| ≤ 0.015 / 0.009).
- **Mixed-beta features: premise rejected** by the beta-label A/B post-mortem (rank-IC Δ ≤ 0;
  the +6.6k/+5.8k replay deltas were overlay-path artifacts). Beta family closed.
- Regime-context features, interactions: closed by prior ledger / model class (addendum 6d).
  **Pack 6 (liquidity) remains unscreened** — closed by low prior + budget only, not by evidence.

## Program conclusions (reviewer-approved wording)

1. **No promotable variant among the tested cells; V0_LEAN stays frozen.** "Frozen" is a
   decision scoped to this candidate set, not a statement about the complete feature universe
   (Pack 6 liquidity was never screened). Improvements, if any, are below the estimator's ~0.002
   Δrank-IC resolution or concentrated in endpoints the bars reject (C3: CI-solid OOS rank-IC
   lift, but negative selection spread in both windows). Program CLOSED per the ≤8-cell budget.
   *Not claimed*: "locally optimal" / "frozen-optimal" — the cells bound improvements; they do
   not confirm optimality.
2. **Univariate screening flags do not survive multivariate redundancy.** Measured pattern, not
   suspicion: C5 (3.4× IC flag) and T1 (univariate t=23.6) were both absorbed by the per-symbol
   Ridge — correlated V0_LEAN features already carry the information.
3. **The estimator law held — and bit deeper than expected.** Judged at book level the cells
   produced clean nulls; judged through live-overlay replays they would have produced false wins
   (beta A/B: ~10-20× amplification on 0.995-correlated preds). This program added a second
   estimator lesson: **paired per-cycle deltas are only paired if both arms share the per-cycle
   population** — a 0.7-symbol mean mismatch was enough to flip one verdict and one CI sign.
4. **Nearest miss for the record**: C3 (residualized 3d momentum) — matched OOS Δrank-IC +0.0016,
   CI excludes zero, hit 21/33; fails recent hit, both spread bars, and the recent era-flip rule.
   Recorded descriptively only (no watchlist status exists in the protocol).
5. The feature layer stays **frozen as-is for the v4 forward test**. Remaining feature-side
   upside requires new data classes (event-level liquidations/borrow; Pack 6 liquidity is
   unscreened) or a model-class change (Ridge → nonlinear) — each a separate, single
   pre-registered question.

## Appendix — adversarial results review (2026-07-08), findings + rulings

Reproduction: all 6 cells replicated from the committed scorer against the books; C3/T1 re-run
bit-consistent, C1/C2/C4/C5 replicated via independent reimplementation (CI endpoints within
seed noise). fwd24 alignment verified correct (X70 `alpha_vs_btc_realized[t]` is forward
close(t)→close(t+4h), so `rolling(6).sum().shift(-5)` = t..t+24h, identical both arms). K-spread
construction matches production (1 long by long-book, 2 shorts by base).

| # | sev | finding | disposition |
|---|---|---|---|
| F1 | HIGH | scorer paired arms on different per-cycle populations (C2 43% of OOS cycles mismatched, T1 100%, C1/C3 18-23%) | scorer FIXED (intersect both arms); all cells re-scored; C2 verdict downgraded, C3 CI reclassified, T1 sign flipped |
| F2 | HIGH | pre-registered 6b population-matched control books never built (C2 trigger: symbol entry-fold shifts; T1 trigger: row ratio 0.9859 > 0.5%) | verdicts annotated as population-confounded with matched-scoring proxy quoted; control outstanding — REJECT wording for C2 withdrawn accordingly |
| F3 | MED | T1 "NaN ⇒ imputation" pin not implemented (train rows dropped; cache-less symbols skipped) | deviation recorded; direction favored variant → KEEP robust |
| F4 | MED | C4 −2 bps non-inferiority margin unpassable by construction (de facto superiority test; neutral-swap pass prob 0.1-0.2%) | C4 reworded as power outcome; alpha wording forfeited; parity question unresolved by this instrument |
| F5 | LOW | pre-registered endpoint 3 (BTC-move quintile split) never computed | absence declared; required before scorer reuse |
| F6 | LOW | C3 "all OOS years ≥0" degenerate (2023 ≈ +0.0001) | claim demoted in wording |
| F7 | COSM | C5 rec CI upper bound seed-fragile at ~0.0000; dead `fwd24()` stub; broad KeyError | immaterial to verdicts; noted |

Verdict rulings: C1 PASS · C2 REVISE→KEEP-incumbent · C3 PASS outcome/REVISE wording ·
C4 PASS outcome/REVISE wording · C5 PASS · T1 PASS (reinforced). Closing wording REVISE:
"locally optimal"/"frozen-optimal" replaced (see conclusion 1). Closure decision itself
legitimate — corrected C3 numbers do not reopen the program under the bars as written.

## Artifacts

- Books: `live/state/convexity/hl_{ret36h,retc1,resid3,ddc2,corr12h,takerls}_{base,long}[_oos]/`
- Generators/scorer: `live/feature_variant_harness.py`, `live/score_variant_cell.py` (F1-fixed),
  `live/gen_beta_label_ab.py` (all currently untracked — commit together)
- Full audit trail: `RESEARCH_LOOP_20260707.md` addenda 3-7
