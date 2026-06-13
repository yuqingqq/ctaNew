# Strategy Re-Engineering Plan v3 (post 2-round 3-agent review)

Created 2026-05-18. Status: **v3 — LOCKED on the enumerated fixes;
proceed to CODE implementation (not more prose).** v1 + Step 94–102
"closure" RETRACTED (flawed harness). v2 was directionally right but
insufficient (round-2 R1/R2/R3). v3 = the convergent fix-list + re-aim.
Production LGBM unaffected throughout.

## 0. ALL validated flaws v3 must fix (rounds 1+2)

Round-1: (1) preprocessing prose-not-wired; (2) "LGBM<Ridge⇒ceiling"
from a mis-tuned LGBM; (3) per-symbol t iid-SE inflated ~6–10×;
(4) 42-sym gate on 19-sym∩; (5) worst-case naive/retail/naked-4h cost;
(6) profitable objective never tested.
Round-2 (deeper): (7) **`sigma_idio` TARGET contaminated** — fold-0
freeze + cross-symbol-median fallback for late-listed (often "best")
symbols; (8) **target/framing may be mis-specified** — arc's own
evidence (Step-77 decile-inversion, Step-80a magnitude/squared
structure, DDI short-side-only) points to asymmetric/magnitude/longer-
horizon; symmetric sign-residual-4h + rank→inv-normal destroys it;
(9) self-checks not in the binding gate (recreates #1's category);
(10) kurtosis>50 = live-routing knob (flips fold-to-fold); (11) §2b
early-stop slice inconsistent with shuffled-time CV (optimistic stop);
(12) native-universe → non-comparable per-block fold calendars;
(13) max()-of-3 noisy baselines; (14) §4 ensemble corr biased by
mismatched aggregation; (15) plan mis-prioritized (grid pre-answered;
§4 is the only real EV).

## 1. Objective (unchanged) — profitable system

Success-A: profitable standalone (rebuilt-harness net Sharpe>+1.5, CI
excl 0, at maker cost, native universe, ≥6/9 windows, beats P2 placebo,
no window>60% PnL). Success-B: portfolio-accretive sleeve vs production
V3.1. Failing both = a **narrow** negative, explicitly NOT "information-
bounded".

## 2. HARNESS REBUILD — code, named functions, blocking self-checks

New module `composite_study/harness_v3.py` REPLACING the
`s94b.grouped_oof` / `feature_reengineering.ceiling` path:

- **2a Preprocessing** wired in `harness_v3.preprocess(df, train_mask)`.
  **Static pre-registered per-feature transform map** (a literal dict in
  code; NO live kurtosis routing — fix #10): each base+interaction
  feature → {xs_rank_invnorm | symz_then_xsrank | raw_xsz}, assigned by
  feature family from domain reasoning, frozen before any run.
  Strict-past per-symbol z `=(x−g.shift(1).rolling(2016,min_periods=504)
  .mean())/std`, std floored at 1e-9. Interactions: raw PIT inputs →
  product → per-cycle rank→inv-normal `(rank−0.5)/N`. NaN: per-symbol
  strict-past rolling-median; no past ⇒ drop row (no cross-symbol).
- **2b TARGET FIX (#7, critical):** σ_idio recomputed **per-fold,
  strictly-past, per-symbol, NO cross-symbol-median fallback** — a
  symbol with no strict-past σ is **ineligible that fold** (same rule as
  features). `tz` rebuilt per fold from that σ. Do NOT consume the
  panel's frozen `sigma_idio`.
- **2c Model envelope (#2,#11):** Ridge α∈{1,3,10,30} + LGBM
  early-stopped on the **chronologically-last block of the train fold**
  (ordered/expanding inner split — NOT a shuffled inner slice). Report
  all members; ceiling = best honest member.
- **2d Honest inference (#3):** per-symbol IC significance via
  block-bootstrap, block_len ≥ 6 (= embargo), n_boot ≥ 1000. Placebos:
  within-symbol perm AND **P2 within-selected-universe**, ≥1000 perms.
- **2e Native universe + PINNED COMMON FOLD CALENDAR (#4,#12):** full
  per-block symbol coverage but a single common walk-forward calendar
  (intersection of test dates) so cross-block ceilings are comparable.
  Report breadth N + breadth-adjusted bar.
- **2f Cost (#5):** gate at maker (~1bps/unit); taker (3.5/side)
  reported as pessimistic disclosed bound.
- **2g Self-checks are BLOCKING gate conditions (#9):** auto-FAIL if
  any of — refit-per-fold vs calendar-causal preprocess ≠ to 1e-12;
  post-preprocess PIT look-ahead |corr|≥0.10; target σ_idio uses any
  future or cross-symbol info. These run before scoring; failure aborts.

## 3. EXECUTION ORDER (re-prioritized #15) — highest EV first

1. **§2 harness rebuild** + 2g self-checks pass (prerequisite).
2. **§4 ensemble-with-production FIRST (kill-fast).** Spec (#14): build
   the naive maker-cost linear-idio book on native universe; align its
   PnL and production V3.1 PnL to a **common non-overlapping settlement
   grid**; per-fold PnL-return correlation; **nested-OOS** blend weight
   (from strictly-past folds). Success-B = blend Sharpe lift ≥ **+0.3**
   AND paired block-bootstrap CI on the difference excludes 0 AND no
   single window > 60% of the lift, measured vs V3.1's **honest-forward**
   Sharpe (NOT in-sample +2.23). **Pre-registered kill:** if §4 lift CI
   includes 0 → linear-idio line is a narrow negative; do NOT run §5.
3. **§3 ONLY IF §4 ambiguous/positive:** a 5-run **horizon sweep**
   {4h,8h,24h,48h,72h}, naive construction, maker cost, native
   universe, rebuilt harness. (Construction & cost sub-axes dropped —
   already answered negative by construction_rankK / cost_sensitivity.)
4. **§3.5 target-framing test (#8, the deepest lever) — run in PARALLEL
   with §4** (independent hypothesis, cheap): predict alternative
   targets the arc's own evidence implies — (i) asymmetric short-leg-
   only payoff, (ii) realized idio-vol / squeeze magnitude, (iii)
   longer-horizon residual — **without** Gaussianizing the magnitude
   (these targets keep magnitude). Same rebuilt harness, same gate.
5. **§5 feature/redundancy LAST, only if something upstream is
   positive** (corrected per R1: cluster rep = LGBM-gain; rule
   text=code; block-bootstrap IC; on preprocessed train-fold features).

## 6. Gate / 7. Strict nested forward validation / 8. Process

Gate per §1 with 2g self-checks blocking and the 60%-window
anti-fragility check in-gate. §7: any adoption requires strict
nested-OOS with preprocessing-fit, model, σ_idio, pruning, blend-weight,
AND cell/target choice ALL decided from strictly-past folds + LOFO + P2
placebo. §8: the 3-agent loop now reviews the **implementation and each
stage's RESULTS vs pre-stated expectation** (converge by running, not
prose); mismatch ⇒ re-initiate that stage. No goalpost-moving; honest
records same-day; production untouched.

## 9. Next action

Implement `harness_v3.py` (§2) + 2g self-checks; run §4 (+§3.5 parallel)
first; 3-agent result-check; proceed only on expected results.

---

## §5-INT — interaction features (OWNER-AUTHORIZED scope re-open)

**Status: LOCKED 2026-05-19 BEFORE any run. Pre-registered; the gate is
the only arbiter.** §5 was pre-registered *off* contingent on §4 failing
(§3 step 2). §4 DID fail. The owner explicitly directed testing
spot-perp / price-volume / order-flow interactions. This is the owner
authorizing a scope change, not a rescue of the §4/§3.5 negatives nor
goalpost-moving — it is a NEW pre-registered hypothesis with its own
locked gate, on the trustworthy `harness_v3` (the prior interaction
runs, Steps 95–102 / feature_reeng, were on the RETRACTED flawed harness
and are NOT authoritative). Why it is the right lever: §4's binding
constraint is standalone signal too weak (inverse-tangency: need
standalone Sharpe ≥+1.45 at ρ≈0.05). Construction (§4) and target
(§3.5) cannot move that; only genuinely more-informative features can.

**Enumeration (LOCKED, the 26 interactions in
`feature_reengineering.py` Stage 1 — products of PIT inputs only):**
no new feature invented post-hoc; the list is frozen as-is.

**Two tiers (honest about the universe-shrink trap R1 caught before):**
- **Tier A — 42 syms, NO universe shrink.** The 13 interactions needing
  only the base panel: price-volume (`x_r1d_volz, x_st_volz, x_r1d_obv,
  x_r1d_vws`), short×long momentum (`x_r1d_r8h, x_r1d_sq, x_st_r1d,
  x_st_autoc`), funding/dom/beta (`x_fz_r1d, x_st_corrb, x_st_betac,
  x_r1d_domz, x_st_idiov`). Directly answers "do price-volume / momentum
  interactions of the data we already have lift on the full universe?"
- **Tier B — 19 syms (perp∩spot∩oi).** Tier-A set PLUS spot-perp
  (`x_basis_r1d, x_spvr_r1d, x_spti_st, x_basis_fz`), order-flow
  (`x_ofi_r1d, x_ofi_st, x_oftfi_volz, x_ofkyle_st, x_ofi_oichg,
  x_ofi_spti`), OI (`x_oic_r1d, x_oiz_st`), `x_lst_r1d`. Joins
  `oi_panel`/`spot_panel`/`oflow_panel` (PIT inner-join on
  symbol+open_time). Gate is **WITHIN-universe**: vs the 19-sym
  matched-grid V3.1 AND the 19-sym base-only — NEVER vs the 42-sym
  baseline (the exact pre-baked-FAIL trap from the retraction).

**Harness & leak guard:** `harness_v3` UNCHANGED. Interactions enter
`preprocess` as **xsr** (per-cycle rank→inv-normal of the raw product) —
plan §2a's prescribed interaction transform; products of PIT inputs
ranked per cycle are PIT-safe. The 3 BLOCKING self-checks run first;
interactions are the classic look-ahead trap and the post-preprocess PIT
check (incl. the magnitude sniff) is the guard — any |corr|≥0.10 aborts
(CLAUDE.md: ">+0.10 IC ⇒ probable hidden look-ahead"). σ_idio target =
the same per-fold strict-past (no cross-symbol). Model envelope: Ridge
α-grid + early-stop LGBM, base-only vs base+ENG (nested superset — fixes
the feature_reeng v1 degenerate-baseline bug).

**Pre-registered gate (per tier, LOCKED):** PASS iff EITHER
- **Success-A:** base+ENG standalone net Sharpe **> +1.5 @ MAKER**,
  block-bootstrap CI excl 0, ≥⌈⅔·n_folds⌉ folds+, no fold >60% of PnL,
  beats P2 within-selected-universe placebo; OR
- **Success-B:** base+ENG nested var-min blend lift **≥ +0.30** vs that
  tier's matched-grid V3.1, paired block-bootstrap CI excl 0, no fold
  >60% of the lift;
- **AND (mandatory marginal-lift):** base+ENG must beat **base-only** by
  **≥ +0.30** net Sharpe within the SAME universe & cycles (isolates the
  interaction contribution from the base). Reported at BOTH MAKER and
  COST=2.25.

**Pre-registered kill:** if NEITHER tier clears (Success-A OR
Success-B) AND the marginal-lift over base-only is < +0.30 in both tiers
at both costs ⇒ the feature-interaction lever (incl. spot-perp /
price-volume / order-flow) is also a negative. The linear line is then
closed **including** the interaction lever — the owner's specific
question answered honestly; no further in-scope variant.

**Process:** 3-agent DESIGN review BEFORE the run (enumeration
completeness; spot/oi/oflow join PIT-correctness; two-tier
within-universe gate fairness) — re-initiate the design on mismatch.
3-agent RESULT review after (genuine lift vs leak/overfit; bug-masked
pass). Same-day honest records. Production LGBM untouched.

### §5-INT-v2 — CORRECTED leak guard (re-initiated 2026-05-19 post DESIGN review; LOCKED BEFORE re-run, before any economic result seen)

3-agent DESIGN review (R1 RE-INITIATE; R2,R3 defer-to-R1) found the
§5-INT magnitude-abort to be a **mis-calibrated leak test**: aborting on
`|corr(|feat|, |αβ_{t+1}|)| ≥ 0.10` is a category error — `|αβ|`
autocorrelation alone is ~0.20 with zero features, so the magnitude
channel false-positives on the whole vol-family (benign GARCH
persistence, not look-ahead; all signed PIT corrs were clean ~0.003–
0.016, prefix-causal Δ=0). Disciplined fix = a leak-SPECIFIC guard,
correct by construction, pre-registered here before re-running BOTH
tiers from scratch. NOT relaxing the bar, NOT dropping the locked
`x_st_idiov`, NOT Tier-B-only, NOT a retro-edit. harness_v3's §4/§3.5
verdicts are UNAFFECTED (its magnitude sniff there *passed* at 0.0645 —
non-binding; only a false *abort* is the failure mode and none occurred;
§4/§3.5 failed on economics, not the guard — do NOT re-open them).

**Corrected BLOCKING guard (LOCKED, exact statistics + thresholds):**
applied to every interaction's xsr transform; abort (do not score) iff
ANY fails:
- **G1 signed-PIT (unchanged, valid):** `|spearman(xsr(feat)_t,
  αβ_{t+1})| < 0.10` (same-symbol next cycle). The project-canonical
  signed look-ahead sniff. (R3 verified all 26 ≈ 0.003–0.016 — clean.)
- **G2 prefix-causal (unchanged):** interior-cut recompute of xsr on
  the strictly-past prefix == whole-panel, max|Δ| < 1e-9.
- **G3 leak-asymmetry (REPLACES the magnitude-abort; the principled
  leak test):** a strictly-past PIT feature must not predict the FUTURE
  better than the CONTEMPORANEOUS bar. For every interaction, in BOTH
  signed and magnitude form, abort iff `|corr(feat_t, αβ_{t+1})| >
  |corr(feat_t, αβ_t)| + 0.02`. Benign persistence decays
  (next ≤ same) ⇒ PASS; a genuine forward leak has next ≳ same ⇒ ABORT.
  (Tolerance 0.02 = sampling noise band, frozen.)

**Success-A P2 wiring (R3 fix, LOCKED):** Success-A additionally
requires the treatment book to beat the **P2 within-selected-universe
placebo p95** (`harness_v3.p2_placebo`, n_perm=1000, computed lazily
only when the other gA conditions already hold). A Success-A pass
without P2 is not a pass.

Everything else (enumeration, two tiers, Success-A/B thresholds,
mandatory ≥+0.30 marginal-lift vs base-only, both cost rates, kill
condition, 3-agent RESULT review) is UNCHANGED from §5-INT above. The
RESULT review must also re-confirm G1/G3 at run time (R3: Tier-B G1
headroom is thin at 0.0835).
