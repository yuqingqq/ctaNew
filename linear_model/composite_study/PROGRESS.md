# Progress Tracker — linear β-residual line (post-retraction)

**Last updated:** 2026-05-19
**Single source of truth for status.** Plan = `FEATURE_REENGINEERING_PLAN.md`
(v3). Full history = `../docs/STATUS.md` + memory. This file = the thin
living tracker: what ran, what it means, what's next. Production LGBM
unaffected throughout.

---

## TL;DR status — ⛔ LINE CLOSED (pre-registered terminus, triple-confirmed)

The linear β-residual line is a **NARROW, HONEST NEGATIVE** within the
fixed 4h-free-Binance-perp scope — explicitly **NOT "information-bounded"**
(the over-claim that was retracted). Conclusion reached HONESTLY on a
rebuilt, 3-agent-cleared harness, not the flawed Steps-94–102 one.

- ✅ **Harness rebuilt & trustworthy.** `harness_v3.py`: per-fold
  strict-past σ_idio (no cross-symbol fallback), static frozen transform
  map, model envelope, 3 *genuine* BLOCKING self-checks (anti-vacuous;
  closes flaw #9). 3-agent-cleared.
- ⛔ **§4 construction test — SUCCESS-B FAIL (3/3-agent).** vs
  matched-grid V3.1 **+2.747**; nested-blend lift **−0.92** (CI
  [−1.95,+0.14]); in-sample UB == V3.1 ⇒ infeasible for ANY weight. The
  one robust positive (corr +0.05) HELD but is necessary-not-sufficient.
  Prelim's "+0.92 lift" was 100% wrong-baseline — formally void.
- ⛔ **§3.5 target-framing test (last independent hypothesis) — NEGATIVE
  (3/3-agent).** Short-only (DDI) & magnitude-conviction (Step-80a),
  ridge & lgbm, both costs: best standalone +1.316 < +1.5 (CI incl 0);
  every Success-B lift negative. Target is **not** the limiter; flaw #8
  tested & negative. DDI short-side alpha **refuted as tradeable** on
  free 4h data (real as cohort IC, portfolio-invisible under cost).
- ⛔ **§5-INT interaction test (OWNER-AUTHORIZED re-open) — NEGATIVE
  (3/3-agent).** Owner pushed: "why not spot-perp / price-volume
  interactions". 26 locked interactions, 2 tiers (A 42-sym price-vol /
  short×long no-shrink; B 19-sym +spot-perp/order-flow/OI). Corrected
  leak-specific guard (G3 leak-asymmetry, re-initiated after R1 caught
  the original magnitude-abort as mis-calibrated) PASSED clean; red-team
  proved it still catches real leaks. Interactions add *marginal*
  structure (ridge TierA +0.34, lgbm TierB +0.50 — but single-fold
  variance-luck per LOFO) yet **every cell fails Success-A & Success-B**;
  best standalone +1.16 < +1.5; all nested lifts vs V3.1 negative.
  spot-perp/price-volume/order-flow add **no tradeable signal**.
- ⛔ §3 horizon-sweep gated off + out-of-scope (user fixed goal to 4h);
  §5 feature/redundancy + §5-INT interactions now BOTH closed.
- **Final P(profitable/accretive in-scope) ≈ 2–4%** (R2; was 3–5% →
  §5-INT resolved the last named in-scope lever negative). The remaining
  hope is OUT-OF-SCOPE only: orthogonal *paid* data / longer horizon
  ≈ 20–30% — deferred by the user's "fix the goal to 4h first"
  directive; re-opening requires a user-decided scope change.
- ➡️ **Action: WIND DOWN the linear line.** No further in-scope runs.
  Production LGBM (V3.1, honest-forward +2.747 matched / +2.23 headline)
  **unaffected throughout** — it remains the production strategy.

## v3 stage checklist

- [x] **§2 harness_v3.py** — built, 3 genuine BLOCKING self-checks PASS,
      3-agent-cleared (R1/R3 ALIGNED, R2 §4-design blocks neutralized).
- [x] **§4 ensemble-with-production** — RUN. **SUCCESS-B FAIL**, 3/3-agent
      confirmed (matched-grid V3.1 +2.747; lift −0.92; in-sample UB==V3.1).
- [x] **§3.5 target-framing test** — RUN. **NEGATIVE**, 3/3-agent
      confirmed (short-only & mag-conviction; best standalone +1.316<+1.5;
      all Success-B lifts negative). Target is not the limiter.
- [x] ~~§3 horizon sweep~~ — gated OFF (§4 not ambiguous/positive) +
      out-of-scope (user fixed goal to 4h).
- [x] ~~§5 feature/redundancy~~ — KILLED by §4 pre-reg kill.
- [x] **§5-INT interaction features (owner-authorized re-open)** — RUN
      (after a re-initiate to fix a mis-calibrated leak guard).
      **NEGATIVE**, 3/3-agent confirmed. Interaction lever closed.
- [—] **§7 strict nested forward validation** — N/A (nothing passed to
      validate; line closed across §4 / §3.5 / §5-INT).

## Run log

| date | run | result | status |
|---|---|---|---|
| 2026-05-18 | Steps 94–102 (D1 ceiling, D1-ext A–G, structural events) | "information-bounded / airtight" | ⛔ **RETRACTED** — flawed harness |
| 2026-05-18 | composite_study: per-symbol idio, raw decomp, cost sweep, construction rank-K, feature_reeng | various negatives | ⚠️ conditional on flawed harness; not authoritative |
| 2026-05-18 | feature_reeng v1 | F_core swept all feats (bug) | ⛔ invalid (disclosed) |
| 2026-05-18 | feature_reeng v2 (bug-fixed) | gate FAIL (pruned −0.28) | ⚠️ conditional — preprocessing never wired (R1) |
| 2026-05-18 | 3-agent review R1/R2/R3 (×2 rounds) | found 15 validated flaws | ✅ drove retraction + v3 |
| 2026-05-18 | **§4 ensemble PRELIMINARY** (triage) | **corr +0.06**; in-sample blend +1.18 (lift +0.92, NOT trusted) | 🟡 **preliminary pulse — warrants clean §4** |
| 2026-05-19 | **V3.1 baseline anomaly RESOLVED** | prelim used wrong CSV (final_sim `net_with_overlay_bps` = K=3-no-sleeve = +0.26). Correct = `vBTC_sleeve_horizon/per_cycle_V3.1_equal6_baseline.csv::net_pnl_bps` = **+2.229** (√2190, matches doc +2.23) | ✅ blocker 1 cleared; "+0.92 lift" formally void (wrong baseline) |
| 2026-05-19 | **harness_v3.py built + 1-fold self-test** | 3 BLOCKING self-checks ALL PASS: prefix-causal Δ=0.0; post-preprocess PIT worst\|corr\|=0.017<0.10; σ_idio strict-past Δ=0.0, exit_time≤open_time. Panel 67538 rows / 42 syms / 1620 cycles / 2025-07-19→2026-04-30 (== V3.1 grid). Smoke ridge IC +0.005 (honestly near-zero) | ✅ built |
| 2026-05-19 | **3-agent harness review (R1/R2/R3)** | R1 ALIGNED (a–e wired, 5 non-blk), R3 ALIGNED (every attack REFUTED w/ indep repro; harness causally clean, no leak), R2 RE-INITIATE — 2 §4-run-design false-pos vectors (NOT harness leaks): cost asymmetry (linear@MAKER vs V3.1@COST=2.25) + fold-coverage (linear=folds3-9/1260cyc; matched-grid V3.1=**+2.747** not headline +2.229) | 🔁 **re-initiate**: harden self-checks (anti-vacuous) + bake R2 blocks into §4 |
| 2026-05-19 | **harness self-checks HARDENED + re-run** | (1) prefix-causal now 5 interior cuts×8 syms on FULL preprocess Δ=0.0; (2) PIT adds magnitude sniff, worst \|corr\|=0.0645 on \|atr_pct\| <0.10; (3) explicit single-symbol-isolation no-xsym assert Δ=0.0. Closes flaw #9 (fake-check category that caused retraction) | ✅ harness genuinely trustworthy |
| 2026-05-19 | **§4 CLEAN NESTED ensemble (THE verdict)** | matched-grid V3.1=**+2.747** (NOT headline +2.228 — R2#2 fix decisive). corr(lin,V31)=**+0.05** (prior finding HELD). But ALL 4 cells negative lift: primary ridge@COST2.25 lift **−0.918** CI[−1.95,+0.14]; best lgbm@MAKER −0.365 CI[−1.11,+0.43]. Gate FAIL (lift, CI, folds+ 1/7) | ⛔ **SUCCESS-B FAIL — pre-registered KILL triggered** |
| 2026-05-19 | **3-agent §4-result review (R1/R2/R3)** | **3/3 ALIGNED.** R1: all numbers reproduced to 4dp, join exact (0 dropped), var-min honest. R3: dropped folds 1-2 Sharpe +0.999 vs kept +2.747 ⇒ FAIL is **conservative against linear book**, not survivorship; joint CI tighter than naive. R2+R1+R3: in-sample UB==V3.1 for ridge ⇒ Success-B **weight-method-independently infeasible**; even cheating lgbm UB +0.20/+0.25 < +0.30 | ✅ FAIL is a genuine honest negative; KILL correct; proceed §3.5 |
| 2026-05-19 | **§3.5 target-framing (LAST independent hypothesis)** | T0/T1-short-only/T2-mag-conviction × ridge/lgbm × MAKER/COST2.25. Best standalone lgbm T0 **+1.316**<+1.5 (CI incl 0); ALL Success-B lifts negative (best ridge T1 −0.372). DDI short-only does NOT yield a profitable book; magnitude framing < +1.5. **Bug caught+fixed+disclosed**: window-share formula `\|f\|/\|Σf\|` exploded >100% on mixed-sign folds → fixed to `max\|f\|/Σ\|f\|` (30-58%); verdict unchanged (gates fail on Sharpe/lift, not window) | ⛔ **§3.5 NEGATIVE — target is NOT the limiter** |
| 2026-05-19 | **§5-INT pre-registered (owner-authorized re-open)** | owner pushed back ("why not spot-perp/price-vol interactions"); 26 locked interactions, 2 tiers (A 42-sym base-only, B 19-sym +spot/oflow/oi); harness_v3 reused; gate locked pre-run | ✅ pre-reg locked |
| 2026-05-19 | **§5-INT 3-agent DESIGN review (R1/R2/R3)** | smoke: Tier-A interaction-PIT guard fired (\|x_st_idiov\| mag 0.14>0.10). **3/3: benign vol-clustering NOT leak** (past 0.18≥same 0.15≥next 0.14 decay; signed −0.009). R1 RE-INITIATE: magnitude-abort is a mis-calibrated leak test (category error); R2/R3 defer to R1. R3: P2 not wired into gA; harness_v3 §4/§3.5 unaffected (guard there passed, non-binding) | 🔁 **re-initiate: fresh pre-reg w/ leak-specific guard, re-run both tiers** |
| 2026-05-19 | **§5-INT-v2 re-run (corrected leak-specific guard, LOCKED pre-econ)** | Guard G1/G2/G3 PASS clean both tiers (G3 leak-asym correctly clears benign vol-clustering, would still abort a real fwd leak). **NEGATIVE**: interactions DO add marginal signal (ridge TierA margin +0.34, lgbm TierB +0.50 — mandatory gate passed 3 cells) BUT absolute hopeless — best base+ENG standalone lgbm TierA +1.16<+1.5 (CI crosses 0), every nested lift vs V3.1 +2.75 ∈[−0.37,−1.69]. lgbm+ENG HURTS TierA (−0.16: trees already auto-search interactions) | ⛔ **§5-INT NEGATIVE — interaction lever closed** |
| 2026-05-19 | **Production model × universe 2×2 (owner-requested)** | prod WINNER_21 LGBM + V3.1 construction verbatim, universe varied: 50-set **+2.748**; drop-only-VVV (adaptive) **+2.569**; 42-set/hl42∩pool (38 syms) **−2.553**. **UNIVERSE dominates** (same model: +2.75→−2.55). **CORRECTED earlier over-claim**: ex-VVV is +2.57 not +1.63 (adaptive construction re-fills) — not a single-VVV lottery, it's the vol-convexity mechanism on whatever tail names exist. Linear "dead" verdicts were on hl42 where even the PROD model = −2.55 ⇒ universe-confounded | ✅ universe ≫ construction(+1.7) > model(+1.2); VVV-fragility corrected |
| 2026-05-19 | **Meme-capture mechanism (owner-requested)** | NOT selector (traded vol≈pool, quartile share uniform). High-vol/long-tail quartile hit-rate **47.9%<coin-flip** but largest PnL via 2× moves ⇒ **vol-amplified CONVEXITY not skill**. VVV: hit 49.4%, edge in 3/9 folds, 55% from 5 long pump-legs. Short side (51.5%, +funding_z squeeze) = only thin genuine edge. Disclosed+corrected a buggy auto-label ("GENUINE skill" → vol-convexity; 5th own-bug). Explains why large-universe also concentrates (tail-name swap, model can't predict which) | ✅ mechanism = fragile lottery-convexity, not meme alpha |
| 2026-05-19 | **Definitive same-construction test (owner-requested)** | linear preds → V3.1's EXACT machinery (phase_ah_sleeve verbatim). CONTROL (prod pred) folds3-9 **+2.748** == doc +2.747 (port faithful). LINEAR ridge → V3.1 = **+1.567** (vs its naive-book −0.10 → construction worth +1.7); LINEAR lgbm → −0.19. vs CONTROL +2.75 → model/features add separable +1.2. **Production +2.75 ≈ ~+1.6 construction + ~+1.2 model.** "Linear signal dead" was an over-claim; naive β-residual *book* was the main killer | ✅ MODEL vs CONSTRUCTION attributed; major honest reframing |
| 2026-05-19 | **Same-universe re-baselining (owner-requested)** | linear book on the EXACT V3.1 50-sym pool (VVV/BIO incl): ridge −0.10/−0.52 (= its hl42 number), lgbm +0.69/+0.64 (worse than hl42 +1.32), corr +0.07–0.09, nested lift −0.49 to −0.77. **§4 negative robust to universe** (not an exclusion artifact). Correct baseline established: V3.1-full +2.747 (VVV-inflated) / ex-VVV ≈+1.63 (honest broad); linear accretes to NEITHER. Production edge is directional-concentrated, structurally not a β-residual signal | ✅ comparison now apples-to-apples; both honest negatives confirmed |
| 2026-05-19 | **3-agent §5-INT-result review (R1/R2/R3)** | **3/3 ALIGNED.** R1: corrected guard verified (injected weak 8%-fwd leak evades G1 but G3 aborts; benign decays pass); marginal isolation correct. R3 red-team: guard blind-spot ≤7%-fwd-blend exists but **proven economically non-exploitable** (margin +0.009/−0.31); +0.34 ridge marginal = single-fold variance-luck (LOFO drop f3 → −0.13); 8/8 cells reproduce. R2: kill fires; real-but-portfolio-invisible (DDI pattern); P(in-scope)≈2–4% | ✅ **NEGATIVE genuine — interaction lever honestly closed** |
| 2026-05-19 | **3-agent §3.5-result review (R1/R2/R3)** | **3/3 ALIGNED.** R1: share-fix verdict-invariant, missing-P2 moot (Sharpe<+1.5 precondition never met), T1/T2 faithful; mag IS predictable (IC+0.12) but dir-IC≈0 ⇒ unmonetizable (flaw#8 tested-neg). R2: Success-A misses on est+CI+breadth; **DDI short-side REFUTED in-scope**; P(in-scope)≈3–5%, P(out-of-scope)≈20–30%. R3: every attack failed; least-bad cell negative GROSS of cost; T1 sign correct (no buried edge) | ✅ **terminus confirmed — LINE CLOSED** |

## What is trustworthy right now

- ✅ **Low linear↔V3.1 correlation (+0.05)** — confirmed on the clean
  harness/matched grid (low in 6/7 folds; only f3 +0.44). Robust. BUT
  necessary-not-sufficient: a near-uncorrelated *but unprofitable* sleeve
  cannot accrete (proven by the in-sample UP­PER BOUND == V3.1).
- ✅ **§4 SUCCESS-B FAIL is a genuine honest negative** — 3/3-agent
  reproduced; conservative against the linear book (dropped folds had
  Sharpe +0.999 < kept +2.747); weight-method-independent (UB==V3.1).
- ✅ **Matched-grid V3.1 honest-forward = +2.747** on the 1260-cyc folds
  3–9 grid (headline +2.228 over all 1620). The prelim's "+0.92 lift"
  was 100% wrong-baseline — formally void.
- ❌ Linear standalone Sharpe — ridge −0.52@COST2.25 / −0.10@MAKER,
  lgbm +1.20/+1.32. Sub-cost (ridge) / below the +1.5 Success-A bar.
- ❌ Every Step 94–102 "closed/terminus" verdict — still retracted (the
  honest re-test confirms a negative, but a NARROW one, not the
  over-claimed "information-bounded").

## Final conclusion (2026-05-19) — what was actually established

- **The linear β-residual line does not produce a profitable or
  portfolio-accretive system on free 4h Binance-perp data, across ALL
  three fixed in-scope levers — construction (§4), target-framing
  (§3.5), AND feature interactions (§5-INT, owner-authorized re-open:
  spot-perp / price-volume / order-flow).** This is a *narrow* honest
  negative: established on a rebuilt harness whose 3 self-checks are
  genuine and 3-agent-cleared; all three decisive tests failed their
  **pre-registered** gates with **independent 3/3-agent reproduction**;
  one re-initiate (mis-calibrated interaction leak guard → fixed via
  fresh pre-registration, not retro-edit); no goalposts moved.
- **Explicitly NOT "information-bounded"** (the retracted over-claim).
  Out-of-scope levers were never tested and are not refuted: orthogonal
  data (e.g. Glassnode), longer holding horizon (the cost-amortization
  mechanism that makes production V3.1 work). R2 P(profitable if scope
  expanded) ≈ 20–30% — but the user fixed the goal to 4h-free-data, so
  these are deferred, not failures.
- **Robust sub-findings that survived the clean harness:** (a) low
  linear↔V3.1 correlation +0.05 (real, but a near-uncorrelated *and
  unprofitable* sleeve cannot accrete — proven by in-sample UB==V3.1);
  (b) magnitude is mildly predictable (IC +0.12) but directional IC ≈0
  ⇒ unmonetizable (flaw #8 tested, negative); (c) DDI short-side alpha
  is real as a cohort IC spread but **portfolio-invisible under 4h
  cost** (T1 short-only negative gross of cost).
- **Production LGBM (V3.1) unaffected throughout** and remains the
  production strategy: honest-forward Sharpe +2.747 on the matched
  1260-cyc folds-3–9 grid (+2.23 headline over all 1620). **CAVEAT
  (verified 2026-05-19, owner-probed):** that +2.23/+2.747 is NOT broad
  alpha — it is single-name concentrated: **~62% of net PnL from
  VVVUSDT alone** (a low-float meme the linear harness *excludes* as
  degenerate), 83% from top-3 names; ex-VVV reconstructed Sharpe
  +3.40→+1.87. So (i) §4's "+2.747 bar" was VVV-inflated and the
  linear-vs-V3.1 comparison was genuinely not same-universe (still does
  not rescue the linear line — it failed on its own universe); (ii) the
  production number is fragile/concentrated, broad component ≈+1.0–1.5.
  Detail: memory `project_vBTC_linear_model.md` (concentration entry).

## Honest probabilities (R2, final) & kill status

- P(linear line profitable/accretive **in-scope**, 4h free data) ≈
  **3–5%** (was 15–22% → 7–11% after §4 → 3–5% after §3.5; the residual
  is unknown-unknown, no specific untested in-scope lever remains).
- P(profitable **only if scope expanded** — orthogonal data / longer
  horizon, both out-of-current-scope) ≈ **20–30%**.
- **Pre-registered kill: BOTH legs fired.** §4 SUCCESS-B FAIL (CI incl
  0) + §3.5 NEGATIVE ⇒ §5 forbidden, §3 gated off, §7 N/A. **Line
  closed.** Re-opening requires a scope change (new orthogonal data or
  a longer horizon), to be decided by the user — not a re-run of any
  in-scope variant.

## Process

3-agent loop (R1 methodology / R2 profitability / R3 red-team) reviews
each stage's **results vs pre-stated expectation**; mismatch ⇒
re-initiate. No goalpost-moving. Honest same-day records. This file
updated after every run.
