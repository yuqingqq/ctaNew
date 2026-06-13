# Portable-Alpha Research Plan — 2026-05-19

## Goal (fixed, non-negotiable)

Build a **profitable** crypto alpha-residual system that is **universe-portable**
(robust to the symbol set) and **not dependent on single-name vol concentration**.
Either deliver such a system with an honest positive Sharpe whose CI excludes
zero on out-of-universe + out-of-time data, or produce a decisive, honest
negative that tells the user where to spend effort next (scope change), with
no goalpost-moving.

## Diagnosis (from code + actual research records, not stale docs)

The reported production "+2.23 Sharpe" is not a deployable broad edge:

1. **`sym_id` memorization.** Dropping `sym_id` (a numeric *alphabetical rank*
   of the symbol name) barely moves per-cycle IC (+0.0235→+0.0230) but
   collapses V3.1 Sharpe **+2.23 → −0.39** (`docs/vBTC_STRATEGY_STATUS.md`
   Test 3). The edge is symbol-identity memorization → this *is* the
   non-portability.
2. **Single-name concentration.** ~62% of net PnL from VVVUSDT alone, 83%
   from top-3; ex-VVV Sharpe +3.40→+1.87 (`linear_model/composite_study/
   PROGRESS.md`, owner-probed 2026-05-19). Vol amplification on a low-float
   meme, not broad skill.
3. **IC selector is noise** (S/N=0.32 at the rank-15/16 cutoff); only the
   *size cap* does work, not the ranking.
4. **Linear β-residual line is CLOSED** (3-agent loop, P≈3-5% in-scope) — not
   re-run here.
5. **PIT smells** (`dom_btc_change_288b` unshifted after `.diff()`, `obv_z_1d`
   current-bar, beta-shift 1 vs 49 inconsistency between 51/111 panels) — must
   be controlled or the "edge" may be partly leak.

The per-cycle IC (~+0.023) is real but tiny. The open question the headline
tests never answered: **does a portable edge exist when measured correctly —
no `sym_id`, no per-universe selector, concentration-controlled, evaluated on
a DISJOINT symbol set?**

## Method principles (locked)

- **Pre-registered.** Each test states its hypothesis, the exact metric, and
  the pass/fail gate *before* running. No moving goalposts. Failures recorded
  same-day.
- **Honest evaluation = out-of-universe AND out-of-time.** The novel,
  never-done-as-headline evaluation: **group-disjoint-by-symbol CV** — train
  on a subset of symbols, predict a *held-out symbol group never seen in
  training*, while also respecting time order. Universe portability cannot be
  claimed any other way.
- **Concentration gate.** Every reported Sharpe is accompanied by per-name PnL
  attribution. A result where top-1 name ≥ 30% of net PnL (or top-3 ≥ 55%)
  FAILS regardless of headline Sharpe — it is not a broad edge.
- **Vol-normalized sizing** is the default (equal-risk per leg) so a single
  high-vol meme cannot manufacture the Sharpe. Raw-notional reported as a
  secondary diagnostic only.
- **CI required.** Block-bootstrap (block = 1 fold) CI on Sharpe; lower bound
  must exceed 0 to be a "pass". Paired block-bootstrap for lift vs baseline.
- **No `sym_id`, no universe-dependent features** (no basket/xs-rank features
  whose value depends on which symbols are in the panel) in any portable
  candidate. Features restricted to BTC-relative + own-symbol + funding.
- **Leak control.** Use the strict/full-PIT panel and re-verify the three
  flagged PIT smells are absent (or rebuild the affected columns) before any
  IC/Sharpe is trusted.

## Tests (pre-registered)

### T0 — Honest baseline reconstruction (measurement, not a strategy)
Rebuild the production-style L/S stack on the 51-panel **but**: (a) vol-normalized
equal-risk sizing; (b) full per-name PnL attribution; (c) a variant with VVV
+ any flagged low-float/extreme-vol name excluded *from construction*, not
post-hoc. No new model — uses existing production predictions.
- **Hypothesis:** the true broad, concentration-controlled Sharpe is materially
  below the +2.23 headline (expect ≈ +1.0 or lower, possibly ≤0).
- **Gate:** none (this defines the honest bar T1/T2 must beat). Output: the
  real baseline Sharpe + CI + attribution table.

### T1 — Portable model under group-disjoint-by-symbol CV (DECISIVE)
Train a universe-invariant model: BTC-relative + own-symbol + funding features
only, **no `sym_id`**, target = clean BTC-residual (`target_beta_btc` /
`alpha_beta`). Evaluation: partition the 51 symbols into G symbol-groups; for
each group, train on the other G−1 groups (time-ordered, embargoed, label-purged)
and predict the held-out group's *future* rows only. Construction: fixed-size
universe cap (no IC ranking), vol-normalized K-per-side, conv-style dispersion
gate allowed (it is universe-agnostic).
- **Hypothesis:** if a portable edge exists, out-of-universe + out-of-time
  Sharpe CI lower bound > 0 with top-1 name < 30% of PnL.
- **Pass:** block-bootstrap Sharpe CI lower bound **> 0** AND top-1 ≤ 30% /
  top-3 ≤ 55% of net PnL AND ≥ 5/9 time-folds positive.
- **Fail action:** the 4h cross-sectional residual edge is **not portable** —
  rigorously confirmed with the correct sym_id-free, disjoint-universe
  evaluation that was never the headline. Proceed to T2.

### T2 — Longer-horizon cost-amortization (only if T1 fails)
The one mechanism with structural (untuned) support is the production sleeve's
cost amortization at longer effective hold. Re-test it **with the T1 portable
construction** (no `sym_id`, vol-normalized, group-disjoint eval): overlapping
sleeves giving effective holds {24h, 48h, 72h}, equal sleeve weights only (no
tuned decay — tuned weights failed nested-OOS historically).
- **Hypothesis:** longer hold amortizes cost enough to surface the tiny IC
  into a portable positive Sharpe.
- **Pass:** same CI>0 + concentration + 5/9-fold gates as T1, AND paired
  block-bootstrap lift over T0 baseline with CI excluding 0.
- **Fail action:** decisive honest negative for 4h-free-data *and*
  longer-horizon-free-data portable alpha.

### T3 — Synthesis & decision
- If T1 or T2 passes: we have a portable, concentration-robust core. Deliver
  it + a deployment-hardening note (the live paper bot currently ships the
  K=4 no-sleeve artifact, *not* the claimed production stack — flag this).
- If both fail: decisive negative. Recommend the only un-refuted lever
  (orthogonal data, e.g. on-chain) and explicitly recommend **stopping
  construction tweaks** (prior sessions prove that path only finds
  overfit local optima). Do not re-run closed lines.

## Review process (per user directive)

1. This plan reviewed by **3 independent agents** (methodology / profitability-
   alignment / red-team) before any test runs. Misalignment → revise plan.
2. After tests, results reviewed by **3 independent agents** vs the
   pre-registered gates. If results don't align with the goal or a gate was
   fudged → re-initiate the offending test (not move the goalpost).

## Out of scope (do not re-litigate)

Linear β-residual line (closed); per-universe IC-selector tuning; construction
micro-tweaks (gates/K/N grid search) on the 51-panel; sym_id encodings;
Glassnode/orthogonal-data acquisition (recommendation only, not executed here).
