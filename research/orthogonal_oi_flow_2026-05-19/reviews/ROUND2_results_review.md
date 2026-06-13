# OI/flow Results Review (2026-05-19) — RE-INITIATE

Verdicts: Methodology **RESULTS-NEED-REWORK** · Profitability **GOAL-MET** ·
Red-team **RE-INITIATE**. Convergent decisive findings:

1. **FATAL implementation bug (red-team, methodology confirm):**
   `oi_flow_test.py` `a0.join(ar, how="inner")` joins on a NON-UNIQUE `time`
   index across the 5 disjoint groups → cartesian product. Same class as the
   B3-flagged duplicate-timestamp inflation (~25×). Effects: (a) n_eff
   1309/3681 and MDE 3.6–6.3 bps are ~25× overstated — must NOT be quoted as
   precision; (b) cross-pairs A0(group i) vs ARM(group j) → NOT lockstep,
   corrupts every Δ/CI; (c) reported A0/ARM Sharpes are a corrupted pooled-
   replicated statistic, NOT the R3c group-disjoint portable number, so they
   cannot be compared to B3's −0.33.
2. **Parity failure:** PLAN guards (per-group level-CI, LOFO sign-flip, top-K
   spread, covered-subset, R0 prefix-causal) were promised but NEVER coded.
   The repeated **Ridge-OI positive** (A1_OI|ridge Δ+0.369, A3_OIFLOW|ridge
   A0+0.645) was dismissed via the corrupted CI — while the equivalent
   bottleneck Ridge-A0 +0.56 thread got a proper level-CI/LOFO/top-K check
   (`ridge_a0_check.json`). Honesty demands the same scrutiny here.
3. LGBM arms nondeterministic ±0.05 > some reported Δ → LGBM deltas
   non-informative without seed-lock or a stated band. Ridge arms
   deterministic (reproduce exactly) → the Ridge-OI positive is the only
   *stable* signal and is the one that was waved away.
4. Conservative-survival (methodology + profitability): the qualitative "no
   detectable portable lift" likely holds because the bug widens CIs / cuts
   toward uncertainty; leak guards genuinely clean (|rankIC| 0.032/0.018;
   coverage admissibility 93–98%, NOT null-by-construction); but the headline
   must NOT be asserted on the corrupted run.

## Action (per discipline: re-initiate the offending test, gate unchanged)
Fix harness: pair A0/ARM WITHIN each disjoint group (group key; mean-of-per-
group portable Sharpe + properly-paired per-cycle diff, honest n_eff =
cycles/BLOCK). Add the promised guards: per-group level block-bootstrap CI,
LOFO single-group sign-flip, top-K realized-alpha spread — applied to all
arms, with explicit Ridge-OI resolution at B3 `ridge_a0_check` parity.
Persist per-group artifacts so re-analysis is free. One corrected run (skip
redundant stage0/full split — data identical). No synthesis / no menu-closure
until corrected. Agent ids: meth `a68615bee8fdb96fe`, prof `a4119b748eebc6efc`,
red `a2eec0618df0b7dba`.
