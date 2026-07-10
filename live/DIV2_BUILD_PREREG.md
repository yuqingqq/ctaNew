# DIV2 crypto-CTA sleeve — build & validation PRE-REGISTRATION (2026-07-10)

Binding pre-registration for building a crypto trend/CTA sleeve as a WITHIN-MANDATE era-diversifier
for v4. Motivation + feasibility evidence: RESEARCH_LOOP addenda 23i–23m (crypto-TSMOM × v4:
corr −0.18, median +90 bps in v4-bad weeks, +24% overall / +48% 2022 matched-vol DD cut, combined
Sharpe +1.55 > v4 +1.27 > trend +0.41; concentration-broad, PIT-clean, reviewer-verified). This
plan commits the tests, gates, and headline BEFORE running, per the estimator-law + W1b (no post-hoc
sweep to rescue a failed gate).

## The binding uncertainty (why this is a validation, not a deploy)

The feasibility result is real but its value rides entirely on the sleeve's OWN forward edge — the
weak link: standalone Sharpe is THIN (+0.41 over choppy 2023-26), nearly all value is crisis-TIMING,
and the +48% 2022 DD cut is ONE sustained-bear episode (H1/H2 are halves of the SAME 2022 bear, not
independent crises). A slow 365d trend also cannot react to a fast V-crash. So the build must clear a
HIGH bar: **validate the sleeve's own edge + its flash-crash weakness FIRST; the diversification
claim is downstream and is inherently single-episode on this data (full crisis-validation is
FORWARD-only).**

## PINNED SLEEVE SPEC (binding — restates 23j; NO re-tuning)

- universe: 20 majors with full 2021-26 coverage (fixed list, div2_crypto_trend.py `MAJORS`).
- signal: canonical time-series momentum = sign(trailing 12-MONTH / 365d return). MOP-2012 default.
- sizing: inverse trailing-30d realized vol, gross-normalized to 1, PIT-shifted (no look-ahead).
- cost: 4.5 bps one-way taker × |Δw| turnover. rebalance: daily.
- combination with v4: PIT trailing inverse-vol weekly weights (as DIV2), matched-vol for DD claims.

**Headline spec is 365d/30d and stays the headline regardless of any robustness-band outcome.**

## Phases, tests, and PRE-COMMITTED gates

### Phase 1 — the sleeve's OWN edge (the weak link). Gate = must pass to proceed.
1a. **Per-period stability.** Split 2021-26 into disjoint half-year sub-periods; report standalone
    net Sharpe per sub-period (2022 halves flagged as crisis).
    - **GATE 1a:** non-2022 (i.e. choppy-regime) sub-periods — sleeve Sharpe ≥ 0 in ≥ 60% of them
      AND aggregate non-2022 Sharpe ≥ 0 (the sleeve must NOT be a drag outside crisis; thin-positive
      is acceptable, negative is not).
1b. **Neighborhood robustness (NOT a sweep-to-pick).** Report Sharpe + v4-bad-week diversification
    sign for lookback ∈ {250, 365, 500}d × vol ∈ {20, 30, 40}d (9 cells). Purpose: confirm the pinned
    365/30 is not a knife-edge. **The pinned cell stays the headline no matter which cell is best.**
    - **GATE 1b:** ≥ 7/9 neighborhood cells show the SAME SIGN of v4-bad-week diversification
      (positive), AND pinned 365/30 standalone Sharpe is within [min, max] of the 9-cell band (not an
      outlier high). If only 365/30 works, the result is knife-edge → FAIL.
1c. **Turnover/cost realism.** Report mean annual turnover + cost drag as % of gross. Sanity only.

### Phase 2 — fast-crash stress (the known weakness). Report + size, no pass/fail.
2a. **Identify fast-V-crash windows** by a pre-defined objective rule: any ISO-week with BTC 4h-close
    return ≤ −15% followed by ≥ 50% retrace within 4 weeks. List the windows found.
2b. Measure the sleeve's PnL and the COMBINED book's PnL vs v4-standalone inside those windows.
    - **Pre-committed read:** quantify the worst fast-crash whipsaw drag. If the combined book is
      WORSE than v4-standalone inside fast-crash windows, that is a material limitation → the sleeve
      must be SIZED so its fast-crash drag cannot exceed a pre-set fraction (≤ 25%) of v4's own
      fast-crash move. (This sizes the weakness; it is not a rescue.)

### Phase 3 — diversification re-confirm (downstream of Phase 1). Gate = the diversification claim.
3a. **Temporal split.** Form the inverse-vol combination on 2023-24, confirm on 2025-26 (weighting is
    parameter-free PIT, so this tests temporal stability of the benefit, not a fit).
    - **GATE 3a:** in the 2025-26 CONFIRMATION window, combined net Sharpe ≥ max(v4, trend) standalone
      AND matched-vol maxDD cut > 0. If the diversification only exists in-form-period → FAIL.
3b. **Single-episode honesty (stated, not gated):** the 2022 crisis DD cut is NOT re-confirmable OOS
    on this data (only one bear). Phase 3 validates the CHOPPY-regime diversification; the CRISIS
    diversification remains a forward claim.

### Phase 4 — build deliverables (ONLY if Phases 1 & 3 gates pass)
- `live/crypto_cta_sleeve.py` — the pinned sleeve as a reusable module (daily positions → PnL).
- combined-book overlay wiring into the paper harness (PIT inverse-vol, matched-vol sizing, the
  Phase-2 fast-crash size cap).
- **Forward protocol** (crisis-validation is forward-only): live sleeve-PnL ledger, and KILL criteria
  pre-committed here — retire the sleeve if (i) standalone rolling-26w Sharpe < −0.5, or (ii) the
  v4-sleeve rolling-26w correlation drifts to > +0.3 for ≥ 8 consecutive weeks (diversification lost).

## What a full PASS means (and does not)
PASS (Phases 1+3 gates + Phase 2 sized) = "a canonical crypto-CTA sleeve has a real, stable,
non-crisis edge that is at worst neutral, is broadly counter-cyclical to v4, and diversifies choppy
regimes OOS; its crisis value is feasibility-shown on 2022 and now on a FORWARD ledger." It does NOT
mean the crisis-diversification is validated (single episode) or that the sleeve is a standalone
strategy (it is thin alone; its role is a diversifying overlay). FAIL of 1a/1b/3a → the sleeve is not
built; the feasibility finding stands as a recorded negative-space result, no deploy.

## Discipline commitments
- No sweep to rescue a failed gate (W1b). The pinned 365/30 headline is binding.
- Book-level / matched-vol metrics (estimator law); no path-coupled overlay replays for the sleeve.
- Every mean reported with its median + concentration (program discipline).
- Feasibility ceiling until a FORWARD crisis is observed.
