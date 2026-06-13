# Research handoff — iter-040

## STATUS: NO-CANDIDATE (regime-gated new-listing short) — the alt-bear gate WORKS mechanically (moonshots are rarer in bear, fade-short profits) but the new-listing data span too few populated bear episodes to certify it forward; the gated PnL is 63-67% one 2025-Q1 alt-bear (50 of 69 events). It's a 1-episode bet wearing a regime-gate costume. New listings stay EXCLUDED via the maturity≥180d filter.

iter-040 (first iter of the NEW-LISTING loop) tested the human's lead hypothesis: short the new-listing
fade ONLY in alt-BEAR (moonshots rare → fade dominates), FLAT in alt-BULL (moonshots cluster → eat the
short). Same 163 listing events as iter-037/039 (5m→1h OHLC, cost 15 bps/leg, iter-039 realistic stop+gap).

## Regime axis (PIT, forward-classifiable — CONFIRMED)
Alt-index = equal-weight trailing-30d cum return over the seasoned universe, `.shift(1)`-lagged
(iter-006/007 definition). Entry at day-3 reads the most-recent alt30 at-or-before entry date. No
look-ahead. Gate: short only if alt30 < X; sweep X ∈ {0, −0.10, −0.20}.

## STEP 2 — mechanism is REAL (gated >> ungated; inverse catastrophic)
| config | n | mean | hit | P(>0) |
|---|---|---|---|---|
| UNGATED naked short | 163 | −0.030 | 66% | 32% |
| UNGATED stop+30% | 163 | +0.035 | 50% | 87% |
| **GATED stop+30%, alt30<−0.10** | 69 | **+0.114** | 57% | **98%** |
| GATED naked, alt30<−0.10 | 69 | +0.164 | 77% | 96% |
| **INVERSE naked, alt30≥0.0 (sanity)** | 71 | **−0.219** | 59% | **1%** |
Gating to alt-bear flips a ≈0-mean short to +0.11/+0.16; the inverse (short in alt-bull) is a disaster
(P>0=1%) — the moonshots that kill the short ARE in alt-bull. Human's diagnosis confirmed cleanly.

## STEP 3 — DECISIVE tests
- **PER-BEAR-EPISODE (central):** data spans 16 alt-bear episodes but listings populate ~10, and
  **ep12 (Jan-Apr 2025 alt-bear) = +5.149 of +8.227 stop-PnL (63%; 67% naked) and holds 50 of 69 bear
  events (73%).** The gate largely re-selects the single 2025-Q1 cohort iter-037/039 already isolated.
- **Leave-dominant-episode-out:** dropping ep12, the residual 19 events still lean positive (stop30
  +0.142 P>0=97%; naked +0.203 P>0=100%) — so NOT a pure 1-episode artifact, the fade-in-bear leaves a
  trace elsewhere — BUT 19 events scattered ≤4 per micro-window can't certify a forward sleeve.
- **PLACEBO:** PASSES random-matched-subset (real p99 vs placebo p95) but **FAILS the honest circular-
  rotation regime placebo (p90 < p95)** — sliding an autocorrelated regime mask reproduces the result
  ~10% of the time; the result is mostly "be short during 2025-Q1," not "detect bears."
- **CI:** thin-event bootstrap +0.114, CI95 [+0.013, +0.241], P(>0)=99% — passes but is ep12-dominated.
- Cost not binding (mean ≈0 ungated even at low cost, per iter-037/039).

## Recommendation
NO new sleeve. **New listings stay EXCLUDED via the maturity≥180d filter (iter-032/035/036).** Champion +
universe standard UNCHANGED. The regime gate is mechanically correct physics (moonshots ARE rarer in
alt-bear) but the new-listing dataset spans too few populated bear regimes to distinguish a forward
regime edge from a 2025-Q1 bet. To ever certify it: many more listings across ≥3-4 distinct, well-
populated alt-bear episodes (a full extra bear cycle's worth) — not available on free perp data. The
LOEO residual (+0.14, n=19) is the only thread worth a future re-look if/when more bear-cohort listings
accumulate.

scripts: agents_system/research/scripts/iter040_regime_gated_short.py ; iter040_episodes_placebo.py
  (+ iter040_events_gated.parquet)
insight: agents_system/research/insights/iter-040.md
