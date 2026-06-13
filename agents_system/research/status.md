# research status
iteration: iter-040
state: done
summary: NO-CANDIDATE — tested the human's lead REGIME-GATED new-listing short (short the fade ONLY in
  alt-BEAR where moonshots are rare, FLAT in alt-BULL where moonshots cluster). First iter of the
  NEW-LISTING loop. Same 163 listing events as iter-037/039 (5m->1h OHLC, cost 15bps/leg, iter-039
  realistic stop+gap fill). Regime axis = alt-index trailing-30d cum return (iter-006/007 def,
  equal-weight seasoned universe, .shift(1)-lagged) -> FORWARD-CLASSIFIABLE PIT CONFIRMED.
  STEP 2 MECHANISM REAL: gating to alt30<-0.10 flips ~0-mean short to +0.114 (stop+30%, P>0=98%) /
  +0.164 (naked, P>0=96%); UNGATED stop+30% only +0.035/P>0=87%; INVERSE (short in alt-bull) is
  catastrophic -0.219/P>0=1% (the moonshots that eat the short ARE in alt-bull). Human's diagnosis
  confirmed: moonshot frequency is regime-driven.
  STEP 3 DECISIVE: (a) PER-BEAR-EPISODE -- data spans 16 alt-bear episodes but listings populate ~10,
  and ep12 (Jan-Apr 2025 alt-bear) ALONE = +5.149 of +8.227 stop-PnL (63%; 67% naked) and holds 50 of 69
  bear events (73%). Gate largely re-selects the single 2025-Q1 cohort iter-037/039 already isolated.
  (b) LEAVE-DOMINANT-EPISODE-OUT: dropping ep12, residual 19 events STILL lean positive (stop30 +0.142
  P>0=97%; naked +0.203 P>0=100%) -> NOT a pure 1-episode artifact, but 19 events scattered <=4 per
  micro-window can't certify forward. (c) PLACEBO: PASSES random-matched-subset (real p99 vs p95) but
  FAILS honest circular-rotation regime placebo (p90 < p95) -- sliding an autocorrelated regime mask
  reproduces it ~10% of the time; result is mostly "be short during 2025-Q1" not "detect bears".
  (d) CI: bootstrap +0.114 CI95[+0.013,+0.241] P>0=99% but ep12-dominated. Cost not binding.
  VERDICT: the alt-bear gate is mechanically CORRECT physics (moonshots rarer in bear, fade-short
  profits) but the new-listing dataset spans too few POPULATED bear regimes to distinguish a forward
  regime edge from a 2025-Q1 bet. 1-episode bet in a regime-gate costume. NO sleeve. New listings STAY
  EXCLUDED via maturity>=180d filter (iter-032/035/036). Champion + universe standard UNCHANGED. The
  LOEO residual (+0.14, n=19) is the only future thread if more bear-cohort listings accumulate.
blockers: none. Champion + universe standard UNCHANGED.
scripts: agents_system/research/scripts/iter040_regime_gated_short.py ; iter040_episodes_placebo.py
  (+ iter040_events_gated.parquet)
insight: agents_system/research/insights/iter-040.md ; handoff: NO-CANDIDATE / regime-gated-newlisting-short
