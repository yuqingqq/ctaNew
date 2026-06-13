# iter-024 — REACTIVE RISK-TRACK: position-level + drawdown-duration mechanisms on top of iter-012

## STATUS: NO-CANDIDATE (honest negative)

## Steer (orchestrator directive)
Alpha-overlay family comprehensively walled (3 mechanisms). Steer to the REACTIVE RISK-TRACK: a
genuinely-DIFFERENT risk mechanism layered ON TOP of the adopted iter-012 vol-norm equity-DD stop that
improves the risk profile **incrementally** beyond what the stop already does. iter-013 already showed
threshold/vol efficiency variants (hysteresis/graded/cooldown) don't Pareto-improve — so a new idea must
be a structurally different MECHANISM, not another knob on the depth-stop.

Picked TWO genuinely-different mechanisms (both with a mechanism + prior reason + online citation),
layered each on the iter-012 depth-stop, measured the INCREMENTAL effect across HL70+EXT+S44, and ran the
decisive construction-layer pre-check (vs random / vs constant-de-gross of equal exposure).

## Online research (citations)
- Position-level vs portfolio-level / selective de-leverage of worst contributors + fractional risk
  sizing; "death by a thousand cuts" caution on mechanical drawdown rules:
  - Samir Varma, "The Stop-Loss That Stops Gains" (Medium, 2024) — mechanical drawdown rules often
    enlarge cumulative loss vs riding the single drawdown.
  - Bayesian drawdown-distribution stop construction — arXiv:1609.00869.
  - Momentum crashes are reversals from bear markets w/ negative beta; vol-scaling exposure tames the
    worst crashes (Barroso-Santa-Clara; alphaarchitect "Minimizing momentum crashes").
- Drawdown DURATION vs DEPTH as DISTINCT risk axes (prolonged shallow grind != sharp deep dip);
  time-based de-risking as a structurally different trigger:
  - tradingmetrics / tradersyard / journalplus drawdown guides; Lévy-model drawdown
    magnitude-vs-duration asymptotics arXiv:1506.08408.

## Mechanism A — POSITION-LEVEL worst-leg stop (vs book-level de-gross)
**Idea/prior:** iter-006 root-cause says the DD is the LONG leg of high-beta fallen alts in a correlated
alt-bear. If the bleeding legs are persistent + identifiable, cutting THEM specifically should cap the
same tail with LESS total exposure removed (better Calmar) than the uniform book-level de-gross. Drop the
held-book legs whose OWN trailing realized contribution (PIT through t-1, trailing-180) is in the worst
q-tile, layered on iter-012.
**Decisive construction-layer placebo (AGENT.md):** worst-leg-cut vs RANDOM-leg-cut of equal count
(40 seeds), measured on incremental maxDD AND Calmar vs iter-012.

| univ | cut | maxDD (incDD% vs i012) | Calmar | random-leg p50 Calmar | rank DD / Cal |
|---|---|---|---|---|---|
| HL70 | 20% | −3912 (−3.1%) | +0.79 | +1.43 | p30 / **p10** |
| HL70 | 33% | −3389 (+10.7%) | −0.05 | +0.68 | p62 / **p5** |
| EXT | 20% | −3764 (−25.5%) | +0.47 | +0.04 | p95 / p100 |
| EXT | 33% | −5534 (−84.4%) | +0.23 | −0.08 | p65 / p100 |
| S44 | 20% | −2901 (+12.3%) | +0.64 | +1.32 | p70 / **p22** |
| S44 | 33% | −3993 (−20.8%) | +0.27 | +0.34 | **p15** / p40 |

**VERDICT: FAIL.** Calmar COLLAPSES vs iter-012 everywhere (+2.01→+0.79 HL70; +2.36→+0.64 S44) — cutting
legs destroys far more return than DD. The Calmar rank vs random-leg-cut is p5/p10/p22/p40 on HL70+S44
(random BEATS the "worst-leg" pick); EXT's p100 is on a maxDD that is WORSE than iter-012 (incDD −25%/−84%,
i.e. it raised DD). No universe shows worst-leg-cut beating random-leg AND improving incrementally.
**Confirms iter-006:** in a correlated alt-bear the "worst leg" is not forward-separable (per-cycle IC
R²≈0.005) — which leg bled most through t−1 is ~random for which bleeds next; cutting it ≈ random and the
strategy needs all legs for its XS structure. Position-level stop has no skill here.

## Mechanism B — DRAWDOWN-DURATION trigger (depth-orthogonal axis)
**Idea/prior:** the iter-012 stop fires on DD DEPTH (≥k·σ). A duration trigger fires when the book has
been continuously underwater ≥ D bars REGARDLESS of depth — a structurally orthogonal axis (lit: duration
and depth are distinct risk axes; a slow shallow grind the depth-stop misses). Layered on iter-012,
de-gross to g_floor when underwater-run ≥ D.

**(1) Incremental headline (looked promising):** D∈{30,60,90} incDD vs iter-012 +12% to +40% on all 3;
Calmar rose on HL70 (+2.01→+2.87 at D=90) and EXT-fire was 69–95% "not already depth-degrossed"
(orthogonal firing). BUT it fires on **774–3673 cycles** (vs depth-stop's 15–92) → avg gross 0.46–0.60 →
that is removing a LOT of average exposure. So the decisive R4 test:

**(2) DECISIVE R4 (does it beat constant / random de-gross of EQUAL average exposure?):**

| univ | dur book maxDD / Calmar (avgG) | (a) CONSTANT ×avgG maxDD / Calmar | (b) RANDOM matched-%-time rank |
|---|---|---|---|
| HL70 | −2574 / +2.87 (0.60) | **−2259 / +2.01** (better tail) | **p38** |
| EXT | −2535 / +0.63 (0.46) | **−1392 / +0.74** (better tail) | **p1** |
| S44 | −2575 / +2.25 (0.58) | **−1911 / +2.36** (better tail) | **p16** |

**VERDICT: FAIL — strictly dominated.** A CONSTANT de-gross of equal average exposure on the depth-stop
book gives a BETTER (less negative) maxDD on ALL THREE universes at EQUAL-or-higher Calmar (constant
shrink is Calmar-scale-invariant → reproduces iter-012's +2.01/+0.74/+2.36). The "duration Calmar rise"
was an artifact of removing average exposure from an over-levered book; uniform shrink does it cleaner.
The matched-%-time RANDOM placebo BEATS the duration trigger (p38/p1/p16 ≪ p95) on every universe.
**The duration axis, despite firing orthogonally to depth, collapses to ~proportional exposure removal
at the R4 layer — and is worse than just running the iter-012 book at a constant ~0.5 gross.** "Run much
smaller, longer," not skill.

## Why NO-CANDIDATE (the wall, sharpened)
Both genuinely-different reactive mechanisms fail the SAME R4/construction-layer wall the iter-012 stop
itself sits at (its honest caveat: ~proportional, R4 p55–70). Mechanism A: the worst contributors are not
forward-separable in a correlated selloff (iter-006), so position-level ≈ random-leg and destroys return.
Mechanism B: a depth-orthogonal duration trigger STILL reduces to ~proportional de-gross (worse than
constant) because what it removes is undifferentiated exposure, not a selectable tail. **There is no
*selective* tail on free data — every reactive lever (depth iter-012, position-level A, duration B)
collapses to "remove exposure" and is bounded by constant-de-gross.** The iter-012 vol-norm depth-stop
remains the efficient reactive choice (it at least concentrates the de-gross WHEN the DD is real, at far
lower turnover — 15 RT vs 394–3673 fires — and PASSES nested-OOS R6 3/3; A and B add only turnover/return
cost with no incremental tail skill).

## Champion unchanged
BASELINE (HL70 regime-hybrid held-book, Calmar +1.68) + adopted iter-012 vol-norm reactive stop (k=2.0).

## Pre-registered gates (had a candidate emerged — for the record)
R1 PIT (all triggers use realized equity/contrib through t−1, lagged); R2 incremental maxDD cut on HL70;
R4 DECISIVE = beat constant-de-gross of equal avg exposure AND matched-random ≥p95 (BOTH FAILED);
R5 episode-LOFO + S44; R6 nested-OOS of D/q on all 3; R7 re-entry sanity. Failed at R4 pre-check → no build.

## Scripts
- research/convexity_portable_2026-05-20/scripts/iter024_reactive_risk_precheck.py (mechanisms A + B)
- research/convexity_portable_2026-05-20/scripts/iter024_duration_R4.py (decisive R4 kill for B)
- logs: /tmp/iter024.log, /tmp/iter024_r4.log
