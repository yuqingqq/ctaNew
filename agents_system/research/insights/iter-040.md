# iter-040 — REGIME-GATED new-listing short — NO-CANDIDATE (1-episode-dominated; mechanism real, can't certify forward)

**Scope (human lead):** iter-037/039 found new perps FADE (median −18.7%/30d, 64% down) but a naked
short's MEAN ≈ 0 because ~8% MOONSHOT 2-5×, and the moonshot FREQUENCY is regime-driven (2023/24/25
moonshot-rate 21%/23%/9%, mean-ret30 +0.24/+0.16/−0.13). The fix tested here: **GATE BY REGIME** — at
the entry decision read the forward-knowable alt-index trailing-30d return; short the fade ONLY when
the alt complex is in a bear (alt30 < X), stand FLAT in alt-bull. First iter of the NEW-LISTING loop.

## Setup (PIT, reuse of prior infra)
- 163 usable listing events (first close ≥2023-02-01; cohorts 28/53/78/4 for 2023/24/25/26), 5m→1h OHLC.
- **Alt-index regime axis** (iter-006/007 definition): equal-weight mean of per-symbol trailing-30d
  cumulative return over the seasoned universe, **`.shift(1)`-lagged** so it is KNOWN at entry. Entry
  decision at day-3 post-listing reads the most-recent alt30 value at-or-before entry date. **Forward-
  classifiable: CONFIRMED PIT** (trailing-30d, lagged; no look-ahead). alt30 span 2021-02→2026-05,
  53% of days < 0, 35% < −0.10.
- Short PnL: iter-039 realistic model (hold day3→day30, stop+30% with gap-fill `max(trigger, breach-bar
  close)`, and a naked variant), cost 15 bps/leg (RT 30).

## STEP 2 — Gated vs ungated vs inverse (the mechanism WORKS directionally)

| config | n | mean | med | hit | t | boot CI95 | P(>0) |
|---|---|---|---|---|---|---|---|
| UNGATED naked short@3d→30d | 163 | −0.030 | +0.173 | 66% | −0.50 | [−0.158,+0.084] | 32% |
| UNGATED stop+30% | 163 | +0.035 | +0.024 | 50% | +1.16 | [−0.023,+0.094] | 87% |
| **GATED naked, alt30<−0.10** | 69 | **+0.164** | +0.230 | 77% | +1.90 | [−0.025,+0.318] | 96% |
| **GATED stop+30%, alt30<−0.10** | 69 | **+0.114** | +0.173 | 57% | +2.16 | [+0.009,+0.216] | **98%** |
| GATED stop+30%, alt30<0.0 | 92 | +0.081 | +0.110 | 53% | +1.86 | [−0.002,+0.168] | 97% |
| GATED stop+30%, alt30<−0.20 | 29 | +0.071 | +0.101 | 52% | +0.85 | [−0.090,+0.237] | 81% |
| **INVERSE naked, alt30≥0.0** | 71 | **−0.219** | +0.116 | 59% | −2.13 | [−0.432,−0.030] | **1%** |
| INVERSE stop+30%, alt30≥−0.10 | 94 | −0.022 | **−0.303** | 46% | −0.65 | [−0.089,+0.048] | 27% |

**The human's mechanism is REAL and clean.** Gating to alt-bear lifts the mean from ≈0 to +0.11/+0.16
(P>0 up to 98%); the INVERSE (short only in alt-bull) is catastrophic (naked −0.22, P>0=1%; the
moonshots that eat the short ARE in alt-bull). The directional story — moonshots cluster in alt-bull,
fade dominates in alt-bear — holds exactly as iter-037/039 diagnosed. alt30<−0.10 is the sweet spot.

## STEP 3 — The DECISIVE honest tests

### COHORT/REGIME TRANSPORT (central) — the gate is dominated by ONE bear episode
The new-listing data (2023-02→2026) spans **16 distinct alt-bear episodes** by the alt-index, but the
listing events land in only ~10 of them, and overwhelmingly in ONE. Per-bear-episode gated-short PnL
(alt30<−0.10):

| episode | window | n events | mean | sumPnL (stop30) |
|---|---|---|---|---|
| ep7 | 2023-05→06 | 2 | +0.222 | +0.443 |
| ep8 | 2023-08→09 | 1 | +0.187 | +0.187 |
| ep9 | 2024-04→05 | 3 | +0.152 | +0.456 |
| ep10 | 2024-06→07 | 2 | −0.065 | −0.130 |
| ep11 | 2024-08→09 | 2 | +0.181 | +0.361 |
| **ep12** | **2024-12→2025-04** | **50** | **+0.103** | **+5.149** |
| ep13 | 2025-06→07 | 1 | +0.415 | +0.415 |
| ep14 | 2025-10→12 | 4 | +0.478 | +1.913 |
| ep15 | 2026-02→03 | 2 | −0.132 | −0.264 |

**ep12 (Jan–Apr 2025 alt-bear) alone = +5.149 of +8.227 total = 63% of stop-PnL (67% of naked-PnL),
and holds 50 of the 69 bear events (73%).** By calendar month: 50/69 bear events fall in Jan-Apr-2025;
the other 19 are scattered 1-4 per tiny window across 2023-26. **The gate is, mechanically, a re-
selection of the single 2025-Q1 alt-bear** — the exact cohort iter-037/039 already flagged as the only
profitable one. This is the run's recurring universe/regime-overfit wall (iter-007 alt-bear gate net-
HURT on multi-episode ext; iter-037/039 cohort sign-flips).

### LEAVE-DOMINANT-EPISODE-OUT (the one nuance) — residual is positive but unbearably thin
Dropping ep12, the remaining **19** bear events still lean positive (stop30 +0.142 P>0=97%; naked
+0.203 P>0=100%). So it is NOT a *pure* 1-episode artifact — the fade-in-bear effect leaves a positive
trace in the other small windows too. **BUT 19 events spread 1-4 across ~8 micro-episodes cannot certify
a forward regime sleeve** — that's the same thin-event problem iter-037/039 hit, now worse after gating
removes 94 of 163 events. No single non-2025 bear episode has enough events to stand on its own (max n=4).

### PLACEBO
- **Random-matched-subset placebo (k=69 of 163):** real gated mean +0.114 ranks **p99** (placebo
  p95=+0.093, max=+0.157). **PASS** — the bear-selected events genuinely outperform a random 69-subset.
- **Circular-rotation regime placebo** (shift the alt-index ±60-360d vs events, preserving regime
  autocorrelation, recompute the gate): real ranks **p90** (placebo p95=+0.126, max=+0.175). **FAIL.**
  When you keep a *realistically autocorrelated* regime mask and just slide its phase, ~10% of phase-
  shifts hit a window as good as the true alignment — because the listing-event density and the one big
  2025 bear largely determine the result regardless of exact phase. The rotation placebo is the honest
  one here, and it does not clear p95.

### CI — passes pooled, but it's a 1-episode CI
Thin-event bootstrap on the 69 gated events: mean +0.114, **CI95 [+0.013, +0.241], P(>0)=99% — PASS**.
But this CI is dominated by the 50 ep12 events; it certifies "2025-Q1 was profitable," not "alt-bear
gating is a forward edge."

## STEP 4 — Verdict: NO-CANDIDATE

**Does regime-gating make the new-listing short tradable? Mechanically yes; honestly no.**
1. The mechanism is REAL and the human's diagnosis is correct: moonshots cluster in alt-bull, the fade
   dominates in alt-bear; gating to alt30<−0.10 flips a ≈0-mean naked short to +0.11/+0.16 (P>0 96-98%),
   and the inverse is catastrophic (P>0=1%). This is the cleanest confirmation yet that the moonshot
   tail is regime-driven, not name-driven.
2. **But the gated PnL is dominated by ONE alt-bear episode** (Jan-Apr 2025 = 63-67% of PnL, 73% of
   events). The "regime gate" is largely re-selecting the 2025-Q1 artifact that iter-037/039 already
   isolated.
3. **The data span too few real bear episodes WITH listings to certify forward.** 16 alt-bear episodes
   exist, but listings cluster in 2024-25; outside ep12 only 19 events remain, scattered ≤4 per micro-
   window. The residual is positive (P>0=97-100%) but uncertifiable on n=19 across heterogeneous tiny
   windows.
4. **Placebo split:** PASSES the easy random-subset placebo (p99) but **FAILS the honest circular-
   rotation regime placebo (p90 < p95)** — phase-shifting an autocorrelated regime mask reproduces the
   result ~10% of the time, because the result is mostly "be short during 2025-Q1," not "detect bears."
5. Cost is not binding (consistent with iter-037/039: mean ≈0 ungated even at low cost).

**Honest mechanism:** the gate WORKS as physics — moonshots ARE rarer in alt-bear, so the fade-short
profits there — but the new-listing dataset spans **too few bear regimes with enough listings to
distinguish "a forward regime edge" from "a bet on the one 2025-Q1 alt-bear."** It's a 1-episode bet
wearing a regime-gate costume; the LOEO residual hints the edge is broader, but n=19 can't certify it
and the rotation placebo can't reject "phase luck."

**New listings stay EXCLUDED via the maturity≥180d filter (iter-032/035/036).** To certify this gate
forward you would need many more listing events across ≥3-4 *distinct, well-populated* alt-bear
episodes (e.g. a full extra bear cycle's worth of listings) — not available. Champion + universe
standard UNCHANGED.

Scripts: `iter040_regime_gated_short.py` (gated/ungated/inverse + threshold sweep + composition),
`iter040_episodes_placebo.py` (16-episode decomposition + random-subset + circular-rotation placebo +
LOEO + thin-event CI). Data: `iter040_events_gated.parquet`.
