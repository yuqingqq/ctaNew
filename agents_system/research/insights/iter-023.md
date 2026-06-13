# iter-023 — cross-sectional MAX / lottery-demand as a SHORT-SIDE overvaluation signal (ONLINE-SOTA)

## STATUS: NO-CANDIDATE (failed the decisive short-side marginal-PnL / G4 pre-check)

## The idea (online-SOTA led, steered to directive headroom (b): the short side)
A coin with a recent **extreme positive return** (high MAX) gets over-bought by
lottery-seeking demand and subsequently **under-performs** — the documented **MAX /
lottery-demand effect**. Because it is an *overvaluation* signal it is intrinsically
**short-side** (high-MAX coins are the ideal shorts), which matches the in-house DDI-2
finding that the short leg carries the real alpha and the long leg is ~beta hedge.

**Mechanism (prior reason to be ERA-STABLE):** lottery preference + limits-to-arbitrage /
short-sale constraints — a behavioral/microstructure effect, not a price-regime. MAX is the
*extreme tail* of recent returns, structurally distinct from `pred` (XS mean-reversion of
the level) and from iter-022's `rel_ret_1d` (the *mean* relative move).

**Citations:**
- "Lottery-like preferences and the MAX effect in the cryptocurrency market", Financial
  Innovation 2021, doi:10.1186/s40854-021-00291-9.
- "Higher moments, extreme returns, and the cross-section of cryptocurrency returns",
  Finance Research Letters 2020, S1544612320303135 (idiosyncratic skewness / MAX negatively
  predict next-period returns).
- Bali, Cakici, Whitelaw (2011, JFE) — the equity MAX origin.
- "Skewness Risk and the Cross-Section of Cryptocurrency Returns", SSRN 4869652 (negative
  skewness–return relation; idiosyncratic).

## Build (PIT, no new data)
MAX = the single largest **trailing 4h return** over the trailing W 4h-blocks. The panel's
`return_pct` is the 48-bar (4h) **forward** return, so the trailing 4h return ending at t is
`return_pct.shift(+48)` per symbol (verified: 4h exit span; trailing/forward corr ≈ 0) — fully
PIT. MAX = rolling-max of that trailing series over W∈{3,6,12} blocks (12h/24h/48h) on the 4h
entry grid (non-overlapping blocks).

## Pre-checks, in fail-fast order (the decisive numbers)

### (ii) G7 transport — PASS (strong, era-stable, beats iter-022's signal)
XS Spearman IC(MAX → fwd 4h alpha-residual), 4h grid:
| W | HL70 | EXT 2021-26 |
|---|---|---|
| 3 | −0.0424 (t−11.6) | −0.0360 (t−15.3) |
| 6 | −0.0454 (t−11.8) | −0.0351 (t−14.7) |
| 12 | −0.0447 (t−11.3) | −0.0369 (t−15.5) |
| (rel_ret_1d ref) | −0.0360 | −0.0302 |

Negative on BOTH universes (where funding/mom180/alt-bear died) and **stronger** than the
iter-022 reversal signal.

### (i) R-marginal IC — PASS at the IC layer (but this is the iter-022 trap)
- IC on **PRED-residualized** fwd alpha_A (HL70): max_6 raw −0.0445 → **pred-resid −0.0416**
  (pred is a weak +0.0056 predictor so it absorbs almost nothing — same as iter-022).
- IC on **rel_ret_1d-residualized** fwd alpha-resid: MAX retains most of it
  (max_12 −0.0447 → **−0.0388**; max_6 → −0.0327) and corr(MAX, rel_ret_1d) only +0.18–0.43,
  so MAX is **genuinely distinct** from the iter-022 reversal (the extreme tail, not the mean).

### (iii) DECISIVE short-side marginal-PnL / G4 pre-check — **FAIL**
The iter-022 lesson: pred-residualized IC is necessary, NOT sufficient — it must add GROSS PnL
*through the held-book construction*. Test: within the short-eligible pool (bottom-half by
pred, sideways), does tilting toward high-MAX pick more-negative-alpha shorts than a
matched-random-K from the SAME pool (200 seeds)?

| W | SHORT highest-MAX | rank vs random-pool | LONG lowest-MAX | rank |
|---|---|---|---|---|
| 3 | +2.12 bps/cyc | **p56** | −0.38 | p72 |
| 6 | +1.39 bps/cyc | **p34** | −2.23 | p28 |
| 12 | −1.09 bps/cyc | **p2** | +0.36 | p90 |

Never ≥p95; sign-unstable across W; at W=12 random shorts beat MAX (p2). The blend
`z(pred)−z(MAX)` ranked **p22**. The production-pred short (+0.45 bps) and the MAX-tilt are
BOTH beaten by random selection from the pool (rand mean +1.9 bps).

## Why it fails (mechanism — the third wall, confirmed for a 3rd signal)
The MAX IC is real cross-sectionally but lives in the **long left tail** of low-MAX bouncers
and is diffuse; **within the pred-conditioned short pool it carries no separable, monetizable
edge** over random selection of equal size. Signal-orthogonality and even pred/rel-residualized
IC do NOT imply marginal contribution given the held-book — exactly the iter-022 finding,
now triple-confirmed (rel_ret_1d i022, MAX i023). The held-book's discrete top/bottom-K
selection on a pred-conditioned pool destroys diffuse cross-sectional IC.

## Verdict
**NO-CANDIDATE — do not build.** Honest fail at the decisive (i)/G4 marginal-PnL layer before
any engine change. The transport + IC + pred-residualized pre-checks all PASS, which is itself
the lesson: those three are insufficient gates; the marginal-PnL-given-construction pre-check
is the one that bites.

## Lesson for the loop (sharpen the R-marginal rule)
**R-marginal must be measured at the CONSTRUCTION layer, not the IC layer.** Add to the
standing pre-check: before proposing a selection-tilt, test it against a matched-random pick
*from the same conditioned pool* the held-book actually selects from (not the full universe) —
diffuse XS IC that survives pred-residualization can still rank < p50 there. This is cheaper
and more decisive than IC, and would have killed iter-022 a layer earlier.

## Scripts
- `research/convexity_portable_2026-05-20/scripts/iter023_max_lottery_precheck.py` (IC + transport + pred-resid)
- `research/convexity_portable_2026-05-20/scripts/iter023_max_resid_on_rel.py` (distinct-from-rel_ret_1d)
- `research/convexity_portable_2026-05-20/scripts/iter023_shortside_pnl_precheck.py` (short-side G4 — the decisive fail)
- `research/convexity_portable_2026-05-20/scripts/iter023_shortside_robust.py` (W-sweep, both sides)

## Sources
- https://link.springer.com/article/10.1186/s40854-021-00291-9 (MAX effect, crypto)
- https://www.sciencedirect.com/science/article/abs/pii/S1544612320303135 (higher moments / extreme returns)
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4869652 (skewness risk, crypto cross-section)
