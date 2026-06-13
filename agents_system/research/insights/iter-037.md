# iter-037 — New-listing dynamics: is there a separate tradable sleeve? — NO-CANDIDATE

**Scope (human idea):** new listings have so far only been EXCLUDED (the maturity≥180d filter,
iter-032/035/036) because the cross-sectional model is blind to their short history. Question:
is there a *distinct, tradable edge in new-listing DYNAMICS* that justifies a SEPARATE sleeve
(event-time, not in the XS book)? Bounded study: characterize ~50+ listing events, test 3
candidate strategies, validate honestly (cohort-transport / few-events CI / cost / disguised-trend).

## Data
- `xs_feats_*.parquet` (5m OHLCV, 218 syms, 2021–26). Listing date = first non-NaN close.
  **Backfill caveat:** ~44 files start exactly at the panel epoch (2023-01-01) — those are NOT
  listings (e.g. BTC/OP/INJ show 2023-01-01). True new-listing events = first close **≥2023-02-01**
  → **164 events** (28 in 2023, 53 in 2024, 78 in 2025, 5 in 2026); 163 with ≥31d history.
- Funding `funding_*.parquet`: mostly backfilled only from **2025-01-01**, so funding-from-listing
  exists only for the **late-2024 / 2025 cohort (146 events)**.
- Conservative cost for thin new perps: **15 bps/leg (30 bps RT)**; cost-swept 5/10/15/20.

## Literature (cited)
- **Long-run new-token / pump underperformance** — Clough & Edwards 2023, *Pump, Dump, and then What?*
  (arXiv:2309.06608, 765 coins): pumped/new tokens average **~−30% one year out** → motivates a FADE.
- **Extreme positive perp funding = crowded longs → reversals** (Coinbase/Phemex funding primers,
  arXiv:2506.08573 funding-as-crowding) → motivates funding-carry SHORT on new perps.

## STEP 2 — Early-life characterization (n=163 events, return measured from first hour)

| horizon | mean | median | %neg |
|---|---|---|---|
| 1d  | +0.017 | **−0.038** | 59% |
| 3d  | +0.030 | **−0.028** | 55% |
| 7d  | +0.042 | **−0.091** | 61% |
| 14d | +0.056 | **−0.136** | 64% |
| 30d | +0.080 | **−0.187** | 64% |

- rv_7d median **263% annualized** (extreme vol); median first-7d **max-run-up +19% then max-DD −20%**.
- **The signature is a FADE: median return is NEGATIVE and monotonically worsens with horizon
  (−3.8%→−18.7%), 64% of listings are down 30d out** — textbook new-token underperformance / listing
  pump-and-fade. BUT the **MEAN is POSITIVE** (a handful of 2–5× moonshots drag it up): the
  distribution is right-skewed with a fat positive tail.

**Cohort medians (the fade is in ALL years, not one regime):**
- 2023 (n=28): 1d +0.02 → 30d −0.12
- 2024 (n=53): 1d −0.05 → 30d −0.02
- 2025 (n=78): 1d −0.03 → 30d **−0.35**
Descriptively the fade is robust across cohorts (strongest in 2025).

## STEP 3 — Candidate strategies (event-pooled, net 15 bps/leg)

| strategy | n | meanPnL | medianPnL | hit | t | boot CI95 |
|---|---|---|---|---|---|---|
| (a) fade short@3d→7d   | 163 | +0.007 | +0.067 | 63% | +0.28 | [−0.041,+0.052] |
| (a) fade short@3d→14d  | 163 | **−0.002** | +0.109 | 64% | −0.05 | [−0.098,+0.079] |
| (a) fade short@7d→30d  | 163 | −0.074 | +0.128 | 66% | −0.85 | [−0.268,+0.072] |
| (a2) fade short@3d→14d if pump>+10% | 46 | −0.031 | +0.107 | 67% | −0.26 | [−0.297,+0.156] |
| (c) momentum long@3d→14d | 163 | −0.004 | −0.115 | 36% | −0.08 | [−0.085,+0.092] |
| (c) momentum long@7d→30d | 163 | +0.068 | −0.134 | 34% | +0.78 | [−0.078,+0.262] |
| **(b) funding-short top-half @3d→14d** | 73 | **+0.074** | +0.186 | 68% | **+1.42** | [−0.033,+0.168] |

**The trap is the same for every variant: high HIT-RATE + strongly-positive MEDIAN, but MEAN ≈ 0
with a |t| < 1.5 and a bootstrap CI that crosses zero.** The fade wins the *body* of the
distribution but a naked short eats the unbounded right tail.

**Moonshot-tail decomposition (short@3d→14d):** 8% of events 2–5× in two weeks (AVNT +475%, IP +249%,
GAS +231%, TIA +153%, FARTCOIN +119%). Short mean **with** tail = −0.0024; **excluding top-5% pumps =
+0.1003**. One 5× pump wipes ~100 winning 1% shorts. The fade edge is real on the body, **uncapturable
naked** because the loss tail is unbounded (you can't size-cap a short against a 5×).

## STEP 4 — Honest validation (the decisive part)

**(i) Cohort transport — FAILS (sign-flips).** Best candidate (funding-short top-half @3d→14d) by year:

| cohort | n | meanPnL | t |
|---|---|---|---|
| 2023 | 10 | **−0.268** | −1.10 |
| 2024 | 24 | +0.028 | +0.36 |
| 2025 | 37 | **+0.187** | **+3.99** |

The entire positive pooled result is the **2025 cohort**; it LOSES in 2023 and is flat in 2024.
This is the run's **universe/regime-overfit wall** (same as funding-alpha iter-021, mom_180d iter-015,
alt-bear gate iter-007): the funding-short "works" only in 2025's alt-bear and reverses in 2023's
alt-bull. The plain fade transports descriptively but not as PnL (the moonshot tail is regime-timed).

**(ii) Few-events CI — FAILS.** Even pooling all cohorts, funding-short event-bootstrap
meanPnL +0.0736, **CI95 [−0.036, +0.166], P(mean>0)=92%** — below the 95% bar, and the pooled positive
is a 2025-regime artifact. ~50–150 events is too thin to clear the bar once cohorts disagree.

**(iii) Cost sensitivity — NOT the binding constraint.** short@3d→14d meanPnL: −0.0004 @5bps →
−0.0024 @15bps → −0.0034 @20bps. The edge is ≈0 even at 5 bps; **cost isn't what kills it — the
zero mean / fat tail does.** (So the "needs <5bps" untradeability test is moot; it fails at 0 bps too.)

**(iv) Disguised rejected family? — YES.** The only positive-leaning candidate (funding-short) is a
**net-directional SHORT on freshly-listed alts** that pays off in the alt-bear (2025) and loses in the
alt-bull (2023) — structurally identical to the already-rejected **net-short-the-alt-bear (iter-008,
NO-CANDIDATE, squeezed by the bounce)** and the **directional/trend family** (iter-008/010/017). The
fade is a directional bet wearing an event-study costume.

## STEP 5 — Verdict: NO-CANDIDATE (new-listing effects are real descriptively, not robustly/cheaply tradable)

New listings ARE special — a real, literature-consistent **FADE** (median −19% at 30d, 64% negative,
present in every cohort) with extreme vol. But it is **NOT a tradable sleeve**:
1. Naked short has **zero mean** despite 64% hit — the **unbounded moonshot tail** (8% of names 2–5×)
   eats the body edge; can't size-cap a short against a 5×.
2. The one variant with a positive lean (funding-short) **sign-flips across cohorts** (2023 −0.27 /
   2025 +0.19) — a 2025-regime artifact, fails transport.
3. **Few-events CI crosses zero** (P(mean>0)=92% < 95%); ~50–150 events too thin to certify.
4. It's the **already-rejected directional/net-short family** in disguise (iter-008/010/017).
5. Cost isn't even the binding limit — the mean is ≈0 at 0 bps.

**New listings stay EXCLUDED via the maturity≥180d filter (iter-032/035/036).** That filter is the
correct treatment: the early-life dynamics are a fat-tailed directional gamble, not a stable
cross-sectional or event edge. The maturity filter already captures the only honest decision here
(don't trade them until seasoned).

**Thin-data + cost honesty:** ~50–164 events, half lack funding-from-listing, the signal is dominated
by an 8% moonshot tail and a single (2025) cohort. No amount of cost-tuning rescues a zero mean with a
sign-flipping cohort split. If pursued further it would need (a) a *capped-loss* instrument (options /
defined-risk) to harvest the fade body without the moonshot tail — out of free-perp scope; (b) many more
listing events across multiple full bull/bear cycles to certify cohort transport — not available.

Scripts: `agents_system/research/scripts/iter037_newlisting.py` (characterization),
`iter037_strategies.py` (3 strategies + transport + cost), and the inline cohort/tail/bootstrap probe.
