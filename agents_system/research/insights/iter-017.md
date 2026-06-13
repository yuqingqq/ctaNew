# Research Insights — iter-017 (TREND-FOLLOWING / CRISIS-ALPHA HEDGE SLEEVE — SCOPING DIAGNOSTIC, NO-BUILD)

**Human idea:** the −57% DD is a sustained correlated alt-bear that bleeds the mean-rev book. Don't
predict/avoid it (proven impossible: iter-006/008/010) and don't flip the book short (iter-008
squeezed). Instead run a SEPARATE, REACTIVE **trend-following / crisis-alpha** sleeve ALONGSIDE the
book — a diversifying directional engine that aims to PROFIT in the prolonged alt-bear so the
COMBINED portfolio has a smaller drawdown. The honest question is NOT "does it predict" but: **are
the trend sleeve's returns negatively correlated with the book DURING the DD episodes, and does
combining shrink portfolio maxDD net of the sleeve's whipsaw cost in calm markets?**

**Verdict: NO-BUILD.** The crisis-alpha mechanism is *partially* present (the sleeve is mildly
negatively correlated with the book in 3 of 4 episodes and profits hugely in 2022_luna), but it
**whipsaws exactly like the iter-008 net-short** — it profits in 1–2 episodes and LOSES in the
others (including 2025_q4, the episode that *is* the −57% HL70 DD, where it gave back −7,834 bps
because the alt complex bounced and the sleeve was net-short). Decisively, the trend sleeve is a
**high-variance standalone stream** (standalone maxDD −39k to −76k bps vs the book's −4,953), so
adding it at ANY hedge weight makes the **combined portfolio maxDD WORSE on every universe**
(ddVSbook −68% to −983%), and the apparent EXT Calmar "lift" **fails episode-LOFO** (vanishes when
2022_luna is dropped) and **fails the G4 sign-placebo** (the real combined DD is no better than a
random-sign sleeve of equal magnitude — p50/p38). It cannot reduce DD; it inflates it.

Script: `research/convexity_portable_2026-05-20/scripts/iter017_trend_hedge_sleeve.py`. Per-cycle:
`results/iter017_trend_sleeve_{EXT,HL70,S44}.parquet`. Reuses X123's alt-index reconstruction and the
X123 EXT/HL70/S44 `pnl_base` (mean-rev book PnL) + regime/fold/episode tags verbatim.

---

## STEP 2 — the trend sleeve (simple, parameter-light, PIT)

Standard time-series momentum (TSMOM): `sign(trailing-Nd cum log-ret).shift(1)` of (a) the eq-weight
ALT-index (panel alts ex BTC/ETH, the thing the book is exposed to) and (b) BTC. Lookbacks 30d
(WIN=180 bars) and 90d (540 bars) + their sign-average (the standard dual-lookback blend). Position
held HOLD=6 bars, earns the forward-HOLD-bar asset return, cost 4.5 bps on position flips. No tuning
to episodes. Variants: `alt_{30d,90d,avg}`, `btc_{30d,90d,avg}`, `altbtc_avg` (equal blend).

**Standalone (EXT, the multi-episode panel):** every variant is a *viable standalone trend
strategy* (Sharpe +0.8 to +1.5, Calmar +0.2 to +0.6) — TSMOM works on crypto as expected. BUT its
**maxDD is enormous relative to the book**: alt_avg −39,005 / altbtc_avg −26,787 / btc_avg −31,327
bps, vs the book's −4,953. It is a directional gross-1.0 stream; the book is a beta-neutral
gross-1.0 stream split over 10 legs. They are on different risk scales — this is the crux.

---

## STEP 3 — the decisive scoping tests (EXT multi-episode panel)

### TEST 1 — crisis-alpha / negative correlation: PARTIAL, weak, and inconsistent across episodes

corr(trend, book), OVERALL vs WITHIN each DD episode (negative-in-crisis = the win condition):

| variant | overall | 2022_luna | 2022_ftx | 2024_summer | 2025_q4 |
|---|---|---|---|---|---|
| alt_avg | +0.039 | **+0.010** | −0.090 | −0.028 | −0.101 |
| btc_avg | +0.021 | **+0.039** | −0.096 | −0.025 | −0.070 |
| altbtc_avg | +0.036 | **+0.023** | −0.099 | −0.030 | −0.096 |

The sleeve IS mildly negatively correlated with the book in **3 of 4 episodes** (2022_ftx,
2024_summer, 2025_q4: −0.03 to −0.10) — a real, directionally-correct crisis-diversification signal
— but it is **POSITIVELY correlated in 2022_luna** (+0.01 to +0.04) and **all magnitudes are tiny
(|corr| ≤ 0.10)**. A |corr| ≈ 0.1 negative relationship is far too weak to offset the sleeve's own
variance (next test).

### TEST 1b — per-episode trend PnL: the iter-008 WHIPSAW pattern (profits in 1, bleeds the rest)

Per-episode totPnL (bps), book vs trend sleeves @4.5 bps:

| episode | book | alt_avg | btc_avg | altbtc_avg |
|---|---|---|---|---|
| 2022_luna | +756 | **+39,204** | +26,037→… | **+32,621** |
| 2022_ftx | −2,039 | +1,377 | +6,250 | +3,813 |
| 2024_summer | −267 | **−6,660** | −15,512 | **−11,086** |
| 2025_q4 | +4,834 | **−7,834** | +8,463 | +315 |

The alt-sleeve **pays big ONLY in 2022_luna** (the one genuine sustained crash where the down-move
persisted — exactly the single episode iter-008's net-short also won), is marginal in 2022_ftx, and
**LOSES in 2024_summer (−6,660) and 2025_q4 (−7,834)**. This is the *same* "profits in 1 episode,
whipsaws the rest" signature that killed the net-short. (btc_avg pays in 2025_q4 but loses −15,512 in
2024_summer — same whipsaw, different episode.) There is no sleeve that profits across all/most DD
episodes.

### TEST 2 — whipsaw cost in CALM periods: the insurance premium is actually a *credit* here

CALM (non-episode) = 7,743 / 10,291 cycles (75%). The standalone sleeve makes MONEY in calm
(alt_avg +66,004, btc_30d +80,338, altbtc_avg +54,908 bps, calm Sharpe +1.3 to +1.8). So unlike a
classic insurance hedge, this trend sleeve does NOT bleed in calm — TSMOM harvests the long calm
uptrends. **The problem is not the calm-period premium; it is the sleeve's directional VARIANCE
(its own −27k to −76k maxDD), which is what wrecks the combined DD.**

### TEST 3 — combined portfolio book + w·trend: DD gets WORSE at EVERY weight, EVERY universe

EXT (multi-episode), maxDD and Calmar vs book by hedge weight w:

| sleeve | w=0.25 | w=0.50 | w=1.0 | book (w=0) |
|---|---|---|---|---|
| alt_avg DD / ddVSbook | −9,433 / **−90%** | −18,231 / −268% | −35,888 / −625% | −4,953 |
| altbtc_avg DD / ddVSbook | −8,634 / **−74%** | −15,110 / −205% | −28,145 / −468% | −4,953 |
| btc_avg DD / ddVSbook | −8,522 / **−72%** | −15,635 / −216% | −31,235 / −531% | −4,953 |

**Every sleeve at every weight ≥0.25 makes the combined maxDD WORSE** (the objective is to *reduce*
DD). The combined *Sharpe* rises at low w (diversification: EXT 0.87→1.48) and totPnL rises (the
sleeve adds standalone return), but **Calmar's "lift" (EXT alt_avg +0.66→+0.87 at w=0.25) is driven
by the added return outrunning the ballooning DD numerator — it is NOT DD reduction.** On the
**production universe HL70 it is unambiguous: alt_avg DESTROYS Calmar 1.68→0.92(w.25)→0.18(w1)** and
maxDD −5,674→−35,430. The only HL70 "win" is `btc_avg` (1.68→2.28 at w=0.25) — but that is a
**single-universe artifact** (the HL70 window 2025 is one BTC up-trend), it does NOT replicate as
crisis-alpha on EXT/S44, and it raises DD too (−5,674→−5,820).

### TEST 3b — episode-LOFO: the EXT Calmar lift is ONE episode (2022_luna) — FAIL

alt_avg @ best w=0.25, FULL lift +0.20. Dropping each episode:

| drop | book Cal | comb Cal | lift |
|---|---|---|---|
| **2022_luna** | +0.67 | +0.59 | **−0.08 NEG** |
| 2022_ftx | +0.79 | +1.02 | +0.23 |
| 2024_summer | +0.77 | +0.98 | +0.21 |
| 2025_q4 | +0.49 | +0.86 | +0.37 |

**The entire combined lift comes from 2022_luna** — drop it and the lift goes NEGATIVE (−0.08;
altbtc_avg identical, −0.06). The "crisis-alpha" is one crash (the LUNA collapse, the only
genuinely-persistent down-move) carrying the whole result — the classic n=1 / single-episode artifact
that has killed every prior DD attempt (iter-002 fold-6, iter-006 LOFO −0.67, iter-008/010 LOFO).
**Episode-LOFO FAILS.**

### TEST 4 — honest framing vs iter-008/010: it whipsaws, esp. in 2025_q4 (which bounced)

alt_avg per episode: net-short throughout (meanPos −0.41 to −0.82), tot +39,204 (luna) but **−7,834
in 2025_q4 with intra-maxDD −25,665** and **−6,660 in 2024_summer**. In **2025_q4 — the −57% HL70 DD
episode — the sleeve was net-short (−0.41) into a bounce and GAVE BACK −7,834 bps**, the exact
squeeze iter-008's net-short suffered. A trend-follower is reactive to the same trailing price the
slow/fast alt-bear detectors used (iter-008/010): by the time the trailing-30/90d return is negative
enough to flip the sleeve short, the alt complex is at/near a bottom and bounces. **It whipsaws like
the net-short.**

### G4-style sign-placebo — is the trend SIGN load-bearing for DD? NO

Combined book + 1·sleeve, vs 200 random-sign sleeves of identical per-cycle |magnitude|:

| sleeve | real combined maxDD | placebo maxDD p50 / p95(best) | DD rank | Calmar rank |
|---|---|---|---|---|
| alt_avg | −35,888 | −35,938 / −19,412 | **p50 (FAIL)** | p96 |
| altbtc_avg | −28,145 | −24,542 / −14,530 | **p38 (FAIL)** | p96 |

**The real combined maxDD is NO BETTER than a random-sign sleeve of equal magnitude (p50 / p38).**
The trend SIGN does not preferentially cap the tail — the DD is driven by the sleeve's *magnitude*
(variance), which the sign is irrelevant to. (The Calmar p96 is misleading: the signed sleeve has
positive standalone *return* so its Calmar beats zero-mean random-sign noise — but that is the
sleeve's own alpha, not DD-capping crisis alpha. For the *objective*, DD, the sign fails p95.)

### G8 cost: not a cost story
Combined Calmar/DD essentially flat across 1/3/4.5 bps (the sleeve flips rarely, low turnover) — the
failure is structural variance, not cost.

---

## STEP 4 — honest decision

**Pre-committed preconditions for proposing the trend hedge sleeve (ALL required):**
(a) robustly negatively-correlated-in-crises (profits across MULTIPLE DD episodes) — **FAIL**
  (corr negative in 3/4 but |≤0.10|; PnL profits in 1–2 episodes, loses in 2 incl. 2025_q4);
(b) combined portfolio maxDD IMPROVES net of whipsaw, across episodes — **FAIL**
  (DD WORSE at every w on every universe; HL70 Calmar destroyed; the EXT "lift" is added return
  outrunning a ballooning DD, not DD reduction);
(c) the DD benefit survives episode-LOFO — **FAIL** (entire lift is 2022_luna; drop it → −0.08);
(d) the trend SIGN beats a random-sign magnitude placebo for DD-capping — **FAIL** (p38–p50).

**→ NO-BUILD. No change proposed. Champion stays = baseline (HL70 Calmar +1.68).**

## Why this is the expected result — what it teaches

1. **Crisis-alpha trend-following does not reliably hedge these crypto alt-bears.** The premise
   (trend profits in prolonged moves) holds in exactly ONE of four episodes — 2022_luna, the only
   *genuinely persistent* crash. The other three DD episodes (ftx, 2024_summer, 2025_q4) are
   choppy/bounce-y at the trade horizon — the SAME forward-coin-flip structure iter-008/010 measured
   — so the reactive trend sleeve flips short near the bottom and whipsaws. A trend-follower IS
   reactive to trailing price; it inherits the iter-008/010 "the alt-bear flag is a coincident
   bottom-detector" pathology.
2. **A directional sleeve cannot reduce a beta-neutral book's DD by addition.** The book is a tight,
   beta-neutral, gross-1.0 stream (maxDD ~5k bps); a TSMOM sleeve is a directional gross-1.0 stream
   with ~6–15× larger DD. Adding any meaningful weight of a higher-variance stream RAISES portfolio
   DD unless its returns are *strongly* negatively correlated with the book in the tail — and here
   the in-crisis correlation is only ≈−0.1 and the wrong sign in luna. The G4 sign-placebo confirms
   the DD is magnitude-driven, sign-irrelevant.
3. **The diversification it DOES provide is a Sharpe/return effect, not a DD effect.** Combined
   Sharpe and totPnL rise at low w (TSMOM is a real, lowly-correlated return stream). If the human's
   objective were Sharpe/return, a small TSMOM allocation is defensible as a return diversifier. But
   the system objective is **Calmar / drawdown**, and on that axis it strictly hurts on the
   production universe and fails LOFO on EXT.

**Converges with iter-006/008/010:** the alt-bear is not forward-separable from a bottom on free
price; FLAT runs-smaller (p72/p0), NET-SHORT squeezes (p10), FAST-onset still bounces (p8–p18), and
now REACTIVE TREND whipsaws (profits 1/4 episodes, DD-placebo p38–p50, LOFO −0.08). All
price-reactive directional overlays fail for the same structural reason. **The remaining honest paths
are unchanged:** (a) a *return* diversifier (small TSMOM sleeve) if the human re-weights toward
Sharpe — explicitly NOT a DD hedge; (b) a paid *leading* deleverage signal (Coinglass liquidation
flow / Glassnode on-chain) — human budget decision; (c) accept the structural DD + live kill-switch.

Artifacts:
- script: `research/convexity_portable_2026-05-20/scripts/iter017_trend_hedge_sleeve.py`
- per-cycle: `results/iter017_trend_sleeve_{EXT,HL70,S44}.parquet`
- console reproduces all tables (standalone, corr-in-crisis, per-episode PnL, calm cost, combined
  DD/Calmar curve, episode-LOFO, TEST 4 framing, G4 sign-placebo, G8 cost).
