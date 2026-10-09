# Signal search: what, at the alert or 4 h later, predicts where a detected move goes? (declared before results, 2026-10-08)

**Question.** The earlier studies found no reliable signal. They are listed in the table below.
This search does two things:
1. It gives the earlier leads a clean test on an **untouched** window.
2. It screens a list of **new**, mechanism-motivated features, then validates the survivors on
   the same untouched window.

Scope: event-level signals around detected moves (the movewatch domain). The repo's
cross-sectional and time-series alpha research on free data is closed (repo `docs/CONCLUSION_2026-08-03.md`,
memory) and is not reopened here.

| study | status of its windows | lead carried here |
|---|---|---|
| signal_features (Aug, Sep 1–19 2026) | consumed | breadth → 4 h fade |
| persistence (Apr–Jul, Aug–Sep 2026) | consumed | vol_persist (sign flipped across windows) |
| btc_hedge, runners (same windows) | consumed | premium_h4, taker_ls_4h, has_spot, age, calm_7d |

## Windows
- **Screen** (consumed; generates hypotheses only):
  - S1 = Aug 2–Sep 24 2026;
  - S2 = Apr 2–Jul 28 2026.
- **Validation V** (UNTOUCHED by movewatch; downloaded only after this file is committed):
  - signals 2025-10-01 → 2026-03-29, from two Binance-USDM 1m panels: `val_a` 2025-09-01 → 2026-01-01
    (signals to 2025-12-29) and `val_b` 2025-12-01 → 2026-04-01 (signals from 2025-12-29);
  - each panel starts ≥ 30 days before its signals, so the lookbacks are inside it;
  - rows are de-duplicated per coin across the seam (72 h).
  - **Read once.** The validate step refuses unless the freeze file is committed, and refuses a
    second run.
  - Deviation stated: rule 11 prefers a LATER window. The only later data is the live forward
    test, which is reserved, so an EARLIER window is used.
  - Other repo programs read this period for different questions: cross-sectional alpha, and
    the pump→dump thread with rvol/age/funding on parabolic run-ups. For A6/A7 below, the V read
    is therefore not perfectly clean; this is flagged in the receipt.
- **Live forward test v2:** untouched. Nothing here reads its outcomes.

## Events (rows are actions)
- Replay NEW alerts with the ≤120-msgs/day preset (size ×1.5, z ×1.0, volume ×1.0, liquidity
  ≥ $5M, alert logic v4). These are the same thresholds as S1/S2 (`study120`, `holdout120`);
  nothing is tuned on V.
- Exclusions, each a counted status:
  - stocks and commodities (static list, plus the venue-labelled price match from `btc_hedge.stock_like`);
  - BTC itself;
  - one alert per coin per 72 h;
  - frozen or delisted bars.
- Populations: **pumps** (dir +1) and **dumps** (dir −1), analysed separately.

## Outcomes
Direction-adjusted log return minus the median liquid coin (≥ $5M/day, the coin excluded), so
**+ = the move continued**. Entry is the alert minute close + 1 min (t + 1).
- `x4h`: t + 1 → t + 241.
- `x72h`: t + 1 → t + 4,321.
- `x4_72`: t + 241 → t + 4,321 (from the +4 h check).

Features knowable only at +4 h are tested on `x4_72` only (rule 9: no shared span).

## Group A: carried leads (straight to V; sign declared from the earlier studies)
| id | feature | pop | outcome | declared sign | source |
|---|---|---|---|---|---|
| A1 | `mkt_breadth` | pumps + dumps | x4h | − | signal_features |
| A2 | `vol_persist` (log10 vol h1–4 ÷ vol h−3–0) | pumps | x4_72 | − | persistence holdout; forward H1 |
| A3 | `premium_h4` (mean perp premium, hours 3–4) | pumps | x4_72 | + | runners |
| A4 | `taker_ls_4h` (mean taker long/short vol ratio 0–4 h) | pumps | x4_72 | + | runners |
| A5 | `has_spot` (Binance spot market exists, point in time) | pumps | x72h | − | runners |
| A6 | `age_days` (log10) | pumps | x72h | − | runners |
| A7 | `calm_7d` (log10 RMS 1m return, alert − 8 d … − 1 d) | pumps | x72h | + | runners |
| A8 | base rate: median `x72h` of pumps | pumps | x72h | < 0 | persistence, btc_hedge |

Descriptive in V only, no test:
- the dumps' median `x72h`;
- the unconditional pump-fade P&L net of costs and funding at 0→4 h and 0→72 h.

## Group B: new features (screened on S1/S2, alert-time data only, through minute t)
± = direction-adjusted (multiplied by dir).

| id | feature | definition | mechanism |
|---|---|---|---|
| B1 | `trade_size` | log10 (USD per trade over [t−59, t] ÷ USD per trade over [t−7 d−60, t−60]); 1m `trades` | whale-driven vs retail-driven move |
| B2 | `wick15` | pumps: (max high [t−14, t] − close t) ÷ (max high − min low over [t−59, t]); dumps mirrored with the low | rejection at the alert |
| B3 | `hi7` ± | pumps: log(close t ÷ max close [t−7 d, t−60]); dumps: −log(close t ÷ min close) | breakout to a new 7 d extreme |
| B4 | `btc_24h` ± | log BTC return over [t−1,440, t] | market regime behind the move |
| B5 | `asia` | alert hour in 00–07 UTC (0/1) | session liquidity |
| B6 | `weekend` | Saturday or Sunday UTC (0/1) | thin weekend books |
| B7 | `upbit_listed` | the Upbit KRW market had traded within the 24 h before the alert (0/1) | Korean retail access |
| B8 | `kimchi` ± | log(Upbit KRW close ÷ KRW-USDT close) − log(perp close ÷ multiplier) at t; Upbit-listed only; \|·\| > 0.5 → status `PRICE_MISMATCH` | Korea leading the move |
| B9 | `upbit_share` | Upbit USD volume over [t−59, t] ÷ (Upbit + Binance perp) | Korean-retail-led move |
| B10 | `oi_chg_1h` | log OI(t) ÷ OI(t − 60 min), Vision metrics point in time | leverage added vs liquidated (flush) |

- Cells: 10 features × 2 populations × 2 outcomes (`x4h`, `x72h`) = **40 cells**.
- **Upbit caveat:** markets Upbit has since delisted cannot be queried. They read as "not listed"
  (B7) or missing (B8/B9), with status `UPBIT_NO_DATA`. This is survivorship on the Upbit side and
  is reported with the counts.

## Statistics
- **IC:** Spearman between feature and outcome.
  - 95% CI: ISO-week block bootstrap, 4,000 draws; weeks with < 3 signal days are merged.
  - One-sided p = (1 + draws on the wrong side of 0) ÷ (draws + 1).
- **Minimum per cell:** n ≥ 100 non-missing rows and ≥ 5 weeks. Otherwise `INSUFFICIENT`, which
  never passes.
- **Screen pass**, computed:
  - the IC has the same sign in S1 and S2; and
  - the pooled S1 ∪ S2 CI excludes 0.
  - `both_windows` is also recorded: each window's own CI excludes 0.
- **Freeze**, computed by the screen step:
  - the carried list is A1–A8 plus every screen-passing B cell, with its pooled sign;
  - k = the list length;
  - tercile cut points from S1 ∪ S2.
  - Written to `docs/SIGNAL_SEARCH_FREEZE.json`, committed before V is built.
- **Validation** on V, per carried cell:
  - **VALIDATED** = declared sign, 95% CI excludes 0, and one-sided p ≤ 0.05 / k (Bonferroni).
  - **SUGGESTIVE** = declared sign and CI excludes 0, but p > 0.05 / k.
  - **FAILED** otherwise.
- **Economics**, for VALIDATED cells, reported for S1 ∪ S2 and V:
  - The trades:
    - fade (side = −dir) in the tercile with the lowest predicted continuation;
    - follow (side = +dir) in the highest tercile;
    - for binary features, the two values in place of the terciles.
  - Hold = the cell's outcome span; coin leg only (BTC does not hedge, `BTC_HEDGE_STUDY.md`).
  - Costs: taker 5 bp per side plus slippage tiered by 24 h volume (5/10/20/40 bp); actual
    Binance funding.
  - Report: mean net with week-block CI, median, win rate, p5, n, and the difference vs the
    same trade on all events.
  - **TRADABLE** = mean net CI > 0 in V, with the same sign of the mean in S1 ∪ S2.
- **Robustness**, for VALIDATED cells, descriptive: partial IC controlling for `liq_pre`,
  `mkt_breadth`, `calm_7d`.

## Self-tests (abort on failure)
1. **Reconciliation:**
   - S1/S2 rows rebuilt by this script match `data/replay/persistence/features_*.parquet` on
     `x72h` (0 mismatches allowed);
   - `premium_h4` matches `data/replay/runners/pumps.parquet`.
2. **Falsifiers for the screen and validation instrument:**
   - a planted feature (outcome rank + noise, IC ≈ 0.15) must pass;
   - a shuffled feature must not pass;
   - an all-missing feature must come back `INSUFFICIENT`.
3. **Look-ahead flag:** any screen |IC| > 0.10 is marked `SUSPICIOUS` and its code is reviewed
   before the freeze.

Memory: `.worktrees/research_slot.sh` at the default 4 GiB cap, `MALLOC_ARENA_MAX=2`, abort
above 2.0 GiB RSS; panels are disk memmaps.

## Screen results (2026-10-08; S1/S2 only, script 01419c7; `data/replay/signal_search/screen_results.json`)

**Run quality:**
- Rows: S2 1,747 (1,041 pumps / 706 dumps) and S1 886 (459 / 427), after these statuses:
  - stock-price matches 39 / 80;
  - BTC 0 / 1.
- Trigger-minute price vs the alert price: 0 mismatches.
- Reconciliation:
  - `x72h` matches the persistence study 1,747/1,747 and 886/886;
  - `premium_h4` matches the runners study 1,415/1,415.
- Falsifiers:
  - the planted feature (IC 0.152) passes and validates;
  - noise (IC 0.026) fails;
  - all-missing comes back `INSUFFICIENT`.
- Upbit: 80% of events have no queryable KRW market (`UPBIT_NO_DATA` 1,389 / 645). `kimchi` and
  `upbit_share` have n ≈ 320 pumps and ≈ 180 dumps; S2 dumps fall below the minimum.

**Passing cells (computed), 9 of 40.** IC S1 / S2 / pooled [CI]:

| cell | S1 | S2 | pooled [95% CI] | both windows |
|---|---:|---:|---|---|
| `hi7` pumps x4h | −0.097 | −0.097 | −0.100 [−0.137, −0.053] | yes |
| `hi7` pumps x72h | −0.119 | −0.085 | −0.091 [−0.133, −0.041] | yes |
| `hi7` dumps x72h | −0.139 | −0.086 | −0.108 [−0.174, −0.037] | no |
| `upbit_listed` pumps x4h | −0.027 | −0.079 | −0.065 [−0.106, −0.021] | no |
| `kimchi` pumps x72h | −0.181 | −0.096 | −0.143 [−0.266, −0.005] | no |
| `oi_chg_1h` pumps x4h | −0.177 | −0.078 | −0.108 [−0.162, −0.044] | no |
| `oi_chg_1h` pumps x72h | −0.145 | −0.078 | −0.096 [−0.140, −0.050] | yes |
| `oi_chg_1h` dumps x4h | +0.108 | +0.110 | +0.112 [+0.055, +0.169] | no (S1 CI) |
| `oi_chg_1h` dumps x72h | +0.089 | +0.094 | +0.094 [+0.049, +0.135] | no |

**Nothing passed:**
- trade size, wick, BTC 24 h, Asia hours and weekend;
- Upbit volume share (a large S1 IC for dumps, +0.35, but S2 is `INSUFFICIENT`).

**Look-ahead review (the declared step for `SUSPICIOUS` cells, |IC| > 0.10).** Each feature was
recomputed from data ending one bar earlier (scratch script, S data only). IC as built →
one bar earlier:

| feature | cell | as built | one bar earlier |
|---|---|---:|---:|
| `oi_chg_1h` (OI 5 min earlier) | pumps x4h | −0.108 | −0.096 |
| | dumps x4h | +0.112 | +0.127 |
| | pumps x72h | −0.096 | −0.081 |
| | dumps x72h | +0.094 | +0.089 |
| `hi7` (close at t − 1) | dumps x72h | −0.108 | −0.091 |
| | pumps x4h | −0.100 | −0.094 |
| | pumps x72h | −0.091 | −0.095 |

- No effect depends on the last bar.
- `kimchi` was reviewed in code: Upbit's `to` is exclusive on the candle start (verified: `to`
  12:00 → last candle 11:59), so it uses the Upbit close of minute t, the same minute as the perp
  close.
- Verdict: not look-ahead.

**Group A on S, for context only** (these leads were chosen on these windows):
- A1 −0.094 [−0.147, −0.034]
- A2 −0.201 [−0.263, −0.139]
- A3 +0.100 [+0.036, +0.167]
- A4 +0.109 [+0.062, +0.156]
- A5 −0.046 [−0.099, +0.008]
- A6 −0.037 [−0.080, +0.012]
- A7 −0.005 [−0.069, +0.052]
- A8 median −8.6% [−9.8, −7.5]

**Freeze:** `docs/SIGNAL_SEARCH_FREEZE.json`.
- Carried: 16 IC cells (A1–A7 plus the 9 above) and the A8 base rate, so **k = 17**.
- Bonferroni bar: one-sided p ≤ 0.0029.
- The same feature at two horizons counts twice, which is conservative.

## Validation results (2026-10-08, the one read; `data/replay/signal_search/validation_receipt.json`)

**V window and run:**
- Signals 2025-10-01 → 2026-03-29: 1,791 pumps and 1,214 dumps, 26 weeks.
- Seam duplicates 8.
- Trigger-price reconciliation 0 mismatches; stock-price matches 1 + 1.
- `UPBIT_NO_DATA` 1,432 / 853; `PRICE_MISMATCH` 16 / 2.
- Freeze 406e7d8; script 01419c7; k = 17; Bonferroni bar one-sided p ≤ 0.0029.
- Before the read, the validate code was rehearsed on the S rows (scratch receipt) so that a crash
  could not force a second look.

| id | cell | declared | V IC [95% CI] | n | verdict (computed) |
|---|---|---|---|---:|---|
| A1 | `mkt_breadth` all x4h | − | −0.137 [−0.202, −0.046], p 0.0007 | 3,002 | **VALIDATED** |
| A2 | `vol_persist` pumps x4_72 | − | −0.202 [−0.235, −0.167], p 0.0002 | 1,773 | **VALIDATED** |
| A3 | `premium_h4` pumps x4_72 | + | +0.121 [+0.059, +0.190], p 0.0002 | 1,755 | **VALIDATED** |
| A4 | `taker_ls_4h` pumps x4_72 | + | +0.054 [−0.007, +0.128] | 1,775 | FAILED |
| A5 | `has_spot` pumps x72h | − | −0.067 [−0.104, −0.018] | 1,775 | SUGGESTIVE |
| A6 | `age_days` pumps x72h | − | −0.030 [−0.069, +0.025] | 1,775 | FAILED |
| A7 | `calm_7d` pumps x72h | + | +0.048 [−0.013, +0.104] | 1,692 | FAILED |
| A8 | pumps median x72h | < 0 | **−6.8% [−7.8, −5.4]**, p 0.0002 | 1,775 | **VALIDATED** |
| B3 | `hi7` pumps x4h | − | −0.017 [−0.121, +0.083] | 1,718 | FAILED |
| B3 | `hi7` pumps x72h | − | −0.086 [−0.178, −0.002] | 1,703 | SUGGESTIVE |
| B3 | `hi7` dumps x72h | − | −0.227 [−0.369, +0.017] | 1,156 | FAILED |
| B7 | `upbit_listed` pumps x4h | − | −0.024 [−0.058, +0.016] | 1,790 | FAILED |
| B8 | `kimchi` pumps x72h | − | −0.023 [−0.147, +0.070] | 397 | FAILED |
| B10 | `oi_chg_1h` pumps x4h | − | −0.028 [−0.118, +0.059] | 1,788 | FAILED |
| B10 | `oi_chg_1h` pumps x72h | − | −0.059 [−0.143, +0.022] | 1,773 | FAILED |
| B10 | `oi_chg_1h` dumps x4h | + | +0.126 [−0.008, +0.181] | 1,212 | FAILED |
| B10 | `oi_chg_1h` dumps x72h | + | +0.148 [+0.013, +0.208] | 1,205 | SUGGESTIVE |

**Partial IC in V**, controlling for liq_pre, breadth and calm_7d: A1 −0.120, A2 −0.201, A3 +0.136.
The three ranking signals are not proxies for liquidity, breadth or volatility.

**Economics of the VALIDATED cells**: net of taker fees, tiered slippage and actual funding; coin
leg only.

| cell | trade | S1 ∪ S2 mean [CI] | V mean [CI] | V median / win / p5 | TRADABLE (declared predicate) |
|---|---|---|---|---|---|
| A1 | fade, top breadth tercile, 4 h | +1.51% [−0.25, +3.33] | +12.51% [+0.43, +23.84] | +4.1% / 64% / −23.7% | **true** (see post-hoc) |
| A1 | follow, bottom tercile, 4 h | −0.56% | −0.25% [−0.81, +0.23] | | |
| A2 | follow, quiet tercile, 4→72 h | +0.60% | +4.45% [−0.37, +8.79] | +0.8% / 52% / −22.2% | false |
| A2 | fade, heavy tercile, 4→72 h | +2.00% | −0.45% [−4.54, +2.88] | +6.3% / 66% / −39.5% | false |
| A3 | fade, premium < 0 tercile, 4→72 h | +2.72% [+0.90, +4.69] | −0.84% [−5.04, +2.75] | +3.5% / 61% / −38.2% | false |

**Descriptive (V):**
- The unconditional pump fade nets −1.09% [−3.65, +0.80] at 4 h and −2.78% [−8.31, +2.02] at
  72 h.
- Its medians are +1.4% / +4.4%, with p5 −28% / −56%.
- Dumps' median x72h is −0.6% [−3.4, +2.5], so dumps do not reliably rebound.

### Post-hoc (after the read; descriptive, decides nothing): `research/signal_search_posthoc.py`
- **A1's V fade mean is one day.**
  - 93% of the fade group's total P&L comes from the week of 2025-10-06. On 2025-10-10 alone:
    296 trades, mean +39.7%. That was the liquidation crash, and fading dumps there means buying
    alts at the wick.
  - Without that week: +1.33% [+0.07, +2.34], median +2.2%, win 61%, n 687.
  - Pumps alone +1.13% [−0.25, +2.36]; dumps alone +1.90% [−1.12, +4.27].
- **S shows the same shape.** 76% of the S fade P&L is the week of 2026-08-17 (2026-08-22: 105
  trades, mean +8.4%).
- Execution in a crash is the open question. The cost model (5–40 bp slippage) is a calm-market
  model, and on 2025-10-10 venues were stressed.

### What it says
- **Three ranking signals and the base rate replicate on untouched data.** All are knowable live:
  - **Breadth.** Moves during broad market moves fade more over 4 h. This is forward test v2 H2.
  - **Post-alert volume.** Heavy volume in hours 1–4 means more fade over 4→72 h; IC −0.20, the
    strongest. This is forward test v2 H1.
  - **Perp premium at +4 h.** A premium still positive means less fade over 4→72 h.
  - **Base rate.** The typical pump fades: median −6.8% at 72 h.
- **None is a clean trade.**
  - The only predicate pass (the A1 fade) comes from two liquidation flushes.
  - Outside them, the fade earns about +1% per 4 h trade, with a CI barely above zero in V and
    spanning zero in S.
  - A2 and A3 rank outcomes, but no tercile trade clears costs in V.
- **Every new feature failed or is only suggestive.** Those are the open-interest flush, the
  distance from the 7-day extreme, the Korean premium and Upbit listing. Their screen passes did
  not survive the untouched window.
- **Candidates, forward only:**
  1. H1 and H2 are already running in forward test v2.
  2. A "buy market-wide liquidation flushes" rule would be a new, declared hypothesis with crash-
     specific execution assumptions. It cannot be validated on these windows, whose crash days have
     now been read.

### Post-hoc: signal 3 (perp premium at +4 h) as a trade (2026-10-09 11:06Z; `research/premium_posthoc.py`; descriptive, decides nothing)
Every window used was already read. Nets are the signal search's (x4_72, stress 1, funding).
- **Pumps, frozen A3 terciles:** the ranking holds, but it is not monotone and no tercile trade
  has the same sign in S and V.

  | window | IC | IC given vol_persist | vol_persist IC given premium | short discount | long premium | long middle (median) |
  |---|---|---|---|---|---|---|
  | S | +0.100 | +0.070 | −0.190 | +2.72% | −0.60% | +1.54% (−2.95%) |
  | V | +0.121 | +0.092 | −0.183 | −0.80% | +0.83% | +2.70% (−0.50%) |

  - The discount tercile fades most (coin median −8.3% / −6.8% from +4 h to +72 h).
  - The middle tercile holds best (−2.7% / −0.1%).
  - The premium tercile has the most +50% runners (6.6% / 4.2% vs 2–4%), but its median coin
    still falls 5.2% / 3.7%.
  - Premium adds a little beyond post-alert volume (partial IC +0.07 / +0.09); volume stays the
    stronger signal.
  - The 2020–25 read of heavy & discount shorts (24 h) was −0.63% (`HEAVY_FADE_V2.md`).
  - So shorting discount pumps was negative in two of three reads.
  - The long-middle mean is positive in both windows, but its median is negative (runner tail),
    and it was noticed after seeing. No untouched pump window is left, so it is forward-only.
- **Dumps, mirror (outcome-blind S terciles −0.57 / +3.83 bp):** no ranking (IC −0.009 S,
  −0.006 V).
  - V's buy-the-discount-dump +6.98% is the 2025-10-10 crash week (25% of V dumps). Without that
    week it is −1.55%, and the group signs flip between S and V.
  - Crash buying is already covered by `EXTREME_CRASH.md`.
- **Verdict:** signal 3 ranks pumps, but it is not a trade on either side, so no test is declared.
  - The 2020–25 dump replays stay unread for alert-level questions.

### Post-hoc: signal 3 from more angles (2026-10-09 11:27Z; `research/premium_angles.py --conf`; descriptive)
**Angles:**
- Pumps: 13 groups × (short / long × 24 h / 72 h, plus short 24 h with a +25% stop). The groups
  are terciles, the extreme deciles (cuts from the S feature distribution), premium × post-alert
  volume, non-meme, and meme.
- Dumps: 6 groups × buy / short, 4 → 72 h.

**Rules, stated before the run:**
- An angle holds if it has the same sign in S and V and both CIs exclude 0.
- Every pump angle with a positive mean in S and V gets a third look on the 2020-01 → 2025-08
  replay pumps. That window was consumed for pump questions but was never split by premium this
  way. 4,823 pumps; premium missing 101.
- The 2020-25 dumps stay unread.

**Results:**
- **S and V:** 77 angles. Only 2 hold, and both are losers (buying discount pumps at 24 h loses in
  both windows). The mirror short does not clear in V.
- **Dumps:** nothing holds. V's means swing on the 2025-10-10 crash week.
- **2020–25 third look:** 16 angles carried; 3 stay positive, none with CI > 0.
  - The short-the-discount family turns negative:
    - discount tercile short 24 h: −1.23% [−2.31, −0.25];
    - deepest 10% (≤ −95 bp) short 24 h: −4.23% [−8.75, −0.05];
    - heavy & discount: −0.63% / −0.81%.
  - In 2020-25 the deeper the discount, the worse the short — the opposite of 2025-26.
  - Survivors:
    - buy the highest-premium 10% (> +14 bp) for 72 h: +2.86 / +2.61 / +1.42% in S / V / C,
      medians −7.3 / −5.9 / −3.9%, every CI spans 0;
    - quiet & premium long 72 h: +0.92%;
    - heavy & premium short 24 h: +0.18%.
  - One survivor in 16 is what chance gives. The buy-high-premium idea is a runner lottery with a
    negative median and stays forward-only.

**Not examined:**
- premium as an exit or stop rule (needs the premium path after +4 h);
- the premium at other hours, or its change;
- other venues' premium;
- market-hedged legs.

**Verdict:** signal 3 is not tradable from any angle examined.
