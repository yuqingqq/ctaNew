# What separates moves that LAST (QNT, LSK) from moves that FADE (G, MAGMA)? — declared before results

Written 2026-09-28, before the holdout data (April–July 2026) was downloaded or looked at.
Scripts: `research/persistence_study.py` (+ `download_usdm_1m.py --out-subdir`, `replay.py --panel-dir`).

## Data and windows
- **Discovery** (already seen in development; generates hypotheses only): NEW signals
  2026-08-02 → 2026-09-24 (so every signal has a 72 h outcome inside the panel).
- **Holdout** (UNTOUCHED until this protocol is committed to disk): NEW signals
  2026-04-02 → 2026-07-28 from a separate Binance-USDM 1m panel (2026-04-01 → 08-01).
  Deviation stated: CLAUDE.md rule 11 prefers a LATER untouched window; none exists yet
  except forward live data, so an EARLIER untouched window is used and the forward live log
  remains the final test.
- Signals: the ≤120-msgs/day preset (size ×1.5, z ×1.0, volume ×1.0, liquidity ≥ $5M; alert
  logic v4) — same thresholds in both windows, nothing tuned on the holdout.
- **De-duplication (rule 2):** per coin, a signal is kept only if no kept signal for that coin
  in the previous 72 h, so kept signals never share a coin's outcome window.
- Primary population: **pumps**. Dumps reported as secondary.

## Outcomes (entry = alert minute close + 1 min; direction-adjusted: + = the move lasted)
- **Primary: x72h** = log return entry → +72 h minus the median liquid coin (≥ $5M/day)
  over the same span.
- Secondary: x24h; `held72` = raw price at +72 h ≥ entry price.

## Features, measured at the alert with data at or before it (± = direction-adjusted)
| # | feature | definition |
|---|---|---|
| 1 | `age_days` | log10 days since the symbol's first Binance-USDM daily kline file |
| 2 | `calm_7d` | log10 RMS 1-min log return over [alert − 8 d, alert − 1 d] (p99-clipped): normal volatility before the run-up |
| 3 | `runup_72h` ± | log return over the 72 h before the alert |
| 4 | `speed` | ±1 h move ÷ max(±24 h move, 1%), clipped [−1, 2]: vertical spike (→1) vs gradual build |
| 5 | `liq_pre` | log10 24 h USD volume ending 24 h before the alert (pre-move liquidity) |
| 6 | `vol_surge` | log10 (24 h volume at alert ÷ mean daily volume over the 7 d ending 24 h before) |
| 7 | `mkt_breadth` | share of liquid coins moving ≥ 2% the same way over the trigger horizon |
| 8 | `oi_chg_24h` | open interest (contracts) % change over the last 24 h (Vision metrics) |
| 9 | `acct_ls` ± | log all-account long/short ratio at the alert (crowding) |
| 10 | `top_ls_chg_24h` ± | change in log top-trader long/short over 24 h (who is adding) |
| 11 | `premium` ± | perp premium over the spot index at the alert |
| 12 | `premium_24h` ± | mean premium over the last 24 h (funding pressure) |
| 13 | `has_spot` | a Binance SPOT market printed klines that day (0/1) |
| 14 | `spot_share_4h` | spot ÷ (spot + perp) USD volume over the last 4 h (0 if no spot) |
| 15 | `prior_signals_30d` | the coin's kept signals in the previous 30 days (serial movers) |
| 16 | `trigger_h` | horizon that fired (1 m = 1 … 24 h = 6) |

**Confirmation (second stage, 4 h after the alert):** `hold_4h` ± = return alert → +4 h;
`vol_persist` = log10 (USD volume in (alert+1 h, alert+4 h] ÷ USD volume in (alert−3 h, alert]).
Their outcome starts AFTER they are measured: x from +4 h to +72 h.

## Tests (block bootstrap because 72 h outcomes overlap across days)
- Spearman IC feature vs x72h; 95% CI by **ISO-week block bootstrap** (1,000 draws).
- **Family-wise null:** 500 permutations shuffling x72h within ISO week; 95th percentile of
  max |IC| over the 16 static features (discovery pumps).
- **VALIDATED difference** = discovery: CI excludes 0 AND |IC| ≥ family-wise null; AND holdout:
  same sign AND CI excludes 0. **Suggestive** = discovery passes, holdout same sign, CI spans 0.
- **Combined score** (declared now): average of ranks of every feature that passes the
  discovery criterion, each signed by its discovery IC → holdout IC and
  top-minus-bottom tercile x72h with CI. If no feature passes discovery, no score is built.
- Exclusions (no metrics / premium / spot file / outcome) are counted per feature.

## Results (2026-09-28, run once after this protocol was written) — `data/replay/persistence/results.json`

**Samples** (after 72 h per-coin de-dup): discovery 1,004 (552 pumps / 452 dumps; from 1,825);
holdout 1,805 (1,060 / 745; from 3,511). Exclusions counted in results.json (e.g. `calm_7d`
148 holdout rows: coins listed < 8 days; `premium_24h` 160). Memory peaks: study 1.08 GiB,
holdout replay 1.43 GiB (default 4 GiB slot cap, 2 GiB self-guard).

**Base rates (replicate):** pumps x72h mean −4.9% / median −6.8%, 32.8% held (discovery);
−5.7% / −8.7%, 29.9% held (**holdout**). Dumps continue mildly: +1.2% / +2.6% further
(market-adjusted), 47% / 58% still lower after 72 h.

**Verdicts (pumps, declared criterion):**
- **VALIDATED — `vol_persist`** (volume in hours 1–4 after the alert ÷ the 3 h up to it):
  discovery IC −0.125 [−0.20, −0.04] (> family null 0.093), holdout −0.240 [−0.29, −0.19].
  Heavy follow-through volume → the pump fades MORE over +4 h → +72 h. Holdout top − bottom
  third: −11.5% [−14.8, −7.9] (quiet 0.59× → +0.1%; heavy 4.5× → −11.4%). Robustness
  (post-hoc, not part of the verdict): negative within every first-4 h price tercile and every
  liquidity tercile; 17/18 holdout weeks negative. Discovery: 8/8 weeks negative but the mean
  difference is small (−0.4% [−5.4, +4.9]) and concentrated in small coins.
- **discovery only (failed holdout):** `liq_pre` +0.185 → holdout −0.012. Combined score (only
  liq_pre qualified) → holdout IC −0.012: no multi-feature score validates.
- **Same sign + significant in the holdout but below the family-wise bar in discovery (not
  validated):** `runup_72h` (−0.120 / −0.101), `trigger_h` (−0.129 / −0.086), `oi_chg_24h`
  (−0.114 / −0.065); confirmation `hold_4h` failed discovery (−0.074, CI touches 0) but holdout
  −0.186 [−0.26, −0.12].
- **Nothing (both windows):** calm_7d, speed, acct_ls, prior_signals_30d, age_days (holdout
  borderline), vol_surge; premium mixed (holdout +0.111, discovery −0.013).
Dumps (secondary): no feature consistent across windows.

## Review 2026-09-28 — supersedes the verdicts above (original text kept as provenance)

Two independent reviewers (code/protocol audit; raw-data re-implementation) plus a claims audit.

**Critical — the "VALIDATED" label is withdrawn.** The protocol declares ONE family-wise null
(max |IC| over the 16 static features, 0.133; re-estimated 0.136 ± 0.004). The code tested the
two confirmation features against an undeclared 2-feature null (0.093). Under the declared bar
`vol_persist` (discovery |IC| 0.125, family-wise p ≈ 0.09) FAILS discovery. **No feature is
VALIDATED under the protocol as written.** `vol_persist` joins runup_72h, trigger_h, oi_chg_24h
and hold_4h as "significant in the holdout, below the discovery bar". It remains the strongest
holdout effect: IC −0.240 [−0.29, −0.19], above the holdout's own joint 18-feature null (0.088),
reproduced exactly from raw files (1,060/1,060 values; −0.234 on raw, non-market-adjusted
returns), partial IC −0.165 controlling for hold_4h and trigger_h. Status: best CANDIDATE;
the holdout is consumed, so confirmation needs forward live data.

**Major.**
- Vision metrics rows stamped T cover [T, T+5 min): `create_time <= alert` looked up to 5 min
  past the alert (oi_chg_24h, acct_ls, top_ls_chg_24h; also docs/SIGNAL_FEATURES.md). PIT
  recompute moves ICs by ≤ 0.05; no pump verdict changes. Fix: `create_time + 5 min <= alert`.
- "%" figures were exp(mean log) − 1, labelled "mean". Arithmetic means: pumps x72h −1.6%
  (discovery) / −1.3% (holdout); vol_persist top − bottom third −8.2% [−12.3, −3.2] holdout and
  **+3.2% [−2.4, +8.0] discovery (sign flips)**. Medians: holdout −2.1 / −4.8 / −11.2%, discovery
  −1.5 / −3.6 / −6.6% (quiet / mid / heavy). Report medians and arithmetic means, never the
  geometric figure as "mean".
- Binance Vision keeps publishing flat zero-volume bars after a delisting: 11 holdout pump
  outcomes (1%) were computed on frozen prices with no exclusion status (rule 4). IC effect
  +0.0014. Fix: detect the last real trade; status DELISTED_IN_WINDOW, counted.

**Minor.** premium_24h NaN for every 00:00-UTC-hour alert (middle day not loaded); panel-start
truncation (vol_surge divides by 7 on clipped windows; calm_7d / runup NaNs are mostly panel
start, not "new listings"); has_spot used same-day post-alert spot listings (5 rows) and wrong
spot tickers for 1000CAT / 1000SATS / 1MBABYDOGE / LUNA2 (8 rows); terciles split ties by time
(zero-inflated features); null Monte Carlo noise ±0.004 (trigger_h misses by 0.0036); discovery
has only 9 week-clusters; 24 discovery pumps are stock/commodity perps; the combined score
omitted the declared top-minus-bottom CI and x24h was never analysed; several robustness numbers
in the results text were not produced by the script (rule 10); the "Nothing" list omitted
mkt_breadth, has_spot, spot_share_4h, premium_24h (holdout-significant) and top_ls_chg_24h.
Claims audit: "30% of pumps still up (vs market)" was a RAW figure — random liquid coins were
up 43% (holdout) / 52% (discovery) over the same 72 h; "quiet volume → held" overstated (quiet
third median −2.1%, 57% faded). Pre-registration is not verifiable by commit (movewatch/ is
untracked; rule 12).

**Checked and correct:** alert timing (idx[t] + 1 min, all alerts); no feature/outcome window
overlap for the confirmation features; outcome windows; premium and spot timestamps (µs switch
handled); direction signs; 72 h de-dup (re-implemented); market adjustment PIT; signal generator
PIT; holdout untouched by thresholds; no survivorship loss; new listings and gaps do not drive
vol_persist.

## Rerun after the bug fixes (v3, 2026-09-28) — `data/replay/persistence/results.json`; v1 kept as `results.v1.json`

Fixes: Vision metrics point-in-time (+5 min span); frozen post-delisting bars excluded as a counted
status (outcomes, market median, random baseline); every lookback day loaded; windows before
listing / panel start -> NaN; spot existence point-in-time + ticker aliases; 37 / 19 stock &
commodity perp signals excluded; value-based terciles; 2,000-draw bootstrap / permutations;
direction-aware simple returns (a short's gain is 1 − exp(−x), never exp(x) − 1); x24h; declared
combined score with CI; robustness computed in the script. Samples: discovery 967 (513 pumps
with outcomes; 5 STALE_OR_UNAVAILABLE), holdout 1,786 (1,043 pumps; 17 STALE_OR_UNAVAILABLE).
Nulls: discovery static-16 0.146, discovery joint-18 0.147, holdout joint-18 0.094.

- **Base rates (hold in both windows):** pumps over 72 h vs the market — median −7.5% / −8.7%,
  arithmetic mean −1.3% / −1.2%, vs random liquid coin-minutes median −0.1% / −0.2%, mean +1.4% /
  +0.7%. Raw price above entry after 72 h: pumps 31.6% / 29.9% vs random 51.6% / 42.2%. Dumps:
  mean −0.8% / −2.1% (slight bounce), median +0.5% / +1.5% — no reliable pattern.
- **Verdicts:** no feature VALIDATED. liq_pre = discovery only (holdout −0.001). Holdout-only (pass
  the holdout joint null, below the discovery bar): vol_persist −0.242, hold_4h −0.188, premium
  +0.116 (discovery +0.006, inconsistent), runup_72h −0.108 (discovery −0.115, nominal). Combined
  score (liq_pre) holdout IC −0.001.
- **vol_persist (best candidate):** holdout heavy third mean −5.8% / median −11.3% vs quiet third
  mean +2.1% / median −2.1%; top − bottom −7.9% [−12.6, −2.9]; 16/17 weeks same sign; negative
  within every hold_4h, liquidity and trigger tercile; partial IC −0.165; raw-return IC −0.235.
  Discovery: IC −0.115 (below its bar), medians −1.9 / −4.4 / −7.2% but means +3.4% [−3.3, +8.5]
  top − bottom (opposite sign). Unproven; needs forward data.
