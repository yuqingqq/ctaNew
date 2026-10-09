# Upbit KRW listing announcements → Binance perp reaction (declared before results, 2026-10-09)

**Question.** When Upbit announces a new KRW market for a coin that already trades as a Binance
USDT perp:
- how much does the perp move, and how fast?
- does a trade on it pay after costs?

This decides whether an Upbit announcement feed (roadmap) is worth adding to movewatch as an
alert.

## Events
- **Source:** `api-manager.upbit.com/api/v1/announcements?category=trade`, 785 notices from
  2017-10 to 2026-10 (`data/hist/upbit_announcements.json`).
- **A listing notice** has a title containing 신규 거래지원 / 디지털 자산 추가 / 신규 상장 /
  코인 추가 (2024+ / 2020–23 / 2019 / 2018) and KRW or 원화.
  - One event per ticker: its FIRST such notice. Later notices about the same listing (delays,
    name changes) are dropped.
- **Time a:** `first_listed_at` floored to the minute (UTC). Perp: `<T>USDT` or `1000<T>USDT`
  in the Binance Vision 1m corpora.
- **Statuses, counted:**
  - no perp, or no data from a − 1 h to a + 5 h (census);
  - `NO_24H_DATA` (paths).

## Cells (screened on discovery; carried cells tested once on confirmation)
- **long_h:** buy the close of a + 1 and sell the close of a + 1 + h, for h ∈ {5, 15, 60, 240}
  minutes.
- **short_o_1440:** short the close of a + o and cover the close of a + 1440 (24 h), for
  o ∈ {15, 60, 240}. This is the "fade the listing pop" trade.
- **Missing closes:** the last close before the minute.
- **Costs:** 2 × (5 bp + 3 × slippage tier by the 24 h volume before a), as the crash study.
- **Funding:** excluded. During a pop, funding usually pays shorts, so leaving it out is
  conservative for the fade.
- **Descriptive:**
  - path returns vs the close of a − 1 at +0, 1, 2, 5, 15, 60, 240 and 1440 minutes;
  - the pre-announcement drift over the 60 minutes before a (a leak check).

## Null
The same coin at 20 random minutes in [a − 30 d, a − 2 d] (same UTC hour where possible), with the
same trade and costs. p_null = (1 + draws with null mean ≥ observed) ÷ 2001; one null minute per
event per draw.

## Windows and rules
- **Discovery** (screen): events from 2025-09-01 on (66 events).
  - Partly consumed: the runners study read listings around 2026 pumps, and the signal search used
    Upbit features.
  - **Screen pass:** n ≥ 20 events, ≥ 5 weeks, mean net > 0 and 95% ISO-week block CI excluding 0.
  - **Freeze:** the screen step writes the carried cells and k to
    `docs/UPBIT_LISTINGS_FREEZE.json`, which is committed before confirmation is read.
- **Confirmation** (UNTOUCHED for this question): events before 2025-09-01 (73 events, 2020-10 →
  2025-08).
  - `paths --window conf` refuses unless the freeze file and the script are committed clean.
  - It is read once.
- **VALIDATED**, per carried cell; all of:
  1. mean > 0, CI excludes 0, and p ≤ 0.05 / k;
  2. p_null ≤ 0.05;
  3. mean > 0 with the best event left out.
- **SUGGESTIVE:** mean > 0 and CI excludes 0, but another condition fails.
- **Falsifiers** in validate, per carried cell: null draws + 10% must pass; null draws alone must
  not.
- **If nothing passes the screen:** k = 0, nothing is confirmed, and the confirmation window
  stays unread.

## Caveats
- Announcement time is the notice's publish time. A live feed polling every ~10 s would see it a
  few seconds later; entry at a + 1 is about 60–120 s after the notice.
- The 1m data cannot resolve the first seconds, where much of a listing pop may already happen.
- Binance tickers can differ from Upbit's (e.g. BEAMX); unmatched tickers are a counted status.

## Result: discovery screen (2026-10-09; `docs/UPBIT_LISTINGS_FREEZE.json`, k = 0)
- 66 events (2025-09 → 2026-08): 65 OK, 1 `NO_24H_DATA`; 1,220 same-coin null rows.
- **Path, median / mean vs the close before the notice:**

  | minutes after | median | mean |
  |---|---|---|
  | 0 (same minute) | **+13.5%** | +17.9% |
  | 1 | +12.0% | +17.7% |
  | 5 | +12.7% | +17.8% |
  | 15 | +12.6% | +18.6% |
  | 60 | +13.9% | +18.8% |
  | 240 | +12.7% | +18.2% |
  | 1440 | **+5.5%** | +13.7% |

  The 60 minutes before the notice: −0.07% median, so no leak.
- **Cells**, mean net at stress 3 [95% CI], win rate:

  | cell | mean net [95% CI] | win |
  |---|---|---|
  | long_5 | −0.99% [−1.82, −0.14] | 35% |
  | long_15 | −0.56% [−1.69, +0.63] | 37% |
  | long_60 | +0.05% [−1.67, +1.90] | 45% |
  | long_240 | −0.20% [−3.83, +3.68] | 37% |
  | short_15_1440 | +2.29% [−4.02, +8.43] | 65% |
  | short_60_1440 | +2.72% [−3.08, +8.53] | 66% |
  | short_240_1440 | +2.36% [−2.77, +7.49] | 66% |

  The same-coin random-time null is −1.1% to −2.3% (costs). **No cell passes the screen: k = 0.**
  The confirmation window (2020 → 2025-08) stays unread.

### What it says
- **The listing pop is complete within the announcement minute.** The median is already +13.5% by
  that minute's close, then flat for 4 h.
- **A live Upbit announcement alert cannot be traded on Binance perps from this box.** Polling
  every few seconds plus a human reaction is too slow; the first seconds go to bots.
- **As information, the alert would only explain a pump after the fact.** It is not worth a
  roadmap slot as a trading signal.
- **Fading the pop over 24 h** (median give-back about 7 points from +4 h to +24 h) is positive but
  noisy at n = 65.
  - A dedicated, declared fade test on the untouched 2020 → 2025-08 window (73 events, k = 1) is
    possible.
  - It was not run: the screen did not carry it.
