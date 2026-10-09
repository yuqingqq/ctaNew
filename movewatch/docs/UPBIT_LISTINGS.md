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
