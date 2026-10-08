# Why do some pumps keep running? (declared before results, 2026-10-08)

**Question.** About 30% of pump alerts are higher 72 h later, and about 5% are 20–50%+ higher.
What separates them from pumps that fade: news, exchange listings, positioning?

**Data: consumed.** These are the replay windows Apr 2–Jul 28 and Aug 2–Sep 24, already read by
the persistence and hedge studies. The study is **exploratory**. Anything consistent becomes a
candidate to record live and validate forward; nothing is adopted from it.

## Population and groups
- **Pumps:** the BTC-hedge study's rows (`data/replay/btc_hedge/events.parquet`, `h == "0_72"`,
  status OK, `dir > 0`): stocks excluded, one alert per coin per 72 h. Each is joined to
  `data/replay/persistence/features_*.parquet` on (window, asset, x72h) for time and pre-alert
  features.
- **Outcome:** y = exp(x72h) − 1, the market-adjusted return from alert + 1 min to + 72 h.
  - **RUNNER:** y ≥ +20%.
  - **FADER:** y ≤ −10%.
  - **MIDDLE:** everything else (reported, not compared).

## Part A: structured, all runners and faders
1. **Listings**, by date, in the window [alert date − 7 d, alert date + 3 d]: Binance spot (first
   Vision 1m file), Upbit KRW and Bithumb KRW (first daily candle), Coinbase USD (first daily
   candle). Each is split into `before` (listing date ≤ alert date) and `after` (later, ≤ +3 d).
   A new Binance perp (age ≤ 14 d at the alert) is reported too.
2. **First 4 h after the alert** (Binance USDM, Vision metrics + premium index; knowable at the
   +4 h check):
   - open-interest change (value, alert → +4 h);
   - mean premium over hours 3–4 (funding pressure; negative = shorts crowded);
   - mean taker long/short volume ratio, 0–4 h;
   - top-trader long/short ratio change, 0–4 h;
   - plus `hold_4h` and `vol_persist` from the features.
3. **Pre-alert context** (features): age, run-up 72 h, calm 7 d, OI change 24 h, account
   long/short ratio, breadth.
- **Statistics, per window and pooled:**
  - Binary: rates with Wilson 95% CIs, runner − fader difference, Fisher exact p.
  - Continuous: medians, Mann–Whitney p.
- **"Consistent"** means the same sign in both windows **and** pooled p < 0.01. Twenty-odd
  comparisons are made, and 0.01 is the guard against picking one by chance. Exclusions
  (no metrics file, no listing data) are counted per feature.

## Part B: news, blind sample
- **Sample:** 30 runners (seeded random, split across windows in proportion), each paired with a
  FADER from the same window with the nearest alert time and a different coin. That gives 60
  events, shuffled.
- **Blind labelling:** search agents get only (token, Binance symbol, alert time UTC, alert
  price), never the outcome or the group. For each event they find news published from
  72 h before to 72 h after the alert. Each item gets one category:
  - `EXCHANGE_LISTING` (which exchange)
  - `PARTNERSHIP_ADOPTION`
  - `PRODUCT_LAUNCH`
  - `TOKENOMICS` (unlock / burn / buyback / airdrop / staking)
  - `FUNDING_INVESTMENT`
  - `REGULATORY_LEGAL`
  - `NARRATIVE_SECTOR`
  - `INFLUENCER_SOCIAL`
  - `SECURITY_INCIDENT`
  - `DELISTING_WARNING`
  - `OTHER`
  - `PRICE_ONLY`: articles that only report the price move, never a catalyst
  - `NONE_FOUND`
  Each item is also tagged by publication time: before the alert, or after it (≤ 72 h).
- **Primary comparison:** share of events with at least one non-price catalyst published
  **before** the alert, runners vs faders (Fisher exact). Per-category counts are secondary.
- **Power and caveat.**
  - With 30 vs 30, only a difference of roughly 35 percentage points or more is detectable.
  - Coverage after the alert partly reflects the outcome, since big moves attract articles.
    That is why the primary comparison uses news published before the alert.
