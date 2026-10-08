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

## Corrections after the first run (2026-10-08; the first run's table HAD been seen)
1. **Instrument bug.** `age_days` is log10(days), and `new_perp_14d` compared it to 14 directly,
   giving 100% for everyone. It now uses 10^age ≤ 14.
2. **Tautology (rule 9).** The first-4h features were compared on the 0→72 h outcome, which
   contains those 4 h.
   - The first run showed `hold_4h` p 1.8e-28 and `oi_chg_4h` p 2e-14: mostly the shared 0–4 h
     move.
   - They are now compared on the **4→72 h** outcome, with groups re-cut on it (`group4`:
     y4 ≥ +20% / ≤ −10%). This is the same split the persistence study used (`x4_72`).
   - Listings and pre-alert features stay on 0→72 h.

## Results (2026-10-08; script 5258d4c+; `data/replay/runners/results.json`, `news_results.json`)

**Population:** 1,484 pumps.
- 0→72 h groups: runners 127 / 53, faders 477 / 202 (holdout / discovery).
- 4→72 h groups: runners 122 / 54, faders 371 / 141.
- Exclusions per feature are in `results.json`: `premium_h4` 34, `calm_7d` 102, `runup_72h` 36,
  others ≤ 5.

### Part A: structured

Runners vs faders, holdout / discovery. "Consistent" = same sign in both windows and pooled p < 0.01.

| feature | outcome | runners | faders | pooled p | consistent |
|---|---|---|---|---:|---|
| any listing ≤ 7 d **before** the alert (Binance spot, Upbit, Bithumb, Coinbase) | 0→72 h | 2.4% / 1.9% | 3.8% / 2.5% | 0.63 | no |
| any listing ≤ 3 d **after** the alert | 0→72 h | 3.1% / 1.9% | 0.2% / 0.5% | 0.006 | yes |
| has a Binance spot market | 0→72 h | 38.6% / 45.3% | 53.9% / 55.9% | 0.001 | yes |
| perp age, median days | 0→72 h | 324 / 381 | 380 / 499 | 0.003 | yes |
| 7 d volatility before (log10) | 0→72 h | −2.64 / −2.71 | −2.70 / −2.81 | 0.002 | yes |
| new perp (≤ 14 d) | 0→72 h | 3.9% / 1.9% | 1.9% / 0.5% | 0.12 | no |
| breadth, OI 24 h, account L/S, run-up | 0→72 h | — | — | ≥ 0.02 | no |
| perp premium, hours 3–4 after (bp) | 4→72 h | +0.6 / +0.1 | −3.0 / −1.9 | 1e-5 | yes |
| taker buy/sell ratio, 0–4 h | 4→72 h | 1.038 / 1.027 | 1.016 / 1.022 | 0.001 | yes |
| volume ratio hours 1–4 (log10) | 4→72 h | 0.17 / 0.24 | 0.34 / 0.32 | 0.015 | no (p) |
| OI change 0–4 h, price 0–4 h, top-trader L/S | 4→72 h | — | — | ≥ 0.02 | no |

**Premium sign at +4 h:**

| window | premium at +4 h | runner rate | median 4→72 h |
|---|---|---:|---:|
| holdout | > 0 | 14.8% | −3.7% |
| holdout | < 0 | 8.9% | −5.7% |
| discovery | > 0 | 15.1% | −3.9% |
| discovery | < 0 | 8.0% | −5.0% |

A positive premium roughly doubles the runner rate, but the typical pump still fades either way.

### Part B: news, blind (30 runners vs 30 time-matched faders, 3 search agents, 4–10 searches/event)

| any non-price item in the window | runners | faders | diff | Fisher p |
|---|---:|---:|---:|---:|
| published **before** the alert (primary) | 9/30 | 7/30 | +6.7 pp | 0.77 |
| before or same day | 11/30 | 9/30 | +6.7 pp | 0.78 |
| after the alert (≤ 72 h) | 7/30 | 5/30 | +6.7 pp | 0.75 |
| anything in −72 h..+72 h | 15/30 | 14/30 | +3.3 pp | 1.00 |

- Categories before the alert are spread thin in both groups: unlocks, listings, delisting
  notices, partnerships, product posts.
- Runners included two delisting/migration notices (RLS: Binance Futures delisting; TRU:
  migration). These are possible short squeezes on a delisting perp, n = 2.
- **Caveats** (from the agents' reports):
  - Blinding was imperfect: search summaries sometimes volunteered later prices, which were not
    used for labels.
  - Many 2026 dates come from aggregator or AI summaries that occasionally mix up years.
  - Exchange timestamps without time zones were read as UTC.
  - Power: only differences ≥ ~35 pp are detectable at 30 vs 30.

### What it says
- **News does not separate runners from faders.** About half of all pumps have no findable
  catalyst, and runners carry no more pre-alert news than faders. Searchable news does not
  explain the continuation.
- **Listings explain a sliver.** A listing within 3 days after the pump (Upbit, Bithumb,
  Binance spot, Coinbase) is ~10× more common among runners, but it covers only ~3% of them.
  Listings before the alert are no more common among runners.
- **The consistent differences are structure and positioning, and all are small:**
  - Runners are more often futures-only (no Binance spot market to sell into).
  - Runners are a bit younger and were more volatile beforehand.
  - Four hours in, runners show the perp still at a premium and more aggressive buying.
- **Nothing here separates well.** The best tilt (premium > 0 at +4 h) moves the runner rate from
  ~9% to ~15%, and the median still fades.
- Everything is exploratory on consumed windows. `premium_h4` and `has_spot` would be the
  candidates to record live, but adding them is a code change that ends forward test v2.
