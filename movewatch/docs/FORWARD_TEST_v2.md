# Forward test v2: declared before any v2 data (2026-10-02)

Supersedes `docs/FORWARD_TEST.md` (v1). No v1 outcome was ever read. At the stop, v1 had 33
alerts with complete rows H2 30, H1 2, B1 2 (counts as of 2026-10-02 14:16Z).

**Why v1 stopped:** the 4-day live check found two problems. Fixing them needs code changes,
and any code change ends v1.
- **Binance USDM discovery:** a 6 s `!miniTicker@arr` window kept only the perps that ticked
  inside it, so restarts kept 347–574 instruments. AIOT, still trading, dropped at a restart.
- **Duplicate rows:** one token under two tickers (`龙虾` / `LONGXIA`) produced one move as two
  NEW alerts.

v1 records stay on disk as a pilot. Under v2 they carry status `OTHER_RUN`.

**The commit that adds this file is the v2 freeze.** Everything in `docs/FORWARD_TEST.md` holds
unless changed below: records, market snapshots, hypotheses H1/H2/B1 with the same definitions,
cutoffs, CIs, minimum sample, reading rule and statuses.

## Code changes in the v2 freeze
- **Binance USDM universe:** every USDT/USDC perp on `!markPrice@arr@1s`, minus those the
  10 s `!miniTicker@arr` window shows below $100k/day.
  - A symbol absent from the window is kept unless markPrice shows it settled: mark == index
    with zero funding, or no next funding time.
  - First run 2026-10-02: 958 on markPrice → 784 kept, 128 settled dropped (347 before the fix).
- **Aliases** (config `aliases`, alias → canonical, applied in `instruments.make`):
  PUMPFUN→PUMP, LONGXIA→龙虾, NIULAI→牛来, QUANT→QNT, RAYSOL/RAYDIUM→RAY, PROS→PHAROS. Evidence:
  same USD price within 1% and 1-min return correlation 0.946–0.991 over a day.
- **Tokenized stocks:** Binance spot lists stocks as ticker + "B" (NVDAB, TSLAB…: 71 on
  2026-10-02, each within 1% of its stock). These are now excluded by price match.
- **Messages:** the venue-agreement line appears on NEW only. On ESCALATE it compared the 1 h
  window, which still held the move before the episode.
- **Fingerprint scope:** `watchers.py` and the config `watchers` / `notify` sections are outside
  the identity. They change messages, never alert or follow-up records, so Upbit-flag tuning does
  not end the test. The evaluator recomputes the identity with the `fingerprint()` function of
  the freeze commit.
- **New recorded fields** (for the BTC-hedge question; no hypothesis yet):
  - `btc_px` / `btc_key` on `FU_OPEN`, `FU_ENTRY`, `FU_4H`, `FU_24H`, `FU_72H`. It is the same
    BTC instrument for the whole follow-up.
  - `beta_btc`, `corr_btc`, `n_beta` at `FU_OPEN`: OLS beta of the reference instrument's 5-min
    log returns on BTC's, minutes [m_a−1440, m_a−60), with at least 100 returns. The window is
    clipped to minutes still in the 25 h ring, so it never reads wrapped slots.
  - `FU_ENTRY` at alert + 60 s: price, `px_key` and BTC. This is the price a person acting on
    the message could get.

## Added status
- `DUPLICATE`: a later admitted `FU_OPEN` with another asset name, the same direction, ≤ 5 min
  after an earlier one and an alert price within 1% of it. Rows are actions (CLAUDE.md rule 2),
  and this is the backstop for aliases not yet in the map. It comes right after `OTHER_RUN` in
  every hypothesis's status order.

## Amendment rule
A hypothesis may be **added**, never changed or removed, by a committed amendment
(`docs/FORWARD_TEST_v2_amend_N.md`). The amendment must be committed before the first `--final`
read and use only fields recorded above. The final read reports multiplicity as the number of
declared hypotheses at that time. A BTC-hedged pump-fade hypothesis is expected as amendment 1,
designed on historical replay data (consumed windows) and validated only here.
