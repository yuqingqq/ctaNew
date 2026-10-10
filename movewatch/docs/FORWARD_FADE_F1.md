# Forward test F1: short heavy-volume pumps, +4 h → +24 h (declared 2026-10-10 01:17Z)

**The commit that adds this file is the freeze** (CLAUDE.md rule 12). Evaluator
`research/forward_fade.py`, falsifiers `tests/test_forward_fade.py` (six deliberate code breaks, each caught), design
numbers `research/fade_forward_design.py`. Paper only: nothing is ordered (project scope).

## Why
Signal 2 (post-alert volume) ranks pumps in every window, and shorting the heavy-volume ones is
positive at stress 1 in every calendar year except 2021 (`docs/HEAVY_FADE_V2.md`). But no single
year's CI excludes 0, the edge is about the size of the cost uncertainty, and only the last year
is clearly above cost. All historical windows are consumed for this question, so only forward
data can test it. This is a check, not a validation already in hand.

## Data
- Records of the live recorder frozen for forward test v2 (`docs/FORWARD_TEST_v2.md`, commit
  55ae98d, identity 2375f81d): `data/alerts/followups-*.jsonl`, admitted runs only, the same
  admission as `research/forward_eval.py`.
- F1 covers v2's whole window from its start (2026-10-02 14:40Z): no F1 outcome has been read.
  Before this declaration only counts were read (01:10Z today: 63 pumps in the v2 window,
  11 complete F1 rows, about 1.6 per day).
- F1 ends when v2's identity ends: the first service start with other code or config (outside
  `watchers.py` and the config `watchers` / `notify` sections). A new test then needs a new
  declaration.

## Population (one row per action)
1. NEW pump alerts (`FU_OPEN`, dir +1) in the test window, not `DUPLICATE` (v2 rule).
2. Per asset, the first alert in any 72 h (any direction), as the research rows did
   (`persistence_study.dedupe_72h`); later ones are `REPEAT_72H`.
3. Reference instrument is a Binance USDM perp (`ref_key` = `binance_usdm:…`).
4. Not a meme coin: the frozen CoinGecko list `docs/coingecko_meme_rank1500_2026-10-09.json`
   (171 symbols, base by the research rule). A coin missing from the list counts as non-meme.
5. Heavy: the Binance-only volume ratio at +4 h (`vol_ratio_binance_usdm`, minutes
   [+60, +240) ÷ [−180, 0)) is above 10^0.35259 = 2.2525× (the A2 heavy cut), with ≤ 5 gap
   minutes in each window.

## Trade
- Short at the `FU_4H` price (alert minute + 240 min + 5 s), cover at the `FU_24H` price
  (alert + 24 h). Both must come from the alert's own instrument and be on time (≤ 600 s and
  ≤ 3,600 s late). The research entry was one minute later (close of minute +240).
- Gross = −(p24 / p4 − 1).
- **Cost: flat 0.40% round trip** (0.05% taker fee + 0.15% slippage, per side). At stress 1 the
  research charged 0.145% slippage per side on average for this population.
- **Funding:** the sum of Binance settled funding rates with fundingTime in (t4, t24], fetched at
  the read from `/fapi/v1/fundingRate`. The short receives positive funding and pays negative
  funding. If funding is unavailable, the row gets status `NO_FUNDING` (never counted as zero).
  In the research it averaged −0.78% per trade (shorts paid), larger than the slippage.
- Net = gross − 0.40% + funding.

## Hypothesis
- **H3:** the mean net per trade is above 0. **Pass** = mean > 0 and the 95% CI (ISO-week block
  bootstrap, 4,000 draws, seed 20261010) lies above 0.
- Also reported, no pass/fail: gross mean, funding mean, net at a 1.00% round trip (slippage ×3),
  win rate, median, worst, share losing > 25%, and losses ÷ wins.

## Minimum, reading rule, power
- **Minimum: ≥ 400 complete rows and ≥ 12 complete ISO weeks.** At about 1.6 rows per day that is
  about eight months (around June 2027). The live rate decides.
- Before the minimum, `python3 -m research.forward_fade` prints counts and statuses only.
  `--final` computes the statistics once: it refuses below the minimum or when a receipt exists
  in `data/forward_fade/`. The receipt records this declaration's commit, the recorder's freeze
  and identity, and the evaluator's HEAD.
- **Power at n = 400** (`research/fade_forward_design.py`; cluster-week SE from 2020–26, normal
  approximation): 43% if the true mean is 2026's +1.53%, 9% if it is 2022–25's +0.52%. A pass
  means something. A fail is weak evidence against a small edge.
- Multiplicity: F1 has one hypothesis. It runs beside v2 (H1, H2). The receipt records three
  forward hypotheses open at the read.

## Cost measurement (descriptive, no pass/fail)
- `FadeCostWatcher` (`movewatch/watchers.py`, config `watchers.fade_cost`) runs outside the v2
  identity: the working-tree identity was 2375f81d before and after it was added.
- At every Binance-priced pump's `FU_4H` (entry) and `FU_24H` (exit) record, it fetches that perp's
  order book (500 levels) and premium index within about 10 s. It writes spread, the
  walk-the-book cost of a market sell (entry) and buy (exit) of $1k / $10k / $50k / $100k, depth
  within 0.5 / 1 / 2%, and the current funding rate to `data/alerts/fadecost-*.jsonl`. It sends
  no messages.
- `python3 -m research.forward_fade --cost` prints cost fields only (never a price), so it reveals
  no trade outcome and may be read at any time. The raw snapshot files carry prices: do not read
  them before the final read.
- Snapshots start at the first service start that runs this commit. Rows before that have no
  measured cost.
