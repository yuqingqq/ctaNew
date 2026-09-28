# Forward test — declared before any forward data (2026-09-28)

Committed before the service ran the code that records it (CLAUDE.md rule 12). **The commit that
adds this file is the freeze.** `research/forward_eval.py` recomputes the code fingerprint
(sha256 over `movewatch/**/*.py` and the loaded `config/movewatch.yaml`,
`movewatch/followup.py::code_fingerprint`) from that commit's tree and admits only records
written by production runs whose `FU_START` carries the same fingerprint. The test starts at the
first such run and ends at the first later production run with a different fingerprint. Dry runs
are ignored.

## What is recorded (live, `data/alerts/followups-YYYYMMDD.jsonl`; every record carries its `run`)
- `FU_START` at every service start: code fingerprint, dry-run flag, open follow-ups.
- `FU_OPEN` at every NEW alert (pumps and dumps, including those held for a burst digest):
  asset, direction, alert price and the instrument that priced it (`ref_key`), trigger horizon,
  **breadth** at the alert, the asset's instruments (`member_keys`), whether the alert was
  messaged, and a **market snapshot** `mkt_px`: USD prices of the market universe (fresh
  primaries with ≥ $5M/day, the alerted asset excluded).
- `FU_4H` at the alert minute + 240 min (+5 s): price and the instrument that priced it
  (`px_key`), `hold`, and the **post-alert volume ratio**: USD volume in minutes
  [m_a+60, m_a+240) ÷ [m_a−180, m_a), summed over the alert-time instruments that have prices in
  ≥ 90% of the minutes of both windows (a venue that lists or goes dark mid-window would bias
  it). Also recorded: the Binance-USDM-only ratio, gap minutes per window (no price on any used
  instrument), the universe now (`mkt_liq`) and a snapshot covering both universes.
- `FU_24H`, `FU_72H` at alert + 24 h / + 72 h: price, `px_key`, snapshot.
- **Market return** over a window = median log return between two snapshots, over the universe at
  the window's start, ≥ 10 coins. This is the replays' `mkt_median` (point-in-time liquidity,
  coin excluded). The service restarts daily (`RuntimeMaxSec`), and snapshots are not affected by
  restarts.

## Hypotheses (one look each, at the minimum sample)
- **H1 — heavy post-alert volume → pumps fade.** Pumps: Spearman IC of log10(volume ratio) vs the
  market-adjusted return +4 h → +72 h (direction-adjusted). **Pass** = IC < 0 and its 95% CI
  (ISO-week block bootstrap, 2,000 draws) below 0. Also reported: the raw-return IC; the same IC
  on the Binance-USDM-only ratio (rows where it exists; the cutoffs came from Binance-only
  replays); heavy (> 2.26×) and quiet (< 0.94×) groups with n, arithmetic and median, and the
  heavy − quiet differences.
- **H2 — market-wide moves fade more.** All NEW alerts: IC of breadth vs the market-adjusted return
  alert → +4 h (direction-adjusted). **Pass** = IC < 0 and CI below 0.
- **B1 — base rate.** Pumps: market-adjusted return alert → +72 h. Median with its week-block CI,
  and the arithmetic mean. Descriptive, no pass/fail.

The cutoffs come from Apr–Jul (volume ratio) and August (breadth) replays. The forward data is
untouched by both.

## Rows, statuses, sample rules
- A row counts for a statistic only if every record it needs is present, written by an admitted
  run inside the test window, on time (`FU_4H` ≤ 600 s late, `FU_72H` ≤ 3,600 s), priced by the
  alert's own instrument, and its market returns have ≥ 10 coins. H1 also needs a volume ratio
  with ≤ 5 gap minutes in each window. H2 needs a finite breadth.
- Otherwise the row carries the **first failing status** for that statistic: `OTHER_RUN`,
  `NO_BREADTH`, `4H_…` / `72H_…` (`MISSING`, `OTHER_RUN`, `LATE`, `NO_PRICE`,
  `PX_SOURCE_CHANGED`), `VOL_GAP`, `NO_MARKET`. Status counts are printed on every evaluator run.
- **Minimum:**
  - H1 and B1: ≥ 200 complete pump rows.
  - H2: ≥ 300 complete rows.
  - ≥ 5 complete ISO weeks (Monday 00:00 UTC to Monday 00:00 UTC) inside the test window.
  - At the replay alert rate (7–10 NEW/day, about half of them pumps), expect late November to
    December 2026. The live rate decides.
- **Reading rule:** before the minimum is met, only counts and statuses may be read
  (`python3 -m research.forward_eval` prints exactly that). `--final` computes the statistics once.
  It refuses below the minimum or when a receipt already exists. The receipt records the freeze
  commit, the fingerprint and the evaluator's HEAD.
- **No changes during the test:** any code or config change alters the fingerprint and ends the
  test at the next service start. A new test needs a new declaration.
- The +4 h Telegram check goes only to pumps whose NEW alert was messaged, when the check runs on
  time. It does not change what is recorded.
- Multiplicity: two hypotheses (H1, H2); B1 is descriptive.
