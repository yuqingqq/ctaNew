# convexity_portable paper-bot — launch guide

## What it is

Forward-test paper bot for the convexity_portable strategy (regime-hybrid
held-book + iter-012 vol-norm stop), running on the full maturity-≥180d
universe at 4h cadence. Records every cycle's predictions, regime classification,
sleeve composition, stop state, and mark-to-market PnL per the schema in
`live/state/convexity/SCHEMA.md`.

**Strategy summary** (from `agents_system/shared/baseline.md` + `current_best.md`):
- 4h decisions; K=5 long / K=5 short; 6 overlapping sleeves (24h hold).
- BULL (BTC 30d > +10%) → trend-follow on mom_30d.
- SIDE → mean-reversion on per-sym Ridge prediction, beta-neutral leg sizing.
- BEAR (BTC 30d < −10%) → flat (no positions).
- Universe: HL∩Binance perps, maturity ≥180d + hygiene + $3M/d liquidity floor.
- Stop: de-gross to 0.40 when DD ≥ 2σ·√180 of trailing-180-bar increments;
  re-enter on 50%-heal or 90-bar timeout.

## Files

- `live/convexity_paper_bot.py` — orchestrator (replay + cycle + state)
- `live/refresh_convexity_panel.py` — data refresh (klines + X132 panel rebuild)
- `live/run_convexity.sh` — tmux supervisor (4h loop)
- `live/train_convexity_artifact.py` — quarterly artifact retrain
- `live/state/convexity/` — all run state + logs (see `SCHEMA.md`)
- `live/models/convexity_portable.pkl` — current deploy artifact (15.6MB; 156 syms;
  XS rank-IC = 1.0000 vs research preds on 200k OOS rows)

## Validation evidence

60-day replay 2026-03-07→2026-05-06 vs research engine (X116 hybrid):
| metric | bot | research | diff |
|---|---|---|---|
| Sharpe (ann) | +1.581 | +1.600 | 0.02 |
| totPnL | +1291 bps | +1307 bps | 1.2% |
| Per-cycle median diff | — | — | 0.000 bps |

The bot reproduces the research backtest to within forward-test noise floor.

## Quick start

### One-time refresh + first batch of cycles (catch up from panel end → today)

```bash
cd /home/yuqing/ctaNew
PYTHONPATH=. python3 -m live.refresh_convexity_panel --days-back 28
PYTHONPATH=. python3 -m live.convexity_paper_bot --cycle    # processes ~125 new cycles
```

The refresh takes ~15-20 min (fetches klines for 156 syms + rebuilds X132 panel).
The cycle run takes ~3 min (compute_mom30 is the bottleneck).

### Launch the supervisor (continuous 4h loop)

```bash
tmux new -d -s convexity 'bash /home/yuqing/ctaNew/live/run_convexity.sh'
tmux ls                       # confirm running
tmux attach -t convexity      # watch live (Ctrl-b d to detach)
tail -f live/state/convexity/run.log
```

The supervisor loops:
1. refresh klines (last 14d) + rebuild X132 panel
2. run `--cycle` to process new cycles since last_open_time (append to logs)
3. sleep until next 4h boundary (UTC: 00, 04, 08, 12, 16, 20)

### Inspect state

```bash
PYTHONPATH=. python3 -m live.convexity_paper_bot --check-state
```

### Quarterly retrain (per iter-033)

```bash
PYTHONPATH=. python3 live/train_convexity_artifact.py
```

Validate the new artifact's XS rank-IC vs research preds on the most recent
fold (should be ≥0.85; current = 1.0000).

## Stopping the supervisor

```bash
tmux kill-session -t convexity
```

State persists in `positions.json` — restart and it resumes from where it left off.

## Resuming after gap

If the bot was stopped for hours/days, the next `--cycle` will catch up all
missed 4h windows in one batch. State machine handles arbitrary gaps.

## Post-trade analysis recipes (per SCHEMA.md)

```python
import pandas as pd
S = "live/state/convexity"
cyc = pd.read_csv(f"{S}/cycles.csv", parse_dates=["open_time"])

# rolling Sharpe (30-cycle window)
import numpy as np
roll = cyc["pnl_bps"]/1e4
sh30 = roll.rolling(30).apply(lambda x: x.mean()/x.std()*np.sqrt(6*365))

# regime attribution
cyc.groupby("regime")["pnl_bps"].agg(["count","mean","sum"])

# cost decomposition
(cyc["cost_bps"]/cyc["gross_pnl_bps"].abs()).describe()

# stop engagement
cyc[cyc["stop_engaged"]==True]
```

## Known caveats

- **Binance Vision lag**: archive publishes ~1-2 days late. For true real-time
  4h cadence we'd need the REST API path (deferred; not wired). Today's loop
  refreshes and processes cycles through ~T−2 days.
- **Liquidity filter**: current $3M/day. The validation script didn't apply this
  (caused per-cycle PnL noise std 30bps); aggregate Sharpe/PnL match is solid.
- **No HL execution**: bot is record-only. Live HL paper or real execution would
  go through `live/hl_executor.py` (used by v6_clean paper bot; not wired here).
- **Beta-neutral leg sizing**: applied only in SIDE regime per spec; bull/bear
  bypass it. Matches X116 engine.

## Failure modes & recovery

- `refresh failed`: bot keeps running on stale panel; next refresh retries.
- `cycle failed`: state not advanced; next cycle reattempts.
- panel rebuild OOMs: revert to incremental rebuild (deferred — currently uses
  X132 full-rebuild ~13min, which has 4h-sample memory fix).
- positions.json corrupted: re-run `--replay 60 && --bootstrap-state` to rebuild.

---

# Two-book forward test (Phase-VII champion, 2026-05-31)

## What changed since the original single-book guide above

Research converged on a **two-book** champion that beats the single book:

| config (K=3, OOS 2025-10-04→2026-05-26) | Sharpe | maxDD |
|---|---|---|
| **two-book (flow BookA + price BookB, 50/50)** | **+3.71** (both-active) / +2.87 (idle-capital) | −1417 |
| flow sleeve alone (flow-syms only) | +3.50 | −2384 |
| one price book, full universe | +3.01 | −4527 |
| unified V0+flow (all syms, one book) | +2.28 | −4549 |

**Phase-VII settled the flow-integration question with full data** (real aggTrade flow for all
175 universe syms): putting flow into ONE unified book is *harmful* (+3.01 → +2.28). Flow and price
predictions correlate ~0.86 on the same universe, so merging adds noise, not signal. The flow edge
only survives as its **own book on its own sub-universe**, combined with a price book at the PnL
level. The two-book's low cross-book correlation (0.17) is the disjoint-universe split, and the
+0.77 Sharpe over a single price book is entirely the flow model in BookA. So the two-book is the
*only* vehicle that captures the flow axis — that's why we run two books, not for diversification.

## Champion config (exact)

- **BookA** — flow model: per-sym RidgeCV on **V0 + 14 flow features**, xs_z target, recency-60,
  monthly walk-forward; universe = flow-eligible syms. `STRAT_K=3`, `SIDE_MODE=default` (model L/S),
  6 sleeves / 24h hold. Preds: `live/state/convexity/flowsub_flow_preds.parquet` (research) →
  regenerate live.
- **BookB** — price model: same machinery on **V0 only**; universe = non-flow syms.
  `STRAT_K=3`, `SIDE_MODE=default`. Preds: `live/state/convexity/priceB_nonflow_preds.parquet`.
- **Combine** — 50/50 PnL via `live/convexity_twobook_combine.py`. Report both bounds:
  *both-active* (+3.71, optimistic — no idle capital) and *fill0* (+2.87, conservative — half the
  book capital sits idle on cycles where one book is flat). **Honest forward expectation: +2.9 to +3.7.**

## Run it

The bot now honors `CONVEXITY_STATE` (per-book state dir) and `CONVEXITY_PREDS_PATH` (per-book preds).

```bash
# one-shot validation (reproduces +3.712 / DD -1417 exactly):
STRAT_K=3 CONVEXITY_PREDS_PATH=live/state/convexity/flowsub_flow_preds.parquet \
  CONVEXITY_STATE=live/state/convexity_bookA python3 -m live.convexity_paper_bot --replay-from 2025-10-04
STRAT_K=3 CONVEXITY_PREDS_PATH=live/state/convexity/priceB_nonflow_preds.parquet \
  CONVEXITY_STATE=live/state/convexity_bookB python3 -m live.convexity_paper_bot --replay-from 2025-10-04
python3 live/convexity_twobook_combine.py \
  --book-a live/state/convexity_bookA/cycles.csv \
  --book-b live/state/convexity_bookB/cycles.csv --out live/state/convexity_twobook

# continuous 4h forward paper test:
tmux new -d -s cvx2 'bash /home/yuqing/ctaNew/live/run_convexity_twobook.sh'
tail -f live/state/convexity_twobook/run.log
```

## Recommended rollout (two phases — be honest about the dependency)

**Phase 1 — start now (no flow dependency):** forward-paper the **price book on the full universe**
(+3.01, the robust core). It uses the existing artifact/pipeline and needs no aggTrade ingestion.
This validates the engine in real time today.
```bash
tmux new -d -s cvx 'STRAT_K=3 bash /home/yuqing/ctaNew/live/run_convexity.sh'
```

**Phase 2 — add the flow book (BookA) once live flow ingestion is wired:** the flow model needs
up-to-date aggTrade features. Two prerequisites, both currently **gaps**:
1. **Daily aggTrade ingestion** — `live/ingest_flow_daily.sh` (cron `0 3 * * *`). KNOWN GAP: the
   builder has no incremental-append mode yet, so it `--force`-rebuilds per sym (heavy). Build an
   incremental appender before scaling the flow universe.
2. **Live preds regeneration** — `run_convexity_twobook.sh` currently replays *frozen* preds forward.
   For a true live two-book, regenerate BookA/BookB preds each cycle from fresh features (the
   `loop2_iter24_unified_fullflow.py` WF generator is the template) and point `BOOKA_PREDS`/
   `BOOKB_PREDS` at the refreshed files in the supervisor's refresh step.

Once both are in place, switch the live book from `cvx` (price-only) to `cvx2` (two-book).

## Monitoring & kill-switch

- **Per-cycle**: `live/state/convexity_twobook/twobook_summary.json` (rolling Sharpe both bounds,
  book corr, per-book Sharpe). Each book's `cycles.csv` / `equity.csv` for drilldown.
- **Health checks**: book PnL corr should stay ≈0.1–0.2 (a jump toward 1.0 means the books
  collapsed onto the same names — diversification gone). BookA active-cycle count dropping means
  flow data is going stale (ingestion broken).
- **Kill-switch** (any trips → flatten, investigate): (a) trailing-30-cycle combined Sharpe < 0 for
  two consecutive weeks; (b) realized maxDD breaches −2500 bps (the validated DD was −1417, so 1.75×);
  (c) BookA flow data > 36h stale (ingestion dead) — fall back to Phase-1 price-only;
  (d) book corr > 0.5 over a trailing month. The bot's built-in vol-norm stop de-grosses to 0.40 at
  ≥2σ DD automatically; the kill-switch is the manual outer layer on top of it.
- **Concentration caveat**: the +3.71 backtest PnL is concentrated in two high-IC months
  (Oct +4100, Apr +2828 of +8782 bps total); per-cycle long-selection skill vs random is weak
  (placebo t≈0.4) — the edge is substantially *structural* (sleeve smoothing + beta-neutral hedge +
  K=3 + flow-quality boost), not sharp name-picking. Expect lumpy returns and long flat stretches;
  judge the forward test over months, not weeks. Universe composition drift will move performance.

## Split rule (Phase-VIII, 2026-05-31) — and an honest forward number

Now that all syms have flow, the flow/price book split is set by a **PIT liquidity rule**:
rank eligible symbols by **trailing-30d dollar volume**, top-**N** → flow book (BookA, V0+flow),
the rest → price book (BookB, V0). Use **N ≈ 70–90**, ranked **statically at each retrain** (NOT
re-ranked every cycle — per-fold re-ranking added churn and *hurt*: +2.11 vs +2.74).

**Honest forward expectation: ~+2.7–2.9 Sharpe, not +3.71.** The thorough study (76 configs +
60 random-split placebos) showed:
- Liquidity is a *mild* real signal (liq70 beats 95% of random same-size splits) and the best
  criterion tested (> flow-quality, > per-fold-dynamic), BUT
- two-book Sharpe is **dominated by composition noise** (random splits span +1.07 to +3.17), and
- **no split beats the single price book (+3.01) on Sharpe** — the two-book's reliable value is
  **drawdown reduction** (~-2900 vs -4527), not return.
- The original +3.71 sat above the *entire* empirical distribution → it was a lucky partition draw.

**Deployment implication:** treat the two-book as a **drawdown-control** structure delivering ~+2.7
forward, OR simply run the **single price book full-universe (+3.01)** if you want one simpler book at
similar Sharpe with higher DD. Do not bank on +3.71. Survivorship audit: OOS window is delisting-clean,
so these numbers aren't survivor-inflated (training tail only, recency-decayed).

## SPLIT RULE UPDATE (system-review loop, 2026-05-31) — volatility beats liquidity
Thorough criterion test (6 pre-registered criteria vs placebo): **high-volatility routing** robustly beats
liquidity. NEW split rule: rank eligible syms by **trailing-30d realized vol (rvol_7d)**, top-N≈**80** →
FLOW book (V0+flow), rest → PRICE book (V0); rank **statically at each retrain** (per-fold re-ranking hurts).
Forward ~**+3.2-3.6** (vs liquidity-split ~+2.7, single-price +3.01). Mechanism: flow microstructure
(vpin/kyle/large-trade) is most informative on high-activity speculative names; price model handles calm
large-caps. hv80 cleared placebo p100 + beats single-book at every N {50-120} → not composition luck.

## FINAL split rule (Phase-IX system-review loop, 2026-05-31)
Thorough 6-criterion test + 12h system review settled the split: **VOLATILITY**, not liquidity.
RULE: rank eligible syms by **trailing-30d realized vol (rvol_7d)**, top-**80** → FLOW book (V0+flow),
rest → PRICE book (V0); rank **statically at each retrain** (per-fold re-ranking hurts). 50/50 combine, K=3.
Forward ~**+3.5 PIT-honest** (vs liquidity-split ~+2.7, single-price +3.01). Validated: flow adds +0.70 Sharpe
specifically on high-vol names; p100 vs placebo across N{50-120}; cost-robust to 12bps. To deploy in
run_convexity_twobook.sh: replace the fixed BookA/BookB symbol lists with the rvol-ranked top-80 split
(recompute the ranking at each retrain from trailing-30d rvol_7d). Everything else unchanged
(K3/HOLD6/HL60/50-50/regime/stop). Recommend also fixing precompute_dvol_cache to trailing-PIT (+0.17 honesty).

## REAL execution slippage (HL L2 orderbook, measured 2026-06-01) — REVISES the forward number
The backtest used a flat 4.5 bps/leg. Measured real HL taker cost (fee + book-walk slippage, l2Book API):
- $5k/leg:  FLOW book (high-vol) total cost med 12.7 bps (p90 25.4); PRICE book med 10.5 (p90 17.4); all fill.
- $25k/leg: FLOW med 29.1 (p90 101), 24/78 NOT fully filled; PRICE med 20.7 (p90 85), 15/74 not filled.
IMPLICATIONS: (1) flat 4.5 bps was ~3× optimistic; real ~10-13 bps even at tiny size → per the cost-sweep
(item 9) honest forward Sharpe ≈ +3.0 (12 bps), NOT +3.5. (2) Flow book genuinely costs more than price book
(high-vol = wider/thinner) — erodes part of the +0.70 flow edge. (3) CAPACITY CEILING: high-vol flow names'
HL books are thin — clean only ~$5k/leg; by $25k a third don't fill. This strategy is SMALL-CAPACITY.
ACTION: forward test wires a per-cycle HL-L2 slippage logger (fetch_hl_l2_book + simulate_taker_fill from
paper_bot.py) → records realized fee+slippage + fully_filled per leg; PnL uses measured cost not flat 4.5.
Probe: /tmp/slippage_probe.py. HONEST forward expectation: ~+3.0 Sharpe at ~$5k/leg, capacity-limited.

## Slippage logger LIVE + size-dependence (validated 2026-06-01)
live/convexity_slippage.py wired into run_convexity_twobook.sh (post-cycle, per book) → logs realized
fee+slippage_bps + fully_filled per leg to live/state/convexity_twobook/slippage.csv. Validated on bookA:
at the bootstrapped ~$38k-book implied ~$12.7k/leg, median total cost ~23 bps (one thin name 61 bps), all
filled. So realistic Sharpe is SIZE-DEPENDENT: ~+3.0 at ~$5k/leg (~12bps), ~+2.2 at ~$12.7k/leg (~23bps),
capacity breaks past ~$25k/leg. RECOMMEND: cap per-leg notional (~$5k) to stay in the clean-cost regime, or
size by per-symbol book depth. The forward test now records realized cost live; judge net-of-real-slippage PnL.

## ============ FORWARD TEST — LIVE (2026-06-01) ============
STATUS: live, advanced to 2026-05-30. Champion = VOLATILITY-SPLIT two-book (rvol_7d top-80→flow book
V0+flow, rest→price book V0; K3/HOLD6/HL60/50-50/regime-prod/stop-prod). Both books bootstrapped over OOS
2025-10-04→05-26 then advanced 05-27→05-30 (+24 cycles). Current two-book Sharpe (modeled cost) +3.25
(bookA flow +3.54, bookB price +0.94, book corr 0.17).

PIPELINE (daily batch — X132 panel rebuild is ~2hr, too slow for 4h):
  live/run_convexity_daily.sh = ingest_flow_daily.py → refresh klines + X132 panel → loop2_iter28_fwd.py
  (forward preds, dynamic CUTS) → rvol-split top-80 → --cycle both books → combine → HL-L2 slippage.
  CRON (install): 30 3 * * *  bash /home/yuqing/ctaNew/live/run_convexity_daily.sh >> live/state/convexity_twobook/daily.log 2>&1

WATCH:
  live/state/convexity_twobook/twobook_summary.json  — rolling Sharpe (both-active + fill0), book corr, per-book Sharpe
  live/state/convexity_twobook/twobook_equity.csv    — combined equity curve
  live/state/convexity_twobook/slippage.csv          — REALIZED per-leg HL cost (fee+book-walk) + fully_filled
  live/state/convexity_book{A,B}/cycles.csv          — per-book cycle detail
  live/state/convexity_twobook/daily.log             — pipeline run log

HONEST FORWARD EXPECTATION (net of REAL slippage, measured 2026-06-01):
  - Modeled-cost (4.5bps) Sharpe ~+3.25; real HL cost ~12bps@$5k/leg, ~22bps@$12.5k/leg → realistic
    SIZE-DEPENDENT Sharpe ~+3.0 (small) to ~+2.2 (current ~$38k book). CAP per-leg ~$5k to stay clean.
  - CAPACITY CEILING: high-vol flow names' HL books are thin (INIT/etc. don't fully fill at $12.5k);
    this is a SMALL-CAPACITY strategy. slippage.csv tracks fully_filled live.
  - Lumpy: edge concentrated in high-IC months; judge over MONTHS not weeks.

KILL-SWITCH (manual outer layer on the bot's vol-stop): flatten + investigate if any of —
  (a) trailing-30-cycle combined Sharpe < 0 for 2 consecutive weeks;
  (b) realized maxDD breaches -2500 bps (validated -1417, so ~1.75×);
  (c) BookA flow data >36h stale (ingest_flow_daily failing) — fall back to price-only;
  (d) book corr > 0.5 over a trailing month (books collapsed onto same names);
  (e) slippage.csv fully_filled=False rate > 30% (capacity breaking → cut size).

CAVEATS: frozen-model between retrains (preds = last-trained per-sym Ridge on fresh features — tests
TRANSPORT; refresh at scheduled retrain); X132 panel rebuild ~2hr (daily-batch not 4h until an incremental
panel builder is added — the main deployment-readiness gap); paper test (no real fills — slippage simulated
from live L2 snapshots at decision time).

## INCREMENTAL panel update (2026-06-01) — fixes the ~2hr rebuild → ~5min, enables 4h cadence
The ~2hr panel rebuild was an artifact: refresh's invalidate_stale_xs_feats() DELETED the xs_feats caches
when klines updated, forcing X132 to recompute compute_kline_features + a slow rolling.apply(autocorr) over
FULL history (~1M bars) for all 175 syms. NOT required — all features are bounded-lookback.
FIX: live/incremental_xs_feats.py recomputes xs_feats over a trailing 60-day WARMUP window (>> max lookback
8640 bars/30d; autocorr 2016/7d) and APPENDS only the new bars. VALIDATED: every V0-relevant column matches
the full rebuild to machine precision (return_1d/atr_pct/vwap_slope_96/bars_since_high/autocorr_pctile_7d=0;
obv_signal=3e-9 — the cumsum offset cancels in obv−obv_ema). Differing cols (obv/obv_ema abs level,
bb_width_pctl_120) are NOT in V0. Timing: ~5min (10 workers) vs ~2hr full rebuild (23×).
PIPELINE: run_convexity_daily.sh now does klines-fetch (--skip-rebuild, no invalidate) → incremental_xs_feats
→ X132 (xs_feats cached+current → skips rebuild). This removes the compute-side blocker to 4h real-time;
remaining gap to true real-time = live REST/ws feed (daily archive lags ~1d). Validate any time:
  python3 live/incremental_xs_feats.py --validate <SYM>

## EFFICIENCY: incremental panel (2026-06-01) — append new bars, memory-safe
live/incremental_panel.py appends ONLY the new 4h bars (windowed build_sym reusing X70.btc_cross/
target_alpha + windowed cohort + xs-rank; build_target_z recomputed cheaply over the full panel). Validated
identical to the full build (only "diff" = flat-coin VINE rvol≈2e-9, a div-by-near-zero artifact; abs diff
2e-9). MEMORY-SAFE: never holds full-history per-sym frames (the 10-worker build_panel_fast OOM'd a 30GB box;
this stays low). xs_feats parquets migrated to row-groups (row_group_size=5000) → 10× faster tail reads.
Daily pipeline (run_convexity_daily.sh) now: klines → incremental_xs_feats(6w) → incremental_panel(6w) →
preds → split → cycle → combine → slippage. build_panel_fast.py kept for full rebuilds (monthly retrain).
MEMORY RULE: keep workers ≤6 (incremental) / ≤4 (full build_panel_fast) on a 30GB box.

## EFFICIENCY: incremental flow ingest (2026-06-01) — Kyle/VPIN only on new bars
ingest_flow_daily.py now runs per_bar_features (incl. the slow per-bar Kyle regression) on NEW daily
aggTrade files ONLY, and reuses the cached per-bar columns (last 2200 bars) as the rolling/VPIN warmup —
instead of recomputing ~16 days of Kyle every cycle. Validated machine-precision vs the full recompute
(kyle/vpin 0.00, signed_volume_z 3e-12). Timing: full universe 29s (no-new) / ~1-1.5min (1 new day),
was ~5min. Same windowing pattern as xs_feats/panel.
