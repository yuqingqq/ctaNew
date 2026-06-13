# Implementation handoff — X132 expanded-universe (~176 sym) V0 panel + preds

iteration: iter-031 (universe expansion 70 → ~176, "breadth = edge")
status: **BUILDING-v2** (background job running; should be MUCH faster than v1)

## v2 MEMORY FIX (2026-05-27)

v1 OOM-killed at the concat/train step. A 176-sym panel at 5m over 2021-2026 is
~50-100M rows — too big for the 30GB box. v1 died right after
"loaded closes for 176 syms" (the concat → build_cohort_fixed → build_target_z step).

**The fix:** the strategy DECIDES only every 4h, the target is 4h-forward, and the
held-book backtest runs ONLY on 4h-aligned rows. So the panel only needs the 4h
decision grid, not all 5m bars. We sample each symbol to the 4h grid
(`open_time.hour % 4 == 0 & open_time.minute == 0`) **before** concatenation
(~48× fewer rows → ~1-2M rows) and downcast features to float32.

### Critical ordering (PIT correctness preserved)

`x6.build_target_z` uses `.shift(HORIZON=48)` where 48 is in **5m bars** (= 4h, the
label horizon). It MUST run on the per-symbol **5m** series BEFORE the 4h sample,
else `.shift(48)` over-shifts by 48×4h = 8 days. `build_target_z` is a pure
per-symbol groupby transform, so running it per-symbol on each 5m df is identical to
running it on the full 5m concat. New per-symbol loop:

```
build_sym(sym) [5m]  → dropna(alpha) → build_target_z [5m, shift(48)]
                     → sample to 4h grid → downcast float32 → append
concat → x6b.build_cohort_fixed → bars_since_high_xs_rank → save
```

- `x6b.build_cohort_fixed` runs AFTER concat on the 4h panel. It loads its OWN 5m
  closes, computes rvol_7d/ret_3d/btc_rvol_7d on the 5m grid (288×7 windows), then
  merges by exact `(symbol, open_time)`. 4h rows are a subset of 5m rows → exact
  match, correct values. (Safe on the 4h panel.)
- `bars_since_high_xs_rank` is a contemporaneous cross-sectional rank per `open_time`
  → correct on the 4h grid.
- Memory guards: stream per-symbol, `del sdf`, `del sdfs; gc.collect()` after concat,
  log running 4h-row count every 20 syms.

Phase A (xs_feats rebuild) is unchanged and **cache-aware** — the 176 needed
`data/ml/cache/xs_feats_<sym>.parquet` are already cached (218 on disk), so Phase A
is skipped/fast.

## How to run

```bash
PYTHONPATH=/home/yuqing/ctaNew nohup python3 \
  research/convexity_portable_2026-05-20/scripts/X132_build_expanded_panel.py \
  > /tmp/x132_build2.log 2>&1 &
echo $! > /tmp/x132_pid
```

(Already launched in background as of this handoff.)

## Outputs

- `outputs/vBTC_features/panel_expanded_v0.parquet` — expanded 4h panel (~1-2M rows)
- `research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet`
  — V0 walk-forward preds
- Does NOT touch validated HL70 / panel_3yr_v0 / panel_ext2021_v0 panels or preds.

## Monitor

```bash
tail -f /tmp/x132_build2.log
cat /tmp/x132_pid                              # PID
ps -p $(cat /tmp/x132_pid)                     # still running?
```

Phase markers: `--- A.` xs_feats (cached, fast), `--- B. Build expanded panel
(4h-sampled)` with `4h-rows so far=` progress, `--- C.` cohort/rank, then `--- D.`
V0 preds, then `DONE [Ns]` with the final IC line.

A SANITY line prints after panel save: off-grid row count (want 0) + HL70 legs
present count (want all present so expanded-vs-70 comparison is valid).

Estimated runtime: **~10-30 min** (Phase A cached; per-sym build is the same as v1
but the 4h sample + float32 keeps the panel tiny so concat/cohort/target/train no
longer blow memory). v1 reached "built 160/175" at ~5670s — most of that was
per-sym `build_sym` (xs read + btc_cross + target_alpha on full 5m history). That
cost is unchanged; the savings are at the concat/train tail that previously OOM'd.

## Next (for Evaluation)

Once `DONE` appears: run the V3.1 sleeve / held-book engine on
`x132_expanded_v0_preds.parquet` and compare vs the HL70 baseline. The universe-overfit
concern from MEMORY: V3.1 (N=15/K=3) was calibrated to the 51-panel; expanded breadth
with the corrected (non-clipped) target_z pipeline is the test of whether more names
recover the edge the clip-hacked 111-panel destroyed.
