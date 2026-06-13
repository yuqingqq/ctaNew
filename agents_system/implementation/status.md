# implementation status
iteration: iter-031 (universe expansion 70 → ~176)
state: working
updated: 2026-05-27T00:00Z
summary: v2 MEMORY FIX of X132_build_expanded_panel.py. v1 OOM-killed at the concat/train tail (176-sym 5m panel over 2021-2026 ~= 50-100M rows > 30GB box; died right after "loaded closes for 176 syms"). FIX — the strategy decides only every 4h, target is 4h-forward, held-book runs only on 4h-aligned rows, so the panel only needs the 4h decision grid. New per-sym loop: build_sym [5m] → dropna(alpha) → x6.build_target_z [5m, .shift(HORIZON=48) PIT-correct] → sample to 4h grid (open_time.hour%4==0 & minute==0) → downcast float32 → append. Then concat → x6b.build_cohort_fixed (loads its own 5m closes, merges by exact (symbol,open_time) so 4h-subset matches) → bars_since_high_xs_rank → save. ORDERING is critical: target_z must run on 5m BEFORE the 4h sample (shift(48)=4h label gap; on a 4h grid that shift would be 8 days). build_target_z is a pure per-symbol groupby transform → per-symbol-on-5m == full-concat. Memory: stream per-sym, del sdf, del sdfs+gc.collect after concat, log running 4h-row count every 20 syms. Phase A xs_feats unchanged + cache-aware (218 cached on disk, all 176 needed present → Phase A skipped/fast). Added SANITY line: off-grid row count (want 0) + HL70 legs present count (expanded-vs-70 comparison validity). Outputs panel_expanded_v0.parquet (~1-2M rows) + x132_expanded_v0_preds.parquet; touches no validated panels/preds. RE-LAUNCHED in background (nohup, PYTHONPATH set) → /tmp/x132_build2.log, PID in /tmp/x132_pid. Est runtime ~10-30 min (Phase A cached; per-sym build_sym unchanged but 4h sample + float32 keep concat/cohort/target/train tiny → no more OOM). Not run to completion.
blockers: none
background_job:
  cmd: PYTHONPATH=/home/yuqing/ctaNew nohup python3 research/convexity_portable_2026-05-20/scripts/X132_build_expanded_panel.py > /tmp/x132_build2.log 2>&1 &
  log: /tmp/x132_build2.log
  pid_file: /tmp/x132_pid
  monitor: ps -p $(cat /tmp/x132_pid) ; tail -f /tmp/x132_build2.log
  outputs:
    - outputs/vBTC_features/panel_expanded_v0.parquet
    - research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet
