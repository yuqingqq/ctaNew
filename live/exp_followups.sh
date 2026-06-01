#!/usr/bin/env bash
# Three follow-up experiments:
#   (i)   Rerun EXP 2 (rolling retrain) with the Feb 29 date bug FIXED
#   (ii)  Per-month Sharpe breakdown of static-baseline vs rolling-retrain
#   (iii) EXP 3: recency-weighted Ridge (halflife = 30d / 90d / 270d) trained
#         through 2026-01-22 and tested on H2 — direct comparison to monthly retrain
#
set -uo pipefail
ROOT=/home/yuqing/ctaNew; cd $ROOT; export PYTHONPATH=$ROOT
S=$ROOT/live/state/convexity

# Helper: last-day-of-month-correct cutoffs
declare -a CUTOFFS=(
  "2025-11-30:2025-12-01:2025-12-31"
  "2025-12-31:2026-01-01:2026-01-31"
  "2026-01-31:2026-02-01:2026-02-28"   # FIX: 2026-02 has 28 days
  "2026-02-28:2026-03-01:2026-03-31"
  "2026-03-31:2026-04-01:2026-04-30"
)

# ============ (i) Rolling retrain — FIXED Feb date ============
echo "==== (i) Rolling retrain (fixed dates) — train+predict each month ===="
for entry in "${CUTOFFS[@]}"; do
  IFS=':' read -r cutoff predict_start predict_end <<< "$entry"
  tag="rfix_$(echo $cutoff | tr -d '-')"
  echo "  -- train through $cutoff, predict $predict_start → $predict_end --"
  python3 live/train_convexity_artifact.py --train-end $cutoff --tag $tag 2>&1 | tail -2
  python3 live/predict_with_artifact.py --artifact $tag --from $predict_start --to $predict_end \
    --out-tag ${tag}_pred 2>&1 | tail -1
done

echo ""
echo "==== (i) stitch rolling preds ===="
python3 -c "
import pandas as pd, glob
from pathlib import Path
S = Path('/home/yuqing/ctaNew/live/state/convexity')
files = sorted(S.glob('x132_rfix_*_pred_preds.parquet'))
print(f'  found {len(files)} files')
dfs = []
for f in files:
    d = pd.read_parquet(f); print(f'    {f.name}: {len(d):,} rows')
    dfs.append(d)
if dfs:
    out = pd.concat(dfs, ignore_index=True).sort_values(['open_time','symbol'])
    out.to_parquet(S/'x132_rfix_stitched_preds.parquet')
    print(f'  stitched: {len(out):,} rows, {out[\"symbol\"].nunique()} syms, '
          f\"{out['open_time'].min()} → {out['open_time'].max()}\")
"

echo ""
echo "==== (i) replay rolling-retrain preds ===="
BULL_MODE=mom REGIME_HYSTERESIS_N=3 \
  CONVEXITY_PREDS_PATH=$S/x132_rfix_stitched_preds.parquet \
  python3 -m live.convexity_paper_bot --replay-from 2025-12-01 --replay-end 2026-04-30 2>&1 | tail -3
cp $S/cycles.csv $S/followup_rolling.csv

# ============ Run static baseline on SAME 5-month window for fair compare ============
echo ""
echo "==== (ii) static baseline same Dec→Apr window ===="
BULL_MODE=mom REGIME_HYSTERESIS_N=3 \
  python3 -m live.convexity_paper_bot --replay-from 2025-12-01 --replay-end 2026-04-30 2>&1 | tail -3
cp $S/cycles.csv $S/followup_static.csv

# ============ (iii) EXP 3: recency-weighted Ridge ============
echo ""
echo "==== (iii) Recency-weighted Ridge experiments ===="
for HL in 30 90 270; do
  echo "  -- halflife = ${HL}d --"
  python3 live/train_convexity_artifact.py --train-end 2026-01-22 \
    --tag recw_hl${HL} --halflife-days $HL 2>&1 | tail -2
  python3 live/predict_with_artifact.py --artifact recw_hl${HL} --from 2026-01-22 --to 2026-05-11 \
    --out-tag recw_hl${HL}_h2 2>&1 | tail -1
  echo "  -- replay --"
  BULL_MODE=mom REGIME_HYSTERESIS_N=3 \
    CONVEXITY_PREDS_PATH=$S/x132_recw_hl${HL}_h2_preds.parquet \
    python3 -m live.convexity_paper_bot --replay-from 2026-01-22 --replay-end 2026-05-11 2>&1 | tail -2
  cp $S/cycles.csv $S/followup_recw_hl${HL}.csv
done

# ============ FINAL SUMMARY ============
echo ""
echo "============ FINAL SUMMARY ============"
python3 -c "
import pandas as pd, numpy as np
S='/home/yuqing/ctaNew/live/state/convexity'
def stats(path, label):
    c = pd.read_csv(path); c['open_time'] = pd.to_datetime(c['open_time'], utc=True)
    p = c['pnl_bps']/1e4
    sh = p.mean()/p.std()*np.sqrt(6*365) if p.std()>0 else float('nan')
    cum = pd.Series(c['pnl_bps']).cumsum(); dd = (cum-cum.cummax()).min()
    side = c[c['regime']=='side']
    side_sh = (side['pnl_bps']/1e4).mean()/(side['pnl_bps']/1e4).std()*np.sqrt(6*365) if (side['pnl_bps']/1e4).std()>0 else float('nan')
    return dict(label=label, n=len(c), Sh=round(sh,3), totPnL=int(c['pnl_bps'].sum()),
                maxDD=int(dd), side_Sh=round(side_sh,3),
                start=str(c['open_time'].iloc[0])[:10], end=str(c['open_time'].iloc[-1])[:10])

# (i)+(ii): rolling vs static on same Dec→Apr window
print('=== Dec→Apr window: rolling-retrain vs static-baseline ===')
print(stats(f'{S}/followup_rolling.csv','ROLLING monthly (Dec→Apr)'))
print(stats(f'{S}/followup_static.csv', 'STATIC baseline (Dec→Apr)'))

# (ii) per-month breakdown
print('\\n=== Per-month breakdown ===')
for name,path in [('rolling',f'{S}/followup_rolling.csv'), ('static',f'{S}/followup_static.csv')]:
    c=pd.read_csv(path); c['open_time']=pd.to_datetime(c['open_time'],utc=True)
    c['month'] = c['open_time'].dt.to_period('M').astype(str)
    mm = c.groupby('month').agg(n=('pnl_bps','count'),
        pnl=('pnl_bps','sum'),
        sh=('pnl_bps', lambda x: (x/1e4).mean()/(x/1e4).std()*np.sqrt(6*365) if (x/1e4).std()>0 else float('nan')))
    print(f'  {name}:'); print(mm.round(2).to_string())

# (iii) recency-weighted on H2 window
print('\\n=== Recency-weighted Ridge — same H2 window as original baseline (Jan22→May11) ===')
for hl in [30, 90, 270]:
    print(stats(f'{S}/followup_recw_hl{hl}.csv', f'RECW halflife={hl}d (H2)'))
print('  (baseline reference H2 static: Sharpe ≈ -0.36 to -2.57 depending on window)')
"
