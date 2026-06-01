#!/usr/bin/env bash
# Two experiments to settle the regularization & rolling-retrain questions.
#
#   EXP 1: extended alpha grid {1, 10, 100, 1000, 10000} — does more reg help H2?
#   EXP 2: rolling retrain (5 monthly cutoffs) — does fresh data eventually catch up?
#
# All within per-sym Ridge framework (no architectural change to model class).
set -uo pipefail
ROOT=/home/yuqing/ctaNew; cd $ROOT; export PYTHONPATH=$ROOT
S=$ROOT/live/state/convexity

# ============ EXP 1: extended alpha grid ============
echo "==== EXP 1: train artifact with extended alpha grid ===="
python3 live/train_convexity_artifact.py \
  --train-end 2026-01-22 --tag val_h1_xa --alphas "1,10,100,1000,10000" 2>&1 | tail -8
echo ""

echo "==== EXP 1: generate H2 preds with extended-alpha artifact ===="
python3 live/predict_with_artifact.py --artifact val_h1_xa --from 2026-01-22 --to 2026-05-11 \
  --out-tag val_h1_xa_h2 2>&1 | tail -3
echo ""

echo "==== EXP 1: replay H2 with extended-alpha preds ===="
BULL_MODE=mom REGIME_HYSTERESIS_N=3 \
  CONVEXITY_PREDS_PATH=$S/x132_val_h1_xa_h2_preds.parquet \
  python3 -m live.convexity_paper_bot --replay-from 2026-01-22 --replay-end 2026-05-11 2>&1 | tail -3
cp $S/cycles.csv $S/exp1_xa_h2.csv

# ============ EXP 2: rolling retrain ============
echo ""
echo "==== EXP 2: rolling monthly retrain ===="
declare -a CUTOFFS=("2025-11-30:2025-12-01:2025-12-31"
                    "2025-12-31:2026-01-01:2026-01-31"
                    "2026-01-31:2026-02-01:2026-02-29"
                    "2026-02-28:2026-03-01:2026-03-31"
                    "2026-03-31:2026-04-01:2026-04-30")

for entry in "${CUTOFFS[@]}"; do
  IFS=':' read -r cutoff predict_start predict_end <<< "$entry"
  tag="rolling_$(echo $cutoff | tr -d '-')"
  echo "  -- train through $cutoff, predict $predict_start → $predict_end --"
  python3 live/train_convexity_artifact.py --train-end $cutoff --tag $tag 2>&1 | tail -3
  python3 live/predict_with_artifact.py --artifact $tag --from $predict_start --to $predict_end \
    --out-tag ${tag}_pred 2>&1 | tail -2
done

# Stitch rolling preds
echo ""
echo "==== EXP 2: stitch rolling preds into one file ===="
python3 -c "
import pandas as pd
from pathlib import Path
S = Path('/home/yuqing/ctaNew/live/state/convexity')
files = sorted(S.glob('x132_rolling_*_pred_preds.parquet'))
print(f'  found {len(files)} rolling preds files')
dfs = []
for f in files:
    d = pd.read_parquet(f); print(f'    {f.name}: {len(d):,} rows')
    dfs.append(d)
if dfs:
    out = pd.concat(dfs, ignore_index=True).sort_values(['open_time','symbol'])
    out.to_parquet(S/'x132_rolling_stitched_preds.parquet')
    print(f'  stitched: {len(out):,} rows, {out[\"symbol\"].nunique()} syms, '
          f\"{out['open_time'].min()} → {out['open_time'].max()}\")
"

echo ""
echo "==== EXP 2: replay stitched-rolling preds on the H2 window covered ===="
BULL_MODE=mom REGIME_HYSTERESIS_N=3 \
  CONVEXITY_PREDS_PATH=$S/x132_rolling_stitched_preds.parquet \
  python3 -m live.convexity_paper_bot --replay-from 2025-12-01 --replay-end 2026-04-30 2>&1 | tail -3
cp $S/cycles.csv $S/exp2_rolling.csv

# ============ Summary ============
echo ""
echo "==== SUMMARY OF BOTH EXPERIMENTS ===="
python3 -c "
import pandas as pd, numpy as np
S = '/home/yuqing/ctaNew/live/state/convexity'
def stats(path, label, window=None):
    c = pd.read_csv(path); c['open_time'] = pd.to_datetime(c['open_time'], utc=True)
    if window: c = c[(c['open_time']>=pd.Timestamp(window[0],tz='UTC')) & (c['open_time']<=pd.Timestamp(window[1],tz='UTC'))]
    p = c['pnl_bps']/1e4
    sh = p.mean()/p.std()*np.sqrt(6*365) if p.std()>0 else float('nan')
    cum = pd.Series(c['pnl_bps']).cumsum(); dd = (cum-cum.cummax()).min()
    side = c[c['regime']=='side']
    side_sh = (side['pnl_bps']/1e4).mean()/(side['pnl_bps']/1e4).std()*np.sqrt(6*365) if (side['pnl_bps']/1e4).std()>0 else float('nan')
    return dict(label=label, n=len(c), Sharpe=round(sh,3), totPnL=int(c['pnl_bps'].sum()), maxDD=int(dd), side_Sharpe=round(side_sh,3))

rows = []
# baseline references (recompute on standard windows for fair compare)
rows.append({'label': 'BASELINE original artifact (H2)', 'note':'-0.36 Sharpe reference from prior runs'})
rows.append(stats(f'{S}/exp1_xa_h2.csv', 'EXP1 extended-alpha H2'))
rows.append(stats(f'{S}/exp2_rolling.csv', 'EXP2 rolling-retrain (Dec→Apr)'))

for r in rows:
    print(r)
"
