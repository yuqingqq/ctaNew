#!/usr/bin/env bash
# A/B compare BULL_MODE variants on the same OOS window, hysteresis N=3 constant.
# Saves each variant's cycles.csv to a separate file so we can compare side-by-side.
set -uo pipefail
ROOT=/home/yuqing/ctaNew
cd $ROOT
export PYTHONPATH=$ROOT
STATE=$ROOT/live/state/convexity
mkdir -p $STATE/ab_bull

for MODE in mom betaneut_mom sidealpha; do
  echo "=== BULL_MODE=$MODE  hyst N=3 ==="
  BULL_MODE=$MODE REGIME_HYSTERESIS_N=3 python3 -m live.convexity_paper_bot --replay-from 2025-10-04 2>&1 | tail -8
  cp $STATE/cycles.csv $STATE/ab_bull/cycles_${MODE}.csv
  cp $STATE/equity.csv $STATE/ab_bull/equity_${MODE}.csv
  echo
done
echo "=== DONE — comparing results ==="
python3 -c "
import pandas as pd, numpy as np
S='/home/yuqing/ctaNew/live/state/convexity/ab_bull'
rows=[]
for mode in ['mom','betaneut_mom','sidealpha']:
    c=pd.read_csv(f'{S}/cycles_{mode}.csv')
    p=c['pnl_bps']/1e4
    sh=p.mean()/p.std()*np.sqrt(6*365) if p.std()>0 else float('nan')
    cum=pd.Series(c['pnl_bps']).cumsum(); dd=(cum-cum.cummax()).min()
    peak_eq=c['equity_post'].cummax(); dd_pct=((c['equity_post']-peak_eq)/peak_eq).min()*100
    bull=c[c['regime']=='bull']; side=c[c['regime']=='side']
    bsh=(bull['pnl_bps']/1e4).mean()/(bull['pnl_bps']/1e4).std()*np.sqrt(6*365) if (bull['pnl_bps']/1e4).std()>0 else float('nan')
    ssh=(side['pnl_bps']/1e4).mean()/(side['pnl_bps']/1e4).std()*np.sqrt(6*365) if (side['pnl_bps']/1e4).std()>0 else float('nan')
    rows.append(dict(mode=mode,Sharpe=round(sh,3),totPnL=int(c['pnl_bps'].sum()),maxDD=int(dd),maxDD_pct=round(dd_pct,1),pct_pos=round(100*(c['pnl_bps']>0).mean(),1),
        bull_n=len(bull),bull_meanbps=round(bull['pnl_bps'].mean(),1) if len(bull) else 0,bull_Sharpe=round(bsh,2) if len(bull) else 0,
        side_n=len(side),side_meanbps=round(side['pnl_bps'].mean(),1) if len(side) else 0,side_Sharpe=round(ssh,2) if len(side) else 0))
df=pd.DataFrame(rows); print(df.to_string(index=False))
"
