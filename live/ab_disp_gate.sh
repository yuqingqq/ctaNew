#!/usr/bin/env bash
# A/B: disp-gate ON vs OFF on full OOS, hysteresis N=3 + mom (production default).
set -uo pipefail
ROOT=/home/yuqing/ctaNew; cd $ROOT; export PYTHONPATH=$ROOT
S=$ROOT/live/state/convexity/ab_disp; mkdir -p $S

for GATE in 0 1; do
  if [ "$GATE" = "1" ]; then LABEL="disp_on"; else LABEL="disp_off"; fi
  echo "==== GATE=$GATE ($LABEL) ===="
  DISP_GATE=$GATE BULL_MODE=mom REGIME_HYSTERESIS_N=3 \
    python3 -m live.convexity_paper_bot --replay-from 2025-10-04 --replay-end 2026-05-26 2>&1 | tail -5
  cp $ROOT/live/state/convexity/cycles.csv $S/${LABEL}.csv
done

echo ""
echo "==== DISP-GATE A/B RESULTS ===="
python3 -c "
import pandas as pd, numpy as np
S='/home/yuqing/ctaNew/live/state/convexity/ab_disp'
rows=[]
for lab in ['disp_off','disp_on']:
    c=pd.read_csv(f'{S}/{lab}.csv')
    c['open_time']=pd.to_datetime(c['open_time'],utc=True)
    p=c['pnl_bps']/1e4
    sh=p.mean()/p.std()*np.sqrt(6*365) if p.std()>0 else float('nan')
    cum=pd.Series(c['pnl_bps']).cumsum(); dd=(cum-cum.cummax()).min()
    side=c[c['regime']=='side']; bull=c[c['regime']=='bull']
    ssh=(side['pnl_bps']/1e4).mean()/(side['pnl_bps']/1e4).std()*np.sqrt(6*365) if (side['pnl_bps']/1e4).std()>0 else float('nan')
    bsh=(bull['pnl_bps']/1e4).mean()/(bull['pnl_bps']/1e4).std()*np.sqrt(6*365) if (bull['pnl_bps']/1e4).std()>0 else float('nan')
    # H1 and H2 split
    h1 = c[c['open_time']<'2026-01-22']
    h2 = c[c['open_time']>='2026-01-22']
    sh_h1=(h1['pnl_bps']/1e4).mean()/(h1['pnl_bps']/1e4).std()*np.sqrt(6*365) if (h1['pnl_bps']/1e4).std()>0 else float('nan')
    sh_h2=(h2['pnl_bps']/1e4).mean()/(h2['pnl_bps']/1e4).std()*np.sqrt(6*365) if (h2['pnl_bps']/1e4).std()>0 else float('nan')
    rows.append(dict(variant=lab,n=len(c),Sharpe=round(sh,3),totPnL=int(c['pnl_bps'].sum()),maxDD=int(dd),
        pct_pos=round(100*(c['pnl_bps']>0).mean(),1),
        side_n=len(side),side_meanbps=round(side['pnl_bps'].mean(),2),side_Sharpe=round(ssh,2),
        bull_n=len(bull),bull_Sharpe=round(bsh,2),
        H1_Sharpe=round(sh_h1,2),H2_Sharpe=round(sh_h2,2)))
df=pd.DataFrame(rows); print(df.to_string(index=False))
"
