#!/usr/bin/env bash
# Nested-OOS validation: run all 3 BULL_MODE variants on each half of the OOS window.
# Pick winner on first half, validate it wins (or doesn't) on second half.
set -uo pipefail
ROOT=/home/yuqing/ctaNew
cd $ROOT
export PYTHONPATH=$ROOT
S=$ROOT/live/state/convexity/ab_nested
mkdir -p $S

# 215 OOS days split: H1 = 2025-10-04..2026-01-22 (~110d), H2 = 2026-01-22..2026-05-11 (~110d)
H1_START=2025-10-04; H1_END=2026-01-22
H2_START=2026-01-22; H2_END=2026-05-11

for HALF in H1 H2; do
  if [ "$HALF" = "H1" ]; then START=$H1_START; END=$H1_END; else START=$H2_START; END=$H2_END; fi
  echo "==== $HALF ($START -> $END) ===="
  for MODE in mom betaneut_mom sidealpha; do
    echo "  -- BULL_MODE=$MODE --"
    BULL_MODE=$MODE REGIME_HYSTERESIS_N=3 \
      python3 -m live.convexity_paper_bot --replay-from $START --replay-end $END 2>&1 | tail -3
    cp $ROOT/live/state/convexity/cycles.csv $S/${HALF}_${MODE}.csv
  done
done

echo ""
echo "==== NESTED-OOS RESULTS ===="
python3 -c "
import pandas as pd, numpy as np
S='/home/yuqing/ctaNew/live/state/convexity/ab_nested'
rows=[]
for half in ['H1','H2']:
    for mode in ['mom','betaneut_mom','sidealpha']:
        c=pd.read_csv(f'{S}/{half}_{mode}.csv')
        p=c['pnl_bps']/1e4
        sh=p.mean()/p.std()*np.sqrt(6*365) if p.std()>0 else float('nan')
        cum=pd.Series(c['pnl_bps']).cumsum(); dd=(cum-cum.cummax()).min()
        bull=c[c['regime']=='bull']
        bsh=(bull['pnl_bps']/1e4).mean()/(bull['pnl_bps']/1e4).std()*np.sqrt(6*365) if (len(bull)>2 and (bull['pnl_bps']/1e4).std()>0) else float('nan')
        rows.append(dict(half=half,mode=mode,n=len(c),Sharpe=round(sh,3),totPnL=int(c['pnl_bps'].sum()),
            maxDD=int(dd),bull_n=len(bull),bull_Sharpe=round(bsh,2) if len(bull) else 0))
df=pd.DataFrame(rows)
print(df.to_string(index=False))
print()
print('=== Honest interpretation ===')
h1_winner = df[df['half']=='H1'].sort_values('Sharpe',ascending=False).iloc[0]['mode']
h2_winner = df[df['half']=='H2'].sort_values('Sharpe',ascending=False).iloc[0]['mode']
print(f'H1 winner: {h1_winner}    H2 winner: {h2_winner}')
print(f'If pick-on-H1 == H2-winner: NO multiple-testing concern (consistent).')
print(f'If different: the winner is fragile / window-specific.')
"
