"""Publish already-declared size/queue summaries before model fitting ends.

No new simulation, selection, fit or hypothesis test. Seven consumed days,
30/50/100 shares, fixed 1,500-share cap, neutral POST/KEEP. Count joint P&L once
per market window, not once per side. Every scenario has identical windows.
Report order retention and gross economic scale separately; no own impact.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pf_entry_repair as P

COUNTS=['orders','accepted','reposts','unknown_initial','filled_orders','multi_fill_orders','exhausted','hard_blocks']
SUMS=COUNTS+['shares','resting_order_ms']


def validate(f,day,markets):
    keys=['day','market','side','size','queue']
    if f.empty or f.duplicated(keys).any() or set(f.day)!={day}:raise ValueError('invalid day/action keys')
    if set(zip(f['size'],f.queue))!={(s,q) for s in P.SIZES for q in ('zero','block')}:raise ValueError('missing scenario')
    for _,g in f.groupby(['size','queue']):
        if set(g.market)!=set(markets) or len(g)!=2*len(markets):raise ValueError('unmatched window population')
        for _,w in g.groupby('market'):
            if set(w.side)!={0,1}:raise ValueError('missing side')
            for c in ('joint_terminal_pnl','max_abs_inventory'):
                if not np.isfinite(w[c]).all() or w[c].nunique()!=1:raise ValueError('inconsistent joint measurement')
    a=f[SUMS+['max_abs_inventory']].to_numpy(float)
    if not np.isfinite(a).all() or (a<0).any():raise ValueError('invalid economic/count fields')
    if (f.multi_fill_orders>f.filled_orders).any() or (f.filled_orders>f.accepted).any() or (f.accepted>f.orders).any():raise ValueError('invalid order counts')
    if (f.reposts>f.orders).any() or (f.shares>f.accepted*f['size']+1e-5).any():raise ValueError('invalid repost/fill accounting')


def summarize(g):
    # Day belongs to the identity: market IDs alone must not collapse across days.
    win=g.drop_duplicates(['day','market']);v={c:float(g[c].sum()) for c in SUMS}
    for c in COUNTS:v[c]=int(v[c])
    n=v['accepted'];filled=v['filled_orders'];shares=v['shares']
    pnl=float(win.joint_terminal_pnl.sum())
    return dict(**v,days=int(g.day.nunique()),windows=len(win),
        gross_terminal_pnl=pnl,gross_terminal_cents_per_filled_share=100*pnl/shares if shares else None,
        multi_fill_fraction_all_orders=v['multi_fill_orders']/n if n else None,
        multi_fill_fraction_filled_orders=v['multi_fill_orders']/filled if filled else None,
        mean_resting_ms=v['resting_order_ms']/n if n else None,
        max_abs_inventory=float(win.max_abs_inventory.max()))


def selftest():
    rows=[]
    for size in P.SIZES:
        for queue in ('zero','block'):
            for market in (1,2):
                for side in (0,1):
                    rows.append(dict(day='20261025',market=market,side=side,size=size,queue=queue,
                        orders=5,accepted=4,reposts=3,unknown_initial=1,filled_orders=2,multi_fill_orders=1,
                        exhausted=1,hard_blocks=0,shares=10.,resting_order_ms=4000.,max_abs_inventory=20.,joint_terminal_pnl=3.))
    f=pd.DataFrame(rows);validate(f,'20261025',[1,2]);s=summarize(f[(f['size']==30)&(f.queue=='zero')])
    assert s['gross_terminal_pnl']==6 and s['shares']==40 and s['multi_fill_fraction_all_orders']==.25
    assert s['gross_terminal_cents_per_filled_share']==15 and s['mean_resting_ms']==1000.
    for bad in (f.iloc[1:].copy(),pd.concat([f,f.iloc[[0]]],ignore_index=True)):
        try:validate(bad,'20261025',[1,2])
        except ValueError:pass
        else:raise ValueError('population falsifier escaped')
    bad=f.copy();bad.loc[0,'joint_terminal_pnl']=10.
    try:validate(bad,'20261025',[1,2])
    except ValueError:pass
    else:raise ValueError('duplicated joint-PnL inconsistency escaped')
    return {'correct_aggregation':True,'missing_side_refused':True,'duplicate_row_refused':True,'inconsistent_joint_pnl_refused':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=P.M.stage(args.out,'summary');controls=selftest();parts=[];receipts={};density=[]
    for day in P.DAYS:
        path=args.base/('validity_'+day);meta=P.N.E.read_json(path/'receipt.json')
        if meta['partial'] or P.M.digest(path/'size_windows.parquet')!=meta['rows_sha256']:raise ValueError('incomplete or changed replay')
        f=pd.read_parquet(path/'size_windows.parquet');validate(f,day,meta['windows']);parts.append(f)
        receipts[day]={'asof':meta['asof'],'receipt_sha256':P.M.digest(path/'receipt.json'),'replay_sha256':meta['rows_sha256'],
                       'source_status':meta['source_status'],'sealed_windows':meta['sealed_lookback_windows']}
        rpath=args.base/('rows_'+day)/'receipt.json'
        if rpath.exists():
            r=P.N.E.read_json(rpath);rp=rpath.parent/'rows.parquet'
            if r['partial'] or P.M.digest(rp)!=r['rows_sha256']:raise ValueError('incomplete/changed dense rows')
            rows=pd.read_parquet(rp,columns=['status','generation','old_selected','sampling_version','inclusion_probability','sampling_weight'])
            e=rows[rows.status=='eligible'];old=e[e.old_selected];w=P.weights(e);oldw=P.N.S.weights(old)
            density.append(dict(day=day,asof=r['asof'],receipt_sha256=P.M.digest(rpath),statuses=r['statuses'],
                old_n=len(old),new_n=len(e),old_later=int((old.generation>0).sum()),new_later=int((e.generation>0).sum()),
                old_effective_n=P.N.S.effective(oldw),new_effective_n=P.N.S.effective(w),all_original_checked=r['original_actions_checked']))
        P.I.check_memory()
    allrows=pd.concat(parts,ignore_index=True)
    byday=[dict(day=d,size=float(s),queue=q,**summarize(g)) for (d,s,q),g in allrows.groupby(['day','size','queue'])]
    pooled=[dict(size=float(s),queue=q,**summarize(g)) for (s,q),g in allrows.groupby(['size','queue'])]
    comparisons=[]
    for q in ('zero','block'):
        control=next(x for x in pooled if x['size']==30. and x['queue']==q)
        for size in (50.,100.):
            target=next(x for x in pooled if x['size']==size and x['queue']==q)
            days=[x for x in byday if x['size']==size and x['queue']==q]
            base={x['day']:x for x in byday if x['size']==30 and x['queue']==q}
            comparisons.append(dict(size=size,queue=q,repost_reduction_fraction=1-target['reposts']/control['reposts'],
                multi_fill_fraction_change=target['multi_fill_fraction_all_orders']-control['multi_fill_fraction_all_orders'],
                pnl_change=target['gross_terminal_pnl']-control['gross_terminal_pnl'],
                cents_per_share_change=target['gross_terminal_cents_per_filled_share']-control['gross_terminal_cents_per_filled_share'],
                days_better_gross_pnl=sum(x['gross_terminal_pnl']>base[x['day']]['gross_terminal_pnl'] for x in days),
                days_better_per_share=sum(x['gross_terminal_cents_per_filled_share']>base[x['day']]['gross_terminal_cents_per_filled_share'] for x in days)))
    result=dict(asof=P.M.now(),controls=controls,days=list(P.DAYS),n_unique_windows=int(allrows[['day','market']].drop_duplicates().shape[0]),
        pooled=pooled,per_day=byday,comparisons=comparisons,density=density,input_receipts=receipts,
        source_sha256=P.M.digest(Path(__file__)),limits=P.LIMITS,
        interpretation='Descriptive predeclared neutral-path size comparison; no test of learned entry, selective keeping, fees, rewards or market impact.',
        prediction_results_available=False)
    P.N.write(out/'result.json',result)
    pd.DataFrame(pooled).to_csv(out/'sizes.csv',index=False);pd.DataFrame(byday).to_csv(out/'size_days.csv',index=False)
    print(json.dumps({'asof':result['asof'],'n_unique_windows':result['n_unique_windows'],'pooled':pooled,'comparisons':comparisons,'density':density},indent=2),flush=True)


if __name__=='__main__':main()
