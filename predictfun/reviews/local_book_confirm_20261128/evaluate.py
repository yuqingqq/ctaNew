"""Verify saved market-level accounting and grade the one declared development candidate."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from run import HERE,OLD,DATA,DAYS,KEYS,load_day,metrics,digest,HASHES
from score import validate

def main():
    path=OLD/'paired_markets.csv'
    assert digest(path)==HASHES[path]
    blanket=pd.read_csv(path,float_precision='round_trip')
    rows=[]
    for day in DAYS:
        f=pd.read_csv(DATA/HERE.name/day/'paired.csv',float_precision='round_trip')
        validate(f,load_day(day))
        bad=f.copy()
        bad.loc[0,'candidate_adverse']+=1
        try:
            validate(bad,load_day(day))
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupted adverse subtotal accepted')
        cols=KEYS+('mark',)
        sub=blanket.loc[blanket.day.eq(int(day))].copy()
        for key in KEYS:
            sub['blanket_'+key]=sub['alert_'+key]
        for key in ['favorable','adverse','shares','censored']:
            sub['blanket_mark_'+key]=sub['candidate_'+key]
        keys=['day','mid','tape_id']
        f=f.merge(sub[keys+[c for c in sub if c.startswith('blanket_')]],on=keys,how='left',validate='one_to_one')
        assert f.blanket_settle.notna().all()
        rows.append(f)
    f=pd.concat(rows,ignore_index=True)
    result={}
    for group,g in [('pooled',f)]+[(str(day),v) for day,v in f.groupby('day')]:
        result[group]={}
        for label,arm,mark in [('current','current','baseline'),('binance','blanket','blanket_mark'),('confirmed','alert','candidate')]:
            r=metrics(g[arm+'_settle'])
            r.update({key:float(g[arm+'_'+key].sum()) for key in KEYS if key not in ('abs_end','settle')})
            r['mark_1s']=float(100*(g[mark+'_favorable']+g[mark+'_adverse']).sum()/(g[mark+'_shares']-g[mark+'_censored']).sum())
            result[group][label]=r
        delta=g.alert_settle-g.current_settle
        stats={key:int(g[key].sum()) for key in ('public_price_updates','grid_pulls','offgrid_pulls','offgrid_one_of_two','waits')}
        stats.update(better=int((delta>1e-6).sum()),worse=int((delta< -1e-6).sum()))
        def markout(tag,subtract=None):
            keys=['favorable','adverse','shares','censored']
            val={key:float(g[tag+'_'+key].sum()-(g[subtract+'_'+key].sum() if subtract else 0)) for key in keys}
            n=val['shares']-val['censored']
            val['net_c_per_share']=100*(val['favorable']+val['adverse'])/n if n else None
            return val
        stats['timely']=markout('timely_baseline')
        stats['remaining']=markout('baseline','timely_baseline')
        result[group]['diagnostics']=stats
    p=result['pooled']
    diff=[result[d]['confirmed']['total']-result[d]['current']['total'] for d in DAYS]
    result['gates']={'settlement':all(p['confirmed']['total']>p[a]['total'] for a in ['binance','current']),
        'sharpe':all(p['confirmed']['sharpe']>p[a]['sharpe'] for a in ['binance','current']),
        'two_days':sum(x>0 for x in diff)>=2,'without_best_day':sum(diff)-max(diff)>0,
        'less_churn_than_binance':p['confirmed']['requests']<p['binance']['requests']}
    result['passed']=all(result['gates'].values())
    f.to_csv(HERE/'paired_markets.csv',index=False)
    (HERE/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['pooled'],indent=2),result['gates'])
if __name__=='__main__':
    main()
