"""Chronological replication of the frozen urgency controller on consumed Oct 27."""
from collections import Counter
import json
import math
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'inventory_urgency_20261125'
sys.path.insert(0, str(PRIOR))
import urgency as U

R = U.R
DAY = '20261027'
ARMS = ('reference', 'retain', 'urgency')


def cohort(frame):
    return dict(markets=len(frame), settlement=frame.settle.sum(), turnover=frame.turnover.sum(),
                residual=frame.residual.sum(), mean_abs_end=frame.abs_end.mean() if len(frame) else None)


def diagnose(markets, inventory):
    out = {}
    wide = markets.pivot(index='mid', columns='arm', values='settle')
    assert not wide.isna().any().any()
    for arm in ARMS:
        g = markets[markets.arm == arm]
        out[arm] = dict(**cohort(g), settle=g.settle.sum(), closed=g.closed.sum(),
                        per_matched=100*g.turnover.sum()/g.closed.sum(),
                        p90_end=g.abs_end.quantile(.9), max_end=g.abs_end.max(),
                        positive=int((g.settle>0).sum()), pulls=int(g.all_pulls.sum()),
                        both=g.both.mean())
    g = markets[markets.arm == 'urgency'].merge(inventory[inventory.arm == 'urgency'],
                                                on=['mid','arm'], validate='one_to_one')
    assert len(g) == len(wide) and (abs(g.abs_end - g.abs_2s)<1e-6).all()
    over = g.abs_60s > U.S.curve(60)+R.EPS
    largest = g.sort_values(['abs_end','mid'], ascending=[False,True]).head(math.ceil(len(g)/10))
    out['inventory'] = dict(over_60=cohort(g[over]), within_60=cohort(g[~over]),
                            tail=cohort(largest), tail_over_60=int(largest.mid.isin(g.mid[over]).sum()))
    out['paired'] = {}
    for arm in ('reference','retain'):
        diff = wide.urgency-wide[arm]
        out['paired'][arm] = dict(delta=diff.sum(), better=int((diff>1e-6).sum()),
                                  worse=int((diff< -1e-6).sum()), without_largest_gain=diff.sum()-diff.max())
    out['gate'] = {k:bool(v) for k,v in U.P.gate(out['reference'],out['urgency']).items()}
    return out


def run():
    prior = U.S.read(PRIOR/'result.json')
    assert R.sha(PRIOR/'urgency.py') == prior['source']
    for name, key in [('markets','markets_sha'), ('inventory','inventory_sha')]:
        assert R.sha(PRIOR/(name+'.csv')) == prior[key]
    old = pd.read_csv(PRIOR/'markets.csv',float_precision='round_trip')
    old_inv = pd.read_csv(PRIOR/'inventory.csv',float_precision='round_trip')
    fit_path = R.HERE/'fit_v2.json'
    fit = U.S.read(fit_path)
    assert R.sha(R.CENSUS) == fit['census_sha']
    all_rows = U.S.read(R.CENSUS)['rows']
    src = R.MM.KP.Tape()
    edges = R.MM.TR.T.P.frozen_edges()
    rows, samples = [], []
    counts = {}
    for day in (R.EVAL,DAY):
        pop = [r for r in all_rows if r['day']==day]
        idx = {r['mid']:r['tape_id'] for r in pop if r['status']=='complete'}
        reg = R.MM.registry(day).set_index('mid')
        assert len(pop)==len(reg)==len(set(r['mid'] for r in pop)) and set(reg.index)=={r['mid'] for r in pop}
        counts[day] = dict(Counter(r['status'] for r in pop))
        if day==R.EVAL:
            assert set(idx)==set(old.mid)
            idx = {min(idx):idx[min(idx)]}
        rg = pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
        off = src.offset(day)
        seen = set()
        for hour,group in reg.groupby('hour'):
            ids = sorted(set(group.index)&idx.keys())
            if not ids:
                continue
            for m,status,path,info in R.MM.worlds(day,ids,src,off,reg):
                assert status=='complete' and m not in seen
                seen.add(m)
                ident = R.MM.ident_of(day,m,1000.*reg.loc[m,'ws'],info,path)
                assert ident['tape_id']==idx[m]
                on = R.triggers(R.MM.make_cache(info['t_start'],info['end_rel']),edges)
                payout = R.VM.V.payout_of(rg.loc[m])
                assert payout in (0,1)
                args = path,info,ident,on,fit['cells'],payout
                for arm in ARMS:
                    if arm=='reference':
                        r, s = U.D.simulate(*args,False), {}
                    else:
                        r, s = U.one(*args,arm)
                    if day==R.EVAL:
                        for record, data in [(r,old),(s,old_inv)]:
                            if record:
                                match=data[(data.mid==m)&(data.arm==arm)].iloc[0]
                                assert all(abs(match[k]-v)<1e-6 for k,v in record.items()), (m,arm)
                    else:
                        rows.append(dict(mid=m,arm=arm,**r))
                        if s:
                            samples.append(dict(mid=m,arm=arm,**s))
            print(day,hour,len(seen),flush=True)
        assert seen==set(idx)
    markets, inventory = pd.DataFrame(rows), pd.DataFrame(samples)
    summary = {R.EVAL:diagnose(old,old_inv), DAY:diagnose(markets,inventory)}
    for day in (R.EVAL,DAY):
        assert counts[day]['complete']==summary[day]['urgency']['markets']
    summary.update(coverage=counts, controller_sha=prior['source'], fit_sha=R.sha(fit_path), census_sha=fit['census_sha'])
    contents=json.dumps(summary,allow_nan=False,indent=2)
    markets.to_csv(HERE/'markets.csv',index=False,mode='x')
    inventory.to_csv(HERE/'inventory.csv',index=False,mode='x')
    with (HERE/'result.json').open('x') as f:
        f.write(contents+'\n')


if __name__=='__main__':
    run()
