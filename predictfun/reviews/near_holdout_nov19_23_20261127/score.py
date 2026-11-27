"""Independent ledger readback and the predeclared five-date screens."""
from pathlib import Path
import hashlib
import json
import math
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
DAYS=[f'202611{d}' for d in range(19,24)]


def ledger(f):
    checks=[np.isclose(f.turnover+f.residual,f.settle,rtol=0,atol=1e-6),
            np.isclose(2*f.closed+f.abs_end,f.shares,rtol=0,atol=1e-6),
            (f.abs_end<=f.peak_position+1e-6)&(f.peak_position<=150+1e-6)]
    if not all(c.all() for c in checks):
        raise ValueError('ledger identity or cap failed')


def metrics(g):
    daily=g.groupby('day').settle.sum()
    n=math.ceil(len(g)/10)
    return dict(markets=len(g),positive_days=int((daily>0).sum()),settle=g.settle.sum(),turnover=g.turnover.sum(),
                residual=g.residual.sum(),closed=g.closed.sum(),zero_fills=int((g.shares==0).sum()),mean_end=g.abs_end.mean(),
                tail_end=g.abs_end.nlargest(n).mean(),tail_loss=(-g.settle).nlargest(n).mean(),peak=g.peak_position.max(),
                without_best_market=g.settle.sum()-g.settle.max(),without_best_day=g.settle.sum()-daily.max())


def run():
    frames=[]
    coverage={}
    hashes={}
    for day in DAYS:
        census,market=(HERE/f'{p}_{day}.csv' for p in ('census','markets'))
        for p in (census,market):
            hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        c=pd.read_csv(census)
        assert len(c)==288 and c.opening_ms.is_unique and c.day.eq(int(day)).all()
        coverage[day]={str(k):int(v) for k,v in c.status.value_counts().items()}
        eligible=c[c.status=='complete']
        if not len(eligible):
            assert not market.read_text().strip()
            continue
        f=pd.read_csv(market,float_precision='round_trip',dtype={'day':str})
        assert f.day.eq(day).all() and len(f)==6*len(eligible)
        for arm in ('nominal','short'):
            for policy in ('original','time_only','near'):
                a=f[(f.arm==arm)&(f.policy==policy)]
                assert a.mid.is_unique and dict(zip(a.mid,a.tape_id))==dict(zip(eligible.mid,eligible.tape_id))
                if arm=='short': assert (a.post_request_shares==0).all()
        ledger(f)
        corrupt=f.iloc[:1].copy()
        corrupt['residual']+=1
        try: ledger(corrupt)
        except ValueError: pass
        else: raise AssertionError('corrupt residual passed')
        frames.append(f)
    sufficient=all(c.get('complete',0)>=30 and not c.get('registry_missing',0) for c in coverage.values())
    screens={}
    panels=[]
    if frames:
        all_rows=pd.concat(frames,ignore_index=True)
        for (arm,policy),g in all_rows.groupby(['arm','policy']):
            for label,sample in [('pooled',g),*g.groupby('day')]:
                panels.append(dict(arm=arm,policy=policy,period=label,**metrics(sample)))
        table=pd.DataFrame(panels)
        table.to_csv(HERE/'comparison.csv',index=False)
        for arm,g in table.query("period=='pooled'").groupby('arm'):
            lookup=g.set_index('policy')
            a=lookup.loc['near']; b=lookup.loc['original']; t=lookup.loc['time_only']
            screens[arm]={k:bool(v) for k,v in dict(positive=a.settle>0,without_best_market=a.without_best_market>0,
                              without_best_day=a.without_best_day>0,majority=a.positive_days>len(DAYS)/2,
                              lower_average_inventory=a.mean_end<b.mean_end,lower_tail_inventory=a.tail_end<b.tail_end,
                              no_worse_tail_loss=a.tail_loss<=b.tail_loss,no_worse_settlement=a.settle>=b.settle,
                              no_fewer_winning_days=a.positive_days>=b.positive_days,beats_time_only=a.settle>t.settle).items()}
    passed=sufficient and all(all(s.values()) for s in screens.values()) and len(screens)==2
    result=dict(coverage=coverage,checks=screens,sufficient=sufficient,passes=passed,hashes=hashes)
    (HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    run()
