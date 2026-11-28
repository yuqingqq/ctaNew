"""Results from unchanged controls; classify paired gross losses only once."""
import json
from pathlib import Path
import pandas as pd
from audit import HERE, DATA, DAYS, load_day
from diagnose import DAYS as ARCHIVED

def total(rows,keys):
    return {k:float(rows[k].sum()) if k in rows else 0. for k in keys}

def run():
    arch=pd.concat([load_day(d) for d in ARCHIVED])
    pair=pd.concat([pd.read_csv(DATA/'cancel_loss_clock_20261128'/d/'pairs.csv',float_precision='round_trip') for d in DAYS],ignore_index=True).fillna(0)
    fill=pd.concat([pd.read_csv(DATA/'cancel_loss_clock_20261128'/d/'fills.csv',float_precision='round_trip') for d in DAYS],ignore_index=True).fillna(0)
    ref=arch[arch.day.isin([int(x) for x in DAYS])]
    assert set(ref.mid)==set(fill.mid)
    keys=['shares','net','bad','good','entry','first_second','later','exit','missing_value','missing_shares']
    amount=['shares','valued','gross_bad_1s','gross_good_1s']
    res=dict(archive=dict(n=len(arch),shares=float(arch.event_shares.sum()),post_request=float(arch.event_post_request_shares.sum()),
                  event_minus_grid=float((arch.event_settle-arch.grid_settle).sum())),markets=len(ref))
    for tag,g in [('pooled',pair),*[(str(d),g) for d,g in pair.groupby('day')]]:
        f=fill if tag=='pooled' else fill[fill.day==int(tag)]
        r=ref if tag=='pooled' else ref[ref.day==int(tag)]
        pending_open=g.opening=='cancel_in_flight'
        pending_close=g.closing=='cancel_in_flight'
        masks={'all':g.index==g.index,'opening_pending':pending_open,'closing_only_pending':~pending_open&pending_close,
               'neither_pending':~pending_open&~pending_close,'neither_warning':(g.opening=='no_prior_live_warning')&(g.closing=='no_prior_live_warning'),
               'losing_pairs':g.sign=='losing','winning_or_flat':g.sign=='other','losing_markets':g.mid.isin(r[r.event_settle<0].mid)}
        row={k:total(g[v],keys) for k,v in masks.items()}
        for dim in ('trade','opening','prefill'):
            row[dim]={str(k):total(v,keys) for k,v in g.groupby(dim)}
        row['fills']={str(k):total(v,amount) for k,v in f.groupby('status')}
        row['fills']['all']=total(f,amount)
        assert abs(row['all']['net']-r.event_turnover.sum())<1e-6 and abs(row['all']['bad']-r.event_gross_loss.sum())<1e-6
        assert abs(sum(row[k]['bad'] for k in ('opening_pending','closing_only_pending','neither_pending'))-row['all']['bad'])<1e-6
        row['markets']=len(r)
        res[tag]=row
    (HERE/'results.json').write_text(json.dumps(res,indent=2,allow_nan=False)+'\n')
    print({day:res[day]['markets'] for day in DAYS})
    p=res['pooled']
    print({k:[round(p[k][j],2) for j in keys] for k in masks})
    print('fill split',p['fills'])
    for dim in ('trade','opening','prefill'):
        print(dim,{k:[round(v[j],2) for j in ('shares','net','bad','good')] for k,v in p[dim].items()})
if __name__=='__main__':
    run()
