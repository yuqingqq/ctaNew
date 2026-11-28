"""Loss accounting on the two previously scored, disjoint archive blocks."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BLOCKS = ('event_pressure_nov14_18_20261127', 'event_pressure_nov24_26_20261127')
DAYS = tuple(str(d) for d in (*range(20261114,20261119),*range(20261124,20261127)))


def load_day(day):
    folder = HERE.parent/BLOCKS[day >= '20261124']
    result = json.loads((folder/'result.json').read_text())
    path = folder/f'paired_{day}.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==result['paired_sha256'][day]
    f = pd.read_csv(path,float_precision='round_trip')
    c = pd.read_csv(folder/f'census_{day}.csv')
    ok = c[c.status=='complete']
    assert len(c)==288 and c.opening_ms.is_unique and len(f)==len(ok)
    assert f.day.eq(int(day)).all() and f.mid.is_unique
    assert dict(zip(f.mid,f.tape_id))==dict(zip(ok.mid,ok.tape_id))
    for a in ('grid','event'):
        np.testing.assert_allclose(f[a+'_gross_gain']+f[a+'_gross_loss'],f[a+'_turnover'],rtol=0,atol=1e-6)
        np.testing.assert_allclose(f[a+'_turnover']+f[a+'_residual'],f[a+'_settle'],rtol=0,atol=1e-6)
    return f


def metrics(pnl):
    return dict(n=len(pnl),mean=float(pnl.mean()),sd=float(pnl.std(ddof=1)),
                sharpe=float(pnl.mean()/pnl.std(ddof=1)),total=float(pnl.sum()),
                worst_decile=float(pnl.nsmallest(math.ceil(len(pnl)/10)).mean()),
                positive=int((pnl>1e-7).sum()),flat=int((pnl.abs()<=1e-7).sum()))


def breakdown(f):
    s=f.event_settle
    r=dict(grid=metrics(f.grid_settle),event=metrics(s))
    for name,selected in [('all',f),('losing',f[s< -1e-7]),('winning',f[s>1e-7]),
                          ('worst_decile',f.nsmallest(math.ceil(len(f)/10),'event_settle'))]:
        v=selected
        row={k:float(v['event_'+k].sum()) for k in ('settle','turnover','residual','gross_gain','gross_loss','closed','win_sh','loss_sh','flat_sh','shares')}
        row.update(n=len(v),negative_round_trip_net=int((v.event_turnover< -1e-7).sum()),
                   adverse_residual=int((v.event_residual< -1e-7).sum()),
                   fifo_won_settlement_lost=int(((v.event_turnover>1e-7)&(v.event_settle< -1e-7)).sum()),
                   flat_at_end=int((v.event_abs_end<=1e-7).sum()))
        r[name]=row
    return r


if __name__=='__main__':
    all_rows=pd.concat([load_day(day) for day in DAYS])
    blocks={k:breakdown(g) for k,g in [('earlier_five',all_rows[all_rows.day<20261124]),('later_three',all_rows[all_rows.day>=20261124]),('all_eight',all_rows),*[(str(d),p) for d,p in all_rows.groupby('day')]]}
    (HERE/'baseline.json').write_text(json.dumps(blocks,indent=2,allow_nan=False)+'\n')
    print(json.dumps(blocks['all_eight'],indent=1))
    print([(k,round(blocks[k]['all']['turnover'],1),round(blocks[k]['all']['residual'],1)) for k in DAYS])
