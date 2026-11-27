"""Reconcile the nine new and five archived paired panels, and score by UTC day."""
from pathlib import Path
import json
import runpy

import numpy as np
import pandas as pd

from extend import HERE, PRIOR, EARLY, SELECTED, H, census, sha, read, baseline


def panel(g):
    daily=g.groupby('day')[['grid_settle','event_settle']].sum()
    delta=daily.event_settle-daily.grid_settle
    best_delta=float(delta.max())
    r=score(g)
    r.update(days=len(daily),positive_delta_days=int((delta>0).sum()),delta_without_best_day=float(delta.sum()-best_delta),
             directional_screen=bool(delta.sum()>0 and delta.sum()-best_delta>0 and (delta>0).sum()>len(delta)/2))
    signs=np.random.default_rng(20261127).choice([-1,1],size=(4096,len(daily)))
    r['flip_ge']=float(np.mean(signs@delta.to_numpy()>=delta.sum()-1e-8))
    for arm in ('grid','event'):
        pnl=g[arm+'_settle']
        days=daily[arm+'_settle']
        signs_count=int((days>0).sum())
        exday=float(pnl.sum()-days.max())
        exmarket=float(pnl.sum()-pnl.max())
        r.update({arm+'_positive_days':signs_count,arm+'_without_best_day':exday,arm+'_without_best_market':exmarket,
                  arm+'_absolute_screen':bool(pnl.sum()>0 and min(exday,exmarket)>0 and signs_count>len(days)/2),
                  arm+'_winning_markets':int((pnl>1e-7).sum()),arm+'_zero_markets':int((pnl.abs()<=1e-7).sum()),
                  arm+'_market_quantiles':pnl.quantile([0,.1,.25,.5,.75,.9,1]).tolist()})
    return r


lib=runpy.run_path(str(PRIOR/'score.py'))
score,validate=lib['score'],lib['validate']


def run():
    prior=read(PRIOR/'result.json')
    held=read(H.HERE/'result.json')
    blocks=[]
    coverage={}
    hashes={}
    for day in (*EARLY,*H.DAYS):
        earlier=day in EARLY
        f=(HERE if earlier else PRIOR)/f'paired_{day}.csv'
        hashes[day]=sha(f)
        if not earlier:
            assert hashes[day]==prior['files'][f.name]
        g=pd.read_csv(f,float_precision='round_trip')
        if earlier:
            ref=baseline(day)
            coverage[day]=census(day).status.value_counts().to_dict()
        else:
            path=H.HERE/f'markets_{day}.csv'
            assert sha(path)==held['hashes'][path.name]
            ref=pd.read_csv(path,float_precision='round_trip').query("policy=='near' and arm=='nominal'").set_index('mid')
            coverage[day]=held['coverage'][day]
        assert len(g)==coverage[day]['complete'] and sum(coverage[day].values())==288
        assert g.day.eq(int(day)).all() and g.mid.is_unique and dict(zip(g.mid,g.tape_id))==ref.tape_id.to_dict()
        checked=g.set_index('mid').sort_index()
        keys=[c for c in ref.columns if 'grid_'+c in g.columns]
        np.testing.assert_allclose(checked[['grid_'+c for c in keys]],ref.sort_index()[keys],rtol=0,atol=1e-6)
        validate(g)
        blocks.append(g)
    corrupt=blocks[0].iloc[:1].copy()
    corrupt['added_favorable']+=1
    try:
        validate(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted attribution accepted')
    joined=pd.concat(blocks,ignore_index=True)
    assert not joined.duplicated(['mid','day']).any() and len(joined)==1041
    earlier=joined[joined.day.isin([int(d) for d in EARLY])]
    later=joined[~joined.index.isin(earlier.index)]
    tables={k:panel(x) for k,x in [('additional_nine',earlier),('previous_five',later),('all_fourteen',joined)]}
    for k,v in prior['pooled'].items():
        assert np.isclose(tables['previous_five'][k],v,rtol=0,atol=1e-6),k
    original=next(r for r in read(SELECTED/'nine_days.json') if r['arm']=='nominal' and r['policy']=='near')
    assert np.isclose(earlier.grid_settle.sum(),original['settle'],rtol=0,atol=1e-6)
    daily={str(d):score(g) for d,g in joined.groupby('day')}
    output=dict(blocks=tables,daily=daily,coverage=coverage,hashes=hashes)
    (HERE/'result.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print({name:r['difference'] for name,r in tables.items()})


if __name__=='__main__':
    run()
