"""Independent readback of each paired row and the frozen directional gate."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'near_holdout_nov19_23_20261127'


def score(g):
    r={}
    for a in ('grid','event'):
        for field in ('settle','turnover','residual','posts','requests','reposts','event_pulls','grid_pressure_pulls','shares','closed','post_request_shares','lead_sum'):
            r[f'{a}_{field}']=g[f'{a}_{field}'].sum().item()
        r[a+'_mean_end']=g[a+'_abs_end'].mean()
        valid=g[a+'_marks_shares'].sum()-g[a+'_marks_censored'].sum()
        r[a+'_mark_c_per_sh']=100*(g[a+'_marks_favorable'].sum()+g[a+'_marks_adverse'].sum()) / valid if valid else None
    for a in ('lost','added'):
        for field in ('shares','favorable','adverse','payout_win','payout_loss','win_sh','loss_sh','flat_sh','censored'):
            r[f'{a}_{field}']=g[f'{a}_{field}'].sum().item()
    r['markets']=len(g)
    r['difference']=r['event_settle']-r['grid_settle']
    return r


def validate(g):
    for a in ('grid','event'):
        np.testing.assert_allclose(g[a+'_settle'],g[a+'_turnover']+g[a+'_residual'],atol=1e-6,rtol=0)
        assert (g[a+'_peak_position']<=150+1e-6).all()
    for mark in ('adverse','favorable','shares','censored'):
        np.testing.assert_allclose(g['event_marks_'+mark]-g['grid_marks_'+mark],g['added_'+mark]-g['lost_'+mark],atol=1e-6,rtol=0)
    np.testing.assert_allclose(g.event_settle-g.grid_settle,g.added_payout_win+g.added_payout_loss-g.lost_payout_win-g.lost_payout_loss,atol=1e-6,rtol=0)


def run():
    chunks=[]
    hashes={}
    for day in range(20261119,20261124):
        p=HERE/f'paired_{day}.csv'
        g=pd.read_csv(p,float_precision='round_trip')
        hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        old=pd.read_csv(PRIOR/f'markets_{day}.csv',float_precision='round_trip').query("arm=='nominal' and policy=='near'")
        assert len(g)==len(old) and g.mid.is_unique and g.day.eq(day).all()
        assert dict(zip(g.mid,g.tape_id))==dict(zip(old.mid,old.tape_id))
        np.testing.assert_allclose(g.set_index('mid').grid_settle.sort_index(),old.set_index('mid').settle.sort_index(),atol=1e-6,rtol=0)
        validate(g)
        chunks.append(g)
    g=pd.concat(chunks)
    corrupt=g.iloc[:1].copy()
    corrupt['added_adverse']+=1
    try:
        validate(corrupt)
    except AssertionError:
        pass
    else:
        raise ValueError('corruption accepted')
    daily={str(d):score(x) for d,x in g.groupby('day')}
    pooled=score(g)
    diff=np.array([r['difference'] for r in daily.values()])
    draws=np.random.default_rng(20261127).choice([-1,1],size=(4096,len(diff)))@diff
    leave_best=float(diff.sum()-diff.max())
    gate=bool(diff.sum()>0 and leave_best>0 and (diff>0).sum()>=3)
    result=dict(daily=daily,pooled=pooled,positive_delta_days=int((diff>0).sum()),delta_without_best_day=leave_best,
                next_shadow_screen=gate,day_sign_flip_ge=float((draws>=diff.sum()-1e-7).mean()),files=hashes)
    (HERE/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Complete:',pooled['markets'],'paired markets. Gate',gate)


if __name__=='__main__':
    run()
