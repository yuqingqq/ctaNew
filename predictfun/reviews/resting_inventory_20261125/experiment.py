"""Run the corrected passive controller on the pinned Oct 26 matched paths."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from policy import Branch, Policy, R

HERE = Path(__file__).resolve().parent
OUT = R.X.DATA / 'resting_inventory_review_20261125'
EVAL = R.EVAL


def read(path):
    return json.loads(path.read_text())


def simulate(path, info, ident, flags, model, payout, enabled):
    b = Branch(path, info['end_rel'], ident)
    policy = Policy(b, flags, model, enabled)
    both = 0
    for k in range(R.LB.n_ticks(b)):
        R.LB.step(b,k,policy)
        g = k*R.BE.STEP
        sides = {s for s in (0,1) for i in b.unclosed[s] if not b.E.o[i]['refused'] and
                 b.E.o[i]['act'] <= g < b.E.o[i]['remove'] and b.E.o[i]['rem'] > R.EPS}
        both += len(sides)==2
    R.LB.finish(b)
    cash,pos,settle = b.E.value(100*payout)
    fills=b.E.fills
    fifo=R.X.fifo(fills,100*payout)
    peak=float(np.max(np.abs(np.cumsum([0.]+[(1-2*s)*q for t,i,s,q,p,it in fills]))))
    assert abs(settle-fifo['completed_c']-fifo['residual_c']) < 1e-5 and peak <= 1000+R.EPS
    assert all(it>=0 and b.E.o[i]['act']<=t<b.E.o[i]['remove'] for t,i,s,q,p,it in fills)
    assert len(b.E.o)==len(policy.proposals)
    pending=sum(x[2]=='post_pending' for x in policy.trims)
    gids={x[1] for x in policy.trims}
    assert len(gids)==len(policy.trims) and gids=={o['gid'] for o in b.E.o if o['why']=='inventory_budget'}
    late_sh=sum(q for t,i,s,q,p,it in fills if i in gids and b.E.o[i]['req']<t)
    result=dict(settle=settle/100,turnover=fifo['completed_c']/100,residual=fifo['residual_c']/100,
                end_pos=pos,peak=peak,shares=sum(f[3] for f in fills),both=both/R.LB.n_ticks(b),
                posts=len(b.E.o),all_pulls=sum(o['req'] is not None for o in b.E.o),
                pressure_pulls=sum(c[7:] == ('PULL','pressure') for c in b.cmd_log),
                first_s=None if policy.opened is None else (ident['start_abs_ms']-ident['opening_ms']+policy.opened*R.BE.STEP)/1000)
    return result,dict(inventory_pulls=len(gids),pending_post_pulls=pending,fill_sh_after_request=late_sh)


def run():
    prior=read(R.HERE/'results_v2.json')
    assert prior['fit_sha']==R.sha(R.HERE/'fit_v2.json')
    fit=read(R.HERE/'fit_v2.json')
    assert fit['census_sha']==R.sha(R.CENSUS)
    idx={r['mid']:r for r in read(R.CENSUS)['rows'] if r['day']==EVAL and r['status']=='complete'}
    assert {f'{EVAL}_{m}.json' for m in idx}==set(prior['inputs'])
    reg=R.MM.registry(EVAL).set_index('mid')
    rg=pd.read_parquet(R.MM.S1/EVAL/'registry.parquet').set_index('mid')
    src=R.MM.KP.Tape()
    off=src.offset(EVAL)
    edges=R.MM.TR.T.P.frozen_edges()
    OUT.mkdir(parents=True,exist_ok=True)
    for hour,group in reg.groupby('hour'):
        mids=[m for m in group.index if m in idx and not (OUT/f'{m}.json').exists()]
        if not mids:
            continue
        for m,st,path,info in R.MM.worlds(EVAL,mids,src,off,reg):
            assert st=='complete' and idx[m]['tape_id']==R.BE.tape_id(path['frames'],path['trades'],path['mark'])
            name=f'{EVAL}_{m}.json'
            saved=R.OUT/'eval'/name
            assert R.sha(saved)==prior['inputs'][name]
            old=read(saved)['arms']['C']['metrics']
            ws=1000.*float(reg.loc[m,'ws'])
            ident=R.MM.ident_of(EVAL,m,ws,info,path)
            flags=R.triggers(R.MM.make_cache(info['t_start'],info['end_rel']),edges)
            args=(path,info,ident,flags,fit['cells'],R.VM.V.payout_of(rg.loc[m]))
            if m==3789484:
                control,counts=simulate(*args,False)
                assert counts['inventory_pulls']==0 and all(abs(control[k]-old[k])<1e-7 for k in control)
            result,counts=simulate(*args,True)
            assert result['first_s']==old['first_s']
            R.write(OUT/f'{m}.json',dict(mid=m,old={k:old[k] for k in result},new=result,counts=counts,
                   baseline_sha=prior['inputs'][name],policy_sha=R.sha(HERE/'policy.py'),runner_sha=R.sha(Path(__file__))))
        print('verified hour',hour,flush=True)


def score():
    prior=read(R.HERE/'results_v2.json')
    rows=[]
    for f in sorted(OUT.glob('*.json')):
        d=read(f)
        assert d['policy_sha']==R.sha(HERE/'policy.py') and d['runner_sha']==R.sha(Path(__file__))
        assert d['baseline_sha']==prior['inputs'][f"{EVAL}_{d['mid']}.json"]
        for arm in ('old','new'):
            row=dict(mid=d['mid'],arm=arm,**d[arm],**{k:(v if arm=='new' else 0) for k,v in d['counts'].items()})
            assert abs(row['settle']-row['turnover']-row['residual'])<1e-5
            rows.append(row)
    frame=pd.DataFrame(rows)
    assert not frame.duplicated(['mid','arm']).any() and len(frame)==2*len(prior['inputs'])
    assert {f'{EVAL}_{m}.json' for m in frame.mid}==set(prior['inputs'])
    frame.to_csv(HERE/'markets.csv',index=False)
    stats={}
    for a,g in frame.groupby('arm'):
        stats[a]={k:float(g[k].sum()) for k in ['settle','turnover','residual','shares','posts','all_pulls','pressure_pulls','inventory_pulls','pending_post_pulls','fill_sh_after_request']}
        stats[a].update(mean_abs_end=float(g.end_pos.abs().mean()),both=float(g.both.mean()),median_peak=float(g.peak.median()))
    for k in ('settle','turnover','residual','shares','mean_abs_end'):
        assert abs(stats['old'][k]-prior['rows']['C'][k])<1e-5
    delta=frame.pivot(index='mid',columns='arm',values='settle').eval('new - old')
    R.write(HERE/'result.json',dict(day=EVAL,n=len(delta),status=prior['status'],arms=stats,
            better=int((delta>1e-7).sum()),worse=int((delta< -1e-7).sum()),equal=int((abs(delta)<=1e-7).sum()),
            improved=stats['new']['settle']>stats['old']['settle'],profitable=stats['new']['settle']>0))
    print(read(HERE/'result.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['run','score'])
    (run if parser.parse_args().command=='run' else score)()
