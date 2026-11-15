"""Audit completed E2SEP sensitivity and strict certificate invariance.

Post hoc, descriptive, nine consumed days. Recompute all three population
means, preserve statuses, prove zero-shift identity and check strict-certified
actions retain key/time/status/value under both shifts. Different scenario
populations are not paired causal effects. No fitting/raw-tape reads.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import resource

import numpy as np
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew/data/predict_fun/research')
DAYS=('20261023','20261024','20261025','20261026','20261027','20261031','20261101','20261102','20261103')
KEY=['day','m','side','rung','generation','decision_ms']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def require(ok,msg):
    if not bool(ok):raise ValueError(msg)


def match(a,b):
    require(not a.duplicated(KEY).any() and not b.duplicated(KEY).any(),'duplicate action')
    j=a.merge(b,on=KEY,how='outer',validate='one_to_one',suffixes=('','_b'),indicator=True)
    require((j._merge=='both').all(),'zero-shift keys differ')
    require((j.status==j.status_b).all(),'zero-shift statuses differ')
    require(np.allclose(j.value_cents,j.value_cents_b,rtol=0,atol=1e-9,equal_nan=True),'zero-shift labels differ')


def run(out):
    src=ROOT/'stop_ev_e2sep_20261115';hashes={};panels={d:[] for d in (-250,0,250)};parity={};ctl={};statuses={}
    def read(p,cols=None):
        hashes[str(p)]=sha(p)
        return pd.read_parquet(p,columns=cols) if p.suffix=='.parquet' else json.loads(p.read_text())
    original_report=read(src/'e2sep.json')
    for day in DAYS:
        rows={d:read(src/f'e2_{day}_{d}.parquet') for d in panels}
        a=read(ROOT/f'stop_ev_20261115/actions_{day}_all_anchors_sensitivity.parquet',KEY+['status','value_cents'])
        match(a,rows[0][a.columns])
        if not ctl:
            for name,b in [('duplicate',pd.concat([rows[0],rows[0].iloc[:1]])),('time',rows[0].assign(decision_ms=rows[0].decision_ms+1.))]:
                try:match(a,b[a.columns])
                except ValueError:ctl[name]=True
                else:ctl[name]=False
            bad=rows[0].copy();ix=bad.index[bad.status.str.startswith('eligible')][0];bad.loc[ix,'value_cents']+=1.
            try:match(a,bad[a.columns])
            except ValueError:ctl['label']=True
            else:ctl['label']=False
            require(all(ctl.values()),'bad parity input accepted')
        cdir=ROOT/'stop_ev_timing_certified_20261115_v1'/day;cr=read(cdir/'receipt.json')
        cp=cdir/'certified_ws.parquet';c=read(cp,KEY+['status','value_cents'])
        require(hashes[str(cp)]==cr['artifacts'][cp.name],'certificate hash changed')
        c.day=c.day.astype('int64');c=c[c.status.str.startswith('eligible')]
        # Some certified rows are refused by the original empirical untimed
        # mask; compare only actions that this experiment admits at zero.
        c=c.merge(rows[0][KEY+['status']],on=KEY,how='left',validate='one_to_one',suffixes=('','_zero'))
        c=c[c.status_zero.str.startswith('eligible')];parity[day]={};statuses[day]={}
        for d,x in rows.items():
            require(not x.duplicated(KEY).any(),'scenario duplicate')
            j=c[KEY+['value_cents']].merge(x[KEY+['status','value_cents']],on=KEY,how='left',validate='one_to_one',suffixes=('','_shift'),indicator=True)
            good=(j._merge=='both') & j.status.str.startswith('eligible') & np.isclose(j.value_cents,j.value_cents_shift,rtol=0,atol=1e-9)
            parity[day][str(d)]=dict(n=len(j),different=int((~good).sum()))
            require(good.all(),'strict certificate shift changed value/action')
            statuses[day][str(d)]={s:int(n) for s,n in x.status.value_counts().items()}
            e=x[x.status.str.startswith('eligible')].copy()
            r=e.reach_ws+e.reach_e2
            e['group']=np.where(r<=0,'QUIET',np.where(e.reach_e2<=0,'WS-REACHED',np.where(e.reach_e2>=r*.5,'E2-REACHED','MIXED')))
            panels[d].append(e)
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2048*1024,'RSS limit')
    report={}
    for d,pp in panels.items():
        p=pd.concat(pp,ignore_index=True)
        report[str(d)]=dict(n=len(p),mean=float(p.value_cents.mean()),groups={g:dict(n=len(q),mean=float(q.value_cents.mean())) for g,q in p.groupby('group')})
    for field,group in [('e2','E2-REACHED'),('ws','WS-REACHED')]:
        require(np.allclose([report[str(d)]['groups'][group]['mean'] for d in panels],original_report['R950'][field],rtol=0,atol=1e-9),'reported timing mean differs')
    for p,h in hashes.items():require(sha(Path(p))==h,'artifact changed')
    result=dict(asof=datetime.now(timezone.utc).isoformat(),source_report_asof='2026-11-15T14:40:31Z',source_sha256=sha(Path(__file__)),
        input_sha256=hashes,days=list(DAYS),scenarios=report,strict_shift_parity=parity,statuses=statuses,controls=ctl,
        interpretation='Within-replay timing sensitivity on changing populations; no estimate of physical timing errors or paired causal effect. Strict certificate invariance independently checked.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(asof=result['asof'],scenarios=report,strict_rows=sum(v['0']['n'] for v in parity.values()),controls=ctl)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
