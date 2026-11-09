"""Replay and artifact identity checks, including exact baseline entry prefix."""
import datetime as dt
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from .pf_late_study import HERE, ARMS, DAYS, DATA, PRIOR, sha, write
from .pf_sizing_run import COMMON, stem, reconcile


def folder(day,tm,mode='main'):
    return DATA/'step4'/f'ladder_tokyoreal_custom_e2_preds_basis_study_late_{mode}_{tm}'/day


def reference(day,tm):
    r=json.loads((PRIOR/'report.json').read_text())
    files=r['artifacts'][f'{tm}/{day}/baseline']
    for v in files.values():
        if sha(v['path'])!=v['sha']:
            raise ValueError('reference artifact changed')
    from pathlib import Path
    return Path(files['win']['path']).parent.parent


def location(label,day,tm):
    return reference(day,tm) if label=='baseline' and day!=DAYS[0] else folder(day,tm)


def pre60_equal(b,a,kind,ws):
    if kind=='fills':
        select=lambda x:x[x.t<60].reset_index(drop=True)
    elif kind=='requests':
        select=lambda x:x[x.t_ms<x.mid.map(ws)+60000].reset_index(drop=True)
    else:
        raise ValueError('unknown prefix artifact')
    pd.testing.assert_frame_equal(select(b),select(a),check_exact=True)


def reconcile_metadata(f,z):
    if len(f)!=len(z):
        raise ValueError('fill metadata population')
    expected=f[['mid','t','side','pt','q']].reset_index(drop=True)
    actual=z[['mid','t','side','pt','q']].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected,actual,check_dtype=False,check_exact=True)
    if not np.array_equal(z['index'],f.groupby('mid',sort=False).cumcount()):
        raise ValueError('fill identity index')
    sg=np.where(z.side==0,1.,-1.)
    expected_net=(f.q*np.where(f.side==0,1.,-1.)).groupby(f.mid,sort=False).cumsum().to_numpy()
    if not np.allclose(z.net_after,expected_net,atol=1e-7,rtol=0):
        raise ValueError('independent fill path identity')
    if not np.allclose(z.net_after-z.net_before,sg*z.q,atol=1e-7,rtol=0):
        raise ValueError('fill path metadata')
    if not np.allclose(z.concession,z.q*sg*(z.pt-z.base_price)/100,atol=1e-7,rtol=0):
        raise ValueError('concession accounting')
    if not np.allclose(z.concession_ticks,sg*(z.pt-z.base_price),atol=1e-7,rtol=0):
        raise ValueError('concession tick identity')
    if not z.concession_ticks.isin([0,1]).all() or not ((z.decision_rel<=z.send_rel)&(z.send_rel<=z.t)&(z.q>0)).all():
        raise ValueError('invalid fill attribution')
    reducing=np.where(z.net_before*sg<0,np.minimum(z.net_before.abs(),z.q),0.)
    if not np.allclose(z.reducing_q,reducing,atol=1e-7,rtol=0):
        raise ValueError('economic reduction attribution')
    shifted=z[z.concession_ticks>0]
    if not ((shifted.decision_rel>=60)&(shifted.decision_rel<240)&(shifted.send_rel<240)).all():
        raise ValueError('concession outside phase')
    if len(shifted) and not (np.where(shifted.side==0,1.,-1.)*shifted.known_net<0).all():
        raise ValueError('concession was not reducing at submission')


def check(day,tm,mode):
    out=folder(day,tm,mode);ref=reference(day,tm)
    b={k:pd.read_parquet(ref/k/stem(ARMS['baseline'])) for k in ['win','fills','requests']}
    mids=b['win'].mid if mode=='main' else b['win'].mid.iloc[:8]
    b={k:v[v.mid.isin(mids)].reset_index(drop=True) for k,v in b.items()}
    ws=b['win'].set_index('mid').ws
    meta=pd.read_parquet(out/'late_fills.parquet')
    checks=[]
    for label,spec in ARMS.items():
        loc=location(label,day,tm) if mode=='main' else out
        w,f,q=(pd.read_parquet(loc/k/stem(spec)) for k in ['win','fills','requests'])
        if not np.array_equal(w.mid,mids) or not f.mid.isin(mids).all() or not q.mid.isin(mids).all():
            raise ValueError('market population')
        reconcile(w,f)
        if not np.allclose(w.pnl,w.mtm_stop+w.carry_stop,atol=1e-7,rtol=0):
            raise ValueError('carry reconciliation')
        for kind,a in [('fills',f),('requests',q)]:
            pre60_equal(b[kind],a,kind,ws)
        if label=='baseline':
            for kind,a in [('win',w),('fills',f),('requests',q)]:
                pd.testing.assert_frame_equal(a,b[kind],check_exact=True)
        if label!='baseline' or loc==out:
            reconcile_metadata(f,meta[meta.variant==label])
        checks.append(label)
    write(out/'passive_checks.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),day=day,timing=tm,
        markets=len(mids),variants=checks,pre60_exact=['fills','requests'],baseline_exact=['win','fills','requests'],
        fresh_baseline=mode=='smoke' or day==DAYS[0],source_sha=sha(HERE/'pf_late_engine.py')))


def run(day,tm,mode):
    if day not in DAYS or tm not in ('J','X') or mode not in ('main','smoke'):
        raise ValueError('undeclared run')
    out=folder(day,tm,mode)
    if out.exists():
        raise ValueError('refuse overwrite')
    labels=list(ARMS) if mode=='smoke' or day==DAYS[0] else ['late_reserved','late_taper1']
    inherited={k:v for k,v in os.environ.items() if not k.startswith('PF_')}
    env=dict(inherited,**COMMON,PF_FORWARD='20261104',PF_PREDS='preds_basis_study',PF_TAG=f'late_{mode}_{tm}',
        PF_MAX_WINDOWS='8' if mode=='smoke' else '0',PF_ARMS=';'.join(ARMS[k] for k in labels),
        PF_TXJIT='frame' if tm=='J' else '',PF_TXSHIFT='0' if tm=='J' else '-50',PF_QACT='' if tm=='J' else 'max')
    subprocess.run([sys.executable,'-m',__package__+'.pf_late_engine',day,'tokyoreal','custom'],env=env,cwd=HERE.parents[1],check=True)
    check(day,tm,mode)
    print('RECONCILED',day,tm,mode,flush=True)


if __name__=='__main__':
    run(*sys.argv[1:])
