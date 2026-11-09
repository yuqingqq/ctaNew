"""Replay and artifact identity checks, including exact baseline entry prefix."""
import datetime as dt
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from .pf_exit_study import HERE, ARMS, DAYS, DATA, PRIOR, REFERENCES, sha, write
from .pf_sizing_run import COMMON, stem, reconcile


def folder(day,tm,mode='main'):
    return DATA/'step4'/f'ladder_tokyoreal_custom_e2_preds_basis_study_exit_{mode}_{tm}'/day


def reference(day,tm,label="baseline"):
    r=json.loads((PRIOR/'report.json').read_text())
    files=r['artifacts'][f'{tm}/{day}/{label}']
    for v in files.values():
        if sha(v['path'])!=v['sha']:
            raise ValueError('reference artifact changed')
    from pathlib import Path
    return Path(files['win']['path']).parent.parent


def location(label,day,tm):
    return reference(day,tm,label) if label in REFERENCES and day!=DAYS[0] else folder(day,tm)


from .pf_late_run import pre60_equal,reconcile_metadata


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
        if label in REFERENCES:
            ref_loc=reference(day,tm,label)
            for kind,a in [('win',w),('fills',f),('requests',q)]:
                expected=pd.read_parquet(ref_loc/kind/stem(spec))
                expected=expected[expected.mid.isin(mids)].reset_index(drop=True)
                pd.testing.assert_frame_equal(a,expected,check_exact=True)
        if label not in REFERENCES or loc==out:
            reconcile_metadata(f,meta[meta.variant==label])
        checks.append(label)
    write(out/'passive_checks.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),day=day,timing=tm,
        markets=len(mids),variants=checks,pre60_exact=['fills','requests'],reference_exact={k:['win','fills','requests'] for k in REFERENCES},
        fresh_references=mode=='smoke' or day==DAYS[0],source_sha=sha(HERE/'pf_exit_engine.py')))


def run(day,tm,mode):
    if day not in DAYS or tm not in ('J','X') or mode not in ('main','smoke'):
        raise ValueError('undeclared run')
    out=folder(day,tm,mode)
    if out.exists():
        raise ValueError('refuse overwrite')
    labels=list(ARMS) if mode=='smoke' or day==DAYS[0] else ['exit_keep','exit_close25']
    inherited={k:v for k,v in os.environ.items() if not k.startswith('PF_')}
    env=dict(inherited,**COMMON,PF_FORWARD='20261104',PF_PREDS='preds_basis_study',PF_TAG=f'exit_{mode}_{tm}',
        PF_MAX_WINDOWS='8' if mode=='smoke' else '0',PF_ARMS=';'.join(ARMS[k] for k in labels),
        PF_TXJIT='frame' if tm=='J' else '',PF_TXSHIFT='0' if tm=='J' else '-50',PF_QACT='' if tm=='J' else 'max')
    subprocess.run([sys.executable,'-m',__package__+'.pf_exit_engine',day,'tokyoreal','custom'],env=env,cwd=HERE.parents[1],check=True)
    check(day,tm,mode)
    print('RECONCILED',day,tm,mode,flush=True)


if __name__=='__main__':
    run(*sys.argv[1:])
