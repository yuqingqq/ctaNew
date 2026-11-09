"""Sequential bounded replays with fresh-reference parity and reconciled paths."""
import datetime as dt
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from .pf_range_study import HERE, ARMS, DAYS, DATA, REFERENCES, sha, write
from .pf_sizing_run import COMMON, stem, reconcile
from .pf_inflight_run import location as reference_location


def folder(day, tm, mode='main'):
    return DATA/'step4'/f'ladder_tokyoreal_custom_e2_preds_basis_study_range_{mode}_{tm}'/day


def location(label, day, tm):
    return reference_location(label,day,tm) if label in REFERENCES and day != DAYS[0] else folder(day,tm)


def check(day, tm, mode):
    out=folder(day,tm,mode); checks=[]; refs={}
    labels=list(ARMS) if mode=='main' else REFERENCES
    reference=pd.read_parquet(reference_location('baseline',day,tm)/'win'/stem(ARMS['baseline']))
    mids=reference.mid if mode=='main' else reference.mid.iloc[:8]
    for label in labels:
        path=location(label,day,tm) if mode=='main' else out
        spec=ARMS[label]
        w,f,q=(pd.read_parquet(path/k/stem(spec)) for k in ['win','fills','requests'])
        if not np.array_equal(w.mid,mids) or not f.mid.isin(mids).all() or not q.mid.isin(mids).all():
            raise ValueError('market population')
        reconcile(w,f)
        if not np.allclose(w.pnl,w.mtm_stop+w.carry_stop,rtol=0,atol=1e-7):
            raise ValueError('carry reconciliation')
        if label in REFERENCES:
            for kind,a in [('win',w),('fills',f),('requests',q)]:
                ref=reference_location(label,day,tm)/kind/stem(spec); b=pd.read_parquet(ref)
                b=b[b.mid.isin(mids)].reset_index(drop=True)
                pd.testing.assert_frame_equal(a,b,check_exact=True)
                refs[label+'/'+kind]=dict(path=str(ref),sha=sha(ref))
        checks.append(label)
    write(out/'passive_checks.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),day=day,timing=tm,
        markets=len(mids),variants=checks,reference_artifacts=refs,
        fresh_reference_parity=mode=='smoke' or day==DAYS[0],source_sha=sha(HERE/'pf_range_engine.py')))


def run(day, tm, mode):
    if day not in DAYS or tm not in ('J','X') or mode not in ('main','smoke'):
        raise ValueError('undeclared run')
    out=folder(day,tm,mode)
    if out.exists():raise ValueError('refuse overwrite')
    labels=REFERENCES if mode=='smoke' else list(ARMS) if day==DAYS[0] else ['range10']
    inherited={k:v for k,v in os.environ.items() if not k.startswith('PF_')}
    env=dict(inherited,**COMMON,PF_FORWARD='20261104',PF_PREDS='preds_basis_study',PF_TAG=f'range_{mode}_{tm}',
        PF_MAX_WINDOWS='8' if mode=='smoke' else '0',PF_ARMS=';'.join(ARMS[k] for k in labels),
        PF_TXJIT='frame' if tm=='J' else '',PF_TXSHIFT='0' if tm=='J' else '-50',PF_QACT='' if tm=='J' else 'max')
    subprocess.run([sys.executable,'-m',__package__+'.pf_range_engine',day,'tokyoreal','custom'],env=env,cwd=HERE.parents[1],check=True)
    check(day,tm,mode)
    if mode=='main':
        from .pf_range_report import summarize
        for label,spec in ARMS.items():summarize(day,tm,label,spec)
    print('RECONCILED',day,tm,mode,flush=True)


if __name__=='__main__':run(*sys.argv[1:])
