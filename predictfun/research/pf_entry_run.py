"""Bounded entry-budget replay with original baseline reconciliation."""
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from .pf_entry_study import HERE,ARMS,DAYS,DATA,sha,write
from .pf_sizing_run import COMMON,stem,name,reconcile,folder as old_folder


def folder(day,tm,mode='main'):
    return DATA/'step4'/f'ladder_tokyoreal_custom_e2_preds_basis_study_entry_{mode}_{tm}'/day


def check(day,tm,mode):
    out=folder(day,tm,mode);base=ARMS['baseline'];w=pd.read_parquet(out/'win'/stem(base))
    checked=[]
    for label,spec in ARMS.items() if mode=='main' else [('baseline',base)]:
        wc,fc,qc=(pd.read_parquet(out/k/stem(spec)) for k in ['win','fills','requests'])
        if len(wc)!=(288 if mode=='main' else 8) or not np.array_equal(w.mid,wc.mid):raise ValueError('market population')
        if not fc.mid.isin(wc.mid).all() or not qc.mid.isin(wc.mid).all():raise ValueError('orphan event')
        reconcile(wc,fc)
        if not np.allclose(wc.pnl,wc.mtm_stop+wc.carry_stop,atol=1e-7,rtol=0):raise ValueError('carry reconciliation')
        checked.append(label)
    for kind in ['win','fills','requests']:
        a=pd.read_parquet(out/kind/stem(base));b=pd.read_parquet(old_folder(day,tm)/kind/stem(base))
        b=b[b.mid.isin(w.mid)].reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_exact=True)
    write(out/'passive_checks.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),day=day,timing=tm,
        markets=len(w),variants=checked,baseline_exact=['win','fills','requests'],
        baseline_origin='retained reference' if (out/'baseline_reference.json').exists() else 'fresh replay',
        source_sha=sha(HERE/'pf_entry_engine.py')))


def run(day,tm,mode):
    if day not in DAYS or tm not in ['J','X'] or mode not in ['main','smoke']:raise ValueError('undeclared run')
    out=folder(day,tm,mode)
    if out.exists():raise ValueError('refuse overwrite')
    reuse=mode=='main' and day!=DAYS[0]
    specs=[v for k,v in ARMS.items() if not reuse or k!='baseline'] if mode=='main' else [ARMS['baseline']]
    inherited={k:v for k,v in os.environ.items() if not k.startswith('PF_')}
    env=dict(inherited,**COMMON,PF_FORWARD='20261104',PF_PREDS='preds_basis_study',PF_TAG=f'entry_{mode}_{tm}',
        PF_MAX_WINDOWS='8' if mode=='smoke' else '0',PF_ARMS=';'.join(specs),
        PF_TXJIT='frame' if tm=='J' else '',PF_TXSHIFT='0' if tm=='J' else '-50',PF_QACT='' if tm=='J' else 'max')
    subprocess.run([sys.executable,'-m',__package__+'.pf_entry_engine',day,'tokyoreal','custom'],env=env,cwd=HERE.parents[1],check=True)
    if reuse:
        ref=old_folder(day,tm);paths={}
        for kind in ['win','fills','requests']:
            src=ref/kind/stem(ARMS['baseline']);shutil.copy2(src,out/kind/src.name)
            paths[kind]=dict(path=str(src),sha256=sha(src))
        write(out/'baseline_reference.json',dict(summary_path=str(ref/'summary.json'),summary_sha=sha(ref/'summary.json'),artifacts=paths))
    check(day,tm,mode)
    if mode=='main':
        from .pf_entry_report import summarize
        for label,spec in ARMS.items():summarize(day,tm,label,spec)
    print('RECONCILED',day,tm,mode,flush=True)


if __name__=='__main__':run(*sys.argv[1:])
