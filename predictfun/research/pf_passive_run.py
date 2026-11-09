"""Run each declared passive variant and verify the shared entry population."""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from .pf_passive_study import ARMS,DAYS,DATA,DEST,HERE,ACK_MS,sha,write
from .pf_sizing_run import COMMON,stem,name,reconcile,folder as old_folder


def folder(day,tm,mode='main'):
    return DATA/'step4'/f'ladder_tokyoreal_custom_e2_preds_basis_study_passive_{mode}_{tm}'/day


def require_prephase_equal(b,c,kind,ws):
    if kind=='fills':
        x=b[b.t<60].reset_index(drop=True);y=c[c.t<60].reset_index(drop=True)
    elif kind=='requests':
        x=b[b.t_ms<b.mid.map(ws)+60000].reset_index(drop=True)
        y=c[c.t_ms<c.mid.map(ws)+60000].reset_index(drop=True)
    else:raise ValueError('unknown artifact')
    pd.testing.assert_frame_equal(x,y,check_exact=True)


def check(day,tm,mode):
    out=folder(day,tm,mode);checks=[]
    base=ARMS['baseline'];w=pd.read_parquet(out/'win'/stem(base));ws=w.set_index('mid').ws
    # ws in win artifacts is already milliseconds (assert against event rows below).
    fb=pd.read_parquet(out/'fills'/stem(base));qb=pd.read_parquet(out/'requests'/stem(base))
    if len(qb) and not ((qb.t_ms-qb.mid.map(ws)).between(-1,301000)).all():raise ValueError('window clock units')
    for label,spec in ARMS.items() if mode!='smoke' else [('baseline',base)]:
        wc=pd.read_parquet(out/'win'/stem(spec));fc=pd.read_parquet(out/'fills'/stem(spec));qc=pd.read_parquet(out/'requests'/stem(spec))
        if len(wc)!=(8 if mode=='smoke' else 288):raise ValueError('market coverage')
        if not np.array_equal(w.mid,wc.mid):raise ValueError('market identity')
        reconcile(wc,fc)
        if not np.allclose(wc.pnl,wc.mtm_stop+wc.carry_stop,atol=1e-7,rtol=0):raise ValueError('carry reconciliation')
        for kind,b,c in [('fills',fb,fc),('requests',qb,qc)]:
            require_prephase_equal(b,c,kind,ws)
            checks.append(dict(variant=label,kind=kind,pre60_exact=True))
    parity=[]
    if mode in ['main','smoke']:
        for kind in ['win','fills','requests']:
            a=pd.read_parquet(out/kind/stem(base));b=pd.read_parquet(old_folder(day,tm)/kind/stem(base))
            b=b[b.mid.isin(w.mid)].reset_index(drop=True)
            pd.testing.assert_frame_equal(a,b,check_exact=True);parity.append(kind)
    report=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),day=day,timing=tm,mode=mode,
        markets=len(w),entry_checks=checks,baseline_exact=parity,
        baseline_origin='retained reference' if (out/'baseline_reference.json').exists() else 'fresh replay',
        source_sha=sha(HERE/'pf_passive_engine.py'))
    write(out/'passive_checks.json',report);print('CHECKED',day,tm,mode,len(w),'markets',len(checks),'entry comparisons',flush=True)


def run(day,tm,mode):
    if day not in DAYS or tm not in ['J','X'] or mode not in ['main','stress','smoke']:raise ValueError('undeclared run')
    if mode=='stress' and day!='20261103':raise ValueError('undeclared stress date')
    out=folder(day,tm,mode)
    if out.exists():raise ValueError('refuse overwrite')
    reuse=mode=='main' and day!=DAYS[0]
    run_arms={k:v for k,v in ARMS.items() if not reuse or k!='baseline'}
    env=dict(os.environ,**COMMON,PF_PREDS='preds_basis_study',PF_TAG=f'passive_{mode}_{tm}',
        PF_MAX_WINDOWS='8' if mode=='smoke' else '0',PF_ARMS=ARMS['baseline'] if mode=='smoke' else ';'.join(run_arms.values()),
        PF_STUDY_EXEC='1',PF_CAP_CACHE='0',PF_STUDY_CANCEL_ACK_MS=str(ACK_MS if mode=='stress' else 0),
        PF_TXJIT='frame' if tm=='J' else '',PF_TXSHIFT='0' if tm=='J' else '-50',PF_QACT='' if tm=='J' else 'max')
    env.pop('PF_LIVE_CAPSTATE_IN',None)
    subprocess.run([sys.executable,'-m','predictfun.research.pf_passive_engine',day,'tokyoreal','custom'],env=env,cwd=HERE.parents[1],check=True)
    if reuse:
        ref=old_folder(day,tm);paths={}
        for kind in ['win','fills','requests']:
            src=ref/kind/stem(ARMS['baseline']);shutil.copy2(src,out/kind/src.name);paths[kind]=dict(path=str(src),sha256=sha(src))
        write(out/'baseline_reference.json',dict(summary_path=str(ref/'summary.json'),summary_sha=sha(ref/'summary.json'),
            artifacts=paths,reason='unchanged baseline; direct full-day control rerun on first date at each timing'))
    check(day,tm,mode)
    if mode!='smoke':
        from .pf_passive_report import summarize
        for label,spec in ARMS.items():summarize(day,tm,mode,label,spec)
        print('METRICS RECONCILED',day,tm,mode,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('day');p.add_argument('timing');p.add_argument('mode',choices=['main','stress','smoke']);a=p.parse_args()
    run(a.day,a.timing,a.mode)
