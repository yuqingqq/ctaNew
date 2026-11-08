"""Bounded runner and reconciliation for the declared sizing variants."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from .pf_sizing_study import DAYS, BASE, ARMS, DATA, DEST, OLD, HERE, sha, write

COMMON={'PF_RL_DDEC':'2','PF_RL_LPLACE':'40','PF_RL_LCANCEL':'22','PF_RL_LPENDCANCEL':'95',
        'PF_RL_LFILL':'100','PF_TIMING':'e2','PF_TXTIME':'ws','PF_TXKEY':'tk',
        'PF_FILLPRIO':'price','PF_OWNCROSS':'on','PF_LEGORDER':'price','PF_REJRESET':'on','PF_POLICYVIEW':'known'}

def specs():
    return {f'{name}_{budget}':spec if budget=='off' else spec.replace('+CAP10','+CAP10+XCAP150')
            for budget in ['off','150'] for name,spec in ARMS.items()}

def name(spec):
    p,r,m,s=spec.split('|');return f'{p}+{r}/{m}@{s}'

def stem(spec):return name(spec).replace('/','__').replace('+','_').replace('@','_sz').replace(':','-')+'.parquet'

def reconcile(w,f):
    if w.mid.duplicated().any():raise ValueError('duplicate window')
    sums=f.groupby('mid').settle.sum().reindex(w.mid,fill_value=0).to_numpy()
    if not np.allclose(w.pnl.to_numpy(),sums,rtol=0,atol=1e-7):raise ValueError('settlement reconciliation')

def folder(day,tm,smoke=False):
    pred='preds_t60k' if smoke else 'preds_basis_study'
    tag='sst_smoke' if smoke else 'sst_'+tm
    return DATA/'step4'/f'ladder_tokyoreal_custom_e2_{pred}_{tag}'/day

def run(day,tm,smoke=False):
    if day not in DAYS or tm not in ['J','X']:raise ValueError('undeclared day or timing')
    out=folder(day,tm,smoke)
    if out.exists():raise ValueError('refuse overwrite of run')
    env=dict(os.environ,**COMMON,PF_PREDS='preds_t60k' if smoke else 'preds_basis_study',
      PF_TAG='sst_smoke' if smoke else 'sst_'+tm,PF_MAX_WINDOWS='8' if smoke else '0',
      PF_ARMS=BASE if smoke else ';'.join(specs().values()),PF_STUDY_EXEC='0' if smoke else '1',
      PF_CAP_CACHE='0' if smoke else '1',
      PF_TXJIT='frame' if tm=='J' else '',PF_TXSHIFT='0' if tm=='J' else '-50',PF_QACT='' if tm=='J' else 'max')
    if smoke:
        p=DATA/'step5'/day/'preds_t60k.parquet'
        if not p.exists():p.symlink_to(OLD/'step5'/day/'preds_t60k.parquet')
    elif not (DEST/'cap_reuse_control.json').exists():
        raise ValueError('missing direct cap-path control')
    elif DAYS.index(day)>0:
        env['PF_LIVE_CAPSTATE_IN']=str(folder(DAYS[DAYS.index(day)-1],tm)/'capstate.json')
    cmd=[sys.executable,'-m','predictfun.research.pf_sizing_engine',day,'tokyoreal','custom']
    subprocess.run(cmd,env=env,cwd=HERE.parents[1],check=True)
    if smoke:
        ref=OLD/'step4'/'ladder_tokyoreal_custom_e2_preds_t60k_i2J'/day
        for typ in ['win','fills','requests']:
            a=pd.read_parquet(out/typ/stem(BASE));b=pd.read_parquet(ref/typ/stem(BASE))
            mids=pd.read_parquet(out/'win'/stem(BASE)).mid
            b=b[b.mid.isin(mids)].reset_index(drop=True)
            pd.testing.assert_frame_equal(a,b,check_exact=True)
        write(DEST/'engine_parity.json',dict(passed=True,windows=8,day=day,timing=tm,
            source_sha=sha(HERE/'pf_sizing_engine.py'),comparison='unchanged frozen v3 win/fill/request data exact'))

def report():
    rows=[];allwins={};counts={};statuses={}
    for tm in ['J','X']:
        for label,spec in specs().items():
            wins=[];fills=[];req=[];ctr={};dayrows=[]
            for day in DAYS:
                path=folder(day,tm);s=json.loads((path/'summary.json').read_text())
                if s['env']['PF_ARMS']!=';'.join(specs().values()) or s['env']['PF_STUDY_EXEC']!='1':raise ValueError('run identity')
                w=pd.read_parquet(path/'win'/stem(spec));f=pd.read_parquet(path/'fills'/stem(spec));q=pd.read_parquet(path/'requests'/stem(spec))
                if w.mid.duplicated().any() or len(w)!=288:raise ValueError('window coverage')
                reconcile(w,f)
                for k,v in s['arms'][name(spec)]['live']['counters'].items():ctr[k]=ctr.get(k,0)+v
                dayrows.append(dict(day=day,settle=float(w.pnl.sum()),plus1=float(f.mo1.sum()),shares=float(w.shares.sum())))
                wins.append(w);fills.append(f);req.append(q)
            w=pd.concat(wins);f=pd.concat(fills);q=pd.concat(req)
            eq=np.r_[0.,w.pnl.cumsum().to_numpy()];dd=eq-np.maximum.accumulate(eq)
            row=dict(timing=tm,variant=label,windows=len(w),settle=float(w.pnl.sum()),plus1=float(f.mo1.sum()),
              mark_stop=float(w.mtm_stop.sum()),carry=float(w.carry_stop.sum()),max_drawdown=float(dd.min()),
              shares=float(w.shares.sum()),fills=len(f),mean_abs_net=float(w.net_end.abs().mean()),
              p95_abs_net=float(w.net_end.abs().quantile(.95)),refusals=int(w.refused.sum()),halted_windows=int(w.halted.sum()),
              requests=q.kind.value_counts().to_dict(),per_day=dayrows,counters=ctr,
              statuses=dict(missing_mark_stop=int(w.mtm_stop.isna().sum()),missing_mo1=int(f.mo1.isna().sum())))
            rows.append(row);allwins[(tm,label)]=w
    result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),days=DAYS,results=rows,
      controls=dict(engine_parity=json.loads((DEST/'engine_parity.json').read_text())),
      interpretation='exploratory fixed-model in-sample strategy comparison; no untouched validation or live parity claim',
      code_sha={p.name:sha(p) for p in HERE.glob('pf_sizing*.py')},forward_data_read=False)
    write(DEST/'strategy_report.json',result)
    for r in rows:print({k:r[k] for k in ['timing','variant','settle','plus1','max_drawdown','shares','refusals','halted_windows','requests']})

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','smoke','report']);p.add_argument('day',nargs='?');p.add_argument('timing',nargs='?',default='J');a=p.parse_args()
    if a.action=='report':report()
    else:run(a.day or DAYS[0],a.timing,a.action=='smoke')

if __name__=='__main__':main()
