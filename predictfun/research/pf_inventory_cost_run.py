"""Bounded replay plus exact reference and accounting checks."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from .pf_inventory_cost_study import HERE,DATA,DEST,DAYS,ARMS,PRIOR,sha,write,check as source_check
from .pf_sizing_run import COMMON,stem,reconcile
from .pf_late_run import pre60_equal,reconcile_metadata


def folder(day,tm,mode='main'):
    return DATA/'step4'/f'ladder_tokyoreal_custom_e2_preds_basis_study_ic_{mode}_{tm}'/day


def verify(day,tm,mode):
    out=folder(day,tm,mode);prior=json.loads(PRIOR.read_text());meta=pd.read_parquet(out/'late_fills.parquet')
    all_data={}
    for label,spec in ARMS.items():
        all_data[label]={k:pd.read_parquet(out/k/stem(spec)) for k in ('win','fills','requests')}
        w,f=all_data[label]['win'],all_data[label]['fills']
        if len(w)!=(8 if mode=='smoke' else 288) or w.mid.duplicated().any():raise ValueError('market population')
        if not f.mid.isin(w.mid).all():raise ValueError('fill outside population')
        reconcile(w,f);reconcile_metadata(f,meta[meta.variant==label])
        if not np.allclose(w.pnl,w.mtm_stop+w.carry_stop,atol=1e-7,rtol=0):raise ValueError('carry')
        if label in ('baseline','late_reserved') and mode!='stress':
            for kind,a in all_data[label].items():
                ref=prior['artifacts'][f'{tm}/{day}/{label}'][kind]
                if sha(ref['path'])!=ref['sha']:raise ValueError('prior artifact changed')
                b=pd.read_parquet(ref['path']);b=b[b.mid.isin(w.mid)].reset_index(drop=True)
                pd.testing.assert_frame_equal(a,b,check_exact=True)
    b=all_data['baseline'];ws=b['win'].set_index('mid').ws
    for label in ('late_reserved','soft_exit','budget_exit'):
        for k in ('fills','requests'):pre60_equal(b[k],all_data[label][k],k,ws)
    budgets=pd.read_parquet(out/'cost_budgets.parquet')
    cost=budgets[budgets.variant=='budget_exit'].set_index('mid').committed_concession
    executed=meta[meta.variant=='budget_exit'].groupby('mid').concession.sum().reindex(cost.index,fill_value=0.)
    if (cost>.2500001).any() or (executed>cost+1e-7).any():raise ValueError('cost budget accounting')
    for label in ('late_reserved','soft_exit','budget_exit','marginal_entry'):
        z=meta[(meta.variant==label)&(meta.send_rel>=60)]
        if ((z.q-z.reducing_q)>1e-7).any() or (z.send_rel>=240).any():raise ValueError('new late order opened reverse exposure')
    audit=pd.read_parquet(out/'exit_episodes.parquet')
    if set(audit.mid)!=set(b['win'].mid):raise ValueError('audit coverage')
    if not np.allclose((audit.end-audit.start).groupby(audit.mid).sum(),240,atol=1e-8):raise ValueError('audit duration')
    write(out/'checks.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),markets=len(b['win']),
        baseline_exact=mode!='stress',control_exact=mode!='stress',exit_entry_prefix_exact=True,
        fill_cash_carry_reconciled=True,cost_budget_verified=True,new_late_reverse_opening=False,
        artifacts={str(p.relative_to(out)):sha(p) for p in out.rglob('*.parquet')}))


def run(day,tm,mode):
    source_check()
    if day not in DAYS or tm not in ('J','X') or mode not in ('main','smoke','stress'):raise ValueError('undeclared run')
    if mode=='stress' and (day,tm)!=('20261103','X'):raise ValueError('undeclared stress population')
    out=folder(day,tm,mode)
    if out.exists():raise ValueError('refuse overwrite')
    inherited={k:v for k,v in os.environ.items() if not k.startswith('PF_')}
    env=dict(inherited,**COMMON,PF_FORWARD='20261104',PF_PREDS='preds_basis_study',PF_TAG=f'ic_{mode}_{tm}',
        PF_MAX_WINDOWS='8' if mode=='smoke' else '0',PF_ARMS=';'.join(ARMS.values()),
        PF_TXJIT='frame' if tm=='J' else '',PF_TXSHIFT='0' if tm=='J' else '-50',PF_QACT='' if tm=='J' else 'max')
    if mode=='stress':env.update(PF_RL_LPLACE='80',PF_RL_LCANCEL='22',PF_RL_LPENDCANCEL='95',PF_RL_LFILL='300')
    subprocess.run([sys.executable,'-m',__package__+'.pf_inventory_cost_engine',day,'tokyoreal','custom'],cwd=HERE.parents[1],env=env,check=True)
    verify(day,tm,mode);print('VERIFIED',day,tm,mode,flush=True)


if __name__=='__main__':run(*sys.argv[1:])
