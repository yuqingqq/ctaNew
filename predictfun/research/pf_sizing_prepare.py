"""Verify rebuilt rows and apply the unchanged deployed guard to the study inputs."""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import json
from .pf_sizing_study import DAYS, HERE, OLD, DATA, DEST, write, sha

def prepare(day):
    if day not in DAYS:raise ValueError('undeclared day')
    old=OLD/'step5'/day/'rows_rl_d2_pk.parquet'
    new=DATA/'step5'/day/'rows_rl_d2_pk_basis_study.parquet'
    a=pd.read_parquet(old);b=pd.read_parquet(new)
    if list(a)!=list(b) or len(a)!=len(b):raise ValueError('row/schema mismatch')
    differences={};boundary_diagnostics={}
    for c in a:
        x,y=a[c].to_numpy(),b[c].to_numpy()
        same=(x==y)|(pd.isna(x)&pd.isna(y));n=int((~same).sum())
        if n:
            differences[c]=n
            if c not in ['basis_bps','gap','p_bn']:raise ValueError('unrelated feature changed: '+c)
            # Historical carry bounds use each file's last Binance timestamp,
            # not wall-clock hour end. Data gaps can move that 10s overlap
            # earlier. The capture's first-70s bound is not a full-day invariant.
            # Enforce unchanged non-basis formulas/columns, record timing.
            phase=b.loc[~same,'tau']%3600000
            boundary_diagnostics[c]=dict(phase_min_ms=float(phase.min()),phase_max_ms=float(phase.max()),
                outside_first70s=int((phase>=70000).sum()),live_outside_first70s=int(((phase>=70000)&(b.loc[~same,'t_rel']<240)).sum()))
    del a
    cfg=json.loads((HERE.parent/'freeze/pm_ladder_v1/params.json').read_text())
    feats=cfg['guard']['features'];models={s:lgb.Booster(model_file=str(HERE.parent/'freeze/pm_ladder_v1'/f'guard_{s}.txt')) for s in ['bid','ask']}
    oldpr=pd.read_parquet(OLD/'step5'/day/'preds_t60k.parquet')
    if not np.array_equal(oldpr.mid,b.mid) or not np.array_equal(oldpr.tau,b.tau):raise ValueError('prediction alignment')
    # Original start-known mask was built from event-carried WS patch times, REST fallback.
    prestart=(oldpr.v_bid==-1)&(oldpr.v_ask==-1)
    pr=oldpr[['mid','tau','pred_1000','pred_250']].copy()
    X=b[feats].to_numpy(dtype=np.float32)
    live=(b.t_rel<240)&b.pf_mid.notna()
    for side,m in models.items():
        v=(m.predict(X,num_threads=1)-cfg['guard']['threshold_c60'][side]).astype(np.float32)
        v=np.where(live,v,np.float32(np.nan));v=np.where(prestart,np.float32(-1),v)
        pr['v_'+side]=v
    pr.to_parquet(DATA/'step5'/day/'preds_basis_study.parquet',index=False)
    # Sparse neutral scheduled decisions, independent of model actions or fills.
    anchors=[15,30,45,60,90,120,180,240]
    take=b.t_rel.isin(anchors);rows=b.loc[take].copy()
    rows['start_known']=~prestart[take].to_numpy();rows['day']=day
    reg=pd.read_parquet(OLD/'step1'/day/'registry.parquet')
    rows=rows.merge(reg[['mid','outcome','outcome_seen_ns','ws','fee_bps','start_price','end_price']],on='mid',how='left',validate='many_to_one')
    rows.to_parquet(DATA/'step5'/day/'settlement_actions.parquet',index=False)
    rep=dict(day=day,rows=len(b),boundary_differences=differences,boundary_diagnostics=boundary_diagnostics,other_columns_equal=True,
      start_withheld=int(prestart.sum()),neutral_actions=len(rows),guard_mode='fixed deployed models, in-sample on eight days',
      input_sha={str(p):sha(p) for p in [old,new,OLD/'step5'/day/'preds_t60k.parquet',OLD/'step1'/day/'registry.parquet']})
    write(DATA/'step5'/day/'prepare.json',rep);print(day,rep['boundary_differences'],flush=True)

if __name__=='__main__':
    for day in sys.argv[1:]:prepare(day)
