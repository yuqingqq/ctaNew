"""Frozen v3 EV at recorded placement/cancel medians; no forecast or refit.

Compare 40/22 with 58/37 ms, all other settings held at the prior latency
study. Every baseline window must reproduce that study. All five consumed
dates, all fills. Deterministic median sensitivity, not full physical latency
validation: feed clocks, notification, pending cancel, queue and tape unchanged.
"""
import argparse
import collections
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb
import pf_stop_ev_latency as L

P,V,E,M,K,AC = L.P,L.V,L.E,L.M,L.K,L.AC
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'reviews/stop_ev_latency_20261115'
DATA=Path(K.IO)/'stop_ev_latency_20261115_v3'
CASES=(('baseline_40_22',40.,22.),('medians_58_37',58.,37.))


@contextmanager
def clocks(place,cancel):
    with L.clock(False):
        K.L_PLACE=place;K.L_CANCEL=cancel
        yield


def parity(a,b):
    for key in ['pnl','shares','orders','pulls','fill_events','max_abs_position']:
        L.require(abs(float(a[key])-float(b[key]))<1e-7,f'baseline window mismatch {key}')
    for j in range(P.NR):
        L.require(abs(float(a[f'live{j}'])-float(b[f'live{j}']))<1e-12,'baseline exposure mismatch')


def selftest():
    checks=L.selftest()
    w=M.toy_lane_window([50.],[50],[130.])
    with clocks(40.,22.):a=L.simulate(w,shift=0.)
    with clocks(58.,37.):b=L.simulate(w,shift=0.)
    L.require(a['fl'] and not b['fl'],'placement median control')
    w=M.toy_lane_window([80.],[50],[130.]);dec={(s,d):(lambda k,t,c:t>=50.) for s in (0,1) for d in range(P.NR)}
    with clocks(40.,22.):a=L.simulate(w,dec=dec,shift=0.)
    with clocks(40.,37.):b=L.simulate(w,dec=dec,shift=0.)
    L.require(not a['fl'] and b['fl'],'cancel median control')
    row={k:1. for k in ['pnl','shares','orders','pulls','fill_events','max_abs_position']+[f'live{j}' for j in range(P.NR)]}
    parity(row,row);bad=dict(row,pnl=2.);caught=False
    try:parity(row,bad)
    except ValueError:caught=True
    L.require(caught,'baseline corruption not refused')
    checks.update(placement_median_fires=True,cancel_median_fires=True,corrupt_baseline_refused=caught)
    return checks


def fresh(heads,W,X):
    pol=L.Policy(heads,W,X,'EV',(0.,1.),collections.Counter());keys=[];rows=[]
    for k,t in enumerate(W[0]['tg']):
        for s in (0,1):
            for d in range(P.NR):
                v=pol._vec(s,d,k,float(t),None)
                if v is not None:keys.append((s,d,k));rows.append(v)
    out=np.full((2,P.NR,W[0]['n']),np.nan)
    if rows:
        x=np.vstack(rows);pred=heads[0].predict(x,num_threads=1)*heads[1].predict(x,num_threads=1)
        for key,p in zip(keys,pred):out[key]=p
    return out


def run(day,out):
    L.require(day in L.DAYS and not out.exists(),'day or existing output');out.mkdir(parents=True)
    start=time.time();checks=selftest();stats=collections.Counter();results=[]
    prior=json.loads((PREVIOUS/f'receipt_{day}_v3.json').read_text())
    attr=json.loads((PREVIOUS/'attribution_v3.json').read_text())
    wp=DATA/day/'windows.parquet'
    L.require(L.sha(wp)==attr['input_hashes'][day]['windows.parquet'],'reference window identity')
    for name,digest in prior['sources'].items():L.require(L.sha(HERE/name)==digest,'prior source changed')
    def inputs_unchanged():
        for path,item in prior['inputs'].items():
            p=Path(path);L.require(p.exists()==item['exists'] and (not item['exists'] or L.sha(p)==item['sha256']),'prior input changed')
    inputs_unchanged()
    reference=pd.read_parquet(wp);reference=reference[(reference.cfg=='current')&(reference.arm=='EV')].set_index('m')
    L.require(reference.index.is_unique,'duplicate reference')
    fit=json.loads((Path(V.OUTD)/f'fit3_{day}.json').read_text())
    L.require(fit['version']=='v3' and fit['reload_pass'] and fit['features']==V.FULL_IN,'model contract')
    heads=[lgb.Booster(model_file=fit['artifacts'][f'full_{h}']['path']) for h in ['p','v']]
    MK=P.market_arrays(day);D=K.load_day(day);cut=M.boundary_cut(day)
    tk=pd.read_parquet(Path(K.IO)/'step1'/day/'legs_tk.parquet',columns=['tk']).tk.to_numpy();LT=D['LT'];ix=LT.index.to_numpy()
    grp=np.where(tk[ix]>=0,tk[ix],-1-ix);WSg=pd.Series(M.TS.ws_groups(LT.how.to_numpy(),grp),index=LT.index);GRP=pd.Series(grp,index=LT.index)
    RW,ust=E.raw_windows(day,D)
    def check(W,cur,key,t,f):
        stats['state_calls']+=1
        if stats['state_calls']%197:return
        g=AC.known_state(W,cur,key+(0,),t,30.,terminal_ms=W[0]['te']+200.,settings=AC.Settings(notification_ms=P.L_NOTE,cancel_ms=K.L_CANCEL))
        L.require(all((math.isnan(float(f[c])) and math.isnan(float(g[c]))) or abs(float(f[c])-float(g[c]))<1e-9 for c in P.ORDER_KEYS),'state parity')
        stats['state_parity']+=1
    for W in M.P1.window_pairs(D):
        if W[0]['ws']<cut:stats['boundary_windows']+=1;continue
        m=int(W[0]['m']);LR={};SR={};GR={}
        for s in (0,1):
            if (m,s) in D['LI']:
                g=LT.iloc[D['LI'][(m,s)]];o=np.lexsort(((-g.ptick.to_numpy() if s==0 else g.ptick.to_numpy()),g.anc.to_numpy()));lr=g.index.to_numpy()[o]
            else:lr=np.zeros(0,dtype=int)
            L.require(len(lr)==len(W[s]['anc']) and np.array_equal(LT.loc[lr,'anc'].to_numpy(),W[s]['anc']),'tape identity')
            LR[s]=lr;SR[s]=WSg.reindex(lr).to_numpy(bool);GR[s]=GRP.reindex(lr).to_numpy('int64')
        W,LR,SR,GR,IM=E.impute_window(W,LR,SR,GR,RW.get(m,{}).get('imp',[]),ust['imp_off_ms'])
        stats['windows']+=1;X={s:MK.get((m,s)) for s in (0,1)};fp=fresh(heads,W,X)
        for cfg,place,cancel in CASES:
            with clocks(place,cancel):
                ast=collections.Counter();pol=L.Policy(heads,W,X,'EV',prior['qr_choice'],ast,check,fresh=fp)
                dec={(s,d):pol.dec(s,d) for s in (0,1) for d in range(P.NR)};gate={(s,d):pol.gate(s,d) for s in (0,1) for d in range(P.NR)}
                result=L.simulate(W,dec,gate,50.);v,live=L.value(W,result,SR,IM)
                row=dict(day=day,m=m,cfg=cfg,**v,**{f'live{j}':x for j,x in enumerate(live)})
                if cfg==CASES[0][0]:parity(row,reference.loc[m]);stats['baseline_window_parity']+=1
                if stats['windows']==1:
                    uncached=L.Policy(heads,W,X,'EV',prior['qr_choice'],collections.Counter())
                    rr=L.simulate(W,{(s,d):uncached.dec(s,d) for s in (0,1) for d in range(P.NR)},
                        {(s,d):uncached.gate(s,d) for s in (0,1) for d in range(P.NR)},50.)
                    L.same_trajectory(result,rr);stats['trajectory_parity']+=1
                results.append(row);stats['runs']+=1
        if stats['windows']%36==0:
            print(json.dumps(dict(day=day,windows=stats['windows'],seconds=round(time.time()-start),rss_mib=K.rss_mib())),flush=True);E.mem_guard('EV median sensitivity')
    L.require(stats['windows']==len(reference)==stats['baseline_window_parity'] and stats['trajectory_parity']==2 and stats['state_parity']>0,'incomplete checks')
    df=pd.DataFrame(results);df.to_parquet(out/'windows.parquet',index=False);totals={}
    for cfg,g in df.groupby('cfg',sort=False):
        sums={c:float(g[c].sum()) for c in ['pnl','shares','orders','pulls','fill_events','markout1_cents','markout5_cents','imputed_pnl','imputed_shares','pend','pend_shares','qunk','qunk_shares']}
        sums.update(windows=len(g),cents_per_share=100*sums['pnl']/sums['shares'] if sums['shares'] else None,
            live_share=float(g[[f'live{j}' for j in range(P.NR)]].to_numpy().mean()),max_abs_position=float(g.max_abs_position.max()))
        totals[cfg]=sums
    inputs_unchanged()
    receipt=dict(asof=datetime.now(timezone.utc).isoformat(),day=day,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE.parents[1],text=True).strip(),
        source_sha256=L.sha(__file__),prior_receipt_sha256=L.sha(PREVIOUS/f'receipt_{day}_v3.json'),reference_windows_sha256=L.sha(wp),
        checks=checks,stats=dict(stats),loader=D['stat'],cases=CASES,unchanged_clock=dict(pending_cancel=K.L_PENDCANCEL,fill_notice=P.L_NOTE,grid=50.,trade_shift=50.),
        totals=totals,seconds=time.time()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        limits='Consumed data, deterministic placement/cancel median sensitivity only. No jitter or physical replay validation, no forecast tested, no new fit or threshold selection.')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(day=day,totals=totals,stats=dict(stats))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--day');p.add_argument('--out',type=Path);a=p.parse_args()
    if a.selftest:print(json.dumps(selftest()))
    else:run(a.day,a.out)
