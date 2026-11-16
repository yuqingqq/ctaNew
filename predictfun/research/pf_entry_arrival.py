"""Frozen-teacher entry anticipation, v1. Research only.

One entry row per placement on the neutral reference at 58/37 ms. Current
83 inputs predict the frozen v3 teacher's score for the SAME absolute price
at activation t+58, with zero modeled feed delay there. Bid and ask residual
regressions are separate. The resting-order guard is unchanged in replay.
Future post-only rejection has zero execution value, not an excluded row.
Missing teacher support is an explicit status. No forecast of cancellation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb
import pf_ev_lag_gap as G

L,P,V,E,M,K,AC=G.L,G.P,G.V,G.E,G.M,G.K,G.AC
ZI=L.ZI
HERE=Path(__file__).resolve().parent
ROOT=Path(K.IO)/'entry_arrival_20261116_v1'
REVIEW=HERE.parent/'reviews/entry_arrival_20261116'
PLACE,CANCEL=58.,37.
FEATURES=tuple(V.FULL_IN)+('current_teacher_score',)
DAYS=tuple(V.DAYS9)


def stamp():return datetime.now(timezone.utc).isoformat()
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def commit():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE.parents[1],text=True).strip()
def score(heads,x):return heads[0].predict(x,num_threads=1)*heads[1].predict(x,num_threads=1)


def teacher(day):
    path=Path(V.OUTD)/f'fit3_{day}.json';j=json.loads(path.read_text())
    L.require(j['version']=='v3' and j['features']==V.FULL_IN and j['reload_pass'],'teacher contract')
    L.require(max(j['train_days'])<j['cal']<day,'teacher chronology')
    heads=[]
    for h in ['p','v']:
        a=j['artifacts'][f'full_{h}'];L.require(L.sha(a['path'])==a['sha256'],'teacher hash')
        heads.append(lgb.Booster(model_file=a['path']))
    return heads,j


def fixed_fresh(w,k,t,side,rung,price,terminal):
    """Fresh quote at a specified price; future best must never change price."""
    jt=int(w['jk'][k])
    if jt<0 or w['best'][k]<0:return None,'missing_teacher_book'
    opp=float(w['opp'][jt])
    if (side==0 and price>=opp) or (side==1 and price<=opp):return None,'reject_at_activation'
    sh=P.shown(w,jt,side,price);best=float(w['P5'][jt][0])
    return dict(side=side,price=float(price),rung=rung,depth=float(best-price if side==0 else price-best),age_ms=0.,
        submitted_quantity=30.,known_remaining=30.,known_filled=0.,queue_initial_known=sh,queue_ahead_known=sh,
        shown_known=sh,queue_initial_unknown=float(not np.isfinite(sh)),effective_horizon_ms=min(1000.,terminal-t)),'supported'


def shifted_data(D,offset):
    return dict(D,ROWS={m:r.assign(tau=r.tau+offset,t_rel=r.t_rel+offset/1000.) for m,r in D['ROWS'].items()})


def at_arrival(W,z,offset=PLACE):
    shifted={s:dict(w,tg=np.asarray(w['tg'])+offset,cache={}) for s,w in W.items()}
    return ZI.window(shifted,z)


def vectors(W,X,F=None,FX=None):
    """All legal current candidates; future data only enters returned targets."""
    keys=[];xs=[];fs=[];status=[]
    pol=L.Policy((None,None),W,X,'EV',(0.,1.),Counter())
    terminal=float(W[0]['te'])+200.
    for k,t in enumerate(W[0]['tg']):
        for s in (0,1):
            for d in range(P.NR):
                x=pol._vec(s,d,k,float(t),None)
                if x is None:continue
                keys.append((s,d,k));xs.append(x)
                if F is None:continue
                price=float(x[V.FULL_IN.index('o_price')]);f,why=fixed_fresh(F[s],k,float(t)+PLACE,s,d,price,terminal)
                target=np.full(len(V.FULL_IN),np.nan,dtype=np.float32)
                if why=='supported':
                    ix,market,scale,mid=FX[s];j=int(ix[k])
                    if j<0:why='missing_teacher_market'
                    else:target=np.r_[market[j],np.asarray(P.order_vector(f,scale[j],mid[j]),np.float32)]
                fs.append(target);status.append(why)
    return np.asarray(keys,np.int64).reshape(-1,3),np.asarray(xs,np.float32).reshape(-1,len(V.FULL_IN)),np.asarray(fs,np.float32).reshape(-1,len(V.FULL_IN)),np.asarray(status)


def future_scores(heads,target,status):
    y=np.full(len(status),np.nan);ok=status=='supported'
    if ok.any():y[ok]=score(heads,target[ok])
    y[status=='reject_at_activation']=0.
    return y


def test_chronology(train,test):
    L.require(bool(train) and len(train)==len(set(train)) and max(train)<test,'student chronology')


def selftest():
    checks=G.selftest()
    W=M.toy_lane_window([],[],[])
    for w in W.values():
        w['kf']=w['tf'];w['jk']=np.searchsorted(w['kf'],w['tg'],side='right')-1
    w=W[0];terminal=W[0]['te']+200.;t=float(w['tg'][0]);p=int(w['best'][0])
    f,status=fixed_fresh(w,0,t,0,0,p,terminal)
    L.require(status=='supported' and f['price']==p,'fixed fresh positive')
    ref=P.fresh_state(w,0,t,0,0,terminal)
    for k in ref:L.require((np.isnan(ref[k]) and np.isnan(f[k])) or ref[k]==f[k],'fresh parity')
    f2,_=fixed_fresh(w,0,t,0,1,p-2,terminal)
    L.require(f2['price']==p-2 and f2['price']!=P.fresh_state(w,0,t,0,1,terminal)['price'],'rung substitution falsifier')
    _,status=fixed_fresh(w,0,t,0,0,float(w['opp'][int(w['jk'][0])]),terminal)
    L.require(status=='reject_at_activation','post-only boundary')
    test_chronology(['20261023'],'20261024');caught=False
    try:test_chronology(['20261024'],'20261024')
    except ValueError:caught=True
    L.require(caught,'same-date training not refused')
    x=np.ones((3,len(V.FULL_IN)),np.float32);y=x.copy();y[:]=999.
    inp=student_matrix(x,np.array([1.,2.,3.]))
    L.require(np.array_equal(inp[:,:-1],x) and not np.array_equal(inp[:,:-1],y),'future input isolation')
    checks.update(fixed_price=True,wrong_rung_price_detected=True,post_only_status=True,chronology_refusal=True,future_input_isolation=True)
    return checks


def student_matrix(x,baseline):
    L.require(x.ndim==2 and x.shape[1]==len(V.FULL_IN) and len(x)==len(baseline),'input schema')
    return np.column_stack([x,baseline]).astype(np.float32)


def data_inputs(day):
    paths=[]
    for name in ['registry.parquet','legs_timed2.parquet','legs_tk.parquet','clock_rl_d2.json','pend_inc.parquet']:
        paths.append(Path(K.IO)/'step1'/day/name)
    paths.extend(p for p in (Path(K.IO)/'step1'/day/'books').glob('part_*.parquet') if K.part_allowed(str(p),day))
    paths.extend(Path(K.IO)/'step5'/day/f for f in ['rows_rl_d2_pk.parquet','preds_t60k.parquet'])
    paths.extend(Path(M.OUTD)/f for f in [f'rows_{day}.parquet',f'rows_{day}_x5.parquet'])
    return paths


def load(day):
    D=K.load_day(day);MK=P.market_arrays(day);LT=D['LT'];ix=LT.index.to_numpy()
    tk=pd.read_parquet(Path(K.IO)/'step1'/day/'legs_tk.parquet',columns=['tk']).tk.to_numpy()
    grp=np.where(tk[ix]>=0,tk[ix],-1-ix)
    WSg=pd.Series(M.TS.ws_groups(LT.how.to_numpy(),grp),index=LT.index);GRP=pd.Series(grp,index=LT.index)
    RW,ust=E.raw_windows(day,D)
    def windows():
        for W in M.P1.window_pairs(D):
            if W[0]['ws']<M.boundary_cut(day):continue
            m=int(W[0]['m']);LR={};SR={};GR={}
            for s in (0,1):
                if (m,s) in D['LI']:
                    g=LT.iloc[D['LI'][(m,s)]];o=np.lexsort(((-g.ptick.to_numpy() if s==0 else g.ptick.to_numpy()),g.anc.to_numpy()));lr=g.index.to_numpy()[o]
                else:lr=np.zeros(0,dtype=int)
                L.require(len(lr)==len(W[s]['anc']) and np.array_equal(LT.loc[lr,'anc'].to_numpy(),W[s]['anc']),'tape identity')
                LR[s]=lr;SR[s]=WSg.reindex(lr).to_numpy(bool);GR[s]=GRP.reindex(lr).to_numpy('int64')
            W,LR,SR,GR,IM=E.impute_window(W,LR,SR,GR,RW.get(m,{}).get('imp',[]),ust['imp_off_ms'])
            yield W,{s:MK.get((m,s)) for s in (0,1)},SR,IM
    return D,windows,ust


def build(day,root=ROOT,limit=None):
    L.require(day in DAYS,'undeclared day');out=root/'data'/day
    L.require(not out.exists(),'data output exists');out.mkdir(parents=True)
    start=time.time();checks=selftest();inputs={};counts=Counter();xx=[];ff=[];ss=[];ids=[]
    def track(p):
        p=Path(p)
        if str(p) not in inputs:inputs[str(p)]=dict(exists=p.exists(),sha256=L.sha(p) if p.exists() else None)
    for p in data_inputs(day):track(p)
    D,windows,imputation=load(day)
    # Exactly 58 ms, never round placement to the next 50 ms review.
    ZERO,zstat=ZI.build(day,shifted_data(D,PLACE),track)
    heads,fit=teacher(day) if day in L.DAYS else (None,None)
    if heads is not None:(out/'oracle').mkdir()
    for W,X,SR,IM in windows():
        if limit is not None and counts['windows']>=limit:break
        m=int(W[0]['m']);F,FX=at_arrival(W,ZERO[m]);keys,x,f,status=vectors(W,X,F,FX)
        L.require(np.array_equal(np.asarray(F[0]['tg']),np.asarray(W[0]['tg'])+PLACE),'arrival clock alignment')
        ix={tuple(k):j for j,k in enumerate(keys)}
        if heads is not None:
            a=np.full((2,P.NR,W[0]['n']),np.nan);y=future_scores(heads,f,status)
            for k,v in zip(keys,y):a[tuple(k)]=v
            np.savez_compressed(out/'oracle'/f'{m}.npz',score=a)
        with G.clocks(PLACE,CANCEL):neutral=AC.replay_neutral(W,size=30.,shift_ms=50.,nr=P.NR)
        L.require(sum(neutral['stat']['viol'].values())==0,'neutral lane violations')
        for (s,d),orders in neutral['order_states'].items():
            for gen,o in enumerate(orders):
                k=int(np.searchsorted(W[s]['tg'],o['ts']));j=ix.get((s,d,k));counts['neutral_posts']+=1
                if j is None:
                    counts['status_missing_current_inputs']+=1
                    xx.append(np.full(len(V.FULL_IN),np.nan,np.float32));ff.append(np.full(len(V.FULL_IN),np.nan,np.float32))
                    ss.append('missing_current_inputs');ids.append((m,s,d,gen,k,float(o['ts']),float(o['pt'])));continue
                L.require(float(x[j,V.FULL_IN.index('o_price')])==o['pt'],'neutral price identity')
                xx.append(x[j]);ff.append(f[j]);ss.append(status[j]);ids.append((m,s,d,gen,k,float(o['ts']),float(o['pt'])))
                counts['status_'+status[j]]+=1
        counts['windows']+=1
        if counts['windows']%24==0:
            print(json.dumps(dict(day=day,stats=dict(counts),seconds=round(time.time()-start),rss_mib=K.rss_mib())),flush=True);E.mem_guard('entry arrival build')
    np.savez_compressed(out/'rows.npz',current=np.asarray(xx,np.float32),future=np.asarray(ff,np.float32),status=np.asarray(ss),ids=np.asarray(ids))
    L.require(len(ids)==len(set(tuple(i[:4]) for i in ids)),'duplicate placement actions')
    for p,v in inputs.items():L.require(Path(p).exists()==v['exists'] and (not v['exists'] or L.sha(p)==v['sha256']),'changed build input')
    receipt=dict(asof=stamp(),day=day,source_commit=commit(),source_sha256=L.sha(__file__),features=V.FULL_IN,counts=dict(counts),
        checks=checks,inputs=inputs,imputation=imputation,zero_inputs=zstat,rows_sha256=L.sha(out/'rows.npz'),
        teacher_fit_sha256=L.sha(Path(V.OUTD)/f'fit3_{day}.json') if heads else None,
        oracle_hashes={p.name:L.sha(p) for p in (out/'oracle').glob('*.npz')} if heads else {},
        seconds=time.time()-start,smoke_only=limit is not None,limits='Neutral placements; exact t+58 fixed-price fresh teacher. No cancellation forecast. Missing support has status; post-only rejections retained as zero. Consumed data.')
    dump(out/'receipt.json',receipt);print(json.dumps(dict(day=day,counts=dict(counts),seconds=receipt['seconds'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['selftest','build']);p.add_argument('--day');p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.task=='selftest':print(json.dumps(selftest()))
    else:build(a.day,a.root,a.limit)


if __name__=='__main__':main()
