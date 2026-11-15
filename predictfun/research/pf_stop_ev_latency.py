"""Frozen-model latency sensitivity and matched-environment model comparison.

All five consumed test dates, all available first-minute windows. No timing
certificate selection; untimed trades imputed with committed b1ea549a builder.
No fit or calibration search. Four configurations: current clocks; zero order
and notification latency with unchanged inputs; zero modeled PF/Binance feed,
decision and order delay on a 50ms grid; last configuration with M0 trade-time
assumption instead of M50. M0 is NOT a network latency setting.

Five rules: keep-all; frozen full EV keep/post >=0; frozen order-only EV >=0;
fill-probability QR with its saved earlier-day thresholds; harm-depth MD1000
with saved thresholds. Every rule uses 50ms review; every entry is freshly
scored (no stale reentry gate). EV's fresh-entry rule is a diagnostic extension
of a keep-value model, not a validated post-vs-wait model. All keep/cancel paths
use the same unmodified activation-queue simulator. All fills are primary;
young/exact/imputed splits are attribution only, never excluded simulations.

No inferential claim, random null, feature selection or policy promotion.
Existing models were trained on the now-qualified timing-selected population;
this tests those frozen models, not a retrained zero-latency model.
"""
import argparse
import collections
from contextlib import contextmanager
from datetime import datetime,timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import pf_stop_ev_play as P
import pf_latency_inputs as ZI

V,E,M,K,AC,QA,SP=P.V,P.E,P.M,P.K,P.AC,P.QA,P.SP
DAYS=tuple(M.TESTS)
CFGS=(('current',False,False,50.),('order_zero',True,False,50.),('modeled_zero',True,True,50.),('modeled_zero_trade0',True,True,0.))
ARMS=('KA','EV','B2','QR','MD1000')
BASE_CLOCK={c:getattr(K,c) for c in ['L_PLACE','L_CANCEL','L_PENDCANCEL','L_FILL']}
BASE_NOTE=M.CL.L_NOTE


def require(ok,msg):
    if not bool(ok):raise ValueError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


@contextmanager
def clock(zero):
    old={c:getattr(K,c) for c in BASE_CLOCK};note=M.CL.L_NOTE;pn=P.L_NOTE
    try:
        for c in old:setattr(K,c,0. if zero else BASE_CLOCK[c])
        M.CL.L_NOTE=P.L_NOTE=0. if zero else BASE_NOTE
        yield
    finally:
        for c,v in old.items():setattr(K,c,v)
        M.CL.L_NOTE=note;P.L_NOTE=pn


class Policy(P.FillPolicy):
    def __init__(self,heads,W,X,arm,choice,stat,check=None,fresh=None):
        super().__init__(heads[0],W,X,choice[0],choice[1],float(W[0]['te'])+200.,stat,check)
        self.heads=heads;self.arm=arm;self.fresh=fresh;self.ix=[V.FULL_IN.index(c) for c in V.B2_IN] if arm=='B2' else None
        self.reg={(s,d):None for s in (0,1) for d in range(P.NR)}

    def _batch(self,k,t):
        self.ck=k;self.cache={};keys=[];rows=[]
        for (s,d),cur in self.reg.items():
            candidates=[None] if self.fresh is None else []
            if cur is not None and cur['tc']>t and not cur['ref'] and cur['ta']<=t:candidates.append(cur)
            for c in candidates:
                v=self._vec(s,d,k,t,c)
                if v is not None:keys.append(((s,d),id(c) if c is not None else None));rows.append(v)
        if rows:
            x=np.vstack(rows);x=x[:,self.ix] if self.ix is not None else x
            p=self.heads[0].predict(x,num_threads=1)
            if self.arm!='QR':p=p*self.heads[1].predict(x,num_threads=1)
            self.cache.update(zip(keys,map(float,p)));self.st['batch_rows']+=len(rows);self.st['batches']+=1

    def prob(self,s,d,k,t,cur):
        if cur is None and self.fresh is not None:
            self.st['fresh_cache_lookups']+=1
            return float(self.fresh[s,d,k])
        if k!=self.ck:self._batch(k,t)
        key=((s,d),id(cur) if cur is not None else None)
        if key not in self.cache:
            v=self._vec(s,d,k,t,cur)
            if v is None:self.cache[key]=math.nan
            else:
                x=v[None,:];x=x[:,self.ix] if self.ix is not None else x
                p=float(self.heads[0].predict(x,num_threads=1)[0])
                if self.arm!='QR':p*=float(self.heads[1].predict(x,num_threads=1)[0])
                self.cache[key]=p
            self.st['single_rows']+=1
        return self.cache[key]

    def dec(self,s,d):
        def f(k,t,cur):
            self.reg[(s,d)]=cur;p=self.prob(s,d,k,t,cur);self.st['resting_decisions']+=1
            return bool(p>self.theta) if self.arm=='QR' else bool(p<0.)
        return f

    def gate(self,s,d):
        def g(k):
            self.reg[(s,d)]=None;p=self.prob(s,d,k,float(self.W[0]['tg'][k]),None);self.st['entry_evaluations']+=1
            return bool(p>=self.off) if self.arm=='QR' else bool(p<0.)
        return g


def fresh_predictions(models,W,X):
    """Fresh-order features do not depend on the simulated policy path.

    Batch exactly the same vectors once per market/knowledge-clock pair, then
    reuse them for every placement and for both execution-latency settings.
    This is an inference optimization, not a rule/threshold change.
    """
    pol=Policy(models['EV'],W,X,'EV',(0.,1.),collections.Counter())
    keys=[];rows=[]
    for k,t in enumerate(W[0]['tg']):
        for s in (0,1):
            for d in range(P.NR):
                v=pol._vec(s,d,k,float(t),None)
                if v is not None:keys.append((s,d,k));rows.append(v)
    out={arm:np.full((2,P.NR,W[0]['n']),np.nan) for arm in ['EV','B2','QR']}
    if not rows:return out
    x=np.vstack(rows);pf=models['EV'][0].predict(x,num_threads=1);vf=models['EV'][1].predict(x,num_threads=1)
    xb=x[:,[V.FULL_IN.index(c) for c in V.B2_IN]];b2=models['B2'][0].predict(xb,num_threads=1)*models['B2'][1].predict(xb,num_threads=1)
    for arm,pred in [('EV',pf*vf),('QR',pf),('B2',b2)]:
        for key,v in zip(keys,pred):out[arm][key]=v
    # Verify batching at scattered rows against independent single-row calls.
    for i in range(0,len(x),997):
        for arm,xx in [('EV',x),('B2',xb)]:
            p=float(models[arm][0].predict(xx[i:i+1],num_threads=1)[0])*float(models[arm][1].predict(xx[i:i+1],num_threads=1)[0])
            require(p==out[arm][keys[i]],'fresh batch/single mismatch')
    return out


def simulate(W,dec=None,gate=None,shift=50.):
    aux={s:[False]*W[s]['n'] for s in (0,1)}
    r=QA.sim_joint_g(W,'M',shift,None,aux,decide=dec,gate=gate,size=30.,mode='none',cooldown=0.)
    require(sum(r['stat']['viol'].values())==0,'lane violation')
    return r


def same_trajectory(a,b):
    for key in ['fills','fill_qunk','orders','cancels','stat']:
        np.testing.assert_equal(a[key],b[key],err_msg=f'cached inference trajectory: {key}')


def value(W,r,src,im):
    tf,mid=W[0]['tf_mff'];v=SP.arm_value2(r,tf,mid,src);term=M.CL.portfolio_terminal(r,tf,mid)
    v.update(orders=sum(len(z) for z in r['orders'].values()),fill_events=sum(len(z) for z in r['fills'].values()),
        terminal_abs_position=abs(term['position']),max_abs_position=r['stat']['maxpos'],imputed_pnl=0.,imputed_shares=0.,
        good_value=0.,gross_loss=0.,markout1_cents=0.,markout5_cents=0.)
    for (s,d),fs in r['fills'].items():
        sg=1 if s==0 else -1
        for f in fs:
            i,price,q=f[:3];x=q*sg*(term['mark']-price)/100.
            v['good_value']+=max(0.,x);v['gross_loss']+=max(0.,-x)
            if im[s][i]:v['imputed_pnl']+=x;v['imputed_shares']+=q
            for h in [1,5]:
                mark=ZI.at(tf,mid,np.array([W[s]['anc'][i]+h*1000.]))[0]
                require(np.isfinite(mark),'missing markout');v[f'markout{h}_cents']+=q*sg*(mark-price)
    return v,SP.live_share(r,float(W[0]['tg'][0]),float(W[0]['te']))


def selftest():
    checks=ZI.selftest();W=M.toy_lane_window([20.],[50],[130.])
    with clock(False):a=simulate(W,shift=0.)
    with clock(True):b=simulate(W,shift=0.)
    require(not a['fl'] and b['fl'],'placement latency control did not fire')
    W=M.toy_lane_window([60.],[50],[130.]);dec={(s,d):(lambda k,t,c:t>=50.) for s in (0,1) for d in range(P.NR)}
    with clock(False):a=simulate(W,dec=dec,shift=0.)
    with clock(True):b=simulate(W,dec=dec,shift=0.)
    require(a['fl'] and not b['fl'],'cancel latency control did not fire')
    same_trajectory(a,a)
    caught=False
    try:same_trajectory(a,b)
    except AssertionError:caught=True
    require(caught,'trajectory mismatch falsifier ineffective')
    class Fake:
        def predict(self,x,**kw):return np.full(len(x),.9)
    w0=E.toy_w(0,[(300.,50,150.,1)])[0];w1=E.toy_w(1,[])[0];W={0:w0,1:w1};n=w0['n']
    X={s:(np.arange(n),np.zeros((n,len(V.F71)),np.float32),np.full(n,100.),np.full(n,51.)) for s in (0,1)}
    pol=Policy((Fake(),None),W,X,'QR',(.1,1.),collections.Counter())
    require(pol.gate(0,0)(1),'off-cadence fresh gate not evaluated')
    # Known-bad stale gate is caught by the same predicate.
    bad=P.FillPolicy(Fake(),W,X,.1,1.,2000.,collections.Counter())
    require(not bad.gate(0,0)(1),'stale gate falsifier ineffective')
    require({c:getattr(K,c) for c in BASE_CLOCK}==BASE_CLOCK and P.L_NOTE==M.CL.L_NOTE==BASE_NOTE,'clock restoration')
    checks.update(placement_clock_fires=True,cancel_clock_fires=True,fresh_gate_every_entry=True,stale_gate_falsifier=True,clock_restores=True,trajectory_mismatch_refused=True)
    return checks


def run(day,out,max_windows=None):
    require(day in DAYS,'unapproved day');require(not out.exists(),'output exists');out.mkdir(parents=True)
    controls=selftest();start=time.time();inputs={};stats=collections.Counter();results=[]
    def track(p):
        p=Path(p);key=str(p)
        if key not in inputs:inputs[key]=dict(exists=p.exists(),sha256=sha(p) if p.exists() else None)
    def js(p):track(p);return json.loads(Path(p).read_text())
    fit=js(Path(V.OUTD)/f'fit2_{day}.json');require(fit['features']==V.FULL_IN,'EV feature contract')
    models={}
    for name,tag in [('EV','full'),('B2','b2')]:
        heads=[]
        for h in ['p','v']:
            art=fit['artifacts'][f'{tag}_{h}'];p=Path(art['path']);track(p);require(sha(p)==art['sha256'],'model hash')
            heads.append(lgb.Booster(model_file=str(p)))
        models[name]=heads
    models['QR']=models['EV']
    qchoice=js(Path(P.OUTD)/f'cal_{day}.json')['choice'];mdchoice=js(Path(M.OUTD)/f'play_cal_{day}_v6.json')['choice'][str(M.HP)]
    mf=js(Path(M.OUTD)/f'fit_{day}_v6.json');require(mf['features']==V.F71 and not mf['norm_inputs'],'harm model representation')
    mheads=[]
    for j in M.JS:
        p=Path(M.OUTD)/'models'/f'fold{day}_H1000_j{j}_v6.txt';track(p);mheads.append(lgb.Booster(model_file=str(p)))
    for f in ['registry.parquet','legs_timed2.parquet','legs_tk.parquet','clock_rl_d2.json','pend_inc.parquet']:track(Path(K.IO)/'step1'/day/f)
    for p in (Path(K.IO)/'step1'/day/'books').glob('part_*.parquet'):
        if K.part_allowed(str(p),day):track(p)
    for f in ['rows_rl_d2_pk.parquet','preds_t60k.parquet']:track(Path(K.IO)/'step5'/day/f)
    for f in [f'rows_{day}.parquet',f'rows_{day}_x5.parquet',f'pred_fold{day}_{day}_v6.parquet']:track(Path(M.OUTD)/f)
    MK=P.market_arrays(day);D=K.load_day(day);cut=M.boundary_cut(day)
    tk=pd.read_parquet(Path(K.IO)/'step1'/day/'legs_tk.parquet',columns=['tk']).tk.to_numpy();LT=D['LT'];ix=LT.index.to_numpy()
    grp=np.where(tk[ix]>=0,tk[ix],-1-ix);WSg=pd.Series(M.TS.ws_groups(LT.how.to_numpy(),grp),index=LT.index);GRP=pd.Series(grp,index=LT.index)
    RW,ust=E.raw_windows(day,D);PA=SP.pred_arrays(str(Path(M.OUTD)/f'pred_fold{day}_{day}_v6.parquet'),M.HP)[0]
    ZERO,zstat=ZI.build(day,D,track)
    def check(W,cur,key,t,f):
        stats['state_calls']+=1
        if stats['state_calls']%197:return
        g=AC.known_state(W,cur,key+(0,),t,30.,terminal_ms=W[0]['te']+200.,settings=AC.Settings(notification_ms=P.L_NOTE,cancel_ms=K.L_CANCEL))
        require(all((math.isnan(float(f[c])) and math.isnan(float(g[c]))) or abs(float(f[c])-float(g[c]))<1e-9 for c in P.ORDER_KEYS),'state parity')
        stats['state_parity']+=1
    for W in M.P1.window_pairs(D):
        if W[0]['ws']<cut:stats['boundary_windows']+=1;continue
        if max_windows is not None and stats['windows']>=max_windows:break
        m=int(W[0]['m']);LR={};SR={};GR={}
        for s in (0,1):
            if (m,s) in D['LI']:
                g=LT.iloc[D['LI'][(m,s)]];o=np.lexsort(((-g.ptick.to_numpy() if s==0 else g.ptick.to_numpy()),g.anc.to_numpy()));lr=g.index.to_numpy()[o]
            else:lr=np.zeros(0,dtype=int)
            require(len(lr)==len(W[s]['anc']) and np.array_equal(LT.loc[lr,'anc'].to_numpy(),W[s]['anc']),'tape identity')
            LR[s]=lr;SR[s]=WSg.reindex(lr).to_numpy(bool);GR[s]=GRP.reindex(lr).to_numpy('int64')
        W,LR,SR,GR,IM=E.impute_window(W,LR,SR,GR,RW.get(m,{}).get('imp',[]),ust['imp_off_ms'])
        stats['windows']+=1;stats['imputed_legs']+=sum(int(IM[s].sum()) for s in (0,1))
        for s in (0,1):stats['tape_legs']+=len(LR[s])
        X={s:MK.get((m,s)) for s in (0,1)};Wz,Xz=ZI.window(W,ZERO[m])
        fresh={False:fresh_predictions(models,W,X),True:fresh_predictions(models,Wz,Xz)}
        for cfg,zero,feed,shift in CFGS:
            w=Wz if feed else W;x=Xz if feed else X
            pp={}
            for s in (0,1):
                if not feed:pp[s]=PA.get((m,s));continue
                ix,xx,_,_=x[s];raw=np.column_stack([b.predict(xx,num_iteration=mf['info'][f'H1000_j{j}']['rounds'],num_threads=1).astype(np.float32) for j,b in zip(M.JS,mheads)])
                if mf['platt']:
                    for j in M.JS:raw[:,j-1]=M.platt_apply(raw[:,j-1],mf['platt'][f'H1000_j{j}'])
                if mf['decisions']['cummin'] and mf['decisions']['target']=='U':raw=np.minimum.accumulate(raw,axis=1)
                pp[s]=[np.where(ix>=0,raw[np.maximum(ix,0),j],np.nan) for j in range(6)]
            with clock(zero):
                for arm in ARMS:
                    dec=gate=None;ast=collections.Counter()
                    if arm in models:
                        pol=Policy(models[arm],w,x,arm,qchoice,ast,check,fresh=fresh[feed][arm])
                        dec={(s,d):pol.dec(s,d) for s in (0,1) for d in range(P.NR)};gate={(s,d):pol.gate(s,d) for s in (0,1) for d in range(P.NR)}
                    elif arm=='MD1000':
                        dec={(s,d):SP.model_dec(pp[s],w[s]['best'],s,mdchoice[0]) for s in (0,1) for d in range(P.NR)}
                        gate={(s,d):SP.model_gate(pp[s],d,mdchoice[0]*mdchoice[1]) for s in (0,1) for d in range(P.NR)}
                    r=simulate(w,dec,gate,shift);val,live=value(w,r,SR,IM)
                    if stats['windows']==1 and arm in models:
                        ref=Policy(models[arm],w,x,arm,qchoice,collections.Counter())
                        rr=simulate(w,{(s,d):ref.dec(s,d) for s in (0,1) for d in range(P.NR)},
                            {(s,d):ref.gate(s,d) for s in (0,1) for d in range(P.NR)},shift)
                        same_trajectory(r,rr);stats['cached_trajectory_parity']+=1
                    results.append(dict(day=day,m=m,hour=int(W[0]['ws']//3600000)%24,cfg=cfg,arm=arm,**val,
                        **{f'live{j}':v for j,v in enumerate(live)},**{f'policy_{k}':v for k,v in ast.items()}))
                    stats['runs']+=1
        if stats['windows']%12==0:
            print(json.dumps(dict(day=day,windows=stats['windows'],seconds=round(time.time()-start),rss_mib=K.rss_mib())),flush=True)
            E.mem_guard('latency replay')
    pd.DataFrame(results).to_parquet(out/'windows.parquet',index=False)
    totals={}
    for (cfg,arm),g in pd.DataFrame(results).groupby(['cfg','arm'],sort=False):
        sums={c:float(g[c].sum()) for c in ['pnl','pnl_old','shares','shares_young','pulls','orders','fill_events','good_value','gross_loss','markout1_cents','markout5_cents','imputed_pnl','imputed_shares','pend_shares','qunk_shares']}
        sums.update(windows=len(g),cents_per_share=100*sums['pnl']/sums['shares'] if sums['shares'] else None,
            live_share=float(g[[f'live{j}' for j in range(P.NR)]].to_numpy().mean()),max_abs_position=float(g.max_abs_position.max()))
        totals.setdefault(cfg,{})[arm]=sums
    for p,z in inputs.items():require(Path(p).exists()==z['exists'] and (not z['exists'] or sha(p)==z['sha256']),'input changed')
    require(stats['state_parity']>0,'no parity control observed')
    require(stats['cached_trajectory_parity']==12,'incomplete cached trajectory parity')
    receipt=dict(asof=datetime.now(timezone.utc).isoformat(),day=day,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
        sources={p.name:sha(p) for p in Path(__file__).parent.glob('pf_*.py')},inputs=inputs,controls=controls,stats=dict(stats),loader=D['stat'],zero_inputs=zstat,
        imputation=ust,base_clock=BASE_CLOCK,base_note=BASE_NOTE,qr_choice=qchoice,md_choice=mdchoice,totals=totals,
        smoke_only=max_windows is not None,seconds=time.time()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        limits='Consumed data; unvalidated queue and event-time clocks; zero modeled transport remains 50ms-grid sampled with fixed pre-window sigma. Frozen models trained on timing-selected data, no refit. EV fresh-entry is diagnostic. No significance/promotion claim; model PnL comparisons have different endogenous volumes.')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(day=day,stats=dict(stats),seconds=receipt['seconds'],totals=totals)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--day');p.add_argument('--out',type=Path);p.add_argument('--max-windows',type=int)
    a=p.parse_args()
    if a.selftest:print(json.dumps(selftest()))
    else:run(a.day,a.out,a.max_windows)
