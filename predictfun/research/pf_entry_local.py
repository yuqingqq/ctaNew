"""Single-order entry surrogate; no recursive continuation-value target.

Unconditional heads estimate fill fraction F and fill-weighted movement D.
Score = known entry edge * E[F|x] + E[D|x]. This preserves fill/move dependence;
neither head trains only on fills. Separate bid/ask models, compact causal
inputs, no thresholds selected on evaluation dates. This is local admission,
not an optimal POST-minus-WAIT Q function. Measurement horizons never trigger
runtime cancels or retries. Cancellation remains a separate policy.
"""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_entry_local_data as DATA
import pf_entry_repair_lane as L
import pf_wait_post_state as N
import pf_entry_information as I

M=N.M; E=N.E; K=M.K
DAYS=DATA.DAYS; TESTS=DAYS[2:]
FOLDS={d:tuple(x for x in DAYS if x<d) for d in TESTS}
KEYS=['day','market','side','rung','generation']
SIZE=30.; CAP=1500.; FILL_MS=5000.; MARK_MS=1000.
ARMS=('DIRECT','FACTOR')
ROUNDS=100
# Explicit research change from pinned 31-leaf/200-round/no-L2 entry control.
# Same predeclared regularization for both arms; no per-day tuning.
PARAMS=dict(objective='regression',learning_rate=.05,num_leaves=7,
            min_data_in_leaf=200,lambda_l2=10.,feature_fraction=1.,
            bagging_fraction=1.,num_threads=1,verbose=-1,seed=20261115)
RAW=['bn_r50','bn_r250','bn_r1000','bn_imb','bn_flow250','bn_flow1000',
     'pf_micro_dev','pf_imb1','pf_imb3','pf_dm250','pf_dm1000']
FEATURES=['price','rung','edge','spread','shown_ratio','better_ratio','opposite_ratio',
          'liquidity_log','history_fraction','queue_known','frame_age','bn_age',
          'sigma','basis_bps','fair_edge','reference_known','time_in_window',
          'time_left','idle_age','inventory_lots','capacity_lots',
          'candidate_shrink250','candidate_grow250','candidate_shrink1000',
          'candidate_grow1000',*['side_'+x for x in RAW]]
BASE=Path(K.IO)/'entry_repair_20261115_v2'
OUT=Path(K.IO)/'entry_local_20261115_v1'
STOP=Path(K.IO)/'stop_model_20261114'
STOP_THRESHOLD=.20; STOP_HORIZON=1000
SCENARIOS=((50.,'zero'),(250.,'zero'),(50.,'block'))
LIMITS=[
    'Consumed development dates only; no forward-validation or promotion.',
    'The local score is a surrogate for admission, not POST-minus-WAIT continuation value.',
    'Labels use neutral KEEP with engine safety; they are potential single-order outcomes, not keeper-conditioned Q values.',
    'Five-second fill horizon and one-second per-tranche markout are measurements, never runtime retry/cancel timers.',
    'Later queue option value, other-slot effects and terminal inventory remain full-replay criteria, absent from the local training label.',
    'Gross marks exclude fees/rewards/own impact. Public queue and fill-clock assumptions are unvalidated.',
    'Clock nuisance calibration is inherited; raw-step5 inputs beyond the reviewed correction retain upstream qualifications.',
]


def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f: json.dump(x,f,indent=2,allow_nan=False); f.write('\n')
def stage(out,name):
    p=Path(out)/name; p.mkdir(parents=True,exist_ok=False); return p
def guard(): I.check_memory()


def sampling(day,market,key):
    h=M.hash_key(day,market,key); old=2 if key[2]==0 else 32
    mod=2 if key[2]==0 else 4
    return dict(selected=h%mod==0,sampling_weight=float(mod),
                requested_delay=M.AGES[(h//old)%len(M.AGES)])


def run(W,policy,observe=None,shift=50.,queue='zero',keeper=None):
    def choose(v):
        if v['kind']=='rest': return True if keeper is None else not keeper(v)
        return policy(v)
    r=L.sim_joint(W,'M',shift,CAP,{s:[False]*len(W[s]['tg']) for s in (0,1)},
                  size=SIZE,nr=M.NR,mode='hard',policy=choose,observe=observe,
                  neutral_keeper=keeper is None,unknown_queue=queue)
    if any(r['stat']['viol'].values()): raise ValueError('engine accounting violation')
    E.terminal(W,r)
    return r


def vector(W,v,idle=0.):
    s=v['side']; sign=1. if s==0 else -1.; w=W[s]; t=v['t']; k=v['k']
    if v['kind']!='entry' or v['status']!='eligible' or w['tg'][k]!=t:
        raise ValueError('invalid entry timestamp/status')
    cache=w.setdefault('entry_local_cache',{}); key=(k,v['rung'],v['price'])
    def dynamic(a):
        a=a.copy(); a[FEATURES.index('idle_age')]=idle
        a[FEATURES.index('inventory_lots')]=sign*v['known_position']/SIZE
        a[FEATURES.index('capacity_lots')]=(CAP-(v['exposure_bid'] if s==0 else v['exposure_ask']))/SIZE
        return a
    if key in cache:
        a,mid=cache[key]; return dynamic(a),mid
    j=int(np.searchsorted(w['kf'],t,side='right'))-1
    if j<0: raise ValueError('no known frame')
    raw=dict(zip(K.G32,w['X'][k,:len(K.G32)]))
    prices=w['P5'][j]; sizes=w['S5'][j]; opp=W[1-s]['S5'][j]
    known=np.asarray(w['kf']); start=int(np.searchsorted(known,t-30000.,side='left'))
    pools=np.r_[w['S5'][start:j+1,0],W[1-s]['S5'][start:j+1,0]]
    pools=pools[np.isfinite(pools)&(pools>0)]
    if not len(pools): raise ValueError('missing causal liquidity scale')
    scale=float(np.median(pools)); p=v['price']; best=prices[0]; opposite=w['opp'][j]
    if best<0 or opposite<0: raise ValueError('invalid known book')
    mid=(best+opposite)/2.; spread=abs(opposite-best)
    outside=prices[-1]>=0 and (p<prices[-1] if s==0 else p>prices[-1])
    shown=np.nan if outside else float(sizes[prices==p].sum())
    better=float(sizes[(prices>=0)&((prices>p) if s==0 else (prices<p))].sum())
    reference=w['vkl'][k]!=-1
    x=dict(price=p,rung=v['rung'],edge=sign*(mid-p),spread=spread,
           shown_ratio=shown/scale,better_ratio=better/scale,opposite_ratio=float(opp[0])/scale,
           liquidity_log=np.log1p(scale/SIZE),history_fraction=min(1.,(t-known[0])/30000.),
           queue_known=float(not outside),frame_age=t-known[j],bn_age=raw['bn_age'],
           sigma=raw['sigma'],basis_bps=raw['basis_bps'],
           fair_edge=sign*(raw['pf_mid']+raw['gap']-p) if reference else np.nan,
           reference_known=float(reference),time_in_window=t-w['ws'],time_left=v['end_time']-t,
           idle_age=idle,inventory_lots=sign*v['known_position']/SIZE,
           capacity_lots=(CAP-(v['exposure_bid'] if s==0 else v['exposure_ask']))/SIZE)
    for h in (250,1000):
        j0=int(np.searchsorted(known,t-h,side='right'))-1
        if j0<0: change=np.nan
        else:
            pp=w['P5'][j0]; zz=w['S5'][j0]
            hidden=pp[-1]>=0 and (p<pp[-1] if s==0 else p>pp[-1])
            before=np.nan if hidden else float(zz[pp==p].sum())
            change=(shown-before)/scale
        x[f'candidate_shrink{h}']=max(-change,0.) if np.isfinite(change) else np.nan
        x[f'candidate_grow{h}']=max(change,0.) if np.isfinite(change) else np.nan
    for name in RAW: x['side_'+name]=sign*raw[name]
    a=np.asarray([x[n] for n in FEATURES],np.float32)
    if np.isinf(a).any(): raise ValueError('infinite input')
    cache[key]=(a.copy(),mid)
    return a,mid


def label(W,order,view,entry_mid):
    """Each actual tranche marked at its own time + MARK_MS; all actions kept.

    Stop counting new fills at probe+FILL_MS; no simulated cancellation there.
    The actual order may live longer, which terminal replay will evaluate.
    """
    t=view['t']; s=view['side']; sg=1. if s==0 else -1.
    cutoff=t+FILL_MS
    tf,mid=W[s]['tf_mff']; tf=np.asarray(tf); mid=np.asarray(mid)
    if cutoff>W[s]['te'] or tf[-1]<cutoff+MARK_MS:
        return dict(status='insufficient_horizon')
    if order['ts']!=t or order['pt']!=view['price'] or order['ta']!=view['activation']:
        raise ValueError('focal order identity/timestamp differs')
    total=drift=value=0.; tranches=[]
    for ft,qty in order['fl']:
        if not order['ta']<=ft<order['tr'] or qty<=0:
            raise ValueError('invalid executable tranche')
        if ft>cutoff: continue
        j=int(np.searchsorted(tf,ft+MARK_MS,side='right'))-1
        if j<0 or not np.isfinite(mid[j]) or ft+MARK_MS-tf[j]>1000.:
            return dict(status='missing_future_mark')
        total+=qty; drift+=qty*sg*(mid[j]-entry_mid)/SIZE
        value+=qty*sg*(mid[j]-order['pt'])/SIZE
        tranches.append((ft,qty,float(tf[j]),float(mid[j])))
    fraction=total/SIZE; edge=sg*(entry_mid-order['pt'])
    if not 0<=fraction<=1+1e-8 or abs(value-(edge*fraction+drift))>1e-7:
        raise ValueError('local-value decomposition failed')
    jt=int(np.searchsorted(tf,W[s]['te']+L.TAIL_MS,side='right'))-1
    if jt<0 or not np.isfinite(mid[jt]): raise ValueError('missing terminal diagnostic mark')
    return dict(status='eligible',fill_fraction=fraction,drift_value=drift,local_value=value,
                refused=bool(order['ref']),tranches=len(tranches),tranche_records=json.dumps(tranches),
                focal_terminal_value=sg*(mid[jt]-order['pt'])*sum(q for _,q in order['fl'])/SIZE)


def fork(W,birth,delay):
    target=M.key(birth); probe=[]; xx=[]; mids=[]; seen=[False]
    def choose(v):
        if not seen[0] and M.key(v)==target:
            if v['t']<birth['t']+delay: return False
            seen[0]=True; probe.append(dict(v)); x,mid=vector(W,v,v['t']-birth['t']); xx.append(x); mids.append(mid)
        return True
    r=run(W,choose)
    if not probe: return dict(status='probe_not_reached'),None
    v=probe[0]; o=r['order_states'][v['side'],v['rung']][v['generation']]
    row=label(W,o,v,mids[0]); row.update(probe_t=v['t'],probe_k=v['k'])
    return row,xx[0]


def future_check(W,v,x):
    changed=copy.deepcopy(W)
    for s in (0,1):
        w=changed[s]; a=np.asarray(w['kf'])>v['t']; w['S5'][a]=12345.
        w['P5'][a]=80; w['X'][np.asarray(w['tg'])>v['t']]=777.
        w.pop('entry_local_cache',None)
    np.testing.assert_equal(vector(changed,v,v.get('idle_age',0.))[0],x)
    changed[v['side']]['X'][v['k'],K.G32.index('bn_r250')]=12345.
    changed[v['side']].pop('entry_local_cache',None)
    if np.array_equal(vector(changed,v,v.get('idle_age',0.))[0],x,equal_nan=True):
        raise ValueError('planted current-input mutation escaped')


def build(day,out,max_windows=None):
    if day not in DAYS: raise ValueError('undeclared day')
    p=stage(out,'rows_'+day); data,fix=DATA.load_day(day); records=[]; windows=[]; checked=0
    # Every source neutral action, including unsampled actions, has an identity.
    rp=BASE/('rows_'+day); rr=json.loads((rp/'receipt.json').read_text())
    if digest(rp/'rows.parquet')!=rr['rows_sha256']: raise ValueError('changed reference rows')
    original=pd.read_parquet(rp/'rows.parquet',columns=KEYS+['birth_t','status'])
    old={int(m):g for m,g in original.groupby('market')}
    for W in E.PAIRS.window_pairs(data):
        market=W[0]['m']; births=[]
        ref=run(W,lambda v:True,observe=lambda v,a:births.append(dict(v)) if v['kind']=='entry' and v['status']=='eligible' else None)
        want={(int(x.side),int(x.rung),int(x.generation)):x.birth_t for x in old[market].itertuples()}
        got={M.key(v):v['t'] for v in births}
        if got!=want or len(got)!=len(births): raise ValueError('30-second history changed neutral action identities')
        for b in births:
            sm=sampling(day,market,M.key(b)); row=dict(day=day,market=market,side=b['side'],rung=b['rung'],generation=b['generation'],birth_t=b['t'],**sm,status='not_sampled')
            if sm['selected']:
                result,x=fork(W,b,sm['requested_delay']); row.update(result)
                if x is not None: row.update(dict(zip(FEATURES,map(float,x))))
            records.append(row)
        if births:
            v=births[0]; x,_=vector(W,v); future_check(W,v,x); checked+=1
        windows.append(market)
        if len(windows)%16==0: print(day,'windows',len(windows),'actions',len(records),flush=True)
        guard()
        if max_windows and len(windows)>=max_windows: break
    if max_windows is None and set(windows)!=set(data['W0']): raise ValueError('incomplete window population')
    sealed=original[original.market.isin(fix['sealed_lookback_windows'])].copy()
    for r in sealed.to_dict('records'): records.append({**r,'status':'sealed_lookback_window'})
    f=pd.DataFrame(records)
    if f.duplicated(KEYS).any(): raise ValueError('duplicate action')
    valid=f[f.status=='eligible']; check_rows(valid)
    f.to_parquet(p/'rows.parquet',index=False)
    write(p/'receipt.json',dict(asof=now(),day=day,partial=max_windows is not None,n=len(f),eligible=len(valid),
        statuses={str(k):int(v) for k,v in f.status.value_counts().items()},windows=windows,
        source_status=data['stat'],data_fix=fix,feature_future_checks=checked,
        rows_sha256=digest(p/'rows.parquet'),reference_receipt_sha256=digest(rp/'receipt.json'),limits=LIMITS))


def check_rows(f):
    if f.empty or f.duplicated(KEYS).any(): raise ValueError('empty/duplicate rows')
    y=f[['fill_fraction','drift_value','local_value']].to_numpy(float)
    if not np.isfinite(y).all() or not f.fill_fraction.between(0,1+1e-8).all(): raise ValueError('invalid local labels')
    if not np.allclose(f.edge*f.fill_fraction+f.drift_value,f.local_value,atol=2e-6,rtol=0): raise ValueError('label reconciliation')
    w=np.where(f.generation.to_numpy()==0,2.,4.)
    if not np.array_equal(w,f.sampling_weight.to_numpy()): raise ValueError('wrong inclusion weights')
    if np.isinf(f[FEATURES].to_numpy(float)).any(): raise ValueError('infinite input')


def load(day,out):
    p=out/('rows_'+day); r=json.loads((p/'receipt.json').read_text())
    if r['day']!=day or r['partial'] or digest(p/'rows.parquet')!=r['rows_sha256']: raise ValueError('incomplete/changed rows')
    f=pd.read_parquet(p/'rows.parquet'); f=f[f.status=='eligible'].copy(); check_rows(f); return f


def baseline(tr,te,target):
    # Remove the mechanical edge/price/rung explanation before claiming signal.
    aa=tr.copy(); bb=te.copy()
    aa['bucket']=(aa.price//10).astype(int); bb['bucket']=(bb.price//10).astype(int)
    aa['weighted']=aa[target]*aa.sampling_weight
    sums=aa.groupby(['rung','bucket'])[['weighted','sampling_weight']].sum()
    means=sums.weighted/sums.sampling_weight
    fallback=float(np.average(aa[target],weights=aa.sampling_weight))
    return np.array([means.get((r,b),fallback) for r,b in zip(bb.rung,bb.bucket)])


def fit(day,out):
    import lightgbm as lgb
    if day not in TESTS: raise ValueError('undeclared fold')
    p=stage(out,'fit_'+day); train=pd.concat([load(d,out) for d in FOLDS[day]],ignore_index=True); test=load(day,out)
    parts=[]; hashes={}; cells=[]
    for side in (0,1):
        tr=train[train.side==side]; te=test[test.side==side]; w=tr.sampling_weight.to_numpy(); wt=te.sampling_weight.to_numpy()
        preds={}; bases={}
        for target in ('fill_fraction','drift_value','local_value'):
            model=lgb.train(PARAMS,lgb.Dataset(tr[FEATURES].to_numpy(np.float32),tr[target].to_numpy(float),weight=w/w.mean(),feature_name=FEATURES),ROUNDS)
            path=p/f'{side}_{target}.txt'; model.save_model(str(path)); hashes[path.name]=digest(path)
            pred=model.predict(te[FEATURES].to_numpy(np.float32)); check=lgb.Booster(model_file=str(path)).predict(te[FEATURES].to_numpy(np.float32))
            np.testing.assert_allclose(pred,check,rtol=0,atol=1e-12)
            preds[target]=np.clip(pred,0.,1.) if target=='fill_fraction' else pred
            bases[target]=baseline(tr,te,target)
        for arm in ARMS:
            pred=preds['local_value'] if arm=='DIRECT' else te.edge.to_numpy()*preds['fill_fraction']+preds['drift_value']
            base=te.edge.to_numpy()*bases['fill_fraction']+bases['drift_value']
            y=te.local_value.to_numpy(); avg=float(np.average(tr.local_value,weights=w))
            cells.append(dict(day=day,side=side,arm=arm,n=len(te),sse=float(np.sum(wt*(pred-y)**2)),baseline_sse=float(np.sum(wt*(base-y)**2)),mean_sse=float(np.sum(wt*(avg-y)**2))))
            f=te[KEYS+['probe_t','sampling_weight','local_value','fill_fraction','focal_terminal_value']].copy()
            f['arm']=arm; f['score']=pred; f['baseline_score']=base; f['post_action']=pred>=0; parts.append(f)
        guard(); print(day,'side',side,'fit complete',flush=True)
    pd.concat(parts,ignore_index=True).to_parquet(p/'predictions.parquet',index=False)
    write(p/'receipt.json',dict(asof=now(),day=day,train_dates=FOLDS[day],models=hashes,cells=cells,features=FEATURES,params=PARAMS,rounds=ROUNDS,
        row_receipts={d:digest(out/('rows_'+d)/'receipt.json') for d in (*FOLDS[day],day)},predictions_sha256=digest(p/'predictions.parquet')))


def matched(f,draws=500):
    if draws<200 or f.duplicated(KEYS).any(): raise ValueError('invalid null input')
    selected=f.post_action.to_numpy(bool); w=f.sampling_weight.to_numpy(); n=int(selected.sum())
    if min(n,len(f)-n)<30: return dict(n=len(f),posts=n,draws=draws,status='insufficient_action_variation',passes=False)
    g=f.copy(); g['hour']=(g.probe_t//3600000).astype('int64'); g['initial']=g.generation==0
    null=np.zeros(draws); rng=np.random.default_rng(20261115); mass=0.
    for _,part in g.groupby(['day','side','hour','rung','initial']):
        k=int(part.post_action.sum()); weights=part.sampling_weight.to_numpy()
        if np.ptp(weights)!=0: raise ValueError('unmatched sampling mass')
        if k:
            ix=np.argsort(rng.random((draws,len(part))),axis=1)[:,:k]
            null+=part.local_value.to_numpy()[ix].sum(axis=1)*weights[0]; mass+=k*weights[0]
    if mass!=w[selected].sum(): raise ValueError('null mass mismatch')
    null/=mass; observed=float(np.average(f.local_value.to_numpy()[selected],weights=w[selected]))
    pv=float((1+(null>=observed).sum())/(1+draws)); adjusted=min(1.,pv*4)
    return dict(n=len(f),posts=n,draws=draws,status='matched',observed_local_value=observed,null_mean=float(null.mean()),p_value=pv,bonferroni_p=adjusted,passes=bool(observed>0 and adjusted<.05))


def report(out):
    p=stage(out,'report'); parts=[]; cells=[]; fits={}; rows={}
    for day in DAYS: rows[day]=json.loads((out/('rows_'+day)/'receipt.json').read_text())
    for day in TESTS:
        fp=out/('fit_'+day); r=json.loads((fp/'receipt.json').read_text())
        if digest(fp/'predictions.parquet')!=r['predictions_sha256']: raise ValueError('changed predictions')
        for name,h in r['models'].items():
            if digest(fp/name)!=h: raise ValueError('changed model')
        for d,h in r['row_receipts'].items():
            if digest(out/('rows_'+d)/'receipt.json')!=h: raise ValueError('changed row receipt')
        fits[day]=digest(fp/'receipt.json'); cells.extend(r['cells']); parts.append(pd.read_parquet(fp/'predictions.parquet'))
    cf=pd.DataFrame(cells); f=pd.concat(parts,ignore_index=True); pooled=[]
    for (arm,side),g in cf.groupby(['arm','side']):
        pooled.append(dict(arm=arm,side=int(side),n=int(g.n.sum()),r2_prior_mean=float(1-g.sse.sum()/g.mean_sse.sum()),
            error_reduction_baseline=float(1-g.sse.sum()/g.baseline_sse.sum()),days_better_baseline=int((g.sse<g.baseline_sse).sum())))
    tests=[dict(arm=a,side=int(s),**matched(g)) for (a,s),g in f.groupby(['arm','side'])]
    cf.to_csv(p/'scores.csv',index=False)
    replay=[]; replay_receipts={}
    for d in TESTS:
        rp=out/('replay_'+d); rr=json.loads((rp/'receipt.json').read_text())
        if digest(rp/'windows.csv')!=rr['windows_sha256']: raise ValueError('changed replay')
        replay.append(pd.read_csv(rp/'windows.csv')); replay_receipts[d]=digest(rp/'receipt.json')
    rf=pd.concat(replay,ignore_index=True); rf.to_csv(p/'replay_windows.csv',index=False)
    totals=[]
    for (shift,queue,arm),g in rf.groupby(['shift','queue','arm']):
        totals.append(dict(shift=float(shift),queue=queue,arm=arm,windows=len(g),terminal_pnl=float(g.terminal_pnl.sum()),
            terminal_pnl_without_young=float(g.terminal_pnl_without_young.sum()),shares=float(g.shares.sum()),posts=int(g.posts.sum()),
            pulls=int(g.pulls.sum()),max_inventory=float(g.max_inventory.max()),
            quote_availability=float(g.quote_live_ms.sum()/g.quote_possible_ms.sum()),
            mean_rest_ms=float(g.accepted_resting_ms.sum()/g.accepted_orders.sum()) if g.accepted_orders.sum() else None,
            inventory_squared_seconds=float(g.inventory_squared_seconds.sum())))
    write(p/'result.json',dict(asof=now(),n_test_actions=len(f)//len(ARMS),pooled=pooled,cells=cells,matched_tests=tests,
        local_checks_passed=sum(x['passes'] for x in tests),rows=rows,fit_receipts=fits,features=FEATURES,
        learned_candidates=2,head_fits=30,primary_fill_ms=FILL_MS,mark_ms=MARK_MS,limits=LIMITS,
        replay_totals=totals,replay_receipts=replay_receipts,keeper=dict(version=6,horizon=STOP_HORIZON,threshold=STOP_THRESHOLD),
        portfolio_evaluated=True,forward_validation=False,promoted=False))


def stop_arrays(day):
    receipt=STOP/f'fit_{day}_v6.json'; r=json.loads(receipt.read_text())
    if r['test']!=day or r['version']!='6' or any(d>=day for d in [*r['train_days'],r['cal']]):
        raise ValueError('nonchronological keeper')
    columns=[f'P{STOP_HORIZON}_{j}' for j in range(1,7)]
    path=STOP/f'pred_fold{day}_{day}_v6.parquet'
    f=pd.read_parquet(path,columns=['m','side','k',*columns])
    if f.duplicated(['m','side','k']).any(): raise ValueError('duplicate keeper key')
    arrays={}
    for (m,s),g in f.groupby(['m','side'],sort=False):
        a=np.full((int(g.k.max())+1,6),np.nan); a[g.k.to_numpy(int)]=g[columns].to_numpy(float)
        if np.any(np.isfinite(a)&((a<0)|(a>1))): raise ValueError('invalid keeper probability')
        arrays[int(m),int(s)]=a
    return arrays,dict(receipt_sha256=digest(receipt),predictions_sha256=digest(path))


def stop_rule(W,arrays):
    def cancel(v):
        s=v['side']; k=v['k']; a=arrays.get((W[s]['m'],s)); best=W[s]['best'][k]
        depth=max(0,int(best-v['price'] if s==0 else v['price']-best))
        if best<0 or depth>5 or a is None or k>=len(a): return False
        p=a[k,depth]
        return bool(np.isfinite(p) and p>STOP_THRESHOLD)
    return cancel


def path_metrics(W,r):
    ws=W[0]['ws']; end=W[0]['te']; live=0.; lifetimes=[]
    for orders in r['orders'].values():
        spans=sorted((max(ws,a),min(end,b)) for a,b,_,ref in orders if not ref and min(end,b)>max(ws,a))
        merged=[]
        for a,b in spans:
            lifetimes.append(b-a)
            if merged and a<=merged[-1][1]: merged[-1][1]=max(b,merged[-1][1])
            else: merged.append([a,b])
        live+=sum(b-a for a,b in merged)
    position=0.; previous=ws; risk=0.
    for s,_,t,_,q in sorted(r['fl'],key=lambda x:x[2]):
        risk+=position**2*max(0.,t-previous)/1000.; previous=t; position+=q if s==0 else -q
    risk+=position**2*max(0.,r['terminal_time']-previous)/1000.
    return dict(quote_live_ms=live,quote_possible_ms=2*M.NR*(end-ws),
                accepted_resting_ms=sum(lifetimes),accepted_orders=len(lifetimes),inventory_squared_seconds=risk)


def replay(day,out,max_windows=None):
    import lightgbm as lgb
    p=stage(out,'replay_'+day); fp=out/('fit_'+day); fit_r=json.loads((fp/'receipt.json').read_text())
    models={}
    for s in (0,1):
        for target in ('fill_fraction','drift_value','local_value'):
            path=fp/f'{s}_{target}.txt'
            if digest(path)!=fit_r['models'][path.name]: raise ValueError('changed entry model')
            b=lgb.Booster(model_file=str(path))
            if b.feature_name()!=FEATURES: raise ValueError('entry schema mismatch')
            models[s,target]=b
    train=pd.concat([load(d,out) for d in FOLDS[day]],ignore_index=True)
    lookup={}
    for s in (0,1):
        tr=train[train.side==s].copy(); tr['bucket']=(tr.price//10).astype(int)
        for target in ('fill_fraction','drift_value'):
            tr['weighted']=tr[target]*tr.sampling_weight
            sums=tr.groupby(['rung','bucket'])[['weighted','sampling_weight']].sum()
            lookup[s,target]=(sums.weighted/sums.sampling_weight,float(np.average(tr[target],weights=tr.sampling_weight)))
    arrays,stop_id=stop_arrays(day); data,fix=DATA.load_day(day); rows=[]; windows=[]
    for W in E.PAIRS.window_pairs(data):
        keeper=stop_rule(W,arrays)
        for shift,queue in SCENARIOS:
            for arm in ('ALWAYS','BASELINE','DIRECT','FACTOR'):
                births={}
                def choose(v):
                    born=births.setdefault(M.key(v),v['t'])
                    if arm=='ALWAYS': return True
                    x,_=vector(W,v,v['t']-born); xx=x[None,:]; s=v['side']
                    if arm=='BASELINE':
                        base={}
                        for target in ('fill_fraction','drift_value'):
                            means,fallback=lookup[s,target]; base[target]=means.get((v['rung'],v['price']//10),fallback)
                        return bool(x[FEATURES.index('edge')]*base['fill_fraction']+base['drift_value']>=0.)
                    if arm=='DIRECT': score=models[s,'local_value'].predict(xx)[0]
                    else:
                        fraction=np.clip(models[s,'fill_fraction'].predict(xx)[0],0.,1.)
                        score=x[FEATURES.index('edge')]*fraction+models[s,'drift_value'].predict(xx)[0]
                    return bool(score>=0.)
                r=run(W,choose,shift=shift,queue=queue,keeper=keeper); terminal=E.terminal(W,r)
                mature=0.
                for (s,d),orders in r['order_states'].items():
                    sg=1. if s==0 else -1.
                    for o in orders:
                        for t,q in o['fl']:
                            if t-o['ta']>=100.:
                                mature+=q*sg*(terminal['mark']-o['pt'])/100.
                rows.append(dict(day=day,market=W[0]['m'],shift=shift,queue=queue,arm=arm,
                    terminal_pnl=terminal['pnl'],terminal_pnl_without_young=mature,
                    shares=sum(x[4] for x in r['fl']),posts=sum(len(x) for x in r['orders'].values()),
                    pulls=sum(r['stat']['pulls']),max_inventory=r['stat']['maxpos'],**path_metrics(W,r)))
        windows.append(W[0]['m'])
        if len(windows)%16==0: print(day,'replay windows',len(windows),flush=True); guard()
        if max_windows and len(windows)>=max_windows: break
    if max_windows is None and set(windows)!=set(data['W0']): raise ValueError('incomplete replay')
    pd.DataFrame(rows).to_csv(p/'windows.csv',index=False)
    write(p/'receipt.json',dict(asof=now(),day=day,partial=max_windows is not None,windows=windows,
        windows_sha256=digest(p/'windows.csv'),fit_receipt_sha256=digest(fp/'receipt.json'),keeper_inputs=stop_id,
        keeper_threshold=STOP_THRESHOLD,keeper_horizon=STOP_HORIZON,data_fix=fix,
        entry_has_no_keeper_veto=True,limits=LIMITS))


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['selftest','build','fit','replay','report']); p.add_argument('day',nargs='?')
    p.add_argument('--out',type=Path,default=OUT); p.add_argument('--max-windows',type=int); a=p.parse_args()
    if a.command=='selftest':
        import unittest
        import test_entry_local
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_entry_local))
        if not result.wasSuccessful(): raise SystemExit(1)
    elif a.command=='build': build(a.day,a.out,a.max_windows)
    elif a.command=='fit': fit(a.day,a.out)
    elif a.command=='replay': replay(a.day,a.out,a.max_windows)
    else: report(a.out)


if __name__=='__main__': main()
