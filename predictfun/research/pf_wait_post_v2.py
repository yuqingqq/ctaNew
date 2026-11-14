"""Aligned admission + continuation-value research model, wait/post v2.

Qpost and Qwait are whole joint-portfolio terminal cash+inventory differences
relative to disabling this empty rung, under ONE frozen continuation policy.
WAIT skips only the current action; future entry is decided by that policy,
which can keep waiting indefinitely. Positive ties POST to retain queue priority.
POST iff predicted Qpost > 0 and Qpost >= Qwait. No cooldown/guard entry veto.
Keeper is fixed neutral KEEP (engine safety still applies), in labels and replay.

Reference population: every neutral POST/KEEP empty generation, independent hash
sampling (first generation 1/2, subsequent 1/32). Each sampled generation is
held empty for a predeclared hash age, then probed at its first legal decision.
The complete prefix is shared by POST, WAIT and NEVER branches. Other slots use
the neutral reference until the probe, then the same frozen continuation. One
row per reference generation; unsampled, unreached and invalid inputs retained.

Chronological policy improvement, no recursive held-out-day leakage:
  bootstrap labels Oct23/24 under FLAT entry continuation -> pi1 admission model;
  labels on all later consumed dates under frozen pi1 -> two value heads/side;
  each pi2 fold trains on those label days strictly before its test date.
Five test dates Oct27,Oct31,Nov1,Nov2,Nov3. No calibration/threshold tuning.
This is finite policy improvement, not an optimality/convergence claim.

Raw economic targets, no clipping or extra inventory penalty; pinned LGBM params.
The zero wait head of FLAT bootstrap is structural, not a failed fit. Learned
heads require adequate support and actual splits/prediction variation.
Acceptance requires absolute/relative portfolio gates AND matched local timing
skill AND a 200-draw M50 portfolio null matching placements per window and side.
The expensive portfolio null runs only if earlier gates pass. A skipped or
insufficient null can NEVER yield acceptance. Consumed dates, no forward claim.
"""
import argparse
import ast
import collections
import functools
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pf_wait_post as V1
import pf_entry_value as E
import pf_wait_post_v2_lane as L

K=E.K; SIZE=E.SIZE; NR=E.NR; CAP=E.CAP; SEED=20261115
DAYS=tuple(K.DAYS9); BOOT_DAYS=DAYS[:2]; LATER=DAYS[2:]; TESTS=DAYS[4:]
FOLDS={d:tuple(x for x in LATER if x<d) for d in TESTS}
AGES=(0.,50.,100.,200.,400.,800.,1600.,3200.,6400.,12800.,25600.,51200.)
FEATURES=V1.FEATURES+['idle_age_ms','side_price_change_while_empty']
DIR_INDEX=[E.FEATURES.index(n) for n in V1.DIRECTIONAL]
ARMS=('VALUE','ADMISSION','ALWAYS','SCHEDULED','TIMER','FLAT'); CFGS=(0.,50.,250.)
BASELINES=('ADMISSION','ALWAYS','SCHEDULED','TIMER')
MIN_N=1000; MIN_NONZERO=100; NULL_DRAWS=200; NULL_MAX_ATTEMPTS=2000
DEFAULT_OUT=Path(K.IO)/'wait_post_v2_20261114_v1'


def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@functools.lru_cache(maxsize=1)
def sources():
    pending=['pf_wait_post_v2']; found=set()
    while pending:
        n=pending.pop()
        if n in found:continue
        found.add(n)
        for node in ast.walk(ast.parse((HERE/(n+'.py')).read_text())):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            pending.extend(x for x in names if x and x.startswith('pf_') and '.' not in x and x not in found)
    return sorted(n+'.py' for n in found)

def stamp():
    return dict(contract='wait-post-learned-continuation-v2',features=FEATURES,decision_ms=50,size=SIZE,nr=NR,cap=CAP,
        seed=SEED,probe_ages_ms=list(AGES),generation_sampling={'first':2,'later':32},keeper='neutral_keep_with_engine_safety',
        target='terminal_POST_and_WAIT_minus_NEVER_under_identical_frozen_policy',raw_targets=True,
        sources={n:digest(HERE/n) for n in sources()})

def require(meta):
    for k,v in stamp().items():
        if meta.get(k)!=v:raise ValueError('v2 artifact differs: '+k)

def stage(out,name):return E.new_stage(out,name)

def key(v):return (v['side'],v['rung'],v['generation'])

def action_key(v):return (v['k'],v['side'],v['rung'],'entry',v['generation'])

def hash_key(day,market,unit):
    return int.from_bytes(hashlib.sha256(json.dumps([SEED,str(day),int(market),list(unit)]).encode()).digest()[:8],'big')


class Tracker:
    def __init__(self):self.births={}
    def see(self,v):
        return self.births.setdefault(key(v),(v['t'],v['price']))


def inputs(W,v,tracker):
    """O(5) received-book entry features; exact v1 features plus waiting state."""
    if v['kind']!='entry' or v['side'] not in (0,1):raise ValueError('entry state required')
    w=W[v['side']]; t=v['t']; j=int(np.searchsorted(w['kf'],t,side='right'))-1; k=v['k']; s=v['side']
    if j<0 or not 0<=k<len(w['X']):raise ValueError('no known input')
    raw=np.asarray(w['X'][k,:len(K.G32)],float)
    if len(raw)!=len(K.G32) or not np.isfinite(raw).any():raise ValueError('missing market input')
    p=v['price']; prices=w['P5'][j]; sizes=w['S5'][j]; best=int(prices[0]); opp=int(w['opp'][j]); sign=1. if s==0 else -1.
    if best<0 or opp<0 or (best>=opp if s==0 else best<=opp):raise ValueError('invalid known book')
    outside=prices[4]>=0 and (p<prices[4] if s==0 else p>prices[4])
    shown=np.nan if outside else float(sizes[prices==p].sum())
    better=float(sizes[(prices>=0)&((prices>p) if s==0 else (prices<p))].sum())
    own=v['exposure_bid'] if s==0 else v['exposure_ask']; other=v['exposure_ask'] if s==0 else v['exposure_bid']
    cap=CAP if v['cap'] is None else v['cap']
    extra=[sign,v['rung'],p,sign*(best-p),abs(opp-best),0.,v['quantity'],v['known_filled'],shown,shown,shown,better,
           1. if np.isfinite(shown) and shown>0 else np.nan,v['known_position'],own,other,cap-own,v['end_time']-t,t-w['kf'][j]]
    old=np.asarray(list(raw)+extra,float); born,price0=tracker.see(v)
    x=np.asarray([*old,*[sign*old[i] for i in DIR_INDEX],sign*v['known_position'],t-born,sign*(p-price0)],np.float32)
    if np.isinf(x).any():raise ValueError('infinite feature')
    return x


def post_rule(post,wait):
    if not np.isfinite(post) or not np.isfinite(wait):return False
    return bool(post>0. and post>=wait)


class Forecast:
    def __init__(self,models=None,identity='FLAT'):
        self.models=models;self.identity=identity;self.cache={};self.calls=0;self.hits=0
    def clear(self):self.cache.clear()
    def values(self,side,x):
        if self.models is None:return (0.,0.)
        item=(side,x.tobytes())
        if item in self.cache:self.hits+=1;return self.cache[item]
        self.calls+=1; heads=self.models[side]
        result=(float(heads['post'].predict(x[None,:])[0]),float(heads['wait'].predict(x[None,:])[0]) if 'wait' in heads else 0.)
        if len(self.cache)>=50000:self.cache.clear()
        self.cache[item]=result;return result


def learned(W,forecast,tracker,v,capture=None):
    tracker.see(v)
    if forecast.models is None:return False
    try:x=inputs(W,v,tracker)
    except ValueError:
        if capture:capture(v,None,None,False,'missing_input')
        return False
    values=forecast.values(v['side'],x); action=post_rule(*values)
    if capture:capture(v,x,values,action,'eligible' if np.isfinite(values).all() else 'invalid_forecast')
    return action


def run(W,choose,shift=50.,observe=None,entry_mode='value'):
    guards={s:(np.asarray(W[s]['vkl'])<=K.GUARD_PULL).tolist() for s in (0,1)}
    r=L.sim_joint(W,'M',shift,CAP,guards,size=SIZE,nr=NR,mode='hard',policy=choose,
                  observe=observe,entry_mode=entry_mode,neutral_keeper=True)
    if any(r['stat']['viol'].values()):raise ValueError('engine accounting violation')
    E.terminal(W,r);return r


def fork(W,birth,delay,continuation,branch):
    """Same neutral prefix, causal delayed opportunity, then common policy."""
    if branch not in ('post','wait','never'):raise ValueError('unknown branch')
    tracker=Tracker(); born=[False]; probe=[]; target=key(birth); started=[False]
    def choose(v):
        tracker.see(v)
        if not born[0]:
            if action_key(v)!=action_key(birth):return True
            if v!=birth:raise ValueError('reference birth state differs')
            born[0]=True
        if not started[0]:
            if key(v)!=target:return True
            if v['t']<birth['t']+delay:return False
            probe.append(dict(v));started[0]=True
            return branch=='post'
        if branch=='never' and (v['side'],v['rung'])==(birth['side'],birth['rung']):return False
        return learned(W,continuation,tracker,v)
    r=run(W,choose)
    if not born[0]:raise ValueError('missing neutral birth')
    return r,probe,tracker


def paired(W,birth,delay,continuation):
    post,pv,pt=fork(W,birth,delay,continuation,'post')
    if not pv:return dict(status='probe_not_reached'),None
    wait,wv,wt=fork(W,birth,delay,continuation,'wait')
    never,nv,nt=(wait,wv,wt) if continuation.models is None else fork(W,birth,delay,continuation,'never')
    if len(pv)!=1 or pv!=wv or pv!=nv:raise ValueError('branch decision states differ')
    target=action_key(pv[0]); prefixes=[]
    for branch,expected in ((post,True),(wait,False),(never,False)):
        actions=branch['policy_decisions'];idx=next(i for i,a in enumerate(actions) if tuple(a[:5])==target)
        if bool(actions[idx][-1])!=expected:raise ValueError('wrong branch action')
        prefixes.append(actions[:idx])
    if prefixes[0]!=prefixes[1] or prefixes[0]!=prefixes[2]:raise ValueError('branch prefixes differ')
    values=[E.terminal(W,z) for z in (post,wait,never)]
    if len({v['time'] for v in values})!=1:raise ValueError('terminal times differ')
    try:x=inputs(W,pv[0],pt)
    except ValueError:return dict(status='missing_input'),None
    # The tracker is append-only: the selected generation's birth is unchanged
    # after subsequent actions. Counterfactual actual inventory never enters x.
    qp=100*(values[0]['pnl']-values[2]['pnl'])/SIZE
    qw=100*(values[1]['pnl']-values[2]['pnl'])/SIZE
    return dict(status='eligible',qpost=qp,qwait=qw,delta=qp-qw,
                value_post=values[0]['pnl'],value_wait=values[1]['pnl'],value_never=values[2]['pnl'],
                probe_k=pv[0]['k'],probe_t=pv[0]['t'],idle_age=pv[0]['t']-birth['t']),x


def build_window(W,day,market,continuation,all_sampled=False):
    births=[];statuses=collections.Counter()
    def observe(v,action):
        statuses[v['status']]+=1
        if v['status']=='eligible':births.append(dict(v))
    run(W,lambda v:True,observe=observe)
    records=[];seen=set();continuation.clear()
    for b in births:
        unit=key(b)
        if unit in seen:raise ValueError('duplicate neutral generation')
        seen.add(unit);h=hash_key(day,market,unit);mod=2 if b['generation']==0 else 32
        selected=all_sampled or h%mod==0;delay=AGES[(h//mod)%len(AGES)]
        row=dict(day=day,market=int(market),side=b['side'],rung=b['rung'],generation=b['generation'],
                 birth_t=b['t'],requested_delay=delay,status='not_sampled')
        if selected:
            values,x=paired(W,b,delay,continuation);row.update(values)
            if x is not None:row.update(dict(zip(FEATURES,map(float,x))))
        records.append(row)
    return records,dict(statuses)


def continuation(out,kind):
    if kind=='flat':return Forecast()
    if kind!='learned':raise ValueError('unknown label continuation')
    return load_models(Path(out)/'bootstrap')


def row_stage(kind,day):return f'rows_{kind}_{day}'

def rows_day(day,out,kind,max_windows=None):
    if day not in (BOOT_DAYS if kind=='flat' else LATER):raise ValueError('undeclared label date')
    if max_windows is not None and max_windows<1:raise ValueError('bad window cap')
    p=stage(out,row_stage(kind,day));initial=stamp();policy=continuation(out,kind);data=K.load_day(day)
    records=[];markets=[];status=collections.Counter();start=time.monotonic()
    for W in E.PAIRS.window_pairs(data):
        rows,st=build_window(W,day,W[0]['m'],policy);records.extend(rows);status.update(st);markets.append(W[0]['m'])
        if len(markets)%4==0 or max_windows is not None:
            print(kind,day,len(markets),'windows',len(records),'generations',round(time.monotonic()-start,1),'s',round(E.rss_mib()),'MiB',flush=True)
        if E.rss_mib()>2048:raise RuntimeError('RSS cap exceeded')
        if max_windows is not None and len(markets)>=max_windows:break
    frame=pd.DataFrame(records)
    if frame.empty:raise ValueError('empty generation population')
    if stamp()!=initial:raise ValueError('source changed')
    frame.to_parquet(p/'rows.parquet',index=False);expected=set(data['W0']);seen=set(markets)
    if len(markets)!=len(seen) or not seen<=expected:raise ValueError('window identities differ')
    E.write_json(p/'receipt.json',dict(**initial,asof=now(),day=day,kind=kind,continuation=policy.identity,
        markets=markets,windows=len(markets),excluded_markets=sorted(expected-seen),complete=seen==expected and max_windows is None,
        partial=max_windows is not None,source_status=data['stat'],decision_status=dict(status),action_status=frame.status.value_counts().to_dict(),
        rows_sha256=digest(p/'rows.parquet'),wall_seconds=time.monotonic()-start))


def load_rows(out,kind,day,partial=False):
    p=Path(out)/row_stage(kind,day);meta=E.read_json(p/'receipt.json');require(meta)
    if meta['day']!=day or meta['kind']!=kind or digest(p/'rows.parquet')!=meta['rows_sha256']:raise ValueError('row identity mismatch')
    if not partial and (meta['partial'] or not meta['complete']):raise ValueError('partial/incomplete rows')
    expected='FLAT' if kind=='flat' else digest(Path(out)/'bootstrap/receipt.json')
    if meta['continuation']!=expected:raise ValueError('label continuation mismatch')
    r=pd.read_parquet(p/'rows.parquet')
    if not (r.day==day).all() or r.duplicated(['day','market','side','rung','generation']).any():raise ValueError('action population mismatch')
    return r


def model_health(model,x,y,structural_zero=False):
    if structural_zero:
        if not np.all(y==0):raise ValueError('nonzero target declared structural zero')
        return dict(structural_zero=True)
    pred=np.asarray(model.predict(x));nodes=[z['num_leaves'] for z in model.dump_model()['tree_info']]
    split=any(n>1 for n in nodes);varies=float(np.ptp(pred))>1e-8
    if not split or not varies:raise ValueError('constant learned head: cannot claim timing skill')
    return dict(trees=len(nodes),max_leaves=max(nodes),prediction_range=float(np.ptp(pred)),has_splits=split)


def fit(out,test=None,partial=False):
    import lightgbm as lgb
    boot=test is None;days=BOOT_DAYS if boot else FOLDS.get(test)
    if not days or (not boot and not all(d<test for d in days)):raise ValueError('undeclared/nonchronological fold')
    p=stage(out,'bootstrap' if boot else 'fit_'+test);initial=stamp();kind='flat' if boot else 'learned'
    rows=pd.concat([load_rows(out,kind,d,partial) for d in days],ignore_index=True);rows=rows[rows.status=='eligible']
    heads=('post',) if boot else ('post','wait');meta={};params=dict(E.PD.LGB_PARAMS,objective='regression')
    for s in (0,1):
        r=rows[rows.side==s]
        if len(r)<MIN_N or r.day.nunique()<2:raise ValueError(f'insufficient side{s} support: {len(r)}')
        if len(r)>E.PD.SUB_N:r=r.sample(E.PD.SUB_N,random_state=SEED)
        x=r[FEATURES].to_numpy(np.float32)
        for head in heads:
            y=r['q'+head].to_numpy(float)
            if not np.isfinite(y).all():raise ValueError('nonfinite economic target')
            if np.count_nonzero(abs(y)>1e-10)<MIN_NONZERO:raise ValueError(f'insufficient nonzero {s}/{head} targets')
            model=lgb.train(params,lgb.Dataset(x,y,feature_name=FEATURES),E.PD.N_ROUNDS)
            health=model_health(model,x,y);name=f'{s}_{head}';model.save_model(str(p/(name+'.txt')))
            meta[name]=dict(n=len(r),nonzero=int((abs(y)>1e-10).sum()),positive=int((y>0).sum()),negative=int((y<0).sum()),
                            health=health,model_sha256=digest(p/(name+'.txt')))
    if stamp()!=initial:raise ValueError('source changed while fitting')
    E.write_json(p/'receipt.json',dict(**initial,asof=now(),kind='bootstrap' if boot else 'improved',test=test,train=list(days),
        continuation='FLAT' if boot else digest(Path(out)/'bootstrap/receipt.json'),partial=partial,heads=meta,
        params=params,rounds=E.PD.N_ROUNDS,threshold='post>0 and post>=wait',
        row_receipts={d:digest(Path(out)/row_stage(kind,d)/'receipt.json') for d in days}))


def load_models(path,allow_partial=False):
    import lightgbm as lgb
    path=Path(path);meta=E.read_json(path/'receipt.json');require(meta)
    if meta['partial'] and not allow_partial:raise ValueError('partial model not allowed')
    boot=meta['kind']=='bootstrap';days=BOOT_DAYS if boot else FOLDS[meta['test']]
    if meta['train']!=list(days):raise ValueError('model training identities differ')
    models={0:{},1:{}}
    for s in (0,1):
        for head in (('post',) if boot else ('post','wait')):
            name=f'{s}_{head}';p=path/(name+'.txt')
            if digest(p)!=meta['heads'][name]['model_sha256']:raise ValueError('model digest mismatch')
            model=lgb.Booster(model_file=str(p))
            if model.feature_name()!=FEATURES:raise ValueError('feature order differs')
            models[s][head]=model
    return Forecast(models,digest(path/'receipt.json'))


def policy(W,predictor,arm='VALUE',capture=None):
    if arm not in ARMS:raise ValueError('unknown replay arm')
    tracker=Tracker()
    def choose(v):
        tracker.see(v)
        if arm=='FLAT':return False
        if arm in ('ALWAYS','TIMER'):return True
        if arm=='SCHEDULED':return bool(W[v['side']]['vkl'][v['k']]>K.GUARD_PULL)
        return learned(W,predictor,tracker,v,capture)
    return choose


def predict_rows(day,out):
    rows=load_rows(out,'learned',day);rows=rows[rows.status=='eligible'].copy();model=load_models(Path(out)/('fit_'+day))
    rows['post_hat']=np.nan;rows['wait_hat']=np.nan
    for side in (0,1):
        m=rows.side==side;x=rows.loc[m,FEATURES].to_numpy(np.float32)
        for head in ('post','wait'):rows.loc[m,head+'_hat']=model.models[side][head].predict(x)
    rows['post_action']=(rows.post_hat>0)&(rows.post_hat>=rows.wait_hat)
    return rows


def eval_day(day,out,max_windows=None):
    if day not in TESTS:raise ValueError('undeclared test')
    p=stage(out,'eval_'+day);initial=stamp();model=load_models(Path(out)/('fit_'+day));admission=load_models(Path(out)/'bootstrap')
    data=K.load_day(day);rows=[];samples=[];counts=collections.Counter();markets=[]
    for W in E.PAIRS.window_pairs(data):
        markets.append(W[0]['m']);model.clear();admission.clear()
        for shift in CFGS:
            for arm in ARMS:
                predictor=admission if arm=='ADMISSION' else model;calls=[0]
                def capture(v,x,values,action,status):
                    counts[f'{arm}:{int(shift)}:{v["side"]}:{v["rung"]}:{status}:{"post" if action else "wait"}']+=1
                    calls[0]+=1
                    if x is not None and values is not None and calls[0]%500==1:
                        samples.append(dict(day=day,market=W[0]['m'],side=v['side'],arm=arm,shift=int(shift),
                            post_hat=values[0],wait_hat=values[1],**dict(zip(FEATURES,map(float,x)))))
                r=run(W,policy(W,predictor,arm,capture),shift,entry_mode='legacy' if arm=='TIMER' else 'value')
                metrics=E.metrics(W,r)
                rows.append(dict(day=day,market=W[0]['m'],arm=arm,shift=int(shift),**metrics,
                    posts_bid=sum(len(r['order_states'][0,d]) for d in range(NR)),posts_ask=sum(len(r['order_states'][1,d]) for d in range(NR))))
                counts.update({f'{arm}:{int(shift)}:engine:{k}':v for k,v in r['policy_status'].items()})
        if len(markets)%4==0:print('replay',day,len(markets),'windows',round(E.rss_mib()),'MiB',flush=True)
        if E.rss_mib()>2048:raise RuntimeError('RSS cap exceeded')
        if max_windows is not None and len(markets)>=max_windows:break
    frame=pd.DataFrame(rows);frame.to_parquet(p/'windows.parquet',index=False);sample=pd.DataFrame(samples)
    if sample.empty:raise ValueError('no inference samples')
    sample.to_parquet(p/'inference.parquet',index=False)
    for (arm,s),g in sample.groupby(['arm','side']):
        forecast=admission if arm=='ADMISSION' else model;x=g[FEATURES].to_numpy(np.float32)
        for h in ('post','wait'):
            pred=forecast.models[s][h].predict(x) if h in forecast.models[s] else np.zeros(len(g))
            if not np.allclose(pred,g[h+'_hat'],rtol=0,atol=1e-9):raise ValueError('inference reconstruction mismatch')
    predict_rows(day,out).to_parquet(p/'local_predictions.parquet',index=False)
    if stamp()!=initial:raise ValueError('source changed during replay')
    expected=set(data['W0']);seen=set(markets)
    if len(markets)!=len(seen) or not seen<=expected:raise ValueError('window identity mismatch')
    E.write_json(p/'receipt.json',dict(**initial,asof=now(),day=day,markets=markets,excluded_markets=sorted(expected-seen),
        complete=seen==expected and max_windows is None,partial=max_windows is not None,source_status=data['stat'],counts=dict(counts),
        files={n:digest(p/n) for n in ('windows.parquet','inference.parquet','local_predictions.parquet')},
        model_identity=model.identity,admission_identity=admission.identity))


def timing_null(rows,draws=500):
    if draws<200:raise ValueError('too few null draws')
    r=rows.copy();r['hour']=(r.probe_t//3600000).astype('int64');selected=r.post_action.to_numpy(bool);n=int(selected.sum())
    if n<30 or len(r)-n<30:return dict(passes=False,status='insufficient_action_variation',n=len(r),posts=n)
    observed=float(r.loc[selected,'delta'].mean());admission=float(r.loc[selected,'qpost'].mean())
    groups=list(r.groupby(['day','side','hour','rung']));rng=np.random.default_rng(SEED);null=[]
    for _ in range(draws):
        total=0.
        for _,g in groups:
            k=int(g.post_action.sum())
            if k:total+=float(rng.choice(g.delta.to_numpy(float),k,replace=False).sum())
        null.append(total/n)
    p95=float(np.quantile(null,.95))
    return dict(passes=bool(observed>p95+1e-12 and observed>0 and admission>0),status='matched_local_decision_value',n=len(r),posts=n,
                observed_delta=observed,observed_absolute_post=admission,null_p95=p95,draws=draws)


def read_eval(day,out):
    p=Path(out)/('eval_'+day);m=E.read_json(p/'receipt.json');require(m)
    if m['day']!=day or not m['complete'] or m['partial']:raise ValueError('incomplete test day')
    if m['model_identity']!=digest(Path(out)/('fit_'+day)/'receipt.json') or m['admission_identity']!=digest(Path(out)/'bootstrap/receipt.json'):
        raise ValueError('replay model identity mismatch')
    for name,sha in m['files'].items():
        if digest(p/name)!=sha:raise ValueError('modified replay')
    r=pd.read_parquet(p/'windows.parquet')
    if not (r.day==day).all() or r.duplicated(['day','market','arm','shift']).any():raise ValueError('wrong/duplicate replay population')
    groups=list(r.groupby(['arm','shift']))
    if len(groups)!=len(ARMS)*len(CFGS) or set((a,int(s)) for (a,s),_ in groups)!={(a,int(s)) for a in ARMS for s in CFGS}:
        raise ValueError('missing replay arms')
    if any(set(g.market)!=set(m['markets']) for _,g in groups):raise ValueError('mismatched market identities')
    return r,pd.read_parquet(p/'local_predictions.parquet'),m


def screen(out):
    parts=[];local=[];statuses={}
    for day in TESTS:
        r,l,m=read_eval(day,out);parts.append(r);local.append(l);statuses[day]={'source':m['source_status'],'counts':m['counts'],'excluded_markets':m['excluded_markets']}
    full=pd.concat(parts,ignore_index=True);pred=pd.concat(local,ignore_index=True)
    cols=['pnl','pnl_without_young','inventory_square_seconds','shares','places','pulls','posts_bid','posts_ask','any_quote_steps','both_quote_steps','grid_steps']
    daily=full.groupby(['day','arm','shift'])[cols].sum();comp=[];rng=np.random.default_rng(SEED)
    for baseline in BASELINES:
        for shift in CFGS:
            a=daily.xs(('VALUE',shift),level=('arm','shift'));b=daily.xs((baseline,shift),level=('arm','shift'))
            for metric in ('pnl','pnl_without_young'):
                d=(a[metric]-b[metric]).to_numpy();boots=rng.choice(d,(1000,len(d)),replace=True).mean(1)
                comp.append(dict(baseline=baseline,shift=shift,metric=metric,n_days=len(d),mean_delta=float(d.mean()),positive_days=int((d>0).sum()),
                    passes=bool(d.mean()>0 and (d>0).sum()>=4),descriptive_day_interval=np.quantile(boots,[.025,.975]).tolist()))
    main=daily.xs(('VALUE',50),level=('arm','shift'))
    positive=bool(main.pnl.sum()>0 and main.pnl_without_young.sum()>0)
    risk=all(main.inventory_square_seconds.sum()<=daily.xs((b,50),level=('arm','shift')).inventory_square_seconds.sum() for b in BASELINES)
    timing={str(s):timing_null(g) for s,g in pred.groupby('side')}
    basic=bool(positive and risk and all(c['passes'] for c in comp) and set(timing)=={'0','1'} and all(v['passes'] for v in timing.values()))
    return dict(basic_pass=basic,absolute_positive_m50=positive,inventory_risk_not_increased=risk,comparisons=comp,timing=timing,statuses=statuses),daily


def quota_policy(quotas,seed,rate):
    if len(quotas)!=2 or any(q<0 or int(q)!=q for q in quotas):raise ValueError('bad placement quotas')
    rng=np.random.default_rng(seed);used=[0,0]
    def choose(v):
        s=v['side']
        post=used[s]<quotas[s] and rng.random()<rate
        if post:used[s]+=1
        return bool(post)
    return choose,used



def validate_null_draws(draws,candidate,n_draws=NULL_DRAWS):
    if n_draws<200:raise ValueError('insufficient portfolio null draws')
    keys=['day','market']
    if candidate.duplicated(keys).any() or draws.duplicated(keys+['draw']).any():raise ValueError('duplicate null/candidate population')
    expected=set(map(tuple,candidate[keys].to_numpy()));actual=set(map(tuple,draws[keys].to_numpy()))
    if actual!=expected:raise ValueError('null market identities differ')
    for _,g in draws.groupby(keys):
        if len(g)!=n_draws or set(g.draw)!=set(range(n_draws)):raise ValueError('incomplete matched null draws')
    joined=draws.merge(candidate[keys+['posts_bid','posts_ask']],on=keys,validate='many_to_one',suffixes=('_null','_candidate'))
    for side in ('bid','ask'):
        if not (joined['posts_'+side+'_null']==joined['posts_'+side+'_candidate']).all():raise ValueError('null action counts not matched')
    metrics=['pnl','pnl_without_young']
    if not np.isfinite(draws[metrics].to_numpy()).all() or not np.isfinite(candidate[metrics].to_numpy()).all():raise ValueError('nonfinite null economics')
    conditions=[]
    for metric in metrics:
        observed=float(candidate[metric].sum());null=draws.groupby('draw')[metric].sum().to_numpy();p95=float(np.quantile(null,.95))
        conditions.append(dict(metric=metric,candidate=observed,null_p95=p95,passes=bool(observed>p95)))
    return conditions


def portfolio_null(out):
    p=stage(out,'portfolio_null');basic,_=screen(out)
    if not basic['basic_pass']:
        E.write_json(p/'receipt.json',dict(**stamp(),asof=now(),status='skipped_basic_screen_failed',passes=False,draws=0));return
    records=[];counts=collections.Counter()
    for day in TESTS:
        f,_,m=read_eval(day,out);cand=f[(f.arm=='VALUE')&(f['shift']==50)].set_index('market');data=K.load_day(day);seen=[]
        for W in E.PAIRS.window_pairs(data):
            market=W[0]['m'];seen.append(market);row=cand.loc[market];quotas=[int(row.posts_bid),int(row.posts_ask)];accepted=0
            for attempt in range(NULL_MAX_ATTEMPTS):
                seed=hash_key(day,market,('portfolio_null',attempt));rate=(.25,.5,.75,.95,1.)[attempt%5]
                choose,used=quota_policy(quotas,seed,rate);r=run(W,choose,50.)
                actual=[sum(len(r['order_states'][s,d]) for d in range(NR)) for s in (0,1)]
                if actual!=used:raise ValueError('quota decisions differ from placements')
                if actual!=quotas:counts['unmatched_draws']+=1;continue
                met=E.metrics(W,r);records.append(dict(day=day,market=market,draw=accepted,posts_bid=actual[0],posts_ask=actual[1],
                    pnl=met['pnl'],pnl_without_young=met['pnl_without_young']));accepted+=1
                if accepted==NULL_DRAWS:break
            if accepted<NULL_DRAWS:
                E.write_json(p/'receipt.json',dict(**stamp(),asof=now(),status='insufficient_matched_draws',passes=False,day=day,market=market,
                    matched=accepted,required=NULL_DRAWS,attempts=NULL_MAX_ATTEMPTS,counts=dict(counts)));return
            if len(seen)%4==0:print('matched portfolio null',day,len(seen),'windows',flush=True)
        if set(seen)!=set(m['markets']):raise ValueError('null population mismatch')
    f=pd.DataFrame(records);f.to_parquet(p/'draws.parquet',index=False)
    candidate=pd.concat([read_eval(d,out)[0] for d in TESTS],ignore_index=True)
    candidate=candidate[(candidate.arm=='VALUE')&(candidate['shift']==50)]
    conditions=validate_null_draws(f,candidate)
    E.write_json(p/'receipt.json',dict(**stamp(),asof=now(),status='complete',passes=all(c['passes'] for c in conditions),
        draws=NULL_DRAWS,counts=dict(counts),conditions=conditions,draws_sha256=digest(p/'draws.parquet')))


def report(out):
    p=stage(out,'report');result,daily=screen(out);npath=Path(out)/'portfolio_null/receipt.json';null=E.read_json(npath);require(null)
    if null['status']=='complete':
        if null['draws']!=NULL_DRAWS or digest(npath.parent/'draws.parquet')!=null['draws_sha256']:raise ValueError('invalid portfolio null')
        candidate=pd.concat([read_eval(d,out)[0] for d in TESTS],ignore_index=True)
        candidate=candidate[(candidate.arm=='VALUE')&(candidate['shift']==50)]
        conditions=validate_null_draws(pd.read_parquet(npath.parent/'draws.parquet'),candidate)
        if conditions!=null['conditions'] or all(c['passes'] for c in conditions)!=null['passes']:raise ValueError('null verdict differs from artifact')
    daily.reset_index().to_parquet(p/'daily.parquet',index=False)
    E.write_json(p/'result.json',dict(**stamp(),asof=now(),**result,portfolio_null=null,
        development_accept=bool(result['basic_pass'] and null['status']=='complete' and null['passes']),
        n_complete_days=len(TESTS),dates=list(TESTS),forward_validation=False,
        limits=['finite policy improvement, no optimality claim','neutral keeper; learned keeper integration not tested',
                'consumed dates; intervals descriptive','gross simulated cash plus inventory; queue/fill assumptions unvalidated']))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['selftest','rows','bootstrap','fit','eval','null','report'])
    p.add_argument('day',nargs='?');p.add_argument('--kind',choices=['flat','learned']);p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--max-windows',type=int);a=p.parse_args()
    if a.command=='selftest':
        import unittest
        import test_wait_post_v2
        r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_wait_post_v2))
        if not r.wasSuccessful():raise SystemExit(1)
    elif a.command=='rows':rows_day(a.day,a.out,a.kind,a.max_windows)
    elif a.command=='bootstrap':fit(a.out)
    elif a.command=='fit':fit(a.out,a.day)
    elif a.command=='eval':eval_day(a.day,a.out,a.max_windows)
    elif a.command=='null':portfolio_null(a.out)
    else:report(a.out)


if __name__=='__main__':main()
