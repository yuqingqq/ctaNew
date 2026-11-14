"""Post hoc diagnosis of frozen wait/post v2; no training or candidate selection.

Saved artifacts: in-sample versus chronological prediction error, earlier-data
constant/rung/price baselines, economic calibration, target tails and state support.
Replay: reproduce VALUE/ALWAYS/TIMER M50 exactly; add one declared guard-exit
sensitivity with the SAME frozen entry models, no cooldown. Full five consumed
dates, first60s/window. Attribute every fill to fill-time spread plus subsequent
mark move, age/rung/side; distinguish any quote from touch presence. Inspect every
legal entry decision and de-duplicate generation-level participation/support.
The keeper sensitivity changes the continuation and is NOT a fitted candidate.
No new skill/acceptance claim: original 500-draw matched tests remain authoritative.
"""
import argparse
import collections
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pf_wait_post_v2 as M
E=M.E
BASE=Path('/home/yuqing/ctaNew/data/predict_fun/research/wait_post_v2_20261114_v1/full')
OUT=Path('/home/yuqing/ctaNew/data/predict_fun/research/wait_post_diagnosis_20261114_v1')
ARMS=('VALUE','ALWAYS','TIMER','VALUE_GUARD_EXIT')
AGES=(100.,500.,2000.,10000.)
STATE_AGES=(50.,250.,1000.,5000.,20000.)
STATE_FEATURES=('known_position','own_exposure','other_exposure','idle_age_ms','side_price_change_while_empty')


def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def metric(y,p,baseline):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if not len(y) or y.shape!=p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('invalid prediction population')
    err=p-y;den=float(np.mean((y-baseline)**2));scale=float(np.mean((y-y.mean())**2))
    rank=lambda z:pd.Series(z).rank().to_numpy()
    return dict(n=len(y),actual_mean=float(y.mean()),predicted_mean=float(p.mean()),bias=float(err.mean()),
        rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(abs(err))),
        r2_vs_train_mean=float(1-np.mean(err**2)/den) if den>0 else None,
        r2_vs_test_mean=float(1-np.mean(err**2)/scale) if scale>0 else None,
        rank_ic=float(np.corrcoef(rank(y),rank(p))[0,1]) if np.ptp(y)>0 and np.ptp(p)>0 else None)


def tail(y):
    y=np.asarray(y,float);sq=np.sort(y*y);n=max(1,int(math.ceil(len(y)*.01)))
    return dict(n=len(y),zero_fraction=float((abs(y)<=1e-10).mean()),mean=float(y.mean()),std=float(y.std()),
        abs_quantiles=dict(zip(('p50','p90','p95','p99','max'),map(float,np.quantile(abs(y),[.5,.9,.95,.99,1])))),
        top1pct_squared_target_fraction=float(sq[-n:].sum()/sq.sum()) if sq.sum()>0 else 0.)


def state_keys(frame):
    return list(zip(frame.rung.astype(int),np.floor(frame.known_position/60).astype(int),
        np.floor(frame.own_exposure/60).astype(int),np.floor(frame.other_exposure/60).astype(int),
        np.searchsorted(STATE_AGES,frame.idle_age_ms.to_numpy(),side='right')))


class Support:
    def __init__(self,frame):
        self.counts=collections.Counter(state_keys(frame));self.bounds={k:(float(frame[k].min()),float(frame[k].max())) for k in STATE_FEATURES}
        self.indices={k:M.FEATURES.index(k) for k in ('rung',*STATE_FEATURES)}
    def check(self,x):
        v={k:float(x[j]) for k,j in self.indices.items()}
        key=(int(v['rung']),math.floor(v['known_position']/60),math.floor(v['own_exposure']/60),math.floor(v['other_exposure']/60),int(np.searchsorted(STATE_AGES,v['idle_age_ms'],side='right')))
        count=self.counts[key];outside=[k for k,(lo,hi) in self.bounds.items() if v[k]<lo or v[k]>hi]
        return count,outside


def training(day):
    x=pd.concat([M.load_rows(BASE,'learned',d) for d in M.FOLDS[day]],ignore_index=True)
    return x[x.status=='eligible'].copy()


def summarize_selection(g,p,w,mask):
    return dict(n=len(g),selected=int(mask.sum()),predicted_post=float(p[mask].mean()) if mask.any() else None,
        actual_post=float(g.qpost.to_numpy()[mask].mean()) if mask.any() else None,
        predicted_delta=float((p-w)[mask].mean()) if mask.any() else None,
        actual_delta=float(g.delta.to_numpy()[mask].mean()) if mask.any() else None)


def baseline(train,test,target):
    # Prior dates only. Fixed10-tick price bins and minimum100 cell support.
    a=train.copy();b=test.copy();a['pricebin']=(a.price//10).astype(int);b['pricebin']=(b.price//10).astype(int)
    by_rung=a.groupby('rung')[target].mean();cell=a.groupby(['rung','pricebin'])[target].agg(['mean','size'])
    pred=[]
    for r,p in zip(b.rung,b.pricebin):
        key=(r,p);pred.append(float(cell.loc[key,'mean']) if key in cell.index and cell.loc[key,'size']>=100 else float(by_rung.get(r,a[target].mean())))
    return np.asarray(pred)


def saved(out):
    p=M.stage(out,'saved');fits=[];bins=[];states=[];inferences=[];label_status={}
    for day in M.TESTS:
        train=training(day);_,test,receipt=M.read_eval(day,BASE);forecast=M.load_models(BASE/('fit_'+day))
        label_status[day]=E.read_json(BASE/M.row_stage('learned',day)/'receipt.json')['action_status']
        for side in (0,1):
            tr=train[train.side==side];te=test[test.side==side];support=Support(tr)
            trp={h:forecast.models[side][h].predict(tr[M.FEATURES].to_numpy(np.float32)) for h in ('post','wait')}
            tep={h:te[h+'_hat'].to_numpy() for h in ('post','wait')}
            for h in ('post','wait'):
                check=forecast.models[side][h].predict(te[M.FEATURES].to_numpy(np.float32))
                if not np.allclose(check,tep[h],atol=1e-9,rtol=0):raise ValueError('test prediction reconstruction differs')
            heads={}
            for h in ('post','wait','delta'):
                name='q'+h if h!='delta' else h;y=tr[name].to_numpy();z=te[name].to_numpy()
                pp=trp[h] if h!='delta' else trp['post']-trp['wait'];qq=tep[h] if h!='delta' else tep['post']-tep['wait']
                heads[h]=dict(train=metric(y,pp,y.mean()),test=metric(z,qq,y.mean()),test_price_rung_baseline=metric(z,baseline(tr,te,name),y.mean()),train_target=tail(y),test_target=tail(z))
            importance={h:sorted(zip(M.FEATURES,map(float,forecast.models[side][h].feature_importance('gain'))),key=lambda x:-x[1])[:12] for h in ('post','wait')}
            fits.append(dict(day=day,side=side,train_dates=list(M.FOLDS[day]),n_train=len(tr),n_test=len(te),heads=heads,
                train_post_wait_correlation=float(tr[['qpost','qwait']].corr().iloc[0,1]),test_post_wait_correlation=float(te[['qpost','qwait']].corr().iloc[0,1]),
                train_selection=summarize_selection(tr,trp['post'],trp['wait'],(trp['post']>0)&(trp['post']>=trp['wait'])),
                test_selection=summarize_selection(te,tep['post'],tep['wait'],te.post_action.to_numpy(bool)),importance=importance))
            for split,g,pp in (('train',tr,trp['post']),('test',te,tep['post'])):
                bucket=np.searchsorted([-5.,-2.,0.,2.,5.,10.],pp,side='right')
                for b in range(7):
                    mask=bucket==b
                    if mask.any():bins.append(dict(day=day,side=side,split=split,bin=b,n=int(mask.sum()),predicted=float(pp[mask].mean()),actual=float(g.qpost.to_numpy()[mask].mean())))
            checks=[support.check(x) for x in te[M.FEATURES].to_numpy(np.float32)]
            states.append(dict(day=day,side=side,n=len(te),unseen=int(sum(c==0 for c,o in checks)),fewer_than5=int(sum(c<5 for c,o in checks)),outside_marginal=int(sum(bool(o) for c,o in checks))))
        inf=pd.read_parquet(BASE/('eval_'+day)/'inference.parquet');inf=inf[(inf.arm=='VALUE')&(inf['shift']==50)]
        for side,g in inf.groupby('side'):
            tr=train[train.side==side];support=Support(tr);checks=[support.check(x) for x in g[M.FEATURES].to_numpy(np.float32)]
            inferences.append(dict(day=day,side=int(side),n=len(g),sampling='first and every500th callback; descriptive only',unseen=sum(c==0 for c,o in checks),fewer_than5=sum(c<5 for c,o in checks)))
    write(p/'result.json',dict(asof=M.now(),fits=fits,score_bins=bins,probe_support=states,saved_inference_support=inferences,label_status=label_status,source_result_sha256=M.digest(BASE/'report/result.json'),forward_validation=False))
    print('Saved prediction audit complete',M.now(),flush=True)


def guard_run(W,forecast,capture=None):
    choose=M.policy(W,forecast,'VALUE',capture)
    def policy(v):
        if v['kind']=='rest':return bool(W[v['side']]['vkl'][v['k']]>M.K.GUARD_PULL)
        return choose(v)
    guards={s:(np.asarray(W[s]['vkl'])<=M.K.GUARD_PULL).tolist() for s in (0,1)}
    r=M.L.sim_joint(W,'M',50.,M.CAP,guards,size=M.SIZE,nr=M.NR,mode='hard',policy=policy,neutral_keeper=False)
    if any(r['stat']['viol'].values()):raise ValueError('guard diagnostic accounting violation')
    E.terminal(W,r);return r


def attribute(W,r):
    terminal=E.terminal(W,r);tf,mid=W[0]['tf_mff'];records=[]
    for (s,d),orders in r['order_states'].items():
        for o in orders:
            for t,q in o['fl']:
                j=int(np.searchsorted(tf,t,side='right'))-1
                if j<0 or not np.isfinite(mid[j]):raise ValueError('missing fill-time mark')
                sign=1 if s==0 else -1;age=t-o['ta'];spread=sign*q*(mid[j]-o['pt'])/100;move=sign*q*(terminal['mark']-mid[j])/100
                records.append(dict(side=s,rung=d,age_band=int(np.searchsorted(AGES,age,side='right')),quantity=q,fill_spread=spread,post_fill_move=move,pnl=spread+move,cancel_pending_shares=q if t>=o['tc'] else 0.))
    if not math.isclose(sum(x['pnl'] for x in records),terminal['pnl'],abs_tol=1e-7):raise ValueError('fill attribution does not reconcile')
    if not math.isclose(sum(x['quantity'] for x in records),sum(z[4] for z in r['fl']),abs_tol=1e-7):raise ValueError('fill quantities differ')
    grid=W[0]['tg'];live=np.zeros((2,len(grid)),bool);touch=live.copy();near=live.copy()
    for (s,d),orders in r['order_states'].items():
        best=np.asarray(W[s]['best']);valid=best>=0
        for o in orders:
            if o['ref']:continue
            active=(grid>=o['ta'])&(grid<min(o['tr'],o['tfill']));depth=(best-o['pt']) if s==0 else (o['pt']-best)
            live[s]|=active;touch[s]|=active&valid&(depth==0);near[s]|=active&valid&(depth>=0)&(depth<=1)
    met=E.metrics(W,r)
    if int(live.all(axis=0).sum())!=met['both_quote_steps']:raise ValueError('presence does not reconcile')
    presence=dict(grid_steps=len(grid),both_live=int(live.all(axis=0).sum()),both_touch=int(touch.all(axis=0).sum()),both_near=int(near.all(axis=0).sum()),
        bid_touch=int(touch[0].sum()),ask_touch=int(touch[1].sum()),bid_live=int(live[0].sum()),ask_live=int(live[1].sum()))
    return records,presence


class EntryAudit:
    def __init__(self,support):self.support=support;self.units={};self.counts=collections.Counter()
    def capture(self,v,x,values,action,status):
        if v['kind']!='entry':raise ValueError('entry audit received resting order')
        self.counts[f'{v["side"]}:{status}:{"post" if action else "wait"}']+=1
        if x is None:return
        key=(v['side'],v['rung'],v['generation']);c,o=self.support[v['side']].check(x)
        if key not in self.units:self.units[key]=dict(side=v['side'],rung=v['rung'],generation=v['generation'],first_t=v['t'],first_post=action,first_unseen=c==0,first_sparse=c<5,first_outside=bool(o),decisions=0,posted=False)
        unit=self.units[key];unit['decisions']+=1
        if action:
            if unit['posted']:raise ValueError('duplicate post for generation')
            unit.update(posted=True,post_delay=v['t']-unit['first_t'],post_unseen=c==0,post_sparse=c<5,post_outside=bool(o),post_hat=values[0],wait_hat=values[1])


def reconcile(actual,expected):
    for k in ('pnl','pnl_without_young','shares','places','inventory_square_seconds','both_quote_steps'):
        if not math.isclose(actual[k],float(expected[k]),rel_tol=1e-10,abs_tol=1e-7):raise ValueError('replay differs: '+k)


def replay(day,out):
    if day not in M.TESTS:raise ValueError('undeclared diagnostic day')
    p=M.stage(out,'replay_'+day);saved,_,receipt=M.read_eval(day,BASE);saved=saved[saved['shift']==50].set_index(['market','arm'])
    forecast=M.load_models(BASE/('fit_'+day));train=training(day);support={s:Support(train[train.side==s]) for s in (0,1)}
    data=M.K.load_day(day);windows=[];fills=[];entries=[];calls=collections.Counter();markets=[];start=time.monotonic()
    for W in E.PAIRS.window_pairs(data):
        market=W[0]['m'];markets.append(market);forecast.clear()
        for arm in ARMS:
            audit=EntryAudit(support) if arm in ('VALUE','VALUE_GUARD_EXIT') else None
            r=guard_run(W,forecast,audit.capture) if arm=='VALUE_GUARD_EXIT' else M.run(W,M.policy(W,forecast,arm,audit.capture if audit else None),entry_mode='legacy' if arm=='TIMER' else 'value')
            metrics=E.metrics(W,r)
            if arm!='VALUE_GUARD_EXIT':reconcile(metrics,saved.loc[(market,arm)])
            f,presence=attribute(W,r);windows.append(dict(day=day,market=market,arm=arm,**metrics,**presence))
            if f:
                agg=pd.DataFrame(f).groupby(['side','rung','age_band'])[['quantity','fill_spread','post_fill_move','pnl','cancel_pending_shares']].sum().reset_index()
                fills.extend(dict(day=day,market=market,arm=arm,**row) for row in agg.to_dict('records'))
            if audit:
                if sum(x['posted'] for x in audit.units.values())!=metrics['places']:raise ValueError('entry generation counts differ')
                entries.extend(dict(day=day,market=market,arm=arm,**v) for v in audit.units.values());calls.update({arm+':'+k:v for k,v in audit.counts.items()})
        if len(markets)%12==0:print(day,len(markets),'windows',round(time.monotonic()-start,1),'s',round(E.rss_mib()),'MiB',flush=True)
        if E.rss_mib()>2048:raise RuntimeError('diagnostic RSS cap')
    if set(markets)!=set(receipt['markets']) or len(markets)!=len(set(markets)):raise ValueError('incomplete diagnostic day')
    for n,f in (('windows',windows),('fills',fills),('entries',entries)):pd.DataFrame(f).to_parquet(p/(n+'.parquet'),index=False)
    write(p/'receipt.json',dict(asof=M.now(),day=day,windows=len(markets),markets=markets,source_status=data['stat'],counts=dict(calls),model_identity=forecast.identity,files={n:M.digest(p/n) for n in ('windows.parquet','fills.parquet','entries.parquet')},original_eval_receipt_sha256=M.digest(BASE/('eval_'+day)/'receipt.json'),baseline_metric_reconciliations=3*len(markets),forward_validation=False))


def finish(out):
    p=M.stage(out,'report');windows=[];fills=[];entries=[];receipts={}
    for day in M.TESTS:
        q=Path(out)/('replay_'+day);m=E.read_json(q/'receipt.json');receipts[day]=m
        for n,parts in (('windows.parquet',windows),('fills.parquet',fills),('entries.parquet',entries)):
            if M.digest(q/n)!=m['files'][n]:raise ValueError('diagnostic artifact changed')
            parts.append(pd.read_parquet(q/n))
    w=pd.concat(windows,ignore_index=True);f=pd.concat(fills,ignore_index=True);e=pd.concat(entries,ignore_index=True)
    cols=['pnl','pnl_without_young','shares','places','inventory_square_seconds','grid_steps','both_live','both_touch','both_near','bid_live','ask_live','bid_touch','ask_touch']
    daily=w.groupby(['day','arm'])[cols].sum().reset_index();tot=w.groupby('arm')[cols].sum().reset_index()
    for frame in (daily,tot):
        for k in ('both_live','both_touch','both_near','bid_touch','ask_touch'):frame[k+'_fraction']=frame[k]/frame.grid_steps
    fs=f.groupby(['arm','side','rung','age_band'])[['quantity','fill_spread','post_fill_move','pnl','cancel_pending_shares']].sum().reset_index()
    entry=[]
    for (day,arm,side),g in e.groupby(['day','arm','side']):
        posted=g[g.posted];entry.append(dict(day=day,arm=arm,side=int(side),generations=len(g),posted=len(posted),initial_posts=int(g.first_post.sum()),
            posted_after_wait=int((~posted.first_post).sum()),post_delay_p50=float(posted.post_delay.median()),post_delay_p90=float(posted.post_delay.quantile(.9)),
            first_unseen_fraction=float(g.first_unseen.mean()),first_sparse_fraction=float(g.first_sparse.mean()),first_outside_fraction=float(g.first_outside.mean()),
            post_unseen_fraction=float(posted.post_unseen.mean()),post_sparse_fraction=float(posted.post_sparse.mean()),post_outside_fraction=float(posted.post_outside.mean())))
    for arm,g in f.groupby('arm'):
        if not math.isclose(float(g.pnl.sum()),float(tot[tot.arm==arm].pnl.iloc[0]),abs_tol=1e-6):raise ValueError('pooled fill attribution differs')
    write(p/'result.json',dict(asof=M.now(),post_hoc_diagnosis=True,forward_validation=False,n_complete_days=len(M.TESTS),n_windows=int(w[['day','market']].drop_duplicates().shape[0]),daily=daily.to_dict('records'),totals=tot.to_dict('records'),fill_attribution=fs.to_dict('records'),entry_generations=entry,receipts=receipts,
        saved_audit=E.read_json(Path(out)/'saved/result.json'),saved_audit_sha256=M.digest(Path(out)/'saved/result.json'),age_edges_ms=AGES,state_age_edges_ms=STATE_AGES,limitations=['No new fitted candidate or threshold selection.','Guard-exit sensitivity uses an entry model trained under neutral KEEP, so it is not a correctly relabeled combined policy.','Historical guard uses LODO inputs.','State-bin support is a descriptive proxy, not a proven cause.','Fill marks/queue remain simulated, at M50 only.','All dates consumed; no forward validation or additional signal claim.']))
    print('Diagnostic report complete',M.now(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['selftest','saved','replay','finish']);p.add_argument('day',nargs='?');p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if a.command=='selftest':
        import unittest,test_wait_post_diagnose
        r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_wait_post_diagnose))
        if not r.wasSuccessful():raise SystemExit(1)
    elif a.command=='saved':saved(a.out)
    elif a.command=='replay':replay(a.day,a.out)
    else:finish(a.out)


if __name__=='__main__':main()
