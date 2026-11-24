"""Independent raw-leg/native-record and dashboard reconciliation; no replay."""
import collections,csv,gzip,itertools,json,math,resource,time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
H=Path(__file__).resolve().parent;R=H.parents[2];D=R/'data/predict_fun/research/maker_trace_v2_20261123'
OUT=R/'data/predict_fun/research/maker3_late_activity_20261124';A='0x57FbC891E89184DA071AE9D97E745531cd7F718a'
resource.setrlimit(resource.RLIMIT_CPU,(20,21));started=time.process_time()
f=pd.read_parquet(OUT/'actor_legs.parquet');te=pd.read_parquet(OUT/'take_events.parquet')
summary=json.loads((H/'summary.json').read_text());checks={}
target=f[f.mid.eq(3789484)].copy()
L=pq.read_table(D/'legs_20261026.parquet',filters=[('mid','=',3789484)],columns=['mid','sid','mk_idx','mk_signer','mk_hash','y_side','y_price','shares','settle_ms','tk_signer','tk_hash','status'],use_threads=False).to_pandas()
raw={}
for r in L.itertuples():
    if r.mk_signer==r.tk_signer:continue
    for role,addr,hh in [('maker',r.mk_signer,r.mk_hash),('taker',r.tk_signer,r.tk_hash)]:
        if addr!=A:continue
        assert r.status=='ok'
        side=r.y_side if role=='maker' else ('ask' if r.y_side=='bid' else 'bid')
        raw[(str(r.sid),int(r.mk_idx),role)]=(r.shares,r.y_price,r.settle_ms,side,hh)
def reconcile(rows):
    keys=[(str(r.sid),int(r.mk_idx),r.role) for r in rows.itertuples()]
    assert len(keys)==len(set(keys)) and set(keys)==set(raw),'membership'
    for r in rows.itertuples():
        q,p,t,s,h=raw[(str(r.sid),int(r.mk_idx),r.role)]
        assert abs(q-r.q)<1e-8 and abs(p-r.p)<1e-8 and t==r.settle_ms and s==r.y_side and h==r.hash,'raw fields'
reconcile(target);checks['raw_target_legs']=len(target)
bad=target.copy();bad.loc[bad.index[0],'y_side']='ask' if bad.iloc[0].y_side=='bid' else 'bid'
refused=0
for z in [bad,pd.concat([target,target.iloc[:1]]),target.assign(q=target.q+1),target.assign(settle_ms=target.settle_ms+1000)]:
    try:reconcile(z)
    except AssertionError:refused+=1
assert refused==4;checks['corruptions_refused']=refused
# All aggregated native events reconcile to the already validated canonical legs.
tk=f[f.role.eq('taker')].groupby(['day','mid','sid'],sort=True).q.sum()
tr=te.set_index(['day','mid','sid']).q.sort_index()
assert tk.index.equals(tr.index) and np.allclose(tk,tr,atol=1e-8,rtol=0)
assert all((r.side=='bid')==((r.tk_quote=='Bid')==(r.tk_index_set==1)) for r in te.itertuples())
checks['native_take_events']=len(te)
# Minute totals independently computed from integer milliseconds; no audit imports.
act=pd.read_csv(H/'minute_activity.csv',float_precision='round_trip')
for r in act.itertuples():
    g=f if r.scope=='pooled' else f[f.mid.eq(3789484)] if r.scope=='market3789484' else f[f.day.eq(str(r.scope))]
    ms=g.settle_ms-g.ws*1000
    g=g[(ms>=60_000*(r.minute-1))&(ms<60_000*r.minute)&g.role.eq(r.role)]
    assert len(g)==r.legs and abs(g.q.sum()-r.shares)<1e-7
    assert len(set(zip(g.day,g.mid,g.sid,g.role)))==r.events
checks['minute_cells']=len(act)
# Public inventory proxy reconstructed independently by timestamp loops.
before={};after={};number={};positions=collections.defaultdict(float)
ordered=sorted(f[['day','mid','settle_ms','sid','role','y_side','q']].itertuples(index=False,name=None))
for k,gg in itertools.groupby(ordered,key=lambda r:r[:3]):
    group=list(gg);market=k[:2];before[k]=positions[market]
    positions[market]+=math.fsum(r[6] if r[5]=='bid' else -r[6] for r in group)
    after[k]=positions[market];number[k]=len({(r[3],r[4]) for r in group})
for r in te.itertuples():
    k=(r.day,r.mid,r.settle_ms)
    assert abs(before[k]-r.recorded_net_before)<1e-6 and abs(after[k]-r.recorded_net_after)<1e-6 and number[k]==r.batch_events
checks['proxy_tie_batches']=len(te)
chart=R/'predictfun/reviews/maker_dashboard_20261124/data.json.gz'
with gzip.open(chart,'rt') as h:data=json.load(h)
c=next(c for c in data['cases'] if c['id']==3789484)
shown=[r for r in c['makers'] if r['rank']==3]
z=target[(target.settle_ms>=c['ws'])&(target.settle_ms<=c['ws']+298_000)]
mapping={(str(r.sid),int(r.mk_idx),r.role):r for r in z.itertuples()}
assert len(mapping)==len(shown)
for r in shown:
    e=mapping[(r['sid'],r['mk_idx'],r['role'])]
    assert r['side']==(0 if e.y_side=='bid' else 1) and abs(r['price']-100*e.p)<1e-6
    assert abs(r['q']-e.q)<1e-5 and abs(r['t']-(e.settle_ms-c['ws'])/1000)<1e-6
checks['chart_maker3_legs']=len(shown)
checks['chart_maker3_taker_legs']=sum(r['role']=='taker' for r in shown)
checks['passed']=True;checks['cpu_s']=time.process_time()-started
(H/'checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
