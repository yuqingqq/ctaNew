import collections,csv,gzip,hashlib,itertools,json,math,resource,time
from pathlib import Path
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;R=H.parents[2];D=R/'data/predict_fun/research'
resource.setrlimit(resource.RLIMIT_CPU,(30,31));start=time.process_time();inputs={}
def bind(p):inputs[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):
    bind(p);return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n')
def union(iv):
    out=[]
    for a,b in sorted(iv):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out
def duration(iv):return sum(b-a for a,b in union(iv))
def intersect(a,b):return sum(max(0,min(y,v)-max(x,u)) for x,y in union(a) for u,v in union(b))
def independent(iv):
    events=collections.defaultdict(int)
    for a,b in iv:
        if b>a:events[a]+=1;events[b]-=1
    last=total=0.;n=0
    for t,delta in sorted(events.items()):
        if n:total+=t-last
        n+=delta;last=t
    return total
assert duration([(0,2),(1,3)])==independent([(0,2),(1,3)])==3
assert intersect([(0,2)],[(1,3)])==1 and duration([(2,2)])==0
# The known-bad sum-of-lifetimes differs from the union on this fixture.
assert duration([(0,2),(1,3)])!=4
old=read(R/'predictfun/reviews/maker_dashboard_20261124/data.json.gz')
own=[];loaded={}
for case in old['cases']:
    mid=case['id'];day=case['day'];original=case['ours']['CURRENT_ASYNC_FAST10']
    for arm,folder in [('ORIGINAL','entry_gate_pilot_20261124'),('FIXED_DEFAULT','entry_gate_pilot_20261124'),('OPENING_ONLY','opening_entry_pilot_20261124')]:
        p=D/folder/f'{mid}_{arm}.json.gz';rec=read(p);ap=bind(p.with_suffix('').with_suffix('.npz'))
        with np.load(ap) as z:arr={k:z[k] for k in ['pose','posc','mid']}
        loaded[(mid,arm)]=(rec,arr);by=collections.defaultdict(list)
        for f in rec['fills']:by[f[1]].append(f)
        stop={}
        for o in rec['orders']:
            e=min(rec['end'],o['remove']);q=0
            for f in by[o['gid']]:
                q+=f[3]
                if q>=o['size']-1e-8:e=min(e,f[0]);break
            stop[o['gid']]=e
        for name,lo in [('full',0),('last_minute',240000)]:
            hi=rec['end'];iv={s:[(max(lo,o['act']),min(hi,stop[o['gid']])) for o in rec['orders'] if not o['refused'] and o['side']==s and o['act']<hi and stop[o['gid']]>lo] for s in [0,1]}
            for s in [0,1]:assert abs(duration(iv[s])-independent(iv[s]))<1e-7
            ff=[f for f in rec['fills'] if lo<=f[0]<=hi];cc=[c for c in rec['commands'] if lo<=c[0]<hi]
            own.append(dict(mid=mid,arm=arm,window=name,duration_s=(hi-lo)/1000,bid_live_s=duration(iv[0])/1000,ask_live_s=duration(iv[1])/1000,
                either_s=duration(iv[0]+iv[1])/1000,both_s=intersect(iv[0],iv[1])/1000,
                posts=sum(c[7]=='POST' for c in cc),pulls=sum(c[7]=='PULL' for c in cc),filled_shares=sum(f[3] for f in ff),fills=len(ff),
                wait_reasons=dict(collections.Counter(c[8] for c in cc if c[7]=='WAIT')),
                first_fill_ms=min([f[0] for f in ff],default=None),last_fill_ms=max([f[0] for f in ff],default=None),
                last_submit_ms=max([o['submit'] for o in rec['orders']],default=None),end_inventory=rec['position'],peak_abs_inventory=float(np.max(np.abs(arr['pose']))),
                last_minute_known_start=float(arr['posc'][4800]),total_usd=rec['value']/100))
fpath=bind(D/'maker3_late_activity_20261124/actor_legs.parquet');f=pd.read_parquet(fpath)
tp=bind(D/'maker3_late_activity_20261124/take_events.parquet');takes=pd.read_parquet(tp)
op=bind(R/'predictfun/reviews/maker_action_population_20261124/outputs/orders.csv');signed=pd.read_csv(op,dtype={'order_hash':str},float_precision='round_trip');signed=signed[signed['rank']==3]
assert signed.order_hash.is_unique
maker=f[f.role.eq('maker') & f.t_rel_s.ge(0) & f.t_rel_s.lt(300)]
orders=maker.groupby(['day','mid','hash'],sort=False).agg(filled_shares=('q','sum'),fills=('q','size'),first_upper_s=('t_rel_s','min'),last_upper_s=('t_rel_s','max'),price=('p','first'),prices=('p','nunique'),side=('y_side','first')).reset_index()
assert orders.prices.eq(1).all()
orders=orders.merge(signed[['order_hash','original_shares','complete_recorded_history','budget_status','post_grade']],left_on='hash',right_on='order_hash',how='left',validate='many_to_one')
matched=orders.original_shares.notna();assert (orders.loc[matched,'filled_shares']<=orders.loc[matched,'original_shares']+1e-4).all()
def dist(v):
    return dict(n=len(v),min=float(v.min()),p25=float(v.quantile(.25)),median=float(v.median()),p75=float(v.quantile(.75)),p90=float(v.quantile(.9)),max=float(v.max())) if len(v) else dict(n=0)
sizes={}
for name,g in [('all_934',orders),('market3789484',orders[orders.mid.eq(3789484)])]:
    q=g[g.original_shares.notna()]
    sizes[name]=dict(filled_order_count=len(g),signed_size_known=len(q),signed_sizes=dist(q.original_shares),signed_histogram=q.original_shares.value_counts().sort_index().to_dict(),
                    observed_filled_per_order=dist(g.filled_shares),signed_selected_not_level_size=True)
orders[orders.mid.eq(3789484)].to_csv(H/'maker3_example_orders.csv',index=False)
# Exact capture-flow extremes and known takes, retained per market.
flow=f.assign(dq=np.where(f.y_side.eq('bid'),f.q,-f.q)).groupby(['day','mid','settle_ms'],sort=True).dq.sum().reset_index()
flow['net']=flow.groupby(['day','mid']).dq.cumsum()
extremes=flow.groupby(['day','mid']).net.agg(['min','max','last']).reset_index();extremes['max_abs']=np.maximum(abs(extremes['min']),abs(extremes['max']))
extremes.to_csv(H/'maker3_market_flow_extremes.csv',index=False)
summary=dict(own=own,sizes=sizes,flow=dict(max_abs_per_market=dist(extremes.max_abs),example=extremes[extremes.mid.eq(3789484)].to_dict('records')))
tt=takes[takes.mid.eq(3789484)].sort_values('t').copy();compare=[]
for t in tt.itertuples():
    row=dict(t=t.t,native_action=t.native_action,shares=t.q,price_native=t.price_native*100,recorded_net_before=t.recorded_net_before,recorded_net_after=t.recorded_net_after)
    for arm in ['ORIGINAL','FIXED_DEFAULT','OPENING_ONLY']:
        rec,arr=loaded[(3789484,arm)];k=min(int(t.t*1000//50),len(arr['posc'])-1)
        row[arm+'_known']=float(arr['posc'][k]);row[arm+'_venue']=float(arr['pose'][k]);row[arm+'_mid_c']=float(arr['mid'][k])
    compare.append(row)
summary['takes_vs_us']=compare
dump(H/'summary.json',summary);dump(H/'inputs.json',inputs)
dump(H/'resources_run.json',dict(cpu_s=time.process_time()-start,peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
print(json.dumps(summary,indent=2,default=str))
