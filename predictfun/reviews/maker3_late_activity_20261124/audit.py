"""Descriptive actor-side and minute census. contract.md is the specification."""
import collections, gzip, hashlib, json, math, resource, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

H=Path(__file__).resolve().parent; R=H.parents[2]
D=R/'data/predict_fun/research/maker_trace_v2_20261123'
ADDR='0x57FbC891E89184DA071AE9D97E745531cd7F718a'
DAYS=['20261024','20261025','20261026','20261027']; MID=3789484

def normalize(quote,index):
    assert quote in ('Bid','Ask') and index in (1,2), (quote,index)
    return 'bid' if (quote=='Bid')==(index==1) else 'ask'
def minute(t):
    return 'pre' if t<0 else 'post' if t>=300 else str(int(t//60)+1)
def position_kind(before,delta):
    if abs(before)<1e-9:return 'from_flat'
    if before*delta>=0:return 'adds_same_direction'
    return 'opposes_reduces_absolute' if abs(before+delta)<abs(before)-1e-9 else 'opposes_without_absolute_reduction'
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def js(p,x):p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def selftest():
    assert [normalize(q,i) for q,i in [('Bid',1),('Ask',1),('Bid',2),('Ask',2)]]==['bid','ask','ask','bid']
    assert [minute(t) for t in [-.001,0,59.999,60,179.999,180,299.999,300]]==['pre','1','1','2','3','4','5','post']
    assert position_kind(-20,30)=='opposes_reduces_absolute'
    assert position_kind(-20,50)=='opposes_without_absolute_reduction'
    assert position_kind(0,30)=='from_flat' and position_kind(20,30)=='adds_same_direction'
    print('normalization, half-open boundaries, crossing-zero distinctions PASS')
def stats(g):
    ev=g[['day','mid','sid','role']].drop_duplicates()
    late=g[g.t_rel_s.ge(180)]
    return dict(legs=len(g),events=len(ev),shares=g.q.sum(),markets=g.mid.nunique(),
                late_legs=len(late),late_events=len(late[['day','mid','sid','role']].drop_duplicates()),late_shares=late.q.sum(),
                late_share_fraction=late.q.sum()/g.q.sum() if g.q.sum()>0 else None)
def run():
    resource.setrlimit(resource.RLIMIT_CPU,(60,61)); start=time.process_time()
    inputs={}; bind=lambda p:inputs.update({str(p.relative_to(R)):sha(p)})
    fp=D/'econ/fills_valued.parquet'; bind(fp)
    cols=['day','mid','sid','mk_idx','role','addr','hash','y_side','s','q','p','self_trade','settle_ms','t_rel_s','status']
    f=pq.read_table(fp,filters=[('addr','=',ADDR),('day','in',DAYS)],columns=cols,use_threads=False).to_pandas()
    for c in ['day','sid','role','addr','hash','y_side','status']: f[c]=f[c].astype(str)
    assert set(f.day)<=set(DAYS),set(f.day)
    assert not f.duplicated(['day','mid','sid','mk_idx','role']).any()
    census=dict(rows=len(f),statuses=f.status.value_counts().to_dict(),self_legs=int(f.self_trade.sum()),nonpositive=int(f.q.le(0).sum()))
    f=f[f.status.eq('ok') & ~f.self_trade & f.q.gt(0)].copy()
    assert np.array_equal(f.s.to_numpy(),np.where(f.y_side.eq('bid'),1,-1))
    markets=[]; takes=[]
    for day in DAYS:
        mp=D/f'markets_{day}.parquet';tp=D/f'records_{day}.parquet';bind(mp);bind(tp)
        mm=pq.read_table(mp,columns=['mid','ws','status','n_poll_incomplete'],use_threads=False).to_pandas();mm['day']=day
        assert len(mm)==288 and mm.mid.is_unique
        markets.append(mm)
        t=pq.read_table(tp,filters=[('tk_signer','=',ADDR)],columns=['mid','sid','tk_signer','tk_hash','tk_quote','tk_index_set','tk_y_side','settle_ms','status'],use_threads=False).to_pandas()
        t['day']=day;t['sid']=t.sid.astype(str);assert not t.duplicated(['mid','sid']).any()
        takes.append(t)
    markets=pd.concat(markets,ignore_index=True);t=pd.concat(takes,ignore_index=True)
    f=f.merge(markets[['day','mid','ws']],on=['day','mid'],validate='many_to_one')
    assert np.allclose(f.t_rel_s,(f.settle_ms-f.ws*1000)/1000,atol=1e-9,rtol=0)
    # Record-level actor events; aggregate the taker's allocated counterparty legs once.
    f['dq']=f.s*f.q;f['qp']=f.q*f.p
    ev=f.groupby(['day','mid','sid','role'],observed=True,sort=False).agg(t=('t_rel_s','first'),settle_ms=('settle_ms','first'),
        side=('y_side','first'),nside=('y_side','nunique'),hash=('hash','first'),nhash=('hash','nunique'),q=('q','sum'),dq=('dq','sum'),qp=('qp','sum'),legs=('q','size')).reset_index()
    # Maker record may contain multiple orders/prices, but all legs are on one normalized side.
    assert ev.nside.eq(1).all()
    te=ev[ev.role.eq('taker')].merge(t,on=['day','mid','sid'],how='left',validate='one_to_one',suffixes=('','_native'),indicator=True)
    assert te._merge.eq('both').all() and te.nhash.eq(1).all()
    assert te.tk_signer.eq(ADDR).all() and te.hash.eq(te.tk_hash).all() and te.settle_ms.eq(te.settle_ms_native).all()
    assert all(normalize(r.tk_quote,int(r.tk_index_set))==r.side==r.tk_y_side for r in te.itertuples())
    te['native_action']=np.where(te.tk_quote.eq('Bid'),'BUY ','SELL ')+np.where(te.tk_index_set.eq(1),'YES','NO')
    te['price_yes']=te.qp/te.q
    te['price_native']=np.where(te.tk_index_set.eq(1),te.price_yes,1-te.price_yes)
    # Batch by public record upper time; no invented ordering inside equal-time sets.
    batch=ev.groupby(['day','mid','settle_ms'],sort=True).agg(batch_dq=('dq','sum'),batch_events=('sid','size'),t=('t','first')).reset_index()
    batch['recorded_net_after']=batch.groupby(['day','mid']).batch_dq.cumsum()
    batch['recorded_net_before']=batch.recorded_net_after-batch.batch_dq
    te=te.merge(batch.drop(columns='t'),on=['day','mid','settle_ms'],validate='many_to_one')
    te['own_delta_class']=[position_kind(a,b) for a,b in zip(te.recorded_net_before,te.dq)]
    te['isolated_record_time']=te.batch_events.eq(1)
    f['minute']=f.t_rel_s.map(minute)
    live=f[f.t_rel_s.ge(0)&f.t_rel_s.lt(300)]
    buckets=[];summ={}
    for name,g in [('pooled',live),('market3789484',live[live.mid.eq(MID)])]+[(d,live[live.day.eq(d)]) for d in DAYS]:
        summ[name]={role:stats(g if role=='all' else g[g.role.eq(role)]) for role in ['all','maker','taker']}
        for b in range(1,6):
            for role in ['maker','taker']:
                z=g[g.minute.eq(str(b))&g.role.eq(role)]
                buckets.append(dict(scope=name,minute=b,role=role,**stats(z)))
    pd.DataFrame(buckets).to_csv(H/'minute_activity.csv',index=False)
    ms=[]
    for (day,mid),g in live.groupby(['day','mid'],observed=True):ms.append(dict(day=day,mid=mid,**stats(g)))
    ms=pd.DataFrame(ms);ms.to_csv(H/'market_activity.csv',index=False)
    summ['market_counts']=dict(registry=len(markets),observed_live=len(ms),majority_last2=int(ms.late_share_fraction.gt(.5).sum()),median_late_share_fraction=ms.late_share_fraction.median(),registry_status=markets.status.value_counts().to_dict())
    summ['excluded']=census
    summ['outside']= {k:stats(f[f.minute.eq(k)]) for k in ['pre','post']}
    live_t=te[te.t.ge(0)&te.t.lt(300)]
    summ['taker_native']=live_t.groupby('native_action').agg(events=('sid','size'),shares=('q','sum')).to_dict('index')
    summ['taker_flow_proxy']=dict(classes=live_t.own_delta_class.value_counts().to_dict(),isolated=int(live_t.isolated_record_time.sum()),total=len(live_t))
    target=te[te.mid.eq(MID)].copy()
    target.to_csv(H/'market3789484_takes.csv',index=False)
    summ['market3789484']['taker_native']=target.groupby('native_action').agg(events=('sid','size'),shares=('q','sum')).to_dict('index')
    # Keep a bounded exact projection for independent checking and optional UI annotation.
    out=R/'data/predict_fun/research/maker3_late_activity_20261124';out.mkdir(exist_ok=True)
    f.to_parquet(out/'actor_legs.parquet',index=False);te.to_parquet(out/'take_events.parquet',index=False)
    f[f.mid.eq(MID)].to_csv(H/'market3789484_legs.csv',index=False)
    js(H/'summary.json',summ);js(H/'inputs.json',inputs)
    js(H/'run_resources.json',dict(cpu_s=time.process_time()-start,peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    print(json.dumps(clean(summ),indent=2))

if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:run()
