"""Saved-state size and passive-before-take diagnostic, not private policy recovery."""
import collections,json,math,resource,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import experiment as E
H,R,T=E.H,E.R,E.T;C=H.parent/'maker_trace_20261122/v2/chains'
pa.set_cpu_count(1);pa.set_io_thread_count(1)
def label(v,fn):
 a=np.asarray(v,float)
 return '|'.join(sorted({fn(x) for x in a})) if len(a) and np.isfinite(a).all() else 'unknown'
def extrema(a):
 a=np.asarray(a,float)
 return (float(a.min()),float(a.max())) if len(a) and np.isfinite(a).all() else (None,None)
def sqrms(x):return np.sqrt(pd.Series(np.asarray(x,float)**2).rolling(10,min_periods=10).mean()).to_numpy()
def inventory_class(x):return 'adding' if x>1e-6 else 'reducing' if x< -1e-6 else 'flat'
def rows_and_orders():
 p=pd.read_parquet(C/'take_logic_v2/panel.parquet',filters=[('rank','=',3)]).sort_values(['day','mid','k']).reset_index(drop=True);p.day=p.day.astype(str)
 assert not p.duplicated(['day','mid','k']).any()
 for d in ['0','1']:
  p['rms'+d]=p.groupby(['day','mid'])['r_1s'+d].transform(sqrms)
 o=pd.read_csv(H.parent/'maker_action_population_20261124/outputs/orders.csv',float_precision='round_trip',dtype={'day':str});o=o[o['rank']==3]
 return p,o
def size_rows(p,o):
 out=[];groups={(d,int(m)):g.set_index('k') for (d,m),g in p.groupby(['day','mid'])}
 for r in o.itertuples(index=False):
  for d in ['0','1']:
   row=dict(hash=r.order_hash,day=r.day,mid=int(r.mid),size=float(r.original_shares),side=r.side,tick=r.tick,lag_ms=int(d)*1000,grade=r.post_grade,
    lo_ms=r.post_lo_ms,hi_ms=r.post_hi_ms,status='unknown',n=0,inventory='unknown',touch='unknown',clean=None,vol_min=None,vol_max=None,move_min=None,move_max=None,I_min=None,I_max=None,distance_min=None,distance_max=None)
   g=groups.get((r.day,int(r.mid)));lo,hi=r.post_lo_ms,r.post_hi_ms
   eligible=(str(r.complete_recorded_history)=='True' and r.budget_status in ['exhausted','positive_remainder_not_a_pull'] and r.post_grade in ['G2_bound_constrained','G4_scenario_only'])
   if eligible and g is not None and np.isfinite([lo,hi,r.ws_ms,r.we_ms]).all() and r.ws_ms<=lo<=hi<r.we_ms:
    a,b=int((lo-r.ws_ms)//1000),int((hi-r.ws_ms)//1000);ix=list(range(a,b+1))
    assert set(ix)<=set(g.index);gg=g.loc[ix];assert (gg.t_ms.to_numpy()==r.ws_ms+np.array(ix)*1000).all()
    sg=1 if r.side=='bid' else -1;I=gg['q'+d].to_numpy();dist=(gg['bid'+d].to_numpy()*100-r.tick) if sg==1 else (r.tick-gg['ask'+d].to_numpy()*100)
    if not gg['kok'+d].all():dist[:]=np.nan
    row.update(status='sampled_window',n=len(gg),inventory=label(sg*I,inventory_class),touch=label(dist,lambda x:'at_touch' if abs(x)<1e-7 else 'behind' if x>0 else 'inside'),clean=bool(gg.clean.all()),
     vol_min=extrema(gg['rms'+d])[0],vol_max=extrema(gg['rms'+d])[1],move_min=extrema(abs(gg['r_5s'+d]))[0],move_max=extrema(abs(gg['r_5s'+d]))[1],
     I_min=extrema(I)[0],I_max=extrema(I)[1],distance_min=extrema(dist)[0],distance_max=extrema(dist)[1])
   out.append(row)
 return pd.DataFrame(out)
def passive_context(p):
 actor=pd.read_parquet(E.X.DATA/'maker3_late_activity_20261124/actor_legs.parquet');actor.day=actor.day.astype(str)
 result=[];coverage=[]
 for (day,mid),g in p.groupby(['day','mid'],sort=True):
  g=g.copy();a=actor[(actor.day==day)&(actor.mid==mid)&(actor.role=='maker')].copy()
  f=pd.read_parquet(E.X.DATA/f'maker_trace_v2_20261123/life/{day}_{mid}_fills.parquet')
  assert not f.duplicated(['sid','mk_idx']).any()
  a=a.merge(f[['sid','mk_idx','hi_ms']],on=['sid','mk_idx'],how='left',validate='one_to_one')
  coverage.append(dict(day=day,mid=int(mid),passive_legs=len(a),missing_hi=int(a.hi_ms.isna().sum())))
  # Unmapped legs do not become zero: any missing hard-hi invalidates this market's context.
  for d in ['0','1']:
   tt=g.t_ms.to_numpy()-int(d)*1000;I=g['q'+d].to_numpy()
   for sec in [5,30]:
    sums={}
    for side in ['bid','ask']:
     af=a[a.y_side==side].sort_values('hi_ms');t=af.hi_ms.to_numpy();cum=np.r_[0,np.cumsum(af.q)]
     sums[side]=cum[np.searchsorted(t,tt,side='right')]-cum[np.searchsorted(t,tt-sec*1000,side='right')]
    red=np.where(I>1e-6,sums['ask'],np.where(I< -1e-6,sums['bid'],np.nan));add=np.where(I>1e-6,sums['bid'],np.where(I< -1e-6,sums['ask'],np.nan))
    if a.hi_ms.isna().any():red[:]=np.nan;add[:]=np.nan
    g[f'reduce_{sec}_{d}']=red;g[f'add_{sec}_{d}']=add
  result.append(g)
 return pd.concat(result,ignore_index=True),coverage
def matched(p):
 out=[]
 for d,clock in [('0','D0'),('1','D1000')]:
  z=p.copy();z['sign']=np.sign(z['q'+d]);z['ibin']=np.searchsorted([50,200,500,1000],abs(z['q'+d]),side='left');z['minute']=z.k//60
  z=z[z.clean&z['q'+d].abs().gt(1e-6)];keys=['day','mid','sign','ibin','minute']
  for sec in [5,30]:
   nm=f'reduce_{sec}_{d}';ev=z[z['O1_'+clock].astype(bool)&z[nm].notna()];ctrl=z[~z.O3.astype(bool)&z[nm].notna()]
   ctrl_groups={k:v for k,v in ctrl.groupby(keys)};ee=[];cc=[];n_control=0;matched_cells=0
   for key,v in ev.groupby(keys):
    c=ctrl_groups.get(key)
    if c is None or not len(c):continue
    ee.extend(v[nm]);cc.extend([float(c[nm].mean())]*len(v));n_control+=len(c);matched_cells+=1
   # Zero-reduction frequency has its own matched calculation, not thresholding a cell mean.
   ez=[];cz=[]
   for key,v in ev.groupby(keys):
    c=ctrl_groups.get(key)
    if c is None or not len(c):continue
    ez.extend((v[nm]<=1e-9).astype(float));cz.extend([float((c[nm]<=1e-9).mean())]*len(v))
   out.append(dict(clock=clock,prior_s=sec,eligible_event_seconds=len(ev),matched_event_seconds=len(ee),unmatched=len(ev)-len(ee),control_seconds=n_control,cells=matched_cells,
    event_mean_reduce=float(np.mean(ee)) if ee else None,matched_control_mean_reduce=float(np.mean(cc)) if cc else None,
    event_zero_reduce_fraction=float(np.mean(ez)) if ez else None,matched_control_zero_reduce_fraction=float(np.mean(cz)) if cz else None))
 return out
def summarize(s,p,ev,cov,mat):
 cells=[]
 for (d,size),g in s.groupby(['lag_ms','size']):
  good=g[g.status=='sampled_window'];cells.append(dict(lag_ms=int(d),size=size,orders=len(g),covered=len(good),inventory=good.inventory.value_counts().to_dict(),touch=good.touch.value_counts().to_dict(),
   rms_covered=int(good.vol_min.notna().sum()),rms_min_median=good.vol_min.median(),rms_max_median=good.vol_max.median(),abs5s_min_median=good.move_min.median(),abs5s_max_median=good.move_max.median()))
 day=[]
 for key,g in s[s.lag_ms.eq(0)].groupby(['day','size']):day.append(dict(day=key[0],size=key[1],orders=len(g),covered=int(g.status.eq('sampled_window').sum()),rms_min_median=g.vol_min.median(),rms_max_median=g.vol_max.median()))
 return T.clean(dict(signed_orders=s.hash.nunique(),cells=cells,by_day=day,neutral_seconds=len(p),markets=p[['day','mid']].drop_duplicates().shape[0],take_context=mat,life_coverage=cov,
  event_records=len(ev),event_statuses=ev.pop_status.value_counts().to_dict(),two_sided_quote_skew_identified=False,private_inventory_cap_identified=False,descriptive_only=True))
def run():
 resource.setrlimit(resource.RLIMIT_CPU,(45,46));start=time.process_time();E.bound();p,o=rows_and_orders();s=size_rows(p,o);p,cov=passive_context(p)
 ev=pd.read_parquet(C/'take_logic_v2/events.parquet',filters=[('rank','=',3)]);ev.day=ev.day.astype(str)
 join=p.copy().rename(columns={'k':'k_anchor'});fields=['day','mid','k_anchor','t_ms','q0','q1','qp','clean','r_5s0','rms0','bid0','ask0','reduce_5_0','reduce_30_0','add_5_0','add_30_0']
 ev=ev.merge(join[fields],on=['day','mid','k_anchor'],how='left',validate='many_to_one');mask=ev.pop_status.eq('on_panel') if 'on_panel' in set(ev.pop_status) else ev.t_ms.notna()
 assert (ev.loc[mask,'t_ms']<ev.loc[mask,'x_lo_hard_ms']).all()
 mat=matched(p);summary=summarize(s,p,ev,cov,mat)
 for fn,obj in [('sizes.csv',s),('take_context_panel.parquet',p),('take_events.csv',ev)]:
  path=E.OUT/fn;assert not path.exists()
  if path.suffix=='.parquet':obj.to_parquet(path,index=False)
  else:obj.to_csv(path,index=False)
 summary['cpu_s']=time.process_time()-start;summary['output_hashes']={fn:T.sha(E.OUT/fn) for fn in ['sizes.csv','take_context_panel.parquet','take_events.csv']};T.dump(H/'observations.json',summary)
 print(json.dumps(T.clean(summary),allow_nan=False));assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1500*1024
if __name__=='__main__':run()
