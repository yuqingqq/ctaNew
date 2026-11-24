"""Frozen descriptive price/best-price decomposition, no new behavioural test."""
import csv,gzip,hashlib,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow as pa
pa.set_cpu_count(1);pa.set_io_thread_count(1)
H=Path(__file__).resolve().parent;R=H.parents[2]
P=R/'predictfun/reviews/maker_trace_20261122/v2/chains/take_logic_v2/panel.parquet'
N=R/'predictfun/reviews/maker_timing_estimates_20261124/outputs/post_estimates.csv.gz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
def classify(lo,hi):
 return 'more_aggressive_relative_touch' if lo>0 else ('less_aggressive_relative_touch' if hi<0 else ('unchanged_relative_touch' if lo==hi==0 else 'timing_ambiguous'))
def window(lo,hi,ws):
 a=(lo-ws)//1000;b=(hi-ws)//1000
 return np.arange(a,b+1,dtype=int) if 0<=a<=b<300 else None
def offsets(old,new,side,b0,b1):
 s=1 if side=='bid' else -1;x=np.asarray(b0)*100;y=np.asarray(b1)*100
 if not (np.isfinite(x).all() and np.isfinite(y).all() and (x>0).all() and (x<100).all() and (y>0).all() and (y<100).all() and np.allclose(x,np.rint(x),atol=1e-8,rtol=0) and np.allclose(y,np.rint(y),atol=1e-8,rtol=0)):raise ValueError('off_grid_book')
 x=np.rint(x).astype(int);y=np.rint(y).astype(int)
 a=s*(old-x);b=s*(new-y);q=s*(new-old);lo=int(b.min()-a.max());hi=int(b.max()-a.min());m=s*(y[:,None]-x[None,:]);d=b[:,None]-a[None,:]
 assert (m+d==q).all()
 return dict(quote_change_ticks=q,best_change_min_ticks=int(m.min()),best_change_max_ticks=int(m.max()),offset_change_min_ticks=lo,offset_change_max_ticks=hi,comparison=classify(lo,hi),old_states=len(x),new_states=len(y))
def build():
 for p,h in json.loads((H/'quote_bindings.json').read_text()).items():assert sha(R/p)==h,p
 c=rows(H/'outputs/chains.csv');nn={x['order_hash']:x for x in rows(N)};p=pd.read_parquet(P);pan={(int(d),int(m),int(r)):g.sort_values('k').set_index('k') for (d,m,r),g in p.groupby(['day','mid','rank'])};out=[]
 for r in c:
  for lag,z in ((0,'0'),(1000,'1')):
   a=dict(day=r['day'],mid=r['mid'],rank=r['rank'],order_hash=r['order_hash'],group=r['group'],successor_hash=r['successor_hash'],sequence=r['successor_relation'],side=r['side'],lag_ms=lag,status='ok',comparison='unknown',quote_change_ticks='',best_change_min_ticks='',best_change_max_ticks='',offset_change_min_ticks='',offset_change_max_ticks='',old_states=0,new_states=0)
   if not r['successor_hash']:a['status']='no_recorded_successor'
   elif r['successor_relation']=='refused_or_missing_window' or r['terminal']=='unresolved':a['status']='refused_window'
   else:
    n=nn[r['successor_hash']];assert (n['maker'].lower(),n['day'],n['mid'],n['side'])==(r['maker'],r['day'],r['mid'],r['side'])
    x=window(int(r['end_lo_ms']),int(r['end_hi_ms']),int(r['ws_ms']));y=window(int(r['successor_post_lo_ms']),int(r['successor_post_hi_ms']),int(r['ws_ms']))
    if x is None or y is None:a['status']='outside_grid'
    else:
     t=pan[(int(r['day']),int(r['mid']),int(r['rank']))];assert t.index.is_unique and all(k in t.index for k in np.r_[x,y]);xx=t.loc[x];yy=t.loc[y]
     assert (xx.t_ms.to_numpy()==int(r['ws_ms'])+1000*x).all() and (yy.t_ms.to_numpy()==int(r['ws_ms'])+1000*y).all()
     if not xx['kok'+z].all() or not yy['kok'+z].all():a['status']='book_unavailable'
     else:
      try:a.update(offsets(int(r['tick']),int(n['tick']),r['side'],xx[r['side']+z].to_numpy(),yy[r['side']+z].to_numpy()))
      except ValueError:a['status']='off_grid_book'
   out.append(a)
 assert len(out)==176
 path=H/'outputs/quote_relative.csv';assert not path.exists()
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(out[0]),lineterminator='\n');w.writeheader();w.writerows(out)
 groups=[]
 for lag in (0,1000):
  for g in ('all','exhausted','conditional_withdrawal','unresolved'):
   rr=[x for x in out if x['lag_ms']==lag and (g=='all' or x['group']==g)];groups.append(dict(lag_ms=lag,group=g,n=len(rr),status=dict(Counter(x['status'] for x in rr)),comparison=dict(Counter(x['comparison'] for x in rr))))
 (H/'quote_result.json').write_text(json.dumps(dict(rows=len(out),groups=groups,sha256=sha(path)),indent=2)+'\n');print(json.dumps(groups,indent=2))
def test():
 assert offsets(50,49,'bid',[.5],[.49])['comparison']=='unchanged_relative_touch'
 assert offsets(50,48,'bid',[.5],[.49])['comparison']=='less_aggressive_relative_touch'
 assert offsets(50,52,'ask',[.5],[.51])['comparison']=='less_aggressive_relative_touch'
 assert offsets(50,49,'bid',[.48,.50],[.48,.50])['comparison']=='timing_ambiguous'
 assert window(-1,3,0) is None and window(299000,300000,0) is None
 try:offsets(50,49,'bid',[.5001],[.49]);raise AssertionError('offgrid accepted')
 except ValueError:pass
 print('6 synthetic groups passed')
if __name__=='__main__':test() if '--test' in sys.argv else build()
