import collections,gzip,hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
H=Path(__file__).resolve().parent;R=H.parents[2];D=R/'data/predict_fun/research';V=D/'maker_trace_v2_20261123';MR=H.parent/'maker_trace_20261122';C=MR/'v2/chains';OUT=D/'maker3_spread_risk_20261124'
ADDR='0x57fbc891e89184da071ae9d97e745531cd7f718a';DAYS=['20261024','20261025','20261026','20261027'];CLOCKS=['pre','upper','lag1']
pa.set_cpu_count(1);pa.set_io_thread_count(1)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def population():
 p=pd.read_csv(MR/'v2/matched_gap_r1/per_market.csv',dtype={'day':str},float_precision='round_trip');p=p[(p.party=='rank3')&(p['view']=='zero')].sort_values(['day','m'])
 assert len(p)==19 and not p.m.duplicated().any();return p
def load_fills(mids):
 cols=['day','mid','sid','mk_idx','px_rank','role','addr','hash','own_index_set','y_side','s','q','p','self_trade','settle_ms','fee_fill','status']
 return pq.read_table(V/'econ/fills_valued.parquet',columns=cols,filters=[('day','in',DAYS),('mid','in',list(map(int,mids))),('addr','=',ADDR)],use_threads=False).to_pandas()
def band(x):return 'unknown' if x is None or not math.isfinite(x) else '<=1' if x<=1+1e-7 else '1-2' if x<=2+1e-7 else '2-4' if x<=4+1e-7 else '>4'
class Book:
 def __init__(self,g):
  g=g.sort_values(['upd_ms','recv_ns']);self.t=g.upd_ms.to_numpy(np.int64);self.recv=g.recv_ns.to_numpy()/1e6;self.bid=g.best_bid.to_numpy(float)*100;self.ask=g.best_ask.to_numpy(float)*100
  self.ok=np.isfinite(self.bid)&np.isfinite(self.ask)&(self.bid>=0)&(self.ask<=100)&(self.bid<=self.ask)&~g.status.fillna('').str.contains('bad_frame|mid_mismatch|out_of_order|off_grid').to_numpy()
 def at(self,t):
  if t is None or not math.isfinite(t):return dict(status='unknown_time',mid=None,spread=None,bid=None,ask=None,frame=None,age=None)
  i=int(np.searchsorted(self.t,t,'right'))-1
  if i<0:return dict(status='no_book',mid=None,spread=None,bid=None,ask=None,frame=None,age=None)
  age=float(t-self.t[i]);st='ok' if self.ok[i] and age<=1000 else 'stale' if age>1000 else 'invalid'
  return dict(status=st,mid=float((self.bid[i]+self.ask[i])/2) if st=='ok' else None,spread=float(self.ask[i]-self.bid[i]) if st=='ok' else None,
   bid=float(self.bid[i]) if st=='ok' else None,ask=float(self.ask[i]) if st=='ok' else None,frame=int(self.t[i]),receipt=float(self.recv[i]),age=age)
 def occupancy(self,start,end):
  edges=np.unique(np.r_[start,end,self.t[(self.t>start)&(self.t<end)],self.t[(self.t+1000>start)&(self.t+1000<end)]+1000])
  out=collections.Counter()
  for a,b in zip(edges[:-1],edges[1:]):out[band(self.at((a+b)/2)['spread'])]+=(b-a)/1000
  assert abs(sum(out.values())-(end-start)/1000)<1e-6;return dict(out)
def fifo(fs,mark):
 lots=collections.deque();cl=[];cash=pos=fees=0.
 for f in fs:
  q=f['q'];s=f['s'];p=f['price'];cash-=s*q*p;pos+=s*q;fees+=f['fee_c']
  while q>1e-9 and lots and lots[0]['f']['s']!=s:
   l=lots[0];a=l['f'];n=min(q,l['q']);gross=a['s']*n*(p-a['price']);fee=n*(a['fee_c']/a['q']+f['fee_c']/f['q'])
   row=dict(entry=a['id'],exit=f['id'],q=n,gross_c=gross,fee_c=fee,net_c=gross-fee,entry_t=a['t'],exit_t=f['t'],entry_role=a['role'],exit_role=f['role'],ordered=a['hi_ms']<f['lo_ms'] if f['lo_ms'] is not None else False,by_clock={})
   for clock in CLOCKS:
    am=a['book'][clock]['mid'];bm=f['book'][clock]['mid'];sa=a['book'][clock]['spread'];sb=f['book'][clock]['spread']
    valid=am is not None and bm is not None
    typ='unknown' if not valid else 'both_wide' if sa>4+1e-7 and sb>4+1e-7 else 'one_wide' if sa>4+1e-7 or sb>4+1e-7 else 'neither_wide'
    e=a['s']*n*(am-a['price']) if valid else None;x=a['s']*n*(p-bm) if valid else None;move=a['s']*n*(bm-am) if valid else None
    if valid:assert abs(e+x+move-gross)<1e-6
    row['by_clock'][clock]=dict(kind=typ,entry_edge_c=e,exit_edge_c=x,drift_c=move)
   cl.append(row);q-=n;l['q']-=n
   if l['q']<1e-9:lots.popleft()
  if q>1e-9:lots.append(dict(f=f,q=q))
 residual=sum(x['f']['s']*x['q']*(mark-x['f']['price']) for x in lots);unclosed_fee=sum(x['q']/x['f']['q']*x['f']['fee_c'] for x in lots)
 net=cash+pos*mark-fees;completed=sum(c['net_c'] for c in cl)
 assert abs(net-completed-residual+unclosed_fee)<1e-5
 return cl,list(lots),dict(cash_c=cash,position=pos,fee_c=fees,net_c=net,completed_c=completed,residual_c=residual,unclosed_fee_c=unclosed_fee,matched_shares=sum(c['q'] for c in cl))
def trend(grid):
 candidates=[]
 for i in range(0,len(grid)-30,5):
  a=grid[i:i+31];v=np.array([x[1] if x[1] is not None else np.nan for x in a]);valid=np.isfinite(v).all()
  if not valid:continue
  move=float(v[-1]-v[0]);length=float(abs(np.diff(v)).sum());eff=abs(move)/length if length else 0.
  candidates.append(dict(start_s=a[0][0],end_s=a[-1][0],move_c=move,efficiency=eff,qualifies=abs(move)>=10 and eff>=.6))
 return candidates
