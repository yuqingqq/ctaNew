import collections,gzip,json,resource,time
import numpy as np
import pandas as pd
import core as C
import build as B
def basis_states(fills,ts):
 f=sorted(fills,key=lambda x:(x['hi_ms'],x['sid'],x['mk_idx'],x['role']));i=0;lots=collections.deque();pos=0.;out=[]
 for t in ts:
  while i<len(f) and f[i]['hi_ms']<=t:
   a=f[i];q=a['q'];s=a['s'];pos+=s*q
   while q>1e-9 and lots and lots[0][0]!=s:
    sg,n,px,fee=lots[0];v=min(q,n);q-=v;n-=v
    if n<1e-9:lots.popleft()
    else:lots[0]=(sg,n,px,fee)
   if q>1e-9:lots.append((s,q,a['price'],a['fee_c']/a['q']))
   i+=1
  assert abs(pos-sum(s*q for s,q,p,f in lots))<1e-6
  out.append([pos,sum(q*p for s,q,p,f in lots)/abs(pos) if abs(pos)>1e-6 else None,sum(q*f for s,q,p,f in lots)/abs(pos) if abs(pos)>1e-6 else None])
 return out
def encode(p,q):
 keys=list(zip(p.day,p.mid,np.sign(q),np.searchsorted([50,200,500,1000],abs(q),side='left'),p.k//60));mapping={v:i for i,v in enumerate(sorted(set(keys)))}
 return np.array([mapping[k] for k in keys]),len(mapping)
def design(code,n,x,valid):
 mask=valid&np.isfinite(x);on=mask&(x>0);off=mask&~(x>0)
 n1=np.bincount(code[on],minlength=n);n0=np.bincount(code[off],minlength=n);both=(n1>0)&(n0>0);w=np.zeros(n);w[both]=n1[both]*n0[both]/(n1[both]+n0[both]);total=w.sum()
 weights=np.zeros(len(code))
 if total:weights[on]=w[code[on]]/n1[code[on]]/total;weights[off]=-w[code[off]]/n0[code[off]]/total
 return dict(weights=weights,on=on&both[code],off=off&both[code],informative=both,mask=mask)
def schedule(events,p):
 inds={(str(d),int(m)):g.index.to_numpy() for (d,m),g in p.groupby(['day','mid'],sort=True)};raw={k:[] for k in inds}
 for e in events.itertuples():
  key=(str(e.day),int(e.mid))
  if key in raw and np.isfinite(e.k_anchor) and 0<=e.k_anchor<300 and e.s in [-1,1]:raw[key].append((int(e.k_anchor),int(e.s)))
 for g in inds.values():assert len(g)==300
 return inds,raw
def outcome(inds,raw,q,offset=None):
 y=np.zeros(len(q),bool)
 for j,(key,g) in enumerate(inds.items()):
  ks=raw[key];off=0 if offset is None else int(offset[j])
  for k,s in ks:
   ix=g[(k+off)%300]
   if q[ix]*s< -1e-6:y[ix]=True
 return y
def run():
 resource.setrlimit(resource.RLIMIT_CPU,(80,81));start=time.process_time();B.bound()
 p=pd.read_parquet(C.D/'maker3_cap_skew_20261124/take_context_panel.parquet').sort_values(['day','mid','k']).reset_index(drop=True);p.day=p.day.astype(str)
 ev=pd.read_parquet(C.C/'take_logic_v2/events.parquet',filters=[('rank','=',3)]);rawstates=json.loads(gzip.decompress((C.OUT/'state_fills.json.gz').read_bytes()));states={(v['day'],v['mid']):v['fills'] for v in rawstates}
 for d in ['0','1']:
  for col in ['basis_position','basis_price','basis_fee']:p[col+d]=np.nan
  for key,g in p.groupby(['day','mid'],sort=True):
   v=np.array(basis_states(states[key],g.t_ms.to_numpy()-int(d)*1000),float);p.loc[g.index,['basis_position'+d,'basis_price'+d,'basis_fee'+d]]=v
 inds,raw=schedule(ev,p);rng=np.random.default_rng(20261124);offsets=rng.integers(0,300,size=(500,len(inds)));rows=[];diagnostic=[]
 for d,clock in [('0','D0'),('1','D1000')]:
  q=p['q'+d].to_numpy();obs=outcome(inds,raw,q);assert np.array_equal(obs,p['O1_'+clock].to_numpy(bool))
  code,n=encode(p,q);spread=(p['ask'+d]-p['bid'+d]).to_numpy()*100
  bp=p['basis_position'+d].to_numpy();agrees=np.isclose(bp,q,atol=1e-6,rtol=0);price=np.where(q>0,p['bid'+d],p['ask'+d])*100
  exit_edge=np.sign(q)*(price-p['basis_price'+d].to_numpy())-p['basis_fee'+d].to_numpy()-.02*np.minimum(price,100-price);exit_edge[~agrees]=np.nan
  valid=p.clean.to_numpy(bool)&(abs(q)>1e-6)&p['kok'+d].to_numpy(bool)&p['bok'+d].to_numpy(bool)
  signals={'adding_burst_100':(p['add_5_'+d]-p['reduce_5_'+d]).to_numpy()>=100,'no_passive_reduction_5s':p['reduce_5_'+d].to_numpy()<=1e-9,
   'btc_against_1bp':q/np.where(q!=0,abs(q),1)*p['r_5s'+d].to_numpy()<=-1,'profitable_exit_1c':np.where(np.isfinite(exit_edge),exit_edge>1,np.nan),'wide_book_4c':spread>4+1e-7}
  Y=np.array([outcome(inds,raw,q,o) for o in offsets],dtype=float)
  assert len(Y)==500
  for name,x in signals.items():
   x=np.asarray(x,float);des=design(code,n,x,valid);w=des['weights'];a,b=des['on'],des['off'];rd=float(obs@w) if np.any(w) else None;draws=Y@w
   markets=len(set(p.loc[(a|b),'mid']));n1=int(a.sum());n0=int(b.sum());e1=int(obs[a].sum());e0=int(obs[b].sum());support=n1>=50 and n0>=50 and e1>=10 and e0>=10 and markets>=5
   perday={day:float(obs[mask]@w[mask]) for day in C.DAYS for mask in [(p.day==day).to_numpy()]}
   rows.append(dict(clock=clock,feature=name,on_seconds=n1,off_seconds=n0,on_events=e1,off_events=e0,informative_markets=markets,matched_RD=rd,
    null_mean=float(draws.mean()),null_p05=float(np.quantile(draws,.05)),null_p95=float(np.quantile(draws,.95)),p_hi=(1+int((draws>=rd).sum()))/501 if rd is not None else None,
    p_lo=(1+int((draws<=rd).sum()))/501 if rd is not None else None,support=support,day_contribution=perday))
  diagnostic.append(dict(clock=clock,raw_takes=sum(len(x) for x in raw.values()),opposing_seconds=int(obs.sum()),base_valid_seconds=int(valid.sum()),basis_disagreement=int((valid&~agrees).sum())))
  p['exit_edge_'+d]=exit_edge
 assert not (C.H/'risk.json').exists();C.dump(C.H/'risk.json',dict(rows=rows,diagnostic=diagnostic,seed=20261124,draws=500,cpu_s=time.process_time()-start,validation=False));p.to_parquet(C.OUT/'risk_states.parquet',index=False)
 print(json.dumps(C.clean(dict(rows=rows,diagnostic=diagnostic)),allow_nan=False));assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1500*1024
if __name__=='__main__':run()
