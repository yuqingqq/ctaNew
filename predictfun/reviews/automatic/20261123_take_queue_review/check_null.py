"""Independent 500-shift recount on saved TAKE panel, without context loading."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew')
H=Path(__file__).resolve().parent
D=ROOT/'predictfun/reviews/maker_trace_20261122/v2/chains/take_logic_v2'
p=pd.read_parquet(D/'panel.parquet'); t=pd.read_parquet(D/'events.parquet')
r=pd.read_csv(D/'ratios.csv',float_precision='round_trip')
t=t[t.in_raw_schedule]
gi=t.gi.to_numpy(int); s=t.s.to_numpy(int)
sideok=~t.status.fillna('').str.split('+').apply(lambda z:'side_mixed' in z).to_numpy()
n=len(p); dst=np.empty((500,len(gi)),int)
for pair,x in p.groupby('pair',sort=False):
    row=x.iloc[0]; base=x.index[0]; raw=p.pair.to_numpy()[gi]==pair
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261123,int(row.day),int(row.mid),int(row['rank'])])))
    shift=rng.integers(0,len(x),size=500)
    dst[:,raw]=base+(p.k.to_numpy()[gi[raw]][None,:]+shift[:,None])%len(x)
q=p.q0.to_numpy(); sg=np.sign(np.where(abs(q)<=1e-6,0,q));rank=p['rank'].to_numpy(int)-1
a5=sg*p.r_5s0.to_numpy(); cat=np.select([~np.isfinite(a5),a5>=1,a5<=-1],[3,0,1],2)
tl=p.tl0.to_numpy();tb=np.select([tl>300,tl>240,tl>120,tl>60,tl>30,tl>10],[0,1,2,3,4,5],6)
qb=np.select([abs(q)<=1e-6,abs(q)<=50,abs(q)<=200,abs(q)<=500],[0,1,2,3],4)
out=[]
for ratio in ['RR_size','RR_end']:
    if ratio=='RR_size':
        group=np.select([qb>=3,qb==1],[1,0],-1); st=rank*28+cat*7+tb; ns=84
    else:
        group=np.select([tb>=5,(tb==2)|(tb==3)],[1,0],-1);group=np.where(qb>0,group,-1)
        st=(rank*4+qb-1)*4+cat;ns=48
    good=group>=0;counts=np.bincount(st[good]*2+group[good],minlength=ns*2).reshape(ns,2)
    a,b=counts[:,1],counts[:,0];valid=(a>0)&(b>0);tot=np.maximum(a+b,1)
    wn=np.zeros(n);wd=np.zeros(n)
    z=good&(group==1);wn[z]=np.where(valid[st[z]],b[st[z]]/tot[st[z]],0.)
    z=good&(group==0);wd[z]=np.where(valid[st[z]],a[st[z]]/tot[st[z]],0.)
    def value(ix):
        ix=np.unique(ix);num=wn[ix].sum();den=wd[ix].sum()
        return num/den if den else (np.inf if num else np.nan)
    for name,pol in [('O1',-1),('O2',1)]:
        obs=value(gi[sideok&(s*sg[gi]==pol)])
        vals=[]
        for j in range(500):
            d=dst[j];vals.append(value(d[sideok&(s*sg[d]==pol)]))
        vals=np.asarray(vals);tol=1e-12*max(abs(obs),1e-300)
        ph=(1+np.count_nonzero(vals>=obs-tol)+np.count_nonzero(np.isnan(vals)))/501
        pl=(1+np.count_nonzero(vals<=obs+tol)+np.count_nonzero(np.isnan(vals)))/501
        row=r[(r.panel=='full')&(r.scope=='pooled')&(r.variant=='D0')&(r.ratio==ratio)&(r.outcome==name)].iloc[0]
        out.append({'outcome':name,'ratio':ratio,'RR_MH':obs,'p_hi':ph,'p_lo':pl,'saved_RR':row.RR_MH,'saved_p_hi':row.p_hi,'saved_p_lo':row.p_lo,
                    'ok':bool(np.allclose([obs,ph,pl],[row.RR_MH,row.p_hi,row.p_lo],rtol=1e-12,atol=1e-12))})
(H/'null_evidence.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
