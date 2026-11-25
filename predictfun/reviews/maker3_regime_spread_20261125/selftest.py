import copy
import numpy as np
import pandas as pd
import common as C
import check as K
import build as B
rows=[]
for mid in [1,2]:
 for k in range(300):
  m=50 if mid==1 else 50+(k%2)*2;s=6 if mid==1 else 1
  rows.append(dict(day='20261024',mid=mid,k=k,t_ms=1000000+k*1000,bid0=(m-s/2)/100,ask0=(m+s/2)/100,kok0=True,r_1s0=0 if mid==1 else (-1)**k,bok0=True))
p=pd.DataFrame(rows);f=C.features(p);K.feature_check(p,f);bounds=C.boundaries(f)
assert f.loc[(f.mid==1)&(f.k>=10),'pf'].eq(0).all()
assert np.allclose(f.loc[(f.mid==2)&(f.k>=10),'pf'],2)
assert C.sb(6)=='wide' and C.vb(0,bounds['pf'])=='calm';assert C.sb(1)=='narrow' and C.vb(3,bounds['pf'])=='volatile'
assert f.loc[f.k<10,['pf','btc']].isna().all().all();assert C.sb(None)=='unknown' and C.vb(None,bounds['pf'])=='unknown'
bad=p.copy();bad.loc[(bad.mid==1)&(bad.k==50),'kok0']=False;bf=C.features(bad);assert bf.loc[(bf.mid==1)&bf.k.between(50,60),'pf'].isna().all()
future=p.copy();mask=future.k>150;future.loc[mask,'bid0']+=future.loc[mask,'k']*.001;future.loc[mask,'ask0']+=future.loc[mask,'k']*.001;ff=C.features(future)
assert f.loc[f.k<=150].equals(ff.loc[ff.k<=150]);assert not f.loc[f.k>150].equals(ff.loc[ff.k>150])
grid={int(x['k']):x for x in f[f.mid==1].to_dict('records')};fill=dict(hi_ms=1050999,lo_ms=1050000,book={c:dict(spread=6) for c in C.CLOCKS})
x=C.context(fill,1000000,grid,'pre','pf',bounds);assert x['anchor_ms']==1049000 and x['vol']=='calm'
y=C.context(fill,1000000,grid,'upper','pf',bounds);assert y['anchor_ms']==1050000
u=copy.deepcopy(fill);u['lo_ms']=None;assert C.context(u,1000000,grid,'pre','pf',bounds)['vol']=='unknown'
# Same positive net can come from a calm-wide book or volatile-narrow price drift;
# classify independently and retain both positive/negative accounting parts.
ex=pd.DataFrame([dict(day='20261024',mid=1,id='1',entry='a',exit='b',entry_hash='a',exit_hash='b',role='MM',clock='pre',asset='pf',axis='entry',vol='calm',spread='wide',q=10,net_c=60,positive_c=60,negative_c=0),dict(day='20261024',mid=1,id='2',entry='c',exit='d',entry_hash='c',exit_hash='d',role='MM',clock='pre',asset='pf',axis='entry',vol='calm',spread='wide',q=10,net_c=-40,positive_c=0,negative_c=-40)])
agg=B.aggregate(ex,'pairs');r=agg[(agg.scope=='all19')&(agg.day=='all')&(agg.role=='all')].iloc[0];assert r.net_c==20 and r.positive_c==60 and r.negative_c==-40 and r.net_c_per_share==1
# Inventory-side symmetry of the remaining-lot identity, fees allocated only once.
for s in [1,-1]:
 entry=50-10*s;exit=50+10*s;mark=50+5*s;q=10;closed=6
 completed=s*closed*(exit-entry)-closed*(1/q+1/closed);residual=s*(q-closed)*(mark-entry)-(q-closed)/q
 cash=-s*q*entry+s*closed*exit;pos=s*(q-closed)
 assert abs(completed+residual-(cash+pos*mark-2))<1e-9
print('PASS calm-wide/volatile-narrow, independent rolling windows, future perturbation and delivered positive control, stale/warmup, clock floors, positive/negative conservation, long/short residual identity')
