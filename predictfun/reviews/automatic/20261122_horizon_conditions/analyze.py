"""Predeclared saved-data horizon/condition analysis; see protocol.md."""
from pathlib import Path
import collections,gzip,hashlib,io,json,subprocess,time
import numpy as np
import pandas as pd
ROOT=Path('/home/yuqing/ctaNew');OUT=Path(__file__).resolve().parent
SRC='2063d31a91ab7e6124d7b8b7b7ec5df01ea66711'
DATA=ROOT/'data/predict_fun/research';H=(0,50,100,1000,5000,30000);t0=time.monotonic()
def blob(p):return subprocess.check_output(['git','-C',str(ROOT),'show',f'{SRC}:{p}'])
def sha(b):return hashlib.sha256(b).hexdigest()
def bind(data,receipt):
 b=Path(data).read_bytes();assert b==blob(receipt),str(data);return b
def gzrows(b):return [json.loads(l) for l in gzip.decompress(b).splitlines() if l]
def mvalue(s,q,p,m):return np.nan if m is None or not np.isfinite(m) else (1 if s==0 else -1)*q*(m-p)
assert mvalue(0,10,49,50)==10 and mvalue(1,10,51,50)==10
assert np.isnan(mvalue(0,10,49,None)) and mvalue(0,10,49,48)==-10
def verify(b,h):assert sha(b)==h,'input hash mismatch'
try:verify(b'changed','0'*64);raise RuntimeError('hash refusal failed')
except AssertionError:pass
rm=json.loads(blob('predictfun/reviews/ladder_pressure_residual_20261121/manifest.json'))
led={};base={};inputs={}
for day in rm['days']:
 pp=DATA/'ladder_pressure_v1_20261121/days'/f'{day}.json'
 b=bind(pp,f'predictfun/reviews/ladder_pressure_20261121/outputs/days/{day}.json');verify(b,rm['parent_day_records'][day]);r=json.loads(b)
 raw=pp.with_suffix('.jsonl.gz').read_bytes();verify(raw,r['sha256']['gz']);inputs[str(pp)]=sha(b)
 for rec in gzrows(raw):base[(day,int(rec['m']))]=rec['disabled']
 for part in ('pilot','rest'):
  pp=DATA/'ladder_pressure_ablation_v1_20261121'/part/f'{day}.json'
  b=bind(pp,f'predictfun/reviews/ladder_pressure_ablation_20261121/outputs/{part}/{day}.json');verify(b,rm['ablation']['part_records'][f'{part}|{day}']);r=json.loads(b)
  if not r['members']:continue
  raw=pp.with_suffix('.jsonl.gz').read_bytes();verify(raw,r['sha256']['gz']);inputs[str(pp)]=sha(b)
  for rec in gzrows(raw):
   k=(day,int(rec['m']));assert k not in led;led[k]=rec['FULL_11']
assert set(led)==set(base) and len(led)==19
rows=[]
for arm,ls in [('BASELINE',base),('FULL',led)]:
 for (day,m),L in ls.items():
  assert len(L['fills'])==len(L['fill_mids']);vs=0.
  for i,(f,mid) in enumerate(zip(L['fills'],L['fill_mids'])):
   tf,gid,s,q,p,_=f;v=mvalue(s,q,p,L['mark']);vs+=v
   z=dict(arm=arm,day=day,m=m,fill=i,gid=int(gid),side=int(s),q=q,t=tf,END=v)
   for j,h in enumerate(H):
    z[f'v{h}']=mvalue(s,q,p,mid[2*j]) if tf+h<=L['end'] else np.nan
    z[f'age{h}']=mid[2*j+1]
   rows.append(z)
  assert np.isclose(vs,L['value'],atol=1e-6,rtol=0)
F=pd.DataFrame(rows);assert not F.duplicated(['arm','day','m','fill']).any()
F['common']=F[[f'v{h}' for h in H]].notna().all(axis=1)
accepted=pd.read_csv(io.BytesIO(blob('predictfun/reviews/ladder_pressure_ablation_20261121/outputs/arms.csv')),dtype={'day':str},float_precision='round_trip')
for arm,an in [('BASELINE','BASELINE_00'),('FULL','FULL_11')]:
 for (day,m),g in F[F.arm==arm].groupby(['day','m']):
  a=accepted[(accepted.arm==an)&(accepted.day==day)&(accepted.m==m)].iloc[0]
  assert np.isclose(g.v1000.sum(),a.all_id_mo1s,atol=1e-6,rtol=0)
out=[];trans=[]
for arm,g0 in F.groupby('arm'):
 for cohort in ('native','common'):
  g=g0 if cohort=='native' else g0[g0.common]
  for day in ['all']+sorted(g.day.unique()):
   for side in ['all',0,1]:
    gg=g[(g.day==day) if day!='all' else np.ones(len(g),bool)];gg=gg[(gg.side==side) if side!='all' else np.ones(len(gg),bool)]
    for h in (*H,'END'):
     k='END' if h=='END' else f'v{h}';u=gg[gg[k].notna()];v=u[k];q=u.q.sum()
     out.append(dict(arm=arm,cohort=cohort,day=day,side=side,horizon_ms=h,rows=len(gg),valid=len(u),censored=len(gg)-len(u),q=q,net=v.sum(),positive=v[v>0].sum(),negative=v[v<0].sum(),per_share=v.sum()/q if q else np.nan,
                     age_median_ms=u[f'age{h}'].median() if h!='END' else np.nan,age_max_ms=u[f'age{h}'].max() if h!='END' else np.nan))
 g=g0[g0.common & (g0.v1000<0)]
 for h in (5000,30000):
  u=g[g[f'v{h}']>=0]
  trans.append(dict(arm=arm,horizon_ms=h,negative_1s_fills=len(g),negative_1s_shares=g.q.sum(),recovered_fills=len(u),recovered_shares=u.q.sum(),recovered_original_loss_cents=-u.v1000.sum(),original_loss_cents=-g.v1000.sum(),net_at_h=g[f'v{h}'].sum()))
pd.DataFrame(out).to_csv(OUT/'horizons.csv',index=False);pd.DataFrame(trans).to_csv(OUT/'transitions.csv',index=False)
# Decision-state diagnostic on the already frozen neutral baseline snapshots.
state_b=blob('predictfun/reviews/ladder_audit_20261121/outputs/states.csv.gz')
verify(state_b,'ec2eaf7cd53628170fe485bb55a9889ae77e7f85188e592b078de6f1ec9a3043')
S=pd.read_csv(io.BytesIO(gzip.decompress(state_b)),dtype={'day':str},float_precision='round_trip');S=S[S.kind=='live'].copy()
assert not S.duplicated(['day','m','gid']).any()
win={};orders={}
for k,L in base.items():
 orders[k]={int(o['gid']):o for o in L['orders']};d=collections.defaultdict(list)
 for f,mid in zip(L['fills'],L['fill_mids']):d[int(f[1])].append((f,mid))
 win[k]=d
sr=[]
for z in S.to_dict('records'):
 k=(z['day'],int(z['m']));o=orders[k][int(z['gid'])];d=z['t'];effect=max(d+37,o['submit']+95) if d<o['act'] else d+37
 assert effect==z['effect'];ff=[(f,mid) for f,mid in win[k][int(z['gid'])] if f[0]>=effect]
 v={};valid={}
 for h,j in ((1000,3),(5000,4),(30000,5)):
  vv=[mvalue(f[2],f[3],f[4],mid[2*j]) if f[0]+h<=base[k]['end'] else np.nan for f,mid in ff]
  valid[h]=all(np.isfinite(a) for a in vv);v[h]=sum(vv) if valid[h] else np.nan
 assert (np.isnan(v[1000]) and pd.isna(z['win_mo1s'])) or np.isclose(v[1000],z['win_mo1s'],atol=1e-6,rtol=0)
 bn=np.float32(z['adv_bn_r250']);pf=z['adv_ofi250'];edge=rm['edges'][str(int(z['side']))][-1]
 b='missing' if not np.isfinite(bn) else 'non_adverse' if bn<=0 else 'q5' if bn>=edge else 'positive_below_q5'
 p='missing' if not np.isfinite(pf) else 'adverse' if pf>0 else 'non_adverse'
 age=z['ctx_generation_age_ms'];inv=z['inv_signed'];depth=z['o_depth'];depl=z['own_d250']
 sr.append(dict(day=z['day'],m=int(z['m']),side=int(z['side']),gid=int(z['gid']),q=sum(f[3] for f,mid in ff),fills=len(ff),no_fill=not ff,
                v1000=v[1000],v5000=v[5000],v30000=v[30000],common=all(valid.values()),pressure_flow=b+'|PF_'+p,
                depletion='missing' if not np.isfinite(depl) else 'depleting' if depl<0 else 'not_depleting',
                depth='missing' if not np.isfinite(depth) else str(depth),age='missing' if not np.isfinite(age) else '<0.5s' if age<500 else '0.5-2s' if age<2000 else '2-10s' if age<10000 else '>=10s',
                inventory='missing' if not np.isfinite(inv) else 'adding' if inv>0 else 'reducing' if inv<0 else 'flat'))
W=pd.DataFrame(sr);tab=[]
for col in ('pressure_flow','depletion','depth','age','inventory'):
 for key,gg in W.groupby(col):
  for day in ['all']+sorted(gg.day.unique()):
   for side in ['all',0,1]:
    g=gg[(gg.day==day) if day!='all' else np.ones(len(gg),bool)];g=g[(g.side==side) if side!='all' else np.ones(len(g),bool)];u=g[g.common]
    for h in (1000,5000,30000):
     v=u[f'v{h}'];q=u.q.sum()
     tab.append(dict(condition=col,band=key,day=day,side=side,horizon_ms=h,rows=len(g),common=len(u),censored=len(g)-len(u),no_fill=int(u.no_fill.sum()),q=q,
                     net=v.sum(),positive=v[v>0].sum(),negative=v[v<0].sum(),per_opportunity=v.mean(),per_share=v.sum()/q if q else np.nan))
pd.DataFrame(tab).to_csv(OUT/'conditions.csv',index=False)
summary=dict(source=SRC,inputs=inputs,markets=len(led),days=rm['days'],full_fills=int((F.arm=='FULL').sum()),full_common_fills=int(((F.arm=='FULL')&F.common).sum()),
             neutral_live_rows=len(W),neutral_common_rows=int(W.common.sum()),neutral_censored_rows=int((~W.common).sum()),neutral_common_no_fill=int(W[W.common].no_fill.sum()),
             checks='hashes, hand marks, missing propagation, unique keys, membership, END and accepted +1s reproduced',elapsed_s=time.monotonic()-t0)
assert summary['elapsed_s']<36,'bounded saved-data audit exceeded 0.01h wall allowance'
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
