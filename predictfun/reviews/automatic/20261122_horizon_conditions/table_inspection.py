from pathlib import Path
import gzip,io,json,subprocess
import numpy as np,pandas as pd
R=Path('/home/yuqing/ctaNew');O=R/'predictfun/reviews/automatic/20261122_horizon_conditions'
F=pd.read_csv(O/'horizons.csv');T=pd.read_csv(O/'transitions.csv');C=pd.read_csv(O/'conditions.csv')
print('COMMON HORIZONS');print(F[(F.cohort=='common')&(F.day=='all')&(F.side=='all')][['arm','horizon_ms','valid','q','net','per_share','age_median_ms','age_max_ms']].to_string(index=False))
print('FULL COMMON BY DAY');print(F[(F.arm=='FULL')&(F.cohort=='common')&(F.side=='all')&(F.day!='all')&F.horizon_ms.isin(['1000','5000','30000'])][['day','horizon_ms','net','per_share']].to_string(index=False))
print('RECOVERY');print(T.to_string(index=False))
print('CONDITIONS (DESCRIPTIVE, NO RANKING)')
for name in ('pressure_flow','depletion','inventory','age'):
 g=C[(C.condition==name)&(C.day=='all')&(C.side=='all')]
 print(name);print(g[['band','horizon_ms','rows','common','no_fill','q','net','per_opportunity','per_share']].to_string(index=False))
b=subprocess.check_output(['git','-C',str(R),'show','2063d31a:predictfun/reviews/ladder_audit_20261121/outputs/states.csv.gz'])
S=pd.read_csv(io.BytesIO(gzip.decompress(b)),float_precision='round_trip');S=S[S.kind=='live']
edges={0:.4666730940341952,1:.285456120967865};bad=[]
for row in S.itertuples():
 x=np.float32(row.adv_bn_r250);e=edges[int(row.side)]
 if (x>=e)!=(float(x)>=e):bad.append([row.day,row.m,row.gid,float(x),e])
print('FLOAT COMPARISON BOUNDARY MISMATCHES',bad)
(O/'boundary_check.json').write_text(json.dumps(dict(count=len(bad),examples=bad),indent=2)+'\n')
