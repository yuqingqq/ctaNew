import sys; sys.path.insert(0,'/home/yuqing/ctaNew/predictfun/research')
import pandas as pd, numpy as np
from pf_review_gate_bands import capped
IO='/home/yuqing/ctaNew/data/predict_fun/research'; d='20261031'; P90=0.0492; INF=float('inf')
p=pd.read_parquet(f'{IO}/step5/{d}/preds_t60k.parquet',columns=['mid','tau','v_bid','v_ask'])
bad=0; n=0; badn=0
for m,g in p.groupby('mid'):
    tau=g.tau.values.astype(float)
    for v in (g.v_bid.values.astype(float), g.v_ask.values.astype(float)):
        s=capped(tau,v,P90,P90,INF)[0]; bad+=int((s!=(np.isfinite(v)&(v>P90))).sum()); n+=len(v)
        sn=capped(tau,v,-INF,-INF,INF)[0]; badn+=int((sn!=np.isfinite(v)).sum())
print(d,'steps',n,'stv3 != (v > PM90):',bad,'stn != isfinite(v):',badn)
