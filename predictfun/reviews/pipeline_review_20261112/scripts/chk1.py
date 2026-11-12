import pandas as pd, numpy as np, sys
IO='/home/yuqing/ctaNew/data/predict_fun/research'
g=pd.read_parquet(f'{IO}/step4/badfill_info_tx_tk/grid.parquet',columns=['day'])
print('grid day dtype',g.day.dtype, g.day.value_counts().sort_index().to_dict())
del g
DAYS=['20261023','20261024','20261025','20261026','20261027','20261031','20261101','20261102','20261103']
for d in DAYS:
    r=pd.read_parquet(f'{IO}/step5/{d}/rows_rl_d2_pk.parquet',columns=['mid','tau','t_rel','pf_mid'])
    p=pd.read_parquet(f'{IO}/step5/{d}/preds_t60k.parquet',columns=['mid','tau','v_bid','v_ask'])
    p0=pd.read_parquet(f'{IO}/step5/{d}/preds_t60.parquet',columns=['mid','tau','v_bid','v_ask'])
    same = len(r)==len(p)==len(p0) and (r.mid.values==p.mid.values).all() and (r.tau.values==p.tau.values).all() and (p0.tau.values==p.tau.values).all()
    live=((r.t_rel<240)&r.pf_mid.notna()).values
    nanok=bool((np.isnan(p0.v_bid.values)==~live).all())
    m1=(p.v_bid.values==-1)
    diff=(p.v_bid.values!=p0.v_bid.values)&~(np.isnan(p.v_bid.values)&np.isnan(p0.v_bid.values))
    print(d,'rows',len(r),'tau dtype',r.tau.dtype,p.tau.dtype,'exact row align',same,'NaN==~live',nanok,
          'v=-1 rows',int(m1.sum()),'changed rows',int(diff.sum()),'changed but not -1',int((diff&~m1).sum()),
          'p0 already -1',int((p0.v_bid.values==-1).sum()), 'tau%50!=0',int((np.mod(r.tau.values,50)!=0).sum()))
