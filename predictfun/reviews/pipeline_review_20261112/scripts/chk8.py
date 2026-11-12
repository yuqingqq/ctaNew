import pandas as pd, numpy as np
IO='/home/yuqing/ctaNew/data/predict_fun/research'; d='20261024'
r=pd.read_parquet(f'{IO}/step5/{d}/rows_rl_d2_pk.parquet',columns=['t_rel','gap','sigma','basis_bps'])
p=pd.read_parquet(f'{IO}/step5/{d}/preds_t60k.parquet',columns=['v_bid'])
w=(p.v_bid.values==-1)
print(d,'withheld rows (start price not yet known)',int(w.sum()),'of which gap finite',int(np.isfinite(r.gap.values[w]).sum()),
      't_rel of withheld rows p50/max %.2f/%.2f s'%(np.median(r.t_rel.values[w]),r.t_rel.values[w].max()))
g=pd.read_parquet(f'{IO}/step4/badfill_info_tx_tk/grid.parquet',columns=['day','t_rel','gap'])
print('guard grid rows with t_rel<2.4 s: %.2f%% of grid'%(100*np.mean(g.t_rel.values<2.4)))
