import pandas as pd, numpy as np
IO='/home/yuqing/ctaNew/data/predict_fun/research'; B='/home/yuqing/ctaNew/data/mm_hf/raw/bookTicker/BTCUSDT'
for h in ['20261023_14','20261103_14']:
    d=pd.read_csv(f'{B}/{h}.csv.gz',header=None,usecols=[0,1],names=['recv','E'])
    bl=d.recv.values/1e6-d.E.values; bs=(d.recv.values//1_000_000_000)
    bx=pd.Series(bl-np.median(bl)).groupby(bs).max()
    p=pd.read_parquet(f'{IO}/step1/{h[:8]}/books/part_{h}.parquet',columns=['t_ms','recv_ns'])
    pl=p.recv_ns.values/1e6-p.t_ms.values; ps=p.recv_ns.values//1_000_000_000
    px=pd.Series(pl-np.median(pl)).groupby(ps).max()
    j=pd.concat([bx.rename('bn'),px.rename('pf')],axis=1).dropna()
    hi=j.bn>200
    print(h,'seconds',len(j),'secs with BN excess>200ms',int(hi.sum()),'of which PF excess>100ms',int((hi&(j.pf>100)).sum()),
          'PF excess>100ms overall share %.3f'%np.mean(j.pf>100),'spearman(bn,pf) %.2f'%j.corr(method='spearman').iloc[0,1])
    print('  BN share of messages with excess >50ms %.4f, >200ms %.4f'%(np.mean(bl-np.median(bl)>50),np.mean(bl-np.median(bl)>200)))
