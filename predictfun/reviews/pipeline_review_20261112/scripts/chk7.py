import pandas as pd, numpy as np
B='/home/yuqing/ctaNew/data/mm_hf/raw/bookTicker/BTCUSDT'
for h in ['20261023_14','20261027_14','20261103_14']:
    d=pd.read_csv(f'{B}/{h}.csv.gz',header=None,usecols=[0,1,3,4,6],names=['recv','E','u','bid','ask']).drop_duplicates('u')
    E=d.E.values.astype(float); lat=d.recv.values/1e6-E; ex=np.maximum(lat-np.median(lat),0)
    m=(d.bid.values+d.ask.values)/2
    kA=E+3.5; oA=np.argsort(kA,kind='stable')
    kB=np.maximum.accumulate((E+3.5+ex)[np.argsort(d.recv.values,kind='stable')]); oB=np.argsort(d.recv.values,kind='stable')
    g=np.arange(np.ceil(kA.min()/50)*50+60000, kA.max()-60000, 50.0)
    iA=np.searchsorted(kA[oA],g,side='right')-1; iB=np.searchsorted(kB,g,side='right')-1
    mA=m[oA][iA]; mB=m[oB][iB]
    jA=np.searchsorted(kA[oA],g-250,side='right')-1; jB=np.searchsorted(kB,g-250,side='right')-1
    rA=1e4*np.log(mA/m[oA][jA]); rB=1e4*np.log(mB/m[oB][jB])
    print(h,'grid steps',len(g),'mid differs %.2f%%'%(100*np.mean(mA!=mB)),'|d bn_r250|>0.5 bps %.2f%%'%(100*np.mean(np.abs(rA-rB)>0.5)))
