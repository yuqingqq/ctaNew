import pandas as pd, numpy as np
IO='/home/yuqing/ctaNew/data/predict_fun/research'
D8=['20261023','20261024','20261025','20261026','20261027','20261031','20261101','20261102']
V={}
for d in D8+['20261103']:
    p=pd.read_parquet(f'{IO}/step5/{d}/preds_t60k.parquet',columns=['mid','tau','v_bid','v_ask'])
    ws=pd.read_parquet(f'{IO}/step1/{d}/registry.parquet',columns=['mid','ws']).set_index('mid').ws
    rel=(p.tau.values-p.mid.map(ws).values*1000.0)/1000.0
    k=(rel>=0)&(rel<60)
    v=np.r_[p.v_bid.values[k],p.v_ask.values[k]]; V[d]=v[np.isfinite(v)]
allv=np.concatenate([V[d] for d in D8])
print('p90 min0 pooled 8d (incl -1):',round(np.percentile(allv,90),4),' excl -1:',round(np.percentile(allv[allv!=-1],90),4),' share -1 in min0 %.2f%%'%(100*np.mean(allv==-1)))
for d in D8:
    o=np.concatenate([V[x] for x in D8 if x!=d])
    own=V[d]
    l=np.percentile(o,90)
    print(d,'LODO p90 %.4f'%l,'own-day p90 %.4f'%np.percentile(own,90),'share own v>0.0492 %.3f vs v>LODO %.3f'%(np.mean(own>0.0492),np.mean(own>l)))
print('20261103 own p90 %.4f share>0.0492 %.3f'%(np.percentile(V['20261103'],90),np.mean(V['20261103']>0.0492)))
