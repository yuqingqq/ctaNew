import pandas as pd, numpy as np, glob, sys, json
IO='/home/yuqing/ctaNew/data/predict_fun/research'
for d in sys.argv[1:]:
    sh=json.load(open(f'{IO}/step1/{d}/clock_rl_d2.json'))['shift_ms']
    meds=[]; fl=[]
    for fn in sorted(glob.glob(f'{IO}/step1/{d}/books/part_*.parquet')):
        b=pd.read_parquet(fn,columns=['t_ms','recv_ns'])
        if fn.split('part_')[1][:8]!=d: continue
        lat=b.recv_ns.values/1e6-b.t_ms.values
        meds.append(np.median(lat)); fl.append(np.mean(b.t_ms.values+13.2 > b.recv_ns.values/1e6-sh))
    meds=np.array(meds)
    print(d,'day shift+53.1=%.1f'%(sh+53.1),'hourly median recv-t: min %.1f max %.1f'%(meds.min(),meds.max()),'hourly share of frames at the 13.2 floor: min %.3f max %.3f'%(min(fl),max(fl)))
