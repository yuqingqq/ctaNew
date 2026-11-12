import pandas as pd, numpy as np, glob, json, sys
IO='/home/yuqing/ctaNew/data/predict_fun/research'; d=sys.argv[1]
ck=json.load(open(f'{IO}/step1/{d}/clock_rl_d2.json')); SH=ck['shift_ms']
reg=pd.read_parquet(f'{IO}/step1/{d}/registry.parquet',columns=['mid','ws'])
d0=pd.Timestamp(f'{d[:4]}-{d[4:6]}-{d[6:]}',tz='UTC').value//1_000_000
W0={int(m):w*1000.0 for m,w in zip(reg.mid,reg.ws) if d0<=w*1000.0<d0+86400000}
parts=[pd.read_parquet(f,columns=['mid','t_ms','recv_ns','b1_p','a1_p']) for f in sorted(glob.glob(f'{IO}/step1/{d}/books/part_{d}_*.parquet'))]
B=pd.concat(parts,ignore_index=True); B=B[B.mid.isin(W0)]
nd=0; ngrid=0; nbest=0
for m,g in B.groupby('mid',sort=False):
    ws=W0[m]
    gf=g.sort_values(['t_ms','recv_ns'],kind='stable')
    kfull=np.maximum.accumulate(np.maximum(gf.t_ms.values+13.2,gf.recv_ns.values/1e6-SH)+2.0)
    rel=g.t_ms.values-ws; gs=g[(rel>=-5000)&(rel<=125000)].sort_values('t_ms',kind='stable')
    ksub=np.maximum.accumulate(np.maximum(gs.t_ms.values+13.2,gs.recv_ns.values/1e6-SH)+2.0)
    tg=ws+np.arange(0,60000,50.0)
    i1=np.searchsorted(kfull,tg,side='right')-1; i2=np.searchsorted(ksub,tg,side='right')-1
    t1=np.where(i1>=0,gf.t_ms.values[np.clip(i1,0,None)],-1); t2=np.where(i2>=0,gs.t_ms.values[np.clip(i2,0,None)],-1)
    b1=np.where(i1>=0,gf.b1_p.values[np.clip(i1,0,None)],-9); b2=np.where(i2>=0,gs.b1_p.values[np.clip(i2,0,None)],-9)
    nd+=int((t1!=t2).sum()); nbest+=int((b1!=b2).sum()); ngrid+=len(tg)
print(d,'grid steps 0-60 s',ngrid,'steps where the latest known frame (by t_ms) differs, replay subset vs features full history:',nd,'best bid differs:',nbest)
