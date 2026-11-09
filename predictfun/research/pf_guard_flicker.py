"""Guard-value flicker at the minute-0 pull line (user 2026-11-09: is the churn a prediction-model issue?). 9 consumed in-sample days,
preds_t60k, 0-60 s, each market and side: |v - line|, |dv| per 50 ms step, side on/off flips per window with a hysteresis band delta
(on -> off at v <= line - delta, off -> on at v > line + delta; delta 0 = the frozen rule), on / off spell lengths. Descriptive only.
Usage: python3 -m predictfun.research.pf_guard_flicker
"""
from predictfun.research import pf_io as io
from predictfun.research.pf_inv_report import DAYS
LINE=json.load(open(f'{io.OUT}/step4/minute_lines_t60k.json'))['lines']['90'][0]
res=[]; runs_on=[]; runs_off=[]; dv=[]; dist=[]
for d in DAYS:
    p=pd.read_parquet(f'{io.OUT}/step5/{d}/preds_t60k.parquet',columns=['mid','tau','v_bid','v_ask'])
    ws=pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet',columns=['mid','ws']).set_index('mid').ws
    p['rel']=(p.tau-p.mid.map(ws)*1000)/1000; p=p[(p.rel>=0)&(p.rel<60)]
    for m,g in p.groupby('mid'):
        g=g.sort_values('tau')
        for col in ('v_bid','v_ask'):
            v=g[col].values; ok=np.isfinite(v)
            if ok.sum()<100: continue
            v=v[ok]; t=g.tau.values[ok]
            dv.append(np.abs(np.diff(v))); dist.append(np.abs(v-LINE))
            for delta in (0.0,0.005,0.01,0.02,0.04):
                on=v[0]>LINE; flips=0; ton=0.0; st=t[0]; seg=[]
                for k in range(1,len(v)):
                    if on and v[k]<=LINE-delta: on=False; flips+=1; seg.append((True,t[k]-st)); st=t[k]
                    elif (not on) and v[k]>LINE+delta: on=True; flips+=1; seg.append((False,t[k]-st)); st=t[k]
                if delta==0.0:
                    runs_on+= [x for o,x in seg if o]; runs_off+=[x for o,x in seg if not o]
                res.append(dict(day=d,delta=delta,flips=flips))
R=pd.DataFrame(res)
dv=np.concatenate(dv); dist=np.concatenate(dist)
print('minute-0 pull line', LINE, 'c | |v - line| p25/50/75', np.round(np.quantile(dist,[.25,.5,.75]),4), '| |dv| per 50 ms step p50/p75/p90', np.round(np.quantile(dv,[.5,.75,.9]),4))
print('flips per side per window (0-60 s) by buffer delta:', R.groupby('delta').flips.mean().round(1).to_dict())
ro=np.array(runs_on); rf=np.array(runs_off)
print('quoting spells ms p25/50/75', np.round(np.quantile(ro,[.25,.5,.75])), '| pulled spells ms p25/50/75', np.round(np.quantile(rf,[.25,.5,.75])), '| pulled spells <= 100 ms %.2f, <= 250 ms %.2f' % ((rf<=100).mean(), (rf<=250).mean()))
