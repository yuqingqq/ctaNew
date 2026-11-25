from collections import Counter
import gzip
import hashlib
import json
import math
import resource
import sys

import numpy as np
import pandas as pd
from measure import H, D, V, DATA, features
from risk import risks, allowed, selftest

sys.path.insert(0,str(H.parent/'maker3_cap_skew_20261124'))
import experiment as cap

LB, BE, X = cap.LB, cap.BE, cap.X
receipt=H.parent/'inventory_exit_exception_20261125'
WIDE=(1000,500,250,False)

class Guard(cap.O.OpeningGate):
    def slot(self,v,r,price):
        action,size,reason=super().slot(v,r,price)
        if self.arm=='WIDE' or action!='POST':
            return action,size,reason
        side=v['side']
        mid=(v['bid']+v['ask'])/200
        cost=None
        if 0<=v['bid']<v['ask']<=100 and 0<=v['t']-v['received']<=1000:
            fair=self.prob[int(v['t']//BE.STEP)]
            q=(mid+fair)/2 if self.arm=='BTC' and np.isfinite(fair) else mid
            cost=risks(mid,q)[side]
        signed=(1-2*side)*v['position']
        reserved=v['A'] if side else v['B']
        new=allowed(size,signed,reserved,cost)
        assert new>=min(size,math.floor(max(0.,-signed-reserved)+1e-9))
        if new==size:
            return action,size,reason
        self.adjusted+=1
        return ('POST',new,'risk_size') if new>=LB.QMIN else ('WAIT',None,'risk_room')

def one(path,cache,old,edges,prob,arm):
    b=cap.Branch(path,old['end'],old['ident'],200,WIDE)
    pol=Guard(b,cache,edges)
    pol.arm,pol.prob,pol.adjusted=arm,prob,0
    assert Guard.live is cap.O.OpeningGate.live
    both=0
    for k in range(LB.n_ticks(b)):
        LB.step(b,k,pol)
        t=k*BE.STEP
        sides={b.E.o[i]['side'] for ids in b.unclosed.values() for i in ids
               if not b.E.o[i]['refused'] and b.E.o[i]['act']<=t<b.E.o[i]['remove'] and b.E.o[i]['rem']>1e-9}
        both+=len(sides)==2
    lg=LB.finish(b)
    fills=lg['fills']
    val=X.fifo(fills,b.mark)
    cash,pos,total=b.E.value(b.mark)
    assert abs(total-val['total_c'])<1e-5 and abs(pos-val['net_shares'])<1e-6
    changes=[(1-2*f[2])*f[3] for f in fills]
    peak=float(np.max(np.abs(np.cumsum([0.]+changes))))
    return dict(arm=arm,net_usd=total/100,turnover_usd=val['completed_c']/100,
                residual_usd=val['residual_c']/100,peak_sh=peak,both=both/LB.n_ticks(b),
                adjusted=pol.adjusted,passive_sh=sum(abs(x) for x in changes),
                pulls=sum(c[7]=='PULL' for c in b.cmd_log)), fills

def main(extend):
    resource.setrlimit(resource.RLIMIT_CPU,(180,181) if extend else (90,91))
    selftest()
    if extend:
        assert json.loads((H/'forecast.json').read_text())['passed']
    cases=json.loads((receipt/'manifest.json').read_text())['cases']
    refs=json.loads((receipt/'summary.json').read_text())['rows']
    hashes={r['mid']:r['sha256'] for r in refs if r['arm']=='WIDE'}
    edges=cap.T.P.frozen_edges()
    rows=[]
    for day,mid,part in cases:
        if (mid in [3702960,3789484,3877472])==extend:
            continue
        old=X.records(day,part)[mid]['CURRENT_ASYNC_FAST10']
        path,cache=X.get_path(day,mid,part,old)
        ident=old['ident']
        ws=int(ident['opening_ms'])
        assert ident['end_abs_ms']==ws+298000
        ticks=int(np.ceil(old['end']/BE.STEP))+1
        times=np.floor(ident['start_abs_ms']+BE.STEP*np.arange(ticks)).astype('int64')
        bt=D.load_btc(V/'btc',day,Counter())
        prob=features(bt,times,np.full(ticks,ws))['pf']
        del bt
        parent=(DATA/'inventory_exit_exception_20261125'/f'{mid}_WIDE.json.gz').read_bytes()
        assert hashlib.sha256(parent).hexdigest()==hashes[mid]
        saved=json.loads(gzip.decompress(parent))
        for arm in ['WIDE','BOOK','BTC']:
            metrics,fills=one(path,cache,old,edges,prob,arm)
            if arm=='WIDE':
                assert [list(f) for f in fills]==saved['fills']
                assert abs(metrics['net_usd']*100-saved['value'])<1e-6
                assert abs(metrics['both']-saved['metrics']['both_grid_fraction'])<1e-10
            rows.append(dict(day=day,mid=mid,missing_btc_ticks=int((~np.isfinite(prob)).sum()),ticks=ticks,**metrics))
        print(mid, 'parity ok',flush=True)
    out=pd.DataFrame(rows)
    out.to_csv(H/('extension.csv' if extend else 'pilot.csv'),index=False)
    print(out.groupby('arm')[['net_usd','turnover_usd','residual_usd','adjusted','passive_sh']].sum().to_string())

if __name__=='__main__':
    main('--extend' in sys.argv)
