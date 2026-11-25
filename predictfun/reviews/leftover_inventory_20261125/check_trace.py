"""Independently rebuild known inventory from fill notices on the charted market."""
import json
from collections import Counter
import numpy as np
import replay as R

Base=R.Branch


class Enumerated(Base):
    def known_filled(self,gid,g):
        return R.LB.LadderBranch.known_filled(self,gid,g)

    def reserve(self,g):
        return R.LB.LadderBranch.reserve(self,g)

    def flow(self,s,g):
        return sum(q for t,i,side,q,p,it in self.E.fills if side==s and g-30000<t+R.BE.FILL_NOTICE<=g)


def trace():
    m,day=3789484,R.EVAL
    reg=R.MM.registry(day).set_index('mid')
    src=R.MM.KP.Tape()
    _,st,p,info=next(R.MM.worlds(day,[m],src,src.offset(day),reg))
    assert st=='complete'
    ws=float(reg.loc[m,'ws'])*1000
    rg=R.pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
    ident=R.MM.ident_of(day,m,ws,info,p)
    obs=R.MM.make_cache(info['t_start'],info['end_rel'])
    flags=R.triggers(obs,R.MM.TR.T.P.frozen_edges())
    btc=R.BTC.load_btc(R.X.DATA/'maker_trace_v2_20261123/btc',day,Counter())
    t=info['t_start']+R.BE.STEP*np.arange(len(obs['x'][0]))
    z=R.BTC.btc_feats(btc,np.floor(t-1000).astype(np.int64),np.full(len(t),int(ws)),np.full(len(t),int(ws)+300000))['z']
    model=json.loads((R.HERE/'fit.json').read_text())['cells']
    stored=json.loads((R.OUT/'eval'/f'{day}_{m}.json').read_text())
    finish=R.LB.finish
    captures=[]
    def capture(b):
        out=finish(b)
        orders=b.E.o
        assert all(o['size']*(o['price'] if o['side']==0 else 100-o['price'])>=100-1e-8 for o in orders)
        assert all(o['act']==o['submit']+58 and o['size']<=220 for o in orders)
        for o in orders:
            r=o['req']
            if r is not None:
                assert o['remove']==(max(r+37,o['submit']+95) if r<o['act'] else r+37)
        captures.append([b.E.fills,[tuple(o[k] for k in ('side','submit','act','req','remove','price','size','refused')) for o in orders]])
        return out
    R.LB.finish=capture
    R.Branch=Enumerated
    try:
        for a in R.ARMS:
            x=R.one(p,info,ident,flags,a,model,z,R.VM.V.payout_of(rg.loc[m]))
            y=stored['arms'][a]
            for k in ('anchors','proposals'):
                assert len(x[k])==len(y[k])
                if k=='proposals': assert np.allclose(x[k],y[k],atol=1e-7,rtol=0)
                else: assert json.loads(json.dumps(x[k]))==y[k]
            for k,val in x['metrics'].items():
                assert abs(val-y['metrics'][k])<1e-7
            if a=='C':
                R.Branch=Base
                R.one(p,info,ident,flags,a,model,z,R.VM.V.payout_of(rg.loc[m]))
                assert captures[-1]==captures[-2]
    finally:
        R.Branch=Base
        R.LB.finish=finish
    R.write(R.HERE/'trace_check.json',dict(mid=m,arms=list(R.ARMS),all_proposals_and_metrics_equal=True,
                                         c_fills_and_orders_exact=True,profile_checked=True))
    print('Uncached notice enumeration agrees; C fills and order lifecycles match exactly.')


if __name__=='__main__':
    trace()
