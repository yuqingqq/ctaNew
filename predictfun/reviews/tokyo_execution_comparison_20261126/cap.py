"""Closed-loop whole-order risk-veto sensitivity, separate from the live-order comparison."""
from unittest.mock import patch
import json

import numpy as np
import pandas as pd

from compare import HERE, LIB, R, U

DAY='20261026'


class Veto(U.S.Smooth):
    blocked=0

    def slot(self,v,rung,price):
        state=self.opened,self.first_side
        action,q,why=super().slot(v,rung,price)
        signed=(1-2*v['side'])*v['position']
        reserved=v['A'] if v['side'] else v['B']
        if action=='POST' and signed+reserved+q>150+R.EPS:
            self.opened,self.first_side=state
            self.proposals.pop()
            self.blocked+=1
            return 'WAIT',None,'tokyo_cap'
        return action,q,why


def run():
    S=LIB['S']
    ref=pd.read_csv(HERE.parent/'urgency_size50_20261126'/f'markets_{DAY}.csv',float_precision='round_trip').query('size==50').set_index('mid',verify_integrity=True)
    raw=U.S.read(R.CENSUS)['rows']
    bound={r['mid']:r['tape_id'] for r in raw if r['day']==DAY and r['status']=='complete'}
    assert set(bound)==set(ref.index)
    fit=U.S.read(R.HERE/'fit_v2.json')['cells']
    assert R.sha(R.HERE/'fit_v2.json')==U.S.read(S['L']['E'].REPLICATION/'result.json')['fit_sha']
    reg=R.MM.registry(DAY).set_index('mid')
    payouts=pd.read_parquet(R.MM.S1/DAY/'registry.parquet').set_index('mid')
    tape=R.MM.KP.Tape()
    off=tape.offset(DAY)
    edges=R.MM.TR.T.P.frozen_edges()
    rows=[]
    for _,group in reg.groupby('hour'):
        batch=sorted(set(group.index)&set(bound))
        if not batch: continue
        for m,status,path,info in R.MM.worlds(DAY,batch,tape,off,reg):
            ident=R.MM.ident_of(DAY,m,reg.at[m,'ws']*1000,info,path)
            assert status=='complete' and ident['tape_id']==bound[m]
            flags=R.triggers(R.MM.make_cache(info['t_start'],info['end_rel']),edges)
            payout=R.VM.V.payout_of(payouts.loc[m])
            if m==min(bound):
                LIB['control'](LIB['replay'](50,path,info,ident,flags,fit,payout),ref.loc[m])
            branch=U.UrgencyBranch(path,info['end_rel'],ident)
            branch.q0=50
            policy=Veto(branch,flags,fit,'linear',{},[])
            with patch.object(R,'max_new',S['size_bound'](50)):
                for k in range(R.LB.n_ticks(branch)): R.LB.step(branch,k,policy)
                R.LB.finish(branch)
            fills=branch.E.fills
            row=LIB['ledger'](fills,100*payout)
            pos=np.r_[0.,np.cumsum([(1-2*s)*q for _,_,s,q,_,_ in fills])]
            assert max(abs(pos))<=150+1e-6 and len(policy.proposals)==len(branch.E.o)
            _,end,val=branch.E.value(100*payout)
            assert abs(val/100-row['turnover']-row['residual'])<1e-6
            row.update(mid=m,cap=150,abs_end=abs(end),settle=val/100,posts=len(branch.E.o),
                       blocked_slots=policy.blocked,peak_position=max(abs(pos)),
                       age_share_seconds=sum((b[0]-a[0])*q/1000 for a,b,q in U.C.inventory_pairs(fills)))
            rows.extend([row,dict(ref.loc[m],mid=m,cap=1000)])
    df=pd.DataFrame(rows)
    assert len(df)==2*len(bound) and not df.duplicated(['cap','mid']).any()
    df.to_csv(HERE/'cap_markets.csv',index=False)
    print(df.groupby('cap')[['settle','turnover','residual','closed','abs_end']].sum())


if __name__=='__main__':
    run()
