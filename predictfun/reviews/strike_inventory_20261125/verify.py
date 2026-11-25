"""Independent risk-room checks and baseline metric reconciliation."""
from pathlib import Path
import json
import math
import pandas as pd
from risk import allowed,risks,BUDGET

H=Path(__file__).resolve().parent

def check_room(choose):
    used=0
    for I in [-400,-200,-25,0,25,200,400]:
        for reserved in (0,20,100,200,400):
            for cost in (None,.01,.1,.35,.9):
                q=choose(100,I,reserved,cost)
                # Enumerate prospective fills, including older same-side reservations.
                valid=[n for n in range(101) if max(0,I+reserved+n)
                       <= (BUDGET/cost if cost is not None else 0)+1e-8]
                assert q==max(valid,default=0)
                used+=1
    return used

n=check_room(allowed)
try:
    check_room(lambda size,I,res,cost:allowed(size,I,0,cost))
except AssertionError:
    pass
else:
    raise RuntimeError('ignored reservation passed')
for b in [.01,.2,.5,.9,.99]:
    for q in [0,.1,.5,.9,1]:
        for side,r in enumerate(risks(b,q)):
            s=1-2*side
            loss_sq=sum(prob*max(0,s*(b-y))**2 for prob,y in [(1-q,0),(q,1)])
            assert math.isclose(r*r,loss_sq,abs_tol=1e-12)
f=json.loads((H/'forecast.json').read_text())
assert f['passed']==all(f['days'][d]['score_blend']<f['days'][d]['score_book'] for d in ('20261026','20261027'))
p=pd.read_csv(H/'pilot.csv')
assert len(p)==9 and not p.duplicated(['mid','arm']).any()
assert (abs(p.net_usd-p.turnover_usd-p.residual_usd)<1e-5).all()
r=pd.read_csv(H.parent/'inventory_exit_exception_20261125/per_market.csv').query("arm=='WIDE'")
x=p[p.arm=='WIDE'].merge(r,on='mid',validate='one_to_one')
assert len(x)==3 and (abs(x.net_c/100-x.net_usd)<1e-6).all()
print('passed',n,'room scenarios; ignored reservations rejected; squared loss identity; forecast gate; 9 replay decompositions; 3 baseline receipt comparisons')
