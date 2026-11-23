"""Synthetic checks only. No tape replay, labels, fitting or development edits."""
from pathlib import Path
import sys,json,time,signal
PIN=Path('/tmp/pf-mt-review-1af85c56');sys.path.insert(0,str(PIN/'predictfun/research'))
import pf_mt_lifecycle as L
import pf_mt_lifecycle_selftest as S
import pf_mt_econ_selftest as E
start=time.process_time();signal.setitimer(signal.ITIMER_PROF,30)
S.main(L)
E.main()
m=S.Mk(99001);m.rec(3000,[('h1','a','bid',50,5)])
m.frames([0,1000,2000,3000,4000],lambda t:{('bid',50):10})
legs,records,frames,levels=m.dfs()
def context(legs,records):
    LL,place,sub,xf=L._prep(legs,records,m.mid)
    B=L._book(frames,levels,m.mid,m.ws*1000,m.we*1000)
    return L._timing(LL,records,B,place,sub)
good=context(legs,records)
assert int(good['lo_h'][0])==S.T0+1000 and not good['rcpt'][0]
legs.first_recv_ns=(S.T0+2500)*10**6;records.first_recv_ns=(S.T0+2500)*10**6
bad=context(legs,records)
assert int(bad['hi'][0])==S.T0+2500
assert int(bad['lo_h'][0])!=-1 and not bad['rcpt'][0]
R={'target':'1af85c5691d5ca4c2f9565d94a13cd9680688d3c',
   'synthetic_receipt_before_UUID':{'hi':int(bad['hi'][0]),'actual_lo_h':int(bad['lo_h'][0]),'contract_lo_h':-1,'actual_receipt_hi':bool(bad['rcpt'][0])},
   'current_artifact_scope':'Saved fill_hi_rule passes; this is a future-input defect, not demonstrated on the measured rows.',
   'lifecycle_selftest':'PASS','econ_selftest':'PASS','cpu_s':time.process_time()-start}
Path(__file__).with_suffix('.json').write_text(json.dumps(R,indent=2)+'\n');print(json.dumps(R,indent=2))
