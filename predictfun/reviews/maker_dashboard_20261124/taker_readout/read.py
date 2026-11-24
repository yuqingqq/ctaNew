"""Descriptive saved rows only. All three dashboard markets, every rank1-3.
Take group = rank+sid (all maker-counterparty legs combined). Inventory is
strict-earlier settlement-clock zero-start net flow, NOT client-known state.
No policy inference, causal attribution, counterfactual removal or fitting.
"""
import collections,gzip,json,csv
from pathlib import Path
H=Path(__file__).resolve().parent;D=json.loads(gzip.decompress((H.parent/'data.json.gz').read_bytes()));out=[]
for m in D['cases']:
 for rank in [1,2,3]:
  fs=[f for f in m['makers'] if f['rank']==rank];takes=[f for f in fs if f['role']=='taker'];makes=[f for f in fs if f['role']=='maker'];groups=collections.defaultdict(list)
  for f in takes:groups[f['sid']].append(f)
  g=[]
  for sid,gg in groups.items():
   ts={f['t'] for f in gg};assert len(ts)==1;t=gg[0]['t'];i=sum((1-2*f['side'])*f['q'] for f in fs if f['t']<t);delta=sum((1-2*f['side'])*f['q'] for f in gg);q=sum(f['q'] for f in gg)
   g.append(dict(sid=sid,t=t,q=q,before=i,net_delta=delta,opposes=i*delta<0,abs_reduces=abs(i+delta)<abs(i)-1e-6,mixed_sides=len({f['side'] for f in gg})>1,self_trade=any(f['self_trade'] for f in gg)))
  total=sum(f['q'] for f in fs);tq=sum(f['q'] for f in takes)
  assert abs(sum(f['q'] for f in makes)+tq-total)<1e-8
  out.append(dict(m=m['id'],rank=rank,all_legs=len(fs),maker_legs=len(makes),taker_legs=len(takes),take_groups=len(g),total_shares=total,taker_shares=tq,taker_share=tq/total if total else None,taker_fees=sum(f['fee'] for f in takes),group_counts={k:sum(x[k] for x in g) for k in ['opposes','abs_reduces','mixed_sides','self_trade']},groups=g))
(H/'readout.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([{k:v for k,v in x.items() if k!='groups'} for x in out]))
