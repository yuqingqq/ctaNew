"""Descriptive complete hash/generation sequence and independent accounting.
All first-minute rows, not a selected profitable-pair sample. Direct cash flow
must agree with FIFO closed value plus the remaining inventory's marked value.
"""
import collections,gzip,json
from pathlib import Path
H=Path(__file__).resolve().parent
D=json.loads((H/'first_minute_turnover.json').read_text())
M=next(x for x in json.loads(gzip.decompress((H/'data.json.gz').read_bytes()))['cases'] if x['id']==3789484)
out={}
for name,x in D.items():
 groups=collections.defaultdict(list)
 for f in x['fills']:groups[f.get('hash',f.get('gid'))].append(f)
 seq=[]
 for key,fs in groups.items():
  assert len({(f['side'],f['price']) for f in fs})==1
  seq.append(dict(key=key,side=fs[0]['side'],price=fs[0]['price'],q=sum(f['q'] for f in fs),first_s=min(f['t'] for f in fs),last_s=max(f['t'] for f in fs),fills=len(fs),post=fs[0].get('post'),grade=fs[0].get('post_grade'),first_hard_lo=fs[0].get('lo')))
 seq.sort(key=lambda x:x['first_s'])
 cash=sum(-(1-2*f['side'])*f['q']*f['price']/100-f.get('fee',0) for f in x['fills'])
 pos=sum((1-2*f['side'])*f['q'] for f in x['fills'])
 mark=x['total']['mark60_c'];direct=cash+pos*mark/100
 assert abs(direct-x['total']['total_marked_usd'])<1e-7
 assert abs(x['total']['fifo_gross_usd']+x['total']['residual_usd']-x['total']['fees_usd']-direct)<1e-7
 # Exact projection-to-projection ledger identity and complete window membership.
 assert sum(r['legs'] for r in x['windows'])==len(x['fills'])
 for side,lab in [(0,'buy'),(1,'sell')]:
  assert abs(sum(r[lab]['shares'] for r in x['windows'])-sum(f['q'] for f in x['fills'] if f['side']==side))<1e-7
 # Refuse corrupted claimed accounting totals, not only successful arithmetic.
 for wrong in [x['total']['fifo_gross_usd']+1,x['total']['fifo_gross_usd']-1]:
  assert abs(wrong+x['total']['residual_usd']-x['total']['fees_usd']-direct)>0.99
 good=sum(c['q'] for c in x['closes'] if c['gross_c']>0);bad=sum(c['q'] for c in x['closes'] if c['gross_c']<0)
 out[name]=dict(order_sequence=seq,profitable_closed_shares=good,losing_closed_shares=bad,good_closed_gross_usd=sum(c['gross_c'] for c in x['closes'] if c['gross_c']>0)/100,bad_closed_gross_usd=sum(c['gross_c'] for c in x['closes'] if c['gross_c']<0)/100,all_rows_accounted=True,closing_quantities=good+bad)
 print(name,json.dumps(out[name],indent=2))
(H/'first_minute_sequences.json').write_text(json.dumps(out,indent=2)+'\n')
