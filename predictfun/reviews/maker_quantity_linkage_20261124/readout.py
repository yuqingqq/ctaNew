"""Declared descriptive readout; signal join occurs only after physical bounds are fixed."""
import csv,gzip,json,statistics,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[2];O=H/'outputs'
def rows(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
b=rows(O/'chain_bounds.csv');orders={r['order_hash']:r for r in rows(O/'order_summary.csv')}
r={}
for tag,field,lo,hi in [('post_A','post_A_lo','old_post_lo','old_post_hi'),('post_B','post_B_lo','old_post_lo','old_post_hi'),('end','end_hi_conditional','old_end_lo','old_end_hi')]:
 valid=[x for x in b if x['bound_status']=='conditional_only' and (tag!='end' or x['group']=='conditional_withdrawal')]
 old=[(int(x[hi])-int(x[lo]))/1000 for x in valid];new=[((int(x[field])-int(x[lo])) if tag=='end' else (int(x[hi])-int(x[field])))/1000 for x in valid]
 r[tag]=dict(n=len(valid),n_narrowed=sum(a>z for a,z in zip(old,new)),old_median_s=statistics.median(old) if old else None,new_median_s=statistics.median(new) if new else None,max_reduction_s=max((a-z for a,z in zip(old,new)),default=None))
 bb={x['order_hash']:x for x in b};p=ROOT/'predictfun/reviews/maker_reversal_hypothesis_20261124/outputs/chronology.csv';ann=[]
for x in rows(p):
 y=bb[x['order_hash']];meta=json.loads((ROOT/f"data/predict_fun/research/maker_trace_v2_20261123/life/{x['day']}_{x['mid']}.json").read_text());rev=meta['ws_ms']+float(x['reversal_s'])*1000
 for method in ('post_A_lo','post_B_lo'):
  lo=int(y[method]);hi=int(y['post_hi_unchanged']);category='before_earliest_post' if rev<lo or (rev==lo and lo>int(y['old_post_lo'])) else ('after_latest_post' if rev>=hi else 'inside_post_window')
  ann.append(dict(order_hash=x['order_hash'],delay_ms=x['delay_ms'],method=method,old_chronology=x['chronology'],new_chronology=category,post_lo_ms=lo,post_hi_ms=hi,reversal_ms=int(rev),grade=y['bound_status'],still_conditional=True))
with (H/'chronology_annotation.csv').open('x') as f:
 w=csv.DictWriter(f,fieldnames=list(ann[0]));w.writeheader();w.writerows(ann)
r.update(chronology_rows=len(ann),chronology_changed=sum(x['old_chronology']!=x['new_chronology'] for x in ann),chronology_orders=len({x['order_hash'] for x in ann}),chronology_source_sha=hashlib.sha256(p.read_bytes()).hexdigest(),repeated_fill_orders=sum(int(x['maker_fills'])>1 for x in orders.values()))
with (H/'readout.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r,indent=2))
