"""Saved-evidence inspection of all maker3 orders filled in [0,60), 3789484.
No private quotes synthesized; no model/null/validation. Reuse previously
graded post windows and original-size evidence. Context is the RANGE across
feasible posting times, never a chosen midpoint or an asserted decision state.
"""
import collections,csv,gzip,hashlib,json,math
from pathlib import Path
import pyarrow.parquet as pq
H=Path(__file__).resolve().parent;R=H.parents[2]
MT=R/'data/predict_fun/research/maker_trace_v2_20261123'
def read(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
M=next(x for x in json.loads(gzip.decompress((H/'data.json.gz').read_bytes()))['cases'] if x['id']==3789484)
WS=M['ws'];posts={r['order_hash']:r for r in read(R/'predictfun/reviews/maker_timing_estimates_20261124/outputs/post_estimates.csv.gz')}
signed={r['order_hash']:r for r in read(R/'predictfun/reviews/maker_action_population_20261124/outputs/orders.csv')}
allf=[f for f in M['makers'] if f['rank']==3 and f['role']=='maker'];wanted={f['hash'] for f in allf if 0<=f['t']<60}
fr=pq.read_table(MT/'depth/20261026_3789484_frames.parquet').to_pandas().sort_values(['upd_ms','recv_ns','fid'])
lv=pq.read_table(MT/'depth/20261026_3789484_levels.parquet',columns=['fid','side','ptick','shares','status']).to_pandas()
lv=lv[(lv.shares>0)&(lv.status=='ok')]
bid=lv[lv.side=='bid'].groupby('fid').ptick.max().to_dict();ask=lv[lv.side=='ask'].groupby('fid').ptick.min().to_dict()
book=[]
for f in fr.to_dict('records'):
 b,a=bid.get(f['fid']),ask.get(f['fid']);ok=f['status']=='ok' and b is not None and a is not None and b<=a
 book.append(dict(t=(f['upd_ms']-WS)/1000,bid=b if ok else None,ask=a if ok else None,receipt_t=(f['recv_ns']/1e6-WS)/1000,status=f['status'],fid=f['fid']))
out=[]
for h in wanted:
 fs=[f for f in allf if f['hash']==h];early=[f for f in fs if 0<=f['t']<60];p=posts[h];s=signed.get(h);pwin=early[0]['post']
 assert (int(p['mid']),int(p['rank']),int(p['tick']))==(3789484,3,int(early[0]['price']))
 assert (p['estimate_enabled']=='True')==(pwin is not None)
 r=dict(hash=h,side='bid' if early[0]['side']==0 else 'ask',price_c=early[0]['price'],filled_first60_sh=sum(f['q'] for f in early),filled_displayed_market_sh=sum(f['q'] for f in fs),n_fills_first60=len(early),first_upper_s=early[0]['t'],last_upper_s=early[-1]['t'],post_window_s=pwin,post_grade=p['grade'],original_shares=float(s['original_shares']) if s else None,end_kind=s['end_kind'] if s else 'unknown',known_private_cancel=False,first60_fills=early)
 # A non-empty inherited certain-presence span still needs lifecycle/no-reentry
 # assumptions; do not call it an exact order lifetime or queue position.
 last_lo=max((f['lo'] for f in early if f['lo'] is not None),default=None)
 r['conditional_presence_span_s']=[early[0]['t'],last_lo] if last_lo is not None and last_lo>=early[0]['t'] else None
 if pwin:
  left=[f for f in book if f['t']<=pwin[0]]
  ctx=(left[-1:] if left else [])+[f for f in book if pwin[0]<f['t']<=pwin[1]]
  good=[f for f in ctx if f['bid'] is not None and f['ask'] is not None];offset=[f['bid']-r['price_c'] if r['side']=='bid' else r['price_c']-f['ask'] for f in good]
  r['possible_post_book']=dict(n_frames=len(ctx),invalid_frames=len(ctx)-len(good),best_bid_range=[min(f['bid'] for f in good),max(f['bid'] for f in good)] if good else None,best_ask_range=[min(f['ask'] for f in good),max(f['ask'] for f in good)] if good else None,ticks_behind_same_side_best_range=[min(offset),max(offset)] if offset else None,classes=sorted({'behind' if o>0 else 'at_touch' if o==0 else 'improves_same_side' for o in offset}),clock='historical native updates; not identified client decision information')
 else:r['possible_post_book']=None
 out.append(r)
out.sort(key=lambda r:r['first_upper_s'])
allquotes=[];later_status=collections.Counter()
for h in {f['hash'] for f in allf}-wanted:
 p=posts.get(h)
 if not p or p['estimate_enabled']!='True':later_status['posting_refused_or_missing']+=1
 elif float(p['effect_lo_ms'])>WS+60000:later_status['enabled_post_window_after60']+=1
 else:later_status['enabled_possible_post_before60']+=1
 if p and p['estimate_enabled']=='True' and float(p['effect_lo_ms'])<=WS+60000:
  fs=[f for f in allf if f['hash']==h]
  allquotes.append(dict(hash=h,price_c=fs[0]['price'],side=fs[0]['side'],first_upper_s=fs[0]['t'],post_window_s=fs[0]['post'],post_grade=fs[0]['post_grade']))
result=dict(market=3789484,rank=3,scope='all observed maker fills whose upper bound is in [0,60)',rows=out,possible_early_orders_first_filled_after60=allquotes,later_order_posting_status=dict(later_status),
 counts=dict(orders=len(out),fills=sum(r['n_fills_first60'] for r in out),signed_original_sizes=sum(r['original_shares'] is not None for r in out),grade=dict(collections.Counter(r['post_grade'] for r in out)),two_or_more_fills=sum(r['n_fills_first60']>1 for r in out),later_orders_with_possible_post_before60=len(allquotes)))
assert result['counts']['fills']==51
(H/'maker3_quotes.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
print(json.dumps(clean(dict(counts=result['counts'],quotes=[{k:v for k,v in r.items() if k!='first60_fills'} for r in out])),indent=2))
