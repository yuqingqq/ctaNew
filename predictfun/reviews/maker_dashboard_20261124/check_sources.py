"""Check projected market/maker rows against saved source fields independently."""
import csv,gzip,json,collections
from pathlib import Path
import pyarrow.parquet as pq
import extract_v2 as E
H=Path(__file__).resolve().parent
D=json.loads(gzip.decompress((H/'data.json.gz').read_bytes()));V,_=E.top_fills();vs={(r['mid'],r['sid'],int(r['mk_idx']),r['role'],r['addr']):r for r in V};post={r['order_hash']:r for r in E.read(E.R/'predictfun/reviews/maker_timing_estimates_20261124/outputs/post_estimates.csv.gz')};counts=collections.Counter()
def eq(a,b):return a is b if a is None or b is None else abs(a-b)<2e-5
def maker(m,f):
 candidates=[v for k,v in vs.items() if k[:4]==(m['id'],f['sid'],f['mk_idx'],f['role']) and v['hash']==f['hash']];assert len(candidates)==1;v=candidates[0]
 assert eq(f['price'],v['p']*100) and eq(f['q'],v['q']) and eq(f['fee'],v['fee_fill']) and f['side']==(0 if v['s']==1 else 1)
 for h in ['1s','10s','60s']:assert eq(f['markouts'][h],100*v['v_'+h]/v['q'] if v['st_'+h]=='ok' else None)
 p=post.get(f['hash']);expected=[(float(p[k])-m['ws'])/1000 for k in ['effect_lo_ms','effect_hi_ms']] if p and p['estimate_enabled']=='True' else None
 assert f['post'] is None if expected is None else all(eq(a,b) for a,b in zip(f['post'],expected))
for m in D['cases']:
 fr=pq.read_table(E.MT/'depth'/f"{m['day']}_{m['id']}_frames.parquet").to_pandas().sort_values(['upd_ms','recv_ns','fid']);lv=pq.read_table(E.MT/'depth'/f"{m['day']}_{m['id']}_levels.parquet",columns=['fid','side','ptick','shares']).to_pandas();groups={int(fid):g for fid,g in lv.groupby('fid')}
 for row in m['book'][::20]:
  t,b,a,*_=row;prev=fr[fr.upd_ms<=m['ws']+t*1000]
  if prev.empty:assert b is None and a is None;continue
  f=prev.iloc[-1];g=groups[int(f.fid)];bid=g[(g.side=='bid')&(g.shares>0)];ask=g[(g.side=='ask')&(g.shares>0)];bad=any(x in f.status for x in ['bad_frame','mid_mismatch','out_of_order','off_grid']);good=not bid.empty and not ask.empty and not bad and m['ws']+t*1000-f.upd_ms<=1000
  if good:good=bid.ptick.max()<=ask.ptick.min()
  assert (b==int(bid.ptick.max()) and a==int(ask.ptick.min())) if good else b is None and a is None
  counts['book_samples']+=1
 life={(r['sid'],int(r['mk_idx']),r['mk_hash']):r for r in pq.read_table(E.MT/'life'/f"{m['day']}_{m['id']}_fills.parquet").to_pylist()}
 for f in m['makers']:
  maker(m,f);counts['maker_legs']+=1
  if f['role']=='maker':
   l=life[(f['sid'],f['mk_idx'],f['hash'])];assert eq(f['lo'],(l['x_lo_hard_ms']-m['ws'])/1000) and eq(f['t'],(l['hi_ms']-m['ws'])/1000);counts['hard_windows']+=1
  else:assert f['lo'] is None
  if f['post']:counts['inferred_post_windows']+=1
 # Source-based falsifier at same identity, changed horizon value.
 f=next(x for x in m['makers'] if x['markouts']['1s'] is not None);bad=dict(f,markouts=dict(f['markouts'],**{'1s':f['markouts']['1s']+1}))
 try:maker(m,bad)
 except AssertionError:counts['falsifiers']+=1
 else:raise RuntimeError('source value corruption escaped')
print(dict(counts));(H/'source_checks.json').write_text(json.dumps(dict(passed=True,counts=dict(counts)),indent=2)+'\n')
