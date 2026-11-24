"""Independent checks of saved clocks and economics, not private queue tenure."""
import csv,gzip,json,math,hashlib,collections
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[2];C=R/'predictfun/reviews/maker_trace_20261122/v2/chains'
def read(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
def eq(a,b):
 if isinstance(b,bool):assert str(a)==str(b),(a,b)
 elif isinstance(b,(int,float)):assert math.isclose(float(a),b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
 else:assert str(a)==str(b),(a,b)
def verify(rows,orders,ps,es,source):
 key=lambda r:tuple(r[k] for k in ['day','mid','sid','mk_idx'])+(r.get('mk_hash',r.get('order_hash')),);groups=collections.defaultdict(list)
 for f in source:groups[key(f)].append(f)
 assert len(rows)==len(groups) and {key(r) for r in rows}==set(groups);assert len(orders)==len(ps) and {r['order_hash'] for r in orders}==set(ps)
 for r in rows:
  fs=groups[key(r)];f=fs[0];p=ps.get(r['order_hash']);e=es.get(r['order_hash']);eq(r['source_occurrences'],len(fs));eq(r['creation_context'],'|'.join(sorted({f['creation_context'] for f in fs})));eq(r['private_queue_continuity'],'False')
  for k in ['side','tick','q','q_valid','valid','excl_reason']:eq(r[k],f[k]);assert all(x[k]==f[k] for x in fs)
  for h in (1,10,60):eq(r[f'gross{h}_usd'],f[f'v_{h}s']);assert all(x[f'v_{h}s']==f[f'v_{h}s'] for x in fs)
  for k in ['lo_ms','hi_ms','addr']:assert all(x[k]==f[k] for x in fs)
  eq(r['good_fill'],f['valid']=='True' and float(f['v_10s'])>0 and float(f['v_60s'])>0);eq(r['grade'],e['grade'] if e else 'missing');eq(r['rank'],p['rank'] if p else 'other')
  if p:
   assert (f['day'],f['mid'],f['side'],int(f['tick']),f['addr'].lower())==(p['day'],p['mid'],p['y_side'],int(p['tick']),p['addr'].lower());eq(r['record_class'],'same_first_record_time' if int(f['hi_ms'])==int(p['first_hi_ms']) else 'later_record_time')
  else:eq(r['record_class'],'unknown')
  status='outside_top3_post_census' if p is None else 'refused_post' if e['estimate_enabled']!='True' or e['grade'] not in ['G2_bound_constrained','G4_scenario_only'] else 'ok'
  if status=='ok':
   a,b,l,u=map(float,[e['effect_lo_ms'],e['effect_hi_ms'],f['lo_ms'],f['hi_ms']]);status='invalid_clock' if not all(math.isfinite(v) for v in [a,b,l,u]) or a<0 or b<a or u<0 or (l>=0 and l>u) else 'chronology_conflict' if u<a else 'ok'
   if status=='ok':
    lo=max(l-b,0)/1000 if l>=0 else 0.;hi=(u-a)/1000;eq(r['age_min_s'],lo);eq(r['age_max_s'],hi);label='upper_le1' if hi<=1 else 'whole_1_5' if lo>1 and hi<=5 else 'whole_5_10' if lo>5 and hi<=10 else 'whole_10_30' if lo>10 and hi<=30 else 'whole_gt30' if lo>30 else 'straddles';eq(r['age_band'],label)
  if status!='ok':eq(r['age_min_s'],'');eq(r['age_max_s'],'');eq(r['age_band'],'unavailable')
  eq(r['timing_status'],status)
 for r in orders:
  p=ps[r['order_hash']];e=es[r['order_hash']];ok=e['estimate_enabled']=='True' and e['grade'] in ['G2_bound_constrained','G4_scenario_only'];eq(r['timing_status'],'ok' if ok else 'refused_post');eq(r['grade'],e['grade']);eq(r['first_age_min_s'],0 if ok else '');eq(r['first_age_max_s'],(int(e['first_record_ms'])-int(e['effect_lo_ms']))/1000 if ok else '')
  for k in ['day','mid','rank','tick']:eq(r[k],p[k])
  eq(r['side'],p['y_side']);eq(r['pnl_fills'],p['pnl_rows']);eq(r['pnl_shares'],p['pnl_shares']);eq(r['gross10_usd'],p['v_10s']);eq(r['gross60_usd'],p['v_60s']);valid=float(p['pnl_shares'])>0 and math.isfinite(float(p['v_10s'])+float(p['v_60s']));eq(r['economics_status'],'counted_fills_available' if valid else 'unavailable');eq(r['good_order'],valid and float(p['v_10s'])>0 and float(p['v_60s'])>0)
def main():
 for path,sha in json.loads((H/'manifest.json').read_text())['files'].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==sha,path
 ps={r['mk_hash']:r for r in read(C/'post_fill/orders.csv')};es={r['order_hash']:r for r in read(R/'predictfun/reviews/maker_timing_estimates_20261124/outputs/post_estimates.csv.gz')};source=read(C/'queue_front_v2/fills.csv');rows=read(H/'fills.csv.gz');orders=read(H/'orders.csv.gz');verify(rows,orders,ps,es,source)
 for cell in read(H/'cells.csv'):
  g=[r for r in rows if r['grade']==cell['grade'] and r['timing_status']==cell['timing_status'] and (cell['dimension']=='all' or r[cell['dimension']]==cell['cell'])];v=[r for r in g if r['valid']=='True'];good=[r for r in v if r['good_fill']=='True'];q=sum(float(r['q_valid']) for r in v)
  for k,val in dict(fills=len(g),orders=len({r['order_hash'] for r in g}),valid_fills=len(v),valid_shares=q,good_fills=len(good),good_shares=sum(float(r['q_valid']) for r in good)).items():eq(cell[k],val)
  for h in (1,10,60):
   x=[float(r[f'gross{h}_usd']) for r in v]
   for k,val in {f'gross{h}_usd':sum(x),f'c_per_share_{h}':100*sum(x)/q if q else '',f'positive_{h}':sum(z>0 for z in x),f'negative_{h}':sum(z<0 for z in x),f'zero_{h}':sum(z==0 for z in x)}.items():eq(cell[k],val)
  for t in (1,5,10,30):eq(cell[f'good_min_ge{t}'],sum(r['timing_status']=='ok' and float(r['age_min_s'])>=t for r in good));eq(cell[f'good_max_ge{t}'],sum(r['timing_status']=='ok' and float(r['age_max_s'])>=t for r in good))
 rng=np.random.default_rng(20261124);draws=read(H/'draws.csv');assert len(draws)==1000
 for m in json.loads((H/'matched.json').read_text()):
  groups=collections.defaultdict(list)
  for r in rows:
   if r['grade']!=m['grade'] or r['timing_status']!='ok' or r['valid']!='True' or '|' in r['creation_context']:continue
   early=float(r['age_min_s'])>=5;recent=float(r['age_max_s'])<=1
   if early or recent:groups[tuple(r[k] for k in ['day','mid','rank','side','tick','record_class','creation_context'])].append((early,r['good_fill']=='True'))
  den=obs=0.;null=np.zeros(500);n1=n0=ng=0;markets=set()
  for k,items in sorted(groups.items()):
   a=[y for e,y in items if e];b=[y for e,y in items if not e]
   if not a or not b:continue
   n,z=len(a),len(b);Y=sum(a)+sum(b);w=1/(1/n+1/z);obs+=w*(sum(a)/n-sum(b)/z);den+=w;draw=rng.hypergeometric(Y,n+z-Y,n,500);null+=w*(draw/n-(Y-draw)/z);n1+=n;n0+=z;ng+=1;markets.add(k[:2])
  eq(m['early'],n1);eq(m['recent'],n0);eq(m['common_strata'],ng);eq(m['common_markets'],len(markets));eq(m['verdict'],'descriptive_only' if n1>=20 and n0>=20 and len(markets)>=5 else 'insufficient')
  if den:
   null/=den;rd=obs/den;eq(m['matched_difference'],rd);eq(m['p_exploratory'],(1+sum(abs(x)>=abs(rd)-1e-15 for x in null))/501)
  else:assert m['matched_difference'] is None and m['p_exploratory'] is None
  ds=[r for r in draws if r['grade']==m['grade']];assert len(ds)==500
  for i,r in enumerate(ds):eq(r['draw'],i);eq(r['value'],float(null[i]));eq(r['evaluated'],bool(den))
 caught=[]
 for field,value in [('age_max_s','999999'),('gross10_usd','999999'),('good_fill','corrupt'),('private_queue_continuity','True'),('grade','invented')]:
  r=next(r for r in rows if r['timing_status']=='ok');old=r[field];r[field]=value
  try:verify(rows,orders,ps,es,source)
  except AssertionError:caught.append(field)
  else:raise AssertionError('escaped '+field)
  finally:r[field]=old
 duplicate=rows+[rows[0]]
 try:verify(duplicate,orders,ps,es,source)
 except AssertionError:caught.append('duplicate')
 else:raise AssertionError('duplicate escaped')
 result=json.loads((H/'result.json').read_text());assert result['orders']==len(orders) and result['source_fill_rows']==len(source) and result['unique_fills']==len(rows) and result['duplicates']==len(source)-len(rows);assert result['status']==dict(collections.Counter(r['timing_status'] for r in rows));assert result['matched']==json.loads((H/'matched.json').read_text())
 (H/'checks.json').write_text(json.dumps(dict(pass_all=True,falsifiers=caught,source_fills=len(source),orders=len(orders),unique_fills=len(rows),scope='Exact accepted-artifact recount; not raw tape or private commands.'),indent=2)+'\n');print('independent timing/value/cells/matching checks pass; six corruptions refused')
if __name__=='__main__':main()
