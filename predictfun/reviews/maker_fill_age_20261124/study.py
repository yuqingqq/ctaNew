import csv,gzip,json,math,hashlib,collections,sys,time,importlib.util
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[2];C=R/'predictfun/reviews/maker_trace_20261122/v2/chains';T=R/'predictfun/reviews/maker_timing_estimates_20261124/outputs/post_estimates.csv.gz'
def read(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
def write(p,rows):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'wt') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def num(x):
 try:return float(x)
 except (ValueError,TypeError):return math.nan
def interval(plo,phi,flo,fhi):
 if not all(math.isfinite(x) for x in (plo,phi,flo,fhi)) or plo<0 or phi<plo or fhi<0 or (flo>=0 and flo>fhi):return 'invalid_clock','',''
 if fhi<plo:return 'chronology_conflict','',''
 return 'ok',max(0,flo-phi)/1000 if flo>=0 else 0.,(fhi-plo)/1000
def ageband(lo,hi):
 if hi<=1:return 'upper_le1'
 for a,b,label in [(1,5,'whole_1_5'),(5,10,'whole_5_10'),(10,30,'whole_10_30')]:
  if lo>a and hi<=b:return label
 return 'whole_gt30' if lo>30 else 'straddles'
def sources():return read(C/'post_fill/orders.csv'),read(T),read(C/'queue_front_v2/fills.csv')
def build():
 orders,estimates,fills=sources();ps={r['mk_hash']:r for r in orders};es={r['order_hash']:r for r in estimates};assert len(ps)==len(orders)==len(es);os=[]
 for h,p in sorted(ps.items()):
  e=es[h];assert int(e['first_record_ms'])==int(p['first_hi_ms'])==int(p['post_hi_ms']);status='ok' if e['estimate_enabled']=='True' and e['grade'] in ('G2_bound_constrained','G4_scenario_only') else 'refused_post';hi=(num(e['first_record_ms'])-num(e['effect_lo_ms']))/1000 if status=='ok' else '';q=num(p['pnl_shares']);v10=num(p['v_10s']);v60=num(p['v_60s']);eligible=q>0 and math.isfinite(v10+v60)
  os.append(dict(order_hash=h,day=p['day'],mid=p['mid'],rank=p['rank'],side=p['y_side'],tick=p['tick'],grade=e['grade'],timing_status=status,first_age_min_s=0 if status=='ok' else '',first_age_max_s=hi,pnl_fills=p['pnl_rows'],pnl_shares=p['pnl_shares'],gross10_usd=p['v_10s'],gross60_usd=p['v_60s'],economics_status='counted_fills_available' if eligible else 'unavailable',good_order=eligible and v10>0 and v60>0))
 unique={};fields=['day','mid','sid','mk_idx','mk_hash','addr','side','tick','q','lo_ms','hi_ms','valid','q_valid','excl_reason','v_1s','v_10s','v_60s'];mult=collections.Counter();contexts=collections.defaultdict(set)
 for f in fills:
  k=tuple(f[x] for x in ('day','mid','sid','mk_idx','mk_hash'));mult[k]+=1;contexts[k].add(f['creation_context'])
  if k in unique:assert all(f[x]==unique[k][x] for x in fields),'duplicate conflict'
  else:unique[k]=f
 rows=[]
 for k,f in sorted(unique.items()):
  h=f['mk_hash'];p=ps.get(h);e=es.get(h);status='outside_top3_post_census' if p is None else 'ok';grade=e['grade'] if e else 'missing';lo=hi=''
  if p:
   assert (f['day'],f['mid'],f['side'],f['tick'],f['addr'].lower())==(p['day'],p['mid'],p['y_side'],p['tick'],p['addr'].lower()),'fill/order identity'
   if e['estimate_enabled']!='True' or grade not in ('G2_bound_constrained','G4_scenario_only'):status='refused_post'
   else:status,lo,hi=interval(num(e['effect_lo_ms']),num(e['effect_hi_ms']),num(f['lo_ms']),num(f['hi_ms']))
  valid=f['valid']=='True';q=num(f['q_valid']);assert (valid and q>0) or (not valid and q==0);vals=[num(f['v_'+t]) for t in ['1s','10s','60s']];assert not valid or all(math.isfinite(v) for v in vals)
  rows.append(dict(day=f['day'],mid=f['mid'],sid=f['sid'],mk_idx=f['mk_idx'],order_hash=h,rank=p['rank'] if p else 'other',side=f['side'],tick=f['tick'],source_occurrences=mult[k],creation_context='|'.join(sorted(contexts[k])),grade=grade,timing_status=status,age_min_s=lo,age_max_s=hi,age_band=ageband(lo,hi) if status=='ok' else 'unavailable',record_class=('same_first_record_time' if f['hi_ms']==p['first_hi_ms'] else 'later_record_time') if p else 'unknown',valid=valid,excl_reason=f['excl_reason'],q=f['q'],q_valid=f['q_valid'],gross1_usd=f['v_1s'],gross10_usd=f['v_10s'],gross60_usd=f['v_60s'],good_fill=valid and vals[1]>0 and vals[2]>0,private_queue_continuity=False))
 return os,rows,len(fills)
def aggregate(rows):
 groups=collections.defaultdict(list)
 for r in rows:
  for dim,value in [('all','all'),('age_band',r['age_band']),('record_class',r['record_class']),('day',r['day'])]:groups[(r['grade'],r['timing_status'],dim,value)].append(r)
 out=[]
 for k,g in sorted(groups.items()):
  v=[r for r in g if r['valid']];q=sum(num(r['q_valid']) for r in v);good=[r for r in v if r['good_fill']];r=dict(grade=k[0],timing_status=k[1],dimension=k[2],cell=k[3],fills=len(g),orders=len({r['order_hash'] for r in g}),valid_fills=len(v),valid_shares=q,good_fills=len(good),good_shares=sum(num(r['q_valid']) for r in good))
  for h in (1,10,60):
   vals=[num(x[f'gross{h}_usd']) for x in v];r.update({f'gross{h}_usd':sum(vals),f'c_per_share_{h}':100*sum(vals)/q if q else '',f'positive_{h}':sum(x>0 for x in vals),f'negative_{h}':sum(x<0 for x in vals),f'zero_{h}':sum(x==0 for x in vals)})
  for t in (1,5,10,30):r[f'good_min_ge{t}']=sum(x['timing_status']=='ok' and num(x['age_min_s'])>=t for x in good);r[f'good_max_ge{t}']=sum(x['timing_status']=='ok' and num(x['age_max_s'])>=t for x in good)
  out.append(r)
 return out
def compare(rows):
 out=[];draw=[];rng=np.random.default_rng(20261124)
 for grade in ('G2_bound_constrained','G4_scenario_only'):
  groups=collections.defaultdict(lambda:[[],[]])
  for r in rows:
   if r['grade']!=grade or r['timing_status']!='ok' or not r['valid'] or '|' in r['creation_context']:continue
   early=num(r['age_min_s'])>=5;recent=num(r['age_max_s'])<=1
   if not early and not recent:continue
   key=tuple(r[x] for x in ('day','mid','rank','side','tick','record_class','creation_context'));groups[key][0 if early else 1].append(int(r['good_fill']))
  den=obs=0.;null=np.zeros(500);n1=n0=0;markets=set();strata=0
  for k,(a,b) in sorted(groups.items()):
   if not a or not b:continue
   n,m=len(a),len(b);Y=sum(a)+sum(b);w=n*m/(n+m);den+=w;obs+=w*(sum(a)/n-sum(b)/m);x=rng.hypergeometric(Y,n+m-Y,n,500);null+=w*(x/n-(Y-x)/m);n1+=n;n0+=m;markets.add(k[:2]);strata+=1
  rd=obs/den if den else None;null=null/den if den else null;p=(1+int((abs(null)>=abs(rd)-1e-15).sum()))/501 if den else None
  out.append(dict(grade=grade,early=n1,recent=n0,common_markets=len(markets),common_strata=strata,matched_difference=rd,p_exploratory=p,verdict='descriptive_only' if n1>=20 and n0>=20 and len(markets)>=5 else 'insufficient'))
  for j,x in enumerate(null):draw.append(dict(grade=grade,draw=j,value=float(x),evaluated=bool(den)))
 return out,draw
def freeze():
 assert interval(100,200,300,400)==('ok',.1,.3);assert interval(100,200,150,250)==('ok',0.,.15);assert interval(300,400,100,200)[0]=='chronology_conflict';assert interval(100,200,-1,400)==('ok',0.,.3);assert ageband(0,.5)=='upper_le1' and ageband(5.1,9)=='whole_5_10' and ageband(0,40)=='straddles'
 spec=importlib.util.spec_from_file_location('cpu',R/'predictfun/reviews/maker_trace_20261122/tools/cpu_ledger.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);task=sum(x['cpu_s'] for x in m.ledger())/3600+m.EXTRA_H;assert task+.0008+.05<=2 and m.LINE_BEFORE+task+.0008+.05<=5.3
 paths=[H/'contract.md',H/'study.py',H/'check.py',C/'post_fill/orders.csv',T,C/'queue_front_v2/fills.csv',C/'post_fill/receipt.json',C/'queue_front_v2/receipt.json'];assert not (H/'manifest.json').exists();dump(H/'manifest.json',dict(files={str(p.relative_to(R)):sha(p) for p in paths},task_before=task,cap_coreh=.0008,comparisons=2));print('interval controls and source/resource gate pass')
def run():
 for p,h in json.loads((H/'manifest.json').read_text())['files'].items():assert sha(R/p)==h,p
 assert not (H/'fills.csv.gz').exists();orders,rows,source_n=build();cells=aggregate(rows);matched,draws=compare(rows);write(H/'orders.csv.gz',orders);write(H/'fills.csv.gz',rows);write(H/'cells.csv',cells);write(H/'draws.csv',draws);dump(H/'matched.json',matched);result=dict(asof=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),orders=len(orders),source_fill_rows=source_n,unique_fills=len(rows),duplicates=source_n-len(rows),status=dict(collections.Counter(r['timing_status'] for r in rows)),matched=matched);dump(H/'result.json',result);print(json.dumps(result,indent=2));print(json.dumps([r for r in cells if r['dimension']=='all'],indent=2))
if __name__=='__main__':freeze() if '--freeze' in sys.argv else run()
