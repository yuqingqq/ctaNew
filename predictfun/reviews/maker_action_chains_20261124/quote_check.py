"""Independent scalar reconstruction; does not import the quote builder."""
import csv,gzip,json,hashlib,math,copy
from pathlib import Path
import pandas as pd
import pyarrow as pa
pa.set_cpu_count(1);pa.set_io_thread_count(1)
H=Path(__file__).resolve().parent;R=H.parents[2]
def read(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
for p,h in json.loads((H/'quote_bindings.json').read_text()).items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
C=read(H/'outputs/chains.csv');N={r['order_hash']:r for r in read(R/'predictfun/reviews/maker_timing_estimates_20261124/outputs/post_estimates.csv.gz')};P=pd.read_parquet(R/'predictfun/reviews/maker_trace_20261122/v2/chains/take_logic_v2/panel.parquet');pan={(int(d),int(m),int(r)):g.set_index('k').to_dict('index') for (d,m,r),g in P.groupby(['day','mid','rank'])}
expected={}
for c in C:
 for lag,z in ((0,'0'),(1000,'1')):
  a={k:c[k] for k in ('day','mid','rank','order_hash','group','successor_hash','side')};a.update(sequence=c['successor_relation'],lag_ms=str(lag),status='ok',comparison='unknown',quote_change_ticks='',best_change_min_ticks='',best_change_max_ticks='',offset_change_min_ticks='',offset_change_max_ticks='',old_states='0',new_states='0')
  if not c['successor_hash']:a['status']='no_recorded_successor'
  elif c['successor_relation']=='refused_or_missing_window' or c['terminal']=='unresolved':a['status']='refused_window'
  else:
   n=N[c['successor_hash']];assert (n['maker'].lower(),n['day'],n['mid'],n['side'])==(c['maker'],c['day'],c['mid'],c['side']);ws=int(c['ws_ms']);bounds=[(int(c['end_lo_ms']),int(c['end_hi_ms'])),(int(c['successor_post_lo_ms']),int(c['successor_post_hi_ms']))];pp=pan[(int(c['day']),int(c['mid']),int(c['rank']))];sets=[]
   for lo,hi in bounds:
    aa=(lo-ws)//1000;bb=(hi-ws)//1000
    if not 0<=aa<=bb<300:a['status']='outside_grid';break
    ss=[]
    for k in range(aa,bb+1):
     r=pp[k];assert r['t_ms']==ws+1000*k;ss.append(r)
    sets.append(ss)
   if a['status']=='ok':
    if any(not r['kok'+z] for ss in sets for r in ss):a['status']='book_unavailable'
    elif any(not math.isfinite(r[c['side']+z]) or not 0<r[c['side']+z]<1 or abs(100*r[c['side']+z]-round(100*r[c['side']+z]))>1e-8 for ss in sets for r in ss):a['status']='off_grid_book'
    else:
     sign=1 if c['side']=='bid' else -1;old=int(c['tick']);new=int(n['tick']);vals=[];moves=[]
     for x in sets[0]:
      for y in sets[1]:
       bx=round(100*x[c['side']+z]);by=round(100*y[c['side']+z]);v=sign*(new-old-by+bx);m=sign*(by-bx);assert m+v==sign*(new-old);vals.append(v);moves.append(m)
     lo=min(vals);hi=max(vals);result='timing_ambiguous'
     if lo>0:result='more_aggressive_relative_touch'
     elif hi<0:result='less_aggressive_relative_touch'
     elif lo==hi==0:result='unchanged_relative_touch'
     a.update(comparison=result,quote_change_ticks=str(sign*(new-old)),best_change_min_ticks=str(min(moves)),best_change_max_ticks=str(max(moves)),offset_change_min_ticks=str(lo),offset_change_max_ticks=str(hi),old_states=str(len(sets[0])),new_states=str(len(sets[1])))
  expected[(c['order_hash'],str(lag))]=a
actual=read(H/'outputs/quote_relative.csv')
def validate(rr):
 got={(r['order_hash'],r['lag_ms']):r for r in rr};assert len(rr)==len(got)==len(expected)==176,'population'
 for k,v in expected.items():assert got[k]==v,(k,got[k],v)
validate(actual);positive=next(i for i,r in enumerate(actual) if r['status']=='ok');bad={}
for name,field,value in [('identity','successor_hash','wrong'),('side','side','wrong'),('direction','comparison','invented'),('offset','offset_change_min_ticks','999'),('book_status','status','invented')]:
 x=copy.deepcopy(actual);x[positive][field]=value
 try:validate(x);raise RuntimeError('missed '+name)
 except AssertionError:bad[name]='refused'
validate(actual)
(H/'quote_checks.json').write_text(json.dumps(dict(pass_all=True,rows=len(actual),positive_control=True,falsifiers=bad),indent=2)+'\n');print('176 rows independently equal; 5 corruptions refused')
