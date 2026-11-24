"""Exact condition-prevalence reference and bounded diagnostic draws."""
import csv,json,collections,hashlib,sys,copy,gzip,io
import numpy as np,pandas as pd,pyarrow as pa
import common as B
pa.set_cpu_count(1);pa.set_io_thread_count(1)
FIELDS=('btc_side','inventory_side','spread','own_depth')
def write_gz(p,rows):
 s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 with p.open('wb') as f:
  with gzip.GzipFile(fileobj=f,filename='',mode='wb',mtime=0,compresslevel=1) as z:z.write(s.getvalue().encode())
def kind(x,which):
 if not np.isfinite(x):return 'unknown'
 if which=='btc_side':return 'with' if x>=1 else ('against' if x<=-1 else 'flat')
 if which=='inventory_side':return 'adding' if x>1e-6 else ('offsetting' if x< -1e-6 else 'flat')
 if which=='spread':return 'wide' if x>=2 else 'tight'
 if which=='own_depth':return 'thin' if x<=50 else ('deep' if x>200 else 'middle')
 return 'last30s' if x<=30 else ('last60s' if x<=60 else 'earlier')
def joined(vals):return 'unknown' if 'unknown' in vals else '|'.join(sorted(set(vals)))
def inputs():
 for p,h in json.loads((B.H/'reference_bindings.json').read_text()).items():assert B.sha(B.R/p)==h,p
 rr=[r for r in B.read(B.H/'outputs/orders.csv') if r['successor_relation']=='sequential_after_under_bounds'];assert len(rr)==30
 cx={(r['order_hash'],r['lag_ms']):r for r in B.read(B.H/'outputs/contexts.csv') if r['stage']=='successor'};p=pd.read_parquet(B.C/'posting_opportunity/rows.parquet');i=pd.read_parquet(B.C/'take_logic_v2/panel.parquet');pg={(str(d),int(m),s):g.sort_values('k').set_index('k') for (d,m,s),g in p.groupby(['day','mid','side'])};ig={(str(d),int(m),int(rank)):g.sort_values('k').set_index('k') for (d,m,rank),g in i.groupby(['day','mid','rank'])};return rr,cx,pg,ig
def tally(rr,windows,cx):
 by=collections.defaultdict(list)
 for r in windows:by[(r['order_hash'],str(r['lag_ms']))].append(r)
 out=[];rng=np.random.default_rng(20261124);draws=[]
 for lag in ('0','1000'):
  chosen=[by[(r['order_hash'],lag)] for r in rr];indices=np.column_stack([rng.integers(0,len(a),500) for a in chosen])
  for f in FIELDS:
   cats=sorted({x[f] for a in chosen for x in a}|{cx[(r['order_hash'],lag)][f] for r in rr})
   for c in cats:
    observed=sum(cx[(r['order_hash'],lag)][f]==c for r in rr);expected=sum(sum(x[f]==c for x in a)/len(a) for a in chosen);sample=[sum(chosen[j][int(indices[k,j])][f]==c for j in range(len(chosen))) for k in range(500)];out.append(dict(lag_ms=lag,field=f,category=c,actions=len(rr),observed_count=observed,expected_uniform_count=expected,observed_share=observed/len(rr),expected_uniform_share=expected/len(rr),difference_pp=100*(observed-expected)/len(rr),draw_mean_count=float(np.mean(sample))))
    for k,x in enumerate(sample):draws.append(dict(lag_ms=lag,field=f,category=c,draw=k,count=x))
 return out,draws
def build():
 rr,cx,pg,ig=inputs();out=[]
 for r in rr:
  pp=pg[(r['day'],int(r['mid']),r['side'])];ii=ig[(r['day'],int(r['mid']),int(r['rank']))];s=1 if r['side']=='bid' else -1
  for lag,z in ((0,'0'),(1000,'1')):
   obs=cx[(r['order_hash'],str(lag))];assert obs['coverage']=='covered';width=int(obs['sampled_states']);sf='D'+str(lag)
   arr={'btc_side':pp['btc_r_5s_bp_'+sf].to_numpy()*s,'inventory_side':ii['q'+z].to_numpy()*s,'spread':pp['spread_ticks_'+sf].to_numpy(),'own_depth':pp['depth_touch_own_'+sf].to_numpy(),'phase':pp['time_left_s_'+sf].to_numpy()};labs={f:np.array([kind(v,f) for v in a]) for f,a in arr.items()};book=pp['status_'+sf].to_numpy();actual=(int(r['successor_post_lo_ms'])-int(r['ws_ms']))//1000
   for k in range(301-width):
    sl=slice(k,k+width);phase=joined(labs['phase'][sl]);
    if phase!=obs['time_left']:continue
    a=dict(order_hash=r['order_hash'],lag_ms=lag,start_k=k,width=width,phase=phase,actual_window=k==actual)
    for f in FIELDS:a[f]=joined(labs[f][sl]) if f not in ('spread','own_depth') or (book[sl]=='ok').all() else 'unknown'
    out.append(a)
   hits=[x for x in out if x['order_hash']==r['order_hash'] and x['lag_ms']==lag and x['actual_window']];assert len(hits)==1 and all(hits[0][f]==obs[f] for f in FIELDS)
 write_gz(B.H/'outputs/reference_windows.csv.gz',out);tot,draw=tally(rr,out,cx);B.write(B.H/'outputs/reference_comparison.csv',tot);write_gz(B.H/'outputs/reference_draws.csv.gz',draw);B.dump(B.H/'reference_result.json',dict(cases=len(rr),case_clocks=2*len(rr),reference_windows=len(out),draws_per_case_clock=500,matched_nonactions=False,validation=False));print(json.dumps(dict(cases=len(rr),reference_windows=len(out),comparison_rows=len(tot))))
def check():
 rr,cx,pg,ig=inputs();expected={}
 for r in rr:
  pp=pg[(r['day'],int(r['mid']),r['side'])].to_dict('index');ii=ig[(r['day'],int(r['mid']),int(r['rank']))].to_dict('index');sgn=1 if r['side']=='bid' else -1
  for lag,z in ((0,'0'),(1000,'1')):
   obs=cx[(r['order_hash'],str(lag))];n=(int(r['successor_post_hi_ms'])-int(r['ws_ms']))//1000-(int(r['successor_post_lo_ms'])-int(r['ws_ms']))//1000+1;assert n==int(obs['sampled_states']);sf='D'+str(lag)
   for k in range(301-n):
    states=[pp[j] for j in range(k,k+n)];phase='|'.join(sorted({'last30s' if p['time_left_s_'+sf]<=30 else ('last60s' if p['time_left_s_'+sf]<=60 else 'earlier') for p in states}))
    if phase!=obs['time_left']:continue
    a=dict(order_hash=r['order_hash'],lag_ms=str(lag),start_k=str(k),width=str(n),phase=phase,actual_window=str(k==(int(r['successor_post_lo_ms'])-int(r['ws_ms']))//1000))
    for field in FIELDS:
     values=[]
     for j,p in zip(range(k,k+n),states):
      if field=='btc_side':v=p['btc_r_5s_bp_'+sf]*sgn;val='with' if v>=1 else ('against' if v<=-1 else 'flat')
      elif field=='inventory_side':v=ii[j]['q'+z]*sgn;val='adding' if v>1e-6 else ('offsetting' if v< -1e-6 else 'flat')
      elif field=='spread':v=p['spread_ticks_'+sf];val='wide' if v>=2 else 'tight'
      else:v=p['depth_touch_own_'+sf];val='thin' if v<=50 else ('deep' if v>200 else 'middle')
      values.append(val if np.isfinite(v) else 'unknown')
     a[field]='unknown' if 'unknown' in values or (field in ('spread','own_depth') and any(p['status_'+sf]!='ok' for p in states)) else '|'.join(sorted(set(values)))
    expected[(r['order_hash'],str(lag),str(k))]=a
 actual=B.read(B.H/'outputs/reference_windows.csv.gz')
 def verify(rs):
  got={(r['order_hash'],r['lag_ms'],r['start_k']):r for r in rs};assert len(rs)==len(got)==len(expected) and set(got)==set(expected)
  for k,e in expected.items():assert got[k]==e,k
 verify(actual);bad={}
 for name,field,value in [('state','btc_side','invented'),('width','width','999'),('future','start_k','300'),('phase','phase','wrong')]:
  z=copy.deepcopy(actual);z[0][field]=value
  try:verify(z);raise RuntimeError('escaped '+name)
  except AssertionError:bad[name]='refused'
 try:verify(actual[:-1]);raise RuntimeError('omission escaped')
 except AssertionError:bad['omission']='refused'
 # Recreate exact aggregates/draws from independently enumerated reference states.
 totals,draws=tally(rr,list(expected.values()),cx)
 for fname,es in [('reference_comparison.csv',totals),('reference_draws.csv.gz',draws)]:
  ar=B.read(B.H/'outputs'/fname);assert len(ar)==len(es)
  for a,e in zip(ar,es):
   for k,v in e.items():assert abs(float(a[k])-v)<1e-9 if isinstance(v,(int,float)) else a[k]==str(v),(fname,k)
 B.dump(B.H/'reference_checks.json',dict(pass_all=True,windows=len(actual),case_clocks=60,falsifiers=bad));print('independent reference windows and aggregates pass; five corruptions refused')
def freeze():
 paths=[B.H/n for n in ('reference.py','reference_contract.md','outputs/orders.csv','outputs/contexts.csv')]+[B.C/'take_logic_v2/panel.parquet',B.C/'posting_opportunity/rows.parquet'];assert not (B.H/'reference_bindings.json').exists();B.dump(B.H/'reference_bindings.json',{str(p.relative_to(B.R)):B.sha(p) for p in paths});print('reference frozen before output')
if __name__=='__main__':
 if '--freeze' in sys.argv:freeze()
 elif '--test' in sys.argv:
  assert joined(['flat','against'])=='against|flat' and joined(['unknown','flat'])=='unknown'
  assert kind(-1,'btc_side')=='against' and kind(1,'btc_side')=='with' and kind(0,'inventory_side')=='flat'
  assert kind(30,'phase')=='last30s' and kind(31,'phase')=='last60s' and kind(61,'phase')=='earlier'
  assert kind(float('nan'),'spread')=='unknown';print('four category/boundary/unknown groups passed')
 elif '--check' in sys.argv:check()
 else:build()
