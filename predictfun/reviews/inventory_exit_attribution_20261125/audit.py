import collections,copy,csv,gzip,hashlib,json,math,resource,subprocess,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2];P=H.parent/'inventory_exit_exception_20261125';D=R/'data/predict_fun/research/inventory_exit_exception_20261125';EPS=1e-8
CATEGORIES=('common_closed','common_close_vs_residual','common_residual','new_entry','foregone_entry','new_role','foregone_role')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def key(f):t,i,s,q,p,it=f;assert it>=0;return (int(it),float(t),int(s),float(p))
def lots(r):
 qevent=collections.defaultdict(float);opening=collections.defaultdict(list);frags=collections.defaultdict(list);queue=collections.deque();completed=residual=0.
 for f in r['fills']:
  t,g,s,q,p,it=f;e=key(f);lo=qevent[e];qevent[e]+=q;sign=1-2*s
  while q>EPS and queue and queue[0]['sign']!=sign:
   lot=queue[0];n=min(q,lot['hi']-lot['lo']);piece=dict(lot,hi=lot['lo']+n,exit_price=p,exit_t=t,closed=True,exit_event=e)
   frags[lot['event']].append(piece);completed+=lot['sign']*n*(p-lot['price']);lot['lo']+=n;q-=n;lo+=n
   if lot['hi']-lot['lo']<=EPS:queue.popleft()
  if q>EPS:
   lot=dict(event=e,lo=lo,hi=lo+q,sign=sign,price=p,t=t);queue.append(lot);opening[e].append((lo,lo+q))
 for lot in queue:
  frags[lot['event']].append(dict(lot,exit_price=r['mark'],exit_t=r['end'],closed=False,exit_event=None));residual+=lot['sign']*(lot['hi']-lot['lo'])*(r['mark']-lot['price'])
 for e,ls in frags.items():
  ls.sort(key=lambda z:z['lo']);assert all(a['hi']<=b['lo']+EPS for a,b in zip(ls,ls[1:]))
  assert abs(sum(z['hi']-z['lo'] for z in ls)-sum(b-a for a,b in opening[e]))<1e-6
  for z in ls:assert -EPS<=z['lo']<z['hi']<=qevent[e]+EPS and z['exit_t']>=z['t']
 return dict(quantity=qevent,fragments=frags,completed=completed,residual=residual)
def independent_fifo(r):
 held=[];value=0.;cash=0.;position=0.
 for t,g,s,q,p,it in r['fills']:
  sg=1-2*s;cash-=sg*q*p;position+=sg*q
  while q>EPS and held and held[0][0]!=sg:
   old=held[0];d=min(q,old[1]);value+=old[0]*d*(p-old[2]);q-=d;old[1]-=d
   if old[1]<=EPS:held.pop(0)
  if q>EPS:held.append([sg,q,p])
 residual=sum(s*q*(r['mark']-p) for s,q,p in held)
 assert abs(value+residual-cash-position*r['mark'])<1e-5
 return value,residual,cash+position*r['mark']
def at(frags,x):return next((z for z in frags if z['lo']<=x<z['hi']),None)
def decompose(a,b):
 A,B=lots(a),lots(b);rows=[]
 for e in sorted(A['fragments'].keys()|B['fragments'].keys()):
  fa,fb=A['fragments'].get(e,[]),B['fragments'].get(e,[])
  bounds=sorted({0.,A['quantity'].get(e,0),B['quantity'].get(e,0)}|{z[v] for z in fa+fb for v in ['lo','hi']})
  for lo,hi in zip(bounds,bounds[1:]):
   if hi-lo<=EPS:continue
   x=(lo+hi)/2;u,v=at(fa,x),at(fb,x)
   if u is None and v is None:continue
   q=hi-lo;ua=0 if u is None else u['sign']*q*(u['exit_price']-u['price']);vb=0 if v is None else v['sign']*q*(v['exit_price']-v['price'])
   if u is not None and v is not None:
    cat='common_closed' if u['closed'] and v['closed'] else 'common_residual' if not u['closed'] and not v['closed'] else 'common_close_vs_residual'
   elif v is not None:cat='new_role' if x<A['quantity'].get(e,0) else 'new_entry'
   else:cat='foregone_role' if x<B['quantity'].get(e,0) else 'foregone_entry'
   rows.append(dict(category=cat,q=q,delta_c=vb-ua,wide_c=ua,cost_c=vb,entry_ms=e[1],side=e[2],entry_price=e[3],
    exit_price_diff=None if u is None or v is None else v['exit_price']-u['exit_price'],
    signed_exit_price_diff=None if u is None or v is None else u['sign']*(v['exit_price']-u['exit_price']),
    hold_ms_diff=None if u is None or v is None else v['exit_t']-u['exit_t'],
    wide_closed=None if u is None else u['closed'],cost_closed=None if v is None else v['closed']))
 assert abs(sum(z['delta_c'] for z in rows)-(B['completed']+B['residual']-A['completed']-A['residual']))<1e-5
 return rows
def verify_record(r):
 a,b,v=independent_fifo(r);L=lots(r)
 assert abs(a-L['completed'])<1e-5 and abs(b-L['residual'])<1e-5
 assert abs(a-r['metrics']['completed_c'])<1e-5 and abs(b-r['metrics']['residual_c'])<1e-5 and abs(v-r['value'])<1e-5
 assert abs(v-r['net_value_c'])<1e-5
def verify_rows(a,b,rows):
 assert rows==decompose(a,b),'row/component differs'
 va=independent_fifo(a)[2];vb=independent_fifo(b)[2];assert abs(sum(z['delta_c'] for z in rows)-vb+va)<1e-5
def group(rows):
 out=[]
 for cat in CATEGORIES:
  rs=[r for r in rows if r['category']==cat];q=sum(r['q'] for r in rs)
  common=cat.startswith('common')
  out.append(dict(category=cat,n=len(rs),shares=q,delta_c=sum(r['delta_c'] for r in rs),positive_c=sum(max(0,r['delta_c']) for r in rs),negative_c=sum(min(0,r['delta_c']) for r in rs),
   signed_exit_price_diff=sum(r['q']*r['signed_exit_price_diff'] for r in rs)/q if common and q else None,
   hold_ms_diff=sum(r['q']*r['hold_ms_diff'] for r in rs)/q if common and q else None))
 return out
def first_divergence(a,b):
 aa,bb=a['commands'],b['commands'];ix=next((i for i,(x,y) in enumerate(zip(aa,bb)) if x!=y),None)
 if ix is None:
  assert aa==bb;assert a['fills']==b['fills'];return dict(status='identical',t=None)
 x,y=aa[ix],bb[ix];assert x[:3]==y[:3] and y[8]=='exit_override';t=x[0]
 assert [f for f in a['fills'] if f[0]<=t]==[f for f in b['fills'] if f[0]<=t]
 z=next(z for z in b['exit_log'] if z['command_index']==ix);assert z['override']
 return dict(status='first_override',command_index=ix,t=t,wide=x,cost=y,context=z)
def prepare():
 assert not (H/'manifest.json').exists();sm=json.loads((D/'summary.json').read_text());rows=[r for r in sm['rows'] if r['arm'] in ['WIDE','WIDE_COST']];assert len(rows)==38
 inputs={str(p.relative_to(R)):sha(p) for p in [P/'manifest.json',P/'checks.json',P/'claims.json',P/'summary.json']}
 for r in rows:
  p=D/f"{r['mid']}_{r['arm']}.json.gz";assert sha(p)==r['sha256'];inputs[str(p.relative_to(R))]=r['sha256']
 dump(H/'manifest.json',dict(asof='2026-11-25',inputs=inputs,sources={f:sha(H/f) for f in ['contract.md','audit.py','selftest.py']},new_candidates=0))
def bound():
 m=json.loads((H/'manifest.json').read_text())
 for p,h in m['inputs'].items():assert sha(R/p)==h,p
 for p,h in m['sources'].items():assert sha(H/p)==h
 for p in list(m['sources'])+['manifest.json']:assert subprocess.check_output(['git','show','HEAD:'+str((H/p).relative_to(R))],cwd=R)==(H/p).read_bytes()
def run():
 resource.setrlimit(resource.RLIMIT_CPU,(50,51));bound();start=time.process_time();sm=json.loads((P/'summary.json').read_text());pairs=sorted({(r['day'],r['mid']) for r in sm['rows']});allrows=[];markets=[];first=[];bad=0
 for day,mid in pairs:
  a,b=[json.loads(gzip.decompress((D/f'{mid}_{arm}.json.gz').read_bytes())) for arm in ['WIDE','WIDE_COST']]
  assert a['tape_id']==b['tape_id'] and a['mark']==b['mark'] and a['end']==b['end']
  for r in [a,b]:verify_record(r)
  rows=decompose(a,b);verify_rows(a,b,rows);delta=sum(r['delta_c'] for r in rows)
  first.append(dict(day=day,mid=mid,**first_divergence(a,b)));markets.append(dict(day=day,mid=mid,delta_c=delta,components=group(rows)))
  allrows.extend(dict(day=day,mid=mid,**r) for r in rows)
  if mid==3789484:
   for change in ['component','quantity','mark','value']:
    ar=copy.deepcopy(a);br=copy.deepcopy(b);rr=copy.deepcopy(rows)
    if change=='component':rr[0]['delta_c']+=1
    elif change=='quantity':rr[0]['q']+=1
    elif change=='mark':br['mark']+=1
    else:br['value']+=1
    try:
     assert ar['mark']==br['mark'];verify_record(ar);verify_record(br);verify_rows(ar,br,rr)
    except AssertionError:bad+=1
    else:raise AssertionError('corruption escaped '+change)
 prior=json.loads((P/'claims.json').read_text())['contrasts']['WIDE_COST minus WIDE'];assert abs(sum(r['delta_c'] for r in allrows)-prior['total'])<1e-5
 assert all(abs(x['delta_c']-next(y['delta'] for y in prior['markets'] if y['mid']==x['mid']))<1e-5 for x in markets)
 out=dict(pooled=group(allrows),markets=markets,days={d:group([r for r in allrows if r['day']==d]) for d,m in pairs},total_c=sum(r['delta_c'] for r in allrows))
 dump(H/'components.json',out);dump(H/'first_divergences.json',first);dump(H/'checks.json',dict(pass_all=True,ledgers=38,pairs=19,partitions=len(allrows),falsifiers=bad,cpu_s=time.process_time()-start))
 with (H/'components.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['day','mid']+list(out['pooled'][0]),lineterminator='\n');w.writeheader()
  for m in markets:
   for z in m['components']:w.writerow(dict(day=m['day'],mid=m['mid'],**z))
 named=next(m for m in markets if m['mid']==3789484)
 lines=['# Inventory exit exception: paired accounting attribution','',
  'Saved WIDE_COST minus WIDE,19consumed markets. Exact FIFO/event-unit attribution under the declared canonical overlap convention; no new policy/replay and no causal isolation of exits from subsequent entries. All values USD.','',
  '| Component | All19 |3789484|','|---|---:|---:|']
 for x,y in zip(out['pooled'],named['components']):lines.append(f"|{x['category']}|{x['delta_c']/100:+.2f}|{y['delta_c']/100:+.2f}|")
 lines+=['',f"Total difference: all19 ${out['total_c']/100:+.2f};3789484 ${named['delta_c']/100:+.2f}. Components sum exactly before rounding.",'',
 'common_closed compares the sale/cover prices for identical entry-event quantity units. common_close_vs_residual compares closing with carrying the same units to END. new_entry and foregone_entry concern opening fills unique to one arm. new_role/foregone_role are common fill units opening inventory in one arm but closing earlier inventory in the other. These roles reflect the changed full trading chain, not a direct intervention on entry policy.',
 'No quantity or loss is silently discarded. Gains/losses and residuals are included; both arms use the same END mark. FIFO attribution and within-event overlap allocation are accounting conventions. They cannot recover hidden Maker3 orders or identify an isolated optimal exit decision. Earlier/later exit times alone do not prove foresight.',
 f"Checks:38ledgers,19paired deltas,all{len(allrows)}interval pieces; four real corruptions refused. Synthetic controls in selftest.py. Full day/market component tables in components.json/.csv; first action/context divergences in first_divergences.json."]
 (H/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(pooled=out['pooled'],named=named,first=next(z for z in first if z['mid']==3789484)),indent=2))
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
