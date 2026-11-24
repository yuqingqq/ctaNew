"""Verify the opening cutoff and the unchanged later slot decisions separately."""
import copy,gzip,json,math,sys
import opening as N
X=N.X;np=N.np;H=N.H;OUT=N.OUT
import check as parent_check
def scope_verify(rec,cache):
 first=min((o['submit'] for o in rec['orders']),default=None)
 assert rec['opening_batch_ms']==first
 assert all(e['t']<=first for e in rec['entry_log']) if first is not None else True
 later=0
 for e in rec['scope_log']:
  t=e['t'];s=e['side'];expected_active=first is None or t<=first
  assert e['gate_active']==expected_active
  c=rec['commands'][e['command_index']]
  assert (c[0],c[1],c[4],c[5],c[7],c[6],c[8])==(t,s,e['rung'],e['price'],e['action'],e['q'],e['why'])
  if expected_active:continue
  later+=1;I=e['position'];B=e['B'];A=e['A'];mode=e['mode']
  if (s==0 and mode=='LONG_RESTRICTED') or (s==1 and mode=='SHORT_RESTRICTED'):
   expected=('WAIT',None,'restricted')
  else:
   f=min(1.,max(0.,1-I/100 if s==0 else 1+I/100));bs=B if s==0 else A;room=200-I-B if s==0 else 200+I-A
   q=float(math.floor(min(100*f,max(0,200*f-bs),max(0,room))+1e-9))
   if q<1:expected=('WAIT',q,'zero_desired' if 100*f<1-1e-9 else 'exposure_full' if 200*f-bs<1-1e-9 else 'no_room')
   else:
    k=int(t/50);x=float(cache[f'x{s}'][k]);ok=int(cache[f'status{s}'][k])==0 and math.isfinite(x)
    edge={0:.4666730940341952,1:.285456120967865}[s]
    expected=('WAIT',None,'pressure_withhold') if ok and x>0 and x>=edge else ('POST',q,'post')
  assert (e['action'],e['q'],e['why'])==expected,('later_policy',t,s)
 return later
def main():
 sm=json.loads((OUT/'summary.json').read_text());assert sm['status']=='COMPLETE' and {r['m'] for r in sm['rows']}=={m for _,m,_ in X.CASES}
 total=0;rows=[];sample=None
 for day,m,part in X.CASES:
  p=OUT/f'{m}_{N.ARM}.json.gz';ap=OUT/f'{m}_{N.ARM}.npz';rec=json.loads(gzip.decompress(p.read_bytes()));arr=np.load(ap)
  sr=next(r for r in sm['rows'] if r['m']==m);assert X.sha(p)==sr['sha256'] and X.sha(ap)==sr['arrays_sha256'] and sr['metrics']==rec['metrics']
  pp='pilot' if part=='pilot' else 'days';z=np.load(X.P.OUT/pp/f'{day}_arrays.npz');cache={n:z[f'{m}__cache__{n}'] for n in ['x0','x1','status0','status1']}
  parent_check.verify(rec,cache,arr);later=scope_verify(rec,cache);assert later>0;total+=later
  baseline=json.loads(gzip.decompress((X.OUT/f'{m}_FIXED_DEFAULT.json.gz').read_bytes()));base=baseline['metrics'];new=rec['metrics']
  rows.append(dict(m=m,opening_batch_ms=rec['opening_batch_ms'],later_slot_calls_verified=later,baseline=base,candidate=new,delta_c=new['total_c']-base['total_c']))
  sample=(rec,cache)
 rec,cache=sample;bad=[]
 for kind in ['scope','late_action','cutoff','late_gate_log']:
  r=copy.deepcopy(rec);late=next(e for e in r['scope_log'] if not e['gate_active'])
  if kind=='scope':late['gate_active']=True
  if kind=='late_action':late['action']='PULL'
  if kind=='cutoff':r['opening_batch_ms']+=50
  if kind=='late_gate_log':r['entry_log'].append(dict(r['entry_log'][0],t=r['opening_batch_ms']+50))
  try:scope_verify(r,cache)
  except AssertionError:bad.append(kind)
  else:raise AssertionError('escaped '+kind)
 result=dict(markets=3,later_slot_calls_verified=total,refusals=bad,rows=rows,
  better_markets=sum(r['delta_c']>0 for r in rows),positive_markets=sum(r['candidate']['total_c']>0 for r in rows),
  pooled_delta_c=sum(r['delta_c'] for r in rows),pooled_baseline_c=sum(r['baseline']['total_c'] for r in rows),pooled_candidate_c=sum(r['candidate']['total_c'] for r in rows),validation=False)
 X.dump(H/'checks.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
