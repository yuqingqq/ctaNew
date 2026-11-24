"""Independent saved-result accounting and causal admission checks; no replay."""
import copy,gzip,json,math,sys
from pathlib import Path
import numpy as np
import replay as X
H=X.H;D=X.DATA;OUT=X.OUT
def verify(r,cache,arr):
 fs=r['fills'];cash=sum((1 if f[2]==1 else -1)*f[3]*f[4] for f in fs);pos=sum((1 if f[2]==0 else -1)*f[3] for f in fs)
 assert abs(cash-r['cash'])<1e-6 and abs(pos-r['position'])<1e-6
 assert abs(cash+pos*r['mark']-r['value'])<1e-6
 v=r['metrics'];assert abs(v['total_c']-r['value'])<1e-6
 assert abs(v['completed_c']+v['residual_c']-v['total_c'])<1e-6
 assert abs(v['filled_shares']-sum(f[3] for f in fs))<1e-6
 assert all(0<=f[0]<=r['end'] and f[3]>0 for f in fs)
 orders={o['gid']:o for o in r['orders']};by={g:sum(f[3] for f in fs if f[1]==g) for g in orders}
 for g,o in orders.items():assert by[g]<=o['size']+1e-6
 if r['arm']!='ORIGINAL':
  assert not np.any((arr['mode']==1)&(arr['posc']< -1e-9))
  assert not np.any((arr['mode']==2)&(arr['posc']>1e-9))
 for e in r['entry_log']:
  t=e['t'];s=e['side'];k=int(t/50);x=cache[f'x{1-s}'][k];st=int(cache[f'status{1-s}'][k]);threshold={0:.4666730940341952,1:.285456120967865}[1-s]
  assert e['other_signal']==(float(x) if math.isfinite(x) else None)
  known=sum((1 if f[2]==0 else -1)*f[3] for f in fs if f[0]+100<=t)
  assert abs(known-e['position'])<1e-6
  posted={int(c[3]) for c in r['commands'][:e['command_index']] if c[7]=='POST'}
  open_g=[g for g in posted if str(g) not in r['closed'] or r['closed'][str(g)][1]>t]
  for side,key in [(0,'B'),(1,'A')]:
   reserved=sum(orders[g]['size']-sum(f[3] for f in fs if f[1]==g and f[0]+100<=t) for g in open_g if orders[g]['side']==side)
   assert abs(reserved-e[key])<1e-6,(t,key,reserved,e[key])
  allowed=e['proposed'] if st==0 and math.isfinite(x) and x>0 and x>=threshold else max(0,math.floor(min(e['proposed'],max(0,-known-e['B'] if s==0 else known-e['A']))+1e-9))
  assert e['quantity']==allowed
  c=r['commands'][e['command_index']];assert c[0]==t and c[1]==s and c[5]==e['price']
  assert (c[7]=='POST' and c[6]==allowed) if allowed>=1 else c[7]=='WAIT'
 return True
def main():
 sm=json.loads((OUT/'summary.json').read_text());assert sm['status']=='COMPLETE'
 assert {(r['m'],r['arm']) for r in sm['rows']}=={(m,a) for _,m,_ in X.CASES for a in X.ARMS}
 checked=0;sample=None
 for day,m,part in X.CASES:
  pp='pilot' if part=='pilot' else 'days';z=np.load(X.P.OUT/pp/f'{day}_arrays.npz');cache={n:z[f'{m}__cache__{n}'] for n in ['x0','x1','status0','status1']}
  for arm in X.ARMS:
   p=OUT/f'{m}_{arm}.json.gz';r=json.loads(gzip.decompress(p.read_bytes()));arr=np.load(OUT/f'{m}_{arm}.npz')
   sr=next(x for x in sm['rows'] if x['m']==m and x['arm']==arm)
   assert X.sha(p)==sr['sha256'] and sr['metrics']==r['metrics']
   verify(r,cache,arr);checked+=1
   if arm=='ORIGINAL':
    old=X.records(day,part)[m]['CURRENT_ASYNC_FAST10']
    for key in ['fills','orders','requests','fingerprint','value']:assert r[key]==old[key]
   if arm=='DIRECTIONAL_ENTRY' and r['entry_log'] and r['fills']:sample=(r,cache,arr)
 assert sample is not None
 bad=[]
 for name in ['cash','total','shares','signal','gate_quantity','known_inventory','reservation']:
  r,c,a=sample;r=copy.deepcopy(r)
  if name=='cash':r['cash']+=1
  if name=='total':r['metrics']['total_c']+=1
  if name=='shares':r['metrics']['filled_shares']+=1
  if name=='signal':r['entry_log'][0]['other_signal']=999
  if name=='gate_quantity':r['entry_log'][0]['quantity']+=1
  if name=='known_inventory':r['entry_log'][0]['position']+=1
  if name=='reservation':r['entry_log'][0]['B']+=1
  try:verify(r,c,a)
  except AssertionError:bad.append(name)
  else:raise AssertionError('escaped '+name)
 contrasts=[]
 for day,m,_ in X.CASES:
  v={r['arm']:r['metrics'] for r in sm['rows'] if r['m']==m}
  contrasts.append(dict(day=day,m=m,inventory_fix_delta_c=v['FIXED_DEFAULT']['total_c']-v['ORIGINAL']['total_c'],entry_delta_c=v['DIRECTIONAL_ENTRY']['total_c']-v['FIXED_DEFAULT']['total_c'],
   entry_completed_delta_c=v['DIRECTIONAL_ENTRY']['completed_c']-v['FIXED_DEFAULT']['completed_c'],entry_residual_delta_c=v['DIRECTIONAL_ENTRY']['residual_c']-v['FIXED_DEFAULT']['residual_c'],
   entry_total_c=v['DIRECTIONAL_ENTRY']['total_c'],shares_delta=v['DIRECTIONAL_ENTRY']['filled_shares']-v['FIXED_DEFAULT']['filled_shares']))
 result=dict(checked=checked,corruptions_refused=bad,contrasts=contrasts,entry_better_each=all(x['entry_delta_c']>0 for x in contrasts),
  entry_positive_each=all(x['entry_total_c']>0 for x in contrasts),entry_pooled_delta_c=sum(x['entry_delta_c'] for x in contrasts),selection_beyond_exposure_tested=False,validation=False)
 X.dump(H/'checks.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
