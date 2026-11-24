"""Saved-data diagnosis only; no replay/fix. Checks modes against exact noticed fills."""
import ast,collections,json,math
import numpy as np
import extract_v2 as E
R=E.load_records();out=[]
src=(E.R/'predictfun/research/pf_branch_ladder.py').read_text();tree=ast.parse(src);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='next_mode'];ns={'TOL':1e-9};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<saved next_mode>','exec'),ns)
assert ns['next_mode']('LONG_RESTRICTED',-99.92717,100,50)=='LONG_RESTRICTED'
assert ns['next_mode']('LONG_RESTRICTED',-30,100,50)=='UNRESTRICTED'
assert ns['next_mode']('LONG_RESTRICTED',-101,100,50)=='SHORT_RESTRICTED'
for day,mid,part,_ in E.CASES:
 with np.load(E.L/part/(day+'_arrays.npz')) as z:
  for arm in E.ARMS:
   r=R[mid][arm];prefix=f'{mid}__{arm}__';pos=z[prefix+'posc'];mode=z[prefix+'mode'];live=z[prefix+'nva'];fills=r['fills']
   # Independent exact noticed inventory at every stored tick.
   for k,x in enumerate(pos):
    p=sum((1 if s==0 else -1)*q for t,g,s,q,p,tr in fills if t+100<=k*50);assert abs(x-p)<1e-7
   bad=((mode==1)&(pos< -50)&(pos> -100))|((mode==2)&(pos>50)&(pos<100));ids=np.flatnonzero(bad);segments=[]
   for i in ids:
    if not segments or i!=segments[-1][-1]+1:segments.append([int(i),int(i)])
    else:segments[-1][-1]=int(i)
   seg=[]
   for a,b in segments:
    known=[(t,g,s,q,p) for t,g,s,q,p,_ in fills if (a-1)*50<t+100<=a*50];seg.append(dict(start_s=a*.05,end_s=(b+1)*.05,ticks=b-a+1,position_before=float(pos[a-1]) if a else None,position_start=float(pos[a]),position_end=float(pos[b]),mode=int(mode[a]),no_live_ticks=int(np.sum(np.all(live[a:b+1]==0,axis=1))),notices_at_start=known))
   out.append(dict(m=mid,arm=arm,wrong_side_restriction_ticks=len(ids),duration_s=len(ids)*.05,segments=seg,source_mode_log=r['mode_log']))
(E.H/'inventory_diagnosis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
