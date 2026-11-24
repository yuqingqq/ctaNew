"""CSV representation correction: missing=None/NaN, bools typed explicitly.
No label, predicate, tolerance or measured output change. V2 stopped comparing
clean=None with CSV clean=NaN; all ledger checks had passed. Preserve unknowns,
never coerce missing to False. The prior independent context checks still run.
"""
import math,runpy,subprocess
import numpy as np
import pandas as pd
import check as V
import observe_v2 as A
def canon(v,column):
 if pd.isna(v):return None
 if column=='clean':
  assert v in [True,False,'True','False']
  return v is True or isinstance(v,np.bool_) and bool(v) or v=='True'
 return v
assert canon(None,'clean') is None and canon(float('nan'),'clean') is None
assert canon(False,'clean') is False and canon('False','clean') is False
assert canon(True,'clean') is True and canon('True','clean') is True
assert canon(None,'clean')!=canon(False,'clean')
def checked(s):
 p,o=A.rows_and_orders();ref=A.size_rows(p,o)
 assert list(s.columns)==list(ref.columns) and len(s)==len(ref)==2*len(o)
 assert not s.duplicated(['hash','lag_ms']).any()
 for col in s:
  for x,y in zip(s[col],ref[col]):
   x,y=canon(x,col),canon(y,col)
   if x is None or y is None:assert x is y,(col,x,y)
   elif isinstance(x,(int,float,np.number)) and isinstance(y,(int,float,np.number)):assert abs(x-y)<1e-12,(col,x,y)
   else:assert x==y,(col,x,y)
 for r in s[s.status.eq('sampled_window')].itertuples():
  assert r.n>=1 and r.lo_ms<=r.hi_ms
  if r.inventory=='reducing':assert (r.I_max<0 if r.side=='bid' else r.I_min>0)
  if r.touch=='at_touch':assert abs(r.distance_min)<1e-7 and abs(r.distance_max)<1e-7
 return True
assert subprocess.check_output(['git','show','HEAD:'+str((A.H/'check_v3.py').relative_to(A.R))],cwd=A.R)==(A.H/'check_v3.py').read_bytes()
V.check_sizes=checked
runpy.run_path(str(A.H/'check_v2.py'),run_name='__main__')
