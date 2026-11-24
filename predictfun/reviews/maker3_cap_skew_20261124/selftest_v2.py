import runpy
import pandas as pd
import numpy as np
import observe_v2 as A
runpy.run_path(str(A.H/'selftest.py'))
p=pd.DataFrame(columns=['day','mid','k'])
o=pd.DataFrame([dict(order_hash='unknown_hash',day=np.nan,mid=np.nan,original_shares=10,side=np.nan,tick=np.nan,post_grade=np.nan,post_lo_ms=np.nan,post_hi_ms=np.nan,
 complete_recorded_history=True,budget_status='exhausted',ws_ms=np.nan,we_ms=np.nan)])
v=A.size_rows(p,o);assert len(v)==2 and v.status.eq('unknown').all() and v.mid.isna().all() and v['size'].eq(10).all()
print('PASS unknown market retained on both clocks')
