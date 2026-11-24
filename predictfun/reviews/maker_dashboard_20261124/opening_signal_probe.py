"""Descriptive saved-data opening trace; no replay, new days, or fitted rule."""
from pathlib import Path
import gzip,json
import numpy as np
import pyarrow.parquet as pq
H=Path(__file__).resolve().parent;R=H.parents[2];D=R/'data/predict_fun/research'
mid=3789484;ws=1792977000000.;day='20261026';arm='CURRENT_ASYNC_FAST10'
z=np.load(D/f'ladder_async_fast_v1_20261122/pilot/{day}_arrays.npz')
c=np.load(D/f'ladder_pressure_v1_20261121/pilot/{day}_arrays.npz')
rows=[]
for k in range(25):
 r={'t_ms':50*k}
 for name in ['bestc','bestv','mid','posc','pose','mode','res']:
  r[name]=z[f'{mid}__{arm}__{name}'][k].tolist()
 for s in (0,1):
  for name in ['x','age','status']:
   r[f'{name}{s}']=c[f'{mid}__cache__{name}{s}'][k].item()
 rows.append(r)
print('TICKS',json.dumps(rows))
root=D/'maker_trace_v2_20261123'
for name in ['legs','records']:
 p=root/f'{name}_{day}.parquet'
 cols=pq.read_schema(p).names; print(name+'_COLS',cols)
 tab=pq.read_table(p,filters=[('mid','=',mid),('settle_ms','>=',ws-100),('settle_ms','<=',ws+1200)]).to_pandas()
 print(name+'_ROWS',tab.to_json(orient='records'))
(H/'opening_signal_ticks.json').write_text(json.dumps(rows,indent=2)+'\n')
