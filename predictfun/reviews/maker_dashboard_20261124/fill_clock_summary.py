"""Condense the user-requested clock diagnosis; source records stay unchanged."""
import gzip,json
from pathlib import Path
import pyarrow.parquet as pq
H=Path(__file__).resolve().parent
MT=Path('/home/yuqing/ctaNew/data/predict_fun/research/maker_trace_v2_20261123')
rows={}
log=H/'fill_clock_probe.log'
source=log.read_text() if log.exists() else gzip.decompress((H/'fill_clock_probe.log.gz').read_bytes()).decode()
for line in source.splitlines():
 for k in ['NATIVE_FRAMES','NATIVE_LEVELS','LIFE_ROWS','PAYLOAD']:
  if line.startswith(k+' '):rows[k]=json.loads(line[len(k)+1:])
out=[]
for f in rows['NATIVE_FRAMES']:
 if not 8.3<=f['rel']<=9.1:continue
 lv=[x for x in rows['NATIVE_LEVELS'] if x['fid']==f['fid']]
 bids=[x for x in lv if x['side']=='bid' and x['shares']>0];asks=[x for x in lv if x['side']=='ask' and x['shares']>0]
 out.append(dict(fid=f['fid'],venue_s=f['rel'],receipt_s=(f['recv_ns']/1e6-1792977000000)/1000,bid=max(x['ptick'] for x in bids),ask=min(x['ptick'] for x in asks),ask_shares={str(x['ptick']):x['shares'] for x in asks if 59<=x['ptick']<=64},pending_asks=f['pend_asks'],status=f['status']))
(H/'fill_clock_evidence.json').write_text(json.dumps(dict(fills=rows['PAYLOAD'],life=rows['LIFE_ROWS'],frames=out),indent=2)+'\n')
print(json.dumps(out,indent=2))
