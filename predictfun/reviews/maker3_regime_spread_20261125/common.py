import gzip,hashlib,json,math,resource,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
H=Path(__file__).resolve().parent;R=H.parents[2];D=R/'data/predict_fun/research';P=H.parent/'maker3_spread_risk_20261124';PAY=D/'maker3_spread_risk_20261124/data.json.gz';OUT=D/'maker3_regime_spread_20261125'
PANEL=H.parent/'maker_trace_20261122/v2/chains/take_logic_v2/panel.parquet';REF=H.parent/'maker_trace_20261122/v2/matched_gap_r1/per_market.csv'
CLOCKS=('pre','upper','lag1');ASSETS=('pf','btc');VOLS=('calm','typical','volatile','unknown');SPREADS=('narrow','intermediate','wide','unknown')
pa.set_cpu_count(1);pa.set_io_thread_count(1)
def limits():resource.setrlimit(resource.RLIMIT_CPU,(90,91))
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def dump(p,x):Path(p).write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def refs():
 x=pd.read_csv(REF,usecols=['day','m','party','view','start_ms','end_ms','end_mark'],dtype={'day':str},float_precision='round_trip');x=x[(x.party=='rank3')&(x['view']=='zero')]
 assert len(x)==19 and x.m.nunique()==19;return x
def neutral():
 x=pd.read_parquet(PANEL,filters=[('rank','=',3)]).sort_values(['day','mid','k']).reset_index(drop=True);x.day=x.day.astype(str)
 assert len(x)==5700 and not x.duplicated(['day','mid','k']).any();return x
def features(p):
 rows=[]
 for (day,mid),g in p.groupby(['day','mid'],sort=True):
  assert np.array_equal(g.k,np.arange(300));assert np.array_equal(np.diff(g.t_ms),np.full(299,1000))
  mids=(g.bid0+g.ask0)*50;delta=mids.diff();valid=g.kok0.astype(bool)&np.isfinite(mids)
  pf=np.sqrt(delta.pow(2).rolling(10,min_periods=10).mean());pf[valid.astype(int).rolling(11,min_periods=11).sum()!=11]=np.nan
  btc=np.sqrt(g.r_1s0.pow(2).rolling(10,min_periods=10).mean());btc[g.bok0.astype(int).rolling(10,min_periods=10).sum()!=10]=np.nan;btc[g.k<10]=np.nan
  for i,r in enumerate(g.itertuples()):rows.append(dict(day=day,mid=int(mid),k=int(r.k),t_ms=int(r.t_ms),pf=float(pf.iloc[i]),btc=float(btc.iloc[i]),spread=(r.ask0-r.bid0)*100 if r.kok0 else np.nan))
 return pd.DataFrame(rows)
def boundaries(f):
 train=f[f.day.isin(['20261024','20261025'])];out={}
 for a in ASSETS:
  v=train[a].dropna();q=np.quantile(v,[.25,.75]);out[a]=dict(q25=float(q[0]),q75=float(q[1]),n=len(v),unit='cents_per_1s_change_rms' if a=='pf' else 'bps_per_1s_return_rms')
 return out
def vb(x,b):return 'unknown' if x is None or not np.isfinite(x) else 'calm' if x<=b['q25'] else 'typical' if x<=b['q75'] else 'volatile'
def sb(x):return 'unknown' if x is None or not np.isfinite(x) else 'narrow' if x<=1+1e-7 else 'intermediate' if x<=4+1e-7 else 'wide'
def clock_ms(f,c):return f['hi_ms'] if c=='upper' else None if f['lo_ms'] is None else f['lo_ms']-(1 if c=='pre' else 1000)
def context(f,ws,rows,clock,asset,bounds):
 t=clock_ms(f,clock);k=math.floor((t-ws)/1000) if t is not None else -1
 r=rows.get(k);v=float(r[asset]) if r is not None else None
 assert r is None or r['t_ms']<=t
 return dict(anchor_ms=int(r['t_ms']) if r is not None else None,value=v,vol=vb(v,bounds[asset]),spread=sb(f['book'][clock]['spread']))
def bind():
 m=json.loads((H/'manifest.json').read_text())
 for fn,h in m['inputs'].items():assert sha(R/fn)==h,fn
 for fn,h in m['sources'].items():
  assert sha(R/fn)==h,fn
  assert subprocess.check_output(['git','show','HEAD:'+fn],cwd=R)==(R/fn).read_bytes(),fn
 return m
def loaded():return json.loads(gzip.decompress(PAY.read_bytes()))
