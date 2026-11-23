"""Independent deduplication of later takes for response and six-chain scopes."""
import json, hashlib
from pathlib import Path
import pandas as pd
R=Path('/home/yuqing/ctaNew'); P=Path('/tmp/pf-competing-review-4eab1efe')
B=Path('predictfun/reviews/maker_trace_20261122/v2/chains'); H=Path(__file__).resolve().parent
C=json.loads((P/B/'continuation_econ/claims.json').read_text())
rec=json.loads((P/B/'continuation_econ/receipt.json').read_text())
ep=R/rec['inputs']['events']['path']
assert hashlib.sha256(ep.read_bytes()).hexdigest()==rec['inputs']['events']['sha256']
T=pd.read_parquet(ep);T['addr']=T.addr.str.lower();T['day']=T.day.astype(str)
L=pd.read_csv(P/B/'continuation_econ/links_econ.csv',float_precision='round_trip'); L['day']=L.day.astype(str)
out={}
for scope,mask in [('response_all',L.kind=='response'),('chains',L.kind=='chain'),('union',L.kind.notna())]:
    keys=set()
    for r in L[mask & (L.takes_status=='evaluated')].itertuples():
        s=T[(T.day==r.day)&(T.mid==r.mid)&(T.addr==r.addr.lower())&(T.x_lo_hard_ms>r.anchor_ms)&(T.hi_ms<=r.end_ms)]
        keys.update(zip(s.day,s.mid,s.addr,s.sid))
    expected=C['pooled'][scope]['takes']['windows']['END']
    assert len(keys)==expected['n_distinct'],(scope,len(keys),expected)
    out[scope]={'independent_take_count':len(keys),'fifo_close_keys':expected['n_fifo_close_keys'],'fifo_closing_shares':expected['fifo_closing_sh']}
(H/'take_scopes.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
