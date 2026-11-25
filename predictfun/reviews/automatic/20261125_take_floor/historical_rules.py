"""Inspect the archived rules, including a tie, without reading any forward day."""
from pathlib import Path
import json,sys
import pandas as pd
R=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(R/'predictfun/research'))
import pf_io
day='20261023';mid=3667409
reg=pd.read_parquet(R/f'data/predict_fun/research/step1/{day}/registry.parquet').set_index('mid').loc[mid]
hour=pd.Timestamp(reg.ws,unit='s',tz='UTC').hour
for _,__,___,pl in pf_io.rest_rows(f'{day}_{hour:02d}','/v1/categories/'):
    c=pl.get('data') or {}
    for m in c.get('markets') or []:
        if m['id']==mid:
            print(json.dumps(dict(mid=mid,start=reg.start_price,end=reg.end_price,category=c.get('description'),market=m.get('description'),
                                  market_keys=sorted(m),category_keys=sorted(c),vd=m.get('variantData')),indent=2))
            sys.exit(0)
raise SystemExit('category unavailable')
