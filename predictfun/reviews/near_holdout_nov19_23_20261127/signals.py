"""Rebuild the frozen BTC state and pressure change points for named dates only."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO/'predictfun/research'))
import pf_mt_btc as B
from pf_mt_decision import prev_day

CONTROL = '20261026'
DAYS = tuple(f'202611{d}' for d in range(19,24))
OUT = REPO/'data/predict_fun/research/near_holdout_nov19_23'
BTC = OUT/'btc'


def build(day):
    assert day in (CONTROL, *DAYS)
    BTC.mkdir(parents=True, exist_ok=True)
    roots = B.Roots()
    opening = B.day_ms(day)
    _,starts = B.ledger_info(roots.ledger,(opening-B.HOUR_MS-60000)*10**6,(opening+B.DAY_MS+60000)*10**6)
    stems = [prev_day(day)+'_23'] + [f'{day}_{h:02d}' for h in range(24)]
    acc = B.BnDay(opening,starts)
    records = []
    for i,stem in enumerate(stems):
        rec,raw = B.load(roots.bn/f'{stem}.csv.gz')
        assert rec['state']=='read',stem
        a,_ = B.parse_bn(raw)
        del raw
        ok = np.isfinite(a['bid']) & np.isfinite(a['ask']) & (a['bid']>0) & (a['ask']>0)
        if not ok.all():
            a = {k:v[ok] for k,v in a.items()}
        rec.update(acc.add(i,a))
        records.append(dict(stem=stem, **rec))
        if i==0 and day in (CONTROL,DAYS[0]):
            padding = B.BnDay(opening-B.DAY_MS,starts)
            padding.add(0,a)
            pad = table(padding, B.bn_holes(records,padding.starts))
            pq.write_table(pad.slice(23*36000),BTC/f'btc_{prev_day(day)}.parquet')
        del a
    p = BTC/f'btc_{day}.parquet'
    tab = table(acc,B.bn_holes(records,acc.starts))
    assert len(tab)==864000
    if day==CONTROL:
        pd.testing.assert_frame_equal(tab.to_pandas(),pq.read_table(B.ROOT_V2/'btc'/p.name).to_pandas(),check_exact=True)
    pq.write_table(tab,p)
    from importlib import import_module
    sys.path.insert(0,str(HERE.parent/'take_exit_replay_20261125'))
    od = import_module('other_days')
    pressure = OUT/f'pressure_{day}.parquet'
    if pressure.exists():
        assert B.sha_file(pressure)==json.loads((OUT/f'pressure_{day}.json').read_text())['out_sha256']
        changes = pd.read_parquet(pressure)
        k,m = changes.k_ms.to_numpy(),changes.mid.to_numpy()
    else:
        k,m = od.build_changes(stems,{},pressure, OUT/f'pressure_{day}.json')
    if day==CONTROL:
        oldk,oldm = od.MM.bn()
        sel = (k>=opening-1000)&(k<opening+B.DAY_MS-1000)
        old = (oldk>=opening-1000)&(oldk<opening+B.DAY_MS-1000)
        assert np.array_equal(k[sel],oldk[old]) and np.array_equal(m[sel],oldm[old])
    (OUT/f'grid_{day}.json').write_text(json.dumps(dict(inputs={r['stem']:r['sha256'] for r in records},grid_sha=B.sha_file(p),pressure_sha=B.sha_file(pressure)),indent=2))
    print(day,'BTC grid and pressure archived',flush=True)


def table(acc, holes):
    g,bid,ask,mid,updates,age,code = acc.finalize(holes)
    return pa.Table.from_arrays([pa.array(v) for v in (g,bid,ask,mid,updates,age)] +
                                [pc.take(pa.array(B.STATUS),pa.array(code))],schema=B.BTC_SCHEMA)


if __name__=='__main__':
    build(sys.argv[1])
