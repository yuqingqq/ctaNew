"""Describe BTC over [120,180] seconds and the already highlighted 30s moves.
Use the 100ms Binance chart clock. Compare median absolute endpoint change
and high-low range with the same midpoint minute in other registry markets on
the selected dates. Require a complete ok grid; missing samples are exclusions.
Public price-selected examples, descriptive only. No quote-decision inference.
"""
import base64
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

folder = Path(__file__).resolve().parent
charts = folder.parent / 'maker_dashboard_20261124/markets'
selected = json.loads((charts/'selection.json').read_text())
registry = pd.read_csv(folder.parent/'maker3_market_coverage_20261124/markets.csv')
btc = folder.parents[2]/'data/predict_fun/research/maker_trace_v2_20261123/btc'
dates = registry.loc[registry.mid.isin(map(int, selected)), 'day'].unique()


def metrics(prices):
    if not np.isfinite(prices).all():
        return (np.nan, np.nan)
    return (10000 * (prices[-1]-prices[0])/prices[0],
            10000 * (prices.max()-prices.min())/prices[0])


assert metrics(np.array([100., 101., 100.])) == (0.,100.)
assert np.isnan(metrics(np.array([100.,np.nan,100.]))[1])
rows = []
for day in sorted(dates):
    tape = pd.read_parquet(btc/f'btc_{day}.parquet', columns=['g_ms','mid','status'])
    assert np.all(np.diff(tape.g_ms) == 100)
    px = tape.mid.where(tape.status.eq('ok')).to_numpy()
    for r in registry[registry.day.eq(day)].itertuples():
        start = (1000*r.ws - tape.g_ms.iloc[0])//100
        assert tape.g_ms.iloc[start] == 1000*r.ws
        window = px[start+1200:start+1801]
        assert len(window) == 601
        net, span = metrics(window)
        row = dict(day=day,mid=r.mid,selected=str(r.mid) in selected,mid_net=net,mid_range=span)
        if row['selected']:
            move = selected[str(r.mid)]
            market = json.loads(gzip.decompress(base64.b64decode((charts/f'{r.mid}.js').read_text().split('"')[1])))[0]
            assert market['ws'] == r.ws*1000 and market['trend'] == move
            times, _, plotted = zip(*market['btc'])
            ix = np.rint(10*np.array(times)).astype(int)
            assert np.allclose(px[start+ix], np.array(plotted,dtype=float), equal_nan=True,atol=1e-6,rtol=0)
            lo,hi = move['start'],move['end']
            trend_net,trend_range=metrics(px[start+10*lo:start+10*hi+1])
            row.update(start_s=lo,end_s=hi,yes_c=move['move'],trend_net=trend_net,trend_range=trend_range,
                       open_to_150_bps=metrics(px[start+np.array([0,1500])])[0])
        rows.append(row)
table = pd.DataFrame(rows)
subset = table[table.selected]
assert len(subset)==7
subset.to_csv(folder/'btc_midpoint.csv',index=False)
for name,g in [('selected',subset),('other_same_dates',table[~table.selected])]:
    valid=g.dropna(subset=['mid_net','mid_range'])
    print(name,len(g),len(valid),'median abs net / range (bps)',valid.mid_net.abs().median(),valid.mid_range.median())
print(subset.round(2).to_string(index=False))
