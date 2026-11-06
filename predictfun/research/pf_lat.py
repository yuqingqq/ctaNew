"""Host latency census: receipt minus source time for Predict books (updateTimestampMs), Binance futures
bookTicker (E event time, T transaction time) and the Predict relay (publish second). One-way path +
clock offset on THIS host; used to re-time the tapes to a hypothetical Tokyo host.
Usage: python3 -m predictfun.research.pf_lat 20261023 20261024 20261025"""
import sys, os
import numpy as np, pandas as pd
from . import pf_io as io
out = {}
for d in sys.argv[1:]:
    S1 = f'{io.OUT}/step1/{d}'
    b = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['t_ms', 'recv_ns']) for f in sorted(os.listdir(f'{S1}/books'))])
    lp = b.recv_ns.values / 1e6 - b.t_ms.values
    bl, tl = [], []
    for h in io.hours_of(d)[::3]:
        df = pd.read_csv(f'{io.BN_DIR}/{h}.csv.gz', header=None, names=['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq'], usecols=['recv', 'E', 'T'])
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        bl.append(df.recv.values / 1e6 - df.E.values); tl.append(df.E.values - df['T'].values)
    bl = np.concatenate(bl); tl = np.concatenate(tl)
    rl = [r[0] / 1e9 - r[1] for h in io.hours_of(d)[::6] for r in io.ref_rows(h) if not r[3]]
    pct = lambda x: dict(zip(['p01', 'p10', 'p50', 'p90', 'p99'], np.percentile(x, [1, 10, 50, 90, 99]).round(1).tolist()))
    out[d] = dict(predict_recv_minus_update_ms=pct(lp), binance_recv_minus_E_ms=pct(bl), binance_E_minus_T_ms=pct(tl),
                  relay_recv_minus_publish_s=pct(np.array(rl)), n=dict(predict=int(len(lp)), binance=int(len(bl)), relay=len(rl)))
    print(d, out[d], flush=True)
io.write_json(f'{io.OUT}/latency_census.json', out)
