"""MAKER3-BOOK-CONDITIONS v1, stage 1 (outcome-blind): the public Predict top-of-book on a 1 s grid for every BTC 5m market of the
consumed days Oct 24-27, to see how spread and touch depth change toward the window end. Reads only markets_<day> and tob_<day>
(already-built day files of maker_trace_v2; no raw hour file, no fill, no P&L). State at grid point g = ws + k s (k = 0..299) is the last
tob frame (ordered upd_ms, recv_ns) with upd_ms <= g; status: no_frame / frame_status (bad_frame, mid_mismatch, out_of_order, off_grid) /
one_sided / crossed / stale (g - upd_ms > 1000) / ok - the econ module's MID rule. boundary_censored markets are excluded as a status.
Usage: run_light.sh mbc_grid -- python3 predictfun/reviews/maker3_book_conditions_20261125/grid.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
H = Path(__file__).resolve().parent
R = H.parents[2]
V = R / 'data/predict_fun/research/maker_trace_v2_20261123'
OUT = R / 'data/predict_fun/research/maker3_book_conditions_20261125'
DAYS = ('20261024', '20261025', '20261026', '20261027')
BAD = ('bad_frame', 'mid_mismatch', 'out_of_order', 'off_grid')


def day_grid(day):
    m = pd.read_parquet(V / f'markets_{day}.parquet', columns=['mid', 'ws', 'we', 'status'])
    assert m.mid.is_unique and len(m) == 288 and ((m.we - m.ws) == 300).all()
    t = pq.read_table(V / f'tob_{day}.parquet', columns=['mid', 'upd_ms', 'recv_ns', 'best_bid', 'best_bid_sh', 'best_ask', 'best_ask_sh',
                                                         'n_bid', 'n_ask', 'order_count', 'status'], use_threads=False).to_pandas()
    t = t.sort_values(['mid', 'upd_ms', 'recv_ns'], kind='mergesort')
    rows = []
    for r in m.itertuples():
        g = (r.ws * 1000 + 1000 * np.arange(300)).astype(np.int64)
        base = dict(day=day, mid=int(r.mid))
        if 'boundary_censored' in str(r.status):
            rows.append(pd.DataFrame(dict(base, k=np.arange(300), g_ms=g, st='boundary_censored')))
            continue
        f = t[t.mid == r.mid]
        u = f.upd_ms.to_numpy(np.int64)
        i = np.searchsorted(u, g, 'right') - 1
        ok_i = i >= 0
        j = np.where(ok_i, i, 0)
        bid = f.best_bid.to_numpy(float)[j] * 100
        ask = f.best_ask.to_numpy(float)[j] * 100
        fst = f.status.fillna('').to_numpy(str)[j]
        age = g - u[j]
        st = np.full(300, 'ok', dtype=object)
        bad = np.array([any(b in s for b in BAD) for s in fst])
        one = ~(np.isfinite(bid) & np.isfinite(ask))
        crossed = ~one & (bid >= ask)
        st[age > 1000] = 'stale'
        st[crossed] = 'crossed'
        st[one] = 'one_sided'
        st[bad] = 'frame_status'
        st[~ok_i] = 'no_frame'
        rows.append(pd.DataFrame(dict(base, k=np.arange(300), g_ms=g, st=st, age_ms=np.where(ok_i, age, -1),
                                      bid_c=bid, ask_c=ask, bid_sh=f.best_bid_sh.to_numpy(float)[j], ask_sh=f.best_ask_sh.to_numpy(float)[j],
                                      n_bid=f.n_bid.to_numpy(float)[j], n_ask=f.n_ask.to_numpy(float)[j],
                                      order_count=f.order_count.to_numpy(float)[j])))
    return pd.concat(rows, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / 'grid.parquet'
    assert not out.exists(), 'write-once: grid.parquet exists'
    g = pd.concat([day_grid(d) for d in DAYS], ignore_index=True)
    assert len(g) == 4 * 288 * 300
    g.to_parquet(out, index=False)
    print(json.dumps(dict(rows=len(g), status=g.st.value_counts().to_dict())))


if __name__ == '__main__':
    main()
