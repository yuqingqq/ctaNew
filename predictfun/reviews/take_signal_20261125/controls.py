"""TAKE-SIGNAL v1 amendment 1 controls: placebo times, +7 s shift, opposite direction, very slow clock (amendment_1_controls.md).
The valuation function below must first reproduce takes.parquet (tokyo and slow) exactly; only then are the controls valued.
Usage: run_light.sh ts_ctrl -- python3 predictfun/reviews/take_signal_20261125/controls.py
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import take_eval as TE  # noqa: E402  (helpers only: frame lookup / mid rule / side price)

OUT = TE.OUT
Q = 50
VCOLS = ['mark_1s', 'mark_3s', 'mark_10s', 'mark_30s', 'rt_1s', 'rt_3s', 'rt_10s', 'end', 'res']


def value(T, midx_v, r, ti, d, c):
    """The take_eval valuation for given trigger times ti and directions d in one market (row r of markets_<day>)."""
    ws, end = int(r.ws) * 1000, int(r.we) * 1000 - 2000
    n = len(ti)
    midx = np.full(n, midx_v, np.int64)
    upb = d == 1
    xk = ti - c['p_in']
    ik = TE.frame_last(T, midx, xk)
    kmid, kok = TE.mid_ok(T, ik, xk)
    kpx, _ = TE.side_px(T, ik, upb)
    ta = ti + c['arrive']
    f1, f2 = TE.frame_last(T, midx, ta), TE.frame_next(T, midx, ta)
    p1, s1 = TE.side_px(T, f1, upb)
    p2, s2 = TE.side_px(T, f2, upb)
    lim = kpx
    exe = np.where(upb, p1 <= lim + 1e-9, p1 >= lim - 1e-9) & np.where(upb, p2 <= lim + 1e-9, p2 >= lim - 1e-9)
    st = np.full(n, 'filled', dtype=object)
    st[~exe] = 'not_executable'
    st[f2 < 0] = 'no_next_frame'
    st[~kok | ~np.isfinite(kpx)] = 'no_known_book'
    avail = np.where(st == 'filled', np.minimum(s1, s2), 0.0)
    fee = r.feeRateBps / 1e4 * np.minimum(lim, 1 - lim)
    vv = {'Up': 1.0, 'Down': 0.0}.get(str(r.outcome), np.nan)
    out = dict(t=ti, dir=d, status=st, avail_sh=avail)
    for h in TE.HORIZONS:
        x = ta + h
        ih = TE.frame_last(T, midx, x)
        mh, okh = TE.mid_ok(T, ih, x)
        cen = x > end
        out[f'mark_{h // 1000}s'] = np.where(cen | ~okh, np.nan, d * (mh - lim) - fee)
        ex, _ = TE.side_px(T, ih, ~upb)
        exf = r.feeRateBps / 1e4 * np.minimum(ex, 1 - ex)
        out[f'rt_{h // 1000}s'] = np.where(cen | ~okh | ~np.isfinite(ex), np.nan, d * (ex - lim) - fee - exf)
    ie = TE.frame_last(T, midx, np.full(n, end))
    me, oke = TE.mid_ok(T, ie, np.full(n, end))
    out['end'] = np.where(oke, d * (me - lim) - fee, np.nan)
    out['res'] = d * (vv - lim) - fee
    return pd.DataFrame(out)


def summ(df):
    sh = np.minimum(Q, df.avail_sh)
    o = dict(triggers=len(df), fill_rate=float((df.status == 'filled').mean()), filled=int((df.status == 'filled').sum()))
    for v in VCOLS:
        ok = df[v].notna() & (sh > 0)
        o[f'{v}_c'] = float(100 * (df[v][ok] * sh[ok]).sum() / sh[ok].sum()) if ok.any() else None
    return o


def main():
    t0 = time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    tk = pd.read_parquet(OUT / 'takes.parquet')
    rng = np.random.default_rng(20261125)
    res, parity_bad, parity_n = {}, 0, 0
    arms = {k: [] for k in ('placebo', 'shift7', 'opposite', 'very_slow')}
    vs = dict(b_in=10, p_in=200, arrive=150)
    for day in TE.DAYS:
        mk = pd.read_parquet(TE.V / f'markets_{day}.parquet', columns=['mid', 'ws', 'we', 'outcome', 'feeRateBps', 'status'])
        mindex = {int(m): i for i, m in enumerate(sorted(mk.mid))}
        T = TE.load_tob(day, mindex)
        mk = mk[~mk.status.astype(str).str.contains('boundary_censored')].set_index('mid')
        for (clk, mid), g in tk[tk.day == day].groupby(['clock', 'mid']):
            r = mk.loc[mid]
            r = r.rename(None)
            row = type('R', (), dict(ws=r.ws, we=r.we, outcome=r.outcome, feeRateBps=r.feeRateBps))
            g = g.sort_values('t')
            v = value(T, mindex[int(mid)], row, g.t.to_numpy(np.int64), g.dir.to_numpy(np.int64), TE.CLOCKS[clk])
            for col in ['avail_sh'] + VCOLS:
                a, b = v[col].to_numpy(float), g[col].to_numpy(float)
                parity_bad += int((~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-12))).sum())
            parity_bad += int((v.status.to_numpy() != g.status.to_numpy()).sum())
            parity_n += len(g)
            if clk != 'tokyo':
                continue
            ws, end = int(r.ws) * 1000, int(r.we) * 1000 - 2000
            c = TE.CLOCKS['tokyo']
            pt = rng.integers(ws, end - 30000, size=len(g))
            arms['placebo'].append(value(T, mindex[int(mid)], row, np.sort(pt), g.dir.to_numpy(np.int64)[np.argsort(pt)], c).assign(day=day))
            s7 = g.t.to_numpy(np.int64) + 7000
            k7 = s7 < end
            if k7.any():
                arms['shift7'].append(value(T, mindex[int(mid)], row, s7[k7], g.dir.to_numpy(np.int64)[k7], c).assign(day=day))
            arms['opposite'].append(value(T, mindex[int(mid)], row, g.t.to_numpy(np.int64), -g.dir.to_numpy(np.int64), c).assign(day=day))
            arms['very_slow'].append(value(T, mindex[int(mid)], row, g.t.to_numpy(np.int64) - 5 + vs['b_in'], g.dir.to_numpy(np.int64), vs).assign(day=day))
        del T
    res['parity'] = dict(rows=parity_n, mismatched_fields=parity_bad, ok=parity_bad == 0)
    assert parity_bad == 0, f'valuation parity failed: {parity_bad} fields'
    for k, lst in arms.items():
        df = pd.concat(lst, ignore_index=True)
        res[k] = dict(all=summ(df), by_day={d: summ(x) for d, x in df.groupby('day')})
    res['primary_tokyo'] = summ(tk[tk.clock == 'tokyo'])
    res['readings'] = dict(placebo_negative=bool(res['placebo']['all']['mark_3s_c'] < 0), opposite_negative=bool(res['opposite']['all']['mark_3s_c'] < 0),
                           shift7_not_positive=bool(res['shift7']['all']['mark_3s_c'] <= 0),
                           very_slow_mark_3s_c=res['very_slow']['all']['mark_3s_c'], very_slow_fill_rate=res['very_slow']['all']['fill_rate'])
    res['cpu_s'] = round(time.process_time() - t0, 2)
    (H / 'controls.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
    print(json.dumps({k: (v['all'] if isinstance(v, dict) and 'all' in v else v) for k, v in res.items()}, indent=1, default=float))


if __name__ == '__main__':
    main()
