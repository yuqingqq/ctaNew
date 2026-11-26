"""How is the Binance move after the top taker's send produced: one large aggressive order, or many small ones? (USER 2026-11-26: "any
other more information we can find"; follow-up of state_value.py: 41% of its decisions come with a thick far side (> 3 BTC) where a random
moment is followed by a move within 100 ms 0.3% of the time, yet it is right 33% of the time). Readout; four consumed days.
Points from state_value_points.parquet, no move their way in the previous 300 ms, the next move their way within 300 ms: the top taker's
decisions (t = its arrival at Predict) and random times (both directions). For each, the aggressive trades their way in (t, move] (trade
stream on E, de-duplicated by id), grouped by ms (one ms group ~ one sweep): number of groups, total BTC, the largest group (BTC, its time
after t and its share of the far-side best size at t), the group at the move; split by the far-side best size at t (<= 1 / 1-3 / > 3 BTC).
A move with no trade their way = the far side was cancelled away. Fingerprint checks: the largest group's size distribution (share at
round sizes 0.5 / 1 / 2 / 5 / 10 BTC +- 0.5 %) and its time after t (decisions only).
Output (write-once): sweep_print.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/sweep_print.py [dir]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import depletion_cause as DC  # noqa: E402

C, TR, WT = DC.C, DC.TR, DC.WT
QS = (.1, .25, .5, .75, .9)
ROUND = (.5, 1., 2., 5., 10.)
FAR_B = (('<=1', -1., 1.), ('1-3', 1., 3.), ('>3', 3., 1e9))


def qs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {f'p{int(q * 100)}': float(np.quantile(x, q)) for q in QS} if len(x) else None


def groups(tr, side):
    """Per-ms aggressive groups of one side: (E, BTC) sorted by E."""
    T = tr[tr.m == side]
    g = T.groupby('E').q.sum()
    return g.index.to_numpy(float), g.to_numpy(float)


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'sweep_print.json'
    C.require(not fn.exists(), 'sweep_print.json exists (write-once)')
    X = pd.read_parquet(WT.OUT / 'state_value_points.parquet')
    X = X[~X.momentum & (X.next_move <= 300.)].copy()
    X['move'] = X.t + X.next_move
    rows = []
    for day in WT.DAYS:
        Xd = X[X.day == day]
        if not len(Xd):
            continue
        tr = DC.read_trades(day).drop_duplicates('id')
        G = {s: groups(tr, s) for s in (DC.BUY_AGGR, 1 - DC.BUY_AGGR)}
        for r in Xd.itertuples():
            E, Q = G[DC.BUY_AGGR if r.d > 0 else 1 - DC.BUY_AGGR]
            i0, i1 = np.searchsorted(E, r.t, 'right'), np.searchsorted(E, r.move, 'right')
            e, q = E[i0:i1], Q[i0:i1]
            k = int(np.argmax(q)) if len(q) else -1
            rows.append(dict(day=day, kind=r.kind, far=r.far, lag=r.next_move, n_groups=len(q), total=float(q.sum()), max_g=float(q[k]) if len(q) else 0.,
                             max_after_t=float(e[k] - r.t) if len(q) else np.nan, at_move=float(q[-1]) if len(q) and e[-1] == r.move else 0.))
        print(day, len(Xd), flush=True)
    R = pd.DataFrame(rows)
    R['max_over_far'] = R.max_g / R.far
    out = dict(asof='2026-11-26', n={k: int((R.kind == k).sum()) for k in ('send', 'random')}, by_far={})
    for fl, lo, hi in FAR_B + (('all', -1., 1e9),):
        o = {}
        for kind in ('send', 'random'):
            Z = R[(R.kind == kind) & (R.far > lo) & (R.far <= hi)]
            if not len(Z):
                continue
            o[kind] = dict(n=int(len(Z)), no_trade=float((Z.n_groups == 0).mean()), one_group=float((Z.n_groups == 1).mean()), groups=qs(Z.n_groups),
                           total_btc=qs(Z.total), max_group_btc=qs(Z.max_g), max_over_far=qs(Z.max_over_far), max_ge_far=float((Z.max_g >= Z.far - 1e-9).mean()),
                           move_lag_ms=qs(Z.lag), max_group_after_t_ms=qs(Z.max_after_t),
                           round_size_share={str(v): float((np.abs(Z.max_g - v) <= .005 * v).mean()) for v in ROUND})
        out['by_far'][fl] = o
    TR.dump(fn, out)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == '__main__':
    main()
