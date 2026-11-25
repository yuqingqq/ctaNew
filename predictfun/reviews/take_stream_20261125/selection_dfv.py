"""Follow-up to selection.py: the repricing a Binance move implies, dFV = FV(now) - FV(250 ms ago) in the take's direction (cents), and the
standardised move r250 / (sigma sqrt(0.25 s)); value after fee by those buckets for WINNERS and OURS. Reads OUT/selection_fills.parquet
only. Output (write-once): selection_dfv.json.
Usage: PF_SLOT=claude predictfun/research/run_light.sh selection_dfv -- python3 predictfun/reviews/take_stream_20261125/selection_dfv.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import stream_replay as S  # noqa: E402

DFV_BINS = [-np.inf, 0., 1., 2., 4., 8., np.inf]
ZM_BINS = [-np.inf, 2., 4., 8., 16., np.inf]


def main():
    fn = H / 'selection_dfv.json'
    S.C.require(not fn.exists(), 'selection_dfv.json exists (write-once)')
    D = pd.read_parquet(S.OUT / 'selection_fills.parquet')
    d = np.where(D.edge_c.notna(), 1, 1)                                         # placeholder, replaced below
    # direction: r250 is already signed in the take's direction; x_bps is BTC's move since the window start (unsigned by direction)
    # FV(now) uses x; FV(250 ms ago) uses x - dir * r250. dir = +1 when fv - price + fee = edge (a buy), -1 otherwise.
    fee = 2. * np.minimum(D.price, 100. - D.price) / 100.
    dirn = np.where(np.isclose(D.fv - D.price - fee, D.edge_c, atol=1e-6), 1, -1)
    ssq = D.sigma.to_numpy(float) * np.sqrt(D.tau.to_numpy(float))
    zn = D.x_bps.to_numpy(float) / ssq
    zb = (D.x_bps.to_numpy(float) - dirn * D.r250.to_numpy(float)) / ssq
    Phi = np.vectorize(lambda z: 0.5 * (1. + math.erf(z / math.sqrt(2.))) if np.isfinite(z) else np.nan)
    D['dfv'] = dirn * 100. * (Phi(zn) - Phi(zb))
    D['zmove'] = D.r250 / (D.sigma * math.sqrt(0.25))
    D['dir'] = dirn

    def wavg(G, c):
        x = G[c].to_numpy(float); w = G.shares.to_numpy(float); ok = np.isfinite(x)
        return float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else None

    def wq(G, c, qs=(.25, .5, .75)):
        x = G[c].to_numpy(float); w = G.shares.to_numpy(float); ok = np.isfinite(x)
        o = np.argsort(x[ok]); cw = np.cumsum(w[ok][o]) / w[ok].sum()
        return [float(x[ok][o][min(np.searchsorted(cw, q), ok.sum() - 1)]) for q in qs]
    out = dict(asof='2026-11-25', quantiles={}, by_dfv={}, by_zmove={})
    for g in ('winners', 'ours'):
        X = D[D.group == g]
        out['quantiles'][g] = dict(dfv=wq(X, 'dfv'), zmove=wq(X, 'zmove'))
        for name, col, bins in (('by_dfv', 'dfv', DFV_BINS), ('by_zmove', 'zmove', ZM_BINS)):
            Y = X.assign(b=pd.cut(X[col], bins))
            out[name][g] = {str(b): dict(share=float(G.shares.sum() / X.shares.sum()), fills=int(len(G)), v1=wavg(G, 'v1'), v10=wavg(G, 'v10'), vset=wavg(G, 'vset'))
                            for b, G in Y.groupby('b', observed=True)}
    S.TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
