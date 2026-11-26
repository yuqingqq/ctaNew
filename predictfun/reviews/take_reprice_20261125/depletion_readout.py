"""Readout of depletion_cause.py's per-point split (OUT/depletion_cause_points.parquet): (1) the traded share of the shrink on points whose
opposite level was still there at the send (same) vs gone; (2) how often the level at the send goes during the hold after one already went
in the 300 ms before; (3) the change of the best-level imbalance over the 300 ms split exactly into the opposite-size and the near-size
change (Shapley: mean of the two orders), f(near, opp) = (near - opp) / (near + opp). Point estimates, four consumed days.
Output (write-once): depletion_readout.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/depletion_readout.py
"""
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import depletion_cause as DC  # noqa: E402

C, TR = DC.C, DC.TR


def imb(n, o):
    return np.where(n + o > 0, (n - o) / np.maximum(n + o, 1e-12), np.nan)


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (300, 301))
    fn = H / 'depletion_readout.json'
    C.require(not fn.exists(), 'depletion_readout.json exists (write-once)')
    X = pd.read_parquet(DC.OUT / 'depletion_cause_points.parquet')
    for c in ('pre_Q', 'pre_left', 'pre_V', 'pre_W', 'near_a', 'near_b', 'opp_b'):
        X[c] = X[c].astype(float)
    out = dict(asof='2026-11-26', points=int(len(X)), groups={})
    for g in DC.GROUPS:
        o = {}
        for y, name in ((1, 'send'), (0, 'null')):
            Z = X[(X.grp == g) & (X.y == y)]
            r = {}
            for st in (DC.SAME, DC.GONE):
                A = Z[Z.pre_st == st]
                sh = A.pre_Q - A.pre_left
                C.require(np.allclose(sh, A.pre_V + A.pre_W, atol=1e-3), f'{g} {name}: shrink != traded + withdrawn')
                r[DC.ST_NAMES[st]] = dict(n=int(len(A)), shrink_btc=float(sh.sum()), traded_btc=float(A.pre_V.sum()), withdrawn_btc=float(A.pre_W.sum()),
                                          traded_share=float(A.pre_V.sum() / sh.sum()) if sh.sum() > 0 else None)
            sp = Z[Z.pre_st.isin([DC.SAME, DC.GONE]) & Z.hold_st.isin([DC.SAME, DC.GONE])]
            pg, hg = sp.pre_st == DC.GONE, sp.hold_st == DC.GONE
            r['ticks'] = dict(n=int(len(sp)), pre_gone=float(pg.mean()), hold_gone=float(hg.mean()),
                              hold_gone_if_pre_gone=float(hg[pg].mean()) if pg.any() else None, hold_gone_if_pre_same=float(hg[~pg].mean()) if (~pg).any() else None)
            A = Z[Z.pre_st.isin([DC.SAME, DC.GONE])]
            na, nb, oa, ob = A.near_a.to_numpy(), A.near_b.to_numpy(), A.pre_Q.to_numpy(), A.opp_b.to_numpy()
            f0, f1 = imb(na, oa), imb(nb, ob)
            d_opp = .5 * ((imb(na, ob) - f0) + (f1 - imb(nb, oa)))
            d_near = .5 * ((imb(nb, oa) - f0) + (f1 - imb(na, ob)))
            C.require(np.nanmax(np.abs(d_opp + d_near - (f1 - f0))) < 1e-9, 'the imbalance split does not add up')
            r['imbalance'] = dict(n=int(len(A)), mean_start=float(np.nanmean(f0)), mean_send=float(np.nanmean(f1)), mean_change=float(np.nanmean(f1 - f0)),
                                  from_opposite=float(np.nanmean(d_opp)), from_near=float(np.nanmean(d_near)))
            o[name] = r
        out['groups'][g] = o
    TR.dump(fn, out)
    import json
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
