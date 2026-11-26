"""Where the smart takers' slippage comes from and what is left after it (USER 2026-11-26: "You mean the smart takers even pay higher than
best price, so he eats the slippage and it still profits"). Readout over stale_quotes' fill rows and stale_paid's rows (aligned as in
stale_paid.merge); no new data read.
Per fill with a best on arrival and a value 1 s later: the identity value = move - slippage - fee, with slippage = d x (price - the best
its order found on arrival) (stale_paid's paid through), fee = 2% x min(p, 100 - p) per share, move = d x (the mid 1 s after the fill - the
best on arrival); share-weighted means per group. Top taker's slippage split by source (declared before the run): quotes posted while its
order waited (stale_quotes' posted_during_hold, incl. levels that appeared within 25 ms); quotes already on the book at arrival and paid
above the arrival best (deeper levels: its order took more than the best level, or others took the best first); at the arrival best or
better. Per taker order (TOP, RANKS2_10): share-weighted slippage and value, distinct price levels taken, the share of orders paying above.
Output (write-once): stale_paid_split.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/stale_paid_split.py [<rehearsal dir>]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import stale_quotes as SQ  # noqa: E402

C, TR, OUT = SQ.C, SQ.TR, SQ.OUT


def main():
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'stale_paid_split.json'
    C.require(not fn.exists(), 'stale_paid_split.json exists (write-once)')
    Fd = pd.concat([pd.read_parquet(OUT / f'{d}_fills.parquet') for d in SQ.DAYS], ignore_index=True)
    P = pd.concat([pd.read_parquet(OUT / f'{d}_paid.parquet') for d in SQ.DAYS], ignore_index=True)
    Fd = Fd[Fd.status != 'outside_path'].reset_index(drop=True)
    C.require(len(Fd) == len(P) and (Fd.t_rel.to_numpy() == P.t_rel.to_numpy()).all() and (Fd.grp.to_numpy() == P.grp.to_numpy()).all(), 'rows do not align')
    Fd['slip'] = P.paid_through.to_numpy()
    Fd['fee'] = .02 * np.minimum(Fd.price, 100. - Fd.price)
    Fd['move'] = Fd.v1 + Fd.fee + Fd.slip
    fresh = Fd.posted_during_hold.astype('boolean').fillna(False).astype(bool) | (Fd.status == 'appeared_late')
    Fd['source'] = np.where(fresh, 'posted while it waited', np.where(Fd.slip > .5, 'deeper levels on the book at arrival', 'at the arrival best or better'))
    out = dict(asof='2026-11-26', groups={}, top_slippage_by_source={}, per_order={})
    for g in SQ.GROUPS:
        X = Fd[(Fd.grp == g) & Fd.slip.notna() & Fd.v1.notna()]
        w = X.q.astype(float)
        m = lambda c: float((X[c] * w).sum() / w.sum())
        o = dict(fills=int(len(X)), shares=float(w.sum()), move_c=m('move'), slippage_c=m('slip'), fee_c=m('fee'), value_c=m('v1'))
        C.require(abs(o['move_c'] - o['slippage_c'] - o['fee_c'] - o['value_c']) < 1e-9, f'{g}: the identity does not hold')
        out['groups'][g] = o
    T = Fd[(Fd.grp == 'TOP') & Fd.slip.notna() & Fd.v1.notna()]
    tw = T.q.sum()
    for k, G in T.groupby('source'):
        out['top_slippage_by_source'][k] = dict(volume_share=float(G.q.sum() / tw), mean_slippage_c=float((G.slip * G.q).sum() / G.q.sum()),
                                                contribution_c=float((G.slip * G.q).sum() / tw), value_c=float((G.v1 * G.q).sum() / G.q.sum()))
    C.require(abs(sum(v['contribution_c'] for v in out['top_slippage_by_source'].values()) - out['groups']['TOP']['slippage_c']) < 1e-9, 'the split does not add up')
    for g in ('TOP', 'RANKS2_10'):
        X = Fd[(Fd.grp == g) & Fd.slip.notna() & Fd.v1.notna()].copy()
        X['ws'], X['wv'] = X.slip * X.q, X.v1 * X.q
        O = X.groupby('order').agg(q=('q', 'sum'), ws=('ws', 'sum'), wv=('wv', 'sum'), levels=('price', 'nunique'))
        O['slip'], O['v1'] = O.ws / O.q, O.wv / O.q
        out['per_order'][g] = dict(orders=int(len(O)), shares_median=float(O.q.median()), levels_median=float(O.levels.median()), multi_level=float((O.levels > 1).mean()),
                                   paid_above=float((O.slip > .5).mean()), slippage_c=SQ.wq(O.slip, np.ones(len(O))), value_c=SQ.wq(O.v1, np.ones(len(O))),
                                   positive_value=float((O.v1 > 0).mean()))
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
