"""Did the takers who beat our 172 ms takes act on Binance futures information or before it? Diagnostic on the consumed v3 days.
Same consuming legs as takers.py (the aimed level, at or through the limit, printed between the decision's known frame and the vanish
+ 100 ms), websocket-timed only (how2 == 1); their arrival at Predict A = print - 150 ms (the hold), an upper bound on when they decided.
Per arrival, from the Binance futures mid change points (event time): the move over (A - 250 ms, A] signed in our take's direction,
whether any mid change happened in that window, the age of the last change, and - for arrivals with no prior move - how long until the
first futures change in our direction after A. No REST read. Output (write-once): takers_timing.json.
Usage: PF_SLOT=claude predictfun/research/run_light.sh takers_timing -- python3 predictfun/reviews/take_exit_venue_20261125/takers_timing.py
"""
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import venue_common as V  # noqa: E402
import who_took as WT  # noqa: E402

F, OD, MM, TR, C = V.F, V.OD, V.MM, V.TR, V.C
S1D, HOLD = WT.S1D, 150.


def main():
    fn = H / 'takers_timing.json'
    C.require(not fn.exists(), 'takers_timing.json exists (write-once)')
    sm = json.loads((S1D / 'summary.json').read_text())
    C.require(sm['delay_ms'] == 172., 'not the 172 ms run')
    want = collections.defaultdict(set)
    for r in sm['rows']:
        want[r['day']].add(int(r['mid']))
    src, rows = MM.KP.Tape(), []
    for day in OD.DAYS3:
        L = pd.read_parquet(MM.S1 / day / 'legs_timed2.parquet', columns=['mid', 'side', 'ptick', 'shares', 't_ms2', 'how2'])
        L = L[L.mid.isin(want[day]) & (L.how2 == 1)]
        by_mid = {m: g for m, g in L.groupby('mid')}
        bk, bm = OD.seg_arrays(OD.SEG_OF[day]); bk = bk.astype(float)
        reg = MM.registry(day).set_index('mid'); off = src.offset(day)
        for h, g in reg.groupby('hour'):
            mids = [int(m) for m in g.index if int(m) in want[day]]
            if not mids:
                continue
            for m, st, p, info, yup in F.worlds_yup(day, mids, src, off, reg):
                C.require(st == 'complete', f'{m} {st}')
                lg = by_mid.get(m)
                if lg is None:
                    continue
                t0 = float(info['t_start']); f = np.asarray(p['frames'], float)
                pt = lg.t_ms2.values.astype(float) - t0
                led = json.loads(gzip.decompress((S1D / f'{m}_BASE.json.gz').read_bytes()))
                for x in led['ioc_results']:
                    if x['reason'] != 'limit_moved':
                        continue
                    fo = WT.follow(p, x)
                    if fo['gone_ms'] is None:
                        continue
                    s, Lp, gd = int(x['side']), float(x['limit']), float(x['submit'])
                    opp = 1 - s; jk = int(np.searchsorted(f[:, 1], gd, side='right') - 1)
                    sel = (lg.side.values == opp) & ((lg.ptick.values >= Lp) if opp == 1 else (lg.ptick.values <= Lp)) & (pt >= f[jk, 0]) & (pt <= gd + fo['gone_ms'] + 100.)
                    d = 1. if s == 0 else -1.
                    for i in np.flatnonzero(sel):
                        A = t0 + pt[i] - HOLD
                        j0 = int(np.searchsorted(bk, A, 'right') - 1); j1 = int(np.searchsorted(bk, A - 250., 'right') - 1)
                        if j0 < 0 or j1 < 0:
                            continue
                        r250 = d * 1e4 * np.log(bm[j0] / bm[j1])
                        changed = j0 > j1                                           # a mid change point inside (A - 250, A]
                        nxt = None
                        if r250 <= 0:                                               # first futures change in our direction after A
                            k = j0 + 1
                            while k < len(bk) and bk[k] <= A + 2000.:
                                if d * (bm[k] - bm[j0]) > 0:
                                    nxt = float(bk[k] - A); break
                                k += 1
                        rows.append(dict(shares=float(lg.shares.values[i]), r250=float(r250), changed=bool(changed), last_age_ms=float(A - bk[j0]),
                                         next_our_way_ms=nxt, rel_decision_ms=float(A - (t0 + gd))))
        print(day, len(rows), flush=True)
    D = pd.DataFrame(rows)
    cls = np.where(D.r250 > 1e-9, 'moved_our_way', np.where(D.r250 < -1e-9, 'moved_against', np.where(D.changed, 'changed_no_net_move', 'no_futures_change_250ms')))
    D['cls'] = cls
    out = dict(asof='2026-11-25', source='stage-1 172 ms BASE missed takes; websocket-timed consuming legs; Binance futures mid change points',
               legs=len(D), shares=float(D.shares.sum()), classes={})
    for c in ('moved_our_way', 'changed_no_net_move', 'no_futures_change_250ms', 'moved_against'):
        X = D[D.cls == c]
        nx = X.next_our_way_ms.dropna()
        out['classes'][c] = dict(legs=len(X), legs_share=len(X) / len(D), shares_share=float(X.shares.sum() / D.shares.sum()),
                                 last_change_age_ms_p50=float(X.last_age_ms.median()) if len(X) else None,
                                 arrival_rel_our_decision_ms_p25_p50_p75=[float(X.rel_decision_ms.quantile(q)) for q in (.25, .5, .75)] if len(X) else None,
                                 futures_move_our_way_followed_within_2s=int(len(nx)),
                                 ms_until_that_move_p25_p50_p75=[float(nx.quantile(q)) for q in (.25, .5, .75)] if len(nx) else None)
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
