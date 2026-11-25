"""TAKE-EXIT review diagnostics (read-only on the v2 PASSIVE_EXIT ledgers + reloaded worlds; no replay):
 D1 every filled IOC classified by the first venue frame after its effect: SURVIVED (size at or better than the limit still >= the fill),
    TRADED_AWAY (the shortfall is covered by public trades at or better than the limit in (effect, next frame]: later takers, we came
    first), CANCEL_UNKNOWN (shortfall not covered by trades: a cancel somewhere inside the frame interval, possibly before our arrival);
    gap statistics; the take legs' 3 s markout value per class.
 D2 every passive exit fill classified by its tape trade: THROUGH (a maker beyond our price was hit: certain if we were resting) vs AT_PRICE
    (queue-dependent, engine rule M); shares and value vs holding to the END mark per class; order age split.
Usage: run_light.sh txrv_diag -- python3 predictfun/reviews/take_exit_review_20261125/diag.py
"""
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / 'take_exit_replay_20261125'))
import more_markets as MM  # noqa: E402

EPS = 1e-9


def levels(fr, side):
    """(prices, sizes) of one side of a frame (0 bid, 1 ask), valid levels only."""
    p = fr[2 + side * 10: 7 + side * 10]
    s = fr[7 + side * 10: 12 + side * 10]
    ok = (p >= 0) & (p <= 100) & (s > 0)
    return p[ok], s[ok]


def main():
    cen = json.loads((MM.OUT2 / 'census.json').read_text())
    sel = {(d, int(m)) for d, m in cen['selected']}
    src = MM.KP.Tape()
    D1, D2 = [], []
    for day in MM.DAYS:
        reg = MM.registry(day).set_index('mid')
        off = src.offset(day)
        for h, g in reg.groupby('hour'):
            want = [int(x) for x in g.index if (day, int(x)) in sel]
            if not want:
                continue
            for m, st, p, info in MM.worlds(day, want, src, off, reg):
                f = np.asarray(p['frames'], float)
                tr = np.asarray(p['trades'], float).reshape(-1, 4)
                rec = json.loads(gzip.decompress((MM.OUT2 / f'{m}_PASSIVE_EXIT.json.gz').read_bytes()))
                io = {int(k): v for k, v in rec['iocs'].items()}
                marks = {int(x['gid']): x for x in rec['take_marks']}
                for x in rec['ioc_results']:
                    if x['filled'] <= EPS:
                        continue
                    t, s, lim, q = float(x['act']), int(io[int(x['gid'])]['side']), float(io[int(x['gid'])]['limit']), float(x['filled'])
                    opp = 1 - s
                    j2 = int(np.searchsorted(f[:, 0], t, side='right'))
                    row = dict(day=day, mid=m, gid=int(x['gid']), q=q, frame_age_ms=t - float(f[int(x['frame']), 0]))
                    if j2 >= len(f):
                        row.update(cls='NO_NEXT_FRAME')
                    else:
                        pr, sz = levels(f[j2], opp)
                        ok = (pr <= lim + EPS) if s == 0 else (pr >= lim - EPS)
                        left = float(sz[ok].sum())
                        row['next_gap_ms'] = float(f[j2, 0] - t)
                        if left >= q - EPS:
                            row['cls'] = 'SURVIVED'
                        else:
                            sel_t = (tr[:, 0] > t) & (tr[:, 0] <= f[j2, 0]) & (tr[:, 1] == opp) & ((tr[:, 2] <= lim + EPS) if s == 0 else (tr[:, 2] >= lim - EPS))
                            traded = float(tr[sel_t, 3].sum())
                            row['cls'] = 'TRADED_AWAY' if left + traded >= q - EPS else 'CANCEL_UNKNOWN'
                    mk = marks.get(int(x['gid']))
                    row['m3s_value_c'] = (mk['m3s_c'] * q) if (mk and mk['m3s_c'] is not None) else np.nan
                    row['end_value_c'] = (mk['end_c'] * q) if mk else np.nan
                    D1.append(row)
                acts = {int(o['gid']): float(o['act']) for o in rec['orders']} if rec['orders'] and isinstance(rec['orders'][0], dict) else {}
                for t, gid, s, q, pz, it in rec['fills']:
                    if int(gid) in io or it < 0:
                        continue
                    tp = float(tr[int(it), 2])
                    through = (tp > pz + EPS) if s == 1 else (tp < pz - EPS)
                    val = ((rec['mark'] - pz) if s == 0 else (pz - rec['mark'])) * q
                    age = t - acts[int(gid)] if int(gid) in acts else np.nan
                    D2.append(dict(day=day, mid=m, q=q, cls='THROUGH' if through else 'AT_PRICE', value_vs_hold_c=val, age_ms=age, young=bool(age < 100) if age == age else None))
    d1, d2 = pd.DataFrame(D1), pd.DataFrame(D2)
    d1.to_csv(H / 'd1_ioc_survival.csv', index=False)
    d2.to_csv(H / 'd2_exit_fill_class.csv', index=False)
    out = dict(
        ioc_fills=len(d1), ioc_shares=float(d1.q.sum()),
        ioc_by_class={k: dict(fills=int(len(x)), shares=float(x.q.sum()), m3s_value_usd=float(x.m3s_value_c.sum() / 100), end_value_usd=float(x.end_value_c.sum() / 100),
                              frame_age_ms_p50=float(x.frame_age_ms.median()), next_gap_ms_p50=float(x.next_gap_ms.median()) if 'next_gap_ms' in x else None)
                      for k, x in d1.groupby('cls')},
        exit_fills=len(d2), exit_shares=float(d2.q.sum()),
        exit_by_class={k: dict(fills=int(len(x)), shares=float(x.q.sum()), value_vs_hold_usd=float(x.value_vs_hold_c.sum() / 100),
                               young_shares=float(x[x.young == True].q.sum())) for k, x in d2.groupby('cls')})
    (H / 'diag.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
