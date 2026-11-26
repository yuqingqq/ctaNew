"""Why does ~12 % of the aggressive volume at the best Binance futures price leave no equal size drop in the bookTicker over (E - 2, E + 1]
(depletion_cause.py rehearsal, 20261031)? Data-quality probe for that split; one consumed day; no strategy change.
Readings for the aggressive trade-ms at the best opposite price: the unreflected volume share over (t - 2, t + k], k = 1, 2, 5, 20 ms, on E
and on T (both streams' transaction time); for the unreflected trade-ms at k = 1 on E: no bookTicker update in the window / size unchanged /
partial drop, and whether the size dips to Q - V +- 0.5 mBTC within (t - 2, t + 5] (a visible drop refilled at once); median size of
reflected vs unreflected trade-ms; aggressive volume priced strictly inside the visible spread at t - 2.
Output (write-once): depletion_probe.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/depletion_probe.py [day [dir]]
"""
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import depletion_cause as DC  # noqa: E402

C, TR, WT = DC.C, DC.TR, DC.WT
KS = (1., 2., 5., 20.)


def book_hours_T(day):
    """bookTicker per hour on T (column 2), the previous hour's last 2 s prepended; returned as (E := T, bid, bq, ask, aq)."""
    C.require(day in WT.DAYS, f'{day} is not a declared day')
    tail = None
    for h in range(24):
        fn = DC.RAW / f'bookTicker/BTCUSDT/{day}_{h:02d}.csv.gz'
        if not fn.exists():
            tail = None
            continue
        b = pd.read_csv(fn, header=None, usecols=[2, 4, 5, 6, 7], names=['E', 'bid', 'bq', 'ask', 'aq'])
        if tail is not None:
            b = pd.concat([tail, b])
        b = b.sort_values('E', kind='stable')
        tail = b[b.E >= b.E.iat[-1] - 2000.]
        yield h, tuple(b[c].to_numpy(float) for c in ('E', 'bid', 'bq', 'ask', 'aq'))


def unreflected(bk, ix, lo, hi, k):
    s = (ix['grp_e'] >= lo) & (ix['grp_e'] < hi)
    e, m, tick = ix['grp_e'][s], ix['grp_m'][s], ix['grp_tick'][s]
    d = np.where(m == DC.BUY_AGGR, 1., -1.)
    r = DC.split(bk, ix, e - 2., e + k, d, np.zeros(len(e), bool))
    at = ((r['st'] == DC.SAME) | (r['st'] == DC.GONE)) & (np.abs(r['P'] * 10. - tick) < 1e-6)
    V, drop = r['V'][at], (r['Q'] - r['left'])[at]
    return dict(volume=float(V.sum()), unreflected=float(np.maximum(V - np.maximum(drop, 0.), 0.).sum())), (e[at], d[at], r['P'][at], r['Q'][at], V, drop)


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (600, 601))
    day = sys.argv[1] if len(sys.argv) > 1 else '20261031'
    fn = (Path(sys.argv[2]) if len(sys.argv) > 2 else H) / 'depletion_probe.json'     # a second argument: rehearsal dir
    C.require(not fn.exists(), 'depletion_probe.json exists (write-once)')
    day0 = WT.OD_day_ms(day)
    tr = DC.read_trades(day)
    ix = DC.build_index(tr, day0)
    trT = tr.copy()
    parts = [pd.read_csv(DC.RAW / f'trade/BTCUSDT/{day}_{h:02d}.csv.gz', header=None, usecols=[2, 3], names=['T', 'id'])
             for h in range(24) if (DC.RAW / f'trade/BTCUSDT/{day}_{h:02d}.csv.gz').exists()]
    Tm = pd.concat(parts, ignore_index=True).drop_duplicates('id').set_index('id')['T']
    trT['E'] = trT.id.map(Tm).to_numpy(np.int64)
    ixT = DC.build_index(trT, day0)
    acc = {f'{c}_{int(k)}': dict(volume=0., unreflected=0.) for c in ('E', 'T') for k in KS}
    cls = dict(no_update=0., size_unchanged=0., partial=0., n_no_update=0, n_size_unchanged=0, n_partial=0)
    dip, n_unref, sz_ref, sz_unref = 0, 0, [], []
    inside, aggr = 0., 0.
    for clock, gen, I in (('E', DC.book_hours(day), ix), ('T', book_hours_T(day), ixT)):
        for h, bk in gen:
            h0 = day0 + h * 3600000.
            lo, hi = h0 + 2000., h0 + 3600000. - DC.EDGE
            for k in KS:
                a, grp = unreflected(bk, I, lo, hi, k)
                for kk in ('volume', 'unreflected'):
                    acc[f'{clock}_{int(k)}'][kk] += a[kk]
                if clock != 'E' or k != 1.:
                    continue
                e, d, P, Q, V, drop = grp
                bad = V - np.maximum(drop, 0.) > DC.TOL
                sz_ref.append(V[~bad]); sz_unref.append(V[bad])
                E, bid, bq, ask, aq = bk
                j0, j1, j5 = WT.last_idx(E, e - 2.), WT.last_idx(E, e + 1.), WT.last_idx(E, e + 5.)
                for i in np.flatnonzero(bad):
                    w = V[i] - max(drop[i], 0.)
                    if j1[i] == j0[i]:
                        cls['no_update'] += w; cls['n_no_update'] += 1
                    elif abs(drop[i]) <= DC.TOL:
                        cls['size_unchanged'] += w; cls['n_size_unchanged'] += 1
                    else:
                        cls['partial'] += w; cls['n_partial'] += 1
                    sl = slice(j0[i] + 1, j5[i] + 1)
                    px, qq = (ask[sl], aq[sl]) if d[i] > 0 else (bid[sl], bq[sl])
                    size_at = np.where(np.abs(px - P[i]) < 1e-9, qq, np.where(d[i] * (px - P[i]) > 0, 0., np.inf))
                    n_unref += 1
                    dip += int(len(size_at) > 0 and size_at.min() <= Q[i] - V[i] + DC.TOL)
            if clock == 'E':
                s = (I['E'] >= lo) & (I['E'] < hi)
                te, tp, tq, tm = I['E'][s], I['p'][s], I['q'][s], I['m'][s]
                E, bid, bq, ask, aq = bk
                j = WT.last_idx(E, te - 2.)
                ok = j >= 0
                jj = np.maximum(j, 0)
                ins = ok & (tp > bid[jj] + 1e-9) & (tp < ask[jj] - 1e-9)
                inside += float(tq[ins].sum()); aggr += float(tq[ok].sum())
            C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
    tot_unref = sum(cls[k] for k in ('no_update', 'size_unchanged', 'partial'))
    out = dict(asof='2026-11-26', day=day, windows_ms=list(KS),
               unreflected_share={k: v['unreflected'] / v['volume'] if v['volume'] > 0 else None for k, v in acc.items()},
               volume_btc={k: v['volume'] for k, v in acc.items()},
               unreflected_E1=dict(groups=n_unref, volume_btc=tot_unref, by_kind_share={k: cls[k] / tot_unref for k in ('no_update', 'size_unchanged', 'partial')} if tot_unref else None,
                                   counts={k: cls[f'n_{k}'] for k in ('no_update', 'size_unchanged', 'partial')}, dip_within_5ms=dip / n_unref if n_unref else None),
               median_size_btc=dict(reflected=float(np.median(np.concatenate(sz_ref))), unreflected=float(np.median(np.concatenate(sz_unref)))),
               inside_spread_volume_share=inside / aggr if aggr else None, aggressive_volume_btc=aggr, tape=ix['tape'])
    TR.dump(fn, out)
    import json
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
