"""What happens on Binance around the top taker's send, on the move clock (USER 2026-11-26: "You mean they send at around -50ms right? Then
we check what happens in Binance around that time"). Readout; four consumed days; no strategy change.
t = 0 = the first Binance futures mid change in the taker's direction (timeline.py's move). Sets: the moves the top taker caught (its 'ahead'
sends with the move within 300 ms, timeline.py's definition) and a seeded sample of all Binance moves (5,000 per direction per day); each
with its own random baseline = the same times shifted by +-5, 10, .., 30 s (same direction). Points within 1.5 s of an hour file boundary or
in an hour without a bookTicker file are excluded.
Per set, 5 ms bins over [-300, +100) ms: the rate of every fire_events event kind (count and BTC; opp = the far side that empties at the
move, near = the other) around the move and around its random times; the top taker's arrival at Predict relative to its move (histogram,
5 ms bins; arrival = settlement - 150 - c, c = fire_events' arrival offset).
Output (write-once): timeline2.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/timeline2.py [dir]
"""
import collections
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import fire_events as FE  # noqa: E402
import timeline as TL  # noqa: E402

DC, C, TR, WT = FE.DC, FE.C, FE.TR, FE.WT
EDGES = np.arange(-300., 101., 5.)


def good(day, t):
    """Not within EDGE_MS of an hour boundary and in an hour whose bookTicker file exists (existence only; declared days only)."""
    C.require(day in WT.DAYS, f'{day} is not a declared day')
    d0 = WT.OD_day_ms(day)
    hrs = np.array([(DC.RAW / f'bookTicker/BTCUSDT/{day}_{h:02d}.csv.gz').exists() for h in range(24)])
    h = ((t - d0) // 3600000).astype(int)
    inday = (h >= 0) & (h < 24)
    edge = np.abs(t - (d0 + np.rint((t - d0) / 3600000.) * 3600000.)) < FE.EDGE_MS
    return inday & ~edge & hrs[np.clip(h, 0, 23)]


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'timeline2.json'
    C.require(not fn.exists(), 'timeline2.json exists (write-once)')
    c = float(json.loads((H / 'fire_events.json').read_text())['groups']['TOP1']['arrival_offset_ms'])
    A = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet')
    A = A[A.t_set.notna() & (A['rank'] == 1)]
    B = A.sort_values(['burst', 'fire'], kind='stable').drop_duplicates('burst', keep='first').reset_index(drop=True)
    B['a'] = B.fire - c
    rng = np.random.default_rng(WT.SEED)
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    lags = []
    for day in WT.DAYS:
        Bd = B[B.day == day].reset_index(drop=True)
        if not len(Bd):
            continue
        P = pd.DataFrame(dict(t=Bd.a.to_numpy(float), d=Bd.d.to_numpy(float)))
        ev, st, _, _ = FE.day_data(day, P)
        t, d = P.t.to_numpy(), P.d.to_numpy()
        nxt, prv = np.full(len(t), np.nan), np.full(len(t), np.nan)
        for dv in (1., -1.):
            m = d == dv
            M = TL.moves(ev, dv)
            j = np.searchsorted(M, t[m], 'right')
            nxt[m] = np.where(j < len(M), M[np.minimum(j, len(M) - 1)], np.nan)
            prv[m] = np.where(j > 0, M[np.maximum(j - 1, 0)], np.nan)
        ok = (st == 'ok') & ((t - prv > TL.AHEAD_MS) | np.isnan(prv)) & (nxt - t <= TL.MAX_LAG)
        lags.append(t[ok] - nxt[ok])
        sets = {'top_caught': (nxt[ok], d[ok])}
        mt, md = [], []
        for dv in (1., -1.):
            M = TL.moves(ev, dv)
            M = M[good(day, M)]
            pick = np.sort(rng.choice(M, min(TL.N_ALL, len(M)), replace=False)) if len(M) else M
            mt.append(pick)
            md.append(np.full(len(pick), dv))
        sets['all_moves'] = (np.concatenate(mt), np.concatenate(md))
        for name, (tt, dd) in sets.items():
            keep = good(day, tt)
            tt, dd = tt[keep], dd[keep]
            rt = np.concatenate([tt + s for s in WT.SHIFTS])
            rd = np.tile(dd, len(WT.SHIFTS))
            rk = good(day, rt)
            for kind, (x, y) in (('event', (tt, dd)), ('random', (rt[rk], rd[rk]))):
                acc[(name, kind)]['n'] += len(x)
                for k, (n, v) in FE.peri(ev, x, y, EDGES).items():
                    acc[(name, kind)][(k, 'n')] = acc[(name, kind)].get((k, 'n'), 0.) + n
                    acc[(name, kind)][(k, 'v')] = acc[(name, kind)].get((k, 'v'), 0.) + v
        C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
        print(day, int(ok.sum()), {k: len(v[0]) for k, v in sets.items()}, flush=True)
    L = np.concatenate(lags)
    h, _ = np.histogram(L, bins=EDGES)
    out = dict(asof='2026-11-26', edges_ms=EDGES.tolist(), arrival_offset_ms=c,
               top1_arrival_rel_move=dict(n=int(len(L)), hist_5ms=h.tolist(), p25=float(np.quantile(L, .25)), p50=float(np.quantile(L, .5)), p75=float(np.quantile(L, .75))),
               sets={})
    for name in ('top_caught', 'all_moves'):
        e, r = acc[(name, 'event')], acc[(name, 'random')]
        o = dict(n_events=int(e['n']), n_random=int(r['n']), rates={})
        for role in FE.ROLES:
            for kind in FE.KINDS:
                k = f'{role}_{kind}'
                o['rates'][k] = dict(event_n_per_ms=(e[(k, 'n')] / e['n'] / 5.).tolist(), random_n_per_ms=(r[(k, 'n')] / r['n'] / 5.).tolist(),
                                     event_btc_per_ms=(e[(k, 'v')] / e['n'] / 5.).tolist(), random_btc_per_ms=(r[(k, 'v')] / r['n'] / 5.).tolist())
        out['sets'][name] = o
    TR.dump(fn, out)
    print(json.dumps(dict(top1_arrival_rel_move={k: v for k, v in out['top1_arrival_rel_move'].items() if k != 'hist_5ms'},
                          n={k: (v['n_events'], v['n_random']) for k, v in out['sets'].items()}), indent=1))


if __name__ == '__main__':
    main()
