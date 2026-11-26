"""Is the price trending before the moves the top taker catches, and is it a bounce? (USER 2026-11-26: "You mean mainly the other side
decreases the flow? For example, if the bid side is trending, the trend continues and the selling side slows down, is this the case").
Readout on timeline2.py's first-move sets (the same construction and seed: the moves the top taker's orders landed ahead of, and the sampled
first moves = no move in the same direction in the previous 300 ms); four consumed days; no strategy change.
Per move (direction d): Binance mid changes in direction d and against it (fire_events' gone / improve events) over (t - 1 s, t - 300 ms]
and (t - 5 s, t - 300 ms] (the windows end where the first-move filter starts); net = with - against; and moves against d in the last 300 ms
(t - 300, t) (a bounce). Readings per set: mean net, share net > 0 / < 0, share with any move against in the last 300 ms, and the windowed
flow ratios of timeline2.json restated for trades against / far-side new orders / trades their way / near-side new orders.
Output (write-once): pretrend.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/pretrend.py [dir]
"""
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
import timeline2 as T2  # noqa: E402

DC, C, TR, WT = FE.DC, FE.C, FE.TR, FE.WT
WINDOWS = ((1000., 300.), (5000., 300.))


def counts(M, t, a, b):
    """Moves in M with time in (t - a, t - b]."""
    return np.searchsorted(M, t - b, 'right') - np.searchsorted(M, t - a, 'right')


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'pretrend.json'
    C.require(not fn.exists(), 'pretrend.json exists (write-once)')
    c = float(json.loads((H / 'fire_events.json').read_text())['groups']['TOP1']['arrival_offset_ms'])
    A = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet')
    A = A[A.t_set.notna() & (A['rank'] == 1)]
    B = A.sort_values(['burst', 'fire'], kind='stable').drop_duplicates('burst', keep='first').reset_index(drop=True)
    B['a'] = B.fire - c
    rng = np.random.default_rng(WT.SEED)
    rows = []
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
        sets = {'top_caught': (nxt[ok], d[ok])}
        mt, md = [], []
        for dv in (1., -1.):                                # timeline2's sample, same seed and call order
            M = TL.moves(ev, dv)
            M = M[T2.good(day, M)]
            pick = np.sort(rng.choice(M, min(TL.N_ALL, len(M)), replace=False)) if len(M) else M
            mt.append(pick)
            md.append(np.full(len(pick), dv))
        sets['all_moves'] = (np.concatenate(mt), np.concatenate(md))
        Mw = {dv: TL.moves(ev, dv) for dv in (1., -1.)}
        for name in ('top_caught', 'all_moves'):
            tt, dd = sets[name]
            keep = T2.good(day, tt)
            tt, dd = tt[keep], dd[keep]
            r = dict(set=name.replace('moves', 'first').replace('caught', 'caught_first'), day=day)
            first = np.zeros(len(tt), bool)
            res = {f'{k}_{int(a)}': np.zeros(len(tt)) for k in ('with', 'against') for a, _ in WINDOWS}
            bounce = np.zeros(len(tt))
            for dv in (1., -1.):
                m = dd == dv
                W_, O_ = Mw[dv], Mw[-dv]
                j = np.searchsorted(W_, tt[m], 'left')
                first[m] = (j == 0) | (tt[m] - W_[np.maximum(j - 1, 0)] > T2.QUIET)
                for a, b in WINDOWS:
                    res[f'with_{int(a)}'][m] = counts(W_, tt[m], a, b)
                    res[f'against_{int(a)}'][m] = counts(O_, tt[m], a, b)
                bounce[m] = np.searchsorted(O_, tt[m], 'left') - np.searchsorted(O_, tt[m] - T2.QUIET, 'right')
            R = pd.DataFrame(dict(day=day, set=r['set'], first=first, bounce=bounce, **res))
            rows.append(R[R['first']])
        C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
        print(day, {k: len(v[0]) for k, v in sets.items()}, flush=True)
    X = pd.concat(rows, ignore_index=True)
    T2J = json.loads((H / 'timeline2.json').read_text())
    E = np.asarray(T2J['edges_ms'])
    lo = E[:-1]
    out = dict(asof='2026-11-26', windows_ms=[list(w) for w in WINDOWS], quiet_ms=T2.QUIET, sets={})
    for name in ('top_caught_first', 'all_first'):
        G = X[X.set == name]
        o = dict(n=int(len(G)), bounce_share=float((G.bounce > 0).mean()))
        for a, _ in WINDOWS:
            net = G[f'with_{int(a)}'] - G[f'against_{int(a)}']
            o[f'net_ticks_{int(a)}'] = dict(mean=float(net.mean()), share_up=float((net > 0).mean()), share_down=float((net < 0).mean()), share_flat=float((net == 0).mean()),
                                            mean_with=float(G[f'with_{int(a)}'].mean()), mean_against=float(G[f'against_{int(a)}'].mean()))
        S = T2J['sets'][name]
        o['flow_ratios'] = {}
        for k in ('near_trade', 'opp_add', 'opp_trade', 'near_add', 'opp_cancel'):
            r = S['rates'][k]
            ev, rd = np.asarray(r['event_btc_per_ms']), np.asarray(r['random_btc_per_ms'])
            o['flow_ratios'][k] = {f'{a:+.0f}..{b:+.0f}': float(ev[(lo >= a) & (lo < b)].sum() / rd[(lo >= a) & (lo < b)].sum())
                                   for a, b in ((-300., -200.), (-200., -100.), (-100., -60.), (-60., -40.), (-40., -20.), (-20., -5.))}
        out['sets'][name] = o
    C.require(out['sets']['top_caught_first']['n'] == T2J['sets']['top_caught_first']['n_events'] and out['sets']['all_first']['n'] == T2J['sets']['all_first']['n_events'],
              'the first-move sets do not reproduce timeline2')
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
