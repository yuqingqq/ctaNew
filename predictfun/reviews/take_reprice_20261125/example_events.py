"""Real examples of the top taker's decisions, ms by ms, and how a thick Binance far side is cleared (USER 2026-11-26: "You mean currently
when there is a big order on far side, the signal makes more sense right? What does it mean, can you give me an example, and what's the
findings so far"). Readout; four consumed days; no strategy change.
Points: state_value_points.parquet (the top taker's decisions, t = its arrival at Predict, and random times in both directions), only points
with no move their way in the previous 300 ms. Far side = the Binance futures best level that must empty for a move their way (the ask for a
bet on up); thick book = far-side best size > 3 BTC with lean < 0.6 (state_value's bucket).
(1) Matched baseline: the random-time rate of a move their way within 100 ms, reweighted to the decisions' cells - far-side size (10 bins) x
lean (6 bins), and those x (no trade against in the last 100 ms) x (far-side new orders <= the random median): the hit rate the book state
alone gives at the top taker's decisions; all decisions and the thick-book ones; cells with < 30 random points are left out and their share
of the decisions is reported.
(2) How the far side is cleared before the move (thick book, the move within 150 ms): the far best size at t (Q0, reconciled with the points
table) and, from the book state at t to the move, the aggressive volume at that price (bought), far-side cancels and new orders (fire_events'
convention: the untraded rest at the emptying is a cancel); residual = Q0 + new - cancels - bought (~0 by construction: the check); the
largest per-ms aggressive group their way (any price) and its time; decisions vs random times. Status per point: ok / far_improved (a better
far price appeared before the move) / near_improve (the move is a better near price, the far level not emptied); Q0 or the move time
differing from the points table is refused.
(3) Examples, drawn before the run (seed WT.SEED + 3, three per class, reported in draw order): thick-book decisions followed by a move within
150 ms; thick-book decisions with no move within 300 ms; thin-book decisions (far <= 1 BTC, lean >= 0.6) followed by a move within 150 ms.
Per example: its burst (market, direction, orders, shares, average price, time left in the window, 1 s value), the Binance per-ms best levels
and per-ms aggressive groups over (t - 300, t + 300] ms, and the points table's flows over the last 100 ms.
Output (write-once): example_events.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/example_events.py \
       [<rehearsal dir> <day>]
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

DC, C, TR, WT = FE.DC, FE.C, FE.TR, FE.WT
THICK, THIN, LEAN, HIT, MISS, WIN = 3., 1., .6, 150., 300., 300.
FAR_BINS = np.array([0., .2, .5, 1., 2., 3., 4., 6., 10., 20., np.inf])          # right-closed (a, b], as state_value's buckets
IMB_BINS = np.array([-np.inf, -.3, 0., .3, .6, .8, np.inf])                       # left-closed [a, b)
MIN_CELL, K_EX = 30, 3
CLASSES = ('thick_hit', 'thick_miss', 'thin_hit')
QS = (.1, .25, .5, .75, .9)


def qs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {f'p{int(q * 100)}': float(np.quantile(x, q)) for q in QS} if len(x) else None


def matched(R, S, keys):
    """The random-time move rate within 100 ms, reweighted to the decisions' cells."""
    g = R.groupby(keys).m100.agg(['mean', 'size'])
    w = S.groupby(keys).size().rename('w')
    J = pd.concat([w, g], axis=1).loc[w.index]
    ok = J['size'].fillna(0.).to_numpy() >= MIN_CELL
    wv, mv = J.w.to_numpy(float), J['mean'].to_numpy(float)
    return dict(n_decisions=int(len(S)), decisions_move_100=float(S.m100.mean()), book_alone_move_100=float((wv[ok] * mv[ok]).sum() / wv[ok].sum()),
                decisions_share_left_out=float(wv[~ok].sum() / wv.sum()), cells_used=int(ok.sum()))


def groups(tr, side):
    """Per-ms aggressive groups of one side: (E, BTC) sorted by E (sweep_print's)."""
    g = tr[tr.m == side].groupby('E').q.sum()
    return g.index.to_numpy(float), g.to_numpy(float)


def in_window(E, cA, a, b):
    """Sum of the events with E in (a, b] (cA = cumulative amounts with a leading 0)."""
    return cA[np.searchsorted(E, b, 'right')] - cA[np.searchsorted(E, a, 'right')]


def clearing(st, ev, ix, G, t, d, far, nxt):
    """Per point (one direction d): Q0 at t, the next move from the hour's events, its status and the far side's fate up to the move."""
    E, bid, bq, ask, aq = st
    up = d > 0
    fs = FE.side_of('opp', d)
    Pf, Qf = (ask, aq) if up else (bid, bq)
    j = WT.last_idx(E, t)
    C.require((j >= 0).all(), "a point before the hour's first book state")
    Q0, P0, Ej = Qf[j], Pf[j], E[j]
    C.require(np.abs(Q0 - far).max(initial=0.) < 1e-9, 'Q0 differs from the points table')
    Mw = TL.moves(ev, d)
    k = np.searchsorted(Mw, t, 'right')
    mv = np.where(k < len(Mw), Mw[np.minimum(k, max(len(Mw) - 1, 0))], np.inf) if len(Mw) else np.full(len(t), np.inf)
    tab = t + nxt
    C.require(np.all((np.abs(mv - tab) < 1e-6) | ((tab > t + WIN) & (mv > t + WIN))), 'the move time differs from the points table')
    gone, imp = ev[(fs, 'gone')][0], ev[(fs, 'improve')][0]
    far_imp = (np.searchsorted(imp, mv, 'right') - np.searchsorted(imp, t, 'right')) > 0
    status = np.where(~np.isfinite(mv) | (mv > t + WIN), 'no_move', np.where(far_imp, 'far_improved', np.where(np.isin(mv, gone), 'ok', 'near_improve')))
    mvc = np.where(np.isfinite(mv), mv, t + WIN)
    side = DC.BUY_AGGR if up else 1 - DC.BUY_AGGR
    bought = DC.traded(ix, np.full(len(t), side), P0, Ej, mvc)
    o = {}
    for kind in ('cancel', 'add'):
        Ek, Ak = ev[(fs, kind)]
        o[kind] = in_window(Ek, np.r_[0., np.cumsum(Ak)], Ej, mvc)
    Eg, Qg = G[side]
    big, big_t, tot = np.zeros(len(t)), np.full(len(t), np.nan), np.zeros(len(t))
    for i in range(len(t)):
        a, b = np.searchsorted(Eg, t[i], 'right'), np.searchsorted(Eg, mvc[i], 'right')
        if b > a:
            m = a + int(np.argmax(Qg[a:b]))
            big[i], big_t[i], tot[i] = Qg[m], Eg[m] - t[i], Qg[a:b].sum()
    return dict(status=status, move=mv, Q0=Q0, P0=P0, bought=bought, cancelled=o['cancel'], added=o['add'], resid=Q0 + o['add'] - o['cancel'] - bought,
                largest=big, largest_after_t=big_t, aggr_total=tot)


def summary(Z):
    ok = Z[Z.status == 'ok']
    left = ok.bought + ok.cancelled
    return dict(n=int(len(Z)), status={k: int(v) for k, v in Z.status.value_counts().sort_index().items()}, n_ok=int(len(ok)),
                far_at_t_btc=qs(ok.Q0), bought_btc=qs(ok.bought), cancelled_btc=qs(ok.cancelled), added_btc=qs(ok.added),
                bought_share_of_far=qs(ok.bought / ok.Q0), bought_share_of_left=qs(ok.bought / left),
                pooled_bought_share_of_left=float(ok.bought.sum() / left.sum()) if len(ok) else None,
                bought_ge_half_of_far=float((ok.bought >= .5 * ok.Q0).mean()) if len(ok) else None,
                resid_btc=qs(ok.resid), resid_abs_gt_001=float((ok.resid.abs() > .01).mean()) if len(ok) else None,
                largest_group_btc=qs(ok.largest), largest_over_far=qs(ok.largest / ok.Q0), largest_after_t_ms=qs(ok.largest_after_t),
                aggr_total_btc=qs(ok.aggr_total), move_lag_ms=qs(ok.move - ok.t))


def burst_info(B, A, legs, reg, r):
    m = B[(B.day == r.day) & (B.a == r.t) & (B.d == r.d)]
    C.require(len(m) >= 1, f'no burst for the example at {r.day} {r.t}')
    O = A[A.burst.isin(m.burst)]
    L = legs[legs.taker_hash.isin(O.taker_hash) & ~legs.ambiguous & legs.how2.isin([1, 3])]
    C.require(abs(L.shares.sum() - O.shares.sum()) < 1e-6, 'legs and orders disagree on shares')
    mids = sorted(int(x) for x in O.mid.unique())
    ws = float(reg.loc[mids[0], 'ws']) * 1000.
    return dict(bursts=int(len(m)), markets=mids, orders=int(len(O)), shares=float(O.shares.sum()), avg_price_up_c=float((L.ptick * L.shares).sum() / L.shares.sum()),
                time_left_s=(ws + 300000. - r.t) / 1000., t_set=int(O.t_set.min()), fire=float(O.fire.min()))


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'example_events.json'
    C.require(not fn.exists(), 'example_events.json exists (write-once)')
    days = WT.DAYS if len(sys.argv) < 3 else (sys.argv[2],)
    C.require(all(x in WT.DAYS for x in days), 'not a declared day')
    SV = json.loads((H / 'state_value.json').read_text())
    X = pd.read_parquet(WT.OUT / 'state_value_points.parquet')
    adds_med = float(X[X.kind == 'random'].far_adds.median())
    C.require(abs(adds_med - SV['far_adds_random_median_btc']) < 1e-12, 'the random far-adds median differs from state_value')
    n0 = len(X)
    X = X[~X.momentum & np.isfinite(X.imb) & np.isfinite(X.far)].reset_index(drop=True)
    X['m100'] = X.next_move <= 100.
    X['fb'] = np.searchsorted(FAR_BINS, X.far.to_numpy(), 'left')
    X['ib'] = np.searchsorted(IMB_BINS, X.imb.to_numpy(), 'right')
    X['no_sells'], X['low_refill'] = X.sells == 0, X.far_adds <= adds_med
    X['thick'] = (X.far > THICK) & (X.imb < LEAN)
    C.require(((X.fb >= 6) & (X.ib <= 4)).eq(X.thick).all(), 'the thick bucket is not a union of cells')
    R, S = X[X.kind == 'random'], X[X.kind == 'send']
    base = {}
    for nm, m_r, m_s in (('all', np.ones(len(R), bool), np.ones(len(S), bool)), ('thick', R.thick.to_numpy(), S.thick.to_numpy())):
        base[nm] = dict(book=matched(R[m_r], S[m_s], ['fb', 'ib']), book_and_flow=matched(R[m_r], S[m_s], ['fb', 'ib', 'no_sells', 'low_refill']))
    sv = SV['by_far_size']['far >3 | imb<0.6']['decisions']
    C.require(base['thick']['book']['n_decisions'] == sv['n'] and abs(base['thick']['book']['decisions_move_100'] - sv['move_100']) < 1e-12,
              'the thick decisions differ from state_value')

    rng = np.random.default_rng(WT.SEED + 3)
    cls = {'thick_hit': S.thick & (S.next_move <= HIT), 'thick_miss': S.thick & (S.next_move > MISS),
           'thin_hit': (S.far <= THIN) & (S.imb >= LEAN) & (S.next_move <= HIT)}
    EX = []
    for k in CLASSES:
        Z = S[cls[k]]
        for n, i in enumerate(rng.choice(len(Z), K_EX, replace=False)):
            EX.append(dict(cls=k, draw=n, n_class=int(len(Z)), **Z.iloc[int(i)][['day', 't', 'd', 'v1', 'imb', 'far', 'near', 'buys', 'sells', 'far_adds', 'far_cxl', 'near_adds', 'next_move']].to_dict()))
    c = float(json.loads((H / 'fire_events.json').read_text())['groups']['TOP1']['arrival_offset_ms'])
    A = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet')
    A = A[A.t_set.notna() & (A['rank'] == 1)]
    B = A.sort_values(['burst', 'fire'], kind='stable').drop_duplicates('burst', keep='first').reset_index(drop=True)
    B['a'] = B.fire - c

    P2 = X[X.thick & (X.next_move <= HIT)].reset_index(drop=True)
    rows, ex_out = [], {k: [] for k in CLASSES}
    for day in days:
        day0 = float(WT.OD_day_ms(day))
        tr = DC.read_trades(day)
        ix = DC.build_index(tr, day0)
        T = tr.drop_duplicates('id')
        G = {s: groups(T, s) for s in (DC.BUY_AGGR, 1 - DC.BUY_AGGR)}
        legs = pd.read_parquet(WT.OUT / f'winners_legs_{day}.parquet', columns=['taker_hash', 'ptick', 'shares', 'how2', 'ambiguous'])
        reg = WT.MM.registry(day).set_index('mid')
        Pd = P2[P2.day == day]
        Xd = [e for e in EX if e['day'] == day]
        hp = ((Pd.t.to_numpy() - day0) // 3600000).astype(int)
        hx = [int((e['t'] - day0) // 3600000) for e in Xd]
        for h, bk in DC.book_hours(day):
            sel = np.flatnonzero(hp == h)
            xs = [e for e, hh in zip(Xd, hx) if hh == h]
            if not len(sel) and not xs:
                continue
            st = FE.collapse(*bk)
            ev = FE.book_events(st, ix)
            for dv in (1., -1.):
                Q = Pd.iloc[sel][Pd.iloc[sel].d == dv]
                if len(Q):
                    r = clearing(st, ev, ix, G, Q.t.to_numpy(float), dv, Q.far.to_numpy(float), Q.next_move.to_numpy(float))
                    rows.append(pd.DataFrame(dict(kind=Q.kind.to_numpy(), t=Q.t.to_numpy(float), **r)))
            E, bid, bq, ask, aq = st
            for e in xs:
                t, dv = float(e['t']), float(e['d'])
                r = {k: v[0] for k, v in clearing(st, ev, ix, G, np.array([t]), dv, np.array([e['far']]), np.array([e['next_move']])).items()}
                j0 = WT.last_idx(E, np.array([t - WIN]))[0]
                b = np.flatnonzero((E > t - WIN) & (E <= t + WIN))
                book = [[float(E[i] - t), float(bid[i]), float(bq[i]), float(ask[i]), float(aq[i])] for i in np.r_[j0, b]]
                W = T[(T.E > t - WIN) & (T.E <= t + WIN)]
                tg = W.groupby(['E', 'm']).agg(q=('q', 'sum'), n=('q', 'size'), pmin=('p', 'min'), pmax=('p', 'max')).reset_index()
                trades = [[float(g.E - t), 'buy' if int(g.m) == DC.BUY_AGGR else 'sell', float(g.q), int(g.n), float(g.pmin), float(g.pmax)] for g in tg.itertuples()]
                info = burst_info(B, A, legs, reg, pd.Series(e))
                rec = dict(e, utc=pd.Timestamp(t, unit='ms', tz='UTC').strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], direction='up' if dv > 0 else 'down', **info,
                           settle_rel_t=float(info['t_set'] - t), status=str(r['status']), move_rel_t=float(r['move'] - t) if np.isfinite(r['move']) else None,
                           far_price=float(r['P0']), bought=float(r['bought']), cancelled=float(r['cancelled']), added=float(r['added']), resid=float(r['resid']),
                           largest_group_btc=float(r['largest']), largest_after_t_ms=float(r['largest_after_t']) if np.isfinite(r['largest_after_t']) else None,
                           aggr_total_btc=float(r['aggr_total']), book=book, trades=trades)
                ex_out[e['cls']].append(rec)
            C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
        print(day, len(Pd), len(Xd), flush=True)
    Z = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['kind', 'status'])
    for k in CLASSES:
        ex_out[k].sort(key=lambda r: r['draw'])
    out = dict(asof='2026-11-26', days=list(days), points_in=int(n0), points_no_momentum=int(len(X)), far_adds_random_median_btc=adds_med,
               thick=dict(far_gt_btc=THICK, lean_lt=LEAN), thin=dict(far_le_btc=THIN, lean_ge=LEAN), hit_ms=HIT, miss_ms=MISS, window_ms=WIN,
               far_bins_btc=[float(x) for x in FAR_BINS[1:-1]], lean_bins=[float(x) for x in IMB_BINS[1:-1]], min_cell=MIN_CELL,
               matched_baseline=base,
               far_clearing={k: summary(Z[Z.kind == k]) for k in ('send', 'random')},
               examples=ex_out)
    TR.dump(fn, out)
    print(json.dumps(dict(matched_baseline=base, far_clearing=out['far_clearing']), indent=1))


if __name__ == '__main__':
    main()
