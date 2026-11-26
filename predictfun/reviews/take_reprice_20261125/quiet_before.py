"""Is the Binance book unchanged in the 300 ms before the top taker's decisions? (USER 2026-11-26: "you mean 46% of such cases, in -300ms -
0, nothing happened in binance orderbook? the orderbook remains the same?"). Readout; four consumed days; no strategy change.
Points: state_value_points.parquet (its 5,930 decisions, t = arrival at Predict; random times in both directions).
Per point over (t - 300, t] on the Binance futures bookTicker (E, collapsed per ms) and trades (E, de-duplicated by id): book updates (per-ms
states), best bid / ask price changes (reconciled with the points table: a change <=> a move their way or against in the window), aggressive
trades (count, BTC; their way / against), far / near best size at t - 300 and at t (t reconciled with the table), gross size churn per side
(sum of |size change| between consecutive states at an unchanged price on that side).
Declared before the run: STATIC = no best price change, no trade, far and near size at t within 1% of t - 300 (the top of the book the same);
STILL = STATIC and gross churn <= 0.05 BTC on each side; QUIET = no best price change, at most 2 trades, far and near size within 10%.
Groups: decisions all / ahead of a move within 300 ms (no move their way in the previous 300 ms) / just after a move / no move within 300 ms /
thick book (far > 3 BTC, lean < 0.6, no prior move) / thin book (far <= 1 BTC, lean >= 0.6, no prior move); random times all / no prior move /
thick / thin. Readings: shares STATIC / STILL / QUIET / price changed / no trade; quantiles of updates, trades, far(t)/far(t-300) and
near(t)/near(t-300) where no price changed, churn per side; share with the far side at t <= half of t - 300.
Output (write-once): quiet_before.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/quiet_before.py \
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

DC, C, TR, WT = FE.DC, FE.C, FE.TR, FE.WT
W, SAME, TINY, Q_TR, Q_SZ, CHURN = 300., .01, 1e-9, 2, .10, .05
THICK, THIN, LEAN = 3., 1., .6
QS = (.1, .25, .5, .75, .9)


def qs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {f'p{int(q * 100)}': float(np.quantile(x, q)) for q in QS} if len(x) else None


def hour_feats(st, t, d):
    """Per point: updates, price change, far / near size at t - W and t, churn per side over (t - W, t]."""
    E, bid, bq, ask, aq = st
    j0, j1 = WT.last_idx(E, t - W), WT.last_idx(E, t)
    C.require((j0 >= 0).all(), "a point whose window starts before the hour's first state")
    pc = np.r_[0, np.cumsum((np.abs(np.diff(bid)) > TINY) | (np.abs(np.diff(ask)) > TINY))]
    cb = np.r_[0., np.cumsum(np.where(np.abs(np.diff(bid)) > TINY, 0., np.abs(np.diff(bq))))]
    ca = np.r_[0., np.cumsum(np.where(np.abs(np.diff(ask)) > TINY, 0., np.abs(np.diff(aq))))]
    up = d > 0
    far0, far1 = np.where(up, aq[j0], bq[j0]), np.where(up, aq[j1], bq[j1])
    near0, near1 = np.where(up, bq[j0], aq[j0]), np.where(up, bq[j1], aq[j1])
    ch_b, ch_a = cb[j1] - cb[j0], ca[j1] - ca[j0]
    return dict(updates=(j1 - j0).astype(float), price_changed=(pc[j1] - pc[j0]) > 0, far0=far0, far1=far1, near0=near0, near1=near1,
                churn_far=np.where(up, ch_a, ch_b), churn_near=np.where(up, ch_b, ch_a))


def readout(Z):
    sp = Z[~Z.price_changed]
    return dict(n=int(len(Z)), static=float(Z.static.mean()), still=float(Z.still.mean()), quiet=float(Z.quiet.mean()), price_changed=float(Z.price_changed.mean()),
                no_trade=float((Z.n_tr == 0).mean()), updates=qs(Z.updates), trades=qs(Z.n_tr), trades_btc=qs(Z.btc_tr),
                trades_their_way=qs(Z.n_their), trades_against=qs(Z.n_against),
                far_ratio_same_price=qs(sp.far1 / sp.far0), near_ratio_same_price=qs(sp.near1 / sp.near0),
                far_halved_same_price=float((sp.far1 <= .5 * sp.far0).mean()) if len(sp) else None,
                churn_far_btc=qs(Z.churn_far), churn_near_btc=qs(Z.churn_near))


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'quiet_before.json'
    C.require(not fn.exists(), 'quiet_before.json exists (write-once)')
    days = WT.DAYS if len(sys.argv) < 3 else (sys.argv[2],)
    C.require(all(x in WT.DAYS for x in days), 'not a declared day')
    X = pd.read_parquet(WT.OUT / 'state_value_points.parquet')
    X = X[X.day.isin(days)].reset_index(drop=True)
    parts = []
    for day in days:
        day0 = float(WT.OD_day_ms(day))
        tr = DC.read_trades(day).drop_duplicates('id').sort_values('E', kind='stable')
        tE, tq, tm = tr.E.to_numpy(float), tr.q.to_numpy(float), tr.m.to_numpy()
        Xd = X[X.day == day]
        hp = ((Xd.t.to_numpy() - day0) // 3600000).astype(int)
        for h, bk in DC.book_hours(day):
            sel = np.flatnonzero(hp == h)
            if not len(sel):
                continue
            st = FE.collapse(*bk)
            Q = Xd.iloc[sel].copy()
            t, d = Q.t.to_numpy(float), Q.d.to_numpy(float)
            for k, v in hour_feats(st, t, d).items():
                Q[k] = v
            for side, nm in ((DC.BUY_AGGR, 'buy'), (1 - DC.BUY_AGGR, 'sell')):
                m = tm == side
                Es, cq = tE[m], np.r_[0., np.cumsum(tq[m])]
                a, b = np.searchsorted(Es, t - W, 'right'), np.searchsorted(Es, t, 'right')
                Q[f'n_{nm}'], Q[f'btc_{nm}'] = (b - a).astype(float), cq[b] - cq[a]
            parts.append(Q)
            C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
        print(day, len(Xd), flush=True)
    Z = pd.concat(parts, ignore_index=True)
    C.require(len(Z) == len(X), 'a point was not processed (no book hour)')
    C.require((np.abs(Z.far1 - Z.far) < 1e-9).all() and (np.abs(Z.near1 - Z.near) < 1e-9).all(), 'the size at t differs from the points table')
    C.require(Z.price_changed.eq(Z.momentum | Z.bounce).all(), "a price change differs from the table's moves")
    up = Z.d > 0
    Z['n_their'] = np.where(up, Z.n_buy, Z.n_sell)
    Z['n_against'] = np.where(up, Z.n_sell, Z.n_buy)
    Z['n_tr'], Z['btc_tr'] = Z.n_buy + Z.n_sell, Z.btc_buy + Z.btc_sell
    rf, rn = (Z.far1 - Z.far0).abs() / Z.far0, (Z.near1 - Z.near0).abs() / Z.near0
    Z['static'] = ~Z.price_changed & (Z.n_tr == 0) & (rf <= SAME) & (rn <= SAME)
    Z['still'] = Z.static & (Z.churn_far <= CHURN) & (Z.churn_near <= CHURN)
    Z['quiet'] = ~Z.price_changed & (Z.n_tr <= Q_TR) & (rf <= Q_SZ) & (rn <= Q_SZ)
    S, R = Z[Z.kind == 'send'], Z[Z.kind == 'random']
    thick = lambda G: G[~G.momentum & (G.far > THICK) & (G.imb < LEAN)]
    thin = lambda G: G[~G.momentum & (G.far <= THIN) & (G.imb >= LEAN)]
    groups = {'decisions_all': S, 'decisions_ahead_move_300': S[~S.momentum & (S.next_move <= 300.)], 'decisions_after_a_move': S[S.momentum],
              'decisions_no_move_300': S[~S.momentum & (S.next_move > 300.)], 'decisions_thick': thick(S), 'decisions_thin': thin(S),
              'random_all': R, 'random_no_prior_move': R[~R.momentum], 'random_thick': thick(R), 'random_thin': thin(R)}
    if tuple(days) == tuple(WT.DAYS):
        sv = json.loads((H / 'state_value.json').read_text())['decisions']
        C.require(len(groups['decisions_all']) == sv['n'] and len(groups['decisions_ahead_move_300']) == sv['by_class']['ahead_move_300']['n'],
                  'the decision classes differ from state_value')
    out = dict(asof='2026-11-26', days=list(days), window_ms=W, rules=dict(static=f'no price change, no trade, far and near within {SAME:.0%}',
               still=f'static and churn <= {CHURN} BTC per side', quiet=f'no price change, <= {Q_TR} trades, far and near within {Q_SZ:.0%}'),
               groups={k: readout(G) for k, G in groups.items()})
    TR.dump(fn, out)
    print(json.dumps({k: {x: v[x] for x in ('n', 'static', 'still', 'quiet', 'price_changed', 'no_trade')} for k, v in out['groups'].items()}, indent=1))


if __name__ == '__main__':
    main()
