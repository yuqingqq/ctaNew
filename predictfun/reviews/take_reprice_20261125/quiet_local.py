"""The Binance book in the 300 ms before the top taker's decisions against the same periods (USER 2026-11-26: "you mean 46% of such cases, in
-300ms - 0, nothing happened in binance orderbook? the orderbook remains the same?"; follow-up of quiet_before.py, whose random times are
uniform over the day and so include hours quieter than the ones it trades in). Readout; four consumed days; no strategy change.
Local baseline (declared with winners_trigger): each decision's time shifted by +-5, 10, .., 30 s, same direction, kept when >= 1.5 s from an
hour file boundary in an hour with a bookTicker file; (a) every shifted time, labelled with its decision's group; (b) only shifted times in
the state the group requires at their own time (no move their way in their previous 300 ms, or one for the after-a-move group; thick / thin
by their own far size and lean). Features and rules as quiet_before.py (STATIC / STILL / QUIET, price change, trades, far / near size
ratios, churn); moves their way / against in the window from the per-ms best prices, reconciled with the points table on the decisions.
Output (write-once): quiet_local.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/quiet_local.py \
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
import quiet_before as QB  # noqa: E402
import timeline2 as T2  # noqa: E402

FE, DC, C, TR, WT = QB.FE, QB.DC, QB.C, QB.TR, QB.WT
W, TINY = QB.W, QB.TINY
GROUPS = ('all', 'ahead_move_300', 'after_a_move', 'no_move_300', 'thick', 'thin')


def moves_flags(st, t, d):
    """A move their way / against within (t - W, t] from the per-ms best prices (their way = a best price moved in direction d)."""
    E, bid, bq, ask, aq = st
    j0, j1 = WT.last_idx(E, t - W), WT.last_idx(E, t)
    db, da = np.diff(bid), np.diff(ask)
    rise, fall = np.r_[0, np.cumsum((db > TINY) | (da > TINY))], np.r_[0, np.cumsum((db < -TINY) | (da < -TINY))]
    r, f = rise[j1] - rise[j0] > 0, fall[j1] - fall[j0] > 0
    return np.where(d > 0, r, f), np.where(d > 0, f, r)


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'quiet_local.json'
    C.require(not fn.exists(), 'quiet_local.json exists (write-once)')
    days = WT.DAYS if len(sys.argv) < 3 else (sys.argv[2],)
    C.require(all(x in WT.DAYS for x in days), 'not a declared day')
    X = pd.read_parquet(WT.OUT / 'state_value_points.parquet')
    S = X[(X.kind == 'send') & X.day.isin(days)].reset_index(drop=True)
    S['dec'] = np.arange(len(S))
    lab = pd.DataFrame(dict(dec=S.dec, g_all=True, g_ahead_move_300=~S.momentum & (S.next_move <= 300.), g_after_a_move=S.momentum,
                            g_no_move_300=~S.momentum & (S.next_move > 300.), g_thick=~S.momentum & (S.far > QB.THICK) & (S.imb < QB.LEAN),
                            g_thin=~S.momentum & (S.far <= QB.THIN) & (S.imb >= QB.LEAN)))          # g_ prefix: never a DataFrame method name
    parts = []
    for day in days:
        day0 = float(WT.OD_day_ms(day))
        Sd = S[S.day == day]
        P = [pd.DataFrame(dict(dec=Sd.dec, t=Sd.t, d=Sd.d, shift_ms=0.))]
        for s in WT.SHIFTS:
            tt = Sd.t.to_numpy(float) + s
            k = T2.good(day, tt)
            P.append(pd.DataFrame(dict(dec=Sd.dec.to_numpy()[k], t=tt[k], d=Sd.d.to_numpy()[k], shift_ms=s)))
        P = pd.concat(P, ignore_index=True)
        tr = DC.read_trades(day).drop_duplicates('id').sort_values('E', kind='stable')
        tE, tq, tm = tr.E.to_numpy(float), tr.q.to_numpy(float), tr.m.to_numpy()
        hp = ((P.t.to_numpy() - day0) // 3600000).astype(int)
        n_done = 0
        for h, bk in DC.book_hours(day):
            sel = np.flatnonzero(hp == h)
            if not len(sel):
                continue
            st = FE.collapse(*bk)
            Q = P.iloc[sel].copy()
            t, d = Q.t.to_numpy(float), Q.d.to_numpy(float)
            for k, v in QB.hour_feats(st, t, d).items():
                Q[k] = v
            Q['their_prior'], Q['against_prior'] = moves_flags(st, t, d)
            for side, nm in ((DC.BUY_AGGR, 'buy'), (1 - DC.BUY_AGGR, 'sell')):
                m = tm == side
                Es, cq = tE[m], np.r_[0., np.cumsum(tq[m])]
                a, b = np.searchsorted(Es, t - W, 'right'), np.searchsorted(Es, t, 'right')
                Q[f'n_{nm}'], Q[f'btc_{nm}'] = (b - a).astype(float), cq[b] - cq[a]
            parts.append(Q)
            n_done += len(Q)
            C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
        C.require(n_done == len(P), f'{day}: a point was not processed (no book hour)')
        print(day, len(Sd), len(P), flush=True)
    Z = pd.concat(parts, ignore_index=True)
    D = Z[Z['shift_ms'] == 0.].merge(S[['dec', 'far', 'near', 'momentum', 'bounce']], on='dec', how='left', validate='one_to_one')
    C.require(len(D) == len(S), 'a decision is missing')
    C.require((np.abs(D.far1 - D.far) < 1e-9).all() and (np.abs(D.near1 - D.near) < 1e-9).all(), 'the size at t differs from the points table')
    C.require(D.their_prior.eq(D.momentum).all() and D.against_prior.eq(D.bounce).all(), "the moves differ from the points table's")
    up = Z.d > 0
    Z['n_their'], Z['n_against'] = np.where(up, Z.n_buy, Z.n_sell), np.where(up, Z.n_sell, Z.n_buy)
    Z['n_tr'], Z['btc_tr'] = Z.n_buy + Z.n_sell, Z.btc_buy + Z.btc_sell
    rf, rn = (Z.far1 - Z.far0).abs() / Z.far0, (Z.near1 - Z.near0).abs() / Z.near0
    Z['static'] = ~Z.price_changed & (Z.n_tr == 0) & (rf <= QB.SAME) & (rn <= QB.SAME)
    Z['still'] = Z.static & (Z.churn_far <= QB.CHURN) & (Z.churn_near <= QB.CHURN)
    Z['quiet'] = ~Z.price_changed & (Z.n_tr <= QB.Q_TR) & (rf <= QB.Q_SZ) & (rn <= QB.Q_SZ)
    Z = Z.merge(lab, on='dec', how='left', validate='many_to_one')
    imb = (Z.near1 - Z.far1) / (Z.near1 + Z.far1)
    own = dict(all=np.ones(len(Z), bool), ahead_move_300=~Z.their_prior, after_a_move=Z.their_prior, no_move_300=~Z.their_prior,
               thick=~Z.their_prior & (Z.far1 > QB.THICK) & (imb < QB.LEAN), thin=~Z.their_prior & (Z.far1 <= QB.THIN) & (imb >= QB.LEAN))
    out = dict(asof='2026-11-26', days=list(days), window_ms=W, shifts_s=[float(s) / 1000. for s in WT.SHIFTS], groups={})
    for g in GROUPS:
        m = Z[f'g_{g}'].to_numpy(bool)
        dec, loc = m & (Z['shift_ms'] == 0.).to_numpy(), m & (Z['shift_ms'] != 0.).to_numpy()
        out['groups'][g] = dict(decisions=QB.readout(Z[dec]), local=QB.readout(Z[loc]), local_same_state=QB.readout(Z[loc & np.asarray(own[g], bool)]))
    if tuple(days) == tuple(WT.DAYS):
        QJ = json.loads((H / 'quiet_before.json').read_text())['groups']
        for g, q in (('all', 'decisions_all'), ('ahead_move_300', 'decisions_ahead_move_300'), ('thick', 'decisions_thick')):
            a, b = out['groups'][g]['decisions'], QJ[q]
            C.require(a['n'] == b['n'] and abs(a['static'] - b['static']) < 1e-12 and abs(a['quiet'] - b['quiet']) < 1e-12, f'{g}: decisions differ from quiet_before')
    TR.dump(fn, out)
    print(json.dumps({g: {k: {x: v[k][x] for x in ('n', 'static', 'still', 'quiet', 'no_trade', 'far_halved_same_price')} for k in v} for g, v in out['groups'].items()}, indent=1))


if __name__ == '__main__':
    main()
