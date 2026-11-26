"""All of the top taker's fires, split by what the Binance book did in the 300 ms before (USER 2026-11-26: "can you tell me what does this
46% mean exactly? ... Among all takers fires, we check the -300 ms to firing time book dynamics, a part of them shows no activity, a part of
them shows notable activity, can you show these data clearly"). Readout; the four consumed days (it traded on three); no strategy change.
Points: its 5,930 decisions (t = its order's arrival at Predict, state_value_points) and nearby moments = the same times shifted +-5, 10, ..,
30 s, same direction (quiet_local's baseline). Windows: before = (t - 300, t], after = (t, t + 300].
Rows (declared before the run, a partition, in this order; thresholds = quiet_before's declared rules): moved its way (a best price moved in
its direction before) / moved against it only / price flat and: far side shrank > 10% / near side grew > 10% (far side not shrinking > 10%) /
other activity (> 2 trades, far side grew or near side shrank > 10%) / small changes only (<= 2 trades, sizes within 10%, not unchanged) /
unchanged (no trade, sizes within 1%).
Per row: share of its fires and of nearby moments; a move its way within 300 / 150 ms after (fires: reconciled with the points table;
nearby: from the per-ms best prices); its 1 s value (c/sh); the row's share of all fires that landed before a move (state_value's 'ahead'
class, no move its way before and one within 300 ms after: the 46%).
Outputs (write-once): fire_breakdown.json; OUT/fire_breakdown_points.parquet (per point).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/fire_breakdown.py \
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
import quiet_local as QL  # noqa: E402
import timeline2 as T2  # noqa: E402

FE, DC, C, TR, WT = QB.FE, QB.DC, QB.C, QB.TR, QB.WT
W, TINY, SZ = QB.W, QB.TINY, QB.Q_SZ
ROWS = ('moved_its_way', 'moved_against_only', 'flat_far_shrank', 'flat_near_grew', 'flat_other_activity', 'flat_small_changes', 'flat_unchanged')


def next_move(st, t, d):
    """ms from t to the first move its way after t (a best price moves in direction d) in the hour's states; inf if none."""
    E, bid, bq, ask, aq = st
    db, da = np.diff(bid), np.diff(ask)
    Eu, Ed = E[1:][(db > TINY) | (da > TINY)], E[1:][(db < -TINY) | (da < -TINY)]
    out = np.full(len(t), np.inf)
    for dv, Ev in ((1., Eu), (-1., Ed)):
        m = d == dv
        if m.any() and len(Ev):
            k = np.searchsorted(Ev, t[m], 'right')
            out[m] = np.where(k < len(Ev), Ev[np.minimum(k, len(Ev) - 1)], np.inf)
    return out - t


def row_of(Z):
    flat = ~Z.price_changed
    far_s, near_g = Z.far1 < (1. - SZ) * Z.far0, Z.near1 > (1. + SZ) * Z.near0
    r = np.select([Z.their_prior, Z.against_prior, flat & far_s, flat & near_g, flat & ~Z.quiet, flat & Z.quiet & ~Z.static, flat & Z.static], list(ROWS), 'none')
    C.require((r != 'none').all(), 'the rows are not a partition')
    return r


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    od = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    fn, fp = (od or H) / 'fire_breakdown.json', (od or WT.OUT) / 'fire_breakdown_points.parquet'
    C.require(not fn.exists() and not fp.exists(), 'an output exists (write-once)')
    days = WT.DAYS if len(sys.argv) < 3 else (sys.argv[2],)
    C.require(all(x in WT.DAYS for x in days), 'not a declared day')
    X = pd.read_parquet(WT.OUT / 'state_value_points.parquet')
    S = X[(X.kind == 'send') & X.day.isin(days)].reset_index(drop=True)
    S['dec'] = np.arange(len(S))
    parts = []
    for day in days:
        day0 = float(WT.OD_day_ms(day))
        Sd = S[S.day == day]
        P = [pd.DataFrame(dict(dec=Sd.dec, day=day, t=Sd.t, d=Sd.d, shift_ms=0.))]
        for s in WT.SHIFTS:
            tt = Sd.t.to_numpy(float) + s
            k = T2.good(day, tt)
            P.append(pd.DataFrame(dict(dec=Sd.dec.to_numpy()[k], day=day, t=tt[k], d=Sd.d.to_numpy()[k], shift_ms=s)))
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
            Q['their_prior'], Q['against_prior'] = QL.moves_flags(st, t, d)
            Q['next_ms'] = next_move(st, t, d)
            n_tr = np.zeros(len(t))
            for side in (DC.BUY_AGGR, 1 - DC.BUY_AGGR):
                Es = tE[tm == side]
                n_tr += np.searchsorted(Es, t, 'right') - np.searchsorted(Es, t - W, 'right')
            Q['n_tr'] = n_tr
            parts.append(Q)
            n_done += len(Q)
            C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
        C.require(n_done == len(P), f'{day}: a point was not processed (no book hour)')
        print(day, len(Sd), len(P), flush=True)
    Z = pd.concat(parts, ignore_index=True)
    rf, rn = (Z.far1 - Z.far0).abs() / Z.far0, (Z.near1 - Z.near0).abs() / Z.near0
    Z['static'] = ~Z.price_changed & (Z.n_tr == 0) & (rf <= QB.SAME) & (rn <= QB.SAME)
    Z['quiet'] = ~Z.price_changed & (Z.n_tr <= QB.Q_TR) & (rf <= SZ) & (rn <= SZ)
    Z['row'] = row_of(Z)
    Z = Z.merge(S[['dec', 'far', 'near', 'momentum', 'bounce', 'next_move', 'v1']], on='dec', how='left', validate='many_to_one')
    Z.loc[Z.shift_ms != 0., 'v1'] = np.nan                                 # a value belongs to the fire only
    D = Z[Z.shift_ms == 0.]
    C.require(len(D) == len(S), 'a decision is missing')
    C.require((np.abs(D.far1 - D.far) < 1e-9).all() and D.their_prior.eq(D.momentum).all() and D.against_prior.eq(D.bounce).all(), 'the state differs from the points table')
    inside = D.next_move <= W
    C.require((np.abs(D.next_ms[inside] - D.next_move[inside]) < 1e-6).all() and (D.next_ms[~inside] > W).all(), 'the next move differs from the points table')
    Z['move300'], Z['move150'] = Z.next_ms <= W, Z.next_ms <= 150.
    Z['ahead'] = ~Z.their_prior & Z.move300
    N = Z[Z.shift_ms != 0.]
    D = Z[Z.shift_ms == 0.]
    mean = lambda x: float(np.nanmean(x)) if np.isfinite(np.asarray(x, float)).any() else None
    rows = {}
    for r in ROWS:
        a, b = D[D.row == r], N[N.row == r]
        rows[r] = dict(fires_n=int(len(a)), fires_share=float(len(a) / len(D)), nearby_n=int(len(b)), nearby_share=float(len(b) / len(N)),
                       fires_move300=mean(a.move300), nearby_move300=mean(b.move300), fires_move150=mean(a.move150), nearby_move150=mean(b.move150),
                       fires_v1_c=mean(a.v1), fires_v1_n=int(a.v1.notna().sum()), share_of_all_fires_ahead=float(a.ahead.sum() / len(D)),
                       fires_far_btc_p50=float(a.far1.median()) if len(a) else None, nearby_far_btc_p50=float(b.far1.median()) if len(b) else None)
    out = dict(asof='2026-11-26', days=list(days), window_ms=W, n_fires=int(len(D)), n_nearby=int(len(N)), size_threshold=SZ, trades_max_small=QB.Q_TR,
               ahead_share=float(D.ahead.mean()), after_a_move_share=float(D.their_prior.mean()), no_move_share=float((~D.their_prior & ~D.move300).mean()),
               fires_move300=float(D.move300.mean()), nearby_move300=float(N.move300.mean()), fires_v1_c=mean(D.v1), rows=rows)
    C.require(abs(sum(v['fires_share'] for v in rows.values()) - 1.) < 1e-12 and abs(sum(v['nearby_share'] for v in rows.values()) - 1.) < 1e-12, 'shares do not sum to 1')
    C.require(abs(sum(v['share_of_all_fires_ahead'] for v in rows.values()) - out['ahead_share']) < 1e-12, 'the ahead shares do not sum')
    if tuple(days) == tuple(WT.DAYS):
        sv = json.loads((H / 'state_value.json').read_text())['decisions']
        C.require(abs(out['ahead_share'] - sv['by_class']['ahead_move_300']['share']) < 1e-12 and abs(out['after_a_move_share'] - sv['by_class']['after_a_move']['share']) < 1e-12,
                  'the classes differ from state_value')
    Z.drop(columns=['far', 'near', 'momentum', 'bounce', 'next_move']).to_parquet(fp, index=False)
    TR.dump(fn, out)
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
    for r, v in rows.items():
        print(f"{r:22s} fires {v['fires_share']:6.1%} nearby {v['nearby_share']:6.1%} move300 {v['fires_move300'] or 0:6.1%} vs {v['nearby_move300'] or 0:6.1%} "
              f"v1 {v['fires_v1_c'] if v['fires_v1_c'] is not None else float('nan'):+.2f} ahead {v['share_of_all_fires_ahead']:6.1%} far {v['fires_far_btc_p50']} / {v['nearby_far_btc_p50']}")


if __name__ == '__main__':
    main()
