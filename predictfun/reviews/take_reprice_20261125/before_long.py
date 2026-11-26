"""Longer windows and Predict before the top taker's fires (USER 2026-11-26: "we check the longer window > 300ms, maybe for those 25% little
activity class, there was a trend, and the trend stops and it shows a calm window, then taker assumes it would be a reversion, or something
happens in predict fun, when nothing happens, but some takers filled on predict fun, then taker fill it back"). Readout; the four consumed
days (it traded on three); no strategy change.
Points: fire_breakdown_points.parquet (its 5,930 fires, t = arrival at Predict, and nearby moments = the same times +-5..30 s) with their
breakdown row; each point keeps its fire's market (the burst's market); a nearby moment outside that market's 5-minute window is left out.
Binance futures mid (bookTicker on E, per-ms states, the mid at its last change): the change its way over (t - 1 s, t], (t - 30 s, t] and
over (t - 5 s, t - 300 ms] (the trend before the 300 ms window), in bps; declared: against <= -1 bp, its way >= +1 bp, else flat.
Predict (winners_legs, timed legs only: websocket or book-drop timed; untimed legs counted as coverage): fills by other takers in the same
market over (t - 5 s, t] and (t - 1 s, t], shares in its direction / against it (a leg's direction = +1 when it buys UP); its own earlier
fills in the market over (t - 5 s, t]. Declared: 'others against' = other takers' shares against its direction > those with it in 5 s.
Added after the one-day rehearsal (11-01 seen: other takers fill within 5 s at ~every moment): the Predict price move its way
over 5 s = d x (median UP-space fill price of timed legs in (t - 1 s, t] - the median in (t - 5 s, t - 4 s]), c; against <= -1 c, its way >= +1 c.
Readings per row group (quiet / far side shrank / other flat / moved its way) and all: fires vs nearby shares; for nearby moments with no
prior Binance move its way, the Binance move within 300 ms after by condition (a public signal?), and after its fires.
Outputs (write-once): before_long.json; OUT/before_long_points.parquet.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/before_long.py \
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
import cross_venue as CV  # noqa: E402

DC, C, TR, WT = FE.DC, FE.C, FE.TR, FE.WT
BP, LONG, PRE, W5, W1 = 1., 30000., 5000., 5000., 1000.
TOP = '0x7cab1c46a5B0557008198b92697F36df855ac0AD'
GROUPS = ('quiet', 'far_shrank', 'other_flat', 'moved_its_way')
QS = (.1, .25, .5, .75, .9)


def day_mid(day):
    """Day-level (E, mid) at every mid change, and a per-hour file flag."""
    Es, Ms = [], []
    for h, bk in DC.book_hours(day):
        E, bid, bq, ask, aq = FE.collapse(*bk)
        mid = (bid + ask) / 2.
        k = np.r_[True, np.abs(np.diff(mid)) > 1e-9]
        Es.append(E[k]); Ms.append(mid[k])
    E, M = np.concatenate(Es), np.concatenate(Ms)
    o = np.argsort(E, kind='stable')
    E, M = E[o], M[o]
    keep = np.r_[True, np.diff(E) > 0]
    hrs = np.array([(DC.RAW / f'bookTicker/BTCUSDT/{day}_{h:02d}.csv.gz').exists() for h in range(24)])
    return E[keep], M[keep], hrs


def mid_at(E, M, hrs, day0, x):
    j = np.searchsorted(E, x, 'right') - 1
    h = ((x - day0) // 3600000).astype(int)
    ok = (j >= 0) & (h >= 0) & (h < 24) & hrs[np.clip(h, 0, 23)]
    return np.where(ok, M[np.maximum(j, 0)], np.nan)


def qs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {f'p{int(q * 100)}': float(np.quantile(x, q)) for q in QS} if len(x) else None


def share(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    od = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    fn, fp = (od or H) / 'before_long.json', (od or WT.OUT) / 'before_long_points.parquet'
    C.require(not fn.exists() and not fp.exists(), 'an output exists (write-once)')
    days = WT.DAYS if len(sys.argv) < 3 else (sys.argv[2],)
    C.require(all(x in WT.DAYS for x in days), 'not a declared day')
    P = pd.read_parquet(WT.OUT / 'fire_breakdown_points.parquet', columns=['dec', 'day', 't', 'd', 'shift_ms', 'row', 'their_prior', 'move300'])
    P = P[P.day.isin(days)].reset_index(drop=True)
    P['grp'] = P.row.map(CV.ROWG)
    P['fire'] = P.shift_ms == 0.
    S = pd.read_parquet(WT.OUT / 'state_value_points.parquet')
    S = S[S.kind == 'send'].reset_index(drop=True)
    S['dec'] = np.arange(len(S))                                             # fire_breakdown's numbering (all days, file order)
    c = float(json.loads((H / 'fire_events.json').read_text())['groups']['TOP1']['arrival_offset_ms'])
    A = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet')
    A = A[A.t_set.notna() & (A['rank'] == 1)]
    C.require(set(A.taker) == {TOP}, 'the top taker differs')
    B = A.sort_values(['burst', 'fire'], kind='stable').drop_duplicates('burst', keep='first').reset_index(drop=True)
    B['a'], B['d'] = B.fire - c, B.d.astype(float)
    Mk = S[['dec', 'day', 't', 'd']].merge(B[['day', 'a', 'd', 'mid']].rename(columns={'a': 't'}), on=['day', 't', 'd'], how='left')
    n_multi = int(Mk.dec.duplicated().sum())
    Mk = Mk.drop_duplicates('dec', keep='first')
    C.require(Mk.mid.notna().all() and len(Mk) == len(S), 'a fire without its market')
    P = P.merge(Mk[['dec', 'mid']], on='dec', how='left', validate='many_to_one')
    Q = P[P.fire].merge(S[['dec', 'day', 't', 'd']], on='dec', suffixes=('', '_s'))
    C.require(len(Q) == int(P.fire.sum()) and (Q.day == Q.day_s).all() and (Q.t == Q.t_s).all() and (Q.d == Q.d_s).all(), 'the fire numbering differs from fire_breakdown')
    cols = {k: np.full(len(P), np.nan) for k in ('r1s', 'r30s', 'r5s_pre', 'oth5_with', 'oth5_against', 'oth1_with', 'oth1_against', 'own5', 'n_oth5', 'pf5')}
    inwin = np.zeros(len(P), bool)
    cov = {}
    for day in days:
        day0 = float(WT.OD_day_ms(day))
        m = (P.day == day).to_numpy()
        t, d, mids = P.t.to_numpy(float)[m], P.d.to_numpy(float)[m], P.mid.to_numpy()[m]
        E, M, hrs = day_mid(day)
        mt = mid_at(E, M, hrs, day0, t)
        for k, x0, x1 in (('r1s', t - W1, t), ('r30s', t - LONG, t), ('r5s_pre', t - PRE, t - 300.)):
            a, b = mid_at(E, M, hrs, day0, x0), (mt if x1 is t else mid_at(E, M, hrs, day0, x1))
            cols[k][m] = d * 1e4 * np.log(b / a)
        reg = WT.MM.registry(day).set_index('mid')
        ws = pd.Series(mids).map(reg.ws).to_numpy(float) * 1000.
        C.require(np.isfinite(ws).all(), 'a market without its window')
        iw = (t >= ws) & (t < ws + 300000.)
        inwin[m] = iw
        L = pd.read_parquet(WT.OUT / f'winners_legs_{day}.parquet', columns=['mid', 'side', 'shares', 't_ms2', 'how2', 'taker', 'ambiguous', 'ptick'])
        cov[day] = dict(legs=int(len(L)), timed=float(L.how2.isin([1, 3]).mean()))
        cov[day]['ambiguous'] = float(L.ambiguous.mean())
        L = L[L.how2.isin([1, 3]) & L.t_ms2.notna() & ~L.ambiguous].copy()          # ambiguous legs: taker unknown, left out
        L['ld'] = np.where(L.side == 1, 1., -1.)
        L['own'] = L.taker == TOP
        L = L.sort_values(['mid', 't_ms2'], kind='stable')
        G = {int(k): (g.t_ms2.to_numpy(float), g.ld.to_numpy(), g.shares.to_numpy(float), g.own.to_numpy(bool), g.ptick.to_numpy(float)) for k, g in L.groupby('mid')}
        res = {k: np.full(len(t), np.nan) for k in ('oth5_with', 'oth5_against', 'oth1_with', 'oth1_against', 'own5', 'n_oth5', 'pf5')}
        for i in np.flatnonzero(iw):
            g = G.get(int(mids[i]))
            for k in res:
                res[k][i] = 0. if k != 'pf5' else np.nan
            if g is None:
                continue
            tl, ld, sh, own, pk = g
            a1, b1 = np.searchsorted(tl, t[i] - W1, 'right'), np.searchsorted(tl, t[i], 'right')
            a0, b0 = np.searchsorted(tl, t[i] - W5, 'right'), np.searchsorted(tl, t[i] - W5 + W1, 'right')
            if b1 > a1 and b0 > a0:
                res['pf5'][i] = d[i] * (np.median(pk[a1:b1]) - np.median(pk[a0:b0]))
            for w, tag in ((W5, '5'), (W1, '1')):
                a, b = np.searchsorted(tl, t[i] - w, 'right'), np.searchsorted(tl, t[i], 'right')
                o, l_, s_ = own[a:b], ld[a:b], sh[a:b]
                res[f'oth{tag}_with'][i] = s_[~o & (l_ == d[i])].sum()
                res[f'oth{tag}_against'][i] = s_[~o & (l_ != d[i])].sum()
                if tag == '5':
                    res['own5'][i] = s_[o].sum()
                    res['n_oth5'][i] = float((~o).sum())
        for k, x in res.items():
            cols[k][m] = x
        C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
        print(day, int(m.sum()), int(iw.sum()), flush=True)
    Z = pd.concat([P, pd.DataFrame(cols)], axis=1)
    Z['in_window'] = inwin
    n_fire_out = int((~Z.in_window & Z.fire).sum())                                    # counted; kept with its features
    Z = Z[Z.in_window | Z.fire].reset_index(drop=True)
    Z['trend'] = np.select([Z.r5s_pre <= -BP, Z.r5s_pre >= BP, Z.r5s_pre.notna()], ['against', 'its_way', 'flat'], 'none')
    Z['oth_against'] = Z.oth5_against > Z.oth5_with
    Z['pf_move'] = np.select([Z.pf5 <= -1., Z.pf5 >= 1., Z.pf5.notna()], ['against', 'its_way', 'flat'], 'none')
    Z['any_oth5'] = Z.n_oth5 > 0
    Fz, N = Z[Z.fire], Z[~Z.fire]

    def blk(F_, N_):
        return dict(fires_n=int(len(F_)), nearby_n=int(len(N_)),
                    trend_against=dict(fires=share(F_.trend == 'against'), nearby=share(N_.trend == 'against')),
                    trend_its_way=dict(fires=share(F_.trend == 'its_way'), nearby=share(N_.trend == 'its_way')),
                    trend_flat=dict(fires=share(F_.trend == 'flat'), nearby=share(N_.trend == 'flat')),
                    r5s_pre_bps=dict(fires=qs(F_.r5s_pre), nearby=qs(N_.r5s_pre)), r30s_bps=dict(fires=qs(F_.r30s), nearby=qs(N_.r30s)),
                    r1s_bps=dict(fires=qs(F_.r1s), nearby=qs(N_.r1s)),
                    any_other_fill_5s=dict(fires=share(F_.any_oth5), nearby=share(N_.any_oth5)),
                    other_fill_1s=dict(fires=share((F_.oth1_with + F_.oth1_against) > 0), nearby=share((N_.oth1_with + N_.oth1_against) > 0)),
                    others_against_5s=dict(fires=share(F_.oth_against), nearby=share(N_.oth_against)),
                    others_against_shares_5s=dict(fires=float(F_.oth5_against.mean()), nearby=float(N_.oth5_against.mean())),
                    others_with_shares_5s=dict(fires=float(F_.oth5_with.mean()), nearby=float(N_.oth5_with.mean())),
                    own_fill_5s=dict(fires=share(F_.own5 > 0), nearby=share(N_.own5 > 0)),
                    predict_moved_against_5s=dict(fires=share(F_.pf_move[F_.pf_move != 'none'] == 'against'), nearby=share(N_.pf_move[N_.pf_move != 'none'] == 'against')),
                    predict_moved_its_way_5s=dict(fires=share(F_.pf_move[F_.pf_move != 'none'] == 'its_way'), nearby=share(N_.pf_move[N_.pf_move != 'none'] == 'its_way')),
                    predict_move_5s_c=dict(fires=qs(F_.pf5), nearby=qs(N_.pf5)), predict_move_coverage=dict(fires=share(F_.pf5.notna()), nearby=share(N_.pf5.notna())))
    out = dict(asof='2026-11-26', days=list(days), n_fires=int(len(Fz)), n_nearby=int(len(N)), n_nearby_outside_market=int((~inwin & ~P.fire.to_numpy()).sum()),
               fires_sharing_a_ms_with_another_burst=n_multi, fires_outside_market_window=n_fire_out, legs_coverage=cov, thresholds=dict(trend_bp=BP), all=blk(Fz, N), by_group={})
    for g in GROUPS:
        out['by_group'][g] = blk(Fz[Fz.grp == g], N[N.grp == g])
    npr = ~Z.their_prior.astype(bool)
    fol = {}
    for nm, cond in (('trend_against', Z.trend == 'against'), ('trend_its_way', Z.trend == 'its_way'), ('trend_flat', Z.trend == 'flat'),
                     ('others_against_5s', Z.oth_against), ('predict_moved_against_5s', Z.pf_move == 'against'), ('predict_moved_its_way_5s', Z.pf_move == 'its_way'),
                     ('predict_flat_5s', Z.pf_move == 'flat')):
        fol[nm] = dict(nearby=share(Z.move300[~Z.fire & npr & cond]), nearby_n=int((~Z.fire & npr & cond).sum()), fires=share(Z.move300[Z.fire & npr & cond]),
                       fires_n=int((Z.fire & npr & cond).sum()))
    out['follow_no_prior_move'] = fol
    Z.to_parquet(fp, index=False)
    TR.dump(fn, out)
    print(json.dumps(dict(n=(out['n_fires'], out['n_nearby'], out['n_nearby_outside_market']), multi=n_multi, cov=cov), indent=0))
    for g in ('all',) + GROUPS:
        b = out['all'] if g == 'all' else out['by_group'][g]
        print(f"{g:14s} n {b['fires_n']}/{b['nearby_n']} | trend against {b['trend_against']['fires']:.3f}/{b['trend_against']['nearby']:.3f} its way {b['trend_its_way']['fires']:.3f}/{b['trend_its_way']['nearby']:.3f} "
              f"flat {b['trend_flat']['fires']:.3f}/{b['trend_flat']['nearby']:.3f} | other fill 5s {b['any_other_fill_5s']['fires']:.3f}/{b['any_other_fill_5s']['nearby']:.3f} "
              f"1s {b['other_fill_1s']['fires']:.3f}/{b['other_fill_1s']['nearby']:.3f} | others against {b['others_against_5s']['fires']:.3f}/{b['others_against_5s']['nearby']:.3f} "
              f"| own 5s {b['own_fill_5s']['fires']:.3f}/{b['own_fill_5s']['nearby']:.3f} | pf against {b['predict_moved_against_5s']['fires']:.3f}/{b['predict_moved_against_5s']['nearby']:.3f} its way {b['predict_moved_its_way_5s']['fires']:.3f}/{b['predict_moved_its_way_5s']['nearby']:.3f}")
    print(json.dumps(fol, indent=0))


if __name__ == '__main__':
    main()
