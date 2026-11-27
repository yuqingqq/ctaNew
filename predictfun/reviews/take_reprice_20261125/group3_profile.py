"""The top taker's group-3 fires (its order lands after a Binance move its way had started): when, at what price, after which moves, and
what our chase model still misses (USER 2026-11-27: "can we check group 3 fires of top taker, what's the distribution of remaining time, firing
probability, include more information for analysis, check what we still miss"). Four consumed days; descriptive; declared before the run.
Fires: its 5,930 decisions (SMART_FIRES_EXPORT's table: arrival = settlement - 150 - c, its fill VWAP, shares, 1 s value, settlement) in
STATE_VALUE's classes (its stored points, joined exactly): group 3 = a Binance futures move its way in the 300 ms before its arrival, group 2 =
none before and one within 300 ms after, group 1 = neither.
Per decision (stage labels): the Binance move it follows (CHASE_TAKES' moves on every tick: onset, the tick index, ms since the onset and
since the last tick, ticks in the last 50 / 100 ms and in the next 300 ms, the mid's move its way since the onset (bps), aggressive volume
its way and against since the onset and over 100 ms, the best-level imbalance and far / near size at arrival, the mid's move over the next
300 ms / 1 s); Predict at its arrival on its market's recorded path (the book known at arrival: mid, the best price of its side, spread; the
mid known at the move's onset; the mid at its match and 1 s / 5 s after it); the spot fair value at arrival (DRIFT_VALUE's) and CHASE_TAKES'
chase gap; its pay-through = its VWAP minus the known best, its way. Statuses for decisions without a path, book or later mid.
Comparison sets (stage read; CHASE_TAKES' rows and held-out predictions, its counts reproduced: a check): every move at its first tick (the
opportunity set) and CHASE_TAKES' model top-10 % actions; its group-2 decisions for the price and time distributions.
Readings: (1) distributions (quantiles and bucket shares, buckets below) of time left, the price of its side (known best and mid), the move
so far, the timing, Predict's move since the onset, the chase and fair-value gaps, the Binance best level, the Predict spread; for its fires
also its pay-through, size, 1 s value, settlement; (2) firing probability = the share of moves it chases (an order of its landing after the
onset and within 300 ms of the move's last tick), by time left, side price, the move's size in its first 300 ms, our model's edge at the
onset and the move's best edge (deciles); (3) coverage: of its group-3 fires, the share whose move is in our population, has a model
top-10 % / 1 % action, and that action lands before its order; our model's percentile at the last tick before its arrival; its 1 s value
by that percentile; (4) what we still miss: the moves it chases against 5 un-chased moves each, same day and direction within 30 min, with
the nearest model edge at the onset: AUC of each of CHASE_TAKES' inputs at the onset; null: the largest |AUC - 0.5| over the inputs with
the chased label shuffled within matched sets (200 draws), inputs beyond its 95th percentile; the realised Predict move and the
conservative / optimistic value of taking at the onset, chased vs matched; the moves' later size (not an input); (5) our model's top-10 %
actions on moves it chased or not: value on both books.
Outputs (write-once): OUT/group3_profile/<day>.parquet (labels), group3_profile.json (read).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/group3_profile.py \
       selftest | labels <day> [<dir>] | read [<dir>]
"""
import collections
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import chase_takes as CT  # noqa: E402

TM, VR, MM, TR, C, F, WT, TK, TE = CT.TM, CT.VR, CT.MM, CT.TR, CT.C, CT.F, CT.WT, CT.TK, CT.TE
FE, TL, DC, DV, RT, CV, BLm = CT.FE, CT.TL, CT.DC, CT.DV, CT.RT, CT.CV, CT.BLm
OUT = WT.OUT / 'group3_profile'
DAYS, Q, fee = CT.DAYS, CT.Q, CT.fee
OUR_ARRIVAL = TM.FEED + TM.DECIDE + CT.NET                        # ms after the tick
N_NULL, K_MATCH, WIN_MS, SEED = 200, 5, 1800000., 20261127
INF = np.inf
EDGES = dict(time_left_s=[0, 30, 60, 120, 180, 240, 300.01], side_best_c=[0, 10, 20, 35, 50, 65, 80, 90, 100.01], side_mid_c=[0, 10, 20, 35, 50, 65, 80, 90, 100.01],
             move_bps=[-INF, .1, .25, .5, 1, 2, 4, INF], tick_index=[1, 2, 3, 4, 6, 10, INF], arrival_after_onset_ms=[0, 10, 25, 50, 100, 200, 300, INF],
             arrival_after_last_tick_ms=[0, 5, 10, 25, 50, 100, 200, 300.01], predict_move_since_onset_c=[-INF, -.5, .5, 2, 5, 10, INF],
             chase_gap_c=[-INF, -2, 0, 2, 5, 10, INF], fv_gap_c=[-INF, -10, -5, 0, 5, 10, INF], imbalance=[-1, -.6, -.3, 0, .3, .6, 1.0001],
             far_size_btc=[0, .1, .5, 1, 3, 10, INF], sweep_100ms_btc=[0, 1e-9, .5, 1, 3, 10, INF], spread_c=[0, 1.5, 2.5, 5, 10, INF],
             pay_through_c=[-INF, -.5, .5, 2, 5, 10, INF], shares=[0, 10, 29.5, 30.5, 50, 100, INF], value_1s_c=[-INF, -5, -1, 0, 1, 5, INF])
QS = (.1, .25, .5, .75, .9)


def bursts():
    """Its 5,930 decisions with STATE_VALUE's class (exact join) -> DataFrame."""
    Fx = pd.read_csv(WT.OUT / 'smart_fires_export' / 'top_taker_fires.csv.gz')
    Fx['day'] = pd.to_datetime(Fx.t_ms, unit='ms', utc=True).dt.strftime('%Y%m%d')
    Fx['d'] = np.where(Fx.bet == 'UP', 1., -1.)
    S = pd.read_parquet(WT.OUT / 'state_value_points.parquet', columns=['day', 'kind', 't', 'd', 'momentum', 'next_move', 'v1'])
    S = S[S.kind == 'send'].drop(columns=['kind'])
    B = Fx.merge(S, left_on=['day', 't_ms', 'd'], right_on=['day', 't', 'd'], how='left', validate='1:1')
    C.require(B.momentum.notna().all(), 'a decision without its STATE_VALUE class')
    C.require(np.allclose(B.v1.to_numpy(float), B.value_1s_c_per_share.to_numpy(float), equal_nan=True), 'the 1 s values differ between the two tables')
    mom = B.momentum.astype(bool).to_numpy()
    B['group'] = np.where(mom, 3, np.where(B.next_move.to_numpy(float) <= 300., 2, 1))
    return B.drop(columns=['t']).rename(columns={'t_ms': 't'})


def count_in(M, lo, hi):
    return np.searchsorted(M, hi, 'right') - np.searchsorted(M, lo, 'right')


# ---------------------------------------------------------------------------------------------------------------- labels
def labels(day, od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    B = bursts()
    B = B[B.day == day].sort_values('t', kind='stable').reset_index(drop=True)
    a, d = B.t.to_numpy(float), B.d.to_numpy(float)
    n = len(B)
    day0 = float(WT.OD_day_ms(day))
    o = {k: np.full(n, np.nan) for k in ('last_tick_ms', 'onset', 'k', 'mkey', 'ticks_50', 'ticks_100', 'ticks_next_300')}
    ev, _, _, _ = FE.day_data(day, pd.DataFrame(dict(t=np.zeros(0), d=np.zeros(0))))
    for dv in (1., -1.):
        m = d == dv
        M = TL.moves(ev, dv)
        if not m.any() or not len(M):
            continue
        mv, onset, k, _, _ = CT.moves_table(M)
        j = np.searchsorted(M, a[m], 'right') - 1
        okj, jj = j >= 0, np.maximum(j, 0)
        o['last_tick_ms'][m] = np.where(okj, a[m] - M[jj], np.nan)
        o['onset'][m] = np.where(okj, onset[jj], np.nan)
        o['k'][m] = np.where(okj, k[jj], np.nan)
        o['mkey'][m] = np.where(okj, mv[jj] * 2 + (1 if dv > 0 else 0), np.nan)
        o['ticks_50'][m], o['ticks_100'][m] = count_in(M, a[m] - 50., a[m]), count_in(M, a[m] - 100., a[m])
        o['ticks_next_300'][m] = count_in(M, a[m], a[m] + 300.)
    del ev
    g3 = B.group.to_numpy() == 3
    C.require(np.array_equal(o['last_tick_ms'] < 300., g3), 'the move before the arrival does not reproduce STATE_VALUE\'s group 3')
    # the Binance best level at arrival and the mid series (one pass over the book hours)
    imb, far, near = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    hour = ((a - day0) // 3600000).astype(int)
    mE, mM = [], []
    for h, bk in DC.book_hours(day):
        E, bid, bq, ask, aq = FE.collapse(*bk)
        mid = (bid + ask) / 2.
        kk = np.r_[True, np.abs(np.diff(mid)) > 1e-9]
        mE.append(E[kk]); mM.append(mid[kk])
        sel = np.flatnonzero(hour == h)
        if len(sel):
            j = np.searchsorted(E, a[sel], 'right') - 1
            okj, jj = j >= 0, np.maximum(j, 0)
            up = d[sel] > 0
            fq, nq = np.where(up, aq[jj], bq[jj]), np.where(up, bq[jj], aq[jj])
            far[sel], near[sel] = np.where(okj, fq, np.nan), np.where(okj, nq, np.nan)
            imb[sel] = np.where(okj, (nq - fq) / np.maximum(nq + fq, 1e-12), np.nan)
    E_, M_ = np.concatenate(mE), np.concatenate(mM)
    oo = np.argsort(E_, kind='stable'); E_, M_ = E_[oo], M_[oo]
    keep = np.r_[np.diff(E_) > 0, True]; E_, M_ = E_[keep], M_[keep]
    hrs = np.array([(DC.RAW / f'bookTicker/BTCUSDT/{day}_{h:02d}.csv.gz').exists() for h in range(24)])
    mid_at = lambda x: BLm.mid_at(E_, M_, hrs, day0, x)
    m_a = mid_at(a)
    on = o['onset']
    o['mv_bps'] = d * 1e4 * np.log(m_a / mid_at(np.where(np.isfinite(on), on - 1., a)))
    o['bn_after_300_bps'] = d * 1e4 * np.log(mid_at(a + 300.) / m_a)
    o['bn_after_1s_bps'] = d * 1e4 * np.log(mid_at(a + 1000.) / m_a)
    del E_, M_
    tr = DC.read_trades(day).drop_duplicates('id')
    tr = tr[(tr.p > 0) & (tr.q > 0)].sort_values('E', kind='stable')
    buy = tr.m.to_numpy() == DC.BUY_AGGR
    Eb, Es = tr.E.to_numpy(float)[buy], tr.E.to_numpy(float)[~buy]
    cb, cs = np.r_[0., np.cumsum(tr.q.to_numpy(float)[buy])], np.r_[0., np.cumsum(tr.q.to_numpy(float)[~buy])]
    del tr
    up = d > 0
    lo = np.where(np.isfinite(on), on - 1., a)
    vb, vs = CT.vol_in(Eb, cb, lo, a), CT.vol_in(Es, cs, lo, a)
    o['its_since_onset_btc'], o['ag_since_onset_btc'] = np.where(up, vb, vs), np.where(up, vs, vb)
    vb, vs = CT.vol_in(Eb, cb, a - 100., a), CT.vol_in(Es, cs, a - 100., a)
    o['its_100_btc'], o['ag_100_btc'] = np.where(up, vb, vs), np.where(up, vs, vb)
    o['imb_0'], o['far_0'], o['near_0'] = imb, far, near
    # Predict on its market's recorded path
    for k_ in ('md_a', 'lim_known', 'spread_a', 'md_onset', 'm_match', 'm1', 'm5'):
        o[k_] = np.full(n, np.nan)
    status = np.full(n, 'market_not_loaded', dtype=object)
    reg = MM.registry(day).set_index('mid')
    rg = pd.read_parquet(MM.S1 / day / 'registry.parquet').set_index('mid')
    mk = B.market_id.astype(int).to_numpy()
    by_mid = pd.Series(np.arange(n)).groupby(mk).indices
    src = MM.KP.Tape(); off = src.offset(day)
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in by_mid]
        if not mids:
            continue
        for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
            ii = by_mid[m]
            if lst != 'loaded':
                status[ii] = f'loader_{lst}'
                continue
            p, st, info = VR.vm_path(W, float(reg.loc[m, 'ws']) * 1000., 50.)
            if p is None:
                status[ii] = f'path_{st}'
                continue
            t0, end = float(info['t_start']), float(info['end_rel'])
            f = np.asarray(p['frames'], float)
            fk = f[:, 1]
            for i in ii:
                g_ = a[i] - t0
                if g_ < 1000. or g_ + TM.HOLD + 1000. > end:
                    status[i] = 'outside_path'
                    continue
                jk = int(np.searchsorted(fk, g_, 'right') - 1)
                if jk < 0 or not (0 <= f[jk, 2] < f[jk, 12] <= 100):
                    status[i] = 'invalid_known_book'
                    continue
                bid, ask = f[jk, 2], f[jk, 12]
                o['md_a'][i], o['lim_known'][i], o['spread_a'][i] = (bid + ask) / 2., (ask if d[i] > 0 else bid), ask - bid
                if np.isfinite(on[i]):
                    jo = int(np.searchsorted(fk, on[i] + TM.FEED + TM.DECIDE - t0, 'right') - 1)
                    o['md_onset'][i] = TM.mid_of(f[jo]) if jo >= 0 else np.nan
                jm = int(np.searchsorted(f[:, 0], g_ + TM.HOLD, 'right') - 1)
                o['m_match'][i] = TM.mid_of(f[jm])
                for k_, hz in (('m1', 1000.), ('m5', 5000.)):
                    x = g_ + TM.HOLD + hz
                    ja = int(np.searchsorted(f[:, 0], x, 'right') - 1)
                    o[k_][i] = TM.mid_of(f[ja]) if (x <= end and ja >= 0 and x - f[ja, 0] <= 1000.) else np.nan
                status[i] = 'ok' if np.isfinite(o['m1'][i]) else 'no_mid_after'
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    tv, pv, _, _, cov, _ = CV.load('binance_spot', day)
    SV = RT.SpotVol(tv, pv, cov[0], cov[1])
    j = np.searchsorted(tv, a - 100., 'right') - 1
    spot = np.where(j >= 0, pv[np.maximum(j, 0)], np.nan)
    drift = 1e4 * np.log(spot / rg.loc[mk, 'start_price'].to_numpy(float))
    tau = (rg.loc[mk, 'we'].to_numpy(float) * 1000. - a) / 1000.
    z, fv, delta = DV.fair(drift, SV.vol(a), tau)
    o['fv_tau_s'], o['fv_gap_its'], o['fv_delta'] = tau, d * (fv - o['md_a']), delta
    o['pf_move_since_onset_c'] = d * (o['md_a'] - o['md_onset'])
    o['chase_implied_c'] = delta * o['mv_bps']
    o['chase_gap_c'] = o['chase_implied_c'] - o['pf_move_since_onset_c']
    o['pay_through_c'] = d * (B.avg_price_up_c.to_numpy(float) - o['lim_known'])
    o['markout_1s_c'] = d * (o['m1'] - o['md_a'])
    L = pd.concat([B[['day', 't', 'd', 'group', 'market_id', 'time_left_s', 'avg_price_up_c', 'shares', 'orders', 'v1', 'bet_won', 'payout_up']].reset_index(drop=True),
                   pd.DataFrame(o)], axis=1)
    L['status'] = status
    od.mkdir(parents=True, exist_ok=True)
    L.to_parquet(fp, index=False)
    cnt = collections.Counter(zip(L.group, L.status))
    TR.dump(od / f'{day}.json', dict(asof='2026-11-27', day=day, decisions=int(n), status={f'group{g}_{s}': int(v) for (g, s), v in cnt.items()}))
    print(day, n, dict(cnt), 'rss', round(F.rss_mib()), flush=True)


# ---------------------------------------------------------------------------------------------------------------- read
def dist(x, edges):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0)
    h = np.histogram(np.clip(x, edges[0] if np.isfinite(edges[0]) else -1e18, edges[-1] - 1e-12 if np.isfinite(edges[-1]) else 1e18),
                     bins=[e if np.isfinite(e) else (-1e18 if e < 0 else 1e18) for e in edges])[0]
    return dict(n=int(len(x)), mean=float(x.mean()), q={f'p{int(100 * q)}': float(np.quantile(x, q)) for q in QS},
                buckets=[dict(lo=float(lo), hi=float(hi), share=float(c_ / len(x))) for lo, hi, c_ in zip(edges[:-1], edges[1:], h)])


def match_sets(day, d, t, e, pos, cand, k=K_MATCH, win=WIN_MS):
    """For each positive row: up to k candidate rows of the same day and direction within win ms with the nearest e -> (n_pos x k) indices, -1 empty."""
    out = np.full((int(pos.sum()), k), -1, np.int64)
    ip = np.flatnonzero(pos)
    groups = pd.Series(np.flatnonzero(cand)).groupby([day[cand], d[cand]]).apply(np.asarray).to_dict()
    for r, i in enumerate(ip):
        c_ = groups.get((day[i], d[i]))
        if c_ is None:
            continue
        c_ = c_[np.abs(t[c_] - t[i]) <= win]
        c_ = c_[np.isfinite(e[c_])]
        if not len(c_) or not np.isfinite(e[i]):
            continue
        o = np.argsort(np.abs(e[c_] - e[i]), kind='stable')[:k]
        out[r, :len(o)] = c_[o]
    return ip, out


def auc_matrix(Xv, lab, sets_lab=None):
    """AUC of every column of Xv (NaN dropped per column) for lab; ranks precomputed once."""
    N, Fn = Xv.shape
    V = np.isfinite(Xv)
    R = np.zeros_like(Xv)
    for j in range(Fn):
        R[V[:, j], j] = rankdata(Xv[V[:, j], j])

    def one(y):
        y = y.astype(bool)
        n1 = (V & y[:, None]).sum(0).astype(float); nv = V.sum(0).astype(float); n0 = nv - n1
        s = (R * y[:, None]).sum(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where((n1 > 0) & (n0 > 0), (s - n1 * (n1 + 1) / 2.) / (n1 * n0), np.nan)
    return one, one(lab)


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'group3_profile.json'
    C.require(not fn.exists(), 'group3_profile.json exists (write-once)')
    src = od or OUT
    G = pd.concat([pd.read_parquet(src / f'{dh}.parquet') for dh in DAYS if (src / f'{dh}.parquet').exists()], ignore_index=True)
    CJ = json.loads((H / 'chase_takes.json').read_text())
    X, feats, _ = CT.frame(CT.OUT)
    Pr = pd.read_parquet(CT.OUT / 'predictions.parquet', columns=['day', 't', 'd', 'pred_1s'])
    X = X.merge(Pr, on=['day', 't', 'd'], how='left', validate='1:1')
    C.require(X.pred_1s.notna().all(), 'a chase tick without its held-out prediction')
    d, lim, md, day = X.d.to_numpy(float), X.lim.to_numpy(float), X.md.to_numpy(float), X.day.to_numpy()
    key = X.key.to_numpy()
    cost = d * (lim - md) + fee(lim)
    e1 = X.pred_1s.to_numpy(float) - cost
    thr10, thr1 = np.quantile(e1, .9), np.quantile(e1, .99)
    on_i = CT.first_per_move(X.mv_k.to_numpy(float) == 1, key)
    t10 = CT.first_per_move(e1 >= thr10, key)
    t1 = CT.first_per_move(e1 >= thr1, key)
    C.require(len(on_i) == CJ['selections']['every_move_onset']['actions'] and len(t10) == CJ['selections']['model1s_top10']['actions']
              and len(t1) == CJ['selections']['model1s_top1']['actions'], 'CHASE_TAKES\' selections not reproduced')
    pctl = (rankdata(e1) - .5) / len(e1)
    chased = np.isfinite(X.tt_chase_lag_ms.to_numpy(float))
    out = dict(asof='2026-11-27', rehearsal=reh, decisions={f'group{g}': int((G.group == g).sum()) for g in (1, 2, 3)},
               label_status={f'group{g}': G[G.group == g].status.value_counts().to_dict() for g in (1, 2, 3)}, chase_reproduced=True,
               distributions={}, firing_probability={}, coverage={}, what_we_miss={}, our_top10_by_chase={})
    # ---- (1) distributions
    okG = G.status.to_numpy() == 'ok'
    gd = G.d.to_numpy(float)
    side = lambda x, dd: np.where(dd > 0, x, 100. - x)
    fG = dict(time_left_s=G.fv_tau_s, side_best_c=side(G.lim_known.to_numpy(float), gd), side_mid_c=side(G.md_a.to_numpy(float), gd), move_bps=G.mv_bps,
              tick_index=G.k, arrival_after_onset_ms=G.t - G.onset, arrival_after_last_tick_ms=G.last_tick_ms, predict_move_since_onset_c=G.pf_move_since_onset_c,
              chase_gap_c=G.chase_gap_c, fv_gap_c=G.fv_gap_its, imbalance=G.imb_0, far_size_btc=G.far_0, sweep_100ms_btc=G.its_100_btc, spread_c=G.spread_a,
              pay_through_c=G.pay_through_c, shares=G.shares, value_1s_c=G.v1)
    fX = dict(time_left_s=X.fv_tau_s, side_best_c=side(lim, d), side_mid_c=X.pf_its_price, move_bps=X.mv_bps, tick_index=X.mv_k,
              arrival_after_onset_ms=X.mv_since_ms + OUR_ARRIVAL, arrival_after_last_tick_ms=pd.Series(np.full(len(X), OUR_ARRIVAL)),
              predict_move_since_onset_c=X.pf_mid_ret_onset, chase_gap_c=X.chase_gap_c, fv_gap_c=X.fv_gap_its, imbalance=X.b_imb1,
              far_size_btc=np.exp(X.b_lfar1.to_numpy(float)), sweep_100ms_btc=X.its_100_btc, spread_c=X.pf_spread)
    sets = dict(its_group3=('G', okG & (G.group.to_numpy() == 3)), its_group2=('G', okG & (G.group.to_numpy() == 2)),
                every_move_onset=('X', on_i), our_model_top10=('X', t10))
    for fld, edges in EDGES.items():
        out['distributions'][fld] = {}
        for snm, (which, sel) in sets.items():
            src_ = fG if which == 'G' else fX
            if fld not in src_:
                continue
            if snm == 'its_group2' and fld in ('move_bps', 'tick_index', 'arrival_after_onset_ms', 'arrival_after_last_tick_ms', 'predict_move_since_onset_c', 'chase_gap_c'):
                continue                                                  # group 2 has no move before it
            v = np.asarray(src_[fld], float)
            out['distributions'][fld][snm] = dist(v[sel], edges)
    g3 = okG & (G.group.to_numpy() == 3)
    out['distributions']['its_group3_outcomes'] = dict(won=float(np.nanmean(G.bet_won.to_numpy(float)[g3])), v1_mean_c=float(np.nanmean(G.v1.to_numpy(float)[g3])),
                                                      markout_1s_mean_c=float(np.nanmean(G.markout_1s_c.to_numpy(float)[g3])),
                                                      bn_after_300_bps_mean=float(np.nanmean(G.bn_after_300_bps.to_numpy(float)[g3])),
                                                      ticks_next_300_mean=float(np.nanmean(G.ticks_next_300.to_numpy(float)[g3])))
    # ---- (2) firing probability: the share of moves it chases
    win_size = X.groupby('key').mv_bps.max()                                  # the move's size in its first 300 ms (not known at the onset)
    best_e = pd.Series(e1, index=X.index).groupby(X.key).max()
    Xo = X.iloc[on_i]
    co = chased[on_i]
    fp = dict(overall=float(co.mean()), moves=int(len(on_i)))
    for nm, v, edges in (('time_left_s', Xo.fv_tau_s.to_numpy(float), EDGES['time_left_s']), ('side_best_c', side(lim[on_i], d[on_i]), EDGES['side_best_c']),
                         ('move_size_first_300ms_bps', win_size.loc[Xo.key].to_numpy(float), [-INF, .1, .25, .5, 1, 2, 4, INF])):
        b = np.digitize(v, [e for e in edges[1:-1]])
        fp[nm] = [dict(lo=float(edges[i]), hi=float(edges[i + 1]), moves=int((b == i).sum()), chased=float(co[b == i].mean()) if (b == i).any() else None) for i in range(len(edges) - 1)]
    for nm, v in (('our_edge_at_onset', e1[on_i]), ('our_best_edge_in_move', best_e.loc[Xo.key].to_numpy(float))):
        qd = np.quantile(v, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(qd[1:-1], v, 'right'), 0, 9)
        fp[nm] = [dict(decile=int(i + 1), lo_c=float(qd[i]), hi_c=float(qd[i + 1]), moves=int((b == i).sum()), chased=float(co[b == i].mean())) for i in range(10)]
    out['firing_probability'] = fp
    # ---- (3) coverage of its group-3 fires
    G3 = G[g3].copy()
    G3['key'] = G3.day.astype(str) + ':' + G3.mkey.astype(np.int64).astype(str)
    in_pop = G3.key.isin(set(key))
    a10 = pd.Series(X.t.to_numpy(float)[t10], index=key[t10]); a1_ = pd.Series(X.t.to_numpy(float)[t1], index=key[t1])
    has10, has1 = G3.key.isin(a10.index), G3.key.isin(a1_.index)
    ours_first10 = has10 & (a10.reindex(G3.key).to_numpy() + OUR_ARRIVAL < G3.t.to_numpy())
    Xt = X[['key', 't']].assign(p=pctl)
    last_p = np.full(len(G3), np.nan)
    by_key = Xt.groupby('key').indices
    tt_ = Xt.t.to_numpy(float); pp = Xt.p.to_numpy(float)
    for r, (k_, aa) in enumerate(zip(G3.key, G3.t.to_numpy(float))):
        ii = by_key.get(k_)
        if ii is None:
            continue
        ii = ii[tt_[ii] + TM.FEED + TM.DECIDE <= aa]                     # our decisions before its arrival
        if len(ii):
            last_p[r] = pp[ii[np.argmax(tt_[ii])]]
    v1g = G3.v1.to_numpy(float)
    pb = np.digitize(last_p, [.5, .9, .99])
    out['coverage'] = dict(group3_ok=int(len(G3)), move_in_our_population=float(in_pop.mean()), with_model_top10_action=float(has10.mean()),
                           with_model_top1_action=float(has1.mean()), our_top10_action_lands_first=float(ours_first10.mean()),
                           beyond_our_300ms_window=float((G3.t - G3.onset > CT.WIN_MS).mean()),
                           our_percentile_before_its_arrival=dist(last_p, [0, .5, .9, .99, 1.0001]),
                           its_v1_by_our_percentile=[dict(bucket=nm, fires=int(((pb == i) & np.isfinite(last_p)).sum()),
                                                          v1_mean_c=float(np.nanmean(v1g[(pb == i) & np.isfinite(last_p)])) if ((pb == i) & np.isfinite(last_p)).any() else None)
                                                     for i, nm in enumerate(('below p50', 'p50-p90', 'p90-p99', 'top 1%'))],
                           its_v1_no_decision_of_ours_before_it_c=float(np.nanmean(v1g[np.isnan(last_p)])) if np.isnan(last_p).any() else None,
                           no_decision_of_ours_before_it=int(np.isnan(last_p).sum()),
                           its_v1_not_in_population_c=float(np.nanmean(v1g[~in_pop.to_numpy()])) if (~in_pop).any() else None)
    # ---- (4) what we still miss: chased moves vs un-chased moves with the nearest edge at the onset
    day_o, d_o, t_o = day[on_i], d[on_i], X.t.to_numpy(float)[on_i]
    ip, M_ = match_sets(day_o, d_o, t_o, e1[on_i], co, ~co)
    full = (M_ >= 0).all(1)
    ip, M_ = ip[full], M_[full]
    rows = np.r_[ip, M_.ravel()]
    lab = np.r_[np.ones(len(ip)), np.zeros(M_.size)]
    Xm = X.iloc[on_i].iloc[rows]
    later = dict(move_size_first_300ms_bps=win_size.loc[Xm.key].to_numpy(float), window_ticks=X.groupby('key').size().loc[Xm.key].to_numpy(float))
    Xv = Xm[feats].to_numpy(float)
    one, A = auc_matrix(Xv, lab)
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(N_NULL if not reh else 20):                         # the chased label shuffled within each matched set (1 + K rows)
        o_ = rng.permuted(np.c_[np.ones(len(ip)), np.zeros((len(ip), K_MATCH))], axis=1)
        perm = np.r_[o_[:, 0], o_[:, 1:].ravel()]
        null.append(float(np.nanmax(np.abs(one(perm) - .5))))
    p95 = float(np.quantile(null, .95))
    rk = sorted(((feats[j], float(A[j])) for j in range(len(feats)) if np.isfinite(A[j])), key=lambda x: -abs(x[1] - .5))
    mo1 = d * (X.m1s.to_numpy(float) - md)
    PA = X[[f'pA_{k}' for k in range(5)]].to_numpy(float); SA = X[[f'sA_{k}' for k in range(5)]].to_numpy(float)
    PB = X[[f'pB_{k}' for k in range(5)]].to_numpy(float); SB = X[[f'sB_{k}' for k in range(5)]].to_numpy(float)
    okB = np.isfinite(X.dtB.to_numpy(float))
    BK = {'A': (PA, SA), 'AB': TK.minbook(PA, SA, PB, SB)}
    MK1 = X.m1s.to_numpy(float)

    def value(idx, b, s):
        P_, S_ = BK[b]
        ii = idx[okB[idx]] if b == 'AB' else idx
        fl, co_, fe_ = TK.vwalk(P_[ii], S_[ii], d[ii], lim[ii] + d[ii] * s)
        v = (d[ii] * (MK1[ii] * fl - co_) - fe_) / Q
        return dict(actions=int(len(ii)), value_c=float(v.mean()) if len(ii) else None, fill=float(fl.sum() / (Q * len(ii))) if len(ii) else None)
    ri = on_i[rows]
    pos_i, neg_i = ri[lab == 1], ri[lab == 0]
    out['what_we_miss'] = dict(matched_sets=int(len(ip)), chased_moves=int(co.sum()), matched_per_set=K_MATCH, null_draws=len(null), null_p95=p95,
                               inputs_beyond_null=[dict(input=f_, auc=a_) for f_, a_ in rk if abs(a_ - .5) > p95], top_inputs=[dict(input=f_, auc=a_) for f_, a_ in rk[:15]],
                               edge_at_onset_mean_c=dict(chased=float(e1[pos_i].mean()), matched=float(e1[neg_i].mean())),
                               realised_predict_move_1s_c=dict(chased=float(mo1[pos_i].mean()), matched=float(mo1[neg_i].mean())),
                               later_size_auc={k_: auc1(v_, lab) for k_, v_ in later.items()},
                               later_size_mean=dict(chased={k_: float(np.nanmean(v_[lab == 1])) for k_, v_ in later.items()},
                                                    matched={k_: float(np.nanmean(v_[lab == 0])) for k_, v_ in later.items()}),
                               value_taking_at_onset={grp: {f'{b}_s{int(s)}': value(ix, b, s) for b in ('A', 'AB') for s in (0., 3., 10.)} for grp, ix in (('chased', pos_i), ('matched', neg_i))})
    # ---- (5) our model's top-10 % actions on moves it chased or not
    ch10 = chased[t10]
    out['our_top10_by_chase'] = {grp: dict(actions=int(m_.sum()), **{f'{b}_s{int(s)}': value(t10[m_], b, s) for b in ('A', 'AB') for s in (0., 3., 10.)})
                                 for grp, m_ in (('it_chased', ch10), ('it_did_not', ~ch10))}
    TR.dump(fn, out)
    print(json.dumps({k: out[k] for k in ('decisions', 'coverage', 'our_top10_by_chase')}, indent=1, default=str)[:4000])
    print(json.dumps({k: v for k, v in out['what_we_miss'].items() if k not in ('top_inputs',)}, indent=1, default=str)[:3000])


def auc1(score, lab):
    ok = np.isfinite(score)
    s, y = score[ok], lab[ok].astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.) / (n1 * n0))


# ---------------------------------------------------------------------------------------------------------------- selftest
def selftest():
    # distributions: shares add to 1, open edges hold the tails
    dd = dist(np.array([-5., 0., .2, .3, 3., 100., np.nan]), [-INF, .1, .25, .5, 1, 2, 4, INF])
    assert dd['n'] == 6 and abs(sum(b['share'] for b in dd['buckets']) - 1.) < 1e-12, dd
    assert [round(b['share'] * 6) for b in dd['buckets']] == [2, 1, 1, 0, 0, 1, 1], dd
    # matching: nearest edge, same day and direction, within the window, never a positive
    day = np.array(['a'] * 6 + ['b']); d = np.array([1., 1., 1., 1., -1., 1., 1.]); t = np.array([0., 10., 20., 4e6, 30., 40., 50.])
    e = np.array([1., 1.1, 3., 1.05, 1., 0.8, 1.])                          # nearest: 1 (0.1), then 5 (0.2); 3 is nearer but out of the window
    pos = np.array([True, False, False, False, False, False, False])
    ip, Mx = match_sets(day, d, t, e, pos, ~pos, k=2)
    assert ip.tolist() == [0] and Mx.tolist() == [[1, 5]], Mx                # 3 is out of the window, 4 the other direction, 6 another day
    # AUC: a perfect separator 1, its mirror 0, NaN dropped; the shuffle keeps one positive per set
    Xv = np.array([[1., 5.], [2., np.nan], [3., 1.], [4., 2.]]); lab = np.array([0., 0., 1., 1.])
    one, A = auc_matrix(Xv, lab)
    assert A[0] == 1. and A[1] == 0., A
    o_ = np.random.default_rng(0).permuted(np.c_[np.ones(3), np.zeros((3, 2))], axis=1)
    assert (o_.sum(1) == 1).all()
    # the group-3 rule: a tick in the 300 ms before the arrival
    M = np.array([100., 250.]); a = np.array([300., 549., 551.])
    j = np.searchsorted(M, a, 'right') - 1
    assert ((a - M[j]) < 300.).tolist() == [True, True, False]
    try:
        labels('20261022', '/nonexistent')
        raise AssertionError('an undeclared day accepted')
    except ValueError as e_:
        assert 'not a declared day' in str(e_), e_
    print('group3_profile selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    if cmd == 'labels':
        labels(a_[0], a_[1] if len(a_) > 1 else None)
    elif cmd == 'read':
        read(a_[0] if a_ else None)
    else:
        dict(selftest=selftest)[cmd]()
