"""Public score v3: the direct factors of our findings instead of raw amounts (USER 2026-11-26: "if refill rate is critical, we should use
refill rate instead of raw amounts, and if multiple level's dynamics make more sense, we should use dynamics instead of raw snapshots, so we
should use the direct factors, not the raw amoutns"). Development evidence on the four consumed days (the top taker traded on three).
Points, labels and evaluation: public_score.py (its per-point table is read). 'Normal' = the same quantity's rate over (t - 30 s, t - 1 s];
a rate factor = log((rate now + e) / (rate normal + e)), e = 1e-5 BTC / ms (1e-4 events / ms for counts): 0 = normal, log 0.3 = 0.3x.
Families (declared before the run):
state: best-level imbalance its way, log far / near size, spread, depth imbalance over 20 levels, log far cushion behind the best, time left.
best_flow (fire_events' events, Binance futures E clock): far-side new orders (refill), far cancels, far trades (consumption), near-side new
orders (posts), near cancels, near trades (counter-trades) as rate factors over 100 and 300 ms; far / near replacement (new orders / (cancels
+ trades)) over 300 ms, now and its change from normal; the counter-trade share (near trades / all trades) over 300 ms, now and its change
from normal; book activity (all events) as a rate factor; imbalance change over 100 / 300 ms; log far / near size change over 300 ms.
deep (depth20, every snapshot against the one before, levels 2-20, a price matched by tick: size up = added, down = removed; a price gone
inside the new snapshot's range = removed, a new price inside the old range = added; the best level of either snapshot excluded):
far / near added and removed as rate factors over 1 s; far-side levels 2-5 removed and near-side levels 2-5 added as rate factors over
300 ms; far replacement (added / removed) over 1 s against normal; depth-imbalance change over 300 ms and 1 s.
venues (Binance spot, Bybit perp, OKX swap, Coinbase, Deribit; each exchange's clock): the price gap its way (venue last trade vs the
Binance futures mid, bps) change over 300 ms and 5 s; volume its way and against as rate factors over 300 ms.
predict: the fill-price move its way over 5 s; other takers' net share its way over 5 s; fills in the last 1 s as a rate factor.
Models: logistic and gradient-boosted trees on the v3 features (public_score's settings), leave-one-day-out, trained on random + nearby
moments, never on fires; compared on the same points with public_score2's held-out v2 scores. The sauce: the trees' family permutation
importance (held-out, seeded), the logistic refit on all days (standardized coefficients), profiles in natural units (x normal for rates)
at the held-out top-1% random moments, all random moments and the top taker's fires.
Outputs (write-once): public_score3.json; OUT/public_score3_points.parquet.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/public_score3.py \
       [<rehearsal dir> <day> <day> ...]
"""
import json
import re
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import fire_events as FE  # noqa: E402
import public_score as PS  # noqa: E402
import public_score2 as P2  # noqa: E402
import cross_venue as CV  # noqa: E402
import before_long as BLm  # noqa: E402

DC, C, TR, WT = FE.DC, FE.C, FE.TR, FE.WT
EPS, EPSN, BASE0, BASE1, BIG = 1e-5, 1e-4, 30000., 1000., 1 << 22
VEN3 = ('binance_spot', 'bybit_perp', 'okx_swap', 'coinbase', 'deribit')
KINDS3 = (('opp', 'add', 'far_new'), ('opp', 'cancel', 'far_cxl'), ('opp', 'trade', 'far_trade'), ('near', 'add', 'near_new'), ('near', 'cancel', 'near_cxl'),
          ('near', 'trade', 'near_trade'))
FAM3 = dict(
    state=['b_imb1', 'b_lfar1', 'b_lnear1', 'b_spread', 'd_imb20', 'd_lfarcush', 'p_tau'],
    best_flow=[f'x_{n}_r{w}' for _, _, n in KINDS3 for w in (100, 300)] + ['x_far_replace', 'x_far_replace_dev', 'x_near_replace', 'x_near_replace_dev',
                                                                         'x_counter_share', 'x_counter_share_dev', 'x_activity_r300', 'x_imb_d100', 'x_imb_d300',
                                                                         'x_far_r300', 'x_near_r300'],
    deep=['z_far_add_r1s', 'z_far_rem_r1s', 'z_near_add_r1s', 'z_near_rem_r1s', 'z_far_rem5_r300', 'z_near_add5_r300', 'z_far_replace_dev', 'z_imb_d300', 'z_imb_d1s'],
    venues=[f'u_{v}_{k}' for v in VEN3 for k in ('gap_d300', 'gap_d5s', 'its_r300', 'ag_r300')],
    predict=['p_pf5', 'q_net_share5', 'q_fill_r1s'])
V3 = [c for f in FAM3.values() for c in f]
NEW = [c for c in V3 if c not in ('b_imb1', 'b_lfar1', 'b_lnear1', 'b_spread', 'd_imb20', 'd_lfarcush', 'p_tau', 'p_pf5')]
PROF3 = dict(far_refill_x_normal_300=('x_far_new_r300', 'xn'), far_cancel_x_normal_300=('x_far_cxl_r300', 'xn'), near_posts_x_normal_300=('x_near_new_r300', 'xn'),
             counter_trades_x_normal_300=('x_near_trade_r300', 'xn'), trades_its_way_x_normal_300=('x_far_trade_r300', 'xn'), activity_x_normal_300=('x_activity_r300', 'xnc'),
             far_replacement_300=('x_far_replace', 'exp'), far_replacement_vs_normal=('x_far_replace_dev', 'exp'), counter_trade_share_300=('x_counter_share', None),
             deep_far_removed_x_normal_1s=('z_far_rem_r1s', 'xn'), deep_far_added_x_normal_1s=('z_far_add_r1s', 'xn'), deep_near_added_x_normal_1s=('z_near_add_r1s', 'xn'),
             deep_far_2_5_removed_x_normal_300=('z_far_rem5_r300', 'xn'), imbalance_change_300=('x_imb_d300', None), depth_imbalance_change_1s=('z_imb_d1s', None),
             imbalance=('b_imb1', None), far_size_btc=('b_lfar1', 'exp'))


def lr(now, w, base, wb, eps):
    return np.log((now / w + eps) / (base / wb + eps))


def wsum(E, cA, a, b):
    return cA[np.searchsorted(E, b, 'right')] - cA[np.searchsorted(E, a, 'right')]


def fast_parse(strs):
    n = len(strs)
    x = np.array(re.split('[@|]', '|'.join(strs)), dtype=float)
    if x.size != n * 40:
        P, Q = PS.DD.parse(strs)
        return P, Q
    X = x.reshape(n, 40)
    return X[:, ::2], X[:, 1::2]


def deep_hour(E, P, Q, side):
    """Per snapshot (against the previous one): added / removed size at levels 2-5 and 6-20 of one side."""
    n = len(E)
    out = {k: np.zeros(n) for k in ('add5', 'add20', 'rem5', 'rem20')}
    if n < 2:
        return out
    T = np.rint(P * 10.).astype(np.int64)
    base = T[np.isfinite(P)].min() - 10 if np.isfinite(P).any() else 0
    ok = np.isfinite(P) & np.isfinite(Q)
    k = np.arange(1, n)
    lvl = np.arange(20)
    prev_T, prev_Q, prev_ok = T[:-1, 1:], Q[:-1, 1:], ok[:-1, 1:]
    cur_T, cur_Q, cur_ok = T[1:, 1:], Q[1:, 1:], ok[1:, 1:]
    kk = np.repeat(k, 19).reshape(n - 1, 19)
    lv = np.broadcast_to(lvl[1:], (n - 1, 19)) + 1                                   # level number 2..20
    pk = (kk * BIG + (prev_T - base))[prev_ok]; pq = prev_Q[prev_ok]; pl = lv[prev_ok]; pkk = kk[prev_ok]
    ck = (kk * BIG + (cur_T - base))[cur_ok]; cq = cur_Q[cur_ok]; cl = lv[cur_ok]; ckk = kk[cur_ok]
    po = np.argsort(pk, kind='stable'); pk, pq, pl, pkk = pk[po], pq[po], pl[po], pkk[po]
    co = np.argsort(ck, kind='stable'); ck, cq, cl, ckk = ck[co], cq[co], cl[co], ckk[co]
    i = np.searchsorted(pk, ck)
    hit = (i < len(pk)) & (pk[np.minimum(i, len(pk) - 1)] == ck)
    dq = np.where(hit, cq - pq[np.minimum(i, len(pk) - 1)], 0.)
    best_prev, best_cur = T[:-1, 0], T[1:, 0]
    last_prev = np.where(ok[:-1, -1], T[:-1, -1], T[:-1, 0])
    last_cur = np.where(ok[1:, -1], T[1:, -1], T[1:, 0])
    ctick = ck - ckk * BIG + base
    ptick = pk - pkk * BIG + base
    inside_prev = (ctick <= last_prev[ckk - 1]) if side == 'ask' else (ctick >= last_prev[ckk - 1])
    new_ok = ~hit & inside_prev & (ctick != best_prev[ckk - 1])
    j = np.searchsorted(ck, pk)
    gone = ~((j < len(ck)) & (ck[np.minimum(j, len(ck) - 1)] == pk))
    inside_cur = (ptick <= last_cur[pkk - 1]) if side == 'ask' else (ptick >= last_cur[pkk - 1])
    rem_ok = gone & inside_cur & (ptick != best_cur[pkk - 1])
    add_v = np.where(hit, np.maximum(dq, 0.), 0.) + np.where(new_ok, cq, 0.)
    remc_v = np.where(hit, np.maximum(-dq, 0.), 0.)
    for band, sel_c, sel_p in (('5', cl <= 5, pl <= 5), ('20', cl > 5, pl > 5)):
        out[f'add{band}'] += np.bincount(ckk[sel_c], weights=add_v[sel_c], minlength=n)
        out[f'rem{band}'] += np.bincount(ckk[sel_c], weights=remc_v[sel_c], minlength=n) + np.bincount(pkk[sel_p & rem_ok], weights=pq[sel_p & rem_ok], minlength=n)
    return out


def day_v3(day, Zd):
    day0 = float(WT.OD_day_ms(day))
    t, d = Zd.t.to_numpy(float), Zd.d.to_numpy(float)
    n = len(t)
    F = {c: np.full(n, np.nan) for c in NEW}
    ev, st, traj, _ = FE.day_data(day, Zd[['t', 'd']])
    C.require((st == 'ok').all(), f'{day}: a point without a book')
    hrs_b = np.array([(DC.RAW / f'bookTicker/BTCUSDT/{day}_{h:02d}.csv.gz').exists() for h in range(24)])
    hrs_d = np.array([(DC.RAW / f'depth20/BTCUSDT/{day}_{h:02d}.csv.gz').exists() for h in range(24)])
    hb = ((t - BASE0 - day0) // 3600000).astype(int)
    base_ok = (t - BASE0 >= day0) & hrs_b[np.clip(hb, 0, 23)] & hrs_d[np.clip(hb, 0, 23)]
    acnt = np.zeros(n); acnt_b = np.zeros(n)
    for dv in (1., -1.):
        m = d == dv
        if not m.any():
            continue
        tt = t[m]
        V = {}
        for r, k, nm in KINDS3:
            E, cA = FE.role_events(ev, r, k, dv)
            for w in (100., 300.):
                F[f'x_{nm}_r{int(w)}'][m] = lr(wsum(E, cA, tt - w, tt), w, wsum(E, cA, tt - BASE0, tt - BASE1), BASE0 - BASE1, EPS)
            V[nm] = (wsum(E, cA, tt - 300., tt), wsum(E, cA, tt - BASE0, tt - BASE1))
        for r in ('opp', 'near'):
            for k in ('trade', 'cancel', 'add', 'gone', 'improve'):
                E, _ = FE.role_events(ev, r, k, dv)
                acnt[m] += np.searchsorted(E, tt, 'right') - np.searchsorted(E, tt - 300., 'right')
                acnt_b[m] += np.searchsorted(E, tt - BASE1, 'right') - np.searchsorted(E, tt - BASE0, 'right')
        for side, a, b, c_ in (('far', 'far_new', 'far_cxl', 'far_trade'), ('near', 'near_new', 'near_cxl', 'near_trade')):
            now = np.log((V[a][0] / 300. + EPS) / ((V[b][0] + V[c_][0]) / 300. + EPS))
            bas = np.log((V[a][1] / (BASE0 - BASE1) + EPS) / ((V[b][1] + V[c_][1]) / (BASE0 - BASE1) + EPS))
            F[f'x_{side}_replace'][m], F[f'x_{side}_replace_dev'][m] = now, now - bas
        tn, tb = V['near_trade'][0] + V['far_trade'][0], V['near_trade'][1] + V['far_trade'][1]
        sh_now = np.where(tn > 0, V['near_trade'][0] / np.maximum(tn, 1e-12), np.nan)
        sh_b = np.where(tb > 0, V['near_trade'][1] / np.maximum(tb, 1e-12), np.nan)
        F['x_counter_share'][m], F['x_counter_share_dev'][m] = sh_now, sh_now - sh_b
    F['x_activity_r300'] = np.log((acnt / 300. + EPSN) / (acnt_b / (BASE0 - BASE1) + EPSN))
    F['x_imb_d100'], F['x_imb_d300'] = traj['imb_0'] - traj['imb_100'], traj['imb_0'] - traj['imb_300']
    F['x_far_r300'], F['x_near_r300'] = np.log(traj['opp_0'] / traj['opp_300']), np.log(traj['near_0'] / traj['near_300'])
    del ev
    # deep dynamics over the day's snapshots, and the Binance mid for the venue gaps
    Es, agg, imbs = [], {s: {k: [] for k in ('add5', 'add20', 'rem5', 'rem20')} for s in ('bid', 'ask')}, []
    mE, mM = [], []
    for h, bk in DC.book_hours(day):
        Eb, bid, bq, ask, aq = FE.collapse(*bk)
        mid = (bid + ask) / 2.
        kk = np.r_[True, np.abs(np.diff(mid)) > 1e-9]
        mE.append(Eb[kk]); mM.append(mid[kk])
        fd = DC.RAW / f'depth20/BTCUSDT/{day}_{h:02d}.csv.gz'
        if not fd.exists():
            continue
        D = pd.read_csv(fd, header=None, usecols=[1, 4, 5], names=['E', 'b', 'a']).sort_values('E', kind='stable')
        E = D.E.to_numpy(float)
        bP, bQ = fast_parse(D.b.to_numpy())
        aP, aQ = fast_parse(D.a.to_numpy())
        for s, (P_, Q_) in (('bid', (bP, bQ)), ('ask', (aP, aQ))):
            o = deep_hour(E, P_, Q_, s)
            for k in o:
                agg[s][k].append(o[k])
        bs, as_ = np.nansum(bQ, 1), np.nansum(aQ, 1)
        imbs.append((bs - as_) / np.maximum(bs + as_, 1e-12))
        Es.append(E)
        C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
    E = np.concatenate(Es)
    o = np.argsort(E, kind='stable')
    E = E[o]
    IMB = np.concatenate(imbs)[o]
    CA = {s: {k: np.r_[0., np.cumsum(np.concatenate(v)[o])] for k, v in agg[s].items()} for s in agg}
    j0 = np.searchsorted(E, t, 'right') - 1
    fresh = (j0 >= 0) & (t - E[np.maximum(j0, 0)] <= PS.DD.STALE)
    up = d > 0
    for role, fs in (('far', np.where(up, 'ask', 'bid')), ('near', np.where(up, 'bid', 'ask'))):
        for kind in ('add', 'rem'):
            now = np.zeros(n); bas = np.zeros(n); now5 = np.zeros(n); bas5 = np.zeros(n)
            for s in ('bid', 'ask'):
                m = fs == s
                c5, c20 = CA[s][f'{kind}5'], CA[s][f'{kind}20']
                now[m] = wsum(E, c5, t[m] - 1000., t[m]) + wsum(E, c20, t[m] - 1000., t[m])
                bas[m] = wsum(E, c5, t[m] - BASE0, t[m] - BASE1) + wsum(E, c20, t[m] - BASE0, t[m] - BASE1)
                now5[m] = wsum(E, c5, t[m] - 300., t[m])
                bas5[m] = wsum(E, c5, t[m] - BASE0, t[m] - BASE1)
            F[f'z_{role}_{kind}_r1s'] = lr(now, 1000., bas, BASE0 - BASE1, EPS)
            if (role, kind) == ('far', 'rem'):
                F['z_far_rem5_r300'] = lr(now5, 300., bas5, BASE0 - BASE1, EPS)
                far_rem_now, far_rem_b = now, bas
            if (role, kind) == ('near', 'add'):
                F['z_near_add5_r300'] = lr(now5, 300., bas5, BASE0 - BASE1, EPS)
            if (role, kind) == ('far', 'add'):
                far_add_now, far_add_b = now, bas
    F['z_far_replace_dev'] = (np.log((far_add_now / 1000. + EPS) / (far_rem_now / 1000. + EPS)) - np.log((far_add_b / (BASE0 - BASE1) + EPS) / (far_rem_b / (BASE0 - BASE1) + EPS)))
    jm3, jm1 = np.searchsorted(E, t - 300., 'right') - 1, np.searchsorted(E, t - 1000., 'right') - 1
    i0 = IMB[np.maximum(j0, 0)]
    F['z_imb_d300'] = np.where(jm3 >= 0, d * (i0 - IMB[np.maximum(jm3, 0)]), np.nan)
    F['z_imb_d1s'] = np.where(jm1 >= 0, d * (i0 - IMB[np.maximum(jm1, 0)]), np.nan)
    for c in [c for c in NEW if c.startswith('z_')]:
        F[c] = np.where(fresh & base_ok, F[c], np.nan)
    for c in NEW:
        if c.startswith('x_') and c.endswith(('_r100', '_r300', '_dev')):                 # factors against normal need the 30 s baseline
            F[c] = np.where(base_ok, F[c], np.nan)
    ME, MM = np.concatenate(mE), np.concatenate(mM)
    o = np.argsort(ME, kind='stable'); ME, MM = ME[o], MM[o]
    keep = np.r_[True, np.diff(ME) > 0]; ME, MM = ME[keep], MM[keep]
    mid_at = lambda x: BLm.mid_at(ME, MM, hrs_b, day0, x)
    for v in VEN3:
        tv, pv, qv, sv, cov, _ = CV.load(v, day)

        def gap(x):
            j = np.searchsorted(tv, x, 'right') - 1
            okv = (j >= 0) & (x >= cov[0]) & (x < cov[1]) & (x - tv[np.maximum(j, 0)] <= CV.STALE_MS)
            mm = mid_at(x)
            return np.where(okv, d * 1e4 * (pv[np.maximum(j, 0)] - mm) / mm, np.nan)
        g0 = gap(t)
        F[f'u_{v}_gap_d300'], F[f'u_{v}_gap_d5s'] = g0 - gap(t - 300.), g0 - gap(t - 5000.)
        for sg, tag in ((1, 'its'), (-1, 'ag')):
            res = np.full(n, np.nan)
            for dv in (1., -1.):
                m = d == dv
                side = sg * int(dv)
                te = tv[sv == side]
                cq = np.r_[0., np.cumsum(qv[sv == side])]
                tt = t[m]
                inside = (tt - BASE0 >= cov[0]) & (tt < cov[1])
                res[m] = np.where(inside, lr(wsum(te, cq, tt - 300., tt), 300., wsum(te, cq, tt - BASE0, tt - BASE1), BASE0 - BASE1, EPS), np.nan)
            F[f'u_{v}_{tag}_r300'] = res
        del tv, pv, qv, sv
    reg = WT.MM.registry(day)
    by_ws = dict(zip((reg.ws.to_numpy(float) * 1000.).round().astype(np.int64), reg.mid.astype(int)))
    wst = (np.floor(t / 300000.) * 300000.).astype(np.int64)
    mids = np.array([by_ws.get(int(x), -1) for x in wst])
    L = pd.read_parquet(WT.OUT / f'winners_legs_{day}.parquet', columns=['mid', 'side', 'shares', 't_ms2', 'how2', 'taker', 'ambiguous'])
    L = L[L.how2.isin([1, 3]) & L.t_ms2.notna() & ~L.ambiguous & (L.taker != BLm.TOP)].sort_values(['mid', 't_ms2'], kind='stable')
    G = {int(k): (g.t_ms2.to_numpy(float), np.where(g.side.to_numpy() == 1, 1., -1.), g.shares.to_numpy(float)) for k, g in L.groupby('mid')}
    for i in range(n):
        g = G.get(int(mids[i]))
        if g is None:
            continue
        tl, ld, sh = g
        b = np.searchsorted(tl, t[i], 'right')
        a5, a1, ab = np.searchsorted(tl, t[i] - 5000., 'right'), np.searchsorted(tl, t[i] - 1000., 'right'), np.searchsorted(tl, t[i] - BASE0, 'right')
        tot = sh[a5:b].sum()
        F['q_net_share5'][i] = float((sh[a5:b] * ld[a5:b]).sum() * d[i] / tot) if tot > 0 else np.nan
        if t[i] - float(wst[i]) < 2. * BASE1:
            continue                                                                    # the window just opened: no fill-rate baseline
        span = min(BASE0 - BASE1, max(t[i] - BASE1 - max(t[i] - BASE0, float(wst[i])), 1.))
        F['q_fill_r1s'][i] = np.log(((b - a1) / 1000. + EPSN) / ((a1 - max(ab, np.searchsorted(tl, float(wst[i]), 'left'))) / span + EPSN))
    return F


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    od = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    fn, fp = (od or H) / 'public_score3.json', (od or WT.OUT) / 'public_score3_points.parquet'
    C.require(not fn.exists() and not fp.exists(), 'an output exists (write-once)')
    days = tuple(sys.argv[2:]) if len(sys.argv) > 2 else WT.DAYS
    C.require(len(days) >= 2 and all(x in WT.DAYS for x in days), 'needs >= 2 declared days')
    keep = ['day', 't', 'd', 'kind', 'y', 'next_ms', 'b_imb1', 'b_lfar1', 'b_lnear1', 'b_spread', 'd_imb20', 'd_lfarcush', 'p_tau', 'p_pf5']
    Z = pd.read_parquet(WT.OUT / 'public_score_points.parquet', columns=keep)
    Z2 = pd.read_parquet(WT.OUT / 'public_score2_points.parquet', columns=['day', 't', 'd', 'kind', 's_logit_v2', 's_gbm_v2'])
    C.require(len(Z) == len(Z2) and (Z.day.to_numpy() == Z2.day.to_numpy()).all() and (Z.t.to_numpy() == Z2.t.to_numpy()).all() and (Z.d.to_numpy() == Z2.d.to_numpy()).all(),
              'the v2 points differ from public_score')
    Z['s_logit_v2'], Z['s_gbm_v2'] = Z2.s_logit_v2.to_numpy(), Z2.s_gbm_v2.to_numpy()
    del Z2
    Z = Z[Z.day.isin(days)].reset_index(drop=True)
    parts = []
    for day in days:
        Zd = Z[Z.day == day].reset_index(drop=True)
        for k, v in day_v3(day, Zd).items():
            Zd[k] = v
        parts.append(Zd)
        print(day, len(Zd), {c: round(float(Zd[c].notna().mean()), 3) for c in ('x_far_new_r300', 'z_far_rem_r1s', 'u_okx_swap_gap_d300', 'q_fill_r1s')}, flush=True)
    Z = pd.concat(parts, ignore_index=True)
    models = dict(logit_v3=('logit', V3), gbm_v3=('gbm', V3))
    for nm in models:
        Z[f's_{nm}'] = np.nan
    rng = np.random.default_rng(PS.SEED + 3)
    perm = {f: dict(auc=[], top1=[]) for f in FAM3}
    base = dict(auc=[], top1=[])
    for day in days:
        tr = ((Z.day != day) & (Z.kind != 'fire')).to_numpy()
        te = (Z.day == day).to_numpy()
        ytr = Z.y.to_numpy()[tr]
        for nm, (kind, cols) in models.items():
            s, mdl = P2.fit_predict(kind, Z.loc[tr, cols].to_numpy(float), ytr, Z.loc[te, cols].to_numpy(float))
            Z.loc[te, f's_{nm}'] = s
            if nm == 'gbm_v3':
                Xte = Z.loc[te, cols].to_numpy(float)
                rnd = Z.kind.to_numpy()[te] == 'random'
                yte, dte, s0 = Z.y.to_numpy()[te][rnd], Z.day.to_numpy()[te][rnd], s[rnd]
                base['auc'].append(P2.roc_auc_score(yte, s0)); base['top1'].append(P2.top_prec(s0, yte, dte))
                for f, fc in FAM3.items():
                    X2 = Xte[rnd].copy()
                    idx = [cols.index(c) for c in fc]
                    X2[:, idx] = X2[rng.permutation(len(X2))][:, idx]
                    s2 = mdl.predict_proba(X2)[:, 1]
                    perm[f]['auc'].append(P2.roc_auc_score(yte, s2)); perm[f]['top1'].append(P2.top_prec(s2, yte, dte))
        print('fold', day, flush=True)
    cover = {c: float(Z[c].notna().mean()) for c in V3}
    out = dict(asof='2026-11-26', days=list(days), n=dict(Z.kind.value_counts().astype(int)), label='move its way within 300 ms', families=FAM3, feature_coverage=cover,
               models={nm: PS.evaluate(Z, f's_{nm}') for nm in ('logit_v2', 'gbm_v2', 'logit_v3', 'gbm_v3')})
    out['permutation_gbm_v3'] = dict(base_auc_mean=float(np.mean(base['auc'])), base_top1_mean=float(np.mean(base['top1'])),
                                     families={f: dict(auc_drop=float(np.mean(base['auc']) - np.mean(v['auc'])), top1_drop=float(np.mean(base['top1']) - np.mean(v['top1'])))
                                               for f, v in perm.items()})
    tr = (Z.kind != 'fire').to_numpy()
    Xa = Z.loc[tr, V3].to_numpy(float)
    A, _ = PS.design(Xa, Xa[:1])
    miss = np.isnan(Xa).any(axis=0)
    names = V3 + [f'{c}__missing' for c, mm in zip(V3, miss) if mm]
    lrm = LogisticRegression(C=1., max_iter=3000).fit(A, Z.y.to_numpy()[tr])
    out['coefficients_logit_v3_all_days'] = [dict(feature=k, coef=float(v)) for k, v in sorted(zip(names, lrm.coef_[0]), key=lambda kv: -abs(kv[1]))]
    R = Z[Z.kind == 'random']
    top = np.zeros(len(R), bool)
    for dd, g in R.groupby('day'):
        thr = np.quantile(g.s_logit_v3, .99)
        top[(R.day == dd).to_numpy() & (R.s_logit_v3 >= thr).to_numpy()] = True
    Fz = Z[Z.kind == 'fire']

    def nat(G, c, how):
        x = G[c].to_numpy(float)
        return np.exp(x) if how in ('exp', 'xn', 'xnc') else x
    out['profiles'] = {nm: dict(top1_random=float(np.nanmedian(nat(R[top], c, how))), all_random=float(np.nanmedian(nat(R, c, how))),
                                fires=float(np.nanmedian(nat(Fz, c, how))) if len(Fz) else None) for nm, (c, how) in PROF3.items()}
    out['profiles_note'] = 'x-normal rates are exp(log ratio) with the e floor: 1 = normal; when both now and normal are 0 the ratio is 1'
    out['profiles_n'] = dict(top1_random=int(top.sum()), all_random=int(len(R)), fires=int(len(Fz)))
    Z.to_parquet(fp, index=False)
    TR.dump(fn, out)
    for nm, o in out['models'].items():
        t1, t3 = o['tops']['0.01'], o['tops']['0.001']
        print(f"{nm:10s} auc {o['auc_random']:.3f} / {o['auc_nearby']:.3f} | top1% {t1['random_precision']:.3f} top0.1% {t3['random_precision']:.3f} | fires in top1% {t1['fires_share_above']:.3f} | fires mean {o['fires_mean_score']:.3f}")
    print(json.dumps(out['permutation_gbm_v3'], indent=0))
    print(json.dumps(out['coefficients_logit_v3_all_days'][:15], indent=0))
    print(json.dumps(out['profiles'], indent=0))


if __name__ == '__main__':
    main()
