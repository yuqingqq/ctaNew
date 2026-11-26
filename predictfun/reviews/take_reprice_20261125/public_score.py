"""One combined public score for 'the Binance futures mid moves its way within 300 ms' and how it compares with the top taker (USER
2026-11-26: "yes we can build that score to check"). Development evidence on the four consumed days (the top taker traded on three); the
features were chosen from descriptive studies on these same days, so this is not a validation; no strategy change, no orders.
Points (decision moments = no move its way in the previous 300 ms): random moments (state_value_points: 30,000 per day, both directions,
uniform over hours with a book file), nearby moments and the top taker's fires (fire_breakdown_points). Label = a move its way within 300 ms
(the first best-price change in its direction; reconciled with the points tables).
Features (public, causal, all computed at t from data before t; Binance futures masked prints dropped), six families:
best level: level-1 imbalance, log far / near size, spread, log change of far / near size over 300 ms, size churn, book updates, a move
against in 300 ms; trades (Binance futures BTC): counts / BTC its way and against over 100 / 300 ms, time since the last trade each way;
depth (depth20 snapshots at or before t, t - 300 ms, t - 1 s): imbalance over 1-5 / 1-20 levels, far / near size behind the best, the $2
cushions, imbalance change over 300 ms / 1 s, far depth change over 300 ms / 1 s; venues (Binance spot, Bybit perp / spot, OKX swap / spot,
Coinbase, Kraken, Deribit, Binance futures ETH): volume its way / against over 100 ms, return over 100 ms; long windows: Binance mid return
over 1 s, 5 s up to -300 ms, 30 s; Predict (the market live at t, timed legs, the top taker's own fills excluded): fill-price move its way
over 5 s, other takers' net shares its way over 5 s, fills in the last 1 s, time left in the window.
Models (declared before the run): logistic regression (train-fold winsorized at 0.5 / 99.5%, standardized, median-imputed with missing flags,
L2 C = 1) on all features and on each family alone; gradient-boosted trees on all features (sklearn HistGradientBoosting, 200 iterations,
learning rate 0.05, 15 leaves, 200 per leaf, L2 1, seed). Trained on random + nearby moments (never on the fires), leave-one-day-out over the
four days; scored on the held-out day.
Readings (pooled over held-out days; G = 4 days: point estimates only): AUC on random and on nearby moments; precision (the label rate) in the
top 1 / 0.3 / 0.1% of each held-out day's random moments; at the same thresholds the share of nearby moments and of its fires above them and
their label rate; its fires' mean score against their label rate; for top-1% random moments followed by a move, the time to the move (share
>= 75 ms: what our path needs, TIMELINE).
Outputs (write-once): public_score.json; OUT/public_score_points.parquet (features, labels, held-out scores).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/public_score.py \
       [<rehearsal dir> <day> <day> ...]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import fire_events as FE  # noqa: E402
import quiet_before as QB  # noqa: E402
import quiet_local as QL  # noqa: E402
import fire_breakdown as FBm  # noqa: E402
import cross_venue as CV  # noqa: E402
import depth_deep as DD  # noqa: E402
import before_long as BLm  # noqa: E402

DC, C, TR, WT = FE.DC, FE.C, FE.TR, FE.WT
W, HIT, LEAD_NEED, CAP_SINCE = 300., 300., 75., 5000.
TOPS = (.01, .003, .001)
VEN = CV.OTHER + ('eth_fut',)
SEED = WT.SEED + 7
FAMILIES = dict(
    best=['b_imb1', 'b_lfar1', 'b_lnear1', 'b_spread', 'b_rfar', 'b_rnear', 'b_lchf', 'b_lchn', 'b_lupd', 'b_against'],
    trades=['t_n100_its', 't_n100_ag', 't_v100_its', 't_v100_ag', 't_v300_its', 't_v300_ag', 't_since_its', 't_since_ag'],
    depth=['d_imb5', 'd_imb20', 'd_lfar25', 'd_lfar220', 'd_lnear220', 'd_lfarcush', 'd_lnearcush', 'd_d300imb', 'd_d1simb', 'd_rfar300', 'd_rfar1s'],
    venues=[f'v_{v}_{k}' for v in VEN for k in ('lv100', 'la100', 'r100')],
    long=['l_r1s', 'l_r5pre', 'l_r30s'],
    predict=['p_pf5', 'p_net5', 'p_n1', 'p_tau'])
ALL = [c for f in FAMILIES.values() for c in f]


def points(days):
    X = pd.read_parquet(WT.OUT / 'state_value_points.parquet', columns=['day', 'kind', 't', 'd', 'momentum', 'next_move'])
    R = X[(X.kind == 'random') & X.day.isin(days) & ~X.momentum].copy()
    R = pd.DataFrame(dict(day=R.day, t=R.t, d=R.d, kind='random', tab_prior=R.momentum, tab_next=R.next_move))
    B = pd.read_parquet(WT.OUT / 'fire_breakdown_points.parquet', columns=['day', 't', 'd', 'shift_ms', 'their_prior', 'next_ms'])
    B = B[B.day.isin(days) & ~B.their_prior]
    B = pd.DataFrame(dict(day=B.day, t=B.t, d=B.d, kind=np.where(B.shift_ms == 0., 'fire', 'nearby'), tab_prior=B.their_prior, tab_next=B.next_ms))
    return pd.concat([R, B], ignore_index=True)


def day_features(day, P):
    """All features and the label for the points of one day (P: t, d)."""
    day0 = float(WT.OD_day_ms(day))
    t, d = P.t.to_numpy(float), P.d.to_numpy(float)
    n = len(t)
    F = {c: np.full(n, np.nan) for c in ALL}
    F.update(their_prior=np.zeros(n, bool), next_ms=np.full(n, np.nan))
    tr = DC.read_trades(day).drop_duplicates('id')
    tr = tr[(tr.p > 0) & (tr.q > 0)].sort_values('E', kind='stable')
    tE, tq, tm = tr.E.to_numpy(float), tr.q.to_numpy(float), tr.m.to_numpy()
    for dv in (1., -1.):
        m = d == dv
        if not m.any():
            continue
        tt = t[m]
        for side, tag in ((DC.BUY_AGGR if dv > 0 else 1 - DC.BUY_AGGR, 'its'), (1 - DC.BUY_AGGR if dv > 0 else DC.BUY_AGGR, 'ag')):
            Es = tE[tm == side]
            cq = np.r_[0., np.cumsum(tq[tm == side])]
            b = np.searchsorted(Es, tt, 'right')
            for w in (100., 300.):
                a = np.searchsorted(Es, tt - w, 'right')
                F[f't_v{int(w)}_{tag}'][m] = np.log1p(cq[b] - cq[a])
                if w == 100.:
                    F[f't_n100_{tag}'][m] = np.log1p(b - a)
            last = np.where(b > 0, tt - Es[np.maximum(b - 1, 0)], CAP_SINCE)
            F[f't_since_{tag}'][m] = np.log1p(np.minimum(last, CAP_SINCE))
    hp = ((t - day0) // 3600000).astype(int)
    mE, mM = [], []
    for h, bk in DC.book_hours(day):
        st = FE.collapse(*bk)
        E, bid, bq, ask, aq = st
        mid = (bid + ask) / 2.
        k = np.r_[True, np.abs(np.diff(mid)) > 1e-9]
        mE.append(E[k]); mM.append(mid[k])
        sel = np.flatnonzero(hp == h)
        if not len(sel):
            continue
        tt, dd = t[sel], d[sel]
        hf = QB.hour_feats(st, tt, dd)
        F['b_imb1'][sel] = (hf['near1'] - hf['far1']) / (hf['near1'] + hf['far1'])
        F['b_lfar1'][sel], F['b_lnear1'][sel] = np.log(hf['far1']), np.log(hf['near1'])
        F['b_rfar'][sel], F['b_rnear'][sel] = np.log(hf['far1'] / hf['far0']), np.log(hf['near1'] / hf['near0'])
        F['b_lchf'][sel], F['b_lchn'][sel], F['b_lupd'][sel] = np.log1p(hf['churn_far']), np.log1p(hf['churn_near']), np.log1p(hf['updates'])
        their, against = QL.moves_flags(st, tt, dd)
        F['their_prior'][sel], F['b_against'][sel] = their, against.astype(float)
        F['next_ms'][sel] = FBm.next_move(st, tt, dd)
        j = WT.last_idx(E, tt)
        F['b_spread'][sel] = np.rint((ask[j] - bid[j]) / .1)
        fd = DC.RAW / f'depth20/BTCUSDT/{day}_{h:02d}.csv.gz'
        if fd.exists():
            D = pd.read_csv(fd, header=None, usecols=[1, 4, 5], names=['E', 'b', 'a']).sort_values('E', kind='stable')
            Ed = D.E.to_numpy(float)
            idx = {lag: np.searchsorted(Ed, tt - w, 'right') - 1 for lag, w in DD.LAGS}
            u = np.unique(np.concatenate([v[v >= 0] for v in idx.values()]))
            if len(u):
                bP, bQ = DD.parse(D.b.to_numpy()[u])
                aP, aQ = DD.parse(D.a.to_numpy()[u])
                up = dd > 0
                S = {}
                for lag, _ in DD.LAGS:
                    kk = np.searchsorted(u, np.maximum(idx[lag], 0))
                    Pf, Qf = np.where(up[:, None], aP[kk], bP[kk]), np.where(up[:, None], aQ[kk], bQ[kk])
                    Pn, Qn = np.where(up[:, None], bP[kk], aP[kk]), np.where(up[:, None], bQ[kk], aQ[kk])
                    S[lag] = {nm: np.where(idx[lag] >= 0, x, np.nan) for nm, x in DD.side_feats(Pf, Qf, Pn, Qn).items()}
                fresh = (idx['1'] >= 0) & (tt - Ed[np.maximum(idx['1'], 0)] <= DD.STALE)
                g = lambda x: np.where(fresh, x, np.nan)
                F['d_imb5'][sel], F['d_imb20'][sel] = g(S['1']['imb_1_5']), g(S['1']['imb_1_20'])
                F['d_lfar25'][sel], F['d_lfar220'][sel], F['d_lnear220'][sel] = g(np.log1p(S['1']['far_2_5'])), g(np.log1p(S['1']['far_2_20'])), g(np.log1p(S['1']['near_2_20']))
                F['d_lfarcush'][sel], F['d_lnearcush'][sel] = g(np.log1p(S['1']['far_cush'])), g(np.log1p(S['1']['near_cush']))
                F['d_d300imb'][sel], F['d_d1simb'][sel] = g(S['1']['imb_1_20'] - S['0']['imb_1_20']), g(S['1']['imb_1_20'] - S['m']['imb_1_20'])
                with np.errstate(divide='ignore', invalid='ignore'):
                    F['d_rfar300'][sel] = g(np.log((S['1']['far_2_20'] + 1e-3) / (S['0']['far_2_20'] + 1e-3)))
                    F['d_rfar1s'][sel] = g(np.log((S['1']['far_2_20'] + 1e-3) / (S['m']['far_2_20'] + 1e-3)))
        C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
    E, M = np.concatenate(mE), np.concatenate(mM)
    o = np.argsort(E, kind='stable')
    E, M = E[o], M[o]
    keep = np.r_[True, np.diff(E) > 0]
    E, M = E[keep], M[keep]
    hrs = np.array([(DC.RAW / f'bookTicker/BTCUSDT/{day}_{h:02d}.csv.gz').exists() for h in range(24)])
    m0 = BLm.mid_at(E, M, hrs, day0, t)
    for k, x0, x1 in (('l_r1s', t - 1000., None), ('l_r30s', t - 30000., None), ('l_r5pre', t - 5000., t - 300.)):
        a = BLm.mid_at(E, M, hrs, day0, x0)
        b = m0 if x1 is None else BLm.mid_at(E, M, hrs, day0, x1)
        F[k] = d * 1e4 * np.log(b / a)
    for v in VEN:
        tv, pv, qv, sv, cov, _ = CV.load(v, day)
        f = CV.feats(tv, pv, qv, sv, cov, t, d)
        F[f'v_{v}_lv100'], F[f'v_{v}_la100'], F[f'v_{v}_r100'] = np.log1p(f['v100']), np.log1p(f['a100']), f['r100']
        del tv, pv, qv, sv
    reg = WT.MM.registry(day)
    by_ws = dict(zip((reg.ws.to_numpy(float) * 1000.).round().astype(np.int64), reg.mid.astype(int)))
    wst = (np.floor(t / 300000.) * 300000.).astype(np.int64)
    mids = np.array([by_ws.get(int(x), -1) for x in wst])
    F['p_tau'] = np.where(mids >= 0, (wst + 300000. - t) / 1000., np.nan)
    L = pd.read_parquet(WT.OUT / f'winners_legs_{day}.parquet', columns=['mid', 'side', 'shares', 't_ms2', 'how2', 'taker', 'ambiguous', 'ptick'])
    L = L[L.how2.isin([1, 3]) & L.t_ms2.notna() & ~L.ambiguous & (L.taker != BLm.TOP)].sort_values(['mid', 't_ms2'], kind='stable')
    G = {int(k): (g.t_ms2.to_numpy(float), np.where(g.side.to_numpy() == 1, 1., -1.), g.shares.to_numpy(float), g.ptick.to_numpy(float)) for k, g in L.groupby('mid')}
    for i in range(n):
        g = G.get(int(mids[i]))
        if g is None:
            continue
        tl, ld, sh, pk = g
        b = np.searchsorted(tl, t[i], 'right')
        a5, a1 = np.searchsorted(tl, t[i] - 5000., 'right'), np.searchsorted(tl, t[i] - 1000., 'right')
        a0, b0 = a5, np.searchsorted(tl, t[i] - 4000., 'right')
        F['p_n1'][i] = np.log1p(b - a1)
        net = float((sh[a5:b] * ld[a5:b]).sum()) * d[i]
        F['p_net5'][i] = np.sign(net) * np.log1p(abs(net))
        if b > a1 and b0 > a0:
            F['p_pf5'][i] = d[i] * (np.median(pk[a1:b]) - np.median(pk[a0:b0]))
    return F


def design(Xtr, Xte):
    """Winsorize, impute, flag and standardize with train-fold statistics (logistic)."""
    C.require(np.isfinite(Xtr).any(axis=0).all(), 'a feature with no training value')
    lo, hi = np.nanquantile(Xtr, .005, axis=0), np.nanquantile(Xtr, .995, axis=0)
    med = np.nanmedian(Xtr, axis=0)
    miss = np.isnan(Xtr).any(axis=0)

    def f(X):
        Xc = np.clip(X, lo, hi)
        flags = np.isnan(Xc[:, miss]).astype(float)
        Xc = np.where(np.isnan(Xc), med, Xc)
        return Xc, flags
    A, Af = f(Xtr)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.
    B, Bf = f(Xte)
    return np.c_[(A - mu) / sd, Af], np.c_[(B - mu) / sd, Bf]


def evaluate(Z, col):
    """Pooled readings of one held-out score column."""
    o = dict(auc_random=None, auc_nearby=None, tops={})
    R, N, Fz = Z[Z.kind == 'random'], Z[Z.kind == 'nearby'], Z[Z.kind == 'fire']
    if R.y.nunique() == 2:
        o['auc_random'] = float(roc_auc_score(R.y, R[col]))
    if N.y.nunique() == 2:
        o['auc_nearby'] = float(roc_auc_score(N.y, N[col]))
    for q in TOPS:
        rows = dict(random_n=0, random_hits=0, nearby_n=0, nearby_above=0, nearby_hits=0, fires_n=0, fires_above=0, fires_hits=0, lead=[])
        for day, Rd in R.groupby('day'):
            thr = np.quantile(Rd[col], 1. - q)
            a = Rd[Rd[col] >= thr]
            rows['random_n'] += len(a); rows['random_hits'] += int(a.y.sum())
            rows['lead'] += list(a.next_ms[a.y.astype(bool)])
            Nd, Fd = N[N.day == day], Fz[Fz.day == day]
            rows['nearby_n'] += len(Nd); rows['nearby_above'] += int((Nd[col] >= thr).sum()); rows['nearby_hits'] += int(Nd.y[Nd[col] >= thr].sum())
            rows['fires_n'] += len(Fd); rows['fires_above'] += int((Fd[col] >= thr).sum()); rows['fires_hits'] += int(Fd.y[Fd[col] >= thr].sum())
        L = np.asarray(rows.pop('lead'), float)
        o['tops'][str(q)] = dict(rows, random_precision=rows['random_hits'] / max(rows['random_n'], 1),
                                 nearby_share_above=rows['nearby_above'] / max(rows['nearby_n'], 1), nearby_precision=rows['nearby_hits'] / max(rows['nearby_above'], 1) if rows['nearby_above'] else None,
                                 fires_share_above=rows['fires_above'] / max(rows['fires_n'], 1), fires_precision_above=rows['fires_hits'] / rows['fires_above'] if rows['fires_above'] else None,
                                 lead_ms=BLm.qs(L), lead_ge_need=float((L >= LEAD_NEED).mean()) if len(L) else None)
    o['fires_mean_score'], o['fires_label_rate'] = (float(Fz[col].mean()), float(Fz.y.mean())) if len(Fz) else (None, None)
    o['random_label_rate'] = float(R.y.mean())
    return o


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    od = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    fn, fp = (od or H) / 'public_score.json', (od or WT.OUT) / 'public_score_points.parquet'
    C.require(not fn.exists() and not fp.exists(), 'an output exists (write-once)')
    days = tuple(sys.argv[2:]) if len(sys.argv) > 2 else WT.DAYS
    C.require(len(days) >= 2 and all(x in WT.DAYS for x in days), 'needs >= 2 declared days')
    P = points(days)
    parts = []
    for day in days:
        Pd = P[P.day == day].reset_index(drop=True)
        F = day_features(day, Pd)
        for k, v in F.items():
            Pd[k] = v
        inside = Pd.tab_next <= HIT
        C.require(Pd.their_prior.eq(Pd.tab_prior).all(), f'{day}: the prior move differs from the points tables')
        C.require((np.abs(Pd.next_ms[inside] - Pd.tab_next[inside]) < 1e-6).all() and (Pd.next_ms[~inside] > HIT).all(), f'{day}: the next move differs from the points tables')
        parts.append(Pd)
        print(day, len(Pd), dict(Pd.kind.value_counts()), flush=True)
    Z = pd.concat(parts, ignore_index=True)
    Z['y'] = (Z.next_ms <= HIT).astype(int)
    cover = {c: float(Z[c].notna().mean()) for c in ALL}
    models = {'logit_all': ('logit', ALL), 'gbm_all': ('gbm', ALL), **{f'logit_{f}': ('logit', cols) for f, cols in FAMILIES.items()}}
    for nm in models:
        Z[f's_{nm}'] = np.nan
    for day in days:
        tr = ((Z.day != day) & (Z.kind != 'fire')).to_numpy()
        te = (Z.day == day).to_numpy()
        ytr = Z.y.to_numpy()[tr]
        for nm, (kind, cols) in models.items():
            Xtr, Xte = Z.loc[tr, cols].to_numpy(float), Z.loc[te, cols].to_numpy(float)
            if kind == 'logit':
                A, B = design(Xtr, Xte)
                mdl = LogisticRegression(C=1., max_iter=3000).fit(A, ytr)
                Z.loc[te, f's_{nm}'] = mdl.predict_proba(B)[:, 1]
            else:
                mdl = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1., random_state=SEED).fit(Xtr, ytr)
                Z.loc[te, f's_{nm}'] = mdl.predict_proba(Xte)[:, 1]
        print('fold', day, flush=True)
    out = dict(asof='2026-11-26', days=list(days), n=dict(Z.kind.value_counts().astype(int)), label='move its way within 300 ms', lead_need_ms=LEAD_NEED,
               feature_coverage=cover, families=FAMILIES, models={nm: evaluate(Z, f's_{nm}') for nm in models})
    Z.to_parquet(fp, index=False)
    TR.dump(fn, out)
    for nm, o in out['models'].items():
        t1 = o['tops']['0.01']; t3 = o['tops']['0.001']
        print(f"{nm:15s} auc rnd {o['auc_random']:.3f} nb {o['auc_nearby']} | top1% prec {t1['random_precision']:.3f} lead>=75 {t1['lead_ge_need']} fires above {t1['fires_share_above']:.3f} "
              f"| top0.1% prec {t3['random_precision']:.3f} (n {t3['random_n']}) | fires mean score {o['fires_mean_score']} label {o['fires_label_rate']} | random rate {o['random_label_rate']:.4f}")


if __name__ == '__main__':
    main()
