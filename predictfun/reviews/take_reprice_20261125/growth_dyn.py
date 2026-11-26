"""Growth rates of the order-book dynamics over the last 3 s, and whether they help (USER 2026-11-26: "In the longer window, can we further
check the dynamics growth rate, etc, and check if they help"). Development evidence on the four consumed days; no strategy change.
Points: public_score3's (random, nearby and the top taker's fires, with the v3 factors and held-out v3 scores) and smart_factors' (ranks
2-10 fires and nearby moments, with the v3 factors); all with no move their way in the previous 300 ms.
Growth factors (declared before the run): four series sampled every 100 ms over [t - 3 s, t] - best-level imbalance its way, log far / near
size at the best (Binance futures bookTicker on E, per-hour states; a sample before the hour's states voids the series), depth imbalance its
way over 20 levels (depth20, the snapshot at or before the sample, <= 250 ms old) - each with the slope per second over 3 s and over the last
1 s, the acceleration (last-second slope minus the slope over [t - 3 s, t - 1 s]), steadiness (share of the 30 steps moving its way:
imbalance / near size / depth imbalance up, far size down), R2 of the 3 s line, and the late share (the last 300 ms' part of the 3 s
change; |change| < 1e-9 -> none; clipped to [-2, 3]); flow growth (fire_events' events) for near-side new orders, far-side cancels,
far-side new orders, trades its way and against: log rate ratios last 300 ms / the 700 ms before, last 1 s / the 2 s before (e = 1e-5 BTC
/ ms). Check: the imbalance and far size sampled at t equal the v3 factors.
Does it help (declared): (1) per group (top taker, rank 2, ranks 3-10) the separation of fires from nearby moments of every growth factor;
(2) imitation: a logistic model of fire vs nearby, v3 factors against v3 + growth, leave-one-day-out over the days with >= 50 fires, held-out
AUC; (3) prediction: the public score for a move within 300 ms, logistic and trees on v3 + growth against public_score3's held-out v3 scores
on the same points, leave-one-day-out, trained on random + nearby moments (public_score's evaluation).
Outputs (write-once): growth_dyn.json; OUT/growth_dyn_points.parquet.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/growth_dyn.py \
       [<rehearsal dir> <day> <day> ...]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import fire_events as FE  # noqa: E402
import public_score as PS  # noqa: E402
import public_score2 as P2  # noqa: E402
import public_score3 as P3  # noqa: E402
import smart_combo as SC  # noqa: E402

DC, C, TR, WT = FE.DC, FE.C, FE.TR, FE.WT
OFFS = np.arange(-3000., 1., 100.)
EPS = 1e-5
SERIES = (('imb', 1.), ('lfar', -1.), ('lnear', 1.), ('dimb', 1.))
MEAS = ('slope3', 'slope1', 'accel', 'steady', 'r2', 'late')
FLOWS = (('near', 'add', 'near_new'), ('opp', 'cancel', 'far_cxl'), ('opp', 'add', 'far_new'), ('opp', 'trade', 'its_trade'), ('near', 'trade', 'ag_trade'))
GROW = [f'g_{s}_{m}' for s, _ in SERIES for m in MEAS] + [f'g_{n}_{k}' for _, _, n in FLOWS for k in ('g1', 'g2')]


def slope(x, S):
    xm = x - x.mean()
    return (S - S.mean(axis=1, keepdims=True)) @ xm / (xm @ xm)


def series_feats(S, sgn):
    """S: n x 31 samples (NaN rows void) -> the six measures."""
    x = OFFS / 1000.
    ok = np.isfinite(S).all(axis=1)
    Sf = np.where(ok[:, None], S, 0.)
    s3, s1, se = slope(x, Sf), slope(x[-11:], Sf[:, -11:]), slope(x[:21], Sf[:, :21])
    d = np.diff(Sf, axis=1)
    steady = (sgn * d > 1e-12).mean(axis=1)
    b = s3[:, None] * (x - x.mean())[None, :] + Sf.mean(axis=1, keepdims=True)
    ss_res, ss_tot = ((Sf - b) ** 2).sum(1), ((Sf - Sf.mean(axis=1, keepdims=True)) ** 2).sum(1)
    r2 = np.where(ss_tot > 1e-18, 1. - ss_res / np.maximum(ss_tot, 1e-18), np.nan)
    tot = Sf[:, -1] - Sf[:, 0]
    late = np.where(np.abs(tot) > 1e-9, np.clip((Sf[:, -1] - Sf[:, -4]) / np.where(np.abs(tot) > 1e-9, tot, 1.), -2., 3.), np.nan)
    o = dict(slope3=s3, slope1=s1, accel=s1 - se, steady=steady, r2=r2, late=late)
    return {k: np.where(ok, v, np.nan) for k, v in o.items()}


def day_growth(day, P):
    day0 = float(WT.OD_day_ms(day))
    t, d = P.t.to_numpy(float), P.d.to_numpy(float)
    n = len(t)
    F = {c: np.full(n, np.nan) for c in GROW}
    F.update(chk_imb=np.full(n, np.nan), chk_lfar=np.full(n, np.nan))
    ev, st, _, _ = FE.day_data(day, P[['t', 'd']])
    C.require((st == 'ok').all(), f'{day}: a point without a book')
    for dv in (1., -1.):
        m = d == dv
        if not m.any():
            continue
        tt = t[m]
        for r, k, nm in FLOWS:
            E, cA = FE.role_events(ev, r, k, dv)
            w = lambda a, b: P3.wsum(E, cA, tt - a, tt - b)
            F[f'g_{nm}_g1'][m] = np.log((w(300., 0.) / 300. + EPS) / (w(1000., 300.) / 700. + EPS))
            F[f'g_{nm}_g2'][m] = np.log((w(1000., 0.) / 1000. + EPS) / (w(3000., 1000.) / 2000. + EPS))
    del ev
    hp = ((t - day0) // 3600000).astype(int)
    for h, bk in DC.book_hours(day):
        sel = np.flatnonzero(hp == h)
        if not len(sel):
            continue
        E, bid, bq, ask, aq = FE.collapse(*bk)
        tt, dd = t[sel], d[sel]
        up = (dd > 0)[:, None]
        tau = tt[:, None] + OFFS[None, :]
        j = np.searchsorted(E, tau, 'right') - 1
        okb = (j >= 0) & (tau >= E[0])
        jj = np.maximum(j, 0)
        far, near = np.where(up, aq[jj], bq[jj]), np.where(up, bq[jj], aq[jj])
        imb = np.where(okb, (near - far) / (near + far), np.nan)
        lfar, lnear = np.where(okb, np.log(far), np.nan), np.where(okb, np.log(near), np.nan)
        F['chk_imb'][sel], F['chk_lfar'][sel] = imb[:, -1], lfar[:, -1]
        fd = DC.RAW / f'depth20/BTCUSDT/{day}_{h:02d}.csv.gz'
        dimb = np.full(imb.shape, np.nan)
        if fd.exists():
            D = pd.read_csv(fd, header=None, usecols=[1, 4, 5], names=['E', 'b', 'a']).sort_values('E', kind='stable')
            Ed = D.E.to_numpy(float)
            bP, bQ = P3.fast_parse(D.b.to_numpy())
            aP, aQ = P3.fast_parse(D.a.to_numpy())
            bs, as_ = np.nansum(bQ, 1), np.nansum(aQ, 1)
            I = (bs - as_) / np.maximum(bs + as_, 1e-12)
            jd = np.searchsorted(Ed, tau, 'right') - 1
            okd = (jd >= 0) & (tau - Ed[np.maximum(jd, 0)] <= PS.DD.STALE)
            dimb = np.where(okd, np.where(up, 1., -1.) * I[np.maximum(jd, 0)], np.nan)
        for nm, S in (('imb', imb), ('lfar', lfar), ('lnear', lnear), ('dimb', dimb)):
            sgn = dict(SERIES)[nm]
            for k, v in series_feats(S, sgn).items():
                F[f'g_{nm}_{k}'][sel] = v
        C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
    return F


def imitation(Z, cols, groups):
    out = {}
    for g in groups:
        G = Z[(Z.grp == g) & Z.kind.isin(['fire', 'nearby'])]
        days = [dd for dd, x in G.groupby('day') if (x.kind == 'fire').sum() >= 50]
        ys, ss = [], []
        for dd in days:
            tr, te = G[(G.day != dd) & G.day.isin(days)], G[G.day == dd]
            A, B = PS.design(tr[cols].to_numpy(float), te[cols].to_numpy(float))
            mdl = LogisticRegression(C=1., max_iter=3000).fit(A, (tr.kind == 'fire').to_numpy(int))
            ys.append((te.kind == 'fire').to_numpy(int)); ss.append(mdl.predict_proba(B)[:, 1])
        out[g] = float(roc_auc_score(np.concatenate(ys), np.concatenate(ss))) if ys else None
    return out


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (2700, 2701))
    od = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    fn, fp = (od or H) / 'growth_dyn.json', (od or WT.OUT) / 'growth_dyn_points.parquet'
    C.require(not fn.exists() and not fp.exists(), 'an output exists (write-once)')
    days = tuple(sys.argv[2:]) if len(sys.argv) > 2 else WT.DAYS
    C.require(len(days) >= 2 and all(x in WT.DAYS for x in days), 'needs >= 2 declared days')
    T1 = pd.read_parquet(WT.OUT / 'public_score3_points.parquet', columns=['day', 't', 'd', 'kind', 'y', 'next_ms', 's_logit_v3', 's_gbm_v3'] + P3.V3)
    T1 = T1[T1.day.isin(days)].reset_index(drop=True)
    T1['grp'] = 'TOP1'
    S2 = pd.read_parquet(WT.OUT / 'smart_factors_points.parquet', columns=['day', 't', 'd', 'grp', 'fire', 'y', 'next_ms'] + P3.V3)
    S2 = S2[S2.day.isin(days)].reset_index(drop=True)
    S2['kind'] = np.where(S2.fire, 'fire', 'nearby')
    Z = pd.concat([T1, S2.drop(columns=['fire'])], ignore_index=True)
    parts = []
    for day in days:
        Zd = Z[Z.day == day].reset_index(drop=True)
        for k, v in day_growth(day, Zd).items():
            Zd[k] = v
        okc = Zd.chk_imb.notna()
        C.require((np.abs(Zd.chk_imb[okc] - Zd.b_imb1[okc]) < 1e-9).all() and (np.abs(Zd.chk_lfar[okc] - Zd.b_lfar1[okc]) < 1e-9).all(), f'{day}: the sampled state differs from the v3 factors')
        parts.append(Zd)
        print(day, len(Zd), {c: round(float(Zd[c].notna().mean()), 3) for c in ('g_imb_slope3', 'g_dimb_slope3', 'g_near_new_g2')}, flush=True)
    Z = pd.concat(parts, ignore_index=True)
    out = dict(asof='2026-11-26', days=list(days), n={f'{g}|{k}': int(v) for (g, k), v in Z.groupby(['grp', 'kind']).size().items()},
               coverage={c: float(Z[c].notna().mean()) for c in GROW}, separation={}, imitation={}, prediction={})
    for g in ('TOP1', 'RANK2', 'RANKS3_10'):
        Fz, N = Z[(Z.grp == g) & (Z.kind == 'fire')], Z[(Z.grp == g) & (Z.kind == 'nearby')]
        rows = []
        for c in GROW:
            r = SC.sep(Fz[c].to_numpy(float), N[c].to_numpy(float))
            if r:
                rows.append(dict(r, factor=c))
        out['separation'][g] = sorted(rows, key=lambda r: -r['separation'])
    out['imitation'] = dict(v3=imitation(Z, P3.V3, ('TOP1', 'RANK2', 'RANKS3_10')), v3_growth=imitation(Z, P3.V3 + GROW, ('TOP1', 'RANK2', 'RANKS3_10')),
                            growth_only=imitation(Z, GROW, ('TOP1', 'RANK2', 'RANKS3_10')))
    keep = ['day', 't', 'd', 'grp', 'kind', 'y'] + GROW
    Z[keep].to_parquet(fp, index=False)
    T = Z[Z.grp == 'TOP1'].drop(columns=['chk_imb', 'chk_lfar']).reset_index(drop=True)
    del Z, parts
    models = dict(logit_v3g=('logit', P3.V3 + GROW), gbm_v3g=('gbm', P3.V3 + GROW))
    for nm in models:
        T[f's_{nm}'] = np.nan
    for day in days:
        tr = ((T.day != day) & (T.kind != 'fire')).to_numpy()
        te = (T.day == day).to_numpy()
        for nm, (kind, cols) in models.items():
            s, _ = P2.fit_predict(kind, T.loc[tr, cols].to_numpy(float), T.y.to_numpy()[tr], T.loc[te, cols].to_numpy(float))
            T.loc[te, f's_{nm}'] = s
        print('fold', day, flush=True)
    out['prediction'] = {nm: PS.evaluate(T, f's_{nm}') for nm in ('logit_v3', 'gbm_v3', 'logit_v3g', 'gbm_v3g')}
    TR.dump(fn, out)
    for g, rows in out['separation'].items():
        print(g, [(r['factor'], round(r['separation'], 3), r['direction']) for r in rows[:8]])
    print('imitation', json.dumps(out['imitation']))
    for nm, o in out['prediction'].items():
        t1, t3 = o['tops']['0.01'], o['tops']['0.001']
        print(f"{nm:10s} auc {o['auc_random']:.3f} / {o['auc_nearby']:.3f} | top1% {t1['random_precision']:.3f} top0.1% {t3['random_precision']:.3f} | fires in top1% {t1['fires_share_above']:.3f}")


if __name__ == '__main__':
    main()
