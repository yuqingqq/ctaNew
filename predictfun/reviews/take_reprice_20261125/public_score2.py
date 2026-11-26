"""Public score v2: the order-book flows of our findings added, and what the score is made of (USER 2026-11-26: "do we include binance ob
refill rate, far side shrink, near side new posts, and everything we need"; "we need to include the findings, and show me the sauce of the
score"). Development evidence on the four consumed days (the top taker traded on three); features chosen from studies on these days.
Points, labels, the six v1 families and the evaluation: public_score.py (its per-point table is read, not recomputed). Added family 'flow'
(fire_events' events on Binance futures, E clock: adds / cancels net of the traded volume at that price, trades by aggressor side):
far / near side x new orders (refill / posts) / cancels / trades over (t - 20, t], (t - 100, t - 20], (t - 300, t - 100]; time since the
last far / near new order and cancel; the far side's 300 ms shrink split into net withdrawal (cancels - new orders) and traded volume, and
the withdrawn share; the imbalance 20 ms before t and its change over 100 / 300 ms; the far / near size change over 100 ms.
Models (declared before the run): logistic on the v1 features (the v1 baseline, refit here), on v1 + flow and on flow alone; gradient-
boosted trees on v1 + flow (public_score's settings). Leave-one-day-out, trained on random + nearby moments, never on fires.
The sauce: (1) the logistic v1 + flow refit on all days (random + nearby): standardized coefficients; (2) family permutation importance of
the trees on the held-out days (each family's columns shuffled together among the held-out rows, seeded): the drop in random-moment AUC and
in top-1% precision; (3) profiles in natural units: medians at the held-out top-1% random moments of the logistic v1 + flow, at all random
moments, and at the top taker's fires.
Outputs (write-once): public_score2.json; OUT/public_score2_points.parquet.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/public_score2.py \
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
import public_score as PS  # noqa: E402

DC, C, TR, WT = FE.DC, FE.C, FE.TR, FE.WT
CAP = 5000.
SUB = ((20., 0., '20'), (100., 20., '100'), (300., 100., '300'))
ROLE = {'opp': 'far', 'near': 'near'}
KIND = {'add': 'new', 'cancel': 'cxl', 'trade': 'trade'}
FLOW = ([f'f_{ROLE[r]}_{KIND[k]}_{w}' for r in ROLE for k in KIND for _, _, w in SUB] + [f'f_since_{ROLE[r]}_{KIND[k]}' for r in ROLE for k in ('add', 'cancel')]
        + ['f_far_netwd', 'f_far_traded', 'f_far_wd_share', 'f_imb_20', 'f_imb_d100', 'f_imb_d300', 'f_far_r100', 'f_near_r100'])
FAM2 = dict(PS.FAMILIES, flow=FLOW)
V1, V2 = PS.ALL, PS.ALL + FLOW
PROFILE = dict(imbalance=('b_imb1', None), far_size_btc=('b_lfar1', 'exp'), near_size_btc=('b_lnear1', 'exp'), far_change_300ms=('b_rfar', 'exp'),
               near_change_300ms=('b_rnear', 'exp'), far_new_last100_btc=('raw_far_new_100', None), far_cxl_last100_btc=('raw_far_cxl_100', None),
               near_new_last100_btc=('raw_near_new_100', None), trades_its_way_last100=('t_n100_its', 'expm1'), trades_against_last100=('t_n100_ag', 'expm1'),
               since_far_new_ms=('f_since_far_new', 'expm1'), far_withdrawn_share_300ms=('f_far_wd_share', None), imbalance_change_300ms=('f_imb_d300', None),
               depth_imbalance_20=('d_imb20', None), spread_ticks=('b_spread', None))


def flow_feats(ev, traj, t, d):
    n = len(t)
    o = {c: np.full(n, np.nan) for c in FLOW}
    o.update(raw_far_new_100=np.full(n, np.nan), raw_far_cxl_100=np.full(n, np.nan), raw_near_new_100=np.full(n, np.nan))
    tot = {}
    for dv in (1., -1.):
        m = d == dv
        if not m.any():
            continue
        tt = t[m]
        for r in ROLE:
            for k in KIND:
                E, cA = FE.role_events(ev, r, k, dv)
                for a, b, w in SUB:
                    v = cA[np.searchsorted(E, tt - b, 'right')] - cA[np.searchsorted(E, tt - a, 'right')]
                    o[f'f_{ROLE[r]}_{KIND[k]}_{w}'][m] = np.log1p(np.maximum(v, 0.))
                v100 = cA[np.searchsorted(E, tt, 'right')] - cA[np.searchsorted(E, tt - 100., 'right')]
                if (r, k) in (('opp', 'add'), ('opp', 'cancel'), ('near', 'add')):
                    o[f"raw_{ROLE[r]}_{KIND[k]}_100"][m] = v100
                tot.setdefault((r, k), np.full(n, np.nan))[m] = cA[np.searchsorted(E, tt, 'right')] - cA[np.searchsorted(E, tt - 300., 'right')]
                if k in ('add', 'cancel'):
                    j = np.searchsorted(E, tt, 'right') - 1
                    since = np.where(j >= 0, tt - E[np.maximum(j, 0)], CAP)
                    o[f'f_since_{ROLE[r]}_{KIND[k]}'][m] = np.log1p(np.minimum(since, CAP))
    net = tot[('opp', 'cancel')] - tot[('opp', 'add')]
    trd = tot[('opp', 'trade')]
    o['f_far_netwd'] = np.sign(net) * np.log1p(np.abs(net))
    o['f_far_traded'] = np.log1p(np.maximum(trd, 0.))
    wd = np.maximum(net, 0.)
    o['f_far_wd_share'] = np.where(wd + trd > 0, wd / np.maximum(wd + trd, 1e-12), np.nan)
    o['f_imb_20'] = traj['imb_20']
    o['f_imb_d100'], o['f_imb_d300'] = traj['imb_0'] - traj['imb_100'], traj['imb_0'] - traj['imb_300']
    o['f_far_r100'], o['f_near_r100'] = np.log(traj['opp_0'] / traj['opp_100']), np.log(traj['near_0'] / traj['near_100'])
    return o


def fit_predict(kind, Xtr, ytr, Xte):
    if kind == 'logit':
        A, B = PS.design(Xtr, Xte)
        return LogisticRegression(C=1., max_iter=3000).fit(A, ytr).predict_proba(B)[:, 1], None
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1., random_state=PS.SEED).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m


def top_prec(score, y, day, q=.01):
    hits = n = 0
    for dd in np.unique(day):
        k = day == dd
        thr = np.quantile(score[k], 1. - q)
        a = score[k] >= thr
        hits += int(y[k][a].sum()); n += int(a.sum())
    return hits / max(n, 1)


def natural(Z, col, how):
    x = Z[col].to_numpy(float)
    return np.exp(x) if how == 'exp' else (np.expm1(x) if how == 'expm1' else x)


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    od = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    fn, fp = (od or H) / 'public_score2.json', (od or WT.OUT) / 'public_score2_points.parquet'
    C.require(not fn.exists() and not fp.exists(), 'an output exists (write-once)')
    days = tuple(sys.argv[2:]) if len(sys.argv) > 2 else WT.DAYS
    C.require(len(days) >= 2 and all(x in WT.DAYS for x in days), 'needs >= 2 declared days')
    Z = pd.read_parquet(WT.OUT / 'public_score_points.parquet', columns=['day', 't', 'd', 'kind', 'y', 'next_ms'] + V1)
    Z = Z[Z.day.isin(days)].reset_index(drop=True)
    parts = []
    for day in days:
        Zd = Z[Z.day == day].reset_index(drop=True)
        ev, st, traj, _ = FE.day_data(day, Zd[['t', 'd']])
        C.require((st == 'ok').all(), f'{day}: a point without a book')
        C.require((np.abs(traj['opp_0'] - np.exp(Zd.b_lfar1.to_numpy())) < 1e-6 * np.maximum(traj['opp_0'], 1.)).all(), f'{day}: the far size differs from public_score')
        for k, v in flow_feats(ev, traj, Zd.t.to_numpy(float), Zd.d.to_numpy(float)).items():
            Zd[k] = v
        parts.append(Zd)
        del ev
        C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
        print(day, len(Zd), flush=True)
    Z = pd.concat(parts, ignore_index=True)
    models = dict(logit_v1=('logit', V1), logit_v2=('logit', V2), gbm_v2=('gbm', V2), logit_flow=('logit', FLOW))
    for nm in models:
        Z[f's_{nm}'] = np.nan
    rng = np.random.default_rng(PS.SEED + 1)
    perm = {f: dict(auc=[], top1=[]) for f in FAM2}
    base = dict(auc=[], top1=[])
    for day in days:
        tr = ((Z.day != day) & (Z.kind != 'fire')).to_numpy()
        te = (Z.day == day).to_numpy()
        ytr = Z.y.to_numpy()[tr]
        for nm, (kind, cols) in models.items():
            s, mdl = fit_predict(kind, Z.loc[tr, cols].to_numpy(float), ytr, Z.loc[te, cols].to_numpy(float))
            Z.loc[te, f's_{nm}'] = s
            if nm == 'gbm_v2':
                Xte = Z.loc[te, cols].to_numpy(float)
                rnd = (Z.kind.to_numpy()[te] == 'random')
                yte = Z.y.to_numpy()[te][rnd]
                dte = Z.day.to_numpy()[te][rnd]
                s0 = s[rnd]
                base['auc'].append(roc_auc_score(yte, s0)); base['top1'].append(top_prec(s0, yte, dte))
                for f, fc in FAM2.items():
                    X2 = Xte[rnd].copy()
                    idx = [cols.index(c) for c in fc]
                    X2[:, idx] = X2[rng.permutation(len(X2))][:, idx]
                    s2 = mdl.predict_proba(X2)[:, 1]
                    perm[f]['auc'].append(roc_auc_score(yte, s2)); perm[f]['top1'].append(top_prec(s2, yte, dte))
        print('fold', day, flush=True)
    out = dict(asof='2026-11-26', days=list(days), n=dict(Z.kind.value_counts().astype(int)), label='move its way within 300 ms', families=FAM2,
               models={nm: PS.evaluate(Z, f's_{nm}') for nm in models})
    out['permutation_gbm_v2'] = dict(base_auc_mean=float(np.mean(base['auc'])), base_top1_mean=float(np.mean(base['top1'])),
                                     families={f: dict(auc_drop=float(np.mean(base['auc']) - np.mean(v['auc'])), top1_drop=float(np.mean(base['top1']) - np.mean(v['top1'])))
                                               for f, v in perm.items()})
    tr = (Z.kind != 'fire').to_numpy()
    Xa = Z.loc[tr, V2].to_numpy(float)
    A, _ = PS.design(Xa, Xa[:1])
    miss = np.isnan(Xa).any(axis=0)
    names = V2 + [f'{c}__missing' for c, mm in zip(V2, miss) if mm]
    lr = LogisticRegression(C=1., max_iter=3000).fit(A, Z.y.to_numpy()[tr])
    coef = sorted(zip(names, lr.coef_[0]), key=lambda kv: -abs(kv[1]))
    out['coefficients_logit_v2_all_days'] = [dict(feature=k, coef=float(v)) for k, v in coef]
    R = Z[Z.kind == 'random']
    top = np.zeros(len(R), bool)
    for dd, g in R.groupby('day'):
        thr = np.quantile(g.s_logit_v2, .99)
        top[(R.day == dd).to_numpy() & (R.s_logit_v2 >= thr).to_numpy()] = True
    Fz = Z[Z.kind == 'fire']
    out['profiles'] = {nm: dict(top1_random=float(np.nanmedian(natural(R[top], c, how))), all_random=float(np.nanmedian(natural(R, c, how))),
                                fires=float(np.nanmedian(natural(Fz, c, how))) if len(Fz) else None,
                                top1_random_mean=float(np.nanmean(natural(R[top], c, how))), all_random_mean=float(np.nanmean(natural(R, c, how))),
                                fires_mean=float(np.nanmean(natural(Fz, c, how))) if len(Fz) else None)
                       for nm, (c, how) in PROFILE.items()}
    out['profiles_n'] = dict(top1_random=int(top.sum()), all_random=int(len(R)), fires=int(len(Fz)))
    Z.to_parquet(fp, index=False)
    TR.dump(fn, out)
    for nm, o in out['models'].items():
        t1, t3 = o['tops']['0.01'], o['tops']['0.001']
        print(f"{nm:16s} auc {o['auc_random']:.3f} / {o['auc_nearby']:.3f} | top1% {t1['random_precision']:.3f} top0.1% {t3['random_precision']:.3f} | fires in top1% {t1['fires_share_above']:.3f} | fires mean {o['fires_mean_score']:.3f}")
    print(json.dumps(out['permutation_gbm_v2'], indent=0))
    print(json.dumps(out['coefficients_logit_v2_all_days'][:15], indent=0))
    print(json.dumps(out['profiles'], indent=0)[:3000])


if __name__ == '__main__':
    main()
