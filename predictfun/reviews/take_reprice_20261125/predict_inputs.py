"""How much of the take / markout models' information comes from Predict? (USER 2026-11-27: "So if we consider predict fun market
conditions, together with the factors, we can actually decide better, can you check how much information we are using from predict").
Readout over TAKE_MODEL's 79 features and TAKE_MARKOUT / TAKE_MARKOUT_EXIT's labels (T = 172 ms); no new data. Declared before the run.
Families (by name, every feature in exactly one, checked): Predict: price level (pf_mid, pf_extreme, pf_its_price), time left (p_tau),
book at the decision (pf_spread, pf_far1, pf_near1, pf_far5, pf_near5), its mid's 1 s move (pf_mid_ret_1000; book on our receipt clock),
other takers' trades (p_pf5, q_net_share5, q_fill_r1s; fills on the venue's match clock, including REST trades timed from book drops:
not all known at the decision); Binance futures: best level (b_*, d_*), flows (x_*, NORM_WINDOWS' rates), deep book (z_*); other venues
(u_*).
Targets: the markout move its way 1 s / 5 s / 30 s after the match and at settlement (100 x payout). Feature sets: all; Predict only;
without Predict; price level + time left (pf_its_price, p_tau); all but other takers' trades. Pinned trees, leave-one-day-out; 'all'
reproduces TAKE_MARKOUT's predictions (a check).
Readouts per target and set: held-out Spearman and R2 (against the pooled mean); the top 1 % by predicted edge at the known best (TAKE_
MARKOUT's rule): realised move and the value of the order at the known best sold back after h (settlement: held), per intended share,
days > 0. Family permutation importance on 'all': the held-out Spearman drop when a family's columns are shuffled together within the
held-out day (2 seeded repeats); positive control: every column shuffled must bring |Spearman| under 0.03.
Output (write-once): predict_inputs.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/predict_inputs.py \
       [<rehearsal dir> <market fraction>]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import take_markout as TK  # noqa: E402
import take_markout_exit as TE  # noqa: E402
import take_model_insample as TI  # noqa: E402

TM, C, TR, Q, DAYS, fee = TK.TM, TK.C, TK.TR, TK.Q, TK.DAYS, TK.fee
FAMILIES = dict(
    predict_price_level=lambda c: c in ('pf_mid', 'pf_extreme', 'pf_its_price'),
    predict_time_left=lambda c: c == 'p_tau',
    predict_book=lambda c: c in ('pf_spread', 'pf_far1', 'pf_near1', 'pf_far5', 'pf_near5'),
    predict_mid_move=lambda c: c == 'pf_mid_ret_1000',
    predict_trades=lambda c: c in ('p_pf5', 'q_net_share5', 'q_fill_r1s'),
    binance_best_level=lambda c: c.startswith('b_') or c.startswith('d_'),
    binance_flows=lambda c: c.startswith('x_') or '|' in c,
    binance_deep=lambda c: c.startswith('z_'),
    other_venues=lambda c: c.startswith('u_'))
SEEDS = (20261127, 20261128)
CONTROL_MAX = .03


def families(feats):
    fam = {f: [c for c in feats if FAMILIES[f](c)] for f in FAMILIES}
    got = [c for v in fam.values() for c in v]
    C.require(sorted(got) == sorted(feats) and len(got) == len(set(got)), 'every feature in exactly one family')
    return fam


def readout(pred, y, ok, cost, val, day):
    p, t = pred[ok], y[ok]
    idx = TK.top(pred - cost, ok, .01)
    v = val[idx] / Q
    by = {dh: (float(v[day[idx] == dh].mean()) if (day[idx] == dh).any() else None) for dh in DAYS}
    return dict(spearman=float(spearmanr(p, t).correlation), r2=float(1. - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)),
                top1=dict(n=int(len(idx)), realised_move_c=float(y[idx].mean()), value_c=float(v.mean()), by_day=by,
                          days_positive=int(sum(1 for x in by.values() if x is not None and x > 0))))


def lodo(X, cols, y, ok, day):
    pred, models = np.full(len(X), np.nan), {}
    for dh in DAYS:
        tr, te = ok & (day != dh), day == dh
        if te.sum() == 0 or tr.sum() == 0:
            continue
        m = HistGradientBoostingRegressor(random_state=0, **TM.HGB).fit(X.loc[tr, cols], y[tr])
        pred[te] = m.predict(X.loc[te, cols]); models[dh] = m
    return pred, models


def permutation(models, X, cols, fam, y, ok, day, base):
    out = {}
    for f in list(fam) + ['every_column']:
        fc = cols if f == 'every_column' else fam[f]
        drops, rhos = [], []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            pred = np.full(len(X), np.nan)
            for dh, m in models.items():
                te = np.flatnonzero(day == dh)
                Z = X.loc[te, cols].copy()
                Z[fc] = Z[fc].to_numpy()[rng.permutation(len(te))]
                pred[te] = m.predict(Z)
            r = float(spearmanr(pred[ok], y[ok]).correlation)
            rhos.append(r); drops.append(base - r)
        out[f] = dict(spearman_drop=float(np.mean(drops)), drops=drops, spearman_permuted=rhos)
    C.require(all(abs(r) < CONTROL_MAX for r in out['every_column']['spearman_permuted']), f'positive control failed: {out["every_column"]}')
    return out


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    reh = len(sys.argv) > 1
    fn = (Path(sys.argv[1]) if reh else H) / 'predict_inputs.json'
    C.require(not fn.exists(), 'predict_inputs.json exists (write-once)')
    X, feats = TI.frame()
    fam = families(feats)
    K = pd.concat([pd.read_parquet(TK.OUT / f'{d}_labels.parquet') for d in DAYS], ignore_index=True)
    E = pd.concat([pd.read_parquet(TE.OUT / f'{d}_labels.parquet') for d in DAYS], ignore_index=True)
    Pr = pd.read_parquet(TK.OUT / 'predictions.parquet')
    C.require(len(K) == len(E) == len(Pr) == len(X), 'labels do not cover the population')
    X = X.merge(K.drop(columns=['day', 'mid', 't', 'd']), on='row', how='left', validate='1:1')
    X = X.merge(E.drop(columns=['day', 'mid', 'd']), on='row', how='left', validate='1:1').merge(Pr.drop(columns=['day']), on='row', how='left', validate='1:1')
    if reh:
        mids = np.unique(X.mid.to_numpy())
        keep = set(np.random.default_rng(1).choice(mids, max(8, int(float(sys.argv[2]) * len(mids))), replace=False).tolist())
        X = X[X.mid.isin(keep)].reset_index(drop=True)
    d, lim, md, pay, day = X.d.to_numpy(float), X.lim.to_numpy(float), X.md.to_numpy(float), X.payout.to_numpy(float), X.day.to_numpy()
    C.require(np.isfinite(pay).all(), 'a moment without its payout')
    cost = d * (lim - md) + fee(lim)
    PA = X[[f'pA172_{k}' for k in range(5)]].to_numpy(float); SA = X[[f'sA172_{k}' for k in range(5)]].to_numpy(float)
    fl, co, fe = TK.vwalk(PA, SA, d, lim)
    targets = {}
    for hs in ('1s', '5s', '30s'):
        mk = X[f'm{hs}_172'].to_numpy(float)
        eP = X[[f'eP{hs}_{k}' for k in range(5)]].to_numpy(float); eS = X[[f'eS{hs}_{k}' for k in range(5)]].to_numpy(float)
        back, hold, _ = TE.exit_values(fl, co, fe, d, eP, eS, pay)
        targets[hs] = (np.where(np.isfinite(mk), d * (mk - md), np.nan), np.isfinite(mk), back, 'sold_back_after_h')
    targets['settlement'] = (d * (100. * pay - md), np.ones(len(X), bool), hold, 'held_to_settlement')
    PRED = [c for f in fam if f.startswith('predict_') for c in fam[f]]
    sets = dict(all=feats, predict_only=PRED, without_predict=[c for c in feats if c not in PRED],
                price_level_and_time_left=['pf_its_price', 'p_tau'], all_but_other_takers_trades=[c for c in feats if c not in fam['predict_trades']])
    out = dict(asof='2026-11-27', rows=int(len(X)), rehearsal=reh, families=fam, set_sizes={k: len(v) for k, v in sets.items()}, targets={})
    for tg, (y, ok, val, vname) in targets.items():
        o = dict(n=int(ok.sum()), value=vname, sets={})
        for sn, cols in sets.items():
            pred, models = lodo(X, cols, y, ok, day)
            r = readout(pred, y, ok, cost, val, day)
            if sn == 'all':
                if tg != 'settlement' and not reh:
                    gap = float(np.nanmax(np.abs(pred - X[f'pred_172_{tg}'].to_numpy(float))))
                    C.require(gap < 1e-9, f'TAKE_MARKOUT not reproduced at {tg}: {gap}')
                r['permutation'] = permutation(models, X, cols, fam, y, ok, day, r['spearman'])
            o['sets'][sn] = r
            print(tg, sn, round(r['spearman'], 3), round(r['top1']['value_c'], 2), flush=True)
        a = o['sets']['all']['spearman']
        o['predicates'] = dict(predict_only_share_of_all=o['sets']['predict_only']['spearman'] / a if a else None,
                               without_predict_share_of_all=o['sets']['without_predict']['spearman'] / a if a else None,
                               largest_family=max(fam, key=lambda f: o['sets']['all']['permutation'][f]['spearman_drop']),
                               predict_families_drop=sum(o['sets']['all']['permutation'][f]['spearman_drop'] for f in fam if f.startswith('predict_')),
                               other_families_drop=sum(o['sets']['all']['permutation'][f]['spearman_drop'] for f in fam if not f.startswith('predict_')))
        out['targets'][tg] = o
        C.require(F_ok(), 'RSS over the self-guard')
    TR.dump(fn, out)
    print(json.dumps({tg: o['predicates'] for tg, o in out['targets'].items()}, indent=1))


def F_ok():
    return TK.F.rss_mib() < TK.F.RSS_GUARD_MIB


if __name__ == '__main__':
    main()
