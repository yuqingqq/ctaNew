"""What separates the top taker's signal from noise? (USER 2026-11-27: "So they react to Orderbook dynamics, and filter out the noise, an
you show me a typical signal they extract and a normal one without the signal, can we check deeply whether there is any difference, is it
possible for us to filter out the noise in normal conditions as well"). Four consumed days; no strategy change. Declared before the run.
Points: public_score3's decision moments (no Binance futures move their way in the previous 300 ms): the top taker's fires (t = its order's
arrival at Predict) and random moments; label y = the Binance futures mid moves their way within 300 ms (public_score's). Random keys that
occur twice are dropped (a status).
Inputs: every input of public_score3 (60), public_score2 (those not in v3), growth_dyn (the top taker's rows for fires) and norm_windows,
joined on (day, t, d, kind); causal at t by their builders. The held-out public score (s_gbm_v3) is used for matching only.
Stage read:
1 look-alikes: each fire's 5 random moments of the same day and direction within +-30 min with the nearest held-out score: moments our
  model rates like the fire. Move rate of fires vs look-alikes; the score gap.
2 one input at a time: AUC of each input, fires vs their look-alikes; null: the largest |AUC - 0.5| over inputs with the fire label
  shuffled within each matched set (200 draws); inputs beyond the null's 95th percentile, with medians in natural units.
3 all inputs together: pinned trees separating fires from look-alikes, leave-one-day-out: held-out AUC with its Hanley-McNeil standard
  error; shuffled-label control (10 draws, must sit near 0.5).
4 filtering the noise ourselves: pinned trees on all inputs predicting y on random moments (leave-one-day-out): precision in the top 1 /
  0.1 % of each held-out day against the public score's and the fires' rate; the fires' share above each threshold.
6 imitation: the section-3 trees (fire vs look-alike), applied to the held-out day's random moments: move precision in the top 1 / 0.1 %
  of fire-likeness, and among the public score's top 5 % the move rate in the top fifth by fire-likeness vs by the score.
5 examples chosen for the chart: the typical fire (followed by a move; nearest the fires' median on key standardized state and flow
  inputs), its nearest look-alike without a move on the same inputs, and the typical random moment without a move.
Stage examples: Binance futures best level (bookTicker, last state per ms) and trades (masked prints dropped) over [-1000, +300] ms around
each example, with the first best-price change their way after t.
Outputs (write-once): signal_vs_noise.json (read), signal_vs_noise_examples.json (examples).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/signal_vs_noise.py \
       selftest | read [<dir> <market fraction>] | examples [<dir>]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import depletion_cause as DC  # noqa: E402
import take_model as TM  # noqa: E402

C, TR, WT = DC.C, DC.TR, DC.WT
BASE = TM.WT.OUT
DAYS = TM.DAYS
K_CTRL, WIN_MS, N_NULL, N_SHUF = 5, 1800000., 200, 10
HGB = dict(max_iter=200, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.)
KEYS = ['day', 't', 'd', 'kind']
NOT_INPUT = {'day', 't', 'd', 'kind', 'y', 'next_ms', 'grp', 'row'}
KEY_INPUTS = ['b_imb1', 'b_lfar1', 'b_lnear1', 'x_far_cxl_r300', 'x_far_new_r300', 'x_near_new_r300', 'x_far_trade_r300', 'x_imb_d300', 'z_far_rem5_r300']
SEED = 20261127


def auc(score, lab):
    """Mann-Whitney AUC of score for lab = 1 vs 0, NaN scores dropped."""
    ok = np.isfinite(score)
    s, y = score[ok], lab[ok].astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.) / (n1 * n0))


def hanley_se(a, n1, n0):
    q1, q2 = a / (2. - a), 2. * a * a / (1. + a)
    return float(np.sqrt((a * (1. - a) + (n1 - 1.) * (q1 - a * a) + (n0 - 1.) * (q2 - a * a)) / (n1 * n0)))


def f32(D):
    num = [c for c in D.columns if c not in ('day', 'kind', 'grp') and D[c].dtype.kind == 'f' and c != 't']
    D[num] = D[num].astype(np.float32)
    return D


def load():
    P3 = pd.read_parquet(BASE / 'public_score3_points.parquet')
    P2 = pd.read_parquet(BASE / 'public_score2_points.parquet')
    C.require(len(P2) == len(P3) and (P2[KEYS].values == P3[KEYS].values).all(), 'public_score2 and 3 rows differ')
    keep = P3.kind.isin(['fire', 'random']).to_numpy()
    extra2 = [c for c in P2.columns if c not in P3.columns and not c.startswith('s_') and c not in NOT_INPUT]
    X = pd.concat([f32(P3[keep].reset_index(drop=True)), f32(P2.loc[keep, extra2].reset_index(drop=True))], axis=1)
    del P3, P2
    dup = X.duplicated(['day', 't', 'd', 'kind'], keep=False)
    status = dict(dropped_duplicate_keys=int(dup.sum()))
    X = X[~dup].reset_index(drop=True)
    G = pd.read_parquet(BASE / 'growth_dyn_points.parquet')
    G = f32(G[(G.kind == 'random') | ((G.kind == 'fire') & (G.grp == 'TOP1'))].drop(columns=['grp', 'y']).reset_index(drop=True))
    G = G[~G.duplicated(['day', 't', 'd', 'kind'], keep=False)]
    NW = pd.concat([f32(n[n.kind.isin(['fire', 'random'])].drop(columns=['y'], errors='ignore').reset_index(drop=True))
                    for n in (pd.read_parquet(BASE / 'norm_windows' / f'{d}.parquet') for d in DAYS)], ignore_index=True)
    NW = NW[~NW.duplicated(['day', 't', 'd', 'kind'], keep=False)]
    n0 = len(X)
    X = X.merge(G, on=KEYS, how='left', validate='1:1').merge(NW, on=KEYS, how='left', validate='1:1', suffixes=('', '_nw'))
    C.require(len(X) == n0, 'a join changed the row count')
    del G, NW
    C.require(TM.F.rss_mib() < TM.F.RSS_GUARD_MIB, 'RSS over the self-guard after loading')
    status.update(fires=int((X.kind == 'fire').sum()), random=int((X.kind == 'random').sum()),
                  growth_missing=int(X['g_imb_slope3'].isna().sum()), nw_missing=int(X['far_new|raw'].isna().sum()))
    inputs = [c for c in X.columns if c not in NOT_INPUT and not c.startswith('s_') and not c.endswith('_nw')]
    C.require(len(inputs) == len(set(inputs)), 'duplicate input names')
    return X, inputs, status


def match(X):
    """-> controls: index of each fire's K_CTRL look-alikes (same day and direction, +-30 min, nearest held-out score)."""
    F = X.index[X.kind == 'fire'].to_numpy()
    out = np.full((len(F), K_CTRL), -1, np.int64)
    s = X.s_gbm_v3.to_numpy(float); t = X.t.to_numpy(float)
    for (day, d), g in X[X.kind == 'random'].groupby(['day', 'd']):
        gi = g.index.to_numpy(); o = np.argsort(t[gi], kind='stable'); gi = gi[o]; tg = t[gi]
        for k, fi in enumerate(F):
            if X.day.iat[fi] != day or X.d.iat[fi] != d:
                continue
            a, b = np.searchsorted(tg, t[fi] - WIN_MS), np.searchsorted(tg, t[fi] + WIN_MS, 'right')
            cand = gi[a:b]
            cand = cand[np.isfinite(s[cand])]
            if len(cand) >= K_CTRL:
                out[k] = cand[np.argsort(np.abs(s[cand] - s[fi]), kind='stable')[:K_CTRL]]
    return F, out


def robust_z(X, cols, ref):
    med = X.loc[ref, cols].median(); iqr = (X.loc[ref, cols].quantile(.75) - X.loc[ref, cols].quantile(.25)).replace(0, 1.)
    return ((X[cols] - med) / iqr).to_numpy(float)


def read(od=None, frac=None):
    resource.setrlimit(resource.RLIMIT_CPU, (3000, 3001))
    reh = od is not None
    fn = (Path(od) if od else H) / 'signal_vs_noise.json'
    C.require(not fn.exists(), 'signal_vs_noise.json exists (write-once)')
    X, inputs, status = load()
    if reh and frac:
        keep = np.random.default_rng(1).random(len(X)) < frac
        X = X[keep | (X.kind == 'fire')].reset_index(drop=True)
    C.require(all(c in inputs for c in KEY_INPUTS), 'a key input is missing')
    out = dict(asof='2026-11-27', rehearsal=reh, status=status, inputs=len(inputs))
    # 1 look-alikes
    F, Ct = match(X)
    okm = (Ct >= 0).all(1)
    F, Ct = F[okm], Ct[okm]
    y, s = X.y.to_numpy(float), X.s_gbm_v3.to_numpy(float)
    out['lookalikes'] = dict(fires=int(len(F)), fires_unmatched=int((~okm).sum()), controls=int(Ct.size), distinct_controls=int(np.unique(Ct).size),
                             fire_move_rate=float(y[F].mean()), lookalike_move_rate=float(y[Ct].mean()), fire_score=float(s[F].mean()), lookalike_score=float(s[Ct].mean()),
                             score_gap_abs_median=float(np.median(np.abs(s[Ct] - s[F][:, None]))),
                             random_move_rate=float(y[X.kind.to_numpy() == 'random'].mean()), by_day={})
    for dh in DAYS:
        m = X.day.to_numpy()[F] == dh
        if m.any():
            out['lookalikes']['by_day'][dh] = dict(fires=int(m.sum()), fire_move_rate=float(y[F][m].mean()), lookalike_move_rate=float(y[Ct[m]].mean()))
    print('look-alikes', out['lookalikes'], flush=True)
    # 2 one input at a time
    idx = np.concatenate([F, Ct.ravel()])
    lab = np.concatenate([np.ones(len(F)), np.zeros(Ct.size)])
    sets = np.concatenate([np.arange(len(F)), np.repeat(np.arange(len(F)), K_CTRL)])
    V = X.loc[idx, inputs].to_numpy(float)
    aucs = np.array([auc(V[:, j], lab) for j in range(V.shape[1])])
    R = np.full(V.shape, np.nan); okv = np.isfinite(V)
    for j in range(V.shape[1]):
        R[okv[:, j], j] = rankdata(V[okv[:, j], j])
    rng = np.random.default_rng(SEED)
    order = np.argsort(sets, kind='stable')
    null_max = []
    for _ in range(N_NULL):
        perm_lab = lab.copy()
        for g in np.split(order, np.flatnonzero(np.diff(sets[order])) + 1):
            perm_lab[g] = rng.permutation(lab[g])
        pos = perm_lab == 1
        n1 = (okv & pos[:, None]).sum(0); n0 = (okv & ~pos[:, None]).sum(0)
        rs = np.where(okv & pos[:, None], R, 0.).sum(0)
        a = (rs - n1 * (n1 + 1) / 2.) / np.maximum(n1 * n0, 1)
        null_max.append(float(np.nanmax(np.abs(a - .5))))
    thr = float(np.quantile(null_max, .95))
    sig = [(inputs[j], float(aucs[j])) for j in np.argsort(-np.abs(np.nan_to_num(aucs, nan=.5) - .5)) if np.isfinite(aucs[j]) and abs(aucs[j] - .5) > thr]
    top = []
    for nm, a in sig[:25] if sig else [(inputs[j], float(aucs[j])) for j in np.argsort(-np.abs(np.nan_to_num(aucs, nan=.5) - .5))[:10]]:
        v = X[nm].to_numpy(float)
        top.append(dict(input=nm, auc=a, fire_median=float(np.nanmedian(v[F])), lookalike_median=float(np.nanmedian(v[Ct.ravel()])),
                        random_median=float(np.nanmedian(v[X.kind.to_numpy() == 'random']))))
    out['one_at_a_time'] = dict(inputs=len(inputs), null_draws=N_NULL, null_p95_abs_dev=thr, beyond_null=len(sig), top=top,
                                max_abs_dev=float(np.nanmax(np.abs(aucs - .5))))
    print('one at a time', len(sig), 'beyond null', round(thr, 4), [(d_['input'], round(d_['auc'], 3)) for d_ in top[:10]], flush=True)
    # 3 all inputs together
    D_ = X.day.to_numpy()[idx]
    Xi = X.loc[idx, inputs].reset_index(drop=True)

    def lodo_auc(labels):
        p = np.full(len(labels), np.nan)
        for dh in DAYS:
            tr, te = D_ != dh, D_ == dh
            if te.sum() == 0 or len(np.unique(labels[tr])) < 2:
                continue
            p[te] = HistGradientBoostingClassifier(random_state=0, **HGB).fit(Xi[tr], labels[tr]).predict_proba(Xi[te])[:, 1]
        return auc(p, labels), p
    a_all, p_all = lodo_auc(lab)
    shuf = []
    for k in range(N_SHUF if not reh else 2):
        pl = lab.copy()
        for g in np.split(order, np.flatnonzero(np.diff(sets[order])) + 1):
            pl[g] = rng.permutation(lab[g])
        shuf.append(lodo_auc(pl)[0])
    imit_models = {}
    for dh in DAYS:
        tr = D_ != dh
        imit_models[dh] = HistGradientBoostingClassifier(random_state=0, **HGB).fit(Xi[tr], lab[tr])
    out['all_together'] = dict(auc=a_all, se=hanley_se(a_all, len(F), Ct.size), shuffled_auc_mean=float(np.mean(shuf)), shuffled_auc=[float(x) for x in shuf],
                               by_day={dh: auc(p_all[D_ == dh], lab[D_ == dh]) for dh in DAYS if (D_ == dh).any()})
    C.require(abs(np.mean(shuf) - .5) < .03, f'shuffled-label control off 0.5: {shuf}')
    print('all together', out['all_together'], flush=True)
    # 4 filtering the noise ourselves
    ra = np.flatnonzero(X.kind.to_numpy() == 'random')
    Xr = X.loc[ra, inputs].reset_index(drop=True); yr = y[ra]; Dr = X.day.to_numpy()[ra]
    pr = np.full(len(ra), np.nan)
    models = {}
    for dh in DAYS:
        tr, te = Dr != dh, Dr == dh
        models[dh] = HistGradientBoostingClassifier(random_state=0, **HGB).fit(Xr[tr], yr[tr])
        pr[te] = models[dh].predict_proba(Xr[te])[:, 1]
    pf = np.full(len(F), np.nan)
    for dh in DAYS:
        m = X.day.to_numpy()[F] == dh
        if m.any():
            pf[m] = models[dh].predict_proba(X.loc[F[m], inputs])[:, 1]
    sr = s[ra]
    filt = dict(auc_all_inputs=auc(pr, yr), auc_public_score=auc(sr, yr), random_move_rate=float(yr.mean()), fire_move_rate=float(y[F].mean()), top={})
    for q in (.01, .001):
        r_ = {}
        for nm, sc, fsc in (('all_inputs', pr, pf), ('public_score', sr, s[F])):
            sel, fire_above = np.zeros(len(ra), bool), []
            for dh in DAYS:
                m = Dr == dh
                if not m.any():
                    continue
                th = np.nanquantile(sc[m], 1. - q)
                sel |= m & (sc >= th)
                fm = X.day.to_numpy()[F] == dh
                fire_above.extend((fsc[fm] >= th).tolist())
            r_[nm] = dict(precision=float(yr[sel].mean()), moments=int(sel.sum()), moves_caught_share=float(yr[sel].sum() / yr.sum()), fires_above=float(np.mean(fire_above)))
        filt['top'][f'{q * 100:g}%'] = r_
    out['filter'] = filt
    print('filter', filt, flush=True)
    C.require(TM.F.rss_mib() < TM.F.RSS_GUARD_MIB, 'RSS over the self-guard')
    # 6 imitation: fire-likeness on the held-out day's random moments
    fl = np.full(len(ra), np.nan)
    for dh in DAYS:
        te = Dr == dh
        if te.any():
            fl[te] = imit_models[dh].predict_proba(Xr[te])[:, 1]
    imi = dict(auc_fire_likeness_for_moves=auc(fl, yr), top={}, within_score_top5={})
    for q in (.01, .001):
        sel = np.zeros(len(ra), bool)
        for dh in DAYS:
            m = Dr == dh
            if m.any():
                sel |= m & (fl >= np.nanquantile(fl[m], 1. - q))
        imi['top'][f'{q * 100:g}%'] = dict(precision=float(yr[sel].mean()), moments=int(sel.sum()))
    t5 = np.zeros(len(ra), bool)
    for dh in DAYS:
        m = Dr == dh
        if m.any():
            t5 |= m & (sr >= np.nanquantile(sr[m], .95))
    for nm, sc in (('by_fire_likeness', fl), ('by_public_score', sr)):
        sel = np.zeros(len(ra), bool)
        for dh in DAYS:
            m = t5 & (Dr == dh)
            if m.any():
                sel |= m & (sc >= np.nanquantile(sc[m], .8))
        imi['within_score_top5'][nm] = dict(precision=float(yr[sel].mean()), moments=int(sel.sum()))
    imi['within_score_top5']['all'] = dict(precision=float(yr[t5].mean()), moments=int(t5.sum()))
    out['imitation'] = imi
    print('imitation', imi, flush=True)
    # 5 examples
    Z = robust_z(X, KEY_INPUTS, X.kind == 'random')
    fy = F[(y[F] == 1) & np.isfinite(Z[F]).all(1)]
    med = np.median(Z[fy], axis=0)
    typ = int(fy[np.argmin(((Z[fy] - med) ** 2).sum(1))])
    k_ = int(np.flatnonzero(F == typ)[0])
    cands = [c for c in Ct[k_] if y[c] == 0 and np.isfinite(Z[c]).all()]
    if not cands:
        cands = [c for c in Ct[k_] if np.isfinite(Z[c]).all()]
    look = int(min(cands, key=lambda c: float(((Z[c] - Z[typ]) ** 2).sum())))
    r0 = ra[(yr == 0) & np.isfinite(Z[ra]).all(1)]
    rmed = np.median(Z[r0], axis=0)
    ordinary = int(r0[np.argmin(((Z[r0] - rmed) ** 2).sum(1))])
    ex = {}
    for nm, i in (('typical_fire', typ), ('lookalike_no_move', look), ('ordinary_moment', ordinary)):
        ex[nm] = dict(day=str(X.day.iat[i]), t=float(X.t.iat[i]), d=float(X.d.iat[i]), y=float(y[i]), score=float(s[i]),
                      key_inputs={c: float(X[c].iat[i]) for c in KEY_INPUTS}, key_z_distance_to_fire=float(((Z[i] - Z[typ]) ** 2).sum() ** .5))
    out['examples'] = ex
    TR.dump(fn, out)
    print(json.dumps(ex, indent=1), flush=True)


# ---------------------------------------------------------------------------------------------------------------- examples
def window(day, t, d, lo_ms=-1000., hi_ms=300.):
    """Best level and trades around t (Binance E clock) -> dict of arrays relative to t, 'their way' signs."""
    day0 = WT.OD_day_ms(day)
    hrs = sorted({int((t + lo_ms - 2000. - day0) // 3600000), int((t + hi_ms - day0) // 3600000)})
    bk = []
    for h in hrs:
        fnb = WT.RAW / f'bookTicker/BTCUSDT/{day}_{h:02d}.csv.gz'
        C.require(fnb.exists(), f'no book file for hour {h}')
        bk.append(pd.read_csv(fnb, header=None, usecols=[1, 4, 5, 6, 7], names=['E', 'bid', 'bq', 'ask', 'aq']))
    b = pd.concat(bk).sort_values('E', kind='stable').drop_duplicates('E', keep='last')
    b = b[(b.E >= t + lo_ms - 2000.) & (b.E <= t + hi_ms)]
    tr = pd.concat([pd.read_csv(WT.RAW / f'trade/BTCUSDT/{day}_{h:02d}.csv.gz', header=None, usecols=[1, 3, 4, 5, 6], names=['E', 'id', 'p', 'q', 'm']) for h in hrs])
    tr = tr.drop_duplicates('id')
    tr = tr[(tr.E >= t + lo_ms) & (tr.E <= t + hi_ms) & (tr.p > 0) & (tr.q > 0)]
    up = d > 0
    E = b.E.to_numpy(float) - t
    far_p, far_q = (b.ask, b.aq) if up else (b.bid, b.bq)
    near_p, near_q = (b.bid, b.bq) if up else (b.ask, b.aq)
    mid = (b.bid.to_numpy(float) + b.ask.to_numpy(float)) / 2.
    j0 = int(np.searchsorted(E, 0., 'right') - 1)
    C.require(j0 >= 0, 'no book state at t')
    m0 = mid[j0]
    after = np.flatnonzero((E > 0) & (d * (mid - m0) > 0))
    move_ms = float(E[after[0]]) if len(after) else None
    tE = tr.E.to_numpy(float) - t
    its = np.where(tr.m.to_numpy() == DC.BUY_AGGR, 1., -1.) * d          # +1 = an aggressive order their way (a buy for an up bet)
    return dict(t_rel_ms=E.tolist(), far_size=far_q.to_numpy(float).tolist(), near_size=near_q.to_numpy(float).tolist(),
                mid_bps=(1e4 * np.log(mid / m0)).tolist(), far_price=far_p.to_numpy(float).tolist(), near_price=near_p.to_numpy(float).tolist(),
                trades_t_rel_ms=tE.tolist(), trades_btc=tr.q.to_numpy(float).tolist(), trades_their_way=its.tolist(), first_move_their_way_ms=move_ms)


def examples(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    src = Path(od) if od else H
    fn = src / 'signal_vs_noise_examples.json'
    C.require(not fn.exists(), 'signal_vs_noise_examples.json exists (write-once)')
    ex = json.loads((src / 'signal_vs_noise.json').read_text())['examples']
    out = {}
    for nm, e in ex.items():
        C.require(e['day'] in DAYS, 'an undeclared day')
        w = window(e['day'], e['t'], e['d'])
        out[nm] = dict(e, **w)
        print(nm, e['day'], 'y', e['y'], 'first move', w['first_move_their_way_ms'], 'book rows', len(w['t_rel_ms']), 'trades', len(w['trades_t_rel_ms']), flush=True)
    TR.dump(fn, out)


def selftest():
    s = np.array([1., 2., 3., 4.]); lab = np.array([0, 0, 1, 1])
    assert auc(s, lab) == 1. and auc(-s, lab) == 0. and abs(auc(np.array([1., 1., 1., 1.]), lab) - .5) < 1e-12
    assert np.isnan(auc(s, np.zeros(4)))
    assert abs(hanley_se(.5, 100, 100) - np.sqrt(((.5 * .5) + 99 * (1 / 3 - .25) + 99 * (1 / 3 - .25)) / 1e4)) < 1e-12
    X = pd.DataFrame(dict(day=['a'] * 8, d=[1.] * 8, t=[0., 10., 20., 30., 40., 50., 60., 2e7], kind=['fire'] + ['random'] * 7,
                          s_gbm_v3=[.5, .1, .49, .52, .9, .51, .48, .5]))
    global K_CTRL
    k0, K_CTRL = K_CTRL, 3
    F, Ct = match(X)
    K_CTRL = k0
    assert F.tolist() == [0] and sorted(Ct[0].tolist()) == [2, 3, 5], Ct       # nearest scores inside the window; the far one (t = 2e7) excluded
    print('signal_vs_noise selftest PASS')


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'read':
        read(a[0] if a else None, float(a[1]) if len(a) > 1 else None)
    elif cmd == 'examples':
        examples(a[0] if a else None)
    else:
        dict(selftest=selftest)[cmd]()
