"""A fill-aware chase model: rank chase moments by what a take is worth after the race for the quotes, not by how far Predict will move
(USER 2026-11-27: "You mean we need to consider both how far it can move and the size to take, yea go ahead"; after OURS_VS_ITS: our move
ranking aims where every chaser races, 37 % of its optimistic fills vanish). Development evidence on the four consumed days, every market,
CHASE_TAKES' population, rows, labels and fill rules unchanged (its move-model selections reproduced: a check); declared before the run.
Targets per labelled tick, a 50-share IOC at the known best + s (s = 3, 10c) on the conservative book (A's levels at their size in the
next snapshot; rows without a next snapshot excluded, counted): the value per intended share at the mid 1 s after the match; the value
taken back after 1 s (CHASE_TAKES' exit: every shown level through the 172 ms hold; the rest held to settlement); the fill fraction; the
net value per filled share (filled rows only).
Models (public_score's pinned trees, leave-one-day-out, CHASE_TAKES' 97 inputs): DIRECT_s3 / DIRECT_s10 on the value at 1 s; DIRECT_TB_s3 on
the value taken back; DECOMP_s3 / DECOMP_s10 = predicted fill x predicted net per filled share ("how much we can take" x "how far it moves
net of what we pay"); MOVE = CHASE_TAKES' held-out move model (edge = predicted move - half spread - fee), the baseline.
Selections per score (one action per move, the first tick picked): top 10 % and top 1 % of ticks (threshold pooled over the held-out days,
as CHASE_TAKES); top 10 % with the threshold from the other days' predictions only; score > 0.
Readings per selection: value per intended share on both books at the known best + 0 / 3 / 10c and every level (1 s mid), taken back after
1 s and held to settlement (+ 3 / + 10c), fill, per filled share, days > 0, the share of the optimistic fill gone by the next snapshot,
time left, Predict's move, and (diagnostic) the share on moves the top taker chased within the fire-set markets. Paying (computed): > 0
pooled AND >= 3 of 4 days AND >= 150 actions. Null (declared, 200 draws): random actions of the same count, conservative book, + 3 / + 10c.
Comparison (computed): each fill-aware score against MOVE at the same selection size, conservative book at 1 s and taken back, pooled and
days better. Seen days: a positive here needs untouched days.
Outputs (write-once): fill_aware.json + OUT/fill_aware/predictions.parquet.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/fill_aware.py \
       selftest | run [<dir> <n rows>]
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
import chase_takes as CT  # noqa: E402

C, TR, TK, TE, TM, WT, F = CT.C, CT.TR, CT.TK, CT.TE, CT.TM, CT.WT, CT.F
OUT = WT.OUT / 'fill_aware'
DAYS, Q, fee = CT.DAYS, CT.Q, CT.fee
SL = (3., 10.)
N_NULL, MIN_ORDERS, SEED = 200, 150, 20261127
ORD = ('s0', 's3', 's10', 'any')


def fit_lodo(Xf, y, ok, day):
    pred = np.full(len(y), np.nan)
    for dh in DAYS:
        tr, te = ok & (day != dh), day == dh
        if te.sum() == 0 or tr.sum() == 0:
            continue
        pred[te] = HistGradientBoostingRegressor(random_state=0, **TM.HGB).fit(Xf[tr], y[tr]).predict(Xf[te])
    return pred


def thr_other_days(score, ok, day, q):
    """Per row: the (1 - q) quantile of the score over the other days' ok rows."""
    th = np.full(len(score), np.nan)
    for dh in DAYS:
        m = day == dh
        o = ok & (day != dh)
        if m.any() and o.any():
            th[m] = np.quantile(score[o], 1. - q)
    return th


def run(od=None, limit=None):
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'fill_aware.json'
    fpp = (od or OUT) / 'predictions.parquet'
    C.require(not fn.exists() and not fpp.exists(), 'an output exists (write-once)')
    X, feats, _ = CT.frame(CT.OUT)
    X = X.merge(pd.read_parquet(CT.OUT / 'predictions.parquet', columns=['day', 't', 'd', 'pred_1s']), on=['day', 't', 'd'], how='left', validate='1:1')
    if limit:                                                         # rehearsal: whole moves, seeded
        u = np.unique(X.key.to_numpy())
        keep = set(np.random.default_rng(SEED).choice(u, min(limit, len(u)), replace=False))
        X = X[X.key.isin(keep)].reset_index(drop=True)
    d, lim, md, day, mid = X.d.to_numpy(float), X.lim.to_numpy(float), X.md.to_numpy(float), X.day.to_numpy(), X.mid.to_numpy()
    key = X.key.to_numpy()
    cost = d * (lim - md) + fee(lim)
    mk = X.m1s.to_numpy(float)
    pay = X.payout.to_numpy(float)
    PA = X[[f'pA_{k}' for k in range(5)]].to_numpy(float); SA = X[[f'sA_{k}' for k in range(5)]].to_numpy(float)
    PB = X[[f'pB_{k}' for k in range(5)]].to_numpy(float); SB = X[[f'sB_{k}' for k in range(5)]].to_numpy(float)
    okB = np.isfinite(X.dtB.to_numpy(float))
    BK = {'A': (PA, SA), 'AB': TK.minbook(PA, SA, PB, SB)}
    fixed = {f's{int(s)}': lim + d * s for s in (0., 3., 10.)}
    fixed['any'] = np.where(d > 0, 99., 1.)
    W = {b: {k: TK.vwalk(BK[b][0], BK[b][1], d, L_) for k, L_ in fixed.items()} for b in BK}
    eP, eS = X[[f'eP1s_{k}' for k in range(5)]].to_numpy(float), X[[f'eS1s_{k}' for k in range(5)]].to_numpy(float)
    V1 = {b: {k: (d * (mk * W[b][k][0] - W[b][k][1]) - W[b][k][2]) / Q for k in fixed} for b in BK}
    TB = {b: {k: TE.exit_values(*W[b][k], d, eP, eS, pay)[0] / Q for k in ('s3', 's10')} for b in BK}
    HOLD = {b: {k: TE.exit_values(*W[b][k], d, eP, eS, pay)[1] / Q for k in ('s3', 's10')} for b in BK}
    Xf = X[feats].to_numpy(np.float32)
    out = dict(asof='2026-11-27', rehearsal=reh, rows=int(len(X)), rows_with_next_snapshot=int(okB.sum()), models={}, selections={}, null={}, comparison={})
    S = {}
    # ---- models
    S['MOVE'] = X.pred_1s.to_numpy(float) - cost
    for s in SL:
        k = f's{int(s)}'
        y = V1['AB'][k]
        S[f'DIRECT_{k}'] = fit_lodo(Xf, y, okB, day)
        fl = W['AB'][k][0] / Q
        fill_hat = np.clip(fit_lodo(Xf, fl, okB, day), 0., 1.)
        filled = okB & (W['AB'][k][0] > 0)
        net = np.where(filled, (d * (mk * W['AB'][k][0] - W['AB'][k][1]) - W['AB'][k][2]) / np.maximum(W['AB'][k][0], 1e-9), np.nan)
        net_hat = fit_lodo(Xf, net, filled, day)
        S[f'DECOMP_{k}'] = fill_hat * net_hat
        out['models'][f'DIRECT_{k}'] = dict(spearman=float(spearmanr(S[f'DIRECT_{k}'][okB], y[okB]).correlation), target_mean_c=float(y[okB].mean()))
        out['models'][f'DECOMP_{k}'] = dict(spearman=float(spearmanr(S[f'DECOMP_{k}'][okB], y[okB]).correlation), fill_spearman=float(spearmanr(fill_hat[okB], fl[okB]).correlation),
                                           net_spearman=float(spearmanr(net_hat[filled], net[filled]).correlation), filled_rows=int(filled.sum()), fill_mean=float(fl[okB].mean()))
        C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    ytb = TB['AB']['s3']
    S['DIRECT_TB_s3'] = fit_lodo(Xf, ytb, okB, day)
    out['models']['DIRECT_TB_s3'] = dict(spearman=float(spearmanr(S['DIRECT_TB_s3'][okB], ytb[okB]).correlation), target_mean_c=float(ytb[okB].mean()))
    out['models']['MOVE'] = dict(spearman_vs_AB_s10=float(spearmanr(S['MOVE'][okB], V1['AB']['s10'][okB]).correlation))
    # ---- selections
    tau = X.fv_tau_s.to_numpy(float)
    mo1 = d * (mk - md)
    A_ = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet', columns=['day', 'rank', 't_set', 'mid'])
    A_ = A_[(A_['rank'] == 1) & A_.t_set.notna()]
    fm = set(zip(A_.day, A_.mid.astype(np.int64)))
    in_fs = np.array([(dy, int(mi)) in fm for dy, mi in zip(day, mid)])
    chased = np.isfinite(X.tt_chase_lag_ms.to_numpy(float))
    fa_any, fb_any = W['A']['any'][0], W['AB']['any'][0]

    def cell(v, idx):
        n = len(idx)
        if n == 0:
            return dict(actions=0)
        by = {dh: float(v[idx][day[idx] == dh].mean()) for dh in DAYS if (day[idx] == dh).any()}
        dp = int(sum(x > 0 for x in by.values()))
        val = float(v[idx].mean())
        return dict(actions=int(n), value_c=val, by_day=by, days_positive=dp, paying=bool(val > 0 and dp >= 3 and n >= MIN_ORDERS))

    def evaluate(idx):
        o = dict(actions=int(len(idx)), markets=int(np.unique(mid[idx]).size), time_left_median_s=float(np.median(tau[idx])) if len(idx) else None,
                 predict_move_1s_c=float(mo1[idx].mean()) if len(idx) else None, gone_by_next_snapshot=float(1. - fb_any[idx].sum() / max(fa_any[idx].sum(), 1e-9)) if len(idx) else None,
                 on_its_chased_moves_in_fire_set=float(chased[idx][in_fs[idx]].mean()) if in_fs[idx].any() else None, books={})
        for b in BK:
            ii = idx[okB[idx]] if b == 'AB' else idx
            o['books'][b] = {k: dict(cell(V1[b][k], ii), fill=float(W[b][k][0][ii].sum() / (Q * max(len(ii), 1)))) for k in ORD}
            for k in ('s3', 's10'):
                o['books'][b][f'{k}_take_back_1s'] = cell(TB[b][k], ii)
                o['books'][b][f'{k}_hold_to_settlement'] = cell(HOLD[b][k], ii)
        return o
    okv = np.isfinite(S['MOVE'])
    sels = {}
    for nm, sc in S.items():
        ok = okv & np.isfinite(sc)
        for q, lab in ((.1, 'top10'), (.01, 'top1')):
            thr = np.quantile(sc[ok], 1. - q)
            sels[f'{nm}_{lab}'] = CT.first_per_move(ok & (sc >= thr), key)
        th = thr_other_days(sc, ok, day, .1)
        sels[f'{nm}_top10_otherdays'] = CT.first_per_move(ok & (sc >= th), key)
        sels[f'{nm}_positive'] = CT.first_per_move(ok & (sc > 0), key)
    if not reh:
        CJ = json.loads((H / 'chase_takes.json').read_text())
        C.require(len(sels['MOVE_top10']) == CJ['selections']['model1s_top10']['actions'] and len(sels['MOVE_top1']) == CJ['selections']['model1s_top1']['actions'],
                  'CHASE_TAKES\' move-model selections not reproduced')
    for nm, idx in sels.items():
        out['selections'][nm] = evaluate(idx)
        C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    # ---- null: random actions of the same count, conservative book
    rng = np.random.default_rng(SEED)
    groups = CT.move_groups(key, okB)
    for nm in [x for x in sels if x.endswith('_top10')]:
        n = int(okB[sels[nm]].sum())
        if n == 0:
            continue
        dr = {k: [] for k in ('s3', 's10')}
        for _ in range(N_NULL if not reh else 20):
            ia = CT.random_actions(groups, n, rng)
            for k in dr:
                dr[k].append(float(V1['AB'][k][ia].mean()))
        out['null'][nm] = {k: dict(mean_c=float(np.mean(v)), p97_5_c=float(np.quantile(v, .975)), selection_c=out['selections'][nm]['books']['AB'][k]['value_c']) for k, v in dr.items()}
    # ---- comparison with MOVE at the same selection size
    for nm in sels:
        base = nm.split('_', 1)[0] if nm.startswith('MOVE') else None
        if nm.startswith('MOVE'):
            continue
        size = nm[len(nm.rsplit('_', 1)[0]) + 1:] if not nm.endswith('otherdays') else 'top10_otherdays'
        mv = f'MOVE_{size}'
        cmp_ = {}
        for k in ('s3', 's10'):
            for what in ('', '_take_back_1s'):
                a_, b_ = out['selections'][nm]['books']['AB'][f'{k}{what}' if what else k], out['selections'][mv]['books']['AB'][f'{k}{what}' if what else k]
                if a_.get('actions') and b_.get('actions'):
                    better = sum(1 for dh in DAYS if dh in a_['by_day'] and dh in b_['by_day'] and a_['by_day'][dh] > b_['by_day'][dh])
                    cmp_[f'{k}{what or "_1s"}'] = dict(fill_aware_c=a_['value_c'], move_c=b_['value_c'], diff_c=a_['value_c'] - b_['value_c'], days_better=better)
        out['comparison'][nm] = dict(against=mv, **cmp_)
    cells = [(nm, b, k, c) for nm, s in out['selections'].items() for b, ex in s['books'].items() for k, c in ex.items() if isinstance(c, dict) and c.get('actions')]
    fa = [x for x in cells if not x[0].startswith('MOVE')]
    out['verdict'] = dict(cells=len(cells), fill_aware_cells=len(fa),
                          paying_conservative=[dict(selection=a, order=k, value_c=c['value_c'], actions=c['actions'], days_positive=c['days_positive']) for a, b, k, c in fa if b == 'AB' and c['paying']],
                          paying_conservative_move=[dict(selection=a, order=k, value_c=c['value_c'], actions=c['actions'], days_positive=c['days_positive']) for a, b, k, c in cells
                                                    if a.startswith('MOVE') and b == 'AB' and c['paying']],
                          best_conservative_take_back=max(((a, k, c['value_c'], c['days_positive'], c['actions']) for a, b, k, c in cells if b == 'AB' and 'take_back' in k and c['actions'] >= MIN_ORDERS),
                                                          key=lambda x: x[2]),
                          beats_move_everywhere={nm: all(v['diff_c'] > 0 for k, v in c.items() if isinstance(v, dict)) for nm, c in out['comparison'].items()})
    (od or OUT).mkdir(parents=True, exist_ok=True)
    P = X[['day', 't', 'd', 'key']].copy()
    for nm, sc in S.items():
        P[nm] = sc
    P.to_parquet(fpp, index=False)
    TR.dump(fn, out)
    print(json.dumps(dict(models=out['models'], verdict=out['verdict']), indent=1, default=str)[:5000])
    for nm, s in out['selections'].items():
        ab = s['books']['AB']
        print(nm, s['actions'], 'left', round(s['time_left_median_s'] or 0), 'gone', round(s['gone_by_next_snapshot'] or 0, 3), 'AB 1s',
              {k: round(ab[k].get('value_c', np.nan), 2) for k in ORD}, 'TB', {k: round(ab[f'{k}_take_back_1s'].get('value_c', np.nan), 2) for k in ('s3', 's10')})


# ---------------------------------------------------------------------------------------------------------------- selftest
def selftest():
    rng = np.random.default_rng(0)
    n = 4000
    day = np.array(DAYS)[rng.integers(0, 4, n)]
    Xf = rng.normal(size=(n, 3)).astype(np.float32)
    y = 2. * Xf[:, 0] + rng.normal(scale=.5, size=n)
    p = fit_lodo(Xf, y, np.ones(n, bool), day)
    assert np.isfinite(p).all() and spearmanr(p, y).correlation > .8                      # learns a plain signal out of day
    yb = y.copy(); yb[day == DAYS[0]] = 1e6                                               # a held-out day's labels never reach its own fit
    p2 = fit_lodo(Xf, yb, np.ones(n, bool), day)
    assert np.allclose(p2[day == DAYS[0]], p[day == DAYS[0]]), 'a held-out day leaked into its fit'
    sc = np.arange(8.); okk = np.ones(8, bool); dd = np.array([DAYS[0]] * 4 + [DAYS[1]] * 4)
    th = thr_other_days(sc, okk, dd, .5)
    assert th[0] == np.quantile(sc[4:], .5) and th[4] == np.quantile(sc[:4], .5), th
    try:
        run('/nonexistent/x', None) if False else C.require(False, 'refused')
    except ValueError as e:
        assert 'refused' in str(e)
    print('fill_aware selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    if cmd == 'run':
        run(a_[0] if a_ else None, int(a_[1]) if len(a_) > 1 else None)
    else:
        dict(selftest=selftest)[cmd]()
