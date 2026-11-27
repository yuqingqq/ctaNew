"""The markout price as the target, and slippage to get the take filled (USER 2026-11-27: "1. We can get slippage to fill the taker order,
2. The training target is the mark out price"). Development evidence on the four consumed days (all seen): leave-one-day-out, no strategy
freeze, no orders. Declared before the run.
Population: TAKE_MODEL's labelled moments (public_score3's random moments, both directions, complete markets), the same rows, recomputed
from the same paths and checked against TAKE_MODEL's labels (the known mid, fills, values and moves at 152 / 172 ms; any difference fails).
Labels (stage labels), per moment and total latency T = 152 / 172 ms: the venue book on the side taken at the match (5 shown levels) from
the last snapshot at or before the match (A, TAKE_MODEL's rule) and the first after it (B); the Predict mid 1 s / 5 s / 30 s after the
match (a snapshot <= 1 s old inside the recorded path, else a counted status).
Target (the USER's): the markout price, as its move from the decision mid the way taken, mo = d x (mid after h - mid at the decision);
primary h = 1 s at T = 172 (TAKE_MODEL's horizon; STALE_PAID_SPLIT's for the top taker); also 5 s and 30 s at 172, 1 s at 152. Model:
public_score's pinned trees on TAKE_MODEL's 79 features, leave-one-day-out; in-sample (fitted and scored on all days) for contrast.
Predicted edge at the known best: e = predicted mo - (known best - decision mid, its way) - fee(known best); selections rank by e.
Orders: a 50-share IOC on the side taken, limit = the known best + s c (s = 0, 1, 2, 3, 5, 10, 20), every shown level ('any'), or adaptive:
the furthest whole-cent price at which the predicted markout still pays the fee and a margin (0 / 1 c), no order when that is short of
the known best. Fill: TAKE_MODEL's walk of book A; pessimistic: of A and B, each of A's levels at the smaller size (a quote B does not
show counts as gone before the match). Per filled share: value = markout from the known best - slippage above it - fee (2% x min(p,
100 - p) per level); per intended share with misses 0.
Selections: the top 1 / 0.3 / 0.1 % by e (pooled), every moment with e > 0, the top 1 % by predicted move (TAKE_MODEL_WHY's, reproduced
there: a check); in-sample top 1 %; the oracle (the realised markout in place of the prediction); the Binance-move score's top 1 %;
random top 1 % (200 draws: mean and 97.5th percentile, fixed limits).
Paying (computed): value per intended share > 0 pooled AND on >= 3 of 4 days AND >= 150 orders. Point estimates (G = 4 days).
Outputs (write-once): OUT/take_markout/<day>_labels.parquet + .json (labels), take_markout.json + OUT/take_markout/predictions.parquet (train).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/take_markout.py \
       selftest | labels <day> [<dir> <n markets>] | train [<dir>]
"""
import collections
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
import take_model as TM  # noqa: E402
import take_model_insample as TI  # noqa: E402

VR, MM, TR, C, F, WT, SRy = TM.VR, TM.MM, TM.TR, TM.C, TM.F, TM.WT, TM.SRy
OUT = WT.OUT / 'take_markout'
DAYS = TM.DAYS
Q = TM.Q
NETS = (2., 22.)
HZ = (1000., 5000., 30000.)
SLIPS = (0., 1., 2., 3., 5., 10., 20.)
MARGINS = (0., 1.)
QTOP = (.01, .003, .001)
N_NULL, MIN_ORDERS, SEED = 200, 150, 20261127
PRIMARY = '172_1s'
SECONDARY = ('172_5s', '172_30s', '152_1s')
fee = TM.fee


# ---------------------------------------------------------------------------------------------------------------- labels
def side_levels(row, side):
    b = 2 + 10 * side
    return row[b:b + 5].copy(), row[b + 5:b + 10].copy()


def parity_row(r0, got):
    """TAKE_MODEL's label fields against the recomputed ones -> the names that differ."""
    bad = []
    for k, v in got.items():
        tol = 1e-12 if k.startswith('fill_') else 1e-9
        if not abs(float(v) - float(r0[k])) <= tol:
            bad.append(k)
    return bad


def labels(day, od=None, limit=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}_labels.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    L0 = pd.read_parquet(TM.OUT / f'{day}_labels.parquet').set_index('row')
    Z = pd.read_parquet(WT.OUT / 'public_score3_points.parquet', columns=['day', 't', 'd', 'kind'])
    Z['row'] = np.arange(len(Z))
    Z = Z[Z.row.isin(set(L0.index))].reset_index(drop=True)
    C.require(len(Z) == len(L0) and (Z.day == day).all() and (Z.kind == 'random').all(), 'the population differs from TAKE_MODEL')
    reg = MM.registry(day).set_index('mid')
    ws_to_mid = {int(float(reg.loc[m, 'ws'])): int(m) for m in reg.index}
    Z['mid'] = (np.floor(Z.t / 300000.) * 300).astype(int).map(ws_to_mid)
    C.require((Z.mid.to_numpy() == L0.loc[Z.row, 'mid'].to_numpy()).all(), 'markets differ from TAKE_MODEL')
    by_mid = Z.groupby('mid').indices
    rows, status, parity = [], collections.Counter(), collections.Counter()
    src = MM.KP.Tape(); off = src.offset(day); n_m = 0
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in by_mid]
        if limit is not None:
            mids = mids[:max(0, limit - n_m)]
        if not mids:
            continue
        for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
            C.require(lst == 'loaded', f'{m} {lst}')
            ws = float(reg.loc[m, 'ws']) * 1000.
            p, st, info = VR.vm_path(W, ws, 50.)
            C.require(st == 'complete', f'{m} {st}')
            n_m += 1
            t0, end = float(info['t_start']), float(info['end_rel'])
            f = np.asarray(p['frames'], float)
            for i in by_mid[m]:
                t, d, rw = float(Z.t.iat[i]), float(Z.d.iat[i]), int(Z.row.iat[i])
                r0 = L0.loc[rw]
                g_ = t + TM.FEED + TM.DECIDE - t0
                rk = f[int(np.searchsorted(f[:, 1], g_, 'right') - 1)]
                bid, ask = rk[2], rk[12]
                side = 1 if d > 0 else 0
                lim = ask if d > 0 else bid
                md = (bid + ask) / 2.
                r = dict(row=rw, day=day, mid=m, t=t, d=d, lim=lim, md=md)
                got = dict(pf_mid=md)
                for net in NETS:
                    T = int(net + TM.HOLD)
                    act = g_ + net + TM.HOLD
                    jv = int(np.searchsorted(f[:, 0], act, 'right') - 1)
                    pa, sa = side_levels(f[jv], side)
                    r[f'dtA{T}'] = act - f[jv, 0]
                    if jv + 1 < len(f):
                        pb, sb = side_levels(f[jv + 1], side)
                        r[f'dtB{T}'] = f[jv + 1, 0] - act
                    else:
                        pb, sb = np.full(5, -1.), np.zeros(5)
                        r[f'dtB{T}'] = np.nan
                        status[f'no_frame_after_act_{T}'] += 1
                    for k in range(5):
                        r[f'pA{T}_{k}'], r[f'sA{T}_{k}'], r[f'pB{T}_{k}'], r[f'sB{T}_{k}'] = pa[k], sa[k], pb[k], sb[k]
                    for hz in HZ:
                        x = act + hz
                        ja = int(np.searchsorted(f[:, 0], x, 'right') - 1)
                        mk = TM.mid_of(f[ja]) if (x <= end and ja >= 0 and x - f[ja, 0] <= 1000.) else np.nan
                        if not np.isfinite(mk):
                            status[f'no_mid_{T}_{int(hz / 1000)}s'] += 1
                        r[f'm{int(hz / 1000)}s_{T}'] = mk
                    ma = r[f'm1s_{T}']
                    for kk in ((0., TM.PAY_K) if net == max(NETS) else (0.,)):
                        q, vw = TM.walk(f[jv], side, d, lim + (kk if d > 0 else -kk), Q)
                        got[f'v_{T}_{int(kk)}'] = q * (d * (ma - vw) - fee(vw)) / Q if q > 0 else 0.
                        got[f'fill_{T}_{int(kk)}'] = q / Q
                    got[f'mv_{T}'] = d * (ma - md)
                parity['rows'] += 1
                for k in parity_row(r0, got):
                    parity[f'{k}_differs'] += 1
                status['ok'] += 1
                rows.append(r)
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    L = pd.DataFrame(rows)
    if limit is None:
        C.require(len(L) == len(L0), 'rows lost against TAKE_MODEL')
    bad = {k: v for k, v in parity.items() if k.endswith('_differs')}
    C.require(not bad, f'parity with TAKE_MODEL failed: {bad}')
    od.mkdir(parents=True, exist_ok=True)
    L.to_parquet(fp, index=False)
    TR.dump(od / f'{day}_labels.json', dict(asof='2026-11-27', day=day, markets=n_m, status=dict(status), parity=dict(parity),
                                             dtA_172_ms_q=[float(x) for x in np.quantile(L.dtA172, [.5, .9])],
                                             dtB_172_ms_q=[float(x) for x in np.nanquantile(L.dtB172, [.5, .9])]))
    print(day, n_m, dict(status), dict(parity), flush=True)


# ---------------------------------------------------------------------------------------------------------------- orders
def vwalk(P, S, d, lim, q=Q):
    """TAKE_MODEL's walk over rows of 5 levels on the side taken (prices P, sizes S, best first), direction d (+1 buys the asks, -1 sells
    to the bids), limit lim -> (filled, cost, fee), the fee per level."""
    P, S = np.asarray(P, float), np.asarray(S, float)
    dd, ll = np.asarray(d, float)[:, None], np.asarray(lim, float)[:, None]
    ok = (P >= 0) & (S > 0) & np.where(dd > 0, P <= ll + 1e-9, P >= ll - 1e-9)
    ok = np.logical_and.accumulate(ok, axis=1)
    avail = np.where(ok, S, 0.)
    take = np.clip(np.minimum(avail, q - (np.cumsum(avail, axis=1) - avail)), 0., None)
    Pz = np.where(ok, P, 0.)
    return take.sum(1), (take * Pz).sum(1), (take * fee(Pz)).sum(1)


def minbook(PA, SA, PB, SB):
    """Of snapshots A and B: each of A's levels at the smaller size (0 when B does not show the price); emptied levels removed in order."""
    hit = (PB[:, None, :] == PA[:, :, None]) & (PA[:, :, None] >= 0) & (SB[:, None, :] > 0)
    s = np.where(PA >= 0, np.minimum(SA, np.where(hit, SB[:, None, :], 0.).sum(2)), 0.)
    o = np.argsort(s <= 0, axis=1, kind='stable')
    P2, S2 = np.take_along_axis(PA, o, 1), np.take_along_axis(s, o, 1)
    return np.where(S2 > 0, P2, -1.), S2


def adaptive_limit(md, d, mhat, margin):
    """The furthest whole-cent price at which the predicted markout price md + d x mhat still pays the fee and the margin:
    a buy P + fee(P) <= M - margin, a sell P - fee(P) >= M + margin."""
    M = np.asarray(md, float) + np.asarray(d, float) * np.asarray(mhat, float)
    X = np.where(d > 0, M - margin, M + margin)
    Pb = np.where(X <= 51., X / 1.02, (X - 2.) / .98)
    Ps = np.where(X <= 49., X / .98, (X + 2.) / 1.02)
    return np.clip(np.where(d > 0, np.floor(Pb + 1e-9), np.ceil(Ps - 1e-9)), 1., 99.)


def cell(sel, fire, w, d, mk, lim, day, mid):
    """One selection under one order rule: sel = selected rows, fire = of those, the ones that send an order, w = (filled, cost, fee) per row."""
    i = sel[fire]
    n = int(len(i))
    out = dict(selected=int(len(sel)), orders=n)
    if n == 0:
        return out
    fl, co, fe = w[0][i], w[1][i], w[2][i]
    dd, m_, l_ = d[i], mk[i], lim[i]
    val = (dd * (m_ * fl - co) - fe) / Q
    Fs = float(fl.sum())
    out.update(fill=Fs / (Q * n), value_per_intended_c=float(val.mean()), total_usd=float(val.sum() * Q / 100.), markets=int(np.unique(mid[i]).size))
    if Fs > 0:
        mv = float((dd * (m_ - l_) * fl).sum() / Fs); sl = float((dd * (co - l_ * fl)).sum() / Fs); fz = float(fe.sum() / Fs)
        C.require(abs((mv - sl - fz) * Fs / (Q * n) - val.mean()) < 1e-6, 'move - slippage - fee does not add up to the value')
        out.update(per_filled_move_c=mv, per_filled_slippage_c=sl, per_filled_fee_c=fz, per_filled_value_c=mv - sl - fz)
    by = {}
    for dh in DAYS:
        mm = day[i] == dh
        by[dh] = dict(orders=int(mm.sum()), value_per_intended_c=float(val[mm].mean()) if mm.any() else None)
    out['by_day'] = by
    out['days_positive'] = int(sum(1 for v in by.values() if v['value_per_intended_c'] is not None and v['value_per_intended_c'] > 0))
    out['paying'] = bool(out['value_per_intended_c'] > 0 and out['days_positive'] >= 3 and n >= MIN_ORDERS)
    return out


# ---------------------------------------------------------------------------------------------------------------- train
def top(score, ok, q):
    i = np.flatnonzero(ok)
    o = np.argsort(-score[i], kind='stable')
    return i[o[:max(1, int(round(q * len(i))))]]


def fit_lodo(X, feats, y, ok):
    pred = np.full(len(X), np.nan)
    for dh in DAYS:
        tr, te = ok & (X.day != dh).to_numpy(), (X.day == dh).to_numpy()
        if te.sum() == 0 or tr.sum() == 0:
            continue
        pred[te] = HistGradientBoostingRegressor(random_state=0, **TM.HGB).fit(X.loc[tr, feats], y[tr]).predict(X.loc[te, feats])
    return pred


def train(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'take_markout.json'
    fpp = (od or OUT) / 'predictions.parquet'
    C.require(not fn.exists() and not fpp.exists(), 'an output exists (write-once)')
    X, feats = TI.frame()
    K = pd.concat([pd.read_parquet((od or OUT) / f'{d}_labels.parquet') for d in DAYS if ((od or OUT) / f'{d}_labels.parquet').exists()], ignore_index=True)
    C.require(len(K) and K.row.is_unique, 'labels missing or duplicated')
    if not reh:
        C.require(len(K) == len(X) and set(K.row) == set(X.row), 'the labels do not cover TAKE_MODEL\'s population')
    X = X[X.row.isin(set(K.row))].reset_index(drop=True)
    X = X.merge(K.drop(columns=['day', 'mid', 't', 'd']), on='row', how='left', validate='1:1')
    C.require(np.allclose(X.md.to_numpy(), X.pf_mid.to_numpy(), rtol=0, atol=1e-12), 'labels and features do not align')
    C.require(np.allclose(X.d.to_numpy() * (X.m1s_172 - X.md).to_numpy(), X.mv_172.to_numpy(), rtol=0, atol=1e-9), 'the 1 s markout differs from TAKE_MODEL')
    d, lim, md = X.d.to_numpy(float), X.lim.to_numpy(float), X.md.to_numpy(float)
    day, mid = X.day.to_numpy(), X.mid.to_numpy()
    cost_known = d * (lim - md) + fee(lim)                    # half the spread and the fee at the known best
    rng = np.random.default_rng(SEED)
    statuses = {dh: json.loads(((od or OUT) / f'{dh}_labels.json').read_text()) for dh in DAYS if ((od or OUT) / f'{dh}_labels.json').exists()}
    out = dict(asof='2026-11-27', rows=int(len(X)), label_status={k: v['status'] for k, v in statuses.items()},
               label_parity={k: v['parity'] for k, v in statuses.items()},
               snapshot_age_ms={k: dict(A_before_match_p50_p90=v['dtA_172_ms_q'], B_after_match_p50_p90=v['dtB_172_ms_q']) for k, v in statuses.items()},
               targets={})
    P = X[['row', 'day']].copy()

    def books(T):
        PA = X[[f'pA{T}_{k}' for k in range(5)]].to_numpy(float); SA = X[[f'sA{T}_{k}' for k in range(5)]].to_numpy(float)
        PB = X[[f'pB{T}_{k}' for k in range(5)]].to_numpy(float); SB = X[[f'sB{T}_{k}' for k in range(5)]].to_numpy(float)
        okB = np.isfinite(X[f'dtB{T}'].to_numpy(float))
        return {'A': (PA, SA, np.ones(len(X), bool)), 'AB': minbook(PA, SA, PB, SB) + (okB,)}

    for tg in (PRIMARY,) + SECONDARY:
        T, hz = int(tg.split('_')[0]), tg.split('_')[1]
        mk = X[f'm{hz}_{T}'].to_numpy(float)
        ok = np.isfinite(mk)
        mo = np.where(ok, d * (mk - md), np.nan)
        full = tg == PRIMARY
        BK = books(T) if full else {'A': books(T)['A']}
        slips = SLIPS if full else (0., 3., 10.)
        margins = MARGINS if full else (0.,)
        fixed = {f's{int(s)}': lim + d * s for s in slips}
        fixed['any'] = np.where(d > 0, 99., 1.)
        W = {b: {k: vwalk(PA, SA, d, L) for k, L in fixed.items()} for b, (PA, SA, _) in BK.items()}
        pl = fit_lodo(X, feats, mo, ok)
        P[f'pred_{tg}'] = pl
        o = dict(n=int(ok.sum()), excluded_no_markout=int((~ok).sum()), spearman_lodo=float(spearmanr(pl[ok], mo[ok]).correlation),
                 move_all_c=float(np.mean(mo[ok])), edge_at_known_best_all_c=float(np.mean(mo[ok] - cost_known[ok])), selections={}, random_null={})
        preds = dict(model=pl)
        if full:
            pin = HistGradientBoostingRegressor(random_state=0, **TM.HGB).fit(X.loc[ok, feats], mo[ok]).predict(X[feats])
            P[f'pred_{tg}_insample'] = pin
            o['spearman_insample'] = float(spearmanr(pin[ok], mo[ok]).correlation)
            preds['insample'] = pin
        preds['oracle'] = np.where(ok, mo, np.nan)
        sels = {}
        for q in (QTOP if full else (.01,)):
            sels[f'model_top{q * 100:g}'] = ('model', top(pl - cost_known, ok, q))
        sels['model_e_positive'] = ('model', np.flatnonzero(ok & (pl - cost_known > 0)))
        if full:
            sels['model_top1_by_move'] = ('model', top(pl, ok, .01))
            sels['insample_top1'] = ('insample', top(preds['insample'] - cost_known, ok, .01))
            sels['binance_score_top1'] = ('model', top(X.s_gbm_v3.to_numpy(float), ok, .01))
        sels['oracle_top1'] = ('oracle', top(mo - cost_known, ok, .01))
        for nm, (src, idx) in sels.items():
            ph = preds[src]
            s = dict(n=int(len(idx)), realised_move_c=float(np.mean(mo[idx])) if len(idx) else None,
                     realised_edge_at_known_best_c=float(np.mean(mo[idx] - cost_known[idx])) if len(idx) else None,
                     predicted_move_c=float(np.mean(ph[idx])) if len(idx) else None, orders={})
            for b, (PA, SA, okb) in BK.items():
                sb = idx[okb[idx]]
                s['orders'][b] = {}
                s['orders'][b]['excluded_no_snapshot_after'] = int(len(idx) - len(sb))
                for k in fixed:
                    s['orders'][b][k] = cell(sb, np.ones(len(sb), bool), W[b][k], d, mk, lim, day, mid)
                for mg in margins:
                    al = adaptive_limit(md, d, ph, mg)
                    wa = vwalk(PA[sb], SA[sb], d[sb], al[sb])
                    wfull = tuple(np.zeros(len(X)) for _ in range(3))
                    for a, z in zip(wfull, wa):
                        a[sb] = z
                    fire = np.where(d[sb] > 0, al[sb] >= lim[sb], al[sb] <= lim[sb])
                    c = cell(sb, fire, wfull, d, mk, lim, day, mid)
                    c['limit_beyond_known_best_c_median'] = float(np.median((d * (al - lim))[sb][fire])) if fire.any() else None
                    s['orders'][b][f'adaptive_m{int(mg)}'] = c
            if full and nm in ('model_top1', 'model_top1_by_move', 'oracle_top1'):
                PA = BK['A'][0]
                there = np.where(d > 0, (PA[:, 0] >= 0) & (PA[:, 0] <= lim), (PA[:, 0] >= 0) & (PA[:, 0] >= lim))[idx]
                s['known_best_at_match'] = {}
                for gnm, gm in (('still_shown', there), ('gone', ~there)):
                    gi = idx[gm]
                    s['known_best_at_match'][gnm] = dict(share=float(gm.mean()), realised_move_c=float(np.mean(mo[gi])) if len(gi) else None,
                                                         s0=cell(gi, np.ones(len(gi), bool), W['A']['s0'], d, mk, lim, day, mid),
                                                         any=cell(gi, np.ones(len(gi), bool), W['A']['any'], d, mk, lim, day, mid))
            o['selections'][nm] = s
        k1 = len(sels['model_top1'][1])
        for b, (PA, SA, okb) in BK.items():
            o['random_null'][b] = {}
            pool = np.flatnonzero(ok & okb)
            for k in fixed:
                fl, co, fe = W[b][k]
                v = (d * (mk * fl - co) - fe) / Q
                draws = np.array([v[rng.choice(pool, k1, replace=False)].mean() for _ in range(N_NULL)])
                o['random_null'][b][k] = dict(draws=N_NULL, k=int(k1), mean_c=float(draws.mean()), p97_5_c=float(np.quantile(draws, .975)))
        out['targets'][tg] = o
        print(tg, 'rho', round(o['spearman_lodo'], 3), {nm: {b: {k: round(c.get('value_per_intended_c', np.nan), 3) for k, c in s['orders'][b].items() if isinstance(c, dict)}
                                                            for b in s['orders']} for nm, s in o['selections'].items() if nm in ('model_top1', 'oracle_top1')}, flush=True)
        C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    # the reconciliation with TAKE_MODEL_WHY: the same model of the same move, its top 1 % at the known best
    pr = out['targets'][PRIMARY]['selections']['model_top1_by_move']
    ref = json.loads((H / 'take_model_why.json').read_text())['components']['top1_by_predicted_move']
    mine = dict(realised_move_c=pr['realised_move_c'], fill=pr['orders']['A']['s0']['fill'])
    out['reconciliation_take_model_why'] = dict(ref=dict(realised_move_c=ref['realised_move_c'], fill=ref['fill']), here=mine)
    if not reh:
        C.require(abs(mine['realised_move_c'] - ref['realised_move_c']) < 1e-9 and abs(mine['fill'] - ref['fill']) < 1e-9, f'TAKE_MODEL_WHY not reproduced: {mine} vs {ref}')
    # verdict (computed)
    po = out['targets'][PRIMARY]
    held = [(nm, b, k, c) for nm in ('model_top1', 'model_top0.3', 'model_top0.1', 'model_e_positive') for b, ex in po['selections'][nm]['orders'].items()
            for k, c in ex.items() if isinstance(c, dict) and c.get('orders')]
    big = [x for x in held if x[3]['orders'] >= MIN_ORDERS]
    best = max(big, key=lambda x: x[3]['value_per_intended_c']) if big else None
    orc = [(b, k, c) for b, ex in po['selections']['oracle_top1']['orders'].items() for k, c in ex.items() if isinstance(c, dict) and c.get('orders')]
    ins = [(b, k, c) for b, ex in po['selections']['insample_top1']['orders'].items() for k, c in ex.items() if isinstance(c, dict) and c.get('orders')]
    sec = {tg: [(nm, k, c['value_per_intended_c'], c['paying']) for nm in ('model_top1', 'model_e_positive') for k, c in out['targets'][tg]['selections'][nm]['orders']['A'].items()
                if isinstance(c, dict) and c.get('orders')] for tg in SECONDARY}
    out['verdict'] = dict(
        heldout_cells=len(held), heldout_cells_with_min_orders=len(big),
        heldout_paying=[dict(selection=a, book=b, order=k, value_c=c['value_per_intended_c'], orders=c['orders'], days_positive=c['days_positive']) for a, b, k, c in held if c['paying']],
        heldout_best=dict(selection=best[0], book=best[1], order=best[2], value_c=best[3]['value_per_intended_c'], orders=best[3]['orders'], fill=best[3]['fill'],
                          days_positive=best[3]['days_positive']) if best else None,
        any_heldout_paying=bool(any(c['paying'] for *_, c in held)),
        insample_best=max(((b, k, c['value_per_intended_c']) for b, k, c in ins), key=lambda x: x[2]),
        oracle_best=max(((b, k, c['value_per_intended_c']) for b, k, c in orc), key=lambda x: x[2]),
        secondary_paying={tg: [x for x in v if x[3]] for tg, v in sec.items()},
        secondary_best={tg: max(v, key=lambda x: x[2]) for tg, v in sec.items() if v})
    (od or OUT).mkdir(parents=True, exist_ok=True)
    P.to_parquet(fpp, index=False)
    TR.dump(fn, out)
    print(json.dumps(out['verdict'], indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------------- selftest
def selftest():
    rng = np.random.default_rng(1)
    n = 3000
    P = np.full((n, 5), -1.); S = np.zeros((n, 5)); d = np.where(rng.random(n) < .5, 1., -1.)
    for i in range(n):
        k = rng.integers(0, 6)
        base = rng.integers(5, 95)
        lv = base + d[i] * np.cumsum(rng.integers(1, 4, k))
        lv = lv[(lv >= 1) & (lv <= 99)]
        P[i, :len(lv)] = lv; S[i, :len(lv)] = rng.integers(1, 80, len(lv)).astype(float)
    lim = np.where(P[:, 0] >= 0, P[:, 0], 50.) + d * rng.integers(-2, 12, n)
    fl, co, fe = vwalk(P, S, d, lim)
    for i in range(n):
        row = np.full(22, -1.)
        b = 2 + 10 * (1 if d[i] > 0 else 0)
        row[b:b + 5] = P[i]; row[b + 5:b + 10] = S[i]
        q, vw = TM.walk(row, 1 if d[i] > 0 else 0, d[i], lim[i], Q)
        assert abs(q - fl[i]) < 1e-9 and (q == 0 or abs(vw - co[i] / fl[i]) < 1e-9), (i, q, vw, fl[i], co[i])
        if q > 0 and np.all(P[i][P[i] >= 0] <= 50.) or q > 0 and np.all(P[i][P[i] >= 0] >= 50.):
            assert abs(fe[i] - q * fee(vw)) < 1e-9          # one side of 50: the per-level fee is the fee at the vwap
    # a zero-size level stops the walk as in TAKE_MODEL's
    row = np.full(22, -1.); row[12:15] = [55., 56., 57.]; row[17:20] = [20., 0., 30.]
    q, vw = TM.walk(row, 1, 1., 57., Q)
    fl, co, fe = vwalk(row[None, 12:17], row[None, 17:22], np.array([1.]), np.array([57.]))
    assert q == 20. and fl[0] == 20. and co[0] == 20 * 55., (q, fl, co)
    # minbook: 55 gone, 56 shrunk to 10, 58 kept at 50; the walk continues past an emptied level
    PA = np.array([[55., 56., 58., -1., -1.]]); SA = np.array([[20., 20., 50., 0., 0.]])
    PB = np.array([[56., 58., 60., -1., -1.]]); SB = np.array([[10., 60., 5., 0., 0.]])
    P2, S2 = minbook(PA, SA, PB, SB)
    assert P2.tolist() == [[56., 58., -1., -1., -1.]] and S2.tolist() == [[10., 50., 0., 0., 0.]], (P2, S2)
    fl, co, fe = vwalk(P2, S2, np.array([1.]), np.array([58.]))
    assert fl[0] == 50. and co[0] == 10 * 56. + 40 * 58.
    # adaptive limit: at P the predicted value pays the margin, one cent further it does not (unless clipped)
    md = rng.uniform(2, 98, 5000); dd = np.where(rng.random(5000) < .5, 1., -1.); mh = rng.normal(0, 6, 5000); mg = rng.choice([0., 1.], 5000)
    al = adaptive_limit(md, dd, mh, mg)
    M = md + dd * mh
    v_at = dd * (M - al) - fee(al)
    v_next = dd * (M - (al + dd)) - fee(al + dd)
    inner = (al > 1.) & (al < 99.)
    assert np.all(v_at[inner] >= mg[inner] - 1e-9), 'adaptive limit does not pay the margin'
    assert np.all(v_next[inner] < mg[inner] + 1e-9), 'adaptive limit is not the furthest price'
    # falsifiers: parity flags a changed label; the split must add up
    r0 = dict(pf_mid=50.5, v_172_0=-1.25, fill_172_0=1.)
    assert parity_row(r0, dict(pf_mid=50.5, v_172_0=-1.25, fill_172_0=1.)) == []
    assert parity_row(r0, dict(pf_mid=50.5, v_172_0=-1.24, fill_172_0=1.)) == ['v_172_0']
    assert parity_row(r0, dict(pf_mid=50.5, v_172_0=-1.25, fill_172_0=.98)) == ['fill_172_0']
    w = (np.array([50., 20.]), np.array([50 * 55., 20 * 56.]), np.array([50 * fee(55.), 20 * fee(56.)]))
    c = cell(np.array([0, 1]), np.array([True, True]), w, np.array([1., 1.]), np.array([58., 54.]), np.array([55., 55.]), np.array(['20261023', '20261031']), np.array([1, 2]))
    v0 = (50 * (58 - 55) - 50 * fee(55.)) / Q; v1 = (20 * (54 - 56) - 20 * fee(56.)) / Q
    assert abs(c['value_per_intended_c'] - (v0 + v1) / 2) < 1e-12 and c['fill'] == 70. / 100.
    assert abs(c['per_filled_value_c'] - (c['per_filled_move_c'] - c['per_filled_slippage_c'] - c['per_filled_fee_c'])) < 1e-12
    assert c['days_positive'] == 1 and c['paying'] is False
    print('take_markout selftest PASS')


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'labels':
        labels(a[0], a[1] if len(a) > 1 else None, int(a[2]) if len(a) > 2 else None)
    elif cmd == 'train':
        train(a[0] if a else None)
    else:
        dict(selftest=selftest)[cmd]()
