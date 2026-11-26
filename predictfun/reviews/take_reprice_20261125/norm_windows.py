"""Do longer 'normal' windows or z-scores make the dynamic factors more informative? (USER 2026-11-26: "our factors like refill rate,
cancelling rate, imbalance ratio, and other dynamic factors may not use a long enough window to show the z-score, how do we get the normal
rate, what's the window we use? Can we try longer window and see the effect"). Readout on the four consumed days; no strategy change.
So far (public_score3): a rate factor = log((rate over the last 100 / 300 ms + e) / (rate over (t - 30 s, t - 1 s] + e)); imbalance and size
changes not normalised.
Points: public_score3's table (the top taker's fires and their nearby moments; random moments with no prior move, label = a Binance move
its way within 300 ms). Events: fire_events.day_data (Binance futures BTC best level, E clock) unchanged.
Declared before the run: factors = far-side new orders (refill), far cancels, far trades (its way), near-side new orders (posts), near
cancels, near trades (counter-trades) and all book activity, each over the last 300 ms; far / near replacement (new / (cancels + trades))
over 300 ms; best-level imbalance its way; its change over 300 ms; far / near best size change over 300 ms (log). Versions: raw (no
normalisation); ratio_B = log((rate now + e) / (mean rate over (t - B, t - 1 s] + e)) (flows; replacement: now minus its value over the
baseline); z_B = (value now - mean) / (sd + floor) over the baseline's 300 ms bins of a fixed 300 ms grid (flows, activity, the
state factors), B = 30 s, 2, 5, 15, 60 min. One population for every version: points with st ok, t >= the day's start + 60 min + 1 s and
the bookTicker hours of (t - 60 min - 1 s, t] present (counted).
Readings: per factor and version, the separation max(AUC, 1 - AUC) of the top taker's fires from their nearby moments, and of the
300 ms label at random moments, with the AUC's direction; the best version per factor against the current one (ratio_30s; raw for the state
factors).
Outputs (write-once): OUT/norm_windows/<day>.parquet; norm_windows.json (merge).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/norm_windows.py \
       day <day> [<dir> <n points>] | merge [<dir>]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import public_score3 as P3  # noqa: E402

FE, C, TR, WT, DC = P3.FE, P3.C, P3.TR, P3.WT, P3.DC
OUT = WT.OUT / 'norm_windows'
DAYS = ('20261023', '20261031', '20261101', '20261102')
BASES = (30000., 120000., 300000., 900000., 3600000.)
BL = {30000.: '30s', 120000.: '2m', 300000.: '5m', 900000.: '15m', 3600000.: '60m'}
GAP, STEP, EPS = 1000., 300., P3.EPS
FLOORS = dict(flow=1e-3, count=1., imb=1e-3, lsz=1e-3)
FLOWS = (('opp', 'add', 'far_new'), ('opp', 'cancel', 'far_cxl'), ('opp', 'trade', 'far_trade'), ('near', 'add', 'near_new'), ('near', 'cancel', 'near_cxl'),
         ('near', 'trade', 'near_trade'))
ACT = [(r, k) for r in ('opp', 'near') for k in ('trade', 'cancel', 'add', 'gone', 'improve')]


def grid_stats(vals, G, t, B, valid=None):
    """Mean and sd of grid values vals[i] (time G[i]) with G[i] in (t - B, t - GAP]; valid marks usable grid values."""
    v = np.where(np.isfinite(vals), vals, 0.)
    ok = np.isfinite(vals).astype(float) if valid is None else valid.astype(float)
    cs, cs2, cn = np.r_[0., np.cumsum(v)], np.r_[0., np.cumsum(v * v)], np.r_[0., np.cumsum(ok)]
    lo, hi = np.searchsorted(G, t - B, 'right'), np.searchsorted(G, t - GAP, 'right')
    n = cn[hi] - cn[lo]
    mean = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1.), np.nan)
    var = np.where(n > 1, (cs2[hi] - cs2[lo]) / np.maximum(n, 1.) - mean ** 2, np.nan)
    return mean, np.sqrt(np.maximum(var, 0.)), n


def bin_sums(E, cA, G):
    """Sum over each 300 ms bin (G[i] - STEP, G[i]] of an event series with cumulative amounts cA."""
    return cA[np.searchsorted(E, G, 'right')] - cA[np.searchsorted(E, G - STEP, 'right')]


def state_grid(day, G):
    """Best-level imbalance (up), log ask / bid size at the grid times, from the day's bookTicker hours (NaN where no state yet)."""
    imb, la, lb = (np.full(len(G), np.nan) for _ in range(3))
    for h, bk in DC.book_hours(day):
        E, bid, bq, ask, aq = FE.collapse(*bk)
        h0 = float(WT.OD_day_ms(day)) + h * 3600000.
        sel = np.flatnonzero((G >= h0) & (G < h0 + 3600000.))
        j = np.searchsorted(E, G[sel], 'right') - 1
        ok = j >= 0
        jj = np.maximum(j, 0)
        imb[sel] = np.where(ok, (bq[jj] - aq[jj]) / np.maximum(bq[jj] + aq[jj], 1e-9), np.nan)
        la[sel] = np.where(ok & (aq[jj] > 0), np.log(np.maximum(aq[jj], 1e-12)), np.nan)
        lb[sel] = np.where(ok & (bq[jj] > 0), np.log(np.maximum(bq[jj], 1e-12)), np.nan)
        C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
    return imb, la, lb


def day_run(day, od=None, limit=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    Z = pd.read_parquet(WT.OUT / 'public_score3_points.parquet', columns=['day', 't', 'd', 'kind', 'y'])
    Z = Z[Z.day == day].reset_index(drop=True)
    if limit:
        Z = Z.iloc[np.sort(np.random.default_rng(1).choice(len(Z), min(limit, len(Z)), replace=False))].reset_index(drop=True)
    day0 = float(WT.OD_day_ms(day))
    t, d = Z.t.to_numpy(float), Z.d.to_numpy(float)
    n = len(t)
    ev, st, traj, _ = FE.day_data(day, Z[['t', 'd']])
    hrs = np.array([(DC.RAW / f'bookTicker/BTCUSDT/{day}_{h:02d}.csv.gz').exists() for h in range(24)])
    h_lo, h_hi = ((t - max(BASES) - GAP - day0) // 3600000).astype(int), ((t - day0) // 3600000).astype(int)
    allh = np.array([bool(hrs[max(a, 0):b + 1].all()) for a, b in zip(h_lo, h_hi)])
    valid = (st == 'ok') & (t - max(BASES) - GAP >= day0) & allh
    G = day0 + STEP * np.arange(1, int(86400000 / STEP) + 1)
    F = {}
    for r, k, nm in FLOWS:
        for vers in ['raw'] + [f'ratio_{BL[b]}' for b in BASES] + [f'z_{BL[b]}' for b in BASES]:
            F[f'{nm}|{vers}'] = np.full(n, np.nan)
    for nm in ('activity',):
        for vers in ['raw'] + [f'ratio_{BL[b]}' for b in BASES] + [f'z_{BL[b]}' for b in BASES]:
            F[f'{nm}|{vers}'] = np.full(n, np.nan)
    for nm in ('far_replace', 'near_replace'):
        for vers in ['raw'] + [f'ratio_{BL[b]}' for b in BASES]:
            F[f'{nm}|{vers}'] = np.full(n, np.nan)
    for nm in ('imbalance', 'imb_change', 'far_size_change', 'near_size_change'):
        for vers in ['raw'] + [f'z_{BL[b]}' for b in BASES]:
            F[f'{nm}|{vers}'] = np.full(n, np.nan)
    act_now, act_bins = np.zeros(n), {dv: np.zeros(len(G)) for dv in (1., -1.)}
    act_base = {b: np.zeros(n) for b in BASES}
    for dv in (1., -1.):
        m = d == dv
        if not m.any():
            continue
        tt = t[m]
        V = {}
        for r, k, nm in FLOWS:
            E, cA = FE.role_events(ev, r, k, dv)
            now = P3.wsum(E, cA, tt - STEP, tt)
            S = bin_sums(E, cA, G)
            F[f'{nm}|raw'][m] = now
            V[nm] = {'now': now}
            for b in BASES:
                base = P3.wsum(E, cA, tt - b, tt - GAP)
                F[f'{nm}|ratio_{BL[b]}'][m] = P3.lr(now, STEP, base, b - GAP, EPS)
                mu, sd, _ = grid_stats(S, G, tt, b)
                F[f'{nm}|z_{BL[b]}'][m] = (now - mu) / (sd + FLOORS['flow'])
                V[nm][b] = base
        for side, a_, c_, tr_ in (('far', 'far_new', 'far_cxl', 'far_trade'), ('near', 'near_new', 'near_cxl', 'near_trade')):
            e3 = EPS * STEP
            now = np.log((V[a_]['now'] + e3) / (V[c_]['now'] + V[tr_]['now'] + e3))
            F[f'{side}_replace|raw'][m] = now
            for b in BASES:
                eb = EPS * (b - GAP)
                F[f'{side}_replace|ratio_{BL[b]}'][m] = now - np.log((V[a_][b] + eb) / (V[c_][b] + V[tr_][b] + eb))
        for r, k in ACT:
            E, cA = FE.role_events(ev, r, k, dv)
            ones = np.r_[0., np.arange(1, len(E) + 1, dtype=float)]
            act_now[m] += P3.wsum(E, ones, tt - STEP, tt)
            act_bins[dv] += bin_sums(E, ones, G)
            for b in BASES:
                act_base[b][m] += P3.wsum(E, ones, tt - b, tt - GAP)
    for dv in (1., -1.):
        m = d == dv
        if not m.any():
            continue
        tt = t[m]
        F['activity|raw'][m] = act_now[m]
        for b in BASES:
            F[f'activity|ratio_{BL[b]}'][m] = np.log((act_now[m] / STEP + P3.EPSN) / (act_base[b][m] / (b - GAP) + P3.EPSN))
            mu, sd, _ = grid_stats(act_bins[dv], G, tt, b)
            F[f'activity|z_{BL[b]}'][m] = (act_now[m] - mu) / (sd + FLOORS['count'])
    del ev
    imb_up, la, lb = state_grid(day, G)
    dimb = np.r_[np.nan, np.diff(imb_up)]
    dla, dlb = np.r_[np.nan, np.diff(la)], np.r_[np.nan, np.diff(lb)]
    imb_now = traj['imb_0']
    imb_chg = traj['imb_0'] - traj['imb_300']
    far_chg, near_chg = np.log(traj['opp_0'] / traj['opp_300']), np.log(traj['near_0'] / traj['near_300'])
    F['imbalance|raw'], F['imb_change|raw'], F['far_size_change|raw'], F['near_size_change|raw'] = imb_now, imb_chg, far_chg, near_chg
    for b in BASES:
        mu, sd, _ = grid_stats(imb_up, G, t, b)
        F[f'imbalance|z_{BL[b]}'] = (imb_now - d * mu) / (sd + FLOORS['imb'])
        mu, sd, _ = grid_stats(dimb, G, t, b)
        F[f'imb_change|z_{BL[b]}'] = (imb_chg - d * mu) / (sd + FLOORS['imb'])
        mua, sda, _ = grid_stats(dla, G, t, b); mub, sdb, _ = grid_stats(dlb, G, t, b)
        up = d > 0
        F[f'far_size_change|z_{BL[b]}'] = (far_chg - np.where(up, mua, mub)) / (np.where(up, sda, sdb) + FLOORS['lsz'])
        F[f'near_size_change|z_{BL[b]}'] = (near_chg - np.where(up, mub, mua)) / (np.where(up, sdb, sda) + FLOORS['lsz'])
    out = pd.DataFrame({k: np.where(valid, v, np.nan) for k, v in F.items()})
    Zo = pd.concat([Z, out], axis=1)
    Zo['valid'] = valid
    od.mkdir(parents=True, exist_ok=True)
    Zo.to_parquet(fp, index=False)
    print(day, n, int(valid.sum()), flush=True)


def merge(od=None):
    fn = (Path(od) if od else H) / 'norm_windows.json'
    C.require(not fn.exists(), 'norm_windows.json exists (write-once)')
    src = Path(od) if od else OUT
    Z = pd.concat([pd.read_parquet(src / f'{d}.parquet') for d in DAYS if (src / f'{d}.parquet').exists()], ignore_index=True)
    Z = Z[Z.valid]
    cols = [c for c in Z.columns if '|' in c]
    Fz, N, R = Z[Z.kind == 'fire'], Z[Z.kind == 'nearby'], Z[Z.kind == 'random']
    res = {}
    for c in cols:
        fac, vers = c.split('|')
        a, b = Fz[c].to_numpy(float), N[c].to_numpy(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        rv, ry = R[c].to_numpy(float), R.y.to_numpy(float)
        ok = np.isfinite(rv)
        o = dict(n_fires=int(len(a)), n_nearby=int(len(b)), n_random=int(ok.sum()))
        if len(a) > 30 and len(b) > 30 and np.r_[a, b].std() > 0:
            au = float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])); o.update(auc_fires=au, sep_fires=max(au, 1. - au))
        if ok.sum() > 100 and ry[ok].std() > 0 and rv[ok].std() > 0:
            au = float(roc_auc_score(ry[ok], rv[ok])); o.update(auc_move=au, sep_move=max(au, 1. - au))
        res.setdefault(fac, {})[vers] = o
    summ = {}
    for fac, vv in res.items():
        cur = 'ratio_30s' if 'ratio_30s' in vv else 'raw'
        best_f = max((k for k in vv if 'sep_fires' in vv[k]), key=lambda k: vv[k]['sep_fires'])
        best_m = max((k for k in vv if 'sep_move' in vv[k]), key=lambda k: vv[k]['sep_move'])
        summ[fac] = dict(current=cur, current_sep_fires=vv[cur].get('sep_fires'), current_sep_move=vv[cur].get('sep_move'), best_fires=best_f, best_sep_fires=vv[best_f]['sep_fires'],
                         best_move=best_m, best_sep_move=vv[best_m]['sep_move'], raw_sep_fires=vv['raw'].get('sep_fires'), raw_sep_move=vv['raw'].get('sep_move'))
    out = dict(asof='2026-11-26', n=dict(fires=int(len(Fz)), nearby=int(len(N)), random=int(len(R)), random_moves=int(R.y.sum())), factors=res, summary=summ)
    TR.dump(fn, out)
    for fac, s in summ.items():
        print(f"{fac:18s} current {s['current']:10s} fires {s['current_sep_fires']:.3f} move {s['current_sep_move']:.3f} | best fires {s['best_fires']:10s} {s['best_sep_fires']:.3f} | best move {s['best_move']:10s} {s['best_sep_move']:.3f} | raw {s['raw_sep_fires']:.3f} / {s['raw_sep_move']:.3f}")


def selftest():
    G = np.arange(1., 11.) * STEP + 0.                         # grid 300 .. 3000
    vals = np.arange(1., 11.)
    mu, sd, n_ = grid_stats(vals, G, np.array([3000. + GAP]), 2500.)
    assert n_[0] == 5 and abs(mu[0] - 8.) < 1e-12 and abs(sd[0] - np.std([6., 7., 8., 9., 10.])) < 1e-12, (n_, mu, sd)   # G in (2500, 3000]... (t - B, t - GAP]
    E = np.array([100., 350., 400., 950.]); cA = np.r_[0., np.cumsum([1., 2., 3., 4.])]
    S = bin_sums(E, cA, np.array([300., 600., 900., 1200.]))
    assert S.tolist() == [1., 5., 0., 4.], S
    print('norm_windows selftest PASS')


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'day':
        day_run(a[0], a[1] if len(a) > 1 else None, int(a[2]) if len(a) > 2 else None)
    elif cmd == 'merge':
        merge(a[0] if a else None)
    else:
        dict(selftest=selftest)[cmd]()
