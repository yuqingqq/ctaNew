"""Combinations of the smart takers' factors, and the same factors over longer look-back windows (USER 2026-11-26: "Yea we can test the
combinations, can we also look back a longer window"). Readout on the four consumed days; no strategy change.
Points: the top taker's fires, nearby and random moments (public_score3) and the ranks 2-10 fires and nearby moments (smart_factors), all
with no move their way in the previous 300 ms.
(A) Combinations, declared before the run: the eight distinct factors of SMART_FACTORS' top twelve - imbalance change 300 ms (up), far-side
size change 300 ms (down), far-side replacement vs normal (down), near-side new orders vs normal 300 ms (up), near-side size change 300 ms
(up), best-level imbalance (up), far-side size at the best (down), depth imbalance change 300 ms (up); a factor is present beyond the top
taker's nearby 80th (20th, for down) percentile (one rule for every group); the count present (0-8); per group the share at each count and
at >= k, fires vs nearby vs random moments, and the Binance move within 300 ms by count; every pair and triple: share at its fires, at
random moments (lift) and the move rate at random moments where present (support >= 200 random moments).
(B) Longer windows, declared before the run: over w = 0.3 / 1 / 3 / 10 s - imbalance change, log far / near size change (best level, Binance
futures bookTicker on E, per-hour states: a window reaching before the hour's states has no value), far / near replacement (new orders /
(cancels + trades), fire_events' events, rates with e = 1e-5 BTC / ms), near-side new orders and far-side cancels against normal (normal =
the rate over (t - 60 s, t - 10 s]), depth imbalance change over 20 levels (depth20 snapshots at or before t and t - w); per group and factor
the separation of fires from nearby moments (max(AUC, 1 - AUC)), its direction and the share of fires beyond the nearby 80th / 20th
percentile.
Outputs (write-once): smart_combo.json; OUT/smart_combo_points.parquet (the longer-window factors).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/smart_combo.py \
       [<rehearsal dir> <day> ...]
"""
import itertools
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import fire_events as FE  # noqa: E402
import public_score as PS  # noqa: E402
import public_score3 as P3  # noqa: E402

DC, C, TR, WT = FE.DC, FE.C, FE.TR, FE.WT
F8 = [('x_imb_d300', 1, 'imbalance rising (300 ms)'), ('x_far_r300', -1, 'far side shrinking (300 ms)'), ('x_far_replace_dev', -1, 'far side not replaced'),
      ('x_near_new_r300', 1, 'near-side new orders above normal'), ('x_near_r300', 1, 'near side growing (300 ms)'), ('b_imb1', 1, 'imbalance its way now'),
      ('b_lfar1', -1, 'thin far side now'), ('z_imb_d300', 1, 'depth imbalance rising (300 ms)')]
WINS = (300., 1000., 3000., 10000.)
BASE_A, BASE_B, EPS, SUPPORT = 60000., 10000., 1e-5, 200
LONG = ('imb_change', 'far_change', 'near_change', 'far_replace', 'near_replace', 'near_new_vs_normal', 'far_cxl_vs_normal', 'depth_imb_change')
LDIR = dict(imb_change=None, far_change=None, near_change=None, far_replace=None, near_replace=None, near_new_vs_normal=None, far_cxl_vs_normal=None, depth_imb_change=None)


def wsum(E, cA, a, b):
    return cA[np.searchsorted(E, b, 'right')] - cA[np.searchsorted(E, a, 'right')]


def long_feats(day, P):
    """The longer-window factors for points P (t, d) of one day."""
    day0 = float(WT.OD_day_ms(day))
    t, d = P.t.to_numpy(float), P.d.to_numpy(float)
    n = len(t)
    O = {f'{k}_{int(w)}': np.full(n, np.nan) for k in LONG for w in WINS}
    ev, st, _, _ = FE.day_data(day, P[['t', 'd']])
    C.require((st == 'ok').all(), f'{day}: a point without a book')
    base_ok = t - BASE_A >= day0
    for dv in (1., -1.):
        m = d == dv
        if not m.any():
            continue
        tt = t[m]
        V = {}
        for r, k in (('opp', 'add'), ('opp', 'cancel'), ('opp', 'trade'), ('near', 'add'), ('near', 'cancel'), ('near', 'trade')):
            E, cA = FE.role_events(ev, r, k, dv)
            V[(r, k)] = (E, cA)
        for w in WINS:
            s = {key: wsum(E, cA, tt - w, tt) for key, (E, cA) in V.items()}
            O[f'far_replace_{int(w)}'][m] = np.log((s[('opp', 'add')] / w + EPS) / ((s[('opp', 'cancel')] + s[('opp', 'trade')]) / w + EPS))
            O[f'near_replace_{int(w)}'][m] = np.log((s[('near', 'add')] / w + EPS) / ((s[('near', 'cancel')] + s[('near', 'trade')]) / w + EPS))
            for key, nm in ((('near', 'add'), 'near_new_vs_normal'), (('opp', 'cancel'), 'far_cxl_vs_normal')):
                E, cA = V[key]
                base = wsum(E, cA, tt - BASE_A, tt - BASE_B) / (BASE_A - BASE_B)
                O[f'{nm}_{int(w)}'][m] = np.where(base_ok[m], np.log((s[key] / w + EPS) / (base + EPS)), np.nan)
    del ev
    hp = ((t - day0) // 3600000).astype(int)
    for h, bk in DC.book_hours(day):
        sel = np.flatnonzero(hp == h)
        if not len(sel):
            continue
        E, bid, bq, ask, aq = FE.collapse(*bk)
        tt, dd = t[sel], d[sel]
        up = dd > 0
        j0 = WT.last_idx(E, tt)
        far0, near0 = np.where(up, aq[j0], bq[j0]), np.where(up, bq[j0], aq[j0])
        imb0 = (near0 - far0) / (near0 + far0)
        fd = DC.RAW / f'depth20/BTCUSDT/{day}_{h:02d}.csv.gz'
        D = pd.read_csv(fd, header=None, usecols=[1, 4, 5], names=['E', 'b', 'a']).sort_values('E', kind='stable') if fd.exists() else None
        if D is not None:
            Ed = D.E.to_numpy(float)
            idx = {w: np.searchsorted(Ed, tt - w, 'right') - 1 for w in (0.,) + WINS}
            u = np.unique(np.concatenate([v[v >= 0] for v in idx.values()]))
            bP, bQ = PS.DD.parse(D.b.to_numpy()[u])
            aP, aQ = PS.DD.parse(D.a.to_numpy()[u])
            bs, as_ = np.nansum(bQ, 1), np.nansum(aQ, 1)
            imbd = (bs - as_) / np.maximum(bs + as_, 1e-12)
            pos = {w: np.searchsorted(u, np.maximum(v, 0)) for w, v in idx.items()}
            fresh = (idx[0.] >= 0) & (tt - Ed[np.maximum(idx[0.], 0)] <= PS.DD.STALE)
        for w in WINS:
            jw = WT.last_idx(E, tt - w)
            ok = (jw >= 0) & (tt - w >= E[0])
            jj = np.maximum(jw, 0)
            farw, nearw = np.where(up, aq[jj], bq[jj]), np.where(up, bq[jj], aq[jj])
            imbw = (nearw - farw) / (nearw + farw)
            O[f'imb_change_{int(w)}'][sel] = np.where(ok, imb0 - imbw, np.nan)
            O[f'far_change_{int(w)}'][sel] = np.where(ok, np.log(far0 / farw), np.nan)
            O[f'near_change_{int(w)}'][sel] = np.where(ok, np.log(near0 / nearw), np.nan)
            if D is not None:
                okd = fresh & (idx[w] >= 0)
                O[f'depth_imb_change_{int(w)}'][sel] = np.where(okd, dd * (imbd[pos[0.]] - imbd[pos[w]]), np.nan)
        C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
    return O


def sep(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 30 or len(b) < 30 or np.r_[a, b].std() == 0:
        return None
    auc = float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))
    hi = auc >= .5
    thr = np.quantile(b, .8 if hi else .2)
    pf = float((a > thr).mean() if hi else (a < thr).mean())
    pn = float((b > thr).mean() if hi else (b < thr).mean())
    return dict(separation=max(auc, 1. - auc), direction='up' if hi else 'down', fires_present=pf, nearby_present=pn, fires_median=float(np.median(a)), nearby_median=float(np.median(b)))


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    od = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    fn, fp = (od or H) / 'smart_combo.json', (od or WT.OUT) / 'smart_combo_points.parquet'
    C.require(not fn.exists() and not fp.exists(), 'an output exists (write-once)')
    days = tuple(sys.argv[2:]) if len(sys.argv) > 2 else WT.DAYS
    C.require(all(x in WT.DAYS for x in days), 'not a declared day')
    cols = [c for c, _, _ in F8]
    T1 = pd.read_parquet(WT.OUT / 'public_score3_points.parquet', columns=['day', 't', 'd', 'kind', 'y'] + cols)
    T1 = T1[T1.day.isin(days)].reset_index(drop=True)
    T1['grp'] = 'TOP1'
    S2 = pd.read_parquet(WT.OUT / 'smart_factors_points.parquet', columns=['day', 't', 'd', 'grp', 'fire', 'y'] + cols)
    S2 = S2[S2.day.isin(days)].reset_index(drop=True)
    S2['kind'] = np.where(S2.fire, 'fire', 'nearby')
    A = pd.concat([T1, S2.drop(columns=['fire'])], ignore_index=True)
    # (A) combinations: one rule from the top taker's nearby moments
    NB = A[(A.grp == 'TOP1') & (A.kind == 'nearby')]
    thr = {c: float(np.nanquantile(NB[c], .8 if s > 0 else .2)) for c, s, _ in F8}
    for c, s, _ in F8:
        A[f'on_{c}'] = (A[c] > thr[c]) if s > 0 else (A[c] < thr[c])
    on = [f'on_{c}' for c, _, _ in F8]
    A['n_on'] = A[on].sum(axis=1)                                  # not 'count': a DataFrame method name
    out = dict(asof='2026-11-26', days=list(days), factors=[dict(factor=c, direction='up' if s > 0 else 'down', label=lb, threshold=thr[c]) for c, s, lb in F8],
               combos={}, pairs_triples={}, long_windows={})
    sets = {'TOP1': ('TOP1',), 'RANK2': ('RANK2',), 'RANKS3_10': ('RANKS3_10',)}
    for g, gs in sets.items():
        Fz, N = A[(A.grp.isin(gs)) & (A.kind == 'fire')], A[(A.grp.isin(gs)) & (A.kind == 'nearby')]
        o = dict(n_fires=int(len(Fz)), n_nearby=int(len(N)), by_count={})
        for k in range(9):
            o['by_count'][str(k)] = dict(fires_share=float((Fz['n_on'] == k).mean()), nearby_share=float((N['n_on'] == k).mean()),
                                         fires_at_least=float((Fz['n_on'] >= k).mean()), nearby_at_least=float((N['n_on'] >= k).mean()),
                                         fires_move300=float(Fz.y[Fz['n_on'] == k].mean()) if (Fz['n_on'] == k).any() else None,
                                         nearby_move300=float(N.y[N['n_on'] == k].mean()) if (N['n_on'] == k).any() else None)
        out['combos'][g] = o
    Rr = A[(A.grp == 'TOP1') & (A.kind == 'random')]
    out['combos']['random'] = dict(n=int(len(Rr)), by_count={str(k): dict(share=float((Rr['n_on'] == k).mean()), at_least=float((Rr['n_on'] >= k).mean()),
                                                                       move300=float(Rr.y[Rr['n_on'] == k].mean()) if (Rr['n_on'] == k).any() else None,
                                                                       move300_at_least=float(Rr.y[Rr['n_on'] >= k].mean()) if (Rr['n_on'] >= k).any() else None)
                                                                for k in range(9)})
    F1 = A[(A.grp == 'TOP1') & (A.kind == 'fire')]
    lab = {f'on_{c}': lb for c, _, lb in F8}
    for size in (2, 3):
        rows = []
        for combo in itertools.combinations(on, size):
            rm = Rr[list(combo)].all(axis=1)
            if rm.sum() < SUPPORT:
                continue
            fs = float(F1[list(combo)].all(axis=1).mean())
            rows.append(dict(factors=[lab[c] for c in combo], fires_share=fs, random_share=float(rm.mean()), lift=fs / float(rm.mean()),
                             random_move300=float(Rr.y[rm].mean()), random_n=int(rm.sum())))
        out['pairs_triples'][str(size)] = dict(by_lift=sorted(rows, key=lambda r: -r['lift'])[:8], by_move=sorted(rows, key=lambda r: -r['random_move300'])[:8], n=len(rows))
    # (B) longer windows on fires and nearby moments
    Q = A[A.kind.isin(['fire', 'nearby'])].reset_index(drop=True)
    parts = []
    for day in days:
        Qd = Q[Q.day == day].reset_index(drop=True)
        if not len(Qd):
            continue
        for k, v in long_feats(day, Qd).items():
            Qd[k] = v
        parts.append(Qd)
        print(day, len(Qd), flush=True)
    Q = pd.concat(parts, ignore_index=True)
    for g, gs in sets.items():
        Fz, N = Q[Q.grp.isin(gs) & (Q.kind == 'fire')], Q[Q.grp.isin(gs) & (Q.kind == 'nearby')]
        out['long_windows'][g] = {f'{k}_{int(w)}': sep(Fz[f'{k}_{int(w)}'].to_numpy(float), N[f'{k}_{int(w)}'].to_numpy(float)) for k in LONG for w in WINS}
    out['long_coverage'] = {c: float(Q[c].notna().mean()) for c in Q.columns if c.split('_')[-1].isdigit() and c[:-len(c.split('_')[-1]) - 1] in LONG}
    Q.drop(columns=on).to_parquet(fp, index=False)
    TR.dump(fn, out)
    for g in ('TOP1', 'RANKS3_10', 'RANK2'):
        o = out['combos'][g]['by_count']
        print(g, 'fires >=k', [round(o[str(k)]['fires_at_least'], 3) for k in range(9)], 'nearby >=k', [round(o[str(k)]['nearby_at_least'], 3) for k in range(9)])
        print('   move by count fires', [o[str(k)]['fires_move300'] and round(o[str(k)]['fires_move300'], 2) for k in range(9)], 'nearby', [o[str(k)]['nearby_move300'] and round(o[str(k)]['nearby_move300'], 3) for k in range(9)])
    rb = out['combos']['random']['by_count']
    print('random >=k', [round(rb[str(k)]['at_least'], 4) for k in range(9)], 'move >=k', [rb[str(k)]['move300_at_least'] and round(rb[str(k)]['move300_at_least'], 3) for k in range(9)])
    for k in LONG:
        print(k, [(int(w), (lambda r: r and round(r['separation'], 3))(out['long_windows']['TOP1'][f'{k}_{int(w)}'])) for w in WINS])


if __name__ == '__main__':
    main()
