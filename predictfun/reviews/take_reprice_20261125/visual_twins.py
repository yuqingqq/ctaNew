"""Do moments that look like the top taker's fires move? (USER 2026-11-27: "Do they fire at 0 in the chart? It looks the cases they fire are
very different from normal cases, why our model can't differentiate"). Readout over public_score3's points; four consumed days; declared
before the run.
Points: SIGNAL_VS_NOISE's: the top taker's fires (t = its order's arrival at Predict) and random moments, all with no Binance futures move
their way in the previous 300 ms; random keys seen twice dropped. Label: a Binance move their way within 300 ms.
Look: SIGNAL_VS_NOISE's nine key book inputs (best-level imbalance, log far / near size, far cancels / new orders / trades and near new
orders over 300 ms, imbalance change over 300 ms, far levels 2-5 removed), each standardized by the random moments' median and IQR.
Twins: each fire's 20 nearest random moments (Euclidean on the nine) of the same day and direction, any time of day; and within +-30 min.
Readouts: move rate of fires, of their twins (all; the nearest; by distance fifth over all pairs) and of random moments; the twins' held-
out public score against the fires'; the distance scale (each fire's nearest other fire of the same day and direction).
Stage examples: the chart's typical fire (SIGNAL_VS_NOISE) and its two nearest twins whatever their outcome, with SIGNAL_VS_NOISE's
Binance windows.
Outputs (write-once): visual_twins.json (read), visual_twins_examples.json (examples).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/visual_twins.py \
       read [<dir>] | examples [<dir>]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import signal_vs_noise as SVN  # noqa: E402

C, TR, DAYS, KEY = SVN.C, SVN.TR, SVN.DAYS, SVN.KEY_INPUTS
K, WIN_MS = 20, 1800000.


def load():
    P = pd.read_parquet(SVN.BASE / 'public_score3_points.parquet', columns=['day', 't', 'd', 'kind', 'y', 's_gbm_v3'] + KEY)
    P = P[P.kind.isin(['fire', 'random'])]
    P = P[~P.duplicated(['day', 't', 'd', 'kind'], keep=False)].reset_index(drop=True)
    ref = (P.kind == 'random').to_numpy()
    med = P.loc[ref, KEY].median(); iqr = (P.loc[ref, KEY].quantile(.75) - P.loc[ref, KEY].quantile(.25)).replace(0, 1.)
    Z = ((P[KEY] - med) / iqr).to_numpy(float)
    return P, Z


def twins(P, Z):
    """-> per fire (row index): K nearest random rows (any time of day) and K nearest within +-30 min, with distances; nearest other fire."""
    out = {}
    t = P.t.to_numpy(float)
    ok = np.isfinite(Z).all(1)
    for (day, d), g in P.groupby(['day', 'd']):
        gi = g.index.to_numpy()
        ra = gi[(g.kind == 'random').to_numpy() & ok[gi]]
        fi = gi[(g.kind == 'fire').to_numpy() & ok[gi]]
        if len(ra) < K or not len(fi):
            continue
        tree = cKDTree(Z[ra])
        dist, ix = tree.query(Z[fi], k=K)
        ftree = cKDTree(Z[fi]) if len(fi) > 1 else None
        fd = ftree.query(Z[fi], k=2)[0][:, 1] if ftree is not None else np.full(len(fi), np.nan)
        for n, f in enumerate(fi):
            near = np.flatnonzero(np.abs(t[ra] - t[f]) <= WIN_MS)
            if len(near) >= K:
                dl = np.sqrt(((Z[ra[near]] - Z[f]) ** 2).sum(1))
                o = np.argsort(dl, kind='stable')[:K]
                loc = (ra[near][o], dl[o])
            else:
                loc = (np.array([], int), np.array([]))
            out[int(f)] = dict(any=(ra[ix[n]], dist[n]), local=loc, nearest_fire=float(fd[n]))
    return out


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    fn = (Path(od) if od else H) / 'visual_twins.json'
    C.require(not fn.exists(), 'visual_twins.json exists (write-once)')
    P, Z = load()
    T = twins(P, Z)
    y, s = P.y.to_numpy(float), P.s_gbm_v3.to_numpy(float)
    F = np.array(sorted(T))
    A = np.stack([T[f]['any'][0] for f in F]); Ad = np.stack([T[f]['any'][1] for f in F])
    loc_ok = np.array([len(T[f]['local'][0]) == K for f in F])
    Lc = np.stack([T[f]['local'][0] for f in F[loc_ok]]) if loc_ok.any() else np.zeros((0, K), int)
    rnd = (P.kind == 'random').to_numpy()
    out = dict(asof='2026-11-27', fires=int(len(F)), fires_without_local_twins=int((~loc_ok).sum()), k=K,
               move_rate=dict(fires=float(y[F].mean()), twins_any_time=float(y[A].mean()), nearest_twin=float(y[A[:, 0]].mean()),
                              twins_within_30min=float(y[Lc].mean()) if len(Lc) else None, random=float(y[rnd].mean())),
               public_score=dict(fires=float(np.nanmean(s[F])), twins_any_time=float(np.nanmean(s[A])), random=float(np.nanmean(s[rnd]))),
               distance=dict(fire_to_twin_median=float(np.median(Ad)), fire_to_nearest_twin_median=float(np.median(Ad[:, 0])),
                             fire_to_nearest_fire_median=float(np.nanmedian([T[f]['nearest_fire'] for f in F]))),
               by_distance_fifth={}, by_day={})
    q = np.quantile(Ad.ravel(), [.2, .4, .6, .8])
    b = np.searchsorted(q, Ad.ravel(), 'right')
    for k in range(5):
        m = b == k
        out['by_distance_fifth'][str(k + 1)] = dict(pairs=int(m.sum()), distance_median=float(np.median(Ad.ravel()[m])), twin_move_rate=float(y[A.ravel()[m]].mean()))
    for dh in DAYS:
        m = P.day.to_numpy()[F] == dh
        if m.any():
            out['by_day'][dh] = dict(fires=int(m.sum()), fire_move_rate=float(y[F][m].mean()), twin_move_rate=float(y[A[m]].mean()))
    ex = json.loads((H / 'signal_vs_noise.json').read_text())['examples']['typical_fire']
    hit = P.index[(P.kind == 'fire') & (P.day == ex['day']) & (P.t == ex['t']) & (P.d == ex['d'])]
    C.require(len(hit) == 1 and int(hit[0]) in T, 'the chart\'s typical fire not found')
    f = int(hit[0])
    tw, td = T[f]['any']
    out['examples'] = dict(typical_fire=dict(day=ex['day'], t=ex['t'], d=ex['d'], y=float(y[f]), score=float(s[f]), distance=0.,
                                              key_inputs={c: float(P[c].iat[f]) for c in KEY}),
                           **{f'twin_{n + 1}': dict(day=str(P.day.iat[tw[n]]), t=float(P.t.iat[tw[n]]), d=float(P.d.iat[tw[n]]), y=float(y[tw[n]]), score=float(s[tw[n]]),
                                                    distance=float(td[n]), key_inputs={c: float(P[c].iat[tw[n]]) for c in KEY}) for n in range(2)})
    out['examples_typical_fire_twins_move_rate'] = float(y[tw].mean())
    TR.dump(fn, out)
    print(json.dumps({k: v for k, v in out.items() if k != 'examples'}, indent=1))
    print(json.dumps(out['examples'], indent=1)[:1500])


def examples(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    src = Path(od) if od else H
    fn = src / 'visual_twins_examples.json'
    C.require(not fn.exists(), 'visual_twins_examples.json exists (write-once)')
    ex = json.loads((src / 'visual_twins.json').read_text())['examples']
    out = {}
    for nm, e in ex.items():
        C.require(e['day'] in DAYS, 'an undeclared day')
        w = SVN.window(e['day'], e['t'], e['d'])
        out[nm] = dict(e, **w)
        print(nm, e['day'], 'y', e['y'], 'distance', round(e['distance'], 3), 'first move', w['first_move_their_way_ms'], flush=True)
    TR.dump(fn, out)


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    {'read': read, 'examples': examples}[cmd](a[0] if a else None)
