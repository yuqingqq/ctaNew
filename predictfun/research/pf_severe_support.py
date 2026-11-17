"""SEVERE-EVENT PROBABILITY: CALIBRATION BY BAND AND TRAINING SUPPORT (2026-11-17, session aea5bfb9; USER "Ok, go" on: work on the
severe-event probability itself - check its calibration by predicted-probability band on past-only folds, and a proper coverage test:
for each test decision, how much nearby training support it has, measured among the training decisions themselves, and whether the
under-prediction grows where that support is thin; separate model error from coverage). Declared before any run; descriptive with
declared predicates; no model change.

MODELS: the past-only expanding folds of RUNNING-ERRORS (49d159e5; gain_loss_20261117/running/fold_<D>): day D in Oct24..Oct27 predicted by
the GAIN-LOSS heads fitted on every eligible decision of the days before D. Severe event = loss beyond 30 c; p = the fold's P(severe) head.
PART 1 CALIBRATION BY BAND: fixed bands of p [0, .005, .01, .02, .05, .10, .20, 1]: n, expected (sum p), observed, O/E - pooled forward
days and by kind, beside the same fold models' in-sample O/E on their own training days.
PART 2 TRAINING SUPPORT: per fold and kind, the 15 inputs with the largest gain in that fold's P(severe) head, standardized on the fold's
training decisions (missing -> 0). Support distance of a decision = distance to its 50th nearest training decision of the same kind; for
training decisions the reference distance excludes their own market (same day and market) - so a test decision (always another market)
is compared with how close training decisions sit to OTHER markets. Support bins by the training reference quantiles: <= p50, p50-p90,
p90-p99, > p99. kNN rate = the observed severe share among those 50 training neighbours. Per bin: share of test decisions, observed /
model expected (model O/E), observed / kNN expected (kNN O/E).
PREDICATES (declared): C1 COVERAGE-GAP: >= 5% of test decisions beyond the training p99 AND model O/E there exceeds the <= p50 bin's by > 0.2.
 C2 MODEL-ERROR: in the <= p90 bins pooled, kNN O/E within 0.9-1.1 AND |log kNN O/E| < |log model O/E| (the training neighbours hold the
 rate the model misses). C3 SHIFT: in the <= p90 bins pooled, kNN O/E > 1.15 AND model O/E > 1.15 (well-supported states had more severe
 events on the test days than their training neighbours). B1 LOW-P-SHORTFALL: pooled O/E over bands p < 0.02 exceeds the O/E over p >= 0.05
 by > 0.2.
CONTROLS (any failure -> no output): every fold's head files are the RUNNING-ERRORS ones and the Oct26 fold reproduces GAIN-LOSS's P(severe)
 exactly; fold training = days before D; the reference neighbour sets contain no decision of the same market (checked on every row); a
 planted error - the reference without the market exclusion - must give smaller reference distances (median) than the excluded reference.
LIMITS: consumed days; four predicted days, no interval (rule 8); kNN in 15 standardized inputs is a support measure, not a density
 estimate; simulated labels; the neutral decision population.
Usage (run_light, PF_SLOT=claude): pf_severe_support.py run
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_gain_loss as GL                                 # noqa: E402
import pf_gain_loss_running as RN                         # noqa: E402

C, pd = GL.C, GL.pd
OUT = GL.OUT / 'severe_support'
BANDS = (0., .005, .01, .02, .05, .10, .20, 1.000001)
K, EXTRA, NFEAT = 50, 60, 15


def knn(train_x, query_x, k, exclude=None, train_grp=None, query_grp=None):
    """Distance to the k-th neighbour, and the neighbour indices; optionally excluding neighbours of the query's own group."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(len(train_x), k + (EXTRA if exclude else 0))).fit(train_x)
    dist, idx = nn.kneighbors(query_x)
    if not exclude:
        return dist[:, k - 1], idx[:, :k]
    keep_d = np.empty(len(query_x)); keep_i = np.empty((len(query_x), k), int)
    for r in range(len(query_x)):
        ok = train_grp[idx[r]] != query_grp[r]
        C.require(ok.sum() >= k, 'not enough neighbours outside the market')
        keep_d[r] = dist[r][ok][k - 1]; keep_i[r] = idx[r][ok][:k]
        C.require(not (train_grp[keep_i[r]] == query_grp[r]).any(), 'a reference neighbour is in the same market')
    return keep_d, keep_i


def run():
    RN.selftest(); start = time.time(); OUT.mkdir(parents=True, exist_ok=False)
    import lightgbm as lgb
    fr = RN.load(); E0 = pd.read_parquet(GL.OUT / 'eval' / 'set_L.parquet'); feats = GL.feature_list('full')
    rows = []; ins = []; sup = []
    for D in RN.PRED_DAYS:
        before = [d for d in RN.DAYS if d < D]; C.require(max(before) < D, 'fold is not past-only')
        fd = RN.OUT / f'fold_{D}'; head = lgb.Booster(model_file=str(fd / 'full_l2_p.txt'))
        train = pd.concat([fr[d] for d in before], ignore_index=True); test = fr[D].copy()
        test['p'] = head.predict(test[feats].to_numpy(np.float32), num_threads=1); train['p'] = head.predict(train[feats].to_numpy(np.float32), num_threads=1)
        if D == '20261026':
            ref = E0[E0.day.astype(str) == D].set_index('uid').loc[test.uid]
            C.require(np.allclose(test.p.values, ref.GL_l2_p.values, rtol=0, atol=1e-12), 'the Oct26 fold does not reproduce GAIN-LOSS P(severe)')
        test['fold'] = D; train['fold'] = D; rows.append(test); ins.append(train)
        gain = head.feature_importance(importance_type='gain'); top = [feats[i] for i in np.argsort(-gain)[:NFEAT]]
        for kind in ('cancel', 'post'):
            tr = train[train.kind == kind].reset_index(drop=True); te = test[test.kind == kind].reset_index(drop=True)
            mu = tr[top].mean(); sd = tr[top].std().replace(0, 1.)
            xt = ((tr[top] - mu) / sd).fillna(0.).to_numpy(float); xq = ((te[top] - mu) / sd).fillna(0.).to_numpy(float)
            grp_t = (tr.day.astype(str) + ':' + tr.m.astype(str)).to_numpy()
            ref_d, _ = knn(xt, xt, K, exclude=True, train_grp=grp_t, query_grp=grp_t)            # reference: 50th neighbour in other markets (self excluded)
            ref_naive, _ = knn(xt, xt, K + 1)                                                       # planted error: no market exclusion
            C.require(np.median(ref_naive) < np.median(ref_d), 'the market exclusion did not change the reference (control)')
            q_d, q_i = knn(xt, xq, K)
            sev_t = tr.sev.to_numpy(); te['knn_rate'] = sev_t[q_i].mean(1); te['dist'] = q_d
            qs = np.quantile(ref_d, [.5, .9, .99]); te['bin'] = np.digitize(q_d, qs)
            te['ref_p50'], te['ref_p99'] = qs[0], qs[2]; sup.append(te)
            print(json.dumps(dict(stage='support', day=D, kind=kind, train=len(tr), test=len(te), ref_median=float(np.median(ref_d)), ref_naive_median=float(np.median(ref_naive)),
                                  test_median=float(np.median(q_d)), beyond_p99=float(np.mean(q_d > qs[2])), seconds=round(time.time() - start))), flush=True)
    T = pd.concat(rows, ignore_index=True); TR = pd.concat(ins, ignore_index=True); S = pd.concat(sup, ignore_index=True)
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    L = [f'SEVERE-EVENT PROBABILITY: calibration by band and training support (past-only folds; no model change); as of {asof}',
         'controls: fold heads = RUNNING-ERRORS; Oct26 fold reproduces GAIN-LOSS P(severe); reference neighbours exclude own market (checked); planted no-exclusion reference smaller', '',
         'PART 1: by band of predicted P(severe) - forward days pooled: n | expected | observed | O/E || in-sample (training days of each fold) O/E']
    bins = pd.cut(T.p, BANDS, right=False); bins_tr = pd.cut(TR.p, BANDS, right=False); oe_band = {}
    for b in bins.cat.categories:
        g = T[bins == b]; h = TR[bins_tr == b]
        if len(g) == 0:
            continue
        oe = g.sev.sum() / max(g.p.sum(), 1e-9); oe_band[str(b)] = (float(g.p.sum()), float(g.sev.sum()))
        L.append(f'  {str(b):<16} n {len(g):6,} | {g.p.sum():7.1f} | {int(g.sev.sum()):5d} | x{oe:.2f} || x{h.sev.sum() / max(h.p.sum(), 1e-9):.2f} (n {len(h):,})')
    L.append('  by kind and day (O/E): ' + ' | '.join(f'{k} {d} x{g.sev.sum() / g.p.sum():.2f}' for (k, d), g in T.groupby(['kind', 'fold'])))
    lo = T[T.p < .02]; hi = T[T.p >= .05]
    oe_lo = lo.sev.sum() / lo.p.sum(); oe_hi = hi.sev.sum() / hi.p.sum(); b1 = (oe_lo - oe_hi) > 0.2
    L += [f'  p < 0.02: O/E x{oe_lo:.2f} (n {len(lo):,}, {int(lo.sev.sum())} events) | p >= 0.05: x{oe_hi:.2f} (n {len(hi):,}, {int(hi.sev.sum())} events)', '',
          'PART 2: training support - bins by the training reference distance (50th neighbour, other markets): share of test decisions | observed rate | model mean p | kNN rate | model O/E | kNN O/E']
    names = {0: '<= p50', 1: 'p50-p90', 2: 'p90-p99', 3: '> p99'}; res = {}
    for b in range(4):
        g = S[S.bin == b]
        if len(g) == 0:
            L.append(f'  {names[b]:<8} share  0.0%'); continue
        mo = g.sev.sum() / max(g.p.sum(), 1e-9); ko = g.sev.sum() / max(g.knn_rate.sum(), 1e-9); res[b] = (mo, ko, len(g))
        L.append(f'  {names[b]:<8} share {len(g) / len(S):5.1%} | {g.sev.mean():.4f} | {g.p.mean():.4f} | {g.knn_rate.mean():.4f} | x{mo:.2f} | x{ko:.2f} (n {len(g):,}, events {int(g.sev.sum())})')
    for (d, k), g in S.groupby(['fold', 'kind']):
        L.append(f'    {d} {k:<6}: beyond p99 {np.mean(g.bin == 3):5.1%} | model O/E x{g.sev.sum() / g.p.sum():.2f} | kNN O/E x{g.sev.sum() / max(g.knn_rate.sum(), 1e-9):.2f}')
    w = S[S.bin <= 1]; mo_w = w.sev.sum() / w.p.sum(); ko_w = w.sev.sum() / max(w.knn_rate.sum(), 1e-9)
    beyond = float(np.mean(S.bin == 3))
    c1 = beyond >= 0.05 and 3 in res and 0 in res and (res[3][0] - res[0][0]) > 0.2
    c2 = 0.9 <= ko_w <= 1.1 and abs(np.log(ko_w)) < abs(np.log(mo_w))
    c3 = ko_w > 1.15 and mo_w > 1.15
    L += ['', f'  well-supported (<= p90) pooled: model O/E x{mo_w:.2f}, kNN O/E x{ko_w:.2f} (n {len(w):,})', '',
          f'B1 LOW-P-SHORTFALL: {"YES" if b1 else "NOT-SHOWN"} (p < 0.02 x{oe_lo:.2f} vs p >= 0.05 x{oe_hi:.2f})',
          f'C1 COVERAGE-GAP: {"YES" if c1 else "NOT-SHOWN"} (beyond training p99: {beyond:.1%} of test decisions' + (f'; model O/E there x{res[3][0]:.2f} vs <= p50 x{res[0][0]:.2f})' if 3 in res and 0 in res else ')'),
          f'C2 MODEL-ERROR: {"YES" if c2 else "NOT-SHOWN"} (well-supported: kNN O/E x{ko_w:.2f} vs model O/E x{mo_w:.2f})',
          f'C3 SHIFT: {"YES" if c3 else "NOT-SHOWN"} (well-supported: kNN O/E x{ko_w:.2f}, model O/E x{mo_w:.2f}; both > 1.15 required)', '',
          'LIMITS: consumed days; four predicted days, no interval; kNN support in 15 standardized inputs is a support measure, not a density; simulated labels; the neutral population']
    (OUT / 'severe_support.txt').write_text('\n'.join(L) + '\n')
    C.dump(OUT / 'severe_support.json', dict(asof=asof, predicates=dict(B1=b1, C1=c1, C2=c2, C3=c3), beyond_p99=beyond, well_supported=dict(model_oe=mo_w, knn_oe=ko_w),
                                             bands=oe_band, seconds=time.time() - start))
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    C.require(len(sys.argv) > 1 and sys.argv[1] == 'run', 'usage: pf_severe_support.py run')
    run()
