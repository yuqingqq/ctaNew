"""GAIN / LOSS DIAGNOSTIC (2026-11-17, session aea5bfb9; USER "Go" on the next step after GAIN-LOSS's G3 NOT-GENERALIZED: where the
out-of-sample loss shortfall sits and whether Oct23-25 cover those states - state representation and training coverage; no model change).
Declared before any run; descriptive with declared predicates.

DATA: the GAIN-LOSS run (fe325d18): training decisions Oct23-25 (27,236; the joint-moment labels and 87 inputs), set L (every market of
Oct26-27, 17,859) with its GL predictions; the GL heads (unchanged) for in-sample predictions on the training decisions.
DIAGNOSTICS: (1) DAY LEVELS: realized loss, severe excess and gain per decision and their frequencies, by day (Oct23-27) and kind.
(2) NEXT-DAY TEST: the same GL loss recipe (the l1 and l2 hurdles; pinned LightGBM, 200 rounds) refitted on two training days and scored on
the third (3 folds): predicted / realized loss and severe totals on the held-out day, by kind. (3) LOCATION: on set L, predicted / realized
loss and the share of the total shortfall by stratum - contract minute, price band (|mid - 50|), spread, liquidity-scale tercile (training
cut points), side, rung, queue ahead (0 / > 0), own position band - beside the in-sample ratio of the same stratum on the training days.
(4) COVERAGE: a LightGBM classifier separating training from set-L decisions on the 87 inputs (5 folds grouped by market, pinned params, 200
rounds; out-of-fold scores): AUC, the inputs carrying the separation (gain), and set L's predicted / realized loss by quintile of that score.
PREDICATES (declared): P1 LOSSES-GREW: realized loss per decision on Oct26 and on Oct27 > 1.15 x the Oct23-25 mean, for both kinds.
 P2 NEXT-DAY-UNDERPREDICTION: held-out training-day loss ratio (pooled kinds) < 0.90 on >= 2 of 3 days. P3 COVERAGE-GAP: separation AUC >= 0.65
 AND set-L loss ratio in the least-familiar quintile below the most-familiar quintile's by > 0.15. P4 SHORTFALL-IN-FAMILIAR-STATES: set-L
 loss ratio < 0.85 in the two most-familiar quintiles pooled.
CONTROLS (any failure -> no output): the GL heads' hashes equal the GAIN-LOSS fit receipt; set L's hash equals its evaluation receipt; the
 training population equals the fit receipt's n; fold refits return finite predictions with probabilities in [0, 1]; every row receives
 exactly one out-of-fold score; markets never span folds.
LIMITS: consumed days; simulated labels; descriptive - no model change, no selection.
Usage (run_light, PF_SLOT=claude): pf_gain_loss_diag.py run
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_gain_loss as GL                                 # noqa: E402

J, JF, F, C, pd = GL.J, GL.JF, GL.F, GL.C, GL.pd
OUT = GL.OUT / 'diag'


def loss_heads(train, feats, out):
    g, l1, l2 = GL.components(train.w.values); X = train[feats].to_numpy(np.float32); arts = {}
    for c, y in (('l1', l1), ('l2', l2)):
        pos = y > 1e-9
        for head, Xh, yh, obj in ((f'{c}_p', X, pos.astype(float), 'binary'), (f'{c}_amt', X[pos], y[pos] / GL.SCALE, 'regression')):
            path = out / f'{head}.txt'; J.FM.fit(Xh, yh, feats, path, obj); arts[head] = str(path)
    return arts


def loss_predict(frame, feats, arts):
    import lightgbm as lgb
    X = frame[feats].to_numpy(np.float32); out = {}
    for c in ('l1', 'l2'):
        p = lgb.Booster(model_file=arts[f'{c}_p']).predict(X, num_threads=1)
        amt = np.maximum(lgb.Booster(model_file=arts[f'{c}_amt']).predict(X, num_threads=1), 0.) * GL.SCALE
        if c == 'l1':
            amt = np.minimum(amt, GL.SEVERE)
        C.require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() and np.isfinite(amt).all(), 'fold prediction contract')
        out[c] = p * amt
    return out


def strata(df, cuts):
    s = {}
    s['minute'] = df.phase.astype(int).astype(str)
    s['price |mid-50|'] = pd.cut((df.mid - 50.).abs(), [-0.1, 10, 25, 40, 50.1], labels=['<10', '10-25', '25-40', '>=40']).astype(str)
    s['spread'] = np.where(df.pf_spread <= 1.5, '1', np.where(df.pf_spread <= 2.5, '2', '3+'))
    s['scale tercile'] = pd.cut(df.jm_scale, [-np.inf, cuts[0], cuts[1], np.inf], labels=['low', 'mid', 'high']).astype(str)
    s['side'] = np.where(df.side == 0, 'bid', 'ask')
    s['kind'] = df.kind.astype(str)
    s['rung'] = np.where(df.jm_rung == 0, '0', np.where(df.jm_rung <= 2, '1-2', '3-5'))
    s['queue ahead'] = np.where(df.jm_ahead <= 1e-12, '0', '>0')
    s['own position'] = pd.cut(df.jm_known_position, [-np.inf, -0.25, 0.25, np.inf], labels=['short side', 'flat', 'long side']).astype(str)
    return s


def run():
    start = time.time(); OUT.mkdir(parents=True, exist_ok=False)
    fr = json.loads((GL.OUT / 'fit' / 'receipt.json').read_text()); arts = fr['artifacts']
    for a_ in arts.values():
        C.require(C.sha(a_['path']) == a_['sha256'], 'GL head changed')
    er = json.loads((GL.OUT / 'eval' / 'receipt.json').read_text())
    C.require(C.sha(GL.OUT / 'eval' / 'set_L.parquet') == er['outputs']['set_L.parquet'], 'set L changed')
    T = pd.concat([JF.read(d)[0] for d in F.TRAIN], ignore_index=True); T['w'] = T.y_w.astype(float); T['day'] = T.day.astype(str)
    C.require(len(T) == fr['n'], 'training population differs from the fit')
    E = pd.read_parquet(GL.OUT / 'eval' / 'set_L.parquet'); E['day'] = E.day.astype(str)
    for df in (T, E):
        g, l1, l2 = GL.components(df.w.values); df['g'] = g; df['l1'] = l1; df['l2'] = l2; df['loss'] = l1 + l2
    ins = GL.predict(T, 'full', arts); T['p_loss'] = ins['l1'] + ins['l2']; T['p_l2'] = ins['l2']
    E['p_loss'] = E.GL_l1 + E.GL_l2; E['p_l2'] = E.GL_l2
    feats = GL.feature_list('full'); L = [f'GAIN / LOSS DIAGNOSTIC (no model change); as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}', '']
    # (1) day levels
    L.append('PART 1: realized c per decision by day and kind: loss | severe excess | gain || P(loss > 0) | P(loss > 30 c) | n')
    lev = {}
    for (day, kind), g in pd.concat([T.assign(set='train'), E.assign(set='eval')]).groupby(['day', 'kind']):
        lev[(day, kind)] = g.loss.mean()
        L.append(f'  {day} {kind:<6} {g.loss.mean():6.2f} | {g.l2.mean():6.2f} | {g.g.mean():5.2f} || {np.mean(g.loss > 1e-9):5.1%} | {np.mean(g.l2 > 1e-9):5.1%} | {len(g):5,}')
    p1 = all(lev[(d, k)] > 1.15 * np.mean([lev[(t, k)] for t in F.TRAIN]) for d in GL.EVAL_DAYS for k in ('cancel', 'post'))
    # (2) next-day test
    L += ['', 'PART 2: leave one training day out - loss heads refitted on the other two days: predicted / realized (ratio) on the held-out day',
          '  (reference: the 3-day heads on Oct26 / Oct27 in GAIN-LOSS PART 3)']
    ratios = {}
    for h in F.TRAIN:
        tr = T[T.day != h]; te = T[T.day == h]; d = OUT / f'fold_{h}'; d.mkdir()
        a2 = loss_heads(tr, feats, d); pr = loss_predict(te, feats, a2); pl = pr['l1'] + pr['l2']
        ratios[h] = float(pl.sum() / te.loss.sum())
        parts = []
        for kind in ('cancel', 'post'):
            mk = (te.kind == kind).to_numpy()
            parts.append(f'{kind} loss x{pl[mk].sum() / te.loss[mk].sum():.2f}, severe x{pr["l2"][mk].sum() / max(te.l2[mk].sum(), 1e-9):.2f}')
        L.append(f'  held out {h}: pooled loss x{ratios[h]:.2f} | ' + ' | '.join(parts) + f' (train {len(tr):,}, held out {len(te):,})')
    p2 = sum(r < 0.90 for r in ratios.values()) >= 2
    # (3) location
    cuts = np.quantile(T.jm_scale, [1 / 3, 2 / 3]); sT = strata(T, cuts); sE = strata(E, cuts)
    short_total = float(E.loss.sum() - E.p_loss.sum())
    L += ['', f'PART 3: set L by stratum - n | realized loss c/decision | predicted / realized (ratio) | share of the total shortfall ({short_total:,.0f} c) '
          '|| training in-sample ratio (n)']
    for name in sE:
        for lvl in sorted(set(sE[name])):
            me = sE[name] == lvl; mt = sT[name] == lvl; ge = E[me]; gt = T[mt]
            if len(ge) == 0:
                continue
            r_e = ge.p_loss.sum() / max(ge.loss.sum(), 1e-9); r_t = gt.p_loss.sum() / max(gt.loss.sum(), 1e-9) if len(gt) else float('nan')
            L.append(f'  {name:<15} {lvl:<11} n {len(ge):5,} | {ge.loss.mean():6.2f} | x{r_e:.2f} | {(ge.loss.sum() - ge.p_loss.sum()) / short_total:6.1%} || x{r_t:.2f} ({len(gt):,})')
    # (4) coverage
    import lightgbm as lgb
    A_ = pd.concat([T[feats].assign(y=0., grp=T.day + ':' + T.m.astype(str)), E[feats].assign(y=1., grp=E.day + ':' + E.m.astype(str))], ignore_index=True)
    groups = A_.grp.unique(); rng = np.random.default_rng(20261117); fold_of = dict(zip(groups, rng.integers(0, 5, len(groups))))
    fold = A_.grp.map(fold_of).to_numpy(); score = np.full(len(A_), np.nan); gain = np.zeros(len(feats))
    for k in range(5):
        tr = fold != k; te = fold == k
        m_ = lgb.train({**C.V.PARAMS, 'objective': 'binary'}, lgb.Dataset(A_.loc[tr, feats].to_numpy(np.float32), label=A_.y[tr].to_numpy()), num_boost_round=200)
        score[te] = m_.predict(A_.loc[te, feats].to_numpy(np.float32), num_threads=1); gain += m_.feature_importance(importance_type='gain')
    C.require(np.isfinite(score).all(), 'every row needs one out-of-fold score')
    C.require(A_.groupby('grp').apply(lambda z: len(set(fold[z.index]))).max() == 1, 'a market spans folds')
    auc = JF.binary_metrics(A_.y.to_numpy(), score)['auc']
    top = sorted(zip(feats, gain), key=lambda z: -z[1])[:10]
    L += ['', f'PART 4: separating Oct23-25 from set L on the 87 inputs (5 folds by market): AUC {auc:.3f}; inputs carrying it (gain share): ' +
          ', '.join(f'{n_} {g_ / gain.sum():.1%}' for n_, g_ in top)]
    for n_, _ in top[:6]:
        L.append(f'    {n_:<22} training median {np.nanmedian(T[n_]):9.3f} (p10 {np.nanquantile(T[n_], .1):8.3f}, p90 {np.nanquantile(T[n_], .9):8.3f}) | set L median '
                 f'{np.nanmedian(E[n_]):9.3f} (p10 {np.nanquantile(E[n_], .1):8.3f}, p90 {np.nanquantile(E[n_], .9):8.3f})')
    se = score[len(T):]; E['fam'] = pd.qcut(pd.Series(se).rank(method='first'), 5, labels=False).to_numpy()
    L.append('  set L by quintile of the separation score (Q1 most training-like): n | realized loss c/decision | predicted / realized loss | severe ratio')
    qr = {}
    for q in range(5):
        g = E[E.fam == q]; qr[q] = g.p_loss.sum() / max(g.loss.sum(), 1e-9)
        L.append(f'    Q{q + 1} n {len(g):5,} | {g.loss.mean():6.2f} | x{qr[q]:.2f} | x{g.p_l2.sum() / max(g.l2.sum(), 1e-9):.2f}')
    fam2 = E[E.fam <= 1]; r_fam = fam2.p_loss.sum() / max(fam2.loss.sum(), 1e-9)
    p3 = auc >= 0.65 and (qr[0] - qr[4]) > 0.15
    p4 = r_fam < 0.85
    L += ['', f'P1 LOSSES-GREW: {"YES" if p1 else "NOT-SHOWN"} (Oct26 / Oct27 vs Oct23-25 mean, both kinds, > 1.15x)',
          f'P2 NEXT-DAY-UNDERPREDICTION: {"YES" if p2 else "NOT-SHOWN"} (held-out training-day ratios ' + ', '.join(f'{h} x{r:.2f}' for h, r in ratios.items()) + ')',
          f'P3 COVERAGE-GAP: {"YES" if p3 else "NOT-SHOWN"} (AUC {auc:.3f}; most- vs least-familiar quintile ratio x{qr[0]:.2f} vs x{qr[4]:.2f})',
          f'P4 SHORTFALL-IN-FAMILIAR-STATES: {"YES" if p4 else "NOT-SHOWN"} (two most-familiar quintiles x{r_fam:.2f})', '',
          'LIMITS: consumed days; simulated labels; descriptive - no model change, no selection']
    (OUT / 'gain_loss_diag.txt').write_text('\n'.join(L) + '\n')
    C.dump(OUT / 'gain_loss_diag.json', dict(asof=C.stamp(), predicates=dict(P1=p1, P2=p2, P3=p3, P4=p4), next_day_ratios=ratios, separation_auc=auc,
                                             familiarity_ratios={f'Q{q + 1}': v for q, v in qr.items()}, seconds=time.time() - start))
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    C.require(len(sys.argv) > 1 and sys.argv[1] == 'run', 'usage: pf_gain_loss_diag.py run')
    run()
