"""Two post-diagnosis cleanup candidates, no row filtering or target clipping.

Uses only the repaired state's frozen labels. CLEAN_TREE keeps pinned LightGBM;
CLEAN_RIDGE uses the prior signal study's fixed alpha100. Compact economic
features, known-liquidity queue scaling, train-only bounds/indicators and exact
constant/duplicate removal. All outcomes retained, inverse inclusion weights
in fitting AND scoring, identical500-draw action-matched null. Development only.
"""
import argparse
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_wait_post_state as N

ARMS = ('CLEAN_TREE', 'CLEAN_RIDGE')
ALPHA = 100.
FIELDS = ('remaining_lots', 'pending_place_lots', 'pending_cancel_lots', 'filled_lots',
          'price_gap_mean', 'oldest_age_log', 'queue_lots', 'queue_unknown', 'empty_age_log')


def compact(f):
    g = N.S.S.compact(f).drop(columns=['basis_bps', 'own_book_log', 'other_book_log'])
    # Basis was already unstable in the earlier development audits. Do not
    # retain duplicate absolute public sizes alongside relative liquidity.
    scale_lots = np.expm1(f.own_book_liquidity_log.to_numpy(float))
    scale_lots = np.where(scale_lots > 0, scale_lots, np.nan)
    for name in ('shown_known', 'size_better'):
        v = f[name].to_numpy(float) / N.M.SIZE / scale_lots
        g[name + '_lots_log'] = np.where(v >= 0, np.log1p(np.maximum(v, 0)), np.nan)
    extra = {}
    for side in ('own', 'opp'):
        for d in range(N.M.NR):
            for name in FIELDS:
                key = f'{side}_{d}_{name}'; v = f[key].to_numpy(float)
                extra[key] = v / scale_lots if name == 'queue_lots' else v
        for name in N.BOOK_FIELDS:
            if name != 'liquidity_log':
                key = f'{side}_book_{name}'; extra[key] = f[key].to_numpy(float)
    for name in ('start_known', 'own_empty_count', 'opp_empty_count'):
        extra[name] = f[name].to_numpy(float)
    return pd.concat([g, pd.DataFrame(extra, index=f.index)], axis=1)


def ridge(x, y, w, alpha=ALPHA):
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    if x.ndim != 2 or len(x) != len(y) or y.shape != w.shape or not np.isfinite(x).all() or not np.isfinite([y, w]).all() or np.any(w <= 0) or alpha <= 0:
        raise ValueError('invalid ridge training inputs')
    w = w / w.mean(); xm = np.average(x, weights=w, axis=0); ym = float(np.average(y, weights=w))
    a = x - xm; yy = y - ym
    coef = np.linalg.solve(a.T @ (a * w[:, None]) + alpha * np.eye(x.shape[1]), a.T @ (w * yy))
    return dict(mean_x=xm.tolist(), mean_y=ym, coef=coef.tolist(), alpha=alpha)


def predict_ridge(model, x):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != len(model['coef']) or not np.isfinite(x).all():
        raise ValueError('invalid ridge inference inputs')
    return model['mean_y'] + (x - model['mean_x']) @ np.asarray(model['coef'])


def probes(base, out):
    import lightgbm as lgb
    out.mkdir(); cells = []; parts = []; transforms = []; params = dict(N.E.PD.LGB_PARAMS, objective='regression')
    model_files = {}; input_receipts = {}
    for day in N.M.TESTS:
        train = pd.concat([N.read_rows(base, d) for d in N.M.FOLDS[day]], ignore_index=True); test = N.read_rows(base, day)
        for d in (*N.M.FOLDS[day], day): input_receipts[d] = N.M.digest(base / ('rows_' + d) / 'receipt.json')
        for side in (0, 1):
            tr = train[train.side == side]; te = test[test.side == side]
            w = N.S.weights(tr); tw = N.S.weights(te)
            processor = N.S.S.Processor().fit(compact(tr)); x = processor.transform(compact(tr)); z = processor.transform(compact(te))
            transforms.append(dict(day=day, side=side, train_dates=list(N.M.FOLDS[day]), train_n=len(tr), test_n=len(te),
                                   processor=processor.metadata(), kept=processor.kept))
            for arm in ARMS:
                predictions = []; heads = []
                for target in ('qpost', 'delta'):
                    y = tr[target].to_numpy(float); prior = float(np.average(y, weights=w))
                    if arm == 'CLEAN_TREE':
                        model = lgb.train(params, lgb.Dataset(x, y, weight=w / w.mean(), feature_name=processor.output), N.E.PD.N_ROUNDS)
                        path = out / f'{day}_{side}_{target}_tree.txt'; model.save_model(str(path))
                        pred = model.predict(z); train_pred = model.predict(x)
                    else:
                        model = ridge(x, y, w); path = out / f'{day}_{side}_{target}_ridge.json'; N.write(path, model)
                        pred = predict_ridge(model, z); train_pred = predict_ridge(model, x)
                    model_files[path.name] = N.M.digest(path); predictions.append(pred)
                    heads.append(dict(target=target, train=N.S.metric(y, train_pred, w, prior), test=N.S.metric(te[target], pred, tw, prior),
                                      price_rung=N.S.metric(te[target], N.S.baseline(tr, te, target), tw, prior)))
                pp, dd = predictions
                f = te[N.KEYS + ['probe_t', 'price', 'qpost', 'qwait', 'delta']].copy()
                f['arm'] = arm; f['post_hat'] = pp; f['delta_hat'] = dd; f['post_action'] = (pp > 0) & (dd >= 0); f['sampling_weight'] = tw
                parts.append(f); cells.append(dict(day=day, side=side, arm=arm, heads=heads, n=len(te), effective_n=N.S.effective(tw)))
        print('Completed cleanup fold', day, N.M.now(), flush=True)
        if N.E.rss_mib() > 2048: raise RuntimeError('RSS self-guard exceeded')
    predictions = pd.concat(parts, ignore_index=True); predictions.to_parquet(out / 'predictions.parquet', index=False)
    tests = [dict(arm=a, side=int(s), **N.S.matched(g)) for (a, s), g in predictions.groupby(['arm', 'side'])]
    pooled = []
    for arm in ARMS:
        heads = {}
        for target in ('qpost', 'delta'):
            hh = [h['test'] for c in cells if c['arm'] == arm for h in c['heads'] if h['target'] == target]
            heads[target] = dict(n=sum(h['n'] for h in hh), r2_vs_fold_prior_mean=1 - sum(h['sse'] for h in hh) / sum(h['baseline_sse'] for h in hh))
        pooled.append(dict(arm=arm, **heads))
    N.write(out / 'result.json', dict(asof=N.M.now(), new_candidates=2, state_study_candidates_total=5,
             dates=list(N.M.TESTS), n_test_actions=len(predictions) // len(ARMS), cells=cells, pooled=pooled,
             matched_tests=tests, tests_passed=sum(t['passes'] for t in tests), preprocessing=transforms,
             models=model_files, input_receipts=input_receipts, predictions_sha256=N.M.digest(out / 'predictions.parquet'),
             params=params, rounds=N.E.PD.N_ROUNDS, ridge_alpha=ALPHA, forward_validation=False, portfolio_evaluated=False,
             limitations=['Post hoc candidates after first-fold diagnosis; consumed dates, no forward claim.',
                          'No rows or labels removed. Feature bounds retain missing/extreme indicators; rare losses remain raw.',
                          'No temporal smoothing across unrelated action rows. Known-time liquidity scaling only.',
                          'Same neutral KEEP terminal targets; compact processing cannot repair that policy mismatch.']))


class Checks(unittest.TestCase):
    def test_ridge_positive_and_bad_schema(self):
        x = np.arange(100., dtype=float)[:, None]; y = 2 * x[:, 0] - 4; w = np.ones(100)
        m = ridge(x, y, w); p = predict_ridge(m, x)
        self.assertGreater(np.corrcoef(p, y)[0, 1], .999)
        with self.assertRaises(ValueError): ridge(x, y, -w)
        with self.assertRaises(ValueError): predict_ridge(m, np.ones((10, 2)))

    def test_training_only_processing_and_no_label_mutation(self):
        f = pd.DataFrame(dict(a=np.arange(100.), b=np.arange(100.), c=np.ones(100)))
        p = N.S.S.Processor().fit(f); before = copy.deepcopy(p.metadata()); z = f.copy(); z.a = 1e9
        p.transform(z); self.assertEqual(before, p.metadata())
        self.assertTrue(any('duplicate' in v for v in p.removed.values()))
        self.assertTrue(any('constant' in v for v in p.removed.values()))
        with self.assertRaises(ValueError): p.transform(z[['a']])
        self.assertNotEqual(before, N.S.S.Processor().fit(z).metadata())

    def test_public_liquidity_scaling_preserves_own_quantities(self):
        f = pd.DataFrame(np.ones((4, len(N.FEATURES))), columns=N.FEATURES)
        f.own_book_liquidity_log = np.log1p(10.); f.opp_book_liquidity_log = np.log1p(10.)
        a = compact(f); z = f.copy(); z.own_book_liquidity_log = np.log1p(20.); z.opp_book_liquidity_log = np.log1p(20.)
        for side in ('own', 'opp'):
            for d in range(N.M.NR): z[f'{side}_{d}_queue_lots'] *= 2.
        z.shown_known *= 2.; z.size_better *= 2.
        b = compact(z)
        np.testing.assert_allclose(a, b)
        self.assertEqual(a.own_0_remaining_lots.iloc[0], 1.)
        self.assertNotIn('basis_bps', a)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['selftest', 'probes']); p.add_argument('--out', type=Path)
    p.add_argument('--base', type=Path, default=N.DEFAULT_OUT / 'full'); a = p.parse_args()
    if a.command == 'selftest':
        r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
        if not r.wasSuccessful(): raise SystemExit(1)
    else: probes(a.base, a.out)
