"""Post hoc first-fold diagnosis; no model fitting, filtering or candidate selection.

Declared after inspecting the first fold's error/importance receipt. Frozen
comparison continues unchanged. Only completed saved Oct25/26 training and
Oct27 evaluation artifacts are read. Tail partitions explain errors; they are
not permissions to remove losses. Bias removal is a test-outcome oracle only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_wait_post_state as N

DAY = '20261027'
TARGETS = ('qpost', 'qwait', 'delta', 'local_post', 'external_post', 'local_delta', 'external_delta')


def arrays(y, w):
    y = np.asarray(y, float); w = np.asarray(w, float)
    if y.ndim != 1 or y.shape != w.shape or not len(y) or not np.isfinite([y, w]).all() or np.any(w <= 0):
        raise ValueError('invalid weighted population')
    return y, w


def moments(y, w):
    y, w = arrays(y, w); mean = float(np.average(y, weights=w))
    return mean, float(np.average((y - mean) ** 2, weights=w))


def tail(y, w):
    y, w = arrays(y, w); mean, var = moments(y, w)
    n = max(1, int(np.ceil(len(y) * .01))); largest = np.argsort(abs(y))[-n:]
    sq = w * y * y; centered = w * (y - mean) ** 2
    return dict(n=len(y), effective_n=N.S.effective(w), mean=mean, std=var ** .5,
                zero_fraction=float(np.mean(abs(y) <= 1e-10)), weighted_zero_fraction=float(np.average(abs(y) <= 1e-10, weights=w)),
                top1pct_n=n, top1pct_squared_target_share=float(sq[largest].sum() / sq.sum()) if sq.sum() else 0.,
                top1pct_variance_share=float(centered[largest].sum() / centered.sum()) if centered.sum() else 0.,
                absolute_quantiles=dict(zip(('p50', 'p90', 'p95', 'p99', 'max'), map(float, np.quantile(abs(y), [.5, .9, .95, .99, 1.])))))


def error_parts(y, p, w, prior):
    y, w = arrays(y, w); p = np.asarray(p, float)
    if p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError('invalid prediction')
    bias, centered = moments(p - y, w); mse = centered + bias * bias
    direct = float(np.average((p - y) ** 2, weights=w)); den = float(np.average((y - prior) ** 2, weights=w))
    if not np.isclose(mse, direct, rtol=1e-10, atol=1e-10):
        raise ValueError('bias/variance decomposition failed')
    return dict(n=len(y), effective_n=N.S.effective(w), bias=bias, mse=mse, rmse=mse ** .5,
                bias_squared_share=bias * bias / mse if mse else 0.,
                r2_vs_training_mean=1 - mse / den if den else None,
                oracle_demeaned_r2=1 - centered / den if den else None,
                oracle_note='Uses test outcomes; diagnostic only, not a deployable calibration.')


def decomposition(local, external, total, w):
    local, w = arrays(local, w); external = np.asarray(external, float); total = np.asarray(total, float)
    if not np.allclose(local + external, total, rtol=0, atol=1e-8):
        raise ValueError('target decomposition mismatch')
    lm, lv = moments(local, w); em, ev = moments(external, w); _, tv = moments(total, w)
    cov = float(np.average((local - lm) * (external - em), weights=w))
    if not np.isclose(lv + ev + 2 * cov, tv, rtol=1e-10, atol=1e-8):
        raise ValueError('variance decomposition mismatch')
    return dict(n=len(local), local_variance=lv, external_variance=ev, twice_covariance=2 * cov, total_variance=tv,
                external_to_local_variance=ev / lv if lv else None,
                correlation=cov / (lv * ev) ** .5 if lv > 0 and ev > 0 else None)


def rank_corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return None
    return float(np.corrcoef(pd.Series(x[ok]).rank(), pd.Series(y[ok]).rank())[0, 1])


def feature_audit(train, test):
    constants = []; duplicates = []; seen = {}; records = []
    for name in N.FEATURES:
        a = train[name].to_numpy(float); b = test[name].to_numpy(float)
        aa = a[np.isfinite(a)]; bb = b[np.isfinite(b)]
        if not len(aa) or (np.ptp(aa) == 0 and len(aa) == len(a)):
            constants.append(name)
        key = hashlib.sha256(np.where(np.isnan(a), np.inf, a).tobytes()).hexdigest()
        if key in seen:
            duplicates.append([name, seen[key]])
        else:
            seen[key] = name
        r = dict(feature=name, train_n=len(a), test_n=len(b), train_missing=int(np.isnan(a).sum()), test_missing=int(np.isnan(b).sum()),
                 ks=float(ks_2samp(aa, bb, method='asymp').statistic) if len(aa) and len(bb) else None,
                 outside_train_range=int(((bb < aa.min()) | (bb > aa.max())).sum()) if len(aa) and len(bb) else None,
                 relationships=[])
        for part in (train, test):
            for day, f in part.groupby('day'):
                r['relationships'].append(dict(day=day, n=len(f), post_rank_ic=rank_corr(f[name], f.qpost), delta_rank_ic=rank_corr(f[name], f.delta)))
        records.append(r)
    return dict(train_n=len(train), test_n=len(test), feature_count=len(N.FEATURES), constant_or_all_missing=constants,
                exact_duplicate_pairs=duplicates, largest_shifts=sorted(records, key=lambda r: r['ks'] or 0., reverse=True)[:20],
                selected_known_concerns=[r for r in records if r['feature'] in ('basis_bps', 'sigma', 'gap', 'bn_r3000', 'own_book_liquidity_log', 'opp_book_liquidity_log')])


def diagnose(base, out):
    out.mkdir(); train = pd.concat([N.read_rows(base, d) for d in N.M.FOLDS[DAY]], ignore_index=True)
    test = N.read_rows(base, DAY); path = base / ('fit_' + DAY); receipt = N.E.read_json(path / 'receipt.json')
    if receipt['predictions_sha256'] != N.M.digest(path / 'predictions.parquet'):
        raise ValueError('prediction digest mismatch')
    predictions = pd.read_parquet(path / 'predictions.parquet')
    noise = []; components = []; populations = []; features = []; errors = []; partitions = []; calibration = []
    for side in (0, 1):
        tr = train[train.side == side]; te = test[test.side == side]; wt = N.S.weights(tr); w = N.S.weights(te)
        for split, f, ww in (('train', tr, wt), ('test', te, w)):
            grouped = pd.DataFrame(dict(day=f.day, market=f.market, weight=ww)).groupby(['day', 'market']).weight.sum()
            populations.append(dict(side=side, split=split, n=len(f), n_days=f.day.nunique(), n_windows=len(grouped),
                                    weight_effective_n=N.S.effective(ww), grouped_weight_effective_n=N.S.effective(grouped.to_numpy()),
                                    warning='Weight effective n is a concentration diagnostic, not independent sample size.',
                                    initial_n=int((f.generation == 0).sum()), later_n=int((f.generation > 0).sum())))
            noise.extend(dict(side=side, split=split, target=h, **tail(f[h], ww)) for h in TARGETS)
            for h, l, e in (('qpost', 'local_post', 'external_post'), ('delta', 'local_delta', 'external_delta')):
                components.append(dict(side=side, split=split, target=h, **decomposition(f[l], f[e], f[h], ww)))
        features.append(dict(side=side, **feature_audit(tr, te)))
        stale_thresholds = {name: float(tr[name].quantile(.95)) for name in ('frame_age', 'bn_age')}
        stale = np.zeros(len(te), bool)
        for name, threshold in stale_thresholds.items():
            stale |= ~np.isfinite(te[name].to_numpy()) | (te[name].to_numpy() > threshold)
        for arm in N.ARMS:
            saved = predictions[(predictions.side == side) & (predictions.arm == arm)]
            merged = te.merge(saved[N.KEYS + ['post_hat', 'delta_hat', 'post_action']], on=N.KEYS, how='left', validate='one_to_one')
            model = N.load_forecast(base, DAY, arm)
            pp, dd = model.predict(side, merged[N.FEATURES].to_numpy(np.float32))
            if not np.allclose(pp, merged.post_hat, atol=1e-9, rtol=0) or not np.allclose(dd, merged.delta_hat, atol=1e-9, rtol=0):
                raise ValueError('saved inference parity failed')
            for head, target, pred in (('post', 'qpost', pp), ('delta', 'delta', dd)):
                prior = float(np.average(tr[target], weights=wt)); y = merged[target].to_numpy()
                errors.append(dict(side=side, arm=arm, target=target, **error_parts(y, pred, w, prior)))
                threshold = float(np.quantile(abs(tr[target]), .99)); extreme = abs(y) > threshold
                masks = {'train_p99_outcome_tail': extreme, 'within_train_p99_outcome': ~extreme,
                         'prior_p95_age_or_missing': stale, 'within_prior_p95_age': ~stale}
                all_sse = np.sum(w * (pred - y) ** 2)
                for name, mask in masks.items():
                    if not mask.any():
                        continue
                    partitions.append(dict(side=side, arm=arm, target=target, partition=name,
                          threshold=threshold if 'outcome' in name else stale_thresholds,
                          weighted_sse_share=float(np.sum(w[mask] * (pred[mask] - y[mask]) ** 2) / all_sse) if all_sse else 0.,
                          **error_parts(y[mask], pred[mask], w[mask], prior)))
            selected = merged.post_action.to_numpy(bool)
            for name, mask in (('selected_post', selected), ('withheld', ~selected)):
                if mask.any():
                    calibration.append(dict(side=side, arm=arm, group=name, n=int(mask.sum()), effective_n=N.S.effective(w[mask]),
                         predicted_post=float(np.average(pp[mask], weights=w[mask])), actual_post=float(np.average(merged.qpost.to_numpy()[mask], weights=w[mask])),
                         predicted_delta=float(np.average(dd[mask], weights=w[mask])), actual_delta=float(np.average(merged.delta.to_numpy()[mask], weights=w[mask]))))
    source_receipts = {d: N.E.read_json(base / ('rows_' + d) / 'receipt.json') for d in (*N.M.FOLDS[DAY], DAY)}
    result = dict(asof=N.M.now(), status='complete_posthoc_first_fold_diagnosis', new_candidates=0, forward_validation=False,
                  train_dates=list(N.M.FOLDS[DAY]), test_dates=[DAY], n_train=len(train), n_test=len(test), source_receipts=source_receipts,
                  populations=populations, target_noise=noise, component_variance=components, features=features,
                  error_decomposition=errors, diagnostic_partitions=partitions, selection_calibration=calibration,
                  fit_receipt_sha256=N.M.digest(path / 'receipt.json'),
                  limitations=['One evaluation day and two training days. No uncertainty intervals or generalizable noise-filter claim.',
                               'Large adverse outcomes may be the very risk the model must predict. They are never deleted.',
                               'Oracle bias removal and outcome-tail partitions use realized outcomes; they are not deployable filters.',
                               'Marginal feature shifts and rank correlations do not establish conditional instability or causation.',
                               'No new candidate, threshold, feature subset, loss function or label horizon was selected.'])
    N.write(out / 'result.json', result)


class Checks(unittest.TestCase):
    def test_bias_and_oracle_controls(self):
        y = np.arange(100.); w = np.ones(100)
        a = error_parts(y, y + 3., w, y.mean())
        self.assertAlmostEqual(a['bias_squared_share'], 1.)
        self.assertAlmostEqual(a['oracle_demeaned_r2'], 1.)
        b = error_parts(y, -y, w, y.mean())
        self.assertLess(b['oracle_demeaned_r2'], 0.)
        with self.assertRaises(ValueError): error_parts(y, y[:-1], w, 0.)

    def test_component_cancellation_and_bad_sign(self):
        a = np.arange(100.); b = -a + np.sin(a); w = np.ones(100)
        r = decomposition(a, b, a + b, w)
        self.assertLess(r['twice_covariance'], 0.)
        with self.assertRaises(ValueError): decomposition(a, b, a - b, w)

    def test_tail_preserves_real_losses_and_bad_weights_refused(self):
        y = np.r_[np.zeros(99), -100.]; w = np.ones(100)
        r = tail(y, w); self.assertEqual(r['top1pct_squared_target_share'], 1.)
        self.assertEqual(r['zero_fraction'], .99); self.assertEqual(r['mean'], -1.)
        with self.assertRaises(ValueError): tail(y, -w)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['selftest', 'diagnose'])
    p.add_argument('--base', type=Path, default=N.DEFAULT_OUT / 'full'); p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.command == 'selftest':
        r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
        if not r.wasSuccessful(): raise SystemExit(1)
    else: diagnose(a.base, a.out)
