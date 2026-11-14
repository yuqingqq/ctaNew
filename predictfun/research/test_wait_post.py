"""Entry-only positive controls and known-bad refusals; synthetic tape only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import pf_wait_post as W
import pf_entry_value as E
from test_entry_value import context, Score


class Model(Score):
    def feature_importance(self, importance_type): return np.zeros(len(W.FEATURES))


def views(ctx):
    out = []; E.run(ctx, lambda v: True, observe=lambda v, a: out.append(v)); return out


class WaitPostTests(unittest.TestCase):
    def test_bid_and_ask_route_independently_and_rest_never_uses_entry_head(self):
        ctx = context(); vv = views(ctx); models = {0: Score(1.), 1: Score(-1.)}
        choose = W.policy(ctx, models)
        b = next(v for v in vv if v['side'] == 0 and v['kind'] == 'entry')
        a = next(v for v in vv if v['side'] == 1 and v['kind'] == 'entry')
        self.assertTrue(choose(b)); self.assertFalse(choose(a))
        result = E.run(ctx, choose)
        self.assertTrue(result['order_states'][(0, 0)]); self.assertFalse(result['order_states'][(1, 0)])
        with mock.patch.object(models[0], 'predict', side_effect=AssertionError('entry head consulted for resting order')):
            self.assertTrue(choose(next(v for v in vv if v['side'] == 0 and v['kind'] == 'rest')))
        models[0], models[1] = Score(-1.), Score(1.)
        self.assertFalse(choose(b)); self.assertTrue(choose(a))  # Detect reversed/shared routing.

    def test_signed_features_inventory_and_future_invariance(self):
        ctx = context(adverse=True); vv = views(ctx)
        b = next(v for v in vv if v['side'] == 0 and v['kind'] == 'entry')
        a = next(v for v in vv if v['side'] == 1 and v['kind'] == 'entry')
        for s in (0, 1): ctx[s]['X'][:, E.K.G32.index('bn_r100')] = 2.
        xb = W.features(ctx, dict(b, known_position=90.)); xa = W.features(ctx, dict(a, known_position=90.))
        self.assertEqual(xb[W.FEATURES.index('side_bn_r100')], 2.)
        self.assertEqual(xa[W.FEATURES.index('side_bn_r100')], -2.)
        self.assertEqual(xb[-1], 90.); self.assertEqual(xa[-1], -90.)
        changed = copy.deepcopy(ctx); changed[0]['P5'][1] -= 10; changed[0]['S5'][1] *= 9
        np.testing.assert_equal(W.features(ctx, b), W.features(changed, b))
        with self.assertRaises(ValueError): W.features(ctx, dict(b, side=2))

    def test_wait_never_expires_and_guard_free_entry(self):
        ctx = context(end=2000.)
        r = E.run(ctx, W.policy(ctx, {0: Score(-1.), 1: Score(-1.)}))
        self.assertFalse(any(r['order_states'].values()))
        r = E.run(ctx, W.policy(ctx, {0: Score(1.), 1: Score(-1.)}))
        self.assertEqual(r['order_states'][(0, 0)][0]['ts'], 0.)
        scheduled = E.run(ctx, W.policy(ctx, {}, 'SCHEDULED'))
        self.assertFalse(any(scheduled['order_states'].values()))
        for s in (0, 1): ctx[s]['vkl'][3:] = [1.] * (len(ctx[s]['vkl'])-3)
        scheduled = E.run(ctx, W.policy(ctx, {}, 'SCHEDULED'))
        self.assertEqual(scheduled['order_states'][(0, 0)][0]['ts'], 150.)
        bad = E.run(ctx, W.policy(ctx, {}, 'TIMER'), entry_mode='legacy')
        self.assertGreater(bad['order_states'][(0, 0)][0]['ts'], 150.)

    def test_entry_rows_include_no_fills_and_only_one_per_generation(self):
        rows, counts = W.build_window(context(), 'synthetic', 1, sample_mod=1)
        f = pd.DataFrame(rows)
        self.assertEqual(len(f), 12); self.assertEqual(set(f.kind), {'entry'})
        self.assertTrue((f.status == 'eligible').all()); self.assertTrue((f.target_c == 0).all())
        self.assertFalse(f.duplicated(['side', 'rung', 'generation']).any())
        self.assertGreater(counts['eligible'], len(f))

    def test_targets_capture_gain_and_harm_not_direction_labels(self):
        for adverse, sign in [(False, 1.), (True, -1.)]:
            r, _ = W.build_window(context([(95., 49, 30.)], adverse=adverse), 'synthetic', 1, sample_mod=1)
            row = next(x for x in r if x['side'] == 0 and x['rung'] == 0)
            self.assertGreater(sign * row['target_c'], 0.)

    def test_fit_separates_sides_and_excludes_calibration_and_test(self):
        test = next(iter(W.FOLDS)); train, cal = W.FOLDS[test]; observed = []
        def load(out, day):
            r = pd.DataFrame(np.zeros((400, len(W.FEATURES))), columns=W.FEATURES)
            r['day'] = day; r['market'] = 1; r['side'] = [0]*200+[1]*200; r['kind'] = 'entry'
            r['generation'] = np.arange(400); r['t'] = np.arange(400)*50.; r['status'] = 'eligible'
            r['target_c'] = np.where(r.side == 0, 1., -1.) if day in train else 9999.
            return r
        def dataset(x, y, feature_name): observed.append(y.copy()); return (x, y)
        import lightgbm as lgb
        with tempfile.TemporaryDirectory() as tmp:
            for day in (*train, cal, test):
                p = Path(tmp) / ('rows_'+day); p.mkdir(); (p/'receipt.json').write_text('{}')
            with mock.patch.object(W, 'load_rows', side_effect=load), mock.patch.object(lgb, 'Dataset', side_effect=dataset), \
                 mock.patch.object(lgb, 'train', side_effect=[Model(1.), Model(-1.)]):
                W.fit_fold(test, tmp)
            self.assertEqual(len(observed), 2)
            self.assertTrue((observed[0] == 1.).all()); self.assertTrue((observed[1] == -1.).all())
            receipt = E.read_json(Path(tmp)/('fit_'+test)/'receipt.json')
            self.assertEqual(receipt['train'], list(train)); self.assertNotIn(cal, receipt['train'])

    def test_refuses_constant_targets_partial_and_incompatible_rows(self):
        with self.assertRaises(ValueError): W.require(E.stamp())
        W.require(json.loads(json.dumps(W.stamp())))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'rows_20261023'; p.mkdir()
            E.write_json(p/'receipt.json', dict(**W.stamp(), day='20261023', partial=True, complete=False))
            with self.assertRaises(ValueError): W.load_rows(tmp, '20261023')
            with self.assertRaises(FileExistsError): E.new_stage(tmp, 'rows_20261023')
        test = next(iter(W.FOLDS))
        r = pd.DataFrame(dict(day=['20261023']*200+['20261024']*200, side=[0]*400, status=['eligible']*400, target_c=[0.]*400))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(W, 'load_rows', return_value=r):
            with self.assertRaisesRegex(ValueError, 'nonzero'): W.fit_fold(test, tmp)
        with self.assertRaises(ValueError): E.check_day('20261109')

    def test_replay_all_arms_keep_flat_and_reconstruct_side_models(self):
        test = next(iter(W.FOLDS)); ctx = context([(400., 49, 30.)])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/('fit_'+test); p.mkdir(); (p/'receipt.json').write_text('{}')
            with mock.patch.object(W, 'load_models', return_value={0: Score(1.), 1: Score(-1.)}), \
                 mock.patch.object(W.K, 'load_day', return_value={'stat': {}, 'W0': {1: 0.}}), \
                 mock.patch.object(E.PAIRS, 'window_pairs', return_value=iter([ctx])):
                W.eval_day(test, tmp)
            r = pd.read_parquet(Path(tmp)/('eval_'+test)/'windows.parquet')
            self.assertEqual(len(r), len(W.ARMS)*len(W.CFGS))
            self.assertTrue((r[r.arm == 'FLAT'].pnl == 0.).all())
            self.assertTrue((r[r.arm == 'VALUE'].shares > 0.).all())
            self.assertTrue(E.read_json(Path(tmp)/('eval_'+test)/'receipt.json')['complete'])

    def test_report_requires_profit_and_exact_market_identity(self):
        def fixture(tmp, profit, mismatch=False):
            for day in W.FOLDS:
                fit = Path(tmp)/('fit_'+day); fit.mkdir()
                predictions = pd.DataFrame(dict(day=[day]*4, side=[0]*4, rung=[0]*4, kind=['entry']*4,
                    t=[0., 50., 100., 150.], target_c=[1., -1., 1., -1.], prediction=[1., -1., 1., -1.]))
                predictions.to_parquet(fit/('pred_'+day+'.parquet'), index=False)
                E.write_json(fit/'receipt.json', dict(**W.stamp(), predictions_sha256={day: E.digest(fit/('pred_'+day+'.parquet'))}))
                stage = Path(tmp)/('eval_'+day); stage.mkdir(); rows = []
                for arm in W.ARMS:
                    for shift in W.CFGS:
                        for market in (1, 2):
                            rows.append(dict(day=day, market=market+2 if mismatch and arm == 'VALUE' else market,
                                arm=arm, shift=int(shift), pnl=profit if arm == 'VALUE' else -2.,
                                pnl_without_young=profit if arm == 'VALUE' else -2., inventory_square_seconds=1.))
                pd.DataFrame(rows).to_parquet(stage/'windows.parquet', index=False)
                pd.DataFrame({'synthetic':[1]}).to_parquet(stage/'inference.parquet', index=False)
                E.write_json(stage/'receipt.json', dict(**W.stamp(), day=day, markets=[1, 2], complete=True, partial=False,
                    source_status={}, counts={}, excluded_markets=[], windows_sha256=E.digest(stage/'windows.parquet'),
                    inference_sha256=E.digest(stage/'inference.parquet'), fit_receipt_sha256=E.digest(fit/'receipt.json')))
        for value, expected in [(1., True), (-1., False)]:
            with tempfile.TemporaryDirectory() as tmp:
                fixture(tmp, value); W.report(tmp)
                self.assertEqual(E.read_json(Path(tmp)/'report/result.json')['development_screen'], expected)
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp, 1., True)
            with self.assertRaisesRegex(ValueError, 'market identities'): W.report(tmp)

    def test_local_null_oracle_and_reversed_score(self):
        r = pd.DataFrame(dict(day=['synthetic']*400, side=[0]*400, kind=['entry']*400, t=np.arange(400)*50.,
                              target_c=[-1.]*200+[1.]*200, prediction=[-1.]*200+[1.]*200))
        self.assertTrue(E.local_null(r)['above_null_p95']); r.prediction *= -1
        self.assertFalse(E.local_null(r)['above_null_p95'])
        with self.assertRaises(ValueError): E.local_null(r, draws=10)


if __name__ == '__main__': unittest.main()
