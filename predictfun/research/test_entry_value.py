"""Synthetic entry-value controls; no historical data or real model fitting."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_entry_value as E
import pf_entry_value_lane as L
import pf_order_harm_client_lane as OLD
import pf_joint_ladder as JL
import pf_joint_ladder_v3 as J3
from test_order_harm_client50 import normalized


def context(legs=(), end=1000., adverse=False, guard=True):
    bid = [(0., [(50, 100), (49, 100), (48, 100), (47, 100), (46, 100)], 52)]
    ask = [(0., [(52, 100), (53, 100), (54, 100), (55, 100), (56, 100)], 50)]
    if adverse:
        bid.append((100., [(40, 100)], 42)); ask.append((100., [(42, 100)], 40))
    W = {0: JL._ctx(0, bid, list(legs), t_end=end), 1: JL._ctx(1, ask, [], t_end=end)}
    for s, w in W.items():
        w['vkl'] = [-1. if guard else 1.] * len(w['tg'])
        w['m'] = 1
        w['tf_mff'] = (w['tf'], np.array([51., 41.] if adverse else [51.]))
    return W


class Score:
    def __init__(self, value): self.value = value
    def predict(self, X): return np.full(len(X), self.value)
    def save_model(self, path): Path(path).write_text('synthetic model\n')


class EntryValueTests(unittest.TestCase):
    def run_lane(self, W, choose, observer=None, entry_mode='value', cap=1500., nr=1):
        guards = {s: [True] * len(W[s]['tg']) for s in (0, 1)}
        return L.sim_joint(W, 'U', 0., cap, guards, size=30., nr=nr, mode='hard',
                           policy=choose, observe=observer, entry_mode=entry_mode)

    def test_legacy_default_parity_and_existing_engine_mutants(self):
        W = context([(400., 47, 40.), (600., 49, 10.)])
        aux = {s: [False] * 4 + [True] * 4 + [False] * 12 for s in (0, 1)}
        for q in ('M', 'U'):
            for shift in (0., 50., 250.):
                old = OLD.sim_joint(copy.deepcopy(W), q, shift, 60., aux, nr=3, mode='hard')
                new = L.sim_joint(copy.deepcopy(W), q, shift, 60., aux, nr=3, mode='hard')
                self.assertEqual(normalized(old), normalized(new))
        with mock.patch.object(J3, 'sim_joint', L.sim_joint):
            self.assertEqual(J3.selftest(), [])
            for name in ('MUT_REF_RELEASE', 'MUT_REST_TRUE', 'MUT_HELD_TRUE', 'MUT_LEG_LEAK', 'MUT_PRIORITY'):
                with self.subTest(mutant=name), mock.patch.object(L, name, 1):
                    self.assertTrue(J3.selftest())

    def test_metric_can_enter_during_guard_without_waiting_for_timer(self):
        W = context()
        choose = lambda v: v['kind'] == 'rest' or v['k'] >= 2
        r = self.run_lane(W, choose)
        self.assertEqual(r['order_states'][(0, 0)][0]['ts'], 100.)
        old_gate = self.run_lane(context(), choose, entry_mode='legacy')
        self.assertFalse(old_gate['order_states'][(0, 0)])  # Known-bad inherited gate blocks the profitable decision.

    def test_negative_score_does_not_expire_into_a_retry(self):
        W = context(end=2000., guard=False)
        r = self.run_lane(W, E.value_policy(W, {'entry': Score(-1.), 'rest': Score(1.)}))
        self.assertFalse(r['order_states'][(0, 0)])
        self.assertEqual(r['policy_status']['eligible'], 80)
        positive = self.run_lane(context(), lambda v: True)
        self.assertEqual(positive['order_states'][(0, 0)][0]['ts'], 0.)

    def test_partial_fills_preserve_generation_and_client_remaining(self):
        views = []
        r = self.run_lane(context([(100., 49, 10.), (200., 49, 10.), (300., 49, 5.)]),
                          lambda v: True, lambda v, a: views.append(v))
        self.assertEqual(len(r['order_states'][(0, 0)]), 1)
        self.assertEqual(len(r['fills'][(0, 0)]), 3)
        by_t = {v['t']: v for v in views if v['side'] == 0 and v['kind'] == 'rest'}
        self.assertEqual(by_t[150.]['quantity'], 30.)
        self.assertEqual(by_t[200.]['quantity'], 20.)
        self.assertEqual(by_t[300.]['quantity'], 10.)
        self.assertTrue(all(v['generation'] == 1 for v in by_t.values()))

    def test_unseen_fill_and_future_refusal_cannot_change_prefix(self):
        def views(W):
            got = []
            self.run_lane(W, lambda v: True, lambda v, a: got.append(v))
            return [v for v in got if v['t'] < 200.]
        self.assertEqual(views(context()), views(context([(80., 49, 30.)])))
        W = context(); changed = copy.deepcopy(W)
        # Refusal at activation changes only the future venue book. The t=0
        # client decision, including reservation for the other side, is equal.
        for s in (0, 1):
            changed[s]['tf'] = np.array([0., 20.])
            changed[s]['kf'] = np.array([0., 50.])
            changed[s]['P5'] = np.vstack([W[s]['P5'], W[s]['P5']])
            changed[s]['S5'] = np.vstack([W[s]['S5'], W[s]['S5']])
            changed[s]['opp'] = np.array([52, 49]) if s == 0 else np.array([50, 50])
        self.assertEqual([v for v in views(W) if v['t'] == 0.], [v for v in views(changed) if v['t'] == 0.])

    def test_pending_cancel_reserves_price_and_capacity(self):
        def choose(v):
            return not (v['kind'] == 'rest' and v['k'] == 2)
        r = self.run_lane(context(), choose, cap=30.)
        orders = r['order_states'][(0, 0)]
        self.assertEqual(orders[0]['tc'], 100.)
        self.assertGreaterEqual(orders[1]['ts'], 222.)  # Cancel effective 122, confirmation 222.
        self.assertEqual(orders[1]['ts'], 250.)
        self.assertGreater(r['policy_status']['price_reserved'], 0)
        self.assertFalse(any(r['stat']['viol'].values()))

    def test_known_capacity_prevents_large_ladder_overbooking(self):
        r = self.run_lane(context(), lambda v: True, cap=30., nr=6)
        self.assertEqual(len(r['order_states'][(0, 0)]), 1)
        self.assertFalse(r['order_states'][(0, 1)])
        self.assertGreater(r['policy_status']['capacity'], 0)

    def test_feature_future_invariance_and_remaining_quantity_positive_control(self):
        W = context(adverse=True); views = []
        self.run_lane(W, lambda v: True, lambda v, a: views.append(v))
        v = next(v for v in views if v['kind'] == 'rest' and v['side'] == 0 and v['t'] == 50.)
        a = E.features(W, v); future = copy.deepcopy(W)
        future[0]['S5'][1] *= 7.; future[0]['P5'][1] -= 3
        np.testing.assert_equal(a, E.features(future, v))
        fewer = dict(v, quantity=10., known_filled=20.)
        self.assertNotEqual(a[E.FEATURES.index('quantity')], E.features(W, fewer)[E.FEATURES.index('quantity')])
        with self.assertRaises(ValueError):
            E.features(W, dict(v, t=-1.))

    def test_actual_post_wait_and_keep_cancel_targets(self):
        # M50 moves this leg's simulated execution to 45 ms: a t=0 post is
        # active; WAIT until the t=50 decision misses it.
        W = context([(95., 49, 30.)]); views = []
        base = E.run(W, lambda v: True, observe=lambda v, a: views.append(v))
        v = next(v for v in views if v['kind'] == 'entry' and v['side'] == 0 and v['rung'] == 0)
        pair = E.paired_probe(W, v, base)
        self.assertAlmostEqual(pair['target_c'], 1.)
        # At t=50 cancelling lands at 72 ms, before the execution at 80 ms.
        for adverse, sign in ((False, 1.), (True, -1.)):
            W = context([(130., 49, 30.)], adverse=adverse); views = []
            base = E.run(W, lambda v: True, observe=lambda v, a: views.append(v))
            v = next(v for v in views if v['kind'] == 'rest' and v['side'] == 0 and v['rung'] == 0 and v['t'] == 50.)
            self.assertGreater(E.paired_probe(W, v, base)['target_c'] * sign, 0.)
            with self.assertRaisesRegex(ValueError, 'identical client'):
                E.paired_probe(W, dict(v, quantity=1.), base)

    def test_no_fill_rows_and_generation_deduplication(self):
        rows, statuses = E.build_window(context(), '20990101', 1, sample_mod=1, age_slots=1)
        frame = pd.DataFrame(rows)
        self.assertEqual(len(frame), 24)  # 12 empty generations + 12 resting generations.
        self.assertTrue((frame.status == 'eligible').all())
        self.assertTrue((frame.target_c == 0.).all())
        self.assertFalse(frame.duplicated(['side', 'rung', 'kind', 'generation']).any())
        self.assertGreater(statuses['eligible'], len(frame))

    def test_model_policy_heads_and_invalid_forecast(self):
        W = context(); views = []
        self.run_lane(W, lambda v: True, lambda v, a: views.append(v))
        entry = next(v for v in views if v['kind'] == 'entry')
        rest = next(v for v in views if v['kind'] == 'rest' and v['status'] == 'eligible')
        p = E.value_policy(W, {'entry': Score(-1.), 'rest': Score(1.)})
        self.assertFalse(p(entry)); self.assertTrue(p(rest))
        p = E.value_policy(W, {'entry': Score(1.), 'rest': Score(-1.)})
        self.assertTrue(p(entry)); self.assertFalse(p(rest))
        p = E.value_policy(W, {'entry': Score(np.nan), 'rest': Score(np.inf)})
        self.assertFalse(p(entry)); self.assertFalse(p(rest))

    def test_refuses_legacy_receipts_partial_rows_and_protected_dates(self):
        E.require(E.stamp())
        with self.assertRaises(ValueError): E.require(OLD.stamp())
        with self.assertRaises(ValueError): E.require(dict(E.stamp(), target='old'))
        with self.assertRaises(ValueError): E.check_day('20261109')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'rows_20261023'; p.mkdir()
            E.write_json(p / 'receipt.json', dict(**E.stamp(), partial=True, day='20261023'))
            with self.assertRaises(ValueError): E.load_rows(tmp, '20261023')
            with self.assertRaises(FileExistsError): E.new_stage(tmp, p.name)

    def test_fit_uses_only_earlier_training_actions(self):
        test = next(iter(E.FOLDS)); train, cal = E.FOLDS[test]; seen = []
        def load(out, day):
            r = pd.DataFrame(np.zeros((400, len(E.FEATURES))), columns=E.FEATURES)
            r['day'] = day; r['market'] = 1; r['side'] = 0; r['kind'] = ['entry'] * 200 + ['rest'] * 200
            r['generation'] = np.arange(400); r['t'] = np.arange(400) * 50.; r['status'] = 'eligible'
            r['target_c'] = 1. if day in train else 9999.
            return r
        def dataset(x, y, feature_name):
            seen.append((len(x), np.array(y))); return (x, y)
        import lightgbm as lgb
        with tempfile.TemporaryDirectory() as tmp:
            for day in (*train, cal, test):
                p = Path(tmp) / f'rows_{day}'; p.mkdir(); (p / 'receipt.json').write_text('{}')
            with mock.patch.object(lgb, 'Dataset', side_effect=dataset), mock.patch.object(lgb, 'train', return_value=Score(1.)), \
                    mock.patch.object(E, 'load_rows', side_effect=load):
                E.fit_fold(test, tmp)
            self.assertEqual(len(seen), 2)
            self.assertTrue(all(n == len(train) * 200 and np.all(y == 1.) for n, y in seen))
            receipt = E.read_json(Path(tmp) / f'fit_{test}' / 'receipt.json')
            self.assertEqual(receipt['train'], list(train))
            self.assertNotIn(cal, receipt['train']); self.assertNotIn(test, receipt['train'])

    def test_full_replay_keeps_flat_windows_and_reconciles(self):
        test = next(iter(E.FOLDS)); models = {'entry': Score(1.), 'rest': Score(1.)}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / f'fit_{test}'; p.mkdir(); (p / 'receipt.json').write_text('{}')
            with mock.patch.object(E, 'load_models', return_value=models), mock.patch.object(E.K, 'load_day', return_value={'stat': {}}), \
                    mock.patch.object(E.PAIRS, 'window_pairs', return_value=iter([context([(400., 49, 30.)])])):
                E.eval_day(test, tmp)
            r = pd.read_parquet(Path(tmp) / f'eval_{test}' / 'windows.parquet')
            self.assertEqual(len(r), 12)
            self.assertTrue((r[r.arm == 'FLAT'].pnl == 0.).all())
            self.assertTrue((r[r.arm == 'FLAT'].any_quote_steps == 0).all())
            self.assertTrue((r[r.arm == 'VALUE'].shares > 0.).all())

    def test_local_null_detects_oracle_and_anti_oracle(self):
        r = pd.DataFrame(dict(day=['synthetic'] * 400, side=[0] * 400, kind=['entry'] * 400,
                              t=np.arange(400) * 50., target_c=[-1.] * 200 + [1.] * 200,
                              prediction=[-1.] * 200 + [1.] * 200))
        self.assertTrue(E.local_null(r)['above_null_p95'])
        r.prediction *= -1
        self.assertFalse(E.local_null(r)['above_null_p95'])
        with self.assertRaises(ValueError): E.local_null(r, draws=10)

    def test_report_rejects_relative_losses_and_mismatched_windows(self):
        def fixture(tmp, candidate_pnl, wrong_ids=False):
            for day in E.FOLDS:
                fit = Path(tmp) / f'fit_{day}'; fit.mkdir(); (fit / 'receipt.json').write_text('{}')
                pr = pd.DataFrame(dict(day=[day] * 4, side=[0] * 4, kind=['entry'] * 4,
                                       t=[0., 50., 100., 150.], prediction=[1., -1., 1., -1.], target_c=[1., -1., 1., -1.]))
                pr.to_parquet(fit / f'pred_{day}.parquet', index=False)
                stage = Path(tmp) / f'eval_{day}'; stage.mkdir(); rows = []
                for arm in E.ARMS:
                    for shift in E.CFGS:
                        for market in (1, 2):
                            m = market + 2 if wrong_ids and arm == 'VALUE' else market
                            rows.append(dict(day=day, market=m, arm=arm, shift=int(shift),
                                             pnl=candidate_pnl if arm == 'VALUE' else (0. if arm == 'FLAT' else -2.),
                                             pnl_without_young=candidate_pnl if arm == 'VALUE' else (0. if arm == 'FLAT' else -2.), young_shares=0.,
                                             inventory_square_seconds=1., shares=30., pulls=1, places=2,
                                             any_quote_steps=10, both_quote_steps=5, grid_steps=20))
                pd.DataFrame(rows).to_parquet(stage / 'windows.parquet', index=False)
                pd.DataFrame({'synthetic': [1]}).to_parquet(stage / 'inference.parquet', index=False)
                E.write_json(stage / 'receipt.json', dict(**E.stamp(), day=day, complete=True,
                             windows_sha256=E.digest(stage / 'windows.parquet'), inference_sha256=E.digest(stage / 'inference.parquet'),
                             fit_receipt_sha256=E.digest(fit / 'receipt.json')))
        for pnl, expected in ((1., True), (-1., False)):
            with tempfile.TemporaryDirectory() as tmp:
                fixture(tmp, pnl)
                with contextlib.redirect_stdout(io.StringIO()): E.report(tmp)
                r = E.read_json(Path(tmp) / 'report/result.json')
                self.assertEqual(r['development_accept'], expected)
                self.assertTrue(all(c['pass_comparison'] for c in r['comparisons']))
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp, 1., wrong_ids=True)
            with self.assertRaisesRegex(ValueError, 'market identities'): E.report(tmp)


if __name__ == '__main__':
    unittest.main()
