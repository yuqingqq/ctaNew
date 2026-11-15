"""Executable client-observation, decomposition and runtime-parity controls."""
import copy
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import pf_wait_post_state as N
import pf_wait_post_v2 as M
from test_entry_value import context, Score
from test_order_harm_client50 import normalized
from test_wait_post_v2 import late, teacher


def births(W):
    got = []
    N.run(W, lambda v: True, observe=lambda v, a: got.append(v) if v['status'] == 'eligible' else None)
    return got


def snapshots(W, choose=None):
    got = []
    N.run(W, choose or (lambda v: True), observe=lambda v, a: got.append(v), snapshot_filter=lambda v: True)
    return got


class StateTests(unittest.TestCase):
    def test_execution_parity_all_clocks_and_waiting(self):
        for shift in M.CFGS:
            for delay in (0., 300.):
                W = late(True)
                old = M.run(copy.deepcopy(W), lambda v: v['t'] >= delay, shift)
                new = N.run(copy.deepcopy(W), lambda v: v['t'] >= delay, shift=shift, snapshot_filter=lambda v: True)
                for field in ('fl', 'fills', 'orders', 'order_states', 'endpos', 'stat', 'terminal_time', 'policy_decisions'):
                    self.assertEqual(normalized(old[field]), normalized(new[field]), field)

    def test_snapshot_reserves_pending_orders_across_sides(self):
        views = snapshots(context())
        v = next(v for v in views if v['t'] == 0. and v['side'] == 1 and v['rung'] == 0)
        N.validate_snapshot(v)
        self.assertEqual(len(v['client_orders']), 6)
        self.assertTrue(all(o[4] > v['t'] for o in v['client_orders']))
        x = N.Features(context()).vector(v, M.Tracker())
        self.assertEqual(x[N.FEATURES.index('opp_0_pending_place_lots')], 1.)
        bad = dict(v, exposure_bid=v['exposure_bid'] - 1.)
        with self.assertRaisesRegex(ValueError, 'reconciliation'): N.validate_snapshot(bad)

    def test_unreceived_partial_fill_does_not_change_snapshot_or_features(self):
        def collect(legs):
            W = context(legs)
            # Keep one slot empty so callbacks continue while another fills.
            views = snapshots(W, lambda v: v['rung'] != 5)
            selected = [v for v in views if v['t'] == 150. and v['side'] == 0 and v['rung'] == 5]
            return W, selected[0]
        # M50 shifts this anchor to execution100, receipt200.
        wa, a = collect([]); wb, b = collect([(150., 49, 10.)])
        self.assertEqual(a, b)
        np.testing.assert_equal(N.Features(wa).vector(a, M.Tracker()), N.Features(wb).vector(b, M.Tracker()))
        later = snapshots(context([(150., 49, 10.)]), lambda v: v['rung'] != 5)
        v = next(v for v in later if v['t'] == 200. and v['side'] == 0 and v['rung'] == 5)
        order = next(o for o in v['client_orders'] if o[:2] == (0, 0))
        self.assertEqual(order[5:7], (20., 10.))
        # A planted exposure derived from the unseen fill must fail the checker.
        bad = copy.deepcopy(b); oo = list(bad['client_orders']); row = list(oo[0]); row[5:7] = [20., 10.]; oo[0] = tuple(row)
        bad['client_orders'] = tuple(oo)
        with self.assertRaises(ValueError): N.validate_snapshot(bad)

    def test_future_venue_events_cannot_change_decision_prefix(self):
        def views(W):
            return [v for v in snapshots(W, lambda v: v['rung'] != 5) if v['t'] <= 150.]
        self.assertEqual(views(context()), views(context([(150., 49, 10.)])))
        self.assertNotEqual(views(context()), views(context([(100., 49, 10.)])))

    def test_queue_proxy_uses_received_frames_and_never_future_activation(self):
        W = context(); w = W[0]
        for s in (0, 1):
            W[s]['tf'] = np.array([0., 100., 200.])
            W[s]['kf'] = np.array([0., 150., 250.])
            W[s]['P5'] = np.repeat(W[s]['P5'], 3, axis=0)
            W[s]['S5'] = np.repeat(W[s]['S5'], 3, axis=0)
        W[0]['S5'][:, 0] = [100., 60., 5.]
        f = N.Features(W)
        self.assertEqual(f.queue(0, 50, 40., 200.), (60., 60.))
        self.assertTrue(np.isnan(f.queue(0, 50, 240., 200.)[1]))
        changed = copy.deepcopy(W); changed[0]['S5'][2, 0] = 9999.
        self.assertEqual(f.queue(0, 50, 40., 200.), N.Features(changed).queue(0, 50, 40., 200.))
        changed[0]['S5'][1, 0] = 12.
        self.assertNotEqual(f.queue(0, 50, 40., 200.), N.Features(changed).queue(0, 50, 40., 200.))
        self.assertTrue(np.isnan(f.queue(0, 44, 40., 200.)[1]))

    def test_ladder_features_react_to_prices_ages_and_remaining(self):
        W = context(); v = snapshots(W)[6]; f = N.Features(W); x = f.vector(v, M.Tracker())
        for col, value in ((2, 49), (3, -50.), (5, 20.)):
            z = copy.deepcopy(v); oo = list(z['client_orders']); o = list(oo[0]); o[col] = value
            if col == 5:
                o[6] = M.SIZE - value; z['exposure_bid'] -= M.SIZE - value
            oo[0] = tuple(o); z['client_orders'] = tuple(oo)
            y = f.vector(z, M.Tracker())
            self.assertFalse(np.array_equal(x, y, equal_nan=True))
        bad = dict(v, snapshot_time=v['t'] + 1)
        with self.assertRaisesRegex(ValueError, 'clock'): f.vector(bad, M.Tracker())
        bad = dict(v, client_slots=v['client_slots'][:-1])
        with self.assertRaisesRegex(ValueError, 'incomplete'): f.vector(bad, M.Tracker())

    def test_future_books_do_not_change_full_vector(self):
        W = late(True); v = snapshots(W)[6]
        a = N.Features(W).vector(v, M.Tracker())
        Z = copy.deepcopy(W)
        for s in (0, 1):
            Z[s]['S5'][1:] = 9999.; Z[s]['P5'][1:] = 10
            Z[s]['X'][1:] = 234.
        np.testing.assert_equal(a, N.Features(Z).vector(v, M.Tracker()))
        Z[0]['S5'][0, 0] = 12.
        self.assertFalse(np.array_equal(a, N.Features(Z).vector(v, M.Tracker()), equal_nan=True))

    def test_start_price_is_masked_in_candidate_and_continuation(self):
        W = context(); v = births(W)[0]; j = M.FEATURES.index('gap')
        W[0]['X'][0, j] = 123.
        self.assertTrue(np.isnan(N.safe_inputs(W, v, M.Tracker())[j]))
        self.assertEqual(M.inputs(W, v, M.Tracker())[j], 123.)
        W[0]['vkl'][0] = 0.
        self.assertEqual(N.safe_inputs(W, v, M.Tracker())[j], 123.)

    def test_components_reconcile_with_economic_controls(self):
        for adverse, positive in ((False, True), (True, False)):
            W = late(adverse); b = births(W)[0]
            pair, x = N.paired(W, b, 0., M.Forecast(), N.Features(W))
            N.check_targets(pd.DataFrame([pair]))
            self.assertEqual(pair['qpost'] > 0, positive)
            self.assertEqual(pair['local_post'] > 0, positive)
            self.assertEqual(len(x), len(N.FEATURES))
            bad = dict(pair, external_post=pair['external_post'] + 1.)
            with self.assertRaisesRegex(ValueError, 'reconciliation'): N.check_targets(pd.DataFrame([bad]))

    def test_local_order_does_not_hide_displaced_ladder_value(self):
        W = context([(95., 49, 30.), (200., 39, 30.)], adverse=True); b = births(W)[0]
        j = M.FEATURES.index('price'); continuation = teacher(lambda x: 1. if x[j] < 45. else -1.)
        pair, _ = N.paired(W, b, 0., continuation, N.Features(W))
        N.check_targets(pd.DataFrame([pair]))
        local_wait = pair['local_post'] - pair['local_delta']
        self.assertGreater(local_wait, 0.)
        self.assertLess(pair['qwait'], 0.)
        self.assertLess(pair['external_post'] - pair['external_delta'], 0.)

    def test_no_fill_is_zero_not_dropped_and_next_order_is_residual(self):
        W = context(); b = births(W)[0]
        pair, _ = N.paired(W, b, 0., M.Forecast(), N.Features(W))
        self.assertEqual(pair['status'], 'eligible'); self.assertEqual(pair['local_post'], 0.)
        result = {'order_states': {(0, 0): [{'pt': 50, 'fl': [(100., 10.)]}, {'pt': 40, 'fl': [(200., 30.)]}]}}
        value, fraction = N.focal_value(result, 0, 0, 0, 51.)
        self.assertAlmostEqual(value, 1 / 3); self.assertAlmostEqual(fraction, 1 / 3)
        self.assertNotEqual(value, (10. + 330.) / M.SIZE)

    def test_training_vector_matches_actual_policy_vector_and_side_heads(self):
        W = context(); b = births(W)[0]
        _, x = N.paired(W, b, 0., M.Forecast(), N.Features(W)); observed = []
        class Capture(N.Forecast):
            def predict(self, side, z):
                observed.append((side, z.copy()))
                return np.array([1.]), np.array([1. if side == 0 else -1.])
        policy = N.policy(W, Capture('BASE', {}))
        r = N.run(W, policy, snapshot_filter=lambda v: v['status'] == 'eligible')
        np.testing.assert_equal(x, observed[0][1][0])
        self.assertTrue(r['order_states'][0, 0]); self.assertFalse(r['order_states'][1, 0])

    def test_components_policy_sums_externalities_and_checks_schema(self):
        heads = {'local_post': Score(2.), 'external_post': Score(-3.), 'local_delta': Score(1.), 'external_delta': Score(0.)}
        f = N.Forecast('COMPONENTS', {0: heads}); x = np.zeros((1, len(N.FEATURES)))
        p, d = f.predict(0, x); self.assertLess(p[0], 0.); self.assertGreater(d[0], 0.)
        with self.assertRaisesRegex(ValueError, 'contract'): f.predict(0, x[:, :-1])


if __name__ == '__main__':
    unittest.main()
