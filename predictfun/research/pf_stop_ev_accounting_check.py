"""Synthetic accounting checks; no historical/raw data or model fitting."""
import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_joint_ladder as J
import pf_stop_ev_accounting as A


def world(legs=((200., 50, 10.),), *, side=0, queue=0., epoch=0., frames=None, marks=None):
    W = {}
    for s in (0, 1):
        best, direction, opp = (50, -1, 52) if s == 0 else (52, 1, 50)
        book = [(t, [(best + direction*d, q) for d in range(5)], opp)
                for t, q in (frames or ((0., queue), (500., queue)))]
        w = J._ctx(s, book, list(legs) if s == side else [])
        w['m'] = 1
        for name in ('tf', 'kf', 'tg', 'anc'):
            w[name] = np.asarray(w[name], float) + epoch
        w['ws'] += epoch; w['te'] += epoch
        mt, mv = zip(*(marks or ((0., 51.), (1000., 51.), (2500., 51.))))
        w['tf_mff'] = (np.asarray(mt) + epoch, np.asarray(mv))
        W[s] = w
    return W


def ledger(W=None, *, size=30., nr=1, result=None, info=None, coverage=None, shift=0.):
    W = world() if W is None else W
    r = A.replay_neutral(W, size=size, nr=nr, shift_ms=shift) if result is None else result
    infos = {(s, i): A.LegInfo(f'{s}:{i}', 'ws') for s in (0, 1) for i in range(len(W[s]['anc']))}
    cov = A.Coverage(W[0]['ws'], W[0]['te'] + 200., W[0]['te'] + 3000., len(infos),
                     W[0]['ws'], W[0]['te'] + 200.)
    return A.Ledger(W, r, submitted_quantity=size, shift_ms=shift,
                    leg_info=infos if info is None else info, coverage=cov if coverage is None else coverage)


def action(L, t=50., side=0, rung=0, settings=A.Settings()):
    return A.label_action(L, (side, rung, 0), L.ws+t, day='synthetic', settings=settings)


class Checks(unittest.TestCase):
    def test_fifo_selects_different_markouts(self):
        marks = ((0., 50.), (1200., 51.), (1400., 49.), (2500., 50.))
        values = []
        for q in (0., 20.):
            l = ledger(world(((200., 50, 20.), (400., 50, 20.)), queue=q, marks=marks), size=20.)
            row = action(l); self.assertEqual(row['status'], 'eligible')
            values.append(row['label']['value_cents'])
            self.assertEqual(row['label']['filled_quantity'], 20.)
        self.assertEqual(values, [20., -20.])
        # Known-bad aggregate-average target is zero at both queue positions.
        self.assertNotEqual(values, [0., 0.])

    def test_prefix_boundary_and_small_through_are_quantity_limited(self):
        l = ledger(world(((60., 50, 5.), (72., 50, 7.), (200., 50, 3.), (300., 49, 4.))))
        y = action(l)['label']
        self.assertEqual(y['common_prefix_quantity'], 5.)
        self.assertEqual(y['actual_remaining_after_prefix'], 25.)
        self.assertEqual(y['filled_quantity'], 14.)
        self.assertEqual(y['value_cents'], 14.)
        self.assertEqual(y['at_price_value_cents'], 10.)
        self.assertEqual(y['through_price_value_cents'], 4.)
        self.assertNotEqual(y['filled_quantity'], 30.)

    def test_shared_leg_does_not_fill_every_level_in_full(self):
        l = ledger(world(((200., 48, 40.),)), nr=2)
        a, b = action(l), action(l, rung=1)
        self.assertEqual([a['label']['filled_quantity'], b['label']['filled_quantity']], [30., 10.])
        self.assertEqual(a['label']['value_cents'] + b['label']['value_cents'], 50.)
        bad = copy.deepcopy(l.W); bad[0]['shl'][0] = 39.
        with self.assertRaisesRegex(ValueError, 'shared taker'):
            ledger(bad, nr=2, result=l.result)

    def test_bid_ask_sign_and_through_can_be_profitable(self):
        for side, price in ((0, 49), (1, 53)):
            l = ledger(world(((200., price, 10.),), side=side))
            y = action(l, side=side)['label']
            self.assertEqual(y['value_cents'], 10.)
            self.assertEqual(y['through_price_value_cents'], 10.)

    def test_size_and_client_remaining(self):
        for size in (30., 100.):
            l = ledger(world(((120., 50, 10.), (300., 50, 5.))), size=size)
            a = action(l, 150.); b = action(l, 250.)
            self.assertEqual(a['features']['known_remaining'], size)
            self.assertEqual(a['label']['past_quantity'], 10.)
            self.assertEqual(a['label']['value_per_known_remaining_share'], 5./size)
            self.assertEqual(b['features']['known_remaining'], size-10.)

    def test_future_frames_and_unacknowledged_fills_are_not_features(self):
        l = ledger(world(((120., 50, 10.), (300., 50, 5.)), frames=((0., 20.), (500., 1.))))
        key = (0, 0, 0); o = l.orders[key]
        kw = dict(terminal_ms=l.end)
        original = A.known_state(l.W, o, key, 150., 30., **kw)
        changed = copy.deepcopy(l.W); changed[0]['S5'][1] = 100000.
        changed[0]['kf'][1] = 151.
        o2 = copy.deepcopy(o); o2['rem'] = 0.; o2['ahead'] = 9999.; o2['fl'] = [(120., 29.), (600., 1.)]
        self.assertEqual(original, A.known_state(changed, o2, key, 150., 30., **kw))
        changed[0]['S5'][0] = 7.
        self.assertNotEqual(original, A.known_state(changed, o2, key, 150., 30., **kw))

    def test_known_queue_never_increases_on_later_additions(self):
        l = ledger(world((), frames=((0., 10.), (100., 20.), (200., 5.), (300., 40.))))
        self.assertEqual(action(l, 350.)['features']['queue_ahead_known'], 5.)

    def test_valid_unfilled_and_unacknowledged_full_fill_are_zero(self):
        l = ledger(world(()))
        self.assertEqual(action(l)['label']['value_cents'], 0.)
        l = ledger(world(((120., 50, 30.), (400., 49, 100.))))
        row = action(l, 150.)
        self.assertEqual(row['status'], 'eligible')
        self.assertEqual(row['label']['value_cents'], 0.)
        self.assertEqual(action(l, 250.)['status'], 'not_client_resting')

    def test_missing_legs_and_ambiguous_reference_prefix_fail_closed(self):
        l = ledger()
        gap = A.MissingLeg('missing', -float('inf'), float('inf'), 'untimed')
        c = replace(l.coverage, expected_legs=2, missing=(gap,))
        self.assertEqual(action(ledger(l.W, result=l.result, coverage=c))['status'], 'missing_trade_timing')
        info = {(0, 0): A.LegInfo('0:0', 'e2', 250.)}
        e = ledger(l.W, result=l.result, info=info)
        self.assertEqual(action(e)['status'], 'ambiguous_reference_timing')
        row = action(e, settings=A.Settings(timing='all_anchors_sensitivity'))
        self.assertEqual(row['status'], 'eligible_estimated_timing')
        self.assertEqual(row['label']['value_cents'], 10.)
        # Even a non-filling uncertain leg before the action changes the path.
        l = ledger(world(((20., 50, 1.),)))
        e = ledger(l.W, result=l.result, info=info)
        self.assertEqual(action(e)['status'], 'ambiguous_reference_timing')

    def test_coverage_and_mark_refusals_are_not_zero_labels(self):
        l = ledger()
        cases = [(replace(l.coverage, trade_end_ms=100.), 'incomplete_trade_coverage'),
                 (replace(l.coverage, book_end_ms=100.), 'incomplete_book_coverage'),
                 (replace(l.coverage, mark_end_ms=100.), 'incomplete_mark_coverage'),
                 (replace(l.coverage, book_gaps=(A.MissingLeg('b', 80., 90., 'gap'),)), 'incomplete_book_coverage')]
        for cov, status in cases:
            row = action(ledger(l.W, result=l.result, coverage=cov))
            self.assertEqual(row['status'], status); self.assertIsNone(row['label'])
        w = copy.deepcopy(l.W); w[0]['tf_mff'] = (np.array([0.]), np.array([51.]))
        self.assertEqual(action(ledger(w, result=l.result))['status'], 'missing_or_stale_mark')
        w[0]['tf_mff'] = (np.array([1200.]), np.array([np.nan]))
        self.assertEqual(action(ledger(w, result=l.result))['status'], 'missing_or_stale_mark')

    def test_marks_follow_execution_clock_not_unshifted_anchor(self):
        w = world(((250., 50, 10.),), marks=((0., 50.), (1200., 51.), (1250., 49.)))
        y = action(ledger(w, shift=50.))['label']
        self.assertEqual(y['value_cents'], 10.)
        self.assertEqual(y['tranches'][0]['requested_mark_time'], 1200.)

    def test_epoch_clock_corruption_is_refused(self):
        l = ledger(world(epoch=1794744000000.))
        self.assertEqual(action(l)['label']['value_cents'], 10.)
        for mutate in ('age', 'flat', 'order'):
            r = copy.deepcopy(l.result)
            if mutate == 'age':
                f = list(r['fills'][0, 0][0]); f[4] += 0.1; r['fills'][0, 0][0] = tuple(f)
            elif mutate == 'flat':
                f = list(r['fl'][0]); f[2] += 0.1; r['fl'][0] = tuple(f)
            else:
                f = list(r['order_states'][0, 0][0]['fl'][0]); f[0] += 0.1
                r['order_states'][0, 0][0]['fl'][0] = tuple(f)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                ledger(l.W, result=r)

    def test_bad_identity_reference_quantity_and_sources_are_refused(self):
        l = ledger()
        for mutate in ('queue', 'neutral', 'remaining', 'position', 'leg'):
            r = copy.deepcopy(l.result)
            if mutate == 'queue': r['order_states'][0, 0][0]['queue_initialization'] = 'old'
            if mutate == 'neutral': del r['local_reference']
            if mutate == 'remaining': r['order_states'][0, 0][0]['rem'] -= 1.
            if mutate == 'position': r['endpos'] += 1.
            if mutate == 'leg':
                f = list(r['fills'][0, 0][0]); f[0] = 10; r['fills'][0, 0][0] = tuple(f)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): ledger(l.W, result=r)
        with self.assertRaises(ValueError): ledger(l.W, result=l.result, info={})
        with self.assertRaises(ValueError): ledger(l.W, result=l.result, coverage=replace(l.coverage, expected_legs=2))
        with self.assertRaises(ValueError): action(l, settings=A.Settings(cancel_ms=23.))
        with self.assertRaises(ValueError): A.Settings(exposure_ms=float('inf'))

    def test_probe_population_is_outcome_independent_and_retains_exclusions(self):
        l = ledger()
        before = list(A.generation_probes(l, day='synthetic'))
        l.orders[0, 0, 0]['rem'] = 999.
        self.assertEqual(before, list(A.generation_probes(l, day='synthetic')))
        l = ledger()
        rows, r = A.build_actions(l, day='synthetic', asof='synthetic', ages_ms=(2000.,))
        self.assertEqual(r['n_actions'], r['n_generations'])
        self.assertEqual(r['statuses'], {'probe_not_reached': 2})
        self.assertTrue(all(x['label'] is None for x in rows))

    def test_matched_action_value_has_positive_and_bad_controls(self):
        row = action(ledger()); rows = []
        for i in range(100):
            r = copy.deepcopy(row); r['action_id'] = ('synthetic', i, 0, 0, 0)
            r['label']['value_cents'] = 1. if i % 2 else -1.
            rows.append(r)
        keep = [bool(i % 2) for i in range(100)]
        good = A.action_metrics(rows, keep, asof='synthetic')
        bad = A.action_metrics(rows, [not x for x in keep], asof='synthetic')
        self.assertEqual(good['local_kept_value_cents'], 50.)
        self.assertEqual(good['local_gain_vs_keep_all_cents'], 50.)
        self.assertLess(good['p_value'], .01); self.assertGreater(bad['p_value'], .9)
        for rr, kk, draws in ((rows + [rows[0]], keep + [keep[0]], 500), (rows, keep, 199), ([], [], 500)):
            with self.assertRaises(ValueError): A.action_metrics(rr, kk, asof='synthetic', draws=draws)
        # Exact remainder matching must not substitute unmatched controls.
        rows[0]['features']['known_remaining'] = 29.
        unique = rows[:1]
        self.assertIsNone(A.action_metrics(unique, [True], asof='synthetic')['p_value'])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
    receipt = dict(asof=datetime.now(timezone.utc).isoformat(), contract=A.CONTRACT,
                   tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                   passed=result.wasSuccessful() and peak <= 2048., peak_rss_mib=peak,
                   source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       (Path(__file__), Path(A.__file__))}, data='synthetic only; no historical performance claim')
    print(json.dumps(receipt, sort_keys=True))
    sys.exit(0 if receipt['passed'] else 1)
