"""Synthetic order-level regressions for activation-only queue initialization."""
import copy
import math
from pathlib import Path
import resource
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_joint_ladder as J
import pf_entry_repair_lane as OLD_ENTRY
import pf_entry_queue_activation_lane as ENTRY
import pf_stop_model as OLD_STOP
import pf_stop_queue_activation_lane as STOP
from pf_queue_activation import initial_queue, CONTRACT


def context(frames=((0., 100.), (60., 300.)), trades=((200., 130.),), side=0,
            unknown=False, becomes_visible=False, through=False):
    """Same frame clock on both sides; all events are synthetic."""
    W = {}
    for s in (0, 1):
        best = 50 if s == 0 else 52
        direction = -1 if s == 0 else 1
        book = []
        for t, amount in frames:
            levels = [(best + direction * d, amount if s == side and d == 0 else 100.) for d in range(5)]
            if unknown and becomes_visible and t > 40.:
                levels[-1] = (best + direction * 5, amount)
            book.append((t, levels, 52 if s == 0 else 50))
        price = best + direction * (5 if unknown else 0) + (direction if through else 0)
        legs = [(t, price, q) for t, q in trades] if s == side else []
        W[s] = J._ctx(s, book, legs, t_end=1000.)
        W[s]['m'] = 1
    return W


def replay(kind, W, side=0, rung=0, size=30., unknown='zero', cancel=None, rule='M', legacy=False):
    W = copy.deepcopy(W)
    n = len(W[0]['tg']); aux = {s: [False] * n for s in (0, 1)}
    if kind == 'entry':
        lane = OLD_ENTRY if legacy else ENTRY
        def choose(v):
            if v['kind'] == 'rest':
                return not (cancel is not None and v['t'] == cancel)
            return v['side'] == side and v['rung'] == rung
        r = lane.sim_joint(W, rule, 0., None, aux, nr=rung + 1, size=size, mode='none',
                           policy=choose, unknown_queue='block' if unknown == 'inf' else unknown)
    else:
        gate = {(s, d): (lambda k, blocked=(s != side or d != rung): blocked)
                for s in (0, 1) for d in range(rung + 1)}
        dec = {(side, rung): lambda k, t, o: cancel is not None and t == cancel}
        lane = OLD_STOP if legacy else STOP
        with mock.patch.object(OLD_STOP, 'QUNK', 'inf' if unknown == 'block' else unknown):
            r = lane.sim_joint_g(W, rule, 0., None, aux, decide=dec, gate=gate,
                                 nr=rung + 1, size=size, mode='none', cooldown=0.)
    if any(r['stat']['viol'].values()):
        raise AssertionError(r['stat']['viol'])
    return r


class Checks(unittest.TestCase):
    def cases(self):
        return ((kind, side) for kind in ('entry', 'stop') for side in (0, 1))

    def test_additions_after_activation_do_not_go_ahead(self):
        for kind, side in self.cases():
            with self.subTest(kind=kind, side=side):
                r = replay(kind, context(side=side), side)
                o = r['order_states'][side, 0][0]
                self.assertEqual(o['ta'], 40.)
                self.assertEqual(o['a0'], 100.)
                self.assertEqual(o['queue_snapshot_time'], 0.)
                self.assertEqual(o['queue_initialization'], CONTRACT)
                self.assertEqual(o['fl'], [(200., 30.)])
                bad = replay(kind, context(side=side), side, legacy=True)
                # Positive control plus known-bad original implementation.
                with self.assertRaises(AssertionError):
                    self.assertEqual(bad['order_states'][side, 0][0]['a0'], 100.)
                self.assertEqual(bad['order_states'][side, 0][0]['fl'], [])

    def test_existing_queue_and_activation_boundary_are_respected(self):
        for frame_time in (20., 40.):
            for kind, side in self.cases():
                with self.subTest(frame_time=frame_time, kind=kind, side=side):
                    W = context(((0., 100.), (frame_time, 200.), (60., 400.)), ((200., 205.),), side)
                    o = replay(kind, W, side)['order_states'][side, 0][0]
                    self.assertEqual(o['a0'], 200.)
                    self.assertEqual(o['queue_snapshot_time'], frame_time)
                    self.assertEqual(o['fl'], [(200., 5.)])

    def test_empty_visible_price_keeps_first_place(self):
        for kind, side in self.cases():
            W = context(((0., 0.), (60., 100.), (300., 1000.)), ((200., 10.), (400., 10.)), side)
            o = replay(kind, W, side)['order_states'][side, 0][0]
            self.assertEqual(o['a0'], 0.)
            self.assertEqual(o['fl'], [(200., 10.), (400., 10.)])

    def test_partial_fills_preserve_the_same_order(self):
        for kind, side in self.cases():
            W = context(trades=((200., 110.), (400., 10.), (800., 10.)), side=side)
            orders = replay(kind, W, side, size=50.)['order_states'][side, 0]
            self.assertEqual(len(orders), 1)
            self.assertEqual(orders[0]['fl'], [(200., 10.), (400., 10.), (800., 10.)])
            self.assertEqual(orders[0]['rem'], 20.)

    def test_cancel_and_repost_join_the_new_existing_queue(self):
        for kind, side in self.cases():
            W = context(((0., 100.), (60., 300.), (360., 500.)), ((150., 110.), (600., 310.)), side)
            orders = replay(kind, W, side, size=50., cancel=200.)['order_states'][side, 0]
            self.assertEqual(len(orders), 2)
            self.assertEqual([o['a0'] for o in orders], [100., 300.])
            self.assertEqual([o['ta'] for o in orders], [40., 290.])
            self.assertEqual(orders[0]['fl'], [(150., 10.)])
            self.assertEqual(orders[1]['fl'], [(600., 10.)])

    def test_unknown_depth_stays_an_explicit_assumption(self):
        for kind, side in self.cases():
            W = context(trades=((200., 10.),), side=side, unknown=True)
            zero = replay(kind, W, side, rung=5)['order_states'][side, 5][0]
            blocked = replay(kind, W, side, rung=5, unknown='block')['order_states'][side, 5][0]
            self.assertEqual(zero['a0'], 0.)
            self.assertTrue(math.isinf(blocked['a0']))
            self.assertEqual(zero['fl'], [(200., 10.)])
            self.assertEqual(blocked['fl'], [])
            through = context(trades=((200., 10.),), side=side, unknown=True, through=True)
            filled = replay(kind, through, side, rung=5, unknown='block')['order_states'][side, 5][0]
            self.assertEqual(filled['fl'], [(200., 10.)])

    def test_unknown_initial_queue_is_not_retroactively_replaced(self):
        for kind, side in self.cases():
            W = context(((0., 100.), (60., 80.)), ((200., 90.),), side, unknown=True, becomes_visible=True)
            o = replay(kind, W, side, rung=5, unknown='block')['order_states'][side, 5][0]
            self.assertTrue(math.isinf(o['a0']))
            # M may bound an unknown queue when the level becomes visible later.
            self.assertEqual(o['fl'], [(200., 10.)])

    def test_later_frame_perturbation_cannot_change_initial_queue(self):
        for kind, side in self.cases():
            original = context(side=side)
            changed = copy.deepcopy(original)
            changed[side]['S5'][1:] = 1e9
            a = replay(kind, original, side)['order_states'][side, 0][0]
            b = replay(kind, changed, side)['order_states'][side, 0][0]
            self.assertEqual((a['a0'], a['queue_snapshot_time']), (b['a0'], b['queue_snapshot_time']))
            self.assertEqual(a['fl'], b['fl'])

    def test_unchanged_initialization_preserves_legacy_replay(self):
        for rule in ('M', 'L', 'U'):
            for kind, side in self.cases():
                W = context(((0., 100.), (60., 100.), (400., 40.)), ((200., 110.), (600., 50.)), side)
                a = replay(kind, W, side, rule=rule)
                b = replay(kind, W, side, rule=rule, legacy=True)
                for key in ('fills', 'orders', 'cancels', 'log', 'stat', 'endpos', 'fl'):
                    self.assertEqual(a[key], b[key], (kind, side, rule, key))

    def test_invalid_assumptions_are_refused(self):
        self.assertEqual(initial_queue(20., 'M'), (20., 20., False))
        self.assertEqual(initial_queue(20., 'U'), (20., 0., False))
        for shown, rule, unknown in ((-1., 'M', 'zero'), (1., 'bad', 'zero'), (np.nan, 'M', 'bad')):
            with self.assertRaises(ValueError):
                initial_queue(shown, rule, unknown)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
    if peak > 2048.:
        raise RuntimeError(f'RSS cap exceeded: {peak:.1f} MiB')
    if not result.wasSuccessful():
        raise SystemExit(1)
