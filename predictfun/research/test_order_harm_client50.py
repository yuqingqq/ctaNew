"""Synthetic controls only. Run with unittest; never loads historical research data."""
import copy
import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_joint_ladder as JL
import pf_joint_ladder_v3 as J3
import pf_order_harm_client_lane as CL
import pf_order_harm_v3 as V3
import pf_order_harm_policy_v2 as P2


def normalized(x):
    if isinstance(x, dict):
        return {k: normalized(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [normalized(v) for v in x]
    if isinstance(x, (float, np.floating)) and math.isnan(x):
        return 'NaN'
    return x


def require_equal_prefix(a, b):
    if a != b:
        raise ValueError('unobserved fill changed decision prefix')


class Client50Controls(unittest.TestCase):
    def test_shared_engine_catches_existing_joint_mutants(self):
        with mock.patch.object(J3, 'sim_joint', CL.sim_joint):
            self.assertEqual(J3.selftest(), [])
            for name in ('MUT_REF_RELEASE', 'MUT_REST_TRUE', 'MUT_HELD_TRUE', 'MUT_LEG_LEAK', 'MUT_PRIORITY'):
                with self.subTest(mutant=name), mock.patch.object(CL, name, 1):
                    self.assertTrue(J3.selftest(), name)

    def test_default_engine_matches_frozen_v3(self):
        frames = {
            0: [(0., [(50, 100), (49, 100), (48, 100)], 52),
                (150., [(51, 50), (50, 40), (49, 30)], 53),
                (300., [(50, 70), (49, 30), (48, 10)], 52)],
            1: [(0., [(52, 100), (53, 100), (54, 100)], 50),
                (150., [(53, 50), (54, 40), (55, 30)], 51),
                (300., [(52, 70), (53, 30), (54, 10)], 50)],
        }
        legs = {0: [(400., 48, 80.), (600., 49, 20.)], 1: [(500., 54, 80.)]}
        guard = {s: [False] * 4 + [True] * 4 + [False] * 12 for s in (0, 1)}
        for q in ('M', 'U'):
            for shift in (0., 50., 250.):
                for mode, cap in (('none', None), ('hard', 60.)):
                    with self.subTest(q=q, shift=shift, mode=mode):
                        W = {s: JL._ctx(s, frames[s], legs[s]) for s in (0, 1)}
                        args = (q, shift, cap, guard)
                        old = J3.sim_joint(copy.deepcopy(W), *args, nr=3, mode=mode)
                        new = CL.sim_joint(copy.deepcopy(W), *args, nr=3, mode=mode)
                        self.assertEqual(normalized(old), normalized({k: new[k] for k in old}))
                        self.assertFalse(any(new['stat']['viol'].values()))

    def label_prefix(self, fill, state='client', notice=None):
        w = JL._ctx(0, [(0., [(50, 100), (49, 100)], 52)], [(400., 49, 30.)] if fill else [])
        guard = [False] * 4 + [True] * 12 + [False] * 4
        _, _, orders, _ = CL.reference_lane(w, 'U', 0., 0, guard, state=state)
        o = orders[0]
        if notice is not None:
            # An actually received cancellation is a positive control.
            o = dict(o, tc=notice)
        steps, _ = V3.decision_steps(4, 16, len(w['tg']), o, w['tg'].tolist(), lambda k: True)
        return [float(w['tg'][k]) for k in steps if w['tg'][k] < 500.]

    def test_unobserved_fill_does_not_remove_label_decisions(self):
        a, b = self.label_prefix(False), self.label_prefix(True)
        require_equal_prefix(a, b)
        self.assertEqual(a, [200., 250., 300., 350., 400., 450.])
        with self.assertRaisesRegex(ValueError, 'unobserved fill'):
            require_equal_prefix(a, self.label_prefix(True, state='compat'))
        self.assertEqual(self.label_prefix(True, notice=400.), [200., 250., 300., 350.])

    def test_reference_rung_offsets_both_sides(self):
        for side in (0, 1):
            for rung in (0, 2, 5):
                best, opp = (50, 52) if side == 0 else (52, 50)
                w = JL._ctx(side, [(0., [(best, 100)], opp)], [])
                _, _, orders, r = CL.reference_lane(w, 'U', 0., rung, [False] * 20)
                self.assertEqual(orders[0]['pt'], best - rung if side == 0 else best + rung)
                self.assertFalse(r['order_states'][(1 - side, 0)])

    def test_pull_fork_entry_same_across_episode(self):
        for side in (0, 1):
            best, opp = (50, 52) if side == 0 else (52, 50)
            trade = (1300., 47 if side == 0 else 55, 40.)
            w = JL._ctx(side, [(0., [(best, 100)], opp)], [trade], t_end=3000.)
            guard = [False] * 4 + [True] * 12 + [False] * 44
            first = V3.run_fork(w, 4, 2, guard, 'U', 0., 2750.)
            later = V3.run_fork(w, 9, 2, guard, 'U', 0., 2750.)
            self.assertEqual(first, later)
            # re-post at the first 50 ms step after the guard clears (800 ms), live 40 ms later; no 250 ms timer (USER 2026-11-14)
            self.assertEqual(first[1][0][0], 840.)

    def terminal_run(self, time):
        frames = {0: [(0., [(50, 100)], 52), (60030., [(39, 100)], 41)],
                  1: [(0., [(52, 100)], 50), (60030., [(41, 100)], 39)]}
        W = {s: JL._ctx(s, frames[s], [(time, 49, 30.)] if s == 0 else [], t_end=60000.) for s in (0, 1)}
        return CL.sim_joint(W, 'M', 0., None, {s: [False] * 1200 for s in (0, 1)}, nr=1, mode='none')

    def test_terminal_includes_cancel_pending_fill_at_later_mark(self):
        r = self.terminal_run(60010.)
        tf, mid = np.array([0., 60000., 60030.]), np.array([51., 51., 40.])
        terminal = CL.portfolio_terminal(r, tf, mid)
        self.assertEqual(terminal, dict(time=60200., mark=40., cash=-15., position=30., pnl=-3.))
        old_terminal = dict(r, terminal_time=60000.)
        with self.assertRaisesRegex(ValueError, 'post-terminal fill'):
            CL.portfolio_terminal(old_terminal, tf, mid)
        for bad, pattern in ((dict(r, endpos=31.), 'position'),
                             (dict(r, order_states={(0, 0): [dict(ref=False, rem=1., tr=float('inf'))]}), 'executable')):
            with self.assertRaisesRegex(ValueError, pattern):
                CL.portfolio_terminal(bad, tf, mid)
        with self.assertRaisesRegex(ValueError, 'finite terminal mark'):
            CL.portfolio_terminal(r, tf, np.array([51., 51., np.nan]))
        before = CL.portfolio_terminal(self.terminal_run(59990.), tf, mid)
        self.assertEqual(before['pnl'], -3.)
        after = CL.portfolio_terminal(self.terminal_run(60030.), tf, mid)
        self.assertEqual(after['pnl'], 0.)

    def test_cadence_and_existing_target_controls(self):
        self.assertEqual(V3.EVERY * 50, 50)
        self.assertEqual(P2.EVERY, V3.EVERY)
        self.assertEqual(V3.selftest(), [])
        self.assertTrue(all(V3.mutation_check().values()))
        self.assertEqual(P2.selftest(), [])
        self.assertTrue(all(P2.mutation_check().values()))

    def test_reject_old_artifacts_and_keep_cooldown(self):
        CL.require_contract(CL.stamp())
        for old in ({}, dict(CL.stamp(), decision_ms=250), dict(CL.stamp(), contract='old')):
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                CL.require_contract(old)
        self.assertIn('client50', V3.OUTD)
        self.assertIn('client50', P2.OUTD)
        self.assertEqual(CL.K.REST_COOL, 250.)
        self.assertEqual((CL.K.L_CANCEL, CL.K.L_PLACE, CL.L_NOTE), (22., 40., 100.))

    def test_synthetic_label_builder_retains_unobserved_fill_row(self):
        # Exercise the actual row writer on one invented window, with all
        # data sources replaced by in-memory frames. No historical reads.
        day = '20990101'
        guard = [False] * 4 + [True] * 12 + [False] * 104
        w = JL._ctx(0, [(0., [(50, 100), (49, 100), (48, 100), (47, 100), (46, 100)], 52)],
                    [(450., 49, 30.)], t_end=6000., v=[-0.2 if g else 0.5 for g in guard])
        w.update(m=1, tf_mff=(w['tf'], np.array([51.])))
        source = pd.DataFrame(np.zeros((120, len(V3.SD.NEW_FEAT))), columns=V3.SD.NEW_FEAT)
        source['m'] = 1; source['side'] = 0; source['k'] = np.arange(120)
        fair = pd.DataFrame(dict(mid=[1] * 120, tau=w['tg'], p_bn=[50.] * 120,
                                 tau_rem=[240.] * 120, sigma=[0.01] * 120))

        def read(path, columns=None, **kwargs):
            path = str(path)
            if path == f'{V3.SD.OUTD}/rows_{day}.parquet':
                frame = source
            elif path == f'{V3.K.IO}/step5/{day}/rows_rl_d2_pk.parquet':
                frame = fair
            else:
                raise AssertionError(f'unexpected data access: {path}')
            return frame.copy() if columns is None else frame[columns].copy()

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(V3, 'OUTD', tmp), mock.patch.object(V3.K, 'load_day', return_value={'W0': {1: 0}}), \
                    mock.patch.object(V3.K, 'window_sides', return_value=iter([w])), \
                    mock.patch.object(pd, 'read_parquet', side_effect=read), contextlib.redirect_stdout(io.StringIO()):
                V3.rows_day(day)
            rows = pd.read_parquet(Path(tmp) / f'rows_{day}.parquet')
            receipt = json.loads((Path(tmp) / f'rows_{day}.json').read_text())
            CL.require_contract(receipt)
            own = rows[(rows.rung_i == 0) & (rows.k < 10)]
            self.assertEqual(own.k.tolist(), list(range(4, 10)))
            self.assertEqual(receipt['stat']['rows'], len(rows))
            self.assertGreater(receipt['stat']['pconst_checks'], 0)
            self.assertEqual(receipt['stat']['partition_bad'], 0)


if __name__ == '__main__':
    unittest.main()
