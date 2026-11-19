"""Bounded synthetic and saved-ledger checks; never fits or evaluates a policy.

Run through PF_SLOT=codex run_light.sh with this file's absolute path.
"""
import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

import pf_pipeline_contract as P

ROOT = Path(__file__).resolve().parents[2]
PINNED_ENGINE = 'a2f13f37d58d546b949bb4919725739978a47c76'
DAY = '20261023'
SCENARIOS = ('M0|A58|L37|N100', 'M50|A58|L37|N100', 'M250|A58|L37|N100')
CONTRACT = 'JOIN30/END=birth+5000/queueM/structuralKeeper/scenarios-v6'
CONTINUATION = 'no_repost_to_common_END;zero_continuation_after_END'


def target(**kwargs):
    base = P.Target('toy:bid:1', 'bid', 'post', 1000., 1058., 6000.,
                    51., 50., 30., SCENARIOS[1], CONTINUATION, CONTRACT)
    return replace(base, **kwargs)


def forecast(t, atoms=(P.Atom(1., 10., 0.),), **kwargs):
    return replace(P.Forecast(t, atoms, t.decision_ms, 0., 'synthetic', 'fixture'), **kwargs)


def state(t, **kwargs):
    live = t.kind == 'retain'
    return replace(P.ClientState(t.order_id, t.side, t.decision_ms,
                   'live' if live else 'absent', t.max_quantity if live else 0.,
                   0., 30. if live and t.side == 'bid' else 0.,
                   30. if live and t.side == 'ask' else 0., 60.), **kwargs)


def decide(t, f, s=None, e=None):
    return P.decide_one(t, f, s or state(t), e or P.Economics(t, 0., 0., 0., 0.), allow_synthetic=True)


class ContractTests(unittest.TestCase):
    def test_joint_dependence(self):
        t = target()
        # Same marginal means, opposite fill/move association: independent
        # multiplication would predict +5 in both, reversing the bad decision.
        bad = forecast(t, (P.Atom(.5, 10., -3.), P.Atom(.5, 0., 3.)))
        good = forecast(t, (P.Atom(.5, 0., -3.), P.Atom(.5, 10., 3.)))
        self.assertEqual(P.map_value(t, bad).retained, -10.)
        self.assertEqual(P.map_value(t, good).retained, 20.)
        self.assertEqual(decide(t, bad).action, 'wait')
        self.assertEqual(decide(t, good).action, 'post')

    def test_complement_and_separate_sides(self):
        b = target(); a = target(order_id='toy:ask:1', side='ask', price=50., reference_mid=49.)
        self.assertEqual(P.map_value(b, forecast(b)), P.map_value(a, forecast(a)))
        for t in (b, a):
            self.assertEqual(decide(t, forecast(t)).action, 'post')
        a = replace(a, reference_mid=51.)
        self.assertEqual(decide(a, forecast(a)).action, 'wait')

    def test_no_fill_and_unclipped_tail(self):
        t = target()
        f = forecast(t, (P.Atom(.8, 0., 0.), P.Atom(.1, 10., 0.), P.Atom(.1, 30., -21.)))
        v = P.map_value(t, f)
        self.assertAlmostEqual(v.retained, -59.)
        self.assertAlmostEqual(v.severe_excess, 57.)
        self.assertAlmostEqual(v.ordinary_loss, 3.)
        self.assertAlmostEqual(v.fill_probability, .2)
        self.assertAlmostEqual(v.severe_probability, .1)
        self.assertLessEqual(v.severe_probability, v.loss_probability)
        self.assertLessEqual(v.loss_probability, v.fill_probability)

    def test_cancel_boundary_and_unavoidable_prefix(self):
        t = target(kind='retain', effective_ms=1037.)
        atom = P.realized_atom(t, [1010., 1036., 1037., 1100.], [5., 2., 3., 4.], 49.)
        self.assertEqual(atom.quantity, 7.)
        self.assertEqual(decide(t, forecast(t, (atom,))).action, 'pull')
        positive = P.realized_atom(t, [1010., 1036., 1037., 1100.], [5., 2., 3., 4.], 52.)
        self.assertEqual(decide(t, forecast(t, (positive,))).action, 'keep')

    def test_zero_advantage_preserves_state(self):
        for kind, action in [('post', 'wait'), ('retain', 'keep')]:
            t = target(kind=kind)
            self.assertEqual(decide(t, forecast(t, (P.Atom(1., 0., 0.),))).action, action)

    def test_missing_forecast_and_pending_actions(self):
        t = target(kind='retain')
        self.assertEqual(decide(t, None).action, 'keep')
        for pending in ('post_pending', 'cancel_pending', 'closed'):
            d = decide(t, None, state(t, lifecycle=pending))
            self.assertEqual(d.action, 'none')
            self.assertEqual(d.reserved_bid_after, 30.)
        t = target()
        self.assertEqual(decide(t, None).action, 'wait')

    def test_cancel_never_releases_reservation(self):
        t = target(kind='retain')
        d = decide(t, forecast(t, (P.Atom(1., 30., -2.),)), state(t, reserved_bid=60.))
        self.assertEqual(d.action, 'pull')
        self.assertEqual(d.reserved_bid_after, 60.)
        new = target(order_id='toy:bid:2')
        self.assertEqual(decide(new, forecast(new), state(new, reserved_bid=d.reserved_bid_after)).reason, 'reservation_cap')
        with self.assertRaisesRegex(ValueError, 'reservation missing'):
            decide(t, None, state(t, reserved_bid=0.))

    def test_two_posts_share_capacity(self):
        t = target()
        d = decide(t, forecast(t), state(t, cap=40.))
        self.assertEqual(d.action, 'post')
        next_t = replace(t, order_id='toy:bid:2')
        d2 = decide(next_t, forecast(next_t), state(next_t, reserved_bid=d.reserved_bid_after, cap=40.))
        self.assertEqual(d2.reason, 'reservation_cap')

    def test_inventory_and_explicit_action_cost(self):
        t = target()
        self.assertEqual(decide(t, forecast(t), state(t, known_position=40.)).reason, 'reservation_cap')
        self.assertEqual(decide(t, forecast(t), e=P.Economics(t, 0., 0., 11., 0.)).action, 'wait')
        self.assertEqual(decide(t, forecast(t), e=P.Economics(t, 0., 0., 0., 11.)).action, 'wait')
        # Continuation is action-relative. Losing queue value can outweigh a pull.
        t = target(kind='retain')
        self.assertEqual(decide(t, forecast(t, (P.Atom(1., 10., -2.),)),
                                e=P.Economics(t, -11., 0., 0., 0.)).action, 'keep')

    def test_economic_and_client_identity(self):
        t = target(); other = replace(t, order_id='other')
        with self.assertRaisesRegex(ValueError, 'economic target mismatch'):
            decide(t, forecast(t), e=P.Economics(other, 0., 0., 0., 0.))
        with self.assertRaisesRegex(ValueError, 'client/target identity'):
            decide(t, forecast(t), state(other))
        with self.assertRaisesRegex(ValueError, 'remaining quantity mismatch'):
            t = target(kind='retain')
            decide(t, forecast(t), state(t, remaining=20.))

    def test_wrong_target_refused(self):
        t = target(); f = forecast(t)
        changes = [dict(order_id='other'), dict(side='ask'), dict(kind='retain'),
                   dict(endpoint_ms=6001.), dict(decision_ms=1001.), dict(effective_ms=1059.),
                   dict(scenario=SCENARIOS[0]), dict(continuation='rolling_1s'),
                   dict(execution_contract='legacy'), dict(reference_mid=52.), dict(max_quantity=20.)]
        for change in changes:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'target mismatch'):
                P.map_value(replace(t, **change), f)
        # Explicitly test epoch-sized timestamps: relative tolerances are unsafe.
        t = target(decision_ms=1790000000000., effective_ms=1790000000058., endpoint_ms=1790000005000.)
        with self.assertRaisesRegex(ValueError, 'target mismatch'):
            P.map_value(replace(t, decision_ms=t.decision_ms+1), forecast(t))

    def test_future_information_and_oracle_refused(self):
        t = target()
        for f in (forecast(t, feature_asof_ms=1001.), forecast(t, trained_through_ms=1000.)):
            with self.assertRaisesRegex(ValueError, 'future information'):
                decide(t, f)
        with self.assertRaisesRegex(ValueError, 'nonpredictive'):
            decide(t, forecast(t, source='realized_accounting'))
        with self.assertRaisesRegex(ValueError, 'nonpredictive'):
            P.decide_one(t, forecast(t), state(t), P.Economics(t, 0., 0., 0., 0.))

    def test_saved_hash_guard(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d); (path/'artifact').write_text('positive-control')
            (path/'receipt.json').write_text(json.dumps(dict(outputs={'artifact': sha(path/'artifact')})))
            self.assertIn('outputs', load_saved(path))
            (path/'artifact').write_text('planted-corruption')
            with self.assertRaisesRegex(ValueError, 'saved artifact changed'):
                load_saved(path)

    def test_invalid_distribution_refused(self):
        t = target()
        for atoms in ((), (P.Atom(.5, 1., 0.),), (P.Atom(-1., 1., 0.), P.Atom(2., 1., 0.)),
                      (P.Atom(1., -1., 0.),), (P.Atom(1., 31., 0.),),
                      (P.Atom(1., 1., 100.),), (P.Atom(1., 1., float('nan')),)):
            with self.subTest(atoms=atoms), self.assertRaises(ValueError):
                P.map_value(t, forecast(t, atoms))

    def test_bad_fill_ledger_refused(self):
        t = target()
        for ts, qs in (([1100.], []), ([1100.], [-1.]), ([1100.], [31.]),
                       ([6100.], [1.]), ([1200., 1100.], [1., 1.])):
            with self.subTest(times=ts), self.assertRaises(ValueError):
                P.realized_atom(t, ts, qs, 52.)

    def test_reference_price_invariance(self):
        t = target(); t2 = replace(t, reference_mid=55.)
        a = P.realized_atom(t, [1100.], [10.], 49.)
        b = P.realized_atom(t2, [1100.], [10.], 49.)
        self.assertEqual(P.map_value(t, forecast(t, (a,))).retained,
                         P.map_value(t2, forecast(t2, (b,))).retained)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_saved(directory):
    r = json.loads((directory/'receipt.json').read_text())
    for name, expected in r['outputs'].items():
        P.require(Path(name).name == name, 'unexpected receipt output path')
        P.require(sha(directory/name) == expected, 'saved artifact changed: '+name)
    return r


def saved_ledger_check():
    import numpy as np
    import pandas as pd
    data = ROOT/'data/predict_fun/research'
    kp = data/'keep_pull_20261117'/DAY
    sc = data/'keep_pull_scenarios_v6_20261118'/DAY
    kr = load_saved(kp); sr = load_saved(sc)
    P.require(sr['keep_pull_receipt'] == sha(kp/'receipt.json'), 'scenario/base receipt mismatch')
    source = subprocess.check_output(['git', 'show', PINNED_ENGINE+':predictfun/research/pf_keep_pull_scenarios.py'], cwd=ROOT)
    P.require(hashlib.sha256(source).hexdigest() == sr['source_sha256'], 'unexpected scenario producer')
    orders = pd.read_parquet(kp/'orders.parquet')
    clock_fills = pd.read_parquet(kp/'fills.parquet')
    scenarios = pd.read_parquet(sc/'orders.parquet')
    fills = pd.read_parquet(sc/'fills.parquet')
    P.require(not orders.oid.duplicated().any() and not scenarios.duplicated(['scen', 'oid']).any(), 'duplicate saved identity')
    P.require(set(orders.day.astype(str)) == {DAY} and orders.side.isin([0, 1]).all(), 'day/side contract')
    path_ids = set(orders.loc[orders.status.isin(['eligible', 'refused_post_only']), 'oid'])
    stats = {}
    for scenario in SCENARIOS:
        # As in the scenario report, M0/M250 are the already re-simulated
        # uniform-clock branches in KEEP/PULL, not rows in the v6 scenario file.
        clock = float(scenario.split('|')[0][1:])
        if clock == 50.:
            rows = scenarios[scenarios.scen == scenario].set_index('oid')
            fs = fills[fills.scen == scenario]
        else:
            tag = '_m0' if clock == 0. else '_m250'
            z = orders[orders.oid.isin(path_ids)]
            rows = pd.DataFrame(dict(oid=z.oid, refused=z.status == 'refused_post_only',
                                     q=z['q'+tag], v=z['v'+tag], t_full=z['t_full'+tag])).set_index('oid')
            fs = clock_fills[clock_fills.clock == clock]
        P.require(set(rows.index) == path_ids and set(fs.oid) <= path_ids, 'scenario population changed')
        groups = {oid: z.sort_values('t') for oid, z in fs.groupby('oid', sort=False)}
        counts = Counter(); max_error = 0.; severe = 0
        for o in orders.itertuples(index=False):
            if o.oid not in path_ids:
                counts['post:'+o.status] += 1; counts['retain:'+o.status] += 1
                continue
            r = rows.loc[o.oid]
            z = groups.get(o.oid)
            times = np.zeros(0) if z is None else z.t.to_numpy(float)
            quantities = np.zeros(0) if z is None else z.q.to_numpy(float)
            P.require(np.all(times >= 58.) and np.all(times <= 5000.), 'scenario fill lifetime')
            sgn = 1. if o.side == 0 else -1.
            # Birth mid is an explicitly causal reference, NOT an invented current mid.
            mid = float(o.p + sgn*o.spread/2.)
            values = quantities*sgn*(o.mark-o.p)
            P.require(abs(float(quantities.sum())-r.q) < 1e-7
                      and abs(float(values.sum())-r.v) < 1e-7, 'saved quantity/value reconciliation')
            if r.refused:
                P.require(r.q == 0 and r.v == 0, 'refused order has a fill')
            for kind in ('post', 'retain'):
                if kind == 'retain':
                    why = ('refused_post_only' if r.refused else
                           'structural_cancel_already_requested' if np.isfinite(o.g_struct) and o.g_struct <= 100. else
                           'full_fill_known' if np.isfinite(r.t_full) and r.t_full+100. <= 100. else '')
                    if why:
                        counts[kind+':'+why] += 1
                        continue
                dt = 0. if kind == 'post' else 100.
                cutoff = 58. if kind == 'post' else 137.
                known = 0. if kind == 'post' else float(quantities[times+100. <= dt].sum())
                t = P.Target(f'{DAY}:{int(o.m)}:{int(o.oid)}', 'bid' if o.side == 0 else 'ask', kind,
                             float(o.t+dt), float(o.t+cutoff), float(o.t+5000.),
                             mid, float(o.p), 30.-known, scenario, CONTINUATION, CONTRACT)
                atom = P.realized_atom(t, (times+o.t).tolist(), quantities.tolist(), float(o.mark))
                f = P.Forecast(t, (atom,), float(o.t), float(o.t-1.), 'realized_accounting', 'oracle-accounting-only')
                value = P.map_value(t, f)
                direct = float(values[times >= cutoff].sum())
                prefix = float(values[times < cutoff].sum())
                err = max(abs(value.retained-direct), abs(prefix+value.retained-r.v))
                max_error = max(max_error, err)
                P.require(err < 1e-7, 'mapped branch does not reconcile to saved engine')
                counts[kind+':refused_post_zero' if kind == 'post' and r.refused else kind+':eligible'] += 1
                counts[kind+':checked'] += 1
                severe += int(value.severe_excess > 0)
        post_n = sum(n for key, n in counts.items() if key.startswith('post:') and not key.endswith(':checked'))
        retain_n = sum(n for key, n in counts.items() if key.startswith('retain:') and not key.endswith(':checked'))
        P.require(post_n == retain_n == len(orders), 'silent population drop')
        stats[scenario] = dict(statuses=dict(counts), original_orders=len(orders),
                              accounting_checks=counts['post:checked']+counts['retain:checked'],
                              nonzero_severe_excess_labels=severe, max_reconciliation_error_cents=max_error)
    return dict(day=DAY, consumed=True, source_asof=sr['asof'], base_asof=kr['asof'],
                pinned_engine=PINNED_ENGINE, receipts={str(p.relative_to(ROOT)): sha(p) for p in (kp/'receipt.json', sc/'receipt.json')},
                inputs={str((d/n).relative_to(ROOT)): sha(d/n) for d, ns in ((kp, ['orders.parquet', 'fills.parquet']), (sc, ['orders.parquet', 'fills.parquet'])) for n in ns},
                scenarios=stats, predictor_fitted=False, policy_evaluated=False,
                realized_values_used_only_for_accounting=True, raw_tape_read=False, untouched_days_read=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['selftest', 'saved-check'])
    p.add_argument('--receipt', type=Path, required=True)
    args = p.parse_args()
    P.require(not args.receipt.exists(), 'receipt exists; use a superseding path')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
    test_count = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    P.require(result.wasSuccessful(), 'synthetic controls failed')
    receipt = dict(asof=datetime.now(timezone.utc).isoformat(),
                   source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   sources={str(p.relative_to(ROOT)): sha(p) for p in (Path(P.__file__), Path(__file__))},
                   synthetic_tests=test_count, passed=True,
                   scope='interface/accounting checks only; no forecast skill, policy acceptance or profitability')
    if args.mode == 'saved-check':
        receipt['saved_check'] = saved_ledger_check()
    receipt['completed_asof'] = datetime.now(timezone.utc).isoformat()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as f:
        json.dump(receipt, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(receipt=str(args.receipt), passed=True, synthetic_tests=test_count)))


if __name__ == '__main__':
    main()
