"""Audit the published maker next-trade study before using it as model evidence.

Reconstruct descriptive report calculations from saved consumed-day tables.
Reuse (do not refit or verify) the published R554 classifier figures. Exercise
the exact producer functions on observable-equivalent exhausted/cancelled and
retained/replaced orders. No new historical policy, forecast, null, interval,
selection, protected-day read or live change.
"""
import ast
import bisect
import copy
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import pf_model_upgrade_spec as M
from . import pf_recovery_experiment_spec as R
from . import model_move_down_integration_review as Previous
from . import model_recovery_stage_review as Identity

PRODUCER = '81ae9c60'
PREFIX = 'learn_pull_integration_review_v1'
STUDY = M.ROOT/'data/predict_fun/research/learn_pull_20261113'
TABLES = ('obs', 'bigfills', 'bigevents')


def compile_functions(tree, names, namespace):
    nodes = [copy.deepcopy(Previous.function(tree, name)) for name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<exact-learn-pull>', 'exec'), namespace)


def controls(tree, exit_tree):
    ns = dict(bisect=bisect, np=np, W_MAX=3000., E2_TOL=250., L_CANCEL=22.,
              MUT_OUTCOME_INV=0, MUT_TAKER_SIGN=0, MUT_PULL_LATE=0, MUT_STOP_TAKEN=0)
    compile_functions(exit_tree, ['classify'], ns)
    ns['EX'] = SimpleNamespace(classify=ns['classify'])
    compile_functions(tree, ['next_trade_obs', 'positions', 'pos_before', 'leg_events',
                            'inv_dir', 'we_pull', 'selftest', 'mutation_check'], ns)
    assert ns['selftest']() == []
    mutations = ns['mutation_check']()
    assert len(mutations) == 3 and all(mutations.values())

    # Both FIFO worlds have identical recorded trades and shown totals.
    # A: initial [maker7:5, maker9:45]; maker7 fully fills at t=100.
    # B: initial [maker7:50]; 45 remain at t=100, cancelled at t=200,
    #    then maker9 posts 45 before the next observed frame/trade.
    # Both show 45 at t=150 and t=250 and maker9 fills 3 at t=300.
    orders = [(100., [(7, 50, 5.)]), (300., [(9, 50, 3.)])]
    pa = lambda p, t: p == 50
    gone, ambiguous, tied = ns['next_trade_obs'](1, orders, pa, {})
    assert len(gone) == 1 and gone[0]['stayed'] == 0 and ambiguous == tied == 0
    world_a = dict(initial_queue=[(7, 5.), (9, 45.)], maker7_residual_after_proof=0.,
                   maker7_cancelled=False, total_after_proof=45., next_trade_maker=9)
    world_b = dict(initial_queue=[(7, 50.)], maker7_residual_after_proof=45.,
                   maker7_cancelled=True, total_after_proof=45., next_trade_maker=9)
    assert world_a['total_after_proof'] == world_b['total_after_proof']
    assert world_a['maker7_cancelled'] != world_b['maker7_cancelled']

    # A later fill by the same signer is compatible with the original residual
    # and with a replacement order from that signer. Order hash is not an input.
    same_signer = [(100., [(7, 50, 5.)]), (300., [(7, 50, 3.)])]
    stayed, _, _ = ns['next_trade_obs'](1, same_signer, pa, {})
    assert len(stayed) == 1 and stayed[0]['stayed'] == 1
    no_presence, _, _ = ns['next_trade_obs'](1, orders, lambda p, t: False, {})
    assert no_presence == []
    ambiguous_rows, ambiguous_n, _ = ns['next_trade_obs'](1, orders, pa, {(7, 50): [200.]})
    assert ambiguous_rows == [] and ambiguous_n == 1

    # Execute the exact nested frame predicate. Showing p anywhere in the
    # depth array does not identify its maker or establish that p is best.
    nested = [n for n in ast.walk(Previous.function(tree, 'day_run'))
              if isinstance(n, ast.FunctionDef) and n.name == 'present_after']
    if len(nested) != 1:
        raise ValueError('frame predicate identity differs')
    frame_ns = dict(bisect=bisect, np=np, tfl=[0., 150., 350.],
                    P5=np.array([[50, 51], [49, 50], [49, 50]]),
                    S5=np.array([[50., 10.], [1., 45.], [1., 42.]]))
    exec(compile(ast.Module(body=[copy.deepcopy(nested[0])], type_ignores=[]),
                 '<exact-presence-predicate>', 'exec'), frame_ns)
    assert frame_ns['present_after'](50, 100.) and frame_ns['P5'][1, 0] != 50
    frame_ns['S5'][1, 1] = 0.
    assert not frame_ns['present_after'](50, 100.)

    # Feature anchor precedes the next trade, but can follow the hidden pull.
    # Exact grid selection from day_run; the action time is not an argument.
    day_node = Previous.function(tree, 'day_run')
    anchor = [n for n in ast.walk(day_node) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'k' for t in n.targets)
              and any(isinstance(t, ast.Name) and t.id == 'DEC_LAG' for t in ast.walk(n))]
    if len(anchor) != 1:
        raise ValueError('feature-time anchor identity differs')
    feature_ns = dict(np=np, tg=np.arange(0., 1050., 50.), o={'t2': 1000.}, DEC_LAG=100.)
    exec(compile(ast.Module(body=[copy.deepcopy(anchor[0])], type_ignores=[]),
                 '<exact-feature-anchor>', 'exec'), feature_ns)
    feature_time = float(feature_ns['tg'][feature_ns['k']])
    assert 100. < 200. < feature_time < 1000.
    gap_assignments = [n for n in ast.walk(Previous.function(tree, 'next_trade_obs'))
                       if isinstance(n, ast.keyword) and n.arg == 'gap']
    assert len(gap_assignments) == 1 and ast.unparse(gap_assignments[0].value) == 't - t1'
    # With unchanged information through t=200, either future next-trade
    # time is possible. This gap is unavailable at that common action time.
    future_gap_cases = [ns['next_trade_obs'](1, [(100., [(7, 50, 5.)]),
                        (t, [(9, 50, 3.)])], pa, {})[0][0]['gap'] for t in (1000., 2000.)]
    assert future_gap_cases == [900., 1900.]

    report = Previous.function(tree, 'report')
    training = [n for n in ast.walk(report) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'tr' for t in n.targets)]
    assert any(ast.unparse(n.value) == 'T5[T5.day != int(d)]' for n in training)
    return dict(producer_positive=True, producer_mutations=mutations,
        gone_not_cancellation=dict(output=gone[0], exhausted_world=world_a,
                                  cancelled_world=world_b, same_observed_trades=True),
        stayed_not_order_retention=dict(output=stayed[0], original_or_replacement_hash_unobserved=True),
        no_presence_and_ambiguous_interval_refused=True,
        price_present_not_at_best=True, absent_price_refused=True,
        feature_time_can_follow_hidden_action=dict(proof_ms=100., cancellation_ms=200.,
                                                  feature_ms=feature_time, next_trade_ms=1000.),
        gap_depends_on_future_next_trade=future_gap_cases,
        leave_one_day_out_includes_later_training_dates=True,
        historical_misclassification_frequency=None)


def report_function(tree, namespace):
    node = copy.deepcopy(Previous.function(tree, 'report'))
    model_loops = [n for n in node.body if isinstance(n, ast.For)
                   and any(isinstance(c, ast.Call) and ast.unparse(c.func) == 'lgb.train'
                           for c in ast.walk(n))]
    if len(model_loops) != 1:
        raise ValueError('model fit block identity differs')
    i = node.body.index(model_loops[0])
    node.body[i:i+1] = ast.parse("out=published['R554']['auc']\nimp=published['R554']['top_gain']").body
    tail = node.body[-3:]
    names = [ast.unparse(n.value.func) for n in tail if isinstance(n, ast.Expr)
             and isinstance(n.value, ast.Call)]
    if len(names) != 3 or names[0] != 'json.dump' or not names[1].endswith('.write') or names[2] != 'print':
        raise ValueError('report publication block differs')
    node.body = node.body[:-3] + ast.parse('return locals()').body
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 '<readonly-report-with-published-model-values>', 'exec'), namespace)
    return namespace['report']


def metadata(record):
    st = record['stat']
    if (record['days'] != 1. or st['windows'] != 288 or st['window_sides'] != 576
            or any(st[k] for k in ('events_differ', 'unpaired_windows', 'inv_missing'))
            or set(st['planted_caught']) != {'MUT_OUTCOME_INV', 'MUT_TAKER_SIGN', 'MUT_PULL_LATE'}
            or not all(st['planted_caught'].values())):
        raise ValueError('producer coverage/control metadata differs')


def run():
    R.check()
    own = Path(__file__).resolve(); rel = str(own.relative_to(M.WORK))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=M.WORK, text=True).strip()
    if subprocess.check_output(['git', 'show', commit+':'+rel], cwd=M.WORK) != own.read_bytes():
        raise ValueError('commit instrument before execution')
    producer = subprocess.check_output(['git', 'rev-parse', PRODUCER], cwd=M.ROOT, text=True).strip()
    sources = {}
    for name in ('pf_learn_pull.py', 'run_learn_pull.sh', 'pf_exit_check.py', 'pf_move_cases.py',
                 'pf_kp_s4.py', 'pf_pull_depth.py', 'pf_robust_check.py', 'pf_qr_queue_race.py'):
        p = M.ROOT/'predictfun/research'/name
        if subprocess.check_output(['git', 'show', producer+':'+str(p.relative_to(M.ROOT))], cwd=M.ROOT) != p.read_bytes():
            raise ValueError('producer source changed: '+name)
        sources[str(p)] = M.sha(p)
    tree = ast.parse((M.ROOT/'predictfun/research/pf_learn_pull.py').read_text())
    exit_tree = ast.parse((M.ROOT/'predictfun/research/pf_exit_check.py').read_text())
    paths = [STUDY/'learn_pull.txt', STUDY/'learn_pull.json',
             M.ROOT/'predictfun/handoffs/20261113_learn_pull_results.md']
    for day in M.DAYS:
        paths += [STUDY/('day_'+day+'.json'), *(STUDY/(t+'_'+day+'.parquet') for t in TABLES)]
    inputs = {str(p): M.sha(p) for p in paths}
    tested = controls(tree, exit_tree)
    original = (STUDY/'learn_pull.txt').read_text()
    match = re.search(r'as of ([0-9TZ:-]+); controls passed', original)
    if not match:
        raise ValueError('published as-of header absent')
    first = json.loads((STUDY/('day_'+M.DAYS[0]+'.json')).read_text()); metadata(first)
    bad = copy.deepcopy(first); bad['stat']['events_differ'] = 1
    Identity.refused(lambda: metadata(bad))
    Identity.refused(lambda: Identity.bind(STUDY/'learn_pull.json', '0'*64))
    R.write(PREFIX+'_declaration.json', dict(as_of=M.now(), source_commit=commit,
        source_sha={str(p.relative_to(M.WORK)):M.sha(p) for p in
                    (own, Path(Previous.__file__), Path(Identity.__file__))},
        producer_commit=producer, producer_source_sha=sources, input_sha=inputs,
        controls=tested, scope=__doc__, days=M.DAYS, published_R554_reused_not_verified=True,
        validation=False, live_changed=False))
    published = json.loads((STUDY/'learn_pull.json').read_text())
    ns = dict(np=np, pd=pd, json=json, DAYS9=M.DAYS, OUTD=str(STUDY), MIN_N=30,
        GAPB=(('0-250 ms',0.,250.), ('250-1000 ms',250.,1000.), ('1000-3000 ms',1000.,3000.)),
        INV_MIN=10., NAMED=['v','P','adv_bn_r250','adv_bn_r1000','adv_bn_flow250','adv_bn_imb',
                           'adv_pf_dm250','adv_pf_imb1','own_d250','opp_d250','pf_spread','size_p',
                           'gap','inv','t_rel','sigma','act_rel'],
        XN=[], published=published,
        time=SimpleNamespace(strftime=lambda *a:match[1], gmtime=lambda:None))
    # XN only builds the removed model's feature list; source assertions above
    # and controls retain its original future-gap and nonchronological design.
    result = report_function(tree, ns)()
    if '\n'.join(result['L'])+'\n' != original:
        raise ValueError('published descriptive report does not reconstruct')
    if M.native(result['R']) != published:
        raise ValueError('published readings differ')
    for record in result['stats'].values(): metadata(record)
    A, BF, BE = (result[k] for k in ('A','BF','BE'))
    for frame in (A, BF, BE):
        if not frame.day.isin([int(d) for d in M.DAYS]).all():
            raise ValueError('undeclared table population')
    counts = {day:{name:int(frame.day.eq(int(day)).sum()) for name,frame in
                   zip(TABLES,(A,BF,BE))} for day in M.DAYS}
    for day in M.DAYS:
        if counts[day]['obs'] != result['stats'][day]['stat']['obs']:
            raise ValueError('observation count differs')
    observed = []
    for group in ('top 5','field'):
        x = A[A.grp == group]
        observed.append(dict(group=group,n=len(x),not_refilled=int(x.gone.sum()),
                             not_refilled_fraction=float(x.gone.mean()),
                             guard_signal_fraction=float(x.we_pull.mean()),
                             absent_original_or_residual_quantity=True,
                             absent_order_hash=True))
    assert not any(c in A.columns for c in ('order_hash','original_quantity','remaining_quantity'))
    statuses = {key:sum(v['stat'][key] for v in result['stats'].values())
                for key in ('e2_ambiguous','same_stamp','inv_missing','unpaired_windows')}
    for path, digest in {**sources, **inputs}.items(): Identity.bind(path,digest)
    receipt = dict(as_of=M.now(), status='DESCRIPTIVE_RECONSTRUCTION_WITH_LABEL_LIMITS',
        producer_as_of=match[1], declaration_sha=M.sha(R.OUT/(PREFIX+'_declaration.json')),
        days=M.DAYS, counts=counts, observed_next_trade_membership=observed,
        producer_readings=published, descriptive_report_exact=True,
        R554_model_values_reused_not_independently_reproduced=True,
        producer_daily_metadata=result['stats'], statuses=statuses, controls=tested,
        cancellation_labels_identified=False, causal_policy_value_verified=False,
        historical_false_gone_or_false_stayed_count=None,
        source_replay_performed=False, validation=False, live_changed=False,
        limits=['A sole-maker stopping fill plus aggregate residual depth does not prove that maker has quantity left.',
                'GONE is lack of a later signer fill; it does not distinguish full fill from cancellation.',
                'STAYED identifies the same signer, not continued existence or priority of the same order.',
                'The frame predicate tests presence among shown levels, not best-price or maker identity.',
                'Features at next-trade minus 100ms may follow the hidden pull; gap uses the future next-trade time.',
                'R554 uses leave-one-day-out including later dates and inherited P score limitations.',
                'The guard is a signal fired on maker-selected intervals, not our sequential order availability.',
                'Next-trade volume-weighted marks are neither our counterfactual fills nor portfolio value.',
                'Part B event fills do not prove continuous quoting; saved aggregate rows lack event/fill identity.',
                'Within-window inferred position is not full inventory; groups are ranked by observed volume.'])
    R.write(PREFIX+'.json', receipt)
    print(json.dumps(dict(as_of=receipt['as_of'], producer_as_of=match[1],
        receipt_sha=M.sha(R.OUT/(PREFIX+'.json')), observed=observed, statuses=statuses,
        descriptive_report_exact=True, model_refitted=False)))


if __name__ == '__main__':
    run()
