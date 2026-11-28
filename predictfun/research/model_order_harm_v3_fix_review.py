"""Synthetic source review only; no historical inputs, fitting, or research drivers."""
import ast
import bisect
import datetime as dt
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path('/home/yuqing/ctaNew/predictfun/research')


def function(file, name, env, repair_rest=False):
    tree = ast.parse((ROOT / file).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    if repair_rest:
        # Positive repair control only, in memory; no research source mutation.
        rest = [n for n in ast.walk(node) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'rest_' for t in n.targets)]
        assert len(rest) == 1
        rest[0].value = ast.parse("cur is not None and not cur['ref'] and cur['ta'] <= t "
                                 "and t < cur['tc'] and t < cur['tfill'] + L_FILL", mode='eval').body
    code = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(code, file, 'exec'), env)
    return env[name]


def config():
    tree = ast.parse((ROOT / 'pf_review_queue_reach.py').read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Tuple)
                      and [x.id for x in n.targets[0].elts] == ['L_PLACE', 'L_CANCEL', 'L_PENDCANCEL', 'L_FILL'])
    values = ast.literal_eval(assignment.value)
    result = dict(zip(['L_PLACE', 'L_CANCEL', 'L_PENDCANCEL', 'L_FILL'], values))
    tree = ast.parse((ROOT / 'pf_kp_s4.py').read_text())
    for key in ('HOLD_K', 'REST_COOL'):
        n = next(n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == key for t in n.targets))
        result[key] = ast.literal_eval(n.value)
    return SimpleNamespace(**result)


def label_case(filled, repair=False, known_immediately=False):
    K = config()
    if known_immediately:
        K.L_FILL = 0.0
    env = dict(np=np, bisect=bisect, INF=float('inf'), LANES=('rest',), MUT_LAG0=0,
               MUT_NO_EXIT=0, MUT_REST_NOCOOL=0, REST_STATE=None, REST_OFFSET=0,
               CTX={}, KP_STAT={}, SIZE=30.0, KP_DECIDE=None,
               KP_TRIG_RESTING_ONLY=0, KP_TRIG_IDLE_ONLY=0, **vars(K))
    function('pf_kp_s4.py', 'sim2', env, repair_rest=repair)
    sim = function('pf_kp_s4.py', 'sim_arm', env)
    v = dict(K=K, EVERY=5, MUT_GRID=0, MUT_VENUE_REST=0)
    function('pf_order_harm_v3.py', 'client_resting', v)
    steps_fn = function('pf_order_harm_v3.py', 'decision_steps', v)
    vg = [False] * 4 + [True] * 12 + [False] * 4
    tg = np.arange(0.0, 1000.0, 50.0); rec = {}; calls = []

    def recorder(k, t, cur):
        rec.setdefault(id(cur), cur); calls.append(t)
        return False
    w = dict(side=0, tf=np.array([0.0]), opp=[52], P5=None, S5=None,
             cache={50: np.array([100.0])}, tg=tg, vkl=[0.5] * 20, stn=[True] * 20,
             best=[50] * 20, bopp=[52] * 20, anc=np.array([400.0] if filled else []),
             ptl=[49] if filled else [], shl=[30.0] if filled else [], te=1000.0,
             kf=[0.0], jk=[0] * 20, X=None, n=20)
    sim(w, 'U', 0.0, 'rest', aux=vg, decide=recorder, idle_only=1, size=30.0)
    o = next(iter(rec.values())); steps, _ = steps_fn(4, 16, 20, o, tg.tolist(), lambda k: True)
    return dict(fill=filled, in_memory_repair=repair, immediate_notice=known_immediately,
                fill_notice_ms=o['tfill'] + K.L_FILL if filled else None,
                cancel_sent_ms=o['tc'] if np.isfinite(o['tc']) else None,
                decisions_before_500=[float(tg[k]) for k in steps if tg[k] < 500.0],
                client_resting_450=v['client_resting'](o, 450.0), callback_at_450=450.0 in calls)


def require_same_decisions(a, b):
    if a['decisions_before_500'] != b['decisions_before_500']:
        raise ValueError('An unobserved fill changes the label decision population')


def terminal_case(trade_time):
    K = config()
    env = dict(np=np, bisect=bisect, SIZE=30.0, NR=1, INF=float('inf'), K=K,
               L_NOTE=K.L_FILL, VIOL=('leg', 'order', 'live', 'priority', 'collision', 'position', 'cap'),
               MUT_REF_RELEASE=0, MUT_REST_TRUE=0, MUT_HELD_TRUE=0,
               MUT_LEG_LEAK=0, MUT_PRIORITY=0, INDEP=0)
    sim = function('pf_joint_ladder_v3.py', 'sim_joint', env)
    W = {}
    for s in (0, 1):
        pt = 50 if s == 0 else 52
        W[s] = dict(tg=np.arange(0.0, 60000.0, 50.0), tf=np.array([0.0, 60000.0, 60030.0, 61050.0]),
                    te=60000.0, ws=0.0, anc=np.array([trade_time] if s == 0 else []),
                    ptl=[49] if s == 0 else [], shl=[30.0] if s == 0 else [], P5=None, S5=None,
                    opp=[52, 52, 41, 41] if s == 0 else [50, 50, 39, 39], best=[pt] * 1200,
                    bopp=[52 if s == 0 else 50] * 1200, vkl=[0.5] * 1200, cache={pt: np.array([100.0] * 4)})
    r = sim(W, 'M', 0.0, None, {0: [False] * 1200, 1: [False] * 1200}, mode='none', nr=1)
    assert not any(r['stat']['viol'].values())
    # Exercise the exact new metric's mark selection, fill inclusion and price expression.
    tree = ast.parse((ROOT / 'pf_order_harm_policy_v2.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eval_day')
    value_env = dict(np=np, W=W, ctx={0: dict(tf=W[0]['tf'], mff=np.array([51.0, 51.0, 40.0, 40.0]))})
    for key in ('tend', 'jte', 'mid_end'):
        nodes = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == key for t in n.targets)]
        assert len(nodes) == 1
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'metric', 'exec'), value_env)
    include = [n for n in ast.walk(fn) if isinstance(n, ast.Compare)
               and ast.unparse(n) == "W[s]['ws'] <= t < W[s]['ws'] + 61000.0"]
    mark = [n for n in ast.walk(fn) if isinstance(n, ast.BinOp) and ast.unparse(n) == 'sg * (mid_end - pt)']
    assert len(include) == len(mark) == 1
    inc_code = compile(ast.Expression(include[0]), 'include', 'eval')
    mark_code = compile(ast.Expression(mark[0]), 'mark', 'eval')
    included = []; vend = 0.0; actual_at_60 = 0.0; common_later = 0.0
    for s, d, t, pt, q in r['fl']:
        value_env.update(s=s, t=t, pt=pt, sg=1 if s == 0 else -1)
        if eval(inc_code, value_env):
            included.append(t); vend += q * eval(mark_code, value_env) / 100.0
        if t <= 60000.0:
            actual_at_60 += q * (1 if s == 0 else -1) * (51.0 - pt) / 100.0
        common_later += q * (1 if s == 0 else -1) * (40.0 - pt) / 100.0
    return dict(trade_time_ms=trade_time, included_fill_times_ms=included, selected_mark_time_ms=60000.0,
                reported_vend_dollars=vend, actual_cash_plus_inventory_at_60s=actual_at_60,
                common_mark_after_cancellations_dollars=common_later, violations=r['stat']['viol'])


def require_terminal_consistency(case):
    if any(t > case['selected_mark_time_ms'] for t in case['included_fill_times_ms']):
        raise ValueError('A post-terminal fill is included in terminal PnL')


def cadence_case():
    v3 = ast.parse((ROOT / 'pf_order_harm_v3.py').read_text())
    step = next(ast.literal_eval(n.value) for n in v3.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'EVERY' for t in n.targets))
    env = dict(EVERY=step, MUT_GRID=0, MUT_HOLD=0, MUT_EVERY_STEP=0)
    wrap = function('pf_order_harm_policy_v2.py', 'cadence', env)
    calls = []; cnt = dict(keep=[0], pull=[0]); o = {'pt': 49}
    def keep(k, t, cur):
        calls.append(t); return k < 2
    cb = wrap(keep, [False] + [True] * 6, [-1] + [1] * 6, cnt, 0)
    pulls = [cb(k, k * 50.0, o) for k in (1, 2, 6)]
    assert calls == [50.0, 300.0] and pulls == [False, False, True]
    return dict(every_steps=step, cadence_ms=step * 50, callback_times_ms=[50, 100, 300],
                actual_score_times_ms=calls, pulls=pulls)


def main():
    names = ('pf_order_harm_v3.py', 'pf_order_harm_policy_v2.py', 'pf_joint_ladder_v3.py',
             'pf_kp_s4.py', 'pf_review_queue_reach.py')
    sha = lambda: {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}
    before = sha()
    a, b = label_case(False), label_case(True)
    label_caught = False
    try:
        require_same_decisions(a, b)
    except ValueError:
        label_caught = True
    assert label_caught
    ar, br = label_case(False, repair=True), label_case(True, repair=True)
    require_same_decisions(ar, br)
    known = label_case(True, repair=True, known_immediately=True)
    assert known['decisions_before_500'] != br['decisions_before_500']
    early, late, after_cancel = [terminal_case(t) for t in (59990.0, 60010.0, 60030.0)]
    require_terminal_consistency(early)
    terminal_caught = False
    try:
        require_terminal_consistency(late)
    except ValueError:
        terminal_caught = True
    assert terminal_caught and not after_cancel['included_fill_times_ms']
    timing = cadence_case()
    assert before == sha(), 'source changed during review'
    print(json.dumps(dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(),
                          kind='synthetic_source_review_no_historical_data_or_fitting', source_sha256=before,
                          label_population=dict(current=[a, b], defect_caught=label_caught,
                                                in_memory_repair=[ar, br], observed_fill_positive_control=known),
                          terminal_metric=dict(before_end_positive_control=early, post_end_counterexample=late,
                                               defect_caught=terminal_caught, after_cancel_no_fill_control=after_cancel),
                          cadence=timing), indent=2))


if __name__ == '__main__':
    main()
