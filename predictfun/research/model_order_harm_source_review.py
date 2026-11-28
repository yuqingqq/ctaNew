"""Tiny source-function controls; no historical data, model imports or replay jobs.

Extract the reviewed functions with AST and run synthetic cases against them.
These cases establish reachable defects, not their frequency in historical rows.
"""
import ast
import bisect
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/home/yuqing/ctaNew')


def function(rel, name, namespace):
    source = (ROOT / rel).read_text()
    tree = ast.parse(source)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError(f'ambiguous function {rel}:{name}')
    node = nodes[0]
    exec(compile(ast.Module(body=[node], type_ignores=[]), rel, 'exec'), namespace)
    return namespace[name], hashlib.sha256(source.encode()).hexdigest(), node.lineno


def require_same_information_same_feature(a, b):
    if a != b:
        raise ValueError('future-only change altered decision feature')


def main():
    np = SimpleNamespace(isfinite=math.isfinite, nan=float('nan'),
                         searchsorted=lambda a, v, side='left':
                         (bisect.bisect_right if side == 'right' else bisect.bisect_left)(a, v))
    cases = {}; sources = {}
    features = []
    for future_size in (100.0, 900.0):
        ns = dict(bisect=bisect, np=np, K=SimpleNamespace(L_PLACE=40.0),
                  tfl=[0.0, 300.0], nf=2, opp=[52, 52], side=0, size=30.0,
                  INF=float('inf'), qrule='M', orders=[], live=[], lv=lambda pt, z=future_size: [100.0, z])
        place, sha, line = function('predictfun/research/pf_queue_place.py', 'place', ns)
        sources['pf_queue_place.py'] = dict(sha256=sha, function_line=line)
        order = place(50, 200.0)  # live 240; decision 250; changed book frame 300
        ahead, sha, line = function('predictfun/research/pf_next_price.py', 'ahead_at', dict(np=np, bisect=bisect))
        sources['pf_next_price.py'] = dict(sha256=sha, function_line=line)
        features.append(dict(a0=order['a0'], ahead=ahead(order, 250.0, [100.0, future_size], [0.0, 300.0])))
    require_same_information_same_feature(100.0, 100.0)  # causal control accepted
    try:
        require_same_information_same_feature(features[0], features[1])
    except ValueError:
        caught = True
    else:
        raise AssertionError('known future leakage was not detected')
    cases['future_queue_feature'] = dict(decision_ms=250, changed_frame_ms=300,
                                        results=features, causal_control_passed=True,
                                        future_change_refused=caught)

    ns = dict(K=SimpleNamespace(L_CANCEL=22.0), SIZE=30.0, MUT_LAT_KEEP=0, MUT_HARM_SIGN=0)
    values, sha, line = function('predictfun/research/pf_order_harm.py', 'branch_values', ns)
    sources['pf_order_harm.py'] = dict(sha256=sha, branch_function_line=line)
    # One 30-share fill, -2c/share, seen in two overlapping decision horizons.
    fill = [(1000.0, -60.0)]
    once = values(250.0, fill, [], 2000.0)[3]
    twice = once + values(500.0, fill, [], 2000.0)[3]
    assert once == -2.0 and twice == -4.0
    assert values(250.0, [(260.0, -60.0)], [], 2000.0) == (0.0, 0.0, 0.0, 0.0)
    cases['overlapping_rows'] = dict(distinct_physical_fills=1, decision_rows=2,
                                    one_row_delta=once, summed_row_delta=twice,
                                    latency_exclusion_control_passed=True)

    captured = []
    def sim_arm(w, *args, **kwargs):
        captured.append(dict(end=w['te'], last_decision=w['tg'][-1], n=w['n']))
        return [], [], 0
    w = dict(tg=list(range(0, 60000, 50)), stn=[True]*1200,
             vkl=[0.5]*1200, best=[50]*1200, bopp=[52]*1200, te=60000)
    fork, _, line = function('predictfun/research/pf_order_harm.py', 'run_fork',
                            dict(np=np, K=SimpleNamespace(sim_arm=sim_arm), SIZE=30.0, MUT_NOWAIT=0))
    fork(w, 1180, 0, [False]*1200, 'M', 50.0, 64000.0)
    assert captured[0]['end'] == 64000.0 and captured[0]['last_decision'] == 59950
    cases['terminal_mismatch'] = dict(reference_end_ms=w['te'], pull=captured[0],
                                      fork_function_line=line)
    print(json.dumps(dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(),
                          kind='synthetic_source_controls_not_historical_frequency',
                          sources=sources, cases=cases), indent=2))


if __name__ == '__main__':
    main()
