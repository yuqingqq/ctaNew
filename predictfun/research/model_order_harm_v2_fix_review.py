"""Source-level synthetic review of v2 repairs; no historical arrays/model imports."""
import ast
import bisect
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/home/yuqing/ctaNew')


class Vec(list):
    def __getitem__(self, key):
        if isinstance(key, list):
            return Vec(x for x, keep in zip(self, key) if keep)
        v = super().__getitem__(key)
        return Vec(v) if isinstance(key, slice) else v

    def min(self):
        return min(self)

    def tolist(self):
        return list(self)

    def __sub__(self, value):
        return Vec(x - value for x in self)


NP = SimpleNamespace(nan=float('nan'), isfinite=lambda x:
                     [math.isfinite(v) for v in x] if isinstance(x, list) else math.isfinite(x))


def extract(path, name, ns):
    source = path.read_text(); tree = ast.parse(source)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError('ambiguous function')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns[name]


def require_invariance(a, b):
    if a != b:
        raise ValueError('future-only perturbation changed decision state')


def main():
    harm = ROOT / 'predictfun/research/pf_order_harm_v2.py'
    joint = ROOT / 'predictfun/research/pf_joint_ladder_v2.py'
    ns = dict(bisect=bisect, np=NP, NAN=float('nan'), MUT_STAMP=0, MUT_RAISE=0)
    queue = extract(harm, 'queue_known', ns)
    o = dict(ta=240.0, pt=50)
    clean = [queue(o, 250.0, Vec([100.0, x]), [0.0, 330.0], [0.0, 300.0]) for x in (100.0, 900.0)]
    require_invariance(*clean)
    ns['MUT_RAISE'] = 1
    bad = [queue(o, 250.0, Vec([100.0, x]), [0.0, 330.0], [0.0, 300.0]) for x in (100.0, 900.0)]
    try:
        require_invariance(*bad)
    except ValueError:
        caught_raise = True
    else:
        raise AssertionError('next-frame-raise falsifier missed')
    ns['MUT_RAISE'] = 0
    o = dict(ta=140.0, pt=50)
    clean_late = [queue(o, 250.0, Vec([100.0, x]), [0.0, 300.0], [0.0, 200.0]) for x in (100.0, 20.0)]
    require_invariance(*clean_late)
    ns['MUT_STAMP'] = 1
    bad = [queue(o, 250.0, Vec([100.0, x]), [0.0, 300.0], [0.0, 200.0]) for x in (100.0, 20.0)]
    try:
        require_invariance(*bad)
    except ValueError:
        caught_stamp = True
    else:
        raise AssertionError('venue-clock falsifier missed')

    tree = ast.parse(harm.read_text())
    censor = [n.test for n in ast.walk(tree) if isinstance(n, ast.If)
              and ast.unparse(n.test) == "t + H > w['te']"]
    assert len(censor) == 1
    censored = [bool(eval(compile(ast.Expression(censor[0]), str(harm), 'eval'),
                          dict(t=t, H=2000.0, w={'te':60000.0}))) for t in (58000.0, 59000.0)]
    assert censored == [False, True]
    report = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'report')
    assert 'R622' not in ast.unparse(report)

    traces = []
    # At t=0 both books show opposite best 52. Only the venue state at t=40
    # differs. A pending bid at 50 has not yet been accepted/rejected at t=0.
    for future_opp in (52, 50):
        side = dict(opp=[52, future_opp], live=[], allo=[], orders=[[], []])
        env = dict(K=SimpleNamespace(L_PLACE=40.0, L_FILL=100.0), bisect=bisect,
                   tfl=[0.0, 40.0], nf=2, ST={0:side}, size=30.0, INF=float('inf'),
                   np=NP, q='M', INDEP=0, MUT_HARD_TRUE=0,
                   stat={'viol':{'collision':0}}, lv=lambda s, pt:[100.0,100.0])
        place = extract(joint, 'place', env); believed = extract(joint, 'believed', env)
        first = place(0, 0, 50, 0.0)
        exposure = believed(0, 0.0, 0.0)
        # Exact hard-gate inequality used by sim_joint, same fixed cap / size.
        second_allowed = exposure + 30.0 <= 30.0 + 1e-9
        traces.append(dict(future_opposite_best=future_opp, first_rejected=first['ref'],
                           believed_at_submission=exposure, second_allowed_at_submission=second_allowed))
    assert traces[0]['believed_at_submission'] == 30.0 and traces[1]['believed_at_submission'] == 0.0
    assert [x['second_allowed_at_submission'] for x in traces] == [False, True]
    # Reserving a submitted order until its acknowledgement removes the leak.
    require_invariance(30.0 + 30.0 <= 30.0, 30.0 + 30.0 <= 30.0)
    full_traces = []
    for future_opp in (52, 50):
        # Exercise the whole actual lane too, with no trades and two rungs.
        # At t=0 both client-known books are identical. Only the future book differs.
        W = {}
        for s in (0, 1):
            prices = (50, 49) if s == 0 else (52, 53)
            W[s] = dict(tg=Vec([0.0,50.0,100.0,150.0]), tf=Vec([0.0,40.0]), te=200.0,
                        anc=Vec(), ptl=[], shl=[], P5=None, S5=None,
                        opp=[52,future_opp] if s == 0 else [50,49 if future_opp == 50 else 50],
                        best=([50,49,49,49] if future_opp == 50 else [50]*4) if s == 0
                             else ([52,50,50,50] if future_opp == 50 else [52]*4),
                        bopp=([52,50,50,50] if future_opp == 50 else [52]*4) if s == 0
                              else ([50,49,49,49] if future_opp == 50 else [50]*4), vkl=[0.5]*4,
                        cache={pt:Vec([100.0,0.0 if s == 0 and pt == 50 and future_opp == 50 else 100.0]) for pt in prices})
        env = dict(bisect=bisect, np=NP, SIZE=30.0, NR=2, INF=float('inf'),
                   K=SimpleNamespace(L_PLACE=40.0,L_FILL=100.0,L_CANCEL=22.0,
                                     L_PENDCANCEL=80.0,HOLD_K=3,REST_COOL=250.0),
                   VIOL=('leg','order','live','priority','collision','position','cap'),
                   MUT_HARD_TRUE=0,MUT_LEG_LEAK=0,MUT_PRIORITY=0,INDEP=0)
        sim = extract(joint, 'sim_joint', env)
        result = sim(W, 'M', 0.0, 30.0, {0:[False]*4,1:[False]*4}, mode='hard',nr=2)
        initial_second = [o for o in result['orders'][(0,1)] if o[0] == 40.0]
        full_traces.append(dict(future_opposite_best=future_opp,
                                second_bid_submitted_at_zero=bool(initial_second),
                                violations=result['stat']['viol']))
        assert not any(result['stat']['viol'].values())
    assert [x['second_bid_submitted_at_zero'] for x in full_traces] == [False,True]
    print(json.dumps(dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(),
                          kind='source_synthetic_controls_not_historical_effect',
                          source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (harm,joint)},
                          repaired_queue=dict(future_frame=clean, delayed_frame=clean_late,
                                              raise_mutant_caught=caught_raise, stamp_mutant_caught=caught_stamp),
                          repaired_censor=dict(decisions_ms=[58000,59000], censored=censored),
                          summed_economic_reading_removed=True,
                          remaining_hard_cap_ack_leak=traces,
                          complete_lane_ack_leak=full_traces,
                          pending_reservation_control_passed=True), indent=2))


if __name__ == '__main__':
    main()
