"""Fix check for review 4 (R7 throttle reservation, R8 view-checker refusal), 2026-11-05.

Synthetic inputs only (source functions executed on hand-built frames / legs; no market data read).
Cases are review 4's own (check_v4.py), with the expectations of a FIXED engine:
  - every earlier control of check_v4 still passes on the fixed engines;
  - R7: with XV, the hidden-fill branch and the no-fill branch send identical requests until the fill is known (343 + 100 ms);
    the physical regression view still diverges (positive control);
  - falsifier: the same R7 case on the pre-fix copies (14:53Z, before the user's fix) MUST diverge, placing at 353 ms;
  - R8: pf_view_check refuses (exit 2) a missing / empty / mismatched / incomplete known run, FAILS (exit 1) a known run with an
    early re-place and a physical run that never fires, PASSES (exit 0) only a firing physical + clean known pair;
    falsifier: the pre-fix checker passes the missing run (exit 0), as review 4 reported.
Usage: python3 predictfun/reviews/simulation_20261105/check_fixes_v4.py [pre-fix dir]
"""
import ast, bisect, collections, heapq, itertools, math, json, hashlib, sys, glob, os, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT = Path('/home/yuqing/ctaNew/predictfun/research')
PRE = Path(sys.argv[1] if len(sys.argv) > 1 else '/tmp/predictfun-repair-loop-20261105')


def load(path):
    src = path.read_text(); tree = ast.parse(src)
    ns = dict(np=np, pd=pd, bisect=bisect, collections=collections, heapq=heapq, itertools=itertools,
              math=math, SIZE=5., V2=True, SPLIT=True, L_O=22., L_FILL=100., L_PL=40., L_CA=22., L_ANS=44.,
              LEGACY_INTENT=False, QACT=False, G2V={}, PRED={}, PRED250={}, PREDVB={}, PREDVA={},
              FILLPRIO='price', OWNCROSS='on', LEGORDER='price', REJRESET='on', POLICYVIEW='known', CK=SimpleNamespace(L_PENDCANCEL=95.),
              LINES={'zero': [0.] * 4, 'inf': [float('inf')] * 4, 'test': [0.] * 4})
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]), str(path), 'exec'), ns)
    ns['reg'] = pd.DataFrame([dict(mid=1, ws=0., outcome='Up')]).set_index('mid')
    return ns, hashlib.sha256(src.encode()).hexdigest()


def fr(t, b=50, a=51, bs=100., az=100.):
    f = dict(t_ms=float(t))
    for s in 'ba':
        for i in range(1, 6):
            f[f'{s}{i}_p'] = -1; f[f'{s}{i}_s'] = 0.
    f.update(b1_p=b, b1_s=bs, a1_p=a, a1_s=az)
    return f


def leg(t, side, pt, sh):
    return dict(t_ms=float(t), side=side, ptick=pt, shares=float(sh), how=1, exec_s=0)


def run(ns, trades, F=None, kf=None, rules=None, size=10., policy='JOIN'):
    ns['LT_BY'] = {1: pd.DataFrame(trades)} if trades else {}
    return ns['run_window'](1, F or [fr(-1)], kf or [0.], rules or {'XNA', 'G5', 'S1'}, policy, 'mid', size)


def prior_controls(ns, file):
    """check_v4's earlier controls, verbatim expectations (all fixed before review 4)."""
    r = {}
    F = [fr(-1, 49, 52, 0., 0.), fr(10, 50, 52, 0., 0.)]
    fixed = run(ns, [leg(60, 0, 48, 5)], F, [0., 10.], size=5.)
    assert fixed['fills'][0][3] == 50
    ns['FILLPRIO'] = 'old'; old = run(ns, [leg(60, 0, 48, 5)], F, [0., 10.], size=5.)
    assert old['fills'][0][3] == 49; ns['FILLPRIO'] = 'price'
    r['price_priority'] = {'fixed_pt': 50, 'regression_pt': 49}
    sweep = run(ns, [leg(60, 0, 50, 100), leg(60, 0, 49, 5)])
    bid_q = sum(x[4] for x in sweep['fills'] if x[2] == 0)
    split_q = sum(x[4] for x in run(ns, [leg(59, 0, 50, 100), leg(60, 0, 49, 5)])['fills'] if x[2] == 0)
    agg_q = sum(x[4] for x in run(ns, [leg(60, 0, 50, 105)])['fills'] if x[2] == 0)
    ask_q = sum(x[4] for x in run(ns, [leg(60, 1, 51, 100), leg(60, 1, 52, 5)])['fills'] if x[2] == 1)
    assert bid_q == split_q == agg_q == ask_q == 5, (bid_q, split_q, agg_q, ask_q)
    ns['LEGORDER'] = 'old'; bad_sweep = run(ns, [leg(60, 0, 50, 100), leg(60, 0, 49, 5)]); ns['LEGORDER'] = 'price'
    assert sum(x[4] for x in bad_sweep['fills'] if x[2] == 0) == 10
    r['atomic_sweep'] = {'bid': bid_q, 'controls': [split_q, agg_q, ask_q], 'regression_bid': 10}
    F = [fr(-1, 50, 52, 0., 0.), fr(30, 49, 50, 0., 0.), fr(60, 50, 52, 0., 0.)]
    rejected = run(ns, [leg(160, 0, 49, 5)], F, [0., 100., 100.], size=5.)
    Fctrl = [fr(-1, 50, 52, 0., 0.), fr(30, 49, 51, 0., 0.), fr(60, 50, 52, 0., 0.)]
    accepted = run(ns, [leg(160, 0, 49, 5)], Fctrl, [0., 100., 100.], size=5.)
    assert rejected['live_ctr'].get('postonly_refused', 0) == 1
    assert rejected['n_fills'] == accepted['n_fills'] == 1
    bid_places = [q for q in rejected['requests'] if q[1] == 0 and q[2] == 'place']
    assert len(bid_places) >= 2 and bid_places[0][0] == 0. and bid_places[1][0] == 100., bid_places
    ns['REJRESET'] = 'off'; bad_retry = run(ns, [leg(160, 0, 49, 5)], F, [0., 100., 100.], size=5.); ns['REJRESET'] = 'on'
    assert bad_retry['n_fills'] == 0
    r['postonly_retry'] = {'refusals': 1, 'fills_after_recovery': 1, 'off_control_fills': 0}
    ns['PREDVB'] = {1: (np.array([0., 10.]), np.array([-1., -1.]))}
    ns['PREDVA'] = {1: (np.array([0., 10.]), np.array([-1., -1.]))}
    assert run(ns, [leg(160, 0, 48, 5)], [fr(-1, 50, 52, 0., 0.)], [0.], {'XNA', 'G5', 'S1', 'V0', 'GS'}, 5.)['n_fills'] == 0
    ns['PREDVB'] = {}; ns['PREDVA'] = {}
    assert run(ns, [leg(60, 0, 48, 5)], [fr(-1, 50, 52, 0., 0.)], [0.], {'XNA', 'G5', 'S1', 'V0', 'GS'}, 5.)['n_fills'] == 0
    r['GS'] = {'missing_fills': 0, 'stale_fills': 0}
    ns['PREDVB'] = {1: (np.array([0., 10.]), np.array([1., 1.]))}
    ns['PREDVA'] = {1: (np.array([0., 10.]), np.array([1., 1.]))}
    if file == 'pf_step4_ladder.py':
        F = [fr(-1, 49, 51, 0., 0.), fr(10, 50, 52, 0., 0.)]
        crossed = run(ns, [], F, [0., 10.], {'XNA', 'G5', 'S1', 'IMtest'}, 5.)
        assert crossed['live_ctr'].get('postonly_refused_own', 0) > 0 and crossed['live_ctr'].get('own_cross', 0) == 0
        ns['OWNCROSS'] = 'off'; bad = run(ns, [], F, [0., 10.], {'XNA', 'G5', 'S1', 'IMtest'}, 5.); ns['OWNCROSS'] = 'on'
        assert bad['live_ctr'].get('own_cross', 0) > 0
        r['own_cross'] = {'fixed': 0, 'off_control': bad['live_ctr']['own_cross']}
        F = [fr(-100, 50, 52, 10., 10.), fr(45, 49, 52, 10., 10.), fr(85, 49, 52, 10., 10.)]
        kf = [0., 60., 100.]
        ns['PREDVB'] = {1: (np.array([0., 60., 100., 150.]), np.array([1., 1., 1., 1.]))}
        ns['PREDVA'] = {1: (np.array([0., 60., 100., 150.]), np.array([-1., -1., -1., -1.]))}
        rules = {'XNA', 'G5', 'S1', 'V0', 'GS', 'LD2'}
        early = lambda res: [q for q in res['requests'] if 90. < q[0] < 190. and q[2] == 'place']
        filled = run(ns, [leg(90, 0, 49, 5)], F, kf, rules, 5.); no_fill = run(ns, [], F, kf, rules, 5.)
        assert filled['fills'][0][1:5] == (0.09, 0, 49, 5.), filled['fills']
        assert not early(filled) and not early(no_fill)
        ns['POLICYVIEW'] = 'physical'; physical = run(ns, [leg(90, 0, 49, 5)], F, kf, rules, 5.); ns['POLICYVIEW'] = 'known'
        assert early(physical)
        ns['L_FILL'] = 1000.; assert not early(run(ns, [leg(90, 0, 49, 5)], F, kf, rules, 5.)); ns['L_FILL'] = 100.
        ns['KB'] = np.arange(-2000., 4000., 100.); ns['MB'] = np.full(60, 100.); ns['BC'] = np.zeros(60, dtype=np.int8)
        ns['BCV'] = {}; ns['_FLIP_CACHE'] = {}
        ns['PREDVB'] = {1: (np.array([0., 60., 100., 150.]), np.full(4, .01))}
        for f in F:
            f.update(b2_p=f['b1_p'] - 1, b2_s=10., a2_p=f['a1_p'] + 1, a2_s=10.)
        full = {'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'V0', 'GS', 'LD2', 'XNA', 'XMIN1', 'XV', 'S60'}
        ff = run(ns, [leg(90, 0, 49, 15)], F, kf, full, 'AV:5:25:0.06', 'MICRO'); fc = run(ns, [], F, kf, full, 'AV:5:25:0.06', 'MICRO')
        assert not early(ff) and not early(fc)
        r['R6_rung_leak'] = {'known_early': 0, 'physical_early': len(early(physical)), 'full_stack_early': 0}
    try:
        run(ns, [], rules={'G5', 'S1'})
    except SystemExit:
        r['known_bad_input_refused'] = True
    else:
        raise AssertionError('unsupported split/amend accepted')
    return r


def throttle_case(ns, file):
    """Review 4's R7 case: a cancel-pending order fully fills at 343 ms (known at 443); frames after the fill are public and
    identical in both branches. Returns the requests before 443 ms for (hit, quiet) with XV and without XV."""
    ns['KB'] = np.arange(-2000., 4000., 100.); ns['MB'] = np.full(60, 100.); ns['BC'] = np.zeros(60, dtype=np.int8)
    ns['BCV'] = {}; ns['_FLIP_CACHE'] = {}
    ns['PREDVB'] = {1: (np.arange(0., 1001., 50.), np.full(21, .01))}
    ns['PREDVA'] = {1: (np.arange(0., 1001., 50.), np.full(21, -1.))}
    tf_fill = 343.; tk = np.r_[np.arange(18) * 19., 348., 353., 358., 363.]
    Fb = [fr(t - 15., 50 + (i % 2), 80, 10., 10.) for i, t in enumerate(tk)]
    for f, src in zip(Fb[-4:], np.linspace(309., 342., 4)):
        f['t_ms'] = src
    for f in Fb:
        f.update(b2_p=f['b1_p'] - 1, b2_s=10., a2_p=81, a2_s=10.)
    rr = {'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'V0', 'GS', 'XNA', 'XMIN1', 'XV', 'S60'}
    before = lambda z: [q for q in z['requests'] if q[0] < tf_fill + 100.]
    hit = run(ns, [leg(tf_fill, 0, 49, 5)], Fb, list(tk), rr, 'AV:5:25:0.06', 'MICRO')
    quiet = run(ns, [], Fb, list(tk), rr, 'AV:5:25:0.06', 'MICRO')
    assert hit['fills'] and hit['fills'][0][1] == tf_fill / 1000. and hit['fills'][0][-1 if file == 'pf_step4_live.py' else 12], hit['fills']
    uh = run(ns, [leg(tf_fill, 0, 49, 5)], Fb, list(tk), rr - {'XV'}, 'AV:5:25:0.06', 'MICRO')
    uq = run(ns, [], Fb, list(tk), rr - {'XV'}, 'AV:5:25:0.06', 'MICRO')
    return before(hit), before(quiet), before(uh), before(uq)


out = {}
for file in ('pf_step4_live.py', 'pf_step4_ladder.py'):
    ns, sha = load(ROOT / file); r = {'sha256': sha}
    r['prior_controls'] = prior_controls(ns, file)
    ns['POLICYVIEW'] = 'known'
    hq, qq, uh, uq = throttle_case(ns, file)
    assert hq == qq, ('R7 still leaks: requests diverge before the fill is known', hq, qq)
    assert not any(q[0] == 353. and q[2] == 'place' for q in hq), hq
    assert uh == uq, (uh, uq)
    ns['POLICYVIEW'] = 'physical'
    ph, pq, _, _ = throttle_case(ns, file)
    ns['POLICYVIEW'] = 'known'
    assert ph != pq, ('physical regression control did not diverge', ph, pq)
    nsb, shab = load(PRE / (file + '.before'))
    bh, bq, _, _ = throttle_case(nsb, file)
    assert bh != bq and any(q[0] == 353. and q[2] == 'place' for q in bh), ('falsifier: pre-fix engine did not reproduce R7', bh, bq)
    r['R7'] = {'known_hit_eq_quiet_before_443ms': True, 'n_requests_before_443ms': len(hq), 'place_at_353': False,
               'unthrottled_equal': True, 'physical_control_diverges': True,
               'prefix_falsifier': {'sha256': shab, 'diverges': True, 'place_at_353': True,
                                    'first_divergence': next((a, b) for a, b in itertools.zip_longest(bh, bq) if a != b)}}
    out[file] = r


def view_fixture(td, run_name, fill_t, place_t, arm='arm', summary=True, empty=False, drop=None):
    base = Path(td) / 'step4' / run_name / 'synthetic'
    for part in ('fills', 'requests', 'win'):
        (base / part).mkdir(parents=True)
    if not empty:
        if drop != 'fills':
            pd.DataFrame([dict(mid=1, lane=0, zombie=False, t=fill_t / 1000.)]).to_parquet(base / 'fills' / f'{arm}.parquet', index=False)
        pd.DataFrame([dict(mid=1, lane=0, side=0, kind='place', t_ms=place_t)]).to_parquet(base / 'requests' / f'{arm}.parquet', index=False)
        pd.DataFrame([dict(mid=1, ws=0.)]).to_parquet(base / 'win' / f'{arm}.parquet', index=False)
    if summary:
        arms = {} if empty else {arm: dict(fills=1, windows=1, live=dict(requests=dict(place=1, cancel=0)))}
        (base / 'summary.json').write_text(json.dumps(dict(day='synthetic', arms=arms)))


def vc_rc(path, td, phys, known):
    src = path.read_text(); tree = ast.parse(src)
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.If))]

    def end(rc):
        raise SystemExit(rc)
    scope = dict(__name__='__main__', np=np, pd=pd, glob=glob, os=os, json=json, io=SimpleNamespace(OUT=td),
                 sys=SimpleNamespace(argv=['pf_view_check.py', phys, known, 'synthetic'], exit=end, stderr=sys.stderr))
    try:
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), scope)
    except SystemExit as e:
        return e.code
    raise AssertionError('checker main did not exit')


with tempfile.TemporaryDirectory(prefix='pf_view_fix4_') as td:
    view_fixture(td, 'physical', 50., 60.)               # early re-place 10 ms after a live fill: fires
    view_fixture(td, 'known_bad', 50., 60.)
    view_fixture(td, 'known_good', 50., 300.)            # re-place 250 ms after the fill: not early
    view_fixture(td, 'physical_silent', 50., 300.)       # instrument never fires
    view_fixture(td, 'known_empty', 50., 300., empty=True)
    view_fixture(td, 'known_other_arm', 50., 300., arm='other')
    view_fixture(td, 'known_no_summary', 50., 300., summary=False)
    view_fixture(td, 'known_no_fills', 50., 300., drop='fills')
    VC = ROOT / 'pf_view_check.py'
    rc = {k: vc_rc(VC, td, 'physical', k) for k in ('missing_known', 'known_empty', 'known_other_arm', 'known_no_summary', 'known_no_fills',
                                                    'known_bad', 'known_good')}
    rc['physical_silent'] = vc_rc(VC, td, 'physical_silent', 'known_good')
    rc['physical_missing'] = vc_rc(VC, td, 'missing_physical', 'known_good')
    want = dict(missing_known=2, known_empty=2, known_other_arm=2, known_no_summary=2, known_no_fills=2, known_bad=1, known_good=0,
                physical_silent=1, physical_missing=2)
    assert rc == want, ('R8 view checker', rc, want)
    pre_missing = vc_rc(PRE / 'pf_view_check.py.before', td, 'physical', 'missing_known')
    assert pre_missing == 0, ('falsifier: pre-fix checker did not reproduce R8', pre_missing)
    out['pf_view_check.py'] = {'sha256': hashlib.sha256(VC.read_bytes()).hexdigest(), 'exit_codes': rc,
                               'prefix_falsifier_missing_known_exit': pre_missing}

out['verdict'] = 'PASS'                                  # reached only if every assertion above held
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2, default=str) + '\n')
print(json.dumps(out, indent=2, default=str))
