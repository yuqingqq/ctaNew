"""MUTATION CHECK of pf_ladder_pressure_residual (PRESSURE-RESIDUAL-TIMING-AUDIT-v1; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with
pf_ladder_pressure_residual_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_pressure_residual.py'
SELFTEST = 'pf_ladder_pressure_residual_selftest'
FAIL = 'RESIDUAL SELFTEST FAILED'
MUTANTS = [
    ('a same-timestamp request counted pending', "if o['req'] is not None and float(o['req']) < t_f:", "if o['req'] is not None and float(o['req']) <= t_f:"),
    ('removal not strict', "        if t_f < float(o['remove']):\n", "        if t_f <= float(o['remove']):\n"),
    ('the pending reason ignored', "return (f'pending_{reason}', d) if reason in", "return ('pending_structural', d) if reason in"),
    ('the feasible window without the cancel latency', "if T1 is None or t_f < T1 + CANCEL_MS:", "if T1 is None or t_f < T1:"),
    ('a feasible tick needs no latency', "feas = [(t, a, r) for t, a, r in live if T1 <= t and t + CANCEL_MS <= t_f]", "feas = [(t, a, r) for t, a, r in live if T1 <= t and t <= t_f]"),
    ('a fallback callback counted evaluable', "e = r != 'fallback' and x[1] == 'ok'", "e = x[1] == 'ok'"),
    ('a non-ok status counted evaluable', "e = r != 'fallback' and x[1] == 'ok'", "e = r != 'fallback'"),
    ('a timely ON absorbed as unflagged', "    if on:\n        return 'event_or_action_mismatch', dict(d, why='pressure ON at a feasible evaluable tick but kept')\n", ""),
    ('nonevaluable never assigned', "    if not ev:\n        return 'nonevaluable', d\n", ""),
    ('the signal of the other side', "s = int(o['side']); live = [(t, a, r) for t, k, a, r in cmds_g if k == 'live']", "s = 1 - int(o['side']); live = [(t, a, r) for t, k, a, r in cmds_g if k == 'live']"),
    ('the eligible cutoff dropped', "el = t <= end - 1000. and mm[6] is not None", "el = mm[6] is not None"),
    ('the exactly-once check dropped', "        C.require(sorted(f.fill.astype(int)) == list(range(n)), f'{day} {m}: the fills are not each assigned exactly once')\n", ""),
    ('a reconcile tolerance of everything', "bad += [(day, m, k) for k, (x, y) in chk.items() if abs(float(x) - float(y)) > 1e-6 * max(1., abs(float(y)))]", "bad += []"),
    ('mismatches do not fail the audit', "answer='AUDIT_FAILED' if mism else", "answer='AUDIT_FAILED' if False else"),
    ('thin days ignored', "'THIN' if thin else", "'THIN' if False else"),
    ('the dominance bar at a quarter', "'TIMING_LIMITED_DOMINANT' if all(x is not None and x > .5 for x in ft)", "'TIMING_LIMITED_DOMINANT' if all(x is not None and x > .25 for x in ft)"),
    ('positives in the negative fractions', "el['neg'] = el.mo1s.astype(float).clip(upper=0.)", "el['neg'] = el.mo1s.astype(float)"),
    ('audit without the freeze check', "        C.require(frozen(), 'the residual audit / selftest / declaration / manifest / launcher are not the committed ones')\n", "        pass\n"),
    ('audit without the write-once check', "C.require(not (od / 'audit.json').exists(), f'{scope}: the audit exists (write-once)');", ""),
]
