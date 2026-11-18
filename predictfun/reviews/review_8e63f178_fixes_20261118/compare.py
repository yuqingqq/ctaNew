"""Reproduction predicates of the review-8e63f178 fixes, declared before the reruns read any data (2026-11-18, session aea5bfb9).
What must NOT move: everything outside the fixed semantics.
 R1 timing case audit v3 vs v2 (the three legs): A0 fields, A1 class, A2 class and ids, A3 (reproduces, domain, u*), A4 (n, B, B_max, B_min,
    statuses, clock step), A5 and the critical offsets identical; the c_hi bound now typed by the answer's premise (CANCELED -> CONDITIONAL).
 R2 census v4 vs v3: per leg B, n, bound status, A0, A1 identical; N0 YES; the CANCELED-and-full order count identical (63).
 R3 known v7 vs v6: K0 identical (the 11-09 definition); K7 identical; K8 still consistent with no universal lag bound.
 R4 scenario replay v5 vs v4 on Oct 23: every scenario's E, score, null, KEEP, PULL-ALL, fills, shares, eligible identical (1e-6); v5 adds
    the C1 / C3 splits only.
What may move (the fixes): the per-domain gate (constituents kept, feasibility), K1-K6 / K8 readings on the evidence-only lifecycle, K5's
cancel statuses, K9 (new), N3 / N4 wording and the snapshot-vs-observed count. Usage: python3 compare.py <out.txt>"""
import json
import sys

import numpy as np
import pandas as pd

D = '/home/yuqing/ctaNew/data/predict_fun/research/'


def tca():
    a, b = (json.load(open(D + f'timing_case_audit_{v}_20261118/audit.json')) for v in ('v2', 'v3'))
    rows = []
    for x, y in zip(a['legs'], b['legs']):
        same = {k: x[k] == y[k] for k in ('leg', 'critical_ms', 'S3_lo', 'c_hi', 'A0', 'A0_fields', 'A1', 'A2', 'A2_binding_ids', 'A3_reproduces', 'A3_domain', 'A3_u',
                                          'A4', 'A5')}
        chi = [z['type'] for z in y['bounds'] if z['name'] == 'match <= cancel answer receipt (c_hi)']
        rows.append(dict(leg=x['leg'], all_same=all(same.values()), differ=[k for k, v in same.items() if not v], c_hi_type=chi, A1=y['A1']))
    return dict(R1=all(r['all_same'] for r in rows) and all(r['c_hi_type'] == (['CONDITIONAL'] if r['A1'] == 'CANCELED' else r['c_hi_type']) for r in rows), legs=rows)


def census():
    a, b = (pd.read_parquet(D + f'known_clock_census_{v}_20261118/legs.parquet') for v in ('v3', 'v4'))
    k = ['run', 'cid', 'trade_id']; m = a.merge(b, on=k, suffixes=('_a', '_b'))
    same = {c: bool(np.array_equal(m[f'{c}_a'].astype(str).values, m[f'{c}_b'].astype(str).values)) for c in ('B', 'n', 'bound_status', 'A0', 'A1')}
    j = json.load(open(D + 'known_clock_census_v4_20261118/census.json')); x3 = json.load(open(D + 'known_clock_census_v3_20261118/census.json'))
    n3 = j['R']['N3_class_x_fill']['CANCELED|full'] == x3['R']['N3_class_x_fill']['CANCELED|full']
    return dict(R2=len(m) == len(a) == len(b) and all(same.values()) and j['readings']['N0 REPRODUCES-THE-CASE-AUDIT'] == 'YES' and n3, same=same, canceled_full_same=n3,
                readings=j['readings'], first_notice_after_answer=j['R'].get('N3_first_notice_after_the_answer'))


def known():
    a, b = (json.load(open(D + f'keep_pull_exec_windows_known_{v}_20261118/known.json')) for v in ('v6', 'v7'))
    k8 = b['R']['K8']
    ok = a['R']['K0'] == b['R']['K0'] and a['R']['K7'] == b['R']['K7'] and bool(k8['consistent']) and k8['lag_bound_ms'] is None
    moved = {k: (a['readings'].get(k), v) for k, v in b['readings'].items() if a['readings'].get(k) != v}
    return dict(R3=ok, moved_readings=moved)


def scenarios():
    a, b = (json.load(open(D + f'keep_pull_scenarios_{v}_20261118/scenarios_20261023.json')) for v in ('v4', 'v5'))
    keys = ('E', 'score', 'null', 'KEEP', 'PULLALL', 'fills', 'shares', 'eligible')
    diff = [(nm, k, a['scenarios'][nm][k], b['scenarios'][nm][k]) for nm in a['scenarios'] for k in keys if abs(float(a['scenarios'][nm][k]) - float(b['scenarios'][nm][k])) > 1e-6]
    return dict(R4=set(a['scenarios']) == set(b['scenarios']) and not diff, differences=diff[:20], scenarios=len(b['scenarios']))


if __name__ == '__main__':
    out = dict(R1=tca(), R2=census(), R3=known(), R4=scenarios())
    out['ALL'] = all(out[k][k] for k in ('R1', 'R2', 'R3', 'R4'))
    txt = json.dumps(out, indent=1, default=str)
    open(sys.argv[1], 'w').write(txt + '\n'); print(txt)
