"""Bounded timing-audit review: synthetic instruments and existing derived artifacts.

Does not reload tape, rerun source-leg linking, or claim to reproduce unpersisted
real-data S1/S2 intervals. Those reproduction limits are recorded explicitly.
"""
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

TARGET = 'd1de75049fdf5a2477e279ca108114b5aef09b27'
ROOT = Path('/home/yuqing/ctaNew')
SNAP = Path(sys.argv[1]).resolve()
OUT = Path(__file__).resolve().parent
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SNAP, text=True).strip() == TARGET
sys.path.insert(0, str(SNAP/'predictfun/research'))
import pf_keep_pull_timing_audit as TA
KP, A = TA.KP, TA.A


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def eq(x, y):
    assert np.allclose(x, y, atol=1e-7, rtol=1e-12), (x, y)


def main():
    saved = SNAP/'predictfun/reviews/keep_pull_timing_audit_20261118'
    J = json.loads((saved/'timing_audit.json').read_text()); R = J['R']
    evidence = dict(target_commit=TARGET, asof=KP.C.stamp(), result_asof=J['asof'], script_sha256=sha(__file__),
                    saved_artifacts={}, per_day={}, controls={}, limitations=[
                        'No full tape/source-leg linking rerun.',
                        'Per-fill linked intervals and source identities not persisted in the result artifact.',
                        'Real S1/S2 extrema checked for aggregation consistency, not independently rebuilt.',
                        'Scenario intervals are not calibrated confidence or venue-time bounds; M50 quantities and eligibility fixed.'])
    for name in ('timing_audit.json', 'timing_audit.txt', 'run.log'):
        actual = ROOT/'predictfun/reviews/keep_pull_timing_audit_20261118'/name if name == 'run.log' else ROOT/'data/predict_fun/research/keep_pull_timing_audit_20261118'/name
        assert sha(saved/name) == sha(actual)
        evidence['saved_artifacts'][name] = sha(saved/name)
    assert sha(SNAP/'predictfun/research/pf_keep_pull_timing_audit.py') == J['audit_source_sha256']
    for name, digest in J['provenance']['source'].items():
        assert sha(SNAP/'predictfun/research'/name) == digest
    TA.selftest()
    # Hand-computed boundary control, with both a positive and a known-bad value.
    # One losing fill: score pulls at 137 ms; matched null inclusion is 1/2.
    CW = np.zeros((1, 100)); CW[0, 2:] = .5
    points = np.array([57., 137.-1e-6, 137., 4999., 5000., np.nextafter(5000., np.inf)])
    F = dict(val=np.full(len(points), -10.), pulled=np.ones(len(points), bool), r=np.full(len(points), 137.),
             s=np.zeros(len(points), int), endj=np.full(len(points), 99), lend=np.full(len(points), np.inf))
    got = TA.contrib(points, F, CW); eq(got, [0., 0., 5., 5., 5., 0.])
    lo = np.array([136., 137., 4999., 5000., 5001., 100.]); hi = np.array([138., 138., 5001., 5000., 5002., 136.])
    mn, mx = TA.bounds(lo, hi, F, CW); eq(mn, [0., 5., 0., 5., 0., 0.]); eq(mx, [5., 5., 5., 5., 0., 0.])
    try:
        eq(got, [0., 0., 5., 5., 0., 0.])
    except AssertionError:
        pass
    else:
        raise AssertionError('known-bad END exclusion accepted')
    evidence['controls'] = dict(pinned_selftest_passed=True, independent_boundary_and_bounds=True, known_bad_END_refused=True)

    data = ROOT/'data/predict_fun/research/keep_pull_20261117'
    totals = dict(counted=0, eligible=0, fills=0, pulled_order_fills=0, collapsed_E=0., score=0., kept=0., gross=0.)
    for day in KP.DAYS:
        receipt = data/day/'receipt.json'
        assert sha(receipt) == J['source_receipts'][day]
        r = json.loads(receipt.read_text())
        for name, digest in r['outputs'].items():
            assert sha(data/day/name) == digest
        O = pd.read_parquet(data/day/'orders.parquet'); Fl = pd.read_parquet(data/day/'fills.parquet').assign(day=day)
        E = O[O.status == 'eligible'].reset_index(drop=True)
        assert not E.oid.duplicated().any() and Fl.oid.isin(E.oid).all()
        assert O.status.value_counts().to_dict() == r['status']
        cell = E.groupby(['m', 'side', 'phase'], sort=True).ngroup().values
        T = KP.score_steps(E[[f'f{j:02d}' for j in range(99)]].values, E.side.values, E.p.values)['A+0']
        endj = A.endj_of(E.t_full.values, E.g_struct.values); act, cw = TA.null_weights(T, cell, endj)
        ri, t, val, src = A.fills_of(E, Fl, 50.)
        F = dict(val=val, pulled=act[ri], r=KP.landing(np.clip(T[ri], 1, 98)), s=cell[ri], endj=endj[ri],
                 lend=np.where(np.isnan(E.land_struct.values[ri]), np.inf, E.land_struct.values[ri]))
        mn, mx = TA.bounds(t, t, F, cw)
        pf = A.per_fill(T, cell, endj, (ri, t, val, src))
        eq(mn, mx); eq(mn, pf['score']-pf['null'])
        eq(mn.sum(), R['S1']['by_day'][day]['E_sim']); eq(mn.sum(), R['S2']['by_day'][day]['E_sim'])
        pl = F['pulled']; removed = pl & (t >= F['r']); kept = pl & ~removed
        row = dict(counted=len(O), eligible=len(E), fills=len(ri), pulled_order_fills=int(pl.sum()),
                   collapsed_E=float(mn.sum()), score=float(-val[removed].sum()), kept=float(-val[kept].sum()), gross=float(np.abs(val[pl]).sum()))
        evidence['per_day'][day] = dict(**row, statuses=r['status'])
        for key, value in row.items():
            totals[key] += value
        # Small pre-existing summaries only. This checks present values, not an
        # original-run hash guarantee: the producer receipt omits these hashes.
        paths = {name: Path(TA.I.IO)/'step1'/day/name for name in ('booktime.json', 'timing.json')}
        book, timing = [json.loads(paths[name].read_text()) for name in ('booktime.json', 'timing.json')]
        eq([book['E2']['p90'], timing['offset_ms_pct']['p01'], timing['offset_ms_pct']['p99']],
           [R['env'][day]['h'], R['env'][day]['q01'], R['env'][day]['q99']])
        evidence['per_day'][day]['current_envelope_input_hashes'] = {name: sha(p) for name, p in paths.items()}
        print(f'{day}: saved-table identity, collapsed fill contributions and envelope summaries pass', flush=True)
        del O, E, Fl
        gc.collect()
    eq(totals['collapsed_E'], R['collapsed_E']); eq(totals['score'], R['score']); eq(totals['kept'], R['sim_order']['kept_c'])
    eq(totals['fills'], R['fills']); eq(sum(R['class_fills'].values()), R['fills'])
    for scenario in ('S1', 'S2'):
        s = R[scenario]
        eq(sum(v['fills'] for v in s['order'].values()), totals['pulled_order_fills'])
        eq(sum(v['removed_c'] for v in s['order'].values()), totals['score']+totals['kept'])
        eq(sum(v['gross_c'] for v in s['order'].values()), totals['gross'])
        eq(sum(v['ambiguous_c'] for v in s['by_class'].values()), s['order']['AMBIGUOUS']['removed_c'])
        for group in ('by_day', 'by_side', 'by_class'):
            for field in ('E_min', 'E_max'):
                eq(sum(v[field] for v in s[group].values()), s[field])
            assert all(v['E_min'] < 0 < v['E_max'] for v in s[group].values())
        expected = 'EXCESS-POSITIVE-IDENTIFIED' if s['E_min'] > 0 else 'EXCESS-NEGATIVE-IDENTIFIED' if s['E_max'] < 0 else 'SIGN-NOT-IDENTIFIED'
        assert J['readings']['T1 SIGN UNDER S1' if scenario == 'S1' else 'T2 SIGN UNDER S2'] == expected
    evidence.update(totals=totals, scenario_readings=J['readings'],
        headline_population_correction=dict(all_fills_on_pulled_orders=totals['pulled_order_fills'],
            all_fills_net_c=totals['score']+totals['kept'], replay_removed_net_c=totals['score'],
            S1_ambiguous_fill_fraction=R['S1']['order']['AMBIGUOUS']['fills']/totals['pulled_order_fills'],
            S1_ambiguous_gross_fraction=R['S1']['order']['AMBIGUOUS']['gross_c']/totals['gross']),
        raw_tape_read=False, real_interval_ledger_recomputed=False, untouched_validation=False)
    (OUT/'receipt_v1.json').write_text(json.dumps(KP.C.clean(evidence), indent=2)+'\n')
    print(json.dumps(dict(totals=totals, headline_population_correction=evidence['headline_population_correction']), indent=2))


if __name__ == '__main__':
    main()
