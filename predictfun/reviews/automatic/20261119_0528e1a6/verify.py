"""Bounded review of 0528e1a6: pinned code, synthetic probes and saved artifacts only."""
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-review-0528e1a6')
OUT = ROOT / 'predictfun/reviews/automatic/20261119_0528e1a6'
DATA = ROOT / 'data/predict_fun/research'
TARGET = '0528e1a614d582d92cf3efbcd26d6afdf7ee8686'
sys.path.insert(0, str(PIN / 'predictfun/research'))
import pf_pipeline_paired as PQ
import pf_pipeline_paired_review as RV


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


verified = {}


def checkhash(p, expected):
    got = sha(p)
    assert got == expected, str(p)
    name = 'pinned/' + str(p.relative_to(PIN)) if p.is_relative_to(PIN) else str(p.relative_to(ROOT))
    verified[name] = got


def save(name, obj):
    obj.update(target_commit=TARGET, asof=datetime.now(timezone.utc).isoformat(), script_sha256=sha(Path(__file__)))
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def synthetic():
    r = pd.DataFrame(dict(oid=[1, 1], j=[2, 3], side=[0, 0], p=[50., 50.], pf_mid=[50., 60.],
                          value_m50=[10., 5.], q_m50=[10., 5.], qm_m50=[10., -45.], pi0_V=[1., -1.], pi0_ok=[True, True]))
    v, _, first = PQ.label_retain(r)
    p = r.iloc[:1].assign(status='eligible')
    vp, _ = PQ.label_post(p, first, r)
    assert v['pair_v'][0] == v['pair_q'][0] == v['pair_qm'][0] == 5.
    assert vp['pair_v'][0] == vp['pair_q'][0] == vp['pair_qm'][0] == 5.
    assert PQ.reached([1, 2], [2, 2], [-1., -1.], {2}).tolist() == [False, True]
    assert PQ.reached([1, 2], [2, 2], [-1., -1.], {1, 2}).tolist() == [True, True]
    # New screen mixes a raw-row zero check with episode-balanced signs and amounts.
    oid = np.r_[np.repeat(np.arange(100), 2), np.arange(100, 200)]
    labels = np.r_[np.ones(200), np.full(100, -2.)]
    pred = np.full(300, -.5)
    screen_bug = RV.screen(oid, pred, labels, False)
    assert labels.sum() == 0 and RV.episode_mean(oid, labels) == -.5
    assert screen_bug == ('NOT_EVALUABLE', 200, 200)
    # Same episode means and prediction but equal representation: PASS.
    balanced_labels = np.r_[np.ones(100), np.full(100, -2.)]
    screen_control = RV.screen(np.arange(200), np.full(200, -.5), balanced_labels, False)
    assert screen_control == ('PASS', 200, 200)
    all_zero = RV.screen(np.arange(200), np.ones(200), np.zeros(200), True)
    assert all_zero[0] == 'NOT_EVALUABLE'
    # Time-check tolerance limitation: END+0.01ms is accepted by default np.isclose rtol.
    ex = pd.DataFrame(dict(kind=['retain'], j=[2], t=[0.], decision_ms=[100.], effective_ms=[137.], endpoint_ms=[5000.]))
    lab = pd.DataFrame(dict(kind=['retain'], j=[2], tstar=[np.nan]))
    clean = RV.v3_timing(ex, lab, (np.array([80.]), np.array([80.])))
    loose = RV.v3_timing(ex.assign(endpoint_ms=5000.01), lab, (np.array([80.]), np.array([80.])))
    bad = RV.v3_timing(ex.assign(endpoint_ms=5001.), lab, (np.array([80.]), np.array([80.])))
    assert clean == loose == (0, 0, 0) and bad == (1, 0, 0)
    # Null moments: exact enumeration as a positive control; a deliberately wrong variance is refused.
    cell = np.array([0, 0, 0, 1, 1, 2]); sel = np.array([True, False, False, True, False, True]); vals = np.array([1., 2., 6., 3., 5., 4.])
    totals = np.array([sum(z) for z in itertools.product(vals[:3], vals[3:5], vals[5:])])
    mean, var = RV.null_moments(cell, sel, vals)
    assert abs(mean - totals.mean()) < 1e-12 and abs(var - totals.var()) < 1e-12
    assert abs(2 * var - totals.var()) > 1.
    save('synthetic.json', dict(R0_counterexamples_fixed=True,
        episode_screen=dict(observed=screen_bug, equal_episode_control=screen_control,
                            raw_sum=0., episode_mean_label=-.5, episode_mean_prediction=-.5, expected='PASS', zero_control=all_zero[0]),
        timing=dict(clean=clean, endpoint_shift_0_01_ms=loose, endpoint_shift_1_ms=bad),
        exact_null_control=dict(mean=mean, variance=var, enumeration_outcomes=len(totals))))
    print('R0 fixes pass; episode-balanced screen counterexample and timing tolerance limitation reproduced.', flush=True)


def artifact_audit():
    p1 = DATA / 'pipeline_paired_20261119'; p2 = DATA / 'pipeline_paired_v2_20261119'
    ed = DATA / 'pipeline_paired_eval_v2_20261119'; rd = DATA / 'pipeline_paired_review_20261119'
    exp = DATA / 'pipeline_export_v2_20261119'; kp = DATA / 'keep_pull_20261117'
    j1, j2 = read(p1 / 'paired.json'), read(p2 / 'paired.json')
    assert j1['models'] == j2['models'] and len(j2['models']) == 44
    checkhash(PIN / 'predictfun/research/pf_pipeline_paired.py', j2['source_sha256'])
    for f, h in j2['models'].items():
        checkhash(p1 / 'models' / f, h); checkhash(p2 / 'models' / f, h)
    result = dict(models_identical=44, paired_asof=j2['asof'], labels={}, days={})
    for day in ('20261024', '20261025', '20261026', '20261027'):
        name = f'labels_{day}.parquet'
        checkhash(p1 / name, j1['outputs'][name]); checkhash(p2 / name, j2['outputs'][name])
        a, b = pd.read_parquet(p1 / name), pd.read_parquet(p2 / name)
        for c in set(a.columns) & set(b.columns) - {'pair_qm'}:
            assert a[c].equals(b[c]), (day, c)
        error = b.pair_v - (np.where(b.side == 0, 1., -1.) * (b.pf_mid - b.p) * b.pair_q + b.pair_qm)
        assert np.isfinite(error).all() and np.max(np.abs(error)) < 1e-6
        result['labels'][day] = dict(rows=len(b), max_identity_error=float(np.abs(error).max()), identical_except_pair_qm=True)
        del a, b; gc.collect()
    for day in ('20261026', '20261027'):
        ej, rj = read(ed / f'eval_{day}.json'), read(rd / f'review_{day}.json')
        old = read(DATA / 'pipeline_paired_eval_20261119' / f'eval_{day}.json')
        assert ej['values'] == old['values'] and ej['admissions'] == old['admissions']
        for key, path in (('paired', p2 / 'paired.json'), ('evaluation', ed / f'eval_{day}.json'), ('export_receipt', exp / day / 'receipt.json')):
            checkhash(path, rj['inputs'][key])
        for name, h in rj['inputs']['parts'].items():
            checkhash(p2 / name, h)
        for name, h in rj['outputs'].items():
            checkhash(rd / name, h)
            if name != f'ledger_{day}.parquet':
                checkhash(PIN / 'predictfun/reviews/pipeline_paired_review_20261119' / name, h)
        checkhash(PIN / 'predictfun/reviews/pipeline_paired_review_20261119' / f'review_{day}.json', sha(rd / f'review_{day}.json'))
        checkhash(PIN / 'predictfun/research/pf_pipeline_paired_review.py', rj['source_sha256'])
        checkhash(PIN / 'predictfun/research/pf_pipeline_paired_eval.py', ej['source_sha256'])
        ledger = pd.read_parquet(rd / f'ledger_{day}.parquet')
        A = ledger[ledger.policy == 'A'].sort_values('oid'); AA = ledger[ledger.policy == 'A-ALL'].sort_values('oid')
        assert np.array_equal(A.oid.values, AA.oid.values)
        draw = pd.read_csv(rd / f'review_null_draws_{day}.csv')
        # Recompute exact variance without calling the production null function.
        groups = AA.groupby(['m', 'side', 'phase'], sort=True)
        admitted_by_oid = A.set_index('oid').admitted
        means = variances = 0.
        for _, g in groups:
            selected = admitted_by_oid.reindex(g.oid.values).sum()
            n = len(g); means += float(selected * g.V.mean())
            if n > 1:
                variances += float(selected * (n - selected) / (n * (n - 1)) * ((g.V - g.V.mean()) ** 2).sum())
        nr = rj['matched_null']['both']; value = float(A.V.sum())
        prob = float((1 + (draw.both >= value).sum()) / (1 + len(draw)))
        assert abs(means - nr['null_mean']) < 1e-6 and abs(np.sqrt(variances) - nr['null_sd']) < 1e-6
        assert prob == nr['p_draws_one_sided'] and len(draw) == 1000
        assert abs(value - ej['values']['A']['both']['M50|A58|L37|N100']) < 1e-6
        # Independent tail/window/component reconciliation over the saved per-order ledger.
        losses = np.maximum(-A.V.values, 0.); rt = rj['risk']['A']['both']; severe = A.V.values < -30.
        assert abs(losses.sum() - rt['total_loss']) < 1e-6
        assert abs(np.maximum(losses - 30., 0.).sum() - rt['total_severe_excess']) < 1e-6
        assert np.allclose(A[['v_w0', 'v_w1', 'v_w2']].sum(axis=1), A.V, rtol=0, atol=1e-6)
        assert np.allclose(A.spread_comp + A.move_comp, A.V, rtol=0, atol=1e-6)
        window = {name: float(A[f'v_w{i}'].values[severe].sum()) for i, name in enumerate(RV.WINDOWS)}
        assert all(abs(window[k] - rj['risk']['A']['where']['window (severe episodes)'][k]) < 1e-6 for k in window)
        # Reconcile the actual A fills without calling either ledger/outcome implementation.
        kr = read(kp / day / 'receipt.json')
        for f in ('orders.parquet', 'fills.parquet'):
            checkhash(kp / day / f, kr['outputs'][f])
        o = pd.read_parquet(kp / day / 'orders.parquet', columns=['oid', 'side', 'p', 'mark'])
        f = pd.read_parquet(kp / day / 'fills.parquet', columns=['oid', 'clock', 't', 'q'])
        f = f[f.clock == 50.].merge(o, on='oid', validate='many_to_one').merge(A[['oid', 'admitted', 'act', 'T']], on='oid', validate='many_to_one')
        mask = f.admitted & (~f.act | (f.t < 50. * f['T'] + 37.))
        vv = f.q * np.where(f.side == 0, 1., -1.) * (f.mark - f.p)
        assert abs(float(vv[mask].sum()) - value) < 1e-6
        predname = f'pred_{day}.parquet'
        checkhash(p1 / predname, j1['outputs'][predname]); checkhash(p2 / predname, j2['outputs'][predname])
        cols = ['oid', 'kind', 'j', 'side', 'A_V', 'AF_V', 'AM_V', 'pi0_V', 'pi0_ok', 'NEU_V', 'NEU_ok', 'pair_v', 'pair_q']
        oldp = pd.read_parquet(p1 / predname, columns=cols); newp = pd.read_parquet(p2 / predname, columns=cols)
        assert all(oldp[c].equals(newp[c]) for c in cols)
        po, re = newp[newp.kind == 'post'], newp[newp.kind == 'retain']
        adm = set(po.oid[po.A_V > 0]); first = re[re.A_V < 0].groupby('oid').j.min()
        reached = re.oid.isin(adm) & (re.j <= re.oid.map(first).fillna(np.inf))
        assert int(reached.sum()) == ej['shift']['reached_rows']
        # Does the new raw-zero/weighted-zero defect change a currently saved group?
        affected = []
        for model, gs in rj['prediction'].items():
            for group, v in gs.items():
                if 'eb_label' in v and ((v['sum_label'] == 0.) != (v['eb_label'] == 0.)):
                    affected.append([model, group])
        result['days'][day] = dict(asof=rj['asof'], original=rj['readings']['V4 POPULATIONS']['original'],
            path_orders=len(A), reached_rows=int(reached.sum()), predictions_unchanged=True,
            baseline_A_value=value, matched_null=dict(mean=means, sd=float(np.sqrt(variances)), p=prob, draws=len(draw)),
            total_loss=float(losses.sum()), severe_episodes=int(severe.sum()), severe_window_values=window,
            screen_bug_saved_groups_affected=affected, validity=rj['readings']['VALIDITY (all five)'])
        print(day, json.dumps(result['days'][day]), flush=True)
        del ledger, A, AA, oldp, newp, re, po, f, o; gc.collect()
    result['verified_hashes'] = verified
    save('receipt.json', result)


if __name__ == '__main__':
    if sys.argv[1] == 'synthetic':
        synthetic()
    elif sys.argv[1] == 'artifacts':
        artifact_audit()
    elif sys.argv[1] == 'selftest':
        RV.selftest()
    else:
        raise ValueError(sys.argv[1])
