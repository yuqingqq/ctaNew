"""Bounded manual review of the paired clock audit; saved tables only.

Independently enumerate inclusion probabilities by order/decision step, then
compare every fill's contribution and every reported aggregate. No tape replay,
training, new strategy or changed production/research implementation.
"""
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

TARGET = '1a1abbaa0e9b572a6d71c9150aa530a11f5b7e6f'
ROOT = Path('/home/yuqing/ctaNew')
SNAP = Path(sys.argv[1]).resolve()
OUT = Path(__file__).resolve().parent
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SNAP, text=True).strip() == TARGET
sys.path.insert(0, str(SNAP / 'predictfun/research'))
import pf_keep_pull_clock_audit as A
KP = A.KP


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def eq(x, y):
    assert np.allclose(x, y, atol=1e-7, rtol=1e-12, equal_nan=True), (x, y)


def weights(T, cell, endj):
    """Explicit per-order inclusion probabilities, independent of audit's cell cumsum."""
    active = (T >= 2) & (T <= 98) & (T < endj)
    w = np.zeros((len(T), 99))
    for step in range(2, 99):
        counts = np.bincount(cell[active & (T == step)], minlength=cell.max()+1)
        live = endj > step
        sizes = np.bincount(cell[live], minlength=cell.max()+1)
        assert (counts <= sizes).all()
        w[live, step] = counts[cell[live]] / sizes[cell[live]]
    eq(w.sum(), active.sum())
    return w.cumsum(axis=1), active


def contributions(T, active, w, ri, t, val):
    landings = np.arange(1, 99) * 50. + 37.
    last = np.searchsorted(landings, t, side='right')
    before_near = np.searchsorted(landings, t-200., side='right')
    removed = active[ri] & (t >= T[ri] * 50. + 37.)
    return dict(score=np.where(removed, -val, 0.), null=-val*w[ri, last],
                score_near=np.where(removed & (t < T[ri]*50.+237.), -val, 0.),
                null_near=-val*(w[ri, last]-w[ri, before_near]))


def total(c, mask=None):
    s = {key: float(v.sum() if mask is None else v[mask].sum()) for key, v in c.items()}
    return dict(score=s['score'], null=s['null'], excess=s['score']-s['null'], near=s['score_near']-s['null_near'])


def main():
    copies = SNAP / 'predictfun/reviews/keep_pull_clock_audit_20261118'
    data = ROOT / 'data/predict_fun/research/keep_pull_20261117'
    audit_data = ROOT / 'data/predict_fun/research/keep_pull_clock_audit_20261118'
    J = json.loads((copies/'clock_audit.json').read_text())
    evidence = dict(target_commit=TARGET, review_asof=KP.C.stamp(), result_asof=J['asof'],
                    script_sha256=sha(__file__), scope='Consumed Oct23-27 saved artifacts only; no raw tape or validation.',
                    hashes={}, population={}, matrix={}, quantity_diagnostic={})
    for name in ('clock_audit.json', 'clock_audit.txt', 'run.log'):
        actual = (ROOT/'predictfun/reviews/keep_pull_clock_audit_20261118'/name
                  if name == 'run.log' else audit_data/name)
        assert sha(copies/name) == sha(actual)
        evidence['hashes'][name] = sha(copies/name)
    assert sha(SNAP/'predictfun/research/pf_keep_pull_clock_audit.py') == J['audit_source_sha256']
    for name, digest in J['provenance']['source'].items():
        assert sha(SNAP/'predictfun/research'/name) == digest

    A.selftest()
    # Independent positive controls and known-bad arithmetic refusal.
    w, act = weights(np.array([2, 99]), np.array([0, 0]), np.array([99, 99]))
    c = contributions(np.array([2, 99]), act, w, np.zeros(4, int),
                      np.array([136.999, 137., 336.999, 337.]), np.full(4, -10.))
    eq(c['score'], [0., 10., 10., 10.]); eq(c['score_near'], [0., 10., 10., 0.])
    eq(c['null'], [0., 5., 5., 5.]); eq(c['null_near'], [0., 5., 5., 0.])
    try:
        eq(c['score'], [10., 10., 10., 10.])
    except AssertionError:
        pass
    else:
        raise AssertionError('known-bad cancellation boundary accepted')

    sums = {k: dict(pulls=0, score=0., null=0., excess=0., near=0., side={}, src={}) for k in J['R']}
    for day in KP.DAYS:
        receipt = data/day/'receipt.json'
        assert sha(receipt) == J['source_receipts'][day]
        assert sha(receipt) == sha(SNAP/f'predictfun/reviews/keep_pull_20261117/receipt_{day}.json')
        r = json.loads(receipt.read_text())
        for filename, digest in r['outputs'].items():
            assert sha(data/day/filename) == digest
        O = pd.read_parquet(data/day/'orders.parquet')
        Fl = pd.read_parquet(data/day/'fills.parquet')
        E = O[O.status == 'eligible'].reset_index(drop=True)
        assert not E.oid.duplicated().any() and Fl.oid.isin(E.oid).all()
        assert O.status.value_counts().to_dict() == r['status']
        evidence['population'][day] = dict(counted=len(O), eligible=len(E), statuses=r['status'])
        fair = E[[f'f{j:02d}' for j in range(99)]].values
        edge = np.where(E.side.values[:, None] == 0, 1., -1.) * (fair-E.p.values[:, None])
        trigger = (edge < 0) & (np.arange(99)[None, :] >= 2)
        T = np.where(trigger.any(axis=1), trigger.argmax(axis=1), 99)
        eq(T, KP.score_steps(fair, E.side.values, E.p.values)['A+0'])
        cell = E.groupby(['m', 'side', 'phase'], sort=True).ngroup().values
        EJ = {}
        for name, field in (('M50', 't_full'), ('M250', 't_full_m250')):
            lim = np.fmin(E.g_struct.values, E[field].values+100.)
            EJ[name] = 1+(50.*np.arange(1, 99)[None, :] < lim[:, None]).sum(axis=1)
            EJ[name][np.isnan(lim)] = 99
            eq(EJ[name], A.endj_of(E[field].values, E.g_struct.values))
        EJ['COMMON'] = np.minimum(EJ['M50'], EJ['M250'])
        ix = pd.Series(np.arange(len(E)), index=E.oid)
        fill_sets = {}
        for clock in (50, 250):
            f = Fl[Fl.clock == clock]
            ri = ix.reindex(f.oid).values.astype(int)
            val = f.q.values * np.where(E.side.values[ri] == 0, 1., -1.) * (E.mark.values[ri]-E.p.values[ri])
            fill_sets[f'M{clock}'] = (ri, f.t.values, val, f.src.values)
        changed_q = ~np.isclose(E.q.values, E.q_m250.values, atol=1e-9, rtol=0.)
        evidence['quantity_diagnostic'][day] = dict(eligible=len(E), orders_with_changed_total_fill_quantity=int(changed_q.sum()),
            shares_M50=float(E.q.sum()), shares_M250=float(E.q_m250.sum()))
        for G, endj in EJ.items():
            w, active = weights(T, cell, endj)
            for F, fl in fill_sets.items():
                key = f'{F}_{G}'; ri, t, val, src = fl
                c = contributions(T, active, w, ri, t, val)
                actual = A.per_fill(T, cell, endj, fl)
                for name, value in c.items():
                    eq(actual[name], value)
                pooled = total(c)
                for name, value in pooled.items():
                    eq(value, J['R'][key]['day'][day][name])
                    sums[key][name] += value
                sums[key]['pulls'] += int(active.sum())
                for kind, labels in (('side', np.where(E.side.values[ri] == 0, 'bid', 'ask')), ('src', src.astype(str))):
                    for label in np.unique(labels):
                        part = total(c, labels == label)
                        dest = sums[key][kind].setdefault(label, dict.fromkeys(part, 0.))
                        for name, value in part.items():
                            dest[name] += value
            del w
        print(f'{day}: source hashes, all six paired cells and per-fill/near contributions pass', flush=True)
        del E, O, Fl, fair, fill_sets
        gc.collect()
    for key, values in sums.items():
        for name in ('pulls', 'score', 'null', 'excess', 'near'):
            eq(values[name], J['R'][key][name])
        for kind in ('side', 'src'):
            for label, part in values[kind].items():
                for name, value in part.items():
                    eq(value, J['R'][key][kind][label][name])
    # Independently recover both paths and their average, by pool, side and source.
    for kind, labels in (('pooled', [None]), ('side', ['bid', 'ask']), ('src', ['0', '1', '2', '3'])):
        for label in labels:
            ex = {key: (v['excess'] if label is None else v[kind][label]['excess']) for key, v in sums.items()}
            a, b, c, d = [ex[k] for k in ('M50_M50', 'M50_M250', 'M250_M50', 'M250_M250')]
            got = dict(total=d-a, fill=(c-a+d-b)/2, eligibility=(b-a+d-c)/2, fill_first=c-a, eligibility_first=b-a)
            saved = J['shapley']['pooled'] if label is None else J['shapley'][kind][KP.SRC[int(label)] if kind == 'src' else label]
            for name, value in got.items():
                eq(value, saved[name])
    assert J['readings']['A1 ATTRIBUTION'] == A.a1(J['shapley']['pooled'])
    assert J['readings']['A2 SAME-OPPORTUNITY STABILITY'] == A.a2({(50., 'C'): J['R']['M50_COMMON'], (250., 'C'): J['R']['M250_COMMON']}, list(KP.DAYS))
    evidence.update(matrix=sums, shapley=J['shapley'], controls=dict(pinned_selftest=True, independent_boundary_control=True,
        known_bad_boundary_refused=True), full_report_rerun=False, raw_tape_read=False, untouched_validation=False)
    (OUT/'receipt_v1.json').write_text(json.dumps(KP.C.clean(evidence), indent=2)+'\n')
    print(json.dumps(dict(shapley=J['shapley']['pooled'], quantity_diagnostic=evidence['quantity_diagnostic']), indent=2))


if __name__ == '__main__':
    main()
