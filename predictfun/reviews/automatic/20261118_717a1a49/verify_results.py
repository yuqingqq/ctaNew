"""Manual review: saved keep/pull tables, absolute values and clock sensitivity.

Bounded saved-artifact checks only. No tape replay, fit or full report rerun.
The analytical STATES expectation is a post-hoc diagnostic, not a new policy.
"""
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

TARGET = '717a1a49f27320def34cb466ca73b7325daf8ea3'
SNAP = Path(sys.argv[1]).resolve()
ROOT = Path('/home/yuqing/ctaNew')
OUT = Path(__file__).resolve().parent
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SNAP, text=True).strip() == TARGET
sys.path.insert(0, str(SNAP / 'predictfun/research'))
import pf_keep_pull as KP  # noqa: E402


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def close(actual, expected):
    assert np.allclose(actual, expected, atol=1e-6, rtol=1e-12, equal_nan=True), (actual, expected)


def expected_weights(endj, P, cell):
    """P(order chosen at j) under the declared independent state sampling.

    Each cell samples k distinct live orders uniformly. Its inclusion probability
    is k / live_count. Linearity gives exact expected removed value, even if the
    same order is sampled at multiple steps. This is not trajectory P&L.
    """
    n = len(P)
    weights = np.zeros((n, KP.NSTEP + 1))
    active = (P >= KP.J0) & (P <= KP.NSTEP) & (P < endj)
    cells = int(cell.max()) + 1
    for step in range(KP.J0, KP.NSTEP + 1):
        count = np.bincount(cell[active & (P == step)], minlength=cells)
        if not count.any():
            continue
        live = endj > step
        available = np.bincount(cell[live], minlength=cells)
        assert (count <= available).all()
        weights[live, step] = count[cell[live]] / available[cell[live]]
    close(weights.sum(), active.sum())
    return weights.cumsum(axis=1)


def main():
    copies = SNAP / 'predictfun/reviews/keep_pull_20261117'
    data = ROOT / 'data/predict_fun/research/keep_pull_20261117'
    v1 = ROOT / 'data/predict_fun/research/queue_place_20261117'
    J = json.loads((copies / 'keep_pull.json').read_text())
    evidence = dict(target_commit=TARGET, review_asof=KP.C.stamp(), result_asof=J['asof'],
                    scope='Consumed Oct23-27 derived artifacts only; no raw tape, training, full report rerun or untouched validation.',
                    review_script_sha256=sha(__file__), artifact_hashes={}, per_day={}, source_clock={})
    for saved in sorted(copies.rglob('*')):
        if not saved.is_file():
            continue
        rel = saved.relative_to(copies)
        actual = (data / saved.stem.removeprefix('receipt_') / 'receipt.json'
                  if saved.name.startswith('receipt_') else data / rel)
        assert sha(saved) == sha(actual), rel
        evidence['artifact_hashes'][str(rel)] = sha(saved)
        if saved.suffix == '.log' and saved.name != 'driver.log':
            assert ' rc=0 ' in saved.read_text().splitlines()[-1]

    # Positive and known-bad arithmetic controls; no implementation changes.
    close(np.array([1., 2.]), np.array([1., 2.]))
    try:
        close(np.array([1., 2.001]), np.array([1., 2.]))
    except AssertionError:
        pass
    else:
        raise AssertionError('value mismatch accepted')
    w = expected_weights(np.array([99, 99]), np.array([2, 3]), np.array([0, 0]))
    close(w[:, 2], [.5, .5]); close(w[:, 3], [1., 1.])
    # Four equally likely independent state draws: both orders have one late
    # fill, worth -2 and +4. Mean improvement over the two sampled states is -2.
    close(-np.dot(np.array([-2., 4.]), w[:, 3]), -2.)
    assert not KP._predicates() and not KP._exhaustion_cases()

    totals = {key: dict(keep=0., pull_all_improvement=0., oracle_improvement=0.,
                       **({'score_improvement': 0.} if key in J['D'] else {})) for key in J['R']}
    fair_live = live_n = 0
    oracle_steps = set()
    clock_scores = {c: [] for c in (50., 250.)}
    for day_index, day in enumerate(KP.DAYS):
        r = json.loads((copies / f'receipt_{day}.json').read_text())
        assert not r['provenance']['uncommitted']
        for name, digest in r['provenance']['source'].items():
            assert sha(SNAP / 'predictfun/research' / name) == digest
        for filename, digest in r['outputs'].items():
            assert sha(data / day / filename) == digest
        O = pd.read_parquet(data / day / 'orders.parquet')
        Fl = pd.read_parquet(data / day / 'fills.parquet').assign(day=day)
        assert not O.oid.duplicated().any()
        assert O.status.value_counts().to_dict() == r['status'] == J['population']['status'][day]
        assert KP.reconcile(O, v1, day) == r['reconciliation']
        ledger = pd.read_parquet(ROOT / 'data/predict_fun/research/queue_place_20261117_v2' / day / 'state_arms.parquet')
        joins = ledger[ledger.arm == 'JOIN']
        check = O.merge(joins[['m', 'side', 't', 'status', 'rej']], on=['m', 'side', 't'], how='outer', suffixes=('', '_ledger'), indicator=True)
        assert check._merge.eq('both').all()
        expect_status = np.where(check.status_ledger.eq('eligible') & (check.rej > 0), 'refused_post_only', check.status_ledger)
        assert np.array_equal(check.status, expect_status)
        E = O[O.status == 'eligible'].reset_index(drop=True)
        fair = E[[f'f{j:02d}' for j in range(99)]].values
        policies = KP.score_steps(fair, E.side.values, E.p.values)
        # Independently regenerate the primary threshold rule from raw fair inputs.
        edge = np.where(E.side.values[:, None] == 0, 1., -1.) * (fair - E.p.values[:, None])
        trigger = (edge < 0) & (np.arange(99)[None, :] >= 2)
        primary = np.where(trigger.any(axis=1), trigger.argmax(axis=1), 99)
        assert np.array_equal(primary, policies['A+0'])
        cell = E.groupby(['m', 'side', 'phase'], sort=True).ngroup().values
        oid_to_row = pd.Series(np.arange(len(E)), index=E.oid)
        old_oracle = None
        for key, saved in J['R'].items():
            parts = key.split('_')
            clock, latency = float(parts[0][1:]), float(parts[-1][1:])
            noyoung = 'noyoung' in parts
            V = KP.variant(E, Fl, clock, noyoung, latency)
            days = np.zeros(len(E), int)
            o = KP.oracle(V)
            oracle_steps.update(o[o < 99].tolist())
            oracle = KP.outcome(V, o, days, 1)
            pull_all = KP.outcome(V, np.full(len(E), 2), days, 1)
            pending = KP.outcome(V, KP.oracle(V, 1), days, 1, 1)
            close(float(np.maximum(-V['Vk'], 0).sum()), saved['keep_loss_by_day'][day_index])
            for name, result in (('oracle', oracle), ('pull_all', pull_all)):
                for metric, value in result.items():
                    close(value[0], saved[name + '_by_day'][metric][day_index])
            if key == 'M50_L0':
                old_oracle = o.copy()
                close(pending['imp'][0], saved['oracle_pending_by_day']['imp'][day_index])
            if key == 'M50_L37':
                close(KP.outcome(V, old_oracle, days, 1)['imp'][0], saved['oracle_0ms_decisions']['imp'][day_index])
                live = (np.arange(1, 99)[None, :] >= 2) & (np.arange(1, 99)[None, :] < V['endj'][:, None])
                live_n += int(live.sum()); fair_live += int((live & np.isnan(fair[:, 1:])).sum())
            totals[key]['keep'] += float(V['Vk'].sum())
            totals[key]['pull_all_improvement'] += float(pull_all['imp'][0])
            totals[key]['oracle_improvement'] += float(oracle['imp'][0])
            if key in J['D']:
                for policy, record in J['D'][key].items():
                    got = KP.outcome(V, policies[policy], days, 1)
                    for metric, value in got.items():
                        close(value[0], record['policy_by_day'][metric][day_index])
                    if policy == 'A+0':
                        totals[key]['score_improvement'] += float(got['imp'][0])

            if key in ('M50_L37', 'M250_L37'):
                f = Fl[Fl.clock == clock].copy()
                ri = oid_to_row.reindex(f.oid).values
                assert np.isfinite(ri).all()
                ri = ri.astype(int)
                sign = np.where(E.side.values == 0, 1., -1.)
                value = f.q.values * sign[ri] * (E.mark.values[ri] - E.p.values[ri])
                active = (primary >= 2) & (primary <= 98) & (primary < V['endj'])
                removed = active[ri] & (f.t.values >= KP.landing(np.minimum(primary, 98), 37.)[ri])
                actual_imp = -float(value[removed].sum())
                close(actual_imp, J['D'][key]['A+0']['policy_by_day']['imp'][day_index])
                W = expected_weights(V['endj'], primary, cell)
                last_step = np.searchsorted(KP.landing(np.arange(1, 99), 37.), f.t.values, side='right')
                expected_count = W[ri, last_step]
                per_source = {}
                for code, source in enumerate(KP.SRC):
                    mask = f.src.values == code
                    si = -float(value[mask & removed].sum())
                    ni = -float(np.dot(value[mask], expected_count[mask]))
                    per_source[source] = dict(keep_value_c=float(value[mask].sum()), score_improvement_c=si,
                                              expected_states_improvement_c=ni, excess_c=si-ni,
                                              removed_shares=float(f.q.values[mask & removed].sum()))
                for side, name in ((0, 'bid'), (1, 'ask')):
                    mask = E.side.values[ri] == side
                    si = -float(value[mask & removed].sum()); ni = -float(np.dot(value[mask], expected_count[mask]))
                    per_source[name] = dict(score_improvement_c=si, expected_states_improvement_c=ni, excess_c=si-ni)
                clock_scores[clock].append(dict(day=day, pulled_orders=int(active.sum()),
                    score_improvement_c=actual_imp, exact_expected_states_improvement_c=-float(np.dot(value, expected_count)),
                    monte_carlo_states_mean_c=J['D'][key]['A+0']['states']['null_by_day_mean']['imp'][day_index], sources=per_source))
                del W
            del V
        evidence['per_day'][day] = dict(asof=r['asof'], counted=len(O), eligible=len(E), statuses=r['status'],
                                       source_hashes=r['provenance']['source'], output_hashes=r['outputs'], original_reconciliation=r['reconciliation'])
        print(f'{day}: hashes, ledger/original reconciliation and policy/oracle arithmetic pass', flush=True)
        del E, O, Fl, ledger
        gc.collect()

    close(live_n, J['population']['live_decisions']); close(fair_live, J['population']['live_decisions_fair_missing'])
    assert oracle_steps == {2}
    for key, row in totals.items():
        close(row['keep'], J['R'][key]['keep_value'])
        row['pull_all_absolute_value_c'] = row['keep'] + row['pull_all_improvement']
        row['oracle_absolute_value_c'] = row['keep'] + row['oracle_improvement']
        if key in J['D']:
            row['score_absolute_value_c'] = row['keep'] + row['score_improvement']
    for key, variants in J['D'].items():
        for policy, result in variants.items():
            assert result['draws'] >= 200
            if not result['traj']['evaluable']:
                assert result['traj']['k2'] is None and result['traj']['k3'] is None
                assert result['estimand'] == 'decision states'
            chosen = result['traj'] if result['traj']['evaluable'] else result['states']
            assert result['k2'] == chosen['k2'] and result['k3'] == chosen['k3']
            assert result['k2'] == (chosen['k2_stats']['imp'] > chosen['k2_stats']['null_p95'] and chosen['k2_stats']['days_above'] >= 4)
            assert result['k3'] == (chosen['k3_stats']['per_share'] < chosen['k3_stats']['null_p5'] and chosen['k3_stats']['days_below'] >= 4)
    for clock, rows in clock_scores.items():
        summed = {k: sum(x[k] for x in rows) for k in ('pulled_orders', 'score_improvement_c', 'exact_expected_states_improvement_c', 'monte_carlo_states_mean_c')}
        summed['excess_over_exact_state_mean_c'] = summed['score_improvement_c'] - summed['exact_expected_states_improvement_c']
        summed['sources'] = {name: {metric: sum(x['sources'][name][metric] for x in rows)
                                           for metric in rows[0]['sources'][name]} for name in rows[0]['sources']}
        summed['by_day'] = rows
        evidence['source_clock'][str(int(clock))] = summed
    evidence.update(absolute_values=totals, oracle_executed_steps=sorted(oracle_steps),
                    controls=dict(value_mismatch_refused=True, state_expectation_control=True, reading_predicate_controls=True, exhaustion_controls=True),
                    caveats='Analytical state expectations are local repeated-state sums. Source attribution is descriptive and changes alongside fills/eligibility across clocks; it does not identify a feed error or deployable policy. Full Monte Carlo quantiles not rerun.')
    evidence['supersedes'] = 'receipt_v1.json: omit uncomputed score fields at latencies with no declared score evaluation; all evaluated results unchanged.'
    (OUT / 'receipt_v2.json').write_text(json.dumps(KP.C.clean(evidence), indent=2) + '\n')
    print(json.dumps(KP.C.clean(dict(absolute_values=totals, source_clock={k:{n:v for n,v in x.items() if n!='by_day'} for k,x in evidence['source_clock'].items()})), indent=2))


if __name__ == '__main__':
    main()
