"""Post-result audit of B12/B13/B14/B16/B18 artifacts; no historical training or replay."""
from pathlib import Path
import copy
import datetime as dt
import hashlib
import json
import math
import subprocess
import numpy as np
import pandas as pd
from .pf_b13a_pressure_model import engineer
from .pf_b18_pressure_fills import classes

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / 'predictfun/reviews'
OUT = REV / 'maker_comparison_20261111'
GROUPS = ['b12b_20261111', 'b12d_20261111', 'b13a_20261111', 'b13bc_20261111',
          'b14_20261111', 'b16_20261111', 'b12g_20261111', 'b18_20261111']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def equal(a, b):
    if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-8):
        raise ValueError(f'identity mismatch: {a} != {b}')


def read(group, name):
    return json.loads((REV / group / name).read_text())


def replay_check(r, frames):
    n = 0
    for arm, frame in frames.items():
        for timing in ('J', 'X'):
            g = frame[frame.timing == timing]
            if len(g) != 9 or g.day.duplicated().any():
                raise ValueError('day population mismatch')
            for metric in ('gross_usd', 'mo1_usd', 'mo60_usd'):
                delta = (g['arm_' + metric] - g['ref_' + metric]).to_numpy()
                saved = r['replay_' + arm][metric + '_' + timing]
                equal(float(delta.sum()), saved['delta_sum'])
                equal(float((delta > 0).sum()), saved['days_positive'])
                for x, y in zip(delta, saved['per_day']):
                    equal(float(x), y)
                n += 1
    return n


def controls():
    fake = {'replay_q': {}}
    frame = pd.DataFrame(dict(timing=['J']*9+['X']*9, day=list(range(9))*2))
    for metric in ('gross_usd', 'mo1_usd', 'mo60_usd'):
        frame['arm_'+metric] = 2.; frame['ref_'+metric] = 1.
        for timing in ('J', 'X'):
            fake['replay_q'][metric+'_'+timing] = dict(delta_sum=9., days_positive=9, per_day=[1.]*9)
    equal(replay_check(fake, {'q': frame}), 6)
    bad = copy.deepcopy(fake); bad['replay_q']['gross_usd_J']['delta_sum'] += 1
    refused = 0
    for result, rows in ((bad, frame), (fake, frame.iloc[:-1])):
        try:
            replay_check(result, {'q': rows})
        except ValueError:
            refused += 1
    if refused != 2:
        raise ValueError('negative control failed')
    return dict(planted_totals_recovered=True, corrupt_total_and_population_refused=refused)


def pressure_examples():
    # Exercise the actual production research functions, not a copied classifier.
    counts = np.array(([0., 100.] * 300)[:-1] + [40.])
    f = pd.DataFrame(dict(mid=1, pf_imb1=0., bn_imb=0., pf_lbs=np.log1p(100.),
                         pf_las=np.log1p(100.), bn_ntr1000=counts, pf_dm250=0., bn_flow250=-1.))
    score = float(engineer(f).bn_act_rel.iloc[-1])
    c = classes(np.array([1.]), np.array([0.]), np.log1p([100.]), np.log1p([100.]),
                np.array([score]), np.array([-1.]), np.array([0.]))
    unknown = classes(*[np.array([x]) for x in [1., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
    positive = classes(*[np.array([x]) for x in [1., -.5, np.log1p(70.), np.log1p(100.), 1., -1., -.4]])
    if not all(x[0] for x in positive.values()):
        raise ValueError('pressure positive control failed')
    if not c['burst'][0] or counts[-1] >= 2 * counts.mean() or any(x[0] for x in unknown.values()):
        raise ValueError('source behavior changed; re-review findings')
    return dict(synthetic_only=True, current_trade_count=float(counts[-1]), arithmetic_mean=float(counts.mean()),
        actual_log_activity_score=score, existing_burst_flag=bool(c['burst'][0]),
        arithmetic_twice_mean_flag=bool(counts[-1] >= 2 * counts.mean()),
        all_missing_inputs_return_any=bool(unknown['any'][0]),
        caveat='No historical impact estimated. Thinning additionally lacks a same-price identity predicate in the source.')


def main():
    ctl = controls(); hashes = {}
    for group in GROUPS:
        for line in (REV / group / 'SHA256SUMS').read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            p = REV / group / name.lstrip('*')
            if sha(p) != digest:
                raise ValueError('changed artifact: ' + str(p))
            hashes[str(p.relative_to(ROOT))] = digest
    r = read('b16_20261111', 'report.json')
    frames = {a: pd.read_csv(REV / 'b16_20261111' / ('replay_days_' + a + '.csv')) for a in ('qa', 'qb')}
    replay_metrics = replay_check(r, frames)
    b12 = read('b12g_20261111', 'b12g.json'); b18 = read('b18_20261111', 'b18.json')
    horizons_checked = 0
    for scope, groups in b12['pooled'].items():
        for group, row in groups.items():
            rows = [d[scope][group] for d in b12['days'].values()]
            weight = sum(x['shares'] for x in rows); equal(weight, row['shares'])
            for key in ('v1', 'v5', 'v60', 'vs'):
                equal(sum(x[key]*x['shares'] for x in rows)/weight, row[key]); horizons_checked += 1
    pressure_checked = 0
    for group, row in b18['pooled'].items():
        rows = [d[group] for d in b18['days'].values()]
        weight = sum(x['shares'] for x in rows); equal(weight, row['shares'])
        for name in ('thin', 'thinning', 'burst', 'bnshift', 'any'):
            w = sum(x['shares']*x[name]['share'] for x in rows)
            equal(w/weight, row[name]['share'])
            for h in ('v1', 'v60'):
                equal(sum(x['shares']*x[name]['share']*x[name][h+'_in'] for x in rows)/w, row[name][h+'_in'])
                equal(sum(x['shares']*(1-x[name]['share'])*x[name][h+'_out'] for x in rows)/(weight-w), row[name][h+'_out'])
                if group in b12['pooled']['window_0_240']:
                    overall = row[name]['share']*row[name][h+'_in']+(1-row[name]['share'])*row[name][h+'_out']
                    equal(overall, b12['pooled']['window_0_240'][group][h])
                pressure_checked += 2
    examples = pressure_examples()
    churn = read('b13bc_20261111', 'b13c.json')
    out = dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(), source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), role='Post-result arithmetic and source audit; no new training, historical replay or validation',
        controls=ctl, source_sha={str(Path(__file__).relative_to(ROOT)):sha(__file__),
            **{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'predictfun/research/pf_b13a_pressure_model.py', ROOT/'predictfun/research/pf_b18_pressure_fills.py', ROOT/'predictfun/research/pf_b16_queue_label.py']}},
        artifact_sha=hashes, counts=dict(artifact_hashes=len(hashes), replay_metric_totals=replay_metrics,
        maker_horizon_aggregates=horizons_checked, pressure_conditional_aggregates=pressure_checked),
        churn=dict(windows=220, observed_date='20261108', pulls_at_best=churn['pulls_at_best'],
            median_life_ms=churn['life_ms_p10_50_90'][1], same_price_replace_share=churn['replace']['share']),
        b16_qb=r['replay_qb'], b12g_pooled=b12['pooled'], b18_pooled=b18['pooled'], b18_status=b18['statuses'],
        pressure_source_examples=examples,
        limits=['Artifact arithmetic only: does not verify missing raw events, fill clocks, queue placement or causal identification.',
                'B18 conditions on fills; fill share is not quote exposure or a maker cancellation rate.',
                'B16 changes whole paths; its differences do not isolate extra fills or prove horizon causality.',
                'No protected Nov4-6/Nov9-13 payload read; no strategy or live changes.'])
    OUT.mkdir(exist_ok=False)
    with (OUT/'audit.json').open('x') as f:
        json.dump(out, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(as_of=out['as_of'], controls=ctl, counts=out['counts'], pressure_examples=examples), indent=2))


if __name__ == '__main__':
    main()
