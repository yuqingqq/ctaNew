"""Declared, bounded INCSIZE comparison on two already-consumed days.

No interval, significance claim, tuning or adoption: two days are descriptive.
The treatment includes the documented reservation/quantity behavior, not just
queue retention. Run via run_incremental_replay.sh (one research process).
"""
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
S4 = ROOT / 'data/predict_fun/research/step4'
DEST = S4 / 'incremental_20261107'
DAYS = ['20261024', '20261101']
BASE = 'MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60|mid|AM:5:25:95'
INC = BASE.replace('+LD2+', '+LD2+INCSIZE+')
ARMS = {'v1': BASE, 'incremental': INC}
PREFIX = 'incsz_20261107'
RESEARCH = ROOT / 'predictfun/research'
CODE = ['pf_step4_ladder.py', 'pf_step4_ladder_incremental.py', 'pf_incremental_size.py',
        'pf_ladder_incremental_build.py', 'pf_ladder_inv_build.py', 'pf_inv_r60.py',
        'pf_ladder_fwd_build.py', 'pf_clock_rl.py', 'pf_io.py', 'pf_incremental_replay.py', 'run_incremental_replay.sh']
COMMON = {'PF_RL_DDEC': '2', 'PF_RL_LPLACE': '40', 'PF_RL_LCANCEL': '22', 'PF_RL_LPENDCANCEL': '95',
          'PF_RL_LFILL': '100', 'PF_TIMING': 'e2', 'PF_TXTIME': 'ws', 'PF_TXKEY': 'tk', 'PF_PREDS': 'preds_t60k',
          'PF_FILLPRIO': 'price', 'PF_OWNCROSS': 'on', 'PF_LEGORDER': 'price', 'PF_REJRESET': 'on', 'PF_POLICYVIEW': 'known'}


def hashes():
    return {p: hashlib.sha256((RESEARCH / p).read_bytes()).hexdigest() for p in CODE}


def name(spec):
    p, r, m, s = spec.split('|')
    return f'{p}+{r}/{m}@{s}'


def stem(spec):
    return name(spec).replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-') + '.parquet'


def directory(tag, day):
    return S4 / f'ladder_tokyoreal_custom_e2_preds_t60k_{tag}' / day


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def declare():
    if DEST.exists() or any(directory(PREFIX + suffix, DAYS[0]).parent.exists() for suffix in ('_smokeJ', '_J', '_X')):
        raise ValueError('REFUSED: incremental comparison destination already exists')
    prior = json.loads((ROOT / 'predictfun/reviews/incremental_size_20261106/receipt.json').read_text())
    if not prior['passed'] or any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h for p, h in prior['code_sha256'].items()):
        raise ValueError('REFUSED: source-only validation receipt does not match current implementation')
    DEST.mkdir(parents=True)
    manifest = dict(declared_at=datetime.now(timezone.utc).isoformat(), days=DAYS, timings=['J', 'X'], arms=ARMS,
                    smoke=dict(day=DAYS[0], timing='J', first_windows=8), code_sha256=hashes(),
                    scope='exploratory in-sample, consumed days; no tuning or adoption',
                    inference='point estimates only; two days, no intervals or hypothesis test',
                    decision_metrics=['settlement_pnl', 'mark_at_common_60s', 'inventory', 'fills', 'markout_1s', 'requests'],
                    control='v1 output must match the existing round-i artifact',
                    positive_control='INCSIZE must engage and change the request stream in at least one full run',
                    negative_controls='summary validator must reject wrong day, arms, timing, and code identity',
                    forward_data_read=False, live_or_frozen_strategy_changed=False)
    write(DEST / 'declaration.json', manifest)
    (DEST / 'arms.txt').write_text(';'.join(ARMS.values()) + '\n')
    print(f'Declared {DEST}')


def validate_summary(s, day, timing, tag, maximum, identities):
    if s.get('day') != day or set(s.get('arms', {})) != {name(x) for x in ARMS.values()}:
        raise ValueError('day/arm identity mismatch')
    expected = dict(COMMON, PF_ARMS=';'.join(ARMS.values()), PF_TAG=tag, PF_MAX_WINDOWS=str(maximum))
    expected.update(PF_TXJIT='frame' if timing == 'J' else '', PF_TXSHIFT='0' if timing == 'J' else '-50',
                    PF_QACT='' if timing == 'J' else 'max')
    if any(s.get('env', {}).get(k) != v for k, v in expected.items()):
        raise ValueError('timing/strategy environment mismatch')
    actual = s.get('code_sha256', {})
    required = {'pf_step4_ladder.py', 'pf_step4_ladder_incremental.py', 'pf_incremental_size.py',
                'pf_ladder_incremental_build.py', 'pf_ladder_inv_build.py', 'pf_inv_r60.py', 'pf_clock_rl.py', 'pf_io.py'}
    if set(actual) != required or any(actual[k] != identities[k] for k in required):
        raise ValueError('implementation identity mismatch')


def summary_controls(s, day, timing, tag, maximum, identities):
    validate_summary(s, day, timing, tag, maximum, identities)
    corruptions = []
    bad = copy.deepcopy(s); bad['day'] = '19000101'; corruptions.append(bad)
    bad = copy.deepcopy(s); bad['arms'] = {}; corruptions.append(bad)
    bad = copy.deepcopy(s); bad['env']['PF_RL_LPLACE'] = '0'; corruptions.append(bad)
    bad = copy.deepcopy(s); bad['code_sha256']['pf_incremental_size.py'] = 'bad'; corruptions.append(bad)
    for bad in corruptions:
        try:
            validate_summary(bad, day, timing, tag, maximum, identities)
        except ValueError:
            continue
        raise ValueError('summary falsifier did not fire')
    return dict(valid_summary_accepted=True, corrupt_summaries_refused=len(corruptions))


def check_run(day, timing, smoke, identities):
    import numpy as np
    import pandas as pd
    tag = PREFIX + ('_smokeJ' if smoke else '_' + timing)
    loc = directory(tag, day)
    summary = json.loads((loc / 'summary.json').read_text())
    controls = summary_controls(summary, day, timing, tag, 8 if smoke else 0, identities)
    baseline = directory('l' + timing + 'i', day)
    reference = json.loads((baseline / 'summary.json').read_text())
    results, panels = {}, {}
    for label, spec in ARMS.items():
        w = pd.read_parquet(loc / 'win' / stem(spec))
        f = pd.read_parquet(loc / 'fills' / stem(spec))
        q = pd.read_parquet(loc / 'requests' / stem(spec))
        a = summary['arms'][name(spec)]
        if len(w) != (8 if smoke else 288) or w.mid.duplicated().any() or not np.isfinite(w[['pnl', 'mtm_stop', 'carry_stop']]).all().all():
            raise ValueError('incomplete/invalid window population')
        if len(f) != a['fills'] or len(w) != a['windows'] or q.kind.value_counts().to_dict() != a['live']['requests']:
            raise ValueError('event totals do not reconcile to summary')
        if not np.allclose(w.pnl, w.mtm_stop + w.carry_stop, atol=1e-8, rtol=0):
            raise ValueError('settlement/mark/carry reconciliation failed')
        if not np.isclose(f.settle.sum(), w.pnl.sum(), atol=1e-8, rtol=0) or not np.isclose(f.q.sum(), w.shares.sum()):
            raise ValueError('fill accounting does not reconcile to windows')
        if not f.mid.isin(w.mid).all() or not q.mid.isin(w.mid).all():
            raise ValueError('events outside declared window population')
        if label == 'incremental':
            for p in (f, q):
                if 'order_slot' not in p or not (p.order_slot % 4 == p.lane).all():
                    raise ValueError('tranche/logical-lane identities disagree')
        else:
            for kind, p in [('win', w), ('fills', f), ('requests', q)]:
                ref = pd.read_parquet(baseline / kind / stem(BASE))
                if smoke:
                    ref = ref[ref.mid.isin(w.mid)]
                if not p.reset_index(drop=True).equals(ref.reset_index(drop=True)):
                    raise ValueError(f'baseline {kind} differs from frozen round-i artifact')
            if not smoke and a != reference['arms'][name(BASE)]:
                raise ValueError('baseline summary differs from frozen round-i artifact')
        missing = dict(mark_at_stop=int(w.mtm_stop.isna().sum()), markout_1s=int(f.mo1.isna().sum()),
                       markout_60s=int(f.mo60.isna().sum()), outcome_windows_excluded=288 - len(w) if not smoke else None)
        shares = float(f.q.sum())
        counters = a['live']['counters']
        if counters.get('own_cross', 0):
            raise ValueError('own resting orders crossed')
        path = w.sort_values('ws').pnl.cumsum().to_numpy()
        peak = np.maximum.accumulate(np.r_[0., path])[1:]
        results[label] = dict(windows=len(w), fills=len(f), shares=shares,
                              settle=float(w.pnl.sum()), mark_60s=float(w.mtm_stop.sum()), carry=float(w.carry_stop.sum()),
                              mo1=float(f.mo1.sum()) if not missing['markout_1s'] else None,
                              mo1_c_per_share=100 * float(f.mo1.sum()) / shares if shares and not missing['markout_1s'] else None,
                              requests=q.kind.value_counts().to_dict(), abs_inventory_mean=float(w.net_end.abs().mean()),
                              abs_inventory_p90=float(w.net_end.abs().quantile(.9)),
                              max_drawdown=float((path - peak).min()), worst_window=float(w.pnl.min()),
                              filled_order_age_median_ms=float(f.age_ms.median()) if len(f) else None,
                              missing=missing, counters=counters)
        panels[label] = (w, q)
    if not panels['v1'][0][['mid', 'ws']].equals(panels['incremental'][0][['mid', 'ws']]):
        raise ValueError('arms cover different windows')
    a, b = results['v1'], results['incremental']
    differences = {key: b[key] - a[key] for key in ('settle', 'mark_60s', 'carry', 'shares', 'fills', 'abs_inventory_mean')}
    differences['requests'] = {k: b['requests'].get(k, 0) - a['requests'].get(k, 0) for k in ('place', 'cancel')}
    changed = not panels['v1'][1].equals(panels['incremental'][1].drop(columns='order_slot'))
    return dict(day=day, timing=timing, baseline_parity=True, controls=controls,
                request_stream_changed=changed, arms=results, delta=differences)


def report(smoke=False):
    declaration = json.loads((DEST / 'declaration.json').read_text())
    if declaration['code_sha256'] != hashes():
        raise ValueError('REFUSED: implementation changed after declaration')
    pairs = [(DAYS[0], 'J')] if smoke else [(day, tm) for tm in ('J', 'X') for day in DAYS]
    runs = [check_run(d, tm, smoke, declaration['code_sha256']) for d, tm in pairs]
    if not smoke and not any(r['request_stream_changed'] and r['arms']['incremental']['counters'].get('inc_same_price_tranches', 0) > 0 for r in runs):
        raise ValueError('INCSIZE engagement positive control failed')
    totals = {}
    for tm in ('J', 'X'):
        subset = [r for r in runs if r['timing'] == tm]
        if not subset:
            continue
        totals[tm] = {}
        for label in ARMS:
            rows = [r['arms'][label] for r in subset]
            totals[tm][label] = {k: sum(r[k] for r in rows) for k in ('windows', 'fills', 'shares', 'settle', 'mark_60s', 'carry')}
            totals[tm][label]['requests'] = {k: sum(r['requests'].get(k, 0) for r in rows) for k in ('place', 'cancel')}
        totals[tm]['delta'] = {k: totals[tm]['incremental'][k] - totals[tm]['v1'][k] for k in ('settle', 'mark_60s', 'carry', 'fills', 'shares')}
    out = dict(as_of=datetime.now(timezone.utc).isoformat(), declaration=declaration, checks_passed=True,
               runs=runs, totals=totals, statistical_inference=False, forward_data_read=False,
               limitation='Two consumed days; descriptive point estimates only. Full order-management variant, not isolated resizing effect.')
    dest = DEST / ('smoke.json' if smoke else 'report.json')
    write(dest, out)
    print(json.dumps(dict(report=str(dest), checks_passed=True, totals=totals), indent=2))


if __name__ == '__main__':
    try:
        mode = sys.argv[1]
        if mode == 'declare':
            declare()
        elif mode == 'smoke':
            report(smoke=True)
        elif mode == 'report':
            report()
        else:
            raise ValueError('expected declare, smoke, or report')
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(f'REFUSED: {exc}')
