"""Entry-only, side-specific POST/WAIT experiment on consumed dates.

Neutral POST/KEEP reference, one entry action per generation. The target is
whole-portfolio POST-now minus WAIT-one-decision under the same neutral
continuation, marked at 60.2s. It is a first policy-improvement target, not an
optimal waiting value. At inference, WAIT is reconsidered every 50ms; it never
expires into a forced post. All arms keep resting orders subject to the same
engine safety constraints. No keep/cancel model is trained here.

Separate bid/ask regressors, shared pinned hyperparameters. Directional
features are also expressed relative to the order side; raw features remain.
Comparators: ALWAYS, SCHEDULED (guard clear, no timer), TIMER (legacy guard and
250ms cooldown), FLAT. These are entry comparisons with a fixed neutral keeper,
not reproductions of the other session's learned cancellation strategy.

A local 500-draw null matches day/side/hour and positive-action count. Local
counterfactual values are never added into portfolio P&L. Five chronological
consumed-date folds only; no forward-validation claim. Fees/rebates excluded,
queue/fill assumptions inherited. Immutable stage outputs and source hashes.
"""
import argparse
import ast
import collections
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_entry_value as E

K = E.K
FOLDS = E.FOLDS
SIDE_NAMES = {0: 'bid', 1: 'ask'}
DIRECTIONAL = ['bn_r25', 'bn_r50', 'bn_r100', 'bn_r250', 'bn_r500', 'bn_r1000',
               'bn_r3000', 'bn_imb', 'bn_flow250', 'bn_flow1000', 'bn_flow5000',
               'pf_micro_dev', 'pf_imb1', 'pf_imb3', 'pf_dm250', 'pf_dm1000', 'pf_dm5000']
FEATURES = E.FEATURES + ['side_' + n for n in DIRECTIONAL] + ['side_position']
ARMS = ('VALUE', 'ALWAYS', 'SCHEDULED', 'TIMER', 'FLAT')
BASELINES = ('ALWAYS', 'SCHEDULED', 'TIMER')
CFGS = E.CFGS
SAMPLE_MOD = 4
DEFAULT_OUT = Path(K.IO) / 'wait_post_20261114_v1'


@functools.lru_cache(maxsize=1)
def sources():
    found = set(); pending = ['pf_wait_post']
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        for node in ast.walk(ast.parse((HERE / (name + '.py')).read_text())):
            names = ([x.name for x in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) else [])
            pending.extend(n for n in names if n and n.startswith('pf_') and '.' not in n and n not in found)
    return sorted(n + '.py' for n in found)


def stamp():
    return dict(contract='wait-post-side-heads-v1', decision_ms=50, features=FEATURES,
                size=E.SIZE, nr=E.NR, cap=E.CAP, keeper='neutral-keep-engine-safety',
                target='post-minus-one-decision-wait-neutral-continuation-terminal',
                side_heads=SIDE_NAMES, sample_mod=SAMPLE_MOD, seed=E.SEED,
                source_sha256={n: E.digest(HERE / n) for n in sources()})


def require(meta):
    # JSON object keys are strings after round trip.
    expected = json.loads(json.dumps(stamp()))
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f'wait/post artifact mismatch: {key}')


def features(W, view):
    if view['side'] not in SIDE_NAMES:
        raise ValueError('invalid side')
    x = E.features(W, view); sign = 1. if view['side'] == 0 else -1.
    return np.r_[x, [sign * x[E.FEATURES.index(n)] for n in DIRECTIONAL],
                 sign * view['known_position']].astype(np.float32)


def build_window(W, day, market, sample_mod=SAMPLE_MOD):
    records = {}; candidates = {}; statuses = collections.Counter()
    def observe(v, action):
        statuses[v['status']] += 1
        if v['kind'] != 'entry' or v['status'] != 'eligible':
            return
        key = E.unit(v)
        if key in records:
            raise ValueError('neutral entry generation was offered more than once')
        chosen = E.hash_unit(day, market, key) % sample_mod == 0
        records[key] = dict(day=str(day), market=int(market), side=v['side'], rung=v['rung'],
                            kind='entry', generation=v['generation'], k=v['k'], t=v['t'],
                            status='pending_label' if chosen else 'not_sampled')
        if chosen:
            candidates[key] = dict(v)
    base = E.run(W, lambda v: True, observe=observe)
    E.terminal(W, base)
    for key, view in candidates.items():
        try:
            x = features(W, view)
        except ValueError:
            records[key]['status'] = 'missing_input'; continue
        pair = E.paired_probe(W, view, base)
        records[key].update(status='eligible', **pair, **dict(zip(FEATURES, map(float, x))))
    return list(records.values()), dict(statuses)


def rows_day(day, out, max_windows=None):
    E.check_day(day)
    if max_windows is not None and max_windows < 1:
        raise ValueError('max_windows must be positive')
    initial = stamp(); stage = E.new_stage(out, 'rows_' + day)
    started = time.monotonic(); data = K.load_day(day); records = []; counts = collections.Counter(); markets = []
    for W in E.PAIRS.window_pairs(data):
        row, count = build_window(W, day, W[0]['m']); records.extend(row); counts.update(count); markets.append(W[0]['m'])
        n = len(markets)
        if n % 4 == 0 or max_windows is not None:
            print(f'{day}: {n} windows, {len(records)} entry generations, RSS {E.rss_mib():.0f} MiB, {time.monotonic()-started:.1f}s', flush=True)
        if E.rss_mib() > 2048:
            raise RuntimeError('RSS cap exceeded')
        if max_windows is not None and n >= max_windows:
            break
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError('no neutral entry generations')
    if stamp() != initial:
        raise ValueError('source changed during labels')
    frame.to_parquet(stage / 'rows.parquet', index=False)
    expected = set(data['W0']); seen = set(markets)
    if len(seen) != len(markets) or not seen <= expected:
        raise ValueError('window identity mismatch')
    diagnostic = []
    for (side, rung), g in frame.groupby(['side', 'rung']):
        eligible = g[g.status == 'eligible']
        diagnostic.append(dict(side=int(side), rung=int(rung), generations=len(g), eligible=len(eligible),
            zero_targets=int((eligible.target_c.abs() < 1e-10).sum()) if len(eligible) else 0,
            positive_targets=int((eligible.target_c > 1e-10).sum()) if len(eligible) else 0,
            negative_targets=int((eligible.target_c < -1e-10).sum()) if len(eligible) else 0))
    E.write_json(stage / 'receipt.json', dict(**initial, asof=now(), day=day, windows=len(markets),
        markets=markets, excluded_markets=sorted(expected-seen), complete=seen == expected,
        partial=max_windows is not None, source_status=data['stat'], decision_status=dict(counts),
        action_status=frame.status.value_counts().to_dict(), support=diagnostic,
        rows_sha256=E.digest(stage / 'rows.parquet'), wall_seconds=time.monotonic()-started))


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def load_rows(out, day):
    path = Path(out) / ('rows_' + day); meta = E.read_json(path / 'receipt.json'); require(meta)
    if meta['day'] != day or meta['partial'] or not meta['complete'] or E.digest(path / 'rows.parquet') != meta['rows_sha256']:
        raise ValueError('partial, incomplete, modified or mismatched rows')
    rows = pd.read_parquet(path / 'rows.parquet')
    if not (rows.day == day).all() or not (rows.kind == 'entry').all() or not rows.side.isin(SIDE_NAMES).all():
        raise ValueError('wrong action population')
    if rows.duplicated(['day', 'market', 'side', 'rung', 'generation']).any():
        raise ValueError('duplicate entry generations')
    return rows


def fit_fold(test, out):
    import lightgbm as lgb
    if test not in FOLDS:
        raise ValueError('undeclared fold')
    stage = E.new_stage(out, 'fit_' + test); initial = stamp(); train, cal = FOLDS[test]
    if not all(d < cal < test for d in train):
        raise ValueError('nonchronological split')
    data = pd.concat([load_rows(out, day) for day in train], ignore_index=True)
    data = data[data.status == 'eligible']; models = {}; support = {}; importances = []
    params = dict(E.PD.LGB_PARAMS, objective='regression')
    for side, name in SIDE_NAMES.items():
        r = data[data.side == side]
        if len(r) < 200 or r.day.nunique() < 2:
            raise ValueError(f'insufficient {name} actions: {len(r)} on {r.day.nunique()} days')
        if len(r) > E.PD.SUB_N:
            r = r.sample(E.PD.SUB_N, random_state=E.SEED)
        y = r.target_c.to_numpy(float)
        if not np.isfinite(y).all():
            raise ValueError('nonfinite target')
        # Refuse a head with no identifiable action contrast instead of treating
        # a constant zero predictor as a model of timing.
        if np.count_nonzero(np.abs(y) > 1e-10) < 20:
            raise ValueError(f'insufficient nonzero {name} targets')
        models[side] = lgb.train(params, lgb.Dataset(r[FEATURES].to_numpy(np.float32), np.clip(y, -10., 10.), feature_name=FEATURES), E.PD.N_ROUNDS)
        models[side].save_model(str(stage / (name + '.txt')))
        support[name] = dict(n=len(r), days=sorted(r.day.unique().tolist()), nonzero=int((abs(y) > 1e-10).sum()),
                             positive=int((y > 1e-10).sum()), negative=int((y < -1e-10).sum()))
        importances.extend(dict(side=name, feature=f, gain=float(g)) for f, g in zip(FEATURES, models[side].feature_importance(importance_type='gain')))
    pd.DataFrame(importances).to_parquet(stage / 'importance.parquet', index=False)
    predictions = {}
    for day in (cal, test):
        r = load_rows(out, day); r = r[r.status == 'eligible'].copy(); r['prediction'] = np.nan
        for side, model in models.items():
            mask = r.side == side
            if mask.any():
                r.loc[mask, 'prediction'] = model.predict(r.loc[mask, FEATURES].to_numpy(np.float32))
        r[['day', 'market', 'side', 'rung', 'kind', 'generation', 't', 'target_c', 'prediction']].to_parquet(stage / ('pred_' + day + '.parquet'), index=False)
        predictions[day] = E.digest(stage / ('pred_' + day + '.parquet'))
    if stamp() != initial:
        raise ValueError('source changed during fitting')
    E.write_json(stage / 'receipt.json', dict(**initial, asof=now(), test=test, train=list(train), cal=cal,
        training_support=support, params=params, rounds=E.PD.N_ROUNDS, threshold_c=0.,
        row_receipts={day: E.digest(Path(out) / ('rows_' + day) / 'receipt.json') for day in (*train, cal, test)},
        predictions_sha256=predictions, model_sha256={name: E.digest(stage / (name + '.txt')) for name in SIDE_NAMES.values()}))


def load_models(test, out):
    import lightgbm as lgb
    path = Path(out) / ('fit_' + test); meta = E.read_json(path / 'receipt.json'); require(meta)
    train, cal = FOLDS[test]
    if meta['test'] != test or meta['train'] != list(train) or meta['cal'] != cal:
        raise ValueError('fold mismatch')
    models = {}
    for side, name in SIDE_NAMES.items():
        if E.digest(path / (name + '.txt')) != meta['model_sha256'][name]:
            raise ValueError('model hash mismatch')
        model = lgb.Booster(model_file=str(path / (name + '.txt')))
        if model.feature_name() != FEATURES:
            raise ValueError('model features mismatch')
        models[side] = model
    return models


def policy(W, models, arm='VALUE', audit=None, capture=None):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    def choose(v):
        if arm == 'FLAT':
            return False
        if v['kind'] == 'rest':
            return True  # Same keeper, every arm; safety remains in the engine.
        if arm in ('ALWAYS', 'TIMER'):
            return True
        if arm == 'SCHEDULED':
            return bool(W[v['side']]['vkl'][v['k']] > K.GUARD_PULL)
        try:
            x = features(W, v)
        except ValueError:
            if audit is not None: audit['missing_input'] += 1
            return False
        score = float(models[v['side']].predict(x[None, :])[0])
        if not np.isfinite(score):
            if audit is not None: audit['nonfinite_prediction'] += 1
            return False
        action = score > 0.
        if capture is not None: capture(v, x, score, action)
        if audit is not None: audit[f'{SIDE_NAMES[v["side"]]}:{v["rung"]}:{"post" if action else "wait"}'] += 1
        return action
    return choose


def eval_day(test, out, max_windows=None):
    if test not in FOLDS:
        raise ValueError('undeclared fold')
    stage = E.new_stage(out, 'eval_' + test); initial = stamp(); models = load_models(test, out); data = K.load_day(test)
    records = []; samples = []; counts = collections.Counter(); markets = []
    for W in E.PAIRS.window_pairs(data):
        markets.append(W[0]['m'])
        for shift in CFGS:
            for arm in ARMS:
                audit = collections.Counter(); calls = [0]
                def capture(v, x, score, action):
                    calls[0] += 1
                    if calls[0] % 1000 == 1:
                        samples.append(dict(day=test, market=W[0]['m'], arm=arm, shift=int(shift), side=v['side'], k=v['k'],
                            prediction=score, action=action, **dict(zip(FEATURES, map(float, x)))))
                r = E.run(W, policy(W, models, arm, audit, capture), shift=shift, entry_mode='legacy' if arm == 'TIMER' else 'value')
                counts.update({f'{arm}:{k}': v for k, v in audit.items()})
                counts.update({f'{arm}:{k}': v for k, v in r['policy_status'].items()})
                records.append(dict(day=test, market=W[0]['m'], arm=arm, shift=int(shift), **E.metrics(W, r)))
        if len(markets) % 4 == 0:
            print(f'{test}: {len(markets)} windows replayed, RSS {E.rss_mib():.0f} MiB', flush=True)
        if E.rss_mib() > 2048: raise RuntimeError('RSS cap exceeded')
        if max_windows is not None and len(markets) >= max_windows: break
    pd.DataFrame(records).to_parquet(stage / 'windows.parquet', index=False)
    frame = pd.DataFrame(samples)
    if frame.empty: raise ValueError('no inference samples')
    frame.to_parquet(stage / 'inference.parquet', index=False)
    for side, group in frame.groupby('side'):
        if not np.allclose(models[side].predict(group[FEATURES].to_numpy(np.float32)), group.prediction, rtol=0, atol=1e-9):
            raise ValueError('inference reconstruction differs')
    if stamp() != initial: raise ValueError('source changed during replay')
    expected = set(data['W0']); seen = set(markets)
    if len(markets) != len(seen) or not seen <= expected: raise ValueError('window identity mismatch')
    E.write_json(stage / 'receipt.json', dict(**initial, asof=now(), day=test, markets=markets, excluded_markets=sorted(expected-seen),
        complete=seen == expected and max_windows is None, partial=max_windows is not None, counts=dict(counts), source_status=data['stat'],
        windows_sha256=E.digest(stage / 'windows.parquet'), inference_sha256=E.digest(stage / 'inference.parquet'),
        fit_receipt_sha256=E.digest(Path(out) / ('fit_' + test) / 'receipt.json')))


def report(out):
    stage = E.new_stage(out, 'report'); parts = []; predictions = []; exclusions = {}
    for day in FOLDS:
        path = Path(out) / ('eval_' + day); meta = E.read_json(path / 'receipt.json'); require(meta)
        if meta['day'] != day or not meta['complete'] or meta['partial']: raise ValueError('incomplete/wrong replay day')
        for name in ('windows', 'inference'):
            if E.digest(path / (name + '.parquet')) != meta[name + '_sha256']: raise ValueError('modified replay')
        fit = Path(out) / ('fit_' + day)
        if E.digest(fit / 'receipt.json') != meta['fit_receipt_sha256']: raise ValueError('modified fit')
        fm = E.read_json(fit / 'receipt.json'); require(fm)
        if E.digest(fit / ('pred_' + day + '.parquet')) != fm['predictions_sha256'][day]: raise ValueError('modified predictions')
        r = pd.read_parquet(path / 'windows.parquet')
        if not (r.day == day).all() or set(r.arm) != set(ARMS) or set(r['shift']) != set(CFGS): raise ValueError('wrong replay population')
        if r.duplicated(['day', 'market', 'arm', 'shift']).any(): raise ValueError('duplicate portfolio window')
        groups = list(r.groupby(['arm', 'shift']))
        if len(groups) != len(ARMS)*len(CFGS) or any(set(g.market) != set(meta['markets']) for _, g in groups):
            raise ValueError('unmatched market identities')
        parts.append(r); predictions.append(pd.read_parquet(fit / ('pred_' + day + '.parquet')))
        exclusions[day] = dict(source=meta['source_status'], decisions=meta['counts'], excluded_markets=meta['excluded_markets'])
    full = pd.concat(parts, ignore_index=True); pred = pd.concat(predictions, ignore_index=True)
    daily = full.groupby(['day', 'arm', 'shift']).sum(numeric_only=True).drop(columns=['market'])
    daily.reset_index().to_parquet(stage / 'daily.parquet', index=False)
    rng = np.random.default_rng(E.SEED); comparisons = []
    for baseline in BASELINES:
        for shift in CFGS:
            a = daily.xs(('VALUE', shift), level=('arm', 'shift')); b = daily.xs((baseline, shift), level=('arm', 'shift'))
            for column in ('pnl', 'pnl_without_young'):
                delta = (a[column]-b[column]).to_numpy(); boot = rng.choice(delta, (1000, len(delta)), replace=True).mean(1)
                comparisons.append(dict(baseline=baseline, shift=shift, metric=column, n_days=len(delta), mean_delta=float(delta.mean()),
                    positive_days=int((delta > 0).sum()), descriptive_day_interval=np.quantile(boot, [.025, .975]).tolist(),
                    passes=bool(delta.mean() > 0 and (delta > 0).sum() >= 4)))
    main = daily.xs(('VALUE', 50), level=('arm', 'shift'))
    positive = bool(main.pnl.sum() > 0 and main.pnl_without_young.sum() > 0)
    risk_ok = all(main.inventory_square_seconds.sum() <= daily.xs((b, 50), level=('arm', 'shift')).inventory_square_seconds.sum() for b in BASELINES)
    side_diagnostics = []
    for (side, rung), group in pred.groupby(['side', 'rung']):
        side_diagnostics.append(dict(side=int(side), rung=int(rung), n=len(group), post_share=float((group.prediction > 0).mean()),
            nonzero_target_share=float((group.target_c.abs() > 1e-10).mean()), local_selection=E.local_null(group)))
    E.write_json(stage / 'result.json', dict(**stamp(), asof=now(), n_days=len(FOLDS), dates=list(FOLDS), n_portfolio_windows=len(full),
        development_screen=bool(positive and risk_ok and all(c['passes'] for c in comparisons)), absolute_positive_m50=positive,
        inventory_risk_not_increased_m50=risk_ok, comparisons=comparisons, side_rung=side_diagnostics,
        local_selection=E.local_null(pred), exclusions=exclusions, forward_validation=False,
        limitations=['neutral keeper only; no combined keep/pull claim', 'one-step neutral continuation, not optimal dynamic waiting',
                     'neutral entry opportunities do not cover all repeated-wait states', 'gross simulated value; queue/fill engine unvalidated',
                     'SCHEDULED and TIMER use inherited LODO guard; exploratory comparisons']))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=['rows', 'fit', 'eval', 'report', 'selftest'])
    p.add_argument('day', nargs='?'); p.add_argument('--out', type=Path, default=DEFAULT_OUT); p.add_argument('--max-windows', type=int)
    a = p.parse_args()
    if a.command == 'selftest':
        import unittest
        import test_wait_post
        r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_wait_post))
        if not r.wasSuccessful(): raise SystemExit(1)
    elif a.command == 'rows': rows_day(a.day, a.out, a.max_windows)
    elif a.command == 'fit': fit_fold(a.day, a.out)
    elif a.command == 'eval': eval_day(a.day, a.out, a.max_windows)
    else: report(a.out)


if __name__ == '__main__': main()
