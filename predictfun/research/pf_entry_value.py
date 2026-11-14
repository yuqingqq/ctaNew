"""POST/WAIT and KEEP/CANCEL with one economic metric, research candidate v1.

An empty rung posts iff E[V(post now) - V(wait this decision)] > 0. A resting
order stays iff E[V(keep now) - V(cancel now)] >= 0. Each is reconsidered at
50 ms decisions. Neither decision uses a retry timer or legacy guard veto.
V is whole-window cash plus inventory at the common 60.2 s terminal, including
cancel-pending fills, for the joint six-rung, two-sided, 30-share ladder.
Actual notification/placement/cancel delays, held-price reservations and a
1500-share hard exposure cap still apply. Costs are the inherited gross-value
convention: fees/rebates are excluded, no new inventory penalty is invented.

Paired targets use the SAME neutral post/keep reference and continuation on
both branches, with exactly one action changed. WAIT defers one decision;
CANCEL may re-enter when a subsequent decision and the actual reservations
allow it. Labels for the old guard/cooldown action set are incompatible.
This is one-step policy improvement from a neutral continuation, NOT an optimal
Q function. Repeated learned actions must be tested in the full replay.

One entry probe per neutral empty generation; one resting probe per neutral
order generation, at a hash-declared age from 0 ms through 51.2 s. Generation hash samples
1/16, independent of its later outcome. Early-ended generations and missing
inputs remain statuses. No selection on fills or favorable outcomes; local
overlapping branch differences are NEVER summed as portfolio profit.

Two regressors use the same pinned PULL-DEPTH parameters. Five chronological
outer folds train before the preceding calibration day; zero thresholds are
fixed and never selected on calibration/test outcomes. No iterative target
backups, upstream guard scores or FRONT-PULL predictions enter the features.
The old loader still requires its input coverage; this is a stated support
limitation. Public queue estimates and the fill engine remain unvalidated.

Candidate: VALUE (both heads). Controls: TIMER (same keep head, old entry gate),
ALWAYS (same keep head, post every legal empty slot), FLAT. Headline cap 50x30;
M0/M50/M250 execution-clock scenarios. Acceptance requires positive M50 gross
terminal value and improvement vs TIMER and ALWAYS pooled and >=4/5 days at
each clock, both with all fills and excluding fills on orders under 100 ms old,
without increasing integrated squared inventory vs either at M50. The age
exclusion is an accounting sensitivity, not a sequential minimum-age policy.
Day bootstrap intervals are descriptive (1000 draws); no forward claim on
the nine already consumed dates. Fixed local null: 500 selections matched by
day, side, UTC hour and action kind, at the model's positive-action count.
It measures local selection only, never portfolio economics.

CLI: selftest | rows DAY [--max-windows N] | fit TESTDAY | eval TESTDAY | report
Use --out for a NEW output namespace. No historical stage runs on import.
"""
import argparse
import ast
import collections
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_entry_value_lane as L
import pf_kp_s4 as K
import pf_order_harm_v2 as OH
import pf_order_harm_policy as PAIRS
import pf_pull_depth as PD

SIZE = 30.0
NR = 6
CAP = 50 * SIZE
SAMPLE_MOD = 16
PROBE_AGES_MS = (0., 50., 100., 200., 400., 800., 1600., 3200., 6400., 12800., 25600., 51200.)
AGE_SLOTS = len(PROBE_AGES_MS)
SEED = 20261114
FOLDS = {K.DAYS9[i]: (tuple(K.DAYS9[:i - 1]), K.DAYS9[i - 1]) for i in range(4, 9)}
DEFAULT_OUT = Path(K.IO) / 'entry_value_20261114_v1'
ARMS = ('VALUE', 'TIMER', 'ALWAYS', 'FLAT')
CFGS = (0., 50., 250.)
EXTRA = ['side_sign', 'rung', 'price', 'depth', 'spread', 'age', 'quantity', 'known_filled',
         'ahead_known', 'initial_ahead_known', 'shown_known', 'size_better', 'queue_fraction',
         'known_position', 'own_exposure', 'other_exposure', 'capacity_left', 'time_remaining', 'frame_age']
FEATURES = list(K.G32) + EXTRA


@functools.lru_cache(maxsize=1)
def source_names():
    """Bind local research dependencies, including imports inside functions."""
    pending = ['pf_entry_value']; found = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        p = HERE / f'{name}.py'
        if not p.is_file():
            raise ValueError(f'missing source dependency {name}')
        found.add(name)
        for node in ast.walk(ast.parse(p.read_text())):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) else [])
            pending.extend(n for n in names if n and n.startswith('pf_') and '.' not in n and n not in found)
    return tuple(f'{name}.py' for name in sorted(found))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rss_mib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return float(line.split()[1]) / 1024.
    raise RuntimeError('RSS is unavailable')


def stamp():
    return dict(**L.stamp(), features=FEATURES, size=SIZE, nr=NR, cap=CAP,
                target='neutral-single-action-terminal-cash-inventory-difference',
                sampling=dict(mod=SAMPLE_MOD, probe_ages_ms=list(PROBE_AGES_MS), seed=SEED),
                source_sha256={n: digest(HERE / n) for n in source_names()})


def require(meta):
    for k, v in stamp().items():
        if meta.get(k) != v:
            raise ValueError(f'incompatible entry-value artifact on {k}')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def new_stage(out, name):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.mkdir()  # Refuse overwriting either a completed or a failed attempt.
    return p


def unit(v):
    return v['side'], v['rung'], v['kind'], v['generation']


def action_key(v):
    return v['k'], *unit(v)


def hash_unit(day, market, u):
    s = json.dumps([SEED, str(day), int(market), list(u)], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], 'big')


def features(W, v):
    """Only received market frames and the engine's sanitized client state."""
    w = W[v['side']]; t = v['t']; k = v['k']
    j = int(np.searchsorted(w['kf'], t, side='right')) - 1
    if j < 0 or not 0 <= k < len(w['X']):
        raise ValueError('missing known market frame/input')
    x = np.asarray(w['X'][k, :len(K.G32)], float)
    if len(x) != len(K.G32) or not np.isfinite(x).any():
        raise ValueError('missing market inputs')
    best, opp = int(w['P5'][j, 0]), int(w['opp'][j]); p = v['price']; s = v['side']
    if best < 0 or opp < 0 or (best >= opp if s == 0 else best <= opp):
        raise ValueError('invalid known book')
    prices, sizes = w['P5'], w['S5']
    shown = np.where((prices[:, 4] >= 0) & ((p < prices[:, 4]) if s == 0 else (p > prices[:, 4])),
                     np.nan, (sizes * (prices == p)).sum(axis=1))
    # A hypothetical new order joins behind the CURRENT known book, not the
    # future activation frame used by the simulator to determine execution.
    activation = t if v['kind'] == 'entry' else v['activation']
    a0, ah, sh = OH.queue_known({'ta': activation}, t, shown, w['kf'].tolist(), w['tf'].tolist())
    better = float(sizes[j][(prices[j] >= 0) & ((prices[j] > p) if s == 0 else (prices[j] < p))].sum())
    own = v['exposure_bid'] if s == 0 else v['exposure_ask']
    other = v['exposure_ask'] if s == 0 else v['exposure_bid']
    cap = CAP if v['cap'] is None else v['cap']
    extra = [1. if s == 0 else -1., v['rung'], p, best - p if s == 0 else p - best,
             abs(opp - best), max(0., t - activation), v['quantity'], v['known_filled'], ah, a0, sh, better,
             ah / sh if np.isfinite(ah) and np.isfinite(sh) and sh > 0 else np.nan,
             v['known_position'], own, other, cap - own, v['end_time'] - t, t - w['kf'][j]]
    out = np.asarray(list(x) + extra, np.float32)
    if np.isinf(out).any():
        raise ValueError('infinite feature')
    return out


def run(W, choose, shift=50., observe=None, entry_mode='value'):
    guards = {s: (np.asarray(W[s]['vkl']) <= K.GUARD_PULL).tolist() for s in (0, 1)}
    result = L.sim_joint(W, 'M', shift, CAP, guards, size=SIZE, nr=NR, mode='hard',
                         policy=choose, observe=observe, entry_mode=entry_mode)
    if any(result['stat']['viol'].values()):
        raise ValueError(f"engine accounting failure: {result['stat']['viol']}")
    return result


def terminal(W, r):
    tf, mid = W[0]['tf_mff']
    return L.portfolio_terminal(r, tf, mid)


def paired_probe(W, view, base=None):
    """Neutral POST/KEEP vs a single WAIT/CANCEL at exactly the same state."""
    base = run(W, lambda v: True) if base is None else base
    target = action_key(view); hit = []
    def fork(v):
        if action_key(v) == target:
            hit.append(dict(v))
            return False
        return True
    changed = run(W, fork)
    if len(hit) != 1 or hit[0] != view:
        raise ValueError('fork did not encounter the identical client decision state')
    a = base['policy_decisions']; b = changed['policy_decisions']
    index = next(i for i, z in enumerate(a) if (z[0], z[1], z[2], z[3], z[4]) == target)
    if a[:index] != b[:index] or a[index][-1] is not True or b[index][-1] is not False:
        raise ValueError('paired fork changed the pre-action decision prefix')
    va, vb = terminal(W, base), terminal(W, changed)
    if va['time'] != vb['time']:
        raise ValueError('unequal terminal times')
    return dict(value_yes=va['pnl'], value_no=vb['pnl'], target_c=100. * (va['pnl'] - vb['pnl']) / SIZE,
                terminal=va['time'])


def build_window(W, day, market, sample_mod=SAMPLE_MOD, age_slots=AGE_SLOTS):
    records = {}; candidates = {}; statuses = collections.Counter()
    def observe(v, action):
        statuses[v['status']] += 1
        if v['status'] != 'eligible':
            return
        u = unit(v); h = hash_unit(day, market, u)
        if u not in records:
            records[u] = dict(day=str(day), market=int(market), side=v['side'], rung=v['rung'],
                              kind=v['kind'], generation=v['generation'], k=v['k'], t=v['t'],
                              status='not_sampled' if h % sample_mod else 'ended_before_probe')
        if h % sample_mod or u in candidates:
            return
        delay = 0. if v['kind'] == 'entry' else PROBE_AGES_MS[(h // sample_mod) % age_slots]
        if v['kind'] == 'rest' and v['t'] - v['activation'] < delay:
            return
        candidates[u] = dict(v)
    base = run(W, lambda v: True, observe=observe)
    terminal(W, base)  # Validate no-fill windows and every terminal, too.
    for u, v in candidates.items():
        record = records[u]; record.update(k=v['k'], t=v['t'])
        try:
            x = features(W, v)
        except ValueError:
            record['status'] = 'missing_input'; continue
        pair = paired_probe(W, v, base)
        record.update(status='eligible', **pair, **dict(zip(FEATURES, map(float, x))))
    return list(records.values()), dict(statuses)


def check_day(day):
    if day not in K.DAYS9:
        raise ValueError('only the declared consumed development dates are allowed')


def rows_day(day, out, max_windows=None):
    if max_windows is not None and max_windows < 1:
        raise ValueError('max_windows must be positive')
    check_day(day); stage = new_stage(out, f'rows_{day}')
    initial = stamp(); data = K.load_day(day); ledger = []; counts = collections.Counter(); n = 0
    for W in PAIRS.window_pairs(data):
        rows, st = build_window(W, day, W[0]['m']); ledger.extend(rows); counts.update(st); n += 1
        if n % 12 == 0:
            print(f'{day}: {n} windows, {len(ledger)} generation records, rss {rss_mib():.0f} MiB', flush=True)
        if rss_mib() > 2048:
            raise RuntimeError('memory cap exceeded')
        if max_windows is not None and n >= max_windows:
            break
    frame = pd.DataFrame(ledger)
    if frame.empty:
        raise ValueError('no neutral actions')
    frame.to_parquet(stage / 'rows.parquet', index=False)
    if stamp() != initial:
        raise ValueError('source changed during label generation')
    write_json(stage / 'receipt.json', dict(**initial, asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               day=day, windows=n, partial=max_windows is not None, source_status=data['stat'], decision_status=dict(counts),
               action_status=frame.status.value_counts().to_dict(), rows_sha256=digest(stage / 'rows.parquet')))


def load_rows(out, day):
    stage = Path(out) / f'rows_{day}'; meta = read_json(stage / 'receipt.json'); require(meta)
    if meta['partial'] or meta['day'] != day or digest(stage / 'rows.parquet') != meta['rows_sha256']:
        raise ValueError('partial, mismatched or modified rows')
    rows = pd.read_parquet(stage / 'rows.parquet')
    if not (rows.day == day).all() or rows.duplicated(['day', 'market', 'side', 'rung', 'kind', 'generation']).any():
        raise ValueError('wrong day or duplicate action generations')
    return rows


def fit_fold(test, out):
    import lightgbm as lgb
    if test not in FOLDS:
        raise ValueError('undeclared fold')
    stage = new_stage(out, f'fit_{test}'); train, cal = FOLDS[test]; initial = stamp()
    if not all(d < cal < test for d in train):
        raise ValueError('nonchronological fold')
    train_rows = pd.concat([load_rows(out, d) for d in train], ignore_index=True)
    train_rows = train_rows[train_rows.status == 'eligible']
    models = {}; counts = {}
    params = dict(PD.LGB_PARAMS, objective='regression')
    for kind in ('entry', 'rest'):
        r = train_rows[train_rows.kind == kind]
        if len(r) < 200 or r.day.nunique() < 2:
            raise ValueError(f'insufficient {kind} training actions: {len(r)}')
        if len(r) > PD.SUB_N:
            r = r.sample(PD.SUB_N, random_state=SEED)
        x = r[FEATURES].to_numpy(np.float32); y = r.target_c.to_numpy(float)
        if not np.isfinite(y).all():
            raise ValueError('nonfinite training target')
        models[kind] = lgb.train(params, lgb.Dataset(x, np.clip(y, -10., 10.), feature_name=FEATURES), PD.N_ROUNDS)
        models[kind].save_model(str(stage / f'{kind}.txt')); counts[kind] = len(r)
    for day in (cal, test):
        r = load_rows(out, day); r = r[r.status == 'eligible'].copy()
        r['prediction'] = np.nan
        for kind, model in models.items():
            sel = r.kind == kind
            if sel.any():
                r.loc[sel, 'prediction'] = model.predict(r.loc[sel, FEATURES].to_numpy(np.float32))
        r[['day', 'market', 'side', 'rung', 'kind', 'generation', 't', 'target_c', 'prediction']].to_parquet(stage / f'pred_{day}.parquet', index=False)
    if stamp() != initial:
        raise ValueError('source changed during fitting')
    write_json(stage / 'receipt.json', dict(**initial, test=test, train=list(train), cal=cal, training_actions=counts,
               thresholds=dict(entry=0., rest=0.), params=params, rounds=PD.N_ROUNDS,
               row_receipts={d: digest(Path(out) / f'rows_{d}' / 'receipt.json') for d in (*train, cal, test)},
               model_sha256={kind: digest(stage / f'{kind}.txt') for kind in models}))


def load_models(test, out):
    import lightgbm as lgb
    stage = Path(out) / f'fit_{test}'; meta = read_json(stage / 'receipt.json'); require(meta)
    train, cal = FOLDS[test]
    if meta['test'] != test or meta['train'] != list(train) or meta['cal'] != cal:
        raise ValueError('fold identity mismatch')
    models = {}
    for kind in ('entry', 'rest'):
        path = stage / f'{kind}.txt'
        if digest(path) != meta['model_sha256'][kind]:
            raise ValueError('model identity mismatch')
        model = lgb.Booster(model_file=str(path))
        if model.feature_name() != FEATURES:
            raise ValueError('model feature order mismatch')
        models[kind] = model
    return models


def value_policy(W, models, arm='VALUE', audit=None, capture=None):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    def choose(v):
        if arm == 'FLAT':
            return False
        if v['kind'] == 'entry' and arm in ('ALWAYS', 'TIMER'):
            return True
        try:
            x = features(W, v)
        except ValueError:
            if audit is not None:
                audit['missing_input'] += 1
            return False
        score = float(models[v['kind']].predict(x[None, :])[0])
        if not np.isfinite(score):
            if audit is not None:
                audit['nonfinite_prediction'] += 1
            return False
        action = score > 0. if v['kind'] == 'entry' else score >= 0.
        if capture is not None:
            capture(v, x, score, action)
        return action
    return choose


def metrics(W, r):
    value = terminal(W, r); grid = np.asarray(W[0]['tg']); live = np.zeros((2, NR, len(grid)), bool)
    for (s, d), orders in r['order_states'].items():
        for o in orders:
            if not o['ref']:
                live[s, d] |= (grid >= o['ta']) & (grid < min(o['tr'], o['tfill']))
    by_side = live.any(axis=1)
    position = 0.; area = 0.; previous = float(grid[0]); fills = sorted(r['fl'], key=lambda z: z[2])
    for s, d, t, pt, quantity in fills:
        area += position ** 2 * max(0., t - previous) / 1000.; previous = t
        position += quantity if s == 0 else -quantity
    area += position ** 2 * (value['time'] - previous) / 1000.
    aged_value = 0.; young_shares = 0.
    for (s, d), order_fills in r['fills'].items():
        for f in order_fills:
            if f[4] >= 100.:
                aged_value += (1. if s == 0 else -1.) * f[2] * (value['mark'] - f[1]) / 100.
            else:
                young_shares += f[2]
    return dict(pnl=value['pnl'], cash=value['cash'], position=value['position'], terminal=value['time'],
                pnl_without_young=aged_value, young_shares=young_shares,
                shares=sum(z[4] for z in fills), pulls=sum(r['stat']['pulls']),
                places=sum(len(a) for a in r['order_states'].values()), max_position=r['stat']['maxpos'],
                inventory_square_seconds=area, grid_steps=len(grid), any_quote_steps=int(by_side.any(axis=0).sum()),
                both_quote_steps=int(by_side.all(axis=0).sum()), rung_live_steps=int(live.sum()))


def eval_day(test, out):
    if test not in FOLDS:
        raise ValueError('undeclared test day')
    stage = new_stage(out, f'eval_{test}'); initial = stamp(); models = load_models(test, out); data = K.load_day(test); rows = []
    counts = collections.Counter(); samples = []
    for n, W in enumerate(PAIRS.window_pairs(data), 1):
        for shift in CFGS:
            for arm in ARMS:
                audit = collections.Counter()
                calls = [0]
                def capture(v, x, score, action):
                    calls[0] += 1
                    if calls[0] % 1000 == 1:
                        samples.append(dict(day=test, market=W[0]['m'], arm=arm, shift=int(shift), kind=v['kind'],
                                            k=v['k'], side=v['side'], prediction=score, action=action,
                                            **dict(zip(FEATURES, map(float, x)))))
                r = run(W, value_policy(W, models, arm, audit, capture), shift=shift, entry_mode='legacy' if arm == 'TIMER' else 'value')
                counts.update({f'{arm}:{k}': v for k, v in audit.items()})
                counts.update({f'{arm}:{k}': v for k, v in r['policy_status'].items()})
                rows.append(dict(day=test, market=W[0]['m'], arm=arm, shift=int(shift), **metrics(W, r)))
        if n % 12 == 0:
            print(f'{test}: {n} windows replayed; rss {rss_mib():.0f} MiB', flush=True)
        if rss_mib() > 2048:
            raise RuntimeError('memory cap exceeded')
    pd.DataFrame(rows).to_parquet(stage / 'windows.parquet', index=False)
    saved = pd.DataFrame(samples)
    if saved.empty:
        raise ValueError('no inference samples recorded')
    saved.to_parquet(stage / 'inference.parquet', index=False)
    for kind, group in saved.groupby('kind'):
        predictions = models[kind].predict(group[FEATURES].to_numpy(np.float32))
        if not np.allclose(predictions, group.prediction.to_numpy(float), atol=1e-9, rtol=0):
            raise ValueError('saved inference does not reconstruct')
    if stamp() != initial:
        raise ValueError('source changed during replay')
    actual_windows = int(pd.DataFrame(rows).market.nunique())
    expected_windows = len(data.get('W0', {})) or actual_windows
    write_json(stage / 'receipt.json', dict(**initial, day=test, source_status=data['stat'],
               expected_windows=expected_windows, actual_windows=actual_windows, complete=actual_windows == expected_windows,
               counts=dict(counts), windows_sha256=digest(stage / 'windows.parquet'),
               inference_sha256=digest(stage / 'inference.parquet'), inference_samples=len(saved),
               fit_receipt_sha256=digest(Path(out) / f'fit_{test}' / 'receipt.json')))


def local_null(frame, draws=500):
    """Matched action-selection diagnostic; NEVER sum these as portfolio P&L."""
    if draws < 200:
        raise ValueError('at least 200 matched draws required')
    r = frame.copy(); r['hour'] = (r.t // 3600000).astype('int64')
    groups = list(r.groupby(['day', 'side', 'hour', 'kind'], sort=True))
    chosen = np.where(r.kind == 'rest', r.prediction >= 0., r.prediction > 0.); n = int(chosen.sum())
    if not n:
        return dict(status='no_positive_actions', n=0)
    observed = float(r.loc[chosen, 'target_c'].mean()); rng = np.random.default_rng(SEED); null = []
    for _ in range(draws):
        total = 0.
        for _, g in groups:
            take = int(np.where(g.kind == 'rest', g.prediction >= 0., g.prediction > 0.).sum())
            if take:
                total += float(rng.choice(g.target_c.to_numpy(float), take, replace=False).sum())
        null.append(total / n)
    return dict(status='local_only', n=n, draws=draws, observed_mean_c=observed,
                null_p95_c=float(np.quantile(null, .95)), above_null_p95=bool(observed > np.quantile(null, .95)))


def report(out):
    stage = new_stage(out, 'report'); parts = []; predictions = []
    for day in FOLDS:
        p = Path(out) / f'eval_{day}'; meta = read_json(p / 'receipt.json'); require(meta)
        if meta['day'] != day:
            raise ValueError('wrong replay receipt day')
        if not meta['complete']:
            raise ValueError('incomplete replay day cannot establish acceptance or a day interval')
        if digest(p / 'windows.parquet') != meta['windows_sha256']:
            raise ValueError('modified replay artifact')
        if digest(p / 'inference.parquet') != meta['inference_sha256']:
            raise ValueError('modified inference samples')
        if digest(Path(out) / f'fit_{day}' / 'receipt.json') != meta['fit_receipt_sha256']:
            raise ValueError('modified fitted artifact')
        f = pd.read_parquet(p / 'windows.parquet')
        if not (f.day == day).all() or set(f.arm) != set(ARMS) or set(f['shift']) != set(map(int, CFGS)):
            raise ValueError('wrong day, arm or execution scenario')
        if f.duplicated(['day', 'market', 'arm', 'shift']).any():
            raise ValueError('duplicate portfolio windows')
        counts = f.groupby(['arm', 'shift']).market.nunique()
        if len(counts) != len(ARMS) * len(CFGS) or counts.nunique() != 1:
            raise ValueError('unmatched replay populations')
        sets = [set(g.market) for _, g in f.groupby(['arm', 'shift'])]
        if any(s != sets[0] for s in sets[1:]):
            raise ValueError('mismatched market identities')
        parts.append(f); predictions.append(pd.read_parquet(Path(out) / f'fit_{day}' / f'pred_{day}.parquet'))
    full = pd.concat(parts, ignore_index=True)
    daily = full.groupby(['day', 'arm', 'shift'])[['pnl', 'pnl_without_young', 'young_shares', 'inventory_square_seconds', 'shares', 'pulls', 'places',
                                                'any_quote_steps', 'both_quote_steps', 'grid_steps']].sum()
    conditions = []; rng = np.random.default_rng(SEED)
    for baseline in ('TIMER', 'ALWAYS'):
        for shift in CFGS:
            a = daily.xs(('VALUE', int(shift)), level=('arm', 'shift')); b = daily.xs((baseline, int(shift)), level=('arm', 'shift'))
            for column in ('pnl', 'pnl_without_young'):
                diff = (a[column] - b[column]).to_numpy(); boots = rng.choice(diff, (1000, len(diff)), replace=True).mean(axis=1)
                conditions.append(dict(vs=baseline, shift=int(shift), metric=column, n_days=len(diff), mean_daily_delta=float(diff.mean()),
                                       positive_days=int((diff > 0).sum()), interval=np.quantile(boots, [.025, .975]).tolist(),
                                       pass_comparison=bool(diff.mean() > 0 and (diff > 0).sum() >= 4)))
    main = daily.xs(('VALUE', 50), level=('arm', 'shift'))
    absolute_positive = bool(main.pnl.sum() > 0 and main.pnl_without_young.sum() > 0)
    risk_ok = all(main.inventory_square_seconds.sum() <= daily.xs((b, 50), level=('arm', 'shift')).inventory_square_seconds.sum()
                  for b in ('TIMER', 'ALWAYS'))
    passed = absolute_positive and risk_ok and all(c['pass_comparison'] for c in conditions)
    daily.reset_index().to_parquet(stage / 'daily.parquet', index=False)
    write_json(stage / 'result.json', dict(**stamp(), asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               dates=list(FOLDS), development_accept=passed, absolute_positive_m50=absolute_positive,
               inventory_risk_not_increased_m50=risk_ok, comparisons=conditions,
               local_selection=local_null(pd.concat(predictions, ignore_index=True)), forward_validation=False))
    print(json.dumps(dict(development_accept=passed, absolute_positive_m50=absolute_positive,
                          inventory_risk_not_increased_m50=risk_ok, comparisons=conditions), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('selftest', 'rows', 'fit', 'eval', 'report'))
    p.add_argument('day', nargs='?'); p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--max-windows', type=int)
    a = p.parse_args()
    if a.command == 'selftest':
        import unittest
        import test_entry_value
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_entry_value))
        if not result.wasSuccessful():
            raise SystemExit(1)
    elif a.command == 'rows':
        rows_day(a.day, a.out, a.max_windows)
    elif a.command == 'fit':
        fit_fold(a.day, a.out)
    elif a.command == 'eval':
        eval_day(a.day, a.out)
    else:
        report(a.out)


if __name__ == '__main__':
    main()
