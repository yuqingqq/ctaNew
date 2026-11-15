"""Wait/post state repair, with order-level target decomposition.

User design: preserve queue priority, decide entry from information, never a
fixed retry timer. This entry-only experiment retains neutral KEEP and the
frozen bootstrap entry continuation. It does NOT integrate the stop policy.

Three declared arms: BASE (causal old inputs), LADDER (same targets plus client
ladder information), COMPONENTS (same rich inputs; separate first-order and
rest-of-portfolio targets whose predictions sum to the same economic objective).
No filled-only training, label clipping, hyperparameter search or threshold fit.
All action generations and exclusions are retained. Known inclusion weights
correct the original 1/2 versus 1/32 sampling. The continuation's gap input is
masked until the start price is known; all labels are regenerated accordingly.

Seven consumed label days, five chronological test days, 500 matched draws.
Local gates are diagnostics, never portfolio/forward acceptance. Every new
source is frozen before historical runs. See the accompanying plan/review.
"""
import argparse
import bisect
import collections
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_wait_post_v2 as M
import pf_wait_post_state_lane as L
import pf_wait_post_sampling as S

E = M.E
BASE = S.BASE
DAYS = M.LATER
ARMS = ('BASE', 'LADDER', 'COMPONENTS')
SLOT_FIELDS = ('orders', 'remaining_lots', 'pending_place_lots', 'pending_cancel_lots',
               'filled_lots', 'price_gap_mean', 'price_gap_min', 'price_gap_max',
               'oldest_age_log', 'newest_age_log', 'active_age_log',
               'shown_lots', 'queue_lots', 'queue_unknown', 'empty_age_log', 'generation_log')
BOOK_FIELDS = ('liquidity_log', 'best_age_log', 'frame_age_log', 'imbalance',
               'size1_relative', 'size2_relative', 'size3_relative', 'size4_relative', 'size5_relative')
EXTRA = [f'{side}_{d}_{field}' for side in ('own', 'opp') for d in range(M.NR) for field in SLOT_FIELDS]
EXTRA += [f'{side}_book_{field}' for side in ('own', 'opp') for field in BOOK_FIELDS]
EXTRA += ['start_known', 'own_empty_count', 'opp_empty_count']
FEATURES = M.FEATURES + EXTRA
HEADS = {'BASE': ('qpost', 'delta'), 'LADDER': ('qpost', 'delta'),
         'COMPONENTS': ('local_post', 'external_post', 'local_delta', 'external_delta')}
KEYS = ['day', 'market', 'side', 'rung', 'generation']
DEFAULT_OUT = Path(M.K.IO) / 'wait_post_state_20261115_v1'


def write(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def safe_inputs(W, view, tracker):
    """Original causal features, with the previously unmasked start-price gap fixed."""
    x = M.inputs(W, view, tracker).copy()
    if W[view['side']]['vkl'][view['k']] == -1:
        x[M.FEATURES.index('gap')] = np.nan
    return x


def teacher(W, predictor, tracker, view):
    tracker.see(view)
    if predictor.models is None:
        return False
    try:
        x = safe_inputs(W, view, tracker)
    except ValueError:
        return False
    return M.post_rule(*predictor.values(view['side'], x))


def run(W, choose, observe=None, snapshot_filter=None, shift=50.):
    aux = {s: [False] * len(W[s]['tg']) for s in (0, 1)}
    r = L.sim_joint(W, 'M', shift, M.CAP, aux, size=M.SIZE, nr=M.NR,
                    mode='hard', policy=choose, observe=observe, neutral_keeper=True,
                    snapshot_filter=snapshot_filter)
    if any(r['stat']['viol'].values()):
        raise ValueError('engine accounting violation')
    E.terminal(W, r)
    return r


def validate_snapshot(v):
    if v.get('snapshot_time') != v['t']:
        raise ValueError('snapshot clock mismatch')
    slots = v.get('client_slots', ())
    if len(slots) != 2 * M.NR or {(s, d) for s, d, _, _ in slots} != {(s, d) for s in (0, 1) for d in range(M.NR)}:
        raise ValueError('incomplete client ladder')
    seen = set()
    remaining = [0., 0.]
    for s, d, p, ts, ta, rem, filled, tc in v['client_orders']:
        if (s, d, ts) in seen or s not in (0, 1) or d not in range(M.NR):
            raise ValueError('duplicate/invalid client order')
        seen.add((s, d, ts))
        if not np.isfinite([p, ts, ta, rem, filled]).all() or ts > v['t'] or ta < ts or (tc is not None and tc > v['t']):
            raise ValueError('noncausal client order')
        if not 1 <= p <= 99 or rem <= 0 or filled < 0 or abs(rem + filled - M.SIZE) > 1e-7:
            raise ValueError('invalid client quantity/price')
        remaining[s] += rem
    for s, name in ((0, 'exposure_bid'), (1, 'exposure_ask')):
        expected = remaining[s] + (1 if s == 0 else -1) * v['known_position']
        if abs(expected - v[name]) > 1e-7:
            raise ValueError('client exposure reconciliation failed')


class Features:
    """One instance per window. Cached public histories never include future frames."""
    def __init__(self, W):
        self.W = W
        self.levels = {}
        self.book_cache = {}
        self.audit_checks = 0

    def level(self, side, price):
        key = (side, price)
        if key not in self.levels:
            w = self.W[side]; p = w['P5']; last = p[:, 4]
            outside = (last >= 0) & ((price < last) if side == 0 else (price > last))
            self.levels[key] = np.where(outside, np.nan, (w['S5'] * (p == price)).sum(axis=1))
        return self.levels[key]

    def queue(self, side, price, activation, t):
        w = self.W[side]; known = w['kf']; jt = int(np.searchsorted(known, t, side='right')) - 1
        if jt < 0:
            return np.nan, np.nan
        lv = self.level(side, price); shown = float(lv[jt])
        if activation > t:
            return shown, np.nan
        ja = int(np.searchsorted(known, activation, side='right')) - 1
        if ja < 0 or not np.isfinite(lv[ja]):
            return shown, np.nan
        seg = lv[ja:jt + 1]; seg = seg[np.isfinite(seg)]
        return shown, float(seg.min()) if len(seg) else np.nan

    def book(self, side, t):
        w = self.W[side]; known = w['kf']; j = int(np.searchsorted(known, t, side='right')) - 1
        if j < 0:
            return [np.nan] * len(BOOK_FIELDS)
        cache_key = (side, j)
        if cache_key not in self.book_cache:
            start = int(np.searchsorted(known, known[j] - 30000., side='left'))
            both = np.r_[self.W[0]['S5'][start:j + 1, 0], self.W[1]['S5'][start:j + 1, 0]]
            both = both[np.isfinite(both) & (both > 0)]
            scale = float(np.median(both)) if len(both) else np.nan
            i = j
            while i > 0 and w['P5'][i - 1, 0] == w['P5'][j, 0] and w['P5'][i - 1, 0] >= 0:
                i -= 1
            own = float(w['S5'][j, 0]); opp = float(self.W[1 - side]['S5'][j, 0])
            imbalance = (own - opp) / (own + opp) if own + opp > 0 else np.nan
            ratios = (w['S5'][j] / scale).tolist() if scale > 0 else [np.nan] * 5
            self.book_cache[cache_key] = (scale, known[i], imbalance, ratios)
        scale, birth, imbalance, ratios = self.book_cache[cache_key]
        return [np.log1p(scale / M.SIZE), np.log1p((t - birth) / 50.),
                np.log1p((t - known[j]) / 50.), imbalance, *ratios]

    def vector(self, v, tracker):
        validate_snapshot(v)
        base = safe_inputs(self.W, v, tracker)
        slots = {(s, d): (g, age) for s, d, g, age in v['client_slots']}
        by_slot = collections.defaultdict(list)
        for o in v['client_orders']:
            by_slot[o[:2]].append(o)
        out = []; empty_counts = []
        for side in (v['side'], 1 - v['side']):
            sign = 1 if side == 0 else -1; empty = 0
            w = self.W[side]; j = int(np.searchsorted(w['kf'], v['t'], side='right')) - 1
            best = w['P5'][j, 0] if j >= 0 else np.nan
            for d in range(M.NR):
                orders = by_slot[side, d]; g, idle = slots[side, d]
                empty += int(not orders)
                if not orders:
                    vals = [0.] * 5 + [np.nan] * 6 + [0., 0., 0.]
                else:
                    qty = np.array([o[5] for o in orders]); gaps = np.array([sign * (best - o[2]) for o in orders])
                    ages = np.array([v['t'] - o[3] for o in orders]); active = [max(0., v['t'] - o[4]) for o in orders]
                    queues = [self.queue(side, o[2], o[4], v['t']) for o in orders]
                    shown = np.array([q[0] for q in queues]); ahead = np.array([q[1] for q in queues])
                    vals = [len(orders), qty.sum() / M.SIZE,
                            sum(o[5] for o in orders if o[4] > v['t']) / M.SIZE,
                            sum(o[5] for o in orders if o[7] is not None) / M.SIZE,
                            sum(o[6] for o in orders) / M.SIZE,
                            float(np.average(gaps, weights=qty)), float(gaps.min()), float(gaps.max()),
                            np.log1p(ages.max() / 50.), np.log1p(ages.min() / 50.), np.log1p(max(active) / 50.),
                            float(shown.sum() / M.SIZE) if np.isfinite(shown).all() else np.nan,
                            float(ahead.sum() / M.SIZE) if np.isfinite(ahead).all() else np.nan,
                            int((~np.isfinite(ahead)).sum())]
                vals += [np.log1p(idle / 50.) if not orders and idle is not None else 0., np.log1p(g)]
                out.extend(vals)
            empty_counts.append(empty)
        for side in (v['side'], 1 - v['side']):
            out.extend(self.book(side, v['t']))
        out.extend([float(self.W[v['side']]['vkl'][v['k']] != -1), *empty_counts])
        x = np.asarray([*base, *out], np.float32)
        if len(x) != len(FEATURES) or np.isinf(x).any():
            raise ValueError('invalid ladder feature vector')
        return x

    def audit(self, view, tracker, expected):
        """Real-row falsifier once per window, before any cached model prediction."""
        if self.audit_checks:
            return
        changed = copy.deepcopy(self.W)
        for side in (0, 1):
            w = changed[side]; after = np.asarray(w['kf']) > view['t']
            w['S5'][after] = w['S5'][after] * 3. + 11.
            w['P5'][after] = np.where(w['P5'][after] >= 0, 10, -1)
            w['X'][np.asarray(w['tg']) > view['t']] = 12345.
        np.testing.assert_equal(expected, Features(changed).vector(view, tracker))
        changed[view['side']]['X'][view['k'], 0] = 12345.
        bad = Features(changed).vector(view, tracker)
        if np.array_equal(expected, bad, equal_nan=True):
            raise ValueError('feature falsifier failed to detect known input change')
        self.audit_checks += 1


def fork(W, birth, delay, predictor, branch, features):
    if branch not in ('post', 'wait', 'never'):
        raise ValueError('unknown branch')
    tracker = M.Tracker(); born = [False]; probe = []; target = M.key(birth); started = [False]; vector = []
    def capture(v):
        return not started[0] and M.key(v) == target and v['t'] >= birth['t'] + delay and v['status'] == 'eligible'
    def choose(v):
        tracker.see(v)
        if not born[0]:
            if M.action_key(v) != M.action_key(birth):
                return True
            clean = {k: v[k] for k in birth}
            if clean != birth:
                raise ValueError('neutral birth differs')
            born[0] = True
        if not started[0]:
            if M.key(v) != target:
                return True
            if v['t'] < birth['t'] + delay:
                return False
            probe.append(dict(v)); vector.append(features.vector(v, tracker)); started[0] = True
            features.audit(v, tracker, vector[-1])
            return branch == 'post'
        if branch == 'never' and (v['side'], v['rung']) == (birth['side'], birth['rung']):
            return False
        return teacher(W, predictor, tracker, v)
    r = run(W, choose, snapshot_filter=capture)
    if not born[0]:
        raise ValueError('neutral birth not reached')
    return r, probe, vector


def focal_value(result, side, rung, generation, mark):
    """Unconditional value of the first order after the probe, zero if never posted.

    All subsequent orders and competing-rung effects belong to the residual.
    This is an economic decomposition, not an independent causal effect.
    """
    orders = result['order_states'][side, rung]
    if generation >= len(orders):
        return 0., 0.
    o = orders[generation]; sign = 1. if side == 0 else -1.
    qty = sum(q for _, q in o['fl'])
    return sign * (mark - o['pt']) * qty / M.SIZE, qty / M.SIZE


def paired(W, birth, delay, predictor, features):
    post, pv, px = fork(W, birth, delay, predictor, 'post', features)
    if not pv:
        return {'status': 'probe_not_reached'}, None
    wait, wv, wx = fork(W, birth, delay, predictor, 'wait', features)
    never, nv, nx = fork(W, birth, delay, predictor, 'never', features)
    if len(pv) != 1 or pv != wv or pv != nv:
        raise ValueError('counterfactual information differs')
    np.testing.assert_equal(px[0], wx[0]); np.testing.assert_equal(px[0], nx[0])
    target = M.action_key(pv[0]); prefixes = []
    for r, action in ((post, True), (wait, False), (never, False)):
        idx = next(i for i, a in enumerate(r['policy_decisions']) if tuple(a[:5]) == target)
        if bool(r['policy_decisions'][idx][-1]) != action:
            raise ValueError('counterfactual action differs')
        prefixes.append(r['policy_decisions'][:idx])
    if prefixes[0] != prefixes[1] or prefixes[0] != prefixes[2]:
        raise ValueError('counterfactual prefix differs')
    values = [E.terminal(W, r) for r in (post, wait, never)]
    if len({v['time'] for v in values}) != 1 or len({v['mark'] for v in values}) != 1:
        raise ValueError('terminal mismatch')
    qp = 100 * (values[0]['pnl'] - values[2]['pnl']) / M.SIZE
    qw = 100 * (values[1]['pnl'] - values[2]['pnl']) / M.SIZE
    lp, fp = focal_value(post, birth['side'], birth['rung'], birth['generation'], values[0]['mark'])
    lw, fw = focal_value(wait, birth['side'], birth['rung'], birth['generation'], values[0]['mark'])
    return dict(status='eligible', qpost=qp, qwait=qw, delta=qp - qw,
                local_post=lp, external_post=qp - lp, local_delta=lp - lw, external_delta=qp - qw - lp + lw,
                post_fill_fraction=fp, wait_fill_fraction=fw, probe_k=pv[0]['k'], probe_t=pv[0]['t'],
                snapshot=json.dumps({k: pv[0][k] for k in ('client_orders', 'client_slots', 'snapshot_time')}, separators=(',', ':'))), px[0]


def check_targets(frame):
    cols = ['qpost', 'qwait', 'delta', 'local_post', 'external_post', 'local_delta', 'external_delta']
    if not np.isfinite(frame[cols].to_numpy()).all():
        raise ValueError('nonfinite economic labels')
    for a, b, c in (('local_post', 'external_post', 'qpost'), ('local_delta', 'external_delta', 'delta'), ('qwait', 'delta', 'qpost')):
        if not np.allclose(frame[a] + frame[b], frame[c], atol=1e-8, rtol=0):
            raise ValueError('component reconciliation failed')


def rows(day, out, max_windows=None):
    if day not in DAYS:
        raise ValueError('undeclared day')
    stage = M.stage(out, 'rows_' + day); begin = time.monotonic()
    original = M.load_rows(BASE, 'learned', day)
    predictor = M.load_models(BASE / 'bootstrap')
    data = M.K.load_day(day); all_rows = []; windows = []; diagnostics = collections.Counter()
    sealed_windows = []
    if day == '20261031':
        boundary = pd.Timestamp('2026-10-31', tz='UTC').timestamp() * 1000. + 1800000.
        sealed_windows = [m for m, start in data['W0'].items() if start < boundary]
        for old in original[original.market.isin(sealed_windows)].to_dict('records'):
            all_rows.append({**{k: old[k] for k in KEYS + ['birth_t', 'requested_delay']}, 'status': 'sealed_lookback_window'})
        data['W0'] = {m: t for m, t in data['W0'].items() if m not in sealed_windows}
    for W in E.PAIRS.window_pairs(data):
        market = W[0]['m']; original_window = original[original.market == market]
        births = []
        run(W, lambda v: True, observe=lambda v, a: births.append(v) if v['status'] == 'eligible' else None)
        source = {tuple(r[k] for k in ('side', 'rung', 'generation')): r for r in original_window.to_dict('records')}
        if {M.key(b) for b in births} != set(source) or len(births) != len(source):
            raise ValueError('neutral action population changed')
        features = Features(W); predictor.clear()
        for b in births:
            old = source[M.key(b)]; mod = 2 if b['generation'] == 0 else 32
            h = M.hash_key(day, market, M.key(b)); sampled = h % mod == 0
            delay = M.AGES[(h // mod) % len(M.AGES)]
            if delay != old['requested_delay'] or sampled != (old['status'] != 'not_sampled'):
                raise ValueError('sampling identity changed')
            row = {k: old[k] for k in KEYS + ['birth_t', 'requested_delay']}
            row['status'] = 'not_sampled'
            if sampled:
                value, x = paired(W, b, delay, predictor, features); row.update(value)
                if x is not None:
                    row.update(zip(FEATURES, map(float, x)))
                    diagnostics['eligible'] += 1
                    diagnostics['changed_label_from_start_mask'] += int(abs(row['qpost'] - old['qpost']) > 1e-8 or abs(row['delta'] - old['delta']) > 1e-8)
            all_rows.append(row)
        windows.append(market)
        diagnostics['feature_future_checks'] += features.audit_checks
        if len(windows) % 8 == 0 or max_windows is not None:
            print(day, len(windows), 'windows', round(time.monotonic() - begin, 1), 's', round(E.rss_mib()), 'MiB', flush=True)
        if E.rss_mib() > 2048:
            raise RuntimeError('RSS self-guard exceeded')
        if max_windows is not None and len(windows) >= max_windows:
            break
    f = pd.DataFrame(all_rows); check_targets(f[f.status == 'eligible'])
    f.to_parquet(stage / 'rows.parquet', index=False)
    write(stage / 'receipt.json', dict(asof=M.now(), day=day, partial=max_windows is not None,
          windows=windows, excluded_windows=sorted(set(data['W0']) - set(windows)), sealed_lookback_windows=sealed_windows,
          source_status=data['stat'], statuses=f.status.value_counts().to_dict(), diagnostics=dict(diagnostics),
          rows_sha256=M.digest(stage / 'rows.parquet'), features=FEATURES,
          continuation='frozen_v2_bootstrap_with_start_gap_mask', continuation_receipt_sha256=M.digest(BASE / 'bootstrap/receipt.json'),
          original_rows_receipt_sha256=M.digest(BASE / M.row_stage('learned', day) / 'receipt.json'), seconds=time.monotonic() - begin))


def read_rows(out, day):
    p = out / ('rows_' + day); meta = E.read_json(p / 'receipt.json')
    if meta['partial'] or meta['excluded_windows'] or meta['features'] != FEATURES or meta['day'] != day:
        raise ValueError('partial/incompatible rows')
    if meta['rows_sha256'] != M.digest(p / 'rows.parquet'):
        raise ValueError('row digest differs')
    f = pd.read_parquet(p / 'rows.parquet')
    if f.duplicated(KEYS).any() or not (f.day == day).all():
        raise ValueError('action identities differ')
    f = f[f.status == 'eligible'].copy(); check_targets(f)
    return f


class Forecast:
    def __init__(self, arm, models):
        if arm not in ARMS:
            raise ValueError('unknown model arm')
        self.arm = arm; self.models = models
        self.names = M.FEATURES if arm == 'BASE' else FEATURES

    def predict(self, side, x):
        x = np.asarray(x, np.float32)
        if x.ndim != 2 or x.shape[1] != len(FEATURES):
            raise ValueError('feature contract differs')
        x = x[:, :len(self.names)]
        p = {h: np.asarray(m.predict(x), float) for h, m in self.models[side].items()}
        if self.arm == 'COMPONENTS':
            return p['local_post'] + p['external_post'], p['local_delta'] + p['external_delta']
        return p['qpost'], p['delta']


def policy(W, predictor):
    tracker = M.Tracker(); features = Features(W)
    def choose(v):
        tracker.see(v)
        x = features.vector(v, tracker)
        p, d = predictor.predict(v['side'], x[None, :])
        return bool(np.isfinite([p[0], d[0]]).all() and p[0] > 0 and d[0] >= 0)
    return choose


def load_forecast(out, test, arm):
    import lightgbm as lgb
    if test not in M.TESTS or arm not in ARMS:
        raise ValueError('undeclared forecast')
    path = out / ('fit_' + test); receipt = E.read_json(path / 'receipt.json')
    if receipt['test'] != test or receipt['train'] != list(M.FOLDS[test]):
        raise ValueError('forecast chronology differs')
    names = M.FEATURES if arm == 'BASE' else FEATURES
    models = {0: {}, 1: {}}
    for side in (0, 1):
        for head in HEADS[arm]:
            name = f'{arm}_{side}_{head}.txt'
            if M.digest(path / name) != receipt['models'][name]:
                raise ValueError('model digest differs')
            model = lgb.Booster(model_file=str(path / name))
            if model.feature_name() != names:
                raise ValueError('model feature schema differs')
            models[side][head] = model
    return Forecast(arm, models)


def run_model(W, predictor, shift=50., observe=None):
    return run(W, policy(W, predictor), observe=observe,
               snapshot_filter=lambda v: v['status'] == 'eligible', shift=shift)


def fit(test, out):
    import lightgbm as lgb
    if test not in M.TESTS or not all(d < test for d in M.FOLDS[test]):
        raise ValueError('undeclared/nonchronological fold')
    stage = M.stage(out, 'fit_' + test)
    train = pd.concat([read_rows(out, d) for d in M.FOLDS[test]], ignore_index=True)
    test_rows = read_rows(out, test); predictions = []; cells = []; health = []
    params = dict(E.PD.LGB_PARAMS, objective='regression')
    for side in (0, 1):
        tr = train[train.side == side]; te = test_rows[test_rows.side == side]
        weights = S.weights(tr); tw = S.weights(te)
        if len(tr) < 1000 or len(te) < 200 or S.effective(weights) < 200:
            raise ValueError('insufficient training/test action support')
        for arm in ARMS:
            names = M.FEATURES if arm == 'BASE' else FEATURES
            x = tr[names].to_numpy(np.float32); z = te[names].to_numpy(np.float32); heads = {}
            for target in HEADS[arm]:
                y = tr[target].to_numpy(float)
                model = lgb.train(params, lgb.Dataset(x, y, weight=weights / weights.mean(), feature_name=names), E.PD.N_ROUNDS)
                model.save_model(str(stage / f'{arm}_{side}_{target}.txt')); heads[target] = model
                health.append(dict(arm=arm, side=side, target=target, n=len(tr), effective_n=S.effective(weights),
                                   train=S.metric(y, model.predict(x), weights, np.average(y, weights=weights)),
                                   test=S.metric(te[target], model.predict(z), tw, np.average(y, weights=weights)),
                                   top_features=sorted(zip(names, map(float, model.feature_importance('gain'))), key=lambda v: -v[1])[:15]))
            predictor = Forecast(arm, {side: heads})
            pp, dd = predictor.predict(side, te[FEATURES].to_numpy(np.float32))
            local = te[KEYS + ['probe_t', 'price', 'qpost', 'qwait', 'delta']].copy()
            local['arm'] = arm; local['post_hat'] = pp; local['delta_hat'] = dd
            local['post_action'] = (pp > 0) & (dd >= 0); local['sampling_weight'] = tw; predictions.append(local)
            cells.append(dict(arm=arm, side=side, day=test, n=len(te), effective_n=S.effective(tw),
                              post=S.metric(te.qpost, pp, tw, np.average(tr.qpost, weights=weights)),
                              delta=S.metric(te.delta, dd, tw, np.average(tr.delta, weights=weights)),
                              baseline_post=S.metric(te.qpost, S.baseline(tr, te, 'qpost'), tw, np.average(tr.qpost, weights=weights)),
                              baseline_delta=S.metric(te.delta, S.baseline(tr, te, 'delta'), tw, np.average(tr.delta, weights=weights))))
    f = pd.concat(predictions, ignore_index=True); f.to_parquet(stage / 'predictions.parquet', index=False)
    write(stage / 'receipt.json', dict(asof=M.now(), test=test, train=list(M.FOLDS[test]), params=params,
          rounds=E.PD.N_ROUNDS, cells=cells, heads=health, predictions_sha256=M.digest(stage / 'predictions.parquet'),
          models={p.name: M.digest(p) for p in stage.glob('*.txt')},
          row_receipts={d: M.digest(out / ('rows_' + d) / 'receipt.json') for d in (*M.FOLDS[test], test)}))


def report(out):
    stage = M.stage(out, 'report'); cells = []; parts = []; receipts = {}
    for day in M.TESTS:
        p = out / ('fit_' + day); r = E.read_json(p / 'receipt.json')
        if r['predictions_sha256'] != M.digest(p / 'predictions.parquet'):
            raise ValueError('prediction identity differs')
        for name, digest in r['models'].items():
            if M.digest(p / name) != digest:
                raise ValueError('model identity differs')
        for row_day, digest in r['row_receipts'].items():
            if M.digest(out / ('rows_' + row_day) / 'receipt.json') != digest:
                raise ValueError('training/label identity differs')
        parts.append(pd.read_parquet(p / 'predictions.parquet')); cells.extend(r['cells'])
    all_predictions = pd.concat(parts, ignore_index=True)
    tests = [dict(arm=a, side=int(s), **S.matched(g)) for (a, s), g in all_predictions.groupby(['arm', 'side'])]
    pooled = []
    for arm in ARMS:
        heads = {}
        for h in ('post', 'delta'):
            cc = [c[h] for c in cells if c['arm'] == arm]
            heads[h] = dict(n=sum(c['n'] for c in cc), r2_vs_fold_prior_mean=1 - sum(c['sse'] for c in cc) / sum(c['baseline_sse'] for c in cc))
        pooled.append(dict(arm=arm, **heads))
    for day in DAYS:
        receipts[day] = E.read_json(out / ('rows_' + day) / 'receipt.json')
    write(stage / 'result.json', dict(asof=M.now(), status='complete_development_study', dates=list(M.TESTS),
          candidates=list(ARMS), new_candidate_count=3, forward_validation=False, portfolio_evaluated=False,
          n_test_actions=int(len(all_predictions) / len(ARMS)), cells=cells, pooled=pooled, matched_tests=tests,
          tests_passed=sum(t['passes'] for t in tests), row_receipts=receipts,
          limitations=['Labels retain neutral KEEP, not the separate stop model. No combined policy claim.',
                       'Same neutral action population: additional jointly waiting states are not generated by reweighting.',
                       'Order queue is a causal public-book estimate, not actual queue position.',
                       'Original frozen bootstrap continuation now masks the start-price gap; its weights were not retrained.',
                       'Components share one terminal mark. Local heads do not remove all long-horizon outcome noise.',
                       'Consumed development dates; later untouched-day validation is still required.',
                       'Model diagnostic only; C1/C2/C3/C5/C12/C15/C17 portfolio gates are not evaluated.']))


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['selftest', 'rows', 'fit', 'report'])
    p.add_argument('day', nargs='?'); p.add_argument('--out', type=Path, default=DEFAULT_OUT); p.add_argument('--max-windows', type=int)
    a = p.parse_args()
    if a.command == 'selftest':
        import unittest, test_wait_post_state
        r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_wait_post_state))
        if not r.wasSuccessful():
            raise SystemExit(1)
    elif a.command == 'rows': rows(a.day, a.out, a.max_windows)
    elif a.command == 'fit': fit(a.day, a.out)
    else: report(a.out)


if __name__ == '__main__':
    main()
