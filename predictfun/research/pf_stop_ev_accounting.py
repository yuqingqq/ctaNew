"""Chronological local KEEP-value labels from a corrected neutral replay.

This module never guesses executions from aggregate volume or price moves.
It reconciles the engine's actual per-order tranches to its per-leg and flat
ledgers, then values the focal order only. The continuation is neutral KEEP
subject to the frozen engine safety/end rules, with no replacement value in
the focal label. Policy re-evaluation every 50 ms is a separate operation.

Inputs and outcomes are separate dictionaries. True queue/remainder are never
model inputs. Missing timing/coverage/marks are statuses, not zero outcomes.
All public-feed execution assumptions remain assumptions, not live fill truth.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np

from pf_queue_activation import CONTRACT as QUEUE_CONTRACT

CONTRACT = 'stop-ev-chronological-order-value-v1'
REFERENCE = 'corrected_stop_neutral_no_model_no_gate_v1'
ORDER_FEATURES = ('side', 'price', 'rung', 'depth', 'age_ms', 'submitted_quantity',
                  'known_remaining', 'known_filled', 'queue_initial_known',
                  'queue_ahead_known', 'shown_known', 'queue_initial_unknown',
                  'effective_horizon_ms')


@dataclass(frozen=True)
class Settings:
    exposure_ms: float = 1000.
    markout_ms: float = 1000.
    cancel_ms: float = 22.
    notification_ms: float = 100.
    max_mark_age_ms: float = 1000.
    timing: str = 'ws_strict'

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.exposure_ms, self.markout_ms,
                   self.cancel_ms, self.notification_ms, self.max_mark_age_ms)):
            raise ValueError('nonfinite label timing contract')
        if not (0 <= self.cancel_ms < self.exposure_ms and self.markout_ms > 0
                and self.notification_ms >= 0 and self.max_mark_age_ms >= 0):
            raise ValueError('invalid label timing contract')
        if self.timing not in ('ws_strict', 'all_anchors_sensitivity'):
            raise ValueError('undeclared timing variant')


@dataclass(frozen=True)
class LegInfo:
    source_id: str
    timing: str                  # ws or e2; omitted/untimed legs go in Coverage
    uncertainty_ms: float = 0.


@dataclass(frozen=True)
class MissingLeg:
    source_id: str
    earliest_ms: float
    latest_ms: float
    reason: str


@dataclass(frozen=True)
class Coverage:
    trade_start_ms: float
    trade_end_ms: float
    mark_end_ms: float
    expected_legs: int
    book_start_ms: float
    book_end_ms: float
    missing: tuple = ()          # MissingLeg records, including unbounded ones
    book_gaps: tuple = ()        # MissingLeg-shaped intervals from loader checks


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-8)


def same_time(a, b):
    # Relative tolerance at epoch milliseconds would silently accept 100+ ms.
    return math.isclose(float(a), float(b), rel_tol=0., abs_tol=0.001)


def compare_fills(a, b, description, time_column):
    a, b = sorted(a), sorted(b)
    if len(a) != len(b) or any(len(x) != len(y) or any(
            not (same_time(i, j) if k == time_column else close(i, j)) for k, (i, j) in enumerate(zip(x, y)))
                               for x, y in zip(a, b)):
        raise ValueError(description + ' differs from execution ledger')


class Ledger:
    """Checked adapter for pf_stop_queue_activation_lane / corrected entry lane.

    submitted_quantity is known at creation, never reconstructed from future
    fills. leg_info must cover every raw timed leg; excluded raw legs retain
    identities in coverage.missing. The caller reconciles expected_legs with
    its source loader receipt before constructing this adapter.
    """
    def __init__(self, W, result, *, submitted_quantity, shift_ms, leg_info, coverage):
        self.W = W; self.result = result; self.size = float(submitted_quantity)
        self.coverage = coverage; self.leg_info = dict(leg_info)
        self.shift_ms = float(shift_ms)
        reference = result.get('local_reference', {})
        if reference.get('contract') != REFERENCE or reference.get('size') != self.size:
            raise ValueError('use replay_neutral; missing/mismatched neutral reference receipt')
        if reference.get('shift_ms') != self.shift_ms:
            raise ValueError('fill clock differs from reference receipt')
        if not math.isfinite(self.size) or self.size <= 0:
            raise ValueError('invalid submitted quantity')
        if any(result['stat']['viol'].values()):
            raise ValueError('execution conservation violation')
        self.ws = float(W[0]['ws'])
        self.end = float(result['terminal_time'])
        if not same_time(self.end, W[0]['te'] + reference['tail_ms']):
            raise ValueError('reference terminal boundary differs from engine')
        if not (math.isfinite(coverage.trade_start_ms) and math.isfinite(coverage.trade_end_ms)
                and math.isfinite(coverage.mark_end_ms) and coverage.trade_start_ms <= coverage.trade_end_ms):
            raise ValueError('invalid coverage interval')
        if not (math.isfinite(coverage.book_start_ms) and math.isfinite(coverage.book_end_ms)
                and coverage.book_start_ms <= coverage.book_end_ms):
            raise ValueError('invalid book coverage interval')
        self.orders = {}; self.fills = defaultdict(list); self.legs = {}
        for side in (0, 1):
            w = W[side]
            if not (len(w['anc']) == len(w['ptl']) == len(w['shl'])):
                raise ValueError('trade columns differ in length')
            for i, (t, p, q) in enumerate(zip(w['anc'], w['ptl'], w['shl'])):
                key = (side, i); t = float(t) - self.shift_ms
                if not (math.isfinite(t) and math.isfinite(q) and q > 0 and math.isfinite(p) and 1 <= p <= 99):
                    raise ValueError('untimed/nonpositive leg must be an exclusion status')
                self.legs[key] = (t, float(p), float(q))
        if set(self.legs) != set(self.leg_info):
            raise ValueError('trade timing provenance does not cover the tape')
        ids = [x.source_id for x in self.leg_info.values()] + [x.source_id for x in coverage.missing]
        if len(ids) != len(set(ids)) or any(not x for x in ids):
            raise ValueError('duplicate/missing source leg identity')
        if coverage.expected_legs != len(ids):
            raise ValueError('raw legs and exclusion statuses do not reconcile')
        for info in self.leg_info.values():
            if info.timing not in ('ws', 'e2') or not math.isfinite(info.uncertainty_ms) or info.uncertainty_ms < 0:
                raise ValueError('invalid trade timing provenance')
            if info.timing == 'e2' and info.uncertainty_ms <= 0:
                raise ValueError('estimated leg needs an uncertainty interval')
        for gap in (*coverage.missing, *coverage.book_gaps):
            if not gap.reason or math.isnan(gap.earliest_ms) or math.isnan(gap.latest_ms) or gap.earliest_ms > gap.latest_ms:
                raise ValueError('invalid missing-leg status')
        by_activation = {}; by_rung = defaultdict(list)
        for (side, rung), orders in result['order_states'].items():
            for generation, order in enumerate(orders):
                key = (side, rung, generation); self.orders[key] = order
                if not order['ref'] and order.get('queue_initialization') != QUEUE_CONTRACT:
                    raise ValueError('historical or unspecified queue initialization')
                identity = (side, rung, float(order['ta']))
                if identity in by_activation:
                    raise ValueError('ambiguous order activation identity')
                by_activation[identity] = key
                by_rung[side, rung].append((float(order['ta']), key))
                total = sum(float(q) for _, q in order['fl'])
                if not close(total + order['rem'], self.size) or total > self.size + 1e-8:
                    raise ValueError('order quantity does not reconcile')
        used = defaultdict(float); flat = []
        for (side, rung), fills in result['fills'].items():
            for fill in fills:
                i, price, qty, through, age = fill[:5]
                leg_key = (side, int(i))
                if leg_key not in self.legs or int(i) != i:
                    raise ValueError('unknown source leg')
                t, trade_price, available = self.legs[leg_key]
                identity = (side, rung, float(t - age))
                # The engine builds age by subtracting its activation clock;
                # tolerate floating subtraction only, never a nearby proxy.
                matches = ([by_activation[identity]] if identity in by_activation else
                           [key for ta, key in by_rung[side, rung] if same_time(ta, identity[2])])
                if len(matches) != 1:
                    raise ValueError('fill timestamp/age has no unique order')
                key = matches[0]; order = self.orders[key]
                expected_through = trade_price < price if side == 0 else trade_price > price
                if not (qty > 0 and close(price, order['pt']) and bool(through) == expected_through
                        and (through or trade_price == price) and order['ta'] <= t < order['tr']):
                    raise ValueError('invalid execution tranche')
                used[leg_key] += qty
                self.fills[key].append(dict(time=t, quantity=float(qty), price=float(price),
                                           leg=leg_key, through=bool(through)))
                flat.append((side, rung, t, price, qty))
        for key, qty in used.items():
            if qty > self.legs[key][2] + 1e-8:
                raise ValueError('shared taker volume counted more than once')
        for key, order in self.orders.items():
            compare_fills([(x['time'], x['quantity']) for x in self.fills[key]], order['fl'], 'per-order fills', 0)
            self.fills[key].sort(key=lambda x: (x['time'], x['leg'][1]))
        compare_fills(flat, result['fl'], 'flat fills', 2)
        signed = sum(q if side == 0 else -q for side, _, _, _, q in flat)
        if not close(signed, result['endpos']):
            raise ValueError('position does not reconcile to tranches')


def shown_at(w, frame, side, price):
    prices, sizes = np.asarray(w['P5'][frame]), np.asarray(w['S5'][frame])
    last = prices[-1]
    outside = last >= 0 and (price < last if side == 0 else price > last)
    return math.nan if outside else float(sizes[prices == price].sum())


def known_state(W, order, key, decision_ms, submitted_quantity, *, terminal_ms, settings=Settings()):
    """Causal order features; no actual remainder/ahead/future outcomes used."""
    side, rung, generation = key; t = float(decision_ms); w = W[side]
    clock = np.asarray(w['kf'], float)
    if not np.isfinite(clock).all() or np.any(np.diff(clock) < 0):
        raise ValueError('nonmonotone known frame clock')
    k = int(np.searchsorted(w['tg'], t))
    if k >= len(w['tg']) or not same_time(w['tg'][k], t):
        raise ValueError('decision is not on the declared engine grid')
    ja, jt = bisect_right(clock, order['ta']) - 1, bisect_right(clock, t) - 1
    filled = sum(float(q) for ft, q in order['fl'] if ft + settings.notification_ms <= t)
    remaining = float(submitted_quantity) - filled
    if remaining < -1e-8:
        raise ValueError('known fills exceed submitted size')
    initial = ahead = shown = math.nan
    if jt >= 0:
        shown = shown_at(w, jt, side, order['pt'])
    if ja >= 0 and ja <= jt:
        initial = shown_at(w, ja, side, order['pt'])
        ahead = initial if math.isfinite(initial) else math.inf
        for j in range(ja + 1, jt + 1):
            size = shown_at(w, j, side, order['pt'])
            if math.isfinite(size):
                ahead = min(ahead, size)
        if not math.isfinite(ahead): ahead = math.nan
    best = float(w['P5'][jt][0]) if jt >= 0 else math.nan
    return dict(side=side, price=float(order['pt']), rung=rung,
                depth=float(best - order['pt'] if side == 0 else order['pt'] - best),
                age_ms=t - order['ta'], submitted_quantity=float(submitted_quantity),
                known_remaining=max(0., remaining), known_filled=filled,
                queue_initial_known=initial, queue_ahead_known=ahead, shown_known=shown,
                queue_initial_unknown=float(not math.isfinite(initial)),
                effective_horizon_ms=min(settings.exposure_ms, terminal_ms - t))


def overlaps(a, b, lo, hi):
    return a <= hi and b >= lo


def valid_marks(w, side):
    """Use actual valid quote timestamps, never the timestamp of an ffilled mid."""
    tf = np.asarray(w['tf'], float)
    best = np.asarray(w['P5'], float)[:, 0]; opp = np.asarray(w['opp'], float)
    if len(tf) != len(best) or len(tf) != len(opp) or not np.isfinite(tf).all() or np.any(np.diff(tf) < 0):
        raise ValueError('invalid raw quote/mark series')
    bid, ask = (best, opp) if side == 0 else (opp, best)
    valid = np.isfinite(bid) & np.isfinite(ask) & (bid >= 0) & (ask <= 100) & (bid < ask)
    return tf[valid], (bid[valid] + ask[valid]) / 2.


def label_action(ledger, key, decision_ms, *, day, settings=Settings()):
    """One neutral order-generation action; no q-grid augmentation here.

    Fills through cancellation arrival are common exposure: they reduce the
    remainder but are excluded from avoidable value. Mark each later tranche
    at its own fill time + markout_ms. Unfilled valid actions receive zero.
    """
    if key not in ledger.orders:
        raise ValueError('unknown neutral action')
    order = ledger.orders[key]; side, rung, generation = key; t = float(decision_ms)
    ref = ledger.result['local_reference']
    if settings.cancel_ms != ref['cancel_ms'] or settings.notification_ms != ref['notification_ms']:
        raise ValueError('label latency differs from reference engine')
    market = int(ledger.W[side]['m'])
    end = min(t + settings.exposure_ms, ledger.end)
    row = dict(contract=CONTRACT, queue_initialization=QUEUE_CONTRACT,
               action_id=(str(day), market, side, rung, generation), decision_ms=t,
               hour=int(t // 3600000), timing=settings.timing, status='pending',
               continuation='neutral_keep_with_engine_safety_no_replacement_value',
               exposure_end_ms=end, reference=dict(ref), label_spec=asdict(settings), label=None, features=None)
    if t < order['ta'] or t >= ledger.W[side]['te']:
        row['status'] = 'probe_not_reached'; return row
    features = known_state(ledger.W, order, key, t, ledger.size, terminal_ms=ledger.end, settings=settings)
    row['features'] = features
    if order['ref'] or order['tc'] <= t or features['known_remaining'] <= 1e-8:
        row['status'] = 'not_client_resting'; return row
    c = ledger.coverage
    if c.trade_start_ms > ledger.ws or c.trade_end_ms < end:
        row['status'] = 'incomplete_trade_coverage'; return row
    if (c.book_start_ms > ledger.ws or c.book_end_ms < end or
            any(overlaps(g.earliest_ms, g.latest_ms, ledger.ws, end) for g in c.book_gaps)):
        row['status'] = 'incomplete_book_coverage'; return row
    w = ledger.W[side]; jt = bisect_right(w['kf'], t) - 1
    if jt < 0 or w['P5'][jt][0] < 0 or w['opp'][jt] < 0 or (
            w['P5'][jt][0] >= w['opp'][jt] if side == 0 else w['P5'][jt][0] <= w['opp'][jt]):
        row['status'] = 'invalid_known_book'; return row
    # Earlier uncertain events can alter the neutral path's queue, inventory
    # and births. Fail closed on the full relevant reference prefix, not just
    # on whether an excluded leg happened to fill this particular order.
    if any(overlaps(g.earliest_ms, g.latest_ms, ledger.ws, end) for g in c.missing):
        row['status'] = 'missing_trade_timing'; return row
    estimated = any(info.timing == 'e2' and overlaps(ledger.legs[k][0] - info.uncertainty_ms,
                     ledger.legs[k][0] + info.uncertainty_ms, ledger.ws, end)
                    for k, info in ledger.leg_info.items())
    if settings.timing == 'ws_strict' and estimated:
        row['status'] = 'ambiguous_reference_timing'; return row
    tf, mids = valid_marks(ledger.W[side], side)
    prefix_end = t + settings.cancel_ms
    past_qty = sum(x['quantity'] for x in ledger.fills[key] if x['time'] <= t)
    prefix_qty = sum(x['quantity'] for x in ledger.fills[key] if t < x['time'] < prefix_end)
    value = quantity = at_price = through_price = 0.; tranches = []
    sign = 1. if side == 0 else -1.
    for fill in ledger.fills[key]:
        if not (t < fill['time'] <= end and fill['time'] >= prefix_end):
            continue
        mark_t = fill['time'] + settings.markout_ms
        if c.mark_end_ms < mark_t:
            row['status'] = 'incomplete_mark_coverage'; return row
        j = bisect_right(tf, mark_t) - 1
        if (j < 0 or not math.isfinite(mids[j]) or not 0 <= mids[j] <= 100
                or mark_t - tf[j] > settings.max_mark_age_ms):
            row['status'] = 'missing_or_stale_mark'; return row
        if any(overlaps(g.earliest_ms, g.latest_ms, tf[j], mark_t) for g in c.book_gaps):
            row['status'] = 'missing_or_stale_mark'; return row
        v = fill['quantity'] * sign * (mids[j] - order['pt'])
        value += v; quantity += fill['quantity']
        if fill['through']: through_price += v
        else: at_price += v
        tranches.append(dict(**fill, mark_time=float(tf[j]), requested_mark_time=mark_t,
                             mark=float(mids[j]), value_cents=v))
    remaining_after_prefix = ledger.size - past_qty - prefix_qty
    if quantity > remaining_after_prefix + 1e-8 or not close(value, at_price + through_price):
        raise ValueError('local quantity/value conservation failed')
    row['status'] = 'eligible_estimated_timing' if estimated else 'eligible'
    row['label'] = dict(value_cents=value,
                        value_per_known_remaining_share=value / features['known_remaining'],
                        filled_quantity=quantity, past_quantity=past_qty, common_prefix_quantity=prefix_qty,
                        actual_remaining_after_prefix=remaining_after_prefix,
                        at_price_value_cents=at_price, through_price_value_cents=through_price,
                        tranches=tranches)
    return row


def generation_probes(ledger, *, day, ages_ms=(0., 250., 500., 1000., 2000.)):
    """One deterministic, outcome-independent probe per neutral generation.

    The caller freezes ages and inclusion design before reading outcomes.
    Unreached probes remain rows; no search for a favorable surviving state.
    """
    if not ages_ms or any(not math.isfinite(x) or x < 0 for x in ages_ms):
        raise ValueError('invalid probe age design')
    for key, order in ledger.orders.items():
        identity = (str(day), int(ledger.W[key[0]]['m']), *key)
        h = int.from_bytes(hashlib.sha256(repr(identity).encode()).digest()[:8], 'big')
        requested = order['ta'] + ages_ms[h % len(ages_ms)]
        grid = ledger.W[key[0]]['tg']; k = int(np.searchsorted(grid, requested))
        t = float(grid[k]) if k < len(grid) else float(ledger.W[key[0]]['te'])
        yield key, t


def replay_neutral(W, *, size, shift_ms, queue='M', unknown_queue='zero', nr=6,
                   cap=None, mode='none'):
    """The only reference entrypoint: no learned pulls, entry gate or retry timer.

    Retains the corrected STOP engine's cross/no-book/drift/end and declared
    cap rules. Those rules are part of this continuation, not learned actions.
    Call with a fresh W (engine caches are mutable); never patch a running job.
    """
    import pf_stop_queue_activation_lane as lane
    M = lane.M
    if queue not in ('M', 'L', 'U') or unknown_queue not in ('zero', 'inf'):
        raise ValueError('undeclared queue assumption')
    if nr < 1 or int(nr) != nr or not math.isfinite(size) or size <= 0:
        raise ValueError('invalid neutral ladder')
    if mode not in ('none', 'soft', 'hard') or (mode != 'none' and (
            cap is None or not math.isfinite(cap) or cap <= 0)):
        raise ValueError('invalid neutral cap contract')
    if mode == 'none' and cap is not None:
        raise ValueError('cap specified but disabled')
    if not math.isfinite(shift_ms) or M.MUT_LANE_NOGATE:
        raise ValueError('invalid clock or engine mutation enabled')
    old = M.QUNK
    try:
        M.QUNK = unknown_queue
        aux = {s: [False] * len(W[s]['tg']) for s in (0, 1)}
        r = lane.sim_joint_g(W, queue, shift_ms, cap, aux, size=size, nr=nr,
                             mode=mode, state='client', cooldown=0.)
    finally:
        M.QUNK = old
    r['local_reference'] = dict(contract=REFERENCE, size=float(size), shift_ms=float(shift_ms),
        queue=queue, unknown_queue=unknown_queue, nr=int(nr), cap=cap, mode=mode,
        cancel_ms=float(M.CL.K.L_CANCEL), notification_ms=float(M.CL.L_NOTE),
        tail_ms=float(M.CL.TAIL_MS), cooldown_ms=0.)
    return r


def build_actions(ledger, *, day, asof, ages_ms=(0., 250., 500., 1000., 2000.), settings=Settings()):
    """Build every sampled neutral generation, retaining all refusal statuses.

    The fixed age design is an instrument default, not a selected trade rule.
    Record it in the full pipeline freeze before historical data are read.
    """
    if not asof:
        raise ValueError('as-of required')
    rows = [label_action(ledger, key, t, day=day, settings=settings)
            for key, t in generation_probes(ledger, day=day, ages_ms=ages_ms)]
    receipt = dict(contract=CONTRACT, asof=asof, day=str(day), ages_ms=list(ages_ms),
                   n_generations=len(ledger.orders), n_actions=len(rows),
                   statuses=dict(Counter(r['status'] for r in rows)),
                   reference=dict(ledger.result['local_reference']), timing=settings.timing,
                   markout_ms=settings.markout_ms, exposure_ms=settings.exposure_ms,
                   max_mark_age_ms=settings.max_mark_age_ms,
                   raw_legs=ledger.coverage.expected_legs, missing_legs=len(ledger.coverage.missing),
                   limitation='Synthetic-checked instrument; raw provenance and model quality require a new data build.')
    return rows, receipt


def action_metrics(rows, keep, *, asof, draws=500, seed=20261115):
    """Local net value and matched null, never terminal portfolio P&L.

    Match day/side/hour/rung/known remainder and weight exactly; q copies or
    repeated steps of one generation are refused. This can have low power in
    sparse strata; no unmatched fallback or per-row confidence interval.
    """
    if not rows or len(rows) != len(keep) or draws < 200 or not asof:
        raise ValueError('invalid evaluation declaration')
    ids = [tuple(r['action_id']) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate neutral action in evaluation')
    if len({r['timing'] for r in rows}) > 1:
        raise ValueError('mixed timing populations')
    if any(r['reference'] != rows[0]['reference'] for r in rows):
        raise ValueError('mixed reference paths or size/clock assumptions')
    if any(r['label_spec'] != rows[0]['label_spec'] for r in rows):
        raise ValueError('mixed label horizons/latencies')
    if any(not isinstance(x, (bool, np.bool_)) for x in keep):
        raise ValueError('keep decisions must be boolean')
    if any(r['status'] not in ('eligible', 'eligible_estimated_timing') for r in rows):
        raise ValueError('invalid labels included in evaluation')
    y = np.array([r['label']['value_cents'] for r in rows], float)
    w = np.array([r.get('sampling_weight', 1.) for r in rows], float)
    selected = np.asarray(keep, bool)
    if not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('invalid value/weight')
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        day, market, side, rung, generation = r['action_id']
        groups[(day, side, r['hour'], rung, r['features']['known_remaining'], w[i])].append(i)
    rng = np.random.default_rng(seed); null = np.zeros(draws); varying = 0
    for ix in groups.values():
        ix = np.asarray(ix); n = int(selected[ix].sum())
        varying += int(0 < n < len(ix))
        if n:
            chosen = np.argsort(rng.random((draws, len(ix))), axis=1)[:, :n]
            null += (y[ix] * w[ix])[chosen].sum(axis=1)
    observed = float(np.sum(y[selected] * w[selected]))
    return dict(asof=asof, n_actions=len(rows), n_days=len({r['action_id'][0] for r in rows}),
                kept=int(selected.sum()), draws=draws, varying_strata=varying,
                local_kept_value_cents=observed, neutral_all_keep_value_cents=float(np.sum(y*w)),
                local_gain_vs_keep_all_cents=float(-np.sum(y[~selected]*w[~selected])),
                null_mean=float(null.mean()),
                p_value=float((1 + np.sum(null >= observed)) / (1 + draws)) if varying else None,
                interpretation='Matched local value only; no portfolio, interval or promotion claim. Multiplicity must be declared externally.')
