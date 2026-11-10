"""Pure research policies; score thresholds are not dollar-valued edge estimates."""
import math
from . import pf_late_engine as old

SOFT_DELTA = .01
CONCESSION_BUDGET = .25
RISK_SCORE_PER_625 = .01
NEW = ('soft_exit', 'budget_exit', 'marginal_entry')


def mapped(mode):
    return 'late_reserved' if mode in NEW else mode


def unique(opens):
    return list({id(o): o for o in opens}.values())


def risk_quantity(net, side, size, opens, value, line, exclude=None):
    """Largest proposed quantity within a fixed score premium for radius-squared growth."""
    if side not in (0, 1) or not all(math.isfinite(x) for x in (net, size, line)) or size < 0:
        raise ValueError('invalid risk state')
    if value is None or not math.isfinite(value):
        return 0.
    pending = [0., 0.]
    for o in unique(opens):
        if o is not exclude:
            if o['side'] not in (0, 1):
                raise ValueError('invalid reservation side')
            pending[o['side']] += old.prior.entry.remainder(o)
    radius = max(abs(net + pending[0]), abs(net - pending[1]))
    endpoint = (net if side == 0 else -net) + pending[side]
    premium = max(0., value - line) / RISK_SCORE_PER_625 * 625.
    return min(size, max(0., math.sqrt(radius * radius + premium) - endpoint))


def soft_flag(mode, rel, net, side, value, line, original):
    if mode not in ('soft_exit', 'budget_exit') or not 60 <= rel < 240:
        return original
    if (1 if side == 0 else -1) * net >= 0 or value is None:
        return original
    return original and not value > line - SOFT_DELTA


def price(net, side, p, bid, ask, rel, mode, state, current):
    if mode != 'budget_exit' or p is None or not 180 <= rel < 240 or abs(net) < 10:
        return p
    if (1 if side == 0 else -1) * net >= 0 or not 0 < bid < ask < 100:
        return p
    proposed = p + (1 if side == 0 else -1)
    if not bid < proposed < ask or not 1 <= proposed <= 99:
        return p
    # Existing paid-for order remains eligible when the submission budget is exhausted.
    if current is not None and current['pt'] == proposed:
        return proposed
    return proposed if state['committed_concession'] < CONCESSION_BUDGET - 1e-9 else p


def plan(net, wants, currents, opens, mode, rel, values, line):
    if mode != 'marginal_entry' or rel >= 60:
        return old.plan(net, wants, currents, opens, mapped(mode), rel)
    planned = []
    reserved = list(opens)
    for lane, (p, size) in enumerate(wants):
        side = lane % 2
        if p is None:
            planned.append((p, size)); continue
        cur = currents[lane]
        same = cur is not None and cur['pt'] == p
        remaining = old.prior.entry.remainder(cur) if same else 0.
        cap = risk_quantity(net, side, size, reserved, values[side], line, cur if same else None)
        if same and 1e-9 < remaining <= cap + 1e-9:
            planned.append((p, cur['size'])); continue
        q = old.parent.lots(risk_quantity(net, side, size, reserved, values[side], line))
        if not old.prior.entry.legal(p, side, q):
            planned.append((None, 0.)); continue
        planned.append((p, q))
        reserved.append(dict(side=side, size=q, kq=0.))
    return tuple(planned)


def admission(net, side, size, opens, mode, rel, value, line, p, meta, state):
    q = old.admission(net, side, size, opens, mapped(mode), rel)
    if mode == 'marginal_entry' and rel < 60:
        q = risk_quantity(net, side, q, opens, value, line)
    if mode == 'budget_exit':
        if meta is None or meta['price'] != p:
            raise ValueError('missing concession intent')
        tick_cost = (1 if side == 0 else -1) * (p - meta['base_price'])
        if tick_cost not in (0, 1):
            raise ValueError('invalid quoted concession')
        if tick_cost:
            q = min(q, max(0., (CONCESSION_BUDGET - state['committed_concession']) * 100))
    return q


def sent(mode, side, price_, size, meta, state):
    if mode != 'budget_exit':
        return
    cost = size * (1 if side == 0 else -1) * (price_ - meta['base_price']) / 100
    if cost < -1e-9 or state['committed_concession'] + cost > CONCESSION_BUDGET + 1e-7:
        raise ValueError('committed concession budget exceeded')
    state['committed_concession'] += cost
    # Conservative lifetime commitment: no refunds for cancellations or rejections.


def feasibility(rel, net, frame, values, line, sizes, btc, opens, wants):
    """All predicates on the unchanged reference decision path; no outcome fields."""
    if not math.isfinite(rel) or not math.isfinite(net):
        raise ValueError('invalid diagnostic state')
    if not 60 <= rel <= 300:
        return None
    side = 0 if net < 0 else 1
    row = dict(side=side if net else -1, net=net, tick=-1, target=0., room=0., actual_quote=False)
    flags = []
    if abs(net) <= 1e-9:
        flags.append('flat')
    if rel >= 240:
        flags.append('stop')
    if frame is None:
        flags.append('no_book')
    elif frame['b1_p'] < 0 or frame['a1_p'] < 0 or frame['a1_p'] <= frame['b1_p']:
        flags.append('invalid_book')
    else:
        bid, ask = frame['b1_p'], frame['a1_p']
        micro = (bid * frame['a1_s'] + ask * frame['b1_s']) / max(frame['b1_s'] + frame['a1_s'], 1e-12)
        half = (ask - bid) / 2
        p = math.floor(micro - half + 1e-9) if side == 0 else math.ceil(micro + half - 1e-9)
        touch = bid if side == 0 else ask
        target = min(sizes[side], abs(net))
        room = old.prior.room(net, side, opens, 'reduce_reserved') if net else 0.
        row.update(tick=p, target=target, room=room, actual_quote=wants[side][0] is not None)
        if not (1 <= p < ask if side == 0 else bid < p <= 99):
            flags.append('invalid_price')
        if (side == 0 and p < bid) or (side == 1 and p > ask):
            flags.append('off_touch')
        nxt = frame['b2_p'] if side == 0 else frame['a2_p']
        if p == touch and (nxt < 0 or abs(touch - nxt) >= 2):
            flags.append('book_gap')
        if btc[side]:
            flags.append('binance')
        if values[side] is None:
            flags.append('missing_score')
        elif values[side] <= line:
            flags.append('score_line')
        if (p if side == 0 else 100-p) * target / 100 < 1-1e-9:
            flags.append('minimum_order')
        if not old.prior.entry.legal(p, side, old.parent.lots(target)):
            flags.append('rounded_minimum')
        if target > room + 1e-9:
            flags.append('reserved_room')
        if not old.prior.entry.legal(p, side, old.parent.lots(min(target, room))):
            flags.append('room_below_minimum')
    # Reservations are an additional feasibility diagnostic, not baseline quote gates.
    baseline_blocks = [x for x in flags if x not in ('reserved_room', 'room_below_minimum', 'rounded_minimum')]
    if net and rel < 240 and bool(baseline_blocks) == row['actual_quote']:
        raise ValueError(f'diagnostic predicates disagree: row={row}, flags={flags}, values={values}, line={line}, frame={frame}')
    row['flags'] = tuple(flags)
    row['first_reason'] = baseline_blocks[0] if baseline_blocks else 'quoted'
    return row


class Episodes:
    """Compress repeated diagnostic states, retain every reference state and its duration."""
    def __init__(self):
        self.rows = []
        self.current = None

    def observe(self, mid, rel, state):
        if state is None:
            return
        key = (mid, tuple(state.items()))
        if self.current:
            if mid != self.current['mid'] or rel < self.current['end']:
                raise ValueError('diagnostic clock/market mismatch')
            self.current['end'] = rel
            if key == self.current['_key']:
                self.current['decisions'] += 1
                return
            self.rows.append({k: v for k, v in self.current.items() if k != '_key'})
        self.current = dict(mid=mid, start=rel, end=rel, decisions=1, **state, _key=key)

    def finish(self):
        if self.current:
            self.current['end'] = 300.
            self.rows.append({k: v for k, v in self.current.items() if k != '_key'})
            self.current = None
