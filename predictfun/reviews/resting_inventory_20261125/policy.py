"""Review resting exposure with the same signed inventory budget used for admission."""
from pathlib import Path
import sys

PARENT = Path(__file__).resolve().parent.parent / 'leftover_inventory_20261125'
sys.path.insert(0, str(PARENT))
from controller_module import controller as R


def excess_orders(orders, room):
    """At most two desired levels. Minimize shares cancelled; prefer the oldest on a tie."""
    if len(orders) > 2:
        raise ValueError('more than two desired levels')
    remaining = {o['gid']: o['size'] - o['known_filled'] for o in orders}
    if sum(remaining.values()) <= room + R.EPS:
        return []
    feasible = [o for o in orders if remaining[o['gid']] <= room + R.EPS]
    if not feasible:
        return orders
    keep = max(feasible, key=lambda o: (remaining[o['gid']], -o['submit'], -o['gid']))
    return [o for o in orders if o['gid'] != keep['gid']]


class Policy(R.Policy):
    def __init__(self, branch, flags, model, enabled=True):
        super().__init__(branch, *flags, 'C', model)
        self.review_resting = enabled
        self.trims = []

    def review(self, g, s):
        b = self.b
        desired = set(b.desired(s))  # excludes requests already in flight, including pending-post cancels
        if not desired:
            return
        v = b.side_view(g, s)
        if not (v['book_ok'] and 0 <= v['bid'] < v['ask'] <= 100 and 0 <= g - v['received'] <= 1000):
            return
        pressure = self.on[s][b.k]
        orders = [o for o in v['orders'] if o['gid'] in desired
                  and not (o['state'] == 'live' and pressure)]  # inherited pressure PULL will remove these
        if not orders:
            return
        signed = (1 - 2*s) * v['position']
        capacity, _ = R.forecast(self.model, self.context(b.k, s))
        mid = (v['bid'] + v['ask']) / 200
        unit = mid if s == 0 else 1-mid
        room = max(0., min(self.Qside, self.cap-signed, capacity + 50/unit - signed))
        pulls = excess_orders(orders, room)
        if not pulls:
            return
        reserved = b.reserve(g)
        for o in pulls:
            gid = o['gid']
            assert b.E.o[gid]['req'] is None and gid in b.unclosed[s]
            b.E.request(gid, g, 'inventory_budget')
            b.decisions.append((g, s, o['state'], 'PULL', gid))
            b._log(g, s, o['state'], gid, o['rung'], o['price'], None, 'PULL', 'inventory_budget')
            self.trims.append((g, gid, o['state'], o['size']-o['known_filled'], room))
        assert b.reserve(g) == reserved
        b.check_position(g)


class Branch(R.Branch):
    def side_commands(self, g, s, pol):
        if pol.review_resting:
            pol.review(g, s)
        super().side_commands(g, s, pol)
