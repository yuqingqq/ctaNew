"""Research-only boundary: joint forecasts -> local economics -> one action.

No tape loading, fitting or exchange actions. See the dated pipeline handoff.
All prices/values are cents; quantities are shares; clocks are absolute ms.
The target is local, with its continuation and execution assumptions named.
"""
from dataclasses import dataclass
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(*values):
    return all(math.isfinite(x) for x in values)


@dataclass(frozen=True)
class Target:
    order_id: str
    side: str
    kind: str                    # post, or retain against a pull now
    decision_ms: float
    effective_ms: float          # activation for post; cancel landing for retain
    endpoint_ms: float           # common to both branches, never implicit t+H
    reference_mid: float         # causal reference; algebra is reference invariant
    price: float
    max_quantity: float          # client-known remaining, not latent true remaining
    scenario: str
    continuation: str
    execution_contract: str

    def validate(self):
        require(bool(self.order_id) and self.side in ('bid', 'ask'), 'order identity/side')
        require(self.kind in ('post', 'retain'), 'target kind')
        require(all((self.scenario, self.continuation, self.execution_contract)), 'unnamed target assumptions')
        require(finite(self.decision_ms, self.effective_ms, self.endpoint_ms,
                       self.reference_mid, self.price, self.max_quantity), 'nonfinite target')
        require(self.decision_ms <= self.effective_ms < self.endpoint_ms, 'target chronology')
        require(0 <= self.reference_mid <= 100 and 0 <= self.price <= 100
                and self.max_quantity > 0, 'target price/quantity bounds')

    @property
    def sign(self):
        return 1. if self.side == 'bid' else -1.


@dataclass(frozen=True)
class Atom:
    probability: float
    quantity: float
    signed_move: float           # side * (endpoint mid - reference mid)


@dataclass(frozen=True)
class Forecast:
    target: Target
    atoms: tuple[Atom, ...]
    feature_asof_ms: float
    trained_through_ms: float
    source: str                 # learned, synthetic, or realized_accounting
    model_id: str

    def validate(self, target):
        target.validate()
        require(self.target == target, 'forecast target mismatch')
        require(self.source in ('learned', 'synthetic', 'realized_accounting') and bool(self.model_id), 'forecast provenance')
        require(finite(self.feature_asof_ms, self.trained_through_ms)
                and self.feature_asof_ms <= target.decision_ms
                and self.trained_through_ms < target.decision_ms, 'forecast future information')
        require(len(self.atoms) > 0, 'empty distribution')
        require(all(finite(a.probability, a.quantity, a.signed_move)
                    and 0 <= a.probability <= 1 and 0 <= a.quantity <= target.max_quantity
                    and -1e-9 <= target.reference_mid + target.sign*a.signed_move <= 100+1e-9
                    for a in self.atoms), 'invalid joint distribution')
        require(abs(math.fsum(a.probability for a in self.atoms)-1.) <= 1e-10, 'probability mass')


@dataclass(frozen=True)
class Value:
    quantity: float
    edge: float
    movement: float
    retained: float
    gain: float
    ordinary_loss: float
    severe_excess: float
    fill_probability: float
    loss_probability: float
    severe_probability: float


def map_value(target, forecast, severe_cents=30.):
    """Exact joint expectation; cancellation orientation belongs to the controller."""
    forecast.validate(target)
    require(finite(severe_cents) and severe_cents > 0, 'severe threshold')
    edge_per_share = target.sign*(target.reference_mid-target.price)
    avg = lambda f: math.fsum(a.probability*f(a) for a in forecast.atoms)
    wealth = lambda a: a.quantity*(edge_per_share+a.signed_move)
    loss = lambda a: max(-wealth(a), 0.)
    q = avg(lambda a: a.quantity)
    edge = edge_per_share*q
    movement = avg(lambda a: a.quantity*a.signed_move)
    gain = avg(lambda a: max(wealth(a), 0.))
    ordinary = avg(lambda a: min(loss(a), severe_cents))
    severe = avg(lambda a: max(loss(a)-severe_cents, 0.))
    require(abs(edge+movement-(gain-ordinary-severe)) <= 1e-7, 'economic decomposition')
    return Value(q, edge, movement, edge+movement, gain, ordinary, severe,
                 avg(lambda a: a.quantity > 0), avg(lambda a: wealth(a) < 0),
                 avg(lambda a: wealth(a) < -severe_cents))


@dataclass(frozen=True)
class Economics:
    # All terms are ACTION minus baseline: post-wait or pull-keep, in cents.
    target: Target
    continuation_delta: float
    other_orders_delta: float
    inventory_cost_delta: float
    action_cost: float

    def validate(self, target):
        require(self.target == target, 'economic target mismatch')
        require(finite(self.continuation_delta, self.other_orders_delta,
                       self.inventory_cost_delta, self.action_cost)
                and self.action_cost >= 0, 'invalid economic adjustment')


@dataclass(frozen=True)
class ClientState:
    order_id: str
    side: str
    decision_ms: float
    lifecycle: str               # absent, live, post_pending, cancel_pending, closed
    remaining: float
    known_position: float
    reserved_bid: float          # includes live, post-pending AND cancel-pending
    reserved_ask: float
    cap: float

    def validate(self, target):
        require((self.order_id, self.side, self.decision_ms) ==
                (target.order_id, target.side, target.decision_ms), 'client/target identity')
        require(self.lifecycle in ('absent', 'live', 'post_pending', 'cancel_pending', 'closed'), 'client lifecycle')
        require(finite(self.remaining, self.known_position, self.reserved_bid,
                       self.reserved_ask, self.cap) and min(self.remaining, self.reserved_bid,
                       self.reserved_ask) >= 0 and self.cap > 0, 'client reservation bounds')
        if self.lifecycle in ('live', 'post_pending', 'cancel_pending'):
            reserve = self.reserved_bid if self.side == 'bid' else self.reserved_ask
            require(reserve >= self.remaining, 'focal order reservation missing')


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    advantage: float | None
    reserved_bid_after: float
    reserved_ask_after: float


def decide_one(target, forecast, state, economics, *, allow_synthetic=False):
    """One marginal action. Caller must revalue after any accepted action.

    No forecasts from realized labels may enter policy decisions. Cancellation
    does not free capacity; fills/acknowledgements must update the client ledger.
    """
    target.validate(); state.validate(target); economics.validate(target)
    bid, ask = state.reserved_bid, state.reserved_ask
    if state.lifecycle in ('post_pending', 'cancel_pending', 'closed'):
        return Decision('none', state.lifecycle, None, bid, ask)
    require((target.kind == 'post') == (state.lifecycle == 'absent'), 'action/lifecycle mismatch')
    if target.kind == 'retain':
        require(target.max_quantity == state.remaining, 'remaining quantity mismatch')
    fallback = 'wait' if target.kind == 'post' else 'keep'
    if forecast is None:
        return Decision(fallback, 'forecast_unavailable', None, bid, ask)
    require(forecast.source == 'learned' or (allow_synthetic and forecast.source == 'synthetic'), 'nonpredictive forecast refused')
    value = map_value(target, forecast)
    advantage = (value.retained if target.kind == 'post' else -value.retained)
    advantage += economics.continuation_delta + economics.other_orders_delta
    advantage -= economics.inventory_cost_delta + economics.action_cost
    if advantage <= 0:
        return Decision(fallback, 'nonpositive_advantage', advantage, bid, ask)
    if target.kind == 'post':
        nb = bid + (target.max_quantity if target.side == 'bid' else 0.)
        na = ask + (target.max_quantity if target.side == 'ask' else 0.)
        if state.known_position+nb > state.cap or -state.known_position+na > state.cap:
            return Decision('wait', 'reservation_cap', advantage, bid, ask)
        return Decision('post', 'positive_advantage', advantage, nb, na)
    return Decision('pull', 'positive_advantage', advantage, bid, ask)


def realized_atom(target, fill_times, fill_quantities, endpoint_mid):
    """Oracle accounting ONLY. Times already come from one re-simulated scenario.

    The simulator convention removes fills at the cancellation landing (>=).
    That is not evidence that a real match/cancel tie is orderable.
    """
    target.validate()
    require(len(fill_times) == len(fill_quantities), 'fill array lengths')
    require(finite(endpoint_mid) and 0 <= endpoint_mid <= 100, 'endpoint mark')
    require(all(finite(t, q) and q >= 0 and t <= target.endpoint_ms
                for t, q in zip(fill_times, fill_quantities)), 'invalid fill ledger')
    require(all(a <= b for a, b in zip(fill_times, fill_times[1:])), 'unsorted fill ledger')
    q = math.fsum(q for t, q in zip(fill_times, fill_quantities) if t >= target.effective_ms)
    require(q <= target.max_quantity+1e-9, 'affected quantity exceeds known remaining')
    return Atom(1., min(q, target.max_quantity), target.sign*(endpoint_mid-target.reference_mid))
