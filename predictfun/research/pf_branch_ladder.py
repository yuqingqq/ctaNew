"""THE MULTI-ORDER LADDER CLIENT branch_ladder_v1 (P-2026-005; 2026-11-21, session aea5bfb9). The current plan
(reviews/automatic/20261121_default_post_ladder/plan.md, section 5): the frozen client pf_branch_engine.Branch stores ONE open gid per side
(:238, :248, :265, :327) and adds 30-share orders (:139); "development must extend the client explicitly to per-order states". The engine
(pf_branch_engine.Engine: add9, the public trades applied ONCE to the whole portfolio in price / activation priority, rule M, removals and the
cancel latency) is reused unchanged; only the client is new. Times in ms from the episode start; prices in whole cents (tick 1: the registry's
decimal_precision 2, verified per market by the runner).
ORDERS: every generation keeps its own gid, price, birth rung (0 at the received best, 1 one tick behind; frozen at birth), quantity, queue
 (the engine's), notices and state: post_pending (before activation + POST_NOTICE 0), live, cancel_pending (a request, not yet acknowledged),
 closed (refused / full fill known / removal acknowledged = effect + 100 ms; known fill = match + 100 ms; the frozen rules per gid). At most
 `levels` unclosed generations per side, pending ones included.
RESERVATIONS: client-known position I (notified fills); B / A = the known remaining of EVERY unclosed bid / ask generation (post-pending,
 live, cancel-pending; unnotified fills stay reserved; released only on closure). Invariants after the notices and after EVERY command:
 I + B <= cap, -I + A <= cap and the engine position within [I - A, I + B].
TICK ORDER (deterministic): the engine's trades up to the tick; the notices of every unclosed gid (side 0 then 1, creation order); the
 invariants; the client structural rule per gid (the frozen rule: no book / more than 3 ticks + birth rung behind the received best / at or
 through the opposite best; every triggering subreason flagged, the primary one the first in that evaluation order, logged with the gid,
 rung, received frame, distance, mode, request, effect and acknowledgement); the inventory mode update (from I: >= S LONG_RESTRICTED, <= -S
 SHORT_RESTRICTED, |I| <= R UNRESTRICTED, else unchanged; transitions logged); then the sides in the frozen alternating order (first side =
 (episode + tick) % 2). Within a side: without a valid received book every actionable order / vacancy is the engine's fallback (KEEP /
 WAIT, logged); else each live gid (creation order) KEEP or PULL, then the vacant target prices from the best outward (best, best -/+ 1
 tick): a vacancy exists while fewer than `levels` generations are unclosed; a price held by an unclosed generation of the side is reserved
 (never a duplicate); a price outside [1, 99] is an unavailable slot (constraint unavailable_price, never clamped); with own_cross a new bid
 at or above / ask at or below any unclosed opposite-side order is own_cross_blocked; the policy then POSTs q whole shares (QMIN <= q <=
 q0) or WAITs; a POST needs the reservation room for q (constraint cap) and activation before END (constraint near_end). The view is
 rebuilt (reservations recomputed) before every command.
LEGACY_5S_Q30_C30 (levels 1, q0 30, cap 30, END 5 s, no mode, no own-cross rule) with the fixed POST / KEEP policy reproduces
 pf_branch_engine.Branch + POST_KEEP exactly: fills, decisions, requests, events, closures, constraints, fallbacks, value (selftest; the
 run's bounded reproduction on real 5-second paths). Tally counters are labels only and are not part of the parity.
IDENTITY: settings (contract branch_ladder_v1, configuration, q0, cap, levels, inventory, END, own_cross, QMIN, TICK and the frozen timing)
 and ident are bound; fingerprint() hashes every order, fill, event, cursor, the tape id and mark, the decisions, closures, unclosed gids,
 fallbacks, constraints, the mode and its log, the structural log, the settings, ident, tick and episode. save_state / resume refuse another
 configuration / END / tape or a changed state before any tick runs. Selftest: pf_ladder_rest_selftest.py."""
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_branch_engine as BE                             # noqa: E402  frozen: the engine, its timing and the frozen client (the parity reference)

C = BE.C
CONTRACT = 'branch_ladder_v1'
MARKET_END_MAX = 298000.
QMIN, TICK, TOL, EPS = 1., 1., 1e-9, 1e-12
CONFIGS = {'LEGACY_5S_Q30_C30': dict(end_ms=5000., q0=30., cap=30., levels=1, inventory=None, own_cross=False),
           'LADDER_REST_INVENTORY_v1': dict(end_ms=None, q0=100., cap=200., levels=2, inventory=dict(Qside=200., S=100., R=50., target=0.), own_cross=True)}
LEGACY, LADDER = 'LEGACY_5S_Q30_C30', 'LADDER_REST_INVENTORY_v1'
STRUCT_FLAGS = ('no_book', 'distance', 'cross')


def next_mode(old, I, S, R):
    """The persistent restriction mode after a notice update."""
    if I >= S - TOL:
        return 'LONG_RESTRICTED'
    if I <= -S + TOL:
        return 'SHORT_RESTRICTED'
    if abs(I) <= R + TOL:
        return 'UNRESTRICTED'
    return old


def book_ok_side(fk, s):
    """pf_branch_engine.book_ok on a received frame: a two-sided book with the side's best in [1, 99]."""
    return bool(0 <= fk[2] < fk[12] and 1 <= (fk[2] if s == 0 else fk[12]) <= 99)


class LadderEngine(BE.Engine):
    """The frozen engine (client mode) with the decided quantity and birth rung at submission."""

    def __init__(self, path, end):
        super().__init__(path, end, client=True)

    def add_order(self, side, price, d, q, rung):
        """A new generation of q shares at price, birth rung `rung`, decided and submitted at d: activation d + 58 ms (pf_branch_engine.Engine.add)."""
        return self.add9([side, float(rung), price, float(q), 0., d + BE.PLACE, 1e30, d, 1.])


class LadderBranch:
    """One branch of the multi-order client (header)."""

    def __init__(self, path, config, end=None, episode=0, ident=None):
        C.require(config in CONFIGS, f'an undeclared configuration {config}'); cf = CONFIGS[config]
        C.require(BE.CANCEL_ACK >= BE.FILL_NOTICE, 'a cancel ACK that can overtake fill notices needs a cumulative-fill watermark (not in this contract)')
        if cf['end_ms'] is not None:
            C.require(end is None or float(end) == cf['end_ms'], 'an END other than the configuration\'s'); end = cf['end_ms']
        C.require(end is not None and 0. < float(end) <= MARKET_END_MAX, 'a market configuration needs its path END in (0, 298,000] ms')
        self.config, self.q0, self.cap, self.levels = config, float(cf['q0']), float(cf['cap']), int(cf['levels'])
        self.inv = dict(cf['inventory']) if cf['inventory'] else None; self.own_cross = bool(cf['own_cross'])
        C.require(QMIN <= self.q0 and self.levels * self.q0 <= self.cap + TOL and self.levels in (1, 2), 'an inconsistent configuration')
        self.E = LadderEngine(path, float(end)); self.end = float(end); self.mark = BE.canon_mark(path.get('mark')); self.episode = int(episode)
        self.settings = dict(BE.SETTINGS, contract=CONTRACT, config=config, end_ms=self.end, q0=self.q0, cap=self.cap, levels=self.levels, inventory=self.inv,
                             own_cross=self.own_cross, qmin=QMIN, tick=TICK)
        self.ident = dict(ident or {}, config=config); self.unclosed = {0: [], 1: []}; self.closed = {}; self.decisions = []; self.fallbacks = []
        self.constraints = []; self.k = 0; self.acted = set(); self.mode = 'UNRESTRICTED' if self.inv else None; self.mode_log = []; self.struct_log = []
        self.cmd_log = []; self.counts = {}

    # ------------------------------------------------------------------ client-known state
    def known_filled(self, gid, g):
        return sum(q for t, i, _, q, _, _ in self.E.fills if i == gid and t + BE.FILL_NOTICE <= g)

    def gid_state(self, gid, g):
        """The state of one unclosed generation after the notices of g (no closure here)."""
        o = self.E.o[gid]
        return 'post_pending' if g < o['act'] + BE.POST_NOTICE else ('cancel_pending' if o['req'] is not None else 'live')

    def notices(self, g):
        """Close every generation whose refusal, full fill or removal acknowledgement is known by g (the frozen status rules, per gid)."""
        for s in (0, 1):
            for gid in list(self.unclosed[s]):
                o = self.E.o[gid]
                if g < o['act'] + BE.POST_NOTICE:
                    continue
                kf = self.known_filled(gid, g); ack = o['remove'] + BE.CANCEL_ACK if o['req'] is not None else np.inf
                why = 'refused' if o['refused'] else 'filled' if kf >= o['size'] - 1e-9 else 'removed' if ack <= g else None
                if why:
                    self.closed[gid] = (why, float(g)); self.unclosed[s].remove(gid); self.E.events.append((float(g), f'closed_{why}', gid, {}))

    def reserve(self, g):
        """(I, B, A): the client-known position and the known remaining of every unclosed bid / ask generation."""
        Ik = sum((1. if s == 0 else -1.) * q for t, _, s, q, _, _ in self.E.fills if t + BE.FILL_NOTICE <= g); res = {0: 0., 1: 0.}
        for s in (0, 1):
            for gid in self.unclosed[s]:
                res[s] += self.E.o[gid]['size'] - self.known_filled(gid, g)
        return Ik, res[0], res[1]

    def check_position(self, g):
        Ik, Bk, Ak = self.reserve(g); P = sum((1. if s == 0 else -1.) * q for t, _, s, q, _, _ in self.E.fills if t <= g)
        C.require(Ik + Bk <= self.cap + TOL and -Ik + Ak <= self.cap + TOL, f'reservation inequality at {g}: I {Ik}, B {Bk}, A {Ak}, cap {self.cap}')
        C.require(Ik - Ak - TOL <= P <= Ik + Bk + TOL, f'position check at {g}: engine {P} outside [{Ik - Ak}, {Ik + Bk}]')
        C.require(all(len(v) <= self.levels for v in self.unclosed.values()), f'more than {self.levels} unclosed generations on a side at {g}')

    def structural(self, g):
        """The frozen client structural rule per unclosed gid (header), every triggering subreason logged."""
        fk = self.E.f[self.E.known]
        for s in (0, 1):
            for gid in self.unclosed[s]:
                o = self.E.o[gid]
                if o['req'] is not None or g <= o['submit']:
                    continue
                best, opp = fk[2 + s * 10], fk[2 + (1 - s) * 10]
                behind = best - o['price'] if s == 0 else o['price'] - best
                cross = o['price'] >= opp if s == 0 else o['price'] <= opp
                flags = dict(no_book=bool(best < 0), distance=bool(behind > BE.HOLD + o['rung']), cross=bool(cross))
                if any(flags.values()):
                    self.E.request(gid, g, 'structural')
                    self.struct_log.append(dict(t=float(g), gid=int(gid), side=int(s), rung=float(o['rung']), price=float(o['price']), best=float(best), opp=float(opp),
                                                behind=float(behind), primary=next(f for f in STRUCT_FLAGS if flags[f]), mode=self.mode, effect=float(o['remove']),
                                                ack=float(o['remove'] + BE.CANCEL_ACK), received=float(fk[1]), frame=int(self.E.known), **flags))

    def update_mode(self, g):
        if not self.inv:
            return
        Ik = self.reserve(g)[0]; new = next_mode(self.mode, Ik, self.inv['S'], self.inv['R'])
        if new != self.mode:
            self.mode_log.append((float(g), float(Ik), self.mode, new)); self.mode = new

    def pre_tick(self, g):
        self.E.advance(g); self.acted = set(); self.notices(g); self.check_position(g); self.structural(g); self.update_mode(g)

    def side_view(self, g, s):
        """The client view of side s at g: position, reservations, mode, the received book, the side's unclosed orders, the opposite prices."""
        fk = self.E.f[self.E.known]; Ik, Bk, Ak = self.reserve(g)
        orders = [dict(gid=int(gid), price=float(self.E.o[gid]['price']), rung=float(self.E.o[gid]['rung']), size=float(self.E.o[gid]['size']), state=self.gid_state(gid, g),
                       known_filled=self.known_filled(gid, g), act=float(self.E.o[gid]['act']), submit=float(self.E.o[gid]['submit'])) for gid in self.unclosed[s]]
        return dict(t=float(g), side=int(s), end=self.end, position=Ik, B=Bk, A=Ak, cap=self.cap, q0=self.q0, mode=self.mode, bid=float(fk[2]), ask=float(fk[12]),
                    bid_size=float(fk[7]), ask_size=float(fk[17]), received=float(fk[1]), orders=orders, book_ok=book_ok_side(fk, s),
                    opp_prices=[float(self.E.o[gid]['price']) for gid in self.unclosed[1 - s]], vacancy=self.levels - len(self.unclosed[s]))

    # ------------------------------------------------------------------ commands
    def _log(self, g, s, kind, gid, rung, price, q, action, reason):
        self.cmd_log.append((float(g), int(s), kind, -1 if gid is None else int(gid), -1. if rung is None else float(rung), np.nan if price is None else float(price),
                             np.nan if q is None else float(q), action, reason))

    def side_commands(self, g, s, pol):
        """The commands of side s at g (header)."""
        C.require(s not in self.acted, f'side {s} already processed at {g}'); self.acted.add(s); v = self.side_view(g, s)
        if not v['book_ok']:
            for o in v['orders']:
                if o['state'] == 'live':
                    self.fallbacks.append((float(g), s, 'live', 'feature_unavailable')); self.decisions.append((float(g), s, 'live', 'KEEP', o['gid']))
                    self._log(g, s, 'live', o['gid'], o['rung'], o['price'], None, 'KEEP', 'fallback')
            if v['vacancy'] > 0:
                self.fallbacks.append((float(g), s, 'absent', 'feature_unavailable')); self.decisions.append((float(g), s, 'absent', 'WAIT', None))
                self._log(g, s, 'slot', None, None, None, None, 'WAIT', 'fallback')
            self.check_position(g); return
        for o in [x for x in v['orders'] if x['state'] == 'live']:
            a, why = pol.live(self.side_view(g, s), o); C.require(a in ('KEEP', 'PULL'), f'a live action {a}')
            if a == 'PULL':
                self.E.request(o['gid'], g, 'policy')
            self.decisions.append((float(g), s, 'live', a, o['gid'])); self._log(g, s, 'live', o['gid'], o['rung'], o['price'], None, a, why); self.check_position(g)
        best = v['bid'] if s == 0 else v['ask']
        for r in range(self.levels):
            v = self.side_view(g, s)
            if v['vacancy'] <= 0:
                break
            price = best - TICK * r if s == 0 else best + TICK * r
            if any(o['price'] == price for o in v['orders']):
                continue
            if not 1 <= price <= 99:
                self.constraints.append((float(g), s, 'unavailable_price')); self._log(g, s, 'slot', None, r, price, None, 'WAIT', 'unavailable_price'); continue
            if self.own_cross and any((price >= p_ if s == 0 else price <= p_) for p_ in v['opp_prices']):
                self.constraints.append((float(g), s, 'own_cross_blocked')); self._log(g, s, 'slot', None, r, price, None, 'WAIT', 'own_cross_blocked'); continue
            a, q, why = pol.slot(v, r, price); gid = None
            if a == 'POST':
                C.require(QMIN <= q <= self.q0 and q == math.floor(q), f'a POST quantity {q} outside the whole shares [{QMIN}, {self.q0}]')
                room = (self.cap - v['position'] - v['B']) if s == 0 else (self.cap + v['position'] - v['A'])
                if q > room + TOL:
                    self.constraints.append((float(g), s, 'cap')); a, why = 'WAIT', 'cap'
                elif g + BE.PLACE >= self.end:
                    self.constraints.append((float(g), s, 'near_end')); a, why = 'WAIT', 'near_end'
                else:
                    gid = self.E.add_order(s, price, g, q, r); self.unclosed[s].append(gid)
            else:
                C.require(a == 'WAIT', f'a slot action {a}')
            self.decisions.append((float(g), s, 'absent', a, gid)); self._log(g, s, 'slot', gid, r, price, q, a, why); self.check_position(g)


def step(B, k, pol):
    """One grid tick k: pre_tick, then both sides in the frozen alternating order."""
    g = k * BE.STEP; B.pre_tick(g); B.k = k
    for s in ((B.episode + k) % 2, 1 - (B.episode + k) % 2):
        B.side_commands(g, s, pol)
    B.check_position(g)


def n_ticks(B):
    return int(np.ceil(B.end / BE.STEP))


def run_episode(B, pol, k0=0, until=None, observe=None):
    """Ticks k0 .. until - 1 (None: the branch's last tick); observe(B, k) after each tick (a read)."""
    C.require(isinstance(B, LadderBranch), 'not a ladder branch'); C.require(getattr(pol, 'horizon', None) == B.end, 'the policy does not declare the branch END')
    for k in range(int(k0), n_ticks(B) if until is None else int(until)):
        step(B, k, pol)
        if observe is not None:
            observe(B, k)
    return B


def finish(B):
    """The trades up to END (inclusive); the notices at END; open exposure disclosed per side and gid; the ledger."""
    B.E.advance(B.end); B.notices(B.end - EPS)
    if B.mark is None or not np.isfinite(B.mark):
        return dict(status='missing_mark', fills=list(B.E.fills))
    cash, pos, val = B.E.value(B.mark)
    og = {int(o['gid']): float(o['rem']) for o in B.E.o if (not o['refused']) and o['remove'] > B.end and o['rem'] > TOL}
    side = {s: sum(v for g_, v in og.items() if B.E.o[g_]['side'] == s) for s in (0, 1)}
    return dict(status='ok', cash=cash, position=pos, value=val, fills=list(B.E.fills), open_at_end=side, open_at_end_gid=og, generations=len(B.E.o),
                fallbacks=list(B.fallbacks), constraints=list(B.constraints), decisions=list(B.decisions), requests=[e_ for e_ in B.E.events if e_[1].endswith('_request')])


class Policy:
    """A frozen causal ladder policy: live(view, order) -> (KEEP / PULL, reason); slot(view, rung, price) -> (POST / WAIT, q, reason)."""

    def __init__(self, name, horizon):
        self.name, self.horizon = name, float(horizon)


class FixedPostKeep(Policy):
    """The legacy fixed policy: KEEP every live order, POST q0 at every considered slot (the engine's constraints decide), as POST_KEEP."""

    def __init__(self, B):
        super().__init__('post_keep_fixed', B.end); self.q0 = B.q0

    def live(self, v, o):
        return 'KEEP', 'keep'

    def slot(self, v, r, price):
        return 'POST', self.q0, 'post'


class LadderInventory(Policy):
    """LADDER_REST_INVENTORY_v1 (plan section 2): side multipliers f_bid = clip(1 - I / S, 0, 1), f_ask = clip(1 + I / S, 0, 1); desired side
    exposure D = Qside f; for each eligible vacant level q = floor(min(q0 f, max(0, D - B_side), max(0, room)) + 1e-9), room = cap - I - B (bid)
    / cap + I - A (ask); below QMIN -> WAIT (zero_desired: q0 f < QMIN; exposure_full: D - B_side < QMIN; else no_room). LONG_RESTRICTED blocks
    new bids and PULLs every live bid (inventory pull); SHORT_RESTRICTED symmetric; the reducing side and every other live order is kept."""

    def __init__(self, B):
        C.require(B.inv is not None, 'the ladder policy needs the inventory configuration'); super().__init__(LADDER, B.end)
        self.q0, self.cap, self.Qside, self.S = B.q0, B.cap, B.inv['Qside'], B.inv['S']

    @staticmethod
    def blocked(v):
        return (v['side'] == 0 and v['mode'] == 'LONG_RESTRICTED') or (v['side'] == 1 and v['mode'] == 'SHORT_RESTRICTED')

    def live(self, v, o):
        return ('PULL', 'inventory_pull') if self.blocked(v) else ('KEEP', 'keep')

    def sizing(self, v):
        s, I = v['side'], float(v['position']); f = min(1., max(0., (1. - I / self.S) if s == 0 else (1. + I / self.S))); D = self.Qside * f
        Bs = v['B'] if s == 0 else v['A']; room = (self.cap - I - v['B']) if s == 0 else (self.cap + I - v['A'])
        return f, D, Bs, room, float(math.floor(min(self.q0 * f, max(0., D - Bs), max(0., room)) + TOL))

    def slot(self, v, r, price):
        if self.blocked(v):
            return 'WAIT', None, 'restricted'
        f, D, Bs, room, q = self.sizing(v)
        if q < QMIN:
            return 'WAIT', q, ('zero_desired' if self.q0 * f < QMIN - TOL else 'exposure_full' if D - Bs < QMIN - TOL else 'no_room')
        return 'POST', q, 'post'


def fingerprint(B):
    """The ladder branch's complete state and identity (header)."""
    return hashlib.sha256(json.dumps([B.E.o, B.E.fills, B.E.events, B.E.it, B.E.known, B.E.end, B.E.client, B.E.nf, BE.tape_id(B.E.f, B.E.tr, B.mark), B.decisions,
                                      sorted(B.closed.items()), B.unclosed, B.fallbacks, B.constraints, B.end, B.settings, B.ident, B.k, B.episode, sorted(B.acted),
                                      B.mode, B.mode_log, B.struct_log, B.cmd_log], sort_keys=True, default=str).encode()).hexdigest()


def save_state(B, next_k):
    C.require(isinstance(B, LadderBranch), 'not a ladder branch')
    return dict(contract=CONTRACT, config=B.config, end=B.end, settings=dict(B.settings), ident=dict(B.ident), tape_id=BE.tape_id(B.E.f, B.E.tr, B.mark), mark=B.mark,
                mode=B.mode, unclosed={str(s): list(v) for s, v in B.unclosed.items()}, next_k=int(next_k), fingerprint=fingerprint(B), blob=pickle.dumps(B))


def resume(saved, config, path, end):
    """The saved branch for the same configuration, END and tape -> (branch, next tick); every disagreement refused before any tick runs."""
    C.require(isinstance(saved, dict) and saved.get('contract') == CONTRACT, 'not a saved branch_ladder_v1 state')
    C.require(saved['config'] == config and config in CONFIGS and float(saved['end']) == float(end), 'a saved state of another configuration / END')
    C.require(BE.tape_id(path['frames'], path['trades'], path.get('mark')) == saved['tape_id'], 'a tape other than the saved state\'s')
    B = pickle.loads(saved['blob']); cf = CONFIGS[config]
    C.require(isinstance(B, LadderBranch) and fingerprint(B) == saved['fingerprint'], 'the saved branch changed after it was saved (its fingerprint)')
    want = dict(BE.SETTINGS, contract=CONTRACT, config=config, end_ms=float(end), q0=cf['q0'], cap=cf['cap'], levels=cf['levels'], inventory=cf['inventory'],
                own_cross=cf['own_cross'], qmin=QMIN, tick=TICK)
    C.require(B.config == config and B.end == float(end) == B.E.end and B.settings == saved['settings'] == want and B.mode == saved['mode']
              and {str(s): list(v) for s, v in B.unclosed.items()} == saved['unclosed'] and B.E.client is True, 'a saved branch whose fields disagree with its identity')
    return B, int(saved['next_k'])
