"""THE CAUSAL BRANCH ENGINE AND ITS VALUE CONTRACT paired_continuation_v2 (plan v2 stage R1; P-2026-005; 2026-11-19, session aea5bfb9).
USER 2026-11-19: "Check my latest commit, next, can you ask codex research session for next priority. continue the implementing and next
priority loop"; review 92d8e52f's next priority: "Develop R1: an executable 50 ms WAIT/re-entry branch engine with common END. The first
reviewable deliverable is code plus synthetic event/ledger traces - not a new model fit." No model is fitted or read here.

ENGINE (one branch of one episode; times in ms from the episode start; one common END = 5,000 ms for every generation and both branches):
 the conserved-volume kernel's rules (pf_flow_control_kernel.cpp, unchanged) ported so that every event carries its own timestamp:
 - an order is [side, rung, price, remaining, queue ahead, activation, removal, submit, new]; a new order's queue ahead is the shown size at its
   price in the last frame with venue time <= its activation; an opposite best at or through its price at activation refuses it (post-only);
 - each public trade is applied ONCE to the whole portfolio in price / activation priority: a trade at the order's price consumes the queue
   ahead first, a trade through it fills up to the remainder; the queue ahead only shrinks to later shown sizes with venue time before the trade
   (rule M); an order fills from its activation (inclusive) to its removal (exclusive); trades at END count;
 - on the 50 ms grid, after the trades up to the grid and with the frames received by then, the structural rule (no book, more than 3 ticks
   + rung behind the best, or at / through the opposite best) requests a removal that lands after the cancel latency (37 ms; before activation
   max(request + 37, submit + 95));
 - TWO PATHS for that rule. Without a client (batch, the C-compatibility path): the kernel's form, skipping orders whose engine remainder is 0,
   applied after an order's submission tick (KERNEL_MODE: the kernel exactly, for P1). With a client (Branch): the CLIENT issues it on
   client-known state - every open generation without a request, after its submission tick - including an order the engine has already filled
   but the client has not heard of (no execution effect; review 043ca403 P1: hidden fills must not change client state). Single-generation fills
   and values are the same on both paths; client traces differ where the kernel's form would leak a hidden fill.
CLIENT (decides on the grid g = 0, 50, ..., < END from client-known state only):
 - named timing fields (review 043ca403 (a); simulation assumptions, not exchange measurements): activation 58 ms after the submission;
   POST_NOTICE 0 ms (acceptance / post-only refusal known at activation); FILL_NOTICE 100 ms after a match; cancel effect 37 ms after a request
   (pre-activation max(c + 37, d + 95)); CANCEL_ACK 100 ms after the effect (CANCEL_ACK >= FILL_NOTICE is required: every pre-removal fill is
   known before the ACK); a generation closes when its full fill is known, its removal acknowledged, or its refusal known;
 - the tie order at a tick: the trades up to the tick; the notices due by it (statuses); the reservation / position invariants; the client's
   structural requests; then the policy decisions in the alternating side order, the invariants checked after every command;
 - per side at most one open generation; states absent / post_pending / live / cancel_pending / terminal (at END); POST or WAIT when absent; KEEP
   or PULL when live; no action in a pending state;
 - reservations (plan v2 section 2): client-known position I; bid / ask exposure B / A = the known remaining of every open bid / ask generation
   (unacknowledged fills stay reserved; released on the acknowledgement, never on a request); a 30-share POST needs I + B + 30 <= 30 (bid) or
   -I + A + 30 <= 30 (ask); an independent check holds the engine position within [I - A, I + B] at every tick;
 - a POST joins the best of its side in the last frame received by the decision (30 shares, rung 0) and needs a two-sided book with the best in
   [1, 99], room under the cap and activation before END; an infeasible POST is a logged constraint (cap / near_end), never a fill;
 - sides are processed in a deterministic alternating order (first side = (episode + tick index) % 2); the second side's view is rebuilt after
   the first side's command;
 - a policy is a frozen causal function of the client view (time, side, both sides' states, generation, known fills, position, reservation room,
   the last received book); a view without a valid book is a fallback (WAIT / KEEP) with its reason, counted; the policy declares its horizon
   (= END) and is refused otherwise.
VALUE CONTRACT paired_continuation_v2 (plan v2 section 4): at a fork tick k and side s the same branch state is copied into two branches;
 Y_post = R_T(POST now, then the policy) - R_T(WAIT this tick, then the policy); Y_keep = R_T(KEEP this tick, then the policy) - R_T(PULL now,
 then the policy); R_T = cash + inventory x the END mark over every generation and both sides of the branch; the common pre-fork fills are
 checked identical and cancel; each branch keeps its fills (time, generation, side, shares, price), cash, inventory, open exposure at END
 (disclosed: never cancelled or liquidated for free) and its event trace. A missing END mark censors the pair (status missing_mark), never 0.
CONTROLS (selftest on synthetic tapes; plan v2 section 4's table; each a valid case and a known-bad one it must reject): P1 numerical
 AGGREGATE parity (KERNEL_MODE against the C library: per-order shares, cash, position and END value within P1_TOL - not event parity, the C
 interface returns no fill times; plus own-order price priority); P2 incremental = batch (an adaptive run replayed from its final order list);
 F1 actual WAIT; F2 re-entry; F3 queue preservation; F4 economic identity; F5 continuation matters (both directions, from ONE saved fork state,
 only the continuation varied); F6 causality; F7 boundaries and capacity (incl. the reservation inequality and released reservations);
 F8 identity / support; F9 shared book; F10 unnoticed fills (zero / partial / full, both sides, a pending-sensitive other side) cannot change
 views, structural requests or reservations before their notice. Mutations: pf_branch_engine_mutants.py.
SLICE (bounded, consumed day; slice <day> <n>): the first n eligible opportunities of the day, the engine under two degenerate policies (POST at
 0 then KEEP; POST at 0 then PULL at the first review) against the committed keep / pull run's M50 fills of the same orders (exact), plus P1 on
 every slice tape.
 The keep / pull run records one fill row per distinct trade time; the engine one per public trade: fills are compared summed per time
 (fills_equal; slice v1 compared rows and failed on 47 + 1 orders whose fills are all equal per time - kept as provenance).
SLICE v3 (review 043ca403 (e); slice_v3 <day>): the first 128 eligible archived opportunities per side by oid - see slice_v3.
Usage (run_light, PF_SLOT=claude): pf_branch_engine.py selftest | traces | slice <day> <n> [tag] | slice_v3 <day>
"""
import copy
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_keep_pull as KP                                 # noqa: E402
import pf_keep_pull_scenarios as SC                       # noqa: E402
import pf_flow_control_kernel as FK                       # noqa: E402
import pf_queue_place_test as QP                          # noqa: E402

C, pd, I = KP.C, KP.pd, KP.I
CONTRACT = 'paired_continuation_v2'
PLACE, CANCEL, PENDING, STEP, END, SIZE, HOLD = KP.PLACE, KP.CANCEL, KP.PENDING, KP.STEP, KP.H, KP.SIZE, 3.
POST_NOTICE = 0.                                           # review 043ca403 (a): post acceptance / refusal known at activation + 0 ms (assumption)
FILL_NOTICE = KP.NOTICE                                    # a fill is known 100 ms after its match
CANCEL_ACK = KP.NOTICE                                     # a removal's effect is acknowledged 100 ms after it (assumption; not a measurement)
CAP = 30.                                                  # plan v2: absolute position cap per instrument (shares)
EPS = 1e-12
SETTINGS = dict(contract=CONTRACT, activation_ms=PLACE, cancel_effect_ms=CANCEL, pending_cancel_ms=PENDING, post_notice_ms=POST_NOTICE,
                fill_notice_ms=FILL_NOTICE, cancel_ack_ms=CANCEL_ACK, step_ms=STEP, end_ms=END, size=SIZE, cap=CAP, hold_ticks=HOLD,
                tie_order='trades <= tick; notices due by the tick; client structural requests; policy decisions in the alternating side order')
P1_TOL = dict(shares=1e-9, cash=1e-7, position=1e-9, value=1e-7)   # numerical aggregate parity with the C library (not event parity)
OUT = Path(I.K.IO) / 'branch_engine_20261119'
MARK = 'BRANCH ENGINE'
KERNEL_MODE = False                                        # True: the kernel's structural rule for every listed order (parity P1 only)
WAIT_IS_NO_ENTRY = False                                   # planted-error switch (selftest only): WAIT never posts later
REENTER_EARLY = False                                      # planted-error switch (selftest only): closes a pulled generation at its request
STALE_VIEW = False                                         # planted-error switch (selftest only): the second side's view is not rebuilt


# ---------------------------------------------------------------------------------------------------------------------- the engine
class Engine:
    """The execution side of one branch: orders, public trades applied once in priority, the structural rule, every fill and request."""

    def __init__(self, path, end=END, client=False):
        self.client = bool(client)                                                          # a client issues the structural requests (Branch)
        self.f = np.asarray(path['frames'], float); self.tr = np.asarray(path['trades'], float).reshape(-1, 4); self.end = float(end)
        C.require(self.f.ndim == 2 and self.f.shape[1] == 22 and len(self.f) > 0, 'engine: frame schema')
        C.require((np.diff(self.f[:, 0]) >= 0).all() and (np.diff(self.f[:, 1]) >= 0).all() and (np.diff(self.tr[:, 0]) >= 0).all(), 'engine: chronology')
        self.nf = len(self.f); self.o = []; self.it = 0; self.known = 0; self.fills = []; self.events = []

    def shown(self, j, side, price):
        p = 2 + side * 10; last = self.f[j, p + 4]
        if last >= 0 and (price < last if side == 0 else price > last):
            return np.nan
        return float(sum(self.f[j, p + 5 + k] for k in range(5) if self.f[j, p + k] >= 0 and self.f[j, p + k] == price))

    def add9(self, o9, action=1):
        """The kernel's order initialisation (pf_flow_control_kernel.cpp lines 42-53) for one 9-field order and its action (0 / 1 / 2)."""
        side, rung, price, size, ahead, act, remove, submit, new = [float(x) for x in o9]
        o = dict(gid=len(self.o), side=int(side), rung=rung, price=price, size=size, rem=0. if action == 0 else size, ahead=ahead if np.isfinite(ahead) else 0.,
                 act=act, remove=remove, submit=submit, new=new > 0, ci=0, refused=False, req=None, why=None)
        if action == 2:
            o['remove'] = min(o['remove'], max(CANCEL, submit + PENDING) if act > 0 else CANCEL)
        if o['new'] and o['rem'] > 0:
            j = 0
            while j + 1 < self.nf and self.f[j + 1, 0] <= act:
                j += 1
            opp = self.f[j, 2 + (1 - o['side']) * 10]
            if (0 <= opp <= price) if o['side'] == 0 else (opp >= price):
                o['rem'] = 0.; o['refused'] = True
            else:
                a = self.shown(j, o['side'], price); o['ahead'] = a if np.isfinite(a) else 0.; o['ci'] = j
        self.o.append(o); self.events.append((submit, 'submit', o['gid'], dict(side=o['side'], price=price, act=act, refused=o['refused'])))
        return o['gid']

    def add(self, side, price, d):
        """A new 30-share JOIN generation decided and submitted at d: activation d + 58 ms."""
        return self.add9([side, 0., price, SIZE, 0., d + PLACE, 1e30, d, 1.])

    def request(self, gid, g, why):
        """A removal requested at g: it lands after the cancel latency (the pending-cancel clock before activation)."""
        o = self.o[gid]; arr = max(g + CANCEL, o['submit'] + PENDING) if g < o['act'] else g + CANCEL
        if arr < o['remove']:
            o['remove'] = arr
        if o['req'] is None:
            o['req'] = float(g); o['why'] = why
        self.events.append((float(g), f'{why}_request', gid, dict(effect=o['remove'])))

    def advance(self, grid):
        """The kernel's loop body for one grid: the trades up to min(grid, END), then (grid < END) the structural rule at grid."""
        limit = min(grid, self.end)
        pri = sorted(range(len(self.o)), key=lambda i: (self.o[i]['side'], -self.o[i]['price'] if self.o[i]['side'] == 0 else self.o[i]['price'], self.o[i]['act'], i))
        while self.it < len(self.tr) and self.tr[self.it, 0] <= limit:
            t, side, p, left = self.tr[self.it]; side = int(side); used = 0.
            for i in pri:
                o = self.o[i]
                if o['side'] != side or o['rem'] <= EPS or o['act'] > t or t >= o['remove']:
                    continue
                j = o['ci']
                while j + 1 < self.nf and self.f[j + 1, 0] < t:
                    j += 1; a = self.shown(j, side, o['price'])
                    if np.isfinite(a):
                        o['ahead'] = min(o['ahead'], a)
                o['ci'] = j
                through = p < o['price'] if side == 0 else p > o['price']; q = 0.
                if p == o['price']:
                    use = min(left, o['ahead']); o['ahead'] -= use; q = min(left - use, o['rem'])
                elif through:
                    q = min(left, o['rem']); o['ahead'] = 0.
                if q > EPS:
                    left -= q; used += q; o['rem'] -= q; self.fills.append((float(t), o['gid'], side, float(q), o['price'], int(self.it)))
            C.require(used <= self.tr[self.it, 3] + 1e-7, 'engine: a public trade filled more than its size')
            self.it += 1
        if grid >= self.end:
            return
        while self.known + 1 < self.nf and self.f[self.known + 1, 1] <= grid:
            self.known += 1
        if self.client:
            return
        fk = self.f[self.known]
        for o in self.o:
            if o['rem'] <= EPS or o['remove'] <= grid or (grid == 0 and o['new']) or (not KERNEL_MODE and grid <= o['submit']):
                continue
            best, opp = fk[2 + o['side'] * 10], fk[2 + (1 - o['side']) * 10]
            behind = best - o['price'] if o['side'] == 0 else o['price'] - best
            cross = o['price'] >= opp if o['side'] == 0 else o['price'] <= opp
            if best < 0 or behind > HOLD + o['rung'] or cross:
                self.request(o['gid'], grid, 'structural')

    def value(self, mark):
        """Cash, inventory and the END value of every fill; the ledger identity is checked by the caller."""
        sg = lambda s: 1. if s == 0 else -1.
        cash = -sum(sg(s) * q * p for _, _, s, q, p, _ in self.fills); pos = sum(sg(s) * q for _, _, s, q, _, _ in self.fills)
        return cash, pos, cash + pos * mark


def batch(path, orders9, actions=None, horizon=END):
    """The kernel run of one branch (no client): orders as listed, the grid loop to the horizon -> the engine after END."""
    E = Engine(path, horizon); orders9 = np.asarray(orders9, float).reshape(-1, 9); actions = np.ones(len(orders9)) if actions is None else np.asarray(actions)
    for o9, a in zip(orders9, actions):
        E.add9(o9, int(a))
    g = 0.
    while True:
        E.advance(g)
        if g >= horizon:
            break
        g += STEP
    return E


def ckernel(path, orders9, actions, horizon=END):
    """The C kernel (pf_flow_control_kernel.cpp through its library, as SC.kernel_eval calls it) for ONE branch with the given actions ->
    (per-order filled shares, cash, position, value)."""
    fo = np.array([0, len(path['frames'])], np.int64); to = np.array([0, len(path['trades'])], np.int64)
    f = np.ascontiguousarray(path['frames'], float); t = np.ascontiguousarray(np.asarray(path['trades'], float).reshape(-1, 4)); delta = np.zeros(1)
    mark = np.array([path['mark'] if path.get('mark') is not None else 50.], float); o = np.ascontiguousarray(np.asarray(orders9, float).reshape(-1, 9))
    acts = np.ascontiguousarray(np.asarray(actions, float).reshape(1, -1)); out = np.zeros((1, 1, 6)); qty = np.zeros((1, 1, len(o)))
    dp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)); ip = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    rc = FK.library().evaluate(1, 1, len(o), ip(fo), ip(to), dp(f), dp(t), dp(delta), dp(mark), dp(o), dp(acts), 0., 50., float(horizon),
                               PLACE, CANCEL, PENDING, HOLD, dp(out), dp(qty))
    C.require(rc == 0, f'the C kernel refused ({rc})')
    return qty[0, 0].copy(), float(out[0, 0, 3]), float(out[0, 0, 2]), float(out[0, 0, 0])


# ---------------------------------------------------------------------------------------------------------------------- the client
class Branch:
    """One branch: the engine plus the client's known state, its reservations, decisions, fallbacks and constraints."""

    def __init__(self, path, episode=0, end=END, settings=None):
        C.require(float(end) == END, 'a branch END other than the contract\'s common END')
        C.require(settings is None or all(SETTINGS.get(k_) == v_ for k_, v_ in settings.items()), 'a timing / contract setting other than the declared scenario')
        C.require(CANCEL_ACK >= FILL_NOTICE, 'a cancel ACK that can overtake fill notices needs a cumulative-fill watermark (not in this contract)')
        self.E = Engine(path, end, client=True); self.counts = {}; self.episode = int(episode); self.end = float(end); self.mark = path.get('mark')
        self.settings = dict(SETTINGS, **(settings or {})); self.open = {0: None, 1: None}; self.closed = {}
        self.decisions = []; self.fallbacks = []; self.constraints = []; self.k = 0

    def known_filled(self, gid, g):
        return sum(q for t, i, _, q, _, _ in self.E.fills if i == gid and t + FILL_NOTICE <= g)

    def status(self, s, g):
        """The client-known state of side s at g (closing a generation whose refusal / full fill / removal is known)."""
        if g >= self.end:
            return 'terminal'
        gid = self.open[s]
        if gid is None:
            return 'absent'
        o = self.E.o[gid]
        if g < o['act'] + POST_NOTICE:
            return 'post_pending'
        kf = self.known_filled(gid, g); ack = o['remove'] + CANCEL_ACK if o['req'] is not None else np.inf
        why = 'refused' if o['refused'] else 'filled' if kf >= o['size'] - 1e-9 else 'removed' if (ack <= g or (REENTER_EARLY and o['req'] is not None)) else None
        if why:
            self.closed[gid] = (why, float(g)); self.open[s] = None; self.E.events.append((float(g), f'closed_{why}', gid, {}))
            return 'absent'
        return 'cancel_pending' if o['req'] is not None else 'live'

    def reserve(self, g):
        """(I, B, A): the client-known position and the known remaining of the open bid / ask generations."""
        sg = lambda s: 1. if s == 0 else -1.
        Ik = sum(sg(s) * q for t, _, s, q, _, _ in self.E.fills if t + FILL_NOTICE <= g); res = {0: 0., 1: 0.}
        for s in (0, 1):
            if self.open[s] is not None:
                res[s] = self.E.o[self.open[s]]['size'] - self.known_filled(self.open[s], g)
        return Ik, res[0], res[1]

    def check_position(self, g):
        """The worst-case reservation inequalities I + B <= cap and -I + A <= cap, and the independent engine-position check: the engine's
        position within [I - A, I + B] (unknown fills are reserved); after every notice and command."""
        Ik, Bk, Ak = self.reserve(g); P = sum((1. if s == 0 else -1.) * q for t, _, s, q, _, _ in self.E.fills if t <= g)
        C.require(Ik + Bk <= CAP + 1e-9 and -Ik + Ak <= CAP + 1e-9, f'reservation inequality at {g}: I {Ik}, B {Bk}, A {Ak}, cap {CAP}')
        C.require(Ik - Ak - 1e-9 <= P <= Ik + Bk + 1e-9, f'position check at {g}: engine {P} outside [{Ik - Ak}, {Ik + Bk}]')

    def structural(self, g):
        """The client's structural rule at tick g on CLIENT-KNOWN state (review 043ca403 P1): every open generation without a request, after
        its submission tick, more than 3 ticks + rung behind the best / at or through the opposite best / without a book in the last received
        frame -> a request (no execution effect if the order is already filled - the client does not know that yet)."""
        fk = self.E.f[self.E.known]
        for s in (0, 1):
            gid = self.open[s]
            if gid is None:
                continue
            o = self.E.o[gid]
            if o['req'] is not None or g <= o['submit']:
                continue
            best, opp = fk[2 + s * 10], fk[2 + (1 - s) * 10]
            behind = best - o['price'] if s == 0 else o['price'] - best
            cross = o['price'] >= opp if s == 0 else o['price'] <= opp
            if best < 0 or behind > HOLD + o['rung'] or cross:
                self.E.request(gid, g, 'structural')

    def tally(self, s, key):
        self.counts[f'{"bid" if s == 0 else "ask"} {key}'] = self.counts.get(f'{"bid" if s == 0 else "ask"} {key}', 0) + 1

    def view(self, g, s):
        """The client view of side s at g: nothing an engine event after g could change."""
        fk = self.E.f[self.E.known]; st = self.status(s, g); other = self.status(1 - s, g); Ik, Bk, Ak = self.reserve(g); gid = self.open[s]
        o = self.E.o[gid] if gid is not None else None
        room = (CAP - (Ik + Bk + SIZE)) if s == 0 else (CAP - (-Ik + Ak + SIZE))
        return dict(t=float(g), side=s, end=self.end, state=st, other=other, gid=gid, price=o['price'] if o else None, act=o['act'] if o else None,
                    known_filled=self.known_filled(gid, g) if gid is not None else 0., position=Ik, room=room, bid=float(fk[2]), bid_size=float(fk[7]),
                    ask=float(fk[12]), ask_size=float(fk[17]), received=float(fk[1]), age=(g - o['act']) if (o and st == 'live') else None)

    def act(self, g, s, action, v):
        """Apply one decision: POST / WAIT when absent, KEEP / PULL when live; constraints and fallbacks logged."""
        st = v['state']
        if st in ('post_pending', 'cancel_pending', 'terminal'):
            C.require(action is None, f'an action in state {st}'); self.tally(s, f'{st} ticks'); return None
        if action == 'FALLBACK':
            action = 'WAIT' if st == 'absent' else 'KEEP'; self.fallbacks.append((float(g), s, st, 'feature_unavailable')); self.tally(s, f'{st} fallback')
        elif st == 'absent':
            self.tally(s, 'absent cap-blocked' if v['room'] < -1e-9 else 'absent near-END' if g + PLACE >= self.end else f'absent feasible, {action}')
        else:
            self.tally(s, f'live, {action}')
        C.require(action in (('POST', 'WAIT') if st == 'absent' else ('KEEP', 'PULL')), f'action {action} in state {st}')
        if action == 'POST':
            if not book_ok(v, s):
                self.constraints.append((float(g), s, 'no_book')); action = 'WAIT'
            elif v['room'] < -1e-9:
                self.constraints.append((float(g), s, 'cap')); action = 'WAIT'
            elif g + PLACE >= self.end:
                self.constraints.append((float(g), s, 'near_end')); action = 'WAIT'
            else:
                self.open[s] = self.E.add(s, v['bid'] if s == 0 else v['ask'], g)
        elif action == 'PULL':
            C.require(v['gid'] is not None and v['gid'] == self.open[s], 'a PULL for a generation that is not the open one')
            self.E.request(self.open[s], g, 'policy')
        self.decisions.append((float(g), s, st, action, self.open[s])); self.check_position(g); return action


def book_ok(v, s):
    """A two-sided book with the side's best in [1, 99] in the client's last received frame."""
    return 0 <= v['bid'] < v['ask'] and 1 <= (v['bid'] if s == 0 else v['ask']) <= 99


def decide(B, g, s, policy, forced=None, v=None):
    """One side's decision at g: none in a pending / terminal state; else the forced action (a fork), a fallback when the view has no valid book,
    or the policy's action on the view."""
    v = v if v is not None else B.view(g, s)
    a = None if v['state'] in ('post_pending', 'cancel_pending', 'terminal') else forced if forced is not None else 'FALLBACK' if not book_ok(v, s) else policy(v)
    return B.act(g, s, a, v)


def pre_tick(B, g):
    """The declared tie order before the decisions at tick g: the engine's trades up to g, the notices due by g (statuses), the invariants, then
    the client's structural requests."""
    B.E.advance(g); B.status(0, g); B.status(1, g); B.check_position(g); B.structural(g)


def fingerprint(B):
    """A hash of a branch's complete state (orders, fills, events, client decisions / closures / counters): equal forks have equal prints."""
    import hashlib
    return hashlib.sha256(json.dumps([B.E.o, B.E.fills, B.E.events, B.decisions, sorted(B.closed.items()), B.open, B.counts, B.E.it, B.E.known],
                                     sort_keys=True, default=str).encode()).hexdigest()


def step(B, k, policy, forced=None):
    """One grid tick k of branch B: pre_tick, then both sides in the alternating order (forced: {side: action})."""
    g = k * STEP; pre_tick(B, g); B.k = k; order = ((B.episode + k) % 2, 1 - (B.episode + k) % 2)
    v0 = {s: B.view(g, s) for s in order} if STALE_VIEW else {}
    for s in order:
        decide(B, g, s, policy, (forced or {}).get(s), v0.get(s))
    B.check_position(g)


def finish(B):
    """The trades up to END (inclusive); open exposure disclosed; returns the branch's ledger."""
    B.E.advance(B.end); B.status(0, B.end - EPS); B.status(1, B.end - EPS)
    if B.mark is None or not np.isfinite(B.mark):
        return dict(status='missing_mark', fills=list(B.E.fills))
    cash, pos, val = B.E.value(B.mark)
    open_x = {s: float(B.E.o[B.open[s]]['rem']) if (B.open[s] is not None and B.E.o[B.open[s]]['remove'] > B.end) else 0. for s in (0, 1)}
    return dict(status='ok', cash=cash, position=pos, value=val, fills=list(B.E.fills), open_at_end=open_x, generations=len(B.E.o),
                fallbacks=list(B.fallbacks), constraints=list(B.constraints), decisions=list(B.decisions), counts=dict(B.counts),
                requests=[e_ for e_ in B.E.events if e_[1].endswith('_request')],
                orders9=[[o['side'], o['rung'], o['price'], o['size'], 0., o['act'], o['remove'] if o['why'] == 'policy' else 1e30, o['submit'], 1.] for o in B.E.o])


def run(path, policy, episode=0, k0=0, B=None, until=None):
    """Branch B (new if None) from tick k0 to `until` (exclusive; None = END) under the policy; returns B (finish() not applied)."""
    C.require(getattr(policy, 'horizon', None) == END, 'the policy does not declare the common END horizon')
    B = B or Branch(path, episode); nt = int(np.ceil(B.end / STEP))
    for k in range(k0, nt if until is None else until):
        step(B, k, policy)
    return B


def fork_state(path, rollin, k, side, episode=0, settings=None):
    """The state at tick k just before `side` decides, reached under the roll-in policy (pre_tick applied; the other side decides first when
    the alternating order puts it first)."""
    C.require(getattr(rollin, 'horizon', None) == END, 'the policy does not declare the common END horizon')
    B0 = run(path, rollin, episode, 0, Branch(path, episode, settings=settings), until=k); g = k * STEP; pre_tick(B0, g); B0.k = k
    if (episode + k) % 2 != side:
        decide(B0, g, 1 - side, rollin)
    return B0


def pair(path, policy, k, side, kind, episode=0, settings=None, rollin=None, state=None):
    """paired_continuation_v2 at tick k and side: the fork state (a saved `state`, or the roll-in policy's - default: the continuation policy)
    is copied; POST / WAIT or KEEP / PULL is applied in the two copies; both continue under the policy to the same END -> dict(Y, the two
    ledgers, checks, the fork state's fingerprint)."""
    C.require(kind in ('post', 'keep'), 'undeclared pair kind'); C.require(getattr(policy, 'horizon', None) == END, 'the policy does not declare the common END horizon')
    Bv = copy.deepcopy(state) if state is not None else fork_state(path, rollin or policy, k, side, episode, settings)
    C.require(Bv.k == k and Bv.episode == episode, 'a saved fork state of another tick / episode'); g = k * STEP; first = (episode + k) % 2; fp = fingerprint(Bv)
    v = Bv.view(g, side)
    if v['state'] != ('absent' if kind == 'post' else 'live'):
        return dict(status=f'not_{kind}_state', state=v['state'])
    if kind == 'post' and not (book_ok(v, side) and v['room'] >= -1e-9 and g + PLACE < Bv.end):
        return dict(status='post_not_feasible', state=v['state'])
    arms = {}
    for name, a in ((('POST', 'POST'), ('WAIT', 'WAIT')) if kind == 'post' else (('KEEP', 'KEEP'), ('PULL', 'PULL'))):
        Bx = copy.deepcopy(Bv); decide(Bx, g, side, policy, forced=a)
        if first == side:
            decide(Bx, g, 1 - side, policy)
        Bx.check_position(g); nt = int(np.ceil(Bx.end / STEP))
        pk = Policy(lambda v_: 'WAIT' if (v_['state'] == 'absent' and v_['side'] == side) else policy(v_), 'no_entry') if (WAIT_IS_NO_ENTRY and a == 'WAIT') else policy
        for kk in range(k + 1, nt):
            step(Bx, kk, pk)
        arms[name] = finish(Bx)
    na, nb = list(arms)
    if any(x['status'] != 'ok' for x in arms.values()):
        return dict(status='missing_mark', arms=arms, fork_fingerprint=fp)
    common = prefork_identical(arms[na]['fills'], arms[nb]['fills'], g)
    return dict(status='ok', kind=kind, k=k, side=side, Y=arms[na]['value'] - arms[nb]['value'], arms=arms, common_prefork_identical=common,
                settings=dict(SETTINGS, **(settings or {})), fork_fingerprint=fp)


def prefork_identical(fa, fb, g):
    """The common history: the two arms' fills up to the fork tick are the same list (no fill counted twice or lost)."""
    pre = lambda x: [f_ for f_ in x if f_[0] <= g]
    return pre(fa) == pre(fb)


class Policy:
    """A frozen causal policy: a function of the client view only, with its declared horizon."""

    def __init__(self, fn, name, horizon=END):
        self.fn, self.name, self.horizon = fn, name, horizon

    def __call__(self, v):
        return self.fn(v)


NEVER = Policy(lambda v: 'WAIT' if v['state'] == 'absent' else 'KEEP', 'never')
POST_KEEP = Policy(lambda v: 'POST' if v['state'] == 'absent' else 'KEEP', 'post_keep')
ONCE_KEEP = Policy(lambda v: ('POST' if v['t'] == 0 else 'WAIT') if v['state'] == 'absent' else 'KEEP', 'once_keep')
ONCE_PULL_FIRST = Policy(lambda v: ('POST' if v['t'] == 0 else 'WAIT') if v['state'] == 'absent' else 'PULL', 'once_pull_first')


HOLD_ONCE = Policy(lambda v: ('POST' if v['t'] == 0 else 'WAIT') if v['state'] == 'absent' else 'KEEP', 'hold_once')
CHASE = Policy(lambda v: 'POST' if v['state'] == 'absent' else ('PULL' if (v['bid'] if v['side'] == 0 else v['ask']) != v['price'] else 'KEEP'), 'chase')


def reactive(adverse=1.):
    """A book-reactive stand-in: post when absent; pull when the opposite best has moved within `adverse` of the order's price (bid: ask - price
    <= adverse)."""
    def fn(v):
        if v['state'] == 'absent':
            return 'POST'
        gap = (v['ask'] - v['price']) if v['side'] == 0 else (v['price'] - v['bid'])
        return 'PULL' if gap <= adverse else 'KEEP'
    return Policy(fn, f'reactive_{adverse:g}')


def side_policy(side, mode):
    """The degenerate policies of the slice: POST at 0 on `side` only, then KEEP (mode keep) or PULL at the first live tick (mode pull)."""
    def fn(v):
        if v['state'] == 'absent':
            return 'POST' if (v['t'] == 0 and v['side'] == side) else 'WAIT'
        return 'PULL' if mode == 'pull' else 'KEEP'
    return Policy(fn, f'side{side}_{mode}')


def fills_equal(ef, kt, kq):
    """The engine's fills (one row per public trade) against the keep / pull run's (one row per distinct trade time): equal shares at equal
    times once the engine's rows are summed per time (1e-9)."""
    ea = {}
    for t_, q_ in ef:
        ea[t_] = ea.get(t_, 0.) + q_
    ka = {}
    for t_, q_ in zip(np.asarray(kt, float), np.asarray(kq, float)):
        ka[t_] = ka.get(t_, 0.) + q_
    return sorted(ea) == sorted(ka) and all(abs(ea[t_] - ka[t_]) <= 1e-9 for t_ in ea)


SIDE10 = [0]                                               # F10's side (selftest only)
SLICE_DAYS = ('20261023', '20261024', '20261025', '20261026', '20261027')   # consumed: never a held-out or untouched day


def slice_check(day, n, out=None, tag=''):
    """The bounded check on a consumed day (see SLICE) -> slice_<day><tag>.json / .csv."""
    C.require(day in SLICE_DAYS, 'slice: consumed days only (Oct 23-27)'); start = time.time(); out = Path(out or OUT); kpd = Path(KP.OUT) / day
    C.require(not (out / f'slice_{day}{tag}.json').exists(), 'output exists')
    O = pd.read_parquet(kpd / 'orders.parquet', columns=['oid', 'm', 'side', 't', 'p', 'status', 'mark', 'v', 'q'])
    sel = O[O.status == 'eligible'].sort_values('oid').head(int(n)); F = pd.read_parquet(kpd / 'fills.parquet')
    F = F[(F.clock == 50.) & F.oid.isin(sel.oid)]; fg = {int(k): g.sort_values('t') for k, g in F.groupby('oid')}
    src = KP.Tape(); want = set(sel.m.astype(int).tolist()); markets = [m for m in src.markets(day) if int(m) in want]; off = src.offset(day); rows = []
    for b in range(0, len(markets), 24):
        data = src.load(day, markets[b:b + 24], off)
        for W in src.pairs(data):
            m = int(W[0]['m'])
            for o in sel[sel.m.astype(int) == m].itertuples():
                s_, t = int(o.side), float(o.t); path, pst = QP.build_path(W, t, t + END); r = dict(oid=int(o.oid), m=m, side=s_, path_status=pst)
                if path is None:
                    rows.append(r); continue
                kf = fg.get(int(o.oid)); kt = kf.t.values.astype(float) if kf is not None else np.zeros(0); kq = kf.q.values.astype(float) if kf is not None else np.zeros(0)
                L = {}
                for mode in ('keep', 'pull'):
                    B = run(path, side_policy(s_, mode)); L[mode] = finish(B)
                ef = {md: [(f_[0], f_[3]) for f_ in L[md]['fills'] if f_[1] == 0] for md in L}
                eq = fills_equal                                                            # v2: per trade time (v1 compared rows: 47 + 1 false, all equal per time)
                pm = kt < KP.landing(KP.J0)
                global KERNEL_MODE
                old = KERNEL_MODE; KERNEL_MODE = True
                try:
                    o9 = [[s_, 0., float(o.p), SIZE, 0., PLACE, 1e30, 0., 1.]]; qc = ckernel(path, o9, [1])[0]; Eb = batch(path, o9, [1])
                finally:
                    KERNEL_MODE = old
                r.update(price_equal=bool((path['bid'] if s_ == 0 else path['ask']) == float(o.p)), mark_equal=bool(abs(path['mark'] - float(o.mark)) <= 1e-9),
                         keep_fills_equal=eq(ef['keep'], kt, kq), pull_fills_equal=eq(ef['pull'], kt[pm], kq[pm]), keep_value_engine=L['keep']['value'],
                         keep_value_kp=float(o.v), keep_value_equal=bool(abs(L['keep']['value'] - float(o.v)) <= 1e-9), n_fills=len(kt),
                         p1_equal=bool(abs(float(qc[0]) - sum(f_[3] for f_ in Eb.fills)) <= 1e-9), pull_generations=L['pull']['generations'],
                         keep_open_at_end=L['keep']['open_at_end'][s_])
                rows.append(r)
        del data; C.guard('branch engine slice')
    D = pd.DataFrame(rows); ok = D[D.path_status == 'eligible']
    R = {'orders': int(len(D)), 'with a path': int(len(ok)), 'with fills': int((ok.n_fills > 0).sum())}
    for c in ('price_equal', 'mark_equal', 'keep_fills_equal', 'pull_fills_equal', 'keep_value_equal', 'p1_equal'):
        R[f'{c}: all'] = bool(ok[c].all()) if len(ok) else False; R[f'{c}: false count'] = int((~ok[c].astype(bool)).sum())
    R['SLICE (every check true on every order with a path, at least one order with fills)'] = 'PASS' if (len(ok) and ok.n_fills.gt(0).any() and all(
        R[f'{c}: all'] for c in ('price_equal', 'mark_equal', 'keep_fills_equal', 'pull_fills_equal', 'keep_value_equal', 'p1_equal'))) else 'FAIL'
    out.mkdir(parents=True, exist_ok=True); D.to_csv(out / f'slice_{day}{tag}.csv', index=False)
    C.dump(out / f'slice_{day}{tag}.json', dict(asof=C.stamp(), day=day, n=int(n), readings=R, settings=SETTINGS, inputs=dict(kp_receipt=C.sha(kpd / 'receipt.json')),
                                          outputs={f'slice_{day}{tag}.csv': C.sha(out / f'slice_{day}{tag}.csv')}, source_sha256=C.sha(Path(__file__).resolve()),
                                          provenance=KP.provenance(), seconds=time.time() - start))
    print(json.dumps(R, indent=1), flush=True)
    return R


ANCHORS = (('post', 0), ('keep', 4))                     # review 043ca403 (e): entry fork at 0 ms, retention fork at 200 ms (tick 4)


def slice_v3(day, per_side=128, out=None):
    """Review 043ca403 (e), selection frozen before any result: the first `per_side` eligible archived opportunities per side by oid of a
    consumed day. (1) single generation: POST at 0 then KEEP, and POST at 0 then PULL at the first review, reconciled to the archive - JOIN
    price, activation, the pull's request / effect, fills per trade time, END mark, cash, inventory and BOTH END values; (2) C-compatibility
    aggregate parity (shares, cash, position, END value within P1_TOL); (3) on the deduplicated (market, start) tapes, POST_KEEP and CHASE on
    both sides with POST / WAIT at 0 ms and KEEP / PULL at 200 ms per side: incremental = batch fill tuples (base run and every arm), the common
    pre-fork history, value identity per arm, no invariant refusal, client causality (views, reservations, requests and decisions up to 200 ms
    equal on the tape truncated after 200 ms); (4) statuses, histories, later-post / re-entry counts, pending / cap counters, per-side coverage.
    -> slice_v3_<day>.json, _orders.csv, _forks.csv, _histories.json. Invariant checks only: no profitability threshold, no policy selection."""
    C.require(day in SLICE_DAYS, 'slice: consumed days only (Oct 23-27)'); start = time.time(); out = Path(out or OUT); kpd = Path(KP.OUT) / day
    C.require(not (out / f'slice_v3_{day}.json').exists(), 'output exists')
    O = pd.read_parquet(kpd / 'orders.parquet', columns=['oid', 'm', 'side', 't', 'p', 'status', 'mark', 'v', 'q']); E_ = O[O.status == 'eligible']
    sel = pd.concat([E_[E_.side == s_].sort_values('oid').head(int(per_side)) for s_ in (0, 1)], ignore_index=True)
    F = pd.read_parquet(kpd / 'fills.parquet'); F = F[(F.clock == 50.) & F.oid.isin(sel.oid)]; fg = {int(k_): g_.sort_values('t') for k_, g_ in F.groupby('oid')}
    src = KP.Tape(); want = set(sel.m.astype(int).tolist()); markets = [m for m in src.markets(day) if int(m) in want]; off = src.offset(day)
    land0 = float(KP.landing(KP.J0)); orows, frows, hist = [], [], {}
    for b in range(0, len(markets), 24):
        data = src.load(day, markets[b:b + 24], off)
        for W in src.pairs(data):
            m = int(W[0]['m']); sm = sel[sel.m.astype(int) == m]
            for o in sm.itertuples():                                                       # (1) and (2): every selected order
                s_, t = int(o.side), float(o.t); path, pst = QP.build_path(W, t, t + END); r = dict(oid=int(o.oid), m=m, side=s_, t=t, path_status=pst)
                if path is None:
                    orows.append(r); continue
                kf = fg.get(int(o.oid)); kt = kf.t.values.astype(float) if kf is not None else np.zeros(0); kq = kf.q.values.astype(float) if kf is not None else np.zeros(0)
                sg = 1. if s_ == 0 else -1.; pm = kt < land0; L = {}; G = {}
                for mode in ('keep', 'pull'):
                    Bm = run(path, side_policy(s_, mode)); L[mode] = finish(Bm); G[mode] = Bm.E.o[0]
                ark = {'keep': (kt, kq), 'pull': (kt[pm], kq[pm])}
                for mode, (tt, qq) in ark.items():
                    ef = [(f_[0], f_[3]) for f_ in L[mode]['fills'] if f_[1] == 0]
                    r[f'{mode}_fills_equal'] = fills_equal(ef, tt, qq); cash_a = -sg * float(o.p) * float(qq.sum()); inv_a = sg * float(qq.sum())
                    r[f'{mode}_cash_equal'] = abs(L[mode]['cash'] - cash_a) <= 1e-7; r[f'{mode}_inventory_equal'] = abs(L[mode]['position'] - inv_a) <= 1e-9
                    r[f'{mode}_value_archive'] = cash_a + inv_a * float(o.mark); r[f'{mode}_value_engine'] = L[mode]['value']
                    r[f'{mode}_value_equal'] = abs(L[mode]['value'] - r[f'{mode}_value_archive']) <= 1e-7
                r['keep_value_equal_archive_v'] = abs(L['keep']['value'] - float(o.v)) <= 1e-7
                r['price_equal'] = G['keep']['price'] == float(o.p) == G['pull']['price']; r['activation_equal'] = G['keep']['act'] == PLACE == G['pull']['act']
                r['mark_equal'] = abs(path['mark'] - float(o.mark)) <= 1e-9; r['pull_request'] = G['pull']['why']; r['pull_request_ms'] = G['pull']['req']
                r['pull_effect_ms'] = G['pull']['remove'] if G['pull']['req'] is not None else None
                r['pull_effect_equal'] = (G['pull']['why'] == 'policy' and G['pull']['req'] == KP.J0 * STEP and G['pull']['remove'] == land0) or G['pull']['why'] == 'structural'
                global KERNEL_MODE
                old = KERNEL_MODE; KERNEL_MODE = True
                try:
                    o9 = [[s_, 0., float(o.p), SIZE, 0., PLACE, 1e30, 0., 1.]]; qc, cc, pc, vc = ckernel(path, o9, [1]); Eb = batch(path, o9, [1]); cp, pp, vp = Eb.value(path['mark'])
                finally:
                    KERNEL_MODE = old
                r['c_parity'] = (abs(float(qc[0]) - sum(f_[3] for f_ in Eb.fills)) <= P1_TOL['shares'] and abs(cc - cp) <= P1_TOL['cash'] and abs(pc - pp) <= P1_TOL['position']
                                 and abs(vc - vp) <= P1_TOL['value']); r['n_fills'] = len(kt); orows.append(r)
            for t in sorted(set(sm.t.astype(float).tolist())):                              # (3) and (4): the deduplicated tapes, both sides jointly
                path, pst = QP.build_path(W, t, t + END)
                if path is None:
                    frows.append(dict(m=m, t=t, status=pst)); continue
                for pol in (POST_KEEP, CHASE):
                    base = dict(m=m, t=t, policy=pol.name)
                    try:
                        Bb = run(path, pol); Lb = finish(Bb); p2 = replay9(path, Lb['orders9']) == [f_[:5] for f_ in Lb['fills']]
                        caus = views_until(path, pol, 5) == views_until(truncate(path, 4 * STEP), pol, 5)
                        frows.append(dict(base, fork='base run', side=-1, status='ok', p2=p2, causality=caus, value=Lb['value'], generations=Lb['generations'],
                                          **{f'count {k_}': v_ for k_, v_ in Lb['counts'].items()}))
                    except ValueError as e:
                        frows.append(dict(base, fork='base run', side=-1, status=f'refused: {str(e)[:120]}')); continue
                    for kind, k_ in ANCHORS:
                        for s_ in (0, 1):
                            row = dict(base, fork=kind, side=s_, k=k_)
                            try:
                                rp_ = pair(path, pol, k_, s_, kind)
                            except ValueError as e:
                                frows.append(dict(row, status=f'refused: {str(e)[:120]}')); continue
                            row['status'] = rp_['status']
                            if rp_['status'] == 'ok':
                                A1, A2 = list(rp_['arms']); a1, a2 = rp_['arms'][A1], rp_['arms'][A2]
                                ident = all(abs(sum((1. if f_[2] == 0 else -1.) * f_[3] * (path['mark'] - f_[4]) for f_ in a_['fills']) - a_['value']) <= 1e-7
                                            and abs(a_['cash'] + a_['position'] * path['mark'] - a_['value']) <= 1e-7 for a_ in (a1, a2))
                                p2a = all(replay9(path, a_['orders9']) == [f_[:5] for f_ in a_['fills']] for a_ in (a1, a2))
                                gens = lambda a_: [d_ for d_ in a_['decisions'] if d_[1] == s_ and d_[3] == 'POST']
                                later = any(d_[0] > k_ * STEP for d_ in gens(a2)) if kind == 'post' else None
                                reent = any(d_[0] > k_ * STEP for d_ in gens(a2)) if kind == 'keep' else None
                                row.update(Y=rp_['Y'], prefork_identical=rp_['common_prefork_identical'], value_identity=ident, p2_arms=p2a, later_post=later, re_entry=reent,
                                           **{f'{A1} {k2}': v2 for k2, v2 in a1['counts'].items()}, **{f'{A2} {k2}': v2 for k2, v2 in a2['counts'].items()})
                                hist[f'{m}|{t:.0f}|{pol.name}|{kind}|{s_}'] = {nm_: dict(value=a_['value'], fills=[f_[:5] for f_ in a_['fills']],
                                                                                         commands=[d_ for d_ in a_['decisions'] if d_[3] in ('POST', 'PULL')],
                                                                                         requests=a_['requests'], open_at_end=a_['open_at_end']) for nm_, a_ in rp_['arms'].items()}
                            frows.append(row)
        del data; C.guard('branch engine slice v3')
    D = pd.DataFrame(orows); Fk = pd.DataFrame(frows); ok = D[D.path_status == 'eligible']; R = {}
    single = ['price_equal', 'activation_equal', 'mark_equal', 'keep_fills_equal', 'pull_fills_equal', 'keep_cash_equal', 'pull_cash_equal', 'keep_inventory_equal',
              'pull_inventory_equal', 'keep_value_equal', 'pull_value_equal', 'keep_value_equal_archive_v', 'pull_effect_equal']
    R['(1) single generation: false counts'] = {c: int((~ok[c].astype(bool)).sum()) for c in single}
    R['(1) single generation (every check on every order with a path)'] = 'PASS' if len(ok) and all(v_ == 0 for v_ in R['(1) single generation: false counts'].values()) else 'FAIL'
    R['(2) C-compatibility aggregate parity'] = 'PASS' if len(ok) and ok.c_parity.astype(bool).all() else 'FAIL'
    br = Fk[Fk.get('fork', pd.Series(dtype=str)) == 'base run'] if len(Fk) else Fk; fo = Fk[(Fk.get('fork', pd.Series(dtype=str)) != 'base run') & (Fk.get('status', pd.Series(dtype=str)) == 'ok')] if len(Fk) else Fk
    refused = int(Fk.status.astype(str).str.startswith('refused').sum()) if len(Fk) else 0
    adapt_ok = (len(br) > 0 and refused == 0 and br.p2.astype(bool).all() and br.causality.astype(bool).all() and len(fo) > 0 and fo.prefork_identical.astype(bool).all()
                and fo.value_identity.astype(bool).all() and fo.p2_arms.astype(bool).all())
    R['(3) adaptive invariants (P2 base and arms, pre-fork history, value identity, no invariant refusal, causality to 200 ms)'] = 'PASS' if adapt_ok else 'FAIL'
    R['(3) counts'] = dict(tapes=int(len(br) // 2) if len(br) else 0, base_runs=int(len(br)), refused=refused, forks_ok=int(len(fo)),
                           p2_false=int((~br.p2.astype(bool)).sum()) if len(br) else 0, causality_false=int((~br.causality.astype(bool)).sum()) if len(br) else 0,
                           prefork_false=int((~fo.prefork_identical.astype(bool)).sum()) if len(fo) else 0, identity_false=int((~fo.value_identity.astype(bool)).sum()) if len(fo) else 0)
    cov = {}
    if len(Fk):
        for (pol, kind, s_), g_ in Fk[Fk.fork != 'base run'].groupby(['policy', 'fork', 'side']):
            cov[f'{pol} {kind} {"bid" if s_ == 0 else "ask"}'] = dict(statuses=g_.status.value_counts().to_dict(), later_post=int(g_.later_post.fillna(False).astype(bool).sum()),
                                                                      re_entry=int(g_.re_entry.fillna(False).astype(bool).sum()))
    R['(4) coverage: fork statuses, later posts (POST / WAIT forks), re-entries (KEEP / PULL forks), per policy and side'] = cov
    R['(4) selected orders per side (with a path)'] = {('bid' if s_ == 0 else 'ask'): int((ok.side == s_).sum()) for s_ in (0, 1)}
    R['SLICE v3 (1), (2) and (3) all PASS'] = 'PASS' if all(R[k_] == 'PASS' for k_ in R if k_.startswith(('(1) single generation (', '(2)', '(3) adaptive'))) else 'FAIL'
    out.mkdir(parents=True, exist_ok=True); D.to_csv(out / f'slice_v3_{day}_orders.csv', index=False); Fk.to_csv(out / f'slice_v3_{day}_forks.csv', index=False)
    C.dump(out / f'slice_v3_{day}_histories.json', hist)
    C.dump(out / f'slice_v3_{day}.json', dict(asof=C.stamp(), day=day, per_side=int(per_side), selection='the first per_side eligible archived opportunities per side by oid (review 043ca403 (e))',
                                             readings=R, settings=SETTINGS, p1_tolerances=P1_TOL, anchors=ANCHORS, inputs=dict(kp_receipt=C.sha(kpd / 'receipt.json')),
                                             outputs={x: C.sha(out / x) for x in (f'slice_v3_{day}_orders.csv', f'slice_v3_{day}_forks.csv', f'slice_v3_{day}_histories.json')},
                                             source_sha256=C.sha(Path(__file__).resolve()), provenance=KP.provenance(), seconds=time.time() - start))
    print(json.dumps(R, indent=1, default=str), flush=True)
    return R


# ---------------------------------------------------------------------------------------------------------------------- synthetic tapes
def frame(venue, recv, bid, ask, bsz=100., asz=100., bid2=None, ask2=None):
    """One 22-column frame: venue / received time, five bid levels (prices, sizes), five ask levels; empty levels -1 / 0."""
    bp = [bid, bid2 if bid2 is not None else -1., -1., -1., -1.]; ap = [ask, ask2 if ask2 is not None else -1., -1., -1., -1.]
    bs = [bsz, 100. if bid2 is not None else 0., 0., 0., 0.]; as_ = [asz, 100. if ask2 is not None else 0., 0., 0., 0.]
    return [venue, recv] + bp + bs + ap + as_


def tape(frames, trades, mark=50.):
    return dict(frames=np.asarray(frames, float), trades=np.asarray(trades, float).reshape(-1, 4), mark=mark)


def rand_tape(rng, n_frames=40, n_trades=60, end=END):
    """A random two-sided tape around 50 (prices in whole ticks), with trades at and through the best levels."""
    tf = np.sort(rng.uniform(0, end, n_frames)); tf[0] = 0.; fr = []; bid = 50.
    for t in tf:
        bid = float(np.clip(bid + rng.choice([-1., 0., 0., 1.]), 40., 60.)); ask = bid + float(rng.choice([1., 1., 2.]))
        fr.append(frame(t, t + float(rng.uniform(0, 30)), bid, ask, float(rng.integers(0, 200)), float(rng.integers(0, 200)), bid - 1., ask + 1.))
    fr = np.asarray(fr); fr[:, 1] = np.maximum.accumulate(fr[:, 1])
    tt = np.sort(rng.uniform(1., end, n_trades)); tr = []
    for t in tt:
        j = int(np.searchsorted(fr[:, 0], t, side='right') - 1); s = int(rng.integers(0, 2)); lvl = fr[j, 2] if s == 0 else fr[j, 12]
        tr.append((t, s, lvl - (1. if (s == 0 and rng.uniform() < .2) else 0.) + (1. if (s == 1 and rng.uniform() < .2) else 0.), float(rng.integers(1, 120))))
    return tape(fr, tr, float(rng.uniform(40, 60)))


# ---------------------------------------------------------------------------------------------------------------------- the controls
def p1_parity(rng, n=40):
    """P1, numerical AGGREGATE parity (not event parity: the C interface returns no fill times): KERNEL_MODE against the C library on random
    tapes, order lists and actions - per-order shares, cash, position and END value within P1_TOL -> mismatches."""
    global KERNEL_MODE
    bad = 0; old = KERNEL_MODE; KERNEL_MODE = True
    try:
        pp2 = tape([frame(0, 0, 50, 52, 0, 0, 49)], [(500, 0, 48, 35)], 51.)               # own bids at 49 and 50, one sell through both
        o2 = [[0, 0, 49, 30, 0, 58, 1e30, 0, 1], [0, 0, 50, 30, 0, 58, 1e30, 0, 1]]; qc2 = ckernel(pp2, o2, [1, 1])[0]; E2 = batch(pp2, o2, [1, 1])
        qp2 = [sum(q for _, i, _, q, _, _ in E2.fills if i == g_) for g_ in (0, 1)]
        bad += not (np.allclose(qc2, qp2, rtol=0, atol=1e-12) and qp2 == [5., 30.])         # price priority: the better bid fills first
        for _ in range(n):
            p = rand_tape(rng); no = int(rng.integers(1, 6)); o = []
            for _ in range(no):
                s = int(rng.integers(0, 2)); act = float(rng.choice([58., rng.uniform(0, 3000)])); sub = float(rng.choice([0., max(act - 58., 0.)]))
                best = p['frames'][0, 2] if s == 0 else p['frames'][0, 12]
                o.append([s, float(rng.integers(0, 2)), best + float(rng.choice([-1., 0., 0., 1.])) * (1 if s == 1 else -1), float(rng.integers(5, 40)),
                          float(rng.choice([0., 50., np.inf])), act, float(rng.choice([1e30, rng.uniform(act, 5000.)])), sub, float(rng.integers(0, 2))])
            o = np.nan_to_num(np.asarray(o), posinf=0.); ac = rng.integers(0, 3, no); qc, cc, pc, vc = ckernel(p, o, ac)
            E = batch(p, o, ac); qp = np.array([sum(q for _, i, _, q, _, _ in E.fills if i == g_) for g_ in range(no)]); cp, pp, vp = E.value(p['mark'])
            bad += not (np.allclose(qc, qp, rtol=0, atol=P1_TOL['shares']) and abs(cc - cp) <= P1_TOL['cash'] and abs(pc - pp) <= P1_TOL['position']
                        and abs(vc - vp) <= P1_TOL['value'])
    finally:
        KERNEL_MODE = old
    return bad


def replay(path, B):
    """P2: the adaptive branch's final order list replayed as a batch (engine mode) -> its fills."""
    o9 = [[o['side'], o['rung'], o['price'], o['size'], 0., o['act'], o['remove'] if o['why'] == 'policy' else 1e30, o['submit'], 1.] for o in B.E.o]
    return batch(path, o9, horizon=B.end).fills if o9 else []


def replay9(path, orders9, end=END):
    """P2 for a finished ledger: its final order list replayed as a batch (engine mode) -> fill tuples (t, gid, side, shares, price)."""
    return [f_[:5] for f_ in batch(path, orders9, horizon=end).fills] if orders9 else []


def views_until(path_, pol, kmax):
    """Per tick up to kmax: the client views of both sides after pre_tick, the reservations and the requests so far, then the decisions."""
    B_ = Branch(path_); out_ = []
    for kk in range(kmax):
        g_ = kk * STEP; pre_tick(B_, g_); B_.k = kk
        out_.append((g_, [B_.view(g_, s_) for s_ in (0, 1)], B_.reserve(g_), [e_ for e_ in B_.E.events if e_[1].endswith('_request')]))
        for s_ in ((B_.episode + kk) % 2, 1 - (B_.episode + kk) % 2):
            decide(B_, g_, s_, pol)
        out_.append(('decisions', list(B_.decisions)))
    return out_


def truncate(path, g):
    """The same tape with every frame of venue time after g and every trade after g removed (events a decision at g cannot know)."""
    q = dict(path); q['frames'] = path['frames'][path['frames'][:, 0] <= g]; q['trades'] = path['trades'][path['trades'][:, 0] <= g]
    return q


def traces(out=None):
    """Example event / ledger traces of the synthetic controls (F1 actual WAIT, F2 re-entry, F9 shared book) for the receipt -> traces.json."""
    out = Path(out or OUT); C.require(not (out / 'traces.json').exists(), 'output exists'); T = {}
    pw = tape([frame(0, 0, 50, 52, 20), frame(20, 30, 51, 53, 20)], [(300, 0, 51, 30), (900, 0, 51, 40), (1500, 0, 50, 60)], 52.)
    r1 = pair(pw, POST_KEEP, 0, 0, 'post'); T['F1 actual WAIT (POST vs WAIT at 0 ms, bid; then post_keep)'] = dict(Y=r1['Y'], arms={k_: {x: v_[x] for x in ('value', 'cash', 'position', 'fills', 'decisions', 'constraints', 'open_at_end')} for k_, v_ in r1['arms'].items()})
    pr = tape([frame(0, 0, 50, 52, 0, 0), frame(110, 112, 49, 51, 0)], [(120, 0, 50, 10), (150, 1, 52, 10), (400, 0, 49, 5)], 50.)
    Br = run(pr, Policy(lambda v: ('POST' if v['state'] == 'absent' else ('PULL' if (v['t'] == 100 and v['side'] == 0) else 'KEEP')), 'bid_pull_at_100')); Lr = finish(Br)
    T['F2 re-entry (bid pull at 100 ms, a fill while cancel-pending, an ask fill offsets it, the acknowledgement, a new generation at the new best)'] = dict(
        ledger={x: Lr[x] for x in ('value', 'cash', 'position', 'fills', 'decisions', 'open_at_end')}, events=Br.E.events, closed={str(k_): v_ for k_, v_ in Br.closed.items()})
    ps = tape([frame(0, 0, 50, 52, 0, 0)], [(100, 0, 50, 30), (700, 1, 52, 30)], 51.)
    B9 = run(ps, Policy(lambda v: ('POST' if (v['side'] == 0 or v['other'] != 'post_pending') else 'WAIT') if v['state'] == 'absent' else 'KEEP', 'ask_waits_for_bid')); L9 = finish(B9)
    T['F9 shared book (a bid fill, then an ask fill; the ask waits while the bid is post-pending)'] = dict(
        ledger={x: L9[x] for x in ('value', 'cash', 'position', 'fills', 'decisions', 'constraints', 'open_at_end')}, events=B9.E.events)
    out.mkdir(parents=True, exist_ok=True); C.dump(out / 'traces.json', dict(asof=C.stamp(), settings=SETTINGS, traces=T, source_sha256=C.sha(Path(__file__).resolve())))
    print(f"traces written: {list(T)}", flush=True)


def _controls(bad):  # noqa: C901
    """The controls (see the module's CONTROLS); failures are appended to bad."""
    rng = np.random.default_rng(20261119)
    n1 = p1_parity(rng)
    if n1:
        bad.append(f'P1: the port differs from the C kernel on {n1} of 40 random tapes')
    for i_ in range(12):                                                                    # P2 on random tapes under the reactive stand-in
        p = rand_tape(rng); B = run(p, reactive(1.), episode=i_); finish(B)
        if [f_[:5] for f_ in replay(p, B)] != [f_[:5] for f_ in B.E.fills]:
            bad.append(f'P2: the adaptive run differs from its batch replay on random tape {i_}'); break
    pg = tape([frame(0, 0, 55, 57, 0), frame(500, 505, 50, 52, 0)], [(1500, 0, 50, 30)], 51.)        # P2: the submission guard matters in a replay
    Bg = run(pg, Policy(lambda v: ('POST' if (v['t'] == 600 and v['side'] == 0) else 'WAIT') if v['state'] == 'absent' else 'KEEP', 'post_at_600')); finish(Bg)
    if not ([f_[:5] for f_ in replay(pg, Bg)] == [f_[:5] for f_ in Bg.E.fills] and len(Bg.E.fills) == 1):
        bad.append(f'P2: a generation submitted at 600 ms (5 ticks behind the best at 100 ms) differs between the run and its replay ({Bg.E.fills} vs {replay(pg, Bg)})')
    if not (fills_equal([(10., 3.), (10., 2.), (20., 1.)], [10., 20.], [5., 1.]) and not fills_equal([(10., 3.)], [10.], [5.])
            and not fills_equal([(10., 5.)], [11.], [5.]) and not fills_equal([(10., 5.)], [10., 20.], [5., 1.])):
        bad.append('the slice comparison (rows summed per trade time; a quantity, time or missing-fill difference must fail)')
    # ---- F1 actual WAIT: bid 50 / ask 52 to 150 ms, then bid 51 / ask 53; trades hit the bid at 51 from 300 ms
    pw = tape([frame(0, 0, 50, 52, 20), frame(20, 30, 51, 53, 20)], [(300, 0, 51, 30), (900, 0, 51, 40), (1500, 0, 50, 60)], 52.)   # the book moves before 50 ms
    r1 = pair(pw, POST_KEEP, 0, 0, 'post')
    sub_w = [o_ for o_ in r1['arms']['WAIT']['decisions'] if o_[1] == 0 and o_[3] == 'POST']
    sub_p = [o_ for o_ in r1['arms']['POST']['decisions'] if o_[1] == 0 and o_[3] == 'POST']
    if not (r1['status'] == 'ok' and sub_w and sub_w[0][0] > 0 and sub_p and sub_p[0][0] == 0 and r1['arms']['WAIT']['fills'] != r1['arms']['POST']['fills']):
        bad.append(f"F1: WAIT does not post later with a different history (WAIT posts {sub_w[:1]}, POST posts {sub_p[:1]})")
    # ---- F2 re-entry: the bid pulled at 100, a fill at 120 while cancel-pending (effect 137); an ask fill at 150 offsets it (the cap needs I + B + 30 <= 30);
    #      the acknowledgement at 237, a new bid generation at 250 at the new best
    pr = tape([frame(0, 0, 50, 52, 0, 0), frame(110, 112, 49, 51, 0)], [(120, 0, 50, 10), (150, 1, 52, 10), (400, 0, 49, 5)], 50.)
    pol_r = Policy(lambda v: ('POST' if v['state'] == 'absent' else ('PULL' if (v['t'] == 100 and v['side'] == 0) else 'KEEP')), 'bid_pull_at_100', END)
    Br = run(pr, pol_r); Lr = finish(Br); gens = [(o_['submit'], o_['price']) for o_ in Br.E.o if o_['side'] == 0]
    f_pend = [f_ for f_ in Lr['fills'] if f_[1] == 0 and 100 <= f_[0] < 137]
    if not (len(gens) >= 2 and gens[0] == (0., 50.) and gens[1][0] >= 237 and gens[1][1] == 49. and f_pend):
        bad.append(f'F2: no re-entry after the acknowledgement at a new price with a cancel-pending fill (generations {gens}, pending fills {f_pend})')
    # ---- F3 queue preservation: 40 ahead at 50; 30 trade at 50 (10 ahead left); a snapshot shows 200 at 50 (arrivals behind us); 15 trade
    pq = tape([frame(0, 0, 50, 52, 40), frame(300, 305, 50, 52, 200)], [(200, 0, 50, 30), (400, 0, 50, 15), (2000, 0, 50, 500)], 51.)
    Eq_ = batch(pq, [[0, 0, 50, 30, 0, 58, 1e30, 0, 1]]); fq = [f_ for f_ in Eq_.fills if f_[0] == 400]
    Eq2 = batch(pq, [[0, 0, 50, 30, 0, 58, 250, 0, 1], [0, 0, 50, 30, 0, 358, 1e30, 300, 1]])      # removed at 250, replaced (activation 358)
    f2 = [f_ for f_ in Eq2.fills if f_[1] == 1]
    if not (fq and abs(fq[0][3] - 5.) < 1e-12 and not [f_ for f_ in f2 if f_[0] < 2000]):
        bad.append(f'F3: queue priority not preserved (kept order at 400: {fq}; the replacement filled before the 500-share trade: {f2})')
    # ---- F4 economic identity on every arm of F1 / F2 and the pair difference; common pre-fork fills identical
    for nm_, L, mk in (('F1 POST', r1['arms']['POST'], pw['mark']), ('F1 WAIT', r1['arms']['WAIT'], pw['mark']), ('F2', Lr, pr['mark'])):
        led = sum((1. if s_ == 0 else -1.) * q * (mk - p_) for _, _, s_, q, p_, _ in L['fills'])
        if not (abs(led - L['value']) <= 1e-9 and abs(L['cash'] + L['position'] * mk - L['value']) <= 1e-9):
            bad.append(f"F4: {nm_} fill ledger {led} vs value {L['value']}")
    if not (r1['common_prefork_identical'] and abs(r1['Y'] - (r1['arms']['POST']['value'] - r1['arms']['WAIT']['value'])) < 1e-12):
        bad.append('F4: the pair difference or its common pre-fork fills')
    fx = [(10., 0, 0, 5., 50., 0)]
    if not (prefork_identical(fx, list(fx), 50.) and not prefork_identical(fx + fx, fx, 50.) and prefork_identical(fx + [(60., 0, 0, 1., 50., 1)], fx, 50.)):
        bad.append('F4: the pre-fork comparison accepts a common fill counted twice (or rejects a post-fork difference)')
    # ---- F5 continuation matters, both directions, engine-generated: Y_keep under a hold continuation vs a chase continuation (seeded tapes)
    found = {}; r5 = np.random.default_rng(55)
    for i_ in range(200):
        p5 = rand_tape(r5); S5 = fork_state(p5, HOLD_ONCE, 4, 0)                                 # ONE saved fork state; only the continuation varies
        a_ = pair(p5, HOLD_ONCE, 4, 0, 'keep', state=S5); b_ = pair(p5, CHASE, 4, 0, 'keep', state=S5)
        if a_['status'] == 'ok' and b_['status'] == 'ok' and a_['fork_fingerprint'] == b_['fork_fingerprint'] == fingerprint(S5):
            if a_['Y'] > 0 > b_['Y']:
                found.setdefault('hold > 0 > chase', (i_, a_['Y'], b_['Y']))
            if a_['Y'] < 0 < b_['Y']:
                found.setdefault('hold < 0 < chase', (i_, a_['Y'], b_['Y']))
        if len(found) == 2:
            break
    if len(found) != 2:
        bad.append(f'F5: from one saved fork state, the continuation never reverses the keep advantage in both directions on 200 seeded tapes: {found}')
    # ---- F6 causality: events after g cannot change the view or action at g; a known one can
    base = rand_tape(np.random.default_rng(5)); k6 = 30; g6 = k6 * STEP
    B6 = run(base, reactive(1.), until=k6); B6.E.advance(g6); v6 = B6.view(g6, 0)
    fut = copy.deepcopy(base); fut['frames'][fut['frames'][:, 0] > g6, 2] += 1.; fut['trades'] = fut['trades'][fut['trades'][:, 0] <= g6]   # venue after g
    B6f = run(fut, reactive(1.), until=k6); B6f.E.advance(g6); v6f = B6f.view(g6, 0)
    known_ = copy.deepcopy(base); jk = int(np.flatnonzero(known_['frames'][:, 1] <= g6)[-1]); known_['frames'][jk, 7] += 17.
    B6k = run(known_, reactive(1.), until=k6); B6k.E.advance(g6); v6k = B6k.view(g6, 0)
    if not (v6 == v6f and v6 != v6k):
        bad.append(f'F6: causality (future-perturbed view equal: {v6 == v6f}; known-perturbed view different: {v6 != v6k})')
    # ---- F10 (review 043ca403 P1) an unnoticed fill cannot change the client's views, structural requests or reservations before its notice
    pend = Policy(lambda v: (('POST' if v['t'] == 0 else 'WAIT') if v['side'] == SIDE10[0] else ('POST' if v['other'] == 'cancel_pending' else 'WAIT'))
                  if v['state'] == 'absent' else 'KEEP', 'post_once_and_other_side_on_pending')
    for side10, jump, px in ((0, (55., 57.), 50.), (1, (45., 47.), 52.)):
        SIDE10[0] = side10; fr10 = [frame(0, 0, 50, 52, 0, 0), frame(90, 95, jump[0], jump[1], 0, 0)]
        runs = {nm_: views_until(tape(fr10, ([(80, side10, px, q_)] if q_ else []), 51.), pend, 4) for nm_, q_ in (('none', 0.), ('partial', 10.), ('full', 30.))}
        req_ = [e_ for e_ in runs['none'][-2][3] if e_[2] == 0 and e_[1] == 'structural_request']
        if not (runs['none'] == runs['partial'] == runs['full'] and req_ and req_[0][0] == 100.):
            diff_ = next((i_ for i_, (a_, b_) in enumerate(zip(runs['none'], runs['full'])) if a_ != b_), None)
            bad.append(f"F10 side {side10}: an unnoticed fill changed the client state before its notice at 180 ms (first differing record {diff_}; "
                       f"structural request at 100 ms: {req_[:1]})")
        late = {nm_: views_until(tape(fr10, ([(80, side10, px, q_)] if q_ else []), 51.), pend, 5)[-2][1][side10]['position'] for nm_, q_ in (('none', 0.), ('full', 30.))}
        if not (late['none'] == 0. and late['full'] == (30. if side10 == 0 else -30.)):                    # the known position after the notice
            bad.append(f'F10 side {side10}: the notice at 180 ms does not reach the client by 200 ms ({late})')
    # ---- F7 boundaries and capacity
    pb = tape([frame(0, 0, 50, 52, 0)], [(137, 0, 50, 5), (136.9, 0, 50, 5)][::-1], 51.)          # a match just before and at the cancel effect (137)
    Eb = batch(pb, [[0, 0, 50, 30, 0, 58, 137, 0, 1]])
    if not ([f_[0] for f_ in Eb.fills] == [136.9]):
        bad.append(f'F7: the match around the cancel effect (fills {[f_[0] for f_ in Eb.fills]}, want only 136.9)')
    pa = tape([frame(0, 0, 50, 52, 0)], [(58, 0, 50, 5)], 51.); Ea = batch(pa, [[0, 0, 50, 30, 0, 58, 1e30, 0, 1]])
    if [f_[0] for f_ in Ea.fills] != [58.]:
        bad.append('F7: a match exactly at activation does not fill (activation is inclusive)')
    pn = tape([frame(0, 0, 50, 52, 0)], [(160, 0, 50, 10)], 51.); Bn = run(pn, POST_KEEP, until=6)
    if not (Bn.known_filled(0, 250) == 0 and Bn.known_filled(0, 260) == 10):
        bad.append('F7: the fill notice is not delayed 100 ms')
    Bq = Branch(tape([frame(0, 0, 50, 52, 0)], [], 51.)); Bq.act(0., 0, 'POST', Bq.view(0., 0)); Bq.E.fills.append((0., 99, 0, 10., 50., -1))      # planted: I = 10 beside an open 30-share bid
    try:
        Bq.check_position(200.); bad.append('F7: the reservation inequality accepts I + B = 40 > 30')
    except ValueError:
        pass
    Bd = copy.deepcopy(Bn); Bd.open[0] = None                                               # a planted double allocation: the reservation released early
    try:
        Bd.check_position(250.); bad.append('F7: the position check accepts a released reservation with an unacknowledged fill')
    except ValueError:
        pass
    pc = tape([frame(0, 0, 50, 52, 0)], [(70, 0, 50, 30)], 51.); Bc = run(pc, POST_KEEP); Lc = finish(Bc)
    if not (('cap' in [c_[2] for c_ in Lc['constraints'] if c_[1] == 0]) and len([o_ for o_ in Bc.E.o if o_['side'] == 0]) == 1
            and Lc['counts'].get('bid absent cap-blocked', 0) > 0 and Lc['counts'].get('bid absent feasible, POST', 0) == 1):
        bad.append(f"F7: cap saturation (a long 30 must stop a second bid; constraints {Lc['constraints'][:3]})")
    pp_ = tape([frame(0, 0, 50, 52, 0), frame(40, 41, 49, 50)], [], 51.); Bp = run(pp_, POST_KEEP, until=3)   # the ask falls to the bid's price before activation
    if not (Bp.E.o and Bp.E.o[0]['refused'] and ('refused', 100.) in Bp.closed.values()):
        bad.append('F7: a post-only refusal at activation (known at the first tick after it)')
    pe = tape([frame(0, 0, 50, 52, 0)], [], 51.); Be = Branch(pe); nt = int(np.ceil(END / STEP))
    for kk in range(nt):
        step(Be, kk, Policy(lambda v: 'POST' if (v['state'] == 'absent' and v['t'] >= END - 50) else ('WAIT' if v['state'] == 'absent' else 'KEEP'), 'late'))
    if not any(c_[2] == 'near_end' for c_ in Be.constraints):
        bad.append('F7: a near-END post is not refused')
    rm = pair(tape([frame(0, 0, 50, 52, 0)], [(500, 0, 50, 5)], None), POST_KEEP, 0, 0, 'post')
    if rm['status'] != 'missing_mark':
        bad.append(f"F7: a missing mark is not censored ({rm['status']})")
    pu = tape([frame(0, 0, 50, 52, 0), frame(200, 205, -1, 52, 0), frame(400, 405, 50, 52, 0)], [], 51.); Bu = run(pu, POST_KEEP); finish(Bu)
    if not any(f_[3] == 'feature_unavailable' for f_ in Bu.fallbacks):
        bad.append('F7: an unavailable book is not a counted fallback')
    # ---- F8 identity / support: a wrong horizon refused; a wrong action / state refused; a stale generation refused
    for nm_, fn_ in (('a policy horizon other than END', lambda: run(pe, Policy(lambda v: 'WAIT', 'h2', 2000.))),
                     ('a PULL with nothing live', lambda: Branch(pe).act(0., 0, 'PULL', Branch(pe).view(0., 0))),
                     ('a POST while post-pending', lambda: (lambda B_: (B_.act(0., 0, 'POST', B_.view(0., 0)), B_.act(50., 0, 'POST', B_.view(50., 0))))(Branch(pe))),
                     ('an undeclared pair kind', lambda: pair(pe, POST_KEEP, 0, 0, 'hold')),
                     ('a branch END other than 5,000 ms', lambda: Branch(pe, end=4000.)),
                     ('a saved fork state of another tick', lambda: pair(pe, POST_KEEP, 1, 0, 'post', state=fork_state(pe, POST_KEEP, 0, 0))),
                     ('a timing scenario other than the declared one', lambda: Branch(pe, settings=dict(cancel=50.))),
                     ('a PULL for a stale generation', lambda: (lambda B_: (B_.act(0., 0, 'POST', B_.view(0., 0)), B_.E.advance(100.),
                                                                           B_.act(100., 0, 'PULL', dict(B_.view(100., 0), gid=7))))(Branch(pe)))):
        try:
            fn_(); bad.append(f'F8: {nm_} was not refused')
        except (SystemExit, ValueError, AssertionError):
            pass
    # ---- F9 shared book: bid fill then ask fill reconcile; the second side's view sees the first side's command
    ps = tape([frame(0, 0, 50, 52, 0, 0)], [(100, 0, 50, 30), (700, 1, 52, 30)], 51.)
    pol9 = Policy(lambda v: ('POST' if (v['side'] == 0 or v['other'] != 'post_pending') else 'WAIT') if v['state'] == 'absent' else 'KEEP', 'ask_waits_for_bid')
    B9 = run(ps, pol9, episode=0); L9 = finish(B9)
    first_ask = min([d_[0] for d_ in B9.decisions if d_[1] == 1 and d_[3] == 'POST'], default=None)
    by_tick = {}
    for d_ in B9.decisions:
        by_tick.setdefault(d_[0], []).append(d_[1])
    if not all(v_[0] == (0 + int(t_ // STEP)) % 2 for t_, v_ in by_tick.items() if len(v_) == 2):
        bad.append('F9: the sides are not processed in the alternating order')
    if not (abs(L9['position']) < 1e-12 and abs(L9['value'] - 30. * (52. - 50.)) < 1e-9 and first_ask is not None and first_ask > 0):
        bad.append(f"F9: the shared book (position {L9['position']}, value {L9['value']}, first ask post {first_ask})")


def selftest():
    """Synthetic tapes only: an engine invariant (the position check, the trade-volume check) refusing a valid case is a control failure."""
    bad = []
    try:
        _controls(bad)
    except ValueError as e:
        bad.append(f'an engine invariant refused a valid synthetic case: {str(e)[:200]}')
    if bad:
        sys.exit(f'{MARK} SELFTEST FAILED: ' + ' | '.join(bad))
    print('branch engine selftests passed (synthetic only): P1 the port = the C kernel on 40 random tapes / order lists / actions; P2 adaptive runs = '
          'their batch replays; F1 WAIT posts later with a different history; F2 re-entry after the acknowledgement at a new price, with a cancel-'
          'pending fill; F3 queue priority kept, replacement behind; F4 fill ledger = cash + inventory value, pair difference, common pre-fork fills; '
          'F5 from one saved fork state the continuation reverses the keep advantage both ways; F6 causality; F10 unnoticed fills (zero / '
          'partial / full, both sides) leave views, structural requests and reservations unchanged before their notice; F7 cancel-effect boundary, delayed notice, cap, post-only refusal, near-END '
          'post, missing mark, unavailable book; F8 wrong horizon / action / state / pair kind refused; F9 shared book', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['traces']:
        traces()
    elif len(a) == 2 and a[0] == 'slice_v3':
        slice_v3(a[1])
    elif len(a) in (3, 4) and a[0] == 'slice':
        slice_check(a[1], int(a[2]), tag=a[3] if len(a) == 4 else '')
    else:
        sys.exit('usage: pf_branch_engine.py selftest | traces | slice <day> <n> [tag] | slice_v3 <day>')
