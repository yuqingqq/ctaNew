"""VENUE-MATCHED COMPARISON stage 2 (declaration.md): BASE and FLOOR on every entry-usable, resolved market of the five v3 days under
the venue rules — takes reach the venue after the speed bump, native BUY / SELL orders (1 USDT buy minimum, buy fees in shares, sells of
held tokens), measured exit timing, queue rule L, trade clocks M50 / M250, settlement 100 / 0 / 50, unresolved orders bounded.
With every venue feature off (OFF) the code reproduces the frozen v1 ledgers exactly (K2).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_exit_venue_20261125/vm_replay.py \
       selftest | day <day> | merge | check | report
"""
import collections
import copy
import gzip
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
R = H.parents[2]
sys.path.insert(0, str(H))
import venue_common as V  # noqa: E402

F, OD, MM, ER, TR, X, LB, BE, C = V.F, V.OD, V.MM, V.ER, V.TR, V.X, V.LB, V.BE, V.C
SR = OD.SR
AF = sys.modules['pf_ladder_async_fast']
QP = sys.modules['pf_queue_place_test']
M = X.M
OUT = X.DATA / 'take_exit_venue_s2_20261125'
EPS = V.EPS
VM_PROFILE = dict(place=float(BE.PLACE), cancel=float(BE.CANCEL), pending=float(BE.PENDING), slot_model='desired_levels')
OFF = dict(delay=58., shift=50., qrule='M', native=False, profile='ASYNC_FAST10')          # every venue feature off (K2 parity)
CONFIGS = {'VM208-M50': dict(delay=208., shift=50., qrule='L', native=True, profile='VM'),
           'VM258-M50': dict(delay=258., shift=50., qrule='L', native=True, profile='VM'),
           'VM208-M250': dict(delay=208., shift=250., qrule='L', native=True, profile='VM')}
PRIMARY = 'VM208-M50'
MIN_BUY_C = 100.          # a buy is worth at least 1 USDT
MIN_MSELL = 0.01          # a market sell is at least 0.01 share
YOUNG_MS = 100.
SCHEDULED = 1440
FIRST_DAY, DAY_CPU_CAP, TOTAL_CPU_CAP = '20261023', 3000., 12500.     # resource amendments R1 / R2 (declaration.md)
PARITY_EVERY = 40
SAME = ['fills', 'orders', 'commands', 'value', 'cash', 'position', 'signal_log', 'struct_log']
FILES = ['declaration.md', 'venue_common.py', 'vm_replay.py']


# ---------------------------------------------------------------------------------------------------------------- the path to the data end
def vm_path(W, ws, shift):
    """market_path's start and QP.build_path's frames / trades, run to min(298 s, data end) with no end mark required. data end = the
    last frame before the first frame gap > 1 s (or the last frame) + 1 s. -> (path, status, info); status complete / complete_no_mark /
    truncated, or the frozen exclusion."""
    t0, seen, none = M.market_start(W, ws)
    info = dict(ws=float(ws), wait_steps_frame_received=int(seen), wait_steps_no_frame=int(none))
    if t0 is None:
        return None, ('no_quotable_book' if seen else 'no_frame_received'), info
    k0 = int(np.searchsorted(np.asarray(W[0]['tg'], float), t0)); j0 = int(np.asarray(W[0]['jk'])[k0])
    info.update(t_start=float(t0), opening_wait_ms=float(t0 - ws), known_frame_receipt_rel_ms=float(np.asarray(W[0]['kf'], float)[j0] - ws))
    w = W[0]; t = t0
    j = int(np.searchsorted(w['kf'], t, side='right') - 1)
    if j < 0:
        return None, 'no_known_book', info
    bid, ask = float(W[0]['P5'][j, 0]), float(W[1]['P5'][j, 0])
    if not 0 <= bid < ask <= 100:
        return None, 'invalid_known_book', info
    E0 = ws + M.SUPPORTED_END
    tf = np.asarray(w['tf'], float)
    seg = tf[j:int(np.searchsorted(tf, E0, side='right'))]
    if len(seg) == 0:
        return None, 'no_usable_span', info
    gaps = np.flatnonzero(np.diff(seg) > 1000.)
    end = min(E0, float(seg[gaps[0]] if len(gaps) else seg[-1]) + 1000.)
    if end <= t + BE.STEP:
        return None, 'no_usable_span', info
    e = int(np.searchsorted(tf, end, side='right'))
    frames = np.column_stack([tf[j:e] - t, np.asarray(w['kf'])[j:e] - t, W[0]['P5'][j:e], W[0]['S5'][j:e], W[1]['P5'][j:e], W[1]['S5'][j:e]]).astype(float)
    tape = []
    for side in (0, 1):
        z = W[side]; tt = np.asarray(z['anc']) - shift - t; ix = np.flatnonzero((tt > 0) & (tt <= end - t))
        tape.extend((float(tt[i]), side, float(z['ptl'][i]), float(z['shl'][i]), int(i)) for i in ix)
    tape.sort(key=lambda x: (x[0], x[1], x[4]))
    trades = np.asarray([x[:4] for x in tape], float).reshape(-1, 4)
    mt, mm = QP.AC.valid_marks(w, 0)
    ix = int(np.searchsorted(mt, end, side='right') - 1)
    endpoint = ix >= 0 and end - mt[ix] <= 1000.
    mark = float(mm[ix]) if ix >= 0 else (bid + ask) / 2.
    p = dict(frames=frames, trades=trades, mark=mark, bid=bid, ask=ask, j=j, mid=(bid + ask) / 2.)
    if not M.whole_cents(p):
        return None, 'tick_unverified', info
    st = 'complete' if (end == E0 and endpoint) else ('complete_no_mark' if end == E0 else 'truncated')
    info.update(end_rel=float(end - t0), endpoint_mark=bool(endpoint), path_end_abs_ms=float(end))
    return p, st, info


# ---------------------------------------------------------------------------------------------------------------- native token book
class NativeBook:
    """YES / NO token balances, USDT cash, average cost and FIFO lots per token (declaration.md): a take's reducing part sells held tokens
    (fee in USDT), its opening part buys YES or NO (fee in shares, fee / p_native); maker fills sell held tokens (no fee); a maker fill
    beyond the held tokens is impossible natively and is dropped (counted as oversell)."""

    def __init__(self):
        self.bal = {'yes': 0., 'no': 0.}; self.cost = {'yes': 0., 'no': 0.}; self.lots = {'yes': collections.deque(), 'no': collections.deque()}
        self.cash = 0.; self.rt = collections.Counter(); self.oversell = 0.; self.fee_cash = 0.; self.fee_tokens_c = 0.

    @property
    def position(self):
        return self.bal['yes'] - self.bal['no']

    def _buy(self, tok, n, px, fee):
        if n <= EPS:
            return
        got = n - fee / px
        self.cash -= n * px; self.bal[tok] += got; self.cost[tok] += n * px; self.lots[tok].append([got, n * px / got]); self.fee_tokens_c += fee

    def _sell(self, tok, x, px, fee, kind):
        if x <= EPS:
            return
        self.cash += x * px - fee; self.fee_cash += fee
        b = self.bal[tok]; self.cost[tok] *= max(0., b - x) / b if b > EPS else 0.; self.bal[tok] = max(0., b - x)
        if self.bal[tok] <= EPS:
            self.bal[tok] = 0.; self.cost[tok] = 0.
        n, matched = x, 0.
        while n > EPS and self.lots[tok]:
            lot = self.lots[tok][0]; y = min(n, lot[0]); matched += y * lot[1]; lot[0] -= y; n -= y
            if lot[0] <= EPS:
                self.lots[tok].popleft()
        pnl = x * px - fee - matched
        for k in ('', kind + '_'):
            self.rt[k + 'net_c'] += pnl; self.rt[k + 'shares'] += x

    def fill(self, s, q, p, fee, maker):
        """One fill in YES space (side s 0 = buy YES / bid, 1 = sell YES / ask) at YES price p (cents); fee = the venue fee for q."""
        if maker:
            tok, px = ('yes', p) if s == 1 else ('no', 100. - p)
            x = min(q, self.bal[tok]); self.oversell += q - x
            self._sell(tok, x, px, 0., 'exit')
            return
        held, other, px_sell, px_buy = ('no', 'yes', 100. - p, p) if s == 0 else ('yes', 'no', p, 100. - p)
        r = min(q, self.bal[held]); o = q - r
        self._sell(held, r, px_sell, fee * r / q, 'take')
        self._buy(other, o, px_buy, fee * o / q)

    def settle(self, y):
        return self.cash + self.bal['yes'] * 100. * y + self.bal['no'] * 100. * (1. - y)

    def limits(self, slack=0.):
        """The exit price limits (YES space): the ask floor of held YES, the bid ceiling of held NO (sell NO >= its average cost)."""
        lim = {0: None, 1: None}
        if self.bal['yes'] > EPS:
            lim[1] = float(math.ceil(self.cost['yes'] / self.bal['yes'] - 1e-9)) - slack
        if self.bal['no'] > EPS:
            lim[0] = float(math.floor(100. - self.cost['no'] / self.bal['no'] + 1e-9)) + slack
        return lim


def fee_of_fill(rec_fees, gid, it):
    return rec_fees[int(gid)] if it < 0 else 0.


# ---------------------------------------------------------------------------------------------------------------- engine and branch
class VMEngine(SR.StrictIOCEngine):
    """StrictIOCEngine with a declared exit timing profile and queue rule: M (frozen frame cap) or L (no credit for cancels ahead)."""

    def __init__(self, path, end, bps, prof, qrule):
        super().__init__(path, end, bps)
        C.require(qrule in ('M', 'L'), f'queue rule {qrule}')
        self.prof = dict(prof); self.qrule = qrule; self._adv = False

    def advance(self, grid):
        self._adv = True
        try:
            super().advance(grid)
        finally:
            self._adv = False

    def shown(self, j, side, price):
        if self._adv and self.qrule == 'L':           # inside advance, shown() only caps the queue ahead (the frame cap)
            return np.nan
        return super().shown(j, side, price)


class VMBranch(F.FloorBranch):
    """FloorBranch (slack 0 FLOOR / out of reach BASE) on the VM engine; native=True: positions, floors and take minimums on tokens."""

    def __init__(self, path, end, ident, bps, host, take, slack=0., cfg=None):
        super().__init__(path, end, ident, bps, host, take, slack=slack)
        cfg = dict(cfg or OFF)
        prof = VM_PROFILE if cfg['profile'] == 'VM' else dict(AF.PROFILES[cfg['profile']])
        C.require(not self.E.o and not self.E.fills, 'the engine is replaced before any order')
        self.E = VMEngine(path, end, bps, prof, cfg['qrule']); self.E.strict = True
        self.prof = dict(prof); self.native = bool(cfg['native']); self.cfg = cfg; self._nb = (None, None); self._nv = (None, None)
        self.settings = dict(self.settings, venue_matched=dict(cfg, profile_values=prof))

    def _book(self, g, known):
        fee = {gid: f for _, gid, f in self.E.fees}
        nb = NativeBook()
        for t, gid, s, q, p, it in self.E.fills:
            if (t + BE.FILL_NOTICE <= g) if known else (t <= g):
                nb.fill(s, q, p, fee_of_fill(fee, gid, it), it >= 0)
        return nb

    def known_book(self, g):
        key = (g, len(self.E.fills))
        if self._nb[0] != key:
            self._nb = (key, self._book(g, True))
        return self._nb[1]

    def venue_book(self, g):
        key = (g, len(self.E.fills))
        if self._nv[0] != key:
            self._nv = (key, self._book(g, False))
        return self._nv[1]

    def reserve(self, g):
        I, B, A = super().reserve(g)
        if not self.native:
            return I, B, A
        return self.known_book(g).position, B, A

    def check_position(self, g):
        if not self.native:
            return super().check_position(g)
        Ik, Bk, Ak = self.reserve(g); Pp = self.venue_book(g).position
        C.require(Ik + Bk <= self.cap + LB.TOL and -Ik + Ak <= self.cap + LB.TOL, f'reservation inequality at {g}: I {Ik}, B {Bk}, A {Ak}, cap {self.cap}')
        C.require(Ik - Ak - 1e-6 <= Pp <= Ik + Bk + 1e-6, f'position check at {g}: native {Pp} outside [{Ik - Ak}, {Ik + Bk}]')
        C.require(all(len(self.desired(s)) <= self.levels for s in (0, 1)), f'more than {self.levels} desired generations on a side at {g}')

    def limits(self, g):
        if not self.native:
            return super().limits(g)
        if self._lim[0] != g:
            nb = self.known_book(g)
            C.require(abs(nb.position - self.reserve(g)[0]) < 1e-9, 'native book differs from the known position')
            self._lim = (g, nb.limits(self.slack))
        return self._lim[1]

    def signal_step(self, k, g):
        """take_replay.SignalBranch.signal_step; native=True adds the venue minimum: the opening part of a take is a buy worth >= 1 USDT,
        else it is cut to the reducing part (min_capped) or not sent (below_min)."""
        if not self.native:
            return super().signal_step(k, g)
        if not self.signal_take:
            return
        for gs in (1, 0):
            if not self.trig[gs][k]:
                continue
            s = 0 if gs == 1 else 1
            I, B, A = self.reserve(g)
            room = self.cap - I - B if s == 0 else self.cap + I - A
            q = float(math.floor(min(TR.TAKE_Q, room) + TR.EPS))
            row = dict(t=float(g), k=int(k), guard_side=gs, side=s, known_I=I, B=B, A=A, room=room, q=q, status='candidate', gid=None)
            if q < 1:
                row['status'] = 'no_room'
            elif self.take_open:
                row['status'] = 'pending_ioc'
            elif g + TR.TAKE_DELAY >= self.end:
                row['status'] = 'near_end'
            else:
                v = self.side_view(g, s)
                limit = v['ask'] if s == 0 else v['bid']
                row.update(limit=limit, book_received_ms=v['received'], known_frame=int(self.E.known))
                if not v['book_ok'] or g - v['received'] > 1000:
                    row['status'] = 'no_fresh_book'
                elif any(limit >= p if s == 0 else limit <= p for p in v['opp_prices']):
                    row['status'] = 'own_cross'
                else:
                    red = min(q, max(0., -I if s == 0 else I)); op = q - red; pn = limit if s == 0 else 100. - limit
                    if op > EPS and op * pn < MIN_BUY_C - 1e-9:
                        if red >= MIN_MSELL:
                            q = red; row.update(q=q, native='min_capped')
                        else:
                            row['status'] = 'below_min'
                    if row['status'] == 'candidate':
                        gid = self.E.submit_ioc(g, s, q, limit)
                        self.take_open.append(gid)
                        row.update(gid=gid, status='sent')
            self.sig_log.append(row)
        self.check_position(g)


def one_vm(p, cache, old, edges, bps, slack, cfg):
    """exit_replay.run_one (PASSIVE_EXIT) on the VM branch with the configuration's take delay."""
    orig, d0 = TR.SignalBranch, TR.TAKE_DELAY
    TR.SignalBranch = lambda *a, **k: VMBranch(*a, slack=slack, cfg=cfg, **k)
    TR.TAKE_DELAY = float(cfg['delay'])
    del F.LAST[:]
    try:
        rec = ER.run_one(p, cache, old, edges, bps, 'PASSIVE_EXIT')
    finally:
        TR.SignalBranch, TR.TAKE_DELAY = orig, d0
    C.require(len(F.LAST) == 1, 'one branch per run')
    C.require(all(abs(v['act'] - v['submit'] - cfg['delay']) < 1e-9 for v in rec['iocs'].values()), 'an IOC delay other than declared')
    rec['floor_log'] = TR.clean(F.LAST[0].floor_log)
    rec['vm_cfg'] = dict(cfg, slack=float(slack))
    del F.LAST[:]
    return json.loads(json.dumps(TR.clean(rec), allow_nan=False))


# ---------------------------------------------------------------------------------------------------------------- native accounting
def native_account(rec, y):
    """Settlement value of one VM ledger on tokens (declaration.md): total, round trips by closing kind, held tokens, C1 young-fill and C3
    cancel-pending contributions, oversell, and the unresolved bound at the path end."""
    C.require(y in (0., .5, 1.), f'payout {y}')
    fee = F.fee_map(rec)
    orders = {int(o['gid']): o for o in rec['orders']}
    end = float(rec['end'])
    nb = NativeBook()
    young = cp = young_sh = cp_sh = 0.
    taker = maker = 0.
    for t, gid, s, q, p, it in rec['fills']:
        if it >= 0:
            tok, px = ('yes', p) if s == 1 else ('no', 100. - p)
            x = min(q, nb.bal[tok]); pay = 100. * (y if tok == 'yes' else 1. - y); contrib = x * (px - pay)
            o = orders[int(gid)]
            if t - o['act'] < YOUNG_MS:
                young += contrib; young_sh += x
            if o['req'] is not None and t >= o['req']:
                cp += contrib; cp_sh += x
            maker += q
        else:
            taker += q
        nb.fill(s, q, p, fee_of_fill(fee, gid, it), it >= 0)
    total = nb.settle(y)
    held_cost = sum(l[0] * l[1] for tok in ('yes', 'no') for l in nb.lots[tok])
    held_val = nb.bal['yes'] * 100. * y + nb.bal['no'] * 100. * (1. - y)
    # identities: round trips + held = total; cash = -sum(buys) + sum(sells)
    C.require(abs(nb.rt['net_c'] + (held_val - held_cost) - total) < 1e-6 * max(1., abs(total) + 1e4), 'round trips + held != total')
    # unresolved: resting exits (sell held tokens) and pending takes at the path end
    lo = hi = 0.; n_unres = 0
    bal = dict(nb.bal)
    for o in rec['orders']:
        if o['why'] == 'IOC' or o['refused'] or o['rem'] <= EPS or o['remove'] <= end:
            continue
        tok, px = ('yes', o['price']) if o['side'] == 1 else ('no', 100. - o['price'])
        x = min(o['rem'], bal[tok]); bal[tok] -= x
        d = x * (px - 100. * (y if tok == 'yes' else 1. - y)); lo += min(0., d); hi += max(0., d); n_unres += int(x > EPS)
    for gid, a in rec['iocs'].items():
        if a['act'] > end:
            b2 = copy.deepcopy(nb); b2.fill(a['side'], a['q'], a['limit'], TR.T.fee_c(a['limit'], a['q'], 200.), False)
            d = b2.settle(y) - total; lo += min(0., d); hi += max(0., d); n_unres += 1
    return dict(total_c=total, rt_net_c=nb.rt['net_c'], rt_exit_net_c=nb.rt['exit_net_c'], rt_take_net_c=nb.rt['take_net_c'],
                rt_shares=nb.rt['shares'], rt_exit_shares=nb.rt['exit_shares'], rt_take_shares=nb.rt['take_shares'],
                held_yes=nb.bal['yes'], held_no=nb.bal['no'], held_settle_net_c=held_val - held_cost, held_cost_c=held_cost,
                fee_cash_c=nb.fee_cash, fee_token_cost_c=nb.fee_tokens_c, taker_shares=taker, maker_shares=maker, oversell_shares=nb.oversell,
                young_exit_c=young, young_exit_shares=young_sh, cancel_pending_exit_c=cp, cancel_pending_exit_shares=cp_sh,
                total_without_young_c=total - young, unresolved_lo_c=total + lo, unresolved_hi_c=total + hi, unresolved_orders=n_unres, payout=y)


# ---------------------------------------------------------------------------------------------------------------- the day run
def bound():
    F.bound()
    for f in FILES:
        C.require(subprocess.check_output(['git', 'show', 'HEAD:' + str((H / f).relative_to(R))], cwd=R) == (H / f).read_bytes(), f'{f} is not committed')


def worlds_all(day, mids, src, off, reg):
    """The unchanged loader for every market of a batch -> (m, W or None, loader status)."""
    data = src.load(day, list(mids), off)
    seen = set()
    for W in src.pairs(data):
        m = int(W[0]['m']); seen.add(m)
        yield m, W, ('precision_not_2' if int(reg.loc[m, 'decimal_precision']) != 2 else 'loaded')
    for m in mids:
        if m not in seen:
            yield int(m), None, 'loader_excluded'


def day_run(day):
    resource.setrlimit(resource.RLIMIT_CPU, (int(DAY_CPU_CAP) + 60, int(DAY_CPU_CAP) + 61))
    bound()
    C.require(day in OD.DAYS3, f'{day} is not a declared day')
    fn = OUT / f'day_{day}.json'
    C.require(not fn.exists(), f'{fn.name} exists (write-once)')
    if day != FIRST_DAY:
        pj = json.loads((H / 'stage2_projection.json').read_text())
        C.require(pj['projected_cpu_s'] <= TOTAL_CPU_CAP, 'the first-day projection exceeds the cap')   # R2: re-evaluated against the amended cap
    OUT.mkdir(parents=True, exist_ok=True)
    v1 = {int(r['mid']): r for r in json.loads((F.OUT / 'summary.json').read_text())['rows'] if r['day'] == day}
    v3 = {int(r['mid']): r for r in json.loads((OD.OUT3 / 'summary.json').read_text())['rows'] if r['setting'] == 'STRICT58' and r['day'] == day}
    edges, src = TR.T.P.frozen_edges(), MM.KP.Tape()
    reg = MM.registry(day).set_index('mid'); rg = pd.read_parquet(MM.S1 / day / 'registry.parquet').set_index('mid')
    off = src.offset(day); t0 = time.process_time()
    rows, census, parity, path_parity = [], [], [], collections.Counter()
    n_complete = 0
    for h, g in reg.groupby('hour'):
        for m, W, lst in worlds_all(day, [int(x) for x in g.index], src, off, reg):
            ws = float(reg.loc[m, 'ws']) * 1000.
            c = dict(day=day, mid=m, hour=int(h), status=lst)
            census.append(c)
            if lst != 'loaded':
                continue
            if ws + 300000. >= OD.FWD_MS:
                c['status'] = 'boundary_censored_forward_tail'; continue
            y = V.payout_of(rg.loc[m]); c['payout'] = y
            paths = {}
            for sh in sorted({cf['shift'] for cf in CONFIGS.values()}):
                paths[sh] = vm_path(W, ws, sh)
            p50, st, info = paths[50.]
            c['status'] = st
            p_ref, st_ref, info_ref = M.market_path(W, ws)                   # K1: the frozen complete path
            if st_ref == 'complete':
                n_complete += 1
                same = st == 'complete' and all(np.array_equal(np.asarray(p50[k]), np.asarray(p_ref[k])) for k in ('frames', 'trades')) and \
                    all(p50[k] == p_ref[k] for k in ('mark', 'bid', 'ask', 'j', 'mid')) and info['end_rel'] == info_ref['end_rel']
                path_parity['complete'] += 1; path_parity['identical'] += int(same)
                C.require(m in v1, f'{m} complete but not in the v1 population')
            if p50 is None:
                continue
            cache = OD.make_cache3(float(info['t_start']), float(info['end_rel']), OD.SEG_OF[day])
            row = dict(day=day, mid=m, hour=int(h), status=st, payout=y, end_rel=info['end_rel'], arms={})
            for name, cf in CONFIGS.items():
                p, st_c, info_c = paths[cf['shift']]
                C.require(p is not None and info_c['end_rel'] == info['end_rel'], f'{m}: {name} path differs in span')
                old = dict(end=float(info_c['end_rel']), ident=MM.ident_of(day, m, ws, info_c, p))
                for arm, slack in V.ARMS.items():
                    rec = one_vm(p, cache, old, edges, 200., slack, cf)
                    rec.update(day=day, mid=m, hour=int(h), arm=arm, config=name, payout=y, path_status=st)
                    sha = F.write_once_gz(OUT / f'{m}_{name}_{arm}.json.gz', rec)
                    a = native_account(rec, y)
                    a.update(sha256=sha, sig_status=dict(collections.Counter(r_['status'] for r_ in rec['signal_log'])),
                             min_capped=sum(r_.get('native') == 'min_capped' for r_ in rec['signal_log']),
                             ioc_status=rec['metrics']['ioc_status'], exit_posts=len(rec['floor_log']))
                    row['arms'][f'{name}|{arm}'] = a
            if st_ref == 'complete' and (n_complete - 1) % PARITY_EVERY == 0:     # K2: every venue feature off == the v1 ledgers
                old = dict(end=float(info_ref['end_rel']), ident=MM.ident_of(day, m, ws, info_ref, p_ref))
                refs = dict(BASE=OD.OUT3 / f'{m}_STRICT58.json.gz', FLOOR=F.OUT / f'{m}_FLOOR.json.gz')
                for arm, slack in V.ARMS.items():
                    r0 = one_vm(p_ref, cache, old, edges, 200., slack, OFF)
                    ref = json.loads(gzip.decompress(refs[arm].read_bytes()))
                    if arm == 'BASE':                                    # the v3 BASE ledger predates the floor-only struct_log keys
                        r0['struct_log'] = [{k: v for k, v in x.items() if k not in F.FLOOR_KEYS} for x in r0['struct_log']]
                    parity.append(dict(mid=m, arm=arm, same={k: r0[k] == ref[k] for k in SAME}))
            rows.append(row)
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    spent = time.process_time() - t0
    if day == FIRST_DAY:
        proj = spent * len(OD.DAYS3) * 1.1
        pfn = H / 'stage2_projection.json'
        C.require(not pfn.exists(), 'stage2_projection.json exists (write-once)')
        TR.dump(pfn, dict(asof='2026-11-25', first_day=day, spent_cpu_s=spent, projected_cpu_s=proj, cap_cpu_s=TOTAL_CPU_CAP, go=proj <= TOTAL_CPU_CAP))
    TR.dump(fn, dict(asof='2026-11-25', day=day, census=census, rows=rows, parity=parity, path_parity=dict(path_parity), cpu_s=spent, peak_rss_mib=F.rss_mib()))
    print(f'{day}: {len(census)} scheduled, {len(rows)} replayed, path parity {dict(path_parity)}, '
          f'K2 parity {all(all(x["same"].values()) for x in parity)} ({len(parity)}), cpu {spent:.0f} s', flush=True)


# ---------------------------------------------------------------------------------------------------------------- selftest
def selftest():
    # native book: buy YES 50 @ 52, fee 48 c -> 50 - 48/52 tokens at cost 2600; floor = ceil(2600 / 49.0769) = 53
    nb = NativeBook(); nb.fill(0, 50., 52., 48., False)
    assert abs(nb.bal['yes'] - (50 - 48 / 52)) < 1e-12 and abs(nb.cash + 2600.) < 1e-9 and nb.limits()[1] == 53. and nb.limits()[0] is None
    # a maker sale of 46 @ 46, then an opposite take of 10 @ 50 that sells the rest and buys NO with the remainder
    nb.fill(1, 46., 46., 0., True)
    left = nb.bal['yes']; nb.fill(1, 10., 50., 10., False)
    r, o = left, 10. - left
    assert nb.bal['yes'] == 0. and abs(nb.bal['no'] - (o - (10. * o / 10.) / 50.)) < 1e-12
    assert abs(nb.cash - (-2600. + 46 * 46. + r * 50. - 10. * r / 10. - o * 50.)) < 1e-9
    # settlement identity on both outcomes and a tie; round trips + held = total
    for y in (0., .5, 1.):
        held_cost = sum(l[0] * l[1] for t_ in ('yes', 'no') for l in nb.lots[t_]); held_val = nb.bal['no'] * 100. * (1 - y)
        assert abs(nb.rt['net_c'] + held_val - held_cost - nb.settle(y)) < 1e-9
    # oversell: a maker sale beyond held tokens is dropped and counted
    nb2 = NativeBook(); nb2.fill(0, 10., 40., 0., False); nb2.fill(1, 12., 45., 0., True)
    assert nb2.bal['yes'] == 0. and abs(nb2.oversell - 2.) < 1e-12
    # short floor: hold NO bought at 60 (YES bid 40): sell NO >= 60 -> YES bid ceiling 40
    nb3 = NativeBook(); nb3.fill(1, 10., 40., 0., False); assert nb3.limits()[0] == 40. and nb3.limits()[1] is None
    # queue L: a shown-size drop must not move a resting order up; M must (synthetic two-frame book, one trade at our price)
    fr = np.full((3, 22), -1.); fr[:, 0] = [0., 100., 200.]; fr[:, 1] = fr[:, 0]
    for i, sz in enumerate((10., 2., 2.)):
        fr[i, 2], fr[i, 7], fr[i, 12], fr[i, 17] = 40., 5., 45., sz        # bid 40 x 5, ask 45 x size
    path = dict(frames=fr, trades=np.array([[250., 1., 45., 3.]]))
    res = {}
    for q in ('M', 'L'):
        E = VMEngine(path, 300., 200., VM_PROFILE, q); gid = E.add_order(1, 45., 0., 5., 0)
        E.advance(300.); res[q] = sum(f[3] for f in E.fills if f[1] == gid)
    assert res['M'] == 1. and res['L'] == 0., res                          # M: ahead 10 -> 2, trade 3 fills 1; L: ahead stays 10
    # the minimum rule: buying 1 share at 52 c is below 1 USDT; 2 shares at 52 c are not
    assert 1 * 52. < MIN_BUY_C <= 2 * 52.
    # native_account on a synthetic ledger: identities, a tie, the unresolved bound order
    rec = dict(fills=[[0., 0, 0, 50., 52., -1], [500., 1, 1, 20., 60., 0]], ioc_results=[dict(gid=0, filled=50., fee_c=48.)], end=1000., iocs={},
               orders=[dict(gid=0, why='IOC', side=0, price=52., rem=0., act=58., remove=58., refused=False, req=None),
                       dict(gid=1, why=None, side=1, price=60., rem=10., act=480., remove=1e30, refused=False, req=None)])
    for y in (0., .5, 1.):
        a = native_account(rec, y)
        assert a['unresolved_lo_c'] <= a['total_c'] + 1e-9 <= a['unresolved_hi_c'] + 2e-9 and a['unresolved_orders'] == 1
        assert abs(a['young_exit_c'] - 20. * (60. - 100. * y)) < 1e-9                # the 500 ms fill is 20 ms after activation
    try:
        native_account(rec, .3)
    except ValueError:
        pass
    else:
        raise AssertionError('payout 0.3 accepted')
    print('vm selftest PASS')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'day':
        day_run(sys.argv[2])
    else:
        dict(selftest=selftest)[cmd]()
