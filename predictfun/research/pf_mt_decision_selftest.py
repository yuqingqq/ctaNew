"""Selftest of pf_mt_decision (TOP-3 DECISION PATHS; P-2026-005, session aea5bfb9, 2026-11-23). SYNTHETIC data only, in a temporary
directory; every expected value comes from an independent reference implementation in this file (rolling sums by convolution, math.erf,
its own as-of book lookup), never from the module's functions. World (YES space; two measurement days):
 D1 20261024: 101 (BOOK, ws 00:00Z: its pre-window needs the Oct 23 23Z padding file), 102; D2 20261025: 201 (BOOK, 10:00Z), 202, 203
 (boundary_censored). BTC: a 100 ms random walk per segment (sigma 1 bps / sqrt s), a 3 s 'gap' patch in 202, a stale ok row in 201;
 btc_20261023 holds 23:50-23:59:59.9 plus one 22:00Z row (before the padding: clipped and counted). Books lag p_fair by 2 s (+ noise),
 frames every 100-400 ms, a 1.6 s gap (stale), a bad_frame, a one-sided and a crossed frame per market.
 rank1 = POSITIVE CONTROL 1: a taker that buys iff g_buy >= 1.5 c and r_1s > 0 (sells iff g_sell >= 1.5 c and r_1s < 0), deciding every
   100 ms on the features at t - 50 ms; 10 % two-leg records; OTHER is often its counterparty.
 rank2 = POSITIVE CONTROL 2: a maker quoting p_fair(t) -+ 2 c - 1 c per 100 shares of inventory, rounded to the cent, re-quoting every 100 ms,
   hit by random takers (inventory-dependent sides). BOOK lifecycle: one order per fill, planted posting / removal orders priced by the same
   rule (cancel_possible windows at top-decile adverse p_fair moves, exhausted_only windows at random) and links whose reprices track p_fair.
 rank3 = noise: random makes / takes, a take after we (outside its span), a src_status row (pair excluded), rows in the censored market and
   one self-trade. OTHER (rank 4) trades everywhere and must never surface; five rank1 rows on the ID day are skipped and counted.
Also: refusals with the reason asserted (forbidden days, a missing BTC grid, an address outside the top 3, identities ranks, a forbidden-day
BTC row, a look-ahead age_ms, write-once, a forbidden-day fills_valued row), book parity with pf_mt_econ.mid_at, a causality falsifier
(later BTC rows, book frames - values AND arrival times - and own fills perturbed: no earlier feature changes; a planted +150 ms look-ahead
does), lifecycle INVALID parity, and a CPU micro-benchmark of the production chunk / fit code for the real-run projection.
Usage (run_light): pf_mt_decision_selftest.py"""
import json
import math
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_mt_common as C  # noqa: E402
import pf_mt_decision as M  # noqa: E402
import pf_mt_econ as E  # noqa: E402

A1, A2, A3 = M.TOP3
OTHER = '0x0000000000000000000000000000000000000Aaa'
CPA, CPA2, CPB = ('0xCc0000000000000000000000000000000000000a', '0xCc0000000000000000000000000000000000000b',
                  '0xCc0000000000000000000000000000000000000c')
D1, D2 = '20261024', '20261025'
D1MS, D2MS = M.day_ms(D1), M.day_ms(D2)
WS = {101: D1MS, 102: D1MS + 300_000, 201: D2MS + 36_000_000, 202: D2MS + 36_300_000, 203: D2MS + 36_600_000}
DAYOF = {101: D1, 102: D1, 201: D2, 202: D2, 203: D2}
BOOK = {D1: [101], D2: [201]}
MSTATUS = {101: 'ok', 102: 'poll_incomplete', 201: 'ok', 202: 'ok', 203: 'boundary_censored'}   # 102: kept and counted (accepted rule)
SIG_FLOOR, SQ2 = 0.2, math.sqrt(2.)
ERF = np.vectorize(math.erf, otypes=[float])
NAN = float('nan')
RES = []
SID = [0]
PA = {'str': pa.string(), 'int64': pa.int64(), 'float64': pa.float64(), 'bool': pa.bool_()}


def check(name, cond, detail=''):
    RES.append((name, bool(cond)))
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  [{str(detail)[:400]}]'), flush=True)


def section(name, fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # a crash is a failure of that section; the other sections still run
        tb = traceback.format_exc().strip().splitlines()
        check(f'{name}: ran without an exception', False, f'{type(e).__name__}: {e} | {" / ".join(tb[-4:])}')
        return None


def quiet(*a, **k):
    return None


def nsid():
    SID[0] += 1
    return f'018f0000-0000-7000-8000-{SID[0]:012d}'


def write(path, cols, types):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({c: pa.array(v, type=PA[types[c]]) for c, v in cols.items()}), path)


# ------------------------------------------------------------------ independent reference (never the module's code) ------------------------------------------------------------------
class Ref:
    """BTC reference on one contiguous 100 ms segment: usability, rolling sums by convolution, math.erf."""

    def __init__(self, S):
        self.G0, self.n = int(S['g'][0] // 100), len(S['g'])
        assert (np.diff(S['g']) == 100).all()
        mid, st, age = S['mid'], S['st'], S['age']
        self.use = (st == 'ok') & (age <= 1000) & np.isfinite(mid) & (mid > 0)
        self.lm = np.full(self.n, NAN)
        self.lm[self.use] = np.log(mid[self.use])
        r = np.full(self.n, NAN)
        r[1:] = 1e4 * (self.lm[1:] - self.lm[:-1])
        self.S600 = np.convolve(np.where(np.isfinite(r), r * r, 0.), np.ones(600), mode='valid')        # S600[j] = sum r2[j .. j + 599]
        self.U601 = np.convolve(self.use.astype(float), np.ones(601), mode='valid')                       # U601[j] = sum use[j .. j + 600]

    def feats(self, x, ws, we):
        x = np.asarray(x, np.int64)
        i = x // 100 - self.G0
        ic = np.clip(i, 600, self.n - 1)
        iw = int(ws // 100 - self.G0)
        pre = x < ws
        valid = (i >= 600) & (i < self.n) & (self.U601[ic - 600] == 601.) & (pre | (0 <= iw < self.n and bool(self.use[iw]))) & (we - x > 0)
        lw = self.lm[iw] if 0 <= iw < self.n else NAN
        r1 = 1e4 * (self.lm[ic] - self.lm[ic - 10])
        rws = np.where(pre, 0., 1e4 * (self.lm[ic] - lw))
        sig = np.maximum(np.sqrt(self.S600[ic - 599] / 60.), SIG_FLOOR)
        tau = (we - x) / 1000.
        pf = 0.5 * (1. + ERF(rws / (sig * np.sqrt(np.maximum(tau, 1e-9))) / SQ2))
        return valid, np.where(valid, r1, NAN), np.where(valid, pf, NAN)


def book_ref(B, x):
    """As-of book: the last frame with upd_ms <= x (frames unique per market); usable = no bad_frame, two-sided, bid < ask, age <= 1 s."""
    upd = B['upd_ms']
    j = np.searchsorted(upd, x, 'right') - 1
    jc = np.maximum(j, 0)
    bb, ba = B['best_bid'][jc], B['best_ask'][jc]
    use = (j >= 0) & (B['status'][jc] != 'bad_frame') & (B['n_bid'][jc] > 0) & (B['n_ask'][jc] > 0) & np.isfinite(bb) & np.isfinite(ba) & \
        (bb < ba) & (x - upd[jc] <= 1000)
    return use, np.where(use, bb, NAN), np.where(use, ba, NAN)


# ------------------------------------------------------------------ the synthetic world ------------------------------------------------------------------
def btc_segment(rng, t0, t1):
    n = (t1 - t0) // 100
    return dict(g=t0 + 100 * np.arange(n, dtype=np.int64), mid=np.exp(np.log(110_000.) + np.cumsum(rng.normal(0., math.sqrt(0.1), n)) / 1e4),
                age=rng.integers(0, 100, n).astype(np.int64), st=np.full(n, 'ok', dtype=object))


def btc_cols(S, k):
    return dict(g_ms=S['g'][k], bid=S['mid'][k] - .05, ask=S['mid'][k] + .05, mid=S['mid'][k], n_upd=np.full(int(k.sum()), 3, np.int64), age_ms=S['age'][k],
                status=S['st'][k].tolist())


BTC_T = dict(g_ms='int64', bid='float64', ask='float64', mid='float64', n_upd='int64', age_ms='int64', status='str')


def make_book(rng, m, ref):
    ws, we = WS[m], WS[m] + 300_000
    t = [ws - 304_000]
    while t[-1] < we + 1_000:
        t.append(t[-1] + int(rng.integers(100, 401)))
    t = np.array(t, np.int64)
    k = int(np.searchsorted(t, ws + 100_000))
    t[k:] += 1_600                                                   # a planted 1.6 s gap (stale reads inside it)
    n = len(t)
    v, _, pf = ref.feats((t - 2_000) // 100 * 100, ws, we)
    c = np.empty(n)
    last = 0.5
    for i in range(n):
        last = pf[i] if v[i] else last
        c[i] = last
    spread = rng.choice([1, 2, 3], n, p=[.5, .3, .2])
    bid = np.clip(np.rint(100. * c - spread / 2. + rng.normal(0., .7, n)), 1, 96).astype(np.int64)
    ask = bid + spread
    st = np.where(np.r_[False, np.diff(t) > 1_000], 'gap', 'ok').astype(object)
    nb, na = np.full(n, 5, np.int64), np.full(n, 5, np.int64)
    bb, ba = bid / 100., ask / 100.
    kb, k1, k2 = (int(np.searchsorted(t, ws + s_)) for s_ in (30_000, 60_000, 90_000))
    st[kb] = 'bad_frame'
    na[k1], ba[k1] = 0, NAN
    ba[k2] = bb[k2]                                                  # crossed (locked)
    return dict(upd_ms=t, best_bid=bb, best_ask=ba, n_bid=nb, n_ask=na, status=st, best_bid_sh=np.round(rng.uniform(20, 500, n), 2),
                best_ask_sh=np.where(na > 0, np.round(rng.uniform(20, 500, n), 2), 0.), recv_ns=t * 1_000_000 + rng.integers(5, 50, n) * 1_000_000)


TOB_T = {c: t for c, t in C.SCHEMAS['tob'].items()}


def tob_cols(books, d):
    cols = {c: [] for c in TOB_T}
    for m in sorted(books):
        if DAYOF[m] != d:
            continue
        B = books[m]
        n = len(B['upd_ms'])
        for c, v in (('day', [d] * n), ('mid', [m] * n), ('upd_ms', B['upd_ms']), ('recv_ns', B['recv_ns']), ('conn', ['c0'] * n), ('order_count', [10] * n),
                     ('n_bid', B['n_bid']), ('n_ask', B['n_ask']), ('best_bid', B['best_bid']), ('best_bid_sh', B['best_bid_sh']),
                     ('best_ask', B['best_ask']), ('best_ask_sh', B['best_ask_sh']), ('pend_bid_usd', [0.] * n), ('pend_ask_usd', [0.] * n),
                     ('status', B['status'].tolist())):
            cols[c].extend(list(v))
    return cols


def rec(m, settle, taker, ts, legs, st='ok', day=None):
    return dict(day=day or DAYOF[m], mid=m, sid=nsid(), settle=int(settle), taker=taker, ts=int(ts), legs=legs, st=st)


def world_a1(rng, REF, books, recs):
    """POSITIVE CONTROL 1 (taker): returns {mid: (n_buy, n_sell)}."""
    out = {}
    for m in (101, 102, 201, 202):
        ws, we = WS[m], WS[m] + 300_000
        ts = np.arange(ws - 300_000, we, 100, dtype=np.int64)
        x = ts - 50
        v, r1, pf = REF[m].feats(x, ws, we)
        bu, bid, ask = book_ref(books[m], x)
        ok = v & bu
        with np.errstate(invalid='ignore'):
            gb, gs = 100. * (pf - ask), 100. * (bid - pf)
            buy, sell = ok & (gb >= 1.5) & (r1 > 0), ok & (gs >= 1.5) & (r1 < 0)
        for k in np.flatnonzero(buy | sell):
            s_ = 1 if buy[k] else -1
            px = float(ask[k] if s_ == 1 else bid[k])
            legs = [(OTHER if rng.random() < .3 else CPA, round(px, 2), round(float(rng.uniform(5, 50)), 2))]
            if rng.random() < .1:
                legs.append((CPA2, round(min(px + .01, .99), 2) if s_ == 1 else round(max(px - .01, .01), 2), round(float(rng.uniform(5, 50)), 2)))
            recs.append(rec(m, ts[k] + int(rng.integers(5, 96)), A1, s_, legs))
        out[m] = (int(buy.sum()), int(sell.sum()))
    return out


def a2_quote(pf, inv, side):
    return int(np.rint(100. * pf + (-2. if side == 1 else 2.) - inv / 100.))


def world_a2(rng, REF, recs):
    """POSITIVE CONTROL 2 (maker): returns {mid: list of (t, settle, side, p, q, inv_before)}."""
    fills = {}
    for m in (101, 102, 201, 202):
        ws, we = WS[m], WS[m] + 300_000
        ts = np.arange(ws - 300_000, we, 100, dtype=np.int64)
        v, _, pf = REF[m].feats(ts, ws, we)
        u, dl, qq = rng.random(len(ts)), rng.integers(1, 91, len(ts)), np.round(rng.uniform(5, 40, len(ts)), 2)
        inv, F_ = 0., []
        for k in range(len(ts)):
            if not v[k]:
                continue
            pb, pa_ = .03 * (1. - math.tanh(inv / 300.)), .03 * (1. + math.tanh(inv / 300.))
            side = 1 if u[k] < pb else (-1 if u[k] < pb + pa_ else 0)
            if side == 0:
                continue
            c = a2_quote(pf[k], inv, side)
            if not 1 <= c <= 99:
                continue
            settle = int(ts[k] + dl[k])
            recs.append(rec(m, settle, OTHER if rng.random() < .2 else CPB, -side, [(A2, c / 100., float(qq[k]))]))
            F_.append((int(ts[k]), settle, side, c / 100., float(qq[k]), inv))
            inv += side * float(qq[k])
        fills[m] = F_
    return fills


def world_a3(rng, recs):
    for m in (101, 102, 201, 202, 203):
        ws, we = WS[m], WS[m] + 300_000
        st = 'boundary_censored' if MSTATUS[m] == 'boundary_censored' else 'ok'     # only the censored market's rows are flagged
        for t in range(ws - 300_000, we, 100):
            u = rng.random()
            px = round(float(np.clip(.5 + rng.normal(0, .05), .05, .95)), 2)
            if u < .004:
                recs.append(rec(m, t + int(rng.integers(1, 99)), CPB, int(rng.choice([-1, 1])), [(A3, px, round(float(rng.uniform(5, 30)), 2))], st=st))
            elif u < .007:
                recs.append(rec(m, t + int(rng.integers(1, 99)), A3, int(rng.choice([-1, 1])), [(CPA, px, round(float(rng.uniform(5, 30)), 2))], st=st))
    recs.append(rec(102, WS[102] + 302_000, A3, 1, [(CPA, .5, 7.)]))                         # after we: outside the span
    recs.append(rec(202, WS[202] + 100_000, CPB, 1, [(A3, .5, 5.)], st='src_status'))       # one flagged row -> pair excluded
    recs.append(rec(201, WS[201] + 50_000, A3, 1, [(A3, .5, 10.)]))                         # self-trade
    for m in (101, 201):                                                                     # OTHER's own trading
        for j in range(50):
            recs.append(rec(m, WS[m] + 1_000 * j + 7, OTHER, 1, [(CPA, .5, 3.)]))
            recs.append(rec(m, WS[m] + 1_000 * j + 9, CPB, -1, [(OTHER, .5, 3.)]))
    for j in range(5):                                                                       # the ID day: skipped, counted
        recs.append(rec(9001, M.day_ms(C.ID_DAY) + 3_600_000 + j * 1_000, A1, 1, [(CPA, .5, 1.)], day=C.ID_DAY))


FV_T = dict(day='str', mid='int64', sid='str', role='str', addr='str', s='int64', q='float64', p='float64', settle_ms='int64', self_trade='bool',
            status='str', hash='str', t_ms='int64')


def fv_cols(recs):
    cols = {c: [] for c in FV_T}
    for r in recs:
        for i, (mk, p, q) in enumerate(r['legs']):
            for role, addr, s in (('maker', mk, -r['ts']), ('taker', r['taker'], r['ts'])):
                for c, v in (('day', r['day']), ('mid', r['mid']), ('sid', r['sid']), ('role', role), ('addr', addr), ('s', s), ('q', q), ('p', p),
                             ('settle_ms', r['settle']), ('self_trade', mk == r['taker']), ('status', r['st']), ('hash', f'h{r["sid"][-6:]}{i}{role[0]}'),
                             ('t_ms', r['settle'])):
                    cols[c].append(v)
    return cols


ORD_T = dict(C.SCHEMAS['orders'])


def order_rows(m, h, signer, side, ptick, first_hi, post_lo, rset, rlo, rhi, status='ok', persist=(-1, -1), size=10.):
    out = []
    for sc in C.SCENARIOS:
        out.append(dict(scenario=sc, mid=m, mk_hash=h, mk_signer=signer, y_side=side, y_ptick=int(ptick), n_fills=1 if persist[0] < 0 else 2,
                        cum_shares=size, first_hi_ms=int(first_hi), last_hi_ms=int(first_hi), size_lb=size, size_exact=NAN, size_class='observed',
                        post_lo_ms=int(post_lo if sc != 'S_NOPRIO' else -1), post_hi_ms=int(first_hi), post_class='constrained' if sc != 'S_NOPRIO' else 'unresolved',
                        post_method='capacity', persist_lo_ms=int(persist[0]), persist_hi_ms=int(persist[1]),
                        persist_class='observed' if persist[0] >= 0 else 'unresolved', ahead_lo=NAN, ahead_hi=NAN, queue_class='unresolved',
                        removal_set=rset, removal_lo_ms=int(rlo), removal_hi_ms=int(rhi), removal_class='constrained', removal_method='end_open',
                        status=status))
    return out


LNK_T = dict(mid='int64', mk_signer='str', y_side='str', kind='str', prev_hash='str', next_hash='str', prev_ptick='int64', next_ptick='int64',
             prev_last_hi_ms='int64', next_first_hi_ms='int64', gap_ms='int64', x_latest='bool', label='str', status='str')


def inv_at(F_, x):
    return float(sum(s * q for (_, settle, s, _, q, _) in F_ if settle < x))


def life_world(rng, REF, a2f):
    """BOOK lifecycle rows: rank2's fill orders, planted posting / removal orders and links (expected values recorded)."""
    life, exp = {}, dict(removal={}, links=[])
    for m in (101, 201):
        ws, we = WS[m], WS[m] + 300_000
        rows, F_ = [], a2f[m]
        for j, (t, settle, side, p, q, _) in enumerate(F_):
            per = (settle - 3_000, settle) if j % 10 == 0 else (-1, -1)
            rows += order_rows(m, f'h2_{m}_{j}', A2, 'bid' if side == 1 else 'ask', round(p * 100), settle, t - 100, 'exhausted', -1, -1, persist=per, size=q)
        xs = np.arange(ws, we - 2_100, 100, dtype=np.int64)
        v0, _, p0 = REF[m].feats(xs, ws, we)
        v1, _, p1 = REF[m].feats(xs + 2_000, ws, we)
        vf, _, pff = REF[m].feats(xs - 1_000, ws, we)
        ok = v0 & v1 & vf
        dp = 100. * (p1 - p0)
        CXL = 'cancelled_or_expired|exhausted'
        planted = []                                                 # (lo, hi, side, removal_set, kind)
        for side, sgn in (('bid', 1), ('ask', -1)):
            adv = np.where(ok, -sgn * dp, -np.inf)
            top = np.argsort(-adv, kind='mergesort')[:max(1, int(ok.sum() // 10))]
            planted += [(int(xs[i]), int(xs[i]) + 2_000, side, CXL, 'live') for i in top[::max(1, len(top) // 30)][:30]]
            rnd = rng.choice(np.flatnonzero(ok), 30, replace=False)
            planted += [(int(xs[i]), int(xs[i]) + 2_000, side, 'exhausted', 'live') for i in rnd]
        xp = np.arange(ws - 250_000, ws - 5_000, 100, dtype=np.int64)  # pre-ws windows: p_fair is flat there, BTC is not
        vp = REF[m].feats(xp, ws, we)[0] & REF[m].feats(xp + 2_000, ws, we)[0] & REF[m].feats(xp - 1_000, ws, we)[0]
        pre_i = rng.choice(np.flatnonzero(vp), 10, replace=False)
        planted += [(int(xp[i]), int(xp[i]) + 2_000, ('bid', 'ask')[j % 2], CXL if j < 5 else 'exhausted', 'pre') for j, i in enumerate(pre_i)]
        planted += [(ws - 1_000, ws + 1_000, 'bid', CXL, 'straddle'), (we - 1_000, we + 1_000, 'ask', CXL, 'beyond'),
                    (ws - 299_000, ws - 4_000, 'bid', CXL, 'long_pool'),     # 50 pre placements < MIN_PLACE: the market-day pool
                    (ws - 310_000, ws - 4_000, 'ask', CXL, 'long_none')]     # wider than the pre phase: no placement (a status)
        kinds = Counter()
        for j, (x0, x1, side, rset, kind) in enumerate(planted):
            fh = int(x0 - 1_000)
            v_, _, pf_ = REF[m].feats(np.array([fh]), ws, we)
            if not v_[0]:
                continue
            c = a2_quote(pf_[0], inv_at(F_, fh), 1 if side == 'bid' else -1)
            if not 1 <= c <= 99:
                continue
            rows += order_rows(m, f'hp_{m}_{j}', A2, side, c, fh, fh - 3_000, rset, x0, x1)
            kinds[kind] += 1
        exp['removal'][m] = kinds
        if m == 101:                                                 # rank3: pulls at EVERY usable live placement (no reaction to moves)
            u = np.arange(3_000, 6_000 - 20, dtype=np.int64)
            xu = ws - 300_000 + 100 * u
            xu = xu[REF[m].feats(xu, ws, we)[0] & REF[m].feats(xu + 2_000, ws, we)[0]]
            for j, x in enumerate(xu):
                rows += order_rows(m, f'h3_{m}_{j}', A3, ('bid', 'ask')[j % 2], 50, int(x) - 1_000, int(x) - 4_000, CXL, int(x), int(x) + 2_000)
            exp['r3'] = (m, xu)
        rows += order_rows(m, f'hbad_{m}', A2, 'bid', 50, ws + 10_000, ws, 'exhausted', -1, -1, status='identity_conflict')
        rows += order_rows(m, f'hoth_{m}', OTHER, 'ask', 60, ws + 20_000, ws, 'cancelled_or_expired|exhausted', ws + 20_000, ws + 22_000)
        links = []
        for j in range(60):
            u = int(rng.choice(xs[ok]))
            gap = 100 * int(rng.integers(10, 51))
            vv, _, pp = REF[m].feats(np.array([u, u + gap]), ws, we)
            if not vv.all() or u + gap >= we:
                continue
            dt_ = int(np.rint(100. * (pp[1] - pp[0])))
            nxt = int(np.clip(50 + dt_, 1, 99))
            links.append(dict(mid=m, mk_signer=A2, y_side=('bid', 'ask')[j % 2], kind='reprice' if nxt != 50 else 'refill', prev_hash=f'l{j}a',
                              next_hash=f'l{j}b', prev_ptick=50, next_ptick=nxt, prev_last_hi_ms=u, next_first_hi_ms=u + gap, gap_ms=gap, x_latest=True,
                              label='descriptive', status='ok'))
            exp['links'].append((m, 100. * (pp[1] - pp[0]), nxt - 50))
        links.append(dict(links[0], status='invalid_order', next_hash='lbad'))
        links.append(dict(links[1], x_latest=False, next_hash='lold'))
        links.append(dict(links[2], mk_signer=OTHER, next_hash='loth'))
        life[m] = (rows, links)
    return life, exp


def build_world(root, rng):
    REF_A, REF_B = None, None
    SA = btc_segment(rng, D1MS - 600_000, D1MS + 900_000)
    SB = btc_segment(rng, WS[201] - 600_000, WS[201] + 1_200_000)
    gp = (SB['g'] >= WS[202] + 120_000) & (SB['g'] < WS[202] + 123_000)
    SB['st'][gp], SB['mid'][gp], SB['age'][gp] = 'gap', NAN, -1
    SB['age'][SB['g'] == WS[201] + 180_000] = 1_500
    REF_A, REF_B = Ref(SA), Ref(SB)
    REF = {101: REF_A, 102: REF_A, 201: REF_B, 202: REF_B, 203: REF_B}
    # BTC files. The Oct 23 file adds one row at 22:00Z (before the padding: clipped, counted) and marks its last 5 s 'gap' (age -1); the
    # Oct 24 file also carries 23:59:00-23:59:59.9 of Oct 23 (600 overlap rows: its first 5 'no_data', age -1). The overlap rule must rebuild
    # the true path: the usable row wins (50 rows filled from the Oct 24 file), else the own-day file.
    a23 = btc_cols(SA, SA['g'] < D1MS)
    tail = np.asarray(a23['g_ms']) >= D1MS - 5_000
    for c in ('mid', 'bid', 'ask'):
        a23[c] = np.where(tail, NAN, a23[c])
    a23['age_ms'], a23['status'] = np.where(tail, -1, a23['age_ms']), ['gap' if t_ else s_ for t_, s_ in zip(tail, a23['status'])]
    extra = dict(g_ms=[D1MS - 7_200_000], bid=[1e5 - .05], ask=[1e5 + .05], mid=[1e5], n_upd=[1], age_ms=[0], status=['ok'])
    write(root / 'btc' / 'btc_20261023.parquet', {c: list(extra[c]) + list(a23[c]) for c in BTC_T}, BTC_T)
    a24 = btc_cols(SA, SA['g'] >= D1MS - 60_000)
    head = np.asarray(a24['g_ms']) < D1MS - 59_500
    for c in ('mid', 'bid', 'ask'):
        a24[c] = np.where(head, NAN, a24[c])
    a24['age_ms'], a24['status'] = np.where(head, -1, a24['age_ms']), ['no_data' if h_ else s_ for h_, s_ in zip(head, a24['status'])]
    write(root / 'btc' / f'btc_{D1}.parquet', a24, BTC_T)
    write(root / 'btc' / f'btc_{D2}.parquet', btc_cols(SB, np.ones(len(SB['g']), bool)), BTC_T)
    books = {m: make_book(rng, m, REF[m]) for m in WS}
    for d in (D1, D2):
        write(root / f'tob_{d}.parquet', tob_cols(books, d), TOB_T)
        ms = [m for m in sorted(WS) if DAYOF[m] == d]
        write(root / f'markets_{d}.parquet', dict(day=[d] * len(ms), mid=ms, slug=[f'btc-updown-5m-{WS[m] // 1000}' for m in ms], ws=[WS[m] // 1000 for m in ms],
                                                  we=[WS[m] // 1000 + 300 for m in ms], outcome=['Up'] * len(ms), status=[MSTATUS[m] for m in ms]),
              dict(day='str', mid='int64', slug='str', ws='int64', we='int64', outcome='str', status='str'))
    recs = []
    a1 = world_a1(rng, REF, books, recs)
    a2f = world_a2(rng, REF, recs)
    world_a3(rng, recs)
    life, exp = life_world(rng, REF, a2f)
    o = rng.permutation(len(recs))
    write(root / 'econ' / 'fills_valued.parquet', fv_cols([recs[i] for i in o]), FV_T)
    (root / 'econ' / 'identities.json').write_text(json.dumps(dict(ex_ante=dict(top10_shares=[dict(rank=1, addr=A1), dict(rank=2, addr=A2),
                                                                                                dict(rank=3, addr=A3), dict(rank=4, addr=OTHER)]))))
    for m, (rows, links) in life.items():
        write(root / 'life' / f'{DAYOF[m]}_{m}_orders.parquet', {c: [r[c] for r in rows] for c in ORD_T}, ORD_T)
        write(root / 'life' / f'{DAYOF[m]}_{m}_x_links.parquet', {c: [r[c] for r in links] for c in LNK_T}, LNK_T)
    return dict(REF=REF, books=books, recs=recs, a1=a1, a2f=a2f, exp=exp, SA=SA, SB=SB)


# ------------------------------------------------------------------ the checks ------------------------------------------------------------------
def rd(out, name):
    return pd.read_csv(out / name)


def pick(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k].astype(str) == str(v)).to_numpy()
    return df[m]


def main_run(root, W):
    out = root / 'out' / 'decision'
    c0 = time.process_time()
    rec_ = M.run(inp=root, out=out, days=(D1, D2), book=BOOK, btc_dir=root / 'btc', log=quiet)
    cpu = time.process_time() - c0
    check('run: write-once output published, partial / fit directories removed',
          out.is_dir() and not list(out.parent.glob('.decision*')) and not list(root.glob('.decision_fit_*')), sorted(p.name for p in out.parent.iterdir()))
    return out, rec_, cpu


def info(tag, df, cols):
    print(f'INFO {tag}: ' + json.dumps(df[[c for c in cols if c in df.columns]].round(4).to_dict('records'), default=str), flush=True)


def checks_controls(out, W):
    TH = rd(out, 'take_threshold.csv')
    info('control1 thresholds (rank1, all days, r1_aligned)', TH[(TH['scope'] == 'rank1') & (TH['day'].astype(str) == 'all') & (TH['slice'] == 'r1_aligned')],
         ['L', 'side', 'threshold_c', 'rate_above', 'rate_below', 'n_above', 'n_below', 'e_above', 'e_below', 'status'])
    SK_ = rd(out, 'post_skew.csv')
    info('control2 skew (rank2)', SK_[(SK_['scope'] == 'rank2') & SK_['clock'].isin(['fill', 'hi'])], ['L', 'clock', 'side', 'n', 'a_c', 'b_c_per_100sh', 'resid_sd'])
    RM_ = rd(out, 'removal.csv')
    info('removal (rank2, adverse, L 0)', RM_[(RM_['scope'] == 'rank2') & (RM_['metric'] == 'adverse') & (RM_['scenario'] == M.PRIMARY) & (RM_['L'] == 0)],
         ['phase', 'group', 'n_orders', 'obs_mean', 'null_mean', 'null_sd', 'z', 'p_one_sided', 'mean_percentile'])
    LK_ = rd(out, 'links.csv')
    info('links (rank2, reprice)', LK_[(LK_['scope'] == 'rank2') & (LK_['kind'] == 'reprice')], ['L', 'side', 'n', 'ols_slope_dtick_on_dpf', 'sign_agree', 'gap_p50'])
    LG_ = rd(out, 'take_logit.csv')
    info('logit (rank1)', LG_[LG_['scope'] == 'rank1'], ['L', 'side', 'status', 'iterations', 'n', 'events', 'b_g_c', 'b_r_1s_bps', 'pseudo_r2'])
    for side in ('buy', 'sell'):
        for day in ('all', D1, D2):
            r = pick(TH, scope='rank1', day=day, L=50, side=side, slice='r1_aligned')
            ok = len(r) == 1 and abs(r['threshold_c'].iloc[0] - 1.5) <= 0.5 + 1e-9 and r['rate_above'].iloc[0] >= 0.8 and r['rate_below'].iloc[0] <= 0.05
            check(f'CONTROL 1 {side} {day}: threshold within one 0.5 c bucket of 1.5 c, rate above >= 0.8, below <= 0.05', ok,
                  r[['threshold_c', 'rate_above', 'rate_below', 'n_above', 'n_below', 'status']].to_dict('records'))
    H2 = rd(out, 'take_hazard_2d.csv')
    h = pick(H2, scope='rank1', day='all', L=50, side='buy', table='g_x_r1sign')
    h = h[h['bucket'].str.endswith('|pos')]
    lo_edge = np.array([float(b.split('|')[0].strip('[<>=').split(',')[0]) for b in h['bucket']])
    up = h[(lo_edge >= 1.5) & ~h['bucket'].str.startswith('<')]
    dn = h[(lo_edge < 1.5) | h['bucket'].str.startswith('<')]
    check('CONTROL 1 exact replication: every r_1s > 0 g bucket at or above 1.5 c rates >= 0.99, every one below <= 0.01',
          len(up) > 0 and len(dn) > 0 and (up['rate'] >= .99).all() and (dn['rate'] <= .01).all(),
          dict(up=up[['bucket', 'n', 'rate']].head(4).to_dict('records'), dn=dn[dn['rate'] > .01][['bucket', 'n', 'rate']].head(4).to_dict('records')))
    st = rd(out, 'statuses.csv')
    nb, ns = sum(v[0] for v in W['a1'].values()), sum(v[1] for v in W['a1'].values())
    got = {s: int(pick(st, stage=f'take_records_{s}', scope='rank1', status='in_span')['n'].sum()) for s in ('buy', 'sell')}
    check('CONTROL 1 take records (two-leg records counted once) in span = the taker\'s decisions', got == dict(buy=nb, sell=ns), (got, nb, ns))
    SK = rd(out, 'post_skew.csv')
    for clock in ('fill', 'hi'):
        for side, a0 in (('bid', -2.), ('ask', 2.)):
            r = pick(SK, scope='rank2', day='all', L=0, clock=clock, scenario='-', side=side)
            ok = len(r) == 1 and abs(r['a_c'].iloc[0] - a0) <= .5 and abs(r['b_c_per_100sh'].iloc[0] + 1.) <= .5
            check(f'CONTROL 2 {clock} {side}: offset {a0:+g} c and skew -1 c / 100 sh recovered within +-0.5 c', ok,
                  r[['n', 'a_c', 'b_c_per_100sh', 'resid_sd']].to_dict('records'))
    CL = json.loads((out / 'claims_decision.json').read_text())['claims']

    def claim(cid, scope, side):
        x = [c for c in CL if c['claim'] == cid and c['scope'] == scope and c['side'] == side]
        return x[0] if x else None
    c = claim('take_rate_above_gap_threshold_ge_10x_below', 'rank1', 'buy/r1_aligned')
    check('claim: rank1 buy 10x threshold holds at L 50 with its n, every L evaluated', c is not None and c['per_L']['50']['holds'] is True and
          c['n_L_evaluated'] == 3 and c['per_L']['50']['n_above'] > 0, c)
    c = claim('quotes_centred_on_p_fair', 'rank2', 'fill')
    check('claim: rank2 quotes centred on p_fair within 0.5 c at L 0 (n recorded)', c is not None and c['per_L']['0']['holds'] is True and
          c['per_L']['0']['n_bid'] > 100, c)
    c = claim('quotes_skew_against_inventory', 'rank2', 'fill')
    check('claim: rank2 quotes skew against inventory at L 0', c is not None and c['per_L']['0']['holds'] is True, c)
    c = claim('next_maker_fill_reduces_position', 'rank2', 'maker')
    check('claim: rank2 next maker fill reduces |position| (inventory-dependent hits)', c is not None and c['per_L']['-']['holds'] is True, c)
    c = claim('take_rate_above_gap_threshold_ge_10x_below', 'rank3', 'buy/all')
    check('claim: the random taker (rank3) claim is computed with its n at every L (no assertion on its value: ~50 random takes)',
          c is not None and len(c['per_L']) == 3, c)
    RM = rd(out, 'removal.csv')
    kw2 = dict(scope='rank2', L=0, scenario=M.PRIMARY)
    rc = pick(RM, **kw2, phase='live', group='cancel_possible', metric='adverse')
    re_ = pick(RM, **kw2, phase='live', group='exhausted_only', metric='adverse')
    rd_ = pick(RM, **kw2, phase='live', group='cancel_minus_exhausted', metric='adverse')
    check('removal: planted top-decile adverse live cancels beyond the phase-matched null (p <= 0.01, >= 200 draws)',
          len(rc) == 1 and rc['p_one_sided'].iloc[0] <= .01 and rc['draws'].iloc[0] >= 200 and rc['n_orders'].iloc[0] >= 50, rc.to_dict('records'))
    check('removal: random-time exhausted live windows sit inside the null (mean percentile 0.35-0.65)',
          len(re_) == 1 and .35 <= re_['mean_percentile'].iloc[0] <= .65, re_.to_dict('records'))
    check('removal: cancel minus exhausted beyond its paired null (p <= 0.01)', len(rd_) == 1 and rd_['p_one_sided'].iloc[0] <= .01, rd_.to_dict('records'))
    K2 = sum((W['exp']['removal'][m] for m in W['exp']['removal']), Counter())
    for ph_, want in (('live', K2['live']), ('pre', K2['pre'] + K2['long_pool'])):
        tot = int(pick(RM, **kw2, phase=ph_, metric='adverse')[lambda x: x['group'] != 'cancel_minus_exhausted']['n_orders'].sum())
        check(f'removal: every planted {ph_} order with a usable window and a placement is measured once (OTHER never)', tot == want, (tot, want))
    rp = pick(RM, **kw2, phase='pre', group='cancel_possible', metric='abs')
    rr = pick(RM, **kw2, phase='pre', group='cancel_possible', metric='adverse_r')
    check('removal: pre-ws windows are drawn from the pre phase only (p_fair flat there: observed and null |dp| exactly 0; BTC null moves)',
          len(rp) == 1 and rp['obs_mean'].iloc[0] == 0 and rp['null_mean'].iloc[0] == 0 and len(rr) == 1 and rr['null_sd'].iloc[0] > 0,
          (rp.to_dict('records'), rr.to_dict('records')))
    st_ = rd(out, 'statuses.csv')
    rs_ = pick(st_, stage='removal', scope='rank2', L=M.PRIMARY)
    rn_ = pick(st_, stage='removal_null', scope='rank2', L=M.PRIMARY)
    got = {k: int(rs_[rs_['status'] == k]['n'].sum()) for k in ('cancel_possible/pre/straddles_ws', 'cancel_possible/live/beyond_we')}
    got.update({k: int(rn_[rn_['status'] == k]['n'].sum()) for k in ('market_day', 'no_phase_placement')})
    want = {'cancel_possible/pre/straddles_ws': K2['straddle'], 'cancel_possible/live/beyond_we': K2['beyond'], 'market_day': K2['long_pool'],
            'no_phase_placement': K2['long_none']}
    check('removal statuses: straddling-ws and beyond-we windows, the market-day pool and no-placement windows counted',
          got == want and all(v >= 1 for v in want.values()), (got, want))
    # ---- the phase-matching case: rank3 pulls at every usable live placement of market 101 (no reaction to any move) ----
    m3, xu = W['exp']['r3']
    ws3, we3 = WS[m3], WS[m3] + 300_000
    R3 = W['REF'][m3]
    uu = np.arange(0, 6_000 - 20, dtype=np.int64)
    x0 = ws3 - 300_000 + 100 * uu
    v0, _, p0 = R3.feats(x0, ws3, we3)
    v1, _, p1 = R3.feats(x0 + 2_000, ws3, we3)
    okp = v0 & v1
    pop = np.abs(100. * (p1 - p0))
    obs = float(np.abs(100. * (R3.feats(xu + 2_000, ws3, we3)[2] - R3.feats(xu, ws3, we3)[2])).mean())
    rng = np.random.default_rng(77)
    pmis, plive = pop[okp], pop[okp & (uu >= 3_000)]
    reps_mis = np.array([pmis[rng.integers(0, len(pmis), len(xu))].mean() for _ in range(500)])
    reps_fix = np.array([plive[rng.integers(0, len(plive), len(xu))].mean() for _ in range(500)])
    p_mis, z_mis = (1 + (reps_mis >= obs).sum()) / 501., (obs - reps_mis.mean()) / reps_mis.std(ddof=1)
    z_fix = (obs - reps_fix.mean()) / reps_fix.std(ddof=1)
    r3 = pick(RM, scope='rank3', L=0, scenario=M.PRIMARY, phase='live', group='cancel_possible', metric='abs')
    print(f'INFO phase case: n {len(xu)} obs |dp| {obs:.4f} c; mismatched null mean {reps_mis.mean():.4f} p {p_mis:.4f} z {z_mis:.1f}; '
          f'reference matched z {z_fix:.2f}; module ' + json.dumps(r3[['n_orders', 'obs_mean', 'null_mean', 'null_sd', 'z', 'p_one_sided',
                                                                        'mean_percentile']].round(5).to_dict('records')), flush=True)
    check('phase case: a phase-MISMATCHED control (draws over [ws - 300 s, we)) gives a false "moves inside random live pulls" signal (p <= 0.01, z > 3)',
          p_mis <= .01 and z_mis > 3, (p_mis, z_mis))
    check('phase case: the module\'s phase-matched control gives none (|z| < 3, p > 0.05, observed mean = the reference)',
          len(r3) == 1 and abs(r3['z'].iloc[0]) < 3 and r3['p_one_sided'].iloc[0] > .05 and abs(r3['obs_mean'].iloc[0] - obs) < 1e-6 and
          r3['n_orders'].iloc[0] == len(xu), (r3.to_dict('records'), obs, len(xu)))
    c = claim('removal_windows_move_beyond_null', 'rank3', f'{M.PRIMARY}/cancel_possible/live')
    check('phase case: claim removal_windows_move_beyond_null is False for rank3 at L 0', c is not None and c['per_L']['0']['holds'] is False, c)
    c = claim('removal_windows_adverse_beyond_null', 'rank2', f'{M.PRIMARY}/cancel_possible/live')
    check('claim: rank2 planted adverse cancels flagged at L 0 (phase-matched)', c is not None and c['per_L']['0']['holds'] is True, c)
    LK = rd(out, 'links.csv')
    lr = pick(LK, scope='rank2', day='all', L=0, kind='reprice')
    check('links: rank2 reprices track p_fair (slope 0.8-1.2, sign agreement >= 0.9, per side)', len(lr) == 2 and lr['ols_slope_dtick_on_dpf'].between(.8, 1.2).all()
          and (lr['sign_agree'] >= .9).all(), lr[['side', 'n', 'ols_slope_dtick_on_dpf', 'sign_agree']].to_dict('records'))
    n_lnk = int(pick(LK, scope='rank2', day='all', L=0)['n'].sum())
    check('links: invalid-status, not-latest and OTHER links excluded', n_lnk == len(W['exp']['links']), (n_lnk, len(W['exp']['links'])))
    LG = rd(out, 'take_logit.csv')
    r = pick(LG, scope='rank1', side='buy', L=50)
    check('logit: rank1 buy fitted with a positive gap coefficient (near-separation: max_iter or converged)', len(r) == 1 and r['b_g_c'].iloc[0] > 0 and
          r['status'].iloc[0] in ('converged', 'max_iter', 'pass_cap'), r.to_dict('records'))
    check('logit: every pooled model ran (24 rows, pooled present)', len(LG) == 24 and (LG['scope'] == 'pooled').sum() == 6, len(LG))


def checks_statuses(out, rec_, W):
    st = rd(out, 'statuses.csv')
    p3 = pick(st, stage='pairs', scope='rank3', day=D2)
    check('pairs: rank3 202 fill_status_src_status and 203 market_boundary_censored excluded with statuses',
          set(p3['status']) >= {'fill_status_src_status', 'market_boundary_censored'}, p3.to_dict('records'))
    o3 = int(pick(st, stage='take_records_buy', scope='rank3', day=D1, status='outside_span')['n'].sum())
    check('take after we counted outside_span (rank3, 1)', o3 == 1, o3)
    gr = pick(st, stage='grid_rows')
    for s_ in ('btc_invalid', 'book_stale', 'book_frame_status', 'book_one_sided', 'book_crossed'):
        check(f'grid status {s_} counted', int(gr[gr['status'] == s_]['n'].sum()) > 0, gr.groupby('status')['n'].sum().to_dict())
    cnt = rec_['counts']
    check('ID-day rows skipped and counted (5)', cnt.get('fv_rows_top3_other_days') == 5, cnt.get('fv_rows_top3_other_days'))
    check('BTC row before the Oct 23 23Z padding clipped and counted (1)', cnt.get('btc_rows_before_padding') == 1, cnt.get('btc_rows_before_padding'))
    check('BTC overlap: 600 rows in two files, 50 own-day gap rows filled by the other file\'s usable rows, 30 age -1 rows missing (the gap patch)',
          (cnt.get('btc_overlap_rows'), cnt.get('btc_overlap_filled_from_other_file'), cnt.get('btc_rows_age_missing')) == (600, 50, 30),
          {k: cnt.get(k) for k in ('btc_overlap_rows', 'btc_overlap_filled_from_other_file', 'btc_rows_age_missing')})
    check('poll_incomplete market kept and counted (market 102: 3 pairs)', cnt.get('pairs_poll_incomplete_kept') == 3, cnt.get('pairs_poll_incomplete_kept'))
    check('self-trade take record counted (1)', cnt.get('take_records_self_trade') == 1, cnt.get('take_records_self_trade'))
    n3 = sum(1 for r in W['recs'] if r['day'] in (D1, D2) for (mk, _, _) in r['legs'] for a in (mk, r['taker']) if a in (A1, A2, A3))
    check('fills_valued top-3 rows on the measurement days = the world count', cnt.get('fv_rows_top3', 0) - cnt.get('fv_rows_top3_other_days', 0) == n3,
          (cnt.get('fv_rows_top3'), n3))
    ps = pick(st, stage='posting', status='post_lo_unbounded')
    check('posting: S_NOPRIO unbounded lo / mid counted as statuses', int(ps['n'].sum()) > 0, ps.head(3).to_dict('records'))
    oi = pick(st, stage='orders', scope='rank2', status='order_invalid')
    check('orders: identity_conflict order counted invalid (2 BOOK markets)', int(oi['n'].sum()) == 2, oi.to_dict('records'))
    H1 = rd(out, 'take_hazard_1d.csv')
    rest = pick(H1, scope='rank2', day='all', L=0, side='buy', table='rest')
    check('rest: rank2 BOOK rows carry resting-order evidence (bid / ask / both buckets present)',
          set(rest['bucket']) & {'bid', 'ask', 'both'} and 'na' in set(rest['bucket']), rest[['bucket', 'n']].to_dict('records'))
    names = sorted(p.name for p in out.iterdir())
    blob = ''.join((out / n_).read_text() for n_ in names)
    check('OTHER never surfaces in any output (address string absent)', OTHER.lower() not in blob.lower(), 'found')
    sc = set()
    for n_ in names:
        if n_.endswith('.csv'):
            df = pd.read_csv(out / n_, nrows=200000)
            if 'scope' in df.columns:
                sc |= set(df['scope'].astype(str))
    check('every scope label is rank1-3 / pooled', sc <= set(M.SCOPES) and len(sc) > 0, sc)
    AS = rd(out, 'address_summary.csv')
    check('address_summary: 4 rows, the contract addresses in rank order', list(AS['address']) == list(M.TOP3) + ['pooled'], list(AS['address']))
    exp_tl = expected_timelines(W)
    got_tl = sorted(n_ for n_ in names if n_.startswith('timeline_'))
    check('timelines: the declared deterministic choice (2 per address here)', got_tl == exp_tl, (got_tl, exp_tl))
    for n_ in got_tl:
        if n_.startswith('timeline_rank1_'):
            tl = pd.read_csv(out / n_)
            m = int(n_.split('_')[-1].split('.')[0])
            nb = sum(1 for r in W['recs'] if r['taker'] == A1 and r['mid'] == m and r['ts'] == 1)
            check(f'timeline {n_}: take_buy buckets = the taker\'s buys there and BTC overlaid', int(tl['take_buy'].sum()) == nb and tl['btc_mid'].notna().mean() > .9,
                  (int(tl['take_buy'].sum()), nb))
    D = json.loads((out / 'decision.json').read_text())
    ca = D.get('constants_accepted', {})
    check('receipt: inputs hashed, the accepted constants printed, the pf_mt_btc semantics and the phase-matched null recorded',
          all('sha256' in v for v in D['inputs'].values()) and D['declared']['N_DRAWS'] >= 200 and '40ee591ea' in D['rules']['btc_semantics'] and
          'SAME PHASE' in D['rules']['removal_null'] and set(D['btc_files']) == {'20261023', D1, D2} and
          (ca.get('sig_60_floor_bps_per_sqrt_s'), ca.get('btc_stale_after_ms'), ca.get('threshold_split_min_rows_per_side'),
           ca.get('logistic_gap_clip_c')) == (0.2, 1000, 200, 20.0) and 'kept and counted' in str(ca.get('poll_incomplete')), (sorted(D), ca))


def expected_timelines(W):
    """The declared choice, recomputed from the world's records."""
    rows, tk, mkr = Counter(), {}, Counter()
    for r in W['recs']:
        if r['day'] not in (D1, D2):
            continue
        for (mk, _, _) in r['legs']:
            for role, a in (('maker', mk), ('taker', r['taker'])):
                if a in M.TOP3:
                    rows[(a, r['day'], r['mid'])] += 1
                    if role == 'taker':
                        tk.setdefault((a, r['day'], r['mid']), set()).add(r['sid'])
                    else:
                        mkr[(a, r['day'], r['mid'])] += 1
    bad = {(A3, D2, 202), (A3, D2, 203)}
    out = []
    for i, a in enumerate(M.TOP3):
        P = [k for k in rows if k[0] == a and k not in bad]
        bk = sorted((k for k in P if k[2] in BOOK[k[1]]), key=lambda k: (-rows[k], k[1], k[2]))
        nb = [k for k in P if k[2] not in BOOK[k[1]]]
        t1 = sorted((k for k in nb if len(tk.get(k, ())) > 0), key=lambda k: (-len(tk[k]), k[1], k[2]))
        chosen = ([bk[0]] if bk else []) + ([t1[0]] if t1 else [])
        t2 = sorted((k for k in nb if mkr[k] > 0 and (not t1 or k != t1[0])), key=lambda k: (-mkr[k], k[1], k[2]))
        chosen += [t2[0]] if t2 else []
        out += [f'timeline_{M.SCOPES[i]}_{k[1]}_{k[2]}.csv' for k in chosen]
    return sorted(out)


def refusals(root, W):
    base = root / 'refuse'
    base.mkdir()
    k = [0]

    def fresh():
        k[0] += 1
        return base / f'o{k[0]}' / 'decision'

    def expect(name, fn, frag, exc=ValueError):
        try:
            fn()
            check(f'refused: {name}', False, 'no refusal')
        except exc as e:
            check(f'refused: {name} (reason asserted)', frag in str(e), str(e))
        except Exception as e:
            check(f'refused: {name}', False, f'{type(e).__name__}: {e}')

    kw = dict(book=BOOK, log=quiet)
    expect('a forbidden day (20261028)', lambda: M.run(inp=root, out=fresh(), days=('20261028',), btc_dir=root / 'btc', **kw), 'is not a measurement day')
    expect('the GT day as a measurement day', lambda: M.run(inp=root, out=fresh(), days=(D1, C.NOV8), btc_dir=root / 'btc', **kw), 'is not a measurement day')
    expect('a forbidden BTC file path (before any access)', lambda: M.btc_path(root / 'btc', '20261028'), 'not an allowed bookTicker day', C.Forbidden)
    b2 = base / 'btc_missing'
    shutil.copytree(root / 'btc', b2)
    (b2 / f'btc_{D2}.parquet').unlink()
    expect('a missing BTC grid', lambda: M.run(inp=root, out=fresh(), days=(D1, D2), btc_dir=b2, **kw), 'missing BTC grid')
    expect('an address outside the top 3', lambda: M.run(inp=root, out=fresh(), days=(D1, D2), addrs=(A1, A2, OTHER), btc_dir=root / 'btc', **kw),
           "is not one of the contract's ex-ante top 3")
    r2 = base / 'root_ident'
    shutil.copytree(root, r2, ignore=shutil.ignore_patterns('out', 'refuse'))
    (r2 / 'econ' / 'identities.json').write_text(json.dumps(dict(ex_ante=dict(top10_shares=[dict(rank=1, addr=A1), dict(rank=2, addr=A2),
                                                                                              dict(rank=3, addr=OTHER), dict(rank=4, addr=A3)]))))
    expect('identities.json ranks 1-3 differ from the contract', lambda: M.run(inp=r2, out=fresh(), days=(D1, D2), btc_dir=r2 / 'btc', **kw),
           'identities.json ex-ante ranks 1-3 differ')
    b3 = base / 'btc_forbidden'
    shutil.copytree(root / 'btc', b3)
    T = pq.read_table(b3 / f'btc_{D2}.parquet').to_pandas()
    T = pd.concat([T, T.tail(1).assign(g_ms=M.BTC_END_MS)], ignore_index=True)
    write(b3 / f'btc_{D2}.parquet', {c: T[c].tolist() for c in BTC_T}, BTC_T)
    expect('a BTC row on Oct 28 (forbidden day)', lambda: M.run(inp=root, out=fresh(), days=(D1, D2), btc_dir=b3, **kw), 'forbidden day')
    b4 = base / 'btc_lookahead'
    shutil.copytree(root / 'btc', b4)
    T = pq.read_table(b4 / f'btc_{D1}.parquet').to_pandas()
    T.loc[100, 'age_ms'] = -5
    write(b4 / f'btc_{D1}.parquet', {c: T[c].tolist() for c in BTC_T}, BTC_T)
    expect('an ok BTC row with age_ms < 0 (look-ahead)', lambda: M.run(inp=root, out=fresh(), days=(D1, D2), btc_dir=b4, **kw), 'age_ms < 0')
    ex = base / 'exists' / 'decision'
    ex.mkdir(parents=True)
    expect('an existing output (write-once)', lambda: M.run(inp=root, out=ex, days=(D1, D2), btc_dir=root / 'btc', **kw), 'write-once')
    r3 = base / 'root_fvday'
    shutil.copytree(root, r3, ignore=shutil.ignore_patterns('out', 'refuse'))
    T = pq.read_table(r3 / 'econ' / 'fills_valued.parquet').to_pandas()
    T = pd.concat([T, T[T['addr'] == A1].head(1).assign(day='20261028')], ignore_index=True)
    write(r3 / 'econ' / 'fills_valued.parquet', {c: T[c].tolist() for c in FV_T}, FV_T)
    expect('a fills_valued top-3 row on a non-consumed day', lambda: M.run(inp=r3, out=fresh(), days=(D1, D2), btc_dir=r3 / 'btc', **kw),
           'non-consumed day')
    left = [p for p in base.rglob('.decision*')] + [p for p in root.glob('.decision_fit_*')] + [p for p in r2.glob('.decision_fit_*')] + \
        [p for p in r3.glob('.decision_fit_*')]
    check('refusals leave no partial or fit directory behind', not left, left)


def btc_rules(root):
    """pf_mt_btc semantics at the loader: age_ms -1 on an ok row is missing (counted, never fresh, no refusal); two usable rows of one g_ms
    in two files that disagree refuse."""
    b1 = root / 'btcrule1'
    b1.mkdir()
    T = pq.read_table(root / 'btc' / f'btc_{D2}.parquet').to_pandas()
    i = 1_000
    g_i = int(T['g_ms'].iloc[i])
    T.loc[i, 'age_ms'] = -1
    write(b1 / f'btc_{D2}.parquet', {c: T[c].tolist() for c in BTC_T}, BTC_T)
    c1 = Counter()
    BT = M.load_btc(b1, D2, c1)
    check('BTC rule: an ok row with age_ms -1 is missing (counted), never fresh, not refused',
          c1['btc_rows_age_missing'] == 31 and not bool(BT['ok'][g_i // 100 - BT['G0']]) and c1['btc_padding_file_missing:20261024'] == 1, dict(c1))
    b2 = root / 'btcrule2'
    shutil.copytree(root / 'btc', b2)
    T = pq.read_table(b2 / f'btc_{D1}.parquet').to_pandas()
    j = int(np.flatnonzero(T['g_ms'].to_numpy() == D1MS - 30_000)[0])
    T.loc[j, 'mid'] = T.loc[j, 'mid'] + 1.
    write(b2 / f'btc_{D1}.parquet', {c: T[c].tolist() for c in BTC_T}, BTC_T)
    try:
        M.load_btc(b2, D1, Counter())
        check('BTC rule: two usable rows of one g_ms that disagree are refused', False, 'no refusal')
    except ValueError as e:
        check('BTC rule: two usable rows of one g_ms that disagree are refused (reason asserted)', 'disagree' in str(e), str(e))


def parity_econ(root):
    T = pd.read_parquet(root / f'tob_{D1}.parquet')
    Fr = E.frames(T)
    rng = np.random.default_rng(5)
    um = np.array([101, 102], np.int64)
    qm = np.repeat(um, 4000)
    x = np.concatenate([rng.integers(WS[m] - 300_000, WS[m] + 300_000, 4000) for m in um]).astype(np.int64)
    me, _, se = E.mid_at(Fr, qm, x)
    FR = M.load_tob(root / f'tob_{D1}.parquet', um, np.array([WS[m] for m in um]) - M.TOB_KEEP_BEFORE_MS, np.array([WS[m] + 300_000 for m in um]), Counter())
    K = M.book_feats(FR, np.searchsorted(um, qm), x)
    ne = np.array(E.MARK_ST, dtype=object)[se]
    nm = np.array([s.replace('book_', '') for s in M.BOOK_ST], dtype=object)[K['code']]
    check('book parity with pf_mt_econ.mid_at: identical statuses on 8000 queries', bool((ne == nm).all()), Counter(zip(ne[ne != nm], nm[ne != nm])))
    mm = (K['bid'] + K['ask']) / 2.
    ok = ne == 'ok'
    check('book parity: identical mids where ok', bool(np.allclose(me[ok], mm[ok], rtol=0, atol=1e-12)) and ok.sum() > 1000, int(ok.sum()))
    import pf_mt_lifecycle as LC
    check('ORDER_INVALID equals pf_mt_lifecycle.INVALID', M.ORDER_INVALID == LC.INVALID, sorted(M.ORDER_INVALID ^ LC.INVALID))


def causality(root):
    rng = np.random.default_rng(9)
    m, d = 201, D2
    ws, we = WS[m], WS[m] + 300_000
    um = np.array([201, 202], np.int64)
    lo, hi = np.array([WS[201], WS[202]]) - M.TOB_KEEP_BEFORE_MS, np.array([WS[201], WS[202]]) + 300_000
    cut = ws + 150_000
    st = np.sort(rng.integers(ws - 300_000, we, 200)).astype(np.int64)
    sq = rng.normal(0, 20, 200)

    def ow(settle, s_):
        o = np.argsort(settle, kind='stable')
        return dict(gid=np.zeros(len(settle), np.int64), settle=settle[o], key=settle[o].copy(), cpos=np.cumsum(s_[o]))
    ctx0 = dict(BT=M.load_btc(root / 'btc', d, Counter()), FR=M.load_tob(root / f'tob_{d}.parquet', um, lo, hi, Counter()), OW=ow(st, sq), RS=None)
    b2 = root / 'causal' / 'btc'
    b2.mkdir(parents=True)
    shutil.copy(root / 'btc' / f'btc_{D1}.parquet', b2 / f'btc_{D1}.parquet')
    T = pq.read_table(root / 'btc' / f'btc_{d}.parquet').to_pandas()
    late = T['g_ms'] > cut
    T.loc[late, 'mid'] = T.loc[late, 'mid'] * np.exp(rng.normal(0, 5e-4, int(late.sum())))
    T.loc[late & (rng.random(len(T)) < .1), 'status'] = 'gap'
    write(b2 / f'btc_{d}.parquet', {c: T[c].tolist() for c in BTC_T}, BTC_T)
    B = pq.read_table(root / f'tob_{d}.parquet').to_pandas()
    lt = B['upd_ms'] > cut
    B.loc[lt, 'best_bid'] = np.clip(B.loc[lt, 'best_bid'] - .01 * rng.integers(0, 3, int(lt.sum())), .01, .5)
    B.loc[lt, 'best_ask'] = np.clip(B.loc[lt, 'best_ask'] + .01 * rng.integers(0, 3, int(lt.sum())), .51, .99)
    B.loc[lt & (rng.random(len(B)) < .1), 'status'] = 'bad_frame'
    B.loc[lt, 'upd_ms'] = B.loc[lt, 'upd_ms'] + 37                                   # later frames arrive at other times
    ins = B[lt].sample(200, random_state=1).copy()
    ins['upd_ms'] = rng.integers(cut + 1, we, 200)
    ins['best_bid'], ins['best_ask'] = .2, .8
    B = pd.concat([B, ins], ignore_index=True)
    write(root / 'causal' / f'tob_{d}.parquet', {c: B[c].tolist() for c in TOB_T}, TOB_T)
    st2 = np.r_[st, rng.integers(cut, we, 50)].astype(np.int64)
    sq2 = np.r_[np.where(st >= cut, sq * 3. + 1., sq), rng.normal(0, 20, 50)]
    ctx1 = dict(BT=M.load_btc(b2, d, Counter()), FR=M.load_tob(root / 'causal' / f'tob_{d}.parquet', um, lo, hi, Counter()), OW=ow(st2, sq2), RS=None)
    t = np.arange(ws - 300_000, cut + 1, 100, dtype=np.int64)
    diff_any = False
    for L in M.LAGS:
        x = t - L
        n = len(x)
        a = dict(ws=np.full(n, ws), we=np.full(n, we), code=np.zeros(n, np.int64), gid=np.zeros(n, np.int64), isbook=np.zeros(n, bool))
        F0 = M.evaluate(ctx0, x, a['ws'], a['we'], a['code'], a['gid'], a['isbook'])
        F1 = M.evaluate(ctx1, x, a['ws'], a['we'], a['code'], a['gid'], a['isbook'])
        bad = [k for k in F0 if not np.array_equal(np.asarray(F0[k], float), np.asarray(F1[k], float), equal_nan=True)]
        check(f'causality L={L}: features at x <= cut unchanged when later BTC rows / frames (values and arrival times) / own fills change', not bad, bad)
        G0 = M.evaluate(ctx0, x + 150, a['ws'], a['we'], a['code'], a['gid'], a['isbook'])
        G1 = M.evaluate(ctx1, x + 150, a['ws'], a['we'], a['code'], a['gid'], a['isbook'])
        diff_any |= any(not np.array_equal(np.asarray(G0[k], float), np.asarray(G1[k], float), equal_nan=True) for k in ('pf', 'bid', 'ask', 'pos'))
    check('causality falsifier fires: a planted +150 ms look-ahead does change features', diff_any)


def bench(root):
    """Marginal CPU of the production code paths at synthetic sizes (no data): BTC / tob loads, grid_chunk (all lags), fit_logits."""
    rng = np.random.default_rng(13)
    bd = root / 'bench'
    g = np.arange(D1MS - 3_600_000, D1MS + 86_400_000, 100, dtype=np.int64)
    S = dict(g=g, mid=np.exp(np.log(110_000.) + np.cumsum(rng.normal(0, math.sqrt(.1), len(g))) / 1e4), age=rng.integers(0, 100, len(g)).astype(np.int64),
             st=np.full(len(g), 'ok', dtype=object))
    write(bd / 'btc' / f'btc_{D1}.parquet', btc_cols(S, np.ones(len(g), bool)), BTC_T)
    c0 = time.process_time()
    BT = M.load_btc(bd / 'btc', D1, Counter())
    t_btc = time.process_time() - c0
    nm, nf = 288, 3_500                                               # 1.0 M frames over a day's 288 markets
    mids = np.arange(1, nm + 1, dtype=np.int64)
    wsm = D1MS + 300_000 * (mids - 1)
    upd = (wsm[:, None] - 300_000 + np.sort(rng.integers(0, 600_000, (nm, nf)), axis=1)).ravel()
    bb = np.round(rng.uniform(.2, .6, nm * nf), 2)
    cols = dict(day=[D1] * (nm * nf), mid=np.repeat(mids, nf), upd_ms=upd, recv_ns=upd * 1_000_000, conn=['c0'] * (nm * nf), order_count=np.full(nm * nf, 10),
                n_bid=np.full(nm * nf, 5), n_ask=np.full(nm * nf, 5), best_bid=bb, best_bid_sh=np.full(nm * nf, 100.), best_ask=bb + .02,
                best_ask_sh=np.full(nm * nf, 100.), pend_bid_usd=np.zeros(nm * nf), pend_ask_usd=np.zeros(nm * nf), status=['ok'] * (nm * nf))
    write(bd / 'tob.parquet', cols, TOB_T)
    c0 = time.process_time()
    FR = M.load_tob(bd / 'tob.parquet', mids, wsm - M.TOB_KEEP_BEFORE_MS, wsm + 300_000, Counter())
    t_tob = time.process_time() - c0
    npair, per = 70, 6_000                                            # 420 k grid rows (one chunk-sized batch per call)
    pm = rng.choice(mids, npair, replace=False)
    pw = wsm[pm - 1]
    t = (np.repeat(pw - 300_000, per) + 100 * np.tile(np.arange(per), npair)).astype(np.int64)
    gr = np.repeat(np.arange(npair, dtype=np.int64), per)
    fs = np.sort((gr[::20] << 42) | (t[::20] + 37))
    OW = dict(gid=fs >> 42, settle=fs & ((1 << 42) - 1), key=fs, cpos=np.cumsum(rng.normal(0, 20, len(fs))))
    ctx = dict(BT=BT, FR=FR, OW=OW, RS=None)
    ACC = M.acc_new(1)
    files = {(l, a): bd / f'fit_{l}_{a}.bin' for l in range(M.NL) for a in range(3)}
    fh = {k: open(p, 'ab') for k, p in files.items()}
    rows_st, ev_st = np.zeros((3, 1, M.NL, M.NRS), np.int64), np.zeros((3, 1, M.NL, 2, M.NRS), np.int64)
    ar = (gr % 3).astype(np.int64)
    yb, ys = (rng.random(len(t)) < .02).astype(float), (rng.random(len(t)) < .02).astype(float)
    R = 0
    c0 = time.process_time()
    for c_ in range(0, len(t), M.CHUNK_ROWS):
        s_ = slice(c_, c_ + M.CHUNK_ROWS)
        M.grid_chunk(ctx, t[s_], ar[s_], gr[s_], np.repeat(pw, per)[s_], np.repeat(pw + 300_000, per)[s_], np.repeat(pm - 1, per)[s_],
                     np.zeros(len(t[s_]), bool), yb[s_], ys[s_], 0, ACC, fh, rows_st, ev_st)
        R += len(t[s_])
    t_grid = time.process_time() - c0
    for f_ in fh.values():
        f_.close()
    base = {}
    for l in range(M.NL):
        for sc in range(4):
            for side in (0, 1):
                N, Y = M.acc_get(ACC, 'pos', sc, None)
                base[(l, sc, side)] = (int(N[l, side].sum()), int(Y[l, side].sum()))
    c0 = time.process_time()
    FITS = M.fit_logits(files, base, quiet)
    t_fit = time.process_time() - c0
    nv = int(rows_st[:, 0, 0, 0].sum())
    passes = sum(max(m_['passes'] for k, m_ in FITS.items() if k[0] == l) for l in range(M.NL))
    c_grid, c_pass = t_grid / (R * M.NL), t_fit / max(1, nv * passes)
    out = dict(rows=R, valid_rows_L0=nv, cpu_btc_day_s=round(t_btc, 3), cpu_tob_per_Mframe_s=round(t_tob / (nm * nf / 1e6), 3),
               cpu_grid_us_per_row_L=round(1e6 * c_grid, 3), cpu_fit_us_per_row_pass=round(1e6 * c_pass, 3), fit_passes_total=passes)
    for Rr in (10e6, 20.7e6):                                        # rows per L; worst case bounded by the per-L fit CPU cap (+ one pass)
        one = Rr * c_pass
        out[f'proj_R{Rr / 1e6:g}M_typical16passes_grid_plus_fit_h'] = round((Rr * M.NL * c_grid + 16 * one) / 3600., 4)
        out[f'proj_R{Rr / 1e6:g}M_worst_capped_grid_plus_fit_h'] = round((Rr * M.NL * c_grid + min(3 * M.LOGIT_MAX_PASS * one,
                                                                                                   M.NL * (M.FIT_CPU_CAP_S + one))) / 3600., 4)
    print('BENCH ' + json.dumps(out), flush=True)
    check('bench: production chunk / fit code ran on 420 k synthetic rows', R == npair * per and passes > 0, out)
    cap0 = M.FIT_CPU_CAP_S
    try:
        M.FIT_CPU_CAP_S = -1.                                         # the resource rule must stop every running model, never claim it
        F2 = M.fit_logits(files, base, quiet)
    finally:
        M.FIT_CPU_CAP_S = cap0
    check('fit CPU cap: models still running end with status cpu_cap (no data pass)',
          all(m_['status'] in ('cpu_cap', 'no_events', 'no_rows', 'all_events') for m_ in F2.values()) and any(m_['status'] == 'cpu_cap' for m_ in F2.values())
          and all(m_['passes'] == 0 for m_ in F2.values()), Counter(m_['status'] for m_ in F2.values()))
    return out


def main():
    c_all = time.process_time()
    with tempfile.TemporaryDirectory(prefix='mt_decision_selftest_') as tmp:
        root = Path(tmp)
        W = build_world(root, np.random.default_rng(20261123))
        print(f'world: A1 takes {W["a1"]}, A2 fills {[len(v) for v in W["a2f"].values()]}, records {len(W["recs"])}', flush=True)
        r = section('main run', main_run, root, W)
        if r is not None:
            out, rec_, cpu = r
            print(f'module run cpu {cpu:.2f} s; rows {rec_["rows"]}', flush=True)
            section('controls', checks_controls, out, W)
            section('statuses', checks_statuses, out, rec_, W)
        section('refusals', refusals, root, W)
        section('btc rules', btc_rules, root)
        section('parity', parity_econ, root)
        section('causality', causality, root)
        section('bench', bench, root)
    n_ok = sum(ok for _, ok in RES)
    print(f'SELFTEST {n_ok}/{len(RES)} checks passed; cpu {time.process_time() - c_all:.1f} s', flush=True)
    sys.exit(0 if n_ok == len(RES) else 1)


if __name__ == '__main__':
    main()
