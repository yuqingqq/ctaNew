"""posting_opportunity_selftest.py - synthetic proof of posting_opportunity.py (P-2026-005 MAKER-TRACE, POSTING-OPPORTUNITY v1;
posting_chain_contract.md sections 1-3). The REAL pipeline runs in-process on synthetic BOOK markets (chain_context.build_market ->
pf_mt_lifecycle._book; queue_front.market_book / market_fills / market_core / _records / _new_life; chain_context.book / level_size / btc_ret /
position / pending; chain_take.Ctx.btc; take_logic.mh_rr / tail_counts / null_counts / tl_codes / align_codes). Frames every 200 ms at
ws + 100 + 200 j, so only the boundary frame interval (t - 100, t + 100] crosses an anchor; regime switches sit on that frame (their creations
are straddles), planted creations on interior frames; every expectation is planted or hand-computed. Fixtures:
(P) a PLANTED world (6 markets): deep creations at 0.25 per frame when spread >= 2 AND fair_gap > 1 at the anchor, 0.01 otherwise, plus
    mechanical improves wherever the live spread is >= 2 -> spread and fair_gap supported for any / deep; btc / depth / flow insufficient (no
    exposure variation); final = D0 unless D1000 differs (timing_sensitive); the improve-availability restriction (spread-1 rows out of every
    improve ratio, spread contrast 3+ vs 2: person-time recounted from the panel; MT_PO_MUT=keep_spread1 caught).
(R) a RANDOM-creation world of 3 TWIN pairs: each twin replays its partner's creation schedule on the swapped regime (spread / fair_gap exposure
    <-> reference) and the time-left windows hold an exact 5 / 30 vs 10 / 60 balance -> pooled RR exactly 1 for spread / fair_gap / time_left,
    reciprocal market RRs, none supported or supported_lower while testable (>= 5 supported markets, >= 30 events): the negative control could fire.
(FS) statuses: a creation interval straddling an anchor -> status straddle, not an event (rows touched counted; MT_PO_MUT=assign_straddle
    caught), a creation across a frame gap -> gap, a cell unknown in f - 1 -> unknown_cell, stale rows at t and at t - 1 s, an improve event.
(FC) causality: moving the ARRIVAL TIMES of every later frame, BTC grid update and fill (+ new later cells and a new fill) leaves every D0 / D1000
    feature at t <= t* unchanged while later ones change (positive control); MT_PO_MUT=leak_1s caught.
(FI) identity: a book-timed G2 front whose posting window contains the creation -> candidate_constrained with its inventory (long -> a bid
    creation is adding); a front whose window opens after the creation -> owner_unknown window_not_containing (MT_PO_MUT=link_any_window
    caught); a G4 front -> candidate_scenario (no inventory); a non-top-3 front -> candidate_scenario without a regrade row; a no-fill level ->
    owner_unknown; (FI_v2) a settlement-only first fill (v1 front_identified by settlement) -> under the v2 straddle rule no candidate
    (MT_PO_MUT=v1_straddle caught); the replicated v1 front logic equals market_core (the tool refuses otherwise).
Also: bins, relation, the MH toy (hand-computed 3.4 / crude 3.889), the verdict rule incl. supported_lower and the tie, the shift (counts per
market-side kept, a bijection, deterministic, sides independent), the queue_front_v2 adapter (same identities, file binding, schema /
vocabulary refusals), the refusals (each asserting its reason, overwrite included) and the pilot path end to end through execute() (seven
outputs, every aggregate reconciled to the saved keyed rows). Usage (run_light, MT_RUN=v2): posting_opportunity_selftest.py"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import posting_opportunity as PO  # noqa: E402

QF, CX, C, TL = PO.QF, PO.CX, PO.C, PO.TL
NAN = float('nan')
RES = []
A1, A2, A3 = PO.TOP3
A1L, A2L, A3L = PO.TOP3_L
OTHER = '0x' + 'ab' * 20
DAY, DAY2 = '20261024', '20261025'
DUR = 120_000                                     # synthetic windows: 120 s (K = 120; the 300-s predicate is reported, not required)
J0 = -15                                          # frames ws + 100 + 200 j, j = -15 .. : from ws - 2.9 s
J1 = (DUR + 1_000) // 200 + 1                     # .. to we + 1.1 s
NG = (400_000 + DUR + 100_000) // 100 + 1         # BTC grid rows [ws - 400 s, we + 100 s]
TY = {'str': object, 'int64': np.int64, 'float64': np.float64, 'bool': bool, 'num': np.float64}
TOBS = {c: C.SCHEMAS['tob'][c] for c in CX.TOB_READ}
LIFE = dict(mid='int64', sid='str', mk_hash='str', mk_idx='int64', y_side='str', y_ptick='int64', shares='float64', hi_ms='int64', lo_ms='int64',
            status='str', x_lo_hard_ms='int64')
ORD = dict(scenario='str', mid='int64', mk_hash='str', mk_signer='str', y_side='str', y_ptick='int64', persist_lo_ms='int64', persist_hi_ms='int64',
           status='str', post_lo_ms='int64', post_hi_ms='int64')
FV_DT = {c: TY[CX.E.FV[c]] for c in CX.FV_READ}
REG = dict(G=(47, 50), H=(51, 52), I=(47, 53), J=(49, 51), K=(48, 49))     # (bid, ask) ticks; p_fair = 0.5 on a flat BTC (fair 50 c)
P_HIGH, P_LOW, P_IMP = 0.25, 0.01, 0.05
CTX = PO.CT.make_ctx(CX)
EXPECT_CATCH = dict(keep_spread1=('P6',), assign_straddle=('FS',), leak_1s=('FC',), link_any_window=('FI',), v1_straddle=('FI_v2',))


def check(name, ok, info=''):
    RES.append((name, bool(ok), str(info)[:900]))


def expect(name, fn, needle):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        check(name, needle in str(e), f'{type(e).__name__}: {e}')
        return
    check(name, False, 'did not raise')


def section(name, fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        check(f'{name}: section crashed', False, f'{type(e).__name__}: {e} | {traceback.format_exc()[-1800:]}')
        return None


def typed(rows, spec):
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=TY[t]) for c, t in spec.items()})
    return pd.DataFrame(rows, columns=list(spec)).astype({c: TY[t] for c, t in spec.items()})


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == 'f' or b.dtype.kind == 'f':
        return bool(np.array_equal(a.astype(float), b.astype(float), equal_nan=True))
    return bool(np.array_equal(a, b))


def bt_from_lm(G0, lm):
    """pf_mt_decision.load_btc's dense layout from a log-mid path (every row usable)."""
    n = len(lm)
    ok = np.isfinite(lm)
    ret = np.zeros(n)
    ret[1:] = 1e4 * (lm[1:] - lm[:-1])
    ret[~np.isfinite(ret)] = 0.
    csq = np.zeros(n + 1)
    csq[1:] = np.cumsum(ret * ret)
    okc = np.zeros(n + 1, np.int64)
    okc[1:] = np.cumsum(ok)
    return dict(G0=int(G0), n=int(n), ok=ok, lm=lm, csq=csq, okc=okc)


# ------------------------------------------------------------------ synthetic BOOK markets ------------------------------------------------------------------
class SM:
    """One synthetic BOOK market: a dense (F, 2, 101) shown array on frames ws + 100 + 200 j (background cells bid 5 / ask 95), dropped frames
    (gaps), unknown cells (a non-ok level row), a 100 ms BTC grid, lifecycle legs (hard window [hi - 2 s, hi]) with their orders (posting
    windows per scenario) and the top-3 fills_valued legs."""

    def __init__(self, day, mid, hour, seed=0, btc_sd=0.0):
        self.day, self.mid = str(day), int(mid)
        self.ws = CX.day_ms(self.day) + 3_600_000 * int(hour)
        self.we = self.ws + DUR
        self.K = DUR // 1000
        self.t = self.ws + 100 + 200 * np.arange(J0, J1, dtype=np.int64)
        self.S = np.zeros((len(self.t), 2, 101))
        self.S[:, 0, 5] = 10.
        self.S[:, 1, 95] = 10.
        self.keep = np.ones(len(self.t), bool)
        self.unk = []
        self.G0 = (self.ws - 400_000) // 100
        rng = np.random.default_rng(seed)
        inc = rng.normal(0., btc_sd * 1e-4, NG) if btc_sd > 0 else np.zeros(NG)
        inc[0] = 0.
        self.lm = np.log(70_000.) + np.cumsum(inc)
        self.legs, self.orders, self.n, self._built = [], {}, 0, None

    def copy(self, mid):
        o = SM.__new__(SM)
        o.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()})
        o.mid, o.unk, o._built = int(mid), list(self.unk), None
        o.legs = [dict(L) for L in self.legs]
        o.orders = {h: dict(v) for h, v in self.orders.items()}
        return o

    def fi(self, rel):
        need_ = (int(rel) - 100) % 200 == 0
        if not need_:
            raise ValueError(f'no frame at ws + {rel}')
        return (int(rel) - 100) // 200 - J0

    def setc(self, ra, rb, side, tick, sh):
        m = (self.t >= self.ws + ra) & (self.t < self.ws + rb)
        self.S[m, 0 if side == 'bid' else 1, int(tick)] = float(sh)
        return self

    def pulse(self, rel, side, tick, sh):
        self.S[self.fi(rel), 0 if side == 'bid' else 1, int(tick)] = float(sh)
        return self

    def drop(self, ra, rb):
        self.keep[(self.t > self.ws + ra) & (self.t < self.ws + rb)] = False
        return self

    def unknown(self, rel, side, tick):
        self.unk.append((self.fi(rel), 0 if side == 'bid' else 1, int(tick)))
        return self

    def fill(self, addr, side, tick, q, rhi, hsh=None, post_lo=None, post_hi=None, sid=None):
        """A maker leg (its own record unless sid is shared); post_lo = (S_FIFO, S_FIFO_LO, S_NOPRIO) relative ms of its order's window."""
        self.n += 1
        hi = self.ws + int(rhi)
        h = hsh or f'0xh{self.mid}{self.n:04d}'
        L = dict(sid=sid or f'{self.mid}-{self.n:04d}', mk_idx=0, mk_hash=h, addr=addr, side=side, tick=int(tick), q=float(q), hi=hi, lo=hi - 2000)
        self.legs.append(L)
        if h not in self.orders:
            pl = tuple(self.ws + int(x) for x in post_lo) if post_lo is not None else (hi - 5000,) * 3
            self.orders[h] = dict(addr=addr, side=side, tick=int(tick), post_lo=pl, post_hi=self.ws + int(post_hi) if post_hi is not None else hi)
        self._built = None
        return L

    def build(self):
        if self._built is not None:
            return self._built
        kp = np.flatnonzero(self.keep)
        t, S = self.t[kp], self.S[kp].copy()
        F = len(t)
        fid = np.arange(F, dtype=np.int64)
        recv = (t + 40) * 1_000_000 + fid
        fr = pd.DataFrame(dict(mid=np.full(F, self.mid, np.int64), fid=fid, upd_ms=t, recv_ns=recv, conn=['c1'] * F, order_count=np.full(F, -1, np.int64),
                               pend_bids=['[]'] * F, pend_asks=['[]'] * F, los_id=[''] * F, los_side=[''] * F, los_price=np.full(F, NAN),
                               los_outcome=[''] * F, status=['ok'] * F))
        U = np.zeros(S.shape, bool)
        pos_of = {int(j): i for i, j in enumerate(kp)}
        for j, s, k in self.unk:
            if j in pos_of:
                U[pos_of[j], s, k] = True
        known = (S > 0) & ~U
        f_, s_, k_ = np.nonzero(known)
        fu, su, ku = np.nonzero(U)
        nl = len(f_) + len(fu)
        lv = pd.DataFrame(dict(mid=np.full(nl, self.mid, np.int64), fid=np.r_[fid[f_], fid[fu]].astype(np.int64),
                               side=np.r_[np.where(s_ == 0, 'bid', 'ask'), np.where(su == 0, 'bid', 'ask')].astype(object),
                               ptick=np.r_[k_, ku].astype(np.int64), shares=np.r_[S[f_, s_, k_], np.ones(len(fu))].astype(float),
                               status=['ok'] * len(f_) + ['bad_level'] * len(fu)))
        kb, ka = known[:, 0, :], known[:, 1, :]
        nb, na = kb.sum(1), ka.sum(1)
        bb = np.where(nb > 0, 100 - np.argmax(kb[:, ::-1], 1), 0)
        ba = np.where(na > 0, np.argmax(ka, 1), 0)
        ar = np.arange(F)
        tob = pd.DataFrame(dict(mid=np.full(F, self.mid, np.int64), upd_ms=t, recv_ns=recv, conn=['c1'] * F, n_bid=nb.astype(np.int64),
                                n_ask=na.astype(np.int64), best_bid=np.where(nb > 0, bb / 100., NAN), best_bid_sh=np.where(nb > 0, S[ar, 0, bb], 0.),
                                best_ask=np.where(na > 0, ba / 100., NAN), best_ask_sh=np.where(na > 0, S[ar, 1, ba], 0.), status=['ok'] * F))
        life = typed([dict(mid=self.mid, sid=L['sid'], mk_hash=L['mk_hash'], mk_idx=L['mk_idx'], y_side=L['side'], y_ptick=L['tick'], shares=L['q'],
                           hi_ms=L['hi'], lo_ms=-1, status='ok', x_lo_hard_ms=L['lo']) for L in self.legs], LIFE)
        orders = typed([dict(scenario=sc, mid=self.mid, mk_hash=h, mk_signer=o['addr'], y_side=o['side'], y_ptick=o['tick'], persist_lo_ms=-1,
                             persist_hi_ms=-1, status='ok', post_lo_ms=o['post_lo'][j], post_hi_ms=o['post_hi'])
                        for h, o in self.orders.items() for j, sc in enumerate(PO.SCEN)], ORD)
        fv = [dict(day=self.day, mid=self.mid, sid=L['sid'], mk_idx=L['mk_idx'], px_rank=0, role='maker', addr=L['addr'], hash=L['mk_hash'], cp_addr=OTHER,
                   kind='regular', y_side=L['side'], s=1 if L['side'] == 'bid' else -1, q=L['q'], p=L['tick'] / 100., self_trade=False, settle_ms=L['hi'],
                   t_ms=L['hi'], t_status='lo', fee=0., fee_type='COLLATERAL', fee_dy=0., status='ok', src_status='ok')
              for L in self.legs if L['addr'].lower() in PO.TOP3_L]
        fvdf = pd.DataFrame(fv, columns=list(CX.FV_READ)).astype(FV_DT) if fv else pd.DataFrame({c: pd.Series(dtype=FV_DT[c]) for c in CX.FV_READ})
        legs = CX.join_windows(fvdf, life)
        BT = bt_from_lm(self.G0, self.lm)
        btc = dict(BT=BT, code=np.where(BT['ok'], 0, 2).astype(np.int8), day=self.day, load_counts={}, files=[], cum=None)
        self._built = CX.build_market(self.day, self.mid, self.ws, self.we, PO.TOP3, btc=btc, book=True, tob=tob, frames=fr, levels=lv, legs=legs,
                                      life_fills=life, orders=orders)
        return self._built

    def part(self, RGm=None, Q2m=None):
        return PO.market_pass(self.day, self.mid, self.build(), CTX, RGm or {}, Q2m)


def regime_seq(rng, names, n):
    seq = []
    while len(seq) < n:
        seq += [names[int(rng.integers(len(names)))]] * int(rng.integers(5, 16))
    return seq[:n]


def apply_regimes(m, regs):
    """second s (-3 .. K + 1) shows regs[s + 3] on its frames ws + 1000 s + 100 .. + 900 (a switch sits on the anchor-crossing frame)."""
    ks = np.floor_divide(m.t - m.ws, 1000)
    ra = np.asarray(regs, dtype=object)[np.clip(ks + 3, 0, len(regs) - 1)]
    for nm, (b, a) in REG.items():
        msk = ra == nm
        m.S[msk, 0, b] = 100.
        m.S[msk, 1, a] = 100.


def high(reg, s):
    b, a = REG[reg]
    return (a - b) >= 2 and ((50 - b) if s == 0 else (a - 50)) > 1


def deep_tick(s, ct):
    ct[s] += 1
    return (30 + ct[s] % 10) if s == 0 else (61 + ct[s] % 10)


def planted_world(seed=11, n_mk=6):
    rng = np.random.default_rng(seed)
    out = []
    for j in range(n_mk):
        m = SM(DAY if j % 2 == 0 else DAY2, 940001 + j, 1 + j)
        regs = regime_seq(rng, tuple(REG), 3 + m.K + 2)
        apply_regimes(m, regs)
        ct = [0, 0]
        for k in range(m.K):
            ra, rl = regs[k - 1 + 3], regs[k + 3]                    # the anchor's regime (last frame <= t) and the live one
            for s in (0, 1):
                p = P_HIGH if high(ra, s) else P_LOW
                for r in (300, 500, 700, 900):
                    if rng.random() < p:
                        m.S[m.fi(1000 * k + r), s, deep_tick(s, ct)] = 50.
            b, a = REG[rl]
            if a - b >= 2:
                for r, s in ((300, 0), (500, 1), (700, 0)):
                    if rng.random() < P_IMP:
                        m.S[m.fi(1000 * k + r), s, (b + 1) if s == 0 else (a - 1)] = 30.
        out.append(m)
    return out


def twin_world(seed=23, n_pair=3):
    rng = np.random.default_rng(seed)
    out = []
    for p in range(n_pair):
        day = (DAY, DAY2, DAY)[p]
        A, B = SM(day, 950001 + 2 * p, 7 + 2 * p), SM(day, 950002 + 2 * p, 8 + 2 * p)
        K = A.K
        regs = regime_seq(rng, ('G', 'H'), 3 + K + 2)
        apply_regimes(A, regs)
        apply_regimes(B, ['H' if x == 'G' else 'G' for x in regs])
        for s in (0, 1):
            ev = set(rng.choice(np.arange(K - 30, K), 5, replace=False).tolist()) | set(rng.choice(np.arange(0, K - 60), 10, replace=False).tolist())
            ev |= {k for k in range(K - 60, K - 30) if rng.random() < 0.17}
            ct = [0, 0]
            for k in sorted(ev):
                r = int(rng.choice((300, 500, 700, 900)))
                tk = deep_tick(s, ct)
                for m in (A, B):
                    m.S[m.fi(1000 * k + r), s, tk] = 50.
        out += [A, B]
    return out


def causality_pair(seed=31):
    rng = np.random.default_rng(seed)
    m = SM(DAY, 970003, 3, seed=seed, btc_sd=0.4)
    regs = regime_seq(rng, ('G', 'I', 'J', 'H'), 3 + m.K + 2)
    apply_regimes(m, regs)
    ct = [0, 0]
    for k in range(m.K):
        for s in (0, 1):
            for r in (300, 500, 700, 900):
                if rng.random() < 0.08:
                    m.S[m.fi(1000 * k + r), s, deep_tick(s, ct)] = 50.
    for i in range(55):
        rel = int(rng.integers(1_000, m.K * 1000 - 500))
        s = int(rng.integers(2))
        b, a = REG[regs[rel // 1000 + 3]]
        m.fill((A1, A2, OTHER)[i % 3], 'bid' if s == 0 else 'ask', b if s == 0 else a, float(rng.integers(5, 60)), rel)
    ts = 60_000
    p = m.copy(970004)
    late = p.t > p.ws + ts
    p.t = np.where(late, p.t + 37, p.t)                                # every later frame arrives 37 ms later
    p.S[p.t > p.ws + ts + 3000, 0, 20] = 15.                            # ... and shows a new cell
    g = (p.G0 + np.arange(len(p.lm), dtype=np.int64)) * 100
    p.lm = np.where(g > p.ws + ts, p.lm + 5e-4, p.lm)                  # every later BTC grid update moves (+5 bp)
    for L in p.legs:
        if L['hi'] > p.ws + ts:                                         # every later fill arrives 300 ms later
            L['hi'] += 300
            L['lo'] = L['hi'] - 2000
    p.fill(OTHER, 'bid', 47, 25., ts + 500)                             # and a new later fill
    p.mid = m.mid                                                       # the same market (the perturbed record of it)
    return m, p, ts


def fixtures():
    FX = {}
    m = SM(DAY, 970001, 1)                                              # FS: statuses
    m.setc(-10 ** 6, 10 ** 7, 'bid', 47, 100.).setc(-10 ** 6, 10 ** 7, 'ask', 50, 100.)
    m.pulse(10100, 'bid', 35, 50.)                                      # interval (9.9 s, 10.1 s]: straddles the anchor ws + 10 s
    m.pulse(20500, 'bid', 36, 50.)                                      # interval (20.3 s, 20.5 s]: an event of row 20
    m.drop(40000, 41600).setc(41700, 42100, 'bid', 37, 40.)             # a frame gap (39.9 s -> 41.7 s) and a creation across it
    m.unknown(50300, 'bid', 38).setc(50500, 50700, 'bid', 38, 30.)      # the cell unknown in f - 1
    m.pulse(60300, 'bid', 48, 30.)                                      # an improve (48 > 47) of row 60
    FX['FS'] = m
    m = SM(DAY, 970002, 2)                                              # FI: identity
    m.setc(-10 ** 6, 10 ** 7, 'bid', 55, 100.).setc(-10 ** 6, 10 ** 7, 'ask', 60, 100.)
    m.fill(A1, 'bid', 55, 20., -20_000, hsh='0xhprior')                 # A1 long 20 before every creation
    m.setc(2300, 4500, 'bid', 45, 100.).setc(4500, 60000, 'bid', 45, 70.)
    m.fill(A1, 'bid', 45, 30., 5000, hsh='0xhC1', post_lo=(1500, 1000, 1500), post_hi=5000)       # C1: G2, window contains
    m.setc(6300, 8500, 'bid', 46, 100.).setc(8500, 60000, 'bid', 46, 70.)
    m.fill(A2, 'bid', 46, 30., 9000, hsh='0xhC2', post_lo=(7000, 7000, 7000), post_hi=9000)       # C2: window opens after the creation
    m.setc(10300, 60000, 'bid', 47, 100.)
    m.fill(A3, 'bid', 47, 30., 12250, hsh='0xhC3', post_lo=(9000, 9000, 9000), post_hi=12250)     # C3: settlement-only (no decrease)
    m.setc(14300, 16500, 'bid', 48, 100.).setc(16500, 60000, 'bid', 48, 70.)
    m.fill(A1, 'bid', 48, 30., 17000, hsh='0xhC4', post_lo=(13000, 13000, 13000), post_hi=17000)  # C4: G4
    m.setc(18300, 20500, 'bid', 49, 100.).setc(20500, 60000, 'bid', 49, 70.)
    m.fill(OTHER, 'bid', 49, 30., 21000, hsh='0xhC5', post_lo=(17000, 17000, 17000), post_hi=21000)  # C5: not top 3
    m.setc(22300, 23500, 'ask', 70, 40.)                                # C6: cancelled, never filled
    FX['FI'] = m
    FX['FC'], FX['FC2'], FX['ts'] = causality_pair()
    return FX


def rg_rows(m, grades):
    """post_regrade_orders rows of the fixture's top-3 orders (window = the lifecycle union, as chain_post_fill builds it)."""
    rows = []
    for h, o in m.orders.items():
        if h not in grades:
            continue
        lo = -1 if min(o['post_lo']) < 0 else min(o['post_lo'])
        rk = PO.TOP3_L.index(o['addr'].lower()) + 1
        rows.append(dict(day=m.day, mid=m.mid, rank=rk, addr=o['addr'], mk_hash=h, y_side=o['side'], tick=o['tick'], post_lo_union_ms=lo,
                         post_hi_ms=o['post_hi'], grade=grades[h]))
    return rows


FI_GRADES = {'0xhC1': PO.G2, '0xhC2': PO.G2, '0xhC3': PO.G2, '0xhC4': 'G4_scenario_only', '0xhprior': 'G4_scenario_only'}


def rg_map(rows):
    return {r['mk_hash']: dict(grade=r['grade'], lo=int(r['post_lo_union_ms']), hi=int(r['post_hi_ms']), addr=r['addr'].lower(), side=r['y_side'],
                               tick=int(r['tick']), rank=int(r['rank'])) for r in rows}


def worlds():
    W = {}
    for nm, mks in (('planted', planted_world()), ('random', twin_world())):
        parts = [m.part() for m in mks]
        days = sorted({m.day for m in mks})
        frames, tables, claims, dg = PO.analyse('selftest', days, parts, Counter())
        W[nm] = dict(mks=mks, parts=parts, P=PO.assemble(parts), claims=claims, tables=tables, E={(e['type'], e['contrast']): e for e in claims['ratios']})
    return W


# ------------------------------------------------------------------ named checks (re-run under the mutants) ------------------------------------------------------------------
def chk_P1(FX, W):
    E = W['planted']['E']
    got = {f'{ty}:{cn}': (E[(ty, cn)]['verdict'], round(float(E[(ty, cn)]['D0']['RR_MH']), 3), E[(ty, cn)]['D0']['p_hi'],
                          E[(ty, cn)]['D0']['n_markets_supported']) for ty in ('any', 'deep') for cn in ('spread', 'fair_gap')}
    ok = all(v[0] == 'supported' and v[1] > 1.5 and v[2] <= 0.05 and v[3] >= 5 for v in got.values())
    return ok, got


def chk_P4(FX, W):
    E = W['planted']['E']
    got = {cn: E[('any', cn)]['verdict'] for cn in ('btc_with', 'btc_against', 'depth_own', 'flow')}
    return all(v == 'insufficient_support' for v in got.values()), got


def chk_P5(FX, W):
    bad = []
    for W_ in (W['planted'], W['random']):
        for (ty, cn), e in W_['E'].items():
            v0, v1 = e['D0']['verdict'], e['D1000']['verdict']
            if e['verdict'] != (v0 if v0 == v1 else 'timing_sensitive') or e['verdict'] not in PO.VERDICTS:
                bad.append((ty, cn, v0, v1, e['verdict']))
    n = len(W['planted']['E'])
    return not bad and n == len(PO.TYPES) * len(PO.CN_NAMES), dict(bad=bad[:5], n=n)


def chk_P6(FX, W):
    """the improve-availability restriction: spread-1 (and missing-spread) rows out of every improve ratio; spread contrast 3+ vs 2."""
    P = W['planted']['P']
    ev = np.flatnonzero(P['E_improve'] > 0)
    SH = PO.shift_dest(P, ev)
    rs = PO.ratio_block(P, 'D0', 'improve', 'spread', ev, SH)[0]
    rf = PO.ratio_block(P, 'D0', 'improve', 'fair_gap', ev, SH)[0]
    rt = PO.ratio_block(P, 'D0', 'improve', 'time_left', ev, SH)[0]
    ok0 = P['st_0'] == 0
    sp, fg, tl = P['sp_c_0'], P['fg_c_0'], P['tl_c_0']
    avail = ok0 & np.isin(sp, (1, 2, 3))
    n_exp, n_ref, n_sp1 = int((ok0 & np.isin(sp, (2, 3))).sum()), int((ok0 & (sp == 1)).sum()), int((ok0 & (sp == 0)).sum())
    n_fg = int((avail & np.isin(fg, (0, 2, 3))).sum())
    n_tl = int((avail & np.isin(tl, (2, 3, 5, 6))).sum())
    n_imp_sp1 = int(((P['E_improve'] > 0) & ok0 & (sp == 0)).sum())
    ok = (rs['T_exposed'] == n_exp and rs['T_reference'] == n_ref and rf['T_exposed'] + rf['T_reference'] == n_fg
          and rt['T_exposed'] + rt['T_reference'] == n_tl and rs['exposed'] == PO.IMPROVE_SPREAD[0] and rs['reference'] == PO.IMPROVE_SPREAD[1]
          and n_sp1 > 0 and rs['rows_availability_excluded'] == n_sp1 and rf['rows_availability_excluded'] == n_sp1)
    return ok, dict(T1=rs['T_exposed'], T0=rs['T_reference'], exp=n_exp, ref=n_ref, sp1_rows=n_sp1, fg=(rf['T_exposed'] + rf['T_reference'], n_fg),
                    tl=(rt['T_exposed'] + rt['T_reference'], n_tl), improve_events_at_anchor_spread1=n_imp_sp1, rr=rs['RR_MH'])


def chk_R1(FX, W):
    E = W['random']['E']
    fired = {f'{ty}:{cn}': e['verdict'] for (ty, cn), e in E.items() if e['verdict'] in ('supported', 'supported_lower')}
    d0 = {f'{ty}:{cn}': e['D0']['verdict'] for (ty, cn), e in E.items() if e['D0']['verdict'] in ('supported', 'supported_lower')}
    return not fired and not d0, dict(final=fired, d0=d0)


def chk_R2(FX, W):
    E = W['random']['E']
    got = {cn: (E[('any', cn)]['D0']['n_markets_supported'], E[('any', cn)]['D0']['n_events_informative']) for cn in ('spread', 'fair_gap', 'time_left')}
    return all(a >= 5 and b >= 30 for a, b in got.values()), got


def chk_R3(FX, W):
    E = W['random']['E']
    got = {cn: (float(E[('any', cn)]['D0']['RR_MH']), E[('any', cn)]['D0']['n_markets_rr_gt1'], E[('any', cn)]['D0']['n_markets_rr_lt1'])
           for cn in ('spread', 'fair_gap', 'time_left')}
    ok = all(abs(v[0] - 1.) < 1e-9 for v in got.values()) and all(got[cn][1] == got[cn][2] for cn in ('spread', 'fair_gap')) \
        and got['time_left'][1] == 0 and got['time_left'][2] == 0
    return ok, got


def chk_FS(FX, W):
    m = FX['FS']
    part = m.part()
    EV, R = part['events'], part['rows']

    def er(tick, rel):
        x = EV[(EV['side'] == 'bid') & (EV['tick'] == tick) & (EV['t_create_ms'] == m.ws + rel)]
        return x.iloc[0].to_dict() if len(x) == 1 else None
    x1, x2, x3, x4, x5 = er(35, 10100), er(36, 20500), er(37, 41700), er(38, 50500), er(48, 60300)
    ok = (x1 is not None and x1['status'] == 'straddle' and int(x1['row_k']) == -1 and int(x1['k_lo']) == 9 and int(x1['k_hi']) == 10
          and x2 is not None and x2['status'] == 'event' and int(x2['row_k']) == 20 and x3 is not None and x3['status'] == 'gap'
          and x4 is not None and x4['status'] == 'unknown_cell' and x5 is not None and x5['status'] == 'event' and int(x5['row_k']) == 60
          and bool(x5['is_improve']))
    ok = ok and (int(R['E_any'][9]) == 0 and int(R['E_any'][10]) == 0 and int(R['n_straddle'][9]) == 1 and int(R['n_straddle'][10]) == 1
                 and int(R['E_any'][20]) == 1 and int(R['N_any'][20]) == 1 and R['keys_any'][20] == x2['key']
                 and [int(x) for x in R['n_gap'][38:43]] == [0, 1, 1, 1, 0] and int(R['n_unknown_cell'][50]) == 1 and int(R['E_any'][50]) == 0
                 and int(R['E_improve'][60]) == 1 and int(R['E_any'][41]) == 0
                 and int(R['st_0'][40]) == 0 and int(R['st_0'][41]) == 2 and int(R['st_0'][42]) == 0 and int(R['st_1'][42]) == 2
                 and int(R['E_any'].sum()) == 2 and int(part['counts']['creations_straddle']) >= 1)
    return ok, dict(x1=None if x1 is None else (x1['status'], x1['row_k']), x3=None if x3 is None else x3['status'],
                    x4=None if x4 is None else x4['status'], E=(int(R['E_any'][9]), int(R['E_any'][10]), int(R['E_any'][20])),
                    st=[int(R['st_0'][k]) for k in (40, 41, 42)], st1_42=int(R['st_1'][42]), n_gap=[int(x) for x in R['n_gap'][38:43]],
                    E_sum=int(R['E_any'].sum()))


def _feats(m):
    M = m.build()
    fl, _ = QF.market_fills(M, None)
    FI = PO.flow_index(fl, Counter())
    return PO.features(M, CTX, FI, M.ws, M.we, (M.we - M.ws) // 1000)[1]


def chk_FC(FX, W):
    base, pert, ts = FX['FC'], FX['FC2'], FX['ts']
    a, b = _feats(base), _feats(pert)
    K = base.K
    ks = ts // 1000
    before, after = [], []
    for s in (0, 1):
        for f in PO.FEAT:
            for sl in (slice(0, ks + 1), slice(K, K + ks + 1)):
                if not same(np.asarray(a[s][f])[sl], np.asarray(b[s][f])[sl]):
                    before.append((s, f, sl.start))
            if not same(np.asarray(a[s][f])[ks + 2:K], np.asarray(b[s][f])[ks + 2:K]):
                after.append((s, f))
    moved_flow = any(f == 'flow' for _, f in after)
    moved_btc = any(f in ('r5', 'pf', 'fg') for _, f in after)
    return not before and moved_flow and moved_btc, dict(changed_at_or_before_t=before[:6], changed_after=after[:12])


def _fi_part(FX, Q2m=None):
    m = FX['FI']
    return m.part(rg_map(rg_rows(m, FI_GRADES)), Q2m)


def _crow(EV, side, tick):
    x = EV[(EV['side'] == side) & (EV['tick'] == tick) & (EV['status'] == 'event')]
    return x.iloc[0].to_dict() if len(x) == 1 else None


def chk_FI(FX, W):
    part = _fi_part(FX)
    EV = part['events']
    c1, c2, c4, c5, c6 = (_crow(EV, 'bid', 45), _crow(EV, 'bid', 46), _crow(EV, 'bid', 48), _crow(EV, 'bid', 49), _crow(EV, 'ask', 70))
    ok = (c1 is not None and c1['identity'] == 'candidate_constrained' and c1['cand_addr'] == A1L and c1['cand_group'] == 'rank1'
          and c1['grade'] == PO.G2 and bool(c1['win_contains']) and c1['inv_relation'] == 'adding' and abs(float(c1['inv_q']) - 20.) < 1e-9
          and not bool(c1['inv_timing_sensitive']) and c1['qf_timing_v2'] == 'book' and c1['cand_hash'] == '0xhC1'
          and c2 is not None and c2['identity'] == 'owner_unknown' and c2['identity_reason'] == 'window_not_containing' and c2['qf_status_v2'] == 'front_identified'
          and c2['cand_addr'] == A2L and not bool(c2['win_contains'])
          and c4 is not None and c4['identity'] == 'candidate_scenario' and c4['grade'] == 'G4_scenario_only' and c4['inv_relation'] == ''
          and np.isnan(float(c4['inv_q']))
          and c5 is not None and c5['identity'] == 'candidate_scenario' and c5['cand_group'] == 'other' and c5['grade'] == PO.NO_GRADE
          and c6 is not None and c6['identity'] == 'owner_unknown' and c6['identity_reason'] == 'no_front_candidate:no_fill'
          and int(part['counts'].get('candidate_lead_not_in_fills_valued', 0)) == 0)
    brief = {k: (None if c is None else (c['identity'], c['identity_reason'], c['cand_addr'][:8], c['grade'], c['inv_relation'], c['qf_status_v2']))
             for k, c in (('c1', c1), ('c2', c2), ('c4', c4), ('c5', c5), ('c6', c6))}
    return ok, brief


def chk_FI_v2(FX, W):
    part = _fi_part(FX)
    c3 = _crow(part['events'], 'bid', 47)
    ok = (c3 is not None and c3['qf_status_v1'] == 'front_identified' and c3['qf_timing_v1'] == 'settlement' and c3['qf_status_v2'] == 'unknown'
          and c3['identity'] == 'owner_unknown' and c3['identity_reason'] == 'no_front_candidate:unknown' and c3['cand_addr'] == ''
          and int(part['counts']['v1_v2_status_changed']) >= 1)
    return ok, None if c3 is None else (c3['qf_status_v1'], c3['qf_timing_v1'], c3['qf_status_v2'], c3['identity'], c3['identity_reason'])


SUITE = (('P1', chk_P1), ('P4', chk_P4), ('P5', chk_P5), ('P6', chk_P6), ('R1', chk_R1), ('R2', chk_R2), ('R3', chk_R3), ('FS', chk_FS),
         ('FC', chk_FC), ('FI', chk_FI), ('FI_v2', chk_FI_v2))
SUITE_FN = dict(SUITE)
SUITE_NAMES = dict(
    P1='planted: creations follow spread >= 2 AND fair_gap > 1 -> spread and fair_gap supported (any, deep: RR > 1.5, p_hi <= 0.05, >= 5 markets)',
    P4='planted: btc with / against, own depth and flow have no exposure variation -> insufficient_support (the verdict can say so)',
    P5='final verdict = D0 unless D1000 differs (timing_sensitive), over all 35 tests of both worlds; every verdict declared',
    P6='improve availability: spread-1 rows leave every improve ratio, improve spread contrast is 3+ vs 2 (person-time recounted from the panel)',
    R1='random twins: no supported / supported_lower verdict (final and D0) on any of the 35 tests',
    R2='random twins: the negative control is testable (any x spread / fair_gap / time_left: >= 5 supported markets, >= 30 events)',
    R3='random twins: pooled MH RR exactly 1 (twin symmetry, exact time-left balance); reciprocal market RRs; time-left ties in every market',
    FS='statuses: straddle -> status (not an event; touched rows counted), gap, unknown_cell, stale rows at t and t - 1 s, events with their keys',
    FC='causality: later frame / BTC / fill ARRIVAL TIMES moved -> every D0 / D1000 feature at t <= t* unchanged; later flow and BTC features change',
    FI='identity: G2 front with a containing window -> candidate_constrained (+ adding inventory); window opens later -> owner_unknown; G4 / '
       'non-top-3 -> candidate_scenario; no-fill -> owner_unknown',
    FI_v2='identity: a settlement-only first fill (v1 front by settlement) has no candidate under the v2 straddle rule')


def run_suite(FX, W):
    for nm, fn in SUITE:
        r = section(nm, fn, FX, W)
        if r:
            check(f'{nm}: {SUITE_NAMES[nm]}', r[0], r[1])


def test_mutants(FX, W):
    for m_, names in EXPECT_CATCH.items():
        os.environ['MT_PO_MUT'] = m_
        try:
            for nm in names:
                try:
                    ok, info = SUITE_FN[nm](FX, W)
                    check(f'mutant {m_}: caught by {nm} (the check evaluates False without raising)', not ok, info)
                except Exception as e:  # noqa: BLE001
                    check(f'mutant {m_}: caught by {nm} (the check evaluates False without raising)', False, f'raised {type(e).__name__}: {e}')
        finally:
            os.environ.pop('MT_PO_MUT', None)
    check('mutants: every declared MT_PO_MUT is exercised', set(EXPECT_CATCH) == set(PO.MUTANTS), sorted(EXPECT_CATCH))


# ------------------------------------------------------------------ pure pieces ------------------------------------------------------------------
def test_bins():
    check('bins: spread 1 / 2 / 3-4 / >= 5 / missing (inf, NaN, crossed)',
          PO.spread_codes([1, 2, 3, 4, 5, 9, np.inf, np.nan, 0, -1]).tolist() == [0, 1, 2, 2, 3, 3, 4, 4, 4, 4])
    check('bins: depth 0 / (0,50] / (50,200] / >200 / missing', PO.depth_codes([0, .5, 50, 50.01, 200, 201, np.nan]).tolist() == [0, 1, 1, 2, 2, 3, 4])
    check('bins: fair_gap <= 0 / (0,1] / (1,3] / >3 / missing', PO.fg_codes([-1, 0, .5, 1, 1.01, 3, 3.5, np.nan, np.inf]).tolist() == [0, 0, 1, 1, 2, 2, 3, 4, 4])
    check('relation: bid while short / ask while long reduce; bid while long / ask while short add; flat',
          PO.relation([0, 0, 1, 1, 0, 1], [-5, 5, 5, -5, 0, 1e-9]).tolist() == ['reducing', 'adding', 'reducing', 'adding', 'from_flat', 'from_flat'])


def toy_part():
    """One market, side 0 only (side 1 rows book_missing): flow exposure; stratum 1 (tl bin 2) T1 = 10 a = 4, T0 = 10 b = 2; stratum 2 (tl bin
    3) T1 = 5 a = 3, T0 = 15 b = 1 -> MH = (4 * 10 / 20 + 3 * 15 / 20) / (2 * 10 / 20 + 1 * 5 / 20) = 3.4, crude (7 / 15) / (3 / 25)."""
    K = 40
    ws = CX.day_ms(DAY) + 3_600_000
    rows = dict(side=np.repeat(np.array([0, 1], np.int8), K), k=np.tile(np.arange(K, dtype=np.int64), 2),
                t=np.tile(ws + 1000 * np.arange(K, dtype=np.int64), 2))
    for d in ('0', '1'):
        rows[f'st_{d}'] = np.r_[np.zeros(K, np.int8), np.ones(K, np.int8)]
        for f, v in (('sp_c', 1), ('fg_c', 1), ('btc_c', 2), ('d_own_c', 2)):
            rows[f'{f}_{d}'] = np.full(2 * K, v, np.int8)
        rows[f'tl_c_{d}'] = np.tile(np.where(np.arange(K) < 20, 2, 3), 2).astype(np.int8)
        fw = np.zeros(K, np.int8)
        fw[0:10] = 1
        fw[20:25] = 1
        rows[f'flow_c_{d}'] = np.r_[fw, fw]
    E = np.zeros(2 * K, np.int8)
    E[[0, 1, 2, 3, 10, 11, 20, 21, 22, 25]] = 1
    rows['E_any'] = E
    rows['imp_pop'] = np.ones(2 * K, bool)
    return dict(day=DAY, mid=990001, ws=ws, we=ws + 1000 * K, K=K, rows=rows, U=PO.offsets(DAY, 990001, K), events=None, counts=Counter())


def test_mh():
    P = PO.assemble([toy_part()])
    ev = np.flatnonzero(P['E_any'] > 0)
    r = PO.ratio_block(P, 'D0', 'any', 'flow', ev, PO.shift_dest(P, ev))[0]
    check('MH: the hand-computed 2-stratum toy (RR_MH 3.4, crude 3.889, T 15 / 25, events 7 / 3, 2 informative strata) through ratio_block',
          abs(r['RR_MH'] - 3.4) < 1e-12 and abs(r['RR_crude'] - (7 / 15) / (3 / 25)) < 1e-12 and r['T_exposed'] == 15 and r['T_reference'] == 25
          and r['a_exposed'] == 7 and r['b_reference'] == 3 and r['n_strata_informative'] == 2 and r['rows_status_excluded'] == 40,
          {k: r[k] for k in ('RR_MH', 'RR_crude', 'T_exposed', 'T_reference', 'a_exposed', 'b_reference', 'n_strata_informative')})


def test_verdict():
    V = PO.verdict
    cases = [((2.0, 1, 400, 40, 6, 5, 1), 'supported'), ((0.5, 400, 1, 40, 6, 1, 5), 'supported_lower'), ((2.0, 1, 400, 29, 6, 5, 1), 'insufficient_support'),
             ((2.0, 1, 400, 40, 4, 4, 0), 'insufficient_support'), ((2.0, 1, 400, 40, 6, 3, 3), 'not_supported'), ((2.0, 26, 1, 40, 6, 6, 0), 'not_supported'),
             ((1.0 + 1e-15, 1, 1, 40, 6, 6, 6), 'not_supported'), ((0.5, 400, 26, 40, 6, 0, 6), 'not_supported'), ((2.0, 25, 400, 30, 5, 4, 1), 'supported'),
             ((float('nan'), None, None, 40, 6, 6, 0), 'not_supported'), ((float('inf'), 1, 501, 40, 6, 6, 0), 'supported')]
    got = [(a, V(*a), e) for a, e in cases]
    check('verdict rule: supported / supported_lower (mirror) / insufficient (< 30 events, < 5 markets) / not_supported (share, p, tie, NaN)',
          all(g == e for _, g, e in got), [(a, g, e) for a, g, e in got if g != e])


def test_shifts(FX, W):
    P = W['random']['P']
    U1, U2 = PO.offsets(DAY, 950001, 120), PO.offsets(DAY, 950001, 120)
    check('shift offsets: deterministic per (seed, day, mid, side), sides independent, in [0, K)',
          np.array_equal(U1, U2) and not np.array_equal(U1[0], U1[1]) and U1.shape == (2, PO.N_SHIFT) and U1.min() >= 0 and U1.max() < 120)
    ev = np.flatnonzero(P['E_any'] > 0)
    SH = PO.shift_dest(P, ev)
    nms = int(P['ms'].max()) + 1
    cnt0 = np.bincount(P['ms'][ev], minlength=nms)
    keep = all(np.array_equal(np.bincount(P['ms'][SH[j]], minlength=nms), cnt0) for j in range(0, PO.N_SHIFT, 25))
    stay = bool((P['ms'][SH] == P['ms'][ev][None, :]).all())
    bij = all(len(np.unique(SH[j])) == len(ev) for j in range(0, PO.N_SHIFT, 25))
    Uz = P['U'].copy()
    P['U'] = np.zeros_like(Uz)
    ident = bool((PO.shift_dest(P, ev) == ev[None, :]).all())
    P['U'] = Uz
    check('shift: every replicate keeps each market-side\'s event-second count, stays in it, is a bijection; offset 0 is the identity',
          keep and stay and bij and ident and SH.shape == (PO.N_SHIFT, len(ev)), dict(keep=keep, stay=stay, bij=bij, ident=ident))


# ------------------------------------------------------------------ queue_front_v2 adapter ------------------------------------------------------------------
def test_qf2(FX, tmp):
    m = FX['FI']
    base = _fi_part(FX)
    E = base['events']
    e1 = E[E['qf_status_v1'] != ''].copy()
    Q = pd.DataFrame(dict(day=e1['day'].astype(str), mid=e1['mid'].astype(np.int64), side=e1['side'], tick=e1['tick'].astype(np.int64),
                          frame=e1['frame'].astype(np.int64), t_open_ms=e1['t_open_ms'], t_create_ms=e1['t_create_ms'], unit='E1',
                          status=np.where(e1['qf_status_v2'] == 'front_identified', 'front_candidate', e1['qf_status_v2']),
                          holder=e1['cand_addr'], front_hash=e1['cand_hash'], extra_col=1))
    p2 = _fi_part(FX, Q.reset_index(drop=True))
    E2 = p2['events']
    n_c = int(E['identity'].isin(PO.IDENT[:2]).sum())
    check('queue_front_v2 adapter: the same identities as the fallback when its candidates are the fallback\'s (source recorded, agreement counted)',
          E2['identity'].tolist() == E['identity'].tolist() and E2['identity_reason'].tolist() == E['identity_reason'].tolist()
          and set(E2.loc[E2['identity'].isin(PO.IDENT[:2]), 'cand_source']) == {'queue_front_v2'} and n_c >= 3
          and int(p2['counts']['qf2_vs_fallback_both_same_address']) == int((E['qf_status_v2'] == 'front_identified').sum())
          and int(p2['counts'].get('qf2_only', 0)) == 0 and int(p2['counts'].get('fallback_only', 0)) == 0,
          dict(n_cand=n_c, counts={k: v for k, v in p2['counts'].items() if k.startswith('qf2') or k.endswith('_only')}))
    qd = tmp / 'qf2'
    qd.mkdir()
    Qd = Q.rename(columns={'holder': 'candidate_addr', 'front_hash': 'candidate_hash'})   # queue_front_v2's actual levels.csv header (integrator alignment)
    Qd.to_csv(qd / 'levels.csv', index=False)
    (qd / 'receipt.json').write_text(json.dumps(dict(outputs={'levels.csv': dict(sha256=PO._sha(qd / 'levels.csv'))})))
    h = {}
    Dq, info = PO.read_qf2(qd, [(m.day, m.mid)], h)
    check('queue_front_v2 reader: levels.csv bound to its receipt, the run markets kept, the candidates counted, both files hashed',
          len(Dq[(m.day, m.mid)]) == len(Q) and info['candidates'] == int((Q['status'] == 'front_candidate').sum()) and len(h) == 2
          and {'holder', 'front_hash'} <= set(Dq[(m.day, m.mid)].columns)
          and Dq[(m.day, m.mid)]['holder'].tolist() == Q['holder'].astype(str).tolist(), info)
    (qd / 'levels.csv').write_text((qd / 'levels.csv').read_text() + '\n')
    expect('queue_front_v2 reader: a levels.csv unlike its receipt -> refused', lambda: PO.read_qf2(qd, [(m.day, m.mid)], {}), "differs from its receipt")
    for nm, Dx, needle in (('without the candidate_addr column', Qd.drop(columns=['candidate_addr']), 'lacks'),
                           ('with the pre-alignment holder header', Q, 'lacks'),
                           ('without any front_candidate status', Qd.assign(status='front_identified'), 'carries no')):
        Dx.to_csv(qd / 'levels.csv', index=False)
        (qd / 'receipt.json').write_text(json.dumps(dict(outputs={'levels.csv': dict(sha256=PO._sha(qd / 'levels.csv'))})))
        expect(f'queue_front_v2 reader: a levels.csv {nm} -> refused (assumed interface)', lambda: PO.read_qf2(qd, [(m.day, m.mid)], {}), needle)


# ------------------------------------------------------------------ refusals ------------------------------------------------------------------
def test_refusals(tmp):
    d = tmp / 'exists'
    d.mkdir()
    expect('refuse: write-once on an existing output directory (overwrite)', lambda: PO.TL2.write_out(d, {}, {}, {}, lambda o: {}), 'write-once')
    for ev_, m_, needle in (('MT_PO_MUT', 'assign_straddle', 'MT_PO_MUT'), ('MT_PO_MUT', 'bogus', 'not a declared mutant'),
                            ('MT_QF_MUT', 'drop_no_fill', 'MT_QF_MUT'), ('MT_TL_MUT', 'leak_1s', 'MT_TL_MUT')):
        os.environ[ev_] = m_
        try:
            expect(f'refuse: production with {ev_}={m_} before any input is read', lambda: PO.execute('pilot', (PO.PILOT_DAY,), tmp / 'never'), needle)
        finally:
            os.environ.pop(ev_, None)
    expect('refuse: a pilot day outside the measurement days (before any path is built)', lambda: PO.execute('pilot', (C.ID_DAY,), tmp / 'never'),
           'measurement days')
    check('refuse: nothing written by a refused production call', not (tmp / 'never').exists())
    g = PO._git([f for f in PO.TOOL_FILES if Path(f).exists()])
    needle = 'freeze' if not g.get('unmodified') else 'no recorded pilot projection'
    expect(f'refuse: the full run before any input ({needle})', lambda: PO.execute('run', C.MEASURE_DAYS, tmp / 'run_out', pilot_receipt=tmp / 'absent.json'),
           needle)
    expect('refuse: the full run without a recorded pilot projection', lambda: PO.preflight_projection(tmp / 'absent.json', {}), 'no recorded pilot projection')
    for nm, J, needle in (('p121.json', dict(kind='pilot', version=PO.VERSION, projection=dict(full_run_cpu_s=121.0)), '> 120 s (refused before launch)'),
                          ('pother.json', dict(kind='pilot', version='queue_front_v1', projection=dict(full_run_cpu_s=10.0)), 'not a posting_opportunity_v1')):
        (tmp / nm).write_text(json.dumps(J))
        expect(f'refuse: a recorded pilot receipt {nm} ({needle})', lambda p=tmp / nm: PO.preflight_projection(p, {}), needle)
    (tmp / 'p60.json').write_text(json.dumps(dict(kind='pilot', version=PO.VERSION, projection=dict(full_run_cpu_s=60.0))))
    h = {}
    r = PO.preflight_projection(tmp / 'p60.json', h)
    check('preflight: a 60 s projection passes and the pilot receipt is hashed as an input', r['full_run_cpu_s'] == 60.0 and len(h) == 1)


# ------------------------------------------------------------------ the pilot path end to end ------------------------------------------------------------------
def write_rg(rdir, rows, n=None):
    rdir.mkdir(parents=True, exist_ok=True)
    D = pd.DataFrame(rows, columns=list(PO.RG_DT))
    D.to_csv(rdir / 'post_regrade_orders.csv', index=False)
    (rdir / 'receipt_post.json').write_text(json.dumps(dict(outputs={'post_regrade_orders.csv': dict(sha256=PO._sha(rdir / 'post_regrade_orders.csv'))},
                                                           rows=dict(post_regrade_orders=len(D) if n is None else n))))
    return D


def test_e2e(tmp, FX):
    mks = [FX['FI'], FX['FS']]
    root = tmp / 'root'
    (root / 'econ').mkdir(parents=True)
    (root / 'econ' / 'identities.json').write_text(json.dumps(dict(ex_ante=dict(day=C.ID_DAY, top10_shares=[dict(addr=a, rank=i + 1)
                                                                                                            for i, a in enumerate(PO.TOP3)]))))
    fvp = root / 'econ' / 'fills_valued.parquet'
    pd.DataFrame(dict(x=[1, 2, 3])).to_parquet(fvp, index=False)
    fsha = PO._sha(fvp)
    rows = rg_rows(FX['FI'], FI_GRADES) + [dict(day=DAY2, mid=999999, rank=1, addr=A1, mk_hash='0xhelsewhere', y_side='bid', tick=40, post_lo_union_ms=1,
                                               post_hi_ms=2, grade=PO.G2)]
    rgd = tmp / 'rg'
    write_rg(rgd, rows)
    hh = {}
    RG, info = PO.read_regrade(rgd, [(m.day, m.mid) for m in mks], hh)
    check('regrade reader: bound to its receipt; the run markets kept (another market dropped); both files hashed',
          info['rows'] == len(rows) and info['rows_run'] == len(rows) - 1 and len(hh) == 2 and '0xhC1' in RG[(DAY, FX['FI'].mid)], info)
    bad = tmp / 'rg_bad'
    write_rg(bad, rows)
    (bad / 'post_regrade_orders.csv').write_text((bad / 'post_regrade_orders.csv').read_text().replace('G4_scenario_only', 'G4_other'))
    expect('regrade reader: a tampered post_regrade_orders.csv -> refused', lambda: PO.read_regrade(bad, [(DAY, FX['FI'].mid)], {}), "differs from its receipt")
    built = {(m.day, m.mid): m.build() for m in mks}
    book = {DAY: sorted(m.mid for m in mks)}
    out = tmp / 'e2e_pilot'
    rec = PO.execute('pilot', (DAY,), out, inp=root, loader=lambda d, mid, r, hc: built[(d, mid)], book=book, regrade_dir=rgd, fv_sha=fsha, log=lambda s: None)
    names = sorted(p.name for p in out.iterdir())
    check('e2e: execute (pilot path, injected loader) writes exactly the seven outputs', names == sorted(PO.OUT_FILES), names)
    Rj = json.loads((out / 'receipt.json').read_text())
    pj = Rj.get('projection') or {}
    dep = Rj.get('dependency') or {}
    check('e2e: receipt.json hashes every other output, the sources and inputs; a finite projection; no mutant; the identity dependency named',
          all(Rj['outputs'][f]['sha256'] == PO._sha(out / f) for f in PO.OUT_FILES if f != 'receipt.json') and Rj['mutant'] == ''
          and len(Rj['sources']) >= 10 and Rj['inputs'].get('econ/fills_valued.parquet') == fsha and isinstance(pj.get('full_run_cpu_s'), float)
          and dep.get('identity_source') == 'v1_functions_v2_straddle' and rec['kind'] == 'pilot'
          and any(k.endswith('post_regrade_orders.csv') for k in Rj['inputs']),
          dict(proj=pj, n_src=len(Rj['sources']), dep=dep.get('identity_source')))
    RW = pd.read_parquet(out / 'rows.parquet')
    EVc = pd.read_csv(out / 'events.csv', dtype={'day': str}, keep_default_na=False)
    RT = pd.read_csv(out / 'ratios.csv')
    IDc = pd.read_csv(out / 'identity.csv', dtype={'day': str}, keep_default_na=False)
    Cl = json.loads((out / 'claims.json').read_text())
    ev = EVc[EVc['status'] == 'event']
    keys = sorted(k for x in RW['keys_any'] if x for k in x.split(';'))
    check('e2e: rows.parquet = 2 sides x K seconds per market; events.csv events = sum N_any; the cell-event keys equal',
          len(RW) == sum(2 * m.K for m in mks) and len(ev) == int(RW['N_any'].sum()) and keys == sorted(ev['key'].tolist()),
          dict(rows=len(RW), events=len(ev), n_any=int(RW['N_any'].sum())))
    r0 = RT[(RT['variant'] == 'D0') & (RT['type'] == 'any') & (RT['contrast'] == 'spread')].iloc[0]
    okr = RW['status_D0'] == 'ok'
    t1 = int((okr & RW['spread_bin_D0'].isin(['2', '3-4', '>=5'])).sum())
    t0 = int((okr & (RW['spread_bin_D0'] == '1')).sum())
    a1 = int((okr & RW['spread_bin_D0'].isin(['2', '3-4', '>=5']) & (RW['E_any'] > 0)).sum())
    check('e2e: ratios.csv = 2 variants x 5 types x 7 contrasts with declared verdicts; D0 any x spread person-time / events recounted from rows.parquet',
          len(RT) == 70 and set(RT['verdict']) <= set(PO.VERDICTS) and int(r0['T_exposed']) == t1 and int(r0['T_reference']) == t0
          and int(r0['a_exposed']) == a1, dict(T1=(int(r0['T_exposed']), t1), T0=(int(r0['T_reference']), t0), a=(int(r0['a_exposed']), a1)))
    po = IDc[(IDc['table'] == 'owner') & (IDc['scope'] == 'pooled') & (IDc['population'] == 'event') & (IDc['group'] == 'all')]
    idn = {r['category']: int(r['n']) for r in po.to_dict('records')}
    check('e2e: identity.csv pooled owner counts = events.csv identities; the FI fixture links two candidates and leaves anonymous creations in',
          idn == {k: int((ev['identity'] == k).sum()) for k in PO.IDENT} and idn.get('candidate_constrained') == 1 and idn.get('candidate_scenario') == 2
          and idn.get('owner_unknown', 0) >= 3, idn)
    vl = Cl['verdict_lists']
    check('e2e: claims - reconciliation predicates all true; the verdict lists partition the 35 tests; descriptive only, no intervals',
          Cl['reconciliation']['all'] is True and sum(len(v) for v in vl.values()) == 35 and Cl['descriptive_only'] is True and Cl['intervals'] is None,
          dict(rec=Cl['reconciliation'], lists={k: len(v) for k, v in vl.items()}))
    expect('e2e: a second write into the written directory is refused (before any input is read)',
           lambda: PO.execute('pilot', (DAY,), out, inp=root, loader=lambda d, mid, r, hc: built[(d, mid)], book=book, regrade_dir=rgd, fv_sha=fsha,
                              log=lambda s: None), 'write-once')
    expect('e2e: a fills_valued unlike the binding -> refused', lambda: PO.execute('pilot', (DAY,), tmp / 'e2e_badfv', inp=root,
                                                                                   loader=lambda d, mid, r, hc: built[(d, mid)], book=book,
                                                                                   regrade_dir=rgd, fv_sha='0' * 64, log=lambda s: None),
           'differs from the accepted binding')


def main():
    t0 = time.process_time()
    check('selftest environment: MT_RUN=v2 and no MT_PO_MUT / reused-tool mutant',
          C.RUN == 'v2' and all(e not in os.environ for e in ('MT_PO_MUT',) + PO.FOREIGN_MUT), C.RUN)
    tmp = Path(tempfile.mkdtemp(prefix='posting_opportunity_selftest_'))
    try:
        section('bins', test_bins)
        section('mh', test_mh)
        section('verdict', test_verdict)
        FX = section('fixtures', fixtures)
        W = section('worlds', worlds)
        if FX and W:
            run_suite(FX, W)
            section('shifts', test_shifts, FX, W)
            section('mutants', test_mutants, FX, W)
            section('qf2', test_qf2, FX, tmp)
            section('e2e', test_e2e, tmp, FX)
        section('refusals', test_refusals, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop('MT_PO_MUT', None)
    fails = [r for r in RES if not r[1]]
    for name, ok, info in RES:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  | {info}'))
    print(json.dumps(dict(checks=len(RES), failed=len(fails), cpu_s=round(time.process_time() - t0, 3), process_cpu_s=round(time.process_time(), 3),
                          vmhwm_mib=round(QF.hwm(), 1))))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
