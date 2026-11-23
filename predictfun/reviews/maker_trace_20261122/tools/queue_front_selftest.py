"""queue_front_selftest.py - synthetic proof of queue_front.py (P-2026-005 MAKER-TRACE, QUEUE-FRONT v1; queue_front_contract.md fixtures 1-7).
The REAL chain_context pipeline (build_market -> pf_mt_lifecycle._book, frame_times, level_at_frames, the per-frame best ticks, level_size)
runs on synthetic BOOK markets built frame by frame (200 ms frames); every expectation is planted or hand-computed. Fixtures: (1) a planted
level creator filled first -> front_identified with the right holder and its behind fill; (2) a cancel ahead (an unexplained drop before the
first fill) -> unexplained_decrease, never identified; Amendment 2 (f4c7e5cbd, records timed by the book): (3) a two-record race with
overlapping settlement windows resolved by the book intervals, (3s) a one-record sweep of two makers -> ambiguous_order with both addresses
in the upper front-share bound only (Amendment 1), (3c) a record with no visible decrease is settlement_only only when its window opens after
upd(f-1), (7) an old-life record settling after the new creation but book-timed before f - 1 is excluded; Amendment 1 (5c8a4b1a4): (3b) one
maker's own records that could each be first -> front_identified; (4) levels
created and cancelled / never filled -> no_fill, in every denominator (levels, shares, the fresh-level mix); (5) mechanisms improve (+ speed
800 ms) / deep / at_empty_touch / re_establish (+ speed 1,000 ms, bounds 800 / 1,200) / a cancel-cleared or 6 s late re-post (not
re_establish) / an unknown touch (other); (6) causality: frames after the front fill's reflection (drops, a zero, a re-creation, a gap) and
a later fill leave its status and holder unchanged, while a cancel before the fill changes it (positive control); (7) the MT_QF_MUT mutants
first_fill_by_settle / ignore_decrease / drop_no_fill / same_maker_ambiguous (the original rule) / settle_only_timing (the r2 rule) /
ignore_old_life, flipped in-process, each caught by a named check evaluating False without raising.
Also: every other status and reason (gap / unknown cell before the fill, creation across a gap, prev cell unknown, the fill window before
creation, an unbounded first fill or member, a straddling front attributed, a fill spanning two levels, a gap after the level end,
unattributable members), the fill-to-cell counts, the declared null (a planted front winner -> p_hi = 1/501, exchangeable labels -> not
small, determinism, per-market independence), the four readings on a planted world (all read) and an exchangeable world (none read while
testable), the markout cohort equal to chain_post_fill.pnl_rows, the input readers (row-group pruning, dictionary columns, refusals with
their reasons), the production / overwrite refusals (each asserting its message) and the pilot path end to end through execute() with an
injected loader. Usage (run_light, MT_RUN=v2): queue_front_selftest.py"""
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
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import queue_front as QF  # noqa: E402

CX, C, E, LC = QF.CX, QF.C, QF.E, QF.LC
NAN = float('nan')
RES = []
A1, A2, A3 = QF.TOP3
A1L, A2L, A3L = QF.TOP3_L
OTHER, OTHER2, OTHER3 = '0x' + 'ab' * 20, '0x' + 'cd' * 20, '0x' + 'ef' * 20
DAY, DAY2 = '20261024', '20261025'
BG = {('bid', 5): 10., ('ask', 95): 10.}
BASE = {('bid', 40): 100., ('ask', 60): 100., **BG}
TY = {'str': object, 'int64': np.int64, 'float64': np.float64, 'bool': bool}
TOBS = {c: C.SCHEMAS['tob'][c] for c in CX.TOB_READ}
LIFE = dict(mid='int64', sid='str', mk_hash='str', mk_idx='int64', y_side='str', y_ptick='int64', shares='float64', hi_ms='int64', lo_ms='int64',
            status='str', x_lo_hard_ms='int64')
ORD = dict(scenario='str', mid='int64', mk_hash='str', mk_signer='str', y_side='str', y_ptick='int64', persist_lo_ms='int64', persist_hi_ms='int64',
           status='str')
FVT = dict(day='str', mid='int64', sid='str', mk_idx='int64', role='str', addr='str', hash='str', y_side='str', s='int64', q='float64', p='float64',
           self_trade='bool', settle_ms='int64', status='str', st_res='str', st_1s='str', v_1s='float64', st_10s='str', v_10s='float64', st_60s='str',
           v_60s='float64')
PFC = ('day', 'mid', 'rank', 'addr', 'mk_hash', 'y_side', 'tick', 'n_fills', 'cum_shares', 'post_lo_union_ms', 'post_hi_ms', 'window_status',
       'queue_class')
BTC = dict(BT=dict(G0=0, n=1, ok=np.ones(1, bool), lm=np.zeros(1), okc=np.array([0, 1], np.int64)), code=np.zeros(1, np.int8), day=DAY,
           load_counts={}, files=[], cum=None)
VERDICTS = set(QF.READINGS) | set(QF.MECHS) | {'unidentified', 'insufficient_support'}
EXPECT_CATCH = dict(first_fill_by_settle=('F3c',), ignore_decrease=('F2', 'F6'), drop_no_fill=('F4',), same_maker_ambiguous=('F3b',),
                    settle_only_timing=('F3', 'F7'), ignore_old_life=('F7',))


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


def _brief(r):
    if r is None:
        return None
    keys = ('unit', 'status', 'reason', 'holder_group', 'holder', 'mechanism', 'n_members', 'n_behind', 'n_unattributed', 'speed_ms', 'speed_lo_ms',
            'speed_hi_ms', 'straddles_end', 'n_overlap', 'overlap_one_address', 'amb_addresses', 'amb_gmask', 'n_unexplained', 'end_type',
            'best_prev_tick', 'front_timing', 'front_book_ms', 'n_old_life')
    return {k: r.get(k) for k in keys}


# ------------------------------------------------------------------ synthetic BOOK markets ------------------------------------------------------------------
class MK:
    """One synthetic BOOK market: usable frames every `step` ms over [ws - 1 s, ws + t1] (cells {(side, tick): shares}, unknown cells as a
    non-ok level row), maker legs (lifecycle fills: hard window [hi - 2 s, hi] or unbounded), their fills_valued maker rows and orders."""

    def __init__(self, day, mid, hour, t1=30_000, cells=None, step=200):
        self.day, self.mid = str(day), int(mid)
        self.ws = CX.day_ms(self.day) + 3_600_000 * int(hour)
        self.we = self.ws + 300_000
        base = dict(BASE if cells is None else cells)
        self.frames = {self.ws + r: dict(base) for r in range(-1000, int(t1) + 1, int(step))}
        self.unk, self.legs, self.n, self._built = {}, [], 0, None

    def setc(self, ra, rb, side, tick, sh):
        for t, bk in self.frames.items():
            if self.ws + ra <= t < self.ws + rb:
                if sh > 0:
                    bk[(side, int(tick))] = float(sh)
                else:
                    bk.pop((side, int(tick)), None)
        return self

    def unknown(self, ra, rb, side, tick):
        for t in self.frames:
            if self.ws + ra <= t < self.ws + rb:
                self.unk.setdefault(t, set()).add((side, int(tick)))
        return self

    def drop(self, ra, rb):
        for t in [x for x in self.frames if self.ws + ra < x < self.ws + rb]:
            del self.frames[t]
        return self

    def fill(self, addr, side, tick, q, rhi, unbounded=False, status='ok', fv=True, fv_status='ok', v=(1., 2., 3.), self_trade=False,
             st=('ok', 'ok', 'ok'), st_res='ok', sid=None, mk_idx=0):
        self.n += 1
        hi = self.ws + int(rhi)
        L = dict(sid=f'{self.mid}-{self.n:04d}' if sid is None else f'{self.mid}-{sid}', mk_idx=int(mk_idx), mk_hash=f'0xh{self.mid}{self.n:04d}',
                 addr=addr, side=side, tick=int(tick), q=float(q),
                 hi=hi, lo=-1 if unbounded else hi - QF.MAX_LAG_MS, status=status, fv=fv, fv_status=fv_status, v=tuple(float(x) for x in v),
                 self_trade=bool(self_trade), st=tuple(st), st_res=st_res)
        self.legs.append(L)
        self._built = None
        return L

    def fv_rows(self, taker=False):
        rows = []
        for L in self.legs:
            if not L['fv']:
                continue
            s = 1 if L['side'] == 'bid' else -1
            r = dict(day=self.day, mid=self.mid, sid=L['sid'], mk_idx=L['mk_idx'], role='maker', addr=L['addr'], hash=L['mk_hash'], y_side=L['side'],
                     s=s, q=L['q'], p=L['tick'] / 100., self_trade=L['self_trade'], settle_ms=L['hi'], status=L['fv_status'], st_res=L['st_res'],
                     st_1s=L['st'][0], v_1s=L['v'][0], st_10s=L['st'][1], v_10s=L['v'][1], st_60s=L['st'][2], v_60s=L['v'][2])
            rows.append(r)
            if taker:
                rows.append(dict(r, role='taker', addr=OTHER3, hash='tk' + L['sid'], y_side='ask' if s > 0 else 'bid', s=-s, v_1s=-L['v'][0],
                                 v_10s=-L['v'][1], v_60s=-L['v'][2]))
        return rows

    def fvm(self):
        df = typed(self.fv_rows(), FVT)
        df['addr_l'] = np.array([str(a).lower() for a in df['addr'].tolist()], dtype=object)
        return df

    def build(self):
        if self._built is not None:
            return self._built
        fr, lv, tb = [], [], []
        for fid, t in enumerate(sorted(self.frames)):
            bk, uk = self.frames[t], self.unk.get(t, set())
            recv = (t + 40) * 1_000_000 + fid
            fr.append(dict(mid=self.mid, fid=fid, upd_ms=t, recv_ns=recv, conn='c1', order_count=-1, pend_bids='[]', pend_asks='[]', los_id='',
                           los_side='', los_price=NAN, los_outcome='', status='ok'))
            known = {k: v for k, v in bk.items() if k not in uk and v > 0}
            for (sd, tk), sh in sorted(known.items()):
                lv.append(dict(mid=self.mid, fid=fid, side=sd, ptick=int(tk), shares=float(sh), status='ok'))
            for (sd, tk) in sorted(uk):
                lv.append(dict(mid=self.mid, fid=fid, side=sd, ptick=int(tk), shares=1.0, status='bad_level'))
            bids = sorted(tk for (sd, tk) in known if sd == 'bid')
            asks = sorted(tk for (sd, tk) in known if sd == 'ask')
            tb.append(dict(mid=self.mid, upd_ms=t, recv_ns=recv, conn='c1', n_bid=len(bids), n_ask=len(asks),
                           best_bid=bids[-1] / 100. if bids else NAN, best_bid_sh=known[('bid', bids[-1])] if bids else 0.,
                           best_ask=asks[0] / 100. if asks else NAN, best_ask_sh=known[('ask', asks[0])] if asks else 0., status='ok'))
        life = [dict(mid=self.mid, sid=L['sid'], mk_hash=L['mk_hash'], mk_idx=L['mk_idx'], y_side=L['side'], y_ptick=L['tick'], shares=L['q'],
                     hi_ms=L['hi'], lo_ms=-1, status=L['status'], x_lo_hard_ms=L['lo']) for L in self.legs]
        orders = [dict(scenario=s, mid=self.mid, mk_hash=L['mk_hash'], mk_signer=L['addr'], y_side=L['side'], y_ptick=L['tick'], persist_lo_ms=-1,
                       persist_hi_ms=-1, status='ok') for L in self.legs for s in ('S_FIFO', 'S_FIFO_LO', 'S_NOPRIO')]
        M = CX.build_market(self.day, self.mid, self.ws, self.we, QF.TOP3, btc=BTC, book=True, tob=typed(tb, TOBS), frames=typed(fr, C.SCHEMAS['frames']),
                            levels=typed(lv, C.SCHEMAS['levels']), legs=None, life_fills=typed(life, LIFE), orders=typed(orders, ORD))
        self._built = (M, self.fvm())
        return self._built

    def part(self, pfm=None):
        M, fvm = self.build()
        return QF.market_pass(self.day, self.mid, M, fvm, pfm)


def lrow(P, side, tick, rt, mk):
    L = P['levels']
    if not len(L):
        return None
    m = ((L['side'] == side).to_numpy() & (L['tick'].to_numpy(np.int64) == tick) & (L['t_create_ms'].to_numpy(np.int64) == mk.ws + rt))
    return L[m].iloc[0].to_dict() if int(m.sum()) == 1 else None


def fixtures():
    F = {}
    m = MK(DAY, 910001, 1)                                            # (1) planted creator, behind fill, an unshown top-3 fill, an off-grid leg
    m.setc(2000, 4600, 'bid', 45, 100).setc(4600, 8800, 'bid', 45, 70).setc(8800, 20000, 'bid', 45, 50)
    m.fill(A1, 'bid', 45, 30, 5000)
    m.fill(OTHER, 'bid', 45, 20, 9000)
    m.fill(A1, 'ask', 80, 5, 10000)
    m.fill(A2, 'bid', 45, 3, 25000, status='subcent_price')
    F['F1'] = m
    m = MK(DAY, 910002, 2)                                            # (2) a cancel ahead at +2.6 s, before the first fill's window
    m.setc(2000, 2600, 'bid', 45, 100).setc(2600, 4600, 'bid', 45, 60).setc(4600, 8800, 'bid', 45, 30).setc(8800, 20000, 'bid', 45, 10)
    m.fill(A1, 'bid', 45, 30, 5000)
    m.fill(OTHER, 'bid', 45, 20, 9000)
    F['F2'] = m
    for nm, mid, hr, b2 in (('F3', 910003, 3, OTHER), ('F3b', 910013, 4, A2)):   # (3) a two-record race / (3b) one maker's own records
        m = MK(DAY, mid, hr)
        m.setc(2000, 4600, 'bid', 45, 100).setc(4600, 5400, 'bid', 45, 70).setc(5400, 20000, 'bid', 45, 50)
        m.fill(A2, 'bid', 45, 30, 5000)
        m.fill(b2, 'bid', 45, 20, 5800)
        if nm == 'F3b':
            m.fill(A2, 'bid', 45, 7, 4800)                            # no 7-sh decrease anywhere: untimed, could precede the front
        F[nm] = m
    m = MK(DAY, 910014, 14)                                           # (3s) one record sweeping two makers' orders at the cell
    m.setc(2000, 4600, 'bid', 45, 100).setc(4600, 20000, 'bid', 45, 50)
    m.fill(A2, 'bid', 45, 30, 5000, sid='SWP', mk_idx=0)
    m.fill(OTHER, 'bid', 45, 20, 5000, sid='SWP', mk_idx=1)
    F['F3s'] = m
    m = MK(DAY, 910015, 15)                                           # (3c) fills inside the creation interval (no visible decrease)
    m.setc(2000, 20000, 'bid', 45, 70).setc(2000, 20000, 'bid', 47, 70)
    m.fill(A1, 'bid', 45, 30, 3900)                                   # window [1.9 s, 3.9 s] opens after upd(f-1) = 1.8 s: settlement-only
    m.fill(A3, 'bid', 47, 30, 2100)                                   # window [0.1 s, 2.1 s] opens before: book_unmatched
    F['F3c'] = m
    m = MK(DAY, 910016, 16, cells={**BASE, ('bid', 45): 50.})        # (7) an old-life record settling after the new creation
    m.setc(3000, 3600, 'bid', 45, 0).setc(3600, 6800, 'bid', 45, 100).setc(6800, 20000, 'bid', 45, 70)
    m.fill(OTHER2, 'bid', 45, 50, 4200)                               # emptied the old life at +3.0 s; settles at +4.2 s (> creation +3.6 s)
    m.fill(A3, 'bid', 45, 30, 7000)                                   # the new life's first record (book-timed +6.8 s)
    F['F7'] = m
    m = MK(DAY, 910004, 5)                                            # (4) a cancelled level and a level alive to the book end; no fill at all
    m.setc(2000, 6000, 'bid', 47, 50).setc(25000, 30001, 'bid', 48, 20)
    F['F4'] = m
    c44 = {('bid', 44): 50., ('bid', 43): 20., ('ask', 60): 100., **BG}
    m = MK(DAY, 910051, 6, cells=c44)                                 # (5a) improve; the best bid changed 800 ms before the creation window
    m.setc(3000, 30001, 'bid', 44, 0).setc(4000, 20000, 'bid', 45, 30)
    F['F5a'] = m
    m = MK(DAY, 910052, 7, cells={('bid', 44): 50., ('ask', 60): 100., **BG})   # (5b) deep (+ a fill without a fills_valued row)
    m.setc(4000, 20000, 'bid', 41, 30)
    m.fill(OTHER2, 'ask', 95, 1, 15000, fv=False)
    F['F5b'] = m
    m = MK(DAY, 910053, 8, cells={('bid', 44): 50., ('bid', 5): 10.})            # (5c) at_empty_touch: no ask in f - 1
    m.setc(4000, 20000, 'ask', 60, 30)
    F['F5c'] = m
    c30 = {('bid', 44): 30., ('bid', 43): 20., ('ask', 60): 100., **BG}
    m = MK(DAY, 910054, 9, cells=c30)                                 # (5d) re_establish: cleared by a fill at +3.0 s, re-posted at +4.0 s
    m.fill(A3, 'bid', 44, 30, 3100)
    m.setc(3000, 4000, 'bid', 44, 0).setc(4000, 30001, 'bid', 44, 25)
    F['F5d'] = m
    m = MK(DAY, 910055, 10, cells=c30)                                # (5e) cleared by a cancel -> not re_establish
    m.setc(3000, 4000, 'bid', 44, 0).setc(4000, 30001, 'bid', 44, 25)
    F['F5e'] = m
    m = MK(DAY, 910056, 11, cells=c30)                                # (5f) cleared by a fill, re-posted 6.2 s later -> not re_establish
    m.fill(A3, 'bid', 44, 30, 3100)
    m.setc(3000, 9200, 'bid', 44, 0).setc(9200, 30001, 'bid', 44, 25)
    F['F5f'] = m
    m = MK(DAY, 910057, 12, cells={('bid', 44): 50., ('ask', 60): 100., **BG})   # (5g) an unknown cell above the best bid -> other
    m.unknown(-1000, 30001, 'bid', 48).setc(4000, 20000, 'bid', 46, 30)
    F['F5g'] = m
    m = MK(DAY, 910061, 13)                                           # (6) F1 with every frame after the fill's reflection perturbed
    m.setc(2000, 4600, 'bid', 45, 100).setc(4600, 5200, 'bid', 45, 70).setc(5200, 6000, 'bid', 45, 40).setc(6000, 7000, 'bid', 45, 0)
    m.setc(7000, 20000, 'bid', 45, 33).drop(10000, 12000)
    m.fill(A1, 'bid', 45, 30, 5000)
    m.fill(OTHER, 'bid', 45, 20, 9000)
    m.fill(OTHER2, 'bid', 45, 7, 15000)
    F['F6a'] = m
    m = MK(DAY, 910071, 15)                                           # S1 a gap between creation and the first fill
    m.setc(2000, 20000, 'bid', 45, 100).setc(5800, 20000, 'bid', 45, 70).drop(3000, 4500)
    m.fill(A1, 'bid', 45, 30, 6000)
    F['S1'] = m
    m = MK(DAY, 910072, 16)                                           # S2 creation across a gap; a cell unknown in f - 1
    m.drop(1000, 2500).setc(2600, 10000, 'bid', 45, 50).unknown(5000, 5001, 'bid', 47).setc(5200, 10000, 'bid', 47, 30)
    F['S2'] = m
    m = MK(DAY, 910073, 17)                                           # S3 an unknown cell between creation and the fill
    m.setc(2000, 20000, 'bid', 45, 100).unknown(3000, 3001, 'bid', 45).setc(5800, 20000, 'bid', 45, 70)
    m.fill(A1, 'bid', 45, 30, 6000)
    F['S3'] = m
    m = MK(DAY, 910074, 18)                                           # S4 the first fill's window opens before the creation window
    m.setc(2000, 20000, 'bid', 45, 100).setc(2800, 20000, 'bid', 45, 70)
    m.fill(A1, 'bid', 45, 30, 3000)
    F['S4'] = m
    m = MK(DAY, 910075, 19)                                           # S5 an unbounded first fill; an unbounded later member
    m.setc(2000, 20000, 'bid', 45, 100).setc(5800, 20000, 'bid', 45, 70)
    m.fill(A1, 'bid', 45, 30, 6000, unbounded=True, status='hi_from_receipt')
    m.setc(2000, 20000, 'bid', 47, 100).setc(5800, 20000, 'bid', 47, 70)
    m.fill(A2, 'bid', 47, 30, 6000)
    m.fill(OTHER, 'bid', 47, 10, 15000, unbounded=True, status='hi_from_receipt')
    F['S5'] = m
    m = MK(DAY, 910076, 20)                                           # S6 the first fill empties the level (straddles its zero frame)
    m.setc(2000, 5000, 'bid', 45, 100)
    m.fill(A1, 'bid', 45, 100, 5100)
    F['S6'] = m
    m = MK(DAY, 910077, 21)                                           # S7 ... while the cell is re-created before the fill's hi
    m.setc(2000, 5000, 'bid', 45, 100).setc(5200, 20000, 'bid', 45, 40)
    m.fill(A1, 'bid', 45, 100, 5300)
    F['S7'] = m
    m = MK(DAY, 910078, 22)                                           # S8 ... with a gap after the zero frame
    m.setc(2000, 5000, 'bid', 45, 100).drop(5000, 6500)
    m.fill(A1, 'bid', 45, 100, 5100)
    F['S8'] = m
    m = MK(DAY, 910080, 23)                                           # S10 a member beyond a re-creation is unattributable
    m.setc(2000, 8000, 'bid', 45, 100).setc(5800, 8000, 'bid', 45, 70).setc(8400, 20000, 'bid', 45, 50)
    m.fill(A1, 'bid', 45, 30, 6000)
    m.fill(OTHER, 'bid', 45, 20, 9000)
    F['S10'] = m
    m = MK(DAY, 910081, 17)                                           # S11 a book_ambiguous record (two equal decreases) could precede the front
    m.setc(2000, 20000, 'bid', 45, 100).setc(3000, 20000, 'bid', 45, 90).setc(3600, 20000, 'bid', 45, 80).setc(5800, 20000, 'bid', 45, 60)
    m.fill(A1, 'bid', 45, 10, 3800)
    m.fill(OTHER, 'bid', 45, 20, 6000)
    F['S11'] = m
    return F


def world(kind):
    """Six markets over two days: 9 fresh bid levels (improve) + 2 re-established asks (400 ms / 3,000 ms) per market. planted: rank 1 holds
    every bid front (behind fills by two others; front v_10s +30 c/sh, behind -20) and the fast re-establish; exchangeable: fronts cycle
    rank 1 / other / other2 with the other two behind, equal markouts, rank 1 holds the slow re-establish."""
    mks = []
    for j in range(6):
        m = MK(DAY if j < 3 else DAY2, (930000 if kind == 'planted' else 930100) + j, 2 + j, t1=75_000, cells={**BASE, ('ask', 70): 50., ('ask', 75): 50.})
        for i in range(9):
            tk, c0 = 41 + i, 2000 + 5000 * i
            hf, h1, h2 = c0 + 3000, c0 + 6000, c0 + 9000
            m.setc(c0, 70000, 'bid', tk, 100).setc(hf - 200, 70000, 'bid', tk, 90).setc(h1 - 200, 70000, 'bid', tk, 80).setc(h2 - 200, 70000, 'bid', tk, 70)
            if kind == 'planted':
                hold, b1, b2, vf, vb = A1, OTHER, OTHER2, 3.0, -2.0
            else:
                cyc = (A1, OTHER, OTHER2)
                hold = cyc[i % 3]
                b1, b2 = [a for a in cyc if a != hold]
                vf = vb = 1.0
            m.fill(hold, 'bid', tk, 10, hf, v=(vf, vf, vf))
            m.fill(b1, 'bid', tk, 10, h1, v=(vb, vb, vb))
            m.fill(b2, 'bid', tk, 10, h2, v=(vb, vb, vb))
        fast, slow = (A1, OTHER) if kind == 'planted' else (OTHER, A1)
        vf = 3.0 if kind == 'planted' else 1.0
        m.setc(3000, 3400, 'ask', 70, 0).setc(3400, 70000, 'ask', 70, 50).setc(6200, 70000, 'ask', 70, 40)
        m.fill(OTHER3, 'ask', 70, 50, 3100)
        m.fill(fast, 'ask', 70, 10, 6400, v=(vf, vf, vf))
        m.setc(20000, 23000, 'ask', 75, 0).setc(23000, 70000, 'ask', 75, 50).setc(25800, 70000, 'ask', 75, 40)
        m.fill(OTHER3, 'ask', 75, 50, 20100)
        m.fill(slow, 'ask', 75, 10, 26000, v=(vf, vf, vf))
        if kind != 'planted':
            m.setc(9200, 70000, 'ask', 70, 30)
            m.fill(slow, 'ask', 70, 10, 9400, v=(1., 1., 1.))
            m.setc(28800, 70000, 'ask', 75, 30)
            m.fill(fast, 'ask', 75, 10, 29000, v=(1., 1., 1.))
        mks.append(m)
    return mks


# ------------------------------------------------------------------ the contract fixtures (named checks; re-run under every mutant) ------------------------------------------------------------------
def chk_F1(FX):
    m = FX['F1']
    P = m.part()
    r = lrow(P, 'bid', 45, 2000, m)
    Fr = P['fronts']
    e1 = int((P['levels']['unit'] == 'E1').sum())
    ok = (r is not None and r['unit'] == 'E1' and r['status'] == 'front_identified' and r['holder'] == A1L and r['holder_group'] == 'rank1'
          and r['mechanism'] == 'improve' and int(r['n_behind']) == 1 and not bool(r['straddles_end']) and e1 == 1 and np.isnan(r['speed_ms'])
          and list(Fr['role']) == ['front', 'behind'] and list(Fr['addr']) == [A1L, OTHER] and list(Fr['group']) == ['rank1', 'other'])
    return ok, dict(row=_brief(r), e1=e1, fronts=Fr[['role', 'addr', 'group']].values.tolist() if len(Fr) else [])


def chk_F2(FX):
    m = FX['F2']
    P = m.part()
    r = lrow(P, 'bid', 45, 2000, m)
    ok = (r is not None and r['status'] == 'unexplained_decrease' and r['reason'] == 'unexplained_drop_before_fill' and r['holder'] == ''
          and int(r['n_unexplained']) >= 1 and len(P['fronts']) == 0)
    return ok, _brief(r)


def chk_F3(FX):
    """amendment 2: a two-record race (two makers, overlapping 2-s settlement windows) resolved by the book intervals (+4.6 s vs +5.4 s)."""
    m = FX['F3']
    P = m.part()
    r = lrow(P, 'bid', 45, 2000, m)
    Fr = P['fronts']
    ok = (r is not None and r['status'] == 'front_identified' and r['reason'] == 'ok' and r['holder'] == A2L and r['holder_group'] == 'rank2'
          and r['front_timing'] == 'book' and int(r['front_book_ms']) == m.ws + 4600 and int(r['n_behind']) == 1
          and list(Fr['role']) == ['front', 'behind'] and list(Fr['addr']) == [A2L, OTHER])
    return ok, dict(row=_brief(r), fronts=Fr[['role', 'addr']].values.tolist() if len(Fr) else [])


def chk_F3b(FX):
    """amendment 1 kept: one maker's own records (the book-timed front and an untimed record that could precede it) -> front_identified."""
    m = FX['F3b']
    P = m.part()
    r = lrow(P, 'bid', 45, 2000, m)
    Fr = P['fronts']
    ok = (r is not None and r['status'] == 'front_identified' and r['reason'] == 'ok' and r['holder'] == A2L and r['holder_group'] == 'rank2'
          and int(r['n_overlap']) == 2 and bool(r['overlap_one_address']) and r['amb_addresses'] == '' and int(r['n_behind']) == 1
          and int(r['n_unattributed']) == 1 and list(Fr['role']) == ['front', 'behind'] and list(Fr['addr']) == [A2L, A2L]
          and [bool(x) for x in Fr['in_overlap']] == [True, False])
    return ok, dict(row=_brief(r), fronts=Fr[['role', 'addr', 'in_overlap']].values.tolist() if len(Fr) else [])


def chk_F3s(FX):
    """amendment 2: one record sweeping two makers' orders -> ambiguous_order with both addresses, in the upper bound only (amendment 1)."""
    m = FX['F3s']
    P = m.part()
    r = lrow(P, 'bid', 45, 2000, m)
    SH = QF.shares_table(P['levels'], P['fronts'], {(m.day, m.mid): P['null']}, [('pooled', 'all', -1)])
    b = {x['group']: x for x in SH.to_dict('records')}
    ok = (r is not None and r['status'] == 'ambiguous_order' and r['reason'] == 'sweep_several_makers' and r['holder'] == ''
          and r['amb_addresses'] == '|'.join(sorted([A2L, OTHER])) and int(r['n_overlap']) == 1 and not bool(r['overlap_one_address'])
          and int(r['amb_gmask']) == (QF.GBIT['rank2'] | QF.GBIT['other']) and len(P['fronts']) == 0 and set(b) == set(QF.SCOPE_GROUPS)
          and all(b[g]['bound_denominator'] == 1 and b[g]['front_share_lower'] == 0.0 for g in QF.SCOPE_GROUPS)
          and all(b[g]['front_share_upper'] == v for g, v in (('rank1', 0.), ('rank2', 1.), ('rank3', 0.), ('other', 1.), ('top3', 1.))))
    return ok, dict(row=_brief(r), bounds={g: (b[g]['front_share_lower'], b[g]['front_share_upper'], b[g]['bound_denominator']) for g in b})


def chk_F3c(FX):
    """amendment 2: a record with no visible decrease belongs to the new life (settlement_only) only if its window opens after upd(f-1)."""
    m = FX['F3c']
    P = m.part()
    a, b = lrow(P, 'bid', 45, 2000, m), lrow(P, 'bid', 47, 2000, m)
    cn = P['counts']
    ok = (a is not None and b is not None and a['status'] == 'front_identified' and a['holder'] == A1L and a['front_timing'] == 'settlement'
          and int(a['front_book_ms']) == -1 and b['status'] == 'unknown' and b['reason'] == 'untimed_record_only' and b['holder'] == ''
          and cn['rec_settlement_only'] == 1 and cn['rec_book_unmatched'] == 1 and cn['rec_book_timed'] == 0)
    return ok, dict(a=_brief(a), b=_brief(b), rec={s_: cn[f'rec_{s_}'] for s_ in QF.REC_ST})


def chk_F7(FX):
    """amendment 2: an old-life record that settles after the new creation but is book-timed before f - 1 is excluded; the new life's first
    record (rank 3) holds the front."""
    m = FX['F7']
    P = m.part()
    r = lrow(P, 'bid', 45, 3600, m)
    ok = (r is not None and r['status'] == 'front_identified' and r['holder'] == A3L and r['holder_group'] == 'rank3' and int(r['n_old_life']) == 1
          and r['front_timing'] == 'book' and int(r['front_book_ms']) == m.ws + 6800 and P['counts']['levels_old_life_excluded'] == 1)
    return ok, dict(row=_brief(r), old=P['counts']['levels_old_life_excluded'])


def chk_F4(FX):
    m = FX['F4']
    P = m.part()
    L = P['levels']
    a, b = lrow(P, 'bid', 47, 2000, m), lrow(P, 'bid', 48, 25000, m)
    nl = {(m.day, m.mid): P['null']}
    SH = QF.shares_table(L, P['fronts'], nl, [('pooled', 'all', -1)])
    r1 = SH[SH['group'] == 'rank1'].iloc[0].to_dict() if len(SH) else {}
    MC = QF.mechanism_table(L, pd.DataFrame({c: [] for c in QF.JOIN_COLS}), [('pooled', 'all')])
    nf = MC[(MC['table'] == 'fresh_levels') & (MC['status'] == 'no_fill')] if len(MC) else MC
    ok = (a is not None and b is not None and a['status'] == 'no_fill' and a['reason'] == 'returned_to_zero' and b['status'] == 'no_fill'
          and b['reason'] == 'window_end_without_fill' and r1.get('n_fresh') == 2 and r1.get('n_no_fill') == 2 and r1.get('n_front_identified') == 0
          and r1.get('front_share_of_fresh') == 0.0 and len(nf) == len(QF.MECHS) and set(nf['denom'].tolist()) == {2} and int(nf['n'].sum()) == 2)
    return ok, dict(a=_brief(a), b=_brief(b), shares={k: r1.get(k) for k in ('n_fresh', 'n_no_fill', 'front_share_of_fresh')},
                    no_fill_mix=nf[['category', 'n', 'denom']].values.tolist() if len(nf) else [])


def chk_F5(FX):
    exp = (('F5a', 'bid', 45, 4000, 'improve', 800., NAN, NAN, dict(is_improve=True, is_re_establish=False)),
           ('F5b', 'bid', 41, 4000, 'deep', NAN, NAN, NAN, dict(is_deep=True)),
           ('F5c', 'ask', 60, 4000, 'at_empty_touch', NAN, NAN, NAN, dict(is_at_empty_touch=True)),
           ('F5d', 'bid', 44, 4000, 're_establish', 1000., 800., 1200., dict(is_re_establish=True, is_improve=True)),
           ('F5e', 'bid', 44, 4000, 'improve', 800., NAN, NAN, dict(is_re_establish=False, is_improve=True)),     # best 44 -> 43 at +3.0 s
           ('F5f', 'bid', 44, 9200, 'improve', 6000., NAN, NAN, dict(is_re_establish=False, is_improve=True)),
           ('F5g', 'bid', 46, 4000, 'other', NAN, NAN, NAN, dict(touch_unknown=True)))
    res, ok = [], True
    for nm, sd, tk, rt, mech, sp, lo, hi, fl in exp:
        m = FX[nm]
        r = lrow(m.part(), sd, tk, rt, m)
        good = (r is not None and r['unit'] == 'E1' and r['mechanism'] == mech and all(bool(r[k]) == v for k, v in fl.items())
                and all((np.isnan(x) and np.isnan(r[k])) or x == r[k] for k, x in (('speed_ms', sp), ('speed_lo_ms', lo), ('speed_hi_ms', hi))))
        ok &= bool(good)
        res.append((nm, bool(good), _brief(r)))
    return ok, [x for x in res if not x[1]] or 'all'


def chk_F6(FX):
    """causality: F6a = F1 with every frame after the front fill's reflection frame (and a later fill) perturbed -> the same level row;
    positive control: F2 = F1 with a cancel before the fill -> a different status."""
    a, b, c = FX['F1'], FX['F6a'], FX['F2']
    ra, rb, rc = lrow(a.part(), 'bid', 45, 2000, a), lrow(b.part(), 'bid', 45, 2000, b), lrow(c.part(), 'bid', 45, 2000, c)
    keys = ('unit', 'status', 'reason', 'holder', 'holder_group', 'mechanism', 'n_unexplained')
    same = (ra is not None and rb is not None and all(ra[k] == rb[k] for k in keys) and rb['status'] == 'front_identified'
            and int(ra['front_hi_ms']) - a.ws == int(rb['front_hi_ms']) - b.ws == 5000)
    moved = ra is not None and rc is not None and rc['status'] != ra['status']
    return bool(same and moved), dict(F1=_brief(ra), F6a=_brief(rb), control=_brief(rc))


SUITE = (('F1', chk_F1), ('F2', chk_F2), ('F3', chk_F3), ('F3b', chk_F3b), ('F3s', chk_F3s), ('F3c', chk_F3c), ('F4', chk_F4), ('F5', chk_F5),
         ('F6', chk_F6), ('F7', chk_F7))
SUITE_NAMES = dict(F1='(1) a planted level creator filled first -> front_identified with the right holder (rank 1) and its behind fill',
                   F2='(2) a decrease explained by no record before the first record (a cancel ahead) -> unexplained_decrease, never identified',
                   F3='(3) a two-record race with overlapping settlement windows resolved by the book intervals -> the earlier record (amendment 2)',
                   F3b="(3b) one maker's own records (front + an untimed record that could precede it) -> front_identified (amendment 1)",
                   F3s='(3s) a single-record sweep of two makers -> ambiguous_order with both addresses, upper bound only (amendments 1-2)',
                   F3c='(3c) no visible decrease: settlement_only when the window opens after upd(f-1), else book_unmatched -> unknown (amendment 2)',
                   F7='(7) an old-life record settling after the new creation, book-timed before f-1, is excluded; the new life holds (amendment 2)',
                   F4='(4) a cancelled level and a level never filled -> no_fill in levels, shares (n_fresh, n_no_fill) and the fresh-level mix',
                   F5='(5) mechanisms improve (800 ms) / deep / at_empty_touch / re_establish (1,000 ms, 800 / 1,200) / not re_establish x2 '
                      '(improve, 800 / 6,000 ms) / other',
                   F6='(6) causality: frames after the front fill change nothing; a cancel before it changes the status (positive control)')


def run_suite(FX):
    out = {}
    for nm, fn in SUITE:
        try:
            ok, info = fn(FX)
            out[nm] = (bool(ok), False, info)
        except Exception as e:  # noqa: BLE001
            out[nm] = (False, True, f'{type(e).__name__}: {e} | {traceback.format_exc()[-800:]}')
    return out


def test_mutants(FX):
    """The unmutated suite is the main section's checks (each must pass there); here every mutant must fail a named check without raising."""
    for m in QF.MUTANTS:
        os.environ['MT_QF_MUT'] = m
        try:
            res = run_suite(FX)
        finally:
            os.environ.pop('MT_QF_MUT', None)
        failed = [k for k, (ok, _, _) in res.items() if not ok]
        caught = [k for k in EXPECT_CATCH[m] if k in failed and not res[k][1]]
        check(f'(7) mutant {m}: caught by {"/".join(EXPECT_CATCH[m])} evaluating False without raising (failed: {failed})', bool(caught),
              {k: res[k][2] for k in failed})
    check('(7) mutants: MT_QF_MUT unset after the suite', 'MT_QF_MUT' not in os.environ)


def test_statuses(FX):
    exp = (('S1', 'bid', 45, 2000, 'E1', 'gap', 'gap_before_fill'), ('S2', 'bid', 45, 2600, 'status_only', 'gap', 'creation_across_gap'),
           ('S2', 'bid', 47, 5200, 'status_only', 'unknown', 'prev_cell_unknown'), ('S3', 'bid', 45, 2000, 'E1', 'unknown', 'unknown_cell_before_fill'),
           ('S3', 'bid', 45, 3200, 'status_only', 'unknown', 'prev_cell_unknown'),
           ('S4', 'bid', 45, 2000, 'E1', 'front_identified', 'ok'),                       # amendment 2: book-timed +2.8 s, after the creation
           ('S5', 'bid', 45, 2000, 'E1', 'unknown', 'untimed_record_only'),               # an unbounded (receipt-hi) record is never timed
           ('S5', 'bid', 47, 2000, 'E1', 'ambiguous_order', 'untimed_record_could_be_first'),
           ('S6', 'bid', 45, 2000, 'E1', 'front_identified', 'ok'),
           ('S7', 'bid', 45, 2000, 'E1', 'front_identified', 'ok'),                       # book-timed at the zero frame: the first life's fill
           ('S7', 'bid', 45, 5200, 'E1', 'no_fill', 'returned_to_zero'),                  # its only member is old life (excluded)
           ('S8', 'bid', 45, 2000, 'E1', 'front_identified', 'ok'),
           ('S10', 'bid', 45, 2000, 'E1', 'front_identified', 'ok'), ('S10', 'bid', 45, 8400, 'E1', 'unknown', 'untimed_record_only'),
           ('S11', 'bid', 45, 2000, 'E1', 'ambiguous_order', 'untimed_record_could_be_first'))
    parts = {}
    for nm, sd, tk, rt, unit, st, rs in exp:
        if nm not in parts:
            parts[nm] = FX[nm].part()
        r = lrow(parts[nm], sd, tk, rt, FX[nm])
        check(f'status {nm}: {sd} {tk} created +{rt} ms -> {unit} {st} / {rs}', r is not None and r['unit'] == unit and r['status'] == st
              and r['reason'] == rs, _brief(r))
    r6 = lrow(parts['S6'], 'bid', 45, 2000, FX['S6'])
    check('S6: the front that emptied its level is attributed (straddles_end) to rank 1, no behind fill',
          r6 is not None and bool(r6['straddles_end']) and r6['holder'] == A1L and int(r6['n_behind']) == 0 and r6['end_type'] == 'zero', _brief(r6))
    r10 = lrow(parts['S10'], 'bid', 45, 2000, FX['S10'])
    check('S10: a member whose hi falls after the re-creation is unattributable (n_behind 0, n_unattributed 1)',
          r10 is not None and int(r10['n_behind']) == 0 and int(r10['n_unattributed']) == 1 and int(r10['n_members']) == 2, _brief(r10))
    r5 = lrow(parts['S5'], 'bid', 47, 2000, FX['S5'])
    check('S5: an untimed (unbounded) record of another maker could precede the book-timed front -> ambiguous with both addresses',
          r5 is not None and int(r5['n_unbounded']) == 1 and int(r5['n_overlap']) == 2 and not bool(r5['overlap_one_address'])
          and r5['amb_addresses'] == '|'.join(sorted([A2L, OTHER])), _brief(r5))
    r7 = lrow(parts['S7'], 'bid', 45, 5200, FX['S7'])
    check('S7: the record that emptied the first life (settling after the re-creation) is old life for the second level (n_old_life 1)',
          r7 is not None and int(r7['n_old_life']) == 1 and parts['S7']['counts']['levels_old_life_excluded'] == 1, _brief(r7))
    r11 = lrow(parts['S11'], 'bid', 45, 2000, FX['S11'])
    check('S11: a record with two equal candidate decreases is book_ambiguous; it could precede the timed front -> ambiguous with both makers',
          r11 is not None and r11['amb_addresses'] == '|'.join(sorted([A1L, OTHER])) and parts['S11']['counts']['rec_book_ambiguous'] == 1
          and parts['S11']['counts']['rec_book_timed'] == 1, dict(row=_brief(r11), rec={s_: parts['S11']['counts'][f'rec_{s_}'] for s_ in QF.REC_ST}))
    allL = pd.concat([FX[k].part()['levels'] for k in FX], ignore_index=True)
    E1 = allL[allL['unit'] == 'E1']
    so = allL[allL['unit'] == 'status_only']
    check('statuses: every E1 status is one of the six, every status_only row is gap / unknown, every row has a reason',
          set(E1['status']) <= set(QF.STATUSES) and set(so['status']) <= {'gap', 'unknown'} and bool((allL['reason'].astype(str) != '').all())
          and set(QF.STATUSES) <= set(E1['status']), sorted(Counter(E1['status']).items()))
    fi = E1[E1['status'] == 'front_identified']
    check('statuses: an identified level always has a holder and group; the others never', bool((fi['holder'] != '').all() and (fi['holder_group'] != '').all())
          and bool((E1.loc[E1['status'] != 'front_identified', 'holder'] == '').all()))


def test_cells(FX):
    P = FX['F1'].part()
    ck, cn = P['checks'], P['counts']
    check('fill-to-cell: F1 has 3 grid fills (2 top-3), the top-3 ask 80 fill is never shown (<= hi and in its window), the sub-cent leg is off grid',
          ck['grid_fills'] == 3 and ck['top3_grid_fills'] == 2 and ck['top3_not_shown_le_hi'] == 1 and ck['top3_not_shown_in_window'] == 1
          and ck['grid_fills_not_shown_le_hi'] == 1 and ck['top3_offgrid_fills'] == 1, ck)
    check('fill join: 4 legs, 1 off grid, 4 fills_valued rows, the lifecycle cell = the fills_valued (y_side, p) cell, addresses agree',
          cn['legs'] == 4 and cn['legs_offgrid'] == 1 and cn['legs_with_fv_row'] == 4 and cn['cell_fv_vs_life_mismatch'] == 0
          and cn['addr_fv_vs_orders_mismatch'] == 0 and cn['q_fv_vs_life_mismatch'] == 0, dict(cn))
    M, fvm = FX['F5b'].build()
    fl, cn2 = QF.market_fills(M, fvm)
    check("fill join: a leg without a fills_valued row takes the lifecycle orders' signer and has no markout (reason no_fv_row)",
          cn2['legs_addr_from_orders'] == 1 and list(fl['addr']) == [OTHER2] and list(fl['reason']) == ['no_fv_row'] and not bool(fl['cohort'][0]), dict(cn2))


def test_null():
    lab = np.r_[np.zeros(40, np.int8), np.full(160, 3, np.int8)]
    front = np.r_[np.ones(40, bool), np.zeros(160, bool)]
    N1 = QF.null_market(DAY, 1, lab, front)
    p1 = QF.p_hi_of(N1['obs'][0], N1['null'][0])
    check('null: a planted front winner (40 / 40 fronts, 40 / 200 fills) -> p_hi = 1/501', abs(p1 - 1 / 501) < 1e-12 and int(N1['obs'][0]) == 40,
          dict(p=p1, null_mean=float(N1['null'][0].mean())))
    lab2 = np.r_[np.zeros(8, np.int8), np.full(32, 3, np.int8), np.zeros(32, np.int8), np.full(128, 3, np.int8)]
    N2 = QF.null_market(DAY, 2, lab2, front)
    p2 = QF.p_hi_of(N2['obs'][0], N2['null'][0])
    check('null: exchangeable labels (front share = fill share = 0.2) -> p_hi > 0.2', p2 > 0.2 and int(N2['obs'][0]) == 8, p2)
    N1b, N1c = QF.null_market(DAY, 1, lab, front), QF.null_market(DAY, 3, lab, front)
    check('null: deterministic per (seed, day, mid), another market draws differently, 500 replicates',
          np.array_equal(N1['null'], N1b['null']) and not np.array_equal(N1['null'], N1c['null']) and N1['null'].shape == (4, QF.N_PERM))
    check('null: every replicate keeps the front count and the label multiset', bool((N1['null'].sum(axis=0) == 40).all())
          and list(N1['fills']) == [40, 0, 0, 160] and int(N1['null'][1].sum()) == 0)
    Z = QF.null_market(DAY, 4, np.zeros(0, np.int8), np.zeros(0, bool))
    check('null: a market without identified fronts has k = 0 and no draws', Z['k'] == 0 and int(Z['null'].sum()) == 0)


def test_cohort():
    import chain_post_fill as CPF
    m = MK(DAY, 990001, 2)
    spec = (('ok', False, 'ok', ('ok', 'ok', 'ok')), ('record_leg_excluded', False, 'ok', ('ok', 'ok', 'ok')), ('ok', True, 'ok', ('ok', 'ok', 'ok')),
            ('ok', False, 'no_resolution', ('ok', 'ok', 'ok')), ('ok', False, 'tie', ('ok', 'ok', 'ok')), ('ok', False, 'ok', ('stale', 'ok', 'ok')),
            ('ok', False, 'ok', ('ok', 'beyond_end', 'ok')), ('ok', False, 'ok', ('ok', 'ok', 'beyond_end')), ('ok', False, 'ok', ('ok', 'ok', 'ok')))
    for i, (fs, sf, sr, st) in enumerate(spec):
        m.fill(A1, 'bid', 45, 1 + i, 5000 + 100 * i, fv_status=fs, self_trade=sf, st_res=sr, st=st, v=(0.1 * i, 0.2 * i - 0.5, 0.3 * i))
    M, fvm = m.build()
    fl, _ = QF.market_fills(M, fvm)
    ref = CPF.pnl_rows(fvm.assign(net_res=0.5, fee_fill=0.01))
    coh = fl['cohort']
    ok = (abs(float(fl['q_c'].sum()) - ref['pnl_shares']) < 1e-9 and int(coh.sum()) == ref['pnl_rows']
          and all(abs(float(np.nansum(fl[f'v_{h}'])) - ref[f'v_{h}']) < 1e-9 for h in QF.HORIZONS)
          and dict(Counter(fl['reason'][~coh].tolist())) == ref['pnl_excluded'])
    check("markouts: the per-fill cohort reproduces chain_post_fill.pnl_rows (rows, shares, v_1s / v_10s / v_60s, exclusions by first reason)", ok,
          dict(ref={k: ref[k] for k in ('pnl_rows', 'pnl_shares', 'v_1s', 'v_10s', 'v_60s', 'pnl_excluded')}, ours=dict(Counter(fl['reason'].tolist()))))
    X = pd.DataFrame(dict(cohort=coh, q=fl['q'], q_c=fl['q_c'], excl_reason=fl['reason'], **{f'v_{h}': fl[f'v_{h}'] for h in QF.HORIZONS}))
    ec = QF._econ(X)
    check('markouts: c/sh = 100 x the cohort sum / the cohort shares', abs(ec['v_10s_c_sh'] - 100. * ref['v_10s'] / ref['pnl_shares']) < 1e-9
          and ec['cohort_rows'] == ref['pnl_rows'], ec)


def test_readings():
    out = {}
    for kind in ('planted', 'exch'):
        mks = world(kind)
        parts = [m.part() for m in mks]
        markets = [(m.day, m.mid) for m in mks]
        tables, claims, dg = QF.analyse('selftest', (DAY, DAY2), markets, parts, Counter())
        out[kind] = (mks, parts, markets, tables, claims)
        V = {(r['scope'], r['reading']): r for r in claims['readings']}
        check(f'readings ({kind}): 4 readings x 2 scopes, every verdict computed from the declared set', len(V) == 8 and all(
            r['verdict'] in VERDICTS for r in V.values()), sorted((k, r['verdict']) for k, r in V.items()))
        pop = claims['population']
        check(f'readings ({kind}): 11 identified levels per market (9 improve + 2 re_establish), 6 supported markets',
              pop['fresh_by_status']['front_identified'] == 66 and pop['supported_markets'] == 6, pop)
        if kind == 'planted':
            want = dict(wins_front_more_than_fills='wins_front_more_than_fills', main_mechanism='improve', faster_than_others='faster_than_others',
                        front_pays='front_pays')
            check('readings (planted): rank 1 wins the front more than its fills (p_hi = 1/501), main mechanism improve, faster re-establish, front pays',
                  all(V[(s, k)]['verdict'] == v for s in QF.READ_SCOPES for k, v in want.items())
                  and abs(V[('rank1', 'wins_front_more_than_fills')]['values']['p_hi'] - 1 / 501) < 1e-12,
                  {f'{s}:{k}': (V[(s, k)]['verdict'], V[(s, k)]['predicates']) for s in QF.READ_SCOPES for k in want})
            sh = tables['shares.csv']
            r1 = sh[(sh['scope'] == 'pooled') & (sh['group'] == 'rank1')].iloc[0]
            check('readings (planted): front share bounds 60/66 = 60/66 (no ambiguous level), fill share 60/174 (hand-computed)',
                  abs(r1['front_share_lower'] - 60 / 66) < 1e-12 and r1['front_share_upper'] == r1['front_share_lower'] and r1['bound_denominator'] == 66
                  and abs(r1['fill_share'] - 60 / 174) < 1e-12, dict(r1))
        else:
            vv = {k: V[('rank1', k)] for k in ('wins_front_more_than_fills', 'faster_than_others', 'front_pays')}
            check('readings (exchangeable): rank 1 reads none of wins / faster / front_pays (unidentified) ...',
                  all(v['verdict'] == 'unidentified' for v in vv.values()), {k: (v['verdict'], v['values'].get('p_hi')) for k, v in vv.items()})
            check('... while each was testable (>= 5 supported / evaluable markets: the negative control could fire)',
                  vv['wins_front_more_than_fills']['predicates']['supported_markets_ge_5'] and vv['faster_than_others']['predicates']['evaluable_markets_ge_5']
                  and vv['front_pays']['predicates']['evaluable_markets_ge_5'], {k: v['predicates'] for k, v in vv.items()})
    sp = out['planted'][3]['speed.csv']
    r = sp[(sp['scope'] == 'pooled') & (sp['mechanism'] == 're_establish') & (sp['group'] == 'rank1')].iloc[0]
    check('speed (planted): rank 1 re_establish median 400 ms (6 levels), others 3,000 ms', r['speed_ms_p50'] == 400. and r['speed_ms_n'] == 6
          and sp[(sp['scope'] == 'pooled') & (sp['mechanism'] == 're_establish') & (sp['group'] == 'other')].iloc[0]['speed_ms_p50'] == 3000., dict(r))
    ec = out['planted'][3]['economics.csv']
    g = ec[(ec['scope'] == 'pooled') & (ec['by'] == 'holder') & (ec['group'] == 'rank1') & (ec['mechanism'] == 'all')]
    fe, be = g[g['role'] == 'front'].iloc[0], g[g['role'] == 'behind'].iloc[0]
    check('economics (planted): rank 1 front fills +30 c/sh (60 fills) vs behind -20 c/sh (108 fills)', abs(fe['v_10s_c_sh'] - 30.) < 1e-9
          and abs(be['v_10s_c_sh'] + 20.) < 1e-9 and fe['n_fills'] == 60 and be['n_fills'] == 108, dict(front=dict(fe), behind=dict(be)))
    return out


def test_refusals(tmp):
    d = tmp / 'exists'
    d.mkdir()
    expect('refuse: write-once on an existing output directory (overwrite)', lambda: QF.write_out(d, {}, {}, lambda o: {}), 'write-once')

    def boom(o):
        raise RuntimeError('boom in the receipt writer')
    try:
        QF.write_out(tmp / 'fails', {'a.csv': pd.DataFrame({'x': [1]})}, {}, boom)
    except RuntimeError:
        pass
    check('write-once: a failed write leaves no output and no partial directory', not (tmp / 'fails').exists()
          and not any(p.name.startswith('.fails.partial') for p in tmp.iterdir()))
    for m, needle in (('first_fill_by_settle', 'MT_QF_MUT'), ('drop_no_fill', 'MT_QF_MUT'), ('bogus', 'not a declared mutant')):
        os.environ['MT_QF_MUT'] = m
        try:
            expect(f'refuse: production with MT_QF_MUT={m} before any input is read', lambda: QF.execute('pilot', (QF.PILOT_DAY,), tmp / 'never'), needle)
        finally:
            os.environ.pop('MT_QF_MUT', None)
    check('refuse: nothing written by a refused production call', not (tmp / 'never').exists())
    expect('refuse: a pilot day outside the measurement days (before any path is built)', lambda: QF.execute('pilot', (C.ID_DAY,), tmp / 'never'),
           'measurement days')
    g = QF._git([f for f in QF.TOOL_FILES if Path(f).exists()])
    needle = 'freeze' if not g.get('unmodified') else 'no recorded pilot projection'
    expect(f'refuse: the full run before any input ({needle})', lambda: QF.execute('run', C.MEASURE_DAYS, tmp / 'run_out', pilot_receipt=tmp / 'absent.json'),
           needle)
    expect('refuse: the full run without a recorded pilot projection', lambda: QF.preflight_projection(tmp / 'absent.json', {}), 'no recorded pilot projection')
    p121 = tmp / 'p121.json'
    p121.write_text(json.dumps(dict(kind='pilot', projection=dict(full_run_cpu_s=121.0))))
    expect('refuse: a recorded pilot projection > 120 CPU s', lambda: QF.preflight_projection(p121, {}), '> 120 s (refused before launch)')
    p60 = tmp / 'p60.json'
    p60.write_text(json.dumps(dict(kind='pilot', projection=dict(full_run_cpu_s=60.0))))
    h = {}
    r = QF.preflight_projection(p60, h)
    check('preflight: a 60 s projection passes and the pilot receipt is hashed as an input', r['full_run_cpu_s'] == 60.0 and len(h) == 1)


# ------------------------------------------------------------------ inputs and the pilot path end to end ------------------------------------------------------------------
def fv_table(rows):
    df = typed(rows, FVT)
    cols = {}
    for c, t in FVT.items():
        if t == 'str':
            cols[c] = pa.array([str(x) for x in df[c].tolist()], type=pa.string()).dictionary_encode()
        elif t == 'bool':
            cols[c] = pa.array(df[c].to_numpy(bool), type=pa.bool_())
        elif t == 'int64':
            cols[c] = pa.array(df[c].to_numpy(np.int64), type=pa.int64())
        else:
            cols[c] = pa.array(df[c].to_numpy(np.float64), type=pa.float64())
    return pa.table(cols)


def write_fv(path, groups):
    tabs = [fv_table(g) for g in groups]
    w = pq.ParquetWriter(path, tabs[0].schema)
    for t in tabs:
        w.write_table(t)
    w.close()


def pf_orders(mks, over):
    rows = []
    for m in mks:
        for L in m.legs:
            if L['addr'].lower() not in QF.TOP3_L:
                continue
            o = dict(day=m.day, mid=m.mid, rank=QF.TOP3_L.index(L['addr'].lower()) + 1, addr=L['addr'], mk_hash=L['mk_hash'], y_side=L['side'],
                     tick=L['tick'], n_fills=1, cum_shares=L['q'], post_lo_union_ms=L['hi'] - 5000, post_hi_ms=L['hi'], window_status='bounded',
                     queue_class='unidentified')
            o.update(over.get(L['mk_hash'], {}))
            rows.append(o)
    return pd.DataFrame(rows, columns=list(PFC))


def write_pf(pf, D, fv_sha, rows=None):
    pf.mkdir(parents=True, exist_ok=True)
    D.to_csv(pf / 'orders.csv', index=False)
    rec = dict(outputs={'orders.csv': dict(sha256=QF._sha(pf / 'orders.csv'))}, inputs={'econ/fills_valued.parquet': dict(sha256=fv_sha)},
               rows=dict(orders=int(len(D) if rows is None else rows)))
    (pf / 'receipt.json').write_text(json.dumps(rec))


def test_e2e(tmp, FX, W):
    planted = W['planted'][0]
    day_mks = [m for m in FX.values() if m.day == DAY] + [m for m in planted if m.day == DAY]
    root = tmp / 'root'
    (root / 'econ').mkdir(parents=True)
    (root / 'econ' / 'identities.json').write_text(json.dumps(dict(ex_ante=dict(day=C.ID_DAY, top10_shares=[dict(addr=a, rank=i + 1)
                                                                                                            for i, a in enumerate(QF.TOP3)]))))
    rg0 = [dict(day='20261023', mid=800001, sid='X1', mk_idx=0, role='maker', addr=OTHER, hash='hx', y_side='bid', s=1, q=1., p=.5, self_trade=False,
                settle_ms=1, status='ok', st_res='ok', st_1s='ok', v_1s=0., st_10s='ok', v_10s=0., st_60s='ok', v_60s=0.)]
    rg1 = [r for m in day_mks for r in m.fv_rows(taker=True)]
    rg2 = [r for m in planted if m.day == DAY2 for r in m.fv_rows(taker=True)]
    fvp = root / 'econ' / 'fills_valued.parquet'
    write_fv(fvp, [rg0, rg1, rg2])
    fsha = QF._sha(fvp)
    f3a2 = FX['F3'].legs[0]['mk_hash']
    s4a1 = FX['S4'].legs[0]['mk_hash']
    D = pf_orders(day_mks, {f3a2: dict(post_lo_union_ms=FX['F3'].ws + 3000, queue_class='behind_at_first_fill'),
                            s4a1: dict(post_lo_union_ms=-1, window_status='window_unbounded')})
    pf = tmp / 'pf'
    write_pf(pf, D, fsha)
    built = {(m.day, m.mid): m.build()[0] for m in day_mks}
    book = {DAY: sorted(m.mid for m in day_mks)}
    markets = [(DAY, mid) for mid in book[DAY]]
    # ---- the readers on these files ----
    cnt = Counter()
    FVr = QF.read_fv_makers(fvp, markets, cnt)
    n_mk = sum(1 for m in day_mks for L in m.legs if L['fv'])
    check('reader: fills_valued maker rows of the run markets only (taker rows and other markets dropped; a row group pruned; dictionary columns)',
          len(FVr) == n_mk and set(FVr['role']) == {'maker'} and cnt['fv_groups_pruned'] >= 1
          and pa.types.is_dictionary(pq.ParquetFile(fvp).schema_arrow.field('role').type) and set(FVr['addr_l']) <= {a.lower() for a in FVr['addr']},
          dict(rows=len(FVr), expected=n_mk, cnt=dict(cnt)))
    bad = tmp / 'fv_dup.parquet'
    write_fv(bad, [rg1[:1] + rg1[:1]])
    expect('reader: a maker leg twice in fills_valued -> refused', lambda: QF.read_fv_makers(bad, markets, Counter()), 'twice')
    bad2 = tmp / 'fv_day.parquet'
    write_fv(bad2, [[dict(rg1[0], day=DAY2)]])
    expect('reader: a BOOK market row under another day -> refused', lambda: QF.read_fv_makers(bad2, markets, Counter()), 'another day')
    PFr, info = QF.read_post_fill(pf, markets, {}, fsha)
    check("reader: chain_post_fill orders.csv bound to its receipt; the run's top-3 orders", len(PFr) == len(D) and info['rows'] == len(D), info)
    for nm, (DD, sha, rows), needle in (('a tampered orders.csv', (D.assign(n_fills=2), fsha, None), "differs from its receipt"),
                                         ('another fills_valued binding', (D, 'f' * 64, None), 'differs from the binding'),
                                         ('a row count unlike its receipt', (D, fsha, len(D) + 1), 'rows differ'),
                                         ('an address unlike its rank', (D.assign(addr=OTHER), fsha, None), "address differs")):
        pfx = tmp / f'pf_{abs(hash(nm)) % 10 ** 8}'
        write_pf(pfx, D if nm == 'a tampered orders.csv' else DD, sha, rows)
        if nm == 'a tampered orders.csv':
            DD.to_csv(pfx / 'orders.csv', index=False)
        expect(f'reader: chain_post_fill with {nm} -> refused', lambda p=pfx: QF.read_post_fill(p, markets, {}, fsha), needle)
    # ---- the pilot path through execute() (loader injected; everything else production) ----
    out = tmp / 'e2e_pilot'
    rec = QF.execute('pilot', (DAY,), out, inp=root, loader=lambda d, mid, r, hc: built[(d, mid)], book=book, pf_dir=pf, fv_sha=fsha, log=lambda s: None)
    names = sorted(p.name for p in out.iterdir())
    check('e2e: execute (pilot path, injected loader) writes exactly the nine contract outputs', names == sorted(QF.OUT_FILES), names)
    Rj = json.loads((out / 'receipt.json').read_text())
    pj = Rj.get('projection') or {}
    check('e2e: receipt.json hashes every other output, the sources and inputs; a finite full-run projection; no mutant',
          all(Rj['outputs'][f]['sha256'] == QF._sha(out / f) for f in QF.OUT_FILES if f != 'receipt.json') and Rj['mutant'] == ''
          and len(Rj['sources']) >= 8 and Rj['inputs'].get('econ/fills_valued.parquet') == fsha and isinstance(pj.get('full_run_cpu_s'), float)
          and rec['kind'] == 'pilot', dict(proj=pj, n_src=len(Rj['sources']), inputs=sorted(Rj['inputs'])[:6]))
    exp_parts = [m.part() for m in day_mks]
    LVc = pd.read_csv(out / 'levels.csv', dtype={'day': str})
    n_lv = sum(len(p['levels']) for p in exp_parts)
    n_nf = sum(int(((p['levels']['unit'] == 'E1') & (p['levels']['status'] == 'no_fill')).sum()) for p in exp_parts)
    SH = pd.read_csv(out / 'shares.csv', dtype={'day': str})
    pr1 = SH[(SH['scope'] == 'pooled') & (SH['group'] == 'rank1')].iloc[0]
    check('e2e: levels.csv = every market part; shares pooled n_fresh = the E1 rows, n_no_fill counted, statuses add up',
          len(LVc) == n_lv and int(pr1['n_no_fill']) == n_nf and int(pr1['n_fresh']) == int((LVc['unit'] == 'E1').sum())
          and int(sum(pr1[f'n_{s}'] for s in QF.STATUSES)) == int(pr1['n_fresh']), dict(levels=len(LVc), expected=n_lv, no_fill=n_nf))
    pool = SH[SH['scope'] == 'pooled']
    up = {r['group']: (float(r['front_share_lower']), float(r['front_share_upper'])) for r in pool.to_dict('records')}
    check('e2e (amendment 1): pooled bounds share one denominator (identified + ambiguous), lower <= upper, and the two-maker sets (F3, S5) '
          'lift rank 2 / other upper above lower',
          bool((pool['bound_denominator'] == int(pr1['n_front_identified']) + int(pr1['n_ambiguous_order'])).all())
          and all(lo_ <= hi_ + 1e-12 for lo_, hi_ in up.values()) and all(up[g][1] > up[g][0] for g in ('rank2', 'other'))
          and int(pool.loc[pool['group'] == 'rank2', 'n_ambiguous_with_group'].iloc[0]) == 2,
          pool[['group', 'n_front_group', 'n_ambiguous_with_group', 'bound_denominator', 'front_share_lower', 'front_share_upper']].values.tolist())
    MC = pd.read_csv(out / 'mechanism.csv', dtype={'day': str})
    J = MC[(MC['table'] == 'joined') & (MC['scope'] == 'pooled') & (MC['group'] == 'top3') & (MC['category'] == '_all')]
    jd = {r['status']: (int(r['n']), int(r['n_front_holder'])) for r in J.to_dict('records')}
    check('e2e: joined counts - the F3 rank-2 order joined an existing queue, the unbounded window is counted, the F1 front holder posted into '
          'an empty cell', jd.get('joined_existing_queue', (0, 0))[0] >= 1 and jd.get('window_unbounded', (0, 0))[0] == 1
          and jd.get('cell_empty_before_window', (0, 0))[1] >= 1, jd)
    CS = pd.read_csv(out / 'consistency.csv', dtype={'day': str})
    Cl = json.loads((out / 'claims.json').read_text())
    bz = Cl.get('beside_readings') or {}
    rt = bz.get('record_timing') or {}
    n_old_exp = sum(int((p['levels']['n_old_life'] > 0).sum()) for p in exp_parts)
    check('e2e (amendment 2): claims carry every record timing status (all four occur) and the old-life exclusion share beside the readings',
          set(rt) == set(QF.REC_ST) and all(rt[s_] >= 1 for s_ in QF.REC_ST) and bz.get('levels_old_life_excluded') == n_old_exp >= 2
          and abs(bz.get('old_life_excluded_share', -1) - n_old_exp / int((LVc['unit'] == 'E1').sum())) < 1e-12, bz)
    check('e2e: consistency.csv = markets x 5 groups; claims.json carries 4 readings x 2 scopes with computed verdicts and the fill-to-cell counts',
          len(CS) == 5 * len(markets) and len(Cl['readings']) == 8 and all(r['verdict'] in VERDICTS for r in Cl['readings'])
          and Cl['fill_to_cell'].get('cellcheck_top3_not_shown_le_hi', -1) >= 1 and Cl['descriptive_only'] is True and Cl['intervals'] is None,
          dict(cons=len(CS), fill_to_cell=Cl['fill_to_cell']))
    expect('e2e: a second write into the written directory is refused (before any input is read)',
           lambda: QF.execute('pilot', (DAY,), out, inp=root, loader=lambda d, mid, r, hc: built[(d, mid)], book=book, pf_dir=pf, fv_sha=fsha,
                              log=lambda s: None), 'write-once')
    # ---- run_core over two days (the full run's tail) ----
    mks, parts, mkts = W['planted'][0], W['planted'][1], W['planted'][2]
    out2 = tmp / 'e2e_two_days'
    rec2, claims2, dg2 = QF.run_core('selftest', (DAY, DAY2), mkts, parts, out2, {}, Counter(), {}, lambda nm: None, {}, 'asof', time.time())
    SH2 = pd.read_csv(out2 / 'shares.csv', dtype={'day': str})
    check('e2e: run_core over two days writes the nine outputs; shares carry pooled + 2 days + 6 markets per group; readings as in memory',
          sorted(p.name for p in out2.iterdir()) == sorted(QF.OUT_FILES) and len(SH2) == 5 * (1 + 2 + 6)
          and [r['verdict'] for r in claims2['readings']] == [r['verdict'] for r in W['planted'][4]['readings']], len(SH2))


def main():
    t0 = time.process_time()
    check('selftest environment: MT_RUN=v2 and no MT_QF_MUT', C.RUN == 'v2' and 'MT_QF_MUT' not in os.environ, C.RUN)
    tmp = Path(tempfile.mkdtemp(prefix='queue_front_selftest_'))
    try:
        FX = section('fixtures', fixtures)
        if FX:
            for nm, fn in SUITE:
                r = section(nm, fn, FX)
                if r:
                    check(SUITE_NAMES[nm], r[0], r[1])
            section('statuses', test_statuses, FX)
            section('fill_to_cell', test_cells, FX)
            section('mutants', test_mutants, FX)
        section('null', test_null)
        section('cohort', test_cohort)
        W = {}
        R = section('readings', test_readings)
        if R:
            W.update(R)
        section('refusals', test_refusals, tmp)
        if FX and W:
            section('e2e', test_e2e, tmp, FX, W)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop('MT_QF_MUT', None)
    fails = [r for r in RES if not r[1]]
    for name, ok, info in RES:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  | {info}'))
    print(json.dumps(dict(checks=len(RES), failed=len(fails), cpu_s=round(time.process_time() - t0, 3), process_cpu_s=round(time.process_time(), 3),
                          vmhwm_mib=round(QF.hwm(), 1))))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
