"""chain_take_selftest.py - synthetic proof of chain_take.py (P-2026-005 MAKER-TRACE logic-chain packet, census tool 7.3; 2026-11-23).
The REAL chain_context functions (btc_ret = pf_mt_decision.btc_feats, book / level_size on pf_mt_lifecycle._book, position) run on synthetic
markets built with chain_context.build_market / join_windows; the real order_estimator.link_hypotheses and chain_pull.order_table. Exact
expected values; every refusal's reason asserted. Sections: classes (hand-computed paths incl. an overshoot, a reversal, a pending-ambiguous
and a simultaneous case) and their agreement with order_estimator.inv_class; mechanisms; the window-instant set against
chain_context.over_window; a burst-reducing taker recovered against matched controls and a random taker with no association; the action
enumeration on a constructed ladder / two-sided / refill / reprice fixture; a look-ahead falsifier (perturbing data after a decision instant
leaves its features unchanged, before it changes them, and a planted leak is caught); refusals; every writer end to end (run + pilot).
Usage (run_light, MT_RUN=v2): chain_take_selftest.py"""
import json
import shutil
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
for _p in (str(RESEARCH), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pf_mt_common as C  # noqa: E402
import pf_mt_econ as E  # noqa: E402
import chain_context as CC  # noqa: E402
import order_estimator as OE  # noqa: E402
import chain_pull as CP  # noqa: E402
import chain_take as T  # noqa: E402

RES = []
NAN = float('nan')


def check(name, ok, info=''):
    RES.append((name, bool(ok), str(info)[:400]))


def expect(name, fn, needle):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        check(name, needle in str(e), f'{type(e).__name__}: {e}')
        return
    check(name, False, 'did not raise')


def section(name, fn, *a):
    try:
        fn(*a)
    except Exception as e:  # noqa: BLE001
        check(f'{name}: section crashed', False, f'{type(e).__name__}: {e} | {traceback.format_exc()[-900:]}')


# ------------------------------------------------------------------ synthetic world ------------------------------------------------------------------
A1, A2, A3 = T.TOP3
OTHER, TAKER_X = '0x' + 'ab' * 20, '0x' + 'cd' * 20
BOOKM = E.load_book(E.BOOK_MANIFEST)
DAY_A, DAY_B, DAY_C = '20261024', '20261025', '20261026'
MID_A, MID_D, MID_B, MID_C = BOOKM[DAY_A][0], BOOKM[DAY_A][1], BOOKM[DAY_B][0], BOOKM[DAY_C][0]
WS0 = 1_792_816_800_000
FV_READ = CC.FV_READ
FV_DT = {c: {'str': object, 'int64': np.int64, 'float64': np.float64, 'bool': bool}[E.FV[c]] for c in FV_READ}
LNK_EMPTY = pd.DataFrame({c: pd.Series(dtype=t) for c, t in (('mid', 'int64'), ('mk_signer', 'object'), ('y_side', 'object'), ('kind', 'object'),
                          ('prev_hash', 'object'), ('next_hash', 'object'), ('prev_ptick', 'int64'), ('next_ptick', 'int64'), ('prev_last_hi_ms', 'int64'),
                          ('next_first_hi_ms', 'int64'), ('gap_ms', 'int64'), ('x_latest', 'bool'), ('status', 'object'))})


def ws_of(day, mid):
    return WS0 + 3_600_000 * (sorted(m for d in BOOKM for m in BOOKM[d]).index(mid))


def bt_from_lm(G0, lm):
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


def make_bt(ws, seed, sigma_bp=0.15, jumps=()):
    """A 100 ms BTC grid over [ws - 400 s, ws + 360 s], every row usable (pf_mt_decision.load_btc's layout); jumps = ((abs ms, bp), ...)."""
    g_lo = ws - 400_000
    G0 = g_lo // 100
    n = 7_601
    inc = np.random.default_rng(seed).normal(0., sigma_bp * 1e-4, n)
    inc[0] = 0.
    for t_ms, d_bp in jumps:
        inc[t_ms // 100 - G0] += d_bp * 1e-4
    return bt_from_lm(G0, np.log(70_000.) + np.cumsum(inc))


def btc_dict(BT, day):
    return dict(BT=BT, code=np.where(BT['ok'], 0, 2).astype(np.int8), day=day, load_counts={}, inputs={}, cum=None)


def book_frames(mid, ws, shift_after=None, shift=0):
    t = np.arange(ws - 310_000, ws + 305_000, 200, dtype=np.int64)
    fid = np.arange(1, len(t) + 1, dtype=np.int64)
    bid = 45 + (t // 7_000) % 3 - 1
    ask = 55 + (t // 11_000) % 3 - 1
    if shift_after is not None:
        m = t > shift_after
        bid, ask = np.where(m, bid - shift, bid), np.where(m, ask - shift, ask)
    bsh, ash = 100. + fid % 7, 80. + fid % 5
    frames = pd.DataFrame(dict(mid=np.int64(mid), fid=fid, upd_ms=t, recv_ns=t * 1_000_000 + 7, conn='c0', order_count=np.int64(0), pend_bids='[]',
                               pend_asks='[]', los_id='', los_side='', los_price=np.nan, los_outcome='', status='ok'))
    levels = pd.DataFrame(dict(mid=np.int64(mid), fid=np.r_[fid, fid, fid], side=['bid'] * (2 * len(t)) + ['ask'] * len(t),
                               ptick=np.r_[bid, bid - 1, ask].astype(np.int64), shares=np.r_[bsh, np.full(len(t), 50.), ash], status='ok'))
    tob = pd.DataFrame(dict(mid=np.int64(mid), upd_ms=t, recv_ns=t * 1_000_000 + 7, conn='c0', n_bid=np.int64(2), n_ask=np.int64(1), best_bid=bid / 100.,
                            best_bid_sh=bsh, best_ask=ask / 100., best_ask_sh=ash, status='ok'))
    return frames, levels, tob


class Mk:
    """One synthetic market: fills_valued rows of the top 3, lifecycle fills (every maker leg), lifecycle orders, x_links."""

    def __init__(self, day, mid):
        self.day, self.mid, self.ws = day, int(mid), ws_of(day, mid)
        self.fv, self.life, self.orders, self.links, self.n = [], [], [], [], 0

    def _sid(self):
        self.n += 1
        return f'{self.mid}-{self.n:06d}'

    def _fv(self, **k):
        r = dict(day=self.day, mid=self.mid, mk_idx=0, px_rank=0, kind='regular', self_trade=False, t_status='lo', fee=0., fee_type='COLLATERAL',
                 fee_dy=0., status='ok', src_status='ok')
        r.update(k)
        r['t_ms'] = r['settle_ms']
        self.fv.append(r)

    def _life(self, sid, i, h, side, tick, q, hi, lh):
        self.life.append(dict(mid=self.mid, sid=sid, mk_hash=h, mk_idx=i, y_side=side, y_ptick=int(tick), shares=float(q), hi_ms=int(hi),
                              lo_ms=int(hi - 100) if lh >= 0 else -1, t_method='pend_inc', f_before=-1, f_after=-1, status='ok', x_lo_hard_ms=int(lh),
                              x_rec_rank=0, x_ahead_hi_S_FIFO=0.))

    def maker_fill(self, addr, t, side, q, p, h=None, status='ok'):
        sid = self._sid()
        h = h or f'0xmk{self.mid}{self.n:06d}'
        self._fv(sid=sid, role='maker', addr=addr, hash=h, cp_addr=TAKER_X, y_side=side, s=1 if side == 'bid' else -1, q=float(q), p=float(p),
                 settle_ms=self.ws + t, status=status, src_status=status)
        self._life(sid, 0, h, side, round(p * 100), q, self.ws + t, self.ws + t - 2_000)
        return sid

    def take(self, addr, t, s, legs, tk=None, status='ok', life_hi=None, life_lh=None):
        """legs: (q, p, maker address); the maker of a leg equal to addr is a self-trade (both rows flagged)."""
        sid = self._sid()
        tk = tk or f'0xtk{self.mid}{self.n:06d}'
        mside = 'ask' if s > 0 else 'bid'
        hi = self.ws + (t if life_hi is None else life_hi)
        lh = hi - 2_000 if life_lh is None else life_lh
        for i, (q, p, mk) in enumerate(legs):
            selfq = mk == addr
            mh = f'0xo{self.mid}{self.n:06d}{i}'
            self._fv(sid=sid, mk_idx=i, px_rank=i, role='taker', addr=addr, hash=tk, cp_addr=mk, y_side='bid' if s > 0 else 'ask', s=int(s), q=float(q),
                     p=float(p), self_trade=selfq, settle_ms=self.ws + t, status=status, src_status=status)
            if selfq:
                self._fv(sid=sid, mk_idx=i, px_rank=i, role='maker', addr=addr, hash=mh, cp_addr=addr, y_side=mside, s=-int(s), q=float(q), p=float(p),
                         self_trade=True, settle_ms=self.ws + t, status=status, src_status=status)
            self._life(sid, i, mh, mside, round(p * 100), q, hi, lh)
        return sid, tk

    def order(self, addr, side, tick, times, post_lo=None, q=5., rhi_extra=1_000):
        h = f'0xord{self.mid}{len(self.orders):04d}'
        his = [self.ws + t for t in times]
        for x in his:
            sid = self._sid()
            self._fv(sid=sid, role='maker', addr=addr, hash=h, cp_addr=TAKER_X, y_side=side, s=1 if side == 'bid' else -1, q=float(q), p=tick / 100.,
                     settle_ms=x)
            self._life(sid, 0, h, side, tick, q, x, x - 2_000)
        lhs = [x - 2_000 for x in his]
        p_lo, p_hi = (his[0], max(lhs)) if max(lhs) >= his[0] else (-1, -1)
        for sc in T.SCEN:
            self.orders.append(dict(scenario=sc, mid=self.mid, mk_hash=h, mk_signer=addr, y_side=side, y_ptick=int(tick), n_fills=len(his),
                                    cum_shares=q * len(his), first_hi_ms=his[0], last_hi_ms=his[-1], size_lb=q * len(his), size_exact=NAN,
                                    post_lo_ms=-1 if post_lo is None else self.ws + post_lo, post_hi_ms=his[0], persist_lo_ms=p_lo, persist_hi_ms=p_hi,
                                    ahead_lo=0., ahead_hi=NAN, removal_set='cancelled_or_expired|exhausted', removal_lo_ms=max(lhs),
                                    removal_hi_ms=his[-1] + rhi_extra, removal_class='constrained', removal_method='book_gone', status='ok', x_size_ub=NAN,
                                    post_class='unresolved', persist_class='observed', queue_class='unresolved', size_class='observed'))
        return h, his

    def link(self, addr, side, kind, P, Q, pt, qt):
        self.links.append(dict(mid=self.mid, mk_signer=addr, y_side=side, kind=kind, prev_hash=P[0], next_hash=Q[0], prev_ptick=int(pt), next_ptick=int(qt),
                               prev_last_hi_ms=int(P[1][-1]), next_first_hi_ms=int(Q[1][0]), gap_ms=int(Q[1][0] - P[1][-1]), x_latest=True, status='ok'))

    def market(self, seed=0, jumps=(), book_shift=None, extra=None):
        fv_rows = self.fv + (extra or [])
        fv = pd.DataFrame(fv_rows, columns=list(FV_READ)).astype(FV_DT) if fv_rows else pd.DataFrame({c: pd.Series(dtype=FV_DT[c]) for c in FV_READ})
        life = pd.DataFrame(self.life) if self.life else None
        legs = CC.join_windows(fv, life)
        fr, lv, tob = book_frames(self.mid, self.ws, *(book_shift or (None, 0)))
        return CC.build_market(self.day, self.mid, self.ws, self.ws + 300_000, T.TOP3, btc=btc_dict(make_bt(self.ws, seed, jumps=jumps), self.day),
                               book=True, tob=tob, frames=fr, levels=lv, legs=legs, life_fills=life,
                               orders=pd.DataFrame(self.orders) if self.orders else None)

    def links_df(self):
        return pd.DataFrame(self.links).astype(dict(mid='int64', prev_ptick='int64', next_ptick='int64', prev_last_hi_ms='int64', next_first_hi_ms='int64',
                                                    gap_ms='int64', x_latest='bool')) if self.links else LNK_EMPTY.copy()


def world_A():
    """Hand-computed take classes for rank 1 (positions from the ok, non-self legs in settle order; see the expected table)."""
    m = Mk(DAY_A, MID_A)
    m.maker_fill(A1, 10_000, 'bid', 30, .45)
    _, tk = m.take(A1, 20_000, -1, [(10, .44, OTHER)])                  # 30 -> 20 reduce_partial
    m.take(A1, 30_000, -1, [(20, .44, OTHER)], tk=tk)                   # 20 -> 0 full_close (same taker order: multi-record)
    m.maker_fill(A1, 40_000, 'bid', 10, .45)
    m.take(A1, 50_000, -1, [(15, .44, OTHER)])                          # 10 -> -5 reversal_reducing
    m.maker_fill(A1, 60_000, 'bid', 15, .45)
    m.take(A1, 70_000, -1, [(15, .44, OTHER), (15, .43, OTHER)])        # 10 -> -20 overshoot (a sweep: two prices)
    m.maker_fill(A1, 80_000, 'bid', 20, .45)
    m.take(A1, 90_000, 1, [(5, .56, OTHER)])                            # 0 -> 5 from_flat
    m.take(A1, 100_000, 1, [(5, .56, OTHER)])                           # 5 -> 10 add
    m.maker_fill(A1, 109_200, 'bid', 15, .45)                           # pending for the next take (inside its hard window)
    m.take(A1, 110_000, -1, [(12, .44, OTHER)])                         # recorded 25 -> 13 reduce; certain 10 -> -2 reversal: timing_sensitive
    m.maker_fill(A1, 120_000, 'bid', 5, .45)                            # simultaneous (same settle ms, another record)
    m.take(A1, 120_000, -1, [(8, .44, OTHER)])                          # 13 -> 5 / incl 18 -> 10: reduce_partial both
    m.maker_fill(A1, 130_000, 'bid', 10, .45)
    m.take(A1, 130_000, -1, [(15, .44, OTHER)])                         # 10 -> -5 reversal / incl 20 -> 5 reduce: timing_sensitive
    m.take(A1, 140_000, 1, [(6, .56, OTHER), (4, .56, A1)])             # self leg: dq +6, 5 -> 11 add, self_trade
    m.take(A1, 150_000, -1, [(3, .44, OTHER)], status='src_status')     # excluded legs: 11 -> 8 reduce_partial, excluded_row
    m.take(A1, 160_000, 1, [(1, .56, OTHER)], life_lh=-1)               # window_unbounded: 11 -> 12 add (no claim)
    m.take(A1, 170_000, 1, [(1, .56, OTHER)], life_hi=169_500)          # receipt_before_uuid: 12 -> 13 add (no claim)
    return m


EXPECT_A = [(20_000, 'reduce_partial'), (30_000, 'full_close'), (50_000, 'reversal_reducing'), (70_000, 'overshoot'), (90_000, 'from_flat'),
            (100_000, 'add'), (110_000, 'timing_sensitive'), (120_000, 'reduce_partial'), (130_000, 'timing_sensitive'), (140_000, 'add'),
            (150_000, 'reduce_partial'), (160_000, 'add'), (170_000, 'add')]
EXPECT_A_REC = {20_000: 30., 30_000: 20., 50_000: 10., 70_000: 10., 90_000: 0., 100_000: 5., 110_000: 25., 120_000: 13., 130_000: 10., 140_000: 5.,
                150_000: 11., 160_000: 11., 170_000: 12.}


def world_B():
    """A burst-reducing taker: three own bid fills of 10 sh in 0.6 s, then a 30 sh sell 2.6 s after the last (certain burst of 3)."""
    m = Mk(DAY_B, MID_B)
    for k in range(20):
        b = 10_000 + 13_000 * k
        for d in (0, 300, 600):
            m.maker_fill(A1, b + d, 'bid', 10, .45)
        m.take(A1, b + 3_200, -1, [(30, .44, OTHER)])
    return m


def world_C(seed=7):
    """A random taker: own maker fills (Poisson, mean 4 s, random side) and takes (Poisson, mean 6 s, random side) drawn independently."""
    m = Mk(DAY_C, MID_C)
    rng = np.random.default_rng(seed)
    t = 5_000.
    while True:
        t += rng.exponential(4_000.)
        if t > 290_000:
            break
        m.maker_fill(A1, int(t), 'bid' if rng.random() < .5 else 'ask', 10, .45)
    u = 7_000.
    while True:
        u += rng.exponential(6_000.)
        if u > 288_000:
            break
        s = 1 if rng.random() < .5 else -1
        m.take(A1, int(u), s, [(5, .56 if s > 0 else .44, OTHER)])
    return m


def world_D():
    """The action fixture for rank 1 (expected counts in the actions section)."""
    m = Mk(DAY_A, MID_D)
    o1 = m.order(A1, 'bid', 45, [10_000, 20_000])
    o3 = m.order(A1, 'bid', 44, [12_000, 25_000])
    m.order(A1, 'ask', 55, [13_000, 16_000])
    o2 = m.order(A1, 'bid', 45, [30_000])
    o5 = m.order(A1, 'bid', 46, [40_000], post_lo=35_000)
    m.link(A1, 'bid', 'refill', o1, o2, 45, 45)
    m.link(A1, 'bid', 'reprice', o2, o5, 45, 46)
    del o3
    return m


class SynthCC:
    """chain_context's API surface over synthetic markets: load_market returns prebuilt chain_context.build_market objects; every query is
    the REAL chain_context function."""
    VERSION = 'synthetic markets / real chain_context functions'
    MOVE_THR, POS_EDGES = CC.MOVE_THR, CC.POS_EDGES
    btc_ret, book, level_size, position = (staticmethod(CC.btc_ret), staticmethod(CC.book), staticmethod(CC.level_size), staticmethod(CC.position))

    def __init__(self, markets, fv_path):
        self.markets, self.fv_path, self.calls = markets, Path(fv_path), Counter()

    def load_market(self, day, mid, addrs, root=None, hash_cache=None):
        self.calls[(day, int(mid))] += 1
        if hash_cache is not None:
            hash_cache.setdefault(str(self.fv_path.resolve()), E.sha256(self.fv_path))
        return self.markets[(day, int(mid))]


def empty_market(day, mid):
    ws = ws_of(day, mid)
    return CC.build_market(day, int(mid), ws, ws + 300_000, T.TOP3, btc=btc_dict(make_bt(ws, 1), day))


def write_root(root, worlds, top3=T.TOP3):
    """identities.json, a stand-in fills_valued file (only its sha256 is bound) and the x_links files of all 19 BOOK markets."""
    (root / 'econ').mkdir(parents=True)
    (root / 'life').mkdir()
    others = [f'0x{i:040x}' for i in range(1, 8)]
    (root / 'econ' / 'identities.json').write_text(json.dumps(dict(ex_ante=dict(day=C.ID_DAY, top10_shares=[
        dict(rank=i + 1, addr=a) for i, a in enumerate(list(top3) + others)]))))
    pd.DataFrame(dict(x=[1, 2, 3])).to_parquet(root / 'econ' / 'fills_valued.parquet', index=False)
    for d, mids in BOOKM.items():
        for mid in mids:
            w = worlds.get((d, mid))
            (w.links_df() if w is not None else LNK_EMPTY.copy()).to_parquet(root / 'life' / f'{d}_{mid}_x_links.parquet', index=False)
    return root / 'econ' / 'fills_valued.parquet'


# ------------------------------------------------------------------ sections ------------------------------------------------------------------
def s_classes():
    cases = [(30., -10., 'reduce_partial'), (20., -20., 'full_close'), (10., -15., 'reversal_reducing'), (10., -30., 'overshoot'),
             (10., -20., 'overshoot'), (0., 5., 'from_flat'), (5., 5., 'add'), (-5., 3., 'reduce_partial'), (-5., 8., 'reversal_reducing'),
             (-5., 5., 'full_close'), (7., 0., 'no_change'), (NAN, 1., 'unclassified'), (1e-7, -3., 'from_flat')]
    got = [T.CLS[c] for c in T.classify([b for b, _, _ in cases], [d for _, d, _ in cases])]
    check('classes: hand-computed (before, dq) cases', got == [e for _, _, e in cases], list(zip(got, [e for _, _, e in cases])))
    grid = [(b, d) for b in (-40., -10., -5., -1e-7, 0., 1e-7, 5., 10., 25.) for d in (-50., -20., -10., -7.5, -5., -1e-7, 0., 1e-7, 5., 10., 30.)]
    mine = [T.CLS[c] for c in T.classify([b for b, _ in grid], [d for _, d in grid])]
    theirs = [OE.inv_class(b, b + d) for b, d in grid]
    bad = [(g, a, b) for g, a, b in zip(grid, mine, theirs) if a != b and not (a == 'no_change' and abs(g[1]) <= 1e-6)]
    check('classes: equal to order_estimator.inv_class on a (before, dq) grid', not bad, bad[:6])
    v = T.combine(np.array([2, 2, 2, 4]), np.array([2, -1, 4, 4]), np.array([-1, 2, 2, -1]))
    check('combine: common class else timing_sensitive (absent variants ignored)', [T.CLS[x] for x in v] ==
          ['reduce_partial', 'reduce_partial', 'timing_sensitive', 'reversal_reducing'], [T.CLS[x] for x in v])
    mi = T.mech_inventory(np.array([2, 1, 0, 2, 5, 7]), np.array([3, 5, 0, 1, -1, 2]), np.array([-1, -1, -1, -1, 5, -1]))
    check('inventory_unwind: compatible / contradicted / unidentified / timing_sensitive', [T.MST[x] for x in mi] ==
          ['compatible', 'contradicted', 'unidentified', 'timing_sensitive', 'contradicted', 'unidentified'], [T.MST[x] for x in mi])
    dc = T.dir_category([5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 3, 0], [5, 0, 0, 2, 3, 0], [0, 5, 0, 1, 0, 0])
    check('direction window category: with / against / flat / mixed / gap / missing', [T.DIRCAT[x] for x in dc] ==
          ['with', 'against', 'flat', 'mixed', 'gap', 'missing'], [T.DIRCAT[x] for x in dc])
    check('directional mechanism map', [T.MST[x] for x in T.mech_directional(dc)] ==
          ['compatible', 'contradicted', 'insufficient_comparison', 'timing_sensitive', 'unidentified', 'unidentified'])
    check('after_fill_burst mechanism', [T.MST[x] for x in T.mech_burst([3, 2, 0, 5], [3, 2, 0, 9])] ==
          ['compatible', 'timing_sensitive', 'contradicted', 'compatible'])
    check('position bins = chain_context POS_EDGES', list(T.pos_bin([0, 49.9, 50, 199, 200, 999, 1000, -60])) == [0, 0, 1, 1, 2, 2, 3, 1]
          and tuple(CC.POS_EDGES) == T.POS_BINS)
    check('moves = chain_context MOVE_THR', all(CC.MOVE_THR[k] == v for k, v in T.MOVES))


def takes_of(M, LK=None, rank=0, mut=None):
    ctx = T.Ctx(SynthCC({(M.day, M.mid): M}, Path(__file__)))
    B = T.Build(ctx, OE.link_hypotheses, CP.order_table, Counter())
    B.market(M.day, M.mid, M, LNK_EMPTY.copy() if LK is None else LK, (rank,))
    return B


def s_world_A():
    w = world_A()
    M = w.market(seed=11)
    B = takes_of(M)
    tk = B.takes_table()
    rel = (tk['settle_ms'] - w.ws).tolist()
    got = dict(zip(rel, tk['take_class']))
    check('A: 13 take records, one row each', len(tk) == 13 and tk['sid'].is_unique, len(tk))
    check('A: hand-computed take classes (D = 0)', all(got.get(t) == c for t, c in EXPECT_A), [(t, got.get(t), c) for t, c in EXPECT_A if got.get(t) != c])
    rec = dict(zip(rel, tk['pos_rec']))
    check('A: recorded position before = the hand-computed path (chain_context.position, reconciled)',
          all(abs(rec[t] - v) < 1e-9 for t, v in EXPECT_A_REC.items()), {t: (rec.get(t), v) for t, v in EXPECT_A_REC.items()})
    r = tk.set_index((tk['settle_ms'] - w.ws).astype(int))
    check('A: pending take - recorded reduce_partial, certain reversal_reducing, one pending leg of +15',
          r.at[110_000, 'cls_rec'] == 'reduce_partial' and r.at[110_000, 'cls_cert_0'] == 'reversal_reducing' and r.at[110_000, 'pend_n_0'] == 1
          and abs(r.at[110_000, 'pend_dq_0'] - 15.) < 1e-9 and abs(r.at[110_000, 'pos_cert_0'] - 10.) < 1e-9, r.loc[110_000, ['cls_rec', 'cls_cert_0', 'pend_n_0']].to_dict())
    check('A: simultaneous actions flagged and evaluated both ways', 'simultaneous_action' in r.at[120_000, 'status'] and r.at[120_000, 'n_tie'] == 1
          and r.at[130_000, 'cls_sim'] == 'reduce_partial' and r.at[130_000, 'cls_rec'] == 'reversal_reducing')
    check('A: the overshoot take is a sweep over two prices', r.at[70_000, 'n_px'] == 2 and r.at[70_000, 'take_class'] == 'overshoot'
          and abs(r.at[70_000, 'p_min'] - .43) < 1e-12)
    check('A: self-trade leg nets out (dq +6), flagged', abs(r.at[140_000, 'dq'] - 6.) < 1e-9 and 'self_trade' in r.at[140_000, 'status'])
    check('A: excluded take legs keep their size, flagged excluded_row', abs(r.at[150_000, 'dq'] + 3.) < 1e-9 and 'excluded_row' in r.at[150_000, 'status'])
    check('A: window_unbounded / receipt_before_uuid are no-claim with unidentified mechanisms',
          (not r.at[160_000, 'claim']) and 'window_unbounded' in r.at[160_000, 'status'] and (not r.at[170_000, 'claim'])
          and 'receipt_before_uuid' in r.at[170_000, 'status'] and r.at[160_000, 'inventory_unwind_0'] == 'unidentified'
          and r.at[170_000, 'directional_0'] == 'unidentified')
    check('A: the multi-record taker order is listed with its two records', r.at[20_000, 'tk_n_records'] == 2 and r.at[30_000, 'tk_seq'] == 2)
    mech = {t: r.at[t, 'inventory_unwind_0'] for t in (20_000, 70_000, 90_000, 100_000, 110_000)}
    # 110 s: the variants' CLASSES differ (reduce_partial vs reversal_reducing -> take_class timing_sensitive) but both actually reduce
    # |position|, so contract 6 makes inventory_unwind compatible (corrected expectation after run mt_take_selftest: the tool was right)
    check('A: inventory_unwind statuses', mech == {20_000: 'compatible', 70_000: 'contradicted', 90_000: 'unidentified', 100_000: 'contradicted',
                                                   110_000: 'compatible'}, mech)
    check('A: burst counts at the 110 s take: certain 0, possible 1 (the pending leg) -> after_fill_burst timing_sensitive',
          r.at[110_000, 'n_cert_0'] == 0 and r.at[110_000, 'n_poss_0'] == 1 and r.at[110_000, 'after_fill_burst_0'] == 'timing_sensitive')
    check('A: no own fill near the 20 s take -> after_fill_burst contradicted', r.at[20_000, 'after_fill_burst_0'] == 'contradicted')
    ts = tk[['touch_start_0', 'spread_start_0', 'depth_taken_start_0']].notna().all(1)
    check('A: the book start state (touch, spread, depth at the taken touch) is read for every take', bool(ts.all()), int(ts.sum()))
    x = r.loc[20_000]
    b = CC.book(M, np.array([w.ws + 18_000]))
    check('A: taken-side touch = chain_context.book bid at the window start (a sell), ticks through = bid - min leg tick',
          x['touch_start_0'] == float(b['bid_tick'][0]) and abs(x['through_worst_tk_0'] - (float(b['bid_tick'][0]) - 44.)) < 1e-9,
          (x['touch_start_0'], float(b['bid_tick'][0])))
    # the window-instant set and categories against chain_context.over_window (both sides)
    claim = tk[tk['claim']]
    for sd in (1, -1):
        sub = claim[claim['s'] == sd]
        if not len(sub):
            continue
        fn = (lambda s_: (lambda M_, t_: (CC.align_cat(s_, CC.btc_ret(M_, t_)['r_5s'], 1.0), CC.btc_ret(M_, t_)['ok'])))(sd)
        R_ = CC.over_window(M, fn, sub['x_lo_hard_ms'].to_numpy(np.int64), sub['hi_ms'].to_numpy(np.int64), D_ms=250, mode='category')
        mapped = [('mixed' if st == 'timing_sensitive' else 'gap' if st == 'partially_missing' else 'missing' if st == 'missing' else c)
                  for st, c in zip(R_['status'], R_['cats'])]
        check(f'A: r_5s window categories (D 250, side {sd:+d}) equal chain_context.over_window',
              mapped == sub['cat_r_5s_250'].tolist() and list(R_['n_points']) == sub['n_inst_250'].tolist(), list(zip(mapped, sub['cat_r_5s_250'])))


def s_association(markets):
    wB, wC = world_B(), world_C()
    MB, MC = wB.market(seed=21), wC.market(seed=22)
    for nm, M in (('B', MB), ('C', MC)):
        markets[(M.day, M.mid)] = M
    B = T.Build(T.Ctx(SynthCC({(MB.day, MB.mid): MB, (MC.day, MC.mid): MC}, Path(__file__))), OE.link_hypotheses, CP.order_table, Counter())
    B.market(MB.day, MB.mid, MB, LNK_EMPTY.copy(), (0,))
    B.market(MC.day, MC.mid, MC, LNK_EMPTY.copy(), (0,))
    tk = B.takes_table()
    ctl, info = B.controls_table((DAY_B, DAY_C))
    tb = tk[tk['day'] == DAY_B]
    check('B: 20 burst takes, each with 3 certain own maker fills in the 5 s before its window, full_close',
          len(tb) == 20 and (tb['n_cert_0'] == 3).all() and (tb['take_class'] == 'full_close').all(), tb[['n_cert_0', 'take_class']].head(3).to_dict())

    def row(day, f, comp='take_vs_control', D=0):
        r = ctl[(ctl['comparison'] == comp) & (ctl['scope'] == 'rank1') & (ctl['day'] == day) & (ctl['D'] == D) & (ctl['feature'] == f)]
        return r.iloc[0] if len(r) else None
    rb = row(DAY_B, 'burst3_certain')
    check('B: the burst association is recovered against matched no-take controls (obs 1, null <= 0.05, p_hi <= 0.01, >= 200 draws)',
          rb is not None and rb['obs'] == 1. and rb['null_mean'] <= .05 and rb['p_hi'] <= .01 and rb['n_draws'] >= T.MIN_DRAWS and rb['n_takes'] == 20,
          None if rb is None else rb[['obs', 'null_mean', 'p_hi', 'n_draws', 'n_takes']].to_dict())
    rc = row(DAY_C, 'burst3_certain')
    check('C: a random taker shows no burst association (|obs - null| < 0.2, p_hi and p_lo > 0.001)',
          rc is not None and rc['n_takes'] >= 25 and abs(rc['obs'] - rc['null_mean']) < .2 and rc['p_hi'] > .001 and rc['p_lo'] > .001,
          None if rc is None else rc[['obs', 'null_mean', 'p_hi', 'p_lo', 'n_takes']].to_dict())
    rr = row(DAY_C, 'tau_s', 'reducing_vs_adding')
    check('C: reducing vs adding contrast rows carry both denominators and a permutation null',
          rr is not None and rr['n_reducing'] > 0 and rr['n_adding'] > 0 and rr['n_draws'] >= T.MIN_DRAWS, None if rr is None else rr.to_dict())
    pl = info['pool_level']
    check('controls: every take drawn from its own stratum or a counted fallback', sum(pl.values()) == len(tk) and pl['no_control'] == 0, pl)


def s_actions(markets):
    w = world_D()
    M = w.market(seed=31)
    markets[(M.day, M.mid)] = M
    B = takes_of(M, w.links_df())
    act = B.actions_table((DAY_A,))
    a = act[(act['scope'] == 'rank1') & (act['day'] == DAY_A)].set_index('action_type')
    n = a['n'].to_dict()
    want = {'maker_fill': 8, 'post_new_level': 4, 'post_same_level': 1, 'keep_after_fill': 3, 'refill': 1, 'reprice': 1, 'ladder_overlap': 1,
            'two_sided': 4, 'pull_possible:S_NOPRIO:cancel_possible': 5, 'pull_possible:S_FIFO:cancel_possible': 5}
    check('D: action counts on the ladder / two-sided / refill / reprice fixture', all(n.get(k) == v for k, v in want.items()),
          {k: (n.get(k), v) for k, v in want.items()})
    check('D: two-sided certain 3 (compatible) and possible-only 1 (timing_sensitive)',
          a.at['two_sided', 'n_compatible'] == 3 and a.at['two_sided', 'n_timing_sensitive'] == 1)
    check('D: refill is tied (timing_sensitive), reprice sequential_identified (compatible) through order_estimator.link_hypotheses',
          a.at['refill', 'n_timing_sensitive'] == 1 and a.at['reprice', 'n_compatible'] == 1, a.loc[['refill', 'reprice'], ['n_compatible', 'n_timing_sensitive']].to_dict())
    ex = json.loads(a.at['reprice', 'extra'])
    check('D: the reprice pair keeps replace feasible (churn bound counted)', ex.get('replace_feasible_sum') == 1., ex)
    check('D: filled orders are labelled outcome-selected', a.at['post_new_level', 'population'] == 'outcome_selected_filled_orders')
    check('D: keep_after_fill is observed persistence (compatible, not timing-sensitive at >= 2 s gaps)', a.at['keep_after_fill', 'n_compatible'] == 3)
    check('D: logic rows carry the declared columns', all(c in a.columns for c in ('state_change', 'alternatives', 'proposed_trade_off', 'observed_action',
                                                                                    'resulting_state', 'supporting_evidence', 'competing_explanation',
                                                                                    'what_would_falsify', 'status', 'next_modal')))


def feat_row(B, sid):
    tk = B.takes_table()
    r = tk[tk['sid'] == sid]
    return r.iloc[0].to_dict() if len(r) else None


def same(a, b):
    diff = []
    for k in a:
        x, y = a[k], b.get(k)
        if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
            continue
        if x != y:
            diff.append(k)
    return diff


def s_falsifier():
    w = world_A()
    sid = [r['sid'] for r in w.fv if r['role'] == 'taker' and r['settle_ms'] == w.ws + 20_000][0]
    hi = w.ws + 20_000
    w2 = world_A()
    w2.maker_fill(A1, 25_000, 'ask', 7, .55)                            # an own fill after the take
    base = feat_row(takes_of(w.market(seed=11)), sid)
    after = feat_row(takes_of(w2.market(seed=11, jumps=((hi + 100, 80.),), book_shift=(hi, 3))), sid)
    inside = feat_row(takes_of(w.market(seed=11, jumps=((hi - 700, 80.),))), sid)
    d_after, d_inside = same(base, after), same(base, inside)
    check('falsifier: data after the decision window (BTC, book, own fills) leave every feature of the take unchanged', not d_after, d_after[:8])
    check('falsifier positive control: a BTC move inside the window changes its features', len(d_inside) > 0, d_inside[:8])
    T._MUT_LEAK_MS = 5_000
    try:
        base_l = feat_row(takes_of(w.market(seed=11)), sid)
        after_l = feat_row(takes_of(w2.market(seed=11, jumps=((hi + 100, 80.),), book_shift=(hi, 3))), sid)
    finally:
        T._MUT_LEAK_MS = 0
    d_leak = same(base_l, after_l)
    check('falsifier catches a planted look-ahead (window end read 5 s late)', len(d_leak) > 0, d_leak[:8])


def s_refusals(tmp):
    expect('refuse: a stub without book', lambda: T.Ctx(type('S', (), dict(load_market=lambda *a, **k: None, btc_ret=lambda *a: None,
                                                                          level_size=lambda *a: None, position=lambda *a: None))()),
           'chain_context: book missing')
    bad = type('S2', (), dict(MOVE_THR={'r_1s': .5, 'r_5s': 2., 'r_30s': 2.}, load_market=lambda *a, **k: None, btc_ret=lambda *a: None, book=lambda *a: None,
                              level_size=lambda *a: None, position=lambda *a: None))
    expect('refuse: a chain_context with other move thresholds', lambda: T.Ctx(bad()), 'MOVE_THR')
    w = world_A()
    M = w.market(seed=11)
    R = T.Ctx.rows(M, A1)
    legs, A, K, ex, sp = T.pair_inputs(R)
    expect('refuse: chain_context recorded position differs from the own action sum',
           lambda: T.take_core(legs, A, K, ex, lambda x: (CC.position(M, A1, x)['recorded'] + 1., CC.position(M, A1, x)['certain'])), 'differs from the own')
    R2 = R.copy()
    R2.loc[R2.index[0], 'lo_hard_ms'] = R2.loc[R2.index[0], 'hi_ms'] - 1_500
    expect('refuse: a status-ok leg whose window is not MAX_LAG wide', lambda: T.pair_inputs(R2), 'neither unbounded nor MAX_LAG_MS wide')
    O = pd.DataFrame(world_D().orders)
    O2 = pd.concat([O, O.assign(mk_signer=O['mk_signer'].str.lower(), mk_hash=O['mk_hash'] + 'x')], ignore_index=True)
    expect('refuse: a top-3 address under two spellings', lambda: T.market_tables(type('Mx', (), dict(orders=O2, life_fills=pd.DataFrame(w.life)))()),
           'two spellings')
    expect('refuse: an x_links path on a non-measurement day (never opened)', lambda: T.links_path(tmp, C.NOV8, MID_A), 'not a measurement day')
    expect('refuse: a pilot on a non-measurement day', lambda: T.pilot('20261028', 1, inp=tmp), 'not a measurement day')


def s_end_to_end(tmp, markets):
    wA, wD = world_A(), world_D()
    for w, sd in ((wA, 11), (wD, 31)):
        M = w.market(seed=sd)
        markets[(M.day, M.mid)] = M
    worlds = {(wA.day, wA.mid): wA, (wD.day, wD.mid): wD}
    for d, mids in BOOKM.items():
        for mid in mids:
            markets.setdefault((d, mid), empty_market(d, mid))
    root = tmp / 'root'
    fvp = write_root(root, worlds)
    sha = E.sha256(fvp)
    cc = SynthCC(markets, fvp)
    out = tmp / 'out_run'
    rec = T.run(inp=root, out=out, cc=cc, require_freeze=False, fv_sha=sha, log=lambda *_: None)
    files = sorted(p.name for p in out.iterdir())
    check('e2e: run writes the five outputs once (partial directory renamed)', files == ['actions.csv', 'receipt.json', 'summary.csv', 'take_controls.csv',
                                                                                         'takes.csv'] and not list(tmp.glob('.out_run.partial*')), files)
    tk = pd.read_csv(out / 'takes.csv')
    check('e2e: takes.csv holds one row per take record of the synthetic markets (13 + 20 + random)', len(tk) == 13 + 20 + len(
        [r for r in world_C().fv if r['role'] == 'taker' and not r['self_trade']]) and tk['sid'].is_unique, len(tk))
    rj = json.loads((out / 'receipt.json').read_text())
    check('e2e: receipt binds inputs, sources, constants, statuses and CPU', all(k in rj for k in ('inputs', 'sources', 'constants', 'statuses', 'cpu_s',
                                                                                                   'class_names', 'pools', 'outputs'))
          and rj['inputs'].get('econ/fills_valued.parquet') == sha and rj['counts'].get('markets') == 19 and rj['outputs']['takes'] == len(tk),
          dict(keys=sorted(rj)[:14], markets=rj.get('counts', {}).get('markets')))
    check('e2e: every one of the 19 BOOK markets was loaded once through the bridge', len(cc.calls) == 19 and set(cc.calls.values()) == {1}, len(cc.calls))
    ct = pd.read_csv(out / 'take_controls.csv')
    check('e2e: take_controls has take_vs_control and reducing_vs_adding rows for every D', set(ct['comparison']) == {'take_vs_control', 'reducing_vs_adding'}
          and set(ct['D']) == set(T.D_PROFILES), sorted(set(ct['D'])))
    sm = pd.read_csv(out / 'summary.csv')
    check('e2e: summary rows per address x day and pooled', {'rank1', 'pooled'} <= set(sm['scope']) and 'all' in set(sm['day'].astype(str)))
    expect('refuse: write-once (the run output exists)', lambda: T.run(inp=root, out=out, cc=cc, require_freeze=False, fv_sha=sha, log=lambda *_: None),
           'write-once')
    expect('refuse: the freeze (untracked tool files)', lambda: T.run(inp=root, out=tmp / 'out_frz', cc=cc, require_freeze=True, fv_sha=sha, log=lambda *_: None),
           'freeze')
    expect('refuse: a fills_valued hash other than the binding', lambda: T.run(inp=root, out=tmp / 'out_sha', cc=SynthCC(markets, fvp), require_freeze=False,
                                                                                fv_sha='0' * 64, log=lambda *_: None), 'differs from the accepted binding')
    bad_root = tmp / 'bad_root'
    write_root(bad_root, worlds, top3=(T.TOP3[1], T.TOP3[0], T.TOP3[2]))
    expect('refuse: a wrong population (ranks 1-3 differ)', lambda: T.run(inp=bad_root, out=tmp / 'out_pop', cc=SynthCC(markets, fvp), require_freeze=False,
                                                                       fv_sha=sha, log=lambda *_: None), 'ranks 1-3 differ')
    prec, _ = T.pilot(DAY_A, 1, inp=root, out=tmp / 'out_pilot', cc=SynthCC(markets, fvp), fv_sha=sha, log=lambda *_: None)
    check('e2e: the pilot writes its own write-once directory with the measured projection',
          (tmp / 'out_pilot' / 'receipt.json').is_file() and prec['projection']['full_run_cpu_s'] > 0 and prec['kind'] == 'pilot')


def main():
    tmp = Path(tempfile.mkdtemp(prefix='chain_take_selftest_'))
    markets = {}
    try:
        section('classes', s_classes)
        section('world A', s_world_A)
        section('association', s_association, markets)
        section('actions', s_actions, markets)
        section('falsifier', s_falsifier)
        section('refusals', s_refusals, tmp)
        section('end to end', s_end_to_end, tmp, markets)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_bad = sum(not ok for _, ok, _ in RES)
    for name, ok, info in RES:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  :: {info}'))
    print(json.dumps(dict(checks=len(RES), failed=n_bad, vmhwm_mib=round(T.hwm(), 1))))
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
