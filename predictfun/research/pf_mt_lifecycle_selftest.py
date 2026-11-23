"""SELFTEST of pf_mt_lifecycle (MAKER-TRACE-RECONSTRUCTION-v1; 2026-11-22, session aea5bfb9; contract amendment 1 in force: makers[] order
is data, never evidence). Synthetic markets only (no data file is read; the CLI rehearsal writes into a temporary directory); every expected
value is computed by hand in the comments. A failed check raises SystemExit with FAIL. Modes: (none) the checks; `simcheck [dur_ms]` scores
every claim against a price-time-priority venue simulator (makers[] listed in random order) - an observed / constrained claim that fails
is a defect, T1 / T2 must not fire; `mutants` every planted defect in MUTANTS, applied alone to a temporary copy of the module, must fail
a check or the light simcheck. Cases (task list a-i plus controls):
 a two histories (5-share order exhausted vs 50-share order filled 5 and cancelled while another maker refills between snapshots): the
   per-order simulations give identical frames, outputs are identical and every scenario keeps exhausted AND cancelled_or_expired;
   S_FIFO bounds the removal by the next record at the price (fifo_front), S_NOPRIO does not -> removal assumption_sensitive
 b same signer, new hash -> refill link (and a reprice link), separate order rows; capacity posting bounds constrained
 c a repeated partial fill (distinct values) -> persistence observed on [min hi, max lo], size_lb = sum, capacity size bound constrained;
   c2 equal values: the earliest pending match is kept (lo only loosens)
 d a non-usable frame is unknown (no posting bound from it, unlike a usable frame's absent price = 0); a frame gap -> no inference inside
   it, removal bounded only by the next real frame and missing_evidence in the set; hu / hd an off_grid level or a doubled cell is an
   unknown cell (never 0); tu an unknown cell is a possible drop (conservative lo)
 e lo unbounded (masked fill): S_FIFO and S_FIFO_LO disagree on posting and queue -> assumption_sensitive; ahead_hi negative -> NaN
 f an impossible overfill -> capacity_contradiction in every scenario, nothing fitted away
 g amendment 1 rule 1 (price priority) on a worst-first and a 'neither' listing, where the old mk_idx rule answers differently; g2 rule 2
   (time priority at the worst price) with the survivor listed first: the other maker exhausted, the survivor's ahead_lo = its shares;
   g3 no survivor -> count constraint only; T1 / T2 violations counted with denominators and their orders get no exhaustion claim;
   p outputs invariant under a reversed makers[] listing; n capacity-tight exhaustion through another proven-resting order (rule 2 too)
 h a posting lower bound from capacity, identical in all scenarios -> constrained
 i H1 identity conflicts (two ticks; a taker leg of another signer)
 t book-drop timing takes the first (conservative) candidate; pending is preferred over an earlier drop (case b); k fifo_through posting;
   s fifo_front skips a record taken by the order's own signer
 ty / hs / tw the real v1 failure patterns of mt_checks (checks_20261123T011521Z): ty1 a tie in one settle ms never voids a rule-2 group
   (the tied member exhausted, the survivor's ahead_lo; priority_exhaustion / ahead_lo), ty2 a T2 violation certain despite a tie (T1_T2),
   ty3 a survivor value below TOL_SH differs from S_NOPRIO's trivial [0, NaN] (class_scenario_consistency); hs a sub-cent order keeps the
   persistence its fills prove, an identity conflict not (H2_presence); tw two orders shown 0.01 short each give neither a false posting
   bound (tp2) nor a false overfill (tv2) (posting_capacity_bound); every hand case and simulator market also asserts the checker's class
   property (cls_inv: a field group differing across scenarios is assumption_sensitive)
 R refusals with the reason asserted (extract invariants, pending text, CLI day, write-once, memory guard, x_ prefix); V venue oddities
   (bad_leg row, zero shares, bad_frame / unlocated dup_level / bad level frames, pending out of range, a doubled cell) kept as counted
   statuses; CLI rehearsal end to end (markets table, depth / no depth, write-once)."""
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_mt_common as C  # noqa: E402

FAIL = 'LIFECYCLE SELFTEST FAILED'
T0 = 1_792_803_600_000                     # 2026-10-24T01:00Z (synthetic)
GRID = list(range(0, 9600, 100))           # frames every 100 ms, t_rel 0..9500
NAN = float('nan')
DAY = '20261024'


class Mk:
    """synthetic market: records / legs / frames / levels in the extract schemas."""

    def __init__(self, mid):
        self.mid = mid; self.L = []; self.R = []; self.Fr = []; self.Lv = []; self.n = 0
        self.we = (T0 + 10_000) // 1000; self.ws = self.we - 300      # we 10 s after T0: the grid's last frame is within GAP_MS of we

    def rec(self, settle_rel, makers, tk='TK', tk_hash=None, status='ok'):
        """makers: [(hash, signer, YES side, tick, shares)] in makers[] order."""
        self.n += 1; settle = T0 + settle_rel
        sid = str(uuid.UUID(int=(settle << 80) | (7 << 76) | (2 << 62) | (self.mid << 16) | self.n))
        recv = (settle + 3000) * 10 ** 6; nm = len(makers); amt = float(sum(m[4] for m in makers))
        tks = 'ask' if makers[0][2] == 'bid' else 'bid'; th = tk_hash or f'tk{self.mid}_{self.n}'; tx = f'0x{self.mid:032x}{self.n:032x}'
        self.R.append(dict(day=DAY, mid=self.mid, sid=sid, tx=tx, settle_ms=settle, exec_ms=(settle // 1000 + 1) * 1000, first_recv_ns=recv,
                           n_recv=1, amount_filled=amt, price_executed=0.5, n_makers=nm, tk_hash=th, tk_signer=tk,
                           tk_quote='Bid' if tks == 'bid' else 'Ask', tk_index_set=1, tk_price=0.5, tk_amount=amt, tk_fee=0., tk_fee_type='SHARES',
                           tk_y_side=tks, status=status))
        for i, (h, g, sd, tick, q) in enumerate(makers):
            self.L.append(dict(day=DAY, mid=self.mid, sid=sid, tx=tx, settle_ms=settle, exec_ms=(settle // 1000 + 1) * 1000, first_recv_ns=recv,
                               mk_idx=i, n_makers=nm, mk_hash=h, mk_signer=g, mk_quote='Bid' if sd == 'bid' else 'Ask', mk_index_set=1,
                               mk_price=tick / 100, shares=float(q), mk_fee=0., mk_fee_type='SHARES', y_side=sd, y_price=round(tick / 100, 6),
                               y_ptick=tick, tk_hash=th, tk_signer=tk, tk_y_side=tks, phase='live', t_rel_s=(settle / 1000 - self.ws), status='ok'))
        return sid

    def frames(self, times, book, pend=None, flags=None, lvstat=None, dups=()):
        """flags: {t_rel: pf_mt_extract WS status}; lvstat: {(t_rel, side, tick): level status}; dups: cells written twice."""
        for tr in times:
            fid = 1000 + len(self.Fr); pb, pa = pend(tr) if pend else ([], [])
            self.Fr.append(dict(mid=self.mid, fid=fid, upd_ms=T0 + tr, recv_ns=(T0 + tr + 50) * 10 ** 6, conn='c0', order_count=0,
                                pend_bids=json.dumps(pb, separators=(',', ':')), pend_asks=json.dumps(pa, separators=(',', ':')), los_id='',
                                los_side='', los_price=0., los_outcome='', status=(flags or {}).get(tr, 'ok')))
            bk = {('bid', 1): 1., ('ask', 99): 1., **book(tr)}                # a far background level: no synthetic frame is a wiped book
            for (sd, tick), q in sorted(bk.items()):
                if q > 0:
                    row = dict(mid=self.mid, fid=fid, side=sd, ptick=tick, shares=float(q), status=(lvstat or {}).get((tr, sd, tick), 'ok'), x_price=NAN)
                    self.Lv.extend([row, dict(row)] if (tr, sd, tick) in dups else [row])

    def dfs(self):
        mk = lambda rows, n, x=(): pd.DataFrame(rows, columns=list(C.SCHEMAS[n]) + [c for c, _ in x]).astype(dict(C.SCHEMAS[n], **dict(x)))  # noqa: E731
        return (mk(self.L, 'legs'), mk(self.R, 'records'), mk(self.Fr, 'frames'),
                mk(self.Lv, 'levels', (('x_price', 'float64'),)))

    def market(self):
        return dict(mid=self.mid, slug=f'btc-{self.mid}', ws=self.ws, we=self.we, outcome='Up', status='ok')


def sim(orders):
    """orders: {(side, tick): [[(t_rel, size), ...] per order]} piecewise-constant sizes -> book(t_rel) (the aggregate the venue shows)."""
    def size(path, tr):
        v = 0.
        for t_, s in path:
            if t_ <= tr:
                v = s
        return v
    return lambda tr: {cell: sum(size(p, tr) for p in paths) for cell, paths in orders.items()}


def win(*spans):
    """pending-entry helper: [(t_from, t_to, price, usd)] -> f(tr) -> list of [price, usd]."""
    return lambda tr: [[p, u] for a, b, p, u in spans if a <= tr < b]


GROUPS_ = dict(size=('size_lb', 'size_exact'), post=('post_lo_ms', 'post_hi_ms'), persist=('persist_lo_ms', 'persist_hi_ms'),
               queue=('ahead_lo', 'ahead_hi'), removal=('removal_set', 'removal_lo_ms', 'removal_hi_ms'))


def cls_inv(o):
    """the class property the independent checker evaluates (mt_checks _cls): a field group whose values differ across the scenario rows of
    a hash is assumption_sensitive (NaN == NaN) -> [(hash, group)] breaking it (real v1: 0.0 vs NaN queue labelled unresolved)."""
    g, out = o.groupby('mk_hash'), []
    for grp, cols in GROUPS_.items():
        diff = pd.concat([g[c].nunique(dropna=False) > 1 for c in cols], axis=1).any(axis=1)
        sens = g[f'{grp}_class'].agg(lambda s: bool((s == C.SENSITIVE).all()))
        out += [(h, grp) for h in diff.index[(diff & ~sens).to_numpy()]]
    return out


def main(X, quiet=False):  # noqa: C901
    bad, n = [], [0]

    def ok(name, cond):
        n[0] += 1
        if not cond:
            bad.append(name)

    def refuses(word, fn, *a, **kw):
        try:
            fn(*a, **kw)
        except (ValueError, MemoryError) as e:
            return word in str(e) or f'{word!r} not in {str(e)[:160]!r}'
        return 'not refused'

    def O(orders, s, h):
        r = orders[(orders.scenario == s) & (orders.mk_hash == h)]
        return r.iloc[0] if len(r) == 1 else None

    def Fh(fills, h, i=0):
        return fills[fills.mk_hash == h].sort_values('hi_ms').iloc[i]

    PM = {}                                      # markets re-run with makers[] reversed (p: mk_idx is never evidence)
    eqn = lambda a, b: (isinstance(a, float) and math.isnan(a) and math.isnan(b)) or a == b  # noqa: E731
    def run(m):                                  # every hand case also asserts the class property (cls_inv)
        res = X.run_market(*m.dfs(), m.market())
        inv = cls_inv(res[1])
        ok(f'class property market {m.mid} ({inv[:3]})', not inv)
        return res
    S_ALL = tuple(C.SCENARIOS)

    # ---- a: two histories, identical observables ----
    try:
        outs = []
        for hist in ('A', 'B'):
            m = Mk(101)
            if hist == 'A':      # h7 5 sh ahead of h9 45 sh; R1 exhausts h7; R2 fills 10 of h9
                b = sim({('bid', 50): [[(-1, 5.), (1060, 0.)], [(-1, 45.), (2060, 35.)]]})
            else:                # h7 50 sh, fills 5 at 1060, cancels 45 at 1070; h9 posts 45 at 1080; R2 fills 10 of h9
                b = sim({('bid', 50): [[(-1, 50.), (1060, 45.), (1070, 0.)], [(1080, 45.), (2060, 35.)]]})
            m.frames(GRID, b, pend=lambda tr: (win((1100, 1500, .5, 2.5), (2100, 2500, .5, 5.))(tr), []))
            m.rec(1060, [('h7', 'M7', 'bid', 50, 5.)], tk='TA'); m.rec(2060, [('h9', 'M9', 'bid', 50, 10.)], tk='TB')
            outs.append((m.dfs(), run(m))); PM['a'] = m
        (dA, (fA, oA, lA, sA)), (dB, (fB, oB, lB, sB)) = outs
        ok('a the two histories show identical frames / levels / records', all(x.equals(y) for x, y in zip(dA, dB)))
        ok('a identical outputs', fA.equals(fB) and oA.equals(oB) and json.dumps(sA, default=str) == json.dumps(sB, default=str))
        f7, f9 = Fh(fA, 'h7'), Fh(fA, 'h9')
        ok(f'a h7 fill window (lo {f7.lo_ms - T0}, hi {f7.hi_ms - T0}, {f7.t_method}, {f7.f_before}, {f7.f_after}, {f7.status})',
           (f7.hi_ms, f7.lo_ms, f7.t_method, f7.f_before, f7.f_after, f7.status) == (T0 + 1060, T0 + 1000, 'pend_inc', 1010, 1011, 'ok'))
        ok('a h9 fill window', (f9.hi_ms, f9.lo_ms, f9.f_before, f9.f_after) == (T0 + 2060, T0 + 2000, 1020, 1021))
        for s in S_ALL:
            r = O(oA, s, 'h7')
            ok(f'a {s} h7 keeps exhausted and cancelled ({r.removal_set})', {'exhausted', 'cancelled_or_expired'} <= set(r.removal_set.split('|')))
            ok(f'a {s} h7 removal_lo = its evidence-free lower end', r.removal_lo_ms == T0 + 1060 - C.MAX_LAG_MS)
        r = O(oA, 'S_FIFO', 'h7')
        ok(f'a S_FIFO h7 removal bounded by R2 ({r.removal_set}, {r.removal_hi_ms - T0}, {r.removal_method})',
           (r.removal_set, r.removal_hi_ms, r.removal_method) == ('cancelled_or_expired|exhausted', T0 + 2060, 'fifo_front'))
        ok('a S_FIFO_LO h7 same bound', O(oA, 'S_FIFO_LO', 'h7').removal_hi_ms == T0 + 2060)
        r = O(oA, 'S_NOPRIO', 'h7')
        ok(f'a S_NOPRIO h7 unbounded ({r.removal_set}, {r.removal_hi_ms}, {r.removal_method})',
           (r.removal_set, r.removal_hi_ms, r.removal_method) == ('cancelled_or_expired|end_censored|exhausted', -1, 'end_open'))
        r = O(oA, 'S_FIFO', 'h7')
        ok(f'a h7 classes ({r.size_class}, {r.post_class}, {r.persist_class}, {r.queue_class}, {r.removal_class})',
           (r.size_class, r.post_class, r.persist_class, r.queue_class, r.removal_class) == (C.OBSERVED, C.UNRESOLVED, C.UNRESOLVED, C.SENSITIVE, C.SENSITIVE))
        ok(f'a h7 S_FIFO queue [{r.ahead_lo}, {r.ahead_hi}] = [0, 50 - 5]', (r.ahead_lo, r.ahead_hi) == (0., 45.))
        r = O(oA, 'S_NOPRIO', 'h7'); ok('a h7 S_NOPRIO queue the trivial [0, NaN]', r.ahead_lo == 0. and math.isnan(r.ahead_hi))
        r = O(oA, 'S_FIFO', 'h9')
        ok(f'a h9 removal unresolved, queue [0, 35] ({r.removal_class}, {r.ahead_hi})', r.removal_class == C.UNRESOLVED and r.ahead_hi == 35.)
        ok('a size_lb / size_exact', O(oA, 'S_FIFO', 'h7').size_lb == 5. and math.isnan(O(oA, 'S_FIFO', 'h7').size_exact))
        ok('a one row per hash per scenario', len(oA) == 6 and not oA.duplicated(['scenario', 'mk_hash']).any())
        ok('a fill statuses reconcile', sum(sA['fills']['status'].values()) == sA['fills']['n'] == 2)
    except Exception as e:  # noqa: BLE001
        bad.append(f'a stopped ({e!r})'[:300])

    # ---- b: same signer, new hash -> refill; then reprice; pending preferred over an earlier drop ----
    try:
        m = Mk(102)
        m.frames(GRID, lambda tr: {('ask', 60): 3 if tr < 1100 else (4 if 2000 <= tr < 3100 else 0), ('ask', 61): 4 if 4000 <= tr < 5100 else 0},
                 pend=lambda tr: ([], win((1100, 1500, .6, 1.8), (3100, 3500, .6, 2.4), (5100, 5500, .61, 2.44))(tr)))
        m.rec(1060, [('hA', 'MB', 'ask', 60, 3.)], tk='T1'); m.rec(3060, [('hB', 'MB', 'ask', 60, 4.)], tk='T2')
        m.rec(5060, [('hC', 'MB', 'ask', 61, 4.)], tk='T3')
        f, o, lk, s_ = run(m)
        lk = lk.sort_values(['next_first_hi_ms', 'kind']).reset_index(drop=True)
        got = [tuple(x) for x in lk[['kind', 'prev_hash', 'next_hash', 'gap_ms', 'x_latest', 'label']].itertuples(index=False)]
        ok(f'b links {got}', got == [('refill', 'hA', 'hB', 2000, True, 'descriptive'), ('reprice', 'hB', 'hC', 2000, True, 'descriptive')])
        ok('b separate order rows', len(o) == 9 and (o.n_fills == 1).all())
        fb_ = Fh(f, 'hB')
        # hA's booking 1.8 < hB's 2.4 and hA's 3-sh drop < hB's 3.5-sh need: no earlier frame could reflect hB (both rules agree)
        ok(f'b hB window (lo {fb_.lo_ms - T0}, {fb_.t_method}, drop fid {fb_.x_drop_fid}, exact {fb_.x_lo_exact_ms - T0})',
           (fb_.lo_ms, fb_.t_method, fb_.x_drop_fid, fb_.x_evidence_fid, fb_.x_lo_exact_ms) == (T0 + 3000, 'pend_inc', 1031, 1031, T0 + 3000))
        r = O(o, 'S_FIFO', 'hB')
        ok(f'b hB capacity posting bound ({r.post_lo_ms - T0}, {r.post_method}, {r.post_class})',
           (r.post_lo_ms, r.post_hi_ms, r.post_method, r.post_class) == (T0 + 1900, T0 + 3060, 'capacity', C.SENSITIVE))
        r = O(o, 'S_FIFO', 'hC')        # the evidence-free bound (fill >= 3060: book 0 before 3000) differs from the clocks' 3900
        ok(f'b hC capacity posting bound ({r.post_lo_ms - T0})', (r.post_lo_ms, r.post_class) == (T0 + 3900, C.SENSITIVE))
        r = O(o, 'S_NOPRIO', 'hA')
        ok(f'b hA gone at the next frame in every scenario ({r.removal_set}, {r.removal_hi_ms - T0}, {r.removal_method}, {r.removal_class})',
           (r.removal_set, r.removal_hi_ms, r.removal_method, r.removal_class) == ('cancelled_or_expired|exhausted', T0 + 1100,
                                                                                    'book_empty_after+book_gone', C.CONSTRAINED)
           and O(o, 'S_FIFO', 'hA').removal_hi_ms == T0 + 1100)
        ok('b post unresolved without a bound', O(o, 'S_FIFO', 'hA').post_class == C.UNRESOLVED and O(o, 'S_FIFO', 'hA').post_lo_ms == -1)
    except Exception as e:  # noqa: BLE001
        bad.append(f'b stopped ({e!r})'[:300])

    # ---- c: repeated partial fill ----
    try:
        m = Mk(103)                                                   # benign extract flags (new_conn, dup, same_ms_changed) keep the frame
        m.frames(GRID, lambda tr: {('bid', 40): 12 if tr < 1100 else (11 if tr < 2100 else (9 if tr < 3100 else 6))},
                 pend=lambda tr: (win((1100, 1500, .4, .4), (2100, 2500, .4, .8), (3100, 3500, .4, 1.2))(tr), []),
                 flags={500: 'new_conn', 1000: 'dup', 2000: 'new_conn+same_ms_changed'})
        for st, q, tk in ((1060, 1., 'T1'), (2060, 2., 'T2'), (3060, 3., 'T3')):
            m.rec(st, [('hR', 'MR', 'bid', 40, q)], tk=tk)
        f, o, lk, s_ = run(m)
        ok('c fill lows from distinct pending values', [int(x - T0) for x in f.sort_values('hi_ms').lo_ms] == [1000, 2000, 3000])
        r = O(o, 'S_FIFO', 'hR')
        ok(f'c persistence observed on [min hi, max lo] ({r.persist_lo_ms - T0}, {r.persist_hi_ms - T0}, {r.persist_class})',
           (r.persist_lo_ms, r.persist_hi_ms, r.persist_class) == (T0 + 1060, T0 + 1060, C.OBSERVED))
        ok(f'c size_lb = sum ({r.size_lb}, {r.n_fills})', (r.size_lb, r.cum_shares, r.n_fills) == (6., 6., 3))
        # presence frames 11..30: need 2 + 3 = 5 on 11..20 (shown 11), 3 on 21..30 (shown 9): min slack 6 -> size <= 12 in every scenario
        ok(f'c capacity size bound only under the clocks ({r.x_size_ub}, {r.size_class})', r.x_size_ub == 12. and math.isnan(r.size_exact)
           and r.size_class == C.SENSITIVE)
        ok(f'c queue at the first fill [0, 12 - 6] ({r.ahead_lo}, {r.ahead_hi})', (r.ahead_lo, r.ahead_hi) == (0., 6.))
        fx = f.sort_values('hi_ms')
        ok(f'c later fills at the front ({fx.x_ahead_hi_S_FIFO.tolist()})', fx.x_ahead_hi_S_FIFO.tolist() == [6., 0., 0.])
        ok('c no contradiction', r.status == 'ok' and s_['capacity']['orders_contradicted'] == 0)
        ok(f'c benign-flag frames used ({s_["inputs"]["frames_usable"]}, {s_["inputs"]["frames_excluded"]})',
           s_['inputs']['frames_usable'] == 96 and s_['inputs']['frames_excluded'] == {})
        pt = s_['priority_tests']['S_FIFO']
        ok(f'c single-maker records: no price-time test units ({pt["T1_price"]["makers"]}, {pt["T2_time"]["groups"]})',
           pt['T1_price']['makers'] == 0 and pt['T2_time']['groups'] == 0)
    except Exception as e:  # noqa: BLE001
        bad.append(f'c stopped ({e!r})'[:300])

    # ---- d1: a non-ok frame is unknown, an ok frame's absent price is 0 ----
    try:
        m = Mk(104)
        m.frames(GRID, lambda tr: {('ask', 70): 3 if 1000 <= tr <= 3000 else 0}, pend=lambda tr: ([], win((3100, 3500, .7, 2.1))(tr)),
                 flags={900: 'out_of_order'})
        m.rec(3060, [('hD', 'MD', 'ask', 70, 3.)], tk='T1')
        f, o, lk, s_ = run(m)
        fd = Fh(f, 'hD')
        ok(f'd1 window (lo {fd.lo_ms - T0}, {fd.f_before}, {fd.f_after})', (fd.lo_ms, fd.f_before, fd.f_after) == (T0 + 3000, 1030, 1031))
        r = O(o, 'S_FIFO', 'hD')
        ok(f'd1 posting bound from the usable frame at 800, not the out_of_order one at 900 ({r.post_lo_ms - T0}, {r.post_class})',
           (r.post_lo_ms, r.post_method, r.post_class) == (T0 + 800, 'capacity', C.CONSTRAINED))
        ok(f'd1 excluded frame counted ({s_["inputs"]["frames_excluded"]})', s_['inputs']['frames_excluded'] == {'out_of_order': 1} and s_['inputs']['frames_usable'] == 95)
    except Exception as e:  # noqa: BLE001
        bad.append(f'd1 stopped ({e!r})'[:300])

    # ---- d2: a frame gap: nothing inferred inside it ----
    try:
        m = Mk(105)
        m.frames(list(range(0, 2200, 100)) + list(range(5000, 9600, 100)), lambda tr: {('bid', 30): 10 if tr < 2100 else (6 if tr < 5000 else 0)},
                 pend=lambda tr: (win((2100, 2500, .3, 1.2))(tr), []), flags={5000: 'gap'})
        m.rec(2060, [('hG', 'MG', 'bid', 30, 4.)], tk='T1')
        f, o, lk, s_ = run(m)
        fg = Fh(f, 'hG')
        ok('d2 fill window', (fg.lo_ms, fg.hi_ms, fg.f_before, fg.f_after, fg.status) == (T0 + 2000, T0 + 2060, 1020, 1021, 'ok'))
        for s in S_ALL:
            r = O(o, s, 'hG')
            ok(f'd2 {s} removal bounded by the first real frame after the gap ({r.removal_set}, {r.removal_hi_ms - T0}, {r.removal_method})',
               (r.removal_set, r.removal_lo_ms, r.removal_hi_ms, r.removal_method) == ('cancelled_or_expired|exhausted|missing_evidence', T0 + 60, T0 + 5000,
                                                                                       'book_gone+unprinted_drop'))
        ok('d2 removal constrained', O(o, 'S_FIFO', 'hG').removal_class == C.CONSTRAINED and s_['inputs']['frame_gaps'] == 1)
    except Exception as e:  # noqa: BLE001
        bad.append(f'd2 stopped ({e!r})'[:300])

    # ---- e: lo unbounded (masked fill): the clocks disagree ----
    try:
        m = Mk(106)
        m.frames(GRID, lambda tr: {('bid', 45): 0 if tr < 4000 else 20})
        m.rec(4990, [('hE', 'ME', 'bid', 45, 5.)], tk='T1')
        f, o, lk, s_ = run(m)
        fe = Fh(f, 'hE')
        ok(f'e unbounded window ({fe.lo_ms}, {fe.t_method}, {fe.f_before}, {fe.f_after}, {fe.status})',
           (fe.lo_ms, fe.t_method, fe.f_before, fe.f_after, fe.status) == (-1, 'none', -1, 1050, 'fill_lo_unbounded'))
        a_, b_ = O(o, 'S_FIFO', 'hE'), O(o, 'S_FIFO_LO', 'hE')
        # hi clock: reflected at 5000 -> frame before 4900 shows 20: [0, 15]; posting after 3900 (book 0 below 4000)
        # lo clock: at 4990 - 2000 -> frame before 2900 shows 0: raw -5 -> NaN; posting after 2900
        ok(f'e S_FIFO ({a_.ahead_lo}, {a_.ahead_hi}, {a_.post_lo_ms - T0})', (a_.ahead_lo, a_.ahead_hi, a_.post_lo_ms) == (0., 15., T0 + 3900))
        ok(f'e S_FIFO_LO ({b_.ahead_hi}, {b_.x_ahead_hi_raw}, {b_.post_lo_ms - T0}, {b_.status})',
           math.isnan(b_.ahead_hi) and b_.x_ahead_hi_raw == -5. and b_.post_lo_ms == T0 + 2900 and 'ahead_hi_negative' in b_.status)
        ok(f'e classes ({a_.post_class}, {a_.queue_class}, {a_.removal_class})', (a_.post_class, a_.queue_class, a_.removal_class) == (C.SENSITIVE,) * 3)
        ok('e S_FIFO_LO removal at the empty frame 3000', (b_.removal_hi_ms, b_.removal_lo_ms) == (T0 + 3000, T0 + 2990) and O(o, 'S_FIFO', 'hE').removal_hi_ms == -1)
        ok('e per-fill queue columns', fe.x_ahead_hi_S_FIFO == 15. and math.isnan(fe.x_ahead_hi_S_FIFO_LO))
    except Exception as e:  # noqa: BLE001
        bad.append(f'e stopped ({e!r})'[:300])

    # ---- f: impossible overfill ----
    try:
        m = Mk(107)
        m.frames(GRID, lambda tr: {('ask', 55): 12 if tr < 1100 else (9 if tr < 5100 else 0)},
                 pend=lambda tr: ([], win((1100, 1500, .55, 1.1), (5100, 5500, .55, 5.5))(tr)))
        m.rec(1060, [('hF', 'MF', 'ask', 55, 2.)], tk='T1'); m.rec(5060, [('hF', 'MF', 'ask', 55, 10.)], tk='T2')
        f, o, lk, s_ = run(m)
        rows = o[o.mk_hash == 'hF']
        ok(f'f contradiction in every scenario ({rows.status.tolist()})', len(rows) == 3 and rows.status.str.contains('capacity_contradiction').all())
        ok('f not fitted away', (rows.n_fills == 2).all() and (rows.cum_shares == 12.).all() and (rows.x_capacity_violation_frames == 20).all())
        ok('f fills flagged', f.status.str.contains('capacity_contradiction').all())
        ok(f'f summary ({s_["capacity"]})', s_['capacity']['orders_contradicted'] == 1 and s_['capacity']['hard_violation_cell_frames'] == 20)
        r = rows.iloc[0]          # B1: no capacity claim survives a contradiction; nothing constrained
        ok(f'f classes ({r.size_class}, {r.post_class}, {r.persist_class}, {r.removal_class}, {r.post_lo_ms}, {r.removal_hi_ms})',
           (r.size_class, r.post_class, r.persist_class, r.removal_class, r.post_lo_ms, r.removal_hi_ms) == (C.OBSERVED, C.UNRESOLVED, C.OBSERVED, C.UNRESOLVED, -1, -1)
           and rows.status.str.contains('capacity_bounds_void').all() and rows.x_size_ub.isna().all())
        # f2 (review B1): bid 40 shows 5 until 2,000 then 0; fills of 3 (fb 10 / fa 11) and 6 (fb 20 / fa 21): every clock contradicted
        m = Mk(126)
        m.frames(GRID, lambda tr: {('bid', 40): 5 if tr <= 2000 else 0}, pend=lambda tr: (win((1100, 1500, .4, 1.2), (2100, 2500, .4, 2.4))(tr), []))
        m.rec(1060, [('hX', 'MX', 'bid', 40, 3.)], tk='T1'); m.rec(2060, [('hX', 'MX', 'bid', 40, 6.)], tk='T2')
        f, o, lk, s_ = run(m)
        fx = f.sort_values('hi_ms')
        ok('f2 the review windows', list(zip(fx.f_before.tolist(), fx.f_after.tolist())) == [(1010, 1011), (1020, 1021)])
        rows = o[o.mk_hash == 'hX']
        ok(f'f2 contradicted under every clock, capacity void, nothing constrained ({rows.status.tolist()}, {set(rows.post_class)}, {set(rows.removal_class)})',
           rows.status.str.contains('clock_capacity_conflict').all() and rows.status.str.contains('capacity_bounds_void').all()
           and C.CONSTRAINED not in set(rows.post_class) | set(rows.removal_class) | set(rows.size_class))
        # f3: V (2 then 5 sh, 5 s apart) cannot rest at ask 57 shown 0: its violated cells are unknown for W, whose removal bound is then 4100, not 1100
        m = Mk(128)
        m.frames(GRID, lambda tr: {('ask', 57): 3 if tr < 1100 else 0}, pend=lambda tr: ([], win((1100, 1500, .57, 1.71), (6100, 6500, .57, 2.85))(tr)))
        m.rec(1060, [('V3', 'MV3', 'ask', 57, 2.), ('W3', 'MW3', 'ask', 57, 1.)], tk='T1'); m.rec(6060, [('V3', 'MV3', 'ask', 57, 5.)], tk='T2')
        f, o, lk, s_ = run(m)
        r = O(o, 'S_NOPRIO', 'W3')
        ok(f'f3 violated cells are unknown for another order ({r.removal_hi_ms - T0}, {r.removal_set}, {O(o, "S_NOPRIO", "V3").status})',
           (r.removal_hi_ms, r.removal_set) == (T0 + 4100, 'cancelled_or_expired|exhausted|missing_evidence')
           and 'capacity_contradiction' in O(o, 'S_NOPRIO', 'V3').status)
    except Exception as e:  # noqa: BLE001
        bad.append(f'f stopped ({e!r})'[:300])

    # ---- g: amendment 1 rule 1 (price priority) on worst-first / neither listings, where the old mk_idx rule answers differently ----
    try:
        m = Mk(108); PM['g'] = m
        m.rec(1060, [('w', 'MW', 'ask', 53, 2.), ('bq', 'MBQ', 'ask', 52, 3.)], tk='T1')                    # worst-first (66 % of real records)
        m.rec(3060, [('p1', 'MP1', 'ask', 51, 1.), ('p3', 'MP3', 'ask', 53, 1.), ('p2', 'MP2', 'ask', 52, 1.)], tk='T2')   # neither
        f, o, lk, s_ = run(m)
        ex = {h: bool(Fh(f, h).x_exh_price) for h in ('w', 'bq', 'p1', 'p3', 'p2')}
        ok(f'g rule 1 by price, not by listing (the mk_idx rule gives bq, p2 False) ({ex})', ex == dict(w=False, bq=True, p1=True, p3=False, p2=True))
        for s in ('S_FIFO', 'S_FIFO_LO'):
            r = O(o, s, 'bq')
            ok(f'g {s} bq exhausted ({r.size_exact}, {r.removal_set}, {r.removal_method}, {r.removal_hi_ms - T0})',
               (r.size_exact, r.removal_set, r.removal_method, r.removal_lo_ms, r.removal_hi_ms) == (3., 'exhausted', 'fifo_price', T0 + 1060 - C.MAX_LAG_MS, T0 + 1060))
            r = O(o, s, 'w')
            ok(f'g {s} the worst-price w keeps every explanation ({r.removal_set})', math.isnan(r.size_exact)
               and {'exhausted', 'cancelled_or_expired'} <= set(r.removal_set.split('|')))
        r = O(o, 'S_NOPRIO', 'bq')
        ok(f'g S_NOPRIO makes no price claim ({r.size_exact}, {r.size_class}, {r.removal_class})',
           math.isnan(r.size_exact) and (r.size_class, r.removal_class) == (C.SENSITIVE, C.SENSITIVE))
        pt = s_['priority_tests']
        ok(f'g T1 units ({pt["S_FIFO"]["T1_price"]})', (pt['S_FIFO']['T1_price']['makers'], pt['S_FIFO']['T1_price']['violations']) == (3, 0))
        ml = pt['makers_listing']
        ok(f'g listing order kept as a descriptive count ({ml})', (ml['multi_price_records'], ml['best_first'], ml['worst_first'], ml['neither']) == (2, 0, 1, 1))
        # g2 rule 2: two makers at the record's worst (only) price, the survivor x2 listed FIRST (the mk_idx rule would exhaust x2)
        m = Mk(109); PM['g2'] = m
        m.rec(1060, [('x2', 'MX2', 'ask', 52, 6.), ('x1', 'MX1', 'ask', 52, 4.)], tk='T1'); m.rec(2060, [('x2', 'MX2', 'ask', 52, 1.)], tk='T2')
        f, o, lk, s_ = run(m)
        x2r1 = f[f.mk_hash == 'x2'].sort_values('hi_ms').iloc[0]
        for s in ('S_FIFO', 'S_FIFO_LO'):
            r = O(o, s, 'x1')
            ok(f'g2 {s} x1 exhausted by time priority ({r.size_exact}, {r.removal_set}, {r.removal_method})',
               (r.size_exact, r.removal_set, r.removal_method) == (4., 'exhausted', 'fifo_time'))
            r = O(o, s, 'x2')
            ok(f'g2 {s} the survivor x2 had x1 ahead ({r.ahead_lo}, exh {bool(x2r1[f"x_exh_{s}"])}, {x2r1[f"x_ahead_lo_{s}"]})',
               r.ahead_lo == 4. and not bool(x2r1[f'x_exh_{s}']) and x2r1[f'x_ahead_lo_{s}'] == 4. and math.isnan(r.size_exact))
        ok('g2 S_NOPRIO no claim', math.isnan(O(o, 'S_NOPRIO', 'x1').size_exact) and O(o, 'S_NOPRIO', 'x2').ahead_lo == 0.
           and O(o, 'S_FIFO', 'x2').queue_class == C.SENSITIVE)
        t2 = s_['priority_tests']['S_FIFO']['T2_time']
        ok(f'g2 T2 units ({t2})', (t2['groups'], t2['violations'], t2['survivor_groups']) == (1, 0, 1))
        # g3 rule 2 without a survivor: only the count constraint; both orders keep every explanation
        m = Mk(119)
        m.rec(1060, [('y1', 'MY1', 'bid', 47, 2.), ('y2', 'MY2', 'bid', 47, 3.)], tk='T1')
        f, o, lk, s_ = run(m)
        t2 = s_['priority_tests']['S_FIFO']['T2_time']
        ok(f'g3 count constraint only ({t2})', (t2['groups'], t2['no_survivor_groups'], t2['min_exhausted_in_no_survivor_groups']) == (1, 1, 1))
        ok('g3 both keep every explanation', all({'exhausted', 'cancelled_or_expired'} <= set(O(o, 'S_FIFO', h).removal_set.split('|'))
                                                  and math.isnan(O(o, 'S_FIFO', h).size_exact) for h in ('y1', 'y2')))
    except Exception as e:  # noqa: BLE001
        bad.append(f'g stopped ({e!r})'[:300])

    # ---- T1 / T2: the priority tests fire on violations (with denominators); violators get no exhaustion claim ----
    try:
        m = Mk(120); PM['t1'] = m
        m.frames(GRID, lambda tr: {('ask', 52): 5 if tr < 1100 else (4 if tr < 4100 else 2), ('ask', 53): 2 if tr < 4100 else 1,
                                   ('ask', 54): 3 if tr < 1100 else 2},
                 pend=lambda tr: ([], win((1100, 1500, .52, .52), (1100, 1500, .54, .54), (4100, 4500, .52, 1.04), (4100, 4500, .53, .53))(tr)))
        m.rec(1060, [('q9', 'MQ9', 'ask', 54, 1.), ('q1', 'MQ1', 'ask', 52, 1.)], tk='T1')
        m.rec(4060, [('q8', 'MQ8', 'ask', 53, 1.), ('q1', 'MQ1', 'ask', 52, 2.)], tk='T2')    # q1 strictly better again at its last fill
        m.rec(6060, [('q7', 'MQ7', 'ask', 54, 1.), ('q5', 'MQ5', 'ask', 51, 1.)], tk='T3')    # q5 twice in the same settle ms: tied
        m.rec(6060, [('q6', 'MQ6', 'ask', 53, 1.), ('q5', 'MQ5', 'ask', 51, 1.)], tk='T4')
        f, o, lk, s_ = run(m)
        t1 = s_['priority_tests']['S_FIFO']['T1_price']
        ok(f't1 a price-exhausted maker fills again; tied legs in the units, never a later fill ({t1})',
           (t1['makers'], t1['violations'], t1['rate'], t1['violations_certain'], t1['ties']) == (4, 1, 0.25, 1, 2))
        r = O(o, 'S_FIFO', 'q5')     # exhausted by rule 1 in BOTH records of 6060: one of them came first and consumed it -> FIFO-inconsistent
        ok(f't1 q5 exhausted twice in one settle ms: fifo_tie_conflict, no claim ({r.status}, {r.size_exact}, '
           f'{s_["priority_tests"]["S_FIFO"]["tie_conflict_orders"]})', 'fifo_tie_conflict' in r.status and math.isnan(r.size_exact)
           and 'exhausted' in r.removal_set and 'cancelled_or_expired' in r.removal_set and s_['priority_tests']['S_FIFO']['tie_conflict_orders'] == 1
           and not f[f.mk_hash == 'q5'].x_exh_S_FIFO.any())
        ok('t1 no per-fill exhaustion for a T1 violator', not f[f.mk_hash == 'q1'].x_exh_S_FIFO.any())
        r = O(o, 'S_FIFO', 'q1')
        ok(f't1 q1 flagged, no exhaustion claim even at its last fill ({r.status}, {r.size_exact})', 'fifo_reappearance' in r.status
           and math.isnan(r.size_exact) and 'fifo_reappearance' not in O(o, 'S_NOPRIO', 'q1').status)
        m = Mk(121)
        m.rec(1060, [('z1', 'MZ1', 'bid', 46, 1.), ('z2', 'MZ2', 'bid', 46, 1.)], tk='T1')
        m.rec(2060, [('z1', 'MZ1', 'bid', 46, 1.)], tk='T2'); m.rec(3060, [('z2', 'MZ2', 'bid', 46, 1.)], tk='T3')
        f, o, lk, s_ = run(m)
        t2 = s_['priority_tests']['S_FIFO']['T2_time']
        ok(f't2 two worst-price makers both fill again ({t2})', (t2['groups'], t2['violations'], t2['rate'], t2['violations_certain']) == (1, 1, 1.0, 0))
        ok('t2 both flagged, no claims', all('fifo_time_violation' in O(o, 'S_FIFO', h).status and math.isnan(O(o, 'S_FIFO', h).size_exact)
                                             and O(o, 'S_FIFO', h).ahead_lo == 0. for h in ('z1', 'z2')))
    except Exception as e:  # noqa: BLE001
        bad.append(f'T1/T2 stopped ({e!r})'[:300])

    # ---- ty: real v1 failure patterns (checks 20261123T011521Z): a tie in one settle ms never voids a rule-2 group ----
    try:
        # ty1 (20261024_3697902 0x0f6299 / 0xbe8a27): yA fills in the group record Q and in P, both at 3060 (P ranked after Q); xB fills
        #     again at 5060 -> xB the survivor, yA exhausted at Q (FIFO puts P first), xB had yA's 0.4 ahead at its first fill
        m = Mk(140)
        m.rec(3060, [('yA', 'MYA', 'ask', 60, .4), ('xB', 'MXB', 'ask', 60, .6)], tk='T1')
        m.rec(3060, [('yA', 'MYA', 'ask', 60, 1.)], tk='T2'); m.rec(5060, [('xB', 'MXB', 'ask', 60, 2.)], tk='T3')
        f, o, lk, s_ = run(m)
        for s in ('S_FIFO', 'S_FIFO_LO'):
            r, rx = O(o, s, 'yA'), O(o, s, 'xB')
            ok(f'ty1 {s} tied yA exhausted by time priority ({r.size_exact}, {r.removal_set}, {r.removal_method}, {r.status}); xB ahead_lo {rx.ahead_lo}',
               (r.removal_set, r.removal_method, r.removal_hi_ms) == ('exhausted', 'fifo_time', T0 + 3060) and abs(r.size_exact - 1.4) < 1e-12
               and not ({'fill_order_tie', 'fifo_tie_conflict', 'fifo_time_violation'} & set(r.status.split('+'))) and abs(rx.ahead_lo - .4) < 1e-12 and rx.queue_class == C.SENSITIVE and math.isnan(rx.size_exact))
        t2 = s_['priority_tests']['S_FIFO']['T2_time']
        ok(f'ty1 the tie group is counted ({t2})', (t2['groups'], t2['violations'], t2['survivor_groups'], t2['groups_tie']) == (1, 0, 1, 1))
        # ty2 (20261025_3734864 / 20261108_4353706): zX also fills in another 3060 record, but BOTH members fill again in a later ms -> T2
        m = Mk(141)
        m.rec(3060, [('zX', 'MZX', 'bid', 45, 1.), ('zY', 'MZY', 'bid', 45, 2.)], tk='T1'); m.rec(3060, [('zX', 'MZX', 'bid', 45, 1.)], tk='T2')
        m.rec(4060, [('zY', 'MZY', 'bid', 45, 1.)], tk='T3'); m.rec(6060, [('zX', 'MZX', 'bid', 45, 1.)], tk='T4')
        f, o, lk, s_ = run(m)
        t2 = s_['priority_tests']['S_FIFO']['T2_time']
        ok(f'ty2 a certain settle-order violation despite the tie ({t2})',
           (t2['groups'], t2['violations'], t2['violations_certain'], t2['groups_tie']) == (1, 1, 0, 1))
        ok('ty2 both flagged, no claims', all('fifo_time_violation' in O(o, 'S_FIFO', h).status and math.isnan(O(o, 'S_FIFO', h).size_exact)
                                              and O(o, 'S_FIFO', h).ahead_lo == 0. for h in ('zX', 'zY')))
        # ty3 (class_scenario_consistency): a survivor value below TOL_SH (0.005 ahead) still differs from S_NOPRIO's trivial 0 -> sensitive
        m = Mk(142)
        m.rec(2060, [('tS', 'MTS', 'ask', 58, 1.), ('tT', 'MTT', 'ask', 58, .005)], tk='T1'); m.rec(4060, [('tS', 'MTS', 'ask', 58, 1.)], tk='T2')
        f, o, lk, s_ = run(m)
        ok(f'ty3 a tiny survivor value is assumption_sensitive ({[O(o, s, "tS").ahead_lo for s in S_ALL]}, {O(o, "S_FIFO", "tS").queue_class})',
           [O(o, s, 'tS').ahead_lo for s in S_ALL] == [.005, .005, 0.] and O(o, 'S_FIFO', 'tS').queue_class == C.SENSITIVE
           and O(o, 'S_FIFO', 'tT').size_exact == .005)
        ok('ty3 an order with no queue evidence is unresolved, identical [0, NaN] rows',
           O(o, 'S_FIFO', 'tT').queue_class == C.UNRESOLVED and all(O(o, s, 'tT').ahead_lo == 0. and math.isnan(O(o, s, 'tT').ahead_hi) for s in S_ALL))
    except Exception as e:  # noqa: BLE001
        bad.append(f'ty stopped ({e!r})'[:300])

    # ---- hs (H2_presence, 20261108_4334454): an invalid (sub-cent) order keeps the persistence its fills prove; an identity conflict not ----
    try:
        m = Mk(143)
        for t_ in (1060, 3060, 6060):
            m.rec(t_, [('sc', 'MSC', 'ask', 56, 1.)], tk='T1')
        m.rec(1060, [('ic', 'MIC', 'ask', 57, 1.)], tk='T2'); m.rec(5060, [('ic', 'MIC', 'ask', 58, 1.)], tk='T3')
        L, R, Fr, Lv = m.dfs()
        L.loc[L.mk_hash == 'sc', ['mk_price', 'y_price', 'status']] = [.555, .555, 'off_grid']     # an extract off_grid leg (tick 56 = round(55.5))
        f, o, lk, s_ = X.run_market(L, R, Fr, Lv, m.market())
        ok(f'hs class property ({cls_inv(o)[:3]})', not cls_inv(o))
        for s in S_ALL:
            r, ri = O(o, s, 'sc'), O(o, s, 'ic')
            ok(f'hs {s} sub-cent order: span [1060, 6060 - 2000] observed ({r.status}, {r.persist_lo_ms - T0}, {r.persist_hi_ms - T0}, {r.persist_class})',
               'subcent_price' in r.status and (r.persist_lo_ms, r.persist_hi_ms, r.persist_class) == (T0 + 1060, T0 + 4060, C.OBSERVED)
               and (r.post_class, r.queue_class, r.removal_class) == (C.UNRESOLVED,) * 3)
            ok(f'hs {s} identity conflict: no span ({ri.status}, {ri.persist_lo_ms})', 'identity_conflict' in ri.status
               and (ri.persist_lo_ms, ri.persist_hi_ms, ri.persist_class) == (-1, -1, C.UNRESOLVED))
    except Exception as e:  # noqa: BLE001
        bad.append(f'hs stopped ({e!r})'[:300])

    # ---- tw (posting_capacity_bound, 20261024_3697902 ask 62): the venue shows some orders 0.01 short (10 -> 9.99) and the shortfalls add
    #      up (279.98 for 10 + 145 + 125); H3 allows TOL_SH per order accounted ----
    try:
        # tp2: wZ (125, shown 124.99) rests from 500 beside wA (11 then 10, shown 10.99 / 9.99); at 1100..7000 shown 134.98 - wA's 10 = 124.98:
        #      one 0.01 tolerance would bound wZ's posting at 5000 (hard) / 7000 (hi clock) although it rests from 500; the bound is 400
        m = Mk(144)
        m.frames(GRID, lambda tr: {('bid', 40): 10.99 if tr < 500 else (135.98 if tr < 1100 else (134.98 if tr < 7100 else (124.99 if tr < 8100 else 0.)))})
        m.rec(1060, [('wA', 'MWA', 'bid', 40, 1.)], tk='T1'); m.rec(7060, [('wA', 'MWA', 'bid', 40, 10.)], tk='T2')
        m.rec(8060, [('wZ', 'MWZ', 'bid', 40, 125.)], tk='T3')
        f, o, lk, s_ = run(m)
        ok(f'tp2 two short-shown orders give no false posting bound ({[int(O(o, s, "wZ").post_lo_ms - T0) for s in S_ALL]}, {O(o, "S_FIFO", "wZ").post_class})',
           all((O(o, s, 'wZ').post_lo_ms, O(o, s, 'wZ').post_method) == (T0 + 400, 'capacity') for s in S_ALL)
           and O(o, 'S_FIFO', 'wZ').post_class == C.CONSTRAINED and not o.status.str.contains('capacity').any())
        # tv2: vA (11, shown 10.99) and vB (6, shown 5.99) both proven resting at bid 41 from 1300: needs 10 + 5 = 15 vs shown 14.98 is no overfill
        m = Mk(145)
        m.frames(GRID, lambda tr: {('bid', 41): 16.98 if tr < 1100 else (15.98 if tr < 1300 else (14.98 if tr < 7100 else (4.99 if tr < 7300 else 0.)))})
        m.rec(1060, [('vA', 'MVA', 'bid', 41, 1.)], tk='T1'); m.rec(1260, [('vB', 'MVB', 'bid', 41, 1.)], tk='T2')
        m.rec(7060, [('vA', 'MVA', 'bid', 41, 10.)], tk='T3'); m.rec(7260, [('vB', 'MVB', 'bid', 41, 5.)], tk='T4')
        f, o, lk, s_ = run(m)
        ok(f'tv2 two short-shown orders are no H3 contradiction ({sorted(set(o.status))}, {s_["capacity"]})',
           not o.status.str.contains('capacity').any() and s_['capacity']['orders_contradicted'] == 0)
        # tq2: a proven-resting order needing < 0.01 sh (tA 0.004) must not loosen the single-order bound: shown 4.989 < 5 - 0.01 proves wQ
        #      absent, while the joint room 4.989 - 0.004 = 4.985 is above 5 - 0.02. a: wQ rests from 5600 (before its hard window) -> every
        #      bound 5500, the hard one included (constrained); b: wQ rests from 7750 (inside it) -> the hi clock bounds at 7700, hard at 6000
        for nm_, mid_, tk_, lastA, postQ in (('tq2a', 146, 63, 7560, 5600), ('tq2b', 147, 64, 8160, 7750)):
            m = Mk(mid_)
            bk = (lambda la, pq: lambda tr: 4.991 if tr < 1100 else (4.989 if tr < pq else (9.979 if tr < 8100 else (4.989 if tr < la + 40 else 4.985)))
                  if la > 8100 else (4.991 if tr < 1100 else (4.989 if tr < pq else (9.979 if tr < la + 40 else (9.975 if tr < 8100 else 4.985)))))(lastA, postQ)
            m.frames(GRID, lambda tr, bk=bk, tk_=tk_: {('ask', tk_): bk(tr)})
            m.rec(1060, [('tA', 'MTA', 'ask', tk_, .002)], tk='T1'); m.rec(lastA, [('tA', 'MTA', 'ask', tk_, .004)], tk='T2')
            m.rec(8060, [('wQ', 'MWQ', 'ask', tk_, 5.)], tk='T3')
            f, o, lk, s_ = run(m)
            pl = [int(O(o, s, 'wQ').post_lo_ms - T0) for s in S_ALL]
            if nm_ == 'tq2a':
                ok(f'tq2a the single-order bound survives a tiny proven order ({pl}, {O(o, "S_FIFO", "wQ").post_class})',
                   pl == [5500, 5500, 5500] and O(o, 'S_FIFO', 'wQ').post_class == C.CONSTRAINED)
            else:
                ok(f'tq2b the scenario single-order bound ({pl}, {O(o, "S_FIFO", "wQ").post_class})', pl[0] == 7700 and pl[2] == 7700
                   and O(o, 'S_FIFO', 'wQ').post_class == C.SENSITIVE)
    except Exception as e:  # noqa: BLE001
        bad.append(f'tw stopped ({e!r})'[:300])

    # ---- h: posting lower bound from capacity ----
    try:
        m = Mk(111)
        m.frames(GRID, lambda tr: {('bid', 35): 4 if tr <= 600 else (14 if tr <= 4000 else 4)}, pend=lambda tr: (win((4100, 4500, .35, 3.5))(tr), []))
        m.rec(4060, [('hH', 'MH', 'bid', 35, 10.)], tk='T1')
        f, o, lk, s_ = run(m)
        for s in S_ALL:
            r = O(o, s, 'hH')
            ok(f'h {s} posting after 600 ({r.post_lo_ms - T0}, {r.post_method})', (r.post_lo_ms, r.post_hi_ms, r.post_method) == (T0 + 600, T0 + 4060, 'capacity'))
        ok('h constrained', O(o, 'S_FIFO', 'hH').post_class == C.CONSTRAINED and O(o, 'S_FIFO', 'hH').ahead_hi == 4.)
        # hu / hd: the 600 cell is unknown (an off_grid level status / the cell written twice under dup_level): never 0, never 4 -> bound from 500
        for nm_, mid_, kw in (('hu', 122, dict(flags={600: 'off_grid'}, lvstat={(600, 'bid', 35): 'off_grid'})),
                              ('hd', 123, dict(flags={600: 'dup_level'}, dups={(600, 'bid', 35)}))):
            m = Mk(mid_)
            m.frames(GRID, lambda tr: {('bid', 35): 4 if tr <= 600 else (14 if tr <= 4000 else 4)}, pend=lambda tr: (win((4100, 4500, .35, 3.5))(tr), []), **kw)
            m.rec(4060, [('hU', 'MH', 'bid', 35, 10.)], tk='T1')
            f, o, lk, s_ = run(m)
            ok(f'{nm_} an unknown cell is no bound ({[int(O(o, s, "hU").post_lo_ms - T0) for s in S_ALL]}, {s_["inputs"]["unknown_cells"]})',
               all(O(o, s, 'hU').post_lo_ms == T0 + 500 for s in S_ALL) and s_['inputs']['unknown_cells'] == 1 and s_['inputs']['frames_excluded'] == {})
    except Exception as e:  # noqa: BLE001
        bad.append(f'h stopped ({e!r})'[:300])

    # ---- i: identity conflicts ----
    try:
        m = Mk(112); m.frames(GRID, lambda tr: {})
        m.rec(1060, [('hI', 'MI', 'bid', 20, 1.)], tk='T1'); m.rec(2060, [('hI', 'MI', 'bid', 21, 1.)], tk='T2')
        m.rec(3060, [('hJ', 'MJ', 'bid', 22, 1.)], tk='T3'); m.rec(4060, [('hK', 'MK', 'ask', 70, 1.)], tk='OTHER', tk_hash='hJ')
        f, o, lk, s_ = run(m)
        for h in ('hI', 'hJ'):
            rows = o[o.mk_hash == h]
            ok(f'i {h} identity_conflict in every row', len(rows) == 3 and rows.status.str.contains('identity_conflict').all())
            r = rows.iloc[0]
            ok(f'i {h} classes unresolved', (r.size_class, r.post_class, r.persist_class, r.queue_class, r.removal_class) == (C.OBSERVED,) + (C.UNRESOLVED,) * 4)
        ok('i hI fills flagged', f[f.mk_hash == 'hI'].status.str.contains('identity_conflict').all())
        ok('i hJ seen as a taker', O(o, 'S_FIFO', 'hJ').x_as_taker == 1 and 'identity_conflict' not in O(o, 'S_FIFO', 'hK').status)
    except Exception as e:  # noqa: BLE001
        bad.append(f'i stopped ({e!r})'[:300])

    # ---- t: book-drop timing, the first candidate ----
    try:
        m = Mk(113)
        m.frames(GRID, lambda tr: {('ask', 65): 20 if tr < 1500 else (14 if tr < 3100 else 9)})
        m.rec(3060, [('hT', 'MT', 'ask', 65, 5.)], tk='T1')
        f, o, lk, s_ = run(m)
        ft = Fh(f, 'hT')
        ok(f't first drop (an unrelated 6-sh drop at 1500) taken ({ft.lo_ms - T0}, {ft.t_method}, {ft.x_drop_fid})',
           (ft.lo_ms, ft.t_method, ft.x_drop_fid, ft.f_before, ft.f_after) == (T0 + 1400, 'book_drop', 1015, 1014, 1031))
        # queue frame = f_before (1400: 20 shown - 5) under both price-time clocks, not the hi clock's frame 3000 (14 - 5)
        ok(f't queue at f_before ({O(o, "S_FIFO", "hT").ahead_hi}, {O(o, "S_FIFO_LO", "hT").ahead_hi})',
           O(o, 'S_FIFO', 'hT').ahead_hi == 15. and O(o, 'S_FIFO_LO', 'hT').ahead_hi == 15.)
        # tu: the cell is unknown at 1500 (off_grid), so the fill may be reflected there: the conservative lo is 1400, not the visible drop at 3100
        m = Mk(124)
        m.frames(GRID, lambda tr: {('ask', 65): 20 if tr < 1500 else (15 if tr < 3100 else 10)}, flags={1500: 'off_grid'}, lvstat={(1500, 'ask', 65): 'off_grid'})
        m.rec(3060, [('hV', 'MV', 'ask', 65, 5.)], tk='T1')
        f, o, lk, s_ = run(m)
        fv = Fh(f, 'hV')
        ok(f'tu an unknown cell is a possible drop ({fv.lo_ms - T0}, {fv.t_method}, {fv.f_before}, {fv.status})',
           (fv.lo_ms, fv.t_method, fv.f_before) == (T0 + 1400, 'book_drop', 1014) and 'drop_at_unknown_cell' in fv.status)
        # tq: no drop at all (refilled book), two pending increments that could include the record: the first one bounds lo
        m = Mk(129)
        m.frames(GRID, lambda tr: {('ask', 67): 10}, pend=lambda tr: ([], win((1100, 1500, .67, .67), (2100, 2500, .67, .67))(tr)))
        m.rec(2060, [('hQ2', 'MQ2', 'ask', 67, 1.)], tk='T1')
        f, o, lk, s_ = run(m)
        ok(f'tq the first pending candidate ({Fh(f, "hQ2").lo_ms - T0}, {Fh(f, "hQ2").t_method})', (Fh(f, 'hQ2').lo_ms, Fh(f, 'hQ2').t_method) == (T0 + 1000, 'pend_inc'))
        # tp: an exact pending booking at 3100 but a drop that could be the fill at 1500: the earlier of the two is the conservative lo
        m = Mk(125)
        m.frames(GRID, lambda tr: {('ask', 66): 10 if tr < 1500 else (6 if tr < 3100 else 3)}, pend=lambda tr: ([], win((3100, 3500, .66, 1.98))(tr)))
        m.rec(3060, [('hP', 'MP', 'ask', 66, 3.)], tk='T1')
        f, o, lk, s_ = run(m)
        fp = Fh(f, 'hP')
        ok(f'tp the earlier evidence of either kind ({fp.lo_ms - T0}, {fp.t_method}, exact {fp.x_lo_exact_ms - T0})',
           (fp.lo_ms, fp.t_method, fp.x_lo_exact_ms) == (T0 + 1400, 'book_drop', T0 + 3000))
    except Exception as e:  # noqa: BLE001
        bad.append(f't stopped ({e!r})'[:300])

    # ---- c2: equal-value repeated fills: the earliest matching pending increment (an earlier record's) is kept: lo only loosens ----
    try:
        m = Mk(115)
        m.frames(GRID, lambda tr: {('bid', 40): 10 if tr < 1100 else (8 if tr < 2100 else 6)},
                 pend=lambda tr: (win((1100, 1500, .4, .8), (2100, 2500, .4, .8))(tr), []))
        m.rec(1060, [('hQ', 'MQ', 'bid', 40, 2.)], tk='T1'); m.rec(2060, [('hQ', 'MQ', 'bid', 40, 2.)], tk='T2')
        f, o, lk, s_ = run(m)
        ok(f'c2 both lows at the first increment ({[int(x - T0) for x in f.sort_values("hi_ms").lo_ms]})',
           [int(x - T0) for x in f.sort_values('hi_ms').lo_ms] == [1000, 1000])
        r = O(o, 'S_FIFO', 'hQ')
        ok(f'c2 empty certain span -> persistence unresolved ({r.persist_lo_ms}, {r.persist_hi_ms})', (r.persist_class, r.persist_lo_ms, r.persist_hi_ms) == (C.UNRESOLVED, -1, -1))
    except Exception as e:  # noqa: BLE001
        bad.append(f'c2 stopped ({e!r})'[:300])

    # ---- k: fifo_through posting bound (no frames, so capacity is silent); a same-price record does not bound it ----
    try:
        m = Mk(116)
        m.rec(1060, [('k1', 'MK1', 'ask', 51, 1.), ('k2', 'MK2', 'ask', 53, 1.)], tk='T1')     # trades through 52
        m.rec(2000, [('k3', 'MK3', 'ask', 52, 1.)], tk='T2')                                   # at 52: kO could be behind k3
        m.rec(3060, [('kO', 'MKO', 'ask', 52, 2.)], tk='T3')
        f, o, lk, s_ = run(m)
        a_, b_, c_ = O(o, 'S_FIFO', 'kO'), O(o, 'S_FIFO_LO', 'kO'), O(o, 'S_NOPRIO', 'kO')
        ok(f'k posting after the trade-through ({a_.post_lo_ms - T0}, {a_.post_method}, {b_.post_lo_ms - T0}, {c_.post_lo_ms})',
           (a_.post_lo_ms, a_.post_method, b_.post_lo_ms, c_.post_lo_ms, a_.post_class) == (T0 + 1060, 'fifo_through', T0 + 1060 - C.MAX_LAG_MS, -1, C.SENSITIVE))
    except Exception as e:  # noqa: BLE001
        bad.append(f'k stopped ({e!r})'[:300])

    # ---- s: fifo_front skips a record whose taker is the order's own signer (self-trade prevention possible) ----
    try:
        m = Mk(118)
        m.rec(1060, [('sO', 'MS', 'bid', 40, 1.)], tk='T1')
        m.rec(2060, [('s2', 'MS2', 'bid', 40, 1.)], tk='MS')
        m.rec(3060, [('s3', 'MS3', 'bid', 40, 1.)], tk='T3')
        f, o, lk, s_ = run(m)
        r = O(o, 'S_FIFO', 'sO')
        ok(f's front bound at the first other-signer record ({r.removal_hi_ms - T0}, {r.removal_method}, {r.removal_set})',
           (r.removal_hi_ms, r.removal_method, r.removal_set) == (T0 + 3060, 'fifo_front', 'cancelled_or_expired|end_censored|exhausted|missing_evidence'))
    except Exception as e:  # noqa: BLE001
        bad.append(f's stopped ({e!r})'[:300])

    # ---- n: capacity-tight exhaustion through another proven-resting order (nB accounts for all shown depth) ----
    try:
        m = Mk(117); PM['n'] = m
        m.frames(GRID, lambda tr: {('bid', 45): 7 if tr < 1100 else (4 if tr < 4100 else 0)},
                 pend=lambda tr: (win((1100, 1500, .45, 1.35), (4100, 4500, .45, 1.8))(tr), []))
        m.rec(1060, [('nA', 'MNA', 'bid', 45, 1.), ('nB', 'MNB', 'bid', 45, 2.)], tk='T1'); m.rec(4060, [('nB', 'MNB', 'bid', 45, 4.)], tk='T2')
        f, o, lk, s_ = run(m)
        for s in S_ALL:
            r = O(o, s, 'nB')    # certain presence 11..20 shows 4 = its later fill: slack 0 -> GONE by 4100, exhausted OR cancelled (two-history rule)
            ok(f'n {s} nB capacity_tight keeps both explanations ({r.size_exact}, {r.x_size_ub}, {r.removal_set}, {r.removal_method}, {r.removal_hi_ms - T0})',
               math.isnan(r.size_exact) and r.x_size_ub == 6. and 'capacity_tight' in r.removal_method
               and (r.removal_set, r.removal_lo_ms, r.removal_hi_ms) == ('cancelled_or_expired|exhausted', T0 + 2060, T0 + 4100))
        ok('n nB constrained', (O(o, 'S_FIFO', 'nB').size_class, O(o, 'S_FIFO', 'nB').removal_class) == (C.CONSTRAINED, C.CONSTRAINED))
        ok(f'n nB exact queue [1, 7 - 6] ({O(o, "S_FIFO", "nB").ahead_lo}, {O(o, "S_FIFO", "nB").ahead_hi})',
           (O(o, 'S_FIFO', 'nB').ahead_lo, O(o, 'S_FIFO', 'nB').ahead_hi) == (1., 1.))
        r = O(o, 'S_NOPRIO', 'nA')    # after its fill the shown 4 is all nB's: no room -> gone at 1100
        ok(f'n S_NOPRIO nA gone by accounting ({r.removal_set}, {r.removal_hi_ms - T0}, {r.removal_method})',
           (r.removal_set, r.removal_hi_ms, r.removal_method) == ('cancelled_or_expired|exhausted', T0 + 1100, 'book_gone'))
        r = O(o, 'S_FIFO', 'nA')
        ok(f'n S_FIFO nA exhausted by rule 2 (nB filled again) ({r.removal_set}, {r.size_exact}, {r.removal_method})',
           (r.removal_set, r.size_exact, r.removal_method) == ('exhausted', 1., 'fifo_time'))
        ok('n T2 survivor', s_['priority_tests']['S_FIFO']['T2_time']['survivor_groups'] == 1)
        # cl: both clocks trust cB's evidence windows, so its later 4 sh fill the shown 4 and cA is gone at 1100 in EVERY scenario; the
        #     evidence-free logic proves it only at 2600 -> the scenarios agree with each other but not with H1-H3: assumption_sensitive
        m = Mk(130)
        m.frames(GRID, lambda tr: {('bid', 44): 7 if tr < 1100 else (4 if tr < 2600 else 0)},
                 pend=lambda tr: (win((1100, 1500, .44, 1.32), (2600, 3000, .44, 1.76))(tr), []))
        m.rec(1060, [('cA', 'MCA', 'bid', 44, 1.)], tk='T1'); m.rec(1060, [('cB', 'MCB', 'bid', 44, 2.)], tk='T2')
        m.rec(2560, [('cB', 'MCB', 'bid', 44, 4.)], tk='T3')
        f, o, lk, s_ = run(m)
        ok(f'cl scenarios agree, H1-H3 does not ({[int(O(o, s, "cA").removal_hi_ms - T0) for s in S_ALL]}, {O(o, "S_FIFO", "cA").removal_class})',
           all(O(o, s, 'cA').removal_hi_ms == T0 + 1100 for s in S_ALL) and O(o, 'S_FIFO', 'cA').removal_class == C.SENSITIVE)
    except Exception as e:  # noqa: BLE001
        bad.append(f'n stopped ({e!r})'[:300])

    # ---- p: outputs are invariant to the makers[] listing (mk_idx reversed within every record) ----
    try:
        for nm_, m in PM.items():
            L, R, Fr, Lv = m.dfs(); L2 = L.assign(mk_idx=L.n_makers - 1 - L.mk_idx)
            (f1, o1, _, _), (f2, o2, _, _) = X.run_market(L, R, Fr, Lv, m.market()), X.run_market(L2, R, Fr, Lv, m.market())
            nf = lambda f_: f_.drop(columns=['mk_idx']).sort_values(['sid', 'mk_hash']).reset_index(drop=True)  # noqa: E731
            no = lambda o_: o_.sort_values(['scenario', 'mk_hash']).reset_index(drop=True)  # noqa: E731
            ok(f'p {nm_}: identical under a reversed makers[] listing', nf(f1).equals(nf(f2)) and no(o1).equals(no(o2)))
        ok(f'p covers the amendment cases ({sorted(PM)})', set(PM) >= {'a', 'g', 'g2', 't1', 'n'})
    except Exception as e:  # noqa: BLE001
        bad.append(f'p stopped ({e!r})'[:300])

    # ---- R: refusals (the base passes; one planted defect each; the reason asserted) ----
    try:
        m = Mk(101)
        m.frames(GRID, sim({('bid', 50): [[(-1, 5.), (1060, 0.)], [(-1, 45.), (2060, 35.)]]}),
                 pend=lambda tr: (win((1100, 1500, .5, 2.5), (2100, 2500, .5, 5.))(tr), []))
        m.rec(1060, [('h7', 'M7', 'bid', 50, 5.)], tk='TA'); m.rec(2060, [('h9', 'M9', 'bid', 50, 10.)], tk='TB')
        L, R, Fr, Lv = m.dfs(); mk = m.market()
        ok('R base passes', isinstance(X.run_market(L, R, Fr, Lv, mk), tuple))
        cases = [
            ('legs missing columns', lambda: X.run_market(L.drop(columns=['mk_quote']), R, Fr, Lv, mk)),
            ('legs hold a market other than', lambda: X.run_market(L.assign(mid=np.where(L.index == 0, 999, L.mid)), R, Fr, Lv, mk)),
            ('duplicate leg key', lambda: X.run_market(pd.concat([L, L.iloc[:1]], ignore_index=True), R, Fr, Lv, mk)),
            ('disagree with yes_space', lambda: X.run_market(L.assign(y_side=np.where(L.index == 0, 'ask', L.y_side)), R, Fr, Lv, mk)),
            ('absent from records', lambda: X.run_market(L, R.iloc[1:], Fr, Lv, mk)),
            ('disagree on settle_ms', lambda: X.run_market(L, R.assign(settle_ms=R.settle_ms + np.where(R.index == 0, 1, 0)), Fr, Lv, mk)),
            ('duplicate record sid', lambda: X.run_market(L, pd.concat([R, R.iloc[:1]], ignore_index=True), Fr, Lv, mk)),
            ('unknown frame fid', lambda: X.run_market(L, R, Fr, pd.concat([Lv, Lv.iloc[:1].assign(fid=999999)], ignore_index=True), mk)),
            ('duplicate frame fid', lambda: X.run_market(L, R, pd.concat([Fr, Fr.iloc[:1]], ignore_index=True), Lv, mk)),
            ('pend text is not a JSON', lambda: X.run_market(L, R, Fr.assign(pend_bids=np.where(Fr.index == 3, '{"a": 1}', Fr.pend_bids)), Lv, mk)),
            ('we <= ws', lambda: X.run_market(L, R, Fr, Lv, dict(mk, we=mk['ws']))),
            ('without the x_ prefix', lambda: X._conform(pd.DataFrame({**{c: pd.Series(dtype=d) for c, d in C.SCHEMAS['fills'].items()}, 'extra': []}), 'fills')),
            ('self-guard', lambda: X.guard('selftest', limit=0.001)),
            ('not a consumed day', lambda: X.market_cli('20261099', 101, out='/nonexistent_selftest_root')),
        ]
        for word, fn in cases:
            res = refuses(word, fn)
            ok(f'R refuses: {word} ({res})', res is True)
    except Exception as e:  # noqa: BLE001
        bad.append(f'R stopped ({e!r})'[:300])

    # ---- V: venue oddities are statuses (rows kept, counted), never refusals or silent drops ----
    try:
        m = Mk(114)
        m.frames(GRID, lambda tr: {('ask', 52): 15 if tr < 1100 else 5}, pend=lambda tr: ([], win((1100, 1500, .52, 5.2))(tr)),
                 flags={300: 'dup_level', 400: 'bad_frame', 800: 'dup_level'}, dups={(300, 'ask', 52)})
        m.rec(1060, [('v1', 'MV1', 'ask', 52, 4.), ('v2', 'MV2', 'ask', 52, 6.)], tk='TV'); m.rec(2060, [('v3', 'MV3', 'ask', 52, 1.)], tk='TW')
        L, R, Fr, Lv = m.dfs()
        bl = L.index[L.mk_hash == 'v2'][0]                           # an extract bad_leg row: no YES side / price, indexSet -1
        L.loc[bl, ['y_side', 'y_ptick', 'y_price', 'mk_index_set', 'status']] = ['', -1, NAN, -1, 'bad_leg']
        L.loc[L.mk_hash == 'v3', 'shares'] = 0.                      # a zero-share leg
        Lv.loc[Lv.fid == Fr.fid[Fr.upd_ms == T0 + 900].iloc[0], 'shares'] = -1.     # a bad level row
        Fr.loc[Fr.upd_ms == T0 + 700, 'pend_asks'] = '[[1.5,2.0]]'  # a pending entry out of range
        Lv = Lv[Lv.fid != Fr.fid[Fr.upd_ms == T0 + 1200].iloc[0]]   # a wiped (level-less) book inside [ws, we]
        f, o, lk, s_ = X.run_market(L, R, Fr, Lv, m.market())
        ok('V the run completes with every row', len(f) == 3 and len(o) == 9)
        fv2 = f[f.mk_hash == 'v2'].iloc[0]
        ok(f'V bad leg kept as a status ({fv2.status})', 'unplaceable' in fv2.status and 'leg_bad_leg' in fv2.status and fv2.y_ptick == -1)
        r = O(o, 'S_FIFO', 'v2')
        ok(f'V bad-leg order invalid ({r.status})', 'unplaceable_leg' in r.status and r.removal_class == C.UNRESOLVED and r.y_side == '')
        fv1 = f[f.mk_hash == 'v1'].iloc[0]
        ok(f'V no price-time inference in a record with an unplaceable leg ({fv1.status})',
           not fv1.x_exh_price and fv1.x_worst_k == 0 and 'record_unplaceable_leg' in fv1.status and math.isnan(O(o, 'S_FIFO', 'v1').size_exact))
        ok('V zero-share leg unplaceable', 'unplaceable' in f[f.mk_hash == 'v3'].status.iloc[0])
        ok(f'V frames excluded by flag / pending range / bad level / unlocated dup_level ({s_["inputs"]["frames_excluded"]})',
           s_['inputs']['frames_excluded'] == {'bad_frame': 1, 'bad_level': 1, 'dup_level_unlocated': 1, 'empty_book_in_window': 1, 'pend_out_of_range': 1}
           and s_['inputs']['frames_usable'] == 91 and s_['inputs']['unknown_cells'] == 1)
        ok(f'V counts ({s_["inputs"]["unplaceable_legs"]})', s_['inputs']['unplaceable_legs'] == 2 and sum(s_['fills']['status'].values()) == 3)
        # r: settle_ms missing, hi from a receipt 90 ms after the match: no lower end at all (the receipt may follow the match by seconds)
        m = Mk(127)
        m.frames(GRID, lambda tr: {('bid', 48): 6 if tr < 1100 else 4}, pend=lambda tr: (win((1100, 1500, .48, .96))(tr), []))
        m.rec(1060, [('hR2', 'MR2', 'bid', 48, 2.)], tk='T1')
        L, R, Fr, Lv = m.dfs()
        for D_ in (L, R):
            D_['settle_ms'] = -1; D_['first_recv_ns'] = (T0 + 1150) * 10 ** 6
        f, o, lk, s_ = X.run_market(L, R, Fr, Lv, m.market())
        fr_ = f.iloc[0]
        ok(f'r receipt-hi fill: no lower end ({fr_.lo_ms}, {fr_.x_lo_hard_ms}, {fr_.t_method}, {fr_.status})',
           (fr_.hi_ms, fr_.lo_ms, fr_.x_lo_hard_ms, fr_.t_method) == (T0 + 1150, -1, -1, 'none') and {'hi_from_receipt', 'fill_lo_unbounded'} <= set(fr_.status.split('+')))
    except Exception as e:  # noqa: BLE001
        bad.append(f'V stopped ({e!r})'[:300])

    # ---- CLI rehearsal: the production read / run / write-once path on a temporary root ----
    tmp = Path(tempfile.mkdtemp(prefix='mt_life_selftest_'))
    try:
        mA = Mk(101)
        mA.frames(GRID, sim({('bid', 50): [[(-1, 5.), (1060, 0.)], [(-1, 45.), (2060, 35.)]]}),
                  pend=lambda tr: (win((1100, 1500, .5, 2.5), (2100, 2500, .5, 5.))(tr), []))
        mA.rec(1060, [('h7', 'M7', 'bid', 50, 5.)], tk='TA'); mA.rec(2060, [('h9', 'M9', 'bid', 50, 10.)], tk='TB')
        mN = Mk(102)                                                  # no depth files: every window unbounded, every removal missing evidence
        mN.rec(1060, [('hA', 'MB', 'ask', 60, 3.)], tk='T1'); mN.rec(3060, [('hB', 'MB', 'ask', 60, 2.)], tk='T2')
        dA, dN = mA.dfs(), mN.dfs()
        pd.concat([dA[0], dN[0]], ignore_index=True).to_parquet(tmp / f'legs_{DAY}.parquet', index=False)
        pd.concat([dA[1], dN[1]], ignore_index=True).to_parquet(tmp / f'records_{DAY}.parquet', index=False)
        (tmp / 'depth').mkdir()
        dA[2].to_parquet(tmp / 'depth' / f'{DAY}_101_frames.parquet', index=False); dA[3].to_parquet(tmp / 'depth' / f'{DAY}_101_levels.parquet', index=False)
        pd.DataFrame([dict(mA.market(), day=DAY, depth=1), dict(mN.market(), day=DAY, depth=0, status='no_frames')]).to_parquet(
            tmp / f'markets_{DAY}.parquet', index=False)
        sm = X.market_cli(DAY, '101', out=tmp)
        life = tmp / 'life'
        got = {nm: pd.read_parquet(life / f'{DAY}_101_{nm}.parquet') for nm in ('fills', 'orders', 'x_links')}
        js = json.loads((life / f'{DAY}_101.json').read_text())
        for nm in ('fills', 'orders'):
            dt_ = {c: str(got[nm][c].dtype) for c in C.SCHEMAS[nm]}
            ok(f'CLI {nm} dtypes read back exactly ({ {c: d for c, d in dt_.items() if d != C.SCHEMAS[nm][c]} })', dt_ == C.SCHEMAS[nm])
        direct = X.run_market(*dA, mA.market())
        ok('CLI output equals the pure function', direct[0].reset_index(drop=True).equals(got['fills'])
           and direct[1].reset_index(drop=True).equals(got['orders']))
        ok('CLI json: summary + provenance', js['mid'] == 101 and js['provenance']['day'] == DAY and js['provenance']['depth'] is True
           and js['fills']['n'] == 2 and len(js['provenance']['module_sha256']) == 64)
        res = refuses('write-once', X.market_cli, DAY, '101', out=tmp)
        ok(f'CLI refuses to overwrite ({res})', res is True)
        sn = X.market_cli(DAY, 102, out=tmp)
        on = pd.read_parquet(life / f'{DAY}_102_orders.parquet'); fn_ = pd.read_parquet(life / f'{DAY}_102_fills.parquet')
        ok('CLI no-depth market: windows unbounded, missing evidence', sn['provenance']['depth'] is False and (fn_.lo_ms == -1).all()
           and on.removal_set.str.contains('missing_evidence').all() and sn['inputs']['frames_usable'] == 0 and sn['market_status'] == 'no_frames')
        ln = pd.read_parquet(life / f'{DAY}_102_x_links.parquet')
        ok('CLI no-depth links still descriptive', ln.kind.tolist() == ['refill'] and ln.label.tolist() == ['descriptive'])
    except Exception as e:  # noqa: BLE001
        bad.append(f'CLI stopped ({e!r})'[:300])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if bad:
        if quiet:
            print(f'   {len(bad)} of {n[0]} checks fail, first: {bad[0][:150]}')
        else:
            print(FAIL, f'{len(bad)} of {n[0]} checks')
            for b in bad:
                print(' -', b)
        raise SystemExit(1)
    if not quiet:
        print(f'LIFECYCLE SELFTEST PASSED: {n[0]} checks; VmHWM {X.vmhwm_mib():.0f} MiB')
    return n[0]


MUTANTS = (  # (name, exact text in pf_mt_lifecycle.py, replacement): every planted defect must fail at least one check
    ('pending evidence: last candidate', "E_p = a0 + int(j[0])", "E_p = a0 + int(j[-1])"),
    ('drop evidence: last candidate', "E_d = a0 + int(j[0])", "E_d = a0 + int(j[-1])"),
    ('pending first, an earlier drop ignored', "cand = [x for x in (E_p, E_d) if x >= 1]", "cand = [E_p] if E_p >= 1 else ([E_d] if E_d >= 1 else [])"),
    ('exact-value-first evidence (the reviewed rule)', "E = min(cand) if cand else -1", "E = (E_x if E_x >= 1 else E_d) if (E_x >= 1 or E_d >= 1) else -1"),
    ('lo at the evidence frame itself', "fb = E - 1 if E >= 1 else -1", "fb = E if E >= 1 else -1"),
    ('unknown cell not a drop candidate', "| np.isnan(dd))", ")"),
    ('receipt-hi fills keep the evidence (B2)', "if F >= 2 and len(gi) and st[ii[0]] > 0:", "if F >= 2 and len(gi):"),
    ('hard logic on the evidence lo (B2 reverted)', "fb, fa, hi = X['fb_h'], X['fa'], X['hi']", "fb, fa, hi = X['fb'], X['fa'], X['hi']"),
    ('removal_lo / persistence from the evidence lo (B2 reverted)', "lb = X['lo_h'][ii][X['lo_h'][ii] >= 0]", "lb = lo[ii][lo[ii] >= 0]"),
    ('hard H3 contradiction silenced', "O['flags'][o].add('capacity_contradiction')", "pass"),
    ('a contradicted order keeps its hard capacity claims (B1)', "if not nv:                                   # a contradicted order",
     "if True:                                     # a contradicted order"),
    ('a contradicted scenario keeps its capacity claims (B1)', "cap_ok = nv == 0 and not Hd['viol'][o]", "cap_ok = not Hd['viol'][o]"),
    ('violated cells known to the others (B1)', "    r[vm] = np.nan\n", "    pass\n"),
    ('no class demotion on a contradiction (B1)', "if Hd['viol'][o] or any(S[s]['clock_viol'][o] for s in SCEN):", "if False:"),
    ('persistence observed with an empty span (B3)', "C.OBSERVED if (O['n'][o] >= 2 and O['p_lo'][o] >= 0) else C.UNRESOLVED",
     "C.OBSERVED if O['n'][o] >= 2 else C.UNRESOLVED"),
    ('rule 1 by the makers[] listing (pre-amendment)', "X['exh_p'][ii] = ~worst", "X['exh_p'][ii] = np.arange(len(ii)) < len(ii) - 1"),
    ('rule 1 includes the worst price', "X['exh_p'][ii] = ~worst", "X['exh_p'][ii] = True"),
    ('rule 2 survivor gets no ahead_lo', "alo[A[0]] = float(q[oth].sum())", "alo[A[0]] = 0."),
    ('rule 2 exhausts the survivor too', "exh[oth] = True", "exh[mem] = True"),
    ('T1 violations not counted', "T1_violations=int((u1 & later).sum())", "T1_violations=0"),
    ('T1 excludes tied legs (fix4 reverted)', "u1 = exh_p & ~unord[oid]", "u1 = exh_p & ~unord[oid] & ~tie"),
    ('T1 violator still claimed exhausted', "exhL = len(xl) == 1 and not t1 and o not in tviol", "exhL = len(xl) == 1 and o not in tviol"),
    ('per-fill exhaustion kept for a T1 violator', "f_exh=exh & ~np.isin(oid, oid[exh_p & later]) & ~tconf[oid])", "f_exh=exh & ~tconf[oid])"),
    ('a tie voids the rule-2 group (fix4 reverted)', "            A = mem[later[mem]]", "            if tie[mem].any():\n                continue\n            A = mem[later[mem]]"),
    ('exhaustion only at the last-ranked fill (fix4 reverted)', "xl = fi[exh[fi] & (sqf == sqf.max())]", "xl = np.array([fL])[exh[[fL]]]"),
    ('a tie conflict claimed exhausted', "exhL = len(xl) == 1 and not t1", "exhL = len(xl) >= 1 and not t1"),
    ('S_NOPRIO queue NaN (fix4 reverted)', "            out['alo'][o] = 0.                     # no priority", "            out['alo'][o] = NAN                    # no priority"),
    ('queue class by a non-trivial value (fix4 reverted)', "cl['queue'][o] = C.UNRESOLVED if _same(qu) else C.SENSITIVE",
     "cl['queue'][o] = C.SENSITIVE if any((not np.isnan(x) and x > TOL_SH) or not np.isnan(y) for x, y in qu) else C.UNRESOLVED"),
    ('persistence only for valid orders (fix4 reverted)', "if not (fl & NO_PRESENCE) and len(ii) >= 2", "if O['valid'][o] and len(ii) >= 2"),
    ('invalid-order persistence unresolved (fix4 reverted)', "cl['size'][o], cl['persist'][o] = C.OBSERVED, span", "cl['size'][o], cl['persist'][o] = C.OBSERVED, C.UNRESOLVED"),
    ('persistence for an identity conflict', "NO_PRESENCE = frozenset(('identity_conflict', 'malformed_identity'))", "NO_PRESENCE = frozenset(('malformed_identity',))"),
    ('one display tolerance for any number of orders (fix4 reverted)', "return TOL_SH * np.maximum(n, 1) + 1e-9", "return TOL_SH + 1e-9 + 0 * n"),
    ('hard posting tolerance without the order itself', "own - _tol(1 + nn[:len(own), co])", "own - _tol(nn[:len(own), co])"),
    ('scenario posting tolerance without the order itself', "cum - _tol(1 + nn[:ao, co])", "cum - _tol(nn[:ao, co])"),
    ('hard posting without the single-order form', "| (shv < own - _tol(1))", "| False"),
    ('scenario posting without the single-order form', "| (np.where(vm[:ao, co], np.nan, SH[:ao, co]) < cum - _tol(1))", "| False"),
    ('T2 violators not flagged', "tviol |= set(oid[A].tolist())", "pass"),
    ('exhausted dropped unless proven', "rset = ({'exhausted', 'cancelled_or_expired'} |", "rset = ({'cancelled_or_expired'} |"),
    ('end_censored never kept', "({'end_censored'} if (rhi < 0 or beyond) else set())", "set()"),
    ('a fifo_front bound past the book drops end_censored', "beyond = rhi >= 0 and (F == 0 or rhi > t[-1])", "beyond = False"),
    ('out_of_order frames used', "USABLE_FRAME = frozenset(('dup', 'same_ms_changed', 'new_conn', 'gap', 'off_grid', 'dup_level'))",
     "USABLE_FRAME = frozenset(('dup', 'same_ms_changed', 'new_conn', 'gap', 'off_grid', 'dup_level', 'out_of_order'))"),
    ('gap-flagged frames excluded', "USABLE_FRAME = frozenset(('dup', 'same_ms_changed', 'new_conn', 'gap', 'off_grid', 'dup_level'))",
     "USABLE_FRAME = frozenset(('dup', 'same_ms_changed', 'new_conn', 'off_grid', 'dup_level'))"),
    ('unlocated dup_level / off_grid frame used', "if use[i] and flag in fl[i] and i not in located:", "if False:"),
    ('a wiped book inside the window used', "for i in np.flatnonzero(use & ~has_lv & (upd >= ws_ms) & (upd <= we_ms)):", "for i in []:"),
    ('unknown cell shown as 0', "shown[tuple(x[unk] for x in ix)] = np.nan", "shown[tuple(x[unk] for x in ix)] = 0."),
    ('absent price unknown instead of 0', "shown = np.zeros((F, 2, NT)); pnd", "shown = np.full((F, 2, NT), np.nan); pnd"),
    ('a gap adds no missing_evidence', "F == 0 or beyond or (g0 <= end_f and bool(gap[span].any() or np.isnan",
     "F == 0 or beyond or (g0 <= end_f and bool(False or np.isnan"),
    ('negative ahead_hi silently 0', "fl.add('ahead_hi_negative')", "out['ahi'][o] = 0."),
    ('classes: every scenario identical', "return all(eq(vals[0], v) for v in vals[1:])", "return True"),
    ('posting: earliest failing frame', "pcap = int(t[j[-1]]) if len(j) else -1", "pcap = int(t[j[0]]) if len(j) else -1"),
    ('persistence span from the min lo', "O['p_lo'][o], O['p_hi'][o] = O['first_hi'][o], int(lb.max())",
     "O['p_lo'][o], O['p_hi'][o] = O['first_hi'][o], int(lb.min())"),
    ('identity unchecked', "fl.add('identity_conflict')", "pass"),
    ('fifo_front searched from the start', "R_ = RS[sd]; i0 = int(np.searchsorted(R_['hi'], seq[fL], 'right'))", "R_ = RS[sd]; i0 = 0"),
    ('fifo_through includes the same price', "thr = (R_['w'][:i1] > p + 1e-9) if sd == 1 else (R_['w'][:i1] < p - 1e-9)",
     "thr = (R_['w'][:i1] >= p - 1e-9) if sd == 1 else (R_['w'][:i1] <= p + 1e-9)"),
    ('fifo_front ignores the taker signer', "jj = np.flatnonzero(reach & (R_['tk'][i0:] != g))", "jj = np.flatnonzero(reach)"),
    ('no unprinted-drop evidence', "if bo + 1 <= end_f and U[end_f + 1, co] - U[bo + 1, co] > 0:", "if False:"),
    ('room ignores the other orders', "    r = np.where(nd <= SH + _tol(nn), SH - nd, SH)\n", "    r = SH.copy()\n"),
    ('queue frame = the clock frame', "qb = np.where(fb >= 0, fb, pos)", "qb = pos"),
    ('lo clock = hi clock', " if loc else pos_hi", " if False else pos_hi"),
    ('book evidence claims exhaustion (integrator decision reverted)', "        if exhL:\n            rset = {'exhausted'}",
     "        if exhL or tight_s:\n            rset = {'exhausted'}"),
    ('removal_lo from the first fill', "O['rlo'][o] = int(lb.max()) if len(lb) else -1", "O['rlo'][o] = int(lb.min()) if len(lb) else -1"),
    ('size class ignores the scenario-free result', "sz = [hz] + [", "sz = ["),
    ('posting class ignores the scenario-free result', "pl = [int(Hd['post'][o])] + [", "pl = ["),
    ('removal class ignores the scenario-free result', "rv = [('|'.join(sorted(hset)), int(O['rlo'][o]), int(rw))] + [", "rv = ["),
)


def mutants(only=None, base_check=True):
    """Each MUTANTS entry (or those named in only) applied alone to a temporary copy of pf_mt_lifecycle.py must fail main() or the light
    simcheck (seeds 11, 12); the unmutated module must pass both (base_check)."""
    import pf_mt_lifecycle as X_
    src = (Path(__file__).resolve().parent / 'pf_mt_lifecycle.py').read_text()
    bad_t = [(nm, src.count(o)) for nm, o, _ in MUTANTS if src.count(o) != 1]
    C.need(not bad_t, f'mutant targets not unique in pf_mt_lifecycle.py: {bad_t}')
    C.need(only is None or set(only) <= {nm for nm, _, _ in MUTANTS}, f'unknown mutant in {only}')
    base = (main(X_, quiet=True), simcheck(X_, seeds=(11, 12), quiet=True))[0] if base_check else None
    res = []
    for name, old, new in [x for x in MUTANTS if only is None or x[0] in only]:
        C.need(src.count(old) == 1, f'mutant {name!r}: its target occurs {src.count(old)} times in pf_mt_lifecycle.py')
        d = Path(tempfile.mkdtemp(prefix='mt_life_mut_'))
        try:
            (d / 'pf_mt_lifecycle_mut.py').write_text(src.replace(old, new))
            spec = importlib.util.spec_from_file_location('pf_mt_lifecycle_mut', d / 'pf_mt_lifecycle_mut.py')
            M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
            print(f'mutant: {name}')
            try:
                main(M, quiet=True)                                          # the simulator only for what the hand cases let through
                simcheck(M, seeds=(11, 12), quiet=True)
                res.append((name, False))
            except SystemExit:
                res.append((name, True))
            except Exception as e:  # noqa: BLE001
                print(f'   stopped: {e!r}'[:160]); res.append((name, True))
        finally:
            shutil.rmtree(d, ignore_errors=True)
    surv = [nm for nm, k in res if not k]
    print(f'MUTANTS: base passes {base} checks; killed {len(res) - len(surv)} of {len(res)}' + (f'; SURVIVED: {surv}' if surv else ''))
    if surv:
        print(FAIL, 'mutant(s) survived')
        raise SystemExit(1)


def simulate(seed, mid, dur_ms=60_000):
    """A price-time-priority venue (the S_FIFO premise): makers post / cancel whole-cent orders (bids 44..49, asks 51..56), takers sweep
    up to 3 levels in price then time order; makers[] is listed in a RANDOM order (amendment 1); frames every 48-52 ms show the aggregate
    depth and the pending USD (booked at the match on the maker side / price, cleared 200-800 ms later, so bookings net); settle_ms =
    match + 0..1,000 ms, out of order (review B2: real creation lags have no finite bound); an exhausted maker re-posts at its price in the
    same instant half of the time (a same-frame refill masks the drop). -> (Mk, truth {hash: size, post, rm, cause, side, tick}, match time per sid, queue snapshot per fid)."""
    rng = np.random.default_rng(seed)
    m = Mk(mid); m.ws = T0 // 1000 - 5; m.we = (T0 + dur_ms) // 1000 + 1
    Q, TR, REC, SNAP, pend, clears = {}, {}, {}, {}, Counter(), []

    def frame(ft):
        keep = []
        for tc, cell, v in clears:
            if tc <= ft:
                pend[cell] -= v
            else:
                keep.append((tc, cell, v))
        clears[:] = keep
        fid = 1000 + len(m.Fr)
        pl = {sd: [[k / 100, round(u, 9)] for (s_, k), u in sorted(pend.items()) if s_ == sd and u > 1e-9] for sd in ('bid', 'ask')}
        m.Fr.append(dict(mid=mid, fid=fid, upd_ms=T0 + ft, recv_ns=(T0 + ft + 120) * 10 ** 6, conn='c0', order_count=0, pend_bids=json.dumps(pl['bid']),
                         pend_asks=json.dumps(pl['ask']), los_id='', los_side='', los_price=0., los_outcome='', status='ok'))
        for (sd, k), lst in sorted(Q.items()):
            tot = sum(r for _, r in lst)
            if tot > 1e-9:
                m.Lv.append(dict(mid=mid, fid=fid, side=sd, ptick=k, shares=round(tot, 9), status='ok', x_price=NAN))
        SNAP[fid] = {c: [(h, r) for h, r in lst] for c, lst in Q.items() if lst}
    t = ft = n = 0
    while True:
        t += 1 + int(rng.exponential(12))
        if t > dur_ms:
            break
        while ft < t:
            frame(ft); ft += 48 + int(rng.integers(0, 5))
        u = rng.random()
        if u < .45:
            sd = 'bid' if rng.random() < .5 else 'ask'; k = int(rng.integers(44, 50)) if sd == 'bid' else int(rng.integers(51, 57))
            n += 1; h = f'o{mid}_{n}'; size = float(rng.choice([1, 2, 3, 5, 8, 10, 20]))
            Q.setdefault((sd, k), []).append([h, size])
            TR[h] = dict(signer=f'M{int(rng.integers(0, 6))}', side=sd, tick=k, size=size, post=t, rm=None, cause='end_censored')
        elif u < .70:
            rest = [(c, i) for c, lst in Q.items() for i in range(len(lst))]
            if rest:
                c, i = rest[int(rng.integers(len(rest)))]; h, r = Q[c].pop(i); TR[h]['rm'], TR[h]['cause'] = t, 'cancelled_or_expired'
        else:
            sd = 'ask' if rng.random() < .5 else 'bid'
            cells = sorted((c for c in Q if c[0] == sd and Q[c]), key=lambda c: c[1], reverse=(sd == 'bid'))[:3]
            left, legs = float(rng.choice([1, 2, 3, 5, 8, 13])), []
            for c in cells:
                while Q[c] and left > 1e-9:
                    h, r = Q[c][0]; x = min(r, left); legs.append((h, x)); left -= x
                    if r - x <= 1e-9:
                        Q[c].pop(0); TR[h]['rm'], TR[h]['cause'] = t, 'exhausted'
                    else:
                        Q[c][0][1] = r - x
                if left <= 1e-9:
                    break
            if legs:
                settle = t + int(rng.integers(0, 1001))                           # real-like creation lag 0-1,000 ms, out of order (review B2)
                sid = m.rec(settle, [(legs[j][0], TR[legs[j][0]]['signer'], sd, TR[legs[j][0]]['tick'], legs[j][1]) for j in rng.permutation(len(legs))],
                            tk=f'T{int(rng.integers(0, 4))}')
                REC[sid] = t
                for h, x in legs:
                    cell = (sd, TR[h]['tick']); v = x * TR[h]['tick'] / 100; pend[cell] += v; clears.append((t + int(rng.integers(200, 801)), cell, v))
                    if TR[h]['cause'] == 'exhausted' and TR[h]['rm'] == t and rng.random() < .5:    # same-frame refill: masks the drop
                        n += 1; h2 = f'o{mid}_{n}'; size = float(rng.choice([1, 2, 3, 5, 8, 10, 20]))
                        Q.setdefault(cell, []).append([h2, size])
                        TR[h2] = dict(signer=TR[h]['signer'], side=sd, tick=TR[h]['tick'], size=size, post=t, rm=None, cause='end_censored')
    while ft <= dur_ms + 1000:
        frame(ft); ft += 48 + int(rng.integers(0, 5))
    return m, TR, REC, SNAP


def simcheck(X, seeds=(11, 12, 13, 14), dur_ms=60_000, quiet=False):
    """Score every claim of run_market against the simulator's truth. An observed / constrained claim that fails is a DEFECT; on this FIFO
    venue an assumption_sensitive miss comes from a clock or the settle-order sequence and is counted. T1 / T2 window-certain must not fire."""
    import time
    n, bad, ex, pts, cpu = Counter(), Counter(), [], [], 0.

    def add(sc, fld, cls, failed, info=''):
        n[(sc, fld, cls)] += 1
        if failed:
            bad[(sc, fld, cls)] += 1
            if cls in (C.OBSERVED, C.CONSTRAINED) and len(ex) < 12:
                ex.append(f'{sc} {fld} {cls}: {info}')
    sizes, evl = [], [0, 0]
    for sd_ in seeds:
        m, TR, REC, SNAP = simulate(sd_, 400 + sd_, dur_ms)
        dfs = m.dfs(); c0 = time.process_time()
        f, o, lk, s_ = X.run_market(*dfs, m.market())
        cpu += time.process_time() - c0; sizes.append((len(dfs[2]), len(dfs[1]), len(dfs[0]), int(o.mk_hash.nunique())))
        inv = cls_inv(o)                         # the checker's class property: a differing field group must be assumption_sensitive
        add('orders', 'class_property', C.OBSERVED, False)
        for h_, grp_ in inv:
            add('orders', 'class_property', C.OBSERVED, True, f'{h_} {grp_} differs across scenarios, not labelled assumption_sensitive')
        for r in f.itertuples():
            tm = T0 + REC[r.sid]
            add('fills', 'window', C.OBSERVED, not ((r.x_lo_hard_ms < 0 or r.x_lo_hard_ms < tm) and tm <= r.hi_ms), f'{r.mk_hash} lo-m {r.x_lo_hard_ms - tm} hi-m {r.hi_ms - tm}')
            if r.lo_ms >= 0:
                evl[0] += 1; evl[1] += int(r.lo_ms >= tm)
        ref = f.sort_values(['hi_ms', 'x_rec_rank']).drop_duplicates('mk_hash').set_index('mk_hash')
        for r in o.itertuples():
            T = TR[r.mk_hash]; post = T0 + T['post']; rm = None if T['rm'] is None else T0 + T['rm']; sc = r.scenario
            add(sc, 'size_lb', C.OBSERVED, T['size'] < r.size_lb - 1e-9, r.mk_hash)
            if not math.isnan(r.size_exact):
                add(sc, 'size_exact', r.size_class, abs(r.size_exact - T['size']) > 1e-6, f'{r.mk_hash} {r.size_exact} vs {T["size"]} {r.removal_method}')
            if not math.isnan(r.x_size_ub):
                add(sc, 'size_ub', r.size_class, T['size'] > r.x_size_ub + 1e-6, f'{r.mk_hash} {T["size"]} > {r.x_size_ub}')
            add(sc, 'post', r.post_class, not ((r.post_lo_ms < 0 or r.post_lo_ms < post) and post <= r.post_hi_ms),
                f'{r.mk_hash} lo-post {r.post_lo_ms - post} hi-post {r.post_hi_ms - post} {r.post_method}')
            if r.persist_lo_ms >= 0:
                add(sc, 'persist', r.persist_class, not (post <= r.persist_lo_ms and (rm is None or r.persist_hi_ms <= rm)), r.mk_hash)
            add(sc, 'removal_member', r.removal_class, T['cause'] not in r.removal_set.split('|'), f'{r.mk_hash} {T["cause"]} not in {r.removal_set}')
            if rm is None:
                add(sc, 'removal_window', r.removal_class, r.removal_hi_ms >= 0, f'{r.mk_hash} still resting, hi {r.removal_hi_ms} {r.removal_method}')
            else:
                add(sc, 'removal_window', r.removal_class, not ((r.removal_lo_ms < 0 or r.removal_lo_ms < rm) and (r.removal_hi_ms < 0 or rm <= r.removal_hi_ms)),
                    f'{r.mk_hash} lo-rm {r.removal_lo_ms - rm} hi-rm {r.removal_hi_ms - rm} {r.removal_method}')
            fb_ = int(ref.loc[r.mk_hash, 'f_before'])
            if C.SCENARIOS[sc]['priority'] == 'price_time' and fb_ >= 0:
                qs = SNAP.get(fb_, {}).get((T['side'], T['tick']), []); hs = [h for h, _ in qs]
                if r.mk_hash in hs:
                    ahead = sum(x for _, x in qs[:hs.index(r.mk_hash)]); top = math.inf if math.isnan(r.ahead_hi) else r.ahead_hi
                    add(sc, 'queue', r.queue_class, not (r.ahead_lo - 1e-6 <= ahead <= top + 1e-6), f'{r.mk_hash} {ahead} not in [{r.ahead_lo}, {r.ahead_hi}]')
        pts.append({s: (v['T1_price']['makers'], v['T1_price']['violations'], v['T1_price']['violations_certain'], v['T2_time']['groups'],
                        v['T2_time']['violations'], v['T2_time']['violations_certain']) for s, v in s_['priority_tests'].items() if s.startswith('S_')})
    defects = sum(v for k, v in bad.items() if k[2] in (C.OBSERVED, C.CONSTRAINED))
    t12 = sum(v[2] + v[5] for p_ in pts for v in p_.values())                  # the window-certain versions must not fire on a FIFO venue
    if quiet:
        if defects or t12:
            print(f'   simcheck: {defects} observed / constrained claims failed, T1 + T2 {t12}; first: {(ex or [""])[0][:120]}')
            raise SystemExit(1)
        return
    print(f'SIMCHECK {len(seeds)} markets x {dur_ms // 1000} s (frames, records, legs, hashes per market: {sizes}); run_market cpu {cpu:.2f} s total')
    for k in sorted(n):
        print(f'  {k[0]:9s} {k[1]:15s} {k[2]:21s} n {n[k]:6d}  failed {bad[k]:5d}')
    print('  priority tests (T1 makers, T1 violations settle order, T1 certain, T2 groups, T2 violations settle order, T2 certain):', pts)
    print(f'  evidence lo_ms (a scenario refinement) at or after the true match: {evl[1]} of {evl[0]} bounded fills'
          f' ({evl[1] / max(evl[0], 1):.4%}); lo < match on {evl[0] - evl[1]} ({(evl[0] - evl[1]) / max(evl[0], 1):.4%})')
    for e in ex:
        print('  DEFECT', e)
    if defects or t12:
        print(FAIL, f'simcheck: {defects} observed / constrained claims failed; T1 + T2 violations {t12} on a FIFO venue')
        raise SystemExit(1)
    print(f'SIMCHECK PASSED: 0 of {sum(v for k, v in n.items() if k[2] in (C.OBSERVED, C.CONSTRAINED))} observed / constrained claims failed; '
          f'T1 / T2 certain = 0; assumption_sensitive misses {sum(v for k, v in bad.items() if k[2] == C.SENSITIVE)}')


if __name__ == '__main__':
    if sys.argv[1:] == ['mutants']:
        mutants()
    elif sys.argv[1:] == ['all']:
        import pf_mt_lifecycle as X_
        main(X_); simcheck(X_)
    elif sys.argv[1:2] == ['final']:                                       # all + the named mutants (no base re-check)
        import pf_mt_lifecycle as X_
        main(X_); simcheck(X_); mutants(only=tuple(sys.argv[2:]), base_check=False)
    elif sys.argv[1:2] == ['simcheck']:
        import pf_mt_lifecycle as X_
        simcheck(X_, dur_ms=int(sys.argv[2]) if len(sys.argv) > 2 else 60_000)
    else:
        import pf_mt_lifecycle as X_
        main(X_)
