"""SELFTEST of tools/chain_post_fill.py (POST + FILL logic-chain census; contract order_estimator_contract.md 4.3 / 4.4 / 5 / 7.0 / 7.1).
SYNTHETIC data only, in a temporary directory (CPF_SELFTEST_TMP or the system default; no data file of the repository is read). The market is
built by the REAL chain_context.build_market from hand-written depth frames / levels / tob, a BTC grid loaded by pf_mt_decision.load_btc,
fills_valued legs joined by chain_context.join_windows and hand-written lifecycle tables; every order goes through the real order_estimator
definitions and the real chain_context queries. Every expected value below is computed by hand (comments), never by the tool.
WORLD (ms after ws; ws = 2026-10-24T01:00Z, we = ws + 300 s; frames every 100 ms on [-30 s, +60 s]; YES ticks):
 book: bid 39 = 50 sh (< 25 s) / 30 sh; bid 40 = 100 (< 28 s), absent [28 s, 35 s), 80 after; ask 42 = 100 except absent [47 s, 49 s); ask 43 =
   50; background bid 1 / ask 99. BTC mid flat, +10 bp from +12 s (aligned with a bid: r_1s on [12, 13) s, r_5s on [12, 17) s, r_30s on
   [12, 42) s).
 A rank1 bid 39, window [10, 30] s, fills 30 sh @30 s + 10 @33 s: positions early_queue_builder on [10, 20] s (the next print at 39 is its own at
   30 s), behind_touch on (20, 28) s, touch_joiner on [28, 30] s -> post_position timing_sensitive (201 instants); post_class
   early_or_late_unidentified (20 s: early and late both compatible); rest [0, 20 s]; queue front_reached_after_clearing (ahead at acceptance
   30..50 sh, S_FIFO ahead_hi 0); front_after_partial 1; next order B (reprice; sequential_identified: B's post_lo 40 s >= A's removal_hi
   34 s); opposite own maker shares in 30 s E 15 + C 10 = 25; a take of 20 sh @35 s, 2 s after its last fill: +40 -> +20 reduce_partial,
   opposing; inventory flat (recorded / certain), possible timing_sensitive (+30 once its first fill's window opens at 28 s); PNL 2 rows, 40
   sh, net .40 -> 1.0 c/sh, gross markouts .1 / .2 / .3 c/sh.
 B rank1 bid 40, [40, 45] s, 25 sh @45 s: at the touch at every instant -> post_position touch_joiner, touch_joiner_only; short_rest (5 s); queue
   unidentified (ahead 80, S_FIFO ahead_hi 5 > TOL, ahead_lo 0); inventory adding (+20); its PNL row excluded (st_10s beyond_end).
 C rank1 ask 41, [50, 50.6] s, 10 sh: inside the spread -> touch_improver_only, depth empty; late_post (identified); front_at_acceptance;
   inventory offsetting (recorded +30, s = -1).
 D rank2 bid 40, S_NOPRIO post_lo -1: unidentified / window_unbounded, phase straddle, no control. E rank1 ask 43, [45.5, 48] s, 15 sh: behind
   until 47 s then at -> timing_sensitive; short_rest; front_reached_after_clearing; next order C (reprice, sequential_identified).
 F rank3 ask 43, [-20, -15] s, 7 sh: behind_touch at every instant, phase pre, control no_eligible (its span is one instant).
 CONTROLS rank1 span [30, 50.6] s: touch_at_or_inside_all obs 2/4 (B, C) and every replicate .5 (A's / E's windows always hold a behind
   instant, B's / C's never) -> p_upper = p_lower = 1; touch_behind_all replicate k = [E's draw k < 44.5 s] / 4; depth_empty_all .25 (C
   only); inv_offsetting_all replicate k = ([C's draw k != 30 s] + [E's draw k != 30 s]) / 4.
CHECKS also: the per-instant code rules, next print (at or after), post_class / post_position / indicator units (post_position can be the
 constant early_queue_builder: a positive control); the market's grid derivations and every order window against chain_context.over_window
 exhaustively (CHECK_EVERY = 1, CHECK_K = all) and 200 random windows against an independent per-instant evaluation; the look-ahead
 falsifier (data after B's window changed: B unchanged; inside: changed; planted _MUT_LEAK_MS: caught; planted _MUT_NO_E: caught); the
 pilot path equals the full path on the pilot address; the cost paths (vectorised light cores = light_core on every order; books / cores
 from cell fills, presorted fills and top-3 rows3_of = the full-table path on the base market, the look-ahead variant with a real P_drop
 and an order with legs at two cells); takes as join keys (first sid, window starts; the dropped take-class columns absent); --compare
 (a second pilot equal to the first; the declared drops of the first pilot's layout ignored; a changed value and a missing row caught);
 refusals with their reasons (forbidden day, write-once, population, fills_valued binding, two spellings, a corrupted context, post_hi !=
 first hi, memory / CPU guards, compare without a reference); end to end run + pilots (outputs, receipt, projection).
 Usage (run_light): chain_post_fill_selftest.py"""
import hashlib
import json
import math
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
import chain_post_fill as CPF  # noqa: E402
import chain_context as CX  # noqa: E402

C, LC, E, OE, DEC = CPF.C, CPF.LC, CPF.E, CPF.OE, CPF.DEC
DAY, MID = '20261024', 101
WS = 1_792_803_600_000
WE = WS + 300_000
R1, R2, R3 = CPF.TOP3
OTHER = '0x' + 'ab' * 20
NAN = float('nan')
RES = dict(n=0, fail=[])


def check(name, cond, detail=''):
    RES['n'] += 1
    if not cond:
        RES['fail'].append(f'{name} {detail}'.strip())
        print(f'FAIL {name} {detail}', flush=True)


def near(a, b, tol=1e-9):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol


def block(name, fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        RES['fail'].append(f'{name}: uncaught {type(e).__name__}: {e}')
        print(f'FAIL {name}: uncaught {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        return None


def expect(name, fn, substr):
    """fn must refuse with a ValueError naming substr."""
    try:
        fn()
    except ValueError as e:
        check(name, substr in str(e), f'(message: {e})')
        return
    except Exception as e:  # noqa: BLE001
        check(name, False, f'(raised {type(e).__name__}: {e})')
        return
    check(name, False, '(no refusal)')


# ------------------------------------------------------------------ the synthetic world ------------------------------------------------------------------
def s39(r):
    return 50. if r < 25_000 else 30.


def s40(r):
    return 100. if r < 28_000 else (0. if r < 35_000 else 80.)


def s42(r):
    return 0. if 47_000 <= r < 49_000 else 100.


def book0(r):
    return {('bid', 1): 1., ('bid', 39): s39(r), ('bid', 40): s40(r), ('ask', 42): s42(r), ('ask', 43): 50., ('ask', 99): 1.}


def book_after(r):
    """the look-ahead variant: bid 39 / 40 wiped AFTER B's window (r >= 45.1 s)."""
    b = book0(r)
    if r >= 45_100:
        b[('bid', 39)] = b[('bid', 40)] = 0.
    return b


FR = list(range(-30_000, 60_001, 100))


def depth_tables(book):
    fr, lv, tb = [], [], []
    for k, r in enumerate(FR):
        fid, t = 1000 + k, WS + r
        rv = (t + 50) * 10 ** 6
        fr.append(dict(mid=MID, fid=fid, upd_ms=t, recv_ns=rv, conn='c0', order_count=0, pend_bids='[]', pend_asks='[]', los_id='', los_side='',
                       los_price=0., los_outcome='', status='ok'))
        bk = {c: q for c, q in book(r).items() if q > 0}
        for (sd, tk), q in sorted(bk.items()):
            lv.append(dict(mid=MID, fid=fid, side=sd, ptick=tk, shares=float(q), status='ok', x_price=tk / 100.))
        bids = [tk for (sd, tk) in bk if sd == 'bid']
        asks = [tk for (sd, tk) in bk if sd == 'ask']
        bb, ba = max(bids), min(asks)
        tb.append(dict(day=DAY, mid=MID, upd_ms=t, recv_ns=rv, conn='c0', order_count=0, n_bid=len(bids), n_ask=len(asks), best_bid=bb / 100.,
                       best_bid_sh=bk[('bid', bb)], best_ask=ba / 100., best_ask_sh=bk[('ask', ba)], pend_bid_usd=0., pend_ask_usd=0., status='ok'))
    frames = pd.DataFrame(fr).astype(C.SCHEMAS['frames'])
    levels = pd.DataFrame(lv).astype(dict(C.SCHEMAS['levels'], x_price='float64'))
    tob = pd.DataFrame(tb).astype(C.SCHEMAS['tob'])
    return frames, levels, tob


def btc_day(root, bump=()):
    """a 100 ms grid on [ws - 400 s, we + 10 s] written as btc_<day>.parquet and loaded by pf_mt_decision.load_btc (unchanged)."""
    g = np.arange(WS - 400_000, WE + 10_001, 100, dtype=np.int64)
    lm = np.log(100_000.) + np.where(g >= WS + 12_000, 0.001, 0.)
    for a, b, dlt in bump:
        lm = lm + np.where((g >= a) & (g < b), dlt, 0.)
    d = Path(root) / 'btc'
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(g_ms=g, bid=np.exp(lm) - .5, ask=np.exp(lm) + .5, mid=np.exp(lm), n_upd=np.ones(len(g), np.int64),
                      age_ms=np.full(len(g), 50, np.int64), status='ok')).to_parquet(d / f'btc_{DAY}.parquet', index=False)
    BT = DEC.load_btc(d, DAY, Counter())
    return dict(BT=BT, code=np.where(BT['ok'], 0, 4).astype(np.int8), cum=None, day=DAY, load_counts={}, inputs={})


# hash, signer, side, tick, fills [(sid, shares, hi_rel, S_FIFO x_ahead_hi)], post_lo (rel; per scenario when a dict), S_FIFO (ahead_lo, ahead_hi), removal_hi_rel
ORD = [('hA', R1, 'bid', 39, [('sA1', 30., 30_000, 0.), ('sA2', 10., 33_000, 0.)], 10_000, (0., 0.), 34_000),
       ('hB', R1, 'bid', 40, [('sB', 25., 45_000, 5.)], 40_000, (0., 5.), 46_000),
       ('hC', R1, 'ask', 41, [('sC', 10., 50_600, 0.)], 50_000, (0., 0.), 51_000),
       ('hD', R2, 'bid', 40, [('sD', 5., 55_000, NAN)], dict(S_FIFO=50_000, S_FIFO_LO=50_000, S_NOPRIO=None), (0., NAN), 56_000),
       ('hE', R1, 'ask', 43, [('sE', 15., 48_000, 0.)], 45_500, (0., 0.), 49_000),
       ('hF', R3, 'ask', 43, [('sF', 7., -15_000, NAN)], -20_000, (0., NAN), -14_000),
       ('hO', OTHER, 'bid', 40, [('sT', 20., 35_000, NAN)], 20_000, (0., NAN), 36_000)]


def life_tables(ords=ORD, post_hi_shift=None):
    recs = sorted((f[2], f[0]) for o in ords for f in o[4])
    rk = {sid: i for i, (_, sid) in enumerate(recs)}
    F, O = [], []
    for h, g, sd, tk, fills, plo, (alo, ahi), rhi in ords:
        for sid, q, hr, xah in fills:
            hi = WS + hr
            F.append(dict(mid=MID, sid=sid, mk_hash=h, mk_idx=0, y_side=sd, y_ptick=tk, shares=q, hi_ms=hi, lo_ms=hi - 200, t_method='pend_inc',
                          f_before=-1, f_after=-1, status='ok', x_evidence_fid=-1, x_drop_fid=-1, x_lo_exact_ms=-1, x_lo_hard_ms=hi - 2_000,
                          x_rec_rank=rk[sid], x_exh_price=False, x_worst_k=0, x_exh_S_FIFO=False, x_exh_S_FIFO_LO=False, x_ahead_lo_S_FIFO=0.,
                          x_ahead_lo_S_FIFO_LO=0., x_ahead_hi_S_FIFO=xah, x_ahead_hi_S_FIFO_LO=xah))
        his = [WS + f[2] for f in fills]
        los = [x - 2_000 for x in his]
        n, cum = len(fills), float(sum(f[1] for f in fills))
        for s in CPF.SCEN:
            pl = plo[s] if isinstance(plo, dict) else plo
            a_lo, a_hi = (alo, ahi) if s != 'S_NOPRIO' else (0., NAN)
            ph = min(his) + ((post_hi_shift or {}).get(h, 0))
            O.append(dict(scenario=s, mid=MID, mk_hash=h, mk_signer=g, y_side=sd, y_ptick=tk, n_fills=n, cum_shares=cum, first_hi_ms=min(his),
                          last_hi_ms=max(his), size_lb=cum, size_exact=NAN, size_class='observed', post_lo_ms=-1 if pl is None else WS + pl,
                          post_hi_ms=ph, post_class='constrained', post_method='capacity', persist_lo_ms=min(his) if n >= 2 else -1,
                          persist_hi_ms=max(los) if n >= 2 else -1, persist_class='observed' if n >= 2 else 'unresolved', ahead_lo=a_lo,
                          ahead_hi=a_hi, queue_class='assumption_sensitive', removal_set='cancelled_or_expired|exhausted', removal_lo_ms=max(los),
                          removal_hi_ms=WS + rhi, removal_class='constrained', removal_method='book_gone', status='ok', x_size_ub=NAN,
                          x_post_cap_ms=-1, x_post_fifo_ms=-1, x_ahead_hi_raw=a_hi, x_n_lo_unbounded=0, x_as_taker=0, x_capacity_violation_frames=0,
                          x_clock_violation_frames=0))
    return pd.DataFrame(F), pd.DataFrame(O)


LINKS = pd.DataFrame([
    dict(mid=MID, mk_signer=R1, y_side='bid', kind='reprice', prev_hash='hA', next_hash='hB', prev_ptick=39, next_ptick=40,
         prev_last_hi_ms=WS + 33_000, next_first_hi_ms=WS + 45_000, gap_ms=12_000, x_latest=True, label='descriptive', status='ok'),
    dict(mid=MID, mk_signer=R1, y_side='ask', kind='reprice', prev_hash='hE', next_hash='hC', prev_ptick=43, next_ptick=41,
         prev_last_hi_ms=WS + 48_000, next_first_hi_ms=WS + 50_600, gap_ms=2_600, x_latest=True, label='descriptive', status='ok')])

# addr, sid, role, hash, y_side, s, q, p, settle_rel, values
FVROWS = [(R1, 'sA1', 'maker', 'hA', 'bid', 1, 30., .39, 30_000, dict(net_res=.30, v_1s=.03, v_10s=.06, v_60s=.09, mq=30.)),
          (R1, 'sA2', 'maker', 'hA', 'bid', 1, 10., .39, 33_000, dict(net_res=.10, v_1s=.01, v_10s=.02, v_60s=.03, mq=10.)),
          (R1, 'sT', 'taker', 'hTK', 'ask', -1, 20., .40, 35_000, dict(net_res=-.05, fee_fill=.002, mq=20.)),
          (R1, 'sB', 'maker', 'hB', 'bid', 1, 25., .40, 45_000, dict(net_res=.25, v_1s=.025, v_10s=NAN, st_10s='beyond_end', v_60s=.05, rq=25.)),
          (R1, 'sE', 'maker', 'hE', 'ask', -1, 15., .43, 48_000, dict(net_res=-.15, v_1s=.015, v_10s=.03, v_60s=.045, mq=15.)),
          (R1, 'sC', 'maker', 'hC', 'ask', -1, 10., .41, 50_600, dict(net_res=.05, v_1s=.01, v_10s=.02, v_60s=.03, mq=10.)),
          (R2, 'sD', 'maker', 'hD', 'bid', 1, 5., .40, 55_000, dict(net_res=.01, rq=5.)),
          (R3, 'sF', 'maker', 'hF', 'ask', -1, 7., .43, -15_000, dict(net_res=.02, rq=7.))]


def fv_frame(rows=FVROWS):
    cols = list(dict.fromkeys(tuple(CPF.FV_COLS) + tuple(CX.FV_READ)))
    out = []
    for a, sid, role, h, ys, s, q, p, tr, v in rows:
        r = dict(day=DAY, mid=MID, sid=sid, mk_idx=0, px_rank=0, role=role, addr=a, hash=h, cp_addr=OTHER, kind='regular', y_side=ys, s=s, q=q,
                 p=p, self_trade=False, settle_ms=WS + tr, t_ms=WS + tr, t_status='lo', fee=0., fee_type='SHARES', fee_dy=0., status='ok',
                 src_status='ok', m_end=.5, v_end=s * q * (.5 - p), fee_fill=0., net_res=0., mq=0., rq=0., st_res='ok', st_1s='ok', v_1s=0.,
                 st_10s='ok', v_10s=0., st_60s='ok', v_60s=0.)
        r.update(v)
        out.append(r)
    return pd.DataFrame(out)[cols].astype({c: E.FV[c] for c in cols})


def make_market(root, book=book0, bump=(), ords=ORD, fv=None, post_hi_shift=None, tables=None):
    frames, levels, tob = depth_tables(book)
    btc = btc_day(root, bump)
    fills, orders = life_tables(ords, post_hi_shift) if tables is None else tables
    fvd = fv_frame() if fv is None else fv
    legs = CX.join_windows(fvd[list(CX.FV_READ)].copy(), fills)
    return CX.build_market(DAY, MID, WS, WE, CPF.TOP3, btc=btc, market_status='ok', book=True, tob=tob, frames=frames, levels=levels, legs=legs,
                           life_fills=fills, orders=orders, inputs={}, counts={})


def write_root(root, fv=None, ids=None):
    """the files run() reads: econ/identities.json, econ/fills_valued.parquet, life/<day>_<mid>_{orders,fills,x_links}.parquet."""
    root = Path(root)
    (root / 'econ').mkdir(parents=True, exist_ok=True)
    (root / 'life').mkdir(parents=True, exist_ok=True)
    top = ids or [dict(rank=i + 1, addr=a) for i, a in enumerate(CPF.TOP3)]
    (root / 'econ' / 'identities.json').write_text(json.dumps(dict(ex_ante=dict(day=C.ID_DAY, top10_shares=top))))
    (fv_frame() if fv is None else fv).to_parquet(root / 'econ' / 'fills_valued.parquet', index=False)
    fills, orders = life_tables()
    orders.to_parquet(root / 'life' / f'{DAY}_{MID}_orders.parquet', index=False)
    fills.to_parquet(root / 'life' / f'{DAY}_{MID}_fills.parquet', index=False)
    LINKS.to_parquet(root / 'life' / f'{DAY}_{MID}_x_links.parquet', index=False)
    return root


class CXW:
    """chain_context with load_market replaced by the prebuilt synthetic markets (every other function is the real one)."""

    def __init__(self, markets):
        self._m = markets
        self.__file__ = CX.__file__

    def __getattr__(self, k):
        return getattr(CX, k)

    def load_market(self, day, mid, addrs, root=None, hash_cache=None, **kw):
        return self._m[(day, int(mid))]


def fv_market(root):
    return CPF.read_fv(Path(root) / 'econ' / 'fills_valued.parquet', [(DAY, MID)], Counter())


def passing(M, root, check_=True, ranks=(0, 1, 2)):
    cnt = Counter()
    rows, fq, st, info = CPF.market_pass(CX, M, LINKS, fv_market(root), ranks, cnt, check=check_)
    return {r['mk_hash']: r for r in rows}, fq, st, info, cnt


# ------------------------------------------------------------------ unit checks ------------------------------------------------------------------
def t_units():
    big = np.iinfo(np.int64).max
    # bid at 40: behind, at, inside, empty own side, crossing (ask 40), unknown distance, unknown opposite, early (next print 10 s later)
    d = np.array([1., 0., -1., -np.inf, -1., NAN, 0., 1.])
    opp = np.array([42., 42., 42., 42., 40., 42., NAN, 42.])
    nxt = np.array([5_000] * 7 + [10_000], np.int64)                  # a print 5 s later (no E) / 10 s later (E)
    post, touch = CPF.touch_codes(d, opp, True, 40, nxt, np.zeros(8, np.int64))
    check('unit: POST codes', post.tolist() == [2, 1, 1, 1, 3, -1, -1, 0], str(post.tolist()))
    check('unit: touch codes', touch.tolist() == [2, 0, 1, 1, 3, -1, -1, 2], str(touch.tolist()))
    post, _ = CPF.touch_codes(np.array([0.]), np.array([39.]), False, 39, np.array([big]), np.zeros(1, np.int64))
    check('unit: an ask at the best bid crosses', post.tolist() == [3])
    nx = CPF.next_print({('bid', 40): np.array([100, 200], np.int64)}, ('bid', 40), np.array([50, 100, 150, 200, 250]))
    check('unit: next print at or after', nx.tolist() == [100, 100, 200, 200, big], str(nx.tolist()))
    check('unit: next print, no print', CPF.next_print({}, ('ask', 3), np.array([5])).tolist() == [big])
    core = dict(flags=set(), post_lo_u=0, clock_bad=False)
    for w, want in ((500, 'late_post'), (1_000, 'late_post'), (1_001, 'short_rest'), (10_000, 'short_rest'), (10_001, 'early_or_late_unidentified'),
                    (300_000, 'early_or_late_unidentified'), (-1, 'unidentified')):
        check(f'unit: post_class width {w}', CPF.post_class(core, w)[0] == want, str(CPF.post_class(core, w)))
    check('unit: post_class invalid', CPF.post_class(dict(core, flags={'identity_conflict'}), 500) == ('unidentified', 'invalid_order'))
    check('unit: post_class clock', CPF.post_class(dict(core, clock_bad=True), 500) == ('unidentified', 'first_fill_clock'))
    check('unit: post_class unbounded', CPF.post_class(dict(core, post_lo_u=-1), 500) == ('unidentified', 'window_unbounded'))
    W = dict(post=[np.array([3]), np.array([0]), np.array([0]), np.array([0])])
    check('unit: post_position can be the constant early_queue_builder (positive control)', CPF.post_position(W, 0) == ('early_queue_builder', 'early_queue_builder'))
    W = dict(post=[np.array([0]), np.array([2]), np.array([1]), np.array([0])])
    check('unit: post_position timing_sensitive', CPF.post_position(W, 0) == ('timing_sensitive', 'behind_touch|touch_joiner'))
    check('unit: post_position unknown', CPF.post_position(dict(post=[np.array([0])] * 4), 0) == ('unknown', ''))
    W = dict(n=np.array([3, 3]), touch_known=np.array([3, 2]), touch=[np.array([3, 2]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])])
    v, e = CPF.indicator(W, 'touch', ('at', 'inside'), 'all')
    check('unit: all needs every instant known', v.tolist() == [True, False] and e.tolist() == [True, False])
    v, e = CPF.indicator(W, 'touch', ('at',), 'some')
    check('unit: some', v.tolist() == [True, True] and e.tolist() == [True, True])
    fo = CPF.follow_opposite(np.array([100, 200]), np.array([50, 150, 31_000, 30_150], np.int64), np.array([1., 2., 4., 8.]))
    check('unit: opposite legs in 30 s (first / union)', fo == (2., 10., 2), str(fo))
    expect('refuse: memory self-guard', lambda: CPF.guard('unit', limit=1.), 'memory self-guard')
    expect('refuse: CPU self-guard', lambda: CPF.guard('unit', cpu0=time.process_time() - 1e6), 'CPU self-guard')


# ------------------------------------------------------------------ the market pass: hand-computed orders ------------------------------------------------------------------
def t_orders(root):
    M = make_market(root)
    R, fq, st, info, cnt = passing(M, root)
    check('orders: six top-3 orders', sorted(R) == ['hA', 'hB', 'hC', 'hD', 'hE', 'hF'], str(sorted(R)))
    A, B, Cc, D, Eo, F = (R.get(h, {}) for h in ('hA', 'hB', 'hC', 'hD', 'hE', 'hF'))
    exp = {
        'hA': dict(rank=1, post_class='early_or_late_unidentified', post_class_reason='width_gt_early_ms', post_position='timing_sensitive',
                   post_cat_set='behind_touch|early_queue_builder|touch_joiner', early_queue_builder_possible=True, touch_joiner_possible=True,
                   touch_joiner_only=False, touch_improver_possible=False, window_width_ms=20_000, window_class='early_possible', rest_lo_ms=0.,
                   rest_hi_ms=20_000., rest_bin='gt_10s', fresh_certain=False, early_possible=True, n_instants=201, n_unknown_instants=0,
                   touch_cat='timing_sensitive', touch_set='at|behind', dist_min=0., dist_max=1., depth_min=30., depth_max=50., spread_min=2.,
                   spread_max=3., spread_cat='wide', depth_cat='nonempty', btc_1s_cat='timing_sensitive', btc_5s_cat='timing_sensitive',
                   btc_5s_set='flat|with', btc_30s_cat='timing_sensitive', a_5s_min=0., inv_cat='flat', inv_certain_cat='flat',
                   inv_possible_cat='timing_sensitive', pend_abs_max=30., phase='live', tau_min_s=270., tau_max_s=290., d_sensitive=False,
                   queue_class='front_reached_after_clearing', ahead_acc_min=30., ahead_acc_max=50., touch_first_ms=WS + 28_000,
                   touch_at_start=False, n_later_fills=1, n_tied_first=0, n_front_after_partial=1, next_kind='reprice',
                   next_link_status='sequential_identified', next_hash='hB', opp_sh_30s_first=25., opp_sh_30s_any=25., n_opp_legs_30s_any=2,
                   mq_sh=40., n_takes_30s_after_last=1, take_after_last_sid='sT', take_after_last_delay_ms=2_000, n_takes_30s_after_first=1,
                   take_after_first_sid='sT', follow_first_lo_ms=WS + 30_000, follow_last_lo_ms=WS + 33_000, pnl_rows=2, pnl_shares=40.,
                   net_res=.40, v_1s=.04, v_10s=.08, v_60s=.12, pnl_excluded_rows=0, eval_lo_ms=WS + 10_000),
        'hB': dict(rank=1, post_class='short_rest', post_position='touch_joiner', post_cat_set='touch_joiner', touch_joiner_only=True,
                   early_queue_builder_possible=False, rest_bin='le_10s', window_class='intermediate', touch_cat='at', dist_min=0., dist_max=0.,
                   depth_min=80., depth_max=80., spread_min=2., spread_max=2., btc_1s_cat='flat', btc_5s_cat='flat', btc_30s_cat='timing_sensitive',
                   inv_cat='adding', inv_certain_cat='adding', inv_possible_cat='adding', queue_class='unidentified', ahead_acc_min=80.,
                   ahead_acc_max=80., next_kind='none', next_link_status='none', opp_sh_30s_first=25., n_takes_30s_after_last=0,
                   take_after_last_sid='', take_after_last_delay_ms=-1, follow_last_lo_ms=WS + 45_000, pnl_rows=0, pnl_excluded_rows=1,
                   pnl_excluded_reasons='{"st_10s": 1}', phase='live', d_sensitive=False),
        'hC': dict(rank=1, post_class='late_post', post_class_reason='width_le_late_ms', fresh_certain=True, rest_bin='le_1s', window_class='late_post',
                   post_position='touch_joiner', touch_cat='inside', touch_improver_only=True, touch_improver_possible=True, touch_joiner_possible=False,
                   dist_min=-1., dist_max=-1., depth_cat='empty', depth_min=0., depth_max=0., queue_class='front_at_acceptance', inv_cat='offsetting',
                   inv_possible_cat='offsetting', next_kind='none', opp_sh_30s_first=0., pnl_rows=1, net_res=.05, n_instants=7, d_sensitive=False),
        'hD': dict(rank=2, post_class='unidentified', post_class_reason='window_unbounded', window_status='window_unbounded', rest_bin='unbounded',
                   early_possible=True, fresh_certain=False, phase='straddle', queue_class='unidentified', window_width_ms=-1, eval_lo_ms=WS - 300_000),
        'hE': dict(rank=1, post_class='short_rest', post_position='timing_sensitive', post_cat_set='behind_touch|touch_joiner',
                   touch_cat='timing_sensitive', touch_set='at|behind', queue_class='front_reached_after_clearing', ahead_acc_min=50., ahead_acc_max=50.,
                   next_kind='reprice', next_link_status='sequential_identified', next_hash='hC', inv_cat='offsetting', btc_30s_cat='flat',
                   d_sensitive=False, pnl_rows=1, net_res=-.15),
        'hF': dict(rank=3, post_class='short_rest', post_position='behind_touch', post_cat_set='behind_touch', touch_cat='behind', phase='pre',
                   queue_class='unidentified')}
    for h, kv in exp.items():
        r = R.get(h, {})
        for k, v in kv.items():
            got = r.get(k, '<missing>')
            ok = near(got, v, 1e-9) if isinstance(v, float) else (got == v)
            check(f'orders {h}.{k}', ok, f'(got {got!r}, want {v!r})')
    check('orders: the dropped take-class columns are absent', not any(k in r_ for r_ in R.values() for k in CPF.TAKE_DROPPED))
    check('orders hA.a_5s_max ~ +10 bp', near(A.get('a_5s_max'), 10., 1e-6), str(A.get('a_5s_max')))
    check('orders hA.a_30s_max ~ +10 bp', near(A.get('a_30s_max'), 10., 1e-6), str(A.get('a_30s_max')))
    for h, w in (('hB', 5_000), ('hC', 600), ('hE', 2_500), ('hF', 5_000)):
        check(f'orders {h} width', R.get(h, {}).get('window_width_ms') == w, str(R.get(h, {}).get('window_width_ms')))
    check('orders: no order has early_queue_builder at every instant (post_hi = first hi)', cnt['position_early_queue_builder_at_every_instant'] == 0)
    check('orders: context checks ran (grid + K windows)', cnt['check_windows'] == 6 and cnt['check_grid_points'] > 0, str(dict(cnt)))
    st_ = Counter({k: v for k, v in cnt.items() if k.startswith('control_status')})
    check('controls: statuses', st_ == Counter({'control_status:ok': 4, 'control_status:window_unbounded': 1, 'control_status:no_eligible': 1}), str(st_))
    ctl, starts = info['ctl'], info['starts']
    hs = list(ctl['hs'])
    lo_s, hi_s = info['spans'][0]
    for h in ('hA', 'hB', 'hC', 'hE'):
        i = hs.index(h)
        w = int(ctl['hi'][i] - ctl['lo'][i])
        s_ = starts[i]
        check(f'controls {h}: 200 draws inside the live pool and the span', len(s_) == 200 and bool((s_ >= max(WS, lo_s)).all())
              and bool((s_ + w <= hi_s).all()) and bool((s_ % 100 == 0).all()), f'({s_.min()}, {s_.max()}, w {w})')
    S = {(r_['field']): r_ for r_ in CPF.ctrl_table(CX, [st], [('rank1', DAY)])}
    t = S['touch_at_or_inside_all']
    check('controls: touch_at_or_inside_all obs .5, null .5, p 1 / 1', near(t['obs'], .5) and t['n_obs'] == 4 and near(t['null_mean'], .5)
          and near(t['null_p05'], .5) and near(t['null_p95'], .5) and near(t['p_upper'], 1.) and near(t['p_lower'], 1.) and t['n_draws'] == 200, str(t))
    iE, iC = hs.index('hE'), hs.index('hC')
    rep = (starts[iE] < WS + 44_500).astype(float) / 4.
    t = S['touch_behind_all']
    check('controls: touch_behind_all = E draws < 44.5 s / 4', near(t['obs'], 0.) and near(t['null_mean'], rep.mean(), 1e-12), f'{t} vs {rep.mean()}')
    t = S['depth_empty_all']
    check('controls: depth_empty_all .25 / .25', near(t['obs'], .25) and near(t['null_mean'], .25) and near(t['null_p05'], .25), str(t))
    rep = ((starts[iC] != WS + 30_000).astype(float) + (starts[iE] != WS + 30_000).astype(float)) / 4.
    t = S['inv_offsetting_all']
    check('controls: inv_offsetting_all hand rule', near(t['obs'], .5) and near(t['null_mean'], rep.mean(), 1e-12), f'{t} vs {rep.mean()}')
    R2_, _, st2, info2, _ = passing(M, root)
    check('controls: deterministic draws', np.array_equal(info2['starts'], starts))
    fqd = pd.DataFrame(fq)
    check('fill queue rows: A first + front_after_partial', sorted(fqd[fqd.mk_hash == 'hA'].kind.tolist()) == ['first', 'front_after_partial'])
    return M, R, info


def t_equivalence(root, M):
    """the grid evaluator against chain_context: exhaustive over_window checks and 200 random windows vs an independent evaluation."""
    old = (CPF.CHECK_EVERY, CPF.CHECK_K)
    CPF.CHECK_EVERY, CPF.CHECK_K = 1, 100
    try:
        R, _, _, info, cnt = passing(M, root)
        check('equivalence: every grid instant and every order window agree with chain_context', cnt['check_windows'] == 6
              and cnt['check_grid_points'] == 2 * info['grid'] * info['cells'], str(dict(cnt)))
    finally:
        CPF.CHECK_EVERY, CPF.CHECK_K = old
    ctx = info['ctx']
    rng = np.random.default_rng(7)
    bad = 0
    for k in range(200):
        lo = int(WS + rng.integers(-25_000, 55_000))
        hi = lo + int(rng.integers(0, 6_000))
        c = int(rng.integers(0, len(ctx.cells)))
        sd, tk = ctx.cells[c]
        a = int(rng.integers(0, 3))
        x = np.unique(np.r_[lo, np.arange((lo // 100 + 1) * 100, hi, 100), hi]).astype(np.int64)
        td, b = CX.touch_distance(M, x, sd, tk), CX.book(M, x)
        dd = np.asarray(td['ticks'], float)
        opp = np.asarray(b['ask_tick'] if sd == 'bid' else b['bid_tick'], float)
        ph = ctx.prints.get((sd, tk), np.zeros(0, np.int64))
        cats = set()
        for j in range(len(x)):                                       # the declared rule, instant by instant
            if np.isnan(dd[j]) or np.isnan(opp[j]):
                continue
            if (sd == 'bid' and opp[j] <= tk) or (sd == 'ask' and opp[j] >= tk):
                cats.add('crossing'); continue
            nxt = [p for p in ph if p >= x[j]]
            if nxt and nxt[0] - x[j] >= 10_000 or (not nxt):
                cats.add('early_queue_builder'); continue
            cats.add('touch_joiner' if dd[j] <= 0 else 'behind_touch')
        W = ctx.windows(np.array([lo]), np.array([hi]), np.array([c]), np.array([0 if sd == 'bid' else 1]), np.array([a]))
        mine = set(CPF.post_position(W, 0)[1].split('|')) - {''}
        n_ok = int(W['n'][0]) == len(x)
        p = CX.position(M, CPF.TOP3[a], x)
        sg = 1 if sd == 'bid' else -1
        inv = {CPF.INV_CATS[int(v)] for v in CPF.inv_code(sg * np.asarray(p['recorded'], float))}
        mine_inv = set(CPF.window_category(W, 'inv', 0)[1].split('|')) - {''}
        if mine != cats or not n_ok or mine_inv != inv:
            bad += 1
            if bad <= 3:
                print(f'  mismatch {k}: lo {lo - WS} hi {hi - WS} cell {(sd, tk)} mine {mine} want {cats} n {int(W["n"][0])} / {len(x)} inv {mine_inv} / {inv}')
    check('equivalence: 200 random windows equal an independent per-instant evaluation', bad == 0, f'({bad} mismatches)')


def t_lookahead(root, R0):
    """data after B's window changed -> B unchanged; inside -> changed; the planted leak and the dropped E rule are caught."""
    cols = ('post_position', 'post_cat_set', 'touch_cat', 'touch_set', 'dist_min', 'dist_max', 'spread_min', 'spread_max', 'depth_min', 'depth_max',
            'btc_1s_cat', 'btc_5s_cat', 'btc_30s_cat', 'a_1s_min', 'a_1s_max', 'a_5s_min', 'a_5s_max', 'a_30s_min', 'a_30s_max', 'inv_cat',
            'inv_certain_cat', 'inv_possible_cat', 'pend_abs_max', 'tau_min_s', 'tau_max_s', 'queue_class', 'ahead_acc_min', 'ahead_acc_max',
            'd_sensitive_fields', 'early_queue_builder_possible')

    def same(r0, r1):
        return all(near(r0[c], r1[c]) if isinstance(r0[c], float) else r0[c] == r1[c] for c in cols)
    Ma = make_market(Path(root) / 'after', book=book_after, bump=[(WS + 45_100, WE + 20_000, .002)])
    Ra = passing(Ma, root)[0]
    check('lookahead: data after B / A / F windows change nothing in them', all(same(R0[h], Ra[h]) for h in ('hA', 'hB', 'hF')),
          str([c for c in cols if not (near(R0['hB'][c], Ra['hB'][c]) if isinstance(R0['hB'][c], float) else R0['hB'][c] == Ra['hB'][c])]))
    check('lookahead: the change after 45.1 s does reach a later window (E)', not same(R0['hE'], Ra['hE']))
    Mi = make_market(Path(root) / 'inside', bump=[(WS + 42_500, WS + 43_000, .0005)])
    Ri = passing(Mi, root)[0]
    check('lookahead positive control: a BTC change inside B changes it', Ri['hB']['btc_1s_cat'] == 'timing_sensitive' and R0['hB']['btc_1s_cat'] == 'flat'
          and same(R0['hA'], Ri['hA']), f"({R0['hB']['btc_1s_cat']} -> {Ri['hB']['btc_1s_cat']})")
    M0 = make_market(Path(root) / 'base2')
    CPF._MUT_LEAK_MS = 300
    try:
        Rl0, Rla = passing(M0, root, check_=False)[0], passing(Ma, root, check_=False)[0]
        check('falsifier: a planted 300 ms look-ahead is caught by the after-window perturbation', not same(Rl0['hB'], Rla['hB']))
        expect('falsifier: the context check refuses the planted look-ahead', lambda: passing(M0, root, check_=True), 'context check')
    finally:
        CPF._MUT_LEAK_MS = 0
    CPF._MUT_NO_E = True
    try:
        Rn = passing(M0, root)[0]
        check('falsifier: dropping the later-print rule changes A (caught)', Rn['hA']['early_queue_builder_possible'] is False
              and R0['hA']['early_queue_builder_possible'] is True)
    finally:
        CPF._MUT_NO_E = False


BOOK_KEYS = ('front_class', 'ahead_lo', 'ahead_hi', 'touch_first_ms', 'touch_at_start', 'nomask_status', 'nomask_n', 'nomask_rep_lo', 'nomask_rep_hi',
             'nomask_span_lo', 'nomask_span_hi', 'nomask_rest_lo', 'nomask_rest_hi', 'nomask_class', 'drop_r', 'drop_frame_ms', 'book_flags')
CORE_KEYS = ('post_lo_u', 'post_hi', 'rest_ub', 'post_class', 'first_lo_hard', 'last_lo_hard', 'n_front_after_partial', 'size_status', 'flags',
             'priors', 'conflict', 'clock_bad', 'union', 'clip', 'fill_hi', 'fill_q', 'fill_ahi_fifo')


def eq(x, y):
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        x, y = np.asarray(x), np.asarray(y)
        return x.shape == y.shape and bool(np.array_equal(x, y, equal_nan=x.dtype.kind == 'f'))
    if isinstance(x, float) and isinstance(y, float):
        return (math.isnan(x) and math.isnan(y)) or x == y
    if isinstance(x, dict):
        return repr(x) == repr(y)
    return x == y


def old_books(M, fvm, addrs):
    """the first pilot's estimator path (rows3_of over every order, per-order light_core, the full fills table, order_core's own sort)."""
    O, Fl, B, ws = M.orders, M.life_fills, M.depth['B'], M.ws
    R3 = OE.rows3_of(O)
    top = {h for h, r3 in R3.items() if str(r3['S_NOPRIO']['mk_signer']).lower() in addrs}
    settle = {(s, h): int(t) for s, h, t, r in zip(fvm.sid, fvm.hash, fvm.settle_ms, fvm.role) if r == 'maker'}
    fb = {h: g for h, g in Fl[Fl.mk_hash.isin(top)].groupby('mk_hash', sort=False)}
    cc = OE.clip_counts(O)
    cores = {}
    for h in sorted(top):
        r3 = R3[h]
        g = str(r3['S_NOPRIO']['mk_signer']).lower()
        fl = fb.get(h, Fl.iloc[0:0])
        clip = OE.clip_for(cc.get(g, Counter()), OE._fin(r3['S_FIFO']['size_exact']))
        c = OE.order_core(r3, fl, settle={sid: settle.get((sid, h)) for sid in fl.sid}, clip=clip)
        c['clip'] = clip
        cores[h] = c
    cell_of = {h: (str(R3[h]['S_NOPRIO']['y_side']), int(R3[h]['S_NOPRIO']['y_ptick'])) for h in R3}
    cells = sorted({cell_of[h] for h in top if CPF.cell_ok(cell_of[h])})
    members = {c: [] for c in cells}
    for h, c in cell_of.items():
        if c in members:
            members[c].append(cores[h] if h in cores else CPF.light_core(R3[h]))
    touch, FF = OE.touch_arrays(B), OE.fill_frames(Fl, B)
    CA = {c: OE.cell_arrays(B, Fl, FF, LC.SIDE[c[0]], c[1]) for c in cells}
    NM = {c: OE.cell_nomask(members[c], B, CA[c], ws) for c in cells}
    books = {h: OE.order_book(cores[h], B, touch, Fl, FF, CA.get(cell_of[h]), NM.get(cell_of[h]), ws) for h in sorted(top)}
    return cores, books


def t_fastpaths(root):
    """the cost paths give the first pilot's results: vectorised light cores = light_core; books / cores (cell fills, presorted fills,
    top-3 rows3_of) = the full-table path, on the base market, the look-ahead variant (a real P_drop) and an order with legs at two cells
    (the full-table fallback)."""
    M = make_market(Path(root) / 'fp_base')
    LCV = CPF.light_cores(M.orders, set(M.orders.mk_hash))
    R3all = OE.rows3_of(M.orders)
    bad = [h for h in R3all if CPF.light_core(R3all[h]) != LCV.get(h)]
    check('fast path: vectorised light cores equal light_core on every order', not bad and len(LCV) == len(R3all), str(bad))
    fills, orders = life_tables(ORD + [('hZ', R2, 'bid', 39, [('sZ1', 4., 20_000, NAN), ('sZ2', 6., 21_000, NAN)], 15_000, (0., NAN), 22_000)])
    fills.loc[fills.sid == 'sZ2', 'y_ptick'] = 40                       # the hash's second leg at another tick
    orders.loc[orders.mk_hash == 'hZ', 'status'] = 'identity_conflict'
    fvm = fv_market(root)
    worlds = (('base', M), ('lookahead_variant', make_market(Path(root) / 'fp_after', book=book_after, bump=[(WS + 45_100, WE + 20_000, .002)])),
              ('legs_at_two_cells', make_market(Path(root) / 'fp_idc', tables=(fills, orders))))
    for nm, MM in worlds:
        co, bo = old_books(MM, fvm, CPF.LOW3)
        cn, bn, _, _, _ = CPF.market_orders(MM.orders, MM.life_fills, LINKS, fvm[fvm.addr_l.isin(CPF.LOW3)], MM.depth['B'], MM.ws, CPF.LOW3)
        diffs = [(h, k) for h in bo for k in BOOK_KEYS if not eq(bo[h][k], bn.get(h, {}).get(k, '<missing>'))]
        diffs += [(h, k) for h in co for k in CORE_KEYS if not eq(co[h][k], cn.get(h, {}).get(k, '<missing>'))]
        check(f'fast path ({nm}): books and cores equal the full-table path', not diffs and sorted(bo) == sorted(bn), str(diffs[:6]))
        if nm == 'lookahead_variant':
            check('fast path: the variant has a real P_drop (B)', near(bn['hB']['drop_r'], 80.) and bn['hB']['drop_frame_ms'] == WS + 45_100,
                  f"{bn['hB']['drop_r']} {bn['hB']['drop_frame_ms']}")
        if nm == 'legs_at_two_cells':
            check('fast path: the two-cell order is in the population, flagged conflict (identity)', ('hZ' in cn) and ('hZ' in bn)
                  and bool(cn.get('hZ', {}).get('conflict')), str(sorted(cn)))


def t_refusals(root, M, info):
    ctx = info['ctx']
    ctx.S['dist'][0, 0] = 7.
    expect('refuse: a corrupted grid distance (context check)', lambda: CPF.check_context(CX, ctx, np.zeros(0), np.zeros(0), np.zeros(0),
                                                                                       np.zeros(0), np.zeros(0), None, None, Counter()), 'context check')
    Mp = make_market(Path(root) / 'posthi', post_hi_shift={'hB': -100})
    expect('refuse: post_hi != first fill hi', lambda: passing(Mp, root), 'post_hi != first fill hi')
    bad = fv_frame(FVROWS + [(R1.lower(), 'sX', 'maker', 'hX', 'bid', 1, 1., .4, 20_000, {})])
    r2 = write_root(Path(root) / 'spell', fv=bad)
    expect('refuse: an address under two spellings', lambda: fv_market(r2), 'two spellings')


# ------------------------------------------------------------------ end to end ------------------------------------------------------------------
def t_e2e(root, M):
    root = write_root(Path(root) / 'e2e')
    fsha = hashlib.sha256((root / 'econ' / 'fills_valued.parquet').read_bytes()).hexdigest()
    cxw = CXW({(DAY, MID): M})
    out = root / 'v2' / 'chains' / 'post_fill'
    expect('refuse: a forbidden day before any path', lambda: CPF.run(cxw, inp=root, out=out, days=('20261028',), fv_sha=fsha),
           'days must be distinct measurement days')
    expect('refuse: the fills_valued binding', lambda: CPF.run(cxw, inp=root, out=out, days=(DAY,), fv_sha='0' * 64), 'differs from the accepted binding')
    check('refusals leave no output', not out.exists() and not list(out.parent.glob('.post_fill.partial*')) if out.parent.exists() else True)
    rec = CPF.run(cxw, inp=root, out=out, days=(DAY,), fv_sha=fsha, log=lambda *a: None)
    files = sorted(p.name for p in out.iterdir())
    check('e2e: outputs', files == ['orders.csv', 'receipt.json', 'summary.csv'], str(files))
    OD = pd.read_csv(out / 'orders.csv')
    check('e2e: one row per top-3 filled order', len(OD) == 6 and sorted(OD.mk_hash) == ['hA', 'hB', 'hC', 'hD', 'hE', 'hF'])
    rA = OD[OD.mk_hash == 'hA'].iloc[0]
    check('e2e: orders.csv carries the hand values', rA.post_class == 'early_or_late_unidentified' and rA.queue_class == 'front_reached_after_clearing'
          and near(rA.net_res, .4) and rA.take_after_last_sid == 'sT' and int(rA.follow_last_lo_ms) == WS + 33_000)
    SM = pd.read_csv(out / 'summary.csv')
    g = SM[(SM.table == 'classes') & (SM['rank'] == 'rank1') & (SM['day'].astype(str) == DAY) & (SM.field == 'post_class')]
    lp = g[g.category == 'late_post']
    check('e2e: classes late_post (rank1): 1 order, 1 execution, 10 sh, net .5 c/sh', len(lp) == 1 and int(lp.n_orders.iloc[0]) == 1
          and int(lp.n_executions.iloc[0]) == 1 and near(lp.shares.iloc[0], 10.) and near(lp.net_res_c_sh.iloc[0], .5), lp.to_dict('records'))
    ts = g[g.category == 'short_rest']
    # B (excluded row) + E (15 sh, net -.15): cohort 15 sh -> -1.0 c/sh; orders 2, executions 2, shares 40
    check('e2e: classes short_rest (rank1): B excluded from the cohort', len(ts) == 1 and int(ts.n_orders.iloc[0]) == 2 and near(ts.shares.iloc[0], 40.)
          and near(ts.pnl_shares.iloc[0], 15.) and near(ts.net_res_c_sh.iloc[0], -1.) and int(ts.pnl_excluded_rows.iloc[0]) == 1, ts.to_dict('records'))
    tot = SM[(SM.table == 'classes') & (SM['rank'] == 'top3') & (SM['day'] == 'all') & (SM.field == 'post_class') & (SM.category == '_total')]
    check('e2e: pooled denominators 6 orders / 7 executions / 102 sh', len(tot) == 1 and int(tot.n_orders.iloc[0]) == 6
          and int(tot.n_executions.iloc[0]) == 7 and near(tot.shares.iloc[0], 102.), tot.to_dict('records'))
    ex = SM[(SM.table == 'executions') & (SM['rank'] == 'rank1') & (SM['day'].astype(str) == DAY)]
    fr = ex[(ex.field == 'first_fill_queue_class') & (ex.category == 'front_reached_after_clearing')]
    fp = ex[(ex.field == 'later_fill') & (ex.category == 'front_after_partial')]
    check('e2e: executions (A, E reached after clearing; A front_after_partial 10 sh)', len(fr) == 1 and int(fr.n_executions.iloc[0]) == 2
          and len(fp) == 1 and int(fp.n_executions.iloc[0]) == 1 and near(fp.shares.iloc[0], 10.))
    ct = SM[(SM.table == 'control') & (SM['rank'] == 'top3') & (SM['day'] == 'all') & (SM.field == 'touch_at_or_inside_all')]
    check('e2e: control row pooled (D, F without controls)', len(ct) == 1 and near(ct.obs.iloc[0], .5) and int(ct.n_obs.iloc[0]) == 4
          and int(ct.n_no_control.iloc[0]) == 2, ct.to_dict('records'))
    J = json.loads((out / 'receipt.json').read_text())
    check('e2e: receipt', J['version'] == CPF.VERSION and J['counts']['orders_top3'] == 6 and J['rows']['orders'] == 6 and 'labels' in J
          and 'declared' in J and set(J['outputs']) == {'orders.csv', 'summary.csv'} and J['inputs']['econ/fills_valued.parquet']['sha256'] == fsha)
    expect('refuse: write-once', lambda: CPF.run(cxw, inp=root, out=out, days=(DAY,), fv_sha=fsha, log=lambda *a: None), 'write-once')
    outp = root / 'v2' / 'chains' / 'post_fill_pilot'
    rp = CPF.run(cxw, inp=root, out=outp, days=(DAY,), ranks=(0,), pilot=(DAY, 1), fv_sha=fsha, log=lambda *a: None)
    P = pd.read_csv(outp / 'orders.csv')
    check('e2e pilot: the pilot address only', sorted(P.mk_hash) == ['hA', 'hB', 'hC', 'hE'])
    a = OD.set_index('mk_hash').loc[sorted(P.mk_hash)].reset_index()
    b = P.sort_values('mk_hash').reset_index(drop=True)
    check('e2e pilot: rows equal the full run (cell members of other addresses kept)', a[b.columns].astype(str).equals(b.astype(str)))
    pj = rp.get('projection') or {}
    check('e2e pilot: projection', pj.get('orders_full') == 6 and pj.get('orders_in_pilot') == 4 and pj.get('full_run_s', -1) >= 0, str(pj))
    outp2 = root / 'v2' / 'chains' / 'post_fill_pilot_v2'
    rp2 = CPF.run(cxw, inp=root, out=outp2, days=(DAY,), ranks=(0,), pilot=(DAY, 1), fv_sha=fsha, log=lambda *a: None, compare=outp)
    cp = rp2.get('compare') or {}
    check('e2e compare: a second pilot equals the first on every column', cp.get('identical') is True and cp['orders.csv']['columns_compared'] > 50
          and cp['summary.csv']['columns_compared'] > 10 and not cp['orders.csv']['new_columns'], str(cp))
    check('e2e compare: in the receipt', json.loads((outp2 / 'receipt.json').read_text()).get('compare', {}).get('identical') is True)
    newc = ['take_after_last_sid', 'take_after_first_sid', 'follow_first_lo_ms', 'follow_last_lo_ms']
    ref = root / 'v2' / 'chains' / 'ref_first_pilot_layout'
    shutil.copytree(outp, ref)
    o = pd.read_csv(ref / 'orders.csv', dtype=str, keep_default_na=False)
    for c in CPF.TAKE_DROPPED:
        o[c] = 'x'
    o.drop(columns=newc).to_csv(ref / 'orders.csv', index=False)
    s = pd.read_csv(ref / 'summary.csv', dtype=str, keep_default_na=False)
    ex_ = s[(s.table == 'classes') & (s.field == 'post_class')].copy()
    ex_['field'] = 'take_after_last_class'
    pd.concat([s, ex_]).to_csv(ref / 'summary.csv', index=False)
    cr = CPF.compare_outputs(outp2, ref)
    check('compare: the declared drops are ignored (the first pilot layout); new join columns listed', cr['identical'] is True
          and sorted(cr['orders.csv']['new_columns']) == sorted(newc), str(cr))
    bad_ = root / 'v2' / 'chains' / 'ref_bad'
    shutil.copytree(outp, bad_)
    o = pd.read_csv(bad_ / 'orders.csv', dtype=str, keep_default_na=False)
    o.loc[o.mk_hash == 'hA', 'post_class'] = 'late_post'
    o.to_csv(bad_ / 'orders.csv', index=False)
    cb = CPF.compare_outputs(outp2, bad_)
    check('compare falsifier: one changed value is caught and named', cb['identical'] is False and cb['orders.csv']['mismatched'] == {'post_class': 1}, str(cb))
    o[o.mk_hash != 'hB'].to_csv(bad_ / 'orders.csv', index=False)
    cb = CPF.compare_outputs(outp2, bad_)
    check('compare falsifier: a missing row is caught', cb['identical'] is False and cb['orders.csv']['same_row_keys'] is False, str(cb))
    expect('refuse: compare without a reference', lambda: CPF.compare_outputs(outp2, root / 'none'), 'no reference outputs')
    bad = write_root(Path(root).parent / 'badids', ids=[dict(rank=1, addr=R2), dict(rank=2, addr=R1), dict(rank=3, addr=R3)])
    expect('refuse: a population other than the top 3', lambda: CPF.run(cxw, inp=bad, out=bad / 'o', days=(DAY,), fv_sha=None), 'differ from pf_mt_decision.TOP3')


def main():
    c0 = time.process_time()
    tmp = Path(tempfile.mkdtemp(prefix='cpf_selftest_', dir=os.environ.get('CPF_SELFTEST_TMP') or None))
    old_book, old_n = E.load_book, CPF.N_BOOK
    E.load_book = lambda path=None: {DAY: [MID]}
    CPF.N_BOOK = 1
    try:
        block('units', t_units)
        root = write_root(tmp / 'w')
        res = block('orders', t_orders, root)
        if res is not None:
            M, R0, info = res
            block('equivalence', t_equivalence, root, M)
            block('fast paths', t_fastpaths, root)
            block('lookahead', t_lookahead, root, R0)
            block('e2e', t_e2e, tmp, M)
            block('refusals', t_refusals, root, M, info)
    finally:
        E.load_book, CPF.N_BOOK = old_book, old_n
        shutil.rmtree(tmp, ignore_errors=True)
    res = dict(result='PASS' if not RES['fail'] else 'FAIL', checks=RES['n'], failed=len(RES['fail']), failures=RES['fail'],
               cpu_s=round(time.process_time() - c0, 2), vmhwm_mib=round(CPF.vmhwm_mib(), 1))
    print(json.dumps(res, indent=1))
    sys.exit(0 if not RES['fail'] else 1)


if __name__ == '__main__':
    main()
