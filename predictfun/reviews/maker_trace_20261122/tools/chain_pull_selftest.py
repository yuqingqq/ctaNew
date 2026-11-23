"""chain_pull selftest (synthetic only: no data file of any day is opened; every path is a temporary directory). The stub serves synthetic
market data through chain_context_v1's call shapes and DELEGATES over_window / matched_times / null_summary / draws_per_target to the real
chain_context module, so the tool's controls, window checks and statistics run through the helper's own code. Cases:
 A  census: classes per view (exhausted_proven only under S_FIFO, capacity_tight in every row, capacity_tight_view, end_censored_possible,
    unidentified x3), end histories, every pulls.csv row an EX / PL tie (identification tied), the successor split.
 B  positive control: a maker that pulls exactly when BTC moves >= 5 bp against the order within 1 s (decision at the jump, effect 250 ms
    later) is recovered against the matched_times presence controls (>= 200 draws per target) at every delay profile.
 C  null: a maker that pulls at random times shows no association.
 D  phase: a live-phase drift; a phase-MISMATCHED control (planted switch) fakes the signal, the declared phase-matched control does not;
    D-sensitivity: a window compatible at D = 0 / 250 and timing_sensitive at D = 1,000 is flagged.
 E  fill anchoring: fills at adverse moves, pulls at random: the presence control shows the construction association, fill_anchored not.
 F  look-ahead falsifier: later BTC rows, frames (values and arrival times) and own legs perturbed -> no earlier feature changes (pending is
    the declared reconstruction exception); a planted +150 ms look-ahead does change features.
 G  units: classify, range min / max, statistics by hand through null_summary, the permutation p's calibration.
 H  refusals (reason asserted): a non-measurement day, write-once, a wrong top 3, three scenario rows, a fill-count mismatch, a chain_context
    without matched_times.
 I  the pilot property: a one-address run reproduces the full run's rows of that address exactly.
 J  equivalence: eval_windows (grid interior + point-query ends) equals chain_context.over_window on random windows (category sets, counts,
    BTC min / max, every profile); matched_times draws lie inside the declared pool.
Usage (run_light, PF_SLOT=claude4, MT_RUN=v2): chain_pull_selftest.py"""
import copy
import json
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / 'research'))
import chain_pull as P  # noqa: E402
import chain_context as CC  # noqa: E402   (the real helper: over_window / matched_times / null_summary / draws_per_target)
import pf_mt_common as C  # noqa: E402

NAN = float('nan')
RES = []


def check(name, ok, info=None):
    RES.append((name, bool(ok)))
    print(('PASS ' if ok else 'FAIL ') + name + ('' if ok or info is None else f'  [{info}]'))


def expect(name, fn, needle):
    try:
        fn()
    except Exception as e:        # noqa: BLE001
        check(name, needle in str(e), f'raised {type(e).__name__}: {e}')
        return
    check(name, False, 'did not raise')


# ------------------------------------------------------------------ the stub chain_context ------------------------------------------------------------------
class Stub:
    """chain_context_v1's call shapes over synthetic arrays (BTC log mid on a 100 ms grid: row g = the state at g; frames: the last frame
    <= t; own legs: settle_ms, s q, x_lo_hard); over_window / matched_times / null_summary / draws_per_target are the real helper's."""
    VERSION = 'selftest_stub'
    over_window = staticmethod(CC.over_window)
    matched_times = staticmethod(CC.matched_times)
    null_summary = staticmethod(CC.null_summary)
    draws_per_target = staticmethod(CC.draws_per_target)

    def __init__(self, mk, drop=()):
        self.mk = mk                                          # (day, mid) -> synthetic market dict
        for d in drop:
            setattr(self, d, None)

    def load_market(self, day, mid, addrs, root=None, *, hash_cache=None, **kw):
        c = self.mk[(day, int(mid))]
        legs = {a: pd.DataFrame(dict(settle_ms=L['settle_ms'])) for a, L in c['legs'].items()}
        return types.SimpleNamespace(day=day, mid=int(mid), ws=c['ws'], we=c['we'], legs=legs, inputs={}, checks={'stub': True}, cost={}, _c=c)

    def btc_ret(self, M, t):
        c = M._c
        t = np.asarray(t, np.int64)
        n = len(c['lm'])
        out = {}
        for nm, h in (('r_1s', 1_000), ('r_5s', 5_000), ('r_30s', 30_000)):
            i, j = t // 100 - c['g0'], (t - h) // 100 - c['g0']
            ok = (j >= 0) & (i < n) & (i >= 0)
            out[nm] = np.where(ok, 1e4 * (c['lm'][np.clip(i, 0, n - 1)] - c['lm'][np.clip(j, 0, n - 1)]), np.nan)
        out['ok'] = np.isfinite(out['r_30s'])
        return out

    def _frame(self, M, t):
        return np.searchsorted(M._c['upd'], np.asarray(t, np.int64), 'right') - 1

    def book(self, M, t):
        j = self._frame(M, t)
        ok = j >= 0
        jc = np.maximum(j, 0)
        bid = np.where(ok, M._c['bid'][jc], np.nan).astype(float)
        ask = np.where(ok, M._c['ask'][jc], np.nan).astype(float)
        return dict(bid_tick=bid, ask_tick=ask, mid_c=(bid + ask) / 2., ok=ok)

    def touch_distance(self, M, t, side, tick):
        t = np.asarray(t, np.int64)
        b = self.book(M, t)
        isb = np.broadcast_to(np.asarray(side, dtype=object) == 'bid', t.shape)
        tk = np.broadcast_to(np.asarray(tick, float), t.shape)
        return dict(ticks=np.where(isb, b['bid_tick'] - tk, tk - b['ask_tick']))

    def level_size(self, M, t, side, tick):
        t = np.asarray(t, np.int64)
        j = self._frame(M, t)
        isa = np.broadcast_to(np.asarray(side, dtype=object) == 'ask', t.shape).astype(np.int64)
        tk = np.broadcast_to(np.asarray(tick, np.int64), t.shape)
        return dict(size=np.where(j >= 0, 100. + 10. * ((np.maximum(j, 0) + tk + isa) % 5), np.nan))

    def _legs(self, M, a):
        L = M._c['legs'].get(str(a).lower())
        if L is None:
            return dict(settle_ms=np.zeros(0, np.int64), sq=np.zeros(0), lo=np.zeros(0, np.int64))
        return L

    def position(self, M, a, t, clock='hi'):
        L = self._legs(M, a)
        o = np.argsort(L['settle_ms'], kind='stable')
        st, cs = L['settle_ms'][o], np.r_[0., np.cumsum(L['sq'][o])]
        j = np.searchsorted(st, np.asarray(t, np.int64), 'left')
        return dict(recorded=cs[j], n=j, pos=cs[j])

    def pending(self, M, a, t):
        L = self._legs(M, a)
        t = np.asarray(t, np.int64)
        ol, os_ = np.argsort(L['lo'], kind='stable'), np.argsort(L['settle_ms'], kind='stable')
        cl, cs = np.r_[0., np.cumsum(L['sq'][ol])], np.r_[0., np.cumsum(L['sq'][os_])]
        n_lo = np.searchsorted(L['lo'][ol], t, 'left')                 # legs with lo < t
        n_st = np.searchsorted(L['settle_ms'][os_], t, 'left')         # legs with settle < t
        return dict(net=cl[n_lo] - cs[n_st])                            # lo < t <= settle

    def tau(self, M, t):
        return (M.we - np.asarray(t, np.int64)) / 1000.


# ------------------------------------------------------------------ synthetic world ------------------------------------------------------------------
DAY_A, DAY_B = '20261024', '20261025'
MA1, MA2, MB1 = 101, 102, 201
WS = {MA1: 1_792_830_000_000, MA2: 1_792_840_000_000, MB1: 1_792_920_000_000}
SIG, JUMP, PERIOD, HALF = 0.05e-4, 6e-4, 16_000, 8_000
TICK = {0: (45, 55), 1: (44, 56), 2: (40, 60)}          # rank -> (bid tick, ask tick)


def make_btc(ws, rng, jumps=True, live_drift=0.):
    we = ws + 300_000
    g_lo, g_hi = ws - 400_000, we + 120_000
    g = np.arange(g_lo, g_hi + 1, 100, dtype=np.int64)
    inc = rng.normal(0., SIG, len(g))
    inc[0] = 0.
    downs = ups = np.zeros(0, np.int64)
    if jumps:
        t0 = g_lo + 3_300
        downs = np.arange(t0, g_hi, PERIOD, dtype=np.int64)
        ups = downs + HALF
        ups = ups[ups <= g_hi]
        inc[(downs - g_lo) // 100] -= JUMP
        inc[(ups - g_lo) // 100] += JUMP
    if live_drift:
        inc[(g >= ws) & (g < we)] += live_drift
    upd = np.arange(g_lo, g_hi + 1, 50, dtype=np.int64)
    bid = 47 + (upd // 7_000) % 3 - 1
    ask = 53 + (upd // 11_000) % 3 - 1
    return dict(g0=int(g_lo // 100), lm=np.log(70_000.) + np.cumsum(inc), downs=downs, ups=ups, upd=upd, bid=bid.astype(float),
                ask=ask.astype(float), ws=ws, we=we, legs={})


class World:
    def __init__(self):
        self.orders, self.fills, self.links = {}, {}, {}      # (day, mid) -> lists
        self.mk = {}
        self.n = 0
        self.meta = {}                                        # hash -> dict(kind, ...)

    def add_order(self, day, mid, a, side, his, rhi, sets, ub=None, status='ok', receipt_hi=False, kind='', q=5.):
        self.n += 1
        h = f'0x{self.n:064x}'
        his = sorted(int(x) for x in his)
        n = len(his)
        tick = TICK[a][0 if side == 'bid' else 1]
        lo_h = [-1 if receipt_hi else x - 2_000 for x in his]
        lb = [x for x in lo_h if x >= 0]
        rlo = max(lb) if lb else -1
        p_lo, p_hi = (his[0], max(lb)) if (n >= 2 and lb and max(lb) >= his[0]) else (-1, -1)
        ub = ub if ub is not None else {s: NAN for s in P.SCEN}
        sets = sets if isinstance(sets, dict) else {s: sets for s in P.SCEN}
        rhi = rhi if isinstance(rhi, dict) else {s: int(rhi) for s in P.SCEN}
        for s in P.SCEN:
            self.orders.setdefault((day, mid), []).append(dict(
                scenario=s, mid=mid, mk_hash=h, mk_signer=P.TOP3[a], y_side=side, y_ptick=tick, n_fills=n, cum_shares=q * n,
                first_hi_ms=his[0], last_hi_ms=his[-1], persist_lo_ms=p_lo, persist_hi_ms=p_hi, ahead_lo=0., ahead_hi=NAN,
                removal_set=sets[s], removal_lo_ms=rlo, removal_hi_ms=int(rhi[s]), removal_class='constrained', removal_method='book_gone',
                status=status, x_size_ub=float(ub[s])))
        for i, x in enumerate(his):
            self.fills.setdefault((day, mid), []).append(dict(mk_hash=h, sid=f'{h[-8:]}-{i}', hi_ms=x, x_lo_hard_ms=lo_h[i], x_rec_rank=i,
                                                              shares=q, x_ahead_hi_S_FIFO=0.))
        ctx = self.mk[(day, mid)]
        L = ctx['legs'].setdefault(P.TOP3[a].lower(), dict(settle_ms=np.zeros(0, np.int64), sq=np.zeros(0), lo=np.zeros(0, np.int64)))
        sg = 1. if side == 'bid' else -1.
        L['settle_ms'] = np.r_[L['settle_ms'], np.array(his, np.int64)]
        L['sq'] = np.r_[L['sq'], np.full(n, sg * q)]
        L['lo'] = np.r_[L['lo'], np.array([x - 2_000 for x in his], np.int64)]
        self.meta[h] = dict(kind=kind, day=day, mid=mid, a=a, side=side, his=his, rhi=rhi, rlo=rlo)
        return h

    def add_link(self, day, mid, a, side, prev, nxt, kind, prev_last, nxt_first, status='ok'):
        self.links.setdefault((day, mid), []).append(dict(mk_signer=P.TOP3[a], y_side=side, kind=kind, prev_hash=prev, next_hash=nxt,
                                                          prev_ptick=TICK[a][0], next_ptick=TICK[a][0], prev_last_hi_ms=int(prev_last),
                                                          next_first_hi_ms=int(nxt_first), gap_ms=int(nxt_first - prev_last), status=status))

    def write(self, root, top3=P.TOP3):
        (root / 'econ').mkdir(parents=True)
        (root / 'life').mkdir()
        (root / 'econ' / 'identities.json').write_text(json.dumps(dict(ex_ante=dict(
            day=C.ID_DAY, top10_shares=[dict(rank=i + 1, addr=x) for i, x in enumerate(top3)]))))
        for d in (DAY_A, DAY_B):
            mids = [m for (dd, m) in self.mk if dd == d]
            pd.DataFrame(dict(mid=mids, ws=[WS[m] / 1000. for m in mids], we=[WS[m] / 1000. + 300. for m in mids],
                              status=['ok'] * len(mids))).to_parquet(root / f'markets_{d}.parquet', index=False)
        for (d, m) in self.mk:
            pd.DataFrame(self.orders.get((d, m), []), columns=list(P.ORD_COLS)).to_parquet(root / 'life' / f'{d}_{m}_orders.parquet', index=False)
            pd.DataFrame(self.fills.get((d, m), []), columns=list(P.FIL_COLS)).to_parquet(root / 'life' / f'{d}_{m}_fills.parquet', index=False)
            lk = pd.DataFrame(self.links.get((d, m), []), columns=list(P.LNK_COLS))
            lk = lk.astype(dict(gap_ms='int64', prev_last_hi_ms='int64', next_first_hi_ms='int64', prev_ptick='int64', next_ptick='int64'))
            lk.to_parquet(root / 'life' / f'{d}_{m}_x_links.parquet', index=False)


CANCEL = 'cancelled_or_expired|exhausted'


def build_world(seed=20261123):
    rng = np.random.default_rng(seed)
    W = World()
    W.mk[(DAY_A, MA1)] = make_btc(WS[MA1], rng)
    W.mk[(DAY_A, MA2)] = make_btc(WS[MA2], rng, jumps=False, live_drift=-0.15e-4)
    W.mk[(DAY_B, MB1)] = make_btc(WS[MB1], rng)

    def phase_ok(ws, a_, b_):
        return (ws - 298_000 <= a_ and b_ < ws - 1_000) or (ws + 1_000 <= a_ and b_ < ws + 299_000)
    # B: rank1 pulls at the first adverse 5 bp jump after its last fill (no adverse jump during its life); effect 250 ms after the jump
    ws = WS[MA1]
    ctx = W.mk[(DAY_A, MA1)]
    for side, adv in (('bid', ctx['downs']), ('ask', ctx['ups'])):
        for k in range(len(adv) - 1):
            j0, j1 = int(adv[k]), int(adv[k + 1])
            if not phase_ok(ws, j0, j1 + 600):
                continue
            h1 = j0 + 2_500 + int(rng.integers(0, 300)); h3 = j1 - 2_600 - int(rng.integers(0, 300))
            h2 = int(rng.integers(h1 + 100, h3 - 100))
            rhi = j1 + 250 + int(rng.integers(20, 200))
            W.add_order(DAY_A, MA1, 0, side, (h1, h2, h3), rhi, CANCEL, kind='trigger')
    # C: rank2 pulls at random times (no reaction to any move)
    n2 = 0
    while n2 < 60:
        side = ('bid', 'ask')[n2 % 2]
        p0 = int(rng.integers(ws - 295_000, ws + 270_000))
        h1 = p0 + 1_000 + int(rng.integers(0, 500)); h3 = p0 + 1_000 + int(rng.integers(11_000, 13_000))
        rhi = h3 + int(rng.integers(500, 6_000)) + int(rng.integers(20, 200))
        if not phase_ok(ws, p0, rhi):
            continue
        h2 = int(rng.integers(h1 + 100, h3 - 100))
        W.add_order(DAY_A, MA1, 1, side, (h1, h2, h3), rhi, CANCEL, kind='random')
        n2 += 1
    # successor links for B: every third trigger order gets a refill 3 s after its last fill; one link 8 s later and one invalid (neither counts)
    trig = [h for h, m in W.meta.items() if m['kind'] == 'trigger']
    for i, h in enumerate(trig):
        m = W.meta[h]
        last = m['his'][-1]
        if i % 3 == 0:
            W.add_link(DAY_A, MA1, 0, m['side'], h, f'0xs{i:063x}', 'refill', last, last + 3_000)
        elif i == 1:
            W.add_link(DAY_A, MA1, 0, m['side'], h, f'0xf{i:063x}', 'reprice', last, last + 8_000)
        elif i == 2:
            W.add_link(DAY_A, MA1, 0, m['side'], h, f'0xv{i:063x}', 'refill', last, last + 2_000, status='invalid_order')
    # D: rank3 bids live through ws (presence in both phases), pull at random in live; BTC drifts against bids in live only
    ws2 = WS[MA2]
    for i in range(30):
        p0 = int(rng.integers(ws2 - 100_000, ws2 - 40_000))
        hn = int(rng.integers(ws2 + 16_000, ws2 + 40_000))
        h1 = p0 + 1_000
        his = (h1, int(rng.integers(h1 + 500, ws2 - 1_000)), int(rng.integers(ws2 + 1_000, hn - 500)), hn)
        rhi = hn + int(rng.integers(500, 4_000)) + int(rng.integers(20, 200))
        W.add_order(DAY_A, MA2, 2, 'bid', his, rhi, CANCEL, kind='phase')
    # D-sensitivity: removal_lo = ws + 900 (the drift is fully in the 1 s window from ws + 1 s): D 0 / 250 compatible, D 1000 reaches ws - 100
    W.add_order(DAY_A, MA2, 2, 'bid', (ws2 - 30_000, ws2 + 2_900), ws2 + 3_500, CANCEL, kind='dsens')
    # E: rank2 on day B: fills exactly at adverse jumps (informed takers), pulls at random after the last fill
    ws3 = WS[MB1]
    ctx3 = W.mk[(DAY_B, MB1)]
    for side, adv in (('bid', ctx3['downs']), ('ask', ctx3['ups'])):
        for k in range(len(adv) - 2):
            his = [int(adv[k + i]) + int(rng.integers(50, 300)) for i in range(3)]
            rhi = his[-1] + int(rng.integers(500, 4_000)) + int(rng.integers(20, 200))
            if not phase_ok(ws3, int(adv[k]) - 500, rhi + 100):
                continue
            W.add_order(DAY_B, MB1, 1, side, his, rhi, CANCEL, kind='adverse_fill')
    # A: rank1 status cases on day B (live)
    b = ws3 + 100_000
    W.add_order(DAY_B, MB1, 0, 'bid', (b, b + 9_000), b + 9_500, {'S_FIFO': 'exhausted', 'S_FIFO_LO': 'exhausted', 'S_NOPRIO': CANCEL},
                kind='ex')
    W.add_order(DAY_B, MB1, 0, 'bid', (b + 20_000, b + 29_000), b + 29_500, CANCEL, ub={s: 10. for s in P.SCEN}, kind='tight')
    W.add_order(DAY_B, MB1, 0, 'ask', (b + 40_000, b + 49_000), -1, 'cancelled_or_expired|end_censored|exhausted', kind='ec')
    W.add_order(DAY_B, MB1, 0, 'ask', (b + 60_000, b + 69_000), b + 70_000, CANCEL, receipt_hi=True, kind='unb')
    W.add_order(DAY_B, MB1, 0, 'bid', (b + 80_000, b + 89_000), b + 90_000, 'cancelled_or_expired|exhausted|missing_evidence', kind='miss')
    W.add_order(DAY_B, MB1, 0, 'ask', (b + 100_000, b + 109_000), b + 110_000, CANCEL, status='identity_conflict', kind='inv')
    W.add_order(DAY_B, MB1, 0, 'bid', (b + 120_000, b + 129_000), b + 129_500, CANCEL, ub={'S_FIFO': 10., 'S_FIFO_LO': NAN, 'S_NOPRIO': NAN},
                kind='tight_view')
    return W


class Missing(dict):
    """a summary row that does not exist: every field NaN (checks on it fail instead of crashing)."""

    def __getitem__(self, k):
        return NAN


def pick(SM, **kw):
    m = np.ones(len(SM), bool)
    for k, v in kw.items():
        m &= (SM[k].astype(str) == str(v)).to_numpy()
    r = SM[m]
    if len(r) != 1:
        print(f'INFO pick: {len(r)} rows for {kw}')
        return Missing()
    return r.iloc[0]


def section(name, fn, *a):
    """run one case group; an exception is a recorded failure, never a hidden one."""
    try:
        fn(*a)
    except Exception as e:    # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(f'{name}: ran without an exception', False, f'{type(e).__name__}: {e}')


def _i(x):
    """int of a summary field (-1 when missing / NaN)."""
    try:
        return int(x) if np.isfinite(float(x)) else -1
    except (TypeError, ValueError):
        return -1


def main():
    t0 = time.process_time()
    tmp = Path(tempfile.mkdtemp(prefix='chain_pull_selftest_'))
    try:
        body(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in RES)
    print(f'chain_pull selftest: {n_ok}/{len(RES)} passed; cpu {time.process_time() - t0:.2f} s; vmhwm {P.vmhwm_mib():.0f} MiB')
    sys.exit(0 if n_ok == len(RES) else 1)


def _life(root, d, m):
    O, Fl, Lk, _ = P.read_life(root / 'life', d, m, P.LOW3)
    return O, Fl, Lk


def market_grid(stub, d, m, cells):
    """the tool's context series and grid of one synthetic market (cells: [(side 0/1, tick)])."""
    M_ = stub.load_market(d, m, P.TOP3)
    ws, we = M_.ws, M_.we
    t = np.arange((ws - P.GRID_PRE_MS) // 100 * 100, (we + P.GRID_POST_MS) // 100 * 100 + 100, 100, dtype=np.int64)
    S = P.context_series(stub, M_, t, [(('bid', 'ask')[sd], k) for sd, k in cells], P.TOP3)
    S['cells'] = cells
    return M_, P.build_grid(S, ws, we)


def body(tmp):
    section('G', units)                                   # independent of the fixture: runs even if the fixture run fails
    W = build_world()
    root = tmp / 'in'
    W.write(root)
    stub = Stub(W.mk)
    section('F', falsifier, W, stub)
    section('J', equivalence, W, stub)
    book = {DAY_A: [MA1, MA2], DAY_B: [MB1]}
    run = lambda out, **kw: P.run(stub, inp=root, out=out, days=(DAY_A, DAY_B), book=book, log=lambda *a: None, **kw)   # noqa: E731
    out = tmp / 'out_full'
    ok_run = []
    section('fixture run', lambda: ok_run.append(run(out)))
    if not ok_run:
        return
    SM = pd.read_csv(out / 'summary.csv', dtype={'day': str, 'D': str})
    PL = pd.read_csv(out / 'pulls.csv', dtype={'day': str})
    CT = pd.read_csv(out / 'controls.csv', dtype={'day': str})
    CE = pd.read_csv(out / 'census.csv', dtype={'day': str})
    R = json.loads((out / 'receipt.json').read_text())
    kind_of = {h: m['kind'] for h, m in W.meta.items()}
    PL['kind'] = PL['mk_hash'].map(kind_of)
    tr = PL[(PL['kind'] == 'trigger') & (PL['view'] == 'S_FIFO')]

    def row(**kw):
        return pick(SM, view='S_FIFO', control=kw.pop('control', 'presence'), group=kw.pop('group', 'cancel_possible'), **kw)

    def case_a():
        T = P.order_table(*_life(root, DAY_B, MB1), WS[MB1], WS[MB1] + 300_000)
        T['kind'] = T['mk_hash'].map(kind_of)
        want = dict(ex=('exhausted_proven', 'cancel_possible'), tight=('capacity_tight', 'capacity_tight'),
                    ec=('end_censored_possible', 'end_censored_possible'), unb=('unidentified', 'unidentified'),
                    miss=('unidentified', 'unidentified'), inv=('unidentified', 'unidentified'), tight_view=('capacity_tight_view', 'cancel_possible'))
        got = {k: tuple(T.loc[T['kind'] == k, f'class_{v}'].iat[0] for v in P.VIEWS) for k in want}
        check('A census: classes per view (exhausted_proven only under S_FIFO, capacity_tight in every row, capacity_tight_view, '
              'end_censored, unidentified x3)', got == want, got)
        why = {k: T.loc[T['kind'] == k, 'why_S_FIFO'].iat[0] for k in ('unb', 'miss', 'inv')}
        check('A census: unidentified reasons window_unbounded / missing_evidence / order_invalid',
              why == dict(unb='window_unbounded', miss='missing_evidence', inv='order_invalid'), why)
        check('A census: an unbounded removal window carries window_status window_unbounded (ec, unb); an invalid order order_invalid',
              all(T.loc[T['kind'] == k, 'wstat_S_FIFO'].iat[0] == 'window_unbounded' for k in ('ec', 'unb'))
              and T.loc[T['kind'] == 'inv', 'wstat_S_FIFO'].iat[0] == 'order_invalid')
        hist = {k: tuple(T.loc[T['kind'] == k, f'hist_{v}'].iat[0] for v in P.VIEWS) for k in ('ex', 'tight', 'ec', 'tight_view')}
        check('A census: end histories (ex EX / EX|PL, tight EX, ec EX|PL|EC, tight_view EX / EX|PL)',
              hist == dict(ex=('EX', 'EX|PL'), tight=('EX', 'EX'), ec=('EX|PL|EC', 'EX|PL|EC'), tight_view=('EX', 'EX|PL')), hist)
        stp = PL[PL['kind'].isin(list(want))]
        check('A census: of the status cases only ex and tight_view enter pulls.csv, and only under S_NOPRIO',
              sorted(zip(stp['kind'], stp['view'])) == [('ex', 'S_NOPRIO'), ('tight_view', 'S_NOPRIO')], sorted(zip(stp['kind'], stp['view'])))
        check('A ties explicit: every pulls.csv row is an EX / PL tie (identification tied, end_histories EX|PL, pull_identified False)',
              len(PL) > 0 and (PL['identification'] == 'tied').all() and (PL['end_histories'] == 'EX|PL').all()
              and not PL['pull_identified'].astype(bool).any())
        n_cp = 0
        for d, m in ((DAY_A, MA1), (DAY_A, MA2), (DAY_B, MB1)):
            T0 = P.order_table(*_life(root, d, m), WS[m], WS[m] + 300_000)
            n_cp += sum(int((T0['class_' + v] == 'cancel_possible').sum()) for v in P.VIEWS)
        check('A census: pulls.csv has exactly one row per cancel-possible (order, view)',
              len(PL) == n_cp and not PL.duplicated(['mk_hash', 'view']).any(), (len(PL), n_cp))
        ce1 = CE[(CE['view'] == 'S_FIFO') & (CE['day'] == DAY_B) & (CE['scope'] == 'rank1')]
        cc = {(r.cls, r.reason): int(r.orders) for r in ce1.itertuples()}
        check('A census: census.csv counts the status cases with reasons and denominators',
              cc == {('exhausted_proven', '-'): 1, ('capacity_tight', '-'): 1, ('capacity_tight_view', '-'): 1, ('end_censored_possible', '-'): 1,
                     ('unidentified', 'window_unbounded'): 1, ('unidentified', 'missing_evidence'): 1, ('unidentified', 'order_invalid'): 1}, cc)
        succ_planted = {x['prev_hash'] for L in W.links.values() for x in L if x['status'] == 'ok' and x['gap_ms'] <= 5_000}
        got_s = set(tr.loc[tr['successor'] != 'none', 'mk_hash'])
        check('A successor split: refill within 5 s found; an 8 s link and an invalid link ignored', got_s == succ_planted and len(got_s) >= 3,
              (len(got_s), len(succ_planted)))
        r1 = row(group='cp_successor_5s', scope='rank1', day=DAY_A, D='0', feature='a_1s_start')
        r2 = row(group='cp_no_successor_5s', scope='rank1', day=DAY_A, D='0', feature='a_1s_start')
        ntr = int((tr['window_status'] == 'ok').sum())
        check('A successor split: summary groups cp_successor_5s / cp_no_successor_5s partition the rank1 cancel-possible rows',
              _i(r1['n_events']) == len(got_s) and _i(r1['n_events']) + _i(r2['n_events']) == ntr, (r1['n_events'], r2['n_events'], ntr))
        check('A receipt: every output hashed, rows reconcile, context checks ran (grid points and over_window windows)',
              set(R['outputs']) == {'pulls.csv', 'controls.csv', 'summary.csv', 'census.csv'} and R['rows']['pulls'] == len(PL)
              and R['counts'].get('check_grid_points', 0) > 0 and R['counts'].get('check_windows', 0) > 0, (R['rows'], R['counts'].get('check_windows')))

    def case_b():
        for Dm in ('0', '250', '1000'):
            b1 = row(scope='rank1', day=DAY_A, D=Dm, feature='a_1s_wmin')
            b2 = row(scope='rank1', day=DAY_A, D=Dm, feature='btc_against_1s_some')
            print(f'INFO B D={Dm}: n {b1["n_obs"]} a_1s_wmin obs {b1["obs"]:.2f} null {b1["null_mean"]:.2f} p_lower {b1["p_lower"]:.4f}; '
                  f'btc_against_1s_some obs {b2["obs"]:.3f} null {b2["null_mean"]:.3f} p_upper {b2["p_upper"]:.4f}; reps {b1["n_reps"]}')
            check(f'B positive control D={Dm}: the most adverse 1 s move is recovered (obs < -4 bp, null > -1.5, p_lower <= 0.01, >= 25 '
                  f'events, 200 replicates)', b1['obs'] < -4 and b1['null_mean'] > -1.5 and b1['p_lower'] <= 0.01 and _i(b1['n_obs']) >= 25
                  and _i(b1['n_reps']) == 200, (b1['obs'], b1['null_mean'], b1['p_lower'], b1['n_obs'], b1['n_reps']))
            check(f'B positive control D={Dm}: btc_against_1s at some instant: rate >= 0.9 vs control <= 0.3, p_upper <= 0.01',
                  b2['obs'] >= 0.9 and b2['null_mean'] <= 0.3 and b2['p_upper'] <= 0.01, (b2['obs'], b2['null_mean'], b2['p_upper']))
        trd = tr[tr['window_status'] == 'ok']
        ts = float((trd['btc_against_1s_D250'] == 'timing_sensitive').mean())
        check('B classes: a 1 s trigger inside a >= 2 s window is timing_sensitive, never compatible at every instant (>= 90 %)', ts >= 0.9, ts)
        st = CT[(CT['scope'] == 'rank1') & (CT['day'] == DAY_A) & (CT['view'] == 'S_FIFO') & (CT['control'] == 'presence')
                & (CT['cls'] == 'cancel_possible')]
        check('B controls: the trigger events draw from the primary matched_times pool (status ok, 200 draws)',
              len(st) > 0 and (st['draw_status'] == 'ok').mean() >= 0.9 and (st['n_draws'] == P.N_DRAWS).mean() >= 0.9,
              st['draw_status'].value_counts().to_dict())
        b1 = row(scope='rank1', day=DAY_A, D='250', feature='a_1s_wmin')
        full = st[st['n_draws'] == P.N_DRAWS]
        cm = float(full['a_1s_wmin_ctl_mean_D250'].mean()) if len(full) == len(st) else NAN
        check('B controls.csv: the per-event control means average to the summary null mean (all draws valid); the events sit low in their '
              'own draws (mean percentile < 0.1)', abs(cm - b1['null_mean']) < 1e-9 and float(st['a_1s_wmin_pct_D250'].mean()) < 0.1,
              (cm, b1['null_mean'], float(st['a_1s_wmin_pct_D250'].mean())))

    def case_c():
        c1 = row(scope='rank2', day=DAY_A, D='250', feature='a_1s_wmin')
        c2 = row(scope='rank2', day=DAY_A, D='250', feature='btc_against_1s_some')
        print(f'INFO C: n {c1["n_obs"]} a_1s_wmin obs {c1["obs"]:.2f} null {c1["null_mean"]:.2f} p_upper {c1["p_upper"]:.3f} p_lower '
              f'{c1["p_lower"]:.3f}; btc_against_1s_some obs {c2["obs"]:.3f} null {c2["null_mean"]:.3f} p_upper {c2["p_upper"]:.3f}')
        check('C null: random pulls show no association (|diff a_1s_wmin| < 2 bp; |diff rate| < 0.25; >= 40 events)',
              abs(c1['diff']) < 2.0 and abs(c2['diff']) < 0.25 and _i(c1['n_obs']) >= 40, (c1['diff'], c2['diff'], c1['n_obs']))

    def case_d():
        d1 = row(scope='rank3', day=DAY_A, D='0', feature='a_1s_start')
        print(f'INFO D matched: n {d1["n_obs"]} obs {d1["obs"]:.3f} null {d1["null_mean"]:.3f} p_lower {d1["p_lower"]:.3f}')
        check('D phase: the phase-matched control shows no live-drift "signal" (|diff| < 0.3 bp, >= 20 events)',
              abs(d1['diff']) < 0.3 and _i(d1['n_obs']) >= 20, (d1['diff'], d1['n_obs']))
        P._MUT_PHASE_FLIP = True
        try:
            P.run(stub, inp=root, out=tmp / 'out_mut_phase', days=(DAY_A,), ranks=(2,), book=book, log=lambda *a: None)
        finally:
            P._MUT_PHASE_FLIP = False
        SMm = pd.read_csv(tmp / 'out_mut_phase' / 'summary.csv', dtype={'day': str, 'D': str})
        dm = pick(SMm, view='S_FIFO', control='presence', group='cancel_possible', scope='rank3', day=DAY_A, D='0', feature='a_1s_start')
        print(f'INFO D mismatched: obs {dm["obs"]:.3f} null {dm["null_mean"]:.3f} p_lower {dm["p_lower"]:.4f}')
        check('D phase: a phase-MISMATCHED control fakes the signal (diff < -0.6 bp, p_lower <= 0.01) - the declared control avoids it',
              dm['diff'] < -0.6 and dm['p_lower'] <= 0.01, (dm['diff'], dm['p_lower']))
        ds = PL[(PL['kind'] == 'dsens') & (PL['view'] == 'S_FIFO')]
        cls3 = tuple(ds[f'btc_against_1s_D{x}'].iat[0] for x in (0, 250, 1000)) if len(ds) else None
        check('D D-sensitivity: compatible at D 0 / 250, timing_sensitive at D 1000 (the window then reaches before the drift), flagged',
              cls3 == ('compatible', 'compatible', 'timing_sensitive') and 'btc_against_1s' in str(ds['d_sensitive'].iat[0]),
              (cls3, None if not len(ds) else ds['d_sensitive'].iat[0]))

    def case_e():
        e1 = row(scope='rank2', day=DAY_B, D='250', feature='a_1s_wmin', control='presence')
        e2 = row(scope='rank2', day=DAY_B, D='250', feature='a_1s_wmin', control='fill_anchored')
        print(f'INFO E: presence obs {e1["obs"]:.2f} null {e1["null_mean"]:.2f} p_lower {e1["p_lower"]:.4f}; fill_anchored null '
              f'{e2["null_mean"]:.2f} diff {e2["diff"]:.2f} p_lower {e2["p_lower"]:.3f} p_upper {e2["p_upper"]:.3f}; n {e1["n_obs"]} / {e2["n_obs"]}')
        check('E fill anchoring: the presence control shows the construction association (diff < -1.5 bp, p_lower <= 0.01)',
              e1['diff'] < -1.5 and e1['p_lower'] <= 0.01, (e1['diff'], e1['p_lower']))
        check('E fill anchoring: the fill_anchored control removes it (|diff| < 1 bp, >= 20 events)', abs(e2['diff']) < 1.0 and _i(e2['n_obs']) >= 20,
              (e2['diff'], e2['n_obs']))

    def case_h():
        expect('H refusal: a non-measurement day (before any path is built)',
               lambda: P.run(stub, inp=root, out=tmp / 'x1', days=('20261028',), book=book), 'measurement days')
        expect('H refusal: write-once', lambda: run(out), 'write-once')
        bad = tmp / 'in_bad'
        W.write(bad, top3=(P.TOP3[0], P.TOP3[2], P.TOP3[1]))
        expect('H refusal: identities ranks 1-3 differ from the contract top 3',
               lambda: P.run(stub, inp=bad, out=tmp / 'x2', days=(DAY_A,), book=book), 'differ from pf_mt_decision.TOP3')
        O, Fl, Lk = _life(root, DAY_A, MA1)
        expect('H refusal: a hash without exactly three scenario rows', lambda: P.order_table(O.iloc[1:], Fl, Lk, WS[MA1], WS[MA1] + 300_000),
               'exactly one row per scenario')
        expect('H refusal: life fills disagree with n_fills', lambda: P.order_table(O, Fl.iloc[1:], Lk, WS[MA1], WS[MA1] + 300_000), 'fill count')
        expect('H refusal: a chain_context without matched_times',
               lambda: P.run(Stub(W.mk, drop=('matched_times',)), inp=root, out=tmp / 'x3', days=(DAY_A,), book=book), 'lacks')

    def case_i():
        P.run(stub, inp=root, out=tmp / 'out_r1', days=(DAY_A,), ranks=(0,), pilot=(DAY_A, 1), book=book, log=lambda *a: None)
        S1 = pd.read_csv(tmp / 'out_r1' / 'summary.csv', dtype={'day': str, 'D': str})
        keys = ['view', 'group', 'control', 'D', 'feature']
        a_ = SM[(SM['scope'] == 'rank1') & (SM['day'] == DAY_A)].sort_values(keys).reset_index(drop=True)
        b_ = S1[(S1['scope'] == 'rank1') & (S1['day'] == DAY_A)].sort_values(keys).reset_index(drop=True)
        cols = keys + ['n_events', 'n_obs', 'n_ctl_valid', 'obs', 'null_mean', 'p_upper', 'p_lower']
        same = len(a_) == len(b_) and len(a_) > 0 and all(a_[c].equals(b_[c]) for c in cols)
        check('I pilot property: a one-address run reproduces the full run rows of that address exactly', same, (len(a_), len(b_)))
        R1 = json.loads((tmp / 'out_r1' / 'receipt.json').read_text())
        check('I pilot receipt: projection present', (R1.get('projection') or {}).get('full_run_s_by_orders', 0) > 0, R1.get('projection'))

    for name, fn in (('A', case_a), ('B', case_b), ('C', case_c), ('D', case_d), ('E', case_e), ('H', case_h), ('I', case_i)):
        section(name, fn)


def falsifier(W, stub):
    ctx0 = W.mk[(DAY_A, MA1)]
    ws, we = ctx0['ws'], ctx0['we']
    cut = ws + 150_000                                   # aligned (multiple of 100)
    rng = np.random.default_rng(7)
    ctx1 = copy.deepcopy(ctx0)
    late = (ctx1['g0'] + np.arange(len(ctx1['lm']))) * 100 > cut
    ctx1['lm'] = np.where(late, ctx1['lm'] + rng.normal(0, 3e-4, len(late)), ctx1['lm'])
    fl = ctx1['upd'] > cut
    ctx1['bid'] = np.where(fl, ctx1['bid'] + rng.integers(-2, 3, len(fl)), ctx1['bid'])
    ctx1['ask'] = np.where(fl, ctx1['ask'] + rng.integers(-2, 3, len(fl)), ctx1['ask'])
    upd = np.where(fl, ctx1['upd'] + 37, ctx1['upd'])                     # later frames arrive at other times
    ins = rng.integers(cut + 1, we, 300)
    o = np.argsort(np.r_[upd, ins], kind='stable')
    ctx1['upd'] = np.r_[upd, ins][o]
    ctx1['bid'] = np.r_[ctx1['bid'], np.full(300, 20.)][o]
    ctx1['ask'] = np.r_[ctx1['ask'], np.full(300, 80.)][o]
    for a, L in ctx1['legs'].items():
        lt = L['settle_ms'] > cut
        L['sq'] = np.where(lt, L['sq'] * 3. + 1., L['sq'])
        add = rng.integers(cut + 1, we, 20)
        L['settle_ms'] = np.r_[L['settle_ms'], add]
        L['sq'] = np.r_[L['sq'], rng.normal(0, 20, 20)]
        L['lo'] = np.r_[L['lo'], add - 2_000]
    s0, s1 = Stub({(DAY_A, MA1): ctx0}), Stub({(DAY_A, MA1): ctx1})
    cells = [(0, 45), (1, 55), (0, 44), (1, 56)]
    (M0, G0), (M1, G1) = market_grid(s0, DAY_A, MA1, cells), market_grid(s1, DAY_A, MA1, cells)
    n = 3_000
    hi = rng.integers(ws - 200_000, cut, n)
    hi[:300] = cut - rng.integers(1, 41, 300)          # windows ending just before the cut (the leak's reach)
    lo = hi - rng.integers(0, 8_000, n)
    side = rng.integers(0, 2, n); cell = rng.integers(0, 4, n); addr = rng.integers(0, 3, n)
    side = np.array([cells[c][0] for c in cell], np.int64)

    def diff():
        bad = []
        for Dm in P.DELAYS:
            a_ = P.windows_at(s0, M0, G0, [lo - Dm], hi, side, cell, addr)[0]
            b_ = P.windows_at(s1, M1, G1, [lo - Dm], hi, side, cell, addr)[0]
            for k in a_:
                if k == 'inside':
                    continue
                x, y = (np.stack(a_[k]), np.stack(b_[k])) if isinstance(a_[k], tuple) else (a_[k], b_[k])
                if not np.array_equal(np.asarray(x, float), np.asarray(y, float), equal_nan=True):
                    bad.append(k)
        return sorted(set(bad))
    bad = diff()
    check('F causality: every decision-time feature of windows ending before the cut is unchanged when later BTC rows, frames (values and '
          'arrival times) and own legs change (pending = the declared reconstruction exception)', set(bad) <= {'spend_start'}, bad)
    P._MUT_LEAK_MS = 150
    try:
        bad2 = diff()
    finally:
        P._MUT_LEAK_MS = 0
    check('F falsifier fires: a planted +150 ms look-ahead changes BTC / book features', bool({'a_1s_start', 'a_1s_wmin', 'opp_tw5_start'} & set(bad2)),
          bad2)


def equivalence(W, stub):
    """eval_windows against chain_context.over_window on random windows (aligned and unaligned ends, every profile)."""
    cells = [(0, 45), (1, 55), (0, 44), (1, 56)]
    M_, G = market_grid(stub, DAY_A, MA1, cells)
    ws = M_.ws
    rng = np.random.default_rng(5)
    n = 24
    lo = rng.integers(ws - 250_000, ws + 250_000, n)
    lo[:6] = lo[:6] // 100 * 100                         # some aligned ends
    hi = lo + rng.integers(0, 6_000, n)
    hi[6:9] = lo[6:9]                                    # single-instant windows
    hi[9:12] = np.maximum((hi[9:12] // 100) * 100, lo[9:12])
    cell = rng.integers(0, 4, n)
    side = np.array([cells[c][0] for c in cell], np.int64)
    addr = rng.integers(0, 3, n)
    bad = []
    for Dm in P.DELAYS:
        Wd = P.windows_at(stub, M_, G, [lo - Dm], hi, side, cell, addr)[0]
        for i in range(n):
            sd_, tk_, a_ = int(side[i]), int(G['cell_tick'][cell[i]]), int(addr[i])
            for name, ser, _, op, thr in P.MECHS:
                def fn(M2, tt, ser=ser, op=op, thr=thr):
                    pv = P.point_values(stub, M2, tt, np.full(len(tt), sd_), np.full(len(tt), tk_), np.full(len(tt), a_))
                    ok_, c_ = P.cond_of(pv[ser], op, thr)
                    return c_, ok_
                ow = CC.over_window(M_, fn, int(lo[i]), int(hi[i]), D_ms=Dm, mode='category')
                t_, v_, n_ = (int(x[i]) for x in Wd[name])
                mine = '|'.join(x for x, has in (('False', v_ - t_ > 0), ('True', t_ > 0)) if has)
                if not (str(ow['cats'][0]) == mine and int(ow['n_points'][0]) == n_ and int(ow['n_valid'][0]) == v_):
                    bad.append((Dm, i, name, mine, ow['cats'][0], n_, int(ow['n_points'][0])))
            for nm in P.WIN_FEATS:
                def fr(M2, tt, nm=nm):
                    pv = P.point_values(stub, M2, tt, np.full(len(tt), sd_), np.full(len(tt), tk_), np.full(len(tt), a_))
                    return pv[nm], ~np.isnan(pv[nm])
                ow = CC.over_window(M_, fr, int(lo[i]), int(hi[i]), D_ms=Dm, mode='sign')
                for q in ('min', 'max'):
                    x = float(Wd[f'{nm}_w{q}'][i])
                    if not (np.isfinite(x) and x == float(ow[q][0])):
                        bad.append((Dm, i, nm, q, x, float(ow[q][0])))
    check('J equivalence: eval_windows equals chain_context.over_window (MECH category sets, instant and valid counts, BTC min / max) on '
          f'{n} random windows x 3 profiles', not bad, bad[:3])
    # matched_times pool: every draw's window lies inside one phase and inside the union presence mask at every grid instant
    T = P.order_table(*_life_from_world(W, DAY_A, MA1), WS[MA1], WS[MA1] + 300_000)
    D = P.market_events(T, M_.ws, M_.we)
    from collections import Counter as _Ct
    C0, st = P.presence_controls(stub, M_, D, T, P.active_spans(M_, P.TOP3), _Ct())
    okw = (D['wst'] == 'ok').to_numpy()
    viol, n_chk = 0, 0
    for e in np.flatnonzero(okw)[:15]:
        if st[e] != 'ok':
            continue
        sel = (T['a'].to_numpy() == D['a'].iat[e]) & (T['s'].to_numpy() == D['s'].iat[e]) & T['valid'].to_numpy(bool) & (T['p_lo'].to_numpy() >= 0)
        plo, phi = T['p_lo'].to_numpy()[sel], T['p_hi'].to_numpy()[sel]
        wb = -(-int(D['w'].iat[e]) // 100) * 100
        p0, p1 = P.phase_bounds(int(D['ph'].iat[e]), M_.ws, M_.we)
        c = C0[e][C0[e] >= 0]
        g = c[:, None] + np.arange(0, wb + 1, 100)[None, :]                     # every grid instant of each draw's window
        cov = ((plo[None, None, :] <= g[:, :, None]) & (phi[None, None, :] >= g[:, :, None])).any(2).all(1)
        viol += int((~(cov & (c >= p0) & (c + wb < p1))).sum())
        n_chk += len(c)
    check('J matched_times: every primary draw lies inside its phase and inside the same-side certain-presence union',
          viol == 0 and n_chk > 0 and (st[okw] != 'n/a').all(), (viol, n_chk))


def _life_from_world(W, d, m):
    O = pd.DataFrame(W.orders[(d, m)], columns=list(P.ORD_COLS))
    O['a'] = O['mk_signer'].str.lower().map({a: i for i, a in enumerate(P.LOW3)}).astype(np.int64)
    Fl = pd.DataFrame(W.fills[(d, m)], columns=list(P.FIL_COLS))
    Lk = pd.DataFrame(W.links.get((d, m), []), columns=list(P.LNK_COLS))
    Lk = Lk.astype(dict(gap_ms='int64', prev_last_hi_ms='int64', next_first_hi_ms='int64', prev_ptick='int64', next_ptick='int64'))
    return O, Fl, Lk


def units():
    tr, va, n = np.array([3, 0, 1, 2, 0, 0]), np.array([3, 3, 3, 2, 2, 0]), np.array([3, 3, 3, 3, 3, 0])
    check('G classify: compatible / contradicted / timing_sensitive / unidentified (missing instants, empty window)',
          P.classify(tr, va, n).tolist() == [0, 1, 2, 3, 3, 3], P.classify(tr, va, n).tolist())
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 997)
    x[rng.integers(0, 997, 40)] = np.nan
    mn, mx = P.sparse(x, np.fmin), P.sparse(x, np.fmax)
    i0 = rng.integers(0, 997, 500); i1 = np.minimum(i0 + rng.integers(0, 300, 500), 996)
    bm = np.array([np.nanmin(x[a:b + 1]) if np.isfinite(x[a:b + 1]).any() else np.nan for a, b in zip(i0, i1)])
    bx = np.array([np.nanmax(x[a:b + 1]) if np.isfinite(x[a:b + 1]).any() else np.nan for a, b in zip(i0, i1)])
    check('G range min / max equal brute force', np.allclose(P.rq(mn, i0, i1, np.fmin), bm, equal_nan=True)
          and np.allclose(P.rq(mx, i0, i1, np.fmax), bx, equal_nan=True))
    acc = P.Acc(1)
    gm = np.zeros((len(P.GROUPS), 2), bool); gm[0] = True
    xc = np.zeros((2, P.N_DRAWS)); xc[1, 0] = np.nan
    acc.add(0, P.Acc.masks(gm, np.array([0, 0])), 0, 0, 0, 0, np.array([1., 2.]), xc)
    acc.ne[0, 0, 0, 0, 0] = 2
    SM = P.summarize(acc, (DAY_A,), CC)
    r = pick(SM, view='S_FIFO', group='cancel_possible', control='presence', scope='rank1', day=DAY_A, D='0', feature=P.FEATS[0])
    check('G statistics by hand through null_summary: obs 1.5, null 0, p_upper 1/201, p_lower 1, n_ctl_valid 399, 200 replicates',
          r['obs'] == 1.5 and r['null_mean'] == 0 and abs(r['p_upper'] - 1 / 201) < 1e-12 and r['p_lower'] == 1. and _i(r['n_ctl_valid']) == 399
          and _i(r['n_reps']) == 200, None if isinstance(r, Missing) else r.to_dict())
    rr = np.random.default_rng(11)
    o = rr.normal(0, 1, (400, 30)).mean(1)
    reps = rr.normal(0, 1, (400, 30, P.N_DRAWS)).mean(1)
    pu = np.array([CC.null_summary(o[i], reps[i])['p_upper'] for i in range(400)])
    pl = np.array([CC.null_summary(o[i], reps[i])['p_lower'] for i in range(400)])
    fu, fl_ = int((pu <= 0.05).sum()), int((pl <= 0.05).sum())
    check('G calibration: under exchangeability P(p <= 0.05) is ~5 % in each tail (8..36 of 400; null_summary)', 8 <= fu <= 36 and 8 <= fl_ <= 36,
          (fu, fl_))


if __name__ == '__main__':
    main()
