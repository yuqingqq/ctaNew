"""TAKE-STREAM v1 (declaration.md): takes decided on the Binance futures stream (STREAM), on Binance futures best-level depletion
(DEPLETION) or on the frozen 50 ms grid (GRID comparator), under the venue-matched harness of take_exit_venue_20261125 (vm_replay).
A trigger's decision time = its Binance event time + 1.5 ms feed + 2 ms decision; the take (a MARKET order at the known best, zero
slippage = one level at the limit or better, the rest cancelled) reaches the book at decision + network + 150 ms.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_stream_20261125/stream_replay.py \
       selftest | triggers <day> | day <day> | merge | report
"""
import collections
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
sys.path.insert(0, str(H.parent / 'take_exit_venue_20261125'))
import vm_replay as VR  # noqa: E402  (the venue-matched harness, frozen at e5ffa314e / 33a0e19d5)

V, F, OD, MM, ER, TR, X, LB, BE, C = VR.V, VR.F, VR.OD, VR.MM, VR.ER, VR.TR, VR.X, VR.LB, VR.BE, VR.C
DATA = F.X.DATA.parents[1]
OUT = X.DATA / 'take_stream_20261125'
FEED, DECIDE, HOLD = 1.5, 2.0, 150.
DEPL_W, DEPL_FRAC, COOL, QUIET = 300., .5, 2000., 2000.
DEPLS_MIN_BASE = 12.609                    # strict variant: the base level >= the p75 best-level size of 2026-10-24 (sizes only, outside the test days)
EDGES = TR.T.P.frozen_edges()
UP_EDGE, DN_EDGE = float(EDGES[1][-1]), float(EDGES[0][-1])
VMCFG = dict(VR.CONFIGS['VM208-M50'], delay=172.)                              # the venue-matched harness at 172 ms (GRID)
CONFIGS = {'GRID-172': dict(kinds=None, net=22.), 'STREAM-172': dict(kinds=('stream',), net=22.), 'STREAM-192': dict(kinds=('stream',), net=42.),
           'STREAM+DEPL-172': dict(kinds=('stream', 'depl'), net=22.), 'STREAM+DEPLS-172': dict(kinds=('stream', 'depls'), net=22.),
           'STREAM+DEPLS-192': dict(kinds=('stream', 'depls'), net=42.)}
DAY_CPU_CAP, TOTAL_CPU_CAP, FIRST_DAY, PARITY_EVERY = 1200., 3600., '20261023', 40
SAME = VR.SAME
FILES = ['declaration.md', 'stream_replay.py']
EPS = V.EPS


# ---------------------------------------------------------------------------------------------------------------- triggers
def stream_from_changes(bk, bm):
    """OFF->ON transitions of the frozen thresholds on a mid stream (change points bk / bm, event time) evaluated at every breakpoint
    (change points and change points + 250 ms), with no ON on that side in the 2 s before -> [(event_ms, dir)], dir +1 up / -1 down."""
    bp = np.unique(np.r_[bk, bk + 250.])
    j0, j1 = np.searchsorted(bk, bp, 'right') - 1, np.searchsorted(bk, bp - 250., 'right') - 1
    ok = (j0 >= 0) & (j1 >= 0)
    r = np.where(ok, 1e4 * np.log(bm[np.maximum(j0, 0)] / bm[np.maximum(j1, 0)]), np.nan)
    out = []
    for d, on in ((1, ok & (r >= UP_EDGE)), (-1, ok & (-r >= DN_EDGE))):
        last_end, prev = -np.inf, False
        for i in range(len(bp)):
            if on[i]:
                if not prev and bp[i] - last_end >= QUIET:
                    out.append((float(bp[i]), d))
                last_end = float(bp[i + 1]) if i + 1 < len(bp) else np.inf       # this ON period ends at the next breakpoint
            prev = bool(on[i])
    return sorted(out)


def depletion_from_book(t, b, bq, a, aq, lo=-np.inf, min_base=0.):
    """Best ask (bid) at an unchanged price at <= DEPL_FRAC of its size DEPL_W ms earlier (that size >= min_base) -> up (down) candidates at t >= lo."""
    j = np.searchsorted(t, t - DEPL_W, 'right') - 1
    ok = (j >= 0) & (t >= lo)
    jj = np.maximum(j, 0)
    up = ok & (a == a[jj]) & (aq[jj] > max(0., min_base - 1e-9)) & (aq <= DEPL_FRAC * aq[jj])
    dn = ok & (b == b[jj]) & (bq[jj] > max(0., min_base - 1e-9)) & (bq <= DEPL_FRAC * bq[jj])
    return [(float(x), 1) for x in t[up]] + [(float(x), -1) for x in t[dn]]


def cooldown(cands):
    last, out = {1: -np.inf, -1: -np.inf}, []
    for x, d in sorted(cands):
        if x - last[d] >= COOL:
            out.append((x, d)); last[d] = x
    return out


def load_book_hour(day, h, carry):
    """Binance futures best levels of one hour (event time E), the previous hour's last second carried in front (same day only)."""
    f = DATA / f'mm_hf/raw/bookTicker/BTCUSDT/{day}_{h:02d}.csv.gz'
    bt = pd.read_csv(f, header=None, usecols=[1, 4, 5, 6, 7], names=['t', 'b', 'bq', 'a', 'aq'],
                     dtype={1: np.int64, 4: np.float64, 5: np.float32, 6: np.float64, 7: np.float32}) if f.exists() else pd.DataFrame(columns=['t', 'b', 'bq', 'a', 'aq'])
    if carry is not None:
        bt = pd.concat([carry, bt])
    return bt.sort_values('t', kind='stable')


def build_triggers(day):
    fn = OUT / f'triggers_{day}.parquet'
    C.require(not fn.exists(), f'{fn.name} exists (write-once)')
    OUT.mkdir(parents=True, exist_ok=True)
    bk, bm = OD.seg_arrays(OD.SEG_OF[day]); bk = bk.astype(float)
    d0 = int(pd.Timestamp(day, tz='UTC').value // 10**6); d1 = d0 + 86400000
    st = [(x, d, 'stream') for x, d in stream_from_changes(bk, bm) if d0 <= x < d1]
    cands, cands_s, carry = [], [], None
    for h in range(24):
        bt = load_book_hour(day, h, carry)
        t = bt.t.to_numpy(float); arr = (bt.b.to_numpy(float), bt.bq.to_numpy(float), bt.a.to_numpy(float), bt.aq.to_numpy(float))
        cands += depletion_from_book(t, *arr, lo=d0 + h * 3600000)
        cands_s += depletion_from_book(t, *arr, lo=d0 + h * 3600000, min_base=DEPLS_MIN_BASE)
        carry = bt[bt.t >= d0 + (h + 1) * 3600000 - 1000]
    dp = [(x, d, 'depl') for x, d in cooldown(cands) if d0 <= x < d1] + [(x, d, 'depls') for x, d in cooldown(cands_s) if d0 <= x < d1]
    T = pd.DataFrame(st + dp, columns=['t_event', 'dir', 'kind']).sort_values(['t_event', 'dir'], ascending=[True, False], kind='stable')
    T.to_parquet(fn, index=False)
    print(day, dict(collections.Counter(T.kind)), flush=True)


# ---------------------------------------------------------------------------------------------------------------- the branch
class StreamBranch(VR.VMBranch):
    """VMBranch whose takes come from a decision-time list: at grid tick g it sends every take decided in [g, g + 50 ms), each on the state
    known at g (never later than its decision) and the Predict book known at its decision; act = decision + network + 150 ms."""

    def __init__(self, path, end, ident, bps, host, take, slack=0., cfg=None, stream=None):
        super().__init__(path, end, ident, bps, host, take, slack=slack, cfg=cfg)
        self.stream, self.si = stream, 0
        self.settings = dict(self.settings, take_stream=dict(n=int(len(stream[0])), feed_ms=FEED, decide_ms=DECIDE))

    def signal_step(self, k, g):
        if not self.signal_take:
            return
        Td, Dd, Kd = self.stream
        f = self.E.f
        while self.si < len(Td) and Td[self.si] < g + BE.STEP:
            t, dd, kind = float(Td[self.si]), int(Dd[self.si]), str(Kd[self.si]); self.si += 1
            if t < g:                                                          # decided before this tick (cannot happen with the loop below)
                continue
            gs = 1 if dd == 1 else 0
            s = 0 if gs == 1 else 1
            I, B, A = self.reserve(g)
            room = self.cap - I - B if s == 0 else self.cap + I - A
            q = float(math.floor(min(TR.TAKE_Q, room) + TR.EPS))
            row = dict(t=t, k=int(k), guard_side=gs, side=s, known_I=I, B=B, A=A, room=room, q=q, status='candidate', gid=None)
            if q < 1:
                row['status'] = 'no_room'
            elif self.take_open:
                row['status'] = 'pending_ioc'
            elif t + TR.TAKE_DELAY >= self.end:
                row['status'] = 'near_end'
            else:
                jk = int(np.searchsorted(f[:, 1], t, side='right') - 1)
                fk = f[max(jk, 0)]
                limit = float(fk[12]) if s == 0 else float(fk[2])
                opp = self.side_view(g, s)['opp_prices']
                row.update(limit=limit, book_received_ms=float(fk[1]), known_frame=jk)
                if jk < 0 or not LB.book_ok_side(fk, s) or t - fk[1] > 1000:
                    row['status'] = 'no_fresh_book'
                elif any(limit >= p_ if s == 0 else limit <= p_ for p_ in opp):
                    row['status'] = 'own_cross'
                elif self.native:
                    red = min(q, max(0., -I if s == 0 else I)); op = q - red; pn = limit if s == 0 else 100. - limit
                    if op > EPS and op * pn < VR.MIN_BUY_C - 1e-9:
                        if red >= VR.MIN_MSELL:
                            q = red; row.update(q=q, native='min_capped')
                        else:
                            row['status'] = 'below_min'
                if row['status'] == 'candidate':
                    gid = self.E.submit_ioc(t, s, q, limit)
                    self.take_open.append(gid)
                    row.update(gid=gid, status='sent')
            row.update(kind=kind)
            self.sig_log.append(row)
        self.check_position(g)


def one_stream(p, cache, old, edges, bps, slack, stream, net):
    orig, d0 = TR.SignalBranch, TR.TAKE_DELAY
    TR.SignalBranch = lambda *a, **k: StreamBranch(*a, slack=slack, cfg=VMCFG, stream=stream, **k)
    TR.TAKE_DELAY = float(net + HOLD)
    del F.LAST[:]
    try:
        rec = ER.run_one(p, cache, old, edges, bps, 'PASSIVE_EXIT')
    finally:
        TR.SignalBranch, TR.TAKE_DELAY = orig, d0
    C.require(len(F.LAST) == 1, 'one branch per run')
    C.require(all(abs(v['act'] - v['submit'] - net - HOLD) < 1e-9 for v in rec['iocs'].values()), 'a take arrival other than decision + network + hold')
    rec['floor_log'] = TR.clean(F.LAST[0].floor_log)
    rec['stream_cfg'] = dict(net=float(net), feed=FEED, decide=DECIDE, n_triggers=int(len(stream[0])))
    del F.LAST[:]
    return json.loads(json.dumps(TR.clean(rec), allow_nan=False))


def grid_stream(cache, edges, end):
    """The frozen grid triggers as a decision-time list (decision = the grid tick; up before down at one tick), for the K1 parity."""
    n = int(np.ceil(end / BE.STEP))
    trig = TR.triggers(TR.on_arrays(cache, edges, n))
    ev = sorted([(k * BE.STEP, 1) for k in np.flatnonzero(trig[1])] + [(k * BE.STEP, -1) for k in np.flatnonzero(trig[0])], key=lambda z: (z[0], -z[1]))
    return (np.array([x for x, _ in ev], float), np.array([d for _, d in ev], int), np.array(['grid'] * len(ev)))


# ---------------------------------------------------------------------------------------------------------------- day run
def bound():
    for f in FILES:
        C.require(subprocess.check_output(['git', 'show', 'HEAD:' + str((H / f).relative_to(R))], cwd=R) == (H / f).read_bytes(), f'{f} is not committed')
    VR.bound()


def take_value(rec):
    acc = collections.defaultdict(lambda: [0., 0.])
    for x in rec['take_marks']:
        for h in ('m1s_c', 'm10s_c'):
            if x[h] is not None:
                acc[h][0] += x[h] * x['q']; acc[h][1] += x['q']
    return {h: v for h, v in acc.items()}


def day_run(day):
    resource.setrlimit(resource.RLIMIT_CPU, (int(DAY_CPU_CAP) + 60, int(DAY_CPU_CAP) + 61))
    bound()
    C.require(day in OD.DAYS3, f'{day} is not a declared day')
    fn = OUT / f'day_{day}.json'
    C.require(not fn.exists(), f'{fn.name} exists (write-once)')
    if day != FIRST_DAY:
        pj = json.loads((H / 'projection.json').read_text())
        C.require(pj['projected_cpu_s'] <= TOTAL_CPU_CAP, 'the first-day projection exceeds the cap')
    vs = json.loads((VR.OUT / 'summary.json').read_text())
    comp = {int(r['mid']): r['payout'] for r in vs['rows'] if r['day'] == day and r['status'] == 'complete'}
    T = pd.read_parquet(OUT / f'triggers_{day}.parquet')
    tdec = T.t_event.to_numpy(float) + FEED + DECIDE
    edges, src = TR.T.P.frozen_edges(), MM.KP.Tape()
    reg = MM.registry(day).set_index('mid'); off = src.offset(day); t0c = time.process_time()
    rows, parity, n = [], [], 0
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in comp]
        if not mids:
            continue
        for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
            C.require(lst == 'loaded', f'{m} {lst}')
            ws = float(reg.loc[m, 'ws']) * 1000.
            p, st, info = VR.vm_path(W, ws, 50.)
            C.require(st == 'complete', f'{m} {st}')
            t0, end = float(info['t_start']), float(info['end_rel'])
            cache = OD.make_cache3(t0, end, OD.SEG_OF[day])
            old = dict(end=end, ident=MM.ident_of(day, m, ws, info, p))
            y = comp[m]
            row = dict(day=day, mid=m, hour=int(h), payout=y, arms={})
            for name, cf in CONFIGS.items():
                if cf['kinds'] is None:
                    rec = VR.one_vm(p, cache, old, edges, 200., V.BASE_SLACK, VMCFG)
                else:
                    sel = (tdec >= t0) & (tdec < t0 + end) & np.isin(T.kind.values, cf['kinds'])
                    stream = (tdec[sel] - t0, T.dir.values[sel].astype(int), T.kind.values[sel])
                    rec = one_stream(p, cache, old, edges, 200., V.BASE_SLACK, stream, cf['net'])
                rec.update(day=day, mid=m, config=name)
                sha = F.write_once_gz(OUT / f'{m}_{name}.json.gz', rec)
                a = VR.native_account(rec, y)
                a.update(sha256=sha, sig_status=dict(collections.Counter(r_['status'] for r_ in rec['signal_log'])),
                         sig_kind_sent=dict(collections.Counter(r_.get('kind', 'grid') for r_ in rec['signal_log'] if r_['status'] == 'sent')),
                         ioc_status=rec['metrics']['ioc_status'], take_value=take_value(rec))
                row['arms'][name] = a
                if name == 'GRID-172' and n % PARITY_EVERY == 0:                   # K1: grid triggers through the stream branch == GRID-172
                    r1 = one_stream(p, cache, old, edges, 200., V.BASE_SLACK, grid_stream(cache, edges, end), 22.)
                    r1['signal_log'] = [{k_: v_ for k_, v_ in x.items() if k_ != 'kind'} for x in r1['signal_log']]
                    parity.append(dict(mid=m, same={k_: r1[k_] == rec[k_] for k_ in SAME}))
            n += 1
            rows.append(row)
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    spent = time.process_time() - t0c
    if day == FIRST_DAY:
        pfn = H / 'projection.json'
        C.require(not pfn.exists(), 'projection.json exists (write-once)')
        TR.dump(pfn, dict(asof='2026-11-25', first_day=day, spent_cpu_s=spent, markets=len(rows), projected_cpu_s=spent * 408 / max(len(rows), 1) * 1.1,
                          cap_cpu_s=TOTAL_CPU_CAP))
    TR.dump(fn, dict(asof='2026-11-25', day=day, rows=rows, parity=parity, cpu_s=spent, peak_rss_mib=F.rss_mib()))
    print(f'{day}: {len(rows)} markets, K1 parity {all(all(x["same"].values()) for x in parity)} ({len(parity)}), cpu {spent:.0f} s', flush=True)


# ---------------------------------------------------------------------------------------------------------------- selftest
def selftest():
    # stream: flat, +1 bps at 10 s (trigger), +1 bps at 11 s (ON again 750 ms after the last ON ended: no trigger), +1 bps at 14 s (trigger)
    bk = np.array([0., 10000., 11000., 14000.]); bm = 100000. * np.exp(np.array([0., 1e-4, 2e-4, 3e-4]))
    tr = stream_from_changes(bk, bm)
    assert tr == [(10000., 1), (14000., 1)], tr
    # a down move needs -0.4667 bps: -0.3 bps does not trigger, -0.5 bps does
    bm2 = 100000. * np.exp(np.array([0., -0.3e-4, -0.3e-4, -0.8e-4]))
    assert stream_from_changes(bk, bm2) == [(14000., -1)], stream_from_changes(bk, bm2)
    # depletion: ask 45 x 10 -> x 4 within 300 ms at the same price (up), then x 1 (cooldown), bid 44 x 10 -> 6 (not halved)
    t = np.array([0., 100., 250., 400., 500.]); a = np.full(5, 45.); aq = np.array([10., 10., 4., 1., 1.]); b = np.full(5, 44.); bq = np.array([10., 10., 10., 6., 6.])
    c = cooldown(depletion_from_book(t, b, bq, a, aq))
    assert c == [(400., 1)], c                                           # the base needs an update 300 ms earlier: first possible at 400
    # a price change at the best is not a same-price depletion
    a2 = np.array([45., 45., 46., 46., 46.]); aq2 = np.array([10., 10., 4., 4., 4.])
    assert cooldown(depletion_from_book(t, b, bq, a2, aq2)) == [], 'price change counted as depletion'
    assert cooldown(depletion_from_book(t, b, bq, a, aq, min_base=12.)) == [], 'a 10-share base passed a 12-share floor'
    print('stream selftest PASS')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'triggers':
        build_triggers(sys.argv[2])
    elif cmd == 'day':
        day_run(sys.argv[2])
    else:
        dict(selftest=selftest)[cmd]()
