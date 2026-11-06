"""Ladder market-making replay (2026-11-03; strategy logic in docs/EXPLORATION_2026-11-01.md §35).

Many resting orders per side, each with its own queue position, at the measured Tokyo clock (pf_clock_rl with
PF_RL_DDEC=2 features): Predict frames known per frame at Tokyo's 53.1 ms median + 2 ms decision, Binance E + 3.5 ms;
orders live 40 ms after sending (post-only checked then), resting cancels effective 22 ms after sending, a cancel sent
before the order is live effective 95 ms after its create; own fills known after L_FILL (100 ms; PF_LFILL overrides);
venue throttle (one bucket of 40 refilling 500 / min; cancels always sent; a place needs tokens >= open orders + 2);
1 USD minimum. Fill model as pf_step4 v2 extended to several own orders: queue = displayed size at our price in the frame
strictly before activation (the 5th shown level's size when our price is beyond the 5 shown levels), lowered when a later
frame shows less there; a trade at our price consumes the queue ahead first; a trade through our price fills us; own
orders on the hit side are processed best price first, then oldest first, sharing the trade's volume.
Arms: PF_LADDER_ARMS = "K=3,z=10,S=60;K=6,z=5,S=240;..." (defaults below). Predictions: PF_PREDS (pred_1000).
Output: step4/ladder_<PF_PREDS>_<PF_TAG>/<day>/{summary.json, win/<arm>.parquet, fills/<arm>.parquet}; with PF_ORDLOG=1 also
orders/<arm>.parquet (one row per order: queue at activation, life, end reason, queue/swept fill; logging only, no effect on P&L)
Usage: PF_PREDS=preds_o3 PF_RL_DDEC=2 python3 -m predictfun.research.pf_ladder <day>     (selftest: ... pf_ladder --selftest)
"""
import sys, os, bisect, heapq, json, itertools, collections
import numpy as np, pandas as pd

DEF = dict(K=3, z=10.0, S=60.0, CAP=50.0, BUD=75.0, RF=1000.0, THIN=30.0, CD=500.0, TH=0.25, BIGBP=2.0, BIGM=2.0)
L_PL, L_CA, L_PEND = 40.0, 22.0, 95.0


def parse_arms(s):
    arms = []
    for spec in s.split(';'):
        a = dict(DEF)
        for kv in spec.split(','):
            k, v = kv.split('=')
            a[k] = int(v) if k == 'K' else float(v)
        a['name'] = ','.join(f'{k}={a[k]:g}' if k != 'K' else f'K={a[k]}' for k in ('K', 'z', 'S', 'TH', 'CAP', 'BUD', 'RF', 'THIN', 'CD'))
        arms.append(a)
    return arms


def disp(f, side, p):
    pre = 'b' if side == 0 else 'a'
    for i in range(1, 6):
        if f[f'{pre}{i}_p'] == p:
            return f[f'{pre}{i}_s']
    last = f[f'{pre}5_p']
    beyond = last >= 0 and ((side == 0 and p < last) or (side == 1 and p > last))
    return f[f'{pre}5_s'] if beyond else 0.0                # beyond the shown levels: assume a 5th-level-sized queue


def disp_rng(f, side, p):
    pre = 'b' if side == 0 else 'a'; last = f[f'{pre}5_p']
    if last >= 0 and ((side == 0 and p < last) or (side == 1 and p > last)):
        return None
    for i in range(1, 6):
        if f[f'{pre}{i}_p'] == p:
            return f[f'{pre}{i}_s']
    return 0.0


def match_leg(cands, side, pt, sh, F, tf, tt):
    """Fill our live orders on the hit side for one trade leg; returns [(order, q, through)]. cands = live orders on side."""
    out = []
    order = sorted(cands, key=(lambda o: (-o['pt'], o['t_act'])) if side == 0 else (lambda o: (o['pt'], o['t_act'])))
    j = bisect.bisect_left(tf, tt) - 1                                  # frames strictly before the trade
    sh_left = sh
    for o in order:
        if sh_left <= 1e-12:
            break
        for k in range(o['ci'] + 1, j + 1):
            d = disp_rng(F[k], side, o['pt'])
            if d is not None and d < o['ahead']:
                o['ahead'] = d
        if j > o['ci']:
            o['ci'] = j
        through = (side == 0 and pt < o['pt']) or (side == 1 and pt > o['pt'])
        if pt == o['pt']:
            use = min(sh_left, o['ahead']); o['ahead'] -= use; q = min(sh_left - use, o['rem'])
        elif through:
            q = min(sh_left, o['rem']); o['ahead'] = 0.0
        else:
            q = 0.0
        if q > 1e-12:
            o['rem'] -= q; sh_left -= q; out.append((o, q, through))
    return out


def selftest():
    """Hand-built cases for the matching rule (must pass before any real run)."""
    fr = dict({f'{s}{i}_{k}': -1.0 for s in 'ba' for i in range(1, 6) for k in 'ps'})
    f0 = dict(fr, b1_p=50, b1_s=100.0, b2_p=49, b2_s=40.0, b3_p=48, b3_s=30.0, a1_p=51, a1_s=80.0)
    F = [f0]; tf = [0.0]
    mk = lambda p, ahead, rem, ta: dict(pt=p, ahead=ahead, rem=rem, ci=0, t_act=ta)
    # 1) queue at our price: 100 ahead, a 120-share sell at 50 fills 20 of our 25
    o = mk(50, 100.0, 25.0, 1.0); r = match_leg([o], 0, 50, 120.0, F, tf, 10.0)
    assert len(r) == 1 and abs(r[0][1] - 20) < 1e-9 and not r[0][2] and abs(o['ahead']) < 1e-9, r
    # 2) a sweep to 48 fills our 49 bid (through) up to the leg's volume; our 47 bid untouched
    a, b = mk(49, 40.0, 10.0, 1.0), mk(47, 0.0, 10.0, 1.0); r = match_leg([a, b], 0, 48, 6.0, F, tf, 10.0)
    assert len(r) == 1 and r[0][0] is a and abs(r[0][1] - 6) < 1e-9 and r[0][2], r
    # 3) two of our bids hit by one through-leg share its volume, better price first
    a, b = mk(50, 0.0, 10.0, 2.0), mk(49, 0.0, 10.0, 1.0); r = match_leg([b, a], 0, 48, 15.0, F, tf, 10.0)
    assert [x[0] for x in r] == [a, b] and abs(r[0][1] - 10) < 1e-9 and abs(r[1][1] - 5) < 1e-9, r
    # 4) a later frame showing less size at our price shortens the queue; asks mirror
    f1 = dict(f0, a1_s=30.0); F2 = [f0, f1]; tf2 = [0.0, 5.0]
    o = mk(51, 80.0, 10.0, 1.0); r = match_leg([o], 1, 51, 35.0, F2, tf2, 10.0)
    assert abs(r[0][1] - 5) < 1e-9 and not r[0][2], r
    # 5) a trade at a worse price for the maker (sell at 49) does not touch a 48 bid
    o = mk(48, 0.0, 10.0, 1.0); assert match_leg([o], 0, 49, 50.0, F, tf, 10.0) == []
    # 6) our price beyond the 5 shown levels: queue = 5th level's size
    f5 = dict(fr, b1_p=50, b1_s=1, b2_p=49, b2_s=1, b3_p=48, b3_s=1, b4_p=47, b4_s=1, b5_p=46, b5_s=77.0)
    assert disp(f5, 0, 44) == 77.0 and disp(f5, 0, 47) == 1 and disp_rng(f5, 0, 44) is None
    print('selftest PASS (6 matching cases)')


if __name__ == '__main__' and '--selftest' in sys.argv:
    selftest(); sys.exit(0)

from . import pf_io as io
from . import pf_clock_rl as CK

day = sys.argv[1]; S1 = f'{io.OUT}/step1/{day}'
PREDS = os.environ['PF_PREDS']; TAG = os.environ.get('PF_TAG', 'ladder')
OUT = f'{io.OUT}/step4/ladder_{PREDS}_{TAG}/{day}'; os.makedirs(f'{OUT}/win', exist_ok=True); os.makedirs(f'{OUT}/fills', exist_ok=True)
ARMS = parse_arms(os.environ['PF_LADDER_ARMS'])
L_FILL = float(os.environ.get('PF_LFILL', '100'))
B_IN = CK.BN_IN + CK.D_DEC
SHIFT = CK.load_shift(S1)

reg = pd.read_parquet(f'{S1}/registry.parquet').set_index('mid')
LT = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2'])
use = (LT.how.values == 0) & np.isfinite(LT.t_e2.values)
LT['t_ms'] = np.where(use, LT.t_e2.values, LT.t_ms.values); LT['how'] = np.where(use, 3, LT.how.values)
LT_BY = {m: g for m, g in LT.groupby('mid', sort=False)}
pr = pd.read_parquet(f'{io.OUT}/step5/{day}/{PREDS}.parquet', columns=['mid', 'tau', 'pred_1000'])
pr = pr[pr.pred_1000.notna()]
PRED = {m: (g.tau.values.astype(np.float64), g.pred_1000.values.astype(np.float64)) for m, g in pr.groupby('mid', sort=False)}
del pr
cols = ['mid', 't_ms', 'recv_ns'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 6) for k in 'ps']
books = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=cols) for f in sorted(os.listdir(f'{S1}/books'))], ignore_index=True)
books.sort_values(['mid', 't_ms', 'recv_ns'], inplace=True, kind='stable')
kb, mb, cb = [], [], []; carry_k = np.empty(0); carry_m = np.empty(0); last_m = np.nan; last_c = 99
for h in io.hours_of(day, extra_after=1, extra_before=1):
    p_ = f'{io.BN_DIR}/{h}.csv.gz'
    if not os.path.exists(p_):
        continue
    df = pd.read_csv(p_, header=None, names=['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq'], usecols=['E', 'u', 'bid', 'ask'])
    df = df.apply(pd.to_numeric, errors='coerce').dropna(); df = df[(df.bid > 0) & (df.ask > df.bid)].drop_duplicates('u', keep='first')
    k = df.E.values + B_IN; m = (df.bid.values + df.ask.values) / 2; del df
    o = np.argsort(k, kind='stable'); k = k[o].astype(np.float64); m = m[o].astype(np.float64)
    K_ = np.concatenate([carry_k, k]); M_ = np.concatenate([carry_m, m])
    j = np.searchsorted(K_, k - 250.0, side='right') - 1
    bp = np.where(j >= 0, 1e4 * np.log(m / M_[np.clip(j, 0, None)]), np.nan)
    c = np.where(bp <= -0.5, -1, np.where(bp >= 0.5, 1, 0)).astype(np.int8)
    keep = (m != np.r_[last_m, m[:-1]]) | (c != np.r_[last_c, c[:-1]])
    kb.append(k[keep]); mb.append(m[keep]); cb.append(c[keep]); last_m = m[-1]; last_c = c[-1]
    tail = k >= k[-1] - 1000.0; carry_k = k[tail]; carry_m = m[tail]
    io.mem_check(f'bn {h}')
KB = np.concatenate(kb); MB = np.concatenate(mb); BC = np.concatenate(cb); o = np.argsort(KB, kind='stable'); KB, MB, BC = KB[o], MB[o], BC[o]
del kb, mb, cb
print(f'[in] {day} books {len(books)} legs {len(LT)} binance kept {len(KB)} [mem] {io.rss_mib()} MiB', flush=True)


def bn_state(t):
    """(g1 small bid/ask, g1 big bid/ask, g2 bid/ask) adverse flags at knowledge time t."""
    i = int(np.searchsorted(KB, t, side='right')) - 1
    if i < 0:
        return (False,) * 6
    s1b = s1a = b1b = b1a = False
    if t - KB[i] <= 1000.0:
        j = int(np.searchsorted(KB, t - 1000.0, side='right')) - 1
        if j >= 0:
            mv = (MB[i] / MB[j] - 1.0) * 1e4
            s1b, s1a, b1b, b1a = mv <= -0.5, mv >= 0.5, mv <= -2.0, mv >= 2.0
    return s1b, s1a, b1b, b1a, BC[i] == -1, BC[i] == 1


def bn_flips(t0, t1):
    a = np.searchsorted(KB, t0 - 1000.0); b = np.searchsorted(KB, t1)
    cand = KB[a:b]; cand = np.concatenate([cand, cand + 1000.0, cand + 1000.0 + 1e-6])
    cand = np.unique(np.concatenate([[t0], cand[(cand >= t0) & (cand <= t1)]]))
    st = [bn_state(x) for x in cand]
    return [x for x, p_, q_ in zip(cand[1:], st[:-1], st[1:]) if p_ != q_]


def model_value(mid, t):
    v = PRED.get(mid)
    if v is None:
        return None
    i = int(np.searchsorted(v[0], t, side='right')) - 1
    if i < 0 or t - v[0][i] > 100.0:
        return None
    return v[1][i]


def model_flips(mid, t0, t1, th, big):
    v = PRED.get(mid)
    if v is None:
        return []
    a = np.searchsorted(v[0], t0); b = np.searchsorted(v[0], t1); tt = v[0][a:b]; pp = v[1][a:b]
    st = np.where(pp <= -big, -2, np.where(pp <= -th, -1, np.where(pp >= big, 2, np.where(pp >= th, 1, 0))))
    out = tt[np.r_[True, st[1:] != st[:-1]]].tolist()
    if len(tt):
        gap = np.r_[np.diff(tt) > 100.0, True]; out += (tt[gap] + 100.0 + 1e-3).tolist() + tt[1:][gap[:-1]].tolist()
    return out


def run_window(mid, F, kf, A):
    R = reg.loc[mid]
    if pd.isna(R.outcome) or R.outcome not in ('Up', 'Down'):
        return None
    y = 1.0 if R.outcome == 'Up' else 0.0; ws = float(R.ws) * 1000.0; S = A['S']; t_stop = ws + S * 1000.0
    tf = [r['t_ms'] for r in F]
    g = LT_BY.get(mid)
    trades = [] if g is None else sorted(zip(np.where(g.how.values > 0, g.t_ms.values, g.exec_s.values * 1000.0).tolist(),
                                             g.side.tolist(), g.ptick.tolist(), g.shares.tolist()))
    ev = []; seq = itertools.count()
    for i, k in enumerate(kf):
        if ws <= k <= t_stop:
            ev.append((k, 2, next(seq), 'D', i))
    for t in bn_flips(ws, t_stop) + model_flips(mid, ws, t_stop, A['TH'], A['BIGM']) + [t_stop]:
        ev.append((t, 2, next(seq), 'D', None))
    for tr in trades:
        if ws <= tr[0] < t_stop + 200.0:
            ev.append((tr[0], 1, next(seq), 'T', tr))
    heapq.heapify(ev)
    orders = {0: [], 1: []}; net = knet = cash = 0.0; fills = []; requests = []; rf = {}; cdu = [-np.inf, -np.inf]; prev = [None, None]
    tok = dict(v=40.0, t=ws); ctr = dict(place=0, cancel=0, refused_post=0, throttled=0, pulls=collections.Counter())
    max_rest = [0.0, 0.0]; max_absnet = 0.0; allo = []

    def refill(t):
        if t > tok['t']:
            tok['v'] = min(40.0, tok['v'] + 500.0 / 60.0 * (t - tok['t']) / 1000.0); tok['t'] = t

    def n_open():
        return sum(1 for s_ in (0, 1) for o in orders[s_] if not o['done'])

    def cancel(o, t, why):
        if o['cancel_sent'] or o['done']:
            return
        refill(t); tok['v'] -= 1.0; ctr['cancel'] += 1; ctr['pulls'][why] += 1; requests.append(t)
        o['cancel_sent'] = True; o['why'] = why
        o['t_rm'] = t + L_CA if o['live'] else max(t + L_CA, o['t_send'] + L_PEND)
        heapq.heappush(ev, (o['t_rm'], 0, next(seq), 'X', o))

    def known_frame(t):
        i = bisect.bisect_right(kf, t) - 1
        return F[i] if i >= 0 else None

    def decide(t):
        rel = (t - ws) / 1000.0
        f = known_frame(t)
        if rel >= S or f is None or f['b1_p'] < 0 or f['a1_p'] <= f['b1_p']:
            for s_ in (0, 1):
                for o in orders[s_]:
                    cancel(o, t, 'stop' if rel >= S else 'nobook')
            return
        b1, a1, bs1, as1 = f['b1_p'], f['a1_p'], f['b1_s'], f['a1_s']
        for s_, cur in ((0, b1), (1, a1)):
            if prev[s_] is not None and ((s_ == 0 and cur < prev[s_]) or (s_ == 1 and cur > prev[s_])):
                cdu[s_] = t + A['CD']; heapq.heappush(ev, (t + A['CD'] + 1e-3, 2, next(seq), 'D', None))
            prev[s_] = cur
        s1b, s1a, b1b, b1a, g2b, g2a = bn_state(t); v = model_value(mid, t)
        for side in (0, 1):
            best = b1 if side == 0 else a1; dsz = bs1 if side == 0 else as1
            near = (s1b or g2b or (v is not None and v <= -A['TH'])) if side == 0 else (s1a or g2a or (v is not None and v >= A['TH']))
            big = (b1b or (v is not None and v <= -A['BIGM'])) if side == 0 else (b1a or (v is not None and v >= A['BIGM']))
            inv = (knet >= A['CAP']) if side == 0 else (knet <= -A['CAP'])
            thin = dsz < A['THIN']; moved = t < cdu[side]
            live_side = [o for o in orders[side] if not o['done'] and not o['cancel_sent']]
            for o in live_side:
                off = (best - o['pt']) if side == 0 else (o['pt'] - best)
                if off < 0:
                    cancel(o, t, 'stale')
                elif big:
                    cancel(o, t, 'big')
                elif inv:
                    cancel(o, t, 'inv')
                elif off <= 1 and near:
                    cancel(o, t, 'near')
                elif off == 0 and (thin or moved):
                    cancel(o, t, 'thin' if thin else 'moved')
                elif off > A['K'] + 2:
                    cancel(o, t, 'deep')
            if big or inv:
                continue
            keep = [o for o in orders[side] if not o['done'] and not o['cancel_sent']]
            resting = sum(o['rem'] for o in keep); have = {o['pt'] for o in keep}
            max_rest[side] = max(max_rest[side], resting)
            for d in range(1, A['K'] + 1):
                if d == 1 and near:
                    continue
                p = best - d if side == 0 else best + d
                if not (1 <= p <= 99) or p in have or t < rf.get((side, p), -np.inf):
                    continue
                if resting + A['z'] > A['BUD'] + 1e-9:
                    break
                if (p if side == 0 else 100 - p) * A['z'] / 100.0 < 1.0 - 1e-9:
                    continue
                refill(t)
                if tok['v'] < n_open() + 2 - 1e-9:
                    ctr['throttled'] += 1; break
                tok['v'] -= 1.0; ctr['place'] += 1; requests.append(t)
                o = dict(side=side, pt=p, size=A['z'], rem=A['z'], ahead=0.0, ci=-1, t_send=t, t_act=t + L_PL, t_rm=np.inf,
                         live=False, cancel_sent=False, done=False, d=d, ahead0=np.nan, fq=0.0, ft=0.0, why='', t_end=np.nan)
                if ORDLOG:
                    allo.append(o)
                orders[side].append(o); heapq.heappush(ev, (o['t_act'], 0, next(seq), 'P', o))
                resting += A['z']; have.add(p)

    while ev:
        t, _, _, kind, pl = heapq.heappop(ev)
        if kind == 'D':
            decide(t)
        elif kind == 'P':
            o = pl
            if o['done']:
                continue
            ci = bisect.bisect_left(tf, t) - 1; f = F[ci] if ci >= 0 else None
            if f is None or (o['side'] == 0 and f['a1_p'] >= 0 and o['pt'] >= f['a1_p']) or (o['side'] == 1 and f['b1_p'] >= 0 and o['pt'] <= f['b1_p']):
                o['done'] = True; ctr['refused_post'] += 1; o['why'] = 'refused'; continue
            o['live'] = True; o['ci'] = ci; o['ahead'] = disp(f, o['side'], o['pt']); o['ahead0'] = o['ahead']
        elif kind == 'X':
            if not pl['done']:
                pl['done'] = True
            heapq.heappush(ev, (t, 2, next(seq), 'D', None))
        elif kind == 'K':
            side, qd, p = pl
            knet += qd; rf[(side, p)] = t + A['RF']
            heapq.heappush(ev, (t, 2, next(seq), 'D', None))
        else:
            tt, side, pt, sh = pl
            cands = [o for o in orders[side] if o['live'] and not o['done'] and o['t_act'] <= tt < o['t_rm']]
            if not cands:
                continue
            for o, q, through in match_leg(cands, side, pt, sh, F, tf, tt):
                px = o['pt'] / 100.0
                if side == 0:
                    net += q; cash -= q * px
                else:
                    net -= q; cash += q * px
                max_absnet = max(max_absnet, abs(net))
                fills.append((tt, side, o['pt'], q, through)); o['ft' if through else 'fq'] += q
                if o['rem'] <= 1e-12:
                    o['done'] = True; o['why'] = 'filled'; o['t_end'] = tt
                heapq.heappush(ev, (tt + L_FILL, 1, next(seq), 'K', (side, q if side == 0 else -q, o['pt'])))
        for s_ in (0, 1):
            if len(orders[s_]) > 64:
                orders[s_] = [o for o in orders[s_] if not o['done']]
    mids = np.array([((r['a1_p'] + r['b1_p']) / 2.0) if (r['b1_p'] >= 0 and r['a1_p'] > r['b1_p']) else np.nan for r in F])

    def mid_at(t, strict=False):
        i = (bisect.bisect_left(tf, t) if strict else bisect.bisect_right(tf, t)) - 1
        while i >= 0 and not np.isfinite(mids[i]):
            i -= 1
        return mids[i] if i >= 0 else np.nan
    m_stop = mid_at(t_stop + 100.0)
    rows = []
    for tt, side, pt, q, through in fills:
        sg = 1.0 if side == 0 else -1.0; m0 = mid_at(tt, strict=True)
        rows.append((mid, (tt - ws) / 1000.0, side, pt, q, through, sg * (y * 100 - pt) * q / 100.0, sg * (m0 - pt) * q / 100.0,
                     sg * (mid_at(tt + 1000.0) - m0) * q / 100.0, sg * (m_stop - pt) * q / 100.0))
    gross = cash + net * y
    olog = []
    for o in allo:                                                      # queue ahead at the order's end (frames strictly before it)
        te = o['t_end'] if o['why'] == 'filled' else (o['t_rm'] if o['why'] != 'refused' else o['t_act'])
        if o['live']:
            j = bisect.bisect_left(tf, te) - 1
            for k in range(o['ci'] + 1, j + 1):
                dd = disp_rng(F[k], o['side'], o['pt'])
                if dd is not None and dd < o['ahead']:
                    o['ahead'] = dd
        olog.append((mid, o['side'], o['pt'], o['d'], (o['t_send'] - ws) / 1000.0, o['live'], te - o['t_act'], o['ahead0'], o['ahead'], o['fq'], o['ft'], o['why']))
    return dict(pnl=gross, mtm=cash + net * m_stop / 100.0 if np.isfinite(m_stop) else np.nan, carry=net * (y - m_stop / 100.0) if np.isfinite(m_stop) else np.nan,
                net_end=net, max_absnet=max_absnet, n_fills=len(fills), shares=sum(x[3] for x in fills), fills=rows, requests=requests,
                place=ctr['place'], cancel=ctr['cancel'], refused_post=ctr['refused_post'], throttled=ctr['throttled'], pulls=dict(ctr['pulls']),
                max_rest_bid=max_rest[0], max_rest_ask=max_rest[1], olog=olog)


FC = ['mid', 't', 'side', 'pt', 'q', 'through', 'settle', 'spread', 'informed', 'to_stop']
WC = ['mid', 'ws', 'pnl', 'mtm', 'carry', 'net_end', 'max_absnet', 'n_fills', 'shares', 'place', 'cancel', 'refused_post', 'throttled', 'max_rest_bid', 'max_rest_ask']
res = {a['name']: [] for a in ARMS}; allf = {a['name']: [] for a in ARMS}; pulls = {a['name']: {} for a in ARMS}; reqs = {a['name']: [] for a in ARMS}
OC = ['mid', 'side', 'pt', 'd', 't_send', 'live', 'life_ms', 'ahead0', 'ahead_end', 'fill_queue', 'fill_swept', 'why']
ORDLOG = os.environ.get('PF_ORDLOG') == '1'; ordl = {a['name']: [] for a in ARMS}
GIDX = books.groupby('mid', sort=False).indices; MAXW = int(os.environ.get('PF_MAX_WINDOWS', '0'))
for nw_, mid in enumerate(sorted((m for m in GIDX if m in reg.index), key=lambda m: float(reg.loc[m].ws))):
    if MAXW and nw_ >= MAXW:
        break
    g = books.iloc[GIDX[mid]]; F = g.to_dict('records')
    kf = np.maximum.accumulate(CK.pf_known(g.t_ms.values, g.recv_ns.values, SHIFT)).tolist()
    for A in ARMS:
        r = run_window(mid, F, kf, A)
        if r is None:
            continue
        res[A['name']].append(tuple([mid, float(reg.loc[mid].ws)] + [r[c] for c in WC[2:]])); allf[A['name']].extend(r['fills'])
        for k, v in r['pulls'].items():
            pulls[A['name']][k] = pulls[A['name']].get(k, 0) + v
        reqs[A['name']].extend(r['requests']); ordl[A['name']].extend(r['olog'])
    io.mem_check(f'win {mid}')
summary = dict(day=day, preds=PREDS, clock=dict(pf_shift_ms=SHIFT, d_dec=CK.D_DEC, binance_in_ms=B_IN, order_live_ms=L_PL, cancel_ms=L_CA,
               pending_cancel_after_create_ms=L_PEND, fill_known_ms=L_FILL), arms={})
for A in ARMS:
    n = A['name']; st = n.replace('=', '').replace(',', '_')
    W = pd.DataFrame(res[n], columns=WC); f = pd.DataFrame(allf[n], columns=FC)
    W.to_parquet(f'{OUT}/win/{st}.parquet', index=False); f.to_parquet(f'{OUT}/fills/{st}.parquet', index=False)
    if ORDLOG:
        os.makedirs(f'{OUT}/orders', exist_ok=True); pd.DataFrame(ordl[n], columns=OC).to_parquet(f'{OUT}/orders/{st}.parquet', index=False)
    q = pd.Series(np.array(reqs[n]) // 60000).value_counts()
    summary['arms'][n] = dict(params={k: A[k] for k in DEF}, windows=int(len(W)), gross_usd=round(float(W.pnl.sum()), 2), mtm_stop_usd=round(float(W.mtm.sum()), 2),
        carry_stop_usd=round(float(W.carry.sum()), 2), carry_sd=round(float(W.carry.std()), 2), fills=int(W.n_fills.sum()), shares=round(float(W.shares.sum()), 1),
        spread_usd=round(float(f.spread.sum()), 2) if len(f) else 0.0, informed_usd=round(float(f.informed.sum()), 2) if len(f) else 0.0,
        through_share=round(float(f.q[f.through].sum() / max(f.q.sum(), 1e-9)), 3) if len(f) else 0.0,
        abs_net_end_mean=round(float(W.net_end.abs().mean()), 1), max_absnet=round(float(W.max_absnet.max()), 1),
        max_rest=round(float(max(W.max_rest_bid.max(), W.max_rest_ask.max())), 1), place=int(W.place.sum()), cancel=int(W.cancel.sum()),
        refused_post=int(W.refused_post.sum()), throttled=int(W.throttled.sum()), req_per_active_min_max=int(q.max()) if len(q) else 0, pulls=pulls[n])
io.write_json(f'{OUT}/summary.json', summary)
print('[done]', day, {n: (v['gross_usd'], v['mtm_stop_usd'], v['fills']) for n, v in summary['arms'].items()}, flush=True)
