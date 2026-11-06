"""Step 4 (exploratory): baseline + D1 rules one at a time, under a Tokyo-host latency scenario.

Clock: venue/UTC ms. Latency scenarios (user 2026-11-01: Tokyo host, ~20 ms to Predict):
  tokyo20  Predict frame known at updateTimestampMs + 20; Binance row known at E + 5; order/cancel active 20 ms after decision
  tape     Predict and Binance known at this host's receipt - 72 ms (both venues ~74 ms one way from here; keeps
           Predict's measured publication delay, ~58 ms median); orders active 20 ms after decision
Decisions happen at every known Predict frame, at every Binance guard flip (D1's extra syncs) and at 240 s.

Rules (D1 pieces, each switchable):
  G1 Binance 1 s against-side withhold: m = (mid_i/mid_j - 1)*1e4, i = last row known <= t, j = last <= t - 1 s;
     BUY withheld if m <= -0.5, SELL if m >= +0.5; unavailable (i older than 1 s) = not adverse
  G2 Binance 250 ms leg (TICK_UNION bps part): per row, 1e4*ln(mid_i/mid_j), j = last row <= row - 250 ms; same thresholds
  G3 gap guard: desired level at the touch and the next level behind >= 2 ticks away (or none) -> withhold
  G4 NO_OFF_TOUCH: desired level behind the touch -> withhold
  G5 STOP240: no quotes from 240 s
  G6 inventory TAPER10_HARD25: increasing side withheld at |net| >= 10; hard 25 incl. live size; final 60 s reduce-only
Fill model as step 3 (FIFO behind displayed size, trades at our price advance us, displayed-size cap,
sweep fills up to the sweeping leg's size, post-only refusal at activation, timed legs exact,
untimed at S-0.5 / S / S+0.5 s).

Additions 2026-11-01 (size / optimisation round; default arms unchanged):
  arms carry an order size; inventory caps scale with it (soft 2 x size, hard 5 x size = D1's 10/25 at size 5)
  M<d> = 1 s model guard at theta 0.<d...> c style tokens: M05 0.5, M10 1.0, M025 0.25, M075 0.75
  Q<d> = 250 ms model guard (pred_250), same token style; S<sec> = stop time for G5 and G6's final rule (default 240)
  PF_PREDS = prediction file stem under step5/<day>/ (default preds); PF_SIZES for the size armset
Adaptive size (2026-11-01, round O2): size spec "K:lo:hi:par" chosen per side at each decision; caps use hi.
  A1 model: hi if the 1 s prediction favours the side (bid: pred >= par, ask: pred <= -par), else lo
  A3 depth: clip(round(par x displayed size at our price), lo, hi)
  A4 inventory: hi on the side that reduces |net|, else lo
  A6 time: hi before par seconds into the window, else lo
  A live order at the same price is amended DOWN in place (queue kept) or cancel/replaced UP (queue reset).
Fill model v2 (default since the 2026-11-01 code review; PF_FILLMODEL=v1 reproduces the earlier numbers):
  - queue position is capped by the displayed size of every venue frame STRICTLY BEFORE the event (a frame stamped at a
    trade's own ms already shows that trade, so v1 counted it twice); frames where our level is beyond the 5 shown
    levels do not cap; activation and post-only use the frame strictly before activation
  - a fill swept through our level leaves no queue ahead of us
  - our own fills are known L_FILL = 20 ms later (inventory, caps, A4, repost after a full fill)
  - decisions also fire when a model prediction goes stale / resumes, and at A1's size threshold crossings
  - reduce-only quotes after the stop are capped at |inventory|
  G6's hard cap (5 x size) never binds given the soft cap (2 x size): |net| < 2c, live <= c, new <= c.
Usage: python3 -m predictfun.research.pf_step4 <day> <scenario> <armset>
  armset: singles  (base, base+G1..G6 for JOIN and MICRO; timing 'mid'; base also early/late)
          stack    (cumulative D1 order G4, G2, G6, G5, G3, G1 for MICRO and JOIN; all three timings)
          ends     (base and ALL only, all three timings)
          size     (MICRO + D1 stack, and + M05 at all timings, for each size in PF_SIZES)
          opt      (MICRO + D1 stack + M05 variants: theta, 250 ms guard, stop time, leave-one-rule-out; mid, size 5)
          custom   (PF_ARMS = "POLICY|R1+R2+...|timing|size;..."; output dir suffix PF_TAG)
"""
import sys, os, math, bisect, heapq, collections
import numpy as np, pandas as pd
from . import pf_io as io

day, SCEN, ARMSET = sys.argv[1], sys.argv[2], sys.argv[3]
S1 = f'{io.OUT}/step1/{day}'
PREDS = os.environ.get('PF_PREDS', 'preds')
OUT = (f"{io.OUT}/step4/{SCEN}_{ARMSET}{'' if os.environ.get('PF_TIMING', 'v1') == 'v1' else '_' + os.environ['PF_TIMING']}"
       f"{'' if PREDS == 'preds' else '_' + PREDS}{('_' + os.environ['PF_TAG']) if os.environ.get('PF_TAG') else ''}/{day}")
os.makedirs(OUT, exist_ok=True)
SIZE = 5.0; NET_MS = 72.0; L_O = 20.0
P_IN = 20.0; B_IN = 5.0
LEGACY_INTENT = os.environ.get('PF_LEGACY_INTENT') == '1'   # diagnostic: reproduce pre-O2 intent handling
FILLMODEL = os.environ.get('PF_FILLMODEL', 'v2')              # v1 = pre-review fill handling (regression only)
V2 = FILLMODEL != 'v1'
L_FILL = 20.0

reg = pd.read_parquet(f'{S1}/registry.parquet').set_index('mid')
TIMING = os.environ.get('PF_TIMING', 'v1')     # v1: WS exact + bracket; e2: WS + E2 point estimate; e4: WS + E4 exact + bracket
POLICIES = os.environ.get('PF_POLICIES', 'JOIN,MICRO').split(',')
if TIMING == 'v1':
    LT = pd.read_parquet(f'{S1}/legs_timed.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how'])
else:
    LT = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 't_e4'])
    alt = LT[f't_{TIMING}'].values
    use = (LT.how.values == 0) & np.isfinite(alt)
    LT['t_ms'] = np.where(use, alt, LT.t_ms.values); LT['how'] = np.where(use, 3, LT.how.values)
    LT = LT.drop(columns=['t_e2', 't_e4'])
LT_BY = {m: g for m, g in LT.groupby('mid', sort=False)}
PRED = {}; PRED250 = {}
if ARMSET in ('model', 'size', 'opt', 'custom'):
    pr = pd.read_parquet(f'{io.OUT}/step5/{day}/{PREDS}.parquet', columns=['mid', 'tau', 'pred_1000', 'pred_250'])
    for col, dst in (('pred_1000', PRED), ('pred_250', PRED250)):
        q_ = pr[pr[col].notna()]
        dst.update({m: (g.tau.values.astype(np.float64), g[col].values.astype(np.float64)) for m, g in q_.groupby('mid', sort=False)})
    del pr, q_
cols = ['mid', 't_ms', 'recv_ns'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 6) for k in 'ps']
books = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=cols) for f in sorted(os.listdir(f'{S1}/books'))], ignore_index=True)
books.sort_values(['mid', 't_ms', 'recv_ns'], inplace=True, kind='stable')

# ---- Binance futures: knowledge time (ms) + mid, whole day +-1 h ----
# Only rows where the mid or the 250 ms-leg class changes are kept. Exact for both guards: G1 between kept
# rows sees an unchanged mid (m = 0, or "unavailable" = not adverse, same decision); G2's class is held per row.
kb, mb, cb = [], [], []
carry_k = np.empty(0); carry_m = np.empty(0); last_m = np.nan; last_c = 99
n_raw = 0
for h in io.hours_of(day, extra_after=1, extra_before=1):
    p = f'{io.BN_DIR}/{h}.csv.gz'
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p, header=None, names=['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq'], usecols=['recv', 'E', 'u', 'bid', 'ask'])
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df = df[(df.bid > 0) & (df.ask > df.bid)].drop_duplicates('u', keep='first')
    k = (df.E.values + B_IN) if SCEN == 'tokyo20' else (df.recv.values / 1e6 - NET_MS)
    m = (df.bid.values + df.ask.values) / 2
    del df
    o = np.argsort(k, kind='stable'); k = k[o].astype(np.float64); m = m[o].astype(np.float64)
    n_raw += len(k)
    K = np.concatenate([carry_k, k]); M = np.concatenate([carry_m, m])
    j = np.searchsorted(K, k - 250.0, side='right') - 1
    bp = np.where(j >= 0, 1e4 * np.log(m / M[np.clip(j, 0, None)]), np.nan)
    c = np.where(bp <= -0.5, -1, np.where(bp >= 0.5, 1, 0)).astype(np.int8)
    keep = (m != np.r_[last_m, m[:-1]]) | (c != np.r_[last_c, c[:-1]])
    kb.append(k[keep]); mb.append(m[keep]); cb.append(c[keep])
    last_m = m[-1]; last_c = c[-1]
    tail = k >= k[-1] - 1000.0; carry_k = k[tail]; carry_m = m[tail]
    del K, M, j, bp, c, keep, k, m
    io.mem_check(f'bn {h}')
KB = np.concatenate(kb); MB = np.concatenate(mb); BC = np.concatenate(cb)
o = np.argsort(KB, kind='stable'); KB = KB[o]; MB = MB[o]; BC = BC[o]
del kb, mb, cb, o
print(f'[in] {day} {SCEN} books {len(books)} legs {len(LT)} binance raw {n_raw} kept {len(KB)} [mem] {io.rss_mib()} MiB', flush=True)


def guard_state(t, use1, use2):
    """(buy_adverse, sell_adverse) at knowledge time t."""
    i = int(np.searchsorted(KB, t, side='right')) - 1
    if i < 0:
        return False, False
    ba = sa = False
    if use1 and t - KB[i] <= 1000.0:
        j = int(np.searchsorted(KB, t - 1000.0, side='right')) - 1
        if j >= 0:
            m = (MB[i] / MB[j] - 1.0) * 1e4
            ba |= m <= -0.5; sa |= m >= 0.5
    if use2:
        ba |= BC[i] == -1; sa |= BC[i] == 1
    return ba, sa


def guard_states_vec(t, use1, use2):
    """Vectorised guard_state over an array of knowledge times -> (buy_adverse, sell_adverse) bool arrays."""
    i = np.searchsorted(KB, t, side='right') - 1
    ok = i >= 0; ic = np.clip(i, 0, None)
    ba = np.zeros(len(t), bool); sa = np.zeros(len(t), bool)
    if use1:
        j = np.searchsorted(KB, t - 1000.0, side='right') - 1
        av = ok & (j >= 0) & (t - KB[ic] <= 1000.0)
        m = (MB[ic] / MB[np.clip(j, 0, None)] - 1.0) * 1e4
        ba |= av & (m <= -0.5); sa |= av & (m >= 0.5)
    if use2:
        ba |= ok & (BC[ic] == -1); sa |= ok & (BC[ic] == 1)
    return ba, sa


_FLIP_CACHE = {}
def guard_flips(t0, t1, use1, use2):
    """Times in [t0, t1] where the guard state changes (candidates: row knowledge times, and +1 s for G1)."""
    key = (t0, use1, use2)
    if key in _FLIP_CACHE:
        return _FLIP_CACHE[key]
    a = np.searchsorted(KB, t0 - 1000.0); b = np.searchsorted(KB, t1)
    cand = KB[a:b]
    if use1:
        cand = np.concatenate([cand, cand + 1000.0, cand + 1000.0 + 1e-6])
    cand = np.unique(np.concatenate([[t0], cand[(cand >= t0) & (cand <= t1)]]))
    ba, sa = guard_states_vec(cand, use1, use2)
    ch = np.r_[False, (ba[1:] != ba[:-1]) | (sa[1:] != sa[:-1])]
    out = cand[ch].tolist()
    _FLIP_CACHE[key] = out
    return out


def model_theta(rules, key='M'):
    for r in rules:
        if r.startswith(key):
            d = r[1:]
            return float(d[0] + '.' + d[1:])   # M05 -> 0.5 c, M10 -> 1.0 c, M025 -> 0.25 c
    return None


def stop_time(rules):
    for r in rules:
        if r.startswith('S') and r[1:].isdigit():
            return float(r[1:])
    return 240.0


def model_state(mid, t, th, P=PRED):
    v = P.get(mid)
    if v is None:
        return False, False
    i = int(np.searchsorted(v[0], t, side='right')) - 1
    if i < 0 or t - v[0][i] > 100.0:
        return False, False
    p = v[1][i]
    return p <= -th, p >= th


def model_flips(mid, t0, t1, th, P=PRED):
    v = P.get(mid)
    if v is None:
        return []
    a = np.searchsorted(v[0], t0); b = np.searchsorted(v[0], t1)
    tt = v[0][a:b]; p = v[1][a:b]
    st = np.where(p <= -th, -1, np.where(p >= th, 1, 0))
    ch = np.r_[True, st[1:] != st[:-1]]
    out = tt[ch].tolist()
    if V2 and len(tt):
        gap = np.r_[np.diff(tt) > 100.0, True]                 # prediction goes stale after the last point of a run
        out += (tt[gap] + 100.0 + 1e-3).tolist() + tt[1:][gap[:-1]].tolist()   # ... and resumes at the next point
    return out


def parse_size(x):
    if isinstance(x, str) and ':' in x:
        k, lo, hi, par = x.split(':')
        return dict(kind=k, lo=float(lo), hi=float(hi), par=float(par))
    return dict(kind='fixed', lo=float(x), hi=float(x), par=0.0)


def model_value(mid, t):
    v = PRED.get(mid)
    if v is None:
        return None
    i = int(np.searchsorted(v[0], t, side='right')) - 1
    if i < 0 or t - v[0][i] > 100.0:
        return None
    return v[1][i]


def run_window(mid, F, kf, rules, policy, mode, size=SIZE):
    SZ = parse_size(size); cap = SZ['hi']
    R = reg.loc[mid]
    if pd.isna(R.outcome) or R.outcome not in ('Up', 'Down'):
        return None
    y = 1.0 if R.outcome == 'Up' else 0.0
    ws = float(R.ws) * 1000.0
    tf = [r['t_ms'] for r in F]
    g = LT_BY.get(mid)
    if g is None:
        trades = []
    else:
        shift = {'early': -500.0, 'mid': 0.0, 'late': 499.0}[mode]
        tt = np.where(g.how.values > 0, g.t_ms.values, g.exec_s.values * 1000.0 + shift)
        trades = sorted(zip(tt.tolist(), g.side.tolist(), g.ptick.tolist(), g.shares.tolist(), (g.how.values > 0).tolist()))
    use1, use2 = 'G1' in rules, 'G2' in rules
    ev = []    # (time, prio, seq, kind, payload) ; prio: activation 0, trade 1, decision 2
    seq = 0
    for i, k in enumerate(kf):
        if ws <= k < ws + 300_000:
            ev.append((k, 2, seq, 'D', i)); seq += 1
    if use1 or use2:
        for t in guard_flips(ws, ws + 300_000, use1, use2):
            ev.append((t, 2, seq, 'D', None)); seq += 1
    th = model_theta(rules); th2 = model_theta(rules, 'Q'); stop = stop_time(rules)
    if V2 and SZ['kind'] == 'A1':
        for t in model_flips(mid, ws, ws + 300_000, SZ['par']):
            ev.append((t, 2, seq, 'D', None)); seq += 1
    if th is not None:
        for t in model_flips(mid, ws, ws + 300_000, th):
            ev.append((t, 2, seq, 'D', None)); seq += 1
    if th2 is not None:
        for t in model_flips(mid, ws, ws + 300_000, th2, PRED250):
            ev.append((t, 2, seq, 'D', None)); seq += 1
    for t in sorted({ws + stop * 1000.0, ws + 240_000, ws + 300_000}):
        ev.append((t, 2, seq, 'D', None)); seq += 1
    for tr in trades:
        ev.append((tr[0], 1, seq, 'T', tr)); seq += 1
    heapq.heapify(ev)
    orders = {0: None, 1: None}; intent = {0: None, 1: None}
    net = 0.0; knet = 0.0; cash = 0.0; fills = []; withheld = collections.Counter()   # knet = inventory as known to us

    def venue_book(t):
        i = bisect.bisect_right(tf, t) - 1
        return F[i] if i >= 0 else None

    def known_frame(t):
        i = bisect.bisect_right(kf, t) - 1
        return F[i] if i >= 0 else None

    def disp(f, side, pt):
        pre = 'b' if side == 0 else 'a'
        for i in range(1, 6):
            if f[f'{pre}{i}_p'] == pt:
                return f[f'{pre}{i}_s']
        return 0.0

    def disp_rng(f, side, pt):
        """Displayed size at pt, or None when pt lies beyond the 5 levels shown (size unknown)."""
        last = f['b5_p'] if side == 0 else f['a5_p']
        if last >= 0 and ((side == 0 and pt < last) or (side == 1 and pt > last)):
            return None
        return disp(f, side, pt)

    def side_size(t, rel, side, f, pq):
        k = SZ['kind']
        if k == 'fixed':
            return SZ['lo']
        lo, hi, par = SZ['lo'], SZ['hi'], SZ['par']
        if k == 'A1':
            v = model_value(mid, t)
            return hi if (v is not None and ((side == 0 and v >= par) or (side == 1 and v <= -par))) else lo
        if k == 'A3':
            return float(min(hi, max(lo, round(par * disp(f, side, pq)))))
        if k == 'A4':
            return hi if ((side == 0 and knet < 0) or (side == 1 and knet > 0)) else lo
        if k == 'A6':
            return hi if rel < par else lo
        raise ValueError(k)

    NOQ = ((None, 0.0), (None, 0.0))

    def desired(t):
        rel = (t - ws) / 1000.0
        if rel >= 300 or ('G5' in rules and rel >= stop):
            return NOQ
        f = known_frame(t)
        if f is None:
            return NOQ
        b1, bs, a1, as_ = f['b1_p'], f['b1_s'], f['a1_p'], f['a1_s']
        if b1 < 0 or a1 < 0 or a1 <= b1:
            return NOQ
        if policy == 'JOIN':
            q = [b1, a1]
        else:
            micro = (b1 * as_ + a1 * bs) / max(bs + as_, 1e-12); half = (a1 - b1) / 2.0
            qb = math.floor(micro - half + 1e-9); qa = math.ceil(micro + half - 1e-9)
            q = [qb if 1 <= qb < a1 else None, qa if b1 < qa <= 99 else None]
        ba, sa = guard_state(t, use1, use2) if (use1 or use2) else (False, False)
        if th is not None:
            mb_, ms_ = model_state(mid, t, th)
            ba = ba or mb_; sa = sa or ms_
        if th2 is not None:
            mb_, ms_ = model_state(mid, t, th2, PRED250)
            ba = ba or mb_; sa = sa or ms_
        live = {s: (orders[s]['rem'] if orders[s] else 0.0) for s in (0, 1)}
        zs = [0.0, 0.0]
        for side in (0, 1):
            p = q[side]
            if p is None:
                continue
            zs[side] = sz_ = side_size(t, rel, side, f, p)
            touch = b1 if side == 0 else a1
            if 'G4' in rules and ((side == 0 and p < b1) or (side == 1 and p > a1)):
                q[side] = None; withheld['G4'] += 1; continue
            if 'G3' in rules and p == touch:
                nxt = f['b2_p'] if side == 0 else f['a2_p']
                if nxt < 0 or abs(touch - nxt) >= 2:
                    q[side] = None; withheld['G3'] += 1; continue
            if (side == 0 and ba) or (side == 1 and sa):
                q[side] = None; withheld['BTC'] += 1; continue
            if 'G6' in rules:
                increasing = (side == 0 and knet >= 0) or (side == 1 and knet <= 0)
                reducing = (side == 0 and knet < 0) or (side == 1 and knet > 0)
                if rel >= stop and not reducing:
                    q[side] = None; withheld['G6_final'] += 1; continue
                if V2 and rel >= stop:
                    zs[side] = sz_ = min(sz_, abs(knet))           # reduce-only: never flip the position
                if increasing and knet != 0 and abs(knet) >= 2 * cap:
                    q[side] = None; withheld['G6_soft'] += 1; continue
                if increasing and abs(knet) + live[side] + sz_ > 5 * cap:
                    q[side] = None; withheld['G6_hard'] += 1; continue
        return tuple((q[i], zs[i] if q[i] is not None else 0.0) for i in (0, 1))

    while ev:
        t, pr, _, kind, pl = heapq.heappop(ev)
        if kind == 'D':
            want = desired(t)
            for side in (0, 1):
                if LEGACY_INTENT and want[side][0] is None and intent[side] is None:
                    continue                              # old behaviour, diagnostic only (see PF_LEGACY_INTENT)
                if want[side] != intent[side]:
                    intent[side] = want[side]
                    heapq.heappush(ev, (t + L_O, 0, seq, 'A', (side,) + want[side])); seq += 1
        elif kind == 'A':
            side, p, zsz = pl
            o = orders[side]
            if p is None:
                orders[side] = None; continue
            if o is not None and o['pt'] == p:
                if zsz < o['size']:                       # amend down in place, queue kept
                    o['rem'] -= o['size'] - zsz; o['size'] = zsz
                    if o['rem'] <= 1e-12:
                        orders[side] = None; intent[side] = None
                    continue
                if zsz <= o['size']:
                    continue                              # same size: keep (D1's same-price keep)
                                                          # larger: cancel/replace below, queue reset
            ci = (bisect.bisect_left(tf, t) if V2 else bisect.bisect_right(tf, t)) - 1
            f = F[ci] if ci >= 0 else None
            if f is not None and ((side == 0 and f['a1_p'] >= 0 and p >= f['a1_p']) or (side == 1 and f['b1_p'] >= 0 and p <= f['b1_p'])):
                orders[side] = None; continue
            orders[side] = dict(pt=p, rem=zsz, size=zsz, ahead=disp(f, side, p) if f is not None else 0.0, t0=t, ci=ci)
        elif kind == 'K':                                   # v2: our fill becomes known
            side, qd, done = pl
            knet += qd
            if done and orders[side] is None:
                intent[side] = None                         # slot vacated: re-place now (D1's repost)
                heapq.heappush(ev, (t, 2, seq, 'D', None)); seq += 1
        else:
            tt_, side, pt, sh, timed = pl
            o = orders[side]
            if o is None:
                continue
            if V2:
                j = bisect.bisect_left(tf, t) - 1           # frames strictly before the trade
                for k in range(o['ci'] + 1, j + 1):
                    d = disp_rng(F[k], side, o['pt'])
                    if d is not None and d < o['ahead']:
                        o['ahead'] = d
                if j > o['ci']:
                    o['ci'] = j
            else:
                f = venue_book(t)
                if f is not None:
                    o['ahead'] = min(o['ahead'], disp(f, side, o['pt']))
            through = (side == 0 and pt < o['pt']) or (side == 1 and pt > o['pt'])
            if pt == o['pt']:
                use = min(sh, o['ahead']); o['ahead'] -= use; q = min(sh - use, o['rem'])
            elif through:
                q = min(sh, o['rem'])
                if V2:
                    o['ahead'] = 0.0                        # a sweep through our level consumed the queue ahead
            else:
                q = 0.0
            if q > 1e-12:
                o['rem'] -= q; px = o['pt'] / 100.0
                if side == 0:
                    net += q; cash -= q * px
                else:
                    net -= q; cash += q * px
                fills.append((t, side, o['pt'], q, 'through' if through else 'at', timed))
                done = o['rem'] <= 1e-12
                if done:
                    orders[side] = None
                if V2:
                    heapq.heappush(ev, (t + L_FILL, 1, seq, 'K', (side, q if side == 0 else -q, done))); seq += 1
                else:
                    knet = net
                    if done:
                        intent[side] = None          # slot vacated: the next decision re-places (D1's repost)
    mids = np.array([((r['a1_p'] + r['b1_p']) / 2.0) if (r['b1_p'] >= 0 and r['a1_p'] > r['b1_p']) else np.nan for r in F])

    def mid_at(t):
        i = bisect.bisect_right(tf, t) - 1
        while i >= 0 and not np.isfinite(mids[i]):
            i -= 1
        return mids[i] if i >= 0 else np.nan
    rows = []
    for t, side, pt, q, kd, timed in fills:
        sg = 1.0 if side == 0 else -1.0
        rows.append((mid, (t - ws) / 1000.0, side, pt, q, kd == 'through', timed, sg * (y * 100 - pt) * q / 100.0,
                     sg * (mid_at(t + 1000) - pt) * q / 100.0, sg * (mid_at(t + 60000) - pt) * q / 100.0,
                     0.25 * 0.02 * min(pt, 100 - pt) / 100.0 * q))
    return dict(pnl=cash + net * y, net_end=net, n_fills=len(fills), shares=sum(x[3] for x in fills), fills=rows, withheld=withheld)


BASE = {'JOIN': [], 'MICRO': []}
SINGLES = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']
STACK = ['G4', 'G2', 'G6', 'G5', 'G3', 'G1']
arms = []
if ARMSET == 'singles':
    for pol in [p for p in ('JOIN', 'MICRO') if p in POLICIES]:
        arms += [(pol, (), m) for m in ('early', 'mid', 'late')]
        arms += [(pol, (r,), 'mid') for r in SINGLES if not (pol == 'JOIN' and r == 'G4')]
        arms += [(pol, ('G1', 'G2'), 'mid')]
elif ARMSET == 'stack':
    for pol in [p for p in ('MICRO', 'JOIN') if p in POLICIES]:
        arms += [(pol, (), m) for m in ('early', 'mid', 'late')]
        for n in range(1, len(STACK) + 1):
            rs = tuple(STACK[:n])
            if pol == 'JOIN' and rs == ('G4',):
                continue
            for m in ('early', 'mid', 'late'):
                arms.append((pol, rs, m))
elif ARMSET == 'model':
    NOBTC = ('G4', 'G6', 'G5', 'G3')
    sets = [NOBTC, tuple(STACK)] + [NOBTC + (f'M{k}',) for k in ('05', '10')] + [tuple(STACK) + (f'M{k}',) for k in ('05', '10')]
    for rs in sets:
        for m in ('early', 'mid', 'late'):
            arms.append(('MICRO', rs, m))
elif ARMSET == 'size':
    for sz in [float(x) for x in os.environ.get('PF_SIZES', '5,10,25,50,100').split(',')]:
        arms.append(('MICRO', tuple(STACK), 'mid', sz))
        for m in ('early', 'mid', 'late'):
            arms.append(('MICRO', tuple(STACK) + ('M05',), m, sz))
elif ARMSET == 'opt':
    ST = tuple(STACK)
    sets = [ST, ST + ('M05',), ST + ('M025',), ST + ('M075',), ST + ('M10',),
            ST + ('M05', 'Q025'), ST + ('M05', 'Q05'), ST + ('Q05',),
            ST + ('M05', 'S270'), ST + ('M05', 'S300')]
    sets += [tuple(g for g in STACK if g != drop) + ('M05',) for drop in ('G1', 'G2', 'G3', 'G4', 'G6')]
    arms = [('MICRO', rs, 'mid') for rs in sets]
elif ARMSET == 'custom':
    for spec in os.environ['PF_ARMS'].split(';'):
        pol, rs, m, sz = spec.split('|')
        arms.append((pol, tuple(x for x in rs.split('+') if x), m, sz if ':' in sz else float(sz)))
else:
    for pol in [p for p in ('MICRO', 'JOIN') if p in POLICIES]:
        for rs in ((), tuple(STACK)):
            for m in ('early', 'mid', 'late'):
                arms.append((pol, rs, m))
arms = [a if len(a) == 4 else a + (SIZE,) for a in arms]
if any(r[0] in 'MQ' for a in arms for r in a[1]) and not PRED:
    sys.exit(f'REFUSED: model-guard arms but no predictions loaded from step5/{day}/{PREDS}.parquet')
if any(r[0] == 'Q' for a in arms for r in a[1]) and not PRED250:
    sys.exit(f'REFUSED: 250 ms guard arms but no pred_250 in step5/{day}/{PREDS}.parquet')

FCOLS = ['mid', 't', 'side', 'pt', 'q', 'through', 'timed', 'settle', 'mo1', 'mo60', 'rebate']
res = collections.defaultdict(list); allf = collections.defaultdict(list); wh = collections.defaultdict(collections.Counter)
MAXW = int(os.environ.get('PF_MAX_WINDOWS', '0')); nw = 0
for mid, g in books.groupby('mid', sort=False):
    if mid not in reg.index:
        continue
    nw += 1
    if MAXW and nw > MAXW:
        break
    F = g.to_dict('records')
    kf = (g.t_ms.values + P_IN) if SCEN == 'tokyo20' else (g.recv_ns.values / 1e6 - NET_MS)
    kf = np.maximum.accumulate(kf.astype(np.float64)).tolist()   # knowledge times non-decreasing in frame order
    for pol, rs, m, sz in arms:
        r = run_window(mid, F, kf, set(rs), pol, m, sz)
        if r is None:
            continue
        name = f"{pol}{'+' + '+'.join(rs) if rs else ''}/{m}{('@' + sz) if isinstance(sz, str) else ('' if sz == SIZE else '@%g' % sz)}"
        res[name].append((mid, r['pnl'], r['net_end'], r['n_fills'], r['shares']))
        wh[name].update(r['withheld'])
        allf[name].extend(r['fills'])
    _FLIP_CACHE.clear()
    io.mem_check(f'win {mid}')
os.makedirs(f'{OUT}/fills', exist_ok=True)
summary = dict(day=day, scenario=SCEN, armset=ARMSET, timing=TIMING, preds=PREDS, fill_model=FILLMODEL, mode='EXPLORATORY', size=SIZE, order_latency_ms=L_O,
               predict_in_ms=(P_IN if SCEN == 'tokyo20' else 'recv-72'), binance_in_ms=(B_IN if SCEN == 'tokyo20' else 'recv-72'), arms={})
for name, v in res.items():
    W = pd.DataFrame(v, columns=['mid', 'pnl', 'net_end', 'n_fills', 'shares'])
    f = pd.DataFrame(allf.pop(name), columns=FCOLS)
    f.to_parquet(f"{OUT}/fills/{name.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')}.parquet", index=False)
    f['kind'] = np.where(f.through, 'through', 'at')
    summary['arms'][name] = dict(
        windows=int(len(W)), gross_usd=round(float(W.pnl.sum()), 2), windows_positive=int((W.pnl > 0).sum()),
        fills=int(W.n_fills.sum()), shares=round(float(W.shares.sum()), 1), abs_net_end_p95=round(float(W.net_end.abs().quantile(0.95)), 1),
        mo1_usd=round(float(f.mo1.sum()), 2), mo60_usd=round(float(f.mo60.sum()), 2), rebate_usd_unverified=round(float(f.rebate.sum()), 2),
        by_kind={k: dict(shares=round(float(x.q.sum()), 1), mo1=round(float(x.mo1.sum()), 2), settle=round(float(x.settle.sum()), 2)) for k, x in f.groupby('kind')} if len(f) else {},
        by_phase_60s=f.assign(ph=(f.t // 60 * 60).astype(int)).groupby('ph').agg(shares=('q', 'sum'), settle=('settle', 'sum')).round(2).to_dict('index') if len(f) else {},
        withheld_decisions=dict(wh[name]))
    del f
io.write_json(f'{OUT}/summary.json', summary)
print('[done]', day, SCEN, ARMSET, {k: (v['gross_usd'], v['fills']) for k, v in summary['arms'].items()})
