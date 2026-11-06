"""Step 3 (exploratory): plain baseline quoting on Predict, no D1 rules.

Policies (5 shares a side, quote over the whole live window, no guards, no stop,
no inventory limit; fills are tagged by window phase so a stop can be read off):
  JOIN   bid at best bid, ask at best ask.
  MICRO  D1's anchor without overlays: bid floor(micro - half spread), ask ceil(micro + half spread),
         withheld only if it would cross.

Venue-time simulation per window. A book frame with source time T is acted on at T + DELAY
(one-way in + one-way out), cancel/replace effective at T + DELAY; the old order is live until then.
Queue: join behind the displayed size at our price when the order becomes effective; only trades
at our price advance us (cancels never do), but queue-ahead is capped by the displayed size.
A trade at a worse price on our side (taker swept through us) fills us up to that trade's size.

Trade timing: ~half the maker legs carry an exact ms time from a matched WS settlementsPending
increment (step 1c); executedAt is the true time rounded to the second, so the untimed rest are
placed at S-0.5 s (early), S (mid) or S+0.499 s (late) - all three reported.

P&L: cash + net YES x outcome (Up=1, tie 0.5); maker fee 0; rebate (25% of taker fee
0.02*min(p,1-p)) reported separately, unverified. Markouts vs Predict mid at +1/10/60 s.

Usage: python3 -m predictfun.research.pf_step3 20261023 [delay_ms]
"""
import sys, os, math, bisect, collections
import numpy as np, pandas as pd
from . import pf_io as io

day = sys.argv[1]
DELAY_MS = float(sys.argv[2]) if len(sys.argv) > 2 else None
S1 = f'{io.OUT}/step1/{day}'
OUT = f'{io.OUT}/step3/{day}' + (f'_d{int(float(sys.argv[2]))}' if len(sys.argv) > 2 else '')
os.makedirs(OUT, exist_ok=True)
SIZE = 5.0

reg = pd.read_parquet(f'{S1}/registry.parquet').set_index('mid')
LT = pd.read_parquet(f'{S1}/legs_timed.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how'])
LT_BY = {m: g for m, g in LT.groupby('mid', sort=False)}
cols = ['mid', 't_ms', 'recv_ns'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 6) for k in 'ps']
books = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=cols) for f in sorted(os.listdir(f'{S1}/books'))], ignore_index=True)
books.sort_values(['mid', 't_ms', 'recv_ns'], inplace=True, kind='stable')
lat = (books.recv_ns.values / 1e6 - books.t_ms.values)
IN_MS = float(np.median(lat))
DELAY = DELAY_MS if DELAY_MS is not None else 2 * IN_MS
print(f'[in] books {len(books)} legs {len(LT)} timed {(LT.how > 0).mean():.3f} median recv-source {IN_MS:.1f} ms -> delay {DELAY:.1f} ms [mem] {io.rss_mib()} MiB', flush=True)


def trade_events(mid, ws, mode, P_idx=None):
    """(t_ms, side, ptick, shares) for one window's live maker legs. Legs matched to a WS
    settlementsPending increment (step 1c) keep that ms time; for the rest executedAt is the true
    time rounded to the second, so mode places them at S-0.5 s (early), S (mid) or S+0.499 s (late)."""
    g = LT_BY[mid] if mid in LT_BY else LT.iloc[0:0]
    shift = {'early': -500, 'mid': 0, 'late': 499}[mode]
    t = np.where(g.how.values > 0, g.t_ms.values, g.exec_s.values * 1000.0 + shift)
    out = sorted(zip(t.tolist(), g.side.tolist(), g.ptick.tolist(), g.shares.tolist(), (g.how.values > 0).tolist()))
    return out


def quote(policy, f):
    """Desired (bid_tick, ask_tick) from one frame row, None = no quote."""
    b1, bs, a1, as_ = f['b1_p'], f['b1_s'], f['a1_p'], f['a1_s']
    if b1 < 0 or a1 < 0 or a1 <= b1:
        return None, None
    if policy == 'JOIN':
        return b1, a1
    micro = (b1 * as_ + a1 * bs) / max(bs + as_, 1e-12)
    half = (a1 - b1) / 2.0
    qb = math.floor(micro - half + 1e-9); qa = math.ceil(micro + half - 1e-9)
    return (qb if qb < a1 and qb >= 1 else None), (qa if qa > b1 and qa <= 99 else None)


def disp(f, side, pt):
    pre = 'b' if side == 0 else 'a'
    for i in range(1, 6):
        if f[f'{pre}{i}_p'] == pt:
            return f[f'{pre}{i}_s']
    return 0.0


def run_window(mid, F, tf, policy, mode, P_idx):
    R = reg.loc[mid]; ws_ms = int(R.ws) * 1000
    if pd.isna(R.outcome) or R.outcome not in ('Up', 'Down'):
        return None
    y = 1.0 if R.outcome == 'Up' else 0.0
    trades = trade_events(mid, int(R.ws), mode, P_idx)
    # command schedule: each frame in the live window -> desired quotes effective at t + DELAY
    cmds = []
    for r in F:
        tr = r['t_ms'] - ws_ms
        if tr < 0 or tr >= 300_000:
            continue
        cmds.append((r['t_ms'] + DELAY, quote(policy, r)))
    cmds.append((ws_ms + 300_000 + DELAY, (None, None)))
    # event loop over commands and trades in venue time
    orders = {0: None, 1: None}    # side -> dict(pt, rem, ahead)
    net = 0.0; cash = 0.0; fills = []
    ci = 0; ti = 0
    def book_at(t):
        i = bisect.bisect_right(tf, t) - 1
        return F[i] if i >= 0 else None
    while ci < len(cmds) or ti < len(trades):
        if ti < len(trades) and (ci >= len(cmds) or trades[ti][0] < cmds[ci][0]):
            t, side, pt, sh, timed = trades[ti]; ti += 1
            o = orders[side]
            if o is None:
                continue
            f = book_at(t)
            if f is not None:
                o['ahead'] = min(o['ahead'], disp(f, side, o['pt']))
            through = (side == 0 and pt < o['pt']) or (side == 1 and pt > o['pt'])
            if pt == o['pt']:
                use = min(sh, o['ahead']); o['ahead'] -= use; q = min(sh - use, o['rem'])
            elif through:
                q = min(sh, o['rem'])     # the taker leg that swept past us would have hit us first, up to its size
            else:
                q = 0.0
            if q > 1e-12:
                o['rem'] -= q
                px = o['pt'] / 100.0
                if side == 0:
                    net += q; cash -= q * px
                else:
                    net -= q; cash += q * px
                fills.append((t, side, o['pt'], q, 'through' if through else 'at', timed, (t - o['t0']) / 1000.0))
                if o['rem'] <= 1e-12:
                    orders[side] = None
            continue
        t, (qb, qa) = cmds[ci]; ci += 1
        f = book_at(t)
        for side, want in ((0, qb), (1, qa)):
            o = orders[side]
            if want is None:
                orders[side] = None
                continue
            if o is not None and o['pt'] == want:
                continue
            # post-only: refuse if it would cross the book at activation
            if f is not None and ((side == 0 and f['a1_p'] >= 0 and want >= f['a1_p']) or (side == 1 and f['b1_p'] >= 0 and want <= f['b1_p'])):
                orders[side] = None
                continue
            orders[side] = dict(pt=want, rem=SIZE, ahead=disp(f, side, want) if f is not None else 0.0, t0=t)
    # valuation
    mids = np.array([((r['a1_p'] + r['b1_p']) / 2.0) if (r['b1_p'] >= 0 and r['a1_p'] > r['b1_p']) else np.nan for r in F])
    def mid_at(t):
        i = bisect.bisect_right(tf, t) - 1
        while i >= 0 and not np.isfinite(mids[i]):
            i -= 1
        return mids[i] if i >= 0 else np.nan
    rows = []
    for t, side, pt, q, kind, timed, age in fills:
        sg = 1.0 if side == 0 else -1.0
        mk = {f'mo{h}': sg * (mid_at(t + h * 1000) - pt) * q / 100.0 for h in (1, 10, 60)}
        rows.append(dict(mid=mid, t=(t - ws_ms) / 1000.0, side=side, pt=pt, q=q, kind=kind, timed=timed, age=age, settle=sg * (y * 100 - pt) * q / 100.0,
                         rebate=0.25 * 0.02 * min(pt, 100 - pt) / 100.0 * q, **mk))
    return dict(mid=mid, pnl=cash + net * y, net_end=net, n_fills=len(fills), shares=sum(f[3] for f in fills), fills=rows)


res = collections.defaultdict(list)
st = collections.Counter()
allfills = []
for mid, g in books.groupby('mid', sort=False):
    if mid not in reg.index:
        continue
    F = g.to_dict('records'); tf = [r['t_ms'] for r in F]
    for policy in ('JOIN', 'MICRO'):
        for mode in ('early', 'mid', 'late'):
            r = run_window(mid, F, tf, policy, mode, None)
            if r is None:
                st[f'{policy}/{mode}:NO_OUTCOME'] += 1
                continue
            res[(policy, mode)].append((mid, r['pnl'], r['net_end'], r['n_fills'], r['shares']))
            for fr in r['fills']:
                fr.update(policy=policy, mode=mode); allfills.append(fr)
    io.mem_check(f'win {mid}')
FL = pd.DataFrame(allfills)
FL.to_parquet(f'{OUT}/fills.parquet', index=False)
summary = dict(day=day, mode='EXPLORATORY', delay_ms=DELAY, median_recv_minus_source_ms=IN_MS, size=SIZE, arms={})
for (policy, mode), v in res.items():
    if len(v) == 0:
        continue
    W = pd.DataFrame(v, columns=['mid', 'pnl', 'net_end', 'n_fills', 'shares'])
    f = FL[(FL.policy == policy) & (FL['mode'] == mode)] if len(FL) else FL
    by_phase = {}
    if len(f):
        f = f.assign(ph=(f.t // 60 * 60).astype(int))
        by_phase = f.groupby('ph').agg(fills=('q', 'size'), shares=('q', 'sum'), settle=('settle', 'sum'), mo1=('mo1', 'sum'), mo60=('mo60', 'sum')).round(2).to_dict('index')
    summary['arms'][f'{policy}/{mode}'] = dict(
        windows=int(len(W)), gross_usd=round(float(W.pnl.sum()), 2), per_window_mean=round(float(W.pnl.mean()), 4),
        per_window_sd=round(float(W.pnl.std()), 4), windows_positive=int((W.pnl > 0).sum()),
        fills=int(W.n_fills.sum()), shares=round(float(W.shares.sum()), 1), abs_net_end_p95=round(float(W.net_end.abs().quantile(0.95)), 1),
        markout_usd={k: round(float(f[k].sum()), 2) for k in ('mo1', 'mo10', 'mo60')} if len(f) else {},
        rebate_usd_unverified=round(float(f.rebate.sum()), 2) if len(f) else 0.0, by_phase_60s=by_phase,
        by_kind={f'{k}/{"timed" if tm else "untimed"}': dict(fills=int(len(x)), shares=round(float(x.q.sum()), 1), mo1=round(float(x.mo1.sum()), 2),
                 settle=round(float(x.settle.sum()), 2), mo1_c_per_sh=round(float(100 * x.mo1.sum() / max(x.q.sum(), 1e-9)), 3))
                 for (k, tm), x in f.groupby(['kind', 'timed'])} if len(f) else {},
        order_age_s_at_fill=dict(zip(['p10', 'p50', 'p90'], f.age.quantile([.1, .5, .9]).round(3).tolist())) if len(f) else {})
summary['statuses'] = dict(st)
io.write_json(f'{OUT}/summary.json', summary)
print('[done]', {k: (v['gross_usd'], v['fills']) for k, v in summary['arms'].items()})
