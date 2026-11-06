"""Step 5a (exploratory): decision-time feature rows for a short-horizon Predict mid model.

Grid: every 50 ms of each live window [ws, ws+300) on the Tokyo clock (Predict frame known at
updateTimestampMs + 20 ms; Binance rows/trades known at E + 5 ms; relay known at this host's receipt - 72 ms).
Every feature uses only data known at the grid time tau. Targets use the VENUE book (latest frame with
updateTimestampMs <= tau + h) minus the KNOWN mid at tau, in cents: y250, y1000.

Usage: python3 -m predictfun.research.pf_step5_features 20261023
Output: step5/<day>/rows.parquet (float32)
"""
import sys, os, math
import numpy as np, pandas as pd
from scipy.special import ndtr
from . import pf_io as io

day = sys.argv[1]
S1 = f'{io.OUT}/step1/{day}'; OUT = f'{io.OUT}/step5/{day}'; os.makedirs(OUT, exist_ok=True)
P_IN, B_IN, NET = 20.0, 5.0, 72.0
STEP = 50.0
TRD = f'{io.ROOT}/data/mm_hf/raw/trade/BTCUSDT'
reg = pd.read_parquet(f'{S1}/registry.parquet')
reg = reg[reg.outcome.isin(['Up', 'Down']) & reg.start_price.notna()].sort_values('ws')
grid = np.concatenate([w * 1000.0 + np.arange(0, 300_000, STEP) for w in reg.ws.values])
gmid = np.repeat(reg.mid.values, 6000)
n = len(grid)
F = {}

# ---------------- Binance bookTicker + trades, streamed per hour ----------------
RET_H = [25, 50, 100, 250, 500, 1000, 3000]
for c in [f'bn_r{h}' for h in RET_H] + ['bn_imb', 'bn_age', 'bn_mid']:
    F[c] = np.full(n, np.nan, np.float64)
for c in ('bn_flow250', 'bn_flow1000', 'bn_flow5000', 'bn_ntr1000'):
    F[c] = np.full(n, np.nan, np.float64)
sec_mid = {}                       # 1 s bucket -> last mid (for sigma)
rel_ratio_t, rel_ratio_v = [], []  # basis samples (relay known time, relay/futures)
ref = [(r[0] / 1e6 - NET, float(r[2])) for h in io.hours_of(day, extra_after=1, extra_before=1) for r in io.ref_rows(h) if not r[3]]
ref = np.array(ref) if ref else np.empty((0, 2))
carry = None; tcarry = None
hours = io.hours_of(day, extra_after=1, extra_before=1)
for h in hours:
    p = f'{io.BN_DIR}/{h}.csv.gz'
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p, header=None, names=['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq'], usecols=['E', 'u', 'bid', 'bq', 'ask', 'aq'])
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df = df[(df.bid > 0) & (df.ask > df.bid)].drop_duplicates('u', keep='first')
    k = df.E.values.astype(np.float64) + B_IN; o = np.argsort(k, kind='stable')
    k = k[o]; m = ((df.bid.values + df.ask.values) / 2)[o]; imb = ((df.bq.values - df.aq.values) / (df.bq.values + df.aq.values))[o]
    del df
    if carry is not None:
        k = np.concatenate([carry[0], k]); m = np.concatenate([carry[1], m]); imb = np.concatenate([carry[2], imb])
    # 1 s mids for sigma
    sb = (k // 1000).astype(np.int64); last = np.r_[sb[1:] != sb[:-1], True]
    for s_, v in zip(sb[last], m[last]):
        sec_mid[int(s_)] = v
    # basis samples from relay prints in this span
    if len(ref):
        rr = ref[(ref[:, 0] >= k[0]) & (ref[:, 0] <= k[-1])]
        if len(rr):
            ii = np.searchsorted(k, rr[:, 0], side='right') - 1
            rel_ratio_t.append(rr[:, 0]); rel_ratio_v.append(rr[:, 1] / m[ii])
    # trades
    tp = f'{TRD}/{h}.csv.gz'
    if os.path.exists(tp):
        td = pd.read_csv(tp, header=None, names=['recv', 'E', 'T', 'id', 'px', 'qty', 'bm'], usecols=['E', 'qty', 'bm'])
        td = td.apply(pd.to_numeric, errors='coerce').dropna()
        tk = td.E.values.astype(np.float64) + B_IN; sq = np.where(td.bm.values == 0, 1.0, -1.0) * td.qty.values
        o = np.argsort(tk, kind='stable'); tk = tk[o]; sq = sq[o]
        del td
    else:
        tk = np.empty(0); sq = np.empty(0)
    if tcarry is not None:
        tk = np.concatenate([tcarry[0], tk]); sq = np.concatenate([tcarry[1], sq])
    csq = np.cumsum(sq)
    # grid points whose time falls inside the fresh part of this hour
    lo = carry[0][-1] if carry is not None and len(carry[0]) else -np.inf
    sel = np.where((grid > lo) & (grid <= k[-1]))[0]
    if len(sel):
        g = grid[sel]
        i = np.searchsorted(k, g, side='right') - 1; ok = i >= 0; ic = np.clip(i, 0, None)
        F['bn_mid'][sel] = np.where(ok, m[ic], np.nan); F['bn_imb'][sel] = np.where(ok, imb[ic], np.nan)
        F['bn_age'][sel] = np.where(ok, g - k[ic], np.nan)
        for hh in RET_H:
            j = np.searchsorted(k, g - hh, side='right') - 1
            F[f'bn_r{hh}'][sel] = np.where(ok & (j >= 0), 1e4 * np.log(m[ic] / m[np.clip(j, 0, None)]), np.nan)
        if len(tk):
            it = np.searchsorted(tk, g, side='right') - 1
            for hh, c in ((250, 'bn_flow250'), (1000, 'bn_flow1000'), (5000, 'bn_flow5000')):
                jt = np.searchsorted(tk, g - hh, side='right') - 1
                F[c][sel] = np.where(it >= 0, csq[np.clip(it, 0, None)], 0.0) - np.where(jt >= 0, csq[np.clip(jt, 0, None)], 0.0)
            F['bn_ntr1000'][sel] = it - (np.searchsorted(tk, g - 1000, side='right') - 1)
    keep = k >= k[-1] - 10_000; carry = (k[keep], m[keep], imb[keep])
    tkeep = tk >= (tk[-1] - 10_000 if len(tk) else 0); tcarry = (tk[tkeep], sq[tkeep])
    io.mem_check(f'bn {h}')
    print(f'[bn] {h} [mem] {io.rss_mib()} MiB', flush=True)
rel_t = np.concatenate(rel_ratio_t) if rel_ratio_t else np.empty(0); rel_v = np.concatenate(rel_ratio_v) if rel_ratio_v else np.empty(0)
o = np.argsort(rel_t); rel_t = rel_t[o]; rel_v = rel_v[o]

# ---------------- per window: Predict book + pending flow + contract ----------------
cols = ['mid', 't_ms'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 4) for k in 'ps']
books = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=cols) for f in sorted(os.listdir(f'{S1}/books'))], ignore_index=True)
books.sort_values(['mid', 't_ms'], inplace=True, kind='stable')
BBY = {m: g for m, g in books.groupby('mid', sort=False)}
pend = pd.read_parquet(f'{S1}/pend_inc.parquet', columns=['mid', 't_ms', 'side', 'inc'])
PBY = {m: g for m, g in pend.groupby('mid', sort=False)}
PF_COLS = ['pf_spread', 'pf_mid', 'pf_micro_dev', 'pf_imb1', 'pf_imb3', 'pf_lbs', 'pf_las', 'pf_age', 'pf_nfr1000',
           'pf_dm250', 'pf_dm1000', 'pf_dm5000', 'pf_onesided', 'pf_pend_bid1000', 'pf_pend_ask1000', 'pf_pend_tot5000',
           't_rel', 'tau_rem', 'sigma', 'basis_bps', 'p_bn', 'gap', 'y250', 'y1000']
for c in PF_COLS:
    F[c] = np.full(n, np.nan, np.float64)
pos = 0
for _, R in reg.iterrows():
    mid = R.mid; ws = R.ws * 1000.0; sl = slice(pos, pos + 6000); pos += 6000
    g = BBY.get(mid)
    tau = grid[sl]
    F['t_rel'][sl] = (tau - ws) / 1000.0; F['tau_rem'][sl] = 300.0 - F['t_rel'][sl]
    # sigma from 1 s mids over [ws-1800, ws)
    s0 = int(R.ws) - 1800
    sm = np.array([sec_mid.get(s0 + i, np.nan) for i in range(1800)])
    sm = pd.Series(sm).ffill().values; lr = np.diff(np.log(sm)); lr = lr[np.isfinite(lr)]
    sig = math.sqrt(np.mean(lr ** 2)) if len(lr) > 1000 else np.nan
    F['sigma'][sl] = sig
    a = np.searchsorted(rel_t, tau - 60_000); b = np.searchsorted(rel_t, tau, side='right')
    bas = np.array([np.median(rel_v[x:y]) if y - x >= 5 else np.nan for x, y in zip(a, b)])
    F['basis_bps'][sl] = 1e4 * (bas - 1)
    S = F['bn_mid'][sl]
    z = np.log(S * bas / float(R.start_price)) / (sig * np.sqrt(np.maximum(F['tau_rem'][sl], 1e-3)))
    F['p_bn'][sl] = 100 * ndtr(z)
    if g is None:
        continue
    tv = g.t_ms.values.astype(np.float64); kn = tv + P_IN
    b1, a1 = g.b1_p.values, g.a1_p.values
    valid = (b1 >= 0) & (a1 > b1)
    midv = np.where(valid, (a1 + b1) / 2.0, np.nan)
    bs1, as1 = g.b1_s.values, g.a1_s.values
    i = np.searchsorted(kn, tau, side='right') - 1; ok = i >= 0; ic = np.clip(i, 0, None)
    vv = ok & valid[ic]
    F['pf_onesided'][sl] = np.where(ok, ~valid[ic], np.nan).astype(float)
    F['pf_spread'][sl] = np.where(vv, (a1 - b1)[ic], np.nan)
    F['pf_mid'][sl] = np.where(vv, midv[ic], np.nan)
    micro = (b1 * as1 + a1 * bs1) / np.maximum(bs1 + as1, 1e-9)
    F['pf_micro_dev'][sl] = np.where(vv, (micro - midv)[ic], np.nan)
    F['pf_imb1'][sl] = np.where(vv, ((bs1 - as1) / np.maximum(bs1 + as1, 1e-9))[ic], np.nan)
    sb3 = sum(np.where(g[f'b{j}_p'].values >= 0, g[f'b{j}_s'].values, 0) for j in (1, 2, 3))
    sa3 = sum(np.where(g[f'a{j}_p'].values >= 0, g[f'a{j}_s'].values, 0) for j in (1, 2, 3))
    F['pf_imb3'][sl] = np.where(vv, ((sb3 - sa3) / np.maximum(sb3 + sa3, 1e-9))[ic], np.nan)
    F['pf_lbs'][sl] = np.where(vv, np.log1p(bs1[ic]), np.nan); F['pf_las'][sl] = np.where(vv, np.log1p(as1[ic]), np.nan)
    F['pf_age'][sl] = np.where(ok, tau - kn[ic], np.nan)
    F['pf_nfr1000'][sl] = i - (np.searchsorted(kn, tau - 1000, side='right') - 1)
    # last valid known mid at tau and at tau - h (forward-filled over invalid frames)
    mf = pd.Series(midv).ffill().values
    cur = np.where(ok, mf[ic], np.nan)
    for hh in (250, 1000, 5000):
        j = np.searchsorted(kn, tau - hh, side='right') - 1
        F[f'pf_dm{hh}'][sl] = cur - np.where(j >= 0, mf[np.clip(j, 0, None)], np.nan)
    F['gap'][sl] = F['p_bn'][sl] - cur
    # targets on the venue clock
    for hh, c in ((250, 'y250'), (1000, 'y1000')):
        j = np.searchsorted(tv, tau + hh, side='right') - 1
        F[c][sl] = np.where(j >= 0, mf[np.clip(j, 0, None)], np.nan) - cur
    pg = PBY.get(mid)
    if pg is not None:
        pk = pg.t_ms.values.astype(np.float64) + P_IN; o = np.argsort(pk, kind='stable'); pk = pk[o]
        for side, c in ((0, 'pf_pend_bid1000'), (1, 'pf_pend_ask1000')):
            cs = np.cumsum(np.where(pg.side.values[o] == side, pg.inc.values[o], 0.0))
            a_ = np.searchsorted(pk, tau, side='right') - 1; b_ = np.searchsorted(pk, tau - 1000, side='right') - 1
            F[c][sl] = np.where(a_ >= 0, cs[np.clip(a_, 0, None)], 0) - np.where(b_ >= 0, cs[np.clip(b_, 0, None)], 0)
        cs = np.cumsum(pg.inc.values[o])
        a_ = np.searchsorted(pk, tau, side='right') - 1; b_ = np.searchsorted(pk, tau - 5000, side='right') - 1
        F['pf_pend_tot5000'][sl] = np.where(a_ >= 0, cs[np.clip(a_, 0, None)], 0) - np.where(b_ >= 0, cs[np.clip(b_, 0, None)], 0)
    else:
        F['pf_pend_bid1000'][sl] = 0; F['pf_pend_ask1000'][sl] = 0; F['pf_pend_tot5000'][sl] = 0
io.mem_check('windows')
D = pd.DataFrame({k: v.astype(np.float32) for k, v in F.items()})
D.insert(0, 'mid', gmid); D.insert(1, 'tau', grid)
D.to_parquet(f'{OUT}/rows.parquet', index=False)
print('[done]', day, 'rows', len(D), 'valid y1000', int(D.y1000.notna().sum()), '[mem]', io.rss_mib(), 'MiB')
print(D.describe().T[['count', 'mean', 'std']].round(3).to_string())
