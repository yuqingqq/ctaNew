"""
iter-037 strategies — event-pooled, conservative cost for thin new perps.
(a) FADE: short at end of initial pump window, hold to horizon.
(b) FUNDING-CARRY SHORT: short when early funding extreme-positive (only late-2024/2025 cohort
    where funding-from-listing exists).
(c) EARLY MOMENTUM: long the post-listing trend.
Validation: cohort-transport, bootstrap CI across events, cost sensitivity.
"""
import pandas as pd, numpy as np, glob, os
np.seterr(all='ignore')
CACHE = '/home/yuqing/ctaNew/data/ml/cache'
events = pd.read_parquet('/home/yuqing/ctaNew/agents_system/research/scripts/iter037_events.parquet')

def load_hourly(sym):
    df = pd.read_parquet(f'{CACHE}/xs_feats_{sym}.parquet', columns=['close']).dropna()
    return df['close'].resample('1h').last().dropna()

H = {}
for s in events['sym']:
    h = load_hourly(s)
    if len(h) >= 24 * 32:
        H[s] = h
ev = events[events['sym'].isin(H)].copy().set_index('sym')
print(f'Usable events: {len(H)}')

def pnl_stats(pnls, label, cohort=None):
    p = np.array([x for x in pnls if np.isfinite(x)])
    n = len(p)
    mean = p.mean(); med = np.median(p); hit = (p > 0).mean()
    sh = mean / p.std() * np.sqrt(n) if p.std() > 0 else 0  # event-pooled t-like (per-event Sharpe scaled)
    t = mean / (p.std() / np.sqrt(n)) if p.std() > 0 else 0
    # bootstrap CI on mean PnL
    rng = np.random.default_rng(0)
    boots = [rng.choice(p, n, replace=True).mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f'  {label:<34} n={n:>3} meanPnL {mean:+.4f} med {med:+.4f} hit {hit:4.0%} '
          f't {t:+.2f} CI[{lo:+.3f},{hi:+.3f}]')
    return dict(label=label, n=n, mean=mean, med=med, hit=hit, t=t, ci_lo=lo, ci_hi=hi)

# ---------- helper: realized fwd return from entry hour to exit hour ----------
def fwd_ret(h, entry_h, exit_h):
    if len(h) <= exit_h:
        return None
    return h.iloc[exit_h] / h.iloc[entry_h] - 1.0

COST_BPS = 15  # conservative per-leg for thin new perps; RT = 2*cost
def net(raw, side, cost_bps=COST_BPS):
    # side +1 long, -1 short; subtract round-trip cost
    return side * raw - 2 * cost_bps / 1e4

# ================= STRATEGY (a) FADE =================
# at end of pump window W, short, hold to exit horizon. Test W in {1d,3d,7d} -> exit at +Hd later.
print('\n=== (a) FADE (short after initial window, hold to exit) ===')
fade_results = {}
for W, Hh in [(1, 7), (3, 7), (3, 14), (7, 14), (7, 30)]:
    pnls = []; per = {}
    for s, h in H.items():
        raw = fwd_ret(h, W * 24, Hh * 24)
        if raw is None: continue
        p = net(raw, -1)  # SHORT
        pnls.append(p); per[s] = p
    r = pnl_stats(pnls, f'short@{W}d -> exit@{Hh}d')
    fade_results[(W, Hh)] = (r, per)

# ================= STRATEGY (c) MOMENTUM =================
print('\n=== (c) EARLY MOMENTUM (long the trend) ===')
mom_results = {}
for W, Hh in [(3, 7), (3, 14), (7, 30), (1, 7)]:
    pnls = []; per = {}
    for s, h in H.items():
        raw = fwd_ret(h, W * 24, Hh * 24)
        if raw is None: continue
        p = net(raw, +1)  # LONG (follow)
        pnls.append(p); per[s] = p
    r = pnl_stats(pnls, f'long@{W}d -> exit@{Hh}d')
    mom_results[(W, Hh)] = (r, per)

# conditional momentum: only follow if pumped (ret first W > 0); fade only if pumped
print('\n=== (a2) CONDITIONAL FADE: short ONLY if first-window pumped (>+X%) ===')
for W, Hh, thr in [(3, 14, 0.0), (3, 14, 0.10), (7, 30, 0.0), (7, 30, 0.20)]:
    pnls = []
    for s, h in H.items():
        if len(h) <= Hh * 24: continue
        early = h.iloc[W * 24] / h.iloc[0] - 1.0
        if early <= thr: continue  # only short the ones that pumped
        raw = fwd_ret(h, W * 24, Hh * 24)
        pnls.append(net(raw, -1))
    pnl_stats(pnls, f'short@{W}d if pump>{thr:+.0%} ->exit@{Hh}d')

# ================= STRATEGY (b) FUNDING-CARRY SHORT =================
print('\n=== (b) FUNDING-CARRY SHORT (early funding extreme-positive) ===')
fund_rows = []
for s in H:
    fp = f'{CACHE}/funding_{s}.parquet'
    if not os.path.exists(fp): continue
    f = pd.read_parquet(fp)
    f['calc_time'] = pd.to_datetime(f['calc_time'], utc=True)
    f = f.set_index('calc_time').sort_index()
    h = H[s]; list_dt = h.index[0]
    # funding must start within 2d of listing to be "from-listing"
    if f.index[0] > list_dt + pd.Timedelta(days=2): continue
    # avg funding over first 3d
    early_f = f.loc[:list_dt + pd.Timedelta(days=3), 'funding_rate']
    if len(early_f) < 3: continue
    fund_rows.append((s, early_f.mean(), ev.loc[s, 'year']))
fund_df = pd.DataFrame(fund_rows, columns=['sym', 'early_fund', 'year'])
print(f'  events with funding-from-listing: {len(fund_df)}  (cohort {fund_df["year"].value_counts().sort_index().to_dict()})')
if len(fund_df):
    print(f'  early_fund (3d avg) dist: median {fund_df["early_fund"].median():+.5f} '
          f'q25 {fund_df["early_fund"].quantile(.25):+.5f} q75 {fund_df["early_fund"].quantile(.75):+.5f}')
    # short the top-funding tercile, hold 3d->14d
    for q, Hh in [(0.5, 14), (0.667, 14), (0.5, 7)]:
        thr = fund_df['early_fund'].quantile(q)
        sel = fund_df[fund_df['early_fund'] >= thr]['sym']
        pnls = []
        for s in sel:
            raw = fwd_ret(H[s], 3 * 24, Hh * 24)
            if raw is None: continue
            pnls.append(net(raw, -1))
        if pnls:
            pnl_stats(pnls, f'short top-fund q>={q:.2f} @3d->{Hh}d')

# ================= COHORT TRANSPORT (best fade config) =================
print('\n=== COHORT TRANSPORT — fade short@3d->exit@14d, by listing year ===')
W, Hh = 3, 14
for yr in [2023, 2024, 2025]:
    syms = ev[ev['year'] == yr].index
    pnls = []
    for s in syms:
        if s not in H: continue
        raw = fwd_ret(H[s], W * 24, Hh * 24)
        if raw is None: continue
        pnls.append(net(raw, -1))
    if pnls:
        pnl_stats(pnls, f'  {yr}', )

# ================= COST SENSITIVITY (best fade) =================
print('\n=== COST SENSITIVITY — fade short@3d->14d ===')
for cb in [5, 10, 15, 20]:
    pnls = []
    for s, h in H.items():
        raw = fwd_ret(h, 3 * 24, 14 * 24)
        if raw is None: continue
        pnls.append(net(raw, -1, cb))
    p = np.array(pnls)
    print(f'  cost {cb:>2}bps/leg: meanPnL {p.mean():+.4f} hit {(p>0).mean():4.0%} t {p.mean()/(p.std()/np.sqrt(len(p))):+.2f}')
