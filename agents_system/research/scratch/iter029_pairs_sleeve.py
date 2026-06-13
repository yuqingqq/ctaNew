"""
iter-029 — Pairs / cointegration stat-arb sleeve (PIT, transport-first).

Mechanism: trade mean-reversion of a stationary log-price spread between
cointegrated crypto pairs. Different alpha source than the cross-sectional
rank book (spread reversion, not XS rank). Tests whether it (a) transports
HL70-era vs EXT 2021-26 and (b) diversifies the baseline book.

PIT discipline:
  - cointegration test + hedge ratio + spread mean/std estimated ONLY on a
    trailing formation window ending at t-1 (re-formed every REFORM bars).
  - trade signal at bar t uses z computed from formation stats + price at t-1
    (decision lagged one bar); PnL realized t->t+1 (4h grid).
  - no full-sample info anywhere.

Parameter-light: ENTRY=2.0, EXIT=0.5, STOP=3.5 (textbook), FORM=360 (60d),
REFORM=180 (30d), ADF p<0.05, top-N pairs by ADF stat.
"""
import numpy as np, pandas as pd, sys
from itertools import combinations

# ADF 5% critical value with constant (MacKinnon, large-sample) ~ -2.86.
ADF_CRIT_5 = -2.86

def adf_tstat(y, lag=1):
    """ADF with constant + `lag` lagged differences. Returns t-stat on rho.
    Δy_t = a + rho*y_{t-1} + sum gamma_i Δy_{t-i} + e.  More negative = more
    stationary; reject unit root if t < ADF_CRIT_5."""
    y = np.asarray(y, float)
    dy = np.diff(y)
    n = len(dy)
    if n < 20:
        return 0.0
    # build design
    yl = y[:-1]            # y_{t-1} aligned with dy
    rows = []
    target = dy[lag:]
    cols = [np.ones(n-lag), yl[lag:]]
    for i in range(1, lag+1):
        cols.append(dy[lag-i:-i] if i < n else np.zeros(n-lag))
    X = np.column_stack(cols)
    try:
        beta, *_ = np.linalg.lstsq(X, target, rcond=None)
        resid = target - X@beta
        dof = len(target) - X.shape[1]
        if dof <= 0: return 0.0
        s2 = (resid@resid)/dof
        XtX_inv = np.linalg.inv(X.T@X)
        se = np.sqrt(s2*XtX_inv[1,1])
        if se <= 0: return 0.0
        return beta[1]/se
    except Exception:
        return 0.0

FORM   = 360     # 60d formation window (4h bars)
REFORM = 180     # re-form every 30d
ENTRY, EXIT, STOP = 2.0, 0.5, 3.5
ADF_P  = 0.05
MAX_PAIRS = 10   # cap active pairs (equal weight)
COST_BPS = 4.5   # per leg
BARS_PER_YR = 6*365  # 4h cycles/yr

def eg_test(la, lb):
    """Engle-Granger: regress la on lb (+const), ADF on residual.
    returns (beta, mu, resid_std, adf_tstat)."""
    x = np.column_stack([np.ones_like(lb), lb])
    coef, *_ = np.linalg.lstsq(x, la, rcond=None)
    mu, beta = coef
    resid = la - (mu + beta*lb)
    stat = adf_tstat(resid, lag=1)
    return beta, mu, resid.std(ddof=1), stat

def run_pairs(logp, label, max_pairs=MAX_PAIRS, entry=ENTRY, exit_=EXIT,
              stop=STOP, form=FORM, reform=REFORM, cost_bps=COST_BPS, verbose=True):
    syms = list(logp.columns)
    P = logp.values            # (T, N) log prices on 4h grid
    T, N = P.shape
    idx = logp.index
    # per-bar log returns (for PnL); r[t] = P[t]-P[t-1]
    R = np.diff(P, axis=0, prepend=P[:1])  # R[0]=0
    # state per pair: position in {-1,0,+1} on the spread (+1 = long spread), beta, mu, sd
    book_pnl = np.zeros(T)     # net pnl per bar (bps of 1 unit gross per active leg-pair)
    book_cost = np.zeros(T)
    n_active = np.zeros(T)
    # active pairs dict: (i,j) -> dict(beta,mu,sd,pos)
    active = {}
    selected = []  # currently selected pair list (re-formed)
    last_form = -10**9
    turnover_legs = 0

    for t in range(form+1, T):
        # ---- re-form pairs (PIT: window ends at t-1) ----
        if t - last_form >= reform:
            last_form = t
            w0, w1 = t-form, t           # [t-form, t) ends at t-1
            cands = []
            sub = P[w0:w1]
            valid = np.where(np.isfinite(sub).all(axis=0))[0]
            for a, b in combinations(valid, 2):
                la, lb = sub[:,a], sub[:,b]
                # require both have moved (avoid degenerate)
                if la.std() < 1e-6 or lb.std() < 1e-6:
                    continue
                beta, mu, sd, stat = eg_test(la, lb)
                if stat < ADF_CRIT_5 and 0.2 < abs(beta) < 5.0 and sd > 1e-4:
                    cands.append((stat, a, b, beta, mu, sd))
            cands.sort(key=lambda x: x[0])  # most-negative ADF stat = strongest
            newsel = cands[:max_pairs]
            # rebuild active: keep position if pair persists, else close
            new_active = {}
            for stat,a,b,beta,mu,sd in newsel:
                key=(a,b)
                pos = active.get(key,{}).get('pos',0)
                new_active[key]=dict(beta=beta,mu=mu,sd=sd,pos=pos)
            # pairs that dropped out: realize close cost at t for any open pos
            for key,st in active.items():
                if key not in new_active and st['pos']!=0:
                    turnover_legs += 2
                    book_cost[t] += 2*cost_bps*1e-4
            active = new_active
            selected = newsel

        if not active:
            continue
        # ---- per-pair signal + pnl (PIT: z from price at t-1) ----
        npair = len(active)
        for (a,b), st in active.items():
            beta, mu, sd = st['beta'], st['mu'], st['sd']
            # spread at t-1
            if not (np.isfinite(P[t-1,a]) and np.isfinite(P[t-1,b])):
                continue
            spread_prev = P[t-1,a] - (mu + beta*P[t-1,b])
            z = spread_prev / sd
            pos = st['pos']
            new_pos = pos
            # entry/exit logic on z (signal decided at t-1, act at t)
            if pos==0:
                if z >  entry: new_pos = -1   # spread rich -> short spread (short A long B)
                elif z < -entry: new_pos = +1
            else:
                if abs(z) < exit_: new_pos = 0
                elif abs(z) > stop: new_pos = 0   # stop out
            # cost on position change (legs traded = 2 per pair on change)
            if new_pos != pos:
                book_cost[t] += 2*cost_bps*1e-4 / npair  # equal-weight across active pairs
                turnover_legs += 2
            st['pos'] = new_pos
            # pnl realized t-1 -> t from the position HELD over [t-1,t]
            # spread return = R[t,a] - beta*R[t,b]; long spread profits if spread rises
            if pos != 0 and np.isfinite(R[t,a]) and np.isfinite(R[t,b]):
                spr_ret = R[t,a] - beta*R[t,b]
                # normalize leg notional: gross = |1| + |beta|; equal weight across pairs
                gross = 1.0 + abs(beta)
                book_pnl[t] += pos * spr_ret / gross / npair
        n_active[t] = npair

    eq = np.cumsum(book_pnl - book_cost) * 1e4  # bps
    net = (book_pnl - book_cost) * 1e4          # per-bar net bps
    gross_bps = book_pnl*1e4
    cost_bps_ser = book_cost*1e4
    # metrics on bars with activity window
    m = net[form+1:]
    mu_, sd_ = m.mean(), m.std()
    sharpe = mu_/sd_*np.sqrt(BARS_PER_YR) if sd_>0 else 0.0
    peak = np.maximum.accumulate(eq); dd = eq-peak; maxdd = dd.min()
    calmar = (eq[-1]/ (len(m)/BARS_PER_YR)) / abs(maxdd) if maxdd<0 else np.nan
    res = dict(label=label, sharpe=sharpe, totpnl=eq[-1], maxdd=maxdd,
               gross=gross_bps.sum(), cost=cost_bps_ser.sum(),
               turnover_legs=turnover_legs, avg_active=np.nanmean(n_active[form+1:]),
               n_bars=len(m))
    if verbose:
        print(f"[{label}] Sharpe {sharpe:+.2f} totPnL {eq[-1]:+.0f}bps maxDD {maxdd:+.0f} "
              f"gross {gross_bps.sum():+.0f} cost {cost_bps_ser.sum():.0f} "
              f"avgPairs {res['avg_active']:.1f} legs {turnover_legs}")
    return res, pd.Series(net, index=idx), eq

if __name__=='__main__':
    out={}
    for lab,path in [('HL70', 'outputs/iter029/hl_logp_4h.parquet'),
                     ('EXT',  'outputs/iter029/ext_logp_4h.parquet')]:
        logp=pd.read_parquet(path)
        res,net,eq=run_pairs(logp,lab)
        net.to_frame('net').to_parquet(f'outputs/iter029/pairs_net_{lab}.parquet')
        out[lab]=res
    pd.DataFrame(out).T.to_csv('outputs/iter029/pairs_summary.csv')
    print('\n', pd.DataFrame(out).T[['sharpe','totpnl','maxdd','gross','cost','avg_active']])
