"""
iter-028 Phase-2: TARGET x HORIZON predictability heatmap, transport-first.

Core question: is the cross-sectional signal MORE predictable + transport-stable
at a DIFFERENT target/horizon than the current 4h BTC-beta-residual?

Grid:
  HORIZONS (forward): 1h, 4h(current), 12h, 1d, 3d, 1w.
  TARGETS:
    (a) beta_resid  : forward beta-residual (4h native = alpha_vs_btc_realized;
                      other H = chained raw ret minus XS-mean-beta * mkt proxy,
                      operationalised as raw ret minus cross-sec mean = MARKET resid,
                      PLUS we report native 4h beta-resid for the 4h column).
    (b) raw         : forward raw return (chained 4h blocks / 5m for 1h).
    (c) mkt_resid   : forward raw - cross-sectional mean (alt-index residual). Clean both panels.
    (d) vol_scaled  : forward raw / trailing realized vol (per-sym), then we IC predictors vs it.
  Note: beta_resid and mkt_resid coincide structurally at non-4h horizons here
        (we don't have per-row beta off-grid); we keep mkt_resid as the residual target
        and report native 4h beta_resid separately for the anchor cell.

PREDICTORS (simple, robust, PIT) tested for predictability of each target cell:
    rev_short  : -trailing 1d return  (cross-sec reversal)
    rev_4h     : -trailing 4h return
    mom_long   : trailing 7d return   (momentum)
    mom_30d    : trailing 30d return  (the bull-regime ranker)
    funding_z  : funding_rate_z_7d
    pred_proxy : per-sym z of -trailing-1d (cheap mean-rev proxy; production pred is on alpha)
  For each (target,horizon) cell we report the BEST |IC| predictor that is SIGN-CONSISTENT
  across HL70 and EXT (transport-stable). IC = mean per-cycle cross-sectional Spearman.

All forward targets built on the 4h (48-bar) non-overlapping grid by CHAINING the
native 4h return_pct (PIT: the predictor uses only info <= t; the target uses [t, t+H]).
1h horizon uses the 5m bars directly (12-bar forward).
"""
import pandas as pd, numpy as np, pyarrow.parquet as pq
pd.options.mode.chained_assignment = None

GRID = 48                       # 4h in 5m bars
H_BLOCKS = {'4h':1, '12h':3, '1d':6, '3d':18, '1w':42}   # in 4h blocks
PANELS = {'HL70':'outputs/vBTC_features/panel_hl70.parquet',
          'EXT' :'outputs/vBTC_features/panel_ext2021_v0.parquet'}

def xs_ic(df, sig, tgt):
    d = df.dropna(subset=[sig, tgt])
    g = d.groupby('ot')
    ics = g.apply(lambda x: x[sig].corr(x[tgt], method='spearman')
                  if (x[sig].nunique() > 3 and x[tgt].nunique() > 3) else np.nan).dropna()
    if len(ics) < 5:
        return np.nan, np.nan, len(ics)
    return ics.mean(), ics.mean()/(ics.std()/np.sqrt(len(ics))), len(ics)

def load(path):
    cols = ['symbol','open_time','return_pct','alpha_vs_btc_realized',
            'funding_rate_z_7d','atr_pct']
    have = set(pq.ParquetFile(path).schema.names)
    cols = [c for c in cols if c in have]
    p = pd.read_parquet(path, columns=cols).sort_values(['symbol','open_time'])
    p['ot'] = pd.to_datetime(p['open_time'])
    return p

def build(path):
    p = load(path)
    # ---- 1h forward raw from 5m bars (12-bar forward), built BEFORE subsetting ----
    # return_pct is 4h-fwd; we need 1h-fwd. Approx forward 1h via trailing-shifted:
    # fwd_1h(t) = product of 5m fwd returns over 12 bars. We don't have 5m fwd ret,
    # but return_pct(t) = 4h fwd. fwd_1h(t) ~ cumret using the 5m grid of return_pct
    # is not directly available. Instead reconstruct 5m log-price proxy from 4h ret:
    # price_ratio over 4h = (1+return_pct). We approximate 1h fwd as the change in the
    # 4h-trailing series. Cleanest: use difference of cumulative.
    # Build a per-sym 5m forward 1h return via overlapping 4h ret is messy; we instead
    # take 1h ~ smallest reliable horizon = use return over next 12 bars from a price proxy.
    grp = p.groupby('symbol')
    # log-price proxy: cumulative of (return over each 5m) -- but we only have 4h ret.
    # Construct 5m simple return from 4h ret by undoing the 48-bar overlap:
    # r5m(t) ~ logprice(t+1)-logprice(t). We have R4h(t)=logprice(t+48)-logprice(t).
    # => logprice(t) reconstructable up to constant via cumulative of R4h on a 48-spaced grid.
    # For 1h we instead use the GRID anyway (cheapest robust): treat 1h as unavailable
    # off-grid and report 4h as the shortest clean horizon. We still attempt 1h on the
    # 48-grid by using a 1h-trailing predictor; but 1h FORWARD target needs sub-grid.
    # PRAGMATIC: reconstruct logprice on 5m grid from chained R4h is impossible (overlap).
    # So 1h-forward target = approximate via (1+R4h)^(1/4)-1 expected per-hour? No — drop 1h
    # as a clean cell and note it. We keep 4h..1w which chain cleanly on the grid.

    # ---- subset to 4h grid ----
    uts = np.sort(p['ot'].unique())
    grid = pd.Index(uts[::GRID])
    g = p[p['ot'].isin(grid)].copy().sort_values(['symbol','ot'])
    grp = g.groupby('symbol')

    # trailing predictors (PIT: end <= t)
    g['trail4h'] = grp['return_pct'].shift(1)            # raw ret over [t-4h, t]
    g['trail1d'] = grp['return_pct'].transform(lambda s: s.shift(1).rolling(6, min_periods=4).sum())
    g['trail7d'] = grp['return_pct'].transform(lambda s: s.shift(1).rolling(42, min_periods=28).sum())
    g['trail30d']= grp['return_pct'].transform(lambda s: s.shift(1).rolling(180, min_periods=120).sum())
    g['rev_short'] = -g['trail1d']
    g['rev_4h']    = -g['trail4h']
    g['mom_long']  = g['trail7d']
    g['mom_30d']   = g['trail30d']
    g['funding_z'] = g.get('funding_rate_z_7d', np.nan)
    # per-sym z of -trail1d (cheap mean-rev pred proxy)
    g['pred_proxy'] = grp['rev_short'].transform(
        lambda s: (s - s.rolling(180, min_periods=60).mean())/s.rolling(180, min_periods=60).std().replace(0,np.nan))
    # trailing realized vol for vol-scaling
    g['rvol4h'] = grp['return_pct'].transform(lambda s: s.shift(1).rolling(42, min_periods=28).std())

    # ---- FORWARD targets at each horizon (chain raw 4h blocks) ----
    # fwd raw over H blocks = sum of return_pct(t), return_pct(t+1)... return_pct(t+H-1)
    # (approx log-additive; return_pct small). For H=1 it's just return_pct.
    for hname, hb in H_BLOCKS.items():
        if hb == 1:
            fwd = g['return_pct']
        else:
            # forward sum of next hb non-overlapping 4h returns
            fwd = grp['return_pct'].transform(
                lambda s: s.rolling(hb, min_periods=hb).sum().shift(-(hb-1)))
        g[f'raw_{hname}'] = fwd
        # market resid = raw - cross-sectional mean (per cycle)
        g[f'mktres_{hname}'] = fwd - fwd.groupby(g['ot']).transform('mean')
        # vol-scaled
        g[f'vs_{hname}'] = fwd / g['rvol4h'].replace(0, np.nan)
    # native 4h beta-residual (the anchor)
    g['betares_4h'] = g['alpha_vs_btc_realized']
    return g

PREDICTORS = ['rev_short','rev_4h','mom_long','mom_30d','funding_z','pred_proxy']

# build both
G = {nm: build(p) for nm, p in PANELS.items()}
for nm in G:
    print(f"{nm}: syms={G[nm].symbol.nunique()} cyc={G[nm].ot.nunique()} "
          f"{G[nm].ot.min().date()}..{G[nm].ot.max().date()}")

# ---- compute IC table: for each (target_family, horizon, predictor) on both panels ----
TARGET_FAMS = {'raw':'raw', 'mktres':'mktres', 'vs':'vs'}
rows = []
for fam_key in TARGET_FAMS:
    for hname in H_BLOCKS:
        tgt = f'{fam_key}_{hname}'
        for pr in PREDICTORS:
            ic_h, t_h, n_h = xs_ic(G['HL70'], pr, tgt) if tgt in G['HL70'] else (np.nan,np.nan,0)
            ic_e, t_e, n_e = xs_ic(G['EXT'],  pr, tgt) if tgt in G['EXT']  else (np.nan,np.nan,0)
            same = (not np.isnan(ic_h) and not np.isnan(ic_e) and np.sign(ic_h)==np.sign(ic_e))
            rows.append(dict(target=fam_key, horizon=hname, pred=pr,
                             HL70_IC=ic_h, HL70_t=t_h, EXT_IC=ic_e, EXT_t=t_e,
                             transport=same, min_abs=min(abs(ic_h) if not np.isnan(ic_h) else 0,
                                                         abs(ic_e) if not np.isnan(ic_e) else 0)))
# anchor cell: native 4h beta-residual
for pr in PREDICTORS:
    ic_h,t_h,_ = xs_ic(G['HL70'], pr, 'betares_4h')
    ic_e,t_e,_ = xs_ic(G['EXT'],  pr, 'betares_4h')
    same = (not np.isnan(ic_h) and not np.isnan(ic_e) and np.sign(ic_h)==np.sign(ic_e))
    rows.append(dict(target='betares', horizon='4h', pred=pr,
                     HL70_IC=ic_h, HL70_t=t_h, EXT_IC=ic_e, EXT_t=t_e,
                     transport=same, min_abs=min(abs(ic_h),abs(ic_e)) if not (np.isnan(ic_h) or np.isnan(ic_e)) else 0))

R = pd.DataFrame(rows)
R.to_csv('research/convexity_portable_2026-05-20/results/iter028_th_grid.csv', index=False)

# ---- BEST transport-stable predictor per (target,horizon) cell ----
print("\n" + "="*100)
print("BEST TRANSPORT-STABLE PREDICTOR per (target x horizon) cell")
print("(min_abs = min(|HL70 IC|,|EXT IC|) among sign-consistent predictors; '-' if none transport)")
print("="*100)
def best_cell(sub):
    s = sub[sub.transport]
    if len(s)==0: return None
    return s.loc[s.min_abs.idxmax()]

print(f"\n{'target':<9} {'horizon':<7} {'best_pred':<11} {'HL70_IC':>9} {'EXT_IC':>9} {'min|IC|':>9}")
HORDER = ['4h','12h','1d','3d','1w']
for fam in ['betares','raw','mktres','vs']:
    for hname in (['4h'] if fam=='betares' else HORDER):
        sub = R[(R.target==fam)&(R.horizon==hname)]
        bc = best_cell(sub)
        if bc is None:
            print(f"{fam:<9} {hname:<7} {'(none)':<11} {'':>9} {'':>9} {'':>9}")
        else:
            print(f"{fam:<9} {hname:<7} {bc['pred']:<11} {bc.HL70_IC:>+9.4f} {bc.EXT_IC:>+9.4f} {bc.min_abs:>9.4f}")

# ---- full per-predictor dump for the strongest cells ----
print("\n" + "="*100)
print("FULL GRID (all predictors, transport flag)")
print("="*100)
for fam in ['betares','raw','mktres','vs']:
    print(f"\n--- target = {fam} ---")
    print(f"{'hor':<5} {'pred':<11} {'HL70_IC':>9} {'HL70_t':>7} {'EXT_IC':>9} {'EXT_t':>7} {'transp':>7}")
    hs = ['4h'] if fam=='betares' else HORDER
    for hname in hs:
        for pr in PREDICTORS:
            r = R[(R.target==fam)&(R.horizon==hname)&(R.pred==pr)]
            if len(r)==0: continue
            r=r.iloc[0]
            print(f"{hname:<5} {pr:<11} {r.HL70_IC:>+9.4f} {r.HL70_t:>+7.1f} "
                  f"{r.EXT_IC:>+9.4f} {r.EXT_t:>+7.1f} {str(r.transport):>7}")
print("\nDONE")
