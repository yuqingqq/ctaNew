"""
iter-027 Phase-2 feature-engineering transport + marginal pre-check.

GOAL: identify new feature families that (a) have cross-sectional IC vs forward
4h alpha-residual that is SIGN-CONSISTENT and non-trivial on BOTH HL70 and EXT
2021-26 (transport, the #1 killer), and (b) add MARGINAL info the production V0
pred lacks (IC on PRED-RESIDUALIZED fwd alpha on HL70). Survivors -> propose a
pooled-LightGBM model rebuild that includes them.

Feature families (SOTA-grounded):
 1. TAKER order-flow imbalance  (Anastasopoulos-Gradojevic 2025; metrics
    sum_taker_long_short_vol_ratio) -- per-sym log + trailing z, PIT.
 2. OI dynamics (OI 1d change, OI-vs-price divergence) -- metrics.
 3. Realized VOL-OF-VOL (std of rolling realized vol; from return_pct, both panels).
 4. Multi-timeframe momentum/reversal (ret over 1d/3d/7d; both panels).
 5. Amihud illiquidity proxy (|ret|/atr as cheap dollar-vol-free proxy; both panels).

Every feature is built on the 4h non-overlapping entry grid, PIT (only info
ending <= t). Cross-sectional Spearman IC per cycle, averaged.
"""
import pandas as pd, numpy as np, glob, os
pd.options.mode.chained_assignment = None

GRID = 48  # 4h cadence in 5m bars
PRED = ('research/convexity_portable_2026-05-20/results/_cache/'
        'x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet')

# ---------- IC helpers (reused from iter-023) ----------
def xs_ic(df, sig, tgt):
    d = df.dropna(subset=[sig, tgt])
    g = d.groupby('ot')
    ics = g.apply(lambda x: x[sig].corr(x[tgt], method='spearman')
                  if x[sig].nunique() > 3 else np.nan).dropna()
    if len(ics) < 5:
        return np.nan, np.nan, len(ics)
    return ics.mean(), ics.mean()/(ics.std()/np.sqrt(len(ics))), len(ics)

def pred_resid_ic(df, sig, pred_col, tgt):
    d = df.dropna(subset=[sig, pred_col, tgt]).copy()
    g = d.groupby('ot')
    d['pred_c'] = d[pred_col] - g[pred_col].transform('mean')
    d['tgt_c']  = d[tgt]      - g[tgt].transform('mean')
    num = (d['pred_c']*d['tgt_c']).groupby(d['ot']).transform('sum')
    den = (d['pred_c']**2).groupby(d['ot']).transform('sum').replace(0, np.nan)
    d['tgt_resid'] = d['tgt_c'] - (num/den)*d['pred_c']
    return xs_ic(d, sig, 'tgt_resid')

# ---------- load panel on 4h grid ----------
def load_grid(path):
    want = ['symbol','open_time','return_pct','alpha_vs_btc_realized',
            'return_1d','atr_pct','rvol_7d']
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(path).schema.names)
    cols = [c for c in want if c in have]
    p = pd.read_parquet(path, columns=cols).sort_values(['symbol','open_time'])
    p['ot'] = pd.to_datetime(p['open_time'])
    uts = np.sort(p['ot'].unique())
    grid = set(uts[::GRID])
    # build trailing multi-bar returns on 5m first (PIT), then subset to grid
    # ret over N 4h-blocks ending at t: use return_pct (4h fwd) shifted back
    p['trail4h'] = p.groupby('symbol')['return_pct'].shift(GRID)  # trailing 4h ret
    g = p[p['ot'].isin(grid)].copy().sort_values(['symbol','ot'])
    return g

# ---------- build klines-only features (BOTH panels) ----------
def build_price_feats(g):
    grp = g.groupby('symbol')
    # multi-TF momentum / reversal from trailing 4h blocks
    g['mom_2'] = grp['trail4h'].transform(lambda s: s.rolling(2,  min_periods=2).sum())   # ~8h
    g['mom_6'] = grp['trail4h'].transform(lambda s: s.rolling(6,  min_periods=4).sum())   # ~1d
    g['mom_18']= grp['trail4h'].transform(lambda s: s.rolling(18, min_periods=12).sum())  # ~3d
    g['mom_42']= grp['trail4h'].transform(lambda s: s.rolling(42, min_periods=28).sum())  # ~7d
    # realized vol over trailing blocks (vol of 4h returns)
    g['rv_6']  = grp['trail4h'].transform(lambda s: s.rolling(6,  min_periods=4).std())
    g['rv_42'] = grp['trail4h'].transform(lambda s: s.rolling(42, min_periods=28).std())
    # VOL-OF-VOL: std of the rolling-6 vol series
    g['vov']   = grp['rv_6'].transform(lambda s: s.rolling(18, min_periods=12).std())
    # vol-of-vol normalised (regime-free)
    g['vov_n'] = g['vov'] / g['rv_42'].replace(0,np.nan)
    # short-term reversal (negative recent ret = expected outperformer)
    g['rev_2'] = -g['mom_2']
    g['rev_6'] = -g['mom_6']
    # Amihud illiquidity proxy: |trailing 4h ret| / atr (no dollar-vol needed)
    g['amihud'] = g['trail4h'].abs() / g['atr_pct'].replace(0,np.nan)
    g['amihud_z'] = grp['amihud'].transform(
        lambda s: (s - s.rolling(126, min_periods=42).mean())
                  / s.rolling(126, min_periods=42).std().replace(0,np.nan))
    return g

# ---------- build metrics (taker / OI) features, merged onto grid ----------
def load_metrics(symbols):
    frames = []
    for s in symbols:
        f = f'data/ml/cache/metrics_{s}.parquet'
        if not os.path.exists(f):
            continue
        m = pd.read_parquet(f, columns=['symbol','sum_taker_long_short_vol_ratio',
                                        'sum_open_interest'])
        m = m.reset_index().rename(columns={'create_time':'ts'})
        frames.append(m)
    if not frames:
        return None
    M = pd.concat(frames, ignore_index=True)
    M['ts'] = pd.to_datetime(M['ts'])
    return M.sort_values(['symbol','ts'])

def build_metric_feats(g, M):
    # resample metrics to 4h grid ends (PIT: last value <= t, then shift 1 grid)
    M = M.copy()
    M['taker_log'] = np.log(M['sum_taker_long_short_vol_ratio'].clip(1e-3, 1e3))
    M['oi'] = M['sum_open_interest']
    out = []
    for s, ms in M.groupby('symbol'):
        ms = ms.set_index('ts').sort_index()
        # trailing aggregates ending at t (use 5m native then sample at grid)
        ms['taker_1d']  = ms['taker_log'].rolling('1D',  min_periods=50).mean()
        ms['taker_z7d'] = ((ms['taker_1d'] - ms['taker_1d'].rolling('7D', min_periods=300).mean())
                           / ms['taker_1d'].rolling('7D', min_periods=300).std().replace(0,np.nan))
        ms['oi_chg_1d'] = ms['oi'].pct_change(periods=288, fill_method=None)
        ms.index = ms.index.astype('datetime64[ns, UTC]')
        gg = g[g.symbol==s][['ot']].copy().sort_values('ot')
        gg['ot'] = gg['ot'].astype('datetime64[ns, UTC]')
        if len(gg)==0:
            continue
        merged = pd.merge_asof(gg, ms[['taker_1d','taker_z7d','oi_chg_1d']].reset_index(),
                               left_on='ot', right_on='ts', direction='backward',
                               tolerance=pd.Timedelta('1D'))
        merged['symbol'] = s
        # PIT lag: shift one grid step so feature uses only <= t-4h info
        for c in ['taker_1d','taker_z7d','oi_chg_1d']:
            merged[c] = merged.groupby('symbol')[c].shift(1)
        out.append(merged[['symbol','ot','taker_1d','taker_z7d','oi_chg_1d']])
    feats = pd.concat(out, ignore_index=True)
    return g.merge(feats, on=['symbol','ot'], how='left')

# ====================== RUN ======================
FAMILIES = ['mom_6','mom_18','mom_42','rev_2','rev_6','vov','vov_n','amihud_z',
            'taker_1d','taker_z7d','oi_chg_1d']

results = {}
for name, path in [('HL70','outputs/vBTC_features/panel_hl70.parquet'),
                   ('EXT','outputs/vBTC_features/panel_ext2021_v0.parquet')]:
    print("="*80); print(f"### {name}  loading...");
    g = load_grid(path)
    g = build_price_feats(g)
    syms = sorted(g.symbol.unique())
    M = load_metrics(syms)
    if M is not None:
        g = build_metric_feats(g, M)
    else:
        for c in ['taker_1d','taker_z7d','oi_chg_1d']:
            g[c] = np.nan
    print(f"   syms={g.symbol.nunique()} cycles={g.ot.nunique()} "
          f"range={g.ot.min().date()}..{g.ot.max().date()}")
    tgt = 'alpha_vs_btc_realized'
    res = {}
    for f in FAMILIES:
        if f not in g.columns or g[f].notna().sum() < 1000:
            res[f] = (np.nan, np.nan, 0)
            continue
        res[f] = xs_ic(g, f, tgt)
    results[name] = res
    print(f"  {'feature':<12} {'rawIC':>9} {'t':>7} {'n_cyc':>7}")
    for f in FAMILIES:
        m,t,n = res[f]
        print(f"  {f:<12} {m:>+9.4f} {t:>+7.2f} {n:>7}")
    if name=='HL70':
        # marginal: pred-residualized IC on HL70
        pred = pd.read_parquet(PRED, columns=['symbol','open_time','pred','alpha_A'])
        pred['ot'] = pd.to_datetime(pred['open_time'])
        mm = g.merge(pred[['symbol','ot','pred','alpha_A']], on=['symbol','ot'], how='inner')
        mu_pred,t_pred,_ = xs_ic(mm,'pred','alpha_A')
        print(f"\n  -- MARGINAL (pred-resid IC on alpha_A), merged n={len(mm)} --")
        print(f"  pred raw IC vs alpha_A: {mu_pred:+.4f} (t{t_pred:+.1f})")
        print(f"  {'feature':<12} {'rawIC_A':>9} {'PREDRESID':>10} {'t_resid':>9}")
        marg = {}
        for f in FAMILIES:
            if f not in mm.columns or mm[f].notna().sum()<1000:
                marg[f]=(np.nan,np.nan); continue
            mr,tr,_ = pred_resid_ic(mm, f, 'pred', 'alpha_A')
            mu,_,_  = xs_ic(mm, f, 'alpha_A')
            marg[f]=(mr,tr)
            print(f"  {f:<12} {mu:>+9.4f} {mr:>+10.4f} {tr:>+9.2f}")
        results['HL70_marg']=marg

print("\n"+"="*80)
print("TRANSPORT + MARGINAL TABLE (sign-consistent both + marginal nonzero -> survivor)")
print(f"  {'feature':<12} {'HL70_IC':>9} {'EXT_IC':>9} {'sign?':>6} {'PREDRESID':>10}")
for f in FAMILIES:
    h = results['HL70'][f][0]; e = results['EXT'][f][0]
    pr = results.get('HL70_marg',{}).get(f,(np.nan,))[0]
    sign = 'same' if (not np.isnan(h) and not np.isnan(e) and np.sign(h)==np.sign(e)) else 'FLIP'
    print(f"  {f:<12} {h:>+9.4f} {e:>+9.4f} {sign:>6} {pr:>+10.4f}")
