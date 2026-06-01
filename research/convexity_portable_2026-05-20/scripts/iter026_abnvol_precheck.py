"""
iter-026 fail-fast pre-check: ABNORMAL TRADING VOLUME / TRADE-COUNT as a
NEGATIVE cross-sectional predictor of forward alpha-residual.

Literature (online, May 2026):
- Garfinkel et al. "Disagreement and returns: the case of cryptocurrencies"
  (Financial Management 2025) -> abnormal volume / disagreement negatively
  predicts the cross-section of next-period crypto returns.
- "Predictability of crypto returns: the impact of trading behavior"
  (Journal of Behavioral & Experimental Finance 2023, S2214635023000266):
  on Binance, the highest-abnormal-volume quintile has SIGNIFICANTLY NEGATIVE
  next-day return minus lowest quintile -> attention/retail-driven overvaluation.
Mechanism (prior to be ERA-STABLE): an activity/attention SPIKE (volume or
trade-count surging above its own trailing norm) marks retail-attention buying
that subsequently underperforms -> high abnormal volume = SHORT, low = LONG.
This is a VOLUME/ACTIVITY input, structurally distinct from price-momentum,
the XS mean-rev pred, rel_ret_1d (i22), MAX (i23), funding (i21), basis (i25).

ABNORMAL VOLUME (PIT): for each symbol, the trailing-4h activity (buy_count_4h,
|signed_volume_4h|) divided by its own trailing-WIN rolling mean (all shifted +1
block so it ends at t, PIT), then cross-sectionally ranked per cycle.

PRE-CHECKS, FAIL-FAST:
(ii) G7 transport: raw univariate XS-IC same sign on HL70 AND 3yr/44-sym.
(i)  R-marginal: IC on PRED-RESIDUALIZED forward alpha (does pred already absorb it?).
(iii) construction-layer matched-random-pool (only if i+ii pass).
"""
import pandas as pd, numpy as np
pd.options.mode.chained_assignment = None

GRID = 48
WINS = [7, 14, 28]   # trailing 4h-blocks for the abnormal-volume baseline (~1d,2d,5d)
COLS = ['symbol','open_time','alpha_vs_btc_realized','return_1d',
        'buy_count_4h','signed_volume_4h','avg_trade_size_4h']

def build(path):
    p = pd.read_parquet(path, columns=COLS)
    p['ot'] = pd.to_datetime(p['open_time'])
    uts = np.sort(p['ot'].unique())
    grid = set(uts[::GRID])
    g4 = p[p['ot'].isin(grid)].copy().sort_values(['symbol','ot'])
    g4['absvol'] = g4['signed_volume_4h'].abs()
    for w in WINS:
        # trailing rolling mean of activity, SHIFTED +1 block (PIT: ends at t-1 block)
        for base in ['buy_count_4h','absvol']:
            roll = g4.groupby('symbol')[base].transform(
                lambda s: s.shift(1).rolling(w, min_periods=max(3, w//2)).mean())
            g4[f'abn_{base}_{w}'] = g4[base] / roll.replace(0, np.nan)
    # rel_ret_1d reference (the iter-022 signal)
    gg = g4.groupby('ot')
    g4['rel_ret_1d'] = g4['return_1d'] - gg['return_1d'].transform('mean')
    return g4

def xs_ic(df, sig, tgt):
    d = df.dropna(subset=[sig, tgt])
    g = d.groupby('ot')
    ics = g.apply(lambda x: x[sig].corr(x[tgt], method='spearman') if x[sig].nunique()>3 else np.nan).dropna()
    if len(ics) < 3: return np.nan, np.nan, 0
    return ics.mean(), ics.mean()/(ics.std()/np.sqrt(len(ics))), len(ics)

def pred_residualized_ic(df, sig, pred_col, tgt):
    d = df.dropna(subset=[sig, pred_col, tgt]).copy()
    g = d.groupby('ot')
    d['pred_c'] = d[pred_col] - g[pred_col].transform('mean')
    d['tgt_c']  = d[tgt]      - g[tgt].transform('mean')
    beta = ((d['pred_c']*d['tgt_c']).groupby(d['ot']).transform('sum') /
            (d['pred_c']**2).groupby(d['ot']).transform('sum').replace(0,np.nan))
    d['tgt_resid'] = d['tgt_c'] - beta*d['pred_c']
    return xs_ic(d, sig, 'tgt_resid')

SIGS = [f'abn_{b}_{w}' for w in WINS for b in ['buy_count_4h','absvol']]
PRED = ('research/convexity_portable_2026-05-20/results/_cache/'
        'x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet')

print("="*80)
print("iter-026 ABNORMAL VOLUME pre-check  (mechanism prior: NEGATIVE IC = high-vol underperforms)")
print("="*80)

panels = {'HL70':'outputs/vBTC_features/panel_hl70_v5_full.parquet',
          '44sy':'outputs/vBTC_features/panel_3yr_v5.parquet'}

g4s = {}
for name, path in panels.items():
    g4 = build(path)
    g4s[name] = g4
    print(f"\n### {name}  syms={g4.symbol.nunique()}  cycles={g4.ot.nunique()}")
    print("  -- (ii) G7 transport: raw univariate XS-IC vs fwd alpha-residual --")
    for s in SIGS:
        m,t,n = xs_ic(g4, s, 'alpha_vs_btc_realized')
        print(f"    IC({s:<22} -> alpha_resid): {m:+.4f}  t {t:+.2f}  n {n}")
    m,t,n = xs_ic(g4,'rel_ret_1d','alpha_vs_btc_realized')
    print(f"    IC({'rel_ret_1d':<22} -> alpha_resid): {m:+.4f}  t {t:+.2f}  n {n}  [i22 ref]")

# R-marginal on HL70
g4 = g4s['HL70']
pred = pd.read_parquet(PRED, columns=['symbol','open_time','pred','alpha_A'])
pred['ot'] = pd.to_datetime(pred['open_time'])
mm = g4.merge(pred[['symbol','ot','pred','alpha_A']], on=['symbol','ot'], how='inner')
print(f"\n  -- (i) R-MARGINAL: IC on PRED-RESIDUALIZED fwd alpha_A (HL70, merged n={len(mm)}) --")
for s in SIGS:
    mr,tr,_ = pred_residualized_ic(mm, s, 'pred', 'alpha_A')
    mu,tu,_ = xs_ic(mm, s, 'alpha_A')
    print(f"    {s:<22}: raw IC {mu:+.4f}(t{tu:+.1f}) | PRED-RESID IC {mr:+.4f}(t{tr:+.1f})")
mu,tu,_ = xs_ic(mm,'pred','alpha_A')
print(f"    {'pred':<22}: raw IC {mu:+.4f}(t{tu:+.1f})  [production predictor]")

print("\n" + "="*80)
print("DECISION (fail-fast): need (i) PRED-RESID IC nonzero same-sign as raw AND")
print(" (ii) raw IC SAME SIGN on HL70 AND 44sy. Else NO-CANDIDATE before any build.")
