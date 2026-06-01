import pandas as pd, numpy as np
pd.options.mode.chained_assignment=None

# --- Load HL70 panel (funding + target) and preds ---
pan = pd.read_parquet('outputs/vBTC_features/panel_hl70.parquet')
pred = pd.read_parquet('research/convexity_portable_2026-05-20/results/_cache/x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet')

# 4h entry cadence = every 48 bars. Use preds' open_time grid (already the entry grid).
# Merge funding features onto pred rows by (symbol, open_time)
fcols=['funding_rate','funding_rate_z_7d','funding_rate_1d_change','alpha_vs_btc_realized']
m = pred.merge(pan[['symbol','open_time']+fcols], on=['symbol','open_time'], how='left')
print('merged', m.shape, 'pred rows', len(pred), 'match rate', m.funding_rate.notna().mean())

# return_pct in pred = forward total return over the 4h horizon (price). alpha_A = price alpha target.
# Use alpha_A as the price-alpha target the model trades; return_pct as raw fwd return.
print('cols', list(m.columns))
m=m.dropna(subset=['funding_rate','pred','alpha_A','return_pct'])

# Cross-sectional IC per cycle (open_time). Spearman.
def xs_ic(df, sig, tgt):
    g=df.groupby('open_time')
    ics=g.apply(lambda x: x[sig].corr(x[tgt],method='spearman') if x[sig].nunique()>3 else np.nan)
    return ics.dropna()

# Signals: funding LEVEL (carry sign: high funding -> shorts pay you, expect short), and z
# Carry hypothesis: SHORT high-funding (longs overpay -> price tends to fall + you collect funding as short)
# So funding should be NEGATIVELY related to forward return if carry+reversal aligned.
for sig in ['funding_rate','funding_rate_z_7d']:
    for tgt in ['alpha_A','return_pct']:
        ics=xs_ic(m,sig,tgt)
        t=ics.mean()/(ics.std()/np.sqrt(len(ics)))
        print(f'IC({sig} -> {tgt}): mean {ics.mean():+.4f}  t {t:+.2f}  n {len(ics)}')

# Orthogonality: corr of funding signal to existing pred (cross-sectionally, per cycle)
for sig in ['funding_rate','funding_rate_z_7d']:
    g=m.groupby('open_time')
    corrs=g.apply(lambda x: x[sig].corr(x['pred'],method='spearman') if x[sig].nunique()>3 else np.nan).dropna()
    print(f'XS corr({sig}, pred): mean {corrs.mean():+.4f}  median {corrs.median():+.4f}')

# IC of pred itself for reference
ics=xs_ic(m,'pred','alpha_A')
print(f'IC(pred -> alpha_A): mean {ics.mean():+.4f}  t {ics.mean()/(ics.std()/np.sqrt(len(ics))):+.2f}')
