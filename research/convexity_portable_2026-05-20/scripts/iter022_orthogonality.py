"""
iter-022 step 2: is the reversal lead-lag signal ORTHOGONAL to the production pred?
And does the residual (signal after projecting out pred) still carry IC on BOTH universes?

If rel_ret_1d is just a proxy for the existing mean-reversion pred -> NOT new (redundant).
We need: (a) XS corr(signal, pred) low/moderate, AND (b) residual-IC survives on BOTH.
HL70 has a cached pred; EXT does not (no HL70-V5 pred for the 23-sym panel) -> for EXT we
use the in-house target_z proxy is NOT a pred. So orthogonality is tested on HL70 (the
production universe where pred exists); transport of the RAW signal already shown on EXT.
"""
import pandas as pd, numpy as np
pd.options.mode.chained_assignment = None

pan = pd.read_parquet('outputs/vBTC_features/panel_hl70.parquet',
                      columns=['symbol','open_time','return_1d','alpha_vs_btc_realized'])
pred = pd.read_parquet('research/convexity_portable_2026-05-20/results/_cache/x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet',
                       columns=['symbol','open_time','pred','alpha_A','return_pct'])

m = pred.merge(pan, on=['symbol','open_time'], how='left')
m['ot'] = pd.to_datetime(m['open_time'])
# use pred's own grid (already 4h entry cadence) -> dedup to entry rows
m = m.dropna(subset=['return_1d','pred','alpha_A'])
g = m.groupby('ot')
m['ret1d_xsmean'] = g['return_1d'].transform('mean')
m['rel_ret_1d'] = m['return_1d'] - m['ret1d_xsmean']

def xs_ic(df, sig, tgt):
    df = df.reset_index(drop=True)
    gg = df.dropna(subset=[sig,tgt]).groupby('ot')
    ics = gg.apply(lambda x: x[sig].corr(x[tgt], method='spearman') if x[sig].nunique()>3 else np.nan).dropna()
    return ics.mean(), ics.mean()/(ics.std()/np.sqrt(len(ics))), len(ics)

# 1) raw IC of signal & pred vs alpha_A (the price-alpha target the book trades)
for sig in ['rel_ret_1d','pred']:
    mm,t,n = xs_ic(m, sig, 'alpha_A')
    print(f"IC({sig:11s} -> alpha_A): {mm:+.4f}  t {t:+.2f}  n {n}")

# 2) cross-sectional corr(signal, pred) per cycle
corrs = g.apply(lambda x: x['rel_ret_1d'].corr(x['pred'], method='spearman') if x['rel_ret_1d'].nunique()>3 else np.nan).dropna()
print(f"\nXS corr(rel_ret_1d, pred): mean {corrs.mean():+.4f}  median {corrs.median():+.4f}")

# 3) residualize rel_ret_1d on pred cross-sectionally per cycle, IC of the residual
def residualize(x):
    a = x['rel_ret_1d'].values.astype(float); b = x['pred'].values.astype(float)
    b = (b - b.mean())
    if (b**2).sum() < 1e-12:
        x['rel_resid'] = a - a.mean(); return x
    beta = (a*b).sum()/(b**2).sum()
    x['rel_resid'] = (a - a.mean()) - beta*b
    return x
m2 = m.groupby('ot', group_keys=False)[['rel_ret_1d','pred','alpha_A','ot']].apply(residualize)
mm,t,n = xs_ic(m2, 'rel_resid', 'alpha_A')
print(f"IC(rel_ret_1d AFTER removing pred -> alpha_A): {mm:+.4f}  t {t:+.2f}  n {n}")

# 4) also: incremental IC of pred AFTER removing rel_ret_1d (is pred just reversal?)
def residualize2(x):
    a = x['pred'].values.astype(float); b = x['rel_ret_1d'].values.astype(float)
    b = (b - b.mean())
    if (b**2).sum() < 1e-12:
        x['pred_resid'] = a - a.mean(); return x
    beta = (a*b).sum()/(b**2).sum()
    x['pred_resid'] = (a - a.mean()) - beta*b
    return x
m3 = m.groupby('ot', group_keys=False)[['rel_ret_1d','pred','alpha_A','ot']].apply(residualize2)
mm,t,n = xs_ic(m3, 'pred_resid', 'alpha_A')
print(f"IC(pred AFTER removing rel_ret_1d -> alpha_A): {mm:+.4f}  t {t:+.2f}  n {n}")
