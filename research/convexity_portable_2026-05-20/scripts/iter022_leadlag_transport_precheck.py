"""
iter-022 pre-check: cross-crypto LEAD-LAG / spillover as alpha, in alpha-residual space.

Literature:
- Jia, Wu, Yan, Yin (2023, J. Empirical Finance) "A Seesaw Effect..." — large coins
  NEGATIVELY predict next-period small-coin returns (flight-to-hot / flee-from-cold).
- Guo, Sang, Tu, Wang (2024, J. Econ. Dynamics & Control) "Cross-cryptocurrency return
  predictability" — lagged returns of OTHER coins predict a focal coin; spillover via
  slow information diffusion + limited attention; LASSO long-short profitable OOS.

Mechanism prior to be ERA-STABLE: capital-rotation / information-diffusion behavioral
effect, not a price-feature regime. TRANSPORT-FIRST: compute XS-IC on BOTH HL70 and
EXT 2021-26; if sign flips or HL70-only -> REJECT as universe-overfit (fail-fast).

All signals are TRAILING (PIT). Target is the 4h-fwd alpha-residual the book trades.
4h entry cadence (every 48 bars) to avoid overlap-inflated t-stats.
"""
import pandas as pd, numpy as np
pd.options.mode.chained_assignment = None

def load(path):
    p = pd.read_parquet(path, columns=['symbol','open_time','return_1d','ret_3d',
                                        'alpha_vs_btc_realized','return_pct'] if 'ext' in path
                        else ['symbol','open_time','return_1d',
                               'alpha_vs_btc_realized','return_pct'])
    return p

def build_signals(p):
    p = p.sort_values(['open_time','symbol']).copy()
    # 4h entry grid: keep one bar per 48-bar block. Use bar index per symbol.
    p['ot'] = pd.to_datetime(p['open_time'])
    # entry grid = bars where minute aligns to a 4h cadence from the series start.
    # Simpler & robust: take every 48th unique timestamp.
    uts = np.sort(p['ot'].unique())
    grid = set(uts[::48])
    p = p[p['ot'].isin(grid)].copy()
    # cross-sectional demean of trailing return = focal coin's RELATIVE recent move
    g = p.groupby('ot')
    p['ret1d_xsmean'] = g['return_1d'].transform('mean')
    p['rel_ret_1d'] = p['return_1d'] - p['ret1d_xsmean']         # own relative lagged move
    p['peer_mean_ex'] = (g['return_1d'].transform('sum') - p['return_1d']) / (g['return_1d'].transform('count') - 1)  # mean of OTHER coins (spillover)
    return p

def xs_ic(df, sig, tgt):
    g = df.dropna(subset=[sig,tgt]).groupby('ot')
    ics = g.apply(lambda x: x[sig].corr(x[tgt], method='spearman') if x[sig].nunique()>3 else np.nan)
    ics = ics.dropna()
    t = ics.mean()/(ics.std()/np.sqrt(len(ics))) if len(ics)>2 else np.nan
    return ics.mean(), t, len(ics)

print("="*70)
for name, path in [('HL70','outputs/vBTC_features/panel_hl70.parquet'),
                   ('EXT ','outputs/vBTC_features/panel_ext2021_v0.parquet')]:
    p = build_signals(load(path))
    print(f"\n### {name}  syms={p.symbol.nunique()}  cycles={p.ot.nunique()}")
    for sig in ['rel_ret_1d','peer_mean_ex','return_1d']:
        for tgt in ['alpha_vs_btc_realized','return_pct']:
            m, t, n = xs_ic(p, sig, tgt)
            tag = tgt.replace('alpha_vs_btc_realized','alpha_resid').replace('return_pct','fwd_ret')
            print(f"  IC({sig:14s} -> {tag:11s}): {m:+.4f}  t {t:+.2f}  n {n}")
print("\n"+"="*70)
print("SEESAW prior: rel_ret_1d (own relative move) -> NEGATIVE (outperformers reverse)")
print("SPILLOVER prior: peer_mean_ex (others' move) -> POSITIVE (slow diffusion follow)")
print("Transport rule: ADOPT-path only if sign CONSISTENT + nonzero on BOTH universes.")
