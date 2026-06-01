"""
iter-023 fail-fast pre-check: cross-sectional MAX / lottery-demand effect as a
SHORT-SIDE overvaluation signal in alpha-residual space.

Literature:
- Cheuathonghua/.. "Lottery-like preferences and the MAX effect in the cryptocurrency
  market" (Financial Innovation 2021, 10.1186/s40854-021-00291-9).
- "Higher moments, extreme returns, and cross-section of cryptocurrency returns"
  (Finance Research Letters 2020, S1544612320303135) -> idiosyncratic skewness /
  MAX NEGATIVELY predict next-period returns.
- Bali/Cakici/Whitelaw (2011 JFE) equity MAX origin.
Mechanism (prior to be ERA-STABLE): lottery preference + limits-to-arbitrage /
short-sale constraints -> coins with a recent extreme up-move get over-bought
(lottery demand) and subsequently UNDER-perform. Short-side concentrated.

MAX definition (PIT): the single largest trailing 4h return over the trailing
WIN 4h-blocks. Built from return_pct shifted +48 bars (trailing 4h return), then
rolling-max on the 4h entry grid (non-overlapping blocks). Cross-sectionally ranked.

PRE-CHECKS, IN FAIL-FAST ORDER:
(i)  R-marginal: IC on PRED-RESIDUALIZED forward alpha-residual (does MAX add info
     pred LACKS? -- the iter-022 lesson: signal-orthogonality != outcome-residual info).
(ii) G7-transport: same sign on HL70 AND EXT 2021-26.
(iii) raw univariate IC + comparison vs rel_ret_1d (the iter-022 signal).
"""
import pandas as pd, numpy as np
pd.options.mode.chained_assignment = None

GRID = 48          # 4h entry cadence (bars)
WINS = [3, 6, 12]  # trailing 4h-blocks for MAX (12h, 24h, 48h)

def load_panel(path):
    return pd.read_parquet(path, columns=['symbol','open_time','return_pct',
                                          'return_1d','alpha_vs_btc_realized'])

def build(p):
    p = p.sort_values(['symbol','open_time']).copy()
    # trailing 4h return ending at t = forward return_pct from t-48 (PIT)
    p['trail4h'] = p.groupby('symbol')['return_pct'].shift(GRID)
    # MAX over trailing WIN non-overlapping 4h blocks -> use the 4h grid then rolling-max
    p['ot'] = pd.to_datetime(p['open_time'])
    uts = np.sort(p['ot'].unique())
    grid = set(uts[::GRID])
    g4 = p[p['ot'].isin(grid)].copy().sort_values(['symbol','ot'])
    for w in WINS:
        # rolling max of the last w trailing-4h blocks (each block PIT, ending <= t)
        g4[f'max_{w}'] = g4.groupby('symbol')['trail4h'].transform(
            lambda s: s.rolling(w, min_periods=max(2, w//2)).max())
    # rel_ret_1d (iter-022 comparison signal)
    gg = g4.groupby('ot')
    g4['rel_ret_1d'] = g4['return_1d'] - gg['return_1d'].transform('mean')
    return g4

def xs_ic(df, sig, tgt):
    d = df.dropna(subset=[sig, tgt])
    g = d.groupby('ot')
    ics = g.apply(lambda x: x[sig].corr(x[tgt], method='spearman') if x[sig].nunique()>3 else np.nan).dropna()
    if len(ics) < 3: return np.nan, np.nan, len(ics)
    return ics.mean(), ics.mean()/(ics.std()/np.sqrt(len(ics))), len(ics)

def pred_residualized_ic(df, sig, pred_col, tgt):
    """Regress tgt on pred PER CYCLE, take residual fwd return, IC of sig vs residual.
    This is what pred LEAVES ON THE TABLE (the R-marginal pre-check)."""
    d = df.dropna(subset=[sig, pred_col, tgt]).copy()
    g = d.groupby('ot')
    # per-cycle center pred and tgt, residualize tgt on pred
    d['pred_c']  = d[pred_col] - g[pred_col].transform('mean')
    d['tgt_c']   = d[tgt]      - g[tgt].transform('mean')
    beta = ((d['pred_c']*d['tgt_c']).groupby(d['ot']).transform('sum') /
            (d['pred_c']**2).groupby(d['ot']).transform('sum').replace(0,np.nan))
    d['tgt_resid'] = d['tgt_c'] - beta*d['pred_c']
    return xs_ic(d, sig, 'tgt_resid')

# ---------------- run ----------------
PRED = ('research/convexity_portable_2026-05-20/results/_cache/'
        'x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet')

print("="*78)
print("iter-023 MAX/lottery pre-check  (signs: lottery prior = NEGATIVE IC; short side)")
print("="*78)

for name, path in [('HL70','outputs/vBTC_features/panel_hl70.parquet'),
                   ('EXT ','outputs/vBTC_features/panel_ext2021_v0.parquet')]:
    g4 = build(load_panel(path))
    print(f"\n### {name}  syms={g4.symbol.nunique()}  cycles={g4.ot.nunique()}")
    print("  -- (ii) G7 transport: raw univariate XS-IC vs fwd alpha-residual --")
    for w in WINS:
        m,t,n = xs_ic(g4, f'max_{w}', 'alpha_vs_btc_realized')
        print(f"    IC(max_{w:>2}  -> alpha_resid): {m:+.4f}  t {t:+.2f}  n {n}")
    m,t,n = xs_ic(g4,'rel_ret_1d','alpha_vs_btc_realized')
    print(f"    IC(rel_ret_1d -> alpha_resid): {m:+.4f}  t {t:+.2f}  n {n}  [iter-022 ref]")

    if name == 'HL70':
        pred = pd.read_parquet(PRED, columns=['symbol','open_time','pred','alpha_A'])
        pred['ot'] = pd.to_datetime(pred['open_time'])
        mm = g4.merge(pred[['symbol','ot','pred','alpha_A']], on=['symbol','ot'], how='inner')
        print(f"  -- (i) R-MARGINAL: IC on PRED-RESIDUALIZED fwd alpha_A  (merged n_rows={len(mm)}) --")
        for w in WINS:
            mr,tr,nr = pred_residualized_ic(mm, f'max_{w}', 'pred', 'alpha_A')
            mu,tu,_  = xs_ic(mm, f'max_{w}', 'alpha_A')
            print(f"    max_{w:>2}: raw IC {mu:+.4f}(t{tu:+.1f}) | PRED-RESID IC {mr:+.4f}(t{tr:+.1f})")
        # baselines
        mr,tr,_ = pred_residualized_ic(mm,'rel_ret_1d','pred','alpha_A')
        mu,tu,_ = xs_ic(mm,'rel_ret_1d','alpha_A')
        print(f"    rel_1d: raw IC {mu:+.4f}(t{tu:+.1f}) | PRED-RESID IC {mr:+.4f}(t{tr:+.1f})  [iter-022 ref]")
        mu,tu,_ = xs_ic(mm,'pred','alpha_A')
        print(f"    pred  : raw IC {mu:+.4f}(t{tu:+.1f})  [production predictor]")

print("\n" + "="*78)
print("DECISION RULE (fail-fast):")
print(" (i) R-marginal: PRED-RESID IC of MAX must be NONZERO & same sign as raw (pred")
print("     must NOT already absorb it -- iter-022 rel_ret_1d died because pred did).")
print(" (ii) transport: MAX raw IC same sign on HL70 AND EXT.")
print(" If both pass for some W -> propose short-side MAX overlay; else NO-CANDIDATE.")
