import sys, numpy as np, pandas as pd
S = '/home/yuqing/ctaNew/data/predict_fun/research/step1'
for d in sys.argv[1:]:
    LT = pd.read_parquet(f'{S}/{d}/legs_timed2.parquet', columns=['shares', 't_ms', 'how', 't_e2', 't_rel'])
    T = pd.read_parquet(f'{S}/{d}/legs_tk.parquet'); tk = T.tk.values; s = T.tk_status.values
    m60 = (LT.t_rel.values >= 0) & (LT.t_rel.values < 60); sh = LT.shares.values
    def anc_of(tkv):
        grp = np.where(tkv >= 0, tkv, -1 - np.arange(len(LT)))
        a_ = pd.Series(np.where(LT.how.values > 0, LT.t_ms.values, np.nan)).groupby(grp, sort=False).transform('min').values
        b_ = pd.Series(np.where((LT.how.values == 0) & np.isfinite(LT.t_e2.values), LT.t_e2.values, np.nan)).groupby(grp, sort=False).transform('min').values
        return np.where(np.isfinite(a_), a_, b_)
    A = anc_of(tk)
    tk2 = np.where(s == 5, -1, tk)          # what 10-31's failed refinement does: status-5 legs become singletons
    A2 = anc_of(tk2)
    ch = m60 & np.isfinite(A) & (np.abs(np.nan_to_num(A2, nan=1e18) - A) > 0)
    dif = (A2 - A)[ch & np.isfinite(A2)]
    print(d, 'first-60 shares in status 5', round(float(sh[m60 & (s == 5)].sum() / sh[m60].sum()), 3), '| anchors changed if refinement fails:', round(float(sh[ch].sum() / sh[m60].sum()), 4), 'of first-60 shares; |change|>100 ms', round(float(sh[ch & (np.abs(np.nan_to_num(A2 - A, nan=1e9)) > 100)].sum() / sh[m60].sum()), 4),
          'became untimed', round(float(sh[m60 & np.isfinite(A) & ~np.isfinite(A2)].sum() / sh[m60].sum()), 4), '| change p10/p50/p90 ms', np.percentile(dif, [10, 50, 90]).round(0).tolist() if len(dif) else None)
