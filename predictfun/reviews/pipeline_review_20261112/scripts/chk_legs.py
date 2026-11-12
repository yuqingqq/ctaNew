import sys, numpy as np, pandas as pd
S = '/home/yuqing/ctaNew/data/predict_fun/research/step1'
for d in sys.argv[1:]:
    LT = pd.read_parquet(f'{S}/{d}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 't_rel'])
    tkd = pd.read_parquet(f'{S}/{d}/legs_tk.parquet'); tk = tkd.tk.values
    assert len(tk) == len(LT)
    grp = np.where(tk >= 0, tk, -1 - np.arange(len(LT)))
    ws_t = np.where(LT.how.values > 0, LT.t_ms.values, np.nan)
    e2_t = np.where((LT.how.values == 0) & np.isfinite(LT.t_e2.values), LT.t_e2.values, np.nan)
    g = pd.Series(ws_t).groupby(grp, sort=False)
    a_ = g.transform('min').values; amax = g.transform('max').values
    b_ = pd.Series(e2_t).groupby(grp, sort=False).transform('min').values
    anc = np.where(np.isfinite(a_), a_, b_)
    m60 = (LT.t_rel.values >= 0) & (LT.t_rel.values < 60)
    sh = LT.shares.values
    own_ws = LT.how.values > 0
    src = np.select([own_ws, np.isfinite(a_), np.isfinite(b_)], ['own_ws', 'group_ws', 'e2'], 'none')
    tot = sh[m60].sum()
    print(d, 'first60 shares by anchor source', {k: round(float(sh[m60 & (src == k)].sum() / tot), 3) for k in ('own_ws', 'group_ws', 'e2', 'none')},
          '| tk<0 share', round(float(sh[m60 & (tk < 0)].sum() / tot), 3), 'status', {int(k): round(float(sh[m60 & (tkd.tk_status.values == k)].sum() / tot), 3) for k in np.unique(tkd.tk_status.values)})
    # spread of WS times inside one taker order (>=2 WS-timed legs)
    n_ws = g.transform('count').values
    sp = (amax - a_)[(tk >= 0) & (n_ws >= 2) & m60]
    print('   WS spread within taker order (legs, ms): p50/p90/p99/max', np.percentile(sp, [50, 90, 99]).round(0).tolist(), sp.max(), 'share>500ms', round(float((sp > 500).mean()), 4), 'share>100ms', round(float((sp > 100).mean()), 4))
    # anchor of how>0 legs moved by the group min
    mv = (LT.t_ms.values - anc)[own_ws & m60]
    print('   own WS t - anchor (ms) p50/p90/p99', np.percentile(mv, [50, 90, 99]).round(0).tolist(), 'share>100', round(float((mv > 100).mean()), 4))
    # e2 (min over group's how==0 legs) vs group WS min, for groups with both
    both = (tk >= 0) & np.isfinite(a_) & np.isfinite(b_) & m60
    df = pd.DataFrame(dict(g=grp[both], e=(b_ - a_)[both])).drop_duplicates('g')
    print('   groups with WS and E2 legs:', len(df), 'E2min - WSmin (ms) p05/p25/p50/p75/p95', np.percentile(df.e, [5, 25, 50, 75, 95]).round(0).tolist(), 'share E2 earlier by >100 ms', round(float((df.e < -100).mean()), 3), 'later >100', round(float((df.e > 100).mean()), 3))
    # per-leg t_e2 vs own WS time (legs timed by WS and carrying a cell e2)
    ok = own_ws & np.isfinite(LT.t_e2.values) & m60
    e = LT.t_e2.values[ok] - LT.t_ms.values[ok]
    print('   per-leg t_e2 - own WS t (ms) p05/p25/p50/p75/p95', np.percentile(e, [5, 25, 50, 75, 95]).round(0).tolist(), 'mean', round(float(e.mean()), 1))
