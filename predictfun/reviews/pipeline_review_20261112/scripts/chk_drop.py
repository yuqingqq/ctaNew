import sys, numpy as np, pandas as pd
S = '/home/yuqing/ctaNew/data/predict_fun/research/step1'
for d in sys.argv[1:]:
    reg = pd.read_parquet(f'{S}/{d}/registry.parquet', columns=['mid', 'ws']); W0 = dict(zip(reg.mid, reg.ws * 1000.0))
    LT = pd.read_parquet(f'{S}/{d}/legs_timed2.parquet', columns=['mid', 'exec_s', 'shares', 't_ms', 'how', 't_e2'])
    tk = pd.read_parquet(f'{S}/{d}/legs_tk.parquet').tk.values
    grp = np.where(tk >= 0, tk, -1 - np.arange(len(LT)))
    a_ = pd.Series(np.where(LT.how.values > 0, LT.t_ms.values, np.nan)).groupby(grp, sort=False).transform('min').values
    b_ = pd.Series(np.where((LT.how.values == 0) & np.isfinite(LT.t_e2.values), LT.t_e2.values, np.nan)).groupby(grp, sort=False).transform('min').values
    anc = np.where(np.isfinite(a_), a_, b_); ws = LT.mid.map(W0).values; ex = LT.exec_s.values * 1000.0
    inwin = (ex >= ws) & (ex < ws + 60000)
    timed = np.isfinite(anc)
    out = timed & ~((anc >= ws - 2000) & (anc <= ws + 61000))
    off = anc - ex
    print(d, 'legs exec in [ws,ws+60s)', int(inwin.sum()), 'timed but anchor outside [ws-2s, ws+61s] (silently dropped)', int((inwin & out).sum()),
          '| legs with anchor in [ws, ws+60s) and exec outside it', int((timed & (anc >= ws) & (anc < ws + 60000) & ~inwin).sum()),
          '| anchor - exec second, first-60 legs: share < -600', round(float((off[inwin & timed] < -600).mean()), 3), 'share >= 600', round(float((off[inwin & timed] >= 600).mean()), 3))
