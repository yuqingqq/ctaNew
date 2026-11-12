import sys, numpy as np, pandas as pd
S = '/home/yuqing/ctaNew/data/predict_fun/research/step1'
for d in sys.argv[1:]:
    LT = pd.read_parquet(f'{S}/{d}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 't_rel'])
    tk = pd.read_parquet(f'{S}/{d}/legs_tk.parquet').tk.values
    m60 = (LT.t_rel.values >= 0) & (LT.t_rel.values < 60)
    X = LT[(tk >= 0) & (LT.how.values > 0) & m60].copy(); X['tk'] = tk[(tk >= 0) & (LT.how.values > 0) & m60]
    X['off'] = X.t_ms - X.exec_s * 1000.0
    g = X.groupby('tk')
    agg = g.agg(n=('off', 'size'), mn=('off', 'min'), mx=('off', 'max'), nh1=('how', lambda h: int((h == 1).sum())), nh2=('how', lambda h: int((h == 2).sum())),
                nmid=('mid', 'nunique'), nsec=('exec_s', 'nunique'), npx=('ptick', 'nunique'))
    a = agg[agg.n >= 2]
    big = a[(a.mx - a.mn) > 500]
    print(d, 'taker orders with >=2 WS legs', len(a), 'spread>500ms', len(big), round(len(big) / len(a), 3),
          '| among them: all how=1', int(((big.nh2 == 0)).sum()), 'any how=2', int((big.nh2 > 0).sum()), 'multi-price', int((big.npx > 1).sum()), 'multi-sec', int((big.nsec > 1).sum()))
    print('   offsets of group min vs exec second: all orders |min|>550', round(float((a.mn.abs() > 550).mean()), 3), '; spread>500 orders: min<-550', round(float((big.mn < -550).mean()), 3), 'max>550', round(float((big.mx > 550).mean()), 3))
    # single-leg offset tails (all WS legs in first 60 s)
    o = X.off.values
    print('   all WS legs first60: share |off|>550', round(float((np.abs(o) > 550).mean()), 3), 'how1', round(float((np.abs(o[X.how.values == 1]) > 550).mean()), 3), 'how2', round(float((np.abs(o[X.how.values == 2]) > 550).mean()), 3))
