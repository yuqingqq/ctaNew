import sys, numpy as np, pandas as pd
S = '/home/yuqing/ctaNew/data/predict_fun/research/step1'
rng = np.random.default_rng(0)
for d in sys.argv[1:]:
    LT = pd.read_parquet(f'{S}/{d}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'taker_index', 'taker_quote', 'first_recv_ns'])
    tk = pd.read_parquet(f'{S}/{d}/legs_tk.parquet').tk.values
    def viol(t):
        X = LT.assign(tk=t)[t >= 0]
        X = X[X.tk.map(X.tk.value_counts()).values >= 2]
        nu = X.groupby('tk')[['mid', 'exec_s', 'side', 'taker_index', 'taker_quote']].nunique()
        return {c: round(float((nu[c] > 1).mean()), 4) for c in nu.columns}, len(nu)
    v, n = viol(tk)
    sh = tk.copy(); m = sh >= 0; sh[m] = rng.permutation(sh[m])
    vs, _ = viol(sh)
    # a shift by one row (misalignment falsifier)
    s1 = np.r_[tk[1:], -1]
    v1, _ = viol(s1)
    print(d, 'orders>=2 legs', n, 'share with >1 value', v, '| shuffled', vs, '| shifted by 1 row', v1)
