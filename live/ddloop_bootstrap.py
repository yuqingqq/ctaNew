"""Block-bootstrap CI on daily-Sharpe and on Sharpe-diff vs base for dd-loop candidate runs.
Stationary block bootstrap (mean block ~5 trading days) on daily PnL. Reports 90% CI + P(diff>0).
usage: python3 live/ddloop_bootstrap.py tag1 tag2 ...   (base auto-included)
"""
import sys
import numpy as np, pandas as pd
ANN = np.sqrt(365); RNG = np.random.default_rng(12345); NB = 2000; MEANBLK = 5

def daily(tag):
    c = pd.read_csv(f"live/state/v3loop/dd_{tag}/state/cycles.csv")
    c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    return c.sort_values("open_time").set_index("open_time")["pnl_bps"].resample("1D").sum().fillna(0)

def block_idx(n):
    idx = []; p = 1.0/MEANBLK
    while len(idx) < n:
        start = RNG.integers(0, n); L = RNG.geometric(p)
        idx.extend(((start + np.arange(L)) % n).tolist())
    return np.array(idx[:n])

def sharpe(x): return x.mean()/x.std()*ANN if x.std() > 0 else np.nan

tags = ["base"] + [t for t in sys.argv[1:] if t != "base"]
D = {t: daily(t) for t in tags}
idx0 = D["base"].index
for t in tags: D[t] = D[t].reindex(idx0).fillna(0).to_numpy()
n = len(idx0)
base = D["base"]
print(f"days={n}  bootstraps={NB}  block~{MEANBLK}d\n")
print(f"{'tag':22s} {'Sharpe':>7} {'CI90':>16} | {'dSharpe vs base':>16} {'CI90(diff)':>18} {'P(diff>0)':>9}")
boot_idx = [block_idx(n) for _ in range(NB)]
for t in tags:
    x = D[t]
    sh = sharpe(pd.Series(x))
    bs = np.array([sharpe(pd.Series(x[bi])) for bi in boot_idx])
    diff = np.array([sharpe(pd.Series(x[bi])) - sharpe(pd.Series(base[bi])) for bi in boot_idx])
    ci = np.nanpercentile(bs, [5, 95]); dci = np.nanpercentile(diff, [5, 95])
    pp = float((diff > 0).mean())
    print(f"{t:22s} {sh:>7.2f} [{ci[0]:>6.2f},{ci[1]:>6.2f}] | "
          f"{(sh-sharpe(pd.Series(base))):>16.2f} [{dci[0]:>7.2f},{dci[1]:>6.2f}] {pp:>9.2f}")
