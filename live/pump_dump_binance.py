"""RELIABILITY TEST (user #1): is the pump->dump blow-off short a reliable STANDALONE Binance strategy? Signal =
high-climax (top tercile) + low-funding (<=era median) blow-off (best from addendum 67). NET of realistic Binance
cost: round-trip spread/slippage (sweep 20/40/60 bps — froth names are wide) + funding drag over the 7d hold (a
SHORT receives funding when funding>0; the low-funding signal is ~0/negative -> the short PAYS -> drag). Both eras,
WEEK-CLUSTERED bootstrap CI. Reliable = both-eras net-positive with CI excluding 0. Uses the existing non-overlapping
entries in pump_both.csv (local universe; broad-froth-universe expansion is the further step if this clears).
"""
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
rng = np.random.default_rng(13); N_FUND = 21   # 7d hold x 3 funding/day

def wk_boot(df, col):
    df = df.copy(); df["wk"] = df["t"].dt.to_period("W").astype(str)
    grps = [g[col].values for _, g in df.groupby("wk")]
    if len(grps) < 4: return (np.nan, np.nan)
    out = [np.concatenate([grps[i] for i in rng.integers(0, len(grps), len(grps))]).mean() for _ in range(2000)]
    return np.percentile(out, [2.5, 97.5])

def main():
    e = pd.read_csv(SD / "pump_both.csv"); e["t"] = pd.to_datetime(e["t"], utc=True)
    e = e.dropna(subset=["funding", "climax", "fwd_ret"])
    e["short_gross"] = -e["fwd_ret"]
    e["fund_pnl"] = e["funding"] * N_FUND                       # short's funding PnL (>0 if funding>0)
    for era, sub in [("OOS 2023-25", e[e.t < pd.Timestamp("2025-10-01", tz="UTC")]),
                     ("RECENT 2025-10+", e[e.t >= pd.Timestamp("2025-10-01", tz="UTC")])]:
        sub = sub.copy()
        thr_c = sub["climax"].quantile(2 / 3); thr_f = sub["funding"].median()
        sig = sub[(sub.climax >= thr_c) & (sub.funding <= thr_f)].copy()    # high-climax + low-funding
        print(f"\n===== {era}: signal n={len(sig)} (high-climax + low-funding), {sig.sym.nunique()} syms, {sig['t'].dt.to_period('W').nunique()} weeks =====")
        g = sig["short_gross"].values; fp = sig["fund_pnl"].values
        print(f"  GROSS short mean {g.mean()*100:+5.1f}% | median {np.median(g)*100:+.1f}% | win {(g>0).mean()*100:.0f}%")
        print(f"  funding drag on the short: mean {fp.mean()*100:+.2f}% (short pays when funding<0)")
        for C in [0.0020, 0.0040, 0.0060]:                                  # 20/40/60 bps round-trip (as fractions)
            sig["net"] = sig["short_gross"] + sig["fund_pnl"] - C
            net = sig["net"].values; lo, up = wk_boot(sig, "net")
            flag = "RELIABLE (CI>0)" if lo > 0 else ("NEG (CI<0)" if up < 0 else "not sig (CI~0)")
            print(f"  NET @ {int(C*10000)}bps round-trip: mean {net.mean()*100:+5.1f}% [wkCI {lo*100:+.1f},{up*100:+.1f}] median {np.median(net)*100:+.1f}% -> {flag}")
    print("\n  (a SHORT can't be held cleanly through a squeeze; froth spread is often >60bps; broad froth universe adds OOS crashes but also delisting-halt risk)")
    print("BINANCEDONE")

if __name__ == "__main__":
    main()
