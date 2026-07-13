"""ENHANCE TEST (user #2): does the robust pump->dump finding (funding separates squeeze from dump — HIGH funding
= squeeze-prone) improve v4's EXISTING short leg? Split v4's bottom-2 short selections by funding-at-cycle, both
eras, at BOOK LEVEL (short-leg selection PnL, path-independent). If HIGH-funding shorts are much worse (they
squeeze), a funding gate (skip/down-weight them) would lift the short book. Self-contained: v4 _honest books +
funding cache, NO klines. Cross-era robustness is the bar.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
FC = Path("/home/yuqing/ctaNew/data/ml/cache"); S = Path("/home/yuqing/ctaNew/live/state/convexity")

def fund_series(sym):
    f = FC / f"funding_{sym}.parquet"
    if not f.exists(): return None
    d = pd.read_parquet(f); tc = "calc_time" if "calc_time" in d.columns else "open_time"
    if tc not in d.columns or "funding_rate" not in d.columns: return None
    d[tc] = pd.to_datetime(d[tc], utc=True)
    s = d.set_index(tc)["funding_rate"].sort_index(); return s[~s.index.duplicated()]

def dsh(x):
    return x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else np.nan

def analyze(base_dir, era):
    b = pd.read_parquet(S / base_dir / "v0full_hl60.parquet"); b["open_time"] = pd.to_datetime(b["open_time"], utc=True)
    rows = []
    for t, g in b.groupby("open_time"):
        if len(g) < 5: continue
        for _, r in g.nsmallest(2, "pred").iterrows():
            rows.append((t, r["symbol"], -float(r["alpha_A"]) * 1e4))     # short PnL bps = -alpha
    d = pd.DataFrame(rows, columns=["t", "sym", "pnl"])
    d["funding"] = np.nan                                                  # batch funding lookup per symbol
    for sym, g in d.groupby("sym"):
        fs = fund_series(sym)
        if fs is None: continue
        d.loc[g.index, "funding"] = fs.reindex(g["t"], method="ffill").values
    d = d.dropna(subset=["funding"])
    d["ft"] = pd.qcut(d["funding"].rank(method="first"), 3, labels=["LO", "mid", "HI"], duplicates="drop")
    print(f"\n===== {era}: {len(d)} short legs w/ funding ({d.sym.nunique()} syms) | ALL short daily Sharpe {dsh(d.assign(day=d.t.dt.date).groupby('day').pnl.sum()):+.2f} =====")
    for ft in ["LO", "mid", "HI"]:
        x = d[d.ft == ft]; day = x.assign(day=x["t"].dt.date).groupby("day")["pnl"].sum()
        print(f"  funding {ft} (med {x.funding.median()*100:+.3f}%): short PnL/leg {x.pnl.mean():+6.1f}bps | daily Sharpe {dsh(day):+.2f} | hit {(x.pnl>0).mean()*100:.0f}%")
    # gated book: short only LO+mid funding (skip HI = squeeze-prone), vs baseline (all)
    for name, sub in [("baseline (all shorts)", d), ("GATED: skip HI-funding shorts", d[d.ft != "HI"])]:
        day = sub.assign(day=sub["t"].dt.date).groupby("day")["pnl"].sum()
        print(f"    {name:32s}: daily Sharpe {dsh(day):+.2f} | total {sub.pnl.sum():+.0f}bps | n {len(sub)}")

if __name__ == "__main__":
    analyze("hl_v4base_oos_honest", "OOS 2023-25")
    analyze("hl_tgt_res_base_honest", "RECENT 2025-10+")
    print("\n  (book-level short-leg PnL, path-independent; funding gate = skip high-funding shorts to avoid squeezes)")
    print("V4SHORTFUNDDONE")
