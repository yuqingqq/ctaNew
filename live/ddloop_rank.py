"""Rank the dd-loop ledger vs baseline. Highlights candidates that improve the DRAWDOWN/tail without
killing Sharpe, AND per-regime (bear is the live-DD regime). Never judge on aggregate Sharpe alone.
usage: python3 live/ddloop_rank.py [sort_key=Sharpe]
"""
import sys, json
import numpy as np, pandas as pd
LED = "live/state/v3loop/ddloop/ledger.jsonl"
rows = [json.loads(l) for l in open(LED) if l.strip().startswith("{")]
df = pd.DataFrame(rows).drop_duplicates("tag", keep="last")
b = df[df.tag == "base"]
if len(b):
    b = b.iloc[0]
    df["dSh"] = df["Sharpe"] - b["Sharpe"]
    df["dMDD"] = df["maxDD"] - b["maxDD"]          # +ve = shallower DD (better)
    df["dCVaR"] = df["CVaR5"] - b["CVaR5"]          # +ve = better (less negative)
    df["dBearSh"] = df["bear_Sh"] - b["bear_Sh"]
    df["dBearMDD"] = df["bear_mDD"] - b["bear_mDD"]
sort = sys.argv[1] if len(sys.argv) > 1 else "Sharpe"
df = df.sort_values(sort, ascending=False)
cols = ["tag", "Sharpe", "dSh", "maxDD", "dMDD", "CVaR5", "dCVaR", "bear_Sh", "dBearSh", "bear_mDD", "dBearMDD", "totPnL"]
cols = [c for c in cols if c in df.columns]
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(df[cols].round(2).to_string(index=False))
print(f"\nbaseline: Sharpe {b['Sharpe']:.2f}  maxDD {b['maxDD']:.0f}  CVaR5 {b['CVaR5']:.0f}  bear_Sh {b['bear_Sh']:.2f}  bear_mDD {b['bear_mDD']:.0f}" if len(rows) and 'base' in df.tag.values else "")
print(f"\n# runs: {len(df)}")
