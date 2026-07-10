"""PRODUCTION (v4 + configs + gatings) long/short by regime — from the full-stack replay cycles.

Reads the KEEPSET4 replay cycles.csv (all overlays: REGIME_GATE, DD-stop, BEAR_MODE=equal,
BULL_GROSS_MULT=0, deep-bull mom1d_long, inv_sqrt_vol; 1.0x parity — the 0.5x live cap is a uniform
final scale on top). long_alpha_bps / short_alpha_bps are the per-cycle GATED leg contributions
(residual). Compare to the vanilla (pre-gate) longshort_regime.py to see what the configs DO to each
regime's raw edge.
"""
import numpy as np, pandas as pd, glob
from pathlib import Path
SCR = "/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad"

def sh_daily(x, col):
    d = pd.to_datetime(x["open_time"], utc=True).dt.date
    dr = x[[col]].groupby(d).sum()[col]
    return dr.mean()/dr.std()*np.sqrt(365) if dr.std() > 0 else np.nan

def main():
    for era, path in (("OOS 2023-25", f"{SCR}/replay_oos_clean/cycles.csv"),
                      ("RECENT 2025-10+", f"{SCR}/replay_recent_clean/cycles.csv")):
        if not Path(path).exists(): print(f"{era}: missing"); continue
        c = pd.read_csv(path); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
        print(f"\n===== {era}: PRODUCTION (v4 + configs + gatings) long/short by regime =====")
        print(f"  {'regime':<9}{'n':>5} | {'gross':>6} {'stop%':>6} | {'LONG net':>9} {'L Sh':>6} | {'SHORT net':>10} {'S Sh':>6} | {'BOOK net':>9} {'B Sh':>6}")
        for rg in ["side","bear","bull","deepbull","ALL"]:
            g = c if rg == "ALL" else c[c["regime"] == rg]
            if len(g) < 3: continue
            la, sa = g["long_alpha_bps"].mean(), g["short_alpha_bps"].mean()
            grr = g["gross_after_stop"].mean() if "gross_after_stop" in g else np.nan
            stp = g["stop_engaged"].mean()*100 if "stop_engaged" in g else np.nan
            print(f"  {rg:<9}{len(g):>5} | {grr:>6.2f} {stp:>5.0f}% | {la:>+9.1f} {sh_daily(g,'long_alpha_bps'):>+6.2f} | "
                  f"{sa:>+10.1f} {sh_daily(g,'short_alpha_bps'):>+6.2f} | {g['pnl_bps'].mean():>+9.1f} {sh_daily(g,'pnl_bps'):>+6.2f}")
    print("\n(gross = mean gross_after_stop [1.0=full, <1=de-grossed by gate/stop]; legs are GATED contributions.")
    print(" bull->~0 = BULL_GROSS_MULT=0 flattens it; deep-bull = mom1d LONG-only overlay.)\nPRODLSDONE")

if __name__ == "__main__":
    main()
