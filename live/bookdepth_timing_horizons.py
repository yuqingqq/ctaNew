"""Horizon sweep for the final signal (pos = -clip(z90(agg imb1), +/-1.5), EWMA-smoothed). Earlier IC strengthened
4h->3d; does it keep strengthening, and what HOLDING horizon trades best net of cost? Two sweeps, both eras:
 (1) IC( agg_imb -> forward H-bar market return ) for H = 1d..10d (where does the predictive edge peak?)
 (2) BACKTEST: position EWMA-smoothed to HL=H (holding horizon), net 10bps -> Sharpe / maxDD / turnover per horizon.
z-window fixed at 90 (15d) to isolate horizon. Single series -> longer H = fewer independent bets, wider CIs.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_market_timing import agg_ob
from live.bookdepth_timing_backtest import market_ret, daily, stats
from live.bookdepth_crowding_horizon import block_ic
CUT = pd.Timestamp("2025-10-01", tz="UTC")
HS = [("1d", 6), ("2d", 12), ("3d", 18), ("5d", 30), ("7d", 42), ("10d", 60)]

def zc(s, W=90):
    return (s - s.rolling(W, min_periods=W // 2).mean()) / s.rolling(W, min_periods=W // 2).std()

def main():
    D = agg_ob().join(market_ret().rename("mkt"), how="inner").sort_index().dropna(subset=["agg_imb", "mkt"])
    D["z"] = zc(D["agg_imb"]); lr = np.log1p(D["mkt"])
    for lab, H in HS:
        D[f"fwd_{lab}"] = np.expm1(lr[::-1].rolling(H, min_periods=H).sum()[::-1])
    eras = {"OOS": D[D.index < CUT], "RECENT": D[D.index >= CUT]}

    print("=== (1) IC( agg_imb -> forward market return ) by horizon (more NEG = stronger contrarian edge) ===")
    print(f"{'horizon':8s} | {'OOS IC [CI]':27s} | {'RECENT IC [CI]':27s}")
    for lab, H in HS:
        blk = max(42, 4 * H)
        oa, ol, ou = block_ic(eras["OOS"]["agg_imb"], eras["OOS"][f"fwd_{lab}"], blk)
        ra, rl, ru = block_ic(eras["RECENT"]["agg_imb"], eras["RECENT"][f"fwd_{lab}"], blk)
        print(f"{lab:8s} | {oa:+.4f} [{ol:+.4f},{ou:+.4f}] | {ra:+.4f} [{rl:+.4f},{ru:+.4f}]")

    print("\n=== (2) BACKTEST by HOLDING horizon (EWMA HL), net 10bps ===")
    print(f"{'HL horizon':10s} {'turn/day':8s} | {'OOS Sharpe':10s} {'maxDD':7s} | {'RECENT Sharpe':13s} {'maxDD':7s}")
    for lab, H in HS:
        pos = (-D["z"]).clip(-1.5, 1.5).ewm(halflife=H).mean()
        strat = (pos * D["mkt"] - 0.0010 * pos.diff().abs().fillna(0)).dropna()
        sd = daily(strat); tpd = pos.diff().abs().sum() / len(sd)
        so, sr = stats(sd[sd.index < CUT]), stats(sd[sd.index >= CUT])
        print(f"{lab:10s} {tpd:8.2f} | {so['sharpe']:+9.2f} {so['maxdd']:+6.0f}% | {sr['sharpe']:+12.2f} {sr['maxdd']:+6.0f}%")
    print("\nread: does the IC keep strengthening past 3d (1), and which holding horizon maximizes net Sharpe both")
    print("eras with sane turnover (2)? longer = lower turnover but slower + fewer independent bets. HORIZONDONE")

if __name__ == "__main__":
    main()
