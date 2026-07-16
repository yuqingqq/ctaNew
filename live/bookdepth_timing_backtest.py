"""Hardened standalone backtest of the agg_imb contrarian market-timing signal (aggregate book bid-lean -> short the
market). CLEAN real-time signal: pos = -clip( z90(agg_imb), +/-1.5 ), trailing-90-bar z (balanced, no look-ahead, no
static short-bias), rebalanced each 4h bar, earning the equal-weight market's forward 4h return, net of 10bps turnover
cost. Reports full/OOS/RECENT stats vs BUY-HOLD, a cost x window robustness sweep, an up/down-market regime breakdown
(is it defensive?), and saves an equity-curve + drawdown chart. PIT-clean: pos_T from agg_imb over [T-4h,T) earns the
market return over [T,T+4h).
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from live.bookdepth_market_timing import agg_ob
PANEL = "/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet"
CUT = pd.Timestamp("2025-10-01", tz="UTC")
OUT = "/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad/timing_backtest.png"

def market_ret():
    p = pd.read_parquet(PANEL, columns=["symbol", "open_time", "return_pct"])
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p = p[(p.open_time.dt.hour % 4 == 0) & (p.open_time.dt.minute == 0)]
    return p.groupby("open_time")["return_pct"].mean().sort_index().rename("mkt")

def position(aggimb, W, cap, smooth_hl=1):
    z = (aggimb - aggimb.rolling(W, min_periods=W // 2).mean()) / aggimb.rolling(W, min_periods=W // 2).std()
    pos = (-z).clip(-cap, cap)
    return pos.ewm(halflife=smooth_hl).mean() if smooth_hl > 1 else pos   # smooth to the signal's 1-2d horizon

def daily(bar_ret):
    return bar_ret.groupby(bar_ret.index.floor("1D")).apply(lambda x: (1 + x).prod() - 1)

def stats(dr):
    if len(dr) < 20: return None
    eq = (1 + dr).cumprod(); dd = (eq / eq.cummax() - 1).min()
    sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else np.nan
    ann = (1 + dr).prod() ** (365 / len(dr)) - 1
    return dict(sharpe=sh, ann=ann * 100, vol=dr.std() * np.sqrt(365) * 100, maxdd=dd * 100,
                calmar=(ann / abs(dd)) if dd < 0 else np.nan, hit=(dr > 0).mean() * 100, n=len(dr))

def line(tag, s):
    if s is None: print(f"  {tag:20s}   (insufficient)"); return
    print(f"  {tag:20s} Sharpe {s['sharpe']:+.2f} | annRet {s['ann']:+6.1f}% | vol {s['vol']:4.0f}% | maxDD {s['maxdd']:+6.1f}% | Calmar {s['calmar']:+.2f} | hit {s['hit']:.0f}% | n{s['n']}")

def main():
    D = agg_ob().join(market_ret(), how="inner").sort_index().dropna(subset=["agg_imb", "mkt"])
    W, CAP, COST, HL = 90, 1.5, 0.0010, 12        # HL=12 bars (2d) EWMA = fair execution of the 1-2d signal (a-priori)
    D["pos"] = position(D["agg_imb"], W, CAP, HL)
    D["turn"] = D["pos"].diff().abs()
    D["strat"] = D["pos"] * D["mkt"] - COST * D["turn"].fillna(0)
    D = D.dropna(subset=["pos", "strat"])
    sd, md = daily(D["strat"]), daily(D["mkt"])
    eras = [("FULL", D.index.min(), D.index.max()), ("OOS", D.index.min(), CUT), ("RECENT", CUT, D.index.max())]
    print(f"=== BACKTEST  agg_imb contrarian timing  (W={W} z, cap={CAP}, EWMA-smooth HL={HL}bars=2d, cost={COST*1e4:.0f}bps) ===")
    print(f"bars {len(D)} | days {len(sd)} | avg pos {D['pos'].mean():+.2f} | avg |pos| {D['pos'].abs().mean():.2f} | %long {(D['pos']>0).mean()*100:.0f}% | turnover/day {D['turn'].sum()/len(sd):.2f}\n")
    for lab, a, b in eras:
        m = (sd.index >= a) & (sd.index < b) if lab != "FULL" else slice(None)
        print(f"[{lab}]"); line("STRATEGY (timing)", stats(sd[m])); line("buy&hold market", stats(md[m]))
    print("\n=== ROBUSTNESS: net Sharpe (OOS / RECENT) over EWMA-smoothing x cost, with turnover/day ===")
    print(f"{'smoothHL':9s} {'turn/day':8s} | " + " | ".join(f"{int(c*1e4):2d}bps" for c in [0, .0010, .0020]))
    for hl in [1, 6, 12, 24]:
        pos = position(D["agg_imb"], W, CAP, hl); turn = pos.diff().abs().fillna(0); tpd = turn.sum() / len(sd); row = []
        for c in [0, .0010, .0020]:
            r = pos * D["mkt"] - c * turn; dd = daily(r)
            so = stats(dd[dd.index < CUT]); sr = stats(dd[dd.index >= CUT])
            row.append(f"{(so['sharpe'] if so else np.nan):+.2f}/{(sr['sharpe'] if sr else np.nan):+.2f}")
        tag = f"{hl}(={hl*4}h)" if hl > 1 else "1(none)"
        print(f"{tag:9s} {tpd:8.2f} | " + " | ".join(f"{x:11s}" for x in row))
    print("\n=== REGIME: strategy daily return on UP vs DOWN market days (defensive if DOWN-day mean > 0) ===")
    for lab, a, b in eras:
        m = (sd.index >= a) & (sd.index < b) if lab != "FULL" else slice(None)
        s, k = sd[m], md[m]; up, dn = k > 0, k <= 0
        print(f"  [{lab:6s}] UP-days: strat {s[up].mean()*1e4:+6.1f}bps (mkt {k[up].mean()*1e4:+6.1f}) | DOWN-days: strat {s[dn].mean()*1e4:+6.1f}bps (mkt {k[dn].mean()*1e4:+6.1f})")
    # equity + drawdown chart
    seq, beq = (1 + sd).cumprod(), (1 + md).cumprod()
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[2.3, 1], sharex=True)
    ax[0].plot(seq.index, seq.values, lw=1.6, color="#1a7f5a", label=f"agg_imb timing (Sharpe {stats(sd)['sharpe']:+.2f})")
    ax[0].plot(beq.index, beq.values, lw=1.3, color="#999", label=f"buy & hold (Sharpe {stats(md)['sharpe']:+.2f})")
    ax[0].axvline(CUT, color="#c0392b", ls="--", lw=1, alpha=.7); ax[0].set_yscale("log")
    ax[0].text(CUT, ax[0].get_ylim()[1], " RECENT →", color="#c0392b", va="top", fontsize=9)
    ax[0].set_ylabel("equity (log, ×)"); ax[0].legend(loc="upper left", fontsize=9)
    ax[0].set_title("Aggregate-book contrarian market-timing — equity curve (10bps cost)", fontsize=11)
    sdd = seq / seq.cummax() - 1
    ax[1].fill_between(sdd.index, sdd.values * 100, 0, color="#1a7f5a", alpha=.35); ax[1].axvline(CUT, color="#c0392b", ls="--", lw=1, alpha=.7)
    ax[1].set_ylabel("drawdown %"); ax[1].grid(alpha=.2)
    plt.tight_layout(); plt.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"\nchart -> {OUT}\nBTDONE")

if __name__ == "__main__":
    main()
