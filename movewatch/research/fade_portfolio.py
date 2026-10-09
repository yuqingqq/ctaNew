"""Run: python3 research/fade_portfolio.py

Descriptive (all windows already read for pump questions; decides nothing): is signal 2's small
per-trade edge significant IN AGGREGATE, given how often it fires? A book that shorts every heavy
pump (vol_persist > A2 heavy cut) at +4 h and covers at +24 h, stress 1 with funding.

Book, fixed before the run:
- each trade is a fixed notional of 1% of capital, no compounding, no cap on open trades;
- P&L is booked on the exit day; days inside a replayed window with no exit book 0;
- days between windows are not counted;
- per window and calendar year: trades, trades per month, total P&L (% of capital), annualised,
  daily Sharpe (annualised, sqrt 365), max drawdown, worst day, and the most trades open at
  once (= gross exposure in % of capital).
Variants: all heavy pumps; non-meme heavy pumps (CoinGecko meme list, retroactive).
`--stress 3` triples the slippage tiers (descriptive cost check).
`--trades` prints per-year trade statistics instead: win rate, median win and loss, the share of trades losing
> 25% / > 50%, p5, worst, and how much of the gross wins the losers took back."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/yuqing/ctaNew/movewatch")
import research.heavy_fade as hf
import research.premium_angles as pa
import research.premium_classes as pc

SIZE = 0.01
STRESS = 3 if "--stress" in sys.argv and sys.argv[sys.argv.index("--stress") + 1] == "3" else 1


def covered_days(p: pd.DataFrame) -> pd.DatetimeIndex:
    days = []
    for _, t in p.groupby("window")["time"]:
        days.append(pd.date_range(t.min().floor("D"), t.max().floor("D"), freq="D"))
    return days[0].append(days[1:]).unique().sort_values()


def book(x: pd.DataFrame, days: pd.DatetimeIndex) -> tuple[pd.Series, pd.DataFrame]:
    x = x.assign(net=hf.net(x, "S24", STRESS)).dropna(subset=["net"])
    x = x.assign(exit_day=(x["time"] + pd.Timedelta(minutes=1441)).dt.floor("D"))
    daily = (SIZE * x.groupby("exit_day")["net"].sum()).reindex(days, fill_value=0.0)
    return daily, x


def summary(daily: pd.Series, x: pd.DataFrame) -> dict:
    if len(daily) < 30 or len(x) == 0:
        return {}
    eq = daily.cumsum()
    t0 = x["time"].sort_values()
    open_at = np.searchsorted(t0.to_numpy(), (t0 + pd.Timedelta(minutes=1200)).to_numpy())  # trades opened before this one closes
    conc = int((open_at - np.arange(len(t0))).max())
    return {"trades": len(x), "per_month": len(x) / (len(daily) / 30.4), "total": 100 * daily.sum(),
            "annual": 100 * daily.mean() * 365, "sharpe": daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else np.nan,
            "maxdd": 100 * (eq - eq.cummax()).min(), "worst_day": 100 * daily.min(), "max_open": conc,
            "mean_trade": 100 * x["net"].mean()}


def show(lab: str, s: dict) -> str:
    if not s:
        return f"{lab:14} (too short)"
    return (f"{lab:14} trades {s['trades']:5d} ({s['per_month']:5.1f}/mo) | per trade {s['mean_trade']:+5.2f}% | total {s['total']:+7.1f}% "
            f"| annual {s['annual']:+6.1f}% | Sharpe {s['sharpe']:+5.2f} | max DD {s['maxdd']:+6.1f}% | worst day {s['worst_day']:+5.1f}% | max open {s['max_open']}")


def trade_table(x: pd.DataFrame, label: str) -> None:
    print(f"\n== {label}: per-trade statistics by year (short +4h -> +24h, stress {STRESS} + funding)")
    print(f"{'year':6} {'n':>5} {'win':>5} {'med win':>8} {'med loss':>9} {'lose>25%':>9} {'lose>50%':>9} {'p5':>7} {'worst':>7} {'mean':>7} {'losses/wins':>11}")
    for y, g in list(x.groupby(x["time"].dt.year)) + [("all", x)]:
        v = g["net"].to_numpy()
        w, l = v[v > 0], v[v <= 0]
        print(f"{str(y):6} {len(v):5d} {100 * (v > 0).mean():4.0f}% {100 * np.median(w):+7.1f}% {100 * np.median(l):+8.1f}% "
              f"{100 * (v < -0.25).mean():8.1f}% {100 * (v < -0.5).mean():8.1f}% {100 * np.percentile(v, 5):+6.1f}% {100 * v.min():+6.0f}% "
              f"{100 * v.mean():+6.2f}% {-l.sum() / w.sum():10.2f}")


if __name__ == "__main__":
    p = pc.frame(require="vol_persist")
    if "--trades" in sys.argv:
        heavy = p[p["vol_persist"] > hf.HEAVY]
        for vname, v in (("ALL HEAVY PUMPS", heavy), ("NON-MEME HEAVY PUMPS", heavy[~heavy["meme"]])):
            trade_table(book(v, covered_days(p))[1], vname)
        sys.exit(0)
    days = covered_days(p)
    heavy = p[p["vol_persist"] > hf.HEAVY]
    for vname, v in (("ALL HEAVY PUMPS", heavy), ("NON-MEME HEAVY PUMPS", heavy[~heavy["meme"]])):
        daily, x = book(v, days)
        print(f"\n== {vname}: short +4h -> +24h, 1% of capital per trade, stress {STRESS} + funding; {len(days)} covered days")
        print(show("2020-01..2026-09", summary(daily, x)))
        for w, lab in (("C", "C 2020-25"), ("V", "V 25-10..26-03"), ("S", "S 26-04..26-09")):
            xi = x[x["win"] == w]
            di = daily[(daily.index >= xi["time"].min().floor("D")) & (daily.index <= xi["time"].max().floor("D"))]
            print(show(lab, summary(di, xi)))
        for y in sorted(x["time"].dt.year.unique()):
            xi = x[x["time"].dt.year == y]
            print(show(f"  {y}", summary(daily[daily.index.year == y], xi)))
        roll = daily.rolling(90).sum().dropna()
        print(f"rolling 90-day P&L: negative in {100 * (roll < 0).mean():.0f}% of windows; worst {100 * roll.min():+.1f}%, best {100 * roll.max():+.1f}%")
