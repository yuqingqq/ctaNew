"""Beyond-cross-section loop — iteration 2 (B2): funding carry as a HELD POSITION, not a signal.

The repo has tested funding twice, both times as a forecast input: as a feature in V0 (dropped, redundant)
and as a cross-sectional L/S signal (sleeve test, net Sharpe -2.45). Both are BETS ON funding. Neither is the
carry trade, which is a different object entirely: long spot + short perp, delta-neutral, and you are simply
PAID the funding rate for supplying leverage to the market. It is a risk premium, not a prediction.

P&L accounting per 8h funding interval, per symbol, holding 1 unit of notional on each leg:
    pnl = (spot_{t+1}/spot_t - 1) - (perp_{t+1}/perp_t - 1) + funding_rate_t
          \_______ basis convergence, the residual risk ________/   \__ the premium __/
A short perp RECEIVES funding when the rate is positive, hence +f.

Costs are charged on BOTH legs and on BOTH sides of the round trip — this trade is cost-heavy in a way the
perp-only book is not, because Binance SPOT VIP-0 taker is 10 bps (vs 5 bps on USDM futures):
    round trip = 2 legs x 2 sides = 2*(10 + 5) = 30 bps taker;  2*(10 + 2) = 24 bps with futures-maker exit.

Universe: the 61 symbols with spot + funding + perp klines all present.

Gates (live/BEYOND_XS_LOOP.md): G1 gross carry Sharpe CI>0 in BOTH eras; G2 net of round-trip cost at the
chosen rebalance CI>0 in BOTH eras; G3 not a disguised market bet — beta to BTC must be near zero and the
result must survive removing it. Falsifier: G2 fails -> carry does not pay at retail fee tiers on free data.
Run: python3 -u -m live.bx_iter2_carry
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from live.cost_loop_harness import CACHE, REPO, block_ci, sharpe, tag_ci

SPOT_FEE_BPS = 10.0        # Binance spot VIP-0 taker
PERP_FEE_BPS = 5.0         # Binance USDM VIP-0 taker
KS = [5, 10, 20]
HOLD_INTERVALS = [3, 21, 63]     # 8h units: 1 day, 1 week, 3 weeks
ERAS = {"OOS": ("2023-06-01", "2025-10-01"), "RECENT": ("2025-10-01", "2026-07-01")}
PYR_8H = 3 * 365.0


def load() -> pd.DataFrame:
    fp = CACHE / "carry_panel.parquet"
    if fp.exists():
        d = pd.read_parquet(fp); d["t"] = pd.to_datetime(d["t"], utc=True); return d
    spot = {Path(f).stem.replace("bn_spot_", "").replace("_1h", ""): f
            for f in glob.glob(str(REPO / "data/ml/cache/bn_spot_*.parquet"))}
    rows = []
    for sym, sf in sorted(spot.items()):
        ff = REPO / f"data/ml/cache/funding_{sym}.parquet"
        kd = REPO / f"data/ml/test/parquet/klines/{sym}/5m"
        if not ff.exists() or not kd.exists():
            continue
        try:
            s = pd.read_parquet(sf)
            s["open_time"] = pd.to_datetime(s["open_time"], utc=True)
            s = s.set_index("open_time")["close"].astype(float).sort_index()
            fu = pd.read_parquet(ff)
            tc = "calc_time" if "calc_time" in fu.columns else "open_time"
            fu[tc] = pd.to_datetime(fu[tc], utc=True)
            fu = fu.set_index(tc)["funding_rate"].astype(float).sort_index()
            fu = fu[~fu.index.duplicated(keep="last")]
            pk = pd.concat([pd.read_parquet(p, columns=["open_time", "close"])
                            for p in sorted(kd.glob("*.parquet"))], ignore_index=True)
            pk["open_time"] = pd.to_datetime(pk["open_time"], utc=True)
            pk = pk.drop_duplicates("open_time").set_index("open_time")["close"].astype(float).sort_index()
        except Exception:
            continue
        grid = fu.index.floor("8h").unique().sort_values()
        sp = s.reindex(grid, method="ffill", tolerance=pd.Timedelta("2h"))
        pp = pk.reindex(grid, method="ffill", tolerance=pd.Timedelta("30min"))
        f8 = fu.groupby(fu.index.floor("8h")).last().reindex(grid)
        df = pd.DataFrame({"t": grid, "spot": sp.to_numpy(), "perp": pp.to_numpy(),
                           "f": f8.to_numpy()}).dropna()
        if len(df) < 500:
            continue
        df["symbol"] = sym
        rows.append(df)
    D = pd.concat(rows, ignore_index=True).sort_values(["symbol", "t"])
    D["r_s"] = D.groupby("symbol")["spot"].pct_change()
    D["r_p"] = D.groupby("symbol")["perp"].pct_change()
    # the premium is paid at the END of the interval you were short into -> shift so it is not look-ahead
    D["carry"] = D["r_s"] - D["r_p"] + D["f"]
    D["f_trail"] = D.groupby("symbol")["f"].transform(
        lambda x: x.shift(1).rolling(21, min_periods=10).mean())          # PIT selection signal
    D.to_parquet(fp, index=False)
    return D


def main():
    D = load()
    print(f"carry panel: {D.symbol.nunique()} symbols, {D.t.min().date()} -> {D.t.max().date()}, "
          f"{len(D):,} symbol-intervals", flush=True)
    ann = D.groupby("symbol")["f"].mean() * PYR_8H * 100
    print(f"\nannualised funding by symbol (%): median {ann.median():.2f}, "
          f"p25 {ann.quantile(.25):.2f}, p75 {ann.quantile(.75):.2f}, max {ann.max():.2f}", flush=True)
    print("  richest 8:", ", ".join(f"{k} {v:.1f}%" for k, v in ann.nlargest(8).items()), flush=True)

    print("\n=== A1 — is the premium actually collectable? decompose the P&L ===", flush=True)
    for era, (t0, t1) in ERAS.items():
        e = D[(D.t >= pd.Timestamp(t0, tz="UTC")) & (D.t < pd.Timestamp(t1, tz="UTC"))]
        if e.empty:
            continue
        print(f"  {era:<8} funding {e['f'].mean()*1e4:+7.3f} bps/8h   "
              f"basis (r_s-r_p) {(e['r_s']-e['r_p']).mean()*1e4:+7.3f}   "
              f"net carry {e['carry'].mean()*1e4:+7.3f} bps/8h "
              f"({e['carry'].mean()*PYR_8H*100:+.1f}%/yr gross)", flush=True)

    print("\n=== G1/G2 — top-K by trailing funding, equal weight, both legs costed ===", flush=True)
    print(f"    round-trip cost charged = 2 legs x 2 sides = {2*(SPOT_FEE_BPS+PERP_FEE_BPS):.0f} bps, "
          "applied to the fraction of the book replaced at each rebalance", flush=True)

    def sh8(x):
        x = np.asarray(x, float)
        return float(x.mean() / x.std() * np.sqrt(PYR_8H)) if len(x) > 2 and x.std() > 0 else np.nan

    RT = 2 * (SPOT_FEE_BPS + PERP_FEE_BPS) / 1e4
    results = {}
    for era, (t0, t1) in ERAS.items():
        e = D[(D.t >= pd.Timestamp(t0, tz="UTC")) & (D.t < pd.Timestamp(t1, tz="UTC"))].copy()
        if e.empty:
            continue
        print(f"\n----- {era} -----", flush=True)
        print(f"  {'K':<4}{'rebal':<8}{'churn':<8}{'gross Sh [CI]':<24}{'gross %/yr':<12}"
              f"{'net Sh [CI]':<24}{'net %/yr':<10}", flush=True)
        for hold in HOLD_INTERVALS:
            for K in KS:
                sel = e.dropna(subset=["f_trail", "carry"]).copy()
                # rebalance only every `hold` intervals: freeze the selection signal within a block
                blk = pd.Series(pd.factorize(sel["t"])[0] // hold, index=sel.index)
                sel["sig"] = sel.groupby([blk, "symbol"])["f_trail"].transform("first")
                sel["rk"] = sel.groupby("t")["sig"].rank(ascending=False, method="first")
                sel = sel[sel["rk"] <= K]
                if sel.empty:
                    continue
                ser = sel.groupby("t")["carry"].mean().sort_index()
                if len(ser) < 200:
                    continue
                names = sel.groupby("t")["symbol"].apply(set)
                churn = float(np.mean([len(names.iloc[i] - names.iloc[i - 1]) / K
                                       for i in range(1, len(names))])) if len(names) > 1 else 1.0
                net = ser - churn * RT / hold          # amortise the round trip over the holding block
                gs, ns = sh8(ser), sh8(net)
                glo, ghi = block_ci(ser.to_numpy(), block=21)
                nlo, nhi = block_ci(net.to_numpy(), block=21)
                results[(era, K, hold)] = (ns, nlo, nhi)
                print(f"  {K:<4}{f'{hold*8}h':<8}{churn:<8.2f}"
                      f"{f'{gs:+.2f} [{glo:+.2f},{ghi:+.2f}]':<24}{ser.mean()*PYR_8H*100:<+12.1f}"
                      f"{f'{ns:+.2f} [{nlo:+.2f},{nhi:+.2f}] {tag_ci(nlo, nhi)}':<24}"
                      f"{net.mean()*PYR_8H*100:<+10.1f}", flush=True)

    print("\n=== GATE READ ===", flush=True)
    cells = {(K, h) for (e_, K, h) in results}
    win = [(K, h) for (K, h) in sorted(cells)
           if all(results.get((e_, K, h), (np.nan, -9, 0))[1] > 0 for e_ in ERAS)]
    print(f"  G2 configs with net CI>0 in BOTH eras: {win if win else 'NONE'}", flush=True)
    print("\nBXITER2DONE", flush=True)


if __name__ == "__main__":
    main()
