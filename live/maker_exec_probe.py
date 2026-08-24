"""Can this book be executed PASSIVELY, and what would it actually cost?

The loop's terminal finding was that execution cost is the binding constraint. Our whole cost model
(`live/state/v3loop/persym_cost_cal.csv`) is TAKER: 5-35 bps/side depth-walking. A resting limit order pays
the fee and earns the spread instead — but converts a KNOWN cost into a STOCHASTIC one (non-fill + adverse
selection). This measures that trade-off on the actual trades our book makes, from owned 5m klines.

Accounting matches the backtest exactly. The backtest prices everything at the 5m close on the decision
stamp (`target_alpha`: my_fwd = close[t+48]/close[t] − 1), so the honest cost of execution is the
IMPLEMENTATION SHORTFALL against that same decision price:

    cost_bps = side * (exec_price / decision_price - 1) * 1e4      (positive = we paid up)

Passive simulation, per trade, per patience window W and offset d:
    BUY  : post at P0*(1-d). Filled if min(low) over the next W minutes <= that price.
    SELL : post at P0*(1+d). Filled if max(high) over the next W minutes >= that price.
    Not filled -> cross at the 5m close at the END of the window and pay that symbol's taker slippage.
Non-fills are kept in the average, so adverse selection is priced: you fail to fill precisely when the
market ran away from you, and then you chase it.

Book = top40/band on the held-out window (2025-01 -> 2026-07), the config iteration 5 pre-committed to.
Run: python3 -u -m live.maker_exec_probe
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from live.cost_loop_harness import (
    ERAS, CACHE, REPO, block_ci, build_panel, get_preds, pit_adv, sharpe, tag_ci,
)
from live.cl_iter4_capacity import build, cost_tiers
from live.mc_oi_universe import topn, N

KL = REPO / "data/ml/test/parquet/klines"
HO0, HO1 = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")
WINDOWS = [3, 6, 12, 24]              # 5m bars of patience = 15m / 30m / 1h / 2h
OFFSETS = [0.0, 1.0, 2.0, 5.0]        # passive offset from the decision price, bps
MAKER_FEE_BPS = 2.0                   # Binance USDM VIP-0 maker; 1.8 with BNB discount, lower at VIP tiers


def load_5m(sym: str, t0, t1) -> pd.DataFrame | None:
    sd = KL / sym / "5m"
    if not sd.exists():
        return None
    paths = [p for p in sorted(sd.glob("*.parquet"))
             if str((t0 - pd.Timedelta(days=2)).date()) <= p.stem <= str(t1.date())]
    if not paths:
        return None
    try:
        d = pd.concat([pd.read_parquet(p, columns=["open_time", "high", "low", "close"]) for p in paths],
                      ignore_index=True)
    except Exception:
        return None
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    return d.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")


def main():
    CT = cost_tiers()
    taker, tmed = CT["cost_10k"]
    PAN = build_panel()
    lab = PAN[["symbol", "open_time", "alpha_vs_btc_realized"]].rename(
        columns={"alpha_vs_btc_realized": "alpha_A"})
    P = pd.concat([get_preds(e) for e in ERAS], ignore_index=True).drop_duplicates(
        ["symbol", "open_time"]).sort_values(["symbol", "open_time"])
    if "alpha_A" not in P.columns:
        P = P.merge(lab, on=["symbol", "open_time"], how="left")
    A = pit_adv(); P["date"] = P["open_time"].dt.floor("1D")
    P = P.merge(A, on=["symbol", "date"], how="left")
    w = topn(P[(P.open_time >= HO0) & (P.open_time < HO1)].dropna(subset=["tadv"]), "tadv", N)

    W, Aa = build(w, "band")
    dW = W.diff(axis=1).iloc[:, 1:]
    trades = dW.stack().rename("dw").reset_index()
    trades = trades[trades["dw"].abs() > 1e-9]
    trades["side"] = np.sign(trades["dw"])
    print(f"held-out book: {W.shape[0]} symbols x {W.shape[1]} bars | "
          f"{len(trades):,} trades ({trades['symbol'].nunique()} syms)", flush=True)
    print(f"taker cost of these trades (cost_10k model): "
          f"{(trades['dw'].abs() * trades['symbol'].map(lambda s: taker.get(s, tmed))).sum() / trades['dw'].abs().sum():.2f} bps/unit traded",
          flush=True)

    # ---- passive fill simulation, per symbol ----
    rows = []
    syms = sorted(trades["symbol"].unique())
    for i, sym in enumerate(syms, 1):
        k = load_5m(sym, HO0, HO1)
        if k is None or k.empty:
            continue
        tk = float(taker.get(sym, tmed))
        g = trades[trades["symbol"] == sym]
        idx = k.index
        pos = idx.searchsorted(g["open_time"].to_numpy())
        ok = (pos < len(idx) - max(WINDOWS))
        pos = pos[ok]; sub = g[ok]
        if len(sub) == 0:
            continue
        P0 = k["close"].to_numpy()[pos]
        lows, highs, closes = k["low"].to_numpy(), k["high"].to_numpy(), k["close"].to_numpy()
        for Wn in WINDOWS:
            fwd_lo = np.array([lows[p + 1:p + 1 + Wn].min() for p in pos])
            fwd_hi = np.array([highs[p + 1:p + 1 + Wn].max() for p in pos])
            endpx = closes[pos + Wn]
            for d in OFFSETS:
                side = sub["side"].to_numpy()
                limit = P0 * (1 - side * d / 1e4)              # buy below, sell above
                filled = np.where(side > 0, fwd_lo <= limit, fwd_hi >= limit)
                # shortfall vs the decision price, in bps, positive = paid up
                sf_fill = side * (limit / P0 - 1) * 1e4 + MAKER_FEE_BPS
                sf_miss = side * (endpx / P0 - 1) * 1e4 + tk    # chase: cross at end of window + slippage
                sf = np.where(filled, sf_fill, sf_miss)
                rows.append(pd.DataFrame({"symbol": sym, "win": Wn, "off": d, "w": sub["dw"].abs().to_numpy(),
                                          "filled": filled.astype(float), "sf": sf,
                                          "sf_fill_only": np.where(filled, sf_fill, np.nan),
                                          "taker": tk}))
        if i % 15 == 0:
            print(f"  [{i}/{len(syms)}] {sym}", flush=True)
    T = pd.concat(rows, ignore_index=True)
    T.to_parquet(CACHE / "maker_exec_trades.parquet", index=False)

    print("\n============ PASSIVE EXECUTION vs TAKER (weighted by traded notional) ============", flush=True)
    print("  cost = implementation shortfall vs the decision price, non-fills chased at end of window\n",
          flush=True)
    print(f"  {'patience':<11}{'offset':<9}{'fill rate':<11}{'cost bps/unit':<15}"
          f"{'cost|filled':<13}{'vs taker':<10}", flush=True)
    base = float((T[(T.win == WINDOWS[0]) & (T.off == OFFSETS[0])]["taker"] *
                  T[(T.win == WINDOWS[0]) & (T.off == OFFSETS[0])]["w"]).sum() /
                 T[(T.win == WINDOWS[0]) & (T.off == OFFSETS[0])]["w"].sum())
    grid = {}
    for Wn in WINDOWS:
        for d in OFFSETS:
            s = T[(T.win == Wn) & (T.off == d)]
            fr = float((s["filled"] * s["w"]).sum() / s["w"].sum())
            cost = float((s["sf"] * s["w"]).sum() / s["w"].sum())
            cf = s.dropna(subset=["sf_fill_only"])
            cfill = float((cf["sf_fill_only"] * cf["w"]).sum() / cf["w"].sum()) if len(cf) else np.nan
            grid[(Wn, d)] = cost
            print(f"  {Wn*5:<3}min{'':<5}{d:<9.1f}{fr*100:<11.1f}{cost:<15.2f}{cfill:<13.2f}"
                  f"{cost - base:<+10.2f}", flush=True)
    print(f"\n  taker baseline (cost_10k, immediate): {base:.2f} bps/unit traded", flush=True)

    # ---- re-price the book's net Sharpe under the measured passive costs ----
    print("\n============ NET SHARPE re-priced with measured passive cost ============", flush=True)
    g = (W * Aa).sum(axis=0)
    turn = (0.25 * W.diff(axis=1).abs().sum(axis=0))
    tvec = pd.Series([taker.get(s, tmed) for s in W.index], index=W.index)
    ch_taker = 0.25 * W.diff(axis=1).abs().mul(tvec, axis=0).sum(axis=0)
    net_taker = (g - ch_taker / 1e4).iloc[1:]
    lo, hi = block_ci(net_taker.to_numpy())
    print(f"  gross                        {sharpe(g.iloc[1:]):+.2f}", flush=True)
    print(f"  net, taker (cost_10k)        {sharpe(net_taker):+.2f} [{lo:+.2f},{hi:+.2f}] "
          f"{tag_ci(lo, hi)}   ({ch_taker.iloc[1:].mean():.2f} bps/bar)", flush=True)
    best = min(grid, key=grid.get)
    for key in sorted(grid, key=grid.get)[:4]:
        c = grid[key]
        net = (g - turn * c / 1e4).iloc[1:]
        lo, hi = block_ci(net.to_numpy())
        star = "  <- best" if key == best else ""
        print(f"  net, passive {key[0]*5:>3}min off{key[1]:.0f}  {sharpe(net):+.2f} [{lo:+.2f},{hi:+.2f}] "
              f"{tag_ci(lo, hi)}   ({(turn*c).iloc[1:].mean():.2f} bps/bar){star}", flush=True)
    print("\nMAKEREXECDONE", flush=True)


if __name__ == "__main__":
    main()
