"""Trade-level test of price-validated order-book reaction across a fixed alt universe.

Primary hypothesis (continuation through displayed liquidity):
  price_z > 0 and ob_z < 0 -> asks are being absorbed -> LONG
  price_z < 0 and ob_z > 0 -> bids are being absorbed -> SHORT

The test is deliberately model-free and compares the reaction trade with price-only
momentum/reversal and book-only controls. Signals are known at decision time T;
``return_pct`` and ``alpha_vs_btc_realized`` are outcomes over [T,T+4h).

This is research, not an execution implementation. The fixed universe is a diagnostic
that removes expanding-universe drift but remains survivor-selected; a strategy must
eventually use point-in-time eligibility including delisted contracts.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from live.bookdepth_reaction_xs import build
from live.bookdepth_timing_corrected import fixed_universe


CUT = pd.Timestamp("2025-10-01", tz="UTC")
COST_ONE_WAY = 0.0006  # 5bp taker + 1bp slippage per unit of notional turnover
RNG = np.random.default_rng(113)


def _weights(g: pd.DataFrame, variant: str, k: int, threshold: float) -> dict[str, float]:
    """Return a dollar-neutral, gross-1 one-bar portfolio for one decision time."""
    q = g[["symbol", "price_z", "ob_z"]].dropna().copy()
    if variant.startswith("reaction_"):
        q = q[
            (q["price_z"] * q["ob_z"] < 0)
            & (q["price_z"].abs() >= threshold)
            & (q["ob_z"].abs() >= threshold)
        ]
        q["score"] = q["price_z"] - q["ob_z"]
        if variant == "reaction_reverse":
            q["score"] = -q["score"]
        longs = q[q["score"] > 0].nlargest(k, "score")
        shorts = q[q["score"] < 0].nsmallest(k, "score")
    elif variant == "price_momentum":
        q["score"] = q["price_z"]
        longs, shorts = q.nlargest(k, "score"), q.nsmallest(k, "score")
    elif variant == "price_reversal":
        q["score"] = -q["price_z"]
        longs, shorts = q.nlargest(k, "score"), q.nsmallest(k, "score")
    elif variant == "book_contrarian":
        q["score"] = -q["ob_z"]
        longs, shorts = q.nlargest(k, "score"), q.nsmallest(k, "score")
    else:
        raise ValueError(variant)

    if len(longs) < k or len(shorts) < k:
        return {}
    w = {s: 0.5 / k for s in longs["symbol"]}
    for s in shorts["symbol"]:
        w[s] = w.get(s, 0.0) - 0.5 / k
    return w


def _turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    names = set(previous) | set(current)
    return float(sum(abs(current.get(s, 0.0) - previous.get(s, 0.0)) for s in names))


def run_variant(
    panel: pd.DataFrame,
    variant: str,
    k: int = 5,
    threshold: float = 0.0,
) -> pd.DataFrame:
    rows = []
    previous: dict[str, float] = {}
    for t, g in panel.groupby("open_time", sort=True):
        weights = _weights(g, variant, k, threshold)
        turn = _turnover(previous, weights)
        outcomes = g.set_index("symbol")
        raw = sum(w * outcomes.at[s, "return_pct"] for s, w in weights.items()) if weights else 0.0
        alpha = (
            sum(w * outcomes.at[s, "alpha_vs_btc_realized"] for s, w in weights.items())
            if weights else 0.0
        )
        cost = COST_ONE_WAY * turn
        rows.append(
            {
                "open_time": t,
                "raw_gross": raw,
                "alpha_gross": alpha,
                "cost": cost,
                "raw_net": raw - cost,
                "alpha_net": alpha - cost,
                "turnover": turn,
                "active": bool(weights),
                "n_names": len(weights),
            }
        )
        previous = weights
    return pd.DataFrame(rows).set_index("open_time")


def daily(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.floor("1D")).apply(lambda x: (1 + x).prod() - 1)


def stats(x: pd.Series) -> dict[str, float]:
    x = x.dropna()
    if len(x) < 20 or x.std(ddof=1) == 0:
        return {"sharpe": np.nan, "mean_bps": np.nan, "maxdd": np.nan, "n": len(x)}
    eq = (1 + x).cumprod()
    return {
        "sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(365)),
        "mean_bps": float(x.mean() * 1e4),
        "maxdd": float((eq / eq.cummax() - 1).min() * 100),
        "n": len(x),
    }


def block_mean_ci(x: pd.Series, block: int = 7, n: int = 2500) -> tuple[float, float]:
    v = x.dropna().to_numpy()
    if len(v) < 30:
        return (np.nan, np.nan)
    nb = int(np.ceil(len(v) / block))
    boot = []
    for _ in range(n):
        starts = RNG.integers(0, len(v) - block + 1, nb)
        take = np.concatenate([v[s : s + block] for s in starts])[: len(v)]
        boot.append(take.mean() * 1e4)
    return tuple(np.percentile(boot, [2.5, 97.5]))


def paired_delta_ci(a: pd.Series, b: pd.Series) -> tuple[float, float, float]:
    """Mean-bps delta and block CI for strategy a minus counterfactual b."""
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    delta = d["a"] - d["b"]
    lo, up = block_mean_ci(delta)
    return float(delta.mean() * 1e4), lo, up


def main():
    syms = fixed_universe()
    panel = build(syms)
    labels = pd.read_parquet(
        "/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
        columns=["symbol", "open_time", "alpha_vs_btc_realized"],
    )
    labels["open_time"] = pd.to_datetime(labels["open_time"], utc=True)
    panel = panel.merge(labels, on=["symbol", "open_time"], how="left")
    panel = panel.dropna(
        subset=["price_z", "ob_z", "return_pct", "alpha_vs_btc_realized"]
    ).sort_values(["open_time", "symbol"])

    configs = [
        ("price_momentum", 0.0),
        ("price_reversal", 0.0),
        ("book_contrarian", 0.0),
        ("reaction_continue", 0.0),
        ("reaction_continue", 0.5),
        ("reaction_reverse", 0.0),
        ("reaction_reverse", 0.5),
    ]
    results: dict[tuple[str, float, int], pd.DataFrame] = {}
    print(
        f"fixed diagnostic universe={len(syms)} | rows={len(panel)} | "
        f"bars={panel.open_time.nunique()} | span={panel.open_time.min()}..{panel.open_time.max()}\n"
    )
    for k in [3, 5]:
        print(f"================ K={k} per side, next-4h hold, 6bps/one-way ================")
        print(
            f"{'variant':25s} {'era':6s} | {'raw net Sh':10s} {'mean bps [7d CI]':25s} "
            f"{'alpha Sh':8s} {'turn/bar':8s} {'active':7s} {'maxDD':7s}"
        )
        for variant, threshold in configs:
            tag = f"{variant}[thr={threshold:g}]"
            r = run_variant(panel, variant, k, threshold)
            results[(variant, threshold, k)] = r
            for era, mask in [("OOS", r.index < CUT), ("REC", r.index >= CUT)]:
                sub = r[mask]
                rd = daily(sub["raw_net"])
                ad = daily(sub["alpha_net"])
                rs, aps = stats(rd), stats(ad)
                lo, up = block_mean_ci(rd)
                print(
                    f"{tag:25s} {era:6s} | {rs['sharpe']:+10.2f} "
                    f"{rs['mean_bps']:+7.2f} [{lo:+6.2f},{up:+6.2f}] "
                    f"{aps['sharpe']:+8.2f} {sub['turnover'].mean():8.2f} "
                    f"{sub['active'].mean()*100:6.1f}% {rs['maxdd']:+6.1f}%"
                )
        print()

        # Primary incremental comparison: proposed continuation reaction vs
        # price-only continuation, and vs the empirically natural price reversal.
        for threshold in [0.0, 0.5]:
            react = results[("reaction_continue", threshold, k)]
            mom = results[("price_momentum", 0.0, k)]
            rev = results[("price_reversal", 0.0, k)]
            print(f"paired daily net-return deltas for reaction_continue[thr={threshold:g}], K={k}")
            for era, mask in [("OOS", react.index < CUT), ("REC", react.index >= CUT)]:
                rr, mm, vv = daily(react.loc[mask, "raw_net"]), daily(mom.loc[mask, "raw_net"]), daily(rev.loc[mask, "raw_net"])
                dm = paired_delta_ci(rr, mm)
                dv = paired_delta_ci(rr, vv)
                print(
                    f"  {era}: vs price momentum {dm[0]:+.2f}bps [{dm[1]:+.2f},{dm[2]:+.2f}] | "
                    f"vs price reversal {dv[0]:+.2f}bps [{dv[1]:+.2f},{dv[2]:+.2f}]"
                )
        print()
    print("REACTIONTRADEDONE")


if __name__ == "__main__":
    main()
