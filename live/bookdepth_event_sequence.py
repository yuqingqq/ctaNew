"""Sequence diagnostic for a price fall followed by a bid-heavy book.

The 4h L2 cache cannot tell whether bids were already present and traded
through, or appeared only after the decline.  This probe goes back to the raw
30-second Binance Vision percentage-band snapshots and aligns them with exact
aggressor-signed aggTrades.

For each point-in-time 15-minute downside shock it observes:

* book depth before the shock;
* taker-sell flow and bid-depth drawdown during the shock;
* bid replenishment for 15 minutes after the shock; and
* returns starting only after that 15-minute classification window.

The labels are deliberately mechanical and fixed before looking at returns:

``late-added``
    Bids were not elevated before the drop, then rose after most of the sell
    impulse had occurred.
``consumed/exhausted``
    Elevated pre-existing bids met net taker selling, visibly drew down, and
    recovered less than one third of the drawdown.
``absorbed/stabilized``
    Consumption evidence, visible depth recovery above two thirds, a bid-heavy
    ending book, and no meaningful new low during the classification window.
``replenished/leaking``
    Visible bids refill, but price still leaks lower; replenishment alone is
    therefore not treated as absorption.
Important limitation: Binance Vision bookDepth is cumulative notional inside
moving percentage bands around the current mid.  It is not market-by-price,
so this probe cannot prove that a particular resting order was executed.  It
is a sequencing/flow proxy, not a queue-reconstruction test.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from live.bookdepth_loader import _fetch_day


REPO = Path("/home/yuqing/ctaNew")
AGG_ROOT = REPO / "data/ml/test/parquet/aggTrades"
DEFAULT_SYMBOLS = [
    "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT",
]
DEFAULT_DAYS = [
    # Distributed OOS observations rather than one hand-picked regime.
    "2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15",
    "2025-01-15", "2025-04-15", "2025-07-15", "2025-09-15",
    # Recent era used by the surrounding L2 research.
    "2025-10-15", "2025-11-15", "2025-12-15", "2026-01-15",
    "2026-02-15", "2026-03-15", "2026-04-15", "2026-05-15",
]
RECENT_CUT = pd.Timestamp("2025-10-01", tz="UTC")


def _weighted_time(index: pd.DatetimeIndex, weights: pd.Series) -> pd.Timestamp | pd.NaT:
    """Weighted-median timestamp for non-negative event intensity."""
    w = np.asarray(weights.fillna(0.0), dtype=float)
    keep = np.isfinite(w) & (w > 0)
    if not keep.any():
        return pd.NaT
    ix = index[keep]
    ww = w[keep]
    return ix[np.searchsorted(np.cumsum(ww), ww.sum() / 2.0)]


def _book_grid(sym: str, day: pd.Timestamp) -> pd.DataFrame | None:
    book = _fetch_day(sym, day)
    if book is None or book.empty:
        return None
    b = book.copy()
    liq1 = np.exp(b["liq1"])
    b["bid1"] = liq1 * (1.0 + b["imb1"].clip(-0.999, 0.999)) / 2.0
    liq02 = liq1 * b["touch"]
    b["bid02"] = liq02 * (1.0 + b["imb02"].clip(-0.999, 0.999)) / 2.0
    b.index = b.index.floor("30s")
    return b[["bid1", "bid02", "imb1", "imb02"]].groupby(level=0).last()


def _trade_grid(sym: str, day: pd.Timestamp) -> pd.DataFrame | None:
    path = AGG_ROOT / sym / f"{day:%Y-%m-%d}.parquet"
    if not path.exists():
        return None
    t = pd.read_parquet(
        path,
        columns=["price", "quantity", "transact_time", "is_buyer_maker"],
    )
    if t.empty:
        return None
    t["transact_time"] = pd.to_datetime(t["transact_time"], utc=True)
    t["quote"] = t["price"].astype(float) * t["quantity"].astype(float)
    t["sell_quote"] = np.where(t["is_buyer_maker"], t["quote"], 0.0)
    t["buy_quote"] = np.where(t["is_buyer_maker"], 0.0, t["quote"])
    t["bucket"] = t["transact_time"].dt.floor("30s")
    return t.groupby("bucket").agg(
        price=("price", "last"),
        sell_quote=("sell_quote", "sum"),
        buy_quote=("buy_quote", "sum"),
    )


def _aligned_day(sym: str, day: pd.Timestamp) -> pd.DataFrame | None:
    b = _book_grid(sym, day)
    t = _trade_grid(sym, day)
    if b is None or t is None:
        return None
    start = day.tz_convert("UTC").floor("1D")
    grid = pd.date_range(start, start + pd.Timedelta("1D") - pd.Timedelta("30s"), freq="30s")
    out = b.reindex(grid).ffill(limit=2).join(t.reindex(grid))
    out["price"] = out["price"].ffill(limit=4)
    out[["sell_quote", "buy_quote"]] = out[["sell_quote", "buy_quote"]].fillna(0.0)
    for c in ["bid1", "bid02"]:
        out[f"log_{c}"] = np.log(out[c].where(out[c] > 0))
    out["ret30"] = out["price"].pct_change()
    out["ret15"] = out["price"].pct_change(30)
    # Volatility estimate is strictly trailing and excludes the current tick.
    out["sig15"] = out["ret30"].rolling(240, min_periods=120).std().shift(1) * np.sqrt(30)
    out["shock_z"] = out["ret15"] / out["sig15"].replace(0.0, np.nan)
    return out


def _median(s: pd.Series) -> float:
    return float(s.median()) if s.notna().any() else np.nan


def _event_row(d: pd.DataFrame, sym: str, end: pd.Timestamp) -> dict | None:
    start = end - pd.Timedelta("15min")
    decision = end + pd.Timedelta("15min")
    if decision + pd.Timedelta("60min") > d.index.max():
        return None

    baseline = d.loc[end - pd.Timedelta("135min"): start - pd.Timedelta("30s")]
    before = d.loc[start - pd.Timedelta("150s"): start + pd.Timedelta("150s")]
    during = d.loc[start:end]
    ending = d.loc[end - pd.Timedelta("120s"):end]
    after = d.loc[end + pd.Timedelta("5min"):decision]
    if min(len(baseline), len(during), len(after)) < 20:
        return None

    pre1, pre02 = _median(before["bid1"]), _median(before["bid02"])
    base1, base02 = _median(baseline["bid1"]), _median(baseline["bid02"])
    end1, end02 = _median(ending["bid1"]), _median(ending["bid02"])
    post1, post02 = _median(after["bid1"]), _median(after["bid02"])
    trough1 = float(during["bid1"].quantile(0.10))
    trough02 = float(during["bid02"].quantile(0.10))
    primary = (pre1, base1, end1, post1, trough1)
    if not np.all(np.isfinite(primary)) or min(primary) <= 0:
        return None
    near = (pre02, base02, end02, post02, trough02)
    near_available = bool(np.all(np.isfinite(near)) and min(near) > 0)

    # Use the complete +/-1% band in every era.  +/-0.2% is absent from older
    # archives and is retained only as a coverage/confirmation diagnostic.
    pre_rel = np.log(pre1 / base1)
    bid_dd = np.log(trough1 / pre1)
    end_change = np.log(end1 / pre1)
    post_from_pre = np.log(post1 / pre1)
    recovery_log = np.log(post1 / trough1)
    recovery_frac = recovery_log / max(-bid_dd, 1e-6)

    sells, buys = float(during["sell_quote"].sum()), float(during["buy_quote"].sum())
    total = sells + buys
    sell_share = sells / total if total > 0 else np.nan
    net_sell_to_bid = (sells - buys) / pre1

    price_sell_time = _weighted_time(during.index, (-during["ret30"]).clip(lower=0))
    add_window = d.loc[start:decision]
    add_intensity = add_window["log_bid1"].diff().clip(lower=0)
    bid_add_time = _weighted_time(add_window.index, add_intensity)
    add_delay_min = (
        (bid_add_time - price_sell_time).total_seconds() / 60.0
        if pd.notna(bid_add_time) and pd.notna(price_sell_time) else np.nan
    )

    p_end = d.at[end, "price"]
    p_dec = d.at[decision, "price"]
    p_15 = d.at[decision + pd.Timedelta("15min"), "price"]
    p_60 = d.at[decision + pd.Timedelta("60min"), "price"]
    if not np.all(np.isfinite([p_end, p_dec, p_15, p_60])):
        return None
    class_ret = p_dec / p_end - 1.0
    class_min_ret = d.loc[end:decision, "price"].min() / p_end - 1.0

    post_imb1 = _median(after["imb1"])
    post_imb02 = _median(after["imb02"])
    final_bid_heavy = post_imb1 > 0

    # Fixed, interpretable mechanism thresholds.  These labels do not inspect
    # any return after `decision`.
    consumed = (
        pre_rel >= 0.0
        and sell_share >= 0.52
        and net_sell_to_bid > 0.0
        and bid_dd <= -0.10
    )
    if consumed and recovery_frac >= 2.0 / 3.0 and final_bid_heavy:
        if class_ret >= 0.0 and class_min_ret >= -0.001:
            label = "absorbed/stabilized"
        else:
            label = "replenished/leaking"
    elif consumed and recovery_frac <= 1.0 / 3.0:
        label = "consumed/exhausted"
    elif consumed:
        label = "consumed/partial"
    elif (
        pre_rel < 0.0
        and post_from_pre >= 0.10
        and end_change > -0.05
        and add_delay_min >= 1.0
        and final_bid_heavy
    ):
        label = "late-added"
    else:
        label = "ambiguous"

    return {
        "symbol": sym,
        "event_end": end,
        "decision": decision,
        "label": label,
        "near_band_available": near_available,
        "drop_bps": float(d.at[end, "ret15"] * 1e4),
        "shock_z": float(d.at[end, "shock_z"]),
        "pre_rel": pre_rel,
        "bid_dd": bid_dd,
        "end_change": end_change,
        "post_from_pre": post_from_pre,
        "recovery_frac": recovery_frac,
        "sell_share": sell_share,
        "net_sell_to_bid": net_sell_to_bid,
        "add_delay_min": add_delay_min,
        "post_imb1": post_imb1,
        "class_ret": class_ret,
        "class_min_ret": class_min_ret,
        "ret_after_15": p_15 / p_dec - 1.0,
        "ret_after_60": p_60 / p_dec - 1.0,
    }


def _events_for_day(sym: str, day: pd.Timestamp) -> list[dict]:
    d = _aligned_day(sym, day)
    if d is None:
        return []
    # Online trigger: a >=2-sigma and >=20-bp 15m fall.  After triggering,
    # wait up to 5m for the local low, then suppress overlapping events 45m.
    eligible = (d["shock_z"] <= -2.0) & (d["ret15"] <= -0.002)
    times = list(d.index[eligible.fillna(False)])
    rows: list[dict] = []
    next_allowed = d.index.min() + pd.Timedelta("135min")
    for trigger in times:
        if trigger < next_allowed:
            continue
        confirm = d.loc[trigger: trigger + pd.Timedelta("5min"), "price"].dropna()
        if confirm.empty:
            continue
        end = confirm.idxmin()
        row = _event_row(d, sym, end)
        if row is not None:
            rows.append(row)
        next_allowed = end + pd.Timedelta("45min")
    return rows


def _cluster_ci(sub: pd.DataFrame, col: str, seed: int = 17) -> tuple[float, float]:
    if sub.empty:
        return np.nan, np.nan
    x = sub[["event_end", col]].dropna().copy()
    x["day"] = x["event_end"].dt.floor("1D")
    groups = [g[col].to_numpy() for _, g in x.groupby("day")]
    if len(groups) < 4:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(2500)
    for i in range(len(boot)):
        pick = rng.integers(0, len(groups), len(groups))
        boot[i] = np.concatenate([groups[j] for j in pick]).mean()
    return tuple(np.percentile(boot, [2.5, 97.5]))


def _print_results(events: pd.DataFrame) -> None:
    events["era"] = np.where(events["event_end"] >= RECENT_CUT, "RECENT", "OOS")
    keep_order = [
        "late-added", "consumed/exhausted", "consumed/partial",
        "absorbed/stabilized", "replenished/leaking", "ambiguous",
    ]
    print(f"\nevents {len(events)} | symbols {events.symbol.nunique()} | days {events.event_end.dt.floor('1D').nunique()}")
    print("Returns start AFTER the 15m classification window (no outcome leakage).")
    for era in ["OOS", "RECENT"]:
        e = events[events["era"] == era]
        print(f"\n### {era}: {len(e)} downside shocks ###")
        print("label                    n   drop    sell%  bidDD  recovery  delay   fwd15 [95% CI]       fwd60 [95% CI]")
        for label in keep_order:
            s = e[e["label"] == label]
            if s.empty:
                continue
            l15, u15 = _cluster_ci(s, "ret_after_15")
            l60, u60 = _cluster_ci(s, "ret_after_60")
            print(
                f"{label:22s} {len(s):3d} "
                f"{s.drop_bps.mean():+6.0f}bp {100*s.sell_share.mean():5.1f}% "
                f"{100*s.bid_dd.mean():+5.1f}% {s.recovery_frac.median():7.2f} "
                f"{s.add_delay_min.median():+5.1f}m "
                f"{1e4*s.ret_after_15.mean():+6.1f} [{1e4*l15:+6.1f},{1e4*u15:+6.1f}] "
                f"{1e4*s.ret_after_60.mean():+6.1f} [{1e4*l60:+6.1f},{1e4*u60:+6.1f}]"
            )

    # Identification check: the two core mechanisms should differ strongly in
    # their pre-depth, depletion and replenishment observables by construction.
    core = events[events.label.isin(["late-added", "consumed/exhausted", "absorbed/stabilized", "replenished/leaking"])]
    if not core.empty:
        print("\nMechanism medians (sanity check; not returns):")
        print(core.groupby("label")[[
            "pre_rel", "bid_dd", "post_from_pre", "recovery_frac",
            "net_sell_to_bid", "add_delay_min", "class_ret",
        ]].median().round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--days", default=",".join(DEFAULT_DAYS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    syms = [s.strip().upper() for s in args.syms.split(",") if s.strip()]
    days = [pd.Timestamp(x.strip(), tz="UTC") for x in args.days.split(",") if x.strip()]

    jobs = [(s, d) for d in days for s in syms]
    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_events_for_day, s, d): (s, d) for s, d in jobs}
        for f in as_completed(futs):
            s, d = futs[f]
            try:
                rows.extend(f.result())
            except Exception as exc:
                print(f"  failed {s} {d:%Y-%m-%d}: {type(exc).__name__}: {exc}", flush=True)
            done += 1
            if done % 16 == 0 or done == len(jobs):
                print(f"  processed {done}/{len(jobs)} symbol-days | events {len(rows)}", flush=True)
    if not rows:
        raise SystemExit("no overlapping raw bookDepth + aggTrade events")
    _print_results(pd.DataFrame(rows).sort_values(["event_end", "symbol"]).reset_index(drop=True))
    print("\nSEQDONE")


if __name__ == "__main__":
    main()
