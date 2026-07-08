"""Block-level policy-regime probe for convexity v3.

The 4h policy-meta probe is too noisy/cost-dominated. This sidecar tests the
regime-level framing instead: choose one policy for a multi-day block using
only state known at the block start.

This is still an approximation because candidate PnL comes from separate
always-on full replays. It is a screen for learnability before implementing a
true stateful meta-policy replay.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from live.convexity_paper_bot import load_close_4h  # noqa: E402

ROOT = REPO / "live/state/longtail"
OUT = REPO / "live/state/policy_regime_probe"
OUT.mkdir(parents=True, exist_ok=True)

POLICY_TAGS = {
    "return1d": "tbull_base",
    "rvol": "tbull_rvol",
    "bd12_rvol": "tbull_bd12_rvol",
    "cause66": "tbull_cause66",
}
ANN = np.sqrt(365)


def load_cycles(tag: str) -> pd.DataFrame:
    p = ROOT / tag / "state/cycles.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    c = pd.read_csv(p)
    c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    return c.sort_values("open_time").set_index("open_time")


def maxdd(vals: list[float] | np.ndarray | pd.Series) -> float:
    s = pd.Series(vals, dtype=float).fillna(0.0).cumsum()
    return float((s - s.cummax()).min())


def btc_features(index: pd.Index) -> pd.DataFrame:
    btc = load_close_4h("BTCUSDT")
    r = btc.pct_change()
    f = pd.DataFrame(index=btc.index)
    for name, n in [("btc_ret_1d", 6), ("btc_ret_3d", 18), ("btc_ret_7d", 42),
                    ("btc_ret_30d", 180), ("btc_ret_90d", 540)]:
        f[name] = btc / btc.shift(n) - 1.0
    f["btc_rvol_7d"] = r.rolling(42).std() * np.sqrt(42)
    f["btc_rvol_30d"] = r.rolling(180).std() * np.sqrt(180)
    f["btc_smooth_30d"] = f["btc_ret_30d"].abs() / (f["btc_rvol_30d"] + 1e-12)
    f["btc_accel_7v30"] = f["btc_ret_7d"] - f["btc_ret_30d"] / 4.0
    return f.reindex(index, method="ffill")


def block_table(block_days: int) -> pd.DataFrame:
    runs = {name: load_cycles(tag) for name, tag in POLICY_TAGS.items()}
    common = None
    for c in runs.values():
        common = c.index if common is None else common.intersection(c.index)
    runs = {k: v.loc[common].copy() for k, v in runs.items()}
    base = runs["return1d"]

    start = common.min().ceil(f"{block_days}D")
    end = common.max().floor(f"{block_days}D")
    starts = pd.date_range(start, end, freq=f"{block_days}D", tz="UTC")
    bf = btc_features(starts)
    rows = []
    for st in starts:
        en = st + pd.Timedelta(days=block_days)
        row = {"block_start": st, "block_end": en, "block_days": block_days}
        cur = bf.loc[st]
        for col, val in cur.items():
            row[col] = float(val) if pd.notna(val) else np.nan

        hist30 = base[(base.index >= st - pd.Timedelta(days=30)) & (base.index < st)]
        hist90 = base[(base.index >= st - pd.Timedelta(days=90)) & (base.index < st)]
        for name, hist in [("h30", hist30), ("h90", hist90)]:
            row[f"{name}_n"] = float(len(hist))
            row[f"{name}_pnl"] = float(hist["pnl_bps"].sum()) if len(hist) else 0.0
            row[f"{name}_pred_disp"] = float(hist["pred_disp"].mean()) if len(hist) else np.nan
            row[f"{name}_gross"] = float(hist["gross_after_stop"].mean()) if len(hist) else np.nan
            row[f"{name}_stop_pct"] = float(hist["stop_engaged"].mean()) if len(hist) else np.nan
            row[f"{name}_bull_frac"] = float((hist["regime"] == "bull").mean()) if len(hist) else 0.0
            row[f"{name}_side_frac"] = float((hist["regime"] == "side").mean()) if len(hist) else 0.0
            row[f"{name}_bear_frac"] = float((hist["regime"] == "bear").mean()) if len(hist) else 0.0

        last = base[base.index < st].tail(1)
        regime = str(last["regime"].iloc[0]) if len(last) else "unknown"
        for reg in ["bull", "side", "bear"]:
            row[f"cur_{reg}"] = float(regime == reg)

        for pol, c in runs.items():
            g = c[(c.index >= st) & (c.index < en)]
            row[f"pnl_{pol}"] = float(g["pnl_bps"].sum())
            row[f"dd_{pol}"] = maxdd(g["pnl_bps"]) if len(g) else 0.0
            row[f"gross_{pol}"] = float(g["gross_after_stop"].mean()) if len(g) else 0.0
        row["pnl_flat"] = 0.0
        rows.append(row)
    out = pd.DataFrame(rows).dropna(subset=["btc_ret_30d"]).sort_values("block_start")
    return out


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, ycol: str,
                features: list[str], lam: float = 10.0) -> np.ndarray:
    x = train[features].astype(float).to_numpy()
    y = train[ycol].astype(float).to_numpy()
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd < 1e-12] = 1.0
    x = np.where(np.isfinite(x), x, mu)
    xs = np.c_[np.ones(len(x)), (x - mu) / sd]
    reg = np.eye(xs.shape[1]) * lam
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xs.T @ xs + reg, xs.T @ y)
    xt = test[features].astype(float).to_numpy()
    xt = np.where(np.isfinite(xt), xt, mu)
    return np.c_[np.ones(len(xt)), (xt - mu) / sd] @ beta


def evaluate_blocks(df: pd.DataFrame) -> pd.DataFrame:
    policy_names = list(POLICY_TAGS) + ["flat"]
    policy_cols = [f"pnl_{p}" for p in policy_names]
    features = [
        "btc_ret_1d", "btc_ret_3d", "btc_ret_7d", "btc_ret_30d", "btc_ret_90d",
        "btc_rvol_7d", "btc_rvol_30d", "btc_smooth_30d", "btc_accel_7v30",
        "h30_pnl", "h30_pred_disp", "h30_gross", "h30_stop_pct",
        "h30_bull_frac", "h30_side_frac", "h30_bear_frac",
        "h90_pnl", "h90_pred_disp", "h90_gross", "h90_stop_pct",
        "h90_bull_frac", "h90_side_frac", "h90_bear_frac",
        "cur_bull", "cur_side", "cur_bear",
    ]
    features = [f for f in features if f in df.columns]
    rows = []
    for margin in [0.0, 25.0, 50.0, 100.0]:
        selected = []
        fixed = {p: [] for p in policy_names}
        oracle = []
        choices = {p: 0 for p in policy_names}
        for year in [2023, 2024, 2025, 2026]:
            train = df[df["block_start"] < pd.Timestamp(f"{year}-01-01", tz="UTC")]
            test = df[(df["block_start"] >= pd.Timestamp(f"{year}-01-01", tz="UTC")) &
                      (df["block_start"] < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))]
            if len(train) < 18 or test.empty:
                continue
            pred_mat = []
            for p in policy_names:
                if p == "flat":
                    pred_mat.append(np.zeros(len(test)))
                else:
                    pred_mat.append(fit_predict(train, test, f"pnl_{p}", features))
            pred_mat = np.vstack(pred_mat).T
            actual = test[policy_cols].to_numpy(float)
            best_i = pred_mat.argmax(axis=1)
            best_v = pred_mat.max(axis=1)
            flat_i = policy_names.index("flat")
            chosen = np.where(best_v > margin, best_i, flat_i)
            pnl = actual[np.arange(len(test)), chosen]
            selected.extend(pnl.tolist())
            oracle.extend(actual.max(axis=1).tolist())
            for i, p in enumerate(policy_names):
                fixed[p].extend(test[f"pnl_{p}"].tolist())
                choices[p] += int((chosen == i).sum())
            rec = {
                "block_days": int(df["block_days"].iloc[0]),
                "margin": margin,
                "year": year,
                "selector": float(pnl.sum()),
                "oracle": float(actual.max(axis=1).sum()),
            }
            for p in policy_names:
                rec[f"fixed_{p}"] = float(test[f"pnl_{p}"].sum())
            rows.append(rec)
        total = {
            "block_days": int(df["block_days"].iloc[0]),
            "margin": margin,
            "year": "TOTAL",
            "selector": float(np.sum(selected)),
            "selector_dd": maxdd(selected),
            "oracle": float(np.sum(oracle)),
            **{f"fixed_{p}": float(np.sum(v)) for p, v in fixed.items()},
            **{f"choose_{p}": int(v) for p, v in choices.items()},
        }
        rows.append(total)
    return pd.DataFrame(rows)


def main() -> None:
    all_results = []
    for block_days in [14, 30, 60]:
        df = block_table(block_days)
        df.to_csv(OUT / f"blocks_{block_days}d.csv", index=False)
        res = evaluate_blocks(df)
        res.to_csv(OUT / f"results_{block_days}d.csv", index=False)
        all_results.append(res)
        print(f"\n=== {block_days}d blocks ===")
        totals = res[res["year"].astype(str) == "TOTAL"]
        keep = ["margin", "selector", "selector_dd", "oracle",
                "fixed_return1d", "fixed_rvol", "fixed_bd12_rvol",
                "fixed_cause66", "fixed_flat",
                "choose_return1d", "choose_rvol", "choose_bd12_rvol",
                "choose_cause66", "choose_flat"]
        print(totals[[c for c in keep if c in totals.columns]].to_string(index=False))
    out = pd.concat(all_results, ignore_index=True)
    out.to_csv(OUT / "results_all.csv", index=False)
    (OUT / "policy_regime_probe.json").write_text(json.dumps({
        "policy_tags": POLICY_TAGS,
        "note": "Block-level approximate selector from separate full replay paths; screen only.",
    }, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
