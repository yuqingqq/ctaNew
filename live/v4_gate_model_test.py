#!/usr/bin/env python3
"""Simple model-native gate test for v4 residual mean-reversion.

This script evaluates whether PIT model/output features can identify cycles
where the v4 residual model is in a favorable distribution.

Metric is model-only 24h residual alpha, not bot PnL:
    edge = mean(alpha24h(selected longs)) - mean(alpha24h(selected shorts))

The strategy shape mirrors current v4 analysis: K_LONG=1 by long-book pred,
K_SHORT=2 by base-book pred.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")

ROOT = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(ROOT))

from live.convexity_paper_bot import compute_btc_30d
HOLD = 6
K_LONG = 1
K_SHORT = 2

BOOKS = {
    "oos_clean": (
        ROOT / "live/state/convexity/hl_v4base_oos_clean/v0full_hl60.parquet",
        ROOT / "live/state/convexity/hl_v4long_oos_clean/v0full_hl60.parquet",
    ),
    "recent_clean": (
        ROOT / "live/state/convexity/hl_tgt_res_base_clean/v0full_hl60.parquet",
        ROOT / "live/state/convexity/hl_tgt_res_long_clean/v0full_hl60.parquet",
    ),
}

FEATURE_COLS = [
    "btc_ret_30d",
    "abs_btc_ret_30d",
    "pred_gap",
    "pred_base_std",
    "pred_long_std",
    "long_pred_mean",
    "short_pred_mean",
    "long_ret1d",
    "short_ret1d",
    "long_ret3d",
    "short_ret3d",
    "long_trail3_resid",
    "short_trail3_resid",
    "long_rvol_7d",
    "short_rvol_7d",
    "long_corr_to_btc",
    "short_corr_to_btc",
    "long_funding_z",
    "short_funding_z",
    "xs_ret1d_std",
    "xs_rvol_mean",
    "n_symbols",
]


def period_label(ts: pd.Timestamp) -> str:
    if ts.year == 2025:
        return "2025H1" if ts < pd.Timestamp("2025-07-01", tz="UTC") else "2025H2"
    return str(ts.year)


def macro_regime(x: float) -> str:
    if x < -0.10:
        return "bear"
    if x > 0.10:
        return "bull"
    return "side"


def btc30_bucket(x: float) -> str:
    if x < -0.20:
        return "bear_deep"
    if x < -0.15:
        return "bear_mid"
    if x < -0.10:
        return "bear_mild"
    if x < -0.05:
        return "side_down"
    if x <= 0.05:
        return "side_flat"
    if x <= 0.10:
        return "side_up"
    if x < 0.15:
        return "bull_mild"
    if x < 0.20:
        return "bull_hot"
    return "bull_deep"


def load_panel_targets() -> pd.DataFrame:
    cols = [
        "symbol",
        "open_time",
        "alpha_vs_btc_realized",
        "return_1d",
        "ret_3d",
        "rvol_7d",
        "corr_to_btc_1d",
        "funding_rate_z_7d",
    ]
    pan = pd.read_parquet(ROOT / "outputs/vBTC_features/panel_expanded_v0.parquet", columns=cols)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan = pan.sort_values(["symbol", "open_time"])
    by_sym = pan.groupby("symbol")
    pan["fwd_resid_24h_bps"] = (
        by_sym["alpha_vs_btc_realized"]
        .transform(lambda s: s.shift(-1).rolling(HOLD).sum().shift(-(HOLD - 1)))
        * 1e4
    )
    pan["trail3_resid_bps"] = (
        by_sym["alpha_vs_btc_realized"].transform(lambda s: s.shift(1).rolling(3).sum()) * 1e4
    )
    return pan.dropna(subset=["fwd_resid_24h_bps"])


def load_book(base_path: Path, long_path: Path, panel: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_parquet(base_path, columns=["symbol", "open_time", "pred"]).rename(
        columns={"pred": "pred_base"}
    )
    long = pd.read_parquet(long_path, columns=["symbol", "open_time", "pred"]).rename(
        columns={"pred": "pred_long"}
    )
    for df in (base, long):
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    data = (
        base.merge(long, on=["symbol", "open_time"], how="inner")
        .merge(panel, on=["symbol", "open_time"], how="inner")
        .dropna(subset=["pred_base", "pred_long", "fwd_resid_24h_bps"])
    )
    return data


def build_cycle_dataset() -> pd.DataFrame:
    panel = load_panel_targets()
    btc30 = compute_btc_30d().dropna().rename("btc_ret_30d").reset_index()
    frames = []
    for window, (base_path, long_path) in BOOKS.items():
        d = load_book(base_path, long_path, panel)
        d = d.merge(btc30, on="open_time", how="left").dropna(subset=["btc_ret_30d"])
        d["window"] = window
        frames.append(d)
    rows = []
    all_rows = pd.concat(frames, ignore_index=True).sort_values(["open_time", "symbol"])
    for (window, open_time), g in all_rows.groupby(["window", "open_time"], sort=False):
        if len(g) < K_LONG + K_SHORT:
            continue
        longs = g.nlargest(K_LONG, "pred_long")
        shorts = g.nsmallest(K_SHORT, "pred_base")
        edge_long = float(longs["fwd_resid_24h_bps"].mean())
        short_pick = float(shorts["fwd_resid_24h_bps"].mean())
        row = {
            "window": window,
            "open_time": open_time,
            "period": period_label(open_time),
            "btc_ret_30d": float(g["btc_ret_30d"].iloc[0]),
            "n_symbols": int(len(g)),
            "long_edge_bps": edge_long,
            "short_pick_bps": short_pick,
            "short_edge_bps": -short_pick,
            "edge_bps": edge_long - short_pick,
            "pred_base_std": float(g["pred_base"].std()),
            "pred_long_std": float(g["pred_long"].std()),
            "long_pred_mean": float(longs["pred_long"].mean()),
            "short_pred_mean": float(shorts["pred_base"].mean()),
            "long_ret1d": float(longs["return_1d"].mean()),
            "short_ret1d": float(shorts["return_1d"].mean()),
            "long_ret3d": float(longs["ret_3d"].mean()),
            "short_ret3d": float(shorts["ret_3d"].mean()),
            "long_trail3_resid": float(longs["trail3_resid_bps"].mean()),
            "short_trail3_resid": float(shorts["trail3_resid_bps"].mean()),
            "long_rvol_7d": float(longs["rvol_7d"].mean()),
            "short_rvol_7d": float(shorts["rvol_7d"].mean()),
            "long_corr_to_btc": float(longs["corr_to_btc_1d"].mean()),
            "short_corr_to_btc": float(shorts["corr_to_btc_1d"].mean()),
            "long_funding_z": float(longs["funding_rate_z_7d"].mean()),
            "short_funding_z": float(shorts["funding_rate_z_7d"].mean()),
            "xs_ret1d_std": float(g["return_1d"].std()),
            "xs_rvol_mean": float(g["rvol_7d"].mean()),
        }
        row["abs_btc_ret_30d"] = abs(row["btc_ret_30d"])
        row["pred_gap"] = row["long_pred_mean"] - row["short_pred_mean"]
        row["macro_regime"] = macro_regime(row["btc_ret_30d"])
        row["btc30_bucket"] = btc30_bucket(row["btc_ret_30d"])
        rows.append(row)
    data = pd.DataFrame(rows).sort_values("open_time").reset_index(drop=True)
    return add_pit_bucket_features(data)


def add_pit_bucket_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    # Conservative thresholds: known only after prior 24h holds would have closed.
    shifted = d.shift(HOLD)
    win = 540
    minp = 180
    d["pred_gap_med_pit"] = shifted["pred_gap"].rolling(win, min_periods=minp).median()
    d["long_ret3d_p25_pit"] = shifted["long_ret3d"].rolling(win, min_periods=minp).quantile(0.25)
    d["short_ret3d_p75_pit"] = shifted["short_ret3d"].rolling(win, min_periods=minp).quantile(0.75)
    d["conf_bucket"] = np.where(d["pred_gap"] >= d["pred_gap_med_pit"], "conf_hi", "conf_lo")
    d.loc[d["pred_gap_med_pit"].isna(), "conf_bucket"] = "conf_unk"
    d["long_state"] = np.where(d["long_ret3d"] <= d["long_ret3d_p25_pit"], "long_knife", "long_normal")
    d.loc[d["long_ret3d_p25_pit"].isna(), "long_state"] = "long_unk"
    d["short_state"] = np.where(d["short_ret3d"] >= d["short_ret3d_p75_pit"], "short_squeeze", "short_normal")
    d.loc[d["short_ret3d_p75_pit"].isna(), "short_state"] = "short_unk"
    d["model_bucket"] = (
        d["macro_regime"] + "|" + d["conf_bucket"] + "|" + d["long_state"] + "|" + d["short_state"]
    )
    return d


def lagged_global_gate(data: pd.DataFrame, window: int = 180, min_periods: int = 60) -> np.ndarray:
    trail = data["edge_bps"].shift(HOLD).rolling(window, min_periods=min_periods).mean()
    return (trail > 0).fillna(True).to_numpy(dtype=bool)


def bucket_report_card_gate(
    data: pd.DataFrame,
    detail_col: str = "model_bucket",
    fallback_col: str = "macro_regime",
    window: int = 540,
    min_detail: int = 30,
    min_fallback: int = 60,
) -> np.ndarray:
    d = data.reset_index(drop=True)
    gates = np.ones(len(d), dtype=bool)
    history: list[dict] = []
    for i, row in d.iterrows():
        # Add cycles whose HOLD horizon has closed.
        close_idx = i - HOLD
        if close_idx >= 0:
            history.append(d.iloc[close_idx].to_dict())
            if len(history) > window:
                history = history[-window:]
        if not history:
            gates[i] = True
            continue
        hist = pd.DataFrame(history)
        detail = hist[hist[detail_col] == row[detail_col]]
        if len(detail) >= min_detail:
            gates[i] = detail["edge_bps"].mean() > 0
            continue
        fallback = hist[hist[fallback_col] == row[fallback_col]]
        if len(fallback) >= min_fallback:
            gates[i] = fallback["edge_bps"].mean() > 0
        else:
            gates[i] = True
    return gates


def folds(data: pd.DataFrame) -> list[tuple[list[str], str]]:
    periods = ["2023", "2024", "2025H1", "2025H2", "2026"]
    out = []
    for i in range(1, len(periods)):
        out.append((periods[:i], periods[i]))
    return out


def evaluate(data: pd.DataFrame, gate: np.ndarray, name: str, eval_mask: np.ndarray) -> dict:
    mask = eval_mask.astype(bool)
    g = gate.astype(bool) & mask
    y = np.where(g, data["edge_bps"].to_numpy(), 0.0)[mask]
    active = data.loc[g, "edge_bps"]
    trade_frac = float(g.sum() / mask.sum()) if mask.sum() else np.nan
    std = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
    cal_sh = float(np.mean(y) / std * np.sqrt(len(y))) if std and std > 0 else np.nan
    return {
        "strategy": name,
        "eval_cycles": int(mask.sum()),
        "traded_cycles": int(g.sum()),
        "trade_frac": trade_frac,
        "calendar_edge_bps": float(np.mean(y)) if len(y) else np.nan,
        "active_edge_bps": float(active.mean()) if len(active) else np.nan,
        "active_hit_rate": float((active > 0).mean()) if len(active) else np.nan,
        "calendar_sh_like": cal_sh,
        "total_edge_bps": float(y.sum()) if len(y) else np.nan,
    }


def random_matched(data: pd.DataFrame, trade_count: int, eval_mask: np.ndarray, seeds: int = 500) -> dict:
    idx = np.flatnonzero(eval_mask)
    if trade_count <= 0:
        return {"random_mean_total": 0.0, "random_p90_total": 0.0, "random_p10_total": 0.0}
    vals = data["edge_bps"].to_numpy()
    totals = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        pick = rng.choice(idx, size=min(trade_count, len(idx)), replace=False)
        totals.append(float(vals[pick].sum()))
    arr = np.array(totals)
    return {
        "random_mean_total": float(arr.mean()),
        "random_p10_total": float(np.quantile(arr, 0.10)),
        "random_p90_total": float(np.quantile(arr, 0.90)),
    }



def frame_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def tree_gates(data: pd.DataFrame, max_depth: int) -> tuple[np.ndarray, list[dict]]:
    gate = np.zeros(len(data), dtype=bool)
    fold_rows = []
    for train_periods, test_period in folds(data):
        train = data["period"].isin(train_periods)
        test = data["period"] == test_period
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=180, random_state=7),
        )
        X_train = data.loc[train, FEATURE_COLS]
        y_train = data.loc[train, "edge_bps"]
        model.fit(X_train, y_train)
        pred = model.predict(data.loc[test, FEATURE_COLS])
        gate[test.to_numpy()] = pred > 0
        tree = model.named_steps["decisiontreeregressor"]
        rules = export_text(tree, feature_names=FEATURE_COLS, decimals=2)
        fold_rows.append(
            {
                "model": f"tree_d{max_depth}",
                "test_period": test_period,
                "train_periods": ",".join(train_periods),
                "test_cycles": int(test.sum()),
                "trade_frac": float((pred > 0).mean()),
                "pred_mean": float(np.mean(pred)),
                "rules": rules,
            }
        )
    return gate, fold_rows


def main() -> None:
    out_dir = ROOT / "live"
    data = build_cycle_dataset()
    data.to_parquet(out_dir / "V4_GATE_MODEL_DATASET.parquet", index=False)

    eval_mask = data["period"].isin(["2024", "2025H1", "2025H2", "2026"]).to_numpy()
    gates: dict[str, np.ndarray] = {
        "always": np.ones(len(data), dtype=bool),
        "btc_no_bull": (data["macro_regime"] != "bull").to_numpy(),
        "btc_bear_only": (data["macro_regime"] == "bear").to_numpy(),
        "lagged_global_edge_180": lagged_global_gate(data, 180, 60),
        "bucket_report_card": bucket_report_card_gate(data),
    }
    fold_info = []
    for depth in (2, 3):
        gate, rows = tree_gates(data, depth)
        gates[f"tree_d{depth}"] = gate
        fold_info.extend(rows)

    decisions = data.copy()
    for name, gate in gates.items():
        decisions[f"gate_{name}"] = gate
    decisions.to_csv(out_dir / "V4_GATE_MODEL_DECISIONS.csv", index=False)

    rows = []
    for name, gate in gates.items():
        row = evaluate(data, gate, name, eval_mask)
        row.update(random_matched(data, row["traded_cycles"], eval_mask))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("calendar_edge_bps", ascending=False)
    summary.to_csv(out_dir / "V4_GATE_MODEL_COMPARISON.csv", index=False)
    pd.DataFrame(fold_info).to_csv(out_dir / "V4_GATE_MODEL_FOLD_RULES.csv", index=False)

    fold_metrics = []
    for period in ["2024", "2025H1", "2025H2", "2026"]:
        mask = (data["period"] == period).to_numpy()
        for name, gate in gates.items():
            r = evaluate(data, gate, name, mask)
            r["period"] = period
            fold_metrics.append(r)
    pd.DataFrame(fold_metrics).to_csv(out_dir / "V4_GATE_MODEL_BY_PERIOD.csv", index=False)

    report = [
        "# V4 Simple Gate Model Comparison",
        "",
        "Metric: model-only 24h residual alpha edge in bps. Evaluation is expanding walk-forward periods 2024, 2025H1, 2025H2, 2026. Skipped cycles contribute zero edge.",
        "",
        "## Summary",
        "",
        frame_to_markdown(summary),
        "",
        "## Notes",
        "",
        "- `always`: trade every v4 model cycle.",
        "- `btc_no_bull`: hardcoded BTC30 gate; trade bear+side, skip bull.",
        "- `btc_bear_only`: farm only the hardcoded bear macro regime.",
        "- `lagged_global_edge_180`: trade only when the prior closed 180-cycle model edge is positive.",
        "- `bucket_report_card`: trade if the current model-state bucket has positive prior closed edge, falling back to macro regime history.",
        "- `tree_d2/tree_d3`: shallow decision-tree regressors trained on prior periods to predict next-period edge; trade if predicted edge > 0.",
        "",
        "Random columns are matched-skip baselines using the same number of traded cycles.",
        "",
    ]
    (out_dir / "V4_GATE_MODEL_REPORT.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(f"WROTE {out_dir / 'V4_GATE_MODEL_REPORT.md'}")


if __name__ == "__main__":
    main()
