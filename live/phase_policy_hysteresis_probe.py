"""Rolling policy-regime selector with hysteresis.

This is a sidecar screen, not an engine change. It uses the existing fixed
full-replay paths and asks whether a past-only regime model can choose among
policies more dynamically than the coarse 14d block probe.

Approximation: PnL is stitched from separately replayed paths. That preserves
policy-level path PnL by cycle, but it does not reproduce sleeve state,
turnover netting, or stop/gate interactions of a true dynamic meta-policy.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from live.convexity_paper_bot import load_close_4h  # noqa: E402

ROOT = REPO / "live/state/longtail"
OUT = REPO / "live/state/policy_hysteresis_probe"
OUT.mkdir(parents=True, exist_ok=True)

POLICY_TAGS = {
    "return1d": "tbull_base",
    "rvol": "tbull_rvol",
    "bd12_rvol": "tbull_bd12_rvol",
    "cause66": "tbull_cause66",
}
POLICIES = list(POLICY_TAGS) + ["flat"]
ANN_4H = math.sqrt(365.0 * 6.0)


@dataclass(frozen=True)
class HysteresisConfig:
    horizon_days: int
    persist_m: int
    persist_n: int
    min_hold_days: int
    min_edge_bps: float
    switch_margin_bps: float
    allow_flat: bool = True


def maxdd(vals: list[float] | np.ndarray | pd.Series) -> float:
    s = pd.Series(vals, dtype=float).fillna(0.0).cumsum()
    return float((s - s.cummax()).min())


def sharpe(vals: list[float] | np.ndarray | pd.Series) -> float:
    s = pd.Series(vals, dtype=float).dropna()
    sd = float(s.std(ddof=0))
    if len(s) == 0 or sd < 1e-12:
        return 0.0
    return float(s.mean() / sd * ANN_4H)


def load_cycles(tag: str) -> pd.DataFrame:
    p = ROOT / tag / "state/cycles.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    c = pd.read_csv(p)
    c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    return c.sort_values("open_time").set_index("open_time")


def load_runs() -> dict[str, pd.DataFrame]:
    runs = {name: load_cycles(tag) for name, tag in POLICY_TAGS.items()}
    common = None
    for c in runs.values():
        common = c.index if common is None else common.intersection(c.index)
    if common is None or len(common) == 0:
        raise RuntimeError("No common cycle timestamps across policy runs")
    return {k: v.loc[common].copy() for k, v in runs.items()}


def btc_feature_frame(index: pd.Index) -> pd.DataFrame:
    btc = load_close_4h("BTCUSDT")
    ret = btc.pct_change()
    f = pd.DataFrame(index=btc.index)
    for name, n in [
        ("btc_ret_1d", 6),
        ("btc_ret_3d", 18),
        ("btc_ret_7d", 42),
        ("btc_ret_14d", 84),
        ("btc_ret_30d", 180),
        ("btc_ret_90d", 540),
    ]:
        f[name] = btc / btc.shift(n) - 1.0
    f["btc_rvol_3d"] = ret.rolling(18).std() * np.sqrt(18)
    f["btc_rvol_7d"] = ret.rolling(42).std() * np.sqrt(42)
    f["btc_rvol_30d"] = ret.rolling(180).std() * np.sqrt(180)
    f["btc_smooth_30d"] = f["btc_ret_30d"].abs() / (f["btc_rvol_30d"] + 1e-12)
    f["btc_accel_7v30"] = f["btc_ret_7d"] - f["btc_ret_30d"] / 4.0
    f["btc_accel_3v14"] = f["btc_ret_3d"] - f["btc_ret_14d"] * (3.0 / 14.0)
    return f.reindex(index, method="ffill")


def build_features(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = runs["return1d"]
    f = btc_feature_frame(base.index)

    for days in [14, 30, 90]:
        win = f"{days}D"
        prefix = f"h{days}"
        f[f"{prefix}_pnl"] = base["pnl_bps"].shift(1).rolling(win).sum()
        f[f"{prefix}_pred_disp"] = base["pred_disp"].shift(1).rolling(win).mean()
        f[f"{prefix}_gross"] = base["gross_after_stop"].shift(1).rolling(win).mean()
        f[f"{prefix}_stop_pct"] = base["stop_engaged"].astype(float).shift(1).rolling(win).mean()
        for reg in ["bull", "side", "bear"]:
            f[f"{prefix}_{reg}_frac"] = (
                (base["regime"] == reg).astype(float).shift(1).rolling(win).mean()
            )

    prev_regime = base["regime"].shift(1).fillna("unknown")
    for reg in ["bull", "side", "bear"]:
        f[f"cur_{reg}"] = (prev_regime == reg).astype(float)

    # Current policy features are intentionally omitted; this is a regime
    # detector, not a Markov policy-state model.
    return f


def forward_sum(s: pd.Series, bars: int) -> pd.Series:
    return s.fillna(0.0).iloc[::-1].rolling(bars, min_periods=bars).sum().iloc[::-1]


def feature_columns(df: pd.DataFrame) -> list[str]:
    banned = {"open_time"}
    return [c for c in df.columns if c not in banned and pd.api.types.is_numeric_dtype(df[c])]


def fit_predict_ridge(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    lam: float = 10.0,
) -> np.ndarray:
    x = train_x.astype(float).to_numpy()
    y = train_y.astype(float).to_numpy()
    ok_y = np.isfinite(y)
    x = x[ok_y]
    y = y[ok_y]
    if len(y) == 0:
        return np.full(len(test_x), np.nan)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd < 1e-12] = 1.0
    x = np.where(np.isfinite(x), x, mu)
    xs = np.c_[np.ones(len(x)), (x - mu) / sd]
    reg = np.eye(xs.shape[1]) * lam
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xs.T @ xs + reg, xs.T @ y)

    xt = test_x.astype(float).to_numpy()
    xt = np.where(np.isfinite(xt), xt, mu)
    return np.c_[np.ones(len(xt)), (xt - mu) / sd] @ beta


def score_policy_values(
    runs: dict[str, pd.DataFrame],
    features: pd.DataFrame,
    horizon_days: int,
    *,
    eval_start: str = "2023-01-01",
    score_step_bars: int = 6,
    min_train_days: int = 180,
) -> pd.DataFrame:
    idx = features.index
    horizon_bars = horizon_days * 6
    min_train = min_train_days * 6
    labels = {
        p: forward_sum(runs[p]["pnl_bps"], horizon_bars)
        for p in POLICY_TAGS
    }
    cols = feature_columns(features)
    eval_times = idx[(idx >= pd.Timestamp(eval_start, tz="UTC"))][::score_step_bars]

    rows = []
    for t in eval_times:
        train_end = t - pd.Timedelta(days=horizon_days)
        train_mask = (idx <= train_end) & features[cols].notna().all(axis=1)
        test = features.loc[[t], cols]
        if int(train_mask.sum()) < min_train or test.isna().any(axis=None):
            continue
        rec = {"open_time": t}
        train_x = features.loc[train_mask, cols]
        for p in POLICY_TAGS:
            rec[f"score_{p}"] = float(
                fit_predict_ridge(train_x, labels[p].loc[train_mask], test)[0]
            )
        rec["score_flat"] = 0.0
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No score rows for horizon_days={horizon_days}")
    return out.set_index("open_time").sort_index()


def choose_desired(scores: pd.Series, min_edge_bps: float) -> str:
    score_vals = {p: float(scores[f"score_{p}"]) for p in POLICIES}
    best = max(score_vals, key=score_vals.get)
    if score_vals[best] <= min_edge_bps:
        return "flat"
    return best


def replay_hysteresis(
    runs: dict[str, pd.DataFrame],
    scores: pd.DataFrame,
    cfg: HysteresisConfig,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    idx = next(iter(runs.values())).index
    pnl_by_policy = {
        p: (np.zeros(len(idx)) if p == "flat" else runs[p]["pnl_bps"].to_numpy(float))
        for p in POLICIES
    }
    score_pos = {t: i for i, t in enumerate(scores.index)}
    score_cols = [f"score_{p}" for p in POLICIES]
    score_arr = scores[score_cols].to_numpy(float)
    score_col_ix = {p: i for i, p in enumerate(POLICIES)}
    choices = POLICIES if cfg.allow_flat else list(POLICY_TAGS)
    current = "flat"
    last_switch = None
    desired_hist: list[str] = []
    active_start = None
    records = []

    for i, t in enumerate(idx):
        j = score_pos.get(t)
        if j is not None:
            row_scores = score_arr[j]
            choice_ix = [score_col_ix[p] for p in choices]
            best_local_i = int(np.nanargmax(row_scores[choice_ix]))
            desired = choices[best_local_i]
            best_score = float(row_scores[score_col_ix[desired]])
            if cfg.allow_flat and best_score <= cfg.min_edge_bps:
                desired = "flat"
            desired_hist.append(desired)
            desired_hist = desired_hist[-cfg.persist_m:]

            enough_history = len(desired_hist) >= cfg.persist_m
            persistent = enough_history and desired_hist.count(desired) >= cfg.persist_n
            hold_ok = (
                last_switch is None
                or t - last_switch >= pd.Timedelta(days=cfg.min_hold_days)
            )
            if desired != current and persistent and hold_ok:
                score_desired = float(row_scores[score_col_ix[desired]])
                score_current = float(row_scores[score_col_ix[current]])
                switch_ok = score_desired >= score_current + cfg.switch_margin_bps
                if current == "flat" and not cfg.allow_flat:
                    switch_ok = True
                if switch_ok:
                    current = desired
                    last_switch = t

            if active_start is None and enough_history and (cfg.allow_flat or current != "flat"):
                active_start = t

        pnl = float(pnl_by_policy[current][i])
        records.append({"open_time": t, "policy": current, "pnl_bps": pnl})

    path = pd.DataFrame(records).set_index("open_time")
    if active_start is None:
        raise RuntimeError("No active_start; persistence window never filled")
    path = path.loc[path.index >= active_start].copy()

    counts = path["policy"].value_counts().to_dict()
    switches = int((path["policy"] != path["policy"].shift(1)).sum())
    stats: dict[str, float | int | str] = {
        **asdict(cfg),
        "start": str(active_start),
        "total": float(path["pnl_bps"].sum()),
        "maxdd": maxdd(path["pnl_bps"]),
        "sharpe": sharpe(path["pnl_bps"]),
        "switches": switches,
    }
    for p in POLICIES:
        stats[f"choose_{p}"] = int(counts.get(p, 0))
    return path, stats


def fixed_stats(runs: dict[str, pd.DataFrame], start: pd.Timestamp) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in POLICIES:
        vals = (
            pd.Series(0.0, index=next(iter(runs.values())).loc[start:].index)
            if p == "flat"
            else runs[p].loc[start:, "pnl_bps"]
        )
        out[f"fixed_{p}_total"] = float(vals.sum())
        out[f"fixed_{p}_maxdd"] = maxdd(vals)
        out[f"fixed_{p}_sharpe"] = sharpe(vals)
    return out


def per_year(path: pd.DataFrame, runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    start = path.index.min()
    for year in sorted(path.index.year.unique()):
        mask = path.index.year == year
        rec = {
            "year": int(year),
            "selector": float(path.loc[mask, "pnl_bps"].sum()),
        }
        for p in POLICIES:
            if p == "flat":
                rec[f"fixed_{p}"] = 0.0
            else:
                vals = runs[p].loc[(runs[p].index >= start) & (runs[p].index.year == year), "pnl_bps"]
                rec[f"fixed_{p}"] = float(vals.sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    runs = load_runs()
    features = build_features(runs)

    horizons = [7, 14, 21, 30]
    force_scores = "--force" in sys.argv
    score_cache: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        score_path = OUT / f"scores_h{horizon}d.csv"
        if score_path.exists() and not force_scores:
            scores = pd.read_csv(score_path, parse_dates=["open_time"]).set_index("open_time")
            print(f"loaded cached horizon={horizon}d rows={len(scores)}")
        else:
            scores = score_policy_values(runs, features, horizon)
            scores.to_csv(score_path)
            print(f"scored horizon={horizon}d rows={len(scores)}")
        score_cache[horizon] = scores

    configs: list[HysteresisConfig] = []
    for horizon in horizons:
        for persist_m in [3, 5, 7, 10, 14]:
            for frac in [0.6, 0.7, 0.8]:
                persist_n = int(math.ceil(persist_m * frac))
                for min_hold in [3, 7, 14]:
                    for allow_flat in [True, False]:
                        min_edges = [0.0, 25.0, 50.0] if allow_flat else [0.0]
                        for min_edge in min_edges:
                            for switch_margin in [0.0, 25.0, 50.0]:
                                configs.append(
                                    HysteresisConfig(
                                        horizon_days=horizon,
                                        persist_m=persist_m,
                                        persist_n=persist_n,
                                        min_hold_days=min_hold,
                                        min_edge_bps=min_edge,
                                        switch_margin_bps=switch_margin,
                                        allow_flat=allow_flat,
                                    )
                                )

    rows = []
    paths: dict[str, pd.DataFrame] = {}
    default_cfg = HysteresisConfig(14, 7, 5, 7, 25.0, 25.0, True)
    default_key = None
    for cfg in configs:
        path, stats = replay_hysteresis(runs, score_cache[cfg.horizon_days], cfg)
        stats.update(fixed_stats(runs, path.index.min()))
        rows.append(stats)
        if cfg == default_cfg:
            default_key = json.dumps(asdict(cfg), sort_keys=True)
            paths[default_key] = path

    res = pd.DataFrame(rows)
    res["delta_vs_best_fixed_total"] = res["total"] - res[
        [f"fixed_{p}_total" for p in POLICIES]
    ].max(axis=1)
    res["delta_vs_bd12_rvol_total"] = res["total"] - res["fixed_bd12_rvol_total"]
    res = res.sort_values(["total", "sharpe"], ascending=False).reset_index(drop=True)
    res.to_csv(OUT / "results_grid.csv", index=False)

    best_cfg = HysteresisConfig(
        int(res.loc[0, "horizon_days"]),
        int(res.loc[0, "persist_m"]),
        int(res.loc[0, "persist_n"]),
        int(res.loc[0, "min_hold_days"]),
        float(res.loc[0, "min_edge_bps"]),
        float(res.loc[0, "switch_margin_bps"]),
        bool(res.loc[0, "allow_flat"]),
    )
    best_path, _ = replay_hysteresis(runs, score_cache[best_cfg.horizon_days], best_cfg)
    best_path.to_csv(OUT / "path_best.csv")
    per_year(best_path, runs).to_csv(OUT / "per_year_best.csv", index=False)
    if default_key is None:
        default_path, _ = replay_hysteresis(runs, score_cache[default_cfg.horizon_days], default_cfg)
    else:
        default_path = paths[default_key]
    default_path.to_csv(OUT / "path_default.csv")
    per_year(default_path, runs).to_csv(OUT / "per_year_default.csv", index=False)

    summary = {
        "note": "Approximate stitched full-replay paths; screen only, not true stateful meta replay.",
        "policy_tags": POLICY_TAGS,
        "default_config": asdict(default_cfg),
        "best_config": asdict(best_cfg),
        "output": str(OUT),
    }
    (OUT / "policy_hysteresis_probe.json").write_text(json.dumps(summary, indent=2))

    keep = [
        "horizon_days", "persist_m", "persist_n", "min_hold_days",
        "min_edge_bps", "switch_margin_bps", "allow_flat", "total", "maxdd", "sharpe",
        "delta_vs_best_fixed_total", "delta_vs_bd12_rvol_total", "switches",
        "fixed_return1d_total", "fixed_rvol_total", "fixed_bd12_rvol_total",
        "fixed_cause66_total", "fixed_flat_total",
        "choose_return1d", "choose_rvol", "choose_bd12_rvol",
        "choose_cause66", "choose_flat",
    ]
    print("\n=== top 12 hysteresis configs ===")
    print(res[keep].head(12).to_string(index=False))

    default_row = res[
        (res["horizon_days"] == default_cfg.horizon_days)
        & (res["persist_m"] == default_cfg.persist_m)
        & (res["persist_n"] == default_cfg.persist_n)
        & (res["min_hold_days"] == default_cfg.min_hold_days)
        & (res["min_edge_bps"] == default_cfg.min_edge_bps)
        & (res["switch_margin_bps"] == default_cfg.switch_margin_bps)
    ]
    print("\n=== default config ===")
    print(default_row[keep].to_string(index=False))

    print("\n=== best per-year ===")
    print(pd.read_csv(OUT / "per_year_best.csv").to_string(index=False))

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
