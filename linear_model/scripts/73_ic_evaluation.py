"""Step 73: IC evaluation for the current linear-model structure.

This script measures signal quality before the trade engine:

- pooled-44 Step 62 predictions
- per-symbol-44 Step 67 predictions
- current Step 71 drop-BIO+VVV all-eligible PIT inputs

For each case and score column it reports:
  * per-cycle Spearman rank IC and Pearson IC vs alpha_beta
  * fold-level Spearman IC
  * per-symbol time-series Spearman IC
  * top/bottom basket alpha spread at the 4h decision cadence
  * decile monotonicity

The current trading rule ranks symbols cross-sectionally every 4h, so the main
metric is per-cycle rank IC on sampled decision cycles.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s64 = _imp("s64", "linear_model/scripts/64_meanrev_v2_backtest.py")
s68 = _imp("s68", "linear_model/scripts/68_persymbol_selfstd.py")
s71 = _imp("s71", "linear_model/scripts/71_battery_alleligible.py")

OUT = REPO / "linear_model/results/step73_ic_evaluation"
OUT.mkdir(parents=True, exist_ok=True)
OOS = set(s64.OOS)
BLOCK = s64.BLOCK


def sampled_decision_frame(apd: pd.DataFrame) -> pd.DataFrame:
    d = apd.copy()
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[d["fold"].isin(OOS)].copy()
    times = sorted(d["open_time"].unique())[::BLOCK]
    return d[d["open_time"].isin(set(times))].copy()


def _cycle_corr(g: pd.DataFrame, score: str, method: str) -> float:
    x = g[score]
    y = g["alpha_beta"]
    m = x.notna() & y.notna()
    if m.sum() < 5:
        return np.nan
    if x[m].std() <= 1e-12 or y[m].std() <= 1e-12:
        return np.nan
    return float(x[m].corr(y[m], method=method))


def cycle_ic(df: pd.DataFrame, score: str, method: str) -> pd.DataFrame:
    rows = []
    for t, g in df.groupby("open_time", sort=True):
        rows.append(
            {
                "open_time": t,
                "fold": int(g["fold"].iloc[0]),
                "ic": _cycle_corr(g, score, method),
                "n": int(g[[score, "alpha_beta"]].dropna().shape[0]),
            }
        )
    return pd.DataFrame(rows).dropna(subset=["ic"])


def top_bottom_spread(df: pd.DataFrame, score: str, k: int = 3) -> pd.DataFrame:
    rows = []
    for t, g in df.dropna(subset=[score, "alpha_beta"]).groupby("open_time", sort=True):
        if len(g) < 2 * k:
            continue
        gs = g.sort_values(score, ascending=False)
        top = gs.head(k)["alpha_beta"].mean() * 1e4
        bot = gs.tail(k)["alpha_beta"].mean() * 1e4
        rows.append(
            {
                "open_time": t,
                "fold": int(g["fold"].iloc[0]),
                "top_bps": float(top),
                "bottom_bps": float(bot),
                "spread_bps": float(top - bot),
                "ls_half_weight_bps": float(0.5 * top - 0.5 * bot),
            }
        )
    return pd.DataFrame(rows)


def deciles(df: pd.DataFrame, score: str) -> pd.DataFrame:
    rows = []
    for t, g in df.dropna(subset=[score, "alpha_beta"]).groupby("open_time", sort=True):
        if len(g) < 20:
            continue
        r = g[score].rank(method="first")
        try:
            bucket = pd.qcut(r, 10, labels=False, duplicates="drop")
        except ValueError:
            continue
        tmp = g.assign(decile=bucket)
        for dec, gg in tmp.groupby("decile"):
            rows.append(
                {
                    "open_time": t,
                    "fold": int(g["fold"].iloc[0]),
                    "decile": int(dec),
                    "alpha_bps": float(gg["alpha_beta"].mean() * 1e4),
                    "n": int(len(gg)),
                }
            )
    return pd.DataFrame(rows)


def symbol_ic(df: pd.DataFrame, score: str) -> pd.DataFrame:
    rows = []
    for sym, g in df.dropna(subset=[score, "alpha_beta"]).groupby("symbol", sort=True):
        if len(g) < 50 or g[score].std() <= 1e-12 or g["alpha_beta"].std() <= 1e-12:
            continue
        rows.append(
            {
                "symbol": sym,
                "n": int(len(g)),
                "spearman": float(g[score].corr(g["alpha_beta"], method="spearman")),
                "pearson": float(g[score].corr(g["alpha_beta"], method="pearson")),
            }
        )
    return pd.DataFrame(rows)


def summarize_case(name: str, apd: pd.DataFrame, scores: list[str]) -> list[dict]:
    df = sampled_decision_frame(apd)
    print(f"\n{name}: {len(df):,} rows, {df['symbol'].nunique()} symbols, "
          f"{df['open_time'].nunique()} decision cycles", flush=True)
    rows = []
    for score in scores:
        if score not in df.columns:
            continue
        valid = df[[score, "alpha_beta"]].dropna()
        if valid.empty:
            continue
        ci_s = cycle_ic(df, score, "spearman")
        ci_p = cycle_ic(df, score, "pearson")
        tb = top_bottom_spread(df, score, 3)
        dec = deciles(df, score)
        si = symbol_ic(df, score)
        dec_avg = dec.groupby("decile")["alpha_bps"].mean() if len(dec) else pd.Series(dtype=float)
        mono = float(dec_avg.corr(pd.Series(dec_avg.index, index=dec_avg.index), method="spearman")) if len(dec_avg) >= 3 else np.nan
        row = {
            "case": name,
            "score": score,
            "n_rows": int(len(valid)),
            "n_cycles": int(ci_s.shape[0]),
            "cycle_spearman_mean": float(ci_s["ic"].mean()) if len(ci_s) else np.nan,
            "cycle_spearman_median": float(ci_s["ic"].median()) if len(ci_s) else np.nan,
            "cycle_spearman_t": float(ci_s["ic"].mean() / (ci_s["ic"].std(ddof=1) / np.sqrt(len(ci_s)))) if len(ci_s) > 2 and ci_s["ic"].std(ddof=1) > 0 else np.nan,
            "cycle_pearson_mean": float(ci_p["ic"].mean()) if len(ci_p) else np.nan,
            "top3_mean_bps": float(tb["top_bps"].mean()) if len(tb) else np.nan,
            "bottom3_mean_bps": float(tb["bottom_bps"].mean()) if len(tb) else np.nan,
            "top_minus_bottom_bps": float(tb["spread_bps"].mean()) if len(tb) else np.nan,
            "ls_half_weight_bps": float(tb["ls_half_weight_bps"].mean()) if len(tb) else np.nan,
            "spread_positive_pct": float((tb["spread_bps"] > 0).mean()) if len(tb) else np.nan,
            "decile_monotonic_spearman": mono,
            "symbol_ic_mean": float(si["spearman"].mean()) if len(si) else np.nan,
            "symbol_ic_median": float(si["spearman"].median()) if len(si) else np.nan,
            "symbols_pos_ic_pct": float((si["spearman"] > 0).mean()) if len(si) else np.nan,
        }
        rows.append(row)
        print(
            f"  {score:14s} cycle rank IC mean={row['cycle_spearman_mean']:+.4f} "
            f"median={row['cycle_spearman_median']:+.4f} t={row['cycle_spearman_t']:+.2f} | "
            f"top-bottom={row['top_minus_bottom_bps']:+.2f} bps "
            f"(half-wt {row['ls_half_weight_bps']:+.2f}) pos={row['spread_positive_pct']*100:.0f}% | "
            f"symIC med={row['symbol_ic_median']:+.4f} pos={row['symbols_pos_ic_pct']*100:.0f}%",
            flush=True,
        )

        ci_s.to_csv(OUT / f"{name}_{score}_cycle_spearman.csv", index=False)
        tb.to_csv(OUT / f"{name}_{score}_top_bottom.csv", index=False)
        si.to_csv(OUT / f"{name}_{score}_symbol_ic.csv", index=False)
        if len(dec):
            dec.groupby("decile")["alpha_bps"].mean().reset_index().to_csv(
                OUT / f"{name}_{score}_deciles.csv", index=False
            )
    return rows


def main():
    print("=" * 92, flush=True)
    print("  STEP 73: IC evaluation", flush=True)
    print("=" * 92, flush=True)
    t0 = time.time()
    rows = []

    pooled = pd.read_parquet(REPO / "linear_model/results/step62_bluechip44/predictions.parquet")
    pooled["open_time"] = pd.to_datetime(pooled["open_time"], utc=True)
    rows += summarize_case("pooled44", pooled, ["pred_z", "pred_B"])

    persym = pd.read_parquet(REPO / "linear_model/results/step67_persymbol/persym_predictions.parquet")
    persym["open_time"] = pd.to_datetime(persym["open_time"], utc=True)
    persym = s68.add_self_z(persym)
    rows += summarize_case("persym44", persym, ["pred_z", "pred_z_self", "pred_B"])

    print("\nRebuilding current drop2 Step-71 inputs for IC...", flush=True)
    drop2 = s71.build(["BIOUSDT", "VVVUSDT"])[0]
    rows += summarize_case("drop2_current", drop2, ["pred_z", "pred_z_self", "pred_B"])

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "summary.csv", index=False)
    print(f"\nSaved summary: {OUT / 'summary.csv'}", flush=True)
    print(f"Total: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
