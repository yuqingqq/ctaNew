"""Pred-momentum entry gate — quick 2-fold sanity test.

Mechanistically distinct from retention/hysteresis (which holds stale names
after they leave top-K). This gate filters NEW entries: a name not currently
held may only enter if its pred was in top-K (or top-band_mult*K) of past
M-1 cycles. Held names auto-keep on sharp boundary (same as baseline).

When persistence rejects a new entry, K shrinks for that cycle (per-name
weight = 1/K_target; leg gross = K_actual / K_target). No fallback into
non-persistent names.

Hypothesis: blip predictions (one-cycle top-K) are noise; persistent
predictions are signal. Filtering blips at entry should improve realized
edge per name. Risk: filters real alpha jumps (rare new top names with
genuine breakout).

If any variant shows positive Δnet at the K=7 baseline, run full multi-OOS.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware
from ml.research.alpha_v8_h48_audit import build_wide_panel

HORIZON = 48
TOP_K = 7
TOP_FRAC = TOP_K / 25.0
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
N_FOLDS = 2
OUT_DIR = REPO / "outputs/pred_momentum"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (label, M_cycles, band_mult)
VARIANTS = [
    ("PM_M2_b1", 2, 1.0),  # strict: top-K at t-1
    ("PM_M2_b2", 2, 2.0),  # loose: top-2K at t-1
    ("PM_M3_b1", 3, 1.0),  # strict 3-cycle
    ("PM_M3_b2", 3, 2.0),  # loose 3-cycle
]


def _beta_neutral_scale(beta_L: float, beta_S: float):
    if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
        return 1.0, 1.0, True
    denom = beta_L + beta_S
    sL = float(np.clip(2.0 * beta_S / denom, 0.5, 1.5))
    sS = float(np.clip(2.0 * beta_L / denom, 0.5, 1.5))
    return sL, sS, False


def portfolio_pnl_pred_momentum_bn(
    test: pd.DataFrame, yt: np.ndarray, *, top_k: int, M_cycles: int,
    band_mult: float, cost_bps_per_leg: float, sample_every: int,
) -> dict:
    """Pred-momentum entry gate. NEW entries require persistence in past
    M-1 cycles within the top-(band_mult*K) band. Held names auto-keep on
    sharp boundary. Variable K downward (per-name weight = 1/top_k)."""
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk"]
    df = test[cols].copy()
    df["pred"] = yt
    times = sorted(df["open_time"].unique())
    if not times:
        return {"n_bars": 0}
    if sample_every > 1:
        keep_times = set(times[::sample_every])
        df = df[df["open_time"].isin(keep_times)]

    band_k = max(top_k, int(round(band_mult * top_k)))
    history: list[dict] = []  # ring buffer of past top/bot-band_k sets

    bars = []
    cur_long: set = set()
    cur_short: set = set()
    prev_long_w: dict[str, float] = {}
    prev_short_w: dict[str, float] = {}

    for t, g in df.groupby("open_time"):
        n = len(g)
        if n < 2 * top_k:
            continue
        sym_arr = g["symbol"].to_numpy()
        pred_arr = g["pred"].to_numpy()

        idx_top = np.argpartition(-pred_arr, top_k - 1)[:top_k]
        idx_bot = np.argpartition(pred_arr, top_k - 1)[:top_k]
        cand_long = set(sym_arr[idx_top])
        cand_short = set(sym_arr[idx_bot])

        # Held names that remain in current top/bot-K auto-stay
        new_long = cur_long & cand_long
        new_short = cur_short & cand_short

        # New entries: must have appeared in past (M-1) cycles' band-K
        if len(history) >= M_cycles - 1 and M_cycles >= 2:
            past_long = [h["long"] for h in history[-(M_cycles - 1):]]
            past_short = [h["short"] for h in history[-(M_cycles - 1):]]
            for s in cand_long - cur_long:
                if all(s in past for past in past_long):
                    new_long.add(s)
            for s in cand_short - cur_short:
                if all(s in past for past in past_short):
                    new_short.add(s)
        else:
            new_long |= cand_long
            new_short |= cand_short

        # Limit to top_k each (in case both cur_long ∩ cand_long and persistent give > K)
        if len(new_long) > top_k:
            ranked = sorted(new_long, key=lambda s: -pred_arr[sym_arr == s][0])[:top_k]
            new_long = set(ranked)
        if len(new_short) > top_k:
            ranked = sorted(new_short, key=lambda s: pred_arr[sym_arr == s][0])[:top_k]
            new_short = set(ranked)

        # Update history with this cycle's band-K sets
        bk = min(band_k, n)
        idx_top_band = np.argpartition(-pred_arr, bk - 1)[:bk] if bk < n else np.arange(n)
        idx_bot_band = np.argpartition(pred_arr, bk - 1)[:bk] if bk < n else np.arange(n)
        history.append({
            "long": set(sym_arr[idx_top_band]),
            "short": set(sym_arr[idx_bot_band]),
        })
        if len(history) > M_cycles:
            history = history[-M_cycles:]

        if not new_long or not new_short:
            cur_long, cur_short = new_long, new_short
            prev_long_w = {s: 1.0 / top_k for s in new_long}
            prev_short_w = {s: 1.0 / top_k for s in new_short}
            continue

        long_g = g[g["symbol"].isin(new_long)]
        short_g = g[g["symbol"].isin(new_short)]
        scale_L, scale_S, degen = _beta_neutral_scale(
            long_g["beta_short_vs_bk"].mean(), short_g["beta_short_vs_bk"].mean()
        )

        # Per-name weight = 1/top_k (constant). Leg gross = (K_actual / K_target) * scale.
        long_w = {s: scale_L / top_k for s in new_long}
        short_w = {s: scale_S / top_k for s in new_short}
        gross_L = scale_L * len(new_long) / top_k
        gross_S = scale_S * len(new_short) / top_k

        long_ret = gross_L * long_g["return_pct"].mean()
        short_ret = gross_S * short_g["return_pct"].mean()
        long_alpha = gross_L * long_g["alpha_realized"].mean()
        short_alpha = gross_S * short_g["alpha_realized"].mean()
        spread_ret = long_ret - short_ret
        spread_alpha = long_alpha - short_alpha

        if not prev_long_w:
            long_to = sum(long_w.values())
            short_to = sum(short_w.values())
        else:
            all_l = set(long_w) | set(prev_long_w)
            long_to = sum(abs(long_w.get(s, 0) - prev_long_w.get(s, 0)) for s in all_l)
            all_s = set(short_w) | set(prev_short_w)
            short_to = sum(abs(short_w.get(s, 0) - prev_short_w.get(s, 0)) for s in all_s)
        cost_bps = cost_bps_per_leg * (long_to + short_to)

        bars.append({
            "time": t, "n_long": len(new_long), "n_short": len(new_short),
            "gross_L": gross_L, "gross_S": gross_S,
            "spread_ret_bps": spread_ret * 1e4,
            "spread_alpha_bps": spread_alpha * 1e4,
            "long_turnover": long_to, "short_turnover": short_to,
            "cost_bps": cost_bps, "net_bps": spread_ret * 1e4 - cost_bps,
            "degen_beta": int(degen),
        })
        cur_long, cur_short = new_long, new_short
        prev_long_w, prev_short_w = long_w, short_w

    bdf = pd.DataFrame(bars)
    if bdf.empty:
        return {"n_bars": 0}
    return {
        "n_bars": len(bdf),
        "spread_ret_bps_mean": float(bdf["spread_ret_bps"].mean()),
        "cost_bps_mean": float(bdf["cost_bps"].mean()),
        "net_bps_mean": float(bdf["net_bps"].mean()),
        "long_turnover_mean": float(bdf["long_turnover"].mean()),
        "short_turnover_mean": float(bdf["short_turnover"].mean()),
        "n_long_mean": float(bdf["n_long"].mean()),
        "n_short_mean": float(bdf["n_short"].mean()),
        "gross_avg": float((bdf["gross_L"] + bdf["gross_S"]).mean() / 2),
        "df": bdf,
    }


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    print(f"Total folds: {len(folds)}; running first {N_FOLDS}")

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_feats = [c for c in v6_clean if c in panel.columns]

    rows = []
    for fold in folds[:N_FOLDS]:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            print(f"  fold {fold['fid']}: skipped"); continue
        Xt = tr[avail_feats].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail_feats].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=s) for s in ENSEMBLE_SEEDS]
        Xtest = test[avail_feats].to_numpy(dtype=np.float32)
        pred_test = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                              for m in models], axis=0)

        results = {}
        results["baseline"] = portfolio_pnl_turnover_aware(
            test, pred_test, top_frac=TOP_FRAC,
            cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON, beta_neutral=True,
        )
        # M=1 sanity (gate inactive — should reproduce baseline modulo weight scheme)
        results["PM_M1_sanity"] = portfolio_pnl_pred_momentum_bn(
            test, pred_test, top_k=TOP_K, M_cycles=1, band_mult=1.0,
            cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON,
        )
        for label, M, b in VARIANTS:
            results[label] = portfolio_pnl_pred_momentum_bn(
                test, pred_test, top_k=TOP_K, M_cycles=M, band_mult=b,
                cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON,
            )

        for label, r in results.items():
            if r.get("n_bars", 0) == 0:
                continue
            df = r["df"]
            sh = df["net_bps"].mean() / df["net_bps"].std() * np.sqrt(CYCLES_PER_YEAR) if df["net_bps"].std() > 0 else 0
            row = {
                "fold": fold["fid"], "variant": label, "n": r["n_bars"],
                "net": df["net_bps"].mean(),
                "spread": df["spread_ret_bps"].mean(),
                "cost": df["cost_bps"].mean(),
                "L_to": df["long_turnover"].mean(),
                "S_to": df["short_turnover"].mean(),
                "sh_fold": float(sh),
            }
            if "n_long" in df.columns:
                row["K_avg"] = (df["n_long"].mean() + df["n_short"].mean()) / 2
                row["gross"] = (df["gross_L"].mean() + df["gross_S"].mean()) / 2
            rows.append(row)
        print(f"  fold {fold['fid']}: total {time.time()-t0:.0f}s")

    summary_df = pd.DataFrame(rows)
    by_var = summary_df.groupby("variant").agg(
        net=("net", "mean"), spread=("spread", "mean"), cost=("cost", "mean"),
        L_to=("L_to", "mean"), S_to=("S_to", "mean"),
        sh_avg=("sh_fold", "mean"),
        K_avg=("K_avg", "mean") if "K_avg" in summary_df.columns else ("net", "mean"),
        gross=("gross", "mean") if "gross" in summary_df.columns else ("net", "mean"),
    ).reset_index()
    order = ["baseline", "PM_M1_sanity"] + [v[0] for v in VARIANTS]
    by_var["sort"] = by_var["variant"].apply(lambda v: order.index(v) if v in order else 999)
    by_var = by_var.sort_values("sort").drop(columns=["sort"]).reset_index(drop=True)

    print("\n" + "=" * 100)
    print(f"PRED-MOMENTUM ENTRY GATE  (h={HORIZON}, K={TOP_K}, {N_FOLDS} folds, 4.5 bps/leg, β-neutral)")
    print("=" * 100)
    print(by_var.to_string(index=False, float_format="%+.2f"))

    base_row = by_var[by_var["variant"] == "baseline"].iloc[0]
    print("\nΔ vs baseline:")
    for _, r in by_var.iterrows():
        if r["variant"] == "baseline":
            continue
        print(f"  {r['variant']:<14}  Δnet={r['net']-base_row['net']:+.2f} bps   "
              f"Δcost={r['cost']-base_row['cost']:+.2f}   Δsh={r['sh_avg']-base_row['sh_avg']:+.2f}")

    summary_df.to_csv(OUT_DIR / "fold_rows.csv", index=False)
    by_var.to_csv(OUT_DIR / "by_variant.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
