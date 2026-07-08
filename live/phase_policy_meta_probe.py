"""Policy-meta probe for convexity v3.

Research-only sidecar. It builds point-in-time counterfactual labels for a
small policy menu, then tests whether a simple expanding policy-value model can
choose among those policies better than fixed rules.

This intentionally does not edit the replay engine. The label is raw new-sleeve
4h net bps, so it is a screening test before any full stateful replay work.
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

OUT = REPO / "live/state/policy_meta_probe"
OUT.mkdir(parents=True, exist_ok=True)

PREDS = REPO / "live/state/v3loop/fullhist_mpit/base.parquet"
PREDS_LONG = REPO / "live/state/v3loop/fullhist_mpit/long.parquet"
PANEL = REPO / "outputs/vBTC_features/panel_expanded_v0.parquet"
BTC_FUNDING = REPO / "data/ml/cache/funding_BTCUSDT.parquet"

K_LONG = 1
K_SHORT = 2
COST_ALT_BPS = 9.0
COST_BTC_BPS = 2.0
BTC_LONG_MULT = 0.25
FUND_CYCLE_FRAC = 0.5
ANN = np.sqrt(365)

POLICIES = ["pred_ls", "pred_btc", "return1d_btc", "rvol_btc"]


def dsh(s: pd.Series) -> float:
    d = (s.fillna(0.0) / 1e4).resample("1D").sum()
    return float(d.mean() / d.std() * ANN) if d.std() > 0 else np.nan


def maxdd(s: pd.Series) -> float:
    eq = s.fillna(0.0).cumsum()
    return float((eq - eq.cummax()).min())


def load_btc_funding() -> pd.Series:
    if not BTC_FUNDING.exists():
        return pd.Series(dtype=float)
    f = pd.read_parquet(BTC_FUNDING)
    f["calc_time"] = pd.to_datetime(f["calc_time"], utc=True)
    return f.set_index("calc_time")["funding_rate"].sort_index()


def asof_val(s: pd.Series, t: pd.Timestamp, default: float = 0.0) -> float:
    if s.empty:
        return default
    v = s.asof(t)
    return float(v) if pd.notna(v) else default


def inv_sqrt_weights(syms: list[str], g: pd.DataFrame, gross: float,
                     feat: str = "idio_vol_to_btc_1h") -> dict[str, float]:
    if not syms:
        return {}
    if feat not in g.columns:
        return {s: gross / len(syms) for s in syms}
    fmap = g.set_index("symbol")[feat].to_dict()
    vals = np.asarray([fmap.get(s, np.nan) for s in syms], dtype=float)
    good = vals[np.isfinite(vals) & (vals > 0)]
    med = float(np.median(good)) if len(good) else 1.0
    vals = np.asarray([v if np.isfinite(v) and v > 0 else med for v in vals], dtype=float)
    raw = 1.0 / np.sqrt(vals)
    raw = raw / (raw.sum() if raw.sum() else 1.0)
    return {s: float(gross * w) for s, w in zip(syms, raw)}


def policy_weights(g: pd.DataFrame, policy: str) -> dict[str, float]:
    gg = g.dropna(subset=["pred", "return_pct"]).copy()
    if len(gg) < K_LONG + K_SHORT:
        return {}

    if policy == "pred_ls":
        lkey = "pred_long" if "pred_long" in gg.columns and gg["pred_long"].notna().sum() >= K_LONG else "pred"
        skey = "pred_short" if "pred_short" in gg.columns and gg["pred_short"].notna().sum() >= K_SHORT else "pred"
        L = gg.dropna(subset=[lkey]).nlargest(K_LONG, lkey)["symbol"].tolist()
        S = gg.dropna(subset=[skey]).nsmallest(K_SHORT, skey)["symbol"].tolist()
        w = {}
        for s, v in inv_sqrt_weights(L, gg, 1.0).items():
            w[s] = w.get(s, 0.0) + v
        for s, v in inv_sqrt_weights(S, gg, 1.0).items():
            w[s] = w.get(s, 0.0) - v
        return w

    if policy == "pred_btc":
        S = gg.nsmallest(K_SHORT, "pred")["symbol"].tolist()
    elif policy == "return1d_btc":
        if "return_1d" not in gg.columns:
            return {}
        S = gg.dropna(subset=["return_1d"]).nlargest(K_SHORT, "return_1d")["symbol"].tolist()
    elif policy == "rvol_btc":
        if "rvol_7d" not in gg.columns:
            return {}
        S = gg.dropna(subset=["rvol_7d"]).nlargest(K_SHORT, "rvol_7d")["symbol"].tolist()
    else:
        raise ValueError(policy)

    if len(S) < K_SHORT:
        return {}
    w = {"_BTC_HEDGE_": BTC_LONG_MULT}
    for s, v in inv_sqrt_weights(S, gg, 1.0).items():
        w[s] = w.get(s, 0.0) - v
    return w


def net_bps_for_weights(g: pd.DataFrame, w: dict[str, float], btc_ret: float,
                        btc_funding: float) -> float:
    if not w:
        return 0.0
    rmap = dict(zip(g["symbol"], g["return_pct"]))
    fmap = dict(zip(g["symbol"], g.get("funding_rate", pd.Series(0.0, index=g.index))))
    gross = 0.0
    cost = 0.0
    fund_unit = 0.0
    for s, wt in w.items():
        if s == "_BTC_HEDGE_":
            ret = btc_ret
            fund = btc_funding
            c = COST_BTC_BPS
        else:
            ret = rmap.get(s, np.nan)
            fund = fmap.get(s, 0.0)
            c = COST_ALT_BPS
        if np.isfinite(ret):
            gross += wt * float(ret)
        cost += abs(wt) * c
        if np.isfinite(fund):
            fund_unit += wt * float(fund)
    return gross * 1e4 - cost - FUND_CYCLE_FRAC * fund_unit * 1e4


def add_policy_basket_features(row: dict, g: pd.DataFrame, policy: str, w: dict[str, float]) -> None:
    shorts = [s for s, wt in w.items() if wt < 0 and s != "_BTC_HEDGE_"]
    longs = [s for s, wt in w.items() if wt > 0 and s != "_BTC_HEDGE_"]
    idx = g.set_index("symbol")
    for side, syms in [("long", longs), ("short", shorts)]:
        sub = idx.loc[[s for s in syms if s in idx.index]] if syms else pd.DataFrame()
        row[f"{policy}_{side}_n"] = float(len(sub))
        for col in ["pred", "return_1d", "rvol_7d", "ret_3d", "corr_to_btc_1d",
                    "idio_vol_to_btc_1h", "funding_rate"]:
            if col in sub.columns and len(sub):
                row[f"{policy}_{side}_{col}_mean"] = float(sub[col].mean())
                row[f"{policy}_{side}_{col}_max"] = float(sub[col].max())
            else:
                row[f"{policy}_{side}_{col}_mean"] = np.nan
                row[f"{policy}_{side}_{col}_max"] = np.nan


def btc_feature_frame(times: pd.Index) -> pd.DataFrame:
    btc = load_close_4h("BTCUSDT")
    ret = btc.pct_change()
    f = pd.DataFrame(index=btc.index)
    for name, n in [("btc_ret_1d", 6), ("btc_ret_3d", 18), ("btc_ret_7d", 42),
                    ("btc_ret_30d", 180)]:
        f[name] = btc / btc.shift(n) - 1.0
    f["btc_rvol_7d"] = ret.rolling(42).std() * np.sqrt(42)
    f["btc_rvol_30d"] = ret.rolling(180).std() * np.sqrt(180)
    f["btc_smooth_30d"] = f["btc_ret_30d"].abs() / (f["btc_rvol_30d"] + 1e-12)
    f["btc_accel_7v30"] = f["btc_ret_7d"] - f["btc_ret_30d"] / 4.0
    f["btc_fwd_4h"] = btc.pct_change().shift(-1)
    return f.reindex(times, method="ffill")


def regime_for_btc30(x: float) -> str:
    if not np.isfinite(x):
        return "unknown"
    if x > 0.10:
        return "bull"
    if x < -0.10:
        return "bear"
    return "side"


def build_labels(force: bool = False) -> pd.DataFrame:
    out = OUT / "policy_labels.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)

    cols = ["symbol", "open_time", "return_pct", "pred"]
    have = set(pd.read_parquet(PREDS, columns=[]).columns) if False else set()
    d = pd.read_parquet(PREDS, columns=cols)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    dl = pd.read_parquet(PREDS_LONG, columns=["symbol", "open_time", "pred"]).rename(columns={"pred": "pred_long"})
    dl["open_time"] = pd.to_datetime(dl["open_time"], utc=True)
    d = d.merge(dl, on=["symbol", "open_time"], how="left")

    pcols = ["symbol", "open_time", "return_1d", "rvol_7d", "ret_3d", "corr_to_btc_1d",
             "idio_vol_to_btc_1h", "funding_rate", "atr_pct"]
    p = pd.read_parquet(PANEL, columns=pcols)
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    d = d.merge(p, on=["symbol", "open_time"], how="left")
    d = d.sort_values(["open_time", "symbol"])

    btc_feat = btc_feature_frame(pd.Index(sorted(d["open_time"].unique())))
    btc_fund = load_btc_funding()
    by_t = {t: g for t, g in d.groupby("open_time", sort=True)}
    rows = []
    for i, (t, g) in enumerate(by_t.items()):
        bf = btc_feat.loc[t]
        btc_fwd = float(bf.get("btc_fwd_4h", 0.0)) if pd.notna(bf.get("btc_fwd_4h", np.nan)) else 0.0
        row = {
            "open_time": t,
            "regime": regime_for_btc30(float(bf.get("btc_ret_30d", np.nan))),
            "n_pred": int(len(g)),
            "pred_disp": float(g["pred"].std()),
            "ret1d_disp": float(g["return_1d"].std()),
            "rvol7d_mean": float(g["rvol_7d"].mean()),
            "rvol7d_disp": float(g["rvol_7d"].std()),
            "corr_mean": float(g["corr_to_btc_1d"].mean()),
            "corr_disp": float(g["corr_to_btc_1d"].std()),
            "corr_low_frac": float((g["corr_to_btc_1d"] < 0.35).mean()),
            "fund_mean": float(g["funding_rate"].mean()),
            "fund_disp": float(g["funding_rate"].std()),
        }
        for col in btc_feat.columns:
            row[col] = float(bf[col]) if pd.notna(bf[col]) else np.nan
        bfnd = asof_val(btc_fund, t)
        for policy in POLICIES:
            w = policy_weights(g, policy)
            row[f"net_{policy}"] = net_bps_for_weights(g, w, btc_fwd, bfnd)
            row[f"gross_{policy}"] = float(sum(abs(v) for v in w.values()))
            add_policy_basket_features(row, g, policy, w)
        rows.append(row)
        if i and i % 2000 == 0:
            print(f"  labels {i}/{len(by_t)}", flush=True)

    lab = pd.DataFrame(rows).sort_values("open_time")
    lab.to_parquet(out, index=False)
    lab.to_csv(OUT / "policy_labels.csv", index=False)
    return lab


def fit_predict_ridge(train: pd.DataFrame, test: pd.DataFrame, ycol: str,
                      features: list[str], lam: float) -> np.ndarray:
    X = train[features].astype(float).values
    y = train[ycol].astype(float).values
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd < 1e-12] = 1.0
    X = np.where(np.isfinite(X), X, mu)
    Xs = (X - mu) / sd
    Xs = np.c_[np.ones(len(Xs)), Xs]
    reg = np.eye(Xs.shape[1]) * lam
    reg[0, 0] = 0.0
    beta = np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ y)
    Xt = test[features].astype(float).values
    Xt = np.where(np.isfinite(Xt), Xt, mu)
    Xts = (Xt - mu) / sd
    Xts = np.c_[np.ones(len(Xts)), Xts]
    return Xts @ beta


def evaluate(lab: pd.DataFrame) -> None:
    lab = lab.copy()
    lab["open_time"] = pd.to_datetime(lab["open_time"], utc=True)
    lab = lab.set_index("open_time").sort_index()

    policy_cols = [f"net_{p}" for p in POLICIES]
    base_features = [
        "btc_ret_1d", "btc_ret_3d", "btc_ret_7d", "btc_ret_30d",
        "btc_rvol_7d", "btc_rvol_30d", "btc_smooth_30d", "btc_accel_7v30",
        "pred_disp", "ret1d_disp", "rvol7d_mean", "rvol7d_disp",
        "corr_mean", "corr_disp", "corr_low_frac", "fund_mean", "fund_disp", "n_pred",
    ]
    basket_features = [c for c in lab.columns if any(c.startswith(f"{p}_short_") for p in POLICIES)]
    features = [c for c in base_features + basket_features if c in lab.columns]
    model_df = lab.dropna(subset=policy_cols).copy()

    rows = []
    for year in [2023, 2024, 2025, 2026]:
        train = model_df[model_df.index < pd.Timestamp(f"{year}-01-01", tz="UTC")]
        test = model_df[(model_df.index >= pd.Timestamp(f"{year}-01-01", tz="UTC")) &
                        (model_df.index < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))]
        if len(train) < 500 or test.empty:
            continue
        preds = {}
        for p in POLICIES:
            preds[p] = fit_predict_ridge(train, test, f"net_{p}", features, lam=50.0)
        pred_mat = np.vstack([preds[p] for p in POLICIES]).T
        names = np.asarray(POLICIES)
        actual_mat = test[policy_cols].to_numpy(float)
        for margin in [0.0, 1.0, 2.0, 5.0]:
            best_i = pred_mat.argmax(axis=1)
            best_v = pred_mat.max(axis=1)
            chosen = np.where(best_v > margin, best_i, -1)
            pnl = np.zeros(len(test))
            for i in range(len(POLICIES)):
                m = chosen == i
                pnl[m] = actual_mat[m, i]
            rec = {
                "year": year,
                "margin": margin,
                "selector_total": float(pnl.sum()),
                "selector_sharpe": dsh(pd.Series(pnl, index=test.index)),
                "selector_maxdd": maxdd(pd.Series(pnl, index=test.index)),
                "flat_rate": float((chosen < 0).mean()),
            }
            for i, p in enumerate(POLICIES):
                rec[f"choose_{p}"] = float((chosen == i).mean())
                rec[f"fixed_{p}"] = float(test[f"net_{p}"].sum())
            rec["oracle"] = float(np.maximum.reduce([actual_mat[:, i] for i in range(actual_mat.shape[1])] +
                                                    [np.zeros(len(test))]).sum())
            rows.append(rec)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "policy_selector_results.csv", index=False)
    print("\n=== Fixed raw-policy labels ===")
    for p in POLICIES:
        s = model_df[f"net_{p}"]
        print(f"{p:14s} total {s.sum():+9.0f}  sh {dsh(s):+5.2f}  maxDD {maxdd(s):+8.0f}")
    oracle = np.maximum.reduce([model_df[f"net_{p}"].to_numpy(float) for p in POLICIES] +
                               [np.zeros(len(model_df))])
    print(f"{'oracle':14s} total {oracle.sum():+9.0f}  sh {dsh(pd.Series(oracle, index=model_df.index)):+5.2f}"
          f"  maxDD {maxdd(pd.Series(oracle, index=model_df.index)):+8.0f}")

    print("\n=== Expanding policy-value selector ===")
    for margin, g in res.groupby("margin"):
        print(f"\nmargin={margin:g}")
        for _, r in g.iterrows():
            print(f"  {int(r['year'])}: selector {r['selector_total']:+8.0f}"
                  f"  oracle {r['oracle']:+8.0f}"
                  f"  flat {r['flat_rate']*100:5.1f}%"
                  f"  choose pred/rvol/ret {r['choose_pred_ls']*100:4.0f}/"
                  f"{r['choose_rvol_btc']*100:4.0f}/{r['choose_return1d_btc']*100:4.0f}")
        print(f"  TOTAL selector {g['selector_total'].sum():+8.0f}"
              f"  oracle {g['oracle'].sum():+8.0f}")

    meta = {
        "rows": int(len(model_df)),
        "start": str(model_df.index.min()),
        "end": str(model_df.index.max()),
        "policies": POLICIES,
        "features": features,
    }
    (OUT / "policy_meta_probe.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    force = "--force" in sys.argv
    lab = build_labels(force=force)
    evaluate(lab)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
