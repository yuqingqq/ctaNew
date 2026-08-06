"""Decompose the live strategy's L/S P&L into ALPHA (beta-hedged) vs BETA (market-exposure) — both eras.
Answers: what do we actually farm stably? Is the leftover beta a farmed edge or a non-stationary side-risk?

Strategy L/S on RAW forward returns (return_pct), per-symbol Ridge predictions. Per bar:
  ls_raw = mean(return_pct | top-20% pred) − mean(return_pct | bottom-20% pred)   [realized L/S]
  mkt    = mean(return_pct | all)                                                  [market]
Regress ls_raw ~ mkt per era: slope = net BETA, intercept = beta-hedged ALPHA. Decompose contributions.
Run: python3 -u -m live.build_alpha_beta_decomp
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.train_v4_artifact import x6

EMB = pd.Timedelta(days=1); HL = 60.0
FULL = "outputs/vBTC_features/panel_expanded_v0_clean.parquet"


def gen_pred(PAN, feats, cuts):
    rec = []
    for i in range(len(cuts) - 1):
        c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
        tr = PAN[(PAN.exit_time < fc) & PAN["z_res"].notna()]
        te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)]
        if tr.empty or te.empty:
            continue
        t_end = tr["open_time"].max()
        for sym, gg in tr.groupby("symbol"):
            if len(gg) < 300:
                continue
            try:
                s, h = x6.fit_preproc(gg, feats); X = x6.apply_preproc(gg, feats, s, h)
                w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["z_res"].to_numpy(), sample_weight=w)
                gte = te[te.symbol == sym]
                if len(gte):
                    rec.append(pd.DataFrame({"symbol": sym, "open_time": gte["open_time"].values,
                                             "pred": m.predict(x6.apply_preproc(gte, feats, s, h))}))
            except Exception:
                pass
    return pd.concat(rec, ignore_index=True) if rec else pd.DataFrame()


def ls_series(df, k=0.2):
    rows = []
    for t, g in df.groupby("open_time"):
        if len(g) < 10:
            continue
        g = g.sort_values("pred"); nk = max(1, int(len(g) * k))
        short = g["return_pct"].iloc[:nk].mean()      # low pred = short
        long = g["return_pct"].iloc[-nk:].mean()       # high pred = long
        rows.append((t, long - short, g["return_pct"].mean()))
    return pd.DataFrame(rows, columns=["open_time", "ls", "mkt"]).set_index("open_time")


def main():
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    print("running per-symbol Ridge predictions (both eras)...", flush=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        pred = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna()
        d = ls_series(pred)
        x = d["mkt"].to_numpy(); y = d["ls"].to_numpy()
        beta = np.polyfit(x, y, 1)[0]
        alpha = (y - beta * x).mean()               # beta-hedged per-bar alpha
        beta_contrib = beta * x.mean()               # avg beta P&L contribution
        print(f"===== {era} =====", flush=True)
        print(f"    net BETA (L/S vs market)   {beta:+.3f}   (negative = net short market)", flush=True)
        print(f"    mean market return         {x.mean()*1e4:+.1f} bps/bar", flush=True)
        print(f"    total L/S return           {y.mean()*1e4:+.2f} bps/bar", flush=True)
        print(f"      = ALPHA (beta-hedged)    {alpha*1e4:+.2f} bps/bar   <- the stable edge?", flush=True)
        print(f"      + BETA contribution      {beta_contrib*1e4:+.2f} bps/bar   <- non-stationary?", flush=True)
        print("", flush=True)
    print("DECOMPDONE", flush=True)


if __name__ == "__main__":
    main()
