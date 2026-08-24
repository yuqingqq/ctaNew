"""CAREFUL incremental test using the STRATEGY'S REAL machinery (not a proxy): x6 preproc + V0_LEAN + per-symbol
RidgeCV + HL=60 exp-decay weighting + exit_time label-purge + 1d embargo, exactly as gen_residual_target.py. Adds
imb_ewma to the base price book and compares rank-IC(pred, alpha_A) vs the V0_LEAN baseline, BOTH eras, with a
day-clustered bootstrap CI on the paired delta. VALIDITY GATE: the V0_LEAN baseline must reproduce the real honest
rank-IC (~+0.030 recent / +0.024 OOS); only then is the delta trustworthy. Old bars lack imb_ewma -> filled 0
(neutral) and ~0 weight under HL=60, so the coef is learned from the covered recent bars.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
os.environ["V4_PANEL"] = str(REPO / "outputs/vBTC_features/panel_expanded_v0_clean.parquet")
import live.train_twobook_models as tt
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr
from live.bookdepth_persist import persist_feats
x6 = tt.x6; V0_LEAN = list(tt.V0_LEAN); EMB = pd.Timedelta(days=1); HL = 60.0
rng = np.random.default_rng(7)
RECENT_CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04", "2025-11-01", "2025-12-01", "2026-01-01",
              "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01", "2026-06-30"]]
OOS_CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2023-06-01", "2023-09-01", "2023-12-01", "2024-03-01",
            "2024-06-01", "2024-09-01", "2024-12-01", "2025-03-01", "2025-06-01", "2025-09-01"]]  # full OOS era (post re-fetch: 69-158 syms/quarter)

def build_panel():
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "alpha_vs_btc_realized"] + V0_LEAN)
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True); PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
    PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)]
    # merge imb persistence features
    rows = []
    import glob
    for f in [x for x in glob.glob(str(REPO / "data/ml/cache/l2_*.parquet")) if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]; d0 = pd.read_parquet(f)[["l2_imb1"]]
        d0.index = pd.to_datetime(d0.index, utc=True) + pd.Timedelta("4h")
        pf = persist_feats(d0["l2_imb1"].sort_index())[["imb_ewma", "imb_mean12", "imb_run"]]
        pf["symbol"] = sym; pf["open_time"] = pf.index; rows.append(pf.reset_index(drop=True))
    L = pd.concat(rows, ignore_index=True)
    PAN = PAN.merge(L, on=["symbol", "open_time"], how="left")
    PAN["_covered"] = PAN["imb_ewma"].notna()
    for c in ["imb_ewma", "imb_mean12", "imb_run"]: PAN[c] = PAN[c].fillna(0.0)   # neutral; old bars ~0 weight (HL)
    g = PAN.groupby("open_time")
    sd = g["alpha_vs_btc_realized"].transform("std").replace(0, np.nan)
    PAN["z_res"] = ((PAN["alpha_vs_btc_realized"] - g["alpha_vs_btc_realized"].transform("mean")) / sd).clip(-10, 10)
    return PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)

def gen(PAN, feats, cuts):
    rec = []
    for i in range(len(cuts) - 1):
        c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
        tr = PAN[(PAN.exit_time < fc) & PAN["z_res"].notna()]; te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)]
        t_end = tr["open_time"].max()
        for sym, gg in tr.groupby("symbol"):
            if len(gg) < 300: continue
            try:
                s, h = x6.fit_preproc(gg, feats); X = x6.apply_preproc(gg, feats, s, h)
                w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["z_res"].to_numpy(), sample_weight=w)
                gte = te[te.symbol == sym]
                if len(gte):
                    rec.append(pd.DataFrame({"open_time": gte["open_time"].values, "alpha_A": gte["alpha_vs_btc_realized"].values,
                                             "pred": m.predict(x6.apply_preproc(gte, feats, s, h)), "cov": gte["_covered"].values}))
            except Exception:
                pass
    return pd.concat(rec, ignore_index=True)

def perbar_ic(P):
    P = P[P["cov"]]  # evaluate only on covered test bars (fair like-for-like)
    return P.groupby("open_time").apply(lambda g: spearmanr(g["pred"], g["alpha_A"]).correlation if len(g) >= 5 else np.nan).dropna()

def main():
    PAN = build_panel()
    print(f"panel rows {len(PAN)} | covered {int(PAN['_covered'].sum())} | V0_LEAN={len(V0_LEAN)} feats\n")
    for era, cuts in [("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)]:
        base = gen(PAN, V0_LEAN, cuts); add = gen(PAN, V0_LEAN + ["imb_ewma"], cuts)
        add3 = gen(PAN, V0_LEAN + ["imb_ewma", "imb_mean12", "imb_run"], cuts)
        ib, ia, i3 = perbar_ic(base), perbar_ic(add), perbar_ic(add3)
        print(f"### {era} (real per-symbol RidgeCV pipeline; rank-IC on covered test bars) ###")
        print(f"  V0_LEAN (baseline)      rank-IC {ib.mean():+.4f}   [validity gate: ~+0.030 rec / +0.024 oos]")
        print(f"  V0_LEAN + imb_ewma      rank-IC {ia.mean():+.4f}")
        print(f"  V0_LEAN + ewma+mean+run rank-IC {i3.mean():+.4f}")
        for lab, iadd in [("imb_ewma", ia), ("ewma+mean+run", i3)]:
            j = pd.concat([ib.rename("a"), iadd.rename("b")], axis=1).dropna(); j["d"] = j["b"] - j["a"]
            j["day"] = pd.to_datetime(j.index, utc=True).floor("1D"); gg = [x["d"].values for _, x in j.groupby("day")]
            boot = [np.concatenate([gg[k] for k in rng.integers(0, len(gg), len(gg))]).mean() for _ in range(3000)]
            lo, up = np.percentile(boot, [2.5, 97.5])
            flag = "ADDS (CI>0)" if lo > 0 else ("HURTS (CI<0)" if up < 0 else "within noise")
            print(f"    Δ(+{lab:14s}) {j['d'].mean():+.4f} [{lo:+.4f},{up:+.4f}] -> {flag}")
        print()
    print("REALABLDONE")

if __name__ == "__main__":
    main()
