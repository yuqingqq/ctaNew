"""Train the DEPLOYABLE v4 artifact (live single-shot inference). Identical machinery to
train_v3_artifact.py (V0_LEAN features, per-symbol RidgeCV, HL=60 sample decay, fit_cut = panel-1d)
with ONE difference — the training TARGET is the RESIDUAL: xs_z(alpha_vs_btc_realized) instead of
xs_z(return_pct) (gen_residual_target.py's z_res; the v4 candidate of V4_PERFORMANCE §2).
Two books:
  base    : V0_LEAN            -> convexity_v4_base_model.pkl      (short ranker + baseline)
  residrev: V0_LEAN + resid_rev-> convexity_v4_residrev_model.pkl  (long ranker)
Usage: python3 live/train_v4_artifact.py [--fit-cut 2026-07-06]
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0_LEAN = tt.V0_LEAN; RR = tt.RR; EMB = pd.Timedelta(days=1); HL = 60.0
OUT = REPO / "live/models"; OUT.mkdir(parents=True, exist_ok=True)

def load_panel():
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "return_pct", "alpha_vs_btc_realized"] + list(tt.V0))
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True); PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
    PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)].sort_values(["symbol", "open_time"])
    a = PAN.groupby("symbol")["alpha_vs_btc_realized"]
    PAN["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
    PAN["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
    for c in RR: PAN[c] = PAN[c].fillna(0.0)
    # v4 target: cross-sectional z of the RESIDUAL (what the strategy farms)
    g = PAN.groupby("open_time"); sd = g["alpha_vs_btc_realized"].transform("std").replace(0, np.nan)
    PAN["xs_z"] = ((PAN["alpha_vs_btc_realized"] - g["alpha_vs_btc_realized"].transform("mean")) / sd).clip(-10, 10)
    return PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)

def fit_book(PAN, feats, fit_cut, name):
    tr = PAN[(PAN.exit_time < fit_cut) & PAN["xs_z"].notna()]
    t_end = tr["open_time"].max()
    models, sstats, hstats = {}, {}, {}
    for sym, gg in tr.groupby("symbol"):
        if len(gg) < 300: continue
        try:
            s, h = x6.fit_preproc(gg, feats)
            X = x6.apply_preproc(gg, feats, s, h)
            w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
            m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["xs_z"].to_numpy(), sample_weight=w)
            models[sym], sstats[sym], hstats[sym] = m, s, h
        except Exception:
            pass
    art = {"feat_cols": feats, "HL": HL, "models": models, "sstats": sstats, "hstats": hstats,
           "meta": {"fit_cut": str(fit_cut), "t_end": str(t_end), "n_train": len(tr), "syms": len(models),
                    "strategy": "convexity_v4", "book": name, "target": "xs_z(alpha_vs_btc_realized)"}}
    path = OUT / f"convexity_v4_{name}_model.pkl"
    pickle.dump(art, open(path, "wb"))
    print(f"  {name}: {len(models)} syms, {len(tr)} train rows, fit_cut {fit_cut.date()} -> {path.name}")
    return art

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fit-cut", default=None); a = ap.parse_args()
    PAN = load_panel()
    fit_cut = pd.Timestamp(a.fit_cut, tz="UTC") if a.fit_cut else (PAN["open_time"].max().normalize() - EMB)
    print(f"v4 artifact train: panel max {PAN['open_time'].max()}, fit_cut {fit_cut}")
    fit_book(PAN, V0_LEAN, fit_cut, "base")
    fit_book(PAN, V0_LEAN + RR, fit_cut, "residrev")
    print("DONE")

if __name__ == "__main__":
    main()
