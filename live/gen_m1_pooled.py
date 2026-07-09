"""M1 model-class cell: pooled LGBM (variant) + pooled Ridge (diagnostic arm) vs per-symbol
Ridge incumbents (RESEARCH_LOOP_20260707 addenda 11 + 11b, PRE-REGISTERED).

Pins (11b): params = x6.LGB_PARAMS_POOLED verbatim, n_estimators 400 FIXED, no early stopping;
sample_weight = exp(-(t_end - open_time)/60d) with GLOBAL fold t_end (never dropped); LGBM arm
has NO preproc — raw features + sym_id pandas-categorical, native NaN; 5 seeds (42,7,123,99,314)
all four seed fields, deterministic, force_row_wise, num_threads=8, mean of RAW preds;
per-fold tripwires printed: sum-w/ESS, per-cycle pred std, n-unique preds, inter-seed rank-corr.
Pooled-Ridge arm (M1_ARM=pridge): global x6.fit_preproc REFIT PER FOLD + sym one-hots
drop_first; diagnostic-only, cannot be promoted. Target/panel construction = gen_winsor lineage
(xs_z clip +-10). No per-fold min-rows floor for pooled training (part of the treatment).
EXCL test-only. Books: hl_m1lgbm_* / hl_m1pridge_*.
Usage: M1_ARM=lgbm|pridge python3 live/gen_m1_pooled.py
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0_LEAN = list(tt.V0_LEAN); EMB = pd.Timedelta(days=1); HL = 60.0
RR = ["resid_rev_2", "resid_rev_3"]
EXCL = {"LITUSDT", "VINEUSDT", "PUMPUSDT"}
ARM = os.environ.get("M1_ARM", "lgbm")
SEEDS = (42, 7, 123, 99, 314)

def build_panel():
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "return_pct",
                                              "alpha_vs_btc_realized"] + V0_LEAN)
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True)
    PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
    PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)].sort_values(["symbol", "open_time"])
    a = PAN.groupby("symbol")["alpha_vs_btc_realized"]
    PAN["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
    PAN["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
    for c in RR: PAN[c] = PAN[c].fillna(0.0)
    g = PAN.groupby("open_time")
    sd = g["alpha_vs_btc_realized"].transform("std").replace(0, np.nan)
    PAN["xs_z"] = ((PAN["alpha_vs_btc_realized"] - g["alpha_vs_btc_realized"].transform("mean")) / sd).clip(-10, 10)
    PAN["sym_id"] = PAN["symbol"].astype("category")
    return PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)

def gen_lgbm(PAN, cuts, feats, outpath, tagn):
    import lightgbm as lgb
    rec = []
    for i in range(len(cuts) - 1):
        c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
        tr = PAN[(PAN.exit_time < fc) & PAN["xs_z"].notna()]
        te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1) & (~PAN.symbol.isin(EXCL))]
        if not len(tr) or not len(te): continue
        t_end = tr["open_time"].max()
        w = np.exp(-((t_end - tr["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
        ess = w.sum() ** 2 / (w * w).sum()
        cols = feats + ["sym_id"]
        Xtr = tr[cols]; ytr = tr["xs_z"].to_numpy(); Xte = te[cols]
        preds = []
        for sd_ in SEEDS:
            p = dict(x6.LGB_PARAMS_POOLED)
            p.update(seed=sd_, feature_fraction_seed=sd_, bagging_seed=sd_, data_random_seed=sd_,
                     deterministic=True, force_row_wise=True, num_threads=8)
            m = lgb.LGBMRegressor(**p)
            m.fit(Xtr, ytr, sample_weight=w, categorical_feature=["sym_id"])
            preds.append(m.predict(Xte))
        P = np.vstack(preds)
        pred = P.mean(axis=0)
        # tripwires (11b-7/8)
        ted = te.assign(_p=pred)
        pcs = ted.groupby("open_time")["_p"].std().mean()
        nun = ted.groupby("open_time")["_p"].nunique().median()
        sc = pd.DataFrame(P.T).corr(method="spearman").to_numpy()
        isc = sc[np.triu_indices_from(sc, 1)].mean()
        print(f"    {tagn} f{i} ({c0.date()}): {tr.symbol.nunique()} syms {len(tr)} rows  "
              f"sumW {w.sum():.0f} ESS {ess:.0f}  predstd/cyc {pcs:.4f} uniq/cyc {nun:.0f}  "
              f"interseed rho {isc:.3f}", flush=True)
        rec.append(pd.DataFrame({"symbol": te["symbol"].values, "open_time": te["open_time"].values,
                                 "alpha_A": te["alpha_vs_btc_realized"].values,
                                 "return_pct": te["return_pct"].values,
                                 "exit_time": te["exit_time"].values, "pred": pred, "fold": i}))
    out = pd.concat(rec, ignore_index=True)
    for c in ("open_time", "exit_time"): out[c] = pd.to_datetime(out[c], utc=True)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(outpath); print(f"  wrote {outpath}", flush=True)

def gen_pridge(PAN, cuts, feats, outpath, tagn):
    from sklearn.linear_model import RidgeCV
    rec = []
    SYMS = sorted(PAN["symbol"].unique())
    for i in range(len(cuts) - 1):
        c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
        tr = PAN[(PAN.exit_time < fc) & PAN["xs_z"].notna()]
        te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1) & (~PAN.symbol.isin(EXCL))]
        if not len(tr) or not len(te): continue
        t_end = tr["open_time"].max()
        w = np.exp(-((t_end - tr["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
        s, h = x6.fit_preproc(tr, feats)                      # GLOBAL preproc, refit per fold
        def mat(df):
            X = x6.apply_preproc(df, feats, s, h)
            oh = pd.get_dummies(pd.Categorical(df["symbol"], categories=SYMS), drop_first=True)
            return np.hstack([X, oh.to_numpy(dtype=np.float32)])
        m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(mat(tr), tr["xs_z"].to_numpy(), sample_weight=w)
        pred = m.predict(mat(te))
        print(f"    {tagn} f{i} ({c0.date()}): {len(tr)} rows alpha={m.alpha_:.2f}", flush=True)
        rec.append(pd.DataFrame({"symbol": te["symbol"].values, "open_time": te["open_time"].values,
                                 "alpha_A": te["alpha_vs_btc_realized"].values,
                                 "return_pct": te["return_pct"].values,
                                 "exit_time": te["exit_time"].values, "pred": pred, "fold": i}))
    out = pd.concat(rec, ignore_index=True)
    for c in ("open_time", "exit_time"): out[c] = pd.to_datetime(out[c], utc=True)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(outpath); print(f"  wrote {outpath}", flush=True)

def main():
    PAN = build_panel()
    last = PAN["open_time"].max().normalize() + pd.Timedelta(days=1)
    REC_CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04", "2025-11-01", "2025-12-01", "2026-01-01",
                "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-05-27"]] + [last]
    OOS_CUTS = list(pd.date_range("2023-01-01", "2025-10-01", freq="MS", tz="UTC"))
    D = REPO / "live/state/convexity"
    tag = "m1lgbm" if ARM == "lgbm" else "m1pridge"
    gen = gen_lgbm if ARM == "lgbm" else gen_pridge
    print(f"ARM={ARM} -> hl_{tag}_*", flush=True)
    print("recent base:", flush=True); gen(PAN, REC_CUTS, V0_LEAN, D / f"hl_{tag}_base/v0full_hl60.parquet", "rb")
    print("recent long:", flush=True); gen(PAN, REC_CUTS, V0_LEAN + RR, D / f"hl_{tag}_long/v0full_hl60.parquet", "rl")
    print("oos base:", flush=True); gen(PAN, OOS_CUTS, V0_LEAN, D / f"hl_{tag}_base_oos/v0full_hl60.parquet", "ob")
    print("oos long:", flush=True); gen(PAN, OOS_CUTS, V0_LEAN + RR, D / f"hl_{tag}_long_oos/v0full_hl60.parquet", "ol")
    print("M1DONE")

if __name__ == "__main__":
    main()
