"""INCREMENTAL test of the WIDE-BOOK imbalance (imb5, ±5% depth) as an added feature in the REAL alpha-residual rank
system — NOT a standalone directional book. Uses the strategy's real machinery (x6 preproc + V0_LEAN + per-symbol
RidgeCV + HL=60 exp-decay + exit_time purge + 1d embargo), the SAME harness that tested imb1. The pilot found the
directional rank-IC grows with book depth (imb1 +0.002 -> imb5 +0.013, both eras) but CIs barely cleared 0 on 50
syms x 3mo; here we test on the full re-fetched panel (148->176 syms, 2023->2026) whether imb5 ADDS rank-IC to
V0_LEAN, both eras, with a day-clustered paired-delta CI.

VALIDITY GATE (unchanged): the V0_LEAN baseline must reproduce the honest rank-IC (~+0.030 recent / +0.024 oos);
only then is the delta trustworthy. Uncovered old bars -> feature filled 0 (neutral), ~0 weight under HL=60, eval
only on covered test bars.

Candidates (pre-registered):
  imb5_ewma  PRIMARY  EWMA(imb5, hl=12bars=2d)   smoothed persistent WIDE-book lean
  imb5_raw            l2_imb5 (4h-mean, PIT)      the pilot's exact feature
  imb1_ewma           EWMA(imb1)                  NEAR-book incumbent (contrast)
  imb5_ewma+imb1_ewma                             does the wide book ADD over the near book?
"""
import os, sys, glob
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
            "2024-06-01", "2024-09-01", "2024-12-01", "2025-03-01", "2025-06-01", "2025-09-01"]]

FEAT_COLS = ["imb5_raw", "imb1_raw", "imb5_ewma", "imb1_ewma", "imb5_z30"]
CANDIDATES = [("imb1_ewma", ["imb1_ewma"]),               # near-book incumbent (contrast)
              ("imb5_raw", ["imb5_raw"]),                 # pilot's exact feature
              ("imb5_ewma", ["imb5_ewma"]),               # PRIMARY: smoothed wide book
              ("imb5_z30", ["imb5_z30"]),                 # relative wide-book strength
              ("imb5_ewma+imb1_ewma", ["imb5_ewma", "imb1_ewma"])]  # does wide add over near?

def build_panel():
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "alpha_vs_btc_realized"] + V0_LEAN)
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True); PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
    PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)]
    rows = []; n_have5 = 0
    for f in [x for x in glob.glob(str(REPO / "data/ml/cache/l2_*.parquet")) if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]
        try:
            d = pd.read_parquet(f)                       # guard: fetch may be mid-write on the T-Z tail
        except Exception:
            continue
        if "l2_imb5" not in d.columns or d["l2_imb5"].notna().sum() == 0: continue   # skip not-yet-refetched
        n_have5 += 1
        d.index = pd.to_datetime(d.index, utc=True) + pd.Timedelta("4h")   # PIT decision bar
        imb1 = d["l2_imb1"].sort_index(); imb5 = d["l2_imb5"].sort_index()
        p1 = persist_feats(imb1); p5 = persist_feats(imb5)
        pf = pd.DataFrame(index=d.index)
        pf["imb1_raw"] = imb1; pf["imb5_raw"] = imb5
        pf["imb1_ewma"] = p1["imb_ewma"]; pf["imb5_ewma"] = p5["imb_ewma"]; pf["imb5_z30"] = p5["imb_z30"]
        pf["symbol"] = sym; pf["open_time"] = pf.index
        rows.append(pf.reset_index(drop=True))
    L = pd.concat(rows, ignore_index=True)
    PAN = PAN.merge(L, on=["symbol", "open_time"], how="left")
    PAN["_covered"] = PAN["imb5_raw"].notna()          # eval only where the WIDE book was observed
    for c in FEAT_COLS: PAN[c] = PAN[c].fillna(0.0)    # neutral; old/uncovered bars ~0 weight under HL
    g = PAN.groupby("open_time")
    sd = g["alpha_vs_btc_realized"].transform("std").replace(0, np.nan)
    PAN["z_res"] = ((PAN["alpha_vs_btc_realized"] - g["alpha_vs_btc_realized"].transform("mean")) / sd).clip(-10, 10)
    print(f"symbols with imb5: {n_have5}", flush=True)
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
    P = P[P["cov"]]
    return P.groupby("open_time").apply(lambda g: spearmanr(g["pred"], g["alpha_A"]).correlation if len(g) >= 5 else np.nan).dropna()

def main():
    PAN = build_panel()
    print(f"panel rows {len(PAN)} | covered {int(PAN['_covered'].sum())} | V0_LEAN={len(V0_LEAN)} feats\n")
    for era, cuts in [("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)]:
        base = gen(PAN, V0_LEAN, cuts); ib = perbar_ic(base)
        print(f"### {era} (real per-symbol RidgeCV; rank-IC on covered test bars) ###")
        print(f"  V0_LEAN (baseline)          rank-IC {ib.mean():+.4f}   [validity gate: ~+0.030 rec / +0.024 oos]")
        for lab, addcols in CANDIDATES:
            iadd = perbar_ic(gen(PAN, V0_LEAN + addcols, cuts))
            j = pd.concat([ib.rename("a"), iadd.rename("b")], axis=1).dropna(); j["d"] = j["b"] - j["a"]
            j["day"] = pd.to_datetime(j.index, utc=True).floor("1D"); gg = [x["d"].values for _, x in j.groupby("day")]
            boot = [np.concatenate([gg[k] for k in rng.integers(0, len(gg), len(gg))]).mean() for _ in range(3000)]
            lo, up = np.percentile(boot, [2.5, 97.5])
            flag = "ADDS (CI>0)" if lo > 0 else ("HURTS (CI<0)" if up < 0 else "within noise")
            print(f"    +{lab:20s} rank-IC {iadd.mean():+.4f}  Δ {j['d'].mean():+.4f} [{lo:+.4f},{up:+.4f}] -> {flag}")
        print()
    print("read: imb5 ADDS only if Δ CI>0 in BOTH eras. IMB5ABLDONE")

if __name__ == "__main__":
    main()
