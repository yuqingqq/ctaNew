"""CAREFUL, single-frame quantification of the ORDER BOOK's influence on the deployed strategy. Uses the VALIDATED
real pipeline (x6 + per-symbol RidgeCV + HL=60 + exit_time purge + 1d embargo; baseline MUST reproduce +0.030 rec /
+0.017 oos). Adds the full OB feature set to V0_LEAN and reports, on the SAME covered universe, both eras:
  - rank-IC (base, +OB, Δ with day-clustered bootstrap CI)   [signal quality]
  - 1L/2S selection-spread daily Sharpe (base, +OB, Δ)        [tradeable performance terms]
so OB's influence is stated in both IC and Sharpe. Validity gate = baseline reproduces the reference numbers.
"""
import os, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
os.environ["V4_PANEL"] = str(REPO / "outputs/vBTC_features/panel_expanded_v0_clean.parquet")
from live.bookdepth_real_ablation import gen, V0_LEAN, RECENT_CUTS, OOS_CUTS
import live.train_twobook_models as tt
from live.bookdepth_persist import persist_feats
from scipy.stats import spearmanr
rng = np.random.default_rng(5)
OB = ["imb_ewma", "l2_imb1", "l2_liq1", "l2_slope", "l2_asym1", "l2_imbstd"]   # representative OB set (dir+liq+shape+instability)

def build():
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "alpha_vs_btc_realized"] + V0_LEAN)
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True); PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
    PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)]
    rows = []
    for f in [x for x in glob.glob(str(REPO / "data/ml/cache/l2_*.parquet")) if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]; d = pd.read_parquet(f); d.index = pd.to_datetime(d.index, utc=True) + pd.Timedelta("4h")
        pf = pd.DataFrame(index=d.index); pf["imb_ewma"] = persist_feats(d["l2_imb1"].sort_index())["imb_ewma"]
        for c in ["l2_imb1", "l2_liq1", "l2_slope", "l2_asym1", "l2_imbstd"]:
            pf[c] = d[c] if c in d.columns else np.nan
        pf["symbol"] = sym; pf["open_time"] = pf.index; rows.append(pf.reset_index(drop=True))
    L = pd.concat(rows, ignore_index=True)
    PAN = PAN.merge(L, on=["symbol", "open_time"], how="left")
    PAN["_covered"] = PAN["imb_ewma"].notna()
    for c in OB: PAN[c] = PAN[c].fillna(0.0)
    g = PAN.groupby("open_time"); sd = g["alpha_vs_btc_realized"].transform("std").replace(0, np.nan)
    PAN["z_res"] = ((PAN["alpha_vs_btc_realized"] - g["alpha_vs_btc_realized"].transform("mean")) / sd).clip(-10, 10)
    return PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)

def per_bar(preds):
    p = preds[preds["cov"]]; ic = {}; sp = {}
    for t, gg in p.groupby("open_time"):
        if len(gg) < 5: continue
        ic[t] = spearmanr(gg["pred"], gg["alpha_A"]).correlation
        L = gg.nlargest(1, "pred"); S = gg.nsmallest(2, "pred")
        sp[t] = 0.5 * L["alpha_A"].iloc[0] - 0.5 * S["alpha_A"].mean()
    return pd.Series(ic).dropna(), pd.Series(sp).dropna()

def sharpe(sp):
    sp = sp.copy(); sp.index = pd.to_datetime(sp.index, utc=True)
    dd = sp.groupby(sp.index.date).sum(); return dd.mean() / dd.std() * np.sqrt(365) if dd.std() > 0 else np.nan

def d_ci(sb, sa):
    """day-clustered bootstrap CI on the mean per-bar difference (rank-IC or selection-spread)."""
    j = pd.concat([sb.rename("b"), sa.rename("a")], axis=1).dropna(); j["d"] = j["a"] - j["b"]
    j["day"] = pd.to_datetime(j.index, utc=True).floor("1D"); g = [x["d"].values for _, x in j.groupby("day")]
    boot = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(3000)]
    return j["d"].mean(), tuple(np.percentile(boot, [2.5, 97.5]))

def main():
    PAN = build()
    print(f"panel {len(PAN)} rows | covered {int(PAN['_covered'].sum())} | OB set = {OB}\n")
    for era, cuts in [("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)]:
        pb = gen(PAN, V0_LEAN, cuts); pa = gen(PAN, V0_LEAN + OB, cuts)
        icb, spb = per_bar(pb); ica, spa = per_bar(pa)
        dic, (il, ih) = d_ci(icb, ica)                         # Δ rank-IC [CI]
        dsp, (sl, sh_) = d_ci(spb, spa)                        # Δ mean per-bar selection-spread [CI], return units
        print(f"### {era} (validity: base rank-IC should be ~+0.030 rec / ~+0.017 oos) ###")
        print(f"  rank-IC:        base {icb.mean():+.4f}  +OB {ica.mean():+.4f}  Δ {dic:+.4f} [{il:+.4f},{ih:+.4f}]"
              f"  -> {'HURTS (CI<0)' if ih < 0 else ('HELPS (CI>0)' if il > 0 else 'within noise')}")
        print(f"  sel-spread:     base Sharpe {sharpe(spb):+.2f}  +OB {sharpe(spa):+.2f} ; Δ mean-spread {dsp*1e4:+.2f}bps "
              f"[{sl*1e4:+.2f},{sh_*1e4:+.2f}] -> {'HELPS (CI>0)' if sl > 0 else ('HURTS (CI<0)' if sh_ < 0 else 'within noise')}\n")
    print("INFLDONE")

if __name__ == "__main__":
    main()
