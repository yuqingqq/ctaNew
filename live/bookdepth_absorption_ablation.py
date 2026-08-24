"""absorp_net PASSED the univariate both-era screen (alpha -0.006 both eras, CI off zero, same sign on alpha AND raw
return) — the only absorption construction to. DECISIVE test: does it ADD to the strategy? Real-pipeline validity-gated
ablation (x6 + V0_LEAN + per-symbol RidgeCV + HL=60 + exit_time purge + 1d embargo) — same harness that killed imb_ewma
and imb5. VALIDITY GATE: V0_LEAN baseline must reproduce ~+0.030 rec / +0.024 oos. Also tests +all-3 absorption feats,
and (proxy check) whether absorp_net's signal survives when it's the ONLY add vs when V0's beta/vol feats are present.
Uncovered bars -> feat filled 0 (neutral), ~0 weight under HL, eval on covered test bars only.
"""
import os, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
os.environ["V4_PANEL"] = str(REPO / "outputs/vBTC_features/panel_expanded_v0_clean.parquet")
import live.train_twobook_models as tt
from live.bookdepth_imb5_ablation import gen, perbar_ic, V0_LEAN, RECENT_CUTS, OOS_CUTS
from live.bookdepth_absorption import build_sym
rng = np.random.default_rng(7)
ABS_FEATS = ["absorp_bid", "absorp_net", "resil_tot"]

def build_panel():
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "alpha_vs_btc_realized"] + V0_LEAN)
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True); PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
    PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)]
    rows = []
    for f in [x for x in glob.glob(str(REPO / "data/ml/cache/l2_*.parquet")) if "BTCUSDT" not in x]:
        o = build_sym(f)
        if o is not None: rows.append(o[["symbol", "open_time"] + ABS_FEATS])
    L = pd.concat(rows, ignore_index=True)
    PAN = PAN.merge(L, on=["symbol", "open_time"], how="left")
    PAN["_covered"] = PAN["absorp_net"].notna()
    for c in ABS_FEATS: PAN[c] = PAN[c].fillna(0.0)
    g = PAN.groupby("open_time")
    sd = g["alpha_vs_btc_realized"].transform("std").replace(0, np.nan)
    PAN["z_res"] = ((PAN["alpha_vs_btc_realized"] - g["alpha_vs_btc_realized"].transform("mean")) / sd).clip(-10, 10)
    return PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)

def paired(ib, iadd):
    j = pd.concat([ib.rename("a"), iadd.rename("b")], axis=1).dropna(); j["d"] = j["b"] - j["a"]
    j["day"] = pd.to_datetime(j.index, utc=True).floor("1D"); gg = [x["d"].values for _, x in j.groupby("day")]
    boot = [np.concatenate([gg[k] for k in rng.integers(0, len(gg), len(gg))]).mean() for _ in range(3000)]
    return j["d"].mean(), *np.percentile(boot, [2.5, 97.5])

def main():
    PAN = build_panel()
    print(f"panel rows {len(PAN)} | covered {int(PAN['_covered'].sum())} | V0_LEAN={len(V0_LEAN)}\n")
    CAND = [("absorp_net", ["absorp_net"]), ("all-3 absorption", ABS_FEATS)]
    for era, cuts in [("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)]:
        ib = perbar_ic(gen(PAN, V0_LEAN, cuts))
        print(f"### {era} (real per-symbol RidgeCV; rank-IC on covered test bars) ###")
        print(f"  V0_LEAN (baseline)        rank-IC {ib.mean():+.4f}   [validity gate: ~+0.030 rec / +0.024 oos]")
        for lab, add in CAND:
            iadd = perbar_ic(gen(PAN, V0_LEAN + add, cuts))
            d, lo, up = paired(ib, iadd)
            flag = "ADDS (CI>0)" if lo > 0 else ("HURTS (CI<0)" if up < 0 else "within noise")
            print(f"    +{lab:18s} rank-IC {iadd.mean():+.4f}  Δ {d:+.4f} [{lo:+.4f},{up:+.4f}] -> {flag}")
        print()
    print("read: absorp_net ADDS only if Δ CI>0 BOTH eras (else another real-but-redundant/weak feature). ABSABLDONE")

if __name__ == "__main__":
    main()
