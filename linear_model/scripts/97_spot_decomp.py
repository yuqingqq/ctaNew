"""Step 97 — DIAGNOSTIC (not a gated test): decompose WHY spot adds nothing,
and isolate the spot-perp VOLUME-DIVERGENCE feature specifically.

Owner Q: "why does spot info add nothing? what about spot-perp volume dif?"
The Step-96 verdict used the 6-feature spot BLOCK — a block can mask one
useful feature, and `sp_volratio_z1d` (spot/perp quote-vol ratio z) is the
piece expected to be orthogonal. This measures, leak-free (same
whole-timestamp+embargo CV):
  (1) each spot feature's UNIVARIATE leak-free Ridge ceiling (standalone),
  (2) single-feature MARGINAL of sp_volratio_z1d (and sp_retdiff_4h) added
      alone to F_core and to F_core+oflow,
  (3) the 2 divergence-only features together.
Reuses cached oflow/spot panels — no rebuild. Pure measurement.
"""
from __future__ import annotations
import importlib.util, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
s94b = _imp("s94b", "linear_model/scripts/94b_info_ceiling_d1_grouped.py")
s95 = _imp("s95", "linear_model/scripts/95_d1ext_orderflow.py")
s96 = _imp("s96", "linear_model/scripts/96_d1ext_spot.py")
build, score, LEAK = s94.build, s94.score, s94.LEAK
grouped_oof, OF, SP = s94b.grouped_oof, s95.OF, s96.SP
OUTD = REPO / "linear_model/results/step97_spot_decomp"
OUTD.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 96, flush=True)
    print("  STEP 97 — spot decomposition: why spot adds nothing; isolate "
          "spot-perp volume-divergence", flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()
    dec, syms, btc, pan = build(universe_oi=False)
    of = pd.read_parquet(REPO / "outputs/vBTC_features_oflow/oflow_panel.parquet")
    sp = pd.read_parquet(REPO / "outputs/vBTC_features_spot/spot_panel.parquet")
    for f in (of, sp):
        f["open_time"] = pd.to_datetime(f["open_time"], utc=True)
    dec = (dec[dec.symbol.isin(of.symbol.unique())]
           .merge(of, on=["symbol", "open_time"], how="inner")
           .merge(sp, on=["symbol", "open_time"], how="inner"))
    dec = dec[dec.open_time <= sp["open_time"].max()]
    base = [c for c in dec.columns if c not in LEAK and c not in OF and
            c not in SP and pd.api.types.is_numeric_dtype(dec[c])] + ["s_t"]
    base = list(dict.fromkeys(base))
    dec = dec.dropna(subset=base + OF + SP + ["tz", "alpha_beta"]
                     ).reset_index(drop=True)
    print(f"  rows={len(dec)} syms={dec.symbol.nunique()} "
          f"cycles={dec.open_time.nunique()}\n", flush=True)

    def ceil(feats, lbl):
        rid, gbm = grouped_oof(dec, feats)
        mk = ~np.isnan(rid)
        dd = dec[mk].reset_index(drop=True)
        return score(dd, rid[mk], lbl)            # Ridge = linear ceiling

    # (0) univariate raw spearman IC of each spot feat vs forward residual
    print("--- (0) spot features: univariate spearman IC vs fwd alpha_beta ---",
          flush=True)
    for f in SP:
        ic = dec[f].corr(dec["alpha_beta"], method="spearman")
        print(f"   {f:18s} IC={ic:+.4f}", flush=True)

    # (1) each spot feature's STANDALONE leak-free Ridge ceiling
    print("\n--- (1) standalone leak-free Ridge ceiling, one spot feat at a "
          "time ---", flush=True)
    for f in SP:
        ceil([f], f"solo|{f}")

    # (2) F_core baselines + single-divergence-feature marginals
    print("\n--- (2) marginal of the DIVERGENCE features (added alone) ---",
          flush=True)
    bF = ceil(base, "F_core")["net_sh"]
    bFO = ceil(base + OF, "F_core+oflow")["net_sh"]
    vF = ceil(base + ["sp_volratio_z1d"], "F_core+sp_volratio_z1d")["net_sh"]
    rF = ceil(base + ["sp_retdiff_4h"], "F_core+sp_retdiff_4h")["net_sh"]
    dvF = ceil(base + ["sp_volratio_z1d", "sp_retdiff_4h"],
               "F_core+[volratio,retdiff]")["net_sh"]
    vFO = ceil(base + OF + ["sp_volratio_z1d"],
               "F_core+oflow+sp_volratio_z1d")["net_sh"]

    print("\n=== SUMMARY (leak-free Ridge NET Sharpe) ===", flush=True)
    print(f"  F_core                       = {bF:+.2f}", flush=True)
    print(f"  F_core + sp_volratio_z1d     = {vF:+.2f}   "
          f"(Δ {vF-bF:+.2f}  ← spot-perp VOLUME divergence, alone)",
          flush=True)
    print(f"  F_core + sp_retdiff_4h       = {rF:+.2f}   (Δ {rF-bF:+.2f})",
          flush=True)
    print(f"  F_core + [volratio,retdiff]  = {dvF:+.2f}   (Δ {dvF-bF:+.2f})",
          flush=True)
    print(f"  F_core+oflow                 = {bFO:+.2f}", flush=True)
    print(f"  F_core+oflow + sp_volratio   = {vFO:+.2f}   (Δ {vFO-bFO:+.2f})",
          flush=True)
    verdict = (
        f"spot-perp volume divergence (sp_volratio_z1d) standalone marginal: "
        f"Δ{vF-bF:+.2f} vs F_core, Δ{vFO-bFO:+.2f} vs F_core+oflow. "
        f"retdiff Δ{rF-bF:+.2f}; both-divergence Δ{dvF-bF:+.2f}. "
        f"{'Volume-divergence DOES carry standalone marginal — block result masked it; worth a focused look.' if (vF-bF) > 0.30 or (vFO-bFO) > 0.30 else 'Volume-divergence carries no material standalone marginal either — confirms spot redundancy is real, not a block-masking artifact (4h horizon arbitrages spot-perp lead-lag; basis≈funding already in F_core).'}")
    print(f"\n  VERDICT: {verdict}", flush=True)
    pd.DataFrame([dict(F_core=bF, volratio=vF, retdiff=rF, both=dvF,
                       F_oflow=bFO, oflow_volratio=vFO,
                       d_volratio_vs_Fcore=vF-bF,
                       d_volratio_vs_Foflow=vFO-bFO,
                       verdict=verdict)]).to_csv(OUTD/"summary.csv",
                                                 index=False)
    print(f"\nSaved {OUTD}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
