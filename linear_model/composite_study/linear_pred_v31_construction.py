"""Definitive same-construction test (owner-requested 2026-05-19).

Isolates MODEL vs CONSTRUCTION: feed the linear model's predictions
through production V3.1's EXACT construction machinery (rolling-IC top-15
→ K=3 → conv_gate/PM/flat_real → 6-sleeve overlap → raw-return PnL),
reusing `scripts/phase_ah_sleeve.py`'s functions VERBATIM. Only the
`pred` column changes.

Faithfulness: a CONTROL run with the unchanged production `pred` must
reproduce V3.1 (+2.747 on the matched folds-3–9 grid). If the control
reproduces, the linear-pred runs are a clean apples-to-apples isolation:
  • linear ≈ control (+2.7) ⇒ edge is the CONSTRUCTION (selection+sleeve),
    not the model/features; the §4 "linear negative" was a naive-
    construction artifact.
  • linear ≪ control            ⇒ the linear signal genuinely lacks what
    WINNER_21 has even under identical construction (model/feature gap).

Linear pred is 4h-cadence; represented at the panel's 5m resolution by
backward merge_asof with a 4h tolerance (a 4h signal IS piecewise-constant
over its window; no stale fill across gaps). Linear covers folds 3–9 ⇒ all
Sharpes computed on the matched folds-3–9 grid for ALL THREE. NOT a §4
re-litigation (that verdict stands). Production LGBM untouched.
"""
from __future__ import annotations
import importlib.util
import sys
import time
import warnings
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


AH = _imp("ah_sleeve", "scripts/phase_ah_sleeve.py")          # VERBATIM reuse
H = _imp("harness_v3", "linear_model/composite_study/harness_v3.py")
SUB = _imp("same_univ", "linear_model/composite_study/same_universe_baseline.py")
OUTD = REPO / "linear_model/composite_study/results"
OOS39 = list(range(3, 10))                                     # linear coverage


def build_fwd_rets(syms):
    """Replicates phase_ah_sleeve.main()'s fwd_rets_4h build (4h MtM),
    restricted to `syms` for speed. Identical formula."""
    frames = []
    for sym in syms:
        sd = AH.KLINES_DIR / sym / "5m"
        if not sd.exists():
            continue
        fs = sorted(sd.glob("*.parquet"))
        if not fs:
            continue
        dl = []
        for f in fs:
            try:
                dl.append(pd.read_parquet(f, columns=["open_time", "close"]))
            except Exception:
                pass
        if not dl:
            continue
        df = pd.concat(dl, ignore_index=True)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True,
                                         errors="coerce")
        df = df.dropna(subset=["open_time"]).drop_duplicates(
            "open_time").set_index("open_time").rename(
            columns={"close": sym})
        frames.append(df[[sym]])
    cw = pd.concat(frames, axis=1).sort_index()
    return (cw.shift(-AH.HORIZON_ENTRY) - cw) / cw


def run_construction(apd, fwd, listings, panel_syms, tag):
    """V3.1 construction VERBATIM (AH.* functions) on a given apd."""
    def elig_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cut = ts - pd.Timedelta(days=AH.MIN_HISTORY_DAYS)
        return {s for s in panel_syms
                if listings.get(s) and listings[s] <= cut}
    tgt = sorted(apd[apd["fold"].isin(AH.OOS_FOLDS)]["open_time"].unique())
    samp = tgt[::AH.HORIZON_ENTRY]
    univ = AH.build_rolling_ic_universe(apd, samp, AH.TOP_N, elig_at)
    recs = AH.run_production_protocol_save_sleeves(apd, univ)
    dfv = AH.aggregate_sleeves(recs, fwd)
    m = dfv[dfv["fold"].isin(OOS39)]                            # matched grid
    net = m["net_pnl_bps"].to_numpy()
    full = AH._sharpe(dfv["net_pnl_bps"].to_numpy())
    fsh = {f: AH._sharpe(m[m.fold == f]["net_pnl_bps"].to_numpy())
           for f in OOS39}
    print(f"  [{tag}] folds3-9 Sharpe={AH._sharpe(net):+.3f} "
          f"(all-folds {full:+.3f}) | totPnL={net.sum():+.0f} | "
          f"traded={recs['traded'].sum()} | per-fold "
          + " ".join(f"f{f}={fsh[f]:+.1f}" for f in OOS39), flush=True)
    return AH._sharpe(net), net


def main():
    print("=" * 94, flush=True)
    print("  DEFINITIVE SAME-CONSTRUCTION TEST — linear preds → V3.1's "
          "exact machinery (verbatim)", flush=True)
    print("=" * 94, flush=True)
    t0 = time.time()
    apd = pd.read_parquet(AH.APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())
    listings = AH.get_listings()
    print(f"  apd: {len(apd)} rows, {len(panel_syms)} syms (5m res). "
          f"Building fwd_rets...", flush=True)
    fwd = build_fwd_rets(panel_syms)
    print(f"  fwd_rets {fwd.shape} ({time.time()-t0:.0f}s)", flush=True)

    # ---- linear OOF preds on the V3.1 universe (harness_v3, cleared) ----
    dec, folds = SUB.load_panel_v31_universe()
    if not H.run_selfchecks(dec):
        print("\n  ABORT — self-checks failed.", flush=True)
        return
    pf = H.walk_forward(dec, folds, members=("ridge_best", "lgbm_es"),
                        verbose=False)
    pf = pf[["symbol", "open_time", "ridge_best", "lgbm_es"]].sort_values(
        "open_time").reset_index(drop=True)
    cov = sorted(dec.loc[dec.open_time.isin(pf.open_time), "fold"].unique())
    print(f"  linear preds: {len(pf)} rows, {pf.symbol.nunique()} syms, "
          f"folds covered {cov}", flush=True)

    results = {}
    # CONTROL: unchanged production pred (must reproduce +2.747)
    sh_c, _ = run_construction(apd, fwd, listings, panel_syms,
                               "CONTROL prod-pred")
    results["control"] = sh_c
    faithful = abs(sh_c - 2.747) < 0.30
    print(f"  → faithfulness: control folds3-9 {sh_c:+.3f} vs documented "
          f"+2.747 → {'OK (port faithful)' if faithful else 'MISMATCH — '
          'do NOT trust linear runs'}", flush=True)

    # LINEAR: substitute pred (4h→5m backward asof, 4h tolerance)
    for member in ("ridge_best", "lgbm_es"):
        sub = apd.drop(columns=["pred"]).merge(
            pd.merge_asof(
                apd[["symbol", "open_time"]].sort_values("open_time"),
                pf[["symbol", "open_time", member]].rename(
                    columns={member: "pred"}).sort_values("open_time"),
                on="open_time", by="symbol", direction="backward",
                tolerance=pd.Timedelta(hours=4)),
            on=["symbol", "open_time"], how="left")
        sub = sub.dropna(subset=["pred"]).reset_index(drop=True)
        sh_l, _ = run_construction(sub, fwd, listings, panel_syms,
                                   f"LINEAR {member}")
        results[member] = sh_l

    print("\n" + "=" * 94, flush=True)
    print("  ATTRIBUTION (matched folds 3-9, V3.1 construction verbatim):",
          flush=True)
    print(f"    CONTROL  (production WINNER_21 pred) : {results['control']:+.3f}"
          f"  {'[faithful]' if faithful else '[PORT MISMATCH]'}", flush=True)
    print(f"    LINEAR ridge → V3.1 construction     : "
          f"{results['ridge_best']:+.3f}", flush=True)
    print(f"    LINEAR lgbm  → V3.1 construction     : "
          f"{results['lgbm_es']:+.3f}", flush=True)
    print(f"    (naive β-residual book, same universe: ridge −0.10 / "
          f"lgbm +0.69 — from same_universe_baseline)", flush=True)
    best_lin = max(results['ridge_best'], results['lgbm_es'])
    if not faithful:
        verd = ("PORT NOT FAITHFUL (control ≠ +2.747) — cannot attribute; "
                "disclose, do not over-read.")
    elif best_lin >= results['control'] - 0.30:
        verd = (f"CONSTRUCTION-DOMINATED: linear pred under V3.1 machinery "
                f"≈ control ({best_lin:+.2f} vs {results['control']:+.2f}). "
                f"V3.1's edge is the selection+sleeve construction, NOT the "
                f"model/features — the §4 'linear negative' was largely a "
                f"naive-construction artifact; production's +2.23 is a "
                f"construction property (incl. its meme concentration), "
                f"reproducible by a weak signal in that machinery.")
    elif best_lin >= 1.0:
        verd = (f"BOTH MATTER: linear under V3.1 machinery {best_lin:+.2f} "
                f"— well above its naive-book +0.69 (construction adds a "
                f"lot) but below control {results['control']:+.2f} "
                f"(WINNER_21 model/features add real, separable value).")
    else:
        verd = (f"MODEL-DOMINATED: even under V3.1's exact machinery the "
                f"linear signal stays weak ({best_lin:+.2f} ≪ control "
                f"{results['control']:+.2f}). Production's edge needs its "
                f"specific WINNER_21 model+features (the ability to rank "
                f"the directional meme moves the β-residual book hedges "
                f"out); construction alone does NOT rescue a weak signal.")
    print(f"\n  VERDICT: {verd}", flush=True)
    pd.DataFrame([dict(control=results['control'],
                       lin_ridge=results['ridge_best'],
                       lin_lgbm=results['lgbm_es'], faithful=faithful,
                       verdict=verd)]).to_csv(
        OUTD / "linear_pred_v31_construction.csv", index=False)
    print(f"\nSaved {OUTD}/linear_pred_v31_construction.csv  "
          f"Total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
