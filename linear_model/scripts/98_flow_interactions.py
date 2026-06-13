"""Step 98 — D1-ext-C: perp-vs-spot FLOW divergence + targeted interactions.

Pre-registered (INFORMATION_DIAGNOSTIC_PLAN.md §D1-ext-C), LOCKED, one run,
no sweep. Motivated: flow is the only real marginal of the arc (+0.63,
Step 95); spot price/vol/basis ≈0 (96/97); perp-vs-spot AGGRESSION
divergence + flow×regime interactions are untested.

Fixed feature block (from cached oflow_panel + spot_panel — no rebuild):
  fd_imb     = of_imb_1d - sp_taker_imb_1d        (perp-led vs spot-led)
  fd_absdiff = |fd_imb|                            (flow dislocation mag)
  fd_prod    = of_imb_1d * sp_taker_imb_1d         (co-aggression)
  x_flow_vol = of_tfi_z1d * vol_zscore_4h_over_7d  (flow × vol regime)
  x_flow_fund= of_imb_1d * funding_rate_z_7d       (aggression × carry)
  x_fd_st    = fd_imb * s_t                        (divergence × deviation)

Block added to F_core+oflow (+1.09 best baseline), SAME leak-free CV
(whole-timestamp 5-fold + 1d embargo) + SAME +1.5 gate. Also univariate IC
+ standalone marginal of fd_imb / x_fd_st (anti block-masking). Pure
measurement, no strategy. Production LGBM unaffected.
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
build, score, GATE, LEAK = s94.build, s94.score, s94.GATE, s94.LEAK
grouped_oof, OF, SP = s94b.grouped_oof, s95.OF, s96.SP
FD = ["fd_imb", "fd_absdiff", "fd_prod", "x_flow_vol", "x_flow_fund",
      "x_fd_st"]
OUTD = REPO / "linear_model/results/step98_flow_interactions"
OUTD.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 96, flush=True)
    print("  STEP 98 — D1-ext-C: perp-vs-spot FLOW divergence + interactions "
          "(LOCKED, +1.5 gate)", flush=True)
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
    dec = dec[dec.open_time <= sp["open_time"].max()].reset_index(drop=True)

    # ---- locked interaction block (built from already-PIT features) ----
    dec["fd_imb"] = dec["of_imb_1d"] - dec["sp_taker_imb_1d"]
    dec["fd_absdiff"] = dec["fd_imb"].abs()
    dec["fd_prod"] = dec["of_imb_1d"] * dec["sp_taker_imb_1d"]
    dec["x_flow_vol"] = dec["of_tfi_z1d"] * dec["vol_zscore_4h_over_7d"]
    dec["x_flow_fund"] = dec["of_imb_1d"] * dec["funding_rate_z_7d"]
    dec["x_fd_st"] = dec["fd_imb"] * dec["s_t"]
    # all inputs are TRAILING+shift(1) PIT ⇒ products are PIT (no new leak)

    base = [c for c in dec.columns if c not in LEAK and c not in OF and
            c not in SP and c not in FD and
            pd.api.types.is_numeric_dtype(dec[c])] + ["s_t"]
    base = list(dict.fromkeys(base))
    dec = dec.dropna(subset=base + OF + FD + ["tz", "alpha_beta"]
                     ).reset_index(drop=True)
    print(f"  rows={len(dec)} syms={dec.symbol.nunique()} "
          f"cycles={dec.open_time.nunique()}", flush=True)

    # ---- PIT sanity: products of <0.10-leak features stay <0.10 ----
    fc = dec[FD].apply(lambda c: c.corr(dec["alpha_beta"], "spearman")).abs()
    print(f"  look-ahead |corr(FD, fwd αβ)| max={fc.max():.3f} "
          f"({fc.idxmax()}); all<0.10={bool((fc < 0.10).all())}", flush=True)
    if not (fc < 0.10).all():
        print("  PIT SANITY FAIL — not run.", flush=True)
        pd.DataFrame([{"audit": "FAIL"}]).to_csv(OUTD/"verdict.csv",
                                                 index=False)
        return

    print("\n--- univariate spearman IC of the locked block ---", flush=True)
    for f in FD:
        print(f"   {f:12s} IC={dec[f].corr(dec['alpha_beta'],'spearman'):+.4f}",
              flush=True)

    def ceil(feats, lbl):
        rid, gbm = grouped_oof(dec, feats)
        mk = ~np.isnan(rid)
        dd = dec[mk].reset_index(drop=True)
        print(f"\n--- {lbl} (feats={len(feats)}) ---", flush=True)
        return (score(dd, rid[mk], f"Ridge|{lbl}"),
                score(dd, gbm[mk], f"LGBM|{lbl}"))

    bF = max(ceil(base, "F_core"), key=lambda r: r["net_sh"])["net_sh"]
    bFO = max(ceil(base + OF, "F_core+oflow"),
              key=lambda r: r["net_sh"])["net_sh"]
    R = ceil(base + OF + FD, "F_core+oflow+FLOWINT (GATED)")
    gv = max(R, key=lambda r: r["net_sh"])
    bG = gv["net_sh"]
    # anti block-masking: standalone marginal of the 2 core features
    sFd = max(ceil(base + ["fd_imb"], "F_core+fd_imb"),
              key=lambda r: r["net_sh"])["net_sh"]
    sXs = max(ceil(base + ["x_fd_st"], "F_core+x_fd_st"),
              key=lambda r: r["net_sh"])["net_sh"]
    score(dec, dec["s_t"].to_numpy()*-1.0, "s_t_rule(ref)")

    PASS = bool(bG > GATE)
    print("\n=== SUMMARY (leak-free best NET Sharpe) ===", flush=True)
    print(f"  F_core                         = {bF:+.2f}", flush=True)
    print(f"  F_core+oflow (best baseline)   = {bFO:+.2f}", flush=True)
    print(f"  F_core+oflow+FLOWINT (GATED)   = {bG:+.2f}  (Δ {bG-bFO:+.2f} vs "
          f"F_core+oflow)  IC {gv['ic']:+.3f}", flush=True)
    print(f"  F_core+fd_imb (solo marginal)  = {sFd:+.2f}  (Δ {sFd-bF:+.2f})",
          flush=True)
    print(f"  F_core+x_fd_st (solo marginal) = {sXs:+.2f}  (Δ {sXs-bF:+.2f})",
          flush=True)
    if PASS:
        v = (f"D1-ext-C PASS — F_core+oflow+flow-interactions best NET "
             f"{bG:+.2f} > +1.5 (Δ {bG-bFO:+.2f} vs F_core+oflow). Perp-vs-"
             f"spot flow divergence/interaction is a REAL lever ⇒ line "
             f"reopens, D2 live. Recommend escalating to aggTrade-granularity "
             f"spot flow for the rigorous version.")
    else:
        v = (f"D1-ext-C FAIL — F_core+oflow+flow-interactions best NET "
             f"{bG:+.2f} ≤ +1.5 (Δ {bG-bFO:+.2f} vs F_core+oflow {bFO:+.2f}); "
             f"solo fd_imb Δ{sFd-bF:+.2f}, x_fd_st Δ{sXs-bF:+.2f} (not a "
             f"block-masking artifact). Perp-vs-spot flow divergence + "
             f"targeted flow×regime interactions add no material ceiling "
             f"info. Only remaining free probe = aggTrade-granularity SPOT "
             f"order-flow (spot aggTrades, heavy, Step-95-style); else the "
             f"free-data terminus is final. Production LGBM unaffected.")
    print(f"\n  PRE-REG GATE(>{GATE:+.1f}): {'PASS' if PASS else 'FAIL'}",
          flush=True)
    print(f"  VERDICT: {v}", flush=True)
    pd.DataFrame([dict(F_core=bF, F_oflow=bFO, gated=bG,
                       d_vs_oflow=bG-bFO, solo_fd_imb=sFd, solo_x_fd_st=sXs,
                       PASS=PASS, verdict=v)]).to_csv(OUTD/"summary.csv",
                                                      index=False)
    pd.DataFrame([{"PASS": PASS, "gated": bG, "verdict": v}]).to_csv(
        OUTD/"verdict.csv", index=False)
    print(f"\nSaved {OUTD}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
