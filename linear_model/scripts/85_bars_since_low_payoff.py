"""Step 85: does the new PIT bars_since_low_xs_rank feature add tradable payoff?

User-seeded orthogonal new lead (added to 80a trend group + 80b distlow_x_qvolz
interaction; volaug2 = volaug + bars_since_low_xs_rank, PIT-audit PASS exact).
A genuinely-new non-monotone "distance-from-recent-LOW" feature, symmetric to
bars_since_high, fitting Step-80a's squared/U-shape finding. NOT dependent on
the (closed) 24h direction — this is an independent feature test.

Engine / per-cycle xsz / fold protocol identical to 80b. Testbed hl42.
Configs: v2_all (ref, reproduces 80b) | v2_all+bsl | trend_ext (user's 80a
expanded trend group incl bsl) | bsl_distlow (isolate the new feature +
its volume interaction) | sqbtcrel+int (incl distlow) | v2_all+int(+distlow).

PRE-REGISTERED GATE (fixed before run) — bakes in the Step-84 G4 lesson so we
are not fooled by an unconstrained-RidgeCV-only artifact again:
  a config CLEARS iff (on ridge_xsz)
    decile rho >= +0.60 AND K3 >= +9.0 bps AND >=6/9 folds K3+
    AND drop-top-2 K3 > 0 AND top-5 <= 60% positive gross
  AND ESTIMATOR-CONSISTENT: signed_equal (no-fit) K3 same sign as ridge AND
    signed_equal K3 > 0 (a real edge is not unique to the free 9-dim fit).
DECISION keyed off hl42. Honest base rate is low (whole arc sub-cost), but
this is a clean PIT new feature the user specifically wanted — test it
rigorously, no backtest.
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


s77 = _imp("s77", "linear_model/scripts/77_orientation_decile_diag.py")
s79 = _imp("s79", "linear_model/scripts/79_broader_universe_attrib.py")
s80b = _imp("s80b", "linear_model/scripts/80b_vol_interaction_payoff.py")
from ml.research.alpha_v4_xs_1d import _multi_oos_splits

VOLAUG2 = REPO / "outputs/vBTC_features_btc_only_111_volaug2/panel_btc_only_111_volaug2.parquet"
HL_MAP = REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv"
OUT = REPO / "linear_model/results/step85_bars_since_low"
OUT.mkdir(parents=True, exist_ok=True)
BLOCK = s80b.BLOCK
BSL = "bars_since_low_xs_rank"
GATE_RHO, GATE_K3, GATE_FOLDS, GATE_TOP5 = 0.60, 9.0, 6, 0.60
TREND_EXT = ["return_1d", "dom_btc_change_288b", "vwap_slope_96",
             "return_8h_orth", "bars_since_high_xs_rank", BSL, "obv_z_1d"]


def evaluate(dec, sub, folds, on_hl):
    """Return payoff dict for ridge_xsz + signed_equal (estimator-consistency)."""
    out = {}
    for mdl in ("ridge_xsz", "signed_equal", "nnls_oriented"):
        df = s80b.score(dec, sub, folds, mdl)
        if df.empty:
            out[mdl] = None
            continue
        p = s79.payoff(df)
        if mdl == "ridge_xsz":
            a, ai = s79.attribution(df, on_hl)
            d2 = df[~df["symbol"].isin(ai["top2"])]
            p["k3_drop2"] = (float(s77._ksweep(d2, ks=(3,)).iloc[0]["spread_bps"])
                             if not d2.empty else np.nan)
            p["top5_share"] = ai["top5_share"]
        out[mdl] = p
    return out


def main():
    print("=" * 100, flush=True)
    print("  STEP 85: bars_since_low payoff (hl42, volaug2) — Step-84 G4 lesson "
          "baked in", flush=True)
    print(f"  GATE: ridge ρ>=+{GATE_RHO} & K3>=+{GATE_K3} & >={GATE_FOLDS}/9 & "
          f"dropT2>0 & top5<= {GATE_TOP5:.0%}  AND  signed_equal K3 same-sign "
          f"& >0", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()

    raw = pd.read_parquet(VOLAUG2)
    raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
    assert BSL in raw.columns, "volaug2 missing bars_since_low_xs_rank"
    hl = pd.read_csv(HL_MAP)
    on_hl = set(hl[hl.on_hl]["symbol"])
    folds = _multi_oos_splits(raw[raw["symbol"] != "BTCUSDT"])
    dec, fc, _ = s79.build_universe(raw, hl, folds, "hl42")
    # BSL already flows into fc via build_v2_features on volaug2 (fc=23).
    # Only VOL needs merging; do NOT re-merge BSL (would create _x/_y suffixes).
    assert BSL in fc, "expected bars_since_low_xs_rank in fc from volaug2"
    dec = dec.merge(raw[["symbol", "open_time"] + s80b.VOL],
                    on=["symbol", "open_time"], how="left")
    for c in s80b.VOL + [BSL]:               # BSL already in dec; just fillna
        dec[c] = dec[c].astype("float32").fillna(0.0)
    INT = s80b.add_interactions(dec)         # distlow_x_qvolz now active
    has_distlow = "distlow_x_qvolz" in INT
    V2_22 = [f for f in fc if f != BSL]       # clean original-22 baseline
    print(f"  hl42: {dec['symbol'].nunique()} syms, "
          f"{dec['open_time'].nunique()} cyc | fc={len(fc)} (22 V2 + bsl) "
          f"+{len(s80b.VOL)} vol +{len(INT)} INT (distlow active={has_distlow})",
          flush=True)

    configs = {
        "v2_22": V2_22,                       # clean baseline (no bsl)
        "v2_23_with_bsl": fc,                 # +bsl: marginal effect
        "trend_ext": [c for c in TREND_EXT if c in dec.columns],
        "bsl_distlow": [BSL] + (["distlow_x_qvolz"] if has_distlow else []),
        "sqbtcrel_plus_int": s80b.SQ + s80b.BTCREL + INT,
        "v2_23_plus_int": fc + INT,
    }
    rows = []
    for cname, sub in configs.items():
        sub = [c for c in sub if c in dec.columns]
        r = evaluate(dec, sub, folds, on_hl)
        rd, re_, rn = r.get("ridge_xsz"), r.get("signed_equal"), r.get("nnls_oriented")
        if rd is None:
            print(f"  {cname:20s} no scores", flush=True)
            continue
        est_consistent = (re_ is not None and np.sign(re_["k3"]) == np.sign(rd["k3"])
                          and re_["k3"] > 0)
        payoff_ok = (rd["decile_rho"] >= GATE_RHO and rd["k3"] >= GATE_K3
                     and rd["k3_folds_pos"] >= GATE_FOLDS
                     and rd.get("k3_drop2", -1) > 0
                     and rd.get("top5_share", 1.0) <= GATE_TOP5)
        clears = bool(payoff_ok and est_consistent)
        rows.append({"config": cname, "n_feat": len(sub),
                     "ridge_ic": rd["ic"], "ridge_rho": rd["decile_rho"],
                     "ridge_k3": rd["k3"], "ridge_k3_drop2": rd.get("k3_drop2"),
                     "ridge_folds": rd["k3_folds_pos"], "ridge_top5": rd.get("top5_share"),
                     "eq_k3": re_["k3"] if re_ else np.nan,
                     "eq_rho": re_["decile_rho"] if re_ else np.nan,
                     "nnls_k3": rn["k3"] if rn else np.nan,
                     "est_consistent": est_consistent, "gate_pass": clears})
        print(f"  {cname:20s}({len(sub):2d}f) ridge: ρ={rd['decile_rho']:+.3f} "
              f"K3={rd['k3']:+6.2f} dT2={rd.get('k3_drop2', float('nan')):+6.2f} "
              f"f+={rd['k3_folds_pos']}/9 top5={rd.get('top5_share', float('nan'))*100:3.0f}% "
              f"| eq K3={re_['k3'] if re_ else float('nan'):+6.2f} "
              f"nnls K3={rn['k3'] if rn else float('nan'):+6.2f} "
              f"| {'PASS' if clears else 'FAIL'}"
              f"{'' if est_consistent else ' (est-inconsistent)'}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    base = df[df.config == "v2_22"].iloc[0] if (df.config == "v2_22").any() else None
    bsl_cfgs = df[df.config.isin(["v2_23_with_bsl", "trend_ext",
                                  "bsl_distlow", "v2_23_plus_int"])]
    clears = df[df.gate_pass]
    print("\n" + "=" * 100, flush=True)
    if not clears.empty:
        w = clears.iloc[0]
        v = (f"{w['config']} CLEARS the gate incl estimator-consistency "
             f"(ridge K3 {w['ridge_k3']:+.2f}, eq K3 {w['eq_k3']:+.2f}). "
             f"bars_since_low adds a robust payoff — first estimator-consistent "
             f"clear of the arc. Deeper robustness/audit next, NO backtest.")
    else:
        lift = ""
        if base is not None and len(bsl_cfgs):
            best = bsl_cfgs.loc[bsl_cfgs["ridge_k3"].idxmax()]
            lift = (f" best bsl-config {best['config']} ridge K3 "
                    f"{best['ridge_k3']:+.2f} vs v2_all {base['ridge_k3']:+.2f} "
                    f"(Δ {best['ridge_k3']-base['ridge_k3']:+.2f}); eq "
                    f"{best['eq_k3']:+.2f}")
        v = (f"No config clears the gate (incl the Step-84 estimator-"
             f"consistency requirement).{lift}. bars_since_low does not add a "
             f"robust tradable payoff at 4h on free data. The linear line "
             f"stays sub-cost; feature engineering (orientation, universe, "
             f"groups, volume, interactions, non-monotone distance-from-"
             f"extreme) is exhausted under pre-registered honest gates. NOT a "
             f"proxy call — direct test. Production LGBM unaffected.")
    print(f"  VERDICT: {v}", flush=True)
    pd.DataFrame([{"any_clears": not clears.empty, "verdict": v}]).to_csv(
        OUT / "verdict.csv", index=False)
    print(f"\nSaved under {OUT}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
