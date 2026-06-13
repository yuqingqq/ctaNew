"""Step 80b: non-monotone / magnitude x VOLUME interaction payoff diagnostic.

Built on Step-80a's finding: the residual-predictive structure is NON-MONOTONE
(squared/U-shape group is the lone essential carrier; signed-linear weightings
top out). So this does NOT build naive signed price*volume — it builds
magnitude / squared * volume interactions, plus the user's high-prior pairs
(dist-from-high*vol, funding*|ret|, btc-rel*vol), from the volaug panel
(=full-PIT spine + 9 PIT-verified volume features, Step 81 PASS).

Testbed: hl42 (the clean executable baseline where the +6-7 bps sub-cost
signal lives; Step 79 showed broader universes degrade / are meme-tail).
Scores: nnls_oriented (best 80a payoff shape) + ridge_xsz. Engine / per-cycle
xsz / fold protocol identical to Steps 76-80a (xsz at score time makes a
separate train-restandardize of the products a provable no-op).

Configs: v2_all (ref) | vol_only (9 raw vol) | int_only (12 interactions) |
sqbtcrel_plus_int (80a surviving groups + interactions = user's "from
surviving groups only") | v2_all_plus_int. Each through the Step-77 payoff
diagnostic + drop-top-2 + per-symbol attribution (+ HL flag, meme-tail guard).

PRE-REGISTERED GATE (fixed before run, same bar as 78/79/80a):
  a config "clears" iff decile rho >= +0.60 AND K3 >= +9.0 bps AND
  >=6/9 folds K3+ AND drop-top-2 K3 > 0 AND top-5 <= 60% positive gross.
DECISION keyed off hl42. If nothing clears -> the linear line is closed even
with proper volume + non-monotone interactions; accept the 4h free-data
ceiling, LGBM stays production. No backtest here.
"""
from __future__ import annotations
import importlib.util, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


s58 = _imp("s58", "linear_model/scripts/58_clean108_train.py")
s76 = _imp("s76", "linear_model/scripts/76_minimal_orientation.py")
s77 = _imp("s77", "linear_model/scripts/77_orientation_decile_diag.py")
s78 = _imp("s78", "linear_model/scripts/78_nnls_poscoef_payoff.py")
s79 = _imp("s79", "linear_model/scripts/79_broader_universe_attrib.py")
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

VOLAUG = REPO / "outputs/vBTC_features_btc_only_111_volaug/panel_btc_only_111_volaug.parquet"
HL_MAP = REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv"
OUT = REPO / "linear_model/results/step80b_vol_interaction"
OUT.mkdir(parents=True, exist_ok=True)
OOS, BLOCK, ALPHAS = s76.OOS, s76.BLOCK, s58.ALPHAS
GATE_RHO, GATE_K3, GATE_FOLDS, GATE_TOP5 = 0.60, 9.0, 6, 0.60

VOL = ["qvol_z_1d", "qvol_z_7d", "qvol_z_30d", "qvol_surge_1h_over_1d",
       "dollar_vol_log_1d", "amihud_illiq_1d", "taker_buy_frac_z_1d",
       "signed_qvol_1h", "trade_size_z_1d"]
SQ = ["return_1d_sq", "corr_to_btc_1d_sq", "beta_to_btc_change_5d_sq",
      "dom_btc_change_288b_sq", "corr_to_btc_change_3d_sq"]
BTCREL = ["corr_to_btc_1d", "corr_to_btc_change_3d", "beta_to_btc_change_5d",
          "dom_btc_z_1d"]


def add_interactions(dec: pd.DataFrame) -> list:
    a = dec["return_1d"].abs()
    a8 = dec["return_8h_orth"].abs()
    adom = dec["dom_btc_change_288b"].abs()
    abr = dec["beta_to_btc_change_5d"].abs()
    I = {
        "abs_ret1d_x_qvolz": a * dec["qvol_z_1d"],
        "ret1d_sq_x_qvolz": dec["return_1d_sq"] * dec["qvol_z_1d"],
        "abs_ret8h_x_qvolz": a8 * dec["qvol_z_1d"],
        "abs_ret1d_x_surge": a * dec["qvol_surge_1h_over_1d"],
        "absdom_x_signedflow": adom * dec["signed_qvol_1h"],
        "ret1d_sq_x_amihud": dec["return_1d_sq"] * dec["amihud_illiq_1d"],
        "abs_ret1d_x_takerz": a * dec["taker_buy_frac_z_1d"],
        "disthigh_x_qvolz": dec["bars_since_high_xs_rank"] * dec["qvol_z_1d"],
        "absbtcrel_x_qvolz": abr * dec["qvol_z_1d"],
        "funding_x_absret": dec["funding_rate"] * a,
        "corrbtc_sq_x_qvolz": dec["corr_to_btc_1d_sq"] * dec["qvol_z_1d"],
        "tradesz_x_absret": dec["trade_size_z_1d"] * a,
    }
    if "bars_since_low_xs_rank" in dec.columns:
        I["distlow_x_qvolz"] = dec["bars_since_low_xs_rank"] * dec["qvol_z_1d"]
    for k, v in I.items():
        dec[k] = v.astype("float32")
    return list(I)


def score(dec, sub, folds, model):
    rows = []
    for k in OOS:
        if k >= len(folds):
            continue
        tr = _slice(dec, folds[k])[0].dropna(subset=["alpha_beta"])
        if len(tr) < 500:
            continue
        w = s76.fit_weights(tr, sub, "alpha_beta")
        sign = np.array([np.sign(w[f]) or 1.0 for f in sub], float)
        coef = None
        Xtr, ytr = [], []
        for _, g in tr.dropna(subset=["target_z"]).groupby("open_time"):
            if len(g) < 5:
                continue
            Z = s78._xsz(g, sub)
            yv = g["target_z"].to_numpy(float)
            if not np.isfinite(yv).all() or np.std(yv) <= 1e-12:
                continue
            Xtr.append(Z if model == "ridge_xsz" else Z * sign)
            ytr.append(yv)
        if not Xtr:
            continue
        Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
        if model == "ridge_xsz":
            coef = RidgeCV(alphas=ALPHAS, scoring="r2",
                           fit_intercept=False).fit(Xtr, ytr).coef_
        else:
            coef, _ = nnls(Xtr, ytr)
        te = dec[dec["fold"] == k].dropna(subset=["alpha_beta"]).copy()
        for t, g in te.groupby("open_time", sort=True):
            if len(g) < 5:
                continue
            Z = s78._xsz(g, sub)
            yv = g["alpha_beta"].to_numpy(float)
            if np.std(yv) <= 1e-12:
                continue
            sc = (Z * sign) @ coef if model == "nnls_oriented" else Z @ coef
            if np.std(sc) <= 1e-12:
                continue
            rows.append(pd.DataFrame({"open_time": t, "fold": k,
                                      "symbol": g["symbol"].to_numpy(),
                                      "score": sc, "y": yv}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    print("=" * 100, flush=True)
    print("  STEP 80b: non-monotone x VOLUME interaction payoff (hl42, volaug)",
          flush=True)
    print(f"  GATE: rho>=+{GATE_RHO} & K3>=+{GATE_K3} & >={GATE_FOLDS}/9 & "
          f"drop-top2 K3>0 & top5<= {GATE_TOP5:.0%} pos-gross", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()

    raw = pd.read_parquet(VOLAUG)
    raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
    hl = pd.read_csv(HL_MAP)
    folds = _multi_oos_splits(raw[raw["symbol"] != "BTCUSDT"])
    dec, fc, on_hl = s79.build_universe(raw, hl, folds, "hl42")
    # carry the 9 PIT-verified vol cols onto the decision frame
    dec = dec.merge(raw[["symbol", "open_time"] + VOL],
                    on=["symbol", "open_time"], how="left")
    for c in VOL:
        dec[c] = dec[c].astype("float32").fillna(0.0)
    INT = add_interactions(dec)
    print(f"  hl42: {dec['symbol'].nunique()} syms, "
          f"{dec['open_time'].nunique()} cycles | 22 V2 + {len(VOL)} vol + "
          f"{len(INT)} interactions", flush=True)

    configs = {
        "v2_all": fc,
        "vol_only": VOL,
        "int_only": INT,
        "sqbtcrel_plus_int": SQ + BTCREL + INT,
        "v2_all_plus_int": fc + INT,
    }
    rows = []
    for mdl in ["nnls_oriented", "ridge_xsz"]:
        print(f"\n--- {mdl} ---", flush=True)
        for cname, sub in configs.items():
            sub = [c for c in sub if c in dec.columns]
            df = score(dec, sub, folds, mdl)
            if df.empty:
                print(f"  {cname:20s} no scores", flush=True)
                continue
            p = s79.payoff(df)
            a, ai = s79.attribution(df, on_hl)
            gate = (p["decile_rho"] >= GATE_RHO and p["k3"] >= GATE_K3
                    and p["k3_folds_pos"] >= GATE_FOLDS
                    and (a.head(5)["gross_bps"].sum()
                         / max(a.loc[a.gross_bps > 0, "gross_bps"].sum(), 1e-9))
                    <= GATE_TOP5)
            # drop-top-2 de-concentration
            d2 = df[~df["symbol"].isin(ai["top2"])]
            k3d2 = (float(s77._ksweep(d2, ks=(3,)).iloc[0]["spread_bps"])
                    if not d2.empty else np.nan)
            gate = gate and (k3d2 > 0)
            rows.append({"model": mdl, "config": cname, "n_feat": len(sub),
                         **p, "k3_drop2": k3d2,
                         "top5_share": ai["top5_share"],
                         "top2": ",".join(ai["top2"]),
                         "gate_pass": bool(gate)})
            print(f"  {cname:20s} ({len(sub):2d}f) IC={p['ic']:+.4f} "
                  f"rho={p['decile_rho']:+.3f} K1={p['k1']:+6.2f} "
                  f"K3={p['k3']:+6.2f} K10={p['k10']:+6.2f} f+={p['k3_folds_pos']}/9 "
                  f"dT2={k3d2:+6.2f} top5={ai['top5_share']*100:3.0f}% | "
                  f"{'PASS' if gate else 'FAIL'}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "summary.csv", index=False)
    clears = out[out["gate_pass"]]
    print("\n" + "=" * 100, flush=True)
    print("  VERDICT", flush=True)
    print("=" * 100, flush=True)
    if not clears.empty:
        w = clears.iloc[0]
        v = (f"{w['model']}/{w['config']} CLEARS the full payoff gate "
             f"(rho {w['decile_rho']:+.2f}, K3 {w['k3']:+.2f}, "
             f"dropT2 {w['k3_drop2']:+.2f}) — proper volume + non-monotone "
             f"interactions revive the linear line. Next = robustness/audit "
             f"of that config, still NO backtest.")
    else:
        bi = out.sort_values("k3", ascending=False).head(1)
        b = (f"best = {bi.iloc[0]['model']}/{bi.iloc[0]['config']} "
             f"K3 {bi.iloc[0]['k3']:+.2f} rho {bi.iloc[0]['decile_rho']:+.2f}"
             if not bi.empty else "n/a")
        v = (f"No config clears the gate even with proper PIT volume features "
             f"+ non-monotone/magnitude interactions ({b}). The linear "
             f"beta-residual line is CLOSED on free 4h Binance data — feature "
             f"engineering exhausted (orientation 76-78, universe 79, groups "
             f"80a, volume+interactions 80b). Accept the 4h free-data ceiling; "
             f"LGBM stays production. proper-24h (Step-76 bug) remains the "
             f"only separately-unresolved item.")
    print(f"  {v}", flush=True)
    pd.DataFrame([{"any_clears": not clears.empty, "verdict": v}]).to_csv(
        OUT / "verdict.csv", index=False)
    print(f"\nSaved under {OUT}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
