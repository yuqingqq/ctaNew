"""Step 80a: feature-group ablation + group-payoff diagnostic.

Per user direction: treat the 22 V2 features as ~5 GROUPS, not 22 independent
signals; ablate by group and gate by GROUP PAYOFF (not feature IC). Answers:
is the economically-dead payoff (Steps 77-79) caused by one noisy group
dominating/diluting, or is every group dead?

SCALING NOTE (verified, intentional skip): the user's "re-standardize *_sq
first" step is a PROVABLE NO-OP on this diagnostic path. s58.build_v2_features
builds `*_sq = winsorize_zscore(raw, train)**2` with no re-standardization
(raw *_sq mean~1 std~1.4-2.1, confirmed). But every score below applies
per-cycle cross-sectional z (`s78._xsz`), and cross-sectional z is invariant
to any train-affine rescale: xsz(a*c+b) == xsz(c). So re-standardizing *_sq
cannot change any score/coef here. It IS valid hygiene for the non-xsz
production s58.train_ridge path (Step-75 raw-Ridge) — flagged for that path,
not run here.

Testbed: hl42 (the clean executable baseline; Step 79 showed hl_all degrades
and Binance-110 is the meme-tail artifact, so group structure is diagnosed
where the +6-7 bps sub-cost signal actually lives). Scores carried:
nnls_oriented (best payoff shape, K3 +6.77) and ridge_xsz (best K1 shape).

Configs per model: all (baseline) | drop_<group> x5 (leave-one-group-out) |
only_<group> x5 (single-group standalone). Each through the Step-77 payoff
diagnostic + drop-top-2 de-concentration.

PRE-REGISTERED GATES (fixed before run):
  * full payoff gate (a config "clears"):
      decile rho >= +0.60  AND  K3 >= +9.0 bps  AND  >=6/9 folds K3+
      AND drop-top-2 K3 > 0
  * group classification vs the `all` baseline K3:
      DOMINATING-NOISE  : drop_<g> raises K3 by >= +3.0 bps
      ESSENTIAL         : drop_<g> lowers K3 by >= +3.0 bps
      NEUTRAL           : |delta K3| < 3.0 bps
DECISION: if no LOGO config clears the full gate, the dead payoff is
structural across groups -> Step 80b builds price x volume interactions only
from groups that are ESSENTIAL or whose single-group payoff is best. If a
DOMINATING-NOISE group exists and dropping it clears the gate -> that is the
finding; carry that subset forward. No backtest here.
"""
from __future__ import annotations

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s58 = _imp("s58", "linear_model/scripts/58_clean108_train.py")
s76 = _imp("s76", "linear_model/scripts/76_minimal_orientation.py")
s77 = _imp("s77", "linear_model/scripts/77_orientation_decile_diag.py")
s78 = _imp("s78", "linear_model/scripts/78_nnls_poscoef_payoff.py")
from ml.research.alpha_v4_xs_1d import _slice

OUT = REPO / "linear_model/results/step80a_group_ablation"
OUT.mkdir(parents=True, exist_ok=True)
OOS, BLOCK, ALPHAS = s76.OOS, s76.BLOCK, s58.ALPHAS
GATE_RHO, GATE_SPREAD, GATE_FOLDS, DELTA = 0.60, 9.0, 6, 3.0

GROUPS = {
    "trend": ["return_1d", "dom_btc_change_288b", "vwap_slope_96",
              "return_8h_orth", "bars_since_high_xs_rank",
              "bars_since_low_xs_rank", "obv_z_1d"],
    "squared": ["return_1d_sq", "corr_to_btc_1d_sq", "beta_to_btc_change_5d_sq",
                "dom_btc_change_288b_sq", "corr_to_btc_change_3d_sq"],
    "vol": ["atr_pct", "idio_vol_to_btc_1h", "idio_vol_to_btc_1d",
            "vol_zscore_4h_over_7d"],
    "funding": ["funding_rate", "funding_rate_z_7d", "funding_rate_1d_change"],
    "btc_rel": ["corr_to_btc_1d", "corr_to_btc_change_3d",
                "beta_to_btc_change_5d", "dom_btc_z_1d"],
}
MODELS = ["nnls_oriented", "ridge_xsz"]


def score(dec, fc_all, fc_sub, folds, model):
    """OOS (cycle,symbol) score using ONLY fc_sub. Same engine as Step 78/79."""
    idx = [fc_all.index(f) for f in fc_sub]
    rows = []
    for k in OOS:
        if k >= len(folds):
            continue
        tr = _slice(dec, folds[k])[0].dropna(subset=["alpha_beta"])
        if len(tr) < 500:
            continue
        w76 = s76.fit_weights(tr, fc_sub, "alpha_beta")
        sign = np.array([np.sign(w76[f]) or 1.0 for f in fc_sub], float)
        coef = None
        if model in ("ridge_xsz", "nnls_oriented"):
            Xtr, ytr = [], []
            for _, g in tr.dropna(subset=["target_z"]).groupby("open_time"):
                if len(g) < 5:
                    continue
                Z = s78._xsz(g, fc_all)[:, idx]
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
            Z = s78._xsz(g, fc_all)[:, idx]
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


def payoff(df):
    ic = df.groupby("open_time").apply(
        lambda g: g["score"].corr(g["y"], method="spearman")).dropna()
    rho = s77._mono(s77._binned(df, 10))
    ks = s77._ksweep(df, ks=(1, 3, 10))
    k3 = float(ks[ks["K"] == 3].iloc[0]["spread_bps"])
    fp = 0
    for _, g in df.groupby("fold"):
        sp = [s.head(3)["y"].mean() - s.tail(3)["y"].mean()
              for _, gg in g.groupby("open_time") if len(gg) >= 6
              for s in [gg.sort_values("score", ascending=False)]]
        if sp and np.mean(sp) > 0:
            fp += 1
    # drop-top-2 by |gross| de-concentration
    contrib = {}
    for _, g in df.groupby("open_time"):
        if len(g) < 6:
            continue
        gs = g.sort_values("score", ascending=False)
        for s, yy in zip(gs.head(3)["symbol"], gs.head(3)["y"]):
            contrib[s] = contrib.get(s, 0.0) + yy
        for s, yy in zip(gs.tail(3)["symbol"], gs.tail(3)["y"]):
            contrib[s] = contrib.get(s, 0.0) - yy
    top2 = [s for s, _ in sorted(contrib.items(),
            key=lambda kv: -abs(kv[1]))[:2]]
    d2 = df[~df["symbol"].isin(top2)]
    k3_d2 = (float(s77._ksweep(d2, ks=(3,)).iloc[0]["spread_bps"])
             if not d2.empty else np.nan)
    return dict(ic=float(ic.mean()), rho=float(rho),
                k1=float(ks[ks["K"] == 1].iloc[0]["spread_bps"]), k3=k3,
                k10=float(ks[ks["K"] == 10].iloc[0]["spread_bps"]),
                folds=fp, k3_drop2=k3_d2,
                gate=bool(rho >= GATE_RHO and k3 >= GATE_SPREAD
                          and fp >= GATE_FOLDS and k3_d2 > 0))


def main():
    print("=" * 100, flush=True)
    print("  STEP 80a: feature-group ablation + group-payoff (hl42 testbed)",
          flush=True)
    print(f"  GATE: rho>=+{GATE_RHO} & K3>=+{GATE_SPREAD} & >={GATE_FOLDS}/9 & "
          f"drop-top2 K3>0 | group |dK3|>={DELTA} = essential/dominating",
          flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    panel, px, fc, folds = s76.s67.build_panel(s76.DROP)        # hl42
    px["open_time"] = pd.to_datetime(px["open_time"], utc=True)
    grid = sorted(px["open_time"].unique())[::BLOCK]
    dec = s76.assign_folds(px[px["open_time"].isin(set(grid))].copy(), folds)

    groups = {g: [f for f in feats if f in fc] for g, feats in GROUPS.items()}
    covered = sorted(sum(groups.values(), []))
    if set(covered) != set(fc):
        print(f"  !! GROUP MAP MISMATCH\n   only_in_fc={sorted(set(fc)-set(covered))}"
              f"\n   only_in_groups={sorted(set(covered)-set(fc))}", flush=True)
        return
    print(f"  hl42: {dec['symbol'].nunique()} syms, {len(fc)} feats, "
          f"5 groups (sizes {[len(v) for v in groups.values()]}) — map OK",
          flush=True)

    rows = []
    for mdl in MODELS:
        print(f"\n--- {mdl} ---", flush=True)
        configs = [("all", fc)]
        for gname, gfeats in groups.items():
            configs.append((f"drop_{gname}", [f for f in fc if f not in gfeats]))
        for gname, gfeats in groups.items():
            configs.append((f"only_{gname}", list(gfeats)))
        base_k3 = None
        for cname, sub in configs:
            df = score(dec, fc, sub, folds, mdl)
            if df.empty:
                print(f"  {cname:16s} no scores", flush=True)
                continue
            p = payoff(df)
            if cname == "all":
                base_k3 = p["k3"]
            cls = ""
            if cname.startswith("drop_") and base_k3 is not None:
                d = p["k3"] - base_k3
                cls = ("DOMINATING-NOISE" if d >= DELTA else
                       "ESSENTIAL" if d <= -DELTA else "neutral")
                cls = f" [{cls} dK3={d:+.2f}]"
            rows.append({"model": mdl, "config": cname, **p})
            print(f"  {cname:16s} IC={p['ic']:+.4f} rho={p['rho']:+.3f} "
                  f"K1={p['k1']:+6.2f} K3={p['k3']:+6.2f} K10={p['k10']:+6.2f} "
                  f"f+={p['folds']}/9 dropT2K3={p['k3_drop2']:+6.2f} "
                  f"{'PASS' if p['gate'] else 'FAIL'}{cls}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "summary.csv", index=False)
    logo_clear = out[(out["config"].str.startswith("drop_")) & out["gate"]]
    print("\n" + "=" * 100, flush=True)
    print("  VERDICT", flush=True)
    print("=" * 100, flush=True)
    if not logo_clear.empty:
        w = logo_clear.iloc[0]
        v = (f"Dropping a group CLEARS the full payoff gate "
             f"({w['model']}/{w['config']}: rho {w['rho']:+.2f} K3 "
             f"{w['k3']:+.2f}) -> that group was dominating noise; carry the "
             f"surviving subset to Step 80b. Still NO backtest.")
    else:
        best = out[out["config"].str.startswith("only_")].sort_values(
            "k3", ascending=False).head(1)
        bstr = (f"best single group = {best.iloc[0]['config']} "
                f"K3 {best.iloc[0]['k3']:+.2f} rho {best.iloc[0]['rho']:+.2f}"
                if not best.empty else "n/a")
        v = (f"No leave-one-group-out config clears the full gate -> the dead "
             f"payoff is STRUCTURAL across all groups, not one noisy group "
             f"dominating ({bstr}). Step 80b: build price x volume "
             f"interactions only from the ESSENTIAL / best-single-group "
             f"features; honest base rate remains low. No backtest.")
    print(f"  {v}", flush=True)
    pd.DataFrame([{"logo_clears": not logo_clear.empty, "verdict": v}]).to_csv(
        OUT / "verdict.csv", index=False)
    print(f"\nSaved under {OUT}\nTotal: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
