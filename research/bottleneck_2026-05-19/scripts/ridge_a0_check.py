"""Mandated follow-up (3-agent results review): is Ridge A0 portable +0.56
a real signal or an artifact? B★b only saved the aggregate. This reproduces
Ridge on A0 (WINNER_21, no sym_id) through the IDENTICAL R3c protocol
(seed 20260519, same groups/folds/disjoint-label/beta-neutral fwd as B★b)
and reports: per-group Sharpe, pooled, LEVEL block-bootstrap CI, and LOFO
group-sign-flip. Pre-registered read:
  - pooled level CI excludes 0 AND >=4/5 groups positive AND no LOFO
    single-group sign-flip  -> +0.56 is a real positive portable signal
    (would reframe: model class IS a lever for the portable baseline).
  - else -> within-noise artifact; dismiss with the number (Ridge 0-imputes
    NaN = shrinkage; per-fold standardize; topK spread was -0.71 i.e.
    negative rank readout; single point < 1 group-sigma from the -0.33
    LGBM cluster).
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "research/portable_alpha_2026-05-19/scripts"))
import phase_ah_sleeve as PA
import R1_baseline_frontier as R1
import R3c_portability_proper as R3c
import ml.research.alpha_vBTC_build_audit_panel as BA
from ml.research.alpha_v4_xs import block_bootstrap_ci
from sklearn.linear_model import Ridge

OUT = REPO / "research/bottleneck_2026-05-19/results"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
G, SEED = 5, 20260519


def _sh(x): return R1._sharpe(np.asarray(x, float))


def fit_ridge(Xt, yt, Xte):           # identical to B_star_b_modelclass.fit_ridge
    mu, sd = np.nanmean(Xt, 0), np.nanstd(Xt, 0) + 1e-9
    Z = np.nan_to_num((Xt - mu) / sd); Zte = np.nan_to_num((Xte - mu) / sd)
    return Ridge(alpha=10.0).fit(Z, yt).predict(Zte)


def main():
    t0 = time.time()
    panel = pd.read_parquet(PANEL)
    feat = [f for f in BA.WINNER_21 if f in panel.columns and f != "sym_id"]
    folds = BA._multi_oos_splits(panel)
    syms = sorted(panel["symbol"].unique())
    listings = PA.get_listings()
    fwd_resid = R3c.build_beta_neutral_fwd()
    sig = pd.read_parquet(R3c.CW)
    sig = sig[[c for c in sig.columns if c.startswith("c_")]].rename(columns=lambda x: x[2:])
    sigma = sig.pct_change().rolling(288, min_periods=48).std().shift(1)
    sigma = sigma.clip(lower=sigma.quantile(0.20), axis=1)
    advc = {s: R1.COST_UNIT for s in syms}
    rng = np.random.RandomState(SEED); shf = syms.copy(); rng.shuffle(shf)
    groups = [shf[i::G] for i in range(G)]

    per_group, pooled = {}, []
    for gi, hold in enumerate(groups):
        hold = set(hold); tr_s = set(syms) - hold
        panel["_tgd"] = R3c.disjoint_target(panel, tr_s).values
        preds = []
        for fid in BA.OOS_FOLDS:
            if fid >= len(folds): continue
            tr, ca, te = BA._slice(panel, folds[fid])
            tr = tr[(tr.autocorr_pctile_7d >= BA.THRESHOLD) & tr.symbol.isin(tr_s)]
            te = te[te.symbol.isin(hold)].copy()
            if len(tr) < 800 or len(te) < 40: continue
            yt = tr["_tgd"].to_numpy(np.float32); mt = ~np.isnan(yt)
            if mt.sum() < 800: continue
            d = te[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
            d["exit_time"] = (te["exit_time"].values if "exit_time" in te
                              else te["open_time"] + pd.Timedelta(minutes=R3c.HOR*5))
            d["fold"] = fid
            d["pred"] = fit_ridge(tr[feat].to_numpy(np.float32)[mt], yt[mt],
                                  te[feat].to_numpy(np.float32))
            preds.append(d)
        if not preds:
            per_group[f"g{gi}"] = None; continue
        apd = pd.concat(preds, ignore_index=True).sort_values(["open_time", "symbol"])
        apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
        apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)

        def elig(b, _h=hold):
            ts = pd.Timestamp(b, unit="ms", tz="UTC") - pd.Timedelta(days=PA.MIN_HISTORY_DAYS)
            return {s for s in _h if listings.get(s) and listings[s] <= ts}
        tt = sorted(apd[apd.fold.isin(PA.OOS_FOLDS)]["open_time"].unique())
        if len(tt) < 40: per_group[f"g{gi}"] = None; continue
        u = PA.build_rolling_ic_universe(apd, tt[::R1.HE], PA.TOP_N, elig)
        rec = PA.run_production_protocol_save_sleeves(apd, u)
        df, _, _ = R1.aggregate_capped(rec, fwd_resid, sigma, advc,
                                       cap_frac=1/3, sizing="equal", cost_mode="flat45")
        per_group[f"g{gi}"] = round(_sh(df["net_pnl_bps"].to_numpy()), 3)
        pooled.append(df[["time", "net_pnl_bps"]])
        print(f"  g{gi}: Ridge A0 portable Sharpe {per_group[f'g{gi}']}", flush=True)

    allp = pd.concat([p.set_index("time") for p in pooled]) if pooled else None
    sh = _sh(allp["net_pnl_bps"].to_numpy()) if allp is not None else None
    gv = [v for v in per_group.values() if v is not None]
    npos = sum(1 for v in gv if v > 0)
    # LEVEL block-bootstrap CI on pooled per-cycle net
    lo = hi = None
    if allp is not None:
        _, lo, hi = block_bootstrap_ci(allp["net_pnl_bps"].to_numpy(),
                                       statistic=_sh, block_size=R1.BLOCK, n_boot=2000)
    # LOFO: drop each group, recompute pooled Sharpe; sign-flip if any flips
    flips = []
    if allp is not None:
        for gi in range(G):
            sub = [p for j, p in enumerate(pooled) if j != gi and p is not None]
            if not sub: continue
            s2 = _sh(pd.concat([p.set_index("time") for p in sub])["net_pnl_bps"].to_numpy())
            if np.sign(s2) != np.sign(sh) and abs(sh) > 1e-9:
                flips.append((f"g{gi}", round(float(s2), 3)))

    real = (sh is not None and lo is not None and lo > 0 and npos >= 4 and not flips)
    res = {"ridge_A0_pooled_sharpe": round(float(sh), 3) if sh is not None else None,
           "per_group": per_group, "groups_positive": f"{npos}/{len(gv)}",
           "level_block_bootstrap_CI": [round(float(lo), 3), round(float(hi), 3)]
                if lo is not None else None,
           "lofo_group_sign_flips": flips,
           "verdict": ("REAL positive portable signal — reframes 'model not a "
                       "lever'" if real else
                       "WITHIN-NOISE ARTIFACT — dismiss (CI incl 0 / <4 groups+ "
                       "/ LOFO flip; Ridge 0-impute shrinkage + per-fold "
                       "standardize; topK spread was -0.71)"),
           "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "ridge_a0_check.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\n  Ridge A0 pooled {res['ridge_A0_pooled_sharpe']} | "
          f"groups+ {res['groups_positive']} | levelCI "
          f"{res['level_block_bootstrap_CI']} | LOFOflips {flips}", flush=True)
    print(f"  VERDICT: {res['verdict']}", flush=True)
    print("RIDGE_A0_DONE", flush=True)


if __name__ == "__main__":
    main()
