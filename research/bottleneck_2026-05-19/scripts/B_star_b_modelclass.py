"""B★b — model-class control for the feature-superset verdict.

Concern (raised on review): B★ measures the de-leaked feature superset
through the PRODUCTION LGBM, which barely fits target_A (best_iter≈7, 20%
at iter 1). A "feature exhausted" verdict through a near-non-learning model
could be a fitting artifact, not a true feature ceiling.

Control: SAME de-leaked A0/A1 feature sets, SAME group-disjoint protocol /
folds / seed / disjoint label / beta-neutral fwd as B★ & R3c, but vary the
LEARNER:
  - RIDGE on per-fold-standardized features (linear; no boosting/early-stop
    pathology — a fundamentally different model class)
  - BIG-LGBM (num_leaves=255, lr=0.01, min_data_in_leaf=20, lambda_l2=0.5,
    4000 rounds, early_stop 200) — high capacity, to test if production
    LGBM's under-training masks signal
Metrics: pooled OOS top-K(=3) realized-alpha_A spread (harness currency,
rank readout — robust to pointwise underfit) + pooled portable R3c Sharpe;
paired Δ(A1−A0) per learner. If NO model class surfaces a portable feature
lever on the superset, "feature ceiling" is earned across model classes.
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
import lightgbm as lgb
from sklearn.linear_model import Ridge

OUT = REPO / "research/bottleneck_2026-05-19/results"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
G, SEED = 5, 20260519


def _sh(x): return R1._sharpe(np.asarray(x, float))


def fit_ridge(Xt, yt, Xte):
    mu, sd = np.nanmean(Xt, 0), np.nanstd(Xt, 0) + 1e-9
    Z = np.nan_to_num((Xt - mu) / sd); Zte = np.nan_to_num((Xte - mu) / sd)
    m = Ridge(alpha=10.0).fit(Z, yt)
    return m.predict(Zte)


def fit_biglgbm(Xt, yt, Xc, yc, seed):
    p = dict(objective="regression", metric="rmse", learning_rate=0.01,
             num_leaves=255, max_depth=-1, min_data_in_leaf=20,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
             lambda_l2=0.5, verbose=-1, seed=seed, feature_fraction_seed=seed,
             bagging_seed=seed, data_random_seed=seed)
    dtr = lgb.Dataset(Xt, yt, free_raw_data=False)
    dc = lgb.Dataset(Xc, yc, reference=dtr, free_raw_data=False)
    mdl = lgb.train(p, dtr, num_boost_round=4000, valid_sets=[dc],
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    return mdl


def main():
    t0 = time.time()
    txt = (OUT / "B_prefeature_ic.txt").read_text()
    SAFE = json.loads(txt.split("SAFE_SUPERSET=")[1].strip())
    panel = pd.read_parquet(PANEL)
    base = [f for f in BA.WINNER_21 if f in panel.columns and f != "sym_id"]
    SAFE = [c for c in SAFE if c in panel.columns and c not in base]
    A = {"A0": base, "A1": base + SAFE}
    fwd_resid = R3c.build_beta_neutral_fwd()
    sig = pd.read_parquet(R3c.CW)
    sig = sig[[c for c in sig.columns if c.startswith("c_")]].rename(columns=lambda x: x[2:])
    sigma = sig.pct_change().rolling(288, min_periods=48).std().shift(1)
    sigma = sigma.clip(lower=sigma.quantile(0.20), axis=1)
    syms = sorted(panel["symbol"].unique())
    advc = {s: R1.COST_UNIT for s in syms}
    listings = PA.get_listings()
    folds = BA._multi_oos_splits(panel)
    rng = np.random.RandomState(SEED); shf = syms.copy(); rng.shuffle(shf)
    groups = [shf[i::G] for i in range(G)]

    LEARNERS = ["ridge", "biglgbm"]
    pooled = {(L, a): [] for L in LEARNERS for a in ("A0", "A1")}
    topk = {(L, a): [] for L in LEARNERS for a in ("A0", "A1")}
    for gi, hold in enumerate(groups):
        hold = set(hold); train_syms = set(syms) - hold
        panel["_tgd"] = R3c.disjoint_target(panel, train_syms).values
        preds = {(L, a): [] for L in LEARNERS for a in ("A0", "A1")}
        for fid in BA.OOS_FOLDS:
            if fid >= len(folds): continue
            tr, ca, te = BA._slice(panel, folds[fid])
            tr = tr[(tr["autocorr_pctile_7d"] >= BA.THRESHOLD) & tr["symbol"].isin(train_syms)]
            ca = ca[(ca["autocorr_pctile_7d"] >= BA.THRESHOLD) & ca["symbol"].isin(train_syms)]
            te = te[te["symbol"].isin(hold)].copy()
            if len(tr) < 1000 or len(ca) < 200 or len(te) < 50: continue
            yt = tr["_tgd"].to_numpy(np.float32); yc = ca["_tgd"].to_numpy(np.float32)
            mt, mc = ~np.isnan(yt), ~np.isnan(yc)
            if mt.sum() < 1000 or mc.sum() < 200: continue
            bd = te[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
            bd["exit_time"] = (te["exit_time"].values if "exit_time" in te.columns
                               else te["open_time"] + pd.Timedelta(minutes=R3c.HOR * 5))
            bd["fold"] = fid
            for a, feat in A.items():
                Xt = tr[feat].to_numpy(np.float32)[mt]
                Xc = ca[feat].to_numpy(np.float32)[mc]
                Xte = te[feat].to_numpy(np.float32)
                # ridge (1 fit)
                d = bd.copy(); d["pred"] = fit_ridge(Xt, yt[mt], Xte)
                preds[("ridge", a)].append(d)
                # big-lgbm (5-seed ensemble)
                ps = [fit_biglgbm(Xt, yt[mt], Xc, yc[mc], s).predict(
                        Xte) for s in BA.SEEDS]
                d2 = bd.copy(); d2["pred"] = np.mean(ps, axis=0)
                preds[("biglgbm", a)].append(d2)
        for L in LEARNERS:
            for a in ("A0", "A1"):
                pl = preds[(L, a)]
                if not pl: continue
                apd = pd.concat(pl, ignore_index=True).sort_values(["open_time", "symbol"])
                apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
                apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
                sp = []
                for t, gg in apd.groupby("open_time"):
                    if len(gg) < 7: continue
                    gg = gg.sort_values("pred")
                    sp.append((gg["alpha_A"].iloc[-3:].mean() - gg["alpha_A"].iloc[:3].mean()) * 1e4)
                topk[(L, a)].append(np.array(sp, float))

                def elig(b, _h=hold):
                    ts = pd.Timestamp(b, unit="ms", tz="UTC") - pd.Timedelta(days=PA.MIN_HISTORY_DAYS)
                    return {s for s in _h if listings.get(s) and listings[s] <= ts}
                tt = sorted(apd[apd["fold"].isin(PA.OOS_FOLDS)]["open_time"].unique())
                if len(tt) < 50: continue
                u = PA.build_rolling_ic_universe(apd, tt[::R1.HE], PA.TOP_N, elig)
                rec = PA.run_production_protocol_save_sleeves(apd, u)
                df, _, _ = R1.aggregate_capped(rec, fwd_resid, sigma, advc,
                                               cap_frac=1/3, sizing="equal",
                                               cost_mode="flat45")
                pooled[(L, a)].append(df[["time", "net_pnl_bps"]].rename(
                    columns={"net_pnl_bps": f"n_{L}_{a}"}))
        print(f"  g{gi} done ({time.time()-t0:.0f}s)", flush=True)

    res = {"safe_n": len(SAFE)}
    for L in LEARNERS:
        e = {}
        for a in ("A0", "A1"):
            if pooled[(L, a)]:
                allp = pd.concat([p.set_index("time") for p in pooled[(L, a)]])
                e[f"{a}_portable_sharpe"] = round(_sh(allp[f"n_{L}_{a}"].to_numpy()), 3)
            if topk[(L, a)]:
                e[f"{a}_topk_spread_bps"] = round(float(np.mean(np.concatenate(topk[(L, a)]))), 2)
        if pooled[(L, "A0")] and pooled[(L, "A1")]:
            a0 = pd.concat([p.set_index("time") for p in pooled[(L, "A0")]])
            a1 = pd.concat([p.set_index("time") for p in pooled[(L, "A1")]])
            j = a0.join(a1, how="inner")
            diff = (j[f"n_{L}_A1"] - j[f"n_{L}_A0"]).to_numpy()
            d_sh = e["A1_portable_sharpe"] - e["A0_portable_sharpe"]
            _, lo, hi = block_bootstrap_ci(diff, statistic=lambda x: float(np.mean(x)),
                                           block_size=R1.BLOCK, n_boot=2000)
            e["delta_sharpe"] = round(d_sh, 3)
            e["diff_ci_bps"] = [round(float(lo), 3), round(float(hi), 3)]
            e["diff_excludes_0"] = bool(lo > 0 or hi < 0)
        res[L] = e
        print(f"  {L}: A0 {e.get('A0_portable_sharpe')} A1 {e.get('A1_portable_sharpe')} "
              f"Δ {e.get('delta_sharpe')} CI {e.get('diff_ci_bps')} | "
              f"topK A0 {e.get('A0_topk_spread_bps')} A1 {e.get('A1_topk_spread_bps')}",
              flush=True)
    res["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "B_star_b_modelclass.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"B★b done [{res['elapsed_s']}s]", flush=True)


if __name__ == "__main__":
    main()
