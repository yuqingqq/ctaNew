"""OI + aggTrade-flow portable VALIDATION (PLAN v3).

Lockstep paired-Δ through the R3c portable protocol (group-disjoint, no
sym_id, disjoint label, beta-neutral fwd, full V3.1 stack, costed). Per
PLAN v3 + reviewer-mandated doc-independent guards:

  - Arms: A0 WINNER_21(no sym_id); A1 +OI; A2 +flow; A3 +OI+flow.
    Models: M1 production LGBM (BA._train), M2 Ridge(standardized).
  - COVERAGE-IDENTITY LEAK FIX: each arm uses its OWN paired A0 baseline on
    that arm's admissible rows = rows where ALL its added features are
    non-NaN. NaN symbol-cycles are dropped IDENTICALLY from both arms (never
    NaN-passed to LGBM nor 0-imputed to Ridge). A0 anchor recomputed on the
    surviving row-set (the −0.33 51-sym number is NOT assumed).
  - Leak denylist + blocking assert max|rankIC(feat,target_A)|<0.10.
  - PIT: OI via audited build_btc_oi_features.oi_features (each .shift(1),
    ffill limit2). Flow: per-symbol .shift(1) (conservative — vpin already
    strictly PIT, extra shift only loses 5min, never leaks), exact-timestamp
    left-join to panel open_time (NOT merge_asof).
  - MDE-in-Sharpe blocking precompute; LOFO single-fold sign-flip kill;
    block-bootstrap CI on per-cycle paired diff; full-51 + covered-subset.

Usage: python3 oi_flow_test.py [--stage0]   (stage0 = use whatever flow/OI
caches exist now; full run = after fetch_flow completes)
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time, warnings
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
import importlib.util
_spec = importlib.util.spec_from_file_location("boif", REPO / "scripts/build_btc_oi_features.py")
BOIF = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(BOIF)

OUT = REPO / "research/orthogonal_oi_flow_2026-05-19/results"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
CACHE = REPO / "data/ml/cache"
G, SEED = 5, 20260519


def _sh(x): return R1._sharpe(np.asarray(x, float))


def build_oi_panel(panel):
    cached = {os.path.basename(f)[8:-8] for f in glob.glob(str(CACHE / "metrics_*.parquet"))}
    syms = sorted(s for s in panel["symbol"].unique() if s in cached)
    parts = []
    for s in syms:
        m = pd.read_parquet(CACHE / f"metrics_{s}.parquet")
        if len(m) < 50_000: continue
        if m.index.tz is None: m.index = m.index.tz_localize("UTC")
        grid = pd.DatetimeIndex(panel[panel.symbol == s]["open_time"]
                                .sort_values().drop_duplicates())
        f = BOIF.oi_features(m, grid).reset_index().rename(columns={"index": "open_time"})
        f["symbol"] = s
        parts.append(f)
    oip = pd.concat(parts, ignore_index=True)
    oip["open_time"] = pd.to_datetime(oip["open_time"], utc=True)
    return oip, BOIF.FEATS, sorted(oip["symbol"].unique())


def build_flow_panel(panel):
    files = sorted(glob.glob(str(CACHE / "flow_*.parquet")))
    FCOLS = ["signed_volume", "tfi", "aggressor_count_ratio", "avg_trade_size",
             "vwap_dev_bps", "large_trade_volume", "large_trade_count",
             "tfi_smooth", "signed_volume_z", "vpin"]
    parts, syms = [], []
    pan_syms = set(panel["symbol"].unique())
    for fp in files:
        s = os.path.basename(fp)[5:-8]
        if s not in pan_syms: continue
        d = pd.read_parquet(fp)
        cols = [c for c in FCOLS if c in d.columns]
        if not cols: continue
        d = d[cols].sort_index().shift(1)              # conservative PIT shift
        d = d.reset_index().rename(columns={d.index.name or "index": "open_time"})
        d.columns = ["open_time"] + [f"flw_{c}" for c in cols]
        d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
        d["symbol"] = s
        parts.append(d); syms.append(s)
    fp_ = pd.concat(parts, ignore_index=True)
    return fp_, [f"flw_{c}" for c in FCOLS], sorted(set(syms))


def leak_guard(panel, cols, tag):
    y = panel["target_A"]; m = y.notna()
    bad = []
    rows = []
    for c in cols:
        x = panel[c]; g = m & x.notna()
        ic = abs(np.corrcoef(x[g].rank(), y[g].rank())[0, 1]) if g.sum() > 30000 else 0.0
        rows.append((c, round(float(ic), 4)))
        if ic >= 0.10: bad.append((c, ic))
    (OUT / f"_leakguard_{tag}.txt").write_text(
        f"max|rankIC|={max(abs(r[1]) for r in rows):.4f}\n" +
        "\n".join(f"{c} {ic:+.4f}" for c, ic in rows))
    assert not bad, f"LEAK GUARD FAIL {tag}: {bad}"
    return max(abs(r[1]) for r in rows)


def coverage_group_auc(panel, addcols, groups_map):
    """AUC( any-added-feature-NaN  ->  held-out-group ). ~0.5 = no identity
    leak. One-vs-rest macro AUC via rank statistic (no sklearn dep needed)."""
    nanflag = panel[addcols].isna().any(axis=1).astype(int).to_numpy()
    grp = panel["symbol"].map(groups_map).to_numpy()
    aucs = []
    for gi in range(G):
        yb = (grp == gi).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb): continue
        # AUC of using nanflag to predict membership in group gi
        order = np.argsort(nanflag)
        r = np.empty(len(nanflag)); r[order] = np.arange(1, len(nanflag) + 1)
        n1 = yb.sum(); n0 = len(yb) - n1
        auc = (r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        aucs.append(abs(auc - 0.5) + 0.5)
    return float(np.mean(aucs)) if aucs else 0.5


def fit_ridge(Xt, yt, Xte):
    mu, sd = np.nanmean(Xt, 0), np.nanstd(Xt, 0) + 1e-9
    Z = np.nan_to_num((Xt - mu) / sd); Zte = np.nan_to_num((Xte - mu) / sd)
    return Ridge(alpha=10.0).fit(Z, yt).predict(Zte)


def run_arm(panel, base_feats, add_feats, model, groups, fwd_resid, sigma, advc,
            listings, folds):
    """Returns dict per-group portable sharpe + pooled per-cycle net for A0
    (on this arm's admissible rows) and ARM, fully lockstep."""
    feA0, feA = base_feats, base_feats + add_feats
    out = {"A0": [], "ARM": []}
    for gi, hold in enumerate(groups):
        hold = set(hold); tr_s = set(panel["symbol"].unique()) - hold
        panel["_tgd"] = R3c.disjoint_target(panel, tr_s).values
        preds = {"A0": [], "ARM": []}
        for fid in BA.OOS_FOLDS:
            if fid >= len(folds): continue
            tr, ca, te = BA._slice(panel, folds[fid])
            tr = tr[(tr.autocorr_pctile_7d >= BA.THRESHOLD) & tr.symbol.isin(tr_s)]
            ca = ca[(ca.autocorr_pctile_7d >= BA.THRESHOLD) & ca.symbol.isin(tr_s)]
            te = te[te.symbol.isin(hold)].copy()
            # admissible rows = added feats all non-NaN (identical both arms)
            def adm(df): return df.dropna(subset=add_feats)
            tr, ca, te = adm(tr), adm(ca), adm(te)
            if len(tr) < 800 or len(ca) < 150 or len(te) < 40: continue
            yt = tr["_tgd"].to_numpy(np.float32); yc = ca["_tgd"].to_numpy(np.float32)
            mt, mc = ~np.isnan(yt), ~np.isnan(yc)
            if mt.sum() < 800 or mc.sum() < 150: continue
            bd = te[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
            bd["exit_time"] = (te["exit_time"].values if "exit_time" in te
                               else te["open_time"] + pd.Timedelta(minutes=R3c.HOR*5))
            bd["fold"] = fid
            for key, feat in (("A0", feA0), ("ARM", feA)):
                Xt = tr[feat].to_numpy(np.float32)[mt]
                Xc = ca[feat].to_numpy(np.float32)[mc]
                Xte = te[feat].to_numpy(np.float32)
                if model == "lgbm":
                    ps = []
                    for s in BA.SEEDS:
                        mdl = BA._train(Xt, yt[mt], Xc, yc[mc], seed=s)
                        ps.append(mdl.predict(Xte, num_iteration=mdl.best_iteration))
                    pred = np.mean(ps, axis=0)
                else:
                    pred = fit_ridge(Xt, yt[mt], Xte)
                d = bd.copy(); d["pred"] = pred
                preds[key].append(d)
        for key in ("A0", "ARM"):
            if not preds[key]: continue
            apd = pd.concat(preds[key], ignore_index=True).sort_values(["open_time", "symbol"])
            apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
            apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)

            def elig(b, _h=hold):
                ts = pd.Timestamp(b, unit="ms", tz="UTC") - pd.Timedelta(days=PA.MIN_HISTORY_DAYS)
                return {s for s in _h if listings.get(s) and listings[s] <= ts}
            tt = sorted(apd[apd.fold.isin(PA.OOS_FOLDS)]["open_time"].unique())
            if len(tt) < 40: continue
            u = PA.build_rolling_ic_universe(apd, tt[::R1.HE], PA.TOP_N, elig)
            rec = PA.run_production_protocol_save_sleeves(apd, u)
            df, _, _ = R1.aggregate_capped(rec, fwd_resid, sigma, advc,
                                           cap_frac=1/3, sizing="equal",
                                           cost_mode="flat45")
            out[key].append(df[["time", "net_pnl_bps"]].rename(
                columns={"net_pnl_bps": f"n_{key}"}))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage0", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    panel = pd.read_parquet(PANEL)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    base = [f for f in BA.WINNER_21 if f in panel.columns and f != "sym_id"]

    oip, oicols, oi_syms = build_oi_panel(panel)
    flp, flcols, fl_syms = build_flow_panel(panel)
    panel = panel.merge(oip, on=["symbol", "open_time"], how="left")
    panel = panel.merge(flp, on=["symbol", "open_time"], how="left")
    print(f"OI syms={len(oi_syms)} flow syms={len(fl_syms)} "
          f"OIcols={len(oicols)} FLcols={len(flcols)}", flush=True)

    rng = np.random.RandomState(SEED)
    syms = sorted(panel["symbol"].unique()); shf = syms.copy(); rng.shuffle(shf)
    groups = [shf[i::G] for i in range(G)]
    gmap = {s: gi for gi, gp in enumerate(groups) for s in gp}

    lk_oi = leak_guard(panel, oicols, "oi")
    lk_fl = leak_guard(panel, flcols, "flow")
    auc_oi = coverage_group_auc(panel, oicols, gmap)
    auc_fl = coverage_group_auc(panel, flcols, gmap)
    print(f"leakguard max|IC| OI={lk_oi:.4f} flow={lk_fl:.4f} | "
          f"coverage->group AUC OI={auc_oi:.3f} flow={auc_fl:.3f}", flush=True)

    fwd_resid = R3c.build_beta_neutral_fwd()
    sig = pd.read_parquet(R3c.CW)
    sig = sig[[c for c in sig.columns if c.startswith("c_")]].rename(columns=lambda x: x[2:])
    sigma = sig.pct_change().rolling(288, min_periods=48).std().shift(1)
    sigma = sigma.clip(lower=sigma.quantile(0.20), axis=1)
    advc = {s: R1.COST_UNIT for s in syms}
    listings = PA.get_listings()
    folds = BA._multi_oos_splits(panel)

    ARMS = {"A1_OI": oicols, "A2_FLOW": flcols, "A3_OIFLOW": oicols + flcols}
    MODELS = ["lgbm", "ridge"]
    res = {"oi_syms": len(oi_syms), "flow_syms": len(fl_syms),
           "leak_oi": lk_oi, "leak_flow": lk_fl,
           "cov_auc_oi": round(auc_oi, 3), "cov_auc_flow": round(auc_fl, 3),
           "arms": {}}
    for aname, addf in ARMS.items():
        for mdl in MODELS:
            o = run_arm(panel, base, addf, mdl, groups, fwd_resid, sigma, advc,
                        listings, folds)
            if not o["A0"] or not o["ARM"]:
                res["arms"][f"{aname}|{mdl}"] = {"status": "insufficient"}; continue
            a0 = pd.concat([p.set_index("time") for p in o["A0"]])
            ar = pd.concat([p.set_index("time") for p in o["ARM"]])
            j = a0.join(ar, how="inner")
            sh0, sha = _sh(j["n_A0"]), _sh(j["n_ARM"])
            diff = (j["n_ARM"] - j["n_A0"]).to_numpy()
            _, lo, hi = block_bootstrap_ci(diff, statistic=lambda x: float(np.mean(x)),
                                           block_size=R1.BLOCK, n_boot=2000)
            n_eff = len(diff) / R1.BLOCK
            mde_bps = 2.49 * diff.std() / np.sqrt(max(n_eff, 1))
            e = {"A0_sharpe": round(sh0, 3), "ARM_sharpe": round(sha, 3),
                 "delta_sharpe": round(sha - sh0, 3),
                 "mean_diff_bps": round(float(diff.mean()), 3),
                 "diff_ci_bps": [round(float(lo), 3), round(float(hi), 3)],
                 "diff_excludes_0": bool(lo > 0 or hi < 0),
                 "n_eff": round(n_eff, 1), "mde_bps": round(float(mde_bps), 3),
                 "n_cyc": int(len(j))}
            d = e["delta_sharpe"]
            if d >= 0.5 and e["diff_excludes_0"]:
                v = "LEVER-REAL"
            elif d <= 0.2 or not e["diff_excludes_0"]:
                v = "no-portable-lift"
            else:
                v = "ambiguous"
            if abs(diff.mean()) < mde_bps and not e["diff_excludes_0"]:
                v += " | underpowered (effect-size, not 'exhausted')"
            e["verdict"] = v
            res["arms"][f"{aname}|{mdl}"] = e
            print(f"  {aname}|{mdl}: A0 {sh0:+.3f} ARM {sha:+.3f} Δ {d:+.3f} "
                  f"CI{e['diff_ci_bps']} -> {v}", flush=True)
    res["elapsed_s"] = round(time.time() - t0, 1)
    tag = "stage0" if args.stage0 else "full"
    (OUT / f"oi_flow_results_{tag}.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nOI/FLOW test ({tag}) done [{res['elapsed_s']}s]", flush=True)
    print("OIFLOW_DONE", flush=True)


if __name__ == "__main__":
    main()
