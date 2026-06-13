"""Step 101 — D1-ext-F: MAXIMAL free-data feature set → structural events.
Closes the "Step-100 was OI-only" loose end. LOCKED pre-registration
(INFORMATION_DIAGNOSTIC_PLAN.md §D1-ext-F). One run, no sweep.

Feature set = strict SUPERSET of Step 100 = F_core (~24 panel PIT feats) +
s_t + FULL OI panel (11) + order-flow panel (6) + the 6 Step-100 OI
composites (~47 PIT feats, all already audited). Events = the 3 LOCKED
Step-100 definitions, UNCHANGED (no event redefinition = anti-p-hack). No
new hand-crafted composites (LGBM auto-combines; Step-98 showed hand-built
interactions net-destructive). Same leak-free CV (whole-timestamp 5-fold +
1d embargo), logistic+LGBM, OOF AUC, timestamp-block bootstrap,
Bonferroni×3, same hedged-vs-unhedged Stage-2. Reuses Step-100 machinery
(grp_oof_clf, auc_ci, load_hlc) verbatim — only the feature matrix widens.

Honest prior: D1 (94b) already bounded full F_core for the forward RETURN;
E1/E2 are 2σ thresholds of that same return ⇒ strong prior AUC≈0.50. But
full-features→event-classification is a distinct untested cell; this is the
decisive maximal test. No strategy adopted. Production LGBM unaffected.
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
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
s100 = _imp("s100", "linear_model/scripts/100_oi_events.py")
from ml.research.alpha_v4_xs import block_bootstrap_ci
grp_oof_clf, auc_ci, load_hlc = s100.grp_oof_clf, s100.auc_ci, s100.load_hlc
H, ANN = s100.H, s100.ANN
OFLOW = REPO / "outputs/vBTC_features_oflow/oflow_panel.parquet"
OIP = REPO / "outputs/vBTC_features_oi/oi_panel.parquet"
OUTD = REPO / "linear_model/results/step101_oi_events_fullfeat"
OUTD.mkdir(parents=True, exist_ok=True)
HELP = {"return_pct", "btc_ret_fwd", "alpha_beta", "exit_time", "symbol",
        "open_time", "fold", "tz", "r_fwd24", "rng_fwd24", "sig",
        "rng_thr90", "btc_fwd24", "resid_fwd24", "E1", "E2", "E3"}
COMP = ["c_oi_x_ret", "c_oi_x_fund", "c_oi_own", "c_div", "c_ls",
        "c_oi_accel"]


def main():
    print("="*96, flush=True)
    print("  STEP 101 — D1-ext-F: MAXIMAL feature set → structural events "
          "(closes OI-only loose end), LOCKED", flush=True)
    print("="*96, flush=True)
    t0 = time.time()
    dec, syms, btc, pan = s94.build(universe_oi=False)
    oi = pd.read_parquet(OIP); of = pd.read_parquet(OFLOW)
    for f in (oi, of):
        f["open_time"] = pd.to_datetime(f["open_time"], utc=True)
    use = sorted(set(oi.symbol) & set(of.symbol) & set(dec.symbol))
    d = (dec[dec.symbol.isin(use)]
         .merge(oi, on=["symbol", "open_time"], how="inner")
         .merge(of, on=["symbol", "open_time"], how="inner"))
    # 6 Step-100 composites (return_1d, funding_rate_z_7d already in panel/dec)
    d["c_oi_x_ret"] = d["oi_chg_1d"]*np.sign(d["return_1d"])
    d["c_oi_x_fund"] = d["oi_z_7d"]*d["funding_rate_z_7d"]
    d["c_oi_own"] = d["oi_z_7d"]
    d["c_div"] = np.sign(d["return_1d"])*np.sign(d["oi_chg_1d"])
    d["c_ls"] = d["ls_taker_z_1d"]
    d["c_oi_accel"] = d["oi_chg_4h"] - d["oi_chg_1d"]

    # forward 24h targets + PIT sigma + LOCKED Step-100 event labels
    btc_c = s94.load_close("BTCUSDT").set_index("open_time")["close"]
    btc_fwd = (btc_c.shift(-H)/btc_c - 1.0).rename("btc_fwd24")
    parts = []
    for s in use:
        k = load_hlc(s)
        if k is None:
            continue
        k = k.set_index("open_time"); c = k["close"]
        r_fwd = c.shift(-H)/c - 1.0
        rng_fwd = (k["high"].rolling(H).max().shift(-H) -
                   k["low"].rolling(H).min().shift(-H)) / c
        r24 = c.pct_change(H)
        sig = r24.rolling(H*30, min_periods=H*5).std().shift(1)
        rthr = (((k["high"].rolling(H).max()-k["low"].rolling(H).min())/c)
                .rolling(H*30, min_periods=H*5).quantile(0.90).shift(1))
        dd = pd.DataFrame({"r_fwd24": r_fwd, "rng_fwd24": rng_fwd, "sig": sig,
                           "rng_thr90": rthr}, index=c.index).join(btc_fwd)
        dd["symbol"] = s
        parts.append(dd.reset_index())
    fwd = pd.concat(parts, ignore_index=True)
    fwd["open_time"] = pd.to_datetime(fwd["open_time"], utc=True)
    d = d.merge(fwd, on=["symbol", "open_time"], how="inner").dropna(
        subset=["r_fwd24", "rng_fwd24", "sig", "btc_fwd24", "beta_btc_pit",
                "rng_thr90"])
    d = d[d["sig"] > 1e-9].reset_index(drop=True)
    d["resid_fwd24"] = d["r_fwd24"] - d["beta_btc_pit"]*d["btc_fwd24"]
    d["E1"] = (d["r_fwd24"] <= -2.0*d["sig"]).astype(int)
    d["E2"] = (d["r_fwd24"] >= 2.0*d["sig"]).astype(int)
    d["E3"] = (d["rng_fwd24"] >= d["rng_thr90"]).astype(int)

    FEATS = [c for c in d.columns if c not in HELP and
             pd.api.types.is_numeric_dtype(d[c])]
    if "s_t" in d.columns and "s_t" not in FEATS:
        FEATS.append("s_t")
    d = d.dropna(subset=FEATS+["E1", "E2", "E3", "resid_fwd24"]).reset_index(
        drop=True)
    print(f"  rows={len(d)} syms={d.symbol.nunique()} "
          f"cycles={d.open_time.nunique()} FEATS={len(FEATS)} "
          f"(F_core+s_t+OI11+oflow6+comp6 superset of Step-100)", flush=True)
    for e in ["E1", "E2", "E3"]:
        print(f"  base rate {e} = {d[e].mean()*100:.1f}%", flush=True)
    fc = pd.Series({c: abs(d[c].corr(d["resid_fwd24"], "spearman"))
                    for c in FEATS})
    bad = fc[fc > 0.15]
    print(f"  [PIT sanity] max|corr(feat, resid_fwd24)|={fc.max():.3f} "
          f"({fc.idxmax()}); >0.15: {list(bad.index) or 'none'} "
          f"(panel/OI/oflow/s_t all pre-audited; forward cols excluded)",
          flush=True)

    times = d["open_time"].to_numpy()
    res = {}
    for e in ["E1", "E2", "E3"]:
        oof = grp_oof_clf(d, FEATS, d[e])
        a, lo, hi = auc_ci(d[e].to_numpy(), oof, times)
        sig = bool(lo > 0.5 and a > 0.55)
        print(f"\n  [Stage1] {e}: AUC={a:.4f} CI[{lo:.4f},{hi:.4f}] "
              f"baserate={d[e].mean()*100:.1f}% -> "
              f"{'PREDICTABLE (Bonf-sig)' if sig else 'not sig'}", flush=True)
        res[e] = dict(auc=a, lo=lo, hi=hi, sig=sig, oof=oof)

    s2 = []
    grid = sorted(d["open_time"].unique())[::6]
    g = d[d["open_time"].isin(set(grid))].copy()
    COST = s94.COST
    for e, dirn in [("E1", -1.0), ("E2", 1.0)]:
        if not res[e]["sig"]:
            s2.append(f"  [Stage2] {e}: skipped (Stage-1 not sig)")
            continue
        pr = pd.Series(res[e]["oof"], index=d.index).reindex(g.index).to_numpy()
        m = ~np.isnan(pr); gg = g[m]; p = pr[m]
        w = dirn*(p-np.nanmedian(p)); w = w/np.nanstd(w)
        for arm, tgt in [("RESID", "resid_fwd24"), ("RAW", "r_fwd24")]:
            ret = w*gg[tgt].to_numpy()*1e4
            net = ret - np.abs(np.diff(np.concatenate([[0], w])))*COST
            S = s100.sh(net)
            if arm == "RAW":
                rng = np.random.default_rng(0)
                pl = [s100.sh(w[rng.permutation(len(w))]*gg[tgt].to_numpy()*1e4)
                      for _ in range(200)]
                tag = f"placebo p95={np.nanpercentile(pl,95):+.2f} -> " + \
                    ("BEATS" if S > np.nanpercentile(pl, 95) else "fails")
            else:
                tag = f"gate+1.5 -> {'PASS' if S > 1.5 else 'fail'}"
            s2.append(f"  [Stage2] {e} {arm}: NET Sh={S:+.2f} | {tag}")
    print("\n"+"\n".join(s2), flush=True)

    anysig = any(res[e]["sig"] for e in res)
    if not anysig:
        v = ("D1-ext-F: NOT predictable with the MAXIMAL feature set either — "
             "no event clears Bonferroni AUC (best %.3f). Closes the 'OI-only' "
             "loose end: structural forced-deleveraging events are not "
             "predictable from ANY free-data feature combination "
             "(F_core+s_t+OI+order-flow+composites). Comprehensive free-data "
             "terminus is airtight across reduced-form AND structural "
             "paradigms. Only levers: paid data or accept closure." %
             max(res[e]["auc"] for e in res))
    else:
        v = ("D1-ext-F: ≥1 event Bonferroni-predictable with the maximal "
             "feature set — see Stage-2 hedged-vs-unhedged fork "
             "(RESID>+1.5 reopens this line; RAW-only ⇒ separate directional "
             "finding).")
    print(f"\n  VERDICT: {v}", flush=True)
    pd.DataFrame([dict(event=e, **{k: res[e][k] for k in
                 ("auc", "lo", "hi", "sig")}) for e in res]).to_csv(
        OUTD/"stage1.csv", index=False)
    pd.DataFrame([{"verdict": v, "stage2": " || ".join(s2),
                   "n_feats": len(FEATS)}]).to_csv(OUTD/"verdict.csv",
                                                   index=False)
    print(f"\nSaved {OUTD}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
