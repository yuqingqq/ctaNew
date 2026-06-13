"""R3 — robustness -> sizing & kill-switch (diagnostic, never a veto).

Chosen deployable config from R1 = equal-weight, cap-1/3 (8/9 folds, +2.06,
reduces single-name dollar dominance). R3 quantifies, per PLAN.md:

  (a) drop-k composition-drift sensitivity: k in {1,3,5}, 30 draws, on the
      chosen config -> mean/worst/std Sharpe.
  (b) held-out-symbol-group portability (the user's core question: "can't
      replay on a different symbol universe"). G=5 disjoint symbol groups;
      for each, retrain the model on the OTHER 4 groups WITHOUT sym_id
      (held-out symbols are unseen -> sym_id meaningless), predict the
      held-out group, evaluate a K=2 long/short spread on alpha_beta
      (BTC-beta-residual = beta-neutral by construction -> guards the
      shared-BTC-factor false positive). Per-group + pooled Sharpe.
  (c) synthesis -> recommended live deployment fraction + max-DD kill-switch.

Diagnostic only: informs sizing; never vetoes the R1 deployable result.
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
import ml.research.alpha_vBTC_build_audit_panel as BA

OUT = REPO / "research/portable_alpha_2026-05-19/results"
APD = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
CAP = 1 / 3
CPY = R1.CPY
G = 5
SEED = 20260519


def _sh(x):
    return R1._sharpe(np.asarray(x, float))


def dropk(apd, panel_syms, listings, fwd, sigma, advc, k, n=30):
    rng = np.random.RandomState(SEED + k)

    def elig(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC") - pd.Timedelta(days=PA.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= ts}
    tt = sorted(apd[apd["fold"].isin(PA.OOS_FOLDS)]["open_time"].unique())
    sm = tt[::R1.HE]
    out = []
    for _ in range(n):
        drop = set(rng.choice(panel_syms, k, replace=False))
        keep = [s for s in panel_syms if s not in drop]
        a2 = apd[apd["symbol"].isin(keep)].copy()
        u2 = PA.build_rolling_ic_universe(a2, sm, PA.TOP_N, elig)
        r2 = PA.run_production_protocol_save_sleeves(a2, u2)
        df, _, _ = R1.aggregate_capped(r2, fwd, sigma, advc, cap_frac=CAP,
                                       sizing="equal", cost_mode="flat45")
        out.append(_sh(df["net_pnl_bps"].to_numpy()))
    a = np.array(out)
    return {"k": k, "mean": round(float(a.mean()), 3),
            "worst": round(float(a.min()), 3), "std": round(float(a.std()), 3),
            "p10": round(float(np.percentile(a, 10)), 3),
            "frac_pos": round(float((a > 0).mean()), 3)}


def group_disjoint_portability(panel_syms):
    """Retrain WITHOUT sym_id on 4/5 symbol groups, predict the held-out
    group, evaluate K=2 spread on alpha_beta (beta-neutral)."""
    panel = pd.read_parquet(PANEL)
    feat = [f for f in BA.WINNER_21 if f in panel.columns and f != "sym_id"]
    if "alpha_beta" not in panel.columns:
        return {"error": "alpha_beta not in panel"}
    folds = BA._multi_oos_splits(panel)
    rng = np.random.RandomState(SEED)
    syms = sorted(panel_syms)
    rng.shuffle(syms)
    groups = [syms[i::G] for i in range(G)]
    per_group = {}
    pooled_cyc = []
    for gi, hold in enumerate(groups):
        hold = set(hold)
        train_syms = set(panel_syms) - hold
        preds = []
        for fid in BA.OOS_FOLDS:
            if fid >= len(folds):
                continue
            tr, ca, te = BA._slice(panel, folds[fid])
            tr = tr[(tr["autocorr_pctile_7d"] >= BA.THRESHOLD) &
                    (tr["symbol"].isin(train_syms))]
            ca = ca[(ca["autocorr_pctile_7d"] >= BA.THRESHOLD) &
                    (ca["symbol"].isin(train_syms))]
            te = te[te["symbol"].isin(hold)].copy()
            if len(tr) < 1000 or len(ca) < 200 or len(te) < 50:
                continue
            Xt, yt = tr[feat].to_numpy(np.float32), tr["target_A"].to_numpy(np.float32)
            Xc, yc = ca[feat].to_numpy(np.float32), ca["target_A"].to_numpy(np.float32)
            mt, mc = ~np.isnan(yt), ~np.isnan(yc)
            if mt.sum() < 1000 or mc.sum() < 200:
                continue
            ps = []
            for s in BA.SEEDS:
                m = BA._train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
                ps.append(m.predict(te[feat].to_numpy(np.float32),
                                    num_iteration=m.best_iteration))
            d = te[["symbol", "open_time", "alpha_beta", "fold"]].copy() \
                if "fold" in te.columns else te[["symbol", "open_time", "alpha_beta"]].copy()
            d["pred"] = np.mean(ps, axis=0)
            d["fold"] = fid
            preds.append(d)
        if not preds:
            per_group[f"g{gi}"] = {"n_held": len(hold), "sharpe": None}
            continue
        pp = pd.concat(preds, ignore_index=True).dropna(subset=["alpha_beta"])
        # K=2 long/short spread on alpha_beta, sampled every HE, beta-neutral
        cyc = []
        for t, gg in pp.groupby("open_time"):
            if len(gg) < 5:
                continue
            gg = gg.sort_values("pred")
            sh_b = gg["alpha_beta"].iloc[:2].mean()
            ln = gg["alpha_beta"].iloc[-2:].mean()
            cyc.append({"open_time": t, "spread": (ln - sh_b) * 1e4})
        c = pd.DataFrame(cyc).sort_values("open_time")
        c = c.iloc[::R1.HE]
        s = _sh(c["spread"].to_numpy()) if len(c) > 5 else None
        per_group[f"g{gi}"] = {"n_held": len(hold),
                               "members": sorted(hold)[:6],
                               "sharpe": round(s, 3) if s is not None else None,
                               "mean_spread_bps": round(float(c["spread"].mean()), 2)
                               if len(c) else None, "n_cyc": len(c)}
        pooled_cyc.append(c)
        print(f"  group {gi} (n={len(hold)}): held-out alpha-only Sharpe "
              f"{per_group[f'g{gi}']['sharpe']}", flush=True)
    pooled = None
    if pooled_cyc:
        allc = pd.concat(pooled_cyc)
        pooled = round(_sh(allc["spread"].to_numpy()), 3)
    return {"per_group": per_group, "pooled_alpha_only_sharpe": pooled}


def main():
    t0 = time.time()
    apd = pd.read_parquet(APD)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())
    listings = PA.get_listings()
    fwd, sigma, advc = R1.build_caches(apd, panel_syms)

    print("R3 (a) drop-k composition drift on equal cap-1/3 ...", flush=True)
    dk = {}
    for k in (1, 3, 5):
        dk[k] = dropk(apd, panel_syms, listings, fwd, sigma, advc, k, 30)
        print(f"  k={k}: mean {dk[k]['mean']:+.2f} worst {dk[k]['worst']:+.2f} "
              f"p10 {dk[k]['p10']:+.2f} fracpos {dk[k]['frac_pos']}", flush=True)

    print("\nR3 (b) group-disjoint (no sym_id) held-out portability ...", flush=True)
    gd = group_disjoint_portability(panel_syms)

    # chosen config maxDD (equal cap-1/3) for kill-switch
    def elig(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC") - pd.Timedelta(days=PA.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= ts}
    tt = sorted(apd[apd["fold"].isin(PA.OOS_FOLDS)]["open_time"].unique())
    u = PA.build_rolling_ic_universe(apd, tt[::R1.HE], PA.TOP_N, elig)
    rec = PA.run_production_protocol_save_sleeves(apd, u)
    base_df, _, _ = R1.aggregate_capped(rec, fwd, sigma, advc, cap_frac=CAP,
                                        sizing="equal", cost_mode="flat45")
    base_sh = _sh(base_df["net_pnl_bps"].to_numpy())
    base_dd = R1._max_dd(base_df["net_pnl_bps"].to_numpy())

    # ---- synthesis: sizing fraction + kill-switch ----------------------
    worst5 = dk[5]["worst"]
    # size so worst-case drop-5 path stays >= +0.3 Sharpe-equivalent risk floor;
    # cap fraction at 1.0. If worst5 already > 0, full size justified.
    rec_frac = 1.0 if worst5 >= 0.3 else round(max(0.3, 0.3 / max(base_sh, 1e-6)), 2)
    kill_dd = round(1.75 * base_dd, 0)   # halt if cum DD breaches 1.75x in-sample maxDD

    out = {"chosen_config": "equal-weight cap-1/3 (24h 6-sleeve)",
           "in_sample_sharpe": round(base_sh, 3),
           "in_sample_maxDD_bps": round(base_dd, 0),
           "dropk": dk,
           "group_disjoint_no_symid": gd,
           "recommended_deploy_fraction": rec_frac,
           "kill_switch_maxDD_bps": kill_dd,
           "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "R3_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  chosen={out['chosen_config']} Sh={base_sh:+.2f} "
          f"maxDD={base_dd:+.0f}", flush=True)
    print(f"  drop5 worst {worst5:+.2f} -> deploy_fraction {rec_frac}, "
          f"kill-switch at cum-DD {kill_dd:+.0f} bps", flush=True)
    print(f"  group-disjoint no-symid pooled alpha-only Sharpe: "
          f"{gd.get('pooled_alpha_only_sharpe')}", flush=True)
    print(f"\nR3 done [{out['elapsed_s']}s]", flush=True)


if __name__ == "__main__":
    main()
