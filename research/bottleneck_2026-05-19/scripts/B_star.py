"""B★ — de-leaked feature-superset, LGBM V3.1 portable (paired Δ A1−A0).

Per PLAN.md v3 (all Round-2 mandatory fixes applied):
  - A0 = WINNER_21 (no sym_id). A1 = A0 + 39 leak-free safe cols from
    results/B_prefeature_ic.txt (denylist + in-script blocking assert
    max|rankIC(feat,target_A)| < 0.10; verified 0.036).
  - A0 and A1 trained in LOCKSTEP: same group-disjoint split (seed 20260519,
    identical to R3c), same folds, same disjoint label, same 5 seeds, same
    held-out rows, same beta-neutral fwd, same full-V3.1 stack — A0/A1 differ
    ONLY in feature columns ⇒ paired Δ is pure feature effect.
  - Decision = paired Δ(A1−A0) portable R3c Sharpe + block-bootstrap CI on the
    per-cycle paired difference + correctly-scaled MDE (not √CPY-doubled).
  - aggTrades arm DROPPED (closed Steps 94b/95/98). Reconciled, not re-run.
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

OUT = REPO / "research/bottleneck_2026-05-19/results"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
G, SEED = 5, 20260519
CPY = R1.CPY


def _sh(x):
    return R1._sharpe(np.asarray(x, float))


def main():
    t0 = time.time()
    # ---- safe superset from the leak-guard evidence file -----------------
    txt = (OUT / "B_prefeature_ic.txt").read_text()
    SAFE = json.loads(txt.split("SAFE_SUPERSET=")[1].strip())
    panel = pd.read_parquet(PANEL)
    base = [f for f in BA.WINNER_21 if f in panel.columns and f != "sym_id"]
    SAFE = [c for c in SAFE if c in panel.columns and c not in base]
    # BLOCKING leak guard (re-assert in-script)
    y = panel["target_A"]; m = y.notna()
    for c in SAFE:
        x = panel[c]; g = m & x.notna()
        ic = abs(np.corrcoef(x[g].rank(), y[g].rank())[0, 1]) if g.sum() > 50000 else 0.0
        assert ic < 0.10, f"LEAK GUARD FAIL: {c} |rankIC|={ic:.4f} >= 0.10"
    A = {"A0": base, "A1": base + SAFE}
    print(f"A0 feat n={len(A['A0'])}; A1 feat n={len(A['A1'])} "
          f"(+{len(SAFE)} safe cols, leak-guard PASS)", flush=True)

    fwd_resid = R3c.build_beta_neutral_fwd()
    sig = pd.read_parquet(R3c.CW)
    sig = sig[[c for c in sig.columns if c.startswith("c_")]].rename(columns=lambda x: x[2:])
    sigma = sig.pct_change().rolling(288, min_periods=48).std().shift(1)
    sigma = sigma.clip(lower=sigma.quantile(0.20), axis=1)
    syms = sorted(panel["symbol"].unique())
    advc = {s: R1.COST_UNIT for s in syms}
    listings = PA.get_listings()
    folds = BA._multi_oos_splits(panel)
    rng = np.random.RandomState(SEED)
    shf = syms.copy(); rng.shuffle(shf)
    groups = [shf[i::G] for i in range(G)]   # identical to R3c

    per = {"A0": {}, "A1": {}}
    pooled = {"A0": [], "A1": []}
    topk = {"A0": [], "A1": []}
    for gi, hold in enumerate(groups):
        hold = set(hold); train_syms = set(syms) - hold
        panel["_tgd"] = R3c.disjoint_target(panel, train_syms).values
        # accumulate predictions per arm, identical rows/seeds
        preds = {"A0": [], "A1": []}
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
            base_d = te[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
            base_d["exit_time"] = (te["exit_time"].values if "exit_time" in te.columns
                                   else te["open_time"] + pd.Timedelta(minutes=R3c.HOR * 5))
            base_d["fold"] = fid
            for arm, feat in A.items():
                Xt = tr[feat].to_numpy(np.float32); Xc = ca[feat].to_numpy(np.float32)
                Xte = te[feat].to_numpy(np.float32)
                ps = []
                for s in BA.SEEDS:                     # identical seeds both arms
                    mdl = BA._train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
                    ps.append(mdl.predict(Xte, num_iteration=mdl.best_iteration))
                d = base_d.copy(); d["pred"] = np.mean(ps, axis=0)
                preds[arm].append(d)
        for arm in ("A0", "A1"):
            if not preds[arm]:
                per[arm][f"g{gi}"] = {"sharpe": None}; continue
            apd = pd.concat(preds[arm], ignore_index=True).sort_values(["open_time", "symbol"])
            apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
            apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
            # pooled top-K(=3) realized-alpha_A spread (harness currency)
            sp = []
            for t, gg in apd.groupby("open_time"):
                if len(gg) < 7: continue
                gg = gg.sort_values("pred")
                sp.append((gg["alpha_A"].iloc[-3:].mean() - gg["alpha_A"].iloc[:3].mean()) * 1e4)
            topk[arm].append(np.array(sp, float))

            def elig(b):
                ts = pd.Timestamp(b, unit="ms", tz="UTC") - pd.Timedelta(days=PA.MIN_HISTORY_DAYS)
                return {s for s in hold if listings.get(s) and listings[s] <= ts}
            tt = sorted(apd[apd["fold"].isin(PA.OOS_FOLDS)]["open_time"].unique())
            if len(tt) < 50:
                per[arm][f"g{gi}"] = {"sharpe": None}; continue
            u = PA.build_rolling_ic_universe(apd, tt[::R1.HE], PA.TOP_N, elig)
            rec = PA.run_production_protocol_save_sleeves(apd, u)
            df, _, _ = R1.aggregate_capped(rec, fwd_resid, sigma, advc,
                                           cap_frac=1/3, sizing="equal", cost_mode="flat45")
            per[arm][f"g{gi}"] = {"sharpe": round(_sh(df["net_pnl_bps"].to_numpy()), 3),
                                  "n_cyc": int(len(df))}
            pooled[arm].append(df[["time", "net_pnl_bps"]].rename(
                columns={"net_pnl_bps": f"net_{arm}"}))
        print(f"  g{gi}: A0 {per['A0'][f'g{gi}'].get('sharpe')} | "
              f"A1 {per['A1'][f'g{gi}'].get('sharpe')}", flush=True)

    res = {"safe_n": len(SAFE), "leak_guard": "PASS (<0.10)"}
    for arm in ("A0", "A1"):
        if pooled[arm]:
            allp = pd.concat([p.set_index("time") for p in pooled[arm]])
            res[f"{arm}_pooled_sharpe"] = round(_sh(allp[f"net_{arm}"].to_numpy()), 3)
        res[f"{arm}_per_group"] = per[arm]
        if topk[arm]:
            res[f"{arm}_topk_spread_bps"] = round(float(np.mean(np.concatenate(topk[arm]))), 2)
    # paired Δ on matched cycles
    if pooled["A0"] and pooled["A1"]:
        a0 = pd.concat([p.set_index("time") for p in pooled["A0"]])
        a1 = pd.concat([p.set_index("time") for p in pooled["A1"]])
        j = a0.join(a1, how="inner")
        diff = (j["net_A1"] - j["net_A0"]).to_numpy()
        d_sh = res["A1_pooled_sharpe"] - res["A0_pooled_sharpe"]
        _, lo, hi = block_bootstrap_ci(diff, statistic=lambda x: float(np.mean(x)),
                                       block_size=R1.BLOCK, n_boot=2000)
        n_eff = len(diff) / R1.BLOCK
        sd = diff.std()
        mde_bps = 2.49 * sd / np.sqrt(max(n_eff, 1))           # correctly-scaled, per-cycle bps
        res["paired"] = {"delta_sharpe": round(d_sh, 3),
                         "mean_diff_bps": round(float(diff.mean()), 3),
                         "diff_ci_bps": [round(float(lo), 3), round(float(hi), 3)],
                         "diff_excludes_0": bool(lo > 0 or hi < 0),
                         "n_eff": round(n_eff, 1),
                         "mde_mean_diff_bps": round(float(mde_bps), 3)}
        # pre-registered verdict
        if d_sh >= 0.5 and (lo > 0 or hi < 0):
            v = "FEATURE-LEVER-REAL"
        elif d_sh <= 0.2 or not (lo > 0 or hi < 0):
            v = "FEATURE-ENGINEERING-EXHAUSTED (earned)"
        else:
            v = "AMBIGUOUS (re-diagnose, gate unchanged)"
        if abs(diff.mean()) < mde_bps and not (lo > 0 or hi < 0):
            v += " | underpowered: effect-size estimate, NOT a false 'exhausted'"
        res["VERDICT"] = v
    res["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "B_star_results.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\n  A0 pooled {res.get('A0_pooled_sharpe')} | A1 pooled "
          f"{res.get('A1_pooled_sharpe')} | Δ {res.get('paired',{}).get('delta_sharpe')} "
          f"CI {res.get('paired',{}).get('diff_ci_bps')}", flush=True)
    print(f"  topK spread bps: A0 {res.get('A0_topk_spread_bps')} "
          f"A1 {res.get('A1_topk_spread_bps')}", flush=True)
    print(f"  VERDICT: {res.get('VERDICT')}", flush=True)
    print(f"B★ done [{res['elapsed_s']}s]", flush=True)


if __name__ == "__main__":
    main()
