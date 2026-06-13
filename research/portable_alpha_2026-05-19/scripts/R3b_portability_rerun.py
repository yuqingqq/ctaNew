"""R3(b) re-initiated: group-disjoint (no sym_id) held-out portability.

Original R3 run errored ('alpha_beta not in panel'). Honest re-run with the
correct BTC-beta-neutral residual that IS in the production panel:
`alpha_vs_btc_realized` (fallback `alpha_A`). This is the user's core
question — does a universe-invariant (no sym_id) model trained on a DISJOINT
symbol set generalise to UNSEEN symbols, evaluated beta-neutral (guards the
shared-BTC-factor false positive). G=5 disjoint groups, K=2 spread.
drop-k results from the prior R3 run are unchanged and retained.
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "research/portable_alpha_2026-05-19/scripts"))
import R1_baseline_frontier as R1
import ml.research.alpha_vBTC_build_audit_panel as BA

OUT = REPO / "research/portable_alpha_2026-05-19/results"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
G, SEED = 5, 20260519


def _sh(x):
    return R1._sharpe(np.asarray(x, float))


def main():
    t0 = time.time()
    panel = pd.read_parquet(PANEL)
    ycol = "alpha_vs_btc_realized" if "alpha_vs_btc_realized" in panel.columns else "alpha_A"
    feat = [f for f in BA.WINNER_21 if f in panel.columns and f != "sym_id"]
    folds = BA._multi_oos_splits(panel)
    panel_syms = sorted(panel["symbol"].unique())
    rng = np.random.RandomState(SEED)
    syms = panel_syms.copy(); rng.shuffle(syms)
    groups = [syms[i::G] for i in range(G)]
    print(f"eval target = {ycol}; feat n={len(feat)} (no sym_id); G={G}", flush=True)

    per_group, pooled = {}, []
    for gi, hold in enumerate(groups):
        hold = set(hold); train_syms = set(panel_syms) - hold
        preds = []
        for fid in BA.OOS_FOLDS:
            if fid >= len(folds): continue
            tr, ca, te = BA._slice(panel, folds[fid])
            tr = tr[(tr["autocorr_pctile_7d"] >= BA.THRESHOLD) & (tr["symbol"].isin(train_syms))]
            ca = ca[(ca["autocorr_pctile_7d"] >= BA.THRESHOLD) & (ca["symbol"].isin(train_syms))]
            te = te[te["symbol"].isin(hold)].copy()
            if len(tr) < 1000 or len(ca) < 200 or len(te) < 50: continue
            Xt, yt = tr[feat].to_numpy(np.float32), tr["target_A"].to_numpy(np.float32)
            Xc, yc = ca[feat].to_numpy(np.float32), ca["target_A"].to_numpy(np.float32)
            mt, mc = ~np.isnan(yt), ~np.isnan(yc)
            if mt.sum() < 1000 or mc.sum() < 200: continue
            ps = []
            for s in BA.SEEDS:
                m = BA._train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
                ps.append(m.predict(te[feat].to_numpy(np.float32),
                                    num_iteration=m.best_iteration))
            d = te[["symbol", "open_time", ycol]].copy()
            d["pred"] = np.mean(ps, axis=0)
            preds.append(d)
        if not preds:
            per_group[f"g{gi}"] = {"n_held": len(hold), "sharpe": None}; continue
        pp = pd.concat(preds, ignore_index=True).dropna(subset=[ycol])
        cyc = []
        for t, gg in pp.groupby("open_time"):
            if len(gg) < 5: continue
            gg = gg.sort_values("pred")
            cyc.append({"open_time": t,
                        "spread": (gg[ycol].iloc[-2:].mean() - gg[ycol].iloc[:2].mean()) * 1e4})
        c = pd.DataFrame(cyc).sort_values("open_time").iloc[::R1.HE]
        s = _sh(c["spread"].to_numpy()) if len(c) > 5 else None
        per_group[f"g{gi}"] = {"n_held": len(hold), "members": sorted(hold),
                               "sharpe": round(s, 3) if s is not None else None,
                               "mean_spread_bps": round(float(c["spread"].mean()), 3) if len(c) else None,
                               "n_cyc": int(len(c))}
        pooled.append(c)
        print(f"  group {gi} (n={len(hold)}): held-out beta-neutral Sharpe "
              f"{per_group[f'g{gi}']['sharpe']}  mean_spread "
              f"{per_group[f'g{gi}']['mean_spread_bps']} bps", flush=True)
    pooled_sh = round(_sh(pd.concat(pooled)["spread"].to_numpy()), 3) if pooled else None
    sharpes = [v["sharpe"] for v in per_group.values() if v.get("sharpe") is not None]
    res = {"eval_target": ycol, "no_sym_id": True,
           "per_group": per_group, "pooled_alpha_only_sharpe": pooled_sh,
           "groups_positive": int(sum(1 for s in sharpes if s > 0)),
           "n_groups_eval": len(sharpes),
           "mean_group_sharpe": round(float(np.mean(sharpes)), 3) if sharpes else None,
           "elapsed_s": round(time.time() - t0, 1)}
    # merge into R3_results.json
    r3 = json.loads((OUT / "R3_results.json").read_text())
    r3["group_disjoint_no_symid"] = res
    (OUT / "R3_results.json").write_text(json.dumps(r3, indent=2, default=str))
    print(f"\n  POOLED beta-neutral held-out (no sym_id) Sharpe: {pooled_sh}; "
          f"{res['groups_positive']}/{res['n_groups_eval']} groups positive; "
          f"mean {res['mean_group_sharpe']}", flush=True)
    print(f"R3b done [{res['elapsed_s']}s]", flush=True)


if __name__ == "__main__":
    main()
