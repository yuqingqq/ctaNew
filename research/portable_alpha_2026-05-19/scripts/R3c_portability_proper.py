"""R3c — re-initiated PROPER portability test (the decisive one).

Round-3 F3/F4: R3b's +1.35 was a costless, gate-free K=2 spread on
`alpha_vs_btc_realized` (NOT the pre-registered residual, NOT the deployable
stack, label not group-disjoint). This is the honest apples-to-apples test of
STATUS Test-3 (no-sym_id full stack = -0.39 in-universe):

  - G=5 DISJOINT symbol groups (seed 20260519, same as R3b).
  - Group-disjoint LABEL: basket = per-timestamp mean return_pct over TRAIN
    symbols only -> alpha_gd -> per-symbol z (expanding-mean.shift(48),
    rolling2016-std.shift(48)). Held-out group's returns never enter the
    training label.
  - Model: WINNER_21 MINUS sym_id (20 feats), 5-seed, exact prod harness,
    trained on train-group rows, predicting held-out (UNSEEN) symbols.
  - FULL deployable stack on the held-out group: rolling-IC universe +
    conv_gate + filter_refill + PM_M2 + K=3 + 6-sleeve + 4.5bps cost.
  - Evaluated on the PRE-REGISTERED residual: trailing-288 PIT beta to BTC
    (.shift(1)), fwd_resid = fwd - beta_pit*fwd_btc. Report regardless of sign.
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
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
CW = REPO / "research/portable_alpha_2026-05-19/results/_cache/close_wide.parquet"
G, SEED, HOR = 5, 20260519, 48


def _sh(x): return R1._sharpe(np.asarray(x, float))


def build_beta_neutral_fwd():
    cw = pd.read_parquet(CW)
    px = cw[[c for c in cw.columns if c.startswith("c_")]].rename(columns=lambda x: x[2:]).sort_index()
    fwd = (px.shift(-HOR) - px) / px
    r5 = px.pct_change()
    btc5 = r5["BTCUSDT"]
    cov = r5.rolling(288).cov(btc5)
    var = btc5.rolling(288).var()
    beta_pit = (cov.div(var, axis=0)).clip(-5, 5).shift(1)   # trailing-288 PIT
    fwd_btc = fwd["BTCUSDT"]
    fwd_resid = fwd.sub(beta_pit.mul(fwd_btc, axis=0))
    return fwd_resid


def disjoint_target(panel, train_syms):
    m = (panel["return_pct"].notna() & panel["atr_pct"].notna()
         & (panel["atr_pct"] > 0) & panel["symbol"].isin(train_syms))
    agg = panel.loc[m].groupby("open_time")["return_pct"].agg(["mean", "count"])
    agg["b"] = np.where(agg["count"] >= 5, agg["mean"], np.nan)
    basket = panel["open_time"].map(agg["b"])
    alpha = (panel["return_pct"] - basket)
    g = alpha.groupby(panel["symbol"], sort=False)
    rmean = g.transform(lambda s: s.expanding(min_periods=288).mean().shift(HOR))
    rstd = g.transform(lambda s: s.rolling(288 * 7, min_periods=288).std().shift(HOR))
    return (alpha - rmean) / rstd.replace(0, np.nan)


def main():
    t0 = time.time()
    panel = pd.read_parquet(PANEL)
    feat = [f for f in BA.WINNER_21 if f in panel.columns and f != "sym_id"]
    folds = BA._multi_oos_splits(panel)
    syms = sorted(panel["symbol"].unique())
    listings = PA.get_listings()
    fwd_resid = build_beta_neutral_fwd()
    sig = pd.read_parquet(CW)
    sig = sig[[c for c in sig.columns if c.startswith("c_")]].rename(columns=lambda x: x[2:])
    sigma = sig.pct_change().rolling(288, min_periods=48).std().shift(1)
    sigma = sigma.clip(lower=sigma.quantile(0.20), axis=1)
    advc = {s: R1.COST_UNIT for s in syms}
    rng = np.random.RandomState(SEED)
    sh = syms.copy(); rng.shuffle(sh)
    groups = [sh[i::G] for i in range(G)]

    per_group, pooled = {}, []
    for gi, hold in enumerate(groups):
        hold = set(hold); train_syms = set(syms) - hold
        tcol = disjoint_target(panel, train_syms)
        panel["_tgd"] = tcol.values
        preds = []
        for fid in BA.OOS_FOLDS:
            if fid >= len(folds): continue
            tr, ca, te = BA._slice(panel, folds[fid])
            tr = tr[(tr["autocorr_pctile_7d"] >= BA.THRESHOLD) & tr["symbol"].isin(train_syms)]
            ca = ca[(ca["autocorr_pctile_7d"] >= BA.THRESHOLD) & ca["symbol"].isin(train_syms)]
            te = te[te["symbol"].isin(hold)].copy()
            if len(tr) < 1000 or len(ca) < 200 or len(te) < 50: continue
            Xt, yt = tr[feat].to_numpy(np.float32), tr["_tgd"].to_numpy(np.float32)
            Xc, yc = ca[feat].to_numpy(np.float32), ca["_tgd"].to_numpy(np.float32)
            mt, mc = ~np.isnan(yt), ~np.isnan(yc)
            if mt.sum() < 1000 or mc.sum() < 200: continue
            ps = []
            for s in BA.SEEDS:
                m = BA._train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
                ps.append(m.predict(te[feat].to_numpy(np.float32),
                                    num_iteration=m.best_iteration))
            d = te[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
            if "exit_time" in te.columns:
                d["exit_time"] = te["exit_time"].values
            else:
                d["exit_time"] = d["open_time"] + pd.Timedelta(minutes=HOR * 5)
            d["pred"] = np.mean(ps, axis=0); d["fold"] = fid
            preds.append(d)
        if not preds:
            per_group[f"g{gi}"] = {"n": len(hold), "sharpe": None}; continue
        apd = pd.concat(preds, ignore_index=True).sort_values(["open_time", "symbol"])
        apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
        apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)

        def elig(b):
            ts = pd.Timestamp(b, unit="ms", tz="UTC") - pd.Timedelta(days=PA.MIN_HISTORY_DAYS)
            return {s for s in hold if listings.get(s) and listings[s] <= ts}
        tt = sorted(apd[apd["fold"].isin(PA.OOS_FOLDS)]["open_time"].unique())
        if len(tt) < 50:
            per_group[f"g{gi}"] = {"n": len(hold), "sharpe": None}; continue
        u = PA.build_rolling_ic_universe(apd, tt[::R1.HE], PA.TOP_N, elig)
        rec = PA.run_production_protocol_save_sleeves(apd, u)
        # FULL deployable stack, evaluated on pre-registered beta-neutral fwd
        df, _, _ = R1.aggregate_capped(rec, fwd_resid, sigma, advc,
                                       cap_frac=1/3, sizing="equal", cost_mode="flat45")
        s = _sh(df["net_pnl_bps"].to_numpy()) if len(df) > 5 else None
        fp = int((df.groupby("fold")["net_pnl_bps"].sum() > 0).sum()) if len(df) else 0
        per_group[f"g{gi}"] = {"n": len(hold), "members": sorted(hold),
                               "sharpe": round(s, 3) if s is not None else None,
                               "folds_pos": fp, "n_cyc": int(len(df)),
                               "totPnL": round(float(df["net_pnl_bps"].sum()), 0) if len(df) else None,
                               "traded_cyc": int((df["net_pnl_bps"] != 0).sum()) if len(df) else 0}
        if len(df):
            pooled.append(df[["time", "net_pnl_bps"]])
        print(f"  g{gi} (n={len(hold)}): FULL-STACK beta-neutral Sharpe "
              f"{per_group[f'g{gi}']['sharpe']} folds+ {fp} "
              f"traded {per_group[f'g{gi}']['traded_cyc']}", flush=True)

    pooled_sh = None
    if pooled:
        allp = pd.concat(pooled)
        pooled_sh = round(_sh(allp["net_pnl_bps"].to_numpy()), 3)
    shs = [v["sharpe"] for v in per_group.values() if v.get("sharpe") is not None]
    res = {"test": "PROPER portability: full deployable stack, no sym_id, "
                    "group-disjoint label+features, trailing-288 PIT-beta-to-BTC "
                    "residual eval, 4.5bps cost, on UNSEEN symbols",
           "per_group": per_group,
           "pooled_fullstack_betaneutral_sharpe": pooled_sh,
           "groups_positive": int(sum(1 for s in shs if s > 0)),
           "n_groups": len(shs),
           "mean_group_sharpe": round(float(np.mean(shs)), 3) if shs else None,
           "std_group_sharpe": round(float(np.std(shs)), 3) if shs else None,
           "vs_STATUS_Test3": "STATUS Test-3 no-sym_id full-stack in-universe = -0.39",
           "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "R3c_portability_proper.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\n  POOLED full-stack beta-neutral (UNSEEN syms, no sym_id) Sharpe: "
          f"{pooled_sh}; {res['groups_positive']}/{res['n_groups']} groups +; "
          f"mean {res['mean_group_sharpe']} std {res['std_group_sharpe']}", flush=True)
    print(f"R3c done [{res['elapsed_s']}s]", flush=True)


if __name__ == "__main__":
    main()
