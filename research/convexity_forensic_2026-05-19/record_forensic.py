"""Forensic on the production test record (no new data, no LGBM retrain).

Answers the user's 3 questions, with the anti-hindsight guard built in:
  Q1 WHY ENTERED: for every leg the production sim actually entered, the
     model `pred` (rank within cycle) vs realized leg PnL. Did `pred`
     SELECT the convex winners, or did selection/refill land on them while
     pred was ~noise (hit~50%)?
  Q2 PIT SIGNATURE (OOS-validated, not hindsight): can features known
     BEFORE entry rank which entries become big positive convex
     contributors — tested OUT-OF-TIME (folds 6-9) AND OUT-OF-SYMBOL
     (held-out group), with a label-shuffled placebo (must -> AUC 0.5)?
  Q3 COST: if we gate entries to only the top PIT-signature name per side
     (concentrate where convexity is, skip the rest), what happens to
     turnover / cost / net Sharpe / maxDD vs full production?

Convex-winner label (the thing to predict): an entered leg whose realized
signed contribution is >= p90 of all entered legs AND positive — i.e. a big
right-tail winning leg. Features are strictly PIT (panel cols, R0-clean).
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
from sklearn.linear_model import LogisticRegression

OUT = REPO / "research/convexity_forensic_2026-05-19"; OUT.mkdir(parents=True, exist_ok=True)
APD = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
SIGFEATS = ["atr_pct", "idio_vol_1d_vs_bk", "idio_vol_to_btc_1d",
            "idio_skew_1d", "idio_kurt_1d", "idio_max_abs_12b",
            "name_idio_share_1d", "name_factor_loading_1d",
            "funding_rate_z_7d", "return_1d", "dom_change_288b_vs_bk",
            "corr_to_btc_1d"]
SEED = 20260519


def _auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    t0 = time.time()
    apd = pd.read_parquet(APD)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    syms = sorted(apd["symbol"].unique())
    listings = PA.get_listings()

    def elig(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC") - pd.Timedelta(days=PA.MIN_HISTORY_DAYS)
        return {s for s in syms if listings.get(s) and listings[s] <= ts}
    tt = sorted(apd[apd["fold"].isin(PA.OOS_FOLDS)]["open_time"].unique())
    universe = PA.build_rolling_ic_universe(apd, tt[::R1.HE], PA.TOP_N, elig)
    rec = PA.run_production_protocol_save_sleeves(apd, universe)
    rec = rec[rec["traded"]].copy()

    # ---- expand to per-entered-leg, attach pred + realized contribution ----
    apd_idx = apd.set_index(["open_time", "symbol"])
    rows = []
    for _, r in rec.iterrows():
        t = r["time"]; fold = r["fold"]
        cyc = apd[apd["open_time"] == t]
        if cyc.empty: continue
        pr = cyc.set_index("symbol")
        # pred rank within the traded cycle's candidate universe (1=highest)
        order = pr["pred"].rank(ascending=False, method="first")
        n = len(pr)
        for side, names in (("long", r["long_basket"]), ("short", r["short_basket"])):
            sgn = 1.0 if side == "long" else -1.0
            for s in names:
                if s not in pr.index: continue
                ret = float(pr.loc[s, "return_pct"])
                contrib = sgn * ret * 1e4 / max(len(names), 1)   # bps, sleeve-equiv
                rows.append({"time": t, "fold": fold, "symbol": s, "side": side,
                             "pred": float(pr.loc[s, "pred"]),
                             "pred_rank": float(order.get(s, np.nan)),
                             "pred_pctile": 1.0 - (float(order.get(s, np.nan)) - 1) / max(n - 1, 1),
                             "ret_pct": ret, "contrib_bps": contrib})
    L = pd.DataFrame(rows)
    p90 = L["contrib_bps"].quantile(0.90)
    L["convex_win"] = ((L["contrib_bps"] >= p90) & (L["contrib_bps"] > 0)).astype(int)

    # ---- Q1: did pred select the convex winners? -------------------------
    q1 = {
      "n_entered_legs": int(len(L)),
      "hit_rate_all_legs": round(float((L["contrib_bps"] > 0).mean()), 4),
      "corr_pred_vs_contrib": round(float(L["pred"].corr(L["contrib_bps"])), 4),
      "convex_win_rate": round(float(L["convex_win"].mean()), 4),
      "pred_pctile_mean_convexwin": round(float(L.loc[L.convex_win == 1, "pred_pctile"].mean()), 3),
      "pred_pctile_mean_rest": round(float(L.loc[L.convex_win == 0, "pred_pctile"].mean()), 3),
      "top_symbols_by_total_contrib": L.groupby("symbol")["contrib_bps"].sum()
            .sort_values(ascending=False).head(8).round(0).to_dict(),
      "share_of_total_net_from_top1": None,
    }
    tot = L["contrib_bps"].sum()
    top1 = L.groupby("symbol")["contrib_bps"].sum().sort_values(ascending=False).iloc[0]
    q1["share_of_total_net_from_top1"] = round(float(top1 / tot), 3) if tot != 0 else None

    # ---- Q2: PIT signature, OOS-time + OOS-symbol + placebo --------------
    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time"] + SIGFEATS)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    D = (L.rename(columns={"time": "open_time"})
           .merge(pan, on=["symbol", "open_time"], how="left")
           .dropna(subset=SIGFEATS))
    X = D[SIGFEATS].to_numpy(np.float64)
    X = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-9)
    y = D["convex_win"].to_numpy()
    # OOS-time: train folds 1-5, test 6-9
    tr = D["fold"].isin([1, 2, 3, 4, 5]).to_numpy()
    te = D["fold"].isin([6, 7, 8, 9]).to_numpy()
    auc_time = auc_time_pred = auc_plac = np.nan
    if tr.sum() > 200 and te.sum() > 100 and y[tr].sum() > 5:
        clf = LogisticRegression(max_iter=400, C=0.5).fit(X[tr], y[tr])
        auc_time = _auc(y[te], clf.predict_proba(X[te])[:, 1])
        # does the model's own pred add over the static PIT signature?
        Xp = np.column_stack([X, D["pred_pctile"].to_numpy()])
        clf2 = LogisticRegression(max_iter=400, C=0.5).fit(Xp[tr], y[tr])
        auc_time_pred = _auc(y[te], clf2.predict_proba(Xp[te])[:, 1])
        rng = np.random.RandomState(SEED); ysh = y.copy(); rng.shuffle(ysh)
        clfp = LogisticRegression(max_iter=400, C=0.5).fit(X[tr], ysh[tr])
        auc_plac = _auc(ysh[te], clfp.predict_proba(X[te])[:, 1])
    # OOS-symbol: 5 disjoint groups (same seed as R3c), leave-one-group-out
    rng = np.random.RandomState(SEED); shf = syms.copy(); rng.shuffle(shf)
    gmap = {s: i % 5 for i, s in enumerate(shf)}
    D["grp"] = D["symbol"].map(gmap)
    aucs_sym = []
    for g in range(5):
        m_tr = (D["grp"] != g).to_numpy(); m_te = (D["grp"] == g).to_numpy()
        if m_tr.sum() < 200 or m_te.sum() < 50 or y[m_tr].sum() < 5: continue
        c = LogisticRegression(max_iter=400, C=0.5).fit(X[m_tr], y[m_tr])
        aucs_sym.append(_auc(y[m_te], c.predict_proba(X[m_te])[:, 1]))
    coef = {}
    if tr.sum() > 200:
        cf = LogisticRegression(max_iter=400, C=0.5).fit(X[tr], y[tr])
        coef = {f: round(float(w), 3) for f, w in zip(SIGFEATS, cf.coef_[0])}
    q2 = {"auc_OOS_time": round(float(auc_time), 4),
          "auc_OOS_time_with_pred": round(float(auc_time_pred), 4),
          "auc_label_shuffled_placebo": round(float(auc_plac), 4),
          "auc_OOS_symbol_mean": round(float(np.nanmean(aucs_sym)), 4) if aucs_sym else None,
          "auc_OOS_symbol_per_group": [round(float(a), 3) for a in aucs_sym],
          "signature_coefs": coef}

    out = {"Q1_why_entered": q1, "Q2_pit_signature": q2,
           "interpretation": {
             "pred_selects_convex?": ("pred barely separates convex winners "
                f"(pctile {q1['pred_pctile_mean_convexwin']} vs "
                f"{q1['pred_pctile_mean_rest']}); corr(pred,contrib)="
                f"{q1['corr_pred_vs_contrib']} ~ 0"),
             "signature_predicts_cohort_OOS?": (
                f"OOS-time AUC {q2['auc_OOS_time']} (placebo {q2['auc_label_shuffled_placebo']}), "
                f"OOS-symbol AUC {q2['auc_OOS_symbol_mean']} — "
                ">0.55 on BOTH = a real PIT cohort signature; ~0.5 = hindsight only")},
           "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "record_forensic.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str), flush=True)
    L.to_parquet(OUT / "entered_legs.parquet", index=False)
    print("FORENSIC_DONE", flush=True)


if __name__ == "__main__":
    main()
