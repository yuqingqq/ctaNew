"""SQ1 predictive falsifier (RESEARCH_LOOP_20260707 addenda 14 + 14b).

Does crowding predict the squeeze event OOS at the name level? Nested walk-forward logistic on
short-shortlist (ranks 1-3) rows. Endpoints: OOS AUC, precision@top-decile, label-permutation
null (200x), and INCREMENTAL vs the ranker's own base pred. Falsifier: crowding AUC must beat
both the permutation p95 AND the pred-only AUC, else the free-data crowding channel is closed.
No portfolio decision, no placebo — this is a pure predictive test (dodges the reorder ceiling).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); D = REPO / "live/state/convexity"
rng = np.random.default_rng(14)
FEATS = ["funding_rate_z_7d", "funding_rate_1d_change", "oi_change_z", "toptrader_ls",
         "ls_ratio", "taker_ls", "funding_dispersion"]
WARMUP_POS = 40

def build_rows():
    bb = pd.read_parquet(D / "hl_v4base_oos_clean/v0full_hl60.parquet",
                         columns=["symbol", "open_time", "pred", "alpha_A", "fold"])
    bb["open_time"] = pd.to_datetime(bb["open_time"], utc=True)
    # forward 24h alpha (short-leg outcome), squeeze threshold = incumbent short p90 (S1 frozen)
    pan = pd.read_parquet(REPO / "outputs/vBTC_features/panel_expanded_v0.parquet",
                          columns=["symbol", "open_time", "alpha_vs_btc_realized"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan = pan[(pan.open_time.dt.hour % 4 == 0) & (pan.open_time.dt.minute == 0)].sort_values(["symbol", "open_time"])
    pan["fwd24"] = pan.groupby("symbol")["alpha_vs_btc_realized"].transform(
        lambda s: s.rolling(6).sum().shift(-5)) * 1e4
    bb = bb.merge(pan[["symbol", "open_time", "fwd24"]], on=["symbol", "open_time"], how="left")
    # short shortlist = ranks 1-3 by base pred (lowest) per cycle
    bb["r"] = bb.groupby("open_time")["pred"].rank(method="first")
    sl = bb[bb.r <= 3].dropna(subset=["fwd24"]).copy()
    # squeeze label: PIT PRIOR-FOLD expanding p90 of the incumbent short leg (review F2 fix — the
    # earlier full-OOS quantile was outcome-informed). Per fold, X = p90 over ranks1-2 fwd24 of
    # STRICTLY PRIOR folds; earliest folds (no prior) fall back to the first-3-fold p90 (labeled).
    s12 = bb[bb.r <= 2].dropna(subset=["fwd24"])
    Xf = {}
    for f in sorted(sl.fold.unique()):
        prior = s12[s12.fold < f]["fwd24"]
        Xf[f] = prior.quantile(0.90) if len(prior) >= 500 else s12[s12.fold < 3]["fwd24"].quantile(0.90)
    sl["Xthr"] = sl.fold.map(Xf)
    sl["sqz"] = (sl["fwd24"] > sl["Xthr"]).astype(int)
    cr = pd.read_parquet(D / "crowding_panel.parquet")
    cr["open_time"] = pd.to_datetime(cr["open_time"], utc=True)
    sl = sl.merge(cr, on=["symbol", "open_time"], how="left")
    print(f"shortlist rows {len(sl)}, squeeze events {sl.sqz.sum()} ({sl.sqz.mean()*100:.2f}%), "
          f"PIT prior-fold threshold X range [{sl.Xthr.min():+.0f},{sl.Xthr.max():+.0f}] bps; "
          f"feat coverage {sl[FEATS].notna().mean().min():.3f}", flush=True)
    return sl

def nested_preds(sl, cols):
    """expanding walk-forward over folds; train prior folds, predict current; min-pos warmup.
    Embargo (review F2): drop the last EMB_DAYS of each prior fold's rows whose 24h outcome window
    could overlap the current fold start (folds are monthly cuts; embargo = 1 day)."""
    EMB = pd.Timedelta(days=1)
    folds = sorted(sl.fold.unique())
    out = np.full(len(sl), np.nan)
    idx = {f: sl.index[sl.fold == f] for f in folds}
    pos = sl.index.get_indexer  # map label idx->positional
    fold_start = sl.groupby("fold")["open_time"].min()
    for k, f in enumerate(folds):
        cut = fold_start[f]
        tr = sl[(sl.fold < f) & (sl.open_time < cut - EMB - pd.Timedelta(hours=24))]
        if tr.sqz.sum() < WARMUP_POS: continue
        Xtr = tr[cols].to_numpy(); ytr = tr.sqz.to_numpy()
        mu = np.nanmean(Xtr, 0); sdv = np.nanstd(Xtr, 0); sdv[sdv == 0] = 1
        Xtr = np.nan_to_num((Xtr - mu) / sdv)
        m = LogisticRegression(C=1.0, max_iter=200).fit(Xtr, ytr)
        te = sl[sl.fold == f]; Xte = np.nan_to_num((te[cols].to_numpy() - mu) / sdv)
        out[pos(te.index)] = m.predict_proba(Xte)[:, 1]
    return out

def main():
    """CANONICAL SQ1 test (review F1 fix — the committed harness now produces the headline
    result). PIT prior-fold threshold + embargo (F2). Endpoints: pred-only vs pred+crowding
    incremental AUC; conditional permutation null (shuffle crowding, KEEP pred); drop-one
    attribution; per-fold stability."""
    sl = build_rows().reset_index(drop=True)
    sl["fx"] = sl["funding_rate_z_7d"] * sl["funding_dispersion"]
    crowd = FEATS + ["fx"]
    pp = nested_preds(sl, ["pred"])                    # ranker-only baseline
    pc = nested_preds(sl, ["pred"] + crowd)            # pred + crowding
    m = ~np.isnan(pp) & ~np.isnan(pc) & sl.sqz.notna().to_numpy()
    y = sl.sqz.to_numpy()[m]
    auc_p = roc_auc_score(y, pp[m]); auc_c = roc_auc_score(y, pc[m])
    print(f"\nscored rows {m.sum()}, events {int(y.sum())} ({y.mean()*100:.2f}%)")
    print(f"pred-only AUC {auc_p:.4f} | pred+crowding AUC {auc_c:.4f} | INCREMENT Δ {auc_c-auc_p:+.4f}")
    # conditional permutation null: shuffle crowding features within cycle, KEEP pred, refit, 200x
    perm = []; slp = sl.copy()
    for _ in range(200):
        for c in crowd:
            slp[c] = sl.groupby("open_time")[c].transform(lambda s: rng.permutation(s.values))
        pcp = nested_preds(slp, ["pred"] + crowd)
        mm = ~np.isnan(pcp) & sl.sqz.notna().to_numpy()
        perm.append(roc_auc_score(sl.sqz.to_numpy()[mm], pcp[mm]))
    perm = np.array(perm); p95 = np.percentile(perm, 95)
    print(f"conditional-permutation null (crowding shuffled, pred kept): mean {perm.mean():.4f} "
          f"p95 {p95:.4f}  -> pred+crowding {auc_c:.4f} "
          f"{'ADDS real increment' if auc_c > p95 else 'increment WITHIN null'}")
    # per-fold incremental stability
    sl["pp"] = pp; sl["pc"] = pc
    dr = []
    for f in sorted(sl.fold.unique()):
        d = sl[(sl.fold == f) & sl.pp.notna() & sl.pc.notna()]
        if len(d) < 50 or d.sqz.nunique() < 2: continue
        dr.append(roc_auc_score(d.sqz, d.pc) - roc_auc_score(d.sqz, d.pp))
    dr = np.array(dr)
    print(f"per-fold increment: {(dr>0).sum()}/{len(dr)} positive, mean {dr.mean():+.4f}, "
          f"worst {dr.min():+.4f}, best {dr.max():+.4f}")
    # drop-one attribution (which crowding feature carries the increment)
    print("drop-one (pred+crowd minus each feature; more negative Δ = more important):")
    full = auc_c
    for c in crowd:
        p = nested_preds(sl, ["pred"] + [x for x in crowd if x != c])
        mm = ~np.isnan(p) & sl.sqz.notna().to_numpy()
        a = roc_auc_score(sl.sqz.to_numpy()[mm], p[mm])
        print(f"  without {c:22s}: {a:.4f}  (Δ {a-full:+.4f})")
    verdict = ("SIGNAL: crowding adds real orthogonal increment over pred" if auc_c > p95
               else "NO INCREMENT: crowding redundant to pred")
    print(f"\nVERDICT: {verdict}")
    print("SQ1DONE")

if __name__ == "__main__":
    main()
