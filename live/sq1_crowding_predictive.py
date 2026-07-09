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
    # squeeze threshold from the incumbent short leg (ranks 1-2) p90, per S1
    X = pd.concat([bb[bb.r <= 2]["fwd24"]]).quantile(0.90)
    sl["sqz"] = (sl["fwd24"] > X).astype(int)
    cr = pd.read_parquet(D / "crowding_panel.parquet")
    cr["open_time"] = pd.to_datetime(cr["open_time"], utc=True)
    sl = sl.merge(cr, on=["symbol", "open_time"], how="left")
    print(f"shortlist rows {len(sl)}, squeeze events {sl.sqz.sum()} ({sl.sqz.mean()*100:.2f}%), "
          f"threshold X={X:+.0f} bps; feat coverage {sl[FEATS].notna().mean().min():.3f}", flush=True)
    return sl

def nested_preds(sl, cols):
    """expanding walk-forward over folds; train prior folds, predict current; min-pos warmup."""
    folds = sorted(sl.fold.unique())
    out = np.full(len(sl), np.nan)
    idx = {f: sl.index[sl.fold == f] for f in folds}
    pos = sl.index.get_indexer  # map label idx->positional
    for k, f in enumerate(folds):
        tr = sl[sl.fold < f]
        if tr.sqz.sum() < WARMUP_POS: continue
        Xtr = tr[cols].to_numpy(); ytr = tr.sqz.to_numpy()
        mu = np.nanmean(Xtr, 0); sdv = np.nanstd(Xtr, 0); sdv[sdv == 0] = 1
        Xtr = np.nan_to_num((Xtr - mu) / sdv)
        m = LogisticRegression(C=1.0, max_iter=200).fit(Xtr, ytr)
        te = sl[sl.fold == f]; Xte = np.nan_to_num((te[cols].to_numpy() - mu) / sdv)
        out[pos(te.index)] = m.predict_proba(Xte)[:, 1]
    return out

def main():
    sl = build_rows()
    sl = sl.reset_index(drop=True)
    sl["fx"] = sl["funding_rate_z_7d"] * sl["funding_dispersion"]   # explicit crowding×regime
    crowd_cols = FEATS + ["fx"]
    # arm 1: crowding
    pc = nested_preds(sl, crowd_cols)
    # arm 2: pred-only baseline (ranker's own base pred — does crowding beat it?)
    pp = nested_preds(sl, ["pred"])
    m = ~np.isnan(pc) & ~np.isnan(pp) & sl.sqz.notna().to_numpy()
    y = sl.sqz.to_numpy()[m]
    auc_c = roc_auc_score(y, pc[m]); auc_p = roc_auc_score(y, pp[m])
    # precision @ top decile of crowding P(squeeze)
    thr = np.quantile(pc[m], 0.9); base = y.mean()
    prec = y[pc[m] >= thr].mean()
    print(f"\nscored rows {m.sum()}, events {int(y.sum())}")
    print(f"crowding OOS AUC {auc_c:.4f} | pred-only AUC {auc_p:.4f} | Δ {auc_c-auc_p:+.4f}")
    print(f"precision@top-decile(crowding) {prec:.3f} vs base rate {base:.3f} (lift {prec/base:.2f}x)")
    # label-permutation null: shuffle sqz WITHIN cycle, refit crowding, 200x
    perm = []
    slp = sl.copy()
    for _ in range(200):
        slp["sqz"] = sl.groupby("open_time")["sqz"].transform(lambda s: rng.permutation(s.values))
        pcp = nested_preds(slp, crowd_cols)
        mm = ~np.isnan(pcp) & slp.sqz.notna().to_numpy()
        try: perm.append(roc_auc_score(slp.sqz.to_numpy()[mm], pcp[mm]))
        except ValueError: pass
    perm = np.array(perm); p95 = np.percentile(perm, 95)
    print(f"permutation-null AUC: mean {perm.mean():.4f} p95 {p95:.4f}  -> crowding {auc_c:.4f} "
          f"{'BEATS p95' if auc_c > p95 else 'WITHIN null'}")
    verdict = ("SIGNAL: crowding beats permutation null AND pred-only" if (auc_c > p95 and auc_c > auc_p)
               else "DEAD: crowding channel closed" if auc_c <= p95
               else "REDUNDANT: beats null but not pred-only (ranker already encodes it)")
    print(f"\nVERDICT: {verdict}")
    print("SQ1DONE")

if __name__ == "__main__":
    main()
