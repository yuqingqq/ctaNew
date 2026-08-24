"""Multivariate squeeze-vs-dump model (user: "we are not just gating by high funding? test it fine-tuned").
The honest test is an ABLATION under a HARD cross-era split — does adding features beyond funding actually separate
dump from squeeze OUT OF SAMPLE, ACROSS eras?

  M0  univariate funding gate            (the "trivial" baseline the user is questioning)
  M1  price/vol action only, NO funding  (climax, runup_3d/1d, parab, rvol, dist_ath, taker, age)
  M2  price/vol + funding                (the full both-eras multivariate model)
  M3  + positioning (recent-only)        (smart$/crowd/OI — recent metrics only, so recent time-split not cross-era)

Two heads because fwd_ret is heavy-tailed (squeeze outliers dominate MSE):
  - REG  : winsorized-fwd_ret regressor -> held-out rank-IC(pred, -fwd_ret)   (does it rank dump-risk?)
  - CLS  : dump = 1{fwd_ret <= DUMP_THR} classifier -> held-out AUC           (does it flag dumps?)
Then the money test: short the model's predicted-worst tercile, NET of spread + funding drag, week-clustered CI.
A feature group EARNS its place only if it beats M0 on the SAME held-out era. Shallow, regularized trees; no random
CV (time-ordered -> would leak). Positioning is recent-only, so M3 is a within-recent time split, flagged as such.
"""
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
rng = np.random.default_rng(13); N_FUND = 21; DUMP_THR = -0.20; COST = 0.0040
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    _kw = dict(n_estimators=250, num_leaves=7, learning_rate=0.03, min_child_samples=30,
               subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
    def mkr(): return LGBMRegressor(**_kw)
    def mkc(): return LGBMClassifier(**_kw)
    HAS_LGB = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    def mkr(): return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=250, l2_regularization=5.0, min_samples_leaf=30)
    def mkc(): return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=250, l2_regularization=5.0, min_samples_leaf=30)
    HAS_LGB = False

PRICE = ["climax", "climax_build", "runup_3d", "runup_1d", "parab", "rvol_7d", "dist_ath", "taker", "age_d"]
FUND = ["funding", "funding_chg", "funding_z"]
POS = ["oi_chg", "tt_ls", "ls", "taker_ls"]

def sp(a, b):
    return pd.Series(np.asarray(a, float)).corr(pd.Series(np.asarray(b, float)), method="spearman")

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s); p = s[y == 1]; n = s[y == 0]
    if len(p) == 0 or len(n) == 0: return np.nan
    return (np.subtract.outer(p, n) > 0).mean() + 0.5 * (np.subtract.outer(p, n) == 0).mean()

def wk_boot(t, x):
    x = np.asarray(x, float); ok = ~np.isnan(x); x = x[ok]; t = pd.to_datetime(np.asarray(t)[ok], utc=True)
    if len(x) < 5: return (np.nan, np.nan)
    wk = pd.Series(t).dt.to_period("W").astype(str).values
    grps = [x[wk == w] for w in pd.unique(wk)]
    if len(grps) < 4: return (np.nan, np.nan)
    out = [np.concatenate([grps[i] for i in rng.integers(0, len(grps), len(grps))]).mean() for _ in range(3000)]
    return tuple(np.percentile(out, [2.5, 97.5]))

def prep(df, feats):
    X = df[feats].replace([np.inf, -np.inf], np.nan)
    keep = [f for f in feats if X[f].notna().mean() > 0.5]
    return X[keep], keep

def fit_eval(tr, te, feats, tag, cross_era=True):
    Xtr, keep = prep(tr, feats)
    if not keep: print(f"    [{tag}] no usable features"); return
    med = Xtr.median()
    Xte = te[keep].replace([np.inf, -np.inf], np.nan)
    yr = tr["fwd_ret"].clip(-0.9, 2.0).values                        # winsorize squeeze tail for MSE
    yc = (tr["fwd_ret"].values <= DUMP_THR).astype(int)
    r = mkr().fit(Xtr.fillna(med), yr); pr = r.predict(Xte.fillna(med))
    ic = sp(pr, -te["fwd_ret"].values)
    au = np.nan
    if yc.sum() >= 8 and yc.sum() <= len(yc) - 8:
        c = mkc().fit(Xtr.fillna(med), yc); pc = c.predict_proba(Xte.fillna(med))[:, 1]
        au = auc((te["fwd_ret"].values <= DUMP_THR).astype(int), pc)
    # short the predicted-worst (lowest predicted fwd_ret) tercile, net of cost
    o = te.copy(); o["pred"] = pr
    o["ct"] = pd.qcut(o["pred"].rank(method="first"), 3, labels=["short", "mid", "long"], duplicates="drop")
    s = o[o.ct == "short"].copy(); s["net"] = -s["fwd_ret"] + s["funding"] * N_FUND - COST
    lo, up = wk_boot(s["t"], s["net"].values)
    imp = ""
    if HAS_LGB:
        fi = sorted(zip(keep, r.feature_importances_), key=lambda z: -z[1])[:5]
        imp = " | top: " + ",".join(f"{f}" for f, _ in fi)
    flag = "NET>0 (CI>0)" if lo > 0 else ("NET<0" if up < 0 else "CI~0")
    print(f"    [{tag}] rank-IC {ic:+.3f} | dump-AUC {au:.3f} | short-net {s['net'].mean()*100:+5.1f}% n={len(s)} [wkCI {lo*100:+.1f},{up*100:+.1f}] -> {flag}{imp}")

def baseline(te, tag):
    s = te[te.funding <= te.funding.quantile(1/3)].copy(); s["net"] = -s["fwd_ret"] + s["funding"] * N_FUND - COST
    lo, up = wk_boot(s["t"], s["net"].values)
    ic = sp(-te["funding"].values, -te["fwd_ret"].values)  # does low funding alone rank dump-risk?
    print(f"    [M0 funding-gate {tag}] rank-IC(-funding) {ic:+.3f} | short-net {s['net'].mean()*100:+.1f}% n={len(s)} [wkCI {lo*100:+.1f},{up*100:+.1f}]")

def main():
    e = pd.read_csv(SD / "pump_enriched.csv"); e["t"] = pd.to_datetime(e["t"], utc=True)
    for c in PRICE + FUND + POS:
        if c not in e.columns: e[c] = np.nan
    e = e.dropna(subset=["fwd_ret", "funding"])
    oos = e[e.t < pd.Timestamp("2025-10-01", tz="UTC")].copy()
    rec = e[e.t >= pd.Timestamp("2025-10-01", tz="UTC")].copy()
    dr = (e["fwd_ret"] <= DUMP_THR).mean()
    print(f"enriched {len(e)} | OOS {len(oos)} recent {len(rec)} | dump base-rate {dr*100:.0f}% (fwd_ret<={DUMP_THR}) | LGBM={HAS_LGB}")
    print(f"feature coverage OOS: " + ", ".join(f"{c}:{oos[c].notna().mean()*100:.0f}%" for c in PRICE + FUND))

    pcov_o = ", ".join(f"{c}:{oos[c].notna().mean()*100:.0f}%" for c in POS)
    print(f"positioning coverage OOS: {pcov_o}  (metrics history 2021-24+, NOT recent-only)")

    print("\n### HARD CROSS-ERA A: train OOS(2023-25) -> test RECENT(2025-10+) ###")
    baseline(rec, "->recent")
    fit_eval(oos, rec, PRICE, "M1 price-only    ->recent")
    fit_eval(oos, rec, PRICE + FUND, "M2 price+funding ->recent")
    fit_eval(oos, rec, PRICE + FUND + POS, "M3 +positioning  ->recent")

    print("\n### HARD CROSS-ERA B: train RECENT -> test OOS ###")
    baseline(oos, "->oos")
    fit_eval(rec, oos, PRICE, "M1 price-only    ->oos")
    fit_eval(rec, oos, PRICE + FUND, "M2 price+funding ->oos")
    fit_eval(rec, oos, PRICE + FUND + POS, "M3 +positioning  ->oos")

    # positioning-covered subset only (no median-impute dilution): cross-era on rows with real metrics
    ec = e[e["tt_ls"].notna()].copy()
    oc = ec[ec.t < pd.Timestamp("2025-10-01", tz="UTC")]; rc = ec[ec.t >= pd.Timestamp("2025-10-01", tz="UTC")]
    if len(oc) >= 60 and len(rc) >= 20:
        print(f"\n### CROSS-ERA on positioning-COVERED rows only (OOS {len(oc)} -> recent {len(rc)}, real metrics, no impute) ###")
        baseline(rc, "->recent")
        fit_eval(oc, rc, PRICE + FUND, "M2 price+funding ->recent")
        fit_eval(oc, rc, PRICE + FUND + POS, "M3 +positioning  ->recent")
    print("\nread: a group earns its place only if it beats M0 on the SAME held-out era with CI>0. MODELDONE")

if __name__ == "__main__":
    main()
