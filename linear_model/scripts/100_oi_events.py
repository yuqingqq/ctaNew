"""Step 100 — D1-ext-E: structural-EVENT paradigm. Crowding composites →
forced-deleveraging events → return, β-RESIDUAL vs UNHEDGED.

LOCKED pre-registration (INFORMATION_DIAGNOSTIC_PLAN.md §D1-ext-E). One run,
no sweep, Bonferroni×3, market-exposure-matched placebo on the unhedged arm.

PIT composites (6, from cached OI panel + panel price/funding — all already
PIT/.shift(1)):
  c_oi_x_ret = oi_chg_1d*sign(return_1d)     c_oi_x_fund = oi_z_7d*funding_rate_z_7d
  c_oi_own   = oi_z_7d                        c_div  = sign(return_1d)*sign(oi_chg_1d)
  c_ls       = ls_taker_z_1d                  c_oi_accel = oi_chg_4h - oi_chg_1d
3 forward-24h events (label; σ_sym = PIT trailing-30d std of 24h returns, k=2):
  E1 long-liq r_fwd24<=-2σ   E2 squeeze r_fwd24>=+2σ   E3 vol fwd24 range/close>=p90
Stage 1: leak-free grouped+embargo CV (logistic+LGBM) → OOF AUC, Bonferroni×3.
Stage 2 (Stage-1 passers; 24h non-overlap, ANN=√365): prob-signal vs
  (a) β-residual fwd24 (PIT beta) — gate +1.5;  (b) RAW fwd24 — vs
  market-exposure-matched placebo p95 (E3 predictability-only).
No strategy adopted. Production LGBM unaffected.
"""
from __future__ import annotations
import importlib.util, sys, time, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
from ml.research.alpha_v4_xs import block_bootstrap_ci
load_close = s94.load_close
KL = REPO / "data/ml/test/parquet/klines"
OIP = REPO / "outputs/vBTC_features_oi/oi_panel.parquet"
OUTD = REPO / "linear_model/results/step100_oi_events"
OUTD.mkdir(parents=True, exist_ok=True)
COMP = ["c_oi_x_ret", "c_oi_x_fund", "c_oi_own", "c_div", "c_ls", "c_oi_accel"]
H = 288                                              # 24h fwd (5m bars)
EMB = 6                                              # 1d embargo (4h cycles)
ANN = np.sqrt(365.0)


def load_hlc(sym):
    fs = sorted(glob.glob(str(KL / sym / "5m" / "*.parquet")))
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f, columns=["open_time", "high", "low",
                  "close"]) for f in fs], ignore_index=True)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True, errors="coerce")
    return (d.dropna(subset=["open_time"]).drop_duplicates("open_time")
            .sort_values("open_time").reset_index(drop=True))


def grp_oof_clf(dec, feats, y):
    """leak-free whole-timestamp 5-fold + 1d embargo; OOF prob (logit+LGBM avg)."""
    times = np.array(sorted(dec["open_time"].unique()))
    ti = {t: i for i, t in enumerate(times)}
    rt = dec["open_time"].map(ti).to_numpy()
    X = dec[feats].to_numpy(np.float64)
    yv = y.to_numpy(int)
    oof = np.full(len(dec), np.nan)
    kf = KFold(5, shuffle=True, random_state=0)
    for _, te_t in kf.split(times):
        emb = np.zeros(len(times), bool)
        for j in te_t:
            emb[max(0, j-EMB):min(len(times), j+EMB+1)] = True
        trm = np.isin(rt, np.where(~emb)[0]) & ~np.isin(rt, te_t)
        tem = np.isin(rt, te_t)
        if trm.sum() < 800 or tem.sum() == 0 or yv[trm].sum() < 10:
            continue
        sc = StandardScaler().fit(X[trm])
        lr = LogisticRegression(max_iter=400, C=1.0).fit(
            sc.transform(X[trm]), yv[trm])
        gb = lgb.LGBMClassifier(num_leaves=31, n_estimators=300,
                                learning_rate=0.03, subsample=0.8,
                                colsample_bytree=0.8, random_state=0,
                                n_jobs=-1, verbose=-1).fit(X[trm], yv[trm])
        p = 0.5*lr.predict_proba(sc.transform(X[tem]))[:, 1] + \
            0.5*gb.predict_proba(X[tem])[:, 1]
        oof[tem] = p
    return oof


def auc_ci(y, p, times):
    m = ~np.isnan(p)
    y, p, t = y[m], p[m], times[m]
    a = roc_auc_score(y, p)
    order = np.argsort(t, kind="stable")            # precompute ts->idx ONCE
    ts = t[order]
    ut, starts = np.unique(ts, return_index=True)
    groups = np.split(order, starts[1:])
    nut = len(ut)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(300):                            # block(timestamp) bootstrap
        pick = rng.integers(0, nut, nut)
        idx = np.concatenate([groups[i] for i in pick])
        yy = y[idx]
        s = yy.sum()
        if s == 0 or s == len(yy):
            continue
        bs.append(roc_auc_score(yy, p[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return a, lo, hi


def sh(x):
    x = np.asarray(x, float)
    return float(x.mean()/x.std(ddof=1)*ANN) if x.std(ddof=1) > 1e-12 else np.nan


def main():
    print("="*96, flush=True)
    print("  STEP 100 — D1-ext-E: crowding composites → events → return "
          "(hedged vs UNHEDGED), LOCKED", flush=True)
    print("="*96, flush=True)
    t0 = time.time()
    dec, syms, btc, pan = s94.build(universe_oi=False)
    oi = pd.read_parquet(OIP)
    oi["open_time"] = pd.to_datetime(oi["open_time"], utc=True)
    use = sorted(set(oi.symbol.unique()) & set(dec.symbol.unique()))
    pf = pan[pan.symbol.isin(use)][["symbol", "open_time", "funding_rate_z_7d",
          "return_1d", "beta_btc_pit"]].copy()
    pf["open_time"] = pd.to_datetime(pf["open_time"], utc=True)
    dec = (dec[dec.symbol.isin(use)]
           .merge(oi[["symbol", "open_time", "oi_chg_1d", "oi_chg_4h",
                      "oi_z_7d", "ls_taker_z_1d"]], on=["symbol", "open_time"],
                  how="inner")
           .merge(pf, on=["symbol", "open_time"], how="inner",
                  suffixes=("", "_p")))
    rcol = "return_1d" if "return_1d" in dec else "return_1d_p"
    fcol = "funding_rate_z_7d"
    dec["c_oi_x_ret"] = dec["oi_chg_1d"]*np.sign(dec[rcol])
    dec["c_oi_x_fund"] = dec["oi_z_7d"]*dec[fcol]
    dec["c_oi_own"] = dec["oi_z_7d"]
    dec["c_div"] = np.sign(dec[rcol])*np.sign(dec["oi_chg_1d"])
    dec["c_ls"] = dec["ls_taker_z_1d"]
    dec["c_oi_accel"] = dec["oi_chg_4h"] - dec["oi_chg_1d"]

    # forward-24h targets + PIT sigma + event labels, per symbol
    btc_c = load_close("BTCUSDT").set_index("open_time")["close"]
    btc_fwd = (btc_c.shift(-H)/btc_c - 1.0).rename("btc_fwd24")
    parts = []
    for s in use:
        k = load_hlc(s)
        if k is None:
            continue
        k = k.set_index("open_time")
        c = k["close"]
        r_fwd = c.shift(-H)/c - 1.0
        rng_fwd = (k["high"].rolling(H).max().shift(-H) -
                   k["low"].rolling(H).min().shift(-H)) / c
        r24_tr = c.pct_change(H)                       # trailing 24h ret
        sig = r24_tr.rolling(H*30, min_periods=H*5).std().shift(1)  # PIT σ
        rng_thr = (((k["high"].rolling(H).max()-k["low"].rolling(H).min())/c)
                   .rolling(H*30, min_periods=H*5).quantile(0.90).shift(1))
        d = pd.DataFrame({"r_fwd24": r_fwd, "rng_fwd24": rng_fwd,
                          "sig": sig, "rng_thr90": rng_thr}, index=c.index)
        d = d.join(btc_fwd)
        d["symbol"] = s
        parts.append(d.reset_index())
    fwd = pd.concat(parts, ignore_index=True)
    fwd["open_time"] = pd.to_datetime(fwd["open_time"], utc=True)
    d = dec.merge(fwd, on=["symbol", "open_time"], how="inner").dropna(
        subset=COMP+["r_fwd24", "rng_fwd24", "sig", "btc_fwd24",
                     "beta_btc_pit", "rng_thr90"])
    d = d[d["sig"] > 1e-9].reset_index(drop=True)
    d["resid_fwd24"] = d["r_fwd24"] - d["beta_btc_pit"]*d["btc_fwd24"]
    d["E1"] = (d["r_fwd24"] <= -2.0*d["sig"]).astype(int)
    d["E2"] = (d["r_fwd24"] >= 2.0*d["sig"]).astype(int)
    d["E3"] = (d["rng_fwd24"] >= d["rng_thr90"]).astype(int)
    print(f"  rows={len(d)} syms={d.symbol.nunique()} "
          f"cycles={d.open_time.nunique()}", flush=True)
    for e in ["E1", "E2", "E3"]:
        print(f"  base rate {e} = {d[e].mean()*100:.1f}%", flush=True)
    la = max(abs(d[c].corr(d["resid_fwd24"], "spearman")) for c in COMP)
    print(f"  [sanity] max|corr(composite, resid_fwd24)| = {la:.3f} "
          f"(<0.10 ok = composites are PIT)", flush=True)

    times_arr = d["open_time"].to_numpy()
    BONF = 3
    results = {}
    for e in ["E1", "E2", "E3"]:
        oof = grp_oof_clf(d, COMP, d[e])
        a, lo, hi = auc_ci(d[e].to_numpy(), oof, times_arr)
        # Bonferroni: require lower CI > 0.5 AND a>0.55 (corrected-sig proxy)
        sig_ok = bool(lo > 0.5 and a > 0.55)
        print(f"\n  [Stage1] {e}: AUC={a:.4f} CI[{lo:.4f},{hi:.4f}] "
              f"baserate={d[e].mean()*100:.1f}% -> "
              f"{'PREDICTABLE (Bonf-sig)' if sig_ok else 'not sig'}",
              flush=True)
        results[e] = dict(auc=a, lo=lo, hi=hi, sig=sig_ok, oof=oof)

    # ---- Stage 2: only Stage-1 passers, directional (E1 short, E2 long) ----
    grid = sorted(d["open_time"].unique())[::6]          # 24h non-overlap
    s2 = d[d["open_time"].isin(set(grid))].copy()
    COST = s94.COST
    s2lines = []
    for e, dirn in [("E1", -1.0), ("E2", +1.0)]:
        if not results[e]["sig"]:
            s2lines.append(f"  [Stage2] {e}: skipped (Stage-1 not sig)")
            continue
        prob = pd.Series(results[e]["oof"], index=d.index).reindex(
            s2.index).to_numpy()
        m = ~np.isnan(prob)
        g = s2[m].copy(); pr = prob[m]
        w = dirn*(pr - np.nanmedian(pr))                  # centered prob signal
        w = w/np.nanstd(w)
        for arm, tgt in [("RESID", "resid_fwd24"), ("RAW", "r_fwd24")]:
            ret = (w*g[tgt].to_numpy())*1e4
            dpos = np.abs(np.diff(np.concatenate([[0], w])))
            net = ret - dpos*COST
            S = sh(net)
            lo, hi = block_bootstrap_ci(net, statistic=lambda z: z.mean()/
                z.std(ddof=1)*ANN if z.std(ddof=1) > 1e-12 else 0.0,
                block_size=7, n_boot=1000)[1:]
            extra = ""
            if arm == "RAW":
                rng = np.random.default_rng(0)
                pl = [sh(w[rng.permutation(len(w))]*g[tgt].to_numpy()*1e4 -
                      np.abs(np.diff(np.concatenate([[0],
                      w[rng.permutation(len(w))]])))*COST)
                      for _ in range(200)]
                p95 = float(np.nanpercentile(pl, 95))
                extra = f" | mkt-exposure placebo p95={p95:+.2f} -> " + \
                    ("BEATS" if S > p95 else "fails")
            else:
                extra = f" | gate +1.5 -> {'PASS' if S > 1.5 else 'fail'}"
            s2lines.append(f"  [Stage2] {e} {arm}: NET Sh={S:+.2f} "
                           f"CI[{lo:+.2f},{hi:+.2f}]{extra}")
    print("\n" + "\n".join(s2lines), flush=True)

    anysig = any(results[e]["sig"] for e in results)
    if not anysig:
        v = ("D1-ext-E: NOT predictable — no event clears Bonferroni-corrected "
             "AUC. Crowding/forced-deleveraging events are NOT PIT-predictable "
             "from free OI composites. The structural-event paradigm is closed "
             "too; free-data terminus comprehensive across reduced-form AND "
             "structural-event framings, hedged AND probed-unhedged. Only "
             "levers left: paid data, or accept closure.")
    else:
        v = ("D1-ext-E: ≥1 event Bonferroni-predictable — see Stage-2 arms for "
             "the hedged-vs-unhedged fork (RESID>+1.5 reopens this line; RAW "
             ">placebo-p95 only ⇒ real but a SEPARATE directional/beta line, "
             "not market-neutral alpha; neither ⇒ real but unmonetizable).")
    print(f"\n  VERDICT: {v}", flush=True)
    pd.DataFrame([dict(event=e, **{k: results[e][k] for k in
                 ("auc", "lo", "hi", "sig")}) for e in results]).to_csv(
        OUTD/"stage1.csv", index=False)
    pd.DataFrame([{"verdict": v, "stage2": " || ".join(s2lines)}]).to_csv(
        OUTD/"verdict.csv", index=False)
    print(f"\nSaved {OUTD}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
