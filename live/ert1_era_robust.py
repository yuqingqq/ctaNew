"""ERT1 — era-robust training vs baseline (ERT1_PREREG.md + addenda 23w/23x).

Retrains v4 walk-forward with inverse-regime-frequency sample weights (era-balanced) vs baseline
(uniform), plus N shuffled-regime-weight placebos. Verdict on the TRADED top/bot-20% SELECTION SPREAD
(spread_alpha_bps_mean from _portfolio_pnl) — NOT rank-IC (review 984619c; rank-IC reported diagnostic
only). Everything except the sample weights is v4-identical (imported machinery, pinned hyperparams,
5-seed ensemble, autocorr filter). Gates: E-1 bad-fold spread improves; E-2 mean spread not flattened;
E-3 real bad-fold spread gain beats shuffled-regime p90.
"""
import sys, warnings, glob, time
sys.path.insert(0, "/home/yuqing/ctaNew")
import numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore")
from pathlib import Path
import gc
from features_ml.cross_sectional import (XS_FEATURE_COLS, list_universe, make_xs_alpha_labels,
    build_kline_features, build_basket, add_basket_features, add_engineered_flow_features)
from ml.research.alpha_v4_xs import (_walk_forward_splits, _slice, _portfolio_pnl,
                                     ENSEMBLE_SEEDS, HORIZON, REGIME_CUTOFF)
rng = np.random.default_rng(23)
N_PLACEBO = 20
CACHE = Path("/home/yuqing/ctaNew/data/ml/cache")

def _build_panel_streaming_4h():
    """FULL 213-sym universe (reviewer f3c3996: no universe-cap — weights depend on the full regime
    mix), streamed per-symbol + subsampled to non-overlapping 4h so it fits memory + compute. Faithful
    on universe/weights; the only deviation is 4h non-overlapping training bars (drops 48x overlap
    redundancy — defensible, and NOT the treatment-confounding universe-cap)."""
    universe = list_universe(min_days=200)
    print(f"  universe {len(universe)} syms; pass1 closes...", flush=True)
    closes = {}; bad = 0
    for s in universe:
        p = CACHE/f"xs_feats_{s}.parquet"
        try:
            if p.exists():
                closes[s] = pd.read_parquet(p, columns=["close"])["close"]
            else:
                f = build_kline_features(s); closes[s] = f["close"] if not f.empty else None
        except Exception:
            bad += 1; continue
    closes = pd.DataFrame({s: c for s, c in closes.items() if c is not None}).sort_index()
    if bad: print(f"  (skipped {bad} unreadable caches)", flush=True)
    basket_ret, basket_close = build_basket(closes)
    sym_to_id = {s: i for i, s in enumerate(sorted(closes.columns))}
    print(f"  basket built ({closes.shape[1]} syms, {closes.shape[0]} 5m bars); pass2 enrich+4h-subsample...", flush=True)
    keep = list(set(list(XS_FEATURE_COLS) + ["autocorr_pctile_7d"]))
    frames = []
    for s in closes.columns:
        try:
            f = build_kline_features(s)
        except Exception:
            continue
        if f.empty: continue
        f = f.reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        lab = make_xs_alpha_labels({s: f}, basket_close, HORIZON)[s]
        avail = [c for c in keep if c in f.columns] + (["sym_id"] if "sym_id" in XS_FEATURE_COLS else [])
        df = f[list(set(avail))].join(lab[["demeaned_target","return_pct","alpha_realized","basket_fwd","exit_time"]], how="inner")
        df = df.reset_index().rename(columns={"index": "open_time", df.index.name or "index": "open_time"})
        if "open_time" not in df.columns: df = df.rename(columns={df.columns[0]: "open_time"})
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        # NON-OVERLAPPING 4h grid (hour%4==0, minute==0) — matches v4 replay cadence
        m = (df["open_time"].dt.hour % 4 == 0) & (df["open_time"].dt.minute == 0)
        df = df[m]
        df["symbol"] = s
        for c in df.select_dtypes("float64").columns: df[c] = df[c].astype("float32")
        frames.append(df); del f;
    del closes, basket_ret, basket_close; gc.collect()
    panel = pd.concat(frames, ignore_index=True, sort=False).dropna(subset=["autocorr_pctile_7d","demeaned_target"])
    del frames; gc.collect()
    return panel

def expanding_folds(panel, test_days=120, cal_days=10, embargo=1.0, start="2022-01-01", min_train_days=365):
    """EXPANDING walk-forward (pre-reg): _slice trains on ALL history < cal_start (multi-regime, so
    era-balancing is a real treatment — unlike 50-day rolling where each window is ~single-regime).
    Test windows tile 2022->end; 2022 folds descriptive, 2023-26 gated."""
    ds, de = panel["open_time"].min(), panel["open_time"].max()
    emb = pd.Timedelta(days=embargo); ts = pd.Timestamp(start, tz="UTC"); folds=[]; fid=0
    while True:
        test_start = ts; test_end = test_start + pd.Timedelta(days=test_days)
        if test_end > de: break
        cal_end = test_start - emb; cal_start = cal_end - pd.Timedelta(days=cal_days)
        if (cal_start - ds).days >= min_train_days:
            folds.append(dict(fid=fid, cal_start=cal_start, cal_end=cal_end, test_start=test_start,
                              test_end=test_end, embargo=emb, train_start=ds, train_end=cal_start)); fid+=1
        ts = test_end
    return folds

def _train_w(X, y, Xc, yc, seed, weight=None):
    """Byte-identical to alpha_v4_xs._train except optional train-row weight (cal stays unweighted)."""
    params = dict(objective="regression", metric="rmse", learning_rate=0.03,
        num_leaves=63, max_depth=8, min_data_in_leaf=100, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5, lambda_l2=3.0, verbose=-1,
        seed=seed, feature_fraction_seed=seed, bagging_seed=seed, data_random_seed=seed)
    dtr = lgb.Dataset(X, y, weight=weight, free_raw_data=False)
    dc = lgb.Dataset(Xc, yc, reference=dtr, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dc],
                     callbacks=[lgb.early_stopping(stopping_rounds=80), lgb.log_evaluation(period=0)])

def btc_ret_30d_daily():
    fs = sorted(glob.glob("/home/yuqing/ctaNew/data/ml/test/parquet/klines/BTCUSDT/5m/*.parquet"))
    b = pd.concat([pd.read_parquet(f, columns=["open_time","close"]) for f in fs], ignore_index=True)
    b["open_time"] = pd.to_datetime(b["open_time"], utc=True)
    b = b.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")["close"]
    d = b.resample("1D").last()
    return (d/d.shift(30) - 1).rename("btc30").reset_index()

def regime_bucket(x):
    if not np.isfinite(x): return "side"
    if x < -0.10: return "bear"
    if x >= 0.15: return "deepbull"
    if x >= 0.10: return "bull"
    return "side"

def inv_freq_weights(buckets):
    vc = pd.Series(buckets).value_counts()
    w = np.array([1.0/vc[b] for b in buckets])
    return w/w.mean()   # normalize mean 1

def spread_and_tail(test_f, yt):
    r = _portfolio_pnl(test_f, yt, top_frac=0.2)
    if r.get("n_bars",0)==0: return None
    df = r["df"]; tail = np.percentile(df["spread_alpha_bps"], 5)  # CVaR5-ish traded-spread tail
    return dict(spread=r["spread_alpha_bps_mean"], ic=r["rank_ic_mean"], tail=tail, n=r["n_bars"])

def main():
    t0=time.time()
    PC = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad/ert1_panel_4h.parquet")
    if PC.exists():
        print(f"loading cached 4h panel {PC.name}...", flush=True); panel = pd.read_parquet(PC)
    else:
        print("assembling v4 panel (FULL universe, streamed, 4h-subsampled)...", flush=True)
        panel = _build_panel_streaming_4h(); panel.to_parquet(PC)
    # attach btc_ret_30d regime per bar (PIT merge_asof backward)
    b30 = btc_ret_30d_daily()
    panel = panel.sort_values("open_time")
    panel = pd.merge_asof(panel, b30, on="open_time", direction="backward")
    panel["regime"] = panel["btc30"].map(regime_bucket)
    print(f"  panel {len(panel)} rows, {panel.open_time.nunique()} bars; regime mix:\n{panel.regime.value_counts()}", flush=True)

    folds = expanding_folds(panel)
    print(f"  {len(folds)} expanding folds {folds[0]['test_start'].date()}..{folds[-1]['test_end'].date()}", flush=True)
    rows=[]
    for fold in folds:
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
        cf = cal[cal["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
        if len(tr) < 1000 or len(cf) < 200: continue
        Xtr, ytr = tr[XS_FEATURE_COLS].to_numpy(), tr["demeaned_target"].to_numpy()
        Xc, yc = cf[XS_FEATURE_COLS].to_numpy(), cf["demeaned_target"].to_numpy()
        w = inv_freq_weights(tr["regime"].values)
        Xte = test[XS_FEATURE_COLS].to_numpy()
        def ens_pred(weight):  # 5-seed ensemble, predict at each model's best_iteration (v4-faithful)
            ms = [_train_w(Xtr,ytr,Xc,yc,s,weight) for s in ENSEMBLE_SEEDS]
            return np.mean([m.predict(Xte, num_iteration=m.best_iteration) for m in ms], axis=0)
        yb = ens_pred(None)      # baseline (uniform)
        ye = ens_pred(w)         # ERT (era-balanced)
        rb, re = spread_and_tail(test, yb), spread_and_tail(test, ye)
        if rb is None or re is None: continue
        # placebos: shuffled-regime weights (permute w across rows), 1 seed each (conservative)
        plc=[]
        for i in range(N_PLACEBO):
            wp = rng.permutation(w)
            mp = _train_w(Xtr,ytr,Xc,yc,ENSEMBLE_SEEDS[0],wp)
            rp = spread_and_tail(test, mp.predict(Xte, num_iteration=mp.best_iteration))
            if rp: plc.append(rp["spread"])
        yr = str(pd.Timestamp(fold["test_start"]).year)
        rows.append(dict(fid=fold["fid"], yr=yr, start=str(pd.Timestamp(fold["test_start"]).date()),
                         base=rb["spread"], ert=re["spread"], base_ic=rb["ic"], ert_ic=re["ic"],
                         base_tail=rb["tail"], ert_tail=re["tail"], plc=plc, n=rb["n"]))
        print(f"  fold {fold['fid']} {yr} {rows[-1]['start']}: base spread {rb['spread']:+.2f} | ERT {re['spread']:+.2f} "
              f"(Δ{re['spread']-rb['spread']:+.2f}) | plc mean {np.mean(plc):+.2f} p90 {np.percentile(plc,90):+.2f} "
              f"| IC base {rb['ic']:+.4f} ERT {re['ic']:+.4f} | tail base {rb['tail']:+.1f} ERT {re['tail']:+.1f}", flush=True)

    df = pd.DataFrame(rows)
    oos = df[df.yr >= "2023"].copy()   # gate window (2022 descriptive)
    print(f"\n===== ERT1 GATES (2023-26 folds, n={len(oos)}; 2022 descriptive) =====")
    if len(oos) < 2:
        print("  INSUFFICIENT 2023-26 folds"); print("ERT1DONE"); return
    # bad eras = worst-baseline-spread folds (bottom ~40%)
    k = max(1, int(round(len(oos)*0.4)))
    bad = oos.nsmallest(k, "base")
    e1_gain = (bad.ert - bad.base).mean()
    e1 = e1_gain > 0
    print(f"  E-1 (bad-fold TRADED spread improves): worst-{k} folds base {bad.base.mean():+.2f} -> ERT {bad.ert.mean():+.2f} (Δ{e1_gain:+.2f})  tail base {bad.base_tail.mean():+.1f}->ERT {bad.ert_tail.mean():+.1f}  >> {'PASS' if e1 else 'FAIL'}")
    e2 = oos.ert.mean() >= oos.base.mean() - abs(oos.base.mean())*0.05
    print(f"  E-2 (no flatten): mean spread base {oos.base.mean():+.2f} -> ERT {oos.ert.mean():+.2f}  >> {'PASS' if e2 else 'FAIL'}")
    # E-3 placebo: real bad-fold gain vs shuffled-regime gain p90
    plc_gains=[]
    for _,r in bad.iterrows():
        plc_gains.append(np.array(r.plc) - r.base)
    plc_gain_dist = np.concatenate(plc_gains) if plc_gains else np.array([0.0])
    p90 = np.percentile(plc_gain_dist, 90)
    e3 = e1_gain > p90
    print(f"  E-3 (placebo): real bad-fold gain {e1_gain:+.2f} vs shuffled-regime gain mean {plc_gain_dist.mean():+.2f} p90 {p90:+.2f}  >> {'PASS' if e3 else 'FAIL'}")
    print(f"  concentration per-fold Δspread: {[f'{y}:{d:+.2f}' for y,d in zip(oos.yr, oos.ert-oos.base)]}")
    print(f"  [diagnostic] mean rank-IC base {oos.base_ic.mean():+.4f} -> ERT {oos.ert_ic.mean():+.4f} (NOT gated)")
    y22 = df[df.yr=="2022"]
    if len(y22): print(f"  [2022 descriptive] base spread {y22.base.mean():+.2f} -> ERT {y22.ert.mean():+.2f} (NOT a gate — holdout spent)")
    verdict = "PASS -> candidate (full-stack replay + forward)" if (e1 and e2 and e3) else "FAIL -> era-robust training not adopted"
    print(f"\n  >>> ERT1 VERDICT (E-1 ∧ E-2 ∧ E-3): {verdict}")
    print(f"  (elapsed {time.time()-t0:.0f}s)")
    print("ERT1DONE")

if __name__ == "__main__":
    main()
