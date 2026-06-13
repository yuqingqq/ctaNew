"""Step 94 — D1: INFORMATION CEILING (answers Q1: enough info for trading?).

Pre-registered in docs/INFORMATION_DIAGNOSTIC_PLAN.md, LOCKED before run.

Random 5-fold SHUFFLED CV = information content assuming stationarity:
predicts HELD-OUT rows (no memorization) but interleaves folds in time (no
temporal-transfer penalty). The non-overlapping 4h dec grid ⇒ no
label-window overlap ⇒ random K-fold is leak-free here.

  Panel    : Step-92 dec — hl42 (42 syms), OOS folds 1–9, 4h non-overlap.
  Target   : tz = clip(alpha_beta / sigma_idio, ±5) (cross-symbol learnable);
             positions & scoring use REALIZED alpha_beta (Step-92 units).
  F_core   : 21 strict-PIT panel features + engineered s_t. Excluded leak:
             return_pct, btc_ret_fwd, alpha_beta, exit_time, ids.
  F_core+OI: + 11 PIT OI features on OI∩hl42 (~23 syms) — context, NOT gated.
  Models   : Ridge (linear ceiling) + LGBM (nonlinear ceiling). Ceiling =
             better of the two. pos = sign(oof_pred), equal-weight; exact
             Step-92 portfolio + VIP-0 cost.
  PIT guard: s_t exact-match audit + forward cols excluded + per-feature
             fwd-corr printed (panel is the _full_pit strict build).

PRE-REGISTERED DECISION (locked):
  F_core random-CV NET Sharpe  > +1.5  ⇒ info sufficient ⇒ proceed D2.
                               ≤ +1.5  ⇒ INFORMATION-BOUNDED, STOP
  (rationale: Steps 87–88 in-sample→nested haircut ~2–3× & sign-destroying;
   a stationary ceiling below +1.5 leaves nothing after that haircut).

No strategy adopted. Production LGBM unaffected.
"""
from __future__ import annotations
import importlib.util, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import lightgbm as lgb

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


s92 = _imp("s92", "linear_model/scripts/92_tsmom_base.py")
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v4_xs import block_bootstrap_ci

load_close, trail = s92.load_close, s92.trailing_ret_pit
L, BLOCK, COST, ANN, OOS = s92.L, s92.BLOCK, s92.COST, s92.ANN, s92.OOS
GATE = 1.5                                             # pre-registered
MAKER = COST * (1.0 / 4.5)                             # ≈HL-maker context only
LEAK = {"return_pct", "btc_ret_fwd", "alpha_beta", "exit_time",
        "symbol", "open_time", "fold", "s_t", "tz"}
OI = REPO / "outputs/vBTC_features_oi/oi_panel.parquet"
OUTD = REPO / "linear_model/results/step94_info_ceiling_d1"
OUTD.mkdir(parents=True, exist_ok=True)


def sh(x):
    x = np.asarray(x, float)
    return float(x.mean()/x.std(ddof=1)*ANN) if x.std(ddof=1) > 1e-12 else np.nan


def portfolio(frame, poscol):
    f = frame.sort_values(["symbol", "open_time"]).copy()
    f["dp"] = f.groupby("symbol")[poscol].diff().abs().fillna(f[poscol].abs())
    f["g"] = f[poscol] * f["alpha_beta"] * 1e4
    f["c"] = f["dp"] * COST
    p = f.groupby(["open_time", "fold"]).agg(
        gross=("g", "mean"), cost=("c", "mean")).reset_index()
    p["net"] = p["gross"] - p["cost"]
    return p.sort_values("open_time"), f


def cv_oof(X, y, seed=0):
    """Random 5-fold shuffled OOF preds: Ridge (std) + LGBM."""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    rid = np.zeros(len(X)); gbm = np.zeros(len(X))
    Xv = X.to_numpy(np.float64)
    yv = y.to_numpy(np.float64)
    for tr, te in kf.split(Xv):
        sc = StandardScaler().fit(Xv[tr])
        r = Ridge(alpha=10.0).fit(sc.transform(Xv[tr]), yv[tr])
        rid[te] = r.predict(sc.transform(Xv[te]))
        m = lgb.LGBMRegressor(num_leaves=63, n_estimators=400,
                              learning_rate=0.03, subsample=0.8,
                              colsample_bytree=0.8, random_state=seed,
                              n_jobs=-1, verbose=-1)
        m.fit(Xv[tr], yv[tr])
        gbm[te] = m.predict(Xv[te])
    return rid, gbm


def score(dec, pred, tag):
    f = dec.copy()
    f["pos"] = np.sign(pred)
    f.loc[f["pos"] == 0, "pos"] = 1.0
    port, perf = portfolio(f, "pos")
    net = port["net"].to_numpy()
    g, n = sh(port["gross"]), sh(net)
    lo, hi = block_bootstrap_ci(
        net, statistic=lambda z: z.mean()/z.std(ddof=1)*ANN
        if z.std(ddof=1) > 1e-12 else 0.0, block_size=7, n_boot=1000)[1:]
    fp = sum(1 for _, gg in port.groupby("fold") if gg["net"].mean() > 0)
    sn = perf.groupby("symbol")["g"].mean() - perf.groupby("symbol")["c"].mean()
    sp = float((sn > 0).mean())
    ic = float(pd.Series(pred).corr(dec["alpha_beta"].reset_index(drop=True),
                                    method="spearman"))
    tn = perf.groupby("open_time")["dp"].mean().mean()
    nmk = (port["gross"] - port["cost"] * (MAKER / COST)).to_numpy()
    print(f"  [{tag}] GROSS Sh={g:+.2f} ({port['gross'].mean():+.2f}bps) | "
          f"NET Sh={n:+.2f} CI[{lo:+.2f},{hi:+.2f}] "
          f"({port['net'].mean():+.2f}bps) | IC={ic:+.4f} turn={tn:.3f} "
          f"syms+={sp*100:.0f}% folds+={fp}/9 | NET@maker Sh={sh(nmk):+.2f}",
          flush=True)
    return dict(tag=tag, gross_sh=g, net_sh=n, ci_lo=lo, ci_hi=hi,
                net_bps=port["net"].mean(), ic=ic, turn=tn, sym_pos=sp,
                folds_pos=fp, net_sh_maker=sh(nmk))


def build(universe_oi=False):
    pan = pd.read_parquet(s92.PANEL)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    hl = pd.read_csv(s92.HL_MAP)
    keep = set(hl[(hl.on_hl) & (hl.hl_day_vol_usd >= 2e6)]["symbol"])
    syms = sorted(s for s in pan["symbol"].unique()
                  if s in keep and s not in {"BIOUSDT", "VVVUSDT", "BTCUSDT"})
    btc = load_close("BTCUSDT").set_index("open_time")["close"]
    btc_rL = trail(btc).rename("ret_btc_L")
    parts = []
    for s in syms:
        c = load_close(s)
        if c is None or len(c) < L + 1000:
            continue
        c = c.set_index("open_time")
        df = pd.concat([trail(c["close"]).rename("ret_asset_L"), btc_rL],
                       axis=1).reset_index()
        df["symbol"] = s
        parts.append(df)
    sig = pd.concat(parts, ignore_index=True)
    sig["open_time"] = pd.to_datetime(sig["open_time"], utc=True)
    d = pan.merge(sig, on=["symbol", "open_time"], how="inner")
    d["s_t"] = (d["ret_asset_L"] - d["beta_btc_pit"] * d["ret_btc_L"]).astype("float32")
    if universe_oi:
        oi = pd.read_parquet(OI)
        oi["open_time"] = pd.to_datetime(oi["open_time"], utc=True)
        d = d.merge(oi, on=["symbol", "open_time"], how="inner")
    d = d.dropna(subset=["s_t", "alpha_beta", "sigma_idio"])
    d = d[d["sigma_idio"] > 1e-12].sort_values(
        ["symbol", "open_time"]).reset_index(drop=True)
    folds = _multi_oos_splits(d)
    d["fold"] = -1
    for fid in range(len(folds)):
        d.loc[_slice(d, folds[fid])[2].index, "fold"] = fid
    oos = d[d["fold"].isin(OOS)]
    grid = sorted(oos["open_time"].unique())[::BLOCK]
    dec = oos[oos["open_time"].isin(set(grid))].copy()
    dec["tz"] = (dec["alpha_beta"] / dec["sigma_idio"]).clip(-5, 5)
    return dec, syms, btc, pan


def main():
    print("=" * 96, flush=True)
    print("  STEP 94 — D1 INFORMATION CEILING (random-CV; answers Q1: "
          "enough info for trading?)", flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()
    dec, syms, btc, pan = build(universe_oi=False)

    # ---- PIT guard ----
    okA = True
    for s in ["SOLUSDT", "ADAUSDT"]:
        c = load_close(s).set_index("open_time")["close"]
        bser = pan[pan.symbol == s].set_index("open_time")["beta_btc_pit"]
        ind = trail(c) - bser.reindex(c.index) * trail(btc).reindex(c.index)
        m = dec[dec.symbol == s].set_index("open_time")[["s_t"]].join(
            ind.rename("ind")).dropna()
        cc = float(m["s_t"].corr(m["ind"])) if len(m) else np.nan
        okA &= (cc > 0.9999)
        print(f"  audit {s}: corr(s_t,indep_PAST)={cc:.6f} -> "
              f"{'OK' if cc > 0.9999 else 'MISMATCH'}", flush=True)
    feats = [c for c in dec.columns if c not in LEAK and
             pd.api.types.is_numeric_dtype(dec[c])] + ["s_t"]
    feats = [c for c in dict.fromkeys(feats)]            # uniq, keep s_t
    dec = dec.dropna(subset=feats + ["tz", "alpha_beta"]).reset_index(drop=True)
    fc = (dec[feats].apply(lambda col: col.corr(dec["alpha_beta"],
          method="spearman")).abs().sort_values(ascending=False))
    print(f"  F_core = {len(feats)} feats. |corr(feat, FWD alpha_beta)| "
          f"max={fc.iloc[0]:.3f} ({fc.index[0]}); >0.15: "
          f"{[f'{i}={v:.2f}' for i, v in fc[fc > 0.15].items()] or 'none'} "
          f"(transparency; PIT = _full_pit panel + s_t audit + leak excl)",
          flush=True)
    if not okA:
        print("\n  PIT GUARD FAIL — D1 not run.", flush=True)
        pd.DataFrame([{"audit": "FAIL"}]).to_csv(OUTD/"verdict.csv", index=False)
        return
    print(f"  [validate] rows={len(dec)} syms={dec.symbol.nunique()} "
          f"cycles={dec.open_time.nunique()} feats={len(feats)}", flush=True)

    # ---- D1 random-CV ceiling on F_core ----
    print("\n--- F_core (primary, hl42, GATED) ---", flush=True)
    rid, gbm = cv_oof(dec[feats], dec["tz"], seed=0)
    R = [score(dec, rid, "Ridge_randCV"), score(dec, gbm, "LGBM_randCV")]
    # context (not gated)
    score(dec, dec["s_t"].to_numpy() * -1.0, "s_t_rule(Step92 ref)")
    best = max(R, key=lambda r: r["net_sh"])
    PASS = bool(best["net_sh"] > GATE)

    # ---- secondary F_core+OI (context, NOT gated) ----
    print("\n--- F_core+OI (secondary, OI∩hl42, context only) ---", flush=True)
    try:
        deco, _, _, _ = build(universe_oi=True)
        deco = deco.dropna()
        fo = [c for c in deco.columns if c not in LEAK and
              pd.api.types.is_numeric_dtype(deco[c])] + ["s_t"]
        fo = [c for c in dict.fromkeys(fo)]
        deco = deco.dropna(subset=fo + ["tz", "alpha_beta"]).reset_index(drop=True)
        print(f"  rows={len(deco)} syms={deco.symbol.nunique()} "
              f"feats={len(fo)}", flush=True)
        ro, go = cv_oof(deco[fo], deco["tz"], seed=0)
        score(deco, ro, "Ridge+OI_randCV"); score(deco, go, "LGBM+OI_randCV")
    except Exception as e:
        print(f"  (F_core+OI skipped: {e})", flush=True)

    # ---- verdict ----
    if PASS:
        v = (f"D1 PASS — best F_core random-CV NET Sharpe {best['net_sh']:+.2f} "
             f"({best['tag']}) > +1.5 gate. Information IS sufficient under "
             f"stationarity ⇒ Q1=YES; the bottleneck is Q2 "
             f"(utilization/non-stationarity). PROCEED to D2 (time-nested).")
    else:
        v = (f"D1 FAIL — best F_core random-CV NET Sharpe {best['net_sh']:+.2f} "
             f"({best['tag']}, GROSS {best.get('gross_sh','?')}) ≤ +1.5 gate. "
             f"Even a best-case, no-memorization, STATIONARY extraction of "
             f"the current features does NOT clear cost. **Q1 = NO: the line "
             f"is INFORMATION-BOUNDED** — no model/utilization fix can rescue "
             f"it; the limiter is the features' information content on free "
             f"4h perp data, not extraction (Q2) or selection (Q3). This is a "
             f"definitive, stronger result than the prior 'sub-cost' finding. "
             f"D2/D3 moot unless new information is added. NET@maker context: "
             f"{best['net_sh_maker']:+.2f}. Production LGBM unaffected.")
    print(f"\n  PRE-REG GATE (>{GATE:+.1f}): {'PASS' if PASS else 'FAIL'}",
          flush=True)
    print(f"  VERDICT: {v}", flush=True)
    pd.DataFrame(R + [dict(tag="VERDICT", net_sh=best["net_sh"],
                 PASS=PASS, verdict=v)]).to_csv(OUTD/"summary.csv", index=False)
    pd.DataFrame([{"PASS": PASS, "best_net_sh": best["net_sh"],
                   "verdict": v}]).to_csv(OUTD/"verdict.csv", index=False)
    print(f"\nSaved {OUTD}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
