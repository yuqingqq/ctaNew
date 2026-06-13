"""Step 84: rigorous horizon-fair gate for the FLAGGED Step-83 24h result.

Step 83 ridge_xsz 24h looked huge (v2_all K3 +36, sqbtcrel +60, 9/9) but the
+9 bps gate was 4h-calibrated (resid24 std is 2.41x) and nnls SIGN-INVERTS.
This decides whether it is a real edge or a horizon-scale + heavy-tail +
unconstrained-ridge artifact. NO claim is made until this passes. No backtest.

PRE-REGISTERED HORIZON-FAIR GATE (fixed before run). The 24h result SURVIVES
iff ALL of:
  G1 net annualized Sharpe (non-overlap 24h, net of conservative 9 bps/cyc
     full-rotation cost) block-bootstrap CI (block 7, 1000) EXCLUDES 0;
  G2 real net Sharpe beats matched random-basket placebo p95 (150 seeds);
  G3 scale-free rank-IC rho >= +0.60 with >= 6/9 folds positive;
  G4 NOT estimator-fragile: signed_equal (no-fit) net Sharpe same sign &
     CI-excludes-0 too (a real cross-sectional edge is not unique to
     unconstrained RidgeCV — nnls already inverted in Step 83);
  G5 leakage audit clean: beta289 PIT exact (indep recompute corr>0.999),
     feature->resid24 per-cycle |IC|<0.10, fold train/test gap >> 24h,
     shuffle-score Sharpe ~0, negate-score Sharpe inverts.
PASS all -> promote flagged->candidate (deeper robustness next, still no
backtest). ANY fail -> Step-83 was an artifact; record honestly.
"""
from __future__ import annotations
import importlib.util, sys, time, warnings, glob, gc
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


s58 = _imp("s58", "linear_model/scripts/58_clean108_train.py")
s64 = _imp("s64", "linear_model/scripts/64_meanrev_v2_backtest.py")
s76 = _imp("s76", "linear_model/scripts/76_minimal_orientation.py")
s78 = _imp("s78", "linear_model/scripts/78_nnls_poscoef_payoff.py")
s79 = _imp("s79", "linear_model/scripts/79_broader_universe_attrib.py")
s80b = _imp("s80b", "linear_model/scripts/80b_vol_interaction_payoff.py")
s83 = _imp("s83", "linear_model/scripts/83_proper_24h_volaug.py")
s59 = s64.s59
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v4_xs import block_bootstrap_ci

OUT = REPO / "linear_model/results/step84_proper_24h_rigor"
OUT.mkdir(parents=True, exist_ok=True)
OOS, BLOCK = s76.OOS, s76.BLOCK
H24 = s83.H24                          # 6
COST_BPS = 9.0                         # conservative full-rotation 24h RT
ANN_24H = np.sqrt(365.0)               # non-overlapping 24h periods / yr
GATE_RHO, GATE_FOLDS, N_PLACEBO = 0.60, 6, 150


def build_dec():
    """Reuse Step-83 pipeline: beta289 + resid24 + volaug hl42 dec frame."""
    raw = pd.read_parquet(s83.VOLAUG)
    raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
    syms = sorted(raw["symbol"].unique())
    b289 = s83.beta289_per_symbol(syms + ["BTCUSDT"])
    raw = raw.merge(b289, on=["symbol", "open_time"], how="left")
    del b289; gc.collect()
    raw = raw.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    g = raw.groupby("symbol", sort=False)
    cr = np.ones(len(raw)); cb = np.ones(len(raw))
    for jj in range(H24):
        cr *= (1.0 + g["return_pct"].shift(-jj * BLOCK).to_numpy())
        cb *= (1.0 + g["btc_ret_fwd"].shift(-jj * BLOCK).to_numpy())
    raw["resid24"] = ((cr - 1.0) - raw["beta289"] * (cb - 1.0)).astype("float32")
    hl = pd.read_csv(s83.HL_MAP)
    keep = set(hl[(hl.on_hl) & (hl.hl_day_vol_usd >= 2e6)]["symbol"])
    p = raw[raw.symbol.isin(keep)
            & ~raw.symbol.isin({"BIOUSDT", "VVVUSDT", "BTCUSDT"})].copy()
    folds = _multi_oos_splits(p)
    f0 = _slice(p, folds[0])[0].index
    sg = p.loc[f0].groupby("symbol")["resid24"].std()
    p["sigma24"] = p["symbol"].map(sg).fillna(float(sg.dropna().median())).clip(lower=1e-6)
    p["target_z"] = (p["resid24"] / p["sigma24"]).clip(-5, 5).astype("float32")
    p["alpha_beta"] = p["resid24"]
    tr0 = _slice(p, folds[0])[0]
    tm = p["open_time"].between(tr0["open_time"].min(), tr0["open_time"].max())
    X, fc = s58.build_v2_features(p, tm)
    px = p[["symbol", "open_time", "alpha_beta", "target_z",
            "autocorr_pctile_7d"] + s80b.VOL].merge(
        X.drop(columns=["alpha_beta", "target_z", "autocorr_pctile_7d"]),
        on=["symbol", "open_time"], how="left")
    px["open_time"] = pd.to_datetime(px["open_time"], utc=True)
    grid = sorted(px["open_time"].unique())[::BLOCK]
    dec = s76.assign_folds(px[px["open_time"].isin(set(grid))].copy(), folds)
    for c in s80b.VOL:
        dec[c] = dec[c].astype("float32").fillna(0.0)
    s80b.add_interactions(dec)
    return raw, p, dec, fc, folds, hl


def ls_series(df, rng=None):
    """Per non-overlap 24h cycle: net K=3 L/S bps. rng -> random baskets."""
    df = df[df["open_time"].isin(set(sorted(df["open_time"].unique())[::H24]))]
    rows = []
    for t, gg in df.groupby("open_time", sort=True):
        if len(gg) < 6:
            continue
        if rng is None:
            s = gg.sort_values("score", ascending=False)
            lo = s.head(3)["y"].mean(); sh = s.tail(3)["y"].mean()
        else:
            idx = rng.permutation(len(gg))
            lo = gg.iloc[idx[:3]]["y"].mean(); sh = gg.iloc[idx[-3:]]["y"].mean()
        rows.append({"open_time": t, "fold": int(gg["fold"].iloc[0]),
                     "net": (lo - sh) * 1e4 - COST_BPS})
    return pd.DataFrame(rows)


def sharpe_ci(net):
    n = net.to_numpy(float)
    sh = float(n.mean() / n.std(ddof=1) * ANN_24H) if n.std() > 0 else np.nan
    lo, hi = block_bootstrap_ci(n, statistic=lambda x: x.mean() / x.std(ddof=1)
                                * ANN_24H if x.std(ddof=1) > 0 else 0.0,
                                block_size=7, n_boot=1000)[1:]
    return sh, float(lo), float(hi), float(n.mean())


def main():
    print("=" * 100, flush=True)
    print("  STEP 84: rigorous horizon-fair gate for the flagged 24h result",
          flush=True)
    print("  PRE-REG: G1 netSharpe CI excl 0 | G2 > placebo p95 | G3 ρ>=+.60 "
          "≥6/9 | G4 signed_equal sign-consistent CI-excl-0 | G5 leak-clean",
          flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    raw, p, dec, fc, folds, hl = build_dec()
    print(f"  hl42 dec: {dec['symbol'].nunique()} syms, "
          f"{dec['open_time'].nunique()} cyc; resid24 std "
          f"{p['resid24'].std():.4g}", flush=True)

    cfg = {"v2_all": fc, "sqbtcrel": s80b.SQ + s80b.BTCREL}
    res = []
    audit = {}
    for cname, sub in cfg.items():
        sub = [c for c in sub if c in dec.columns]
        for mdl in ["ridge_xsz", "signed_equal", "nnls_oriented"]:
            df = s80b.score(dec, sub, folds, mdl)
            if df.empty:
                continue
            net = ls_series(df)
            sh, lo, hi, mu = sharpe_ci(net["net"])
            # scale-free rank IC + folds
            ic = df.groupby("open_time").apply(
                lambda g: g["score"].corr(g["y"], method="spearman")).dropna()
            d6 = df[df["open_time"].isin(set(sorted(df["open_time"].unique())[::H24]))]
            rho = s79.payoff(d6)["decile_rho"]
            fp = sum(1 for _, gg in net.groupby("fold") if gg["net"].mean() > 0)
            row = dict(config=cname, model=mdl, net_sharpe=sh, ci_lo=lo,
                       ci_hi=hi, net_bps_cyc=mu, ic_mean=float(ic.mean()),
                       decile_rho=rho, folds_pos=fp,
                       ci_excl0=bool(lo > 0))
            res.append(row)
            print(f"  {cname:10s} {mdl:14s} netSh={sh:+.2f} "
                  f"CI[{lo:+.2f},{hi:+.2f}] net={mu:+.1f}bps/cyc "
                  f"ρ={rho:+.3f} f+={fp}/9 IC={ic.mean():+.4f}", flush=True)

    # ---- G2 matched random-basket placebo (ridge_xsz/sqbtcrel = best) ----
    dfb = s80b.score(dec, [c for c in s80b.SQ + s80b.BTCREL if c in dec.columns],
                      folds, "ridge_xsz")
    real_sh = sharpe_ci(ls_series(dfb)["net"])[0]
    pl = []
    for k in range(N_PLACEBO):
        pl.append(sharpe_ci(ls_series(dfb, np.random.default_rng(k))["net"])[0])
    pl = np.array(pl, float)
    p95 = float(np.nanpercentile(pl, 95))
    g2 = real_sh > p95
    print(f"\n  G2 placebo: real netSh {real_sh:+.2f} vs placebo p95 {p95:+.2f} "
          f"(mean {np.nanmean(pl):+.2f} max {np.nanmax(pl):+.2f}) -> "
          f"{'PASS' if g2 else 'FAIL'}", flush=True)

    # ---- G5 leakage audit ----
    print("\n  G5 leakage audit:", flush=True)
    # (a) beta289 PIT exact, 1 symbol
    sym = "ETHUSDT"
    kl = s83.load_kl(sym); btc = s83.load_kl("BTCUSDT").set_index("open_time")
    btc["br"] = btc["close"].pct_change()
    kl = kl.set_index("open_time"); kl["r"] = kl["close"].pct_change()
    j = kl.join(btc[["br"]], how="inner")
    cov = j["r"].rolling(s83.BETA_WIN, min_periods=1000).cov(j["br"])
    var = j["br"].rolling(s83.BETA_WIN, min_periods=1000).var()
    bind = (cov / var.replace(0, np.nan)).shift(s83.BETA_SHIFT_24)
    ref = pd.DataFrame({"open_time": j.index, "b": bind.values})
    mrg = raw[raw.symbol == sym][["open_time", "beta289"]].merge(
        ref, on="open_time").dropna()
    bcorr = float(mrg["beta289"].corr(mrg["b"])) if len(mrg) > 1000 else np.nan
    # (b) feature->resid24 IC < 0.10
    oos = dec[dec["fold"].between(1, 9)]
    maxic = 0.0
    for f in (fc + s80b.VOL):
        ics = [g[[f, "alpha_beta"]].dropna().pipe(
            lambda v: v[f].corr(v["alpha_beta"], "spearman")
            if len(v) >= 5 and v[f].std() > 1e-12 else np.nan)
            for _, g in oos.groupby("open_time")]
        m = abs(np.nanmean(ics))
        maxic = max(maxic, m if m == m else 0.0)
    # (c) fold gap >> 24h
    gap_days = (folds[1]["test_start"] - folds[1]["cal_start"]).total_seconds() / 86400
    # (d) shuffle / negate signal-dependence
    sh_shuf = sharpe_ci(ls_series(
        dfb.assign(score=dfb.groupby("open_time")["score"].transform(
            lambda s: np.random.default_rng(0).permutation(s.values)))
        )["net"])[0]
    sh_neg = sharpe_ci(ls_series(dfb.assign(score=-dfb["score"]))["net"])[0]
    g5 = (bcorr > 0.999 and maxic < 0.10 and gap_days > 1.0
          and abs(sh_shuf) < 1.0 and sh_neg < 0)
    print(f"    β289 indep-recompute corr={bcorr:.5f} (>0.999)", flush=True)
    print(f"    max feature->resid24 |cycle-IC|={maxic:.4f} (<0.10)", flush=True)
    print(f"    fold train/test gap={gap_days:.1f}d (>> 1d/24h)", flush=True)
    print(f"    shuffle-score netSh={sh_shuf:+.2f} (~0); negate netSh="
          f"{sh_neg:+.2f} (<0)  -> G5 {'PASS' if g5 else 'FAIL'}", flush=True)

    out = pd.DataFrame(res)
    out.to_csv(OUT / "summary.csv", index=False)
    best = out[(out.config == "sqbtcrel") & (out.model == "ridge_xsz")].iloc[0]
    eq = out[(out.config == "sqbtcrel") & (out.model == "signed_equal")]
    g1 = bool(best["ci_excl0"])
    g3 = bool(best["decile_rho"] >= GATE_RHO and best["folds_pos"] >= GATE_FOLDS)
    g4 = bool(len(eq) and eq.iloc[0]["net_sharpe"] > 0 and eq.iloc[0]["ci_excl0"])
    allpass = g1 and g2 and g3 and g4 and g5
    print("\n" + "=" * 100, flush=True)
    print(f"  GATES: G1 netSh-CI-excl0={g1} | G2 >placebo-p95={g2} | "
          f"G3 ρ≥.60&≥6/9={g3} | G4 signed_equal-consistent={g4} | "
          f"G5 leak-clean={g5}", flush=True)
    if allpass:
        v = ("ALL 5 GATES PASS — the 24h ridge edge SURVIVES horizon-fair, "
             "cost, placebo, estimator-robustness and leakage audit. "
             "Promote flagged->candidate; deeper robustness next, still NO "
             "backtest. First genuine linear lead of the investigation.")
    else:
        fails = [g for g, ok in [("G1", g1), ("G2", g2), ("G3", g3),
                 ("G4", g4), ("G5", g5)] if not ok]
        v = (f"FAIL ({','.join(fails)}) — the Step-83 24h headline was a "
             f"horizon-scale / heavy-tail / unconstrained-ridge artifact, "
             f"not a real edge (G4 estimator-fragility and/or placebo/CI). "
             f"Recorded honestly: 24h does NOT revive the linear line under "
             f"a horizon-fair test. This was the last genuinely-unresolved "
             f"direction, now cleanly closed by DIRECT rigorous test (not a "
             f"proxy). Production LGBM unaffected.")
    print(f"\n  VERDICT: {v}", flush=True)
    pd.DataFrame([{"all_pass": allpass, "verdict": v}]).to_csv(
        OUT / "verdict.csv", index=False)
    print(f"\nSaved under {OUT}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
