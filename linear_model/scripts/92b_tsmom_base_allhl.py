"""Step 92b — Phase 1 on the PRE-REGISTERED ALTERNATIVE universe (all-on_hl).

NOT a retry of Step 92. hl42 (the locked PRIMARY universe) FAILED the
pre-registered gate — that verdict is FINAL. This runs the §7-U1 alternative
("all-on_hl ≈70", drop the $2M floor) purely as a ROBUSTNESS / BREADTH
diagnostic: is the hl42 failure composition-specific to those 42, or
structural across the broader executable set?

Identical locked contract to Step 92 (reuses its audited code): same s_t
(klines, strict-PIT, panel forward fields excluded), same convergence/fade
`pos=-sign(s_t)`, L=24h, 4h hold no-sleeve, VIP-0, mandatory PIT audit,
within-symbol-permutation matched placebo, pre-registered P1-P4. ONLY the
universe filter changes: on_hl == True (no volume floor), minus BTC.

Honest prior: GUARDED-NEGATIVE. The extra ~28 names are <$2M-vol (less
liquid; VIP-0 cost model under-charges them). Step 79 already showed this
exact broadening (hl_all≈70) DEGRADES vs hl42 for the cross-sectional
residual. Most likely confirms/broadens the negative. Multiple-comparison
rule: a pass here (2nd universe after primary fail) is NOT a clean win —
would need clearly-dominate + own-placebo + explicit caveat. Both universes
reported regardless; no cherry-pick.
"""
from __future__ import annotations
import importlib.util, sys, time, warnings
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


s92 = _imp("s92", "linear_model/scripts/92_tsmom_base.py")
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v4_xs import block_bootstrap_ci

load_close, trail = s92.load_close, s92.trailing_ret_pit
L, BLOCK, COST, ANN, OOS, N_PLACEBO = (s92.L, s92.BLOCK, s92.COST, s92.ANN,
                                       s92.OOS, s92.N_PLACEBO)
OUTD = REPO / "linear_model/results/step92b_tsmom_allhl"
OUTD.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 96, flush=True)
    print("  STEP 92b — Phase 1 ALTERNATIVE universe (all-on_hl) — ROBUSTNESS "
          "check, NOT a retry (hl42 FAILED, final)", flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()
    pan = pd.read_parquet(s92.PANEL, columns=["symbol", "open_time",
                                              "alpha_beta", "beta_btc_pit"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    hl = pd.read_csv(s92.HL_MAP)
    keep = set(hl[hl.on_hl]["symbol"])                 # ONLY change: no $2M floor
    syms = sorted(s for s in pan["symbol"].unique()
                  if s in keep and s != "BTCUSDT")
    print(f"  all-on_hl universe: {len(syms)} symbols "
          f"(vs hl42=42; +{len(syms)-42} lower-liquidity HL names)", flush=True)

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
    d = d.dropna(subset=["s_t", "alpha_beta"]).sort_values(
        ["symbol", "open_time"]).reset_index(drop=True)
    folds = _multi_oos_splits(d)
    d["fold"] = -1
    for fid in range(len(folds)):
        te = _slice(d, folds[fid])[2]
        d.loc[te.index, "fold"] = fid
    n_eff = d["symbol"].nunique()
    print(f"  effective (klines+data): {n_eff} symbols", flush=True)

    # ---- PIT audit (same gate as Step 92) ----
    okA = True
    for s in ["SOLUSDT", "ADAUSDT"]:
        if s not in syms:
            continue
        c = load_close(s).set_index("open_time")["close"]
        bser = pan[pan.symbol == s].set_index("open_time")["beta_btc_pit"]
        ind = trail(c) - bser.reindex(c.index) * trail(btc).reindex(c.index)
        m = d[d.symbol == s].set_index("open_time")[["s_t"]].join(
            ind.rename("ind")).dropna()
        cc = float(m["s_t"].corr(m["ind"])); md = float((m["s_t"]-m["ind"]).abs().max())
        good = cc > 0.9999 and md < 1e-5
        okA &= good
        print(f"  audit {s}: corr={cc:.6f} maxdiff={md:.1e} "
              f"-> {'OK' if good else 'MISMATCH'}", flush=True)
    oos = d[d["fold"].isin(OOS)]
    grid = sorted(oos["open_time"].unique())[::BLOCK]
    dec = oos[oos["open_time"].isin(set(grid))].copy()
    icp = float(dec["s_t"].corr(dec["alpha_beta"], method="spearman"))
    sic = dec.groupby("symbol").apply(
        lambda g: g["s_t"].corr(g["alpha_beta"], "spearman")
        if len(g) > 30 and g["s_t"].std() > 1e-12 else np.nan).dropna()
    worst = float(sic.abs().max()) if len(sic) else np.nan
    okB = abs(icp) < 0.10 and (np.isnan(worst) or worst < 0.10)
    print(f"  look-ahead IC pooled {icp:+.4f} worst-sym {worst:.4f} "
          f"-> {'OK' if okB else 'SUSPECT'}", flush=True)
    if not (okA and okB):
        print("\n  PIT AUDIT: FAIL — not run.", flush=True)
        pd.DataFrame([{"audit": "FAIL"}]).to_csv(OUTD / "verdict.csv", index=False)
        return
    print("  PIT AUDIT: PASS", flush=True)

    dec = dec.sort_values(["symbol", "open_time"]).copy()
    dec["pos"] = -np.sign(dec["s_t"]).astype(float)
    dec.loc[dec["pos"] == 0, "pos"] = 1.0

    def portfolio(f):
        f = f.sort_values(["symbol", "open_time"]).copy()
        f["dp"] = f.groupby("symbol")["pos"].diff().abs().fillna(f["pos"].abs())
        f["g"] = f["pos"] * f["alpha_beta"] * 1e4
        f["c"] = f["dp"] * COST
        p = f.groupby(["open_time", "fold"]).agg(
            gross=("g", "mean"), cost=("c", "mean")).reset_index()
        p["net"] = p["gross"] - p["cost"]
        return p.sort_values("open_time"), f

    def sh(x):
        x = np.asarray(x, float)
        return float(x.mean()/x.std(ddof=1)*ANN) if x.std(ddof=1) > 1e-12 else np.nan

    port, perf = portfolio(dec)
    net = port["net"].to_numpy()
    gsh, nsh = sh(port["gross"]), sh(net)
    lo, hi = block_bootstrap_ci(
        net, statistic=lambda z: z.mean()/z.std(ddof=1)*ANN
        if z.std(ddof=1) > 1e-12 else 0.0, block_size=7, n_boot=1000)[1:]
    fp = sum(1 for _, g in port.groupby("fold") if g["net"].mean() > 0)
    snet = perf.groupby("symbol")["g"].mean() - perf.groupby("symbol")["c"].mean()
    spos = float((snet > 0).mean())
    t2 = snet.abs().sort_values(ascending=False).head(2).index.tolist()
    pd2 = portfolio(dec[~dec.symbol.isin(t2)])[0]
    nsh2 = sh(pd2["net"])
    pl = []
    for sd in range(N_PLACEBO):
        rng = np.random.default_rng(sd)
        pp = dec.copy()
        pp["pos"] = pp.groupby("symbol")["pos"].transform(
            lambda v: rng.permutation(v.values))
        pl.append(sh(portfolio(pp)[0]["net"]))
    p95 = float(np.nanpercentile(pl, 95))
    P1, P2 = bool(lo > 0), bool(nsh > p95)
    P3, P4 = bool(spos >= 0.60 and nsh2 > 0), bool(fp >= 6)
    allp = P1 and P2 and P3 and P4
    print(f"\n  GROSS Sh={gsh:+.2f} ({port['gross'].mean():+.2f}bps) | "
          f"NET Sh={nsh:+.2f} CI[{lo:+.2f},{hi:+.2f}] "
          f"({port['net'].mean():+.2f}bps)", flush=True)
    print(f"  syms NET+={spos*100:.0f}% dropT2({','.join(t2)}) {nsh2:+.2f} "
          f"folds+={fp}/9 placebo_p95={p95:+.2f} | autocorr {icp:+.4f}",
          flush=True)
    print(f"  GATES P1={P1} P2={P2} P3={P3} P4={P4}", flush=True)
    cmp = (f"vs hl42(primary): GROSS +0.78, NET +0.25 CI[-2.07,+2.47], "
           f"55% syms, 4/9 folds, FAILED.")
    if allp:
        v = (f"all-on_hl PASSES P1-P4 — but this is the 2nd universe after the "
             f"primary (hl42) FAILED → a multiple-comparison, NOT a clean win. "
             f"Requires clearly-dominate + own-placebo + caveat before any "
             f"claim; do NOT treat as validation. {cmp}")
    else:
        v = (f"all-on_hl ALSO FAILS (P1={P1},P2={P2},P3={P3},P4={P4}): NET "
             f"Sh {nsh:+.2f} CI[{lo:+.2f},{hi:+.2f}], {spos*100:.0f}% syms, "
             f"{fp}/9 folds. The hl42 failure is STRUCTURAL, not composition-"
             f"specific — broadening the executable universe does not rescue "
             f"it (consistent with Step-79). Convergence signal is real but "
             f"sub-cost+breadth/fold-fragile across BOTH executable universes. "
             f"{cmp} Production LGBM unaffected.")
    print(f"\n  VERDICT: {v}", flush=True)
    pd.DataFrame([dict(universe="all_on_hl", n_sym=n_eff, gross_sh=gsh,
                       net_sh=nsh, ci_lo=lo, ci_hi=hi, sym_pos=spos,
                       drop2_net=nsh2, folds_pos=fp, placebo_p95=p95,
                       autocorr=icp, P1=P1, P2=P2, P3=P3, P4=P4,
                       PASS=allp, verdict=v)]).to_csv(
        OUTD / "verdict.csv", index=False)
    print(f"\nSaved {OUTD}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
