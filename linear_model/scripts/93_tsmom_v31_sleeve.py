"""Step 93 — Phase 1b of MOMENTUM_GATE_PLAN: V3.1 cost-amortizing sleeve.

CONTRACT (MOMENTUM_GATE_PLAN.md §6 Phase 1b), run once, nothing tuned:
  Run-if   : Phase 1 GROSS-positive (Step 92 hl42 GROSS +0.78 — SATISFIED).
  What     : Phase 1's LOCKED base rule (s_t, convergence/fade pos=-sign(s_t),
             hl42, L=24h, VIP-0) held 24h via the V3.1 equal-weight
             6-overlapping-sleeve at the 4h entry cadence — a FIXED,
             non-tunable discrete construction (like K=3). Sole purpose:
             turnover/cost amortization.
  Adopted  : NET-Sharpe lift >= +0.5 vs Phase-1 primary (hl42 NET +0.25)
             AND P3/P4 not degraded (>=60% syms NET+, drop-top-2 NET+,
             >=6/9 folds NET+) AND still beats matched placebo p95.
             Else: NOT adopted, context only. Not mandatory.

SLEEVE = trailing 6-step (24h) MA of the ±1 convergence position per symbol
on the 4h decision grid:  w_t = mean(pos_{t-5..t}).  6 staggered 4h-entry
sleeves held 24h at 1/6 each is exactly this trailing MA. Applied to the
SAME per-step forward-4h alpha_beta Phase 1 used. NO forward shift anywhere
(only a trailing MA of strictly-PIT positions) ⇒ structurally immune to the
Step-76 24h-shift bug. |Δw|=(1/6)|pos_t-pos_{t-6}| ⇒ turnover = 1/6 of
no-sleeve ⇒ ~6× cost reduction (the contract's amortization mechanism).

Reuses Step-92's audited code (same s_t, universe, folds, PIT audit, matched
within-symbol-permutation placebo). Step 92 kept pristine. Production LGBM
unaffected.
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
SLEEVES = 6                                    # 24h / 4h, fixed (non-tunable)
OUTD = REPO / "linear_model/results/step93_tsmom_v31_sleeve"
OUTD.mkdir(parents=True, exist_ok=True)
P1_BASE = REPO / "linear_model/results/step92_tsmom_base/summary.csv"


def sh(x):
    x = np.asarray(x, float)
    return float(x.mean()/x.std(ddof=1)*ANN) if x.std(ddof=1) > 1e-12 else np.nan


def portfolio(frame, poscol):
    f = frame.sort_values(["symbol", "open_time"]).copy()
    f["dp"] = f.groupby("symbol")[poscol].diff().abs().fillna(
        f[poscol].abs())
    f["g"] = f[poscol] * f["alpha_beta"] * 1e4
    f["c"] = f["dp"] * COST
    p = f.groupby(["open_time", "fold"]).agg(
        gross=("g", "mean"), cost=("c", "mean")).reset_index()
    p["net"] = p["gross"] - p["cost"]
    return p.sort_values("open_time"), f


def add_sleeve(frame, src="pos", dst="w"):
    """w_t = trailing SLEEVES-step MA of src per symbol on the 4h grid
    (equal-weight overlapping sleeves; min_periods=1 = natural ramp)."""
    f = frame.sort_values(["symbol", "open_time"]).copy()
    f[dst] = f.groupby("symbol")[src].transform(
        lambda v: v.rolling(SLEEVES, min_periods=1).mean())
    return f


def main():
    print("=" * 96, flush=True)
    print("  STEP 93 — Phase 1b: V3.1 cost-amortizing 6-sleeve on the LOCKED "
          "Phase-1 rule (hl42)", flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()

    # ---- rebuild Phase-1 d/dec EXACTLY as Step 92 (hl42 locked primary) ----
    pan = pd.read_parquet(s92.PANEL, columns=["symbol", "open_time",
                                              "alpha_beta", "beta_btc_pit"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    hl = pd.read_csv(s92.HL_MAP)
    keep = set(hl[(hl.on_hl) & (hl.hl_day_vol_usd >= 2e6)]["symbol"])
    syms = sorted(s for s in pan["symbol"].unique()
                  if s in keep and s not in {"BIOUSDT", "VVVUSDT", "BTCUSDT"})
    print(f"  hl42 universe: {len(syms)} symbols (Phase-1 locked primary)",
          flush=True)
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
        d.loc[_slice(d, folds[fid])[2].index, "fold"] = fid

    # ---- PIT audit (same gate as Step 92; same s_t) ----
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
        print(f"  audit {s}: corr={cc:.6f} maxdiff={md:.1e} -> "
              f"{'OK' if good else 'MISMATCH'}", flush=True)
    oos = d[d["fold"].isin(OOS)]
    grid = sorted(oos["open_time"].unique())[::BLOCK]
    dec = oos[oos["open_time"].isin(set(grid))].copy()
    icp = float(dec["s_t"].corr(dec["alpha_beta"], method="spearman"))
    okB = abs(icp) < 0.10
    print(f"  look-ahead IC pooled {icp:+.4f} -> {'OK' if okB else 'SUSPECT'}",
          flush=True)
    if not (okA and okB):
        print("\n  PIT AUDIT: FAIL — Phase 1b NOT run.", flush=True)
        pd.DataFrame([{"audit": "FAIL"}]).to_csv(OUTD/"verdict.csv", index=False)
        return
    print("  PIT AUDIT: PASS", flush=True)

    dec = dec.sort_values(["symbol", "open_time"]).copy()
    dec["pos"] = -np.sign(dec["s_t"]).astype(float)
    dec.loc[dec["pos"] == 0, "pos"] = 1.0
    dec = add_sleeve(dec, "pos", "w")

    # ---- intermediate validations (user: verify intermediates) ----
    assert dec["w"].between(-1.0 - 1e-9, 1.0 + 1e-9).all(), "w out of [-1,1]"
    p1port, p1perf = portfolio(dec, "pos")     # Phase-1 reproduction
    svport, svperf = portfolio(dec, "w")       # V3.1 sleeve
    tn_p1 = p1perf.groupby("open_time")["dp"].mean().mean()
    tn_sv = svperf.groupby("open_time")["dp"].mean().mean()
    print(f"\n  [validate] cycles={dec['open_time'].nunique()} "
          f"syms={dec['symbol'].nunique()}", flush=True)
    print(f"  [validate] mean |Δ| per cycle: Phase-1={tn_p1:.4f} "
          f"sleeve={tn_sv:.4f}  ratio={tn_sv/tn_p1:.3f} "
          f"(expect ≈1/6={1/6:.3f} — cost amortization)", flush=True)
    print(f"  [validate] Phase-1 reproduced here: GROSS {p1port['gross'].mean():+.2f} "
          f"NET {p1port['net'].mean():+.2f} bps/cyc Sh {sh(p1port['net']):+.2f} "
          f"(Step-92 hl42 was GROSS +1.29 NET +0.42 Sh +0.25)", flush=True)

    # ---- Phase-1 primary baseline (from Step-92 artifact) ----
    b = pd.read_csv(P1_BASE).iloc[0]
    nsh_p1 = float(b["net_sh"])
    print(f"  Phase-1 primary baseline (Step-92 summary.csv): NET Sh "
          f"{nsh_p1:+.2f}", flush=True)

    # ---- V3.1 sleeve metrics ----
    net = svport["net"].to_numpy()
    gsh, nsh = sh(svport["gross"]), sh(net)
    lo, hi = block_bootstrap_ci(
        net, statistic=lambda z: z.mean()/z.std(ddof=1)*ANN
        if z.std(ddof=1) > 1e-12 else 0.0, block_size=7, n_boot=1000)[1:]
    fp = sum(1 for _, g in svport.groupby("fold") if g["net"].mean() > 0)
    snet = svperf.groupby("symbol")["g"].mean() - svperf.groupby("symbol")["c"].mean()
    spos = float((snet > 0).mean())
    t2 = snet.abs().sort_values(ascending=False).head(2).index.tolist()
    nsh2 = sh(portfolio(add_sleeve(dec[~dec.symbol.isin(t2)], "pos", "w"),
                         "w")[0]["net"])
    pl = []
    for sd in range(N_PLACEBO):
        rng = np.random.default_rng(sd)
        pp = dec.copy()
        pp["pos"] = pp.groupby("symbol")["pos"].transform(
            lambda v: rng.permutation(v.values))
        pp = add_sleeve(pp, "pos", "w")        # placebo gets SAME sleeve smoothing
        pl.append(sh(portfolio(pp, "w")[0]["net"]))
    p95 = float(np.nanpercentile(pl, 95))

    g_bps, n_bps = svport["gross"].mean(), svport["net"].mean()
    c_bps = svport["cost"].mean()
    cg_p1 = abs(p1port["cost"].mean()) / max(abs(p1port["gross"].mean()), 1e-9)
    cg_sv = abs(c_bps) / max(abs(g_bps), 1e-9)
    LIFT = nsh - nsh_p1
    P1 = bool(lo > 0)
    P2 = bool(nsh > p95)
    P3 = bool(spos >= 0.60 and nsh2 > 0)
    P4 = bool(fp >= 6)
    ADOPT = bool(LIFT >= 0.5 and P3 and P4 and nsh > p95)

    print(f"\n  V3.1 SLEEVE: GROSS Sh={gsh:+.2f} ({g_bps:+.2f}bps) | "
          f"NET Sh={nsh:+.2f} CI[{lo:+.2f},{hi:+.2f}] ({n_bps:+.2f}bps) "
          f"cost {c_bps:+.2f}bps", flush=True)
    print(f"  cost/gross: Phase-1={cg_p1*100:.0f}% sleeve={cg_sv*100:.0f}% "
          f"(contract claim ~21%→~12%)", flush=True)
    print(f"  LIFT vs Phase-1 primary = {nsh:+.2f} − ({nsh_p1:+.2f}) = "
          f"{LIFT:+.2f}  (adopt needs ≥ +0.50)", flush=True)
    print(f"  syms NET+={spos*100:.0f}% dropT2({','.join(t2)})={nsh2:+.2f} "
          f"folds+={fp}/9 placebo_p95={p95:+.2f} (mean {np.nanmean(pl):+.2f})",
          flush=True)
    print(f"  GATES: P1(CI>0)={P1} P2(>p95)={P2} P3(breadth)={P3} "
          f"P4(folds)={P4} | LIFT≥0.5={LIFT>=0.5}", flush=True)

    if ADOPT:
        v = (f"Phase 1b ADOPTED: V3.1 sleeve lifts NET Sharpe {nsh_p1:+.2f}→"
             f"{nsh:+.2f} (Δ {LIFT:+.2f} ≥ +0.5) WITHOUT degrading robustness "
             f"({spos*100:.0f}% syms+, dropT2 {nsh2:+.2f}, {fp}/9 folds) and "
             f"beats matched placebo p95 {p95:+.2f}. Cost amortized "
             f"{cg_p1*100:.0f}%→{cg_sv*100:.0f}%. Phase 2 gate now eligible "
             f"per §6 precondition (Phase 1b adopted & NET-passes). One run.")
    else:
        why = []
        if LIFT < 0.5: why.append(f"lift {LIFT:+.2f}<+0.5")
        if not P3: why.append(f"P3 fail ({spos*100:.0f}%syms/dropT2 {nsh2:+.2f})")
        if not P4: why.append(f"P4 fail ({fp}/9 folds)")
        if nsh <= p95: why.append(f"≤placebo p95 {p95:+.2f}")
        v = (f"Phase 1b NOT adopted ({'; '.join(why)}). V3.1 sleeve NET Sh "
             f"{nsh:+.2f} CI[{lo:+.2f},{hi:+.2f}] vs Phase-1 {nsh_p1:+.2f} "
             f"(Δ {LIFT:+.2f}); cost amortized as designed "
             f"({cg_p1*100:.0f}%→{cg_sv*100:.0f}%, |Δ| ratio {tn_sv/tn_p1:.2f}) "
             f"but {'cost was not the only binding constraint — P3/P4 '
             'breadth/fold fragility (cost-independent, confirmed Step-92b) '
             'persists' if (P3 is False or P4 is False) else 'the residual '
             'net edge stays statistically ≈0'}. Phase-1 result stands as the "
             f"headline; gate stays blocked (§6 precondition unmet). No "
             f"direction flip. Production LGBM unaffected.")
    print(f"\n  VERDICT: {v}", flush=True)
    pd.DataFrame([dict(gross_sh=gsh, net_sh=nsh, ci_lo=lo, ci_hi=hi,
                       gross_bps=g_bps, net_bps=n_bps, cost_bps=c_bps,
                       nsh_phase1=nsh_p1, lift=LIFT, sym_pos=spos,
                       drop2_net=nsh2, folds_pos=fp, placebo_p95=p95,
                       cg_phase1=cg_p1, cg_sleeve=cg_sv,
                       turn_ratio=tn_sv/tn_p1, P1=P1, P2=P2, P3=P3, P4=P4,
                       ADOPT=ADOPT, verdict=v)]).to_csv(
        OUTD/"summary.csv", index=False)
    pd.DataFrame([{"ADOPT": ADOPT, "verdict": v}]).to_csv(
        OUTD/"verdict.csv", index=False)
    print(f"\nSaved {OUTD}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
