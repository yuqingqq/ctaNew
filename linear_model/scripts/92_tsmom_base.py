"""Step 92 — Phase 1 of MOMENTUM_GATE_PLAN (LOCKED contract, run once).

Pre-registered (MOMENTUM_GATE_PLAN.md §5/§7), nothing tuned:
  Universe   : hl42 = on_hl & hl_day_vol_usd>=$2M, minus BIO/VVV/BTC
  Signal     : s_t = ret_asset[t-L,t) - beta_pit_t * ret_btc[t-L,t)
               L = 24h (288 5m bars); ret_* from KLINES close, strictly
               shifted so decision bar t uses only bars <= t-1; beta_pit =
               panel beta_btc_pit (strict-PIT shift-49).
               *** panel return_pct / btc_ret_fwd are FORWARD — NOT used. ***
  Direction  : FIXED convergence/fade  pos_t = -sign(s_t) ∈ {-1,+1}
  Hold       : 4h forward bar at 4h NON-overlapping decision cadence,
               NO sleeve. PnL_i = pos_i * alpha_beta_i (fwd 4h β-residual).
  Sizing     : equal-weight over symbols active that cycle.
               GROSS_bps = mean_i(pos_i*ab_i)*1e4
               COST_bps  = mean_i(|Δpos_i|)*COST_PER_UNIT_ABS_DELTA  (s64;
                           pos∈{-1,1} ⇒ flip |Δ|=2 ⇒ 4.5bps @ VIP-0;
                           first appearance |Δ|=|pos| = entry)
               NET = GROSS - COST. Report GROSS and NET separately.
  Cost       : VIP-0 = s64.COST_PER_UNIT_ABS_DELTA (2.25 bps/unit |Δw|)

PART 1 — MANDATORY PIT AUDIT (Phase 1 strategy does NOT run unless PASS):
  A independent strictly-past recompute of s_t exact-match (≥2 syms;
    corr 1.0, maxdiff ≤ float32-eps)
  B |corr(s_t, forward alpha_beta)| < 0.10 (pooled & worst-symbol) —
    catches gross forward-alignment (a legit predictor is ~0.01-0.05; this
    bar only rejects blatant leak, NOT the signal itself)

PART 2 — pre-registered PASS (ALL, on NET); GROSS reported for the Step-91
decomposition (signal-vs-cost):
  P1 portfolio NET annualized Sharpe block-bootstrap CI excludes 0
  P2 NET Sharpe > matched placebo p95 (150 seeds; placebo = within-symbol
     permutation of the pos sequence — preserves each symbol's turnover &
     sign frequency exactly, destroys timing)
  P3 ≥60% of symbols individually NET-positive AND drop-top-2-symbol
     portfolio still NET-positive
  P4 ≥6/9 OOS folds NET-positive
Autocorrelation diagnostic corr(s_t, fwd αβ) reported (TRANSPARENCY ONLY;
does NOT flip the locked direction).
"""
from __future__ import annotations
import importlib.util, sys, time, warnings, glob
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


s64 = _imp("s64", "linear_model/scripts/64_meanrev_v2_backtest.py")
s59 = s64.s59
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v4_xs import block_bootstrap_ci

PANEL = REPO / "outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet"
HL_MAP = REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv"
KL = REPO / "data/ml/test/parquet/klines"
OUTD = REPO / "linear_model/results/step92_tsmom_base"
OUTD.mkdir(parents=True, exist_ok=True)
L = 288                                  # 24h trailing (5m bars)
BLOCK = 48                               # 4h non-overlap decision cadence
COST = s64.COST                          # COST_PER_UNIT_ABS_DELTA = 2.25
ANN = np.sqrt(365.0 * 6.0)               # 6 non-overlap 4h cycles/day
OOS = list(range(1, 10))
N_PLACEBO = 150


def load_close(sym):
    fs = sorted(glob.glob(str(KL / sym / "5m" / "*.parquet")))
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f, columns=["open_time", "close"])
                   for f in fs], ignore_index=True)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True, errors="coerce")
    return (d.dropna(subset=["open_time"]).drop_duplicates("open_time")
              .sort_values("open_time").reset_index(drop=True))


def trailing_ret_pit(close: pd.Series) -> pd.Series:
    """close[t-1]/close[t-1-L]-1 at row t (strictly past): pct_change(L).shift(1)."""
    return close.pct_change(L).shift(1)


def main():
    print("=" * 96, flush=True)
    print("  STEP 92 — Phase 1 (LOCKED): convergence/fade β-resid momentum, "
          "hl42, L=24h, VIP-0", flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()

    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time", "alpha_beta",
                                          "beta_btc_pit"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    hl = pd.read_csv(HL_MAP)
    keep = set(hl[(hl.on_hl) & (hl.hl_day_vol_usd >= 2e6)]["symbol"])
    syms = sorted(s for s in pan["symbol"].unique()
                  if s in keep and s not in {"BIOUSDT", "VVVUSDT", "BTCUSDT"})
    print(f"  hl42 universe: {len(syms)} symbols", flush=True)

    btc = load_close("BTCUSDT").set_index("open_time")["close"]
    btc_rL = trailing_ret_pit(btc).rename("ret_btc_L")           # PIT BTC trail
    parts = []
    for s in syms:
        c = load_close(s)
        if c is None or len(c) < L + 1000:
            continue
        c = c.set_index("open_time")
        a_rL = trailing_ret_pit(c["close"]).rename("ret_asset_L")
        df = pd.concat([a_rL, btc_rL], axis=1).reset_index()
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

    # ================= PART 1 — MANDATORY PIT AUDIT =================
    print("\n--- PART 1: PIT audit (Phase 1 runs only if PASS) ---", flush=True)
    okA = True
    for s in ["SOLUSDT", "ADAUSDT"]:
        if s not in syms:
            continue
        c = load_close(s).set_index("open_time")["close"]
        ind = (trailing_ret_pit(c) - pan[pan.symbol == s].set_index("open_time")
               ["beta_btc_pit"].reindex(c.index) * trailing_ret_pit(btc).reindex(c.index))
        m = d[d.symbol == s].set_index("open_time")[["s_t"]].join(
            ind.rename("ind")).dropna()
        cc = float(m["s_t"].corr(m["ind"])); md = float((m["s_t"]-m["ind"]).abs().max())
        good = cc > 0.9999 and md < 1e-5
        okA &= good
        print(f"  {s}: corr(stored,indep_PAST)={cc:.6f} maxdiff={md:.1e} "
              f"-> {'OK' if good else 'MISMATCH'}", flush=True)

    oos = d[d["fold"].isin(OOS)]
    grid = sorted(oos["open_time"].unique())[::BLOCK]
    dec = oos[oos["open_time"].isin(set(grid))].copy()
    ic_pool = float(dec["s_t"].corr(dec["alpha_beta"], method="spearman"))
    sym_ic = dec.groupby("symbol").apply(
        lambda g: g["s_t"].corr(g["alpha_beta"], "spearman")
        if len(g) > 30 and g["s_t"].std() > 1e-12 else np.nan).dropna()
    worst = float(sym_ic.abs().max()) if len(sym_ic) else np.nan
    okB = abs(ic_pool) < 0.10 and (np.isnan(worst) or worst < 0.10)
    print(f"  look-ahead IC: pooled corr(s_t,fwd αβ)={ic_pool:+.4f} "
          f"worst-sym |IC|={worst:.4f}  (<0.10 = not a blatant leak; a legit "
          f"predictor is ~0.01-0.05) -> {'OK' if okB else 'SUSPECT'}",
          flush=True)
    if not (okA and okB):
        print("\n  PIT AUDIT: FAIL — Phase 1 strategy NOT run (per contract).",
              flush=True)
        pd.DataFrame([{"audit": "FAIL", "okA": okA, "okB": okB}]).to_csv(
            OUTD / "verdict.csv", index=False)
        return
    print("  PIT AUDIT: PASS", flush=True)

    # ================= PART 2 — Phase 1 strategy =================
    dec = dec.sort_values(["symbol", "open_time"]).copy()
    dec["pos"] = -np.sign(dec["s_t"]).astype(float)            # convergence/fade
    dec.loc[dec["pos"] == 0, "pos"] = 1.0

    def portfolio(frame, poscol="pos"):
        f = frame.sort_values(["symbol", "open_time"]).copy()
        f["dpos"] = f.groupby("symbol")[poscol].diff().abs()
        f["dpos"] = f["dpos"].fillna(f[poscol].abs())          # entry
        f["g"] = f[poscol] * f["alpha_beta"] * 1e4
        f["c"] = f["dpos"] * COST
        port = f.groupby(["open_time", "fold"]).agg(
            gross=("g", "mean"), cost=("c", "mean")).reset_index()
        port["net"] = port["gross"] - port["cost"]
        return port.sort_values("open_time"), f

    port, perf = portfolio(dec)

    def sh(x):
        x = np.asarray(x, float)
        return float(x.mean() / x.std(ddof=1) * ANN) if x.std(ddof=1) > 1e-12 else np.nan

    net = port["net"].to_numpy()
    gsh = sh(port["gross"]); nsh = sh(net)
    lo, hi = block_bootstrap_ci(
        net, statistic=lambda z: z.mean()/z.std(ddof=1)*ANN
        if z.std(ddof=1) > 1e-12 else 0.0, block_size=7, n_boot=1000)[1:]
    fp = sum(1 for _, g in port.groupby("fold") if g["net"].mean() > 0)
    sym_net = perf.groupby("symbol")["net" if False else "g"].mean() - \
        perf.groupby("symbol")["c"].mean()
    sym_pos = float((sym_net > 0).mean())
    top2 = sym_net.abs().sort_values(ascending=False).head(2).index.tolist()
    pd2, _ = portfolio(dec[~dec.symbol.isin(top2)])
    nsh_d2 = sh(pd2["net"])

    # matched placebo: within-symbol permutation of pos (turnover preserved)
    pl = []
    for sd in range(N_PLACEBO):
        rng = np.random.default_rng(sd)
        pp = dec.copy()
        pp["pos"] = pp.groupby("symbol")["pos"].transform(
            lambda v: rng.permutation(v.values))
        pl.append(sh(portfolio(pp)[0]["net"]))
    p95 = float(np.nanpercentile(pl, 95))

    P1 = bool(lo > 0)
    P2 = bool(nsh > p95)
    P3 = bool(sym_pos >= 0.60 and nsh_d2 > 0)
    P4 = bool(fp >= 6)
    allp = P1 and P2 and P3 and P4
    print(f"\n  GROSS Sharpe={gsh:+.2f} ({port['gross'].mean():+.2f} bps/cyc) | "
          f"NET Sharpe={nsh:+.2f} CI[{lo:+.2f},{hi:+.2f}] "
          f"({port['net'].mean():+.2f} bps/cyc)", flush=True)
    print(f"  symbols NET+={sym_pos*100:.0f}%  drop-top2({','.join(top2)}) "
          f"NET Sh={nsh_d2:+.2f}  folds NET+={fp}/9  | placebo p95={p95:+.2f} "
          f"(mean {np.nanmean(pl):+.2f})", flush=True)
    print(f"  [diag, transparency only] autocorr corr(s_t,fwd αβ) pooled "
          f"{ic_pool:+.4f} ⇒ {'reversion-consistent (fade is right sign)' if ic_pool<0 else 'trend-consistent (fade is WRONG sign)'} "
          f"— does NOT flip locked direction", flush=True)

    pd.DataFrame([dict(gross_sh=gsh, net_sh=nsh, ci_lo=lo, ci_hi=hi,
                       gross_bps=port["gross"].mean(), net_bps=port["net"].mean(),
                       sym_pos_frac=sym_pos, drop2_net_sh=nsh_d2, folds_pos=fp,
                       placebo_p95=p95, autocorr=ic_pool,
                       P1=P1, P2=P2, P3=P3, P4=P4, PASS=allp)]).to_csv(
        OUTD / "summary.csv", index=False)
    print("\n" + "=" * 96, flush=True)
    print(f"  GATES: P1 CI-excl-0={P1} | P2 >placebo-p95={P2} | "
          f"P3 ≥60%syms&dropT2={P3} | P4 ≥6/9 folds={P4}", flush=True)
    if allp:
        v = (f"PASS — fixed-convergence β-residual rule has a real, "
             f"parameter-free, net-of-cost, executable pulse: NET Sharpe "
             f"{nsh:+.2f} CI[{lo:+.2f},{hi:+.2f}] > placebo p95 {p95:+.2f}, "
             f"{sym_pos*100:.0f}% syms+, drop-top2 {nsh_d2:+.2f}, {fp}/9 folds. "
             f"HEADLINE RESULT. Enhancements (Phase 1b sleeve / Phase 2 gate) "
             f"now OPTIONAL per plan §6 preconditions.")
    else:
        gross_pos = gsh > 0 and port["gross"].mean() > 0
        tag = (" GROSS-positive but NET fails on cost ⇒ Phase 1b "
               "(cost-amortization) WARRANTED per plan §6." if gross_pos
               else " GROSS ≤ 0 ⇒ no pulse; enhancements moot; the fixed-"
               "convergence thesis has no parameter-free executable edge.")
        v = (f"FAIL (P1={P1},P2={P2},P3={P3},P4={P4}). NET Sharpe {nsh:+.2f} "
             f"CI[{lo:+.2f},{hi:+.2f}], GROSS {gsh:+.2f}, placebo p95 "
             f"{p95:+.2f}.{tag} Recorded honestly; no direction flip "
             f"(anti-p-hack rule). Production LGBM unaffected.")
    print(f"  VERDICT: {v}", flush=True)
    pd.DataFrame([{"PASS": allp, "verdict": v}]).to_csv(
        OUTD / "verdict.csv", index=False)
    print(f"\nSaved {OUTD}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
