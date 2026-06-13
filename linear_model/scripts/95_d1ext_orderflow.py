"""Step 95 — D1-extension A: does perp aggTrade ORDER-FLOW lift the
leak-free information ceiling?

Pre-registered (INFORMATION_DIAGNOSTIC_PLAN.md §Status, D1-ext-A): D1 (94b)
proved the perp-OHLCV panel family is information-bounded. This adds a
genuinely-untested FREE family — trade-level order-flow built directly from
perp aggTrades — and re-runs the IDENTICAL leak-free ceiling (whole-timestamp
shuffled 5-fold + 1-day embargo, s94b.grouped_oof) with the SAME +1.5 gate.

Universe: hl42 ∩ aggTrades = 20 liquid majors (the only ones with aggTrades;
same subset-universe situation as the OI panel). Apples-to-apples: F_core
vs F_core+oflow on the SAME 20-sym rows ⇒ the delta isolates order-flow's
marginal information; the +1.5 gate is the pre-registered bar.

Order-flow features (per 5m bar from aggTrades; is_buyer_maker=True ⇒ taker
SELL, False ⇒ taker BUY), all TRAILING + .shift(1) ⇒ strictly PIT (bar t
uses only bars ≤ t-1), mirroring the panel convention:
  of_tfi_z1d  z(trailing 288) of per-bar tfi=(bv-sv)/(bv+sv)
  of_imb_4h   Σ_48(bv-sv)/Σ_48 vol      (signed flow imbalance, 4h)
  of_imb_1d   Σ_288(bv-sv)/Σ_288 vol    (1d)
  of_vol_z7d  z(trailing 2016) of bar taker volume   (flow regime)
  of_kyle_1d  trailing-288 corr(ret5, signed_vol)    (price-impact proxy)
  of_tsz_z1d  z(trailing 288) of avg trade size vol/nt

Mandatory PIT audit (same discipline as Step-92/94): independent
strictly-past recompute of of_tfi_z1d exact-match (SOL/ADA) + per-feature
|corr(feat, FWD alpha_beta)| < 0.10. Cached to outputs/vBTC_features_oflow/.
No strategy adopted. Production LGBM unaffected.
"""
from __future__ import annotations
import importlib.util, sys, time, glob, os, warnings
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


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
s94b = _imp("s94b", "linear_model/scripts/94b_info_ceiling_d1_grouped.py")
build, score, GATE, LEAK = s94.build, s94.score, s94.GATE, s94.LEAK
grouped_oof = s94b.grouped_oof
AGG = REPO / "data/ml/test/parquet/aggTrades"
CACHE = REPO / "outputs/vBTC_features_oflow"
CACHE.mkdir(parents=True, exist_ok=True)
PANEL_F = CACHE / "oflow_panel.parquet"
OUTD = REPO / "linear_model/results/step95_d1ext_orderflow"
OUTD.mkdir(parents=True, exist_ok=True)
OF = ["of_tfi_z1d", "of_imb_4h", "of_imb_1d", "of_vol_z7d", "of_kyle_1d",
      "of_tsz_z1d"]


def _z(x, w):
    m = x.rolling(w, min_periods=max(20, w // 5)).mean()
    s = x.rolling(w, min_periods=max(20, w // 5)).std()
    return ((x - m) / s.replace(0, np.nan))


def build_oflow(syms, d_lo, d_hi):
    """Per-symbol 5m bars from aggTrades → trailing PIT order-flow feats."""
    if PANEL_F.exists():
        p = pd.read_parquet(PANEL_F)
        p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
        if set(syms).issubset(set(p["symbol"].unique())):
            print(f"  [cache] oflow_panel {p.shape} reused", flush=True)
            return p
    rows = []
    for k, s in enumerate(syms):
        t0 = time.time()
        fs = sorted(glob.glob(str(AGG / s / "*.parquet")))
        fs = [f for f in fs if d_lo <= os.path.basename(f)[:-8] <= d_hi]
        bars = []
        for f in fs:
            df = pd.read_parquet(f, columns=["price", "quantity",
                                             "transact_time", "is_buyer_maker"])
            df["bar"] = pd.to_datetime(df["transact_time"], utc=True).dt.floor("5min")
            df["bv"] = np.where(~df["is_buyer_maker"], df["quantity"], 0.0)
            df["sv"] = np.where(df["is_buyer_maker"], df["quantity"], 0.0)
            g = df.groupby("bar").agg(
                bv=("bv", "sum"), sv=("sv", "sum"),
                nt=("price", "size"), close=("price", "last")).reset_index()
            bars.append(g)
        if not bars:
            continue
        b = (pd.concat(bars, ignore_index=True).groupby("bar", as_index=False)
             .agg(bv=("bv", "sum"), sv=("sv", "sum"), nt=("nt", "sum"),
                  close=("close", "last")).sort_values("bar")
             .reset_index(drop=True))
        b["vol"] = b["bv"] + b["sv"]
        b["sig"] = b["bv"] - b["sv"]
        b["tfi"] = b["sig"] / b["vol"].replace(0, np.nan)
        b["ret5"] = b["close"].pct_change()
        b["tsz"] = b["vol"] / b["nt"].replace(0, np.nan)
        # all TRAILING then .shift(1) ⇒ bar t uses only bars ≤ t-1 (PIT)
        b["of_tfi_z1d"] = _z(b["tfi"], 288).shift(1)
        i4 = b["sig"].rolling(48, min_periods=10).sum() / \
            b["vol"].rolling(48, min_periods=10).sum().replace(0, np.nan)
        i1 = b["sig"].rolling(288, min_periods=60).sum() / \
            b["vol"].rolling(288, min_periods=60).sum().replace(0, np.nan)
        b["of_imb_4h"] = i4.shift(1)
        b["of_imb_1d"] = i1.shift(1)
        b["of_vol_z7d"] = _z(b["vol"], 2016).shift(1)
        b["of_kyle_1d"] = (b["ret5"].rolling(288, min_periods=60)
                           .corr(b["sig"])).shift(1)
        b["of_tsz_z1d"] = _z(b["tsz"], 288).shift(1)
        b["symbol"] = s
        rows.append(b[["symbol", "bar"] + OF].rename(columns={"bar": "open_time"}))
        print(f"  [{k+1}/{len(syms)}] {s:12s} bars={len(b)} "
              f"{time.time()-t0:.0f}s", flush=True)
    pan = pd.concat(rows, ignore_index=True)
    pan.to_parquet(PANEL_F, index=False)
    print(f"  built+cached oflow_panel {pan.shape} -> {PANEL_F}", flush=True)
    return pan


def main():
    print("=" * 96, flush=True)
    print("  STEP 95 — D1-ext-A: perp aggTrade ORDER-FLOW vs leak-free "
          "ceiling (same +1.5 gate)", flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()
    dec, syms, btc, pan = build(universe_oi=False)
    have = sorted(s for s in syms
                  if glob.glob(str(AGG / s / "*.parquet")))
    print(f"  universe hl42∩aggTrades = {len(have)} syms", flush=True)
    d_lo = (dec.open_time.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    d_hi = dec.open_time.max().strftime("%Y-%m-%d")
    ofp = build_oflow(have, d_lo, d_hi)

    dec = dec[dec.symbol.isin(have)].copy()
    dec = dec.merge(ofp, on=["symbol", "open_time"], how="inner")
    agg_hi = ofp["open_time"].max()
    dec = dec[dec.open_time <= agg_hi]
    print(f"  merged dec: rows={len(dec)} syms={dec.symbol.nunique()} "
          f"cycles={dec.open_time.nunique()} (clip ≤ {agg_hi.date()})",
          flush=True)

    # ---- PIT audit ----
    ok = True
    for s in ["SOLUSDT", "ADAUSDT"]:
        # independent strictly-past recompute of of_tfi_z1d from raw aggTrades
        fs = sorted(glob.glob(str(AGG / s / "*.parquet")))
        fs = [f for f in fs if d_lo <= os.path.basename(f)[:-8] <= d_hi]
        bb = []
        for f in fs:
            df = pd.read_parquet(f, columns=["price", "quantity",
                                             "transact_time", "is_buyer_maker"])
            df["bar"] = pd.to_datetime(df["transact_time"], utc=True).dt.floor("5min")
            df["bv"] = np.where(~df["is_buyer_maker"], df["quantity"], 0.0)
            df["sv"] = np.where(df["is_buyer_maker"], df["quantity"], 0.0)
            bb.append(df.groupby("bar").agg(bv=("bv", "sum"),
                      sv=("sv", "sum")).reset_index())
        b = (pd.concat(bb).groupby("bar", as_index=False).sum()
             .sort_values("bar").reset_index(drop=True))
        b["tfi"] = (b["bv"]-b["sv"]) / (b["bv"]+b["sv"]).replace(0, np.nan)
        ind = _z(b["tfi"], 288).shift(1)
        b["ind"] = ind.values
        m = dec[dec.symbol == s].merge(
            b[["bar", "ind"]].rename(columns={"bar": "open_time"}),
            on="open_time", how="inner").dropna(subset=["of_tfi_z1d", "ind"])
        cc = float(m["of_tfi_z1d"].corr(m["ind"])) if len(m) > 10 else np.nan
        ok &= (cc > 0.9999)
        print(f"  audit {s}: corr(stored,indep_PAST of_tfi_z1d)={cc:.6f} -> "
              f"{'OK' if cc > 0.9999 else 'MISMATCH'}", flush=True)
    fc = dec[OF].apply(lambda c: c.corr(dec["alpha_beta"], "spearman")).abs()
    print(f"  look-ahead |corr(oflow, FWD αβ)| max={fc.max():.3f} "
          f"({fc.idxmax()}); all<0.10={bool((fc < 0.10).all())}", flush=True)
    if not (ok and (fc < 0.10).all()):
        print("\n  PIT AUDIT FAIL — not run.", flush=True)
        pd.DataFrame([{"audit": "FAIL"}]).to_csv(OUTD/"verdict.csv", index=False)
        return
    print("  PIT AUDIT: PASS", flush=True)

    base = [c for c in dec.columns if c not in LEAK and c not in OF and
            pd.api.types.is_numeric_dtype(dec[c])] + ["s_t"]
    base = list(dict.fromkeys(base))
    dec = dec.dropna(subset=base + OF + ["tz", "alpha_beta"]).reset_index(drop=True)
    print(f"  [validate] rows={len(dec)} syms={dec.symbol.nunique()} "
          f"F_core={len(base)} +oflow={len(OF)}", flush=True)

    def run(feats, lbl):
        rid, gbm = grouped_oof(dec, feats)
        mk = ~np.isnan(rid)
        dd = dec[mk].reset_index(drop=True)
        print(f"\n--- {lbl} (feats={len(feats)}, OOF {mk.mean()*100:.0f}%) ---",
              flush=True)
        return [score(dd, rid[mk], f"Ridge|{lbl}"),
                score(dd, gbm[mk], f"LGBM|{lbl}")]

    R0 = run(base, "F_core (20-sym baseline)")
    R1 = run(base + OF, "F_core+ORDERFLOW (GATED)")
    score(dec, dec["s_t"].to_numpy()*-1.0, "s_t_rule(ref)")
    b0 = max(R0, key=lambda r: r["net_sh"])["net_sh"]
    b1d = max(R1, key=lambda r: r["net_sh"])
    b1 = b1d["net_sh"]
    PASS = bool(b1 > GATE)
    delta = b1 - b0
    if PASS:
        v = (f"D1-ext-A PASS — F_core+orderflow best NET Sharpe {b1:+.2f} "
             f"({b1d['tag']}, IC {b1d['ic']:+.3f}) > +1.5. Order-flow lifts "
             f"the leak-free ceiling (Δ vs same-universe F_core {delta:+.2f}) "
             f"⇒ a REAL untested-free lever; line reopens, D2 becomes live.")
    else:
        v = (f"D1-ext-A FAIL — F_core+orderflow best NET Sharpe {b1:+.2f} "
             f"({b1d['tag']}, IC {b1d['ic']:+.3f}) ≤ +1.5; same-universe "
             f"F_core baseline {b0:+.2f}, Δ {delta:+.2f}. Perp aggTrade "
             f"order-flow does NOT lift the leak-free information ceiling. "
             f"Information-bound now broadens to include perp order-flow. "
             f"Remaining untested free family = spot microstructure "
             f"(D1-ext-B); else terminus. Production LGBM unaffected.")
    print(f"\n  baseline(F_core,20-sym)={b0:+.2f}  +orderflow={b1:+.2f}  "
          f"Δ={delta:+.2f}  | PRE-REG GATE(>{GATE:+.1f}): "
          f"{'PASS' if PASS else 'FAIL'}", flush=True)
    print(f"  VERDICT: {v}", flush=True)
    pd.DataFrame(R0 + R1 + [dict(tag="VERDICT", net_sh=b1, base=b0,
                 delta=delta, PASS=PASS, verdict=v)]).to_csv(
        OUTD/"summary.csv", index=False)
    pd.DataFrame([{"PASS": PASS, "base": b0, "with_oflow": b1,
                   "delta": delta, "verdict": v}]).to_csv(
        OUTD/"verdict.csv", index=False)
    print(f"\nSaved {OUTD}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
