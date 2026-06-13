"""Composite Study — RAW-price influence, honestly decomposed (LOCKED,
descriptive). Owner: "forget the residual, check the composite influence."

raw_fwd = beta·btc_fwd (market/exposure) + alpha_beta (idiosyncratic).
Per composite, when fired, in its bet direction, report:
  RAW total | = MKT(beta·btc_fwd) part + IDIO(alpha_beta) part | hit% |
  RAW vs random-timing placebo pct | RAW vs market-exposure-matched placebo.
If RAW ≈ MKT-part and RAW doesn't beat the exposure-matched placebo, the
"influence" is just directional market exposure (drift/beta), not skill.
4h and 24h horizons. All 14 reported. In-sample/descriptive. NOT a strategy.
"""
from __future__ import annotations
import importlib.util, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
OUTD = REPO / "linear_model/composite_study/results"


def main():
    print("=" * 112, flush=True)
    print("  COMPOSITE STUDY — RAW-price influence decomposed (RAW = MKT/beta "
          "+ IDIO/residual). Descriptive.", flush=True)
    print("=" * 112, flush=True)
    dec, syms, btc, pan = s94.build(universe_oi=False)
    oi = pd.read_parquet(REPO / "outputs/vBTC_features_oi/oi_panel.parquet")
    of = pd.read_parquet(REPO / "outputs/vBTC_features_oflow/oflow_panel.parquet")
    for f in (oi, of):
        f["open_time"] = pd.to_datetime(f["open_time"], utc=True)
    d = (dec[dec.symbol.isin(set(oi.symbol) & set(of.symbol))]
         .merge(oi, on=["symbol", "open_time"], how="inner")
         .merge(of, on=["symbol", "open_time"], how="inner"))
    need = ["return_1d", "return_8h", "s_t", "funding_rate_z_7d", "oi_chg_1d",
            "oi_z_7d", "ls_taker_z_1d", "of_imb_1d", "vol_zscore_4h_over_7d",
            "obv_z_1d", "corr_to_btc_1d", "alpha_beta", "beta_btc_pit",
            "fold"]
    d = d.dropna(subset=need).sort_values(["open_time", "symbol"]).reset_index(drop=True)
    gp = d.groupby("open_time")["s_t"]
    d["sz"] = ((d["s_t"]-gp.transform("mean")) /
               gp.transform("std").replace(0, np.nan)).fillna(0.0)

    # forward raw + BTC forward → MKT (beta·btc_fwd) and IDIO (alpha_beta)
    parts = []
    for sym in d.symbol.unique():
        c = s94.load_close(sym)
        if c is None:
            continue
        c = c.set_index("open_time")["close"]
        parts.append(pd.DataFrame({"symbol": sym, "open_time": c.index,
            "fwd4_raw": (c.shift(-48)/c-1.0).values,
            "fwd24_raw": (c.shift(-288)/c-1.0).values}))
    fr = pd.concat(parts, ignore_index=True)
    fr["open_time"] = pd.to_datetime(fr["open_time"], utc=True)
    bdf = pd.DataFrame({"open_time": btc.index,
                        "btc4": (btc.shift(-48)/btc-1.0).values,
                        "btc24": (btc.shift(-288)/btc-1.0).values})
    d = d.merge(fr, on=["symbol", "open_time"], how="left").merge(
        bdf, on="open_time", how="left")
    d["mkt4"] = d["beta_btc_pit"]*d["btc4"]            # market/exposure part
    d["mkt24"] = d["beta_btc_pit"]*d["btc24"]
    d["idio4"] = d["alpha_beta"]                        # idiosyncratic part
    # idio24 = raw24 − beta·btc24 (forward 24h residual, decomposition only)
    d["idio24"] = d["fwd24_raw"] - d["mkt24"]

    r, r8, st, sz = d["return_1d"], d["return_8h"], d["s_t"], d["sz"]
    fz, oic = d["funding_rate_z_7d"], d["oi_chg_1d"]
    ls, ofi, vz = d["ls_taker_z_1d"], d["of_imb_1d"], d["vol_zscore_4h_over_7d"]
    obv, cb = d["obv_z_1d"], d["corr_to_btc_1d"]
    sgn = lambda x: np.sign(x).astype(float)
    C = {
      "O1": np.where((r > 0) & (oic > 0) & (fz > 0) & (ls > 0), -1.0, 0),
      "O2": np.where((r < 0) & (oic > 0) & (fz < 0) & (ls < 0), 1.0, 0),
      "O3": np.where((r < 0) & (oic < 0) & (vz > 1), 1.0, 0),
      "O4": np.where((r > 0) & (oic < 0), -1.0, 0),
      "O5": np.where(fz.abs() > 1, -sgn(fz), 0),
      "P1": np.where(sz.abs() > 1, -sgn(st), 0),
      "P2": np.where((sgn(r) == sgn(r8)) & (r != 0), -sgn(r), 0),
      "V1": np.where((vz > 1) & (r != 0), -sgn(r), 0),
      "V2": np.where((vz < -0.5) & (r != 0), -sgn(r), 0),
      "V3": np.where(sgn(obv) != sgn(r), sgn(obv), 0),
      "V4": np.where((sgn(ofi) != sgn(st)) & (sz.abs() > 0.5), -sgn(st), 0),
      "R1": np.where(sz.abs() > 1.5, -sgn(st), 0),
      "R2": np.where((vz > 1) & (r != 0), sgn(r), 0),
      "X1": np.where((cb < 0) & (sz.abs() > 1), -sgn(st), 0),
    }
    M = pd.DataFrame(C, index=d.index)
    raw4 = d["fwd4_raw"].to_numpy()*1e4
    mkt4 = d["mkt4"].to_numpy()*1e4
    idi4 = d["idio4"].to_numpy()*1e4
    raw24 = d["fwd24_raw"].to_numpy()*1e4
    mkt24 = d["mkt24"].to_numpy()*1e4
    rng = np.random.default_rng(0)
    Nrows = len(d)

    print(f"\n  rows={Nrows} syms={d.symbol.nunique()}  "
          f"(sample drift: mean raw fwd4 = {raw4.mean():+.2f}bps, "
          f"mean mkt4 = {mkt4.mean():+.2f}bps)\n", flush=True)
    print(f"  {'C':3s} {'fire%':>6s} | {'RAW4':>8s} = {'MKT4':>8s} + "
          f"{'IDIO4':>7s} {'hitR4':>6s} {'plcT':>5s} {'plcEXP':>6s} | "
          f"{'RAW24':>9s} = {'MKT24':>9s} + {'IDIO24':>8s} {'hitR24':>6s}",
          flush=True)
    rows = []
    for c in M.columns:
        v = M[c].to_numpy(); m = v != 0; n = int(m.sum())
        if n < 80:
            print(f"  {c:3s} {m.mean()*100:6.1f}  (n={n} too few)", flush=True)
            continue
        dv = v[m]
        R4, K4, I4 = (dv*raw4[m]).mean(), (dv*mkt4[m]).mean(), (dv*idi4[m]).mean()
        R24, K24 = (dv*raw24[m]).mean(), (dv*mkt24[m]).mean()
        I24 = (dv*(raw24[m]-mkt24[m])).mean()
        hR4 = (np.sign(dv) == np.sign(raw4[m])).mean()*100
        hR24 = np.nanmean(np.sign(dv) == np.sign(raw24[m]))*100
        # random-timing placebo on RAW (same #fires, same dir rule)
        plT = np.mean([ (dv*raw4[rng.choice(Nrows, n, replace=False)]).mean()
                        for _ in range(200)] < np.float64(R4))*100
        # market-exposure-matched placebo: same per-symbol net dir, random time
        netdir = pd.Series(dv, index=d.index[m]).groupby(
            d.loc[m, "symbol"]).mean()
        plE = []
        for _ in range(200):
            rr = rng.choice(Nrows, n, replace=False)
            ss = d["symbol"].to_numpy()[rr]
            sd = np.array([netdir.get(x, 0.0) for x in ss])
            plE.append((sd*raw4[rr]).mean())
        plEXP = np.mean(np.array(plE) < R4)*100
        print(f"  {c:3s} {m.mean()*100:6.1f} | {R4:+8.2f} = {K4:+8.2f} + "
              f"{I4:+7.2f} {hR4:6.1f} {plT:5.0f} {plEXP:6.0f} | {R24:+9.2f} "
              f"= {K24:+9.2f} + {I24:+8.2f} {hR24:6.1f}", flush=True)
        rows.append(dict(composite=c, fire_pct=round(m.mean()*100, 1), n=n,
            raw4=round(R4, 2), mkt4=round(K4, 2), idio4=round(I4, 2),
            hit_raw4=round(hR4, 1), plc_time=round(plT, 0),
            plc_expmatch=round(plEXP, 0), raw24=round(R24, 2),
            mkt24=round(K24, 2), idio24=round(I24, 2),
            hit_raw24=round(hR24, 1)))
    pd.DataFrame(rows).to_csv(OUTD/"raw_decomposition.csv", index=False)
    print("""
  READING (descriptive, in-sample):
   • RAW4 = MKT4 + IDIO4. If RAW's sign/size lives in MKT4 (beta·btc_fwd),
     the "influence on price" is just directional MARKET EXPOSURE, not
     composite skill. IDIO4 (= alpha_beta) is the skill part — and it is
     the ≈0 we have established across the whole arc.
   • plcT = vs random-timing (raw). plcEXP = vs MARKET-EXPOSURE-MATCHED
     placebo (same net per-symbol direction, random time). plcEXP≈50 ⇒ the
     raw "influence" does NOT beat simply holding that net exposure ⇒ it is
     drift/beta, not prediction. Only plcEXP>95 AND IDIO carrying it would
     be a real (and separate, DIRECTIONAL/beta-laden) finding — and would
     still need the loop-closed nested-OOS gate (not run here).
""", flush=True)
    print(f"Saved {OUTD}/raw_decomposition.csv", flush=True)


if __name__ == "__main__":
    main()
