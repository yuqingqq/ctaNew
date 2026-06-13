"""Composite Study — pure per-composite event study (LOCKED, descriptive).
See README.md. When each composite fires, what is the forward performance
in the direction it bets? All 14 reported; matched random-timing placebo =
the pure-effect test. In-sample/descriptive — NOT a gated strategy.
"""
from __future__ import annotations
import importlib.util, sys, glob, warnings
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
from ml.research.alpha_v4_xs import block_bootstrap_ci
ANN4 = np.sqrt(365.0 * 6.0)
COST = s94.COST
OUTD = REPO / "linear_model/composite_study/results"
OUTD.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 104, flush=True)
    print("  COMPOSITE STUDY — pure per-composite event study (LOCKED, "
          "descriptive; scaling bug FIXED)", flush=True)
    print("=" * 104, flush=True)
    dec, syms, btc, pan = s94.build(universe_oi=False)
    oi = pd.read_parquet(REPO / "outputs/vBTC_features_oi/oi_panel.parquet")
    of = pd.read_parquet(REPO / "outputs/vBTC_features_oflow/oflow_panel.parquet")
    for f in (oi, of):
        f["open_time"] = pd.to_datetime(f["open_time"], utc=True)
    d = (dec[dec.symbol.isin(set(oi.symbol) & set(of.symbol))]
         .merge(oi, on=["symbol", "open_time"], how="inner")
         .merge(of, on=["symbol", "open_time"], how="inner"))
    need = ["return_1d", "return_8h", "s_t", "funding_rate_z_7d", "oi_chg_1d",
            "oi_chg_4h", "oi_z_7d", "ls_taker_z_1d", "of_imb_1d",
            "vol_zscore_4h_over_7d", "obv_z_1d", "corr_to_btc_1d",
            "alpha_beta", "fold"]
    d = d.dropna(subset=need).sort_values(["open_time", "symbol"]).reset_index(drop=True)

    # FIX: per-cycle cross-sectional z of s_t for ALL magnitude gates
    g = d.groupby("open_time")["s_t"]
    d["sz"] = ((d["s_t"] - g.transform("mean")) /
               g.transform("std").replace(0, np.nan)).fillna(0.0)

    # forward raw returns (price performance) — clean shift, per symbol
    parts = []
    for sym in d.symbol.unique():
        c = s94.load_close(sym)
        if c is None:
            continue
        c = c.set_index("open_time")["close"]
        parts.append(pd.DataFrame({
            "symbol": sym, "open_time": c.index,
            "fwd4_raw": (c.shift(-48)/c - 1.0).values,
            "fwd24_raw": (c.shift(-288)/c - 1.0).values}))
    fr = pd.concat(parts, ignore_index=True)
    fr["open_time"] = pd.to_datetime(fr["open_time"], utc=True)
    d = d.merge(fr, on=["symbol", "open_time"], how="left")

    r, r8, st, sz = d["return_1d"], d["return_8h"], d["s_t"], d["sz"]
    fz, oic, oiz = d["funding_rate_z_7d"], d["oi_chg_1d"], d["oi_z_7d"]
    ls, ofi, vz = d["ls_taker_z_1d"], d["of_imb_1d"], d["vol_zscore_4h_over_7d"]
    obv, cb = d["obv_z_1d"], d["corr_to_btc_1d"]
    sg = lambda x: np.sign(x).astype(float)
    C = {
      "O1": np.where((r > 0) & (oic > 0) & (fz > 0) & (ls > 0), -1.0, 0),
      "O2": np.where((r < 0) & (oic > 0) & (fz < 0) & (ls < 0), 1.0, 0),
      "O3": np.where((r < 0) & (oic < 0) & (vz > 1), 1.0, 0),
      "O4": np.where((r > 0) & (oic < 0), -1.0, 0),
      "O5": np.where(fz.abs() > 1, -sg(fz), 0),
      "P1": np.where(sz.abs() > 1, -sg(st), 0),                 # FIXED (sz)
      "P2": np.where((sg(r) == sg(r8)) & (r != 0), -sg(r), 0),
      "V1": np.where((vz > 1) & (r != 0), -sg(r), 0),
      "V2": np.where((vz < -0.5) & (r != 0), -sg(r), 0),
      "V3": np.where(sg(obv) != sg(r), sg(obv), 0),
      "V4": np.where((sg(ofi) != sg(st)) & (sz.abs() > 0.5), -sg(st), 0),  # FIXED
      "R1": np.where(sz.abs() > 1.5, -sg(st), 0),               # FIXED (extreme-dev)
      "R2": np.where((vz > 1) & (r != 0), sg(r), 0),            # momentum
      "X1": np.where((cb < 0) & (sz.abs() > 1), -sg(st), 0),    # FIXED
    }
    M = pd.DataFrame(C, index=d.index)
    ab = d["alpha_beta"].to_numpy() * 1e4
    f24 = d["fwd24_raw"].to_numpy() * 1e4
    fold = d["fold"].to_numpy()
    rng = np.random.default_rng(0)

    print(f"\n  rows={len(d)} syms={d.symbol.nunique()} "
          f"cycles={d.open_time.nunique()}\n", flush=True)
    print(f"  {'C':3s} {'fire%':>6s} {'dir':>4s} | {'4hRes_bps':>9s} "
          f"{'CI':>17s} {'hit%':>5s} {'netSh':>6s} {'fold+':>5s} "
          f"{'plc_pct':>7s} | {'24hRaw_bps':>10s} {'hit%':>5s}", flush=True)
    rows = []
    for c in M.columns:
        v = M[c].to_numpy()
        m = v != 0
        n = int(m.sum())
        if n < 80:
            print(f"  {c:3s} {m.mean()*100:6.1f}  (fires too rarely n={n})",
                  flush=True)
            continue
        sp = v[m] * ab[m]                     # signed fwd-4h residual (bps)
        sp24 = v[m] * f24[m]
        hit = (np.sign(v[m]) == np.sign(ab[m])).mean() * 100
        hit24 = (np.sign(v[m]) == np.sign(f24[m]))
        hit24 = np.nanmean(hit24[~np.isnan(f24[m])]) * 100
        lo, hi = block_bootstrap_ci(sp, statistic=np.mean, block_size=7,
                                    n_boot=800)[1:]
        # net-of-cost: enter/exit the fired position
        net = sp - np.where(np.r_[True, np.diff(v[m]) != 0], abs(1.0), 0)*COST*2
        nsh = float(net.mean()/net.std(ddof=1)*ANN4) if net.std(ddof=1) > 1e-9 else np.nan
        fp = sum(1 for fl in np.unique(fold[m])
                 if sp[fold[m] == fl].mean() > 0)
        nf = len(np.unique(fold[m]))
        # matched RANDOM-TIMING placebo: same #fires, same dir-rule, random rows
        pl = []
        dirv = v[m]
        for _ in range(300):
            ridx = rng.choice(len(d), n, replace=False)
            pl.append((dirv * ab[ridx]).mean())
        pct = (np.array(pl) < sp.mean()).mean() * 100
        print(f"  {c:3s} {m.mean()*100:6.1f}  n={n:5d} | {sp.mean():+9.2f} "
              f"[{lo:+6.1f},{hi:+6.1f}] {hit:5.1f} {nsh:+6.2f} {fp:2d}/{nf:<2d} "
              f"{pct:6.1f} | {sp24.mean():+10.2f} {hit24:5.1f}", flush=True)
        rows.append(dict(composite=c, fire_pct=round(m.mean()*100, 1), n=n,
                         res4h_bps=round(sp.mean(), 2), ci_lo=round(lo, 1),
                         ci_hi=round(hi, 1), hit=round(hit, 1),
                         net_sh=round(nsh, 2), folds_pos=f"{fp}/{nf}",
                         placebo_pct=round(pct, 1),
                         raw24h_bps=round(sp24.mean(), 2),
                         hit24=round(hit24, 1)))
    pd.DataFrame(rows).to_csv(OUTD / "per_composite.csv", index=False)

    print("""
  READING / VERDICT (descriptive, in-sample):
   • 4hRes_bps = mean β-residual over next 4h IN the composite's bet
     direction, when it fires. hit% ≈ direction reliability.
   • plc_pct = where the real mean sits vs 300 random-timing draws (same
     #fires, same direction rule). ~50 = the composite's STATE/TIMING adds
     NOTHING over trading that direction at random times (the pure-effect
     test). >95 would be a candidate; <50 means worse than random timing.
   • 24hRaw_bps = longer price view, BETA-LADEN (unhedged) — context only.
   Any composite at plc_pct>95 AND CI-excludes-0 AND ≥6/9 folds is only a
   CANDIDATE; per the arc such in-sample standouts fail the loop-closed
   nested-OOS+placebo gate. This study does NOT run that gate. Not a
   strategy. Production LGBM unaffected.
""", flush=True)
    print(f"Saved {OUTD}/per_composite.csv", flush=True)


if __name__ == "__main__":
    main()
