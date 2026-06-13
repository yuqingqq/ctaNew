"""Step 83: PROPER 24h forward beta-residual on the volaug panel.

The genuinely-unresolved direction. Step-76's 24h was invalidated by a real
bug: `shift(-j*BLOCK)` on an already-4h-sampled frame jumped ~8 days/block
(should be one decision row), and it reused the 4h-horizon beta. Pre-bug it
showed +19 bps spread; never cleanly tested. This builds it correctly:

  1. ret24 / btc24 : compound the 6 NON-OVERLAPPING 4h forward blocks at the
     5m-BAR level — ret24[t] = prod_{j=0..5}(1+return_pct[t+j*48]) - 1
     (return_pct is the per-bar 4h fwd return; j*48 5m-bars = 4h apart =
     correct; this is the original full-PIT builder convention, NOT the
     Step-76 mistake of *48 on a 4h-sampled frame).
  2. beta @ 24h : rebuilt from klines as rolling-90d cov/var of 5m returns
     shifted by HORIZON_24h + 1 = 289 bars (matched strict-PIT, vs the 49
     used for the 4h target). Removes the Step-76 reused-4h-beta error.
  3. resid24 = ret24 - beta289 * btc24 ; sigma24 = fold-0 per-sym std frozen.
  4. Evaluate at NON-OVERLAPPING decision cadence (stride 6) so K3/IC/t are
     not autocorrelation-inflated.

Testbed hl42 (clean executable baseline). Configs: v2_all + the squared/
non-monotone carriers (only_squared, sqbtcrel, sqbtcrel_plus_int) since 80a
showed the 4h structure is non-monotone — test if 24h is different.
Models ridge_xsz + nnls_oriented. Step-77 payoff + drop-top-2 + per-symbol
attribution + HL flag (meme guard). SAME pre-registered gate as 78/79/80:
  clears iff decile rho >= +0.60 AND K3 >= +9.0 bps AND >=6/9 folds K3+
  AND drop-top-2 K3 > 0 AND top-5 <= 60% positive gross.
No backtest.
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
s76 = _imp("s76", "linear_model/scripts/76_minimal_orientation.py")
s77 = _imp("s77", "linear_model/scripts/77_orientation_decile_diag.py")
s78 = _imp("s78", "linear_model/scripts/78_nnls_poscoef_payoff.py")
s79 = _imp("s79", "linear_model/scripts/79_broader_universe_attrib.py")
s80b = _imp("s80b", "linear_model/scripts/80b_vol_interaction_payoff.py")
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

VOLAUG = REPO / "outputs/vBTC_features_btc_only_111_volaug/panel_btc_only_111_volaug.parquet"
KL = REPO / "data/ml/test/parquet/klines"
HL_MAP = REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv"
OUT = REPO / "linear_model/results/step83_proper_24h"
OUT.mkdir(parents=True, exist_ok=True)
OOS, BLOCK, ALPHAS = s76.OOS, s76.BLOCK, s58.ALPHAS
DAY = 288
H24 = 6                       # 6 non-overlapping 4h blocks = 24h
BETA_WIN = 90 * DAY
BETA_SHIFT_24 = 6 * BLOCK + 1  # 289 — matched strict PIT for 24h fwd window
GATE_RHO, GATE_K3, GATE_FOLDS, GATE_TOP5 = 0.60, 9.0, 6, 0.60
CARRIERS = {"v2_all": None, "only_squared": s80b.SQ,
            "sqbtcrel": s80b.SQ + s80b.BTCREL,
            "sqbtcrel_plus_int": s80b.SQ + s80b.BTCREL}  # +INT added at runtime


def load_kl(sym):
    fs = sorted(glob.glob(str(KL / sym / "5m" / "*.parquet")))
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f, columns=["open_time", "close"])
                   for f in fs], ignore_index=True)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True, errors="coerce")
    return (d.dropna(subset=["open_time"]).drop_duplicates("open_time")
              .sort_values("open_time").reset_index(drop=True))


def beta289_per_symbol(syms):
    """Rolling-90d cov/var of 5m returns vs BTC, shift(289) — matched 24h PIT."""
    btc = load_kl("BTCUSDT").set_index("open_time")
    btc["btc_r"] = btc["close"].pct_change()
    out = []
    for i, s in enumerate(syms):
        sd = load_kl(s)
        if sd is None or len(sd) < 1000:
            continue
        sd = sd.set_index("open_time")
        sd["r"] = sd["close"].pct_change()
        j = sd.join(btc[["btc_r"]], how="inner")
        cov = j["r"].rolling(BETA_WIN, min_periods=1000).cov(j["btc_r"])
        var = j["btc_r"].rolling(BETA_WIN, min_periods=1000).var()
        b = (cov / var.replace(0, np.nan)).shift(BETA_SHIFT_24)
        o = pd.DataFrame({"open_time": j.index, "beta289": b.values})
        o["symbol"] = s
        out.append(o)
        if (i + 1) % 30 == 0:
            print(f"    beta289 {i+1}/{len(syms)}", flush=True)
    a = pd.concat(out, ignore_index=True)
    a["open_time"] = pd.to_datetime(a["open_time"], utc=True)
    return a


def main():
    print("=" * 100, flush=True)
    print("  STEP 83: PROPER 24h fwd beta-residual (volaug/hl42, beta@289)",
          flush=True)
    print(f"  GATE: rho>=+{GATE_RHO} & K3>=+{GATE_K3} & >={GATE_FOLDS}/9 & "
          f"drop-top2>0 & top5<= {GATE_TOP5:.0%} | non-overlap stride {H24}",
          flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()

    raw = pd.read_parquet(VOLAUG)
    raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
    syms = sorted(raw["symbol"].unique())
    print(f"  volaug {len(raw):,} rows, {len(syms)} syms; building beta289 "
          f"from klines...", flush=True)
    b289 = beta289_per_symbol(syms + ["BTCUSDT"])
    raw = raw.merge(b289, on=["symbol", "open_time"], how="left")
    del b289
    gc.collect()

    # ret24 / btc24 by 5m-bar non-overlapping compounding (CORRECT)
    raw = raw.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    g = raw.groupby("symbol", sort=False)
    cr = np.ones(len(raw))
    cb = np.ones(len(raw))
    for jj in range(H24):
        cr *= (1.0 + g["return_pct"].shift(-jj * BLOCK).to_numpy())
        cb *= (1.0 + g["btc_ret_fwd"].shift(-jj * BLOCK).to_numpy())
    raw["ret24"] = cr - 1.0
    raw["btc24"] = cb - 1.0
    raw["resid24"] = (raw["ret24"] - raw["beta289"] * raw["btc24"]).astype("float32")
    raw["exit_time"] = raw["open_time"] + pd.Timedelta(hours=24)
    cov = raw["resid24"].notna().mean()
    print(f"  resid24 coverage {cov*100:.1f}%  std={raw['resid24'].std():.4g}",
          flush=True)

    # hl42 testbed
    hl = pd.read_csv(HL_MAP)
    keep = set(hl[(hl.on_hl) & (hl.hl_day_vol_usd >= 2e6)]["symbol"])
    p = raw[raw.symbol.isin(keep)
            & ~raw.symbol.isin({"BIOUSDT", "VVVUSDT", "BTCUSDT"})].copy()
    folds = _multi_oos_splits(p)
    f0 = _slice(p, folds[0])[0].index
    sg = p.loc[f0].groupby("symbol")["resid24"].std()
    med = float(sg.dropna().median())
    p["sigma24"] = p["symbol"].map(sg).fillna(med).clip(lower=1e-6)
    p["target_z"] = (p["resid24"] / p["sigma24"]).clip(-5, 5).astype("float32")
    p["alpha_beta"] = p["resid24"]                    # payoff target = 24h resid
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
    INT = s80b.add_interactions(dec)
    print(f"  hl42: {dec['symbol'].nunique()} syms, "
          f"{dec['open_time'].nunique()} cycles (4h grid; eval stride {H24})",
          flush=True)

    cfg = {"v2_all": fc, "only_squared": s80b.SQ,
           "sqbtcrel": s80b.SQ + s80b.BTCREL,
           "sqbtcrel_plus_int": s80b.SQ + s80b.BTCREL + INT}
    rows = []
    for mdl in ["ridge_xsz", "nnls_oriented"]:
        print(f"\n--- {mdl} ---", flush=True)
        for cname, sub in cfg.items():
            sub = [c for c in sub if c in dec.columns]
            df = s80b.score(dec, sub, folds, mdl)
            if df.empty:
                print(f"  {cname:18s} no scores", flush=True)
                continue
            # NON-OVERLAPPING evaluation: keep every 6th decision cycle
            tt = sorted(df["open_time"].unique())[::H24]
            d = df[df["open_time"].isin(set(tt))]
            p_ = s79.payoff(d)
            a, ai = s79.attribution(d, set(hl[hl.on_hl]["symbol"]))
            d2 = d[~d["symbol"].isin(ai["top2"])]
            k3d2 = (float(s77._ksweep(d2, ks=(3,)).iloc[0]["spread_bps"])
                    if not d2.empty else np.nan)
            gate = (p_["decile_rho"] >= GATE_RHO and p_["k3"] >= GATE_K3
                    and p_["k3_folds_pos"] >= GATE_FOLDS and k3d2 > 0
                    and ai["top5_share"] <= GATE_TOP5)
            rows.append({"model": mdl, "config": cname, **p_,
                         "k3_drop2": k3d2, "top5_share": ai["top5_share"],
                         "top2": ",".join(ai["top2"]),
                         "gate_pass": bool(gate)})
            print(f"  {cname:18s} IC={p_['ic']:+.4f} rho={p_['decile_rho']:+.3f} "
                  f"K1={p_['k1']:+6.2f} K3={p_['k3']:+6.2f} K10={p_['k10']:+6.2f} "
                  f"f+={p_['k3_folds_pos']}/9 dT2={k3d2:+6.2f} "
                  f"top5={ai['top5_share']*100:3.0f}% | "
                  f"{'PASS' if gate else 'FAIL'}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "summary.csv", index=False)
    clears = out[out["gate_pass"]]
    best = out.sort_values("k3", ascending=False).head(1)
    print("\n" + "=" * 100, flush=True)
    if not clears.empty:
        w = clears.iloc[0]
        v = (f"PROPER-24h CLEARS — {w['model']}/{w['config']} rho "
             f"{w['decile_rho']:+.2f} K3 {w['k3']:+.2f} dropT2 "
             f"{w['k3_drop2']:+.2f}. The 24h horizon revives the line; next = "
             f"robustness/audit, still NO backtest.")
    else:
        bb = (f"{best.iloc[0]['model']}/{best.iloc[0]['config']} K3 "
              f"{best.iloc[0]['k3']:+.2f} rho {best.iloc[0]['decile_rho']:+.2f}"
              if not best.empty else "n/a")
        v = (f"Proper-24h does NOT clear the pre-registered gate (best {bb}). "
             f"The Step-76 +19 bps was the shift+beta bug, not a real 24h "
             f"edge. This was the last genuinely-unresolved direction; with it "
             f"cleanly closed the linear beta-residual line has no clean "
             f"tradable win on free Binance data at 4h OR 24h. NOT a proxy "
             f"call — measured directly. Production LGBM unaffected.")
    print(f"  VERDICT: {v}", flush=True)
    pd.DataFrame([{"any_clears": not clears.empty, "verdict": v}]).to_csv(
        OUT / "verdict.csv", index=False)
    print(f"\nSaved under {OUT}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
