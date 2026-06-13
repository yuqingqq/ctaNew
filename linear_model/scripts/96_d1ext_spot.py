"""Step 96 — D1-ext-B: does SPOT microstructure lift the leak-free ceiling,
STACKED on order-flow? (the honest free-data-stack test).

Pre-registered (INFORMATION_DIAGNOSTIC_PLAN.md §Status): D1-ext-A showed
perp order-flow adds a clean +0.63 marginal but the ceiling stays ≤ +1.5.
The two free families are tested STACKED — order-flow carries forward; the
question is whether F_core + order-flow + SPOT clears +1.5.

Spot 5m klines downloaded from data.binance.vision/data/spot (egress
confirmed), cached. 6 PIT spot features (all TRAILING + .shift(1) ⇒ bar t
uses only ≤ t-1; same convention as panel/oflow), using consistent
quote_volume both sides + spot taker-buy columns:
  sp_basis_z1d   z288 of basis=(perp_close/spot_close-1)   (basis dislocation)
  sp_basis_4h    trailing-48 mean basis
  sp_taker_imb_1d  trailing-288 mean of (2*s_takerbuy - s_vol)/s_vol  (spot CVD)
  sp_volratio_z1d  z288 of spot_qvol/perp_qvol            (spot-vs-perp lead)
  sp_vol_z7d     z2016 of spot quote-volume               (spot vol regime)
  sp_retdiff_4h  trailing-48 mean of (spot_ret5 - perp_ret5)  (spot lead-lag)

Universe = hl42 ∩ aggTrades ∩ spot (the 20 liquid majors). Same leak-free
CV (whole-timestamp 5-fold + 1d embargo) + SAME +1.5 gate. Reports F_core /
+oflow / +spot / +oflow+spot(GATED) on the SAME rows. Mandatory PIT audit.
No strategy adopted. Production LGBM unaffected.
"""
from __future__ import annotations
import importlib.util, sys, time, io, zipfile, warnings
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests

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
s95 = _imp("s95", "linear_model/scripts/95_d1ext_orderflow.py")
build, score, GATE, LEAK = s94.build, s94.score, s94.GATE, s94.LEAK
grouped_oof, OF = s94b.grouped_oof, s95.OF
SPOT_URL = "https://data.binance.vision/data/spot/daily/klines"
PERP = REPO / "data/ml/test/parquet/klines"
CACHE = REPO / "outputs/vBTC_features_spot"
(CACHE / "spot_klines").mkdir(parents=True, exist_ok=True)
SPANEL = CACHE / "spot_panel.parquet"
OFLOW = REPO / "outputs/vBTC_features_oflow/oflow_panel.parquet"
OUTD = REPO / "linear_model/results/step96_d1ext_spot"
OUTD.mkdir(parents=True, exist_ok=True)
SP = ["sp_basis_z1d", "sp_basis_4h", "sp_taker_imb_1d", "sp_volratio_z1d",
      "sp_vol_z7d", "sp_retdiff_4h"]
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
         "ignore"]


def _z(x, w):
    m = x.rolling(w, min_periods=max(20, w // 5)).mean()
    s = x.rolling(w, min_periods=max(20, w // 5)).std()
    return (x - m) / s.replace(0, np.nan)


def _dl_day(sym, d):
    url = f"{SPOT_URL}/{sym}/5m/{sym}-5m-{d:%Y-%m-%d}.zip"
    try:
        r = requests.get(url, timeout=25)
        if r.status_code != 200:
            return None
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        raw = pd.read_csv(zf.open(zf.namelist()[0]), header=None)
        if not str(raw.iloc[0, 0]).replace(".", "").isdigit():
            raw = raw.iloc[1:]                       # header row present
        raw = raw.iloc[:, :12]
        raw.columns = KCOLS
        return raw
    except Exception:
        return None


def fetch_spot(sym, d_lo, d_hi):
    pq = CACHE / "spot_klines" / f"{sym}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    days = []
    dd = d_lo
    while dd <= d_hi:
        days.append(dd); dd += timedelta(days=1)
    out = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(_dl_day, sym, d): d for d in days}
        for f in as_completed(futs):
            r = f.result()
            if r is not None:
                out.append(r)
    if not out:
        return None
    df = pd.concat(out, ignore_index=True)
    ot = pd.to_numeric(df["open_time"], errors="coerce").dropna().astype("int64")
    mx = int(ot.max())                       # Binance switched ms→µs in 2025
    unit = "ns" if mx > 10**17 else "us" if mx > 10**14 else "ms"
    df = df.loc[ot.index]
    df["open_time"] = pd.to_datetime(ot, unit=unit, utc=True)
    for c in ["close", "volume", "quote_volume", "taker_buy_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df[["open_time", "close", "volume", "quote_volume",
              "taker_buy_volume"]].dropna().drop_duplicates("open_time")
          .sort_values("open_time").reset_index(drop=True))
    df.to_parquet(pq, index=False)
    return df


def perp_kl(sym):
    import glob
    fs = sorted(glob.glob(str(PERP / sym / "5m" / "*.parquet")))
    d = pd.concat([pd.read_parquet(f, columns=["open_time", "close",
                  "quote_volume"]) for f in fs], ignore_index=True)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True, errors="coerce")
    return (d.dropna(subset=["open_time"]).drop_duplicates("open_time")
            .sort_values("open_time").reset_index(drop=True)
            .rename(columns={"close": "p_close", "quote_volume": "p_qvol"}))


def spot_features(sym, d_lo, d_hi):
    s = fetch_spot(sym, d_lo, d_hi)
    if s is None or len(s) < 3000:
        return None
    p = perp_kl(sym)
    b = s.merge(p, on="open_time", how="inner").sort_values(
        "open_time").reset_index(drop=True)
    b["basis"] = b["p_close"] / b["close"] - 1.0
    b["s_imb"] = (2 * b["taker_buy_volume"] - b["volume"]) / \
        b["volume"].replace(0, np.nan)
    b["vr"] = b["quote_volume"] / b["p_qvol"].replace(0, np.nan)
    b["s_ret"] = b["close"].pct_change()
    b["p_ret"] = b["p_close"].pct_change()
    b["sp_basis_z1d"] = _z(b["basis"], 288).shift(1)
    b["sp_basis_4h"] = b["basis"].rolling(48, min_periods=10).mean().shift(1)
    b["sp_taker_imb_1d"] = b["s_imb"].rolling(288, min_periods=60).mean().shift(1)
    b["sp_volratio_z1d"] = _z(b["vr"], 288).shift(1)
    b["sp_vol_z7d"] = _z(b["quote_volume"], 2016).shift(1)
    b["sp_retdiff_4h"] = (b["s_ret"] - b["p_ret"]).rolling(
        48, min_periods=10).mean().shift(1)
    b["symbol"] = sym
    return b[["symbol", "open_time"] + SP]


def build_spot_panel(syms, d_lo, d_hi):
    if SPANEL.exists():
        p = pd.read_parquet(SPANEL)
        p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
        if set(syms).issubset(set(p["symbol"].unique())):
            print(f"  [cache] spot_panel {p.shape} reused", flush=True)
            return p
    rows = []
    for k, s in enumerate(syms):
        t0 = time.time()
        fx = spot_features(s, d_lo, d_hi)
        if fx is not None:
            rows.append(fx)
            print(f"  [{k+1}/{len(syms)}] {s:12s} rows={len(fx)} "
                  f"{time.time()-t0:.0f}s", flush=True)
        else:
            print(f"  [{k+1}/{len(syms)}] {s:12s} SKIP (no/short spot)",
                  flush=True)
    pan = pd.concat(rows, ignore_index=True)
    pan.to_parquet(SPANEL, index=False)
    print(f"  built+cached spot_panel {pan.shape} -> {SPANEL}", flush=True)
    return pan


def main():
    print("=" * 96, flush=True)
    print("  STEP 96 — D1-ext-B: SPOT microstructure STACKED on order-flow "
          "vs leak-free ceiling (+1.5 gate)", flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()
    dec, syms, btc, pan = build(universe_oi=False)
    of = pd.read_parquet(OFLOW)
    of["open_time"] = pd.to_datetime(of["open_time"], utc=True)
    have = sorted(set(of["symbol"].unique()))
    d_lo = (dec.open_time.min() - pd.Timedelta(days=10)).date()
    d_hi = dec.open_time.max().date()
    print(f"  universe (hl42∩aggTrades) = {len(have)}; spot span "
          f"{d_lo}→{d_hi}", flush=True)
    sp = build_spot_panel(have, d_lo, d_hi)

    dec = dec[dec.symbol.isin(have)].merge(
        of, on=["symbol", "open_time"], how="inner").merge(
        sp, on=["symbol", "open_time"], how="inner")
    dec = dec[dec.open_time <= sp["open_time"].max()]
    print(f"  merged dec rows={len(dec)} syms={dec.symbol.nunique()} "
          f"cycles={dec.open_time.nunique()}", flush=True)

    # ---- PIT audit ----
    ok = True
    for s in ["SOLUSDT", "ADAUSDT"]:
        sx = fetch_spot(s, d_lo, d_hi)
        sx["s_imb"] = (2*sx["taker_buy_volume"] - sx["volume"]) / \
            sx["volume"].replace(0, np.nan)
        ind = sx.sort_values("open_time")["s_imb"].rolling(
            288, min_periods=60).mean().shift(1)
        sx = sx.assign(ind=ind.values)
        m = dec[dec.symbol == s].merge(sx[["open_time", "ind"]],
            on="open_time", how="inner").dropna(subset=["sp_taker_imb_1d",
                                                        "ind"])
        cc = float(m["sp_taker_imb_1d"].corr(m["ind"])) if len(m) > 10 else np.nan
        ok &= (cc > 0.9999)
        print(f"  audit {s}: corr(stored,indep_PAST sp_taker_imb_1d)="
              f"{cc:.6f} -> {'OK' if cc > 0.9999 else 'MISMATCH'}", flush=True)
    fc = dec[SP].apply(lambda c: c.corr(dec["alpha_beta"], "spearman")).abs()
    print(f"  look-ahead |corr(spot, FWD αβ)| max={fc.max():.3f} "
          f"({fc.idxmax()}); all<0.10={bool((fc < 0.10).all())}", flush=True)
    if not (ok and (fc < 0.10).all()):
        print("\n  PIT AUDIT FAIL — not run.", flush=True)
        pd.DataFrame([{"audit": "FAIL"}]).to_csv(OUTD/"verdict.csv", index=False)
        return
    print("  PIT AUDIT: PASS", flush=True)

    base = [c for c in dec.columns if c not in LEAK and c not in OF and
            c not in SP and pd.api.types.is_numeric_dtype(dec[c])] + ["s_t"]
    base = list(dict.fromkeys(base))
    dec = dec.dropna(subset=base + OF + SP + ["tz", "alpha_beta"]
                     ).reset_index(drop=True)
    print(f"  [validate] rows={len(dec)} syms={dec.symbol.nunique()} "
          f"F_core={len(base)} oflow={len(OF)} spot={len(SP)}", flush=True)

    def run(feats, lbl):
        rid, gbm = grouped_oof(dec, feats)
        mk = ~np.isnan(rid)
        dd = dec[mk].reset_index(drop=True)
        print(f"\n--- {lbl} (feats={len(feats)}, OOF {mk.mean()*100:.0f}%) ---",
              flush=True)
        return [score(dd, rid[mk], f"Ridge|{lbl}"),
                score(dd, gbm[mk], f"LGBM|{lbl}")]

    R0 = run(base, "F_core")
    R1 = run(base + OF, "F_core+oflow")
    R2 = run(base + SP, "F_core+spot")
    R3 = run(base + OF + SP, "F_core+oflow+spot (GATED)")
    score(dec, dec["s_t"].to_numpy()*-1.0, "s_t_rule(ref)")
    b = {k: max(v, key=lambda r: r["net_sh"])["net_sh"]
         for k, v in [("F", R0), ("FO", R1), ("FS", R2), ("FOS", R3)]}
    g = max(R3, key=lambda r: r["net_sh"])
    PASS = bool(b["FOS"] > GATE)
    if PASS:
        v = (f"D1-ext-B PASS — STACKED F_core+oflow+spot best NET Sharpe "
             f"{b['FOS']:+.2f} ({g['tag']}, IC {g['ic']:+.3f}) > +1.5. The "
             f"full free-data stack clears the bar ⇒ free microstructure DOES "
             f"carry tradeable-ceiling info; LINE REOPENS, D2 (utilization) "
             f"becomes live. Marginals: F {b['F']:+.2f} → +oflow {b['FO']:+.2f}"
             f" → +spot {b['FS']:+.2f} → stacked {b['FOS']:+.2f}.")
    else:
        v = (f"D1-ext-B FAIL — STACKED F_core+oflow+spot best NET Sharpe "
             f"{b['FOS']:+.2f} ({g['tag']}, IC {g['ic']:+.3f}) ≤ +1.5. "
             f"Marginals: F {b['F']:+.2f} → +oflow {b['FO']:+.2f} → +spot "
             f"{b['FS']:+.2f} → stacked {b['FOS']:+.2f}. The FULL free-data "
             f"stack (perp-OHLCV + order-flow + spot microstructure) does NOT "
             f"clear the leak-free ceiling ⇒ **free-data information bound is "
             f"DEFINITIVE; TERMINUS.** Only remaining lever = paid/orthogonal "
             f"data domain. Production LGBM unaffected.")
    print(f"\n  STACK marginals: F={b['F']:+.2f} +oflow={b['FO']:+.2f} "
          f"+spot={b['FS']:+.2f} +oflow+spot={b['FOS']:+.2f} | "
          f"GATE(>{GATE:+.1f}): {'PASS' if PASS else 'FAIL'}", flush=True)
    print(f"  VERDICT: {v}", flush=True)
    pd.DataFrame(R0+R1+R2+R3+[dict(tag="VERDICT", net_sh=b["FOS"], PASS=PASS,
                 verdict=v)]).to_csv(OUTD/"summary.csv", index=False)
    pd.DataFrame([{"PASS": PASS, **{f"best_{k}": x for k, x in b.items()},
                   "verdict": v}]).to_csv(OUTD/"verdict.csv", index=False)
    print(f"\nSaved {OUTD}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
