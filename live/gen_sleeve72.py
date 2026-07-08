"""B3 sleeve books (RESEARCH_LOOP_20260707 addendum 8c, PRE-REGISTERED).

72h-label sleeve cell: does resid_ret_3d add book-level value on a 72h residual target?
Builds BOTH arms with identical machinery:
  baseline arm : V0_LEAN features, retrained on the h72 label       -> hl_slv72base_{base,long}[_oos]
  variant arm  : V0_LEAN + resid_ret_3d, same label/cuts            -> hl_slv72res3_{base,long}[_oos]
Pins (8c):
- Label = xs_z (per-cycle) of the 18-cycle forward alpha sum (alpha[t] fwd close(t)->t+4h),
  grid-guarded (open_time[t+17] - open_time[t] == 17*4h), clipped +-10 like production xs_z.
- PURGE: embargo uses exit72 = open_time + 72h (NOT the panel 4h exit_time) — the label window
  must clear the fold cut minus 1d embargo, else training leaks the label.
- Population: BOTH arms train on the variant row mask (resid_ret_3d notna) — the baseline IS the
  matched control (8b-3). Test rows identical both arms (EXCL applied).
- Two-book structure preserved (long book adds resid_rev_2/3), RidgeCV, HL=60, min 300 rows.
Scoring: score_variant_cell.py with SCORE_FWD_CYCLES=18 SCORE_BLOCK_DAYS=3
         SCORE_BASELINE_TAG=slv72base, tag slv72res3.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0_LEAN = list(tt.V0_LEAN); EMB = pd.Timedelta(days=1); HL = 60.0
RR = ["resid_rev_2", "resid_rev_3"]
EXCL = {"LITUSDT", "VINEUSDT", "PUMPUSDT"}
KLINES = REPO / "data/ml/test/parquet/klines"
K72 = 18

def load_closes(sym):
    sd = KLINES / sym / "5m"
    if not sd.exists(): return None
    dfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(sd.glob("*.parquet"))]
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time")["close"].astype(np.float32)

def main():
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "return_pct",
                                              "alpha_vs_btc_realized"] + V0_LEAN)
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True)
    PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)]
    PAN = PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    g = PAN.groupby("symbol")
    a = g["alpha_vs_btc_realized"]
    PAN["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
    PAN["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
    for c in RR: PAN[c] = PAN[c].fillna(0.0)
    # h72 label: 18-cycle forward alpha sum, grid-guarded; xs_z per cycle (production convention)
    lab = a.transform(lambda s: s.rolling(K72).sum().shift(-(K72 - 1)))
    ok = (g["open_time"].shift(-(K72 - 1)) - PAN["open_time"]) == (K72 - 1) * pd.Timedelta(hours=4)
    PAN["alpha72"] = lab.where(ok)
    gt = PAN.groupby("open_time")
    sd = gt["alpha72"].transform("std").replace(0, np.nan)
    PAN["xs_z72"] = ((PAN["alpha72"] - gt["alpha72"].transform("mean")) / sd).clip(-10, 10)
    PAN["exit72"] = PAN["open_time"] + pd.Timedelta(hours=72)   # purge horizon (8c pin)

    print("building resid_ret_3d (C3 construction)...", flush=True)
    btc = load_closes("BTCUSDT")
    btc_ret = np.log(btc / btc.shift(1))
    parts = []
    for i, sym in enumerate(sorted(PAN["symbol"].unique())):
        c = load_closes(sym)
        if c is None: continue
        my_ret = np.log(c / c.shift(1))
        ci = my_ret.index.intersection(btc_ret.index)
        mr = my_ret.reindex(ci); br = btc_ret.reindex(ci)
        cov = mr.rolling(288, min_periods=72).cov(br); var = br.rolling(288, min_periods=72).var()
        beta = (cov / var.replace(0, np.nan)).shift(1)
        idio = mr - beta * br
        v = idio.rolling(864, min_periods=432).sum().shift(1).rename("resid_ret_3d")
        v = v.reset_index(); v["symbol"] = sym
        parts.append(v)
        if (i + 1) % 40 == 0: print(f"  {i+1} syms", flush=True)
    V = pd.concat(parts, ignore_index=True)
    V["open_time"] = pd.to_datetime(V["open_time"], utc=True)
    PAN = PAN.merge(V, on=["symbol", "open_time"], how="left")
    PAN = PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    print(f"xs_z72 coverage {PAN.xs_z72.notna().mean():.3f}; "
          f"resid_ret_3d coverage {PAN.resid_ret_3d.notna().mean():.3f}", flush=True)

    def gen(cuts, feats, outpath, tagn):
        rec = []
        for i in range(len(cuts) - 1):
            c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
            # PURGE on the 72h label window + BOTH arms on the variant row mask (8c pins)
            tr = PAN[(PAN.exit72 < fc) & PAN["xs_z72"].notna() & PAN["resid_ret_3d"].notna()]
            te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1) & (~PAN.symbol.isin(EXCL))]
            if not len(tr) or not len(te): continue
            t_end = tr["open_time"].max()
            n_tr, n_sym = 0, 0
            for sym, gg in tr.groupby("symbol"):
                if len(gg) < 300: continue
                try:
                    s, h = x6.fit_preproc(gg, feats); X = x6.apply_preproc(gg, feats, s, h)
                    w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
                    m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["xs_z72"].to_numpy(), sample_weight=w)
                    gte = te[te.symbol == sym]
                    if len(gte):
                        rec.append(pd.DataFrame({"symbol": sym, "open_time": gte["open_time"].values,
                            "alpha_A": gte["alpha_vs_btc_realized"].values, "return_pct": gte["return_pct"].values,
                            "exit_time": gte["exit_time"].values,
                            "pred": m.predict(x6.apply_preproc(gte, feats, s, h)), "fold": i}))
                    n_tr += len(gg); n_sym += 1
                except Exception:
                    pass
            print(f"    {tagn} fold {i} ({c0.date()}): {n_sym} syms, {n_tr} train rows", flush=True)
        out = pd.concat(rec, ignore_index=True)
        for c in ("open_time", "exit_time"): out[c] = pd.to_datetime(out[c], utc=True)
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(outpath)
        print(f"  wrote {outpath}", flush=True)

    last = PAN["open_time"].max().normalize() + pd.Timedelta(days=1)
    REC_CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04", "2025-11-01", "2025-12-01", "2026-01-01",
                "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-05-27"]] + [last]
    OOS_CUTS = list(pd.date_range("2023-01-01", "2025-10-01", freq="MS", tz="UTC"))
    D = REPO / "live/state/convexity"
    FB = V0_LEAN                       # baseline arm
    FV = V0_LEAN + ["resid_ret_3d"]    # variant arm
    for arm, feats, tag in (("baseline", FB, "slv72base"), ("variant", FV, "slv72res3")):
        print(f"== {arm} arm ==", flush=True)
        gen(REC_CUTS, feats, D / f"hl_{tag}_base/v0full_hl60.parquet", f"{tag}-rb")
        gen(REC_CUTS, feats + RR, D / f"hl_{tag}_long/v0full_hl60.parquet", f"{tag}-rl")
        gen(OOS_CUTS, feats, D / f"hl_{tag}_base_oos/v0full_hl60.parquet", f"{tag}-ob")
        gen(OOS_CUTS, feats + RR, D / f"hl_{tag}_long_oos/v0full_hl60.parquet", f"{tag}-ol")
    print("SLV72DONE")

if __name__ == "__main__":
    main()
