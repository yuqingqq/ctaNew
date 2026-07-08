"""Beta-window LABEL A/B (queued by the 2026-07-08 pipeline audit; pre-registered in
RESEARCH_LOOP_20260707 addendum 5). Tests the textbook SHRUNK beta label against the incumbent
1-day beta label, isolating the LABEL only:

  incumbent: alpha = my_fwd − β_288·btc_fwd            (β on rolling 288×5min, shift(1))
  variant  : alpha* = my_fwd − (0.5·β_288 + 0.5·β_1440)·btc_fwd   (β_1440 = 5d, min_periods 360)

Isolation pins (fixed before running):
- ONLY the training target changes. Features — including resid_rev_2/3, which are derived from
  the incumbent alpha — stay exactly as in the incumbent books. Bot replay mark-to-market uses raw
  returns and is unaffected; the comparison is therefore purely "what the model learned".
- fwd returns stay ROW-based (matching the incumbent) — no _fill_grid here, or the A/B would
  confound two changes.
- Same WF machinery/cuts as the incumbent books: recent = gen_residual_target CUTS
  (2025-10-04, monthly, panel-end+1d); OOS = gen_oos_v4 CUTS (2023-01-01..2025-10-01 MS).
  Same V0_LEAN / +RR books, RidgeCV, HL=60, embargo 1d, min 300 rows.
- Clean universe: output rows exclude {LITUSDT, VINEUSDT, PUMPUSDT}.
Outputs: live/state/convexity/hl_shrunkB_{base,long}[_oos]/v0full_hl60.parquet
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
HORIZON = 48  # 4h in 5-min bars (mirrors X70)
EXCL = {"LITUSDT", "VINEUSDT", "PUMPUSDT"}
KLINES = REPO / "data/ml/test/parquet/klines"

def load_closes(sym):
    sd = KLINES / sym / "5m"
    if not sd.exists(): return None
    dfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(sd.glob("*.parquet"))]
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time")["close"].astype(np.float32)

def shrunk_alpha(my_close, btc_close):
    """X70 target_alpha with the SHRUNK beta (0.5·β_288 + 0.5·β_1440); identical conventions."""
    my_ret = np.log(my_close / my_close.shift(1)); btc_ret = np.log(btc_close / btc_close.shift(1))
    ci = my_ret.index.intersection(btc_ret.index)
    my_ret = my_ret.reindex(ci); btc_ret = btc_ret.reindex(ci)
    def beta(w, mp):
        cov = my_ret.rolling(w, min_periods=mp).cov(btc_ret)
        var = btc_ret.rolling(w, min_periods=mp).var()
        return (cov / var.replace(0, np.nan)).shift(1)
    import os
    mode = os.environ.get("BETA_AB_MODE", "shrunk")
    b = beta(1440, 360) if mode == "pure5d" else 0.5 * beta(288, 72) + 0.5 * beta(1440, 360)
    mc = my_close.reindex(ci); bc = btc_close.reindex(ci)
    my_fwd = (mc.shift(-HORIZON) / mc - 1)
    btc_fwd = (bc.shift(-HORIZON) / bc - 1)
    return (my_fwd - b * btc_fwd).astype(np.float32)

TAG = __import__("os").environ.get("BETA_AB_TAG", "shrunkB")

def main():
    print("building shrunk-beta alpha for all panel symbols...", flush=True)
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "return_pct",
                                              "alpha_vs_btc_realized"] + V0_LEAN)
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True)
    PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
    PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)].sort_values(["symbol", "open_time"])
    a = PAN.groupby("symbol")["alpha_vs_btc_realized"]      # resid_rev from INCUMBENT alpha (isolation pin)
    PAN["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
    PAN["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
    for c in RR: PAN[c] = PAN[c].fillna(0.0)
    btc = load_closes("BTCUSDT")
    parts = []
    for i, sym in enumerate(sorted(PAN["symbol"].unique())):
        c = load_closes(sym)
        if c is None: continue
        al = shrunk_alpha(c, btc).rename("alpha_shrunk").reset_index()
        al["symbol"] = sym
        parts.append(al)
        if (i + 1) % 40 == 0: print(f"  {i+1} syms", flush=True)
    A = pd.concat(parts, ignore_index=True)
    A["open_time"] = pd.to_datetime(A["open_time"], utc=True)
    PAN = PAN.merge(A, on=["symbol", "open_time"], how="left")
    g = PAN.groupby("open_time"); sd = g["alpha_shrunk"].transform("std").replace(0, np.nan)
    PAN["z_shrunk"] = ((PAN["alpha_shrunk"] - g["alpha_shrunk"].transform("mean")) / sd).clip(-10, 10)
    PAN = PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    print(f"panel {len(PAN)} rows; z_shrunk coverage {PAN.z_shrunk.notna().mean():.3f}; "
          f"corr(z_shrunk, incumbent-alpha xs) sanity next", flush=True)

    def gen(cuts, feats, outpath):
        rec = []
        for i in range(len(cuts) - 1):
            c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
            tr = PAN[(PAN.exit_time < fc) & PAN["z_shrunk"].notna()]
            te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1) & (~PAN.symbol.isin(EXCL))]
            if not len(tr) or not len(te): continue
            t_end = tr["open_time"].max()
            for sym, gg in tr.groupby("symbol"):
                if len(gg) < 300: continue
                try:
                    s, h = x6.fit_preproc(gg, feats); X = x6.apply_preproc(gg, feats, s, h)
                    w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
                    m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["z_shrunk"].to_numpy(), sample_weight=w)
                    gte = te[te.symbol == sym]
                    if len(gte):
                        rec.append(pd.DataFrame({"symbol": sym, "open_time": gte["open_time"].values,
                            "alpha_A": gte["alpha_vs_btc_realized"].values, "return_pct": gte["return_pct"].values,
                            "exit_time": gte["exit_time"].values,
                            "pred": m.predict(x6.apply_preproc(gte, feats, s, h)), "fold": i}))
                except Exception:
                    pass
            print(f"    fold {i} ({c0.date()}) done", flush=True)
        out = pd.concat(rec, ignore_index=True)
        for c in ("open_time", "exit_time"): out[c] = pd.to_datetime(out[c], utc=True)
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(outpath)
        print(f"  wrote {outpath} ({out.symbol.nunique()} syms, {len(out)} rows)", flush=True)

    last = PAN["open_time"].max().normalize() + pd.Timedelta(days=1)
    REC_CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04", "2025-11-01", "2025-12-01", "2026-01-01",
                "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-05-27"]] + [last]
    OOS_CUTS = list(pd.date_range("2023-01-01", "2025-10-01", freq="MS", tz="UTC"))
    D = REPO / "live/state/convexity"
    print("recent base:", flush=True); gen(REC_CUTS, V0_LEAN, D / f"hl_{TAG}_base/v0full_hl60.parquet")
    print("recent long:", flush=True); gen(REC_CUTS, V0_LEAN + RR, D / f"hl_{TAG}_long/v0full_hl60.parquet")
    print("oos base:", flush=True); gen(OOS_CUTS, V0_LEAN, D / f"hl_{TAG}_base_oos/v0full_hl60.parquet")
    print("oos long:", flush=True); gen(OOS_CUTS, V0_LEAN + RR, D / f"hl_{TAG}_long_oos/v0full_hl60.parquet")
    print("BETAABDONE")

if __name__ == "__main__":
    main()
