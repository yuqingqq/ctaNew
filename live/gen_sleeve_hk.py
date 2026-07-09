"""h12 sleeve-aligned completion cells (RESEARCH_LOOP_20260707 addendum 17, PRE-REGISTERED).

Generalized k-cycle sleeve generator. Baseline = V0_LEAN retrained on the h-label; variant =
baseline + ONE feature; BOTH arms on the IDENTICAL population (variant row mask = matched
control built in). h12 = 3-cycle residual-alpha sum, grid-guarded, purge exit = open_time+12h.

Usage: HK_CELL=base|ret24|resid24|dd3d python3 live/gen_sleeve_hk.py
"""
import os, sys
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
K = 3                                   # h12 = 3 cycles
CELL = os.environ.get("HK_CELL", "ret24")
CYC = pd.Timedelta(hours=4)
# cell -> (added feature name, tag stem). Each cell builds BOTH arms on the feature's row mask:
#   baseline hl_<stem>base_* (V0_LEAN) + variant hl_<stem>_* (V0_LEAN+feature), identical population.
SPEC = {"ret24": ("ret_24h", "slv12_ret24"), "resid24": ("resid_ret_24h", "slv12_resid24"),
        "dd3d": ("dd_3d", "slv12_dd3d")}

def load_closes(sym):
    sd = KLINES / sym / "5m"
    if not sd.exists(): return None
    dfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(sd.glob("*.parquet"))]
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time")["close"].astype(np.float32)

def build_feature(fname, syms, BTC):
    """the ONE added feature, per symbol at 4h cadence, X6b conventions."""
    parts = []
    btc_ret = np.log(BTC / BTC.shift(1)) if fname == "resid_ret_24h" else None
    for sym in syms:
        c = load_closes(sym)
        if c is None: continue
        if fname == "ret_24h":
            v = c.pct_change(288).shift(1).rename(fname)                 # 24h raw return, shift1
        elif fname == "dd_3d":
            v = (c / c.rolling(864).max() - 1).rename(fname)            # SAME-BAR (17-fix-1)
        else:  # resid_ret_24h: incumbent beta_288 idio 24h sum (C3 construction)
            my = np.log(c / c.shift(1)); ci = my.index.intersection(btc_ret.index)
            mr = my.reindex(ci); br = btc_ret.reindex(ci)
            cov = mr.rolling(288, min_periods=72).cov(br); var = br.rolling(288, min_periods=72).var()
            beta = (cov / var.replace(0, np.nan)).shift(1)
            v = (mr - beta * br).rolling(288, min_periods=144).sum().shift(1).rename(fname)
        v = v.reset_index(); v["symbol"] = sym
        parts.append(v)
    out = pd.concat(parts, ignore_index=True)
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    return out

def main():
    fname, tag = SPEC[CELL]
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "return_pct",
                                             "alpha_vs_btc_realized"] + V0_LEAN)
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True)
    PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
    PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)].sort_values(["symbol", "open_time"])
    g = PAN.groupby("symbol"); a = g["alpha_vs_btc_realized"]
    PAN["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
    PAN["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
    for c in RR: PAN[c] = PAN[c].fillna(0.0)
    # h12 label = 3-cycle forward residual-alpha sum, GRID-GUARDED (17-fix-3)
    lab = a.transform(lambda s: s.rolling(K).sum().shift(-(K - 1)))
    ok = (g["open_time"].shift(-(K - 1)) - PAN["open_time"]) == (K - 1) * CYC
    PAN["alpha12"] = lab.where(ok)
    gt = PAN.groupby("open_time"); sd = gt["alpha12"].transform("std").replace(0, np.nan)
    PAN["xs_z12"] = ((PAN["alpha12"] - gt["alpha12"].transform("mean")) / sd).clip(-10, 10)
    PAN["exit12"] = PAN["open_time"] + pd.Timedelta(hours=12)        # purge horizon (17)
    # the ONE added feature (defines the matched population for BOTH arms of this cell)
    BTC = load_closes("BTCUSDT")
    V = build_feature(fname, sorted(PAN["symbol"].unique()), BTC)
    PAN = PAN.merge(V, on=["symbol", "open_time"], how="left")
    maskcol = fname                            # BOTH arms train on the variant's row mask
    PAN = PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    print(f"CELL={CELL} tag={tag} feature={fname}; xs_z12 cov {PAN.xs_z12.notna().mean():.3f}; "
          f"mask({maskcol}) cov {PAN[maskcol].notna().mean():.3f}", flush=True)

    def gen(cuts, ff, outpath, tagn):
        rec = []
        for i in range(len(cuts) - 1):
            c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
            tr = PAN[(PAN.exit12 < fc) & PAN["xs_z12"].notna() & PAN[maskcol].notna()]
            te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1) & (~PAN.symbol.isin(EXCL))]
            if not len(tr) or not len(te): continue
            t_end = tr["open_time"].max(); n_sym = 0
            for sym, gg in tr.groupby("symbol"):
                if len(gg) < 300: continue
                try:
                    s, h = x6.fit_preproc(gg, ff); Xtr = x6.apply_preproc(gg, ff, s, h)
                    w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
                    m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(Xtr, gg["xs_z12"].to_numpy(), sample_weight=w)
                    gte = te[te.symbol == sym]
                    if len(gte):
                        rec.append(pd.DataFrame({"symbol": sym, "open_time": gte["open_time"].values,
                            "alpha_A": gte["alpha_vs_btc_realized"].values, "return_pct": gte["return_pct"].values,
                            "exit_time": gte["exit_time"].values,
                            "pred": m.predict(x6.apply_preproc(gte, ff, s, h)), "fold": i}))
                    n_sym += 1
                except Exception:
                    pass
            print(f"    {tagn} f{i} ({c0.date()}): {n_sym} syms", flush=True)
        out = pd.concat(rec, ignore_index=True)
        for c in ("open_time", "exit_time"): out[c] = pd.to_datetime(out[c], utc=True)
        Path(outpath).parent.mkdir(parents=True, exist_ok=True); out.to_parquet(outpath)
        print(f"  wrote {outpath}", flush=True)

    last = PAN["open_time"].max().normalize() + pd.Timedelta(days=1)
    REC_CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04", "2025-11-01", "2025-12-01", "2026-01-01",
                "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-05-27"]] + [last]
    OOS_CUTS = list(pd.date_range("2023-01-01", "2025-10-01", freq="MS", tz="UTC"))
    D = REPO / "live/state/convexity"
    FB = V0_LEAN                      # baseline arm (matched control: V0_LEAN on the variant mask)
    FV = V0_LEAN + [fname]            # variant arm
    for arm, ff, stem in (("baseline", FB, tag + "base"), ("variant", FV, tag)):
        print(f"== {arm} ({stem}) ==", flush=True)
        gen(REC_CUTS, ff, D / f"hl_{stem}_base/v0full_hl60.parquet", f"{stem}-rb")
        gen(REC_CUTS, ff + RR, D / f"hl_{stem}_long/v0full_hl60.parquet", f"{stem}-rl")
        gen(OOS_CUTS, ff, D / f"hl_{stem}_base_oos/v0full_hl60.parquet", f"{stem}-ob")
        gen(OOS_CUTS, ff + RR, D / f"hl_{stem}_long_oos/v0full_hl60.parquet", f"{stem}-ol")
    print("HKDONE")

if __name__ == "__main__":
    main()
