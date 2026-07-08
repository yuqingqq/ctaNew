"""Feature-variant harness (RESEARCH_LOOP_20260707 addendum 6, BINDING protocol).

Runs ONE pre-registered feature-variant cell: build the variant column from klines, swap it into
V0_LEAN (replacement, not addition), train the frozen two-book WF machinery on both windows, and
score the PRIMARY book-level endpoints vs the incumbent books per the estimator law (replay
through path-coupled overlays is NOT a valid variant estimator — see V4_PERFORMANCE §8).

Cells (addendum 6b final numbering):
  C1: ret_3d -> ret_36h        C2: ret_3d -> ret_6d        C3: ret_3d -> resid_ret_3d (incumbent-alpha)
  C4: bars_since_high (+xs_rank) -> dd_from_high_288 (+xs_rank)  [parity ladder]

Population accounting (mandatory): per-fold training rows + symbols, variant vs incumbent.
Usage: python3 live/feature_variant_harness.py C1|C2|C3|C4|C5|T1
Output book tags (score with live/score_variant_cell.py <tag>):
  C1=ret36h  C2=retc1  C3=resid3  C4=ddc2  C5=corr12h  T1=takerls
KNOWN DEVIATIONS (results review 2026-07-08): variant-NaN train rows are DROPPED (line ~130),
not imputed — this violated the T1 "NaN => preproc imputation" pin and shifts symbol entry folds
for history-hungry variants (C2); the 6b population-matched control books were never built. Any
reuse for a cell where the variant changes row coverage must either implement imputation or
build the matched control book first.
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

CELLS = {
    # reviewer-final numbering (addendum 6b). NB: books already generated under interim names:
    # tag retc1 == C2 (ret_6d), tag ddc2 == C4 (dd_from_high).
    "C1": (["ret_3d"], "ret_36h", "ret36h"),
    "C2": (["ret_3d"], "ret_6d", "retc1"),
    "C3": (["ret_3d"], "resid_ret_3d", "resid3"),
    "C4": (["bars_since_high", "bars_since_high_xs_rank"], "dd_from_high_288", "ddc2"),
    # C5 granted by the screening-extension directional flag (addendum 6b gate): corr window 288->144
    "C5": (["corr_to_btc_1d"], "corr_to_btc_12h", "corr12h"),
    # T1 (addendum 6d): ADDITION cell — replaces nothing; adds the lagged taker long/short ratio
    "T1": ([], "taker_ls_24h_lag36h", "takerls"),
}

def load_closes(sym):
    sd = KLINES / sym / "5m"
    if not sd.exists(): return None
    dfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(sd.glob("*.parquet"))]
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time")["close"].astype(np.float32)

BTC = None

def build_variant_col(cell, syms):
    """Per-symbol variant series at 5-min, to be sampled onto panel rows. Shift conventions copied
    verbatim from the addendum-4 feature map (ret_3d: shift(1); bars_since_high: same-bar)."""
    parts = []
    for i, sym in enumerate(syms):
        c = load_closes(sym)
        if c is None: continue
        if cell == "C1":
            v = c.pct_change(432).shift(1).rename("ret_36h")             # mirrors X6b ret_3d convention
        elif cell == "C2":
            v = c.pct_change(288 * 6).shift(1).rename("ret_6d")          # mirrors X6b ret_3d convention
        elif cell == "C3":
            # residualized 3d return: trailing 864-bar sum of per-bar INCUMBENT alpha (beta_288, shift 1)
            my_ret = np.log(c / c.shift(1)); btc_ret = np.log(BTC / BTC.shift(1))
            ci = my_ret.index.intersection(btc_ret.index)
            mr = my_ret.reindex(ci); br = btc_ret.reindex(ci)
            cov = mr.rolling(288, min_periods=72).cov(br); var = br.rolling(288, min_periods=72).var()
            beta = (cov / var.replace(0, np.nan)).shift(1)
            idio = mr - beta * br
            v = idio.rolling(864, min_periods=432).sum().shift(1).rename("resid_ret_3d")
        elif cell == "T1":
            f = REPO / "data/ml/cache" / f"metrics_{sym}.parquet"
            if not f.exists(): continue
            m = pd.read_parquet(f, columns=["sum_taker_long_short_vol_ratio"]).sort_index()
            m = m[~m.index.duplicated(keep="last")]
            m.index = pd.to_datetime(m.index, utc=True, format="mixed")
            grid = pd.date_range(m.index.min().ceil("5min"), m.index.max(), freq="5min", tz="UTC")
            tls = m["sum_taker_long_short_vol_ratio"].astype(float).reindex(grid)
            t24 = tls.rolling("24h").mean().where(tls.notna().rolling("24h").count() >= 230)
            v = t24.shift(432).rename("taker_ls_24h_lag36h")   # 36h worst-case Vision availability lag
            v.index.name = "open_time"                         # metrics grid index is unnamed (closes' is not)
        elif cell == "C5":
            my_ret = np.log(c / c.shift(1)); btc_ret = np.log(BTC / BTC.shift(1))
            ci = my_ret.index.intersection(btc_ret.index)
            v = my_ret.reindex(ci).rolling(144, min_periods=36).corr(btc_ret.reindex(ci)).shift(1).rename("corr_to_btc_12h")
        else:
            v = (c / c.rolling(288).max() - 1).rename("dd_from_high_288")  # same-bar, mirrors bars_since_high PIT basis
        v = v.reset_index(); v["symbol"] = sym
        parts.append(v)
        if (i + 1) % 40 == 0: print(f"  {i+1} syms", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    return out

def main():
    cell = sys.argv[1]
    replaced, vname, tag = CELLS[cell]
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "return_pct",
                                              "alpha_vs_btc_realized"] + V0_LEAN)
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True)
    PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
    PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)].sort_values(["symbol", "open_time"])
    a = PAN.groupby("symbol")["alpha_vs_btc_realized"]
    PAN["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
    PAN["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
    for c in RR: PAN[c] = PAN[c].fillna(0.0)
    g = PAN.groupby("open_time"); sd = g["alpha_vs_btc_realized"].transform("std").replace(0, np.nan)
    PAN["xs_z"] = ((PAN["alpha_vs_btc_realized"] - g["alpha_vs_btc_realized"].transform("mean")) / sd).clip(-10, 10)

    print(f"building variant {vname}...", flush=True)
    global BTC
    if cell in ("C3", "C5"):
        BTC = load_closes("BTCUSDT")
    V = build_variant_col(cell, sorted(PAN["symbol"].unique()))
    PAN = PAN.merge(V, on=["symbol", "open_time"], how="left")
    if cell == "C4":  # xs_rank of the variant, mirroring bars_since_high_xs_rank
        PAN["dd_from_high_288_xs_rank"] = PAN.groupby("open_time")["dd_from_high_288"].rank(pct=True)
        feats_variant = [f for f in V0_LEAN if f not in replaced] + ["dd_from_high_288", "dd_from_high_288_xs_rank"]
    else:
        feats_variant = [f for f in V0_LEAN if f not in replaced] + [vname]
    PAN = PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    _inc = f" (incumbent {replaced[0]}: {PAN[replaced[0]].notna().mean():.4f})" if replaced else " (ADDITION cell)"
    print(f"variant coverage: {PAN[vname].notna().mean():.4f}" + _inc)

    def gen(cuts, feats, outpath, tagn):
        rec = []
        for i in range(len(cuts) - 1):
            c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
            tr = PAN[(PAN.exit_time < fc) & PAN["xs_z"].notna()]
            trv = tr.dropna(subset=[vname]) if cell in ("C1", "C2", "C3", "C5", "T1") else tr
            te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1) & (~PAN.symbol.isin(EXCL))]
            if not len(trv) or not len(te): continue
            t_end = trv["open_time"].max()
            n_tr, n_sym = 0, 0
            for sym, gg in trv.groupby("symbol"):
                if len(gg) < 300: continue
                try:
                    s, h = x6.fit_preproc(gg, feats); X = x6.apply_preproc(gg, feats, s, h)
                    w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
                    m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["xs_z"].to_numpy(), sample_weight=w)
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
    print("recent base:", flush=True); gen(REC_CUTS, feats_variant, D / f"hl_{tag}_base/v0full_hl60.parquet", "rb")
    print("recent long:", flush=True); gen(REC_CUTS, feats_variant + RR, D / f"hl_{tag}_long/v0full_hl60.parquet", "rl")
    print("oos base:", flush=True); gen(OOS_CUTS, feats_variant, D / f"hl_{tag}_base_oos/v0full_hl60.parquet", "ob")
    print("oos long:", flush=True); gen(OOS_CUTS, feats_variant + RR, D / f"hl_{tag}_long_oos/v0full_hl60.parquet", "ol")
    print(f"{cell}GENDONE")

if __name__ == "__main__":
    main()
