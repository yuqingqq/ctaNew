"""Pilot: does WIDER-book imbalance (imb5, at +-5% of mid) carry a directional signal the near-book imb1 (1%) missed?
Re-derives imb2/imb3/imb5 from raw bookDepth (in-memory via load_symbol, no cache write) for ~50 both-era symbols over
a 2mo/era window, and tests the full imbalance curve (imb02/imb1/imb2/imb3/imb5) vs RAW return + ALPHA, both eras,
x-sec rank-IC + day-clustered CI. If imb5 shows a both-era signal imb1 didn't, the wider book helps.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_loader import load_symbol
rng = np.random.default_rng(23)
LVL = ["imb02", "imb1", "imb2", "imb3", "imb5"]

def pilot_syms(n=50):
    ok = []
    for f in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet"):
        if "BTCUSDT" in f: continue
        ix = pd.to_datetime(pd.read_parquet(f, columns=["l2_imb1"]).index, utc=True)
        if ((ix >= "2024-05-01") & (ix < "2024-07-01")).sum() > 50 and ((ix >= "2026-05-01") & (ix < "2026-07-01")).sum() > 50:
            ok.append(Path(f).stem[3:])
    return sorted(ok)[:n]

def xic(df, feat, tgt):
    return df.groupby("open_time").apply(lambda g: g[feat].corr(g[tgt], method="spearman") if g[[feat, tgt]].dropna().shape[0] >= 8 else np.nan).dropna()

def dayci(ic):
    s = pd.DataFrame({"ic": ic.values}, index=pd.to_datetime(ic.index, utc=True)); s["d"] = s.index.floor("1D")
    g = [x["ic"].values for _, x in s.groupby("d")]
    if len(g) < 5: return (np.nan, np.nan)
    o = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(2500)]
    return tuple(np.nanpercentile(o, [2.5, 97.5]))

def main():
    syms = pilot_syms(50)
    days = pd.date_range("2026-05-01", "2026-06-30").append(pd.date_range("2024-05-01", "2024-06-30"))
    print(f"pilot: {len(syms)} syms x {len(days)} days, re-deriving imb2/imb3/imb5 from raw bookDepth...", flush=True)
    rows = []
    for i, sym in enumerate(syms, 1):
        out = load_symbol(sym, days)
        if out is None: continue
        cols = [f"l2_{c}" for c in LVL if f"l2_{c}" in out.columns]
        out = out[cols].copy(); out.columns = [c[3:] for c in cols]
        out.index = pd.to_datetime(out.index, utc=True) + pd.Timedelta("4h")   # PIT
        out["symbol"] = sym; out["open_time"] = out.index; rows.append(out.reset_index(drop=True))
        if i % 10 == 0: print(f"  {i}/{len(syms)}", flush=True)
    L = pd.concat(rows, ignore_index=True)
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time", "return_pct", "alpha_vs_btc_realized"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    m = pan.merge(L, on=["symbol", "open_time"], how="inner")
    cut = pd.Timestamp("2025-10-01", tz="UTC"); eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"\nmerged {len(m)} | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])} | corr(imb1,imb5)={m['imb1'].corr(m['imb5']):.2f}")
    for tgt, lab in [("return_pct", "RAW 4h return (direction)"), ("alpha_vs_btc_realized", "ALPHA vs BTC")]:
        print(f"\n### {lab} — imbalance across book depth (0.2% -> 5%) ###")
        for feat in LVL:
            if feat not in m.columns: continue
            cells = {}
            for era, sub in eras.items():
                ic = xic(sub, feat, tgt); lo, up = dayci(ic); cells[era] = (ic.mean(), lo, up)
            (ra, rl, ru), (oa, ol, ou) = cells["RECENT"], cells["OOS"]
            both = "BOTH-ERA" if (np.sign(ra) == np.sign(oa) and (rl > 0 or ru < 0) and (ol > 0 or ou < 0)) else "no"
            print(f"  {feat:6s} RECENT {ra:+.4f} [{rl:+.4f},{ru:+.4f}] | OOS {oa:+.4f} [{ol:+.4f},{ou:+.4f}] | {both}")
    print("\nread: does imb5 (wide book) beat imb1 (near book)? both-era = a signal. PILOTDONE")

if __name__ == "__main__":
    main()
