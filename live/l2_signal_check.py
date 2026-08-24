"""Comprehensive OB SIGNAL check on the COMPLETE data. For every OB feature, cross-sectional rank-IC vs BOTH targets
— return_pct (RAW 4h fwd direction; exit_time=open_time+4h) and alpha_vs_btc (BETA-NEUTRAL 4h fwd, what the book
trades) — in BOTH eras (full 2023→2026 cache), day-clustered CI. Answers "is there ANY OB signal now": if OB shows up
on RAW return but not on ALPHA, it's market-beta the book discards; if nothing both-era on either, no signal at all.
Lightweight: l2 cache + panel forward returns, no kline reload.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_persist import persist_feats
rng = np.random.default_rng(11)
FEATS = ["imb1", "imb_ewma", "imb_run", "liq1", "slope", "asym1", "imbstd"]
RAW = {"imb1": "l2_imb1", "liq1": "l2_liq1", "slope": "l2_slope", "asym1": "l2_asym1", "imbstd": "l2_imbstd"}

def build():
    rows = []
    for f in [x for x in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet") if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]; d = pd.read_parquet(f); d.index = pd.to_datetime(d.index, utc=True)
        pf = persist_feats(d["l2_imb1"].sort_index())[["imb_ewma", "imb_run"]]
        for k, c in RAW.items(): pf[k] = d[c] if c in d.columns else np.nan
        pf.index = pf.index + pd.Timedelta("4h")                       # PIT: obs bar -> decision bar
        pf["symbol"] = sym; pf["open_time"] = pf.index; rows.append(pf.reset_index(drop=True))
    L = pd.concat(rows, ignore_index=True)
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time", "return_pct", "alpha_vs_btc_realized"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    return pan.merge(L, on=["symbol", "open_time"], how="inner")

def xic(df, feat, tgt):
    return df.groupby("open_time").apply(lambda g: g[feat].corr(g[tgt], method="spearman") if g[[feat, tgt]].dropna().shape[0] >= 8 else np.nan).dropna()

def dayci(ic):
    s = pd.DataFrame({"ic": ic.values}, index=pd.to_datetime(ic.index, utc=True)); s["d"] = s.index.floor("1D")
    g = [x["ic"].values for _, x in s.groupby("d")]
    if len(g) < 5: return (np.nan, np.nan)
    o = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(3000)]
    return tuple(np.nanpercentile(o, [2.5, 97.5]))

def main():
    m = build(); cut = pd.Timestamp("2025-10-01", tz="UTC")
    eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"merged {len(m)} rows | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])} | full-data cross-sectional rank-IC\n")
    for tgt, lab in [("return_pct", "RAW 4h return (direction)"), ("alpha_vs_btc_realized", "ALPHA vs BTC (beta-neutral, what the book trades)")]:
        print(f"### target = {lab} ###")
        print(f"{'feature':9s} | {'RECENT IC [CI]':24s} | {'OOS IC [CI]':24s} | both-era signal?")
        for feat in FEATS:
            cells = {}
            for era, sub in eras.items():
                ic = xic(sub, feat, tgt); lo, up = dayci(ic); cells[era] = (ic.mean(), lo, up)
            (ra, rl, ru), (oa, ol, ou) = cells["RECENT"], cells["OOS"]
            both = "YES" if (np.sign(ra) == np.sign(oa) and (rl > 0 or ru < 0) and (ol > 0 or ou < 0) and abs(ra) > 0.02 and abs(oa) > 0.02) else "no"
            print(f"{feat:9s} | {ra:+.4f} [{rl:+.4f},{ru:+.4f}] | {oa:+.4f} [{ol:+.4f},{ou:+.4f}] | {both}")
        print()
    print("read: both-era on RAW-return but not ALPHA = market-beta (book discards it); nothing both-era = no signal. SIGCHECKDONE")

if __name__ == "__main__":
    main()
