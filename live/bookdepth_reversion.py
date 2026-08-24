"""User: L2 should carry SHORT-TERM alpha usable at 4h as a REVERSION signal. Test it properly (all prior tests
framed imbalance as CONTINUATION). Reversion = fade the imbalance / fade the recent move; could be linear, tail-only
(exhaustion), or conditional on a move. 4h holds, both eras, day-clustered CI, net of cost.

Signals (PIT at bar T; fwd_4h = raw return over [T,T+4h]):
  A linear:  IC(-imb1, fwd_4h)               reversion if >0 (fade bid-heavy). [continuation test gave IC(imb1)~+0.005]
  B price-rev: IC(-ret_prev4h, fwd_4h)        is there raw 4h price reversal at all? (strategy's core is resid-reversion)
  C L2-enhanced: does imb1 add to price reversion? partial IC(-imb1, fwd_4h | ret_prev4h)
  D tail:    extreme imb1 deciles -> do bid-heavy fall & ask-heavy rise? (non-linear exhaustion)
  E divergence: price up but book ask-heavy (or vice versa) -> revert? IC(-sign(ret_prev4h)*|imb1 opposing|, fwd_4h)
  F BOOK:    contrarian long-short (long bottom-imb decile, short top-imb decile), 4h hold, NET of 8bps -> tradeable?
             + the price-reversion book (long losers/short winners) for comparison.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_persist import persist_feats, close_4h
rng = np.random.default_rng(19); COST = 0.0008   # 8bps round-trip (liquid names, taker+spread)

def day_boot(ic):
    s = pd.DataFrame({"ic": np.asarray(ic)}, index=pd.to_datetime(ic.index, utc=True)); s["d"] = s.index.floor("1D")
    g = [x["ic"].values for _, x in s.groupby("d")]
    if len(g) < 5: return (np.nan, np.nan)
    o = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(2500)]
    return tuple(np.nanpercentile(o, [2.5, 97.5]))

def xic(df, feat, tgt="fwd_4h"):
    return df.groupby("open_time").apply(lambda g: g[feat].corr(g[tgt], method="spearman") if g[[feat, tgt]].dropna().shape[0] >= 8 else np.nan).dropna()

def main():
    rows = []
    for f in [x for x in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet") if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]; d = pd.read_parquet(f)
        d.index = pd.to_datetime(d.index, utc=True) + pd.Timedelta("4h")
        imb = d["l2_imb1"].sort_index()
        df = pd.DataFrame({"imb1": imb, "imb_ewma": persist_feats(imb)["imb_ewma"]})
        c = close_4h(sym, str(df.index.min().date()), str((df.index.max() + pd.Timedelta("2D")).date()))
        if c is None: continue
        c = c.reindex(df.index.union(c.index)).sort_index()
        df["fwd_4h"] = (c.shift(-1) / c - 1).reindex(df.index)
        df["ret_prev4h"] = (c / c.shift(1) - 1).reindex(df.index)          # just-happened 4h (PIT, known at T)
        df["symbol"] = sym; df["open_time"] = df.index; rows.append(df.reset_index(drop=True))
    m = pd.concat(rows, ignore_index=True).dropna(subset=["fwd_4h", "imb1", "ret_prev4h"])
    m["neg_imb"] = -m["imb1"]; m["neg_ewma"] = -m["imb_ewma"]; m["neg_prev"] = -m["ret_prev4h"]
    cut = pd.Timestamp("2025-10-01", tz="UTC"); eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}

    def line(tag, sub, feat):
        ic = xic(sub, feat); lo, up = day_boot(ic)
        f = "SIG" if (lo > 0 or up < 0) else "~0"
        return f"{ic.mean():+.4f} [{lo:+.4f},{up:+.4f}]{f}"

    print("=== A/B/C reversion ICs (rank-IC vs fwd_4h; >0 = the reversion signal predicts) ===")
    for era, sub in eras.items():
        print(f"  {era}:")
        print(f"    A fade-imbalance    (-imb1)      {line('A', sub, 'neg_imb')}")
        print(f"    A' fade-sustained   (-imb_ewma)  {line('A', sub, 'neg_ewma')}")
        print(f"    B price-reversion   (-ret_prev4h){line('B', sub, 'neg_prev')}")
    # C: does fade-imbalance add beyond price reversion? partial (residualize both on ret_prev4h ranks, per bar pooled)
    print("\n=== C: fade-imbalance PARTIAL of price-reversion (does L2 add to price reversion?) ===")
    for era, sub in eras.items():
        s = sub.copy()
        R = s.groupby("open_time")[["neg_imb", "neg_prev", "fwd_4h"]].rank(pct=True)
        def res(y, x): A = np.column_stack([np.ones(len(x)), x]); b, *_ = np.linalg.lstsq(A, y, rcond=None); return y - A @ b
        rx = res(R["neg_imb"].values, R["neg_prev"].values); ry = res(R["fwd_4h"].values, R["neg_prev"].values)
        raw = np.corrcoef(R["neg_imb"], R["fwd_4h"])[0, 1]; part = np.corrcoef(rx, ry)[0, 1]
        print(f"  {era}: raw {raw:+.4f} -> partial(|price-rev) {part:+.4f}")

    print("\n=== D tail reversion: mean fwd_4h in extreme imbalance deciles (bid-heavy should FALL, ask-heavy RISE) ===")
    for era, sub in eras.items():
        s = sub.copy(); s["dec"] = s.groupby("open_time")["imb1"].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop"))
        top = s[s.dec == 9]["fwd_4h"]; bot = s[s.dec == 0]["fwd_4h"]   # top=most bid-heavy, bot=most ask-heavy
        print(f"  {era}: top-imb(bid-heavy) fwd {top.mean()*1e4:+.1f}bps | bot-imb(ask-heavy) fwd {bot.mean()*1e4:+.1f}bps | rev-spread(bot-top) {(bot.mean()-top.mean())*1e4:+.1f}bps")

    print("\n=== F tradeable contrarian long-short books (4h hold, NET 8bps), both eras ===")
    for name, feat in [("fade-imbalance (long bot-imb/short top-imb)", "imb1"), ("price-reversion (long losers/short winners)", "ret_prev4h")]:
        print(f"  {name}:")
        for era, sub in eras.items():
            s = sub.copy(); s["dec"] = s.groupby("open_time")[feat].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop"))
            per = s.groupby("open_time").apply(lambda g: g[g.dec == 0]["fwd_4h"].mean() - g[g.dec == 9]["fwd_4h"].mean()).dropna()
            net = per - COST; lo, up = day_boot(net)
            f = "TRADEABLE (CI>0)" if lo > 0 else "~0/neg"
            print(f"    {era}: net/leg-pair {net.mean()*1e4:+.1f}bps/4h [{lo*1e4:+.1f},{up*1e4:+.1f}] {f}")
    print("REVDONE")

if __name__ == "__main__":
    main()
