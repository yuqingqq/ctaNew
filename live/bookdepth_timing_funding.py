"""Fee realism — FUNDING. The corrected backtest had flat 10bps turnover only. A directional perp held ~days also
accrues funding every 8h. HYPOTHESIS: this contrarian strategy is on the side that RECEIVES funding (short when book
crowded-long = funding high = shorts earn; long on capitulation = funding low/neg = longs earn), so funding should be
a TAILWIND, not a drag. Measure funding_pnl = -pos * funding_rate on BTC/ETH (funding cache is 2025+ only -> 2025-tail
of OOS + recent). Does it flip the verdict? (Caveat: even a tailwind is separate CARRY, not the price signal; partial data.)
"""
import os
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_timing_corrected import fixed_universe, agg_z, instruments, daily, block_sharpe
CUT = pd.Timestamp("2025-10-01", tz="UTC")

def daily_funding(sym):
    p = f"data/ml/cache/funding_{sym}.parquet"
    if not os.path.exists(p): return None
    d = pd.read_parquet(p); t = pd.to_datetime(d["calc_time"], utc=True)
    fr = pd.Series(d["funding_rate"].values, index=t).sort_index()
    return fr.groupby(fr.index.floor("1D")).sum()          # daily total funding rate (sum of 8h prints)

def main():
    syms = fixed_universe(); z = agg_z(syms); insts, _ = instruments(syms)
    pos = (-z).clip(-1.5, 1.5).ewm(halflife=42).mean()
    dpos = pos.groupby(pos.index.floor("1D")).mean()
    dturn = pos.diff().abs().groupby(pos.index.floor("1D")).sum()
    print("FUNDING impact (7d hold, fixed univ). funding_pnl = -pos * funding_rate; TAILWIND if short-when-funding-high.\n")
    for name, sym in [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")]:
        fr = daily_funding(sym)
        if fr is None: print(f"  {name}: no funding file"); continue
        price = daily((pos * insts[name].reindex(pos.index)).dropna())
        idx = price.index.intersection(dpos.index).intersection(fr.index)
        p, dp, f = price.reindex(idx), dpos.reindex(idx), fr.reindex(idx)
        fund = -dp * f; cost = (0.0010 * dturn).reindex(idx).fillna(0)
        print(f"  {name}: funding covers {str(idx.min())[:10]}..{str(idx.max())[:10]}")
        for era, sub in [("OOS(2025 tail)", idx[idx < CUT]), ("RECENT", idx[idx >= CUT])]:
            if len(sub) < 20: continue
            pe, fe, ce, dpe, fre = p.reindex(sub), fund.reindex(sub), cost.reindex(sub), dp.reindex(sub), f.reindex(sub)
            corr = dpe.corr(fre)
            sp, sf = block_sharpe(pe - ce), block_sharpe(pe + fe - ce)
            print(f"    {era:14s} | funding {fe.mean()*1e4:+.2f}bps/day ({'TAILWIND' if fe.mean()>0 else 'DRAG'}) "
                  f"corr(pos,fund) {corr:+.2f} | Sharpe price-only {sp[0]:+.2f} -> +funding {sf[0]:+.2f} [{sf[1]:+.2f},{sf[2]:+.2f}]")
    print("\nread: funding is a TAILWIND if short-when-funding-high (corr<0). But it's separate CARRY, partial (2025+)")
    print("data, and can't rescue a price signal whose CI already spans zero. FUNDDONE")

if __name__ == "__main__":
    main()
