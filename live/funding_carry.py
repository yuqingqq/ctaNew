"""Funding-carry realism: the +4.22 backtest is PRICE-RETURN ONLY (grep funding live/convexity_paper_bot.py == empty).
Perps pay/charge funding every 8h. This reconstructs the ACTUAL aggregated book per cycle (from sleeves.csv, the same
1/HOLD sleeve-average the bot uses) and accrues realized funding from the panel's funding_rate, then recomputes
Sharpe/totPnL WITH funding. Honest sign: long pays funding when fr>0, short receives; book funding = -sum_s w_s*fr_s.
Accrue 0.5*fr per 4h cycle (fr is the 8h rate). Reports the carry drag + funding-adjusted performance + per-regime split.

  python3 live/funding_carry.py [run_dir=live/state/v3loop/iter5_tilt0]
"""
import sys, json
import numpy as np, pandas as pd
ROOT = "/home/yuqing/ctaNew"
RUN = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/live/state/v3loop/iter5_tilt0"
HOLD = 6; ANN = np.sqrt(6*365)

cyc = pd.read_csv(f"{RUN}/cycles.csv"); cyc["open_time"] = pd.to_datetime(cyc["open_time"], utc=True)
sl = pd.read_csv(f"{RUN}/sleeves.csv"); sl = sl[sl["event"] == "enter"].copy()
sl["w"] = sl["weights_json"].apply(lambda j: json.loads(j) if isinstance(j, str) and j.strip() else {})
w_by_cid = dict(zip(sl["cycle_id"], sl["w"]))

pan = pd.read_parquet(f"{ROOT}/outputs/vBTC_features/panel_expanded_v0.parquet", columns=["symbol", "open_time", "funding_rate"])
pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
# funding_rate structure: forward-filled over the 8h window or stamped at settlement?
s0 = pan[pan.symbol == "ETHUSDT"].sort_values("open_time")["funding_rate"].dropna().head(10).round(6).tolist()
print(f"ETH funding_rate consecutive 4h bars: {s0}")
fund = pan.dropna(subset=["funding_rate"]).set_index(["open_time", "symbol"])["funding_rate"]

rows = []
for t in cyc["cycle_id"].tolist():
    ot = cyc.loc[cyc.cycle_id == t, "open_time"].iloc[0]
    book = {}
    for c in range(t - HOLD + 1, t + 1):
        w = w_by_cid.get(c)
        if not w: continue
        for s, wt in w.items(): book[s] = book.get(s, 0.0) + wt / HOLD
    fp = 0.0
    for s, bw in book.items():
        try: fr = fund.loc[(ot, s)]
        except KeyError: fr = np.nan
        if np.isfinite(fr): fp += -bw * fr           # long pays / short receives when fr>0
    rows.append((t, fp * 0.5 * 1e4))                 # 0.5 = 4h/8h ; ->bps
fpnl = pd.DataFrame(rows, columns=["cycle_id", "fund_bps"]).set_index("cycle_id")["fund_bps"]
cyc = cyc.set_index("cycle_id")
cyc["fund_bps"] = fpnl
cyc["pnl_with_fund"] = cyc["pnl_bps"] + cyc["fund_bps"]

def sharpe(x): return x.mean() / x.std() * ANN
print(f"\n=== FUNDING-CARRY REALISM ({RUN.split('/')[-1]}) ===")
print(f"price-only:      Sharpe {sharpe(cyc['pnl_bps']):+.3f}  totPnL {cyc['pnl_bps'].sum():+.0f}  mean/cyc {cyc['pnl_bps'].mean():+.3f}")
print(f"funding total:   {cyc['fund_bps'].sum():+.0f} bps  ({cyc['fund_bps'].mean():+.3f}/cyc, {cyc['fund_bps'].mean()*6*365/100:+.1f}% annualized on book)")
print(f"WITH funding:    Sharpe {sharpe(cyc['pnl_with_fund']):+.3f}  totPnL {cyc['pnl_with_fund'].sum():+.0f}  mean/cyc {cyc['pnl_with_fund'].mean():+.3f}")
print(f"  -> funding haircut: Sharpe {sharpe(cyc['pnl_with_fund'])-sharpe(cyc['pnl_bps']):+.3f}  PnL {cyc['fund_bps'].sum()/cyc['pnl_bps'].sum()*100:+.0f}%")
print(f"\nper-regime funding drag (bps total / mean per cyc):")
for r, g in cyc.groupby("regime"):
    print(f"  {r:5s} n={len(g):4d}  fund {g['fund_bps'].sum():+7.0f} ({g['fund_bps'].mean():+.3f}/cyc)  price {g['pnl_bps'].sum():+7.0f}  with-fund Sharpe {sharpe(g['pnl_with_fund']):+.2f} (price {sharpe(g['pnl_bps']):+.2f})")
print(f"\nfunding by sign: cycles paying {(cyc['fund_bps']<0).mean()*100:.0f}%  receiving {(cyc['fund_bps']>0).mean()*100:.0f}%  worst cycle {cyc['fund_bps'].min():+.1f}  best {cyc['fund_bps'].max():+.1f}")
cyc.reset_index()[["cycle_id", "open_time", "regime", "pnl_bps", "fund_bps", "pnl_with_fund"]].to_csv(f"{ROOT}/live/state/v3loop/funding_carry.csv", index=False)
print("wrote live/state/v3loop/funding_carry.csv ; DONE funding_carry")
