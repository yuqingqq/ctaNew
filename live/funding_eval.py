"""Funding-ADJUSTED evaluator: score a replay run on (price PnL + reconstructed funding carry), per-fold, vs the
baseline's funding-adjusted performance (iter5_tilt0 = +3.784 Sh / +14973 after funding). Funding is reconstructed
from the run's own sleeves.csv (the actual filtered book) + panel funding_rate, accrued 0.5*fr/cycle (fr=8h rate),
sign -sum_s w_s*fr_s (long pays / short receives when fr>0). Emits one JSON line like opt_eval.py.

  python3 live/funding_eval.py <run_dir> [label]
"""
import sys, json
import numpy as np, pandas as pd
ROOT = "/home/yuqing/ctaNew"; HOLD = 6; ANN = np.sqrt(6*365)
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27","2026-06-30"]]
_PAN = pd.read_parquet(f"{ROOT}/outputs/vBTC_features/panel_expanded_v0.parquet", columns=["symbol","open_time","funding_rate"])
_PAN["open_time"] = pd.to_datetime(_PAN["open_time"], utc=True)
_FUND = _PAN.dropna(subset=["funding_rate"]).set_index(["open_time","symbol"])["funding_rate"]

def fold(t):
    for i in range(len(CUTS)-1):
        if CUTS[i] <= t < CUTS[i+1]: return i
    return -1

def fund_adj(run):
    cyc = pd.read_csv(f"{run}/cycles.csv"); cyc["open_time"] = pd.to_datetime(cyc["open_time"], utc=True)
    sl = pd.read_csv(f"{run}/sleeves.csv"); sl = sl[sl["event"]=="enter"].copy()
    sl["w"] = sl["weights_json"].apply(lambda j: json.loads(j) if isinstance(j,str) and j.strip() else {})
    w_by_cid = dict(zip(sl["cycle_id"], sl["w"]))
    fb = []
    for t, ot in zip(cyc["cycle_id"], cyc["open_time"]):
        book = {}
        for c in range(t-HOLD+1, t+1):
            w = w_by_cid.get(c)
            if w:
                for s, wt in w.items(): book[s] = book.get(s,0.0) + wt/HOLD
        fp = 0.0
        for s, bw in book.items():
            try: fr = _FUND.loc[(ot, s)]
            except KeyError: fr = np.nan
            if np.isfinite(fr): fp += -bw*fr
        fb.append(fp*0.5*1e4)
    cyc["fund_bps"] = fb
    cyc["net"] = cyc["pnl_bps"] + cyc["fund_bps"]
    cyc["f"] = cyc["open_time"].map(fold)
    pf = cyc.groupby("f")["net"].sum().reindex(range(9)).fillna(0)
    p = cyc["net"]/1e4; eq = cyc["net"].fillna(0).cumsum()
    return dict(sharpe=float(p.mean()/p.std()*ANN), totpnl=float(cyc["net"].sum()),
                maxdd=float((eq-eq.cummax()).min()), fund=float(cyc["fund_bps"].sum()),
                price_sharpe=float((cyc["pnl_bps"]/1e4).mean()/(cyc["pnl_bps"]/1e4).std()*ANN)), pf

def main():
    run = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else run.split("/")[-1]
    s, pf = fund_adj(run)
    bs, bpf = fund_adj(f"{ROOT}/live/state/v3loop/iter5_tilt0")
    d = (pf - bpf)
    print(json.dumps(dict(label=label,
        fund_adj_sharpe=round(s["sharpe"],3), fund_adj_totpnl=round(s["totpnl"],0), fund_adj_maxdd=round(s["maxdd"],0),
        funding_bps=round(s["fund"],0), price_only_sharpe=round(s["price_sharpe"],3),
        baseline_fund_adj_sharpe=round(bs["sharpe"],3), baseline_funding_bps=round(bs["fund"],0),
        lift_fund_adj=round(s["sharpe"]-bs["sharpe"],3), maxdd_change_pct=round((s["maxdd"]-bs["maxdd"])/abs(bs["maxdd"])*100,0),
        folds_positive=int((d>0).sum()), per_fold_delta=[round(x) for x in d.values])))

if __name__ == "__main__":
    main()
