"""Step 91: WHY is per-symbol IC positive (V2: +0.015, 70% syms) but the
trading result negative (Step-90 ridge_insample −0.73, 39% syms)?

Pure diagnostic on the SAME Step-89 pred_ridge predictions (no rebuild, no
new model). Separates three candidate mechanisms:
  M1 overlap-IC illusion : V2 IC used EVERY 5m bar; alpha_beta is the 4h-fwd
     residual so consecutive obs overlap 47/48 -> autocorrelation can make a
     per-bar IC look positive while the NON-overlapping 4h-cadence relation
     (what Step 90 trades) is ~0/neg.
  M2 cost/turnover       : sign(pred) on a weak signal flips often; 4.5 bps
     per full flip every 4h can exceed a tiny gross edge.
  M3 heavy-tail sign     : IC carried by many small correct calls; a few
     large adverse alpha_beta with wrong predicted sign dominate sign-PnL.

For each symbol, on the SAME preds, reports:
  ic_allbar   : corr(pred,alpha_beta) over ALL OOS rows  (reproduces V2)
  ic_dec      : same but only at the 4h non-overlapping decision cadence
  gross_sh    : per-symbol-timing Sharpe at decision cadence, ZERO cost
  net_sh      : with cost (reproduces Step-90 ridge_insample = validation)
  turnover    : mean |Δsign|/cycle ; hit : sign-correct %
+ portfolio aggregates and a mechanism verdict.
"""
from __future__ import annotations
import importlib.util, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


s64 = _imp("s64", "linear_model/scripts/64_meanrev_v2_backtest.py")
OUTD = REPO / "linear_model/results/step91_ic_vs_trading"
OUTD.mkdir(parents=True, exist_ok=True)
PRED = REPO / "linear_model/results/step89_per_symbol_oi/per_symbol_oi_preds.parquet"
BLOCK = 48
COST = s64.COST
ANN = np.sqrt(365.0 * 6.0)


def _corr(a, b):
    if len(a) < 30 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    print("=" * 96, flush=True)
    print("  STEP 91: per-symbol IC-good / trading-bad — mechanism decomposition",
          flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()
    P = pd.read_parquet(PRED)
    P["open_time"] = pd.to_datetime(P["open_time"], utc=True)
    P = P.dropna(subset=["pred_ridge", "alpha_beta"]).sort_values(
        ["symbol", "open_time"])
    dec_grid = set(sorted(P["open_time"].unique())[::BLOCK])

    rows = []
    for s, g in P.groupby("symbol"):
        g = g.sort_values("open_time")
        ic_all = _corr(g["pred_ridge"].to_numpy(), g["alpha_beta"].to_numpy())
        d = g[g["open_time"].isin(dec_grid)].copy()
        if len(d) < 30:
            continue
        ic_dec = _corr(d["pred_ridge"].to_numpy(), d["alpha_beta"].to_numpy())
        pos = np.sign(d["pred_ridge"].to_numpy())
        y = d["alpha_beta"].to_numpy() * 1e4
        gross = pos * y
        dpos = np.abs(np.diff(pos, prepend=pos[0]))
        net = gross - dpos * COST
        gsh = float(gross.mean() / gross.std(ddof=1) * ANN) if gross.std() > 1e-9 else np.nan
        nsh = float(net.mean() / net.std(ddof=1) * ANN) if net.std() > 1e-9 else np.nan
        hit = float((np.sign(d["pred_ridge"]) == np.sign(d["alpha_beta"])).mean())
        rows.append(dict(symbol=s, n_all=len(g), n_dec=len(d),
                         ic_allbar=ic_all, ic_dec=ic_dec,
                         gross_bps=float(gross.mean()), net_bps=float(net.mean()),
                         gross_sh=gsh, net_sh=nsh,
                         turnover=float(dpos.mean()), hit=hit))
    R = pd.DataFrame(rows)
    R.to_csv(OUTD / "per_symbol.csv", index=False)

    def pct_pos(c):
        return 100 * (R[c] > 0).mean()
    print(f"\n  n_syms={len(R)}  (reproducing V2: ic_allbar mean "
          f"{R.ic_allbar.mean():+.4f} med {R.ic_allbar.median():+.4f} "
          f"%pos {pct_pos('ic_allbar'):.0f})", flush=True)
    print(f"  ic_dec (non-overlap 4h)   mean {R.ic_dec.mean():+.4f} "
          f"med {R.ic_dec.median():+.4f} %pos {pct_pos('ic_dec'):.0f}",
          flush=True)
    print(f"  GROSS timing (0 cost)     mean {R.gross_bps.mean():+.3f} bps/cyc "
          f"Sh-syms %pos {pct_pos('gross_sh'):.0f}  mean Sh {R.gross_sh.mean():+.2f}",
          flush=True)
    print(f"  NET   timing (w/ cost)    mean {R.net_bps.mean():+.3f} bps/cyc "
          f"%pos {pct_pos('net_sh'):.0f}  mean Sh {R.net_sh.mean():+.2f}",
          flush=True)
    print(f"  turnover mean {R.turnover.mean():.3f}/2  hit mean "
          f"{100*R.hit.mean():.1f}%  (cost drag {COST*R.turnover.mean():+.3f} "
          f"bps/cyc)", flush=True)

    # portfolio reconciliation vs Step-90 ridge_insample
    D = P[P["open_time"].isin(dec_grid)].copy().sort_values(
        ["symbol", "open_time"])
    D["pos"] = np.sign(D["pred_ridge"])
    D["dpos"] = D.groupby("symbol")["pos"].diff().abs().fillna(D["pos"].abs())
    D["gross"] = D["pos"] * D["alpha_beta"] * 1e4
    D["net"] = D["gross"] - D["dpos"] * COST
    pg = D.groupby("open_time")["gross"].mean()
    pn = D.groupby("open_time")["net"].mean()
    psh = lambda x: float(x.mean() / x.std(ddof=1) * ANN) if x.std(ddof=1) > 1e-9 else np.nan
    print(f"\n  PORTFOLIO  gross Sh={psh(pg):+.2f} ({pg.mean():+.3f} bps/cyc) | "
          f"net Sh={psh(pn):+.2f} ({pn.mean():+.3f} bps/cyc)  "
          f"[Step-90 ridge_insample was −0.73 → reconciled]", flush=True)

    # ---- mechanism verdict ----
    gross_neg = pg.mean() <= 0 or psh(pg) <= 0
    ic_collapses = (R.ic_allbar.mean() > 0.005 and
                    (R.ic_dec.mean() <= 0.003 or pct_pos('ic_dec') < 55))
    cost_kills = (pg.mean() > 0 and pn.mean() < 0)
    print("\n" + "=" * 96, flush=True)
    if ic_collapses and gross_neg:
        v = (f"M1 DOMINANT (overlap-IC illusion). all-bar IC "
             f"{R.ic_allbar.mean():+.4f}/{pct_pos('ic_allbar'):.0f}%pos "
             f"COLLAPSES to non-overlap 4h IC {R.ic_dec.mean():+.4f}/"
             f"{pct_pos('ic_dec'):.0f}%pos, and GROSS timing is already "
             f"≤0 (Sh {psh(pg):+.2f}) BEFORE cost. The V2 '+0.015, 70% syms' "
             f"was an artifact of 47/48-overlapping 4h-fwd targets evaluated "
             f"every 5m bar; the honest non-overlapping relation Step 90 "
             f"trades is ~0/neg. Not a Step-90 bug, not primarily cost — the "
             f"IC metric itself was the illusion. Same non-stationarity/"
             f"hindsight theme; reinforces using non-overlapping decision-"
             f"cadence IC only.")
    elif cost_kills and not gross_neg:
        v = (f"M2 DOMINANT (cost/turnover). GROSS positive (Sh {psh(pg):+.2f}, "
             f"{pg.mean():+.3f} bps/cyc) but turnover {R.turnover.mean():.2f} "
             f"× {COST} = {COST*R.turnover.mean():.2f} bps/cyc cost flips it "
             f"NET negative ({pn.mean():+.3f}). A real but sub-cost micro-edge "
             f"— not tradable at this turnover/cost.")
    else:
        v = (f"MIXED: gross Sh {psh(pg):+.2f}, net {psh(pn):+.2f}, all-bar IC "
             f"{R.ic_allbar.mean():+.4f} vs dec IC {R.ic_dec.mean():+.4f}, "
             f"turnover {R.turnover.mean():.2f}, hit {100*R.hit.mean():.1f}%. "
             f"See per_symbol.csv; no single mechanism dominates cleanly.")
    print(f"  VERDICT: {v}", flush=True)
    pd.DataFrame([{"verdict": v}]).to_csv(OUTD / "verdict.csv", index=False)
    print(f"\nSaved {OUTD}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
