"""A/B: PRICE vs FLOW vs COMBINED model — re-run HONESTLY (PIT dvol gate on).

Re-validates the foundational two-book champion claim (docs: combined +3.71 > flow +3.50 > price +2.95,
single-price-full +3.01) under the PIT liquidity gate (CONVEXITY_PIT_DVOL=1), removing the end-of-sample
dvol look-ahead. Configs (all single-book unless noted), each run through the identical replay+combine:

  price_full   V0 model, FULL eligible universe              (single book)
  flow_full    V0+flow model, FULL universe (= "unified")     (single book; was claimed harmful)
  flow_hv80    V0+flow on the top-80-rvol subset = BookA       (single book)
  price_rest   V0 on the non-hv80 rest        = BookB          (single book)
  combined     0.5*BookA + 0.5*BookB                           (two-book champion)

flow_hv80 / price_rest / combined are read from the monthly-split PIT rerun (live/state/ab_split_pit/monthly);
this script only adds the two FULL-universe single-book replays. Run AFTER ab_split_rerank.py (PIT) finishes.

Usage: CONVEXITY_PIT_DVOL=1 python3 live/ab_model_compare.py
"""
import sys, os, json, subprocess
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
HLDIR  = REPO/"live/state/convexity/hl"
OUT    = REPO/"live/state/ab_model"
RERUN  = REPO/os.environ.get("AB_RERUN","live/state/ab_split_pit/monthly")   # two-book pieces (monthly split, PIT)
OOS_START = pd.Timestamp("2025-10-04", tz="UTC")


def sharpe(p):
    p = np.asarray(p, float); p = p[np.isfinite(p)]
    return float(p.mean()/p.std()*np.sqrt(6*365)) if p.std() > 0 else np.nan


def run_replay(preds_path, state_dir):
    env = dict(os.environ, PYTHONPATH=str(REPO), CONVEXITY_PREDS_PATH=str(preds_path),
               CONVEXITY_STATE=str(state_dir), STRAT_K="3", SIDE_MODE="default",
               CONVEXITY_PIT_DVOL=os.environ.get("CONVEXITY_PIT_DVOL", "1"))
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "-m", "live.convexity_paper_bot", "--replay-all"],
                       env=env, cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"\n[replay FAIL {state_dir}]\n{r.stderr[-2000:]}\n"); r.check_returncode()
    return Path(state_dir)/"cycles.csv"


def book_stats(cycles_csv):
    c = pd.read_csv(cycles_csv); p = c["pnl_bps"].to_numpy()
    return sharpe(p), float(p.sum()), float((c["equity_post"].cummax() - c["equity_post"]).max() * -1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # --- full-universe single books (new replays) ---
    ca = run_replay(HLDIR/"v0full_hl60.parquet",   OUT/"price_full")
    s, pnl, dd = book_stats(ca); rows.append(("price_full", s, pnl, dd))
    cf = run_replay(HLDIR/"fullflow_hl60.parquet", OUT/"flow_full")
    s, pnl, dd = book_stats(cf); rows.append(("flow_full", s, pnl, dd))

    # --- two-book pieces from the monthly PIT rerun ---
    summ = json.load(open(RERUN/"combine/twobook_summary.json"))
    sA, pA, ddA = book_stats(RERUN/"stateA/cycles.csv")
    sB, pB, ddB = book_stats(RERUN/"stateB/cycles.csv")
    rows.append(("flow_hv80 (BookA)", sA, pA, ddA))
    rows.append(("price_rest (BookB)", sB, pB, ddB))
    rows.append(("combined (two-book)", summ["sharpe_both_active"], summ["totPnL_both_active"], summ["maxDD_both_active"]))

    R = pd.DataFrame(rows, columns=["config", "sharpe", "totPnL", "maxDD"]).set_index("config")
    print("\n========  PRICE vs FLOW vs COMBINED  (PIT dvol honest, K=3, full OOS)  ========")
    print(R.round({"sharpe": 3, "totPnL": 0, "maxDD": 0}).to_string())
    print("\nprior (look-ahead) docs: combined +3.71, flow_hv80 +3.50, price_rest +2.95, single-price-full +3.01")
    R.to_csv(OUT/"summary.csv"); print(f"\nsaved -> {OUT/'summary.csv'}")


if __name__ == "__main__":
    main()
