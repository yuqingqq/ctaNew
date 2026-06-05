"""v1 optimization loop engine (book-B-only + resid_rev + K=3). Runs a BATCH of env-toggle configs through the
walk-forward harness, reports sharpe_B + per-fold-positive + maxDD vs baseline. Cheap screen (no retrain) — winners
go to rigorous per-fold + matched-placebo validation separately.

Usage: python3 live/v1_opt_loop.py BATCHNAME   (batches defined below)
"""
import sys, os, json, subprocess
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27","2026-05-31"]]

BATCHES = {
  # iteration 1 — broad env-toggle re-test on the NEW book-B structure (prior tests were two-book/51-panel)
  "batch1": {
    "baseline":      {},
    "gate_off":      {"REGIME_BULL_THR":"99","REGIME_BEAR_THR":"-99"},   # remove dynamic regime gate (user ask)
    "bull_pred":     {"BULL_MODE":"sidealpha"},                          # bull regime use pred not mom
    "bull_bnmom":    {"BULL_MODE":"betaneut_mom"},
    "size_invvol":   {"SIZING_MODE":"inv_vol"},                          # vol-scaled leg sizing
    "size_invsqrt":  {"SIZING_MODE":"inv_sqrt_vol"},
    "size_volcap":   {"SIZING_MODE":"volcap"},
    "idioskip90":    {"LONG_IDIO_SKIP_PCT":"0.90"},                      # falling-knife idio-vol skip (long)
    "idioskip80":    {"LONG_IDIO_SKIP_PCT":"0.80"},
    "dispgate":      {"DISP_GATE":"1"},                                  # conviction/dispersion gate
    "stop_off":      {"STOP_G_FLOOR":"1.0"},                             # disable DD stop
    "stop_tight":    {"STOP_K_SIGMA":"1.5"},
    "sidemode_lms":  {"SIDE_MODE":"longmom_shortmr"},
    "sidemode_lds":  {"SIDE_MODE":"longdef_shortmr"},
  },
  # iteration 3 — hold/sleeve horizon (cost-amortization lever; 24h/6-sleeve was tuned on the OLD structure)
  "batch3": {
    "hold1_4h":   {"AB_HOLD_B":"1"},    # 4h hold (no overlap)
    "hold3_12h":  {"AB_HOLD_B":"3"},
    "hold6_24h":  {"AB_HOLD_B":"6"},    # = production baseline
    "hold9_36h":  {"AB_HOLD_B":"9"},
    "hold12_48h": {"AB_HOLD_B":"12"},
  },
  # iteration 2 — SIDE_MODE variants wired CORRECTLY (via AB_SIDEMODE_B, not SIDE_MODE which the harness overrides)
  "batch2": {
    "baseline":      {},
    "lms":           {"AB_SIDEMODE_B":"longmom_shortmr"},
    "lds":           {"AB_SIDEMODE_B":"longdef_shortmr"},
    "regime_switch": {"AB_SIDEMODE_B":"regime_switch"},
    "conf_btc_hedge":{"AB_SIDEMODE_B":"confidence_btc_hedge"},
  },
}


def run_one(name, extra_env, outbase):
    od = outbase/name
    env = dict(os.environ, STRAT_K="3",
               AB_HLDIR="live/state/convexity/hl", AB_HLDIR_LONG="live/state/convexity/hl_residrev",
               AB_OUTBASE=str(od), AB_SIDEMODE_A="default")
    env.setdefault("AB_SIDEMODE_B", "default")
    env.update(extra_env)
    r = subprocess.run([sys.executable, "live/ab_split_rerank.py", "--n", "80", "--policies", "monthly"],
                       cwd=str(REPO), env=env, capture_output=True, text=True)
    sj = od/"monthly/combine/twobook_summary.json"
    if not sj.exists():
        return {"config": name, "sharpe_B": np.nan, "note": "FAIL:"+r.stderr.strip().split(chr(10))[-1][:60]}
    s = json.load(open(sj))
    c = pd.read_csv(od/"monthly/stateB/cycles.csv")
    pcol = "pnl_bps" if "pnl_bps" in c else [x for x in c.columns if "pnl" in x][0]
    pb = c[pcol] if pcol == "pnl_bps" else c[pcol]*1e4
    ot = pd.to_datetime(c["open_time"], utc=True) if "open_time" in c else None
    # per-fold positive count
    fpos = np.nan
    if ot is not None:
        fold = pd.cut(ot, bins=CUTS, labels=False, right=False)
        fp = pb.groupby(fold).sum()
        fpos = int((fp > 0).sum()); nf = int(fp.notna().sum())
    e = pb.cumsum(); dd = (e-e.cummax()).min()
    return {"config": name, "sharpe_B": s["sharpe_bookB"], "maxDD": round(dd), "totPnL": round(pb.sum()),
            "folds_pos": f"{fpos}/{nf}" if ot is not None else "?"}


def main():
    batch = sys.argv[1] if len(sys.argv) > 1 else "batch1"
    cfgs = BATCHES[batch]
    outbase = REPO/f"live/state/opt_loop/v1loop_{batch}"; outbase.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, env in cfgs.items():
        print(f"... running {name} {env}", flush=True)
        rows.append(run_one(name, env, outbase));
        pd.DataFrame(rows).to_csv(outbase/"results.csv", index=False)
    R = pd.DataFrame(rows)
    base = R[R.config=="baseline"]["sharpe_B"].iloc[0] if "baseline" in R.config.values else np.nan
    R["lift"] = (R["sharpe_B"]-base).round(3)
    R = R.sort_values("sharpe_B", ascending=False)
    print(f"\n================ v1 OPT LOOP [{batch}] (book-B + resid_rev, K=3; baseline={base:+.3f}) ================")
    print(R.to_string(index=False))
    R.to_csv(outbase/"results.csv", index=False)
    print(f"\nsaved -> {outbase/'results.csv'}")
    print("BATCHDONE")


if __name__ == "__main__":
    main()
