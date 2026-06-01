"""Walk-forward halflife meta-CV.

For each monthly retrain cutoff C:
  1) Validation slice = (C - 4w) to C
  2) For each halflife in {7, 14, 30, 60, 90, 999=no-decay}:
       a) Train inner model through (C - 4w) with this halflife
       b) Predict on validation slice
       c) Score by validation top-K minus bot-K spread (= strategy proxy)
  3) Pick best halflife for THIS cutoff
  4) Train final model through C with picked halflife
  5) Predict next month (C+1d → C+1m end)

Then stitch all next-month preds + replay through bot to get realistic OOS Sharpe.

Honest about overfit: validation slice is BEFORE deploy window (no look-ahead).
Logs which halflife each cutoff picks (stability check).
"""
from __future__ import annotations
import subprocess, sys, time, json, os
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
S = REPO/"live/state/convexity"
HALFLIVES = [7, 14, 30, 60, 90, 999]   # 999 = effectively no decay
K = 5   # selection K used to score val spread (same as strategy)

# Monthly cutoffs and the next-month prediction windows
CUTOFFS = [
    ("2025-11-30", "2025-12-01", "2025-12-31"),
    ("2025-12-31", "2026-01-01", "2026-01-31"),
    ("2026-01-31", "2026-02-01", "2026-02-28"),
    ("2026-02-28", "2026-03-01", "2026-03-31"),
    ("2026-03-31", "2026-04-01", "2026-04-30"),
]

def run(cmd, label=""):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"  !!! {label} FAILED: {res.stderr[-500:]}", flush=True)
        return False
    return True

def score_val(val_preds_path):
    """Score a val preds parquet by per-cycle top-K minus bot-K mean realized return.
    The bigger this is, the better the predictions rank syms for our strategy."""
    d = pd.read_parquet(val_preds_path)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)]
    spreads = []
    for ot, g in d.groupby("open_time"):
        if len(g) < 2*K: continue
        g = g.sort_values("pred")
        spread = g.tail(K)["return_pct"].mean() - g.head(K)["return_pct"].mean()
        if np.isfinite(spread): spreads.append(spread)
    if not spreads: return float("nan")
    return float(np.mean(spreads))

def main():
    t0 = time.time()
    chosen = {}   # cutoff -> best halflife
    for cutoff, pred_start, pred_end in CUTOFFS:
        # Validation slice = (C - 28d) to C; inner train cuts at start of val slice
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
        val_start = (cutoff_ts - pd.Timedelta(days=28)).strftime("%Y-%m-%d")
        val_end = cutoff   # = cutoff_ts
        inner_train_end = val_start

        print(f"\n=== cutoff {cutoff} | val {val_start}..{val_end} | inner-train through {inner_train_end} ===", flush=True)
        val_scores = {}
        for hl in HALFLIVES:
            tag = f"mcv_inner_{cutoff.replace('-','')}_hl{hl}"
            # train inner
            ok = run(["python3", str(REPO/"live/train_convexity_artifact.py"),
                      "--train-end", inner_train_end, "--tag", tag, "--halflife-days", str(hl)],
                     f"train {tag}")
            if not ok: val_scores[hl] = float("nan"); continue
            # predict on validation slice
            out_tag = f"{tag}_val"
            ok = run(["python3", str(REPO/"live/predict_with_artifact.py"),
                      "--artifact", tag, "--from", val_start, "--to", val_end, "--out-tag", out_tag],
                     f"predict {out_tag}")
            if not ok: val_scores[hl] = float("nan"); continue
            val_preds = S/f"x132_{out_tag}_preds.parquet"
            val_scores[hl] = score_val(val_preds)
            print(f"   hl={hl:>3}d: val spread {val_scores[hl]*1e4:+.2f} bps/cycle", flush=True)
        # pick best
        valid = {hl: s for hl, s in val_scores.items() if np.isfinite(s)}
        if not valid:
            print(f"   !!! no valid hl, defaulting to 30d"); best_hl = 30
        else:
            best_hl = max(valid, key=valid.get)
            print(f"   → BEST halflife for cutoff {cutoff}: {best_hl}d (val spread {valid[best_hl]*1e4:+.2f} bps)", flush=True)
        chosen[cutoff] = best_hl

        # Final train at cutoff with best hl
        final_tag = f"mcv_final_{cutoff.replace('-','')}"
        print(f"   final train through {cutoff} with hl={best_hl}d ...", flush=True)
        ok = run(["python3", str(REPO/"live/train_convexity_artifact.py"),
                  "--train-end", cutoff, "--tag", final_tag, "--halflife-days", str(best_hl)],
                 f"final-train {final_tag}")
        if not ok: continue
        # predict next month
        out_tag = f"{final_tag}_deploy"
        ok = run(["python3", str(REPO/"live/predict_with_artifact.py"),
                  "--artifact", final_tag, "--from", pred_start, "--to", pred_end, "--out-tag", out_tag],
                 f"deploy-predict {out_tag}")
        print(f"   deploy preds saved: x132_{out_tag}_preds.parquet [{time.time()-t0:.0f}s elapsed]", flush=True)

    # Stitch deploy preds across all cutoffs
    print(f"\n=== stitching deploy preds ===", flush=True)
    dfs = []
    for cutoff, _, _ in CUTOFFS:
        f = S/f"x132_mcv_final_{cutoff.replace('-','')}_deploy_preds.parquet"
        if not f.exists():
            print(f"  missing {f.name}"); continue
        d = pd.read_parquet(f); print(f"  {f.name}: {len(d):,} rows, picked hl={chosen[cutoff]}d")
        dfs.append(d)
    out = pd.concat(dfs, ignore_index=True).sort_values(["open_time","symbol"])
    stitched_path = S/"x132_mcv_stitched_preds.parquet"
    out.to_parquet(stitched_path)
    print(f"  stitched: {len(out):,} rows, {out['open_time'].min()} → {out['open_time'].max()}")

    print(f"\n=== HALFLIFE PICKS PER CUTOFF (stability check) ===")
    print(json.dumps(chosen, indent=2))

    # Replay
    print(f"\n=== replaying through bot (Dec→Apr) ===", flush=True)
    import os
    env = os.environ.copy()
    env.update({"PYTHONPATH":str(REPO),"CONVEXITY_PREDS_PATH":str(stitched_path),
                "BULL_MODE":"mom","REGIME_HYSTERESIS_N":"3"})
    res = subprocess.run(["python3","-m","live.convexity_paper_bot",
                          "--replay-from","2025-12-01","--replay-end","2026-04-30"],
                         capture_output=True, text=True, env=env)
    print(res.stdout[-400:] if res.stdout else "(no stdout)")
    if res.returncode != 0: print("STDERR:", res.stderr[-500:])
    # save cycles + summary
    import shutil
    shutil.copy(S/"cycles.csv", S/"mcv_cycles.csv")

    # Summary
    print(f"\n=== META-CV WALKFORWARD RESULT ===")
    c = pd.read_csv(S/"mcv_cycles.csv")
    c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    p = c["pnl_bps"]/1e4
    sh = p.mean()/p.std()*np.sqrt(6*365) if p.std()>0 else float("nan")
    cum = pd.Series(c["pnl_bps"]).cumsum(); dd = (cum-cum.cummax()).min()
    print(f"  cycles {len(c)}, Sharpe {sh:+.3f}, totPnL {int(c['pnl_bps'].sum()):+d} bps, maxDD {int(dd):+d}")
    print(f"  static reference (Dec→Apr): Sharpe +1.57")
    print(f"  rolling hl=fixed-each-month reference (Dec→Apr): Sharpe +1.61")
    print(f"\nDONE [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
