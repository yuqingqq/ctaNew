"""Step 39: V2 fixed WITHOUT conv_gate.

Re-run V2 fixed predictions through V3.1 sleeve with conv_gate disabled
(monkey-patch GATE_PCTILE to 0 so no cycle is skipped). Other gates intact:
  - rolling-IC top-15 universe filter
  - PM_M=2 persistence
  - filter_refill (winner-stay, loser-out)
  - V3.1 6-sleeve × 24h hold

Reference: V2 fixed WITH conv_gate (Step 34/35) = Sharpe +2.19.
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
from ml.research.alpha_v4_xs import block_bootstrap_ci

# --- DISABLE conv_gate ---
ORIG_GATE_PCTILE = psl.GATE_PCTILE
psl.GATE_PCTILE = 0.0  # Setting to 0 means thr = min(hist) → disp < min never True → never skip

PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
PREDS_DIR = REPO / "linear_model/results/step34_v1_fixed"
OUT = REPO / "linear_model/results"

OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
HOLD_BARS = 288


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


def get_listings():
    L = {}
    for d in KLINES_DIR.iterdir():
        if not d.is_dir(): continue
        m5 = d / "5m"
        if not m5.exists(): continue
        f = sorted(m5.glob("*.parquet"))
        if not f: continue
        try: L[d.name] = pd.Timestamp(f[0].stem, tz="UTC")
        except Exception: pass
    return L


def aggregate_hold_through(records, alpha_wide):
    sleeve_queue = deque(maxlen=psl.N_SLEEVES)
    prev_weights = {}
    bar_freq = pd.Timedelta(minutes=5)
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]; fold = rec["fold"]
        if rec["traded"]:
            sleeve_queue.append({"entry_time":t, "longs":list(rec["long_basket"]),
                                  "shorts":list(rec["short_basket"])})
        max_age = HOLD_BARS * bar_freq
        sleeve_queue = deque(
            [s for s in sleeve_queue if (t - s["entry_time"]) < max_age],
            maxlen=psl.N_SLEEVES)
        tw = defaultdict(float)
        sw = 1.0 / psl.N_SLEEVES
        for sl in sleeve_queue:
            nL, nS = len(sl["longs"]), len(sl["shorts"])
            if nL == 0 or nS == 0: continue
            for s in sl["longs"]: tw[s] += sw * (1.0/nL)
            for s in sl["shorts"]: tw[s] -= sw * (1.0/nS)
        gross = 0.0
        if t in alpha_wide.index:
            a = alpha_wide.loc[t]
            for sym, w in prev_weights.items():
                if sym in a.index and not pd.isna(a[sym]):
                    gross += w * a[sym] * 1e4
        syms = set(tw.keys()) | set(prev_weights.keys())
        abs_d = sum(abs(tw.get(s,0)-prev_weights.get(s,0)) for s in syms)
        cost = abs_d * psl.COST_PER_UNIT_ABS_DELTA
        rows.append({"time":t,"fold":fold,"gross_pnl_bps":gross,"cost_bps":cost,
                     "net_pnl_bps":gross-cost,"turnover":abs_d})
        prev_weights = dict(tw)
    return pd.DataFrame(rows)


def main():
    print("=" * 100, flush=True)
    print("  STEP 39: V2 fixed WITHOUT conv_gate", flush=True)
    print("=" * 100, flush=True)
    print(f"  GATE_PCTILE: {ORIG_GATE_PCTILE} → {psl.GATE_PCTILE} (disabled)", flush=True)
    print(f"  Kept gates: universe filter, PM_M={psl.PM_M}, filter_refill", flush=True)
    print()
    t0 = time.time()
    listings = get_listings()

    apd_full = pd.read_parquet(PREDS_DIR / "v2_fixed_predictions.parquet")
    apd_full["open_time"] = pd.to_datetime(apd_full["open_time"], utc=True)
    apd_full["alpha_A"] = apd_full["alpha_beta"]
    if "exit_time" not in apd_full.columns or "return_pct" not in apd_full.columns:
        extra = pd.read_parquet(PANEL,
                                  columns=["symbol","open_time","exit_time","return_pct"])
        extra["open_time"] = pd.to_datetime(extra["open_time"], utc=True)
        extra["exit_time"] = pd.to_datetime(extra["exit_time"], utc=True)
        apd_full = apd_full.merge(extra, on=["symbol","open_time"], how="left")

    panel_syms = set(apd_full["symbol"].unique())
    for s, t in apd_full.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t
    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd_full[apd_full["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]

    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]
    apd_full["pred"] = apd_full["pred_z"]
    universe = psl.build_rolling_ic_universe(apd_full, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide = apd_full.pivot_table(index="open_time", columns="symbol",
                                        values="alpha_A", aggfunc="first").sort_index()

    print(f"\n{'='*100}", flush=True)
    print(f"  V2 fixed predictions + V3.1 sleeve, NO conv_gate", flush=True)
    print(f"{'='*100}", flush=True)
    results = []
    for sub, col in [("A","pred_z"), ("B","pred_B")]:
        apd_v = apd_full.copy(); apd_v["pred"] = apd_v[col]
        records = psl.run_production_protocol_save_sleeves(apd_v, universe)
        df_v = aggregate_hold_through(records, alpha_wide)
        net = df_v["net_pnl_bps"].to_numpy()
        sh = _sharpe(net)
        sh_lo, sh_hi = block_bootstrap_ci(net, statistic=_sharpe,
                                            block_size=7, n_boot=1000)[1:]
        n_traded = (df_v["gross_pnl_bps"] != 0).sum()
        df_v.to_csv(OUT / f"step39_no_conv_gate_{sub}.csv", index=False)
        sub_label = "baseline (pred_z)" if sub == "A" else "IC_signed (pred_B)"
        print(f"  {sub} {sub_label}: Sharpe={sh:+.2f} [{sh_lo:+.2f},{sh_hi:+.2f}]  "
              f"folds+={folds_positive(df_v)}/9  gross={df_v['gross_pnl_bps'].mean():+.2f}  "
              f"traded={n_traded}/{len(df_v)}", flush=True)
        results.append({"sub":sub, "sharpe":sh, "sh_lo":sh_lo, "sh_hi":sh_hi,
                         "folds_pos":folds_positive(df_v), "n_traded":n_traded})

    sh_B = results[1]["sharpe"]
    df_v_B = pd.read_csv(OUT / "step39_no_conv_gate_B.csv")
    print(f"\n  LOFO on B (Sharpe = {sh_B:+.2f}):", flush=True)
    for excl in range(1, 10):
        rem = df_v_B[df_v_B["fold"] != excl]["net_pnl_bps"].to_numpy()
        sh_rem = _sharpe(rem)
        d = sh_rem - sh_B
        flag = "  ← drives" if d < -0.4 else ""
        print(f"    excl {excl}: {sh_rem:+.2f} (Δ {d:+.2f}){flag}", flush=True)

    print(f"\n{'='*100}", flush=True)
    print(f"  COMPARISON", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  V2 fixed + sleeve + ALL gates (Step 34/35):  Sharpe +2.19 (7/9 folds+)", flush=True)
    print(f"  V2 fixed + sleeve + NO conv_gate (THIS):     Sharpe {sh_B:+.2f} "
          f"({results[1]['folds_pos']}/9 folds+)", flush=True)
    print(f"  V2 fixed + raw 4h cycle, no sleeve (Step 36): Sharpe -7.45 (1/9 folds+)",
          flush=True)
    print(f"\n  conv_gate contribution: {2.19 - sh_B:+.2f} Sharpe", flush=True)
    print(f"  Cycles traded: {results[1]['n_traded']}/{len(df_v_B)} = "
          f"{results[1]['n_traded']/len(df_v_B)*100:.0f}% (vs ~50% with conv_gate)",
          flush=True)
    print(f"\n  Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
