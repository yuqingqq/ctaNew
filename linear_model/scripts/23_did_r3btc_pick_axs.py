"""Step 23: Did R3_BTC actually pick AXSUSDT during the fold-6 winning window?

Re-run the protocol with detailed pick tracking. Look at:
  1. Was AXSUSDT in the long basket during Jan 17-22 cycles?
  2. What was AXS's pred_z value in those cycles?
  3. When did R3_BTC START longing AXS — early in rally or late?
  4. What were the shorts during this period (e.g. TIA, JTO, ENA)?
  5. Confirm by simulating: AXS-only PnL during the window
"""
from __future__ import annotations
import sys, warnings, importlib.util
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)

R3_BTC_PREDS = REPO / "linear_model/results/ridge_r3_btc_preds.parquet"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

WIN_START = pd.Timestamp("2026-01-15", tz="UTC")
WIN_END = pd.Timestamp("2026-01-25", tz="UTC")
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60


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


def compute_trailing_ic(apd, sampled_t, win_days=90):
    apd_s = apd[apd["open_time"].isin(set(sampled_t))].sort_values(
        ["symbol","open_time"]).reset_index(drop=True)
    cycles_per_day = 6
    win_cycles = win_days * cycles_per_day
    rows = []
    for sym, g in apd_s.groupby("symbol"):
        g = g.sort_values("open_time").reset_index(drop=True)
        pred = g["pred_z"].to_numpy(); alpha = g["alpha_beta"].to_numpy()
        n = len(g)
        ics = np.full(n, np.nan)
        for i in range(50, n):
            lo = max(0, i - win_cycles)
            p, a = pred[lo:i], alpha[lo:i]
            mask = ~np.isnan(p) & ~np.isnan(a)
            if mask.sum() < 50: continue
            pr = pd.Series(p[mask]).rank().to_numpy()
            ar = pd.Series(a[mask]).rank().to_numpy()
            if pr.std() < 1e-6 or ar.std() < 1e-6: continue
            ics[i] = np.corrcoef(pr, ar)[0,1]
        for j, t in enumerate(g["open_time"]):
            rows.append({"symbol":sym, "open_time":t, "trail_ic":ics[j]})
    return pd.DataFrame(rows).fillna(0)


def main():
    print("=== Step 23: Did R3_BTC pick AXS during fold-6 winning window? ===\n",
          flush=True)
    listings = get_listings()

    apd = pd.read_parquet(R3_BTC_PREDS)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    apd["alpha_A"] = apd["alpha_beta"]
    if "return_pct" not in apd.columns:
        base = pd.read_parquet(PANEL, columns=["symbol","open_time","return_pct"])
        base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
        apd = apd.merge(base, on=["symbol","open_time"], how="left")
    apd["pred"] = apd["pred_z"]

    panel_syms = set(apd["symbol"].unique())
    for s, t in apd.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    universe = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, elig_pit)

    # IC-signed pred
    df_ic = compute_trailing_ic(apd, sampled_t, 90)
    apd_full = apd.merge(df_ic, on=["symbol","open_time"], how="left")
    apd_full["trail_ic"] = apd_full["trail_ic"].fillna(0)
    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]
    apd_full["pred"] = apd_full["pred_B"]

    print("Running protocol to get picks records...", flush=True)
    records = psl.run_production_protocol_save_sleeves(apd_full, universe)
    records["time"] = pd.to_datetime(records["time"], utc=True)

    # Filter to winning window
    win = records[(records["time"] >= WIN_START) & (records["time"] <= WIN_END)
                  & records["traded"]]
    print(f"\nCycles in window: {len(win)}", flush=True)

    # ===== Was AXS in long basket? =====
    print(f"\n--- AXSUSDT in long basket during winning window ---", flush=True)
    print(f"  {'time':<20}  {'long_basket':<40}  {'short_basket':<40}", flush=True)
    axs_long = 0; axs_short = 0
    for _, rec in win.iterrows():
        lb = rec["long_basket"]; sb = rec["short_basket"]
        t = rec["time"]
        in_long = "AXSUSDT" in lb
        in_short = "AXSUSDT" in sb
        if in_long: axs_long += 1
        if in_short: axs_short += 1
        marker = " ← AXS LONG" if in_long else (" ← AXS SHORT" if in_short else "")
        print(f"  {str(t)[:19]}  {str(lb):<40}  {str(sb):<40}{marker}", flush=True)

    print(f"\n  AXSUSDT in LONG basket: {axs_long}/{len(win)} cycles",
          flush=True)
    print(f"  AXSUSDT in SHORT basket: {axs_short}/{len(win)} cycles",
          flush=True)

    # ===== AXS pred_z values during window =====
    print(f"\n--- AXS prediction values (pred_z and pred_B) ---", flush=True)
    axs_in_win = apd_full[(apd_full["symbol"] == "AXSUSDT")
                          & (apd_full["open_time"] >= WIN_START)
                          & (apd_full["open_time"] <= WIN_END)
                          & (apd_full["open_time"].isin(set(sampled_t)))]
    print(f"  {'time':<20}  {'pred_z':>10}  {'trail_ic':>10}  {'pred_B':>10}  "
          f"{'alpha_β bps':>12}", flush=True)
    for _, r in axs_in_win.iterrows():
        ab_bps = r["alpha_beta"] * 1e4 if not pd.isna(r["alpha_beta"]) else 0
        print(f"  {str(r['open_time'])[:19]}  {r['pred_z']:>+10.4f}  "
              f"{r['trail_ic']:>+10.4f}  {r['pred_B']:>+10.4f}  "
              f"{ab_bps:>+12.1f}", flush=True)

    # ===== Top symbols by pick frequency in winning window =====
    print(f"\n--- Top symbols in long basket (winning window) ---", flush=True)
    long_freq = defaultdict(int)
    short_freq = defaultdict(int)
    for _, rec in win.iterrows():
        for s in rec["long_basket"]: long_freq[s] += 1
        for s in rec["short_basket"]: short_freq[s] += 1
    long_sorted = sorted(long_freq.items(), key=lambda x: -x[1])
    short_sorted = sorted(short_freq.items(), key=lambda x: -x[1])
    print(f"  Long-basket frequency:", flush=True)
    for s, c in long_sorted[:10]:
        print(f"    {s:<14} {c:>3}/{len(win)} cycles", flush=True)
    print(f"  Short-basket frequency:", flush=True)
    for s, c in short_sorted[:10]:
        print(f"    {s:<14} {c:>3}/{len(win)} cycles", flush=True)


if __name__ == "__main__":
    main()
