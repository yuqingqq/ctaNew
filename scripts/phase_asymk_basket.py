"""Phase ASYMK: asymmetric K (K_long=5, K_short=3) with equal-capital basket.

Pre-registered (no fitted threshold):
  - K_LONG = 5 (dilute the noisier long side: 47.8% correct rate)
  - K_SHORT = 3 (concentrate the higher-signal short side: 57.4% correct rate)
  - Each side weighted 1/K per pick → gross long = gross short = 1.0
  - BETA NEUTRAL preserved
  - All other V3.1 machinery unchanged (rolling-IC universe, conv_gate,
    filter_refill, flat_real, 6-sleeve equal-weight overlay)

Hypothesis: diluting long-side noise (where model is worse than random) across
more picks should improve cohort PnL via noise averaging, while keeping the
short side (real alpha) concentrated.

Validation: 6-gate same as iter loop. Plus inverse asymmetry (K_long=3, K_short=5)
as a sanity-check placebo — if the symmetric inverse is BETTER than our chosen
direction, the asymmetry hypothesis is wrong.
"""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(psl)
spec2 = importlib.util.spec_from_file_location(
    "svar", REPO / "scripts/phase_ah_sleeve_variants.py")
svar = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(svar)

OUT = REPO / "outputs/vBTC_phase_ASYMK"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON_ENTRY = 48
HOLD_BARS = 288
N_SLEEVES = 6
COST_PER_UNIT_ABS_DELTA = 2.25
CYCLES_PER_YEAR = (288 * 365) / 48
OOS_FOLDS = list(range(1, 10))
V31_REF_SHARPE = 2.23
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def block_bootstrap_ci(x, stat=_sharpe, block_size=7, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < block_size + 2: return stat(x), stat(x), stat(x)
    rng = np.random.RandomState(seed)
    n = len(x); nb = n // block_size + 1
    boots = []
    for _ in range(n_boot):
        starts = rng.randint(0, n - block_size + 1, size=nb)
        blocks = np.concatenate([x[s:s+block_size] for s in starts])[:n]
        boots.append(stat(blocks))
    boots = np.array(boots)
    return float(stat(x)), float(np.percentile(boots, 100 * alpha / 2)), \
           float(np.percentile(boots, 100 * (1 - alpha / 2)))


def build_asymk_sleeves(apd, universe, K_LONG, K_SHORT):
    """Production protocol with asymmetric K. Equal-capital per side."""
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON_ENTRY])
    df = df[df["open_time"].isin(keep_t)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()
    hist_disp = deque(maxlen=GATE_LOOKBACK)
    hist_basket = []
    cur_long, cur_short = set(), set()
    is_flat = False
    picks_hist = defaultdict(list)
    by_t = {t: g for t, g in df.groupby("open_time")}
    records = []
    for t in times:
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < K_LONG + K_SHORT + 1:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "traded": False})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))
        ret_l = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        # Top K_LONG and bottom K_SHORT
        idx_t = np.argpartition(-pred_arr, K_LONG - 1)[:K_LONG]
        idx_b = np.argpartition(pred_arr, K_SHORT - 1)[:K_SHORT]
        disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
        skip = False
        if len(hist_disp) >= 30:
            thr = float(np.quantile(list(hist_disp), GATE_PCTILE))
            if disp < thr: skip = True
        hist_disp.append(disp)
        if skip:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "traded": False})
            if not is_flat and (cur_long or cur_short):
                is_flat = True; cur_long, cur_short = set(), set()
            continue
        order_d = np.argsort(-pred_arr); order_a = np.argsort(pred_arr)
        long_r = [sym_arr[i] for i in order_d]
        short_r = [sym_arr[i] for i in order_a]
        cand_l, _ = psl.select_refill(long_r, "long", K_LONG, picks_hist, 90, t)
        cand_s, _ = psl.select_refill(short_r, "short", K_SHORT, picks_hist, 90, t)
        c_ls = set(cand_l); c_ss = set(cand_s)
        hist_basket.append({"long": c_ls, "short": c_ss})
        if len(hist_basket) > PM_M:
            hist_basket = hist_basket[-PM_M:]
        if len(hist_basket) >= PM_M:
            p_l = [h["long"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            p_s = [h["short"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            nl = cur_long & c_ls; ns = cur_short & c_ss
            for s_ in c_ls - cur_long:
                if all(s_ in p for p in p_l): nl.add(s_)
            for s_ in c_ss - cur_short:
                if all(s_ in p for p in p_s): ns.add(s_)
            if len(nl) > K_LONG:
                nl = set(sorted(nl, key=lambda s_: -pred_arr[np.where(sym_arr == s_)[0][0]])[:K_LONG])
            if len(ns) > K_SHORT:
                ns = set(sorted(ns, key=lambda s_: pred_arr[np.where(sym_arr == s_)[0][0]])[:K_SHORT])
        else:
            nl, ns = c_ls, c_ss
        if not nl or not ns:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "traded": False})
            if not is_flat and (cur_long or cur_short):
                is_flat = True; cur_long, cur_short = set(), set()
            continue
        for s_ in nl:
            picks_hist[(s_, "long")].append((t, exit_l[s_], ret_l[s_] * 1e4 / len(nl)))
        for s_ in ns:
            picks_hist[(s_, "short")].append((t, exit_l[s_], -ret_l[s_] * 1e4 / len(ns)))
        records.append({"time": t, "fold": fold_lookup.get(t, 0),
                          "long_basket": sorted(list(nl)),
                          "short_basket": sorted(list(ns)),
                          "traded": True})
        cur_long, cur_short = nl, ns
        is_flat = False
    return pd.DataFrame(records)


def main():
    print("=== Phase ASYMK: K_long=5, K_short=3 (with K_long=3,K_short=5 placebo) ===\n",
          flush=True)

    # Load audit panel
    apd = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet")
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    print(f"  loaded {len(apd):,} predictions", flush=True)

    # Build rolling-IC universe (same as V3.1)
    listings = psl.get_listings()
    panel_first_obs = apd.groupby("symbol")["open_time"].min()
    for sym, t in panel_first_obs.items():
        if sym not in listings:
            listings[sym] = t
    def elig_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in apd["symbol"].unique()
                  if listings.get(s) and listings[s] <= cutoff}
    tgt = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled = tgt[::HORIZON_ENTRY]
    print(f"  building rolling-IC universe...", flush=True)
    t0 = time.time()
    universe = psl.build_rolling_ic_universe(apd, sampled, psl.TOP_N, elig_at)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # Build close prices for fwd_rets
    all_syms = sorted(apd["symbol"].unique())
    print(f"  loading close prices...", flush=True)
    t0 = time.time()
    close_wide = svar.load_close_wide(all_syms)
    fwd_rets_4h = (close_wide.shift(-HORIZON_ENTRY) - close_wide) / close_wide
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # Three variants
    variants = [
        ("V3.1_baseline_K3_K3", 3, 3),
        ("ASYMK_K5_K3 (long dilute)", 5, 3),
        ("ASYMK_K3_K5 (placebo inverse)", 3, 5),
    ]
    results = {}
    for label, KL, KS in variants:
        print(f"\n  Building sleeves: {label}...", flush=True)
        t0 = time.time()
        sleeves = build_asymk_sleeves(apd, universe, KL, KS)
        n_tr = sleeves["traded"].sum()
        print(f"  done: traded {n_tr}/{len(sleeves)} cycles ({time.time()-t0:.0f}s)",
              flush=True)
        sleeves["time"] = pd.to_datetime(sleeves["time"], utc=True)
        sleeves.to_parquet(OUT / f"sleeves_{label.split()[0]}.parquet", index=False)

        # V3.1 aggregation
        df_v = svar.aggregate_sleeves_variant(sleeves, fwd_rets_4h, N_SLEEVES,
                                                    HOLD_BARS, sleeve_weights=[1/6]*6)
        results[label] = df_v
        sh = _sharpe(df_v["net_pnl_bps"])
        dd = _max_dd(df_v["net_pnl_bps"])
        npos = sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)
        print(f"  → Sharpe = {sh:+.3f}  maxDD = {dd:+.0f}  "
              f"PnL = {df_v['net_pnl_bps'].sum():+.0f}  folds+ = {npos}/9",
              flush=True)
        df_v.to_csv(OUT / f"per_cycle_{label.split()[0]}.csv", index=False)

    # Comparison
    print(f"\n=== Comparison ===\n", flush=True)
    base = results["V3.1_baseline_K3_K3"]
    asym = results["ASYMK_K5_K3 (long dilute)"]
    inv = results["ASYMK_K3_K5 (placebo inverse)"]

    sh_base = _sharpe(base["net_pnl_bps"])
    sh_asym = _sharpe(asym["net_pnl_bps"])
    sh_inv = _sharpe(inv["net_pnl_bps"])
    print(f"  V3.1 K3-K3 baseline:        Sharpe = {sh_base:+.3f}", flush=True)
    print(f"  ASYMK K5-K3 (long dilute):  Sharpe = {sh_asym:+.3f}  "
          f"(Δ = {sh_asym-sh_base:+.3f})", flush=True)
    print(f"  ASYMK K3-K5 (inverse):      Sharpe = {sh_inv:+.3f}  "
          f"(Δ = {sh_inv-sh_base:+.3f})", flush=True)
    print(f"\n  Direction validates hypothesis if K5-K3 > K3-K3 AND K5-K3 > K3-K5",
          flush=True)

    # Per-fold breakdown
    print(f"\n  Per-fold breakdown:", flush=True)
    print(f"  {'fold':>4}  {'K3-K3':>8}  {'K5-K3':>8}  {'K3-K5':>8}  "
          f"{'asym-base':>10}", flush=True)
    fold_diffs = {}
    for f in OOS_FOLDS:
        b = base[base["fold"] == f]["net_pnl_bps"].sum()
        a = asym[asym["fold"] == f]["net_pnl_bps"].sum()
        i = inv[inv["fold"] == f]["net_pnl_bps"].sum()
        d = a - b
        fold_diffs[f] = d
        print(f"  {f:>4}  {b:>+8.0f}  {a:>+8.0f}  {i:>+8.0f}  {d:>+10.0f}",
              flush=True)
    pos_lift = sum(v for v in fold_diffs.values() if v > 0)
    max_fold_contribution = (max(fold_diffs.values()) / pos_lift * 100) if pos_lift > 0 else 0
    print(f"\n  Max single fold contribution: {max_fold_contribution:.0f}%", flush=True)

    # Paired bootstrap
    print(f"\n  Paired K5-K3 vs K3-K3 bootstrap:", flush=True)
    paired = base[["time", "fold", "net_pnl_bps"]].rename(
        columns={"net_pnl_bps": "base"}).merge(
        asym[["time", "net_pnl_bps"]].rename(columns={"net_pnl_bps": "asym"}),
        on="time")
    paired["diff"] = paired["asym"] - paired["base"]
    def _mean(x): return float(np.mean(x))
    mu, lo, hi = block_bootstrap_ci(paired["diff"].to_numpy(), stat=_mean,
                                        block_size=7, n_boot=2000)
    print(f"  Mean diff: {mu:+.3f} bps/cycle  CI [{lo:+.3f}, {hi:+.3f}]", flush=True)
    diff_sig = (lo > 0) or (hi < 0)
    print(f"  Paired diff CI excludes 0: {'YES' if diff_sig else 'NO'}", flush=True)

    # Verdict
    print(f"\n=== Phase ASYMK Verdict ===\n", flush=True)
    g_dir = (sh_asym > sh_base) and (sh_asym > sh_inv)
    g_lift = (sh_asym - sh_base) >= 0.10
    g_sig = diff_sig
    g_folds = sum(1 for _, g in asym.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)
    g_fold_count = g_folds >= 6
    g_concentration = max_fold_contribution <= 40
    print(f"  [{'PASS' if g_dir else 'FAIL'}]  Asymmetry direction correct (K5-K3 > base AND > inverse)",
          flush=True)
    print(f"  [{'PASS' if g_lift else 'FAIL'}]  Sharpe lift ≥ +0.10  ({sh_asym - sh_base:+.2f})",
          flush=True)
    print(f"  [{'PASS' if g_sig else 'FAIL'}]  Paired diff CI excludes 0  ([{lo:+.3f}, {hi:+.3f}])",
          flush=True)
    print(f"  [{'PASS' if g_fold_count else 'FAIL'}]  ≥ 6/9 folds positive ({g_folds}/9)",
          flush=True)
    print(f"  [{'PASS' if g_concentration else 'FAIL'}]  Max fold contribution ≤ 40% ({max_fold_contribution:.0f}%)",
          flush=True)


if __name__ == "__main__":
    main()
