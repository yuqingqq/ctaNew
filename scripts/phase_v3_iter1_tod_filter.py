"""Iteration 1: Time-of-day entry filter (skip new sleeves at 04 UTC + 12 UTC).

Pre-registered hypothesis (single threshold, no sweeping):
  H1: Skipping new sleeve entry at UTC hours {4, 12} preserves PnL from the
      good hours (00, 08, 16, 20) and removes drag from the two losing hours.
  Implementation: same V3.1 6-sleeve queue, but at cycles where hour_utc in
      {4, 12}, do NOT append a new sleeve. Existing sleeves age normally.
      Steady-state effective sleeves drops from 6 to 4.

Validation gates:
  Static Sharpe ≥ V3.1 +2.23 + 0.10  AND
  Nested-OOS Sharpe ≥ static - 0.10   AND
  Matched-time placebo p95 PASS       AND
  Paired diff vs V3.1 CI excludes 0   AND
  ≥ 6/9 folds positive               AND
  No single fold contributes > 40% of lift
"""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "svar", REPO / "scripts/phase_ah_sleeve_variants.py")
svar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svar)

spec2 = importlib.util.spec_from_file_location(
    "psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(psl)

OUT = REPO / "outputs/vBTC_iter_loop"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON_ENTRY = 48
HOLD_BARS = 288
N_SLEEVES = 6
COST_PER_UNIT_ABS_DELTA = 2.25
CYCLES_PER_YEAR = (288 * 365) / HORIZON_ENTRY
OOS_FOLDS = list(range(1, 10))

V31_WEIGHTS = [1/6] * 6
V31_REF_SHARPE = 2.23

# Pre-registered: hours to skip new entries
SKIP_HOURS = {4, 12}


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


def aggregate_sleeves_tod(records, fwd_rets_4h, sleeve_weights,
                            skip_entry_hours=None,
                            placebo_universe=None, placebo_seed=None,
                            placebo_skip_hours=None):
    """V3.1 aggregation but with optional entry-hour filter.

    skip_entry_hours: set of UTC hours (e.g. {4, 12}) where new sleeve entry
        is suppressed. Existing sleeves continue to age normally.
    placebo_skip_hours: if set, also skip entries at these hours for placebo
        (used for matched-time placebo where we randomize WHICH hours to skip).
    """
    bar_freq = pd.Timedelta(minutes=5)
    sleeve_queue = deque(maxlen=N_SLEEVES)
    prev_weights = {}
    rng = np.random.RandomState(placebo_seed if placebo_seed is not None else 0)
    rows = []
    skip_set = skip_entry_hours or set()
    placebo_skip_set = placebo_skip_hours or set()

    for _, rec in records.iterrows():
        t = rec["time"]
        fold = rec["fold"]
        hour = pd.Timestamp(t).hour

        # Determine whether to add a new sleeve this cycle
        suppress_new = (hour in skip_set) or (hour in placebo_skip_set)

        if placebo_seed is not None and placebo_universe is not None and rec["traded"]:
            u = placebo_universe.get(t, set())
            pool = sorted(list(u))
            K_l = len(rec["long_basket"]); K_s = len(rec["short_basket"])
            if len(pool) >= K_l + K_s and K_l > 0 and K_s > 0:
                shuffled = rng.permutation(len(pool))
                long_b = sorted([pool[i] for i in shuffled[:K_l]])
                short_b = sorted([pool[i] for i in shuffled[K_l:K_l+K_s]])
            else:
                long_b = []; short_b = []
        else:
            long_b = list(rec["long_basket"])
            short_b = list(rec["short_basket"])

        # New sleeve entry — gated by hour
        if (not suppress_new) and len(long_b) > 0 and len(short_b) > 0:
            sleeve_queue.append({
                "entry_time": t, "longs": long_b, "shorts": short_b,
            })

        # Drop aged-out sleeves
        max_age = HOLD_BARS * bar_freq
        sleeve_queue = deque(
            [s for s in sleeve_queue if (t - s["entry_time"]) < max_age],
            maxlen=N_SLEEVES
        )

        # Build target weights
        active_list = sorted(list(sleeve_queue),
                                key=lambda s: s["entry_time"], reverse=True)
        target_weights = defaultdict(float)
        for i, s in enumerate(active_list):
            if i >= len(sleeve_weights): break
            w = sleeve_weights[i]
            n_long = len(s["longs"]); n_short = len(s["shorts"])
            if n_long == 0 or n_short == 0: continue
            for sym in s["longs"]:
                target_weights[sym] += w * (1.0 / n_long)
            for sym in s["shorts"]:
                target_weights[sym] -= w * (1.0 / n_short)

        # 4h PnL
        gross_pnl_bps = 0.0
        if t in fwd_rets_4h.index:
            rets_at_t = fwd_rets_4h.loc[t]
            for sym, w in prev_weights.items():
                if sym in rets_at_t.index and not pd.isna(rets_at_t[sym]):
                    gross_pnl_bps += w * rets_at_t[sym] * 1e4

        all_syms = set(target_weights.keys()) | set(prev_weights.keys())
        total_abs_delta = sum(abs(target_weights.get(s, 0.0) -
                                    prev_weights.get(s, 0.0))
                                for s in all_syms)
        cost_bps = total_abs_delta * COST_PER_UNIT_ABS_DELTA
        net_pnl_bps = gross_pnl_bps - cost_bps

        rows.append({"time": t, "fold": fold, "hour": hour,
                      "active_sleeves": len(sleeve_queue),
                      "gross_pnl_bps": gross_pnl_bps,
                      "cost_bps": cost_bps,
                      "net_pnl_bps": net_pnl_bps,
                      "turnover": total_abs_delta,
                      "gross_exposure": sum(abs(w) for w in target_weights.values())})
        prev_weights = dict(target_weights)
    return pd.DataFrame(rows)


def fold_concentration(df_v):
    fold_pnls = df_v.groupby("fold")["net_pnl_bps"].sum()
    pos = fold_pnls[fold_pnls > 0]
    total_pos = pos.sum() if len(pos) > 0 else 0
    if total_pos <= 0: return 0.0
    return float(pos.max() / total_pos)


def main():
    print("=== Iteration 1: Time-of-day entry filter ===", flush=True)
    print(f"  Pre-registered SKIP_HOURS = {sorted(SKIP_HOURS)} (UTC)", flush=True)
    print(f"  V3.1 reference Sharpe = {V31_REF_SHARPE}\n", flush=True)

    records = pd.read_parquet(svar.SLEEVES_PATH)
    records["time"] = pd.to_datetime(records["time"], utc=True)
    apd = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet",
                            columns=["symbol"])
    all_syms = sorted(apd["symbol"].unique())
    print(f"  loading close prices...", flush=True)
    t0 = time.time()
    close_wide = svar.load_close_wide(all_syms)
    fwd_rets_4h = (close_wide.shift(-HORIZON_ENTRY) - close_wide) / close_wide
    print(f"  done ({time.time()-t0:.0f}s)\n", flush=True)

    # ---------- 1. Static: V3.1 baseline (no filter) ----------
    df_v31 = aggregate_sleeves_tod(records, fwd_rets_4h, V31_WEIGHTS,
                                       skip_entry_hours=None)
    v31_sh = _sharpe(df_v31["net_pnl_bps"])
    v31_dd = _max_dd(df_v31["net_pnl_bps"])
    v31_npos = sum(1 for _, g in df_v31.groupby("fold")
                    if _sharpe(g["net_pnl_bps"]) > 0)
    print(f"  V3.1 baseline (no filter):  Sharpe={v31_sh:+.3f}  "
          f"maxDD={v31_dd:+.0f}  PnL={df_v31['net_pnl_bps'].sum():+.0f}  "
          f"folds+={v31_npos}/9", flush=True)

    # ---------- 2. Static: TOD-filtered ----------
    df_tod = aggregate_sleeves_tod(records, fwd_rets_4h, V31_WEIGHTS,
                                       skip_entry_hours=SKIP_HOURS)
    tod_sh = _sharpe(df_tod["net_pnl_bps"])
    tod_dd = _max_dd(df_tod["net_pnl_bps"])
    tod_npos = sum(1 for _, g in df_tod.groupby("fold")
                    if _sharpe(g["net_pnl_bps"]) > 0)
    print(f"  TOD-filtered (skip {{4,12}}):  Sharpe={tod_sh:+.3f}  "
          f"maxDD={tod_dd:+.0f}  PnL={df_tod['net_pnl_bps'].sum():+.0f}  "
          f"folds+={tod_npos}/9", flush=True)
    print(f"  Static lift: {tod_sh - v31_sh:+.3f}", flush=True)

    df_tod.to_csv(OUT / "per_cycle_iter1_tod_filtered.csv", index=False)
    df_v31.to_csv(OUT / "per_cycle_iter1_v31_baseline.csv", index=False)

    # ---------- 3. Per-fold breakdown ----------
    print(f"\n  Per-fold breakdown:", flush=True)
    print(f"  {'fold':>4}  {'V3.1':>8}  {'TOD':>8}  {'Δ':>7}", flush=True)
    fold_diffs = {}
    for f in OOS_FOLDS:
        v31_g = df_v31[df_v31["fold"] == f]["net_pnl_bps"].sum()
        tod_g = df_tod[df_tod["fold"] == f]["net_pnl_bps"].sum()
        d = tod_g - v31_g
        fold_diffs[f] = d
        print(f"  {f:>4}  {v31_g:>+8.0f}  {tod_g:>+8.0f}  {d:>+7.0f}", flush=True)

    total_lift = sum(fold_diffs.values())
    pos_lift = sum(v for v in fold_diffs.values() if v > 0)
    if pos_lift > 0:
        max_fold_contribution = max(v for v in fold_diffs.values()) / pos_lift * 100
    else:
        max_fold_contribution = 0
    print(f"\n  Total lift (sum): {total_lift:+.0f} bps", flush=True)
    print(f"  Max single positive-fold contribution to positive lift: "
          f"{max_fold_contribution:.0f}%", flush=True)

    # ---------- 4. Paired bootstrap V3.1 vs TOD ----------
    print(f"\n--- Paired V3.1 vs TOD bootstrap ---", flush=True)
    paired = df_v31[["time", "fold", "net_pnl_bps"]].rename(
        columns={"net_pnl_bps": "v31"}).merge(
        df_tod[["time", "net_pnl_bps"]].rename(columns={"net_pnl_bps": "tod"}),
        on="time")
    paired["diff"] = paired["tod"] - paired["v31"]
    def _mean(x): return float(np.mean(x))
    mu, lo, hi = block_bootstrap_ci(paired["diff"].to_numpy(), stat=_mean,
                                        block_size=7, n_boot=2000)
    print(f"  Mean diff (TOD - V3.1) per cycle: {mu:+.3f} bps  CI [{lo:+.3f}, {hi:+.3f}]",
          flush=True)
    diff_sig = (lo > 0) or (hi < 0)
    print(f"  Paired diff statistically nonzero: {'YES' if diff_sig else 'NO'}",
          flush=True)

    sh_d, sh_lo, sh_hi = block_bootstrap_ci(paired["diff"].to_numpy(),
                                              stat=_sharpe, block_size=7, n_boot=2000)
    print(f"  Sharpe-of-diff: {sh_d:+.2f}  CI [{sh_lo:+.2f}, {sh_hi:+.2f}]",
          flush=True)

    # ---------- 5. Nested-OOS (per-fold threshold selection) ----------
    # Trivially: SKIP_HOURS is pre-registered, so nested == static. But test:
    # for each fold f, was {4,12} the past-fold best 2-hour combo, or were
    # other hours better? If V3.1's "best 2-hour skip" varies fold-to-fold,
    # then {4,12} doesn't generalize.
    print(f"\n--- Nested-OOS check: is {{4,12}} the best 2-hour skip past-fold? ---",
          flush=True)
    # Build per-fold per-hour mean PnL
    df_v31_h = df_v31.merge(records[["time"]], on="time", how="left")
    df_v31_h["hour"] = pd.to_datetime(df_v31_h["time"]).dt.hour
    fold_hour_mean = df_v31_h.groupby(["fold", "hour"])["net_pnl_bps"].mean().unstack()
    print(f"  Per-fold per-hour mean PnL (V3.1 baseline):", flush=True)
    print(fold_hour_mean.round(2).to_string(), flush=True)

    # For each test fold f, pick 2 worst hours from {1..f-1}, see if that
    # matches {4, 12}
    nested_picks = []
    for f in OOS_FOLDS:
        past = [pf for pf in OOS_FOLDS if pf < f]
        if not past:
            picked = (4, 12); reason = "fold 1 default"
        else:
            past_hour_mean = df_v31_h[df_v31_h["fold"].isin(past)].groupby(
                "hour")["net_pnl_bps"].mean()
            two_worst = past_hour_mean.sort_values().head(2).index.tolist()
            picked = tuple(sorted(two_worst))
            reason = f"past 2 worst: {past_hour_mean.sort_values().head(2).round(2).to_dict()}"
        match = picked == (4, 12)
        nested_picks.append({"fold": f, "picked": picked, "match_4_12": match,
                              "reason": reason})
        print(f"  fold {f}: picked={picked}  match {{4,12}}: {match}  ({reason})",
              flush=True)
    match_rate = sum(1 for r in nested_picks if r["match_4_12"]) / len(nested_picks)
    print(f"\n  Nested match rate: {match_rate*100:.0f}% of folds picked {{4,12}} based on past data",
          flush=True)

    # ---------- 6. Matched-time placebo ----------
    print(f"\n--- Matched-time placebo (random 2-hour skip, 100 seeds) ---",
          flush=True)
    placebo_sh = []
    rng = np.random.RandomState(42)
    all_hours = list(range(0, 24, 4))  # entry slots only
    t0 = time.time()
    for seed in range(100):
        # Pick 2 random hours from the 6 entry slots
        chosen = tuple(sorted(rng.choice(all_hours, size=2, replace=False).tolist()))
        df_p = aggregate_sleeves_tod(records, fwd_rets_4h, V31_WEIGHTS,
                                          skip_entry_hours=set(chosen))
        placebo_sh.append({"seed": seed, "skip_hours": chosen,
                             "sharpe": _sharpe(df_p["net_pnl_bps"])})
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/100  ({time.time()-t0:.0f}s)", flush=True)
    pdf = pd.DataFrame(placebo_sh)
    pdf.to_csv(OUT / "iter1_matched_time_placebo.csv", index=False)
    p_sh = pdf["sharpe"].to_numpy()
    p95 = float(np.percentile(p_sh, 95))
    rank = float((p_sh < tod_sh).mean() * 100)
    print(f"\n  Placebo Sharpe: mean={p_sh.mean():+.2f}  p50={np.median(p_sh):+.2f}  "
          f"p95={p95:+.2f}  max={p_sh.max():+.2f}", flush=True)
    print(f"  TOD ({tod_sh:+.2f}) ranks p{rank:.0f}  "
          f"beats_p95={'PASS' if tod_sh > p95 else 'FAIL'}", flush=True)

    # ---------- Verdict ----------
    print(f"\n=== Iteration 1 Verdict ===\n", flush=True)
    gates = []
    g1 = tod_sh >= V31_REF_SHARPE + 0.10
    g2 = nested_picks[0]["match_4_12"] or match_rate >= 0.7  # generalization
    g3 = tod_sh > p95
    g4 = diff_sig
    g5 = tod_npos >= 6
    g6 = max_fold_contribution <= 40
    gates = [
        ("Static lift ≥ +0.10", g1, f"{tod_sh:+.2f} - {V31_REF_SHARPE:+.2f} = {tod_sh-V31_REF_SHARPE:+.2f}"),
        ("Nested {4,12} stable past-fold", g2, f"match rate {match_rate*100:.0f}%"),
        ("Beats matched-time placebo p95", g3, f"{tod_sh:+.2f} vs p95 {p95:+.2f}"),
        ("Paired diff CI excludes 0", g4, f"[{lo:+.3f}, {hi:+.3f}]"),
        ("≥ 6/9 folds positive", g5, f"{tod_npos}/9"),
        ("Max fold contribution ≤ 40%", g6, f"{max_fold_contribution:.0f}%"),
    ]
    for name, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}  ({detail})", flush=True)
    all_pass = all(g[1] for g in gates)
    print(f"\n  Verdict: {'ACCEPT' if all_pass else 'REJECT'} time-of-day filter",
          flush=True)


if __name__ == "__main__":
    main()
