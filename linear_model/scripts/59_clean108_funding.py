"""Step 59: clean-108 re-measured WITH realized funding PnL.

User challenge (2026-05-15): the sim never modeled perpetual funding. funding_rate
is in the panel but used ONLY as a model feature; alpha_beta is price-return only.
On the PnL-driving meme names (PIPPIN, BROCCOLI714) per-interval |funding| is
same-order-or-larger than the +9.55 bps/cycle gross. So the +2.34 magnitude — and
possibly its sign on those names — is untrustworthy until funding is simulated.

This script:
  1. Loads Step 58 clean-108 predictions (no retrain)
  2. Builds per-symbol funding settlement cadence INFERRED from the data
     (median gap between funding_rate change timestamps), not assumed
  3. Adds realized funding PnL to the causal aggregator:
       funding_pnl[t] = -Σ_sym tw[sym] × funding_rate(sym,t) × (4h / interval_h[sym])
       (long pays positive funding → negative PnL; short receives)
     This continuously pro-rates the per-interval funding rate over the 4h block
     each held weight spans; summed across cycles = funding over the full 24h hold.
  4. Reports Sharpe with vs without funding, per-fold, LOFO, P1/P2 placebos.

Decisive test of: does clean-108 +2.34 survive realized funding, and which
direction (shorting crowded/pumping memes EARNS funding → may help).
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

PANEL = REPO / "outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
PREDS_DIR = REPO / "linear_model/results/step58_clean108"
OUT = REPO / "linear_model/results/step59_clean108_funding"
OUT.mkdir(parents=True, exist_ok=True)

OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
HOLD_BARS = 288
BAR_MIN = 5
BLOCK_HOURS = psl.HORIZON_ENTRY * BAR_MIN / 60.0   # 4h cycle block
N_PLACEBO = 100


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


FUND_GRID = {8.0: {0, 8, 16}, 4.0: {0, 4, 8, 12, 16, 20}}
FUND_CLIP = 0.025   # 250 bps hard safety guard (above Binance's ±2% cap)


def infer_funding(panel_syms, sampled_t):
    """Realized funding over the forward 4h held block per (cycle_t × symbol).

    Funding is charged ONCE per funding settlement on the standard UTC grid
    (8h: 00/08/16 UTC; 4h: 00/04/08/12/16/20). `funding_rate` in the panel is
    a continuously re-marked rate (changes many times per interval), so summing
    change-points over-counts; instead we sample the ffilled rate AT each grid
    settlement and charge it once. Per-symbol interval = 4h if ≥30% of rate-
    change timestamps fall on the 4h-only hours {4,12,20}, else 8h (default;
    the Binance norm). A 4h block then spans 0–1 settlements (8h) or exactly 1
    (4h), bounded by the ±2% cap — physically ≤ ~200 bps, never −682.

    funding_block[t] = Σ settlement funding_rate in (t, t+4h]   (forward, to
    match the causal convention where weights set at t earn fwd alpha[t])."""
    fp = pd.read_parquet(PANEL, columns=["symbol", "open_time", "funding_rate"])
    fp["open_time"] = pd.to_datetime(fp["open_time"], utc=True)
    fp = fp[fp["symbol"].isin(set(panel_syms))]
    idx = pd.DatetimeIndex(sorted(sampled_t))
    block = pd.Timedelta(hours=BLOCK_HOURS)
    cols = {}
    intervals = {}
    n_clipped = 0
    for sym, g in fp.groupby("symbol"):
        g = g.dropna(subset=["funding_rate"]).sort_values("open_time")
        if len(g) < 10:
            continue
        s = g.set_index("open_time")["funding_rate"]
        s = s[~s.index.duplicated(keep="last")]
        # interval detection from rate-change timestamps' hour-of-day
        chg = s[s.diff().fillna(1) != 0]
        if len(chg) >= 5:
            hrs = chg.index.round("1h").hour
            frac_4h_only = np.mean(np.isin(hrs, [4, 12, 20]))
            interval_h = 4.0 if frac_4h_only >= 0.30 else 8.0
        else:
            interval_h = 8.0
        intervals[sym] = interval_h
        # settlement grid timestamps spanning the data range
        t0 = s.index.min().floor("1D"); t1 = s.index.max().ceil("1D")
        all_h = pd.date_range(t0, t1, freq="1h", tz="UTC")
        settle = all_h[np.isin(all_h.hour, list(FUND_GRID[interval_h]))]
        # ffilled rate sampled AT each settlement (rate in effect that interval)
        f_at = s.reindex(s.index.union(settle)).sort_index().ffill().reindex(settle)
        f_at = f_at.fillna(0.0).clip(-FUND_CLIP, FUND_CLIP)
        n_clipped += int((f_at.abs() >= FUND_CLIP).sum())
        cum = pd.Series(f_at.values, index=settle).cumsum()
        cum = cum.reindex(cum.index.union(idx + block).union(idx)).sort_index().ffill()
        at_t = cum.reindex(idx).fillna(0.0).values
        at_te = cum.reindex(idx + block).fillna(0.0).values
        cols[sym] = at_te - at_t            # funding settled in (t, t+4h]
    fund_block = pd.DataFrame(cols, index=idx).fillna(0.0)
    if n_clipped:
        print(f"  WARN: {n_clipped} settlement rates clipped at ±{FUND_CLIP*1e4:.0f} bps",
              flush=True)
    return fund_block, intervals


def aggregate_causal_funding(records, alpha_wide, fund_block):
    """58b causal aggregator + realized funding over each 4h held block.

    gross[t]   = Σ tw[sym] × alpha[sym,t]                       (new weights earn fwd 4h)
    funding[t] = -Σ tw[sym] × fund_block[sym,t]                 (long pays + funding)
    cost[t]    = Σ|tw - prev_tw| × COST_PER_UNIT_ABS_DELTA
    net        = gross + funding - cost
    """
    sleeve_queue = deque(maxlen=psl.N_SLEEVES)
    prev_weights = {}
    bar_freq = pd.Timedelta(minutes=BAR_MIN)
    has_fund = set(fund_block.columns)
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]; fold = rec["fold"]
        if rec["traded"]:
            sleeve_queue.append({"entry_time": t, "longs": list(rec["long_basket"]),
                                  "shorts": list(rec["short_basket"])})
        max_age = HOLD_BARS * bar_freq
        sleeve_queue = deque(
            [s for s in sleeve_queue if (t - s["entry_time"]) < max_age],
            maxlen=psl.N_SLEEVES)
        tw = defaultdict(float)
        sw = 1.0 / psl.N_SLEEVES
        for sl in sleeve_queue:
            nL, nS = len(sl["longs"]), len(sl["shorts"])
            if nL == 0 or nS == 0: continue
            for s in sl["longs"]: tw[s] += sw * (1.0 / nL)
            for s in sl["shorts"]: tw[s] -= sw * (1.0 / nS)
        gross = 0.0
        if t in alpha_wide.index:
            a = alpha_wide.loc[t]
            for sym, w in tw.items():
                if sym in a.index and not pd.isna(a[sym]):
                    gross += w * a[sym] * 1e4
        funding = 0.0
        if t in fund_block.index:
            fr = fund_block.loc[t]
            for sym, w in tw.items():
                if sym in has_fund:
                    fv = fr[sym]
                    if not pd.isna(fv):
                        funding += -w * fv * 1e4   # long (w>0) pays + funding
        syms = set(tw.keys()) | set(prev_weights.keys())
        abs_d = sum(abs(tw.get(s, 0) - prev_weights.get(s, 0)) for s in syms)
        cost = abs_d * psl.COST_PER_UNIT_ABS_DELTA
        rows.append({"time": t, "fold": fold, "gross_pnl_bps": gross,
                      "funding_pnl_bps": funding, "cost_bps": cost,
                      "net_pnl_bps": gross + funding - cost,
                      "net_nofund_bps": gross - cost, "turnover": abs_d})
        prev_weights = dict(tw)
    return pd.DataFrame(rows)


def build_liquidity_universe(sampled_t, panel_syms, n_top=30):
    print(f"  Loading kline volumes for {len(panel_syms)} symbols...", flush=True)
    daily_dv = {}
    for sym in panel_syms:
        m5 = KLINES_DIR / sym / "5m"
        if not m5.exists(): continue
        files = sorted(m5.glob("*.parquet"))
        if not files: continue
        df = pd.concat([pd.read_parquet(f, columns=["open_time", "quote_volume"])
                          for f in files], ignore_index=True)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df["date"] = df["open_time"].dt.floor("1D")
        daily_dv[sym] = df.groupby("date")["quote_volume"].sum()
    dv_wide = pd.DataFrame(daily_dv).sort_index()
    universe = {}
    for t in sampled_t:
        win = dv_wide[(dv_wide.index >= t - pd.Timedelta(days=90)) & (dv_wide.index < t)]
        if len(win) < 10: continue
        universe[t] = set(win.mean(axis=0).sort_values(ascending=False).head(n_top).index)
    return universe


def main():
    print("=" * 100, flush=True)
    print("  STEP 59: clean-108 WITH realized funding PnL (causal aggregator)", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    listings = get_listings()

    apd_full = pd.read_parquet(PREDS_DIR / "predictions.parquet")
    apd_full["open_time"] = pd.to_datetime(apd_full["open_time"], utc=True)
    apd_full["alpha_A"] = apd_full["alpha_beta"]
    panel_syms = sorted(apd_full["symbol"].unique())
    print(f"\nPanel symbols: {len(panel_syms)} (BTC excluded)", flush=True)

    for s, t in apd_full.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t
    panel_syms_set = set(panel_syms)
    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms_set if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd_full[apd_full["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]

    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]
    apd_full["pred"] = apd_full["pred_z"]
    universe_V2 = psl.build_rolling_ic_universe(apd_full, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide = apd_full.pivot_table(index="open_time", columns="symbol",
                                        values="alpha_A", aggfunc="first").sort_index()

    print("\nInferring per-symbol funding cadence from data...", flush=True)
    fund_block, intervals = infer_funding(panel_syms, sampled_t)
    from collections import Counter
    print(f"  funding interval distribution (hours): "
          f"{dict(Counter(intervals.values()))}", flush=True)
    print(f"  fund_block shape {fund_block.shape}, "
          f"mean |block funding| = {fund_block.abs().mean().mean()*1e4:.2f} bps", flush=True)

    apd_v = apd_full.copy(); apd_v["pred"] = apd_v["pred_B"]
    records_real = psl.run_production_protocol_save_sleeves(apd_v, universe_V2)
    df = aggregate_causal_funding(records_real, alpha_wide, fund_block)
    df.to_csv(OUT / "per_cycle_real_funding.csv", index=False)

    sh_fund = _sharpe(df["net_pnl_bps"].to_numpy())
    sh_nofund = _sharpe(df["net_nofund_bps"].to_numpy())
    lo, hi = block_bootstrap_ci(df["net_pnl_bps"].to_numpy(), statistic=_sharpe,
                                  block_size=7, n_boot=1000)[1:]
    print(f"\n{'='*100}", flush=True)
    print("  REAL clean-108 B_IC_signed — funding impact", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  Sharpe NO funding (== 58b check): {sh_nofund:+.2f}", flush=True)
    print(f"  Sharpe WITH funding            : {sh_fund:+.2f}  [{lo:+.2f}, {hi:+.2f}]",
          flush=True)
    print(f"  mean gross   = {df['gross_pnl_bps'].mean():+.2f} bps/cyc", flush=True)
    print(f"  mean funding = {df['funding_pnl_bps'].mean():+.2f} bps/cyc "
          f"(>0 = strategy EARNS funding)", flush=True)
    print(f"  mean cost    = {df['cost_bps'].mean():+.2f} bps/cyc", flush=True)
    print(f"  mean net     = {df['net_pnl_bps'].mean():+.2f} bps/cyc", flush=True)
    print(f"  folds+ (with funding) = {folds_positive(df)}/9", flush=True)

    print(f"\n  Per-fold (gross / funding / cost / net):", flush=True)
    for fid, g in df.groupby("fold"):
        print(f"    fold {fid}: g={g['gross_pnl_bps'].mean():+6.2f}  "
              f"f={g['funding_pnl_bps'].mean():+6.2f}  "
              f"c={g['cost_bps'].mean():5.2f}  "
              f"net={g['net_pnl_bps'].mean():+6.2f}  Sh={_sharpe(g['net_pnl_bps']):+.2f}",
              flush=True)

    print(f"\n  LOFO (with funding, Sharpe={sh_fund:+.2f}):", flush=True)
    for excl in range(1, 10):
        rem = df[df["fold"] != excl]["net_pnl_bps"].to_numpy()
        d = _sharpe(rem) - sh_fund
        print(f"    excl {excl}: {_sharpe(rem):+.2f} (Δ {d:+.2f})"
              f"{'  ← drives' if d < -0.4 else ''}", flush=True)

    # Placebos WITH funding
    universe_liq = build_liquidity_universe(sampled_t, panel_syms, n_top=30)
    for name, univ in [("P1 (liq-univ random)", universe_liq),
                        ("P2 (V2-univ random)", universe_V2)]:
        ps = []
        for seed in range(N_PLACEBO):
            rp = psl.run_production_protocol_save_sleeves(apd_v, univ, placebo_seed=seed)
            dp = aggregate_causal_funding(rp, alpha_wide, fund_block)
            ps.append(_sharpe(dp["net_pnl_bps"].to_numpy()))
        ps = np.array(ps); p95 = float(np.percentile(ps, 95))
        rank = (ps < sh_fund).mean() * 100
        print(f"\n  {name} ×{N_PLACEBO} WITH funding: p95={p95:+.2f}  "
              f"real rank p{rank:.0f}  edge {sh_fund - p95:+.2f}  "
              f"{'PASS' if sh_fund > p95 else 'FAIL'}", flush=True)
        pd.DataFrame({name: ps}).to_csv(
            OUT / f"placebo_{name.split()[0]}_funding.csv", index=False)

    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
