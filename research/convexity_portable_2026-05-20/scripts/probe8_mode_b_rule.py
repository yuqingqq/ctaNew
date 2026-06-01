"""Probe #8 — Direct rule-based test of Mode B (V3.1's outlier-state mechanism).

Hypothesis from Probe #7 VVV signature:
  When a name is in an EXTREME OUTLIER STATE on (idio vol + BTC decoupling +
  deep negative funding), and has had recent positive momentum, it's in a
  short-squeeze setup → expected positive forward return.

Pre-committed rules (no tuning — these come from VVV's observed mean signature):

  LONG RULE (squeeze-fuel pattern):
    idio_vol_1d_vs_bk_xsz   > +2.0σ  (extreme idio vol vs basket)
    corr_to_btc_1d_xsz      < -1.5σ  (decoupled from BTC)
    funding_rate_xsz        < -1.5σ  (shorts paying longs heavily)
    return_1d_xsz           > 0      (already rallying)

  SHORT RULE (over-extension / long-crowded pattern):
    idio_vol_1d_vs_bk_xsz   > +2.0σ  (extreme idio vol)
    corr_to_btc_1d_xsz      < -1.5σ  (decoupled — own dynamic)
    funding_rate_xsz        > +1.5σ  (longs paying shorts — over-positioned long)
    return_1d_xsz           > +0.5σ  (recent rally extended)

Test:
  - Compute xs-z per cycle across symbols at that time (PIT — features are
    already strictly PIT in the panel).
  - Apply rules, count firings.
  - For each firing: realized alpha_vs_btc_realized over next 4h (the
    BTC-residual target, costed at 9 bps RT).
  - Time-OOS folds (9 expanding-window).
  - Matched placebo: rule fires K bars per cycle; for each cycle, draw K
    random symbols and compute matched random returns. 100 placebo seeds.

Gates (pre-committed, BINARY):
  PASS-LONG:  net mean ≥ +20 bps AND Sharpe net ≥ +0.5 AND ≥6/9 folds positive
              AND real Sharpe > matched-placebo p95
  PASS-SHORT: analogous (negative mean = good, sign flipped)
  PASS-COMBINED: both legs PASS individually OR combined long-short Sharpe ≥ +1.0

Test on 51-panel first (validated mechanism source), then 110-panel (portability).
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/convexity_portable_2026-05-20/results"; OUT.mkdir(parents=True, exist_ok=True)
PANEL_51 = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
PANEL_110 = REPO / "outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet"

RULE_FEATS = ["idio_vol_to_btc_1d", "corr_to_btc_1d", "funding_rate", "return_1d"]
TARGET_COL_51 = "alpha_vs_btc_realized"      # 4h forward BTC residual
TARGET_COL_110 = "alpha_beta"                # same thing, different naming
COST_RT_BPS = 9.0                            # production convention
N_FOLDS = 9
N_PLACEBO = 100
SEED = 20260520


def compute_xsz(p, cols):
    """Cross-sectional z per open_time."""
    pz = p.copy()
    for c in cols:
        pz[c + "_xsz"] = pz.groupby("open_time")[c].transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1.0))
    return pz


def apply_rule_long(pz):
    return ((pz["idio_vol_to_btc_1d_xsz"] > 2.0) &
            (pz["corr_to_btc_1d_xsz"] < -1.5) &
            (pz["funding_rate_xsz"] < -1.5) &
            (pz["return_1d_xsz"] > 0)).fillna(False)


def apply_rule_short(pz):
    return ((pz["idio_vol_to_btc_1d_xsz"] > 2.0) &
            (pz["corr_to_btc_1d_xsz"] < -1.5) &
            (pz["funding_rate_xsz"] > 1.5) &
            (pz["return_1d_xsz"] > 0.5)).fillna(False)


def sharpe_4h(r):
    r = np.asarray(r, dtype=float); r = r[~np.isnan(r)]
    if len(r) < 2 or r.std() == 0: return 0.0
    return float(r.mean() / r.std() * np.sqrt(288 * 365 / 48))


def evaluate_side(fires, returns_bps_signed, name, n_folds=N_FOLDS):
    """Evaluate one side (long or short) — signed returns already aligned (
    long=positive when symbol up; short=positive when symbol down)."""
    if fires.sum() == 0:
        return {"n_fires": 0, "verdict": "NO FIRES"}
    fired = returns_bps_signed[fires].dropna()
    n = len(fired)
    gross_mean = float(fired.mean())
    net_mean = gross_mean - COST_RT_BPS
    sh_gross = sharpe_4h(fired)
    sh_net = sharpe_4h(fired - COST_RT_BPS)
    # per-fold
    fires_df = pd.DataFrame({
        "fired": fires.values,
        "ret": returns_bps_signed.values,
        "t": fires.index if hasattr(fires, 'index') else range(len(fires)),
    })
    return {
        "n_fires": int(n),
        "fire_rate_pct": round(float(fires.mean()) * 100, 4),
        "gross_mean_bps": round(gross_mean, 1),
        "net_mean_bps": round(net_mean, 1),
        "median_bps": round(float(fired.median()), 1),
        "std_bps": round(float(fired.std()), 1),
        "sharpe_gross_4h_ann": round(sh_gross, 3),
        "sharpe_net_4h_ann": round(sh_net, 3),
        "long_share_outcome": round(float((fired > 0).mean()), 3),
    }


def per_fold_evaluation(pz, fires, returns_signed, n_folds=N_FOLDS):
    """Time-OOS per-fold evaluation."""
    times = sorted(pz["open_time"].unique())
    n_times = len(times); fold_size = n_times // n_folds
    pf = []
    for f in range(n_folds):
        i0 = f * fold_size
        i1 = min((f + 1) * fold_size, n_times - 1) if f < n_folds - 1 else n_times
        ts = pd.Timestamp(times[i0]); te = pd.Timestamp(times[i1 - 1])
        mask = (pz["open_time"] >= ts) & (pz["open_time"] <= te)
        f_fires = fires & mask
        n = int(f_fires.sum())
        if n < 3:
            pf.append({"fold": f, "n": n, "mean": None, "sharpe_net": None})
            continue
        r = returns_signed[f_fires].dropna()
        mean = float(r.mean()); sn = sharpe_4h(r - COST_RT_BPS)
        pf.append({"fold": f, "n": n,
                   "mean_bps": round(mean, 1),
                   "net_mean_bps": round(mean - COST_RT_BPS, 1),
                   "sharpe_net": round(sn, 3),
                   "from": str(ts.date()), "to": str(te.date())})
    return pf


def matched_placebo(pz, fires, returns_signed, seeds=N_PLACEBO):
    """For each cycle where rule fired K_t bars, draw K_t random symbols
    from same cycle. 100 seeds. Compute placebo Sharpe distribution."""
    rng = np.random.RandomState(SEED)
    by_cycle = pz.groupby("open_time")
    fires_by_t = pz.assign(fire=fires.values).groupby("open_time")["fire"].sum()
    cycle_idx = pz.groupby("open_time").indices
    placebo_sharpes = []
    for s in range(seeds):
        local_rng = np.random.RandomState(SEED + s)
        sampled_returns = []
        for t, k in fires_by_t.items():
            if k == 0: continue
            idx_pool = cycle_idx[t]
            if len(idx_pool) <= k:
                pick = idx_pool
            else:
                pick = local_rng.choice(idx_pool, size=k, replace=False)
            sampled_returns.extend(returns_signed.iloc[pick].dropna().values)
        if len(sampled_returns) < 5:
            placebo_sharpes.append(0.0); continue
        sampled = np.array(sampled_returns) - COST_RT_BPS
        placebo_sharpes.append(sharpe_4h(sampled))
    return {
        "p05": round(float(np.percentile(placebo_sharpes, 5)), 3),
        "p50": round(float(np.percentile(placebo_sharpes, 50)), 3),
        "p95": round(float(np.percentile(placebo_sharpes, 95)), 3),
        "mean": round(float(np.mean(placebo_sharpes)), 3),
        "max": round(float(np.max(placebo_sharpes)), 3),
        "min": round(float(np.min(placebo_sharpes)), 3),
    }


def run_test(panel_path, target_col, panel_name):
    print(f"\n========== {panel_name} ==========", flush=True)
    cols = ["symbol", "open_time", target_col] + RULE_FEATS
    p = pd.read_parquet(panel_path, columns=cols)
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    # 4h-aligned subsample
    p = p[(p["open_time"].dt.minute == 0) & (p["open_time"].dt.hour % 4 == 0)]
    p = p.dropna(subset=[target_col]).reset_index(drop=True)
    print(f"  panel: {len(p):,} rows × {p['symbol'].nunique()} syms (4h-aligned)", flush=True)

    pz = compute_xsz(p, RULE_FEATS)
    fires_long = apply_rule_long(pz)
    fires_short = apply_rule_short(pz)
    print(f"  long-rule fires:  {fires_long.sum():,} ({fires_long.mean()*100:.4f}%)", flush=True)
    print(f"  short-rule fires: {fires_short.sum():,} ({fires_short.mean()*100:.4f}%)", flush=True)

    # signed returns in bps
    # long fires: r_long = +target_col (in bps)
    # short fires: r_short = -target_col (in bps)
    pz["target_bps"] = pz[target_col] * 10000.0  # to bps

    long_eval = evaluate_side(fires_long, pz["target_bps"], "LONG")
    short_eval = evaluate_side(fires_short, -pz["target_bps"], "SHORT")

    long_fold = per_fold_evaluation(pz, fires_long, pz["target_bps"])
    short_fold = per_fold_evaluation(pz, fires_short, -pz["target_bps"])

    long_placebo = matched_placebo(pz, fires_long, pz["target_bps"])
    short_placebo = matched_placebo(pz, fires_short, -pz["target_bps"])

    print(f"\n  LONG RULE:", flush=True)
    print(f"    n_fires={long_eval['n_fires']}, gross_mean={long_eval.get('gross_mean_bps','—'):+} bps, "
          f"net_mean={long_eval.get('net_mean_bps','—'):+} bps, Sharpe_net={long_eval.get('sharpe_net_4h_ann','—')}",
          flush=True)
    print(f"    placebo: p50={long_placebo['p50']}, p95={long_placebo['p95']}, "
          f"real_beats_p95: {long_eval.get('sharpe_net_4h_ann',-99) > long_placebo['p95']}",
          flush=True)
    pos = sum(1 for r in long_fold if r.get('sharpe_net') and r['sharpe_net'] > 0)
    print(f"    folds positive: {pos}/{sum(1 for r in long_fold if r.get('sharpe_net') is not None)}", flush=True)

    print(f"\n  SHORT RULE:", flush=True)
    print(f"    n_fires={short_eval['n_fires']}, gross_mean={short_eval.get('gross_mean_bps','—'):+} bps, "
          f"net_mean={short_eval.get('net_mean_bps','—'):+} bps, Sharpe_net={short_eval.get('sharpe_net_4h_ann','—')}",
          flush=True)
    print(f"    placebo: p50={short_placebo['p50']}, p95={short_placebo['p95']}, "
          f"real_beats_p95: {short_eval.get('sharpe_net_4h_ann',-99) > short_placebo['p95']}",
          flush=True)
    pos = sum(1 for r in short_fold if r.get('sharpe_net') and r['sharpe_net'] > 0)
    print(f"    folds positive: {pos}/{sum(1 for r in short_fold if r.get('sharpe_net') is not None)}", flush=True)

    return {
        "panel": panel_name,
        "panel_rows": int(len(p)),
        "n_syms": int(p["symbol"].nunique()),
        "long": {"eval": long_eval, "per_fold": long_fold, "placebo": long_placebo},
        "short": {"eval": short_eval, "per_fold": short_fold, "placebo": short_placebo},
    }


def main():
    t0 = time.time()
    print("=== Probe #8 Mode B rule test ===", flush=True)
    print(f"  LONG rule:  idio_vol_vs_bk>+2σ AND corr_btc<-1.5σ AND funding<-1.5σ AND ret_1d>0", flush=True)
    print(f"  SHORT rule: idio_vol_vs_bk>+2σ AND corr_btc<-1.5σ AND funding>+1.5σ AND ret_1d>+0.5σ", flush=True)
    print(f"  Cost: {COST_RT_BPS} bps RT", flush=True)

    r51 = run_test(PANEL_51, TARGET_COL_51, "51-PANEL")
    r110 = run_test(PANEL_110, TARGET_COL_110, "110-PANEL")

    # report fold tables
    for r in (r51, r110):
        print(f"\n=== {r['panel']} per-fold (LONG) ===", flush=True)
        for f in r["long"]["per_fold"]:
            print(f"  fold {f['fold']}: n={f['n']}, "
                  f"mean_bps={f.get('mean_bps','—')}, "
                  f"net_mean={f.get('net_mean_bps','—')}, "
                  f"sharpe_net={f.get('sharpe_net','—')}, "
                  f"{f.get('from','')}→{f.get('to','')}", flush=True)
        print(f"\n=== {r['panel']} per-fold (SHORT) ===", flush=True)
        for f in r["short"]["per_fold"]:
            print(f"  fold {f['fold']}: n={f['n']}, "
                  f"mean_bps={f.get('mean_bps','—')}, "
                  f"net_mean={f.get('net_mean_bps','—')}, "
                  f"sharpe_net={f.get('sharpe_net','—')}, "
                  f"{f.get('from','')}→{f.get('to','')}", flush=True)

    out = {
        "rules": {
            "long": "idio_vol_vs_bk_xsz>+2.0 AND corr_btc_xsz<-1.5 AND funding_xsz<-1.5 AND return_1d_xsz>0",
            "short": "idio_vol_vs_bk_xsz>+2.0 AND corr_btc_xsz<-1.5 AND funding_xsz>+1.5 AND return_1d_xsz>+0.5",
        },
        "cost_rt_bps": COST_RT_BPS,
        "panel_51": r51,
        "panel_110": r110,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "probe8_mode_b_rule.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nPROBE8_DONE [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
