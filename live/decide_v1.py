"""v1 DECIDE driver — fires the decision at the bar boundary (T+~1min), not 4h35m later at settle.

Reuses predict_at_close.build_bar to construct the current (unlabeled) 4h bar with PIT features (byte-
identical to the panel, proven by predict_at_close --verify), computes the resid_rev long-ranker features
for that bar from the labeled trailing alpha (shift(1) → uses only settled prior bars), scores the frozen
base(short) + resid_rev(long) models, filters to the low-vol universe, and writes decide-preds
(base_decide.parquet / long_decide.parquet — current bar only). `bot --decide` reads those to select the
bar's legs + turnover into decision.json, which the HL probe prices at the real boundary.

Usage:
  python3 live/decide_v1.py            # build+score current bar → decide-preds for `bot --decide`
  python3 live/decide_v1.py --verify   # build a PAST labeled bar, assert its base/long preds == the panel's
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.predict_at_close as pac
import live.train_twobook_models as tt

x6 = pac.x6; V0 = tt.V0; RR = tt.RR
MODELS = REPO/"live/models"
STATE = REPO/"live/state"/os.environ.get("CONVEXITY_BOOK", "convexity_v1")
UNIV = REPO/"live/models/convexity_v1_universe.json"


def _close_ref_at(syms, boundary) -> dict:
    """Binance close at the bar boundary per sym (the price the signal saw) — read from the xs_feats caches
    (5m-indexed, have a `close` col). Used by the ledger for the latency-drift leg of the cost decomposition."""
    from live.incremental_xs_feats import CACHE
    out = {}
    for s in syms:
        p = CACHE / f"xs_feats_{s}.parquet"
        if not p.exists():
            continue
        try:
            c = pd.read_parquet(p, columns=["close"])
            c.index = pd.to_datetime(c.index, utc=True)
            v = c["close"].asof(boundary)
            if np.isfinite(v):
                out[s] = float(v)
        except Exception:
            continue
    return out


def _ret3d_at(syms, boundary) -> dict:
    """3-day backward return per sym from the xs_feats close (5m close.asof) — the long-winner gate input,
    computed LIVE for the CURRENT decide bar. The labeled panel lags a cycle (the decide bar isn't in it yet,
    its forward label needs the next boundary), so the bot's panel-merged ret_3d is NaN here and the gate
    silently no-ops; this supplies it from the same live close source as _close_ref_at."""
    from live.incremental_xs_feats import CACHE
    out = {}
    for s in syms:
        p = CACHE / f"xs_feats_{s}.parquet"
        if not p.exists():
            continue
        try:
            c = pd.read_parquet(p, columns=["close"]); c.index = pd.to_datetime(c.index, utc=True)
            now = c["close"].asof(boundary); ago = c["close"].asof(boundary - pd.Timedelta(days=3))
            if np.isfinite(now) and np.isfinite(ago) and ago > 0:
                out[s] = float(now / ago - 1.0)
        except Exception:
            continue
    return out


def _score(bar_feats: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Apply frozen per-sym models to the bar — identical path to predict_twobook_incremental._predict."""
    rec = []
    for sym, g in bar_feats.groupby("symbol"):
        if sym not in models:
            continue
        m, s, h, feats = models[sym]
        try:
            pred = m.predict(x6.apply_preproc(g, feats, s, h))
        except Exception:
            continue
        rec.append(pd.DataFrame({"symbol": sym, "open_time": g["open_time"].values,
                                 "alpha_A": g["alpha_vs_btc_realized"].values, "return_pct": np.nan,
                                 "exit_time": g["exit_time"].values, "pred": pred, "fold": -1}))
    return pd.concat(rec, ignore_index=True) if rec else pd.DataFrame()


def _with_residrev(bar: pd.DataFrame, boundary) -> pd.DataFrame:
    """Append the current bar to the labeled panel, compute resid_rev for it from settled trailing alpha."""
    pan = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "return_pct",
                                             "alpha_vs_btc_realized"] + V0)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan["exit_time"] = pd.to_datetime(pan["exit_time"], utc=True)
    pan = pan[(pan.open_time.dt.hour % 4 == 0) & (pan.open_time.dt.minute == 0)]
    cur = bar.copy()
    if "alpha_vs_btc_realized" not in cur: cur["alpha_vs_btc_realized"] = np.nan      # unlabeled current bar
    cur["return_pct"] = np.nan
    keep = ["symbol", "open_time", "exit_time", "return_pct", "alpha_vs_btc_realized"] + V0
    comb = pd.concat([pan[keep], cur[keep]], ignore_index=True)
    comb = comb.drop_duplicates(["symbol", "open_time"], keep="last").sort_values(["symbol", "open_time"])
    _a = comb.groupby("symbol")["alpha_vs_btc_realized"]                              # same def as the trainer
    comb["resid_rev_2"] = -_a.transform(lambda s: s.shift(1).rolling(2).sum())
    comb["resid_rev_3"] = -_a.transform(lambda s: s.shift(1).rolling(3).sum())
    for c in RR: comb[c] = comb[c].fillna(0.0)
    return comb[comb["open_time"] == boundary].copy()


def _freshness(univ_syms) -> dict:
    """Inputs for the decision gate: funding age + residual klines gaps among the decision universe (the latter
    from backfill_klines_gaps' data_health.json). Lets us REFUSE a decision on degraded data instead of trading
    on ffill-patched/stale features — the live-test analogue of the gap bug we just fixed historically."""
    import glob
    fa_h = 999.0
    try:
        fs = glob.glob(str(REPO / "data/ml/cache/funding_*.parquet"))
        mx = max((pd.to_datetime(pd.read_parquet(f, columns=["calc_time"])["calc_time"], utc=True).max()
                  for f in fs[:40]), default=None)
        if mx is not None:
            fa_h = (pd.Timestamp.utcnow() - mx).total_seconds() / 3600
    except Exception:
        pass
    gappy = []
    hp = REPO / "live/state/data_health.json"
    if hp.exists():
        try:
            h = json.load(open(hp))
            gappy = sorted((set(h.get("residual_gappy", [])) | set(h.get("fapi_errors", []))) & set(univ_syms))
        except Exception:
            pass
    return {"funding_age_h": round(fa_h, 1), "gappy_universe": gappy}


def run(boundary=None) -> dict:
    boundary = boundary if boundary is not None else pac._latest_closed_boundary()
    bar = pac.build_bar(boundary, drop_unlabeled=False)
    if bar is None:
        print(f"[decide_v1] no current bar @ {boundary}"); return {}
    ot = bar["open_time"].iloc[0]
    cur = _with_residrev(bar, ot)                                                     # adds resid_rev_2/3
    short = pickle.load(open(MODELS/"convexity_v1_short_model.pkl", "rb"))["models"]  # base V0 -> shorts
    longm = pickle.load(open(MODELS/"convexity_v1_long_model.pkl", "rb"))["models"]   # V0+resid_rev -> longs
    base = _score(cur, short)
    longp = _score(cur, longm)
    excl = set(json.load(open(UNIV))["exclude_high_vol"])                             # low-vol universe only
    base = base[~base["symbol"].isin(excl)]; longp = longp[~longp["symbol"].isin(excl)]
    ddir = STATE/"decide"; ddir.mkdir(parents=True, exist_ok=True)
    # FRESHNESS GATE — refuse to decide on a degraded feed (systematic gaps FAPI couldn't fill, or stale funding)
    fr = _freshness(set(base["symbol"]) | set(longp["symbol"]))
    fund_max = float(os.environ.get("FRESHNESS_FUNDING_MAX_H", "5"))
    gap_max = int(os.environ.get("FRESHNESS_GAP_MAX", "10"))
    # COHORT GUARD: bars_since_high_xs_rank is ranked over the symbols predict_at_close builds for THIS bar
    # (predict_at_close.py:78), and the model trained on 174. The 80 non-traded high-vol names are peers in that
    # rank; if their klines go stale they don't build a current bar and silently drop out, mis-scaling the rank
    # for the traded names (the 174→94 collapse on 06-04). The traded-universe gappy check above is blind to it
    # (those peers aren't in the decision universe), so guard directly on the built-bar cohort size.
    # ABORT only on a SEVERE collapse (most of the cohort gone — the original 94-collapse bug). A moderate dip
    # (e.g. the transient boundary-bar race leaving the slowest peers behind) WARNS but still trades — freezing
    # the book on every boundary is worse than a slightly-drifted xs-rank. The wait_boundary_bar step normally
    # keeps the cohort ≈174; this guard is the floor for genuine data failure.
    cohort_n = int(bar["symbol"].nunique())
    cohort_min = int(os.environ.get("FRESHNESS_COHORT_MIN", "60"))
    cohort_bad = 0 < cohort_n < cohort_min
    degraded = fr["funding_age_h"] > fund_max or len(fr["gappy_universe"]) > gap_max or cohort_bad
    json.dump({**fr, "cohort_n": cohort_n, "degraded": degraded, "open_time": str(ot)}, open(ddir/"freshness.json", "w"))
    if degraded:
        print(f"[decide_v1] DEGRADED FEED @ {ot}: funding {fr['funding_age_h']}h "
              f"(max {fund_max}), {len(fr['gappy_universe'])} gappy univ syms "
              f"{fr['gappy_universe'][:8]} (max {gap_max}), xs-rank cohort {cohort_n} (min {cohort_min}) "
              f"→ ABORT (no decision this cycle)")
        return {}
    if 0 < cohort_n < 165:
        print(f"[decide_v1] WARN: xs-rank cohort {cohort_n} < 174 (some peer klines stale) — rank may drift, trading anyway")
    if fr["gappy_universe"]:
        print(f"[decide_v1] feed OK but {len(fr['gappy_universe'])} univ syms on ffill-patched bars: "
              f"{fr['gappy_universe'][:8]}")
    r3 = _ret3d_at(sorted(set(base["symbol"]) | set(longp["symbol"])), ot)   # LIVE long-winner gate input
    base["ret_3d"] = base["symbol"].map(r3); longp["ret_3d"] = longp["symbol"].map(r3)
    if not r3:    # empty -> ret_3d all-NaN -> the long-winner gate keeps everything (silent no-op); surface it
        print(f"[decide_v1] WARN: live ret_3d EMPTY @ {ot} — long-winner gate input all-NaN, gate NO-OPS this cycle")
    base.to_parquet(ddir/"base_decide.parquet", index=False)
    longp.to_parquet(ddir/"long_decide.parquet", index=False)
    # Binance bar-close per sym = the price the signal saw at B; the ledger uses it for the latency-drift
    # leg of the execution-cost decomposition (HL exec mid at B+~90s vs this).
    cref = _close_ref_at(base["symbol"].tolist(), ot)
    json.dump(cref, open(ddir/"close_ref.json", "w"))
    print(f"[decide_v1] bar {ot}: scored base {len(base)} / long {len(longp)} syms (low-vol) → {ddir}")
    return {"open_time": str(ot), "n_base": len(base), "n_long": len(longp)}


def verify():
    """Build a PAST labeled bar via the decide path; assert its base/long preds match the settle preds."""
    pan = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time"]); pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    B = sorted(pan["open_time"].unique())[-2]
    print(f"[verify] decide-path preds vs settle preds @ {pd.Timestamp(B)}")
    bar = pac.build_bar(pd.Timestamp(B), drop_unlabeled=True)
    cur = _with_residrev(bar, pd.Timestamp(B))
    short = pickle.load(open(MODELS/"convexity_v1_short_model.pkl", "rb"))["models"]
    longm = pickle.load(open(MODELS/"convexity_v1_long_model.pkl", "rb"))["models"]
    dec_base = _score(cur, short).set_index("symbol")["pred"]
    dec_long = _score(cur, longm).set_index("symbol")["pred"]
    for name, dec, path in [("base", dec_base, STATE/"base.parquet"), ("long", dec_long, STATE/"long.parquet")]:
        ref = pd.read_parquet(path); ref["open_time"] = pd.to_datetime(ref["open_time"], utc=True)
        ref = ref[ref["open_time"] == B].set_index("symbol")["pred"]
        common = dec.index.intersection(ref.index)
        diff = (dec.loc[common] - ref.loc[common]).abs().max()
        print(f"  {name}: {len(common)} syms, max |pred diff| {diff:.2e} {'MATCH ✓' if diff < 1e-6 else 'DIFF ✗'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        verify()
    else:
        r = run()
        # exit non-zero if the current bar isn't buildable yet → cycle_once aborts instead of letting
        # bot --decide read a STALE base_decide from a prior boundary.
        sys.exit(0 if r else 1)
