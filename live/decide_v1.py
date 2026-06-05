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
