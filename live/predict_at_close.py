"""Predict-at-close DECIDE builder — surfaces the just-opened 4h bar so the decision fires at the bar
boundary (~1 min after, once klines flush) instead of 4h35m later when the forward label settles.

The settle path (incremental_panel) drops any bar without its realized forward return; that's correct for
SCORING but wrong for DECIDING — the model only needs the point-in-time features, which are ready as soon
as the bar's 5m klines are in. This builds the latest bar with drop_unlabeled=False (PIT features, NaN
label), so predict_twobook can score it and the bot can trade it immediately. The reference PnL is still
booked later from return_pct (A); the real HL fill is captured here at decide time.

Build is byte-identical to the panel's feature computation (reuses incremental_panel._build_sym_window) —
--verify proves a PAST bar built this way matches the labeled panel exactly (no train/serve skew).

Usage:
  python3 live/predict_at_close.py            # build current bar → predict_panel.parquet
  python3 live/predict_at_close.py --verify    # build a past bar, assert V0 matches the labeled panel
"""
from __future__ import annotations
import argparse, json, os, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
import live.incremental_panel as ip
import live.train_twobook_models as tt

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO/"live/state/convexity/predict_panel.parquet"
DECIDE_DIR = REPO/"live/state/convexity/decide"
MODELS = REPO/"live/models"
SPLIT = MODELS/"twobook_split.json"
V0 = ip.x6.BASE + ip.x6.COHORT_EXTRAS
x6 = ip.x6


def _latest_closed_boundary():
    """Latest 4h boundary B such that every panel sym has 5m klines at/just-past B (bar has opened)."""
    now = pd.Timestamp.utcnow()
    if now.tz is None: now = now.tz_localize("UTC")
    return now.floor("4h")


def _build_one(args):
    """Pool worker: build one sym's window and keep only the boundary row. Inherits ip._BTC_FULL via fork
    (set in build_bar before the pool is created), so no per-worker BTC reload."""
    s, since, boundary, drop_unlabeled = args
    try:
        g = ip._build_sym_window(s, since, drop_unlabeled=drop_unlabeled)
    except Exception:
        return None
    if g is not None and len(g):
        g = g[g["open_time"] == boundary]
        if len(g):
            return g
    return None


def build_bar(boundary, drop_unlabeled=False, workers=6):
    """Build the V0 panel rows for a single 4h `boundary` (one row per sym), PIT features, label may be NaN.
    Per-sym builds are independent → fanned across `workers` processes (forked so they share the loaded BTC
    series). The cross-sectional cohort/xs_rank below still run over ALL syms, so values are unchanged."""
    import multiprocessing as mp
    t0 = time.time()
    ip._BTC_FULL = ip.X70.load_closes("BTCUSDT")
    ip._BTC_FULL.index = pd.DatetimeIndex(ip._BTC_FULL.index).tz_convert("UTC")
    panel_syms = pd.read_parquet(ip.PANEL, columns=["symbol"])["symbol"].unique()
    syms = sorted(s for s in panel_syms if s != "BTCUSDT")
    since = boundary - pd.Timedelta(hours=4)                       # so open_time > since keeps exactly `boundary`
    # BTC-completeness gate — the DECIDE side (which books fills) must refuse a sparse BTC series just like the
    # settle path does (incremental_panel: SystemExit(3)); otherwise beta_to_btc_change_5d / idio-vol are computed
    # on a gappy BTC tail and silently wrong (the NIL -0.95 vs +0.80 beta flip). None -> decide_v1 returns {} ->
    # cycle_once falls through to the settle without booking a trade.
    _ok, _msg = ip._btc_completeness_ok(ip._BTC_FULL, since)
    if not _ok and os.environ.get("CONVEXITY_SKIP_BTC_GATE") != "1":
        print(f"[predict_at_close] BTC SPARSE — refusing to decide ({_msg})", flush=True); return None
    if not _ok:
        print(f"[predict_at_close] WARN: BTC sparse but CONVEXITY_SKIP_BTC_GATE=1 — proceeding ({_msg})", flush=True)
    args = [(s, since, boundary, drop_unlabeled) for s in syms]
    if workers > 1:
        with mp.get_context("fork").Pool(min(workers, len(syms))) as pool:   # fork → workers inherit _BTC_FULL
            results = pool.map(_build_one, args)
    else:
        results = [_build_one(a) for a in args]
    parts = [g for g in results if g is not None]
    if not parts:
        print(f"[predict_at_close] no rows built for {boundary}"); return None
    new = pd.concat(parts, ignore_index=True)
    new = ip._cohort_window(new, since)
    new["bars_since_high_xs_rank"] = new.groupby("open_time")["bars_since_high"].rank(pct=True).astype("float32")
    print(f"[predict_at_close] built {len(new)} syms @ {boundary} [{time.time()-t0:.0f}s]")
    return new


def verify():
    """Build a PAST bar with the decide path and assert its V0 features match the labeled panel exactly."""
    pan = pd.read_parquet(ip.PANEL); pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    B = sorted(pan["open_time"].unique())[-2]                      # a fully-labeled recent bar
    print(f"[verify] rebuilding {pd.Timestamp(B)} via the decide path and diffing V0 vs the labeled panel")
    got = build_bar(pd.Timestamp(B), drop_unlabeled=True)
    ref = pan[pan["open_time"] == B]
    m = ref.merge(got, on=["symbol", "open_time"], suffixes=("_p", "_d"))
    worst, wc = 0.0, None
    for c in V0 + ["bars_since_high_xs_rank"]:
        if c+"_p" in m and c+"_d" in m:
            a, b = m[c+"_p"].astype(float), m[c+"_d"].astype(float)
            d = (a - b).abs().max()
            if d > worst: worst, wc = d, c
    print(f"[verify] {len(m)} syms compared; V0 max abs diff {worst:.2e} ({wc}) "
          f"{'MATCH ✓' if worst < 1e-4 else 'DIFF ✗'}")


def _predict_book(panel_bar, models):
    """Apply the frozen per-sym models to the current bar — same path as predict_twobook_incremental."""
    rec = []
    for sym, g in panel_bar.groupby("symbol"):
        if sym not in models: continue
        m, s, h, feats = models[sym]
        try:
            pred = m.predict(x6.apply_preproc(g, feats, s, h))
        except Exception:
            continue
        rec.append(pd.DataFrame({"symbol": sym, "open_time": g["open_time"].values,
                                 "alpha_A": np.nan, "return_pct": np.nan,        # unlabeled: settle fills these
                                 "exit_time": g["exit_time"].values, "pred": pred, "fold": -1}))
    return pd.concat(rec, ignore_index=True) if rec else pd.DataFrame()


def decide():
    """Predict the current bar with both frozen books, split flow/price, write decide-preds for the bot."""
    bar = build_bar(_latest_closed_boundary(), drop_unlabeled=False)
    if bar is None: return
    OUT.parent.mkdir(parents=True, exist_ok=True); bar.to_parquet(OUT, index=False)
    ot = bar["open_time"].iloc[0]
    flow = pickle.load(open(MODELS/"twobook_flow_models.pkl", "rb"))["models"]
    price = pickle.load(open(MODELS/"twobook_price_models.pkl", "rb"))["models"]
    F = tt.build_flow()[0]
    bar_flow = bar.merge(F, on=["symbol", "open_time"], how="left")        # flow features for the flow book
    fullflow = _predict_book(bar_flow, flow)                               # flow model on all syms
    v0full = _predict_book(bar, price)                                     # price model on all syms
    flow_book = set(json.load(open(SPLIT))["flow_book"])                   # FROZEN rvol split (PIT, as-of retrain)
    bookA = fullflow[fullflow["symbol"].isin(flow_book)].copy()            # flow book = flow syms
    bookB = v0full[~v0full["symbol"].isin(flow_book)].copy()              # price book = the rest
    DECIDE_DIR.mkdir(parents=True, exist_ok=True)
    bookA.to_parquet(DECIDE_DIR/"bookA_flow.parquet", index=False)
    bookB.to_parquet(DECIDE_DIR/"bookB_price.parquet", index=False)
    print(f"[predict_at_close] DECIDE bar {ot}: bookA(flow) {len(bookA)} syms, bookB(price) {len(bookB)} syms "
          f"→ {DECIDE_DIR}")


def main():
    new = build_bar(_latest_closed_boundary(), drop_unlabeled=False)
    if new is not None:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        new.to_parquet(OUT, index=False)
        print(f"[predict_at_close] → {OUT} (current bar {new['open_time'].iloc[0]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--decide", action="store_true", help="predict current bar + split → decide-preds for the bot")
    a = ap.parse_args()
    if a.verify: verify()
    elif a.decide: decide()
    else: main()
