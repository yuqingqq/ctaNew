"""X132 — Build EXPANDED-UNIVERSE (~176-sym) V0 panel + walk-forward preds.

v2 MEMORY FIX (2026-05-27): the v1 build OOM-killed at the concat step — a 176-sym
panel at 5m over 2021-2026 is ~50-100M rows (>30GB box). The strategy DECIDES only
every 4h, the target is 4h-forward, and the held-book backtest runs only on
4h-aligned rows — so the panel only needs 4h-aligned rows. v2 samples each symbol to
the 4h decision grid (open_time.hour %4==0 & minute==0) BEFORE concatenation
(~48x fewer rows → ~1-2M rows, trivial memory) and downcasts features to float32.

  CRITICAL ORDERING: build_target_z uses .shift(HORIZON=48) where 48 is in 5m bars
  (=4h, the label horizon). It MUST run on the per-symbol 5m series BEFORE the 4h
  sample, else .shift(48) would over-shift by 48*4h=8 days. build_target_z is a pure
  per-symbol groupby transform, so running it per-symbol on the 5m df is identical to
  running it on the full 5m concat. After target_z, sample to 4h, then concat.
  build_cohort_fixed runs AFTER concat on the 4h panel — it loads its own 5m closes,
  computes rvol/ret/btc_rvol on 5m, and merges by exact (symbol, open_time), so 4h
  rows (a subset of 5m rows) match correctly. bars_since_high_xs_rank is a contemporaneous
  cross-sectional rank per open_time → correct on the 4h grid.


Universe expansion 70 -> ~176 (iter-031: breadth = edge). Klines + funding for the
106 new Binance symbols (values of /tmp/addable.json) are already fetched back to
listing (2021-2026). This adapts X113 to the union universe.

SYMS = symbols in panel_hl70.parquet  ∪  the 106 addable Binance syms  MINUS 'TSTUSDT'
       (no live HL match). Dedup. (~175-176 syms.)

Pipeline (reuses X70 helpers verbatim):
  - rebuild_xs_feats per sym, CACHE-AWARE: skip if xs_feats_<sym>.parquet already
    exists (prior HL70/EXT builds cached ~72) -> only the ~104 new ones rebuild.
  - build_sym(sym, btc_close): target alpha_vs_btc_realized (4h/HORIZON=48), btc_cross,
    funding; concat; dropna(alpha).
  - x6b.build_cohort_fixed; x6.build_target_z (FAITHFUL — per-symbol rmean/rstd with
    .shift(HORIZON) + HEAVY_TAIL winsorization; NO clip-at-±N target hack); bars_since_high_xs_rank.
  - V0 walk-forward: x6.get_folds; feats = x6.BASE + present COHORT_EXTRAS;
    x6.train_per_sym_ridge.

CLIP-BUG NOTE: x6.build_target_z is reused verbatim. Its only clip is a defensive
.clip(-50,50) on the z-score (guards inf from rstd≈0), NOT the clip-at-±5 target hack
that degraded the 111-panel. The forward label itself is never clipped. (per spec)

Outputs:
  outputs/vBTC_features/panel_expanded_v0.parquet
  research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet
Does NOT touch the validated HL70 / 3yr / ext panels or their cached preds.
"""
from __future__ import annotations
import json, time, gc, importlib.util
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
# import X70 as a module to reuse its helpers (no main() executed)
spec = importlib.util.spec_from_file_location(
    "x70mod", REPO/"research/convexity_portable_2026-05-20/scripts/X70_build_3yr_and_regime_test.py")
X70 = importlib.util.module_from_spec(spec); spec.loader.exec_module(X70)
x6, x6b = X70.x6, X70.x6b

CACHE = REPO/"data/ml/cache"
OUT = REPO/"research/convexity_portable_2026-05-20/results/_cache"
PANEL_OUT = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
PRED_OUT = OUT/"x132_expanded_v0_preds.parquet"


def build_universe():
    hl70 = sorted(pd.read_parquet(
        REPO/"outputs/vBTC_features/panel_hl70.parquet", columns=["symbol"])["symbol"].unique().tolist())
    addable = list(json.loads(Path("/tmp/addable.json").read_text()).values())
    union = sorted((set(hl70) | set(addable)) - {"TSTUSDT"})
    return union, hl70, addable


def main():
    t0 = time.time()
    syms, hl70, addable = build_universe()
    # BTC needed for cross/target but is not a tradable leg in panel; keep separate.
    syms_no_btc = [s for s in syms if s != "BTCUSDT"]
    print(f"=== X132 EXPANDED-UNIVERSE V0 panel ===", flush=True)
    print(f"  hl70={len(hl70)}  addable={len(addable)}  union(-TST)={len(syms)}  "
          f"(tradable legs ex-BTC = {len(syms_no_btc)})\n", flush=True)

    # --- A. Rebuild xs_feats (CACHE-AWARE: only the ~104 new syms) ---
    print("--- A. Rebuild xs_feats (cache-aware) ---", flush=True)
    n_rebuilt = n_cached = n_fail = 0
    for i, sym in enumerate(syms, 1):
        xs_path = CACHE/f"xs_feats_{sym}.parquet"
        if xs_path.exists():
            n_cached += 1
            continue
        tf = time.time()
        try:
            ok = X70.rebuild_xs_feats(sym)
            if ok:
                n_rebuilt += 1
                print(f"  [{i}/{len(syms)}] {sym} rebuilt [{time.time()-tf:.0f}s]", flush=True)
            else:
                n_fail += 1
                print(f"  [{i}/{len(syms)}] {sym} NO-DATA (skip)", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"  [{i}/{len(syms)}] {sym} ERR {type(e).__name__}: {e}", flush=True)
    print(f"  xs_feats: {n_cached} cached, {n_rebuilt} rebuilt, {n_fail} failed "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # --- B. Build panel (robust per-sym; build_target_z on 5m, THEN sample to 4h) ---
    print("\n--- B. Build expanded panel (4h-sampled) ---", flush=True)
    btc_close = X70.load_closes("BTCUSDT")
    if btc_close is None:
        raise RuntimeError("BTCUSDT closes missing — cannot build cross/target")

    # non-feature columns kept as-is; numeric feature cols downcast to float32
    KEEP_OBJ = {"symbol", "open_time", "exit_time"}

    def _sample_4h(df):
        ot = df["open_time"]
        if not pd.api.types.is_datetime64_any_dtype(ot):
            ot = pd.to_datetime(ot, utc=True)
        m = (ot.dt.hour % 4 == 0) & (ot.dt.minute == 0)
        return df[m]

    sdfs = []
    n_ok = n_skip = 0
    for i, sym in enumerate(syms_no_btc, 1):
        try:
            sdf = X70.build_sym(sym, btc_close)
            if sdf is None or len(sdf) == 0:
                n_skip += 1
                print(f"  [{i}/{len(syms_no_btc)}] {sym} build empty (skip)", flush=True)
                continue
            sdf["open_time"] = pd.to_datetime(sdf["open_time"], utc=True)
            sdf["exit_time"] = pd.to_datetime(sdf["exit_time"], utc=True)
            sdf = sdf.dropna(subset=["alpha_vs_btc_realized"])
            if len(sdf) == 0:
                n_skip += 1; continue
            # PIT target_z on the per-symbol 5m series (HORIZON=48 5m-bars = 4h label gap)
            sdf = x6.build_target_z(sdf)
            # NOW sample to the 4h decision grid (the engine only uses 4h-aligned rows)
            sdf = _sample_4h(sdf)
            if len(sdf) == 0:
                n_skip += 1; continue
            # downcast numeric feature cols to float32 to keep concat small
            for c in sdf.columns:
                if c in KEEP_OBJ:
                    continue
                if pd.api.types.is_float_dtype(sdf[c]):
                    sdf[c] = sdf[c].astype("float32")
            sdfs.append(sdf); n_ok += 1
            del sdf
        except Exception as e:
            n_skip += 1
            print(f"  [{i}/{len(syms_no_btc)}] {sym} build ERR {type(e).__name__}: {e}", flush=True)
        if i % 20 == 0:
            tot = sum(len(d) for d in sdfs)
            print(f"  ...built {i}/{len(syms_no_btc)} ({n_ok} ok) "
                  f"4h-rows so far={tot:,} [{time.time()-t0:.0f}s]", flush=True)
    print(f"  per-sym build: {n_ok} ok, {n_skip} skipped", flush=True)

    panel = pd.concat(sdfs, ignore_index=True)
    del sdfs; gc.collect()
    print(f"  concat 4h panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    # --- C. Cohort (on 4h grid via exact open_time merge) + xs rank ---
    #     target_z already computed per-symbol on 5m above.
    panel = x6b.build_cohort_fixed(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    panel["bars_since_high_xs_rank"] = (panel.groupby("open_time")["bars_since_high"]
                                        .rank(pct=True).astype("float32"))
    panel.to_parquet(PANEL_OUT, index=False)
    print(f"\n  panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms  "
          f"{panel['open_time'].min().date()}→{panel['open_time'].max().date()}", flush=True)
    # SANITY: 4h-grid only, original HL70 set present (so we can compare expanded vs 70)
    panel_syms = set(panel["symbol"].unique())
    hl70_legs = [s for s in hl70 if s != "BTCUSDT"]
    hl70_present = [s for s in hl70_legs if s in panel_syms]
    hl70_missing = [s for s in hl70_legs if s not in panel_syms]
    bad_hours = panel.loc[(panel["open_time"].dt.hour % 4 != 0) |
                          (panel["open_time"].dt.minute != 0)]
    print(f"  SANITY: 4h-grid off-grid rows={len(bad_hours)} (want 0); "
          f"HL70 legs present {len(hl70_present)}/{len(hl70_legs)}"
          + (f" MISSING={hl70_missing}" if hl70_missing else " (all present)"), flush=True)
    print(f"  saved {PANEL_OUT.name} [{time.time()-t0:.0f}s]", flush=True)

    # --- D. V0 walk-forward preds ---
    print("\n--- D. V0 walk-forward preds (per-sym ridge) ---", flush=True)
    folds = x6.get_folds(panel)
    print(f"  {len(folds)} folds:", flush=True)
    for f, ts, te, ec in folds:
        trend, ret, av = X70.classify_fold(btc_close, ts, te)
        print(f"    f{f} {str(ts)[:10]}→{str(te)[:10]} BTC{ret*100:+.0f}% vol{av*100:.0f}% {trend}", flush=True)
    feats = [f for f in x6.BASE + x6.COHORT_EXTRAS if f in panel.columns]
    print(f"  V0 features ({len(feats)}): {feats}", flush=True)
    apd = x6.train_per_sym_ridge(panel, folds, feats, label="x132_expanded_v0")
    apd.to_parquet(PRED_OUT, index=False)
    ic = float(apd["pred"].corr(apd["alpha_A"]))
    print(f"\n  trained {len(apd):,} preds × {apd['symbol'].nunique()} syms, "
          f"full-sample IC={ic:+.4f}", flush=True)
    print(f"  pred range [{apd['pred'].min():+.3f}, {apd['pred'].max():+.3f}] "
          f"mean {apd['pred'].mean():+.4f} std {apd['pred'].std():.4f}", flush=True)
    print(f"  saved {PRED_OUT.name}", flush=True)
    print(f"DONE [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
