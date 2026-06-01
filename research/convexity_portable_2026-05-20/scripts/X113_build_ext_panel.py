"""X113 — Build EXTENDED 2021-2026 V0 panel + walk-forward preds for the 23 pre-2023 symbols.

Reuses X70's pipeline (rebuild_xs_feats / build_sym / cohort / target_z / get_folds /
train_per_sym_ridge) but over the extended klines (2021-2022 just fetched + 2023-2026
existing) and restricted to the 23 symbols that existed pre-2023. Spans 2021 bull /
2022 LUNA-FTX bear / 2023-24 recovery / 2025 peak / 2026 decay → MULTIPLE regime flips
so the sign-flip predictor (task #109, X111) has flip events to learn from.

Outputs:
  outputs/vBTC_features/panel_ext2021_v0.parquet
  research/convexity_portable_2026-05-20/results/_cache/x113_ext_v0_preds.parquet
Does NOT touch the validated 44-sym panel_3yr_v0 / x70 preds.
"""
from __future__ import annotations
import json, time, importlib.util
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
# import X70 as a module to reuse its helpers (no main() executed)
spec = importlib.util.spec_from_file_location(
    "x70mod", REPO/"research/convexity_portable_2026-05-20/scripts/X70_build_3yr_and_regime_test.py")
X70 = importlib.util.module_from_spec(spec); spec.loader.exec_module(X70)
x6, x6b = X70.x6, X70.x6b

SYMS = json.loads(Path("/tmp/pre23_syms.json").read_text())   # 23 pre-2023 syms (excl BTC)
OUT = REPO/"research/convexity_portable_2026-05-20/results/_cache"
PANEL_OUT = REPO/"outputs/vBTC_features/panel_ext2021_v0.parquet"
PRED_OUT = OUT/"x113_ext_v0_preds.parquet"


def main():
    t0=time.time()
    print(f"=== X113 extended 2021-2026 V0 panel ({len(SYMS)} syms) ===\n", flush=True)

    print("--- A. Funding 2021-01..2026-05 ---", flush=True)
    for i,sym in enumerate(SYMS,1):
        try: X70.load_funding_rate(sym, start_month="2021-01", end_month="2026-05")
        except Exception as e: print(f"  {sym} funding ERR {e}", flush=True)
    print(f"  funding done [{time.time()-t0:.0f}s]", flush=True)

    print("\n--- B. Rebuild xs_feats over extended klines ---", flush=True)
    for i,sym in enumerate(SYMS,1):
        tf=time.time()
        try:
            ok=X70.rebuild_xs_feats(sym)
            print(f"  [{i}/{len(SYMS)}] {sym} {'rebuilt' if ok else 'no-data'} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e: print(f"  [{i}/{len(SYMS)}] {sym} ERR {e}", flush=True)

    print("\n--- C. Build panel ---", flush=True)
    btc_close=X70.load_closes("BTCUSDT")
    sdfs=[]
    for i,sym in enumerate(SYMS,1):
        try:
            sdf=X70.build_sym(sym, btc_close)
            if sdf is not None and len(sdf)>0: sdfs.append(sdf)
        except Exception as e: print(f"  {sym} build ERR {e}", flush=True)
    panel=pd.concat(sdfs, ignore_index=True)
    panel["open_time"]=pd.to_datetime(panel["open_time"],utc=True)
    panel["exit_time"]=pd.to_datetime(panel["exit_time"],utc=True)
    panel=panel.dropna(subset=["alpha_vs_btc_realized"])
    panel=x6b.build_cohort_fixed(panel)
    panel=x6.build_target_z(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    panel["bars_since_high_xs_rank"]=panel.groupby("open_time")["bars_since_high"].rank(pct=True).astype("float32")
    panel.to_parquet(PANEL_OUT, index=False)
    print(f"  panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms  "
          f"{panel['open_time'].min().date()}→{panel['open_time'].max().date()}", flush=True)
    print(f"  saved {PANEL_OUT.name} [{time.time()-t0:.0f}s]", flush=True)

    print("\n--- D. V0 walk-forward preds ---", flush=True)
    folds=x6.get_folds(panel)
    print(f"  {len(folds)} folds:")
    for f,ts,te,ec in folds:
        trend,ret,av=X70.classify_fold(btc_close,ts,te)
        print(f"    f{f} {str(ts)[:10]}→{str(te)[:10]} BTC{ret*100:+.0f}% vol{av*100:.0f}% {trend}", flush=True)
    feats=[f for f in x6.BASE+x6.COHORT_EXTRAS if f in panel.columns]
    apd=x6.train_per_sym_ridge(panel, folds, feats, label="x113_ext_v0")
    apd.to_parquet(PRED_OUT, index=False)
    ic=float(apd["pred"].corr(apd["alpha_A"]))
    print(f"\n  trained {len(apd):,} preds, full-sample IC={ic:+.4f}, saved {PRED_OUT.name}")
    print(f"DONE [{time.time()-t0:.0f}s]")


if __name__=="__main__":
    main()
