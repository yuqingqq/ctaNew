"""Side-by-side comparison: paper_bot predictions from HL vs Binance Vision.

Verifies that swapping the data feed from Binance to HL doesn't substantially
change v6_clean's portfolio choices. Concern: HL volume is in coin units and
much smaller than Binance — `volume_ma_50` and other volume-derived features
may shift distribution and confuse trees trained on Binance scale.

If HL and Vision pick the same names (or near-same), the swap is safe.
If picks diverge, we need to mix sources (HL prices + Vision volume).
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from live.paper_bot import (
    refresh_klines_cache, build_panel_for_inference, predict_for_bar,
    select_portfolio, load_model_artifact, LOOKBACK_DAYS, TOP_K,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compare")


def _run_one(source: str, models, meta, target_cutoff=None) -> dict:
    feat_cols = list(meta["feat_cols"])
    sym_to_id = meta["sym_to_id"]
    universe = sorted(sym_to_id.keys())
    log.info("source=%s: refreshing klines (LOOKBACK_DAYS=%d)", source, LOOKBACK_DAYS)
    klines_by_sym = refresh_klines_cache(universe, days=LOOKBACK_DAYS, source=source)
    log.info("source=%s: got %d/%d symbols", source, len(klines_by_sym), len(universe))
    if target_cutoff is not None:
        klines_by_sym = {s: kl[kl.index <= target_cutoff] for s, kl in klines_by_sym.items()}
        log.info("source=%s: trimmed klines to <= %s", source, target_cutoff)
    panel = build_panel_for_inference(klines_by_sym, sym_to_id)
    target_time = panel["open_time"].max()
    preds = predict_for_bar(models, panel, target_time, feat_cols)
    log.info("source=%s: target_time=%s, n_preds=%d", source, target_time, len(preds))
    top, bot, scale_L, scale_S = select_portfolio(preds, top_k=TOP_K)
    return {
        "source": source,
        "target_time": str(target_time),
        "preds": preds,
        "long": top["symbol"].tolist(),
        "short": bot["symbol"].tolist(),
        "scale_L": scale_L,
        "scale_S": scale_S,
    }


def main():
    log.info("Loading model artifact...")
    models, meta = load_model_artifact()

    log.info("=== Running vision first to establish a common target_time ===")
    res_v = _run_one("vision", models, meta)
    cutoff = pd.Timestamp(res_v["target_time"])
    log.info("=== Running HL with target_cutoff=%s for apples-to-apples ===", cutoff)
    res_h = _run_one("hl", models, meta, target_cutoff=cutoff)

    print("\n" + "=" * 100)
    print("DATA-SOURCE COMPARISON: HL vs Binance Vision")
    print("=" * 100)
    print(f"\n  vision target_time: {res_v['target_time']}")
    print(f"  hl     target_time: {res_h['target_time']}")
    print()
    print(f"  vision long  top-{TOP_K}: {res_v['long']}")
    print(f"  hl     long  top-{TOP_K}: {res_h['long']}")
    print(f"  vision short bot-{TOP_K}: {res_v['short']}")
    print(f"  hl     short bot-{TOP_K}: {res_h['short']}")
    print()
    overlap_long = set(res_v["long"]) & set(res_h["long"])
    overlap_short = set(res_v["short"]) & set(res_h["short"])
    print(f"  long overlap:  {len(overlap_long)}/{TOP_K}  {sorted(overlap_long)}")
    print(f"  short overlap: {len(overlap_short)}/{TOP_K}  {sorted(overlap_short)}")
    print()

    pv = res_v["preds"].set_index("symbol")["pred"]
    ph = res_h["preds"].set_index("symbol")["pred"]
    common_syms = pv.index.intersection(ph.index)
    pv_c = pv.loc[common_syms]
    ph_c = ph.loc[common_syms]
    rank_corr = pv_c.rank().corr(ph_c.rank())
    print(f"  Per-symbol pred Spearman correlation: {rank_corr:+.4f}  (1.0 = identical ranks)")

    sbs = pd.DataFrame({"vision_pred": pv_c, "hl_pred": ph_c}).sort_values("vision_pred")
    sbs["vision_rank"] = sbs["vision_pred"].rank().astype(int)
    sbs["hl_rank"] = sbs["hl_pred"].rank().astype(int)
    sbs["rank_diff"] = sbs["hl_rank"] - sbs["vision_rank"]
    print()
    print("  Side-by-side predictions (sorted by vision_pred):")
    print(sbs.round(4).to_string())

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    sbs.to_csv(out / "compare_data_sources.csv")

    if rank_corr >= 0.85 and len(overlap_long) >= TOP_K - 1 and len(overlap_short) >= TOP_K - 1:
        print(f"\n  ✓ HL feed produces near-identical portfolio (rank corr ≥0.85, overlap ≥{TOP_K - 1}/{TOP_K})")
    elif rank_corr >= 0.7:
        print(f"\n  ⚠️  Moderate divergence (rank corr {rank_corr:.2f}). Forward test will reveal Sharpe impact.")
    else:
        print(f"\n  ⚠️  HIGH divergence (rank corr {rank_corr:.2f}). Consider mixing sources.")


if __name__ == "__main__":
    main()
