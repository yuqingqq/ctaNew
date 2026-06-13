"""Stage-1 heavy fetch: aggTrades for the 26 missing symbols, per-symbol
pull -> 5m flow features -> delete raw (disk-bounded). Produces
data/ml/cache/flow_<SYM>.parquet (schema-identical to the cached 25)."""
import logging, shutil, sys, time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, "/home/yuqing/ctaNew")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("fetch_flow")
from data_collectors.binance_vision_loader import LoaderConfig, list_aggtrade_paths
from features_ml.trade_flow import TradeFlowConfig, aggregate_trades_streaming

REPO = Path("/home/yuqing/ctaNew")
DATA_DIR = REPO / "data/ml/test/parquet"
CACHE = REPO / "data/ml/cache"
START, END = datetime(2025, 3, 23).date(), datetime(2026, 5, 6).date()
MISS = ("AAVEUSDT ASTERUSDT AXSUSDT BIOUSDT ENAUSDT ETCUSDT GMXUSDT HBARUSDT "
        "HYPEUSDT ICPUSDT JTOUSDT JUPUSDT LDOUSDT ONDOUSDT ORDIUSDT PENDLEUSDT "
        "PENGUUSDT PUMPUSDT STRKUSDT TAOUSDT TONUSDT TRBUSDT VIRTUALUSDT VVVUSDT "
        "WIFUSDT ZECUSDT").split()
cfg_tf = TradeFlowConfig(bar_interval="5min", compute_kyle_lambda=True)
t0 = time.time()
done, failed = [], []
for i, sym in enumerate(MISS, 1):
    out = CACHE / f"flow_{sym}.parquet"
    if out.exists():
        log.info("[%s] flow cache exists; skip", sym); done.append(sym); continue
    try:
        lc = LoaderConfig(symbol=sym, out_dir=DATA_DIR, max_workers=8)
        paths = list_aggtrade_paths(START, END, cfg=lc)   # downloads daily parquets
        paths = sorted(p for p in paths if p and Path(p).exists())
        if not paths:
            log.warning("[%s] no aggTrade parquets (likely no Vision history)", sym)
            failed.append(sym); continue
        feats = aggregate_trades_streaming(paths, cfg_tf)
        feats = feats.dropna(subset=["vpin"], how="all")
        feats.to_parquet(out, compression="zstd")
        log.info("[%s] flow %d rows %s..%s (%d/%d, %.0fs)", sym, len(feats),
                 feats.index.min() if len(feats) else "NA",
                 feats.index.max() if len(feats) else "NA",
                 i, len(MISS), time.time()-t0)
        done.append(sym)
    except Exception as e:
        log.error("[%s] FAILED: %s", sym, e); failed.append(sym)
    finally:
        raw = DATA_DIR / "aggTrades" / sym       # disk-bound: drop raw aggTrades
        if raw.exists():
            shutil.rmtree(raw, ignore_errors=True)
            log.info("[%s] raw aggTrades removed", sym)
log.info("FLOW fetch done [%.0fs] done=%d failed=%d failed_syms=%s",
         time.time()-t0, len(done), len(failed), failed)
print("FLOW_FETCH_DONE", flush=True)
