# Convexity two-book — transfer to live-data server

Goal: run the forward test on another server WITHOUT re-downloading data or retraining.
Code comes via git; models + data caches + run-state transfer via rsync (too large for git).

## 1. Code (git)
On the new server:  git clone <repo> && git checkout main   (or: git pull)

## 2. Artifacts + data (rsync from THIS box → new server)
Set:  DST=user@newhost:/home/yuqing/ctaNew
```
# (a) trained models — ~44MB (avoids retrain)
rsync -av live/models/twobook_flow_models.pkl live/models/twobook_price_models.pkl  $DST/live/models/

# (b) panel + preds + book state — ~190MB (current forward-test state to continue from)
rsync -av outputs/vBTC_features/panel_expanded_v0.parquet                 $DST/outputs/vBTC_features/
rsync -av live/state/convexity/hl/fullflow_hl60.parquet live/state/convexity/hl/v0full_hl60.parquet  $DST/live/state/convexity/hl/
rsync -av live/state/convexity/split2/bookA_hv80.parquet live/state/convexity/split2/bookB_hv80.parquet $DST/live/state/convexity/split2/
rsync -av live/state/convexity_bookA/ $DST/live/state/convexity_bookA/
rsync -av live/state/convexity_bookB/ $DST/live/state/convexity_twobook/ 2>/dev/null; rsync -av live/state/convexity_bookB/ $DST/live/state/convexity_bookB/

# (c) feature/flow/kline caches — ~60GB (needed for incremental warmup going forward)
rsync -av --progress data/ml/cache/         $DST/data/ml/cache/          # flow_*.parquet + xs_feats_*.parquet + funding (~55GB)
rsync -av --progress data/ml/test/parquet/klines/  $DST/data/ml/test/parquet/klines/   # ~4.4GB

# (d) telegram creds (or create fresh on the new server)
rsync -av .env $DST/    # contains TELEGRAM_BOT_TOKEN / CHAT_ID
```

## 3. On the new server
- Install cron (4h notify + daily pipeline + monthly retrain) — see CONVEXITY_BOT_LAUNCH.md.
- Swap the daily-archive ingestion for the LIVE Binance REST/websocket feed (the incremental compute
  layer — xs_feats/panel/preds/mom-beta — carries over unchanged; only ingestion changes).
- Verify: python3 live/predict_twobook_incremental.py  (should append a few bars, ~7s).

## Minimal vs full
(c) is the bulk (~60GB). The xs_feats caches (~52GB) are the largest; they're needed for the incremental
panel/xs_feats warmup. If bandwidth-constrained, they CAN be rebuilt on the new server from klines
(build_panel_fast.py / incremental_xs_feats.py) — but that's the ~2hr rebuild we're avoiding by transferring.
