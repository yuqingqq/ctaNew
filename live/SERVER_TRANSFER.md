# Convexity two-book — two-server architecture

## Roles
- **Training box (this one):** keeps full-history data, retrains models MONTHLY, commits the ~44MB
  twobook_{flow,price}_models.pkl to git + pushes. Runs `live/monthly_retrain.sh` (cron: 1st, 02:00 UTC).
- **Live box (new server):** `git pull` → gets code + fresh models. Connects to LIVE Binance feed,
  builds features incrementally from its own data, predicts with the pulled models, trades. Pulls fresh
  models each month. **Does NOT need the training box's 60GB caches** — only models (git) + a recent
  warmup it fetches live.

## Bringing up the live box
1. `git clone` / `git pull` (code + the two twobook model .pkl come with it).
2. **Warmup data (fetched live, one-time):** pull the last ~60 days of 5m klines + aggTrades for the 175
   universe syms (small, minutes), build the feature caches once:
     python3 live/incremental_xs_feats.py --workers 6   # builds xs_feats from the warmup klines
     python3 live/ingest_flow_daily.py --workers 4       # builds flow caches from warmup aggTrades
     python3 live/build_panel_fast.py --workers 4        # one-time full panel from the warmup
   (Or rsync these caches from the training box if you'd rather not refetch — optional, not required.)
3. **Swap ingestion to the LIVE feed:** replace the daily-archive loader with Binance REST/websocket for
   klines+aggTrades+funding. The incremental compute layer (xs_feats / panel / preds / mom-beta) is
   unchanged — only the data source changes.
4. Install crons on the LIVE box: 4h notify (convexity_notify.py), the cycle (run_convexity_daily.sh or a
   4h variant), and a daily `git pull` to receive fresh models. Set up .env (Telegram creds).
5. Verify: python3 live/predict_twobook_incremental.py  (loads pulled models, appends new-bar preds, ~7s).

## What flows where
- **git (small, versioned):** all code + the two twobook model .pkl (refreshed monthly by the training box).
- **NOT in git / fetched-or-rsynced:** the 55GB xs_feats+flow caches, 4.4GB klines, run-state. The live box
  builds these from its own live feed; the training box keeps the full history for retraining.

## Models train on the latest data
train_twobook_models.py defaults to fit_cut = latest_panel_date - 1d embargo (the freshest labelled data),
recency-60 weighted. Monthly cadence matches the validated walk-forward. Current pushed models: fit_cut 05-29.
