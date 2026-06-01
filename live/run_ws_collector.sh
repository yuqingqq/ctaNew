#!/usr/bin/env bash
# Persistent real-time Binance USDⓈ-M market-data collector for the convexity two-book live test.
# Subscribes to @aggTrade + @kline_5m for the model universe on the ROUTED /market endpoint and writes
# Vision-format daily parquets the pipeline reads. Auto-restarts on crash.
#
# Launch:  tmux new -d -s wscol 'bash /home/yuqing/ctaNew/live/run_ws_collector.sh'
# Watch:   tail -f /home/yuqing/ctaNew/live/state/ws_collector.log
# Stop:    tmux kill-session -t wscol
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd $ROOT
PY=$ROOT/.venv/bin/python
LOG=$ROOT/live/state/ws_collector.log
SYMS=$ROOT/live/state/universe174.txt

# universe = price-model keys (authoritative; survives retrains) + BTCUSDT (regime/cross reference —
# excluded from traded legs but its klines drive the bull/side/bear regime, so we must stream them).
$PY - <<'PYEOF'
import pickle
syms=set(pickle.load(open("/home/yuqing/ctaNew/live/models/twobook_price_models.pkl","rb"))["models"].keys())
syms.add("BTCUSDT")
open("/home/yuqing/ctaNew/live/state/universe174.txt","w").write(" ".join(sorted(syms)))
print(f"universe: {len(syms)} syms (incl BTCUSDT regime ref)")
PYEOF

while true; do
  echo "[$(date -u '+%F %T')] starting ws collector ($(wc -w < $SYMS) syms)" | tee -a $LOG
  $PY data_collectors/binance_ws_collector.py --syms-file $SYMS >> $LOG 2>&1
  echo "[$(date -u '+%F %T')] collector exited ($?) — restarting in 10s" | tee -a $LOG
  sleep 10
done
