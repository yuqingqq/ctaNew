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
SYMS=$ROOT/live/state/collector_syms.txt   # only what we MONITOR: low-vol book + BTC (no high-vol, no flow)

# Two lists: universe174.txt = full model set (kept for the maturity gate); collector_syms.txt = the
# subset we actually stream = low-vol book (model keys MINUS the frozen top-80 high-vol) + BTCUSDT
# (regime/cross-asset ref). v1 trades only the low-vol book, so the high-vol 80 aren't worth streaming.
$PY - <<'PYEOF'
import pickle, json
R = "/home/yuqing/ctaNew"
mk = set(pickle.load(open(R+"/live/models/convexity_v1_short_model.pkl","rb"))["models"].keys()); mk.add("BTCUSDT")
open(R+"/live/state/universe174.txt","w").write(" ".join(sorted(mk)))            # full set — maturity gate
excl = set(json.load(open(R+"/live/models/convexity_v1_universe.json"))["exclude_high_vol"])
mon = (mk - excl); mon.add("BTCUSDT")                                            # low-vol + BTC — what we stream
open(R+"/live/state/collector_syms.txt","w").write(" ".join(sorted(mon)))
print(f"monitor: {len(mon)} low-vol+BTC syms (dropped {len(excl)} high-vol; kline+markPrice only, no aggTrade)")
PYEOF

while true; do
  echo "[$(date -u '+%F %T')] starting ws collector ($(wc -w < $SYMS) syms)" | tee -a $LOG
  $PY data_collectors/binance_ws_collector.py --syms-file $SYMS >> $LOG 2>&1
  echo "[$(date -u '+%F %T')] collector exited ($?) — restarting in 10s" | tee -a $LOG
  sleep 10
done
