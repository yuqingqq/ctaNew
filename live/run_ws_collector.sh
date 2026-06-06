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
SYMS=$ROOT/live/state/collector_syms.txt   # what we stream: FULL live universe + BTC (xs-rank cohort)

# Stream the FULL canonical live universe, NOT just the traded low-vol book. bars_since_high_xs_rank is
# ranked over the whole 175-XS panel (validated 2026-06-06), so the live cross-section must see EVERY symbol
# or the feature drifts from the backtest — the 174->94 cohort collapse that came from streaming only the 94
# traded names while the 80 high-vol peers froze. collector_universe.txt (panel syms minus durably-dead/halted,
# e.g. VINE) is the canonical list, regenerated at each retrain and git-pulled here. universe174.txt = full
# model set (maturity gate). The high-vol 80 are klines-only peers — never scored, never traded.
$PY - <<'PYEOF'
import pickle
R = "/home/yuqing/ctaNew"
mk = set(pickle.load(open(R+"/live/models/convexity_v1_short_model.pkl","rb"))["models"].keys()); mk.add("BTCUSDT")
open(R+"/live/state/universe174.txt","w").write(" ".join(sorted(mk)))            # full model set — maturity gate
canon = open(R+"/live/collector_universe.txt").read().split()                    # canonical live-feed set (dead dropped)
mon = set(canon); mon.add("BTCUSDT")                                             # full live universe + BTC ref
open(R+"/live/state/collector_syms.txt","w").write(" ".join(sorted(mon)))
print(f"monitor: {len(mon)} live-universe syms (collector_universe.txt + BTC; kline+markPrice)")
PYEOF

while true; do
  echo "[$(date -u '+%F %T')] starting ws collector ($(wc -w < $SYMS) syms)" | tee -a $LOG
  $PY data_collectors/binance_ws_collector.py --syms-file $SYMS >> $LOG 2>&1
  echo "[$(date -u '+%F %T')] collector exited ($?) — restarting in 10s" | tee -a $LOG
  sleep 10
done
