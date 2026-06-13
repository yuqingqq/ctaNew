#!/bin/bash
# Full V5 3-year pipeline: wait for downloads → build flow features → X78
cd /home/yuqing/ctaNew
echo "$(date) pipeline start"

# 1. Wait for OKX+CB extension
while pgrep -f "extend_okx_cb_to_2023.py" >/dev/null 2>&1; do sleep 120; done
echo "$(date) OKX+CB extension done"

# 2. Wait for aggTrades download
while pgrep -f "extend_aggtrades_to_2023.py" >/dev/null 2>&1; do sleep 120; done
echo "$(date) aggTrades download done"

# 3. Build flow features over full 3-year range for the 45 candidates
SYMS="ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT ADAUSDT AVAXUSDT LINKUSDT DOTUSDT ATOMUSDT LTCUSDT BCHUSDT NEARUSDT UNIUSDT TIAUSDT SUIUSDT SEIUSDT INJUSDT ARBUSDT APTUSDT OPUSDT AAVEUSDT AXSUSDT FILUSDT ETCUSDT TRBUSDT WLDUSDT ICPUSDT ONDOUSDT PENDLEUSDT LDOUSDT JTOUSDT ENAUSDT HBARUSDT TONUSDT STRKUSDT WIFUSDT ORDIUSDT JUPUSDT GMXUSDT TAOUSDT RUNEUSDT SUSDT ZECUSDT BTCUSDT"
echo "$(date) building flow features (force rebuild over 3yr)"
python3 -u -m scripts.build_aggtrade_features --symbols $SYMS --force > /tmp/build_flow_3yr.log 2>&1
echo "$(date) flow features done"

# 4. Launch X78
echo "$(date) launching X78"
python3 -u research/convexity_portable_2026-05-20/scripts/X78_build_v5_3yr.py > /tmp/x78.log 2>&1
echo "$(date) X78 done — PIPELINE COMPLETE"
