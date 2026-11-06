#!/usr/bin/env bash
# Confirmation C2 driver: runs each candidate day once it is complete (next day's 01Z hour closed, +10 min), in order,
# until 5 days are ADMITTED (a day with SOURCE_GAP / WINDOW_GAP / FILL_INVARIANT / FAILED is replaced by the next day),
# then writes the report. Gives up after 20261110 (report then says INSUFFICIENT_DAYS). Launcher refusals retry every 10 min.
# Started detached (setsid nohup); heartbeat in data/predict_fun/research/c2/auto.heartbeat.
set -u
cd /home/yuqing/ctaNew
C=data/predict_fun/research/c2; A=$C/admission.jsonl; mkdir -p $C; echo $$ > $C/auto.pid
n_adm() { [ -f $A ] && python3 -c "
import json,sys
last={}
for l in open(sys.argv[1]):
    if l.strip(): r=json.loads(l); last[r['day']]=r['status']
print(sum(v=='ADMITTED' for v in last.values()))" $A || echo 0; }
has_rec() { [ -f $A ] && grep -q "\"day\": \"$1\"" $A; }
for D in 20261101 20261102 20261103 20261104 20261105 20261106 20261107 20261108 20261109 20261110; do
  [ "$(n_adm)" -ge 5 ] && break
  has_rec $D && continue
  READY=$(date -u -d "$(date -u -d "$D +1 day" +%F) 02:10" +%s)
  while [ "$(date -u +%s)" -lt "$READY" ]; do date -u +%FT%TZ" waiting for $D" > $C/auto.heartbeat; sleep 300; done
  while true; do
    date -u +%FT%TZ" running $D" > $C/auto.heartbeat
    predictfun/research/run_c2_day.sh $D >> $C/auto.log 2>&1; rc=$?
    [ $rc -eq 75 ] && { sleep 600; continue; }
    echo "$(date -u +%FT%TZ) day $D rc=$rc" >> $C/auto.log; break
  done
  [ $rc -ne 0 ] && { echo "$(date -u +%FT%TZ) stopping: day $D rc=$rc" >> $C/auto.log; date -u +%FT%TZ" STOPPED at $D rc=$rc" > $C/auto.heartbeat; exit 1; }
done
until predictfun/research/run_light.sh c2_report -- python3 -m predictfun.research.pf_c2_report > data/predict_fun/research/logs/c2_report.log 2>&1; do
  grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' data/predict_fun/research/logs/c2_report.log || { date -u +%FT%TZ" REPORT FAILED" > $C/auto.heartbeat; exit 1; }
  sleep 600; done
date -u +%FT%TZ" DONE" > $C/auto.heartbeat
