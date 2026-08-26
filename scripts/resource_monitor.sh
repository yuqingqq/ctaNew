#!/usr/bin/env bash
# Continuous box resource sampler (user directive 2026-08-26, R-163).
# One CSV line per 60s to the journal + ALERT lines on threshold breach.
# View: journalctl --user -u resource-monitor -f
set -u
CG=/sys/fs/cgroup/user.slice/user-1001.slice/user@1001.service
prev_r_cpu=0; prev_c_cpu=0
echo "ts,mem_avail_mib,swap_used_mib,research_mib,collectors_mib,docker_mib,load1,research_cpu_pct,collectors_cpu_pct"
while true; do
  ma=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  sw=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)
  r=$(( $(cat $CG/research.slice/memory.current 2>/dev/null || echo 0) / 1048576 ))
  c=$(( $(cat $CG/collectors.slice/memory.current 2>/dev/null || echo 0) / 1048576 ))
  d=0; for f in /sys/fs/cgroup/system.slice/docker-*.scope/memory.current; do
    [ -r "$f" ] && d=$((d + $(cat "$f") / 1048576)); done
  l1=$(cut -d' ' -f1 /proc/loadavg)
  rc=$(awk '/usage_usec/{print $2}' $CG/research.slice/cpu.stat 2>/dev/null || echo 0)
  cc=$(awk '/usage_usec/{print $2}' $CG/collectors.slice/cpu.stat 2>/dev/null || echo 0)
  rp=$(( (rc - prev_r_cpu) / 600000 )); cp=$(( (cc - prev_c_cpu) / 600000 ))
  [ $prev_r_cpu -eq 0 ] && rp=0; [ $prev_c_cpu -eq 0 ] && cp=0
  prev_r_cpu=$rc; prev_c_cpu=$cc
  echo "$(date -u +%FT%TZ),$ma,$sw,$r,$c,$d,$l1,$rp,$cp"
  [ "$ma" -lt 4096 ] && echo "ALERT MEM_AVAIL_LOW ${ma}MiB"
  [ "$sw" -gt 512 ] && echo "ALERT SWAP_IN_USE ${sw}MiB"
  ra=$(( $(awk '/^anon /{print $2}' $CG/research.slice/memory.stat 2>/dev/null || echo 0) / 1048576 ))
  [ "$ra" -gt 16384 ] && echo "ALERT RESEARCH_ANON_HIGH ${ra}MiB"
  for u in collect-hf collect-hl pm-collector-clob pm-collector-prices; do
    systemctl --user is-active --quiet "$u" || echo "ALERT UNIT_DOWN $u"
  done
  sleep 60
done
