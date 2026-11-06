#!/usr/bin/env bash
# run_light.sh -- launch ONE light P-2026-005 research process (user, 2026-11-01:
# "allow a light process, and watch out memory usage and disk usage").
# Refuses if another of our pf_ jobs runs, MemAvailable < 8 GiB, or the data
# disk has < 100 GiB free. Pins core 8, caps address space at 4 GiB, logs the
# measured peak RSS (/usr/bin/time) and the disk-free delta per run.
# Usage: predictfun/research/run_light.sh <tag> -- python3 -m predictfun.research.<mod> args...
set -u
ROOT=/home/yuqing/ctaNew; LOG=$ROOT/predictfun/research/run_light.log
TAG="${1:?tag}"; shift; [ "${1:-}" = "--" ] && shift
if ps -eo args | grep -E '^(/\S*/)?python(3(\.[0-9]+)?)? .*predictfun\.research\.' | grep -v grep -q; then
  echo "REFUSED: a predictfun research process is already running" >&2; exit 1; fi
AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
DF=$(df -BG --output=avail $ROOT/data | tail -1 | tr -dc 0-9)
[ "$AV" -lt 8 ] && { echo "REFUSED: MemAvailable ${AV} GiB < 8" >&2; exit 1; }
[ "$DF" -lt 100 ] && { echo "REFUSED: disk free ${DF} GiB < 100" >&2; exit 1; }
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cd $ROOT
# pyarrow reserves ~1 GiB of address space under an AS cap unless told to use the system allocator
export ARROW_DEFAULT_MEMORY_POOL=system MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1
taskset -c 8 prlimit --as=$((4*1024*1024*1024)) -- /usr/bin/time -f '%M %e %U' -o /tmp/claude-1001/pf_time.$$ "$@"
RC=$?
read PEAK_KB WALL USER < <(tail -1 /tmp/claude-1001/pf_time.$$ 2>/dev/null) || true
case "${PEAK_KB:-}" in ''|*[!0-9]*) PEAK_KB=0;; esac; rm -f /tmp/claude-1001/pf_time.$$
DF2=$(df -BG --output=avail $ROOT/data | tail -1 | tr -dc 0-9)
echo "$T0 tag=$TAG rc=$RC peak_rss_mib=$(( ${PEAK_KB:-0} / 1024 )) wall_s=${WALL:-?} cpu_user_s=${USER:-?} memavail_start_gib=$AV disk_free_gib=${DF}->${DF2}" | tee -a $LOG
exit $RC
