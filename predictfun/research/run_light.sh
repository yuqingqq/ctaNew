#!/usr/bin/env bash
# run_light.sh -- launch ONE light P-2026-005 research process in a slot (user, 2026-11-01:
# "allow a light process, and watch out memory usage and disk usage"; 2026-11-12: "Can we launch 2 parallel
# processes" / "One for codex and one here").
# Two slots, one core each: PF_SLOT=codex (the default; core 8, the Codex session) and PF_SLOT=claude (core 9, the
# Claude session). Refuses if a research process already runs on the slot's core, if 2 already run (the cap),
# if MemAvailable < 8 GiB, or if the data disk has < 100 GiB free. A research process is a python whose argv names
# predictfun.research.<mod> or a predictfun/research/pf_*.py script (script paths were invisible to the check
# before 2026-11-12). Pins the slot's core, caps address space at 4 GiB, logs the measured peak RSS
# (/usr/bin/time), the disk-free delta and the slot per run.
# Usage: [PF_SLOT=claude] predictfun/research/run_light.sh <tag> -- python3 -m predictfun.research.<mod> args...
set -u
ROOT=/home/yuqing/ctaNew; LOG=$ROOT/predictfun/research/run_light.log
TAG="${1:?tag}"; shift; [ "${1:-}" = "--" ] && shift
SLOT="${PF_SLOT:-codex}"
case "$SLOT" in codex) CORE=8;; claude) CORE=9;; *) echo "REFUSED: PF_SLOT must be codex or claude, not '$SLOT'" >&2; exit 1;; esac
N=0; BUSY=
for P in $(ps -eo pid=,args= | awk '$2 ~ /^(\/[^ ]*\/)?python(3(\.[0-9]+)?)?$/ && (/predictfun\.research\./ || /predictfun\/research\/pf_/) {print $1}'); do
  N=$((N+1))
  [ "$(awk '/^Cpus_allowed_list/{print $2}' /proc/$P/status 2>/dev/null)" = "$CORE" ] && BUSY=$P
done
[ -n "$BUSY" ] && { echo "REFUSED: research slot $SLOT (core $CORE) is in use by pid $BUSY" >&2; exit 1; }
[ "$N" -ge 2 ] && { echo "REFUSED: $N predictfun research processes already running (cap 2)" >&2; exit 1; }
AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
DF=$(df -BG --output=avail $ROOT/data | tail -1 | tr -dc 0-9)
[ "$AV" -lt 8 ] && { echo "REFUSED: MemAvailable ${AV} GiB < 8" >&2; exit 1; }
[ "$DF" -lt 100 ] && { echo "REFUSED: disk free ${DF} GiB < 100" >&2; exit 1; }
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cd $ROOT
# pyarrow reserves ~1 GiB of address space under an AS cap unless told to use the system allocator
export ARROW_DEFAULT_MEMORY_POOL=system MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1
taskset -c $CORE prlimit --as=$((4*1024*1024*1024)) -- /usr/bin/time -f '%M %e %U' -o /tmp/claude-1001/pf_time.$$ "$@"
RC=$?
read PEAK_KB WALL USER < <(tail -1 /tmp/claude-1001/pf_time.$$ 2>/dev/null) || true
case "${PEAK_KB:-}" in ''|*[!0-9]*) PEAK_KB=0;; esac; rm -f /tmp/claude-1001/pf_time.$$
DF2=$(df -BG --output=avail $ROOT/data | tail -1 | tr -dc 0-9)
echo "$T0 tag=$TAG rc=$RC peak_rss_mib=$(( ${PEAK_KB:-0} / 1024 )) wall_s=${WALL:-?} cpu_user_s=${USER:-?} memavail_start_gib=$AV disk_free_gib=${DF}->${DF2} slot=$SLOT" | tee -a $LOG
exit $RC
