#!/bin/bash
# ENTRY-TARGET-AUDIT-v1 (review d6d5d55e): one job at a time via run_light (absolute paths). Its own launcher; never edited while it runs.
#   run | checks
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; RC=$R/predictfun/reviews/entry_target_audit_20261121; L=$RC/logs; mkdir -p $L; J=$L/jobs_rc.txt
rl() { env PF_SLOT=claude $P/run_light.sh $1 -- python3 "${@:2}" > $L/$1.log 2>&1; echo "$1 rc=$?" >> $J; }
case "$1" in
  run) rl audit_run $P/pf_entry_audit.py run ;;
  checks) rl audit_checks $RC/audit_checks.py ;;
esac
