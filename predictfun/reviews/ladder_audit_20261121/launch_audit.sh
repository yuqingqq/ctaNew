#!/bin/bash
# RECEIPT-LEAD-TIME-AUDIT-v1 launcher (session aea5bfb9, 2026-11-21; declaration.md). Stages: saved | causal | checks. run_light, absolute paths,
# the development slots (one day per slot for the causal audit).
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/ladder_audit_20261121; L=$D/logs; J=$L/jobs_rc.txt; mkdir -p $L
rl() { env PF_SLOT=$1 ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
case "${1:-}" in
  saved) rl claude au_saved $P/pf_ladder_audit.py saved ;;
  causal)
    ( rl claude au_causal_20261024 $P/pf_ladder_audit.py causal 20261024 ) & p1=$!
    ( rl claude2 au_causal_20261025 $P/pf_ladder_audit.py causal 20261025 ) & p2=$!
    ( rl claude3 au_causal_20261026 $P/pf_ladder_audit.py causal 20261026 ) & p3=$!
    ( rl claude4 au_causal_20261027 $P/pf_ladder_audit.py causal 20261027 ) & p4=$!
    wait $p1 $p2 $p3 $p4 ;;
  checks) rl claude au_checks $D/audit_checks.py ;;
  *) echo "usage: launch_audit.sh saved | causal | checks"; exit 2 ;;
esac
