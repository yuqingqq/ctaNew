#!/bin/bash
# PRESSURE-RESIDUAL-TIMING-AUDIT-v1 launcher (session aea5bfb9, 2026-11-21; declaration.md). Stages, in order:
#   manifest | pilot | rehearse | project | full | checks
# run_light, absolute paths, one job at a time; the full audit runs only if the write-once projection allows it. No tape load, no replay.
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/ladder_pressure_residual_20261121; L=$D/logs; J=$L/jobs_rc.txt; mkdir -p $L
rl() { env PF_SLOT=$1 ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
case "${1:-}" in
  manifest) rl claude gr_manifest $P/pf_ladder_pressure_residual.py manifest ;;
  pilot) rl claude gr_audit_pilot $P/pf_ladder_pressure_residual.py audit pilot ;;
  rehearse) rl claude gr_rehearse_checks $D/residual_checks.py rehearse ;;
  project) rl claude gr_projection $D/tools/pilot_projection_residual.py ;;
  full)
    python3 -c "import json, sys; sys.exit(0 if json.load(open('$L/pilot_projection.json'))['run_full'] is True else 1)" || { echo "full REFUSED by the pilot projection" >> $J; exit 1; }
    rl claude gr_audit_full $P/pf_ladder_pressure_residual.py audit full ;;
  checks) rl claude gr_checks $D/residual_checks.py ;;
  *) echo "usage: launch_residual.sh manifest | pilot | rehearse | project | full | checks"; exit 2 ;;
esac
