#!/usr/bin/env bash
# STOP-EV v2 driver (2026-11-15, session aea5bfb9; after the Codex review relayed by the USER, "remain: 1 ... 4"): run_stop_ev_model2.sh all ->
# selftests, E2SEP report2 (certificate groups, fixed identities; no rebuild), the certificate flags of the nine days, the five v2 fits (mean
# value head, saved heads), a first report2 (verdict PENDING until T5v2), the T5v2 falsifier (200 permuted-label refits per fold), the final
# report2. Each step through run_light on the claude slot; a busy slot is waited for.
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?all}"
OUT=data/predict_fun/research/stop_ev_model_20261115; mkdir -p $OUT/logs
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
TESTS="20261027 20261031 20261101 20261102 20261103"
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver2.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver2.log; return 1
  done; }
case "$PHASE" in
  all)
    job evm2_selftest python3 predictfun/research/pf_stop_ev_model.py selftest || exit 1
    job e2s2_selftest python3 predictfun/research/pf_stop_ev_e2sep.py selftest || exit 1
    job e2s2_report python3 predictfun/research/pf_stop_ev_e2sep.py report2
    for d in $DAYS; do job evm2_prep_$d python3 predictfun/research/pf_stop_ev_model.py prep2 $d || exit 1; done
    for t in $TESTS; do job evm2_fit_$t python3 predictfun/research/pf_stop_ev_model.py fit2 $t || exit 1; done
    job evm2_report_pending python3 predictfun/research/pf_stop_ev_model.py report2
    for t in $TESTS; do job evm2_t5_$t python3 predictfun/research/pf_stop_ev_model.py t5v2 $t || exit 1; done
    job evm2_report python3 predictfun/research/pf_stop_ev_model.py report2;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) STOP-EV-MODEL2 $PHASE DONE" >> $OUT/logs/driver2.log
