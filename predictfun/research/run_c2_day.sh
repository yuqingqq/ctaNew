#!/usr/bin/env bash
# Confirmation C2, one fresh day (declared in STATUS.yml before any C2 day was read). One light process at a time.
# Exit 0 = chain ran and an admission record was appended; 75 = launcher busy/low memory/low disk (retry later);
# anything else = a step failed (a FAILED admission record is appended).
set -u
cd /home/yuqing/ctaNew
D=${1:?day}; R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs; A=data/predict_fun/research/c2/admission.jsonl
mkdir -p "$L" data/predict_fun/research/c2
[[ "$D" =~ ^202611[0-9]{2}$ ]] && [ "$D" -ge 20261101 ] || { echo "REFUSED: $D is not a C2 day (complete UTC days from 20261101)"; exit 2; }
rec() { python3 - "$D" "$1" "$2" >> "$A" <<'EOF'
import sys, json, datetime as dt
d, st, why = sys.argv[1:4]
print(json.dumps(dict(day=d, status=st, why=why, at=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))))
EOF
}
# source coverage: every hour file from d-1 23Z to d+1 01Z in the Predict WS/REST, reference relay and Binance tape
PF=data/predict_fun/runs/20261021T162823Z_0dd45f2d5a; REF=data/predict_reference/runs/20261022T132901Z_8c043d17b6
BN=data/mm_hf/raw/bookTicker/BTCUSDT; miss=0
for h in $(python3 -c "import datetime as t,sys; d=t.datetime.strptime(sys.argv[1],'%Y%m%d'); print(' '.join((d+t.timedelta(hours=k)).strftime('%Y%m%d_%H') for k in range(-1,26)))" "$D"); do
  for f in $PF/predict_ws/$h.jsonl.gz $PF/rest/$h.jsonl.gz $REF/raw_ws/$h.jsonl.gz $REF/reference_prices/$h.jsonl.gz $BN/$h.csv.gz; do
    [ -s "$f" ] || miss=$((miss+1)); done; done
[ "$miss" -eq 0 ] || { rec SOURCE_GAP "$miss hour files missing in d-1 23Z..d+1 01Z"; exit 0; }
run() { tag=$1; shift
  $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 && return 0
  grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' "$L/$tag.log" && exit 75
  rec FAILED "$tag: $(tail -1 "$L/$tag.log" | cut -c1-200)"; exit 1; }
ST=G4+G2+G6+G5+G3+G1; CB=G5+G1+M025; A1=A1:5:25:0
ARMS="MICRO|$ST+M025|mid|$A1;MICRO|$ST+M025|early|$A1;MICRO|$ST+M025|late|$A1;MICRO|$CB|mid|$A1;MICRO|$CB|early|$A1;MICRO|$CB|late|$A1"
ARMS="$ARMS;MICRO|$ST+M025|mid|25;MICRO|$CB|mid|25;MICRO|$ST+M025|mid|5;MICRO|$ST|mid|5;MICRO|$ST|mid|25;MICRO|$ST+M05|mid|5;MICRO|$CB|mid|5"
[ -s data/predict_fun/research/step1/$D/summary.json ] || run c2s1_$D python3 -m predictfun.research.pf_step1 $D
run c2t_$D  python3 -m predictfun.research.pf_step1_timing $D
run c2bt_$D python3 -m predictfun.research.pf_step1_booktime $D
run c2s5f_$D python3 -m predictfun.research.pf_step5_features $D
run c2ap_$D python3 -m predictfun.research.pf_step5_c2 apply $D
PF_ARMS="$ARMS" PF_TIMING=e2 PF_PREDS=preds_c2 PF_TAG=c2 run c2s4_$D python3 -m predictfun.research.pf_step4 $D tokyo20 custom
run c2chk_$D python3 -m predictfun.research.pf_check_fills tokyo20_custom_e2_preds_c2_c2 $D
# admission: outcome-independent only (window coverage from step 1, fill invariants)
W=$(python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print(min(s['n_windows_with_live_frames'], s['n_windows_with_outcome'], s['n_windows_with_start_price']))" data/predict_fun/research/step1/$D/summary.json)
V=$(grep -oE 'with violations [0-9]+' "$L/c2chk_$D.log" | grep -oE '[0-9]+$')
if [ "${V:-x}" != 0 ]; then rec FILL_INVARIANT "violations=${V:-unparsed}"
elif [ "$W" -lt 270 ]; then rec WINDOW_GAP "complete windows $W < 270 of 288"
else rec ADMITTED "complete windows $W/288, fill invariants 0 violations"; fi
exit 0
