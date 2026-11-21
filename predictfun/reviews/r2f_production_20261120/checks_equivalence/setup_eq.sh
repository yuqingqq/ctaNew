#!/bin/bash
# The v1.1 / v1.2 equivalence check of r2f_checks.py (declaration_v1_2.txt): both copies read the parity output (the already-read pilot
# rows) and write into their own scratch dirs; the logs are the receipt's (read only).
set -eu
E=/home/yuqing/ctaNew/predictfun/reviews/r2f_production_20261120/checks_equivalence; R=/home/yuqing/ctaNew/predictfun/reviews/r2f_production_20261120
for v in v11 v12; do rm -rf $E/$v; mkdir -p $E/$v; cp $R/parity.json $E/$v/; done
cp $E/r2f_checks_v1_1.py $E/v11/r2f_checks.py; cp $R/r2f_checks.py $E/v12/r2f_checks.py
for v in v11 v12; do
  f=$E/$v/r2f_checks.py
  python3 - "$f" "$R/logs" <<'PY'
import sys
p, logs = sys.argv[1], sys.argv[2]; s = open(p).read()
a = "P, PILOT = D / 'production_v1', D / 'attempt4_pilot_A'"; b = "HERE / 'logs'"
assert s.count(a) == 1 and s.count(b) == 1
s = s.replace(a, "P, PILOT = D / 'parity_pilot_e35', D / 'attempt4_pilot_A'").replace(b, f"Path('{logs}')")
open(p, 'w').write(s)
PY
done
grep -n "parity_pilot_e35\|LOGS = " $E/v11/r2f_checks.py $E/v12/r2f_checks.py
