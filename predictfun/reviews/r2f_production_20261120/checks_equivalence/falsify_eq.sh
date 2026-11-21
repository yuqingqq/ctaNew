#!/bin/bash
# Falsifiers of compare_eq.py. x0 is the neutral control: paths.csv and checks.json rewritten through csv / json round trips (the writers'
# own dialect, \r\n line ends) must compare identical, so x1 / x2 differ from v12 only by their planted change. x1: one count + 1 in
# paths.csv; x2: one control flipped in checks.json. Each must be flagged (exit 1).
E=/home/yuqing/ctaNew/predictfun/reviews/r2f_production_20261120/checks_equivalence; cd $E; fails=0
python3 compare_eq.py v11 v12 > /dev/null || { echo "intact pair differs"; fails=1; }
rm -rf x0 x1 x2; cp -r v12 x0; cp -r v12 x1; cp -r v12 x2
python3 - <<'PY'
import csv, json
def recsv(p, bump):
    with open(p, newline='') as f:
        R = list(csv.reader(f))
    if bump:
        R[1][-1] = str(int(R[1][-1]) + 1)
    with open(p, 'w', newline='') as f:
        csv.writer(f).writerows(R)
def rejson(p, flip):
    j = json.load(open(p)); c = j['readings']['controls'][0]; k = sorted(c['refused'])[0]
    if flip:
        c['refused'][k] = not c['refused'][k]
    json.dump(j, open(p, 'w'), indent=1, sort_keys=True)
recsv('x0/paths.csv', False); rejson('x0/checks.json', False)
recsv('x1/paths.csv', True); rejson('x2/checks.json', True)
PY
cmp -s v12/paths.csv x0/paths.csv || { echo "the neutral csv rewrite changed bytes"; fails=1; }
python3 compare_eq.py v11 x0 > /dev/null || { echo "neutral control x0 flagged"; fails=1; }
for x in x1 x2; do python3 compare_eq.py v11 $x > /dev/null && { echo "falsifier $x NOT flagged"; fails=1; }; done
rm -rf x0 x1 x2
[ $fails = 0 ] && echo "compare_eq falsifiers: the intact pair and the neutral rewrite are identical; a changed CSV count and a flipped control are each flagged" || echo "compare_eq falsifiers FAILED"
exit $fails
