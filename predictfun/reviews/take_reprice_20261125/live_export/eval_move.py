"""Score the frozen chase MOVE model from move_model.json and check the fixture rows; Python standard library only.
Per row: prediction = baseline + the leaf value of every tree in order (at a split, input NaN -> the missing side, else left iff
input <= threshold); MOVE = prediction - (d x (lim - md) + 0.02 x min(lim, 100 - lim)); above = MOVE >= threshold. Per move (key, rows in t
order), the action is the first row above. Every value must equal the fixture's exactly (the fixture's floats are written with 17 digits).
Usage: python3 eval_move.py move_model.json move_fixtures.csv.gz   (exit 0: every row and every move agrees)."""
import csv
import gzip
import json
import math
import sys


def predict(model, x):
    s = model['baseline']
    for tr in model['trees']:
        i = 0
        while not tr['is_leaf'][i]:
            v = x[tr['feature'][i]]
            left = tr['missing_left'][i] if math.isnan(v) else v <= tr['threshold'][i]
            i = tr['left'][i] if left else tr['right'][i]
        s += tr['value'][i]
    return s


def cost(d, lim, md):
    return d * (lim - md) + 0.02 * min(lim, 100. - lim)


def check(model, rows):
    """rows: dicts of the fixture's columns (strings) -> (rows, rows that differ, moves, moves whose action differs)."""
    names, thr = model['inputs'], model['threshold']
    bad, first = 0, {}
    acts = {}
    for r in rows:
        x = [float(r[k]) if r[k] != '' else math.nan for k in names]
        p = predict(model, x)
        mv = p - cost(float(r['d']), float(r['lim']), float(r['md']))
        above = mv >= thr
        if p != float(r['pred']) or mv != float(r['MOVE']) or above != (r['above'] == '1'):
            bad += 1
        k, t = r['key'], float(r['t'])
        if above and (k not in first or t < first[k]):
            first[k] = t
        if r['act'] == '1':
            acts.setdefault(k, []).append(t)
    keys = {r['key'] for r in rows}
    moves_bad = sum(1 for k in keys if (([first[k]] if k in first else []) != sorted(acts.get(k, []))))
    return len(rows), bad, len(keys), moves_bad


def main(mp, fp):
    with open(mp) as f:
        model = json.load(f)
    with gzip.open(fp, 'rt', newline='') as f:
        rows = list(csv.DictReader(f))
    n, bad, nk, kb = check(model, rows)
    print(f'{n} rows, {bad} differ; {nk} moves, {kb} with a different action')
    return 0 if n and not bad and not kb else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
